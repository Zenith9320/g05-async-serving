# g05-async-serving

Galaxea **G0.5 VLA**（YAM 双臂机器人）的**异步前瞻（look-ahead）推理服务**与测试。

单次模型推理约需 **1.5 s**（开发机估计；前瞻能藏住的上限是 0.53 s，须在工控机实测，见 §8），而机器人需以 **30 Hz**（YAM 训练数据的相机对齐帧率）连续控制。本项目采用**分块滚动时域（receding horizon / "Plan B"）**方案：一次推理生成一整段未来动作，机器人逐帧回放的同时后台提前计算下一段，从而把推理延迟藏起来。

> **动作生成模式：AR（离散 ActionCodec）**。本仓库部署使用 G0.5 的 **AR 头**出动作，不使用 FM（连续 flow matching）。切法是关 FM、留 AR：`model.model_arch.continuous_action=false`（`discrete_action` 保持 `true`）。AR 模式下动作来自 ActionCodec 解码，`_absent_keys`（不置信部件）作为唯一动作来源的置信度掩码——被 `serve_policy` 丢弃、客户端保持上一帧值。§6 所有加载模型的命令都已带此开关。

---

## 目录

1. [文件说明](#1-文件说明)
2. [异步推理实现思路](#2-异步推理实现思路)
3. [测试目的与内容](#3-测试目的与内容)
4. [运行依赖（重要）](#4-运行依赖重要)
5. [环境配置与创建（miniforge）](#5-环境配置与创建miniforge)
6. [测试指令](#6-测试指令)
7. [工控机上需要确认的内容](#7-工控机上需要确认的内容)
8. [如何根据 GPU 推理时间定 `action_steps`](#8-如何根据-gpu-推理时间定-action_steps)
9. [关键参数与频率说明](#9-关键参数与频率说明)

---

## 1. 文件说明

| 文件 | 作用 |
|---|---|
| `serve_policy.py` | WebSocket 异步推理服务器。加载模型/processor，接收客户端原始观测，返回反归一化动作字典。异步前瞻逻辑在 `chunked_policy.py` 里。 |
| `chunked_policy.py` | `ChunkedPolicyWrapper`：分块缓存 + 后台前瞻的**状态机**。无 torch/g05 依赖，可单独 import 单测。 |
| `fake_policy_client.py` | 合成观测客户端（端到端 WS 测试）。模拟真实机器人协议：按固定频率发真实观测/空观测，遵守 `need_obs` 握手。 |
| `offline_chunk_harness.py` | 离线 chunk 时序测试。不走网络，直接驱动 `ChunkedPolicyWrapper`，单独验证分块/前瞻的时序逻辑。 |
| `tests/` | 无 GPU 单元测试（`test_codec.py` / `test_build_obs.py` / `test_chunked_wrapper.py`），用 pytest 跑。 |
| `conftest.py` | pytest 引导：把仓库根加进 `sys.path`，让测试能 import 同级脚本。 |
| `.python-version` | Python 版本标记（3.10.16），供 `rootutils.setup_root` 定位项目根。 |
| `.gitignore` | 排除 `__pycache__`、`.pytest_cache`、日志、`.env`、模型权重等。 |

> `serve_policy.py` / `fake_policy_client.py` / `offline_chunk_harness.py` 源自 `GalaxeaVLA/scripts/` 同名文件；`chunked_policy.py` 是把原来内嵌在 `serve_policy.py` 里的 `ChunkedPolicyWrapper` 抽成独立模块（去掉 torch/g05 依赖，便于无 GPU 单测）。改 `serve_policy.py` 时注意与 `GalaxeaVLA/scripts/` 同步；`chunked_policy.py` 的改动只在本仓库。

## 2. 异步推理实现思路

### 2.1 为什么需要异步

- 模型一次推理（生成一段未来动作）约 1.5 s（开发机估计；前瞻能藏住的上限是 0.53 s，见 §8）。
- 机器人控制频率 30 Hz（每 33 ms 一步）。若每步都等推理（RTC，`action_steps=1`），会每步卡 1.5 s，无法实时控制。

### 2.2 Plan B：分块滚动时域（receding horizon）

- 一次推理输出**一整段**未来动作，长度 = `chunk_len`（= 模型动作视野 `action_size`，YAM 为 **32** 步）。
- 机器人按控制频率逐个回放这 32 个动作。
- 回放到第 `action_steps` 步时（默认 **16**），后台**提前**用最新观测启动下一段推理（`asyncio.create_task`）。
- 当前段回放完（第 32 步）时，若后台推理已完成，直接"换段"；若没完成，同步等待（超时兜底）。

推理延迟因此被藏在 **`chunk_len - action_steps` = 16 步** 的"尾巴"里：

```
 时间轴（30 Hz，每步 33 ms）
 |---------------- chunk_len = 32 步（一次推理）----------------|
 |--- action_steps = 16 ---|--------- look-ahead 尾巴 16 步 ---------|
        回放前 16 步             后台算下一段（覆盖这 16 步）
                                          ↑ 换段点，理想情况下下一段已就绪
```

### 2.3 关键机制（对应 `ChunkedPolicyWrapper`）

- **`need_obs` 握手**：告诉客户端"下一步该发真实观测还是空观测"。只有触发新推理时才需要真实观测，其余步发 `{}`，省网络开销。
- **首段同步、后续异步**：第一段没有可回放内容，同步算；之后的段都在后台算。
- **超时兜底**：后台推理若超过尾巴长度（16 步），换段时退化为同步等待——该步延迟 = 满推理延迟（测试里表现为 SLOW）。
- **重规划重叠**：后续每段只服务"新鲜的后 `action_steps` 步"，前 `action_steps` 步与上一段重叠（滚动时域）。

## 3. 测试目的与内容

两条测试都只验证**时序/协议**，不验证动作质量（喂的是全零合成观测）。

### 3.1 离线 chunk 时序测试（`offline_chunk_harness.py`）

- **目的**：把"分块 + 前瞻"的时序逻辑从网络层剥离，单独验证。
- **内容**：加载真实模型，用合成观测驱动 `ChunkedPolicyWrapper` 跑 64 步（2 段），测每步返回延迟。
- 可选 `--compare_rtc`：与 `action_steps=1`（RTC）对比，量化 look-ahead 藏掉了多少延迟。

### 3.2 端到端 WebSocket 测试（`serve_policy.py` + `fake_policy_client.py`）

- **目的**：验证完整服务链路（msgpack 编解码、`need_obs` 握手、异步 serving、时序）。
- **内容**：客户端按 30 Hz 连发 64 步合成观测，服务器返回动作。

### 3.3 无 GPU 单元测试（`tests/`）

- **目的**：把"分块 + 前瞻"状态机与 codec/obs 构造逻辑从模型、GPU、网络里彻底剥离，用 mock `compute` 验证，任何机器（含纯 CPU 开发机）都能秒跑。
- **内容**：
  - `test_codec.py`：msgpack + numpy 编解码 round-trip（与 `g05.utils.websocket` 线上兼容）。
  - `test_build_obs.py`：合成观测的 image/state/frequency 形状与类型正确。
  - `test_chunked_wrapper.py`：`ChunkedPolicyWrapper` 状态机——首段同步、`action_steps` 触发点后台推理、换段复用、`need_obs` 握手、RTC（`action_steps=1`）、`reset()`。
- **运行**：`python -m pytest tests/ -v`（仅需 `msgpack` + `numpy` + `pytest`）。

### 3.4 看什么（成败判定）

| 步骤 | 预期 | 含义 |
|---|---|---|
| step 0 | ≈ 满推理延迟 | 单步 RTC 成本基线 |
| step 1–31 | ≈ 0 ms | 从缓存段回放 |
| step 32（换段） | ≈ 0 ms | look-ahead 成功盖住推理；≈ 满延迟 = 超时兜底 |

> 慢阈值 `SLOW_MS = 20`，任何一步 ≥ 20 ms 会被标成 SLOW 并打印。

## 4. 运行依赖（重要）

本仓库是"服务 + 测试"这一**薄层**，运行依赖 **GalaxeaVLA** 仓库（提供 `g05` 包、configs、checkpoints）。

需要：

1. **`g05` 包**：`pip install -e GalaxeaVLA`（editable 安装，`.pth` 指向 `GalaxeaVLA/src`）。
2. **checkpoints**（从 Hugging Face `OpenGalaxea/G05` 下载）：

   ```
   GalaxeaVLA/
   └── checkpoints/
       ├── action_tokenizer.pt              # 官方 ActionCodec
       └── g05-base/                        # run dir（load_config_from_run_dir 从这找 .hydra/config.yaml）
           ├── .hydra/config.yaml
           ├── dataset_stats.json
           └── checkpoints/model_state_dict.pt
   ```

3. **GPU + CUDA**：模型推理需要 GPU（默认 `device=cuda`）。

## 5. 环境配置与创建（miniforge）

```bash
# 1. 安装 miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh

# 2. 建 Python 3.10 环境（与 requires-python >=3.10.16,<3.11 对齐）
conda create -n g05 python=3.10 -y
conda activate g05

# 3. 安装 GalaxeaVLA（g05 包 + 全部依赖，含 torch cu128）
git clone https://github.com/OpenGalaxea/GalaxeaVLA.git
cd GalaxeaVLA
pip install -e .

#    注：torch 需 CUDA 12.8 wheel。原始仓库用 uv（uv.lock + [tool.uv.sources] 指向 pytorch-cu128）。
#    若 pip 装的 torch 不带 GPU，先手动装再 -e .：
#    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 4. 补 serving 脚本额外依赖（msgpack 不在 pyproject 依赖里，客户端直接 import）
pip install msgpack

#    无 GPU 单元测试需要 pytest（纯逻辑，不加载模型）
pip install pytest

# 5. 环境变量（HF_TOKEN 等一律走环境变量，切勿写进代码/提交）
cd GalaxeaVLA
cp .env.example .env
# 编辑 .env：填 PROJECT_ROOT、HF_HOME、HF_TOKEN（如需私有/gated 资源）等
source .env
```

## 6. 测试指令

前置：`g05` 已可 import，checkpoints 已就位，GPU 可用。

```bash
cd /path/to/GalaxeaVLA          # 以 GalaxeaVLA 为工作目录（稳妥起见，与官方运行方式一致）

# 6.1 离线 chunk 时序测试
python /path/to/g05-async-serving/offline_chunk_harness.py \
    --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
    --action_steps 16 --steps 64 --rate 30 [--compare_rtc] \
    model.model_arch.continuous_action=false

# 6.2 端到端 WS 测试
# 终端 1 —— 服务器
python /path/to/g05-async-serving/serve_policy.py \
    --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
    --host 127.0.0.1 --port 8765 \
    eval_embodiment=galaxea_r1lite model.model_weights_to_bf16=true \
    model.model_arch.continuous_action=false

# 终端 2 —— 客户端
python /path/to/g05-async-serving/fake_policy_client.py \
    --host 127.0.0.1 --port 8765 --steps 64 --rate 30
```

> 客户端 `--rate` 必须 = 真实控制频率（YAM = 30）。`--rate 0` 只用于测裸推理延迟（此时 look-ahead 尾巴塌缩成 0，换段必超时兜底）。

```bash
# 6.3 无 GPU 单元测试（不加载模型；msgpack + numpy + pytest 即可）
cd /path/to/g05-async-serving
python -m pytest tests/ -v
```

## 7. 工控机测试清单

开发机与工控机 GPU 不同，**时序数字不能直接迁移**。上工控机后按下述顺序逐项确认（`action_steps` 的估算流程见 §8）：

### 7.1 环境 / 资源自检（最基础）

- `g05` 可 import、GPU 可见、CUDA 版本正确。
- checkpoints 就位、模型能正常加载。
- **VRAM 够不够装模型**：不够用 `model.model_weights_to_bf16=true`，或减小 batch/视野。

### 7.2 端到端 WS 链路（协议 / 链路轴）

```bash
# 终端 1 —— 服务器
python serve_policy.py \
    --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
    --host 127.0.0.1 --port 8765 \
    eval_embodiment=galaxea_r1lite model.model_arch.continuous_action=false
# 终端 2 —— 客户端
python fake_policy_client.py --host 127.0.0.1 --port 8765 --steps 64 --rate 30
```

验证：msgpack 编解码、`need_obs` 握手、`action_steps` 宣告、异步 serving、`frequency` 字段，在工控机 GPU 环境下整条链路走通。`offline_chunk_harness.py` 只测 chunk 状态机（不过网络），这一步补上完整链路。

### 7.3 时序：推理延迟 vs look-ahead 窗口（`action_steps`）

实测工控机 GPU 单次推理延迟 `T_infer`，确认 ≤ **0.53 s**（30 Hz 下 16 步窗口）。若 `T_infer` > 0.53 s，换段必掉进超时兜底——解法是加速推理或加长 `chunk_len`，**调 `action_steps` 救不了**（16 已经是 `chunk_len=32` 下能取到的最大窗口）。完整估算 + 验证指令见 **§8**。

### 7.4 AR 动作正确性（动作轴 ★最关键）

AR 模式下动作完全来自 ActionCodec 解码，正确性取决于 **slot 对齐**：YAM 的 6 个关节 + 2 个夹爪在 codec 的 27 维 slot 里，**顺序/符号必须与 R1-Lite 一致**，否则不是"不准"而是"直接崩"（符号反、量级错）。

- **怎么测**：喂真实观测（真机相机 + 关节，或一段录制的 MCAP），看输出动作的**量级合理、符号正确**。
- 注：仓库里已无真数据校验脚本（`offline_eval.py` 已删），这块只能靠真机实测，或临时喂一段真实观测人工看输出。

### 7.5 真机 IO + 频率（数据轴）

- **3 路相机**：`head_rgb` / `left_wrist_rgb` / `right_wrist_rgb` 能采到，shape `3×480×640` uint8。
- **关节读数**：`left_arm[6] + left_gripper[1] + right_arm[6] + right_gripper[1]` 的顺序/符号与模型 schema 一致（也是 7.4 的 slot 对齐一部分）。
- **控制频率**：客户端 `--rate` 与实际控制频率 = 30 Hz；`frequency` 字段必须显式发（服务端缺省是 15，YAM 应是 30）。
- **`action_steps` 对齐**：`action_steps = obs_stride × (control_fps / data_fps)`，与实际控制频率、训练数据帧率一致，避免分布漂移。

### 7.6 安全状态机（上真机前最优先）

实验室的 `scripts/safety`（SAFE HOLD / gravity / off）——上真机前先确认急停与软限位可用。这与模型无关，但比任何测试都优先。

## 8. 如何根据 GPU 推理时间定 `action_steps`

`action_steps` 不是自由旋钮，它由"GPU 推理时间"和"训练数据帧距"共同决定。

### 8.1 能藏住的推理时间由前瞻窗口决定

```
前瞻窗口（步） = min(action_steps, chunk_len − action_steps)
前瞻窗口（秒） = min(action_steps, chunk_len − action_steps) / 控制频率
```

`chunk_len = 32` 是模型 `action_size`（固定），所以窗口在 **`action_steps = chunk_len/2 = 16`** 时取到最大：

| `action_steps` | 窗口（步） | 30 Hz 下窗口 |
|---|---|---|
| 8 | min(8, 24) = 8 | 0.27 s |
| **16** | min(16, 16) = **16** | **0.53 s** |
| 20 | min(20, 12) = 12 | 0.40 s |

**结论：`action_steps=16` 已经是"窗口最大"的取值，不是待调的占位值。** 真正要拿 GPU 时间校验的是"0.53 s 够不够"。

### 8.2 决策规则

1. 在工控机 GPU 上实测单次推理延迟 `T_infer`。
2. 若 `T_infer ≤ 0.53 s`（留余量，建议 ≤ 0.45 s）→ 16 成立，换段 ≈ 0 ms。
3. 若 `T_infer > 0.53 s`：光调 `action_steps` 无解，只能
   - 加速推理（bf16 / `torch.compile` / 减视野或 batch / 换 GPU），或
   - 加长 `chunk_len`（= 模型 `action_size`，需改模型重训，不是运行时参数），或
   - 降低控制频率（偏离 YAM 30 Hz 会分布漂移，不推荐），或
   - 接受换段阻塞（等价于每 16 步卡一次满推理）。

### 8.3 `action_steps` 的下界：数据对齐

窗口最大化之外，`action_steps` 还要满足训练数据帧距对齐：

```
action_steps = obs_stride × (control_fps / data_fps)
```

即"每次重规划对应的观测间隔"要和训练数据一致，否则有分布漂移。默认 16 在 YAM（30 Hz、obs_steps=1）下同时满足窗口最大与数据对齐。**若要让 `action_steps` 成为真正可调参数（例如为更快的闭环响应取 8），代码已支持任意 `1 ≤ action_steps ≤ chunk_len/2`**（见 `chunked_policy.py` 的 `_serve_start`/`_serve_len` 实现与 `tests/test_chunked_wrapper.py::test_general_action_steps_contiguous`）。

### 8.4 具体指令（工控机上实测）

**第 1 步：量 `T_infer`（单次推理延迟）**

```bash
cd /path/to/GalaxeaVLA
python /path/to/g05-async-serving/offline_chunk_harness.py \
    --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
    --action_steps 1 --steps 8 --rate 0 \
    model.model_arch.continuous_action=false
```

`--action_steps 1` 即 RTC，每一步都是一次完整推理；汇总里的 `first-step latency (inference baseline)` 就是 `T_infer`。多跑几次取中位数，避开首次模型加载 / `torch.compile`。

**第 2 步：套充要条件**

```
T_infer ≤ (chunk_len − action_steps) / 30 Hz = (32 − action_steps) / 30
```

- `T_infer ≤ 0.53 s`（建议留余量 `≤ 0.45 s`）→ `action_steps=16` 成立，换段 ≈ 0 ms。
- `T_infer > 0.53 s` → 调 `action_steps` 无解（16 已是最大窗口），只能加速推理（bf16 / `torch.compile` / 减视野或 batch）。

**第 3 步：验证换段点**

```bash
python /path/to/g05-async-serving/offline_chunk_harness.py \
    --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
    --action_steps 16 --steps 64 --rate 30 \
    model.model_arch.continuous_action=false
```

看 **step 32（换段）**：≈ 0 ms 说明藏住了；标 `SLOW`（≥20 ms）说明 `T_infer` 超了窗口，回到第 1 步。

## 9. 关键参数与频率说明

- **YAM 数据 = 30 Hz**：`tools/yam/mcap_to_lerobot.py` 把 action（200 Hz）、state（~260 Hz）重采样到 head 相机 30 Hz 栅格（`FPS = 30`）。
- **`frequency` 是模型的编码输入**：`g05/tokenizer/models/protocol.py` 注释为 "Control frequency in Hz"，从 `data["frequency"]` 读取并编码进动作 token。客户端必须发，否则服务端缺省 15（那是 SO-100 的帧率，对 YAM 是分布漂移）。
- **look-ahead 窗口** = `min(action_steps, chunk_len - action_steps)` 步（推导见 §8）。默认 chunk_len=32、action_steps=16，即 16 步；**30 Hz 下 = 0.53 s**（10 Hz 下才是 1.6 s，旧文档里的 10 Hz / 1.6 s 是错误占位，已修正为 30 Hz）。
- **帧距对齐**：`action_steps = obs_stride × (control_fps / data_fps)`（见 `GalaxeaVLA/docs/deployment/serve_policy_mem.md`）。

---

## 备注

- 本仓库只含服务/测试脚本与文档，不含模型权重与数据。
- 安全：云 AK/SK、`HF_TOKEN`、实验室凭据等一律走环境变量，不硬编码、不提交。
