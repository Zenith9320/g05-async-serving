# 真机部署 TODO

> 目标：在工控机上跑通 G0.5 AR 异步推理 → YAM 双臂控制。
> **部署根目录：`~/workspace/g05-yam-deployment`**（含 `GalaxeaVLA` / `gello_software` / `g05-async-serving` / `i2rt`）。
> 决策已定：AR（`model.model_arch.continuous_action=false`）、`frequency=30`、异步 look-ahead（`action_steps=16`）。
> 环境用 **uv**（GalaxeaVLA 自带 `uv.lock` + `.python-version`=3.10.16 + torch→cu128 index）。
> **供帧率（已定论）**：chunked 服务是「16 步 @30Hz 突发 + chunk 边界阻塞等待推理」的锯齿状，**不是连续 ≥5Hz**。
> 因此用 **untimed 控制**（正确），`T_infer` 不再卡正确性——它只决定平均供帧率 `16 / max(0.53s, T_infer)`，即任务快慢，不决定能不能跑。

## 待确认（不阻塞第 1/2 步，阻塞第 3 步验收）

- [x] **关节对齐（14 维，已定，纯透传）**：
  - 坐标系 = 从臂(i2rt)帧：记录 `applied_joint_state`；顺序/符号 = **恒等映射**，`left_arm=act[:6] / left_gripper=act[6] / right_arm=act[7:13] / right_gripper=act[13]`，无重排、无符号翻转。
  - action = **绝对关节值**：serve_policy `postprocess` 已做去 padding + 反归一化 + 相对→绝对。
  - bridge 按 14 维直发 `command_joint_state()`，零转换。
  - **codec 输出顺序已端到端核对（静态追码）**：decode = 训练顺序 = `/action/accepted.position`（follower 14 维，`mcap_recording.py:515`）→ `mcap_to_lerobot.py:308` 恒等切 `left_arm=accepted[:6]` → serve_policy unpad → bridge 恒等 → `BimanualRobot` 切 `[左(7), 右(7)]`（`robot.py:105`）→ `YAMRobot.command_joint_state` 7 维 = `[关节 0x01..0x06, gripper]`（gripper 最后，`motor_chain_robot.py:110`）。整条链恒等。
  - **符号**：follower `yam.yml directions` 全 +1、gripper 无翻号 → 无翻转。⚠️ `yam_left/right.yaml` 的 `agent.joint_signs=[1,-1,-1,-1,1,1]` 是 **GELLO 主手(leader)** 符号，只在遥操作主→从映射时用；训练 action 取**从臂** `/action/accepted`（已在从臂系），此符号与 VLA 推理无关，别误当 follower 翻号。
- [x] **相机 shape_meta（已定）**：3 路 `head_rgb/left_wrist_rgb/right_wrist_rgb` 各 `(3,480,640) uint8`；state 4 维 float32；`embodiment_type="galaxea_r1lite"`。
- [x] **安全接入方式（已定）**：safety daemon（`gello.safety`），bridge 走 `connect_safety` + `SupervisedMotionSession`，**不再直接 `YAMRobot.create_pair`**（会与 daemon 抢 CAN）。
- [x] **相机型号（已定）**：RealSense rig（`CameraRig`），`capture_images()` 已实现；serials 已填（见第 2 步，★左/右为「先左后右」假设，待物理确认）。
- [x] **★ ActionCodec 27-dim ↔ 6 关节 slot 顺序/符号（已定：codec 已在 YAM 微调）**：
  - 布局已核对（非 bug）：`action_dim:27` / `parts_meta(16)` / `shape_meta(14)` 三层不冲突；27 = `[left_control(9)|left_gripper(1)|right_control(9)|right_gripper(1)|lower_body(7)]`，由 `GroupedPaddingMerger` 从 `parts_meta/r1lite.yaml` + `merge_spec` 生成；YAM 14 维落点 `left_arm[0:6] / left_gripper[9] / right_arm[10:16] / right_gripper[19]`，其余 pad，可逆无重排。
  - **codec 不是冻结的**：用 `ActionCodec-finetune-pkg`（https://github.com/Zenith9320/ActionCodec-finetune-pkg）在 YAM 上微调，产物 `checkpoints/action_tokenizer_yam.pt`（工控机即此版本）。codec 已「说」YAM 自己的关节约定，**i2rt joint1..6 vs R1-Lite 的顺序/符号问题作废**。
  - 配套：VLA（VLM+head）也用 `configs/task/yam.yaml`（`ckpt_dir: action_tokenizer_yam.pt`）微调，run dir 与基座同名 `g05-base`（工控机上已覆盖）。serving 用 `checkpoints/g05-base/.../model_state_dict.pt` + `action_tokenizer_yam.pt` 这一对；注意 **dev 机上的旧 `g05-base`/`action_tokenizer.pt` 是基座+冻结 codec**，rsync 时会被工控机版本覆盖，别混。
- [ ] **任务名样例**：真机要跑哪些任务。

---

## 第 0 步：工控机环境（前置，不跳）

> 代码已在 `~/workspace/g05-yam-deployment`（GalaxeaVLA / gello_software / g05-async-serving / i2rt 四个仓库）。以下只补装环境/依赖。

```bash
# 1. 查 GPU 型号 → 装对齐的 CUDA driver（工控机 GPU 与 dev 机不同，先确认）

# 2. 装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # 或重开终端

# 3. GalaxeaVLA 3.10 env（serving 侧）
cd ~/workspace/g05-yam-deployment/GalaxeaVLA
uv sync
uv pip install msgpack pytest        # serving 脚本额外依赖（不在 pyproject 里）

# 4. 环境变量（HF_TOKEN 走环境变量，勿硬编码/提交）
cp .env.example .env                 # 填 PROJECT_ROOT / HF_HOME / HF_TOKEN 等
source .env

# 5. YAM 控制栈（★独立环境 Python 3.11，与上面 3.10 分开，两进程走 WS 通信）
cd ~/workspace/g05-yam-deployment/gello_software
uv sync --frozen --extra yam --extra realsense     # Python 3.11
uv pip install msgpack websockets                  # bridge/vla_client.py 客户端依赖（gello 不带）
# i2rt 不用手动装：`--extra yam` 的 [tool.uv.sources] 已把它指向 ../i2rt 做 editable 安装
# （python-can 也随 yam extra 进来）。所以 gello venv 里直接 `import i2rt` 即可；
# §1.1 命令里的 PYTHONPATH 只是保险，可省略。

# 6. 推理环境自检：应打印 torch 版本 + True
cd ~/workspace/g05-yam-deployment/GalaxeaVLA
uv run python -c "import g05, torch; print(torch.__version__, torch.cuda.is_available())"
```

**✅ 通过标准（预期输出）：**
- 步骤 2：`uv --version` 打印版本号。
- 步骤 3：`uv sync` 无报错结束（GalaxeaVLA `.venv` 建好）。
- 步骤 4：`source .env` 后 `echo "$HF_TOKEN"` 非空。
- 步骤 5：`uv sync --frozen --extra yam --extra realsense` 无报错；gello venv 内
  `python -c "import i2rt, can; print('ok')"` 打印 `ok`（i2rt editable + python-can 就位）。
- 步骤 6：打印 `<torch版本> True`（例 `2.6.0 True`；`True`=CUDA 可用，`False` 先修 driver）。

---

## ~~已完成：模型 + 测 T_infer~~

> 不做 T_infer 测量门禁：untimed 控制下 chunk 边界阻塞是正常行为，`T_infer` 只影响平均帧率。
> 模型 checkpoint 仍需就位，且必须是 **YAM 微调后的那一对**（codec + VLA checkpoint 配套）：
> 若工控机还没有，照旧 rsync 整目录到 `~/workspace/g05-yam-deployment/GalaxeaVLA/checkpoints/`：
>
> ```
> checkpoints/
> ├── action_tokenizer_yam.pt                 # ★ YAM 微调后的 ActionCodec（ActionCodec-finetune-pkg 产物）
> └── g05-base/                               # YAM 微调后的 VLA run dir（与基座同名，工控机上已覆盖；dev 机的旧 g05-base 不算）
>     ├── .hydra/config.yaml                  # 里面 ckpt_dir 应指向 action_tokenizer_yam.pt（或本 run dir 的 sidecar action_tokenizer.pt）
>     ├── dataset_stats.json                  # YAM stats（yam_stats.json）
>     └── checkpoints/model_state_dict.pt     # YAM 微调后的 VLM+head
> ```
>
> ⚠️ 三个易错点：
>   1. codec 用 `action_tokenizer_yam.pt`，**不是**冻结的 `action_tokenizer.pt`。`g05-base` 在工控机上是 YAM 微调后的 VLA（与基座同名），
>      但 **dev 机上的 `g05-base` 还是 r1lite+r1pro 旧基座**——rsync 时用工控机的 `g05-base` + `action_tokenizer_yam.pt` 覆盖 dev 机，别让旧文件混进来。
>   2. config 里 `ckpt_dir` 是相对路径，serve_policy 要从 GalaxeaVLA 根目录启动（§3 的 `cd` 不能省，省了相对路径就找不到 codec）。
>   3. 若 run dir 内有 `action_tokenizer.pt` sidecar，serve_policy 优先用它（`_apply_action_tokenizer_sidecar`），否则回退 config `ckpt_dir`。
>      两种都要是 YAM 微调后的 codec，别让冻结的 `checkpoints/action_tokenizer.pt` 被捡到。

**✅ 通过标准（预期输出）：**
- `checkpoints/action_tokenizer_yam.pt` 在位；`g05-base/.hydra/config.yaml` 里 `ckpt_dir` 指向 YAM codec（`action_tokenizer_yam.pt` 或 `g05-base/action_tokenizer.pt` sidecar），**不是** `checkpoints/action_tokenizer.pt`。
- 三份文件在位：`g05-base/.hydra/config.yaml` + `g05-base/dataset_stats.json` + `g05-base/checkpoints/model_state_dict.pt`。
- 在 GalaxeaVLA 根目录跑 `ls checkpoints/action_tokenizer_yam.pt` 能列出（验证相对路径成立）。

---

## ~~已完成：端到端 serving 冒烟~~

> 协议已由 `fake_policy_client.py` + 纯函数单测（`tests/test_vla_client.py`）覆盖。
> 不单独做无机器人 serving 冒烟，直接在第 3 步真机联调时一并验证。

---

## 第 1 步：机器人控制打通（daemon 路径，风险最高）

> ⚠️ 先做 §1.1 看门狗（最优先），再做 1.2 抱持冒烟。急停 / SAFE_HOLD 可用之后才动命令。
> 安全接入已定：safety daemon 独占 CAN；bridge 是受监督客户端，**不直接 create_pair**。

### 1.1 看门狗寄存器确认 + 写入（daemon 停止；纯 i2rt 自带，不用 read_arm_params.py）

> ⚠️ **先确保机械臂有物理支撑/挡块**：本节会 `pkill` daemon，且 `set_timeout.py` 会对每个电机 `motor_off`，
> 两者都会切断电机保持力。若臂正被 daemon 抱持（或由电机保持），一停就**瞬间垂落（limp）**。
> 执行前先确认臂已落在安全位置或有物理支撑。

```bash
pgrep -af yam_safety_daemon      # 有输出 = daemon 在跑，先停掉
pkill -f yam_safety_daemon

# set_timeout.py --timeout 读写一体：读当前值 → 写 8000 → save_to_memory(写 flash) → 再读回，
# 每条命令把每个 motor 的 id/master_id/timeout 当前值与写后值都打出来，读回即验证。
cd ~/workspace/g05-yam-deployment/i2rt/i2rt/motor_config_tool   # 必须在此目录跑（脚本 `from utils import ...`）
PYTHONPATH=~/workspace/g05-yam-deployment/i2rt python set_timeout.py --channel can_follower_l --timeout
PYTHONPATH=~/workspace/g05-yam-deployment/i2rt python set_timeout.py --channel can_follower_r --timeout
# 判定：左右两条各 7 个 motor（1–7，含 gripper=7），写后全部 timeout=8000 才算过。
#   - 某 motor 读不到/报错 → 没上电 / ID 不对 / 线没接；脚本停在该 motor，正好定位，修好重跑。
#   - timeout=0 = 看门狗被关（危险），本命令强制写回 8000。
#   - save_to_memory 把 8000 写进电机 flash，断电不丢 → 跑一次即可，别频繁刷。
# 可选 liveness 快查（不崩、列出在线 motor；会短暂 motor_on→off 使能，不送位置/力矩命令）：
#   PYTHONPATH=~/workspace/g05-yam-deployment/i2rt python ping_motors.py --channel can_follower_l
#   PYTHONPATH=~/workspace/g05-yam-deployment/i2rt python ping_motors.py --channel can_follower_r
# daemon 启动时会 verify_motor_watchdogs（14 个全 8000 才放行使能）。
```

**✅ 通过标准（预期输出）：**
- 左右两条命令各打出 7 个 motor（1–7），每个 motor 写后 `current setting: timeout = 8000`。
- 全程无 `timeout = 0`、无 CAN 读不到/报错。

### 1.2 抱持冒烟（`bridge/hold_in_place.py`，已写）

```bash
cd ~/workspace/g05-yam-deployment/g05-async-serving
python bridge/hold_in_place.py --seconds 10    # 有界自退
# 验：daemon 启动/复用、两路心跳租约(parent+collector)、command_joint_state 往返、
#     live_max_dev≈0（命令到当前位置不产生位移）、Ctrl+C→SAFE_HOLD。
```

**✅ 通过标准（预期输出）：**
- daemon 启动/复用成功（日志显示 SAFE_HOLD → ACTIVE 使能链路）。
- 打印 `live_max_dev ≈ 0`（命令到当前位置不产生位移）。
- 脚本 10s 到期自退，或 Ctrl+C 后进入 SAFE_HOLD（不是 crash）。

### 1.3 teardown

```bash
cd ~/workspace/g05-yam-deployment/gello_software
./scripts/safety off     # 交互式：运行后再输入一次 off → DISARM（锁存，需重启 daemon 才能再动）
```

**✅ 通过标准（预期输出）：**
- 输入 `off` 后打印 DISARM（状态机进入 DISARMED；再动需重启 daemon）。

---

## 第 2 步：相机接入（RealSense rig）

> 已定：`gello.cameras.realsense_rig.CameraRig`，三路 `main/left/right`。
> serials 已填：`main=D435 408122072021`；`left=D405 260322274929` / `right=D405 260422272011`
> （★「先左后右」**假设**，尚未物理确认）。
> config：`~/workspace/g05-yam-deployment/gello_software/configs/realsense_bimanual.yaml`（bridge 默认读这份，serials 需填进这里）。

### 2.1 ★ 确认相机 ↔ 观测键对应（真机推理前硬门槛，必做）

> 左右腕反了 = 模型看到左右镜像的双目输入 → 双臂动作错位（安全问题，不是性能问题）。
> 序列号无法区分左右腕，必须**端到端**确认：`left_wrist_rgb` / `right_wrist_rgb` 收到的
> 是否真是左/右腕的图像。此步只碰相机，不碰机械臂。

**推荐 · 挡镜头法（端到端，验证模型实际收到的图）：**

```python
from pathlib import Path
import sys
DEPLOY = Path("~/workspace/g05-yam-deployment").expanduser()
sys.path.insert(0, str(DEPLOY / "g05-async-serving"))
from bridge.vla_client import build_camera_rig, capture_images

rig = build_camera_rig(DEPLOY / "gello_software/configs/realsense_bimanual.yaml")
imgs = capture_images(rig)   # {head_rgb, left_wrist_rgb, right_wrist_rgb}，各 CHW uint8
for name, arr in sorted(imgs.items()):
    print(f"{name:16s} shape={arr.shape}  mean={float(arr.mean()):7.1f}")
rig.close()
```

1. 三台无遮挡跑一次，记三个 `mean` 基线（都亮）。
2. 用手/胶带挡住**左腕镜头**，再跑一次 → `left_wrist_rgb` 的 `mean` 应跌到≈0（变黑）。
   （config `auto_exposure: false` + 固定曝光，挡镜头不会被打增益补亮，测试干净。）
3. 判读：
   - `left_wrist_rgb` 变黑、其余亮 → ✅ 映射正确。
   - `right_wrist_rgb` 变黑而 `left_wrist_rgb` 仍亮 → ❌ **左右反了**，把 config 里 `left`/`right` 两个序列号互换。
4. 反过来挡右腕再跑一次，确认 `right_wrist_rgb` 变黑，交叉验证。

**快查 · 拔线法（仅定位「序列号 ↔ 物理位置」）：**

1. 拔掉左腕 USB，跑枚举 → 剩下的 D405 = right；插回后新出现的 = left。
2. 只确定序列号归属，**仍须用上面的挡镜头法验证 config 映射**。

**✅ 通过标准（预期输出）：**
- 无遮挡：三台 `mean` 都在合理亮度范围（场景正常即可，记下做基线）。
- 挡左腕：`left_wrist_rgb mean≈0`、`right_wrist_rgb`/`head_rgb` 仍亮 → 映射正确。
- 挡右腕：`right_wrist_rgb mean≈0`、`left_wrist_rgb` 仍亮 → 交叉验证通过。
- 若左右反过来（挡左腕变黑的是 right）→ 换 config 里 left/right 序列号，重测直到正确。

### 2.2 枚举 + 采样冒烟

```bash
# 枚举确认（只读，不碰机器人、不启流）：
python - <<'PY'
from gello.cameras.realsense_rig import enumerate_realsense_devices
for d in enumerate_realsense_devices():
    print(f"{d.serial:>16}  {d.model}")
PY

# 采样冒烟：三路同步采到 (480,640,3) uint8 即过。
# bridge 的 build_camera_rig() + capture_images() 已实现（sample → HWC→CHW，映射
# main→head_rgb / left→left_wrist_rgb / right→right_wrist_rgb）。
```

**✅ 通过标准（预期输出）：**
- 枚举打印 3 台，序列号 = `408122072021 (D435)`、`260322274929 (D405)`、`260422272011 (D405)`。
- 采样三路各得 `(480,640,3)` uint8（capture 转 CHW 后 `(3,480,640)`），无 CameraMiss。

---

## 第 3 步：bridge 驱动 G0.5 推理完成任务（主验收）

> `bridge/vla_client.py` 已落地：daemon 客户端 + `SupervisedMotionSession` + 真机相机 + `need_obs` 握手。
>
> ⚠️ **首次真机推理安全姿态**：这是模型第一次真正驱动双臂。动手前确认——操作员手放急停、臂周无遮挡、
> 先用短/温和任务（小幅度）首跑，目视确认动作量级与方向合理后，再上正式任务。

```bash
# 终端 1 —— 服务器（3.10 GalaxeaVLA env）
cd ~/workspace/g05-yam-deployment/GalaxeaVLA
uv run python ~/workspace/g05-yam-deployment/g05-async-serving/serve_policy.py \
    --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
    --host 0.0.0.0 --port 8765 \
    eval_embodiment=galaxea_r1lite model.model_weights_to_bf16=true \
    model.model_arch.continuous_action=false

# 终端 2 —— 真机桥接（3.11 YAM env，g05-async-serving 根目录）
cd ~/workspace/g05-yam-deployment/g05-async-serving
python bridge/vla_client.py --host 127.0.0.1 --port 8765 --task "任务名"
```

**✅ 预期输出（终端）：**
- 终端 1（server）：打印模型加载完成 + `WS listening on 0.0.0.0:8765`（或等价监听日志）。
- 终端 2（bridge）：`WS connected` → `need_obs` 握手成功 → 持续打印下发帧（30Hz，chunk 边界停顿正常）→ 任务执行完正常退出。

验收点：
- [ ] 机器人控制：hold_in_place 通过（ACTIVE / `live_max_dev≈0` / Ctrl+C→SAFE_HOLD）
- [ ] 相机：三路同步采样、形状 `(3,480,640) uint8` 对齐模型 shape_meta；★左右腕对应已确认（§2.1 挡镜头法，硬门槛）
- [ ] bridge：WS 连接 + `need_obs` 握手 + 30Hz 下发（chunk 边界阻塞是正常，不误判卡死）
- [ ] **★ G0.5 推理正确性**：输出动作量级合理、符号正确（codec 已 YAM 微调 = `action_tokenizer_yam.pt`；此验收点 = 真机首跑目视量级/方向合理，不再是 slot 对齐风险）
- [ ] 急停有效：Ctrl+C → SAFE_HOLD；`scripts/safety off`（再输入 off）→ DISARM
- [ ] 任务指令正确下发（`[Low]: {task}`，英文）

