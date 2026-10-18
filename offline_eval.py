#!/usr/bin/env python
"""Offline real-data evaluation: baseline (sync full-chunk) vs async look-ahead.

Feeds **real recorded LeRobot observations** (converted by
``tools/yam/mcap_to_lerobot.py``) through the exact same model/processor as
``serve_policy.py``, and drives two ``ChunkedPolicyWrapper`` modes over the same
frames:

  baseline   = G05 original: serve one full action chunk, then infer the next
               chunk synchronously at the swap point. Expressed as
               ``action_steps == chunk_len``, which turns look-ahead off.
  lookahead  = Plan B: overlap the next chunk's inference behind the current
               chunk's tail (``action_steps == chunk_len/2``, default 16).

Two axes are reported per mode:

  1. 卡顿 (stutter): per-step serve latency (mean/p50/p95/max + SLOW count) and
     the inference time T_infer (mean/p95/max). Because we drive the wrapper as
     fast as possible, a step's serve latency == how long the real robot would
     stall at that control step (every step must fit in 1/30Hz = 33 ms).
  2. action error (success proxy): |served action - GT action| L1 per body part,
     in raw joint space. GT is the recorded ``/action/accepted`` (LeRobot
     ``action.*`` step 0 of each frame).

This is an OPEN-LOOP proxy, **not** a success rate: there is no environment
feedback, so "action error small" is a necessary-not-sufficient sanity check
("动作误差不要太大" is the acceptance bar here, not a precision metric).

Caveats:
  - Absolute latency is meaningful only on the deployment GPU (工控机); on the
    dev box it is a correctness/regression signal.
  - The dataset dirs come from the checkpoint's run-dir config (already points
    at ``data/yam/...``). Override with ``dataset_dirs=[...]`` if needed.

Usage:
    python offline_eval.py \
        --ckpt_path <run_dir>/checkpoints/step_X/model_state_dict.pt \
        [--action_steps 16] [--steps 0] [--max_episodes 0] [--eval_split val] \
        [eval_embodiment=galaxea_r1lite]

Needs the g05 env + GPU + the converted LeRobot data present.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse serve_policy.py's config-loading / model-loading / inference path.
_SERVE_POLICY = Path(__file__).resolve().parent / "serve_policy.py"
_spec = importlib.util.spec_from_file_location("_serve_policy", _SERVE_POLICY)
_sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)

from g05.utils.data.processor_utils import instantiate_dataset  # noqa: E402

SLOW_MS = 20.0


def _to_np(x):
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _raw_to_obs_and_gt(raw: dict, embodiment: str | None) -> tuple[dict, dict]:
    """Convert a raw dataset sample (processor=None) into a serve_policy raw_obs
    plus the ground-truth action (step 0 of the action chunk, raw joint space).

    Raw sample layout (base_lerobot_dataset.__getitem__, processor=None):
      images[key]: Tensor[obs_size=1, C, H, W] uint8
      state[key]:  Tensor[obs_size=1, D] float
      action[key]: Tensor[action_size, D] float (raw physical; step 0 = this frame)
      frequency:   model_fps (30 for YAM)
      task:        str
    """
    obs = {"images": {}, "state": {}, "task": raw["task"]}
    for k, v in raw["images"].items():
        arr = _to_np(v[-1])  # [C, H, W]
        obs["images"][k] = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
    for k, v in raw["state"].items():
        arr = _to_np(v[-1])  # [D]
        obs["state"][k] = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    obs["frequency"] = float(raw.get("frequency", 30.0))
    if embodiment is not None:
        obs["embodiment_type"] = embodiment

    gt = {}
    for k, v in raw["action"].items():
        arr = _to_np(v[0])  # [D] — expert action issued at this frame
        gt[k] = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    return obs, gt


def _make_timed_compute(inferencer, processor, record: list):
    """Wrap serve_policy's compute closure to record each inference's wall time."""
    base = _sp._make_compute_chunk(inferencer, processor)

    async def compute(raw_obs):
        t0 = time.monotonic()
        res = await base(raw_obs)
        record.append((time.monotonic() - t0) * 1000)
        return res

    return compute


async def _drive(wrapper, episodes) -> tuple[list, list]:
    """Drive a wrapper over real frames, resetting at each episode boundary.

    Returns (served_actions, per_step_latency_ms) in frame order.
    """
    served = []
    lat_ms = []
    for frames in episodes:
        wrapper.reset()
        need_obs = True
        for obs, _gt in frames:
            o = obs if need_obs else {}
            t0 = time.monotonic()
            action, _cot = await wrapper.get_action(o)
            lat_ms.append((time.monotonic() - t0) * 1000)
            served.append(action)
            need_obs = wrapper.need_obs
    return served, lat_ms


def _latency_summary(lat_ms: list) -> dict:
    a = np.asarray(lat_ms, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
        "slow_steps": int((a >= SLOW_MS).sum()),
    }


def _error_summary(served: list, gt_list: list) -> tuple[dict, float]:
    per_part = defaultdict(list)
    for s, g in zip(served, gt_list):
        for part, gv in g.items():
            if part in s:
                per_part[part].append(float(np.abs(s[part] - gv).mean()))
    parts = {k: float(np.mean(v)) for k, v in per_part.items()}
    overall = float(np.mean(list(parts.values()))) if parts else float("nan")
    return parts, overall


def _fmt_infer(ms: list) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean_ms": float(a.mean()),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser(description="Offline baseline vs look-ahead eval on real data")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--action_steps", type=int, default=16,
        help="look-ahead action_steps (default 16 = chunk_len/2; must be <= chunk_len/2)",
    )
    ap.add_argument("--steps", type=int, default=0, help="max frames to eval (0 = all)")
    ap.add_argument("--max_episodes", type=int, default=0, help="max episodes (0 = all)")
    ap.add_argument("--eval_split", default=None, help="val|train (default: cfg eval_split)")
    ap.add_argument("--output", default="offline_eval_results.json")
    args, remaining = ap.parse_known_args()
    overrides = [r for r in remaining if "=" in r]

    run_dir = _sp.find_run_dir(args.ckpt_path)
    cfg = _sp.load_config_from_run_dir(run_dir, args.ckpt_path, overrides)
    eval_embodiment = cfg.get("eval_embodiment", None)
    if eval_embodiment and "embodiment_datasets" in cfg.data:
        _sp.filter_embodiment(cfg, eval_embodiment)

    print(f"[offline_eval] loading model + processor (device={args.device}) ...")
    policy, processor = _sp.setup(cfg, device=args.device)
    inferencer = _sp.PolicyInferencer(policy, processor, device=args.device)
    chunk_len = int(cfg.data.action_size)

    if args.action_steps > chunk_len // 2:
        print(
            f"[offline_eval] WARNING: action_steps={args.action_steps} > chunk_len/2={chunk_len // 2}; "
            "look-ahead degrades to baseline (sync full-chunk)."
        )

    eval_split = args.eval_split or cfg.get("eval_split", "val")
    is_training_set = eval_split == "train"
    print(f"[offline_eval] loading dataset (split={eval_split}) ...")
    dataset = instantiate_dataset(cfg, is_training_set=is_training_set)
    # YAM is configured as a MixtureLerobotDataset (single embodiment galaxea_r1lite).
    # Reduce it to its one per-embodiment GalaxeaLerobotDataset.
    if dataset.__class__.__name__ == "MixtureLerobotDataset":
        if len(dataset.datasets) != 1:
            raise SystemExit(
                f"Expected 1 embodiment after filtering, got {len(dataset.datasets)}; "
                "pass eval_embodiment=<type> to reduce to a single embodiment."
            )
        dataset = dataset.datasets[0]
    dataset.processor = None  # fetch raw samples (no normalization)

    # ---- Materialize frames grouped by episode (for wrapper.reset()) ----
    start = int(getattr(dataset, "_start_idx", 0))
    n = len(dataset)
    ep_from = [int(x) for x in np.asarray(dataset.episode_data_index["from"])]
    ep_to = [int(x) for x in np.asarray(dataset.episode_data_index["to"])]
    ep_starts = set()
    for f, t in zip(ep_from, ep_to):
        if t > start and f < start + n:
            ep_starts.add(max(f, start) - start)

    episodes = []
    cur = []
    total = 0
    for idx in range(n):
        if args.max_episodes and len(episodes) >= args.max_episodes:
            break
        if idx in ep_starts and cur:
            episodes.append(cur)
            cur = []
        cur.append(_raw_to_obs_and_gt(dataset[idx], eval_embodiment))
        total += 1
        if args.steps and total >= args.steps:
            break
    if cur:
        episodes.append(cur)

    if total == 0:
        raise SystemExit(
            f"No frames in split '{eval_split}' (YAM val_set_proportion is tiny). "
            "Pass --eval_split train."
        )
    print(f"[offline_eval] {len(episodes)} episodes, {total} frames")

    # ---- Run both modes ----
    results = {}
    for name, a_steps in (("baseline", chunk_len), ("lookahead", args.action_steps)):
        record = []
        wrapper = _sp.ChunkedPolicyWrapper(
            _make_timed_compute(inferencer, processor, record), action_steps=a_steps
        )
        served, lat_ms = asyncio.run(_drive(wrapper, episodes))
        gt_list = [gt for ep in episodes for _obs, gt in ep]
        parts, overall = _error_summary(served, gt_list)
        results[name] = {
            "action_steps": a_steps,
            "frames": len(served),
            "infer_ms": _fmt_infer(record),
            "serve_ms": _latency_summary(lat_ms),
            "action_error_l1": {"overall": overall, "per_part": parts},
        }

    # ---- Report ----
    b, l = results["baseline"], results["lookahead"]
    print("\n================= offline eval (open-loop proxy) =================")
    for name, r in (("baseline (sync full-chunk)", b), ("lookahead (Plan B)", l)):
        s = r["serve_ms"]
        print(f"\n[{name}]  action_steps={r['action_steps']}  frames={r['frames']}")
        print(
            f"  serve latency: mean={s['mean_ms']:.1f}ms  p50={s['p50_ms']:.1f}ms  "
            f"p95={s['p95_ms']:.1f}ms  max={s['max_ms']:.1f}ms  slow(>={SLOW_MS:.0f}ms)={s['slow_steps']}"
        )
        im = r["infer_ms"]
        print(
            f"  T_infer:       mean={im['mean_ms']:.1f}ms  p95={im['p95_ms']:.1f}ms  "
            f"max={im['max_ms']:.1f}ms  (n={im['n']})"
        )
        ae = r["action_error_l1"]
        print(f"  action L1 vs GT: overall={ae['overall']:.5f}")
        for part, v in ae["per_part"].items():
            print(f"      {part:16s} {v:.5f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
