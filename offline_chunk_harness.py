#!/usr/bin/env python
"""Offline harness: drive ChunkedPolicyWrapper with the real model, no WebSocket.

Loads the same model/processor as ``scripts/serve_policy.py``, then runs the
Plan B ``ChunkedPolicyWrapper`` directly for ``--steps`` control steps, feeding
synthetic observations whose shapes are read from the processor's shape_meta (so
it adapts to whatever embodiment you serve). This isolates the chunk/timing logic
from the network layer for a quick sanity check before the end-to-end WS test.

Optionally compare against RTC (``--compare_rtc``) to quantify how much the
look-ahead hides inference latency.

Usage:
    python scripts/offline_chunk_harness.py \
        --ckpt_path <run_dir>/checkpoints/step_X/model_state_dict.pt \
        --action_steps 16 --steps 64 --rate 30 [--compare_rtc]

Needs the g05 env + GPU. ``--rate 0`` runs as fast as possible (see
fake_policy_client.py for why rate matters for the look-ahead window).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import time
from pathlib import Path

import numpy as np

# Load serve_policy.py as a module to reuse its setup()/config-loading logic.
_SERVE_POLICY = Path(__file__).resolve().parent / "serve_policy.py"
_spec = importlib.util.spec_from_file_location("_serve_policy", _SERVE_POLICY)
_sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)


def _dim(m):
    s = m["raw_shape"]
    return tuple(s) if isinstance(s, (list, tuple)) else s


def build_synthetic_obs(processor, task: str, frequency: float = 30.0) -> dict:
    """Build a zero-valued obs matching the processor's shape_meta."""
    if isinstance(processor, _sp.MixtureProcessor):
        emb = next(iter(processor.processors))
        sub = processor.processors[emb]
    else:
        emb = None
        sub = processor

    sm = sub.shape_meta
    obs = {
        "images": {m["key"]: np.zeros(_dim(m), dtype=np.uint8) for m in sm["images"]},
        "state": {m["key"]: np.zeros(_dim(m), dtype=np.float32) for m in sm["state"]},
        "task": task,
        "frequency": float(frequency),
    }
    if emb is not None:
        obs["embodiment_type"] = emb
    return obs


SLOW_MS = 20.0


def summarize(records, wall_s, label):
    if not records:
        return
    lats = [r[1] for r in records]
    slow = [r for r in records if r[1] >= SLOW_MS]
    triggers = [r[0] for r in records if r[2]]
    max_lat, max_step = max((r[1], r[0]) for r in records)
    avg = sum(lats) / len(lats)
    p99 = sorted(lats)[int(len(lats) * 0.99) - 1] if len(lats) > 1 else lats[0]
    print(f"\n=== {label} ===")
    print(f"steps={len(records)}  wall={wall_s:.2f}s  effective_rate={len(records)/wall_s:.1f}Hz")
    print(f"latency: avg={avg:.1f}ms  p99={p99:.1f}ms  max={max_lat:.1f}ms @ step {max_step}")
    print(f"first-step latency (inference baseline): {records[0][1]:.1f}ms")
    print(f"real-obs steps (chunk triggers): {triggers}")
    print(f"slow steps (>= {SLOW_MS}ms, blocked recompute/swap): {[r[0] for r in slow]}")


async def run_wrapper(wrapper, processor, steps, rate, task, verbose):
    period = 1.0 / rate if rate > 0 else 0.0
    records = []
    need_obs = True
    t_start = time.monotonic()
    next_tick = t_start

    for step in range(steps):
        obs = build_synthetic_obs(processor, task, frequency=(rate if rate > 0 else 30.0)) if need_obs else {}
        t0 = time.monotonic()
        action, _ = await wrapper.get_action(obs)
        dt = (time.monotonic() - t0) * 1000
        sent_real = bool(obs)
        records.append((step, dt, sent_real))
        need_obs = wrapper.need_obs

        if verbose or sent_real or dt >= SLOW_MS or step == 0:
            tag = ("OBS " if sent_real else "    ") + ("SLOW" if dt >= SLOW_MS else "")
            print(f"step {step:4d}: {dt:8.1f}ms  {tag}  need_obs={need_obs}")

        if rate > 0:
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    return records, time.monotonic() - t_start


def main():
    ap = argparse.ArgumentParser(description="Offline ChunkedPolicyWrapper harness")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--action_steps", type=int, default=16, help="16 = Plan B; 1 = RTC")
    ap.add_argument("--steps", type=int, default=64, help="control steps (64 = 2 chunks)")
    ap.add_argument("--rate", type=float, default=30.0, help="control Hz (30 = YAM data_fps); 0 = as fast as possible")
    ap.add_argument("--task", default="fold the towel")
    ap.add_argument("--compare_rtc", action="store_true", help="also run action_steps=1 baseline")
    ap.add_argument("--verbose", action="store_true")
    args, remaining = ap.parse_known_args()
    overrides = [r for r in remaining if "=" in r]

    # Mirrors serve_policy.main(): resolve run dir + config, then setup().
    run_dir = _sp.find_run_dir(args.ckpt_path)
    cfg = _sp.load_config_from_run_dir(run_dir, args.ckpt_path, overrides)
    eval_embodiment = cfg.get("eval_embodiment", None)
    if eval_embodiment and "embodiment_datasets" in cfg.data:
        _sp.filter_embodiment(cfg, eval_embodiment)

    print(f"[harness] loading model + processor (device={args.device})...")
    policy, processor = _sp.setup(cfg, device=args.device)
    inferencer = _sp.PolicyInferencer(policy, processor, device=args.device)

    wrapper = _sp.ChunkedPolicyWrapper(
        _sp._make_compute_chunk(inferencer, processor), action_steps=args.action_steps
    )
    records, wall = asyncio.run(
        run_wrapper(wrapper, processor, args.steps, args.rate, args.task, args.verbose)
    )
    summarize(records, wall, f"chunk action_steps={args.action_steps} ({args.steps} steps @ {args.rate}Hz)")

    if args.compare_rtc:
        print("\n[harness] running RTC baseline (action_steps=1)...")
        wrapper_rtc = _sp.ChunkedPolicyWrapper(
            _sp._make_compute_chunk(inferencer, processor), action_steps=1
        )
        records_rtc, wall_rtc = asyncio.run(
            run_wrapper(wrapper_rtc, processor, args.steps, args.rate, args.task, args.verbose)
        )
        summarize(records_rtc, wall_rtc, f"RTC action_steps=1 ({args.steps} steps @ {args.rate}Hz)")


if __name__ == "__main__":
    main()
