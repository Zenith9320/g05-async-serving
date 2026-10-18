#!/usr/bin/env python
"""Fake policy client — stress-test the Plan B chunk server without a robot.

Connects to ``scripts/serve_policy.py`` and streams synthetic observations at a
fixed control rate, honouring the server's ``need_obs`` flag (send a real obs only
when requested, an empty obs otherwise). This is exactly the protocol the real
robot client uses, so it validates the *timing* behaviour of the async look-ahead
(Plan B), not action quality.

What to look for in the per-step log / summary:

  * step 0          -> full inference latency (this is the per-step cost of RTC)
  * steps 1..31     -> ~0 ms (served from the cached chunk)
  * step 32 (swap)  -> ~0 ms if the background inference finished inside the 16-step
                       tail (look-ahead worked), or ~full inference latency if it
                       overran the tail (timeout fallback)

``--rate`` must match the real robot cadence — 30 Hz for YAM (the training data is
resampled onto the 30 Hz camera grid, see tools/yam/mcap_to_lerobot.py). The
look-ahead has ``chunk_len - action_steps`` = 16 steps to hide one inference; at
30 Hz that is a ~0.53 s window. If ``--rate 0`` (as fast as possible) the tail
collapses to ~0 s, so the swap always hits the timeout fallback — only use rate 0
to measure raw latency.

Usage:
    # terminal 1 — server (needs a finetune run dir: .hydra/config.yaml,
    #              dataset_stats.json, action_tokenizer.pt)
    python scripts/serve_policy.py \
        --ckpt_path <run_dir>/checkpoints/step_X/model_state_dict.pt --port 8765

    # terminal 2 — client
    python scripts/fake_policy_client.py --host 127.0.0.1 --port 8765 --steps 64 --rate 30

Only needs ``websockets``, ``msgpack``, ``numpy`` (all in the g05 env).
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import time

import msgpack
import numpy as np
import websockets


# ─── msgpack + numpy codec (wire-compatible with g05.utils.websocket) ─────────

def _pack(obj):
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tobytes(), "dtype": obj.dtype.str, "shape": obj.shape}
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _bytes_to_str_keys(obj):
    if isinstance(obj, dict):
        return {(k.decode() if isinstance(k, bytes) else k): _bytes_to_str_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_bytes_to_str_keys(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_bytes_to_str_keys(v) for v in obj)
    return obj


def _unpack(obj):
    obj = _bytes_to_str_keys(obj)
    if "__ndarray__" in obj:
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]), shape=tuple(obj["shape"]))
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)


# ─── synthetic observation (must match the embodiment being served) ───────────
# Edit these to match the embodiment you pass to serve_policy.py. The defaults are
# galaxea_r1lite / YAM (configs/data/yam.yaml): 3x [C,H,W] uint8 images + 4x 1-D
# state vectors.

EMBODIMENT_TYPE = "galaxea_r1lite"
IMAGE_SPECS = {
    "head_rgb": (3, 480, 640),
    "left_wrist_rgb": (3, 480, 640),
    "right_wrist_rgb": (3, 480, 640),
}
STATE_SPECS = {
    "left_arm": 6,
    "left_gripper": 1,
    "right_arm": 6,
    "right_gripper": 1,
}


def build_obs(task: str, frequency: float) -> dict:
    return {
        "embodiment_type": EMBODIMENT_TYPE,
        "images": {k: np.zeros(shape, dtype=np.uint8) for k, shape in IMAGE_SPECS.items()},
        "state": {k: np.zeros(d, dtype=np.float32) for k, d in STATE_SPECS.items()},
        "task": task,
        # Control frequency in Hz (YAM = 30), encoded by the model as a per-sample
        # input. The real client sends action_fps here too; the server falls back
        # to 15 if omitted, which is wrong for YAM.
        "frequency": float(frequency),
    }


def summarize(records, wall_s, label):
    """records: list[(step, latency_ms, sent_real_obs)]. Print a compact summary."""
    if not records:
        return
    lats = [r[1] for r in records]
    slow = [r for r in records if r[1] >= SLOW_MS]
    triggers = [r[0] for r in records if r[2]]
    max_lat, max_step = max((r[1], r[0]) for r in records)
    avg = sum(lats) / len(lats)
    p99 = sorted(lats)[int(len(lats) * 0.99) - 1] if len(lats) > 1 else lats[0]
    print(f"\n=== {label} summary ===")
    print(f"steps={len(records)}  wall={wall_s:.2f}s  effective_rate={len(records)/wall_s:.1f}Hz")
    print(f"latency: avg={avg:.1f}ms  p99={p99:.1f}ms  max={max_lat:.1f}ms @ step {max_step}")
    print(f"first-step latency (inference baseline): {records[0][1]:.1f}ms")
    print(f"real-obs sent at steps (chunk triggers): {triggers}")
    print(f"slow steps (>= {SLOW_MS}ms, i.e. blocked recompute/swap): {[r[0] for r in slow]}")


SLOW_MS = 20.0


async def run(host, port, steps, rate, task, verbose):
    period = 1.0 / rate if rate > 0 else 0.0
    records = []

    async with websockets.connect(f"ws://{host}:{port}", max_size=None) as ws:
        # First server message is the action_steps announcement.
        hello = unpackb(await ws.recv())
        print(f"[client] connected, server action_steps={hello.get('action_steps')}")

        need_obs = True
        t_start = time.monotonic()
        next_tick = t_start

        for step in range(steps):
            obs = build_obs(task, rate if rate > 0 else 30.0) if need_obs else {}
            t0 = time.monotonic()
            await ws.send(packb(obs))
            resp = unpackb(await ws.recv())
            dt = (time.monotonic() - t0) * 1000

            if "error" in resp:
                print(f"step {step:4d}: SERVER ERROR {resp['error']}")
                break

            sent_real = bool(obs)
            records.append((step, dt, sent_real))

            tag = []
            if sent_real:
                tag.append("OBS")
            if dt >= SLOW_MS:
                tag.append("SLOW")
            if step == 0:
                tag.append("FIRST")
            line = f"step {step:4d}: {dt:8.1f}ms  {' '.join(tag)}"
            if not sent_real:
                line += f"  need_obs={resp.get('need_obs')}"
            if step == 0:
                shapes = {k: tuple(v.shape) for k, v in resp.get('action', {}).items() if isinstance(v, np.ndarray)}
                line += f"  action={shapes}"
            if verbose or sent_real or dt >= SLOW_MS or step == 0:
                print(line)

            need_obs = bool(resp.get("need_obs", True))

            if rate > 0:
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif verbose:
                    print(f"step {step:4d}: behind schedule by {-delay*1000:.0f}ms")

    summarize(records, time.monotonic() - t_start, f"ws://{host}:{port} ({steps} steps @ {rate}Hz)")


def main():
    ap = argparse.ArgumentParser(description="Fake policy client (synthetic obs)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--steps", type=int, default=64, help="control steps to run (64 = 2 chunks)")
    ap.add_argument("--rate", type=float, default=30.0, help="control Hz (30 = YAM data_fps); 0 = as fast as possible")
    ap.add_argument("--task", default="fold the towel", help="VLM task instruction")
    ap.add_argument("--verbose", action="store_true", help="print every step")
    args = ap.parse_args()

    asyncio.run(run(args.host, args.port, args.steps, args.rate, args.task, args.verbose))


if __name__ == "__main__":
    main()
