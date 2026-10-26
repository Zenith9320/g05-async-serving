#!/usr/bin/env python3
"""Minimal daemon-client hold test: read joints, command to the current position, hold.

Validates the exact daemon path the bridge now uses — ``resolve_safety_config`` +
``connect_safety`` + ``SupervisedMotionSession`` — **without moving the arm**: it
activates the session, then repeatedly commands the robot to the position it was
already at, so arm deltas are ~0 and the daemon's slew limiter never trips.

This exercises, end-to-end and for real:
  * daemon startup / reuse (``ensure_safety_daemon``) and watchdog gate,
  * the two heartbeat leases (parent + collector) keeping the robot ACTIVE,
  * the ``command_joint_state`` round-trip through ``SafetyController``,
  * clean Ctrl+C → confirmed SAFE HOLD.

Run in the YAM 3.11 env, same as ``yam return-zero`` / the bridge:

    python bridge/hold_in_place.py                 # resolve from machine profile
    python bridge/hold_in_place.py --seconds 10    # bounded auto-exit (also tests clean exit)

Teardown (two distinct states):
  1. Ctrl+C ends the session → the daemon issues a confirmed **SAFE HOLD** — motors
     are still energized and actively holding position.
  2. To fully de-energize, run ``yam safety off`` (interactive, type ``off``) → the
     daemon **DISARMs** and cuts motor power. DISARM is latched: to move again you
     must restart the daemon (e.g. ``yam return-zero``).
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import numpy as np

# Make the repo root importable when run as `python bridge/hold_in_place.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.vla_client import RATE, connect_safety, resolve_safety_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Hold-in-place smoke test of the safety daemon client path")
    ap.add_argument("--profile", type=Path, help="machine profile (defaults to YAM_PROFILE/XDG)")
    ap.add_argument("--left-config", type=Path)
    ap.add_argument("--right-config", type=Path)
    ap.add_argument("--hardware-port", type=int)
    ap.add_argument("--safety-port", type=int)
    ap.add_argument("--seconds", type=float, default=0.0, help="bounded hold; 0 = until Ctrl+C")
    args = ap.parse_args()

    robot = safety = None
    try:
        left_config, right_config, hardware_port, safety_port = resolve_safety_config(ap, args)

        import gello
        repo_root = Path(gello.__file__).resolve().parents[1]

        robot, safety, status = connect_safety(
            repo_root=repo_root,
            python=sys.executable,
            left_config=left_config,
            right_config=right_config,
            hardware_port=hardware_port,
            safety_port=safety_port,
        )
        limits = np.asarray(safety.status()["joint_limits"], dtype=np.float32)
        print(f"safety state : {status['state'].upper()}  ({status['reason']})")
        print(f"joint limits  : shape={limits.shape}")

        from gello.safety.motion_session import SupervisedMotionSession

        session_id = uuid.uuid4().hex
        with SupervisedMotionSession(safety, session_id, operation="hold-in-place-test") as session:
            hold = np.asarray(robot.get_joint_state(), dtype=np.float32).reshape(-1)
            print(f"hold target   : {np.array2string(hold, precision=4)}")
            print("holding in place — Ctrl+C to end\n")

            period = 1.0 / RATE
            next_tick = time.monotonic()
            start = time.monotonic()
            ticks = 0
            while args.seconds <= 0 or time.monotonic() - start < args.seconds:
                session.check()

                # No-op hold: command exactly where the arm already is.
                result = robot.command_joint_state(hold)
                applied = (
                    np.asarray(result["applied_joint_state"], dtype=np.float32)
                    if isinstance(result, dict) and "applied_joint_state" in result
                    else None
                )

                ticks += 1
                if ticks == 1 or ticks % RATE == 0:
                    live = np.asarray(robot.get_joint_state(), dtype=np.float32).reshape(-1)
                    dev = float(np.abs(live - hold).max())
                    app_dev = float(np.abs(applied - hold).max()) if applied is not None else float("nan")
                    print(
                        f"t={time.monotonic() - start:5.1f}s  live_max_dev={dev:.4f} rad  "
                        f"applied_dev={app_dev:.4f} rad  limited={result.get('limited')}  "
                        f"consec_limited={result.get('consecutive_limited')}"
                    )

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

        print("session exited cleanly → SAFE HOLD issued")
        return 0
    except KeyboardInterrupt:
        # The `with` block's __exit__ already ran and issued SAFE HOLD before we
        # get here; only report it and do NOT disarm.
        print("\ninterrupted — SupervisedMotionSession already issued SAFE HOLD.")
        print("motors are still energized (holding). To cut power:  yam safety off   (then type 'off')")
        return 130
    finally:
        if robot is not None:
            robot.close()  # ZMQ socket only; the daemon keeps owning motors
        if safety is not None:
            safety.close()


if __name__ == "__main__":
    raise SystemExit(main())
