#!/usr/bin/env python3
"""Real-machine bridge: cameras + joint state → ``serve_policy`` (WS) → YAM bimanual.

Runs in the **YAM control stack env (Python 3.11)**, separate from the GalaxeaVLA
3.10 env that hosts ``serve_policy.py``; the two processes talk over WebSocket +
msgpack. This is the exact client protocol validated by ``fake_policy_client.py``,
now with real joint observations and real action command instead of synthetic data.

Data contract (all resolved — see TODO.md 待补信息 and [[g05-action-codec-alignment]]):

* **Action frame** = follower (i2rt) native frame. Recorded actions are
  ``applied_joint_state``; leader raw values live separately in ``leader_action``.
* **Joint order/sign = identity**: 14-dim = ``[left_arm(6), left_gripper(1),
  right_arm(6), right_gripper(1)]``. No reorder, no sign flip — the bridge is a
  pure passthrough.
* **Action value = absolute**: ``serve_policy`` postprocess already does
  ``merger.backward`` (unpad) → ``normalizer.backward`` (denorm) →
  ``RelativeJointTransform.backward`` (relative→absolute). The client must NOT add
  state back on.
* **Obs to send**: ``{embodiment_type, images, state, task, frequency=30.0}`` with
  3 images ``(3,480,640) uint8`` and 4 state vectors ``(6,1,6,1) float32``.
* **Action received**: ``resp["action"]`` is ``{left_arm, left_gripper, right_arm,
  right_gripper}`` of 1-D ndarrays. Keys the AR head dropped (``_absent_keys``) are
  absent from the dict — the client holds the last commanded value for them.

Safety: the bridge is a supervised client of the persistent safety daemon — a
``SupervisedMotionSession`` holds the daemon's two heartbeat leases and issues a
confirmed SAFE HOLD on exit. It never opens the CAN buses directly: ``create_pair``
belongs to the daemon and conflicts with it.

Usage:
    # server (3.10 GalaxeaVLA env, see TODO.md 第 2/6 步)
    uv run python serve_policy.py --ckpt_path ... --host 0.0.0.0 --port 8765 \\
        eval_embodiment=galaxea_r1lite model.model_arch.continuous_action=false

    # client (3.11 YAM env) — resolves follower config/ports from the same machine
    # profile as `yam return-zero` and auto-starts (or reuses) the safety daemon.
    python bridge/vla_client.py --host <ip> --port 8765 --task "fold the towel"
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import msgpack
import numpy as np

logger = logging.getLogger("vla_client")


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


# ─── embodiment schema (matches configs/data/yam.yaml + fake_policy_client) ───

EMBODIMENT_TYPE = "galaxea_r1lite"
RATE = 30  # Hz — YAM data is resampled onto the 30 Hz camera grid
IMAGE_SPECS = {
    "head_rgb": (3, 480, 640),
    "left_wrist_rgb": (3, 480, 640),
    "right_wrist_rgb": (3, 480, 640),
}
# configs/realsense_bimanual.yaml camera name → observation key the server expects.
CAMERA_NAME_MAP = {
    "main": "head_rgb",
    "left": "left_wrist_rgb",
    "right": "right_wrist_rgb",
}
# 14-dim joint layout → per-part slices. Order is the identity mapping confirmed by
# tools/yam/mcap_to_lerobot.py: left_arm=act[:6], left_gripper=act[6],
# right_arm=act[7:13], right_gripper=act[13].
STATE_SPLIT = {
    "left_arm": (0, 6),
    "left_gripper": (6, 7),
    "right_arm": (7, 13),
    "right_gripper": (13, 14),
}
ACTION_KEYS = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
NUM_JOINTS = 14


# ─── pure joint mapping (unit-tested without gello / cameras / websockets) ────

def split_state(state: np.ndarray) -> dict[str, np.ndarray]:
    """14-dim joint vector → ``{left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)}``.

    Identity mapping — no reorder or sign flip.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS}-dim joint state, got {state.shape[0]}")
    return {key: state[lo:hi] for key, (lo, hi) in STATE_SPLIT.items()}


def action_to_target(action: dict, last_target: np.ndarray) -> np.ndarray:
    """Merge the server's action dict back into a 14-dim absolute target.

    Missing keys (dropped by the AR head via ``_absent_keys``) fall back to the
    previous commanded value in ``last_target``, so the robot holds that part.
    """
    last_target = np.asarray(last_target, dtype=np.float32).reshape(-1)
    parts = []
    for key in ACTION_KEYS:
        lo, hi = STATE_SPLIT[key]
        val = action.get(key)
        if val is None:
            parts.append(last_target[lo:hi])
            continue
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        if arr.shape[0] != (hi - lo):
            raise ValueError(f"action['{key}'] has {arr.shape[0]} dims, expected {hi - lo}")
        parts.append(arr)
    return np.concatenate(parts)


def build_obs(state: np.ndarray, images: dict[str, np.ndarray], task: str, frequency: float) -> dict:
    """Build the raw observation dict the server's ``_validate_obs`` expects."""
    return {
        "embodiment_type": EMBODIMENT_TYPE,
        "images": images,
        "state": split_state(state),
        "task": task,
        # ★ must be 30 for YAM — the server falls back to 15 (SO-100) if omitted.
        "frequency": float(frequency),
    }


# ─── cameras (gello CameraRig; API confirmed from gello/cameras/realsense_rig.py) ─

def build_camera_rig(config_path: Path):
    """Load and start the fixed RealSense rig (``CameraRig.start`` warms up 5 frames)."""
    from gello.cameras.realsense_rig import CameraRig, load_camera_rig_config

    rig = CameraRig(load_camera_rig_config(Path(config_path)))
    rig.start(warmup_frames=5, timeout_s=5.0)
    logger.info("Camera rig started: %s", ", ".join(rig.config.cameras))
    return rig


def capture_images(camera_rig, max_retries: int = 5) -> dict[str, np.ndarray]:
    """Capture the three synchronized streams as ``[C,H,W] uint8``.

    ``camera_rig.sample()`` returns a synchronized ``CameraBundle`` (frames keyed
    ``main``/``left``/``right``, each ``frame.rgb`` is HWC ``(480,640,3) uint8``) or a
    ``CameraMiss`` when the bundle is incomplete/stale/skewed. Retry briefly, then fail
    loudly so the bridge holds rather than feeding a stale frame.
    """
    from gello.cameras.realsense_rig import CameraMiss

    for attempt in range(max_retries):
        result = camera_rig.sample(time.monotonic())
        if not isinstance(result, CameraMiss):
            break
        time.sleep(0.005)
    else:
        raise RuntimeError(
            f"camera sync failed after {max_retries} tries: "
            f"{result.reason} ({result.detail})"
        )

    images = {}
    for cam_name, obs_name in CAMERA_NAME_MAP.items():
        rgb = np.asarray(result.frames[cam_name].rgb, dtype=np.uint8)  # HWC (H, W, 3)
        images[obs_name] = np.ascontiguousarray(rgb.transpose(2, 0, 1))  # CHW
    return images


# ─── safety daemon connection (lazy import: gello lives in the 3.11 env) ──────

def load_channels(left_config: Path, right_config: Path) -> dict[str, str]:
    """Read the follower CAN channel names the daemon must already own."""
    import yaml

    channels = {}
    for side, path in (("left", left_config), ("right", right_config)):
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        channels[side] = str(cfg["robot"]["channel"])
    return channels


def connect_safety(
    *,
    repo_root: Path,
    python: str,
    left_config: Path,
    right_config: Path,
    hardware_port: int,
    safety_port: int,
) -> tuple:
    """Ensure the safety daemon owns the CAN chains and return ``(robot, safety, status)``.

    The daemon is the persistent owner of both CAN buses, so the bridge must NOT
    call ``create_pair`` — doing so grabs the same channels and conflicts.
    ``ensure_safety_daemon`` probes first and only spawns if one isn't already
    running (e.g. started by ``yam return-zero``). Mirrors ``scripts/return_zero.py``.
    """
    from gello.safety.daemon_client import ensure_safety_daemon
    from gello.safety.rpc import SafetyClient
    from gello.zmq_core.robot_node import ZMQClientRobot

    left_config = Path(left_config).resolve()
    right_config = Path(right_config).resolve()
    channels = load_channels(left_config, right_config)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p
    )
    env["PYTHONUNBUFFERED"] = "1"
    state_dir = Path(env.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "yam-teleop"
    log_path = state_dir / datetime.now().strftime("vla-safety-%Y%m%d-%H%M%S.log")

    status, _process = ensure_safety_daemon(
        repo_root=repo_root,
        python=python,
        left_config=left_config,
        right_config=right_config,
        channels=channels,
        hardware_port=hardware_port,
        safety_port=safety_port,
        log_path=log_path,
        env=env,
    )
    safety = SafetyClient(port=safety_port)
    robot = ZMQClientRobot(port=hardware_port)
    logger.info("Safety daemon ready: state=%s, num_dofs=%d", status["state"], robot.num_dofs())
    return robot, safety, status


def resolve_safety_config(ap, args):
    """Resolve follower config + ports from the machine profile (as ``yam return-zero``).

    Returns ``(left_config, right_config, hardware_port, safety_port)``. Using the
    same profile/overrides as ``return_zero`` keeps the daemon identity fingerprint
    identical, so the bridge reuses an already-running daemon instead of being refused.
    """
    from gello.command_profile import command_profile, validate_motion_configuration

    profile = command_profile(ap, args.profile)
    left_config = args.left_config or profile.arms["left"].config_path
    right_config = args.right_config or profile.arms["right"].config_path
    hardware_port = args.hardware_port if args.hardware_port is not None else profile.network.hardware_port
    safety_port = args.safety_port if args.safety_port is not None else profile.network.safety_port
    validate_motion_configuration(
        ap,
        {"left": left_config, "right": right_config},
        hardware_port=hardware_port,
        safety_port=safety_port,
    )
    return left_config, right_config, hardware_port, safety_port


# ─── main loop ────────────────────────────────────────────────────────────────

async def run(host, port, task, steps, camera_rig, robot, safety):
    """Drive the robot from ``serve_policy`` under a supervised safety session."""
    import websockets
    from gello.safety.motion_session import SupervisedMotionSession

    images_fn = lambda: capture_images(camera_rig)
    # ZMQClientRobot exposes no get_joint_limits(); the daemon publishes them here.
    limits = np.asarray(safety.status()["joint_limits"], dtype=np.float32)  # (14, 2)

    period = 1.0 / RATE
    session_id = uuid.uuid4().hex

    # The session holds the daemon's two heartbeat leases (parent + collector),
    # activates on enter, and issues a confirmed SAFE HOLD on any exit.
    with SupervisedMotionSession(safety, session_id, operation="vla-inference") as session:
        last_target = np.asarray(
            robot.get_observations()["joint_positions"], dtype=np.float32
        ).reshape(-1)

        async with websockets.connect(f"ws://{host}:{port}", max_size=None) as ws:
            hello = unpackb(await ws.recv())
            logger.info("Connected to %s:%d, server action_steps=%s", host, port, hello.get("action_steps"))

            need_obs = True
            next_tick = time.monotonic()
            step = 0

            while steps <= 0 or step < steps:
                # Surfaces daemon FAULT/DISARMED (latched by the heartbeat threads).
                session.check()

                state = np.asarray(
                    robot.get_observations()["joint_positions"], dtype=np.float32
                ).reshape(-1)

                obs = build_obs(state, images_fn(), task, RATE) if need_obs else {}

                await ws.send(packb(obs))
                resp = unpackb(await ws.recv())

                if "error" in resp:
                    raise RuntimeError(f"server error: {resp['error']}")

                action = resp.get("action", {})
                target = action_to_target(action, last_target)
                target = np.clip(target, limits[:, 0], limits[:, 1])  # soft-limit clamp

                # The daemon slew-limits arms (0.20 rad/frame) and clamps grippers
                # [0,1]; trust its "applied" target so last_target stays honest.
                result = robot.command_joint_state(target)
                if isinstance(result, dict) and "applied_joint_state" in result:
                    last_target = np.asarray(
                        result["applied_joint_state"], dtype=np.float32
                    ).reshape(-1)
                else:
                    last_target = target

                need_obs = bool(resp.get("need_obs", True))
                step += 1

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif step % 30 == 0:
                    logger.warning("Behind schedule by %.0f ms at step %d", -delay * 1000, step)


def main():
    ap = argparse.ArgumentParser(description="Real-machine YAM bridge client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--task", required=True, help="VLM task instruction (English, sent as [Low]: task)")
    ap.add_argument("--steps", type=int, default=0, help="bounded run for smoke tests; 0 = run until Ctrl+C")
    ap.add_argument("--verbose", action="store_true", help="debug logging")
    # safety daemon connection — mirrors scripts/return_zero.py
    ap.add_argument("--profile", type=Path, help="machine profile (defaults to YAM_PROFILE/XDG)")
    ap.add_argument("--left-config", type=Path, help="override left follower config")
    ap.add_argument("--right-config", type=Path, help="override right follower config")
    ap.add_argument("--hardware-port", type=int, help="override daemon hardware RPC port")
    ap.add_argument("--safety-port", type=int, help="override daemon safety RPC port")
    ap.add_argument("--camera-config", type=Path, help="RealSense rig config (default: <gello-root>/configs/realsense_bimanual.yaml)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    robot = safety = camera_rig = None
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
        logger.info("Safety state: %s (%s)", status["state"].upper(), status.get("reason"))

        camera_config = args.camera_config or (repo_root / "configs" / "realsense_bimanual.yaml")
        camera_rig = build_camera_rig(camera_config)

        asyncio.run(
            run(
                args.host, args.port, args.task, args.steps,
                camera_rig=camera_rig, robot=robot, safety=safety,
            )
        )
    except KeyboardInterrupt:
        if safety is not None:
            safety.safe_hold("vla inference cancelled by operator")
        logger.info("Cancelled · followers remain in SAFE HOLD")
    finally:
        if camera_rig is not None:
            camera_rig.close()
        if robot is not None:
            robot.close()  # closes the ZMQ socket only; the daemon keeps owning motors
        if safety is not None:
            safety.close()


if __name__ == "__main__":
    main()
