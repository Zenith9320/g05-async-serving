"""Unit tests for the pure joint-mapping helpers in bridge/vla_client.py.

These verify the *identity* passthrough that is the correctness core of the YAM
deployment (see TODO.md 待补信息): 14-dim = [left_arm(6), left_gripper(1),
right_arm(6), right_gripper(1)], no reorder, no sign flip, absolute values.

No gello / cameras / websockets / GPU needed — only numpy + msgpack.
"""

import numpy as np
import pytest

from bridge.vla_client import (
    ACTION_KEYS,
    EMBODIMENT_TYPE,
    IMAGE_SPECS,
    NUM_JOINTS,
    STATE_SPLIT,
    action_to_target,
    build_obs,
    split_state,
)


def _arbitrary_state():
    return np.arange(NUM_JOINTS, dtype=np.float32) + 0.5


def test_split_state_identity_layout():
    state = _arbitrary_state()
    parts = split_state(state)
    assert list(parts) == ACTION_KEYS
    np.testing.assert_array_equal(parts["left_arm"], state[0:6])
    np.testing.assert_array_equal(parts["left_gripper"], state[6:7])
    np.testing.assert_array_equal(parts["right_arm"], state[7:13])
    np.testing.assert_array_equal(parts["right_gripper"], state[13:14])


def test_split_state_rejects_wrong_dim():
    with pytest.raises(ValueError):
        split_state(np.zeros(13, dtype=np.float32))


def test_split_state_accepts_any_input_array():
    # A python list must be coerced to (14,) float32, matching get_observations().
    parts = split_state([0.0] * NUM_JOINTS)
    assert parts["left_arm"].shape == (6,)
    assert parts["left_arm"].dtype == np.float32


def test_action_to_target_roundtrip():
    state = _arbitrary_state()
    target = action_to_target(split_state(state), last_target=np.zeros(NUM_JOINTS, dtype=np.float32))
    np.testing.assert_array_equal(target, state)


def test_action_to_target_holds_missing_parts():
    last = _arbitrary_state()
    # Server drops left_gripper and right_arm (AR _absent_keys); client must hold.
    action = split_state(last)
    del action["left_gripper"]
    del action["right_arm"]
    target = action_to_target(action, last_target=last)
    np.testing.assert_array_equal(target, last)


def test_action_to_target_rejects_wrong_dim():
    with pytest.raises(ValueError):
        action_to_target({"left_arm": np.zeros(5, dtype=np.float32)}, last_target=np.zeros(NUM_JOINTS))


def test_build_obs_schema():
    images = {k: np.zeros(shape, dtype=np.uint8) for k, shape in IMAGE_SPECS.items()}
    obs = build_obs(_arbitrary_state(), images, "fold the towel", 30)
    assert obs["embodiment_type"] == EMBODIMENT_TYPE
    assert obs["task"] == "fold the towel"
    assert obs["frequency"] == 30.0 and isinstance(obs["frequency"], float)
    assert set(obs["images"]) == set(images)
    for key in ACTION_KEYS:
        assert obs["state"][key].dtype == np.float32
    assert obs["state"]["left_arm"].shape == (6,)
    assert obs["state"]["right_gripper"].shape == (1,)
