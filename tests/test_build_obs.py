"""Unit tests for build_obs in fake_policy_client.py.

Verifies the synthetic observation matches the served embodiment schema
(galaxea_r1lite / YAM): 3x [3,480,640] uint8 images, 4x 1-D float32 state
vectors, and the ``frequency`` field that the model encodes as a per-sample
input. No GPU / g05 / torch needed — only numpy.
"""

import numpy as np

from fake_policy_client import (
    EMBODIMENT_TYPE,
    IMAGE_SPECS,
    STATE_SPECS,
    build_obs,
)


def test_images_shapes_and_dtype():
    obs = build_obs("fold the towel", frequency=30.0)
    assert set(obs["images"]) == set(IMAGE_SPECS)
    for key, (c, h, w) in IMAGE_SPECS.items():
        arr = obs["images"][key]
        assert arr.shape == (c, h, w)
        assert arr.dtype == np.uint8


def test_state_shapes_and_dtype():
    obs = build_obs("fold", 30.0)
    assert set(obs["state"]) == set(STATE_SPECS)
    for key, d in STATE_SPECS.items():
        arr = obs["state"][key]
        assert arr.shape == (d,)
        assert arr.dtype == np.float32


def test_meta_fields():
    obs = build_obs("fold", frequency=30.0)
    assert obs["embodiment_type"] == EMBODIMENT_TYPE
    assert obs["task"] == "fold"
    assert obs["frequency"] == 30.0
    assert isinstance(obs["frequency"], float)
