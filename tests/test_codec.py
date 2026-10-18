"""Unit tests for the msgpack + numpy codec in fake_policy_client.py.

These are wire-compatible with g05.utils.websocket, so a round-trip here
guarantees the synthetic observations the client sends survive the transport.
No GPU / g05 / torch needed — only msgpack + numpy.
"""

import numpy as np

from fake_policy_client import packb, unpackb


def test_ndarray_roundtrip():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = unpackb(packb(a))
    assert isinstance(b, np.ndarray)
    assert b.dtype == a.dtype
    assert b.shape == a.shape
    assert np.array_equal(b, a)


def test_ndarray_uint8_preserves_dtype():
    a = np.zeros((3, 4, 5), dtype=np.uint8)
    b = unpackb(packb(a))
    assert b.dtype == np.uint8
    assert b.shape == (3, 4, 5)


def test_np_generic_roundtrip():
    x = np.float32(3.5)
    y = unpackb(packb(x))
    assert isinstance(y, np.float32)
    assert float(y) == 3.5


def test_nested_dict_roundtrip():
    obj = {
        "images": {"head_rgb": np.zeros((3, 4, 5), dtype=np.uint8)},
        "state": {"left_arm": np.zeros(6, dtype=np.float32)},
        "task": "fold",
        "frequency": 30.0,
    }
    r = unpackb(packb(obj))
    assert np.array_equal(r["images"]["head_rgb"], obj["images"]["head_rgb"])
    assert np.array_equal(r["state"]["left_arm"], obj["state"]["left_arm"])
    assert r["task"] == "fold"
    assert r["frequency"] == 30.0


def test_bytes_keys_decoded_to_str():
    obj = {b"__reset__": True, b"action": [1, 2, 3]}
    r = unpackb(packb(obj))
    assert "__reset__" in r
    assert r["action"] == [1, 2, 3]
