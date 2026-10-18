"""Unit tests for ChunkedPolicyWrapper's chunk-cache / look-ahead state machine.

Uses a fake ``compute`` callable (no model, no GPU) that returns a ramp where
``chunk[i] = obs_step + i``, i.e. every served value equals its *absolute* control
step. That makes the served stream directly assertable: it must be exactly
``0, 1, 2, ..., steps-1`` — any replay or skipped step shows up as a duplicate or
a gap. This holds for any ``action_steps``, including ``chunk_len != 2*action_steps``.

Only needs ``numpy``.
"""

import asyncio

import numpy as np

from chunked_policy import ChunkedPolicyWrapper


class RampCompute:
    """``chunk[i] = raw_obs['__step__'] + i``; records calls and obs presence."""

    def __init__(self, chunk_len=32, dim=1):
        self.chunk_len = chunk_len
        self.dim = dim
        self.calls = 0
        self.got_real_obs = []

    async def __call__(self, raw_obs):
        self.calls += 1
        self.got_real_obs.append(bool(raw_obs))
        base = float(raw_obs.get("__step__", 0.0))
        arr = (base + np.arange(self.chunk_len, dtype=np.float32)).reshape(
            self.chunk_len, self.dim
        )
        return {"action": arr}, None


async def drive(wrapper, steps, empty_at=()):
    """Drive the wrapper exactly like the real client: send a real obs only when
    ``need_obs`` is True (optionally forcing an empty obs at given steps, to model
    a misbehaving client). ``asyncio.sleep(0)`` lets any background task finish."""
    need_obs = True
    served = []
    real_steps = []
    for step in range(steps):
        obs = {} if (step in empty_at or not need_obs) else {"__step__": float(step)}
        if obs:
            real_steps.append(step)
        action, _cot = await wrapper.get_action(obs)
        served.append(float(action["action"][0]))
        need_obs = wrapper.need_obs
        await asyncio.sleep(0)
    return served, real_steps


def test_lookahead_call_schedule():
    # chunk_len=32, action_steps=16: one sync inference at step 0, then a
    # background inference at the trigger (16) and immediately after each swap
    # (32, 48). 64 steps = 2 full chunks served => 4 inferences total.
    fc = RampCompute(chunk_len=32)
    wrapper = ChunkedPolicyWrapper(fc, action_steps=16)
    served, real_steps = asyncio.run(drive(wrapper, 64))

    assert fc.calls == 4
    assert real_steps == [0, 16, 32, 48]
    assert all(fc.got_real_obs)              # every inference from a real obs
    assert served == list(map(float, range(64)))  # no replay / no gap


def test_general_action_steps_contiguous():
    # chunk_len=32, action_steps=8 (N != 2A): the served stream must still be a
    # contiguous 0..63 with no replay/gap — the generalization must not only work
    # at the special 2:1 ratio.
    fc = RampCompute(chunk_len=32)
    wrapper = ChunkedPolicyWrapper(fc, action_steps=8)
    served, real_steps = asyncio.run(drive(wrapper, 64))

    assert served == list(map(float, range(64)))
    # chunk 0 (32 steps) + chunks launched at 8, 32, 40, 48, 56 => 6 inferences
    assert fc.calls == 6


def test_served_action_is_single_step():
    fc = RampCompute(chunk_len=32, dim=3)
    wrapper = ChunkedPolicyWrapper(fc, action_steps=16)
    served = []
    need_obs = True
    async def _run():
        nonlocal need_obs
        for step in range(64):
            obs = {"__step__": float(step)} if need_obs else {}
            action, cot = await wrapper.get_action(obs)
            served.append(action["action"])
            need_obs = wrapper.need_obs
            await asyncio.sleep(0)
    asyncio.run(_run())
    for arr in served:
        assert arr.shape == (3,)
        assert arr.dtype == np.float32


def test_first_chunk_served_before_next_launch():
    # Only the first (synchronous) chunk is computed through step 15; the trigger
    # fires at step 16, which is when the background inference launches.
    fc = RampCompute(chunk_len=32)
    wrapper = ChunkedPolicyWrapper(fc, action_steps=16)
    asyncio.run(drive(wrapper, 16))
    assert fc.calls == 1
    assert wrapper.need_obs is True

    asyncio.run(drive(wrapper, 1))  # step 16 -> launches the background inference
    assert fc.calls == 2


def test_rtc_recomputes_every_step():
    # action_steps=1 is the RTC baseline: one inference per control step.
    fc = RampCompute(chunk_len=32)
    wrapper = ChunkedPolicyWrapper(fc, action_steps=1)
    served, _ = asyncio.run(drive(wrapper, 8))
    assert fc.calls == 8
    assert served == list(map(float, range(8)))


def test_sync_fallback_serves_fresh_window():
    # If the client fails to send a real obs at the trigger (step 16), no
    # background launch happens; at the swap the wrapper recomputes synchronously
    # from the *current* obs and serves its first action_steps steps. The action
    # stream must stay contiguous (the old code served the wrong future window).
    fc = RampCompute(chunk_len=32)
    wrapper = ChunkedPolicyWrapper(fc, action_steps=16)
    served, _ = asyncio.run(drive(wrapper, 64, empty_at={16}))

    assert served == list(map(float, range(64)))
    # chunk 0 (step 0) + synchronous recomputes at swaps (32, 48) => 3 inferences
    assert fc.calls == 3


def test_reset_clears_cache():
    fc = RampCompute(chunk_len=32)
    wrapper = ChunkedPolicyWrapper(fc, action_steps=16)
    asyncio.run(drive(wrapper, 16))
    assert fc.calls == 1

    wrapper.reset()
    assert wrapper.need_obs is True
    asyncio.run(drive(wrapper, 1))  # fresh chunk -> another synchronous inference
    assert fc.calls == 2
