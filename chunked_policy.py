"""Chunked look-ahead policy wrapper (Plan B receding horizon).

Extracted from ``serve_policy.py`` so it can be imported and unit-tested without
torch / g05 / a GPU. It owns only the chunk-caching + async-look-ahead *state
machine*; the actual model inference is injected as an async ``compute`` callable.

    wrapper = ChunkedPolicyWrapper(compute=my_async_compute, action_steps=16)
    need_obs = wrapper.need_obs
    action, cot = await wrapper.get_action(raw_obs_or_empty)

``compute(raw_obs) -> (chunk, cot_text)`` must return a chunk dict mapping each
action part to ``ndarray[chunk_len, dim]`` (the leading dim is the chunk length).
"""

from __future__ import annotations

import asyncio

import numpy as np


class ChunkedPolicyWrapper:
    """Wraps model inference with chunk caching and step-by-step serving.

    One instance per client connection.

    Plan B (async look-ahead, receding horizon): the *first* chunk is served in
    full (``chunk_len`` steps = the model's action horizon); each subsequent
    chunk is served only for a fresh ``action_steps``-step window, the rest being
    steps already executed by the previous chunk (the overlap). The next chunk's
    inference is launched in the background from a fresh observation collected
    every ``action_steps`` steps, so its latency is hidden behind the current
    chunk's tail and the swap at the window boundary does not stall. Blocking at
    the boundary is the timeout fallback when inference overruns the tail.

    Args:
        compute: async callable ``raw_obs -> (chunk_dict, cot_text)`` that runs
            one model inference and returns a chunk of future actions.
        action_steps: control period in steps — a fresh observation is collected
            and the next chunk's inference is launched every ``action_steps``
            steps. The first chunk serves ``chunk_len`` steps; subsequent chunks
            serve only ``action_steps`` fresh steps.
            1 = RTC (recompute every call), >1 = chunk look-ahead mode. Look-ahead
            requires ``chunk_len >= 2 * action_steps``; otherwise it degrades to
            synchronous full-chunk recompute.
    """

    def __init__(self, compute, *, action_steps: int = 16):
        self._compute = compute
        self.action_steps = action_steps
        self._cached_chunk = None  # {part: ndarray[chunk_len, dim]}
        self._chunk_len = 0        # array leading dim (the model's action horizon)
        self._chunk_step = 0       # next array index to serve
        self._serve_start = 0      # array index where the current serve window began
        self._serve_len = 0        # number of steps in the current serve window
        self._next_task = None     # asyncio.Task computing the next chunk, or None
        self._next_start = 0       # array index where the pending chunk's window begins
        self._cot_text = None      # str or None, from the most recent chunk

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_len_of(chunk: dict) -> int:
        """Number of steps in a chunk dict (leading dim of the first array part)."""
        for arr in chunk.values():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                return int(arr.shape[0])
        return 0

    async def _compute_chunk(self, raw_obs: dict) -> tuple[dict, str | None]:
        """Run one injected inference and return ``(chunk_dict, cot_text)``."""
        return await self._compute(raw_obs)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @property
    def need_obs(self) -> bool:
        """Whether the next call to get_action needs a real observation.

        True when no chunk exists yet, when the current serve window is exhausted
        (the next call swaps in the pending chunk *and* needs a fresh observation
        to launch the following chunk), or when the observation-collection trigger
        has been reached and the next chunk's inference has not been launched yet.
        False while serving from cache or while a pending inference is running
        (the client can then send an empty observation).
        """
        if self._cached_chunk is None:
            return True
        if self._chunk_step >= self._serve_start + self._serve_len:
            return True
        if self._next_task is not None:
            return False
        return self._chunk_step >= self.action_steps

    async def get_action(self, raw_obs: dict) -> tuple[dict, str | None]:
        """Return single-step action ``{part: ndarray[dim]}``."""
        rtc = self.action_steps == 1

        if self._cached_chunk is None:
            # First chunk: nothing to serve while we wait, so compute synchronously
            # and serve its full horizon. Serving the full horizon gives the first
            # background inference its maximum lead time (chunk_len - action_steps).
            self._cached_chunk, self._cot_text = await self._compute_chunk(raw_obs)
            self._chunk_len = self._chunk_len_of(self._cached_chunk)
            self._chunk_step = 0
            self._serve_start = 0
            self._serve_len = self._chunk_len
        elif rtc:
            # RTC: recompute synchronously on every call (original behavior).
            if self._chunk_step >= self._serve_start + self._serve_len:
                self._cached_chunk, self._cot_text = await self._compute_chunk(raw_obs)
                self._chunk_len = self._chunk_len_of(self._cached_chunk)
                self._chunk_step = 0
                self._serve_start = 0
                self._serve_len = self._chunk_len
        else:
            # Chunk look-ahead (receding horizon). Requires an overlap window and a
            # fresh window to both fit in the chunk horizon: chunk_len >= 2*action_steps.
            lookahead = self._chunk_len >= 2 * self.action_steps
            if self._chunk_step >= self._serve_start + self._serve_len:
                # Serve window exhausted: swap in the pending async chunk, or fall
                # back to a synchronous recompute (inference overran the tail, or the
                # background launch never fired).
                if lookahead and self._next_task is not None:
                    task = self._next_task
                    self._next_task = None
                    self._cached_chunk, self._cot_text = await task
                    self._chunk_len = self._chunk_len_of(self._cached_chunk)
                    # Resume the pending chunk where its observation lead ends.
                    self._chunk_step = self._next_start
                    self._serve_start = self._next_start
                    self._serve_len = self.action_steps
                else:
                    # Fresh obs: compute now, serve its first action_steps steps.
                    self._cached_chunk, self._cot_text = await self._compute_chunk(raw_obs)
                    self._chunk_len = self._chunk_len_of(self._cached_chunk)
                    self._chunk_step = 0
                    self._serve_start = 0
                    self._serve_len = self.action_steps if lookahead else self._chunk_len
            # Launch the next chunk's inference from a fresh observation once the
            # trigger is reached. This fires mid-window for the first chunk, and
            # immediately after a swap for subsequent chunks.
            if (
                lookahead
                and self._chunk_step >= self.action_steps
                and self._next_task is None
                and raw_obs
            ):
                # Number of steps until the current window ends == how far into the
                # next chunk the swap will land (its observation lead).
                self._next_start = self._serve_start + self._serve_len - self._chunk_step
                self._next_task = asyncio.create_task(self._compute_chunk(raw_obs))

        single_step = {}
        for part, arr in self._cached_chunk.items():
            if arr.ndim >= 1 and self._chunk_step < arr.shape[0]:
                single_step[part] = arr[self._chunk_step]
            else:
                single_step[part] = arr[0] if arr.ndim >= 1 else arr
        self._chunk_step += 1
        return single_step, self._cot_text

    def reset(self):
        """Invalidate cache and cancel any pending inference (episode boundary)."""
        self._cancel_pending()
        self._cached_chunk = None
        self._chunk_step = 0
        self._serve_start = 0
        self._serve_len = 0
        self._cot_text = None

    def close(self):
        """Cancel any pending inference (connection teardown)."""
        self._cancel_pending()

    def _cancel_pending(self):
        if self._next_task is not None and not self._next_task.done():
            self._next_task.cancel()
        self._next_task = None
