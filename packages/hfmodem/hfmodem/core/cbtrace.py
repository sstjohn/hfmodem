# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The audio callback's own account of itself.

`rates.lost_step` says a block was captured and never handed over. It cannot say
what the callback thread was doing when that happened, and a finished recording
cannot either -- the hole is the one thing not in it. So this records, from
inside the callback, the six facts the counter is a function of: when Python
reached the callback, when it left, what the converter's clock said, and what the
delivered count said.

PREALLOCATED, AND WRITTEN TO BY INDEX. A trace that appends to a list is a trace
that allocates inside the thing it is measuring, and an allocation is one of the
mechanisms on the list of suspects. Nothing here builds a tuple, grows a buffer
or takes a lock; a full ring stops recording rather than wrapping, because a
wrapped ring loses the beginning of the session, which is where the phase is.

`watch_gc` is on the same timeline and for the same reason: "it is probably the
collector" is a sentence that has to be settled by a measurement, and the
collector will say so itself for the asking.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

from . import rates


class CallbackTrace:
    """One row per callback, and one per garbage collection beside it."""

    def __init__(self, capacity: int = 1 << 19, *, blocksize: int = 128,
                 fs: int = rates.CARD_RATE_HZ) -> None:
        self.blocksize, self.fs = blocksize, fs
        self.n = 0
        self.dropped = 0          # callbacks past `capacity`, recorded nowhere
        self.t_in = np.zeros(capacity)     # monotonic, at callback entry
        self.t_out = np.zeros(capacity)    # ...and at the last line of the body
        self.adc = np.zeros(capacity)      # t.inputBufferAdcTime
        self.dac = np.zeros(capacity)      # t.outputBufferDacTime
        self.cur = np.zeros(capacity)      # t.currentTime
        self.n0 = np.zeros(capacity, np.int64)     # delivered before this block
        self.frames = np.zeros(capacity, np.int32)
        self.status = np.zeros(capacity, np.int32)
        self.gc_n = 0
        self.gc_t = np.zeros(4096)
        self.gc_gen = np.zeros(4096, np.int8)
        self.gc_phase = np.zeros(4096, np.int8)    # 0 start, 1 stop
        self._gc_hooked = False

    def stamp(self, t_in: float, t_out: float, t, frames: int, n0: int,
              status) -> None:
        i = self.n
        if i >= self.t_in.size:
            self.dropped += 1
            return
        self.t_in[i] = t_in
        self.t_out[i] = t_out
        self.adc[i] = t.inputBufferAdcTime
        self.dac[i] = t.outputBufferDacTime
        self.cur[i] = t.currentTime
        self.n0[i] = n0
        self.frames[i] = frames
        # `sounddevice.CallbackFlags` is not an int and will not become one;
        # its `_flags` is the PortAudio word, which is what a reader wants.
        self.status[i] = getattr(status, "_flags", 0)
        self.n = i + 1

    # -- the collector, on the same clock ----------------------------------

    def watch_gc(self) -> None:
        if not self._gc_hooked:
            gc.callbacks.append(self._gc)
            self._gc_hooked = True

    def unwatch_gc(self) -> None:
        if self._gc_hooked:
            gc.callbacks.remove(self._gc)
            self._gc_hooked = False

    def _gc(self, phase, info) -> None:
        i = self.gc_n
        if i >= self.gc_t.size:
            return
        self.gc_t[i] = time.monotonic()
        self.gc_gen[i] = info["generation"]
        self.gc_phase[i] = 0 if phase == "start" else 1
        self.gc_n = i + 1

    # -- reading it back ---------------------------------------------------

    @property
    def ahead(self) -> np.ndarray:
        """The converter's instant for each block minus where the delivered count
        puts it -- the series `rates.lost_step` watches."""
        i = self.n
        return self.adc[:i] - self.n0[:i] / self.fs

    def losses(self) -> np.ndarray:
        """Indices of the callbacks `lost_step` charges a loss to."""
        a = self.ahead
        step = np.diff(a, prepend=a[:1])
        step[0] = 0.0
        return np.nonzero(step > self.blocksize / 2 / self.fs)[0]

    def lost_samples(self) -> int:
        a = self.ahead
        step = np.diff(a, prepend=a[:1])
        step[0] = 0.0
        return int(np.round(step[step > self.blocksize / 2 / self.fs] * self.fs).sum())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        i, g = self.n, self.gc_n
        np.savez_compressed(
            path, blocksize=self.blocksize, fs=self.fs, dropped=self.dropped,
            t_in=self.t_in[:i], t_out=self.t_out[:i], adc=self.adc[:i],
            dac=self.dac[:i], cur=self.cur[:i], n0=self.n0[:i],
            frames=self.frames[:i], status=self.status[:i],
            gc_t=self.gc_t[:g], gc_gen=self.gc_gen[:g], gc_phase=self.gc_phase[:g])
        return path

    def report(self) -> str:
        i = self.n
        if i < 3:
            return f"{i} callbacks recorded -- nothing to say about them"
        t_in, t_out = self.t_in[:i], self.t_out[:i]
        body = (t_out - t_in) * 1e3
        gap = np.diff(t_in) * 1e3
        adc_step = np.diff(self.adc[:i]) * self.fs
        lost = self.losses()
        span = t_in[-1] - t_in[0]
        out = [
            f"{i} callbacks over {span:.1f} s ({self.dropped} past the ring), "
            f"{int(self.frames[:i].sum())} frames delivered",
            f"body      ms: median {np.median(body):.3f}  p99 {np.percentile(body, 99):.3f}"
            f"  max {body.max():.3f}",
            f"entry gap ms: median {np.median(gap):.3f}  p99 {np.percentile(gap, 99):.3f}"
            f"  max {gap.max():.3f}",
            f"adc step samples: median {np.median(adc_step):.1f}  max {adc_step.max():.1f}",
            f"lost: {self.lost_samples()} samples in {lost.size} steps",
        ]
        if lost.size:
            out.append("  the callbacks the loss is charged to:")
            for k in lost[:12]:
                since = (t_in[k] - t_in[k - 1]) * 1e3 if k else 0.0
                before = body[k - 1] if k else 0.0
                out.append(
                    f"    n0={self.n0[k]:<9d} +{adc_step[k - 1]:6.0f} samples of adc"
                    f"  {since:8.2f} ms since the previous entry"
                    f"  (that callback's body {before:.3f} ms, this one's "
                    f"{body[k]:.3f} ms, status {self.status[k]})")
            if lost.size > 12:
                out.append(f"    ... and {lost.size - 12} more")
            out.append(self._gc_verdict(lost))
        return "\n".join(out)

    def _gc_verdict(self, lost: np.ndarray) -> str:
        """Whether a collection was running when the capture went missing.

        The hole is between two callbacks, so the question is whether a
        collection overlapped that interval -- not whether one happened nearby.
        """
        g = self.gc_n
        if not g:
            return "  gc: not watched"
        starts = self.gc_t[:g][self.gc_phase[:g] == 0]
        gens = self.gc_gen[:g][self.gc_phase[:g] == 0]
        t_in = self.t_in[:self.n]
        hit = 0
        for k in lost:
            if k and np.any((starts >= t_in[k - 1]) & (starts <= t_in[k])):
                hit += 1
        by_gen = "  ".join(f"gen{n}:{int((gens == n).sum())}" for n in (0, 1, 2))
        return (f"  gc: {int((self.gc_phase[:g] == 0).sum())} collections "
                f"({by_gen}); {hit} of {lost.size} loss intervals had one "
                f"running inside them")
