# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Per-frame-local acquisition: the frames the old whole-buffer scan dropped.

Every step of acquisition now works on a bounded region around the candidate
frame — burst onset, carrier-offset window, leader threshold, and the body
transform — instead of on global whole-capture quantities. These are the three
faults that fixed, each a frame the receiver used to lose:

- a strong on-tune frame killed because a weaker earlier frame skewed the global
  carrier-offset estimate into a bogus buffer-wide de-rotation;
- a weaker earlier frame skipped because the leader threshold was a fraction of a
  later, stronger frame's peak;
- the second of two close frames lost because the cursor chased trailing energy
  past its leader instead of stepping by the first frame's known length.

Each case fails on the pre-local receiver and passes now. The clean single-frame
on-tune path (the interop burst) is covered byte-for-byte by
``test_demodulator``; here the point is the multi-frame and mistuned captures the
fixtures cannot express.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmodem.besra.phy import demodulator as D
from hfmodem.besra.phy import modulator as M
from hfmodem.besra.frame import frame as F

SR = 12000
_LEAD = np.zeros(2400, dtype=np.int16)
_TAIL = np.zeros(4800, dtype=np.int16)


def _pl(ftype: int) -> bytes:
    return bytes((i * 7 + 3) & 0xFF for i in range(F.FRAMES[ftype].net_payload))


def _framed(*chunks: np.ndarray) -> np.ndarray:
    return np.concatenate([_LEAD, *chunks, _TAIL])


def _scaled(ftype: int, amp: float, session: int) -> np.ndarray:
    return M.render_frame(ftype, payload=_pl(ftype), session_id=session).astype(np.float64) * amp


def _analytic(x: np.ndarray) -> np.ndarray:
    X = np.fft.fft(x)
    n = x.size
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def _tune(au: np.ndarray, hz: float) -> np.ndarray:
    """Single-sideband carrier offset — the whole passband slides by ``hz``."""
    n = np.arange(au.size)
    return np.real(_analytic(au.astype(float)) * np.exp(2j * np.pi * hz * n / SR))


# --------------------------------------------------------------------------- #
# Bug #1 — a weak leading frame must not skew a strong following frame off-tune.
# --------------------------------------------------------------------------- #

def test_weak_then_strong_pair_both_decode():
    """A weak frame (~6 dB down) ahead of a strong on-tune one: the old global
    carrier-offset estimate anchored on the weak frame's body, returned a bogus
    offset, de-rotated the whole buffer and lost both. Local estimation keeps each
    frame's offset to itself, so both decode."""
    gap = np.zeros(3000, dtype=np.float64)
    weak = _scaled(0x40, 0.5, 0x5A)
    strong = _scaled(0x44, 1.0, 0x33)
    frames = D.decode(_framed(weak, gap, strong))

    assert [f.type for f in frames] == [0x40, 0x44]
    assert all(f.ok for f in frames)
    assert frames[0].payload == _pl(0x40)
    assert frames[1].payload == _pl(0x44)


def test_much_weaker_leading_frame_still_found():
    """The leader threshold is local now, so a frame ~10 dB below a later one is
    still acquired — a global-peak fraction skipped it entirely."""
    gap = np.zeros(3000, dtype=np.float64)
    weak = _scaled(0x40, 0.3, 0x5A)
    strong = _scaled(0x44, 1.0, 0x33)
    frames = D.decode(_framed(weak, gap, strong))

    assert [f.type for f in frames] == [0x40, 0x44]
    assert frames[0].payload == _pl(0x40) and frames[1].payload == _pl(0x44)


# --------------------------------------------------------------------------- #
# Bug #3 — close-following frame lost to a trailing-energy overshoot.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gap_ms", [25, 100, 200])
def test_close_equal_pair_both_decode(gap_ms):
    """Two equal frames a short gap apart. The old cursor walked past the first
    frame's trailer to re-acquire and, unable to tell trailer from the next
    leader, overshot the second frame's leader. Stepping by the first frame's
    known length lands cleanly in the gap, so the second decodes too."""
    gap = np.zeros(int(SR * gap_ms / 1000), dtype=np.float64)
    first = _scaled(0x40, 1.0, 0x5A)
    second = _scaled(0x40, 1.0, 0x33)
    frames = D.decode(_framed(first, gap, second))

    assert [f.type for f in frames] == [0x40, 0x40]
    assert [f.session_id for f in frames] == [0x5A, 0x33]
    assert all(f.payload == _pl(0x40) for f in frames)
    assert frames[0].offset < frames[1].offset


# --------------------------------------------------------------------------- #
# Bug #1 / CFO — a mistuned multi-frame capture, each frame de-rotated on itself.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("hz", [150, -150])
def test_offset_multiframe_decodes(hz):
    """A ±150 Hz-mistuned capture holding three frames, the first ~7 dB down. The
    old estimator took one offset from a global energy onset that the weak frame
    pushed into PSK data — a bogus offset that de-rotated the whole buffer and lost
    every frame. Estimating each frame's offset on its own leader recovers all
    three from the mistuned capture."""
    gap = np.zeros(3000, dtype=np.float64)
    a = _scaled(0x40, 0.45, 0x5A)
    b = _scaled(0x44, 1.0, 0x33)
    c = M.render_frame(0x34, caller="W9SSJ", target="K7ABC",
                       session_id=0xFF).astype(np.float64)
    frames = D.decode(_tune(_framed(a, gap, b, gap, c), hz))

    assert [f.type for f in frames] == [0x40, 0x44, 0x34]
    assert all(f.ok for f in frames)
    assert frames[0].payload == _pl(0x40)
    assert frames[1].payload == _pl(0x44)
    assert (frames[2].caller, frames[2].target) == ("W9SSJ", "K7ABC")
