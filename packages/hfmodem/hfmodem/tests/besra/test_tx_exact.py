# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The bulk TX filter must be bit-identical to the filter stepped one sample at a time.

`_TxFilter.render` runs the comb + resonator bank in bulk (shift-and-subtract
comb, one `scipy.signal.lfilter` per bin, vectorized output stage). `lfilter`'s
transposed direct-form-II associates the resonator sum differently from stepping
`y = x + c·y1 - R²·y2` by hand, so the float64 states can disagree at the
sub-ULP level — but the output stage quantizes to int16, ~10⁹× coarser, so the
shipped waveform is bit-identical. These tests pin that against a per-sample
oracle and assert max|Δ| = 0, including the whole-number inner bins and the
drive≠100 integer floor-division path.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmodem.besra.frame import frame as F
from hfmodem.besra.phy import modulator


def _int16_wrap(v: int) -> int:
    return ((int(v) + 0x8000) & 0xFFFF) - 0x8000


def _ref_filter(src: np.ndarray, f: modulator._TxFilter) -> np.ndarray:
    """The transmit filter stepped one output sample at a time, from its
    definition: drive, comb, notch, then every resonator by hand. It takes the bin
    constants from ``f``, so only the arithmetic form differs from the bulk path.
    Drive is integer floor division wrapped into int16."""
    x = [_int16_wrap(int(v) * f._drive // 100) for v in src]
    y1 = np.zeros(f._coef.size)
    y2 = np.zeros(f._coef.size)
    back1 = back2 = 0.0                       # the comb's last two outputs
    out: list[int] = []
    for n, v in enumerate(x):
        combed = float(v) - (f._rn * x[n - f._N] if n >= f._N else 0.0)
        y = (combed - f._r2 * back2) + f._coef * y1 - f._r2 * y2
        back2, back1 = back1, combed
        y2, y1 = y1, y
        if n >= f._N // 2:
            level = float(np.sum(np.where(f._whole, np.trunc(y), y) * f._weight)) * 0.00833333333
            out.append(_int16_wrap(int(min(32700.0, max(-32700.0, level)))))
    return np.array(out, dtype="<i2")


_WIDTHS = [200, 500, 1000, 2000]  # 500/1000/2000 carry the truncating middle bins


@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("drive", [100, 75, 50])
def test_bulk_filter_matches_per_sample(width: int, drive: int) -> None:
    rng = np.random.default_rng(0xA5 ^ width ^ (drive << 8))
    src = rng.integers(-32767, 32768, size=4000)
    # A block of negatives exercises the floor-vs-trunc divergence of the drive
    # scaling that a float `x*drive/100` would get wrong.
    src[:50] = rng.integers(-32767, -1, size=50)

    bulk = modulator._TxFilter(width, drive=drive).render(src)
    ref = _ref_filter(src, modulator._TxFilter(width, drive=drive))

    assert bulk.dtype == ref.dtype == np.dtype("<i2")
    assert len(bulk) == len(ref)
    diff = int(np.abs(bulk.astype(np.int64) - ref.astype(np.int64)).max())
    assert diff == 0, f"width={width} drive={drive}: max|Δ| = {diff}"


def _meta(fd: F.FrameDef) -> dict:
    n = fd.name
    if n.startswith("ConAck"):
        return {"timing": 120}
    if n == "PingAck":
        return {"sn": 15, "quality": 50}
    if n == "IDFrame":
        return {"caller": "W9SSJ", "grid": "EN63"}
    if n.startswith("ConReq") or n == "Ping":
        return {"caller": "W9SSJ", "target": "K1ABC"}
    return {}


# One frame per distinct filter width, so the end-to-end path exercises every
# resonator geometry including the truncating middle bins.
_FRAME_PER_WIDTH: dict[int, int] = {}
for _ft, _fd in F.FRAMES.items():
    _w = (modulator._FILTER_WIDTH_FOR_FSK[_fd.baud] if _fd.mod is F.Mod.FSK4
          else modulator._FILTER_WIDTH[_fd.carriers])
    _FRAME_PER_WIDTH.setdefault(_w, _ft)
_REP_FRAMES = sorted(_FRAME_PER_WIDTH.values())


@pytest.mark.parametrize("frame_type", _REP_FRAMES)
@pytest.mark.parametrize("payload", [b"", b"besra42", bytes(range(24))])
@pytest.mark.parametrize("drive", [100, 75])
def test_render_frame_matches_per_sample_oracle(frame_type, payload, drive):
    """render_frame's bulk filter reproduces the per-sample oracle bit-for-bit,
    end to end, over frame types / payloads / a session and a drive≠100 case."""
    fd = F.FRAMES[frame_type]
    kw = dict(session_id=0x5A, leader_ms=40, trailer_ms=10, drive=drive, **_meta(fd))

    real = modulator.render_frame(frame_type, payload=payload, **kw)

    orig = modulator._TxFilter

    class _OracleFilter(orig):
        def render(self, src):
            return _ref_filter(np.asarray(src), self)

    modulator._TxFilter = _OracleFilter
    try:
        oracle = modulator.render_frame(frame_type, payload=payload, **kw)
    finally:
        modulator._TxFilter = orig

    assert len(real) == len(oracle)
    diff = int(np.abs(real.astype(np.int64) - oracle.astype(np.int64)).max())
    assert diff == 0, f"{fd.name} drive={drive}: max|Δ| = {diff}"
