# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The bulk TX filter must be bit-identical to ardopcf's per-sample recurrence.

`_TxFilter.render` runs the comb + resonator bank in bulk (shift-and-subtract
comb, one `scipy.signal.lfilter` per bin, vectorized output stage). `lfilter`'s
transposed direct-form-II associates the resonator sum differently from the
hand-rolled `zcomb + coef*z1 - r2*z2`, so the float64 states can disagree at the
sub-ULP level — but the output stage quantizes to int16, ~10⁹× coarser, so the
shipped waveform is bit-identical. These tests pin that: they re-implement the
original per-sample filter as an oracle and assert max|Δ| = 0, including the
truncating middle-bin filters and the drive≠100 integer floor-division path.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmodem.besra.frame import frame as F
from hfmodem.besra.phy import modulator


def _int16_wrap(v: int) -> int:
    return ((int(v) + 0x8000) & 0xFFFF) - 0x8000


def _ref_filter(src: np.ndarray, f: modulator._TxFilter) -> np.ndarray:
    """ardopcf's transmit filter run one output sample at a time — the exact
    recurrence `_TxFilter.render` replaced. Reuses the bin coefficients built by
    `_TxFilter.__init__`, so only the arithmetic form differs. The drive scaling
    is integer floor-division then an int16 wrap, matching `(short)(x*drive/100)`."""
    coef, tcoef, trunc = f._coef, f._tcoef, f._trunc
    rn, r2, N, drive = f._rn, f._r2, f._N, f._drive

    z1 = np.zeros(coef.size)
    z2 = np.zeros(coef.size)
    last120 = np.zeros(121, dtype=np.int64)
    zin1 = zin2 = 0.0
    get, put = 0, 120
    out: list[int] = []
    for n, sample in enumerate(src):
        s = _int16_wrap(int(sample) * drive // 100)
        zin = float(s) if n < N else s - rn * last120[get]
        get = (get + 1) % 121
        zcomb = zin - zin2 * r2
        zin2, zin1 = zin1, zin
        z0 = zcomb + coef * z1 - r2 * z2
        z2, z1 = z1, z0
        if n >= N // 2:
            shaped = np.where(trunc, np.trunc(z0), z0) * tcoef
            filt = float(np.sum(shaped)) * 0.00833333333
            filt = 32700.0 if filt > 32700 else (-32700.0 if filt < -32700 else filt)
            out.append(_int16_wrap(int(filt)))
        last120[put] = s
        put = (put + 1) % 121
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
