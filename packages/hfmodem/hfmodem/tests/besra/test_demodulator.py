# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate besra's receiver end to end.

Two arbiters, one decode path:

- **Ground truth** — the 59 ``txframe_*.wav`` ardopcf renders. Padded with leading
  and trailing silence exactly as the reference RX test does, each must decode to
  the manifest's frame type and its exact payload/metadata (the data bytes; ConAck
  timing; PingAck S:N + quality; ConReq/Ping/ID callsigns + grid).
- **Loopback** — besra's own modulator feeds its own demodulator for every mode,
  closing the TX↔RX round trip on the byte stream.

The SDFT primitive in :mod:`besra.dsp.detect` is checked against a plain Goertzel
so the off-grid tone detector is proven, not just used.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmodem.besra.dsp import detect
from hfmodem.besra.phy import demodulator
from hfmodem.besra.phy import modulator
from hfmodem.besra.frame import frame as F
from . import groundtruth as gt

_LEAD_PAD = 2400
_TAIL_PAD = 4800


def _pad(au: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(_LEAD_PAD, dtype=np.int16), au,
                           np.zeros(_TAIL_PAD, dtype=np.int16)])


def _check(frame: demodulator.DecodedFrame, entry: dict, name: str) -> None:
    assert frame is not None, f"{name}: nothing decoded"
    assert frame.type == entry["type"], (
        f"{name}: type 0x{frame.type:02X} != 0x{entry['type']:02X}")
    assert frame.ok, f"{name}: decoded but flagged not-ok"

    meta = entry.get("meta")
    if "payload" in entry:
        assert frame.payload == entry["payload"], f"{name}: payload mismatch"
    elif meta is None:
        assert frame.payload == b""
    elif "timing" in meta:
        assert frame.conack_timing_ms == int(meta["timing"]), f"{name}: ConAck timing"
    elif "sn" in meta:
        assert frame.pingack_sn_db == int(meta["sn"]), f"{name}: PingAck S:N"
        assert frame.pingack_quality == int(meta["quality"]), f"{name}: PingAck quality"
    elif "grid" in meta:
        assert frame.caller == meta["caller"], f"{name}: ID caller"
        assert frame.grid == meta["grid"], f"{name}: ID grid"
    else:
        assert frame.caller == meta["caller"], f"{name}: caller"
        assert frame.target == meta["target"], f"{name}: target"


# --------------------------------------------------------------------------- #
# Ground truth: decode every ardopcf-rendered fixture.
# --------------------------------------------------------------------------- #

@gt.requires_reference
@pytest.mark.parametrize("name", gt.wav_params())
def test_decode_fixture(name: str) -> None:
    entry = gt.txframe_manifest()[name]
    frames = demodulator.decode(_pad(gt.load_wav(name)))
    assert len(frames) == 1, f"{name}: expected 1 frame, got {len(frames)}"
    _check(frames[0], entry, name)


@gt.requires_reference
def test_all_fixtures_exact() -> None:
    """A single roll-up so the whole ground-truth set passing is one green line."""
    manifest = gt.txframe_manifest()
    for name in gt.wav_names():
        frames = demodulator.decode(_pad(gt.load_wav(name)))
        assert len(frames) == 1, f"{name}: {len(frames)} frames"
        _check(frames[0], manifest[name], name)


@gt.requires_reference
def test_callsign_loop_closed() -> None:
    """ConReq and ID recover the transmitted callsigns/grid byte for byte."""
    con = demodulator.decode(_pad(gt.load_wav("txframe_ConReq2000M.wav")))[0]
    assert (con.caller, con.target) == ("M7TFF", "GB7RDG-15")
    idf = demodulator.decode(_pad(gt.load_wav("txframe_IDFrame.wav")))[0]
    assert (idf.caller, idf.grid) == ("M7TFF-3", "IO81VK")


# --------------------------------------------------------------------------- #
# Loopback: besra's modulator into besra's demodulator, every mode.
# --------------------------------------------------------------------------- #

def _loopback(entry: dict) -> demodulator.DecodedFrame:
    meta = entry.get("meta", {})
    au = modulator.render_frame(
        entry["type"], payload=entry.get("payload", b""),
        session_id=entry["session_id"], **meta)
    frames = demodulator.decode(_pad(au))
    assert len(frames) == 1
    return frames[0]


@pytest.mark.parametrize("name", gt.wav_params())
def test_loopback_every_mode(name: str) -> None:
    """Round-trip besra TX → besra RX for each manifest frame (covers all modes)."""
    entry = gt.txframe_manifest()[name]
    _check(_loopback(entry), entry, name)


def test_loopback_synthetic_data() -> None:
    """A payload not drawn from the fixtures, round-tripped on a spread of modes —
    proves the loop independently of the ground-truth vectors."""
    rng = np.random.default_rng(1)
    for ftype in (0x48, 0x4A, 0x40, 0x44, 0x46, 0x50, 0x54, 0x62, 0x64, 0x74, 0x7A):
        fd = F.FRAMES[ftype]
        payload = bytes(rng.integers(0, 256, fd.net_payload, dtype=np.uint8))
        au = modulator.render_frame(ftype, payload=payload, session_id=0xFF)
        frames = demodulator.decode(_pad(au))
        assert len(frames) == 1, f"0x{ftype:02X}: {len(frames)} frames"
        assert frames[0].type == ftype and frames[0].ok
        assert frames[0].payload == payload, f"0x{ftype:02X} ({fd.name}) payload"


# --------------------------------------------------------------------------- #
# The off-grid SDFT primitive.
# --------------------------------------------------------------------------- #

def test_sdft_lands_on_offgrid_tones_like_goertzel() -> None:
    """The ½N-offset streaming SDFT matches a single-block Goertzel magnitude on the
    off-grid FSK tones a plain 50 Hz DFT grid cannot reach."""
    rng = np.random.default_rng(7)
    for n, tones in ((240, detect.FSK_TONES[50]), (120, detect.FSK_TONES[100])):
        seg = rng.normal(0, 3000, n)
        vec = detect.SlidingDFT(n, tones).block(seg)
        for i, f in enumerate(tones):
            ref = abs(detect.goertzel(seg, 0, n, f))
            assert abs(abs(vec[i]) - ref) < 1e-6 * (ref + 1)


# --------------------------------------------------------------------------- #
# The session id, which nothing in the header protects.
#
# Both header parity symbols are computed from the raw type, so symbols 5-8 ride
# unchecked and one slipped tone rewrites the session id. On
# `logs/onair/20260829T015913Z-besra-3590508.wav` at t=103.075 s a
# `4PSK.200.100S.E` decoded ok=True at q=60 reading `sess=0xa4` against the live
# session's `0xac` — one symbol out, symbol 7, read 1 where 3 was second-strongest
# by 6.6 dB. `ArqSession.on_receive` discarded it, KN4LQN retransmitted, and the
# repeat 3.0 s later decoded at 0xac carrying byte-identical payload. These build
# that geometry: the same two ids, the same differing symbol, the same runner-up.
# --------------------------------------------------------------------------- #

_OURS, _THEIRS = 0xAC, 0xA4
_SLIPPED = 7                      # the one header symbol the two ids differ in


def _headered(ftype: int, session: int, payload: bytes = b"") -> np.ndarray:
    return _pad(modulator.render_frame(ftype, payload=payload, session_id=session))


def _second_strongest(au: np.ndarray, header: int, symbol: int,
                      tone: int, amp: float = 0.5) -> np.ndarray:
    """``au`` with ``tone`` laid over one header symbol hard enough to come second
    at it and no harder — the channel putting a slipped tone within reach."""
    lo = header + symbol * demodulator._SYM_50
    t = np.arange(demodulator._SYM_50) / demodulator.SAMPLE_RATE
    out = au.astype(np.float64)
    out[lo:lo + demodulator._SYM_50] += amp * 8000 * np.cos(2 * np.pi * tone * t)
    return out


def _wanted_tone(ftype: int, symbol: int) -> int:
    return detect.FSK_TONES[50][F.header_symbols(ftype, _OURS)[symbol]]


def test_a_slipped_session_tone_is_put_back_when_the_body_vouches_for_it() -> None:
    payload = bytes(range(16))
    au = _headered(0x42, _THEIRS, payload)
    (plain,) = demodulator.Demodulator().decode(au)
    assert plain.session_id == _THEIRS and plain.ok

    au = _second_strongest(au, plain.offset, _SLIPPED, _wanted_tone(0x42, _SLIPPED))
    (got,) = demodulator.Demodulator(expect_session=lambda: _OURS).decode(au)
    assert got.session_id == _OURS
    assert got.type == 0x42 and got.ok and got.payload == payload


def test_a_crisp_reading_of_another_station_stays_another_station() -> None:
    """The runner-up is the whole test. A header whose session symbols read cleanly
    is a station saying what it meant, and it is left saying it."""
    au = _headered(0x42, _THEIRS, bytes(range(16)))
    (got,) = demodulator.Demodulator(expect_session=lambda: _OURS).decode(au)
    assert got.session_id == _THEIRS and got.ok


def test_two_slipped_tones_are_not_one_and_the_frame_stays_a_stranger() -> None:
    """One tone is a read error; two is a guess at eight bits. `0xed` sits two
    dibits from `0xac`, and both are put within reach at once."""
    far = 0xED
    payload = bytes(range(16))
    au = _headered(0x42, far, payload)
    (plain,) = demodulator.Demodulator().decode(au)
    ours, theirs = F.header_symbols(0x42, _OURS), F.header_symbols(0x42, far)
    for sym in (i for i in range(5, 9) if ours[i] != theirs[i]):
        au = _second_strongest(au, plain.offset, sym, _wanted_tone(0x42, sym))
    (got,) = demodulator.Demodulator(expect_session=lambda: _OURS).decode(au)
    assert got.session_id == far and got.ok


def test_nothing_is_repaired_when_this_station_is_party_to_no_session() -> None:
    """A listening receiver has no id to read a frame back onto, and reports what
    the tones spelled."""
    au = _headered(0x42, _THEIRS, bytes(range(16)))
    (plain,) = demodulator.Demodulator().decode(au)
    au = _second_strongest(au, plain.offset, _SLIPPED, _wanted_tone(0x42, _SLIPPED))
    (got,) = demodulator.Demodulator().decode(au)
    assert got.session_id == _THEIRS


def test_a_bare_controls_session_id_is_never_manufactured() -> None:
    """A BREAK has no body, so `ok` is the crispness floor and a leader — and its
    session id is the eight bits `_crisp_floor`, `_addressed` and the `_scan_header`
    fallback all lean on for want of anything else. The reading would fire: it is
    the same one-symbol, runner-up geometry the data frame above is repaired on.
    It is not spent here, and the strict unsolicited floor refuses the frame
    outright rather than admitting it under our own id."""
    au = _headered(0x23, _THEIRS)
    (plain,) = demodulator.Demodulator().decode(au)
    assert plain.name == "BREAK" and plain.session_id == _THEIRS

    sym = next(i for i in range(5, 9)
               if F.header_symbols(0x23, _OURS)[i] != F.header_symbols(0x23, _THEIRS)[i])
    au = _second_strongest(au, plain.offset, sym, _wanted_tone(0x23, sym))
    tones = tuple(float(f) for f in detect.FSK_TONES[50])
    lo = plain.offset + 5 * demodulator._SYM_50
    col = detect.tone_mag_series(au[lo:lo + 4 * demodulator._SYM_50],
                                demodulator._SYM_50,
                                tones)[:, ::demodulator._SYM_50][:, :4]
    assert demodulator._slipped_session(col, 0x23, _THEIRS, _OURS) == _OURS
    assert all(f.session_id != _OURS
               for f in demodulator.Demodulator(expect_session=lambda: _OURS).decode(au))
