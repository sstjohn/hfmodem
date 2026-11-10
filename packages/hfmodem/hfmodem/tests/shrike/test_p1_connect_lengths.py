# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1, pactor2, rxfront

TREE = Path(__file__).resolve().parents[5]
CALLS = ("A", "A1", "A1B", "W1AW", "N3HYM", "WS8EOC", "N3HYM-1",
         "N3HYM-10", "AA1AAA-1", "WS8EOC-10", "3DA0XYZ-15", "N3HYM-15", "DL6MAA-1")
SESSIONS = (
    ("onair-1002-2138", "N3HYM-10", 2749, 9,
     "pactor-current-n3hym-10-40-force-20261003T023836Z.log"),
    ("onair-1002-2037", "WS8EOC", 2745, 14,
     "pactor-p4-bench-ws8eoc-80-force-20261003T013745Z-p4-mail.log"),
)


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("invert", (False, True))
@pytest.mark.parametrize("variant", ("normal", "longpath"))
def test_connect_lengths(call, invert, variant):
    if len(call) > 8:
        with pytest.raises(ValueError, match="1..8 characters"):
            pactor1.connect_signal(call, variant=variant, invert=invert)
        return
    audio = pactor1.connect_signal(call, variant=variant, invert=invert)
    got = p1rx.decode_connect(audio)
    assert got is not None
    assert (got.callsign, got.variant, got.inverted) == (call, variant.title(), invert)
    assert p1rx.decode_connect(audio, tones=p1rx.tone_series(audio)) == got


@pytest.mark.parametrize("call", ("A1B", "K1AB", "AA1AB", "AA1ABC", "AA1ABCD",
                                  "AA1ABCDE", "A-1", "A1-1", "A1B-1",
                                  "K1AB-1", "AA1AB-1", "AA1AAA-1", "N3HYM-10"))
@pytest.mark.parametrize("invert", (False, True))
@pytest.mark.parametrize("variant", ("normal", "longpath"))
@pytest.mark.parametrize("offset,phase", ((0, 1.22), (137, 0.0), (479, 2.7)))
def test_connect_lock_plateaus(call, invert, variant, offset, phase):
    audio = pactor1.connect_signal(call, variant=variant, invert=invert, phase0=phase)
    audio = np.pad(audio, (offset, 0))
    got = p1rx.decode_connect(audio)
    assert got is not None
    assert (got.callsign, got.variant, got.inverted) == (call, variant.title(), invert)


def test_full_address_bytes():
    assert pactor1.connect_frame_bytes("N3HYM-10") == (
        b"\x55N3HYM-10", b"N3HYM-")
    assert pactor1.connect_frame_bytes("N3HYM-1") == (
        b"\x55N3HYM-1\x0f", b"N3HYM-")


def _frame(addr, red, invert=False, red_sps=None):
    freqs = np.concatenate((
        pactor1._fsk_freqs(pactor1._bits_lsb_first(addr), p1rx.FS // 100, invert),
        pactor1._fsk_freqs(pactor1._bits_lsb_first(red),
                          red_sps or p1rx.FS // 200, invert)))
    burst = 0.11 * np.cos(2 * np.pi * np.cumsum(freqs) / p1rx.FS)
    pad = np.zeros(p1rx.FS // 10)
    return np.concatenate((pad, burst, pad))


@pytest.mark.parametrize("invert", (False, True))
@pytest.mark.parametrize("variant", ("normal", "longpath"))
@pytest.mark.parametrize("byte", range(6))
def test_full_address_requires_every_redundancy_byte(invert, variant, byte):
    addr, red = pactor1.connect_frame_bytes("N3HYM-10", variant)
    bad = bytearray(red)
    bad[byte] ^= 1
    assert p1rx.decode_connect(_frame(addr, bad, invert)) is None


@pytest.mark.parametrize("invert", (False, True))
@pytest.mark.parametrize("byte", range(9))
def test_full_address_requires_sync_and_valid_characters(invert, byte):
    addr = bytearray(b"\x55N3HYM-10")
    addr[byte] = 0
    assert p1rx.decode_connect(_frame(addr, b"N3HYM-", invert)) is None


@pytest.mark.parametrize("invert", (False, True))
def test_full_address_last_character_high_bit_is_read(invert):
    assert p1rx.decode_connect(_frame(b"\x55N3HYM-1\xb0", b"N3HYM-", invert)) is None


@pytest.mark.parametrize("byte", (6, 7, 8))
def test_short_address_requires_padding_after_terminator(byte):
    addr = bytearray(b"\x55N3HYM\x0f\x0f\x0f")
    addr[byte] = 0
    assert p1rx.decode_connect(_frame(addr, b"N3HYM\x0f")) is None


@pytest.mark.parametrize("invert", (False, True))
def test_terminated_address_survives_bad_redundancy(invert):
    got = p1rx.decode_connect(_frame(b"\x55N3HYM\x0f\x0f\x0f", bytes(6), invert))
    assert got is not None
    assert (got.callsign, got.variant, got.inverted) == ("N3HYM", "Normal", invert)


@pytest.mark.parametrize("invert", (False, True))
@pytest.mark.parametrize("n", range(6))
def test_full_address_requires_complete_redundancy(invert, n):
    assert p1rx.decode_connect(_frame(b"\x55N3HYM-10", b"N3HYM-"[:n], invert)) is None


@pytest.mark.parametrize("invert", (False, True))
def test_full_address_requires_the_rate_change(invert):
    assert p1rx.decode_connect(_frame(b"\x55N3HYM-10", b"N3HYM-", invert,
                                      red_sps=p1rx.FS // 100)) is None


@pytest.mark.parametrize("session,call,first,count,logname", SESSIONS)
def test_recorded_connect_bursts(session, call, first, count, logname):
    audio_path = TREE / "captures" / session / "stream.flac"
    log_path = TREE / "working" / session / logname
    if not audio_path.exists() or not log_path.exists():
        pytest.skip(f"individual recording/log links absent for {session}")
    bounds = []
    for line in log_path.read_text().splitlines():
        if "tx_event" not in line or "completed" not in line:
            continue
        event = json.loads(line[line.index("{"):])
        if event.get("event") == "completed" and event.get("label") == f"connect->{call}":
            bounds.append((event["audio_start_sample"], event["audio_end_sample"]))
    assert bounds == [(first + i * 60000, first + i * 60000 + 46080)
                      for i in range(count)]
    audio = p1rx.load_wav(str(audio_path))
    results = [p1rx.decode_connect(audio[max(0, lo - 960):hi + 960])
               for lo, hi in bounds]
    assert results[0] is None
    assert [(c.callsign, c.variant, c.inverted) if c else None
            for c in results[1:]] == [(call, "Normal", bool(i % 2))
                                      for i in range(1, count)]


def test_noise_and_other_fsk_are_not_connects():
    rng = np.random.default_rng(10022138)
    for _ in range(60):
        assert p1rx.decode_connect(rng.normal(0, 0.11, 2 * p1rx.FS)) is None
    for invert in (False, True):
        for baud in (100, 200):
            audio = pactor1.packet_signal(b"N3HYM-10", baud=baud, invert=invert)
            assert p1rx.decode_connect(audio) is None
        audio = pactor1.connect_signal("N3HYM-10", variant="robust", invert=invert)
        assert p1rx.decode_connect(audio) is None
    for hz in (1400, 1500, 1600):
        carrier = 0.11 * np.cos(2 * np.pi * hz * np.arange(2 * p1rx.FS) / p1rx.FS)
        assert p1rx.decode_connect(carrier) is None


@pytest.mark.parametrize("bits_per_cell", (1, 2, 3))
def test_pactor2_is_not_a_connect(bits_per_cell):
    rng = np.random.default_rng(bits_per_cell)
    bits = rng.integers(0, 2, (2, 200 * bits_per_cell), dtype=np.uint8)
    audio = pactor2.frame(*bits, k=2, bits_per_cell=bits_per_cell, fs=p1rx.FS)
    assert p1rx.decode_connect(audio) is None


@pytest.mark.parametrize("name", ("ws8eoc-0908-p3-rms.wav",
                                 "ws8eoc-0909-p3-banner.wav",
                                 "ws8eoc-0909-p3-rms.wav"))
def test_recorded_pactor3_is_not_a_connect(name):
    path = Path(__file__).with_name("fixtures") / name
    if not path.exists():
        pytest.skip(f"PACTOR-3 fixture absent: {name}")
    assert p1rx.decode_connect(p1rx.load_wav(str(path))) is None


@pytest.mark.parametrize("call", ("N3HYM-10", "AA1AAA-1"))
@pytest.mark.parametrize("invert", (False, True))
def test_live_front_reports_full_address(call, invert):
    events = list(rxfront.decode_events(pactor1.connect_signal(call, invert=invert)))
    connects = [event.connect for event in events if event.kind == "connect"]
    assert connects
    assert all((c.callsign, c.variant, c.inverted) == (call, "Normal", invert)
               for c in connects)
