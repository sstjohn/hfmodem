# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""MIL-STD-188-110 Appendix B, the 39-tone parallel modem, end to end
through ``bateleur.sim``: self-consistency, and the receiver's tolerance of
what real senders and channels do (offsets, clocks, fading, other editions'
framing, preambles laid out a little differently)."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import sim, tone39
from hfmodem.bateleur.spec import FS
from hfmodem.bateleur.tone39 import T

MSG = bytes(np.random.default_rng(39).integers(0, 256, 120, dtype=np.uint8))
TEXT = b"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789\r\n" * 2


def _air(mode, data=MSG, **ch):
    a = np.concatenate([np.zeros(FS // 2), tone39.transmit(data, mode), np.zeros(FS // 2)])
    ch.setdefault("seed", 1)
    return sim.channel(a, **ch)


def _want(mode, data):
    m = tone39.mode(mode)
    return bytes(b & ((1 << m.data_bits) - 1) for b in data) if m.char else data


# --------------------------------------------------------------------------
# The tables and the framing sequence

def test_the_framing_sequence_ends_with_nine_marks_and_a_space():
    s = tone39.framing_bits(252)
    assert s.size == 252 and s[-1] == 0 and s[-10:-1].all()
    # One period of a nine-stage register: 511 bits.
    assert tone39._MSEQ.size == 511
    cycle = tone39.framing_bits(511)
    assert cycle.sum() == 256
    for i in range(511):
        assert cycle[i] == (cycle[(i-2) % 511] ^ cycle[(i-3) % 511]
                            ^ cycle[(i-5) % 511] ^ cycle[(i-9) % 511])
    for divisor in (1, 7, 73):
        assert not np.array_equal(cycle, np.roll(cycle, divisor))


def test_mode_names_round_trip():
    for m in tone39.all_modes():
        assert tone39.mode(m.name) == m
    assert tone39.mode("600/99").char == 10
    assert tone39.mode("75/36/SF").diversity == "f"
    assert tone39.mode("1200/567/SX").extended
    assert tone39.mode("75/4/S:384").framing == (0, 0, 384)
    with pytest.raises(ValueError):
        tone39.mode("600/98/S")


def test_rates_and_codes():
    assert tone39.mode("2400/288/S").n == 14 and tone39.mode("2400/288/S").k == 10
    for rate in (75, 150, 300, 600, 1200):
        m = tone39.mode(f"{rate}/1/S")
        assert (m.n, m.k) == (7, 3)
        # Coded bits an element, times its 44.4 elements a second, are the
        # rate over the code rate (the framing aside).
        assert m.word_bits * FS / T >= rate * 7 / 3


# --------------------------------------------------------------------------
# Loopback

@pytest.mark.parametrize("mode", ["2400/9/S", "1200/63/S", "600/33/A10.8", "300/17/S",
                                  "150/9/SF", "75/5/A7.5", "2400/36/A9.7"])
def test_modes_round_trip(mode):
    r = tone39.receive(_air(mode, snr_db=25, freq_hz=17.0), mode)
    assert r is not None
    assert r.data[: len(MSG)] == _want(mode, MSG)


def test_the_extended_preamble():
    r = tone39.receive(_air("600/33/SX", snr_db=25, freq_hz=-11.0), "600/33/SX")
    assert r is not None and r.data[: len(MSG)] == MSG


def test_async_ends_at_the_eom():
    r = tone39.receive(_air("600/33/A10.8", TEXT, snr_db=25), "600/33/A10.8")
    assert r.eom and r.data == TEXT


@pytest.mark.parametrize("hz", [-75.0, -23.0, 10.0, 15.0, 60.0])
def test_carrier_offsets(hz):
    """The offset turns every phase change by 2 pi df T, 8 degrees a hertz:
    past 5.5 Hz that is beyond what the fourth power can tell apart, so the
    known part must come off first."""
    r = tone39.receive(_air("1200/63/S", snr_db=25, freq_hz=hz), "1200/63/S")
    assert r is not None and abs(r.freq_hz - hz) < 0.5
    assert r.data[: len(MSG)] == MSG


@pytest.mark.parametrize("ppm", [-3000.0, 2000.0])
def test_a_sender_whose_clock_is_far_off(ppm):
    r = tone39.receive(_air("600/33/S", snr_db=25, freq_hz=9.0, clock_ppm=ppm), "600/33/S")
    assert r is not None and abs(r.clock_ppm - ppm) < 200
    assert r.data[: len(MSG)] == MSG


def test_sound_card_clocks_are_measured_from_the_data():
    """A few hundred ppm walks a long message's elements out of the guard
    (300 ppm: 65 samples in 22 s); the data says by how much, and each
    element is read where it falls."""
    long_msg = MSG * 14
    r = tone39.receive(_air("600/33/S", long_msg, snr_db=10, freq_hz=7.0, clock_ppm=-300.0,
                            paths=sim.two_paths(2.0, 2.0), seed=1), "600/33/S")
    assert r is not None and abs(r.clock_ppm + 300) < 40
    assert r.data[: len(long_msg)] == long_msg


@pytest.mark.parametrize("ppm", [-300.0, 0.0, 300.0])
def test_clock_tracking_on_two_fading_paths(ppm):
    long_msg = MSG * 14
    r = tone39.receive(_air("600/33/S", long_msg, snr_db=10, freq_hz=7.0, clock_ppm=ppm,
                            paths=sim.two_paths(2.0, 2.0), seed=0), "600/33/S")
    assert r is not None and abs(r.clock_ppm - ppm) < 40
    assert r.data[: len(long_msg)] == long_msg


# --------------------------------------------------------------------------
# What the receiver has to work out for itself

@pytest.mark.parametrize("mode", ["600/33/A10.8", "300/17/S", "2400/9/S", "1200/63/A9.7"])
def test_the_mode_is_detected(mode):
    data = TEXT if tone39.mode(mode).char else MSG
    r = tone39.receive(_air(mode, data, snr_db=25, freq_hz=5.0))
    assert r is not None and r.mode == tone39.mode(mode)
    assert r.data[: len(data)] == _want(mode, data)


def test_framing_of_another_length_is_measured():
    """A sender working from another edition's table: a framing sequence
    the tables here do not have is measured from its end, the SPACE after
    nine MARKs, and the degree found from the code words."""
    sent = tone39.Mode39(75, 4, "tf", False, 0, 8, seq=384)
    r = tone39.receive(_air(sent, MSG[:20], snr_db=25))
    assert r is not None and r.mode.seq == 384 and r.mode.degree == 4
    assert r.data[:20] == MSG[:20]


def test_a_preamble_an_element_longer():
    """Senders lay out the preamble's parts a little differently; the data
    is found where the framing sequence opens it."""
    m = tone39.mode("600/33/S")
    a = tone39.transmit(MSG, m)
    p3 = 22 * T
    a = np.concatenate([np.zeros(FS // 2), a[:p3 + T], a[p3:p3 + T], a[p3 + T:],
                        np.zeros(FS // 2)])
    r = tone39.receive(sim.channel(a, snr_db=25, seed=2), m)
    assert r is not None and r.data[: len(MSG)] == MSG


def test_characters_framed_as_on_the_line_are_recognised():
    m = tone39.mode("600/33/A9.7")
    c = tone39.chars(b"HELLO", m).reshape(-1, 9)
    assert tone39._formed(c.ravel(), m) == 1.0            # all SPACE, per B.5.7
    c[:, -1] = 1                                          # stop bits MARK
    assert tone39._formed(c.ravel(), m) == 1.0
    c[:, 0] = 1                                           # no start bits
    assert tone39._formed(c.ravel(), m) < 0.5


# --------------------------------------------------------------------------
# Decoding and channels

def test_soft_decoding_goes_past_two_symbol_errors():
    """Three nibbles wrong in a (7,3) word is beyond the hard decoder; with
    their bits marked unsure the nearest code word is still the one sent."""
    m = tone39.mode("300/1/S")
    data = np.random.default_rng(5).integers(0, 2, m.superblock_data).astype(np.uint8)
    coded = tone39.encode_superblock(m, data)
    llr = (1.0 - 2.0 * coded) * 4.0
    for sym in (0, 3, 5):                                  # one bit in each
        llr[sym * 4] = -0.5 * np.sign(llr[sym * 4])
    bits, fixed, failed = tone39.decode_superblock(m, llr)
    assert np.array_equal(bits, data) and fixed == 1


def test_two_fading_paths():
    """Table B-XII's channel: two equal paths 2 ms apart, 2 Hz of spread."""
    r = tone39.receive(_air("1200/63/S", snr_db=20, freq_hz=23.0,
                            paths=sim.two_paths(2.0, 2.0), seed=3), "1200/63/S")
    assert r is not None and r.data[: len(MSG)] == MSG


def test_found_at_low_snr_through_fading():
    a = _air("75/36/S", MSG[:10], snr_db=4, freq_hz=23.0, paths=sim.two_paths(2.0, 2.0), seed=5)
    got = tone39.find_preamble(a)
    assert got is not None
    part3 = tone39.settle(a, *got)
    assert abs(part3 - (FS // 2 + 22 * T + (tone39._BANDPASS.size - 1) // 2) + 20) <= 40      # inside the guard


def test_nothing_in_noise():
    rng = np.random.default_rng(0)
    assert tone39.receive(rng.normal(0, 1000, 8 * FS)) is None


_LONG = b"".join(b"line %04d of a long message sent in the 39-tone mode\r\n" % i
                 for i in range(80))


@pytest.mark.parametrize("name,cut_s,preset", [
    ("1200/63", 5.0, "awgn"), ("600/33", 8.0, "poor"), ("2400/9/S", 3.0, "awgn"),
    ("2400/36/A7.5", 4.0, "awgn")])
def test_late_entry_joins_at_the_next_framing(name, cut_s, preset):
    """B.5.4.3: the framing goes back into the data every few super blocks,
    and from each insertion the data runs on as it did from the first; a
    receiver that missed the preamble starts there. The last mode's
    insertions begin inside an element."""
    m = tone39.mode(name)
    msg = _LONG[:2500]
    if m.char == 7:
        msg = bytes(b & 0x1F for b in msg)
    x = tone39.transmit(msg, m)
    y = sim.channel(np.r_[np.zeros(3000), x, np.zeros(9600)], snr_db=20, preset=preset,
                    freq_hz=13.0, seed=2)[3000 + int(cut_s * FS):]
    assert tone39.receive(y, m) is None or tone39.receive(y, m).data != msg
    r = tone39.late_entry(y, m)
    assert r is not None and abs(r.freq_hz - 13.0) < 1.0
    head = r.data[:64]
    at = msg.find(head)
    assert at > 0 and len(head) == 64
    if m.char:
        assert r.eom and r.data == msg[at:]
    else:
        n = min(len(r.data), len(msg) - at)
        assert r.data[:n] == msg[at:at + n] and n > 200


def test_late_entry_finds_nothing_in_noise():
    noise = np.random.default_rng(1).normal(0, 0.1, FS * 20)
    assert tone39.late_entry(noise, "600/33") is None
