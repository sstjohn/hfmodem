# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 16-tone DPSK mode (MIL-STD-188-110B appendix A) against itself."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import sim, stream, tone16
from hfmodem.bateleur.spec import FS

MSG = b"The quick brown fox jumps over the lazy dog 0123456789 " * 4


def _ber(got: bytes, want: bytes) -> float:
    n = min(len(got), len(want))
    a = np.unpackbits(np.frombuffer(got[:n], np.uint8))
    b = np.unpackbits(np.frombuffer(want[:n], np.uint8))
    return float(np.mean(a != b)) + (len(want) - n) * 8 / (8 * len(want))


def test_the_tones_and_the_element():
    """A.5.1, A.5.2, table A-I."""
    assert tone16.T / FS == pytest.approx(1 / 75)
    assert list(tone16.TONE_HZ[[0, 7, 15]]) == [935.0, 1705.0, 2585.0]
    assert tone16.SLOT_HZ not in tone16.TONE_HZ
    # The window read makes the tones orthogonal: 1/110 s, to a sample.
    assert abs(tone16.W / FS - 1 / 110) < 1 / FS


def test_table_a_ii_and_the_diversity_of_table_a_i():
    # 2400: a dibit a tone; (odd, even) = (MARK, SPACE) is +225 degrees.
    d = tone16.changes(2400, bytes([0b00000001]) + bytes(3))
    assert d.shape == (1, 16) and d[0, 0] == 225.0 and d[0, 1] == 135.0
    # 1200: tones 9-16 repeat 1-8.
    d = tone16.changes(1200, bytes([0x5A, 0xC3]))
    assert np.array_equal(d[0, :8], d[0, 8:])
    # 150: the first bit on the odd tones, the second on the even.
    d = tone16.changes(150, bytes([0b01]))
    assert set(d[0, ::2]) == {315.0} and set(d[0, 1::2]) == {135.0}
    # 75: one bit on all sixteen.
    assert set(tone16.changes(75, bytes([1]))[0]) == {315.0}


def test_the_preamble_and_the_reference_element():
    """A.5.4: 605 and 1705 Hz, the latter reversing each element; then one
    element of every tone."""
    x = tone16.transmit(b"", 2400)
    n = tone16.PREAMBLE_ELEMENTS
    assert x.size == (n + 1) * tone16.T
    at = np.arange(n) * tone16.T + tone16.GUARD
    z = tone16.read(x, at, 0.0, np.r_[tone16.DOPPLER_HZ, tone16.TONE_HZ])
    p = np.abs(z) ** 2
    assert np.all(p[:, 8] > 50 * np.delete(p, [0, 8], axis=1).max(axis=1))
    assert 10 * np.log10(np.mean(p[:, 0]) / np.mean(p[:, 8])) == pytest.approx(7.0, abs=0.2)
    assert np.allclose(np.abs(np.angle(z[1:, 8] * np.conj(z[:-1, 8]))), np.pi, atol=0.02)
    ref = tone16.read(x, np.array([n * tone16.T + tone16.GUARD]), 0.0, tone16.TONE_HZ)[0]
    assert np.all(np.abs(ref) > 0.5 * np.abs(ref).max())


@pytest.mark.parametrize("rate", tone16.RATES)
def test_every_rate_round_trips(rate):
    x = np.r_[np.zeros(3000), tone16.transmit(MSG, rate), np.zeros(4000)]
    r = tone16.receive(x, rate)
    assert r.data[:len(MSG)] == MSG
    assert set(r.data[len(MSG):]) <= {0}           # the last element's fill
    # Anywhere in the run of windows inside the element will do.
    assert abs(r.start_sample - (3000 + tone16.PREAMBLE_ELEMENTS * tone16.T)) <= 16


@pytest.mark.parametrize("rate,snr,preset,df", [
    (2400, 15, "awgn", 23.0), (1200, 10, "awgn", -41.0), (600, 15, "poor", -11.0),
    (300, 10, "poor", 7.0), (75, 0, "awgn", 0.0)])
def test_through_noise_fading_and_offset(rate, snr, preset, df):
    x = np.r_[np.zeros(4000), tone16.transmit(MSG, rate), np.zeros(9600)]
    y = sim.channel(x, snr_db=snr, preset=preset, freq_hz=df, seed=1)
    r = tone16.receive(y, rate)
    assert r is not None and abs(r.freq_hz - df) < 2.0
    assert _ber(r.data, MSG) < 0.01


@pytest.mark.parametrize("ppm", [-100, 100])
def test_a_sender_s_clock_is_followed(ppm):
    msg = bytes(np.random.default_rng(3).integers(0, 256, 6000, dtype=np.uint8))
    x = np.r_[np.zeros(4000), tone16.transmit(msg, 2400), np.zeros(9600)]
    y = sim.channel(x, snr_db=20, seed=2, clock_ppm=ppm)
    assert tone16.receive(y, 2400).data[:len(msg)] == msg


def test_noise_starts_nothing():
    noise = np.random.default_rng(5).normal(0, 0.1, FS * 120)
    r = stream.Receiver16(600)
    events = []
    for i in range(0, noise.size, 960):
        events += r.push(noise[i:i + 960])
    assert not events


def test_the_stream_gives_what_the_whole_gives():
    x = np.r_[np.zeros(4000), tone16.transmit(MSG, 1200), np.zeros(9600),
              tone16.transmit(MSG[::-1], 1200), np.zeros(9600)]
    y = sim.channel(x, snr_db=15, seed=4)
    r = stream.Receiver16(1200)
    events = []
    for i in range(0, y.size, 777):
        events += r.push(y[i:i + 777])
    kinds = [type(e).__name__ for e in events if not isinstance(e, stream.Data)]
    assert kinds == ["Started", "Ended", "Started", "Ended"]
    got = b"".join(e.data for e in events if isinstance(e, stream.Data))
    assert got.replace(b"\0", b"") == (MSG + MSG[::-1]).replace(b"\0", b"")
