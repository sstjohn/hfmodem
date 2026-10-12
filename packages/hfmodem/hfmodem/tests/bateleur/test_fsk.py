# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Binary FSK: STANAG 4481 annex C and MIL-STD-188-110B 5.1, against itself."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import fsk, sim, stream
from hfmodem.bateleur.modem import Mode, ModemError
from hfmodem.bateleur.spec import FS

MSG = b"NAVAL BROADCAST TEST 0123456789 the quick brown fox\r\n" * 3


def _cer(got: bytes, want: bytes) -> float:
    """Edit distance over the length wanted."""
    prev = np.arange(len(want) + 1)
    for i, a in enumerate(got, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        for j, b in enumerate(want, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return float(prev[-1]) / len(want)


def test_the_characteristic_frequencies():
    """110B table IV (HF: mark 1575, space 2425, centre 2000; mark the
    lower, 4.2.2), 4481 C.2 (850 Hz wide, space higher, the mark ending in
    75 Hz) and C.3 (255 + 170n, 42.5 Hz either way, n twinned with n + 8)."""
    assert fsk.mode("75W").pairs() == [(1575.0, 2425.0)]
    assert fsk.mode("75W/1700").pairs() == [(1275.0, 2125.0)]
    assert fsk.mode("50N").pairs() == [(1915.0, 2085.0)]
    assert fsk.mode("75V3").pairs() == [(722.5, 807.5)]
    assert fsk.mode("75V3D").pairs() == [(722.5, 807.5), (2082.5, 2167.5)]
    assert [fsk.vft_twin(n) for n in (1, 8, 9, 16)] == [9, 16, 1, 8]
    for name in ("75W", "75W/1700"):
        assert fsk.mode(name).pairs()[0][0] % 100 == 75
    with pytest.raises(ValueError):
        fsk.mode("75X")
    with pytest.raises(ValueError):
        fsk.mode("600N")                          # faster than the shift carries


def test_mark_is_the_lower_tone():
    x = fsk.transmit(b"", "75W")                 # all idle: mark
    s = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(x.size, 1 / FS)
    assert abs(f[np.argmax(s)] - 1575) < 5


@pytest.mark.parametrize("name", ["75W", "75W/1700", "50N", "150W", "300W", "75V3", "75V3D"])
def test_round_trips(name):
    x = np.r_[np.zeros(4800), fsk.transmit(MSG, name), np.zeros(9600)]
    r = fsk.receive(sim.channel(x, snr_db=25, seed=1), name)
    assert r.data == MSG


def test_the_synchronous_stream():
    x = np.r_[np.zeros(4800), fsk.transmit(MSG, "75WS"), np.zeros(9600)]
    r = fsk.receive(sim.channel(x, snr_db=25, seed=1), "75WS")
    assert r.data.startswith(MSG)                # the idle tail may follow as 0xFF


@pytest.mark.parametrize("name,snr,preset,df,most", [
    ("75W", 6, "awgn", 25.0, 0.01), ("75W/1700", 10, "awgn", -30.0, 0.02),
    ("50N", 10, "awgn", 12.0, 0.01), ("75W", 20, "poor", 0.0, 0.10),
    ("75V3D", 6, "poor", 0.0, 0.05)])
def test_noise_offset_and_fading(name, snr, preset, df, most):
    x = np.r_[np.zeros(4800), fsk.transmit(MSG, name), np.zeros(9600)]
    r = fsk.receive(sim.channel(x, snr_db=snr, preset=preset, freq_hz=df, seed=3), name)
    assert r is not None and abs(r.freq_hz - df) < 3
    assert _cer(r.data, MSG) <= most


def test_noise_starts_nothing():
    noise = np.random.default_rng(5).normal(0, 0.1, FS * 60)
    r = stream.ReceiverFsk("75W")
    events = []
    for i in range(0, noise.size, 960):
        events += r.push(noise[i:i + 960])
    assert not events


def test_the_stream_in_pieces_and_back_to_back():
    x = np.r_[np.zeros(4800), fsk.transmit(MSG, "75W"), np.zeros(3 * FS),
              fsk.transmit(MSG[::-1], "75W"), np.zeros(9600)]
    y = sim.channel(x, snr_db=20, seed=4)
    r = stream.ReceiverFsk("75W")
    events = []
    for i in range(0, y.size, 777):
        events += r.push(y[i:i + 777])
    events += r.flush()
    kinds = [type(e).__name__ for e in events if not isinstance(e, stream.Data)]
    assert kinds == ["Started", "Ended", "Started", "Ended"]
    assert b"".join(e.data for e in events if isinstance(e, stream.Data)) == MSG + MSG[::-1]


def test_the_modem_keys_only_inside_the_declared_span():
    assert Mode.parse("FSK", "75v3d") == Mode("FSK", "75V3D")
    for name in ("75V1", "75V16", "75V8D"):
        with pytest.raises(ModemError, match="outside"):
            Mode.parse("FSK", name)
    m = Mode.parse("FSK", "75W")
    assert m.seconds(100) == pytest.approx(fsk.transmit(bytes(100), "75W").size / FS)
