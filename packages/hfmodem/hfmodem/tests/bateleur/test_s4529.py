# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 4529, against the figures its conformance test procedures give."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import s4285, s4529, sim, stream
from hfmodem.bateleur.modem import Mode, ModemError
from hfmodem.bateleur.spec import FS, MODES4285

MSG = b"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789\r\n" * 4


def test_the_sync_pattern_the_procedures_print():
    """Table 1.6, step 6: the 80 2-PSK sync symbols as bits."""
    want = ("01011001111100011011" "10101000010010110011"
            "11100011011101010000" "10010110011111000110")
    assert "".join(str(v // 4) for v in s4285.SYNC) == want


@pytest.mark.parametrize("rate,long_,short", [
    (1200, 24678, 2150), (600, 12390, 1126), (300, 6246, 614), (150, 3174, 358),
    (75, 1638, 230)])
def test_the_flush_each_rate_takes(rate, long_, short):
    """Paragraph 1.2 j: the flush zeros, which are 4285's for twice the rate."""
    assert MODES4285[s4529.mode(f"{rate}L").as_4285].flush_bits == long_
    assert MODES4285[s4529.mode(f"{rate}S").as_4285].flush_bits == short


def test_1200_baud_and_99_percent_in_1240_hz():
    """1.2 b and d: 1200 Bd; 99% of the power within 1240 Hz of the
    sub-carrier."""
    x = s4529.transmit(bytes(np.random.default_rng(1).integers(0, 256, 600, np.uint8)),
                       "1800U")
    s = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / FS)
    assert np.sum(s[np.abs(f - 1700) <= 620]) >= 0.99 * np.sum(s)
    # The same symbols as 4285's at 2400 Bd, each twice as long.
    data = bytes(np.random.default_rng(1).integers(0, 256, 600, np.uint8))
    assert x.size == 2 * s4285.transmit(data, "3600U", trim=False).size


def test_the_sub_carrier_is_selectable():
    m = s4529.mode("600L/2000")
    assert (m.name, m.carrier, m.full_name) == ("600L", 2000.0, "600L/2000")
    for bad in ("600L/750", "600L/2450", "600L/1750", "900L"):
        with pytest.raises(ValueError):
            s4529.mode(bad)


@pytest.mark.parametrize("name,snr,preset,df", [
    ("1200S", 20, "awgn", -15.0), ("600L/2000", 15, "awgn", 20.0),
    ("300S", 8, "awgn", 0.0), ("75S", 3, "awgn", 30.0), ("150L", 12, "poor", 0.0),
    ("1800U", 25, "awgn", 0.0)])
def test_round_trips(name, snr, preset, df):
    """1.2 i: offsets to 37.5 Hz."""
    x = np.r_[np.zeros(4801), s4529.transmit(MSG, name), np.zeros(19200)]
    r = s4529.receive(sim.channel(x, snr_db=snr, preset=preset, freq_hz=df, seed=2), name)
    assert r is not None and r.data[:len(MSG)] == MSG
    assert abs(r.freq_hz - df) < 2


def test_the_stream_gives_what_the_whole_gives():
    x = np.r_[np.zeros(4801), s4529.transmit(MSG, "600S/1500"), np.zeros(19200)]
    y = sim.channel(x, snr_db=15, freq_hz=12.0, seed=5)
    r = stream.Receiver4529("600S/1500")
    events = []
    for i in range(0, y.size, 777):
        events += r.push(y[i:i + 777])
    started = [e for e in events if isinstance(e, stream.Started)]
    assert started and started[0].standard == "4529" and started[0].mode == "600S/1500"
    assert abs(started[0].freq_hz - 12.0) < 2
    got = b"".join(e.data for e in events if isinstance(e, stream.Data))
    assert got == s4529.receive(y, "600S/1500").data


def test_the_modem_keys_only_inside_the_declared_span():
    assert Mode.parse("4529", "600l/2000") == Mode("4529", "600L/2000")
    for name in ("600L/800", "600L/2400"):
        with pytest.raises(ModemError, match="outside"):
            Mode.parse("4529", name)
