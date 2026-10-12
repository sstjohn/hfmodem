# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""MIL-STD-188-110 serial end to end, through the channels in ``bateleur.sim``.

This end against itself: every claim here is self-consistency, and says
nothing about agreement with any other implementation of the standard.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import mil110, sim, spec

MSG = bytes(np.random.default_rng(11).integers(0, 256, 300, dtype=np.uint8))


def _air(mode, data=MSG, trim=True, **ch):
    a = np.concatenate([np.zeros(3000), mil110.transmit(data, mode, trim=trim),
                        np.zeros(3000)])
    return sim.channel(a, **ch)


@pytest.mark.parametrize("mode", [m for m in spec.MODES110 if not m.endswith("L")])
def test_every_short_mode_round_trips_and_names_itself(mode):
    """The preamble names the mode; zero interleave is the one thing the
    receiver has to be told (5.3.2.3.7.2.1)."""
    r = mil110.receive(_air(mode, snr_db=20, freq_hz=41.0, seed=1),
                       zero_interleave=mode.endswith("Z"))
    assert r.mode.name == mode
    assert r.eom and r.data == MSG
    assert r.freq_hz == pytest.approx(41.0, abs=1.0)


def test_a_long_interleave_round_trips():
    r = mil110.receive(_air("1200L", snr_db=15, freq_hz=-20.0, seed=2))
    assert r.mode.name == "1200L" and r.eom and r.data == MSG


def test_awgn_at_the_standard_s_figure():
    """2400 bit/s in 10 dB of white noise, where the standard asks for a
    bit error rate of 1e-5; this has to be error-free on a 300-byte message."""
    for seed in range(3):
        r = mil110.receive(_air("2400S", snr_db=10, freq_hz=12.0, seed=seed))
        assert r.data == MSG


def test_the_untrimmed_pulse_is_received_the_same_way():
    """A far end whose rig passes 3.2 kHz sends the whole pulse; nothing on
    the air says so, and nothing here needs telling."""
    for mode in ("2400S", "600L"):
        r = mil110.receive(_air(mode, trim=False, snr_db=15, freq_hz=-30.0, seed=5))
        assert r.mode.name == mode and r.eom and r.data == MSG


@pytest.mark.parametrize("mode", ["600S", "300S"])
def test_poor_channel(mode):
    """CCIR poor: two paths 2 ms apart, each fading with 1 Hz of spread."""
    for seed in range(2):
        r = mil110.receive(_air(mode, snr_db=12, preset="poor", freq_hz=-25.0, seed=seed))
        assert r.data == MSG, seed


def test_a_rig_filter_narrower_than_the_signal():
    """The signal spans 490-2985 Hz; an eighth-order 300-2700 Hz filter, a
    common SSB transmit default, takes its top 300 Hz and the equalizer puts
    the symbols back."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(8, [300, 2700], btype="band", fs=spec.FS, output="sos")
    a = np.concatenate([np.zeros(3000), mil110.transmit(MSG, "2400S"), np.zeros(3000)])
    r = mil110.receive(sim.channel(sosfiltfilt(sos, a), snr_db=15, seed=8))
    assert r.data == MSG


def test_sound_card_clocks_that_disagree():
    """150 ppm over half a minute is more than two symbols of slip."""
    data = MSG * 30
    r = mil110.receive(_air("2400S", data, snr_db=15, clock_ppm=150, seed=4))
    assert r.data == data


def test_two_transmissions_in_one_recording():
    a = mil110.transmit(b"first", "600S")
    b = mil110.transmit(b"second", "1200S")
    audio = sim.channel(np.concatenate([np.zeros(2000), a, np.zeros(9000), b, np.zeros(2000)]),
                        snr_db=25, seed=5)
    r1 = mil110.receive(audio)
    r2 = mil110.receive(audio, start=r1.end_sample)
    assert (r1.data, r1.mode.name) == (b"first", "600S")
    assert (r2.data, r2.mode.name) == (b"second", "1200S")


def test_noise_is_not_a_transmission():
    noise = np.random.default_rng(6).normal(0, 0.1, 5 * spec.FS)
    assert mil110.receive(noise) is None


def test_empty_message():
    r = mil110.receive(_air("600S", b"", snr_db=20, seed=7))
    assert r.eom and r.data == b""


def test_a_transmission_late_in_a_long_recording():
    """Acquisition scans a few seconds at a time; the signal is past the first
    few windows, and one straddles a window edge."""
    for lead_s in (19.0, 7.99):
        lead = np.random.default_rng(9).normal(0, 0.02, int(lead_s * spec.FS))
        a = np.concatenate([lead, mil110.transmit(b"late", "600S"), np.zeros(2000)])
        r = mil110.receive(sim.channel(a, snr_db=20, seed=9))
        assert r.data == b"late", lead_s


def test_the_alternate_count_reading_is_received():
    """Receive the alternate 01/10 row order when consecutive counts agree.
    This synthetic preamble does not establish which edition prints that order.
    """
    mode = spec.MODES110["600S"]
    pre = np.concatenate([
        mil110._expand(spec.PREAMBLE_SYNC + (mode.d1, mode.d2)
                       + mil110.count_symbols(c, table_b=True) + (0,))
        for c in (2, 1, 0)])
    from hfmodem.bateleur import phy
    audio = phy.modulate(np.concatenate([pre, mil110.data_symbols(mode, MSG)]))
    y = sim.channel(np.concatenate([np.zeros(3000), audio, np.zeros(3000)]),
                    snr_db=20, seed=11)
    r = mil110.receive(y)
    assert r.mode.name == "600S" and r.data == MSG


def test_count_readings_are_distinct():
    assert mil110.count_symbols(23) == (5, 5, 7)            # the text's example
    assert mil110.count_symbols(2) == (4, 4, 6)
    assert mil110.count_symbols(2, table_b=True) == (4, 4, 5)


def test_75_bits_on_the_poor_channel():
    r = mil110.receive(_air("75S", MSG[:40], snr_db=5, preset="poor",
                            freq_hz=-20.0, seed=12))
    assert r.mode.name == "75S" and r.data == MSG[:40]


def test_75_uses_the_exceptional_sets_at_block_ends():
    """Table XIVb every 45th set: the 45th set of a short-interleave message
    is one of the complemented patterns."""
    mode = spec.MODES110["75S"]
    sym = mil110.data_symbols(mode, b"x" * 20)
    unscr = (sym - np.resize(mil110._SCR, sym.size)) % 8
    sets = unscr.reshape(-1, 32)
    normal = {tuple(spec.CHANNEL_PATTERNS[i]) for i in range(4)}
    special = {tuple(spec.CHANNEL_PATTERNS[i]) for i in range(4, 8)}
    for i, s in enumerate(sets):
        assert tuple(s) in (special if (i + 1) % 45 == 0 else normal), i


# --------------------------------------------------------------------------
# The 5 Hz channels of table XX, through the path-model equalizer
# --------------------------------------------------------------------------

def test_2400_on_the_2ms_5hz_channel():
    """Table XX asks 1e-3 of 2400 bit/s here at 30 dB; the RLS equalizer
    alone lost 0.29 of the bits."""
    data = MSG * 2
    for seed in (1, 2):
        r = mil110.receive(_air("2400L", data, snr_db=30, preset="2ms-5hz",
                                freq_hz=23.0, seed=seed))
        assert r.eom and r.data == data, seed


def test_75_bits_on_the_5ms_5hz_channel_at_2db():
    """The table's hardest row. The Walsh sets' error runs near 1 with the
    signal there at 2 dB, so the loss detector's level is set above that."""
    data = MSG[:60]
    r = mil110.receive(_air("75L", data, snr_db=2, preset="5ms-5hz",
                            freq_hz=23.0, seed=1))
    assert r.eom and r.data == data


def test_the_preamble_is_read_across_segments():
    """One segment's D and C symbols, read alone, misread 150 bit/s as 75 at
    5 dB on the 5 ms / 5 Hz channel; read against three segments, and with
    both paths' energy, it is read right."""
    a = np.concatenate([np.zeros(4000), mil110.transmit(bytes(300), "150L"), np.zeros(4000)])
    y = sim.channel(a, snr_db=5, preset="5ms-5hz", freq_hz=23.0, seed=7)
    from hfmodem.bateleur import phy
    bb = phy.downconvert(y)
    hit = phy.find(bb, mil110.SYNC_REF)
    mode, count = mil110.read_preamble(bb, hit)
    assert (mode.name, count) == ("150L", 23)


def test_a_path_that_appears_after_the_preamble_is_taken_in(monkeypatch):
    """A second fading path arrives after the preamble; the equalizer
    searches again and adds it without losing the message."""
    from hfmodem.bateleur import phy
    from hfmodem.bateleur.patheq import PathEqualizer
    data = bytes(np.random.default_rng(106).integers(0, 256, 1500, dtype=np.uint8))
    a = np.concatenate([np.zeros(4000), mil110.transmit(data, "2400L"), np.zeros(4000)])
    onset = 4000 + phy.leadin_samples() + len(mil110.preamble(spec.MODES110["2400L"])) * spec.SPS
    # A seed alone need not hide a path during acquisition. Hold the second
    # path off through the preamble, then bring it up over two seconds.
    envelopes = iter((1.0, np.clip((np.arange(a.size) - onset) / (2 * spec.FS), 0, 1)))
    fading = sim._fading

    def arriving_path(n, spread, rng):
        return fading(n, spread, rng) * next(envelopes)

    with monkeypatch.context() as patch:
        patch.setattr(sim, "_fading", arriving_path)
        y = sim.channel(a, snr_db=18, preset="poor", freq_hz=23.0, seed=6)
    bb = phy.downconvert(y)
    hit = phy.find(bb, mil110.SYNC_REF)
    mode, count = mil110.read_preamble(bb, hit)
    plan = mil110._plan(mode, count, 200)
    eq = PathEqualizer()
    eq.start(hit.sample, hit.freq_hz, hit.gain, plan.constellation)
    eq.advance(bb[:eq.needs()], 0, plan)
    assert len(eq.delays) == 1
    eq.advance(bb, 0, plan)
    assert len(eq.delays) == 2
    # Delays are relative to whichever equal-power path acquisition found.
    assert round(max(eq.delays) - min(eq.delays)) == round(0.002 * spec.FS)
    r = mil110.receive(y)
    assert r is not None and r.eom and r.data == data


def test_clock_error_on_a_fading_channel():
    """100 ppm moves the paths ten samples in a ten-second message; the
    path model's timing loop follows them."""
    data = MSG * 10
    r = mil110.receive(_air("2400L", data, snr_db=18, preset="poor", clock_ppm=100,
                            freq_hz=23.0, seed=0))
    assert r.eom and r.data == data
