# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Level measurement. The failure this guards is silent by nature."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.core import levels as L


def test_the_working_point_is_defined_here():
    assert L.WORKING_POINT == 0.040


def test_digital_silence_is_not_a_quiet_band():
    """A stalled capture returns -inf. It must not read as a usable quiet
    channel — that is how a deaf receiver looks like a dead one."""
    quiet = np.zeros(4800)
    assert L.peak_dbfs(quiet) == float("-inf")
    ok, why = L.usable(quiet)
    assert not ok and "silence" in why


def test_a_hot_input_is_a_failure_not_a_warning():
    hot = np.clip(np.random.default_rng(0).normal(0, 1.5, 48000), -1, 1)
    ok, why = L.usable(hot)
    assert not ok
    assert "rail" in why and "0.04" in why


def test_railed_is_measured_against_full_scale_not_the_capture_peak():
    """A normalising WAV writer destroys this: 9.6% railed off the codec reads
    0.00% from a file scaled to 0.8 peak."""
    a = np.full(1000, 0.999)
    assert L.railed_ppm(a) == pytest.approx(1e6)
    assert L.railed_ppm(a * 0.8) == 0.0


def test_the_measured_working_point_reads_as_usable():
    """Real traffic at the working point sits at -4 to -8 dBFS peak, nothing
    railed. A criterion that calls -5 a fault is one nobody will keep."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, 48000)
    a = a / np.abs(a).max() * 10 ** (-5.0 / 20)
    ok, why = L.usable(a)
    assert ok and "-5.0 dBFS" in why


def test_lost_headroom_is_reported_even_without_railing():
    a = np.zeros(1000); a[0] = 10 ** (-1.0 / 20)
    ok, why = L.usable(a)
    assert not ok and "headroom" in why


#: What the receiver was delivering on the 2026-08-22 session taps, in the
#: currency :func:`hfmodem.core.busy.occupancy_db` reads: the two that read none
#: of their own keyings back (`-0922`, `-0924`), and four that read all of theirs
#: (`-0929`, `-0933`, `-0936`, `-0938`). The bound between them is not a taste,
#: and this is what says so.
UNREADABLE_DBFS = (-39.6, -40.1)
READABLE_DBFS = (-23.5, -24.1, -24.3, -24.3)


def _at(level_dbfs, seconds=2.0, fs=48000):
    """Band noise the level gate reads at `level_dbfs`, measured on its own
    instrument rather than assumed off the RMS."""
    from hfmodem.core.busy import occupancy_db
    a = np.random.default_rng(7).normal(0, 0.05, int(seconds * fs))
    return a * 10 ** ((level_dbfs - float(occupancy_db(a, fs).max())) / 20)


def test_the_floor_lies_between_the_taps_that_read_and_the_taps_that_did_not():
    assert max(UNREADABLE_DBFS) < L.RX_FLOOR_DBFS < min(READABLE_DBFS)


def test_a_receiver_that_cannot_hold_a_mute_is_quiet_and_not_deaf():
    """The measured failure. It still delivers audio -- a gain, not a dead
    interface -- and calling it deaf sends the operator to the wrong end."""
    for level in UNREADABLE_DBFS:
        state, line = L.rx_verdict(_at(level))
        assert state == "quiet", line
        assert "TOO QUIET" in line and "NO AUDIO" not in line


def test_the_receivers_that_read_their_own_keyings_back_still_pass():
    """A guard that refuses the levels this station works at is one that gets
    forced every time, and then it is not a guard."""
    for level in READABLE_DBFS:
        state, line = L.rx_verdict(_at(level))
        assert state == "live", line


def test_a_codec_delivering_nothing_is_named_as_that_and_not_as_a_quiet_band():
    state, line = L.rx_verdict(np.zeros(96000))
    assert state == "silent"
    assert L.NO_AUDIO_CAUSE in line


def test_a_window_on_the_rail_is_clipping_whatever_its_level_reads():
    """Asked ahead of the level, so a window loud enough to rail is never
    reported as a quiet one."""
    rng = np.random.default_rng(3)
    state, line = L.rx_verdict(np.clip(rng.normal(0, 0.5, 96000), -1, 1))
    assert state == "hot" and "CLIPPING" in line


def test_the_gain_that_stopped_this_station_decoding_is_refused_before_it_keys():
    """2026-08-29 22:15z and 22:18z: one arm flown twice three minutes apart,
    everything but the capture gain held. At 0.209 the gate read rms 0.245 and
    passed, then 23 of 25 windows railed and 3 control signals came out; at 0.099
    nothing railed and 19 came out. Reproduced here at the measured levels --
    railing inside the clipping arm's 0.001-0.056% and an RMS under the hot bound,
    so the rail count is the only thing that can refuse it."""
    live = np.random.default_rng(11).normal(0, 0.12, 96000)
    hot = np.clip(live * 2.1, -1.0, 1.0)
    assert 0.001 <= L.railed_ppm(hot) / 1e4 <= 0.056
    assert np.sqrt(np.mean(hot ** 2)) < L.RX_HOT_RMS
    assert L.rx_verdict(hot)[0] == "hot"
    assert L.rx_verdict(live)[0] == "live"


def test_a_receiver_over_the_hot_bound_is_clipping_before_it_rails():
    """0.283 ran a session clean and 0.327 railed 16% of its samples, so the
    bound is between them and a reading past it is reported on its own."""
    t = np.arange(96000) / 48000
    a = (L.RX_HOT_RMS + 0.02) * np.sqrt(2) * np.sin(2 * np.pi * 1500 * t)
    assert L.railed_ppm(a) == 0.0
    state, _ = L.rx_verdict(a)
    assert state == "hot"


def test_every_refusal_names_the_control_that_reaches_this_converter():
    """`AF` drives the speaker tap, not the modem's feed. That one cost a
    recalibration, and a message is where it costs the next one."""
    for audio in (_at(UNREADABLE_DBFS[0]), np.clip(
            np.random.default_rng(4).normal(0, 0.5, 96000), -1, 1)):
        _, line = L.rx_verdict(audio)
        assert "codec_gain" in line and "NOT the rig's AF" in line


def test_a_quiet_band_is_sent_to_the_control_that_is_per_band():
    """2026-08-20: 20 m read 0.0046 RMS while 40 m read 0.055-0.065 on the same
    codec setting, and RF gain recovered 18 dB. A message that offers only the
    shared controls sends the operator to overdrive the band that was fine."""
    _, line = L.rx_verdict(_at(UNREADABLE_DBFS[0]))
    assert "RF gain" in line and "hamlib's `RF`" in line
    assert "per-band" in line and "never the codec" in line


def test_a_quiet_band_with_nothing_left_to_turn_says_so():
    """The reading an operator most needs to recognise fast. Hunting a control
    that is already at its stop costs the slot, and the band may just be shut."""
    _, line = L.rx_verdict(_at(UNREADABLE_DBFS[0]))
    assert "already wide open" in line
    assert "antenna" in line and "simply closed" in line
    assert "no control fixes either" in line
