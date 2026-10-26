# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Kestrel as BW500 caller answers stock where stock's own receiver would, and
asks for the turn behind stock's burst, not under it, with noise on its input.

Bench 1, Pat at BW500 with 0.3-0.6 RMS on kestrel's input, 2026-10-21
(pat-gate/cnl-b2a*): every noisy session
keyed two turn-requests 0.28-0.38 s into stock's 0.48 s answer to our release,
because the energy gate never opened on it; clean, the same request came 0.15-0.24
s behind it. Our answers reached stock anywhere from 0.04 to 0.30 s behind its
frame, keyed whenever the half-second scan named the frame; stock's receivers
answer 0.195-0.235 s behind.
"""

import numpy as np
import pytest
from scipy.signal import firwin, lfilter

from hfmodem.kestrel.arq import phy
from hfmodem.tests import evidence
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_idle_poll import CALLED, CALLER, _Clock, _station, clock  # noqa: F401

CABLES = (evidence.WORKING / "varac-on-air/pat-gate")
BLOCK = 960
_BAND = firwin(801, [300, 2700], pass_zero=False, fs=MK.FS)


def _band_noise(seconds, seed):
    return lfilter(_BAND, 1, np.random.default_rng(seed).standard_normal(int(seconds * MK.FS))) * 0.05


def _burst(pairs, snr_db=3.0, seed=1):
    """A two-tone burst in band noise at ``snr_db`` over the 1150-1850 Hz band."""
    x = MK.synth_tone_pairs(pairs)
    noise = _band_noise(len(x) / MK.FS, seed)
    gain = np.sqrt(np.mean(noise ** 2) * 2400 / 700 * 10 ** (snr_db / 10) / np.mean(x ** 2))
    return x * gain + noise


def _caller():
    hs, io = _station("500")
    hs.high_speed = hs._link_high_speed = True
    hs._track_band(_band_noise(4.0, 9))
    return hs, io


def _feed(hs, io, x):
    """Feed until something is keyed; the samples heard by then, or None."""
    sent = len(io.sent)
    for at in range(0, len(x), BLOCK):
        _Clock.t += BLOCK / MK.FS
        hs.on_rx_stream(x[at:at + BLOCK])
        if len(io.sent) > sent:
            return at + BLOCK
    return None


def test_the_ask_behind_our_release_waits_out_stocks_answer_under_noise():
    hs, io = _caller()
    hs.turn = VA._TURN_OURS
    assert hs._release_turn()
    hs.send(b"x" * 6)
    lead = _band_noise(0.08, 2)
    burst = _burst(VF.control_bursts(CALLER, "500", 1)[1])
    x = np.concatenate([lead, burst, _band_noise(1.0, 3)])
    keyed = _feed(hs, io, x)
    assert keyed is not None and io.msgs[-1].startswith("tx session-turn-request")
    behind = (keyed - len(lead) - len(burst)) / MK.FS
    assert 0.05 <= behind <= 0.2, f"asked {behind:+.2f} s behind stock's burst"


def test_the_release_behind_stocks_ack_is_keyed_at_the_answer_age():
    hs, io = _caller()
    hs.turn = VA._TURN_OURS
    hs._tx_pending = (phy.vara_body(b"x" * 20, CALLER), 1, 4)
    burst = _burst(VF.control_bursts(CALLER, "500", 1)[1], snr_db=6.0)
    x = np.concatenate([_band_noise(0.3, 4), burst, _band_noise(1.0, 5)])
    keyed = _feed(hs, io, x)
    assert keyed is not None and io.msgs[-1].startswith("turn released"), io.msgs[-3:]
    behind = (keyed - int(0.3 * MK.FS) - len(burst)) / MK.FS
    assert -0.03 <= behind - VA._ANSWER_HELD_S <= 0.03 + BLOCK / MK.FS, behind


#: Bench 1 cnl sessions, b2a cable: the end of stock's 0.48 s answer to our
#: release (band envelope), and where the live turn-request started.
OVERLAPS = [("cnl-b2a0.3-1", 24.40, 24.17), ("cnl-b2a0.3-1", 51.28, 51.05),
            ("cnl-b2a0.3-2", 24.29, 24.06), ("cnl-b2a0.3-2", 51.17, 50.94),
            ("cnl-b2a0.45-1", 48.08, 47.96), ("cnl-b2a0.45-1", 76.85, 76.62),
            ("cnl-b2a0.45-2", 33.83, 33.71), ("cnl-b2a0.45-2", 61.16, 60.93),
            ("cnl-b2a0.6-1", 64.00, 63.92), ("cnl-b2a0.6-1", 92.77, 92.58)]


@pytest.mark.skipif(not CABLES.exists(), reason="bench 1 recordings not kept here")
@pytest.mark.parametrize("session, end, live", OVERLAPS)
def test_the_recorded_asks_go_behind_stocks_burst(session, end, live):
    from scipy.io import wavfile
    fs, x = wavfile.read(CABLES / session / "cables" / f"pat-{session}__b2a.wav")
    x = np.asarray(x, float)
    hs, io = _station("500")
    hs.called, hs.high_speed, hs._link_high_speed = "W1AW", True, True
    hs._track_band(x[int((live - 5.4) * fs):int((live - 1.9) * fs)])
    released = live - 0.365
    _Clock.t = released
    hs._key(False)
    hs.turn, hs._released = VA._TURN_PEER, True
    hs._ask_not_before = released + VA._RELEASE_ASK_GAP_S
    _Clock.t = released + 0.05
    hs.send(b"x" * 6)
    _Clock.t = released + 0.1
    for at in range(int(_Clock.t * fs), int((released + 3) * fs), BLOCK):
        _Clock.t += BLOCK / fs
        hs.on_rx_stream(x[at:at + BLOCK])
        if io.sent:
            break
    assert io.sent and 0.05 <= _Clock.t - end <= 0.2, (_Clock.t - end, live - end)


#: Bench 1 with the locator at 0.6 RMS:
#: unanswered turn-requests re-keyed on our idle clock, 0.06-1.13 s into stock's
#: 1.39-1.40 s query or keepalive. (session, live start, stock's end from its PTT,
#: a stretch of band with stock silent for the floor.)
RETRIES = [("loc2-b2a0.6-1", 94.33, 94.65, 84.4), ("loc2-b2a0.6-1", 149.36, 149.71, 142.5),
           ("loc2-b2a0.6-2", 87.11, 87.70, 77.3), ("loc2-b2a0.6-2", 118.94, 120.27, 111.8),
           ("loc2-b2a0.6-2", 142.03, 142.29, 131.8)]


@pytest.mark.skipif(not CABLES.exists(), reason="bench 1 recordings not kept here")
@pytest.mark.parametrize("session, live, end, quiet", RETRIES)
def test_a_retried_ask_waits_out_stocks_query(session, live, end, quiet):
    from scipy.io import wavfile
    fs, x = wavfile.read(CABLES / session / "cables" / f"pat-{session}__b2a.wav")
    x = np.asarray(x, float)
    hs, io = _station("500")
    hs.called, hs.high_speed, hs._link_high_speed = "W1AW", True, True
    hs._track_band(x[int(quiet * fs):int((quiet + 3) * fs)])
    _Clock.t = live - 3.0
    hs._key(False)
    hs.turn, hs._asked, hs._txq = VA._TURN_ASKED, 2, [b"x" * 6]
    ticked = False
    for at in range(int(_Clock.t * fs), int((live + 3) * fs), BLOCK):
        _Clock.t += BLOCK / fs
        hs.on_rx_stream(x[at:at + BLOCK])
        if not ticked and _Clock.t >= live:
            ticked = True
            hs.idle_keepalive()
        if io.sent:
            break
    assert io.sent and 0.05 <= _Clock.t - end <= 0.25, (_Clock.t - end, io.msgs[-2:])


def _quiet_directly(hs):
    """The band reading `_band_quiet_s` keeps incrementally, from the whole envelope."""
    env, hist = hs._em_env, hs._em_hist
    cols = min(VA._LIT_COLS, len(env))
    if cols < 3 or len(hist) < VA._FLOOR_SPAN:
        return None
    lit_at = VA._LIT_RATIO * np.median(hist[-VA._LIT_HIST:])
    lit = np.flatnonzero(np.convolve(env, np.ones(cols) / cols, "valid") > lit_at)
    if not len(lit):
        return None
    k = lit[-1]
    fine = np.flatnonzero(np.convolve(env[k:], np.ones(3) / 3, "valid") > lit_at)
    end = k + (fine[-1] + 2 if len(fine) else cols // 2)
    return -1.0 if end >= len(env) - 1 else (len(env) - 1 - end) * VA._OVER500_COL / MK.FS


def test_the_band_reading_is_the_whole_envelopes_at_every_block():
    hs, _io = _station("500")
    pairs = VF.control_bursts(CALLER, "500", 1)[1]
    x = np.concatenate([_band_noise(2.0, 11)]
                       + [np.concatenate([_burst(pairs, snr_db=snr, seed=12 + i),
                                          _band_noise(0.2 + 0.3 * i, 20 + i)])
                          for i, snr in enumerate((6.0, 0.0, 3.0, -2.0, 9.0))]
                       + [_band_noise(12.0, 30)])
    read, at = [], 0
    for i, size in enumerate(np.resize([960, 1024, 4800, 300, 2048], 400)):
        if at >= len(x):
            break
        if i in (60, 61, 150):
            hs._key(False)                  # an unkey starts the envelope over
        hs._track_band(x[at:at + size])
        at += size
        read.append((hs._band_quiet_s(), _quiet_directly(hs)))
    assert all(a == b for a, b in read)
    assert {type(a) for a, _ in read} >= {type(None), float}
    assert any(a is not None and a >= 0 for a, _ in read) and any(a == -1.0 for a, _ in read)


def test_a_fall_wakes_the_scans_as_it_is_seen_and_at_the_answer_age():
    hs, _io = _caller()
    woken = []
    col = VA._OVER500_COL / MK.FS
    hs._wake_answer_scan = lambda wait: woken.append(
        (len(hs._em_env) - round(hs._band_quiet_s() / col), hs._band_quiet_s()))
    pairs = VF.control_bursts(CALLER, "500", 1)[1]
    x = np.concatenate([_band_noise(0.3, 4), _burst(pairs, snr_db=6.0), _band_noise(1.0, 5)])
    for at in range(0, len(x), BLOCK):
        hs._track_band(x[at:at + BLOCK])
        hs._wake_at_band_fall()
    falls = {fall for fall, _ in woken}
    assert falls and len(woken) == 2 * len(falls), woken
    for fall in falls:
        seen, aged = (quiet for at, quiet in woken if at == fall)
        assert 0 <= seen < VA._ANSWER_HELD_S <= aged < VA._ANSWER_HELD_S + col, woken
