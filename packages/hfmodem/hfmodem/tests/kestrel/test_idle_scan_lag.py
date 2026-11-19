# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A peer idle named before its last symbols are read is read again, not dropped.

The hold behind a BW500 idle counts the scan's own decode lag, so a slow host
reaches it while the tail is still arriving and the fit cannot be exact yet.
K5FIT's over-idle on the 2026-09-24 80 m arm keyed whenever the lag stayed under
about 81 ms and went silent at 82 ms, its idle 0.105 s old against a 0.25 s
limit. Nothing here opens a device or keys a radio.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.host import station_runtime as SR
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests import evidence

from . import test_reack_over as reack
from .test_data_over_gate import _HostIO
from .test_delivery_lag import Capture, transport

ARMS = evidence.WORKING / "vara-level-air-0915" / "arms"
BW500 = "vara-current-k5fit-80m-3584200-500-pounce-20260924T090430Z"
BW2300 = "vara-current-k5fit-40m-7102300-2300-pounce-20260924T120618Z"
PEER = "K5FIT"
AGAIN = "answering again for over #0: continue"


class LaggedClock:
    """Audio arrives in real time, and a scan reads the clock ``lag`` past
    the arrival of the newest sample it holds. A callback's audio reaches the
    scans a block at a time, ``_callback_lag`` short of the callback's end."""

    def __init__(self):
        self.t, self.lag, self.scanning, self.arrived = 1000.0, 0.0, None, False

    def monotonic(self):
        hs = self.scanning
        if hs is None:
            return self.t + self.lag
        if not self.arrived:
            self.arrived = True
            return self.t
        return self.t - getattr(hs, "_callback_lag", 0) / MK.FS + self.lag


@pytest.fixture
def clock(monkeypatch):
    c = LaggedClock()
    monkeypatch.setattr(VA, "time", SimpleNamespace(monotonic=c.monotonic, time=c.monotonic))
    original = VA.VaraStationHandshake.on_rx_stream

    def receive(hs, samples, *args, **kwargs):
        if c.scanning is not None:
            return original(hs, samples, *args, **kwargs)
        c.scanning, c.arrived = hs, False
        try:
            return original(hs, samples, *args, **kwargs)
        finally:
            c.scanning = None
    monkeypatch.setattr(VA.VaraStationHandshake, "on_rx_stream", receive)
    return c


def owing_an_over(bw="500"):
    io = _HostIO(b"")
    hs = VA.VaraStationHandshake([reack.MYCALL], io, bw=bw)
    hs.role, hs.called, hs.caller = "initiator", PEER, reack.MYCALL
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    io.hs = hs
    hs._link_high_speed = False
    hs._peer_over_state = None
    hs._deliver([phy.vara_body(b"A" * 89, hs.caller, tail=0x81)])
    hs._key_over_answer(last=False, owes_release=False)
    assert hs._answer_owed == VA._OWED_OVER, io.msgs
    return hs, io


def replay(clock, audio, lag, bw="500"):
    """Feed ``audio`` through the station transport, recording each key's
    idle hold and freshness age, and each rescan's freshness age."""
    clock.lag = 0.0
    hs, io = owing_an_over(bw)
    keys, rescans, tx, wake = [], [], io.tx, hs._wake_answer_scan

    def fresh():
        return hs._idle_age + hs._stream_trail()

    def timed_tx(samples):
        keys.append((hs._idle_held, fresh()))
        tx(samples)

    def counted(wait):
        if wait == MK.HOP // 2:
            rescans.append(fresh())
        wake(wait)
    io.tx, hs._wake_answer_scan = timed_tx, counted
    clock.lag = lag
    t = transport(Capture(int(lag * MK.FS)))
    for i in range(0, len(audio), SR.CHUNK):
        chunk = audio[i:i + SR.CHUNK]
        clock.t += len(chunk) / MK.FS
        t.feed(hs._input_clock, chunk)
        while (burst := t.next_rx_burst(.01, hs)) is not None:
            hs.on_rx_audio(burst, sample_start=t.burst_start, backlog=t.burst_backlog)
    return io, keys, rescans


def tap(arm, start):
    found = sorted((ARMS / arm).glob("tap.*"))
    if not found:
        pytest.skip(f"{arm} tap is not in this checkout's working record")
    return wav.read(found[0], MK.FS)[start * MK.FS:(start + 6) * MK.FS]


def k5fit():
    return tap(BW500, 276)


def stale(io):
    return [m for m in io.msgs if "nothing keyed" in m]


def unanswered(io):
    """Idles named with neither a key nor a stale line before the next."""
    out = []
    for m in io.msgs:
        if "is still transmitting, named" in m:
            out.append(m)
        elif out and (m == AGAIN or "nothing keyed" in m):
            out.pop()
    return out


# Under this clock the 276 s idle reaches the hold with about 18 ms of its tail
# in from 0.087 s of lag, and went unkeyed and unlogged until the rescan.
@pytest.mark.parametrize("lag", [.09, .105])
def test_an_idle_named_ahead_of_its_tail_under_decode_lag_is_answered(clock, lag):
    io, keys, rescans = replay(clock, k5fit(), lag)
    assert io.msgs.count(AGAIN) == 1, io.msgs
    (held, age), = keys
    assert VA._ANSWER_HELD_S <= held and age <= VA._IDLE_ANSWER_S, keys
    assert rescans and max(rescans) <= VA._IDLE_ANSWER_S, rescans
    assert not unanswered(io) and not stale(io), io.msgs


def test_an_idle_rescanned_past_the_gap_before_it_fits_is_stale_not_keyed(clock):
    io, keys, rescans = replay(clock, k5fit(), .115)
    assert not keys and AGAIN not in io.msgs, io.msgs
    assert len(stale(io)) == 1 and "over-idle" in stale(io)[0], io.msgs
    assert rescans and max(rescans) <= VA._IDLE_ANSWER_S, rescans


def test_a_wide_idle_is_rescanned_to_its_own_gap_and_keyed_inside_it(clock):
    """Off the BW500 initiator the gap is one block, not the 0.25 s a BW500
    responder's poll allows."""
    io, keys, rescans = replay(clock, tap(BW2300, 213), .25, bw="2300")
    (held, age), = keys
    assert max(rescans) > VA._IDLE_ANSWER_S and age > VA._IDLE_ANSWER_S, (rescans, keys)
    assert max(rescans) <= VA._GRANT_FRESH_S and age <= VA._GRANT_FRESH_S, (rescans, keys)
    named = [i for i, m in enumerate(io.msgs) if "is still transmitting, named" in m]
    assert len(named) == 1 and io.msgs[named[0] + 1].startswith("tx "), io.msgs
    assert not stale(io), io.msgs


def test_a_wide_idle_rescanned_past_its_gap_is_stale_not_keyed(clock):
    io, keys, rescans = replay(clock, tap(BW2300, 213), .3, bw="2300")
    assert not keys, io.msgs
    assert max(rescans) > VA._IDLE_ANSWER_S and max(rescans) <= VA._GRANT_FRESH_S, rescans
    assert len(stale(io)) == 1 and "over-idle" in stale(io)[0], io.msgs


def test_an_idle_on_a_healthy_host_keys_where_it_always_did(clock):
    io, keys, rescans = replay(clock, k5fit(), .03)
    assert io.msgs.count(AGAIN) == 1 and not rescans, io.msgs
    # The hold and age 5a8437084 keyed this idle at under the same clock.
    (held, age), = keys
    assert held == pytest.approx(.0833, abs=5e-4) and age == pytest.approx(.1133, abs=5e-4)


def corrupted_idle():
    """K5FIT's idle with its last two symbols another station's, so the loose
    reader names it and the exact fit never takes it."""
    kind = VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE, "500")
    x = MK.synth_burst(PEER, kind)
    tail = 2 * MK.HOP
    x[-tail:] = MK.synth_burst("N0CALL", kind)[-tail:]
    return x


@pytest.mark.parametrize("lag", [0.0, .03, .09, .2])
def test_an_idle_that_never_fits_exactly_ends_stale_and_is_never_keyed(clock, lag):
    audio = np.r_[np.zeros(MK.FS // 2), corrupted_idle(), np.zeros(2 * MK.FS)]
    audio = audio + np.random.default_rng(3).normal(0, 1e-3, len(audio))
    io, keys, rescans = replay(clock, audio, lag)
    assert not keys and AGAIN not in io.msgs, io.msgs
    assert len(stale(io)) == 1 and "over-idle" in stale(io)[0], io.msgs
    # Short of the limit at its first naming, the frame is rescanned up to it.
    assert bool(rescans) == (lag < .2), rescans
    assert all(age <= VA._IDLE_ANSWER_S for age in rescans), rescans
