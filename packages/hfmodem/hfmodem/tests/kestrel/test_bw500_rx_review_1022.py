# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""BW500 receiver review, 2026-10-22: alternate epochs, the decode budget,
the carry's edge, pooled noise and the ident screen."""
import numpy as np
import pytest

from hfmodem.kestrel.rx import bw500_locator as loc, varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA, vara_mfsk as MK
from .test_bw500_stock_ident_1021 import CALLED, CALLER, _IO, stock_ident

ONE = (rx.NSYM + 9) * rx.H


def _junk_pair_over(start):
    candidate = loc.Candidate(4, start, 1., 0, start,
                              (loc.Epoch(start, 1., 0), loc.Epoch(start + 3000, 1., 0)))
    return loc.Over(candidate, next_check=start + ONE + rx._PAD)


def test_a_distant_alternate_epoch_is_read_on_fresh_audio_at_its_own_onsets():
    """The failed epoch's onsets, outside the alternate's own lock search, fixed
    its alignment, and the alternate read the snapshot the failed epoch had
    taken, cut short of it."""
    payload = bytes(range(43))
    burst = tx.synth_burst(tx.build_frame(payload, 0x02))
    x = np.zeros(48000 + len(burst) + 5 * 48000)
    x[48000:48000 + len(burst)] += burst
    x += np.random.default_rng(3).normal(0, .02 * np.std(burst), len(x))
    found = loc.Receiver().feed(x, eof=True)
    start = next(o.candidate.start for o in found if o.result.complete)
    receiver = loc.Receiver()
    late = start - 6000
    candidate = loc.Candidate(4, late, 1., 0, start,
                              (loc.Epoch(late, 1., 0), loc.Epoch(start, 1., 0)))
    receiver.pending.append(loc.Over(candidate, next_check=late + ONE + rx._PAD))
    out = []
    for j in range(0, len(x), 1024):
        out += receiver.feed(x[j:j + 1024], batches=[])
        if out:
            break
    assert [f.frame_bytes[:43] for o in out for f in o.frames if f.crc_ok] == [payload]


def test_an_over_at_the_carrys_edge_is_retired_not_raised():
    receiver = loc.Receiver()
    over = loc.Over(loc.Candidate(4, 0, 1., 0, 0), next_check=1 << 40)
    receiver.pending.append(over)
    for _ in range(loc.CARRY // 1024 + 16):
        assert receiver.feed(np.zeros(1024), batches=[]) == []
    assert receiver.pending == [] and receiver.expired == 1


@pytest.mark.parametrize('size', [2048, 4096])
def test_failing_pair_candidates_finish_inside_the_carry_at_large_callbacks(size):
    """Two junk pair locks at one step a callback starved past the ~11.8 s carry
    at 4096-sample callbacks, which reset the whole receiver."""
    x = np.random.default_rng(5).normal(0, .1, 48000 * 16)
    receiver = loc.Receiver()
    receiver.pending = [_junk_pair_over(s) for s in (48000, 60000)]
    done = []
    for j in range(0, len(x), size):
        done += receiver.feed(x[j:j + size], may_pair=True)
        if len(done) == 2:
            break
    assert sorted(o.candidate.start for o in done) == [51000, 63000]
    assert receiver.expired == 0


@pytest.mark.parametrize('size,steps', [(1024, 1), (2048, 2), (4096, 4), (5000, 5)])
def test_the_station_budgets_a_step_per_1024_samples(monkeypatch, size, steps):
    hs = VA.VaraStationHandshake([CALLER], _IO(), bw="500")
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    budget = []
    original = loc.Receiver.feed
    def feed(self, samples, *, _budget=None, **kw):
        budget[:] = budget or [_budget[0]]
        return original(self, samples, _budget=_budget, **kw)
    monkeypatch.setattr(loc.Receiver, 'feed', feed)
    hs._feed_shared_front(np.zeros(size))
    assert budget == [steps]


@pytest.mark.parametrize('level', [1, 2, 3])
def test_burst_decoders_read_without_pooled_noise(monkeypatch, level):
    """Outside the locator no pair partner checks a pooled read."""
    modes = []
    original = rx._l3_llr
    def llr(*args, pooled):
        modes.append(pooled)
        return original(*args, pooled=pooled)
    monkeypatch.setattr(rx, '_l3_llr', llr)
    r = rx.INDEX_RECORDS[level]
    n = np.random.default_rng(level).normal(0, .1, 24000 + 2 * (r.lead + r.ncols) * r.dw)
    assert not rx.decode_l3_burst(n, 24000, len(n), level).crc_ok
    assert not any(f.crc_ok for f in rx.decode_burst_frames(n, 24000, len(n), level))
    assert modes and not any(modes)


def test_an_over_keyed_right_behind_the_ident_is_read(monkeypatch):
    """The ident screen reads the 6 s behind the stream, which can hold a whole
    3.25 s ident ahead of an over's start only within 2.75 s of it, and no over
    is decoded before its own 4.3 s are in."""
    from . import test_bw500_stock_ident_1021 as ident
    monkeypatch.setattr(VA, "time", ident._Clock)
    io = _IO()
    io.host = []
    io.data = io.host.append
    hs = VA.VaraStationHandshake([CALLER], io, bw="500")
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    payload = bytes(range(43))
    over = tx.synth_burst(tx.build_frame(payload, 0x02))
    x = np.r_[np.zeros(int(.6 * MK.FS)), stock_ident(), .33 * over / np.max(abs(over)),
              np.zeros(3 * MK.FS)]
    x += .03 * np.random.default_rng(1).normal(size=len(x))
    for at in range(0, len(x), 1024):
        ident._Clock.t += 1024 / MK.FS
        hs.on_rx_stream(x[at:at + 1024])
    assert io.host == [payload], io.msgs
    assert not any("CW ident" in m for m in io.msgs), io.msgs
