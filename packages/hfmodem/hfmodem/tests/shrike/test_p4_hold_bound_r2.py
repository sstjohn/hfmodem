# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Unit 2: fixed recording evidence followed by explicitly scripted recovery.

1029's 3.985 s is a rounded log duration, not a claimed capture cut. The
scripted window uses that duration and its measured P1 CS raster. The peer
words below replace the continuation; no old local TX is admitted as RX.
"""
import numpy as np
import pytest

from hfmodem.shrike import onair, p4oracle, p4robusttx, p4breakintx
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.tests.shrike.r2_replay import Clocks, FS, Tape
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx


def recovery(word, skipped, base_slot=26):
    tape = Tape('1001-1029')
    assert 'captured 3.985 s' in tape.input_path(tape.contract['log']).read_text()
    grid = onair._MasterGrid(2703, 60000, 8880, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    grid.d_n = 4563  # 1793346 - boundary(29) - 46080
    start = grid.rx_due(base_slot)-round(.275*FS)
    audio = np.zeros(round(3.985*FS), np.float32)
    onset = grid.rx_due(base_slot+skipped)
    wave = p4robusttx.robust_word(word)
    audio[onset-start:onset-start+len(wave)] = wave
    live = Stream(np.zeros(start+len(audio)+2*FS, np.float32))
    live.pos = start+len(audio)
    tx = Tx(live)
    tx.slot = base_slot+4
    tx.tx_end = grid.boundary(base_slot-1)+grid.packet_n
    tx.p4_probe.requested = False
    tx.p4_probe.emissions.clear()
    host = tx.p4_link_host
    host.p4_after_first_ack = True
    tx.sessrx.host = host
    return tx, host, live, grid, start, audio, onset


@pytest.mark.parametrize('word', ['X27', 'Y27'])
@pytest.mark.parametrize('skipped', [0, 1, 2])
def test_r2_regrid_retains_takeover(tmp_path, monkeypatch, word, skipped):
    tape = Tape('0926-1342')
    a, b = tape.contract['trusted_rx_intervals'][-4]
    peer = P4Receiver(FS, admitted=True)
    tape.read(peer, a, b)
    peer.read_breakin(tape.audio[a:b], a, carrier_end=a)
    assert any(o.word == 'Y27' and abs(o.stream_sample-1786811) < FS//100
               for o in peer.observations)
    tape.diverge(b)

    base_slot = 31
    _, _, _, grid, start, audio, onset = recovery(word, skipped, base_slot)
    from hfmodem.tests.shrike.test_p4_regrid_handoff_0929 import recovery_scene
    live, _, tx, host, rx, _, _ = recovery_scene(tmp_path, monkeypatch, None)
    rx.p4 = peer
    assert not peer.emissions  # the RX-only prefix invents no local P4 key
    live.audio = np.zeros(48*FS, np.float32)
    live.limit = len(live.audio)
    live.audio[start:start+len(audio)] = audio
    live.read(start+len(audio)-live.pos)
    tx.tx_end = grid.boundary(base_slot-1)+grid.packet_n
    tx.aim(grid, base_slot)
    receiver = tx.sessrx.p4
    broad = []
    read = receiver.read_breakin
    def measured(audio, origin, **kwargs):
        broad.append(len(audio))
        return read(audio, origin, **kwargs)
    monkeypatch.setattr(receiver, 'read_breakin', measured)
    _, recovered, recovered_origin = onair._regrid(
        live, grid, tx, host, rx, base_slot, audio, start, round(.040*FS))
    assert not broad or max(broad) <= round(.680*FS), 'unbounded recovered word/body read'
    assert tx.p4_probe.handoff_slot == base_slot+skipped
    assert tx.p4_probe.requested and tx.p4_probe.peer_breakin
    assert tx.p4_probe.receiver is receiver and not tx.slots_used
    origin, saved = tx.p4_probe.handoff_audio
    assert origin == start == recovered_origin
    np.testing.assert_array_equal(saved, recovered)
    audio[:] = 0  # the capture buffer is reusable
    tx.p4_hold_audio.retain(audio, start)
    assert np.any(saved) and np.any(tx.p4_hold_audio.audio)
    assert any(o.word == word and abs(o.stream_sample-onset) < round(.010*FS)
               for o in receiver.observations)


@pytest.mark.parametrize('stage_ms', [42.20, 216.75, 577.31])
@pytest.mark.parametrize('jump', ['sample', 'wall'])
def test_r2_hold_deadline(monkeypatch, stage_ms, jump):
    tx, host, live, grid, start, audio, onset = recovery('Y27', 0)
    body = p4breakintx.changeover_packet(b'unit two', 0)
    audio[onset-start:onset-start+len(body)] = body
    render, settle = round(.090*FS), round(tx.settle*FS)
    reserve = render+settle
    tx.cycle_cost_n = lambda: render
    tx.slot = 32
    key = grid.boundary(tx.slot)
    cost = round(stage_ms/1000*FS)
    initial = key-reserve-cost-round(.640*FS)
    clocks = Clocks(sample=initial, wall=initial)
    monkeypatch.setattr(live, 'sample_now', lambda: clocks.sample)
    monkeypatch.setattr(onair.time, 'monotonic', lambda: clocks.wall/FS)
    calls, budgets = [], []
    record, scan = p4oracle._word_record, p4oracle.word_scan

    def expensive(*args, **kwargs):
        result = record(*args, **kwargs)
        if not calls:
            calls.append(cost)
            setattr(clocks, jump, getattr(clocks, jump)+cost)
        return result

    def budgeted_scan(*args, **kwargs):
        budgets.append(kwargs.get('work'))
        return scan(*args, **kwargs)

    monkeypatch.setattr(p4oracle, '_word_record', expensive)
    monkeypatch.setattr(p4oracle, 'word_scan', budgeted_scan)
    heard = onair._p4_hold_word(tx, host, live, grid, audio, start, tx.tx_end)
    assert calls == [cost]
    assert getattr(clocks, jump) == initial+cost
    assert getattr(clocks, 'wall' if jump == 'sample' else 'sample') == initial
    assert heard == 26 and tx.p4_probe.peer_breakin and not tx.slots_used
    receiver = tx.sessrx.p4
    assert not any(o.frame is not None for o in receiver.observations)
    work = budgets[0]
    assert work.sample_deadline == key-reserve
    assert work.wall_deadline == pytest.approx((key-reserve)/FS)
    assert work.remaining() == pytest.approx(.640)
    assert 'window' in work.deferred

    setattr(clocks, jump, getattr(clocks, jump)+FS)
    outstanding = reserve-clocks.lead
    next_key = clocks.feasible_due(key, grid.cycle_n, outstanding=outstanding)
    assert next_key == key+grid.cycle_n
    assert min(key-clocks.sample, key-clocks.wall) < reserve
    assert min(next_key-clocks.sample, next_key-clocks.wall) >= reserve
    retained = tx.p4_hold_audio
    audio[:] = 0
    retained.retain(audio, start)
    monkeypatch.setattr(p4oracle, '_word_record', record)
    body_calls = []
    decode = p4oracle.body_after

    def decode_cost(*args, **kwargs):
        frame = decode(*args, **kwargs)
        body_calls.append(frame)
        clocks.work(read=round(.57731*FS))
        return frame

    monkeypatch.setattr(p4oracle, 'body_after', decode_cost)
    tx.slot = (next_key-grid.anchor)//grid.slot_n
    before = len(receiver.registrations)
    assert onair._p4_hold_word(tx, host, live, grid, retained.audio,
                              retained.origin, tx.tx_end) == 26
    frames = [r for r in receiver.registrations[before:] if r.observation.frame is not None]
    assert len(frames) == 1 and frames[0].observation.frame.data.startswith(b'unit two')
    assert len(body_calls) == 1 and body_calls[0] is not None
    onair._p4_hold_word(tx, host, live, grid, retained.audio, retained.origin, tx.tx_end)
    assert len(body_calls) == 1
    clocks.work(render=render)
    assert min(next_key-clocks.sample, next_key-clocks.wall) >= settle


def test_r2_regrid_never_calls_the_whole_hold_reader(tmp_path, monkeypatch):
    from hfmodem.tests.shrike.test_p4_regrid_handoff_0929 import recovery_scene
    live, grid, tx, host, rx, audio, _ = recovery_scene(tmp_path, monkeypatch, 'bare')
    def forbidden(*args, **kwargs):
        pytest.fail('_regrid invoked the unbounded hold reader')
    monkeypatch.setattr(onair, '_p4_hold_word', forbidden)
    onair._regrid(live, grid, tx, host, rx, 1, audio, 0, round(.040*FS))
    assert tx.p4_probe.requested and not live.emissions


def test_r2_retention_keeps_gaps_and_expires_work(monkeypatch):
    from hfmodem.shrike.p4holdwork import HoldAudio
    from hfmodem.shrike.p4budget import WorkBudget
    tx, _, live, grid, start, audio, onset = recovery('Y27', 0)
    retained = HoldAudio(FS)
    retained.retain(audio[:FS], start)
    retained.retain(audio[2*FS:], start+2*FS)
    assert retained.gaps == [(start+FS, start+2*FS)]
    assert not retained.audio[FS:2*FS].any()
    now = [live.pos, 0.]
    available = WorkBudget(FS, now[0]+2*FS, lambda: now[0], lambda: now[1])
    tx.sessrx.p4.read_hold_word(audio[:onset-start+round(.215*FS)], start,
        due=onset, carrier_end=tx.tx_end, work=available)
    budget = WorkBudget(FS, now[0], lambda: now[0], lambda: now[1])
    retained.read(tx.sessrx.p4, grid, tx.tx_end, budget, lambda: now[1])
    assert retained.pending
    expiry = retained.wall_expires
    now[1] = expiry
    retained.read(tx.sessrx.p4, grid, tx.tx_end, budget, lambda: now[1])
    assert not retained.pending
    retained.retain(np.ones(8*FS), start+4*FS)
    assert len(retained.audio) <= 8*FS


def test_r2_pending_receive_work_refuses_a_p1_callback(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hfmodem.tests.shrike.test_p4_regrid_handoff_0929 import recovery_scene
    live, grid, tx, host, _, _, _ = recovery_scene(tmp_path, monkeypatch, None)
    tx.aim(grid, 3)
    tx.host = host
    tx.p4_hold_audio = SimpleNamespace(pending=True)
    tx._tx(np.zeros(48000, np.float32), 'P1 packet callback')
    assert tx.refused and not tx.slots_used and not live.emissions


@pytest.mark.parametrize('word', ['X27', 'Y27'])
def test_r2_partial_takeover_reserves_then_confirms(word):
    tx, host, live, grid, start, audio, onset = recovery(word, 0)
    live.pos = onset+round(.215*FS)
    tx.slot = 28
    prefix = audio[:live.pos-start]
    assert onair._p4_hold_word(tx, host, live, grid, prefix, start, tx.tx_end) is None
    assert tx.p4_hold_audio.pending and not tx.p4_probe.requested
    assert any(o.word == word and o.provisional for o in tx.sessrx.p4.observations)
    live.pos = onset+round(.275*FS)
    heard = onair._p4_hold_word(tx, host, live, grid, audio[:live.pos-start], start, tx.tx_end)
    assert heard == 26 and tx.p4_probe.requested and not tx.slots_used
    assert any(o.word == word and not o.provisional for o in tx.sessrx.p4.observations)


@pytest.mark.parametrize('tail_gap', [False, True])
def test_r2_censored_word_tail_preserves_capture_gaps(tail_gap):
    onset, carrier_end = FS, round(1.020*FS)
    audio = np.zeros(2*FS, np.float32)
    wave = p4robusttx.robust_word('Y27')
    audio[onset:onset+len(wave)] = wave
    live = Stream(audio)
    live.pos = round(1.30*FS)
    tx = Tx(live)
    tx.slot, tx.tx_end, tx.tx_audio_start = 2, carrier_end, 0
    tx.p4_probe.requested = False
    tx.p4_probe.emissions.clear()
    tx.p4_link_host.p4_after_first_ack = True
    if tail_gap:
        tx.sessrx.p4.note_gap(onset+round(.080*FS), onset+round(.095*FS))
    grid = onair._MasterGrid(0, 60000, 8880, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    heard = onair._p4_hold_word(tx, tx.p4_link_host, live, grid,
                               audio[:live.pos], 0, carrier_end)
    assert (heard is not None) == (not tail_gap)
    if not tail_gap:
        assert tx.p4_probe.peer_breakin
        assert (0, carrier_end) in tx.p4_hold_audio.gaps


@pytest.mark.parametrize('jump', ['sample', 'wall'])
def test_r2_expired_candidate_preserves_unread_recorded_takeover(monkeypatch, jump):
    """Conditional fixed-schedule RX; no claim about a peer after changed keys."""
    from hfmodem.shrike.p4receiver import Registered
    tape = Tape('1001-0344')
    a, c = (row['audio_end'] for row in tape.contract['emissions'][:2])
    first_onset, second_onset = 707957, 767965
    d = second_onset+round(.275*FS)
    cut = first_onset+round(.215*FS)
    control = P4Receiver(FS, admitted=True)
    tape.read(control, a, cut, needed=(a, cut))
    tape.read(control, c, d, needed=(c, d))
    live = Stream(tape.audio)
    live.pos = cut
    clocks = Clocks(sample=cut)
    monkeypatch.setattr(live, 'sample_now', lambda: clocks.sample)
    monkeypatch.setattr(onair.time, 'monotonic', lambda: clocks.wall/FS)
    tx = Tx(live)
    tx.slot, tx.tx_end = 13, a
    tx.p4_probe.requested = False
    tx.p4_probe.emissions.clear()
    host = tx.p4_link_host
    host.p4_after_first_ack = True
    grid = onair._MasterGrid(1877, 60000, 8880, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    assert onair._p4_hold_word(tx, host, live, grid, tape.audio[a:cut], a, a) is None
    retained = tx.p4_hold_audio
    receiver = tx.sessrx.p4
    assert retained.pending and not tx.p4_probe.requested
    candidate, = [o for o in receiver.observations if o.word == 'X27' and o.provisional]
    assert abs(candidate.stream_sample-first_onset) < FS//100
    retained.retain(tape.audio[c:d], c)
    live.pos = d
    clocks.sample = d
    if jump == 'sample':
        clocks.sample = retained.expires
    else:
        clocks.wall = round(retained.wall_expires*FS)
    tx.slot = 16
    heard = onair._p4_hold_word(tx, host, live, grid, tape.audio[c:d], c, a)
    assert heard == 12 and tx.p4_probe.peer_breakin and not tx.slots_used
    assert any(r.outcome == Registered.EXPIRED and r.observation == candidate
               for r in receiver.registrations)
    assert not retained.pending
    assert any(o.word == 'Y27' and not o.provisional
               and abs(o.stream_sample-second_onset) < FS//100
               for o in receiver.observations)
    assert (cut, c) in retained.gaps
