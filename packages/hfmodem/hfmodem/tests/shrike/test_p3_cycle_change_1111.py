# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import re

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3frame, placement, rxfront
from hfmodem.core import wav
from hfmodem.tests import evidence
from hfmodem.tests.shrike.onair_support import _Rig as _Rig, _Session as _Session
from hfmodem.tests.shrike.p3_cycle_support import (
    NativeCycleClock as NativeCycleClock,
    _arm,
    emit,
    grant,
    packet,
    put_packet,
    scene,
)
from hfmodem.tests.shrike.p3_reply_support import mail_link, owe


@pytest.mark.parametrize('old_long, expected', [
    (True, [(3, 1), (6, 1), (9, 1), (12, 1)]),
    (False, [(3, 1), (6, 1), (9, 1), (12, 1)]),
])
def test_pending_silence_preserves_old_packet_control_boundaries(tmp_path, monkeypatch,
                                                old_long, expected):
    s, tx, g, clock = _arm(tmp_path, np.zeros(20 * onair.FS), old_long=old_long)
    s.host.arq.cfg.speed_up = 'hold'
    monkeypatch.setattr(onair, '_scan_frame', lambda *a, **k: None)
    emitted = []
    slot = 1
    for _ in range(4):
        tx.aim(g, slot)
        slot, _, _, _ = onair._p3_transition_window(
            clock, g, tx, s.host, s.rx, slot, np.zeros(0), clock.pos, 1920)
        tx.aim(g, slot)
        tx._send_p3_control(arq.CS_REQUEST)
        emitted.append((tx.slot, tx._p3_emitted_cs))
        slot = g.next_slot(slot)
    assert emitted == expected
    assert s.host.arq.cycle_command_emitted


@pytest.mark.parametrize('target', [False, True])
def test_grant_arms_first_commanded_boundary(tmp_path, target):
    s, tx, live, g, _, index = mail_link(tmp_path)
    a = s.host.arq
    a.cfg.speed_up = 'hold'
    a._cycle_long = not target
    g.regear(not target)
    a._cycle_request, a._cycle_command_emitted = target, False
    tx.aim(g, index + 1)
    owe(s, tx, arq.CS_CYCLE_TOG)
    tx.emit_pending_cs()
    assert a.cycle_command_emitted
    k = tx.slot
    slot, _ = onair._regear_next_slot(g, g.next_slot(k), a.cycle_long)
    assert slot == k + (3 if target else 1)
    tx.aim(g, slot)
    tx.arm_p3_reply()
    assert tx._p3_armed is not None
    assert tx._p3_armed.slot == slot


@pytest.mark.parametrize('old_long, response, expected', [
    (False, 'accept', [(0, 5), (3, 1), (6, 1), (9, 1)]),
    (True, 'accept', [(0, 5), (1, 1), (2, 1), (3, 1)]),
    (False, 'duplicate', [(0, 5), (1, 0), (2, 0), (3, 0)]),
    (True, 'duplicate', [(0, 5), (3, 0), (6, 0), (9, 0)]),
    (False, 'new', [(0, 5), (1, 5), (4, 1), (7, 1)]),
    (True, 'new', [(0, 5), (3, 5), (6, 1), (9, 1)]),
])
def test_real_arq_cycle_outcomes(tmp_path, monkeypatch, old_long, response, expected):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=old_long)
    emitted = [grant(s, tx, g, live, old_long)]
    long = not old_long if response == 'accept' else old_long
    seq = 0 if response == 'duplicate' else 1
    status = seq | (0x20 if not old_long else 0)
    put_packet(s, g, live, 2, long, status)
    target = g.next_slot(tx.slot)
    tx.aim(g, target)
    tx.arm_p3_reply()
    onair._p3_transition_window(
        live, g, tx, s.host, s.rx, target, np.zeros(0), live.pos, 1920)
    emitted.append(emit(s, tx, g, live))
    for _ in range(2):
        slot = g.next_slot(tx.slot)
        tx.aim(g, slot)
        onair._p3_transition_window(
            live, g, tx, s.host, s.rx, slot, np.zeros(0), live.pos, 1920)
        emitted.append(emit(s, tx, g, live))
    assert emitted == expected
    assert tx.slots_used == [slot for slot, _ in expected]
    assert s.host.arq.cycle_command_emitted == (response == 'new')


def test_pending_short_interval_listens_through_long_repeat(tmp_path, monkeypatch):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=True)
    assert grant(s, tx, g, live, True) == (0, arq.CS_CYCLE_TOG)
    row = put_packet(s, g, live, 2, True, 0)
    tx.aim(g, 1)
    tx.arm_p3_reply()
    slot, _, _, _ = onair._p3_transition_window(
        live, g, tx, s.host, s.rx, 1, np.zeros(0), live.pos, 1920)
    assert slot == 3
    assert abs(s.rx._p3_delivered_at - row) <= 120
    assert s.packets[-1].packet == (2, 0, b'X', True)
    assert not s.host.arq.cycle_command_emitted
    assert not g._p3_interval.pending and g._p3_interval.long
    assert emit(s, tx, g, live) == (3, arq.CS_ACK)
    assert tx.slots_used == [0, 3]
    assert [at for at, _ in live.submitted] == [g.boundary(0), g.boundary(3)]


@pytest.mark.parametrize('level', [1, 2, 3, 4, 5, 6])
def test_short_header_keeps_first_pending_reply(tmp_path, monkeypatch, level):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=True)
    grant(s, tx, g, live, True)
    put_packet(s, g, live, level, False, 1, header_only=True)
    tx.aim(g, 1)
    tx.arm_p3_reply()
    slot, _, _, _ = onair._p3_transition_window(
        live, g, tx, s.host, s.rx, 1, np.zeros(0), live.pos, 1920)
    assert slot == 1
    assert s.host.arq.cycle_command_emitted and g._p3_interval.pending
    assert emit(s, tx, g, live) == (1, arq.CS_ACK)
    assert tx.slots_used == [0, 1]


@pytest.mark.parametrize('old_long, expected', [
    (False, [(0, 5), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0)]),
    (True, [(0, 5), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0)]),
])
def test_silent_peer_does_not_expire_pending_interval(tmp_path, monkeypatch,
                                                     old_long, expected):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=old_long)
    emitted = [grant(s, tx, g, live, old_long)]
    for _ in range(5):
        slot = g.next_slot(tx.slot)
        tx.aim(g, slot)
        onair._p3_transition_window(
            live, g, tx, s.host, s.rx, slot, np.zeros(0), live.pos, 1920)
        emitted.append(emit(s, tx, g, live))
    assert emitted == expected
    assert s.host.arq.cycle_command_emitted
    assert g._p3_command_slot == 0


@pytest.mark.parametrize('old_long, confirmation, delay, expected', [
    (False, True, 0, [(0, 5), (3, 1), (6, 1), (9, 1)]),
    (False, True, 1, [(0, 5), (6, 1), (9, 1), (12, 1)]),
    (False, True, 3, [(0, 5), (9, 1), (12, 1), (15, 1)]),
    (False, False, 0, [(0, 5), (1, 0), (2, 0), (3, 0)]),
    (False, False, 1, [(0, 5), (3, 0), (4, 0), (5, 0)]),
    (False, False, 3, [(0, 5), (5, 0), (6, 0), (7, 0)]),
    (True, True, 0, [(0, 5), (1, 1), (2, 1), (3, 1)]),
    (True, True, 1, [(0, 5), (3, 1), (4, 1), (5, 1)]),
    (True, True, 3, [(0, 5), (5, 1), (6, 1), (7, 1)]),
    (True, False, 0, [(0, 5), (3, 0), (6, 0), (9, 0)]),
    (True, False, 1, [(0, 5), (6, 0), (9, 0), (12, 0)]),
    (True, False, 3, [(0, 5), (9, 0), (12, 0), (15, 0)]),
])
def test_crc_decision_crosses_boundaries_without_retroactive_emissions(
        tmp_path, monkeypatch, old_long, confirmation, delay, expected):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=old_long)
    emitted = [grant(s, tx, g, live, old_long)]
    long = not old_long if confirmation else old_long
    end = 3 if long else 1
    decision = g.boundary(end + delay) + (100 if delay else -5000)
    packet(s, tx, g, live, end_slot=end, long=long,
           status=(1 if confirmation else 0) | (0 if old_long else 0x20),
           decision=decision)
    emitted.append(emit(s, tx, g, live))
    for _ in range(2):
        emitted.append(emit(s, tx, g, live, g.next_slot(tx.slot)))
    assert emitted == expected
    assert tx.slots_used == [slot for slot, _ in expected]


@pytest.mark.parametrize('allow_long', [False, True])
@pytest.mark.parametrize('asks_long, expected', [
    (True, [(3, 0), (6, 0), (9, 0)]),
    (False, [(3, 5), (4, 0), (5, 0)]),
])
def test_unsolicited_long_reconciles_independently_of_preference(
        tmp_path, monkeypatch, allow_long, asks_long, expected):
    s, tx, g, live = scene(tmp_path, monkeypatch, allow_long=allow_long)
    packet(s, tx, g, live, end_slot=3, long=True, status=0x20 if asks_long else 0,
           decision=g.boundary(3) - 1000)
    # Cnew admits this boundary even though nominal PTT-start is past.
    assert live.armed_at == g.boundary(3)
    emitted = [emit(s, tx, g, live)]
    for _ in range(2):
        emitted.append(emit(s, tx, g, live, g.next_slot(tx.slot)))
    assert emitted == expected
    assert tx.slots_used == [slot for slot, _ in expected]


@pytest.mark.parametrize('placement_mode', ['audio-start', 'pulse-center'])
@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_reconciliation_cnew_uses_native_admission_not_nominal_ptt(
        tmp_path, monkeypatch, delta, placement_mode):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    tx.p3_control_placement = placement_mode
    tx.prekey_cost_n = 0
    lead = tx._p3_control_lead(0) if placement_mode == 'pulse-center' else 0
    packet(s, tx, g, live, end_slot=3, long=True, status=0x20,
           decision=g.boundary(3) - lead - live.key_notice + delta)
    expected = [(3, 0), (6, 0)] if delta <= 0 else [(6, 0), (9, 0)]
    emitted = [emit(s, tx, g, live)]
    emitted.append(emit(s, tx, g, live, g.next_slot(tx.slot)))
    assert emitted == expected


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_cnew_reserves_preparation_before_native_queue(tmp_path, monkeypatch, delta):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    tx.prekey_cost_n = 240
    live.spend(g.boundary(3) - live.key_notice - 240 + delta)
    render = tx._keyed_audio

    def charged(*args, **kwargs):
        live.spend(240)
        return render(*args, **kwargs)

    monkeypatch.setattr(tx, '_keyed_audio', charged)
    packet(s, tx, g, live, end_slot=3, long=True, status=0x20)
    assert live.armed_at == g.boundary(3 if delta <= 0 else 6)
    assert emit(s, tx, g, live) == (3 if delta <= 0 else 6, 0)


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_crc_replaces_committed_word_after_collection_close(
        tmp_path, monkeypatch, delta):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    grant(s, tx, g, live, False)
    tx.aim(g, 3)
    tx.arm_p3_reply()
    assert tx._p3_armed.cs == 0
    decision = g.boundary(3) - live.key_notice + delta
    assert decision > onair._p3_decode_deadline(live, g.boundary(3), 1920)
    live.spend(decision - live.now)
    assert not live.disarm()
    packet(s, tx, g, live, end_slot=3, long=True, status=1)
    expected = [(3, 1), (6, 1)] if delta <= 0 else [(3, 0), (6, 1)]
    emitted = [emit(s, tx, g, live)]
    emitted.append(emit(s, tx, g, live, g.next_slot(tx.slot)))
    assert emitted == expected


@pytest.mark.parametrize('committed, expected', [
    (False, [(3, 1), (6, 1)]),
    (True, [(2, 1), (3, 1), (6, 1)]),
])
def test_reconciliation_preserves_only_committed_off_interval_bursts(
        tmp_path, monkeypatch, committed, expected):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    tx.aim(g, 2)
    tx.arm_p3_reply()
    live.spend(g.boundary(2) - (1000 if committed else 3000))
    # A delayed earlier long CRC sets residue zero while slot two is armed.
    packet(s, tx, g, live, end_slot=0, long=True, status=0x21)
    emitted = []
    if committed:
        emitted.append((tx.slots_used[-1], tx._p3_emitted_cs))
    emitted.append(emit(s, tx, g, live))
    emitted.append(emit(s, tx, g, live, g.next_slot(tx.slot)))
    assert emitted == expected
    assert tx.slots_used == [slot for slot, _ in expected]


@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('admitted', [False, True])
def test_first_long_packet_uses_bounded_prekey_reader(
        tmp_path, monkeypatch, level, admitted):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    grant(s, tx, g, live, False)
    first = g._p3_command_row0
    waveform = placement.link_packet(level, b'Y', 0x21, long_cycle=True)
    offset = first - p3frame.DATA_OFFSET * rxfront.SPS - (
        placement.protocol_config().pulse().size - 1) // 2
    live.audio[offset:offset + waveform.size] = waveform
    s.rx.sync.packet_level = level
    s.rx._p3_clock_role = arq.IRS
    tx.aim(g, 3)
    tx.arm_p3_reply()
    calls = []
    reader = getattr(s.rx.sync, f'sl{level}_packet_at')

    def read(audio, at, *, long_cycle):
        calls.append((at, long_cycle))
        live.spend(round((.003 if level == 1 else .007) * onair.FS))
        return reader(audio, at, long_cycle=long_cycle)

    monkeypatch.setattr(s.rx.sync, f'sl{level}_packet_at', read)
    monkeypatch.setattr(s.rx, '_read_p3_packet',
                        lambda *a, **k: pytest.fail('unbounded packet reader'))
    if not admitted:
        live.spend(g.boundary(3) - live.key_notice - 280 - live.now)
    slot, _, _, _ = onair._p3_transition_window(
        live, g, tx, s.host, s.rx, 3, np.zeros(0), live.pos, 1920)
    assert slot == 3
    assert any(long for _, long in calls) == admitted
    assert bool(s.packets[-1].cycle_long) == admitted
    assert emit(s, tx, g, live) == (3, 1 if admitted else 0)


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_collection_is_capped_at_ccol_independently_of_decision(
        tmp_path, monkeypatch, delta):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    grant(s, tx, g, live, False)
    tx.aim(g, 3)
    tx.arm_p3_reply()
    cutoff = onair._p3_decode_deadline(live, g.boundary(3), 1920)
    # Move the synthetic packet's last filter sample around the collection
    # cutoff, independently of the later decision's completion time.
    monkeypatch.setattr(rxfront, '_packet_span', lambda span: cutoff + delta
                        - g._p3_command_row0 - rxfront.FILTER_REACH_N)
    ends = []
    take = live.take_until

    def collect(until):
        ends.append(until)
        return take(until)

    monkeypatch.setattr(live, 'take_until', collect)
    monkeypatch.setattr(onair, '_scan_frame', lambda *a, **k: None)
    onair._p3_transition_window(live, g, tx, s.host, s.rx, 3,
                               np.zeros(0), live.pos, 1920)
    assert ends and max(ends) == min(cutoff, cutoff + delta)
    assert live.pos <= cutoff
    assert emit(s, tx, g, live) == (3, 0)


@pytest.mark.parametrize('capture, anchor, costs', [
    ('onair-20261107T174054102950Z-882884a0-13a2-4462-9776-992b4e0f542f',
     31725, (2.50, 10.42, 368.74, 82.44, 389.25, 80.67, 78.61, 86.45)),
    ('onair-20261107T174400635638Z-4f0fa00b-7b15-4390-9c51-0da6a4fec078',
     31652, (1.86, 8.86, 373.31, 112.85, 355.90, 92.92, 31.02, 53.12)),
])
def test_recorded_cs6_walk_preserves_long_control_boundaries(
        tmp_path, monkeypatch, capsys, capture, anchor, costs):
    cap = evidence.RECORD / 'operator-data' / 'captures' / capture
    identity = json.loads(evidence.require_input(cap / 'capture-identity.json').read_text())
    log = evidence.require_input(evidence.RECORD / 'operator-data' / 'working'
                                 / capture / identity['transcript']['path']).read_text()
    assert re.findall(r'SLOT (\d+) IS GONE', log) == ['30', '33', '36']
    assert 'REGRID GAVE UP' in log
    evidence.require_input(next(cap.glob('stream-*.flac'), cap / 'stream.wav'))
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=True)
    g.anchor = anchor
    live._lat, live.tx_latency_n, live._blk, live.holdback = 756, 336, 128, 916
    live._dac = (756 / onair.FS, 0)
    live.audio = wav.read(cap / 'stream.wav').astype(np.float32)
    live.limit = live.audio.size
    tx.aim(g, 27)
    packet(s, tx, g, live, end_slot=27, long=True, status=1,
           decision=g.boundary(27) - 10000, payload=b'S via WS8EOC >\r',
           phase=1489232)
    assert emit(s, tx, g, live) == (27, 5)
    assert g._p3_command_row0 == 1673552
    s.rx.sync.packet_level = 2
    s.rx._p3_clock_role = arq.IRS
    broad = []

    def slow_general(*args, **kwargs):
        charge = costs[min(len(broad) + 2, len(costs) - 1)]
        broad.append(charge)
        live.spend(round(charge * onair.FS / 1000))
        return None

    monkeypatch.setattr(s.rx, '_read_p3_packet', slow_general)
    # Pin the unread walk. CRC recovery is tested separately, since a valid
    # old-cycle repeat must end this short interval as soon as it is decided.
    for level in (1, 2):
        reader = getattr(s.rx.sync, f'sl{level}_packet_at')

        def unread(*args, _reader=reader, _cost=costs[level - 1], **kwargs):
            _reader(*args, **kwargs)
            live.spend(round(_cost * onair.FS / 1000))
            return None

        monkeypatch.setattr(s.rx.sync, f'sl{level}_packet_at', unread)
    capsys.readouterr()
    emitted = []
    for _ in range(12):
        slot = g.next_slot(tx.slot)
        tx.aim(g, slot)
        tx.arm_p3_reply()
        slot, audio, origin, _ = onair._p3_transition_window(
            live, g, tx, s.host, s.rx, slot, np.zeros(0), live.pos, 1920)
        slot, _, _, _ = onair._regrid(live, g, tx, s.host, s.rx,
                                  slot, audio, origin, 1920)
        emitted.append(emit(s, tx, g, live))
    assert emitted == [(slot, 1) for slot in range(30, 66, 3)]
    assert not broad
    out = capsys.readouterr().out
    assert 'IS GONE' not in out and 'REGRID GAVE UP' not in out


@pytest.mark.parametrize('interference', ['corrupt-length', 'header-only', 'energy', 'gap'])
def test_non_crc_evidence_cannot_change_pending_interval(
        tmp_path, monkeypatch, interference):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=True)
    emitted = [grant(s, tx, g, live, True)]
    row = g._p3_command_row0
    waveform = placement.link_packet(1, b'Y', 0x21, long_cycle=True)
    head = row - p3frame.DATA_OFFSET * rxfront.SPS - 900
    if interference == 'energy':
        waveform = np.random.default_rng(4).normal(0, .1, waveform.size)
    elif interference == 'corrupt-length':
        short = placement.link_packet(1, b'Y', 0x21)
        short[:10000] = waveform[:10000]
        waveform = short
    elif interference == 'header-only':
        waveform[10000:] = 0
    if interference in ('corrupt-length', 'header-only'):
        header = s.rx.sync.narrow_header_at(waveform, 5220, 1)
        assert header is not None and header.long_cycle
    live.audio[head:head + waveform.size] = waveform
    s.rx.sync.packet_level = 2
    for slot in (1, 4, 7, 10):
        tx.aim(g, slot)
        tx.arm_p3_reply()
        if interference == 'gap':
            live.flush_to(g.boundary(slot) - 2000)
        found, _, _, _ = onair._p3_transition_window(
            live, g, tx, s.host, s.rx, slot, np.zeros(0), live.pos, 1920)
        assert found == slot + 2
        emitted.append(emit(s, tx, g, live))
    assert emitted == [(0, 5), (3, 0), (6, 0), (9, 0), (12, 0)]
    assert g._p3_interval.pending and g.ticks == 1


@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('old_long', [False, True])
@pytest.mark.parametrize('admitted', [False, True])
def test_pending_default_scan_reads_observed_cycle_within_budget(
        tmp_path, monkeypatch, level, old_long, admitted):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=old_long)
    grant(s, tx, g, live, old_long)
    row = put_packet(s, g, live, level, old_long, 0 if old_long else 0x20)
    s.host.arq.speed_level = level
    tx.aim(g, 3)
    tx.arm_p3_reply()
    paths = placement.LONG_PATHS if old_long else placement.SPEED_PATHS
    end = row + rxfront._packet_span(rxfront._frame_span(paths[level]))
    origin = live.pos
    audio = live.read(end + rxfront.FILTER_REACH_N - origin)
    before = s.rx._p3_delivered_at
    calls = []
    read = getattr(s.rx.sync, f'sl{level}_packet_at')

    def charged(*args, **kwargs):
        calls.append(kwargs['long_cycle'])
        live.spend(round(getattr(s.rx, f'P3_SL{level}_RESERVE_S') * onair.FS))
        return read(*args, **kwargs)

    monkeypatch.setattr(s.rx.sync, f'sl{level}_packet_at', charged)
    monkeypatch.setattr(s.rx, '_read_p3_packet',
                        lambda *a, **k: pytest.fail('unbounded packet reader'))
    if not admitted:
        live.spend(g.boundary(3) - live.key_notice - 280 - live.now)
    assert s.rx._p3_read_long is None
    onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    assert calls == ([old_long] if admitted else [])
    if admitted:
        assert abs(s.rx._p3_delivered_at - row) <= 120
        assert not s.host.arq.cycle_command_emitted
        assert emit(s, tx, g, live) == (3 if old_long else 1, 0)
    else:
        assert s.rx._p3_delivered_at == before
        assert s.host.arq.cycle_command_emitted
        assert emit(s, tx, g, live) == (3, 0)


@pytest.mark.parametrize('old_long', [False, True])
def test_repeated_emitted_cs6_refreshes_command_without_restarting_interval(
        tmp_path, monkeypatch, old_long):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=old_long)
    s.host.arq.cfg.p3_unread_after_gear = 'repeat'
    assert grant(s, tx, g, live, old_long) == (0, arq.CS_CYCLE_TOG)
    interval = g._p3_interval
    assert g._p3_command_row0 == 620640
    assert g._p3_command_slot == 0
    assert s.host.arq.cycle_command_emitted
    assert emit(s, tx, g, live, 3) == (3, arq.CS_CYCLE_TOG)
    assert g._p3_command_slot == 3
    assert g._p3_command_row0 == 800640
    assert g._p3_interval is interval
    assert s.host.arq.cycle_command_emitted
