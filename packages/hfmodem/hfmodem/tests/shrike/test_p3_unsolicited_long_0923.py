# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import hashlib

import numpy as np

from hfmodem.core import wav
from hfmodem.shrike import arq, onair, placement, rxfront
from hfmodem.tests import evidence
from hfmodem.tests.shrike.p3_cycle_support import emit, grant, put_packet, scene
from hfmodem.tests.shrike.p3_cycle_support import _arm
from hfmodem.tests.shrike.p3_cycle_support import _duplex


def _receive(tmp_path, monkeypatch, audio, *, level, row, status):
    s, tx, g, _ = _arm(tmp_path, audio, sl=level, expected_seq=status & 3)
    a = s.host.arq
    a.cfg.long_cycle = True
    a._cycle_request = None
    a._cycle_command_emitted = False
    a._rx_seen = True
    g.anchor += row - 7200
    g.p3_interval(False, 0)
    s.rx._p3_row0 = s.rx._p3_delivered_at = row - 60000
    s.rx._p3_clock_role = arq.IRS
    s.rx.sync.packet_level = level
    s.rx.p3_receive_offset_hz = 0.
    clock = _duplex(tx, s, audio, 1536)
    transmit = clock.transmit

    def muted(*args, **kwargs):
        result = transmit(*args, **kwargs)
        clock.audio[clock.keyed_at:result[1]] = 0
        return result

    monkeypatch.setattr(clock, 'transmit', muted)
    initial_end = row + rxfront._packet_span(s.rx._p3_span)
    clock.pos = clock.samples = clock.end = initial_end
    audio, origin = clock.audio[:initial_end].copy(), 0
    slot = 1
    while slot <= 3:
        tx.aim(g, slot)
        tx._pending_p3_cs = arq.CS_REQUEST
        slot, audio, origin, _ = onair._p3_transition_window(
            clock, g, tx, s.host, s.rx, slot, audio, origin, 1920)
        tx.emit_pending_cs()
        slot = g.next_slot(slot)
    return s, tx, g


def test_half_duplex_unsolicited_long_packet_has_only_its_endpoint_reply(
        tmp_path, monkeypatch):
    payload = b' WS8EOC >\r'
    wave = placement.link_packet(3, payload, 0x23, long_cycle=True)
    audio = np.pad(wave, (4800, 12000))
    row = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + 4320
    s, tx, g = _receive(tmp_path, monkeypatch, audio, level=3, row=row, status=0x23)
    assert tx.slots_used == [3]
    assert not tx.refused
    assert [ev.packet for ev in s.packets] == [(3, 0x23, payload, True)]
    assert s.host.arq.cycle_long and g.ticks == 3


def test_recorded_unsolicited_long_packet_stays_audible_through_short_slots(
        tmp_path, monkeypatch):
    cap = evidence.CAPTURES / 'onair-0923-2244'
    evidence.require_input(cap / 'stream.wav')
    first = 3381920
    start = first - 7200
    audio = wav.read(cap / 'stream.wav', start=start,
                     stop=start + 4 * onair.FS).astype(np.float32)
    s, tx, g = _receive(tmp_path, monkeypatch, audio, level=1, row=7200, status=0x22)
    assert tx.slots_used == [3]
    assert not tx.refused
    assert len(s.packets) == 1
    level, status, payload, long_cycle = s.packets[0].packet
    assert (level, status, long_cycle) == (1, 0x22, True)
    assert len(payload) == 36
    assert hashlib.sha256(payload).hexdigest() == (
        '600e49aaad118327bf1f7b20d34925e6e988920c43425f2668b28a988d793d30')
    assert s.host.arq.cycle_long and g.ticks == 3


def test_long_header_withdraws_prearmed_interior_reply_before_ptt(
        tmp_path, monkeypatch):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    grant(s, tx, g, live, False)
    row = put_packet(s, g, live, 3, True, 0x21)
    s.host.arq._cycle_request = None
    s.host.arq._cycle_command_emitted = False
    g.p3_interval(False, 0)
    tx.aim(g, 1)
    tx.arm_p3_reply()
    burst = live._armed
    edges = list(live.edges)
    origin = live.pos
    audio = onair._listen_until_answer(
        live, row + 10000 - live.pos, s.host, s.rx, 0,
        ahead=lambda audio, origin: onair._read_ahead(s.rx, g, audio, origin))
    assert burst.stop.is_set() and not burst.keying
    assert live.edges == edges
    assert tx.slot == tx._p3_armed.slot == 3
    audio, origin, _ = onair._collect(
        live, s.rx, audio, origin, row + rxfront._packet_span(s.rx._p3_span))
    slot, _, _, _ = onair._p3_transition_window(
        live, g, tx, s.host, s.rx, tx.slot, audio, origin, 1920)
    assert slot == 3
    assert emit(s, tx, g, live)[0] == 3
    assert tx.slots_used == [0, 3]
    assert s.packets[-1].packet == (3, 0x21, b'X', True)
    assert s.packets[-1].cycle_long


def test_header_withdrawal_race_preserves_committed_burst_and_defers_next_reply(
        tmp_path, monkeypatch):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    grant(s, tx, g, live, False)
    packets, delivered = list(s.packets), s.rx._p3_delivered_at
    row = put_packet(s, g, live, 1, True, 0x21, header_only=True)
    s.host.arq._cycle_request = None
    s.host.arq._cycle_command_emitted = False
    g.p3_interval(False, 0)
    tx.aim(g, 1)
    tx.arm_p3_reply()
    burst, armed = live._armed, tx._p3_armed
    disarm = live.disarm
    decisions = []

    def commit_before_disarm():
        live.spend(max(0, g.boundary(1) - 1000 - live.now))
        decisions.append(disarm())
        return decisions[-1]

    origin = live.pos
    with monkeypatch.context() as racing:
        racing.setattr(live, 'disarm', commit_before_disarm)
        audio = onair._listen_until_answer(
            live, row + 10000 - live.pos, s.host, s.rx, 0,
            ahead=lambda audio, origin: onair._read_ahead(s.rx, g, audio, origin))
    assert decisions and not any(decisions)
    assert tx.slot == 1 and tx._p3_armed is armed and live._armed is burst
    assert not burst.stop.is_set() and not burst.played
    assert audio.size == row + 10000 - origin
    assert origin + audio.size == live.pos
    assert tx.slots_used == [0]
    assert emit(s, tx, g, live) == (1, armed.cs)
    np.testing.assert_array_equal(live.submitted[-1][1], armed.audio)
    tx.aim(g, 2)
    tx.arm_p3_reply()
    following = live._armed
    onair._read_ahead(s.rx, g, audio, origin)
    assert following.stop.is_set() and not following.keying
    assert tx.slot == tx._p3_armed.slot == 3
    endpoint = tx._p3_armed
    for slot in (0, 3, 4):
        assert onair._p3_reply_after_long(tx, slot, 3) == slot
        assert tx._p3_armed is endpoint
    assert tx.slots_used == [0, 1]
    assert s.packets == packets and s.rx._p3_delivered_at == delivered
    assert not s.host.arq.cycle_long and g.ticks == 1
