# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, onair, p3acquire, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_p3_terminal_confirm_arq import Peer, wire


class ClosingPeer(Peer):
    def __init__(self):
        super().__init__()
        self.controls = []
        self.keyed_bursts = []
        self.defer_p3_cs = False

    def send_cs(self, cs_index):
        self.controls.append(cs_index)

    def send_packet(self, sl, payload, status, breakin=False):
        result = super().send_packet(sl, payload, status, breakin=breakin)
        if result != arq.REFUSED:
            self.keyed_bursts.append(self.last)
        return result


def link(role=arq.ISS):
    peer = ClosingPeer()
    host = PtcHost(peer, mycall='W9SSJ')
    host.protocol = spec.Protocol.PACTOR3
    host.p3_qrt_confirm = True
    host.arq.role = role
    host.arq._enter_connected()
    host.arq._next_seq = 2
    host.arq.cfg.speed_up = 'hold'
    host.arq.cfg.repeat_gear = 0
    return host, peer


def packet(host, status=0x40, *, ok=True):
    host.on_rx_event(rxfront.Event(0, 'packet', 'close disruption',
        protocol=spec.Protocol.PACTOR3, packet=(1, status, b'', ok),
        breakin=True))


def keyed(host):
    host.arq.on_host_disconnect()
    host.tick()
    assert host.arq.said_goodbye
    return host.arq._inflight


def test_qrt_survives_cs3_and_repeated_changeover_until_fresh_ack():
    host, peer = link()
    a = host.arq
    first = keyed(host)
    wire(host, arq.CS_BREAKIN)
    assert a._inflight is None and a._qrt_keyed is first
    wire(host, 0)
    assert not a.terminal_confirm_pending
    packet(host)
    assert peer.controls == [0]
    host.tick()
    assert a.role == arq.IRS and len(peer.bursts) == 1
    packet(host)
    host.tick()
    assert a.role == arq.ISS
    fresh = a._inflight
    assert fresh is not first and a._qrt_keyed is fresh
    assert peer.last[3] and peer.last[2] == 0x98
    assert peer.controls == [0]
    wire(host, 1)
    assert not a.terminal_confirm_pending
    wire(host, 0)
    assert a.terminal_confirm_pending and not a.goodbye_acked
    host.tick()
    assert peer.marker_bits == [1]
    wire(host, 0)
    assert not a.goodbye_acked
    wire(host, 1)
    assert a.goodbye_acked and a.state == arq.State.DISCONNECTED
    assert a._qrt_keyed is None


@pytest.mark.parametrize('cs', [arq.CS_SPEED_UP, arq.CS_CYCLE_TOG])
def test_gear_command_cannot_confirm_qrt(cs):
    host, peer = link()
    first = keyed(host)
    wire(host, cs)
    assert host.arq._inflight is first
    assert not host.arq.terminal_confirm_started
    wire(host, 0)
    assert host.arq.terminal_confirm_pending


@pytest.mark.parametrize('sl', [2, 3])
def test_cs5_downshifts_qrt_without_confirming_close(sl):
    host, peer = link()
    a = host.arq
    a.speed_level = sl
    first = keyed(host)
    counter, status = first.seq, first.status
    assert peer.last[0] == sl
    for expected_sl in range(sl - 1, 0, -1):
        transmissions = len(peer.keyed_bursts)
        wire(host, arq.CS_NAK)
        assert a.speed_level == expected_sl
        assert len(peer.keyed_bursts) == transmissions + 1
        assert peer.last == (expected_sl, first.payload, status, first.breakin)
        assert a._inflight is first and a._qrt_keyed is first
        assert a.tx_seq == counter and first.seq == counter
        assert a.state == arq.State.DISCONNECTING
        assert not a.terminal_confirm_started and not a.goodbye_acked
        assert peer.marker_calls == 0

    # A repeat request still cannot acknowledge the QRT at the speed floor.
    wire(host, arq.CS_NAK)
    assert a.speed_level == 1 and a._qrt_keyed is first
    wire(host, (counter & 1) ^ 1)
    assert a._inflight is first and not a.terminal_confirm_started
    host.tick()
    assert peer.marker_calls == 0 and not a.goodbye_acked
    wire(host, counter & 1)
    assert a.terminal_confirm_pending and not a.goodbye_acked
    host.tick()
    assert peer.marker_bits == [(counter & 1) ^ 1]
    wire(host, counter & 1)
    assert not a.goodbye_acked
    wire(host, (counter & 1) ^ 1)
    assert a.goodbye_acked and a.state == arq.State.DISCONNECTED


def test_unkeyed_qrt_cannot_be_confirmed_by_matching_control():
    host, peer = link()
    peer.refuse = True
    host.arq.on_host_disconnect()
    host.tick()
    wire(host, host.arq.tx_seq & 1)
    assert not host.arq.terminal_confirm_started
    assert not host.arq.goodbye_acked
    peer.refuse = False
    host.tick()
    wire(host, host.arq.tx_seq & 1)
    assert host.arq.terminal_confirm_pending


def test_peer_ack_must_key_before_qrt_reclaims_turn():
    host, peer = link()
    keyed(host)
    peer.defer_p3_cs = True
    packet(host)
    host.tick()
    assert host.arq.role == arq.IRS and len(peer.bursts) == 1
    packet(host)
    host.tick()
    assert host.arq.role == arq.IRS
    host.on_cs_emitted(0)
    host.tick()
    assert host.arq.role == arq.ISS and peer.last[3]


def test_qrt_budget_is_short_and_survives_role_changes():
    host, peer = link()
    first = keyed(host)
    host.arq.cycle_long = True
    wire(host, arq.CS_CYCLE_TOG)
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES)
    assert host.arq.state == arq.State.DISCONNECTING
    assert host.arq._goodbye_cycles == arq.GOODBYE_CYCLES
    packet(host)
    assert host.arq._qrt_keyed is first
    host.tick()
    assert host.arq.state == arq.State.DISCONNECTED
    assert not host.arq.goodbye_acked
    assert 'QRT sent but unconfirmed' in onair._goodbye_result(host.arq)


@pytest.mark.parametrize('stage', ['place', 'qrt', 'marker'])
def test_expiry_names_unconfirmed_step(stage):
    host, peer = link()
    if stage == 'place':
        peer.refuse = True
        host.arq.on_host_disconnect()
        host.tick(elapsed_ticks=arq.GOODBYE_PLACE_TICKS)
        expected = 'QRT unplaceable'
    else:
        keyed(host)
        if stage == 'marker':
            wire(host, 0)
            host.tick()
        host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
        expected = 'marker unconfirmed' if stage == 'marker' else 'QRT sent but unconfirmed'
    assert host.arq.state == arq.State.DISCONNECTED
    assert not host.arq.goodbye_acked
    assert expected in onair._goodbye_result(host.arq)


def test_abort_clears_keyed_qrt_before_next_link():
    host, peer = link()
    keyed(host)
    host.arq.on_host_abort()
    assert host.arq._qrt_keyed is None
    host.arq.role = arq.ISS
    host.arq._enter_connected()
    assert host.arq._qrt_keyed is None
    assert not host.arq.said_goodbye


def test_refused_recovery_keeps_first_keyed_qrt_and_its_clock():
    host, peer = link()
    first = keyed(host)
    packet(host)
    host.tick()
    peer.refuse = True
    host.tick()
    assert host.arq.role == arq.IRS
    assert host.arq._qrt_keyed is first
    assert host.arq._goodbye_cycles == 2
    peer.refuse = False
    host.tick()
    assert host.arq.role == arq.ISS
    assert host.arq._qrt_keyed is host.arq._inflight
    assert host.arq._qrt_keyed is not first
    assert host.arq._goodbye_cycles == 3
    host.tick(elapsed_ticks=2)
    assert host.arq.state == arq.State.DISCONNECTED
    assert not host.arq.goodbye_acked


def test_new_peer_packet_needs_its_own_emitted_ack_before_recovery():
    host, peer = link()
    keyed(host)
    packet(host)
    host.tick()
    peer.defer_p3_cs = True
    packet(host, status=0x41)
    host.on_cs_emitted(arq.CS_SPEED_UP)
    host.tick()
    assert host.arq.role == arq.IRS
    assert host.arq._turn_ack_owed
    host.on_cs_emitted(1)
    host.tick()
    assert host.arq.role == arq.ISS
    assert peer.last[2] == 0x98 and peer.last[3]


def test_answered_peer_repeat_cannot_restore_old_driver_raster():
    host, peer = link()
    recovered = []
    peer.recover_p3_turn = lambda ev: recovered.append(ev.packet) or True
    keyed(host)
    packet(host)
    host.tick()
    host.tick()
    fresh = host.arq._inflight
    packet(host)
    assert recovered == []
    assert host.arq.role == arq.ISS and host.arq._inflight is fresh
    packet(host, status=0x41)
    assert recovered == [(1, 0x41, b'', True)]
    assert host.arq.role == arq.IRS and host.arq._qrt_keyed is fresh


def test_qrt_recovery_answers_with_ack_even_after_a_repeat_gear_run():
    host, peer = link()
    a = host.arq
    keyed(host)
    wire(host, arq.CS_BREAKIN)
    a.cfg.speed_up = 'auto'
    a.cfg.repeat_gear = 1
    a._repeat_answer = (1, 0, arq.CS_ACK, 0)
    a._repeat_run = 1
    a.on_rx_packet(1, b'', 0, True, protocol=spec.Protocol.PACTOR3)
    assert peer.controls == [arq.CS_ACK]
    host.tick()
    host.tick()
    assert a.role == arq.ISS and peer.last[2:] == (0x98, True)


TRACE = Path(__file__).parent / 'fixtures' / 'p3-clean-shutdown-1002.json'
FIXTURE = json.loads(evidence.require_input(TRACE, allow_module_level=True).read_text())
FS = spec.SAMPLE_RATE


@pytest.fixture(scope='module')
def close_audio():
    root = Path(__file__).resolve().parents[5]
    result = {}
    for name, pin in FIXTURE['audio'].items():
        path = root / 'captures/onair-1002-1137' / name
        if not path.is_file():
            path = evidence.CAPTURES / 'onair-1002-1137' / name
        if not path.is_file():
            pytest.skip(f'close recording unavailable: {path}')
        assert path.stat().st_size == pin['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin['sha256']
        result[name] = rxfront.load_wav(str(path))
    return result


def recorded_control(audio, t, *, offsets=p3acquire.CONTROL_OFFSETS_HZ):
    lo = round((t - .07) * FS)
    c = p3acquire.control_signal(audio[lo:lo + round(.38 * FS)], offsets=offsets)
    assert c is not None
    assert abs(c.event.t + lo / FS - t) < .003
    return c.event


def recorded_packet(audio, at):
    lo = at - 4800
    c = p3acquire.changeover(audio[lo:lo + round(1.03 * FS)], offsets=(-13.,))
    assert c is not None and c.event.packet is not None
    assert c.event.packet[3]
    return c.event


@pytest.mark.parametrize('marker_ack', [False, True])
def test_1002_audio_replay_preserves_qrt_through_disruption(close_audio, marker_ack):
    host, peer = link(role=arq.IRS)
    a = host.arq
    a.on_host_data(b'FQ\r')
    a.on_host_breakin()
    a._breakin_armed = True
    host.tick()
    assert peer.last[1] == b'FQ\r' and peer.last[3]
    first = fresh = None
    for row in FIXTURE['failure']['events']:
        if row['kind'] == 'control':
            ev = recorded_control(close_audio['stream.flac'], row['t'], offsets=(-13.,))
            assert ev.cs == row['cs']
            host.on_rx_event(ev)
            if row['t'] == 215.33:
                host.tick()
            elif row['t'] == 216.58:
                a.on_host_disconnect()
                host.tick()
                first = a._inflight
                assert first is a._inflight and first.status == 0x9a
            elif row['cs'] == arq.CS_BREAKIN:
                assert a.said_goodbye and a.role == arq.IRS
                host.tick()
            else:
                assert a.terminal_confirm_pending
                assert not a.goodbye_acked and not peer.marker_bits
                host.tick()
        elif row['kind'] == 'packet':
            ev = recorded_packet(close_audio['stream.flac'], row['at'])
            assert ev.packet == (1, 0x40, b'', True)
            host.on_rx_event(ev)
            if fresh is None:
                assert a.said_goodbye and not a.terminal_confirm_pending
                host.tick()
                if a.role == arq.ISS:
                    fresh = a._inflight
                    assert fresh is a._inflight and fresh is not first
                    assert peer.last[2:] == (0x98, True)
                    assert not a.cycle_long
            else:
                assert a._inflight is fresh
                assert a.role == arq.ISS and not a.terminal_confirm_pending
    assert first is not None and fresh is not None
    assert peer.controls.count(0) == 1
    assert peer.marker_bits == [1]
    wire(host, 0)
    assert not a.goodbye_acked
    if marker_ack:
        wire(host, 1)
        assert a.goodbye_acked
    else:
        host.tick(elapsed_ticks=arq.GOODBYE_CYCLES)
        assert 'marker unconfirmed' in onair._goodbye_result(a)
        assert not a.goodbye_acked
    assert a.state == arq.State.DISCONNECTED
    counts = len(peer.bursts), len(peer.controls), len(peer.marker_bits)
    host.tick(elapsed_ticks=20)
    assert counts == (len(peer.bursts), len(peer.controls), len(peer.marker_bits))


def test_1002_witness_keeps_the_four_unconfirmed_tail_cs1_reads(close_audio):
    witness = close_audio['witness.wav']
    lo = round(222.35 * FS)
    c = p3acquire.changeover(witness[lo:round(223.45 * FS)])
    assert c is not None and c.event.packet == (1, 0x98, b'', True)
    assert abs(c.event.t + lo / FS - 222.4475) < .003
    words = [recorded_control(close_audio['witness.wav'], t).cs
             for t in FIXTURE['witness_tail_s']]
    assert words == [0, 0, 0, 0]
    times = FIXTURE['witness_tail_s']
    assert all(abs(b - a - 1.25) < .003 for a, b in zip(times, times[1:]))


@pytest.mark.parametrize('case', FIXTURE['clean'], ids=['ve1yz-0432', 've1yz-2310'])
def test_ve1yz_clean_close_control_sequence(case):
    path = evidence.RECORD / case['source']
    if path.is_file():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
    disrupted = '0432' in case['source']
    host, peer = link(role=arq.ISS if disrupted else arq.IRS)
    a = host.arq
    keyed(host)
    for row in case['events']:
        if row['kind'] == 'control':
            wire(host, row['cs'])
            if row['cs'] == arq.CS_BREAKIN:
                assert a.role == arq.IRS and not a.terminal_confirm_pending
                peer.refuse = True
                host.tick()
                peer.refuse = False
            elif a.terminal_confirm_pending:
                assert not a.goodbye_acked
                host.tick()
        elif row['kind'] == 'packet':
            packet(host, status=row['status'])
        elif (disrupted and row['kind'] == 'tx'
              and row.get('status') == 0x98):
            host.tick()
            if a.role == arq.IRS:
                host.tick()
    assert a.state == arq.State.DISCONNECTED and a.goodbye_acked
    assert peer.marker_bits == [1]
    qrts = [b for b in peer.keyed_bursts if b[2] & spec.STATUS_QRT]
    assert [b[2] for b in qrts] == ([0x9a, 0x9a, 0x98] if disrupted else [0x98])
