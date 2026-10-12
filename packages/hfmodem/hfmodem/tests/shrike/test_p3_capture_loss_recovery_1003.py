# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

FS, CYCLE = onair.FS, 60000
CASES = json.loads(evidence.require_input(
    Path(__file__).parent / 'fixtures' / 'p3-capture-loss-1003.json',
    allow_module_level=True).read_text())


def measured_grid(case):
    g = onair._MasterGrid(case['control_phase'] % CYCLE, CYCLE, 8880,
                         packet_n=46080, cs_n=5760, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.d_n, g.d_ref_n = 4413, 17280
    phase, width = case['peer_phase'], case['width']
    identity = (False, 1, case['status'], bytes.fromhex(case['payload_hex']))
    for at in (phase - 2*CYCLE, phase - CYCLE):
        g.note_p3_packet(at, width, CYCLE, identity=identity, swapped=False)
        g.cycles += 1
    g.note_p3_control(case['control_phase'])
    g.note_p3_packet(phase, width, CYCLE, identity=identity, swapped=False)
    assert g._p3_reply_timing() is not None
    g.reverse(to_iss=True)
    return g


@pytest.fixture(params=CASES, ids=lambda c: c['session'])
def recorded(request):
    case = request.param
    paths = [evidence.require_input(evidence.RECORD / name) for name in case['files']]
    for path in paths:
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == case['files'][
                str(path.relative_to(evidence.RECORD))]
    return case, rxfront.load_wav(str(paths[0]))


def station(case, audio, tmp_path):
    g, s = measured_grid(case), _Session(role=arq.ISS)
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.live = _Bench(seconds=300)
    tx.live.audio[:len(audio)] = audio
    tx.attach(s.host)
    s.host.peer = tx
    tx.raster, tx.sessrx = g, s.rx
    tx.defer_p3_cs = True
    tx.breakin_due = True
    a = s.host.arq
    a._inflight = arq._Packet(status=0, payload=b';FW', sl=1, breakin=True)
    a._outbuf.extend(b' queued')
    a._buffer_raw = 10
    a.cfg.speed_up = 'hold'
    s.rx.p3_receive_offset_hz = case['offset_hz']
    s.rx._p3_row0 = s.rx._p3_delivered_at = case['peer_phase'] + 4320
    s.rx._p3_cycle_n = CYCLE
    s.rx._p3_span = rxfront._frame_span(placement.SPEED_PATHS[1])
    s.rx.sync.packet_level = 1
    return s, tx, g


def read_packet(s, tx, g, audio, row0):
    s.rx.new_cycle()
    g.cycles += 1
    origin, end = row0 - 6000, row0 + 36000
    tx.live.now = tx.live.pos = end
    onair._scan_frame(s.rx, audio[origin:end], origin, tracked_only=True)
    assert s.packets[-1].packet[3] is True
    assert abs(s.rx._p3_delivered_at - row0) <= rxfront.SPS


@pytest.mark.parametrize('overrun', [False, True])
def test_recorded_crc_restores_changeover_in_next_available_cycle(
        recorded, overrun, tmp_path):
    case, audio = recorded
    s, tx, g = station(case, audio, tmp_path)
    a = s.host.arq
    inflight, queued, buffered = a._inflight, bytes(a._outbuf), a._buffer_raw
    gap = g._p3_reply_timing().reverse_gap
    for samples, at in case['losses']:
        g.note_capture_loss(samples, at=at)
        assert g._p3_reply_timing() is None
    tx.aim(g, case['slot'])
    historical, fresh = case['reads']
    read_packet(s, tx, g, audio, historical[0])
    assert g._p3_reply_timing() is None
    assert tx._place_breakin() == ''
    slot = case['slot']
    tx.aim(g, slot)
    if overrun:
        tx.live.now = max(tx.live.now,
                          tx.key_instant(g, slot) - tx.live.key_notice + round(.0306*FS))
        slot, _, _ = onair._regrid(tx.live, g, tx, s.host, s.rx, slot,
                                  np.zeros(0, np.float32), tx.live.pos, 1920)
    else:
        read_packet(s, tx, g, audio, fresh[0])
        early = g.boundary(slot) - tx.key_instant(g, slot)
        slot = onair._keyable_slot(tx.live, g, slot, 1920 + early, listen=False)
    assert g._p3_reply_timing().reverse_gap == gap
    assert s.packets[-1].packet == (1, case['status'],
                                   bytes.fromhex(case['payload_hex']), True)
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = max(tx.live.pos,
                                   tx.key_instant(g, slot) - round(.04*FS))
    assert tx.send_packet(inflight.sl, inflight.payload, inflight.status,
                          breakin=True) != arq.REFUSED
    assert len(tx.live.emissions) == 1 and not tx.refused
    assert case['slot'] <= tx.slot <= case['slot'] + 2
    phase = tx.tx_audio_start + min(tx.tx_pulse_offsets)
    expected = case['control_phase'] - sum(n for n, _ in case['losses'])
    assert abs((phase - expected + CYCLE//2) % CYCLE - CYCLE//2) <= round(.005*FS)
    assert tx.tx_key_up >= g.peer_packet_end(tx.slot).at_slot
    assert a._inflight is inflight and bytes(a._outbuf) == queued
    assert a._buffer_raw == buffered and a.unconfirmed_breakin


@pytest.mark.parametrize('long', [False, True])
@pytest.mark.parametrize('miss_ms', [5.6, 5.8, 8.2])
def test_late_ack_is_rendered_in_next_slot_carrier_order(
        long, miss_ms, tmp_path, monkeypatch):
    from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid
    g = irs_grid()
    g.regear(long)
    cycle = g.cycle_n
    width = 158880 if long else 39120
    phase = 1500000
    g.note_p3_packet(phase, width, cycle, swapped=False,
                     identity=(False, 1, 2, b'>\r'))
    tx = driver(g, tmp_path)
    tx.p3_control_waveform = 'current'
    slot = (phase + round((onair.P3_LONG_REPLY_S if long else onair.P3_REPLY_S)*FS)
            - g.anchor) // CYCLE
    slot = onair._p3_place_reply(g, tx, slot)
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - 1920
    offset = tx._p3_offset
    orders = []

    def render(*args, **kwargs):
        orders.append(tx._p3_reply_carrier_order(g.boundary(tx.slot), False))
        audio = offset(*args, **kwargs)
        if len(orders) == 1:
            tx.live.now = g.boundary(tx.slot) - tx.live.key_notice + round(miss_ms*FS/1000)
            tx.live.now += tx.live._blk - tx.live.now % tx.live._blk
        return audio

    monkeypatch.setattr(tx, '_p3_offset', render)
    assert tx._send_p3_control(arq.CS_ACK) is None
    assert len(tx.live.emissions) == 1 and not tx.refused
    assert tx.slot == g.next_slot(slot)
    assert orders == [False, True]
    assert tx.tx_audio_start == g.boundary(tx.slot)
    assert tx.tx_pulse_offsets[0] > tx.tx_pulse_offsets[1]


@pytest.mark.parametrize('stage', ['drain', 'enqueue'])
def test_ack_recovers_a_deadline_missed_after_render(stage, tmp_path, monkeypatch):
    from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid
    g = irs_grid()
    tx = driver(g, tmp_path)
    slot = 30
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = g.boundary(slot) - 1920
    calls = []
    if stage == 'enqueue':
        emit = tx.live.transmit

        def enqueue(*args, **kwargs):
            calls.append(tx.slot)
            if len(calls) == 1:
                raise onair._MissedTxSlot(53)
            return emit(*args, **kwargs)

        monkeypatch.setattr(tx.live, 'transmit', enqueue)
    else:
        take = tx.live.take_until

        def drain(*args, **kwargs):
            calls.append(tx.slot)
            got = take(*args, **kwargs)
            if len(calls) == 1:
                tx.live.now = g.boundary(tx.slot) - tx.live.key_notice + 480
            return got

        monkeypatch.setattr(tx.live, 'take_until', drain)
    assert tx._send_p3_control(arq.CS_ACK) is None
    assert len(tx.live.emissions) == 1 and not tx.refused
    assert tx.slot == g.next_slot(slot)


def test_repeated_control_overruns_remain_bounded(tmp_path, monkeypatch):
    from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid
    g = irs_grid()
    tx = driver(g, tmp_path)
    tx.aim(g, 30)
    tx.live.now = tx.live.pos = g.boundary(30) - 1920
    offset = tx._p3_offset
    renders = []

    def overrun(*args, **kwargs):
        renders.append(tx.slot)
        audio = offset(*args, **kwargs)
        tx.live.now = g.boundary(tx.slot) - tx.live.key_notice + 480
        return audio

    monkeypatch.setattr(tx, '_p3_offset', overrun)
    assert tx._send_p3_control(arq.CS_ACK) == arq.REFUSED
    assert len(renders) == onair.REGRID_TRIES
    assert not tx.live.emissions and tx.refused


@pytest.mark.parametrize('problem', ['no_pair', 'unknown_splice', 'large_loss',
                                     'spanning_packet', 'age', 'period', 'width',
                                     'phase', 'head'])
def test_loss_candidate_never_supplies_unqualified_placement(problem):
    case = CASES[0]
    g = measured_grid(case)
    if problem == 'no_pair':
        g._clear_p3_reply_timing()
    splice = case['peer_phase'] + 39119 if problem == 'spanning_packet' else 10138112
    g.note_capture_loss(153904 if problem == 'large_loss' else 188,
                        at=None if problem == 'unknown_splice' else splice)
    at, width, period = 10309168, 39120, CYCLE
    if problem == 'age':
        g.cycles += onair.ONSET_MAX_CYCLES + 1
    if problem == 'period':
        period *= 3
    if problem == 'width':
        width -= 240
    if problem == 'phase':
        at += round(.021*FS)
    identity = None if problem == 'head' else (False, 1, 2, b'>\r')
    g.note_p3_packet(at, width, period, identity=identity)
    assert g._p3_reply_timing() is None
    assert g.breakin_refusal(g.peer_packet_end(174)) is not None


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['session'])
def test_known_loss_above_twenty_ms_recovers_from_matching_packets(case):
    g = measured_grid(case)
    gap = g._p3_reply_timing().reverse_gap
    lost = 1000
    g.note_capture_loss(lost, at=case['losses'][0][1])
    first = max(3, (case['losses'][0][1] + lost - case['peer_phase']) // CYCLE + 1)
    for k in range(first, first + 7):
        g.cycles += 1
        g.note_p3_packet(case['peer_phase'] + k*CYCLE - lost, case['width'], CYCLE,
                         identity=(False, 1, case['status'],
                                   bytes.fromhex(case['payload_hex'])), swapped=False)
        assert g._p3_reply_timing().reverse_gap == gap
        assert g.breakin_refusal(g.peer_packet_end(case['slot'] + k)) is None


def publish_losses(tx, case):
    tx.live.lost = sum(n for n, _ in case['losses'])
    tx.live.lost_at = [(at, n, n/FS) for n, at in case['losses']]


@pytest.mark.parametrize('ordering', ['during_collect', 'decode_before_notice'])
def test_recorded_loss_order_restores_and_keys_changeover(
        recorded, ordering, tmp_path, monkeypatch):
    case, audio = recorded
    s, tx, g = station(case, audio, tmp_path)
    evidence_of = onair._CycleEvidence(tx.live, tx, g, tmp_path)
    evidence_of.input_loss('pre-collect')
    gap = g._p3_reply_timing().reverse_gap
    row = case['reads'][1][0]
    if ordering == 'during_collect':
        read_ready = tx.live.read_ready

        def collect():
            got = read_ready()
            publish_losses(tx, case)
            return got

        monkeypatch.setattr(tx.live, 'read_ready', collect)
        s.rx.new_cycle()
        g.cycles += 1
        tx.live.now = tx.live.pos = row - 6000
        seg, origin, _ = onair._collect(tx.live, s.rx, np.zeros(0, np.float32),
                                        tx.live.pos, row + 36000)
        assert tx.live.lost > evidence_of.lost
        onair._scan_frame(s.rx, seg, origin, tracked_only=True)
        assert evidence_of.lost == tx.live.lost
    else:
        read_packet(s, tx, g, audio, row)
        publish_losses(tx, case)
        evidence_of.input_loss('post-key')
        assert g._p3_loss_timing is not None
        for k in range(1, 4):
            read_packet(s, tx, g, audio, row + k*CYCLE)
            assert g._p3_reply_timing() is not None
    assert s.packets[-1].packet == (1, case['status'],
                                   bytes.fromhex(case['payload_hex']), True)
    assert g._p3_reply_timing().reverse_gap == gap
    a = s.host.arq
    inflight, queued = a._inflight, bytes(a._outbuf)
    slot = case['slot'] + (3 if ordering == 'decode_before_notice' else 0)
    early = g.boundary(slot) - tx.key_instant(g, slot)
    slot = onair._keyable_slot(tx.live, g, slot, 1920 + early, listen=False)
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = max(tx.live.pos, tx.key_instant(g, slot) - 1920)
    assert tx.send_packet(inflight.sl, inflight.payload, inflight.status,
                          breakin=True) != arq.REFUSED
    assert len(tx.live.emissions) == 1 and tx.slot <= slot + 1
    assert a.unconfirmed_breakin and a._inflight is inflight
    assert bytes(a._outbuf) == queued
    phase = tx.tx_audio_start + min(tx.tx_pulse_offsets)
    expected = case['control_phase'] - sum(n for n, _ in case['losses'])
    assert abs((phase - expected + CYCLE//2) % CYCLE - CYCLE//2) <= round(.005*FS)
    assert tx.tx_key_up >= g.peer_packet_end(tx.slot).at_slot


def test_review_ack_retry_observes_loss_before_keying(tmp_path, monkeypatch):
    from hfmodem.tests.shrike.test_p3_reply_placement import REPLY_N, driver, irs_grid
    phase, lost = 1500000, 20000
    g = irs_grid()
    g.note_p3_packet(phase, 39120, CYCLE, swapped=False,
                     identity=(False, 1, 2, b'>\r'))
    tx = driver(g, tmp_path)
    tx.p3_control_waveform = 'current'
    slot = onair._p3_place_reply(g, tx, round((phase + REPLY_N - g.anchor) / CYCLE))
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - 1920
    offset, take = tx._p3_offset, tx.live.take_until
    renders = []

    def late(*args, **kwargs):
        renders.append(tx.slot)
        audio = offset(*args, **kwargs)
        if len(renders) == 1:
            tx.live.now = g.boundary(tx.slot) - tx.live.key_notice + 480
        return audio

    def splice(*args, **kwargs):
        tx.live.lost = lost
        tx.live.lost_at = [(tx.live.pos, lost, lost/FS)]
        return take(*args, **kwargs)

    monkeypatch.setattr(tx, '_p3_offset', late)
    monkeypatch.setattr(tx.live, 'take_until', splice)
    assert tx._send_p3_control(arq.CS_ACK) == arq.REFUSED
    assert g._p3_lost_n == lost and not tx.live.emissions


@pytest.mark.parametrize('stage', ['retry_wait', 'final_drain'])
def test_recorded_ack_replaces_after_loss_in_wait(recorded, stage, tmp_path,
                                                monkeypatch):
    case, audio = recorded
    s, tx, g = station(case, audio, tmp_path)
    g.sending, s.host.arq.role = False, arq.IRS
    tx.breakin_due = False
    tx.p3_control_waveform = 'current'
    row = case['reads'][1][0]
    phase = row - 4320 - CYCLE
    g.note_p3_packet(phase, case['width'], CYCLE, swapped=False,
                     identity=(False, 1, case['status'],
                               bytes.fromhex(case['payload_hex'])))
    s.rx._p3_row0 = s.rx._p3_delivered_at = row - CYCLE
    slot = onair._p3_place_reply(g, tx,
                                round((phase + round(onair.P3_REPLY_S*FS)
                                       - g.anchor) / CYCLE))
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - 1920
    offset, take = tx._p3_offset, tx.live.take_until
    renders, losses = [], []

    def render(*args, **kwargs):
        renders.append(tx.slot)
        wave = offset(*args, **kwargs)
        if stage == 'retry_wait' and len(renders) == 1:
            tx.live.now = g.boundary(tx.slot) - tx.live.key_notice + 480
        return wave

    def splice(*args, **kwargs):
        got = take(*args, **kwargs)
        if not losses:
            publish_losses(tx, case)
            losses.append(tx.slot)
        return got

    monkeypatch.setattr(tx, '_p3_offset', render)
    monkeypatch.setattr(tx.live, 'take_until', splice)
    cs = case['status'] & 1
    assert tx._send_p3_control(cs) is None
    assert tx._cycle_evidence.lost == tx.live.lost
    assert len(tx.live.emissions) == 1 and not tx.refused
    assert slot < tx.slot <= slot + onair.REGRID_TRIES
    assert s.packets[-1].packet[3]
    peer = g._p3_peer
    assert peer[0] >= row - 4320
    assert tx.tx_audio_start - peer[0] == round(onair.P3_REPLY_S*FS)
    assert tx.tx_key_up >= peer[0] + peer[1]


def test_recorded_unconfirmed_iss_silence_remains_bounded(recorded, tmp_path):
    case, audio = recorded
    s, tx, g = station(case, audio, tmp_path)
    a = s.host.arq
    publish_losses(tx, case)
    read_packet(s, tx, g, audio, case['reads'][1][0])
    assert a.role == arq.ISS and a.unconfirmed_breakin
    tx.live.audio[:] = 0
    inflight, queued = a._inflight, bytes(a._outbuf)
    bound = arq.P3_CRC_WAIT_TICKS
    for k in range(bound):
        s.rx.new_cycle()
        g.cycles += 1
        slot = case['slot'] + 1 + k
        tx.listening = a.breakin_listen_due
        tx.aim(g, slot)
        tx.live.now = tx.live.pos = tx.key_instant(g, slot) - 1920
        origin = tx.live.pos - CYCLE
        onair._scan_frame(s.rx, tx.live.audio[origin:tx.live.pos], origin,
                          tracked_only=True)
        token = a.begin_receive_opportunity()
        s.host.tick()
        a.finish_receive_opportunity(token)
        if a.state in (arq.State.DISCONNECTED, arq.State.LISTENING):
            break
        if a.state == arq.State.CONNECTED and not a._qrt_pending:
            assert a.role == arq.ISS and a._inflight is inflight
            assert bytes(a._outbuf) == queued
    assert a.state in (arq.State.DISCONNECTED, arq.State.LISTENING)
    assert k == bound - 1
    assert a._p3_crc_wait_ticks == bound
    assert a.transfer_unconfirmed['field'] == inflight.payload
    assert a.transfer_unconfirmed['suffix'] == queued
    assert a._dark_windows == bound
