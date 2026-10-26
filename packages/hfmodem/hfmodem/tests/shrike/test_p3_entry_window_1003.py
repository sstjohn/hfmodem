# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.tests import evidence

from hfmodem.shrike import arq, onair, pactor1, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_granted_entry_retry import granted, repeat
from hfmodem.tests.shrike.test_grid import _Bench

MANIFEST = Path(__file__).with_name('fixtures') / 'p3-entry-window-1003.json'
ARMS = {a['arm']: a for a in json.loads(MANIFEST.read_text())}


@cache
def recording(arm):
    meta = ARMS[arm]
    folder = evidence.CAPTURES / ('onair-' + arm)
    for name, digest in meta['sha256'].items():
        path = folder / name
        if not path.exists():
            pytest.skip(f'recording absent: {path}')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return rxfront.load_wav(str(folder / 'stream.flac'))


def grid(arm, row):
    folder = evidence.CAPTURES / ('onair-' + arm)
    tx = json.loads((folder / row['tx']).read_text())
    boundary = tx['audio_start'] - 234
    g = onair._MasterGrid(boundary - row['slot'] * 60000, 60000, 8880,
                         packet_n=46080, cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.d_n, g.d_ref_n = round(ARMS[arm]['d_ms'] * 48), 46080
    g.keyed_slot = row['slot']
    g._p3_keyed_reply = (row['slot'], tx['audio_start'], tx['audio_end'], 60000)
    assert g.rx_due(g.keyed_slot) == row['nominal']
    return g


def collected_end(rx, g, row):
    rx.host.peer.live = SimpleNamespace(holdback=916)
    target = rx.control_collect_until(row['until'], row['origin'], g)
    assert target <= g.rx_due(g.keyed_slot) + g.p1_cs_n + 720 + 916
    return target - 916


def observe_entry_answered(monkeypatch, host):
    """Observe the original bound transition before its caller can send DATA."""
    original = host._entry_answered
    transitions = []

    def snapshot():
        peer, a = host.peer, host.arq
        present = hasattr(peer, '_p3_entry_listen_slot')
        state = dict(marker_present=present,
                     marker=peer._p3_entry_listen_slot if present else None,
                     pending=a.entry_pending, slot=peer.slot,
                     grant=None if host._entry_grant is None else tuple(host._entry_grant),
                     confirmed=a._p3_entry_confirmed, protocol=host.protocol,
                     state=a.state, role=a.role, queued=bytes(a._outbuf),
                     inflight_empty=a._inflight is None,
                     acked_bytes=a.tx_acked_bytes)
        if isinstance(peer, onair.RadioTx):
            state.update(n=peer.n, slots_used=tuple(peer.slots_used),
                         keyed=tuple(peer.keyed), emissions=tuple(peer.live.emissions),
                         wavs=tuple(sorted(p.name for p in peer.outdir.glob('*.wav'))))
        return state

    def observe(*args, **kwargs):
        before = snapshot()
        result = original(*args, **kwargs)
        after = snapshot()
        transitions.append((before, after))
        return result

    monkeypatch.setattr(host, '_entry_answered', observe)
    return transitions


def assert_entry_retired(transitions, host, queued, acked_bytes):
    assert len(transitions) == 1
    before, after = transitions[0]
    assert before['marker_present'] and before['marker'] == 18
    assert before['pending'] is True and before['grant'] is not None
    # on_rx_event marks a genuine P3 reply before dispatching its ACK;
    # entry ownership nevertheless remains live until this original method.
    assert before['confirmed'] is True
    assert after['marker_present'] and after['marker'] is None
    assert after['pending'] is False and after['grant'] is None
    assert after['confirmed'] is True
    for state in (before, after):
        assert state['slot'] == 18
        assert state['protocol'] == spec.Protocol.PACTOR3
        assert state['state'] == arq.State.CONNECTED and state['role'] == arq.ISS
        assert state['queued'] == queued
        # ACK processing clears the entry inflight BEFORE calling this seam.
        assert state['inflight_empty']
        assert state['acked_bytes'] == acked_bytes
    assert hasattr(host.peer, '_p3_entry_listen_slot')
    assert host.peer._p3_entry_listen_slot is None


def saved_entry_window(g, row, listen_slot):
    return dict(keyed=tuple(g._p3_keyed_reply), keyed_slot=g.keyed_slot,
                origin=row['origin'], listen_slot=listen_slot,
                boundary=g.boundary(listen_slot))


def assert_original_entry_capture(result, saved, audio, live):
    _, _, captured, origin = result
    keyed_slot, start, end, cycle = saved['keyed']
    assert keyed_slot == saved['keyed_slot'] == saved['listen_slot'] - 1
    assert cycle == 60000 and live.holdback == 916
    assert live._blk == 128 and live.lat_in == 0
    coordinates = {
        17: (16, 962973, 1003451, 1022973, 1023889, 1023872),
        18: (17, 1022973, 1063451, 1082973, 1083889, 1083776),
    }
    reply_end = start + cycle
    target = reply_end + live.holdback
    returned_end = target // live._blk * live._blk
    observed_end = origin + captured.size
    print(f'original entry capture: {saved}, reply_end={reply_end}, '
          f'target={target}, returned=[{origin}, {observed_end})')
    assert (keyed_slot, start, end, reply_end, target, returned_end) == coordinates[saved['listen_slot']]
    assert saved['origin'] == end == origin
    assert observed_end == returned_end
    assert observed_end >= reply_end
    np.testing.assert_array_equal(captured, audio[origin:returned_end])


def assert_pending_candidate(rx, result, transitions, packet, queued):
    host = rx.host
    assert result[0] is None
    assert transitions == []
    candidate = rx._p3_head_candidate
    assert candidate is not None
    assert candidate[0] == rx._p3_acquisition_cycle
    # Logs round this instant; use the acquisition's existing 20 ms tolerance.
    assert candidate[1] / onair.FS == pytest.approx(21.034, abs=.020)
    assert not any(e.protocol == spec.Protocol.PACTOR3 for e in rx.cs_log)
    assert host.arq.entry_pending is True
    assert host.protocol == spec.Protocol.PACTOR3
    assert host.arq.state == arq.State.CONNECTED and host.arq.role == arq.ISS
    assert host.peer._p3_entry_listen_slot == host.peer.slot == 17
    assert host.arq._inflight is packet and packet.entry and packet.seq == 2
    assert host.arq.tx_seq == 2 and bytes(host.arq._outbuf) == queued
    return tuple(candidate)


def assert_distinct_entry_ack(rx, result, candidate):
    assert result[0] == arq.CS_ACK
    events = [e for e in rx.cs_log if e.protocol == spec.Protocol.PACTOR3]
    assert len(events) == 1 and events[0].cs == arq.CS_ACK
    assert events[0].t == pytest.approx(22.29, abs=.020)
    accepted_at = rx._p3_answer_at
    assert accepted_at / onair.FS == pytest.approx(events[0].t, abs=1 / onair.FS)
    assert rx._p3_acquisition_cycle - candidate[0] == 1
    assert abs(accepted_at - candidate[1] - 60000) <= round(.020 * onair.FS)
    assert abs(rx.p3_receive_offset_hz - candidate[2]) <= 25
    assert rx._p3_head_candidate is None


@pytest.mark.parametrize('arm', list(ARMS)[:5])
def test_recorded_cutoff_and_selected_window(arm):
    audio = recording(arm)
    for row in ARMS[arm]['rows']:
        g = grid(arm, row)
        old = _Session(entry_pending=True)
        old.rx.new_cycle()
        assert old.rx.control_signal_in(audio[row['origin']:row['end']],
                                        row['origin'], g) == (None, None)
        assert old.events == []
        new = _Session(entry_pending=True)
        new.rx.new_cycle()
        end = collected_end(new.rx, g, row)
        assert g.rx_due_in(row['origin'], end) == row['nominal']
        new.rx.control_signal_in(audio[row['origin']:end], row['origin'], g)
        assert [(e.kind, e.cs, e.spare, str(e.protocol)) for e in new.events] == [
            tuple(e) for e in row['new_events']]


@pytest.mark.parametrize('arm,count,confirmed,level', [
    ('1003-0004', 9, False, 1), ('1003-0007', 9, False, 1),
    ('1003-0008', 0, True, 2), ('1003-0010', 7, True, 1),
    ('1003-0012', 7, False, 1),
])
@pytest.mark.parametrize('known_grant', [False, True])
def test_recorded_entry_campaign(arm, count, confirmed, level, known_grant, monkeypatch):
    audio = recording(arm)
    host, _ = granted()
    transitions = (observe_entry_answered(monkeypatch, host)
                   if known_grant and arm == '1003-0012' else [])
    if not known_grant:
        host._entry_grant = None
    host.no_p3_fallback = True
    rx = onair._SessionRx(host, tag='ENTRY')
    packet, queued = host.arq._inflight, bytes(host.arq._outbuf)
    acked_bytes = host.arq.tx_acked_bytes
    candidate = None
    for row in ARMS[arm]['rows']:
        rx.new_cycle()
        g = grid(arm, row)
        if known_grant:
            if arm == '1003-0012' and row in ARMS[arm]['rows'][-2:]:
                assert host.arq._inflight is packet and packet.entry and packet.seq == 2
                assert host.arq.tx_seq == 2 and bytes(host.arq._outbuf) == queued
                assert host.arq.tx_acked_bytes == acked_bytes
            result, saved, live = resolved_entry_reply(rx, g, row, audio)
            if arm == '1003-0012' and row in ARMS[arm]['rows'][-2:]:
                assert_original_entry_capture(result, saved, audio, live)
                if saved['listen_slot'] == 17:
                    candidate = assert_pending_candidate(rx, result, transitions, packet, queued)
                    assert host.arq._upgrade_requests == 7
                else:
                    assert candidate is not None
                    assert_distinct_entry_ack(rx, result, candidate)
                    assert_entry_retired(transitions, host, queued, acked_bytes)
        else:
            end = collected_end(rx, g, row)
            rx.control_signal_in(audio[row['origin']:end], row['origin'], g)
        host.tick()
        if not host.arq.entry_pending:
            break
        assert host.arq._inflight is packet
        assert bytes(host.arq._outbuf) == queued
        assert host.arq.role == arq.ISS
        assert host.arq.state == arq.State.CONNECTED
    if known_grant and arm == '1003-0012':
        confirmed = True
        assert_entry_retired(transitions, host, queued, acked_bytes)
        data = host.arq._inflight
        assert data is not None and data is not packet
        assert not data.entry and not data.breakin and data.status == 0x23
        assert data.seq == host.arq.tx_seq == 3 and data.sl == 1
        assert data.payload == b'pendi' and data.retries == 0 and data.repeats == 0
        assert data.payload + bytes(host.arq._outbuf) == queued
        assert host.arq.tx_acked_bytes == acked_bytes
    assert host.arq.entry_pending is not confirmed
    assert host.arq._upgrade_requests == count
    assert host.arq._sl == level
    if arm in ('1003-0004', '1003-0007'):
        assert host.arq._unanswered_upgrade == 0
        assert packet.retries == 1
        assert not any('silence rule' in s for s in host.log_lines)


@pytest.mark.parametrize('condition', ['normal', 'wrong_slot', 'unkeyed',
                                       'receiving', 'confirmed', 'stale'])
def test_truncation_is_unavailable_only_for_the_retained_keyed_entry(condition):
    arm, row = '1003-0004', ARMS['1003-0004']['rows'][0]
    audio = recording(arm)
    host, _ = granted()
    rx, g = onair._SessionRx(host, tag='ENTRY'), grid(arm, row)
    origin = row['origin']
    if condition == 'wrong_slot':
        g.keyed_slot += 1
    elif condition == 'unkeyed':
        g._p3_keyed_reply = None
        g.keyed_slot = None
    elif condition == 'receiving':
        host.arq.role, g.sending = arq.IRS, False
    elif condition == 'confirmed':
        host.arq.entry_pending = False
    elif condition == 'stale':
        origin = row['nominal'] + 1
    rx.new_cycle()
    rx.control_signal_in(audio[origin:row['end']], origin, g)
    assert host.arq._entry_window_truncated is (condition == 'normal')
    assert rx.cs_log == []
    if condition == 'normal':
        host.tick()
        assert host.arq._unanswered_upgrade == 0
        assert host.arq._upgrade_requests == 1
        assert host.arq._inflight.retries == 1


def test_unavailable_reads_are_bounded_and_do_not_invent_answers():
    host, _ = granted()
    host.no_p3_fallback = True
    for n in range(host.arq.cfg.max_retries):
        host.arq.note_entry_window_truncated()
        host.tick()
        assert host.arq._upgrade_requests == 0
        if n + 1 < host.arq.cfg.max_retries:
            assert host.arq._unanswered_upgrade == 0
            assert host.arq.role == arq.ISS
            assert host.arq._inflight.retries == 0
    assert host.arq.state == arq.State.DISCONNECTED
    assert any('entry receive budget exhausted' in s for s in host.log_lines)
    assert not any('silence rule' in s for s in host.log_lines)


def test_repeated_grants_reach_the_existing_bound_without_fallback():
    host, _ = granted()
    host.no_p3_fallback = True
    for n in range(arq.ENTRY_GRANT_CYCLES):
        repeat(host)
        if n + 1 < arq.ENTRY_GRANT_CYCLES:
            assert host.arq.state == arq.State.CONNECTED
            assert host.arq.entry_pending and host.arq.role == arq.ISS
    assert host.arq.state == arq.State.CONNECTED and host.arq.terminal_pending
    assert host.protocol == spec.Protocol.PACTOR1 and not host.arq.entry_pending
    host.tick()
    assert host.arq.said_goodbye and host.arq.state == arq.State.DISCONNECTING
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
    assert host.arq.state == arq.State.DISCONNECTED and not host.arq.goodbye_acked
    assert any('repeated-grant rule' in s for s in host.log_lines)


def test_late_grants_after_listen_refusals_spend_the_emitted_reply_budget():
    host, keyed = granted()
    host.no_p3_fallback = True
    send = keyed.send_packet
    packet, queued = host.arq._inflight, bytes(host.arq._outbuf)
    for n in range(40):
        keyed.send_packet = lambda *a, **kw: arq.REFUSED
        host.tick()
        keyed.send_packet = send
        repeat(host)
        if n + 1 < arq.ENTRY_GRANT_CYCLES:
            assert host.arq.state == arq.State.CONNECTED
            assert host.arq._inflight is packet and host.arq.entry_pending
            assert host.arq._upgrade_requests == n + 1
            assert host.arq._unanswered_upgrade == 0
            assert packet.retries == 0
            assert bytes(host.arq._outbuf) == queued
        else:
            if n + 1 == arq.ENTRY_GRANT_CYCLES:
                assert host.arq.state == arq.State.CONNECTED and host.arq.terminal_pending
            elif 2 * (n + 1 - arq.ENTRY_GRANT_CYCLES) - 1 <= arq.GOODBYE_CYCLES:
                assert host.arq.state == arq.State.DISCONNECTING and host.arq.said_goodbye
            else:
                assert host.arq.state == arq.State.DISCONNECTED and not host.arq.goodbye_acked
            assert host.protocol == spec.Protocol.PACTOR1 and not host.arq.entry_pending
    assert host.arq._upgrade_requests == arq.ENTRY_GRANT_CYCLES
    assert len(keyed.p3) == arq.ENTRY_GRANT_CYCLES
    assert any('repeated-grant rule' in s for s in host.log_lines)
    host.tick()
    assert host.arq.said_goodbye
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
    assert host.arq.state == arq.State.DISCONNECTED and not host.arq.goodbye_acked


def test_one_emitted_reply_owns_only_one_grant_across_refusals():
    host, keyed = granted()
    keyed.send_packet = lambda *a, **kw: arq.REFUSED
    host.tick()
    repeat(host)
    assert host.arq._upgrade_requests == 1
    assert len(keyed.p3) == 1
    repeat(host)
    assert host.arq._upgrade_requests == 1


def test_complete_quiet_windows_still_spend_the_silence_budget():
    arm, row = '1003-0004', ARMS['1003-0004']['rows'][0]
    recording(arm)
    host, _ = granted()
    host._entry_grant = None
    rx, g = onair._SessionRx(host, tag='QUIET'), grid(arm, row)
    for _ in range(arq.UPGRADE_SILENCE_CYCLES):
        rx.new_cycle()
        end = collected_end(rx, g, row)
        rx.control_signal_in(np.zeros(end - row['origin']), row['origin'], g)
        host.tick()
    assert host.protocol == spec.Protocol.PACTOR1
    assert any('silence rule' in s for s in host.log_lines)


@pytest.mark.parametrize('word', [pactor1.CS_ACK_A, pactor1.CS_ACK_B])
def test_p1_ack_words_cannot_acknowledge_the_p3_entry(word):
    arm, row = '1003-0004', ARMS['1003-0004']['rows'][0]
    recording(arm)
    host, _ = granted()
    host._entry_grant = None
    host.no_p3_fallback = True
    rx, g = onair._SessionRx(host, tag='P1'), grid(arm, row)
    packet, queued = host.arq._inflight, bytes(host.arq._outbuf)
    end = collected_end(rx, g, row)
    window = np.zeros(end - row['origin'])
    signal = onair._trim_silence(pactor1.control_signal(word))
    at = g.rx_due(g.keyed_slot) - row['origin']
    window[at:at + signal.size] = signal
    rx.new_cycle()
    rx.control_signal_in(window, row['origin'], g)
    assert host.arq.entry_pending and host.arq.role == arq.ISS
    assert host.arq._inflight is packet and host.arq.tx_seq == 2
    assert bytes(host.arq._outbuf) == queued


def test_a_complete_read_supersedes_an_earlier_partial_bridge():
    arm, row = '1003-0004', ARMS['1003-0004']['rows'][0]
    recording(arm)
    host, _ = granted()
    host._entry_grant = None
    rx, g = onair._SessionRx(host, tag='BRIDGE'), grid(arm, row)
    rx.new_cycle()
    rx.control_signal_in(np.zeros(row['end'] - row['origin']), row['origin'], g)
    assert host.arq._entry_window_truncated
    end = collected_end(rx, g, row)
    rx.control_signal_in(np.zeros(end - row['origin']), row['origin'], g)
    assert not host.arq._entry_window_truncated
    host.tick()
    assert host.arq._unanswered_upgrade == 1


@pytest.mark.parametrize('grant_present', [False, True])
def test_entry_acquisition_can_give_up_one_key_without_weakening_admission(grant_present):
    arm, row = '1003-0004', ARMS['1003-0004']['rows'][0]
    audio = recording(arm)
    host, _ = granted()
    rx, g = onair._SessionRx(host, tag='BUDGET'), grid(arm, row)
    peer, live = host.peer, _Bench(seconds=0)
    peer.live, peer.raster, peer.slot = live, g, g.keyed_slot + 1
    peer.key_instant = lambda raster, slot: raster.boundary(slot)
    peer.listening = False
    live.now = live.pos = (g.boundary(peer.slot) - live.key_notice
                          - round((onair.P3_DECODE_RESERVE_S + .008) * onair.FS))
    assert not rx._p3_acquisition_fits(rx.P3_ANSWER_ACQUIRE_RESERVE_S)
    rx.new_cycle()
    end = collected_end(rx, g, row)
    peer.live = live
    window = audio[row['origin']:end] if grant_present else np.zeros(end - row['origin'])
    rx.control_signal_in(window, row['origin'], g)
    assert (getattr(peer, '_p3_entry_listen_slot', None) == peer.slot) is not grant_present
    if grant_present:
        assert host.arq._upgrade_asked and host.arq.entry_pending
    else:
        assert rx._p3_acquisition_fits(rx.P3_ANSWER_ACQUIRE_RESERVE_S)
        assert live.clamp_late(peer.key_instant(g, peer.slot)
                               - round((.012 + onair.P3_DECODE_RESERVE_S) * onair.FS))
        peer.slot += 1
        live.now = live.pos = (g.boundary(peer.slot) - live.key_notice
                              - round((onair.P3_DECODE_RESERVE_S + .008) * onair.FS))
        assert not rx._p3_acquisition_fits(rx.P3_ANSWER_ACQUIRE_RESERVE_S)


@pytest.mark.parametrize('arm,head,status,data_at,offset', [
    ('1003-0000', 529386, 0, 653706, -14.5),
    ('1002-2349', 1129309, 0x20, 1253629, -12.9),
])
def test_successful_changeover_and_first_mail_data(arm, head, status, data_at, offset):
    audio = recording(arm)
    s = _Session(entry_pending=True)
    s.rx.p3_receive_offset_hz = offset
    s.rx.new_cycle()
    origin = head - 3840
    assert s.rx.control_signal(audio[origin:head + 48000], origin, head,
                               p1_allowed=False) == arq.CS_BREAKIN
    assert s.host.arq.role == arq.IRS and not s.host.arq.entry_pending
    assert any(e.packet == (1, status, b'RMS', True) for e in s.events)
    s.rx.new_cycle()
    onair._scan_frame(s.rx, audio[data_at - 4800:data_at + 48000], data_at - 4800)
    assert any(e.packet == (1, 0x21, b' Trim', True) for e in s.events)
    assert s.rx.p3_receive_offset_hz == pytest.approx(offset, abs=.5)


def test_ve3kpg_last_controls_fit_the_bounded_listening_window(tmp_path, monkeypatch, capsys):
    arm = '1003-0012'
    audio = recording(arm)
    host, _ = granted()
    host.no_p3_fallback = True
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(host)
    host.peer = tx
    transitions = observe_entry_answered(monkeypatch, host)
    packet, queued = host.arq._inflight, bytes(host.arq._outbuf)
    acked_bytes = host.arq.tx_acked_bytes
    live = _Bench(seconds=audio.size / onair.FS + 1, holdback=916)
    live.limit = 1 << 40
    live.audio[:audio.size] = audio
    tx.live = live
    rx = onair._SessionRx(host, tag='TAIL')
    for row in ARMS[arm]['rows'][:-2]:
        g = grid(arm, row)
        rx.new_cycle()
        end = max(collected_end(rx, g, row), row['nominal'] + 5760 + 720)
        rx.control_signal_in(audio[row['origin']:end], row['origin'], g)
    assert len(rx.cs_log) == 7
    assert [(e.kind, e.protocol, e.spare, e.cs) for e in rx.cs_log] == [
        ('unassigned', spec.Protocol.PACTOR1, pactor1.CS_59A, None)] * 7
    candidate = None
    for row in ARMS[arm]['rows'][-2:]:
        g = grid(arm, row)
        tx.aim(g, g.keyed_slot + 1)
        rx.new_cycle()
        end = collected_end(rx, g, row)
        tx.live = live
        live.pos = end
        live.now = max(end, tx.key_instant(g, tx.slot) - live.key_notice
                       - round((onair.P3_DECODE_RESERVE_S + .008) * onair.FS))
        listen_slot = tx.slot
        saved = saved_entry_window(g, row, listen_slot)
        # Keep the original entry identity/bytes until the real reader accepts.
        assert host.arq._inflight is packet and packet.entry and packet.seq == 2
        assert host.arq.tx_seq == 2 and bytes(host.arq._outbuf) == queued
        assert host.arq.tx_acked_bytes == acked_bytes
        assert tx.n == 0 and tx.slots_used == [] and tx.keyed == []
        assert live.emissions == [] and list(tmp_path.glob('*.wav')) == []
        result = onair._read_held_window(
            live, g, tx, rx, tx.slot, audio[row['origin']:end],
            row['origin'], g.boundary(tx.slot))
        heard, _, captured, origin = result
        assert_original_entry_capture(result, saved, audio, live)
        if listen_slot == 17:
            assert getattr(tx, '_p3_entry_listen_slot', None) == listen_slot
            assert tx.entry_listen_due(17)
            candidate = assert_pending_candidate(rx, result, transitions, packet, queued)
            assert tx.n == 0 and tx.slots_used == [] and tx.keyed == []
            assert live.emissions == [] and list(tmp_path.glob('*.wav')) == []
        else:
            assert listen_slot == 18 and candidate is not None
            assert_distinct_entry_ack(rx, result, candidate)
            assert_entry_retired(transitions, host, queued, acked_bytes)
            assert not tx.entry_listen_due(18)
    assert heard == arq.CS_ACK
    assert not host.arq.entry_pending and host.arq.role == arq.ISS
    assert host.arq.state == arq.State.CONNECTED
    assert host.protocol == spec.Protocol.PACTOR3 and host.arq._sl == 1
    for state in transitions[0]:
        assert state['n'] == 0 and state['slots_used'] == ()
        assert state['keyed'] == state['emissions'] == state['wavs'] == ()
    assert rx.p3_transmit_offset_hz == 0
    assert live.emissions == []
    assert tx.transmit is False and tx.keyed == []
    assert tx.n == 1 and tx.slots_used == [19] and tx.slot == 19
    assert not tx.refused and tx.refusal_cause is None
    assert tx.raster is g and g.next_slot(18) == 19
    assert g.boundary(18) == saved['boundary'] == 1082739
    assert tx.boundary == g.boundary(19) == 1142739
    assert onair.KEY_CLAMP_TOL_S == .005
    assert not host.arq.cycle_long
    data = host.arq._inflight
    assert data is not None and data is not packet
    assert not data.entry and not data.breakin and data.status == 0x23
    assert data.seq == host.arq.tx_seq == 3 and data.sl == 1
    assert data.payload == b'pendi' and data.retries == 0 and data.repeats == 0
    assert data.payload + bytes(host.arq._outbuf) == queued
    assert host.arq.tx_acked_bytes == acked_bytes
    wavs = sorted(tmp_path.glob('*.wav'))
    assert wavs == [tmp_path / 'tx_01.wav']
    waveform = rxfront.load_wav(str(wavs[0]))
    assert waveform.size > 0 and np.any(waveform)
    assert g.keyed_slot == 19
    assert g._p3_keyed_reply == (19, 1142739, 1142739 + waveform.size, 60000)
    output = capsys.readouterr().out
    print(output, end='')  # Retain the real decoder/admission/output evidence.
    refusal = ('SLOT 18 GIVEN UP BEFORE THE RENDER. The unrendered burst would '
               'miss boundary 1082739 by +53.6 ms (+53.6 ms on the bare clamp), '
               'past the 5 ms the reader forgives with the 0.0 ms this cycle '
               'still owes the tick and the render; aiming at slot 19 before '
               'choosing its shift.')
    assert output.count(refusal) == 1
    assert output.count('GIVEN UP BEFORE THE RENDER') == 1
    assert output.count('TX[1] SL1 pkt 5B status=0x23') == 1
    assert output.count(' -- dry run -> ') == 1
    assert f' -- dry run -> {wavs[0]}' in output
    assert output.count('TX(dry-run) P3 SL1 DATA short/1.25s seq=3 type=0 '
                        'status=0x23 attempt=1 retry=0 [CRC-ENCODED]') == 1
    assert "TX(dry-run) payload 5B hex=70656e6469 bytes=b'pendi'" in output
    assert 'hex=70656e6469233745' in output
    events = [json.loads(line.removeprefix('tx_event '))
              for line in output.splitlines() if line.startswith('tx_event ')]
    assert len(events) == 1
    assert {key: events[0][key] for key in (
        'event', 'tx_id', 'protocol', 'direction', 'packet_kind', 'speed_level',
        'payload_bytes', 'status_byte', 'long_cycle', 'outcome')} == dict(
            event='waveform', tx_id=1, protocol='PACTOR-3', direction='tx',
            packet_kind='DATA', speed_level=1, payload_bytes=5, status_byte=0x23,
            long_cycle=False, outcome='dry_run')


@pytest.mark.parametrize('arm', ['1003-0000', '1002-2349'])
def test_successful_mail_packet_replay_is_unchanged(arm):
    audio = recording(arm)
    meta = ARMS[arm]['mail_replay']
    packets = []
    for at, level, offset in meta['rows']:
        s = _Session(role=arq.IRS)
        s.host.arq._sl = level
        s.rx.p3_receive_offset_hz = offset
        s.rx.new_cycle()
        origin = max(0, at - 4800)
        onair._scan_frame(s.rx, audio[origin:at + 48000], origin)
        packets.append([
            dict(status=e.packet[1],
                 payload_sha256=hashlib.sha256(e.packet[2]).hexdigest(),
                 payload_bytes=len(e.packet[2]), at=round(e.t * onair.FS))
            for e in s.events if e.packet is not None and e.packet[3]
            and abs(round(e.t * onair.FS) - at) < 960])
    assert hashlib.sha256(json.dumps(packets, sort_keys=True).encode()).hexdigest() == meta['packets_sha256']
    assert sum(p['payload_bytes'] for row in packets for p in row) == meta['payload_bytes']


def resolved_entry_reply(rx, g, row, audio):
    peer = rx.host.peer
    live = _Bench(seconds=0, holdback=916)
    live.audio, live.limit = audio, 1 << 40
    peer.live, peer.raster, peer.slot = live, g, g.keyed_slot + 1
    peer.settle, peer.listening = .04, False
    peer.key_instant = lambda raster, slot: raster.boundary(slot)
    peer.still_owed_n = lambda: 0
    target = rx.control_collect_until(row['until'], row['origin'], g)
    end = target - live.holdback
    live.pos, live.now = end, target
    saved = saved_entry_window(g, row, peer.slot)
    result = onair._read_held_window(live, g, peer, rx, peer.slot,
                                   audio[row['origin']:end], row['origin'],
                                   g.boundary(peer.slot))
    return result, saved, live
