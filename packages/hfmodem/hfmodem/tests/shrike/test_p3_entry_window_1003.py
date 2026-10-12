import hashlib
import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, pactor1, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_granted_entry_retry import granted, repeat
from hfmodem.tests.shrike.test_grid import _Bench

ROOT = Path(__file__).resolve().parents[5]
MANIFEST = Path(__file__).with_name('fixtures') / 'p3-entry-window-1003.json'
ARMS = {a['arm']: a for a in json.loads(MANIFEST.read_text())}


@cache
def recording(arm):
    meta = ARMS[arm]
    folder = ROOT / 'captures' / ('onair-' + arm)
    for name, digest in meta['sha256'].items():
        path = folder / name
        if not path.exists():
            pytest.skip(f'recording absent: {path}')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return rxfront.load_wav(str(folder / 'stream.flac'))


def grid(arm, row):
    folder = ROOT / 'captures' / ('onair-' + arm)
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
def test_recorded_entry_campaign(arm, count, confirmed, level):
    audio = recording(arm)
    host, _ = granted()
    host.no_p3_fallback = True
    rx = onair._SessionRx(host, tag='ENTRY')
    packet, queued = host.arq._inflight, bytes(host.arq._outbuf)
    for row in ARMS[arm]['rows']:
        rx.new_cycle()
        g = grid(arm, row)
        end = collected_end(rx, g, row)
        rx.control_signal_in(audio[row['origin']:end], row['origin'], g)
        host.tick()
        if not host.arq.entry_pending:
            break
        assert host.arq._inflight is packet
        assert bytes(host.arq._outbuf) == queued
        assert host.arq.role == arq.ISS
        assert host.arq.state == arq.State.CONNECTED
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
        assert host.arq._upgrade_requests == 0
        assert host.arq._inflight.retries == 0


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


def test_ve3kpg_last_controls_fit_the_bounded_listening_window(tmp_path):
    arm = '1003-0012'
    audio = recording(arm)
    host, _ = granted()
    host.no_p3_fallback = True
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(host)
    host.peer = tx
    live = _Bench(seconds=audio.size / onair.FS + 1, holdback=916)
    live.limit = 1 << 40
    live.audio[:audio.size] = audio
    tx.live = live
    rx = onair._SessionRx(host, tag='TAIL')
    for row in ARMS[arm]['rows'][:-2]:
        g = grid(arm, row)
        rx.new_cycle()
        end = collected_end(rx, g, row)
        rx.control_signal_in(audio[row['origin']:end], row['origin'], g)
    assert len(rx.cs_log) == 7
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
        reply_end = g._p3_keyed_reply[1] + 60000
        heard, _, captured, origin = onair._read_held_window(
            live, g, tx, rx, tx.slot, audio[row['origin']:end],
            row['origin'], g.boundary(tx.slot))
        assert getattr(tx, '_p3_entry_listen_slot', None) == listen_slot
        assert origin + captured.size >= reply_end
    assert heard == arq.CS_ACK
    assert not host.arq.entry_pending and host.arq.role == arq.ISS
    assert rx.p3_transmit_offset_hz == 0
    assert live.emissions == []


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
