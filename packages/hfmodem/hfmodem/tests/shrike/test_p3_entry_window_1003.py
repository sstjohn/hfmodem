# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
import time
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.tests import evidence

from hfmodem.shrike import arq, onair, p3acquire, pactor1, placement, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_granted_entry_retry import granted, repeat
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

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
        new.rx.control_signal_in(audio[row['origin']:end], row['origin'], g)
        events = [(e.kind, e.cs, e.spare, str(e.protocol)) for e in new.events]
        if g.rx_due_in(row['origin'], end) is None:
            # The P1 reply runs past our next key: the read stops at that key,
            # which goes out, and the P1 word is left to be read behind it.
            assert end + 916 == row['until']
            assert events == [tuple(e) for e in row['new_events'] if e[3] != 'PACTOR-1']
        else:
            assert g.rx_due_in(row['origin'], end) == row['nominal']
            assert events == [tuple(e) for e in row['new_events']]


@pytest.mark.parametrize('arm,known_grant,count,confirmed,level', [
    ('1003-0004', False, 9, False, 1), ('1003-0004', True, 9, False, 1),
    ('1003-0007', False, 8, False, 1), ('1003-0007', True, 8, False, 1),
    ('1003-0008', False, 0, True, 2), ('1003-0008', True, 0, True, 2),
    ('1003-0010', False, 0, True, 1), ('1003-0010', True, 7, True, 1),
    ('1003-0012', False, 3, True, 1), ('1003-0012', True, 7, True, 1),
])
@pytest.mark.wall_clock
def test_recorded_entry_campaign(arm, known_grant, count, confirmed, level):
    """These arms flew with the entry listen; here every entry slot keys, on
    the FTDX10's clock, and each reply is read to the next entry's key. The
    first bridge holds 13-15 symbols of a P3 answer and aims the read at the
    key, which takes the whole word: 0012's first ACK at 21.031 s and its
    distinct ACK where it repeats, inside the last block before the next key."""
    audio = recording(arm)
    host, _ = granted()
    if not known_grant:
        host._entry_grant = None
    host.no_p3_fallback = True
    tx = ftdx10(host, audio)
    packet, queued = host.arq._inflight, bytes(host.arq._outbuf)
    for row in ARMS[arm]['rows']:
        keyed, margin = held_cycle(tx, grid(arm, row), row['slot'] + 1)
        assert keyed, 'every entry slot keys'
        assert margin is None or margin > 0, 'decided past the notice'
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
def test_an_entry_short_of_acquisition_time_still_keys_its_next_slot(grant_present):
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
    # No slot is given to the reply's acquisition: the next entry keys.
    assert not peer.listening
    assert not rx._p3_acquisition_fits(rx.P3_ANSWER_ACQUIRE_RESERVE_S)
    if grant_present:
        assert host.arq._upgrade_asked and host.arq.entry_pending


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


FTDX10 = dict(holdback=916, lat_in=916 - 128)
"""The FTDX10's stream: 916 samples of holdback, all but a block of them the
input latency (`_LiveInput.holdback`)."""


def ftdx10(host, audio):
    """A keying transmitter on the FTDX10's clock, reading `audio`."""
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=Path('/unused'),
                       settle=.04, record_audio=False)
    tx.attach(host)
    host.peer = tx
    tx.live = _Bench(seconds=0, **FTDX10)
    tx.live.audio, tx.live.limit = np.asarray(audio, np.float32), 1 << 40
    tx.defer_p3_cs = True
    tx.sessrx = onair._SessionRx(host, tag='FTDX10')
    return tx


def charged(live, work, *args):
    started = time.perf_counter()
    try:
        return work(*args)
    finally:
        live.spend(round((time.perf_counter() - started) * onair.FS))


def held_cycle(tx, g, slot):
    """One held-link cycle in the hold loop's order and on its deadlines: the
    first bridge `d` in front of the key and the sweep there, the read to the
    key, the regrid and the tick, each charged to the bench clock at what it
    costs here. Returns the audio keyed, and the room left in front of the
    key's notice for the tick where the read itself did not key. A slower or
    busier machine keys a later slot, so every test keyed through here is
    wall_clock."""
    live, rx, host, settle = tx.live, tx.sessrx, tx.host, 1920
    tx.aim(g, slot)
    tx.tx_end = g._p3_keyed_reply[2]
    rx.new_cycle()
    origin = live.pos = tx.tx_end
    live.now = max(live.now, origin + live.lat_in)
    key = tx.key_instant(g, slot)
    final = key - onair._prekey_lead(live, settle)
    bridge = rx.control_bridge_until(key - max(g.d, settle), final, origin, g)
    whole, origin, _ = onair._collect(live, rx, np.empty(0, np.float32), origin, bridge)
    charged(live, rx.acquire_ahead, whole, origin, g)
    until = rx.control_collect_until(final, max(live.pos, int(live.sample_now())), g)
    whole, origin, _ = onair._collect(live, rx, whole, origin, until)
    before = len(live.emissions)
    _, _, whole, origin = charged(live, onair._read_held_window,
                                  live, g, tx, rx, slot, whole, origin)
    if rx._p3_entry_body_candidate is not None and rx._entry_repeat_eligible(g):
        whole, origin = charged(live, onair._finish_entry_candidate,
                                live, g, tx, host, rx, slot, whole, origin, settle)
    whole = charged(live, onair._receive_changeover, live, g, tx, host, rx, slot,
                    whole, origin, np.empty(0, np.float32), 0, settle)
    charged(live, onair._reverse_before_key, g, host, tx, slot)
    slot, whole, origin = charged(live, onair._regrid,
                                  live, g, tx, host, rx, slot, whole, origin, settle)
    margin = (None if len(live.emissions) > before else
              (tx.key_instant(g, slot) - live.key_notice - live.now) / onair.FS)
    charged(live, host.tick)
    return live.emissions[before:], margin


SLOT = 60000


def p3_grid(keyed_slot):
    g = onair._MasterGrid(0, SLOT, 8880, packet_n=46080, cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.d_n, g.d_ref_n = round(.111 * onair.FS), 46080
    g.keyed_slot = keyed_slot
    g._p3_keyed_reply = (keyed_slot, keyed_slot * SLOT + 234,
                         keyed_slot * SLOT + 40712, SLOT)
    return g


def rendered(audio, head, wave, hz):
    """`wave` at `hz` with its acquired head at sample `head` of `audio`."""
    shifted = p3acquire.compensate(np.pad(wave, (4800, 4800)), -hz)
    got = p3acquire.control_signal(shifted, offsets=p3acquire.ENTRY_CONTROL_OFFSETS_HZ)
    at = head - got.event.start
    audio[at:at + shifted.size] = shifted
    return got


@pytest.mark.wall_clock
def test_a_changeover_a_cycle_after_a_bare_answer_is_read_before_the_key():
    """An ACK at 103,200 and the RMS changeover one short cycle later, both
    at -50 Hz: the entry that answers the ACK keys, and the next one does not
    go out over the changeover's CRC-valid body at 163,200-202,080."""
    host, _ = granted()
    audio = np.zeros(7 * onair.FS, np.float32)
    rendered(audio, 103200, placement.control_signal(arq.CS_ACK), -50)
    body = rendered(audio, 163200, placement.changeover_packet(b'RMS', 0), -50)
    assert body.event.packet == (1, 0, b'RMS', True)
    tx = ftdx10(host, audio)
    keyed, _ = held_cycle(tx, p3_grid(1), 2)
    assert keyed and host.arq.entry_pending
    held_cycle(tx, p3_grid(2), 3)
    assert not [k for k in tx.keyings if k[0] < 202080 and k[1] > 163200]
    assert host.arq.role == arq.IRS and not host.arq.entry_pending
    assert bytes(host.channel(host.ptchn).rx) == b'RMS'


AIMED = {  # (arm, slot): the word read at the key where the bridge aimed
    ('1003-0000', 8): 'BREAK-IN', ('1003-0008', 15): 'SPEED-UP',
    ('1003-0008', 16): 'SPEED-UP', ('1003-0008', 17): 'SPEED-UP',
    ('1003-0008', 18): 'SPEED-UP', ('1003-0008', 19): 'SPEED-UP',
    ('1003-0008', 20): 'SPEED-UP', ('1003-0008', 21): 'ACK',
    ('1003-0008', 22): 'NAK', ('1003-0008', 23): 'BREAK-IN',
    ('1003-0010', 19): 'ACK', ('1003-0012', 16): 'ACK', ('1003-0012', 17): 'ACK',
    # Started 986 ms past our boundary: its word ends behind our key.
    ('1003-0010', 20): None,
}


def test_the_bridge_aims_the_key_s_read_at_every_recorded_p3_answer_and_nowhere_else():
    """On the FTDX10's clock the first bridge ends 1171.6 ms past our boundary
    and the read at the key 1191.6: every P3 answer of these arms is aimed at
    there and read whole at one frequency where its word ends in front of the
    key, and no PACTOR-1 answer draws an aim."""
    for arm in ('1003-0000', '1003-0004', '1003-0007', '1003-0008', '1003-0010', '1003-0012'):
        audio = recording(arm)
        for row in ARMS[arm]['rows']:
            b = grid(arm, row).boundary(row['slot'])
            lo, bridge, end = b + round(.851 * onair.FS), b + 56236, b + 57197
            aims = p3acquire.head_aims(audio[lo:bridge])
            if (arm, row['slot']) not in AIMED:
                assert aims == [], (arm, row['slot'])
                continue
            head, hz = aims[0]
            window = audio[end - round(.4 * onair.FS):end]
            ev = rxfront.SyncedRx(memory={}).control_signal_tracked(
                p3acquire.compensate(window, hz), lo + head - (end - window.size))
            word = None if ev is None else spec.CS_NAMES[ev.cs]
            assert word == AIMED[arm, row['slot']], (arm, row['slot'])


@pytest.mark.parametrize('head_ms,d_ms', [(986, 111), (1005, 111), (1020, 111),
                                           (1030, 111), (1045, 111),
                                           (1002, 134), (1040, 134), (1045, 134)])
@pytest.mark.wall_clock
def test_a_changeover_the_key_cuts_short_is_read_before_the_next_entry(head_ms, d_ms):
    """An RMS changeover whose word ends behind the key's read: 986 ms past our
    boundary is the latest of these arms' answers, and a peer turning round in
    134 ms answers P3 about 1002. 1045 ms is the latest head whose first fourteen
    bits the read in front of the key holds. The bits there are CS3's alone, and
    the next entry does not key over its body."""
    host, _ = granted()
    audio = np.zeros(5 * onair.FS, np.float32)
    head = SLOT + round(head_ms / 1000 * onair.FS)
    body = rendered(audio, head, placement.changeover_packet(b'RMS', 0), -50)
    assert body.event.packet == (1, 0, b'RMS', True)
    tx = ftdx10(host, audio)
    g = p3_grid(1)
    g.d_n = round(d_ms / 1000 * onair.FS)
    held_cycle(tx, g, 2)
    end = head + round(placement.PACKET_S * onair.FS)
    assert not [k for k in tx.keyings if k[0] < end and k[1] > head]
    assert host.arq.role == arq.IRS and not host.arq.entry_pending
    assert bytes(host.channel(host.ptchn).rx) == b'RMS'


@pytest.mark.parametrize('head_ms', [950, 975])
@pytest.mark.parametrize('symbols', [16, 19, None])
@pytest.mark.wall_clock
def test_an_answer_cut_off_inside_its_word_confirms_no_entry(head_ms, symbols):
    """Two ACKs a cycle apart, the second gone `symbols` after its head, short
    of its 21: a word read only in part corroborates nothing, and the next slot
    keys the entry again at its own instant. Whole, the same ACK confirms it,
    and the slot keys our first data packet."""
    host, _ = granted()
    audio = np.zeros(7 * onair.FS, np.float32)
    head = round(head_ms / 1000 * onair.FS)
    rendered(audio, SLOT + head, placement.control_signal(arq.CS_ACK), -50)
    rendered(audio, 2 * SLOT + head, placement.control_signal(arq.CS_ACK), -50)
    if symbols is not None:
        audio[2 * SLOT + head + symbols * rxfront.SPS:] = 0
    tx = ftdx10(host, audio)
    held_cycle(tx, p3_grid(1), 2)
    keyed, _ = held_cycle(tx, p3_grid(2), 3)
    assert host.arq.entry_pending is (symbols is not None)
    assert keyed and keyed[0][0] == 3 * SLOT + (234 if symbols is not None else 0)


@pytest.mark.wall_clock
def test_a_symbol_faded_inside_a_changeover_s_head_is_read_before_the_next_entry():
    """An RMS changeover 975 ms past our boundary with its tenth symbol faded
    to a twentieth: the word is still read whole, and the next entry does not
    key over its body."""
    host, _ = granted()
    audio = np.zeros(7 * onair.FS, np.float32)
    rendered(audio, 106800, placement.changeover_packet(b'RMS', 0), -50)
    audio[111360:111840] *= .05
    tx = ftdx10(host, audio)
    held_cycle(tx, p3_grid(1), 2)
    end = 106800 + round(placement.PACKET_S * onair.FS)
    assert not [k for k in tx.keyings if k[0] < end and k[1] > 106800]
    assert bytes(host.channel(host.ptchn).rx) == b'RMS'


@pytest.mark.wall_clock
def test_a_symbol_faded_inside_a_corroborating_ack_still_confirms_the_entry():
    """The second of two ACKs 975 ms past our boundary with its tenth symbol
    faded to a twentieth: a fade is not a cut, and the slot keys our first data
    packet."""
    host, _ = granted()
    audio = np.zeros(7 * onair.FS, np.float32)
    rendered(audio, 106800, placement.control_signal(arq.CS_ACK), -50)
    rendered(audio, 166800, placement.control_signal(arq.CS_ACK), -50)
    audio[171360:171840] *= .05
    tx = ftdx10(host, audio)
    held_cycle(tx, p3_grid(1), 2)
    keyed, _ = held_cycle(tx, p3_grid(2), 3)
    assert not host.arq.entry_pending
    assert keyed and keyed[0][0] == 3 * SLOT


@pytest.mark.parametrize('end', [116552, 116900])
@pytest.mark.wall_clock
def test_a_burst_that_ends_before_the_key_refuses_no_slot(end):
    """Tones on both control carriers from where an answer starts, gone before
    our carrier comes up: not the peer on the air, and the entry keys."""
    host, _ = granted()
    audio = np.random.default_rng(1107).normal(0, .002, 5 * onair.FS).astype(np.float32)
    t = np.arange(104552, end)
    audio[t] += .1 * (np.sin(2 * np.pi * 1080 * t / onair.FS)
                      + np.sin(2 * np.pi * 1920 * t / onair.FS))
    tx = ftdx10(host, audio)
    keyed, _ = held_cycle(tx, p3_grid(1), 2)
    assert keyed and keyed[0][0] == 2 * SLOT + 234


@pytest.mark.wall_clock
def test_a_bare_answer_running_into_the_key_s_read_refuses_no_slot():
    """An ACK 986 ms past our boundary on a noiseless channel, still on the air
    where the read stops: the entry keys."""
    host, _ = granted()
    audio = np.zeros(5 * onair.FS, np.float32)
    rendered(audio, SLOT + round(.986 * onair.FS), placement.control_signal(arq.CS_ACK), -50)
    tx = ftdx10(host, audio)
    keyed, _ = held_cycle(tx, p3_grid(1), 2)
    assert keyed and keyed[0][0] == 2 * SLOT + 234
