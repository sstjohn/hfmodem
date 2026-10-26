# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
from functools import cache
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p1rx, pactor1, rxfront, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_granted_entry_retry import granted
from hfmodem.tests.shrike.test_grid import _Bench
from hfmodem.tests.shrike.test_late_entry_loop import run_late, MESSAGE, SLOT

FS = onair.FS
FOLDER = evidence.CAPTURES / 'onair-20261010T234613641109Z-f8689e8a-9847-4b5c-a8b7-988694aa49ef'


@cache
def stream():
    meta = json.loads(evidence.require_input(FOLDER / 'stream.json').read_text())
    assert meta['artifact_path'] == 'stream.wav' and meta['wav'] == 'stream.flac'
    path = evidence.require_input(FOLDER / meta['wav'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['container_sha256'] == '40bd73f8fa6172e487db610cdc7d05ac205005139c6c121767a713e8ed0f3671'
    return rxfront.load_wav(str(path))


def grant_pcm():
    clip = stream()[1003443:1023488].copy()
    anchor = 1014121 - 1003443
    assert p1rx.cs_anchored(clip, anchor / FS) == p1rx.CS(5, 0, 0)
    return clip, anchor


def grant_train():
    clip, anchor = grant_pcm()
    inverse = stream()[1063443:1083488].copy()
    assert p1rx.cs_anchored(inverse, anchor / FS) == p1rx.CS(5, 0, 1)
    return (clip, inverse), anchor


@pytest.mark.parametrize('sense', [0, 1])
@pytest.mark.parametrize('bits', [9, 10, 11, 12])
def test_recorded_expected_grant(bits, sense):
    clips, anchor = grant_train()
    clip = clips[sense]
    got = p1rx.entry_grant_match(clip[:anchor + bits * 480], anchor / FS)
    assert got is not None
    assert 9 <= got.observed_bits <= bits
    assert got.sense == sense and not got.complete
    assert abs(got.at - anchor / FS) <= p1rx.CS_SEARCH_HALF_S


@pytest.mark.parametrize('kind', ['wrong', 'ack', 'corrupt', 'eight', 'noise', 'quiet', 'gap', 'ambiguous'])
def test_expected_grant_rejects(kind):
    signal = onair._trim_silence(pactor1.control_signal(pactor1.CS_59A))
    signal = np.pad(signal, (960, 0))[:960 + 9 * 480]
    if kind in ('wrong', 'ack'):
        word = pactor1.CS_6A9 if kind == 'wrong' else pactor1.CS_ACK_A
        signal = np.pad(onair._trim_silence(pactor1.control_signal(word)), (960, 0))[:signal.size]
    elif kind == 'corrupt':
        signal[960 + 4 * 480:960 + 5 * 480] = .11 * np.sin(2*np.pi*pactor1.SPACE*np.arange(480)/FS)
    elif kind == 'eight':
        signal = signal[:960 + 8 * 480]
    elif kind == 'noise':
        signal = np.random.default_rng(1010).normal(0, .02, signal.size)
    elif kind == 'quiet':
        signal[:] = 0
    elif kind == 'gap':
        signal[960 + 3 * 480:960 + 4 * 480] = 0
    elif kind == 'ambiguous':
        signal = np.pad(onair._trim_silence(pactor1.control_signal(pactor1.CS_59A)), (960, 0))[:960 + 9 * 480]
        signal += np.pad(onair._trim_silence(pactor1.control_signal(pactor1.CS_6A9)), (960, 0))[:signal.size]
    assert p1rx.entry_grant_match(signal, 960 / FS) is None


def setup_entry():
    host, _ = granted()
    rx = onair._SessionRx(host, tag='PIN')
    g = onair._MasterGrid(0, SLOT, 8880, packet_n=46080, cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.d_n, g.d_ref_n = round(.111 * FS), 46080
    g.keyed_slot = 1
    g._p3_keyed_reply = (1, SLOT + 234, SLOT + 40712, SLOT)
    host.peer.raster, host.peer.slot = g, 2
    host.peer.live = SimpleNamespace(holdback=916, lost=0)
    rx.new_cycle()
    at = g.rx_due(1)
    clip, anchor = grant_pcm()
    origin = at - anchor
    return host, rx, g, clip, origin, at


def test_partial_owned_repeat_order_credit_and_complete_supersession():
    host, rx, g, clip, origin, at = setup_entry()
    end = at + 10 * 480
    order = []
    control, match = rx.control_signal, p1rx.entry_grant_match
    def read(*a, **kw):
        order.append('p3' if not kw.get('p1_allowed', True) else 'p1')
        return control(*a, **kw)
    def matching(*a, **kw):
        order.append('match')
        return match(*a, **kw)
    packet, queued, seq = host.arq._inflight, bytes(host.arq._outbuf), host.arq.tx_seq
    with patch.object(rx, 'control_signal', read), patch.object(p1rx, 'entry_grant_match', matching):
        for _ in range(3):
            rx.control_signal_in(clip[:end-origin], origin, g)
    assert order.index('p3') < order.index('match')
    assert host.arq._entry_window_truncated and not host.arq._entry_p1_current
    assert not host.prepare_p1_entry_goodbye()
    host.tick()
    assert host.arq._upgrade_requests == 1 and host.arq._unanswered_upgrade == 0
    assert host.arq._inflight is packet and bytes(host.arq._outbuf) == queued
    assert host.arq.tx_seq == seq and host.arq.role == arq.ISS
    rx.control_signal_in(clip, origin, g)
    assert not host.arq._entry_window_truncated
    assert host.arq._upgrade_requests == 1
    rx.control_signal_in(clip[:end-origin], origin, g)
    assert not host.arq._entry_window_truncated


@pytest.mark.parametrize('condition', ['missing', 'wrong_slot', 'unkeyed', 'stale', 'gap', 'terminal', 'answered', 'p2', 'receiving'])
def test_repeat_eligibility_boundaries(condition):
    host, rx, g, clip, origin, at = setup_entry()
    if condition == 'missing':
        host._entry_grant = None
    elif condition == 'wrong_slot':
        g.keyed_slot += 1
    elif condition == 'unkeyed':
        g._p3_keyed_reply = None
    elif condition == 'stale':
        origin = at + 1
    elif condition == 'gap':
        host.peer.live.lost += 480
    elif condition == 'terminal':
        host.arq.on_host_disconnect()
    elif condition == 'answered':
        host._entry_answered()
    elif condition == 'p2':
        host.protocol = spec.Protocol.PACTOR2
    elif condition == 'receiving':
        g.sending, host.arq.role = False, arq.IRS
    assert not rx._entry_repeat_eligible(g, origin)


@pytest.mark.parametrize('d,options', [(.111, ()), (.1058, ()), (.111, ('--no-p3-entry-stagger',)), (.111, ('--no-p3-rise',))])
def test_three_recorded_grant_entries_are_consecutive(monkeypatch, tmp_path, d, options):
    clip, anchor = grant_train()
    got = run_late(monkeypatch, tmp_path, 3, False, d=d,
                   grant_pcm=clip, grant_anchor=anchor, options=options,
                   acquire_cost=.012, match_cost=.001)
    entries = got['peer'].entries
    assert len(entries) == 3
    assert np.diff(entries).tolist() == [SLOT, SLOT]
    assert 'partial entry grant' in got['log']
    assert len(got['entry_reads']) >= 2
    for n, r in enumerate(got['entry_reads'][:2]):
        assert r['d'] == pytest.approx(d, abs=p1rx.CS_SEARCH_HALF_S)
        assert r['truncated'] and not r['current'] and not r['owed']
        assert r['silence'] == 0 and r['requests'] == n
        assert r['sense'] == n % 2
        assert r['keyed'][1] in entries[:2]
    assert {'p3', 'match'} <= {kind for kind, _ in got['entry_work']}
    order = [kind for kind, _ in got['entry_work']]
    assert order.index('p3') < order.index('match')
    assert got['peer'].p3_phase == entries[2] + round(.90*FS)
    assert got['entry_heads']
    keyed, head = got['entry_heads'][0]
    assert keyed[1] == entries[2] and keyed[2] <= head < keyed[1] + keyed[3]
    assert 'changeover -> IRS' in got['log']
    assert got['host'].entry_result is not None
    before, after = got['host'].entry_result
    assert before == after and MESSAGE.endswith(before)
    assert bytes(got['host'].channel(got['host'].ptchn).rx) == b'RMS Trimode 1.4.3.0\r\nW9SSJ has 118 d'


def test_slot35_mismatch_listen_keeps_candidate_and_distinct_answer(tmp_path):
    audio = stream()
    txmeta = json.loads((FOLDER / 'tx_26.json').read_text())
    assert (txmeta['slot'], txmeta['audio_start'], txmeta['audio_end'], txmeta['sha256']) == (35, 2102785, 2143263, '2a97e4d824d20eee993bf842a2c907b2ef960ef37979e780bdaa2caea26da2cb')
    host, _ = granted()
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(host)
    host.peer = tx
    live = _Bench(seconds=51, holdback=916)
    live.limit = 1 << 40
    live.audio[:51*FS] = audio[:51*FS]
    tx.live = live
    boundary = 2102551
    g = onair._MasterGrid(boundary - 35*SLOT, SLOT, 8880, packet_n=46080, cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.d_n, g.d_ref_n = round(.111*FS), 46080
    g.keyed_slot = 35
    g._p3_keyed_reply = (35, 2102785, 2143263, SLOT)
    tx.aim(g, 36)
    rx = onair._SessionRx(host, tag='S35')
    rx.new_cycle()
    origin = 2143263
    end = boundary + round(1.170*FS)
    live.pos, live.now = end, end + 916
    with patch.object(p1rx, 'entry_grant_match', wraps=p1rx.entry_grant_match) as matcher:
        assert rx.control_signal_in(audio[origin:end], origin, g) == (None, None)
        assert rx._p3_head_candidate is None and host.arq.entry_pending
        heard, _, captured, start = onair._read_held_window(live, g, tx, rx, 36, audio[origin:end], origin, g.boundary(36))
        assert matcher.call_count == 1
    assert tx._p3_entry_listen_slot == 36
    assert start + captured.size >= 2102785 + SLOT
    assert heard is None and host.arq.entry_pending and host.arq.role == arq.ISS
    assert rx._p3_head_candidate[1] / FS == pytest.approx(44.7788125, abs=1/FS)
    assert rx._p3_head_candidate[2] == pytest.approx(-51.6, abs=.1)
    rx.new_cycle()
    rx.control_signal_in(captured, start, g)
    assert host.arq.entry_pending and host.arq.role == arq.ISS
    follow = json.loads((FOLDER / 'tx_27.json').read_text())
    assert (follow['slot'], follow['audio_start'], follow['audio_end']) == (37, 2222785, 2263263)
    g.keyed_slot = follow['slot']
    g._p3_keyed_reply = (follow['slot'], follow['audio_start'], follow['audio_end'], SLOT)
    tx.aim(g, 38)
    rx.new_cycle()
    origin = follow['audio_end']
    rx.control_signal_in(audio[origin:follow['audio_start']+SLOT+916], origin, g)
    assert not host.arq.entry_pending and host.arq.role == arq.IRS
    assert rx.cs_at / FS == pytest.approx(47.28, abs=.004)


@pytest.mark.parametrize('no_fallback', [False, True])
def test_partial_campaign_is_bounded_without_terminal_p1_proof(no_fallback):
    host, _ = granted()
    host.no_p3_fallback = no_fallback
    packet, queued, seq = host.arq._inflight, bytes(host.arq._outbuf), host.arq.tx_seq
    for n in range(arq.ENTRY_GRANT_CYCLES):
        host.note_entry_grant_repeat()
        assert host.arq._entry_window_truncated
        assert not host.arq._entry_p1_current
        assert not host.prepare_p1_entry_goodbye()
        host.tick()
        assert host.arq._upgrade_requests == n + 1
        if n + 1 < arq.ENTRY_GRANT_CYCLES:
            assert host.arq._unanswered_upgrade == 0
            assert host.arq._inflight is packet and host.arq.tx_seq == seq
            assert bytes(host.arq._outbuf) == queued and host.arq.role == arq.ISS
    assert any('partial repeat' in s for s in host.log_lines)
    if no_fallback:
        assert host.arq.state == arq.State.DISCONNECTED
    else:
        assert host.protocol == spec.Protocol.PACTOR1
    assert not host.arq._entry_p1_close


def test_accepted_p3_wins_without_partial_matching():
    host, rx, g, clip, origin, at = setup_entry()
    def accepted(*args, **kwargs):
        assert not kwargs.get('p1_allowed', True)
        host._entry_answered()
        return arq.CS_ACK
    with patch.object(rx, 'control_signal', accepted), patch.object(p1rx, 'entry_grant_match') as matcher:
        assert rx.control_signal_in(clip[:at+10*480-origin], origin, g)[0] == arq.CS_ACK
        assert not matcher.called
    assert not host.arq.entry_pending


def test_collection_preserves_unknown_grant_control_and_bridge_plan():
    host, rx, g, _, origin, _ = setup_entry()
    until = g.boundary(2) - round(.06*FS)
    expected = until - round(rx.P3_ANSWER_ACQUIRE_RESERVE_S*FS)
    assert rx.control_collect_until(until, origin, g) == expected
    assert getattr(host.peer, '_p3_entry_listen_slot', None) is None
    assert rx.control_bridge_until(until-200, until, origin, g) == expected
    assert rx.control_collect_until(until, expected, g) == expected
    host._entry_grant = None
    rx._p3_control_collect_plan = None
    ready = rx.control_collect_until(until, origin, g)
    assert ready >= g.rx_due(1)+5760+916
    assert host.peer._p3_entry_listen_slot == 2


def test_old_policy_first_three_keys_are_two_cycles_apart(monkeypatch, tmp_path):
    clip, anchor = grant_train()
    with patch.object(onair._SessionRx, '_entry_repeat_eligible', return_value=False, create=True):
        got = run_late(monkeypatch, tmp_path, 3, False, d=.111,
                       grant_pcm=clip, grant_anchor=anchor, acquire_cost=.012,
                       match_cost=.001)
    assert len(got['peer'].entries) >= 3
    assert np.diff(got['peer'].entries[:3]).tolist() == [2*SLOT, 2*SLOT]


def test_matching_repeat_cannot_clear_existing_listen_or_cede():
    host, rx, g, clip, origin, at = setup_entry()
    host.peer._p3_entry_listen_slot = host.peer.slot
    host.peer.listening = host.peer.cede_listening = True
    rx.control_signal_in(clip[:at+10*480-origin], origin, g)
    assert host.peer._p3_entry_listen_slot == host.peer.slot
    assert host.peer.listening and host.peer.cede_listening
    assert host.arq._entry_window_truncated


def test_full_flush_of_partial_credit_cannot_spend_the_next_emission():
    host, rx, g, clip, origin, at = setup_entry()
    rx.control_signal_in(clip[:at+10*480-origin], origin, g)
    host.tick()
    assert host.arq._upgrade_requests == 1 and host.arq._upgrade_reply_owed
    g.keyed_slot += 1
    old = g._p3_keyed_reply
    g._p3_keyed_reply = (g.keyed_slot, old[1]+SLOT, old[2]+SLOT, SLOT)
    host.peer.slot += 1
    rx.new_cycle()
    event = rx._p1_cs(clip, at-origin)
    rx._on(onair.replace(event, t=event.t+origin/FS))
    assert host.arq._upgrade_reply_owed and not host.arq._upgrade_owned_grant
    assert not host.arq._entry_p1_current
    assert host.arq._entry_window_truncated
    host.tick()
    assert host.arq._upgrade_requests == 1 and host.arq._unanswered_upgrade == 0


def test_late_match_work_gives_up_the_key_on_the_original_grid(monkeypatch, tmp_path):
    clip, anchor = grant_train()
    got = run_late(monkeypatch, tmp_path, 3, False, d=.111,
                   grant_pcm=clip, grant_anchor=anchor, acquire_cost=.05,
                   match_cost=.001)
    entries = got['peer'].entries
    assert len(entries) >= 3 and any(n > SLOT for n in np.diff(entries[:3]))
    assert all((first-entries[0]) % SLOT == 0 for first in entries)
    assert 'IS GONE' in got['log'] or 'LATE TO THE KEY' in got['log']


@pytest.mark.parametrize('options', [('--no-p3-rise',), ('--no-p3-entry-stagger',)])
def test_mail_keeps_waveform_requirements(monkeypatch, tmp_path, options):
    with pytest.raises(SystemExit, match='--p3-mail requires'):
        run_late(monkeypatch, tmp_path, 3, False, options=('--p3-mail', *options))


@pytest.mark.parametrize('sense', [0, 1])
def test_repeat_match_uses_observed_prefix_weight_in_both_senses(sense):
    signal = onair._trim_silence(pactor1.control_signal(pactor1.CS_59A, invert=bool(sense)))
    signal = np.pad(signal, (960, 0))[:960+9*480]
    match = p1rx.entry_grant_match(signal, 960/FS)
    assert match is not None and match.observed_bits == 9 and match.sense == sense
    assert not match.complete
    assert p1rx.cs_anchored(signal, 960/FS) is None


@pytest.mark.parametrize('condition', ['uninvited', 'robust', '6a9', 'fallback', 'reset'])
def test_entry_grant_state_boundaries(condition):
    host, rx, g, _, origin, _ = setup_entry()
    if condition == 'uninvited':
        host._uninvited_entry = True
    elif condition == 'robust':
        host.robust_entry_from_bridge = True
    elif condition == '6a9':
        host.p1_6a9 = True
    elif condition == 'fallback':
        host.fall_back('pin fallback')
        assert host._entry_grant is None
    elif condition == 'reset':
        host.arq._finish_disconnected()
        assert host._entry_grant is None
    assert not rx._entry_repeat_eligible(g, origin)


def test_partial_repeat_cannot_schedule_a_later_unkeyed_decision():
    host, rx, g, _, origin, _ = setup_entry()
    host.peer.slot += 1
    assert not rx._entry_repeat_eligible(g, origin)


@pytest.mark.parametrize('no_fallback', [False, True])
def test_complete_wrong_p1_ack_cannot_block_the_listen_p3_retry(no_fallback):
    host, rx, g, _, _, at = setup_entry()
    host.no_p3_fallback = no_fallback
    origin = g._p3_keyed_reply[2]
    boundary = g.boundary(2)
    live = _Bench(seconds=3, holdback=916)
    host.peer.live, host.peer.settle = live, .04
    host.peer.listening = False
    host.peer.key_instant = lambda raster, slot: raster.boundary(slot)
    host.peer.still_owed_n = lambda: 0
    word = onair._trim_silence(pactor1.control_signal(pactor1.CS_ACK_A))
    live.audio[at:at+len(word)] = word
    end = at + 5760 + 720
    live.pos, live.now = end, end + live.holdback
    def reader(seg, relative_at, **kwargs):
        if kwargs['seg_start'] + len(seg) < g._p3_keyed_reply[1] + SLOT:
            return None
        # The lower-mode ACK leaves the unconfirmed entry owned, with the
        # bounded listen still active, even when fallback is normally allowed.
        assert host.protocol == spec.Protocol.PACTOR3 and host.arq.entry_pending
        assert host.peer._p3_entry_listen_slot == 2 and not rx.cs_seen
        return rxfront.Event(relative_at/FS, 'cs', 'accepted P3 control',
                             protocol=spec.Protocol.PACTOR3, cs=arq.CS_SPEED_UP)
    with patch.object(rx, '_p3_cs', reader):
        heard, _, _, _ = onair._read_held_window(
            live, g, host.peer, rx, 2, live.audio[origin:end], origin, boundary)
    assert heard == arq.CS_SPEED_UP
    assert not host.arq.entry_pending and host.arq.role == arq.ISS


@pytest.mark.parametrize('marker_before', [False, True])
@pytest.mark.parametrize('capture_loss_during_p3', [False, True])
def test_complete_p1_fallback_preserves_word_anchor_and_body(
        tmp_path, marker_before, capture_loss_during_p3):
    host, rx, g, _, _, at = setup_entry()
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(host)
    host.peer = tx
    boundary = tx.aim(g, 2)
    if marker_before:
        tx._p3_entry_listen_slot = 2
    live = _Bench(seconds=4, holdback=916)
    origin = g._p3_keyed_reply[2]
    body = onair._trim_silence(pactor1.breakin_signal(
        b'P1 BODY', 100, invert=True, lead_s=0, tail_s=0))
    live.audio[at:at+len(body)] = body
    end = at + 5760 + 720
    live.pos, live.now = end, end + live.holdback
    queued = bytes(host.arq._outbuf)
    delivered = []
    receive = host.on_rx_event

    def delivery(ev):
        delivered.append(ev)
        receive(ev)

    def p3_read(*args, **kwargs):
        if capture_loss_during_p3:
            live.lost += 480
        return None

    tx.live = live
    with patch.object(host, 'on_rx_event', delivery), \
            patch.object(rx, '_p3_cs', p3_read), \
            patch.object(p1rx, 'entry_grant_match') as matcher, \
            patch.object(live, 'wait_until', wraps=live.wait_until) as waiting:
        heard, anchor, audio, start = onair._read_held_window(
            live, g, tx, rx, 2, live.audio[origin:end], origin, boundary)
        assert (heard, anchor) == (pactor1.CS_CHANGEOVER, at)
        assert not waiting.called  # No retired keyed-P3 extension after fallback.
        assert not matcher.called
        assert host.protocol == spec.Protocol.PACTOR1
        assert host.arq.role == arq.IRS and not host.arq.entry_pending
        assert getattr(tx, '_p3_entry_listen_slot', None) is None
        assert g.peer_cs.at == at and g.peer_cs.protocol == spec.Protocol.PACTOR1
        onair._align_shift(g, rx, tx, 2, anchor)
        assert g.shift(g.rx_slot(anchor))

        # Reinstall the retired marker inside the synchronous ACK callback.
        # The consumer guard must already be inert before _tx can return.
        send = tx._tx
        guard_at_send = []

        def reply(*args, **kwargs):
            tx._p3_entry_listen_slot = tx.slot
            guard_at_send.append(tx.entry_listen_due(tx.slot))
            return send(*args, **kwargs)

        with patch.object(tx, '_tx', reply):
            audio = onair._receive_changeover(
                live, g, tx, host, rx, 2, audio, start,
                np.zeros(0, np.float32), origin, round(tx.settle*FS),
                reserve_s=onair.PREKEY_RESERVE_S)
        assert not g.sending and g.protocol == spec.Protocol.PACTOR1
        packets = [ev for ev in delivered if ev.kind == 'packet']
        assert len(packets) == 1 and packets[0].packet[3]
        assert packets[0].breakin and packets[0].packet[2] == b'P1 BODY'
        assert guard_at_send == [False] and not tx.refused and tx.n == 1
        assert bytes(host.channel(host.ptchn).rx) == b'P1 BODY'
        assert bytes(host.arq._outbuf) == queued
        assert len([ev for ev in delivered if ev.kind == 'cs']) == 1

        # Bridges, a held re-read and a sweep cannot deliver that head/body twice.
        rx.control_signal_in(audio, start, g)
        onair._scan_frame(rx, audio, start)
        rx._on(delivered[0])
        assert len(delivered) == 2 and guard_at_send == [False]
        assert not tx.entry_listen_due(tx.slot)


@pytest.mark.parametrize('condition', ['different', 'none', 'fallback', 'answered', 'reset', 'terminal'])
def test_entry_listen_ownership_does_not_survive_other_decisions(tmp_path, condition):
    host, _, g, _, _, _ = setup_entry()
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.attach(host)
    host.peer = tx
    tx.aim(g, 2)
    tx._p3_entry_listen_slot = 2
    assert tx.entry_listen_due(2)
    tx.listening = tx.cede_listening = True
    if condition == 'different':
        assert not tx.entry_listen_due(3)
    elif condition == 'none':
        tx._p3_entry_listen_slot = None
        assert not tx.entry_listen_due(None)
    else:
        if condition == 'fallback':
            host.fall_back('ownership control')
        elif condition == 'answered':
            host._entry_answered()
        elif condition == 'reset':
            host.arq._finish_disconnected()
        elif condition == 'terminal':
            host.no_p3_fallback = True
            host.arq._entry_p1_current = True
            assert host.prepare_p1_entry_goodbye()
        assert not tx.entry_listen_due(2)
        assert tx._p3_entry_listen_slot is None
        # A new entry cannot inherit the old slot, even with the same number.
        host.arq.entry_pending = True
        assert not tx.entry_listen_due(2)
    assert tx.listening and tx.cede_listening


@pytest.mark.parametrize('capture_loss_during_p3', [False, True])
def test_mismatch_owns_one_bounded_original_reply_listen(capture_loss_during_p3):
    host, rx, g, _, _, at = setup_entry()
    tx = host.peer
    live = _Bench(seconds=3, holdback=916)
    tx.live, tx.settle, tx.listening = live, .04, False
    tx.key_instant = lambda raster, slot: raster.boundary(slot)
    tx.still_owed_n = lambda: 0
    origin, end = g._p3_keyed_reply[2], at + 8*480
    word = onair._trim_silence(pactor1.control_signal(pactor1.CS_6A9))
    live.audio[at:at+len(word)] = word
    live.pos, live.now = end, end + live.holdback
    def p3_read(*args, **kwargs):
        if capture_loss_during_p3:
            live.lost += 480
        return None

    with patch.object(rx, '_p3_cs', side_effect=p3_read) as p3, \
            patch.object(p1rx, 'entry_grant_match', wraps=p1rx.entry_grant_match) as matcher, \
            patch.object(live, 'wait_until', wraps=live.wait_until) as waiting:
        heard, anchor, audio, start = onair._read_held_window(
            live, g, tx, rx, 2, live.audio[origin:end], origin, g.boundary(2))
        assert heard is None and host.arq.entry_pending
        assert tx._p3_entry_listen_slot == 2
        assert onair.RadioTx.entry_listen_due(tx, 2)
        waiting.assert_called_once_with(g._p3_keyed_reply[1]+SLOT+916)
        assert start + len(audio) >= g._p3_keyed_reply[1]+SLOT
        assert host.arq._unanswered_upgrade == 0
        assert host.arq._upgrade_requests == 0
        if capture_loss_during_p3:
            # The extension's existing ineligible path reads complete P1 first;
            # this supported unassigned word returns before another P3 read.
            assert p3.call_count == 1 and anchor == at
            assert len(rx.cs_log) == 1 and rx.cs_log[0].spare == pactor1.CS_6A9
            assert not matcher.called and live.lost == 480
            assert not rx._entry_repeat_eligible(g, origin)
        else:
            assert p3.call_count == 2
        assert not tx.listening


@pytest.mark.parametrize('scan_s,extra_owed_s', [(0, 0), (.05, 0), (0, .05)])
def test_post_fallback_scan_and_remaining_work_keep_admission_guards(
        tmp_path, capsys, scan_s, extra_owed_s):
    host, rx, g, _, _, at = setup_entry()
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(host)
    host.peer = tx
    host.on_rx_event(rxfront.Event(at/FS, 'cs', 'P1 fallback head',
                                  protocol=spec.Protocol.PACTOR1,
                                  cs=pactor1.CS_CHANGEOVER, sense=0))
    tx.aim(g, 2)
    onair._reverse_before_key(g, host, tx, 2)
    assert host.arq.role == arq.IRS and host.protocol == spec.Protocol.PACTOR1
    live = _Bench(seconds=5)
    tx.live = live
    tx.prekey_cost_n = 416  # Same remaining work as the granted cadence scene.
    anchor, boundary = g.anchor, tx.boundary
    deadline = boundary - onair._prekey_lead(
        live, round(tx.settle*FS), tx.still_owed_n())
    live.pos = live.now = deadline
    scan = rx.deep_scan

    def charged(audio):
        live.spend(round(scan_s*FS))
        return scan(audio)

    with patch.object(rx, 'deep_scan', charged):
        onair._scan_frame(rx, np.zeros(round(.96*FS), np.float32),
                          deadline-round(.96*FS))
    tx.prekey_cost_n += round(extra_owed_s*FS)
    tx._advance_aim(before_render=True)
    assert live.samples % live._blk == 0 and g.anchor == anchor
    assert live.emissions == []
    if scan_s or extra_owed_s:
        assert tx.slot > 2 and tx.boundary == g.boundary(tx.slot)
        assert onair.SLOT_UNRENDERED in capsys.readouterr().out
    else:
        assert tx.slot == 2 and tx.boundary == boundary


def test_bounded_entry_fallback_emits_data_with_retired_marker(tmp_path):
    from hfmodem.tests.shrike.test_grid import _Rig

    host, _, g, _, _, _ = setup_entry()
    queued = bytes(host.arq._outbuf)
    entry = host.arq._inflight
    # A P1 ACK cannot end an unconfirmed entry. Spend the real silence bound
    # instead, retaining the queued bytes through every unconfirmed attempt.
    for silence in range(1, arq.UPGRADE_SILENCE_CYCLES):
        host.tick()
        assert host.arq.entry_pending and host.arq.upgrade_unanswered
        assert host.arq._unanswered_upgrade == silence
        assert host.arq._inflight is entry and bytes(host.arq._outbuf) == queued
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(host)
    host.peer = tx
    live = _Bench(seconds=6)
    tx.live, tx.rig, tx.transmit = live, _Rig(), True
    tx.aim(g, 2)
    tx._p3_entry_listen_slot = 2
    assert tx.entry_listen_due(2)
    send = tx._tx
    data = []

    def emission(audio, what, **kwargs):
        # Fallback must retire the producer's marker before synchronous TX.
        assert tx._p3_entry_listen_slot is None
        tx._p3_entry_listen_slot = tx.slot
        assert not tx.entry_listen_due(tx.slot)
        result = send(audio, what, **kwargs)
        if not tx.refused:
            data.append((what, tx.boundary, live.emissions[-1][0]))
        return result

    with patch.object(tx, '_tx', emission), \
            patch.object(host, 'fall_back', wraps=host.fall_back) as fallback:
        host.tick()
    fallback.assert_called_once()
    assert 'silence rule' in fallback.call_args.args[0]
    assert host.protocol == spec.Protocol.PACTOR1 and host.arq.role == arq.ISS
    assert not host.arq.entry_pending
    assert not host.arq.upgrade_unanswered and host.arq._upgrade_requests == 0
    assert len(data) == 1 and 'P1 pkt#' in data[0][0]
    assert data[0][1] == data[0][2] == g.boundary(tx.slot)
    assert host.arq._inflight is not None and host.arq._inflight.payload
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == queued


@pytest.mark.parametrize('change', ['fallback', 'answered', 'reset', 'rekeyed', 'generation', 'gap'])
def test_p3_read_rechecks_live_entry_and_emission_before_partial_matching(change):
    host, rx, g, clip, origin, at = setup_entry()
    requests = host.arq._upgrade_requests

    def callback(*args, **kwargs):
        if change == 'fallback':
            host.fall_back('synchronous read transition')
        elif change == 'answered':
            host._entry_answered()
        elif change == 'reset':
            host.arq._finish_disconnected()
        elif change == 'rekeyed':
            keyed = g._p3_keyed_reply
            g.keyed_slot += 1
            g._p3_keyed_reply = (g.keyed_slot, keyed[1]+SLOT, keyed[2]+SLOT, SLOT)
            host.peer.slot += 1
        elif change == 'generation':
            host.arq._link_generation += 1
        elif change == 'gap':
            host.peer.live.lost += 480
        return None

    with patch.object(rx, '_p3_cs', callback), \
            patch.object(p1rx, 'entry_grant_match') as matcher:
        assert rx.control_signal_in(clip[:at+10*480-origin], origin, g) == (None, None)
    assert not matcher.called and rx._entry_reply_results == {}
    assert getattr(host.peer, '_p3_entry_listen_slot', None) == (2 if change == 'gap' else None)
    assert host.arq._upgrade_requests == requests


@pytest.mark.parametrize('word', [pactor1.CS_59A, pactor1.CS_ACK_A, None],
                         ids=['full-grant', 'other-word', 'miss'])
def test_complete_owned_grant_precedes_p3_and_other_reads_survive_capture_loss(word):
    host, rx, g, _, _, at = setup_entry()
    origin = g._p3_keyed_reply[2]
    signal = onair._trim_silence(pactor1.control_signal(
        pactor1.CS_ACK_A if word is None else word))
    if word is None:
        signal[:] = 0
    audio = np.pad(signal, (at-origin, 720))
    packet, queued = host.arq._inflight, bytes(host.arq._outbuf)
    order = []
    control, deliver, full = rx.control_signal, rx._on, rx._p1_cs
    full_reads = []

    def read(*args, **kwargs):
        if not kwargs.get('p1_allowed', True):
            order.append('p3')
        return control(*args, **kwargs)

    def delivered(ev, **kwargs):
        if ev.protocol == spec.Protocol.PACTOR1:
            order.append('p1')
        return deliver(ev, **kwargs)

    def full_read(*args, **kwargs):
        full_reads.append(True)
        return full(*args, **kwargs)

    def p3_read(*args, **kwargs):
        host.peer.live.lost += 480
        return None

    with patch.object(rx, '_p3_cs', p3_read), \
            patch.object(rx, 'control_signal', read), \
            patch.object(rx, '_on', delivered), \
            patch.object(rx, '_p1_cs', full_read), \
            patch.object(p1rx, 'entry_grant_match') as matcher:
        heard, anchor = rx.control_signal_in(audio, origin, g)
    assert len(full_reads) == 1 and not matcher.called
    assert order == (['p1'] if word == pactor1.CS_59A else
                     ['p3'] if word is None else ['p3', 'p1'])
    assert rx._entry_repeat_eligible(g, origin) == (word == pactor1.CS_59A)
    assert len(rx.cs_log) == (0 if word is None else 1)
    assert host.arq.entry_pending and host.protocol == spec.Protocol.PACTOR3
    assert host.arq._inflight is packet and bytes(host.arq._outbuf) == queued
    assert not host.arq._entry_window_truncated
    result, = rx._entry_reply_results.values()
    assert result['complete'] and result['grant'] == (word == pactor1.CS_59A)
    if word == pactor1.CS_59A:
        assert (heard, anchor) == (None, at)  # A grant is an unassigned word.
        assert host.arq._entry_p1_current and result['credited']
        assert getattr(host.peer, '_p3_entry_listen_slot', None) is None
        host.tick()
        assert host.arq._upgrade_requests == 1 and host.arq._unanswered_upgrade == 0
    else:
        assert (heard, anchor) == (None, None) and not rx.cs_seen
        assert not host.arq._entry_p1_current and not result['credited']
        assert host.peer._p3_entry_listen_slot == 2


@pytest.mark.parametrize('admitted,expected', [(True, False), (False, True)])
def test_cached_complete_grant_keeps_p4_admission_preemption(admitted, expected):
    host, rx, g, _, _, at = setup_entry()
    origin = g._p3_keyed_reply[2]
    audio = np.pad(onair._trim_silence(pactor1.control_signal(pactor1.CS_59A)),
                   (at-origin, 720))
    host.arq.p4_admitted = admitted
    requests = host.arq._upgrade_requests
    with patch.object(rx, 'p4', SimpleNamespace(expected=expected, consumed=True)), \
            patch.object(rx, '_p4_read') as read, patch.object(rx, '_on') as deliver:
        rx.control_signal_in(audio, origin, g)
    read.assert_called_once()
    deliver.assert_not_called()
    assert not rx.cs_log and not host.arq._entry_p1_current
    assert host.arq._upgrade_requests == requests
    assert not any(row['grant'] for row in rx._entry_reply_results.values())


@pytest.mark.parametrize('change', ['fallback', 'answered', 'reset', 'rekeyed', 'generation'])
def test_cached_other_word_cannot_survive_synchronous_reply_retirement(change):
    host, rx, g, _, _, at = setup_entry()
    origin = g._p3_keyed_reply[2]
    audio = np.pad(onair._trim_silence(pactor1.control_signal(pactor1.CS_ACK_A)),
                   (at-origin, 720))
    requests = host.arq._upgrade_requests

    def retire(*args, **kwargs):
        if change == 'fallback':
            host.fall_back('synchronous cached-read transition')
        elif change == 'answered':
            host._entry_answered()
        elif change == 'reset':
            host.arq._finish_disconnected()
        elif change == 'generation':
            host.arq._link_generation += 1
        else:
            keyed = g._p3_keyed_reply
            g.keyed_slot += 1
            g._p3_keyed_reply = (g.keyed_slot, keyed[1]+SLOT, keyed[2]+SLOT, SLOT)
        return None

    with patch.object(rx, '_p3_cs', retire), patch.object(rx, '_on') as deliver:
        assert rx.control_signal_in(audio, origin, g) == (None, None)
    deliver.assert_not_called()
    assert not rx.cs_log and rx._entry_reply_results == {}
    assert host.arq._upgrade_requests == requests


def test_wrong_ack_with_live_capture_loss_still_owns_entry_listen():
    host, rx, g, _, _, at = setup_entry()
    host.no_p3_fallback = True
    origin = g._p3_keyed_reply[2]
    word = onair._trim_silence(pactor1.control_signal(pactor1.CS_ACK_A))
    audio = np.pad(word, (at-origin, 720))
    control = rx.control_signal

    def read(*args, **kwargs):
        heard = control(*args, **kwargs)
        if kwargs.get('p1_allowed', True):
            host.peer.live.lost += 480
        return heard

    with patch.object(rx, '_p3_cs', return_value=None), \
            patch.object(rx, 'control_signal', read), \
            patch.object(p1rx, 'entry_grant_match') as matcher:
        assert rx.control_signal_in(audio, origin, g) == (None, None)
    assert not matcher.called and not rx._entry_repeat_eligible(g, origin)
    assert host.arq.entry_pending and host.protocol == spec.Protocol.PACTOR3
    assert host.peer._p3_entry_listen_slot == 2


@pytest.mark.parametrize('cache', [None, {'unrelated': None}])
def test_standalone_p1_read_needs_no_entry_owner_attributes(cache):
    rx = object.__new__(onair._SessionRx)
    rx.cs_seen = False
    rx.host = SimpleNamespace(protocol=spec.Protocol.PACTOR1,
                              arq=SimpleNamespace(state=next(iter(onair.LINKED))))
    rx.p4 = SimpleNamespace(expected=False)
    if cache is not None:
        rx._p3_ahead = cache
    with patch.object(rx, '_p1_cs', return_value=None) as read:
        assert rx.control_signal(np.zeros(32), 0, 0) is None
    read.assert_called_once()
