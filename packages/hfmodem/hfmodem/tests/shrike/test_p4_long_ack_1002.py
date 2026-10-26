# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Opt-in robust LONG ACK selection on accepted, keyed source packets."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpy as np

from hfmodem.shrike import onair, p4breakintx, p4chirp, p4oracle, p4robustrx, p4turn, session
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
from hfmodem.shrike.p4turn import Proposal, State
from hfmodem.shrike.p4receiver import Kind, Registered, RegisteredObservation
from hfmodem.tests.shrike.test_p4turn import FS, ack_frame, machine, propose_ack


def opening(turn):
    reply = propose_ack(turn, ack_frame(FS, b'opening', 0, Kind.BODY))
    assert reply.word == 'X27'
    turn.keyed()


@pytest.mark.parametrize('policy,expected', [
    ('fixed-x27', ('X27', 'X27', 'X27')),
    ('alternate-packet', ('X27', 'V27', 'V27')),
])
def test_k4msu_c1_then_33_c2_copies(policy, expected):
    turn, _ = machine(role='irs', normal_ack='parity')
    opening(turn)
    c1 = ack_frame(300000, b'first packet', 0x61, Kind.LONG)
    first = propose_ack(turn, c1)
    assert first.word == expected[0]
    turn.keyed()
    turn.long_ack = policy
    words = []
    for i in range(33):
        c2 = ack_frame(480000 + i * 180000, b'second packet', 0x62, Kind.LONG,
                       header='A' if i % 2 == 0 else 'B')
        reply = propose_ack(turn, c2)
        assert reply.stream_sample == c2.observation.stream_sample + round(3.4 * FS)
        words.append(reply.word)
        turn.keyed()
    assert (first.word, words[0], words[-1]) == expected
    assert len(set(words)) == 1
    assert turn.acked_counter == 2
    assert turn.adapter.result.duplicate


def test_long_refusal_unread_and_status_do_not_advance_history():
    turn, _ = machine(role='irs', normal_ack='parity')
    opening(turn)
    propose_ack(turn, ack_frame(300000, b'first packet', 0x61, Kind.LONG))
    turn.keyed()
    turn.long_ack = 'alternate-packet'
    c2 = ack_frame(480000, b'second packet', 0x62, Kind.LONG)
    assert propose_ack(turn, c2).word == 'V27'
    turn.refused('deadline')
    assert turn._packet_ack[1] == 'X27' and turn.acked_counter == 1
    unread = replace(c2.observation, stream_sample=660000, frame=None)
    turn.observe(RegisteredObservation(unread, Registered.NEW), now=660000)
    assert turn._packet_ack[1] == 'X27' and turn.acked_counter == 1
    turn.adapter.terminal = False
    turn.observe(ack_frame(840000, b'wrong status', 0x82, Kind.LONG), now=840000)
    assert turn._packet_ack[1] == 'X27' and turn.acked_counter == 1
    assert propose_ack(turn, ack_frame(1020000, b'second packet', 0x62, Kind.LONG,
                                       header='B')).word == 'V27'
    turn.keyed()
    assert turn._packet_ack[1] == 'V27' and turn.acked_counter == 2


def test_cached_long_reselected_when_policy_changes():
    turn, _ = machine(role='irs', normal_ack='parity', long_ack='fixed-x27')
    opening(turn)
    propose_ack(turn, ack_frame(300000, b'first packet', 0x61, Kind.LONG))
    turn.keyed()
    assert propose_ack(turn, ack_frame(480000, b'second packet', 0x62, Kind.LONG)).word == 'X27'
    turn.keyed()
    turn.long_ack = 'alternate-packet'
    repeat = propose_ack(turn, ack_frame(660000, b'second packet', 0x62, Kind.LONG, header='B'))
    assert repeat.word == 'V27' and repeat.ack_reason == 'new packet after X27'
    assert turn._packet_ack[1] == 'X27'
    turn.keyed()
    assert turn._packet_ack[1] == 'V27'
    turn.long_ack = 'fixed-x27'
    repeat = propose_ack(turn, ack_frame(840000, b'second packet', 0x62, Kind.LONG))
    assert repeat.word == 'X27' and repeat.ack_reason is None


def test_late_long_keeps_history_until_deferred_answer_is_keyed():
    turn, _ = machine(role='irs', prekey_lead=.049)
    opening(turn)
    propose_ack(turn, ack_frame(300000, b'first packet', 0x61, Kind.LONG))
    turn.keyed()
    turn.long_ack = 'alternate-packet'
    c2 = ack_frame(3528420, b'second packet', 0x62, Kind.LONG)
    due = c2.observation.stream_sample + round(3.4 * FS)
    cutoff = due - turn.lead
    assert due == 3691620 and cutoff == 3689268
    turn.observe(c2, now=cutoff + 146)
    current = turn.decide(cutoff + 146)
    assert current.word == 'V27' and current.stream_sample == due
    turn.refused('deadline')
    assert turn._deferred is not None
    assert turn._packet_ack[1] == 'X27' and turn.acked_counter == 1
    later = due + round(3.75 * FS)
    reply = turn.decide(later - turn.lead)
    assert reply.word == 'V27' and reply.stream_sample == later
    assert turn._packet_ack[1] == 'X27' and turn.acked_counter == 1
    turn.keyed()
    assert turn._packet_ack[1] == 'V27' and turn.acked_counter == 2


def test_distinct_long_identity_does_not_advance_without_key():
    turn, _ = machine(role='irs', long_ack='alternate-packet')
    opening(turn)
    c1 = ack_frame(300000, b'first packet', 0x61, Kind.LONG)
    assert propose_ack(turn, c1).word == 'V27'
    turn.keyed()
    c2 = ack_frame(480000, b'second packet', 0x62, Kind.LONG)
    assert propose_ack(turn, c2).word == 'X27'
    turn.refused('deadline')
    assert turn._packet_ack[1] == 'V27' and turn.acked_counter == 1
    repeat = propose_ack(turn, ack_frame(660000, b'first packet', 0x61, Kind.LONG,
                                         header='B'))
    assert repeat.word == 'V27'
    turn.keyed()
    assert turn._packet_ack[1] == 'V27' and turn.acked_counter == 1


def test_protocol_three_normal_preserves_merged_handback_counter_policy():
    for policy, expected in [('fixed-x27', 'X27'), ('alternate-packet', 'X27')]:
        turn, _ = machine(role='irs', normal_ack='parity', long_ack=policy)
        opening(turn)
        reply = propose_ack(turn, ack_frame(300000, b'protocol three', 0x62))
        assert reply.word == expected


def test_chirp_policy_is_unchanged_by_long_ack():
    turn, _ = machine(role='irs', chirp_ack='alternate', long_ack='alternate-packet')
    words = []
    for i, status in enumerate((0x61, 0x62, 0x62, 0x63)):
        reply = propose_ack(turn, ack_frame(48000 + i * 180000, b'chirp', status,
                                            Kind.CHIRP))
        words.append(reply.word)
        turn.keyed()
    assert words == ['X27', 'V27', 'V27', 'X27']


def test_ve3kpg_normal_c1_copies_stay_v27():
    turn, _ = machine(role='irs', long_ack='alternate-packet')
    opening(turn)
    words = []
    for i in range(6):
        reply = propose_ack(turn, ack_frame(300000 + i * 60000, b'normal c1', 0x01,
                                            header='A' if i % 2 == 0 else 'B'))
        words.append(reply.word)
        turn.keyed()
    assert words == ['V27'] * 6


def test_k4msu_recorded_long_crc_and_identity():
    root = Path(__file__).resolve().parents[5]
    path = root / 'captures/onair-1002-2150/stream.flac'
    if not path.exists():
        path = root.parents[2] / 'captures/onair-1002-2150/stream.flac'
    if not path.exists():
        pytest.skip('K4MSU recording absent')
    audio = session.load_wav(str(path), FS)
    frames = []
    for at in (1008369, 1188446, 1368444, 1548442):
        crop = audio[at-4800:at+round(3.4*FS)]
        coherence, symbols, _ = p4oracle.placed_despread(crop, FS, .1, 2.434, count=373)
        decoded = p4robustrx.decode_symbols(symbols)
        assert coherence > .5 and decoded is not None and decoded.speedlevel == 3
        frames.append(decoded.frame)
    assert [frame.status.byte for frame in frames] == [0x61, 0x61, 0x62, 0x62]
    assert frames[0].data == frames[1].data
    assert frames[2].data == frames[3].data
    assert frames[0].data != frames[2].data


def test_invalid_long_ack_policy():
    with pytest.raises(ValueError, match='long ack policy'):
        machine(long_ack='unknown')


@pytest.mark.parametrize('long_policy', ['fixed-x27', 'alternate-packet'])
def test_refused_chirp_does_not_advance_either_selection_history(long_policy):
    turn, _ = machine(role='irs', chirp_ack='alternate', long_ack=long_policy)
    first = propose_ack(turn, ack_frame(FS, b'first chirp', 0x61, Kind.CHIRP))
    assert first.word == 'X27'
    assert turn._last_ack is None and turn._packet_ack is None
    turn.refused('deadline')
    assert turn._last_ack is None and turn._packet_ack is None
    second = propose_ack(turn, ack_frame(300000, b'different chirp', 0x62, Kind.CHIRP))
    assert second.word == 'V27'
    turn.keyed()
    assert turn._last_ack[0] == second.ack_identity
    assert turn._packet_ack[0][1] == Kind.CHIRP and turn._packet_ack[1] == 'V27'
    repeat = propose_ack(turn, ack_frame(480000, b'different chirp', 0x62, Kind.CHIRP,
                                        header='B'))
    assert repeat.word == 'V27'
    turn.keyed()
    third = propose_ack(turn, ack_frame(660000, b'third chirp', 0x63, Kind.CHIRP))
    assert third.word == 'X27'


def test_keyed_shared_packet_history_across_body_chirp_long_and_normal():
    turn, _ = machine(role='irs', long_ack='alternate-packet',
                      normal_ack='alternate-packet', chirp_ack='alternate')
    opening(turn)
    for i, (kind, status, expected) in enumerate([
        (Kind.CHIRP, 0x61, 'X27'), (Kind.LONG, 0x62, 'V27'),
        (Kind.NORMAL, 0x03, 'X27'), (Kind.CHIRP, 0x60, 'V27')]):
        reg = ack_frame(300000+i*180000, f'packet {i}'.encode(), status, kind)
        before = turn._packet_ack
        reply = propose_ack(turn, reg)
        assert reply.word == expected and turn._packet_ack == before
        turn.keyed()
        assert turn._packet_ack[0][1] == kind and turn._packet_ack[1] == expected


@pytest.mark.parametrize('policy,words', [
    ('v27', ('V27', 'V27')), ('x27', ('X27', 'X27')),
    ('parity', ('V27', 'X27')), ('x27-u27', ('U27', 'X27')),
    ('alternate-packet', ('V27', 'X27')),
])
@pytest.mark.parametrize('long_policy', ['fixed-x27', 'alternate-packet'])
def test_protocol_three_normal_iss_expectations_follow_selected_policy(policy, words, long_policy):
    turn, _ = machine(normal_ack=policy, long_ack=long_policy)
    for status, expected in zip((0x61, 0x62), words, strict=True):
        turn._data_packet = Proposal(Kind.NORMAL, 0, status=status)
        assert turn._expected_data_ack() == expected
        if policy != 'alternate-packet':
            assert turn._normal_ack((status, b'normal'), None)[0] == expected


def test_alternate_long_trial_enabled_from_opening_c0_is_v27_x27_x27():
    turn, _ = machine(role='irs', long_ack='alternate-packet', normal_ack='parity')
    opening(turn)
    words = []
    for i, status in enumerate((0x61, 0x62, 0x62)):
        reply = propose_ack(turn, ack_frame(300000+i*180000,
                                            b'c1' if i == 0 else b'c2', status, Kind.LONG))
        words.append(reply.word)
        turn.keyed()
    assert words == ['V27', 'X27', 'X27']


def test_history_neutral_repeat_request_preserves_late_long_deferred_answer():
    turn, _ = machine(role='irs', prekey_lead=.049)
    opening(turn)
    propose_ack(turn, ack_frame(300000, b'first packet', 0x61, Kind.LONG))
    turn.keyed()
    turn.long_ack = 'alternate-packet'
    c2 = ack_frame(3528420, b'second packet', 0x62, Kind.LONG)
    due = 3691620
    turn.observe(c2, now=3689414)
    assert turn.decide(3689414).stream_sample == due
    turn.refused('deadline')
    deferred = turn._deferred
    assert deferred[1]['due'] == due+round(3.75*FS)
    before = (turn.ack_word, turn.acked_counter, turn._packet_ack,
              turn._last_ack, turn.first_body)
    turn._next = Proposal(Kind.WORD, due, word='X27', state=State.IRS_ANSWER,
                          current_slot_accepted=False)
    request = turn.keyed()
    assert request.ack_identity is None and request.ack_reason is None
    assert turn._deferred == deferred
    assert (turn.ack_word, turn.acked_counter, turn._packet_ack,
            turn._last_ack, turn.first_body) == before
    reply = turn.decide(due+round(3.75*FS)-turn.lead)
    assert reply.word == 'V27' and reply.ack_identity == (0x62, c2.observation.frame.data)
    turn.keyed()
    assert turn.acked_counter == 2 and turn._packet_ack[1] == 'V27'


@pytest.mark.parametrize('delta', [0, 1, 100])
@pytest.mark.parametrize('kind,status,first_word', [
    (Kind.BODY, 0, 'X27'), (Kind.NORMAL, 0x61, 'V27'),
    (Kind.LONG, 0x61, 'X27'), (Kind.CHIRP, 0x61, 'X27'),
])
def test_physical_key_shift_keeps_admitted_family_history_and_repeats(
        delta, kind, status, first_word):
    turn, _ = machine(role='irs', normal_ack='alternate-packet',
                      long_ack='alternate-packet', chirp_ack='alternate')
    first = propose_ack(turn, ack_frame(FS, b'first packet', status, kind))
    assert first.word == first_word
    assert turn._packet_ack is None and turn._last_ack is None
    keyed = turn.keyed(first.stream_sample+delta)
    assert keyed.stream_sample == first.stream_sample+delta
    assert turn._packet_ack == ((turn._receive_epoch, kind, first.ack_identity), first_word)
    assert turn._last_ack[0] == first.ack_identity
    repeat = propose_ack(turn, ack_frame(300000, b'first packet', status, kind, header='B'))
    before = turn._packet_ack
    assert repeat.word == first_word
    turn.keyed(repeat.stream_sample+delta)
    assert turn._packet_ack == before
    next_packet = propose_ack(turn, ack_frame(480000, b'different packet', 0x62, Kind.CHIRP))
    assert next_packet.word == 'V27'
    assert turn._packet_ack == before
    turn.keyed(next_packet.stream_sample+delta)
    assert turn._last_ack[0] == next_packet.ack_identity
    assert turn._packet_ack[0][1] == Kind.CHIRP
    assert turn._packet_ack[1] == next_packet.word


@pytest.mark.parametrize('delta', [0, 1, 100])
@pytest.mark.parametrize('case', ['refused', 'identity', 'generation', 'epoch', 'control'])
def test_physical_key_shift_does_not_credit_unrelated_admission_metadata(delta, case):
    turn, _ = machine(role='irs', chirp_ack='alternate', long_ack='alternate-packet')
    reply = propose_ack(turn, ack_frame(FS, b'admitted packet', 0x61, Kind.CHIRP))
    if case == 'refused':
        turn.refused('deadline')
        with pytest.raises(RuntimeError, match='no proposal'):
            turn.keyed(reply.stream_sample+delta)
    elif case == 'epoch':
        turn._receive_epoch += 1
        with pytest.raises(RuntimeError, match='stale receive answer'):
            turn.keyed(reply.stream_sample+delta)
    else:
        if case == 'identity':
            turn._next = replace(reply, ack_identity=(0x62, b'unrelated source'))
        elif case == 'generation':
            turn._peer_generation += 1
        else:
            turn._next = Proposal(Kind.WORD, reply.stream_sample, word='X27',
                                  state=State.IRS_ANSWER, current_slot_accepted=False)
        turn.keyed(reply.stream_sample+delta)
    assert turn._packet_ack is None and turn._packet_words == {}


@pytest.mark.parametrize('delta', [0, 1, 100])
def test_driver_records_shifted_body_and_chirp_keys_using_actual_writer_sample(monkeypatch, delta):
    audio = np.zeros(15*FS, np.float32)
    body_at = round(.923*FS)
    body = p4breakintx.changeover_packet(b'opening', 0)
    audio[body_at:body_at+len(body)] = body
    for i, (payload, status) in enumerate([(b'first chirp', 0x61),
                                          (b'first chirp', 0x61),
                                          (b'next chirp', 0x62)]):
        wave = p4chirp.entry_packet(payload, status)
        at = round((2.173+3.75*i)*FS)+delta
        audio[at:at+len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_ladder = 'chirp'
    tx.p4_link_options.p4_long_ack = 'alternate-packet'
    tx.p4_link_options.p4_chirp_ack = 'alternate'
    tx.p4_link_options.p4_link_timeout = 14.5
    machines = []
    keyed_history = []
    constructor = p4turn.P4Turn

    def tracked(*args, **kwargs):
        result = constructor(*args, **kwargs)
        machines.append(result)
        key = result.keyed

        def observed(at=None):
            sent = key(at)
            keyed_history.append((sent, result._packet_ack, result._last_ack))
            return sent

        result.keyed = observed
        return result

    key = tx._tx

    def shifted(audio, what):
        key(audio, what)
        tx.tx_audio_start += delta
        tx.tx_end += delta
        stream.pos = tx.tx_end

    monkeypatch.setattr(p4turn, 'P4Turn', tracked)
    monkeypatch.setattr(tx, '_tx', shifted)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    emitted = tx.p4_probe.emissions[1:]
    assert [e['word'] for e in emitted] == ['X27', 'X27', 'X27', 'V27']
    assert all(e['keyed_at'] == e['audio_start'] == e['proposed_at']+delta for e in emitted)
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Refused']
    turn, = machines
    assert len(turn._packet_words) == 3
    assert {key[1] for key in turn._packet_words} == {Kind.BODY, Kind.CHIRP}
    assert keyed_history[-1][1][1] == keyed_history[-1][2][1].word == 'V27'
    assert [sent.stream_sample for sent, _, _ in keyed_history] == [
        e['audio_start'] for e in emitted]


COUNTER_POLICIES = ('counter-odd-t27', 'counter-even-t27')


@pytest.mark.parametrize('policy,expected', [
    (None, ('X27', 'X27')), ('fixed-x27', ('X27', 'X27')),
    ('counter-odd-t27', ('X27', 'T27')),
    ('counter-even-t27', ('T27', 'X27')),
])
def test_counter_recorded_fields_duplicates_refusal_and_held_unread(policy, expected):
    from hfmodem.tests.shrike.test_p4_late_long_ack_1010 import (
        DATA_62, DATA_03, accepted, seed_v60)
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    options = {} if policy is None else dict(long_ack=policy)
    turn, _ = machine(role='irs', normal_ack='parity', prekey_lead=.049, **options)
    turn.require_long_crc = True
    delivered = []
    turn.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    seed_v60(turn, 1412434)
    words = []
    for index, (at, now, status, data, word) in enumerate([
        (1428171, 1589338, 0x62, DATA_62, expected[0]),
        (1608171, 1769338, 0x62, DATA_62, expected[0]),
        (1788166, 1950320, 0x03, DATA_03, expected[1]),
        (2148166, 2310320, 0x03, DATA_03, expected[1]),
    ]):
        assert len(data) == 69
        reg = accepted(at, data, status)
        reg = replace(reg, observation=replace(reg.observation, header='B' if index % 2 else 'A'))
        turn.observe(reg, now=now)
        reply = turn.decide(now)
        assert (reply.word, reply.stream_sample, reply.ack_identity) == (word, at+163200, (status, data))
        words.append(reply.word)
        if index == 2:
            history = turn._last_ack, turn._packet_ack, dict(turn._packet_words)
            turn.refused('deadline')
            assert (turn._last_ack, turn._packet_ack, turn._packet_words) == history
            assert turn._deferred[1]['due'] == 2131366
            slot = at+180000
            turn.observe(unread(Kind.LONG, slot), now=slot+158400)
            reply = turn.decide(2131366-turn.lead)
            assert (reply.word, reply.stream_sample, reply.accepted_target) == (word, 2131366, at)
            assert not reply.current_slot_accepted and not turn.pending[slot]['accepted']
        turn.keyed()
        assert turn._packet_ack[1] == word
        assert turn._packet_words[(turn._receive_epoch, Kind.LONG, (status, data))] == word
    assert words == [expected[0], expected[0], expected[1], expected[1]]
    assert delivered == [DATA_62, DATA_03[:39]]
    history = turn._last_ack, turn._packet_ack, dict(turn._packet_words)
    due = 2148166+180000+163200
    turn.silence(due-turn.lead, answer_due=due)
    silent = turn.decide(due-turn.lead)
    assert silent.word == expected[1] and not silent.current_slot_accepted
    assert silent.current_slot is None and silent.accepted_target == 2148166
    assert (turn._last_ack, turn._packet_ack, turn._packet_words) == history
    if policy in COUNTER_POLICIES:
        events = [e for e in turn.link_events if e['event'] == 'long-answer']
        assert events and all(e['ack_policy'] == policy for e in events)
        assert {e['counter'] for e in events} == {2, 3}
        assert any(e['repeat'] for e in events)


@pytest.mark.parametrize('policy', COUNTER_POLICIES)
@pytest.mark.parametrize('protocol', [0, 3])
@pytest.mark.parametrize('header', ['A', 'B'])
def test_counter_maps_all_counters_and_wrap(policy, protocol, header):
    turn, _ = machine(role='irs', long_ack=policy)
    for index, counter in enumerate((0, 1, 2, 3, 0)):
        reg = ack_frame(FS+index*180000, f'packet {index}'.encode(),
                        protocol*32+counter, Kind.LONG, header=header)
        reply = propose_ack(turn, reg)
        t27 = counter % 2 == (1 if policy == 'counter-odd-t27' else 0)
        assert reply.word == ('T27' if t27 else 'X27')
        assert reply.ack_reason == f'{policy} counter {counter}'
        for upper in (0x04, 0x1c, 0x80, 0x9c):
            assert turn._long_ack(((protocol*32+counter) | upper, b'other payload')) == (
                reply.word, reply.ack_reason)
        turn.keyed()
        assert turn._long_policies[(turn._receive_epoch, Kind.LONG, reply.ack_identity)] == policy


@pytest.mark.parametrize('policy', COUNTER_POLICIES)
@pytest.mark.parametrize('status', [0x62, 3])
def test_counter_acceptance_retains_word_before_refused_first_key(policy, status):
    from hfmodem.tests.shrike.test_p4_late_long_ack_1010 import accepted, seed_v60
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    turn, _ = machine(role='irs', long_ack=policy)
    seed_v60(turn, 1412434)
    reg = accepted(1428171, b'held'.ljust(69, b'\x1e'), status)
    turn.observe(reg)
    selected = turn.decide(1591371-turn.lead)
    assert selected.word == ('T27' if status % 2 == (1 if policy == 'counter-odd-t27' else 0) else 'X27')
    turn.refused('deadline')
    held = turn._held_answer
    assert turn._last_ack is None and turn._packet_ack is None and not turn._packet_words
    for invalid in [
        unread(Kind.LONG, 1608171),
        replace(reg, observation=replace(reg.observation, stream_sample=1608171, provisional=True)),
        replace(reg, observation=replace(reg.observation, stream_sample=1608171,
            frame=accepted(1608171, b'rejected'.ljust(69, b'\x1e'), 0x82).observation.frame)),
        replace(reg, observation=replace(reg.observation, stream_sample=1428170), outcome=Registered.DUPLICATE),
    ]:
        turn.observe(invalid)
        assert turn._held_answer is held
    repeated = turn.decide(1771371-turn.lead)
    assert repeated.word == selected.word and repeated.ack_identity == selected.ack_identity
    assert not repeated.current_slot_accepted
    event = next(e for e in reversed(turn.link_events) if e['event'] in ('long-answer', 'acked-repeat'))
    assert (event['ack_policy'], event['word'], event['counter'], event['status'], event['repeat']) == (
        policy, selected.word, status & 3, status, True)
    assert event['accepted_target'] == 1428171 and event['current_slot'] == 1608171
    assert not event['current_slot_accepted']
    turn.keyed()
    assert turn._packet_ack[1] == selected.word


@pytest.mark.parametrize('policy', COUNTER_POLICIES)
def test_counter_policy_reselection_and_epoch_keys(policy):
    turn, _ = machine(role='irs', long_ack=policy)
    first = propose_ack(turn, ack_frame(FS, b'packet', 3, Kind.LONG))
    turn.keyed()
    turn.long_ack = COUNTER_POLICIES[1] if policy == COUNTER_POLICIES[0] else COUNTER_POLICIES[0]
    repeat = propose_ack(turn, ack_frame(FS+180000, b'packet', 3, Kind.LONG))
    assert repeat.word != first.word
    turn.keyed()
    assert turn._packet_ack[1] == repeat.word
    old_key = turn._packet_ack[0]
    turn._close_receive_epoch('test')
    assert turn._held_answer is None
    turn._open_receive_epoch(FS+360000)
    newer = propose_ack(turn, ack_frame(FS+360000, b'packet', 3, Kind.LONG))
    turn.keyed()
    new_key = turn._packet_ack[0]
    assert new_key[0] != old_key[0] and new_key[1:] == old_key[1:]
    assert turn._packet_words[new_key] == newer.word == repeat.word
    held = turn._held_answer
    turn.observe(ack_frame(FS+180000, b'stale', 2, Kind.LONG))
    assert turn._held_answer is held


def test_omitted_and_explicit_fixed_long_proposals_and_events_are_identical():
    turns = [machine(role='irs', **options)[0] for options in ({}, {'long_ack': 'fixed-x27'})]
    for index, status in enumerate((0x62, 3, 3)):
        reg = ack_frame(FS+index*180000, b'packet', status, Kind.LONG)
        assert propose_ack(turns[0], reg) == propose_ack(turns[1], reg)
        assert turns[0].keyed() == turns[1].keyed()
        assert turns[0].link_events == turns[1].link_events


@pytest.mark.parametrize('policy', COUNTER_POLICIES)
def test_counter_long_policy_preserves_short_normal_chirp_and_iss_t27_rules(policy):
    from hfmodem.tests.shrike.test_p4turn import body, word
    turn, _ = machine(role='irs', long_ack=policy, normal_ack='parity')
    turn.observe(replace(body(FS), observation=replace(body(FS).observation, kind=Kind.SHORT, frame=None, value=12, header='A')))
    assert turn.decide(FS+44160-turn.lead).word == 'T27'
    turn.keyed()
    for index, (kind, status, expected) in enumerate([
        (Kind.NORMAL, 0x61, 'V27'), (Kind.NORMAL, 0x62, 'X27'),
        (Kind.CHIRP, 0x61, 'X27'), (Kind.CHIRP, 0x62, 'V27')]):
        reply = propose_ack(turn, ack_frame(300000+index*180000, b'other', status, kind))
        assert reply.word == expected
        turn.keyed()
    iss, _ = machine(long_ack=policy)
    iss.state = State.ISS_DATA_ACK
    iss._data_packet = Proposal(Kind.LONG, 0, status=3)
    assert 'T27' in iss.expected_words()
    iss.observe(word('T27', FS))
    assert iss.acked_counter is None


@pytest.mark.parametrize('policy', ['fixed-x27', 'alternate-packet', *COUNTER_POLICIES])
def test_onair_forwards_chosen_long_policy_to_turn(monkeypatch, policy):
    stream = Stream(np.zeros(2*FS, np.float32))
    tx = Tx(stream)
    tx.p4_link_options.p4_long_ack = policy
    class ReachedTurn(Exception):
        pass
    original = p4turn.P4Turn
    def checked(*args, **kwargs):
        turn = original(*args, **kwargs)
        assert turn.long_ack == policy
        raise ReachedTurn
    monkeypatch.setattr(p4turn, 'P4Turn', checked)
    with pytest.raises(ReachedTurn):
        onair._run_p4_link(tx, stream, onair._MasterGrid(
            0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
