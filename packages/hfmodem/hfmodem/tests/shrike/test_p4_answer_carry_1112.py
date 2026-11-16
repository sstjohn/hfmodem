# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A read packet owns its ACK until an acknowledging word goes on air."""

from dataclasses import replace

import numpy as np
import pytest

from hfmodem.shrike import onair, p4oracle, p4turn, p4robusttx
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, Registered, RegisteredObservation
from hfmodem.tests.shrike.test_p4_ack_toggle_1110 import key_packet, short
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.shrike.test_p4_late_long_ack_1010 import accepted
from hfmodem.tests.shrike.test_p4turn import body, machine, propose_ack


FORMS = [(Kind.NORMAL, {}), (Kind.LONG, {}),
         (Kind.LONG, {'sf8': True}), (Kind.LONG, {'speedlevel': 'SL5-long'})]


def packet(at, counter, kind, tags):
    data = (b'first' if counter == 0 else b'second').ljust(
        40 if kind == Kind.NORMAL else 141 if tags.get('sf8') else 69, b'\x1e')
    reg = accepted(at, data, counter, kind)
    return replace(reg, observation=replace(reg.observation, **tags))


def test_silent_slot_defers_reacquisition_until_after_repeat_request(monkeypatch):
    cycle = round(3.75 * FS)
    at = cycle + FS
    due = at + round(p4turn.T[Kind.LONG] * FS)
    live = Stream(np.zeros(due + FS // 10, np.float32))
    live.pos = at - FS // 2
    live.wall = live.pos
    live.sample_now = lambda: max(live.pos, live.wall)
    tx = Tx(live)
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = (len(live.audio) - live.pos) / FS
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, 0., misses=2)
    turn_type = p4turn.P4Turn

    def factory(*args, **kwargs):
        turn = turn_type(*args, **{**kwargs, 'role': 'irs'})
        turn.observe(packet(at - cycle, 0, Kind.LONG, {}))
        assert turn.decide(due - cycle - turn.lead).word == 'X27'
        turn.keyed()
        return turn

    monkeypatch.setattr(p4turn, 'P4Turn', factory)
    monkeypatch.setattr(rx, 'sync_peer_watch', lambda *args, **kwargs: None)
    monkeypatch.setattr(rx, 'read_partial', lambda *args, **kwargs: [])
    monkeypatch.setattr(rx, '_word', lambda *args, **kwargs: None)
    monkeypatch.setattr(rx, '_packet', lambda *args, **kwargs: None)
    monkeypatch.setattr(rx, 'prepare_sf8', lambda *args: None)
    monkeypatch.setattr(rx, 'prepare_long', lambda *args: None)
    scans = []

    def scan(*args, **kwargs):
        scans.append(len(tx.slots_used))
        live.wall = live.sample_now() + round(.150 * FS)
        return []

    monkeypatch.setattr(p4oracle, 'word_scan', scan)
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    decision, = [e for e in tx.p4_probe.link_events if e['event'] == 'Decision']
    keyed, = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert decision['now_sample'] < decision['decision_at']
    assert keyed['at'] == due and keyed['word'] == 'X27'
    assert keyed['purpose'] == 'repeat-request' and keyed['ack_identity'] is None
    assert scans == [1]
    assert rx.silent_cycles == 1


@pytest.mark.parametrize('kind,tags', FORMS)
@pytest.mark.parametrize('next_packet', ['unread', 'same', 'different'])
@pytest.mark.parametrize('miss', ['deadline', 'emission'])
def test_driver_redecides_missed_answer(monkeypatch, kind, tags, next_packet, miss):
    cycle = round(p4turn.CYCLE[kind] * FS)
    at = cycle + FS
    due = at + round(p4turn.T[kind] * FS)
    live = Stream(np.zeros(due + cycle + FS, np.float32))
    live.pos = at - FS // 2
    stalled_until = [0]
    live.sample_now = lambda: max(live.pos, stalled_until[0])
    live.holdback = round(.020 * FS) if tags.get('sf8') else 0
    tx = Tx(live)
    tx.p4_probe.emissions.clear()
    tx.p4_probe.peer_breakin = True
    tx.p4_link_options.p4_link_timeout = (len(live.audio) - live.pos) / FS
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, at, .330, cycle / FS, 0.)
    delivered, machines, missed, keyed = [], [], [], []
    tx.p4_link_host.deliver = delivered.append
    turn_type = p4turn.P4Turn

    def factory(*args, **kwargs):
        turn = turn_type(*args, **{**kwargs, 'role': 'irs'})
        seed = packet(at - cycle, 0, kind, tags)
        turn.observe(seed)
        assert turn.decide(due - cycle - turn.lead).word == 'X27'
        turn.keyed()
        machines.append(turn)
        return turn

    monkeypatch.setattr(p4turn, 'P4Turn', factory)
    # The primary packet clock owns both slots; no optional acquisition is needed.
    monkeypatch.setattr(rx, 'sync_peer_watch', lambda *args, **kwargs: None)
    emitted = set()

    def partial(*args, **kwargs):
        onset = rx.clock.next_due
        result = []
        if onset in (at, at + cycle) and ('head', onset) not in emitted:
            head = replace(packet(onset, 1, kind, tags).observation,
                           frame=None, provisional=True)
            rx._emit(head, result)
            emitted.add(('head', onset))
        return result

    def window(audio, origin, **kwargs):
        onset = rx.clock.next_due
        result = []
        support = round((3.280 if kind == Kind.LONG else .810) * FS)
        if (onset in (at, at + cycle) and origin + len(audio) >= onset + support
                and ('body', onset) not in emitted):
            counter = 2 if onset > at and next_packet == 'different' else 1
            obs = packet(onset, counter, kind, tags).observation
            if onset > at and next_packet == 'unread':
                obs = replace(obs, frame=None)
            rx._emit(obs, result)
            emitted.add(('body', onset))
        return result

    monkeypatch.setattr(rx, 'read_partial', partial)
    monkeypatch.setattr(rx, 'read_window', window)
    monkeypatch.setattr(rx, 'refine', lambda **kwargs: [])
    monkeypatch.setattr(rx, 'prepare_sf8', lambda *args: None)
    monkeypatch.setattr(rx, 'prepare_long', lambda *args: None)
    keyable = onair._keyable_slot

    def admission(*args, **kwargs):
        proposal = machines[0]._next
        if miss == 'deadline' and not missed and proposal.stream_sample == due:
            missed.append(proposal)
            return 1
        return keyable(*args, **kwargs)

    monkeypatch.setattr(onair, '_keyable_slot', admission)
    emit = tx._tx

    def transmit(audio, what):
        proposal = machines[0]._next
        if miss == 'emission' and not missed and proposal.stream_sample == due:
            missed.append(proposal)
            if kind == Kind.LONG:
                stalled_until[0] = due + 1
                tx.p4_tx_admit()
                pytest.fail('late LONG answer admitted')
            raise onair._MissedTxSlot('late answer')
        if tx.p4_tx_admit is not None:
            tx.p4_tx_admit()
        assert np.array_equal(audio, p4robusttx.robust_word(proposal.word))
        emit(audio, what)
        keyed.append(proposal)

    tx._tx = transmit
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    ack = 'T27' if tags.get('speedlevel') else 'V27'
    assert len(missed) == 1 and missed[0].word == ack
    assert ('body', at + cycle) in emitted
    assert keyed and keyed[0].stream_sample == due + cycle
    assert keyed[0].word == ack
    assert keyed[0].ack_identity is not None
    assert keyed[0].current_slot == at + cycle
    assert keyed[0].current_slot_accepted == (next_packet != 'unread')
    assert delivered == [b'first', b'second'] + ([b'second'] if next_packet == 'different' else [])
    assert any(e['event'] == 'Refused' and e['reason'] == 'deadline'
               for e in tx.p4_probe.link_events)


@pytest.mark.parametrize('kind,tags', FORMS)
@pytest.mark.parametrize('miss', ['deadline', 'expiry', 'decode'])
def test_deferred_unread_then_read_keeps_ack_chain(kind, tags, miss):
    turn, _ = machine(role='irs')
    cycle = round(p4turn.CYCLE[kind] * FS)
    at = FS
    due = at + round(turn.t[kind] * FS)
    turn.observe(packet(at, 0, kind, tags))
    assert turn.decide(due - turn.lead).word == 'X27'
    turn.keyed()
    reg = packet(at + cycle, 1, kind, tags)
    turn.observe(reg, now=due + cycle + 1 if miss == 'decode' else None)
    if miss != 'decode':
        answer = turn.decide(due + cycle - turn.lead)
        assert answer.word == ('T27' if tags.get('speedlevel') else 'V27')
        if miss == 'deadline':
            turn.refused('deadline')
        else:
            assert turn.decide(due + cycle + 1) is None
    unread = replace(packet(at + 2 * cycle, 2, kind, tags),
                     observation=replace(reg.observation, stream_sample=at + 2 * cycle,
                                         frame=None))
    turn.observe(unread)
    repeat = turn.decide(due + 2 * cycle - turn.lead)
    assert repeat.word == ('T27' if tags.get('speedlevel') else 'V27')
    assert repeat.ack_identity == (reg.observation.frame.status.byte, reg.observation.frame.data)
    assert not repeat.current_slot_accepted
    turn.keyed()
    turn.observe(packet(at + 3 * cycle, 1, kind, tags))
    answer = turn.decide(due + 3 * cycle - turn.lead)
    assert answer.word == ('T27' if tags.get('speedlevel') else 'V27')


@pytest.mark.parametrize('jitter', [0, 96])
@pytest.mark.parametrize('heard', ['unread', 'nothing'])
def test_deferred_data_after_control_acknowledges_read_packet(jitter, heard):
    turn, _ = machine(role='irs')
    assert key_packet(turn, FS, 1, Kind.LONG).word == 'X27'
    assert key_packet(turn, 200000, 2, Kind.LONG).word == 'V27'
    reg = short(turn.last.stream_sample + round(.330 * FS), 59)
    reg = replace(reg, observation=replace(reg.observation, placed=True),
                  responds_to=turn.last.stream_sample)
    assert propose_ack(turn, reg) is not None
    turn.keyed()
    cycle = round(p4turn.CYCLE[Kind.LONG] * FS)
    at = (turn.last.stream_sample + round(1.25 * FS)
          - round(turn.t[Kind.LONG] * FS) + cycle)
    due = at + round(turn.t[Kind.LONG] * FS)
    decoded = packet(at, 3, Kind.LONG, {})
    turn.observe(decoded, now=due + 1)
    if heard == 'unread':
        turn.observe(replace(decoded, observation=replace(decoded.observation,
            stream_sample=at + cycle + jitter, frame=None)))
    else:
        turn.silence(due + cycle - turn.lead, answer_due=due + cycle)
    answer = turn.decide(due + cycle - turn.lead)
    assert answer.word == 'X27' and answer.peer_control is None
    assert answer.ack_identity == (decoded.observation.frame.status.byte, decoded.observation.frame.data)
    assert answer.purpose != 'repeat-request'


@pytest.mark.parametrize('tags,partner', [({'speedlevel': 'SL5-long'}, 'T27'), ({}, 'V27')])
@pytest.mark.parametrize('stale_form', [None, 'normal long', 'robust'])
def test_deferred_first_read_long_keeps_ack(tags, partner, stale_form):
    turn, _ = machine(role='irs')
    turn._ack_form = stale_form
    cycle = round(p4turn.CYCLE[Kind.LONG] * FS)
    due = FS + round(turn.t[Kind.LONG] * FS)
    decoded = packet(FS, 0, Kind.LONG, tags)
    turn.observe(decoded, now=due + 1)
    turn.observe(replace(decoded, observation=replace(decoded.observation,
        stream_sample=FS + cycle, frame=None)))
    answer = turn.decide(due + cycle - turn.lead)
    assert answer.word == 'X27' and answer.purpose != 'repeat-request'
    assert answer.ack_identity == (decoded.observation.frame.status.byte, decoded.observation.frame.data)
    assert not answer.current_slot_accepted and answer.current_slot == FS + cycle
    turn.keyed()
    assert turn._ack_chain == (0, 'X27') and turn._unread_data_due is None


@pytest.mark.parametrize('tags,partner', [({'speedlevel': 'SL5-long'}, 'T27'), ({}, 'V27')])
@pytest.mark.parametrize('stale_form', [None, 'normal long', 'robust'])
def test_unread_first_long_uses_packet_form(tags, partner, stale_form):
    turn, _ = machine(role='irs')
    turn._ack_form = stale_form
    at = 480000
    unread = replace(body(at).observation, kind=Kind.LONG, frame=None, **tags)
    due = at + round(turn.t[Kind.LONG] * FS)
    turn.observe(RegisteredObservation(unread, Registered.NEW), now=due - turn.lead)
    answer = turn.decide(due - turn.lead)
    assert answer.word == partner and answer.ack_identity is None
    assert answer.purpose == 'repeat-request'


@pytest.mark.parametrize('kind,tags', FORMS)
@pytest.mark.parametrize('miss', ['decode', 'deadline'])
def test_deferred_repeat_log_matches_keyed_word(kind, tags, miss):
    turn, _ = machine(role='irs')
    key_packet(turn, FS, 0, Kind.NORMAL)
    key_packet(turn, 120000, 1, Kind.NORMAL)
    at = 240000
    cycle = round(p4turn.CYCLE[kind] * FS)
    due = at + round(turn.t[kind] * FS)
    turn.observe(packet(at, 2, kind, tags), now=due + 1 if miss == 'decode' else None)
    if miss == 'deadline':
        assert turn.decide(due - turn.lead) is not None
        turn.refused('deadline')
    answer = turn.decide(due + cycle - turn.lead)
    assert answer.word == 'X27' and answer.purpose != 'repeat-request'
    assert answer.ack_identity is not None and not answer.current_slot_accepted
    assert answer.current_slot == at + cycle
    event = turn.link_events[-1]
    assert (event['event'], event['word'], event['status'], event['counter']) == (
        'long-answer' if kind == Kind.LONG else 'normal-answer', answer.word,
        answer.ack_identity[0], 2)
    assert not event['current_slot_accepted']
    assert event['current_slot'] == answer.current_slot
    assert turn.keyed() == answer
    assert turn._ack_chain == (2, 'X27') and turn._unread_data_due is None


@pytest.mark.parametrize('kind,width', [(Kind.LONG, 40), (Kind.CHIRP, 22)])
@pytest.mark.parametrize('current', ['absent', 'read', 'unread'])
def test_deferred_control_owner_and_current_data_slot(kind, width, current):
    turn, _ = machine(role='irs')
    at = FS
    cycle = round(p4turn.CYCLE[kind] * FS)
    due = at + round(turn.t[kind] * FS)
    reg = accepted(at, b'\x1e' * width, 0x40, kind)
    turn.observe(reg)
    selected = turn.decide(due - turn.lead)
    assert selected.word == 'X27' and selected.ack_identity == (0x40, reg.observation.frame.data)
    turn.refused('deadline')
    if current != 'absent':
        obs = replace(reg.observation, stream_sample=at + cycle,
                      frame=None if current == 'unread' else reg.observation.frame)
        turn.observe(replace(reg, observation=obs))
    answer = turn.decide(due + cycle - turn.lead)
    assert answer.stream_sample == due + cycle
    assert answer.word == selected.word and answer.ack_identity == selected.ack_identity
    assert answer.current_slot_accepted == (current == 'read')
    assert turn.grants == turn.breakins == 0


def test_unread_form_change_keeps_read_unacknowledged_owner():
    turn, _ = machine(role='irs', normal_ack='parity', first_ack='v27')
    reg = body(FS, b'prior')
    turn.observe(reg)
    selected = turn.decide(FS + round(turn.t[Kind.BODY] * FS) - turn.lead)
    assert selected.word == 'X27' and selected.ack_identity is not None
    turn.refused('deadline')
    at = FS + round(p4turn.CYCLE[Kind.BODY] * FS)
    unread = replace(reg.observation, stream_sample=at, kind=Kind.LONG, frame=None)
    turn.observe(replace(reg, observation=unread))
    answer = turn.decide(at + round(turn.t[Kind.LONG] * FS) - turn.lead)
    assert answer.word == 'X27' and answer.ack_identity == selected.ack_identity
    assert answer.purpose != 'repeat-request' and not answer.current_slot_accepted
    turn.keyed()
    assert turn._last_ack[0] == selected.ack_identity
    assert turn._ack_chain == (selected.ack_identity[0] & 3, 'X27')


@pytest.mark.parametrize('tags,partner', [({}, 'V27'), ({'speedlevel': 'SL5-long'}, 'T27')])
def test_chain_start_repeat_request_does_not_start_epoch(tags, partner):
    turn, _ = machine(role='irs')
    cycle = round(p4turn.CYCLE[Kind.LONG] * FS)
    due = FS + round(turn.t[Kind.LONG] * FS)
    decoded = packet(FS, 0, Kind.LONG, tags)
    turn.observe(replace(decoded, observation=replace(decoded.observation, frame=None)))
    turn.decide(due + 1)
    turn.observe(replace(decoded, observation=replace(decoded.observation,
        stream_sample=FS + cycle, frame=None)))
    answer = turn.decide(due + cycle - turn.lead)
    assert answer.word == partner and answer.purpose == 'repeat-request'
    turn.keyed()
    assert turn.keyed_ack['chain_epoch'] == 0
    assert answer.ack_reason is None
    events = [e for e in turn.link_events if e.get('event') == 'repeat-request']
    assert len(events) == 1 and events[0]['reason'] is None
    at = FS + 2 * cycle
    turn.observe(packet(at, 0, Kind.LONG, tags), now=at)
    answer = turn.decide(at + round(turn.t[Kind.LONG] * FS) - turn.lead)
    assert answer.word == 'X27' and answer.ack_identity is not None
    turn.keyed()
    assert turn.keyed_ack['chain_epoch'] == 1


@pytest.mark.parametrize('chain_reset,word', [('turn,form', 'T27'), ('turn', 'V27')])
@pytest.mark.parametrize('provisional', [False, True])
def test_unread_normal_long_copies_and_silence_keep_one_word(chain_reset, word, provisional):
    turn, _ = machine(role='irs', chain_reset=chain_reset)
    assert key_packet(turn, FS, 0, Kind.NORMAL).word == 'X27'
    assert key_packet(turn, 120000, 1, Kind.NORMAL).word == 'V27'
    at = 240000
    due = at + round(turn.t[Kind.LONG] * FS)
    cycle = round(p4turn.CYCLE[Kind.LONG] * FS)
    unread = replace(packet(at, 3, Kind.LONG, {'speedlevel': 'SL5-long'}).observation,
                     frame=None, provisional=provisional)
    turn.observe(RegisteredObservation(unread, Registered.NEW))
    first = turn.decide(due - turn.lead)
    assert first.word == word and not first.current_slot_accepted
    turn._next = None
    turn.silence(due - turn.lead, answer_due=due)
    assert turn.decide(due - turn.lead).word == word
    turn.keyed()
    turn.silence(due + cycle - turn.lead, answer_due=due + cycle)
    assert turn.decide(due + cycle - turn.lead).word == word
    turn.keyed()
    turn.observe(RegisteredObservation(replace(unread, stream_sample=at + 2 * cycle,
                                               provisional=False), Registered.NEW))
    assert turn.decide(due + 2 * cycle - turn.lead).word == word


@pytest.mark.parametrize('kind,tags', FORMS)
def test_deferred_read_copy_is_accepted_and_logged_once(kind, tags):
    turn, _ = machine(role='irs')
    cycle = round(p4turn.CYCLE[kind] * FS)
    due = FS + round(turn.t[kind] * FS)
    decoded = packet(FS, 0, kind, tags)
    turn.observe(decoded, now=due + 1)
    turn.observe(replace(decoded, observation=replace(decoded.observation,
        stream_sample=FS + cycle)))
    start = len(turn.link_events)
    answer = turn.decide(due + cycle - turn.lead)
    assert answer.word == 'X27' and answer.ack_identity is not None
    assert answer.current_slot_accepted and answer.purpose != 'repeat-request'
    events = [e for e in turn.link_events[start:] if e.get('event') in (
        'acked-repeat', 'normal-answer', 'long-answer', 'repeat-request')]
    assert len(events) == 1
    assert events[0]['status'] == answer.ack_identity[0]
    assert events[0]['word'] == turn.keyed().word


@pytest.mark.parametrize('kind,tags', FORMS)
def test_unread_packet_read_before_deferred_redecision_is_accepted(kind, tags):
    turn, _ = machine(role='irs')
    due = FS + round(turn.t[kind] * FS)
    decoded = packet(FS, 0, kind, tags)
    unread = replace(decoded, observation=replace(decoded.observation, frame=None))
    turn.observe(unread)
    request = turn.decide(due - turn.lead)
    assert request.purpose == 'repeat-request'
    turn.keyed()
    turn.observe(replace(decoded, outcome=Registered.CONFIRMED, replaced=unread.observation),
                 now=due + 1)
    cycle = round(p4turn.CYCLE[kind] * FS)
    answer = turn.decide(due + cycle - turn.lead)
    assert answer.word == 'X27' and answer.ack_identity is not None
    assert answer.purpose != 'repeat-request'
    assert turn.keyed().ack_identity == (decoded.observation.frame.status.byte,
                                         decoded.observation.frame.data)


def test_deferred_retained_control_answer_logs_once():
    turn, _ = machine(role='irs')
    due = FS + round(turn.t[Kind.LONG] * FS)
    decoded = accepted(FS, b'\x1e' * 40, 0x40, Kind.LONG)
    turn.observe(decoded)
    selected = turn.decide(due - turn.lead)
    turn.refused('deadline')
    start = len(turn.link_events)
    answer = turn.decide(due + round(p4turn.CYCLE[Kind.LONG] * FS) - turn.lead)
    events = [e for e in turn.link_events[start:] if e.get('event') in (
        'acked-repeat', 'normal-answer', 'long-answer', 'repeat-request')]
    assert len(events) == 1
    assert answer.ack_identity == selected.ack_identity
    assert events[0]['status'] == answer.ack_identity[0]
    assert events[0]['word'] == turn.keyed().word


@pytest.mark.parametrize('kind,tags', FORMS)
@pytest.mark.parametrize('miss', ['request', 'refused', 'expired', 'late'])
@pytest.mark.parametrize('heard', ['unread', 'nothing'])
@pytest.mark.parametrize('first_ack', ['x27', 'v27'])
def test_read_packet_owns_next_ack_until_keyed(kind, tags, miss, heard, first_ack):
    turn, _ = machine(role='irs', first_ack=first_ack)
    key_packet(turn, FS, 0, kind, **tags)
    cycle = round(p4turn.CYCLE[kind] * FS)
    at = FS + cycle
    due = at + round(turn.t[kind] * FS)
    decoded = packet(at, 1, kind, tags)
    identity = decoded.observation.frame.status.byte, decoded.observation.frame.data
    expected = ('T27' if tags.get('speedlevel') else 'V27') if first_ack == 'x27' else 'X27'
    turn.observe(decoded, now=due + 1 if miss in ('request', 'late') else at)
    if miss in ('refused', 'expired'):
        assert turn.decide(due - turn.lead).word == expected
        if miss == 'refused':
            turn.refused('deadline')
        else:
            assert turn.decide(due + 1) is None
    if miss == 'request':
        turn._next = p4turn.Proposal(Kind.WORD, due, word=turn.last.word,
            state=p4turn.State.IRS_ANSWER, purpose='repeat-request', current_slot_accepted=False)
        turn.keyed()
        assert turn._read_unacknowledged()
    if heard == 'unread':
        turn.observe(replace(decoded, observation=replace(decoded.observation,
            stream_sample=at + cycle, frame=None)))
    else:
        turn.silence(due + cycle - turn.lead, answer_due=due + cycle)
    answer = turn.decide(due + cycle - turn.lead)
    assert answer.word == expected and answer.ack_identity == identity
    assert answer.accepted_target == at and answer.purpose != 'repeat-request'
    assert not answer.current_slot_accepted
    turn.keyed()
    assert turn._unread_data_due is None and not turn._read_unacknowledged()
    turn.observe(replace(decoded, observation=replace(decoded.observation,
        stream_sample=at + 2 * cycle, frame=None)))
    repeat = turn.decide(due + 2 * cycle - turn.lead)
    assert repeat.word == expected and repeat.ack_identity is None
    assert repeat.purpose == 'repeat-request'
