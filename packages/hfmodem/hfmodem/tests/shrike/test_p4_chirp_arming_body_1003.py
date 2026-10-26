# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Admitted none/body links receive the peer's existing SL1 chirp form."""

from dataclasses import replace
from contextlib import redirect_stdout
from inspect import getsource
from os import devnull
from time import perf_counter
import json
from pathlib import Path

import pytest

from hfmodem.shrike import onair, p4chirp, p4chirprx
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.shrike.p4turn import P4Turn, Proposal, State
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike import p4_chirp_deadline_support as support
from hfmodem.tests.shrike.test_p4turn import ack_frame
from hfmodem.tests.shrike.p4_nak_qam_support import (
    neutral_keys, waveform_keys, without_neutral_requests,
)


STARTS = (1248792, 1608702, 1788749, 1968723)
FIELD = b'RMS Trimode 1.4.4.0\r\nW'


def recorded(monkeypatch):
    """Charge every packet read, including the selected chirp CRC, to sample time."""
    calls = []
    deliver = []
    receivers = []
    headers = []
    decoding = [False]
    original_tx = support.Tx
    streams = []
    refusal_clock = []
    machines = []
    transmitters = []
    observations = []

    def snapshot(stage, proposal, due=None):
        # The seed is preloaded, not a decision/physical key in this replay.
        events = transmitters[0].p4_probe.link_events
        decision = next(i for i in range(len(events)-1, -1, -1)
                        if events[i]['event'] == 'Decision')
        stream = streams[0]
        row = dict(stage=stage, decision=decision, proposal=proposal,
                   facts=vars(proposal).copy(),
                   due=proposal.stream_sample if due is None else due,
                   now=stream.sample_now(), notice=stream.key_notice)
        observations.append(row)
        return row

    class ChargedTx(original_tx):
        def __init__(self, stream):
            super().__init__(stream)
            streams.append(stream)
            transmitters.append(self)
            self.p4_link_host.deliver = deliver.append
            rx = self.sessrx.p4
            receivers.append(rx)
            assert not rx.chirp
            read = rx.read_chirp

            def timed(*args, **kwargs):
                started = perf_counter()
                decoding[0] = True
                try:
                    records = read(*args, **kwargs)
                finally:
                    decoding[0] = False
                seconds = perf_counter()-started
                stream.wall = stream.sample_now()+round(seconds*support.FS)
                calls.append((kwargs['onset'], seconds, records is not None))
                return records

            rx.read_chirp = timed

        def _tx(self, audio, what):
            row = snapshot('physical', machines[0]._next,
                           self.raster.boundary(self.slot))
            result = super()._tx(audio, what)
            row['start'] = self.tx_audio_start
            return result

    seed = P4Turn.seed_opening

    def waiting(machine, *args, **kwargs):
        result = seed(machine, *args, **kwargs)
        machine.allow_breakin = False
        machines.append(machine)
        return result

    acquire = p4chirprx.acquire_header
    refuse = P4Turn.refused
    can_key = P4Turn.can_key
    keyable_slot = onair._keyable_slot

    def eligible(machine, proposal):
        row = snapshot('eligible', proposal)
        held = machine._held_answer
        row['held'] = None if held is None else dict(
            answer=vars(held.answer).copy(), generation=held.generation)
        row['epoch'] = machine._receive_epoch
        row['generation'] = machine._peer_generation
        row['allowed'] = can_key(machine, proposal)
        return row['allowed']

    def admission(live, raster, slot, lead_n, *, listen=True, shift=None):
        row = snapshot('admission', machines[0]._next, raster.boundary(slot))
        row.update(slot=slot, lead=lead_n, listen=listen, shift=shift)
        row['returned'] = keyable_slot(live, raster, slot, lead_n,
                                       listen=listen, shift=shift)
        return row['returned']

    def timed_refusal(machine,reason):
        proposal=machine._next
        snapshot('refusal', proposal)['reason'] = reason
        if proposal is not None and not proposal.current_slot_accepted:
            stream=streams[0]
            now=stream.sample_now()
            refusal_clock.append(dict(at=proposal.stream_sample,word=proposal.word,
                purpose=proposal.purpose,kind=proposal.kind,state=proposal.state,reason=reason,
                now_sample=now,key_notice=stream.key_notice,
                margin_samples=proposal.stream_sample-stream.key_notice-now))
        return refuse(machine,reason)

    def tracked_header(*args, **kwargs):
        before = receivers[0].clock.cycle_s
        header = acquire(*args, **kwargs)
        if header is not None and not decoding[0]:
            at = receivers[0].clock.next_due+header.start_sample-kwargs['nominal_sample']
            if not any(abs(prior-at) < 100 for prior, _, _ in headers):
                headers.append((at, header.marker, before))
        return header

    with monkeypatch.context() as patch, open(devnull, 'w') as sink, redirect_stdout(sink):
        patch.setattr(support, 'Tx', ChargedTx)
        patch.setattr(P4Turn, 'seed_opening', waiting)
        patch.setattr(P4Turn, 'refused', timed_refusal)
        patch.setattr(P4Turn, 'can_key', eligible)
        patch.setattr(onair, '_keyable_slot', admission)
        patch.setattr(p4chirprx, 'acquire_header', tracked_header)
        result = support.replay('onair-1003-1110', measured_work=True)
    result['headers'] = headers
    result['neutral_refusal_clock'] = refusal_clock
    result['arming_observations'] = observations
    return result, calls, deliver


def assert_charged_chirp_receive(result, calls, delivered, neutral):
    complete = [row for row in result['heard']
                if row['kind'] == 'chirp' and not row['provisional']]
    assert [row['onset_sample'] for row in complete] == list(STARTS)
    assert all(row['status'] == 0x61 and bytes.fromhex(row['data']) == FIELD
               for row in complete)
    assert result['headers'][0] == (STARTS[0], 8, 1.25)
    assert [marker for _, marker, _ in result['headers']] == [8, 9, 9, 9, 9]
    assert [at for at, _, read in calls if read] == list(STARTS)
    assert all(seconds > 0 for _, seconds, _ in calls)
    # Actual decode work is charged to sample_now before the real admission check.
    ordinary = {row['at'] for row in result['events']
                if row['event'] == 'Keyed' and row.get('current_slot_accepted', True)}
    replies = [row['audio_start'] for row in result['emissions']
               if row.get('word') == 'X27' and row['audio_start'] >= STARTS[0]
               and row['audio_start'] in ordinary]
    expected = [at+round(3.4*support.FS) for at in STARTS]
    assert replies and set(replies) <= set(expected)
    assert set(expected[1:]) <= set(replies)
    assert expected[0] in replies or expected[1] in replies
    refusals = result['neutral_refusal_clock']
    # Any new refusal must belong to a captured neutral request. Ordinary
    # receive ACKs retain the original no-refusal condition.
    assert [(e['at'],e['kind'],e['state'],e['reason'])
            for e in result['events'] if e['event']=='Refused'] == [
        (e['at'],e['kind'],e['state'],e['reason']) for e in refusals]
    assert all(e['purpose'] is None and e['margin_samples']<0
               and e['reason']=='deadline' for e in refusals)
    assert not {e['at'] for e in refusals} & {e['audio_start'] for e in result['emissions']}
    assert all(not (at < row['audio_start'] < at+round(p4chirp.PACKET_S*support.FS))
               for at in STARTS for row in result['emissions'])
    ticks = [at for at, _ in result['cycles'] if at >= STARTS[0]]
    assert len(ticks) == 5
    assert all(abs((right-left)-round(3.75*support.FS)) < 100
               for left, right in zip(ticks, ticks[1:]))
    assert result['reason'] != 'peer silent'
    assert delivered == [b'RMS Trimode 1.4.4.0\r\nW9SSJ has ', FIELD]
    # The next complete local copy is unread; the final marker is truncated.
    assert not any(abs(row['onset_sample']-1428792) < 100 for row in complete)
    assert any(row['kind'] == 'chirp' and row['provisional']
               and abs(row['onset_sample']-2148862) < 100 for row in result['heard'])


def assert_proposal_admission(result, neutral, refused):
    """Join ordered decisions to real eligibility, admission and physical seams."""
    decisions = []
    for index, event in enumerate(result['events']):
        if event['event'] == 'Decision':
            decisions.append((index, event, []))
        elif event['event'] in ('Keyed', 'Refused'):
            assert decisions, event
            decisions[-1][2].append(event)
    observations = iter(result['arming_observations'])
    pending = next(observations, None)
    neutral_outcomes, neutral_refusals, clocks, physical_starts = [], [], [], []
    for index, decision, outcomes in decisions:
        rows = []
        while pending is not None and pending['decision'] == index:
            rows.append(pending)
            pending = next(observations, None)
        if decision['proposed_at'] is None:
            assert not rows and not outcomes, (decision, rows, outcomes)
            continue
        assert rows and len(outcomes) == 1, (decision, rows, outcomes)
        first = rows[0]
        proposal, facts = first['proposal'], first['facts']
        due = decision['proposed_at']
        assert facts['stream_sample'] == due
        assert first['stage'] == 'eligible', rows
        assert all(row['proposal'] is proposal and row['facts'] == facts
                   and row['due'] == due for row in rows), rows
        gates = [row for row in rows if row['stage'] == 'eligible']
        assert all(row['allowed'] is True for row in gates), gates
        admissions = [row for row in rows if row['stage'] == 'admission']
        assert len(admissions) == 1, rows
        admission = admissions[0]
        assert (admission['slot'], admission['lead'], admission['listen'],
                admission['shift']) == (0, 0, False, False)
        margin = due-admission['notice']-admission['now']
        assert (admission['returned'] == 0) == (margin >= 0), admission
        outcome, = outcomes
        assert outcome['at'] == due and outcome['kind'] == facts['kind'], outcome
        is_neutral = not facts['current_slot_accepted']
        if is_neutral:
            assert (facts['kind'], facts['state'], facts['payload'],
                    facts['purpose']) == ('word', 'IRS_ANSWER', b'', None)
            held = first['held']
            valid = (held is not None
                     and held['answer']['receive_epoch'] == first['epoch']
                     and held['generation'] == first['generation'])
            expected = held['answer'] if valid else vars(Proposal(
                facts['kind'], due, word='V27', state=State.IRS_ANSWER))
            # Only placement and acceptance change when copying the held answer.
            assert {k: v for k, v in facts.items() if k not in (
                'stream_sample', 'current_slot', 'current_slot_accepted')} == {
                k: v for k, v in expected.items() if k not in (
                    'stream_sample', 'current_slot', 'current_slot_accepted')}
        record = dict(at=due, **{k: facts[k] for k in (
            'word', 'purpose', 'kind', 'state')})
        if margin >= 0:
            assert [row['stage'] for row in rows] == [
                'eligible', 'admission', 'eligible', 'physical'], rows
            assert len(gates) == 2 and outcome['event'] == 'Keyed', outcome
            physical = rows[-1]
            assert due-physical['notice']-physical['now'] >= 0, physical
            assert physical['start'] == due
            physical_starts.append(due)
            for name in ('word', 'purpose', 'accepted_target', 'current_slot',
                         'current_slot_accepted'):
                assert outcome[name] == facts[name], outcome
            identity = facts['ack_identity']
            assert outcome['ack_identity'] == (
                (identity[0], identity[1].hex()) if identity is not None else None)
            if is_neutral:
                assert outcome['state'] == facts['state']
                neutral_outcomes.append(record)
        else:
            assert [row['stage'] for row in rows] == [
                'eligible', 'admission', 'refusal'], rows
            assert len(gates) == 1 and is_neutral, rows
            assert outcome['event'] == 'Refused' and outcome['reason'] == 'deadline'
            assert outcome['state'] == facts['state']
            refusal = rows[-1]
            assert refusal['reason'] == 'deadline'
            refusal_margin = due-refusal['notice']-refusal['now']
            assert refusal_margin < 0, refusal
            record = dict(record, reason='deadline')
            neutral_refusals.append(record)
            clocks.append(dict(record, now_sample=refusal['now'],
                               key_notice=refusal['notice'],
                               margin_samples=refusal_margin))
    assert pending is None, pending
    assert neutral_outcomes == neutral
    assert neutral_refusals == refused
    assert clocks == result['neutral_refusal_clock']
    # The first emission is the explicitly preloaded opening seed.
    assert [row['audio_start'] for row in result['emissions'][1:]] == physical_starts
    assert all(row['proposed_at'] == row['keyed_at'] == row['audio_start']
               for row in result['emissions'][1:])


def assert_word_waveforms(result, waves, pin):
    words = {row['at'] for row in result['events']
             if row['event'] == 'Keyed' and row.get('word') == 'X27'}
    assert words == {row['at'] for row in waves}
    assert [row['at'] for row in waves] == [
        row['start'] for row in result['arming_observations']
        if row['stage'] == 'physical']
    assert all({k: row[k] for k in ('samples', 'dtype', 'sha256')} == pin
               for row in waves)


def test_none_body_driver_receives_chirps_with_charged_work(monkeypatch):
    refused=[]
    with neutral_keys(refused=refused) as neutral, waveform_keys() as waves:
        result, calls, delivered = recorded(monkeypatch)
    pins = json.loads((Path(__file__).with_name('fixtures')/
                       'nak-stack1003e-controls.json').read_text())['onair-1003-1110']
    assert_proposal_admission(result, neutral, refused)
    assert_charged_chirp_receive(result, calls, delivered, neutral)
    # Both real ACKs and neutral requests key the frozen X27 waveform; only
    # whether the keyed slot was accepted distinguishes the two, never the word's name.
    assert_word_waveforms(result, waves, pins['word_waveform'])


def arming_example():
    """One equality-boundary held key and one late default-word refusal."""
    keyed = Proposal('word', 100, word='X27', state=State.IRS_ANSWER,
                     receive_epoch=3, source_root=('body', 40),
                     accepted_target=40, current_slot=90,
                     ack_identity=(0, b'field'), current_slot_accepted=False)
    late = Proposal('word', 200, word='V27', state=State.IRS_ANSWER,
                    current_slot=190, current_slot_accepted=False)
    held = replace(keyed, stream_sample=50, current_slot=40,
                   current_slot_accepted=True)
    first = dict(decision=0, proposal=keyed, facts=vars(keyed).copy(),
                 due=100, now=90, notice=10)
    second = dict(decision=2, proposal=late, facts=vars(late).copy(),
                  due=200, now=191, notice=10)
    source = dict(held=dict(answer=vars(held).copy(), generation=4),
                  epoch=3, generation=4)
    neutral = [dict(at=100, word='X27', purpose=None, kind='word', state='IRS_ANSWER')]
    refused = [dict(at=200, word='V27', purpose=None, kind='word',
                    state='IRS_ANSWER', reason='deadline')]
    result = dict(
        events=[dict(event='Decision', proposed_at=100),
                dict(event='Keyed', **neutral[0], accepted_target=40,
                     current_slot=90, current_slot_accepted=False,
                     ack_identity=(0, b'field'.hex())),
                dict(event='Decision', proposed_at=200),
                dict(event='Refused', at=200, kind='word', state='IRS_ANSWER',
                     reason='deadline')],
        arming_observations=[
            dict(first, stage='eligible', allowed=True, **source),
            dict(first, stage='admission', slot=0, lead=0, listen=False,
                 shift=False, returned=0),
            dict(first, stage='eligible', allowed=True, **source),
            dict(first, stage='physical', start=100),
            dict(second, stage='eligible', allowed=True, held=None,
                 epoch=3, generation=4),
            dict(second, stage='admission', slot=0, lead=0, listen=False,
                 shift=False, returned=1),
            dict(second, stage='refusal', reason='deadline')],
        neutral_refusal_clock=[dict(refused[0], now_sample=191, key_notice=10,
                                    margin_samples=-1)],
        emissions=[dict(audio_start=0),
                   dict(audio_start=100, proposed_at=100, keyed_at=100)])
    return result, neutral, refused


def test_proposal_admission_accepts_on_time_and_late_examples():
    assert_proposal_admission(*arming_example())


@pytest.mark.parametrize('fault', [
    'late key', 'lost on-time key', 'unexplained refusal', 'duplicate outcome',
    'missing outcome', 'ineligible', 'wrong held source', 'ordinary refusal',
])
def test_proposal_admission_rejects_broken_contract(fault):
    result, neutral, refused = arming_example()
    rows = result['arming_observations']
    if fault == 'late key':
        rows[3]['now'] = 91
    elif fault == 'lost on-time key':
        rows.pop(3)
        rows.pop(2)
        rows.insert(2, dict(rows[0], stage='refusal', reason='deadline'))
        result['events'][1] = dict(event='Refused', at=100, kind='word',
                                   state='IRS_ANSWER', reason='deadline')
        result['emissions'].pop()
    elif fault == 'unexplained refusal':
        result['events'][-1]['reason'] = 'admission'
        rows[-1]['reason'] = 'admission'
    elif fault == 'duplicate outcome':
        result['events'].append(dict(result['events'][-1]))
    elif fault == 'missing outcome':
        result['events'].pop()
    elif fault == 'ineligible':
        rows[2]['allowed'] = False
    elif fault == 'wrong held source':
        rows[0]['held']['answer']['source_root'] = ('body', 41)
    else:
        for row in rows[4:]:
            row['facts']['current_slot_accepted'] = True
    with pytest.raises(AssertionError):
        assert_proposal_admission(result, neutral, refused)


def test_neutral_history_mutation_is_rejected(monkeypatch):
    from hfmodem.tests.shrike.test_p4turn import machine

    turn, _ = machine()
    turn.state = State.IRS_ANSWER
    turn._next = Proposal('word', 100, word='V27', state=State.IRS_ANSWER,
                          current_slot_accepted=False)
    keyed = P4Turn.keyed

    def mutate_history(self, *args, **kwargs):
        sent = keyed(self, *args, **kwargs)
        self.ack_word = 'mutated history'
        return sent

    monkeypatch.setattr(P4Turn, 'keyed', mutate_history)
    with neutral_keys(), pytest.raises(AssertionError):
        turn.keyed(100)


def test_word_waveform_digest_mutation_is_rejected():
    result, _, _ = arming_example()
    pin = json.loads((Path(__file__).with_name('fixtures')/
                      'nak-stack1003e-controls.json').read_text())[
                          'onair-1003-1110']['word_waveform']
    waves = [dict(at=100, **pin)]
    assert_word_waveforms(result, waves, pin)
    waves[0]['sha256'] = '0'*64
    with pytest.raises(AssertionError):
        assert_word_waveforms(result, waves, pin)


def test_charged_chirp_without_neutral_requests_keeps_original_receive(monkeypatch):
    without_neutral_requests(monkeypatch)
    with neutral_keys() as neutral:
        result, calls, delivered = recorded(monkeypatch)
    assert neutral == []
    assert result['neutral_refusal_clock'] == []
    assert_charged_chirp_receive(result, calls, delivered, neutral)


def test_recording_pin_fails_with_previous_transmit_option_gate(monkeypatch):
    source = getsource(onair._run_p4_link)
    original = 'receiver.chirp = True'
    assert original in source
    source = source.replace(original,
        "receiver.chirp |= args.p4_ladder == 'chirp' or args.p4_data == 'chirp'", 1)
    namespace = dict(vars(onair))
    exec(compile(source, 'previous_chirp_arming', 'exec'), namespace)
    monkeypatch.setattr(onair, '_run_p4_link', namespace['_run_p4_link'])
    result, calls, delivered = recorded(monkeypatch)
    assert not calls
    assert not [row for row in result['heard'] if row['kind'] == 'chirp']
    assert result['reason'] == 'peer silent'
    assert delivered == [b'RMS Trimode 1.4.4.0\r\nW9SSJ has ']


@pytest.mark.parametrize('enabled', [False, True])
def test_standalone_disabled_receiver_keeps_its_gate(enabled):
    path = CAPTURES/'onair-1003-1110/stream.flac'
    require_input(path)
    audio = onair.session.load_wav(str(path), support.FS)
    nominal = 1248463
    rx = P4Receiver(support.FS, admitted=True, chirp=enabled)
    rx.expected = True
    rx.clock = Clock(Expect.PACKET, nominal, .330, 1.25, 2.13481)
    rx.read_partial(audio[nominal-2400:nominal+round(.81*support.FS)],
                    nominal-2400, deadline=nominal+round(.81*support.FS))
    assert bool(rx._chirp_headers) is enabled
    assert rx.clock.cycle_s == (3.75 if enabled else 1.25)


@pytest.fixture(scope='module')
def unread_marker_audio():
    path = CAPTURES/'onair-0929-0936/stream.flac'
    require_input(path)
    return onair.session.load_wav(str(path), support.FS)


def unread_marker_receiver(audio, cycle=1.25):
    nominal, onset = 2147800, 2149224
    origin = nominal-round(.5*support.FS)
    rx = P4Receiver(support.FS, admitted=True, chirp=True)
    rx.expected = True
    rx.clock = Clock(Expect.PACKET, nominal, .330, cycle, -2.5703200253367777)
    early = nominal+round(.845*support.FS)
    heard = rx.read_partial(audio[origin:early], origin, deadline=early)
    assert any(reg.observation.kind == 'chirp' and reg.observation.provisional
               and reg.observation.stream_sample == onset for reg in heard)
    assert rx.has_chirp_header(onset) and rx.clock.cycle_s == 3.75
    return rx, origin, onset


@pytest.mark.parametrize('previous', [1.25, 3.75])
@pytest.mark.parametrize('path', ['fast', 'ordinary', 'expiry'])
def test_recorded_unread_marker_releases_only_its_provisional_cadence(
        unread_marker_audio, monkeypatch, previous, path):
    audio = unread_marker_audio
    rx, origin, onset = unread_marker_receiver(audio, previous)
    end = onset+round(p4chirp.PACKET_S*support.FS)
    if path == 'fast':
        assert rx.read_chirp(audio[origin:end], origin, onset=onset) is None
    elif path == 'ordinary':
        rx.read_window(audio[origin:end], origin, whole=True)
    else:
        # Expiry must release the marker even when a decoder is not dispatched.
        monkeypatch.setattr(rx, '_packet', lambda *a, **kw: None)
        rx.read_window(audio[origin:onset+round(3.36*support.FS)], origin, whole=True)
    assert not rx.has_chirp_header(onset)
    assert rx.clock.cycle_s == previous
    assert not any(o.kind == 'chirp' and o.frame is not None for o in rx.observations)
    assert [row['restored_cycle_s'] for row in rx.live_events
            if row['event'] == 'chirp header unread'] == [previous]
    # The same retained buffer must not reacquire the rejected marker.
    rx._packet(audio[origin:end], origin)
    assert rx.clock.cycle_s == previous


@pytest.mark.parametrize('new_owner', ['clock', 'generation'])
def test_expired_chirp_marker_cannot_restore_a_stale_owner_cadence(
        unread_marker_audio, new_owner):
    rx, _, onset = unread_marker_receiver(unread_marker_audio)
    if new_owner == 'clock':
        rx.clock = replace(rx.clock)
    else:
        rx._train_generation += 1
    rx._reject_chirp_header(onset)
    assert rx.clock.cycle_s == 3.75
    assert not [row for row in rx.live_events if row['event'] == 'chirp header unread']


@pytest.mark.parametrize('kind,cycle', [('long', 3.75), ('normal', 1.25)])
def test_expired_chirp_marker_cannot_displace_a_checked_packet_cadence(
        unread_marker_audio, kind, cycle):
    rx, _, onset = unread_marker_receiver(unread_marker_audio, cycle=3.75)
    checked = ack_frame(onset, b'checked source', 0x61, kind=kind).observation
    rx._emit(checked, [])
    assert rx.clock.cycle_s == cycle
    rx._reject_chirp_header(onset)
    assert rx.clock.cycle_s == cycle
    assert not [row for row in rx.live_events if row['event'] == 'chirp header unread']
