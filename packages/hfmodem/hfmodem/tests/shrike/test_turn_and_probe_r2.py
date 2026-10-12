# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Unit 1 pins: evidence integrity and replay mechanics, no protocol fix claim."""
from dataclasses import replace
from types import SimpleNamespace
from collections import Counter
from copy import deepcopy
import json

import numpy as np
import pytest

from hfmodem.shrike import onair, p4breakintx, p4frame, p4robusttx
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Emission, Expect, P4Receiver, Role
from hfmodem.shrike.p4turn import Proposal, State
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
from hfmodem.tests.shrike.test_p4turn import body, machine
from hfmodem.tests.shrike.r2_replay import (
    BoundaryError, ByteLedger, Clocks, FIXTURES, FS, KeyOutcomes,
    ReactivePeer, ReplayStream, Tape, canonical, digest, manifest,
    snapshot, require_input,
)

if not (FIXTURES/'census.json').exists():
    pytest.skip('turn-and-probe-r2 fixtures is not in this tree', allow_module_level=True)

REQUIRED = {'1001-1022', '1001-1036', '1001-0611', '0930-0146',
            '0930-2218', '0930-2223', '1001-0514', '1001-1045',
            '1001-1042', '1001-1012', '0929-0121', '1001-0344',
            '1001-1029', '0926-1342'}


def test_r2_fixture_contract():
    m = manifest()
    assert REQUIRED <= m['fixtures'].keys()
    for arm in m['fixtures']:
        tape = Tape(arm)
        assert tape.contract['options']
        assert tape.contract['own_tx_masks'] or tape.contract['trusted_rx_intervals']
        # Explicitly resolve old .wav pathnames to the hashed physical file.
        assert tape.contract['stream'].endswith(('stream.flac', 'stream.wav'))
    assert m['fixtures']['0930-2223']['options']['p4-ladder'] == 'none'
    assert m['fixtures']['1001-1022']['metadata']['sha256'] == '82f725baa89532b5c18028f27ad3fc9a5757b2ca8779a8d16fb43dd1f744fb1e'
    with pytest.raises(BoundaryError, match='unmanifested'):
        Tape('invented')
    for field in ('origin', 'hash', 'mask'):
        row = deepcopy(m['fixtures']['1001-1022'])
        if field == 'origin':
            row['metadata']['first_stream_sample'] = 1
        elif field == 'hash':
            row['files'][row['stream']] = '0' * 64
        else:
            row['own_tx_masks'] = []
        with pytest.raises(BoundaryError):
            Tape('1001-1022', contract=row)
    tape = Tape('1001-1022')
    rx = P4Receiver(FS, admitted=True)
    a, b = tape.contract['own_tx_masks'][0]
    with pytest.raises(BoundaryError, match='admitted as RX'):
        tape.read(rx, a, b, mask=False)
    with pytest.raises(BoundaryError, match='overlaps'):
        tape.read(rx, a, b, needed=(a, b))
    assert tape.read(rx, a, b) == []
    assert (a, b) in rx.capture_gaps
    first = tape.contract['emissions'][0]
    em = Emission(Kind.SHORT, Role.ISS, first['audio_start'],
                  (first['audio_end']-first['audio_start'])/FS)
    with pytest.raises(BoundaryError, match='future'):
        tape.register_recorded(rx, em, available_at=em.stream_sample-1)
    tape.diverge(em.stream_sample)
    with pytest.raises(BoundaryError, match='post-divergence'):
        tape.register_recorded(rx, em, available_at=first['audio_end'])
    with pytest.raises(BoundaryError, match='after divergence'):
        tape.read(rx, first['audio_end'], first['audio_end']+FS//10)


def recount(logs):
    result = []
    for log in logs:
        words = sorted(set(map(tuple, log['words'])))
        i = 0
        while i < len(words):
            j = i + 1
            while j < len(words) and words[j][1] == words[i][1] and words[j][0]-words[j-1][0] < 1.5:
                j += 1
            if words[i][1] in ('T27', 'W27', 'Z27') and j-i >= 3:
                f = words[j] if j < len(words) else None
                category = ('end' if f is None else 'gap' if f[0]-words[j-1][0] >= 1.5
                            else 'x27' if f[1] == 'X27' else 'other')
                result.append(dict(arm=log['path'].split('/')[1][6:],
                                   answers=words[i:j], following=f, category=category))
            i = j
    return canonical(result)


def test_r2_census_contract():
    assert digest(FIXTURES/'census.json') == manifest()['census_sha256']
    c = json.loads((FIXTURES/'census.json').read_text())
    assert recount(c['logs']) == c['expanded']
    assert len(c['expanded']) == 101
    assert Counter(r['category'] for r in c['expanded']) == dict(x27=43, other=38, end=14, gap=6)
    saved = c['saved']
    assert len(saved) == 36
    assert Counter(r['answers'][0][1] for r in saved) == {'T27': 32, 'Z27': 3, 'W27': 1}
    assert all(r in c['expanded'] and len(r['answers']) == 6 for r in saved)
    assert all(7.498-1e-9 <= r['following'][0]-r['answers'][0][0] <= 7.502+1e-9 for r in saved)
    assert len(next(r for r in saved if r['answers'][0][1] == 'W27')['answers']) == 6
    costs = c['costs']
    assert len(costs) == 109 and sum(bool(r['reads']) for r in costs) == 105
    reads = [x for r in costs for x in r['reads']]
    assert len(reads) == 898
    assert sum(x['cost_ms'] > 150 for x in reads) == 201
    assert sum(x['cost_ms'] > 150 and x['overrun_next3'] for x in reads) == 131
    assert sum(len(r['overrun_lines']) for r in costs) == 314


@pytest.mark.parametrize('arm', sorted({r['arm'] for r in json.loads((FIXTURES/'census.json').read_text())['expanded']}))
def test_r2_census_recordings_available(arm):
    from hfmodem.tests.evidence import CAPTURES
    directory = CAPTURES/('onair-'+arm)
    path = directory/'stream.flac'
    require_input(path if path.exists() else directory/'stream.wav')


@pytest.mark.parametrize('branch', ['ACK', 'refusal', 'body', 'silence', 'P1 continuation'])
def test_r2_reactive_peer_and_keyed_only_registration(branch):
    turn, arq = machine()
    arq.on_host_data(b'A'*30 + b'\x1c\x1d\x1eB')
    original = b'A'*30 + b'\x1c\x1d\x1eB'
    ledger = ByteLedger(turn, original)
    ledger.check()
    turn.outbox.transfer()
    ledger.check()
    clocks = Clocks()
    stream = ReplayStream(5*FS, clocks)
    rx = P4Receiver(FS, admitted=True)
    peer = ReactivePeer(branch)
    tx = KeyOutcomes(stream, rx, turn, peer)
    turn._short(FS, 0)
    proposal = turn.decide(0)
    assert proposal is not None
    before = snapshot(turn, rx)
    assert tx.transmit(keyed=False) is None
    assert rx.emissions == [] and peer.triggers == []
    assert snapshot(turn, rx)['turn']['last'] == before['turn']['last']
    ledger.check()
    proposal = turn._next
    assert proposal is not None
    emission = tx.transmit()
    assert peer.triggers == [emission.stream_sample]
    assert len(rx.emissions) == 1
    assert [x[0] for x in tx.trace] == ['Refused', 'Keyed']
    end = emission.stream_sample + round((1.80 if branch == 'body' else 1.24)*FS)
    regs = stream.read_into(rx, turn, end)
    assert all(reg in turn.events for reg in regs)
    if branch in ('ACK', 'refusal'):
        named = [r for r in regs if r.observation.word == ('X27' if branch == 'ACK' else 'Z27')]
        assert named and any(not r.observation.provisional for r in named)
        assert all(r.responds_to == emission.stream_sample for r in named)
    elif branch == 'body':
        assert any(r.observation.kind == Kind.BODY and r.observation.frame is not None for r in regs)
    elif branch == 'silence':
        assert regs == []
    elif branch == 'P1 continuation':
        from hfmodem.shrike import p1rx
        onset = emission.stream_sample + round(.935*FS)
        hit = p1rx.cs_anchored(stream.audio[:end], onset/FS, FS)
        assert hit == p1rx.CS(index=1, errors=0, sense=0)
    ledger.check()
    turn.outbox.requeue()
    ledger.check()
    assert bytes(arq._outbuf).endswith(b'\x1c\x7c\x1c\x7d\x1c\x7eB')


def test_r2_deadline_sample_wall_costs_and_clock_jumps():
    c = Clocks(sample=FS, wall=FS, lag=480)
    due = FS + 6000
    assert c.feasible_due(due, 60000, outstanding=240) == due
    c.work(read=480, refinement=480, build=480, render=480)
    assert c.feasible_due(due, 60000, outstanding=240) == due+60000
    c.wall += 10*FS
    assert c.feasible_due(due, 60000) == due+9*60000
    c.sample += 20*FS
    advanced = c.feasible_due(due, 60000)
    assert advanced >= c.sample+c.lag+c.lead
    assert (advanced-due) % 60000 == 0
    turn, _ = machine()
    turn._short(FS, 0)
    stream = ReplayStream(5*FS, Clocks(wall=FS))
    rx = P4Receiver(FS, admitted=True)
    tx = KeyOutcomes(stream, rx, turn, ReactivePeer('silence'))
    with pytest.raises(BoundaryError, match='infeasible'):
        tx.transmit()
    assert rx.emissions == [] and tx.peer.triggers == []


def test_r2_prefix_state_is_reconstructed_and_replayable():
    tape = Tape('1001-0344')
    first = tape.contract['emissions'][0]
    second = tape.contract['emissions'][1]
    end = second['audio_start']-1
    turn, _ = machine(ladder='none', data='body')
    rx = P4Receiver(FS, admitted=True)
    cut, regs = tape.prefix(turn, rx, end=end)
    assert cut['turn']['last']['stream_sample'] == first['audio_start']
    assert cut['turn']['state'] == 'ISS_LADDER'
    assert cut['queued'] == {'hex': ''} and cut['inflight'] == {'hex': ''}
    assert any(r.observation.word == 'X27' and not r.observation.provisional for r in regs)
    assert all(r.responds_to == first['audio_start'] for r in regs if r.observation.word == 'X27')
    other_turn, _ = machine(ladder='none', data='body')
    other, _ = Tape('1001-0344').prefix(other_turn, P4Receiver(FS, admitted=True), end=end)
    assert other == cut
    frozen = json.loads((FIXTURES/'default-prefix-state.json').read_text())
    assert frozen == dict(arm='1001-0344', end=end, state=cut)
    assert cut['role'] == 'iss'
    with pytest.raises(BoundaryError, match='longer driver'):
        Tape('1001-0344').prefix(other_turn, rx, end=second['audio_end']+1)


def driver_trace():
    """Bounded baseline driver control: body acceptance, duplicate, and repeat."""
    audio = np.zeros(6*FS)
    for seconds, payload in ((.923, b'hello'), (2.173, b'hello'), (3.423, b'next')):
        wave = p4breakintx.changeover_packet(payload, 0 if payload == b'hello' else 1)
        at = round(seconds*FS)
        audio[at:at+len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_link_timeout = 5
    delivered = []
    tx.p4_link_host.deliver = delivered.append
    onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    return canonical(dict(delivered=delivered, emissions=tx.p4_probe.emissions,
                          registrations=[dict(kind=r.observation.kind, word=r.observation.word,
                                              at=r.observation.stream_sample, outcome=r.outcome,
                                              responds_to=r.responds_to)
                                         for r in tx.sessrx.p4.registrations]))


def test_r2_default_driver_trace():
    expected = json.loads((FIXTURES/'default-driver-trace.json').read_text())
    actual = driver_trace()
    # The default schedule is unchanged; timing now records both coordinates.
    # Check the added metadata before comparing every original golden field.
    for emission in actual['emissions'][1:]:
        assert emission.pop('answer_residual_samples') == 0
        assert emission.pop('output_enqueue_sample') is None
        assert emission.pop('nominal_due') == emission['audio_start']
        assert emission.pop('local_due') == emission['audio_start']
        # Keep word-render noise reduction off to preserve the pinned samples.
        assert emission.pop('word_render') == 'off'
        assert emission.pop('word_offset_hz') == 0
    assert actual == expected
    assert expected['delivered'] == [{'hex': b'hello'.hex()}, {'hex': b'next'.hex()}]


def _close_prefix(*, end=75, answer=.912, held=False):
    wall = [0.]
    turn, arq = machine(prekey_lead=.074, answer_offset=answer)
    if hasattr(turn, '_wall_clock'):
        turn._wall_clock = lambda: wall[0]
        turn.session_end = round(end*FS)
    arq.on_host_data(b'pending mail')
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    body_at = 2011485
    short_at = 2071485
    if held:
        turn._peer_word = 'V27'
        turn._word_yield = True
        turn._data_refusals[body_at-60000] = 'V27'
    turn._body(body_at, 0x20, State.ISS_CLOSE)
    body = turn.keyed()
    rx.note_emission(Emission(Kind.BODY, Role.ISS, body.stream_sample, .798))
    turn.silence(short_at-turn.lead)
    assert turn._next.kind == Kind.SHORT and turn._next.value == 37
    short = turn.keyed(short_at)
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, short.stream_sample, .758))
    if hasattr(rx, 'arm_close_lease'):
        rx.arm_close_lease(short.stream_sample, end=turn.close_context.end)
    return turn, arq, rx, wall


def _close_audio(turn, rx, at, wave, *, ready=.810):
    origin = at-round(.20*FS)
    end = at+round(ready*FS)
    audio = np.zeros(end-origin, np.float32)
    audio[at-origin:at-origin+len(wave)] = wave
    regs = rx.read_window(audio, origin, whole=True)
    for reg in regs:
        turn.observe(reg, now=end)
    return regs


def test_r2_1042_exhaustion_proposes_close(monkeypatch):
    tape = Tape('1001-1042')
    rows = [e for e in tape.contract['emissions'] if e['audio_start'] >= 900780]
    original = bytes.fromhex(rows[0]['payload']) + bytes.fromhex(rows[-1]['payload'])
    turn, arq = machine(ladder='chirp', data='chirp', chirp_ack='alternate',
                        prekey_lead=2131/FS)
    turn._wall_clock = lambda: 0.
    arq.on_host_data(original)
    ledger = ByteLedger(turn, original)
    rx = P4Receiver(FS, admitted=True, chirp=True, breakin=True)
    rx.enable_hold()
    turn._begin_data_turn(rows[0]['audio_start'], State.ISS_DATA_ACK)
    calls = []
    requeue = turn.outbox.requeue
    def tracked_requeue():
        if turn.outbox.inflight:
            calls.append(turn.outbox.inflight)
        requeue()
    monkeypatch.setattr(turn.outbox, 'requeue', tracked_requeue)
    checkpoint = None
    for i, row in enumerate(rows):
        at = row['audio_start']
        if i:
            turn.decide(at-turn.lead)
            turn.silence(at-turn.lead)
        proposal = turn._next
        assert proposal is not None
        assert (proposal.value, proposal.status, proposal.payload) == (
            row['value'], row['status'], bytes.fromhex(row['payload']))
        keyed = turn.keyed(at)
        tape.register_recorded(rx, Emission(keyed.kind, Role.ISS, at,
                               (row['audio_end']-at)/FS), available_at=row['audio_end'])
        if i == len(rows)-1:
            assert turn.counter == 1 and turn.data_repeats == 6
            turn.lead = round(.110*FS)
        end = rows[i+1]['audio_start'] if i+1 < len(rows) else len(tape.audio)
        if at < 55*FS < end:
            checkpoint = snapshot(turn, rx)
        records = tape.read(rx, row['audio_end'], end)
        for reg in records:
            turn.observe(reg, now=min(end, reg.observation.stream_sample+round(.280*FS)))
        if arq._buffer_raw < len(original):
            ledger.committed = original[:31]
        ledger.check()
    assert checkpoint['turn']['state'] == State.ISS_DATA
    assert checkpoint['turn']['last']['stream_sample'] == 2580780
    assert checkpoint['turn']['counter'] == 1
    assert checkpoint['turn']['ladder'] == [13, 49, 53]
    assert checkpoint['turn']['index'] == 2 and checkpoint['role'] == Role.ISS
    assert checkpoint['inflight'] == {'hex': rows[-1]['payload']}
    assert checkpoint['queued'] == {'hex': ''}
    assert checkpoint['turn']['_next'] is None and not checkpoint['turn']['pending']
    assert checkpoint['turn']['_waiting_ack'] is None
    assert checkpoint['turn']['_rollback_until'] is None
    assert any(r.observation.word == 'U27' and not r.observation.provisional
               and abs(r.observation.stream_sample/FS-68.446) < .010
               and r.responds_to == 3120780 for r in records)
    now = len(tape.audio)
    turn.silence(now)
    proposal = turn.decide(now)
    tape.diverge(now)
    assert turn.state == State.ISS_DATA
    assert proposal is not None and proposal.kind == Kind.BODY
    assert (proposal.status, proposal.payload, proposal.state) == (0x20, b'', State.ISS_CLOSE)
    assert proposal.stream_sample == 3480780
    assert calls == [bytes.fromhex(rows[-1]['payload'])]
    assert turn._data_packet is None and turn.outbox.inflight == b''
    ledger.check()
    turn.silence(proposal.stream_sample-turn.lead)
    assert turn._next == proposal
    close = turn.keyed()
    assert close.status == 0x20 and turn.state == State.ISS_CLOSE
    turn.silence(close.stream_sample+FS)
    assert turn._next.value == 37
    ledger.check()


@pytest.mark.parametrize('kind,state,path', [
    (Kind.BODY, State.ISS_DATA, 'repeat'),
    (Kind.NORMAL, State.ISS_DATA, 'repeat'),
    (Kind.CHIRP, State.ISS_DATA, 'repeat'),
    (Kind.BODY, State.ISS_DATA_ACK, 'repeat'),
    (Kind.BODY, State.ISS_DATA_ACK, 'retry'),
])
def test_r2_data_exhaustion_branches(kind, state, path, monkeypatch):
    wall = [0.]
    turn, arq = machine(ladder='none', data='body', prekey_lead=.110)
    turn._wall_clock = lambda: wall[0]
    original = b'A'*20 + b'\x1c\x1d\x1e' + b'B'*45
    arq.on_host_data(original)
    ledger = ByteLedger(turn, original)
    turn._data(FS, state, kind)
    turn.keyed()
    budget = 8 if path == 'retry' else turn.retry_budget()[1]
    for _ in range(budget-1):
        due = turn.last.stream_sample + round((3.75 if kind == Kind.CHIRP else 1.25)*FS)
        if path == 'retry':
            turn._retry_data(due)
        else:
            turn._repeat(due-turn.lead)
        turn.keyed()
        ledger.check()
    lent = turn.outbox.inflight
    calls = []
    requeue = turn.outbox.requeue
    def tracked_requeue():
        if turn.outbox.inflight:
            calls.append(turn.outbox.inflight)
        requeue()
    monkeypatch.setattr(turn.outbox, 'requeue', tracked_requeue)
    turn._speculative[123] = (State.ISS_DATA_ACK, 0, 0)
    turn.weak_ack = 123
    turn._rollback_until = 123
    turn._blocked_redue = True
    turn.pending[123] = {'word': None}
    turn._waiting_ack = turn._waiting_until = 123
    queued = bytes(arq._outbuf)
    due = turn.next_raster(turn.last.stream_sample+FS)
    if path == 'retry':
        turn._retry_data(due)
    else:
        turn._repeat(due-turn.lead)
    close = turn._next
    assert close is not None and close.kind == Kind.BODY and close.status == 0x20
    assert close.payload == b'' and close.state == State.ISS_CLOSE
    assert turn.state == state and calls == [lent]
    assert turn._data_packet is None and not turn._speculative and not turn.pending
    assert turn.weak_ack is None and turn._waiting_ack is None
    assert turn._rollback_until is None and not turn._blocked_redue
    assert arq._buffer_raw == len(original)+3
    ledger.check()
    ctx = turn.close_context
    assert ctx.inflight == lent and ctx.queued == queued
    turn.refused('waveform not prepared')
    assert turn.state == state and turn.last.payload == lent
    assert turn._next.status == 0x20 and turn._next.kind == Kind.BODY
    assert turn.close_context is ctx and calls == [lent]
    turn.silence(turn._next.stream_sample-turn.lead)
    assert turn._next.status == 0x20
    close = turn.keyed()
    turn.silence(close.stream_sample+FS)
    assert turn._next.value == 37
    for _ in range(3):
        short = turn.keyed()
        turn.silence(short.stream_sample+FS)
        ledger.check()
    assert turn.state == State.DOWN and turn.reason == 'close silence'
    assert calls == [lent]


@pytest.mark.parametrize('branch', ['terminal-ack', 'held-v27', 'y27', 'refused', 'sample', 'wall'])
def test_r2_data_exhaustion_continuations(branch):
    wall = [0.]
    turn, arq = machine(ladder='chirp', data='chirp', prekey_lead=.110)
    turn._wall_clock = lambda: wall[0]
    original = b'unsent\x1c\x1d\x1e mail'*4
    arq.on_host_data(original)
    ledger = ByteLedger(turn, original)
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    turn.counter = 1
    turn._data(FS)
    for i in range(7):
        sent = turn.keyed()
        rx.note_emission(Emission(sent.kind, Role.ISS, sent.stream_sample, 3.292))
        q = sent.stream_sample+round(3.403*FS)
        regs = _close_audio(turn, rx, q, p4robusttx.robust_word('U27'), ready=.280)
        assert any(r.observation.word == 'U27' and not r.observation.provisional
                   and r.responds_to == sent.stream_sample for r in regs)
        turn.silence(q+round(.280*FS))
        ledger.check()
        if i < 6:
            assert turn._next.kind == Kind.CHIRP
    assert turn._next.kind == Kind.BODY and turn._next.status == 0x20
    assert turn.state == State.ISS_DATA and turn.outbox.inflight == b''
    ctx = turn.close_context
    if branch in ('refused', 'sample', 'wall'):
        if branch == 'refused':
            for _ in range(40):
                turn.refused('waveform not prepared')
                assert turn._next.kind == Kind.BODY and turn._next.status == 0x20
                turn.decide(ctx.started)
                ledger.check()
        if branch == 'wall':
            wall[0] = 30.
            turn.decide(ctx.started)
        else:
            turn.decide(ctx.end)
        assert turn.state == State.DOWN and turn.reason == 'close deadline'
        ledger.check()
        return
    if branch == 'held-v27':
        turn._word_yield = True
        turn._peer_word = 'V27'
        turn._data_refusals[turn.last.stream_sample] = 'V27'
    close = turn.keyed()
    rx.note_emission(Emission(close.kind, Role.ISS, close.stream_sample, .798))
    assert turn.state == State.ISS_CLOSE
    turn.silence(close.stream_sample+FS)
    short = turn.keyed()
    rx.note_emission(Emission(short.kind, Role.ISS, short.stream_sample, .758))
    rx.arm_close_lease(short.stream_sample, end=ctx.end)
    q = short.stream_sample+round(.935*FS)
    regs = _close_audio(turn, rx, q,
                        p4robusttx.robust_word('Y27' if branch == 'y27' else 'V27'), ready=.800)
    assert any(r.responds_to == short.stream_sample for r in regs)
    if branch == 'terminal-ack':
        assert turn.state == State.DOWN and turn.reason == 'close acknowledged'
    elif branch == 'held-v27':
        assert turn.state == State.ISS_CLOSE and turn.reason != 'close acknowledged'
        turn.decide(ctx.end)
        assert turn.state == State.DOWN and turn.reason == 'close deadline'
    else:
        _close_audio(turn, rx, q, p4robusttx.robust_word('Y27'),
                     ready=.810+(ctx.last_request-q)/FS)
        grant = turn.decide(ctx.last_request+round(.920*FS)-turn.lead)
        assert grant.purpose == 'close-grant' and grant.status == 0x40
        assert turn.state == State.ISS_CLOSE and ctx.grants == 0
        grant = turn.keyed()
        assert turn.state == State.IRS_ANSWER and ctx.grants == 1
        turn.silence(grant.stream_sample+5*FS)
        assert turn._next.status == 0x20 and ctx.resumed
        turn.decide(ctx.end)
        assert turn.state == State.DOWN and turn.reason == 'close deadline'
    ledger.check()


@pytest.mark.parametrize('state,ladder', [
    (State.ISS_DATA_ACK, 'long'), (State.ISS_GRANT, 'long'),
    (State.ISS_GRANT, 'none'), (State.ISS_LADDER, 'long'),
    (State.IRS_BREAKIN, 'none'),
])
def test_r2_data_exhaustion_preserves_other_budgets(state, ladder):
    turn, arq = machine(ladder=ladder, data='body')
    arq.on_host_data(b'pending')
    ledger = ByteLedger(turn, b'pending')
    if state == State.ISS_DATA_ACK:
        turn._data(FS, state)
    elif state == State.ISS_LADDER:
        turn.start(FS)
    else:
        turn._body(FS, 0x40, state)
    turn.keyed()
    budget = turn.retry_budget()[1]
    for _ in range(budget-1):
        turn._repeat()
        turn.keyed()
    turn._repeat()
    if state in (State.ISS_DATA_ACK, State.ISS_GRANT) and ladder != 'none':
        assert turn._next.value == 13 and turn.close_context is None
        assert turn.outbox.inflight == b''
    elif state == State.ISS_GRANT:
        assert turn.state == State.ISS_GRANT and turn._next is None
        rx = P4Receiver(FS, admitted=True)
        rx.note_emission(Emission(Kind.BODY, Role.ISS, turn.last.stream_sample, .798))
        q = turn.last.stream_sample+round(.935*FS)
        _close_audio(turn, rx, q, p4robusttx.robust_word('V27'), ready=.280)
        assert turn._next.payload == b'pending' and turn.close_context is None
    else:
        assert turn.state == State.DOWN and turn.reason == 'retry budget exhausted'
        assert turn.close_context is None
    ledger.check()


def test_r2_0514_close_head_owns_slot():
    turn, _, rx, _ = _close_prefix()
    tape = Tape('1001-0514')
    close = turn.last.stream_sample
    assert close == next(e['audio_start'] for e in tape.contract['emissions']
                         if 43*FS < e['audio_start'] < 44*FS)
    start, early = round(43.95*FS), round(44.32*FS)
    assert not tape.intervals(start, early)
    partial = rx.read_partial(tape.audio[start:early].copy(), start, deadline=early)
    for reg in partial:
        turn.observe(reg, now=early)
    turn.silence(round(44.34*FS))
    assert turn._next is None
    records = tape.read(rx, round(43.95*FS), round(44.40*FS))
    for reg in records:
        turn.observe(reg, now=round(44.40*FS))
    heads = [r for r in records if r.observation.kind == Kind.WORD
             and r.observation.word in (None, 'Y27')]
    assert heads and all(r.responds_to == close for r in heads)
    assert turn.close_context.first_request is not None
    assert turn.decide(round(44.34*FS)) is None
    assert turn._next is None
    assert rx.pending_body is not None or any(e['event'] == 'body decode'
                                               for e in rx.live_events)
    tape.diverge(round(44.406*FS))


@pytest.mark.parametrize('shape', ['bare', 'empty'])
def test_r2_close_grant_handshake(shape):
    turn, arq, rx, wall = _close_prefix()
    original = b'pending mail'
    ledger = ByteLedger(turn, original)
    q = round(44.093*FS)
    wave = (p4robusttx.robust_word('Y27') if shape == 'bare'
            else p4breakintx.changeover_packet(b'', 0x40))
    regs = _close_audio(turn, rx, q, wave)
    q = next(r.observation.stream_sample for r in regs if r.observation.word == 'Y27')
    due = q+round(.912*FS)
    proposal = turn.decide(due-turn.lead)
    assert proposal is not None and getattr(proposal, 'purpose', None) == 'close-grant'
    assert any(r.responds_to == turn.close_context.emission for r in regs)
    assert proposal.kind == Kind.BODY and proposal.status == 0x40
    assert proposal.stream_sample == due and turn.state == State.ISS_CLOSE
    turn.refused('admission')
    assert turn.state == State.ISS_CLOSE and turn.close_context.grants == 0
    assert turn._next is None
    ledger.check()
    repeat = q+round(1.25*FS)
    rx.new_cycle(repeat)
    regs = _close_audio(turn, rx, repeat, p4robusttx.robust_word('Y27'))
    assert any(r.responds_to == turn.close_context.emission for r in regs)
    repeat = next(r.observation.stream_sample for r in regs if r.observation.word == 'Y27')
    due = repeat+round(.912*FS)
    proposal = turn.decide(due-turn.lead)
    assert proposal is not None and proposal.stream_sample == due
    keyed = turn.keyed()
    rx.note_emission(Emission(keyed.kind, Role.IRS, keyed.stream_sample, .798,
                              packet_reply=True), close_grant=True)
    assert turn.state == State.IRS_ANSWER and turn.close_context.grants == 1
    assert rx.clock.next_due == keyed.stream_sample+round(.330*FS)
    assert rx.clock.cycle_s == 1.25
    assert rx.clock.expects.value == 'packet'
    ledger.check()
    assert bytes(arq._outbuf)
    wall[0] = 31
    assert turn.decide(keyed.stream_sample+FS) is None
    assert turn.state == State.DOWN and turn.reason == 'close deadline'
    ledger.check()


def _key_close_grant(*, end=75):
    turn, arq, rx, wall = _close_prefix(end=end)
    records = _close_audio(turn, rx, round(44.093*FS), p4robusttx.robust_word('Y27'))
    q = records[0].observation.stream_sample
    grant = turn.decide(q+round(.912*FS)-turn.lead)
    assert grant.purpose == 'close-grant'
    grant = turn.keyed()
    rx.note_emission(Emission(Kind.BODY, Role.IRS, grant.stream_sample, .798,
                              packet_reply=True), close_grant=True)
    return turn, arq, rx, wall, grant


@pytest.mark.parametrize('status,payload,word,state', [
    (0x40, b'peer', 'X27', State.IRS_ANSWER),
    (0x20, b'', 'V27', State.IRS_CLOSE),
])
def test_r2_reserved_close_head_enriched_to_non_grant_body(status, payload, word, state):
    turn, _, rx, _ = _close_prefix()
    records = _close_audio(turn, rx, round(44.093*FS),
                           p4breakintx.changeover_packet(payload, status))
    body = next(r for r in records if r.observation.kind == Kind.BODY)
    assert body.observation.frame.status.byte == status
    assert any(r.observation.kind == Kind.WORD and r.observation.word == 'Y27'
               for r in records)
    unread = turn.unread
    due = body.observation.stream_sample+round(.912*FS)
    proposal = turn.decide(due-turn.lead)
    assert proposal is not None and proposal.word == word and proposal.state == state
    assert turn.unread == unread and turn.close_context.grants == 0


def test_r2_close_grant_reads_v27_at_word_answer_slot():
    turn, _, rx, _, grant = _key_close_grant()
    at = grant.stream_sample+round(.935*FS)
    origin = at-round(.20*FS)
    wave = p4robusttx.robust_word('V27')
    audio = np.zeros(at-origin+round(.35*FS), np.float32)
    audio[at-origin:at-origin+len(wave)] = wave
    records = rx.read_turn_word(audio, origin)
    assert any(r.observation.word == 'V27' and r.responds_to == grant.stream_sample
               for r in records)
    for record in records:
        turn.observe(record, now=origin+len(audio))
    assert turn.state == State.IRS_ANSWER and turn._next is None


def test_r2_close_grant_held_v27_keeps_packet_raster():
    turn, _, rx, _, grant = _key_close_grant()
    packet_due = grant.stream_sample+round(.330*FS)
    held_at = grant.stream_sample+round(.935*FS)
    origin = held_at-round(.20*FS)
    wave = p4robusttx.robust_word('V27')
    audio = np.zeros(held_at-origin+round(.35*FS), np.float32)
    audio[held_at-origin:held_at-origin+len(wave)] = wave
    records = rx.read_turn_word(audio, origin)
    assert any(r.observation.word == 'V27' and r.responds_to == grant.stream_sample
               for r in records)
    for record in records:
        turn.observe(record, now=origin+len(audio))
    assert turn.state == State.IRS_ANSWER and turn._next is None
    assert rx.clock.expects == Expect.PACKET
    assert rx.clock.next_due == packet_due

    payload_at = grant.stream_sample+round(1.580*FS)
    assert payload_at == packet_due+round(1.25*FS)
    rx.new_cycle(payload_at)
    records = _close_audio(turn, rx, payload_at,
                           p4breakintx.changeover_packet(b'peer', 0x40))
    body = next(r for r in records if r.observation.kind == Kind.BODY)
    assert body.observation.frame.data.startswith(b'peer')
    assert abs(body.observation.stream_sample-payload_at) <= 1
    assert turn.close_context.progress_at == body.observation.stream_sample


def test_r2_close_grant_terminal_branches():
    held, _, held_rx, _ = _close_prefix(held=True)
    assert held._close_held_v27
    records = _close_audio(held, held_rx, held.last.stream_sample+round(.935*FS),
                           p4robusttx.robust_word('V27'))
    assert any(r.observation.word == 'V27' for r in records)
    assert held.state == State.ISS_CLOSE and held.reason != 'close acknowledged'
    ByteLedger(held, b'pending mail').check()

    turn, _, rx, _, grant = _key_close_grant()
    ledger = ByteLedger(turn, b'pending mail')
    v = _close_audio(turn, rx, grant.stream_sample+round(.330*FS),
                     p4robusttx.robust_word('V27'))
    assert any(r.observation.word == 'V27' and r.responds_to == grant.stream_sample for r in v)
    assert turn.state == State.IRS_ANSWER and turn._next is None
    ledger.check()

    turn, _, rx, _, grant = _key_close_grant()
    peer_at = grant.stream_sample+round(.330*FS)
    records = _close_audio(turn, rx, peer_at, p4breakintx.changeover_packet(b'peer', 0x40))
    body = next(r for r in records if r.observation.kind == Kind.BODY)
    due = body.observation.stream_sample+round(.912*FS)
    proposal = turn.decide(due-turn.lead)
    assert proposal.word == 'X27' and proposal.state == State.IRS_ANSWER
    assert turn.close_context.progress_at == body.observation.stream_sample
    count = len([r for r in turn.events if r.observation.kind == Kind.BODY])
    turn.observe(rx.register(body.observation), now=due-turn.lead)
    assert len([r for r in turn.events if r.observation.kind == Kind.BODY]) == count == 1
    assert turn.close_context.grants == 1
    ByteLedger(turn, b'pending mail').check()

    turn, _, rx, _, grant = _key_close_grant()
    peer_at = grant.stream_sample+round(.330*FS)
    records = _close_audio(turn, rx, peer_at, p4breakintx.changeover_packet(b'', 0x20))
    body = next(r for r in records if r.observation.kind == Kind.BODY)
    due = body.observation.stream_sample+round(.912*FS)
    ack = turn.decide(due-turn.lead)
    assert ack.word == 'V27' and ack.state == State.IRS_CLOSE
    ack = turn.keyed()
    rx.note_emission(Emission(Kind.WORD, Role.IRS, ack.stream_sample, .244))
    peer_at = body.observation.stream_sample+round(1.25*FS)
    rx.new_cycle(peer_at)
    records = _close_audio(turn, rx, peer_at, p4robusttx.robust_short(37))
    assert any(r.observation.kind == Kind.SHORT and r.observation.value == 37
               for r in records)
    assert turn._next.word == 'V27' and turn._next.state == State.IRS_CLOSE
    turn.keyed()
    turn.silence(peer_at+round(2.5*FS))
    assert turn.state == State.DOWN and turn.reason == 'peer closed'
    ByteLedger(turn, b'pending mail').check()

    turn, _, rx, _, grant = _key_close_grant()
    peer_at = grant.stream_sample+round(.330*FS)
    records = _close_audio(turn, rx, peer_at, p4breakintx.changeover_packet(b'', 0x40))
    body = next(r for r in records if r.observation.kind == Kind.BODY)
    due = body.observation.stream_sample+round(.912*FS)
    resumed = turn.decide(due-turn.lead)
    assert resumed.status == 0x20 and resumed.stream_sample == due
    assert turn.close_context.resumed and turn.close_context.grants == 1
    resumed = turn.keyed()
    rx.note_emission(Emission(Kind.BODY, Role.ISS, resumed.stream_sample, .798))
    turn.silence(resumed.stream_sample+round(1.25*FS)-turn.lead)
    assert turn._next.value == 37
    short = turn.keyed()
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, short.stream_sample, .758))
    rx.arm_close_lease(short.stream_sample, end=turn.close_context.end)
    records = _close_audio(turn, rx, short.stream_sample+round(.935*FS),
                           p4robusttx.robust_word('V27'))
    assert any(r.observation.word == 'V27' for r in records)
    assert turn.state == State.DOWN and turn.reason == 'close acknowledged'
    ByteLedger(turn, b'pending mail').check()

    turn, _, rx, _, grant = _key_close_grant()
    repeated = grant.stream_sample+round((.330+1.25)*FS)
    rx.new_cycle(repeated)
    records = _close_audio(turn, rx, repeated, p4robusttx.robust_word('Y27'))
    q = next(r.observation.stream_sample for r in records if r.observation.word == 'Y27')
    assert turn.decide(q+round(.912*FS)-turn.lead) is None
    assert turn.close_context.grants == 1
    turn.silence(grant.stream_sample+round(5*FS))
    assert turn.close_context.resumed and turn._next.status == 0x20
    assert turn.close_context.grants == 1
    turn.decide(turn.close_context.started+round(30*FS))
    assert turn.state == State.DOWN and turn.reason == 'close deadline'
    ByteLedger(turn, b'pending mail').check()

    turn, _, _, _, _ = _key_close_grant(end=50)
    turn.decide(50*FS)
    assert turn.state == State.DOWN and turn.reason == 'close deadline'

    turn, _, _, wall = _close_prefix()
    for i in range(30):
        assert turn._next is None or turn._next.state == State.ISS_CLOSE
        if turn._next is None:
            turn.silence(round((44.4+i*1.25)*FS))
        if turn._next is not None:
            turn.refused('admission')
        if turn.state == State.DOWN:
            break
        turn.decide(round((44.4+i*1.25)*FS))
    assert turn.state == State.DOWN and turn.reason == 'close deadline'
    ByteLedger(turn, b'pending mail').check()
    assert wall[0] == 0


def test_r2_close_grant_lease_rejects_invalid_evidence():
    for variant in ('off-raster', 'gap', 'tail-gap', 'expiry'):
        turn, _, rx, _ = _close_prefix()
        q = round(44.093*FS)
        if variant == 'off-raster':
            q += round(.10*FS)
        if variant == 'gap':
            rx.note_gap(q+round(.10*FS), q+round(.20*FS))
        if variant == 'tail-gap':
            rx.note_gap(q+round(.30*FS), q+round(.40*FS))
        if variant == 'expiry':
            q += round(3.75*FS)
        records = _close_audio(turn, rx, q, p4robusttx.robust_word('Y27'))
        assert not any(r.responds_to == turn.close_context.emission for r in records)
        assert turn.decide(q+round(.912*FS)-turn.lead) is None
        assert turn.close_context.grants == 0


@pytest.mark.parametrize('delay', ['late', 'unresolved'])
def test_r2_close_grant_handshake_deferred(delay):
    turn, _, rx, _ = _close_prefix()
    q = round(44.093*FS)
    if delay == 'late':
        records = _close_audio(turn, rx, q, p4robusttx.robust_word('Y27'))
        q = records[0].observation.stream_sample
        late = q+round(.912*FS)-turn.lead+1
        assert turn.decide(late) is None
    else:
        wave = p4breakintx.changeover_packet(b'', 0x40).copy()
        wave[round(.45*FS):round(.65*FS)] = 0
        records = _close_audio(turn, rx, q, wave)
        q = records[0].observation.stream_sample
        assert not any(r.observation.kind == Kind.BODY for r in records)
        assert turn.decide(q+round(.912*FS)-turn.lead) is None
    turn.silence(q+round(1.25*FS)-turn.lead)
    assert turn._next is None
    repeat = q+round(1.25*FS)
    rx.new_cycle(repeat)
    records = _close_audio(turn, rx, repeat, p4robusttx.robust_word('Y27'))
    repeat = next(r.observation.stream_sample for r in records if r.observation.word == 'Y27')
    assert any(r.responds_to == turn.close_context.emission for r in records)
    due = repeat+round(.912*FS)
    assert turn.decide(due-turn.lead).stream_sample == due


def _phase_frame(kind, at, *, outcome=Registered.NEW, provisional=False,
                 weak=False, unexpected=False, replaced=None):
    frame = p4frame.decode_field(
        p4frame.encode_field(b'hello'.ljust(31, b'\x1e'), p4frame.Status.from_byte(0)),
        data_bytes=31,
    )
    obs = Observation(kind, FS, 0, at, at, 3.29 if kind == Kind.LONG else .81,
                      0, .9, frame=frame, provisional=provisional, weak=weak,
                      unexpected=unexpected)
    return RegisteredObservation(obs, outcome, replaced)


def _breakin_prefix():
    tape = Tape('0930-2218')
    receiver = P4Receiver(FS, admitted=True, chirp=True, breakin=True)
    receiver.enable_hold()
    turn, arq = machine(ladder='none', data='body', role='irs', prekey_lead=.110)
    first_ack = next(e for e in tape.contract['emissions']
                     if 68*FS < e['audio_start'] < 69*FS)
    turn._next = Proposal(Kind.WORD, first_ack['audio_start'], word='X27',
                          state=State.IRS_ANSWER)
    turn.keyed()
    tape.register_recorded(receiver, Emission(Kind.WORD, Role.IRS,
        first_ack['audio_start'], (first_ack['audio_end']-first_ack['audio_start'])/FS),
        available_at=first_ack['audio_end'])
    for reg in tape.read(receiver, round(68.3*FS), round(72.2*FS)):
        turn.observe(reg, now=round(72.2*FS))
    turn.decide(round(72.2*FS))
    receiver.new_cycle(round(71.9*FS))
    seen = tape.read(receiver, round(71.9*FS), round(75.55*FS))
    full = [reg for reg in seen if reg.observation.kind == Kind.LONG
            and reg.observation.frame is not None]
    assert len(full) == 1
    peer = full[0].observation.stream_sample
    assert abs(peer/FS - 72.252) < .010
    for reg in seen:
        turn.observe(reg, now=round(75.5*FS))
    actual_ack = next(e for e in tape.contract['emissions']
                      if 75*FS < e['audio_start'] < 76*FS)
    ack = turn.decide(actual_ack['audio_start'] - turn.lead)
    assert ack.word == 'X27'
    assert abs(ack.stream_sample - actual_ack['audio_start']) <= 1
    turn.keyed(actual_ack['audio_start'])
    tape.register_recorded(receiver, Emission(Kind.WORD, Role.IRS,
        actual_ack['audio_start'], (actual_ack['audio_end']-actual_ack['audio_start'])/FS),
        available_at=actual_ack['audio_end'])
    arq.on_host_data(b'our mail')
    start, end = round(75.91*FS), round(76.81*FS)
    assert not tape.intervals(start, end)
    heads = receiver.read_partial(tape.audio[start:end].copy(), start, deadline=end)
    assert len(heads) == 1 and heads[0].observation.provisional
    head = heads[0].observation
    assert abs(head.stream_sample/FS - 76.019) < .010
    for reg in heads:
        turn.observe(reg, now=end)
    assert turn._next is None
    turn._break_in(peer + round((3.400 + 3.750) * FS))
    return turn, arq, peer, head


def test_r2_2218_first_request_on_peer_phase():
    turn, _, peer, head = _breakin_prefix()
    first = peer + round((3.400 + 3.750) * FS)
    assert turn._next.stream_sample == first
    assert turn._next.kind == Kind.BODY and turn._next.status == 0x40
    assert turn.breakins == 0
    assert turn.breakin_due(first - turn.lead) == first
    assert turn.breakin_due(first - turn.lead + 1) == first + round(3.750 * FS)
    assert onair._p4_breakin_phase_live(turn, [])
    assert onair._p4_link_due(turn, [], head.stream_sample, .920,
                              first-turn.lead, turn.lead, False) == first
    assert onair._p4_link_due(turn, [], head.stream_sample, .920,
                              first-turn.lead+1, turn.lead, False) == first+round(3.750*FS)
    costly, _, costly_peer, _ = _breakin_prefix()
    assert costly_peer == peer
    costly.reschedule_breakin(first - costly.lead + 1)
    assert costly._next.stream_sample == first + round(3.750 * FS)
    assert costly.breakins == 0
    retimed = costly.decide(first - costly.lead + 1)
    assert retimed is costly._next and costly.can_key(retimed)
    turn.keyed()
    turn.silence(first + round(3.750 * FS) - turn.lead)
    assert turn._next.stream_sample == first + round(3.750 * FS)
    assert turn.breakins == 1


def test_r2_breakin_phase_transitions():
    turn, _, peer, head = _breakin_prefix()
    first = peer + round((3.400 + 3.750) * FS)
    assert turn.peer_phase.onset == peer and turn.peer_phase.kind == Kind.LONG
    assert turn._next.stream_sample == first
    turn.refused('waveform not prepared')
    assert turn.breakins == 0
    assert turn._next.stream_sample == first + round(3.750 * FS)
    turn.observe(RegisteredObservation(head, Registered.EXPIRED, head))
    assert turn.peer_phase.onset == peer
    later = round(76.002 * FS)
    turn.observe(_phase_frame(Kind.LONG, later), now=later + round(3.25 * FS))
    assert turn.peer_phase.onset == later and turn.peer_phase.cycle == round(3.750 * FS)
    normal = round(80.000 * FS)
    turn.observe(_phase_frame(Kind.NORMAL, normal), now=normal + round(.80 * FS))
    assert turn.peer_phase.onset == normal and turn.peer_phase.cycle == round(1.250 * FS)
    assert turn.breakin_due(normal + round(.80*FS)) == normal + round(.920*FS)
    for bad in (_phase_frame(Kind.LONG, normal + FS, provisional=True),
                _phase_frame(Kind.LONG, normal + 2*FS, weak=True),
                _phase_frame(Kind.LONG, normal + 3*FS, unexpected=True),
                _phase_frame(Kind.LONG, normal + 4*FS, outcome=Registered.CONTRADICTED),
                _phase_frame(Kind.LONG, normal + 5*FS, outcome=Registered.EXPIRED)):
        turn.observe(bad, now=bad.observation.stream_sample + round(3.25*FS))
        assert turn.peer_phase.onset == normal
    short_at = normal + 6*FS
    short = Observation(Kind.SHORT, FS, 0, short_at, short_at, .758, 0, .9,
                        header='A', value=13, placed=True)
    turn.observe(RegisteredObservation(short, Registered.CONFIRMED),
                 now=short_at + round(.81*FS))
    assert turn.peer_phase.onset == short_at and turn.peer_phase.cycle == round(1.25*FS)
    long_at = short_at - round(.017*FS)
    turn.observe(_phase_frame(Kind.LONG, long_at, outcome=Registered.ENRICHED,
                              replaced=short), now=long_at + round(3.25*FS))
    assert turn.peer_phase.onset == long_at and turn.peer_phase.kind == Kind.LONG
    bounded, bounded_arq, bounded_peer, _ = _breakin_prefix()
    ledger = ByteLedger(bounded, b'our mail')
    ledger.check()
    due = bounded_peer + round((3.400 + 3.750) * FS)
    for keyed_count in range(4):
        assert bounded._next.stream_sample == due
        bounded.keyed()
        assert bounded.breakins == keyed_count + 1
        due += round(3.750 * FS)
        bounded.silence(due - bounded.lead)
        ledger.check()
    assert bounded.state == State.DOWN
    assert bounded.reason == 'break-in unanswered'
    assert bytes(bounded_arq._outbuf) == b'our mail'


@pytest.mark.parametrize('arm', ['1001-0344', '1001-1022', '1001-1036', '0930-0117', '0929-2253', '0930-2157'])
def test_r2_default_recorded_receiver_trace(arm):
    from hfmodem.tests.shrike.r2_replay import recorded_receiver_trace
    expected = json.loads((FIXTURES/'default-receiver-traces.json').read_text())
    assert recorded_receiver_trace(arm) == expected[arm]
    assert expected[arm]


@pytest.mark.parametrize('limit', ['sample', 'wall', 'refusals'])
def test_r2_terminal_limits_requeue_without_ack_credit(limit):
    turn, arq = machine()
    arq.on_host_data(b'bytes\x1e')
    ledger = ByteLedger(turn, b'bytes\x1e')
    turn.outbox.transfer()
    turn._short(FS, 0)
    clocks = Clocks()
    rx = P4Receiver(FS, admitted=True)
    tx = KeyOutcomes(ReplayStream(5*FS, clocks), rx, turn, ReactivePeer('ACK'), attempts=2)
    if limit == 'refusals':
        tx.transmit(keyed=False)
        ledger.check()
        tx.transmit(keyed=False)
    else:
        setattr(clocks, limit, 60*FS)
    with pytest.raises(BoundaryError, match='terminal'):
        tx.transmit(keyed=False)
    assert turn.state.value == 'DOWN'
    assert not turn.outbox.inflight
    assert ledger.committed == b''
    ledger.check()
    assert rx.emissions == [] and tx.peer.triggers == []


def test_r2_pre_journal_capture_boundary():
    tape = Tape('0926-1342')
    assert not tape.contract['journal_available']
    rx = P4Receiver(FS, admitted=True)
    with pytest.raises(BoundaryError, match='journal unavailable'):
        tape.read(rx, 0, FS)
    a, b = tape.contract['trusted_rx_intervals'][0]
    tape.read(rx, a, b)


def test_r2_census_raw_log_inputs_are_frozen():
    import re
    c = json.loads((FIXTURES/'census.json').read_text())
    tape = Tape('1001-1022')
    pattern = re.compile(r'word ([A-Z]27) confirmed at ([0-9.]+)')
    for row in c['logs']:
        path = tape.input_path(row['path'])
        assert digest(path) == row['sha256']
        actual = sorted(set((float(t), w) for w, t in pattern.findall(path.read_text(errors='replace'))))
        assert canonical(actual) == row['words']
    hold = re.compile(r'\[p1 stage\] slot (\S+) p4-hold=([0-9.]+)ms')
    overrun = re.compile(r'REGRID GAVE UP|IS GONE|GIVEN UP BEFORE THE RENDER|BEHIND THE GRID')
    tx = re.compile(r'^\s+TX\[\d+\]')
    linked, expensive = 0, 0
    for row in c['costs']:
        path = tape.input_path(row['path'])
        assert digest(path) == row['sha256']
        lines = path.read_text(errors='replace').splitlines()
        actual = [dict(line=i, cost_ms=float(m[2]), overrun_next3=any(overrun.search(s) for s in lines[i+1:i+4]))
                  for i, s in enumerate(lines) if (m := hold.search(s))]
        assert actual == row['reads']
        assert [i for i, s in enumerate(lines) if overrun.search(s)] == row['overrun_lines']
        last = None
        for i, s in enumerate(lines):
            if (m := hold.search(s)):
                last = i, float(m[2])
                continue
            if tx.match(s):
                last = None
            if overrun.search(s) and last and i-last[0] <= 3:
                linked += 1
                expensive += last[1] > 150
    assert (linked, expensive) == (182, 149)


@pytest.mark.parametrize('arm', ['1001-0215', '1001-0321'])
def test_r2_default_p3_first_stint_control_trace(arm, tmp_path):
    from hfmodem.tests.shrike.r2_replay import p3_control_trace
    expected = json.loads((FIXTURES/'default-p3-traces.json').read_text())
    assert p3_control_trace(arm, tmp_path) == expected[arm]
    assert expected[arm][-1]['packets']


@pytest.mark.parametrize('arm', ['1001-1022', '0930-2157'])
def test_r2_default_p1_control_trace(arm):
    from hfmodem.tests.shrike.r2_replay import p1_control_trace
    expected = json.loads((FIXTURES/'default-p1-traces.json').read_text())
    assert p1_control_trace(arm) == expected[arm]
    assert all(hit is not None for hit in expected[arm])
    if arm == '1001-1022':
        assert [hit[2] for hit in expected[arm]] == [1, 0, 1, 1]


@pytest.mark.parametrize('n', [15, 16, 17])
def test_r2_p3_offset_inputs_are_exact_and_decodable(n):
    from hfmodem.shrike import p3acquire
    from hfmodem.tests.shrike.r2_replay import held_window
    audio, origin = held_window('1001-0215', n)
    got = p3acquire.changeover(audio, offsets=p3acquire.CONTROL_OFFSETS_HZ)
    assert got is not None
    assert got.event.packet == (1, 0x20, b'RMS', True)
    assert -10 < got.offset_hz < -9
    assert origin == manifest()['fixtures']['1001-0215']['held_windows'][str(n)]['origin']


def test_r2_default_trace_artifacts_are_frozen():
    for name, expected in manifest()['default_trace_hashes'].items():
        assert digest(FIXTURES/name) == expected


def test_r2_mutated_physical_fixture_is_rejected(tmp_path):
    tape = Tape('1001-0344')
    for name in tape.contract['files']:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('stream.json'):
            target.write_bytes(tape.input_path(name).read_bytes() + b' ')
        else:
            target.symlink_to(tape.input_path(name))
    with pytest.raises(BoundaryError, match='mutated fixture'):
        Tape('1001-0344', root=tmp_path)


@pytest.mark.parametrize('sample_limit, wall_limit, wall, due', [
    (60*FS, 120*FS, 0, 61*FS),
    (120*FS, 60*FS, 0, 61*FS),
    (120*FS, 60*FS, FS, 59*FS),
    (60*FS, 120*FS, 0, 60*FS),
])
def test_r2_scheduled_emission_terminal_limits(sample_limit, wall_limit, wall, due):
    turn, arq = machine()
    arq.on_host_data(b'pending\x1e')
    ledger = ByteLedger(turn, b'pending\x1e')
    turn.outbox.transfer()
    turn._short(due, 0)
    clocks = Clocks(wall=wall)
    rx = P4Receiver(FS, admitted=True)
    peer = ReactivePeer('ACK')
    tx = KeyOutcomes(ReplayStream(65*FS, clocks), rx, turn, peer,
                     sample_limit=sample_limit, wall_limit=wall_limit)
    with pytest.raises(BoundaryError, match='terminal.*scheduled emission'):
        tx.transmit()
    assert clocks.sample == 0 and clocks.wall == wall
    assert turn.state.value == 'DOWN'
    assert not turn.outbox.inflight
    assert rx.emissions == [] and peer.triggers == [] and tx.trace == []
    assert ledger.committed == b''
    ledger.check()


def test_r2_unavailable_recording_skips(tmp_path):
    with pytest.raises(pytest.skip.Exception, match='unavailable recording input'):
        Tape('1001-0344', root=tmp_path)
    row = deepcopy(manifest()['fixtures']['1001-0344'])
    row['own_tx_masks'] = []
    with pytest.raises(BoundaryError, match='contract differs'):
        Tape('1001-0344', root=tmp_path, contract=row)


def test_r2_unavailable_raw_log_skips(tmp_path):
    with pytest.raises(pytest.skip.Exception, match='unavailable recording input'):
        require_input(tmp_path/'missing.log')


def _word_request_prefix(ladder='none'):
    tape = Tape('0930-2223')
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.enable_hold()
    turn, arq = machine(ladder=ladder, data='body', role='irs',
                        breakin='y27', prekey_lead=.047, normal_ack='parity')
    ack = next(e for e in tape.contract['emissions'] if 24*FS < e['audio_start'] < 25*FS)
    turn._next = Proposal(Kind.WORD, ack['audio_start'], word=ack['word'],
                          state=State.IRS_ANSWER)
    turn.keyed()
    tape.register_recorded(receiver, Emission(Kind.WORD, Role.IRS,
        ack['audio_start'], (ack['audio_end']-ack['audio_start'])/FS),
        available_at=ack['audio_end'])
    arq.on_host_data(b'our mail')
    end = round(25.59*FS)
    records = receiver.read_partial(tape.audio[ack['audio_end']:end].copy(),
                                    ack['audio_end'], deadline=end)
    records += tape.read(receiver, ack['audio_end'], end)
    assert any(r.observation.kind == Kind.NORMAL and r.observation.frame is not None
               for r in records)
    for reg in records:
        turn.observe(reg, now=end)
    request = next(e for e in tape.contract['emissions'] if 25*FS < e['audio_start'] < 26*FS)
    proposal = turn.decide(request['audio_start']-turn.lead)
    assert proposal.kind == Kind.WORD and proposal.word == 'Y27'
    assert abs(proposal.stream_sample-request['audio_start']) <= 1
    turn.keyed(request['audio_start'])
    turn.lead = round(.110*FS)
    assert any(r['event'] == 'Keyed' and r.get('state') == 'IRS_BREAKIN'
               and r['at'] == request['audio_start'] for r in tape.contract['control_trace'])
    receiver.note_emission(Emission(Kind.WORD, Role.ISS, request['audio_start'],
                                    (request['audio_end']-request['audio_start'])/FS))
    assert turn.breakins == 1 and turn._requesting()
    return tape, receiver, turn, arq, request


@pytest.mark.parametrize('ladder,kind,value', [('none', Kind.BODY, None),
                                              ('chirp', Kind.SHORT, 13)])
def test_r2_2223_reserves_v27(monkeypatch, ladder, kind, value):
    tape, receiver, turn, arq, request = _word_request_prefix(ladder)
    ledger = ByteLedger(turn, b'our mail')
    divergence = next(e['audio_start'] for e in tape.contract['emissions']
                      if 26*FS < e['audio_start'] < 27*FS)
    tape.diverge(divergence)
    audio = np.zeros(30*FS, np.float32)
    start = request['audio_end']
    assert not tape.intervals(start, divergence)
    audio[start:divergence] = tape.audio[start:divergence]
    stream = Stream(audio)
    stream.pos = start
    tx = _run_word_request(monkeypatch, stream, receiver, turn, arq, request, ladder)
    keyed = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert keyed and keyed[0]['kind'] == kind and keyed[0]['value'] == value
    assert keyed[0]['at'] == request['audio_start'] + round(2*1.25*FS)
    assert not any(e['word'] == 'Y27' for e in keyed)
    confirmed = [r for r in tx.p4_probe.heard if r['word'] == 'V27'
                 and not r['provisional']]
    assert len(confirmed) == 1 and confirmed[0]['responds_to'] == request['audio_start']
    assert confirmed[0]['answered'] is None
    assert abs(confirmed[0]['onset_sample']/FS-26.595) < .010
    assert not any(e['state'] == State.IRS_BREAKIN for e in keyed)
    ledger.check()


def _run_word_request(monkeypatch, stream, receiver, turn, arq, request, ladder='none',
                      refuse=False):
    from hfmodem.shrike import p4turn

    tx = Tx(stream)
    if refuse:
        def refused_key(*a):
            tx.refused = True
        tx._tx = refused_key
    tx.sessrx.p4 = receiver
    tx.p4_probe.peer_breakin = True
    tx.p4_probe.emissions = [request]
    tx.p4_link_options.p4_ladder = ladder
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_breakin = 'y27'
    tx.p4_link_options.p4_link_timeout = 2.8
    tx.p4_link_host.arq = arq
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: round(.110*FS))
    onair._run_p4_link(tx, stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    return tx


@pytest.mark.parametrize('case', ['weak', 'unrelated', 'expired', 'unexpected', 'late'])
def test_r2_request_head_resolution(case):
    tape, receiver, turn, arq, request = _word_request_prefix()
    ledger = ByteLedger(turn, b'our mail')
    start, early, end = request['audio_end'], round(26.81*FS), round(26.86*FS)
    assert not tape.intervals(start, end)
    heads = receiver.read_partial(tape.audio[start:early].copy(), start, deadline=early)
    assert len(heads) == 1 and heads[0].observation.provisional
    head = heads[0].observation
    if case == 'weak':
        head = replace(head, weak=True)
    turn.observe(replace(heads[0], observation=head), now=early)
    receiver.observations = [head]
    assert turn.breakins == 1 and turn._next is None
    full = receiver.read_window(tape.audio[start:end].copy(), start, whole=True)
    confirmed = next(r for r in full if r.outcome == Registered.CONFIRMED)
    assert confirmed.responds_to == request['audio_start'] and confirmed.answered is None
    if case == 'weak':
        confirmed = replace(confirmed, observation=replace(confirmed.observation, weak=True))
    elif case == 'unrelated':
        confirmed = replace(confirmed, responds_to=None)
    elif case == 'unexpected':
        confirmed = replace(confirmed, observation=replace(confirmed.observation, unexpected=True))
    elif case == 'expired':
        confirmed = replace(confirmed, outcome=Registered.EXPIRED)
    now = round(28.10*FS) if case == 'late' else end
    turn.observe(confirmed, now=now)
    assert turn.breakins == 1
    if case == 'late':
        assert turn._next.kind == Kind.BODY
        assert turn._next.stream_sample >= now + turn.lead
    else:
        assert turn._requesting()
        assert turn._next is None or turn._next.kind == Kind.WORD
        turn.silence(turn.next_raster(now)-turn.lead)
        assert turn._next.word == 'Y27' and turn.breakins == 1
        turn.keyed()
        assert turn.breakins == 2
        for count in (3, 4):
            turn.silence(turn.next_raster(turn.last.stream_sample)-turn.lead)
            assert turn.breakins == count-1
            turn.keyed()
            assert turn.breakins == count
        turn.silence(turn.next_raster(turn.last.stream_sample)-turn.lead)
        assert turn.state == State.DOWN and turn.reason == 'break-in unanswered'
    ledger.check()


@pytest.mark.parametrize('branch', ['body', 'failed', 'late', 'refused'])
def test_r2_request_head_resolution_driver(monkeypatch, branch):
    _, receiver, turn, arq, request = _word_request_prefix()
    ledger = ByteLedger(turn, b'our mail')
    stream = ReplayStream(30*FS, Clocks())
    stream.pos = request['audio_end']
    peer = ReactivePeer('body' if branch == 'body' else 'ACK',
                        payload=b'peer mail', word='V27')
    [(onset, wave)] = peer.keyed(turn.last)
    if branch == 'failed':
        wave[round(.120*FS):] = 0
    stream.append_peer([(onset, wave)])
    deliveries = []
    turn.adapter.host = SimpleNamespace(deliver=deliveries.append, supervisor=lambda *a: None)
    if branch == 'late':
        read = receiver.read_window
        def costly_read(*a, **kw):
            records = read(*a, **kw)
            stream.clocks.wall += round(.200*FS)
            return records
        monkeypatch.setattr(receiver, 'read_window', costly_read)
        monkeypatch.setattr(stream, 'sample_now', lambda: stream.pos + stream.clocks.wall)
    tx = _run_word_request(monkeypatch, stream, receiver, turn, arq, request,
                           refuse=branch == 'refused')
    keyed = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    if branch == 'refused':
        assert not keyed and turn.breakins == 1
        assert any(e['event'] == 'Refused' and e['kind'] == Kind.BODY
                   for e in tx.p4_probe.link_events)
        ledger.check()
        return
    assert keyed
    if branch == 'body':
        assert keyed[0]['kind'] == Kind.WORD and keyed[0]['word'] == 'V27'
        assert deliveries == [b'peer mail']
        assert sum(r['kind'] == Kind.BODY for r in tx.p4_probe.heard) == 1
        assert not any(e['word'] == 'Y27' for e in keyed)
    elif branch == 'failed':
        assert keyed[0]['word'] == 'Y27' and turn.breakins == 2
        assert sum(r['outcome'] == Registered.EXPIRED for r in tx.p4_probe.heard) == 1
    else:
        assert keyed[0]['kind'] == Kind.BODY and keyed[0]['at'] >= round(28.165*FS)
        assert not any(e['word'] == 'Y27' for e in keyed)
    assert keyed[0]['at'] > request['audio_start'] + round(1.25*FS)
    ledger.check()


def test_r2_request_head_resolution_horizons():
    turn, _ = machine(ladder='none', data='body', role='irs', breakin='y27')
    turn._break_in(0)
    turn.keyed()
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.note_emission(Emission(Kind.WORD, Role.ISS, 0, .242))
    obs = Observation(Kind.WORD, FS, 0, round(.935*FS), round(.935*FS), .2, 0, .9,
                      word=None, provisional=True)
    receiver.observations = [obs]
    horizon = obs.stream_sample + round(receiver._word_horizon(obs)*FS)
    assert onair._p4_word_confirmation_end(receiver, turn, obs.stream_sample,
                                          horizon-1, False) == horizon
    turn.state = State.ISS_HOLD
    old_horizon = obs.stream_sample+round(.245*FS)
    assert onair._p4_word_confirmation_end(receiver, turn, obs.stream_sample,
                                          old_horizon-1, False) is None
    assert onair._p4_word_confirmation_end(receiver, turn, obs.stream_sample,
                                          old_horizon, False) == old_horizon
    turn.state = State.IRS_BREAKIN
    receiver.emissions = [Emission(Kind.SHORT, Role.ISS, 0, .758)]
    assert receiver._word_horizon(obs) == .300
    assert onair._p4_word_confirmation_end(receiver, turn, obs.stream_sample,
                                          obs.stream_sample+round(.300*FS), False) == obs.stream_sample+round(.300*FS)


def _r2_opening(value=38, *, policy='repeat'):
    turn, arq = machine(ladder='long', data='hold', train_policy=policy)
    turn.seed_opening([dict(audio_start=0, value=value, header='A')], [],
                      values=(value,))
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, 0, .75, train=True))
    return turn, arq, rx


def _r2_word(rx, turn, name, at, *, provisional=False, weak=False, extent=.24):
    obs = Observation(Kind.WORD, FS, 0, at, at, .24, 0, .9, word=name,
                      score=.99, provisional=provisional, weak=weak, extent_s=extent)
    reg = rx.register(obs)
    turn.observe(reg, now=at + round(.28*FS))
    return reg


def test_r2_v38_opening_wait():
    turn, arq, rx = _r2_opening()
    arq.on_host_data(b'queued mail')
    at = round(.938*FS)
    reg = _r2_word(rx, turn, 'X27', at)
    assert reg.responds_to == 0
    assert turn.state == State.ISS_LADDER and turn._opening_wait
    assert turn._opening and turn._next is None
    turn.silence(6*FS)
    assert turn.state == State.ISS_LADDER and turn._next is None
    turn.decide(round(12.5*FS))
    assert turn.state == State.DOWN and 'opening' in turn.reason
    assert arq._buffer_raw == len(arq._outbuf)
    other, _, other_rx = _r2_opening()
    unnamed = _r2_word(other_rx, other, None, at)
    assert unnamed.responds_to is None
    assert not other._opening_wait
    pending, _, pending_rx = _r2_opening()
    head = _r2_word(pending_rx, pending, None, at, provisional=True)
    assert head.responds_to == 0
    confirmed = _r2_word(pending_rx, pending, None, at)
    assert confirmed.responds_to == 0 and not pending._opening_wait


def test_r2_bare_y27_resolves_opening_before_mail_turn():
    turn, arq = machine(ladder='none', data='body')
    arq.on_host_data(b'queued mail')
    turn.seed_opening([dict(audio_start=0, value=38, header='A')], [], values=(38,))
    rx = P4Receiver(FS, admitted=True)
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, 0, .75, train=True))
    _r2_word(rx, turn, 'X27', round(.935*FS))
    assert turn._opening_wait
    y_at = round(2.185*FS)
    _r2_word(rx, turn, 'Y27', y_at, extent=.245)
    grant_due = y_at + round(turn.t[Kind.BODY]*FS)
    grant = turn.decide(grant_due-turn.lead)
    assert grant is not None and grant.state == State.ISS_GRANT
    assert not turn._opening_wait
    turn.keyed()
    turn.decide(round(12.5*FS))
    assert turn.state != State.DOWN


@pytest.mark.parametrize('arm,at_s', [('1001-0611', 11.304729),
                                     ('0930-0146', 35.059938)])
def test_r2_v38_recorded_x27(arm, at_s):
    tape = Tape(arm)
    rows = tape.contract['emissions']
    row = min(rows, key=lambda e: abs(e['audio_start']/FS-at_s))
    following = next(e for e in rows if e['audio_start'] > row['audio_start'])
    turn, _ = machine(ladder='long', data='hold')
    turn.seed_opening([e for e in rows if e.get('value') == 38
                       and e['audio_start'] <= row['audio_start']], [], values=(38,))
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.expected_words = turn.expected_words()
    em = Emission(Kind.SHORT, Role.ISS, row['audio_start'],
                  (row['audio_end']-row['audio_start'])/FS, train=True,
                  train_words=frozenset(('X27', 'Z27', 'Y27')))
    tape.register_recorded(rx, em, available_at=row['audio_end'])
    regs = tape.read(rx, row['audio_end'], following['audio_start'])
    ack = next(r for r in regs if r.observation.word == 'X27')
    assert ack.responds_to == row['audio_start'] and ack.response_slot == 0
    turn.observe(ack, now=ack.observation.stream_sample+round(.275*FS))
    assert turn._opening_wait and turn.state == State.ISS_LADDER
    assert turn._next is None


def test_r2_payload_ladder_completion_is_not_opening_wait():
    turn, _ = machine(ladder='long', data='hold')
    rx = P4Receiver(FS, admitted=True)
    turn._short(0, 0)
    turn.keyed()
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, 0, .75))
    _r2_word(rx, turn, 'X27', round(.935*FS))
    second = turn.keyed()
    assert second.value == 60
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, second.stream_sample, .75))
    _r2_word(rx, turn, 'U27', second.stream_sample+round(.935*FS))
    assert turn._next is not None and turn._next.kind == Kind.LONG
    assert not turn._opening_wait


def test_r2_train_receiver_lease():
    turn, _, rx = _r2_opening(7, policy='passive')
    first = round(.935*FS)
    provisional = _r2_word(rx, turn, 'W27', first, provisional=True)
    assert provisional.responds_to == 0
    confirmed = _r2_word(rx, turn, 'W27', first)
    assert confirmed.responds_to == 0 and confirmed.response_slot == 0
    for slot in range(1, 6):
        at = first + slot*round(1.25*FS)
        reg = _r2_word(rx, turn, 'W27', at)
        assert reg.responds_to == 0 and reg.response_slot == slot
    ack = _r2_word(rx, turn, 'X27', first+6*round(1.25*FS))
    assert ack.responds_to == 0 and ack.response_slot == 6
    assert all(abs(t-.935) < .01 for t in rx.turnarounds['short'])
    off = _r2_word(rx, turn, 'W27', first+round(7.1*1.25*FS))
    assert off.responds_to is None


@pytest.mark.parametrize('arm,start_s', [('1001-1022', 20.054667),
                                       ('1001-1036', 18.756563)])
def test_r2_recorded_train_conditional_first_key(arm, start_s):
    tape = Tape(arm)
    rows = tape.contract['emissions']
    index = min(range(len(rows)), key=lambda i: abs(rows[i]['audio_start']/FS-start_s))
    rows = rows[index:index+8]
    first = rows[0]
    body_train = arm == '1001-1036'
    if body_train:
        turn, arq = machine(ladder='none', data='body', train_policy='passive')
        arq.on_host_data(bytes.fromhex(first['payload']))
        turn._begin_data_turn(first['audio_start'])
        assert turn.keyed().payload == bytes.fromhex(first['payload'])
    else:
        turn, _ = machine(ladder='long', data='hold', train_policy='passive')
        turn.seed_opening([first], [], values=(7, 38))
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.expected_words = turn.expected_words()
    tape.register_recorded(rx, Emission(Kind.BODY if body_train else Kind.SHORT,
                           Role.ISS, first['audio_start'],
                           (first['audio_end']-first['audio_start'])/FS, train=True),
                           available_at=first['audio_end'])
    seen = []
    for slot in range(7):
        row = rows[slot]
        end = rows[slot+1]['audio_start']
        regs = tape.read(rx, row['audio_end'], end)
        named = [r for r in regs if r.observation.word in ('W27', 'X27')]
        assert len(named) == 1
        reg = named[0]
        assert reg.responds_to == first['audio_start']
        assert reg.response_slot == slot
        seen.append(reg.observation.word)
        turn.observe(reg, now=reg.observation.stream_sample)
        if slot < 6:
            assert turn._next is None and turn._train_active
            rx.note_gap(end, rows[slot+1]['audio_end']+tape.contract['settle_samples'])
            rx.new_cycle()
    assert seen == ['W27']*6 + ['X27']
    assert turn._next is not None
    assert turn._next.kind == (Kind.NORMAL if body_train else Kind.SHORT)
    assert turn._next.status == (1 if body_train else 0)
    assert turn._next.value == (None if body_train else 38)
    assert not turn._train_active


def test_r2_passive_elapsed_budget():
    turn, _, rx = _r2_opening(7, policy='passive')
    first = round(.935*FS)
    _r2_word(rx, turn, 'W27', first)
    for slot in range(1, 8):
        now = slot*round(1.25*FS)-turn.lead
        turn.silence(now)
        assert turn.decide(now) is None
    k8 = 8*round(1.25*FS)
    turn.silence(k8-turn.lead)
    fallback = turn.decide(k8-turn.lead)
    assert fallback is not None and fallback.stream_sample == k8
    assert fallback.value == 7 and fallback.header == 'B'
    turn.keyed()
    turn.decide(9*round(1.25*FS))
    assert turn.state == State.DOWN and turn.reason == 'passive train expired'


def test_r2_train_lease_gaps_generation_and_enrichment():
    turn, _, rx = _r2_opening(7, policy='passive')
    first = round(.935*FS)
    initial = _r2_word(rx, turn, 'W27', first)
    assert initial.response_slot == 0
    gap_at = first + round(1.25*FS)
    rx.note_gap(gap_at+1, gap_at+round(.1*FS))
    assert _r2_word(rx, turn, 'W27', gap_at).responds_to is None
    intact = first + 2*round(1.25*FS)
    reg = _r2_word(rx, turn, 'W27', intact, extent=None)
    assert reg.responds_to == 0 and reg.response_slot == 2
    enriched = rx.register(replace(reg.observation, extent_s=.25))
    assert enriched.outcome == Registered.EXTENDED
    assert enriched.responds_to == 0 and enriched.response_slot == 2
    rx.note_emission(Emission(Kind.NORMAL, Role.ISS, 4*FS, .75))
    stale = rx.register(replace(reg.observation, extent_s=.26))
    assert stale.responds_to is None
    assert stale.response_generation is None


def test_r2_revoked_registrations_cannot_advance_ladder_or_commit_mail():
    turn, _, rx = _r2_opening(38)
    ack_at = round(.935*FS)
    ack = rx.register(Observation(Kind.WORD, FS, 0, ack_at, ack_at, .24, 0,
                                  .9, word='X27', score=.99, extent_s=.24))
    assert ack.response_generation == rx.train_lease.generation
    rx.revoke_train()
    turn.observe(ack, now=ack_at+round(.28*FS))
    assert not turn._opening_wait

    # Reusing the receiver and sample origin must not resurrect the old lease.
    rx.new_call()
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, 0, .75, train=True))
    assert ack.response_generation != rx.train_lease.generation
    turn.observe(ack, now=ack_at+round(.28*FS))
    assert not turn._opening_wait

    data, arq = machine(ladder='none', data='body', train_policy='passive')
    arq.on_host_data(b'pending mail')
    data._begin_data_turn(0)
    data.keyed()
    data_rx = P4Receiver(FS, admitted=True)
    data_rx.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8, train=True))
    first = round(.935*FS)
    _r2_word(data_rx, data, 'W27', first)
    ack_at = first+round(1.25*FS)
    ack = data_rx.register(Observation(Kind.WORD, FS, 0, ack_at, ack_at, .24,
                                       0, .9, word='X27', score=.99, extent_s=.24))
    assert ack.response_generation == data_rx.train_lease.generation
    data_rx.revoke_train()
    data.observe(ack, now=ack_at+round(.28*FS))
    assert data.outbox.inflight == b'pending mail'
    assert arq._buffer_raw == len(b'pending mail')


def test_r2_passive_v38_z27_and_body_resolution():
    turn, _, rx = _r2_opening(38, policy='passive')
    at = round(.935*FS)
    reg = _r2_word(rx, turn, 'Z27', at)
    assert reg.responds_to == 0
    assert turn._opening_wait and not turn._train_active
    assert turn.decide(8*round(1.25*FS)-turn.lead) is None
    peer_body = body(8*round(1.25*FS), b'greeting').observation
    _r2_word(rx, turn, 'Y27', peer_body.stream_sample)
    turn.observe(rx.register(peer_body), now=peer_body.stream_sample)
    assert not turn._opening_wait
    assert turn.pending
    assert turn.decide(peer_body.stream_sample + round(.920*FS)-turn.lead) is not None


def test_r2_passive_v38_z27_recorded():
    tape = Tape('1001-1022')
    rows = tape.contract['emissions']
    index = next(i for i, e in enumerate(rows) if e.get('value') == 38)
    rows = rows[index:index+9]
    first = rows[0]
    turn, _ = machine(ladder='long', data='hold', train_policy='passive')
    turn.seed_opening([first], [], values=(38,))
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.expected_words = turn.expected_words()
    tape.register_recorded(rx, Emission(Kind.SHORT, Role.ISS, first['audio_start'],
                           (first['audio_end']-first['audio_start'])/FS, train=True,
                           train_words=frozenset(('X27', 'Z27', 'Y27'))),
                           available_at=first['audio_end'])
    seen = []
    for slot in range(8):
        row, following = rows[slot:slot+2]
        regs = tape.read(rx, row['audio_end'], following['audio_start'])
        for reg in regs:
            if reg.observation.word in ('Z27', 'Y27') or reg.observation.kind == Kind.BODY:
                seen.append((reg.observation.word or 'BODY', reg.response_slot))
                turn.observe(reg, now=reg.observation.stream_sample)
        if slot < 7:
            assert turn._opening_wait and turn._next is None
            rx.note_gap(following['audio_start'],
                        following['audio_end']+tape.contract['settle_samples'])
            rx.new_cycle()
    assert seen == [('Z27', i) for i in range(7)] + [('Y27', 7), ('BODY', 7)]
    assert not turn._opening_wait and turn.pending
    assert rx.train_lease is None


def test_r2_recorded_greeting_body_keeps_replacement_authority():
    tape = Tape('1001-1022')
    rows = tape.contract['emissions']
    index = next(i for i, row in enumerate(rows) if row.get('value') == 38)
    rows = rows[index:index+9]
    first = rows[0]
    turn, _ = machine(ladder='long', data='hold', train_policy='passive')
    turn.seed_opening([first], [], values=(38,))
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.expected_words = turn.expected_words()
    tape.register_recorded(rx, Emission(Kind.SHORT, Role.ISS, first['audio_start'],
                           (first['audio_end']-first['audio_start'])/FS, train=True,
                           train_words=frozenset(('X27', 'Z27', 'Y27'))),
                           available_at=first['audio_end'])
    head = greeting = None
    for slot in range(8):
        row, following = rows[slot:slot+2]
        for reg in tape.read(rx, row['audio_end'], following['audio_start']):
            if reg.observation.word in ('Z27', 'Y27') or reg.observation.kind == Kind.BODY:
                if reg.observation.word == 'Y27':
                    head = reg
                elif reg.observation.kind == Kind.BODY:
                    greeting = reg
                turn.observe(reg, now=reg.observation.stream_sample)
        if slot < 7:
            rx.note_gap(following['audio_start'],
                        following['audio_end']+tape.contract['settle_samples'])
            rx.new_cycle()
    assert head is not None and greeting is not None
    assert greeting.observation.frame.data.startswith(b'RMS Trimode 1.4.4.0')
    assert greeting.response_generation == head.response_generation
    assert rx.train_lease is None
    assert not turn._current_response(head)
    assert turn._current_response(greeting)
    assert not turn._opening_wait
    assert turn.pending[greeting.observation.stream_sample]['body'] is greeting.observation


def test_r2_passive_data_distinct_keyed_refusals():
    turn, arq = machine(ladder='none', data='body', train_policy='passive')
    arq.on_host_data(b'initial mail')
    turn._begin_data_turn(0)
    first = turn.keyed()
    rx = P4Receiver(FS, admitted=True)
    rx.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8, train=True))
    at = round(.935*FS)
    _r2_word(rx, turn, 'Z27', at)
    assert turn._train_active and len(turn._data_refusals) == 1
    _r2_word(rx, turn, 'Z27', at+round(1.25*FS))
    assert len(turn._data_refusals) == 1
    k8 = 8*round(1.25*FS)
    turn.silence(k8-turn.lead)
    fallback = turn.decide(k8-turn.lead)
    assert fallback.status == first.status and fallback.payload == first.payload
    turn.keyed()
    rx.note_emission(Emission(Kind.BODY, Role.ISS, k8, .8,
                              train=True, train_root=0))
    _r2_word(rx, turn, 'Z27', k8+at)
    assert turn._next is not None and turn._next.state == State.ISS_CLOSE
    assert arq._buffer_raw == len(arq._outbuf)


def test_r2_passive_wall_jump_exhausts_without_credit():
    wall = [0.]
    turn, arq = machine(ladder='none', data='body', train_policy='passive',
                        wall_clock=lambda: wall[0])
    arq.on_host_data(b'pending')
    turn._begin_data_turn(0)
    turn.keyed()
    rx = P4Receiver(FS, admitted=True)
    rx.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8, train=True))
    _r2_word(rx, turn, 'W27', round(.935*FS))
    assert turn._train_active
    wall[0] = 12.
    turn.decide(2*FS)
    assert turn._next is not None and turn._next.state == State.ISS_CLOSE
    assert bytes(arq._outbuf) == b'pending'


@pytest.mark.parametrize('expiry', ['k9', 'wall'])
def test_r2_retained_x27_after_passive_expiry_cannot_commit(expiry):
    wall = [0.]
    turn, arq = machine(ladder='none', data='body', train_policy='passive',
                        wall_clock=lambda: wall[0])
    arq.on_host_data(b'pending mail')
    turn._begin_data_turn(0)
    turn.keyed()
    rx = P4Receiver(FS, admitted=True)
    rx.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8, train=True))
    first = round(.935*FS)
    _r2_word(rx, turn, 'W27', first)
    slot = 8 if expiry == 'k9' else 1
    ack_at = first+slot*round(1.25*FS)
    ack = rx.register(Observation(Kind.WORD, FS, 0, ack_at, ack_at, .24, 0,
                                  .9, word='X27', score=.99, extent_s=.24))
    assert ack.responds_to == 0 and ack.response_slot == slot
    if expiry == 'wall':
        wall[0] = 12.
        observed_at = ack_at+round(.28*FS)
    else:
        observed_at = 9*round(1.25*FS)
    turn.observe(ack, now=observed_at)
    assert turn.outbox.inflight == b''
    assert bytes(arq._outbuf) == b'pending mail'
    assert turn._next is not None and turn._next.state == State.ISS_CLOSE


def test_r2_complete_r8_x27_before_k9_still_commits():
    turn, arq = machine(ladder='none', data='body', train_policy='passive')
    arq.on_host_data(b'pending mail')
    turn._begin_data_turn(0)
    turn.keyed()
    rx = P4Receiver(FS, admitted=True)
    rx.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8, train=True))
    first = round(.935*FS)
    _r2_word(rx, turn, 'W27', first)
    ack_at = first+8*round(1.25*FS)
    ack = rx.register(Observation(Kind.WORD, FS, 0, ack_at, ack_at, .24, 0,
                                  .9, word='X27', score=.99, extent_s=.24))
    turn.observe(ack, now=9*round(1.25*FS)-1)
    assert arq._buffer_raw == 0
    assert turn.outbox.inflight == b''


@pytest.mark.parametrize('case', ['missed-k8', 'refused-k8', 'silent-k9', 'live-head'])
def test_r2_passive_fallback_is_one_elapsed_opportunity(case):
    turn, _, rx = _r2_opening(7, policy='passive')
    _r2_word(rx, turn, 'W27', round(.935*FS))
    cycle = round(1.25*FS)
    if case == 'live-head':
        _r2_word(rx, turn, 'W27', round(.935*FS)+7*cycle, provisional=True)
        turn.silence(8*cycle-turn.lead)
        assert turn._next is None
    elif case == 'missed-k8':
        turn.silence(8*cycle-turn.lead+1)
        assert turn._next is None
    else:
        turn.silence(8*cycle-turn.lead)
        assert turn._next is not None and turn._next.stream_sample == 8*cycle
        if case == 'refused-k8':
            turn.refused('test refusal')
            assert turn._next is None and not turn._train_fallback_keyed
        else:
            turn.keyed()
            assert turn._train_fallback_keyed
    turn.decide(9*cycle-turn.lead)
    assert turn.state == State.DOWN and turn.reason == 'passive train expired'


@pytest.mark.parametrize('word_yield,later', [(True, False), (False, True)])
def test_r2_passive_scope_excludes_word_yield_and_later_data(word_yield, later):
    turn, arq = machine(ladder='none', data='body', train_policy='passive')
    arq.on_host_data(b'pending data')
    if later:
        turn._first_data = False
        turn.counter = 1
        turn._data(0)
    else:
        turn._begin_data_turn(0, word_yield=word_yield)
    sent = turn.keyed()
    assert turn.train_root_for(sent) is None
    rx = P4Receiver(FS, admitted=True)
    rx.note_emission(Emission(Kind.BODY if not later else Kind.NORMAL,
                              Role.ISS, 0, .8))
    _r2_word(rx, turn, 'W27', round(.935*FS))
    assert not turn._train_active


@pytest.mark.parametrize('policy', ['repeat', 'passive'])
@pytest.mark.parametrize('answer', ['X27', 'Z27'])
@pytest.mark.parametrize('train_next', [True, False])
@pytest.mark.parametrize('revoke_before_key', [False, True])
def test_r2_keyed_opening_head_keeps_its_own_refinement(policy, answer, train_next, revoke_before_key):
    turn, _ = machine(ladder='long', data='hold', train_policy=policy)
    turn.seed_opening([dict(audio_start=0, value=7, header='A')], [], values=(7, 38))
    rx = P4Receiver(FS, admitted=True)
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, 0, .75, train=True))
    at = round(.935*FS)
    head = _r2_word(rx, turn, 'X27', at, provisional=True)
    if revoke_before_key:
        rx.revoke_train()
    sent = turn.keyed()
    assert sent.value == 38
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, sent.stream_sample, .75,
                             train=train_next,
                             train_words=frozenset(('X27', 'Z27', 'Y27')) if train_next else None))
    assert not turn._current_response(head)
    full = rx.register(replace(head.observation, provisional=False, word=answer))
    assert full.responds_to == 0 and full.answered == sent.stream_sample
    assert turn._current_response(full)
    turn.observe(full, now=sent.stream_sample+round(.01*FS))
    assert at not in turn._speculative
    assert turn.index == (1 if answer == 'X27' else 0)
    if answer == 'Z27':
        assert turn._next.value == 7
    if train_next:
        rx.revoke_train()
    else:
        rx.note_emission(Emission(Kind.NORMAL, Role.IRS, sent.stream_sample+FS, .75))
    assert not turn._current_response(full)


def test_r2_saved_unassociated_body_keeps_its_association():
    rx = P4Receiver(FS, admitted=True)
    decoded = body(round(.935*FS)).observation
    head = replace(decoded, kind=Kind.WORD, word='Y27', frame=None, extent_s=.245)
    saved = rx.register(head)
    assert saved.responds_to is None
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, 0, .75))
    completed = rx.register(decoded)
    assert completed.responds_to is None
