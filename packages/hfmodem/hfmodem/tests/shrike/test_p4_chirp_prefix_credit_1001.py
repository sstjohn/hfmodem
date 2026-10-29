# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""R2 source-wire transactions and pending turn integration, without audio reads."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.shrike import compress, p4frame, spec
from hfmodem.shrike.p4credit import ReceiptContext, SourceReceipt
from hfmodem.shrike.p4host import P4Host, P4StreamFailure
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import Proposal, State
from hfmodem.tests.shrike.test_p4turn import FS, machine
from hfmodem.tests.shrike.test_p4_opening_recut_0929 import packet, receive

ROWS_PATH = Path(__file__).parent / 'fixtures/p4-chirp-prefix-1001.json'
if not (ROWS_PATH).exists():
    pytest.skip('p4-chirp-prefix-1001.json is not in this tree', allow_module_level=True)
ROWS = json.loads(ROWS_PATH.read_text())
PREFIX = b'RMS Trimode 1.4.4.0 Welcome to '


class Sink:
    def __init__(self):
        self.events = []

    def deliver(self, data):
        self.events.append(('data', data))

    def supervisor(self, code, params):
        self.events.append(('block', code, params))

    @property
    def payload(self):
        return b''.join(e[1] for e in self.events if e[0] == 'data')

    @property
    def blocks(self):
        return [e[1:] for e in self.events if e[0] == 'block']


def frame(data, status=0):
    return p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)),
                                data_bytes=len(data))


def setup(prefix=PREFIX, *, provenance='opening', armed=True, terminal=True):
    sink = Sink()
    host = P4Host(sink, terminal=terminal)
    body = frame(prefix, 0x40 if provenance == 'changeover' else 0)
    receipt = SourceReceipt(1, (Kind.BODY, FS), (body.status.byte, body.data))
    host.accept(body, changeover=provenance == 'changeover',
                receipt_context=ReceiptContext(receipt, Kind.BODY))
    host.candidate(receipt, prefix, provenance)
    if armed:
        assert host.arm(receipt) == 'armed'
    return host, sink, receipt


def chirp(host, receipt, data, counter=1, *, status=None):
    return host.accept(frame(data, 0x60 | counter if status is None else status),
                       receipt_context=ReceiptContext(receipt, Kind.CHIRP))


def independent(fields):
    supervisor, sink = compress.Supervisor(), Sink()
    for field in fields:
        payload = supervisor.feed(field)
        if payload:
            sink.deliver(payload)
        for block in supervisor.blocks:
            sink.supervisor(*block)
        supervisor.blocks.clear()
    return sink


@pytest.mark.parametrize('name', ROWS)
def test_recorded_host_stream_and_per_identity_diagnostics(name):
    row = ROWS[name]
    prefix = bytes.fromhex(row['body']['data'])
    host, sink, receipt = setup(prefix)
    last = receipt.identity
    credits, cursors = [0], [0]
    for index, copy in enumerate(row['copies']):
        raw = bytes.fromhex(copy['data'])
        identity = copy['status'], raw
        output = chirp(host, receipt, raw, status=copy['status'])
        assert host.result.duplicate == (identity == last)
        if index == 0:
            assert output == b'' and host.result.held_total == 22
        elif index == 1:
            assert output is None and host.result.held_total == 22
        elif index == 2:
            assert output == (b'96 daily minu' if name == '0930-2226' else b'K4MSU Winlink')
        else:
            assert host.last_credit == 0
        credits.append(host.last_credit)
        cursors.append(host.result.cursor)
        last = identity
    assert credits[:4] == [0, 0, 0, 31]
    assert cursors[:4] == [0, 22, 22, 31]
    assert sink.payload == bytes.fromhex(row['expected'])
    assert sink.blocks == row['callbacks']


@pytest.mark.parametrize('prefix,fields', [
    (b'abc', [(1, b'abcTAIL')]),
    (b'abc', [(1, b'abc')]),
    (PREFIX, [(1, PREFIX[:22]), (2, PREFIX[22:])]),
    (b'abc', [(1, b''), (2, b'abcZ')]),
    (PREFIX, [(1, PREFIX[:22]), (2, b''), (3, PREFIX[22:]+b'Z')]),
    (b'abcde', [(1, b'a'), (2, b'b'), (3, b'c'), (0, b'd'), (1, b'eZ')]),
])
def test_short_exact_empty_and_wrapped_counters(prefix, fields):
    host, sink, receipt = setup(prefix, provenance='changeover')
    cursor = 0
    for counter, data in fields:
        before = len(sink.events)
        chirp(host, receipt, data or spec.field_fill(22), counter)
        cursor = min(len(prefix), cursor+len(data))
        assert host.result.cursor == cursor
        assert host.result.expected_counter == (counter+1) % 4
        assert host.last_credit == (len(prefix) if cursor == len(prefix) else 0)
        snapshot = host.result
        chirp(host, receipt, data or spec.field_fill(22), counter)
        assert host.result.duplicate and host.last_credit == 0
        assert host.result.cursor == snapshot.cursor
        assert host.result.held_delta == host.result.released_source == 0
        assert len(sink.events) <= before+1
    assert sink.payload == b''.join(data for _, data in fields)
    assert host.arm(receipt) == 'spent'


@pytest.mark.parametrize('first,current,counter', [
    (b'X'+PREFIX[1:22], PREFIX[22:]+b'Z', 2),
    (PREFIX[:22], PREFIX[22:30]+b'X'+b'Z', 2),
    (PREFIX[:22], PREFIX[22:]+b'Z', 3),
    (PREFIX[:22], b'', 3),
])
def test_mismatch_and_counter_gap_release_entire_wire(first, current, counter):
    host, sink, receipt = setup()
    chirp(host, receipt, first)
    chirp(host, receipt, current, counter)
    assert sink.payload == PREFIX+first+current
    assert host.last_credit == 0 and host.result.held_total == 0
    assert host.result.released_source == (22 if first == PREFIX[:22] else 0)
    assert host.arm(receipt) == 'spent'


@pytest.mark.parametrize('boundary', [22, 31])
@pytest.mark.parametrize('block', [False, True])
def test_transparency_and_real_blocks_cross_credit_boundaries(boundary, block):
    wire = (b'A'*(boundary-2)+compress.supervisor_block('B 7')+b'Z' if block else
            compress.transparent(b'A'*(boundary-1)+b'\x1e'+b'Z'*8))
    wire = wire.ljust(40, b'Z')
    prefix = wire[:31]
    host, sink, receipt = setup(prefix)
    chirp(host, receipt, wire[:22])
    chirp(host, receipt, wire[:22])
    chirp(host, receipt, wire[22:44], 2)
    expected = independent([wire])
    assert sink.payload == expected.payload
    assert sink.blocks == ([('B', b'7')] if block else [])
    assert host.last_credit == 31 and host.result.held_total == 0


def test_suffix_is_not_normalized_again(monkeypatch):
    host, sink, receipt = setup(b'abc', provenance='changeover')
    original, calls = spec.field_payload, []
    def normalize(data):
        calls.append(data)
        return original(data)
    monkeypatch.setattr(spec, 'field_payload', normalize)
    suffix = spec.field_fill(8)
    chirp(host, receipt, b'abc'+suffix)
    assert calls == [b'abc'+suffix]
    assert sink.payload == independent([b'abc'+suffix]).payload


@pytest.mark.parametrize('operation', ['mismatch', 'cancel', 'epoch', 'unsupported'])
def test_rollback_callback_order_and_conservation(operation):
    prefix = (compress.supervisor_block('B 7')+b'A'*27)[:31]
    held = prefix[:22]
    current = compress.supervisor_block('C 9')+b'Z'
    host, sink, receipt = setup(prefix)
    chirp(host, receipt, held)
    if operation == 'mismatch':
        chirp(host, receipt, current, 2)
    elif operation == 'cancel':
        result = host.cancel(receipt, 'finish')
        assert result == host.result
    elif operation == 'epoch':
        host.end_epoch(1, 'ownership')
    else:
        with pytest.raises(P4StreamFailure):
            chirp(host, receipt, current, status=0x65)
    expected = independent([prefix, held]+([current] if operation == 'mismatch' else []))
    assert sink.events == expected.events
    assert sink.blocks == [('B', b'7'), ('B', b'7')]+([('C', b'9')] if operation == 'mismatch' else [])
    assert host.result.released_source == 22 and host.result.held_delta == -22
    assert host.result.held_total == host.last_credit == 0
    events = list(sink.events)
    host.cancel(receipt, 'again')
    assert sink.events == events and host.result.released_source == 0


@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize('status', [0x65, 0xe1])
def test_unsupported_flushes_before_failure_without_claiming_identity(terminal, status):
    host, sink, receipt = setup(terminal=terminal)
    chirp(host, receipt, PREFIX[:22])
    last = host.last
    with pytest.raises(P4StreamFailure):
        chirp(host, receipt, b'unsupported', status=status)
    assert sink.payload == PREFIX+PREFIX[:22]
    assert host.last == last and host.last_credit == 0
    assert host.result.released_source == 22
    if terminal:
        with pytest.raises(P4StreamFailure):
            chirp(host, receipt, b'new', 2)
        assert host.result.released_source == host.last_credit == 0
    else:
        assert chirp(host, receipt, b'new', 2) == b'new'
        assert sink.payload == PREFIX+PREFIX[:22]+b'new'


def test_terminal_closure_reports_incomplete_supervisor_and_epoch_keeps_state():
    host, sink, receipt = setup(b'A'*30+b'\x1c')
    chirp(host, receipt, b'A'*22)
    result = host.end_epoch(1, 'close', terminal=True)
    assert result.incomplete and host.failure and host.last is None
    assert sink.payload == b'A'*30
    with pytest.raises(P4StreamFailure):
        host.accept(frame(b'7 ', 2))
    host, sink, receipt = setup(b'A'*29+bytes((compress.SB,))+b'B')
    result = host.end_epoch(1, 'ownership')
    assert result.incomplete and not host.failure
    host.accept(frame(b'7 Z', 2))
    assert sink.payload == b'A'*29+b'Z' and sink.blocks == [('B', b'7')]


@pytest.mark.parametrize('provenance', ['opening', 'changeover'])
@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG])
@pytest.mark.parametrize('match', [False, True])
def test_legacy_attempt_spends_shared_receipt(provenance, kind, match):
    host, sink, receipt = setup(provenance=provenance)
    data = (PREFIX if match else b'X'*31)+b'tail'
    kwargs = dict(opening_prefix=PREFIX if provenance == 'opening' and kind == Kind.LONG else None,
                  counted=True, receipt_context=ReceiptContext(receipt, kind))
    output = host.accept(frame(data, 0x61), **kwargs)
    eligible = provenance == 'changeover' or kind == Kind.LONG
    assert output == (b'tail' if match and eligible else data)
    assert host.arm(receipt) == 'spent'
    assert host.accept(frame(data, 0x61), **kwargs) is None
    assert host.last_credit == host.result.committed_credit == 0
    before = sink.payload
    chirp(host, receipt, PREFIX[:22])
    # The BODY receipt remains spent. Counted source continuity independently
    # suppresses an exact retransmission of the NORMAL/LONG field's prefix.
    assert sink.payload == before + (b'' if match else PREFIX[:22])
    assert host.last_credit == host.result.committed_credit == 0


@pytest.mark.parametrize('provenance', ['opening', 'changeover'])
@pytest.mark.parametrize('complete', [False, True])
def test_chirp_selection_retires_both_legacy_paths(provenance, complete):
    host, sink, receipt = setup(provenance=provenance)
    chirp(host, receipt, PREFIX[:22])
    if complete:
        chirp(host, receipt, PREFIX[22:]+b'Z', 2)
    current = PREFIX+b'tail'
    assert host.accept(frame(current, 0x61), opening_prefix=PREFIX, counted=True,
                       receipt_context=ReceiptContext(receipt, Kind.LONG)) == (
                           b'' if complete else PREFIX[:22])+current
    assert host.last_credit == 0
    assert sink.payload == PREFIX+(b'Z' if complete else PREFIX[:22])+current


def test_unarmed_candidate_is_spent_and_registration_cannot_rearm():
    host, sink, receipt = setup(armed=False)
    assert chirp(host, receipt, PREFIX[:22]) == PREFIX[:22]
    host.candidate(receipt, PREFIX, 'opening')
    assert host.arm(receipt) == 'spent'
    assert chirp(host, receipt, PREFIX[22:], 2) == PREFIX[22:]
    assert sink.payload == PREFIX*2


def test_arm_and_body_repeat_do_not_reset_active_cursor():
    host, sink, receipt = setup()
    assert host.accept(frame(PREFIX), receipt_context=ReceiptContext(receipt, Kind.BODY)) is None
    chirp(host, receipt, PREFIX[:22])
    host.candidate(receipt, PREFIX, 'opening')
    assert host.arm(receipt) == 'armed'
    assert chirp(host, receipt, PREFIX[22:], 2) == b''
    assert host.result.cursor == 31 and sink.payload == PREFIX


def test_lost_context_and_incompatible_body_release_held_bytes():
    for context in (ReceiptContext(None, Kind.CHIRP), ReceiptContext(None, Kind.BODY)):
        host, sink, receipt = setup()
        chirp(host, receipt, PREFIX[:22])
        host.accept(frame(b'new', 2), receipt_context=context)
        assert sink.payload == PREFIX+PREFIX[:22]+b'new'
        assert host.result.released_source == 22 and host.arm(receipt) == 'spent'


def test_later_same_identity_after_counter_wrap_is_new_source():
    host, sink, receipt = setup(b'a', provenance='changeover')
    for counter, data in [(1, b'a'), (2, b'b'), (3, b'c'), (0, b'd'), (1, b'a')]:
        chirp(host, receipt, data, counter)
    assert sink.payload == b'abcda'


def test_legacy_context_free_repeat_reports_zero_new_credit():
    host = P4Host()
    current = frame(PREFIX+b'tail', 0x61)
    assert host.accept(current, opening_prefix=PREFIX) == b'tail'
    assert host.last_credit == 31
    assert host.accept(current, opening_prefix=PREFIX) is None
    assert host.last_credit == 0
    assert host.result.committed_credit == 0


def turn_open(prefix=PREFIX, *, negotiate=True):
    turn, _ = machine(role='irs', chirp_ack='alternate', data='body')
    sink = Sink()
    turn.adapter.host = sink
    body = packet(Kind.BODY, FS, prefix)
    receive(turn, body)
    if negotiate:
        controls(turn)
    return turn, sink, body


def controls(turn, *, start=3*FS, key=True, values=(49, 53)):
    for index, value in enumerate(values):
        at = start+index*FS
        obs = Observation(Kind.SHORT, FS, 0, at, at, .758, 0., .9,
                          value=value, header='B', walsh_peak=64)
        receive(turn, RegisteredObservation(obs, Registered.NEW), key=key)


@pytest.mark.parametrize('name', ROWS)
def test_turn_recorded_stream_and_unchanged_alternate_words(name):
    row = ROWS[name]
    turn, _ = machine(role='irs', chirp_ack='alternate', prekey_lead=2132/FS)
    sink = Sink()
    turn.adapter.host = sink
    receive(turn, packet(Kind.BODY, row['body']['onset'], bytes.fromhex(row['body']['data'])))
    for control in row['controls']:
        at = control['onset_sample']
        obs = Observation(Kind.SHORT, FS, at, 0, at, .758, 0., .9,
                          value=control['value'], header=control['header'], walsh_peak=control['walsh_peak'])
        receive(turn, RegisteredObservation(obs, Registered.NEW))
    words = []
    for copy in row['copies']:
        raw = bytes.fromhex(copy['data'])
        incoming = packet(Kind.CHIRP, copy['onset'], raw, copy['status'], len(raw))
        incoming = replace(incoming, observation=replace(incoming.observation,
            chirp=SimpleNamespace(marker=copy['marker'], rank0_upper=copy['lane'] == 'upper')))
        proposal = receive(turn, incoming)
        words.append(proposal.word)
    assert words == row['words']
    assert sink.payload == bytes.fromhex(row['expected'])
    assert sink.blocks == []


@pytest.mark.parametrize('rejection', ['provisional', 'weak', 'unexpected', 'crc', 'expired',
                                       'duplicate', 'extended', 'contradicted'])
@pytest.mark.parametrize('outcome', [Registered.CONFIRMED, Registered.ENRICHED])
def test_turn_rejected_root_can_refine_without_source_side_effects(rejection, outcome):
    turn, sink, _ = turn_open()
    accepted = packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22)
    obs = accepted.observation
    if rejection in ('provisional', 'weak', 'unexpected'):
        rejected = replace(accepted, observation=replace(obs, **{rejection: True}))
    elif rejection == 'crc':
        rejected = replace(accepted, observation=replace(obs, frame=None))
    else:
        rejected = replace(accepted, outcome=Registered(rejection))
    turn.observe(rejected)
    assert sink.payload == PREFIX and turn.adapter.last == (0, PREFIX)
    turn.observe(replace(accepted, outcome=outcome, replaced=rejected.observation))
    turn.observe(packet(Kind.CHIRP, 10*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX+b'Z'


def test_turn_unexpected_to_valid_same_outcome_and_expired_to_valid():
    turn, sink, _ = turn_open()
    first = packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22)
    turn.observe(replace(first, observation=replace(first.observation, unexpected=True)))
    turn.observe(first)
    second = packet(Kind.CHIRP, 10*FS, PREFIX[22:]+b'Z', 0x62, 22)
    turn.observe(replace(second, outcome=Registered.EXPIRED))
    turn.observe(second)
    assert sink.payload == PREFIX+b'Z'


@pytest.mark.parametrize('operation', ['finish', 'close', 'link_down', 'unsupported'])
def test_turn_lifecycle_flushes_tentative_source(operation):
    turn, sink, _ = turn_open()
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    assert sink.payload == PREFIX
    if operation == 'finish':
        turn.finish_application(10*FS)
    elif operation == 'close':
        receive(turn, packet(Kind.BODY, 10*FS, b'', 0x20))
    elif operation == 'link_down':
        turn.link_down('test')
    else:
        receive(turn, packet(Kind.CHIRP, 10*FS, b'bad', 0x65, 22))
    assert sink.payload == PREFIX+PREFIX[:22]


@pytest.mark.parametrize('action', ['proposed', 'refused', 'keyed'])
def test_turn_only_keyed_local_acquisition_flushes(action):
    turn, sink, _ = turn_open()
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    assert sink.payload == PREFIX
    turn._next = Proposal(Kind.BODY, 10*FS, status=0x40, state=State.ISS_GRANT)
    if action == 'refused':
        turn.refused('deadline')
    elif action == 'keyed':
        turn.keyed()
    assert sink.payload == PREFIX+(PREFIX[:22] if action == 'keyed' else b'')
    if action != 'keyed':
        receive(turn, packet(Kind.CHIRP, 14*FS, PREFIX[22:]+b'Z', 0x62, 22))
        assert sink.payload == PREFIX+b'Z'


@pytest.mark.parametrize('associated', [False, True])
def test_turn_handoff_ticket_and_closed_epoch_refinement(associated):
    turn, sink, old = turn_open()
    turn._next = Proposal(Kind.BODY, 6*FS, status=0x40, state=State.ISS_GRANT)
    grant = turn.keyed()
    turn.observe(replace(old, outcome=Registered.CONFIRMED, replaced=old.observation))
    assert sink.payload == PREFIX
    new = packet(Kind.BODY, 8*FS, PREFIX, 0x40)
    receive(turn, replace(new, responds_to=grant.stream_sample if associated else None))
    turn.observe(replace(old, outcome=Registered.CONFIRMED, replaced=old.observation,
                         observation=replace(old.observation, stream_sample=8*FS+3)))
    assert sink.payload == PREFIX*(2 if associated else 1)
    controls(turn, start=10*FS)
    receive(turn, packet(Kind.CHIRP, 13*FS, PREFIX[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 17*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == (PREFIX*2+b'Z' if associated else PREFIX)
    assert turn.adapter.last_credit == (31 if associated else 0)


@pytest.mark.parametrize('mode', ['none', 'unkeyed', 'refused', 'v13', 'v53-only'])
def test_turn_no_positive_sequence_delivers_chirps_without_credit(mode):
    turn, sink, _ = turn_open(negotiate=False)
    if mode != 'none':
        controls(turn, key=mode not in ('unkeyed', 'refused'),
                 values=(13,) if mode == 'v13' else (53,) if mode == 'v53-only' else (49, 53))
        if mode == 'refused':
            turn.refused('deadline')
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 10*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX*2+b'Z' and turn.adapter.last_credit == 0


@pytest.mark.parametrize('reply', ['data', 'idle', 'unassociated'])
def test_turn_ordinary_protocol_two_reply_does_not_arm_chirp_credit(reply):
    turn, arq = machine(role='irs', data='body', ladder='none')
    sink = Sink()
    turn.adapter.host = sink
    responds_to = None
    if reply == 'data':
        arq.on_host_data(b'ours')
        turn._data(0)
        turn.keyed()
        responds_to = 0
    elif reply == 'idle':
        turn._next = Proposal(Kind.NORMAL, 0, status=1, state=State.ISS_HOLD)
        turn.keyed()
        responds_to = 0
    first = replace(packet(Kind.BODY, FS, PREFIX, 0x40), responds_to=responds_to)
    receive(turn, first)
    controls(turn)
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 10*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX*2+b'Z' and turn.adapter.last_credit == 0


def test_turn_accepted_root_and_older_unseen_source_cannot_deliver_again():
    turn, sink, _ = turn_open(negotiate=False)
    first = packet(Kind.CHIRP, 6*FS, b'first', 0x61, 22)
    receive(turn, first)
    receive(turn, packet(Kind.CHIRP, 10*FS, b'next', 0x62, 22))
    before = sink.payload
    turn.observe(replace(first, outcome=Registered.CONFIRMED, replaced=first.observation,
                         observation=replace(first.observation, stream_sample=6*FS+3)))
    turn.observe(packet(Kind.CHIRP, 5*FS, b'old', 0x63, 22))
    assert sink.payload == before


def test_turn_new_epoch_accepts_same_identity_and_rejects_closed_emission():
    turn, sink, body = turn_open()
    ack_word = turn.ack_word
    assert ack_word is not None
    old_answer = next(p.stream_sample for p in turn._emissions.values() if p.ack_identity)
    turn._next = Proposal(Kind.BODY, 6*FS, status=0x40, state=State.ISS_HANDBACK)
    grant = turn.keyed()
    assert turn.adapter.last is None
    for name in ('_last_ack', '_last_body_ack', 'acked_counter'):
        assert getattr(turn, name) is None
    assert turn.ack_word == ack_word
    new = packet(Kind.BODY, 8*FS, PREFIX)
    receive(turn, replace(new, responds_to=grant.stream_sample))
    assert sink.payload == PREFIX*2
    stale = packet(Kind.CHIRP, 9*FS, b'late', 0x61, 22)
    turn.observe(replace(stale, responds_to=old_answer))
    turn.observe(replace(body, outcome=Registered.ENRICHED, replaced=body.observation,
                         observation=replace(body.observation, stream_sample=9*FS)))
    assert sink.payload == PREFIX*2


def test_closed_receive_epoch_rejects_new_chirp_for_old_ack():
    turn, sink, _ = turn_open()
    old_answer = next(p.stream_sample for p in turn._emissions.values() if p.ack_identity)
    turn._next = Proposal(Kind.BODY, 6*FS, status=0x20, state=State.ISS_CLOSE)
    turn.keyed()
    stale = packet(Kind.CHIRP, 8*FS, b'late source', 0x61, 22)
    turn.observe(replace(stale, responds_to=old_answer))
    assert sink.payload == PREFIX
    assert not turn._receive_active


def test_handoff_after_unassociated_body_accepts_identical_ticketed_body():
    turn, sink, _ = turn_open()
    turn._next = Proposal(Kind.BODY, 6*FS, status=0x40, state=State.ISS_GRANT)
    grant = turn.keyed()
    intervening = packet(Kind.BODY, 7*FS, b'new source', 0x40)
    receive(turn, intervening)
    ticketed = packet(Kind.BODY, 8*FS, b'new source', 0x40)
    receive(turn, replace(ticketed, responds_to=grant.stream_sample))
    assert sink.payload == PREFIX+b'new source'
    assert turn._receive_active and turn._receive_epoch == 2
    assert turn._receipt is not None and turn._receipt.epoch == 2
    assert turn.adapter._active == turn._receipt


def test_handoff_retires_intervening_receipt_and_resets_identity():
    turn, sink, _ = turn_open()
    turn._next = Proposal(Kind.BODY, 6*FS, status=0x40, state=State.ISS_GRANT)
    grant = turn.keyed()
    body = packet(Kind.BODY, 8*FS, b'new source', 0x40)
    old = SourceReceipt(turn._receive_epoch, (Kind.BODY, 7*FS),
                        (body.observation.frame.status.byte, body.observation.frame.data))
    turn.adapter.last = old.identity
    turn.adapter.candidate(old, b'new source', 'changeover')
    turn._receipt = old
    assert turn.adapter.arm(old) == 'armed'
    turn.adapter.accept(frame(b'new ', 0x61),
                        receipt_context=ReceiptContext(old, Kind.CHIRP))
    turn.adapter.last = old.identity
    receive(turn, replace(body, responds_to=grant.stream_sample))
    assert sink.payload == PREFIX+b'new '+b'new source'
    assert turn.adapter.arm(old) == 'spent'
    assert turn._receipt is not None and turn._receipt.epoch == 2
    assert turn.adapter._active == turn._receipt


def test_turn_ladder_recovery_and_repeated_v53_preserve_spent_receipt():
    turn, sink, _ = turn_open()
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    controls(turn, start=10*FS)
    receive(turn, packet(Kind.CHIRP, 13*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX+b'Z'
    controls(turn, start=17*FS, values=(53, 53))
    receive(turn, packet(Kind.CHIRP, 21*FS, PREFIX[:22], 0x61, 22))
    assert sink.payload == PREFIX+b'Z'+PREFIX[:22]


@pytest.mark.parametrize('status', [0x60, 0x62, 0xe1, 0x01])
def test_chirp_transaction_requires_nonextended_protocol_three_c1(status):
    host, sink, receipt = setup()
    data = b'\x01\x01\x01'+PREFIX[:19] if status == 0xe1 else PREFIX[:22]
    chirp(host, receipt, data, status=status)
    expected = data[3:] if status == 0xe1 else data
    assert sink.payload == PREFIX+expected and host.last_credit == 0
    assert host.arm(receipt) == 'spent'


def test_recognized_extended_c3_can_complete_matching_transaction():
    host, sink, receipt = setup()
    chirp(host, receipt, PREFIX[:22])
    chirp(host, receipt, b'', 2)
    chirp(host, receipt, b'\x01\x01\x01'+PREFIX[22:]+b'Z', status=0xe3)
    assert sink.payload == PREFIX+b'Z' and host.last_credit == 31


def test_epoch_reset_admits_same_identity_without_resetting_ledger():
    host, sink, receipt = setup()
    chirp(host, receipt, PREFIX[:22])
    host.end_epoch(1, 'local acquisition')
    assert host.last is None and host.arm(receipt) == 'spent'
    body = frame(PREFIX)
    new = SourceReceipt(2, (Kind.BODY, 20*FS), receipt.identity)
    host.accept(body, receipt_context=ReceiptContext(new, Kind.BODY))
    host.candidate(new, PREFIX, 'opening')
    assert host.arm(new) == 'armed'
    chirp(host, new, PREFIX[:22])
    chirp(host, new, PREFIX[22:]+b'Z', 2)
    assert sink.payload == PREFIX+PREFIX[:22]+PREFIX+b'Z'


def test_unknown_receipt_cannot_arm_and_new_candidate_requires_retirement():
    host, _, receipt = setup()
    other = replace(receipt, epoch=2)
    assert host.arm(other) == 'ineligible'
    with pytest.raises(ValueError, match='cancel'):
        host.candidate(other, PREFIX, 'opening')
    host.end_epoch(1, 'ownership')
    host.accept(frame(PREFIX), receipt_context=ReceiptContext(other, Kind.BODY))
    host.candidate(other, PREFIX, 'opening')
    assert host.arm(other) == 'armed'


@pytest.mark.parametrize('mode', ['unkeyed-body', 'refused-body'])
def test_turn_positive_control_sequence_without_keyed_body_cannot_arm(mode):
    turn, _ = machine(role='irs', chirp_ack='alternate', data='body')
    sink = Sink()
    turn.adapter.host = sink
    receive(turn, packet(Kind.BODY, FS, PREFIX), key=False)
    if mode == 'refused-body':
        turn.refused('deadline')
    controls(turn)
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 10*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX*2+b'Z'


def test_frozen_recorded_oracles():
    expected = {
        '0930-2206': b'RMS Trimode 1.4.4.0 Welcome to K4MSU Winlink Gateway at Murray ',
        '0930-2226': b'RMS Trimode 1.4.4.0\r\nW9SSJ has 96 daily minutes remaining with ',
        '1001-0952': b'RMS Trimode 1.4.4.0 Welcome to K4MSU Winlink',
    }
    assert {name: bytes.fromhex(row['expected']) for name, row in ROWS.items()} == expected
    assert [len(row['copies']) for row in ROWS.values()] == [17, 4, 3]
    assert ROWS['1001-0952']['words'] == ['X27', 'X27', 'V27']
    for row in ROWS.values():
        first, repeated = row['copies'][:2]
        assert first['marker'] == 8 and repeated['marker'] == 9
        assert first['lane'] != repeated['lane']
        assert first['status'] == repeated['status'] == 0x61
        assert first['data'] == repeated['data']


@pytest.mark.parametrize('rejection', ['provisional', 'weak', 'unexpected', 'crc'])
def test_turn_rejected_body_cannot_deliver_or_create_candidate(rejection):
    turn, _ = machine(role='irs', chirp_ack='alternate', data='body')
    sink = Sink()
    turn.adapter.host = sink
    first = packet(Kind.BODY, FS, PREFIX)
    bad = replace(first, observation=replace(first.observation,
                  **({'frame': None} if rejection == 'crc' else {rejection: True})))
    turn.observe(bad)
    assert sink.payload == b'' and turn.adapter.last is None
    receive(turn, replace(first, outcome=Registered.CONFIRMED, replaced=bad.observation))
    controls(turn)
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 10*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX+b'Z'


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG])
def test_turn_legacy_receipt_stays_spent_while_counted_source_deduplicates_chirps(kind):
    turn, sink, _ = turn_open()
    receive(turn, packet(kind, 6*FS, PREFIX+b'Z', 0x61, 40))
    assert sink.payload == PREFIX*2+b'Z'
    receive(turn, packet(Kind.CHIRP, 10*FS, PREFIX[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 14*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX*2+b'Z'
    assert turn.adapter.last_credit == turn.adapter.result.committed_credit == 0


def test_turn_tentative_chirp_interrupts_legacy_long_credit():
    turn, sink, _ = turn_open()
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    assert sink.payload == PREFIX
    controls(turn, start=10*FS, values=(49,))
    receive(turn, packet(Kind.LONG, 13*FS, PREFIX+b'Z', 0x61, 40))
    assert sink.payload == PREFIX+PREFIX[:22]+PREFIX+b'Z'
    assert turn.adapter.last_credit == 0


def test_turn_controls_and_rejected_unsupported_source_preserve_tentative_match():
    turn, sink, _ = turn_open()
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    assert sink.payload == PREFIX
    rejected = packet(Kind.CHIRP, 10*FS, b'unsupported', 0x65, 22)
    turn.observe(replace(rejected, observation=replace(rejected.observation, provisional=True)))
    controls(turn, start=11*FS, values=(13,))
    receive(turn, packet(Kind.CHIRP, 14*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX+b'Z' and not turn.adapter.failure


def test_turn_reversed_control_sequence_cannot_arm():
    turn, sink, _ = turn_open(negotiate=False)
    controls(turn, values=(53, 49))
    receive(turn, packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 10*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX*2+b'Z'


@pytest.mark.parametrize('prefix', [b'A', b'B'*22, PREFIX])
@pytest.mark.parametrize('state', [State.ISS_GRANT, State.ISS_HANDBACK])
@pytest.mark.parametrize('status', [0, 0x40])
def test_turn_handoff_short_prefix_and_one_use_ticket(prefix, state, status):
    turn, sink, _ = turn_open()
    turn._next = Proposal(Kind.BODY, 6*FS, status=0x40, state=state)
    grant = turn.keyed()
    first = packet(Kind.BODY, 8*FS, prefix, status)
    receive(turn, replace(first, responds_to=grant.stream_sample))
    controls(turn, start=10*FS)
    receive(turn, packet(Kind.CHIRP, 13*FS, prefix[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 17*FS, prefix[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX+prefix+b'Z'
    repeated_ticket = packet(Kind.BODY, 21*FS, prefix, 0x40)
    receive(turn, replace(repeated_ticket, responds_to=grant.stream_sample))
    controls(turn, start=23*FS)
    receive(turn, packet(Kind.CHIRP, 26*FS, prefix[:22], 0x61, 22))
    receive(turn, packet(Kind.CHIRP, 30*FS, prefix[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX+prefix+b'Z'+prefix*2+b'Z'


@pytest.mark.parametrize('terminate', [False, True])
def test_rollback_preserves_partial_supervisor_parameters(terminate):
    prefix = b'A'*29+bytes((compress.SB,))+b'B'
    host, sink, receipt = setup(prefix)
    chirp(host, receipt, prefix[:22])
    if terminate:
        host.end_epoch(1, 'close', terminal=True)
    else:
        chirp(host, receipt, b'7 Z', 2)
    expected = independent([prefix, prefix[:22]]+([] if terminate else [b'7 Z']))
    assert sink.events == expected.events
    assert sink.blocks == ([] if terminate else [('B', b'A'*22+b'7')])
    assert host.result.incomplete == terminate
    assert host.result.released_source == 22


def test_turn_boundary_discards_old_deferred_source_answers_but_keeps_emissions():
    turn, sink, _ = turn_open()
    history = dict(turn._emissions)
    incoming = packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22)
    due = 6*FS+round(turn.t[Kind.CHIRP]*FS)
    turn.observe(incoming, now=due-turn.lead+1)
    assert turn.decide(due-turn.lead+1).stream_sample == due
    turn.refused('deadline')
    assert sink.payload == PREFIX and turn._deferred is not None
    turn._next = Proposal(Kind.BODY, 10*FS, status=0x40, state=State.ISS_GRANT)
    grant = turn.keyed()
    first = packet(Kind.BODY, 12*FS, b'new epoch', 0x40)
    receive(turn, replace(first, responds_to=grant.stream_sample))
    assert sink.payload == PREFIX+PREFIX[:22]+b'new epoch'
    assert turn._deferred is None
    assert not any(p.get('body') and p['body'].stream_sample < 12*FS
                   for p in turn.pending.values())
    assert all(turn._emissions[at] == emission for at, emission in history.items())


@pytest.mark.parametrize('answer', ['refused', 'deferred'])
def test_turn_ack_loss_does_not_undo_source_acceptance(answer):
    turn, sink, _ = turn_open()
    incoming = packet(Kind.CHIRP, 6*FS, PREFIX[:22], 0x61, 22)
    if answer == 'refused':
        receive(turn, incoming, key=False)
        turn.refused('deadline')
    else:
        due = 6*FS+round(turn.t[Kind.CHIRP]*FS)
        turn.observe(incoming, now=due-turn.lead+1)
    assert sink.payload == PREFIX
    receive(turn, packet(Kind.CHIRP, 10*FS, PREFIX[22:]+b'Z', 0x62, 22))
    assert sink.payload == PREFIX+b'Z' and turn.adapter.last_credit == 31
