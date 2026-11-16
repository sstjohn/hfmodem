# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json

import pytest

from hfmodem.station.chat_transport import (
    Events, KiB, MiB, TerminalResults, TokenBucket, encode_event,
)


def test_encoded_bytes_include_escaped_json_and_sse_framing():
    events = Events()
    payload = {'text': '\x00"\\\n😀' * 100}
    event = events.publish('tape', payload)
    expected = (f"id: {event['seq']}\nevent: tape\ndata: {json.dumps(event, separators=(chr(44), chr(58)), ensure_ascii=False)}\n\n").encode()
    assert event.encoded == expected
    assert events.encoded_bytes == len(expected)
    assert events.encoded_bytes > len(payload['text'].encode())
    payload['text'] = 'changed'
    assert events.after(0)[0]['data']['text'] != 'changed'
    assert encode_event(event) is event.encoded
    assert encode_event(dict(event)) == expected


def test_partition_eviction_reports_exact_interior_and_trailing_holes():
    events = Events(command_bytes=2000, other_bytes=900)
    events.publish('command', {'id': 'first', 'state': 'done'})  # 1
    for i in range(2):
        events.publish('tape', {'text': 'x' * 400})  # 2, 3 (2 evicted)
    events.publish('command', {'id': 'second', 'state': 'done'})  # 4
    events.publish('tape', {'text': 'x' * 400})  # 5 (3 evicted)
    replay = events.after(0)
    assert [event['seq'] for event in replay] == [1, 3, 4, 5]
    assert replay[1]['type'] == 'gap'
    assert replay[1]['data'] == {'from': 1, 'to': 4}
    assert events.byte_counts['command'] <= 2000
    assert events.byte_counts['other'] <= 900
    events = Events(capacity=2)
    events.publish('command', {'n': 1})
    events.publish('command', {'n': 2})
    events.publish('tape', {'n': 3})
    replay = events.after(0)
    assert [e['seq'] for e in replay] == [1, 2, 3]
    assert replay[0]['data'] == {'from': 0, 'to': 2}
    assert replay[-1]['type'] == 'tape'
    assert events.after(3) == []


def test_separate_interior_holes_and_batch_count_progress():
    events = Events(command_bytes=500, other_bytes=500, batch_count=2)
    events.publish('command', {'text': 'x' * 150})  # 1
    events.publish('tape', {'text': 'x' * 150})  # 2
    events.publish('command', {'text': 'x' * 150})  # 3, evicts 1
    events.publish('tape', {'text': 'x' * 150})  # 4, evicts 2
    events.publish('command', {'text': 'x' * 150})  # 5, evicts 3
    assert [(e['type'], e['seq']) for e in events.after(0)] == [('gap', 3), ('tape', 4)]
    assert [(e['type'], e['seq']) for e in events.after(4)] == [('command', 5)]


def test_slow_reader_never_copies_an_unbounded_batch():
    events = Events()
    for i in range(180):
        events.publish('tape', {'text': '\x00' * 500, 'n': i})
    cursor, observed = 0, []
    while cursor < events.seq:
        batch = events.after(cursor)
        assert batch
        assert len(batch) <= 128
        assert sum(len(encode_event(event)) for event in batch) <= 256 * KiB
        observed.extend(e['data']['n'] for e in batch)
        cursor = batch[-1]['seq']
    assert observed == list(range(180))
    assert events.encoded_bytes <= 8 * MiB
    tiny = Events(batch_count=3)
    for n in range(10):
        tiny.publish('tape', {'n': n})
    assert len(tiny.after(0)) == 3


def test_age_count_and_future_cursor_compatibility():
    now = [0]
    events = Events(capacity=2, age=10, clock=lambda: now[0])
    for n in range(3):
        events.publish('tape', {'n': n})
    assert [e['seq'] for _, e in events.ring] == [2, 3]
    assert events.after(0)[0]['data'] == {'from': 0, 'to': 2}
    assert events.after(100)[0]['data'] == {'from': 100, 'to': 2}
    now[0] = 11
    events.publish('tape', {'n': 3})
    assert [e['seq'] for e in events.after(2)] == [3, 4]
    now[0] = 21
    assert events.after(4) == []
    assert not events.ring and events.encoded_bytes == 0
    assert events.after(0)[0]['data'] == {'from': 0, 'to': 5}


def test_oversized_escaped_event_does_not_consume_sequence():
    events = Events()
    with pytest.raises(ValueError, match='reference'):
        events.publish('tape', {'text': '\x00' * (64 * KiB // 6)})
    assert events.seq == 0 and events.encoded_bytes == 0
    with pytest.raises(ValueError, match='unknown'):
        events.publish('invented', {})


@pytest.mark.parametrize('factory,budget', [(TokenBucket.sse, 256 * KiB),
                                           (TokenBucket.bulk, 4 * MiB)])
def test_rate_budget_burst_refill_and_retry(factory, budget):
    now = [0.0]
    limiter = factory(clock=lambda: now[0])
    assert limiter.consume(budget)
    assert not limiter.consume(1)
    assert limiter.retry_after(budget // 2) == .5
    now[0] = .5
    assert limiter.consume(budget // 2)
    assert not limiter.consume(1)
    now[0] = 100
    assert not limiter.consume(budget + 1)
    assert limiter.retry_after(budget + 1) == float('inf')
    assert limiter.consume(budget)


def test_command_lookup_survives_chat_overflow_and_is_identity_scoped():
    results = TerminalResults()
    result = {'state': 'done', 'result': {'controller_id': 'controller'}}
    results.remember('incarnation', 'client', 'startup', result)
    result['state'] = 'mutated'
    events = Events(other_bytes=1000)
    for n in range(100):
        events.publish('tape', {'n': n, 'text': 'x' * 200})
    assert events.after(0)[0]['type'] == 'gap'
    retained = results.lookup('incarnation', 'client', 'startup')
    assert retained['status'] == 'found' and retained['result']['state'] == 'done'
    retained['result']['state'] = 'also mutated'
    assert results.lookup('incarnation', 'client', 'startup')['result']['state'] == 'done'
    assert results.lookup('other', 'client', 'startup') == {'status': 'expired'}
    assert results.lookup('incarnation', 'other', 'startup') == {'status': 'expired'}


def test_result_cache_eviction_bounds_count_bytes_and_replacement():
    results = TerminalResults(capacity=2, byte_limit=500)
    for n in range(3):
        results.remember('inc', 'client', str(n), {'n': n})
    assert results.lookup('inc', 'client', '0') == {'status': 'expired'}
    assert results.lookup('inc', 'client', '1')['status'] == 'found'
    results.remember('inc', 'client', '1', {'text': '\x00' * 100})
    assert results.lookup('inc', 'client', '1')['result'] == {'n': 1}
    assert results.encoded_bytes <= 500
    defaults = TerminalResults()
    for n in range(1001):
        defaults.remember('inc', 'client', str(n), {'n': n})
    assert defaults.lookup('inc', 'client', '0') == {'status': 'expired'}
    assert defaults.lookup('inc', 'client', '1000')['status'] == 'found'
    assert defaults.encoded_bytes <= 2 * MiB


def test_default_partitions_bound_maximum_escaped_traffic():
    events = Events()
    for n in range(40):
        events.publish('command', {'id': str(n), 'state': 'done', 'text': '\x00' * 10000})
    command_sequences = [e['seq'] for _, e in events.ring if e['type'] == 'command']
    assert events.byte_counts['command'] <= 2 * MiB
    assert command_sequences[0] > 1  # command partition really overflowed
    for n in range(150):
        events.publish('tape', {'n': n, 'text': '\x00' * 10000})
    assert events.byte_counts['other'] <= 6 * MiB
    assert events.encoded_bytes <= 8 * MiB
    assert [e['seq'] for _, e in events.ring if e['type'] == 'command'] == command_sequences
    assert events.encoded_bytes == sum(len(e.encoded) for _, e in events.ring)


def test_multiple_interior_gaps_are_reported_before_each_later_event():
    events = Events(other_bytes=400)
    events.publish('command', {'n': 1})
    events.publish('tape', {'text': 'x' * 200})
    events.publish('command', {'n': 3})
    events.publish('tape', {'text': 'x' * 200})
    events.publish('command', {'n': 5})
    events.publish('tape', {'text': 'x' * 200})
    replay = events.after(1)
    assert [(e['type'], e['seq']) for e in replay] == [
        ('gap', 2), ('command', 3), ('gap', 4), ('command', 5), ('tape', 6)]
    assert [e['data'] for e in replay if e['type'] == 'gap'] == [
        {'from': 1, 'to': 3}, {'from': 3, 'to': 5}]


def test_result_cache_aggregate_byte_eviction():
    results = TerminalResults(byte_limit=400)
    results.remember('inc', 'client', 'one', {'text': '\x00' * 30})
    results.remember('inc', 'client', 'two', {'text': '\x00' * 30})
    assert results.lookup('inc', 'client', 'one') == {'status': 'expired'}
    assert results.lookup('inc', 'client', 'two')['status'] == 'found'
    assert results.encoded_bytes <= 400


def test_result_pages_capture_immutable_values_and_stable_cursors():
    from hfmodem.station.chat_transport import ResultPages
    from hfmodem.contact import chat_state as s
    pages = ResultPages(page_bytes=650)
    value = {'_type': 'FileEntries', 'generation': 4,
             'entries': [{'n': n, 'name': '\x00😀' * 15} for n in range(20)]}
    first = pages.capture(value, read_id='captured')
    value['entries'].clear()
    combined, current = [], first
    while True:
        assert len(pages._encode(current)) <= 650
        assert current['value']['generation'] == 4
        combined.extend(current['value']['entries'])
        assert pages.page('captured', current['cursor']) == current
        if current['next_cursor'] is None:
            break
        current = pages.page('captured', current['next_cursor'])
    assert [row['n'] for row in combined] == list(range(20))
    first['value']['entries'].clear()
    assert pages.page('captured')['value']['entries']
    typed = s.ReadPage('read', '0', None, s.HelpText(None, 'hello'))
    assert s.from_wire(s.to_wire(typed)) == typed
    for cursor in ('-1', '01', '99999', '☃', 1):
        with pytest.raises(ValueError, match='cursor'):
            pages.page('captured', cursor)


def test_result_pages_text_age_byte_and_count_limits():
    from hfmodem.station.chat_transport import ResultPages
    now = [0]
    pages = ResultPages(page_bytes=250, capacity=1, byte_limit=3000, age=10,
                        clock=lambda: now[0])
    text = '😀\x00' * 100
    current = pages.capture({'_type': 'HelpText', 'topic': None, 'text': text}, read_id='one')
    reconstructed = ''
    while True:
        reconstructed += current['value']['text']
        if current['next_cursor'] is None:
            break
        current = pages.page('one', current['next_cursor'])
    assert reconstructed == text
    pages.capture({'_type': 'HelpText', 'topic': None, 'text': 'small'}, read_id='two')
    with pytest.raises(ValueError, match='expired'):
        pages.page('one')
    now[0] = 10
    with pytest.raises(ValueError, match='expired'):
        pages.page('two')
    assert pages.encoded_bytes == 0
    with pytest.raises(ValueError, match='budget'):
        pages.capture({'rows': [{'huge': 'x' * 500}, {'other': 1}]})
    with pytest.raises(ValueError, match='retention'):
        ResultPages(page_bytes=250, byte_limit=500).capture({'text': 'x' * 1000})


def test_chat_live_state_limit_counts_complete_sse_envelope():
    events = Events()
    with pytest.raises(ValueError, match='transport limit'):
        events.publish('chat.state', {'text': 'x' * (8 * KiB)})
    assert events.seq == 0
    event = events.publish('chat.event', {'text': '😀' * (48 * KiB // 4)})
    assert len(event.encoded) <= 64 * KiB


def test_count_pressure_retains_new_command_after_other_partition_fills_ring():
    events = Events(capacity=10)
    for n in range(12):
        events.publish('tape', {'n': n})
    command = events.publish('command', {'id': 'new', 'state': 'accepted'})
    assert events.after(command['seq'] - 1) == [command]
    assert [e['seq'] for _, e in events.ring] == list(range(4, 14))
    assert events.encoded_bytes == sum(len(e.encoded) for _, e in events.ring)
