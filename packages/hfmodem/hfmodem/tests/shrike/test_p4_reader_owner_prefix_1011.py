# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""R1 physical read-origin proof before the unchanged actual-key assertions."""
from hfmodem.tests.shrike.test_p4_second_delta_driver_1011 import lifecycle
from hfmodem.tests.shrike.reader_owner_support_1011 import record


def _assert_prefix(result, row):
    m, rx, tx, live, captures, keys, counts, feeds, replays = result
    owner = row['first_owner']
    assert owner['machine'] is m and owner['receiver'] is rx
    assert owner['proposal'] is keys[0] is captures[0][0]
    generation = owner['generation']
    assert generation is not None
    assert generation == owner['proposal'].peer_watch_generation
    assert generation == owner['watch'].generation == owner['receiver_watch'].generation
    reads = row['reads']
    word = 8236750
    packet = 8208670
    ordinary = [r for r in reads if r[1] is None and r[2] == word]
    watched = [r for r in reads if r[1] == generation and r[2] == packet]
    forbidden = [r for r in ordinary if r[4] >= word + 159840 and r[4] < word + 180000]
    record('r1_prefix_boundary', ordinary=ordinary, watched=watched, forbidden=forbidden,
           actual_keys=[(p.stream_sample, p.word) for p in keys],
           first_assertion='watch support then WORD full-read boundary, before key-prefix')
    assert {r[0] for r in watched} >= {'read_partial', 'read_window'}
    assert all(r[4] == packet + 161040 for r in watched)
    assert {r[4] for r in ordinary} >= {8247070, 8277310, 8284750}
    assert not forbidden, 'associated WORD origin borrowed saved-phase LONG full completion'
    assert [(p.stream_sample, p.word) for p in keys[:4]] == [
        (8191870, 'Y27'), (8371870, 'Y27'), (8551870, 'Y27'), (8731870, 'Y27')]
    assert [(p.stream_sample, p.word) for p in keys[4:6]] == [(8911870, 'X27'), (9091870, 'X27')]
    assert next(p.stream_sample for p in keys if p.word == 'Y27' and p.stream_sample > 8731870) == 9811870
    assert counts[:5] == [0, 1, 2, 3, 4]

    return generation


def _run_prefix():
    from hfmodem.tests.shrike.second_delta_support_1011 import EVENTS
    begin = len(EVENTS)
    result = lifecycle(True, 'exhaustion')
    rows = [r for r in EVENTS[begin:] if r['label'] == 'driver_lifecycle']
    assert len(rows) == 1
    row = rows[0]  # Capture this invocation before any subsequent lifecycle.
    return _assert_prefix(result, row), row


def test_r1_actual_word_origin_does_not_borrow_watch_completion():
    _run_prefix()


def test_r1_two_real_lifecycles_bind_distinct_increasing_owners():
    first, first_row = _run_prefix()
    second, second_row = _run_prefix()
    assert first_row is not second_row
    assert first < second
    assert not [r for r in second_row['reads'] if r[1] == 1 and r[2] == 8208670]
