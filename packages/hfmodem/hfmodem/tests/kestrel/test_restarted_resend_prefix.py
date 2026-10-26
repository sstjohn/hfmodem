# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KQ4QWH: retrying a partially matched resend cannot release duplicates."""
import json
import hashlib
from pathlib import Path
import pytest
from hfmodem.kestrel.vara import compression as C
from hfmodem.tests.kestrel.test_compression import station


def capture():
    d = json.loads((Path(__file__).parent / 'fixtures/kq4qwh-prefix-1025/payloads.json').read_text())
    return (bytes.fromhex(d['original']), bytes.fromhex(d['partial_retry']),
            [bytes.fromhex(x) for x in d['restarted_chunks']])


def arm(h, payload):
    h._delivered = (payload,)
    h._retry_payload = lambda b: b
    h._expect_resend()


def test_recorded_restarted_prefix_is_suppressed_without_losing_new_bytes():
    h, io = station()
    original, partial, chunks = capture()
    arm(h, original)
    assert h._past_resend(partial, False) == b''
    arm(h, partial)
    # Several queries about the same over must not add duplicate obligations.
    for _ in range(3): h._expect_resend()
    out = b''.join(h._past_resend(x, False) for x in chunks)
    replay = b''.join(chunks)
    assert replay.startswith(original)
    assert out == replay[len(original):]


def test_peer_continuing_instead_of_restarting_retains_every_new_byte():
    h, io = station()
    original, partial, chunks = capture()
    arm(h, original)
    assert h._past_resend(partial, False) == b''
    arm(h, partial)
    assert h._past_resend(original[len(partial):] + b'new', False) == b'new'


def test_second_restart_inside_first_retry_remains_exact():
    h, io = station()
    original, partial, chunks = capture()
    arm(h, original)
    assert h._past_resend(partial, False) == b''
    arm(h, partial)
    first = chunks[0]
    assert h._past_resend(first, False) == b''
    arm(h, first)
    assert h._past_resend(first, False) == b''
    out = b''.join(h._past_resend(x, False) for x in chunks[1:])
    assert out == b''.join(chunks)[len(original):]


def test_unrelated_or_early_final_data_is_not_suppressed():
    for last, data in [(False, b'unrelated'), (True, b'a')]:
        h, io = station()
        arm(h, b'abcdef')
        assert h._past_resend(b'abc', False) == b''
        arm(h, b'abc')
        assert h._past_resend(data, last) == b'abc' + data


@pytest.mark.parametrize('mode', ['OFF', 'TEXT'])
@pytest.mark.parametrize('repeat_partial', [False, True])
def test_recorded_payloads_through_delivery_and_compression(mode, repeat_partial):
    h, io = station(mode)
    h._start_compression(mode)
    original, partial, chunks = capture()
    def body(data):
        return (C.encode_text(data) + bytes([C.TEXT_FLAG]) if mode == 'TEXT'
                else data + b'\x00')
    h._deliver([body(original)])
    assert b''.join(io.data_received) == original
    h._expect_resend()
    h._deliver([body(partial)])
    assert b''.join(io.data_received) == original
    if repeat_partial:
        h._deliver([body(partial)])
    h._expect_resend()
    for chunk in chunks:
        h._deliver([body(chunk)])
    assert b''.join(io.data_received) == b''.join(chunks)
    assert not io.warnings
    # The corrupt host stream crossed the application's declared length.
    # With replay suppression it remains an incomplete, well-framed download.
    from hfmodem.contact.varac import MessageParser
    p = Path(__file__).parent / 'fixtures/kq4qwh-prefix-1025'
    meta = json.loads((p / 'provenance.json').read_text())
    raw = (p / 'received.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == meta['received_sha256']
    with pytest.raises(ValueError, match='Invalid VarAC message length'):
        MessageParser().feed(raw)
    corrected = raw[:meta['original_chunk_offset']] + b''.join(io.data_received)
    parser = MessageParser()
    assert parser.feed(corrected) == []
    assert len(corrected) == len(raw) - meta['duplicate_bytes']
    assert len(parser.pending) == 12233
