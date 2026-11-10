# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""N0BBS: retrying a partially matched resend cannot release duplicates."""
import json
import hashlib
from pathlib import Path
import pytest
from hfmodem.kestrel.vara import compression as C
from hfmodem.kestrel.vara import vara_arq as A
from hfmodem.kestrel.arq import phy
from hfmodem.tests.kestrel.test_compression import station


def capture():
    d = json.loads((Path(__file__).parent / 'fixtures/n0bbs-prefix-1025/payloads.json').read_text())
    return (bytes.fromhex(d['original']), bytes.fromhex(d['partial_retry']),
            [bytes.fromhex(x) for x in d['restarted_chunks']])


def receive(h, data, last=False):
    wire = C.encode_text(data) if h.session_compression == 'TEXT' else data
    body = phy.vara_body(wire, h.caller, tail=0x81,
                         body_len=len(wire) + (20 if last else 1))
    if h.session_compression == 'TEXT':
        body = body[:-1] + bytes([body[-1] | C.TEXT_FLAG])
    assert phy.vara_payload(body, caller=h.caller, body_len=len(body)) == wire
    before = len(b''.join(h.io.data_received))
    h._deliver([body])
    h._finish_delivery_window()
    return b''.join(h.io.data_received)[before:]


def arm(h, payload):
    if h._rx_window.K is None:
        h.state = A.VaraState.CONNECTED
        h.role, h.called, h.turn = 'responder', 'WI9YZA', A._TURN_PEER
        receive(h, payload)
    assert h._rx_window.K.end - h._rx_window.K.start == len(payload)
    h._rx_query(query_start=h._unkeyed_at + 60)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert h._rx_window.T is not None
    assert h._rx_window.T.frontier == h._rx_window.S


def test_recorded_restarted_prefix_is_suppressed_without_losing_new_bytes():
    h, io = station()
    original, partial, chunks = capture()
    arm(h, original)
    assert receive(h, partial, False) == b''
    arm(h, partial)
    # Several queries about the same over must not add duplicate obligations.
    for _ in range(3):
        h._rx_query(query_start=h._unkeyed_at + 60)
        assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    out = b''.join(receive(h, x, False) for x in chunks)
    replay = b''.join(chunks)
    assert replay.startswith(original)
    assert out == replay[len(original):]


def test_peer_continuing_instead_of_restarting_retains_every_new_byte():
    h, io = station()
    original, partial, chunks = capture()
    arm(h, original)
    assert receive(h, partial, False) == b''
    arm(h, partial)
    assert receive(h, original[len(partial):] + b'new', False) == b'new'


def test_second_restart_inside_first_retry_remains_exact():
    h, io = station()
    original, partial, chunks = capture()
    arm(h, original)
    assert receive(h, partial, False) == b''
    arm(h, partial)
    first = chunks[0]
    assert receive(h, first, False) == b''
    arm(h, first)
    assert receive(h, first, False) == b''
    out = b''.join(receive(h, x, False) for x in chunks[1:])
    assert out == b''.join(chunks)[len(original):]


def test_unrelated_or_early_final_data_is_not_suppressed():
    """DESIGN §5: behind a nested NAK the next window must agree with R at
    exactly one candidate. Unrelated DATA agrees at neither and closes the
    link (R8) unanswered; an early terminal agreeing at the window's start is
    the resend and takes the final answer. Neither releases a byte."""
    for last, data in [(False, b'unrelated'), (True, b'a')]:
        h, io = station()
        arm(h, b'abcdef')
        assert receive(h, b'abc') == b''
        arm(h, b'abc')
        assert receive(h, data, last) == b''
        if last:
            assert h._rx_window.A.kind == 'final' and h._rx_window.A.sent
        else:
            assert h._rx_window.closed and h._receive_reports[-1].code == 'R8'
        assert b''.join(io.data_received) == b'abcdef'


@pytest.mark.parametrize('mode', ['OFF', 'TEXT'])
@pytest.mark.parametrize('repeat_partial', [False, True])
def test_recorded_payloads_through_delivery_and_compression(mode, repeat_partial):
    h, io = station(mode)
    h._start_compression(mode)
    original, partial, chunks = capture()
    receive(h, original)
    assert b''.join(io.data_received) == original
    receive(h, partial)
    assert b''.join(io.data_received) == original + partial
    if repeat_partial:
        receive(h, partial)
    for chunk in chunks:
        receive(h, chunk)
    assert b''.join(io.data_received) == original + partial * (2 if repeat_partial else 1) + b''.join(chunks)
    assert not io.warnings


    from hfmodem.contact.varac import MessageParser
    meta, raw = _recorded_stream()
    assert hashlib.sha256(raw).hexdigest() == meta['received_sha256']
    with pytest.raises(ValueError, match='Invalid VarAC message length'):
        MessageParser().feed(raw)
    corrected = raw[:meta['original_chunk_offset']] + b''.join(io.data_received)
    parser = MessageParser()
    with pytest.raises(ValueError, match='Invalid VarAC message length'):
        parser.feed(corrected)
    assert len(corrected) == len(raw) + len(partial) * int(repeat_partial)


@pytest.mark.parametrize('mode', ['OFF', 'TEXT'])
@pytest.mark.parametrize('repeat_partial', [False, True])
def test_recorded_payloads_after_each_qualifying_aired_nak(mode, repeat_partial):
    h, io = station(mode)
    h._start_compression(mode)
    original, partial, chunks = capture()
    arm(h, original)
    assert receive(h, partial) == b''
    if repeat_partial:
        arm(h, partial)
        assert receive(h, partial) == b''
    arm(h, partial)
    for chunk in chunks:
        receive(h, chunk)
    assert b''.join(io.data_received) == b''.join(chunks)
    assert not io.warnings
    from hfmodem.contact.varac import MessageParser
    meta, raw = _recorded_stream()
    assert hashlib.sha256(raw).hexdigest() == meta['received_sha256']
    with pytest.raises(ValueError, match='Invalid VarAC message length'):
        MessageParser().feed(raw)
    corrected = raw[:meta['original_chunk_offset']] + b''.join(io.data_received)
    parser = MessageParser()
    assert parser.feed(corrected) == []
    assert len(corrected) == len(raw) - meta['duplicate_bytes']
    assert len(parser.pending) == 12233


def _recorded_stream():
    p = Path(__file__).parent / 'fixtures/n0bbs-prefix-1025'
    raw = json.loads((p / 'payloads.json').read_text())['received'].encode('ascii')
    return json.loads((p / 'provenance.json').read_text()), raw


@pytest.mark.parametrize('split', [False, True], ids=['one-delivery', 'at-the-resend'])
def test_recorded_prefix_failure_is_rejected_whole_and_the_link_resyncs(split):
    """The 12709-byte length frames bytes the duplicate shifted; nothing in the
    broken feed is delivered, every byte is in the evidence, and the next
    delivery opening with a length header is read."""
    from hfmodem.contact.varac import frame_message
    from hfmodem.tests.contact.test_varac import Host
    meta, raw = _recorded_stream()
    host = Host()
    cut = meta['original_chunk_offset'] if split else len(raw)
    host.data.incoming.extend([raw[:cut]] + ([raw[cut:]] if split else []))
    host.client.poll()
    if split:
        host.client.poll()
    errors = [json.loads(e.raw) for e in host.events if e.kind == 'peer_protocol_error']
    assert len(errors) == 1 and errors[0]['bytes'] == len(raw) and errors[0]['truncated']
    assert not host.client.messages and host.client._resyncing
    host.data.incoming.extend([b'CgSCCpEFrY4C', frame_message('<SFB>next')])
    host.client.poll()
    host.client.poll()
    assert [json.loads(e.raw)['bytes'] for e in host.events
            if e.kind == 'peer_protocol_error'][1:] == [12]
    assert [text for _, text in host.client.messages] == ['<SFB>next']
    assert not host.client._resyncing and not host.command.closed


def test_rearmed_prefix_disagreement_closes_at_r8_without_releasing_bytes():
    h, io = station()
    transmissions = []
    io.tx = transmissions.append
    arm(h, b'abcdef')
    assert receive(h, b'abc') == b''
    arm(h, b'abc')
    sent = len(transmissions)
    assert receive(h, b'unrelated') == b''
    assert h._rx_window.closed and h._receive_reports[-1].code == 'R8'
    assert len(transmissions) == sent
    assert b''.join(io.data_received) == b'abcdef'


def test_rearmed_early_terminal_does_not_release_committed_prefix():
    h, io = station()
    arm(h, b'abcdef')
    assert receive(h, b'abc') == b''
    arm(h, b'abc')
    assert receive(h, b'a', last=True) == b''
    assert h._rx_window.A.kind == 'final' and h._rx_window.A.sent
    assert b''.join(io.data_received) == b'abcdef'
