# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import os
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy, rxwindow as RW
from hfmodem.kestrel.rx import varahf500 as RX500
from hfmodem.kestrel.tx import varahf500_tx as TX500
from hfmodem.kestrel.vara import compression as C
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_data_over_gate import _IO, _connected


class TraceIO(_IO):
    def __init__(self):
        super().__init__()
        self.events = []
        self.warnings = []
        self.accept = True

    def tx(self, samples):
        super().tx(samples)
        self.events.append(('key', b''))

    def tx_went_out(self):
        return self.accept

    def data(self, data):
        super().data(data)
        self.events.append(('host', bytes(data)))

    def compression_error(self, warning):
        self.warnings.append(warning)


def station(role='responder', turn=A._TURN_PEER, mode='OFF', high=True, bw='2300'):
    io = TraceIO()
    hs, _ = _connected(role=role, bw=bw, io=io)
    hs.turn = turn
    hs._link_high_speed = high
    hs._start_compression(mode)
    hs._rx_read = (7 if high else 4, None)
    hs._expect_level = hs._rx_read[0]
    return hs, io


def receive(monkeypatch, hs, *bodies, hold=False, audio=None):
    monkeypatch.setattr(hs, '_peer_data_over', lambda _: list(bodies))
    return hs._answer_data_over(np.zeros(10) if audio is None else audio, hold=hold)


def full(payload=b'next', size=90, field=0x85):
    return payload.ljust(size - 1, b'.') + bytes([field])


def short(hs, payload, *, compressed=False, size=90):
    body = bytearray(phy.vara_body(payload, hs.caller, body_len=size))
    if compressed:
        body[-1] |= C.TEXT_FLAG
    return bytes(body)


def query(hs, phase=None, delay=6.4, **kw):
    return hs._rx_query(phase=phase, query_start=hs._unkeyed_at + delay, **kw)


def stored(data):
    return len(data).to_bytes(2, 'big') + data


def deflated(data):
    encoder = zlib.compressobj(wbits=-15)
    body = encoder.compress(data) + encoder.flush()
    return (0x8000 | len(body)).to_bytes(2, 'big') + body


@pytest.mark.parametrize('caller', ['W1AGC', 'W0AEB', 'W1AEU', 'W0AGT'])
@pytest.mark.parametrize('marker_form', ['raw', 'xara'])
def test_computed_crc_marker_is_accepted(caller, marker_form):
    marker = VF.crc16_genibus(caller.encode()) >> 8
    assert marker in (0, 0x14)
    body = b'payload\x14' + bytes([1 if marker_form == 'xara' else marker]) + b'\0' * 13 + b'\x80'
    assert phy.vara_payload(body, caller=caller, body_len=len(body)) == b'payload'


@pytest.mark.parametrize('compressed', [False, True])
def test_bare_delimiter_and_full_literal_controls(compressed):
    field = 0x80 | (C.TEXT_FLAG if compressed else 0)
    body = b'bcdefghi\x14' + bytes([field])
    assert phy.vara_payload(body, caller='W9SSJ', body_len=10) == b'bcdefghi'
    literal = body[:-1] + bytes([field | 1])
    assert phy.vara_payload(literal, caller='W9SSJ', body_len=10) == b'bcdefghi\x14'
    with pytest.raises(ValueError, match='delimiter'):
        phy.vara_payload(b'abcdefghi' + bytes([field]), caller='W9SSJ', body_len=10)


def test_raw_14_whole_trailer_precedes_bare_delimiter():
    assert phy.vara_payload(b'abc\x14\x14\x80', caller='W1AEU', body_len=6) == b'abc'


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_malformed_short_is_a_hole_and_never_emits(monkeypatch, role):
    hs, io = station(role)
    assert receive(monkeypatch, hs, b'badframe!\x80')
    assert not io.host
    assert hs._rx_window.A.kind == 'nak'
    assert hs._rx_window.A.sent
    assert hs._rx_window.K is None


def test_located_unread_bw500_after_commit_keeps_retry_new(monkeypatch):
    hs, io = station(role='initiator', high=False, bw='500')
    body = b'A' * 43 + b'\x99'
    assert receive(monkeypatch, hs, body)
    assert b''.join(io.host) == b'A' * 43
    hs._base_rx = SimpleNamespace(history=[SimpleNamespace(
        refined_score=A._UNREAD_LOCK, candidate=SimpleNamespace(level=4),
        result=SimpleNamespace(complete=False), pair=False, frames=(), end=0)], pending=[])
    hs._unkeyed_at = A.time.monotonic() - 6.4
    hs._heard = int(6.4 * MK.FS)
    hs._ans_skip = 0
    hs._stream_answer(np.concatenate([
        MK.synth_burst(hs.called, VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE, hs.bw)),
        np.zeros(MK.FS // 10)]))
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert hs._rx_window.T is None
    assert receive(monkeypatch, hs, body)
    assert b''.join(io.host) == b'A' * 86
    assert hs._rx_window.T is None
    assert (hs._rx_window.K.start, hs._rx_window.K.end) == (43, 86)


@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_located_unread_bw500_before_phased_query_keeps_retry_new(monkeypatch, route):
    now = [100.0]
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    hs, io = station(role='initiator', high=False, bw='500')
    body = b'A' * 43 + b'\x99'
    assert receive(monkeypatch, hs, A._UNDECODED)
    assert query(hs)
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert not hs._rx_window.trusted and not io.host
    assert receive(monkeypatch, hs, body)
    assert b''.join(io.host) == b'A' * 43
    assert hs._rx_window.A.phase == 1 and hs._rx_window.A.sent
    assert not hs._rx_window.trusted and hs._rx_window.T is None
    hs._base_rx = SimpleNamespace(history=[SimpleNamespace(
        refined_score=A._UNREAD_LOCK, candidate=SimpleNamespace(level=4),
        result=SimpleNamespace(complete=False), pair=False, frames=(), end=0)], pending=[])
    pairs = VF.over_continue_frames(hs.called, hs.bw, (11,))[11][1]
    audio = np.concatenate([np.zeros(MK.HOP), MK.synth_tone_pairs(pairs), np.zeros(MK.HOP)])
    answer_at = hs._unkeyed_at
    now[0] = answer_at + 6.4 + (len(audio) - MK.HOP) / MK.FS
    hs._heard = int((now[0] - answer_at) * MK.FS)
    hs._ans_skip = 0
    before = len(io.sent)
    if route == 'stream':
        assert hs._stream_answer(audio)
    else:
        hs.on_rx_audio(audio)
    assert len(io.sent) == before + 1
    assert hs._query_start == pytest.approx(answer_at + 6.4, abs=0.02)
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert hs._rx_window.W is None and hs._rx_window.T is None
    assert b''.join(io.host) == b'A' * 43
    assert receive(monkeypatch, hs, body)
    assert len(b''.join(io.host)) == 86
    assert b''.join(io.host) == b'A' * 86
    assert hs._rx_window.T is None
    assert (hs._rx_window.K.start, hs._rx_window.K.end) == (43, 86)


@pytest.mark.parametrize('field', [0x80, 0xa0])
def test_nonfinal_trailer_padding_hole_naks_whole_window(monkeypatch, field):
    hs, io = station(role='initiator')
    bad = bytes.fromhex('61 62 63 14 f6 00 00 00 5a') + bytes([field])
    assert receive(monkeypatch, hs, bad, short(hs, b'end', size=10))
    assert b''.join(io.host) == b''
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert hs._rx_window.K is None
    assert hs._receive_reports[-1].code == 'malformed'
    assert hs._receive_reports[-1].rejected == bad


def test_terminal_trailer_keeps_level_byte_allowance(monkeypatch):
    hs, io = station(role='initiator')
    body = bytes.fromhex('61 62 63 14 f6 00 00 00 0d 82')
    assert receive(monkeypatch, hs, body)
    assert b''.join(io.host) == b'abc'
    assert hs._rx_window.A.kind == 'final' and hs._rx_window.A.sent


@pytest.mark.parametrize('verdict', ['NONE', 'CERTAIN'])
@pytest.mark.parametrize('phase', [0, 1])
def test_streamed_validated_query_ends_held_window(monkeypatch, verdict, phase):
    hs, io = station(role='initiator')
    hs._lv_owed = True
    hs._holds_epoch = 6 if verdict == 'CERTAIN' else 0
    assert receive(monkeypatch, hs, full(b'head'), hold=True)
    assert hs._rx_window.W.verdict == verdict
    assert hs._held_answer is not None and not io.host and not io.sent
    hs._decoded_frame_end = A.time.monotonic() - 1
    pairs = VF.over_continue_frames(hs.called, hs.bw, (11 + phase,))[11 + phase][1]
    audio = np.concatenate([np.zeros(MK.HOP), MK.synth_tone_pairs(pairs), np.zeros(MK.HOP)])
    hs._stream_answer(audio)
    hs._release_held_answer(len(audio))
    assert hs._rx_window.W is None
    assert hs._held_answer is None
    assert hs._rx_window.A.kind == ('continue' if verdict == 'NONE' else 'nak')
    assert hs._rx_window.A.sent and len(io.sent) == 1
    assert b''.join(io.host) == (full(b'head')[:-1] if verdict == 'NONE' else b'')


@pytest.mark.parametrize('phase', [0, 1])
def test_initiator_bracket_query_preserves_phase_and_answer_debt(monkeypatch, phase):
    hs, io = station(role='initiator')
    assert receive(monkeypatch, hs, full(b'first'))
    assert receive(monkeypatch, hs, full(b'second'))
    assert hs._rx_window.trusted and hs._rx_window.A.phase == 1
    before = b''.join(io.host)
    hs._unkeyed_at = A.time.monotonic() - 6.4
    pairs = VF.over_continue_frames(hs.called, hs.bw, (11 + phase,))[11 + phase][1]
    audio = np.concatenate([np.zeros(MK.HOP), MK.synth_tone_pairs(pairs), np.zeros(MK.HOP)])
    hs.on_rx_audio(audio)
    assert len(io.sent) == 3
    assert hs._answer_owed == A._OWED_OVER
    assert hs._rx_window.A.kind == ('continue' if phase else 'nak')
    assert hs._rx_window.A.sent and hs._rx_window.T is None
    assert b''.join(io.host) == before


@pytest.mark.parametrize('turn', [A._TURN_PEER, A._TURN_OURS])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_text_preflight_fault_precedes_both_answer_orders(monkeypatch, turn, role):
    hs, io = station(role, turn, 'TEXT')
    prefix = b'verified text'
    good = short(hs, C.encode_text(prefix), compressed=True)
    bad = short(hs, b'\0', compressed=True)
    assert receive(monkeypatch, hs, good, bad)
    assert b''.join(io.host) == prefix
    assert not io.sent
    assert len(io.warnings) == 1
    assert hs.state == A.VaraState.DISCONNECTED
    assert hs._receive_reports[-1].code == 'fault'


@pytest.mark.parametrize('turn', [A._TURN_PEER, A._TURN_OURS])
def test_each_text_frame_is_prepared_once_before_key(monkeypatch, turn):
    hs, io = station(turn=turn, mode='TEXT')
    decoded = C.decode_text
    calls = []

    def decode(payload):
        calls.append(payload)
        assert not io.sent
        return decoded(payload)

    payloads = [C.encode_text(b'first'), C.encode_text(b'second')]
    bodies = [short(hs, p, compressed=True) for p in payloads]
    monkeypatch.setattr(C, 'decode_text', decode)
    receive(monkeypatch, hs, *bodies)
    assert calls == payloads
    assert b''.join(io.host) == b'firstsecond'
    assert len(io.sent) == 1
    assert [event[0] for event in io.events] == (
        ['host', 'key'] if turn == A._TURN_PEER else ['key', 'host'])


@pytest.mark.parametrize('turn', [A._TURN_PEER, A._TURN_OURS])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_files_fault_preflight_emits_no_failed_block(monkeypatch, turn, role):
    hs, io = station(role, turn, 'FILES')
    receive(monkeypatch, hs, short(hs, stored(b'held') + b'\x80\x01\xff'))
    assert not io.host and not io.sent
    assert hs.state == A.VaraState.DISCONNECTED
    assert len(io.warnings) == 1
    assert b''.join(b.data for b in hs._receive_reports[-1].blocks) == b'held'


def test_files_stored_and_following_deflate_wait_for_clean_close(monkeypatch):
    hs, io = station(mode='FILES')
    wire = stored(b'stored') + deflated(b'later' * 3)
    head = wire + b'\x14\xf6' + b'\0' * (87 - len(wire)) + b'\xa0'
    receive(monkeypatch, hs, head)
    assert not io.host
    assert hs._rx_window.files.blocks
    receive(monkeypatch, hs, short(hs, b''))
    assert b''.join(io.host) == b'stored' + b'later' * 3
    assert not hs._rx_window.files.blocks


def test_deflate_releases_when_its_structure_validates():
    decoder = C.ReceiveCompression('FILES')
    wire = deflated(b'x' * 32500)
    assert decoder.feed(wire[:-1], 0, False) == (b'', None)
    assert decoder.feed(wire[-1:], 0, False) == (b'x' * 32500, None)


@pytest.mark.parametrize('incomplete', [False, True])
def test_files_abort_discards_held_and_partial_and_restarts_decoder(monkeypatch, incomplete):
    hs, io = station(mode='FILES')
    wire = stored(b'held') + (b'\0\x20part' if incomplete else b'')
    head = wire + b'\x14\xf6' + b'\0' * (87 - len(wire)) + b'\xa0'
    receive(monkeypatch, hs, head)
    assert not io.host
    old = hs._rx_compression
    hs.turn = A._TURN_OURS
    assert hs._rx_compression is not old
    assert not hs._rx_compression.buffer and not hs._rx_compression.damaged
    assert hs._receive_reports[-1].code == 'abort'
    assert b''.join(b.data for b in hs._receive_reports[-1].blocks) == b'held'
    hs.turn = A._TURN_PEER
    receive(monkeypatch, hs, short(hs, stored(b'new')))
    assert b''.join(io.host) == b'new'


def test_files_hold_bound_reports_every_held_block():
    decoder = C.ReceiveCompression('FILES')
    blocks = stored(b's' * 32500)
    for _ in range(4):
        assert decoder.feed(blocks, 0, False) == (b'', None)
    data, warning = decoder.feed(stored(b'x' * 1073), 0, False)
    assert not data and 'hold exhausted' in warning and decoder.damaged


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('phase', [0, 1])
def test_only_proving_row_skips_late_query_bound(monkeypatch, role, phase):
    hs, io = station(role)
    receive(monkeypatch, hs, full(b'first'))
    if phase:
        receive(monkeypatch, hs, full(b'second'))
    before = b''.join(io.host)
    query(hs, phase)
    assert b''.join(io.host) == before
    assert hs._rx_window.A.kind == ('continue' if phase else 'nak')
    assert (hs._rx_window.T is None) == bool(phase)


@pytest.mark.parametrize('phase', [0, 1])
@pytest.mark.parametrize('level', [7, 10, 12])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_responder_reads_caller_eight_symbol_phase_at_l7_up(monkeypatch, phase, level, route):
    hs, io = station()
    hs._rx_read = (level, None)
    receive(monkeypatch, hs, full(b'first'))
    receive(monkeypatch, hs, full(b'second'))
    before = b''.join(io.host)
    hs._unkeyed_at = A.time.monotonic() - 6.4
    pairs = VF.over_continue_frames(hs.called, hs.bw, (11 + phase,))[11 + phase][0]
    audio = np.concatenate([np.zeros(MK.HOP), MK.synth_tone_pairs(pairs), np.zeros(MK.HOP)])
    if route == 'stream':
        hs._stream_responder_control(audio)
    else:
        hs.on_rx_audio(audio)
    assert hs._rx_window.A.kind == ('continue' if phase else 'nak')
    assert b''.join(io.host) == before
    assert len(io.sent) == 3


@pytest.mark.parametrize('group', ['short', 'long'])
@pytest.mark.parametrize('phase', [0, 1])
def test_responder_passes_caller_tone_query_phase(monkeypatch, group, phase):
    hs, io = station()
    receive(monkeypatch, hs, full(b'first'))
    receive(monkeypatch, hs, full(b'second'))
    hs._unkeyed_at = A.time.monotonic() - 6.4
    kinds = VF.SESSION_ANSWER_QUERIES_SHORT if group == 'short' else VF.SESSION_INTERMEDIATE_ANSWER_QUERIES
    hs.on_rx_audio(MK.synth_burst(hs.called, kinds[phase]))
    assert hs._rx_window.A.kind == ('continue' if phase else 'nak')
    assert b''.join(io.host) == full(b'first')[:-1] + full(b'second')[:-1]


@pytest.mark.parametrize('kind', ['continue', 'frame10', 'final'])
@pytest.mark.parametrize('inside', [False, True])
def test_silent_timer_never_authorizes_positive_or_partial(monkeypatch, kind, inside):
    hs, io = station(bw='500' if kind == 'frame10' else '2300', high=kind != 'frame10')
    if kind == 'frame10':
        hs._answer_pair_loss_500(SimpleNamespace(level=4, quality=None), full(size=44))
    else:
        receive(monkeypatch, hs, short(hs, b'final') if kind == 'final' else full())
    before = len(io.sent)
    hs._unkeyed_at = A.time.monotonic() - (0.1 if inside else 6.4)
    hs._reack(recognized=False)
    if kind == 'final':
        assert hs._rx_window.A.kind == 'final'
    elif inside:
        assert len(io.sent) == before
    else:
        assert hs._rx_window.A.kind == 'nak'
        assert hs._rx_window.T is not None


@pytest.mark.parametrize('phase', [0, 1])
def test_abort_phase_stays_untrusted_until_delivery_finishes(monkeypatch, phase):
    hs, io = station()
    for _ in range(phase + 1):
        receive(monkeypatch, hs, full())
    hs.turn = A._TURN_OURS
    hs.turn = A._TURN_PEER
    assert hs._rx_window.alt == 0 and not hs._rx_window.trusted
    receive(monkeypatch, hs, full(b'fresh'))
    receive(monkeypatch, hs, full(b'another'))
    query(hs, 1)
    assert hs._rx_window.A.kind == 'nak'
    receive(monkeypatch, hs, short(hs, b'end'))
    assert hs._rx_window.alt == 0 and hs._rx_window.trusted


def test_possible_single_backstop_closes_before_replay_is_armed(monkeypatch):
    hs, io = station()
    hs._holds_any = 6
    receive(monkeypatch, hs, full())
    assert hs._rx_window.K.possible_single
    before = len(io.sent)
    query(hs, 0)
    assert hs.state == A.VaraState.DISCONNECTED
    assert hs._rx_window.T is None and len(io.sent) == before
    assert hs._receive_reports[-1].code == 'R8'
    assert b''.join(io.host) == full()[:-1]


def test_certainty_survives_quiet_without_companion(monkeypatch):
    hs, io = station()
    hs._holds_epoch = hs._holds_any = 6
    receive(monkeypatch, hs, full())
    assert not io.host and not io.sent
    assert hs._rx_window.state == 'OPEN'
    assert hs._rx_window.W.verdict == 'CERTAIN'
    assert query(hs)
    assert hs._rx_window.A.kind == 'nak'
    assert not io.host
    assert hs._rx_window.K is None


@pytest.mark.parametrize('service', ['initiator-idle', 'responder-idle', 'responder-tick'])
@pytest.mark.parametrize('evidence', ['slot', 'ceiling'])
@pytest.mark.parametrize('busy', [False, True], ids=['quiet', 'receiving'])
def test_held_certain_clock_services_slot_and_ceiling(monkeypatch, service, evidence, busy):
    now = [100.0]
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    role, timer = service.split('-')
    hs, io = station(role=role, high=False)
    hs._holds_epoch = hs._holds_any = 6
    if evidence == 'slot':
        hs._note_frame_end(10, 10)
    assert receive(monkeypatch, hs, full(field=0x99), hold=True)
    assert hs._rx_window.state == 'OPEN' and hs._held_answer is not None
    deadline = 100 + (hs._companion_samples() if evidence == 'slot'
                      else A._ANSWER_HOLD_MAX) / MK.FS
    events = []
    event = hs._rx_window.event

    def record(kind, **kwargs):
        events.append(kind)
        return event(kind, **kwargs)

    monkeypatch.setattr(hs._rx_window, 'event', record)
    tick = hs.idle_keepalive if timer == 'idle' else hs._answerer_tick
    io.receiving = busy
    now[0] = deadline - .001
    tick()
    assert hs._rx_window.state == 'OPEN' and not io.host and not io.sent
    now[0] = deadline + .001
    tick()
    assert events == ['E' if evidence == 'slot' else 'C']
    assert hs._rx_window.W is None and hs._held_answer is None
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent == (not busy)
    assert not io.host and not hs._rx_window.R
    if busy:
        assert not io.sent
        io.receiving = False
        assert hs._rx_query()
        assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert len(io.sent) == 1
    assert not io.host and not hs._rx_window.R


@pytest.mark.parametrize('bw', ['2300', '2750', '500'])
@pytest.mark.parametrize('close', ['data', 'answer', 'finish'])
@pytest.mark.parametrize('holds', [0, 6], ids=['NONE', 'CERTAIN'])
def test_quiet_close_routes_respect_certain_slot(monkeypatch, bw, close, holds):
    now = [100.0]
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    hs, io = station(role='initiator', high=False, bw=bw)
    hs._holds_epoch = hs._holds_any = holds
    if bw == '500':
        hs._peer_run_500 = holds
        hs._peer_over_state = (4, 0x99)
    hs._note_frame_end(10, 10)
    body = full(size=44 if bw == '500' else 90, field=0x99)
    receive(monkeypatch, hs, body, hold=close != 'data')
    if close == 'answer':
        hs._key_over_answer(False, False)
    elif close == 'finish':
        hs._finish_delivery_window()
    if not holds:
        assert b''.join(io.host) == body[:-1]
        assert hs._rx_window.W is None and len(io.sent) == 1
    else:
        assert not io.host and not io.sent and hs._rx_window.state == 'OPEN'
        deadline = hs._decoded_frame_end + hs._companion_samples() / MK.FS
        now[0] = deadline - .001
        hs._release_held_answer(1)
        assert not io.host and not io.sent and hs._rx_window.state == 'OPEN'
        now[0] = deadline + .001
        hs._release_held_answer(1)
        assert b''.join(io.host) == (body[:-1] if bw == '500' else b'')
        assert len(io.sent) == 1
        assert hs._rx_window.W is None
        assert hs._rx_window.A.kind == ('frame10' if bw == '500' else 'nak')
        assert hs._rx_window.A.sent


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_bw500_bracket_records_frame_end_before_trailing_quiet(monkeypatch, level):
    now = [100.0]
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    hs, io = station(role='initiator', high=False, bw='500')
    hs._rx_read, hs._expect_level = (level, None), level
    payload = b'A' * RX500._PAYLOAD[level]
    frame = (TX500.build_frame(payload, 0x99) if level == 4 else
             TX500.build_l3_frame(payload, 0x99, level))
    audio = np.pad(TX500.synth_burst(frame, onset=24000, level=level), (0, MK.FS // 2))
    rec = RX500.INDEX_RECORDS.get(level)
    end = 24000 + ((rec.lead + rec.ncols) * rec.dw if rec else
                   (9 + RX500.NSYM) * RX500.H)
    hs.on_rx_audio(audio)
    assert b''.join(io.host) == payload and len(io.sent) == 1
    assert hs._decoded_frame_end == pytest.approx(100 - (len(audio) - end) / MK.FS, abs=.025)
    assert hs._decoded_frame_end <= 99.5


def test_refused_commit_reanswer_flips_and_counts_once(monkeypatch):
    hs, io = station()
    io.accept = False
    receive(monkeypatch, hs, full())
    assert b''.join(io.host) == full()[:-1]
    assert hs._rx_window.alt == 0 and hs._holds_any == 0
    io.accept = True
    query(hs, 0, delay=0.1)
    assert hs._rx_window.alt == 1 and hs._holds_any == 1
    query(hs, 0, delay=0.1)
    assert hs._rx_window.alt == 1 and hs._holds_any == 1
    assert b''.join(io.host) == full()[:-1]


def test_query_budget_closes_without_spending_refusals(monkeypatch):
    hs, io = station()
    receive(monkeypatch, hs, full())
    receive(monkeypatch, hs, full(b'phase one'))
    io.accept = False
    for _ in range(3):
        query(hs, 1)
    assert hs._since_progress == 0
    io.accept = True
    idle = hs.idle_keyed
    for _ in range(A._MAX_WITHOUT_PROGRESS):
        query(hs, 1)
    assert hs.idle_keyed == idle
    assert hs._since_progress == A._MAX_WITHOUT_PROGRESS
    query(hs, 1)
    assert hs.state == A.VaraState.DISCONNECTED
    assert hs._receive_reports[-1].code == 'budget'


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('explicit', [False, True], ids=['idle', 'query'])
@pytest.mark.parametrize('first', ['F', 'X'])
def test_greeting_idles_are_free_until_first_data(monkeypatch, role, explicit, first):
    hs, io = station(role)
    hs._grant_rx_window()
    before = hs._reacks, hs._since_progress
    for _ in range(A._MAX_WITHOUT_PROGRESS + A._REACK_MAX):
        assert query(hs, explicit=explicit)
        assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
        assert (hs._reacks, hs._since_progress) == before
        assert hs.state == A.VaraState.CONNECTED
    if first == 'F':
        receive(monkeypatch, hs, full(b'greeting'))
    else:
        receive(monkeypatch, hs, A._UNDECODED)
    for _ in range(A._MAX_WITHOUT_PROGRESS + A._REACK_MAX + 1):
        query(hs, explicit=explicit)
        if hs.state == A.VaraState.DISCONNECTED:
            break
    assert hs.state == A.VaraState.DISCONNECTED
    assert hs._receive_reports[-1].code == 'budget'
    assert b''.join(io.host) == (full(b'greeting')[:-1] if first == 'F' else b'')


def certain_single_nak(monkeypatch, hs, body):
    hs._reset_peer_holds(confirmed=True)
    hs._holds_epoch = hs._holds_any = 6
    hs._decoded_frame_end = A.time.monotonic() - 100
    receive(monkeypatch, hs, body)
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert hs._rx_window.W is None


def lesson_audio(hs, *bodies, inconclusive=False):
    from hfmodem.kestrel.tx import varahf2300_tx as TX
    key = phy.level_key(hs.bw, hs._rx_read[0])
    size = phy.body_size(hs.bw, key)
    # The decoder bodies above are stubbed, including the lower-level axis's
    # old 90-byte body. Render a valid frame at the actual read level for timing.
    bodies = [body[:-1][:size - 1].ljust(size - 1, b'.') + body[-1:]
              for body in bodies]
    frames = [TX.synth_frame(TX.build_frame(body, key), key,
                             over=7 if n == 0 else None)
              for n, body in enumerate(bodies)]
    if inconclusive:
        frames.append(frames[0][:len(frames[0]) // 2])
    audio = np.pad(np.concatenate(frames), (MK.FS // 4, 8 * MK.FS))
    extent = 'inconclusive' if inconclusive else 'single' if len(bodies) == 1 else 'double'
    assert phy.burst_extent(audio, hs.bw, hs._rx_read[0]) == extent
    return audio


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_speed_limited_wide_level_four_after_six_holds_is_certain(monkeypatch, bw):
    hs, io = station(high=False, bw=bw)
    bodies = [full(bytes([n]), field=0x99) for n in range(6)]
    for body in bodies:
        receive(monkeypatch, hs, body)
    assert hs._holds_epoch == hs._holds_any == 6
    assert b''.join(io.host) == b''.join(body[:-1] for body in bodies)
    assert hs._peer_double_verdict() == 'CERTAIN'


def test_older_kestrel_single_resend_teaches_none_for_session(monkeypatch):
    hs, io = station(high=False)
    bodies = [full(bytes([n])) for n in range(12)]
    for body in bodies[:6]:
        receive(monkeypatch, hs, body)
    assert hs._peer_double_verdict() == 'CERTAIN'
    hs._decoded_frame_end = A.time.monotonic() - 100
    receive(monkeypatch, hs, bodies[6])
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert b''.join(io.host) == b''.join(b[:-1] for b in bodies[:6])
    # DESIGN §3: "a same-level resend that ends cleanly as a single teaches NONE".
    receive(monkeypatch, hs, bodies[6], audio=lesson_audio(hs, bodies[6]))
    assert hs._peer_double_verdict() == 'NONE'
    assert hs._rx_window.K.frames[0].body == bodies[6]
    # The lesson changes expectation, never placement: this resend is POSSIBLE.
    assert hs._rx_window.K.possible_single
    for body in bodies[7:]:
        receive(monkeypatch, hs, body)
        assert hs._peer_double_verdict() == 'NONE'
    hs._holds_epoch = hs._holds_any = 12
    receive(monkeypatch, hs, short(hs, b'end'))
    assert b''.join(io.host) == b''.join(b[:-1] for b in bodies) + b'end'
    assert hs._rx_window.A.kind == 'final'


@pytest.mark.parametrize('resend', ['peer-double', 'xara-lower', 'inconclusive', 'hole', 'refused'])
def test_resend_without_same_level_clean_single_teaches_nothing(monkeypatch, resend):
    hs, io = station(high=False)
    body = full(b'retry')
    if resend == 'refused':
        io.accept = False
        hs._holds_epoch = hs._holds_any = 6
        hs._decoded_frame_end = A.time.monotonic() - 100
        receive(monkeypatch, hs, body)
        assert not hs._rx_window.A.sent
        io.accept = True
    elif resend == 'hole':
        hs._holds_epoch = hs._holds_any = 6
        receive(monkeypatch, hs, body, None)
        query(hs)
    else:
        certain_single_nak(monkeypatch, hs, body)
    before = b''.join(io.host)
    if resend == 'peer-double':
        # DESIGN §3: "A double, a lower-level resend, a located failure or an
        # inconclusive end teaches nothing."
        receive(monkeypatch, hs, body, full(b'tail'),
                audio=lesson_audio(hs, body, full(b'tail')))
        assert b''.join(io.host) == before + body[:-1] + full(b'tail')[:-1]
    elif resend == 'xara-lower':
        hs._rx_read = (3, None)
        receive(monkeypatch, hs, body)
        assert b''.join(io.host) == before + body[:-1]
    else:
        # The inconclusive axis needs an actual inconclusive end (§3 above);
        # refused/hole history still cannot teach from a clean single resend.
        receive(monkeypatch, hs, body, hold=resend == 'inconclusive',
                audio=lesson_audio(hs, body, inconclusive=resend == 'inconclusive'))
        if resend == 'inconclusive':
            assert hs._rx_window.W is not None and b''.join(io.host) == before
            hs._perform_rx_actions(hs._rx_window.event('C'))
    hs._reset_peer_holds(confirmed=True)
    hs._holds_epoch = hs._holds_any = 6
    assert hs._peer_double_verdict() == 'CERTAIN'
    receive(monkeypatch, hs, full(b'next head'), full(b'next tail'))
    certain_single_nak(monkeypatch, hs, body)
    # DESIGN §3: "a same-level resend that ends cleanly as a single teaches NONE".
    receive(monkeypatch, hs, body, audio=lesson_audio(hs, body))
    assert hs._peer_double_verdict() == 'NONE'


@pytest.mark.parametrize('reset', ['turn', 'session', 'disconnect'])
def test_t_clears_learned_single_peer(monkeypatch, reset):
    hs, io = station(high=False)
    body = full(b'retry')
    certain_single_nak(monkeypatch, hs, body)
    # DESIGN §3: "a same-level resend that ends cleanly as a single teaches NONE";
    # "T clears the lesson."
    receive(monkeypatch, hs, body, audio=lesson_audio(hs, body))
    hs._holds_epoch = hs._holds_any = 6
    assert hs._peer_double_verdict() == 'NONE'
    if reset == 'disconnect':
        hs.disconnect()
    else:
        hs._perform_rx_actions(hs._rx_window.event('T', new_session=reset == 'session'))
    hs._reset_peer_holds(confirmed=True)
    hs._holds_epoch = hs._holds_any = 6
    assert hs._peer_double_verdict() == 'CERTAIN'
    assert b''.join(io.host) == body[:-1]


def test_xara_record_zero_bare_head_and_empty_close_deliver_exact_256(monkeypatch):
    hs, io = station(high=False)
    hs._rx_read = (1, None)
    payload = bytes(range(256))
    receive(monkeypatch, hs, full(payload[:248], size=249))
    head = payload[248:] + b'\x14\x80'
    close = phy.vara_body(b'', hs.caller, body_len=10)
    assert phy.vara_payload(head, caller=hs.caller, body_len=10) == payload[248:]
    receive(monkeypatch, hs, head, hold=True)
    assert hs._rx_window.W.expected == frozenset((2,))
    assert b''.join(io.host) == payload[:248]
    receive(monkeypatch, hs, close)
    assert b''.join(io.host) == payload
    assert len(io.sent) == 2
    assert hs._rx_window.A.kind == 'final'


def test_raw_waveform_does_not_record_a_window_answer(monkeypatch):
    hs, io = station()
    receive(monkeypatch, hs, full())
    before = (hs._rx_window.alt, hs._rx_window.A.sent, hs._rx_window.T,
              hs._peer_answered, hs._holds_epoch, hs._holds_any)
    assert hs._send_burst(VF.SESSION_OVER_NAK)
    assert (hs._rx_window.alt, hs._rx_window.A.sent, hs._rx_window.T,
            hs._peer_answered, hs._holds_epoch, hs._holds_any) == before


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_malformed_frames_exhaust_nak_budget_and_report_rejected_bytes(monkeypatch, role):
    hs, io = station(role)
    malformed = b'badframe!\x80'
    for _ in range(A._REACK_MAX + 1):
        receive(monkeypatch, hs, malformed)
    assert hs.state == A.VaraState.DISCONNECTED
    assert not io.host and len(io.sent) == A._REACK_MAX
    assert hs._receive_reports[-1].code == 'budget'
    assert any(r.code == 'malformed' and r.rejected == malformed for r in hs._receive_reports)


def test_bw500_sixth_single_uses_the_previous_five_answers(monkeypatch):
    hs, io = station(high=False, bw='500')
    hs._rx_read = (1, None)
    hs._peer_over_state = (1, 0x99)
    hs._peer_run_500 = 5
    body = full(b'sixth', size=10, field=0x99)
    def decoded(_):
        hs._note_peer_500(1, [body])
        return [body]
    monkeypatch.setattr(hs, '_peer_data_over', decoded)
    assert hs._answer_data_over(np.zeros(10))
    assert b''.join(io.host) == body[:-1]
    assert hs._peer_run_500 == 6
    assert hs._peer_double_verdict() == 'CERTAIN'
    assert hs._rx_window.K.frames[0].body == body


def test_query_ending_an_open_window_returns_its_keying_result(monkeypatch):
    hs, io = station()
    receive(monkeypatch, hs, full(), hold=True)
    assert not io.sent and not io.host
    assert hs._rx_query(query_start=hs._unkeyed_at + .1)
    assert len(io.sent) == 1 and b''.join(io.host) == full()[:-1]
    assert hs._rx_window.A.sent and hs._rx_window.W is None


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('chat', [False, True])
def test_final_query_preserves_the_original_control_frame(monkeypatch, role, chat):
    hs, io = station(role=role, high=False, bw='500')
    hs.chat = hs._chat_profile = chat
    assert receive(monkeypatch, hs, short(hs, b'last'))
    answer = io.sent[-1].copy()
    assert hs._rx_window.A.original == ('control', int(chat))
    assert query(hs, delay=60)
    np.testing.assert_array_equal(io.sent[-1], answer)
    assert b''.join(io.host) == b'last'


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_limited_pair_head_commits_at_frame_ten_and_reanswers_exactly(monkeypatch, role):
    hs, io = station(role=role, high=False, bw='500')
    hs._rx_read = 1, None
    hs._peer_over_state, hs._peer_run_500 = (1, 0x99), 6
    body = full(b'head', size=10, field=0x99)
    assert receive(monkeypatch, hs, body, None)
    assert hs._rx_window.A.kind == 'frame10' and hs._rx_window.A.sent
    assert hs._owed_block and b''.join(io.host) == body[:-1]
    answer = io.sent[-1].copy()
    assert query(hs, phase=0, delay=.1)
    np.testing.assert_array_equal(io.sent[-1], answer)
    assert b''.join(io.host) == body[:-1]


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_proving_next_phase_uses_missing_data_nak_on_plain_wide_link(monkeypatch, bw):
    hs, io = station(role='initiator', high=False, bw=bw)
    body = full(field=0x99)
    receive(monkeypatch, hs, body)
    assert hs._rx_window.A.phase == 0 and hs._rx_window.trusted
    assert query(hs, phase=1, delay=.1)
    want = MK.synth_burst(hs.called, VF.for_bw(VF.SESSION_OVER_NAK_SHORT, bw))
    np.testing.assert_array_equal(io.sent[-1], want)
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.phase == 1
    assert hs._rx_window.T is None and b''.join(io.host) == body[:-1]


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_grant_does_not_reuse_an_earlier_direct_nak_form(role, bw):
    hs, io = station(role=role, high=False, bw=bw)
    hs._rx_nak_form = 'direct'
    hs._grant_rx_window()
    assert query(hs, phase=0, delay=.1)
    kind = (VF.SESSION_OVER_NAK_RESPONDER_SHORT if role == 'responder'
            else VF.SESSION_OVER_NAK if bw == '500' else VF.SESSION_OVER_NAK_SHORT)
    np.testing.assert_array_equal(io.sent[-1], MK.synth_burst(hs.called, VF.for_bw(kind, bw)))
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert not io.host and hs._rx_window.T is None


REPLAY_DERIVATIVES = [('test_compression', 'test_off_ignores_text_bit_and_preserves_duplicate_handling'), ('test_receive_turn_duplicates', 'test_identical_message_after_completed_handover_is_new_data'), ('test_receive_turn_duplicates', 'test_asking_for_turn_does_not_forget_unconfirmed_peer_delivery'), ('test_bw2750_nak', 'test_partial_window_recovery_delivers_only_missing_suffix_once'), ('test_fit_hint', 'test_u4_peer_confirms_reset'), ('test_fit_hint', 'test_u4_reply_bits_do_not_create_another_counting_window'), ('test_fit_hint', 'test_u4_equal_payload_new_window_occurrences'), ('test_fit_hint', 'test_u4_repeated_pair_retains_window_occurrence'), ('test_fit_hint', 'test_u4_idle_completed_single_retransmission_counts_once'), ('test_bw500_pairing', 'test_six_delivered_singles_then_pairs_until_level_change_duplicate_or_close'), ('test_bw500_pairing', 'test_duplicate_read_resets_pair_run_without_redelivery'), ('test_bw500_pairing', 'test_fetch_013206_retry_starts_six_singles'), ('test_bw500_oracle_0923', 'test_a_close_resent_at_a_lower_record_is_not_delivered_twice'), ('test_two_over_window', 'test_a_repeated_window_on_the_stream_is_not_delivered_twice'), ('test_two_over_window', 'test_identical_blocks_in_one_window_keep_their_multiplicity'), ('test_two_over_window', 'test_a_refused_missing_block_nak_is_retained_for_the_next_gap'), ('test_two_over_window', 'test_the_repeat_of_a_two_over_window_is_not_delivered_twice'), ('test_chat_receive_silence', 'test_reply_at_previously_requested_rate_is_delivered_once_after_extra_nak'), ('test_vara_mail_rx', 'test_a_repeated_over_is_answered_but_delivered_once'), ('test_data_over_gate', 'test_a_repeated_greeting_draws_the_per_over_response_not_a_second_request'), ('test_limited_responder_query_recovery', 'test_caller_nak_routes_name_the_missing_or_saved_over_through_refusals'), ('test_kl7qr_retry_rate', 'test_recorded_retry_after_possibly_heard_direct_nak_is_delivered_once'), ('test_n0bbs_retry_rate', 'test_recorded_retry_delivered_once_and_acknowledged'), ('test_intermediate_over_answer', 'test_a_repeated_full_over_repeats_the_continue_answer'), ('test_vara_giveup', 'test_a_repeated_over_is_traffic_and_not_progress'), ('test_w5jab_receive_recovery', 'test_good_resend_delivers_once_then_queries_repeat_ack_not_nak'), ('test_wide_answer_query', 'test_lost_final_behind_deferred_ask_is_delivered_once')]


@pytest.mark.parametrize('source,scenario', REPLAY_DERIVATIVES,
                         ids=[name for _, name in REPLAY_DERIVATIVES])
def test_held_scenario_repeat_after_aired_nak(monkeypatch, source, scenario):
    from .receive_answer_helpers import arm_replay
    bw = '2750' if source in ('test_bw2750_nak', 'test_bw500_oracle_0923') else '500' if source in ('test_bw500_pairing', 'test_chat_receive_silence', 'test_kl7qr_retry_rate', 'test_n0bbs_retry_rate', 'test_w5jab_receive_recovery') else '2300'
    hs, io = station(high=False, bw=bw)
    if source == 'test_two_over_window':
        from .test_two_over_window import _window, _stream, _over, _lead
        hs, io = station(role='initiator', high=False)
        a = b'A' * 89
        b = a if scenario == 'test_identical_blocks_in_one_window_keep_their_multiplicity' else b'B' * 89
        window = _window(a, b)
        streamed = scenario != 'test_the_repeat_of_a_two_over_window_is_not_delivered_twice'
        if scenario == 'test_a_refused_missing_block_nak_is_retained_for_the_next_gap':
            _stream(hs, _over(a, 0), _lead(_over(b, 1)), trail=0.0)
            _stream(hs, *[_over(b'C' * 89, 2)[::-1]] * 3, trail=0.0)
            assert hs._owed_block and not io.host
            io.accept = False
            for _ in range(2):
                assert not hs._reack()
                assert not hs._rx_window.A.sent and hs._rx_window.T is None
            io.accept = True
            assert hs._reack()
            assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
        for replay in (False, True):
            before = b''.join(io.host)
            keys = len(io.sent)
            if streamed:
                split = len(_over(a, 0)) + len(_lead(_over(b, 1)))
                _stream(hs, window[:split], trail=0.0)
                assert hs._held_answer is not None
                assert len(hs._rx_window.W.frames) == 1
                assert b''.join(io.host) == before and len(io.sent) == keys
                _stream(hs, window[split:])
            else:
                hs.on_rx_audio(window)
            assert hs._rx_window.W is None and hs._held_answer is None
            assert len(hs._rx_window.K.frames) == 2
            assert [f.body[:-1] for f in hs._rx_window.K.frames] == [a, b]
            assert hs._rx_window.T is None
            assert hs._rx_window.A.kind == 'continue' and hs._rx_window.A.sent
            assert len(io.sent) == keys + 1
            assert b''.join(io.host) == a + b
            if not replay:
                arm_replay(hs)
        return
    if source in ('test_fit_hint', 'test_bw500_pairing'):
        level = 10 if source == 'test_fit_hint' else 1
        hs._rx_read, hs._expect_level = (level, 40.0), level
        hs._next_500 = level
        monkeypatch.setattr(A, '_irs_frame', lambda *a: 0)
        capacity = phy.body_size(bw, phy.level_key(bw, level)) - 1
        def data(payload, field=0x99):
            hs._rx_read = level, 40.0
            assert receive(monkeypatch, hs, payload + bytes([field]))
        if scenario in ('test_u4_repeated_pair_retains_window_occurrence',
                        'test_six_delivered_singles_then_pairs_until_level_change_duplicate_or_close'):
            prefix = b''
            if source == 'test_bw500_pairing':
                for c in range(6):
                    data(bytes([c]) * capacity)
                prefix = b''.join(bytes([c]) * capacity for c in range(6))
            pair = [b'A' * capacity + b'\x99', b'B' * capacity + b'\x99']
            assert receive(monkeypatch, hs, *pair)
            for _ in range(3):
                arm_replay(hs)
                assert receive(monkeypatch, hs, *pair)
            assert b''.join(io.host) == prefix + b'A' * capacity + b'B' * capacity
        elif scenario == 'test_u4_equal_payload_new_window_occurrences':
            for c in b'ABACDE':
                data(bytes([c]) * capacity)
                arm_replay(hs)
                data(bytes([c]) * capacity)
            assert b''.join(io.host) == b''.join(bytes([c]) * capacity for c in b'ABACDE')
        elif scenario == 'test_u4_idle_completed_single_retransmission_counts_once':
            for c in b'ABCDE':
                data(bytes([c]) * capacity)
            for field in (0x99, 0x19):
                arm_replay(hs)
                data(b'E' * capacity, field)
            data(b'F' * capacity)
            assert b''.join(io.host) == b''.join(bytes([c]) * capacity for c in b'ABCDEF')
        elif scenario in ('test_u4_peer_confirms_reset', 'test_u4_reply_bits_do_not_create_another_counting_window'):
            for c in b'ABCDEF':
                data(bytes([c]) * capacity)
            arm_replay(hs)
            data(b'F' * capacity)
            assert b''.join(io.host) == b''.join(bytes([c]) * capacity for c in b'ABCDEF')
        elif scenario == 'test_fetch_013206_retry_starts_six_singles':
            data(b'\x01' * capacity)
            data(b'\x02' * capacity)
            arm_replay(hs)
            data(b'\x02' * capacity)
            for c in range(3, 8):
                data(bytes([c]) * capacity)
            assert b''.join(io.host) == b''.join(bytes([c]) * capacity for c in range(1, 8))
        else:
            assert scenario == 'test_duplicate_read_resets_pair_run_without_redelivery'
            data(bytes([31]) * capacity)
            arm_replay(hs)
            data(bytes([31]) * capacity)
            assert b''.join(io.host) == bytes([31]) * capacity
        assert hs._rx_window.T is None
        assert hs._rx_window.A.sent
        return
    payloads = {
        'test_off_ignores_text_bit_and_preserves_duplicate_handling': b'\xffplain\x00\xfe',
        'test_identical_message_after_completed_handover_is_new_data': b'same message\r',
        'test_asking_for_turn_does_not_forget_unconfirmed_peer_delivery': b'repeat',
        'test_partial_window_recovery_delivers_only_missing_suffix_once': b'A' * 89 + b'B' * 89,
        'test_a_close_resent_at_a_lower_record_is_not_delivered_twice': b'FS +\r',
        'test_reply_at_previously_requested_rate_is_delivered_once_after_extra_nak': b'recovered mail',
        'test_a_repeated_over_is_answered_but_delivered_once': b'once only\r',
        'test_a_repeated_greeting_draws_the_per_over_response_not_a_second_request': b'gateway greeting >\r',
        'test_caller_nak_routes_name_the_missing_or_saved_over_through_refusals': bytes(range(43)),
        'test_a_repeated_full_over_repeats_the_continue_answer': b'A' * 89,
        'test_good_resend_delivers_once_then_queries_repeat_ack_not_nak': b'hi',
        'test_lost_final_behind_deferred_ask_is_delivered_once': b'last over',
    }
    payload = payloads.get(scenario)
    if source in ('test_kl7qr_retry_rate', 'test_n0bbs_retry_rate'):
        import importlib
        original = importlib.import_module('hfmodem.tests.kestrel.' + source)
        from .test_idle_poll import _Clock
        monkeypatch.setattr(A.time, 'monotonic', _Clock.monotonic)
        captured, _ = original.station()
        bodies = captured._peer_data_over(original.recording())
        payload = b''.join(phy.vara_payload(b, caller=captured.caller, body_len=len(b)) for b in bodies)
        assert payload
    if scenario == 'test_a_repeated_over_is_traffic_and_not_progress':
        from .test_vara_giveup import _greeting
        payload = phy.vara_payload(_greeting(), caller=hs.caller)
    assert payload is not None
    anchor = payload + b'\x99'
    if source == 'test_compression':
        anchor = anchor[:-1] + bytes([anchor[-1] | C.TEXT_FLAG])
    assert receive(monkeypatch, hs, anchor)
    assert b''.join(io.host) == payload
    arm_replay(hs)
    assert receive(monkeypatch, hs, anchor)
    assert hs._rx_window.T is None
    assert b''.join(io.host) == payload
    assert hs._rx_window.A.kind == 'continue' and hs._rx_window.A.sent


def test_floor_tape_late_idle_naks_and_contract_retry_fills_offset_89(monkeypatch):
    import json
    from pathlib import Path
    from hfmodem.kestrel.tx import varahf2300_tx as TX
    fixture = json.loads((Path(__file__).with_name('fixtures') / 'floor-tape-run1.json').read_text())
    l4, l3, l2 = [bytes.fromhex(row['raw']) for row in fixture['frames']]
    for raw in (l4, l3, l2):
        assert VF.crc16_genibus(raw[:-2]) == int.from_bytes(raw[-2:], 'big')
        assert raw[-3] == 0x99
    prbs = l4[:-3] + l3[:-3] + l2[:-3]
    assert len(prbs) == 158
    hs, io = station(role='initiator', high=False)
    hs.caller, hs.called = 'WI9YZA', 'W9SSJ'
    now = [24.972]
    monkeypatch.setattr(A.time, 'monotonic', lambda: now[0])
    hs._level_read = phy.LevelRead(4, (l4[:-2],), 40.0, 1)
    assert hs._answer_data_over(np.zeros(100))
    hs._level_read = None
    assert hs._rx_window.R == prbs[:89] and hs._rx_window.A.sent
    now[0] = hs._unkeyed_at + .014 + 1.395
    hs.on_rx_audio(MK.synth_burst(hs.called, VF.SESSION_RESPONDER_IDLE))
    assert hs._rx_window.A.kind == 'continue' and hs._rx_window.A.sent
    assert hs._rx_window.T is None
    now[0] = hs._unkeyed_at + 6.14 + 1.4
    hs.on_rx_audio(MK.synth_burst(hs.called, VF.SESSION_RESPONDER_OVER_IDLE))
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    np.testing.assert_array_equal(io.sent[-1], MK.synth_tone_pairs(VF.nak(hs.called, hs.bw)[0]))
    assert (hs._rx_window.T.start, hs._rx_window.T.frontier) == (0, 89)
    assert b''.join(io.host) == prbs[:89]
    retry_body = prbs[89:111] + l2[-3:-2]
    retry = TX.build_frame(retry_body, 1)
    assert len(retry) == len(l2)
    hs._level_read = phy.LevelRead(2, (retry[:-2],), 40.0, 1)
    assert hs._answer_data_over(np.zeros(100))
    hs._level_read = None
    assert (hs._rx_window.K.start, hs._rx_window.K.end) == (89, 111)
    assert hs._rx_window.T is None
    assert hs._rx_window.R == prbs[:111]
    assert b''.join(io.host) == prbs[:111]


def feed_envelope_stream(hs, io, audio, now):
    before = len(io.sent)
    for start in range(0, len(audio), 1024):
        block = audio[start:start + 1024]
        now[0] += len(block) / MK.FS
        hs.on_rx_stream(block)
        if len(io.sent) > before:
            return start + len(block)
    return len(audio)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('high', [False, True], ids=['limited', 'high-speed'])
@pytest.mark.parametrize('reset', ['turn', 'session', 'disconnect'])
def test_certain_single_envelope_stream_naks_without_lesson(monkeypatch, bw, high, reset):
    from hfmodem.kestrel.tx import varahf2300_tx as TX
    now = [100.0]
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    hs, io = station(role='initiator', high=high, bw=bw)
    hs._rx_read, hs._expect_level = (4, None), 4
    hs._holds_epoch = hs._holds_any = 6
    assert hs._peer_double_verdict() == 'CERTAIN'
    key = phy.level_key(bw, 4)
    size = phy.body_size(bw, key)
    body = full(bytes([31]), size=size, field=0x99)
    audio = np.pad(TX.synth_frame(TX.build_frame(body, key), key, over=7),
                   (MK.FS // 4, 7 * MK.FS))
    assert phy.burst_extent(audio, bw, 4) == 'single'
    feed_envelope_stream(hs, io, audio, now)
    assert hs._rx_window.W is None
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert b''.join(io.host) == b''
    assert not hs._rx_window.R and hs._rx_window.K is None
    assert not hs._rx_window.no_doubles
    hs._holds_epoch = hs._holds_any = 12
    assert hs._peer_double_verdict() != 'NONE'
    assert len(io.sent) == 1
    if reset == 'disconnect':
        hs.disconnect()
    else:
        hs._perform_rx_actions(hs._rx_window.event('T', new_session=reset == 'session'))
    hs._reset_peer_holds(confirmed=True)
    hs._holds_epoch = hs._holds_any = 6
    assert not hs._rx_window.no_doubles
    assert hs._peer_double_verdict() == 'CERTAIN'


@pytest.mark.parametrize('extent', ['double', 'inconclusive', 'located-hole'])
def test_certain_missing_companion_stream_naks_without_lesson(monkeypatch, extent):
    from hfmodem.kestrel.tx import varahf2300_tx as TX
    now = [100.0]
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    hs, io = station(role='initiator', high=False)
    hs._holds_epoch = hs._holds_any = 6
    head = full(b'head', field=0x99)
    tail = full(b'tail', field=0x99)
    first = TX.synth_frame(TX.build_frame(head, 3), 3, over=7)
    second = TX.synth_frame(TX.build_frame(tail, 3), 3, over=None)
    if extent == 'inconclusive':
        second = second[:len(second) // 2]
    audio = np.pad(np.concatenate([first, second]), (MK.FS // 4, 8 * MK.FS))
    assert phy.burst_extent(audio, hs.bw, 4) == ('inconclusive' if extent == 'inconclusive' else 'double')
    decode = hs._peer_data_over
    def erase_companion(samples):
        if hs._rx_window.W is not None:
            return [None] if extent == 'located-hole' else []
        return decode(samples)[:1]
    monkeypatch.setattr(hs, '_peer_data_over', erase_companion)
    end = feed_envelope_stream(hs, io, audio, now)
    assert end >= MK.FS // 4 + len(first) + len(second)
    assert hs._rx_window.W is None
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert not io.host and not hs._rx_window.R and hs._rx_window.K is None
    assert not hs._rx_window.no_doubles
    hs._reset_peer_holds(confirmed=True)
    hs._holds_epoch = hs._holds_any = 6
    assert hs._peer_double_verdict() == 'CERTAIN'


def test_peer_i2_double_stream_erased_companion_naks(monkeypatch):
    from pathlib import Path
    from scipy.io import wavfile
    path = Path(os.environ.get('KESTREL_I2_CAPTURE', 'chat-ks-hold2-800-1__b2a.wav'))
    if not path.exists():
        pytest.skip('DOUBLE-LAYOUT I2 capture unavailable')
    rate, tape = wavfile.read(path, mmap=True)
    assert rate == MK.FS
    audio = np.asarray(tape[3698015:4199987], dtype=float)
    if np.issubdtype(tape.dtype, np.integer):
        audio /= np.iinfo(tape.dtype).max
    assert phy.burst_extent(audio, '2300', 2) == 'double'
    now = [100.0]
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    hs, io = station(role='initiator', high=False)
    hs.caller, hs.called = 'WI9YZA', 'W9SSJ'
    hs._rx_read, hs._expect_level = (2, None), 2
    hs._holds_epoch = hs._holds_any = 6
    decode = hs._peer_data_over
    seen = []
    def erase_companion(samples):
        if hs._rx_window.W is not None:
            return []
        bodies = decode(samples)
        seen.extend(b for b in bodies[:1] if isinstance(b, bytes))
        return bodies[:1]
    monkeypatch.setattr(hs, '_peer_data_over', erase_companion)
    feed_envelope_stream(hs, io, np.pad(audio, (0, 8 * MK.FS)), now)
    assert len(seen) == 1 and len(seen[0]) == 23
    assert hs._rx_window.W is None
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert not io.host and not hs._rx_window.R and hs._rx_window.K is None
    assert not hs._rx_window.no_doubles


@pytest.mark.parametrize('envelope', ['single', 'double', 'inconclusive'])
@pytest.mark.parametrize('shape', ['head', 'short-head', 'terminal-tail', 'hole', 'keying', 'ceiling'])
def test_certain_end_envelope_never_commits_or_teaches(envelope, shape):
    machine = RW.ReceiveWindow()
    frame = RW.Frame(b'head', full=shape != 'short-head', last=shape == 'terminal-tail')
    machine.event('F', frame=frame, verdict='CERTAIN', level=4, envelope=envelope)
    if shape == 'hole':
        machine.event('X')
    actions = machine.event('C' if shape == 'ceiling' else 'E', single=True,
                            peer_keying=shape == 'keying')
    assert not machine.no_doubles
    assert not any(isinstance(a, (RW.Ready, RW.Emit)) for a in actions)
    assert machine.A.kind == 'nak'
    assert not machine.R and machine.K is None
