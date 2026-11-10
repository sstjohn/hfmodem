# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fresh VARA HF responder query controls and VE6DNL's exact on-air request."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF
from hfmodem.kestrel.arq import rxwindow as RW
from .test_idle_poll import clock as clock, _Clock as _Clock, _IO as _IO
from .test_responder_control_stream import feed

QUERY = F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]


def setup(*, bw='500', chat=True, marked=False, refused=False, caller='VE6DNL', ask=None):
    io = _IO()
    hs = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    hs.role, hs.called, hs.caller = 'responder', 'W9SSJ', caller
    hs.state = A.VaraState.CONNECTING
    hs.chat = hs._chat_profile = chat
    hs.high_speed = True
    if refused:
        io.tx_went_out = lambda: False
    frame = F.link_setup_frame(caller, bw=bw, ask=1, high_speed=marked)
    if ask is not None:
        # The generator uses ask 6 for marked callers; N0BBS actually asked 1.
        body = bytearray(frame[:-2])
        body[-2] = ask
        frame = bytes(body) + F.crc16_genibus(body).to_bytes(2, 'big')
    hs._take_link_setup(frame)
    return hs, io


def query(call='W9SSJ', bw='500'):
    return M.synth_burst(call, F.for_bw(QUERY, bw))


def padded(x):
    return np.concatenate([np.zeros(4800), x, np.zeros(24000)])


@pytest.mark.parametrize('marked', [False, True])
@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('queued', [False, True])
@pytest.mark.parametrize('block', [960, 4096, 24000])
def test_complete_setup_query_gets_peer_idle_without_payload_credit(native, queued, block, marked):
    hs, io = setup(marked=marked)
    if queued:
        hs.send(b'15 de W9SSJ <R-02>12 <LOC:EN63bb>')
    before = (hs.payload_ledger.pending, hs.payload_ledger.confirmed, hs.progress,
              hs._expect_level, hs._lv_owed, hs._lv_armed)
    if native:
        fs, x = wavfile.read(Path(__file__).with_name('fixtures') / 've6dnl-1024-query.wav')
        assert fs == M.FS
        x = x.astype(float) / 32768
    else:
        x = padded(query())
    feed(hs, x, block)
    assert len(io.sent) == 2, io.msgs
    assert np.array_equal(io.sent[-1], M.synth_burst('VE6DNL', F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, '500')))
    assert hs.turn == A._TURN_PEER and hs._tx_pending is None
    assert (hs.payload_ledger.pending, hs.payload_ledger.confirmed, hs.progress,
            hs._expect_level, hs._lv_owed, hs._lv_armed) == before
    hs.on_rx_audio(x)
    assert len(io.sent) == 2  # stream owns this frame


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
@pytest.mark.parametrize('block', [960, 4096, 24000])
def test_lost_connected_ack_query_gets_idle_at_every_bandwidth(bw, chat, block):
    hs, io = setup(bw=bw, chat=chat)
    assert len(io.sent) == 1 and hs.state == A.VaraState.CONNECTED
    hs.send(b'queued greeting')
    before = (hs.payload_ledger.pending, hs.payload_ledger.confirmed, hs.progress,
              hs._expect_level, hs._lv_owed, hs._lv_armed)
    short = M.synth_burst(hs.called, F.for_bw(F.SESSION_ANSWER_QUERIES_SHORT[0], bw))
    feed(hs, padded(short), block)
    assert len(io.sent) == 1, io.msgs
    idle = M.synth_burst(hs.caller, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, bw))
    for count in (2, 3):
        feed(hs, padded(query(bw=bw)), block)
        assert len(io.sent) == count, io.msgs
        assert np.array_equal(io.sent[-1], idle)
        assert hs.turn == A._TURN_PEER and hs._tx_pending is None
        assert hs.state == A.VaraState.CONNECTED
        assert (hs.payload_ledger.pending, hs.payload_ledger.confirmed, hs.progress,
                hs._expect_level, hs._lv_owed, hs._lv_armed) == before


@pytest.mark.parametrize('bw,case', [
    ('500', case) for case in ('wrong-call', 'partial', 'query-b', 'short-query', 'stale',
                              'refused-ack', 'progress', 'keepalive', 'release')
] + [
    ('2300', case) for case in ('wrong-call', 'partial', 'query-b', 'short-query', 'stale',
                               'refused-ack')
])
@pytest.mark.parametrize('marked', [False, True])
def test_query_recovery_is_only_for_a_fresh_actual_setup(bw, case, marked):
    hs, io = setup(bw=bw, marked=marked, refused=case == 'refused-ack')
    x = query('W1AW' if case == 'wrong-call' else 'W9SSJ', bw=bw)
    if case == 'partial':
        x = x[:-4*M.HOP]
    elif case == 'query-b':
        x = M.synth_burst('W9SSJ', F.for_bw(F.SESSION_INTERMEDIATE_ANSWER_QUERIES[1], bw))
    elif case == 'short-query':
        x = M.synth_burst('W9SSJ', F.for_bw(F.SESSION_ANSWER_QUERIES_SHORT[0], bw))
    elif case == 'stale':
        _Clock.t += A._MAX_WITHOUT_PROGRESS * A._IDLE_POLL_S
    elif case == 'progress':
        hs._deliver([A._phy.vara_body(b'peer payload', 'VE6DNL', body_len=44)])
        hs._finish_delivery_window()
        assert io.host == [b'peer payload']
    elif case == 'release':
        feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_TURN_RELEASE, bw))))
    elif case == 'keepalive':
        feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_KEEPALIVE_B, bw))))
    count = len(io.sent)
    previous = io.sent[-1].copy()
    feed(hs, padded(x))
    if case == 'progress':
        assert len(io.sent) == count + 1, io.msgs
        np.testing.assert_array_equal(io.sent[-1], previous)
        assert hs._rx_window.A.kind == 'final' and io.host == [b'peer payload']
    else:
        assert len(io.sent) == count, io.msgs


def test_query_never_keys_before_the_complete_tail():
    hs, io = setup()
    x = query()
    feed(hs, x[:-M.HOP])
    assert len(io.sent) == 1
    feed(hs, np.concatenate([x[-M.HOP:], np.zeros(24000)]))
    assert len(io.sent) == 2


def test_refused_idle_preserves_obligation_and_queued_ledger():
    hs, io = setup()
    hs.send(b'queued greeting')
    progress = hs.progress
    io.tx_went_out = lambda: False
    feed(hs, padded(query()))
    assert hs.payload_ledger.pending == 15 and hs.payload_ledger.confirmed == 0
    assert hs._setup_answer_for is not None and hs.progress == progress
    io.tx_went_out = lambda: True
    feed(hs, padded(query()))
    assert len(io.sent) == 3, io.msgs
    assert hs.payload_ledger.pending == 15 and hs._tx_pending is None


def test_setup_query_cannot_replace_existing_data_answer_debt(monkeypatch):
    hs, io = setup()
    hs._answer_owed = A._OWED_OVER
    hs._rx_window.A = RW.Answer(1, 0, 'continue',
        original=A.OVER_CONTINUE_GENERATED, sent=True)
    hs._rx_window.alt = 1
    hs._answer_responder_control(QUERY)
    assert len(io.sent) == 2 and not io.host
    assert hs._rx_window.A.kind == 'continue' and hs._rx_window.A.sent
    assert len(io.sent[-1]) == A._span(F.SESSION_OVER_RESPONSE)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('chat', [False, True])
def test_setup_query_ends_decoded_open_window_before_answering(bw, route, chat):
    hs, io = setup(bw=bw, chat=chat)

    def receive_query():
        audio = padded(query(bw=bw))
        if route == 'stream':
            feed(hs, audio)
        else:
            _Clock.t += len(audio) / M.FS
            hs.on_rx_audio(audio)

    receive_query()
    assert len(io.sent) == 2 and hs._rx_window.before_data
    np.testing.assert_array_equal(io.sent[-1], M.synth_burst(
        hs.caller, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, bw)))
    # Decode a real full first frame with the answer held, as when the
    # transport still hears the peer keying. No injected G or DATA verdict.
    level = A._phy.level_key(bw, 4)
    payload = bytes(range(A._phy.body_size(bw, level) - 1))
    body = payload + b'\x81'
    assert hs._answer_data_over(OF.data_over_tx(body, bw=bw, level=level), hold=True)
    assert hs._rx_window.W is not None and hs._held_answer is not None
    assert hs._rx_window.W.frames[0].body == body
    assert hs._answer_owed is None and not io.host and len(io.sent) == 2
    receive_query()
    assert b''.join(io.host) == payload  # BW500: all 43 host bytes.
    assert hs._rx_window.W is None and hs._held_answer is None
    assert hs._rx_window.A.kind == 'continue' and hs._rx_window.A.sent
    assert len(io.sent) == 3
    assert not np.array_equal(io.sent[-1], M.synth_burst(
        hs.caller, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, bw)))


@pytest.mark.parametrize('opening', [True, False])
@pytest.mark.parametrize('marked', [False, True])
def test_a_peer_chat_caller_gets_the_greeting(opening, marked):
    """A VARA HF CHAT caller keys no confirm: VE6DNL's setup query, answered with
    the VARA HF idle even with a greeting queued, then either its own opening
    over, whose answer says traffic and draws its release (N0BBS 2026-10-25,
    our away greeting went on that release), or its keepalive while it holds
    the turn idle, answered the same way."""
    hs, io = setup(marked=marked)
    greeting = b'<AWQ>Away, please leave a VMail'
    hs.send(greeting)
    fs, x = wavfile.read(Path(__file__).with_name('fixtures') / 've6dnl-1024-query.wav')
    feed(hs, x.astype(float) / 32768)
    assert hs._tx_pending is None
    if opening:
        level = A._phy.level_key('500', hs._expect_level)
        body = A._phy.vara_body(b'73', 'VE6DNL', body_len=A._phy.body_size('500', level))
        feed(hs, padded(OF.data_over_tx(body, bw='500', level=level)))
        assert io.host == [b'73'], io.msgs
    else:
        feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_KEEPALIVE_B, '500'))))
    traffic = M.synth_tone_pairs(F.control_bursts('VE6DNL', '500', 0)[1])
    assert np.array_equal(io.sent[-1], traffic), io.msgs
    feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_TURN_RELEASE, '500'))))
    assert hs._tx_pending is not None, io.msgs


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('handover', [F.SESSION_TURN_RELEASE, A._CONFIRM_LONG[4]],
                         ids=['release', 'confirm-32'])
def test_non_chat_caller_keepalive_after_setup_query_draws_traffic(bw, handover):
    hs, io = setup(bw=bw, chat=False)
    hs.send(b'<AWQ>Away, please leave a VMail')
    feed(hs, padded(query(bw=bw)))
    assert len(io.sent) == 2, io.msgs
    idle = M.synth_burst(hs.caller, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, bw))
    assert np.array_equal(io.sent[-1], idle)
    assert hs.turn == A._TURN_PEER and hs._tx_pending is None
    keepalive = M.synth_burst('W9SSJ', F.for_bw(F.SESSION_KEEPALIVE_B, bw))
    traffic = M.synth_tone_pairs(F.control_bursts(hs.caller, bw, 0)[1])
    for count in (3, 4):
        feed(hs, padded(keepalive))
        assert len(io.sent) == count, io.msgs
        assert np.array_equal(io.sent[-1], traffic), io.msgs
        assert hs.turn == A._TURN_ASKED and hs._tx_pending is None
    feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(handover, bw))))
    assert hs._tx_pending is not None, io.msgs


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_non_chat_caller_keepalive_after_setup_query_without_traffic_draws_idle(bw):
    hs, io = setup(bw=bw, chat=False)
    feed(hs, padded(query(bw=bw)))
    assert len(io.sent) == 2 and hs._tx_pending is None, io.msgs
    turn = hs.turn
    feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_KEEPALIVE_B, bw))))
    assert len(io.sent) == 3, io.msgs
    idle = M.synth_tone_pairs(F.control_bursts(hs.caller, bw, 1)[1])
    assert np.array_equal(io.sent[-1], idle), io.msgs
    assert hs.turn == turn and hs._tx_pending is None


def test_peer_data_after_recovery_is_received_and_retires_setup_context():
    hs, io = setup()
    hs.send(b'local greeting')
    feed(hs, padded(query()))
    body = A._phy.vara_body(b'peer message', 'VE6DNL', body_len=44)
    feed(hs, padded(OF.data_over_tx(body, bw='500', level=4)))
    assert io.host == [b'peer message'], io.msgs
    assert hs._setup_answer_for is None
    assert hs.payload_ledger.pending == len(b'local greeting')


@pytest.mark.parametrize('block', [960, 4096, 24000])
@pytest.mark.parametrize('ask', [1, 6])
def test_n0bbs_unfiltered_setup_query_with_adjacent_channel(block, ask):
    hs, io = setup(marked=True, caller='N0BBS', ask=ask)
    hs.send(b'hello')
    before = (hs.progress, hs.turn, hs._expect_level, hs._lv_owed, hs._lv_armed)
    fs, x = wavfile.read(Path(__file__).with_name('fixtures') / 'n0bbs-1025-setup-query.wav')
    assert fs == M.FS
    x = x.astype(float) / 32768
    # Stop before the final symbols: no reply may be inferred from the prefix.
    feed(hs, x[:65000], block)
    assert len(io.sent) == 1
    feed(hs, x[65000:], block)
    assert len(io.sent) == 2, io.msgs
    assert np.array_equal(io.sent[-1], M.synth_burst(
        'N0BBS', F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, '500')))
    assert hs.payload_ledger.pending == 5 and hs.payload_ledger.confirmed == 0
    assert (hs.progress, hs.turn, hs._expect_level, hs._lv_owed, hs._lv_armed) == before
    assert hs._tx_pending is None and not io.host
    hs.on_rx_audio(x)
    assert len(io.sent) == 2
    # A repeated solicitation can recover a lost idle reply, but cannot keep
    # extending the no-progress deadline or settle the still-unsent greeting.
    feed(hs, x, block)
    assert len(io.sent) == 3
    assert hs.progress == before[0]
    assert hs.payload_ledger.pending == 5 and hs.payload_ledger.confirmed == 0


@pytest.mark.parametrize('case', ['wrong-call', 'expired', 'data-debt', 'released'])
def test_n0bbs_recording_does_not_bypass_setup_context(case):
    hs, io = setup(marked=True, caller='N0BBS', ask=1)
    hs.send(b'hello')
    if case == 'wrong-call':
        hs.called = 'W1AW'
        hs._setup_answer_for = (hs.caller, hs.called, hs.progress)
    elif case == 'expired':
        _Clock.t += A._MAX_WITHOUT_PROGRESS * A._IDLE_POLL_S
    elif case == 'data-debt':
        hs._answer_owed = A._OWED_OVER
    else:
        hs._setup_answer_for = None
    fs, x = wavfile.read(Path(__file__).with_name('fixtures') / 'n0bbs-1025-setup-query.wav')
    assert fs == M.FS
    feed(hs, x.astype(float) / 32768)
    assert len(io.sent) == 1
    assert hs.payload_ledger.pending == 5 and hs.payload_ledger.confirmed == 0
