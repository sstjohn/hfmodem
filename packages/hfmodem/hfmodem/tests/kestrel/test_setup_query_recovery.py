# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fresh stock responder query controls and VE6DNL's exact on-air request."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF
from .test_idle_poll import clock as clock, _Clock as _Clock, _IO as _IO
from .test_responder_control_stream import feed

QUERY = F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]


def setup(*, chat=True, marked=False, refused=False, caller='VE6DNL', ask=None):
    io = _IO()
    hs = A.VaraStationHandshake(['W9SSJ'], io, bw='500')
    hs.role, hs.called, hs.caller = 'responder', 'W9SSJ', caller
    hs.state = A.VaraState.CONNECTING
    hs.chat = hs._chat_profile = chat
    hs.high_speed = True
    if refused:
        io.tx_went_out = lambda: False
    frame = F.link_setup_frame(caller, bw='500', ask=1, high_speed=marked)
    if ask is not None:
        # The generator uses ask 6 for marked callers; KQ4QWH actually asked 1.
        body = bytearray(frame[:-2])
        body[-2] = ask
        frame = bytes(body) + F.crc16_genibus(body).to_bytes(2, 'big')
    hs._take_link_setup(frame)
    return hs, io


def query(call='W9SSJ'):
    return M.synth_burst(call, F.for_bw(QUERY, '500'))


def padded(x):
    return np.concatenate([np.zeros(4800), x, np.zeros(24000)])


@pytest.mark.parametrize('marked', [False, True])
@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('queued', [False, True])
@pytest.mark.parametrize('block', [960, 4096, 24000])
def test_complete_setup_query_gets_stock_idle_without_payload_credit(native, queued, block, marked):
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


@pytest.mark.parametrize('case', ['wrong-call', 'partial', 'query-b', 'short-query', 'stale',
                                  'chat-off', 'refused-ack', 'progress', 'keepalive', 'release'])
@pytest.mark.parametrize('marked', [False, True])
def test_query_recovery_is_only_for_fresh_actual_chat_setup(case, marked):
    hs, io = setup(chat=case != 'chat-off', marked=marked, refused=case == 'refused-ack')
    x = query('W1AW' if case == 'wrong-call' else 'W9SSJ')
    if case == 'partial':
        x = x[:-4*M.HOP]
    elif case == 'query-b':
        x = M.synth_burst('W9SSJ', F.for_bw(F.SESSION_INTERMEDIATE_ANSWER_QUERIES[1], '500'))
    elif case == 'short-query':
        x = M.synth_burst('W9SSJ', F.for_bw(F.SESSION_ANSWER_QUERIES_SHORT[0], '500'))
    elif case == 'stale':
        _Clock.t += A._MAX_WITHOUT_PROGRESS * A._IDLE_POLL_S
    elif case == 'progress':
        hs._deliver_block(A._phy.vara_body(b'peer payload', 'VE6DNL', body_len=44))
        assert io.host == [b'peer payload']
    elif case == 'release':
        feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_TURN_RELEASE, '500'))))
    elif case == 'keepalive':
        feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_KEEPALIVE_B, '500'))))
    count = len(io.sent)
    feed(hs, padded(x))
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
    called = []
    monkeypatch.setattr(hs, '_reanswer_over', lambda: called.append(True))
    hs._answer_responder_control(QUERY)
    assert called == [True] and len(io.sent) == 1


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
def test_kq4qwh_unfiltered_setup_query_with_adjacent_channel(block, ask):
    hs, io = setup(marked=True, caller='KQ4QWH', ask=ask)
    hs.send(b'hello')
    before = (hs.progress, hs.turn, hs._expect_level, hs._lv_owed, hs._lv_armed)
    fs, x = wavfile.read(Path(__file__).with_name('fixtures') / 'kq4qwh-1025-setup-query.wav')
    assert fs == M.FS
    x = x.astype(float) / 32768
    # Stop before the final symbols: no reply may be inferred from the prefix.
    feed(hs, x[:65000], block)
    assert len(io.sent) == 1
    feed(hs, x[65000:], block)
    assert len(io.sent) == 2, io.msgs
    assert np.array_equal(io.sent[-1], M.synth_burst(
        'KQ4QWH', F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, '500')))
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
def test_kq4qwh_recording_does_not_bypass_setup_context(case):
    hs, io = setup(marked=True, caller='KQ4QWH', ask=1)
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
    fs, x = wavfile.read(Path(__file__).with_name('fixtures') / 'kq4qwh-1025-setup-query.wav')
    assert fs == M.FS
    feed(hs, x.astype(float) / 32768)
    assert len(io.sent) == 1
    assert hs.payload_ledger.pending == 5 and hs.payload_ledger.confirmed == 0
