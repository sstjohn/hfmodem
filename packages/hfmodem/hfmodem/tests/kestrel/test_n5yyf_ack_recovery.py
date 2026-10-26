# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock 4.9.0: recover a lost final ACK before acquiring the transmit turn."""
from pathlib import Path
import wave
import numpy as np
import pytest
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_data_over_gate import _connected
from .test_responder_control_stream import feed
from .test_idle_poll import clock, _Clock  # noqa: F401


def audio(name):
    p = Path(__file__).parent / 'fixtures/n5yyf-ack-recovery-1025' / (name + '.wav')
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float)/32768


def station():
    h, io = _connected(bw='500')
    h.chat = True
    h.caller, h.called = 'W9SSJ', 'N5YYF'
    h.turn = A._TURN_PEER
    h._answer_owed = A._OWED_RELEASE
    h._txq = [b'25 <QSYF>000014103500</QSYF>']
    h._unkeyed_at = _Clock.t
    return h, io


def assert_frame(x, kind):
    call = 'W9SSJ' if kind.keyed_by == 'caller' else 'N5YYF'
    assert np.array_equal(x, M.synth_burst(call, F.for_bw(kind, '500')))


@pytest.mark.parametrize('confirmation,grant', [('confirmed-long', 'grant'),
                                                ('confirmed-control', 'grant-short'),
                                                ('confirmed-idle-short', 'grant')])
@pytest.mark.parametrize('block', [512, 4096])
def test_recorded_recovery_confirms_ack_then_requests_then_sends(confirmation, grant, block):
    h, io = station()
    progress = h.progress
    feed(h, audio('idle'), block)
    assert len(io.sent) == 1, io.msgs
    assert_frame(io.sent[-1], F.SESSION_TURN_IDLE)
    assert h._receive_ack_recovery == 'ack' and h._asked == 0
    assert h._tx_pending is None and h.progress == progress
    feed(h, audio('idle'), block)
    assert len(io.sent) == 2
    assert_frame(io.sent[-1], F.SESSION_TURN_IDLE)
    assert h.progress == progress and h._tx_pending is None
    feed(h, audio(confirmation), block)
    assert len(io.sent) == 3, io.msgs
    assert_frame(io.sent[-1], F.SESSION_TURN_REQUEST)
    assert h._receive_ack_recovery == 'request'
    assert h.turn == A._TURN_ASKED and h._tx_pending is None
    feed(h, np.r_[audio(grant), np.zeros(12000)], block)
    assert h.turn == A._TURN_OURS and h._tx_pending is not None, io.msgs
    assert h._receive_ack_recovery is None


def test_refused_ack_does_not_claim_peer_confirmation_or_send_data(monkeypatch):
    h, io = station()
    monkeypatch.setattr(h, '_send_burst', lambda *a: False)
    assert not h._ask_in_gap()
    assert h._receive_ack_recovery is None and h._asked == 0
    assert h._answer_owed == A._OWED_RELEASE and h._txq


def test_retry_cannot_skip_pending_ack_confirmation():
    h, io = station()
    assert h._ask_in_gap()
    assert h._tx_turn_request()
    assert_frame(io.sent[-1], F.SESSION_TURN_IDLE)
    assert h._asked == 0 and h._tx_pending is None
    h._owe_nothing()
    assert h._receive_ack_recovery is None


@pytest.mark.parametrize('case', ['unread', 'nonchat', 'no-final-ack'])
def test_only_completed_chat_receive_delivery_enters_new_recovery(case):
    h, io = station()
    if case == 'unread': h._answer_owed, h._owed_block = A._OWED_OVER, True
    if case == 'nonchat': h.chat = False
    if case == 'no-final-ack': h._answer_owed = None
    assert not h._needs_final_ack_recovery()


@pytest.mark.parametrize('confirmation', ['confirmed-long', 'confirmed-control',
                                         'confirmed-idle-short'])
def test_confirmation_requires_expected_peer_and_does_not_grant_turn(confirmation):
    h, io = station()
    h._ask_in_gap()
    h.called = 'KQ4QWH'
    feed(h, audio(confirmation), 512)
    assert h._receive_ack_recovery == 'ack'
    assert h._tx_pending is None and len(io.sent) == 1
    assert not any('invariant failure' in m for m in io.msgs)


def test_duplicate_confirmation_is_not_progress_or_permission_to_send():
    h, io = station()
    h._ask_in_gap()
    feed(h, audio('confirmed-long'), 512)
    progress, sent = h.progress, len(io.sent)
    feed(h, audio('confirmed-long'), 512)
    h._took_idle_response()  # also the bracket route
    assert h.progress == progress and len(io.sent) == sent
    assert h._receive_ack_recovery == 'request' and h._tx_pending is None


def test_refused_request_preserves_confirmed_state_for_long_retry(monkeypatch):
    h, io = station()
    h._ask_in_gap()
    with monkeypatch.context() as m:
        m.setattr(h, '_send_burst', lambda *a: False)
        h._took_idle_response()
    assert h._receive_ack_recovery == 'request' and h._answer_owed is None
    assert h._tx_turn_request()
    assert_frame(io.sent[-1], F.SESSION_TURN_REQUEST)
    assert h._tx_pending is None


@pytest.mark.parametrize('confirmation', ['confirmed-long', 'confirmed-idle-short',
                                         'confirmed-control'])
def test_partial_confirmation_never_keys_across_its_tail(confirmation):
    h, io = station()
    h._ask_in_gap()
    x = audio(confirmation)
    cut = len(x) // 2
    progress = h.progress
    feed(h, x[:cut], 512)
    assert len(io.sent) == 1 and h._receive_ack_recovery == 'ack'
    assert h.progress == progress and h._tx_pending is None
    feed(h, x[cut:], 512)
    assert len(io.sent) == 2 and h._receive_ack_recovery == 'request', io.msgs
    assert not any('invariant failure' in m for m in io.msgs)
