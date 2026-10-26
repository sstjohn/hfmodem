# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Lost retry/poll: bounded recovery without acknowledging unread DATA."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.tx import varahf500_tx as TX
from .test_idle_poll import clock as clock, _Clock as _Clock, _station as _station
from .test_responder_control_stream import feed


def waiting():
    h, io = _station('500')
    h.called = 'KQ4QWH'
    h.chat = True
    h.turn = A._TURN_PEER
    h._link_high_speed = True
    h._start_compression('OFF')
    h._progressed()
    h._expect_level = 5
    h._irs_nak(phy.LevelRead(5, (), None, 1))
    assert h._expect_level == 4
    return h, io


def quiet(h, seconds=13):
    feed(h, np.random.default_rng(42).normal(0, 1e-5, int(48000 * seconds)))


def notices(io, state):
    return [m for m in io.msgs if m.startswith(f'link-recovery {state}:')]


def test_quiet_retry_is_solicited_twice_without_resetting_progress_or_credit():
    h, io = waiting()
    progress = h.progress
    h.send(b'hello')
    assert notices(io, 'queued')
    for count in (1, 2):
        quiet(h)
        before = len(io.sent)
        h.idle_keepalive()
        assert len(io.sent) == before + 1, io.msgs
        assert h._rx_silence_tries == count
        assert h.progress == progress and h.payload_ledger.pending == 5
        assert h.turn == A._TURN_PEER
    assert len(notices(io, 'requested')) == 3
    quiet(h)
    before = len(io.sent)
    h.idle_keepalive()
    assert len(io.sent) == before
    for _ in range(8):
        _Clock.t += 13
        h.idle_keepalive()
        if h.state != A.VaraState.CONNECTED:
            break
    assert h.state != A.VaraState.CONNECTED
    assert len(notices(io, 'closing')) == 1
    assert '3 NAK(s)' in notices(io, 'closing')[0]
    assert '5 outgoing bytes unacknowledged' in notices(io, 'closing')[0]


def test_reply_at_previously_requested_rate_is_delivered_once_after_extra_nak():
    h, io = waiting()
    quiet(h)
    h.idle_keepalive()
    assert h._rx_silence_tries == 1 and h._expect_level == 1
    body = phy.vara_body(b'recovered mail', h.caller, body_len=44)
    packet = TX.synth_burst(TX.build_frame(body[:-1], body[-1]))
    feed(h, np.concatenate([np.zeros(4800), packet, np.zeros(24000)]))
    assert b''.join(io.host) == b'recovered mail', io.msgs
    assert len(notices(io, 'received')) == 1
    assert 'level 4' in notices(io, 'received')[0]
    assert h._rx_resend is None
    assert h._rx_silence_tries == 0
    feed(h, np.concatenate([np.zeros(4800), packet, np.zeros(24000)]))
    assert b''.join(io.host) == b'recovered mail'
    assert len(notices(io, 'received')) == 1
    before = len(notices(io, 'requested'))
    quiet(h)
    h.idle_keepalive()
    assert len(notices(io, 'requested')) == before


@pytest.mark.parametrize('guard', ['short-wait', 'busy', 'band-busy', 'lagging-stream',
    'pending-candidate', 'held-answer', 'own-turn', 'not-chat', 'no-stream', 'closed', 'awaiting-end'])
def test_retry_never_interrupts_receive_or_uses_another_context(guard):
    h, io = waiting()
    quiet(h, 1 if guard == 'short-wait' else 13)
    if guard == 'busy': io.receiving = True
    elif guard == 'band-busy': h._band_busy = lambda: True
    elif guard == 'lagging-stream': _Clock.t += 20
    elif guard == 'pending-candidate': h._base_rx = SimpleNamespace(pending=[object()])
    elif guard == 'held-answer': h._held_answer = object()
    elif guard == 'own-turn': h.turn = A._TURN_OURS
    elif guard == 'not-chat': h.chat = False
    elif guard == 'no-stream': h._stream_owns_over = False
    elif guard == 'closed': h.state = A.VaraState.DISCONNECTED
    elif guard == 'awaiting-end': h._lv_await_end = True
    before = (len(io.sent), h._rx_resend, h._expect_level, h._since_progress)
    assert not h._retry_receive_on_silence()
    assert (len(io.sent), h._rx_resend, h._expect_level, h._since_progress) == before


def test_refused_recovery_transmission_spends_neither_attempt_nor_rate():
    h, io = waiting()
    quiet(h)
    io.tx_went_out = lambda: False
    before = (h._rx_resend, h._expect_level, h._recovery_naks, h._since_progress)
    assert not h._retry_receive_on_silence()
    assert (h._rx_resend, h._expect_level, h._recovery_naks, h._since_progress) == before
    assert h._rx_silence_tries == 0 and len(notices(io, 'requested')) == 1


@pytest.mark.parametrize('case', ['unrequested-rate', 'our-echo', 'our-turn', 'incomplete', 'bad-crc'])
def test_earlier_rate_acceptance_still_requires_a_complete_clean_owned_reply(case):
    h, io = waiting()
    quiet(h)
    h.idle_keepalive()
    body = phy.vara_body(b'recovered mail', h.caller, body_len=44)
    packet = TX.synth_burst(TX.build_frame(body[:-1], body[-1]))
    decoded = A._RX500.decode_stream(packet)
    assert decoded.complete
    if case == 'unrequested-rate':
        h._rx_retry_rates.clear()
        h._rx_resend = (1, (True,))
    elif case == 'our-echo': h._keyed_bodies.add(body)
    elif case == 'our-turn': h.turn = A._TURN_OURS
    elif case == 'incomplete': decoded.observation_complete = False
    elif case == 'bad-crc': decoded.data_frames[0].crc_ok = False
    h._resend_decoded = decoded
    assert h._peer_data_over_500(packet, len(packet) / 48000) == []
    assert h._rx_resend is not None and not io.host
    assert not notices(io, 'received')


def test_new_retry_cannot_borrow_the_previous_transactions_requested_rates():
    h, _ = waiting()
    quiet(h)
    h.idle_keepalive()
    assert h._previous_base_retry_rate(4)
    h._rx_resend = None
    h._expect_level = 1
    h._advance_resend(missing=True)
    assert not h._previous_base_retry_rate(4)
