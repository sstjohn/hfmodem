# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A quick poll cannot prove the preceding direct NAK went unheard."""
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A
from .test_resend_ladders_1020 import receiver
from .test_responder_control_stream import feed
from .test_idle_poll import clock  # noqa: F401


def recording():
    p = Path(__file__).parent / 'fixtures/kl7qr-retry-1025/retry.wav'
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float)/32768


def station():
    h, io = receiver('500', 6)
    h.role, h.caller, h.called = 'initiator', 'W9SSJ', 'KL7QR'
    h.turn = A._TURN_PEER
    h._peer_delivery_open = True
    h._start_compression('OFF')
    h._irs_nak(A._phy.LevelRead(6, (), None, 1))
    assert h._expect_level == 5
    h._nak_unheard = True
    assert h._nak_owed_over()
    assert h._expect_level == 4
    assert h._nak_owed_over()
    assert h._expect_level == 1
    return h, io


@pytest.mark.parametrize('block', [960, 4096])
def test_recorded_retry_after_possibly_heard_direct_nak_is_delivered_once(block):
    h, io = station()
    assert h._previous_base_retry_rate(3)
    feed(h, recording(), block)
    delivered = b''.join(io.host)
    assert delivered, io.msgs
    assert h._rx_resend is None and not h._rx_failed
    feed(h, recording(), block)
    assert b''.join(io.host) == delivered


def test_alternative_rate_does_not_allow_failed_data_or_leak_to_next_delivery():
    h, io = station()
    assert not h._resend_matches(3)  # Failed candidates must still match prediction.
    assert not h._previous_base_retry_rate(2)
    h._rx_resend, h._expect_level = None, 1
    h._advance_resend(missing=True)
    assert not h._previous_base_retry_rate(3)


def test_refused_query_reply_adds_no_alternative_rate(monkeypatch):
    h, io = receiver('500', 6)
    h._irs_nak(A._phy.LevelRead(6, (), None, 1))
    h._nak_unheard = True
    before = set(h._rx_retry_rates)
    monkeypatch.setattr(h, '_send_burst', lambda *a: False)
    assert not h._nak_owed_over()
    assert h._rx_retry_rates == before
    assert h._expect_level == 5


@pytest.mark.parametrize('query', [97, 103])
@pytest.mark.parametrize('block', [512, 960, 4096])
def test_recorded_early_query_repeats_ack_instead_of_restarting_delivery(query, block):
    from .test_idle_poll import _Clock
    p = Path(__file__).parent / f'fixtures/kl7qr-retry-1025/query{query}.wav'
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float)/32768
    h, io = receiver('500', 6)
    h.role, h.caller, h.called = 'initiator', 'W9SSJ', 'KL7QR'
    h.turn = A._TURN_PEER
    h._answer_owed, h._owed_block = A._OWED_OVER, False
    h._reack_frame = A.OVER_CONTINUE_BURST
    h._unkeyed_at = _Clock.t
    h._delivery_read, h._peer_over = True, 35
    h._rx_read = (6, None)  # Saved ACK held L6 after a clean L6 over.
    # Keep a little trailing silence so every callback size sees the whole query.
    feed(h, np.concatenate([x, np.zeros(4800)]), block)
    assert len(io.sent) == 1, io.msgs
    assert any('re-keying it' in m for m in io.msgs)
    assert not any('was not read' in m for m in io.msgs)
    assert h._rx_resend is None and h._expect_level == 6


def test_query_after_time_for_expected_data_still_requires_recovery():
    from hfmodem.kestrel.vara import vara_frames as F
    h, io = receiver('500', 6)
    h.role = 'initiator'
    h._unkeyed_at = 1000
    h._idle_kind = F.SESSION_RESPONDER_IDLE
    h._idle_end = 1000 + A._phy.burst_samples('500', 6)/48000 + .1 + A._span(h._idle_kind)/48000
    assert not h._answer_unheard()
    h._answer_owed, h._owed_block = A._OWED_OVER, False
    h._reack_frame = A.OVER_CONTINUE_BURST
    assert h._reack_over()
    assert h._rx_resend is not None
