# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Generated DATA continues carry the receiver role's answer stream."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_data_over_gate import _connected


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('answer', [A.OVER_CONTINUE_GENERATED, A.OVER_CONTINUE_SHORT])
def test_generated_continue_matches_the_role_and_called_identity(bw, role, answer):
    h, io = _connected(bw=bw, role=role)
    h._start_compression('OFF')
    h._queue_host_payload(b'outbound debt is unrelated to a receive ACK')
    pending = h.payload_ledger.pending
    assert h._tx_continue_answer(answer) == answer
    assert len(io.sent) == 1
    # The responder's long continue is independently pinned by the existing
    # stock intermediate-query ACK. The short form is its 289 stream (the
    # connect-response stream) at the first frame, with the short preamble.
    if role == 'responder':
        descriptor = (F.SESSION_INTERMEDIATE_QUERY_ANSWER
                      if answer == A.OVER_CONTINUE_GENERATED
                      else replace(F.SESSION_OVER_RESPONSE_SHORT, seed_off=289))
    else:
        descriptor = (F.SESSION_OVER_RESPONSE if answer == A.OVER_CONTINUE_GENERATED
                      else F.SESSION_OVER_RESPONSE_SHORT)
    kind = F.for_bw(descriptor, bw)
    expected = M.synth_burst(h.called, kind)
    assert np.array_equal(io.sent[0], expected)
    assert not np.array_equal(io.sent[0], M.synth_burst(h.caller, kind))
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == pending
    assert h._tx_pending is None


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_short_refusal_falls_back_to_the_same_role_long_continue(bw, role):
    h, io = _connected(bw=bw, role=role)
    verdicts = iter([False, True])
    io.tx_went_out = lambda: next(verdicts)
    assert h._tx_continue_answer(A.OVER_CONTINUE_SHORT) == A.OVER_CONTINUE_GENERATED
    assert len(io.sent) == 2
    descriptor = (F.SESSION_INTERMEDIATE_QUERY_ANSWER if role == 'responder'
                  else F.SESSION_OVER_RESPONSE)
    assert np.array_equal(io.sent[-1], M.synth_burst(h.called, F.for_bw(descriptor, bw)))


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_high_speed_generated_continue_retains_its_rate_step(monkeypatch, role):
    h, io = _connected(bw='2300', role=role)
    h._link_high_speed = True
    monkeypatch.setattr(h, '_irs_step', lambda: 5)
    assert h._tx_continue_answer(A.OVER_CONTINUE_GENERATED) == A.OVER_CONTINUE_GENERATED
    descriptor = (F.SESSION_INTERMEDIATE_QUERY_ANSWER if role == 'responder'
                  else F.SESSION_OVER_RESPONSE)
    expected = replace(descriptor, preadv=1+62*5)
    assert np.array_equal(io.sent[-1], M.synth_burst(h.called, expected))


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_two_tone_continue_keeps_its_existing_role_specific_stream(role):
    h, io = _connected(bw='500', role=role)
    h._peer_over_state = (4, 0x89)
    assert h._tx_continue_answer(A.OVER_CONTINUE_BURST) == A.OVER_CONTINUE_BURST
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(
        F.over_continue(h.called, h.bw)[role == 'responder']))
