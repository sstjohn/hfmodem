# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Do not strand a peer after DATA delivery when the caller leaves frequency.

The 2026-10-25 stock-modem departure bench withheld all later caller audio.
At record 6, the fixed pair-137 handback left KI5IXX polling every 2.4 s;
stock's pair-135 form and the complete f20 both established a 10 s idle hold.
At stock-negotiated record 2, SESSION_DRAINED failed the same boundary and
f20 passed. All cases delivered the invitation to stock's host data socket.
Consequently host delivery alone was not a handback interoperability test.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from hfmodem.tests.kestrel.test_bw500_query_turn_idle_1024 import _pending, clock as clock  # noqa: F401


@pytest.mark.parametrize('called', ['K5CG', 'KI5IXX', 'KL7QR', 'KQ4QWH'])
@pytest.mark.parametrize('record', [1, 2, 4, 6, 12])
def test_empty_final_ack_confirms_handoff_without_a_later_host_write(clock, called, record):
    hs, io = _pending(clock, payload=b'12 <QF:7104250>', compression='TEXT')
    hs.called = called
    body, over, _ = hs._tx_pending
    hs._tx_pending = body, over, record
    hs._rec500 = record
    io.sent.clear()

    hs._took_control_burst(idle=True)

    assert hs.pending_bytes == 0 and hs._tx_pending is None
    assert hs.turn == A._TURN_PEER and hs._released and not hs._handed_high
    assert len(io.sent) == 1
    assert np.array_equal(io.sent[0], M.synth_burst(
        called, F.for_bw(F.SESSION_TURN_IDLE_SHORT, '500')))


def test_refused_idle_confirmation_remains_owed(clock):
    hs, io = _pending(clock)
    io.tx_went_out = lambda: False
    hs._took_control_burst(idle=True)
    assert hs._release_owed and hs.turn == A._TURN_OURS
    io.tx_went_out = lambda: True
    assert hs._release_turn(idle=True)
    assert not hs._release_owed and hs.turn == A._TURN_PEER
