# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A whole-over NAK retains both paired frames until a retry reaches the air."""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F
from .test_data_over_gate import _connected


def paired(monkeypatch, *, final=False):
    h, io = _connected(bw='500')
    h.chat = True
    h.turn = A._TURN_OURS
    rendered = []

    def render(bodies, over, level, lead):
        rendered.append((tuple(bodies), level))
        return np.zeros(1)

    monkeypatch.setattr(h, '_render_over', render)
    capacity = phy.body_size(h.bw, 4) - 1
    original = bytes(i % 251 for i in range(capacity * (7 if final else 8) + 7))
    h.send(original)
    for _ in range(6):
        h._continue_frame = (h._tx_pending, 0)
        h._took_over_continue()
    assert h._tx_pair is not None
    assert len(rendered[-1][0]) == 2
    assert h.payload_ledger.confirmed == 6 * capacity
    h.send(b'later delivery')
    return h, io, original[6 * capacity:], rendered


def outstanding(h):
    return (b''.join(h._tx_payload(b) for b in h._over_bodies(h._tx_pending[0]))
            + b''.join(h._txq))


@pytest.mark.parametrize('final', [False, True])
@pytest.mark.parametrize('refused', [False, True])
def test_pair_data_nak_repacketizes_atomically(monkeypatch, final, refused):
    h, io, remaining, rendered = paired(monkeypatch, final=final)
    pending, pair, queue = h._tx_pending, h._tx_pair, list(h._txq)
    confirmed = h.payload_ledger.confirmed
    kind = F.for_bw(F.SESSION_DATA_NAK_SHORT, h.bw)
    assert h._data_nak_fresh(kind)
    h._matched_data_nak = (pending, h._pending_answer_at, kind)
    io.tx_went_out = lambda: not refused
    h._took_data_nak()
    assert outstanding(h) == remaining + b'later delivery'
    assert h.payload_ledger.confirmed == confirmed
    assert h._txq[-1] == b'later delivery'
    # The NAK authorizes the existing lower-record plan for the entire over.
    assert rendered[-1][1] == 0
    assert len(rendered[-1][0]) == 1
    if refused:
        assert h._tx_pending is pending and h._tx_pair is pair
        assert h._txq == queue and h._tx_retries == 0
        io.tx_went_out = lambda: True
        h._took_control_burst(continue_reply=True)
        assert h.payload_ledger.confirmed == confirmed + sum(
            len(h._tx_payload(b)) for b in (pair[1], pending[0]))
    else:
        assert h._tx_pending[2] == 0 and h._tx_pair is None
        assert h._tx_retries == 1


def test_unqualified_repacketization_retries_whole_original_pair(monkeypatch):
    h, io, remaining, rendered = paired(monkeypatch)
    original = rendered[-1]
    pending, pair = h._tx_pending, h._tx_pair
    # No solicited query: the missing-DATA lower-record plan is unavailable.
    h._retry_data_over(peer_asked=True, adapt=True)
    assert rendered[-1] == original
    assert h._tx_pending is pending and h._tx_pair is pair
    assert outstanding(h) == remaining + b'later delivery'
