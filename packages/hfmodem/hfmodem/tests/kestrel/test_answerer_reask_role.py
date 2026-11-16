# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An answerer whose answer to the caller's last over went unheard, with text queued.

Kestrel answering Kestrel, OpenSlot clients, 3 dB moderate fading, seed
20261111 (2026-11-10), 69.99 s: we answered the caller's last over, it did not
hear us and queried; with our text queued we re-answered each query with the
caller's own 17-symbol turn-request, keyed to the caller's call, four times in
16 s. The caller read none of them and queried on until our own 1078/1 at
87.29 s, which it granted as the over's acknowledgement at once. A VARA HF BW500
responder that had read the over answered the caller's short query with 1078/1,
twice (2026-09-27, W9SSJ -> WI9YZA).

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF
from .test_idle_poll import clock as clock, _Clock as _Clock
from .test_responder_control_stream import feed
from .test_setup_query_recovery import setup, padded

CALLER, CALLED = 'W9SSJ-1', 'W9SSJ'


def _answered_last_over():
    hs, io = setup(caller=CALLER, ask=2)
    feed(hs, padded(M.synth_burst(CALLER, F.for_bw(F.SESSION_TURN_REQUEST, '500'))))
    level = phy.level_key('500', hs._expect_level)
    body = phy.vara_body(b'ok 73\r', CALLER, body_len=phy.body_size('500', level), level=level)
    feed(hs, padded(OF.data_over_tx(body, bw='500', level=level)))
    assert hs._answer_owed == A._OWED_RELEASE and io.host, io.msgs[-4:]
    return hs, io


def test_a_queried_answer_with_text_queued_asks_with_the_responders_request():
    hs, io = _answered_last_over()
    hs.send(b'de W9SSJ text queued')
    sent = len(io.sent)
    query = M.synth_burst(CALLED, F.for_bw(F.SESSION_ANSWER_QUERIES_SHORT[1], '500'))
    feed(hs, padded(query))
    callers = [M.synth_burst(call, F.for_bw(F.SESSION_TURN_REQUEST_SHORT, '500')) for call in (CALLER, CALLED)]
    ours = M.synth_burst(CALLER, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, '500'))
    new = io.sent[sent:]
    assert not any(len(s) == len(c) and np.allclose(s, c) for s in new for c in callers), io.msgs[-4:]
    assert len(new) == 1 and len(new[0]) == len(ours) and np.allclose(new[0], ours), io.msgs[-4:]
    assert hs.turn == A._TURN_ASKED and hs._tx_pending is None and hs._txq, io.msgs[-4:]
