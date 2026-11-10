# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A caller holding its answer to the answerer's over, with traffic queued.

Kestrel calling Kestrel over the OpenSlot air loop, 3 dB moderate fading,
2026-11-10: the caller read the answerer's last over while it was still
keying, held the answer, and with two blocks queued keyed a turn-request into
that hold. The ask dropped the held window, the answerer's seven queries drew
nothing, and it closed with the over unconfirmed. A VARA HF receiver never
asks while an over of the peer's is owed, and under CHAT answers the last
over with frame 0 when it has traffic (2026-10-18 and 2026-10-21 VARA HF pairs).

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock
from .test_limited_responder_query_recovery import on_air, pending, query, receiver
from .test_responder_control_stream import feed


def test_the_held_answer_is_keyed_at_the_query_with_frame_zero(clock):
    h, hi = pending(final=True)
    h.chat = h._chat_profile = True
    c, ci = receiver(h)
    c.chat, c._released = True, True
    # Our release's answer is still due, so the host's lines owe the ask.
    c._ask_not_before = _Clock.t + 1
    c.send(b'hello there')
    c.send(b'x' * 40)
    assert c._ask_owed and not ci.sent
    _Clock.t += 2
    assert c._answer_data_over(on_air(hi.sent[-1]), hold=True)
    assert c._rx_window.W is not None and not ci.sent, ci.msgs[-3:]
    feed(c, np.zeros(2 * M.FS))
    assert not ci.sent, ci.msgs[-3:]
    query(h, hi, 11)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS)])
    assert len(ci.sent) == 1, ci.msgs[-3:]
    traffic = M.synth_tone_pairs(F.control_bursts(c.caller, '500', 0)[0])
    np.testing.assert_array_equal(ci.sent[0], traffic)
    assert ci.host == [b'last over']
    feed(h, on_air(ci.sent[-1]))
    assert h._tx_pending is None and h.payload_ledger.confirmed == 9, h.io.msgs[-3:]
    # The answerer grants the turn the frame 0 asked for, and our line goes.
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS)])
    assert c.turn == A._TURN_OURS and c._tx_pending is not None, ci.msgs[-3:]
