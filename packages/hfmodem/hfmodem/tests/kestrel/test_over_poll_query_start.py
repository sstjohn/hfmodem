# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The answerer's frame-12 query behind a continue of ours, timed from its own audio.

Kestrel calling Kestrel over the OpenSlot air loop, 3 dB moderate fading,
2026-11-10: the caller's continue to the answerer's over was lost, and of the
answerer's queries only the third, 5 s on, was read. A phase-0 query behind a
phase-0 answer is not proof and the timing bound put it past the over's span,
so the caller keyed the missing-over NAK, and the answerer resent  [DESIGN
section 6]. The caller read the resend and its continue was lost again; the
answerer queried with frame 12 0.9 s behind it. Frame 12 is also BW500's
over-poll, and its reader left the query's start where the earlier query had
put it, so the bound read this query as one from before our continue and the
caller NAKed a delivered over a second time. Timed from its own audio, the
query falls inside the span and draws the continue again.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np

from hfmodem.kestrel.vara import vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock
from .test_limited_responder_query_recovery import on_air, pending, receiver
from .test_responder_control_stream import feed


def test_a_frame_12_query_behind_our_continue_draws_the_continue_again(clock):
    h, hi = pending(payload=bytes(range(200)))
    c, ci = receiver(h)
    c._link_high_speed = True
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS // 2)])
    answer, sent = c._rx_window.A, len(ci.sent)
    assert answer.kind == 'continue' and answer.phase == 0 and c._lv_owed, ci.msgs[-4:]
    # As in the cell: a NAK of ours earlier in this delivery, so the phase is
    # not proof, and the start of the query that drew it, 8 s back.
    c._rx_window.trusted = False
    c._query_start = _Clock.t - 8
    # Our continue is lost; the answerer's frame-12 query follows it.
    feed(c, np.r_[on_air(M.synth_tone_pairs(F.answer_query(c.called, '500', 1))), np.zeros(M.FS)])
    assert len(ci.sent) == sent + 1, ci.msgs[-4:]
    assert c._rx_window.A is answer and answer.reanswers == 1, ci.msgs[-4:]
