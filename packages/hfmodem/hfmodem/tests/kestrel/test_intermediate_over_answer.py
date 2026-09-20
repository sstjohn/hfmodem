# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""An over with another one behind it is not answered the way the last one is.

Both the 2026-08-28 connects and five gateway sessions before them delivered
exactly 89 payload bytes and stopped. 89 is one base BW2300 over, so the reading
that they hit a ceiling was never available: they sent one over and were told to
stop, and the only thing in those sessions that could have told them is this
station's answer.

A bench of two stock instances on 2026-08-30, one cable per direction, keyed the
whole answer out. A station answers each INTERMEDIATE over of a multi-over
delivery with a short two-tone burst and only the LAST over with the 11-symbol
control burst — 6 + 1 across a 600-byte delivery, 4 + 1 across a 200-byte BW500
one, and 2 + 1 in each direction of a 178-byte exchange. This build keyed the
11-symbol burst at every over, so a gateway's first over drew the frame a stock
station keys only for its last, and a stock responder answers that by releasing
the turn: handed a 356-byte greeting it delivered 89 bytes and released with 267
still queued, which is the gateway shape byte for byte.

WHAT SAYS WHICH IS THE OVER ITSELF. The discriminating run is the one handed
exactly one over's worth: 89 bytes queued, and the sender keyed TWO overs — a
full one, then a second delivering nothing to the host at all — which the caller
answered with the short burst and then the control burst. At the first of those
the caller already held every byte of the message, so neither its own appetite
nor any negotiated length can be what it read. A full body carries no trailer and
has another over behind it; a short body carries the trailer that ends the
delivery, and a sender with an exact multiple to move keys an empty over rather
than end on a full one.

TWO FRAMES CONTINUE A DELIVERY AND NEITHER EVER ENDS ONE. A stock responder
handed 356 bytes delivers all 356 against the 32-symbol `session-over-response`
and against the 8-symbol continue burst alike, and keyed at a CLOSING over both
leave it repeating that over and then idling with the release never coming. So
the bench cannot choose between them and provenance does: the 32-symbol frame is
generated per callsign and is the only one a real gateway has been recorded
sending a further over against (KE8LVA, 2026-08-26, three overs), while the
8-symbol one is a copy of another link's tail. So provenance says generated and
length says captured, and length is what a whole message turns on at the bench:
8 symbols is 0.341 s where 32 is 1.366, and a stock responder keying its own
repeat 1.43-1.65 s after its unkey leaves only the shorter answer room. A real
gateway waits the generated frame out and answers it through its first delivery,
so `over_continue` picks, and this evidence originally selected generated32.
The exact KC9GHZ/225-byte stock greeting now completes with short16 where the
late generated32 re-ACK overlaps the next DATA on the host clocks and a middle
block is lost. Short16 is the current BW2300 initial default; generated32 remains
explicitly selectable for its real-gateway evidence [vara_arq, OVER_CONTINUE_ANSWERS].

THE PHASE IS THE OTHER TERM. The first over a gateway keys after taking the turn
back drew the idle cadence from the generated frame five times at three gateways
on 2026-09-06, and at the bench a stock responder re-keys 1.37-1.43 s after its
own unkey in that phase, before the 1.366 s frame has ended, where during its
first delivery it waits the frame out. So after a handover the captured burst
answers whatever is set: it ends at +0.5 s and carried six consecutive
post-handover overs at the bench, byte-exact, in every fetch of 2026-09-03/04.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel.test_data_over_gate import _HostIO, _connected

MYCALL, CALLED = "W9SSJ", "KC9GHZ"

#: The three overs of a 178-byte delivery, as a stock sender frames them: two
#: full, then the empty one that closes it.
_DELIVERY = (b"A" * 89, b"B" * 89, b"")


def _over(payload: bytes, index: int = 0) -> np.ndarray:
    """One base-level DATA over carrying ``payload``, framed as a sender frames it."""
    return tx.synth_burst(_phy.vara_body(payload, MYCALL), over=index)


def _answer(hs, io, payload: bytes, index: int = 0) -> list[tuple[int, int]]:
    """Drive one over in and return the tone pairs of the burst it drew."""
    before = len(io.sent)
    hs.on_rx_audio(np.asarray(_over(payload, index), float))
    assert len(io.sent) == before + 1, "the over drew no burst"
    burst = io.sent[-1]
    return [tuple(sorted(p)) for p in
            MK.demod_tone_pairs(burst, round(len(burst) / MK.HOP))]


def _captured_burst() -> list[tuple[int, int]]:
    return [tuple(sorted(p)) for p in VF.OVER_CONTINUE_CALLER_2300]


def _control_burst() -> list[tuple[int, int]]:
    return [tuple(sorted(p)) for p in VF.CONTROL_BURST_CALLER_2300]


def _generated_answer(hs, io) -> bool:
    """Is the last burst keyed the generated 32-symbol `session-over-response`,
    keyed to the station we called?"""
    burst = io.sent[-1]
    n = round(len(burst) / MK.HOP)
    return (n == len(VF.SESSION_OVER_RESPONSE.preamble)
            + VF.SESSION_OVER_RESPONSE.n_payload
            and MK.demod_tones(burst, n)
            == VF.handshake_tones(hs.called, VF.SESSION_OVER_RESPONSE))


def _generated(io=None):
    return _connected(io=io, over_continue=VA.OVER_CONTINUE_GENERATED)


def _captured(io=None):
    return _connected(io=io, over_continue=VA.OVER_CONTINUE_CAPTURED)


def _peer_control() -> np.ndarray:
    """The peer's 11-symbol control burst: the grant to our turn-request, and its
    answer to each over we key."""
    return MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)


def _handed_over(hs, io) -> None:
    """Take the turn for one over of our own and give it back, the way a login
    goes out behind a greeting."""
    hs.send(b"our login block")
    hs.on_rx_audio(_peer_control())           # granted -> our over
    hs.on_rx_audio(_peer_control())           # answered, nothing queued -> release
    assert hs.turn == VA._TURN_PEER
    assert any("turn released" in m for m in io.msgs), io.msgs


# --------------------------------------------------------------------------- #
# What the over says.
def test_a_full_over_says_another_one_follows_it():
    body = _phy.vara_body(b"A" * 89, MYCALL)
    assert len(body) == 90
    assert not _phy.over_is_last(body, MYCALL)


@pytest.mark.parametrize("n", [0, 1, 43, 66, 88])
def test_a_short_over_closes_the_delivery(n):
    assert _phy.over_is_last(_phy.vara_body(b"x" * n, MYCALL), MYCALL)


def test_an_exact_multiple_still_owes_a_closing_over():
    """The measurement that separates the over's framing from the receiver's own
    knowledge: 89 bytes is a whole over and cannot be framed as a last one, so a
    sender with 89 bytes to move has to key a second, empty over to end on."""
    assert not _phy.over_is_last(_phy.vara_body(b"A" * 89, MYCALL), MYCALL)
    assert _phy.over_is_last(_phy.vara_body(b"", MYCALL), MYCALL)


def test_the_record_below_base_is_read_against_its_own_body():
    """A peer that drops a level keys a 48-byte body, and its trailer sits at the
    end of that one. Measured against 90 the same over reads as full and the
    delivery never ends."""
    body = _phy.vara_body(b"short", MYCALL)[:48]
    assert _phy.over_is_last(body, MYCALL)


# --------------------------------------------------------------------------- #
# What we key back.
def test_an_intermediate_over_draws_the_short_frame_by_default():
    """BW2300's exact stock greeting advances without the generated32 re-ACK
    overlap. Explicit generated32 retains the separate real-gateway option."""
    hs, io = _connected()
    _answer(hs, io, b"A" * 89)
    assert _short_answer(hs, io)


def test_the_default_answer_follows_the_bandwidth():
    """BW2300/BW2750 use short16 after the exact KC9GHZ greeting tests;
    BW500 retains captured8. Explicit selection still overrides the default.
    """
    assert _connected()[0].over_continue == VA.OVER_CONTINUE_SHORT
    assert _generated()[0].over_continue == VA.OVER_CONTINUE_GENERATED
    assert _connected(bw="500")[0].over_continue == VA.OVER_CONTINUE_CAPTURED
    assert _connected(bw="2750")[0].over_continue == VA.OVER_CONTINUE_SHORT
    hs, _ = _connected(bw="500", over_continue=VA.OVER_CONTINUE_GENERATED)
    assert hs.over_continue == VA.OVER_CONTINUE_GENERATED
    hs, _ = _connected(bw="2750", over_continue=VA.OVER_CONTINUE_GENERATED)
    assert hs.over_continue == VA.OVER_CONTINUE_GENERATED


def test_an_intermediate_over_draws_the_captured_burst_when_asked_for():
    """The bench's frame: 3 fetches of 3 byte-exact at a stock responder that
    re-keys 1.43-1.65 s after its unkey, where the generated frame took 0 of 3."""
    hs, io = _captured()
    assert _answer(hs, io, b"A" * 89) == _captured_burst()


def _short_answer(hs, io) -> bool:
    """Is the last burst keyed the 16-symbol `session-over-response-short`, keyed
    to the station we called?"""
    burst = io.sent[-1]
    n = round(len(burst) / MK.HOP)
    return (n == len(VF.SESSION_OVER_RESPONSE_SHORT.preamble)
            + VF.SESSION_OVER_RESPONSE_SHORT.n_payload == 16
            and MK.demod_tones(burst, n)
            == VF.handshake_tones(hs.called, VF.SESSION_OVER_RESPONSE_SHORT))


@pytest.mark.parametrize("setting", [None, VA.OVER_CONTINUE_GENERATED])
def test_a_delivery_after_a_handover_is_answered_with_the_short_frame(setting):
    """The generated frame took every greeting over on 2026-09-06 and drew the idle
    cadence from the first over the gateway keyed after taking the turn back, at
    KE8LVA, KC9GHZ and K0SI alike. The phase, not the first delivery's setting,
    picks the frame from there; the closing over still draws the control burst."""
    hs, io = _connected(over_continue=setting)
    _answer(hs, io, b"the whole greeting")
    _handed_over(hs, io)
    _answer(hs, io, b"A" * 89, 1)
    assert _short_answer(hs, io)
    assert "generated 16-symbol" in io.msgs[-1] and "after a handover" in io.msgs[-1]
    _answer(hs, io, b"B" * 89, 2)
    assert _short_answer(hs, io)
    assert _answer(hs, io, b"FC EM ABCDEF 100 90 0\rF> 3B\r", 3) == _control_burst()


def test_the_answer_after_a_handover_is_its_own_setting():
    """Two phases with separate evidence get separate choices: the first delivery
    keeps whatever it was given while the post-handover answer changes."""
    hs, io = _connected(over_continue=VA.OVER_CONTINUE_GENERATED,
                        over_continue_after=VA.OVER_CONTINUE_CAPTURED)
    _answer(hs, io, b"A" * 89)
    assert _generated_answer(hs, io)
    _answer(hs, io, b"the rest of the greeting", 1)
    _handed_over(hs, io)
    assert _answer(hs, io, b"A" * 89, 2) == _captured_burst()
    assert "captured 8-symbol" in io.msgs[-1] and "after a handover" in io.msgs[-1]
    hs, io = _connected(over_continue_after=VA.OVER_CONTINUE_GENERATED)
    _answer(hs, io, b"the whole greeting")
    _handed_over(hs, io)
    _answer(hs, io, b"A" * 89, 1)
    assert _generated_answer(hs, io)


def test_the_short_frame_is_the_lattice_sibling_and_ends_early():
    """`short` is off this station's own NS0A tape, keyed to the called station
    like the 32-symbol frame, and half its length: 0.68 s against 1.366, so it
    ends inside the 1.37 s a responder waits after a handover."""
    hs, io = _connected(over_continue=VA.OVER_CONTINUE_SHORT)
    _answer(hs, io, b"A" * 89)
    assert _short_answer(hs, io)
    assert MK.demod_tones(io.sent[-1], 16) != VF.handshake_tones(
        "KB5LZK", VF.SESSION_OVER_RESPONSE_SHORT)
    assert len(io.sent[-1]) / MK.FS == pytest.approx(0.683, abs=0.01)
    assert len(io.sent[-1]) < VA._DATA_OVER_MIN


def test_the_first_delivery_keeps_the_selected_generated_frame_until_handover():
    """A queued reply is not a handover: the explicitly selected initial
    generated32 answer stays selected throughout the remaining greeting."""
    io = _HostIO(b"a reply the host queued early")
    hs, _ = _connected(io=io, over_continue=VA.OVER_CONTINUE_GENERATED)
    _answer(hs, io, b"A" * 89)
    assert hs._txq and _generated_answer(hs, io)
    _answer(hs, io, b"B" * 89, 1)
    assert _generated_answer(hs, io)


def test_the_last_over_draws_the_control_burst_on_all_settings():
    """A closing over draws the final control for every continuation choice."""
    for hs, io in (_connected(), _generated(), _captured()):
        assert _answer(hs, io, b"the whole message") == _control_burst()


def test_a_three_over_delivery_is_answered_two_and_one():
    """The shape a gateway greeting arrives in, and the whole of what stopped it:
    every byte reaches the host, and only the closing over frees the turn."""
    hs, io = _captured()
    answers = [_answer(hs, io, p, i) for i, p in enumerate(_DELIVERY)]
    assert answers == [_captured_burst(), _captured_burst(), _control_burst()]
    assert b"".join(io.host) == b"A" * 89 + b"B" * 89


def test_a_three_over_delivery_is_answered_two_and_one_on_the_generated_frame():
    hs, io = _generated()
    for i, payload in enumerate(_DELIVERY[:-1]):
        _answer(hs, io, payload, i)
        assert _generated_answer(hs, io)
    assert _answer(hs, io, _DELIVERY[-1], 2) == _control_burst()
    assert b"".join(io.host) == b"A" * 89 + b"B" * 89


@pytest.mark.parametrize("setting", VA.OVER_CONTINUE_ANSWERS)
def test_a_repeated_full_over_repeats_the_continue_answer(setting):
    """A lost ACK does not turn a full DATA frame into a delivery ending.

    The stock measurement above distinguishes continue from final ACK: the latter
    releases a sender with the rest of its greeting still queued. A repeated full
    frame still needs the continue answer, with its bytes delivered only once.
    """
    hs, io = _connected(over_continue=setting)
    first = _answer(hs, io, b"A" * 89)
    assert _answer(hs, io, b"A" * 89) == first
    assert hs._answer_owed == VA._OWED_OVER
    assert io.host == [b"A" * 89], "the repeat must not reach the host twice"
    _answer(hs, io, b"B" * 89, 1)
    assert _answer(hs, io, b"ending", 2) == _control_burst()
    assert b"".join(io.host) == b"A" * 89 + b"B" * 89 + b"ending"


@pytest.mark.parametrize("setting", VA.OVER_CONTINUE_ANSWERS)
def test_the_turn_is_not_taken_anywhere_inside_the_peers_delivery(setting):
    """A host that answers before the delivery has ended must not cost the rest of
    it, and it must not cost the acknowledgement either.

    Every over of the peer's draws the frame it asks for, the closing one
    included: the release that hands us the channel follows that acknowledgement,
    so a turn-request keyed in its place buys the ask and loses the release. What
    the queue does is put the ask in the peer's own cadence
    [see `VaraStationHandshake._reack_release`].
    """
    io = _HostIO(b"a reply the host queued early")
    hs, _ = _connected(io=io, over_continue=setting)
    _answer(hs, io, b"A" * 89)
    assert hs._txq, "the host's reply is queued"
    assert VF.SESSION_TURN_REQUEST.name not in " ".join(io.msgs)
    _answer(hs, io, b"and the rest of it")
    assert VF.SESSION_TURN_REQUEST.name not in " ".join(io.msgs)
    assert _control_burst() == [tuple(sorted(p)) for p in
                                MK.demod_tone_pairs(io.sent[-1], 11)]
    assert hs.turn == VA._TURN_PEER
    assert hs._answer_owed == VA._OWED_RELEASE


@pytest.mark.parametrize("setting", VA.OVER_CONTINUE_ANSWERS)
def test_the_ask_goes_out_in_the_peers_own_cadence(setting):
    """Where the deferred ask ends up: the peer's idle burst opens the only gap
    wide enough for a 1.37 s frame, and the rung before it is the acknowledgement
    the peer may simply have missed."""
    io = _HostIO(b"a reply the host queued early")
    hs, _ = _connected(io=io, over_continue=setting)
    _answer(hs, io, b"the whole greeting")
    idle = MK.synth_burst(CALLED, VF.SESSION_RESPONDER_OVER_IDLE)
    hs.on_rx_audio(idle)
    assert _control_burst() == [tuple(sorted(p)) for p in
                                MK.demod_tone_pairs(io.sent[-1], 11)], io.msgs
    hs.on_rx_audio(idle)
    assert MK.demod_tones(io.sent[-1], 32) == VF.handshake_tones(
        MYCALL, VF.SESSION_TURN_REQUEST), io.msgs
    assert hs.turn == VA._TURN_ASKED


def test_the_log_names_the_frame_that_went_out():
    """A slot flies one setting and the report has to state which without reading
    the build. On 2026-08-30 an arm flew a change and the write-up could only
    infer what had been keyed."""
    hs, io = _connected()
    _answer(hs, io, b"A" * 89)
    assert any("generated 16-symbol" in m for m in io.msgs), io.msgs
    hs, io = _generated()
    _answer(hs, io, b"A" * 89)
    assert any("generated 32-symbol" in m for m in io.msgs), io.msgs
    hs, io = _captured()
    _answer(hs, io, b"A" * 89)
    assert any("captured 8-symbol" in m for m in io.msgs), io.msgs


def test_the_four_answers_are_four_different_bursts():
    """Guards the whole point: if any two of these became the same frame the fix
    would be silently gone."""
    assert _captured_burst() != _control_burst()
    assert len(_captured_burst()) == VF.OVER_CONTINUE_NSYM == 8
    assert len(_control_burst()) == VF.CONNECTED_ACK_NSYM == 11
    assert (len(VF.SESSION_OVER_RESPONSE_SHORT.preamble)
            + VF.SESSION_OVER_RESPONSE_SHORT.n_payload) == 16
    assert (len(VF.SESSION_OVER_RESPONSE.preamble)
            + VF.SESSION_OVER_RESPONSE.n_payload) == 32


def test_the_generated_answer_is_keyed_to_the_station_we_called():
    """What makes it right for a peer nobody has recorded, and the whole of what
    separates it from the captured copy: its tones follow the callsign, so a
    different link draws a different frame."""
    hs, io = _generated()
    _answer(hs, io, b"A" * 89)
    ours = MK.demod_tones(io.sent[-1], 32)
    assert ours == VF.handshake_tones(CALLED, VF.SESSION_OVER_RESPONSE)
    assert ours != VF.handshake_tones("KB5LZK", VF.SESSION_OVER_RESPONSE)


# --------------------------------------------------------------------------- #
# The frame itself.
def test_the_continue_burst_is_two_tone_throughout():
    """Every symbol lights two carriers, as the 11-symbol burst's do — which is
    what keeps it out of the single-tone session family."""
    assert all(a != b for a, b in VF.OVER_CONTINUE_CALLER_2300)


def test_the_continue_burst_renders_and_reads_back():
    x = MK.synth_tone_pairs(VF.OVER_CONTINUE_CALLER_2300)
    heard = [tuple(sorted(p)) for p in
             MK.demod_tone_pairs(x, VF.OVER_CONTINUE_NSYM)]
    assert heard == _captured_burst()


def test_the_continue_burst_is_shorter_than_an_over_the_receiver_would_answer():
    """It is keyed into a turnaround, and a burst that reached ``_DATA_OVER_MIN``
    would come back through the receiver as something to answer."""
    x = MK.synth_tone_pairs(VF.OVER_CONTINUE_CALLER_2300)
    assert len(x) < VA._DATA_OVER_MIN


def test_the_first_over_is_a_whole_one_at_this_bandwidth():
    """The 89 the seven gateways stopped at is the base level's own per-over
    payload and nothing else."""
    assert _phy.payload_size("2300") == rx.payload_bytes(rx.BASE_LEVEL) - 1 == 89
