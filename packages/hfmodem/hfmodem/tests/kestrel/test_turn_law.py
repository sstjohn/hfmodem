# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Whose turn it is to transmit, and the frames that say so.

A VARA session is strictly one over each way, and which station may key a DATA
over is carried in the post-CONNECTED session frames  [spec 05 §5.3.3, §5.4].
Those frames were a hardcoded capture here until their generator parameters were
recovered: each one's 31 payload tones pin a single 24-bit generator state out of
2**24, and a discrete log against the seeding map gives (SEED_OFF, PREADV).

The arbiter for every parameter below is a real VARA's own symbols, demodulated
off recordings — not a round trip through kestrel's own encoder, which would stay
green with the parameters wrong. Each frame is graded against two independent
recordings keyed to DIFFERENT callsigns, which is what separates the state the
frame reports from the identity it is keyed to:

  * a logged VARA-to-VARA BW2300 session, AAAA1 calling BBBB2, station A's own
    transmit path;
  * two off-air BW2300 sessions in which this station (W9SSJ) called the Winlink
    gateways NS0A and KC9GHZ, our own transmissions read back through the
    receiver mute that follows every keying.

The recordings are large and git-ignored, so the tone sequences they yielded are
inlined here. The tail of a frame read through our own unmuting receiver is not
recoverable — the last one or two symbols read a different carrier in every
capture — so those captures are graded over the symbols the recording resolves.

A NAME COST THIS PROJECT A SEARCH, and the shape is worth keeping in view. The
17-symbol burst that releases the turn was in this tree the whole time, filed as
a disconnect request off a loopback tape. A five-candidate sweep for "the frame
that releases the turn" found nothing, because the frame it was looking for was
already held under an event that ends sessions and was therefore never in the
candidate set. Two stock 4.9.0 instances on one cable settled it on 2026-08-26:
17 of 17 tones against the same generator parameters, and both sessions ran on
for two further overs after keying it. A frame's name here records when it was
seen, not what it does, and a name that asserts a meaning is a name that can hide
the frame from the next search.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_ofdm as OF

_MYCALL = "W9SSJ"
_CALLED = "KC9GHZ"

# (frame, callsign it is keyed to, tones the recording resolved, source)
_OFF_AIR = [
    # --- logged VARA-to-VARA session, AAAA1 -> BBBB2 (station A's transmit path)
    (VF.SESSION_CONFIRM, "BBBB2",
     [62, 82, 73, 46, 59, 64, 69, 60, 37, 42, 79, 30, 87, 88, 63, 58], "loopback"),
    (VF.SESSION_KEEPALIVE_A, "BBBB2",
     [74, 98, 39, 60, 87, 68, 97, 52, 51, 90, 69, 92, 33, 36, 43, 40,
      37, 98, 59, 72, 51, 58, 37, 38, 63, 64, 33, 38, 39, 66, 37, 50], "loopback"),
    (VF.SESSION_KEEPALIVE_B, "BBBB2",
     [74, 42, 43, 32, 39, 58, 43, 90, 35, 74, 59, 62, 83, 82, 35, 44,
      53, 92, 63, 54, 73, 46, 57, 52, 33, 62, 49, 86, 89, 62, 47, 42], "loopback"),
    # keyed to AAAA1, the CALLER, and keyed 0.17 s after that station's host
    # handed it a payload — the first thing it put on the air before its overs.
    (VF.SESSION_TURN_REQUEST, "AAAA1",
     [74, 64, 29, 84, 71, 80, 65, 90, 55, 94, 81, 98, 67, 98, 93, 30,
      73, 94, 91, 50, 75, 90, 51, 76, 33, 96, 95, 32, 59, 70, 37, 90], "loopback"),
    # --- off air, W9SSJ -> NS0A
    (VF.SESSION_CONFIRM, "NS0A",
     [62, 44, 57, 64, 47, 40, 83, 64, 75, 50, 93, 36, 45, 60, 97], "NS0A"),
    (VF.SESSION_OVER_RESPONSE_SHORT, "NS0A",
     [62, 74, 81, 68, 59, 60, 53, 82, 79, 86, 93, 66, 59, 44], "NS0A"),
    (VF.SESSION_OVER_RESPONSE, "NS0A",
     [74, 40, 29, 80, 75, 96, 87, 72, 77, 64, 73, 34, 51, 90, 61, 62,
      91, 72, 59, 52, 61, 72, 37, 86, 81, 82, 79, 88, 79, 30, 57], "NS0A"),
    # --- off air, W9SSJ -> KC9GHZ
    (VF.SESSION_KEEPALIVE_A, "KC9GHZ",
     [74, 98, 35, 30, 89, 64, 87, 40, 97, 52, 75, 44, 81, 56, 97, 70,
      97, 72, 49, 62, 61, 88, 93, 88, 37, 86, 65, 60, 33, 80, 81], "KC9GHZ"),
    (VF.SESSION_OVER_RESPONSE, "KC9GHZ",
     [74, 36, 57, 44, 55, 86, 91, 82, 51, 34, 67, 58, 55, 94, 85, 56,
      73, 80, 71, 84, 57, 98, 61, 94, 67, 62, 89, 44, 67, 34], "KC9GHZ"),
    # The turn-request keyed to W9SSJ — byte-identical off both gateways, which
    # is exactly what a frame keyed to the caller does and what one keyed to the
    # called station cannot.
    (VF.SESSION_TURN_REQUEST, _MYCALL,
     [74, 50, 49, 42, 97, 48, 31, 40, 55, 94, 81, 98, 51, 56, 55, 74,
      33, 82, 97, 82, 67, 52, 73, 36, 97, 96, 57, 58, 41, 60], "NS0A+KC9GHZ"),
    # The gateways' own answer to our idle frame, and the only entry in this
    # table read off a transmitter that is not ours — so, unlike every capture
    # above, no receiver mute eats its tail and all 32 symbols resolve. Five
    # occurrences behind each of these, every one identical.
    (VF.SESSION_IDLE_RESPONSE, "NS0A",
     [74, 92, 39, 84, 97, 58, 47, 92, 69, 30, 45, 44, 35, 76, 71, 88,
      47, 30, 49, 34, 69, 50, 31, 42, 55, 70, 73, 90, 87, 90, 53, 98], "NS0A"),
    (VF.SESSION_IDLE_RESPONSE, "KC9GHZ",
     [74, 84, 85, 62, 69, 70, 93, 78, 93, 68, 57, 96, 71, 68, 57, 78,
      69, 94, 39, 76, 41, 86, 33, 52, 55, 56, 47, 92, 91, 56, 75, 96], "KC9GHZ"),
    # And the second frame here read off a gateway's transmitter: thirteen
    # keyings, all 32 symbols, seven of them clear of the band at every one.
    (VF.SESSION_RESPONDER_IDLE, "KE8LVA",
     [74, 80, 31, 74, 39, 40, 29, 80, 97, 88, 67, 60, 65, 62, 39, 44,
      91, 46, 95, 72, 67, 48, 69, 72, 45, 64, 49, 40, 61, 74, 87, 42], "KE8LVA"),
    # The turn release, off the transmitter of a stock VARA HF 4.9.0 calling
    # another one across a fake cable on 2026-08-26 — one cable per direction, so
    # the recording holding it holds nothing else. Neither end is ours and no mute
    # crosses it, so all 17 symbols resolve, preamble
    # included. Two sessions, the same burst at 21.050 s and 21.015 s, each
    # 0.13-0.14 s after the peer's answer to the caller's own over and each
    # answered 0.07-0.10 s later by the peer's next over.
    (VF.SESSION_TURN_RELEASE, "W1AW",
     [62, 67, 36, 45, 80, 47, 74, 73, 74, 89, 48, 83, 54, 61, 44, 91, 60],
     "two-VARA bench"),
    # And the close, off the same transmitter: one burst at 32.947 s of the third
    # session, keyed at a host DISCONNECT and followed by nothing but the CW ident.
    (VF.SESSION_DISCONNECT_REQ, "W1AW",
     [74, 38, 83, 72, 87, 96, 49, 48, 59, 84, 39, 92, 77, 62, 35, 76, 67, 62,
      33, 90, 73, 62, 75, 70, 89, 32, 45, 86, 83, 78, 59, 74], "two-VARA bench"),
]


@pytest.mark.parametrize(
    "kind,call,tones,src", _OFF_AIR,
    ids=[f"{k.name}-{c}-{s}" for k, c, tones, s in _OFF_AIR])
def test_generator_reproduces_a_real_stations_symbols(kind, call, tones, src):
    got = VF.handshake_tones(call, kind)[:len(tones)]
    assert got == tones, f"{kind.name} keyed to {call} ({src})"


def test_the_turn_idle_frame_is_the_whole_capture_byte_for_byte():
    """The one session frame a recording resolved to its last symbol.

    It was carried here as a captured constant on the reading that no generator
    could produce it. The generator produces it, all 32 tones, from the callsign
    of the station that recorded it — which is the point: the frame names US, and
    that is why the same 32 tones came back from three different gateways.
    """
    assert (VF.handshake_tones(VF.SESSION_RESPONSE_2300_CALL, VF.SESSION_TURN_IDLE)
            == list(VF.SESSION_RESPONSE_2300))


# --------------------------------------------------------------------------- #
# Counterexamples. Each of these is the same measurement read the wrong way, and
# each must fail.
def test_the_turn_frames_are_not_keyed_to_the_called_station():
    """Read un-reversed — keyed to the gateway instead of to us — the turn frames
    do not reproduce either capture, and would differ between the two gateways."""
    for kind in (VF.SESSION_TURN_REQUEST, VF.SESSION_TURN_IDLE):
        assert kind.keyed_by == "caller"
        for gw in ("NS0A", "KC9GHZ"):
            assert VF.handshake_tones(gw, kind) != VF.handshake_tones(_MYCALL, kind)
    assert (VF.handshake_tones("NS0A", VF.SESSION_TURN_IDLE)
            != list(VF.SESSION_RESPONSE_2300))


def test_the_over_response_is_not_keyed_to_the_caller():
    """And the mirror: the per-over response IS keyed to the called station, so
    keying it to ourselves must miss both gateways' captures."""
    assert VF.SESSION_OVER_RESPONSE.keyed_by == "called"
    for kind, call, tones, _src in _OFF_AIR:
        if kind is not VF.SESSION_OVER_RESPONSE:
            continue
        assert VF.handshake_tones(_MYCALL, kind)[:len(tones)] != tones


def test_the_idle_response_is_keyed_to_the_gateway_and_to_nothing_else():
    """The cross-callsign test that pins (288, 1241), stated as a check.

    The seeding map is many-to-one: for either gateway alone, a whole ladder of
    (SEED_OFF, PREADV) pairs reaches the same generator state, so a pair read off
    one recording proves nothing. What is decisive is that this ONE pair produces
    both gateways' bursts from their two different callsigns — and that reading it
    keyed to the caller, or to the wrong gateway, reproduces neither.
    """
    for gw in ("NS0A", "KC9GHZ"):
        ours = VF.handshake_tones(_MYCALL, VF.SESSION_IDLE_RESPONSE)
        theirs = VF.handshake_tones(gw, VF.SESSION_IDLE_RESPONSE)
        assert ours != theirs
        assert sum(1 for a, b in zip(ours[1:], theirs[1:]) if a == b) <= 3
    assert (VF.handshake_tones("NS0A", VF.SESSION_IDLE_RESPONSE)
            != VF.handshake_tones("KC9GHZ", VF.SESSION_IDLE_RESPONSE))
    assert VF.SESSION_IDLE_RESPONSE.keyed_by == "called"


def test_the_idle_response_is_not_the_keepalive_it_shares_a_preadv_with():
    """(288, 1241) against keepalive-b's (60, 1241): same pre-advance, and a
    different frame. Collapsing them would name a gateway's answer with the frame
    an initiator keys at it."""
    for call in ("NS0A", "KC9GHZ", "BBBB2"):
        a = VF.handshake_tones(call, VF.SESSION_IDLE_RESPONSE)
        b = VF.handshake_tones(call, VF.SESSION_KEEPALIVE_B)
        assert a != b
        assert sum(1 for u, v in zip(a[1:], b[1:]) if u == v) <= 3


def test_the_responder_idle_is_the_only_reading_its_state_admits():
    """One recording and one callsign, so the fit states its own argument here.

    KE8LVA's frame pins generator state 0xcb89c9, unique over all 2**24. The four
    SEED_OFFs this family already uses, discrete-logged from each of the session's
    two callsigns, give eight pre-advances with nothing left to fit — and only one
    of the eight is a number a burst could carry. Every 32-symbol frame in the
    table sits at PREADV = 1 (mod 62), 62 draws being one frame's payload, so they
    are all positions on one generator stream; 683 is position 11 on it, which
    none of the others occupies.
    """
    assert VF.SESSION_RESPONDER_IDLE.seed_off == 288
    assert VF.SESSION_RESPONDER_IDLE.preadv % 62 == 1
    assert VF.SESSION_RESPONDER_IDLE.keyed_by == "called"
    lattice = {k.preadv for k in VF.BURSTS.values() if k.n_payload == 31}
    assert all(p % 62 == 1 for p in lattice)
    assert VF.SESSION_RESPONDER_IDLE.preadv not in lattice - {683}
    ours = VF.handshake_tones(_MYCALL, VF.SESSION_RESPONDER_IDLE)
    theirs = VF.handshake_tones("KE8LVA", VF.SESSION_RESPONDER_IDLE)
    assert sum(1 for a, b in zip(ours[1:], theirs[1:]) if a == b) <= 3
    for kind in VF.BURSTS.values():
        if kind is VF.SESSION_RESPONDER_IDLE or kind.n_payload != 31:
            continue
        other = VF.handshake_tones("KE8LVA", kind)
        assert sum(1 for a, b in zip(theirs[1:], other[1:]) if a == b) <= 3


def test_the_turn_frames_are_distinct_states():
    """Holding the turn is not asking for it: one frame would collapse the two."""
    req = VF.handshake_tones(_MYCALL, VF.SESSION_TURN_REQUEST)
    idle = VF.handshake_tones(_MYCALL, VF.SESSION_TURN_IDLE)
    assert req != idle
    assert req[0] == idle[0] == 74, "the fixed preamble tone is state-invariant"
    assert sum(1 for a, b in zip(req[1:], idle[1:]) if a == b) <= 3


def test_control_tails_do_not_cross_classify():
    """The three tails a responder reports differ in all seven symbols, so no one
    of them may be read as another even with a symbol lost to the channel."""
    pre = list(VF.CONNECTED_ACK_PREAMBLE)
    for name, tail in VF.CONTROL_TAILS_500.items():
        assert VF.control_state(pre + list(tail)) == name
        for other, ref in VF.CONTROL_TAILS_500.items():
            if other != name:
                assert sum(1 for a, b in zip(tail, ref) if a != b) == len(tail)
    assert VF.control_state(pre + [(30, 40)] * 7) is None
    assert VF.control_state(pre) is None, "a burst with no tail states nothing"


# --------------------------------------------------------------------------- #
# The schedule: when a connected initiator may key a DATA over.
class _IO(VA.VaraIO):
    """Records what was keyed. Nothing here opens a device or reaches a radio."""

    def __init__(self):
        self.keys = 0
        self.msgs: list[str] = []
        self.sent: list[np.ndarray] = []

    def key(self, on): self.keys += bool(on)

    def tx(self, samples): self.sent.append(np.asarray(samples, float))

    def pending(self): ...

    def connected(self, *a): ...

    def log(self, msg): self.msgs.append(msg)


def _connected(bw: str = "2300"):
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw=bw)
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def _peer_control() -> np.ndarray:
    """The peer's 11-symbol two-tone control burst — what answers our every over."""
    return MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)


def _overs(k: int) -> bytes:
    """A payload that splits into exactly ``k`` DATA overs, the last one short.

    A whole multiple of the block owes an empty over behind it to end the delivery
    on, which is one more transmission than the turn law being measured here
    [see test_over_continue_answer].
    """
    return b"x" * (_phy.payload_size("2300") * (k - 1) + 40)


def _their_over(text: bytes, index: int) -> np.ndarray:
    """A BW2300 DATA over as the station at the other end keys it.

    Framed with OUR callsign, which is not a courtesy to the test: the body
    trailer ``arq.phy.vara_body`` writes is keyed to the CALLER from either end,
    and on this link the caller is us. So what arrives is byte-for-byte what a
    gateway sends, and carries nothing that names the station that keyed it.
    """
    return OF.data_over_tx(_phy.vara_body(text, _MYCALL), over=index)


def test_a_queued_payload_asks_for_the_turn_and_does_not_key_an_over():
    hs, io = _connected()
    hs.send(b"x" * 200)
    assert hs.turn == VA._TURN_ASKED
    assert any("session-turn-request" in m for m in io.msgs), io.msgs
    assert not any("tx DATA over" in m for m in io.msgs), (
        "keyed a DATA over before the peer answered the turn-request")


def test_the_turn_request_goes_out_keyed_to_us_not_to_the_gateway():
    hs, io = _connected()
    hs.send(b"x" * 10)
    heard = MK.demod_tones(io.sent[-1], 32)
    assert heard == VF.handshake_tones(_MYCALL, VF.SESSION_TURN_REQUEST)
    assert heard != VF.handshake_tones(_CALLED, VF.SESSION_TURN_REQUEST)


def test_the_peers_answer_takes_the_turn_and_starts_the_overs():
    hs, io = _connected()
    hs.send(b"x" * 200)
    io.msgs.clear()
    hs.on_rx_audio(_peer_control())
    assert hs.turn == VA._TURN_OURS
    assert any("tx DATA over #1" in m for m in io.msgs), io.msgs


def test_each_answered_over_draws_the_next_one_and_then_the_release():
    """And the release goes out in that same turnaround, which is where the
    2026-08-26 two-VARA capture puts it: over, the peer's answer, the release
    0.13-0.14 s later, the peer's own over 0.07-0.10 s after that."""
    hs, io = _connected()
    hs.send(b"x" * 200)                       # 89-byte blocks -> 3 overs
    hs.on_rx_audio(_peer_control())           # grant + over 1
    hs.on_rx_audio(_peer_control())           # over 2
    hs.on_rx_audio(_peer_control())           # over 3
    assert [m for m in io.msgs if "tx DATA over" in m][-1].startswith(
        "tx DATA over #3")
    io.msgs.clear()
    io.sent = []
    hs.on_rx_audio(_peer_control())           # queue empty
    assert hs.turn == VA._TURN_PEER
    assert any("session-turn-release" in m for m in io.msgs), io.msgs
    heard = MK.demod_tones(io.sent[-1], 17)
    assert heard == VF.handshake_tones(_CALLED, VF.SESSION_TURN_RELEASE)


def test_our_own_turn_idle_coming_back_does_not_key_another():
    """The runaway: while we hold the turn, every short burst used to draw the
    next over, and our own frames arrive back through the receiver. A turn-idle
    answering a turn-idle would key for as long as the link lasted."""
    hs, io = _connected()
    hs.send(b"x" * 89)
    hs.on_rx_audio(_peer_control())
    assert hs.turn == VA._TURN_OURS
    io.keys = 0
    io.msgs.clear()
    hs.on_rx_audio(MK.synth_tones(VF.handshake_tones(_MYCALL, VF.SESSION_TURN_IDLE)))
    assert io.keys == 0, "our own turn-idle, back through the receiver, keyed again"
    hs.on_rx_audio(MK.synth_tones(VF.handshake_tones(_CALLED, VF.SESSION_KEEPALIVE_B)))
    assert io.keys == 0, "a single-tone session frame is not the peer's answer"


def test_an_idle_link_with_nothing_queued_answers_an_over_the_way_a_caller_does():
    """No queue, no turn: the burst the over asks for, and the turn stays put.

    The arbiter is two stock VARA HF 4.9.0 instances over a fake cable, one
    recording per direction. On 2026-08-26 every over of three sessions drew the
    11-symbol control burst — but every over in those three was the last of its
    delivery, and on 2026-08-30 a run long enough to have an intermediate over
    put a continue-class frame behind each one and the control burst only behind
    the last. Which continue-class frame is `over_continue`'s, and the turn stays
    put on either  [vara_arq, OVER_CONTINUE_ANSWERS].

    This is a link's FIRST delivery — nothing keyed, nothing handed over — which
    is the state `over_continue` governs, and at BW2300 the frame it picks is now
    the generated 16-symbol one: against stock 4.9.0 on 2026-09-11 the 32-symbol
    answer's re-key landed across stock's middle DATA and 89 of the 225 greeting
    bytes never arrived, where the short frame took all three overs and the whole
    mail exchange behind them  [stock/runs/20260911T005601Z-2300-clean,
    20260911T011356Z-2300-short-ack]. Both other forms stay selectable.
    """
    hs, io = _connected()
    assert hs.turn == VA._TURN_PEER
    hs._tx_over_response(last=True)
    heard = MK.demod_tone_pairs(io.sent[-1], VF.CONNECTED_ACK_NSYM)
    assert ([tuple(sorted(p)) for p in heard]
            == [tuple(sorted(p)) for p in VF.CONTROL_BURST_CALLER_2300])

    hs._tx_over_response(last=False)
    assert (MK.demod_tones(io.sent[-1], 16)
            == VF.handshake_tones(_CALLED, VF.SESSION_OVER_RESPONSE_SHORT))

    hs.over_continue = VA.OVER_CONTINUE_GENERATED
    hs._tx_over_response(last=False)
    assert (MK.demod_tones(io.sent[-1], 32)
            == VF.handshake_tones(_CALLED, VF.SESSION_OVER_RESPONSE))

    hs.over_continue = VA.OVER_CONTINUE_CAPTURED
    hs._tx_over_response(last=False)
    heard = MK.demod_tone_pairs(io.sent[-1], VF.OVER_CONTINUE_NSYM)
    assert ([tuple(sorted(p)) for p in heard]
            == [tuple(sorted(p)) for p in VF.OVER_CONTINUE_CALLER_2300])
    assert hs.turn == VA._TURN_PEER, "answering an over must not move the turn"


def test_a_bandwidth_with_no_continue_frame_of_its_own_keys_the_generated_one(
        monkeypatch):
    """The 11-symbol tails travel between bandwidths and this frame does not.

    Three fetches on 2026-09-04 keyed BW2300's copy at BW2750: each took the
    gateway's first over, answered it, and drew the responder's own idle cadence
    to the timeout with nothing at its host, `gateway_rx_bytes` 0 in all three.
    A generated-answer run the same hour closed its delivery on the stale-state
    11-symbol burst, so that fallback stays; this one goes, and a bandwidth whose
    copy is not measured keys the frame it can build.

    All three bandwidths hold a copy today — BW2750's was measured off the cables
    the same week — so the table is emptied for the session's own bandwidth to
    ask the question the table can no longer pose.
    """
    monkeypatch.delitem(VF.OVER_CONTINUE_BY_CALLER, (VF.CONTROL_BURST_CALLER, "2750"))
    hs, io = _connected(bw="2750")
    hs.over_continue = VA.OVER_CONTINUE_CAPTURED
    hs._tx_over_response(last=False)
    assert (MK.demod_tones(io.sent[-1], 32, band=MK.band_for("2750"))
            == VF.handshake_tones(_CALLED,
                                  VF.for_bw(VF.SESSION_OVER_RESPONSE, "2750")))
    assert hs.turn == VA._TURN_PEER


def test_the_2750_session_keys_its_own_continue_frame_and_not_the_wide_ones():
    hs, io = _connected(bw="2750")
    hs.over_continue = VA.OVER_CONTINUE_CAPTURED
    hs._tx_over_response(last=False)
    heard = [tuple(sorted(p)) for p in
             MK.demod_tone_pairs(io.sent[-1], VF.OVER_CONTINUE_NSYM,
                                 MK.band_for("2750"))]
    assert heard == [tuple(sorted(p)) for p in VF.OVER_CONTINUE_CALLER_2750]
    assert heard != [tuple(sorted(p)) for p in VF.OVER_CONTINUE_CALLER_2300]


def test_the_idle_cadence_follows_the_turn():
    hs, io = _connected()
    hs.idle_keepalive()
    hs.idle_keepalive()
    hs.idle_keepalive()
    # The peer's turn is the peer's: a stock caller keys nothing on its own clock
    # there, and answers the peer's idles in the gaps they open instead
    # [see idle_keepalive, _answer_peer_idle].
    assert not [m for m in io.msgs if m.startswith("tx session-keepalive")], io.msgs
    # The turn-idle frame prompts for the answer that draws the next over, so it
    # is the frame for a turn with something still behind it.
    hs.turn, hs._txq = VA._TURN_OURS, [b"x" * 89]
    io.msgs.clear()
    hs.idle_keepalive()
    assert any("session-turn-idle" in m for m in io.msgs), io.msgs


def test_an_unanswered_last_over_is_queried_on_the_cadence():
    """An empty unsent queue does not mean the peer received the last block."""
    hs, io = _connected()
    hs.send(_overs(1))
    hs.on_rx_audio(_peer_control())
    original = io.sent[-1].copy()
    pending = hs._tx_pending
    assert hs.turn == VA._TURN_OURS and not hs._txq
    io.keys, io.sent = 0, []
    hs.idle_keepalive()
    assert hs.turn == VA._TURN_OURS
    assert io.keys == 1
    np.testing.assert_array_equal(
        io.sent[-1], MK.synth_burst(_CALLED, VF.SESSION_FINAL_ANSWER_QUERY))
    assert not np.array_equal(io.sent[-1], original)
    assert hs._tx_pending == pending and hs._tx_retries == 0
    assert hs._final_query_attempts == 1


def test_a_turn_still_carrying_data_is_not_given_back(monkeypatch):
    """Bounded feedback queries retain the turn and all unconfirmed bytes."""
    now = [100.0]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: now[0])
    hs, io = _connected()
    hs.send(_overs(3))
    hs.on_rx_audio(_peer_control())
    pending, queued = hs._tx_pending, list(hs._txq)
    assert hs.turn == VA._TURN_OURS and len(queued) == 2
    for attempt in range(VA._FINAL_QUERY_MAX):
        now[0] += 5.0  # Beyond the intermediate query's 4.5 s reply window.
        hs.idle_keepalive()
        assert hs.turn == VA._TURN_OURS
        assert hs._tx_pending == pending and hs._txq == queued
        assert hs._intermediate_query_attempts == attempt + 1
        assert hs._tx_retries == 0
    assert not any('retry DATA over' in m for m in io.msgs)


def test_the_released_turn_lets_the_gateways_next_over_through():
    """Release is only worth anything if the exchange goes on from there: the
    gateway's reply arrives as an ordinary over, reaches the host, and what the
    host hands back asks for the turn again."""
    hs, io = _connected()
    hs.send(_overs(1))
    hs.on_rx_audio(_peer_control())
    hs.on_rx_audio(_peer_control())  # Acknowledge the data, then release.
    assert hs.turn == VA._TURN_PEER
    io.msgs.clear()
    hs.on_rx_audio(_their_over(b"FC EM ABCDEF 100 90 0\r", index=1))
    assert any("delivered" in m for m in io.msgs), io.msgs
    hs.send(b"FS Y\r")
    assert hs.turn == VA._TURN_ASKED, "the queue had no way back onto the air"


def test_the_over_we_key_is_the_over_a_gateway_would_read():
    """End of the chain: what goes on the air has to be a DATA over, not merely
    audio our own decoder likes.

    So it is put through the recogniser that identifies a *gateway's* over —
    reference-column alignment, turbo decode, CRC-16 — and then checked not to be
    a link-setup, which is the one wideband frame that means something else.
    """
    from hfmodem.kestrel.rx import varahf2300 as RX

    hs, io = _connected()
    hs.send(b"Hello from kestrel. FF")
    hs.on_rx_audio(_peer_control())
    over = io.sent[-1]
    assert len(over) / MK.FS > 4.0, "a DATA over is a ~4.4 s wideband burst"

    hits, _bins = hs._rec3_alignment(over)
    assert hits >= VA._OVER_GUARD_MIN, (
        f"{hits}/{VA._OVER_REF_COLS} reference columns — a gateway would not "
        "recognise this as an over at all")
    fr = RX.decode_over(over, 0, len(over))
    assert fr.crc_ok
    assert not VF.is_link_setup(bytes(fr.frame_bytes))
    from hfmodem.kestrel.arq import phy as _phy
    assert _phy.vara_payload(bytes(fr.payload)) == b"Hello from kestrel. FF"


def test_an_unanswered_turn_request_is_repeated_and_then_given_up():
    """No recording holds a gateway answering a turn-request, so a peer that
    never does has to leave the link up rather than a queue stuck forever."""
    hs, io = _connected()
    hs.send(b"x" * 89)
    assert hs.turn == VA._TURN_ASKED
    for _ in range(VA._TURN_MAX_ASK):
        hs.idle_keepalive()
    assert hs.turn == VA._TURN_PEER
    assert hs._txq, "the queue was thrown away rather than deferred"
    assert any("giving the turn back" in m for m in io.msgs), io.msgs


def test_a_peer_that_takes_the_turn_back_ends_up_with_it():
    """The move the real Winlink flow makes every session, and the one state this
    machine had no way out of.

    Greeting, turn-request, grant, our proposal — and then the gateway takes the
    turn back to deliver the mail. Its overs arrive while our model still says the
    turn is ours, and nothing in an over names the station that keyed it, so the
    first is answered and concedes nothing. A run of them is the traffic
    disagreeing with the model, and the model is what gives way: the turn goes
    back and the queue does not. The over that yields it is still acknowledged —
    the acknowledgement is what the peer's release follows, and a turn-request in
    its place buys the ask and loses the release [see `_answer_data_over`] — so
    the ask is owed from there and goes out in the peer's own cadence
    [see `_reack_release`].

    Held the other way — the turn ours and no way out of it — the queue never
    drains, a later ``send`` keys a 5.4 s over on top of the gateway, and the idle
    cadence tells a station that is transmitting that we hold the turn and have
    nothing to put in it.
    """
    hs, io = _connected()
    hs.send(_overs(3))
    hs.on_rx_audio(_peer_control())                  # granted; over #1 keyed
    pending, queued = hs._tx_pending, list(hs._txq)
    assert hs.turn == VA._TURN_OURS
    assert len(hs._txq) == 2

    for i in range(VA._TURN_YIELD_AFTER - 1):
        io.msgs.clear()
        hs.on_rx_audio(_their_over(b"gateway mail %d\r" % i, index=i + 1))
        assert hs.turn == VA._TURN_OURS, "one over cost us the turn"
        assert any("per-over response" in m for m in io.msgs), io.msgs
        assert not any("session-turn-idle" in m for m in io.msgs), (
            f"told a station that is transmitting we hold the turn: {io.msgs}")

    io.msgs.clear()
    hs.on_rx_audio(_their_over(b"gateway mail last\r", index=VA._TURN_YIELD_AFTER))
    assert hs.turn == VA._TURN_PEER, (
        f"the turn never left us after {VA._TURN_YIELD_AFTER} overs: {io.msgs}")
    assert len(hs._txq) == 2, "the queue went with the turn"
    assert any("per-over response" in m for m in io.msgs), io.msgs
    assert hs._answer_owed == VA._OWED_RELEASE, io.msgs

    io.msgs.clear()
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_OVER_IDLE))
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_OVER_IDLE))
    assert hs.turn == VA._TURN_ASKED, (
        f"the ask never went out in the peer's cadence: {io.msgs}")
    assert any("session-turn-request" in m for m in io.msgs), io.msgs

    io.msgs.clear()
    hs.on_rx_audio(_peer_control())                  # the request is answered
    assert hs.turn == VA._TURN_OURS
    assert len(hs._txq) == 2  # Reacquiring the turn did not acknowledge over #1.
    assert hs._tx_pending == pending and hs._txq == queued
    assert hs._intermediate_query_for == pending
    assert hs._tx_pending[1] == 1 and hs._tx_retries == 0
    np.testing.assert_array_equal(io.sent[-1], MK.synth_burst(
        _CALLED, VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]))


def _their_idle_response() -> np.ndarray:
    """The frame both gateways keyed back at our turn-idle, ten times over."""
    return MK.synth_tones(VF.handshake_tones(_CALLED, VF.SESSION_IDLE_RESPONSE))


def test_the_gateways_answer_to_our_idle_frame_is_named_and_keys_nothing():
    """The state this station has ended every VARA session in.

    Holding the turn with nothing to put in it, the gateway answers each idle
    frame 0.13 s later — and the whole vocabulary had no name for what arrived,
    so the log said "not an MFSK handshake burst" about the one thing on the
    channel that was addressed to us. Naming it is the point; keying at it is not,
    because an answer to an answer is a 1.5 s ping-pong where the real VARA that
    recorded these sessions said nothing for the next 11.7 s.
    """
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    io.keys = 0
    hs.on_rx_audio(_their_idle_response())
    assert any("session-idle-response" in m for m in io.msgs), io.msgs
    assert not any("not an MFSK handshake burst" in m for m in io.msgs), io.msgs
    assert io.keys == 0, "answered an idle answer"
    assert hs.turn == VA._TURN_OURS


def test_an_idle_answer_queries_the_outstanding_over():
    """The peer is listening, but has not acknowledged the outstanding data."""
    hs, io = _connected()
    hs.send(_overs(2))
    hs.on_rx_audio(_peer_control())                  # granted; over #1 keyed
    pending, queued = hs._tx_pending, list(hs._txq)
    assert hs.turn == VA._TURN_OURS and len(hs._txq) == 1
    io.msgs.clear()
    hs.on_rx_audio(_their_idle_response())
    assert hs._tx_pending == pending and hs._txq == queued
    assert hs._intermediate_query_for == pending
    assert hs._tx_pending[1] == 1 and hs._tx_retries == 0
    np.testing.assert_array_equal(io.sent[-1], MK.synth_burst(
        _CALLED, VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]))
    assert len(hs._txq) == 1


def test_the_gateways_answer_is_not_read_as_a_turn_grant():
    """No recording holds it answering a turn-request — both sessions' requests
    drew a different burst — so it must not take a turn nobody granted."""
    hs, io = _connected()
    hs.send(b"x" * 89)
    assert hs.turn == VA._TURN_ASKED
    io.keys = 0
    io.msgs.clear()
    hs.on_rx_audio(_their_idle_response())
    assert hs.turn == VA._TURN_ASKED, "took the turn on a frame that never granted one"
    assert io.keys == 0
    assert hs._txq, "the queue went with a turn that was never given"
    assert any("session-idle-response" in m for m in io.msgs), io.msgs


def test_the_peer_answering_one_of_our_overs_puts_the_count_back():
    """The count is of overs *in a row* that our peer did not answer, so a link
    working normally never accumulates one.

    Without that, three overs scattered across a long session — a station we are
    not in session with keying on the frequency, or the segmenter handing over
    somebody else's traffic — would add up to a turn given away while our own peer
    was answering every over we keyed.
    """
    hs, _io = _connected()
    # Enough blocks that the queue outlasts the loop: an answered over with the
    # queue empty is a release, which is a different measurement  [see
    # test_each_answered_over_draws_the_next_one_and_then_the_release].
    hs.send(b"x" * 89 * 6)
    hs.on_rx_audio(_peer_control())                  # granted; over #1 keyed
    for i in range(VA._TURN_YIELD_AFTER + 1):
        hs.on_rx_audio(_their_over(b"not our peer %d\r" % i, index=1))
        assert hs.turn == VA._TURN_OURS, f"the turn went on over {i + 1}"
        hs.on_rx_audio(_peer_control())              # our own over is answered
        assert hs.turn == VA._TURN_OURS
    assert hs._into_our_turn == 0


# --------------------------------------------------------------------------- #
# What may be read as the peer's answer, measured on the population that matters.
def test_nothing_we_transmit_reads_as_the_peers_answer():
    """The runaway again, over the whole vocabulary rather than one frame.

    Every burst kestrel puts on the air comes back through its own receiver, and
    while we hold the turn any accepted burst keys the next over. Not one of the
    nine may be accepted; the peer's control burst must be.
    """
    hs, _io = _connected()
    for call, kind in [
            (_CALLED, VF.CR), (_CALLED, VF.CONNECT_RESPONSE),
            (_CALLED, VF.SESSION_CONFIRM), (_CALLED, VF.SESSION_KEEPALIVE_A),
            (_CALLED, VF.SESSION_KEEPALIVE_B), (_CALLED, VF.SESSION_OVER_RESPONSE),
            (_CALLED, VF.SESSION_OVER_RESPONSE_SHORT),
            (_MYCALL, VF.SESSION_TURN_REQUEST), (_MYCALL, VF.SESSION_TURN_IDLE)]:
        audio = MK.synth_burst(call, kind)
        assert not hs._peer_control_burst(audio), kind.name
        assert not hs._peer_idle_response(audio), kind.name
    assert hs._peer_control_burst(_peer_control())
    assert hs._peer_idle_response(_their_idle_response())
    assert not hs._peer_control_burst(_their_idle_response())


def _off_frequency(x: np.ndarray, carriers: int) -> np.ndarray:
    """``x`` as a station ``carriers`` off frequency would have put it on the air."""
    from scipy.signal import hilbert

    t = np.arange(len(x)) / MK.FS
    return np.real(hilbert(x) * np.exp(2j * np.pi * MK.carrier_to_hz(carriers) * t))


def _session_with_an_off_frequency_peer(carriers: int = -1):
    """A live session with a peer transmitting ``carriers`` off frequency, opened
    the way the air opens one: its connect-response is found at the shift it
    arrived on, and ``_peer_shift`` is what that measurement leaves behind."""
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300", mfsk_only=True)
    hs.originate(_CALLED, _MYCALL)
    resp = _off_frequency(MK.synth_burst(_CALLED, VF.CONNECT_RESPONSE), carriers)
    resp = np.concatenate([resp, np.zeros(MK.FS)])
    for i in range(0, len(resp), 4800):
        hs.on_rx_stream(resp[i:i + 4800])
    assert hs._peer_shift == carriers, (
        "the connect-response was not taken at the offset it arrived on")
    hs.on_rx_audio(_off_frequency(_peer_control(), carriers))     # step 5
    assert hs.state is VA.VaraState.CONNECTED
    return hs, io


def test_the_off_frequency_peers_control_burst_still_answers_our_turn_request():
    """One peer, one offset, two readers of the same burst.

    ``_peer_shift`` is how far off frequency the peer's transmissions arrive,
    measured off the connect-response that opened the session and carried from
    there. The connected-ack is read at that offset — it is the same eight carriers
    in the data phase as in step 5, keyed by the same station on the same dial —
    so an in-session control burst scored at zero is a gateway answering into a
    reader that cannot hear it: the turn is never granted, the queue never drains,
    and the log says the peer went quiet.
    """
    hs, io = _session_with_an_off_frequency_peer()
    hs.send(b"x" * 89)
    assert hs.turn == VA._TURN_ASKED
    io.msgs.clear()
    hs.on_rx_audio(_off_frequency(_peer_control(), hs._peer_shift))
    assert hs.turn == VA._TURN_OURS, (
        "the peer answered our turn-request one carrier low and the answer was "
        "scored as though it were centred")
    assert any("tx DATA over #1" in m for m in io.msgs), io.msgs


#: Where each gateway's answer to one of our turn-idles begins, to the second,
#: measured by template score over the whole of both recordings. Five in each
#: session, 0.112-0.150 s after our own burst ended, and nothing else in either
#: recording is this frame.
_IDLE_RESPONSES = {"NS0A_2300": (52.9, 66.1, 79.3, 92.5, 105.7),
                   "KC9GHZ_2300": (50.0, 63.2, 76.4, 89.6, 102.7)}


@corpora.requires_gateway_session
@pytest.mark.parametrize("session", sorted(_IDLE_RESPONSES))
def test_the_gateways_answer_is_found_in_the_recording_that_holds_it(session):
    """The arbiter: a real gateway's own symbols, off the air, both sessions.

    Read at the burst and one second either side of it, because a segmenter does
    not hand over burst-accurate audio — the recogniser locates the frame by its
    payload, so a window that holds most of it is enough.
    """
    path = corpora.OFFAIR / session / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air recording for {session} not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    hs, _io = _connected()
    hs.called = session.split("_")[0]
    for t in _IDLE_RESPONSES[session]:
        i = int(t * MK.FS)
        assert hs._peer_idle_response(x[i:i + int(1.6 * MK.FS)]), (
            f"{session}: the gateway's answer at {t} s was not recognised")


@corpora.requires_regress_fixtures
def test_the_gateways_answer_does_not_fire_on_audio_addressed_to_nobody():
    """The false-accept floor, on the population that matters: 33 real off-air
    recordings from four continents — VARA, PACTOR-1/2/3, ARDOP, FT8, WSPR and
    band noise, some through a rig and some through a websdr — none of it
    addressed to us. 2 s windows at 50% overlap, five callsigns each."""
    fixtures = sorted(corpora.REGRESS_FIXTURES.glob("*.wav"))
    if not fixtures:
        pytest.skip("shared regression corpus not present")
    hs, _io = _connected()
    w = int(2.0 * MK.FS)
    accepts, trials = [], 0
    for path in fixtures:
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() or 1.0)
        for i in range(0, max(1, len(x) - w), w // 2):
            for call in ("NS0A", "KC9GHZ", "W9SSJ", "KB9MMT", "KD9USW"):
                hs.called = call
                trials += 1
                if hs._peer_idle_response(x[i:i + w]):
                    accepts.append((path.name, round(i / MK.FS, 2), call))
    assert trials > 4000, f"only {trials} trials — the corpus is not all there"
    assert not accepts, f"{len(accepts)} of {trials} trials accepted: {accepts[:8]}"


@corpora.requires_gateway_session
@corpora.requires_clear_channel
def test_real_band_audio_is_not_read_as_the_peers_answer():
    """981 half-second windows of real off-air HF at 50% overlap — two gateway
    sessions and a verified-clear 40 m frequency through the same rig and codec.

    Two windows accept, 17.75 s and 18.00 s of the NS0A session, which is where
    that recording's one genuine gateway control burst sits (its preamble locks
    4/4 at 17.959 s). Nothing else in the three recordings accepts.
    """
    hs, _io = _connected()
    for path, allowed in ((corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav", (17.5, 18.5)),
                          (corpora.GATEWAY_SESSION / "rig_rx.wav", None),
                          (corpora.CLEAR_CHANNEL, None)):
        if not path.exists():
            continue
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() or 1.0)
        w = int(0.5 * MK.FS)
        bad = [round(i / MK.FS, 2) for i in range(0, len(x) - w, w // 2)
               if hs._peer_control_burst(x[i:i + w])
               and not (allowed and allowed[0] <= i / MK.FS <= allowed[1])]
        assert not bad, f"{path.name} accepted band audio at t = {bad} s"


_GRANT_WAV, _GRANT_CALL, _GRANT_WINDOWS = corpora.ONAIR_TURN_GRANTS


@pytest.mark.parametrize("start,end,granted", _GRANT_WINDOWS)
@corpora._requires(_GRANT_WAV, what="the 2026-08-23 KB3AC-10 session")
def test_the_turn_grant_is_found_on_the_stream_the_gate_never_opened_for(
        start, end, granted):
    """The deadlock that cost this station two gateways, off the tape it happened on.

    The gate opens at 2.0x its tracked floor, 6.02 dB, and everything KB3AC-10 put
    on the air that session stands +0.90 to +5.00 dB over it — the three DATA overs
    and all three answers to our turn-requests. The overs still reached the host,
    because the stream search owns them; the grant had no stream route at all, so
    `turn-request unanswered 3x — giving the turn back, 2 block(s) still queued`
    was logged with the gateway's answer already recorded.

    Fed the same audio the same way the transport feeds it, the turn is now taken
    in the turnaround it was granted in — and is not taken in the turnaround where
    the gateway keyed something else, nor over its greeting over.
    """
    x = corpora.wav_mono(_GRANT_WAV)
    x = x / 32768.0 if np.abs(x).max() > 1.5 else x
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", _GRANT_CALL, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn, hs._asked, hs._txq = VA._TURN_ASKED, 1, [b"x" * 99]
    # Where `AudioVaraIO.tx` leaves its cursor: past our own burst and the echo
    # guard, which on this session's changeovers is still inside the mute.
    window = x[int((start + 0.11) * MK.FS):int(end * MK.FS)]
    for i in range(0, len(window), MK.FS // 10):
        hs.on_rx_stream(window[i:i + MK.FS // 10])
    assert (hs.turn == VA._TURN_OURS) is granted, (
        f"turn={hs.turn} over {end - start:.1f} s from {start:.3f} s: " + str(io.msgs))
    if granted:
        assert any("tx DATA over #1" in m for m in io.msgs), io.msgs


_IDLE_WAV, _IDLE_CALL, _IDLE_STARTS = corpora.ONAIR_RESPONDER_IDLE


@corpora._requires(_IDLE_WAV, what="the 2026-08-23 KE8LVA session")
def test_the_responder_idle_is_found_in_the_recording_it_was_fitted_from():
    """The arbiter for (288, 683): a gateway's own symbols off the air, at every
    one of the thirteen keyings, located by payload the way a bracket is.

    The negative half is the same recording read with the same frame keyed to us
    instead — the reading `keyed_by` rejects — and the other two 2026-08-23 arms
    that reached CONNECTED, neither of whose gateways ever keys it.
    """
    x = corpora.wav_mono(_IDLE_WAV)
    x = x / (np.abs(x).max() or 1.0)
    kind = VF.SESSION_RESPONDER_IDLE
    want = np.asarray(VF.handshake_tones(_IDLE_CALL, kind), dtype=np.int32)
    mine = np.asarray(VF.handshake_tones(_MYCALL, kind), dtype=np.int32)
    for t in _IDLE_STARTS:
        i = int(t * MK.FS)
        seg = x[max(0, i - MK.FS // 4):i + int(1.8 * MK.FS)]
        heard = VA._payload_alignment(seg, kind, want)
        assert VF.recognize([int(v) for v in heard], _IDLE_CALL, kind), (
            f"the gateway's frame at {t} s was not recognised")
        assert not VF.recognize([int(v) for v in heard], _MYCALL, kind), (
            f"the frame at {t} s reads as keyed to us")
        assert (heard[1:] == mine[1:]).sum() <= 3


@corpora._requires(_IDLE_WAV, what="the 2026-08-23 KE8LVA session")
def test_the_responder_idle_is_named_and_is_not_progress():
    """A gateway keying every 3.4 s, read and answered but not banked.

    Nothing read the frame KE8LVA was transmitting on 2026-08-23, so the link
    closed on a station that had never stopped talking, and the fix was to name it
    — which this checks. Banking it as progress was the other half of that fix and
    is gone: the frame says the peer is transmitting, and a peer transmitting is
    what a stalled session looks like too. At the bench on 2026-08-31 a reply over
    that would not decode was followed by eighteen of these, each one zeroing the
    budget, and the session hung ~100 s with no route out but the mail timeout.

    What it DOES draw is the one burst stock keys back at it — 683 takes a
    `session-over-nak`, one per idle, in the gap the idle opens (2026-09-16, 45 of
    45 at both wide bandwidths)  [see _answer_peer_idle]. That answer is neither
    progress nor charged to the budget, which is what the two assertions above
    still pin.
    """
    x = corpora.wav_mono(_IDLE_WAV)
    x = x / (np.abs(x).max() or 1.0)
    hs, io = _connected()
    hs.called = _IDLE_CALL
    hs.turn = VA._TURN_PEER
    hs._since_progress = VA._MAX_WITHOUT_PROGRESS - 1
    window = x[int((_IDLE_STARTS[0] - 0.3) * MK.FS):int(_IDLE_STARTS[2] * MK.FS)]
    for i in range(0, len(window), MK.FS // 10):
        hs.on_rx_stream(window[i:i + MK.FS // 10])
    named = sum("session-responder-idle" in m for m in io.msgs)
    assert named, io.msgs
    assert hs._since_progress == VA._MAX_WITHOUT_PROGRESS - 1, (
        f"the peer's own idle cadence paid into our budget: {io.msgs}")
    assert io.keys == named, (
        f"{named} idles named and {io.keys} keyings: the answer is one burst per "
        f"idle, in the gap that idle opens", io.msgs)
    assert all("session-over-nak" in m
               for m in io.msgs if m.startswith("tx ")), io.msgs


@corpora._requires(_GRANT_WAV, what="the 2026-08-23 KB3AC-10 session")
def test_the_answer_search_does_not_fire_on_the_session_that_holds_no_answer():
    """The other side of it: KB3AC-10 keyed a responder-idle never, and 187 s of
    the band it was calling over does not read as one."""
    x = corpora.wav_mono(_GRANT_WAV)
    x = x / (np.abs(x).max() or 1.0)
    hs, io = _connected()
    hs.called = _GRANT_CALL
    hs.turn = VA._TURN_PEER
    for i in range(0, len(x), MK.FS // 10):
        hs.on_rx_stream(x[i:i + MK.FS // 10])
    assert not any("session-responder-idle" in m for m in io.msgs), io.msgs


def test_the_peers_answer_to_an_over_is_taken_off_the_stream_too():
    """The bracket route found this one and the stream route did not exist, so a
    gateway whose answer never opened the gate stalled with a queue in hand."""
    hs, io = _connected()
    hs.turn, hs._txq = VA._TURN_OURS, [b"x" * 99]
    io.msgs.clear()
    burst = _peer_control()
    x = np.concatenate([np.zeros(MK.FS // 2), burst, np.zeros(2 * MK.FS)])
    for i in range(0, len(x), MK.FS // 10):
        hs.on_rx_stream(x[i:i + MK.FS // 10])
    assert any("tx DATA over #1" in m for m in io.msgs), io.msgs


_ANS_WAV, _ANS_CALL, _ANS_REFUSED_AT, _ANS_OVER_AT = corpora.ONAIR_OVER_ANSWERED

# KE8LVA's own symbols in the two turnarounds of the 2026-08-26 12:59z session
# that the live log recorded as silence. Payload only: both frames open in the
# turnaround off one of our own keyings and our receiver was still muting through
# their first symbol, so the preamble each carries is not read from either.
_ANSWERS = [
    (VF.SESSION_TURN_RELEASE_RESPONDER,
     [70, 97, 40, 47, 70, 29, 40, 29, 58, 71, 36, 85, 96, 83, 62]),
    (VF.SESSION_RESPONDER_OVER_ANSWER,
     [46, 69, 52, 49, 70, 89, 96, 29, 84, 35, 34, 97, 78, 31, 60, 91,
      80, 41, 86, 47, 76, 81, 32, 59, 76, 61, 62, 59, 96, 65, 72]),
]


@pytest.mark.parametrize("kind,tones", _ANSWERS, ids=[k.name for k, _ in _ANSWERS])
def test_the_generator_reproduces_the_answers_a_gateway_keyed_at_us(kind, tones):
    assert VF.payload_bins(_ANS_CALL, kind) == tones


@pytest.mark.parametrize("kind,tones", _ANSWERS, ids=[k.name for k, _ in _ANSWERS])
def test_each_answer_is_the_only_reading_its_state_admits(kind, tones):
    """One session apiece, so the lattice the fit landed on is what is held here.

    A frame's payload is two generator draws a symbol, so the 16-symbol family
    sits on a 30-draw lattice and the 32-symbol one on a 62-draw lattice, and
    every member of each is at position 1 (mod its own). That is what makes the
    single pre-advance the discrete log returned a number a burst could carry
    rather than one of the seven in the millions beside it — and it holds only
    while the family it is a position in stays on that lattice, which is what
    this asserts. Read as keyed to us instead, the tones are chance.
    """
    step = 2 * kind.n_payload
    assert kind.preadv % step == 1
    family = {k.preadv for k in VF.BURSTS.values() if k.n_payload == kind.n_payload}
    assert all(p % step == 1 for p in family)
    assert kind.seed_off in {288, 289}, "keyed by the responder's own seed"
    assert kind.keyed_by == "called"
    ours = VF.payload_bins(_MYCALL, kind)
    assert sum(1 for a, b in zip(ours, tones) if a == b) <= 3
    for other in VF.BURSTS.values():
        if other is kind or other.n_payload != kind.n_payload:
            continue
        theirs = VF.payload_bins(_ANS_CALL, other)
        assert sum(1 for a, b in zip(theirs, tones) if a == b) <= 3


@corpora._requires(_ANS_WAV, what="the 2026-08-26 KE8LVA session")
@pytest.mark.parametrize(
    "kind,at,other",
    [(VF.SESSION_TURN_RELEASE_RESPONDER, _ANS_REFUSED_AT, VF.SESSION_DRAINED_RESPONDER),
     (VF.SESSION_RESPONDER_OVER_ANSWER, _ANS_OVER_AT, VF.SESSION_RESPONDER_IDLE)],
    ids=["responder-release", "over-answer"])
def test_the_answers_are_found_in_the_recording_they_were_fitted_from(kind, at, other):
    """The arbiter for both: a gateway's own symbols off the air, located by
    payload the way a bracket is, and rejected read as keyed to us."""
    x = corpora.wav_mono(_ANS_WAV)
    x = x / (np.abs(x).max() or 1.0)
    seg = x[int((at - 0.3) * MK.FS):int((at + 1.8) * MK.FS)]
    want = np.asarray(VF.handshake_tones(_ANS_CALL, kind), dtype=np.int32)
    heard = [int(v) for v in VA._payload_alignment(seg, kind, want)]
    assert VF.recognize(heard, _ANS_CALL, kind)
    assert not VF.recognize(heard, _MYCALL, kind)
    every = np.asarray(VF.handshake_tones(_ANS_CALL, other), dtype=np.int32)
    assert not VF.recognize([int(v) for v in
                             VA._payload_alignment(seg, other, every)],
                            _ANS_CALL, other), (
        f"{at} s reads as {other.name} as well as {kind.name}")


@corpora._requires(_ANS_WAV, what="the 2026-08-26 KE8LVA session")
def test_the_turn_grant_is_not_read_as_a_refusal():
    """The two answers to a turn-request are the same length off the same
    transmitter 12 s apart, and only the tones separate them."""
    x = corpora.wav_mono(_ANS_WAV)
    x = x / (np.abs(x).max() or 1.0)
    hs, io = _connected()
    hs.called = _ANS_CALL
    hs.turn = VA._TURN_ASKED
    grant = x[int(62.3 * MK.FS):int(64.2 * MK.FS)]
    assert hs._peer_drained(grant)
    assert not hs._peer_responder_release(grant)


def test_the_bracket_route_will_not_take_a_grant_out_of_a_short_bracket():
    """`_peer_drained` carries no length guard — it takes the first half of a
    grant as readily as the whole — so the route that acts on it owes one. A grant
    accepted while the frame is still arriving keys a 4.4 s DATA over across the
    tail of a half-duplex peer, which decodes none of it."""
    hs, io = _connected()
    grant = MK.synth_burst(_CALLED, VF.SESSION_DRAINED_RESPONDER)
    short = grant[:VA._SESSION_NEED - 1]
    assert hs._peer_drained(short), "the fragment this guards against is gone"

    hs.turn = VA._TURN_ASKED
    hs.on_rx_audio(short)
    assert hs.turn == VA._TURN_ASKED, "took the turn off a frame still arriving"
    assert not io.sent, f"keyed an over across the peer's tail: {io.msgs}"

    hs.on_rx_audio(grant)
    assert hs.turn == VA._TURN_OURS, io.msgs


def test_a_grant_named_a_block_behind_the_buffers_newest_audio_is_not_taken():
    """The 2026-09-03 22:01z fetch, and `220602Z` five minutes later.

    Our turn-request ended at 117.84 as the peer keyed a 1.39 s poll of its own;
    that poll closed at 119.23, the peer keyed again at 119.58, and the grant was
    named at 120.06 out of a buffer that by then held both. The frame was
    complete, so no last-symbol guard could see anything wrong with it — what was
    wrong with it was its age. The 4.4 s DATA over went out across the peer's
    second keying and the responder logged no RX bitrate for any of it.

    Over the 222 grants of the 2026-09-03/04 bench the 210 whose over the peer
    then read were named 0.00-0.37 s behind the newest audio, and the twelve whose
    over it did not read 0.91-2.33 s.
    """
    hs, io = _connected()
    hs.turn = VA._TURN_ASKED
    poll = MK.synth_burst(_CALLED, VF.SESSION_DRAINED_RESPONDER)
    again = MK.synth_burst(_CALLED, VF.SESSION_TURN_REQUEST)[:int(0.48 * MK.FS)]
    buf = np.concatenate([poll, np.zeros(int(0.35 * MK.FS)), again])
    assert hs._peer_drained(buf), "the frame the run named is still named"
    assert hs._grant_held > VA._GRANT_FRESH_S

    assert not hs._stream_grant(buf)
    assert hs.turn == VA._TURN_ASKED, "took the turn off a burst the peer keyed over"
    assert not io.sent, f"keyed an over across the peer's own keying: {io.msgs}"
    assert any("behind the newest audio" in m for m in io.msgs), io.msgs
    assert not len(hs._grant_buf), "the next ask is heard on audio recorded since"


def test_a_grant_still_in_the_turnaround_it_was_asked_in_is_taken():
    hs, io = _connected()
    hs.turn = VA._TURN_ASKED
    buf = np.concatenate([MK.synth_burst(_CALLED, VF.SESSION_DRAINED_RESPONDER),
                          np.zeros(int(0.3 * MK.FS))])
    assert hs._stream_grant(buf), io.msgs
    assert hs._grant_held < VA._GRANT_FRESH_S
    assert hs.turn == VA._TURN_OURS, io.msgs


@corpora._requires(_ANS_WAV, what="the 2026-08-26 KE8LVA session")
def test_the_over_answer_keeps_the_give_up_budget_off_a_gateway_that_answered():
    """`6 idle bursts and nothing back from KE8LVA` was logged over an answer that
    arrived 0.17 s after our own over ended. Fed the same audio the transport
    feeds, the stream route names it and puts the counter back."""
    x = corpora.wav_mono(_ANS_WAV)
    x = x / (np.abs(x).max() or 1.0)
    hs, io = _connected()
    hs.called = _ANS_CALL
    hs.turn = VA._TURN_OURS
    hs._since_progress = VA._MAX_WITHOUT_PROGRESS - 1
    window = x[int((_ANS_OVER_AT - 0.3) * MK.FS):int((_ANS_OVER_AT + 2.0) * MK.FS)]
    for i in range(0, len(window), MK.FS // 10):
        hs.on_rx_stream(window[i:i + MK.FS // 10])
    assert hs._since_progress == 0, io.msgs
    assert any("session-responder-over-answer" in m for m in io.msgs), io.msgs
    assert io.keys == 0, "no recording says what an answer to this frame is"


@corpora._requires(_ANS_WAV, what="the 2026-08-26 KE8LVA session")
def test_neither_answer_fires_on_the_stretch_that_holds_no_gateway_at_all():
    """The 110 s after the over-answer, which the live log read as ten of its
    twelve undecoded high-speed-level overs and which holds no gateway
    transmission at all."""
    x = corpora.wav_mono(_ANS_WAV)
    x = x / (np.abs(x).max() or 1.0)
    hs, io = _connected()
    hs.called = _ANS_CALL
    for kind, fn in ((VF.SESSION_TURN_RELEASE_RESPONDER, hs._peer_responder_release),
                     (VF.SESSION_RESPONDER_OVER_ANSWER, hs._peer_over_answer)):
        for t in np.arange(72.0, 178.0, 0.5):
            seg = x[int(t * MK.FS):int((t + 1.9) * MK.FS)]
            assert not fn(seg), f"{kind.name} accepted band noise at {t:.1f} s"


# --------------------------------------------------------------------------- #
# The peer's turn-request, and the release that answers it.
#
# Both frames are read off a real VARA's transmitter and neither is a round trip:
# the tones below are what came back off the cable, and the arbiter for the
# release is a station that is not ours at either end of the link.
#: Keyed to the CALLER, so read back against this station's own callsign.
_ASK_TONES = [
    90, 59, 94, 83, 42, 51, 88, 37, 96, 39, 86, 55, 54, 79, 96, 31,
    88, 59, 86, 75, 68, 45, 64, 31, 50, 49, 36, 67, 36, 43, 46]
_RESPONDER_IDLE_TONES = [
    50, 73, 78, 77, 64, 87, 52, 55, 90, 81, 40, 59, 82, 55, 46, 81,
    70, 45, 32, 95, 52, 69, 38, 69, 78, 93, 66, 63, 56, 63, 60]
_RESPONDER_RELEASE_TONES = [
    62, 67, 32, 69, 68, 77, 32, 69, 38, 65, 94, 49, 56, 97, 42, 49, 64]


def test_the_generator_reproduces_the_bench_responder_frames():
    """The three frames a stock VARA HF 4.9.0 keyed at this station on the bench.

    The turn-request is keyed to the CALLER, as both turn frames in the file are,
    which is what its SEED_OFF says as well: 1078 is 850 + 228, and 228 is the
    offset between every responder frame here and its initiator counterpart.
    """
    assert VF.payload_bins(_MYCALL, VF.SESSION_TURN_REQUEST_RESPONDER) \
        == _ASK_TONES
    assert VF.payload_bins("W1AW", VF.SESSION_RESPONDER_OVER_IDLE) \
        == _RESPONDER_IDLE_TONES
    assert VF.handshake_tones("W1AW", VF.SESSION_TURN_RELEASE_RESPONDER) \
        == _RESPONDER_RELEASE_TONES


def test_the_two_releases_are_one_stream_position_keyed_from_either_end():
    """What the naming error cost, stated as the arithmetic that exposed it.

    The responder's release sits at the caller's release's own lattice position
    with the responder's SEED_OFF, and that is why a gateway answering our first
    turn-request with it was read as a refusal for three months: 289 is to 61
    what 288 is to 60, and both are PREADV 391 on the 30-draw lattice a 16-symbol
    payload sits on.
    """
    ours, theirs = VF.SESSION_TURN_RELEASE, VF.SESSION_TURN_RELEASE_RESPONDER
    assert ours.preadv == theirs.preadv
    assert theirs.seed_off - ours.seed_off == 228
    assert ours.preamble == theirs.preamble == (62, 67)
    assert (ours.preadv - 1) % (2 * ours.n_payload) == 0
    req = VF.SESSION_TURN_REQUEST_RESPONDER
    assert req.seed_off - VF.SESSION_TURN_REQUEST.seed_off == 228
    assert req.preadv == VF.SESSION_TURN_REQUEST.preadv == 1
    assert req.keyed_by == VF.SESSION_TURN_REQUEST.keyed_by == "caller"


def test_the_ask_is_answered_by_releasing_the_turn():
    """A caller with an empty queue owes the peer the release, and keys it.

    Eleven of these went unanswered in the run of 2026-08-26 that had to be
    freed by hand, and twelve in each of three runs that moved no payload at all.
    """
    hs, io = _connected()
    hs.caller = _MYCALL
    ask = MK.synth_tones(VF.handshake_tones(
        _MYCALL, VF.SESSION_TURN_REQUEST_RESPONDER))
    hs.on_rx_audio(np.asarray(ask, float))
    assert hs.turn == VA._TURN_PEER
    heard = MK.demod_tones(io.sent[-1], 17)
    assert heard == VF.handshake_tones(_CALLED, VF.SESSION_TURN_RELEASE)


def test_a_queue_of_our_own_keeps_the_turn_through_an_ask():
    """The other half: the release follows our own drain, not the peer's asking."""
    hs, io = _connected()
    hs.caller = _MYCALL
    hs.turn = VA._TURN_OURS
    hs._txq.append(b"x" * 8)
    before = len(io.sent)
    hs.on_rx_audio(np.asarray(MK.synth_tones(VF.handshake_tones(
        _MYCALL, VF.SESSION_TURN_REQUEST_RESPONDER)), float))
    assert hs.turn == VA._TURN_OURS
    assert len(io.sent) == before


_ASK_WAV = corpora.GATEWAY_SESSION / "rig_rx.wav"
_ASK_AT = 15.88


@corpora._requires(_ASK_WAV, what="the KC9GHZ 2300 session")
def test_a_gateway_keyed_the_ask_at_us_too_and_nowhere_else_in_the_session():
    """The frame off somebody else's transmitter, and the session's own negative.

    The bench is one responder and one callsign pair, which is what makes this
    occurrence worth its runtime: a Winlink RMS gateway on 40 m keyed it once,
    at 30 of the 30 payload tones its own alignment finds comparable, and this
    station answered it with silence. Swept every half second over the rest of
    the recording, nothing else in it reaches the recogniser at all.
    """
    x = corpora.wav_mono(_ASK_WAV)
    x = x / (np.abs(x).max() or 1.0)
    hs, _io = _connected()
    hs.caller = _MYCALL
    kind = VF.SESSION_TURN_REQUEST_RESPONDER
    frame = (len(kind.preamble) + kind.n_payload - 1) * MK.HOP + MK.STRIDE
    at = int(_ASK_AT * MK.FS)
    assert hs._peer_wants_turn(x[at:at + frame])
    took = [round(i / MK.FS, 2) for i in range(0, len(x) - frame, MK.FS // 2)
            if abs(i - at) > MK.FS and hs._peer_wants_turn(x[i:i + frame])]
    assert not took, f"the ask was also read at t = {took} s"


def _keyed_regions(a: np.ndarray, thr: float = 0.12, lo_s: float = 0.15):
    """Envelope regions of a one-way cable — every one of them is that end's."""
    win = MK.FS // 50
    env = np.sqrt(np.convolve(a * a, np.ones(win) / win, "same"))
    on = env > env.max() * thr
    edges = np.diff(on.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if on[0]:
        starts.insert(0, 0)
    if on[-1]:
        ends.append(len(a))
    return [(x, y) for x, y in zip(starts, ends) if (y - x) / MK.FS >= lo_s]


def _control_tails(session: str) -> dict:
    """The seven state symbols each end of one two-VARA session keyed.

    One recording per direction, so which station keyed a burst is a property of
    the file. The burst is 0.47 s and opens on the fixed four-symbol preamble,
    which is what locates it inside a region the envelope bracketed.
    """
    pre = [tuple(sorted(p)) for p in VF.CONNECTED_ACK_PREAMBLE]
    out = {}
    for path, who in zip(corpora.control_burst_sides(session),
                         ("caller", "responder")):
        a = corpora.wav_mono(path)
        a = a / max(float(np.abs(a).max()), 1e-9)
        tails = []
        for start, end in _keyed_regions(a):
            if not 0.40 <= (end - start) / MK.FS <= 0.55:
                continue
            seg = a[max(0, start - 2880):end + MK.NFFT]
            best = (-1, ())
            for off in range(0, min(3000, max(1, len(seg) - MK.NFFT)), 8):
                pairs = [tuple(sorted(p)) for p in
                         MK.demod_tone_pairs(seg[off:], VF.CONNECTED_ACK_NSYM)]
                hit = sum(1 for x, y in zip(pairs, pre) if x == y)
                if hit > best[0]:
                    best = (hit, tuple(pairs[len(pre):]))
            if best[0] == len(pre):
                tails.append(best[1])
        out[who] = tails
    return out


@corpora.requires_control_burst_key
def test_the_control_burst_tail_is_keyed_to_the_caller_and_not_to_the_role():
    """Eight sessions, one variable: who called whom.

    Three callees against one caller leave both tails where they were; one
    changed caller moves both, and no two callers share a single symbol at a
    single position. Role picks one of a link's two bursts and does not name
    either — which is what the three recordings of a single pair could not
    separate, and what a constant carried on that reading would have got wrong at
    every gateway this station ever answers.

    One symbol per tail is allowed to disagree between sessions with the same
    caller: its two carriers sit within 1% of each other in the recording and the
    stronger-second rule ranks them either way.
    """
    seen: dict = {}
    for session, caller, _called in corpora.CONTROL_BURST_CALLER_KEY:
        tails = _control_tails(session)
        for who in ("caller", "responder"):
            assert tails[who], f"{session}: no {who} control burst located"
            assert len(set(tails[who])) == 1, (
                f"{session}: the {who} keyed more than one tail")
            seen.setdefault((caller, who), []).append((session, tails[who][0]))

    for (caller, who), rows in seen.items():
        ref = rows[0][1]
        for session, tail in rows[1:]:
            differ = sum(1 for x, y in zip(ref, tail) if x != y)
            assert differ <= 1, f"{caller} {who}: {session} differs in {differ}"

    for who in ("caller", "responder"):
        held = {c: rows[0][1] for (c, w), rows in seen.items() if w == who}
        for a in held:
            for b in held:
                if a < b:
                    shared = sum(1 for x, y in zip(held[a], held[b]) if x == y)
                    assert shared == 0, f"{a}/{b} {who} share {shared} symbols"


@corpora.requires_control_burst_key
def test_the_pair_this_station_keys_is_the_pair_of_a_link_W9SSJ_called():
    """The two constants against a recording of the link they belong to.

    One symbol of slack, and the same one: the caller's symbol 4 carries carriers
    61, 80 and 29 within 1% of each other and the responder's symbol 1 carries 80,
    97 and 34, so whichever alignment a reader lands on decides which of them
    ranks second. Everything else is exact.
    """
    tails = _control_tails("20260826-214913-pair1ctl")
    held = VF.control_bursts("W9SSJ", "2300")
    for who, burst in zip(("caller", "responder"), held):
        ref = tuple(tuple(sorted(p))
                    for p in burst[len(VF.CONNECTED_ACK_PREAMBLE):])
        differ = sum(1 for x, y in zip(ref, tails[who][0]) if x != y)
        assert differ <= 1, f"{who}: {differ} symbols differ from the recording"


def test_answering_someone_elses_call_says_the_tail_is_another_links():
    hs, io = _connected()
    hs.role, hs.caller = "responder", "KE8LVA"
    hs._tx_control_burst()
    assert any("another link's tail" in m for m in io.msgs)


# --------------------------------------------------------------------------- #
# What the log says a session read, and what it says about the turnaround.


@corpora._requires(_IDLE_WAV, what="the 2026-08-23 KE8LVA session")
def test_one_gateway_burst_is_named_once():
    """Ten keyings on the tape have to reach the log ten times.

    A frame this end does not key back at leaves its own tail on the stream, and
    the buffer that named it starts again on that tail: two accepts a burst, the
    second out of fourteen-odd symbols of a frame already read. Over this stretch
    it was 20 lines for 10 bursts, and KB3AC-10's fifteen responder-idles of
    2026-08-30 reached the session report as thirty.

    The tape's first ten keyings and no further: they run unbroken at 3.4 s, and
    what follows holds a transmission of our own. Two seconds past the last of
    them, because a frame is named when its own last symbol has arrived — the
    gap it opens is what a re-acknowledgement is keyed into, and how much of that
    gap is left cannot be read off a frame still on the air  [see
    `_peer_responder_idle`, `_reack`].
    """
    x = corpora.wav_mono(_IDLE_WAV)
    x = x / (np.abs(x).max() or 1.0)
    hs, io = _connected()
    hs.called = _IDLE_CALL
    hs.turn = VA._TURN_PEER
    window = x[int((_IDLE_STARTS[0] - 0.5) * MK.FS):
               int((_IDLE_STARTS[9] + 2.0) * MK.FS)]
    for i in range(0, len(window), MK.FS // 10):
        hs.on_rx_stream(window[i:i + MK.FS // 10])
    named = [m for m in io.msgs if "session-responder-idle" in m]
    assert len(named) == 10, (len(named), io.msgs)


def test_the_close_line_says_what_the_session_read():
    """`6 idle bursts and nothing this build can read back from KB3AC-10` closed a
    link whose log held fifteen frames named from that gateway, and it reads as
    the gateway having gone silent.

    The counter spans the run of bursts since the session last moved, never the
    session, so the session's own count goes out with it and no reader can take
    the one for the other.
    """
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    x = np.concatenate([np.zeros(MK.FS // 2), _peer_control(), np.zeros(2 * MK.FS)])
    for i in range(0, len(x), MK.FS // 10):
        hs.on_rx_stream(x[i:i + MK.FS // 10])
    assert hs.progress > 0, io.msgs
    io.msgs.clear()
    hs._since_progress = VA._MAX_WITHOUT_PROGRESS
    hs.idle_keepalive()
    closed = [m for m in io.msgs if "closing the stalled link" in m]
    assert len(closed) == 1, io.msgs
    assert f"{hs.progress} progress event(s)" in closed[0], closed[0]
    assert str(VA._MAX_WITHOUT_PROGRESS) in closed[0], closed[0]


def test_the_release_is_logged_with_the_audio_it_was_held_behind():
    """The one thing arm 11 of 2026-08-30 could not answer from its own log.

    The fix that answers a release promptly flew once and the log recorded the
    read and the keying with nothing between them, so the figure had to come off
    the recording afterwards. What this end owns is the audio it was already
    sitting on when it named the frame, and that is what goes out.
    """
    hs, io = _connected()
    hs.turn = VA._TURN_PEER
    burst = MK.synth_burst(_CALLED, VF.SESSION_TURN_RELEASE_RESPONDER)
    for held in (0, MK.FS // 4):
        assert hs._peer_responder_release(
            np.concatenate([burst, np.zeros(held)]))
        assert hs._grant_held == pytest.approx(held / MK.FS, abs=0.02)
    io.msgs.clear()
    x = np.concatenate([burst, np.zeros(3 * MK.FS)])
    for i in range(0, len(x), MK.FS // 10):
        hs.on_rx_stream(x[i:i + MK.FS // 10])
    line = [m for m in io.msgs if "turn-release-responder" in m]
    assert len(line) == 1, io.msgs
    assert re.search(r"named [-+]\d\.\d{3} s from its last symbol", line[0]), line[0]
