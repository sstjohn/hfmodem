# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the caller keys during the peer's turn.

States measured 2026-09-16, each a keying the old build got wrong while the turn
was the peer's: a held ACK left 11.6 s late, a keepalive on our own clock, and a
final over retained behind the turn-request that read it.
"""
import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_vara_giveup import _connected, _CALLED, _MYCALL, _IO

kc = corpora.harness("kestrel_connect")


def _peer_turn(bw="2300"):
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw=bw)
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    return hs, io


def _idle(hs, kind):
    """Drive one of the peer's idle bursts through the stream, on this session's
    own alphabet, with the lead a stock caller answers it after."""
    audio = np.concatenate([MK.synth_burst(_CALLED, VF.for_bw(kind, hs.bw)),
                            np.zeros(int(0.13 * MK.FS))])
    for at in range(0, len(audio), VA._STREAM_BLOCK):
        hs.on_rx_stream(audio[at:at + VA._STREAM_BLOCK])


# --- A held answer is released on the second fresh idle under it ---------------

def _hold(hs, kind=VF.SESSION_RESPONDER_OVER_IDLE):
    hs._held_answer = (True, False)          # a last over, no release owed
    hs._held_samples = hs._held_idles = hs._held_idle_at = 0
    hs._idle_pair_seen = False
    hs._idle_kind = kind


def _idle_under_hold(hs, gap_s):
    """One named idle under the hold, ``gap_s`` of held audio behind the last."""
    hs._a_gap()
    hs._release_held_answer(int(gap_s * MK.FS))


def test_a_second_idle_under_the_hold_releases_the_ack_before_the_ceiling():
    hs, io = _peer_turn()
    _hold(hs)

    # First fresh idle under the hold: counted, and not released — it can be the
    # tail of an over that is genuinely still mid-window  [see _a_gap].
    _idle_under_hold(hs, 3.5)
    assert hs._held_answer is not None, "released on the first idle"
    assert not io.sent

    # Second fresh idle a cadence later: two emissions, so the peer has finished
    # its window and is waiting for the ACK.
    _idle_under_hold(hs, 3.5)
    assert hs._held_answer is None, "held the ACK to the _ANSWER_HOLD_MAX ceiling"
    assert io.sent, "the held ACK never went out on the second idle"
    assert any("window has ended" in m for m in io.msgs)
    assert not hs._owed_block, "a positive ACK, not a NAK-retained window"


def test_two_namings_of_one_emission_do_not_release_the_ack():
    """The recogniser is named off a third of the frame and can take an over-idle
    out of a block it is sitting in, so a pair closer than the peer's own cadence
    is one emission read twice  [see _a_gap, _IDLE_PAIR_MIN_S]."""
    hs, io = _peer_turn()
    _hold(hs)
    _idle_under_hold(hs, 0.5)
    _idle_under_hold(hs, 0.5)
    assert hs._held_idles == 2
    assert hs._held_answer is not None, "released on one emission named twice"
    assert not io.sent
    # AND THE VERDICT DOES NOT CHANGE AS AUDIO PILES UP. The pair was one
    # emission; a block arriving 3 s later does not make it two, so nothing may
    # release until a genuinely later idle is NAMED  [see _a_gap].
    hs._release_held_answer(int(3.5 * MK.FS))
    assert hs._held_answer is not None, (
        "one emission named twice released the ACK once 3 s of audio had passed")
    assert not io.sent
    _idle_under_hold(hs, 3.5)                # a real second emission at last
    assert hs._held_answer is None
    assert io.sent


def test_the_released_window_is_closed_and_not_left_open():
    """Left open, the window's bodies join the NEXT over's and a peer repeat of
    that over reaches the host twice  [see _finish_delivery_window]."""
    hs, io = _peer_turn()
    a = phy.vara_body(b"A" * 89, hs.caller)
    hs._deliver([a], hold=True)
    _hold(hs)
    assert hs._window_bodies
    _idle_under_hold(hs, 3.5)
    _idle_under_hold(hs, 3.5)
    assert hs._held_answer is None
    assert not hs._window_bodies, "the answered window was left open"


def test_the_ceiling_still_backstops_a_window_that_only_ever_idled_once():
    hs, io = _peer_turn()
    _hold(hs)
    hs._a_gap()                              # one idle only
    hs._release_held_answer(VA._ANSWER_HOLD_MAX)
    assert hs._held_answer is None
    assert io.sent, "one idle at the ceiling is still a positive ACK"


# --- The peer's turn keys nothing on our own clock -----------------------------

def test_the_peers_turn_keys_nothing_on_our_own_clock():
    hs, io = _peer_turn()
    for _ in range(VA._MAX_WITHOUT_PROGRESS):        # every tick up to the close
        hs.idle_keepalive()
        assert hs.state is VA.VaraState.CONNECTED
        assert not io.sent, "keyed a keepalive on our own clock in the peer's turn"
    hs.idle_keepalive()                              # the silence budget is spent
    assert hs.state is VA.VaraState.DISCONNECTED, (
        "a silent peer was not closed on the give-up budget")


def test_the_peers_poll_is_still_answered_though_our_own_clock_is_silent():
    """The one prompted burst in the peer's turn, and the distinction the silence
    turns on: a poll is the peer asking us to speak, our own clock is not.

    A stock responder handed the channel with a reply queued keys its control
    burst every 2.3 s and releases only once a burst of ours lands in the gap
    behind one — 5 of 5 on 2026-09-03  [see _took_poll]. Silencing the cadence
    took that answer away with it; the flag is what puts it back.
    """
    hs, io = _peer_turn()
    hs._released = True                      # the poll only follows our release
    hs.idle_keepalive()
    assert not io.sent, "keyed on our own clock in the peer's turn"

    hs._took_poll()
    assert len(io.sent) == 1, "the peer's poll drew no answer"
    assert any("control-burst poll" in m for m in io.msgs), io.msgs
    assert any("session-keepalive-a" in m for m in io.msgs), io.msgs


def test_a_down_transmitter_does_not_stall_the_silence_budget():
    """`tx_went_out` is decided at key-up and is stale outside a keyed region, so
    a latched refusal must not hold the budget at zero on a link nothing is
    answering  [see VaraIO.tx_went_out]."""
    hs, _ = _connected(transmits=False)
    hs.turn = VA._TURN_PEER
    for _ in range(VA._MAX_WITHOUT_PROGRESS):
        hs.idle_keepalive()
    assert hs._since_progress == VA._MAX_WITHOUT_PROGRESS
    hs.idle_keepalive()
    # The close is begun; a rig that cannot key cannot complete it on the air.
    assert hs.state is not VA.VaraState.CONNECTED


# --- The peer's idle draws one answer in the gap it opens ----------------------

def test_a_745_over_idle_draws_one_reactive_keepalive_a():
    hs, io = _peer_turn()
    _idle(hs, VF.SESSION_RESPONDER_OVER_IDLE)
    assert len(io.sent) == 1, "the 745 idle drew none, or more than one, keepalive"
    assert any("session-keepalive-a" in m for m in io.msgs)
    assert any("session-responder-over-idle" in m for m in io.msgs)


def test_a_683_idle_draws_one_over_nak_at_both_wide_bandwidths():
    for bw in ("2300", "2750"):
        hs, io = _peer_turn(bw)
        _idle(hs, VF.SESSION_RESPONDER_IDLE)
        assert len(io.sent) == 1, f"BW{bw}: the 683 idle drew {len(io.sent)} bursts"
        assert any("session-over-nak" in m for m in io.msgs), f"BW{bw}: {io.msgs}"


def test_an_answered_idle_is_neither_progress_nor_charged_to_the_budget():
    """The budget counts the peer's silence, and a frame we read and answered is
    the opposite of silence; charging it closed a live link in twenty seconds at
    the peer's own idle rate. Nor is it progress: a peer that only idles is what a
    stalled session looks like  [see _answer_peer_idle]."""
    hs, _ = _peer_turn()
    hs._since_progress = 2
    before = hs.progress
    _idle(hs, VF.SESSION_RESPONDER_OVER_IDLE)
    assert hs._since_progress == 2, "an answered idle was charged to the budget"
    assert hs.progress == before, "an idle was read as progress"


def test_the_answered_gap_is_not_keyed_into_twice():
    """`_keyed_on_peer_burst` is what stops the idle tick putting a second burst
    into a gap the stream already answered. Driven with NOTHING owed, so it is
    `_answer_peer_idle` that sets the flag and not the re-ack ladder."""
    hs, io = _peer_turn()
    assert hs._answer_owed is None
    _idle(hs, VF.SESSION_RESPONDER_OVER_IDLE)
    assert len(io.sent) == 1 and hs._keyed_on_peer_burst
    hs._answer_owed = VA._OWED_OVER      # a rung is owed as the tick comes round
    hs.idle_keepalive()
    assert len(io.sent) == 1, "keyed a second burst into the answered gap"
    # And the flag is spent, so the NEXT tick may key.
    hs.idle_keepalive()
    assert len(io.sent) == 2


def test_the_idle_answer_does_not_defer_the_give_up_tick():
    """`mail_session` restarts its keepalive deadline whenever
    `(progress, idle_keyed)` moves, so an answer counted there pushes the give-up
    tick back by a cadence — and the answers go out at the PEER's rate. Modelled
    against a 745 every 3.5 s that held the link up for 115 transmissions in
    400 s, which is the 2026-08-16 failure the budget exists to stop."""
    hs, io = _peer_turn()
    before = (hs.progress, hs.idle_keyed)
    for _ in range(4):
        _idle(hs, VF.SESSION_RESPONDER_OVER_IDLE)
    assert hs.idle_answers == 4, "the answers were not counted at all"
    assert (hs.progress, hs.idle_keyed) == before, (
        "an idle answer moved the pair mail_session restarts its clock on")


def test_the_driver_still_ticks_while_a_fast_peer_cadence_is_answered(monkeypatch):
    """The failure this counter split exists to stop, driven through the real
    `mail_session` loop rather than modelled.

    That loop restarts its keepalive deadline whenever `(progress, idle_keyed)`
    moves. With the idle answer counted in `idle_keyed`, a peer idling faster than
    `keepalive_s` deferred the tick for ever: 0 ticks, `_since_progress` at 0, and
    the link held to the mail timeout while transmitting at the peer's rate.
    """
    idle = np.concatenate([
        MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_OVER_IDLE),
        np.zeros(int(0.13 * MK.FS))])

    class _Io(VA.VaraIO):
        def __init__(self):
            self.now = 0.0
            self.next_idle = 3.5          # the peer's cadence, faster than ours
            self.keyed: list[float] = []
            self.msgs: list[str] = []

        def key(self, on):
            if on:
                self.keyed.append(self.now)

        def tx(self, samples): ...

        def log(self, msg): self.msgs.append(msg)

        def next_rx_burst(self, timeout, hs=None):
            self.now += timeout
            if self.now >= self.next_idle:
                self.next_idle += 3.5
                for at in range(0, len(idle), VA._STREAM_BLOCK):
                    hs.on_rx_stream(idle[at:at + VA._STREAM_BLOCK])
            return None

    class _Waiting:
        done = False

    io = _Io()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    monkeypatch.setattr(kc.time, "time", lambda: io.now)
    kc.mail_session(hs, io, _Waiting(), timeout=400.0, keepalive_s=10.0)

    assert hs.idle_answers, "the peer's idles were never answered at all"
    assert hs.state is not VA.VaraState.CONNECTED, (
        f"answered {hs.idle_answers} idles over {io.now:.0f} s and the link never "
        f"closed: the give-up tick was deferred by our own answers", io.msgs[-4:])
    assert io.now < 400.0, "the loop sat out the whole mail timeout"


def test_a_stalled_gateway_still_closes_while_its_idles_are_answered():
    """The whole point of the counter split: answering the cadence must not hold a
    dead exchange open. Ticks and idles interleaved, as the driver runs them."""
    hs, io = _peer_turn()
    for _ in range(VA._MAX_WITHOUT_PROGRESS):
        _idle(hs, VF.SESSION_RESPONDER_OVER_IDLE)
        assert hs.state is VA.VaraState.CONNECTED
        hs.idle_keepalive()
    assert hs._since_progress == VA._MAX_WITHOUT_PROGRESS, (
        f"answered {hs.idle_answers} idles and the budget stood at "
        f"{hs._since_progress}")
    _idle(hs, VF.SESSION_RESPONDER_OVER_IDLE)
    hs.idle_keepalive()                      # the budget is spent
    assert hs.state is VA.VaraState.DISCONNECTED, (
        f"answered {hs.idle_answers} idles and never spent the give-up budget")


def test_holding_the_turn_the_peers_idle_draws_no_keepalive():
    """Holding the turn, the burst due in the peer's gap is our own DATA over or
    the grant we are reading for — never a keepalive on top of either."""
    for turn in (VA._TURN_OURS, VA._TURN_ASKED):
        hs, io = _peer_turn()
        hs.turn = turn
        _idle(hs, VF.SESSION_RESPONDER_OVER_IDLE)
        assert not io.sent, f"turn={turn}: keyed a keepalive while holding the turn"


# --- A turn-request behind our final short over grants the turn ----------------

def _final_over(bw="2300"):
    hs, io = _connected()
    hs.bw = bw
    hs.turn = VA._TURN_OURS
    hs.send(b"only block")
    assert hs._tx_pending is not None and not hs._txq
    assert phy.over_is_last(hs._tx_pending[0], hs.caller)
    hs._stream_owns_turn_request = True       # the stream's own complete frame
    return hs, io


def test_a_turn_request_behind_our_final_over_solicits_confirmation_at_2300():
    hs, io = _final_over()
    old = hs._tx_pending
    assert hs._took_turn_request(raw=True)
    assert hs._tx_pending == old and hs.turn == VA._TURN_OURS
    assert hs._final_query_attempts == 1
    assert len(io.sent) == 2


def test_the_k0si_turn_request_delays_are_inside_the_answer_window():
    """The case the bound exists for. On the K0SI 40 m tape our final over's last
    sample is at 95.648 s and the responder's two turn-requests at 101.262 and
    104.922 s — +5.61 s and +9.27 s. A bound borrowed from the query reply
    (2.0 s) never fires on either  [docs/protocols/vara/20-peer-turn-measurements.md]."""
    for delay in (5.61, 9.27):
        hs, io = _final_over()
        hs._pending_answer_at -= delay
        assert hs._took_turn_request(raw=True), (
            f"a turn-request {delay} s behind our final over was not read as "
            f"its answer")
        assert hs._tx_pending is not None and hs._final_query_attempts == 1


def test_a_stale_turn_request_does_not_retire_the_final_over():
    """A responder asks for the turn on its own schedule too, and a final over
    that faded draws exactly that ask a cadence later  [see _final_over_was_read]."""
    hs, io = _final_over()
    original = hs._tx_pending
    hs._pending_answer_at -= VA._TURN_REQUEST_ANSWER_S + 1.0
    assert hs._took_turn_request(raw=True)  # Query; the request alone proves nothing.
    assert hs._tx_pending == original, "retired the final over on a stale request"
    assert hs.turn == VA._TURN_OURS


def test_a_bracket_turn_request_does_not_retire_the_final_over():
    hs, io = _final_over()
    original = hs._tx_pending
    hs._stream_owns_turn_request = False
    assert hs._took_turn_request()  # Query, without retiring anything.
    assert hs._tx_pending == original, "retired on a bracket's guess at the frame"


def test_a_declined_release_retains_the_final_over_and_owes_no_release():
    """Retiring behind a release the transport refused drops the last block of the
    message on a burst the peer never heard.

    And the release must not be left OWED: `_release_turn` owes it on every
    refusal, and the `_release_owed` branch runs ahead of the retry — it would
    hand the turn away with the over still pending, stranding it where nothing
    retries it. The over comes first  [see idle_keepalive]."""
    hs, io = _final_over()
    original = hs._tx_pending
    io.transmits = False
    assert not hs._took_turn_request(raw=True)
    assert hs._tx_pending == original, "retired the over behind a declined release"
    assert hs.turn == VA._TURN_OURS
    assert not hs._release_owed, "owed a release with the final over still pending"

    # The next tick retries the over rather than giving the turn away under it.
    io.transmits = True
    io.sent.clear()
    hs.idle_keepalive()
    assert hs.turn == VA._TURN_OURS, "handed the turn away with the over pending"
    assert io.sent, "the tick keyed nothing at all"
    assert any("final-answer query" in m for m in io.msgs), io.msgs
    assert hs._tx_pending == original, "the retry lost the pending over"


def test_a_turn_request_behind_an_intermediate_over_still_retains_the_turn():
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b"A" * phy.payload_size("2300") + b"second block")   # two overs
    assert hs._txq, "the second block should still be queued"
    hs._stream_owns_turn_request = True
    assert not hs._took_turn_request(raw=True)
    assert hs._tx_pending is not None
    assert hs.turn == VA._TURN_OURS
    assert any("retaining the turn" in m for m in io.msgs)
