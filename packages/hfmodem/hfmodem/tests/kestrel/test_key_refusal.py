# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A key-up the rig refused must not be followed by a burst.

`Rig.key` returns whether the transmitter went up, and for one release nothing
read it: `AudioVaraIO.key` called it and dropped the bool on the floor. `VaraIO`
declares `key` as returning None, so both state machines key, transmit and log the
burst as sent whatever the rig said — a full connect-request played into a
transmitter that is down, on every burst of the attempt, because a refused key-up
does not retire the rig and nothing else in the loop notices.

The refusal knowledge belongs to the IO that owns the rig, so that is where it is
answered: the burst becomes silence and the attempt ends through the handshake's
own timeout and retry ladder. Everything here runs against a rig object and a
sound card the test provides — no device is opened and nothing is keyed.
"""
from __future__ import annotations

import sys

import numpy as np

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_ofdm as OF
from hfmodem.kestrel.vara.vara_arq import VaraStationHandshake
from hfmodem.tests.kestrel import corpora

kc = corpora.harness("kestrel_connect")
oa = corpora.harness("onair_session")

MYCALL = "W9SSJ"
GATEWAY = "K9WRA"


class _Rig:
    """The keying surface `AudioVaraIO` uses, with the key-up under the test's hand.

    `refuse` is the rig that would not key: `Rig._key_locked` exhausts its budget,
    logs NOT TRANSMITTING and drops PTT to be sure, then returns False. It does
    *not* retire the rig — the transmitter is down and confirmed down, so there is
    nothing here for the `retired` guard to catch.
    """

    def __init__(self, armed: bool = True, refuse: bool = True):
        self.armed = armed
        self.refuse = refuse
        self.retired = False
        self.ptt: list[bool] = []

    def key(self, on: bool) -> bool:
        self.ptt.append(bool(on))
        if not self.armed:
            return not on            # the arm gate's own answer [vara_rig_bridge.Rig.key]
        return not (on and self.refuse)


class _Out:
    """A transmit stream that records what it was handed, in no time at all."""

    latency = 0.0

    def __init__(self, card: _Card):
        self._card = card

    def start(self): ...
    def write(self, block): self._card.played.append(np.asarray(block, float))
    def stop(self): ...
    def close(self): ...


class _In:
    def start(self): ...
    def stop(self): ...
    def close(self): ...


class _Card:
    """Enough of sounddevice to build the transport, keeping every transmission."""

    def __init__(self):
        self.played: list[np.ndarray] = []

    def query_devices(self, name): return {"max_output_channels": 2}
    def InputStream(self, **kw): return _In()
    def OutputStream(self, **kw): return _Out(self)


def _transport(monkeypatch, rig: _Rig) -> tuple[object, _Card]:
    card = _Card()
    monkeypatch.setitem(sys.modules, "sounddevice", card)
    # One codec both ways, as the station this is about has: a rig keying its
    # own monitor into the receive path  [kestrel_connect.AudioVaraIO.tx].
    return kc.AudioVaraIO("codec", "codec", rig=rig, tx_tail=0.0), card


def test_a_refused_key_up_puts_nothing_on_the_air(monkeypatch, capsys):
    """The defect, at the burst: the rig said no and the connect-request played."""
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    VaraStationHandshake([MYCALL], io, bw="2300").originate(GATEWAY)

    assert rig.ptt == [True, False], f"the burst did not key normally: {rig.ptt}"
    assert not card.played, (
        f"{len(card.played)} transmission(s) went to the card after the rig refused "
        f"the key-up — that is a burst played into a transmitter that is down")
    assert "NOT TRANSMITTING" in capsys.readouterr().out, (
        "the refusal was not reported: an operator reading the session sees a burst "
        "logged as transmitted and nothing to say it was not")


def test_a_refusal_does_not_silence_the_next_burst_the_rig_takes(monkeypatch):
    """The flag is the keyed region's, not the session's.

    A rig that refuses one key-up and takes the next is the ordinary case — a
    single timed-out `T 1` against a daemon that answers again — and a connect
    that went silent for the rest of the attempt over one of those would be this
    defect the other way up.
    """
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    hs = VaraStationHandshake([MYCALL], io, bw="2300")
    hs.originate(GATEWAY)
    assert not card.played

    rig.refuse = False
    hs.originate(GATEWAY)
    assert len(card.played) == 1, (
        "the burst the rig keyed for was still suppressed: the refusal outlived the "
        "key-up it belonged to")


def test_a_disarmed_rig_still_transmits_into_the_loopback(monkeypatch):
    """`Rig.key` returns False for a disarmed key-up by design — nothing is keyed
    and the audio goes wherever `tx_device` points. That is the rehearsal, not a
    refusal, and suppressing it would take the radioless path down with it."""
    rig = _Rig(armed=False)
    io, card = _transport(monkeypatch, rig)
    VaraStationHandshake([MYCALL], io, bw="2300").originate(GATEWAY)
    assert len(card.played) == 1, "the disarmed rehearsal stopped transmitting"


def test_an_attempt_against_a_rig_that_will_not_key_reaches_no_transmitter(
        monkeypatch, capsys):
    """The whole connect loop, not one burst: every CR is refused, so every CR is
    silence and the attempt ends unconnected on its own clock.

    What the attempt still prints before its first CR is ``originating:``, which
    `onair_session` reads as proof the air was reached; that marker goes out before
    anything is keyed and is not this seam's to answer.
    """
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    connected = kc.connect(GATEWAY, MYCALL, "2300", io, timeout=3.0,
                           cr_interval=0.3, max_cr=3, listen_first=0.0)

    assert connected is False
    assert not card.played, (
        f"{len(card.played)} burst(s) reached the card across an attempt in which "
        f"the transmitter never came up")
    assert rig.ptt.count(True) >= 2, f"the attempt stopped re-keying: {rig.ptt}"
    assert capsys.readouterr().out.count("NOT TRANSMITTING") >= 2


# --------------------------------------------------------------------------- #
# The connect-request budget counts transmissions.

def test_a_refused_connect_request_tells_its_caller_it_never_went_out(monkeypatch):
    """`originate` dropped `_send_burst`'s verdict, so an attempt's ``max_cr``
    connect-requests were spent on intentions.

    A rig that will not key puts nothing on the band, and the caller counting the
    train has no way to know: it charges the budget, and eight refusals end an
    attempt that never transmitted once.
    """
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    hs = VaraStationHandshake([MYCALL], io, bw="2300")
    assert hs.originate(GATEWAY) is False, (
        "the connect-request reported nothing back, so a caller counting the CR "
        "train cannot tell a transmission from a refusal")
    assert not card.played


def test_a_transmitted_connect_request_says_so(monkeypatch):
    """The other half: the budget must still be charged for what does go out."""
    rig = _Rig(refuse=False)
    io, card = _transport(monkeypatch, rig)
    hs = VaraStationHandshake([MYCALL], io, bw="2300")
    assert hs.originate(GATEWAY) is True
    assert len(card.played) == 1


def test_a_refused_connect_request_leaves_the_retry_ladder_where_it_was(monkeypatch):
    """A refusal must cost the budget nothing and the ladder nothing.

    The caller re-sends on `_I_CR_SENT`, so a refused request that failed to reach
    that step would be an attempt that keys once, silently, and then waits out its
    whole window with nothing left to trigger a retry.
    """
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    hs = VaraStationHandshake([MYCALL], io, bw="2300")
    hs.originate(GATEWAY)
    assert not card.played
    assert hs.step == VA._I_CR_SENT and hs.state == VA.VaraState.CONNECTING, (
        f"step={hs.step} state={hs.state} — the resend condition the caller "
        "retries on is gone, so the refusal ended the attempt instead of costing "
        "it nothing")


def test_a_rig_that_keys_spends_exactly_the_connect_requests_it_always_did(
        monkeypatch):
    """These budgets are tuned against real gateway behaviour: the attempt that
    reaches the air must key the same train it keyed before."""
    rig = _Rig(refuse=False)
    io, card = _transport(monkeypatch, rig)
    kc.connect(GATEWAY, MYCALL, "2300", io, timeout=3.4, cr_interval=0.2,
               max_cr=2, listen_first=0.0)
    assert len(card.played) == 2, (
        f"{len(card.played)} connect-requests reached the card for a budget of 2 "
        "— the window holds more than that many cadence ticks, and only the "
        "budget stops the train")


# --------------------------------------------------------------------------- #
# An over that never happened must cost nothing.

def _connected_initiator(io) -> VA.VaraStationHandshake:
    """A CONNECTED initiator holding the transmit turn, as the mail phase has it."""
    hs = VA.VaraStationHandshake([MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", GATEWAY, MYCALL
    hs.state = VA.VaraState.CONNECTED
    hs.turn = VA._TURN_OURS
    return hs


MAIL = b"QTC 1 W9SSJ DE K9WRA"


def test_a_refused_over_leaves_the_mail_exactly_where_it_was(monkeypatch):
    """The data loss: the queue entry, the over counter and the duplicate gate
    were all spent before key-up, so an over the rig refused cost a block of the
    operator's mail — permanently, three ways."""
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    hs = _connected_initiator(io)
    hs.send(MAIL)

    assert not card.played
    assert hs._txq == [MAIL], (
        f"the queue holds {hs._txq!r} — the block was spent on an over that "
        "never reached the air")
    assert hs._over == 0, f"the over counter advanced to {hs._over} for silence"
    assert not hs._keyed_bodies, (
        "the duplicate gate learned a body that never went out — the peer's first "
        "real sight of this block could be read back as our own echo")


def test_the_held_block_goes_out_whole_when_the_rig_relents(monkeypatch):
    """The other half of costing nothing: the held block is still first in line,
    and the next keyed over carries it with the counter and gate agreeing."""
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    hs = _connected_initiator(io)
    hs.send(MAIL)
    assert not card.played

    rig.refuse = False
    hs._tx_data_over()
    assert len(card.played) == 1
    assert hs._txq == []
    assert hs._over == 1
    assert hs._keyed_bodies == {_phy.vara_body(MAIL, MYCALL)}


def test_the_transcript_never_claims_a_burst_the_transport_declined(monkeypatch, capsys):
    """``tx CONNECT_REQUEST keyed-by=...`` after a tx() that did nothing is a
    transmission in the log and silence on the band. The refusal line precedes it,
    but the tx line alone must not lie."""
    rig = _Rig(refuse=True)
    io, _ = _transport(monkeypatch, rig)
    VaraStationHandshake([MYCALL], io, bw="2300").originate(GATEWAY)
    out = capsys.readouterr().out
    assert "NOT TRANSMITTING" in out
    assert "keyed-by=" not in out, (
        "the line after the refusal still claims the burst was transmitted")


# --------------------------------------------------------------------------- #
# The session's "did we reach the air" verdict, read off the child's transcript.

def test_an_attempt_of_nothing_but_refusals_is_not_a_transmission(monkeypatch, capsys):
    """``originating:`` prints before anything is keyed, so it cannot say whether
    the air was reached; the per-burst verdicts in the same transcript can."""
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    kc.connect(GATEWAY, MYCALL, "2300", io, timeout=1.2, cr_interval=0.3,
               max_cr=2, listen_first=0.0)
    out = capsys.readouterr().out
    assert "originating:" in out          # the marker the session used to trust
    assert not card.played                # and yet nothing reached the air
    assert oa.attempt_reached_air(out) is False, (
        "a session in which every key-up was refused still counts as a "
        "transmission — it sets owes_id and burns the target's campaign slot")


def test_one_confirmed_burst_makes_the_attempt_real(monkeypatch, capsys):
    rig = _Rig(refuse=False)
    io, card = _transport(monkeypatch, rig)
    kc.connect(GATEWAY, MYCALL, "2300", io, timeout=1.0, cr_interval=0.4,
               max_cr=1, listen_first=0.0)
    out = capsys.readouterr().out
    assert card.played
    assert oa.attempt_reached_air(out) is True


def test_with_no_refusal_on_record_the_verdict_stays_transmitted():
    """Identification is owed for a transmission that happened, so uncertainty
    resolves toward having transmitted: a transcript reporting no refusal keeps
    the launch marker's old meaning, and one that never printed ``originating:``
    keeps its old meaning too — our own process never started."""
    assert oa.attempt_reached_air(
        f"originating: {MYCALL} -> {GATEWAY} (BW2300)\n") is True
    assert oa.attempt_reached_air("Traceback (most recent call last):\n") is False


# --------------------------------------------------------------------------- #
# A connect the peer cannot have heard is not a connect.

def test_a_responder_whose_ack_never_went_out_stays_connecting(monkeypatch):
    """CONNECTED, and the host told so, off a burst the rig refused.

    The initiator heard no ack and is still re-sending its link-setup; a responder
    that has already declared the link up answers those repeats as data-phase
    bursts and refuses them, so the connect cannot recover from either end.
    """
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    hs = VaraStationHandshake([MYCALL], io, bw="2300")
    hs.listen(True)
    hs.on_rx_audio(MK.synth_burst(MYCALL, VF.CR))
    hs.on_rx_audio(OF.link_setup_tx(GATEWAY))
    assert not card.played
    assert hs.state == VA.VaraState.CONNECTING, (
        "the link is up on our side and dead on the peer's — the ack that would "
        "have made it real was never transmitted")

    rig.refuse = False
    hs.on_rx_audio(OF.link_setup_tx(GATEWAY))
    assert hs.state == VA.VaraState.CONNECTED, (
        "the initiator's repeat found nothing left to ack it: staying CONNECTING "
        "has to keep the recovery route open, not close it")
    assert len(card.played) == 1


def test_a_refused_close_leaves_the_session_open(monkeypatch):
    """A close is closed by the burst reaching the air, and by nothing else — a
    stock 4.9.0 keys it once and nothing answers it. So a close the transmitter
    was down for leaves the peer holding the link, and `close_verdict` must not
    read that as a frequency the operator may call again."""
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    hs = _connected_initiator(io)
    hs.disconnect()
    assert not card.played
    assert hs.state == VA.VaraState.DISCONNECTING, (
        "the session is recorded as closed on a burst nobody transmitted")
    hs.on_rx_audio(MK.synth_tone_pairs(VF.CONNECTED_ACK_2300))
    assert not card.played, "the peer's burst was answered on a closing link"

    rig.refuse = False
    hs.disconnect()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert len(card.played) == 1


def test_an_idle_answer_that_never_went_out_does_not_count_as_keyed(monkeypatch):
    """The peer's idle draws one answer, and a refused one is not that answer.

    A burst the transmitter was down for is a burst the peer has not heard, so it
    must not stand in for the answer its gap was owed: `_keyed_on_peer_burst` is
    what stops the idle tick keying a second rung into a gap already answered, and
    setting it on a burst that never reached the air silences the tick as well
    [see _answer_peer_idle, _reack].
    """
    rig = _Rig(refuse=True)
    io, card = _transport(monkeypatch, rig)
    hs = _connected_initiator(io)
    hs.turn = VA._TURN_PEER
    hs._idle_kind = VF.SESSION_RESPONDER_OVER_IDLE

    assert hs._answer_peer_idle() is False
    assert not card.played
    assert hs._keyed_on_peer_burst is False, (
        "a burst that never reached the air was counted as the gap's answer")

    rig.refuse = False
    assert hs._answer_peer_idle() is True
    assert len(card.played) == 1
    assert hs._keyed_on_peer_burst is True
