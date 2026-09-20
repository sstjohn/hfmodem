# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The post-CONNECTED session bursts: step-6 confirm + the two idle keepalives.

These are PRNG-MFSK handshake frames keyed to the CALLED callsign (spec 05 §5.3.3),
not OFDM data frames. Tone sequences below were demodulated from live Wine VARA
BW2300 sessions at 2048-sample symbol resolution, five independent called callsigns;
each predictor reproduces its capture exactly from the callsign alone.
"""
import pytest

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF

# call -> {burst kind name: observed tone sequence}  [live VARA capture, 2026-07-24]
_CAPTURED = {
    "BBBB2": {
        "session-confirm": [62, 82, 73, 46, 59, 64, 69, 60, 37, 42, 79, 30, 87, 88, 63, 58],
        "session-keepalive-a": [74, 98, 39, 60, 87, 68, 97, 52, 51, 90, 69, 92, 33, 36, 43,
                                40, 37, 98, 59, 72, 51, 58, 37, 38, 63, 64, 33, 38, 39, 66,
                                37, 50],
        "connect-response": [62, 67, 55, 66, 59, 72, 68, 55, 36, 43, 40, 63, 92, 63, 70, 59,
                             80, 59, 58, 29, 74, 81, 58],
    },
    "EEEE2": {
        "session-keepalive-a": [74, 66, 45, 96, 97, 48, 63, 48, 53, 66, 63, 50, 57, 58, 93,
                                38, 63, 70, 77, 62, 31, 98, 77, 98, 49, 54, 75, 88, 85, 86,
                                75, 74],
    },
}


@pytest.mark.parametrize("call,kind_name,tones", [
    (c, k, t) for c, d in _CAPTURED.items() for k, t in d.items()
])
def test_session_burst_predicts_capture(call, kind_name, tones):
    assert VF.handshake_tones(call, VF.BURSTS[kind_name]) == tones


def test_session_bursts_are_called_callsign_keyed():
    """Different called callsigns must give different bursts — the frames carry the
    called identity, so a recognizer keyed to the wrong call has to miss."""
    for kind in (VF.SESSION_CONFIRM, VF.SESSION_KEEPALIVE_A, VF.SESSION_KEEPALIVE_B):
        a = VF.handshake_tones("BBBB2", kind)
        b = VF.handshake_tones("BBBB5", kind)
        assert a != b
        assert a[0] == b[0], "the fixed preamble tone is callsign-invariant"


def test_par0_default_preserves_the_original_bursts():
    """The parity flag added for the session bursts must not perturb CR/response/ack."""
    assert VF.CR.par0 == 1 and VF.CONNECT_RESPONSE.par0 == 1
    # bin = 29 + parity + 14P + 2D, so bin parity is the complement of the tone parity
    # (29 is odd): par0=1 on the first tone => even bin, alternating from there.
    bins = VF.payload_bins("BBBB2", VF.CONNECT_RESPONSE)
    assert [b % 2 for b in bins[:4]] == [0, 1, 0, 1]


# --------------------------------------------------------------------------- #
# Driver wiring: the session bursts must actually be keyed on the link.
class _IO(VA.VaraIO):
    def __init__(self, peer_inbox):
        self.peer_inbox = peer_inbox
        self.sent = []

    def key(self, on): pass

    def tx(self, samples): self.peer_inbox.append(samples)

    def pending(self): pass

    def connected(self, caller, called, bw): pass

    def log(self, msg): self.sent.append(msg)


def _data_over():
    """One real base-level BW2300 DATA over — the burst a gateway keys, which is
    what an initiator has to recognise before it may answer."""
    from hfmodem.kestrel.rx import varahf2300 as rx
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    return tx.synth_burst(b"x" * rx.payload_bytes(rx.BASE_LEVEL), over=0)


def _handshake():
    i_inbox, r_inbox = [], []
    io_i, io_r = _IO(r_inbox), _IO(i_inbox)
    init = VA.VaraStationHandshake(["MYCALL"], io_i, mfsk_only=True)
    resp = VA.VaraStationHandshake(["GATE1"], io_r, mfsk_only=True)
    resp.listen(True)
    init.originate("GATE1")
    for _ in range(100):
        if r_inbox:
            resp.on_rx_audio(r_inbox.pop(0))
        elif i_inbox:
            init.on_rx_audio(i_inbox.pop(0))
        else:
            break
    return init, resp, io_i, io_r


def test_initiator_keys_the_session_confirm_after_connected():
    init, resp, io_i, io_r = _handshake()
    assert init.state is VA.VaraState.CONNECTED
    assert any("tx session-confirm" in m for m in io_i.sent), io_i.sent


def test_responder_answers_the_session_confirm():
    init, resp, io_i, io_r = _handshake()
    assert any("rx session-confirm" in m for m in io_r.sent), io_r.sent
    assert any("tx connect-response" in m for m in io_r.sent)


def test_the_keepalive_is_answered_never_originated_and_never_a_b():
    """In the peer's turn the caller keys a keepalive only when asked, and only A.

    The tape this used to be written from and the bench that replaced it agree
    once the prompts are put back into the reading. The logged VARA-to-VARA
    BW2300 session keyed keepalive-A at 10.2 s and keepalive-B at 12.3, 24.4 and
    36.5 s — a ~12 s spacing, which is the responder's poll cadence of 12.07 s
    measured on the 2026-08-30 cross-wired pair. So those four bursts are four
    answers, not a cadence of the caller's own: on the wine bench of 2026-09-16 a
    stock caller alone in the peer's turn keyed nothing for 60 s and then
    disconnected on a silence timer.

    What is left is the invariant: our own clock keys nothing, the peer's poll
    draws one keepalive-A  [see _took_poll], and keepalive-B is never keyed at all.
    """
    init, resp, io_i, io_r = _handshake()
    assert init.turn == VA._TURN_PEER and init.state is VA.VaraState.CONNECTED
    io_i.sent.clear()

    for _ in range(3):
        init.idle_keepalive()
    assert not [m for m in io_i.sent if m.startswith("tx")], (
        f"keyed on our own clock while the turn was the peer's: {io_i.sent}")

    init._took_poll()
    keyed = [m for m in io_i.sent if m.startswith("tx")]
    assert len(keyed) == 1 and "session-keepalive-a" in keyed[0], keyed
    assert init.called in keyed[0], "the answer is keyed to the CALLED station"

    for _ in range(VA._POLLS_PER_ANSWER * 2):
        init._took_poll()
        init.idle_keepalive()
    assert not any("keepalive-b" in m for m in io_i.sent), io_i.sent


def test_the_frame_three_gateways_returned_names_this_station():
    """Why the same 32 tones came back from three different gateways.

    The frame was read as carrying no identity at all, on the strength of not
    varying with the gateway. It carries one: it is keyed to the CALLER, and the
    caller was this station every time  [spec 05 §5.3.3]. Which is a stronger
    claim, so it is checked both ways — the caller reproduces it exactly and no
    gateway comes close. The turn law that follows from it is in test_turn_law.
    """
    t = VF.SESSION_RESPONSE_2300
    assert len(t) == 32
    assert all(29 <= x <= 98 for x in t)          # spec 04 §4.2.3 tone range
    assert t[0] == 74                             # same fixed preamble tone as the long bursts
    assert VF.handshake_tones(VF.SESSION_RESPONSE_2300_CALL,
                              VF.SESSION_TURN_IDLE) == list(t)
    for call in ("KC9GHZ", "KO2F", "NS0A"):
        for kind in (VF.SESSION_KEEPALIVE_A, VF.SESSION_KEEPALIVE_B,
                     VF.SESSION_TURN_IDLE, VF.SESSION_OVER_RESPONSE):
            got = VF.handshake_tones(call, kind)
            assert sum(1 for a, b in zip(got, t) if a == b) < 9


def test_bw2300_per_over_response_round_trips():
    from hfmodem.kestrel.vara import vara_mfsk as MK
    tones = list(VF.SESSION_RESPONSE_2300)
    assert MK.demod_tones(MK.synth_tones(tones), len(tones)) == tones


def test_initiator_answers_a_data_over():
    """A gateway stops sending if its DATA over goes unanswered, so a connected
    initiator must key the per-over response  [spec 05 §5.3.3].

    The over has to be the real waveform: 4.4 s of anything at all used to be enough
    to key, which is the defect ``test_data_over_gate`` exists to hold shut.
    """
    init, resp, io_i, io_r = _handshake()
    assert init.state is VA.VaraState.CONNECTED
    init.bw = "2300"
    io_i.sent.clear()
    init.on_rx_audio(_data_over())
    assert any("rx DATA over" in m for m in io_i.sent), io_i.sent
    assert any("tx per-over response" in m for m in io_i.sent), io_i.sent


def test_responder_does_not_answer_with_the_initiator_frame():
    """The response is role-asymmetric: a responder uses the short DBPSK data-ack,
    so it must not key the initiator's 1.38 s frame."""
    init, resp, io_i, io_r = _handshake()
    resp.bw = "2300"
    io_r.sent.clear()
    resp.on_rx_audio(_data_over())
    assert not any("tx per-over response" in m for m in io_r.sent), io_r.sent
