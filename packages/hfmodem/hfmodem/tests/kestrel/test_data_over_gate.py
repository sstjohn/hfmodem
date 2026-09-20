# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What has to be true before a received burst may key the transmitter.

Once connected, an initiator answers the gateway's every DATA over — an over that
goes unanswered stops the gateway  [spec 05 §5.3.3]. The decision used to be made
on length alone: any delivered piece of at least ``_DATA_OVER_MIN`` (2.5 s) drew a
keyed response, with no test that the audio was a VARA burst at all.

That was survivable only while the segmenter cut at the first envelope falling
edge, so nothing much longer than a burst ever reached the state machine. The
bracket segmenter force-closes at 6 s instead (``tools/kestrel_connect.SEG_MAX_S``,
``test_rx_segmenter``), and rides through fades rather than cutting at them — so on
a busy channel, with the station's other modem running, or on band noise above the
gate's enter threshold, it hands over six-second pieces of whatever is on the
frequency. Answering those means keying at an unrelated station, which is both bad
practice and the failure hardest to explain afterwards.

Measured on the 246 s of real off-air HF in ``corpora`` (two live gateway sessions
and one verified-clear channel), scanned as every 6 s window at 1 s steps:

    windows scanned                                   229
    windows that key                                    9
    windows that key without containing a gateway over  0

and the reference-column populations the gate sits between:

    the five real gateway DATA overs      20, 23, 24, 24, 24  of 24
    everything else in the same 246 s     <= 8
    9 s of synthetic gaussian noise          9

so ``_OVER_GUARD_MIN`` = 16 has eight columns of room on each side.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_control as VC
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_ofdm as OF

FS = MK.FS
_CALLED = "KC9GHZ"
_MYCALL = "W9SSJ"
_WINDOW = int(6 * FS)                      # the segmenter's force-close
_SESSIONS = ("KC9GHZ_2300", "NS0A_2300")


class _IO(VA.VaraIO):
    """Counts key-ups, keeps what went on the air and what reached the host.
    Nothing here opens a device or reaches a transmitter."""

    def __init__(self):
        self.keys = 0
        self.msgs: list[str] = []
        self.sent: list[np.ndarray] = []
        self.host: list[bytes] = []

    def key(self, on): self.keys += bool(on)

    def tx(self, samples): self.sent.append(np.asarray(samples, float))

    def pending(self): ...

    def connected(self, *a): ...

    def log(self, msg): self.msgs.append(msg)

    def data(self, payload): self.host.append(bytes(payload))


class _HostIO(_IO):
    """The same, with the one thing the live mail path does that a sink does not:
    answer.

    ``AudioVaraIO.data`` hands the payload to a Winlink client, which reads the
    gateway's greeting and calls ``send`` straight back into the state machine
    before it returns  [tools/kestrel_connect._MailSeam.data -> MailClient.
    on_link_data -> B2FSession.feed]. A delivery is therefore a re-entry, and an
    IO that only collects payloads cannot see anything ordered against one.
    """

    def __init__(self, reply: bytes):
        super().__init__()
        self.hs = None
        self.reply = reply

    def data(self, payload):
        super().data(payload)
        reply, self.reply = self.reply, b""
        if reply and self.hs is not None:
            self.hs.send(reply)


def _connected(bw: str = "2300", role: str = "initiator", io: _IO | None = None,
               over_continue: str | None = None,
               over_continue_after: str | None = None):
    io = io or _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw=bw,
                                 over_continue=over_continue,
                                 over_continue_after=over_continue_after)
    hs.role, hs.called, hs.caller = role, _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    if isinstance(io, _HostIO):
        io.hs = hs
    return hs, io


def _keys(audio, bw: str = "2300", role: str = "initiator") -> bool:
    """Does this burst make a connected station key the transmitter?"""
    hs, io = _connected(bw, role)
    hs.on_rx_audio(np.asarray(audio, float))
    return io.keys > 0


def _over(text: bytes = b"kestrel test over ", index: int = 0) -> np.ndarray:
    """One base-level BW2300 DATA over, the waveform a gateway answers with."""
    n = rx.payload_bytes(rx.BASE_LEVEL)
    return tx.synth_burst((text * 8)[:n].ljust(n, b" "), over=index)


def _closing_over(text: bytes, index: int = 0) -> np.ndarray:
    """An over framed as the LAST of a delivery, which is what draws the answer
    that frees the turn. A body filled to capacity has another over behind it and
    is answered *keep sending*, so a greeting that ends here has to end short —
    which is what a real one does  [see `arq.phy.over_is_last`]."""
    return tx.synth_burst(_phy.vara_body(text, _MYCALL), over=index)


def _link_setup(caller: str) -> np.ndarray:
    """The step-4 caller-ID over — the only wideband frame that names a station."""
    return tx.synth_frame(VF.link_setup_frame(caller), over=0)


def _on_air(x, lead: float = 0.5, tail: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(int(lead * FS)), np.asarray(x, float),
                        np.zeros(int(tail * FS))])
    return y + rng.standard_normal(len(y)) * 1e-4


def _faded(x, at_s: float, dur_s: float, depth_db: float) -> np.ndarray:
    y = np.array(x, float)
    y[int(at_s * FS):int((at_s + dur_s) * FS)] *= 10 ** (depth_db / 20.0)
    return y


def _windows(x, step: float = 1.0):
    for i in range(0, max(0, len(x) - _WINDOW), int(step * FS)):
        yield i, x[i:i + _WINDOW]


# --------------------------------------------------------------------------- #
# The negatives, on the population that matters: real band audio.
@corpora.requires_clear_channel
def test_real_band_noise_of_over_length_never_keys():
    """30 s of a verified-clear 40 m frequency through the same rig and codec.

    Nothing in it is a VARA burst, and every 6 s window of it is longer than
    ``_DATA_OVER_MIN`` — which is all the old test asked for, so every one of these
    windows used to key the transmitter. The best any alignment in this recording
    reaches is 7 of the 24 reference columns.
    """
    x = corpora.wav_mono(corpora.CLEAR_CHANNEL)
    x = x / (np.abs(x).max() or 1.0)
    keyed = [round(i / FS, 1) for i, w in _windows(x) if _keys(w)]
    assert not keyed, f"band noise keyed the transmitter at t = {keyed} s"


@pytest.mark.parametrize("session", _SESSIONS)
def test_only_the_gateway_overs_key_in_a_recorded_session(session):
    """The false-key rate on a busy channel, checked against an independent detector.

    These are live sessions with a Winlink RMS gateway on 40 m: gateway bursts, our
    own transmissions, and whatever else was on the frequency. Sliding the
    segmenter's whole 6 s bracket across the recording, every window that keys must
    hold a gateway over — either a whole frame's worth (395 columns, 4.21 s) of one
    the envelope detector independently finds, or one it decodes to a clean CRC.

    The second arm is not slack, it is the stronger check, and the ladder's bottom
    records are why it is needed: ``detect_overs`` opens on 12 % of the recording's
    peak, which is a base-level over's amplitude, and the KC9GHZ session holds a
    record-1 gateway over at t = 10 s that it never segments. That over pins 24 of
    24 reference columns at its own record and decodes CRC-clean, so a window that
    keys on it is keying on a gateway.
    """
    path = corpora.OFFAIR / session / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air recording for {session} not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    overs = rx.detect_overs(x)
    assert overs, "the recording should hold wideband overs to be measuring anything"
    frame = rx._BASE_NCOLS * rx.RECORDS[rx.BASE_LEVEL].dw50
    def held(i, w):
        if any(min(i + _WINDOW, e) - max(i, s) >= frame for s, e in overs):
            return True
        return any(rx.decode_over(w, 0, len(w), level=lv).crc_ok
                   for lv in rx.INDEX_LEVELS)

    stray = [round(i / FS, 1) for i, w in _windows(x)
             if _keys(w) and not held(i, w)]
    assert not stray, (
        f"{len(stray)} window(s) of {session} keyed on audio holding no gateway "
        f"over, at t = {stray} s")


@pytest.mark.parametrize("session", _SESSIONS)
def test_every_real_gateway_over_still_keys(session):
    """The other half: the gate must not have bought its silence by going deaf.

    Each of the five overs across the two sessions is answered, delivered as the
    segmenter delivers it (the keyed region, with its lead-in and tail).
    """
    path = corpora.OFFAIR / session / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air recording for {session} not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    overs = rx.detect_overs(x)
    assert overs
    for s, e in overs:
        assert _keys(x[max(0, s - FS // 4):e + FS // 4]), (
            f"the gateway over at {s / FS:.1f}-{e / FS:.1f} s of {session} went "
            "unanswered — an unanswered over stops the gateway")


def test_synthetic_noise_of_over_length_never_keys():
    """400 blobs of 2.5-6 s, the lengths the segmenter delivers, at every level from
    a whisper to full scale. None may key; the closest any came was 9 of 24."""
    rng = np.random.default_rng(3)
    for _ in range(400):
        n = int(rng.uniform(2.5, 6.0) * FS)
        assert not _keys(rng.standard_normal(n) * rng.uniform(0.01, 1.0)), \
            "noise keyed the transmitter"


def test_a_carrier_of_over_length_never_keys():
    """Someone tuning up holds the segmenter's bracket open to its 6 s maximum."""
    t = np.arange(_WINDOW) / FS
    for hz in (500.0, 1000.0, 1500.0, 2200.0):
        assert not _keys(0.9 * np.sin(2 * np.pi * hz * t)), f"a {hz:.0f} Hz carrier keyed"


# --------------------------------------------------------------------------- #
# The four things that are not our peer's over but do look like real audio.
def test_a_strangers_connect_on_our_frequency_does_not_key():
    """A station opening a VARA session on the frequency we are working transmits a
    wideband link-setup: a real BW2300 frame, clean CRC, every reference column in
    place. It is not an over — it is step 4 of somebody else's connect — and the
    caller it names is not the station we are in session with.
    """
    for stranger in ("K7ABC", "W1AW-10", "NS0A"):
        audio = _on_air(_link_setup(stranger))
        assert not _keys(audio), f"a link-setup from {stranger} keyed the transmitter"


def test_our_own_transmission_coming_back_does_not_key():
    """The rig monitors its own transmit into the receive codec, so the link-setup
    we key arrives back at the segmenter at full scale, 4.4 s long and decoding
    perfectly — because it is real, and it is ours. It names us, which is how it is
    told from the gateway's answer.
    """
    hs, io = _connected()
    hs._tx_link_setup()                       # exactly what goes on the air
    ours = io.sent[-1]
    assert len(ours) / FS > 4.0, "the link-setup should be a wideband over"
    assert not _keys(_on_air(ours)), (
        "our own link-setup, back through the receiver, keyed the transmitter")


def test_our_own_per_over_response_coming_back_does_not_key():
    """The only other thing we put on the air in the data phase, and the one that
    could run away: answering our own answer keys again, and again. It is 1.37 s of
    MFSK, so the length precondition stops it — asserted on the burst the module
    actually transmits, since that is what makes the length argument true."""
    resp = MK.synth_tones(list(VF.SESSION_RESPONSE_2300))
    assert len(resp) < VA._DATA_OVER_MIN
    assert not _keys(_on_air(resp, lead=2.0, tail=2.0)), (
        "our own per-over response, back through the receiver, drew another one")


def _our_own_over(text: bytes = b"gateway's\r\n"):
    """A station holding the turn, and the DATA over it has just put on the air.

    The audio is the transmitter's own output, which is what the rig's monitor
    feeds back into the receive codec: just after 51 s of the 2026-08-06 KC9GHZ
    recording our own link-setup returns that way at 24 of 24 reference columns
    with a clean CRC, naming this station. A DATA over returns the same way, and
    unlike the link-setup it names nobody.
    """
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(text)
    over = io.sent[-1]
    assert len(over) / FS > 4.0, "a DATA over is a ~4.4 s wideband burst"
    io.keys, io.sent, io.host, io.msgs = 0, [], [], []
    return hs, io, over


def test_our_own_data_over_coming_back_is_not_delivered_or_answered():
    """The over we are transmitting is not the peer transmitting to us.

    It decodes — it is real, and it is ours — so every test in front of the
    delivery passes: 24 reference columns, a clean CRC, and not a link-setup,
    which is the one wideband frame that carries a callsign to be told by. What
    the recogniser had left was the ninety bytes themselves, which are the ones we
    keyed: the peer answers our overs with a short control burst and keys none of
    its own  [spec 05 §5.4], so this audio is our own transmission, and our own
    outgoing mail must not come back to the host as received mail.
    """
    hs, io, ours = _our_own_over()
    hs.on_rx_audio(_on_air(ours))
    assert not io.host, f"our own outgoing mail came back to the host: {io.host}"
    assert io.keys == 0, f"our own DATA over drew a transmission: {io.msgs}"


def test_our_own_data_over_on_the_receive_stream_is_not_delivered_or_answered():
    """The same audio by the route that actually carries it.

    ``_stream_over`` searches every sample the receiver produces, with no energy
    gate in front of it and no bracket to be too quiet for — which is the point of
    it, and which also means our own transmission reaches it whenever the
    transport's transmit cursor does not skip every sample of the monitor path.
    """
    hs, io, ours = _our_own_over()
    echo = _on_air(ours)
    for i in range(0, len(echo), VA._STREAM_BLOCK):
        hs.on_rx_stream(echo[i:i + VA._STREAM_BLOCK])
    assert hs._stream_owns_over, "the stream route did not run"
    assert not io.host, f"our own outgoing mail came back to the host: {io.host}"
    assert io.keys == 0, f"our own DATA over drew a transmission: {io.msgs}"


# --------------------------------------------------------------------------- #
# The positives: a working link must stay working.
def test_a_genuine_over_is_answered():
    hs, io = _connected()
    hs.on_rx_audio(_on_air(_over()))
    assert io.keys == 1, "a genuine DATA over went unanswered"
    assert any("tx per-over response" in m for m in io.msgs), io.msgs


def test_an_over_arriving_while_we_hold_the_turn_is_answered():
    """Why the turn is not a gate in front of the over search.

    Nothing in the corpus shows a gateway asking for the turn back — the one
    turn-request ever captured is the caller's — so a peer that simply starts
    sending again is a live possibility, and refusing its overs on a turn state
    that stale is a station that has stopped listening to a gateway still talking.
    The over is answered with the per-over response, which means *keep sending*.
    """
    hs, io, _ours = _our_own_over()
    hs.on_rx_audio(_on_air(_over(b"gateway greeting ", index=7)))
    assert io.host, "a genuine gateway over went undelivered"
    assert io.keys == 1, io.msgs
    assert any("tx per-over response" in m for m in io.msgs), io.msgs


def _stranger_over(text: bytes = b"mail for somebody else\r\n", who: str = "K7ABC"):
    """A third station's BW2300 DATA over, framed with ITS caller's callsign.

    Decoded off air this is 90 bytes of payload and a CRC-16 — no address, no
    session id, nothing that names a station  [vara_arq._peer_data_over]. So it
    reaches the over recogniser looking exactly like our peer's: 24 of 24
    reference columns, a clean CRC, not a link-setup, and not one of ours.
    """
    return OF.data_over_tx(_phy.vara_body(text, who), over=3)


def test_one_strangers_over_does_not_cost_us_the_turn():
    """The third case the turn law leaves open, and the price of reading it wrong
    in the cheap direction.

    While we hold the turn the peer answers with short control bursts and keys no
    overs  [spec 05 §5.4], so an over arriving then contradicts the law — and the
    contradiction reads the same whether the transmitter is our peer breaking it
    or a station we are not in session with. Yielding the turn on the first one
    costs a stranger's single transmission the turn, three more keyings as the
    turn-request goes unanswered, and every block still queued behind it. So the
    first is answered and concedes nothing; what a run of them does is
    ``test_a_peer_that_takes_the_turn_back_ends_up_with_it`` in test_turn_law.
    """
    hs, io, _ours = _our_own_over()
    hs._txq = [b"A" * 89, b"B" * 89, b"C" * 89]
    hs.on_rx_audio(_on_air(_stranger_over()))
    assert io.keys == 1, f"a stranger's over cost {io.keys} keyings: {io.msgs}"
    assert hs.turn == VA._TURN_OURS, "a stranger's over took the turn off us"
    assert len(hs._txq) == 3, "a stranger's over cost us the queue"
    for _ in range(VA._TURN_MAX_ASK + 1):
        hs.idle_keepalive()
    assert len(hs._txq) == 3, (
        f"the queue was abandoned after a stranger's over: {io.msgs}")


def test_a_body_we_keyed_stops_being_ours_once_another_station_has_transmitted():
    """The echo window on ``_keyed_bodies``, and what an unbounded one refuses.

    Short payloads frame identically in both directions: ``arq.phy.vara_body``
    ends a short body with 0x14, the high byte of the CALLER callsign's CRC-16,
    and zeros — the caller's from either end — so a gateway echoing `FQ\\r` back
    at us produces byte for byte the body we sent. Kept forever, that byte
    pattern is refused for the rest of the session and the link stalls on it.

    What makes a body ours is that we have just keyed it: our own transmission
    returns through the rig's monitor before the next burst off the air, never
    after one. So a received control burst closes the window — whoever keyed it,
    which is the cost of the window and is set out in ``_close_echo_window``.
    """
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b"FQ\r")
    keyed, = hs._keyed_bodies
    # What the gateway frames for the same three bytes: the trailer is keyed to
    # the CALLER, and on this link the caller is us.
    theirs = _phy.vara_body(b"FQ\r", hs.caller)
    assert theirs == keyed, "the two directions should frame `FQ` identically"
    hs.on_rx_audio(_on_air(MK.synth_tone_pairs(VF.CONNECTED_ACK_2300), 0.2, 0.2))
    io.keys, io.sent, io.host, io.msgs = 0, [], [], []
    hs.on_rx_audio(_on_air(OF.data_over_tx(theirs, over=9)))
    assert io.host == [b"FQ\r"], f"the peer's own `FQ` was refused: {io.msgs}"
    assert io.keys == 1, io.msgs


def test_the_over_a_host_keys_inside_a_delivery_is_still_ours_afterwards():
    """The echo window against the seam that actually delivers.

    A delivery is not a sink call: the mail client answers the greeting from
    inside it, so a DATA over is keyed and recorded as ours *during* the call that
    delivers the burst which prompted it. Closed after delivering rather than
    before, the window forgets that over the instant it is recorded, and our own
    outgoing mail comes back through the rig's monitor as received mail — which is
    the delivery this whole gate exists to refuse.

    The turn is ours here, which is the state where the host's answer changes
    nothing about the frame we key back [see :meth:`_answer_data_over`], so the
    order this pins is the answer-then-deliver one and the over the host keys
    lands inside the delivery, as it always did.

    Measured at 233b1fb: ``_keyed_bodies`` empty after the gateway's over, and our
    own over back through the monitor delivered to the host as
    ``FC EM ABC 100 80 0\\rF> aa\\r``. It is a layer rather than a live fault —
    ``AudioVaraIO.tx`` skips the receive stream through the end of every
    transmission plus 0.1 s, so on the live audio path that echo does not reach
    the state machine at all — but this file is that layer.
    """
    reply = b"FC EM ABC 100 80 0\rF> aa\r"
    hs, io = _connected(io=_HostIO(reply))
    hs.turn = VA._TURN_OURS                  # the turn has already been granted
    hs.on_rx_audio(_on_air(_over(b"gateway greeting ", index=7)))
    assert io.host, f"the gateway's greeting went undelivered: {io.msgs}"
    ours = [s for s in io.sent if len(s) / FS > 4.0]
    assert ours, f"the host's answer never went on the air: {io.msgs}"
    remembered = set(hs._keyed_bodies)
    delivered = list(io.host)
    io.msgs.clear()
    hs.on_rx_audio(_on_air(ours[-1]))
    assert io.host == delivered, (
        f"our own outgoing mail came back to the host: {io.host[len(delivered):]}")
    assert remembered == {_phy.vara_body(reply, hs.caller)}, (
        "the over the host keyed inside the delivery was not remembered as ours")


def test_one_over_answered_with_a_host_reply_keys_exactly_one_burst():
    """The Winlink greeting, at the turn state it really arrives in — and the
    count no loopback can make, because both ends of a loopback tolerate whatever
    we key.

    The gateway holds the turn and sends its greeting; the mail client answers
    from inside the delivery with a proposal, which is data we have nowhere to
    put until we hold the turn. Both are true of one turnaround, and a turnaround
    holds one frame. Across three complete VARA-to-VARA sessions in the loopback
    corpus — 120 keyings and 51 DATA overs each, read off the harness's own PTT
    ledger — the stations alternate at all 118 changes of transmitter, and no
    station ever keyed twice into a turnaround its peer had opened. Off air a
    gateway keys its next over 0.27-0.29 s after our answer ends; a second 1.4 s
    frame goes out on top of that.

    So one burst answers this over, and which burst says what we want: the
    acknowledgement the over asks for. A queue of our own used to replace it with
    the turn-request, and that is what stopped the 2026-09-08 mailbox — the peer's
    release follows the acknowledgement, so with none keyed no release came and
    the ask went out on our own clock at a peer that had gone back to idling. The
    ask is owed from here and keyed into the peer's own cadence instead
    [see `VaraStationHandshake._reack_release`].
    """
    hs, io = _connected(io=_HostIO(b"FC EM ABC 100 80 0\rF> aa\r"))
    assert hs.turn == VA._TURN_PEER
    hs.on_rx_audio(_on_air(_closing_over(b"gateway greeting >\r", index=7)))
    assert io.host, f"the gateway's greeting went undelivered: {io.msgs}"
    assert io.keys == 1, (
        f"{io.keys} bursts keyed into one turnaround: "
        f"{[m for m in io.msgs if m.startswith('tx ')]}")
    assert any("per-over response" in m for m in io.msgs), io.msgs
    assert not any("session-turn-request" in m for m in io.msgs), io.msgs
    assert hs.turn == VA._TURN_PEER
    assert hs._answer_owed == VA._OWED_RELEASE, io.msgs
    assert hs._txq, "the host's reply was thrown away with the ask"


def test_a_repeated_greeting_draws_the_per_over_response_not_a_second_request():
    """No recording shows whether a turn-request also acknowledges the over it
    answers. If it does not, the gateway repeats its greeting — and a station that
    answered every repeat with another request would ask forever and never
    acknowledge anything.

    One request goes out, and the turn state carries that: the repeat draws the
    per-over response the gateway is listening for, and the repeated payload is
    not handed to the host twice.
    """
    hs, io = _connected(io=_HostIO(b"FC EM ABC 100 80 0\rF> aa\r"))
    greeting = _on_air(_closing_over(b"gateway greeting >\r", index=7))
    hs.on_rx_audio(greeting)
    delivered = list(io.host)
    io.msgs.clear()
    io.keys = 0
    hs.on_rx_audio(greeting)
    assert io.keys == 1, (
        f"{io.keys} bursts answered the repeat: "
        f"{[m for m in io.msgs if m.startswith('tx ')]}")
    assert any("per-over response" in m for m in io.msgs), io.msgs
    assert not any("session-turn-request" in m for m in io.msgs), io.msgs
    assert io.host == delivered, f"the repeat was delivered twice: {io.host}"


_FADES = [(at, dur, db)
          for at in (0.5, 1.5, 2.5, 3.5)
          for dur in (0.025, 0.050, 0.090, 0.200)
          for db in (-60, -16)]


@pytest.mark.parametrize("at,dur,db", _FADES, ids=lambda v: str(v))
def test_a_faded_over_is_still_answered(at, dur, db):
    """The degradation the connected-ack path was rewritten to tolerate, applied to
    the over: notches from 25 to 200 ms, down to -60 dB, anywhere in the burst. The
    segmenter rides through these rather than splitting on them (test_rx_segmenter),
    so the whole burst arrives with a hole in it. Measured on the five real off-air
    overs over this same grid: 160 of 160 still answered.
    """
    assert _keys(_on_air(_faded(_over(), at, dur, db))), (
        f"a {dur * 1000:.0f} ms {db} dB fade at {at} s cost the gateway its answer")


@pytest.mark.parametrize("snr_db", [20, 12, 6, 0])
def test_a_noisy_over_is_still_answered(snr_db):
    """Fail-closed must not mean fragile: the identification is a frame decode, and
    the frame is turbo-coded. Measured on the real off-air overs, all five are still
    answered with white noise at the burst's own power added to them (0 dB SNR)."""
    rng = np.random.default_rng(snr_db)
    x = _over()
    p = float(np.sqrt((x * x).mean()))
    assert _keys(x + rng.standard_normal(len(x)) * p / 10 ** (snr_db / 20.0)), \
        f"an over at {snr_db} dB SNR went unanswered"


def test_a_burst_too_short_to_hold_a_frame_is_not_answered():
    """Fail closed on the piece the segmenter cuts short: 2.5 s clears the length
    precondition, but a base over is 395 columns (4.21 s) and there is no way to
    tell whose a fragment of one is."""
    x = _over()
    for secs in (2.6, 3.5, 4.2):
        assert not _keys(x[:int(secs * FS)]), f"answered a {secs} s piece of an over"


# --------------------------------------------------------------------------- #
# The rest of the transmit decisions in the data phase.
def test_a_session_never_keys_at_another_bandwidths_over():
    """A BW500 session has its own over recogniser now, and a BW2300 over is not
    one — a different waveform on a comb four times as wide, which
    ``decode_stream`` finds no data frame in. This used to key the idle keepalive
    at any 2.5 s burst: a frame nobody had shown was the right answer, sent to a
    burst nobody had identified."""
    hs, io = _connected(bw="500")
    hs.on_rx_audio(_on_air(_over()))
    hs.on_rx_audio(np.random.default_rng(0).standard_normal(_WINDOW))
    assert io.keys == 0, io.msgs
    assert any("no data frame in it" in m for m in io.msgs), io.msgs


def _undecodable_over(text: bytes = b"gateway greeting ") -> np.ndarray:
    """A real rec3 over carrying a frame whose CRC-16 is wrong.

    All 24 reference columns light — the burst is positively an over, which is the
    whole point — and nothing behind them can be trusted, so it reaches exactly the
    branch that identified KC9GHZ's 4.8 s burst at 18 of 24 on 2026-08-18 and then
    said nothing. Corrupting the trailer rather than the audio is what keeps the
    identification clean: noise over the payload columns loses the alignment too,
    and then the test is measuring the guard instead of the decode.
    """
    n = rx.payload_bytes(rx.BASE_LEVEL)
    frame = tx.build_frame((text * 8)[:n].ljust(n, b" "), rx.BASE_LEVEL)
    return tx.synth_frame(bytes(frame[:-2]) + b"\x00\x00", rx.BASE_LEVEL, over=0)


def test_an_over_that_will_not_decode_keys_the_measured_nak_at_bw2300():
    """The 2026-08-18 defect, and the answer to it that BW2300 actually has.

    The turnaround used to be left empty: the only NAK on hand came out of a
    2026-07-23 table no real VARA was recorded keying, and it drew no repeat off
    KC9GHZ. A register-and-reject bench on 2026-09-07 measured the real one — the
    receiver keys an 8-symbol burst and the sender drops a speed level and
    re-sends [see vara_frames.nak]. This station is the caller here, so it keys
    the caller's burst of a W9SSJ-called link, and the over never reaches the host.
    """
    hs, io = _connected()
    hs.on_rx_audio(_on_air(_undecodable_over()))
    assert io.keys == 1, f"the NAK did not go out: {io.msgs}"
    pairs = [tuple(sorted(p)) for p in MK.demod_tone_pairs(io.sent[-1], 8)]
    assert pairs == list(VF.NAK_CALLER_2300), pairs
    assert any("tx NAK" in m for m in io.msgs), io.msgs
    assert not io.host, "an undecodable frame reached the host"


def test_the_nak_this_station_does_have_is_one_a_peer_can_read():
    """BW500's NAK is a promoted spec fact, read off the loopback corpus. It is not
    reachable through an over — BW500 has no over recogniser — so it is asserted
    where it lives, which is also the only place anything now keys a token."""
    hs, io = _connected(bw="500")
    assert hs._send_token("nak")
    m = VC.detect_token(io.sent[-1])
    assert m is not None and m.name == "nak", m


def test_an_over_that_will_not_decode_is_reported_once_off_the_receive_stream():
    """The route the 4.8 s burst actually arrived by. An energy gate cannot see
    these bursts at all, so a fix that only reaches the bracket route would leave
    the live path exactly as silent as it was — and the stream must not report the
    same over twice on its way past."""
    hs, io = _connected()
    echo = _on_air(_undecodable_over())
    for i in range(0, len(echo), VA._STREAM_BLOCK):
        hs.on_rx_stream(echo[i:i + VA._STREAM_BLOCK])
    assert hs._stream_owns_over, "the stream route did not run"
    said = [m for m in io.msgs if "tx NAK" in m]
    assert len(said) == 1, f"the stream route answered {len(said)} times: {io.msgs}"


def test_the_undecodable_overs_are_bounded_and_the_link_closes_rather_than_holding():
    """The other half, and the reason the answer is not simply "wait for the next".

    An empty turnaround asks for the over again, which repairs a fade; a repeat
    that fails the same way says the decode is ours to fix, and waiting a third
    time is the VARA keepalive loop of #104. Both ways of getting this wrong end
    with a peer transmitting to a station that has stopped taking part.
    """
    hs, io = _connected()
    for _ in range(VA._OVER_NAK_MAX + 1):
        hs.on_rx_audio(_on_air(_undecodable_over()))
    said = [m for m in io.msgs if "tx NAK" in m]
    assert len(said) == VA._OVER_NAK_MAX, io.msgs
    assert hs.state == VA.VaraState.DISCONNECTED, (
        f"the link was still held after {VA._OVER_NAK_MAX} unrepaired overs")
    assert any("disconnect" in m for m in io.msgs), io.msgs


def test_an_over_that_does_decode_clears_the_nak_budget():
    """The budget counts overs this station could not read in a row, not overs it
    was sent — a link that is working after a fade must not be closed by what the
    fade cost."""
    hs, io = _connected()
    hs.on_rx_audio(_on_air(_undecodable_over()))
    hs.on_rx_audio(_on_air(_over(b"gateway greeting ", index=7)))
    assert io.host, "a genuine over went undelivered"
    for _ in range(VA._OVER_NAK_MAX):
        hs.on_rx_audio(_on_air(_undecodable_over()))
    assert hs.state == VA.VaraState.CONNECTED, (
        "the budget carried across an over that decoded cleanly")


def test_the_echo_window_survives_an_over_that_will_not_decode():
    """A frame that will not decode names nobody, ours included. Clearing the
    bodies we keyed on the strength of one would let our own outgoing mail come
    back through the rig's monitor and reach the host as received mail."""
    hs, io, ours = _our_own_over()
    keyed = set(hs._keyed_bodies)
    assert keyed, "the station kept no record of what it put on the air"
    hs.on_rx_audio(_on_air(_undecodable_over()))
    assert hs._keyed_bodies == keyed, "an unreadable burst closed the echo window"
    hs.on_rx_audio(_on_air(ours))
    assert not io.host, f"our own outgoing mail came back to the host: {io.host}"


def test_a_responder_never_answers_an_over():
    """Role-asymmetric  [spec 05 §5.3.3]: a responder uses the short DBPSK data-ack,
    so it must not key the initiator's per-over frame at a genuine over either."""
    assert not _keys(_on_air(_over()), role="responder")


def test_an_unconnected_station_never_answers_an_over():
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300")
    hs.listen(True)
    hs.on_rx_audio(_on_air(_over()))
    assert io.keys == 0, io.msgs


# --------------------------------------------------------------------------- #
def test_the_guard_threshold_sits_between_the_two_measured_populations():
    """The margin the gate depends on, restated as an assertion so it cannot drift.

    A real frame lights its 24 reference columns; anything else matches at chance.
    Measured: 20-24 for the five real off-air overs, <= 8 for everything else in the
    same 246 s, 9 for synthetic noise. Both distances to ``_OVER_GUARD_MIN`` are
    checked here, since a threshold is only as good as the gap it sits in.
    """
    hs, _ = _connected()
    rng = np.random.default_rng(5)
    clean, _ = hs._rec3_alignment(_over())
    assert clean == 24, f"a clean over scores {clean}/24 reference columns"
    worst = max(hs._rec3_alignment(rng.standard_normal(_WINDOW))[0] for _ in range(20))
    assert worst < VA._OVER_GUARD_MIN, (
        f"noise reaches {worst}/24 reference columns, at or above the "
        f"{VA._OVER_GUARD_MIN} the gate opens on")


def test_the_check_stays_inside_the_per_burst_budget():
    """It runs inside a live ARQ turnaround, once per delivered burst, alongside the
    connected-ack matcher's 0.28 s. Measured: 24 ms to reject a 6 s bracket, 35 ms
    for 9 s, 41-55 ms to accept a real over (the decode is only reached once the
    reference columns already line up, so noise never pays for it).
    """
    rng = np.random.default_rng(1)
    blob = rng.standard_normal(9 * FS) * 0.05
    over = _on_air(_over())
    t0 = time.perf_counter()
    _keys(blob)
    reject = time.perf_counter() - t0
    t0 = time.perf_counter()
    _keys(over)
    accept = time.perf_counter() - t0
    assert reject < 0.30, f"rejecting a 9 s bracket took {reject * 1000:.0f} ms"
    assert accept < 0.30, f"accepting an over took {accept * 1000:.0f} ms"
