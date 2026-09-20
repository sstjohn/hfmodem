# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Mail over kestrel's own byte path: a B2F exchange across the VARA loopback.

The station side is the real thing — `VaraStationHandshake` queueing host
bytes, claiming the turn, keying rec3 overs, delivering decoded payload
through `VaraIO.data` — and the far end is `station.mail`'s scripted answering
peer. Every payload byte both ways is synthesised BW2300 audio decoded by the
interop-validated receiver.

What this file cannot prove is a real VARA's turn grant: it has never been
observed off air at BW2300, and both ends here share kestrel's assumption, so
a wrong grant law stays green by construction. The counterexamples below pin
the failure modes that ARE local — a swallowed delivery, reordered payload, a
turn never claimed — and each must leave the exchange visibly dead, because a
gate that cannot fail is not measuring anything.
"""
from __future__ import annotations

import random
import time

import pytest

from hfmodem.station.mail import VaraLoopback, VaraMail
from hfmodem.winlink import B2FSession, MailExchange, compose

from . import corpora

kc = corpora.harness("kestrel_connect")


def _exchange(outbox=None, rms_outbox=None, rig=None, max_steps=100,
              idle_cadence=False, pair_after=0):
    session = B2FSession("W9SSJ", role="calling", target="K7ABC",
                         outbox=outbox or [])
    transport = VaraLoopback("W9SSJ", "K7ABC", rms_outbox=rms_outbox or [],
                             idle_cadence=idle_cadence, pair_after=pair_after)
    if rig is not None:
        rig(transport)
    report = MailExchange(session, transport, max_steps=max_steps).run()
    return session, transport, report


def test_a_message_crosses_kestrels_air_byte_exact_both_ways():
    out = compose("W9SSJ", "SMTP:op@example.net", "vara loopback proof",
                  b"a message carried over kestrel's own byte path.\r\n")
    back = compose("K7ABC", "W9SSJ", "return mail",
                   b"and one the other way.\r\n")
    session, t, report = _exchange(outbox=[out], rms_outbox=[back])

    assert session.done and not session.failure, report
    assert t.rms.done and not t.rms.failure
    assert [m.render() for m in t.rms.inbox] == [out.render()]
    assert [m.render() for m in session.inbox] == [back.render()]
    assert session.sent_mids == [out.mid]
    assert "closed" in report
    # The bytes moved by the turn law, not around it: the station asked for
    # the turn, was granted it, and keyed real DATA overs.
    assert t.hs._over >= 2, "the station never keyed a DATA over"
    assert not t.connected, "the transport left the link up"
    # And the link came down by the 3-burst graceful close [spec 05 §5.6],
    # not by falling silent into the peer's inactivity timeout.
    assert t.clean_close, "the session ended by silence, not by the close"


def test_an_empty_exchange_still_closes_cleanly():
    session, t, report = _exchange()
    assert session.done and t.rms.done
    assert not session.failure and not t.rms.failure
    assert not session.inbox and not t.rms.inbox
    assert "closed" in report


# --------------------------------------------------------------------------- #
# The whole mailbox, and the acknowledgement that goes missing out of it.
#
# The 2026-09-08 arm: KE8LVA proposed four messages of these rendered sizes, we
# accepted all four, and after the seventh over our answer was lost under a
# co-channel occupant. The gateway idled 57 s with three messages still in its
# queue and closed. Both arms below carry that mailbox.
_MAILBOX = (643, 662, 651, 699)
_WORDS = ("gateway station mailbox message evening propagation antenna receiver "
          "operator schedule traffic forecast harbour weather bulletin roster "
          "shipment arrival departure inventory quarterly summary attached "
          "coordination frequency bandwidth transmitter waveform").split()


def _prose(n, seed):
    r = random.Random(seed)
    words = []
    while sum(len(w) + 1 for w in words) <= n:
        words.append(r.choice(_WORDS))
    return " ".join(words).encode()[:n]


def _sized(sender, to, subject, size, seed):
    n = size
    for _ in range(6):
        msg = compose(sender, to, subject, _prose(n, seed))
        short = size - len(msg.render())
        if not short:
            return msg
        n += short
    raise AssertionError(f"no body renders {sender} to {size} bytes")


def _mailbox():
    return [_sized("K7ABC", "W9SSJ", f"mailbox {i}", size, i)
            for i, size in enumerate(_MAILBOX, 1)]


@pytest.mark.slow
def test_four_messages_cross_the_loopback_and_one_goes_back():
    held = _mailbox()
    ours = _sized("W9SSJ", "SMTP:op@example.net", "the one going out", 643, 9)
    session, t, report = _exchange(outbox=[ours], rms_outbox=held,
                                   max_steps=400)
    assert session.done and not session.failure, report
    assert [m.render() for m in session.inbox] == [m.render() for m in held], report
    assert [m.render() for m in t.rms.inbox] == [ours.render()], report
    assert session.sent_mids == [ours.mid], report
    assert not session.unfetched, report
    assert t.clean_close, report


@pytest.mark.slow
def test_a_dropped_acknowledgement_is_recovered_by_a_re_ack():
    """The arm's own defect, off air: one over-answer is lost, the peer holds
    the unacknowledged over and idles on its cadence with a full queue, and the
    station must say the answer again. Nothing else recovers it — a keepalive
    is not an acknowledgement, and the peer will not send the eighth over until
    it has one."""
    held = _mailbox()
    ours = _sized("W9SSJ", "SMTP:op@example.net", "the one going out", 643, 9)
    session, t, report = _exchange(
        outbox=[ours], rms_outbox=held, max_steps=300, idle_cadence=True,
        rig=lambda t: t.peer.deafen_to_answer(7))

    assert t.peer.dropped == 1, report
    assert t.peer.idles, "the peer never keyed its cadence at the stall"
    assert [m.render() for m in session.inbox] == [m.render() for m in held], report
    assert [m.render() for m in t.rms.inbox] == [ours.render()], report
    assert len(t.peer.answers) > len(set(t.peer.answers)), (
        "no over drew a second answer: the station never re-acknowledged the "
        f"one that went missing\n{report}")


@pytest.mark.slow
def test_two_blocks_in_one_window_both_reach_the_session():
    """The 2026-09-09 fetch bench, off air: past a run of clean acknowledgements
    a stock sender stops keying one block per transmission and keys two, back to
    back inside one 8.6 s PTT window, and expects ONE answer at the end of it.
    A station that answers the first block is answering into a transmitter that
    is still keyed — nobody reads it, the second block is never delivered, and
    the mailbox stops where the bench's did.

    So the gate is the payload: every block of a two-block window has to reach
    the session, which is only true of a station that answers where the peer
    unkeys."""
    held = _mailbox()
    ours = _sized("W9SSJ", "SMTP:op@example.net", "the one going out", 643, 9)
    session, t, report = _exchange(outbox=[ours], rms_outbox=held,
                                   max_steps=400, idle_cadence=True,
                                   pair_after=6)
    assert t.peer.pairs, "the sender never grew to two blocks in a window"
    assert [m.render() for m in session.inbox] == [m.render() for m in held], report
    assert [m.render() for m in t.rms.inbox] == [ours.render()], report
    assert session.done and not session.failure, report


def test_the_peer_re_keys_the_over_a_nak_refused():
    """The last rung of the recovery ladder, with a peer that answers it: the
    station's own 8-symbol NAK, read by its tones off the lead pair it shares
    with the continue burst, and the over goes out again — which is what two
    stock 4.9.0s did on the cables on 2026-09-07  [vara_frames, NAK_BY_CALLER].
    """
    import numpy as np
    from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
    from hfmodem.kestrel.vara import vara_mfsk as MK
    t = VaraLoopback("W9SSJ", "K7ABC")
    t.peer.hs.state = VA.VaraState.CONNECTED
    t.peer.send(b"the over the station could not decode" * 2)
    t.peer._next_over()
    keyed = len(t._to_station)

    t.peer.on_air(MK.synth_tone_pairs(VF.OVER_CONTINUE_CALLER_2300))
    assert t.peer.naks == 0, "the continue burst was read as a NAK"

    t.peer.on_air(MK.synth_tone_pairs(VF.NAK_CALLER_2300))
    assert t.peer.naks == 1
    assert len(t._to_station) == keyed + 1, "the NAK drew nothing"
    assert np.array_equal(t._to_station[-1], t._to_station[keyed - 1]), (
        "the peer keyed something other than the over it was NAKed for")


# --------------------------------------------------------------------------- #
# The delivery seam, on its own: mail must not vanish into a missing sink.
def test_an_unattached_seam_reports_rather_than_swallows():
    heard = []
    io = VaraMail(lambda samples: None, log=heard.append)
    io.data(b"decoded payload")
    assert any("[unrouted]" in line for line in heard), (
        "a delivery with no client attached vanished without a trace")


# --------------------------------------------------------------------------- #
# Planted counterexamples: corrupt one wire, watch the exchange go visibly red.
def test_a_swallowed_delivery_leaves_the_exchange_dead():
    """Drop the first decoded delivery on the seam: the greeting never reaches
    the session, so the exchange must stall in 'awaiting greeting' — not
    report success around the hole."""
    def rig(t):
        orig, state = t.io.data, {"dropped": False}
        def swallow(payload):
            if not state["dropped"]:
                state["dropped"] = True
                return
            orig(payload)
        t.io.data = swallow
    session, t, report = _exchange(rms_outbox=[
        compose("K7ABC", "W9SSJ", "held", b"never fetched\r\n")], rig=rig)
    assert not session.done, "the exchange claimed success past a lost delivery"
    assert "stalled in: awaiting greeting" in report, report


def test_reordered_payload_cannot_pass_for_mail():
    """Deliver the peer's payload pairwise-swapped: the B2F stream is order,
    so the far session must fail or stall — byte-exact success over a
    reordering transport would mean the gate is not reading the bytes."""
    out = compose("W9SSJ", "SMTP:op@example.net", "ordering proof",
                  b"a body long enough to cross in several blocks.\r\n" * 4)
    def rig(t):
        orig, held = t.peer.client.on_link_data, []
        def swap(blob):
            if held:
                orig(blob)
                orig(held.pop())
            else:
                held.append(blob)
        t.peer.client.on_link_data = swap
    session, t, report = _exchange(outbox=[out], rig=rig)
    assert not (session.done and not session.failure
                and t.rms.done and not t.rms.failure), report
    assert [m.render() for m in t.rms.inbox] != [out.render()], (
        "the message arrived byte-exact through a reordering transport")


def test_a_turn_never_claimed_stalls_and_the_record_says_so():
    """Sever the turn-request: the session composes its reply, nothing may key
    it, and the exchange dies with the queue named in the stage record."""
    def rig(t):
        # Both ways this station may become the sender: the ask, and the
        # release its peer keys unasked at the end of a delivery. Severing only
        # the ask leaves the turn arriving anyway and the counterexample
        # measures nothing  [vara_frames, SESSION_TURN_RELEASE_RESPONDER].
        t.hs._tx_turn_request = lambda: None
        t.hs._took_responder_release = lambda: None
    session, t, report = _exchange(rms_outbox=[
        compose("K7ABC", "W9SSJ", "held", b"never fetched\r\n")], rig=rig)
    assert not session.done
    assert "stalled in:" in report, report
    assert "vara: turn=peer" in report and "block(s) queued here" in report, report


def test_an_unanswered_grant_is_named_for_the_post_mortem():
    """The honest edge, rehearsed: kestrel takes the turn on an answer arriving
    at all, and the BW2300 grant has never been observed off air. Make the
    grant unrecognisable and the stage record must name exactly that — the
    line a live session's post-mortem needs if a real gateway grants some
    other way."""
    def rig(t):
        t.hs._peer_control_burst = lambda samples: False
        # The unasked release, deaf as well: it hands the turn over without any
        # request, so with it readable the ask this test is about is never made.
        t.hs._peer_responder_release = lambda samples, track=None: False
    session, t, report = _exchange(rms_outbox=[
        compose("K7ABC", "W9SSJ", "held", b"never fetched\r\n")], rig=rig,
        idle_cadence=True)
    assert not session.done
    assert "turn-request unanswered" in report, report
    assert "never been observed off air" in report, report


# --------------------------------------------------------------------------- #
# The over framing that makes binary mail possible at all.
def test_binary_payload_carrying_a_trailer_byte_round_trips():
    """Compressed mail contains 0x14 like any other byte. The trailer is found
    by its whole caller-keyed pattern, so the byte alone must not cut the
    payload — and the old first-byte inference is exactly what it cannot do."""
    from hfmodem.kestrel.arq import phy
    pl = bytes([0x10, 0x14, 0x00, 0x22]) + b"x" * 30
    body = phy.vara_body(pl, "W9SSJ")
    assert phy.vara_payload(body, caller="W9SSJ") == pl
    assert phy.vara_payload(body) != pl

    ends_in_trailer_byte = b"abc\x14"
    body = phy.vara_body(ends_in_trailer_byte, "W9SSJ")
    assert phy.vara_payload(body, caller="W9SSJ") == ends_in_trailer_byte


def test_a_full_block_of_trailer_bytes_is_not_cut():
    from hfmodem.kestrel.arq import phy
    pl = b"\x14" * 89
    assert phy.vara_payload(phy.vara_body(pl, "W9SSJ"), caller="W9SSJ") == pl


def test_the_loopback_peer_reads_a_close_keyed_at_record_2():
    """A stock delivery closes one record down [arq.phy.close_level], and the
    peer here read every over at the base level: a record-2 close decoded to
    nothing, drew no answer, and the both-ways exchange died in 'their turn'.
    The record rides in the waveform, so the peer reads it off the reference
    columns first, the way the station's own receiver does."""
    import numpy as np
    from hfmodem.kestrel.arq import phy
    from hfmodem.kestrel.rx import varahf2300 as rx
    from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
    from hfmodem.kestrel.vara import vara_mfsk as MK, vara_ofdm as OF
    t = VaraLoopback("W9SSJ", "K7ABC")
    t.peer.hs.state = VA.VaraState.CONNECTED
    got = []
    t.peer.client.on_link_data = got.append
    body = phy.vara_body(b"tail", "W9SSJ", body_len=rx.payload_bytes(2))
    t.peer.on_air(OF.data_over_tx(body, over=1, bw="2300", level=2))
    assert got == [b"tail"], "the record-2 close was not delivered"
    assert np.array_equal(t._to_station[-1],
                          MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)), (
        "the close drew no control burst")


# --------------------------------------------------------------------------- #
# The graceful close [spec 05 §5.6], and its planted counterexample.
def test_a_peer_deaf_to_the_close_leaves_it_unclaimed():
    """Make the peer deaf to the disconnect-request alone: the exchange still
    completes, and `clean_close` must read False — if it reads True here, the
    close is being claimed rather than measured."""
    from hfmodem.kestrel.vara import vara_frames as VF, vara_mfsk as MK
    kind = VF.SESSION_DISCONNECT_REQ
    n_req = len(kind.preamble) + kind.n_payload

    def rig(t):
        orig = t.peer.on_air

        # By its tones, not by its length: the close shares a symbol count with
        # the turn-request and the keepalives, and deafening the peer to all of
        # them would stall the exchange this test needs to complete.
        def deaf_to_close(samples):
            n_sym = max(1, round((len(samples) - MK.STRIDE) / MK.HOP) + 1)
            if n_sym == n_req and MK.demod_tones(samples, n_sym) == \
                    VF.handshake_tones(t.peer.mycall, kind):
                return
            orig(samples)
        t.peer.on_air = deaf_to_close
    session, t, report = _exchange(rig=rig)
    assert session.done and not session.failure, report
    assert not t.clean_close, "clean_close claimed a close the peer never heard"


def test_the_loopback_tapes_closing_bursts_are_the_release_and_the_final():
    """The two initiator bursts at the end of the one captured loopback session,
    read back and named for what the 2026-08-26 bench then showed them to be.

    The 17-symbol burst at 71.3 s is the turn release, not a disconnect-request:
    a real 4.9.0 keys it in the turnaround of every answered over and its peer
    answers by transmitting  [vara_frames, SESSION_TURN_RELEASE]. What that
    leaves open is the 16-symbol burst behind it, which no bench session has
    reproduced. Both are pinned here so the parameters stay measurements."""
    import numpy as np
    from scipy.io import wavfile
    from hfmodem.kestrel.vara import vara_frames as VF, vara_mfsk as MK
    wav = (corpora.ROOT / "analysis" / "caps" / "bw2300"
           / "AAAA1-BBBB2-bw2300-counter512__a2b.wav")
    if not wav.exists():
        pytest.skip("bw2300 close capture not present (installed-wheel run)")
    a = np.asarray(wavfile.read(str(wav))[1], float)
    for t0, kind in ((71.3, VF.SESSION_TURN_RELEASE),
                     (72.7, VF.SESSION_DISCONNECT_FINAL)):
        seg = a[int(t0 * MK.FS):int((t0 + 1.1) * MK.FS)]
        n_sym = len(kind.preamble) + kind.n_payload
        at = MK.lock_preamble(seg, kind) if len(kind.preamble) >= 2 else None
        if at is None:
            env = np.abs(seg)
            at = int(np.argmax(env > 0.1 * env.max()))
        seg = seg[at:]
        need = (n_sym - 1) * MK.HOP + MK.STRIDE
        seg = np.concatenate([seg, np.zeros(max(0, need - len(seg)))])
        heard = list(MK.demod_tones(seg, n_sym))
        assert heard == VF.handshake_tones("BBBB2", kind), kind.name


# --------------------------------------------------------------------------- #
# The live transport's mail seam (tools/kestrel_connect), radiolessly.
class _Fed:
    def __init__(self):
        self.blobs = []
        self.done = False

    def link_up(self):
        self.blobs.append(b"<link up>")

    def on_link_data(self, blob):
        self.blobs.append(blob)
        self.done = True


def test_the_live_seam_routes_deliveries_to_the_client():
    io = kc._DryIO()
    io.client = fed = _Fed()
    io.connected("W9SSJ", "K7ABC", "2300")
    io.data(b"decoded payload")
    assert fed.blobs == [b"<link up>", b"decoded payload"]


def test_the_live_seam_without_a_client_reports_rather_than_swallows(capsys):
    """The counterexample's other arm: no client attached, and the delivery
    must land in the log — silence is indistinguishable from a gateway that
    sent nothing."""
    io = kc._DryIO()
    io.data(b"decoded payload")
    assert "[unrouted]" in capsys.readouterr().out


def test_mail_session_pumps_the_link_and_a_severed_seam_goes_red():
    """The loop wiring, both arms. A burst reaching the handshake delivers
    through the seam and the exchange completes; the identical run with the
    seam severed must come back False — mail that cannot fail is mail nobody
    is measuring."""
    class _Hs:
        def __init__(self, io):
            self.io = io
            self.state = kc.VaraState.CONNECTED
            self.progress = self.idle_keyed = 0

        def on_rx_audio(self, samples):
            self.io.data(b"[SID]\r")     # the burst decoded: deliver
            self.progress += 1           # ...which is what the loop paces on

        def idle_keepalive(self):
            pass

    def run(sever):
        io = kc._DryIO()
        io.client = fed = _Fed()
        if sever:
            io.data = lambda payload: None
        hs = _Hs(io)
        io.next_rx_burst = lambda timeout, hs=None: object()
        return kc.mail_session(hs, io, fed, timeout=0.5), fed

    done, fed = run(sever=False)
    assert done and fed.blobs == [b"[SID]\r"]
    done, fed = run(sever=True)
    assert not done and fed.blobs == []


def test_the_mail_loop_stops_when_the_link_layer_gives_the_session_up():
    """The loop keys nothing without a link, so a link the state machine has
    closed — overs that would not decode, NAKed to their budget — used to cost
    the whole remaining mail timeout in silence and report "out of time"."""
    class _Hs:
        state = kc.VaraState.DISCONNECTING
        progress = idle_keyed = 0

        def on_rx_audio(self, samples):
            raise AssertionError("a closed session was still being pumped")

        def idle_keepalive(self):
            raise AssertionError("a closed session was still being kept alive")

    io = kc._DryIO()
    io.client = fed = _Fed()
    io.next_rx_burst = lambda timeout, hs=None: object()
    started = time.time()
    assert not kc.mail_session(_Hs(), io, fed, timeout=30.0)
    assert time.time() - started < 5.0, "the loop sat out the mail timeout"


def test_live_mail_loop_carries_fq_to_the_peer_before_returning(monkeypatch):
    """Use separate audio events so the FF and the queued FQ cannot collapse
    into one loopback pump, hiding the live driver's early exit."""
    from hfmodem.winlink import MailClient

    transport = VaraLoopback("W9SSJ", "K7ABC")
    message = compose("W9SSJ", "test@example.net", "drain proof", "hello\n")
    session = B2FSession("W9SSJ", target="K7ABC", outbox=[message])
    client = MailClient(session, transport.send)
    transport.attach(client)
    transport.peer.hs.listen(True)
    transport.hs.originate("K7ABC", "W9SSJ")
    for _ in range(20):
        if transport.connected:
            break
        transport._pump(rounds=1)
    assert transport.connected and not session.done

    clock = [100.]
    monkeypatch.setattr(kc.time, "time", lambda: clock[0])
    monkeypatch.setattr(kc.time, "monotonic", lambda: clock[0])

    class IO:
        rig = None
        receiving = False

        def next_rx_burst(self, timeout, hs):
            clock[0] += min(timeout, .1)
            if transport._to_peer:
                transport.peer.on_air(transport._to_peer.pop(0))
            if transport._to_station:
                return transport._to_station.pop(0)
            return None

    assert kc.mail_session(transport.hs, IO(), client, timeout=30)
    assert session.our_fq and not transport.hs.data_pending
    assert transport.rms.done and not transport.rms.failure
    assert (False, "FQ", True) in transport.rms.exchange
    assert transport.rms.inbox[0].render() == message.render()


def test_attach_mail_wires_both_seams_or_neither():
    """The two wires mail hangs on, held by name: deliveries route to the
    client through the transport, and the session's answers queue on the
    handshake's own send. `main` makes them through this one function, so
    losing either wire fails here rather than silently on the air."""
    import types
    io = kc._DryIO()
    hs = types.SimpleNamespace(send=lambda b: None)
    args = types.SimpleNamespace(mail_send=None, mail_fetch=True, mail_to="",
                                 mail_subject="", mail_password="",
                                 mail_sid="", mycall="W9SSJ",
                                 gateway="K7ABC")
    mail, session = kc.attach_mail(args, hs, io)
    assert io.client is mail, "deliveries would never reach the client"
    assert mail._send is hs.send, "the session's answers would never key"
    assert session.target == "K7ABC"

    io2 = kc._DryIO()
    args.mail_fetch = False
    assert kc.attach_mail(args, hs, io2) == (None, None)
    assert io2.client is None


def test_disconnect_session_retries_and_never_claims_an_unanswered_close():
    from hfmodem.kestrel.vara.vara_arq import VaraState

    class _Hs:
        state = VaraState.CONNECTED
        keyed = 0

        def __init__(self, answers):
            self.answers = answers

        def disconnect(self):
            self.state = VaraState.DISCONNECTING
            self.keyed += 1

        def on_rx_audio(self, samples):
            if self.answers:
                self.state = VaraState.DISCONNECTED

    io = kc._DryIO()
    io.next_rx_burst = lambda timeout, hs=None: object()
    hs = _Hs(answers=True)
    assert kc.disconnect_session(hs, io, tries=3, wait=0.5)
    assert hs.keyed == 1

    io = kc._DryIO()
    hs = _Hs(answers=False)
    assert not kc.disconnect_session(hs, io, tries=2, wait=0.1)
    assert hs.keyed == 2, "an unanswered request was not re-keyed"


def test_the_close_verdict_always_names_an_outcome():
    """Every session ends on one of two lines, and the session wrapper reads
    the clean-close one off the child's output to decide whether the
    identification may be keyed — so the exact marker is a contract."""
    up = kc.close_verdict(True, "k7abc")
    down = kc.close_verdict(False, "k7abc")
    assert "DISCONNECTED: clean close" in up and "K7ABC" in up
    assert "peer receipt unconfirmed" in up and "released the link" not in up
    assert down.startswith("NOT DISCONNECTED") and "peer state is unknown" in down
    wrapper = corpora.TOOLS / "onair_session.py"
    if wrapper.exists():
        assert '"DISCONNECTED: clean close"' in wrapper.read_text(), (
            "the session wrapper no longer greps the marker the child prints")


def test_onair_mail_dry_run_reports_its_stage(tmp_path):
    """The keying tool end to end on its radioless path: flags parse, the
    session attaches, the run completes with no rig and no audio device, and
    the summary names the stage — a typo dies here, not at the radio."""
    import os
    import subprocess
    import sys
    gw_csv = tmp_path / "gateways.csv"
    gw_csv.write_text("Callsign,Mode\n")
    # THE CHILD'S OUTPUT ENCODING IS PINNED, NOT INHERITED FROM WHOEVER STARTED
    # PYTEST. `kestrel_connect` prints an em dash on its first line, and its
    # stdout here is a pipe rather than a terminal, so Python encodes it with
    # whatever the ambient locale says. Under `LC_ALL=en_US.ISO8859-1` -- or any
    # environment PEP 538 cannot coerce to UTF-8 -- that first print raises
    # UnicodeEncodeError and the tool dies before it does anything.
    #
    # The failure that produced was unreadable, which is the part that matters:
    # the child exits 1 from its own traceback, so the returncode assertion below
    # PASSES, and the run dies on an empty stdout with the real cause truncated
    # inside a CompletedProcess repr. A gate over a transmit tool going red for
    # the shell that started it, and saying nothing about why, is how a suite
    # teaches people to read past red.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        [sys.executable, str(corpora.TOOLS / "kestrel_connect.py"),
         "--gateway", "K7ABC", "--mycall", "W9SSJ", "--dry-run",
         "--listen-first", "0", "--timeout", "1", "--no-record",
         "--mail-fetch", "--mail-out", str(tmp_path / "mail"),
         "--gateways", str(gw_csv)],
        capture_output=True, encoding="utf-8", timeout=120, cwd=tmp_path, env=env)
    assert r.returncode == 1, r.stdout + r.stderr
    # A tool that said nothing died on the way in, and the next three assertions
    # would report that as a missing stage. Its own stderr is the diagnosis.
    assert r.stdout, f"the tool printed nothing at all:\n{r.stderr}"
    assert "NOT connected" in r.stdout
    assert "mail: stage awaiting greeting" in r.stdout
    assert "mail: nothing moved" in r.stdout
    # Twice: once live off the client's own watch as the stage changed, once
    # out of the teardown summary. A run that prints it only at the end is the
    # 2026-08-28 arm, where a live log showing nothing meant nothing.
    assert r.stdout.count("mail: stage") >= 2, r.stdout


# --------------------------------------------------------------------------- #
def test_cli_mail_verb_runs_the_vara_rehearsal(tmp_path, capsys):
    from hfmodem import cli
    rc = cli.main(["mail", "--gateway", "K7ABC", "--protocol", "vara",
                   "--mycall", "W9SSJ", "--fetch",
                   "--out", str(tmp_path / "in")])
    outtext = capsys.readouterr().out
    assert rc == 0, outtext
    assert "stage closed" in outtext
    assert len(list((tmp_path / "in").glob("*.b2f"))) == 1
    assert "vara-mail K7ABC" in outtext, "the on-air command is spelled out"
    # And the launcher answers to the verb the printed line names.
    launcher = corpora.TOOLS / "onair.sh"
    if launcher.exists():
        assert "vara-mail)" in launcher.read_text()
