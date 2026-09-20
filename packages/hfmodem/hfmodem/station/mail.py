# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Mail over this station's own modems: transports for the winlink client.

The client itself — the pump, the exchange engine, the stage record — lives in
`hfmodem.winlink.client` and knows no modem. This module is the other half of
that bargain: one thin transport per modem, each built from the modem's own
loopback machinery, and the `hfmodem mail` verb that drives them.

The loopback transports are the rehearsal for a rig slot: a message composed,
carried across the modem's real byte path — chunked, acknowledged, the channel
handed back and forth by the link's own law — received by a genuine answering
session, decompressed and read back. What that leaves untested is the RF and
the far end's opinion, which is exactly the list a morning session wants.
"""
from __future__ import annotations

import os
import sys

from hfmodem.besra.host.modem_core import ModemObserver
from hfmodem.core import config
from hfmodem.winlink import B2FSession, MailClient, MailExchange, compose


class ArdopMail(ModemObserver):
    """A mail client as besra's `ModemObserver`: hand it to `modem.start`.

    Only ARQ payload feeds the session. FEC broadcasts, uncorrectable blocks
    and decoded IDs are not link bytes, and a session fed an ERR block would
    fail a transfer the link never broke. The other events are a host server's
    concern, and in this arrangement there is no host server.
    """

    def __init__(self, client: MailClient):
        self.client = client

    def modem_connected(self, remote: str, bw: int) -> None:
        self.client.link_up()

    def modem_data_received(self, kind: str, blob: bytes) -> None:
        if kind == "ARQ":
            self.client.on_link_data(blob)

    def modem_newstate(self, state: str) -> None:
        pass

    def modem_disconnected(self) -> None:
        pass

    def modem_ptt(self, on: bool) -> None:
        pass

    def modem_buffer(self, nbytes: int) -> None:
        pass


class VaraMail:
    """A mail client on kestrel's `VaraIO` seam: payload decoded off a peer's
    DATA over feeds the session in arrival order, the connected edge is
    `link_up`, and sending stays the handshake's own — `VaraStationHandshake.
    send` queues, asks for the turn, and drains by the turn law. Duck-typed
    against `kestrel.vara.vara_arq.VaraIO` so importing this module costs no
    DSP.

    With no client attached a delivery is reported on the transport's log,
    never consumed in silence — the `VaraIO.data` contract, because silence is
    indistinguishable from a gateway that sent nothing.
    """

    def __init__(self, transmit, log=None):
        self.transmit = transmit
        self.client: MailClient | None = None
        self._log = log or (lambda msg: None)

    def key(self, on: bool) -> None:
        pass

    def tx(self, samples) -> None:
        self.transmit(samples)

    def pending(self) -> None:
        pass

    def connected(self, caller: str, called: str, bw: str) -> None:
        if self.client is not None:
            self.client.link_up()

    def data(self, payload: bytes) -> None:
        if self.client is None:
            self.log(f"[unrouted] {len(payload)} decoded payload bytes: "
                     "no data sink")
        else:
            self.client.on_link_data(bytes(payload))

    def log(self, msg: str) -> None:
        self._log(msg)


class PactorLoopback:
    """shrike's loopback as a mail transport: a `PtcHost` against a `SimPeer`
    whose far end is a real answering session, every caller-side byte crossing
    the ARQ link — chunked into packets, acknowledged, carried across
    changeovers by `PtcHost.app_turns` on this side and the same discipline on
    the gateway's."""

    def __init__(self, mycall: str, gateway: str,
                 rms_outbox: list | None = None):
        from hfmodem.shrike.arq import State
        from hfmodem.shrike.ptc import PtcHost, SimPeer
        self._State = State
        self.gateway = gateway.upper()
        self.rms = B2FSession(self.gateway, role="answering", target=mycall,
                              outbox=rms_outbox or [])
        self.peer = SimPeer()
        # The far end's application: feed the answering session, and queue what
        # it says on the far link layer — the honest path, transmitted only
        # when the far end holds the channel, not short-circuited back.
        self.peer.reply = lambda blob: self.peer.send(self.rms.feed(blob))
        self.host = PtcHost(self.peer, mycall=mycall.upper())

    @property
    def send(self):
        return self.host.arq.on_host_data

    def attach(self, client: MailClient) -> None:
        self.host.app = client

    def connect(self) -> bool:
        self.host.arq.on_host_connect(self.host.mycall, self.gateway)
        for _ in range(20):
            self.host.tick()
            if self.connected:
                self.peer.send(self.rms.start())
                return True
        return False

    def step(self) -> None:
        self.host.tick()
        self.host.app_turns()
        # The gateway's half of the same turn law.
        if not self.rms.done:
            if self.peer.role == "iss" and not self.peer.sending:
                self.peer.over()
            elif self.peer.role == "irs" and self.peer.sending:
                self.peer.breakin()

    @property
    def connected(self) -> bool:
        return self.host.arq.state == self._State.CONNECTED

    def disconnect(self) -> None:
        if not self.connected:
            return
        self.host.arq.on_host_disconnect()
        for _ in range(30):
            self.host.tick()
            if not self.connected:
                return

    def carried(self) -> int:
        """Payload bytes the far end's link layer actually received."""
        return len(self.peer.received)


class ArdopLoopback:
    """besra's loopback as a mail transport: two `BesraModem`s over `StepAir`,
    every byte in both directions modulated, flown, and demodulated. ARDOP's
    turn law is the session's own — an IRS with data queued BREAKs — so no
    discipline is added here."""

    def __init__(self, mycall: str, gateway: str,
                 rms_outbox: list | None = None, bandwidth: int = 500):
        from hfmodem.besra.arq.modem import BesraModem
        from hfmodem.besra.sim.air import StepAir
        self.gateway = gateway.upper()
        self.rms = B2FSession(self.gateway, role="answering", target=mycall,
                              outbox=rms_outbox or [])
        self.air = StepAir()
        self.modem = BesraModem(bandwidth=bandwidth)
        self.modem.set_mycall(mycall.upper())
        self.far = BesraModem(bandwidth=bandwidth)
        self.far.set_mycall(self.gateway)
        self.far.set_listen(True)

    @property
    def send(self):
        return self.modem.transmit

    def attach(self, client: MailClient) -> None:
        self.modem.start(ArdopMail(client))
        self.far.start(ArdopMail(MailClient(self.rms, self.far.transmit)))
        self.air.join(self.modem)
        self.air.join(self.far)

    def connect(self) -> bool:
        self.modem.connect(self.gateway)
        self.air.run(max_time=60)
        return self.modem.connected and self.far.connected

    def step(self) -> None:
        self.air.run(max_time=30)

    @property
    def connected(self) -> bool:
        return self.modem.connected

    def disconnect(self) -> None:
        if self.modem.connected:
            self.modem.disconnect()
            self.air.run(max_time=60)
        self.modem.stop()
        self.far.stop()


class _VaraPeer:
    """The answering station of a VARA loopback: kestrel's real responder FSM
    for the connect, and past it the gateway's half of the session, scripted
    the way shrike's `SimPeer` is. Every station burst is positively
    identified — a wideband over by the same alignment/turbo/CRC decode a
    gateway's own overs are held to, a session frame by its demodulated tones
    against the generator — and answered with the frames a real gateway keys.

    What the script cannot claim is a real gateway's turn law. The BW2300
    grant has never been observed off air, so the grant here is the control
    burst kestrel accepts positionally: our own assumption, played back at us.
    """

    def __init__(self, mycall: str, caller: str, rms: B2FSession, transmit,
                 pair_after: int = 0):
        import numpy as np
        from hfmodem.kestrel.arq import phy
        from hfmodem.kestrel.rx import varahf2300
        from hfmodem.kestrel.vara import vara_arq, vara_frames, vara_mfsk, vara_ofdm
        self._np = np
        self._phy, self._rx, self._va = phy, varahf2300, vara_arq
        self._vf, self._mk, self._of = vara_frames, vara_mfsk, vara_ofdm
        self.mycall, self.caller = mycall.upper(), caller.upper()
        self.transmit = transmit
        self.hs = vara_arq.VaraStationHandshake(
            [self.mycall], VaraMail(transmit), bw="2300")
        self.client = MailClient(rms, self.send)
        self._outq: list[bytes] = []
        self._over = 0                  # position in the session's preamble stream
        self._await_ack = False
        self._last_body: bytes | None = None
        self._greeted = False
        self._deaf_at = 0
        #: The over each tone-identified short answer arrived against.
        self.answers: list[int] = []
        self.dropped = 0
        self.idles = 0
        self.naks = 0
        self.releases = 0
        self._last_tx = None
        self._released_at = -1
        #: Consecutive full overs this end has been acknowledged for, and the
        #: run at which a stock sender starts keying two blocks per window.
        self.pair_after = pair_after
        self.pairs = 0
        self._clean = 0
        self._full = False

    def deafen_to_answer(self, nth: int) -> None:
        """Lose the nth over-answer, the way a co-channel occupant lost one on
        2026-09-08: the over stands unacknowledged with the queue behind it
        still full, and this end falls to its idle cadence."""
        self._deaf_at = nth

    def idle(self) -> None:
        """One turn of the responder's own cadence: with an over outstanding a
        stock station keys `SESSION_RESPONDER_IDLE` every 3.3-3.5 s until the
        answer comes. Not the over-idle beside it: 745 is what the same station
        keys at a turn-request while it still holds the turn  [vara_frames,
        SESSION_RESPONDER_IDLE]."""
        if not self._await_ack:
            return
        self.idles += 1
        self._clean = 0
        self.transmit(self._mk.synth_burst(
            self.mycall, self._vf.SESSION_RESPONDER_IDLE))

    def send(self, blob: bytes) -> None:
        n = self._phy.payload_size("2300")
        self._outq += [bytes(blob[i:i + n]) for i in range(0, len(blob), n)]

    def _next_over(self) -> None:
        """One transmission, which past a run of clean acknowledgements is TWO
        overs keyed back to back inside one PTT window and answered once: a
        stock sender grew to that at the seventh over of the 2026-09-09 fetch
        bench, 8.6 s keyed with no gap between the blocks."""
        if self._await_ack or not self._outq:
            return
        n = self._phy.payload_size("2300")
        window = [self._outq.pop(0)]
        if (self.pair_after and self._clean >= self.pair_after
                and len(window[0]) == n and self._outq):
            window.append(self._outq.pop(0))
            self.pairs += 1
        bursts = []
        for payload in window:
            self._over += 1
            bursts.append(self._of.data_over_tx(
                self._phy.vara_body(payload, self.caller), over=self._over))
        self._full = len(window[-1]) == n
        self._last_tx = (bursts[0] if len(bursts) == 1
                         else self._np.concatenate(bursts))
        self.transmit(self._last_tx)
        self._await_ack = True

    def _acknowledged(self) -> None:
        self._await_ack = False
        self._clean = self._clean + 1 if self._full else 0

    def _is_nak(self, pairs) -> bool:
        pair = self._vf.nak(self.caller, "2300")
        return pair is not None and all(
            tuple(sorted(a)) == tuple(sorted(b))
            for a, b in zip(pairs, pair[0]))

    def _is_control_burst(self, pairs) -> bool:
        pair = self._vf.control_bursts(self.caller, "2300")
        return pair is not None and all(
            tuple(sorted(a)) == tuple(sorted(b))
            for a, b in zip(pairs, pair[0]))

    def _release_turn(self) -> None:
        """The release a stock responder keys UNASKED, 0.08-0.15 s behind the
        acknowledgement of its last over, once its queue is empty  [vara_frames,
        SESSION_TURN_RELEASE_RESPONDER]. Once per delivery: the station that
        acknowledges rather than asks is waiting for exactly this burst, and
        without it the channel is nobody's."""
        if self._outq or self._released_at == self._over:
            return
        self._released_at = self._over
        self.releases += 1
        self.transmit(self._mk.synth_burst(
            self.mycall, self._vf.SESSION_TURN_RELEASE_RESPONDER))

    def _answer(self) -> None:
        """The control burst a gateway keys back — over-answer and turn grant
        in one, which is as much as the recordings distinguish.

        A real station keys this only at the LAST over of a delivery and a
        continue-class frame at every intermediate one  [vara_arq,
        OVER_CONTINUE_ANSWERS]. Modelling that half here would make the loopback
        assert a capability the station does not have: nothing in `vara_arq` can
        read either continue-class frame back yet, so the peer would answer our
        intermediate overs with a frame we cannot hear. The gap is real and it is
        this end of it that is missing, so it is named rather than papered over.
        """
        self._await_ack = False
        self.transmit(self._mk.synth_tone_pairs(self._vf.CONNECTED_ACK_2300))

    def _rx_over(self, samples) -> None:
        level = self._rx.index_guard(samples)[0][0]
        fr = self._rx.decode_over(samples, 0, len(samples), level=level)
        if not fr.crc_ok or self._vf.is_link_setup(bytes(fr.frame_bytes)):
            return
        body = bytes(fr.payload)
        if body != self._last_body:     # a repeat is answered, not re-fed
            self._last_body = body
            self.client.on_link_data(
                self._phy.vara_payload(body, caller=self.caller,
                                       body_len=len(body)))
        self._answer()

    def on_air(self, samples) -> None:
        if self.hs.state != self._va.VaraState.CONNECTED:
            self.hs.on_rx_audio(samples)
            if (self.hs.state == self._va.VaraState.CONNECTED
                    and not self._greeted):
                self._greeted = True
                self.client.link_up()   # the gateway speaks first
                self._next_over()
            return
        if len(samples) >= self._va._DATA_OVER_MIN:
            self._rx_over(samples)
            return
        vf, mk = self._vf, self._mk
        n_sym = max(1, round((len(samples) - mk.STRIDE) / mk.HOP) + 1)
        if n_sym == vf.OVER_CONTINUE_NSYM:
            # *Keep sending*, in the captured 8-symbol copy of it: a sender
            # that waits for the control burst here stops after one over with its
            # queue full. That is the shape seven gateways returned, and a bench
            # responder reproduced it from a queue we controlled  [vara_frames,
            # OVER_CONTINUE_CALLER_2300].
            pairs = mk.demod_tone_pairs(samples, vf.OVER_CONTINUE_NSYM)
            if all(tuple(sorted(a)) == tuple(sorted(b)) for a, b in
                   zip(pairs, vf.OVER_CONTINUE_CALLER_2300)):
                self._acknowledged()
                self._next_over()
            elif self._is_nak(pairs) and self._last_tx is not None:
                # A stock sender answers the NAK by re-sending the over, one
                # speed level down [vara_frames, NAK_BY_CALLER]. The same
                # record goes out here: the drop is the sender's own choice,
                # and this end holds no other rendering of the over.
                self.naks += 1
                self._clean = 0
                self.transmit(self._last_tx)
            return
        if n_sym == vf.CONNECTED_ACK_NSYM:
            # The control burst, which is what answers a DATA over: two stock
            # VARA HF 4.9.0 instances over a fake cable on 2026-08-26 answered
            # every over of three sessions with it and with nothing else, in both
            # directions, and neither ever keyed a 32-symbol frame at all. This
            # end used to wait for SESSION_OVER_RESPONSE here, which is a frame
            # no station in those recordings sends.
            pairs = mk.demod_tone_pairs(samples, vf.CONNECTED_ACK_NSYM)
            if all(tuple(sorted(a)) == tuple(sorted(b)) for a, b in
                   zip(pairs, vf.CONNECTED_ACK_PREAMBLE)):
                self._acknowledged()
                self._next_over()
                if self._is_control_burst(pairs):
                    self._release_turn()
            return
        rel = vf.SESSION_TURN_RELEASE
        if n_sym == len(rel.preamble) + rel.n_payload:
            # The initiator has drained and handed the channel back, which is the
            # one thing this end needs before it may key  [vara_frames,
            # SESSION_TURN_RELEASE].
            if mk.demod_tones(samples, n_sym) == vf.handshake_tones(
                    self.mycall, rel):
                self._next_over()
            return
        short = vf.SESSION_OVER_RESPONSE_SHORT
        if n_sym == len(short.preamble) + short.n_payload:
            # *Keep sending*, in the 32-symbol frame's 16-symbol sibling on the
            # same lattice: what the station keys at a full over once the turn
            # has been its and gone back, and what a stock responder reads
            # 18 of 18 [vara_arq, OVER_CONTINUE_AFTER_DEFAULT].
            if mk.demod_tones(samples, n_sym) == vf.handshake_tones(
                    self.mycall, short):
                self.answers.append(self._over)
                if len(self.answers) == self._deaf_at:
                    self.dropped += 1
                    return
                self._acknowledged()
                self._next_over()
            return
        kind = vf.SESSION_TURN_REQUEST
        if n_sym != len(kind.preamble) + kind.n_payload:
            return                      # confirm / keepalive: nothing owed
        tones = mk.demod_tones(samples, n_sym)
        if tones == vf.handshake_tones(self.mycall, vf.SESSION_OVER_RESPONSE):
            # *Keep sending*, in the other of the two frames that say it — the
            # generated one, keyed to this end's own callsign, which is what the
            # station keys at a full over by default  [vara_arq,
            # OVER_CONTINUE_ANSWERS]. Same length as the turn-request below, so it
            # is separated by its tones and not by its symbol count.
            self._acknowledged()
            self._next_over()
            return
        if tones == vf.handshake_tones(self.mycall, vf.SESSION_DISCONNECT_REQ):
            # The initiator is closing [spec 05 §5.6]: the responder FSM already
            # knows the whole answering side, so hand the burst back to it.
            self.hs.on_rx_audio(samples)
        elif tones == vf.handshake_tones(self.caller, kind):
            self._answer()              # the grant, as kestrel assumes it
        elif tones == vf.handshake_tones(self.caller, vf.SESSION_TURN_IDLE):
            self._next_over()           # the station is idle: channel is ours



class VaraLoopback:
    """kestrel's loopback as a mail transport: the station side is the real
    thing — `VaraStationHandshake` queueing host bytes, claiming the turn,
    keying rec3 overs, delivering through `VaraIO.data` — against the
    answering `_VaraPeer`, every payload byte both ways a synthesised BW2300
    burst decoded by the interop-validated receiver.

    Both ends are this implementation, so the loopback cannot settle whether a
    real gateway grants the turn the way kestrel assumes (the BW2300 grant has
    never been observed off air); `stall_note` names that edge whenever an
    exchange dies waiting on it.
    """

    def __init__(self, mycall: str, gateway: str,
                 rms_outbox: list | None = None, idle_cadence: bool = False,
                 pair_after: int = 0):
        from hfmodem.kestrel.vara import vara_arq
        self._va = vara_arq
        self.idle_cadence = idle_cadence
        self.mycall, self.gateway = mycall.upper(), gateway.upper()
        self.rms = B2FSession(self.gateway, role="answering", target=mycall,
                              outbox=rms_outbox or [])
        self._to_peer: list = []
        self._to_station: list = []
        self.io = VaraMail(self._to_peer.append)
        self.hs = vara_arq.VaraStationHandshake([self.mycall], self.io,
                                                bw="2300")
        self.peer = _VaraPeer(self.gateway, self.mycall, self.rms,
                              self._to_station.append, pair_after=pair_after)
        self.clean_close = False

    @property
    def send(self):
        return self.hs.send

    def attach(self, client: MailClient) -> None:
        self.io.client = client

    def connect(self) -> bool:
        self.peer.hs.listen(True)
        self.hs.originate(self.gateway, self.mycall)
        self._pump()
        return self.connected

    def step(self) -> None:
        # Both queues empty is a turnaround nobody answered. Off by default:
        # the stall counterexamples measure a station that keys nothing on the
        # cadence, and this would fill their record.
        if self.idle_cadence and not self._to_peer and not self._to_station:
            self.peer.idle()
            self._pump()
            self.hs.idle_keepalive()
        self._pump()

    def _pump(self, rounds: int = 400) -> None:
        for _ in range(rounds):
            if self._to_peer:
                self.peer.on_air(self._to_peer.pop(0))
            elif self._to_station:
                self.hs.on_rx_audio(self._to_station.pop(0))
            else:
                return

    @property
    def connected(self) -> bool:
        return self.hs.state == self._va.VaraState.CONNECTED

    @property
    def stall_note(self) -> str:
        """The link-layer state a dead exchange leaves behind, for the record."""
        if self.hs.turn == self._va._TURN_ASKED:
            return ("vara: turn-request unanswered — kestrel takes the turn on "
                    "any answer arriving at all, and the BW2300 grant has never "
                    "been observed off air")
        return (f"vara: turn={self.hs.turn}, {len(self.hs._txq)} block(s) "
                f"queued here, {len(self.peer._outq)} at the peer")

    def disconnect(self) -> None:
        # The graceful close [spec 05 §5.6], driven for real: the station keys
        # the disconnect-request, the peer's responder FSM acknowledges, the
        # final answers it — every burst synthesised and demodulated like the
        # rest of the exchange. `clean_close` records whether that happened; a
        # close the peer never answered falls back to what a live link does —
        # both ends stop, nothing keys, the peer times the link out — and the
        # record says so rather than claiming a close that did not complete.
        self.hs.disconnect()
        self._pump()
        self.clean_close = (
            self.hs.state == self._va.VaraState.DISCONNECTED
            and self.peer.hs.state == self._va.VaraState.DISCONNECTED)
        self.hs.state = self._va.VaraState.DISCONNECTED
        self.peer.hs.state = self._va.VaraState.DISCONNECTED


TRANSPORTS = {
    "pactor": PactorLoopback,
    "ardop": ArdopLoopback,
    "vara": VaraLoopback,
}


def _onair_command(args) -> str:
    """The keying path for this mail session, spelled out.

    `hfmodem mail` itself cannot key — deliberately. Each protocol has exactly
    one validated way onto the air at this station, with the panic handling a
    keying path owes the operator, so the on-air session is that command with
    the mail flags carried through.

    PACTOR's driver is `shrike.onair`, which owns the CAT port itself and keys
    the RTS line beside it. Its two serial ports have no shipped default —
    deliberately, since a shipped default once put 22.1 s of reported carrier
    into a radio nobody had opened — so they ride the printed line as shell
    references to CAT_PORT and PTT_PORT, the environment contract every
    launcher in tools/ answers to; unset, they expand to nothing and shrike's
    arm gate refuses by name. ARDOP and VARA go through the `onair.sh`
    launcher's mail verbs: it checks the hardware, starts rigctld with
    ptt_type=None so the daemon only ever tunes — its rig side stops answering
    the moment PTT is asserted (2026-08-13) — leaves the keying line to the
    modem itself, and proves PTT low afterwards. The launcher is printed by
    name without a directory: it is a station instrument that lives outside
    the installable package, and shipping code may not spell its path.
    """
    import shlex
    mail = []
    for f in args.send or []:
        mail += ["--mail-send", f]
    if args.fetch:
        mail += ["--mail-fetch"]
    if args.to:
        mail += ["--mail-to", args.to]
    if args.subject:
        mail += ["--mail-subject", args.subject]
    # Carried only when the rehearsal was overridden. Left off, the on-air tool
    # reads the station file for itself and the printed line stays true after
    # the file is edited.
    if args.sid:
        mail += ["--mail-sid", args.sid]
    mail += ["--mail-out", args.out]
    # The printed line lands in scrollback and redirected session logs, so the
    # password never rides it as a literal — only as the file it came out of or
    # as the reference that will expand. A password given as a bare --password
    # has neither behind it, and the line then carries a placeholder that fails
    # loudly when it is pasted unfilled. The alternative, a reference to an
    # unexported variable, expands to nothing and logs in with an empty password
    # against an attempt budget that does not refill.
    if not args.password:
        pw = ""
    elif args.password_file:
        pw = " --mail-password-file " + shlex.quote(args.password_file)
    elif args.password == os.environ.get(config.PASSWORD_ENV):
        pw = f' --mail-password "${config.PASSWORD_ENV}"'
    else:
        pw = " --mail-password-file <YOUR-PASSWORD-FILE>"
    if args.protocol == "pactor":
        cmd = [sys.executable, "-m", "hfmodem.shrike.onair",
               "--mycall", args.mycall, "--dxcall", args.gateway]
        if args.freq:
            cmd += ["--center", str(args.freq)]
        # The ports ride as shell references for the same reason the password
        # does: a per-machine value may not be baked into a printed line, and
        # a reference that expands to nothing meets the arm gate, not a radio.
        return (shlex.join(cmd + mail + ["--transmit"])
                + ' --serial "$CAT_PORT" --ptt-port "$PTT_PORT"' + pw)
    centre = str(args.freq) if args.freq else "<CENTRE-HZ>"
    if args.protocol == "ardop":
        cmd = ["onair.sh", "ardop-mail", args.gateway, centre,
               str(getattr(args, "bandwidth", 500))]
    else:
        cmd = ["onair.sh", "vara-mail", args.gateway, centre]
    return shlex.join(cmd + ["--mycall", args.mycall] + mail) + pw


def run(args) -> int:
    """The `hfmodem mail` verb: a full B2F exchange over the chosen modem's
    own byte path, and the on-air command that carries the same exchange to
    a rig."""
    from hfmodem.winlink import load_outbound, write_inbox

    if not args.send and not args.fetch:
        print("nothing to do: give --send FILE, --fetch, or both.",
              file=sys.stderr)
        return 2

    args.password = config.mail_password(args.password, args.password_file)

    outbox = [load_outbound(p, args.mycall, to=args.to, subject=args.subject)
              for p in args.send or []]
    session = B2FSession(args.mycall, role="calling", target=args.gateway,
                         password=args.password, outbox=outbox,
                         client_sid=config.client_sid(args.sid))
    # The rehearsal gateway holds a message when a fetch is asked for, so the
    # receive path is proved rather than merely not refused.
    rms_outbox = ([compose(args.gateway, args.mycall, "loopback rehearsal",
                           b"carried the other way, gateway to station.\r\n")]
                  if args.fetch else [])
    extra = ({"bandwidth": args.bandwidth}
             if args.protocol == "ardop" and getattr(args, "bandwidth", None)
             else {})
    transport = TRANSPORTS[args.protocol](args.mycall, args.gateway,
                                          rms_outbox=rms_outbox, **extra)

    print(f"rehearsal: {args.protocol} loopback, "
          f"{args.mycall.upper()} -> {args.gateway.upper()}, "
          f"announcing as {session.sid}")
    print(MailExchange(session, transport).run())
    for p in write_inbox(session, args.out):
        print(f"mail: wrote {p}")
    print("\non air, this exchange is:\n  " + _onair_command(args))

    ok = session.done and not session.failure
    if args.send and len(session.sent_mids) != len(outbox):
        ok = False
    if args.fetch and not session.inbox:
        ok = False
    return 0 if ok else 1
