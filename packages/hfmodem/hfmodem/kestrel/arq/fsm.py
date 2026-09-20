# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel ARQ / session state machine — transport-agnostic.

Implements the observable state machine of spec
`05-arq-session-state-machine.md`: the five states (§5.1), the transitions
(§5.2), the connect handshake (§5.3), stop-and-wait ARQ with per-over ACK,
retransmit and 1->2 block/over gear-shift (§5.4), turnaround + keepalive timing
(§5.5) and the 3-burst graceful disconnect (§5.6).

The FSM is pure logic: it consumes events (host commands, decoded RX frames,
timer ticks) and drives I/O only through the injected :class:`ArqIO`. All entry
points are expected to be called serially (the modem funnels every event through
one worker thread), so no internal locking.

Every behaviour is tagged inline with what fixes it:
    [spec 05 §x]  -> the spec fixes it; changing it changes interoperability.
    [ours]        -> kestrel's own choice, where the spec is silent (the value
                     lives inside the CONTROL waveform, spec 05 §5.7 / spec 04).
                     Retunable, within what the link's own timing tolerates.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from . import frames as F
from . import phy


class State(StrEnum):
    """Members compare equal to their names, so log lines and host-API strings
    keep working unchanged."""
    DISCONNECTED = "DISCONNECTED"
    LISTENING = "LISTENING"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTING = "DISCONNECTING"


@dataclass
class ArqConfig:
    # Turnaround: spec 05 §5.5 min ~85 ms (mean 88.5, 52-122). Our value in-range.
    turnaround_s: float = 0.085                 # [spec 05 §5.5]
    # ACK timeout: spec 05 §5.5 UNRESOLVED (no ACK ever timed out on clean
    # loopback). Value below is [ours], sized > one frame round-trip.
    ack_timeout_s: float = 6.0                  # [ours — spec gap 5.5]
    max_retries: int = 8                        # [ours — spec gap 5.4 max-retries]
    # Connect handshake retransmit — [ours] (spec silent on connect retry timing
    # for P2P clean loopback; doc mentions ~4.6 s P2P retry cycle, §5.7).
    connect_timeout_s: float = 8.0              # [ours]
    max_connect_retries: int = 4                # [ours]
    # Gear-shift: spec 05 §5.4 "grows 1->2 blocks/over after several consecutive
    # successes". Threshold count itself is [ours].
    gear_up_after: int = 3                      # [ours — "several", spec 5.4]
    max_blocks_per_over: int = 2                # [spec 05 §5.4: 1->2]
    # Speed-level gear-shift (spec 06 §6.2/6.3): up one level on sustained success,
    # back to the robust base on failure. Off by default (single robust level) so
    # the standard loopback runs at the fast base level; enable to climb the ladder.
    speed_gearshift: bool = False               # [spec 06 §6.2/6.3 policy]
    speed_up_after: int = 2                      # [ours — spec 6.2 "≥1 good over"]
    # On-air keepalive: spec 05 §5.5 ~10-12 s.
    keepalive_s: float = 10.0                   # [spec 05 §5.5]
    # Link death. Absence of traffic must eventually mean absence of a link:
    # a peer that aborts, dies, drops carrier or is powered off otherwise holds
    # this session open forever, and a CONNECTED station refuses every later
    # connect request. Sized at several missed keepalives so a slow-but-alive
    # link is never torn down. This is the half that protects against peers we
    # do not control, which on air is all of them.
    link_death_s: float = 60.0                  # [ours — spec silent]
    block_bytes: int = F.PAYLOAD                # 43 B/block at level 4 [spec 5.4]
    # Speak VARA's real control channel: emit/read the control burst a recording
    # fixes for ACK / connect-answer / keepalive instead of kestrel's own control
    # frames. At BW500 that is four DBPSK tokens [spec 02 §2.6]; at BW2300 it is
    # one index-modulated waveform for the two answers and nothing for the rest
    # [spec 04 §4.2C]. connect-request/confirm (MFSK, spec 04) and disconnect stay
    # native at both, and so does the BW2300 NAK — no recording fixes it.
    vara_compat: bool = False                   # [spec 02 §2.6 tokens]


# Control type <-> VARA token name, for the tokens whose waveform spec 02 §2.6
# fixes.
_TOKEN_FOR = {F.ACK: "data-ack", F.NAK: "nak",
              F.CA: "connected-ack", F.KA: "alive"}
_CTYPE_FOR = {v: k for k, v in _TOKEN_FOR.items()}

#: And what BW2300 has instead, which is one waveform and two answers: the
#: responder's CONNECT-time burst and its first per-over answer are the same
#: keying to correlation 1.00 [spec 04 §4.2C, see arq.phy.ANSWER_2300]. Nothing
#: here for the NAK or the keepalive, because no real VARA has been recorded
#: keying either at this bandwidth — those fall through to kestrel's own control
#: frames, which is a link kestrel speaks to itself rather than a waveform it
#: asserts of somebody else.
_TOKEN_2300_FOR = {F.ACK: phy.ANSWER_2300, F.CA: phy.ANSWER_2300}


def _vara_token(ctype: int, bw: str) -> str | None:
    """The VARA control token ``bw`` keys for ``ctype``, or None if it keys none."""
    if bw == "2300":
        return _TOKEN_2300_FOR.get(ctype)
    return _TOKEN_FOR.get(ctype) if bw == "500" else None


class ArqIO:
    """Sink the FSM drives. The modem implements this over the PHY + host API."""
    def key(self, on: bool) -> None: ...              # PTT ON/OFF around an over
    def tx(self, payload: bytes, marker: int, bw: str = "500",
           level=None) -> None: ...                   # render+send one burst
    def tx_token(self, name: str, bw: str = "500") -> None: ...   # send one VARA token
    def on_busy(self, on: bool) -> None: ...             # BUSY ON/OFF
    def on_pending(self, cancel: bool = False) -> None: ...
    def on_connected(self, src: str, dst: str, bw: str) -> None: ...
    def on_disconnected(self) -> None: ...
    def on_buffer(self, nbytes: int) -> None: ...        # BUFFER n
    def on_deliver(self, blob: bytes) -> None: ...       # payload -> host data port
    def log(self, msg: str) -> None: ...


@dataclass
class _Over:
    """One keyed transmission: 1..max_blocks_per_over data blocks, ACKed as a unit."""
    blocks: list = field(default_factory=list)        # list[(seq, block43)]
    retries: int = 0

    @property
    def last_seq(self) -> int:
        return self.blocks[-1][0]


class ArqFsm:
    def __init__(self, io: ArqIO, cfg: ArqConfig | None = None, clock=time.monotonic):
        self.io = io
        self.cfg = cfg or ArqConfig()
        self._clock = clock
        self.state = State.DISCONNECTED
        self.role = None                # "initiator" | "responder"
        self.src = self.dst = self.bw = ""
        self._listen = False

        # bandwidth-derived PHY parameters (recomputed when bw is set)
        self._psize = phy.payload_size("500")   # ARQ block payload bytes
        self._ladder = phy.speed_ladder("500")  # ascending speed levels for this bw
        self._level = self._ladder[0]           # current data-over speed level
        self._speed_successes = 0               # consecutive good overs at level

        # ISS (send) engine
        self._blockq: deque = deque()   # pending (seq, block43)
        self._next_seq = 0
        self._over: _Over | None = None
        self._blocks_per_over = 1
        self._successes = 0
        self._buffer_raw = 0            # raw host bytes still queued (BUFFER n)
        self._preconnect: list = []     # host writes before CONNECTED [spec 07 §7.4]
        self._disconnect_pending = False

        # IRS (receive) engine
        self._reasm = F.Reassembler()
        self._over_base = 0             # expected seq at start of current inbound over
        self._over_blocks: list = []    # tentative blocks of the inbound over
        self._connect_retries = 0

        # timers (monotonic deadlines; None = disarmed)
        self._t_ack: float | None = None
        self._t_connect: float | None = None
        self._t_keepalive: float | None = None
        self._last_activity = self._clock()

    # ---- helpers ---------------------------------------------------------
    def _to(self, state: State) -> None:
        if state != self.state:
            self.io.log(f"state {self.state} -> {state}")
            self.state = state

    def _touch(self) -> None:
        self._last_activity = self._clock()

    def _set_bw(self, bw: str) -> None:
        """Adopt a connection bandwidth: resize the ARQ block + speed ladder."""
        self.bw = bw or self.bw or "500"
        self._psize = phy.payload_size(self.bw)
        self._ladder = phy.speed_ladder(self.bw)
        self._level = self._ladder[0]
        self._speed_successes = 0

    def _send_control(self, ctype: int, seq: int = 0) -> None:
        # `finally`, always: everything between the key and the key-down is a keyed
        # transmitter, and an exception in there is a transmitter left on a shared
        # band with nobody modulating it.
        # VARA-compat: emit the control burst a real VARA keys wherever a recording
        # fixes its waveform, and kestrel's own frame wherever none does.
        name = _vara_token(ctype, self.bw) if self.cfg.vara_compat else None
        if name is not None:
            self.io.key(True)
            try:
                self.io.tx_token(name, self.bw)
            finally:
                self.io.key(False)
            self.io.log(f"tx TOKEN {name}")
            return
        # control/handshake always rides the robust base level [spec 06 §6.4]
        c = F.Control(ctype=ctype, seq=seq, src=self.src, dst=self.dst, bw=self.bw)
        payload, marker = F.encode_control(c, size=self._psize)
        self.io.key(True)
        try:
            self.io.tx(payload, marker, self.bw, phy.base_level(self.bw))
        finally:
            self.io.key(False)
        self.io.log(f"tx CONTROL {c.name} seq={seq}")

    def on_rx_token(self, name: str) -> None:
        """A received VARA control token -> the equivalent control action.

        VARA tokens carry no sequence field (§5.7); for the data-ACK the seq is
        implicit in stop-and-wait — it acks whichever over is outstanding.
        """
        self.io.on_busy(True)
        self.io.log(f"rx TOKEN {name}")
        ctype = _CTYPE_FOR.get(name)
        if name == phy.ANSWER_2300:
            # One waveform for both answers, so which one this is, is the state we
            # are in and not anything in the audio  [see arq.phy.ANSWER_2300].
            ctype = F.ACK if self.state is State.CONNECTED else F.CA
        if ctype == F.ACK:
            if self._over is not None:
                self._rx_ack(self._over.last_seq)
        elif ctype == F.NAK:
            self._rx_nak(self._over_base)                 # seqless: resend the over
        elif ctype == F.CA:
            self._rx_connect_answer(
                F.Control(ctype=F.CA, seq=0, src=self.dst, dst=self.src, bw=self.bw))
        elif ctype == F.KA:
            self._touch()
        else:
            self._touch()                                 # 'ready'/unknown: benign

    # ==================================================================== #
    # Host-driven events
    # ==================================================================== #
    def on_host_listen(self, on: bool) -> None:
        self._listen = on
        if on and self.state == State.DISCONNECTED:
            self._to(State.LISTENING)                     # [spec 05 §5.2]
        elif not on and self.state == State.LISTENING:
            self._to(State.DISCONNECTED)

    def on_host_connect(self, src: str, dst: str, bw: str) -> None:
        # DISCONNECTED --host CONNECT--> CONNECTING, key connect-request [spec 5.2/5.3]
        self.src, self.dst = src, dst
        self._set_bw(bw)
        self.role = "initiator"
        self._to(State.CONNECTING)
        self.io.on_busy(True)                                # [spec 05 §5.2]
        self._connect_retries = 0
        self._send_control(F.CR)                          # step 1 [spec 05 §5.3]
        self._t_connect = self._clock() + self.cfg.connect_timeout_s

    def on_host_data(self, blob: bytes) -> None:
        if self.state != State.CONNECTED:
            # Queue, do not drop [spec 07 §7.4]: bytes written before the link
            # comes up are buffered against the TX queue, reported via BUFFER n,
            # and flushed the moment CONNECTED fires. The source previously said
            # "a real modem drops it" and dropped them, while this package's own
            # LoopbackModem buffered — the two halves disagreed about the
            # contract, and the spec settles it against the dropping half.
            self._preconnect.append(blob)
            self._buffer_raw += len(blob)
            self.io.on_buffer(self._buffer_raw)
            return
        if self.role != "initiator":
            # Bidirectional turn (ISS<->IRS) exchange is a spec gap (§5.7,
            # in-waveform). Milestone: only the initiator sends data overs.
            self.io.log("responder-side host_data ignored (turn exchange = spec gap)")
            return
        framed = F.frame_outbound(blob)                   # length-prefix [ours]
        # chop into bandwidth-sized blocks (43 B at BW500, 89 B at BW2300)
        for i in range(0, len(framed), self._psize):
            chunk = framed[i:i + self._psize]
            chunk = chunk + bytes(self._psize - len(chunk))            # pad last
            self._blockq.append((self._next_seq % F.SEQ_MOD, chunk))
            self._next_seq += 1
        self._buffer_raw += len(blob)
        self.io.on_buffer(self._buffer_raw)                  # BUFFER n [spec 05 §5.4]
        self._maybe_start_over()

    def on_host_disconnect(self) -> None:
        if self.state not in (State.CONNECTED, State.CONNECTING):
            return
        if self._over is not None or self._blockq:
            self._disconnect_pending = True               # flush first [spec 5.2/5.6]
        else:
            self._begin_disconnect()

    def _hard_close(self, notify: bool) -> None:
        """Drop the session now: discard the queues, disarm, optionally tell the far
        end, and report DISCONNECTED. `notify` sends one unacknowledged disconnect
        burst — right for an ABORT, pointless for a link declared dead because the
        peer stopped answering."""
        self._blockq.clear()
        self._over = None
        self._buffer_raw = 0
        self._preconnect = []
        self._disarm_all()
        if notify:
            self._send_control(F.DR)
        self._finish_disconnected()

    def on_host_abort(self) -> None:
        # Immediate dirty close [spec 05 §5.6 ABORT]: the queue is discarded
        # rather than flushed, and we do not wait for the 3-burst exchange.
        #
        # But we do tell the far end. A purely local close leaves a peer that
        # has no link-death timer CONNECTED forever, and a connected station
        # refuses every later connect request — so one ABORT wedges the pair
        # permanently. On air that presents as a gateway that answered once and
        # then refuses you for no visible reason. Best-effort: one disconnect
        # burst, unacknowledged, because ABORT does not wait. [ours — spec 5.6
        # records ABORT as immediate but is silent on whether it notifies]
        notify = self.state in (State.CONNECTED, State.CONNECTING,
                                State.DISCONNECTING)
        self._hard_close(notify=notify)

    # ==================================================================== #
    # RX-frame events  (one decoded proven burst)
    # ==================================================================== #
    def on_rx_frame(self, fr: F.RxFrame) -> None:
        self.io.on_busy(True)
        if fr.is_control:
            if not fr.crc_ok or fr.control is None:
                return
            self._on_control(fr.control)
        else:
            self._on_data(fr)

    def _on_control(self, c: F.Control) -> None:
        self.io.log(f"rx CONTROL {c.name} seq={c.seq}")
        t = c.ctype
        if t == F.CR:
            self._rx_connect_request(c)
        elif t == F.CA:
            self._rx_connect_answer(c)
        elif t == F.CF:
            self._rx_connect_final(c)
        elif t == F.ACK:
            self._rx_ack(c.seq)
        elif t == F.NAK:
            self._rx_nak(c.seq)
        elif t == F.DR:
            self._rx_disc_request()
        elif t == F.DC:
            self._rx_disc_answer()
        elif t == F.DF:
            pass                                          # final; idempotent
        elif t == F.KA:
            self._touch()                                 # keepalive heard

    # ---- connect handshake (spec 05 §5.3) --------------------------------
    def _rx_connect_request(self, c: F.Control) -> None:
        if self.state not in (State.LISTENING, State.CONNECTING):
            return
        # responder: reply connect-answer  [spec 05 §5.2/5.3 step 2]
        self.role = "responder"
        # the connect-request carries the initiator's src/dst; from the
        # responder's view src/dst swap for its own CONNECTED report.
        self.src, self.dst = c.dst, c.src
        self._set_bw(c.bw or self.bw or "500")
        self._to(State.CONNECTING)
        self.io.on_busy(True)
        self.io.on_pending()                                 # PENDING [spec 05 §5.2]
        self._send_control(F.CA)                          # step 2
        self._t_connect = self._clock() + self.cfg.connect_timeout_s

    def _rx_connect_answer(self, c: F.Control) -> None:
        if self.state != State.CONNECTING or self.role != "initiator":
            return
        self._send_control(F.CF)                          # step 4/6 confirm [spec 5.3]
        self._t_connect = None
        self._enter_connected()

    def _rx_connect_final(self, c: F.Control) -> None:
        if self.state != State.CONNECTING or self.role != "responder":
            return
        self._t_connect = None
        self._enter_connected()

    def _enter_connected(self) -> None:
        self._to(State.CONNECTED)                         # [spec 05 §5.2]
        self._blocks_per_over = 1
        self._successes = 0
        self._over_base = 0
        self._next_seq = 0
        self._touch()
        self._t_keepalive = self._clock() + self.cfg.keepalive_s
        self.io.on_connected(self.src, self.dst, self.bw)    # CONNECTED src dst bw
        # Bytes the host wrote before the link came up go over the air now
        # [spec 07 §7.4]. _buffer_raw already counts them, so re-queue through
        # the framing path without double-counting.
        # Only the initiator sends data overs in this milestone — the turn exchange
        # is a spec gap — so flushing on a responder fed every held blob straight
        # into the role check in on_host_data, which logged and dropped it. The host
        # had already been told BUFFER n for those bytes, _buffer_raw was decremented
        # to match, and no further BUFFER was ever emitted: counted bytes vanished
        # and the host's flow-control view stayed wrong with nothing reporting it.
        # Hold them instead, and keep the reported depth honest.
        if self.role == "initiator":
            held, self._preconnect = self._preconnect, []
            for blob in held:
                self._buffer_raw -= len(blob)
                self.on_host_data(blob)
        elif self._preconnect:
            self.io.log(f"{self._buffer_raw} pre-connect bytes held: this station is "
                        "the responder and cannot start an over [spec 05 §5.7]")
            self.io.on_buffer(self._buffer_raw)
        self._maybe_start_over()

    # ---- ISS: send engine (stop-and-wait, per-over ACK) ------------------
    def _maybe_start_over(self) -> None:
        if self.state != State.CONNECTED or self._over is not None:
            return
        if not self._blockq:
            if self._disconnect_pending:
                self._begin_disconnect()
            return
        over = _Over()
        for _ in range(self._blocks_per_over):
            if not self._blockq:
                break
            over.blocks.append(self._blockq.popleft())
        self._over = over
        self._transmit_over()

    def _transmit_over(self) -> None:
        over = self._over
        assert over is not None
        # Encoded before the key, not between the key and the audio: framing, CRC and
        # FEC are work, and work inside the keyed region is unmodulated carrier — or,
        # if it raises, a transmitter held up by an exception on the way past the
        # key-down. The `finally` is the other half of that: the key comes down on
        # every path out, including a transmit that dies mid-over.
        n = len(over.blocks)
        bursts = [F.encode_data(block, seq, last_of_over=(i == n - 1))
                  for i, (seq, block) in enumerate(over.blocks)]
        self.io.key(True)                                 # PTT ON (one keyed over)
        try:
            for payload, marker in bursts:
                self.io.tx(payload, marker, self.bw, self._level)  # current speed level
        finally:
            self.io.key(False)                            # PTT OFF
        self.io.log(f"tx DATA over: seqs={[s for s,_ in over.blocks]} "
                    f"bpo={self._blocks_per_over} level={self._level}")
        self._t_ack = self._clock() + self.cfg.ack_timeout_s
        self._touch()

    def _rx_ack(self, acked_seq: int) -> None:
        if self._over is None:
            return
        if acked_seq != self._over.last_seq:
            return                                        # stale/duplicate ACK
        acked_blocks = len(self._over.blocks)
        self._over = None
        self._t_ack = None
        # BUFFER decrements per ACKed block [spec 05 §5.4]
        self._buffer_raw = max(0, self._buffer_raw - acked_blocks * self._psize)
        # gear-shift: 1 -> 2 blocks/over after gear_up_after successes [spec 5.4]
        self._successes += 1
        if (self._successes >= self.cfg.gear_up_after
                and self._blocks_per_over < self.cfg.max_blocks_per_over):
            self._blocks_per_over += 1
            self.io.log(f"gear-shift: blocks/over -> {self._blocks_per_over}")
        self._speed_up_maybe()                            # speed-level up-shift [spec 6.2]
        if not self._blockq:
            self._buffer_raw = 0
        self.io.on_buffer(self._buffer_raw)
        self._maybe_start_over()

    def _rx_nak(self, expected_seq: int) -> None:
        # decode failure at the peer -> retransmit the over [spec 05 §5.4]
        self._retransmit_over()

    def _retransmit_over(self) -> None:
        if self._over is None:
            return
        self._over.retries += 1
        self._successes = 0
        if self._blocks_per_over > 1:                     # back off the gear
            self._blocks_per_over = 1
        self._speed_down()                                # speed-level down-shift [spec 6.3]
        if self._over.retries > self.cfg.max_retries:     # give up [ours — spec gap]
            self.io.log("max retries exceeded -> abort")
            self.on_host_abort()
            return
        self.io.log(f"retransmit over (retry {self._over.retries})")
        self._transmit_over()

    # ---- speed-level gear-shift (spec 06 §6.2/6.3) -----------------------
    def _speed_up_maybe(self) -> None:
        if not self.cfg.speed_gearshift or len(self._ladder) < 2:
            return
        self._speed_successes += 1
        i = self._ladder.index(self._level)
        if self._speed_successes >= self.cfg.speed_up_after and i + 1 < len(self._ladder):
            self._level = self._ladder[i + 1]
            self._speed_successes = 0
            self.io.log(f"speed up-shift: level -> {self._level}")

    def _speed_down(self) -> None:
        if not self.cfg.speed_gearshift:
            return
        self._speed_successes = 0
        if self._level != self._ladder[0]:
            self._level = self._ladder[0]                 # drop to robust base
            self.io.log(f"speed down-shift: level -> {self._level}")

    # ---- IRS: receive engine --------------------------------------------
    def _on_data(self, fr: F.RxFrame) -> None:
        if self.state != State.CONNECTED:
            return
        self._touch()
        if not fr.crc_ok:
            self._nak_reset()                             # decode failure
            return
        seq = fr.seq
        expected_in_over = (self._over_base + len(self._over_blocks)) % F.SEQ_MOD
        if seq == expected_in_over:
            self._over_blocks.append(fr.payload)
            if fr.last_of_over:                           # commit the whole over
                for blk in self._over_blocks:
                    for blob in self._reasm.feed(blk):
                        self.io.on_deliver(blob)             # -> host data port
                self._over_base = (seq + 1) % F.SEQ_MOD
                self._over_blocks = []
                self._send_control(F.ACK, seq=seq)        # one ACK/over [spec 5.4]
        elif seq == (self._over_base - 1) % F.SEQ_MOD and not self._over_blocks:
            # duplicate of the last already-ACKed over -> re-ACK, suppress [spec 5.4]
            self._send_control(F.ACK, seq=seq)
        else:
            self._nak_reset()                             # gap / wrong seq

    def _nak_reset(self) -> None:
        self._over_blocks = []
        self._send_control(F.NAK, seq=self._over_base)

    # ---- disconnect (spec 05 §5.6, 3-burst) ------------------------------
    def _begin_disconnect(self) -> None:
        self._disconnect_pending = False
        self.role = self.role or "initiator"
        self._to(State.DISCONNECTING)
        self._send_control(F.DR)                          # burst 1
        self._t_connect = self._clock() + self.cfg.connect_timeout_s

    def _rx_disc_request(self) -> None:
        self._to(State.DISCONNECTING)
        self._send_control(F.DC)                          # burst 2
        self._finish_disconnected()

    def _rx_disc_answer(self) -> None:
        self._send_control(F.DF)                          # burst 3
        self._finish_disconnected()

    def _finish_disconnected(self) -> None:
        self._disarm_all()
        self._blockq.clear(); self._over = None; self._over_blocks = []
        self._buffer_raw = 0
        self.io.on_disconnected()                            # DISCONNECTED [spec 5.2]
        self.io.on_busy(False)                               # BUSY OFF
        self._to(State.LISTENING if self._listen else State.DISCONNECTED)

    # ==================================================================== #
    # Timer tick
    # ==================================================================== #
    def on_timer(self) -> None:
        now = self._clock()
        if self._t_ack is not None and now >= self._t_ack:
            self._t_ack = None
            self.io.log("ACK timeout")
            self._retransmit_over()
        if self._t_connect is not None and now >= self._t_connect:
            self._t_connect = None
            self._on_connect_timeout()
        if (self._t_keepalive is not None and self.state == State.CONNECTED
                and now >= self._t_keepalive):
            self._t_keepalive = now + self.cfg.keepalive_s
            if self._over is None and now - self._last_activity >= self.cfg.keepalive_s:
                self._send_control(F.KA)                  # on-air keepalive [spec 5.5]
        if (self.state == State.CONNECTED and self.cfg.link_death_s
                and now - self._last_activity >= self.cfg.link_death_s):
            self.io.log(f"link dead: no traffic for {self.cfg.link_death_s:.0f}s")
            self._hard_close(notify=False)

    def _on_connect_timeout(self) -> None:
        if self.state == State.CONNECTING and self.role == "initiator":
            self._connect_retries += 1
            if self._connect_retries > self.cfg.max_connect_retries:
                self.io.log("connect retries exhausted -> abort")
                self.on_host_abort()
                return
            self._send_control(F.CR)                      # resend connect-request
            self._t_connect = self._clock() + self.cfg.connect_timeout_s
        elif self.state == State.CONNECTING and self.role == "responder":
            self._send_control(F.CA)                      # resend connect-answer
            self._t_connect = self._clock() + self.cfg.connect_timeout_s
        elif self.state == State.DISCONNECTING:
            self._finish_disconnected()                   # disconnect stuck -> force

    def _disarm_all(self) -> None:
        self._t_ack = self._t_connect = self._t_keepalive = None
