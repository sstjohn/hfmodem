# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ARDOP ARQ session state machine — besra's NEWSTATE engine.

Pure frame-level protocol logic: it decides which ARDOP frame to transmit, drives
the eight-token NEWSTATE machine (`besra.host.protocol`), and reports link events
to a host `Observer`. No DSP — it speaks to the radio only through a `Transport`
that renders one frame (`besra.frame`) at a time, and it is clocked externally so
tests are deterministic.

The transitions implement the ARDOP ARQ rules of `docs/protocols/ardop/11-WAVEFORM.md` §2.3: the
ConReq/ConAck/ConAck/ACK connect handshake, stop-and-wait data with even/odd
repeat detection and quality ACK/NAK, BREAK turnover, and DISC/END teardown.

v1 simplifications (reference behaviour noted):
  * The FSK→PSK→QAM ladder is climbed as far as 4PSK at every bandwidth and
    no further (`_DATA_LADDER`); the 8PSK and 16QAM rungs the reference puts
    above it stay out of the table.
    In the other direction the *peer* shifts on the decode quality we report in
    every DATAACK/DATANAK (`Gearshift_9`), so those numbers are measured
    (:mod:`besra.phy.quality`) rather than asserted — a receiver that always claims
    100 talks its peer up into modes it cannot read.
  * No memory-ARQ sample averaging (that lives in the demodulator, not here).
  * A NAK retransmits immediately rather than waiting for the repeat timer. The
    outcome is NOT identical, and calling it so was wrong: in the reference a NAK
    means SHIFT DOWN, not "send that again". `ARQ.c:2210` gearshifts on the NAK's
    quality and `ARQ.c:2212` reaches `SendData()` only `if (intShiftUpDn != 0)`
    — an unshifted NAK changes nothing, and what puts the frame back on the air
    is the repeat timer the ISS was already running from its own frame end
    (2.0 s against the gateways this station has worked). Only an ACK stops that
    timer (`ARQ.c:2143`). This end still retransmits on the NAK, which keeps a
    synchronous relay deterministic without an interleaved tick; what it must not
    do is read a peer's NAK as a repeat request, or read a peer's repeat as
    evidence its NAK was heard.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol

from .. import crc
from ..frame import callsign
from ..frame import frame as F
from ..host import protocol as P

log = logging.getLogger(__name__)


class Transport(Protocol):
    def send(self, frame_type: int, payload: bytes, session_id: int) -> float:
        """Transmit a frame; returns its transmit duration in seconds so repeats
        can be scheduled from when the frame ends, not when it starts."""
        ...


class Observer(Protocol):
    def newstate(self, state: str) -> None: ...
    def connected(self, remote: str, bw: int) -> None: ...
    def disconnected(self) -> None: ...
    def data_received(self, kind: str, blob: bytes) -> None: ...
    def buffer(self, nbytes: int) -> None: ...
    def pending(self, cancel: bool = False) -> None: ...
    def target(self, call: str) -> None: ...
    def status(self, text: str) -> None: ...


# -- frame-type lookups from the catalog -------------------------------------

def _type_of(name: str) -> int:
    for t, fd in F.FRAMES.items():
        if fd.name == name:
            return t
    raise KeyError(name)


BREAK = _type_of("BREAK")
IDLE = _type_of("IDLE")
DISC = _type_of("DISC")
END = _type_of("END")
CONREJ_BW = _type_of("ConRejBW")

_CONREQ_MAX = {200: "ConReq200M", 500: "ConReq500M", 1000: "ConReq1000M", 2000: "ConReq2000M"}
_CONREQ_FORCED = {200: "ConReq200F", 500: "ConReq500F", 1000: "ConReq1000F", 2000: "ConReq2000F"}
_CONACK = {200: "ConAck200", 500: "ConAck500", 1000: "ConAck1000", 2000: "ConAck2000"}

CONREQ_MAX = {bw: _type_of(n) for bw, n in _CONREQ_MAX.items()}
CONREQ_FORCED = {bw: _type_of(n) for bw, n in _CONREQ_FORCED.items()}
CONACK = {bw: _type_of(n) for bw, n in _CONACK.items()}
_CONREQ_BW = {t: bw for bw, t in CONREQ_MAX.items()} | {t: bw for bw, t in CONREQ_FORCED.items()}
_CONACK_BW = {t: bw for bw, t in CONACK.items()}

#: Transmit data modes per session bandwidth, most robust rung first (even types;
#: the odd twin is +1), each paired with the average ACK quality that has to be
#: beaten to leave it — None on the top rung, which is never left.
#:
#: Modes and thresholds are the reference's own, in its order — `ARQ.c`'s
#: DataModes200/500/1000/2000 read against the byt200/500/1000/2000 of
#: `GetShiftUpThresholds`, whose numbers were measured over a pink-noise channel
#: rather than chosen: the quality the next mode down reads at the minimum S:N the
#: mode above it decodes at.
#:
#: Every ladder is cut below the 8PSK and 16QAM rungs, which is where this
#: station's evidence stops and the reference's table goes on. 4PSK the paths
#: carry: WW2MI sent us 4PSK.200.100 all through 2026-08-23 while grading our 4FSK
#: at a median 96 — 45 requests to speed up that went nowhere, at 4.9 B/s against
#: the 12.2 B/s of the rung already in use — and KE8LVA's greeting is
#: 4PSK.500.100. The phase-dense modes stay out until a run says the same of them.
_DATA_LADDER: dict[int, tuple[tuple[int, int | None], ...]] = {
    200: ((0x48, 82), (0x42, 84), (0x40, None)),
    500: ((0x48, 80), (0x42, 84), (0x40, 84), (0x50, None)),
    1000: ((0x4C, 80), (0x4A, 80), (0x50, 80), (0x60, None)),
    2000: ((0x4C, 80), (0x4A, 80), (0x50, 80), (0x60, 76), (0x70, None)),
}

#: Weight the exponential averager gives each newly reported quality, and the counts
#: that gate a shift: two ACKs at the current rung to leave it at all, five before a
#: rung that failed the instant we reached it is tried a second time, and two NAKs to
#: come back down — one where the rung has never ACKed anything, which is
#: `Gearshift_9`'s `DownNAKS` read against `ModeHasWorked` (ARQ.c ComputeQualityAvg,
#: Gearshift_9; `ArdopGearshift.Gearshift9` in the M0LTE port is the same test).
_QUALITY_ALPHA = 0.5
_ACKS_TO_SHIFT_UP = 2
_ACKS_TO_RETRY_RUNG = 5
_NAKS_TO_SHIFT_DOWN = 2

DATAACK_MIN, DATAACK_MAX = 0xE0, 0xFF
DATANAK_MIN, DATANAK_MAX = 0x00, 0x1F

#: nominal received-leader time carried by ConAck, tens of ms ×3 (spec §2.3).
#: besra measures no leader, so it states the one it sends —
#: `phy.modulator.DEFAULT_LEADER_MS` / 10, which is what a peer running the same
#: leader would have measured off us.
_LEADER_TENS = 24

#: No session has been party to this end yet. It has to be a value no wire id can
#: take, and 0 is not one: `crc.session_id` remaps its reserved 0xFF to 0x00, so
#: zero is a *doubly* likely real id — 0.77% of random call pairs hash to it,
#: measured, among them ``NS0A -> K5DAT-13``, both callsigns this station has
#: worked. Starting at 0 made a fresh besra a party to every one of those
#: sessions: it answered a stranger's DISC with an END, which is the exact defect
#: the session filter in `on_receive` was added to stop.
_NO_SESSION = -1


def _is_conreq(t: int) -> bool:
    return t in _CONREQ_BW


def _is_conack(t: int) -> bool:
    return t in _CONACK_BW


def _is_dataack(t: int) -> bool:
    return DATAACK_MIN <= t <= DATAACK_MAX


def _is_datanak(t: int) -> bool:
    return DATANAK_MIN <= t <= DATANAK_MAX


def _is_data(t: int) -> bool:
    fd = F.FRAMES.get(t)
    return fd is not None and (fd.name.endswith(".E") or fd.name.endswith(".O"))


def _canon_call(call: str) -> str:
    """The canonical StationId spelling ardopcf hashes into the session ID:
    uppercase, SSID normalised, ``-0`` dropped. Round-tripping through the wire
    codec yields exactly the form both ends recompute from the ConReq payload, so
    besra's session ID agrees with a real ardopcf's for any input spelling."""
    return callsign.unpack_callsign(callsign.pack_callsign(call))


def _dataack_for(quality: int) -> int:
    """DATAACK frame type carrying a decode quality (Q = 38 + 2·code)."""
    return DATAACK_MIN + max(0, min(31, (quality - 38) // 2))


def _datanak_for(quality: int) -> int:
    return DATANAK_MIN + max(0, min(31, (quality - 38) // 2))


def peer_quality(frame_type: int) -> int | None:
    """The decode quality a DATAACK/DATANAK carries: the peer's grade of *our*
    transmission, and the only transmit-rate feedback ARDOP has."""
    if _is_dataack(frame_type):
        return 38 + 2 * (frame_type - DATAACK_MIN)
    if _is_datanak(frame_type):
        return 38 + 2 * (frame_type - DATANAK_MIN)
    return None


def _mode_name(frame_type: int) -> str:
    return F.FRAMES[frame_type].name.removesuffix(".E").removesuffix(".O")


#: The quality reported before anything with a body has been decoded. Every path that
#: reaches a data ACK has demodulated a ConAck or a data frame first, so this stands
#: only for a session driven frame-by-frame from a test.
_QUALITY_UNMEASURED = 100

#: Turnover and keepalive ACKs acknowledge a control frame, not data — there is no
#: decode to grade, and the reference passes a literal 100 at exactly these sites
#: (ARQ.c: BREAK in IRSData, IDLE in IRSData, BREAK in ISS/IDLE). They never reach the
#: peer's gearshift either, which averages quality only over ACKs of data frames
#: (``blnLastFrameSentData``), so the constant is the honest value here.
_QUALITY_NOT_DATA = 100

#: Consecutive frames this end could not read that draw a NAK before a stint
#: which has delivered nothing is closed deliberately, because an unbounded NAK
#: train at a peer we have never read is the keepalive problem again with a
#: different frame in it, and it is a gateway's transmitter it spends as well as
#: ours.
#:
#: It used to close the link on the count alone. The premise was that a NAK buys
#: a more robust mode over the same fade, so a frame that fails identically after
#: two of them is no longer about the channel and the fault must be in this
#: receiver. W6IDS disproved that on 2026-08-26: RMS Trimode 1.4.2.2 sent
#: 4PSK.200.100 before both NAKs and after both, with two rungs of the 500 Hz
#: ladder unused beneath it, and the receiver it was blamed on had read 46 of the
#: session's 51 frames — the five that failed graded 55-63 where every frame that
#: read graded 64-80, which is a fade at the margin. The link ended four frames
#: short of the second of three messages the CMS had been told we would take.
#: Nothing this end can read tells a peer that will not downshift from its own
#: deafness, so the count cannot carry that conclusion and no longer draws it.
#:
#: Delivery decides instead. A stint that has passed payload to the application
#: has shown this receiver reading this peer, and its NAKs stand — bounded, as
#: the reference bounds them, by the progress clock: ARQ.c NAKs an undecodable
#: data frame without limit and does not touch `dttTimeoutTrip` doing it, so a
#: stint that goes on failing runs `_timeout_s` down and ends there.
#:
#: The budget is spent per RUNG, not per stint, because that is what a NAK buys.
#: `Gearshift_9` drops the ISS one rung per `DownNAKS` NAKs — 2, or 1 on a rung
#: that has never ACKed — so a peer answering correctly cannot reach the bottom of
#: a six- or seven-rung ladder on two NAKs, and a budget that never renews ends the
#: link somewhere in the middle of the descent. KN4LQN, 2026-08-26 02:24z: two NAKs
#: at `4PSK.500.100`, the gateway shifted down exactly as the reference says it
#: should, and its first `4FSK.500.100` — the best-received body of the whole slot,
#: and the one rung from the floor — drew a DISC instead of the third NAK. A frame
#: type this stint has not NAKed before is the peer having moved, so the count
#: restarts on it; a type already tried buys nothing and does not renew, which
#: keeps the whole descent finite.
_UNREPAIRED_BUDGET = 2

#: The same count for the stint that has delivered nothing, where the reasoning
#: above does not reach. Delivery is what licenses the standing NAK train, and the
#: greeting is by construction the one stint that has never delivered — so the
#: budget that was only ever meant to bound a peer this end has never read was
#: also, every session, the gate the link had to get through to start.
#:
#: W6IDS closed two of four connects on it on 2026-08-29, both 41 s in and both
#: before the banner: three header-only reads at q 51-54 and 52-54, `4PSK.200.100`
#: throughout, no descent, so the per-rung renewal never fired. The premise that a
#: repeat failing the same way is this receiver's fault was refuted three ways that
#: afternoon. The operator heard the far end still transmitting on both dead arms.
#: The channel sense read `+4.8 dB tone at 1512 Hz -> OCCUPIED` 60 s after arm 1's
#: DISC — our own peer, still keying on our own centre, after we had left. And on
#: arm 3 the very next copy read, on a stint whose quality climbed 59 -> 84 and
#: which went on to a completed B2F login. `theirq` ran 94, 96 and 100: nothing
#: about those arms was a transmit-side fault.
#:
#: Six, because arm 3 is the measurement: its greeting cost twelve frames over 56 s
#: with unreadable ones among them, and survived only because good frames kept
#: resetting the count. Six is half that run, and it is where the count stops being
#: the binding bound — W6IDS's greeting frames arrived 12.4-14.5 s apart, so the
#: seventh lands about 82 s past the turnover against a 90 s deadline, and a slower
#: peer reaches the deadline first. Either way the stint is finite: an unreadable
#: frame is the one arrival that postpones nothing (`_unreadable` NAKs without
#: `_progress`). The cost is that a link nobody is filling holds the channel for up
#: to `_timeout_s` from the turnover rather than the 41 s it cost W6IDS.
_UNDELIVERED_BUDGET = 6

#: Sized to carry an ardopcf IRS's first ConAck and five repeats of it: that end
#: repeats on the 2.0 s grid with no budget at all, so a peer that has answered goes
#: on answering well after our own calling stops. Nothing has ever been heard beyond
#: it — five recordings kept 92-182 s past the last ConReq hold no gateway frame.
#: ardopcf tears down where the repeats run out, so this departs from the reference.
_CONNECT_TAIL_S = 15.0

#: Conceding ACKs to repeat at a peer that goes on breaking. A concession that is
#: read is acted on at once — the one KY4RY demonstrably read on 2026-08-14 was
#: followed by its next data frame with the even/odd parity advanced, which nothing
#: but an ACK does, and by nothing else on the air in between — so a run of them is
#: a run of answers the peer is not reading, and neither end bounds that run on its
#: own. ardopcf breaks on a randomised 1-3 s clock with no repeat limit at all
#: (ARQ.c ComputeInterFrameInterval; IRStoISS falls past GetNextARQFrame's budgeted
#: cases to the catch-all), and this end re-ACKed every one and refreshed the
#: liveness clock doing it, so the 90 s session timeout stayed out of reach too.
#: Measured that day: 16 BREAKs, 16 conceding ACKs, 84 s in which the gateway
#: repeated its whole welcome greeting and the mail parser above was handed both
#: copies concatenated. Five spans ~10 s of the peer's own repeat clock — long
#: enough that only a link failing anyway reaches it, short enough that the host
#: gets the failure while a reconnect is still cheap.
_CONCESSION_BUDGET = 5


# -- ARQ sub-states within the connect handshake and data phase --------------

class _Sub(Enum):
    NONE = 0
    ISS_CONREQ = 1     # caller repeating ConReq, awaiting ConAck
    ISS_CONACK = 2     # caller sent confirming ConAck, awaiting first ACK
    ISS_DATA = 3       # caller exchanging data
    IRS_CONACK = 4     # callee sent ConAck, awaiting the confirming ConAck
    IRS_DATA = 5       # callee receiving data
    IRS_FROM_ISS = 6   # new IRS after a role swap, first data completes it


class ArqSession:
    """One end of an ARDOP ARQ link. Feed it decoded frames via `on_receive`,
    outbound payload via `queue_data`, and wall-clock via `tick`; it drives the
    `Transport` and reports through the `Observer`."""

    def __init__(self, mycall: str, transport: Transport, observer: Observer,
                 *, bandwidth: int = 500, timeout_s: float = 90.0,
                 listen: bool = False) -> None:
        self.mycall = mycall
        self._tx = transport
        self._obs = observer
        self.bandwidth = bandwidth           # our ARQBW ceiling (MAX-style)
        self._timeout_s = timeout_s
        self.listen = listen

        self._state = P.ArdopState.DISC
        self._sub = _Sub.NONE
        self._session = _NO_SESSION
        self._remote = ""
        self._session_bw = bandwidth
        self._now = 0.0

        # outbound / stop-and-wait bookkeeping (ISS side)
        self._outbound = bytearray()
        self._tx_even = True                 # next new data frame's even/odd parity
        self._last_data_type = -1            # data frame currently outstanding
        self._in_process = 0                 # bytes of _outbound in that frame
        self._data_interval = 2.0            # DATA repeat period (ardopcf intFrameRepeatInterval)
        self._next_data_repeat = 0.0         # when to retransmit the outstanding DATA
        self._frame_was_repeated = False     # outstanding frame has been sent more than once
        self._ack_guard_until = 0.0          # ignore stale ACKs until this time (half-duplex)
        self._turnaround_s = 1.0             # one link turnaround for the stale-ACK guard

        # transmit rate selection (ardopcf Gearshift_9)
        self._rung = 0                       # index into _DATA_LADDER[self._session_bw]
        self._avg_quality = 0                # exponential average of the peer's grades
        self._acks_at_rung = 0               # ACKs since the last shift
        self._naks_at_rung = 0               # ...and NAKs, which an ACK also clears
        self._rungs_tried: set[int] = set()  # rungs shifted up into this session
        self._rungs_worked: set[int] = set() # ...and those that then ACKed anything

        # IRS-side dedup
        self._last_rx_type = -1              # last accepted data frame
        self._last_rx_payload: bytes | None = None   # ...and its bytes, which outlive a reversal
        self._stint_delivered = False        # ...and whether one reached the application
        self._last_acked_type = -1           # last data frame we ACKed
        self._heard_once = -1                # bare control awaiting its corroborating repeat
        # Decode quality of the last frame that had a body to grade — the reference's
        # `intLastRcvdFrameQuality`. Bare control frames carry no body, so the previous
        # reading stands, there and here.
        self._rx_quality = _QUALITY_UNMEASURED

        # connect / teardown timers
        self._conreq_type = 0
        self._conreq_repeats = 0
        self._conreq_budget = 0
        self._connect_deadline = 0.0
        self._connect_interval = 2.0
        self._next_conreq = 0.0
        self._listening_out = False          # calling over, still hearing (see `_listen_out`)
        self._conack_repeats = 0             # ConAck-leg repeat counter (both roles)
        self._conack_budget = 0
        self._next_conack = 0.0
        self._disc_repeating = False
        self._disc_repeats = 0
        self._last_progress = 0.0            # last time the exchange got somewhere
        self._next_chirp = 0.0               # next IDLE-keepalive time
        self._next_break_repeat = 0.0        # next BREAK resend while IRStoISS
        self._concessions = 0                # conceding ACKs sent since we yielded
        self._unrepaired = 0                 # frames that arrived and would not read
        self._unrepaired_modes: set[int] = set()   # the rungs those frames came on
        self._rx_epoch = 0                   # bumps where memory ARQ must be dropped

    # -- public state ------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def connected(self) -> bool:
        return self._state in P.CONNECTED_STATES

    @property
    def expected_session(self) -> int | None:
        """The session id a bare control addressed to us carries right now — known
        from the moment a connect is initiated or answered — or None when no
        session is in progress. The demodulator trades it for sensitivity: a
        control frame matching it is corroborated, one from nobody is not."""
        return self._session if self._state != P.ArdopState.DISC else None

    @property
    def queued(self) -> int:
        """Outbound bytes still to go — what the host reads as BUFFER."""
        return len(self._outbound)

    @property
    def rx_epoch(self) -> int:
        """Counter the demodulator keys its memory ARQ on. It moves exactly where
        the reference calls `ResetMemoryARQ` from ARQ.c — every one of those five
        sites is a `SetARDOPProtocolState(IRS)`, this end taking the receiving role
        — plus the teardown to DISC. Across either of those a repeat of a frame
        type is a new block, not another copy of the last one, and averaging the
        two would hand a fresh carrier the phases of a frame from before the turn.
        """
        return self._rx_epoch

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            if state in (P.ArdopState.IRS, P.ArdopState.DISC):
                self._rx_epoch += 1
            self._obs.newstate(state)

    # -- host verbs --------------------------------------------------------

    def connect(self, target: str, repeats: int = 10) -> None:
        """Become the caller: fire the first ConReq and await ConAck."""
        if self._state != P.ArdopState.DISC:
            return
        self._remote = _canon_call(target)
        self._session = crc.session_id(_canon_call(self.mycall), self._remote)
        self._session_bw = self.bandwidth
        forced = False                       # v1: MAX-style negotiation only
        self._conreq_type = (CONREQ_FORCED if forced else CONREQ_MAX)[self.bandwidth]
        self._conreq_repeats = 0
        self._conreq_budget = repeats
        self._listening_out = False
        self._reset_rate()
        self._connect_deadline = self._now + self._timeout_s
        self._sub = _Sub.ISS_CONREQ
        self._set_state(P.ArdopState.ISS)
        self._progress()
        payload = callsign.pack_callsign(self.mycall) + callsign.pack_callsign(target)
        # Schedule the repeat from when the ConReq *ends*: a ConReq is ~1.75 s and
        # the ConAck arrives after the turnaround, so anchoring at send-start left
        # only ~0.25 s to hear a 0.70 s ConAck — the next ConReq keyed over it.
        dur = self._tx.send(self._conreq_type, payload, self._session)
        self._next_conreq = self._now + dur + self._connect_interval

    def queue_data(self, data: bytes) -> None:
        if not data:
            return
        self._outbound += data
        self._obs.buffer(len(self._outbound))
        # Kick the pipe if we're idle and own (or can seize) the link.
        if self._state == P.ArdopState.ISS and self._sub == _Sub.ISS_DATA and self._last_data_type < 0:
            self._send_data()
        elif self._state == P.ArdopState.IDLE:
            self._set_state(P.ArdopState.ISS)
            self._sub = _Sub.ISS_DATA
            self._send_data()
        elif self._state == P.ArdopState.IRS and self._sub == _Sub.IRS_DATA:
            self._begin_break()

    def purge(self) -> None:
        """Empty the outbound queue, leaving the link alone (host PURGEBUFFER).

        The frame already in flight is kept: stop-and-wait deletes those bytes
        only when the peer ACKs them, so dropping them here would step the queue
        past data the peer never received."""
        del self._outbound[self._in_process:]
        self._obs.buffer(len(self._outbound))

    def disconnect(self) -> None:
        """Graceful close: DISC (retried up to 5×) awaiting END."""
        if self._state == P.ArdopState.DISC:
            return
        self._obs.status("INITIATING ARQ DISCONNECT")
        self._disc_repeating = True
        self._disc_repeats = 1
        self._progress()
        self._tx.send(DISC, b"", self._session)

    def abort(self) -> None:
        """Dirty disconnect: drop everything immediately."""
        self._teardown(notify=True)

    def tick(self, now: float) -> None:
        self._now = now

        if self._sub == _Sub.ISS_CONREQ:
            self._tick_connect()
            return

        if self._sub in (_Sub.IRS_CONACK, _Sub.ISS_CONACK):
            self._tick_conack()
            return

        if self._disc_repeating:
            self._tick_disc()
            return

        if self.connected and now - self._last_progress > self._timeout_s:
            self._obs.status(f"ARQ Timeout from Protocol State: {self._state}")
            self._progress()
            self._tx.send(DISC, b"", self._session)
            self._teardown(notify=True)
            return

        # Repeat an outstanding DATA frame until it is ACKed: a lost DATA or lost
        # DATAACK is otherwise never retried. The resend does not `_touch()`, so a
        # truly dead channel still trips the session timeout above.
        if self._state == P.ArdopState.ISS and self._sub == _Sub.ISS_DATA \
                and self._last_data_type >= 0 and now >= self._next_data_repeat:
            self._frame_was_repeated = True
            dur = self._tx.send(self._last_data_type,
                                bytes(self._outbound[:self._in_process]), self._session)
            self._next_data_repeat = now + dur + self._data_interval

        # A seizure is half done until the concession is heard: if the peer's
        # conceding ACK is lost, both ends would otherwise sit silent — the
        # peer as a conceded IRS, this end waiting in IRStoISS — until the
        # session timeout. So the BREAK repeats on its own clock, like an
        # outstanding DATA frame, and without `_touch()` for the same reason.
        if self._state == P.ArdopState.IRStoISS \
                and now >= self._next_break_repeat:
            dur = self._tx.send(BREAK, b"", self._session)
            self._next_break_repeat = now + dur + self._data_interval

        # Idle chirp: ISS with an empty queue keeps the link alive with IDLE.
        # It advances only its own timer, never _last_progress, so a dead channel still
        # trips the session timeout above (reference: repeats bypass SendData).
        if self._state in (P.ArdopState.ISS, P.ArdopState.IDLE) and self._sub == _Sub.ISS_DATA \
                and not self._outbound and self._last_data_type < 0 \
                and now >= self._next_chirp:
            self._set_state(P.ArdopState.IDLE)
            dur = self._tx.send(IDLE, b"", self._session)
            self._next_chirp = now + dur + 2.0

    # -- receive dispatch --------------------------------------------------

    def on_receive(self, frame_type: int, payload: bytes, session_id: int, ok: bool,
                   quality: int | None = None, header_only: bool = False) -> None:
        fd = F.FRAMES.get(frame_type)
        if fd is None:
            return
        # ardopcf demodulates the frame-type header against the session id it is
        # party to (SoundInput.c, ComputeDecodeDistance), so another session's
        # frames never reach its protocol layer. besra decodes (type, session)
        # jointly, so the equivalent address filter lives here: a frame stamped
        # with a foreign session id is someone else's traffic and must not drive
        # this state machine. Measured on air (2026-08-05, 7103.5 kHz): a
        # third-party QSO's DISC was answered with an END bearing our stale
        # session id. ConReq/Ping/ID travel with the forced 0xFF wire id and are
        # addressed by callsign instead — their handlers check the target. In
        # DISC state ``_session`` is the id of the last session, which makes the
        # DISC branch below exactly ardopcf's rule 1.5 (END for a peer whose
        # earlier END was lost, keyed by ``bytLastARQSessionID``) — and before the
        # first session it is `_NO_SESSION`, which no wire id can equal, so a
        # station that has never connected is party to nothing.
        if not fd.forces_session and session_id != self._session:
            return
        # A header-only frame's type is read off ten tones and corroborated by
        # nothing else (`Demodulator._scan_header`): it is evidence that a frame
        # arrived and no evidence of which one. So it may ask for that frame again
        # and may do nothing else — a wrong guess costs one retransmit that way
        # and the payload every other way. Replayed over the ARDOP captures in
        # logs/onair: 26 of 161 header-only data frames read as the type this end
        # had just ACKed, and six of those drew a DATAACK onto the air 0.34-1.08 s
        # later for a body that never decoded. Two more of the 163 read as a
        # control type and fell through `_rx_irs` unanswered, since every branch
        # there needs an `ok` a guess does not have. None of the 163 was reported
        # ok=True, so the other states, whose handlers all require it, saw none of
        # them before this gate and see none after it.
        #
        # ONLY FOR A FRAME OF OURS. The forced-session types reach this line by
        # being addressed by callsign rather than by session, and the callsigns
        # live in a payload a guess does not have — so a guessed one names nobody,
        # and answering it NAKs and then disconnects a peer that sent none of it.
        # A control frame is corroborated by its repeat, and only a *consecutive*
        # one: a data frame in between is the peer getting on with the link, which
        # leaves any control heard before it stale.
        if _is_data(frame_type):
            self._heard_once = -1
        if header_only:
            if not fd.forces_session and self._state is P.ArdopState.IRS \
                    and self._sub is not _Sub.IRS_CONACK:
                self._unreadable(frame_type, quality)
            return
        # What we report back grades *this session's* frames, so the reading is taken
        # only from frames addressed to it. ConReq/Ping/ID cleared the filter above on
        # the forced 0xFF wire id rather than by being ours: they are unconnected
        # traffic, from a stranger as readily as from the peer, and a 12-byte 4FSK ID
        # is not a grade of the data path whatever it reads. Measured on
        # logs/onair/20260805T235717Z-besra-7102000.wav: the ID frames the demodulator
        # reads off it score 76 (t=46.9) and 90 (t=169.3), while the session's own
        # frames read 81 at the ConAck and 19-54 across the data frames it was failing
        # to decode. The reference's global is set at demodulation and so takes both,
        # which is how a gateway gets told the path is better than it is and
        # gearshifts into modes we cannot read. A guess is not a grade either, and
        # its own reading goes to the NAK it draws instead.
        if quality is not None and not fd.forces_session:
            self._rx_quality = quality
        handler = {
            P.ArdopState.DISC: self._rx_disc,
            P.ArdopState.ISS: self._rx_iss,
            P.ArdopState.IRS: self._rx_irs,
            P.ArdopState.IDLE: self._rx_idle,
            P.ArdopState.IRStoISS: self._rx_irs_to_iss,
        }.get(self._state)
        if handler is not None:
            handler(frame_type, payload, ok)

    # DISC: a listening responder answers a ConReq addressed to us.
    def _rx_disc(self, ft: int, payload: bytes, ok: bool) -> None:
        if ok and ft == DISC:
            # A stray DISC from a prior session whose END was lost.
            self._tx.send(END, b"", self._session)
            return
        if not ok or not _is_conreq(ft) or not self.listen:
            return
        if len(payload) < 12:
            return
        # A heard ConReq pauses a scanning host until we know whose it is; every
        # path out of here either answers it or withdraws the PENDING.
        self._obs.pending()
        caller = callsign.unpack_callsign(payload[0:6])
        target = callsign.unpack_callsign(payload[6:12])
        if target != self.mycall:
            self._obs.pending(cancel=True)
            return
        self._obs.target(target)
        self._answer_conreq(ft, caller, target)

    def _answer_conreq(self, ft: int, caller: str, target: str) -> None:
        reply = self._negotiate(ft)
        self._remote = caller
        self._session = crc.session_id(caller, target)
        if reply is None:
            self._obs.status(f"ARQ CONNECTION FROM {caller} REJECTED, INCOMPATIBLE BW")
            self._obs.pending(cancel=True)
            self._tx.send(CONREJ_BW, b"", self._session)
            return
        self._session_bw = _CONACK_BW[reply]
        self._sub = _Sub.IRS_CONACK
        self._reset_rate()
        self._reset_stint()
        self._set_state(P.ArdopState.IRS)
        self._progress()
        dur = self._tx.send(reply, bytes([_LEADER_TENS]) * 3, self._session)
        self._arm_conack(dur)

    def _negotiate(self, conreq_type: int) -> int | None:
        """The ConAck type granting a ConReq under our bandwidth ceiling, or None
        (ConRejBW). v1: MAX-style — negotiate down to the lower of the two."""
        req_bw = _CONREQ_BW[conreq_type]
        forced = conreq_type in CONREQ_FORCED.values()
        granted = min(req_bw, self.bandwidth)
        if forced and req_bw > self.bandwidth:
            return None
        return CONACK[granted]

    # ISS: caller through the handshake, then data.
    def _rx_iss(self, ft: int, payload: bytes, ok: bool) -> None:
        if self._sub == _Sub.ISS_CONREQ:
            if ok and _is_conack(ft):
                # Grant received — echo a confirming ConAck, await first ACK.
                self._session_bw = _CONACK_BW[ft]
                self._sub = _Sub.ISS_CONACK
                self._progress()
                dur = self._tx.send(ft, bytes([_LEADER_TENS]) * 3, self._session)
                self._arm_conack(dur)
            elif ok and ft == CONREJ_BW:
                self._obs.status(f"ARQ CONNECTION REJECTED BY {self._remote}, INCOMPATIBLE BW")
                self._teardown(notify=False)
            return

        if self._sub == _Sub.ISS_CONACK:
            if ok and _is_dataack(ft):
                self._sub = _Sub.ISS_DATA
                self._tx_even = True
                self._progress()
                self._obs.connected(self._remote, self._session_bw)
                self._obs.status(f"ARQ CONNECTION ESTABLISHED WITH {self._remote}, "
                                 f"SESSION BW = {self._session_bw} HZ")
                if self._outbound:
                    self._send_data()
            return

        # ISS_DATA
        if ok and _is_dataack(ft):
            self._advance_on_ack(ft)
        elif ok and _is_datanak(ft):
            self._average_quality(38 + 2 * (ft - DATANAK_MIN))
            self._shift_down()
            # `ARQ.c`'s ISSData DataNAK branch sets `intACKctr = 0` outside the
            # `if (intShiftUpDn != 0)` beside it and after `Gearshift_9`, so a NAK
            # costs the rung's earned ACKs whether or not it shifted, and the shift
            # still sees them; `ArdopGearshift.RecordNak` does the same. It is the
            # reference's rule, not a strictness of ours — see
            # `test_a_nak_zeroes_the_acks_earned_at_the_rung_as_the_reference_does`.
            self._acks_at_rung = 0
            self._retransmit()
        elif ok and ft == BREAK:
            self._concede_to_break()
        elif ok and ft == DISC:
            self._tx.send(END, b"", self._session)
            self._teardown(notify=True)
        elif ok and ft == END:
            self._teardown(notify=True)

    # IRS: callee through the handshake, then receiving data.
    def _rx_irs(self, ft: int, payload: bytes, ok: bool) -> None:
        if self._sub == _Sub.IRS_CONACK:
            if not ok:
                return
            if _is_conreq(ft):
                # The caller missed our first ConAck — resend it.
                caller = callsign.unpack_callsign(payload[0:6])
                target = callsign.unpack_callsign(payload[6:12])
                if target == self.mycall:
                    self._answer_conreq(ft, caller, target)
            elif _is_conack(ft):
                # The confirming ConAck — session is up; first ACK completes it.
                self._session_bw = _CONACK_BW[ft]
                self._sub = _Sub.IRS_DATA
                self._progress()
                self._obs.connected(self._remote, self._session_bw)
                self._obs.status(f"ARQ CONNECTION FROM {self._remote}: "
                                 f"SESSION BW = {self._session_bw} HZ")
                self._dataack(self._rx_quality)
            return

        # IRS_DATA / IRS_FROM_ISS. The peer holds the link, so any transmission of
        # its could be a data frame this receiver failed to read as one, and the
        # bare controls that would spend it are acted on only once a repeat has
        # corroborated them (`_corroborated`) — never on a single report.
        if ok and _is_conack(ft):
            # The caller missed our ACK — re-ACK.
            if self._corroborated(ft):
                self._dataack(self._rx_quality)
                self._progress()
        elif ok and ft == DISC:
            if self._corroborated(ft):
                self._tx.send(END, b"", self._session)
                self._teardown(notify=True)
        elif ok and ft == END:
            self._teardown(notify=True)
        elif ok and ft == BREAK:
            if self._corroborated(ft):
                self._concede_again()
        elif ok and ft == IDLE:
            confirmed_idle = self._corroborated(ft)
            if confirmed_idle:
                self._last_rx_payload = None
            # An idle ISS is one that has said its piece. An IRS holding bytes
            # queued mid-handover — the instant between conceding and the new
            # ISS's first frame, when `queue_data` cannot kick — seizes the
            # link now; one with nothing to say acknowledges the keepalive.
            if self._outbound:
                self._begin_break()
            else:
                # Answering the chirp is not progress: the peer has said its piece
                # and we have nothing to add, which is the shape of an exchange
                # that is over rather than one that is alive. Deferring only our
                # own chirp lets the session deadline run, so the link ends on the
                # timeout it already has instead of on a signal from the operator.
                if confirmed_idle:
                    self._dataack(_QUALITY_NOT_DATA)
                    self._defer_chirp()
        elif _is_data(ft):
            self._receive_data(ft, payload, ok)

    # IDLE: connected, ISS neither sending; hearing an ACK keeps idling.
    def _rx_idle(self, ft: int, payload: bytes, ok: bool) -> None:
        if not ok:
            return
        # Any valid inbound frame is proof the link is live — reset the idle
        # timeout, so IDLE chirps a peer keeps ACKing never trip the drop.
        self._progress()
        if _is_dataack(ft):
            if self._outbound:
                self._set_state(P.ArdopState.ISS)
                self._sub = _Sub.ISS_DATA
                self._send_data()
            return
        if ft == BREAK:
            self._concessions = 0
            self._dataack(_QUALITY_NOT_DATA)
            self._set_state(P.ArdopState.IRS)
            self._sub = _Sub.IRS_FROM_ISS
            self._reset_stint()
            self._progress()
        elif ft == DISC:
            self._tx.send(END, b"", self._session)
            self._teardown(notify=True)
        elif ft == END:
            self._teardown(notify=True)

    # IRStoISS: repeating BREAK; any ACK completes the seizure.
    def _rx_irs_to_iss(self, ft: int, payload: bytes, ok: bool) -> None:
        if ok and _is_dataack(ft):
            self._set_state(P.ArdopState.ISS)
            self._sub = _Sub.ISS_DATA
            self._tx_even = True
            self._last_data_type = -1
            self._progress()
            self._send_data()
        elif ok and (_is_data(ft) or ft == IDLE):
            # The old ISS hasn't heard our BREAK — a data frame and an idle
            # chirp say so equally. Repeat it. IDLE matters exactly when the
            # IRS speaks first (a mail answerer owns the greeting): an ISS
            # with an empty queue sends nothing else, so without this the
            # seizure of a quiet link could never complete.
            dur = self._tx.send(BREAK, b"", self._session)
            self._next_break_repeat = self._now + dur + self._data_interval
        elif ok and ft == DISC:
            # The peer is leaving (mail done, gateway closing) — a BREAK
            # repeated at a station that is gone holds the seizure open
            # forever. Answer the disconnect and let the queue go with it.
            self._tx.send(END, b"", self._session)
            self._teardown(notify=True)
        elif ok and ft == END:
            self._teardown(notify=True)

    # -- data engine -------------------------------------------------------

    def _data_type(self) -> int:
        base = _DATA_LADDER[self._session_bw][self._rung][0]
        return base if self._tx_even else base + 1

    def _send_data(self) -> None:
        if not self._outbound:
            return
        ft = self._data_type()
        cap = F.FRAMES[ft].net_payload
        self._in_process = min(cap, len(self._outbound))
        self._last_data_type = ft
        self._frame_was_repeated = False
        self._progress()
        dur = self._tx.send(ft, bytes(self._outbound[:self._in_process]), self._session)
        self._next_data_repeat = self._now + dur + self._data_interval

    def _retransmit(self) -> None:
        """Put the outstanding chunk back on the air at the mode now in force. A
        shift down since it went out means a smaller frame, so the chunk is re-cut
        from the queue head at the new cap — safe because bytes leave the queue
        only on an ACK, and clamped so the cut never *grows*, which would delete
        more on the next ACK than the peer was ever sent."""
        if self._last_data_type < 0:
            return
        ft = self._data_type()
        self._in_process = min(F.FRAMES[ft].net_payload, self._in_process)
        self._last_data_type = ft
        self._frame_was_repeated = True
        self._progress()
        dur = self._tx.send(ft, bytes(self._outbound[:self._in_process]), self._session)
        self._next_data_repeat = self._now + dur + self._data_interval

    # -- transmit rate selection -------------------------------------------

    def _reset_rate(self) -> None:
        self._rung = 0
        self._avg_quality = 0
        self._acks_at_rung = 0
        self._naks_at_rung = 0
        self._rungs_tried.clear()
        self._rungs_worked.clear()

    def _average_quality(self, reported: int) -> None:
        self._avg_quality = reported if self._avg_quality == 0 else int(
            self._avg_quality * (1 - _QUALITY_ALPHA) + _QUALITY_ALPHA * reported + 0.5)

    def _shift_to(self, rung: int) -> None:
        log.info("rate %s -> %s (peer quality avg %d over %d ACKs)",
                 _mode_name(_DATA_LADDER[self._session_bw][self._rung][0]),
                 _mode_name(_DATA_LADDER[self._session_bw][rung][0]),
                 self._avg_quality, self._acks_at_rung)
        self._rung = rung
        self._avg_quality = 0                # the next grade becomes the new average
        self._acks_at_rung = 0
        self._naks_at_rung = 0

    def _shift_up(self) -> None:
        """Every arm of this says so. The 2026-08-26 W6IDS fetch climbed on one
        session and not the next, and the record could not separate a brake from
        a grade from a shift never attempted — the four ways to hold a rung look
        identical from outside, and only one of them is about the path."""
        ladder = _DATA_LADDER[self._session_bw]
        threshold = ladder[self._rung][1]
        nxt = self._rung + 1
        tail = len(self._outbound)
        one_frame = F.FRAMES[self._data_type()].net_payload
        # Both of the reference's brakes, kept because the gateways are tuned
        # against that ladder and a peer met with a different one behaves oddly:
        # nothing is won by climbing for a tail that already fits in one frame
        # here, and a rung that failed the moment we reached it is not retried on
        # the same two ACKs that took us there the first time.
        if threshold is None:
            held = "no rung above it"
        elif self._acks_at_rung < _ACKS_TO_SHIFT_UP:
            held = f"{_ACKS_TO_SHIFT_UP} ACKs needed to leave a rung"
        elif self._avg_quality <= threshold:
            held = f"the bar is {threshold}"
        elif tail <= one_frame:
            held = f"the {tail} B tail already fits one {one_frame} B frame"
        elif nxt in self._rungs_tried and nxt not in self._rungs_worked \
                and self._acks_at_rung < _ACKS_TO_RETRY_RUNG:
            held = (f"{_mode_name(ladder[nxt][0])} failed when reached, "
                    f"{_ACKS_TO_RETRY_RUNG} ACKs needed to retry it")
        else:
            self._rungs_tried.add(nxt)
            self._shift_to(nxt)
            return
        log.info("rate holds %s (peer quality avg %d over %d ACKs): %s",
                 _mode_name(ladder[self._rung][0]), self._avg_quality,
                 self._acks_at_rung, held)

    def _shift_down(self) -> None:
        """`Gearshift_9`'s `DownNAKS`, and it says so at every arm the way the climb
        does: two NAKs leave a rung that has ACKed something, one leaves a rung that
        has not. An isolated bad frame on a rung the path was carrying is a fade,
        not a verdict on the mode."""
        ladder = _DATA_LADDER[self._session_bw]
        self._naks_at_rung += 1
        needed = _NAKS_TO_SHIFT_DOWN if self._rung in self._rungs_worked else 1
        if not self._rung:
            held = "no rung below it"
        elif self._naks_at_rung < needed:
            held = f"{needed} NAKs needed to leave a rung that has ACKed"
        else:
            self._shift_to(self._rung - 1)
            return
        log.info("rate holds %s (peer quality avg %d over %d NAKs): %s",
                 _mode_name(ladder[self._rung][0]), self._avg_quality,
                 self._naks_at_rung, held)

    def _advance_on_ack(self, ft: int) -> None:
        if self._last_data_type < 0:
            return
        # Half-duplex stale-ACK guard: once we have advanced past a *repeated*
        # frame and launched the next one, an ACK arriving within a turnaround
        # cannot be for that new frame — the peer has not heard it yet — so it is
        # a duplicate of the frame we just cleared. Ignoring it stops a stale ACK
        # from deleting an unacked frame. Un-repeated frames ACK exactly once, so
        # the guard never touches the clean path (it stays disarmed there).
        if self._now < self._ack_guard_until:
            return
        repeated = self._frame_was_repeated
        del self._outbound[:self._in_process]
        self._in_process = 0
        self._last_data_type = -1
        self._frame_was_repeated = False
        self._tx_even = not self._tx_even
        self._progress()
        self._obs.buffer(len(self._outbound))
        self._acks_at_rung += 1
        self._naks_at_rung = 0
        self._rungs_worked.add(self._rung)
        self._average_quality(38 + 2 * (ft - DATAACK_MIN))
        self._shift_up()
        self._ack_guard_until = self._now + self._turnaround_s if repeated else 0.0
        if self._outbound:
            self._send_data()

    def _reset_stint(self) -> None:
        # Type alternation only separates frames within one stint as IRS. What
        # a station sends first on taking the link is its own business — WW2MI
        # answered an ACKed even frame with another even frame across the
        # turnover — so a type held from the previous stint is no evidence of a
        # repeat. Both halves clear together: graded against a stale
        # `_last_acked_type`, an undecodable frame draws the ACK for one
        # already delivered and the ISS moves on with the payload still aboard.
        # The NAK budget clears here for the same reason it is graded against
        # `_stint_delivered`: both are about what THIS stint has got through.
        self._last_rx_type = -1
        self._last_acked_type = -1
        self._heard_once = -1
        self._stint_delivered = False
        self._unrepaired = 0
        self._unrepaired_modes.clear()

    def _is_replay(self, payload: bytes) -> bool:
        """Match the first frame after a reversal against the unsettled payload."""
        return self._last_rx_type < 0 and bytes(payload) == self._last_rx_payload

    def _unreadable(self, frame_type: int, quality: int | None = None) -> None:
        """A frame arrived and this end could not read it. That is neither an empty
        channel nor a repeat, and the NAK is what says so on the air. What it asks
        for is a shift DOWN, not a repeat — the peer's own repeat timer is what
        puts the frame back — so the recovery here is the mode getting more
        robust, and the frame coming round again is the metronome rather than an
        answer.

        A rung this stint has not failed on before is the peer having answered the
        last NAK, so the budget starts again there: the descent costs the reference
        two NAKs a rung and there are up to six rungs beneath the one a session
        opens on. Which budget renews depends on whether the stint has delivered —
        the two are independent, and a peer that never descends, as W6IDS did not,
        gets exactly one of them. Only a data frame names a rung — a header-only
        control is a guess at a type and not a mode the peer chose — and the
        even/odd twin is the block counter rather than the mode, so both are
        stripped before the comparison.

        ``quality`` is for a reading this session did not take as its own."""
        if _is_data(frame_type) and (frame_type | 1) not in self._unrepaired_modes:
            self._unrepaired_modes.add(frame_type | 1)
            self._unrepaired = 0
        self._unrepaired += 1
        # `_last_rx_type` is the dedup record and answers a different question:
        # a replay across a reversal sets it and hands the application nothing,
        # so reading it as delivery gave a stint that had delivered nothing the
        # shorter budget and then never let it reach the close at all.
        delivered = self._stint_delivered
        budget = _UNREPAIRED_BUDGET if delivered else _UNDELIVERED_BUDGET
        if self._unrepaired > budget:
            head = (f"{self._unrepaired} {_mode_name(frame_type)} FRAMES FROM "
                    f"{self._remote} UNREADABLE AFTER {budget} NAKS")
            if not delivered:
                self._obs.status(f"{head} AND NOTHING DELIVERED THIS STINT, "
                                 f"DISCONNECTING")
                self.disconnect()
                return
            if self._unrepaired == budget + 1:
                self._obs.status(f"{head}, BUT THIS STINT HAS DELIVERED, "
                                 f"HOLDING THE LINK")
        self._tx.send(_datanak_for(self._rx_quality if quality is None else quality),
                      b"", self._session)

    def _dataack(self, quality: int) -> None:
        """Acknowledge — which on this link means one thing and cannot be made to
        mean less: the frame the ISS is holding may be let go of.

        A DATAACK carries no sequence number and no reference to what it answers,
        so the ISS applies it to whatever it has outstanding whatever this end
        meant by it. Every caller therefore has to satisfy one of two conditions:
        the payload has reached the application (`_receive_data`, and the repeat
        rule that stands in for it), or the peer provably holds no data frame at
        all — through the connect handshake, and while it is breaking for a link it
        has not been given yet.

        Twice this station has sent one that satisfied neither, and both times the
        message opened on its 65th byte: WW2MI 2026-08-18, `0xb1`, a repeat rule
        graded against a type held across a turnover; `WWTD6QMC61TV` 2026-08-23,
        `0xfd`, a second concession answering a data frame the receiver had minted
        a BREAK out of. The routes were different and the defect was one.
        """
        self._tx.send(_dataack_for(quality), b"", self._session)

    def _corroborated(self, ft: int) -> bool:
        """A control frame from the station that holds the link, answered only once
        a second copy of it has arrived.

        As IRS every transmission the peer makes could be a data frame, and this
        receiver mints bodyless controls out of one when the half-duplex mute
        splice lands on its leader: on 2026-08-23 the 4.4 s `4PSK.200.100` carrying
        the head of `WWTD6QMC61TV` was reported as a BREAK, and the concession that
        drew was read by the CMS as the ACK of that frame. A real BREAK, IDLE or
        ConAck runs on a repeat timer of its own, so waiting for the second copy
        costs one repeat interval; a minted one arrives once, beside the frame it
        came from, and is never answered at all.

        DISC is the worst of them and was the last one left ungated: a minted one
        tears the link down and loses the whole queue, where a minted BREAK costs
        one frame. It is on the rule for the same reason and at the same price —
        the reference repeats DISC until an END comes back or the count runs out
        (`ARQ.c` `blnDISCRepeating`, and four copies 2.05 s apart in this station's
        own 2026-08-23 log). END is not, and cannot be: the reference sends it once
        at every site, so there is no second copy to wait for, and a station that
        ignored the only one would hold a link its peer had already left.

        `SoundInput.c:2305` is the reference drawing the same line — DataACK/NAK,
        BREAK, END and DISC are its "critical" frames, held to a tighter decode
        distance than every other type because those are the ones that damage a
        session. This is that list, answered here by corroboration rather than by
        a threshold.
        """
        if self._heard_once == ft:
            return True
        self._heard_once = ft
        return False

    def _receive_data(self, ft: int, payload: bytes, ok: bool) -> None:
        if not ok:
            if ft == self._last_acked_type:
                self._dataack(self._rx_quality)              # already delivered
                return
            self._unreadable(ft)
            return
        # The frame is acknowledged BEFORE its payload reaches the application,
        # and the handshake-tail sub-state is retired before either. Both
        # matter to an application that answers from inside `data_received`,
        # which the mail client does: `queue_data`'s break-in kick reads the
        # sub-state, so an answer queued against IRS_FROM_ISS sat in the buffer
        # while this end ACKed idle chirps forever; and a break-in provoked by
        # the delivery used to overtake the DATAACK, so the deposed ISS unwound
        # a frame this end had already delivered and re-sent it — a duplicate
        # the type-alternation dedupe cannot catch across a role reversal.
        if self._sub == _Sub.IRS_FROM_ISS:
            self._sub = _Sub.IRS_DATA
        self._concessions = 0                # the peer took the link: the seizure completed
        self._unrepaired = 0                 # this stint is reading this peer again
        self._unrepaired_modes.clear()
        self._last_acked_type = ft
        self._progress()
        self._dataack(self._rx_quality)
        if ft != self._last_rx_type:
            replay = self._is_replay(payload)
            self._last_rx_type = ft
            if not replay:
                self._last_rx_payload = bytes(payload)
                self._stint_delivered = True
                self._obs.data_received("ARQ", bytes(payload))

    # -- turnover ----------------------------------------------------------

    def _begin_break(self) -> None:
        self._set_state(P.ArdopState.IRStoISS)
        self._obs.status("QUEUE BREAK new Protocol State IRStoISS")
        self._progress()
        dur = self._tx.send(BREAK, b"", self._session)
        self._next_break_repeat = self._now + dur + self._data_interval

    def _concede_to_break(self) -> None:
        # Deposed ISS: ACK and hand the link over, but keep `_outbound` — the
        # unsent/unacked bytes replay once we regain ISS (ardopcf SaveQueueOnBreak).
        # The outstanding frame is unwound so it resends from the queue head.
        self._last_data_type = -1
        self._in_process = 0
        self._frame_was_repeated = False
        self._concessions = 0
        self._obs.status("BREAK received from Protocol State ISS, new state IRS")
        self._dataack(_QUALITY_NOT_DATA)
        self._set_state(P.ArdopState.IRS)
        self._sub = _Sub.IRS_FROM_ISS
        self._reset_stint()
        self._progress()

    def _concede_again(self) -> None:
        """A corroborated repeated BREAK from a peer we already yielded to: it has
        not acted on our conceding ACK. Concede again — the ACK is what the seizure
        waits on, and it is the only frame that completes one (ardopcf leaves
        IRStoISS on a DataACK and on nothing else; a data frame it answers with
        another BREAK, and `ProcessRcvdARQFrame`'s IRStoISS case does nothing at
        all). But a budget: past `_CONCESSION_BUDGET` the peer is not going to act
        on this answer, and re-sending it is a station transmitting into a link it
        cannot finish."""
        # Already leaving: a further BREAK must not re-enter `disconnect`, which
        # would restart the DISC retry clock on every one and hang the teardown
        # exactly the way the concession itself hung.
        if self._disc_repeating:
            return
        self._concessions += 1
        if self._concessions > _CONCESSION_BUDGET:
            self._obs.status(f"BREAK FROM {self._remote} NOT COMPLETED AFTER "
                             f"{_CONCESSION_BUDGET} ACKS, DISCONNECTING")
            self.disconnect()
            return
        self._dataack(_QUALITY_NOT_DATA)
        self._progress()

    # -- timers ------------------------------------------------------------

    def _tick_connect(self) -> None:
        if self._now < self._next_conreq:
            return
        if self._listening_out:
            self._tick_listen_out()
            return
        if self._conreq_repeats >= self._conreq_budget or self._now >= self._connect_deadline:
            self._listen_out()
            return
        self._conreq_repeats += 1
        payload = callsign.pack_callsign(self.mycall) + callsign.pack_callsign(self._remote)
        dur = self._tx.send(self._conreq_type, payload, self._session)
        self._next_conreq = self._now + dur + self._connect_interval

    def _tick_conack(self) -> None:
        # Repeat the ConAck leg of the handshake (ardopcf repeats ConAck in both
        # IRSConAck and ISSConAck) so a lost confirming ConAck or completing
        # DATAACK no longer wedges the connect for the whole session timeout.
        if self._now < self._next_conack:
            return
        if self._listening_out:
            self._tick_listen_out()
            return
        if self._conack_repeats >= self._conack_budget:
            self._listen_out()
            return
        self._conack_repeats += 1
        dur = self._tx.send(CONACK[self._session_bw], bytes([_LEADER_TENS]) * 3, self._session)
        self._next_conack = self._now + dur + self._connect_interval

    def _listen_out(self) -> None:
        """The calling is over; the listening is not.

        Both handshake legs end here rather than in a teardown, because the state
        that keeps this end hearing *is* the connect state: `expected_session` is
        what lets the demodulator corroborate a bare ConAck out of the noise, and a
        teardown retires it, drops the host's link and stops the capture with it.
        Nothing is keyed from here.
        """
        self._listening_out = True
        self._connect_deadline = self._now + _CONNECT_TAIL_S
        self._obs.status(f"CALLING {self._remote} ENDED, LISTENING "
                         f"{_CONNECT_TAIL_S:.0f} S FOR A LATE ANSWER")

    def _tick_listen_out(self) -> None:
        if self._now >= self._connect_deadline:
            self._obs.status(f"CONNECT TO {self._remote} FAILED!")
            self._teardown(notify=False)

    def _arm_conack(self, sent_dur: float) -> None:
        # A ConAck answered is the call alive again: the tail belongs to the leg
        # that has run out of repeats, not to the whole attempt.
        self._listening_out = False
        self._conack_repeats = 0
        self._conack_budget = 10
        self._next_conack = self._now + sent_dur + self._connect_interval

    def _tick_disc(self) -> None:
        if self._now - self._last_progress < self._connect_interval:
            return
        self._disc_repeats += 1
        if self._disc_repeats > 5:
            self._obs.status(f"END NOT RECEIVED CLOSING ARQ SESSION WITH {self._remote}")
            self._teardown(notify=True)
            return
        self._progress()
        self._tx.send(DISC, b"", self._session)

    def _progress(self) -> None:
        """Payload moved, or the link's state advanced.

        The only thing entitled to postpone the session deadline, and the reason
        the stamp is named for progress rather than for traffic: an ARQ layer
        cannot know whether the exchange above it is finished, so the one signal
        it can trust is that the exchange is still getting somewhere. Answering a
        peer that is only chirping is traffic — ours — and reading it as liveness
        is what let 25 cycles of RX IDLE draw 25 DATAACKs over 71 s after the
        gateway had refused our login, ending on the operator's SIGTERM.
        """
        self._last_progress = self._now
        self._defer_chirp()

    def _defer_chirp(self) -> None:
        """Something went out, so our own keepalive is not due yet. Says nothing
        about whether the link is getting anywhere."""
        self._next_chirp = self._now + 2.0

    def _teardown(self, *, notify: bool) -> None:
        self._disc_repeating = False
        self._sub = _Sub.NONE
        self._outbound.clear()
        self._in_process = 0
        self._last_data_type = -1
        self._reset_stint()
        self._last_rx_payload = None         # a new link is a new stream
        self._concessions = 0
        if notify and self._state in P.CONNECTED_STATES:
            self._obs.disconnected()
        self._set_state(P.ArdopState.DISC)
