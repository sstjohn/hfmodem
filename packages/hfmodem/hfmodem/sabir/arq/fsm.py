# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The sabir link layer: a HARQ FSM with carrier-group selective ACK.

Pure logic behind an injected sink: events in (host commands, decoded control
blocks, per-codeword body LLRs, timer ticks), transmissions out through the
injected :class:`ArqIO`. The FSM owns the framing codecs (deterministic
compute) but renders no waveform and reads no clock beyond the injected
one -- the seam the M4 host API and a real audio channel drop into.

The discipline:

- **Stop-and-wait per frame, selective repeat within it.** A frame is up to
  64 codewords; the ACK carries a cumulative per-codeword bitmap and only
  the failed codewords are retransmitted. Grouped gears confine each
  codeword to one carrier group (`arq.layout`), so a notched group costs
  only its own codewords.
- **HARQ (Chase).** The receiver keeps a running per-codeword LLR sum and
  folds every retransmission in before decoding, so independent equal-quality observations can gain 3 dB per doubling.
- **Gearshift** (TX-driven, announced in the DATA header's gear field): up
  one rung after ``up_after`` consecutive single-round frames *and* the
  ACK-reported SNR clearing the next rung's entry threshold; down one rung
  -- rebuilding the frame's payload at the new gear under a fresh seq -- on
  a stalled round (no newly-acked codewords), round exhaustion, or repeated
  ACK timeouts; and softly (at the next frame boundary) when the reported
  SNR sags below the current rung's threshold.
- **Loading** (TX, from the ACK's per-group SNR): groups measured below
  ``dead_db`` per-carrier SNR are transmit-masked; on grouped gears each
  live group independently picks QPSK/16-QAM/64-QAM by SNR threshold. A
  masked group is no longer measured, so it stays masked for the session
  (re-probing is future work). Pilots are transmitted on masked carriers
  regardless, so the channel stays sounded for the estimator.
- **Adaptive control tier.** Control blocks ride the fast coherent
  OFDM burst when the measured link clears ``fast_ctrl_on_db``, and fall
  back to the noncoherent floor burst -- the safety net that keeps
  the link from dropping -- when marginal: at connect (SNR unknown), below
  ``fast_ctrl_off_db`` (hysteretic), or on any control loss (ACK/turn
  timeout, a duplicate DATA round showing our ACK died). The IRS never
  replies faster than the header it just heard, so one lost fast ACK costs
  exactly one conservative floor round before the tier re-earns itself.
  The receive side always tries both tiers, so tier choice can never
  break a decode, only its speed.
- **Turn exchange.** One end holds the send role (ISS) at a time; the
  connect initiator starts with it. The role holder hands over (TURN)
  when its queue drains and the peer has flagged traffic -- in every ACK's
  ``gear`` bit, or by an explicit TURN_REQ from a quiescent link. A
  pending host disconnect counts as traffic, so the mic always comes back
  for the goodbye. On the fast tier the TURN round trip disappears: a
  draining ISS offers handover in its DATA header (``wire.HANDOVER``),
  and a completing IRS with traffic answers with its own DATA over, the
  final ACK piggybacked ahead of the header -- no separate control over
  at all on a healthy two-way link.
- **Station ID (§97.119).** Every callsign-bearing control block (connect,
  disconnect, the dedicated ID block) restamps the clock; either side
  transmitting with ``id_interval_s`` elapsed emits an ID block first, and
  DISC/DISC_ACK carry the callsign so a session always ends identified.
"""

from __future__ import annotations

import secrets
import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.phy.modem import GEARS, SPACING

from . import wire


class SessionState(StrEnum):
    """The five link-lifecycle states, shared vocabulary across the flock.

    ``StrEnum`` members are ``str`` subclasses, so every existing comparison,
    f-string and CBOR encode behaves exactly as it did with bare constants --
    ``encode(SessionState.CONNECTED)`` is byte-identical to
    ``encode("CONNECTED")`` -- while the set becomes closed and introspectable.
    """

    DISCONNECTED = "DISCONNECTED"
    LISTENING = "LISTENING"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTING = "DISCONNECTING"


@dataclass(frozen=True)
class RungCfg:
    name: str            # phy.GEARS key
    frame_cws: int       # codewords per frame at this rung
    grouped: bool        # carrier-group-localized codewords (selective unit)
    up_snr_db: float     # SNR (3 kHz) needed to shift up into this rung


# Entry thresholds sit near each rung's M2 FER~0.1 point; the bottom rung is
# unconditional. Frame sizes keep bodies in the ~5-9 s range per rung.
DATA_LADDER = (
    RungCfg("robust", 4, False, -99.0),
    RungCfg("workhorse", 12, False, 3.0),
    RungCfg("workhorse34", 16, False, 9.0),
    RungCfg("fast", 48, True, 15.0),
    RungCfg("max", 60, True, 21.0),
)
_RUNG_OF_GEAR = {wire.GEAR_ID[r.name]: i for i, r in enumerate(DATA_LADDER)}


# why a session ended, surfaced to the host as StateChanged.reason (HOST-API §5;
# values equal the wire RS_* codes there).
DISC_REMOTE = 1        # peer sent DISC
DISC_LOCAL = 2         # host disconnect or abort
DISC_REFUSED = 4       # unsupported critical capability extension
DISC_LINK_FAILED = 3   # rebuild limit, no progress, or connect never established


@dataclass
class ArqConfig:
    callsign: str = "N0CALL"
    turnaround_s: float = 0.25
    control_s: float = 8.72           # one floor control burst (endpoint sets)
    ack_margin_s: float = 1.5
    connect_timeout_s: float = 15.0
    max_connect_retries: int = 4
    max_rounds: int = 4              # transmissions per frame before rebuild
    max_rebuilds: int = 8            # consecutive rebuilds before abort
    up_after: int = 2                # clean frames before an up-shift
    probe_margin_db: float = 5.0     # SNR slack for the success-driven probe
    down_hysteresis_db: float = 2.0
    start_rung: int = 1              # workhorse
    dead_db: float = 0.0             # per-carrier SNR below which a group dies
    t16_db: float = 12.0             # per-carrier SNR for 16-QAM loading
    t64_db: float = 21.0             # per-carrier SNR for 64-QAM loading
    selective: bool = True           # False: retransmit whole frames
    harq: bool = True                # False: no soft-combining (blind repeat)
    loading: bool = True             # False: uniform loading always
    deflate: bool = True             # advertise DEFLATE: we accept 0x11 records
    fast_ctrl: bool = True           # False: every control on the floor burst
    fast_ctrl_on_db: float = 6.0     # SNR (3 kHz) to enter the fast tier
    fast_ctrl_off_db: float = 3.0    # below this, back to the floor burst
    max_rung: int = len(DATA_LADDER) - 1   # host bandwidth cap on the ladder
    id_interval_s: float = 540.0     # §97.119 cadence, with a frame in hand
    inactivity_timeout_s: float = 180.0  # receiver-local silent-peer expiry
    max_turn_reqs: int = 6
    advertise_gears: tuple = ()      # additional receive-profile names
    data_profile: str | None = None  # preferred DATA profile, sender-gated
    bandwidth_hz: int = 2750         # actual waveform limit, including extensions
    feedback_iters: int = 0          # receiver-local decoder-aided refinement
    impulse_blank: float = 0.0       # receiver-local threshold; 0 disables


class ArqIO:
    """Sink the FSM drives; both senders return the transmission duration.

    The FSM publishes the control tier as ``fsm.ctrl_fast`` and a
    piggybacked ACK (to ride ahead of the next DATA header) as
    ``fsm.pb_ack``; a waveform-rendering IO reads both at send time."""

    def send_control(self, ctrl: wire.Control) -> float: ...
    def send_data(self, ctrl_seq: int, gear: int, present: list[int],
                  n_cw: int, nibbles: tuple, coded: np.ndarray, offset: int) -> float: ...
    def connected(self, session: int, peer: str) -> None: ...
    def disconnected(self) -> None: ...
    def deliver(self, blob: bytes) -> None: ...
    def log(self, msg: str) -> None: ...
    def state_changed(self, state: str) -> None: ...


@dataclass
class _TxFrame:
    seq: int
    rung: int
    chunks: list
    coded: np.ndarray
    profile: str | None = None
    offset: int = 0
    acked: set = field(default_factory=set)
    tx_count: int = 0
    retx_cws: int = 0
    timeouts: int = 0

    @property
    def clean(self) -> bool:
        """First-shot delivery, or a single small salvage round (<= 1/4 of
        the codewords): the selective mechanism working, not a failure."""
        return self.tx_count == 1 or (self.tx_count == 2
                                      and self.retx_cws * 4 <= len(self.chunks))


@dataclass
class _RxFrame:
    seq: int
    gear: int
    n_cw: int
    store: np.ndarray
    chunks: list
    offset: int = 0


class ArqFsm:
    def __init__(self, io: ArqIO, cfg: ArqConfig | None = None, clock=None,
                 session_id_factory=None):
        import time
        self.io = io
        self.cfg = cfg or ArqConfig()
        if (not 0 <= self.cfg.max_rung < len(DATA_LADDER)
                or not 0 <= self.cfg.start_rung < len(DATA_LADDER)
                or self.cfg.bandwidth_hz not in (500, 1500, 2300, 2750)
                or not 0 <= self.cfg.feedback_iters <= 3
                or not 0 <= self.cfg.impulse_blank <= 20
                or not math.isfinite(self.cfg.inactivity_timeout_s)
                or self.cfg.inactivity_timeout_s <= 0):
            raise ValueError("invalid DATA/receiver configuration")
        from .profiles import EXTENDED, PROFILES
        if (len(set(self.cfg.advertise_gears)) != len(self.cfg.advertise_gears)
                or any(g not in EXTENDED for g in self.cfg.advertise_gears)):
            raise ValueError("advertise_gears must name distinct implemented profiles")
        if self.cfg.data_profile is not None and self.cfg.data_profile not in PROFILES:
            raise ValueError("unknown DATA profile")
        self._profile_failed = False
        self._floor_fallback: str | None = None
        self._clock = clock or time.monotonic
        self._session_id_factory = session_id_factory or (lambda: secrets.randbits(64) or 1)
        self.state = SessionState.DISCONNECTED
        self.session = 0
        self.peer = ""
        self.rung = min(self.cfg.start_rung, self.cfg.max_rung)
        self.ctrl_fast = False           # current control tier (IO reads it)
        self.bootstrap_fast = False      # CONNECT transport, independent of data-tier policy
        self._fast_probe_done = False
        self.pb_ack = None               # ACK to piggyback on the next DATA
        self._listen = False
        self._codecs: dict[str, FrameCodec] = {}

        # Exact peer receive permissions; no feature is usable before the reply.
        self.peer_capabilities = 0
        self.peer_profiles: list[int] = []
        self.use_fastctl = False
        self.use_pback = False
        self.use_loading = False
        self.use_deflate = False
        self._usable_rungs = list(range(self.cfg.max_rung + 1))
        self._eff_max_rung: int | None = self.cfg.max_rung
        self._accepted_offer = None
        self.begin_burst()
        self.disc_reason = DISC_LOCAL                  # why the last session ended

        # TX (whichever end currently holds the send role)
        self._queue = bytearray()
        self._frame: _TxFrame | None = None
        self._next_seq = 1
        self._streak = 0
        self._rebuilds = 0
        self._nibbles = [0] * 8
        self._gsnr: dict[int, float] = {}
        self._link_snr3k: float | None = None
        self._disc_pending = False
        self._connect_retries = 0
        self._iss = False
        self._peer_pending = False
        self._turn_reqs = 0
        self._last_id = 0.0

        # RX
        self._rx: _RxFrame | None = None
        self._done_seq: int | None = None
        self._last_session = 0

        # single timer: (deadline, kind)
        self._deadline: float | None = None
        self._kind = ""
        self._idle_deadline: float | None = None
        self._tx_busy_until = 0.0
        self._peer_generation = 0

        self.stats = {"frames": [], "sends": [], "shifts": [], "ids": [],
                      "ctrl": [], "rounds": 0, "timeouts": 0, "rebuilds": 0,
                      "pb_acks": 0}

    # -- helpers -----------------------------------------------------------
    def _to(self, state: str) -> None:
        if state != self.state:
            self.io.log(f"state {self.state} -> {state}")
            self.state = state
            self.io.state_changed(state)

    def _arm(self, kind: str, dt: float) -> None:
        self._deadline = self._clock() + dt
        self._kind = kind

    def next_deadline(self) -> float | None:
        deadlines = [self._deadline]
        if self._idle_deadline is not None:
            deadlines.append(max(self._idle_deadline, self._tx_busy_until))
        return min((d for d in deadlines if d is not None), default=None)

    def _heard_peer(self, body_duration_s: float = 0.0) -> None:
        """Refresh only after accepting session traffic; protect an announced body."""
        self._idle_deadline = self._clock() + max(
            self.cfg.inactivity_timeout_s,
            body_duration_s + self.cfg.control_s + 2 * self.cfg.turnaround_s + self.cfg.ack_margin_s)

    def on_data_header(self, c: wire.Control, body_duration_s: float) -> bool:
        """Lease for a structurally validated DATA header, including long legal bodies.

        The waveform front-end must derive duration from validated codeword geometry,
        never merely trust the received symbol count. Whole-burst decoders call this
        before body demodulation; a streaming frontend may call it at header decode.
        """
        _, n_cw, _ = wire.parse_data_aux(c.aux)
        if (self.state != SessionState.CONNECTED or c.session != self.session
                or self.rx_profile(c.gear) is None or not 0 < n_cw <= 64
                or c.offset > self._rx_offset or c.seq < self._peer_generation
                or c.seq == 0 or not math.isfinite(body_duration_s) or body_duration_s < 0):
            return False
        self._peer_generation = max(self._peer_generation, c.seq)
        self._heard_peer(body_duration_s)
        return True

    def _codec(self, rung_idx: int) -> FrameCodec:
        name = DATA_LADDER[rung_idx].name
        return self.profile_codec(name)

    def profile_codec(self, name):
        from .profiles import codec_for
        if name not in self._codecs:
            self._codecs[name] = codec_for(name)
        return self._codecs[name]

    def local_profiles(self):
        from .profiles import PROFILES
        names = ([r.name for r in DATA_LADDER[:self.cfg.max_rung + 1]]
                 + list(self.cfg.advertise_gears))
        return sorted(PROFILES[g].id for g in names
                      if PROFILES[g].bandwidth <= self.cfg.bandwidth_hz)

    def rx_profile(self, gear):
        from .profiles import BY_ID
        gid = gear & wire.GEAR_MASK
        p = BY_ID.get(gid)
        if p is None or p.bandwidth > self.cfg.bandwidth_hz:
            return None
        if gid not in self.local_profiles():
            return None
        return p

    def tx_profile(self):
        from .profiles import PROFILES
        desired = self._floor_fallback or (None if self._profile_failed
                                           else self.cfg.data_profile)
        if self.cfg.bandwidth_hz < 1500 and not desired:
            desired = "narrow"
        if (not desired and not self._profile_failed and self.rung == len(DATA_LADDER) - 1
                and self._streak >= 2 * self.cfg.up_after):
            desired = "wide256"  # measured clean top-gear streak earns one probe
        if desired:
            p = PROFILES[desired]
            if p.id in self.peer_profiles and p.bandwidth <= self.cfg.bandwidth_hz:
                return p
        if self._eff_max_rung is None or self.cfg.bandwidth_hz < 1500:
            # No baseline profile is usable: choose an implemented explicitly
            # permitted profile. Prefer narrow modes before wider alternatives.
            for name in ("narrow", "narrow2", "narrow4", "sparse34", "doppler", "wide256"):
                p = PROFILES[name]
                if p.id in self.peer_profiles and p.bandwidth <= self.cfg.bandwidth_hz:
                    return p
            return None
        return PROFILES[DATA_LADDER[self.rung].name]

    def tx_gear(self, name):
        from .profiles import PROFILES
        gid = PROFILES[name].id
        return gid

    def _snr_offset_db(self, rung_idx: int) -> float:
        """Per-carrier SNR -> SNR-in-3-kHz conversion for this rung's gear."""
        gear = GEARS[DATA_LADDER[rung_idx].name]
        return 10 * np.log10(3000.0 / (gear.n_carriers * SPACING))

    def _my_capabilities(self) -> int:
        """Advertise only DATA profiles and features this receiver accepts."""
        feats = wire.PBACK
        if self.cfg.fast_ctrl and self.cfg.bandwidth_hz >= 1500:
            feats |= wire.FASTCTL
        if self.cfg.loading:
            feats |= wire.LOADING
        if self.cfg.deflate:
            feats |= wire.DEFLATE
        return wire.capabilities([i for i in self.local_profiles() if i < 24], feats)

    def _my_caps(self):
        ids = [i for i in self.local_profiles() if i >= 24]
        groups = [ids[i:i + 14] for i in range(0, len(ids), 14)]
        if len(groups) > wire.MAX_CAPS:
            raise ValueError("too many extended profiles")
        return [wire.Caps.build(self.session, wire.pack_tlvs([
            (wire.GEARSET, b"".join(i.to_bytes(2, "big") for i in group))]),
            len(groups), idx) for idx, group in enumerate(groups)]

    def begin_burst(self):
        self._caps_positions = {}
        self._rx_caps = []
        self._caps_invalid = False

    def _on_caps(self, c):
        parsed = wire.parse_tlvs(c.tlvs)
        if parsed is None:
            self._caps_invalid = True
            return
        previous = self._caps_positions.get(c.idx)
        if previous is not None:
            self._caps_invalid |= previous != c
            return
        self._caps_positions[c.idx] = c
        self._rx_caps.extend(parsed)

    def _caps_complete(self, c):
        return (not self._caps_invalid and set(self._caps_positions) == set(range(c.n_ext))
                and all(b.session == c.session and b.total == c.n_ext
                        for b in self._caps_positions.values()))

    def _unsupported_critical(self):
        return any(t not in wire.IMPLEMENTED_TLV for t in wire.crit_tlvs(self._rx_caps))

    def _intersect(self, c: wire.Control) -> None:
        """Derive transmission permission from the peer's complete fixed bitmap."""
        pc = self.peer_capabilities = c.capability_word
        self.use_fastctl = bool(self.cfg.fast_ctrl and self.cfg.bandwidth_hz >= 1500 and pc & wire.FASTCTL)
        self.use_pback = bool(pc & wire.PBACK)
        self.use_loading = bool(self.cfg.loading and pc & wire.LOADING)
        self.use_deflate = bool(self.cfg.deflate and pc & wire.DEFLATE)
        self.peer_profiles = wire.supported_profiles(pc) + wire.gearset_ids(self._rx_caps)
        from .profiles import PROFILES
        self._usable_rungs = [i for i, r in enumerate(DATA_LADDER)
                              if i <= self.cfg.max_rung
                              and wire.GEAR_ID[r.name] in self.peer_profiles
                              and PROFILES[r.name].bandwidth <= self.cfg.bandwidth_hz]
        self._eff_max_rung = max(self._usable_rungs, default=None)
        if self._usable_rungs:
            self._set_rung(self.rung)

    def _set_ctrl(self, fast: bool) -> None:
        if fast != self.ctrl_fast:
            self.ctrl_fast = fast
            self.stats["ctrl"].append(
                (self._clock(), "fast" if fast else "floor"))
            self.io.log(f"control tier -> {'fast' if fast else 'floor'}")

    def _update_ctrl_tier(self, snr3k: float) -> None:
        if not self.use_fastctl:
            return
        if self.ctrl_fast:
            if snr3k < self.cfg.fast_ctrl_off_db:
                self._set_ctrl(False)
        elif snr3k >= self.cfg.fast_ctrl_on_db:
            self._set_ctrl(True)

    def _maybe_id(self) -> None:
        if self._clock() - self._last_id >= self.cfg.id_interval_s:
            self._last_id = self._clock()
            self.stats["ids"].append(self._last_id)
            self.io.send_control(
                wire.Control.ident(wire.ID, self.session, self.cfg.callsign))

    @property
    def tx_pending(self) -> int:
        """Bytes accepted from the host and not yet delivered (BUFFER n)."""
        f = self._frame
        inflight = sum(map(len, f.chunks)) if f else 0
        return len(self._queue) + inflight

    # ==================================================================== #
    # Host events
    # ==================================================================== #
    def on_host_listen(self, on: bool) -> None:
        self._listen = on
        if on and self.state == SessionState.DISCONNECTED:
            self._to(SessionState.LISTENING)
        elif not on and self.state == SessionState.LISTENING:
            self._to(SessionState.DISCONNECTED)

    def on_host_connect(self, peer: str) -> None:
        self.peer = peer.upper()
        self.session = self._session_id_factory()
        if not 0 < self.session < 2**64:
            raise ValueError("session identifier must be a nonzero uint64")
        self._to(SessionState.CONNECTING)
        self._connect_retries = 0
        self._fast_probe_done = False
        self._send_connect()

    def on_host_data(self, blob: bytes) -> None:
        self._queue += blob
        if self.state != SessionState.CONNECTED:
            return
        if self.tx_profile() is None:
            self._fail_no_data_gear()
            return
        if self._iss:
            if self._frame is None:
                self._next_frame()
        elif self._deadline is None and self._rx is None:
            # quiescent IRS: after one control-burst's grace (an active peer
            # would have spoken by then), ask for the mic
            self._turn_reqs = 0
            self._arm("turnreq", self.cfg.turnaround_s + self.cfg.control_s)

    def on_host_disconnect(self) -> None:
        if self.state not in (SessionState.CONNECTED, SessionState.CONNECTING):
            return
        if self._frame is not None or self._queue:
            self._disc_pending = True
        elif self.state == SessionState.CONNECTED and not self._iss:
            self._disc_pending = True
            if self._deadline is None:
                self._turn_reqs = 0
                self._arm("turnreq", self.cfg.turnaround_s + self.cfg.control_s)
        else:
            self._begin_disconnect()

    def on_host_abort(self) -> None:
        """Dirty disconnect: discard queued TX, one DISC courtesy, drop."""
        if self.state in (SessionState.DISCONNECTED, SessionState.LISTENING):
            return
        self._queue.clear()
        self._frame = None
        if self.state == SessionState.CONNECTED:
            self.io.send_control(
                wire.Control.ident(wire.DISC, self.session, self.cfg.callsign))
        self._finish_disconnected()

    # ==================================================================== #
    # Decoded control blocks
    # ==================================================================== #
    def on_control(self, c: wire.Control, hdr_fast: bool = False) -> None:
        t = c.type
        if t == wire.CAPS:
            self._on_caps(c)
        elif t == wire.CONNECT:
            self._rx_connect(c, hdr_fast)
        elif t == wire.CONNECT_ACK:
            if (self.state == SessionState.CONNECTING and c.session == self.session
                    and c.call == self.peer and c.destination == self.cfg.callsign):
                if not self._caps_complete(c):
                    return
                if c.gear & wire.CFAIL or self._unsupported_critical():
                    self._finish_disconnected(DISC_REFUSED)
                    return
                self._deadline = None
                self._intersect(c)
                self._enter_connected(iss=True)
        elif t == wire.ID and c.session == self.session:
            if self.state == SessionState.CONNECTED and c.call == self.peer:
                self._heard_peer()
        elif t == wire.ACK and c.session == self.session:
            self._on_ack(c)
        elif t == wire.TURN and c.session == self.session:
            self._on_turn()
        elif t == wire.TURN_REQ and c.session == self.session:
            if self.state == SessionState.CONNECTED:
                self._heard_peer()
                self._peer_pending = True
                if self._iss and self._frame is None and not self._queue:
                    self._send_turn()
        elif t == wire.DISC:
            if c.session in (self.session, self._last_session):
                self.io.send_control(wire.Control.ident(
                    wire.DISC_ACK, c.session, self.cfg.callsign))
                if c.session == self.session and self.state in (SessionState.CONNECTED, SessionState.CONNECTING,
                                  SessionState.DISCONNECTING):
                    self._finish_disconnected(DISC_REMOTE)
        elif t == wire.DISC_ACK:
            if self.state == SessionState.DISCONNECTING and c.session == self.session:
                self._deadline = None
                self._finish_disconnected()

    def _rx_connect(self, c: wire.Control, hdr_fast: bool = False) -> None:
        # An optional-extension reply cannot fit the short probe response window.
        # Narrow or floor-only endpoints wait for the caller's floor retry.
        if hdr_fast and (not self.cfg.fast_ctrl or self.cfg.bandwidth_hz < 1500
                         or c.n_ext or self._my_caps()):
            return
        peer = c.call
        if not self._caps_complete(c):
            return
        if not c.session or (self.state == SessionState.LISTENING and c.session == self._last_session):
            return
        if c.destination != self.cfg.callsign or not peer:   # explicitly addressed session
            return
        if self.state == SessionState.LISTENING:
            self.session = c.session
            self.peer = peer
        elif self.state == SessionState.CONNECTED:
            # §3.6(a) idempotent re-offer: must match tag *and* identity, else
            # it is a colliding session or corruption -- ignore, never splice.
            if c.session != self.session or peer != self.peer:
                return
        else:
            return
        offer = (c, tuple(self._caps_positions[i] for i in sorted(self._caps_positions)))
        if self.state == SessionState.CONNECTED and offer != self._accepted_offer:
            return
        self._accepted_offer = offer
        if self.state == SessionState.CONNECTED:
            self._heard_peer()
        self._intersect(c)
        caps = self._my_caps()
        self.bootstrap_fast = hdr_fast
        refuse = self._unsupported_critical()
        self.io.send_control(wire.Control.connect(
            self.session, self._my_capabilities(), self.cfg.callsign,
            ack=True, xh=len(caps) | (wire.CFAIL if refuse else 0), destination=self.peer))
        for block in caps:
            self.io.send_control(block)
        if refuse:
            self._finish_disconnected(DISC_REFUSED)
        elif self.state != SessionState.CONNECTED:
            self._enter_connected(iss=False)

    def _send_connect(self) -> None:
        self._last_id = self._clock()
        caps = self._my_caps()
        self.bootstrap_fast = bool(not self._fast_probe_done and self.cfg.fast_ctrl
                                   and self.cfg.bandwidth_hz >= 1500 and not caps)
        self._fast_probe_done = True
        dur = self.io.send_control(wire.Control.connect(
            self.session, self._my_capabilities(), self.cfg.callsign,
            xh=len(caps), destination=self.peer))
        for block in caps:
            self.io.send_control(block)
        # The peer's optional extensions are unknown until its reply arrives.
        # A floor reply can occupy one CONNECT_ACK plus MAX_CAPS blocks; the
        # caller must not retransmit over a still-valid long response.
        floor_reply_budget = max(
            self.cfg.connect_timeout_s,
            (1 + wire.MAX_CAPS) * self.cfg.control_s
            + 2 * self.cfg.turnaround_s + self.cfg.ack_margin_s)
        wait = (2 * dur + 2 * self.cfg.turnaround_s + self.cfg.ack_margin_s
                if self.bootstrap_fast else dur + floor_reply_budget)
        self._arm("connect", wait)

    def _enter_connected(self, iss: bool) -> None:
        self._to(SessionState.CONNECTED)
        self._rx = None
        self._done_seq = None
        self._done_identity = None
        self._tx_offset = 0
        self._rx_offset = 0
        self._peer_generation = 0
        self._tx_busy_until = 0.0
        self._heard_peer()
        self._streak = 0
        self._rebuilds = 0
        self._profile_failed = False
        self._floor_fallback = None
        self._set_ctrl(False)               # SNR unknown until measured
        self._iss = iss
        self._peer_pending = False
        self._last_id = self._clock()   # the handshake carried the callsign
        self.io.connected(self.session, self.peer)
        if self._queue and self.tx_profile() is None:
            self._fail_no_data_gear()
            return
        if self._iss:
            if self._queue:
                self._next_frame()
            elif self._disc_pending:
                self._begin_disconnect()
        elif self._queue:
            self._turn_reqs = 0
            self._arm("turnreq", self.cfg.turnaround_s + self.cfg.control_s)

    # -- turn exchange -----------------------------------------------------
    def _send_turn(self) -> None:
        dur = self.io.send_control(wire.Control(wire.TURN, self.session))
        self._iss = False
        self._peer_pending = False
        self._arm("turn", dur + 2 * self.cfg.turnaround_s
                  + self.cfg.control_s + self.cfg.ack_margin_s)

    def _on_turn(self) -> None:
        if self.state != SessionState.CONNECTED or self._iss:
            return
        self._heard_peer()
        if self._kind in ("turn", "turnreq"):
            self._deadline = None
        self._iss = True
        self._peer_pending = False
        self._turn_reqs = 0
        self._next_frame()

    # ==================================================================== #
    # TX engine
    # ==================================================================== #
    def _fail_no_data_gear(self) -> None:
        """Host data queued for a peer advertising no usable data profile
        (SPEC.md §5): nothing conformant can carry it -- R1 forbids
        transmitting outside the advertised set -- so fail the link loud
        rather than stall on a queue that can never drain."""
        self._queue.clear()
        self.io.log("link failed: no data gear within the peer's supported profiles")
        self.io.send_control(wire.Control.ident(
            wire.DISC, self.session, self.cfg.callsign))
        self._finish_disconnected(DISC_LINK_FAILED)

    def _next_frame(self) -> None:
        if (self._frame is not None or self.state != SessionState.CONNECTED
                or not self._iss):
            return
        if not self._queue:
            if self._peer_pending:
                self._send_turn()
            elif self._disc_pending:
                self._begin_disconnect()
            return
        if self.tx_profile() is None:
            self._fail_no_data_gear()
            return
        if self._next_seq >= 2**32 or self._tx_offset + len(self._queue) >= 2**64:
            self._finish_disconnected(DISC_LINK_FAILED)
            return
        rc = self.tx_profile()
        codec = self.profile_codec(rc.name)
        take = min(len(self._queue), rc.frame_cws * codec.data_bytes)
        payload = bytes(self._queue[:take])
        del self._queue[:take]
        chunks = codec.chunk(payload)
        coded = np.stack([codec.encode_cw(ch) for ch in chunks])
        self._frame = _TxFrame(seq=self._next_seq, rung=_RUNG_OF_GEAR.get(rc.id, self.rung),
                               chunks=chunks, coded=coded, profile=rc.name, offset=self._tx_offset)
        self._next_seq += 1
        self._tx_offset += take
        self._send_round()

    def _send_round(self) -> None:
        self._maybe_id()
        f = self._frame
        n_cw = len(f.chunks)
        present = (sorted(set(range(n_cw)) - f.acked)
                   if self.cfg.selective else list(range(n_cw)))
        name = f.profile or DATA_LADDER[f.rung].name
        # Extended profiles use their specified constellation; baseline loading
        # nibbles cannot silently down-map a 256-QAM profile to 64-QAM.
        nibbles = (tuple(self._nibbles) if self.use_loading and wire.GEAR_ID[name] < 15
                   else (0,) * 8)
        gear = self.tx_gear(name)
        if self._peer_pending and not self._queue and self.use_pback:
            gear |= wire.HANDOVER
        dur = self.io.send_data(f.seq, gear,
                                present, n_cw, nibbles, f.coded[present], f.offset)
        self.pb_ack = None
        # A long legal local transmission precludes any peer response. Do not
        # expire mid-burst; allow its floor ACK/turnaround window to finish.
        self._tx_busy_until = self._clock() + dur + 2 * self.cfg.turnaround_s + self.cfg.control_s + self.cfg.ack_margin_s
        if f.tx_count:
            f.retx_cws += len(present)
        f.tx_count += 1
        self.stats["sends"].append(
            dict(t=self._clock(), seq=f.seq, round=f.tx_count,
                 rung=name, n_cw=len(present), dur=dur))
        self._arm("ack", dur + 2 * self.cfg.turnaround_s + self.cfg.control_s
                  + self.cfg.ack_margin_s)

    def _on_ack(self, c: wire.Control) -> None:
        f = self._frame
        if (self.state != SessionState.CONNECTED or f is None or c.seq != f.seq
                or int.from_bytes(c.mask, "little") >> len(f.chunks)
                or c.gear & ~(wire.ACK_TRAFFIC | wire.ACK_TOOK_ROLE)):
            return
        self._heard_peer()
        self._tx_busy_until = 0.0
        self._deadline = None
        f.timeouts = 0
        self._peer_pending = bool(c.gear & wire.ACK_TRAFFIC)
        if c.gear & wire.ACK_TOOK_ROLE:     # piggybacked: DATA follows
            self._iss = False
            self._peer_pending = False
        self.stats["rounds"] += 1
        newly = set(wire.mask_indices(c.mask, len(f.chunks))) - f.acked
        f.acked |= newly
        if not f.profile or wire.GEAR_ID[f.profile] < 15:
            self._update_quality(wire.parse_ack_aux(c.aux), f.rung)
        elif f.profile in GEARS:
            live = [v for v in wire.parse_ack_aux(c.aux) if v is not None]
            if live:
                g = GEARS[f.profile]
                self._update_ctrl_tier(float(np.median(live)) -
                    10 * np.log10(3000 / (g.n_carriers * 48000 / g.n_fft)))
        if len(f.acked) == len(f.chunks):
            self._frame_done()
        elif not newly or f.tx_count >= self.cfg.max_rounds:
            self._rebuild()
        else:
            self._send_round()

    def _frame_done(self) -> None:
        f = self._frame
        self._frame = None
        self._rebuilds = 0
        self.stats["frames"].append(
            dict(t=self._clock(), seq=f.seq, rung=f.profile or DATA_LADDER[f.rung].name,
                 tx_count=f.tx_count, n_cw=len(f.chunks),
                 snr3k=self._link_snr3k))
        self._streak = self._streak + 1 if f.clean else 0
        if f.profile and wire.GEAR_ID[f.profile] >= 16:
            self._next_frame()
            return
        r = f.rung
        snr = self._link_snr3k
        next_rung = min((i for i in self._usable_rungs if i > r), default=None)
        if (snr is not None and r > 0
                and snr < DATA_LADDER[r].up_snr_db
                - self.cfg.down_hysteresis_db):
            self._set_rung(r - 1)
        elif (self._streak >= self.cfg.up_after and next_rung is not None
              and snr is not None
              and snr >= DATA_LADDER[next_rung].up_snr_db):
            self._set_rung(next_rung)
        elif (self._streak >= 2 * self.cfg.up_after
              and next_rung is not None
              and (snr is None or snr >= DATA_LADDER[next_rung].up_snr_db
                   - self.cfg.probe_margin_db)):
            # the SNR measurement saturates near ~20 dB (channel-estimation
            # error floor), so sustained clean frames earn a probe of the
            # next rung; a failed probe costs one HARQ round + rebuild
            self._set_rung(next_rung)
        self._next_frame()

    def _rebuild(self) -> None:
        """Give up combining this frame: requeue its payload and start a new
        seq one rung down (the receiver drops its store on the seq change)."""
        f = self._frame
        self._frame = None
        self.stats["rebuilds"] += 1
        self._rebuilds += 1
        if f.profile and (wire.GEAR_ID[f.profile] >= 16 or f.profile == self.cfg.data_profile):
            self._profile_failed = True
        if f.rung == 0 or self._floor_fallback or self.cfg.bandwidth_hz < 1500:
            candidates = ("narrow", "narrow2", "narrow4")
            start = (candidates.index(self._floor_fallback) + 1
                     if self._floor_fallback in candidates else 0)
            for name in candidates[start:]:
                if wire.GEAR_ID[name] in self.peer_profiles:
                    self._floor_fallback = name
                    break
        self._tx_offset = f.offset
        self._queue[:0] = b"".join(f.chunks)
        if self._rebuilds > self.cfg.max_rebuilds:
            self.io.log("link failed: rebuild limit")
            self._finish_disconnected(DISC_LINK_FAILED)
            return
        if f.rung > 0:
            self._set_rung(f.rung - 1)
        self._streak = 0
        self._next_frame()

    def _set_rung(self, r: int) -> None:
        if self._eff_max_rung is None:
            return                          # no DATA to this peer at all
        candidates = [i for i in self._usable_rungs if i <= r]
        r = max(candidates) if candidates else min(self._usable_rungs)
        if r == self.rung:
            return
        self.io.log(f"gearshift {DATA_LADDER[self.rung].name} -> "
                    f"{DATA_LADDER[r].name}")
        self.stats["shifts"].append(
            dict(t=self._clock(), rung=DATA_LADDER[r].name))
        self.rung = r
        self._streak = 0
        self._nibbles = [0] * 8
        self._gsnr = {}

    def _update_quality(self, group_snr, rung_idx: int) -> None:
        rc = DATA_LADDER[rung_idx]
        for g, db in enumerate(group_snr):
            if db is None:
                continue
            self._gsnr[g] = db
            if db < self.cfg.dead_db:
                self._nibbles[g] = 1
            elif rc.grouped:
                self._nibbles[g] = (4 if db >= self.cfg.t64_db else
                                    3 if db >= self.cfg.t16_db else 2)
            else:
                self._nibbles[g] = 0
        if all(n == 1 for n in self._nibbles):
            self._nibbles = [0] * 8         # never mask the whole band
        live = [db for g, db in self._gsnr.items() if self._nibbles[g] != 1]
        if live:
            self._link_snr3k = float(np.median(live)) - self._snr_offset_db(
                rung_idx)
            self._update_ctrl_tier(self._link_snr3k)

    # ==================================================================== #
    # RX engine (HARQ combining + selective ACK)
    # ==================================================================== #
    def on_data(self, c: wire.Control, cw_llrs, group_snr,
                hdr_fast: bool = False) -> None:
        if self.state != SessionState.CONNECTED or c.session != self.session:
            return
        profile = self.rx_profile(c.gear)
        _, n_cw, _ = wire.parse_data_aux(c.aux)
        if profile is None or not self.on_data_header(c, 0.0):
            return
        if self._kind in ("turn", "turnreq"):
            self._deadline = None           # valid peer DATA satisfies the turn wait
        self._turn_reqs = 0
        if self._frame is None:
            self._iss = False               # the peer audibly holds the role
        rung_idx = _RUNG_OF_GEAR.get(profile.id)
        if rung_idx is not None and group_snr is not None:
            live = [db for db in group_snr if db is not None]
            if live:
                self._update_ctrl_tier(float(np.median(live))
                                       - self._snr_offset_db(rung_idx))
        elif group_snr is not None and profile.name in GEARS:
            live = [v for v in group_snr if v is not None]
            if live:
                g = GEARS[profile.name]
                self._update_ctrl_tier(float(np.median(live)) -
                    10 * np.log10(3000 / (g.n_carriers * 48000 / g.n_fft)))
        if not hdr_fast:
            # never answer faster than the sender spoke: a floor header means
            # the ISS thinks the link is marginal (or just lost our fast ACK)
            self._set_ctrl(False)
        identity = (c.seq, c.offset, n_cw, c.gear & wire.GEAR_MASK)
        if identity == self._done_identity:
            self._send_ack(c.seq, range(n_cw), group_snr)
            return
        codec = self.profile_codec(profile.name)
        rx = self._rx
        # The store is keyed by frame identity: seq, gear (HANDOVER may toggle
        # between rounds of one frame), and n_cw. A colliding identity restarts
        # it -- summing LLRs of different frames could only waste rounds, and
        # a geometry change would make the rows unindexable.
        if (rx is None or rx.seq != c.seq or rx.n_cw != n_cw
                or rx.offset != c.offset or (rx.gear ^ c.gear) & ~wire.HANDOVER):
            rx = self._rx = _RxFrame(
                seq=c.seq, gear=c.gear, n_cw=n_cw,
                store=np.zeros((n_cw, codec.code.n)), chunks=[None] * n_cw, offset=c.offset)
        if cw_llrs is not None:
            present = wire.mask_indices(c.mask, n_cw)
            for row, j in zip(cw_llrs, present):
                rx.store[j] = rx.store[j] + row if self.cfg.harq else row
            todo = [j for j in present if rx.chunks[j] is None]
            if todo:
                got, _ = codec.decode_cws(rx.store[todo])
                for j, chunk in zip(todo, got):
                    if chunk is not None:
                        rx.chunks[j] = chunk
        ok = [j for j, ch in enumerate(rx.chunks) if ch is not None]
        if len(ok) == n_cw:
            payload = b"".join(rx.chunks)
            skip = self._rx_offset - c.offset
            if skip < len(payload):
                self.io.deliver(payload[skip:])
                self._rx_offset = c.offset + len(payload)
            self._done_seq = c.seq
            self._done_identity = identity
            self._rx = None
            if (c.gear & wire.HANDOVER and self.ctrl_fast and self._queue
                    and self._frame is None):
                # take the offered role: our DATA over carries the final ACK
                self._iss = True
                self._turn_reqs = 0
                self.stats["pb_acks"] += 1
                self.pb_ack = wire.Control(
                    wire.ACK, self.session, seq=c.seq,
                    gear=wire.ACK_TOOK_ROLE, mask=wire.cw_mask(ok),
                    aux=wire.ack_aux(group_snr))
                self._next_frame()
                return
        self._send_ack(c.seq, ok, group_snr)

    def _send_ack(self, seq: int, ok, group_snr) -> None:
        self._maybe_id()
        gear = (wire.ACK_TRAFFIC if self._queue or self._disc_pending else 0)
        if self._iss and self._frame is not None:
            # re-ACK after our piggybacked over was lost: we still hold the
            # role, so the peer must not turn around or hand it out again
            gear |= wire.ACK_TOOK_ROLE
        self.io.send_control(wire.Control(
            wire.ACK, self.session, seq=seq, gear=gear,
            mask=wire.cw_mask(ok), aux=wire.ack_aux(group_snr)))

    # ==================================================================== #
    # Disconnect
    # ==================================================================== #
    def _begin_disconnect(self) -> None:
        self._idle_deadline = None
        self._disc_pending = False
        self._to(SessionState.DISCONNECTING)
        self._connect_retries = 0
        self._last_id = self._clock()
        dur = self.io.send_control(
            wire.Control.ident(wire.DISC, self.session, self.cfg.callsign))
        self._arm("disc", dur + self.cfg.connect_timeout_s)

    def _finish_disconnected(self, reason: int = DISC_LOCAL) -> None:
        self._deadline = None
        self._idle_deadline = None
        self._tx_busy_until = 0.0
        self._frame = None
        self._rx = None
        self._queue.clear()
        self._disc_pending = False
        self.ctrl_fast = False          # silent: teardown, not a fallback
        self.pb_ack = None
        self._iss = False
        self._peer_pending = False
        self._last_session = self.session
        self.session = 0
        self.disc_reason = reason       # host reads it in modem_disconnected
        self.io.disconnected()
        self._to(SessionState.LISTENING if self._listen else SessionState.DISCONNECTED)

    # ==================================================================== #
    # Timer
    # ==================================================================== #
    def on_timer(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        if (self.state == SessionState.CONNECTED and self._idle_deadline is not None
                and now >= max(self._idle_deadline, self._tx_busy_until)):
            self.io.log("link failed: peer inactivity")
            self.io.send_control(wire.Control.ident(wire.DISC, self.session, self.cfg.callsign))
            self._finish_disconnected(DISC_LINK_FAILED)
            return
        if self._deadline is None or now < self._deadline:
            return
        kind, self._deadline = self._kind, None
        if kind == "ack":
            f = self._frame
            if f is None:
                return
            f.timeouts += 1
            self.stats["timeouts"] += 1
            self.io.log(f"ACK timeout (frame {f.seq})")
            self._set_ctrl(False)           # the control round trip died
            if f.timeouts >= 2 or f.tx_count >= self.cfg.max_rounds:
                self._rebuild()
            else:
                self._send_round()
        elif kind == "connect":
            if self.bootstrap_fast:
                self._send_connect()       # probe is additional to the floor retry budget
                return
            self._connect_retries += 1
            if self._connect_retries > self.cfg.max_connect_retries:
                self.io.log("connect failed")
                self._finish_disconnected(DISC_LINK_FAILED)
            else:
                self._send_connect()
        elif kind == "turn":
            # the peer never picked the role up: take it back
            self._set_ctrl(False)
            self._iss = True
            self._next_frame()
        elif kind == "turnreq":
            if (self.state == SessionState.CONNECTED and not self._iss
                    and (self._queue or self._disc_pending)
                    and self._turn_reqs < self.cfg.max_turn_reqs):
                if self._turn_reqs:
                    self._set_ctrl(False)   # first ask went unanswered
                self._turn_reqs += 1
                dur = self.io.send_control(
                    wire.Control(wire.TURN_REQ, self.session))
                self._arm("turnreq", dur + 2 * self.cfg.turnaround_s
                          + self.cfg.control_s + self.cfg.ack_margin_s)
            elif (self.state == SessionState.CONNECTED and not self._iss
                  and (self._queue or self._disc_pending)):
                self.io.log("link failed: unanswered turn requests")
                self.io.send_control(wire.Control.ident(
                    wire.DISC, self.session, self.cfg.callsign))
                self._finish_disconnected(DISC_LINK_FAILED)
        elif kind == "disc":
            self._connect_retries += 1
            if self._connect_retries > 2:
                self._finish_disconnected()
            else:
                self._set_ctrl(False)
                dur = self.io.send_control(wire.Control.ident(
                    wire.DISC, self.session, self.cfg.callsign))
                self._arm("disc", dur + self.cfg.connect_timeout_s)
