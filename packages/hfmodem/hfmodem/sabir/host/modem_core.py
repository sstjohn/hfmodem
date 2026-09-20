# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Sabir host commands and telemetry over a serialized modem execution port.

The host reader submits commands to the air thread through ``submit``. The
simulator and StationAir provide that execution port; station wiring alone is
not evidence of a completed two-station radio exchange.
"""

from __future__ import annotations

import abc
import math
import zlib
from typing import Optional

from hfmodem.sabir import compress
from hfmodem.sabir.arq import ArqConfig, LinkModem, SessionState, wire

from hfmodem.sabir.arq.profiles import EXTENDED, PROFILES
from hfmodem.sabir.frame.datagram import Fragment, Reassembler, fragment_object
from collections import deque
from dataclasses import replace


class ModemObserver(abc.ABC):
    """Link events -> host wire messages. Implemented by the server session;
    methods may be called from the air thread, so they must be thread-safe."""

    @abc.abstractmethod
    def modem_connected(self, src: str, dst: str, bw: Optional[str]) -> None:
        """Link established; report the peer identity."""

    @abc.abstractmethod
    def modem_disconnected(self, reason: int = 0) -> None:
        """Link closed -> ``DISCONNECTED``. ``reason`` is the FSM ``DISC_*``
        cause (0 = unset), surfaced as StateChanged.reason (HOST-API §5)."""

    @abc.abstractmethod
    def modem_ptt(self, on: bool) -> None:
        """Key/unkey the radio -> ``PTT ON`` / ``PTT OFF``."""

    @abc.abstractmethod
    def modem_data_received(self, blob: bytes) -> None:
        """A complete record arrived from the air."""

    def modem_busy(self, on: bool) -> None:
        """Channel busy detector -> ``BUSY ON`` / ``BUSY OFF``."""

    def modem_capabilities(self, image: dict) -> None:
        """Air capability intersection -> CapabilitiesNegotiated."""

    def modem_link_stats(self, snapshot: dict) -> None:
        """Link telemetry -> LinkStats."""

    def modem_id_sent(self, station_id: str, t: float) -> None:
        """A §97.119 identification went out -> IdSent."""

    def modem_send_progress(self, msg_id: int, delivered: bool) -> None:
        """A tracked message's bytes were all peer-ACKed -> SendProgress."""

    def modem_state_changed(self, state: str) -> None:
        """A link-lifecycle transition -> StateChanged (LISTENING / CONNECTING /
        DISCONNECTING; CONNECTED/DISCONNECTED come via the callbacks above,
        which carry the peer_id and the disconnect reason respectively)."""

    def modem_peer_observed(self, image: dict) -> None:
        """A sabir-capable peer seen via sounding/beacon -> PeerObserved."""

    def modem_object_received(self, image: dict) -> None:
        """An integrity-checked connectionless object; no ACK was sent."""


class ModemCore(abc.ABC):
    """Commands run on the owning modem thread through ``submit``.
    Link results arrive asynchronously through the observer."""

    @abc.abstractmethod
    def start(self, observer: ModemObserver) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    def submit(self, command) -> None:
        command()

    def set_identity(self, station_id: str) -> None: ...
    def set_profile(self, profile: int) -> None: ...
    def set_listen(self, on: bool) -> None: ...

    @abc.abstractmethod
    def connect(self, src: str, dst: str) -> None: ...

    @abc.abstractmethod
    def transmit(self, blob: bytes, msg_id: Optional[int] = None,
                 *, deflate: bool = False) -> bool: ...

    def beacon(self, addressee: int = 0) -> None: ...

    def configure_data(self, **options) -> None: ...

    def send_object(self, data, **options):
        raise NotImplementedError("connectionless objects unavailable")

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def abort(self) -> None: ...

    @property
    @abc.abstractmethod
    def connected(self) -> bool: ...


class _HostLink(LinkModem):
    """The ARQ endpoint with its link events forwarded to the host modem."""

    owner: "SabirModem"

    def connected(self, session, peer):
        super().connected(session, peer)
        self.owner._link_connected(peer)

    def disconnected(self):
        super().disconnected()
        self.owner._link_down()

    def deliver(self, blob: bytes):
        super().deliver(blob)
        self.owner._link_data(blob)

    def on_beacon(self, beacon):
        super().on_beacon(beacon)
        self.owner._link_beacon(beacon)

    def state_changed(self, state: str):
        self.owner._link_state(state)

    def on_datagram(self, data):
        self.owner._link_datagram(data)


class SabirModem(ModemCore):
    def __init__(self, air, cfg: ArqConfig | None = None, dd: int = 1):
        self.air = air
        self.link = _HostLink(cfg or ArqConfig(advertise_gears=EXTENDED), air.clock, dd=dd)
        self.link.owner = self
        self._observer: Optional[ModemObserver] = None
        self._rx = compress.Unpacker()
        self._profile = 0                   # operating profile (HOST-API §7)
        self.link_bytes_tx = 0              # record bytes handed to the ARQ
        self._raw_tx = 0                    # app bytes before compression
        self._delivered = 0                 # app bytes handed up this session
        self._t0 = 0.0                      # session start (for throughput)
        self._ids_seen = 0                  # stats["ids"] length last observed
        self._snap = None                   # last LinkStats snapshot emitted
        self._tx_base = 0                   # link_bytes_tx at session start
        self._tx_app = 0                    # app-payload bytes submitted, session
        self._marks: list = [(0, 0)]        # (record_bytes, app_bytes) per Send
        self._pending: list = []            # [msg_id, end_offset] awaiting ACK
        self._objects = deque()
        self._object_pending = False
        self._reassembler = Reassembler(clock=air.clock)
        air.register(self)

    @property
    def fsm(self):
        return self.link.fsm

    # -- ModemCore ---------------------------------------------------------
    def start(self, observer: ModemObserver) -> None:
        self._observer = observer

    def stop(self) -> None:
        self._observer = None

    def set_identity(self, station_id: str) -> None:
        self.fsm.cfg.callsign = station_id.upper()

    def submit(self, command) -> None:
        self.air.post(command)

    def set_profile(self, profile: int) -> None:
        if profile not in (1, 2):
            raise ValueError("unsupported operating profile")
        self._profile = profile

    def set_listen(self, on: bool) -> None:
        self.fsm.on_host_listen(on)

    def connect(self, src: str, dst: str) -> None:
        if self.fsm.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
            raise ValueError("connection already active")
        if self._object_pending:
            raise ValueError("connectionless transmission is pending")
        def go():
            self.fsm.cfg.callsign = src.upper()
            self.fsm.on_host_connect(dst)
        go()

    def transmit(self, blob: bytes, msg_id: Optional[int] = None,
                 *, deflate: bool = False) -> bool:
        rec = compress.pack(blob, deflate and self.fsm.use_deflate
                            and self.fsm.state == SessionState.CONNECTED)
        self._raw_tx += len(blob)
        self.link_bytes_tx += len(rec)
        self._tx_app += len(blob)
        self._marks.append((self.link_bytes_tx - self._tx_base, self._tx_app))
        if msg_id is not None:
            self._pending.append([msg_id, self.link_bytes_tx - self._tx_base])
        self.fsm.on_host_data(rec)
        return bool(rec[0] & compress.DEFLATE)

    def beacon(self, addressee: int = wire.ALL_CALL) -> None:
        if self._object_pending or self.fsm.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
            raise ValueError("beacon requires an idle link")
        def go():
            bc = wire.Beacon.build(self.fsm._my_capabilities(), self.fsm.cfg.callsign,
                                   addressee=addressee, profile=self._profile)
            self.link.send_beacon(bc)
        go()

    def configure_data(self, **options):
        allowed = {"data_profile", "receive_profiles", "feedback_iters", "impulse_blank", "bandwidth_hz", "inactivity_timeout_s"}
        if options.keys() - allowed:
            raise ValueError("unknown DATA option")
        if self.fsm.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
            raise ValueError("DATA configuration is fixed for the session")
        timeout = options.get("inactivity_timeout_s", self.fsm.cfg.inactivity_timeout_s)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("inactivity timeout must be finite and positive")
        names = tuple(options.get("receive_profiles", self.fsm.cfg.advertise_gears))
        if len(names) > 8 or len(set(names)) != len(names) or any(n not in EXTENDED for n in names):
            raise ValueError("invalid receive profiles")
        preferred = options.get("data_profile", self.fsm.cfg.data_profile)
        if preferred is not None and preferred not in PROFILES:
            raise ValueError("invalid DATA profile")
        if not 0 <= options.get("feedback_iters", 0) <= 3 or not 0 <= options.get("impulse_blank", 0) <= 20:
            raise ValueError("invalid receiver options")
        if options.get("bandwidth_hz", 2750) not in (500, 1500, 2300, 2750):
            raise ValueError("unsupported bandwidth")

        def apply():
            if self.fsm.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
                return
            self.fsm.cfg.advertise_gears = names
            for name, value in options.items():
                if name != "receive_profiles":
                    setattr(self.fsm.cfg, name, value)
        apply()

    def send_object(self, data, *, destination="*", service=0, data_profile="workhorse",
                    parity=True, repeats=1, message_id=None):
        if self._object_pending or self.fsm.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
            raise ValueError("connectionless send requires an idle link")
        p = PROFILES[data_profile]
        if p.bandwidth > self.fsm.cfg.bandwidth_hz or not 1 <= repeats <= 8:
            raise ValueError("invalid object transmission profile")
        records = fragment_object(data, self.fsm.cfg.callsign, destination, service=service,
                                  fragment_bytes=32 if p.floor else 1024, parity=parity,
                                  message_id=message_id)
        self._object_pending = True

        def go():
            if self._objects or self.fsm.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
                self.link.log("object send rejected: modem busy")
                self._object_pending = False
                return
            # Bytes are queued, not multi-minute waveforms. Render one packet
            # per air step, bounding audio memory independently of file size.
            packets = [replace(r, source=self.fsm.cfg.callsign).pack() for r in records]
            self._objects.extend((packet, data_profile) for _ in range(repeats) for packet in packets)
        go()
        return records[0].message_id

    def disconnect(self) -> None:
        self.fsm.on_host_disconnect()

    def abort(self) -> None:
        self._objects.clear()
        self._object_pending = False
        self.fsm.on_host_abort()

    @property
    def connected(self) -> bool:
        return self.fsm.state == SessionState.CONNECTED

    # -- telemetry -------------------------------------------------------
    def capability_image(self) -> dict:
        """The air capability intersection, as a CapabilitiesNegotiated body.
        Everything here is what the FSM negotiated (NEGOTIATION.md §3.1)."""
        f = self.fsm
        return {"peer_id": f.peer, "peer_capabilities": f.peer_capabilities,
                "peer_profiles": list(f.peer_profiles),
                "usable": {"fastctl": f.use_fastctl, "pback": f.use_pback,
                           "loading": f.use_loading, "deflate": f.use_deflate}}

    def _queued_app_bytes(self) -> int:
        """Application-payload bytes submitted this session and not yet
        peer-ACKed (HOST-API §6). `tx_pending` is record bytes -- compressed
        body plus a 4-byte record header -- so it is mapped back to submitted
        payload bytes through the per-Send (record, app) marks and reported in
        the units a host progress bar expects."""
        confirmed = max(0, (self.link_bytes_tx - self._tx_base)
                        - self.fsm.tx_pending)
        app_done = self._tx_app                      # all, if fully drained
        for (r0, a0), (r1, a1) in zip(self._marks, self._marks[1:]):
            if confirmed < r1:                       # confirmed lands in (r0,r1]
                span = r1 - r0
                app_done = a0 + (a1 - a0) * (confirmed - r0) / span if span \
                    else a0
                break
        return max(0, int(self._tx_app - app_done))

    def link_snapshot(self) -> dict:
        """A LinkStats body from live FSM state -- gear, per-group SNR, control
        tier, queue depth, HARQ accounting, throughput, compression."""
        f = self.fsm
        elapsed = max(self.air.clock() - self._t0, 1e-3)
        profile = f.tx_profile()
        snap = {"gear": profile.name if profile else "none", "rung": f.rung,
                "snr3k_db": f._link_snr3k,
                "group_snr_db": [f._gsnr.get(g) for g in range(8)],
                "control_tier": 1 if f.ctrl_fast else 0,
                "queue_bytes": self._queued_app_bytes(),
                "harq_rounds": f.stats["rounds"],
                "rebuilds": f.stats["rebuilds"],
                "throughput_bps": self._delivered * 8.0 / elapsed}
        if self._raw_tx:
            snap["compression_ratio"] = self.link_bytes_tx / self._raw_tx
        return snap

    # -- link events (air thread) ------------------------------------------
    def _link_connected(self, peer: str) -> None:
        self._rx = compress.Unpacker()
        self._delivered = 0
        self._t0 = self.air.clock()
        # Preserve records submitted before CONNECT, including their IDs.
        self._ids_seen = len(self.fsm.stats["ids"])
        if self._observer:
            self._observer.modem_connected(self.fsm.cfg.callsign, peer,
                                           str(self.fsm.cfg.bandwidth_hz))
            self._observer.modem_capabilities(self.capability_image())

    def _link_down(self) -> None:
        self._pending = []                  # undelivered messages abandoned
        self._tx_base = self.link_bytes_tx
        self._tx_app = 0
        self._marks = [(0, 0)]
        self._snap = None
        if self._observer:
            self._observer.modem_disconnected(self.fsm.disc_reason)

    def _link_data(self, blob: bytes) -> None:
        try:
            for payload in self._rx.feed(blob):
                self._delivered += len(payload)
                if self._observer:
                    self._observer.modem_data_received(payload)
        except (ValueError, zlib.error) as e:
            # never deliver a body under an unknown encoding (SPEC.md §6.3.1);
            # posted so the abort runs after the FSM finishes this burst
            self.link.log(f"record stream poisoned ({e}); dropping the link")
            self.air.post(self.fsm.on_host_abort)

    def _link_beacon(self, beacon) -> None:
        if self._observer:
            self._observer.modem_peer_observed(
                {"peer_id": beacon.call, "profile": beacon.profile,
                 "capabilities": wire.decode_capabilities(beacon.capability_word)})

    def _link_datagram(self, data):
        try:
            fragment = Fragment.unpack(data)
            if fragment.destination not in ("*", self.fsm.cfg.callsign.upper()):
                return
            result = self._reassembler.accept(data)
        except (ValueError, UnicodeError):
            return
        if result and self._observer:
            f, payload = result
            self._observer.modem_object_received({"peer_id": f.source,
                "destination": f.destination, "message_id": f.message_id,
                "service": f.service, "data": payload, "t": float(self.air.clock())})

    def _link_state(self, state: str) -> None:
        if self._observer:
            self._observer.modem_state_changed(state)

    # -- air-side port ------------------------------------------------------
    def ptt(self, on: bool) -> None:
        if self._observer:
            self._observer.modem_ptt(on)

    def after_step(self) -> None:
        if self._object_pending and not self._objects and not self.link.outbox:
            self._object_pending = False
        if self._objects and not self.link.outbox and self.fsm.state in (SessionState.DISCONNECTED, SessionState.LISTENING):
            packet, profile = self._objects.popleft()
            self.link.send_datagram(packet, profile)
        if not self._observer:
            return
        ids = self.fsm.stats["ids"]
        if len(ids) > self._ids_seen:
            self._ids_seen = len(ids)
            self._observer.modem_id_sent(self.fsm.cfg.callsign, ids[-1])
        if self.fsm.state == SessionState.CONNECTED:
            if self._pending:
                acked = (self.link_bytes_tx - self._tx_base
                         - self.fsm.tx_pending)
                still = []
                for msg_id, end in self._pending:
                    if end <= acked:
                        self._observer.modem_send_progress(msg_id, True)
                    else:
                        still.append([msg_id, end])
                self._pending = still
            snap = self.link_snapshot()
            material = {k: snap[k] for k in ("gear", "control_tier",
                                             "queue_bytes", "rebuilds")}
            if material != self._snap:
                self._snap = material
                self._observer.modem_link_stats(snap)
