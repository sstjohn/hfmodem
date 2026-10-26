# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import math
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from typing import Callable

import numpy as np

from hfmodem.core import cwid, levels, resample
from hfmodem.core.audio import BurstInvalidated
from hfmodem.core.operations import LeaseTimeout
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.regulatory import Emission
from hfmodem.core.rig import Rig, RigError, _WATCHDOG_FRACTION, _WATCHDOG_SLACK_S

#: Priorities. Two is enough; a third would need a reason.
LIVE = 0
ERRAND = 1

_log = logging.getLogger(__name__)


def longest_burst_s(max_key_s: float, lead_s: float = 0.0) -> float:
    """The longest burst admitted after ``lead_s`` within ``max_key_s``, once the
    rig's watchdog slack is counted. Admission refuses anything longer."""
    room = max_key_s - lead_s
    return min(room - _WATCHDOG_SLACK_S, room / (1 + _WATCHDOG_FRACTION))


class Refused(Exception):
    """The burst was not transmitted, and the message says why.

    `submit` can also raise `BurstInvalidated` — the burst was keyed and what went
    out is not what was composed. It is deliberately not a `Refused`, because the
    two need different responses: one is retry, the other is find out why.
    """


class RequestState:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = "waiting"

    @property
    def value(self):
        with self._lock:
            return self._value

    def cancel(self):
        with self._lock:
            if self._value != "waiting":
                return False
            self._value = "cancelled"
            return True

    def commit(self):
        with self._lock:
            if self._value != "waiting":
                return False
            self._value = "committed"
            return True


@dataclass(frozen=True)
class TxResult:
    outcome: str
    state: str
    end: int | None = None
    reason: str = ""
    displacement: int = 0
    key_index: int | None = None
    keyed_s: float = 0.0
    emission_start: int | None = None
    emission_end: int | None = None


@dataclass(frozen=True)
class TransmitHistory:
    """A coherent receive freshness barrier between transmit requests."""

    sequence: int = 0
    end: int = 0
    receive_floor: int = field(default=0, compare=False)

    @property
    def capture_floor(self):
        """Receive retirement may outlive an interrupted physical emission."""
        return max(self.end, self.receive_floor)


@dataclass(order=True)
class TxRequest:
    priority: int
    seq: int = field(compare=True)
    protocol: str = field(compare=False, default="")
    audio: np.ndarray = field(compare=False, default_factory=lambda: np.zeros(0))
    rate: int = field(compare=False, default=48000)
    emission: Emission | None = field(compare=False, default=None)
    #: Scheduled on-air capture index; finite placement subtracts TX latency.
    at: int | None = field(compare=False, default=None)
    settle_s: float = field(compare=False, default=0.0)
    #: True when this answers an interrogation from a station under local control.
    responding: bool = field(compare=False, default=False)
    state: RequestState | None = field(compare=False, default=None)
    eligible: Callable[[], bool] | None = field(compare=False, default=None)
    lead_s: float | None = field(compare=False, default=None)
    expires: float | None = field(compare=False, default=None)
    authority: tuple[str, int] | None = field(compare=False, default=None)
    created: float = field(compare=False, default_factory=time.monotonic)
    on_ptt: Callable[[bool], None] | None = field(compare=False, default=None)
    gain: float = field(compare=False, default=1.0)
    admit: Callable[[], None] | None = field(compare=False, default=None)
    owner_poll: Callable[[], None] | None = field(compare=False, default=None)


    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.rate if self.rate else 0.0


class Identity:
    """When the callsign is next owed, and whether it may be deferred.

    This scheduler requests identification at the end of a communication and at
    the configured interval during one. The default leaves a margin below ten
    minutes so the last frame in hand can finish before the next identification.
    """

    def __init__(self, mycall: str, *, interval_s: float = 540.0,
                 mode: str = "cw") -> None:
        self.mycall = mycall
        self.interval_s = interval_s
        self.mode = mode
        self._last = time.monotonic()
        self._in_communication = False

    def note_sent(self) -> None:
        self._last = time.monotonic()

    def begin(self) -> None:
        self._in_communication = True

    def end(self, *, identified=False) -> None:
        """A communication ended, so an identification is owed now.

        The configured schedule includes a callsign at session end, so this
        makes one due rather than merely noting the session closed.
        """
        self._in_communication = False
        self._last = time.monotonic() if identified else time.monotonic() - self.interval_s

    @property
    def due(self) -> bool:
        return (self.mode != "none"
                and time.monotonic() - self._last >= self.interval_s)

    def burst(self, rate: int = 48000) -> np.ndarray:
        return cwid.audio(self.mycall, fs=rate)


class _Flight:
    def __init__(self, request):
        self.request = request
        self.token = None
        self.key_started = None
        self.key_done = threading.Event()
        self.done = threading.Event()
        self.drained = threading.Event()
        self.stop = threading.Event()
        self.finish = threading.Event()
        self.cut = None
        self.reason = ""
        self.error = None
        self.end = None
        self.worker = None
        self.containment_start_failures = 0
        self.takeover = False
        self.emission_start = None
        self.receive_end = None
        self.keyed_s = 0.0
        self.ptt_lock = threading.Lock()
        self.ptt_announced = False


class TxArbiter:
    def __init__(self, audio, rig: Rig, *, identity: Identity | None = None,
                 drive: float = levels.TX_DRIVE, settle_s: float = 0.0,
                 attended=None) -> None:
        self.audio, self.rig, self.identity = audio, rig, identity
        self.drive, self.settle_s = drive, settle_s
        self._admit = threading.Condition(threading.Lock())
        self._owner = None
        self._lease_epoch = 0
        self._inflight = None
        self._flight = None
        self._revocations = {}
        self._closed = False
        self._revoked_at = 0.0
        self._wake = threading.Event()
        self._supervisor = None
        self._errand = None
        self._attended = attended or (lambda session: False)
        self.records = queue.SimpleQueue()
        self.sent = self.dropped_late = self.refused = 0
        self._active_protocols = lambda: ()
        self.disabled = set()
        self.last_sent = {}
        self._history = TransmitHistory()
        self._polling = threading.local()
        self._scan_hold = None
        self._contact_lock = threading.RLock()
        self._contact = None
        self._receive_boundary = (0, int(getattr(audio, "samples", 0)))

    def claim_contact(self, protocol, token=None, *, receive_epoch=None):
        """Reserve acquisition through terminal RF output, without retaining a mutex.

        A new receiver cannot enter while a transmission or retune is underway.
        An existing owner's token remains valid inside its own keyed callbacks.
        """
        if not isinstance(protocol, str) or not protocol or protocol == 'ident':
            raise ValueError('contact requires a protocol')
        with self._contact_lock:
            if receive_epoch is not None and receive_epoch != self._receive_boundary[0]:
                return None
            if token is not None:
                if self._contact != (protocol, token):
                    raise ValueError('contact token is stale or belongs to another owner')
                return token
            if self._contact is not None:
                return None
        if not self._admit.acquire(blocking=False):
            return None
        try:
            with self._contact_lock:
                if (receive_epoch is not None and receive_epoch != self._receive_boundary[0]
                        or self._closed or self._owner is not None
                        or self._contact is not None or self._scan_hold is not None
                        or protocol in self.disabled
                        or set(self.active_protocols()) - {protocol}
                        or self.identity is not None and self.identity.due):
                    return None
                token = object()
                self._contact = (protocol, token)
                return token
        finally:
            self._admit.release()

    def release_contact(self, protocol, token):
        with self._contact_lock:
            if token is None or self._contact != (protocol, token):
                raise ValueError('contact token is stale or belongs to another owner')
            self._contact = None

    @property
    def contact_protocol(self):
        with self._contact_lock:
            return self._contact[0] if self._contact else None

    @property
    def scan_held(self):
        return self._scan_hold is not None

    @property
    def receive_boundary(self):
        """Channel generation and first potentially usable capture sample."""
        with self._contact_lock:
            return self._receive_boundary

    def _retire_receive(self):
        # Called with key/retune ownership. Publish a new immutable boundary
        # before admission can resume; owners reset their decoders themselves.
        with self._contact_lock:
            epoch, floor = self._receive_boundary
            self._receive_boundary = (epoch+1, max(floor, int(getattr(self.audio, 'samples', 0))))

    def acquire_scan_hold(self, protocol, *, eligible):
        """Grant a retune lease only between keyings and with every lane idle.

        The opaque token blocks all transmissions, including identification,
        until its owner releases it. No mutex is retained across host polling.
        Eligibility runs in a nontransmitting scope under the key lock.
        """
        if (not isinstance(protocol, str) or not protocol or protocol == 'ident'
                or not callable(eligible)):
            raise ValueError('scanner requires a protocol and eligibility predicate')
        if not self._admit.acquire(blocking=False):
            return None
        try:
            if (self._closed or self._owner is not None or self._scan_hold is not None
                    or self.contact_protocol is not None or tuple(self.active_protocols())):
                return None
            if not self.run_owner_poll(eligible) or tuple(self.active_protocols()):
                return None
            token = object()
            self._scan_hold = (protocol, token)
            self._retire_receive()
            return token
        finally:
            self._admit.release()

    def release_scan_hold(self, token):
        with self._admit:
            if self._scan_hold is None or self._scan_hold[1] is not token:
                raise ValueError('scanner lease does not belong to this token')
            self._retire_receive()
            self._scan_hold = None

    @property
    def active_protocols(self):
        def active():
            protocols = set(self._active_protocols())
            owner = self._owner
            if isinstance(owner, TxLease):
                protocols.add(owner.protocol)
            return tuple(sorted(protocols))
        return active

    @active_protocols.setter
    def active_protocols(self, provider):
        self._active_protocols = provider

    def authority(self, session_id="station"):
        with self._admit:
            return session_id, self._revocations.setdefault(session_id, (0, 0))[0]

    def _validate(self, request, *, committed=False):
        self._check_scan_hold(request)
        if self._closed:
            raise Refused("arbiter closed")
        if request.created < self._revoked_at:
            raise Refused("stopped: station authority revoked")
        if request.authority is None:
            epoch, since = self._revocations.get("station", (0, 0))
            if request.created < since:
                raise Refused("stopped: authority revoked")
            request.authority = ("station", epoch)
        session, epoch = request.authority
        if epoch != self._revocations.get(session, (0, 0))[0]:
            raise Refused("stopped: authority revoked")
        if committed:
            return
        if request.protocol in self.disabled:
            raise Refused(f"{request.protocol} is disabled on this station")
        if request.expires is not None and time.monotonic() >= request.expires:
            raise Refused("request expired")
        if isinstance(request.state, RequestState) and request.state.value == "cancelled":
            raise Refused("cancelled")

    def _acquire(self, request, *, wait):
        self._not_from_owner_poll()
        with self._admit:
            epoch = self._lease_epoch
            while True:
                self._validate(request)
                if epoch != self._lease_epoch:
                    raise Refused("an intervening lease overtook this request")
                if isinstance(self._owner, TxLease):
                    raise Refused(f"transmitter leased by {self._owner.protocol}")
                if self._owner is None:
                    self._owner = request
                    if isinstance(request, TxLease):
                        self._lease_epoch += 1
                    self._flight = _Flight(request)
                    self._ensure_supervisor()
                    return self._flight
                if not wait:
                    raise Refused("transmitter busy")
                self._admit.wait(0.02)

    def _ensure_supervisor(self):
        if self._supervisor is None:
            self._supervisor = threading.Thread(target=self._supervise,
                                                name="tx-supervisor", daemon=True)
            self._supervisor.start()

    def _release(self, flight):
        with self._admit:
            if self._flight is flight:
                self._owner = self._flight = self._inflight = None
                self._admit.notify_all()
        self._wake.set()

    @contextmanager
    def try_hold(self, *, scan_token=None):
        """Reserve a retune through the same owner as finite and leased TX."""
        self._not_from_owner_poll()
        flight = None
        with self._admit:
            if (not self._closed and self._owner is None
                    and self.contact_protocol is None and not tuple(self.active_protocols())
                    and (self._scan_hold is None or self._scan_hold[1] is scan_token)):
                self._owner = TxRequest(ERRAND, -1, protocol="hold")
                self._flight = flight = _Flight(self._owner)
                self._retire_receive()
        try:
            yield flight is not None
        finally:
            if flight is not None:
                with self._admit:
                    self._retire_receive()
                self._release(flight)

    def transmit_history(self):
        return self._history

    def submit(self, req):
        if req.state is not None or req.eligible is not None or req.lead_s is not None:
            raise ValueError("stateful requests use submit_result")
        flight = self._acquire(req, wait=req.priority == LIVE)
        try:
            return self._send(req, flight)
        finally:
            self._release_when_safe(flight)

    def submit_result(self, req):
        if req.state is None:
            req.state = RequestState()
        flight = None
        try:
            flight = self._acquire(req, wait=True)
            if req.eligible is not None and not req.eligible():
                return TxResult("deferred", req.state.value)
            with self._admit:
                self._validate(req)
            end = self._send(req, flight)
            return TxResult("sent", req.state.value, end=end,
                            key_index=flight.token.card_index, keyed_s=flight.keyed_s,
                            emission_start=flight.emission_start, emission_end=end)
        except Refused as exc:
            return TxResult("refused", req.state.value, reason=str(exc))
        except BurstInvalidated as exc:
            return TxResult("invalidated", req.state.value, reason=str(exc),
                            displacement=getattr(exc, "displacement", 0))
        except Exception as exc:
            return TxResult("error", req.state.value, reason=str(exc))
        finally:
            if flight is not None:
                self._release_when_safe(flight)

    def _not_from_owner_poll(self):
        if getattr(self._polling, 'active', False):
            raise Refused('an owner I/O poll cannot submit a transmission')

    def run_owner_poll(self, action):
        """Service bounded owner I/O, including outside a transmit wait.

        The scope forbids this owner from acquiring the key recursively. It
        does not create a worker, acquire ownership or advance a modem clock.
        """
        previous = getattr(self._polling, 'active', False)
        self._polling.active = True
        try:
            return action()
        finally:
            self._polling.active = previous

    def _release_when_safe(self, flight):
        if flight.key_started is None or flight.drained.is_set():
            self._release(flight)

    def _policy(self, req):
        self._check_scan_hold(req)
        self._check_active_protocols(req)
        if not 0 <= getattr(req, 'gain', 1.0) <= 1:
            raise Refused('host output gain must be finite and within 0..1')
        if getattr(req, 'owner_poll', None) is not None and not callable(req.owner_poll):
            raise Refused('owner I/O poll must be callable')
        if req.emission is None:
            raise Refused(f"{req.protocol}: no emission described")
        session = req.authority[0]
        if (self.identity is not None and self.identity.due and req.protocol != "ident"
                and not self._attended(session)):
            raise Refused(f"{req.protocol}: identification is due")

    def _check_active_protocols(self, req):
        owners = set(self.active_protocols()) - {req.protocol}
        contact = self.contact_protocol
        if contact is not None and contact != req.protocol:
            owners.add(contact)
        if owners and req.protocol != "ident":
            raise Refused(f"{req.protocol}: station link active on {', '.join(sorted(owners))}")

    def _send(self, req, flight):
        self._policy(req)
        if req.at is not None and req.at < self.audio.sample_now():
            self.dropped_late += 1
            raise Refused(f"{req.protocol}: scheduled index has passed")
        lead = max(req.settle_s, req.lead_s or 0)
        duration = req.duration_s + (0.1 if req.lead_s is not None else 0)
        if duration > longest_burst_s(self.rig.max_key_s, lead):
            self.refused += 1
            raise Refused("complete keyed interval exceeds max_key_s. Refused rather than truncated")
        self.audio.alive()
        return self._transmit(req, flight, lead, duration)

    def _cancelled(self, flight):
        return flight.stop.is_set() or (flight.request.state is not None
                                       and flight.request.state.value == "cancelled")

    def _commit(self, req, flight):
        with self._admit:
            self._validate(req)
            if flight.stop.is_set():
                raise BurstInvalidated("stopped")
            if req.state is not None and not req.state.commit():
                raise BurstInvalidated("cancelled")

    def _key(self, req, flight, **kwargs):
        if getattr(req, 'admit', None) is not None:
            req.admit()
        with self._admit:
            self._check_scan_hold(req)
            self._check_active_protocols(req)
            self._validate(req)
            if flight.stop.is_set():
                raise BurstInvalidated("stopped")
            flight.key_started = time.monotonic()
        self._wake.set()
        try:
            token = self.rig.key(req.emission, why=f"{req.protocol} transmit",
                                 responding=req.responding,
                                 attended=self._attended(req.authority[0]),
                                 sample_now=self.audio.sample_now, **kwargs)
            with flight.ptt_lock:
                self._notify_ptt(req, True)
                flight.ptt_announced = True
            with self._admit:
                flight.token = token
                try:
                    # Takeover accepts the already committed over, STOP does not.
                    if not flight.takeover:
                        self._validate(req, committed=True)
                except Refused:
                    self._halt_locked(flight, "stopped")
                if not flight.stop.is_set():
                    self._inflight = (req, token)
        except BaseException as exc:
            with self._admit:
                stopped = flight.stop.is_set()
                self._halt_locked(flight, "key-up failed")
            if not (stopped and isinstance(exc, RigError)):
                raise
        finally:
            flight.key_done.set()
            self._wake.set()
        if flight.stop.is_set():
            self._await_finish(flight)
            raise BurstInvalidated("overtaken" if isinstance(req, TxLease) else "stopped")
        return token

    @staticmethod
    def _notify_ptt(req, on):
        if getattr(req, 'on_ptt', None) is not None:
            try:
                req.on_ptt(on)
            except Exception:
                # Host notification cannot strand the station's physical key.
                logging.getLogger(__name__).exception('PTT host notification failed')

    def _check_scan_hold(self, req):
        if self._scan_hold is not None:
            raise Refused(f'{req.protocol}: scanner lease held by {self._scan_hold[0]}')

    def _wait_dac(self, frame, *, request=None, **kwargs):
        if request is not None and getattr(request, 'owner_poll', None) is not None:
            cancelled = kwargs.get('cancelled')
            def poll():
                request.owner_poll()
                return bool(cancelled is not None and cancelled())
            kwargs['cancelled'] = lambda: self.run_owner_poll(poll)

        wait = getattr(self.audio, "wait_until_dac", None)
        if wait is None:
            # ReplayAudio's expanded transmit surface lands in U14.
            if kwargs.get("cancelled", lambda: False)():
                raise BurstInvalidated("stopped")
            self.audio.wait_until(frame, **kwargs)
        else:
            wait(frame, **kwargs)

    def _transmit(self, req, flight, lead, duration):
        waveform = levels.at_drive(resample.for_card(req.audio, req.rate), self.drive * getattr(req, 'gain', 1.0))
        loop_lat = getattr(self.audio, "loop_lat", 0)
        tx_latency = self.audio.tx_latency_n
        dac_lat = loop_lat - tx_latency
        settle = round(lead * CARD_RATE_HZ)
        deadline = time.monotonic() + lead + duration + 2
        try:
            hold = self.rig.hold() if hasattr(self.rig, "hold") else nullcontext(True)
            with hold as held:
                if not held:
                    raise Refused("CAT hold missed the key deadline")
                if req.lead_s is None:
                    if req.at is None:
                        frame_at = getattr(self.audio, "dac_frame_at", None)
                        want = (math.ceil(frame_at(time.monotonic() + lead)) + dac_lat
                                if frame_at else int(self.audio.sample_now()) + settle)
                    else:
                        want = req.at - tx_latency
                    at = self.audio.arm_burst(waveform, at=want)
                    if req.at is not None and at != want:
                        self.dropped_late += 1
                        raise Refused(f'{req.protocol}: output cannot meet card index {req.at}')
                    flight.emission_start = at + tx_latency
                    flight.receive_end = flight.emission_start + len(waveform)
                    frame = at - dac_lat
                    deadline += max(0, (at - self.audio.sample_now()) / CARD_RATE_HZ)
                    self._wait_dac(frame - settle, request=req, deadline=deadline, healthy=True,
                                   cancelled=lambda: self._cancelled(flight))
                elif req.at is not None:
                    self.audio.wait_until(req.at, deadline=deadline,
                                          cancelled=lambda: self._cancelled(flight))
                self._commit(req, flight)
                token = self._key(req, flight, duration_s=duration, settle_s=lead)
                if req.lead_s is not None:
                    if token.card_index is None or token.card_deadline is None:
                        raise BurstInvalidated("key did not report its card interval")
                    want = token.card_index + settle + tx_latency
                    at = self.audio.reserve_burst(waveform, want, token.card_deadline)
                    flight.emission_start = at
                    flight.receive_end = at + len(waveform)
                    frame = at - loop_lat
            self._wait_dac(frame + len(waveform), request=req, deadline=deadline, healthy=True,
                           cancelled=lambda: self._cancelled(flight))
            flight.finish.set()
            self._wake.set()
            end = self._await_finish(flight)
            with self._admit:
                if flight.stop.is_set():
                    raise BurstInvalidated("stopped")
                if self.rig.retired:
                    raise BurstInvalidated("rig retired during burst")
                self.sent += 1
                self.last_sent[req.protocol] = end
                if self.identity is not None and req.protocol == 'ident':
                    self.identity.note_sent()
            return end
        except BaseException as exc:
            stopped = flight.stop.is_set()
            if flight.key_started is not None:
                with self._admit:
                    self._halt_locked(flight, "stopped" if stopped else str(exc))
                self._await_finish(flight)
            else:
                self.audio.cancel_burst()
            if isinstance(exc, RigError):
                self.refused += 1
                raise Refused(str(exc)) from exc
            if stopped and isinstance(exc, BurstInvalidated):
                raise BurstInvalidated("stopped") from exc
            raise

    def _halt_locked(self, flight, why):
        if flight.done.is_set() or flight.stop.is_set():
            return
        flight.cut = int(self.audio.sample_now())
        flight.reason = why
        flight.stop.set()
        if isinstance(flight.request, TxLease):
            flight.request.state_name = "halted"
        self.records.put(dict(protocol=flight.request.protocol,
                              authority=flight.request.authority, outcome=why,
                              end=flight.cut, cut=flight.cut,
                              key_index=getattr(flight.token, "card_index", None)))
        self._wake.set()

    def _revoke(self, session):
        epoch = self._revocations.get(session, (0, 0))[0]
        self._revocations[session] = epoch + 1, time.monotonic()
        self._admit.notify_all()

    def stop(self, session_id="station", why="stopped"):
        if session_id is None:
            self.revoke()
            return
        with self._admit:
            self._revoke(session_id)
            flight = self._flight
            if flight is not None and (session_id is None or
                    flight.request.authority[0] == session_id):
                self._halt_locked(flight, why)
        self._wake.set()

    def revoke(self):
        with self._admit:
            self._revoked_at = time.monotonic()
            for session in {"station", *self._revocations}:
                self._revoke(session)
            if self._flight is not None:
                self._halt_locked(self._flight, "stopped")

    def takeover(self, session_id):
        with self._admit:
            self._revoke(session_id)
            flight = self._flight
            if flight is not None and flight.request.authority[0] == session_id:
                flight.takeover = True
                if flight.key_started is None:
                    self._halt_locked(flight, "superseded")
        self._wake.set()

    def _supervise(self):
        while True:
            self._wake.wait(0.02)
            self._wake.clear()
            flight = self._flight
            if flight is None:
                with self._admit:
                    if self._flight is None:
                        self._supervisor = None
                        return
                continue
            worker_exited = False
            try:
                owner = flight.request
                if flight.done.is_set():
                    continue
                if flight.worker is not None and not flight.worker.is_alive():
                    with self._admit:
                        if (self._flight is not flight or flight.done.is_set()
                                or flight.drained.is_set()):
                            continue
                    # It may have left an unkey thread running. Never replace
                    # an exited containment worker; escalate on this pass.
                    worker_exited = True
                    raise flight.error or RuntimeError("transmit containment worker exited")
                why = None
                if not flight.stop.is_set():
                    if (flight.key_started is not None and not flight.key_done.is_set()
                            and time.monotonic() - flight.key_started >= 1):
                        why = "key-up timeout"
                    elif isinstance(owner, TxLease) and flight.key_done.is_set():
                        why = owner._fault()
                        if (flight.takeover and owner.state_name != "keying"
                                and not owner._taking_over):
                            owner._taking_over = True
                            threading.Thread(target=owner._drain_takeover,
                                             name="tx-takeover", daemon=True).start()
                if why:
                    with self._admit:
                        self._revoke(owner.authority[0])
                        self._halt_locked(flight, why)
                if (flight.stop.is_set() or flight.finish.is_set()) and flight.worker is None:
                    try:
                        worker = threading.Thread(target=self._contain_guarded, args=(flight,),
                                                  name="tx-containment", daemon=True)
                        worker.start()
                    except Exception:
                        flight.containment_start_failures += 1
                        raise
                    flight.worker = worker
            except Exception as exc:
                if not self._recover_supervisor(flight, exc, worker_exited=worker_exited):
                    return

    def _recover_supervisor(self, flight, exc, *, worker_exited=False):
        # One retry for containment starts, independently of the inspection
        # exception that initiated recovery. An exited containment worker or
        # broken halt escalates immediately; neither may kill supervision.
        halt_failed = False
        with self._admit:
            flight.error = exc
            self._revoke(flight.request.authority[0])
            try:
                self._halt_locked(flight, str(exc))
            except Exception:
                halt_failed = True
                flight.reason = str(exc)
                flight.stop.set()
                if isinstance(flight.request, TxLease):
                    flight.request.state_name = "halted"
        _log.exception("transmit supervisor failed; containing current flight")
        if flight.containment_start_failures < 2 and not worker_exited and not halt_failed:
            return True

        # No new thread can be relied on here. Enter the rig's bounded panic
        # ladder directly, then hand off to fatal even if panic itself raises
        # (for example because its own Thread.start fails). Without fatal, or
        # if a test fatal returns, fail closed and retain undrained ownership.
        why = "transmit supervisor recovery failed"
        with self._admit:
            self._closed = True
            flight.error = LeaseTimeout(why)
            self._admit.notify_all()
        try:
            self.rig.panic_unkey(why)
        except BaseException:
            _log.exception("transmit supervisor panic failed")
        finally:
            try:
                fatal = getattr(self.rig, "fatal", None)
                if fatal is not None:
                    fatal(why)
            finally:
                flight.done.set()
        return False

    def _contain_guarded(self, flight):
        try:
            self._contain(flight)
        except Exception as exc:
            # The supervisor detects this worker's exit and applies the same
            # bounded recovery policy as a failure in the guard itself.
            with self._admit:
                flight.error = exc
            self._wake.set()

    def _contain(self, flight):
        returned = threading.Event()
        unkeyed = threading.Event()
        errors = []

        def unkey():
            self.rig.unkey(flight.token)
            with flight.ptt_lock:
                if flight.ptt_announced:
                    self._notify_ptt(flight.request, False)
                    flight.ptt_announced = False
            unkeyed.set()
            flight.keyed_s = time.monotonic() - flight.key_started

        def drop():
            try:
                lock = (flight.request._stream_lock if isinstance(flight.request, TxLease)
                        else nullcontext())
                if flight.stop.is_set():
                    with lock:
                        self.audio.cancel_burst()
                if flight.key_started is not None:
                    pending = not flight.key_done.is_set()
                    unkey()
                    if pending:
                        flight.key_done.wait()
                        unkey()
                if flight.stop.is_set():
                    with lock:
                        self.audio.cancel_burst()
                    self.audio.finish_burst()
                    flight.end = flight.cut
                else:
                    end = self.audio.finish_burst()
                    if flight.end is None:
                        flight.end = end
            except BaseException as exc:
                errors.append(exc)
            finally:
                returned.set()

        worker = threading.Thread(target=drop, name="tx-unkey", daemon=True)
        worker.start()
        returned.wait(1.0)
        confirmed = unkeyed.is_set() and getattr(self.rig, "line_state", None) == "low"
        pending_key = flight.key_started is not None and not flight.key_done.is_set()
        if not returned.is_set() or (flight.key_started is not None and
                                               (not confirmed or pending_key)):
            try:
                self.rig.panic_unkey(flight.reason or "unkey unconfirmed or blocked")
                flight.error = errors[0] if errors else BurstInvalidated(
                    "rig retired during transmit containment")
            except BaseException as exc:
                flight.error = exc
            if not returned.is_set() or pending_key:
                flight.error = LeaseTimeout("transmit containment did not drain")
                fatal = getattr(self.rig, "fatal", None)
                if fatal is not None:
                    fatal(str(flight.error))
                else:
                    with self._admit:
                        self._halt_locked(flight, str(flight.error))
                    # Complete the caller with LeaseTimeout, but retain ownership
                    # until the old worker can no longer touch PTT or audio.
                    flight.done.set()
                # Ownership cannot pass a worker or ioctl still able to touch PTT.
                returned.wait()
        if errors and flight.error is None:
            flight.error = errors[0]
        if flight.error is not None:
            with self._admit:
                self._halt_locked(flight, str(flight.error))
        if flight.key_started is not None:
            self._history = TransmitHistory(self._history.sequence + 1,
                flight.end or int(self.audio.sample_now()), flight.receive_end or 0)
        if not flight.stop.is_set():
            self.records.put(dict(protocol=flight.request.protocol,
                                  authority=flight.request.authority,
                                  outcome="released", end=flight.end,
                                  key_index=getattr(flight.token, "card_index", None),
                                  keyed_s=flight.keyed_s, emission_start=flight.emission_start,
                                  emission_end=flight.end,
                                  cut=None))
        if isinstance(flight.request, TxLease):
            flight.request.state_name = "halted" if flight.stop.is_set() else "released"
        flight.drained.set()
        flight.done.set()
        if isinstance(flight.request, TxLease) or flight.error is not None:
            self._release(flight)

    def _await_finish(self, flight):
        while not flight.done.wait(0.02):
            if flight.error is not None:
                raise flight.error
        if flight.error is not None:
            raise flight.error
        return flight.end

    def lease(self, *, protocol, emission, settle_s=0.0, idle="fault",
              authority=None, capacity=CARD_RATE_HZ):
        return TxLease(self, protocol, emission, settle_s, idle, authority, capacity)

    def identify(self, emission):
        self._not_from_owner_poll()
        if self.identity is None:
            raise Refused("no identity configured")
        if self._attended(None):
            raise Refused("attended session holds the rig")
        request = TxRequest(ERRAND, -1, protocol="ident", audio=self.identity.burst(),
                            emission=emission, settle_s=self.settle_s)
        end = self.submit(request)
        self.identity.note_sent()
        return end

    def identify_if_free(self, emission):
        self._not_from_owner_poll()
        if self.identity is None or not self.identity.due or self._attended(None):
            return False
        with self._admit:
            if self._owner is not None or self._closed or (
                    self._errand is not None and self._errand.is_alive()):
                return False
            request = TxRequest(ERRAND, -1, protocol="ident",
                                emission=emission, settle_s=self.settle_s)
            def identify():
                try:
                    request.audio = self.identity.burst()
                    self.submit(request)
                    self.identity.note_sent()
                except (Refused, BurstInvalidated, RigError):
                    pass
            self._errand = threading.Thread(target=identify, name="tx-ident", daemon=True)
            self._errand.start()
            return True

    def disable(self, protocol, why):
        with self._admit:
            self.disabled.add(protocol)
            self._admit.notify_all()

    def close(self, timeout=5.0):
        self.revoke()
        with self._admit:
            self._closed = True
            self._admit.notify_all()
        deadline = time.monotonic() + timeout
        while self._owner is not None:
            if time.monotonic() >= deadline:
                raise LeaseTimeout("transmitter owner did not drain")
            time.sleep(0.01)
        for thread in (self._supervisor, self._errand):
            if thread is not None:
                thread.join(max(0, deadline - time.monotonic()))
                if thread.is_alive():
                    raise LeaseTimeout("arbiter thread did not stop")


class TxLease:
    def __init__(self, arbiter, protocol, emission, settle_s, idle, authority, capacity):
        if idle not in ("fault", "silence") or capacity <= 0 or settle_s < 0:
            raise ValueError("invalid lease policy")
        self.arbiter = arbiter
        self.protocol, self.emission = protocol, emission
        self.settle_s, self.idle, self.capacity = settle_s, idle, capacity
        self.authority = authority
        self.created = time.monotonic()
        self.expires = None
        self.responding = False
        self.state_name = "acquiring"
        self.ceiling_s = None
        self.heartbeat_s = 1.0
        self.key_index = None
        self._flight = None
        self._raw = None
        self._stream_lock = threading.Lock()
        self._completion_lock = threading.RLock()
        self._touched = self.created
        self._progress_at = self.created
        self._progress = None
        self._starved_since = None
        self._taking_over = False
        self._proxy = _LeaseStream(self)

    @property
    def state(self):
        return self.state_name

    def __enter__(self):
        if self._flight is not None:
            raise Refused("lease already entered")
        arb = self.arbiter
        self._flight = flight = arb._acquire(self, wait=True)
        try:
            arb._policy(self)
            attended = arb._attended(self.authority[0])
            self.ceiling_s = None if attended else arb.rig.max_key_s
            # The 5 s attended client acknowledgement belongs to the control
            # session; every lease holder/pump must still touch within 1 s.
            self.heartbeat_s = 1.0
            self.state_name = "keying"
            token = arb._key(self, flight, open=True)
            self.key_index = token.card_index
            if self.key_index is None:
                raise BurstInvalidated("key did not report its card index")
            self.touch()
            self.state_name = "filling"
            return self
        except BaseException:
            if flight.key_started is not None:
                self.halt("key-up failed")
                arb._await_finish(flight)
            arb._release_when_safe(flight)
            raise

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.halt(str(value))
            self.arbiter._await_finish(self._flight)
        elif not self._flight.done.is_set():
            self.end()

    def touch(self):
        self._touched = time.monotonic()

    @property
    def keyed_s(self):
        flight = self._flight
        return 0.0 if flight is None or flight.key_started is None else (
            time.monotonic() - flight.key_started)

    def _check(self, *, writing=False):
        flight = self._flight
        if flight is None:
            raise Refused("lease has not been entered")
        if flight.stop.is_set() or flight.done.is_set():
            raise BurstInvalidated(flight.reason or "lease released")
        if writing and flight.takeover:
            raise Refused("superseded")
        self.touch()

    def stream(self):
        self._check(writing=True)
        with self._stream_lock:
            self._check(writing=True)
            if self._raw is None:
                self._raw = self.arbiter.audio.arm_stream(
                    at=self.key_index + round(self.settle_s * CARD_RATE_HZ),
                    capacity=self.capacity)
                self._raw.idle = self.idle
        return self._proxy

    def _deadline(self):
        audio = self.arbiter.audio
        raw = self._raw
        at = raw.at if raw is not None else self.key_index + round(self.settle_s * CARD_RATE_HZ)
        queued = raw.queued if raw is not None else 0
        latency = getattr(getattr(audio, "_stream", None), "latency", 0)
        if isinstance(latency, tuple):
            latency = latency[1]
        return (time.monotonic() + max(0, at - audio.sample_now() + queued) / CARD_RATE_HZ
                + getattr(audio, "loop_lat", 0) / CARD_RATE_HZ + latency + 1)

    def _waiting(self):
        self._check()
        self.touch()
        raw = self._raw
        progress = raw.progress if raw is not None else None
        if progress != self._progress:
            self._progress, self._progress_at = progress, time.monotonic()
        self.arbiter.audio.check_callbacks()

    def complete(self, written):
        with self._completion_lock:
            self._check()
            raw = self._raw
            if written < 0 or written > (raw.written if raw is not None else 0):
                raise ValueError("completion position was not written")
            draining = self.state_name == "draining"
            if not draining:
                self.state_name = "waiting_card"
            self._progress_at = time.monotonic()
            deadline = self._deadline()
            try:
                while True:
                    self._waiting()
                    endpoint = (self.key_index + round(self.settle_s * CARD_RATE_HZ)
                                if written == 0 else raw.progress.card_end(written))
                    if endpoint is not None and self.arbiter.audio.sample_now() >= endpoint:
                        return endpoint
                    if time.monotonic() >= deadline:
                        raise BurstInvalidated("completion deadline expired")
                    time.sleep(0.002)
            except BaseException as exc:
                self.halt(str(exc))
                raise
            finally:
                if self.state_name == "waiting_card":
                    self.state_name = "filling"

    def end(self):
        with self._completion_lock:
            if self._flight.done.is_set():
                return self.arbiter._await_finish(self._flight)
            self._check()
            self.state_name = "draining"
            with self._stream_lock:
                if self._raw is not None:
                    self._raw.close()
            try:
                endpoint = self.complete(self._raw.written if self._raw is not None else 0)
                self._flight.end = endpoint
                self._flight.finish.set()
                self.arbiter._wake.set()
                self.arbiter._await_finish(self._flight)
                return endpoint
            except BaseException as exc:
                self.halt(str(exc))
                self.arbiter._await_finish(self._flight)
                raise

    def halt(self, why="stopped"):
        arb = self.arbiter
        with arb._admit:
            if (self._flight is not None and not self._flight.stop.is_set()
                    and not self._flight.done.is_set()):
                arb._revoke(self.authority[0])
                arb._halt_locked(self._flight, why)

    def _drain_takeover(self):
        try:
            self.end()
        except BaseException as exc:
            self.halt(str(exc))

    def _fault(self):
        if self._flight.stop.is_set() or self._flight.finish.is_set():
            return None
        now = time.monotonic()
        if self.ceiling_s is not None and self.keyed_s >= self.ceiling_s:
            return "ceiling"
        if now - self._touched >= self.heartbeat_s:
            return "holder heartbeat expired"
        try:
            self.arbiter.audio.check_callbacks()
        except Exception as exc:
            return str(exc)
        raw = self._raw
        if raw is not None:
            progress = raw.progress
            if progress != self._progress:
                self._progress, self._progress_at = progress, now
            if (self.state_name in ("waiting_card", "draining") and raw.queued
                    and now - self._progress_at >= 1):
                return "card progress stalled"
            if raw.starved_at is not None and self.idle == "fault":
                if self._starved_since is None:
                    self._starved_since = now
                elif now - self._starved_since >= 1:
                    return "stream starved"
            else:
                self._starved_since = None
        return None


class _LeaseStream:
    def __init__(self, lease):
        self._lease = lease

    def write(self, samples):
        lease = self._lease
        samples = np.asarray(samples, np.float32)
        with lease.arbiter._admit:
            lease._check(writing=True)
            lease.arbiter._validate(lease)
            lease._raw.write(samples)

    def write_blocking(self, samples):
        lease = self._lease
        samples = np.asarray(samples, np.float32)
        if len(samples) > lease.capacity:
            raise ValueError("write exceeds stream capacity")
        lease._check(writing=True)
        lease.state_name = "waiting_card"
        lease._progress_at = time.monotonic()
        deadline = lease._deadline()
        try:
            while lease._raw.queued + len(samples) > lease.capacity:
                lease._check(writing=True)
                lease._waiting()
                if time.monotonic() >= deadline:
                    raise BurstInvalidated("write deadline expired")
                time.sleep(0.002)
            self.write(samples)
        except BurstInvalidated as exc:
            lease.halt(str(exc))
            raise
        finally:
            if lease.state_name == "waiting_card":
                lease.state_name = "filling"

    def close(self):
        lease = self._lease
        with lease.arbiter._admit:
            lease._check(writing=True)
            lease._raw.close()

    def __getattr__(self, name):
        if name not in {"at", "capacity", "written", "queued", "emitted", "progress",
                        "end", "closed", "drained", "starved_at"}:
            raise AttributeError(name)
        self._lease._check()
        return getattr(self._lease._raw, name)
