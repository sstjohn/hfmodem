# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One keyer, one owner, and no claim service.

There is one `TxArbiter` in one process, and it is the only thing that calls
`StationAudio.arm_burst()` or `Rig.key()`. That is the whole answer to contention:
there is nothing to claim, because there is one of it. Four processes needed a
lock, an arbitration protocol, and operator discipline — and discipline is what
failed, when one modem transmitted while another recorded and voided eight on-air
sessions reported as a silent band.

The policy is short on purpose. No weights, no fairness knobs, no configuration.

1. One request in flight. Priority 0 (a live ARQ turn) beats priority 1 (a beacon
   or an identification); ties are FIFO.
2. **Half-duplex is enforced here, once.** `StationAudio.finish_burst()` advances
   every lane past our own carrier, which is the one place that knows the sample
   index. That replaces besra's `muted` flag and shrike's per-session flush with
   one mechanism.
3. A request whose scheduled index has already passed is **dropped, not sent
   late**. A cycle that overran is absorbed rather than displacing every cycle
   after it — shrike's `take_until` discipline, applied to transmit.
4. Over-long audio is **refused before keying**, never truncated. A clipped
   transmission is a worse artefact than a missing one.
5. **Identification is an interlock, not a request.** When one is due, priority-0
   traffic is refused until it has gone out. The station goes quiet rather than
   transmitting unidentified — which is the opposite of what a priority scheme
   would do to it, since a busy station is exactly one that never gets round to
   a low-priority errand.

There is no channel-busy gate. One shipped here for a while and could never
fire: the window of receive audio it judged was a stub returning no samples, so
the gate read every channel as clear behind three comments promising otherwise.
Listening before transmitting is the operator's duty, and `core.busy` serves it
through the operator's tools — a check the station cannot perform, described as
one it does, is worse than none.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from hfmodem.core import cwid, levels, resample
from hfmodem.core.audio import BurstInvalidated
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.regulatory import Emission
from hfmodem.core.rig import Rig, RigError

#: Priorities. Two is enough; a third would need a reason.
LIVE = 0
ERRAND = 1


class Refused(Exception):
    """The burst was not transmitted, and the message says why.

    `submit` can also raise `BurstInvalidated` — the burst was keyed and what went
    out is not what was composed. It is deliberately not a `Refused`, because the
    two need different responses: one is retry, the other is find out why.
    """


@dataclass(order=True)
class TxRequest:
    priority: int
    seq: int = field(compare=True)
    protocol: str = field(compare=False, default="")
    audio: np.ndarray = field(compare=False, default_factory=lambda: np.zeros(0))
    rate: int = field(compare=False, default=48000)
    emission: Emission | None = field(compare=False, default=None)
    at: int | None = field(compare=False, default=None)
    settle_s: float = field(compare=False, default=0.0)
    #: True when this answers an interrogation from a station under local control.
    responding: bool = field(compare=False, default=False)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.rate if self.rate else 0.0


class Identity:
    """When the callsign is next owed, and whether it may be deferred.

    §97.119(a) wants identification at the end of a communication and at least
    every ten minutes during one. A margin below the limit exists so the last
    frame in hand can finish rather than being cut off by the deadline.
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

    def end(self) -> None:
        """A communication ended, so an identification is owed now.

        §97.119(a) wants the callsign at the end of a communication as well as
        every ten minutes during one, so this makes one due rather than merely
        noting the session closed.
        """
        self._in_communication = False
        self._last = time.monotonic() - self.interval_s

    @property
    def due(self) -> bool:
        return (self.mode != "none"
                and time.monotonic() - self._last >= self.interval_s)

    def burst(self, rate: int = 48000) -> np.ndarray:
        return cwid.audio(self.mycall, fs=rate)


class TxArbiter:
    """The only caller of `Rig.key()` and `StationAudio.arm_burst()`."""

    def __init__(self, audio, rig: Rig, *, identity: Identity | None = None,
                 drive: float = levels.TX_DRIVE, settle_s: float = 0.0) -> None:
        self.audio = audio
        self.rig = rig
        self.identity = identity
        #: Carrier the rig is given before audio starts. Held on the station
        #: rather than passed per request, because the identification is the one
        #: transmission with no protocol behind it to remember — and it forgot,
        #: which put the leading dit of the callsign out ahead of the key.
        self.settle_s = settle_s
        #: Peak every burst leaves at — `core.levels.TX_DRIVE`, or the station
        #: file's override of it.
        self.drive = drive
        self._lock = threading.Lock()
        self._seq = 0
        self.sent = 0
        self.dropped_late = 0
        self.refused = 0
        self.disabled: set[str] = set()

    # -- submission --------------------------------------------------------

    def submit(self, req: TxRequest) -> int:
        """Transmit one burst, or raise `Refused` saying why not.

        Synchronous: the caller is a protocol's own thread and its next move
        depends on whether this went out. A queue here would let a protocol
        believe it had transmitted while the request sat behind another.

        An errand does not wait. A plain blocking lock made priority a fiction —
        a live ARQ answer arriving during a beacon blocked for the beacon's whole
        duration and was then dropped as late, inside a 0.29 s answer window. So an
        errand takes the lock only if it is free, and a live turn is the only thing
        allowed to wait.
        """
        blocking = req.priority == LIVE
        if not self._lock.acquire(blocking=blocking):
            raise Refused(
                f"{req.protocol}: the transmitter is busy and this is not a live "
                "turn. An errand that waits is an answer that arrives late.")
        try:
            return self._send(req)
        finally:
            self._lock.release()

    def _send(self, req: TxRequest) -> int:
        if req.protocol in self.disabled:
            raise Refused(f"{req.protocol} is disabled on this station")

        if req.at is not None and req.at < self.audio.sample_now():
            # Absorbed rather than allowed to displace every later cycle.
            self.dropped_late += 1
            raise Refused(
                f"{req.protocol}: scheduled for card index {req.at}, which has "
                f"passed. Dropped rather than sent late.")

        if req.duration_s > self.rig.max_key_s:
            self.refused += 1
            raise Refused(
                f"{req.protocol}: {req.duration_s:.1f} s of audio exceeds "
                f"max_key_s {self.rig.max_key_s:.1f}. Refused rather than "
                "truncated — a clipped transmission is the worse artefact.")

        if self.identity is not None and self.identity.due and req.priority == LIVE:
            # The station goes quiet rather than transmitting unidentified.
            raise Refused(
                f"{req.protocol}: identification is due. The station will not "
                "carry traffic until the callsign has gone out.")

        return self._transmit(req)

    def _transmit(self, req: TxRequest) -> int:
        if req.emission is None:
            raise Refused(
                f"{req.protocol}: no emission described. The rig cannot check an "
                "unstated one, and will not key without a checked one.")
        alive = getattr(self.audio, "alive", None)
        if alive is not None:
            alive()             # a dead stream and a quiet band look identical

        # arm_burst clamps to the earliest index it can actually meet and returns
        # it. The lead is the ADC->DAC offset plus a few blocks of notice — not
        # `holdback`, which is a reader's constant and measured 10986 samples on
        # this machine against an offset of 1152. Recording where we asked rather
        # than where the carrier went is recording nothing.
        # The card plays at one rate in one scale, and a protocol composes at its
        # own. ARDOP's 12 kHz int16 armed as-is would go out a quarter as long and
        # two octaves high — 1500 Hz landing at 6000, outside the passband the
        # emission gate just checked and approved. The gate cannot see this: it is
        # handed the waveform's declared band, not the samples.
        audio = levels.at_drive(resample.for_card(req.audio, req.rate), self.drive)
        settle_n = int(req.settle_s * CARD_RATE_HZ)
        # A burst that names no instant still needs its settle to be real. Armed
        # at zero it lands at `arm_burst`'s minimum notice, which on this card is
        # the converter offset plus three blocks — near the configured 40 ms by
        # arithmetic coincidence, and not by anything that stays true if the
        # offset or the blocksize changes. Asking for the settle explicitly makes
        # the configured number mean what it says.
        want = req.at if req.at else int(self.audio.sample_now()) + settle_n
        at = self.audio.arm_burst(audio, at=want)
        try:
            # Wait for the burst's own start before keying, so the lead is not
            # spent as dead carrier, and so a far-future `at` cannot outrun the
            # rig's per-burst watchdog.
            self.audio.wait_until(at - settle_n)
            token = self.rig.key(req.emission, why=f"{req.protocol} burst",
                                 duration_s=req.duration_s, settle_s=req.settle_s,
                                 responding=req.responding)
        except RigError as exc:
            # Discarded, not finished: a burst the rig refused never went out, and
            # `finish_burst` is what records one as sent.
            self.audio.cancel_burst()
            self.refused += 1
            raise Refused(f"{req.protocol}: {exc}") from None
        try:
            self.audio.wait_until(at + len(audio))
        finally:
            # finish_burst must run even if unkey panics, or the callback keeps
            # feeding audio to a transmitter we have just declared stuck.
            try:
                self.rig.unkey(token)
            finally:
                end = self.audio.finish_burst()
        if self.rig.retired:
            raise BurstInvalidated(
                f"{req.protocol}: the rig retired mid-burst "
                f"({self.rig.retired_why}) — what went out is not what was composed.")
        self.sent += 1
        if self.identity is not None and req.priority != LIVE:
            self.identity.note_sent()
        return end

    # -- identification ----------------------------------------------------

    def identify(self, emission: Emission) -> int:
        """Send the callsign. Never droppable."""
        if self.identity is None:
            raise Refused("no identity configured")
        audio = self.identity.burst()
        with self._lock:
            end = self._transmit(TxRequest(
                priority=ERRAND, seq=-1, protocol="ident", audio=audio,
                emission=emission, settle_s=self.settle_s))
        self.identity.note_sent()
        return end

    # -- containment -------------------------------------------------------

    def disable(self, protocol: str, why: str) -> None:
        """Stop one protocol without taking the station down.

        The property four processes gave for free and one process has to build.
        """
        self.disabled.add(protocol)
