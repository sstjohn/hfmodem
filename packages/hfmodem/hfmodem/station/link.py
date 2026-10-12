# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Binding a protocol's ARQ core to the station's radio.

Each modem already has a sample-level seam — somewhere it takes received audio and
somewhere it hands back audio to transmit. Before the merge each also had its own
sound card, its own PTT, and its own idea of when it was allowed to key. This is
the adapter that replaces all of that with the station's one lane and one arbiter,
and it is deliberately thin: **a protocol's timing is its own business, and the
adapter's only job is to be the thing that carries it to the radio.**

Two rules the adapter enforces on every protocol, because they are properties of
the station rather than of any waveform:

  * **A protocol never keys.** It submits, and the arbiter decides. That is what
    makes contention impossible rather than unlikely — four modems each holding
    their own PTT is the arrangement that voided eight on-air sessions.
  * **A protocol never sees the card.** It receives at its own rate and hands back
    audio at its own rate; the lane and the arbiter convert at the edges. ARDOP's
    12 kHz is normative to ARDOP and resampling it would cost the byte-exact
    cross-decode that is besra's strongest evidence.

Every burst must describe itself. `Rig.key()` takes an `Emission`, so a protocol
that cannot say what it is about to put on the air cannot put anything on the air —
which is the point.
"""
from __future__ import annotations

import sys
import logging
import threading
import time
from dataclasses import dataclass
from collections import deque
from contextlib import nullcontext

from hfmodem.bateleur.spec import FS as BATELEUR_RATE
from hfmodem.bateleur.ale.fsk import FS as ALE_RATE
from hfmodem.bateleur.selcall.fsk import FS as SELCALL_RATE

import numpy as np

from hfhost.config import ConfigError
from hfmodem.core import band
from hfmodem.core.audio import RollingLane, StreamLane
from hfmodem.core.band import amateur_hf
from hfmodem.core.busy import POUNCE_WINDOW_S, is_busy
from hfmodem.core.config_validation import check_linking_protection
from hfmodem.core.occupied import keyed_hz, occupied_hz
from hfmodem.core.operations import LeaseTimeout
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.regulatory import Emission
from hfmodem.core.rig import RigError
from hfmodem.core.tx_events import TxEvents
from hfmodem.station.arbiter import LIVE, ERRAND, Refused, RequestState, TxRequest


@dataclass(frozen=True, slots=True)
class Waveform:
    """What a protocol occupies, in the terms the emission-profile check asks about.

    The passband is looked up rather than restated here. It used to be four pairs
    of literals beside four class definitions, and three of the four were inside
    the emission they described — pactor by 43 Hz at the bottom, sabir by 222 at
    the top, VARA BW2300 by 184 at the bottom, all measured at -26 dB against the
    modulators themselves. Every one of those is a segment edge the gate would
    have reported us safely inside while the emission straddled it.

    `core.occupied` is the one place that answers this, so the band the station
    senses before it keys and the band the gate judges when it keys cannot differ.
    A protocol names no bandwidth here, which is what makes the answer the widest
    burst it can reach rather than the narrowest it opens with: the daemon holds
    no protocol to a ceiling, so no protocol may claim one. For bateleur, ``bw`` selects the occupied row
    (``docs/CONFIG.md``).
    """

    name: str
    rate: int
    bw: str = ""

    def emission(self, dial_hz: int, **kw) -> Emission:
        lo, hi = keyed_hz(self.name, self.bw)
        return Emission(dial_hz, lo, hi, technique=self.name, **kw)


class ProtocolLink:
    """One protocol on the station's radio.

    Subclasses supply `decode` and `on_frame`; everything else — the lane, the
    request, the refusal handling — is the same for all four, because the parts
    that differ are the waveform and the ARQ, and neither of those belongs here.
    """

    waveform: Waveform
    active = False

    @classmethod
    def validate_options(cls, options):
        if not isinstance(options, dict) or options:
            from hfhost.config import ConfigError

            raise ConfigError(f"{cls.__name__}: unsupported protocol options")

    def __init__(self, station, *, depth: int = 4096) -> None:
        self.station = station
        self.lane = self.make_lane(depth)
        self.refusals = deque(maxlen=256)
        self.sent = 0

    def make_lane(self, depth: int):
        """How this protocol wants to be shown the stream.

        Overlapping windows suit a decoder that is handed audio and asked whether
        there is a frame in it, which is three of the four. A protocol whose
        receiver carries state across chunks wants the stream itself and says so
        here — the lane cannot infer it, and getting it wrong is invisible from
        inside the decoder.
        """
        return RollingLane(self.waveform.rate, decode=self.decode,
                           on_frame=self.on_frame, depth=depth)

    # -- receive -----------------------------------------------------------

    def decode(self, samples: np.ndarray):
        raise NotImplementedError

    def on_frame(self, frame) -> None:
        raise NotImplementedError

    def poll(self):
        return self.lane.poll()

    # -- transmit ----------------------------------------------------------

    def transmit(self, audio: np.ndarray, *, why: str = "", at: int | None = None,
                 responding: bool = False, power_w: float | None = None) -> int | None:
        """Ask the station to put `audio` on the air. Returns the card index it
        ended at, or None if it was refused.

        The refusal reason is retained for the protocol and its host events.
        """
        arb = self.station.arbiter
        if arb is None:
            self.refusals.append("no arbiter: this station cannot transmit")
            return None
        try:
            req = TxRequest(
                priority=LIVE if responding or at is not None else ERRAND, seq=0, protocol=self.waveform.name,
                audio=np.asarray(audio), rate=self.waveform.rate,
                emission=self.waveform.emission(
                    self.station.rig.dial(),
                    power_w=power_w if power_w is not None else self.station.cfg.rig.power_w),
                at=at, responding=responding,
                settle_s=self.station.cfg.rig.ptt.settle_s)
            end = arb.submit(req)
        except (Refused, RigError) as exc:
            self.refusals.append(str(exc))
            return None
        self.sent += 1
        return end


class BesraLink(ProtocolLink):
    """ARDOP.

    Its core already separates decoding from the session: `receive_frames` takes
    frames a rolling decoder found, precisely because overlapping windows would
    otherwise deliver the same frame twice — and a replayed DATAACK clears a data
    frame the peer never acknowledged. So the lane decodes and the frames go
    straight in.
    """

    #: The 12 kHz core rate is normative to the protocol and is what the ardopcf
    #: cross-decode holds at.
    waveform = Waveform("ardop", 12000)

    def __init__(self, station, modem, **kw) -> None:
        self.modem = modem
        super().__init__(station, **kw)

    def decode(self, samples: np.ndarray):
        return self.modem._demod.decode(np.asarray(samples, "<i2"))

    def on_frame(self, frame) -> None:
        self.modem.receive_frames([frame])


class ShrikeLink(ProtocolLink):
    """PACTOR-1/2/3.

    The one protocol here with a raster: its cycle is 1.25 s of *sample indices*,
    which is why the station keeps a `CycleLane` at all. This adapter uses the
    rolling lane for monitoring; a cycle-locked session wants `CycleLane` and the
    arbiter's `at` parameter, so that the burst lands on the grid rather than
    wherever the scheduler got to.

    `rxfront.decode_events` reports each event once and derives the protocol from
    the decoder rather than from the event kind — a shared kind lies, and one did:
    a PACTOR-1 link setup announced itself as PACTOR-3 on an operator's screen.
    """

    waveform = Waveform("pactor", 48000)

    def __init__(self, station, *, on_event=None, **kw) -> None:
        self._on_event = on_event
        super().__init__(station, **kw)

    def decode(self, samples: np.ndarray):
        from hfmodem.shrike import rxfront
        return list(rxfront.decode_events(np.asarray(samples, float)))

    def on_frame(self, frame) -> None:
        if self._on_event:
            self._on_event(frame)


class SabirLink(ProtocolLink):
    """The clean-sheet protocol, and the only one whose ARQ stack this station
    also *runs* rather than merely feeds.

    The other three adapters decode into a session that some other code drives.
    sabir's session was written against a simulated air — two endpoints, one
    thread, virtual time — so binding it here means supplying that same port over
    a real radio, which is `station.air.StationAir` and is the whole of M5.

    That is why this one takes the stream rather than windows of it: the segmenter
    that decides where a burst starts carries an adaptive floor, and overlapping
    windows would re-teach it the same audio on every poll.
    """

    waveform = Waveform("sabir", 48000)

    def __init__(self, station, modem=None, **kw) -> None:
        self.modem = modem
        self.air = None
        super().__init__(station, **kw)

    @property
    def active(self):
        end = self.air._end if self.air is not None else None
        return end is not None and str(end.link.fsm.state) not in ("DISCONNECTED", "LISTENING")

    def make_lane(self, depth: int):
        from hfmodem.station.air import StationAir
        lane = StreamLane(self.waveform.rate, depth=depth)
        self.air = StationAir(self, lane, log=lambda m: print(m, file=sys.stderr))
        return lane

    def poll(self):
        """The air owns the endpoint, so audio is handed across rather than
        decoded here — everything that touches the state machine runs on its
        thread."""
        self.air.pump()
        return []

    def start(self) -> None:
        self.air.start()

    def stop(self) -> None:
        self.air.stop()

    def decode(self, samples: np.ndarray):
        raise AssertionError("sabir receives through its air, not a lane decode")

    def on_frame(self, frame) -> None:
        raise AssertionError("sabir receives through its air, not a lane decode")


class KestrelLink(ProtocolLink):
    """VARA.

    BW500 is what has been demodulated byte-exact from real audio; the wideband
    records are validated against our own encoder. Nothing holds this adapter to
    either bandwidth, so it declares both: the union in `core.occupied` is
    BW2300's, the wider of the two.
    """

    waveform = Waveform("vara", 48000)

    def decode(self, samples: np.ndarray):
        from hfmodem.kestrel.rx import varahf500 as v5
        try:
            res = v5.decode_stream(np.asarray(samples, float))
        except Exception:                       # noqa: BLE001
            return []
        return [res] if res else []

    def on_frame(self, frame) -> None:
        return None


class _ThreadedModemLink(ProtocolLink):
    indexed_audio = True

    def __init__(self, station, **kw):
        self._expect_audio = None
        super().__init__(station, **kw)

    def make_lane(self, depth: int):
        return StreamLane(CARD_RATE_HZ, depth=depth)

    def _feed(self, at, chunk):
        if self.indexed_audio:
            self.modem.feed(at, chunk)
        else:
            self.modem.feed(chunk)

    def poll(self):
        for at, chunk in self.lane.poll():
            expect = at if self._expect_audio is None else self._expect_audio
            if at > expect:
                gap = at - expect
                if gap > CARD_RATE_HZ:
                    self.modem.mark_audio_gap()
                else:
                    self._feed(expect, np.zeros(gap, dtype=np.float32))
            self._feed(at, chunk)
            self._expect_audio = at + len(chunk)
        return []

    def _send(self, audio: np.ndarray, rate: int):
        if rate != self.waveform.rate:
            raise ValueError(f"{self.waveform.name}: expected {self.waveform.rate} Hz, got {rate}")
        end = self.transmit(audio)
        return self.refusals[-1] if end is None else end

    def start(self) -> None:
        self.modem.start()

    def stop(self) -> None:
        self.modem.stop()


class BateleurLink(_ThreadedModemLink):
    """The bateleur modem (docs/protocols/bateleur).

    A transmission can run for minutes — a long-interleave message has a
    4.8 s preamble before its first byte — which no rolling window holds, and
    decoding it again on every poll would cost more than real time. So it takes
    the stream, every sample once, and runs its own streaming receiver on the
    modem's thread, the arrangement sabir's air uses and for the same reason.

    The modem renders at 9600 Hz; the arbiter converts at the edge, as for
    every protocol. A burst longer than the rig's ``max_key_s`` is never asked
    for: the modem splits the message into transmissions that each fit.

    ``bandwidth`` is read once, here, for both the modem and the gate
    (docs/CONFIG.md).
    """

    waveform = Waveform("bateleur", BATELEUR_RATE, "3000")

    def __init__(self, station, **kw) -> None:
        from hfmodem.bateleur.modem import BateleurModem
        proto = station.cfg.protocols.get("bateleur")
        bw = (proto.bandwidth if proto else "") or "3000"
        self.waveform = Waveform("bateleur", BATELEUR_RATE, bw)
        self.modem = BateleurModem(
            self._send, card_rate=CARD_RATE_HZ, max_key_s=station.cfg.rig.max_key_s,
            log=lambda m: print(m, file=sys.stderr), trim=(bw == "3000"))
        super().__init__(station, **kw)

    indexed_audio = False

    def _send(self, audio: np.ndarray, rate: int) -> str:
        result = super()._send(audio, rate)
        return result if isinstance(result, str) else ""


class AleLink(_ThreadedModemLink):
    """2G ALE (MIL-STD-188-141B appendix A).

    The one lane that moves the radio. While available it scans the channels
    in ``[protocols.ale] channels``, retuning the rig to each; linked, it stays
    on the channel the link was made on, and whatever else the station runs
    then runs there. A retune is refused while anything is keyed, and the scan
    tries again on its next step.

    The channels are dial frequencies, as ALE channel lists publish them. The
    rig is told the channel centre (``Rig.tune`` derives the dial from it), so
    the dial is exactly the listed frequency.

    Like bateleur it takes the stream, every sample once, and runs on its own
    thread; its receiver and controller carry state across chunks.
    """

    waveform = Waveform("ale", ALE_RATE)

    def __init__(self, station, **kw) -> None:
        from hfmodem.bateleur.ale.controller import Config
        from hfmodem.bateleur.ale.modem import AleModem
        self.station = station
        p = station.cfg.protocols["ale"]
        check_linking_protection(station.cfg.station.regulatory, p.lp_level)
        self.dials = [int(round(f)) for f in p.channels]
        self._others_sent = 0
        names = [f"{d / 1e3:.1f}" for d in self.dials]
        nets = {n["address"].upper(): (n["member"].upper(), n["slot"]) for n in p.nets}
        common = dict(scan_rate=p.scan_rate, scanning=len(self.dials) > 1,
                      listen_s=POUNCE_WINDOW_S, lp_level=p.lp_level, lp_key=p.lp_key,
                      channel_hz=[float(d) for d in self.dials],
                      max_key_s=station.cfg.rig.max_key_s)
        if p.conventions == "hflink":
            from hfmodem.bateleur.ale import hflink
            pilots = [self.dials.index(int(round(f))) for f in p.pilot_channels] or None
            cfg = hflink.config(p.addresses[0], names, base=p.base, pilot_channels=pilots,
                                **common)
            cfg.addresses = [a.upper() for a in p.addresses]
            cfg.nets.update(nets)
        else:
            cfg = Config(list(p.addresses), names, activity_s=p.activity_s,
                         sound_s=p.sound_s, nets=nets, **common)
        if p.lp_level:
            print(f"ale: linking protection AL-{p.lp_level} enabled; ALE words use the configured key", file=sys.stderr)
            for dial in self.dials:
                self._warn_protection(dial, p.lp_level)
        self.modem = AleModem(cfg, send=self._send, tune=self._tune, busy=self._busy,
                              card_rate=CARD_RATE_HZ, hold=self._held,
                              log=lambda m: print(m, file=sys.stderr))
        super().__init__(station, **kw)

    def _warn_protection(self, dial: int, level: int) -> None:
        if level and amateur_hf(dial):
            profile = self.station.cfg.station.regulatory
            print(f"ale: warning: linking protection AL-{level} at {dial} Hz "
                  f"is inside an amateur band-plan range; your chosen {profile} "
                  "profile applies a warning here.",
                  file=sys.stderr)

    def transmit(self, audio: np.ndarray, **kw) -> int | None:
        level = self.modem.ctl.cfg.lp_level
        try:
            check_linking_protection(self.station.cfg.station.regulatory, level)
            if self.station.arbiter is not None and self.station.rig is not None:
                self._warn_protection(self.station.rig.dial(), level)
        except (ConfigError, RigError) as exc:
            self.refusals.append(str(exc))
            return None
        return super().transmit(audio, **kw)

    def _held(self):
        for name, lane in self.station.lanes.items():
            if name != "ale" and lane.active:
                return True
        return False

    def _send(self, audio: np.ndarray, rate: int):
        rig = self.station.rig
        try:
            if rig is not None and abs(rig.dial() - self.dials[self.modem.ctl.channel]) > band.QSY_TOLERANCE_HZ:
                return "rig dial differs from ALE channel"
        except RigError as exc:
            return f"dial read failed: {exc}"
        return super()._send(audio, rate)

    def _tune(self, channel: int):
        if self._held():
            return None
        rig = self.station.rig
        audio = self.station.audio
        arbiter = self.station.arbiter
        try:
            with arbiter.try_hold() if arbiter is not None else nullcontext(True) as held:
                if not held:
                    return None
                if rig is not None:
                    try:
                        rig.tune(self.dials[channel] + band.DIAL_OFFSET_HZ)
                    finally:
                        # Publish before releasing retune ownership.  A queued
                        # RPR request then fails eligibility on the new dial.
                        rpr = self.station.lanes.get("rpr")
                        if rpr is not None:
                            rpr.retuned(int(audio.sample_now()) if audio is not None else 0)
        except RigError as exc:
            print(f"ale: rig would not tune to {self.dials[channel] / 1e3:.1f} kHz ({exc})", file=sys.stderr)
            return None
        return int(audio.sample_now()) if audio is not None else 0

    def _busy(self, audio: np.ndarray) -> bool:
        return is_busy(audio, CARD_RATE_HZ, occupied_hz("ale"))[0]

    def _load_directory(self, sources) -> None:
        from hfmodem.bateleur.ale import directory as D
        srcs = [None if s == "default" else s for s in sources]
        try:
            d = D.load(srcs)
        except (OSError, ValueError) as exc:
            print(f"ale: directory: {exc}", file=sys.stderr)
            return
        self.modem.directory = d
        print(f"ale: directory: {d.summary()}", file=sys.stderr)

    def poll(self):
        super().poll()
        # Other protocols' bursts are the link's traffic (Twa, A.5.5.3.5).
        sent = self.station.arbiter.last_sent if self.station.arbiter is not None else {}
        latest = max((e for p, e in list(sent.items()) if p not in ("ale", "ident")), default=0)
        if latest > self._others_sent:
            self._others_sent = latest
            self.modem.activity(latest)
        return []

    def start(self) -> None:
        self.modem.start()
        sources = self.station.cfg.protocols["ale"].directory
        if sources:
            threading.Thread(target=self._load_directory, args=(list(sources),),
                             name="ale-directory", daemon=True).start()

    def stop(self) -> None:
        self.modem.stop()
        rig = self.station.rig
        if rig is not None:
            try:
                rig.tune(self.station.cfg.rig.centre_hz)
            except RigError as exc:
                print(f"ale: could not restore station frequency ({exc})", file=sys.stderr)



class SelcallLink(_ThreadedModemLink):
    """HF selcall (CCIR 493-4 land mobile, the UN open 6-digit selcall).

    It listens on whatever channel the radio is on, reports every call it
    hears, and sends calls on command. It never moves the radio: a network
    that scans for selcall wants the radio's own scanning, or the ALE lane's.
    """

    waveform = Waveform("selcall", SELCALL_RATE)

    def __init__(self, station, **kw) -> None:
        from hfmodem.bateleur.selcall.modem import SelcallModem
        p = station.cfg.protocols["selcall"]
        self.modem = SelcallModem(list(p.addresses), send=self._send, card_rate=CARD_RATE_HZ,
                                  dots_s=p.dots_s, invert=p.invert, acknowledge=p.acknowledge,
                                  log=lambda m: print(m, file=sys.stderr))
        super().__init__(station, **kw)



@dataclass
class RprRequest:
    identifier: int
    frame: bytes
    state: RequestState
    queued: float
    attempts: int = 0
    deferrals: int = 0


class RprLink(ProtocolLink):
    """A streaming packet lane with one FIFO transmit worker and one KISS owner."""

    waveform = Waveform("rpr", CARD_RATE_HZ)

    @classmethod
    def validate_options(cls, options):
        from hfhost.config import ConfigError

        if not isinstance(options, dict):
            raise ConfigError("rpr options must be a table")
        allowed = {"limiter", "level", "txdelay", "persistence", "slot_ms", "tx_events"}
        if options.keys() - allowed:
            raise ConfigError(f"unknown rpr options: {sorted(options.keys() - allowed)}")
        if "limiter" in options and type(options["limiter"]) is not bool:
            raise ConfigError("rpr limiter must be boolean")
        for key, lo, hi in (("level", 0, 1), ("txdelay", 0, 2550),
                            ("persistence", 0, 255), ("slot_ms", 1, 2550)):
            if key in options and (type(options[key]) is not int or
                                   not lo <= options[key] <= hi):
                raise ConfigError(f"rpr {key} must be an integer in {lo}..{hi}")
        if "tx_events" in options and (not isinstance(options["tx_events"], str) or
                                       not options["tx_events"] or "\0" in options["tx_events"]):
            raise ConfigError("rpr tx_events must be a nonempty path")

    def __init__(self, station, *, on_frame=None, **kw):
        from hfmodem.brubru import rx, tx, waveform
        from hfmodem.brubru.access import Access

        options = dict(station.cfg.protocols["rpr"].options)
        self.limiter = options.get("limiter", True)
        self.level = options.get("level", 0)
        self.txdelay = options.get("txdelay", 100)
        self.access = Access(options.get("persistence", 63), options.get("slot_ms", 100))
        self.receiver = rx.Receiver()
        self.decimator = waveform.Decimator()
        self._expected = None
        self._discard_fault = None
        self._on_frame = on_frame
        self.host = None
        self.received = deque(maxlen=64)
        self.rx_events = deque(maxlen=64)
        self.events = TxEvents("rpr")
        self.outcomes = deque(maxlen=64)
        self._journal_path = options.get("tx_events")
        self._journal_lock = threading.Lock()
        self._condition = threading.Condition()
        self._queue = deque()
        self._active = None
        self._stopping = False
        self._worker = None
        self._identifier = 0
        self._history = None
        super().__init__(station, **kw)
        # Measure after card conversion before accepting any client parameters.
        tx.onset_s(False)
        tx.onset_s(True)

    def make_lane(self, depth):
        return StreamLane(CARD_RATE_HZ, depth=depth)

    def retuned(self, index):
        """Invalidate both queued receive samples and the prior channel floor."""
        self.lane.fault(index, "ALE retune")
        self.access.retune(index, self.lane.fault_serial)

    def poll(self):
        result = []
        for chunk in self.lane.poll():
            at, samples = chunk
            if chunk.faults:
                fault = max(chunk.faults, key=lambda f: (f.index, f.serial))
                self._discard_fault = fault.index
                self.rx_events.append((fault.index, ", ".join(dict.fromkeys(
                    f.cause for f in chunk.faults))))
                self.receiver.reset()
                self.access.hold(False)
                self.access.fault(fault.index, chunk.faults[-1].serial)
                self._expected = None
            if self._discard_fault is not None:
                boundaries = [b for b in chunk.boundaries if b > self._discard_fault]
                if not boundaries:
                    continue
                resume = min(boundaries)
                skip = max(0, resume - at)
                samples, at = samples[skip:], max(at, resume)
                self._discard_fault = None
                self.access.fault(at, self.lane.fault_serial)
            if not len(samples):
                continue
            if self._expected != at:
                self.decimator.reset(at)
                native = (at + (self.decimator.DELAY - at) % 5 - self.decimator.DELAY) // 5
                self.receiver.reset(native)
            self._expected = at + len(samples)
            # DCD and acquisition advance together, so a held frame freezes its floor.
            for offset in range(0, len(samples), self.access.WINDOW):
                card = samples[offset:offset + self.access.WINDOW]
                _, values = self.decimator.feed(at + offset, card)
                held = self.receiver.held
                decoded = self.receiver.feed(values)
                if self.receiver.preambles:
                    self.access.preamble(at + offset + len(card))
                    self.receiver.preambles.clear()
                self.access.hold(held or self.receiver.held or bool(decoded))
                self.access.feed(at + offset, card)
                self.access.hold(self.receiver.held)
                for burst in decoded:
                    position = 5 * burst.ref + self.decimator.DELAY
                    self.received.append((position, burst))
                    for raw in burst.frames:
                        self.on_frame(raw)
                    result.append((position, burst))
            while self.receiver.events:
                index, reason = self.receiver.events.popleft()
                self.rx_events.append((5 * index + self.decimator.DELAY, reason))
        with self._condition:
            self._condition.notify_all()
        return result

    def on_frame(self, raw):
        if self.host is not None:
            self.host.send_frame(raw)
        if self._on_frame is not None:
            self._on_frame(raw)

    def set_parameter(self, command, value):
        with self._condition:
            if command == 1:
                self.txdelay = value * 10
            elif command == 2:
                self.access.persistence = value
            elif command == 3:
                self.access.slot_ms = value * 10
            self._condition.notify_all()

    def enqueue(self, raw):
        with self._condition:
            self._identifier += 1
            request = RprRequest(self._identifier, bytes(raw), RequestState(), time.monotonic())
            if self._stopping or len(self._queue) + (self._active is not None) >= 16:
                reason = "stopped" if self._stopping else "inbound queue full"
            else:
                self._queue.append(request)
                self._condition.notify_all()
                return request
        self._record(request, "refused", reason)
        return request

    def _record(self, request, outcome, reason="", **fields):
        if not self._journal_lock.acquire(timeout=2):
            self._fatal("rpr journal producer did not drain")
        try:
            event = self.events.record("completed", request.identifier, outcome=outcome,
                                       reason=reason, attempt=request.attempts,
                                       deferrals=request.deferrals, state=request.state.value,
                                       **fields)
            self.outcomes.append(event)
            self._drain_events()
        finally:
            self._journal_lock.release()

    def _fatal(self, why):
        fatal = getattr(self.station, "fatal_shutdown", None)
        if fatal is not None:
            fatal(why)
        raise LeaseTimeout(why)

    def _drain_events(self):
        lines = self.events.lines()
        if self._journal_path and lines:
            with open(self._journal_path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")

    def flush_events(self, timeout=None):
        acquired = (self._journal_lock.acquire() if timeout is None else
                    self._journal_lock.acquire(timeout=timeout))
        if acquired:
            try:
                self._drain_events()
            finally:
                self._journal_lock.release()
        return acquired

    def drop_pending(self):
        with self._condition:
            pending = list(self._queue)
            self._queue.clear()
            for request in pending:
                request.state.cancel()
            self._condition.notify_all()
        for request in pending:
            self._record(request, "error", "cancelled")

    def start(self):
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="rpr-tx", daemon=True)
            self._worker.start()

    def stop(self):
        with self._condition:
            self._stopping = True
            for request in self._queue:
                request.state.cancel()
            if self._active is not None:
                self._active.state.cancel()
            self._condition.notify_all()
        arbiter = self.station.arbiter
        if arbiter is not None:
            arbiter.disable("rpr", "lane stopped")
        self.drop_pending()
        if self._worker is not None:
            limit = self.station.cfg.rig.max_key_s + self.station.cfg.rig.ptt.settle_s + 5
            self._worker.join(limit)
            if self._worker.is_alive():
                self._fatal("rpr worker did not stop")
        if not self.flush_events(timeout=2):
            self._fatal("rpr journal did not stop")

    def _fresh_after(self):
        arbiter = self.station.arbiter
        if arbiter is not None:
            history = arbiter.transmit_history()
            if history != self._history:
                self._history = history
                return history.end
        return None

    def _fresh(self):
        after = self._fresh_after()
        if after is not None:
            self.access.require_fresh(after)

    def _eligible(self):
        return self.access.eligible(self.lane.fault_serial, fresh_after=self._fresh_after(),
                                    transmitting=True)

    def _run(self):
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait(0.05)
                if self._stopping:
                    return
                request = self._queue.popleft()
                self._active = request
            try:
                self._serve(request)
            except Exception as exc:
                logging.getLogger(__name__).exception("rpr transmit worker: %s", exc)
                self._record(request, "error", str(exc))
            finally:
                with self._condition:
                    self._active = None

    def _serve(self, request):
        from hfmodem.brubru.tx import transmit

        arbiter, rig = self.station.arbiter, self.station.rig
        if arbiter is None or rig is None:
            self._record(request, "refused", "station has no transmitter")
            return
        settle = self.station.cfg.rig.ptt.settle_s
        try:
            bursts = transmit([request.frame], self.level, limiter=self.limiter,
                              max_key_s=rig.max_key_s, txdelay=self.txdelay, settle_s=settle)
            if len(bursts) != 1:
                raise ValueError("frame fits no permitted class")
            burst, = bursts
        except ValueError as exc:
            self._record(request, "refused", str(exc))
            return
        while True:
            if request.state.value == "cancelled":
                self._record(request, "error", "cancelled")
                return
            if time.monotonic() - request.queued >= 60:
                self._record(request, "refused", "request expired")
                return
            self._fresh()
            if not self.access.contend(time.monotonic(), self.lane.fault_serial):
                with self._condition:
                    self._condition.wait(0.02)
                continue
            req = TxRequest(LIVE, request.identifier, "rpr", burst.audio, 9600,
                            self.waveform.emission(rig.dial(), power_w=self.station.cfg.rig.power_w),
                            settle_s=settle, state=request.state, eligible=self._eligible,
                            lead_s=burst.lead_s(self.txdelay, settle), expires=request.queued + 60)
            try:
                result = arbiter.submit_result(req)
            except Exception as exc:
                request.state.commit()
                request.attempts += 1
                self._record(request, "error", str(exc), arbiter_state=None)
                return
            finally:
                if self.access.transmitting:
                    self.access.tx(False, int(self.station.audio.sample_now()))
            if result.outcome == "deferred":
                request.deferrals += 1
                self.access.require_fresh(max(arbiter.transmit_history().end, self.access.evaluated))
                continue
            request.attempts += 1
            outcome = "error" if result.outcome == "invalidated" else result.outcome
            self._record(request, outcome, result.reason, end=result.end,
                         displacement=result.displacement)
            if result.outcome == "sent":
                self.sent += 1
            if (result.outcome == "invalidated" and result.state == "waiting"
                    and request.attempts == 1):
                self.access.require_fresh(max(arbiter.transmit_history().end, self.access.evaluated))
                continue
            return


#: What `station/process.py` builds when a protocol is enabled. A dict rather than
#: a registry class: these links are known at import time.
LINKS = {
    "pactor": ShrikeLink,
    "vara": KestrelLink,
    "ardop": BesraLink,
    "sabir": SabirLink,
    "bateleur": BateleurLink,
    "ale": AleLink,
    "selcall": SelcallLink,
    "rpr": RprLink,
}
