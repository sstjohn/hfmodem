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
from dataclasses import dataclass

import numpy as np

from hfmodem.core.audio import RollingLane, StreamLane
from hfmodem.core.occupied import keyed_hz
from hfmodem.core.regulatory import Emission
from hfmodem.station.arbiter import LIVE, Refused, TxRequest


@dataclass(frozen=True, slots=True)
class Waveform:
    """What a protocol occupies, in the terms the regulatory gate asks about.

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
    no protocol to a ceiling, so no protocol may claim one.
    """

    name: str
    rate: int

    def emission(self, dial_hz: int, **kw) -> Emission:
        lo, hi = keyed_hz(self.name)
        return Emission(dial_hz, lo, hi, technique=self.name, **kw)


class ProtocolLink:
    """One protocol on the station's radio.

    Subclasses supply `decode` and `on_frame`; everything else — the lane, the
    request, the refusal handling — is the same for all four, because the parts
    that differ are the waveform and the ARQ, and neither of those belongs here.
    """

    waveform: Waveform

    def __init__(self, station, *, depth: int = 4096) -> None:
        self.station = station
        self.lane = self.make_lane(depth)
        self.refusals: list[str] = []
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

        A refusal is not an error here. The station refuses for reasons a protocol
        has no view of — an identification is owed, the burst would arrive after
        the instant it was scheduled for — and the protocol's correct response to
        all of them is the same as to a lost frame.
        """
        arb = self.station.arbiter
        if arb is None:
            self.refusals.append("no arbiter: this station cannot transmit")
            return None
        req = TxRequest(
            priority=LIVE, seq=0, protocol=self.waveform.name,
            audio=np.asarray(audio), rate=self.waveform.rate,
            emission=self.waveform.emission(
                self.station.rig.dial(),
                power_w=power_w if power_w is not None else self.station.cfg.rig.power_w),
            at=at, responding=responding,
            settle_s=self.station.cfg.rig.ptt.settle_s)
        try:
            end = arb.submit(req)
        except Refused as exc:
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


#: What `station/process.py` builds when a protocol is enabled. A dict rather than
#: a registry class: there are four, they are known at import time, and an
#: abstraction over four entries is ceremony.
LINKS = {
    "pactor": ShrikeLink,
    "vara": KestrelLink,
    "ardop": BesraLink,
    "sabir": SabirLink,
}
