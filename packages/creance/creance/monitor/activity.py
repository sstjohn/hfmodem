# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The merged activity view: what is on this frequency right now, and how sure.

Every modem monitor emits normalised detections onto one timeline; this module
merges them and renders the result two ways — a human-readable live log, and a
structured per-protocol summary an automatic connect-targeter can read.

The grading discipline is the same one ``creance.propagation`` uses for bands,
and for the same reason: a false positive here costs a wasted transmit, which is
the exact failure a monitor exists to prevent. So a detection is CONFIRMED only
when it decoded to certainty — a CRC validated, a callsign resolved, a gateway
matched — and TENTATIVE when a signal was clearly present and classified but its
identity was never proven. A tentative detection is *never* promoted to a
confident "gateway here"; the summary carries the two grades all the way out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

CONFIRMED = "confirmed"    # decoded to certainty: CRC ok / callsign / gateway match
TENTATIVE = "tentative"    # signal present and classified, identity unproven
GRADES = (CONFIRMED, TENTATIVE)

#: per-protocol channel state, worst-to-best. A protocol is only ACTIVE_CONFIRMED
#: when something under it decoded to certainty in the window.
QUIET = "quiet"
ACTIVE_TENTATIVE = "active_tentative"
ACTIVE_CONFIRMED = "active_confirmed"

#: protocols a monitor may name. UNKNOWN is real energy that classified as no
#: known waveform — it counts as "something is here" but names no protocol.
UNKNOWN = "UNKNOWN"

#: which clock a rendered log line stamps its events with. STREAM counts audio
#: this process actually received; WALL is the time of day that audio arrived.
STREAM_CLOCK = "stream"
WALL_CLOCK = "wall"


@dataclass(frozen=True, slots=True)
class Detection:
    """One normalised event on the common timeline, whatever modem found it."""
    t: float                   # seconds of audio into the stream THIS PROCESS
                               # received — not an offset into any recording
    modem: str                 # "kestrel" | "shrike" | ...
    protocol: str              # VARA | PACTOR-1/2/3 | ARDOP | UNKNOWN
    kind: str                  # the modem's own event label
    grade: str                 # CONFIRMED | TENTATIVE
    station: str = ""          # callsign, if resolved
    role: str = ""             # "gateway" | "caller" | ""
    detail: str = ""
    wall: float = 0.0          # unix time the audio carrying it arrived

    @classmethod
    def from_json(cls, obj: dict) -> "Detection":
        return cls(
            t=float(obj["t"]), modem=str(obj["modem"]),
            protocol=str(obj.get("protocol", UNKNOWN)),
            kind=str(obj.get("kind", "")),
            grade=obj["grade"] if obj.get("grade") in GRADES else TENTATIVE,
            station=str(obj.get("station", "")),
            role=str(obj.get("role", "")), detail=str(obj.get("detail", "")),
            wall=float(obj.get("wall", 0.0)))


def _stamp(d: Detection, clock: str) -> str:
    """The leading field of a log line, in a form that says what it is.

    Bare seconds read as an offset into the recording of the session, and on
    live audio that is provably what they are not. Measured 2026-08-14: a live
    pass printed 309.57 and 355.56 for two events a recording of the same window
    puts at 380.18 and 436.27, because the figure counted only the samples that
    survived the capture — 0.820 x the recorder's own clock, itself 0.910 x real
    time. So a live line is stamped with the wall clock, which no amount of lost
    audio can shift, and a replay line with the offset into the file, marked
    ``s`` so the two can never be mistaken for one another.
    """
    if clock == WALL_CLOCK:
        # tenths off the fractional part, not off `wall * 10`: a unix time is
        # large enough that the second multiplication loses the digit.
        return (time.strftime("%H:%M:%S", time.localtime(d.wall))
                + f".{int(d.wall % 1 * 10)}")
    return f"{d.t:9.2f}s"


def format_detection(d: Detection, *, clock: str = STREAM_CLOCK) -> str:
    """One human log line, confidence marked, aligned with its peers."""
    mark = "  " if d.grade == CONFIRMED else " ?"
    who = f" {d.station}" if d.station else ""
    tail = f"  {d.detail}" if d.detail else ""
    return (f"[{_stamp(d, clock)}]{mark} {d.modem:<8} {d.protocol:<9} "
            f"{d.kind:<16}{who}{tail}".rstrip())


@dataclass(slots=True)
class _Track:
    """The last time a station was heard, and how surely, within the window."""
    last_t: float
    grade: str
    role: str


@dataclass
class ActivityView:
    """Rolling merge of every modem's detections over the last ``window_s``.

    Feed it detections as they arrive and ask it, at any wall time, for the
    state of each protocol and the stations heard under it. It forgets anything
    older than the window so a burst that has stopped stops counting.
    """
    window_s: float = 20.0
    #: protocol -> best grade seen in window, and last time
    _proto: dict[str, _Track] = field(default_factory=dict)
    #: (protocol, station) -> track
    _stations: dict[tuple[str, str], _Track] = field(default_factory=dict)
    last_t: float = -1e9

    def add(self, d: Detection) -> None:
        self.last_t = max(self.last_t, d.t)
        cur = self._proto.get(d.protocol)
        if cur is None or _stronger(d.grade, d.t, cur.grade, cur.last_t):
            self._proto[d.protocol] = _Track(d.t, d.grade, d.role)
        else:
            cur.last_t = max(cur.last_t, d.t)
        if d.station:
            key = (d.protocol, d.station)
            st = self._stations.get(key)
            if st is None or _stronger(d.grade, d.t, st.grade, st.last_t):
                self._stations[key] = _Track(d.t, d.grade, d.role)

    def _fresh(self, t_now: float, track: _Track) -> bool:
        return (t_now - track.last_t) <= self.window_s

    def summary(self, t_now: float) -> dict:
        """Structured state for a targeter: per protocol a QUIET/ACTIVE verdict,
        the stations heard under it, and one overall verdict."""
        protocols: dict[str, dict] = {}
        for proto, track in self._proto.items():
            if proto == UNKNOWN or not self._fresh(t_now, track):
                continue
            state = (ACTIVE_CONFIRMED if track.grade == CONFIRMED
                     else ACTIVE_TENTATIVE)
            stations = [
                {"call": call, "role": st.role or None, "grade": st.grade,
                 "age_s": round(t_now - st.last_t, 1)}
                for (p, call), st in self._stations.items()
                if p == proto and self._fresh(t_now, st)]
            stations.sort(key=lambda s: s["age_s"])
            protocols[proto] = {"state": state,
                                "age_s": round(t_now - track.last_t, 1),
                                "stations": stations}
        unknown = self._proto.get(UNKNOWN)
        energy = bool(unknown and self._fresh(t_now, unknown))
        overall = _overall(protocols, energy)
        return {"t": round(t_now, 2), "window_s": self.window_s,
                "protocols": protocols, "unidentified_energy": energy,
                "overall": overall}


def render_summary(summary: dict) -> str:
    """A single operator-facing line: what is here, and whether to try it."""
    parts = []
    for proto, info in sorted(summary["protocols"].items()):
        if info["state"] == ACTIVE_CONFIRMED:
            named = ", ".join(
                f"{s['call']}({s['role']})" if s["role"] else s["call"]
                for s in info["stations"] if s["grade"] == CONFIRMED)
            parts.append(f"{proto} active"
                         + (f": {named}" if named else " (link decoded)"))
        else:
            parts.append(f"{proto}? tentative")
    if not parts and summary["unidentified_energy"]:
        parts.append("unidentified energy")
    body = " | ".join(parts) if parts else "quiet"
    return f"{body}  ->  {_ADVICE[summary['overall']]}"


_ADVICE = {
    ACTIVE_CONFIRMED: "a station/link is confirmed here — target it, don't call blind",
    ACTIVE_TENTATIVE: "signal present but unproven — listen more before transmitting",
    QUIET: "nothing heard in the window",
}


def _stronger(g1: str, t1: float, g2: str, t2: float) -> bool:
    """A confirmed detection always outranks a tentative one; among equals the
    more recent wins. Confidence never decays into a stale confirmed masking a
    channel that has since gone tentative — freshness is judged separately in
    the window, this only picks which track to keep."""
    if g1 != g2:
        return g1 == CONFIRMED
    return t1 >= t2


def _overall(protocols: dict[str, dict], energy: bool) -> str:
    if any(p["state"] == ACTIVE_CONFIRMED for p in protocols.values()):
        return ACTIVE_CONFIRMED
    if protocols or energy:
        return ACTIVE_TENTATIVE
    return QUIET
