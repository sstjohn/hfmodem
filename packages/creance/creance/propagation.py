# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Propagation context for a session, from WSPR spot data.

The problem this solves is the biggest confounder in on-air testing: a session
that fails tells you nothing on its own, because "the far end refused us" and
"the band was shut" look identical from one end of a one-way path. A conformance
result that cannot separate those is not a result.

Two sources, in order of strength:

**Our own beacon.** Transmit WSPR from this station and ask who logged it. That
measures *this* station — antenna, feedline, power, local noise — and is the
only thing that actually answers "can that gateway hear me". Preferred whenever
a beacon has been sent recently.

**Everyone else's spots.** Spots between stations near us and stations near the
target, in the recent past. Available with no transmission at all, and covers
paths we have never beaconed, but it measures somebody else's station rather
than ours.

**The evidence is asymmetric, and the code says so rather than leaving it to be
assumed.** No spots anywhere on a band is a strong negative: do not spend a
session slot there. Spots present is a *weak positive* — WSPR is a 6 Hz beacon
decoding near -30 dB SNR, and a 2.3 kHz ARQ link needs vastly more margin. A
path that carries WSPR may still refuse PACTOR. So a summary carries a verdict
of OPEN / CLOSED / UNKNOWN where CLOSED is trustworthy and OPEN means "worth
trying", never "will work".

Network is optional throughout. A remote site has opportunistic IP at best, and
a propagation lookup must never delay or fail a session — every entry point
returns UNKNOWN on any error and records why.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

ENDPOINT = "https://db1.wspr.live/"

OPEN = "open"          # spots seen: worth trying, not a guarantee
CLOSED = "closed"      # nothing heard: a trustworthy negative
UNKNOWN = "unknown"    # no data, no network, or never asked

#: WSPR band identifiers are the MHz of the dial, with 0/-1 for LF/MF. Only the
#: HF bands a Winlink station would use are mapped.
BANDS = {"160m": 1, "80m": 3, "60m": 5, "40m": 7, "30m": 10, "20m": 14,
         "17m": 18, "15m": 21, "12m": 24, "10m": 28}


@dataclass(frozen=True, slots=True)
class BandReport:
    band: str
    verdict: str
    spots: int = 0
    best_snr: int | None = None
    median_km: int | None = None
    source: str = "none"          # "own" | "path" | "none"
    detail: str = ""

    @property
    def worth_trying(self) -> bool:
        """UNKNOWN counts as worth trying: absence of evidence is not evidence
        of a shut band, and refusing to transmit because a web service was
        unreachable would be its own failure mode."""
        return self.verdict != CLOSED


@dataclass(frozen=True, slots=True)
class Propagation:
    """What was known about the path when a session ran."""
    asked_at: float = 0.0
    window_h: float = 0.0
    bands: dict[str, BandReport] = field(default_factory=dict)
    error: str = ""

    def best(self) -> BandReport | None:
        """The band with the strongest evidence, or None if nothing is known."""
        seen = [b for b in self.bands.values() if b.verdict == OPEN]
        if not seen:
            return None
        return max(seen, key=lambda b: (b.spots, b.best_snr or -99))

    def as_dict(self) -> dict:
        return {"asked_at": self.asked_at, "window_h": self.window_h,
                "error": self.error,
                "bands": {k: {"verdict": v.verdict, "spots": v.spots,
                              "best_snr": v.best_snr, "median_km": v.median_km,
                              "source": v.source}
                          for k, v in self.bands.items()}}


def _query(sql: str, timeout: float) -> list[dict]:
    url = ENDPOINT + "?" + urllib.parse.urlencode(
        {"query": sql + " FORMAT JSONEachRow"})
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        body = fh.read().decode("utf-8", "replace")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def own_beacon(call: str, near: tuple[float, float], radius_km: float = 300.0,
               hours: float = 3.0, timeout: float = 15.0,
               fetch=_query) -> Propagation:
    """Who logged our own WSPR beacon, by band.

    The strongest evidence available, because it measures this station. `near`
    is the receiver's neighbourhood — the target we care about being heard by.
    """
    lon, lat = near[1], near[0]
    sql = (f"SELECT band, count() AS n, max(snr) AS best, "
           f"round(median(distance)) AS km FROM wspr.rx "
           f"WHERE time > subtractHours(now(),{hours:g}) "
           f"AND tx_sign = '{_safe(call)}' "
           f"AND greatCircleDistance(rx_lon,rx_lat,{lon:g},{lat:g}) "
           f"< {radius_km * 1000:g} GROUP BY band ORDER BY band")
    return _summarize(sql, hours, "own", timeout, fetch)


def path(from_: tuple[float, float], to: tuple[float, float],
         radius_km: float = 300.0, hours: float = 3.0, timeout: float = 15.0,
         fetch=_query) -> Propagation:
    """Anyone's spots between our neighbourhood and the target's.

    Weaker than our own beacon — it measures somebody else's station — but it
    needs no transmission and covers paths we have never beaconed.
    """
    sql = (f"SELECT band, count() AS n, max(snr) AS best, "
           f"round(median(distance)) AS km FROM wspr.rx "
           f"WHERE time > subtractHours(now(),{hours:g}) "
           f"AND greatCircleDistance(tx_lon,tx_lat,{from_[1]:g},{from_[0]:g}) "
           f"< {radius_km * 1000:g} "
           f"AND greatCircleDistance(rx_lon,rx_lat,{to[1]:g},{to[0]:g}) "
           f"< {radius_km * 1000:g} GROUP BY band ORDER BY band")
    return _summarize(sql, hours, "path", timeout, fetch)


def _summarize(sql: str, hours: float, source: str, timeout: float,
               fetch) -> Propagation:
    now = time.time()
    try:
        rows = fetch(sql, timeout)
    except Exception as exc:                # offline is normal, not exceptional
        return Propagation(asked_at=now, window_h=hours,
                           error=f"{type(exc).__name__}: {exc}")

    by_id = {int(r["band"]): r for r in rows if "band" in r}
    bands = {}
    for name, ident in BANDS.items():
        row = by_id.get(ident)
        if row is None:
            bands[name] = BandReport(name, CLOSED, source=source,
                                     detail=f"no spots in {hours:g}h")
        else:
            bands[name] = BandReport(
                name, OPEN, spots=int(row.get("n", 0)),
                best_snr=_int(row.get("best")), median_km=_int(row.get("km")),
                source=source)
    return Propagation(asked_at=now, window_h=hours, bands=bands)


def unknown(reason: str = "not queried") -> Propagation:
    return Propagation(error=reason)


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe(call: str) -> str:
    """Callsigns are [A-Z0-9/-] and this string is interpolated into SQL, so
    anything else is refused rather than escaped."""
    ok = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-")
    up = call.upper()
    if not up or not set(up) <= ok:
        raise ValueError(f"implausible callsign {call!r}")
    return up
