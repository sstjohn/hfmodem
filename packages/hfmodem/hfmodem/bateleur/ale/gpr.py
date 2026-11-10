# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ALE-GPR, the amateur ALE geo position report: a position as a plain AMD
message, so that any ALE radio can show it and an operator can key one in.

Seven fields between "*"s: GPR, the object (a callsign; "#" for "/"), the
latitude and longitude in decimal degrees with the point written N, S, E or
W, the altitude with M or F after it, the time as YYYYMMDD, a zone letter
(Z for UTC) and hhmmss, and a comment of up to 25 characters. A "#" in any
of the position or time fields marks it as keyed in by hand or not a valid
fix. Only letters, digits, space, "*" and "#" are used: those are what ALE
radios' displays and keypads have in common.

Written from HFLINK's published description of the format; it is sent to
the "GPR" net ending TWAS (`Controller.broadcast`), or in a call to one
station.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 #")


@dataclass
class Report:
    object: str                         # usually a callsign, "/" as itself
    lat: float | None                   # degrees, north positive; None if not given
    lon: float | None                   # degrees, east positive
    alt_m: float | None = None
    when: datetime | None = None        # UTC if the zone letter was Z
    comment: str = ""
    zone: str = "Z"
    #: A "#" in the position or time: keyed in by hand, or no valid fix.
    manual: bool = False


def _degrees(v: float, pos: str, neg: str, width: int, whole: int) -> str:
    """``v`` in at most ``width`` characters, the point written as the
    hemisphere; at least ``whole`` digits of degrees."""
    sign = pos if v >= 0 else neg
    places = max(0, width - max(whole, len(str(int(abs(v))))) - 1)
    deg, _, frac = f"{abs(v):.{places}f}".partition(".")
    return f"{int(deg):0{whole}d}{sign}{frac}"[:width]


def _clean(text: str, n: int) -> str:
    t = text.upper().replace("/", "#")
    return "".join(c if c in SAFE else "#" for c in t)[:n]


def format(r: Report) -> str:
    """The AMD text of a report (at most 90 characters: an AMD's limit)."""
    obj = _clean(r.object, 15)
    if len(obj) < 3:
        raise ValueError("the object is 3 to 15 characters")
    mark = "#" if r.manual else ""
    lat = "##" if r.lat is None else _degrees(r.lat, "N", "S", 9 - len(mark), 2) + mark
    lon = "###" if r.lon is None else _degrees(r.lon, "E", "W", 10 - len(mark), 3) + mark
    if r.lat is not None and not -90 <= r.lat <= 90 or r.lon is not None and not -180 <= r.lon <= 180:
        raise ValueError("latitude or longitude out of range")
    alt = "#M" if r.alt_m is None else f"{int(round(r.alt_m))}M"
    if len(alt) > 7:
        raise ValueError("the altitude is at most six digits of metres")
    if r.when is None:
        when = "#" * 15
    else:
        t = r.when.astimezone(timezone.utc) if r.zone == "Z" and r.when.tzinfo else r.when
        when = t.strftime("%Y%m%d") + r.zone + t.strftime("%H%M%S")
    comment = _clean(r.comment, 25) or "#"           # no "*" at the end
    return "*".join(["GPR", obj, lat, lon, alt, when, comment])


_DEG = re.compile(r"^(\d*)([NSEW])(\d*)$")


def _parse_deg(f: str, pos: str) -> float | None:
    m = _DEG.match(f.replace("#", ""))
    if not m or not m.group(1):
        return None
    v = float(f"{m.group(1)}.{m.group(3) or '0'}")
    return v if m.group(2) == pos else -v


def parse(text: str) -> Report | None:
    """A report from an AMD's text; None if it is not one."""
    f = text.strip().upper().split("*")
    if len(f) < 6 or f[0] != "GPR":
        return None
    f += [""] * (7 - len(f))
    manual = any("#" in x for x in f[2:6])
    alt = None
    m = re.match(r"^(\d+)([MF])$", f[4].replace("#", ""))
    if m:
        alt = float(m.group(1)) * (0.3048 if m.group(2) == "F" else 1.0)
    when, zone = None, "Z"
    m = re.match(r"^(\d{8})([A-Z])(\d{6})$", f[5])
    if m:
        zone = m.group(2)
        try:
            when = datetime.strptime(m.group(1) + m.group(3), "%Y%m%d%H%M%S")
        except ValueError:
            when = None
        if when is not None and zone == "Z":
            when = when.replace(tzinfo=timezone.utc)
    return Report(object=f[1].replace("#", "/"), lat=_parse_deg(f[2], "N"),
                  lon=_parse_deg(f[3], "E"), alt_m=alt, when=when,
                  comment="*".join(f[6:]).strip().strip("#"), zone=zone, manual=manual)
