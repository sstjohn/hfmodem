# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Grammar of the VARA-style TCP host interface: vocabulary and line classifier.

Written from the host-API mapping in ``docs/protocols/vara/07-host-api-mapping.md``.
This is the client half, written from the specification rather than from any
server — which is what makes creance's grading mean anything.
Where spec and both implementations diverge, this module follows the
implementations and records the delta in SPEC_DELTAS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CR = b"\r"
DEFAULT_CMD_PORT = 8300
DEFAULT_DATA_PORT = 8301

BANDWIDTHS = ("500", "2300", "2750")
COMPRESSION_MODES = ("OFF", "TEXT", "FILES")

SPEC_DELTAS = (
    "IAMALIVE flows from attach, not only while connected: spec §7.3 says "
    "~every 60 s while connected, but both servers heartbeat unconditionally",
)

# Line kinds.
REPLY_OK = "reply_ok"
REPLY_WRONG = "reply_wrong"
NOTIFICATION = "notification"
UNKNOWN = "unknown"

OK = "OK"
WRONG = "WRONG"


@dataclass(frozen=True, slots=True)
class Line:
    kind: str                       # REPLY_OK | REPLY_WRONG | NOTIFICATION | UNKNOWN
    name: str                       # leading verb ("" if the line was blank)
    fields: dict = field(default_factory=dict)
    raw: str = ""


def _bare(args: list[str]) -> dict | None:
    return {} if not args else None


def _on_off(args: list[str]) -> dict | None:
    if args == ["ON"]:
        return {"on": True}
    if args == ["OFF"]:
        return {"on": False}
    return None


def _connected(args: list[str]) -> dict | None:
    # "CONNECTED src dst bw" on HF/FM; bw absent on SAT.
    if len(args) == 2:
        return {"src": args[0], "dst": args[1], "bw": None}
    if len(args) == 3:
        return {"src": args[0], "dst": args[1], "bw": args[2]}
    return None


def _buffer(args: list[str]) -> dict | None:
    if len(args) == 1 and args[0].isdigit():
        return {"n": int(args[0])}
    return None


def _version(args: list[str]) -> dict | None:
    # Reply to the VERSION command, e.g. "VERSION 4.9.0.KestrelOpen".
    return {"version": " ".join(args)} if args else None


def _bitrate(args: list[str]) -> dict | None:
    # Spec §7.3: "BITRATE (N) x bps TX" / "... RX". Neither modem emits it yet.
    if len(args) != 4:
        return None
    level, bps, unit, direction = args
    if (level.startswith("(") and level.endswith(")") and level[1:-1].isdigit()
            and bps.isdigit() and unit == "bps" and direction in ("TX", "RX")):
        return {"level": int(level[1:-1]), "bps": int(bps), "direction": direction}
    return None


def _sn(args: list[str]) -> dict | None:
    if len(args) != 1:
        return None
    try:
        return {"value": float(args[0])}
    except ValueError:
        return None


def _registered(args: list[str]) -> dict | None:
    return {"call": args[0]} if len(args) == 1 else None


def _link(args: list[str]) -> dict | None:
    if args == ["REGISTERED"]:
        return {"registered": True}
    if args == ["UNREGISTERED"]:
        return {"registered": False}
    return None


def _cqframe(args: list[str]) -> dict | None:
    return {"src": args[0], "bw": args[1]} if len(args) == 2 else None


_NOTIFICATIONS = {
    "CONNECTED": _connected,
    "DISCONNECTED": _bare,
    "PTT": _on_off,
    "BUFFER": _buffer,
    "PENDING": _bare,
    "CANCELPENDING": _bare,
    "BUSY": _on_off,
    "IAMALIVE": _bare,
    "REGISTERED": _registered,
    "LINK": _link,
    "BITRATE": _bitrate,
    "SN": _sn,
    "CQFRAME": _cqframe,
    "VERSION": _version,
}


def classify(line: str) -> Line:
    """Classify one CR-delimited line from the command channel.

    Stateless: a VERSION reply classifies as a NOTIFICATION named VERSION;
    the client correlates it with its command. Malformed arguments to a known
    verb classify as UNKNOWN — grammar violations are conformance findings.
    """
    text = line.strip()
    if text == OK:
        return Line(REPLY_OK, OK, raw=line)
    if text == WRONG:
        return Line(REPLY_WRONG, WRONG, raw=line)
    parts = text.split()
    if not parts:
        return Line(UNKNOWN, "", raw=line)
    parser = _NOTIFICATIONS.get(parts[0])
    fields = parser(parts[1:]) if parser else None
    if fields is None:
        return Line(UNKNOWN, parts[0], raw=line)
    return Line(NOTIFICATION, parts[0], fields, raw=line)
