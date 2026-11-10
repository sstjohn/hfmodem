# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Text reporting over SessionMetrics: per-session one-pager, aggregate
table, results-tree scan, and the session.json writer."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median

from .metrics import META_KEYS, SessionMetrics, from_files
from .payloads import size_bytes

#: outcomes that count as a successful session, everywhere: the aggregate's
#: ok column, `run` and `campaign` all grade against this one set
GOOD_OUTCOMES = ("ok", "echo_peer")


def _bps(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v / 1000:.2f} kbps" if v >= 1000 else f"{v:.0f} bps"


def _secs(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f} s"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _yn(v: bool | None) -> str:
    return "-" if v is None else ("yes" if v else "no")


def session_report(m: SessionMetrics) -> str:
    out: list[str] = []

    def kv(key: str, val: str) -> None:
        out.append(f"  {key + ':':<20}{val}")

    def sec(name: str) -> None:
        out.append("")
        out.append(name)

    out.append(f"session {m.sid or '?'}  --  {m.outcome or 'outcome unknown'}")
    kv("site", m.site or "-")
    modem = m.modem or "-"
    if m.version:
        modem += f"  ({m.version})"
    tag = " ".join(x for x in (m.rev, m.label) if x)
    if tag:
        modem += f"  [{tag}]"
    kv("modem", modem)
    params = " ".join(f"{k}={v}" for k, v in (m.params or {}).items())
    kv("scenario", (m.scenario or "-") + (f"  {params}" if params else ""))
    kv("wall", f"{m.wall_start or '?'} .. {m.wall_end or '?'}")
    if m.suppressed:
        kv("suppressed", ", ".join(m.suppressed) + "  (rate/duty figures withheld)")

    sec("latencies")
    kv("connect", _secs(m.connect_s))
    kv("handshake", _secs(m.handshake_s))
    kv("disconnect", _secs(m.disconnect_s))

    sec("throughput")
    far = _bps(m.goodput_bps_far) + "  (far-end REPORT, authoritative"
    if m.far_bytes is not None and m.far_dur_s is not None:
        far += f": {m.far_bytes} B / {m.far_dur_s:g} s"
    kv("goodput_bps_far", far + ")")
    drain = _bps(m.drain_bps_local) + "  (local modem drain"
    if m.drain_window_s is not None:
        drain += f": {m.drain_bytes} B / {m.drain_window_s:g} s"
    kv("drain_bps_local", drain + ")")

    sec("ptt")
    kv("keys", str(m.ptt_keys))
    kv("duty", _pct(m.ptt_duty))
    if m.ptt_unpaired:
        kv("unpaired", str(m.ptt_unpaired))

    sec("integrity")
    kv("crc failures", str(m.crc_failures))
    kv("end sha match", _yn(m.end_sha_match))
    kv("report sha match", _yn(m.report_sha_match))

    if m.findings:
        sec(f"findings ({len(m.findings)})")
        for f in m.findings:
            extra = "  ".join(f"{k}={v}" for k, v in f.items()
                              if k not in ("t", "modem", "check", "verdict"))
            out.append(f"  [{f.get('verdict', '?')}] {f.get('check', '?')}"
                       + (f"  {extra}" if extra else ""))
    if m.warnings:
        sec(f"warnings ({len(m.warnings)})")
        out.extend(f"  {w}" for w in m.warnings)
    return "\n".join(out) + "\n"


def table(rows: list[tuple[str, ...]]) -> str:
    """Column-aligned text table; the first row is the header, and a rule goes
    under it. No trailing newline."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
             for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _tag(m: SessionMetrics) -> str:
    return " ".join(x for x in (m.rev, m.label) if x) or "-"


def _day(m: SessionMetrics) -> str:
    return m.wall_start[:10] if m.wall_start else "-"


def _size_bucket(m: SessionMetrics) -> str:
    # Always bucket by byte count, never by the raw param string: "10k" and a
    # measured 10240 must land in one row, or a size splits the aggregate in
    # two whenever some sessions carry params and others do not.
    b = m.far_bytes or m.bytes_tx or m.bytes_rx
    if not b:
        size = (m.params or {}).get("size")
        try:
            b = size_bytes(size) if size else 0
        except (TypeError, ValueError):
            b = 0
    if not b:
        return "-"
    kib = b / 1024
    if kib < 3:
        return "1K"
    if kib < 30:
        return "10K"
    if kib < 300:
        return "100K"
    return "1M+"


def aggregate(metrics: list[SessionMetrics]) -> str:
    groups: dict[tuple, list[SessionMetrics]] = {}
    for m in metrics:
        key = (m.modem or "-", _tag(m), m.scenario or "-", _size_bucket(m), _day(m))
        groups.setdefault(key, []).append(m)

    header = ("modem", "rev/label", "scenario", "size", "day", "n", "ok",
              "gp med", "gp min", "gp max", "connect", "duty", "findings")
    rows = [header]
    for key in sorted(groups):
        ms = groups[key]
        n = len(ms)
        ok = sum(1 for m in ms if m.outcome in GOOD_OUTCOMES)
        gps = sorted(m.goodput_bps_far for m in ms if m.goodput_bps_far is not None)
        conns = [m.connect_s for m in ms if m.connect_s is not None]
        duties = [m.ptt_duty for m in ms if m.ptt_duty is not None]
        finds = sum(1 for m in ms for f in m.findings if f.get("verdict") != "PASS")
        rows.append(key + (
            str(n), f"{round(100 * ok / n)}%",
            _bps(median(gps)) if gps else "-",
            _bps(gps[0]) if gps else "-",
            _bps(gps[-1]) if gps else "-",
            _secs(median(conns)) if conns else "-",
            _pct(median(duties)) if duties else "-",
            str(finds)))
    return table(rows) + "\n"


def scan(results_dir: str | Path, since: str | None = None) -> list[SessionMetrics]:
    """Walk results/<site>/<date>/<sid>/, loading each session; damaged
    directories are skipped with a note on stderr. since filters by the
    date directory name (YYYY-MM-DD, inclusive).

    `creance conform` writes its transcripts into the same tree under
    ``conform-<timestamp>/`` (cli.py). Those are probe runs, not sessions: no
    modem, no scenario, no outcome. Counting them produced an aggregate row
    reading "3 sessions, 0% ok" when nothing had failed — a report that
    manufactures its own bad news is worse than one that omits them."""
    out: list[SessionMetrics] = []
    root = Path(results_dir)
    if not root.is_dir():
        return out
    for site in sorted(p for p in root.iterdir() if p.is_dir()):
        for date in sorted(p for p in site.iterdir() if p.is_dir()):
            if since and date.name < since:
                continue
            for sid in sorted(p for p in date.iterdir() if p.is_dir()):
                if sid.name.startswith("conform-"):
                    continue
                try:
                    out.append(from_files(sid))
                except Exception as exc:
                    print(f"creance: skipping {sid}: {exc}", file=sys.stderr)
    return out


def write_session(sid_dir: str | Path, m: SessionMetrics) -> Path:
    """Write session.json (meta + full metrics) into the session directory."""
    d = asdict(m)
    obj = {"meta": {k: d[k] for k in META_KEYS}, "metrics": d}
    path = Path(sid_dir) / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return path
