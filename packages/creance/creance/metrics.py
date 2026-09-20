# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Pure metrics: session transcript records in, SessionMetrics out.

The transcript is ground truth — any old session reprocesses for free when a
metric definition improves. compute() does no I/O; from_files() adds tolerant
loading of a session directory (transcript.jsonl + session.json).

Recorder contract, so the initiator/responder and these functions agree:
- session meta keys: META_KEYS below (report.write_session round-trips them),
  among them peer_sid/peer_call: the far end's session id and callsign, the
  only key that joins the two sites' records of one exchange — they share no
  clock, so wall times cannot;
- data_tx records carry label=CONTROL_LABEL for CWP control frames (HELLO,
  HELLO_ACK, END, REPORT) and the payload generator's name for payload; the
  drain measurement below depends on telling the two apart;
- hproto event texts name their direction: end_sent/report_sent are ours,
  end_rx/report_rx are the peer's, each carrying the frame body (sha256/
  bytes/dur_s, plus match=bool where the recorder verified that sha against
  bytes it held itself); "desync" marks a CWP framing/CRC defect. Without an
  explicit match, END-vs-REPORT sha equality stands in for both match fields.

goodput_bps_far and drain_bps_local are different measurements, never
interchangeable.

goodput_bps_far is the payload rate the *receiving* end measured over its own
clock and put in the REPORT: payload bytes divided by the window from the
completed handshake (HELLO_ACK) to the arrival of the last payload byte. That
window deliberately includes the sender's turnaround before it started
transmitting — a whole-exchange figure, conservative on half-duplex HF, and
the one a user experiences. It is None whenever the receiver could not time an
honest window (see scenarios._rx_duration).

drain_bps_local clocks only how fast the local modem drained our TX buffer:
payload frame bytes divided by the window from our first *payload* write to
the guarded terminal condition — a BUFFER 0 after both the last local write
and an observed BUFFER rise. Starting at the HELLO would fold the peer's
handshake turnaround into the window; the naive enqueue-to-BUFFER-0 clock
lies under TCP buffering. Both are None when the terminal condition was never
observed. Sessions tagged sim_time report no rate/duty figures at all:
wall-clock rates against simulated air are meaningless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from hfhost import wire
from hfhost.transcript import Kind, Record, read

META_KEYS = ("sid", "site", "modem", "version", "rev", "label", "scenario",
             "params", "peer_sid", "peer_call", "wall_start", "wall_end",
             "sim_time", "outcome")

CONTROL_LABEL = "cwp"     # data_tx label for CWP control frames, never payload
MIN_MEASURE_S = 1e-3      # shorter windows are clock noise, not a measurement


@dataclass
class SessionMetrics:
    sid: str = ""
    site: str = ""
    modem: str = ""
    version: str = ""
    rev: str = ""
    label: str = ""
    scenario: str = ""
    params: dict = field(default_factory=dict)
    peer_sid: str = ""
    peer_call: str = ""
    wall_start: str = ""
    wall_end: str = ""
    sim_time: bool = False
    outcome: str | None = None
    suppressed: list = field(default_factory=list)
    connect_s: float | None = None
    handshake_s: float | None = None
    disconnect_s: float | None = None
    goodput_bps_far: float | None = None
    far_bytes: int | None = None
    far_dur_s: float | None = None
    drain_bps_local: float | None = None
    drain_window_s: float | None = None
    drain_bytes: int | None = None
    bytes_tx: int = 0
    bytes_rx: int = 0
    ptt_keys: int = 0
    ptt_duty: float | None = None
    ptt_unpaired: int = 0
    buffer_curve: list = field(default_factory=list)      # [t, n]
    bitrate_series: list = field(default_factory=list)    # [t, level, bps, dir]
    sn_series: list = field(default_factory=list)         # [t, value]
    crc_failures: int = 0
    end_sha: str | None = None
    report_sha: str | None = None
    end_sha_match: bool | None = None
    report_sha_match: bool | None = None
    findings: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


# The structured dialect's message names, mapped onto the vocabulary the rest of
# this module already speaks. Normalizing here rather than teaching every
# consumer two dialects is the same call the Link seam makes: one body of logic,
# two front ends.
_ST_CONNECTED, _ST_DISCONNECTED = 3, 0
_STRUCTURED_CMD = {"Connect": "CONNECT", "Disconnect": "DISCONNECT",
                   "Abort": "ABORT", "Listen": "LISTEN"}


def _tx_verb(r: Record) -> str:
    """The command verb of a cmd_tx record, whichever dialect wrote it."""
    text = str(r.fields.get("text", ""))
    if isinstance(r.fields.get("fields"), dict):
        return _STRUCTURED_CMD.get(text, text.upper())
    head = text.split()[:1]
    return head[0] if head else ""


def _normalize(recs: list[Record]) -> dict[int, tuple[str, dict]]:
    """Received notifications as (name, fields), from either dialect.

    A structured record carries a `fields` dict beside its `text`; a VARA record
    carries only the raw line. PTT is reported by both but only the structured
    one distinguishes it from a channel-busy report, so the mapping is explicit
    rather than by name collision.
    """
    out: dict[int, tuple[str, dict]] = {}
    for i, r in enumerate(recs):
        if r.kind != Kind.CMD_RX:
            continue
        text = str(r.fields.get("text", ""))
        sf = r.fields.get("fields")
        if not isinstance(sf, dict):
            ln = wire.classify(text)
            if ln.kind == wire.NOTIFICATION:
                out[i] = (ln.name, ln.fields)
            continue
        if text == "StateChanged":
            state = sf.get("state")
            if state == _ST_CONNECTED:
                out[i] = ("CONNECTED", {"src": sf.get("peer_id"),
                                        "dst": None, "bw": None})
            elif state == _ST_DISCONNECTED:
                out[i] = ("DISCONNECTED", {})
        elif text == "LinkStats":
            if isinstance(sf.get("queue_bytes"), int):
                out[i] = ("BUFFER", {"n": sf["queue_bytes"]})
            # snr and throughput ride the same message; emit the richer one only
            # when there is no queue reading to lose
            elif isinstance(sf.get("snr3k_db"), (int, float)):
                out[i] = ("SN", {"value": sf["snr3k_db"]})
        elif text == "PhysicalState" and "ptt" in sf:
            out[i] = ("PTT", {"on": bool(sf["ptt"])})
    return out


def compute(records: Iterable[Record], session_meta: dict) -> SessionMetrics:
    meta = session_meta or {}
    recs = list(records)
    m = SessionMetrics(
        sid=str(meta.get("sid") or (recs[0].sid if recs else "")),
        site=str(meta.get("site") or ""),
        modem=str(meta.get("modem") or ""),
        version=str(meta.get("version") or ""),
        rev=str(meta.get("rev") or ""),
        label=str(meta.get("label") or ""),
        scenario=str(meta.get("scenario") or ""),
        params=dict(meta.get("params") or {}),
        peer_sid=str(meta.get("peer_sid") or ""),
        peer_call=str(meta.get("peer_call") or ""),
        wall_start=str(meta.get("wall_start") or ""),
        wall_end=str(meta.get("wall_end") or ""),
        sim_time=bool(meta.get("sim_time")),
        outcome=meta.get("outcome"),
    )
    if m.sim_time:
        m.suppressed.append("sim_time")
    if not recs:
        m.warnings.append("empty transcript")
    if m.modem:
        mine = [r for r in recs if r.modem == m.modem]
        if recs and not mine:
            m.warnings.append(f"no records for modem {m.modem!r}; using all records")
        else:
            recs = mine

    events = _normalize(recs)
    if not events and any(r.kind == Kind.CMD_RX for r in recs):
        m.warnings.append("no recognized notifications in the transcript; "
                          "link timings and buffer curve are unavailable")

    for i, (name, fields) in events.items():
        t = recs[i].t
        if name == "BUFFER":
            m.buffer_curve.append([t, fields["n"]])
        elif name == "BITRATE":
            m.bitrate_series.append(
                [t, fields["level"], fields["bps"], fields["direction"]])
        elif name == "SN":
            m.sn_series.append([t, fields["value"]])

    def notif_at(name: str, start: int) -> int | None:
        return next((i for i, (n, _) in events.items()
                     if i >= start and n == name), None)

    def cmd_at(verb: str) -> int | None:
        return next((i for i, r in enumerate(recs) if r.kind == Kind.CMD_TX
                     and _tx_verb(r) == verb), None)

    def hproto_at(prefix: str, start: int = 0) -> int | None:
        return next((i for i in range(start, len(recs))
                     if recs[i].kind == Kind.HPROTO
                     and str(recs[i].fields.get("text", "")).startswith(prefix)), None)

    i_connect = cmd_at("CONNECT")
    i_connected = notif_at("CONNECTED", i_connect + 1 if i_connect is not None else 0)
    if i_connect is not None and i_connected is not None:
        m.connect_s = round(recs[i_connected].t - recs[i_connect].t, 6)
    if i_connected is not None:
        i_ack = hproto_at("hello_ack", i_connected + 1)
        if i_ack is not None:
            m.handshake_s = round(recs[i_ack].t - recs[i_connected].t, 6)
    i_disconnect = cmd_at("DISCONNECT")
    i_disconnected = notif_at("DISCONNECTED", (i_disconnect or 0) + 1)
    if i_disconnect is not None and i_disconnected is not None:
        m.disconnect_s = round(recs[i_disconnected].t - recs[i_disconnect].t, 6)

    tx = [(r.t, _num(r.fields.get("len")) or 0.0,
           str(r.fields.get("label") or "")) for r in recs
          if r.kind == Kind.DATA_TX]
    m.bytes_tx = int(sum(n for _, n, _ in tx))
    m.bytes_rx = int(sum(_num(r.fields.get("len")) or 0.0
                         for r in recs if r.kind == Kind.DATA_RX))
    payload_tx = [(t, n) for t, n, label in tx if label != CONTROL_LABEL]

    if payload_tx and not m.sim_time:
        # The clock starts at the first payload write, not the HELLO: the
        # handshake turnaround is the peer's think time, not modem drain.
        t_first, t_last = payload_tx[0][0], tx[-1][0]
        prev, rise, t_zero = 0, False, None
        for t, n in m.buffer_curve:
            if t < t_first:
                prev = n            # carried-over buffer draining is not a rise
                continue
            if n > prev:
                rise = True
            prev = n
            if n == 0 and rise and t >= t_last:
                t_zero = t
                break
        if t_zero is not None and t_zero - t_first >= MIN_MEASURE_S:
            m.drain_window_s = round(t_zero - t_first, 6)
            # Payload frame bytes only, so the figure stays comparable with
            # goodput_bps_far; the control frames sharing the window (END,
            # REPORT) make this a slight under-read, never an over-read.
            m.drain_bytes = int(sum(n for _, n in payload_tx))
            if m.drain_bytes:
                m.drain_bps_local = m.drain_bytes * 8 / m.drain_window_s
        else:
            m.warnings.append(
                "drain_bps_local unavailable: no usable drain window (a BUFFER "
                "rise then BUFFER 0 after the last write, spanning more than "
                f"{MIN_MEASURE_S} s)")

    ptt = [(r.t, r.fields.get("text") == "ON") for r in recs if r.kind == Kind.PTT]
    keyed: list[tuple[float, float]] = []
    on_since: float | None = None
    for t, on in ptt:
        if on:
            m.ptt_keys += 1
            if on_since is not None:
                m.ptt_unpaired += 1
            on_since = t
        elif on_since is None:
            m.ptt_unpaired += 1
        else:
            keyed.append((on_since, t))
            on_since = None
    if on_since is not None:
        m.ptt_unpaired += 1
    if m.ptt_unpaired:
        m.warnings.append(f"{m.ptt_unpaired} unpaired PTT event(s)")
    # Duty over the session window, not the whole transcript: pre-HELLO idle
    # and the post-DISCONNECT settle tail would dilute it away. Keying is
    # clipped to that window too — a carrier held past DISCONNECTED belongs to
    # no session, and counting it whole is how a duty cycle exceeds 100%.
    t_open = recs[i_connected].t if i_connected is not None else (
        recs[0].t if recs else 0.0)
    t_close = recs[i_disconnected].t if i_disconnected is not None else (
        recs[-1].t if recs else 0.0)
    if keyed and t_close > t_open and not m.sim_time:
        on_time = sum(max(0.0, min(off, t_close) - max(on, t_open))
                      for on, off in keyed)
        m.ptt_duty = on_time / (t_close - t_open)

    # Pair END with REPORT within one direction. end_sent/report_rx describe
    # what we transmitted; end_rx/report_sent what we received. Crossing them
    # (the bidir trap) compares two unrelated payloads' sha256.
    i_rep_rx, i_rep_sent = hproto_at("report_rx"), hproto_at("report_sent")
    if i_rep_rx is not None or i_rep_sent is None:
        end_i, rep_i = hproto_at("end_sent"), i_rep_rx
    else:
        end_i, rep_i = hproto_at("end_rx"), i_rep_sent
    if end_i is None:
        end_i = hproto_at("end_local")          # echo_peer self-check
    end_ev = recs[end_i] if end_i is not None else None
    rep_ev = recs[rep_i] if rep_i is not None else None
    desyncs = sum(1 for r in recs if r.kind == Kind.HPROTO
                  and str(r.fields.get("text", "")).startswith("desync"))

    if end_ev is not None:
        m.end_sha = str(end_ev.fields.get("sha256") or "") or None
    if rep_ev is not None:
        m.report_sha = str(rep_ev.fields.get("sha256") or "") or None
        b = _num(rep_ev.fields.get("bytes"))
        d = _num(rep_ev.fields.get("dur_s"))
        m.far_bytes = int(b) if b is not None else None
        m.far_dur_s = d
        if not m.sim_time:
            if b and d is not None and d >= MIN_MEASURE_S:
                m.goodput_bps_far = b * 8 / d
            else:
                m.warnings.append(
                    "goodput_bps_far unavailable: REPORT carries no usable "
                    "bytes/dur_s (the far end could not time the receive leg)")

    def sha_match(ev: Record | None, other: str | None) -> bool | None:
        if ev is None:
            return None
        if "match" in ev.fields:
            return bool(ev.fields["match"])
        sha = str(ev.fields.get("sha256") or "")
        return sha == other if sha and other else None

    m.end_sha_match = sha_match(end_ev, m.report_sha)
    m.report_sha_match = sha_match(rep_ev, m.end_sha)

    m.findings = [{"t": r.t, "modem": r.modem, **r.fields}
                  for r in recs if r.kind == Kind.CONF]
    m.crc_failures = desyncs + sum(
        1 for f in m.findings
        if f.get("verdict") == "FAIL" and "crc" in str(f.get("check", "")).lower())

    if m.outcome is None:
        if desyncs:
            m.outcome = "failed:cwp_desync"
        else:
            m.warnings.append("outcome missing")
    elif m.outcome == "ok" and desyncs:
        m.warnings.append("outcome ok but desync events recorded")
    return m


def from_files(sid_dir: str | Path) -> SessionMetrics:
    """Load one results/<site>/<date>/<sid>/ directory. Tolerates truncated
    transcripts and missing/unreadable session.json (warnings, not errors);
    raises only when transcript.jsonl itself is absent."""
    sid_dir = Path(sid_dir)
    meta: dict = {}
    meta_warning = None
    session_json = sid_dir / "session.json"
    if session_json.exists():
        try:
            obj = json.loads(session_json.read_text(encoding="utf-8"))
            meta = obj.get("meta", obj) if isinstance(obj, dict) else {}
            if not isinstance(meta, dict):
                meta, meta_warning = {}, "session.json: meta is not an object"
        except ValueError:
            meta_warning = "session.json: unreadable JSON"
    else:
        meta_warning = "session.json missing"
    records, damaged = read(str(sid_dir / "transcript.jsonl"), tolerant=True)
    m = compute(records, meta)
    if damaged:
        m.warnings.append(f"transcript: skipped {damaged} damaged line(s)")
    if meta_warning:
        m.warnings.append(meta_warning)
    return m
