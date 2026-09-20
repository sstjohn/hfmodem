# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

# Adapted from an earlier session transcript writer by the same author.
"""JSONL session transcripts: thread-safe writer plus a reader for metrics.

One JSON object per line, append-only, line-buffered so a live session can be
tailed. Every record carries monotonic seconds since the session epoch, wall
time, the session id, the modem it concerns, and a monotone sequence number.
Data blocks are summarized: length, sha1 and the first 16 bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass


class Kind:
    CMD_TX = "cmd_tx"      # command string written to the cmd port
    CMD_RX = "cmd_rx"      # reply/notification read from the cmd port
    DATA_TX = "data_tx"    # payload bytes written to the data port
    DATA_RX = "data_rx"    # payload bytes read from the data port
    STATE = "state"        # connection/session state change
    PTT = "ptt"            # PTT ON/OFF notification from the modem
    HPROTO = "hproto"      # CWP event (frame sent/received, desync, fallback)
    CONF = "conf"          # conformance finding (check + verdict + evidence)
    NOTE = "note"          # free-form annotation from the harness
    ERROR = "error"        # something went wrong


@dataclass(frozen=True, slots=True)
class Record:
    seq: int
    t: float
    wall: str
    sid: str
    modem: str
    chan: str
    dir: str
    kind: str
    fields: dict


class Transcript:
    """JSONL writer. One instance per session, shared across threads."""

    def __init__(self, path: str, sid: str, *, epoch: float | None = None,
                 echo: bool = False) -> None:
        self.path = path
        self.sid = sid
        self.epoch = time.monotonic() if epoch is None else epoch
        self.echo = echo
        self._seq = 0
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # line-buffered so `tail -f` works during a live session
        self._fh = open(path, "a", buffering=1, encoding="utf-8")

    @property
    def closed(self) -> bool:
        return self._fh.closed

    def _emit(self, modem: str, chan: str, direction: str, kind: str,
              fields: dict) -> None:
        t = round(time.monotonic() - self.epoch, 6)
        wall = (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                + ".%03d" % int((time.time() % 1) * 1000))
        with self._lock:
            # session teardown closes this while reader threads are still
            # delivering late traffic; dropping those records is correct, and
            # raising would kill the reader mid-loop
            if self._fh.closed:
                return
            self._seq += 1
            record = {"seq": self._seq, "t": t, "wall": wall, "sid": self.sid,
                      "modem": modem, "chan": chan, "dir": direction,
                      "kind": kind, **fields}
            self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            if self.echo:
                detail = fields.get("text",
                                    "%d bytes sha1=%s" % (fields.get("len", 0),
                                                          fields.get("sha1", "")[:12])
                                    if "sha1" in fields else fields)
                print("[%8.3f] %-10s %-8s %s" % (t, modem, kind, detail))

    def cmd_tx(self, modem: str, text: str) -> None:
        self._emit(modem, "cmd", "tx", Kind.CMD_TX, {"text": text})

    def cmd_rx(self, modem: str, text: str) -> None:
        self._emit(modem, "cmd", "rx", Kind.CMD_RX, {"text": text})

    def data(self, modem: str, direction: str, blob: bytes, label: str = "") -> None:
        if self.closed:
            return                      # skip the hashing too
        fields = {"len": len(blob), "sha1": hashlib.sha1(blob).hexdigest(),
                  "head": blob[:16].hex(), "label": label}
        kind = Kind.DATA_TX if direction == "tx" else Kind.DATA_RX
        self._emit(modem, "data", direction, kind, fields)

    def state(self, modem: str, state: str, detail: str = "") -> None:
        self._emit(modem, "sys", "", Kind.STATE, {"text": state, "detail": detail})

    def ptt(self, modem: str, on: bool) -> None:
        self._emit(modem, "ptt", "rx", Kind.PTT, {"text": "ON" if on else "OFF"})

    def hproto(self, modem: str, event: str, **meta) -> None:
        self._emit(modem, "hproto", "", Kind.HPROTO, {"text": event, **meta})

    def conf(self, modem: str, check: str, verdict: str, **meta) -> None:
        self._emit(modem, "sys", "", Kind.CONF,
                   {"check": check, "verdict": verdict, **meta})

    def note(self, modem: str, text: str, **meta) -> None:
        self._emit(modem, "sys", "", Kind.NOTE, {"text": text, **meta})

    def error(self, modem: str, text: str, **meta) -> None:
        self._emit(modem, "sys", "", Kind.ERROR, {"text": text, **meta})

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


_ENVELOPE = ("seq", "t", "wall", "sid", "modem", "chan", "dir", "kind")


def read(path: str, *, tolerant: bool = False) -> tuple[list[Record], int]:
    """Parse a transcript into records plus a count of damaged lines;
    kind-specific payload lands in Record.fields. tolerant skips lines that
    will not parse — a crashed session leaves a torn tail, and metrics must
    still come out of what survived."""
    records: list[Record] = []
    damaged = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                env = {k: obj.pop(k, "" if k not in ("seq", "t") else 0)
                       for k in _ENVELOPE}
                records.append(Record(**env, fields=obj))
            except (ValueError, TypeError, AttributeError):
                if not tolerant:
                    raise
                damaged += 1
    return records, damaged
