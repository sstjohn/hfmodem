# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Transcript writer/reader round-trip, seq monotonicity, data summaries."""

import hashlib

import pytest

from hfhost import transcript
from hfhost.transcript import Kind, Transcript


def write_sample(path, **kw):
    tr = Transcript(str(path), "sid-1", **kw)
    tr.cmd_tx("kestrel", "MYCALL N0CAL")
    tr.cmd_rx("kestrel", "OK")
    tr.data("kestrel", "tx", b"hello CWP world!!", label="prbs9")
    tr.data("kestrel", "rx", b"echo")
    tr.state("kestrel", "connected", "CONNECTED N0CAL K7ABC 2300")
    tr.ptt("kestrel", True)
    tr.hproto("kestrel", "hello_sent", scenario="unidir")
    tr.conf("kestrel", "buffer_cadence", "PASS", evidence="BUFFER 0")
    tr.note("kestrel", "session start")
    tr.error("kestrel", "boom", detail="test")
    tr.close()
    return tr


def test_round_trip_and_seq(tmp_path):
    path = tmp_path / "t.jsonl"
    write_sample(path)
    records, _ = transcript.read(str(path))
    assert [r.seq for r in records] == list(range(1, 11))
    assert all(r.sid == "sid-1" and r.modem == "kestrel" for r in records)
    assert [r.kind for r in records] == [
        Kind.CMD_TX, Kind.CMD_RX, Kind.DATA_TX, Kind.DATA_RX, Kind.STATE,
        Kind.PTT, Kind.HPROTO, Kind.CONF, Kind.NOTE, Kind.ERROR]
    ts = [r.t for r in records]
    assert ts == sorted(ts)
    assert records[0].fields == {"text": "MYCALL N0CAL"}
    assert records[6].fields == {"text": "hello_sent", "scenario": "unidir"}
    assert records[7].fields == {"check": "buffer_cadence", "verdict": "PASS",
                                 "evidence": "BUFFER 0"}


def test_data_summarized(tmp_path):
    path = tmp_path / "t.jsonl"
    write_sample(path)
    rec = next(r for r in transcript.read(str(path))[0] if r.kind == Kind.DATA_TX)
    blob = b"hello CWP world!!"
    assert rec.fields["len"] == len(blob)
    assert rec.fields["sha1"] == hashlib.sha1(blob).hexdigest()
    assert rec.fields["head"] == blob[:16].hex()
    assert rec.fields["label"] == "prbs9"


def test_tolerant_read_counts_a_torn_tail(tmp_path):
    path = tmp_path / "t.jsonl"
    write_sample(path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"seq": 99, "t": 1.0, "wal')       # crashed mid-write
    with pytest.raises(ValueError):
        transcript.read(str(path))
    records, damaged = transcript.read(str(path), tolerant=True)
    assert len(records) == 10 and damaged == 1


def test_writes_after_close_are_dropped_not_fatal(tmp_path):
    # the responder closes a session transcript while modem reader threads are
    # still delivering late traffic into it
    path = tmp_path / "t.jsonl"
    tr = write_sample(path)
    before = path.read_text()
    tr.cmd_rx("kestrel", "DISCONNECTED")
    tr.data("kestrel", "rx", b"late echo")
    tr.state("kestrel", "detached")
    tr.ptt("kestrel", False)
    tr.hproto("kestrel", "late")
    tr.conf("kestrel", "x", "PASS")
    tr.note("kestrel", "late")
    tr.error("kestrel", "late")
    tr.close()
    assert tr.closed
    assert path.read_text() == before


def test_append_continues_seq(tmp_path):
    path = tmp_path / "t.jsonl"
    write_sample(path)
    tr = Transcript(str(path), "sid-2")
    tr.note("harrier-A", "second writer")
    tr.close()
    records, _ = transcript.read(str(path))
    assert len(records) == 11
    assert records[-1].sid == "sid-2" and records[-1].modem == "harrier-A"
