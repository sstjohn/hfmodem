# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""session_report / aggregate / scan / write_session."""

import json

from creance.metrics import SessionMetrics, compute, from_files
from creance.report import aggregate, scan, session_report, write_session
from hfhost.transcript import read

from test_metrics import META, write_clean


def test_session_report_labels(tmp_path, clock):
    m = compute(read(str(write_clean(tmp_path / "t.jsonl", clock)))[0], META)
    txt = session_report(m)
    assert "goodput_bps_far" in txt and "drain_bps_local" in txt
    assert "authoritative" in txt and "local modem drain" in txt
    assert "8.19 kbps" in txt        # far: 8192 B * 8 / 8.0 s
    assert "16.43 kbps" in txt       # local drain: 2 frames * 8 / 4.0 s
    assert "2.500 s" in txt          # connect latency
    assert "--  ok" in txt


def test_session_report_sim_time(tmp_path, clock):
    m = compute(read(str(write_clean(tmp_path / "t.jsonl", clock)))[0],
                dict(META, sim_time=True))
    txt = session_report(m)
    assert "sim_time" in txt and "withheld" in txt
    assert "kbps" not in txt


def _mk(**kw):
    base = dict(modem="kestrel", rev="r1", scenario="unidir",
                params={"size": "10k"}, wall_start="2026-07-19T10:00:00",
                outcome="ok")
    base.update(kw)
    return SessionMetrics(**base)


def test_aggregate():
    ms = [
        _mk(sid="a", goodput_bps_far=1000.0, connect_s=1.0),
        _mk(sid="b", goodput_bps_far=2000.0, connect_s=2.0),
        _mk(sid="c", outcome="failed:cwp_desync", connect_s=3.0),
        _mk(sid="d", modem="sabir", goodput_bps_far=5000.0),
    ]
    txt = aggregate(ms)
    rows = [ln for ln in txt.splitlines() if ln and not ln.startswith(("modem", "-"))]
    assert len(rows) == 2
    kestrel = next(r for r in rows if r.startswith("kestrel"))
    assert "67%" in kestrel                              # 2 of 3 ok
    assert "1.50 kbps" in kestrel                        # median goodput
    assert "1.00 kbps" in kestrel and "2.00 kbps" in kestrel
    assert "2.000 s" in kestrel                          # median connect
    sabir = next(r for r in rows if r.startswith("sabir"))
    assert "100%" in sabir and "5.00 kbps" in sabir


def test_aggregate_ok_column_uses_the_one_definition_of_a_good_outcome():
    # echo_peer is the kestrel loopback's success; counting only "ok" here
    # would read a wholly healthy loopback sweep as 0%
    ms = [_mk(sid="a", outcome="echo_peer"), _mk(sid="b", outcome="echo_peer")]
    row = next(ln for ln in aggregate(ms).splitlines()
               if ln.startswith("kestrel"))
    assert "100%" in row


def test_aggregate_size_buckets_are_normalized():
    # Same 10 KiB transfer, three ways of knowing its size: one row, not three.
    ms = [
        _mk(sid="a", params={"size": "10k"}, far_bytes=10240,
            goodput_bps_far=1000.0),
        _mk(sid="b", params={}, far_bytes=10240, goodput_bps_far=1000.0),
        _mk(sid="c", params={"size": "10K"}, bytes_tx=10240,
            goodput_bps_far=1000.0),
    ]
    rows = [ln for ln in aggregate(ms).splitlines()
            if ln and not ln.startswith(("modem", "-"))]
    assert len(rows) == 1
    assert " 3 " in f" {rows[0]} "


def test_aggregate_size_bucket_from_params_only():
    # A session that died before moving bytes still buckets by its plan.
    ms = [_mk(sid="a", params={"size": "10k"}, outcome="failed:connect_timeout"),
          _mk(sid="b", params={"size": "1k"}, outcome="failed:connect_timeout")]
    rows = [ln for ln in aggregate(ms).splitlines()
            if ln and not ln.startswith(("modem", "-"))]
    assert len(rows) == 2


def _make_tree(root, clock):
    for date, sid in (("2026-07-18", "s1"), ("2026-07-19", "s2")):
        sid_dir = root / "home" / date / sid
        write_clean(sid_dir / "transcript.jsonl", clock, sid=sid)
        meta = dict(META, sid=sid, wall_start=f"{date}T10:00:00")
        (sid_dir / "session.json").write_text(json.dumps({"meta": meta}))
    (root / "home" / "2026-07-19" / "broken").mkdir(parents=True)


def test_scan(tmp_path, clock, capsys):
    _make_tree(tmp_path, clock)
    ms = scan(tmp_path)
    assert sorted(m.sid for m in ms) == ["s1", "s2"]
    assert all(m.outcome == "ok" and m.warnings == [] for m in ms)
    assert "broken" in capsys.readouterr().err
    assert [m.sid for m in scan(tmp_path, since="2026-07-19")] == ["s2"]


def test_write_session_roundtrip(tmp_path, clock):
    sid_dir = tmp_path / "s1"
    write_clean(sid_dir / "transcript.jsonl", clock)
    m = compute(read(str(sid_dir / "transcript.jsonl"))[0], META)
    write_session(sid_dir, m)
    obj = json.loads((sid_dir / "session.json").read_text())
    assert obj["meta"]["sid"] == "s1" and obj["meta"]["outcome"] == "ok"
    assert obj["metrics"]["goodput_bps_far"] == m.goodput_bps_far
    m2 = from_files(sid_dir)
    assert m2.sid == m.sid and m2.goodput_bps_far == m.goodput_bps_far
    assert m2.warnings == []
