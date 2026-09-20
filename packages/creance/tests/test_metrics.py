# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""compute()/from_files() over transcripts written by the real writer."""

import json
from dataclasses import asdict

import pytest

from creance.metrics import compute, from_files
from hfhost.transcript import Transcript, read


SHA = "ab" * 32

META = {"sid": "s1", "site": "home", "modem": "kestrel",
        "version": "4.9.0.KestrelOpen", "rev": "r1", "label": "",
        "scenario": "unidir", "params": {"size": "2k", "payload": "prbs9"},
        "wall_start": "2026-07-19T10:00:00", "wall_end": "2026-07-19T10:00:11",
        "sim_time": False, "outcome": "ok"}


FRAME = 4096 + 11        # one full CWP DATA frame on the wire


def write_clean(path, clock, sid="s1"):
    """A unidir initiator transcript exactly as the real recorder writes one:
    the HELLO goes out as a labelled control data_tx, the peer takes 2.4 s to
    answer, then two payload frames drain over 4.0 s."""
    tr = Transcript(str(path), sid, epoch=0.0)
    c = clock
    c.now = 1.0; tr.cmd_tx("kestrel", "CONNECT K7ABC")
    c.now = 3.5; tr.cmd_rx("kestrel", "CONNECTED N0CAL K7ABC 2300")
    c.now = 4.0; tr.data("kestrel", "tx", b"\x00" * 90, label="cwp")   # HELLO
    c.now = 4.05; tr.hproto("kestrel", "hello_sent", scenario="unidir")
    c.now = 4.1; tr.cmd_rx("kestrel", "BUFFER 90")
    c.now = 4.2; tr.cmd_rx("kestrel", "BUFFER 0")
    c.now = 6.5; tr.hproto("kestrel", "hello_ack_rx", accept=True)
    c.now = 6.6; tr.data("kestrel", "tx", b"\x00" * FRAME, label="prbs9")
    c.now = 6.7; tr.ptt("kestrel", True)
    c.now = 6.8; tr.cmd_rx("kestrel", f"BUFFER {FRAME}")
    c.now = 7.0; tr.cmd_rx("kestrel", "BITRATE (3) 1200 bps TX")
    c.now = 7.1; tr.cmd_rx("kestrel", "SN 12.5")
    c.now = 8.0; tr.data("kestrel", "tx", b"\x01" * FRAME, label="prbs9")
    c.now = 8.2; tr.cmd_rx("kestrel", f"BUFFER {2 * FRAME}")
    c.now = 8.3; tr.data("kestrel", "tx", b"\x00" * 132, label="cwp")  # END
    c.now = 9.4; tr.ptt("kestrel", False)
    c.now = 10.6; tr.cmd_rx("kestrel", "BUFFER 0")
    c.now = 10.65; tr.hproto("kestrel", "end_sent", sha256=SHA, bytes=8192,
                             dur_s=1.7)
    c.now = 11.6; tr.hproto("kestrel", "report_rx", sid=sid, sha256=SHA,
                            bytes=8192, dur_s=8.0, match=True)
    c.now = 12.0; tr.cmd_tx("kestrel", "DISCONNECT")
    c.now = 13.0; tr.cmd_rx("kestrel", "DISCONNECTED")
    tr.close()
    return path


def test_clean_session(tmp_path, clock):
    recs = read(str(write_clean(tmp_path / "t.jsonl", clock)))[0]
    m = compute(recs, META)
    assert m.sid == "s1" and m.modem == "kestrel" and m.outcome == "ok"
    assert m.connect_s == 2.5
    assert m.handshake_s == 3.0
    assert m.disconnect_s == 1.0
    assert m.goodput_bps_far == 8192 * 8 / 8.0
    assert m.far_bytes == 8192 and m.far_dur_s == 8.0
    assert m.drain_window_s == 4.0                  # 6.6 (first payload) .. 10.6
    assert m.drain_bytes == 2 * FRAME               # control frames excluded
    assert m.drain_bps_local == 2 * FRAME * 8 / 4.0
    assert m.bytes_tx == 2 * FRAME + 90 + 132 and m.bytes_rx == 0
    assert m.ptt_keys == 1 and m.ptt_unpaired == 0
    assert m.ptt_duty == pytest.approx((9.4 - 6.7) / (13.0 - 3.5))
    assert m.buffer_curve == [[4.1, 90], [4.2, 0], [6.8, FRAME],
                              [8.2, 2 * FRAME], [10.6, 0]]
    assert m.bitrate_series == [[7.0, 3, 1200, "TX"]]
    assert m.sn_series == [[7.1, 12.5]]
    assert m.crc_failures == 0
    assert m.end_sha == SHA and m.report_sha == SHA
    assert m.end_sha_match is True and m.report_sha_match is True
    assert m.suppressed == [] and m.warnings == []
    json.dumps(asdict(m))  # must stay JSON-serializable


def test_drain_rate_is_the_true_rate(tmp_path, clock):
    """Ground truth: 8 KiB of payload leaves the buffer in 8.0 s, so the drain
    rate is 8192 bps — and the peer sat on the HELLO for 20 s first, which the
    window must not absorb."""
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "gt", epoch=0.0)
    c = clock
    c.now = 0.0; tr.cmd_rx("kestrel", "CONNECTED N0CAL K7ABC 2300")
    c.now = 1.0; tr.data("kestrel", "tx", b"\x00" * 90, label="cwp")
    c.now = 1.5; tr.cmd_rx("kestrel", "BUFFER 90")
    c.now = 2.0; tr.cmd_rx("kestrel", "BUFFER 0")
    c.now = 21.0; tr.hproto("kestrel", "hello_ack_rx", accept=True)
    c.now = 22.0; tr.data("kestrel", "tx", b"\x00" * 4096, label="prbs9")
    c.now = 22.5; tr.cmd_rx("kestrel", "BUFFER 4096")
    c.now = 26.0; tr.data("kestrel", "tx", b"\x00" * 4096, label="prbs9")
    c.now = 26.5; tr.cmd_rx("kestrel", "BUFFER 4096")
    c.now = 30.0; tr.cmd_rx("kestrel", "BUFFER 0")
    tr.close()
    m = compute(read(str(path))[0], {"sid": "gt", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.drain_window_s == 8.0
    assert m.drain_bps_local == pytest.approx(8192.0, rel=1e-9)
    # the pre-fix clock (first data_tx, i.e. the HELLO) would have said this:
    assert m.drain_bps_local != pytest.approx((8192 + 90) * 8 / 29.0)


def test_drain_ignores_control_only_traffic(tmp_path, clock):
    # A connect scenario writes HELLO and END and no payload: there is nothing
    # to rate, and inventing one from control frames is the bug.
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "c1", epoch=0.0)
    clock.now = 1.0; tr.data("kestrel", "tx", b"\x00" * 90, label="cwp")
    clock.now = 1.2; tr.cmd_rx("kestrel", "BUFFER 90")
    clock.now = 3.0; tr.cmd_rx("kestrel", "BUFFER 0")
    tr.close()
    m = compute(read(str(path))[0], {"sid": "c1", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.bytes_tx == 90
    assert m.drain_bps_local is None and m.drain_bytes is None
    assert m.warnings == []


def test_goodput_rate_is_the_true_rate(tmp_path, clock):
    # goodput_bps_far is the far end's own measurement, carried in the REPORT.
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "g1", epoch=0.0)
    clock.now = 1.0; tr.hproto("kestrel", "end_sent", sha256=SHA, bytes=10240,
                               dur_s=9.0)
    clock.now = 40.0; tr.hproto("kestrel", "report_rx", sid="g1", sha256=SHA,
                                bytes=10240, dur_s=64.0, match=True)
    tr.close()
    m = compute(read(str(path))[0], {"sid": "g1", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.goodput_bps_far == pytest.approx(1280.0)   # 10240 B in 64 s
    assert m.far_bytes == 10240 and m.far_dur_s == 64.0


def test_goodput_none_when_far_end_could_not_time_it(tmp_path, clock):
    # dur_s 0.0 is the far end saying "I had no honest window" (see
    # scenarios._rx_duration); a missing number beats a fabricated one.
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "g2", epoch=0.0)
    clock.now = 1.0; tr.hproto("kestrel", "report_rx", sid="g2", sha256=SHA,
                               bytes=1024, dur_s=0.0)
    tr.close()
    m = compute(read(str(path))[0], {"sid": "g2", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.goodput_bps_far is None
    assert m.far_bytes == 1024 and m.far_dur_s == 0.0
    assert any(w.startswith("goodput_bps_far unavailable") for w in m.warnings)


def test_bidir_sha_pairing_is_per_direction(tmp_path, clock):
    """bidir carries two independent payloads. Pairing our END (TX sha) with
    our own REPORT (RX sha) compares unrelated transfers and only looks sane
    when both legs happen to send the same bytes."""
    tx_sha, rx_sha = "aa" * 32, "bb" * 32
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "b1", epoch=0.0)
    c = clock
    c.now = 1.0; tr.hproto("kestrel", "end_sent", sha256=tx_sha, bytes=4096,
                           dur_s=4.0)
    c.now = 2.0; tr.hproto("kestrel", "end_rx", sha256=rx_sha, bytes=2048,
                           dur_s=2.0, match=True)
    c.now = 3.0; tr.hproto("kestrel", "report_sent", sid="b1", sha256=rx_sha,
                           bytes=2048, dur_s=2.5)
    c.now = 4.0; tr.hproto("kestrel", "report_rx", sid="b1", sha256=tx_sha,
                           bytes=4096, dur_s=5.0, match=True)
    tr.close()
    m = compute(read(str(path))[0], {"sid": "b1", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.end_sha == tx_sha and m.report_sha == tx_sha   # both our TX leg
    assert m.end_sha_match is True and m.report_sha_match is True
    assert m.far_bytes == 4096 and m.far_dur_s == 5.0
    assert m.goodput_bps_far == pytest.approx(4096 * 8 / 5.0)


def test_bidir_integrity_failure_is_not_masked(tmp_path, clock):
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "b2", epoch=0.0)
    c = clock
    c.now = 1.0; tr.hproto("kestrel", "end_sent", sha256="aa" * 32, bytes=4096,
                           dur_s=4.0)
    c.now = 2.0; tr.hproto("kestrel", "end_rx", sha256="bb" * 32, bytes=2048,
                           dur_s=2.0, match=True)
    c.now = 3.0; tr.hproto("kestrel", "report_sent", sid="b2", sha256="bb" * 32,
                           bytes=2048, dur_s=2.5)
    c.now = 4.0; tr.hproto("kestrel", "report_rx", sid="b2", sha256="cc" * 32,
                           bytes=4096, dur_s=5.0, match=False)
    tr.close()
    m = compute(read(str(path))[0], {"sid": "b2", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.end_sha_match is False and m.report_sha_match is False


def test_responder_side_pairs_its_own_report(tmp_path, clock):
    # A pure receive leg has no report_rx: the RX pair carries the truth.
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "r1", epoch=0.0)
    clock.now = 1.0; tr.hproto("kestrel", "hello_ack_sent", accept=True)
    clock.now = 9.0; tr.hproto("kestrel", "end_rx", sha256=SHA, bytes=4096,
                               dur_s=7.0, match=True)
    clock.now = 9.5; tr.hproto("kestrel", "report_sent", sid="r1", sha256=SHA,
                               bytes=4096, dur_s=8.0)
    tr.close()
    m = compute(read(str(path))[0], {"sid": "r1", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.end_sha == SHA and m.report_sha == SHA
    assert m.end_sha_match is True and m.report_sha_match is True
    assert m.goodput_bps_far == pytest.approx(4096 * 8 / 8.0)


def test_ptt_duty_uses_the_session_window(tmp_path, clock):
    # Idle before CONNECTED and the settle tail after DISCONNECTED must not
    # dilute the duty cycle.
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "d1", epoch=0.0)
    c = clock
    c.now = 0.0; tr.note("kestrel", "session_start")
    c.now = 100.0; tr.cmd_tx("kestrel", "CONNECT K7ABC")
    c.now = 110.0; tr.cmd_rx("kestrel", "CONNECTED N0CAL K7ABC 2300")
    c.now = 120.0; tr.ptt("kestrel", True)
    c.now = 130.0; tr.ptt("kestrel", False)
    c.now = 150.0; tr.cmd_tx("kestrel", "DISCONNECT")
    c.now = 160.0; tr.cmd_rx("kestrel", "DISCONNECTED")
    c.now = 300.0; tr.note("kestrel", "session_end")
    tr.close()
    m = compute(read(str(path))[0], {"sid": "d1", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.ptt_duty == pytest.approx(10.0 / 50.0)     # not 10/300


def test_ptt_duty_clips_keying_outside_the_session(tmp_path, clock):
    # A carrier that outlives DISCONNECTED belongs to no session; counting it
    # whole against the session window pushes duty over 100%.
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "d2", epoch=0.0)
    c = clock
    c.now = 10.0; tr.ptt("kestrel", True)               # keyed before CONNECTED
    c.now = 20.0; tr.cmd_rx("kestrel", "CONNECTED N0CAL K7ABC 2300")
    c.now = 25.0; tr.ptt("kestrel", False)
    c.now = 30.0; tr.ptt("kestrel", True)
    c.now = 40.0; tr.cmd_rx("kestrel", "DISCONNECTED")
    c.now = 60.0; tr.ptt("kestrel", False)              # and held after it
    tr.close()
    m = compute(read(str(path))[0], {"sid": "d2", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.ptt_keys == 2 and m.ptt_unpaired == 0
    assert m.ptt_duty == pytest.approx((25.0 - 20.0 + 40.0 - 30.0) / 20.0)
    assert m.ptt_duty <= 1.0


def test_buffer_zero_trap(tmp_path, clock):
    # Spurious BUFFER 0 before any rise, and a mid-transfer BUFFER 0 before
    # the last write: neither may stop the drain clock.
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "s3", epoch=0.0)
    c = clock
    c.now = 5.0; tr.data("kestrel", "tx", b"\x00" * 1024)
    c.now = 5.05; tr.cmd_rx("kestrel", "BUFFER 0")
    c.now = 5.2; tr.cmd_rx("kestrel", "BUFFER 1024")
    c.now = 5.8; tr.cmd_rx("kestrel", "BUFFER 0")
    c.now = 6.0; tr.data("kestrel", "tx", b"\x00" * 1024)
    c.now = 6.2; tr.cmd_rx("kestrel", "BUFFER 2048")
    c.now = 9.0; tr.cmd_rx("kestrel", "BUFFER 0")
    tr.close()
    m = compute(read(str(path))[0], {"sid": "s3", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.drain_window_s == 4.0
    assert m.drain_bps_local == 4096.0
    assert m.warnings == []


def test_drain_requires_rise(tmp_path, clock):
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "s4", epoch=0.0)
    clock.now = 5.0; tr.data("kestrel", "tx", b"\x00" * 512)
    clock.now = 6.0; tr.data("kestrel", "tx", b"\x00" * 512)
    clock.now = 9.0; tr.cmd_rx("kestrel", "BUFFER 0")
    tr.close()
    m = compute(read(str(path))[0], {"sid": "s4", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.drain_bps_local is None and m.drain_window_s is None
    assert any(w.startswith("drain_bps_local") for w in m.warnings)


def test_sim_time_suppression(tmp_path, clock):
    recs = read(str(write_clean(tmp_path / "t.jsonl", clock)))[0]
    m = compute(recs, dict(META, sim_time=True))
    assert m.suppressed == ["sim_time"]
    assert m.goodput_bps_far is None
    assert m.drain_bps_local is None and m.drain_window_s is None
    assert m.ptt_duty is None
    assert m.ptt_keys == 1
    assert m.far_bytes == 8192                     # raw counts survive
    assert m.bytes_tx == 2 * FRAME + 90 + 132
    assert m.warnings == []


def test_unpaired_ptt(tmp_path, clock):
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "s5", epoch=0.0)
    clock.now = 1.0; tr.cmd_rx("kestrel", "CONNECTED A B 2300")
    clock.now = 2.0; tr.ptt("kestrel", True)
    clock.now = 3.0; tr.ptt("kestrel", False)
    clock.now = 4.0; tr.ptt("kestrel", True)
    clock.now = 5.0; tr.cmd_rx("kestrel", "DISCONNECTED")
    tr.close()
    m = compute(read(str(path))[0], {"sid": "s5", "modem": "kestrel",
                                    "outcome": "ok"})
    assert m.ptt_keys == 2 and m.ptt_unpaired == 1
    assert m.ptt_duty == 0.25          # only the paired key counts
    assert any("unpaired" in w for w in m.warnings)


def test_desync_session(tmp_path, clock):
    path = tmp_path / "t.jsonl"
    tr = Transcript(str(path), "s9", epoch=0.0)
    clock.now = 1.0; tr.cmd_rx("kestrel", "CONNECTED N0CAL K7ABC 2300")
    clock.now = 2.0; tr.hproto("kestrel", "hello_ack_sent", accept=True)
    clock.now = 3.0; tr.data("kestrel", "rx", b"\xde\xad")
    clock.now = 3.1; tr.hproto("kestrel", "desync", reason="crc mismatch",
                               evidence="43524e31")
    clock.now = 3.1; tr.conf("kestrel", "cwp_desync", "FAIL",
                             evidence="crc mismatch")
    tr.close()
    recs = read(str(path))[0]
    m = compute(recs, {"sid": "s9", "modem": "kestrel",
                       "outcome": "failed:cwp_desync"})
    assert m.outcome == "failed:cwp_desync"
    assert m.crc_failures == 1
    assert m.handshake_s == 1.0
    assert m.bytes_rx == 2
    assert [f["verdict"] for f in m.findings] == ["FAIL"]
    # no recorded outcome: derived from the desync evidence
    m2 = compute(recs, {"sid": "s9", "modem": "kestrel"})
    assert m2.outcome == "failed:cwp_desync"
    # outcome/evidence disagreement is flagged
    m3 = compute(recs, {"sid": "s9", "modem": "kestrel", "outcome": "ok"})
    assert any("desync" in w for w in m3.warnings)


def test_truncated_transcript(tmp_path, clock):
    sid_dir = tmp_path / "s1"
    write_clean(sid_dir / "transcript.jsonl", clock)
    with open(sid_dir / "transcript.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"seq": 99, "t": 12.0, "wal')      # torn tail from a crash
    (sid_dir / "session.json").write_text(json.dumps({"meta": META}))
    m = from_files(sid_dir)
    assert m.connect_s == 2.5
    assert any("damaged" in w for w in m.warnings)


def test_missing_session_json(tmp_path, clock):
    sid_dir = tmp_path / "s1"
    write_clean(sid_dir / "transcript.jsonl", clock)
    m = from_files(sid_dir)
    assert m.sid == "s1"                              # falls back to the records
    assert any("session.json" in w for w in m.warnings)


def test_empty_input():
    m = compute([], {})
    assert m.connect_s is None and m.goodput_bps_far is None
    assert m.drain_bps_local is None and m.ptt_duty is None
    assert "empty transcript" in m.warnings
