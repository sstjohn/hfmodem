# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`--p1-robust-stall`: a first P1 packet that only draws repeat requests.

KB5LZK twice and VE3KPG once answered our first P1 packet with CS1 every cycle
and never acknowledged it. Recorded after our teardown, KB5LZK asked on, and
then with us silent left P1 on its own, keying Y27 once a cycle about 0.89 s
after our boundary. The probe goes silent on the stall, waits for that word,
and answers it."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys

import numpy as np

from hfmodem.shrike import onair, p4oracle, p4robusttx, pactor1
from hfmodem.tests.shrike.test_n9vc_links import _load
from hfmodem.tests.shrike.test_p1_robust_0926 import FS, PACKET_N, SLOT_N

STALL, SILENT, WORDS = 4, 3, 8
WORD_AT_S = 0.891


def _stalled_peer_wav(path, requests):
    meta, x, fs = _load("ve3kpg-0926-1228-answers")
    assert fs == FS
    lead = round(0.020 * FS)
    at = round(meta["words"][1]["at_s"] * FS) - lead
    y27 = x[at:at + round(0.300 * FS)]
    slots = requests + SILENT + WORDS + 8
    out = np.zeros(slots * SLOT_N, np.float32)
    for k in range(requests + 1):
        rf = onair._trim_silence(np.asarray(
            pactor1.control_signal(pactor1.CS_ACK_A, invert=k % 2), np.float32))
        at = k * SLOT_N + PACKET_N + round(0.090 * FS)
        out[at:at + rf.size] += rf
    for k in range(requests + 1 + SILENT, requests + 1 + SILENT + WORDS):
        at = k * SLOT_N + round(WORD_AT_S * FS) - lead
        out[at:at + y27.size] += y27
    onair.session.write_wav(str(path), out)


def _stalled_arm(tmp_path, monkeypatch, requests, *extra, ends=None):
    tmp_path.mkdir(exist_ok=True)
    wav = tmp_path / "peer.wav"
    _stalled_peer_wav(wav, requests)
    keyed, key = [], onair.RadioTx._tx

    def tx(self, audio, what, *a, **k):
        before = len(self.slots_used)
        key(self, audio, what, *a, **k)
        if len(self.slots_used) > before:
            keyed.append((self.slot, what))
            if ends is not None and not what.startswith("connect"):
                ends.append(self.tx_end)

    monkeypatch.setattr(onair.RadioTx, "_tx", tx)
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", [
        "shrike.onair", "--replay", str(wav), "--max-cycles", "3",
        "--mycall", "W9SSJ", "--dxcall", "KB5LZK", "--dial", "7101500",
        "--outdir", str(tmp_path / "out"), "--after-connect", "p1-robust",
        "--p1-status-bits45", "2", "--p4-probe-values", "7,38",
        "--p4-probe-headers", "A,A", "--p4-probe-cadence", "consecutive",
        "--p1-robust-stall", str(STALL), "--p4-probe-reply", "X,X,X",
        "--p4-probe-listen", "2", "--p4-entry-timeout", "90", *extra])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    log = log.getvalue()
    aired = [(slot, what) for slot, what in keyed if not what.startswith("connect")]
    p1 = [what for _, what in aired if what.startswith("P1 ")]
    assert p1 and all(what.startswith("P1 pkt#1") for what in p1), aired
    replies = aired[len(p1):]
    assert [what.split(" (")[0] for _, what in replies] == [
        "P4 PROBE REPLY 1/3 X27", "P4 PROBE REPLY 2/3 X27", "P4 PROBE REPLY 3/3 X27"], log
    assert f"drew {STALL} repeat requests" in log, log
    probe = json.loads((tmp_path / "out" / "p4-probe.json").read_text())
    return aired, p1, log, probe


def test_a_stalled_first_packet_falls_silent_and_answers_the_peers_word(
        tmp_path, monkeypatch):
    aired, p1, log, probe = _stalled_arm(tmp_path, monkeypatch, 6)
    assert probe["path"] == "stall"
    assert probe["trigger_word"] == f"P1 stall: {STALL} repeat requests"
    first = probe["heard"][0]
    assert first["phase"] == "silent listen" and first["word"] == "Y27", probe["heard"]
    assert abs(first["after_boundary_s"] - WORD_AT_S) < 0.03, first
    assert f"heard Y27 in slot {first['slot']}" in log, log
    # Nothing between our last packet 1 and the peer's word; then the replies in
    # the first slot the clock still allows, one per slot.
    slots = [e["slot"] for e in probe["emissions"]]
    assert aired[len(p1) - 1][0] < first["slot"] < slots[0], (aired, first)
    assert slots == list(range(first["slot"] + 2, first["slot"] + 5)), (slots, first)
    x27 = hashlib.sha256(p4robusttx.robust_word("X27").tobytes()).hexdigest()
    assert all(e["waveform_sha256"] == x27 and e["word"] == "X27"
               for e in probe["emissions"])
    listened = [w for w in probe["windows"] if w["kind"] == "stall-listen"]
    assert listened and listened[0]["end"] >= first["onset_sample"], probe["windows"]


def test_the_goodbye_form_keys_the_holds_teardown_before_the_listen(tmp_path, monkeypatch):
    """The idle hold's goodbye as 0926-1237 flew it: the QRT queues behind the
    unacknowledged packet 1, so its cycles air packet 1 again."""
    silent = _stalled_arm(tmp_path / "silent", monkeypatch, 14)[1]
    ends = []
    aired, p1, log, probe = _stalled_arm(tmp_path / "goodbye", monkeypatch, 14,
                                         "--p1-robust-stall-then", "goodbye", ends=ends)
    assert probe["path"] == "stall after goodbye"
    # The stall's own cycle and QRT_CYCLES after it, as the idle hold keys its
    # deadline cycle and four more: 0926-1308's hold 12 aired 17.
    assert len(p1) == len(silent) + 1 + onair.QRT_CYCLES, (silent, p1)
    assert "the goodbye was unacknowledged" in log, log
    first = probe["heard"][0]
    assert first["phase"] == "silent listen" and aired[len(p1) - 1][0] < first["slot"]
    # The deadline runs from the end of the teardown's last packet.
    assert probe["grant_sample"] >= ends[len(p1) - 1], (probe["grant_sample"], ends)


HOLD_Y27, HOLD_CS1 = "kb5lzk-0926-1342-hold-y27", "ve3kpg-0926-1409-hold-cs1"


def _pieces(name):
    meta, x, fs = _load(name)
    return meta, np.split(x, len(meta["slots"])), fs


def test_a_word_behind_our_carrier_is_read_by_its_tail():
    """1342: KB5LZK keyed Y27 while we still keyed packet 1, and the hold's
    window past our mute holds only its tail, which the scan cannot see."""
    meta, pieces, fs = _pieces(HOLD_Y27)
    for piece in pieces:
        assert p4oracle.word_scan(piece, fs) == []
        hit, = p4oracle.word_tail(piece, fs)
        assert hit.word == "Y27" and hit.score > 0.99, hit
        onset = meta["cut_s"][0] + (hit.first_chip / FS)
        assert abs(onset - meta["onset_after_boundary_s"]) < 0.01, onset
    meta, x, fs = _load("ve3kpg-0926-1228-answers")
    for word in meta["words"]:
        at = round((word["at_s"] - 0.05) * fs)
        hit, = p4oracle.word_tail(x[at:at + round(0.35 * fs)], fs)
        assert hit.word == word["word"], (hit, word)
        assert abs((hit.first_chip / FS) - 0.05) < 0.01, hit


def test_the_tail_reader_names_nothing_in_repeat_requests():
    """1409: VE3KPG's CS1 in the same window, six holds of it."""
    _, pieces, fs = _pieces(HOLD_CS1)
    assert [p4oracle.word_tail(piece, fs) for piece in pieces] == [[]] * len(pieces)
    noise = np.random.default_rng(1409).standard_normal(20 * fs).astype(np.float32)
    assert all(p4oracle.word_tail(noise[a:a + round(0.3 * fs)], fs) == []
               for a in range(0, noise.size - fs, fs // 4))


def test_a_word_heard_in_the_hold_goes_straight_to_the_replies(tmp_path, monkeypatch):
    requests, words = 3, 4
    _, pieces, fs = _pieces(HOLD_Y27)
    assert fs == FS
    out = np.zeros((requests + words + 12) * SLOT_N, np.float32)
    for k in range(requests + 1):
        rf = onair._trim_silence(np.asarray(
            pactor1.control_signal(pactor1.CS_ACK_A, invert=k % 2), np.float32))
        at = k * SLOT_N + PACKET_N + round(0.090 * FS)
        out[at:at + rf.size] += rf
    for j, k in enumerate(range(requests + 1, requests + 1 + words)):
        piece = pieces[j % len(pieces)]
        at = k * SLOT_N + round(0.90 * FS)
        out[at:at + piece.size] += piece
    wav = tmp_path / "peer.wav"
    onair.session.write_wav(str(wav), out)
    keyed, key = [], onair.RadioTx._tx

    def tx(self, audio, what, *a, **k):
        before = len(self.slots_used)
        key(self, audio, what, *a, **k)
        if len(self.slots_used) > before:
            keyed.append((self.slot, what))

    monkeypatch.setattr(onair.RadioTx, "_tx", tx)
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", [
        "shrike.onair", "--replay", str(wav), "--max-cycles", "3",
        "--mycall", "W9SSJ", "--dxcall", "KB5LZK", "--dial", "10144900",
        "--outdir", str(tmp_path / "out"), "--after-connect", "p1-robust",
        "--p1-status-bits45", "2", "--p4-probe-values", "7,38",
        "--p4-probe-headers", "A,A", "--p4-probe-cadence", "consecutive",
        "--p1-robust-stall", "12", "--p4-probe-reply", "X,X,X",
        "--p4-probe-listen", "2", "--p4-entry-timeout", "90"])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    log = log.getvalue()
    probe = json.loads((tmp_path / "out" / "p4-probe.json").read_text())
    assert probe["path"] == "hold word", log
    first = probe["heard"][0]
    assert first["phase"] == "hold" and first["word"] == "Y27", probe["heard"]
    assert abs(first["after_boundary_s"] - 0.918) < 0.03, first
    aired = [(slot, what) for slot, what in keyed if not what.startswith("connect")]
    replies = [(slot, what.split(" (")[0]) for slot, what in aired
               if what.startswith("P4 PROBE")]
    assert [what for _, what in replies] == [
        "P4 PROBE REPLY 1/3 X27", "P4 PROBE REPLY 2/3 X27", "P4 PROBE REPLY 3/3 X27"], log
    assert all(what.startswith("P1 pkt#1") for _, what in aired[:-3]), aired
    assert replies[0][0] > first["slot"], (replies, first)
    assert "drew 12 repeat requests" not in log, log
