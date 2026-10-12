# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`--after-connect p1-robust`: N9VC's caller's entry, keyed by the session loop.

On both N9VC links on tape the caller connects in PACTOR-1, keys one P1 data
packet, and on the peer's CS1 keys robust shorts in its own next ISS slot."""
from __future__ import annotations

import contextlib
import io
import json
import sys

import numpy as np

from hfmodem.shrike import onair, pactor1, rxfront, spec

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
PACKET_N = round(spec.P1_PACKET_S * FS)


def _peer_wav(path, slots=24):
    out = np.zeros((slots + 2) * SLOT_N, np.float32)
    for k in range(slots):
        cs = (pactor1.CS_SPEED if k == 0 else
              pactor1.CS_ACK_A if k % 2 else pactor1.CS_ACK_B)
        rf = onair._trim_silence(
            np.asarray(pactor1.control_signal(cs, invert=k % 2), np.float32))
        at = k * SLOT_N + PACKET_N + round(0.090 * FS)
        out[at:at + rf.size] += rf
    onair.session.write_wav(str(path), out)


def test_the_first_acked_p1_packet_starts_the_robust_shorts(tmp_path, monkeypatch):
    wav = tmp_path / "peer.wav"
    _peer_wav(wav)
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
        "--mycall", "W9SSJ", "--dxcall", "N9VC", "--dial", "7100910",
        "--outdir", str(tmp_path / "out"), "--after-connect", "p1-robust",
        "--p1-status-bits45", "2", "--p4-probe-values", "7,38",
        "--p4-probe-headers", "A,A", "--p4-probe-cadence", "consecutive"])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    log = log.getvalue()
    aired = [(slot, what) for slot, what in keyed if not what.startswith("connect")]
    assert [what.split()[0] for _, what in aired[:3]] == ["P1", "P4", "P4"], log
    assert "first with bits 4-5 set: status 0x21" in log, log
    assert "P4 PROBE ROBUST 1/2 v=7 hdr=A" in aired[1][1], aired
    assert "P4 PROBE ROBUST 2/2 v=38 hdr=A" in aired[2][1], aired
    assert len(aired) == 3, aired
    probe = json.loads((tmp_path / "out" / "p4-probe.json").read_text())
    assert probe["trigger_word"] == "P1 ACK"
    assert [e["slot"] for e in probe["emissions"]] == [aired[0][0] + 1, aired[0][0] + 2], (
        keyed, log)
