# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Radio configuration gate: all CAT/process/PTT boundaries are fakes."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hfmodem.core import rxreadiness as rx
from hfmodem.shrike import ota


@pytest.mark.parametrize("width", [2400, 3000])
def test_ft891_accepts_documented_width_covering_profile(width):
    actual = rx.parse_mode_readback(f"PKTUSB\n{width}\n", 0)
    proof = rx.validate_mode_width(actual, expected_mode="PKTUSB", model=1036, span_hz=(380, 2620))
    assert proof["minimum_nominal_width_hz"] == 2240
    assert proof["full_receiver_verified"] is False


@pytest.mark.parametrize("text,code,err", [("", 0, ""), ("PKTUSB\n3000\n", 1, ""),
    ("PKTUSB\n3000\n", 0, "RPRT -4"), ("RPRT -4\n", 0, ""),
    ("PKTUSB\n0\n", 0, ""), ("PKTUSB\nnan\n", 0, ""),
    ("PKTUSB\n3000\nRPRT 0\n", 0, "")])
def test_missing_unsupported_failed_or_ambiguous_mode(text, code, err):
    with pytest.raises(rx.ReceiverReadbackError): rx.parse_mode_readback(text, code, err)


@pytest.mark.parametrize("mode,width", [("USB", 3000), ("PKTLSB", 3000),
                                      ("PKTUSB", 2000), ("PKTUSB", 2240)])
def test_bad_actual_mode_or_width(mode, width):
    with pytest.raises(rx.ReceiverReadbackError):
        rx.validate_mode_width(rx.RadioMode(mode, width, ""), expected_mode="PKTUSB",
                               model=1036, span_hz=(380, 2620))


def test_other_rig_not_restricted_to_ft891_width_steps():
    rx.validate_mode_width(rx.RadioMode("PKTUSB", 2800, ""), expected_mode="PKTUSB",
                           model=3088, span_hz=(380, 2620))


def fake_rig(monkeypatch, result=None, fail=None):
    rig = ota.Rig.__new__(ota.Rig)
    rig.rigctl = Path("/mock/rigctl"); rig.model = 1036
    rig.serial = "/mock/CAT"; rig.baud = 38400; rig.ptt_port = "/mock/PTT"
    calls = []
    monkeypatch.setattr(rig, "_close", lambda: calls.append("close"))
    def run(argv, **kwargs):
        calls.append(argv)
        assert calls[0] == "close" and kwargs["timeout"] == 8
        assert "-p" not in argv and "/mock/PTT" not in argv
        assert argv[-3:] == ["-P", "NONE", "m"]
        if fail: raise fail
        return result or SimpleNamespace(stdout="PKTUSB\n2400\n", stderr="", returncode=0)
    monkeypatch.setattr(ota.subprocess, "run", run)
    return rig, calls


def test_real_rig_getter_closes_owner_and_reads_cat_only(monkeypatch):
    rig, calls = fake_rig(monkeypatch)
    assert rig.get_mode().passband_hz == 2400
    assert len(calls) == 2


@pytest.mark.parametrize("fail", [OSError("gone"), subprocess.TimeoutExpired("m", 8)])
def test_real_getter_timeout_error_fails_closed(monkeypatch, fail):
    rig, _ = fake_rig(monkeypatch, fail=fail)
    with pytest.raises(rx.ReceiverReadbackError): rig.get_mode()


class StartupRig:
    def __init__(self, reply="PKTUSB\n2400\n", accepted=True):
        self.reply = reply; self.accepted = accepted; self.calls = []; self.keys = 0
    def set_mode(self, mode, passband):
        self.calls.append(("set", mode, passband)); return self.accepted
    def get_mode(self):
        self.calls.append(("read",)); return rx.parse_mode_readback(self.reply, 0)
    def ptt(self, keyed):
        self.keys += int(keyed)


@pytest.mark.parametrize("reply,accepted", [("USB\n3000\n", True), ("PKTUSB\n2000\n", True),
    ("RPRT -4\n", True), ("PKTUSB\n2400\n", False)])
def test_startup_refusal_prevents_following_rf_and_records_reason(tmp_path, reply, accepted):
    rig = StartupRig(reply, accepted)
    with pytest.raises(SystemExit, match="NOT KEYING"):
        rx.verify_receiver_startup(rig, expected_mode="PKTUSB", model=1036,
                                   pactor1_only=False, output_dir=tmp_path)
        rig.ptt(True)
    assert rig.keys == 0
    proof = json.loads((tmp_path / "radio-readback.json").read_text())
    assert proof["status"] == "refused" and proof["reason"]


def test_success_records_requested_actual_and_unknown_calibration(tmp_path):
    rig = StartupRig()
    proof = rx.verify_receiver_startup(rig, expected_mode="PKTUSB", model=1036,
                                       pactor1_only=False, output_dir=tmp_path)
    assert rig.calls == [("set", "PKTUSB", 3000), ("read",)] and rig.keys == 0
    assert proof["actual"]["passband_hz"] == 2400
    assert proof["requested"]["passband_hz"] == 3000
    assert all(value is None for value in proof["calibration"].values())
    assert proof["rf_authorized"] is False


def test_p1_only_uses_its_enforced_narrower_span(tmp_path):
    proof = rx.verify_receiver_startup(StartupRig("PKTUSB\n800\n"), expected_mode="PKTUSB",
                                      model=1036, pactor1_only=True, output_dir=tmp_path)
    assert proof["occupied_hz"] == [1200., 1800.]


def test_stale_output_refuses_before_cat_request(tmp_path):
    (tmp_path / "radio-readback.json").write_text("prior evidence")
    rig = StartupRig()
    with pytest.raises(SystemExit):
        rx.verify_receiver_startup(rig, expected_mode="PKTUSB", model=1036,
                                   pactor1_only=False, output_dir=tmp_path)
    assert rig.calls == [] and rig.keys == 0
