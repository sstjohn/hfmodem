# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Actual startup dispatch and owned-CAT cleanup, entirely fake hardware."""
import json
import subprocess
from unittest import mock

import pytest

from hfmodem.core import rxreadiness as rx
from hfmodem.shrike import onair, ota
from hfmodem.tests.shrike.test_armdefaults import _parsed


class FakeRig:
    def __init__(self, *, failure=None, mode="PKTUSB", cleanup_failure=False):
        self.failure = failure
        self.mode = mode
        self.cleanup_failure = cleanup_failure
        self.calls = []

    def set_freq(self, value):
        self.calls.append(("set_freq", value))
        return True

    def get_freq(self):
        self.calls.append(("get_freq",))
        return "3595000"

    def set_mode(self, mode, passband=3000):
        self.calls.append(("set_mode", mode, passband))
        if self.failure == "write":
            return False
        if self.failure == "abort":
            raise KeyboardInterrupt("operator stop")
        return True

    def get_mode(self):
        self.calls.append(("get_mode",))
        if self.failure == "timeout":
            raise subprocess.TimeoutExpired("mock-getter", 8)
        if self.failure == "query":
            raise rx.ReceiverReadbackError("original query failure")
        if self.failure == "malformed":
            return rx.parse_mode_readback("PKTUSB\n0\n", 0)
        return rx.RadioMode(self.mode, 2400, self.mode + "\n2400\n")

    def _close(self):
        self.calls.append(("close",))
        if self.cleanup_failure:
            raise RuntimeError("secondary cleanup failure")

    def ptt(self, *_args, **_kwargs):
        raise AssertionError("RF is forbidden in this fixture")


@pytest.mark.parametrize("failure", ["write", "query", "timeout", "malformed", "abort"])
def test_failure_closes_only_owned_cat_and_never_keys(tmp_path, failure):
    rig = FakeRig(failure=failure)
    expected = KeyboardInterrupt if failure == "abort" else SystemExit
    with pytest.raises(expected):
        rx.verify_receiver_startup(rig, expected_mode="PKTUSB", model=1036,
                                   pactor1_only=False, output_dir=tmp_path)
    assert rig.calls[-1] == ("close",)
    assert json.loads((tmp_path / "radio-readback.json").read_text())["status"] == "refused"


def test_cleanup_failure_preserves_original_diagnostic(tmp_path):
    rig = FakeRig(failure="query", cleanup_failure=True)
    with pytest.raises(SystemExit, match="original query failure") as caught:
        rx.verify_receiver_startup(rig, expected_mode="PKTUSB", model=1036,
                                   pactor1_only=False, output_dir=tmp_path)
    assert "secondary cleanup failure" in caught.value.__cause__.__notes__[0]


def test_metadata_failure_closes_cat_before_rf(tmp_path, monkeypatch):
    rig = FakeRig()
    monkeypatch.setattr(rx.json, "dump", mock.Mock(side_effect=OSError("disk full")))
    with pytest.raises(SystemExit, match="disk full"):
        rx.verify_receiver_startup(rig, expected_mode="PKTUSB", model=1036,
                                   pactor1_only=False, output_dir=tmp_path)
    assert rig.calls[-1] == ("close",)


@pytest.mark.parametrize("mode,failure", [("PKTUSB", None), ("USB", None),
    ("PKTUSB", "write"), ("PKTUSB", "query"), ("PKTUSB", "timeout"),
    ("PKTUSB", "malformed"), ("PKTUSB", "abort")])
def test_real_startup_reaches_readback_before_any_rf(tmp_path, monkeypatch, mode, failure):
    args = _parsed("--transmit", "--serial", "/mock/CAT", "--ptt-port", "/mock/PTT",
                   "--dial", "3595000", "--outdir", str(tmp_path), "--p1-grant-only")
    rig = FakeRig(mode=mode, failure=failure)
    reached = []
    original = rx.verify_receiver_startup

    class StopBeforeRF(Exception):
        pass

    def guard(*pos, **kw):
        reached.append(True)
        original(*pos, **kw)
        raise StopBeforeRF("verified, deliberately stop before audio/RF")

    def forbidden(*pos, **kw):
        raise AssertionError("unexpected hardware/process access")

    monkeypatch.setattr(rx, "verify_receiver_startup", guard)
    monkeypatch.setattr(ota, "Rig", lambda *a, **k: rig)
    monkeypatch.setattr(ota, "ptt_device", lambda *a, **k: "/mock/PTT")
    monkeypatch.setattr(onair, "find_device", forbidden)
    monkeypatch.setattr(onair, "_LiveInput", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(onair.config, "tx_drive", lambda *a, **k: 0.5)
    monkeypatch.setattr(onair.config, "tx_latency_ms", lambda *a, **k: 20)
    expected = (KeyboardInterrupt if failure == "abort" else
                StopBeforeRF if mode == "PKTUSB" and failure is None else SystemExit)
    with pytest.raises(expected):
        onair.run(args)
    assert reached == [True]
    assert ("set_mode", "PKTUSB", 3000) in rig.calls
    assert (("get_mode",) in rig.calls) == (failure not in ("write", "abort"))
    report = json.loads((tmp_path / "radio-readback.json").read_text())
    if failure is None:
        assert report["actual"]["mode"] == mode
    else:
        assert report["actual"] is None
    if expected is not StopBeforeRF:
        assert rig.calls[-1] == ("close",)
    assert report["rf_authorized"] is False
