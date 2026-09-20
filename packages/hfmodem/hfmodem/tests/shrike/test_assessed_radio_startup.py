# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Real startup readback-only policy; all radio and process boundaries fake."""
import json
from pathlib import Path
import subprocess
import time
from unittest import mock
import pytest
from hfmodem.core import rxreadiness as rx
from hfmodem.shrike import onair, ota
from hfmodem.tests.shrike.test_radio_startup_flow import FakeRig
from hfmodem.tests.shrike.test_armdefaults import _parsed


class AssessedRig(FakeRig):
    def get_frequency_readback(self):
        self.calls.append(("get_frequency_readback",))
        return getattr(self, "frequency", 3595000)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    args = _parsed("--transmit", "--serial", "/mock/CAT", "--ptt-port", "/mock/PTT",
                   "--dial", "3595000", "--outdir", str(tmp_path/"child"),
                   "--p1-grant-only", "--tx-drive", "0.5", "--tx-latency-ms", "20")
    onair._arm_defaults(args)
    args.rx_assessment = str(tmp_path/"receipt.json")
    claim = {"device": 1, "inode": 2, "cat": args.serial}
    monkeypatch.setattr(rx, "verify_inherited_claim", lambda *a: claim)
    now = time.monotonic()
    record = {"schema": "rx-assessment-1", "assessment_complete": True,
              "claim": claim, "profile_sha256": rx.assessment_profile(args),
              "sources": rx.assessment_sources(), "issued_monotonic": now,
              "expires_monotonic": now+60,
              "actual": {"frequency_hz": 3595000, "mode": "PKTUSB", "passband_hz": 2400}}
    rig = AssessedRig()
    constructors = []
    monkeypatch.setattr(ota, "Rig", lambda *a, **k: constructors.append(True) or rig)
    monkeypatch.setattr(ota, "ptt_device", lambda *a, **k: "/mock/PTT")
    def forbidden(*a, **k):
        raise AssertionError("hardware/process access is forbidden")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(onair, "find_device", forbidden)
    monkeypatch.setattr(onair, "_LiveInput", forbidden)
    return args, record, rig, constructors


@pytest.mark.parametrize("case", ["valid", "frequency", "mode", "width", "expired", "source",
                                  "profile", "pending", "missing", "ownership", "query_error"])
def test_actual_run_assessed_policy(scene, monkeypatch, case):
    args, record, rig, constructors = scene
    if case == "frequency": rig.frequency = 3595001
    if case == "mode": rig.mode = "USB"
    if case == "width": record["actual"]["passband_hz"] = 3000
    if case == "expired": record["expires_monotonic"] = record["issued_monotonic"] - 1
    if case == "source": record["sources"] = {}
    if case == "profile": record["profile_sha256"] = "wrong"
    if case == "pending": record["assessment_complete"] = False
    if case == "ownership":
        monkeypatch.setattr(rx, "verify_inherited_claim", mock.Mock(side_effect=RuntimeError("claim changed")))
    if case == "query_error": rig.failure = "query"
    if case != "missing": Path(args.rx_assessment).write_text(json.dumps(record))
    original = rx.verify_assessed_receiver_startup
    reached = []
    class StopBeforeRF(Exception): pass
    def guard(*a, **k):
        result = original(*a, **k)
        reached.append(result)
        raise StopBeforeRF
    monkeypatch.setattr(rx, "verify_assessed_receiver_startup", guard)
    with pytest.raises(StopBeforeRF if case == "valid" else SystemExit):
        onair.run(args)
    assert not any(c[0] in ("set_freq", "set_mode") for c in rig.calls)
    if case == "valid":
        assert len(reached) == 1 and reached[0]["actual"] == record["actual"]
    if case in ("expired", "source", "profile", "pending", "missing", "ownership"):
        assert constructors == []


@pytest.mark.parametrize("stdout,rc", [("3595000\n",0),("3595000\n",1),("RPRT -1\n",0),
                                       ("",0),("3595000\nRPRT 0\n",0),("0\n",0)])
def test_strict_frequency_parse(stdout, rc):
    if stdout == "3595000\n" and rc == 0:
        assert rx.parse_frequency_readback(stdout, rc) == 3595000
    else:
        with pytest.raises(rx.ReceiverReadbackError): rx.parse_frequency_readback(stdout, rc)
