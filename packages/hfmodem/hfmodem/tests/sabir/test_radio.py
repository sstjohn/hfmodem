# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import pytest

from hfmodem.core.claim import ClaimError
from hfmodem.sabir import radio
from hfmodem.tests import fakeclaim


@pytest.mark.parametrize("daemon", [None, "127.0.0.1:1"])
def test_cat_arguments_disable_serial_ptt(monkeypatch, daemon):
    monkeypatch.setattr(radio, "rigctl_path", lambda: "/fake/rigctl")
    rig = radio.Rig("ft891", "/dev/fake-ptt", profile=None, control=None,
                    serial="/dev/fake-cat", rigctld=daemon, log=lambda msg: None)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="7103500")

    class OneShot:
        returncode = 0

        def communicate(self, timeout):
            calls[-1][1]["timeout"] = timeout
            return b"7103500\n", b""

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(_process=OneShot())

    monkeypatch.setattr(radio.subprocess, "run", run)
    admissions = fakeclaim.install(monkeypatch, spawner=spawn)
    assert rig._run("F", "7103500", "f") == (0, "7103500")
    argv, kwargs = calls[0]
    if daemon:
        assert argv == ["/fake/rigctl", "-m", "2", "-r", daemon,
                        "F", "7103500", "f"]
    else:
        assert argv == ["/fake/rigctl", "-m", str(rig.model), "-r", "/dev/fake-cat",
                        "-s", str(rig.baud), "-P", "NONE", "F", "7103500", "f"]
        assert kwargs["claim_fd"] == 9 and "long_lived" not in kwargs
    assert kwargs["timeout"] <= 30
    assert not admissions


def test_a_refused_cat_admission_is_a_failed_command(monkeypatch):
    monkeypatch.setattr(radio, "rigctl_path", lambda: "/fake/rigctl")
    said = []
    rig = radio.Rig("ft891", "/dev/fake-ptt", profile=None, control=None,
                    serial="/dev/fake-cat", log=said.append)

    def refused(argv, **kwargs):
        raise ClaimError("cleanup in progress (pid 1)")

    fakeclaim.install(monkeypatch, spawner=refused)
    assert rig.qsy(7103500) is None
    assert said == ["  CAT refused: cleanup in progress (pid 1)"]
