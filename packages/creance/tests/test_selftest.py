# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The M1 gate as a pytest: same stages as `creance selftest`, smaller payload.

Skipped wholesale when neither target imports here; the selftest itself skips
whichever half is missing, so a machine with only kestrel still gates on it.

Set ``CREANCE_REQUIRE=kestrel,sabir`` on a bench that is supposed to have both
and a target that stops resolving fails the gate instead of quietly reducing it.
The list cannot be derived from ``have_kestrel()``/``have_sabir()`` — that is the
same predicate that goes False when a path moves, so it would excuse exactly the
outage it is meant to catch. It is an assertion about the machine, kept outside
the code that checks it, and unset on a clean clone so hermeticity survives.
"""

import os
import hashlib
from pathlib import Path
import shutil
import stat

import pytest

from creance import selftest


def _finish_qualification_selftest(source: Path, observations: Path,
                                   failed: bool) -> None:
    """Preserve a failed M1 run before removing its temporary work tree."""
    if not failed:
        shutil.rmtree(source)
        return
    try:
        if source.is_symlink() or not source.is_dir():
            raise ValueError("M1 work tree is not a regular directory")
        files = []
        for path in sorted(source.rglob("*")):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"M1 work tree contains a nonregular entry: {path}")
            files.append(path)
        if observations.is_symlink() or not observations.is_dir():
            raise ValueError("qualification observations is not a regular directory")
        retained = observations / "creance-m1"
        retained.mkdir()
        for path in files:
            target = retained / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(
                    target.read_bytes()).digest():
                raise ValueError(f"M1 evidence copy changed: {path}")
    except Exception:
        print(f"creance M1 original failure evidence: {source}")
        raise
    print(f"creance M1 failure evidence: {retained}")
    shutil.rmtree(source)


@pytest.mark.skipif(not (selftest.have_kestrel() or selftest.have_sabir()),
                    reason="neither the kestrel loopback nor the sabir server "
                           "is importable")
def test_m1_gate(capsys, tmp_path):
    require = os.environ.get("CREANCE_REQUIRE", "").split(",")
    qualification_observations = os.environ.get("HFMODEM_QUALIFICATION_OBSERVATIONS")
    observations = (Path(qualification_observations) if qualification_observations
                    else tmp_path / "creance-observations")
    if not qualification_observations:
        observations.mkdir()
    run = selftest.Selftest(size="2k", require=require, keep=True)
    rc = 1
    try:
        rc = run.run()
    finally:
        _finish_qualification_selftest(run.tmp, observations, rc != 0)
    detail = "\n".join(f"{s.status} {s.name}: {s.detail}" for s in run.stages)
    assert rc == 0, detail
    assert any(s.status == selftest.PASS for s in run.stages), detail


def test_qualification_failure_retains_m1_artifacts(tmp_path, capsys):
    source = tmp_path / "work"
    (source / "results" / "init").mkdir(parents=True)
    (source / "results" / "init" / "transcript.jsonl").write_text(
        'session failed: sabir-A: not attached\n')
    (source / "sabir.log").write_text("server trace\n")
    observations = tmp_path / "observations"
    observations.mkdir()

    _finish_qualification_selftest(source, observations, True)

    retained = observations / "creance-m1"
    assert (retained / "results" / "init" / "transcript.jsonl").read_text() == (
        'session failed: sabir-A: not attached\n')
    assert (retained / "sabir.log").read_text() == "server trace\n"
    assert str(retained) in capsys.readouterr().out
    assert not source.exists()


def test_qualification_success_removes_m1_work_tree(tmp_path):
    source = tmp_path / "work"
    source.mkdir()
    (source / "sabir.log").write_text("server trace\n")
    observations = tmp_path / "observations"
    observations.mkdir()

    _finish_qualification_selftest(source, observations, False)

    assert not source.exists()
    assert list(observations.iterdir()) == []


def test_qualification_m1_artifacts_reject_symlink(tmp_path):
    source = tmp_path / "work"
    source.mkdir()
    (source / "sabir.log").write_text("server trace\n")
    (source / "linked.log").symlink_to(source / "sabir.log")
    observations = tmp_path / "observations"
    observations.mkdir()

    with pytest.raises(ValueError, match="nonregular entry"):
        _finish_qualification_selftest(source, observations, True)

    assert source.exists()
    assert list(observations.iterdir()) == []


def test_m1_gate_without_qualification_env_retains_failure(tmp_path, monkeypatch,
                                                            capsys):
    monkeypatch.delenv("HFMODEM_QUALIFICATION_OBSERVATIONS", raising=False)
    source = tmp_path / "fake-selftest"

    class FakeSelftest:
        def __init__(self, *, size, require, keep):
            assert size == "2k"
            assert keep is True
            source.mkdir()
            (source / "sabir.log").write_text("server failed\n")
            self.tmp = source
            self.stages = [selftest.Stage("cwp unidir 2k over the pair",
                                          selftest.FAIL, "failed:modemerror")]

        def run(self):
            return 1

    monkeypatch.setattr(selftest, "Selftest", FakeSelftest)
    with pytest.raises(AssertionError, match="failed:modemerror"):
        test_m1_gate(capsys, tmp_path)

    retained = tmp_path / "creance-observations" / "creance-m1"
    assert (retained / "sabir.log").read_text() == "server failed\n"
    assert str(retained) in capsys.readouterr().out
    assert not source.exists()
