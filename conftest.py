# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent

#: Load average per core at which a live rendezvous test in this repository has
#: already failed for want of CPU rather than for cause: 7.97 on 18 cores, with
#: seven of `creance`'s twenty probes reporting no session established between the
#: pair, where the same test alone passes in 27 s four times out of four. Per core
#: so it carries to another machine.
_CONTENDED_PER_CORE = 0.4

_LOAD_AT_ENTRY = pytest.StashKey[float]()

# The editable installs in .venv resolve hfmodem/hfhost/creance to wherever they
# were `pip install -e`d, almost always the main tree — so a worktree's own
# pytest run would otherwise test someone else's code under its own name.
# setuptools' editable finder is appended to sys.meta_path, after the stdlib
# PathFinder that walks sys.path, so putting this tree's packages ahead in
# sys.path resolves them here first and the editable finder is never reached.
#
# The tests that run a tool or the console as a child process need the same
# answer, and a child reads PYTHONPATH, not this process's sys.path: without it a
# worktree's tool imports the main tree's hfhost and fails on a module only this
# tree has.
_PACKAGES = [str(p) for p in sorted(ROOT.glob("packages/*")) if (p / "pyproject.toml").is_file()]
for pkg_dir in _PACKAGES:
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
_inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
os.environ["PYTHONPATH"] = os.pathsep.join(
    _PACKAGES + [p for p in _inherited if p not in _PACKAGES])

from hfmodem.tests.hermetic import captures_untouched, station_kept_out  # noqa: F401  (autouse fixtures)


def pytest_configure(config):
    import importlib
    from hfmodem.tests.hermetic import GUARD
    for arg in config.invocation_params.args:
        path = Path(config.invocation_params.dir, str(arg).split("::")[0]).resolve()
        if path == GUARD.resolve() or GUARD.resolve() in path.parents:
            # Collecting it puts the guard's startup directory on sys.path, and
            # its tripwire sounddevice then answers every later import.
            raise pytest.UsageError(f"{arg} is the hermetic guard's startup, not a test")
    for pkg_dir in sorted(ROOT.glob("packages/*")):
        if not (pkg_dir / "pyproject.toml").is_file():
            continue
        for candidate in pkg_dir.iterdir():
            if not (candidate / "__init__.py").is_file():
                continue
            mod = importlib.import_module(candidate.name)
            resolved = Path(mod.__file__).resolve()
            if ROOT not in resolved.parents:
                pytest.exit(
                    f"{candidate.name} imports from {resolved}, outside {ROOT} "
                    "— this run is testing the wrong tree's code",
                    returncode=1,
                )


@pytest.fixture(autouse=True)
def isolated_claim_state(tmp_path_factory, monkeypatch):
    """Keep every claim, in this process and any it spawns, out of the operator's
    real state directory: the claim layer resolves it from the password database,
    not $HOME, and honours HFMODEM_STATE_DIR only under the coexist harness.
    A value other than "1" leaves the sounddevice and ffmpeg shims off."""
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "pytest")
    monkeypatch.setenv("HFMODEM_STATE_DIR", str(tmp_path_factory.mktemp("claim-state")))
    for key in ("HFMODEM_COEXIST_LOG", "HFMODEM_COEXIST_BARRIERS", "HFMODEM_COEXIST_FAULT"):
        monkeypatch.delenv(key, raising=False)


def pytest_collection_modifyitems(config, items):
    """`posix_only(reason)` is a visible skip off POSIX that says what is missing."""
    for item in items:
        marker = item.get_closest_marker("posix_only")
        if marker is None:
            continue
        if not marker.args:
            raise pytest.UsageError(f"{item.nodeid}: posix_only needs a reason")
        if os.name != "posix":
            item.add_marker(pytest.mark.skip(reason=f"POSIX only: {marker.args[0]}"))


def pytest_runtest_setup(item):
    if item.get_closest_marker("realtime") and hasattr(os, "getloadavg"):
        item.stash[_LOAD_AT_ENTRY] = os.getloadavg()[0]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.failed and _LOAD_AT_ENTRY in item.stash:
        report.sections.append(("the machine this ran on", _contention(item)))
    return report


def _contention(item) -> str:
    """What else the machine was doing, said where the assertion is read.

    Seven concurrent jobs and six simultaneous pytest runs is an ordinary evening here, and
    a test that has to rendezvous in real time fails under that for a reason that
    is not in the code. Dismissing such a failure took someone re-running it in
    isolation; not dismissing it invites a "fix" to a non-bug, a good change
    reverted, or a threshold loosened that was right. Neither is a judgement the
    reader can make without knowing what was on the machine, so it goes in the
    failure.
    """
    cores = os.cpu_count() or 1
    entry, leaving = item.stash[_LOAD_AT_ENTRY], os.getloadavg()[0]
    runs = _pytest_runs()
    contended = max(entry, leaving) >= cores * _CONTENDED_PER_CORE or runs > 1
    return (
        f"{cores} cores, load average {entry:.2f} entering this test and "
        f"{leaving:.2f} leaving it, {runs} pytest process(es) on the machine.\n"
        + ("Something else was competing for the machine, so contention is a "
           "candidate for this failure ahead of the code. Re-run this test alone "
           "before believing it, and before changing anything it measures."
           if contended else
           "Nothing else was competing for the machine, so contention does not "
           "explain this failure."))


def _pytest_runs() -> int:
    found = subprocess.run(["pgrep", "-f", "pytest"], capture_output=True, text=True)
    return len(found.stdout.split())
