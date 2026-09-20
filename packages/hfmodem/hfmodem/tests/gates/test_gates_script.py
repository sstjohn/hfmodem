# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`tools/gates.sh` run for real, against a stub tree built to be red.

Nothing executed this script. Every one of its three exit-code repairs could be
undone with the tree still green: `exit "${rc:-1}"` back to a `printf`, `-rfEs`
back to `-rf -rs`, the ruff `PIPESTATUS` check deleted -- 70 passed each time.
The headline defect of the commit that repaired them ("reported no failures and
always exited 0") was therefore reintroducible in silence, and it is the one
defect that costs the most, because the script's whole job is to be the thing
whose exit status gets quoted.

The suites are read out of the script rather than restated here, and a stub tree
is built at exactly the paths it names: a temporary repo with `tools/gates.sh`
copied in, a `.venv/bin/python` that execs this interpreter, a stub `ruff`, and
one tiny test module in each suite directory. That is what makes running the real
script cheap enough to be a test -- what is under examination is the script's
control flow, not the 45 minutes of arithmetic it usually drives.

Parameterising `gates.sh` on its suite list was the alternative and it was
declined: the list would then be settable from outside, and a gate whose scope an
environment variable can shrink is the same failure wearing a sleeve.

Each case below is one of the reverts, and goes red under it: the green tree
proves the harness can pass at all, the failing suite proves the exit plumbing
and the `-r` letters, the unarmed arbiter proves the skipped-arbiter rule, the
failing oracle proves the sequential loop's `PIPESTATUS`, and the failing ruff
proves the last one.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
GATES = REPO / "tools" / "gates.sh"

pytestmark = pytest.mark.skipif(
    not GATES.exists(), reason="tools/ not present (installed-wheel run)")

_GREEN = "def test_the_stub_suite_ran():\n    assert True\n"

#: A failure, a skip and an error, because the summary letters are three separate
#: store actions and `-rf -rs` looks like both and is only the second.
_RED = '''
import pytest


def test_the_stub_suite_ran():
    assert True


def test_the_planted_failure():
    assert False, "planted"


@pytest.mark.skip(reason="planted skip")
def test_the_planted_skip():
    pass


@pytest.fixture
def broken():
    raise RuntimeError("planted error")


def test_the_planted_error(broken):
    pass
'''


def _suites() -> tuple[list[str], list[str]]:
    """The targets the script runs, and the ones it adds under `--all`."""
    text = GATES.read_text()
    base = re.search(r"SUITES=\((.*?)\)", text, re.S).group(1).split()
    extra = re.search(r"SUITES\+=\((.*?)\)", text).group(1).split()
    return base, extra


def _tree(root: Path, *, bulk: str = "green", oracle: str = "green",
          ruff: int = 0, armed: bool = True) -> None:
    base, extra = _suites()
    for i, d in enumerate(base + extra):
        body = _RED if bulk == "red" and i == 0 else _GREEN
        if d.endswith(".py"):           # the array names one file as well as dirs
            (root / d).parent.mkdir(parents=True, exist_ok=True)
            (root / d).write_text(body)
            continue
        (root / d).mkdir(parents=True, exist_ok=True)
        # Unique module names: pytest imports test files by basename, and two
        # `test_stub.py` in unpackaged directories collide before either runs.
        (root / d / f"test_stub_{d.replace('/', '_')}.py").write_text(body)

    shrike = root / "packages/hfmodem/hfmodem/tests/shrike"
    shrike.mkdir(parents=True, exist_ok=True)
    for t in ("p1", "p2", "p3"):
        (shrike / f"test_{t}_oracle.py").write_text(
            _RED if oracle == "red" and t == "p2" else _GREEN)

    venv = root / ".venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "ruff").write_text(f'#!/bin/sh\necho "stub ruff"\nexit {ruff}\n')
    for name in ("python", "ruff"):
        (venv / name).chmod(0o755)

    (root / "working/ardop/interop").mkdir(parents=True)
    (root / "working/ardop/interop/ardopcf-vm").touch()

    tools = root / "tools"
    tools.mkdir()
    shutil.copy(GATES, tools / "gates.sh")
    (tools / "gates.sh").chmod(0o755)
    if armed:
        for name in ("pmon-oracle", "pmon-stage"):
            (root / name).touch()
        (root / "kestrel-corpus").mkdir()
        (tools / "arbiters.env").write_text(
            f'export PMON_ORACLE="{root}/pmon-oracle"\n'
            f'export PMON_STAGE="{root}/pmon-stage"\n'
            f'export KESTREL_CORPUS="{root}/kestrel-corpus"\n')


def _run(root: Path) -> subprocess.CompletedProcess:
    """The script, with this machine's own arbiters out of the environment.

    A developer with `PMON_ORACLE` exported would otherwise arm the unarmed case
    from outside and it would pass without checking anything.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("PMON_")}
    env.pop("ARDOPCF", None)
    env.pop("KESTREL_CORPUS", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(["bash", str(root / "tools/gates.sh")],
                          capture_output=True, text=True, timeout=600, env=env)


def test_the_gate_script_parses():
    r = subprocess.run(["bash", "-n", str(GATES)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_a_green_tree_passes(tmp_path):
    """The control. Every case below asserts a non-zero exit, and none of them
    would mean anything if the harness could not produce a zero one."""
    _tree(tmp_path)
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_failing_suite_fails_the_gate_and_names_the_test(tmp_path):
    """The defect this file exists for, and the `-r` letters in one case.

    A 35-minute run that ends "1 failed" without saying which costs another 35
    minutes, so the summary is asserted alongside the exit status: under
    `-rf -rs` the second `-r` replaces the first and only the skips are listed.
    """
    _tree(tmp_path, bulk="red")
    r = _run(tmp_path)
    assert r.returncode != 0, r.stdout + r.stderr
    for pattern in (r"^FAILED .*test_the_planted_failure",
                    r"^ERROR .*test_the_planted_error",
                    r"^SKIPPED"):
        assert re.search(pattern, r.stdout, re.M), (pattern, r.stdout)


def test_an_unarmed_arbiter_fails_the_gate(tmp_path):
    """A skipped arbiter is not a green one, and the suites here are all green:
    the only thing that can fail this run is the unset variable."""
    _tree(tmp_path, armed=False)
    r = _run(tmp_path)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "PMON_ORACLE unset" in r.stdout


def test_a_failing_oracle_fails_the_gate(tmp_path):
    """The oracles run one at a time after the bulk suite, each piped into
    `tail -1`, which is a second place for a pipe to report its own success."""
    _tree(tmp_path, oracle="red")
    r = _run(tmp_path)
    assert r.returncode != 0, r.stdout + r.stderr


def test_a_failing_ruff_fails_the_gate(tmp_path):
    _tree(tmp_path, ruff=1)
    r = _run(tmp_path)
    assert r.returncode != 0, r.stdout + r.stderr
