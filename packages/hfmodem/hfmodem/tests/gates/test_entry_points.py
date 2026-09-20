# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every declared entry point resolves and starts.

This exists because it caught something the moment it was written: the `hfmodem`
console script was declared in `pyproject.toml` pointing at `hfmodem.cli:main`, and
`hfmodem/cli.py` did not exist. Nothing noticed — a wheel would have installed
cleanly and produced a command that fails on first use, which is the same class of
defect as a package-data glob that ships no assets.

shrike's `validate_static` had this check and it was the best idea in it; it was
not carried into the shared gates until this file. It reads the declarations rather
than a hand-written list, so a new entry point is covered without anyone
remembering — with one exception, below.

`-m`-only modules are listed explicitly. Enumerating from `[project.scripts]`
*alone* would be a coverage regression: `shrike.ota` and `shrike.beacon` are
runnable only as `python -m`, and they are the two that key a transmitter.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
PACKAGES = REPO / "packages"

#: Runnable with `python -m` but not declared as console scripts. Named here so
#: the union of covered entry points cannot silently shrink.
MODULE_ENTRY_POINTS = [
    # The station itself. At the radio the venv is addressed by path rather
    # than activated, so this and the console script must agree.
    "hfmodem",
    "hfmodem.shrike.monitor",
    "hfmodem.shrike.live",
    "hfmodem.shrike.onair",
    "hfmodem.shrike.ota",
    "hfmodem.shrike.beacon",
    "hfmodem.shrike.ptc",
    "hfmodem.besra.monitor",
    "hfmodem.besra.host.run_server",
    "hfmodem.sabir.monitor",
    "hfmodem.sabir.offair",
    "hfmodem.sabir.onair",
    "hfmodem.sabir.host.run_server",
    "hfmodem.kestrel.host.run_server",
]


def _scripts() -> list[tuple[str, str, str]]:
    """(dist, command, "module:function") for every declared console script."""
    out = []
    for py in sorted(PACKAGES.glob("*/pyproject.toml")):
        data = tomllib.loads(py.read_text(encoding="utf-8"))
        for cmd, target in data.get("project", {}).get("scripts", {}).items():
            out.append((py.parent.name, cmd, target))
    return out


def test_some_console_scripts_are_declared():
    """A gate that iterates an empty list passes for the wrong reason.

    Two, deliberately: `hfmodem` and `creance`. The four repos declared ten
    between them — `shrike-monitor`, `besra-modem`, `sabir-server` and the rest —
    and they collapse into one command with verbs rather than one command per
    modem, because a station is one thing. The modules themselves are still
    runnable and `MODULE_ENTRY_POINTS` is what keeps them covered.
    """
    declared = _scripts()
    assert len(declared) == 2, declared
    assert {c for _, c, _ in declared} == {"hfmodem", "creance"}


@pytest.mark.parametrize("dist,cmd,target", _scripts(),
                         ids=[f"{c}" for _, c, _ in _scripts()])
def test_a_declared_console_script_resolves(dist, cmd, target):
    """The defect this file was written for: a declared script whose module does
    not exist installs fine and fails on first use."""
    mod_name, _, func_name = target.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        pytest.fail(f"{dist} declares `{cmd} = {target}` and {mod_name} will not "
                    f"import: {exc}")
    fn = getattr(mod, func_name, None)
    assert callable(fn), f"{target} is not callable"


@pytest.mark.parametrize("mod", MODULE_ENTRY_POINTS)
def test_a_module_entry_point_starts(mod):
    """`--help` and exit 0. It proves the module imports, its argument parser is
    well formed, and nothing at import time needs a radio."""
    r = subprocess.run([sys.executable, "-m", mod, "--help"],
                       capture_output=True, text=True, timeout=120,
                       cwd=PACKAGES / "hfmodem")
    assert r.returncode == 0, (
        f"python -m {mod} --help exited {r.returncode}\n{r.stderr[-1500:]}")


#: Every entry point that must stay covered, by name. A `>=` on a count let an
#: audit delete `hfmodem.shrike.beacon` — one of the two that key a transmitter —
#: and the gate stayed green, which is the whole failure mode this file is about.
REQUIRED_COVERAGE = frozenset(MODULE_ENTRY_POINTS) | {"hfmodem.cli", "creance.cli"}


def test_the_covered_set_does_not_shrink():
    """Both halves, by name rather than by count. Enumerating only the
    declarations would drop the `-m`-only modules, two of which key a
    transmitter."""
    covered = {t.partition(":")[0] for _, _, t in _scripts()} | set(MODULE_ENTRY_POINTS)
    missing = REQUIRED_COVERAGE - covered
    assert not missing, f"entry points no longer covered: {sorted(missing)}"
