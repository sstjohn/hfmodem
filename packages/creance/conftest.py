# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""What the repo-root conftest does, for a run that never reaches it.

This package declares its own `[tool.pytest.ini_options]` so its suite works from
a checkout of the package alone, and that makes `packages/creance` the rootdir --
which is also where pytest stops collecting conftests. So the root one is invisible
to every invocation that names a path in here, however it was launched.

That costs the `realtime` contention note, and it costs it on the suite the note
was written for: `tests/test_conformance_live.py`'s pair gate is the failure the
root conftest quotes. The hooks are taken from there rather than copied, so there
is still one implementation of what a contended machine looks like.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("_repo_conftest", ROOT / "conftest.py")
_repo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_repo)

_contention = _repo._contention
pytest_runtest_setup = _repo.pytest_runtest_setup
pytest_runtest_makereport = _repo.pytest_runtest_makereport
isolated_claim_state = _repo.isolated_claim_state

for pkg_dir in sorted(ROOT.glob("packages/*")):
    if (pkg_dir / "pyproject.toml").is_file() and str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))


def pytest_configure(config):
    import importlib
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
