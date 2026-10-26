# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# This package's own pytest config makes it the rootdir, where conftest
# collection stops, so the repo-root conftest's claim isolation is taken from there.
_spec = importlib.util.spec_from_file_location("_repo_conftest", ROOT / "conftest.py")
_repo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_repo)

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
