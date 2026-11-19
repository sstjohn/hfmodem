#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$PWD
PYTHON=${PYTHON:-python}

usage() {
    echo "Usage: $0 {install|test [PYTEST_ARGS...]|build [DIST]|check-dist [DIST]|smoke [DIST]|lock}" >&2
    exit 2
}

check_dist() {
    local dist=$1 package wheel failed=0
    local wheels sdists
    shopt -s nullglob
    for package in hfhost hfmodem creance; do
        wheels=("$dist/$package/"*.whl)
        sdists=("$dist/$package/"*.tar.gz)
        if [[ ${#wheels[@]} -ne 1 || ${#sdists[@]} -ne 1 ]]; then
            echo "$dist/$package: expected one sdist and one wheel; found ${#sdists[@]} sdists and ${#wheels[@]} wheels" >&2
            printf '  %s\n' "$dist/$package/"* >&2
            failed=1
        fi
        for wheel in "${wheels[@]}"; do
            if [[ $wheel != *-py3-none-any.whl ]]; then
                echo "Non-pure wheel: $wheel" >&2
                failed=1
            fi
        done
    done
    return "$failed"
}

smoke() (
    local dist=$1 scratch
    dist=$(cd "$dist" && pwd)
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    # A checkout on PYTHONPATH would satisfy pip and hide the wheels under test.
    unset PYTHONPATH
    "$PYTHON" -m venv "$scratch/venv"
    "$scratch/venv/bin/python" -m pip install -c "$ROOT/publish/ci/constraints.txt" "$dist"/*/*.whl
    cd "$scratch"
    "$scratch/venv/bin/python" -I - <<'PY'
import hfmodem
import hfhost
import creance
from hfmodem.shrike import p4longnative

assert p4longnative.backend is None, p4longnative.backend
for module in (hfmodem, hfhost, creance, p4longnative):
    print(module.__name__, getattr(module, "__version__", None), module.__file__)
PY
)

lock() (
    local scratch
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    "$PYTHON" -m venv "$scratch/venv"
    HFMODEM_PURE=1 "$scratch/venv/bin/python" -m pip install \
        -e 'packages/hfhost[dev]' -e 'packages/hfmodem[dev]' -e 'packages/creance[dev]' \
        pytest-xdist build twine
    "$scratch/venv/bin/python" -m pip freeze --exclude-editable > "$scratch/freeze.txt"
    "$scratch/venv/bin/python" - "$scratch/freeze.txt" <<'PY'
from pathlib import Path
import re
import sys

lines = Path(sys.argv[1]).read_text().splitlines()
pins = [line for line in lines if not re.match(r"(hfhost|hfmodem|creance)\b", line, re.I)]
Path("publish/ci/constraints.txt").write_text(
    "# SPDX-License-Identifier: AGPL-3.0-only\n"
    "# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)\n"
    "# Regenerate: publish/ci/release-check.sh lock (on Linux x86-64, Python 3.13).\n"
    + "\n".join(pins) + "\n"
)
PY
)

[[ $# -ge 1 ]] || usage
command=$1
shift
case "$command" in
    install)
        [[ $# -eq 0 ]] || usage
        "$PYTHON" -m pip install -c publish/ci/constraints.txt \
            -e 'packages/hfhost[dev]' -e 'packages/hfmodem[dev]' \
            -e 'packages/creance[dev]' pytest-xdist
        ;;
    test)
        export HFMODEM_NATIVE=0
        selection='not slow and not realtime and not wall_clock and not corpus and not hardware and not ardopcf'
        failed=0
        nice -n 19 "$PYTHON" -m pytest -c pyproject.toml -q -n "${RELEASE_CHECK_WORKERS:-auto}" --dist worksteal --durations=50 \
            -m "$selection" --ignore packages/hfmodem/hfmodem/tests/coexist packages "$@" || {
            echo 'Release test step (a) failed (packages)' >&2
            failed=1
        }
        nice -n 19 "$PYTHON" -m pytest -c pyproject.toml -q -n 0 --durations=50 \
            -m "$selection" packages/hfmodem/hfmodem/tests/coexist "$@" || {
            echo 'Release test step (b) failed (coexist)' >&2
            failed=1
        }
        exit "$failed"
        ;;
    build)
        [[ $# -le 1 ]] || usage
        dist=${1:-dist}
        "$PYTHON" -m pip install -c publish/ci/constraints.txt build twine
        for p in hfhost hfmodem creance; do
            HFMODEM_PURE=1 "$PYTHON" -m build "packages/$p" --outdir "$dist/$p"
        done
        "$PYTHON" -m twine check --strict "$dist"/*/*
        check_dist "$dist"
        smoke "$dist"
        ;;
    check-dist)
        [[ $# -le 1 ]] || usage
        check_dist "${1:-dist}"
        ;;
    smoke)
        [[ $# -le 1 ]] || usage
        smoke "${1:-dist}"
        ;;
    lock)
        [[ $# -eq 0 ]] || usage
        lock
        ;;
    *) usage ;;
esac
