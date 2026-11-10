#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
set -euo pipefail
here=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
root=$(cd -P -- "$here/../../.." && pwd -P)
if [[ -n "${HFMODEM_INTEROP_PYTHON:-}" ]]; then
    python="$HFMODEM_INTEROP_PYTHON"
elif [[ -x "$root/.venv/bin/python" ]]; then
    python="$root/.venv/bin/python"
else
    python=python3
    # An extracted source kit has no Git metadata. Only a real checkout may
    # borrow its shared repository's prepared environment.
    if [[ "$(git -C "$root" rev-parse --show-toplevel 2>/dev/null || true)" == "$root" ]]; then
        common=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)
        if [[ -x "$(dirname -- "$common")/.venv/bin/python" ]]; then
            python="$(dirname -- "$common")/.venv/bin/python"
        fi
    fi
fi
exec "$python" "$here/cli.py" "$@"
