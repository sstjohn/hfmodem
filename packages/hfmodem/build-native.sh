#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/hfmodem-cargo}"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        PYTHON="$ROOT/.venv/bin/python"
    else
        PYTHON=python3
        if COMMON_DIR="$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
            MAIN="$(dirname "$COMMON_DIR")"
            if [[ -x "$MAIN/.venv/bin/python" ]]; then
                PYTHON="$MAIN/.venv/bin/python"
            fi
        fi
    fi
fi

(
    cd "$SCRIPT_DIR"
    HFMODEM_NATIVE=1 "$PYTHON" setup.py build_rust --inplace --release
)
for artifact in "$SCRIPT_DIR"/hfmodem/core/_dsp*.so; do
    printf 'Built %s\n' "$artifact"
done

cargo build --offline --release --locked --manifest-path "$SCRIPT_DIR/native/p4long/Cargo.toml"
if [[ "$(uname -s)" == Darwin ]]; then
    library=libhfmodem_p4long.dylib
else
    library=libhfmodem_p4long.so
fi
destination="$SCRIPT_DIR/native/p4long/target/release"
mkdir -p "$destination"
cp "$CARGO_TARGET_DIR/release/$library" "$destination/$library"
printf 'Built %s\n' "$destination/$library"
