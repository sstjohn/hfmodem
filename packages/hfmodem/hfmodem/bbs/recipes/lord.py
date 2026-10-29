# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Inspected LORD candidate, deliberately unavailable as a validated recipe.

No original LORD payload, unclear-rights asset mirror, package-runner fallback,
or imitation LORD is provided. Operator-supplied artifacts need a separate
curated hash, tested launch profile and asset provenance before activation.
"""
CANDIDATE = {
    "game": "lord",
    "project": "LORD.ts",
    "revision": "1e3b30e0cf3b0d93af80d6c6833ba55b31705328",
    "version": "1.4.6",
    "source": "https://github.com/talisto/LORD.ts",
    "package_lock_sha256": "5625d9466d6cf594e0b63f7fbce8fa65233f5c5521d3b388a5c710bfad0fc45f",
    "runtime": "Node.js >=22; frozen npm workspaces; better-sqlite3 native module",
    "validated": False,
    "acquisition": "operator-supplied artifacts only until asset rights are established",
    "blockers": [
        "Original DOS LDY assets and reverse-engineered IGMs have unverified individual distribution terms",
        "No dependency-closed offline runtime artifact has passed a macOS smoke test",
        "No verified cursor/screen reconstruction, key/line hook, shared-realm or hangup test",
    ],
}
