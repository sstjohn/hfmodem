# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Transport-free, nonblocking backend interface.

An opaque `ready` releases only a write slot; it is not command completion.
Each adapter instance belongs to exactly one core run generation.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GameEvent:
    kind: str
    text: str = ""
    mode: str = "unknown"
    turn_id: str | None = None
    outcome: str = ""
    checkpoint_id: str | None = None


class GameAdapter(Protocol):
    capabilities: frozenset[str]

    def start(self, player: str, directory: Path) -> None: ...
    def submit(self, text: str, mode: str = "default") -> bool: ...
    def poll(self) -> list[GameEvent]: ...
    def close(self, reason: str) -> None: ...
