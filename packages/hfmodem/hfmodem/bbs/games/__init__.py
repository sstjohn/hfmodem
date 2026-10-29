# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Optional, separately installed game runtimes."""
from .base import GameAdapter, GameEvent
from .registry import configured_games

__all__ = ["GameAdapter", "GameEvent", "configured_games"]
