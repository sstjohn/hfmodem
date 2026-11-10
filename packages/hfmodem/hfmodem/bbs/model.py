# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Transport independent BBS attachment and output values."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Presentation:
    mode: str = 'plain'
    width: int = 80
    height: int = 24
    page_bytes: int = 600
    ansi_capable: bool = False

    def __post_init__(self):
        if self.mode not in ('plain', 'ansi16'):
            raise ValueError('Unknown terminal mode')
        if not 20 <= self.width <= 160 or not 8 <= self.height <= 100:
            raise ValueError('Terminal dimensions out of range')
        if not 64 <= self.page_bytes <= 32768:
            raise ValueError('Page byte limit out of range')
        if self.mode == 'ansi16' and not self.ansi_capable:
            raise ValueError('ANSI requires explicit attachment capability')


@dataclass(frozen=True)
class OutputEvent:
    kind: str
    output_id: str
    text: str
    session_id: str
    run_id: str | None = None
    turn_id: str | None = None
    policy_revision: int = 0
    more: bool = False
    payload: object = None


class BusyError(RuntimeError):
    """One caller or an unreaped backend still owns this BBS."""
