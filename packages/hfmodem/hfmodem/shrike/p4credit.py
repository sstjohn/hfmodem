# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive-source receipt and host transaction boundary; epochs belong to P4Turn."""

from dataclasses import dataclass

from .p4observation import Kind


@dataclass(frozen=True)
class SourceReceipt:
    epoch: int
    root: tuple[Kind, int]
    identity: tuple[int, bytes]


@dataclass(frozen=True)
class ReceiptContext:
    receipt: SourceReceipt | None
    kind: Kind


@dataclass(frozen=True)
class SourceResult:
    delivery: bytes | None = b""
    duplicate: bool = False
    held_delta: int = 0
    held_total: int = 0
    committed_credit: int = 0
    released_source: int = 0
    state: str = "idle"
    cursor: int = 0
    expected_counter: int | None = None
    receipt: SourceReceipt | None = None
    reason: str | None = None
    incomplete: bool = False
    replayed_source_bytes: int = 0
    new_source_bytes: int = 0
    resegmented: bool = False
