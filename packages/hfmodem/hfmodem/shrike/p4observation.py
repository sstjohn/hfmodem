# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Shared P4 receive coordinates and immutable evidence records."""

from dataclasses import dataclass
from enum import StrEnum
from . import p4frame

WORD_HORIZON_S = 0.275
RESPONSE_S = 0.935
"""Centre of the 0.90-0.97 s bracket after our packet's start where a peer's word
answers it: VE3KPG answers at +0.944-0.950, N9VC reaches +0.968."""


class Kind(StrEnum):
    WORD = "word"
    BODY = "body"
    SHORT = "short"
    LONG = "long"
    NORMAL = "normal"
    CHIRP = "chirp"


class Registered(StrEnum):
    NEW = "new"
    ENRICHED = "enriched"
    DUPLICATE = "duplicate"
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    EXPIRED = "expired"
    EXTENDED = "extended"


@dataclass(frozen=True)
class Observation:
    kind: Kind
    fs: int
    origin: int
    first_chip: int
    stream_sample: int
    duration_s: float
    cfo_hz: float
    coherence: float
    slot: int | None = None
    provisional: bool = False
    symbols_read: int = 0
    extent_s: float | None = None
    word: str | None = None
    score: float = 0.0
    shift: int = 0
    after: int | None = None
    frame: p4frame.Frame | None = None
    header: str | None = None
    value: int | None = None
    walsh_peak: int | None = None
    lattice: float = 0.0
    chirp: object | None = None
    weak: bool = False
    unexpected: bool = False
    placed: bool = False
    peer_raster_tail: bool = False
    sf8: bool = False
    speedlevel: str | None = None


@dataclass(frozen=True)
class RegisteredObservation:
    observation: Observation
    outcome: Registered
    replaced: Observation | None = None
    answered: int | None = None
    responds_to: int | None = None
    response_generation: int | None = None
    response_slot: int | None = None
    response_owner: object | None = None
