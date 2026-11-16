# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Standalone, explicitly placed robust LONG C2 bench decoder.

No acquisition, live dispatch, callbacks, delivery or cross-call saved work.
"""
from dataclasses import dataclass, field
from numbers import Integral
import time

import numpy as np

from . import p4c2, p4frame, p4longdecode, p4robustrx, p4sf8rx


@dataclass(frozen=True)
class _Member:
    weighting: str
    quarter_turn: int
    fields: np.ndarray = field(repr=False)
    syndrome_score: float | None = None


@dataclass(frozen=True)
class MethodOutcome:
    member_index: int
    weighting: str
    quarter_turn: int
    method: str
    iteration: int
    crc_valid: bool
    identity_index: int | None


@dataclass(frozen=True)
class DecodeOutcome:
    complete: bool
    frame: p4frame.Frame | None = field(repr=False)
    reason: str
    speedlevel: int
    iterations: int
    eligible_members: int
    completed_members: int
    component_calls: int
    constituent_positions: int
    crc_checks: int
    identity_count: int
    ambiguous: bool
    identities: tuple = field(repr=False)
    provenance: tuple[MethodOutcome, ...]
    selected_provenance: MethodOutcome | None
    stage_seconds: tuple
    elapsed_seconds: float


def _integer(value, minimum, maximum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + ' outside supported range')
    return value


def _geometry(speedlevel):
    level = _integer(speedlevel, 2, 4, 'speedlevel')
    if level not in (2, 4):
        raise ValueError('speedlevel must be 2 or 4')
    return p4longdecode.geometry(level)


def _resources():
    # Validate both immutable C2 orders before a no-hit/eligibility decision.
    output = {}
    for sl in (2, 4):
        size = p4longdecode.geometry(sl).systematic
        order = p4c2.permutation(size)
        p4c2.validate(order, size)
        output[sl] = order
    return output


def _eligible_members(raw, g):
    """Preserve this bench API's declared placement/layout/weighting domain."""
    with np.errstate(over='ignore', invalid='ignore'):
        differential = raw[1:] * raw[:-1].conjugate()
    if not np.all(np.isfinite(differential)):
        raise ValueError('nonfinite differential scores')
    if g.speedlevel == 4:
        members = tuple(_Member('amplitude' if amplitude else 'unit', turn, fields)
            for amplitude in (False, True)
            for turn, fields in p4sf8rx._scores(differential, amplitude=amplitude, stride=29))
    else:
        unit = differential / np.maximum(abs(differential), 1e-30)
        unit *= np.exp(-1j*np.angle(np.mean(unit**4))/4)
        data = unit[p4robustrx._SYMBOL_ORDER]
        candidates = []
        for turn in range(4):
            rotated = data * (1j**turn)
            fields = 4*np.stack((rotated.real+rotated.imag,
                                 rotated.real-rotated.imag), axis=-1).ravel()
            for level in (2, 3):
                score = p4robustrx.syndrome(fields, level)
                if score >= .80:
                    candidates.append((score, level, turn, fields))
        if not candidates:
            return ()
        best = max(row[0] for row in candidates)
        retained = [row for row in candidates if row[0] == best]
        if {row[1] for row in retained} != {2}:
            return ()
        members = tuple(_Member('unit', turn, fields, score)
                        for score, _, turn, fields in retained)
    if any(not np.all(np.isfinite(member.fields)) for member in members):
        raise ValueError('nonfinite member scores')
    return members


def decode_placed(symbols, *, speedlevel: int, iterations: int = 8,
                  component_budget: int | None = None) -> DecodeOutcome:
    started = time.monotonic()
    g = _geometry(speedlevel)
    iterations = _integer(iterations, 0, 8, 'iterations')
    if component_budget is not None:
        component_budget = _integer(component_budget, 0, None, 'component_budget')
    raw = np.asarray(symbols, dtype=complex)
    if raw.ndim != 1 or len(raw) != g.symbols+1 or not np.all(np.isfinite(raw)):
        raise ValueError('expected finite one-dimensional complete placed body')
    _resources()
    members = _eligible_members(raw, g) if np.any(raw) else ()
    canonical = p4longdecode.geometry(speedlevel)
    job = p4longdecode.LongSearch([
        p4longdecode.LongMember(canonical, m.fields, m.quarter_turn, m.weighting)
        for m in members], policy='exhaustive', iterations=iterations,
        component_budget=component_budget)
    result = job.advance()
    provenance = tuple(MethodOutcome(p.member_index, members[p.member_index].weighting,
        members[p.member_index].quarter_turn, p.method, p.iteration,
        p.crc_valid, p.identity_index) for p in result.provenance)
    selected = next((p for p in provenance if p.identity_index == 0), None)
    reason = result.state if result.state in ('unique', 'ambiguous') else (
        'incomplete-work' if not result.complete else
        'layout-ineligible' if np.any(raw) and not members else 'no-hit')
    return DecodeOutcome(result.complete, result.frame, reason, speedlevel,
        iterations, len(members), len(members) if result.complete else 0,
        result.component_calls, result.constituent_positions,
        len(provenance), len(result.identities), len(result.identities) > 1,
        result.identities, provenance, selected if len(result.identities) == 1 else None,
        result.stage_seconds, time.monotonic()-started)
