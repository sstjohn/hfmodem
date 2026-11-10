# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Standalone, explicitly placed robust LONG C2 bench decoder.

No acquisition, live dispatch, callbacks, delivery or cross-call saved work.
"""
from dataclasses import dataclass, field
from numbers import Integral
import time

import numpy as np

from . import p4c2, p4decode, p4frame, p4robustrx, p4sf8rx

_TABLES = ((2, 348), (4, 1160))


@dataclass(frozen=True)
class _Geometry:
    speedlevel: int
    symbols: int
    systematic: int
    data_bytes: int
    padding: int
    period: int
    c2_phase: int

    @property
    def parity(self):
        return self.systematic // self.period


@dataclass(frozen=True)
class _Member:
    weighting: str
    quarter_turn: int
    fields: np.ndarray = field(repr=False)
    syndrome_score: float | None = None


@dataclass(frozen=True)
class _Candidate:
    method: str
    iteration: int
    bits: np.ndarray = field(repr=False)


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


class _BudgetExhausted(Exception):
    pass


class _Work:
    def __init__(self, budget):
        self.budget = budget
        self.calls = 0
        self.positions = 0
        self.costs = []

    def call(self, stage, count, function, *args, **kwargs):
        if self.budget is not None and self.calls >= self.budget:
            raise _BudgetExhausted
        started = time.monotonic()
        self.calls += 1
        self.positions += count
        result = function(*args, **kwargs)
        self.costs.append((stage, time.monotonic()-started))
        return result


def _integer(value, minimum, maximum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + ' outside supported range')
    return value


def _geometry(speedlevel):
    level = _integer(speedlevel, 2, 4, 'speedlevel')
    if level == 2:
        return _Geometry(2, 348, 348, 40, 1, 2, 1)
    if level == 4:
        return _Geometry(4, 696, 1160, 141, 5, 10, 4)
    raise ValueError('speedlevel must be 2 or 4')


def _resources():
    # Validate both immutable C2 orders before a no-hit/eligibility decision.
    output = {}
    for sl, size in _TABLES:
        order = p4c2.permutation(size)
        p4c2.validate(order, size)
        output[sl] = order
    return output


def _eligible_members(raw, g):
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
                if score >= p4robustrx.MIN_SYNDROME:
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


def _frame(bits, g):
    net = 8*(g.data_bytes+3)
    wire = np.packbits(np.asarray(bits)[:net], bitorder='little').tobytes()
    return p4frame.decode_field(wire, data_bytes=g.data_bytes)


def _decode_member(member, g, order, iterations, work):
    fields = member.fields
    systematic = fields[:g.systematic]
    yield _Candidate('systematic', 0, systematic < 0)
    if not iterations:
        return
    # All D observations remain: no legacy suffix erasure or forced padding.
    p1, p2 = np.zeros(g.systematic), np.zeros(g.systematic)
    p1[::g.period] = fields[g.systematic:g.systematic+g.parity]
    p2[g.c2_phase::g.period] = fields[g.systematic+g.parity:]
    first = work.call('c1-only', g.systematic, p4decode.component_decode,
                     systematic, p1, initial_state=0, end_state=0)
    yield _Candidate('c1-only', 0, first.posterior < 0)
    prior = np.zeros(g.systematic)
    for iteration in range(1, iterations+1):
        first = work.call('iteration-c1', g.systematic, p4decode.component_decode,
                         systematic, p1, prior, initial_state=0, end_state=0)
        second = work.call('iteration-c2', g.systematic, p4decode.component_decode,
                          systematic[order], p2, first.extrinsic[order],
                          initial_state=0, end_state=None)
        prior[order] = second.extrinsic
        bits = np.empty(g.systematic, dtype=bool)
        bits[order] = second.posterior < 0
        yield _Candidate('turbo', iteration, bits)


def decode_placed(symbols, *, speedlevel: int, iterations: int = 8,
                  component_budget: int | None = None) -> DecodeOutcome:
    """Decode exactly one caller-placed reference plus robust LONG body."""
    started = time.monotonic()
    g = _geometry(speedlevel)
    iterations = _integer(iterations, 0, 8, 'iterations')
    if component_budget is not None:
        component_budget = _integer(component_budget, 0, None, 'component_budget')
    raw = np.asarray(symbols, dtype=complex)
    if raw.ndim != 1 or len(raw) != g.symbols+1 or not np.all(np.isfinite(raw)):
        raise ValueError('expected finite one-dimensional complete placed body')
    tables = _resources()
    preparation = time.monotonic()-started
    work = _Work(component_budget)
    found, provenance = {}, []
    completed = checks = 0
    members = ()
    ranking = crc_seconds = 0.0
    def finish(complete, reason):
        frames = tuple(found.values())
        selected = next((p for p in provenance if p.identity_index == 0), None)
        return DecodeOutcome(complete,
            frames[0] if complete and len(frames) == 1 else None,
            reason, g.speedlevel, iterations, len(members), completed,
            work.calls, work.positions, checks, len(frames), len(frames)>1,
            frames, tuple(provenance), selected if len(frames)==1 else None,
            (('preparation',preparation),('ranking',ranking),('crc',crc_seconds),*work.costs),
            time.monotonic()-started)
    if not np.any(raw):
        return finish(True, 'no-hit')
    ranking_started = time.monotonic()
    members = _eligible_members(raw, g)
    ranking = time.monotonic()-ranking_started
    if not members:
        return finish(True, 'layout-ineligible')
    try:
        for member_index, member in enumerate(members):
            for candidate in _decode_member(member, g, tables[g.speedlevel], iterations, work):
                checked = time.monotonic()
                frame = _frame(candidate.bits, g)
                crc_seconds += time.monotonic()-checked
                checks += 1
                index = None
                if frame is not None:
                    identity = (g.speedlevel, frame.data, frame.status.byte, frame.crc)
                    if identity not in found:
                        found[identity] = frame
                    index = tuple(found).index(identity)
                provenance.append(MethodOutcome(member_index, member.weighting,
                    member.quarter_turn, candidate.method, candidate.iteration,
                    frame is not None, index))
            completed += 1
    except _BudgetExhausted:
        return finish(False, 'incomplete-work')
    return finish(True, 'ambiguous' if len(found)>1 else 'unique' if found else 'no-hit')
