# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Exact MAP LONG search over a frozen domain of serialized LLRs."""
from dataclasses import dataclass, field
from functools import lru_cache
import logging
from math import exp, log
from numbers import Integral, Real
import time

import numpy as np

from . import p4c2, p4coding, p4decode, p4frame

# Bit one needs a posterior below -TIE in both backends.
TIE = 1e-9


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < low or (high is not None and value > high):
        raise ValueError(f'{name} outside supported integer range')
    return int(value)


@dataclass(frozen=True)
class LongGeometry:
    speedlevel: int
    systematic: int
    data_bytes: int
    padding: int
    c1_period: int
    c1_phase: int
    c2_period: int
    c2_phase: int

    @property
    def symbols(self):
        return self.serialized_size // 2 if self.speedlevel <= 4 else 4968

    @property
    def order(self):
        return _order(self.speedlevel)

    @property
    def c1_slice(self):
        return slice(self.c1_phase, None, self.c1_period)

    @property
    def c2_slice(self):
        return slice(self.c2_phase, None, self.c2_period)

    @property
    def parity(self):
        return len(range(self.c1_phase, self.systematic, self.c1_period))

    @property
    def serialized_size(self):
        return self.systematic + self.parity + len(range(self.c2_phase, self.systematic, self.c2_period))

    @property
    def regions(self):
        return (('systematic', 0, self.systematic),
                ('retained-c1', self.systematic, self.systematic + self.parity),
                ('retained-c2', self.systematic + self.parity, self.serialized_size))


_GEOMETRIES = tuple(LongGeometry(*row) for row in (
    (2, 348, 40, 1, 2, 0, 2, 1),
    (3, 580, 69, 1, 10, 0, 10, 4),
    (4, 1160, 141, 5, 10, 0, 10, 4),
    (5, 1656, 203, 5, 1, 0, 1, 0),
    (6, 4140, 514, 1, 10, 0, 10, 4),
    (7, 8280, 1031, 5, 10, 0, 10, 4),
    (8, 12420, 1549, 1, 10, 0, 10, 4),
    (9, 16560, 2066, 5, 10, 0, 10, 4),
    (10, 20700, 2584, 1, 10, 0, 10, 4),
))


def geometry(speedlevel):
    return _GEOMETRIES[_integer(speedlevel, 2, 10, 'speedlevel') - 2]


@lru_cache(maxsize=None)
def _order(speedlevel):
    g = geometry(speedlevel)
    if speedlevel == 3:
        order = np.concatenate([np.arange(j, g.systematic, 24) for j in range(24)])
        if not np.array_equal(np.sort(order), np.arange(g.systematic)):
            raise ValueError('invalid SL3 gather order')
    else:
        order = p4c2.permutation(g.systematic)
        p4c2.validate(order, g.systematic)
    return np.frombuffer(order.astype(np.intp).tobytes(), dtype=np.intp)


@dataclass(frozen=True)
class LongMember:
    geometry: LongGeometry
    fields: np.ndarray = field(repr=False, compare=False)
    quarter_turn: int = 0
    weighting: str = 'unit'
    polarity: int = 1
    label: str = ''

    def __post_init__(self):
        if not isinstance(self.geometry, LongGeometry) or self.geometry != geometry(self.geometry.speedlevel):
            raise ValueError('expected canonical LONG geometry')
        _integer(self.quarter_turn, 0, 3, 'quarter_turn')
        if (isinstance(self.polarity, (bool, np.bool_)) or self.polarity not in (1, -1)
                or not isinstance(self.polarity, Integral)
                or not isinstance(self.weighting, str) or not isinstance(self.label, str)):
            raise ValueError('expected immutable LONG provenance with polarity +1 or -1')
        a = np.asarray(self.fields)
        if a.shape != (self.geometry.serialized_size,) or a.dtype.kind not in 'fiu' or not np.all(np.isfinite(a)):
            raise ValueError('expected finite real serialized LLR vector of declared length')
        self.geometry.order
        object.__setattr__(self, 'fields', np.frombuffer(a.astype(np.float64).tobytes(), dtype=np.float64))


@dataclass(frozen=True)
class RegionAgreement:
    region: str
    hard_disagreements: int
    agreement_fraction: float
    weighted_support: float


@dataclass(frozen=True)
class CandidateOutcome:
    member_index: int
    method: str
    iteration: int
    crc_valid: bool
    padding_valid: bool
    flush_valid: bool
    identity_index: int | None
    agreements: tuple[RegionAgreement, ...] = ()


@dataclass(frozen=True)
class LongResult:
    state: str
    frame: p4frame.Frame | None
    identities: tuple
    completed_tier: int
    provenance: tuple[CandidateOutcome, ...]
    selected_provenance: CandidateOutcome | None
    stage_seconds: tuple
    elapsed_seconds: float
    component_calls: int
    constituent_positions: int

    @property
    def complete(self):
        return self.state != 'incomplete'


def _check(bits, member, member_index, tier, found):
    bits = np.asarray(bits, dtype=np.uint8)
    g = member.geometry
    net = 8 * (g.data_bytes + 3)
    frame = p4frame.decode_field(np.packbits(bits[:net], bitorder='little').tobytes(), data_bytes=g.data_bytes)
    padding = not np.any(bits[net:-3])
    agreements, index = (), None
    flush = False
    if frame is not None:
        first = p4coding.rsc_encode(bits, initial_state=(0, 0, 0))
        flush = first.final_state == (0, 0, 0)
        second = p4coding.rsc_encode(bits[g.order], initial_state=(0, 0, 0))
        coded = np.concatenate((bits, np.asarray(first.parity)[g.c1_slice], np.asarray(second.parity)[g.c2_slice]))
        rows = []
        for name, lo, hi in g.regions:
            observed, expected = member.fields[lo:hi], coded[lo:hi]
            errors = int(np.count_nonzero((observed < 0) != expected))
            # Scaling before summation keeps support finite for extreme LLRs.
            scale = np.max(abs(observed))
            weights = observed / scale if scale else observed
            # Match the native left-to-right reduction at the strict zero boundary.
            total = support = 0.0
            for w, b in zip(weights, expected):
                value = float(w)
                total += abs(value)
                support += value * (1 - 2 * int(b))
            support = support / total if total else 0.0
            rows.append(RegionAgreement(name, errors, 1 - errors / len(observed), support))
        agreements = tuple(rows)
        if padding and flush and all(r.weighted_support > 0 for r in agreements):
            key = (g.speedlevel, frame.status.byte, frame.data, frame.crc)
            if key not in found:
                found[key] = frame
            index = tuple(found).index(key)
    return CandidateOutcome(member_index, 'systematic' if tier == 0 else 'c1-only' if tier == 1 else 'turbo',
                            max(0, tier - 1), frame is not None, bool(padding), flush, index, agreements)


def _member_step(g, fields, prior, first_extrinsic, tier):
    """One boundary; C1-only supplies the C1 half of turbo iteration one."""
    n, order = g.systematic, g.order
    systematic = fields[:n]
    if tier == 0:
        return systematic < -TIE, prior, first_extrinsic, 0
    c1, c2 = np.zeros(n), np.zeros(n)
    c1[g.c1_slice] = fields[n:n + g.parity]
    c2[g.c2_slice] = fields[n + g.parity:]
    calls = 0
    if tier != 2:
        posterior, first = _components((systematic + prior)[None], c1[None], end_state=0)
        first_extrinsic = first[0]
        calls += 1
        if tier == 1:
            return posterior[0] < -TIE, prior, first_extrinsic, calls
    posterior, second = _components((systematic[order] + first_extrinsic[order])[None],
                                    c2[None], end_state=None)
    next_prior, bits = np.empty(n), np.empty(n, dtype=bool)
    next_prior[order], bits[order] = second[0], posterior[0] < -TIE
    return bits, next_prior, first_extrinsic, calls + 1


def _probabilities(llrs):
    tail = np.fromiter((exp(-abs(x)) for x in llrs.flat), float, llrs.size).reshape(llrs.shape)
    hi, lo = 1 / (1 + tail), tail / (1 + tail)
    return np.stack((np.where(llrs >= 0, hi, lo), np.where(llrs >= 0, lo, hi)), axis=-1)


def _components(systematic, parity, *, end_state):
    """Probability-domain MAP with the native kernel's scalar reduction order.

    Member batching never changes a member's arithmetic. Scalar libm exp/log
    and ordered sums keep threshold decisions on the same quantity as Rust.
    """
    sys, par = np.clip(systematic, -40, 40), np.clip(parity, -40, 40)
    count, n = sys.shape
    ip, pp = _probabilities(sys), _probabilities(par)
    gamma = ip[:, :, None, :] * pp[:, :, p4decode._PARITY]
    metrics = np.empty((n+1, count, 2, 8))
    metrics[0] = 0.
    metrics[0, :, 0, 0] = 1.
    metrics[0, :, 1] = .125
    if end_state is not None:
        metrics[0, :, 1] = 0.
        metrics[0, :, 1, end_state] = 1.
    edges = np.stack((gamma[:, :, p4decode._PREV, p4decode._INCOMING_INPUT],
                      gamma[:, ::-1]), axis=2).transpose(1, 0, 2, 3, 4)
    flat = metrics.reshape(n+1, count, 16)
    workspace = np.empty((count, 2, 8, 2))
    for t in range(n):
        np.take(flat[t], p4decode._FLAT_NEIGHBORS, axis=1,
                out=workspace, mode='clip')
        np.multiply(workspace, edges[t], out=workspace)
        row = metrics[t+1]
        np.add(workspace[..., 0], workspace[..., 1], out=row)
        row /= np.add.accumulate(row, axis=2)[:, :, -1, None]
    alpha = metrics[:, :, 0].transpose(1, 0, 2)
    beta = metrics[::-1, :, 1].transpose(1, 0, 2)
    branch = alpha[:, :-1, :, None] * pp[:, :, p4decode._PARITY] * beta[:, 1:, p4decode._NEXT]
    marginal = np.add.accumulate(branch, axis=2)[:, :, -1]
    logs = np.fromiter((log(x) for x in marginal.flat), float, marginal.size).reshape(marginal.shape)
    extrinsic = logs[:, :, 0] - logs[:, :, 1]
    return extrinsic + sys, extrinsic


def _batch_step(job, indices, tier):
    g = job.members[indices[0]].geometry
    n, order = g.systematic, g.order
    fields = job.llrs[indices]
    systematic = fields[:, :n]
    c1, c2 = np.zeros_like(systematic), np.zeros_like(systematic)
    c1[:, g.c1_slice] = fields[:, n:n+g.parity]
    c2[:, g.c2_slice] = fields[:, n+g.parity:g.serialized_size]
    first = job.first[indices, :n]
    prior = job.priors[indices, :n]
    if tier != 2:
        posterior, first = _components(systematic+prior, c1, end_state=0)
        if tier == 1:
            return posterior < -TIE, prior, first
    posterior, second = _components(systematic[:, order]+first[:, order], c2,
                                    end_state=None)
    prior = np.empty_like(second)
    bits = np.empty_like(second, dtype=bool)
    prior[:, order], bits[:, order] = second, posterior < -TIE
    return bits, prior, first


# Validation has its own checkpoint; its reserve cannot fund a constituent.
# Warm and fresh-input calls were measured at nice 19. Reserves cover its
# measured maxima (SL7 32.450 ms, SL8 34.902 ms) with at least 25% margin.
_PYTHON_COMPONENT_SECONDS = (.0032, .0045, .006, .009, .030, .050, .060, .075, .094)


def _python_check_seconds(g):
    return .0005 + g.systematic * .00000075


def _python_step_seconds(g, tier, count=1):
    if not tier:
        return _python_check_seconds(g)
    calls = 1 if tier in (1, 2) else 2
    if count == 1:
        return calls * _PYTHON_COMPONENT_SECONDS[g.speedlevel - 2]
    # The independent member axis amortizes the recurrence loop. The 24-member
    # SL3 component measured 10.070 ms at nice 19; reserve 17.08 ms per
    # component. Smaller grants shrink the batch, never the frozen domain.
    return calls * (.002 + g.systematic * (2 + count) * .000001)


def _python_advance(job, deadline):
    """Whole-grant backend seam: owned stacked LLRs/priors, geometries, tier.

    A native replacement owns this loop and checks identities at each boundary;
    it must not call Python once per member or constituent.
    """
    while job.tier <= job.last_tier:
        if deadline is not None and time.monotonic() >= deadline:
            return
        tier = job.tier
        calls = 0 if tier == 0 else 1 if tier in (1, 2) else 2
        while job.member_index < len(job.members):
            i = job.member_index
            needed = 0 if i in job.pending else calls
            if (deadline is not None and time.monotonic() >= deadline
                    or job.component_budget is not None and job.calls + needed > job.component_budget):
                return
            member = job.members[i]
            indices = []
            if (i not in job.pending and tier and job.component_budget is None
                    and member.geometry.systematic <= 1160 and len(job.members) >= 16):
                indices = [j for j in range(i, len(job.members))
                           if job.members[j].geometry == member.geometry][:24]
            if deadline is not None:
                remaining = deadline - time.monotonic()
                while len(indices) > 1 and _python_step_seconds(member.geometry, tier, len(indices)) > remaining:
                    indices.pop()
                step_tier = 0 if i in job.pending else tier
                if _python_step_seconds(member.geometry, step_tier, max(1, len(indices))) > remaining:
                    return
            started = time.monotonic()
            n = member.geometry.systematic
            if i in job.pending:
                bits, prior, first = job.pending.pop(i)
                used = 0
            elif len(indices) > 1:
                # Bound memory and the interval between deadline checks, even
                # for diagnostic domains with more placements than live SF16.
                bits, priors, firsts = _batch_step(job, indices, tier)
                job.calls += calls*len(indices)
                job.positions += calls*len(indices)*n
                job.pending.update((j, (b, p, f))
                                   for j, b, p, f in zip(indices, bits, priors, firsts))
                bits, prior, first = job.pending.pop(i)
                used = 0
            else:
                bits, prior, first, used = _member_step(
                    member.geometry, job.llrs[i, :member.geometry.serialized_size],
                    job.priors[i, :n], job.first[i, :n], tier)
            job.priors[i, :n], job.first[i, :n] = prior, first
            job.calls += used
            job.positions += used * n
            job.costs.append((f'tier-{tier}-decode', time.monotonic() - started))
            if (tier and deadline is not None
                    and deadline-time.monotonic() < _python_check_seconds(member.geometry)):
                # The complete recurrence is owned evidence. Park its bits
                # and messages until validation has its own admitted grant;
                # neither recompute the constituent nor publish a partial tier.
                job.pending[i] = bits, prior, first
                return
            checked = time.monotonic()
            job.outcomes.append(_check(bits, member, i, tier, job.found))
            job.costs.append((f'tier-{tier}-check', time.monotonic() - checked))
            job.member_index += 1
        job.member_index = 0
        job.tier += 1
        if job.found and job.policy == 'live':
            job.done = True
            return
    job.done = True


class LongSearch:
    def __init__(self, members, *, policy='live', iterations=8, component_budget=None):
        if policy not in ('live', 'exhaustive'):
            raise ValueError('policy must be live or exhaustive')
        self.members = tuple(members)
        if any(not isinstance(m, LongMember) for m in self.members):
            raise ValueError('expected LONG members')
        self.policy = policy
        iterations = _integer(iterations, 0, 8, 'iterations')
        self.last_tier = iterations + 1 if iterations else 0
        self.component_budget = None if component_budget is None else _integer(component_budget, 0, None, 'component_budget')
        width = max((m.geometry.systematic for m in self.members), default=0)
        self.llrs = np.zeros((len(self.members), max((m.geometry.serialized_size for m in self.members), default=0)))
        for i, m in enumerate(self.members):
            self.llrs[i, :len(m.fields)] = m.fields
        self.llrs = np.frombuffer(self.llrs.tobytes(), dtype=np.float64).reshape(self.llrs.shape)
        self.priors = np.zeros((len(self.members), width))
        self.first = np.zeros_like(self.priors)
        self.tier = self.calls = self.positions = 0
        self.member_index = 0
        self.backend_failed = False
        self.pending = {}
        self.found, self.outcomes, self.costs = {}, [], []
        self.elapsed = 0.0
        self.done = not any(np.any(m.fields) for m in self.members)

    def advance(self, deadline=None):
        if deadline is not None and (isinstance(deadline, (bool, np.bool_))
                                     or not isinstance(deadline, Real) or not np.isfinite(deadline)):
            raise ValueError('deadline must be finite monotonic seconds')
        started = time.monotonic()
        if not self.done:
            backend = None if self.backend_failed else p4longnative.backend
            if backend is None:
                _python_advance(self, deadline)
            else:
                try:
                    backend(self, deadline)
                except RuntimeError as error:
                    logging.getLogger(__name__).warning('P4 LONG backend failed: %s; retrying in Python', error)
                    elapsed = self.elapsed
                    fresh = LongSearch(self.members, policy=self.policy,
                                       iterations=max(0, self.last_tier - 1),
                                       component_budget=self.component_budget)
                    self.__dict__.clear()
                    self.__dict__.update(fresh.__dict__)
                    self.elapsed = elapsed
                    self.backend_failed = True
                    _python_advance(self, deadline)
        self.elapsed += time.monotonic() - started
        identities = tuple(self.found.values())
        state = 'incomplete' if not self.done else 'ambiguous' if len(identities) > 1 else 'unique' if identities else 'unread'
        selected = next((p for p in self.outcomes if p.identity_index == 0), None) if len(identities) == 1 else None
        return LongResult(state, identities[0] if state == 'unique' else None, identities,
                          self.tier - 1, tuple(self.outcomes), selected, tuple(self.costs),
                          self.elapsed, self.calls, self.positions)


# Load and self-test before the first work grant, after the reference is defined.
from . import p4longnative
