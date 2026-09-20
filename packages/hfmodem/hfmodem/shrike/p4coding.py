# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline primitives from SCS PACTOR-4 §§7–8, figures 7.1–7.2.

This is not a packet encoder. Initial states, C2's input permutation and
on-wire tail/padding placement remain unknown. Callers supply states and any
permutation explicitly; component termination is deliberately separate.

State (s1, s2, s3) lists the newest delay first. The diagram gives recursive
input u = x ^ s2 ^ s3, parity u ^ s1 ^ s3, next state (u, s1, s2).
Thus zero-state transfer is (1+D+D**3)/(1+D**2+D**3); octal tuple ordering
alone is insufficient to specify the convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

Bits = tuple[int, ...]
State = tuple[int, int, int]


def _bits(values: Iterable[int], name: str) -> Bits:
    result = tuple(values)
    if any(not isinstance(v, Integral) or v not in (0, 1) for v in result):
        raise ValueError(f"{name} must contain integer bits (0 or 1)")
    return tuple(int(v) for v in result)


def _state(values: Iterable[int]) -> State:
    state = _bits(values, "state")
    if len(state) != 3:
        raise ValueError("state must contain exactly three bits")
    return state[0], state[1], state[2]


@dataclass(frozen=True)
class ComponentResult:
    parity: Bits
    final_state: State


@dataclass(frozen=True)
class TerminationResult:
    systematic: Bits
    parity: Bits
    final_state: State


@dataclass(frozen=True)
class CodedFields:
    systematic: Bits
    c1: Bits
    c2: Bits

    @property
    def serialized(self) -> Bits:
        """Figure 7.2 field order; no symbol mapping or interleaving."""
        return self.systematic + self.c1 + self.c2


def rsc_encode(bits: Iterable[int], *, initial_state: State) -> ComponentResult:
    """Encode exactly the supplied bits without implicit reset or tail bits."""
    source = _bits(bits, "bits")
    s1, s2, s3 = _state(initial_state)
    parity = []
    for x in source:
        u = x ^ s2 ^ s3
        parity.append(u ^ s1 ^ s3)
        s1, s2, s3 = u, s1, s2
    return ComponentResult(tuple(parity), (s1, s2, s3))


def rsc_terminate(state: State) -> TerminationResult:
    """Three local flush inputs forcing u=0; NOT a P4 tail packing rule."""
    s1, s2, s3 = _state(state)
    systematic, parity = [], []
    for _ in range(3):
        systematic.append(s2 ^ s3)
        parity.append(s1 ^ s3)
        s1, s2, s3 = 0, s1, s2
    return TerminationResult(tuple(systematic), tuple(parity), (s1, s2, s3))


def encode_components(
    bits: Iterable[int], *, permutation: Iterable[int],
    initial_states: tuple[State, State],
) -> tuple[ComponentResult, ComponentResult]:
    """Encode two unterminated components using caller-supplied hypotheses.

    C2 input position j receives bits[permutation[j]]. No default P4
    interleaver is known. This permutation is distinct from symbol interleaving.
    """
    source = _bits(bits, "bits")
    order = tuple(permutation)
    if (any(not isinstance(v, Integral) or isinstance(v, bool) for v in order)
            or sorted(order) != list(range(len(source)))):
        raise ValueError("permutation must contain each input index exactly once")
    states = tuple(initial_states)
    if len(states) != 2:
        raise ValueError("initial_states must contain two component states")
    return (rsc_encode(source, initial_state=states[0]),
            rsc_encode((source[i] for i in order), initial_state=states[1]))


def puncture(
    systematic: Iterable[int], c1: Iterable[int], c2: Iterable[int], *, rate: str,
) -> CodedFields:
    """Puncture equal-length fields, restarting the pattern at their first bit.

    Rates are nominal: partial periods are allowed, so actual length ratios can
    differ. C1/C2 keep zero-based indices 0/1 modulo 2 for '1/2', 0/4 modulo
    10 for '5/6'. '1/3' retains both fields in full (derived unpunctured case).
    No tail-specific mask or reset is inferred. To continue a pattern across
    chunks, concatenate the unpunctured fields before calling this function.
    """
    v0, v1, v2 = (_bits(systematic, "systematic"), _bits(c1, "c1"),
                  _bits(c2, "c2"))
    if not len(v0) == len(v1) == len(v2):
        raise ValueError("unpunctured fields must have equal lengths")
    if rate == "1/3":
        return CodedFields(v0, v1, v2)
    if rate == "1/2":
        return CodedFields(v0, v1[0::2], v2[1::2])
    if rate == "5/6":
        return CodedFields(v0, v1[0::10], v2[4::10])
    raise ValueError("rate must be '1/3', '1/2', or '5/6'")
