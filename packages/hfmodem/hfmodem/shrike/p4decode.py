# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline P4 component/turbo observations with explicit input permutation.

LLRs are log P(bit0)/P(bit1). The figure-derived K4 component uses a uniform
initial state and unconstrained final state; neither packet tail packing nor
termination is inferred. This module has no RF or packet-acceptance integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

_STATE = np.arange(8)[:, None]
_INPUT = np.arange(2)[None, :]
_S1, _S2, _S3 = (_STATE >> 2) & 1, (_STATE >> 1) & 1, _STATE & 1
_U = _INPUT ^ _S2 ^ _S3
_NEXT = (_U << 2) | (_S1 << 1) | _S2
_PARITY = _U ^ _S1 ^ _S3
# Each destination has two incoming branches, indexed by its predecessor s3.
_DEST = np.arange(8)[:, None]
_OLD_S3 = np.arange(2)[None, :]
_PREV = (((_DEST >> 1) & 1) << 2) | ((_DEST & 1) << 1) | _OLD_S3
_INCOMING_INPUT = ((_DEST >> 2) & 1) ^ (_DEST & 1) ^ _OLD_S3


def _llrs(values, name):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real LLRs")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite real LLRs") from error
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    return result


@dataclass(frozen=True)
class ComponentDecode:
    posterior: np.ndarray
    extrinsic: np.ndarray


def component_decode(systematic, parity, apriori=None) -> ComponentDecode:
    """Exact log-MAP BCJR; zero parity LLR means an erased observation.

    Extrinsic excludes the local systematic observation and local a-priori LLR.
    All eight start states have equal weight; all end states are allowed.
    Inputs are not modified. Extremely large finite inputs that overflow the
    log-domain arithmetic raise ValueError instead of returning invalid results.
    """
    sys = _llrs(systematic, "systematic")
    par = _llrs(parity, "parity")
    prior = np.zeros_like(sys) if apriori is None else _llrs(apriori, "apriori")
    if sys.shape != par.shape or sys.shape != prior.shape:
        raise ValueError("systematic, parity and apriori must have equal lengths")
    with np.errstate(over="ignore", invalid="ignore"):
        local = sys + prior
        input_log = -np.logaddexp(0., -(1 - 2 * _INPUT) * local[:, None])
        parity_log = -np.logaddexp(0., -(1 - 2 * _PARITY) * par[:, None, None])
        gamma = input_log[:, None, :] + parity_log
    if not np.all(np.isfinite(gamma)):
        raise ValueError("LLR magnitudes overflow component arithmetic")
    n = len(sys)
    alpha = np.empty((n + 1, 8))
    beta = np.empty((n + 1, 8))
    alpha[0] = -np.log(8.)
    beta[n] = 0.
    for t in range(n):
        branch = alpha[t, _PREV] + gamma[t, _PREV, _INCOMING_INPUT]
        alpha[t + 1] = np.logaddexp(branch[:, 0], branch[:, 1])
        alpha[t + 1] -= np.logaddexp.reduce(alpha[t + 1])
    for t in range(n - 1, -1, -1):
        branch = gamma[t] + beta[t + 1, _NEXT]
        beta[t] = np.logaddexp(branch[:, 0], branch[:, 1])
        beta[t] -= np.logaddexp.reduce(beta[t])
    # Omitting local input evidence here avoids posterior-minus-large-prior
    # cancellation and explicitly defines what may be exchanged by turbo passes.
    branch = alpha[:-1, :, None] + parity_log + beta[1:, _NEXT]
    marginal = np.logaddexp.reduce(branch, axis=1)
    extrinsic = marginal[:, 0] - marginal[:, 1]
    posterior = extrinsic + local
    if not np.all(np.isfinite(posterior)) or not np.all(np.isfinite(extrinsic)):
        raise ValueError("LLR magnitudes overflow component arithmetic")
    return ComponentDecode(posterior, extrinsic)


@dataclass(frozen=True)
class IterationDiagnostic:
    iteration: int
    hard_changes: int
    mean_abs_posterior: float
    max_abs_extrinsic: float


@dataclass(frozen=True)
class TurboDecode:
    posterior: np.ndarray
    bits: np.ndarray
    diagnostics: tuple[IterationDiagnostic, ...]


def turbo_decode(systematic, c1, c2, permutation, iterations=8) -> TurboDecode:
    """Iterative rate1/3 decode under a caller-supplied C2 input ordering.

    C2 input[j] = systematic[permutation[j]]. C2 parity stays in its own time
    order. Exchange only constituent extrinsic information; never feed a full
    posterior back as a prior. No clipping, tail removal, early stopping or
    automatic packet acceptance. Posterior0 is a tie, represented as hard bit0.
    Convergence and confidence diagnostics do not establish a valid packet.
    """
    sys = _llrs(systematic, "systematic")
    p1, p2 = _llrs(c1, "c1"), _llrs(c2, "c2")
    if sys.shape != p1.shape or sys.shape != p2.shape:
        raise ValueError("systematic and both parity fields must have equal lengths")
    raw_order = np.asarray(permutation)
    if (raw_order.ndim != 1 or raw_order.shape != sys.shape
            or raw_order.dtype.kind not in "iu"
            or not np.array_equal(np.sort(raw_order), np.arange(len(sys)))):
        raise ValueError("permutation must contain each input index exactly once")
    order = raw_order.astype(np.intp)
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, Integral) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    prior = np.zeros_like(sys)
    posterior = sys.copy()
    previous_bits = sys < 0
    diagnostics = []
    for iteration in range(1, int(iterations) + 1):
        first = component_decode(sys, p1, prior)
        second = component_decode(sys[order], p2, first.extrinsic[order])
        prior[order] = second.extrinsic
        posterior[order] = second.posterior
        bits = posterior < 0
        diagnostics.append(IterationDiagnostic(
            iteration, int(np.count_nonzero(bits != previous_bits)),
            float(np.mean(np.abs(posterior))), float(np.max(np.abs(prior))),
        ))
        previous_bits = bits
    return TurboDecode(posterior, previous_bits.astype(np.uint8), tuple(diagnostics))
