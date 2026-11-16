# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Placed normal-LONG SL6--10 diagnostics and live receiver dispatch.

Constants are immutable generated arrays. The symbol noise parameter
is complex power E|n|**2 (each real axis has variance noise/2). Body-derived
noise is a diagnostic residual-ISI proxy, not a calibrated channel estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral
import time

import numpy as np
from scipy.special import logsumexp

from . import p4c2, tablegen, p4coding, p4frame, p4long, p4normal, p4rx
from . import p4longdecode

LEVELS = (6, 7, 8, 9, 10)
SYMBOLS = 4968
SYSTEMATIC = (4140, 8280, 12420, 16560, 20700)
NET_BYTES = (517, 1034, 1552, 2069, 2587)


@dataclass(frozen=True)
class Geometry:
    speedlevel: int
    width: int
    systematic: int
    net_bytes: int

    @property
    def data_bytes(self):
        return self.net_bytes - 3

    @property
    def padding(self):
        return self.systematic - 8 * self.net_bytes - 3

    @property
    def parity_each(self):
        return self.systematic // 10


def geometry(speedlevel):
    if isinstance(speedlevel, (bool, np.bool_)) or not isinstance(speedlevel, Integral) or speedlevel not in LEVELS:
        raise ValueError("speedlevel must be an integer from 6 through 10")
    i = int(speedlevel) - 6
    return Geometry(int(speedlevel), i + 1, SYSTEMATIC[i], NET_BYTES[i])


@lru_cache(maxsize=1)
def _tables():
    result = {"symbols": tablegen.p4_symbol_permutation()}
    for sl, size in zip(LEVELS, SYSTEMATIC):
        order = p4c2.permutation(size)
        p4c2.validate(order, size)
        result[f"c2_{sl}"] = order
        points = tablegen.p4_constellation_points(sl)
        values = (points[:, 0].astype(float) + 1j * points[:, 1]) / tablegen.P4_POINT_SCALE
        result[f"points_{sl}"] = np.frombuffer(values.tobytes(), dtype=values.dtype)
    return result


def constellation(speedlevel):
    return _tables()[f"points_{geometry(speedlevel).speedlevel}"]


def component_permutation(speedlevel):
    return _tables()[f"c2_{geometry(speedlevel).speedlevel}"]


def symbol_permutation():
    return _tables()["symbols"]


def _real_array(values, length):
    a = np.asarray(values)
    if a.shape != (length,) or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
        raise ValueError("expected a finite real vector of the declared length")
    return a.astype(float)


def _symbols(values):
    a = np.asarray(values)
    if a.shape != (SYMBOLS,) or a.dtype.kind not in "cfiu" or not np.all(np.isfinite(a)):
        raise ValueError("expected 4968 finite complex symbols")
    return a.astype(complex)


def _iterations(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 1 <= value <= 8:
        raise ValueError("iterations must be an integer from 1 through 8")
    return int(value)


def _budget(value):
    if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0):
        raise ValueError("max_work must be None or a nonnegative iteration count")
    return value


def _noise(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value) or not np.isreal(value) or not np.isfinite(value) or value <= 0:
        raise ValueError("noise must be finite positive complex noise power")
    return float(value)


@lru_cache(maxsize=None)
def _demap_groups(speedlevel):
    """Both bit classes, so tiny complements are never formed by subtraction."""
    g = geometry(speedlevel)
    labels = np.arange(1 << g.width)
    groups = np.column_stack([
        ((labels >> bit) & 1) == value
        for bit in range(g.width - 1, -1, -1) for value in (0, 1)
    ]).astype(float)
    return np.frombuffer(groups.tobytes(), dtype=float).reshape(groups.shape)


def demap(symbols, noise, speedlevel):
    """Return field-order, MSB label LLRs; positive values mean bit zero."""
    g = geometry(speedlevel)
    z, n0 = _symbols(symbols), _noise(noise)
    points = constellation(speedlevel)
    # The symbol-only energy cancels in every bit-class likelihood ratio.
    metrics = (2 * (z.real[:, None] * points.real + z.imag[:, None] * points.imag)
               - np.abs(points)**2) / n0
    if not np.all(np.isfinite(metrics)):
        raise ValueError("demapper metric overflow")
    if g.width == 1:
        # Each bit class has one point, so no log-sum-exp is needed.
        llr = (metrics[:, :1] - metrics[:, 1:2])
    else:
        # Every bit class shares the row normalization. Exponentiate each
        # constellation likelihood once, rather than once per label bit.
        weights = np.exp(metrics - np.max(metrics, axis=1)[:, None])
        sums = weights @ _demap_groups(speedlevel)
        paired = sums.reshape(len(z), g.width, 2)
        stable = np.all(paired >= np.finfo(float).tiny, axis=2)
        logs = np.log(np.maximum(paired, np.finfo(float).tiny))
        llr = logs[:, :, 0] - logs[:, :, 1]
        # Extreme finite inputs may underflow one whole bit class under
        # the shared normalization. Preserve separately shifted sums.
        if not np.all(stable):
            labels = np.arange(1 << g.width)
            for column, bit in enumerate(range(g.width - 1, -1, -1)):
                missing = ~stable[:, column]
                if not np.any(missing):
                    continue
                subset = -abs(z[missing, None] - points)**2 / n0
                llr[missing, column] = (
                    logsumexp(subset[:, ((labels >> bit) & 1) == 0], axis=1)
                    - logsumexp(subset[:, ((labels >> bit) & 1) == 1], axis=1))
    return llr[symbol_permutation()].ravel()


def build_members(observed):
    """Rank the complete normal domain using one training-fixed noise model."""
    noise = max(observed.noise(6), 1e-12)
    subset = observed.symbols[::16]
    ranked = []
    for sl in range(5, 11):
        points = constellation(max(6, sl))
        metrics = -abs(subset[:, None] - points)**2 / noise
        score = float(np.mean(logsumexp(metrics, axis=1) - np.log(len(points))))
        ranked.append((score, sl))
    fields = {}
    members = []
    for _, sl in sorted(ranked, key=lambda row: (-row[0], row[1])):
        modulation = max(6, sl)
        if modulation not in fields:
            fields[modulation] = demap(observed.symbols, noise, modulation)
        for polarity in ((1, -1) if sl <= 6 else (1,)):
            members.append(p4longdecode.LongMember(p4longdecode.geometry(sl),
                fields[modulation] * polarity, polarity=polarity, weighting='pilot'))
    return tuple(members)


@dataclass(frozen=True)
class DecodeResult:
    complete: bool
    reason: str
    frame: p4frame.Frame | None = field(repr=False)
    identity_count: int = 0
    modes_completed: tuple[int, ...] = ()
    iterations_done: int = 0
    crc_checks: int = 0
    elapsed_seconds: float = 0.0
    declared_modes: tuple[int, ...] = ()
    declared_iterations: int = 8
    identities: tuple = field(default=(), repr=False)
    outcomes: tuple = ()
    stage_seconds: tuple = ()
    coordinates: object = None
    noise_diagnostics: tuple = ()
    observation: object = field(default=None, repr=False)
    observation_eligible: bool = True


@dataclass(frozen=True)
class SourceCoordinates:
    input_stage: str
    first_training_start: int | None = None
    frequency_offset: float | None = None
    input_samples: int | None = None


@dataclass(frozen=True)
class NoiseDiagnostic:
    speedlevel: int
    mean_symbol_energy: float
    complex_variance: float
    provenance: str
    convention: str = "total-complex-variance"
    calibrated_recording_snr: bool = False


@dataclass(frozen=True)
class RegionDisagreement:
    region: str
    serialized_start: int
    serialized_stop: int
    hard_disagreements: int
    normalized_complex_distance: float | None


@dataclass(frozen=True)
class ReencodingDiagnostic:
    regions: tuple
    full_hard_disagreements: int
    normalized_complex_distance: float | None
    normalization: str = "mean-squared-complex-error / mean-reference-symbol-energy"
    tail_definition: str = "systematic alignment padding and C1 flush; no received samples erased"


@dataclass(frozen=True)
class IterationOutcome:
    speedlevel: int
    iteration: int
    crc_valid: bool
    padding_valid: bool
    flush_valid: bool
    identity_index: int | None
    reencoding: ReencodingDiagnostic | None
    method: str = "turbo"


def _check_frame(bits, g):
    net = 8 * g.net_bytes
    padding_ok = not bool(np.any(bits[net:net + g.padding]))
    wire = np.packbits(bits[:net], bitorder="little").tobytes()
    decoded = p4frame.decode_field(wire, data_bytes=g.data_bytes)
    encoded = p4coding.rsc_encode(bits[:-3], initial_state=(0, 0, 0))
    termination = p4coding.rsc_terminate(encoded.final_state)
    flush_ok = bool(np.array_equal(bits[-3:], termination.systematic))
    return decoded, padding_ok, flush_ok


def _reencoding(bits, g, fields, symbols):
    # Regenerate the complete canonical codeword, including alignment and flush.
    # Observations, equalizer coefficients and acceptance never depend on this.
    net = 8 * g.net_bytes
    prefix = np.concatenate((bits[:net], np.zeros(g.padding, dtype=np.uint8)))
    first = p4coding.rsc_encode(prefix, initial_state=(0, 0, 0))
    flush = p4coding.rsc_terminate(first.final_state)
    systematic = np.concatenate((prefix, flush.systematic))
    c1 = p4coding.rsc_encode(systematic, initial_state=(0, 0, 0)).parity
    c2 = p4coding.rsc_encode(systematic[component_permutation(g.speedlevel)],
                             initial_state=(0, 0, 0)).parity
    coded = np.concatenate((systematic, c1[::10], c2[4::10]))
    hard = fields < 0
    disagreement = hard != coded
    expected = constellation(g.speedlevel)[
        coded.reshape(-1, g.width) @ (1 << np.arange(g.width - 1, -1, -1))]
    observed = None if symbols is None else symbols[symbol_permutation()]
    energy = float(np.mean(np.abs(expected)**2))
    def distance(start, stop):
        if observed is None:
            return None
        # Tail bit boundaries can intersect symbols; include each intersecting
        # symbol once. Coding-region boundaries are exact symbol boundaries.
        lo, hi = start // g.width, (stop + g.width - 1) // g.width
        return float(np.mean(np.abs(observed[lo:hi] - expected[lo:hi])**2) / energy)
    bounds = (("systematic", 0, g.systematic),
              ("retained-c1", g.systematic, g.systematic + g.parity_each),
              ("retained-c2", g.systematic + g.parity_each, len(coded)),
              ("alignment-padding", net, net + g.padding),
              ("c1-flush", g.systematic - 3, g.systematic))
    regions = tuple(RegionDisagreement(name, start, stop,
                    int(np.count_nonzero(disagreement[start:stop])),
                    distance(start, stop)) for name, start, stop in bounds)
    return ReencodingDiagnostic(regions, int(np.count_nonzero(disagreement)),
                                distance(0, len(coded)))


def _decode(members, iterations, max_work, started, *, declared_modes=None,
            symbols=None, coordinates=None, noise_diagnostics=(),
            observation=None, stage_seconds=None):
    """Adapt the shared exhaustive engine result to the diagnostic API.

    Work is measured in member turbo iterations: two constituent calls
    per work unit. The engine admits complete tiers across the whole domain,
    and also reports systematic and C1-only outcomes explicitly.
    """
    members = tuple(members)
    domain = tuple(g.speedlevel for g, _ in members) if declared_modes is None else tuple(declared_modes)
    canonical = tuple(p4longdecode.LongMember(p4longdecode.geometry(g.speedlevel), fields)
                      for g, fields in members)
    search = p4longdecode.LongSearch(canonical, policy='exhaustive', iterations=iterations,
                                   component_budget=None if max_work is None else 2 * max_work)
    result = search.advance()
    outcomes, diagnostics = [], {}
    reencode_seconds = 0.0
    for outcome in result.provenance:
        member = canonical[outcome.member_index]
        g = geometry(member.geometry.speedlevel)
        diagnostic = None
        if outcome.identity_index is not None:
            key = (outcome.member_index, outcome.identity_index)
            if key not in diagnostics:
                reencode_started = time.monotonic()
                frame = result.identities[outcome.identity_index]
                wire = p4frame.encode_field(frame.data, frame.status)
                prefix = np.r_[np.unpackbits(np.frombuffer(wire, np.uint8), bitorder='little'),
                               np.zeros(g.padding, dtype=np.uint8)]
                first = p4coding.rsc_encode(prefix, initial_state=(0, 0, 0))
                bits = np.r_[prefix, p4coding.rsc_terminate(first.final_state).systematic]
                diagnostics[key] = _reencoding(bits, g, member.fields, symbols)
                reencode_seconds += time.monotonic() - reencode_started
            diagnostic = diagnostics[key]
        outcomes.append(IterationOutcome(g.speedlevel, outcome.iteration,
            outcome.crc_valid, outcome.padding_valid, outcome.flush_valid,
            outcome.identity_index, diagnostic, outcome.method))
    costs = dict(stage_seconds or ())
    costs.update(result.stage_seconds)
    costs['reencode'] = costs.get('reencode', 0.0) + reencode_seconds
    completed = domain if result.complete else ()
    reason = 'no-hit' if result.state == 'unread' else result.state
    return DecodeResult(result.complete, reason, result.frame,
        len(result.identities), completed,
        sum(o.method == 'turbo' for o in result.provenance), len(result.provenance),
        time.monotonic() - started, domain, iterations, result.identities,
        tuple(outcomes), tuple(costs.items()), coordinates, tuple(noise_diagnostics),
        observation)


def decode_fields(fields, speedlevel, iterations=8, *, max_work=None):
    """Decode one mode through the exhaustive shared LONG engine.

    ``max_work`` grants two constituent calls per unit. Whole-domain tiers
    complete together; systematic checks cost no constituent calls. Outcomes
    include systematic, C1-only, and turbo checks, identified by ``method``.
    """
    started = time.monotonic()
    g = geometry(speedlevel)
    iterations, max_work = _iterations(iterations), _budget(max_work)
    values = _real_array(fields, SYMBOLS * g.width)
    component_permutation(speedlevel)  # Validate constants even for zero input.
    return _decode(((g, values),), iterations, max_work, started,
                   coordinates=SourceCoordinates("serialized-llrs"))


def _levels(levels):
    values = tuple(levels)
    modes = tuple(geometry(v) for v in values)
    if not modes or len({g.speedlevel for g in modes}) != len(modes):
        raise ValueError("speedlevels must be a nonempty unique declared sequence")
    return modes


def decode_equalized(symbols, noise, *, speedlevels=LEVELS, iterations=8, max_work=None):
    """Decode a declared mode domain; supplied N0 is complex variance.

    ``max_work`` counts pairs of constituent calls across complete engine tiers,
    as in :func:`decode_fields`; it cannot complete only an early mode.
    """
    started = time.monotonic()
    z, n0 = _symbols(symbols), _noise(noise)
    modes = _levels(speedlevels)
    iterations, max_work = _iterations(iterations), _budget(max_work)
    _tables()
    if not np.any(z):
        return DecodeResult(True, "no-hit", None,
            elapsed_seconds=time.monotonic()-started,
            declared_modes=tuple(g.speedlevel for g in modes),
            declared_iterations=iterations,
            coordinates=SourceCoordinates("equalized-symbols"),
            observation_eligible=False)
    demap_started = time.monotonic()
    members = tuple((g, demap(z, n0, g.speedlevel)) for g in modes)
    demap_seconds = time.monotonic() - demap_started
    noises = tuple(NoiseDiagnostic(g.speedlevel,
        float(np.mean(np.abs(constellation(g.speedlevel))**2)), n0,
        "caller-supplied") for g in modes)
    return _decode(members, iterations, max_work, started, symbols=z,
        coordinates=SourceCoordinates("equalized-symbols"),
        noise_diagnostics=noises, stage_seconds={"demap": demap_seconds})


@dataclass(frozen=True)
class BodyObservation:
    symbols: np.ndarray = field(repr=False)
    gain: complex
    fitted_noise: float
    inverse_energy: float
    residual_energy: float
    coordinates: SourceCoordinates | None = None

    @property
    def noise_diagnostics(self):
        return tuple(NoiseDiagnostic(sl,
            float(np.mean(np.abs(constellation(sl))**2)), float(self.noise(sl)),
            "pilot-residual-isi-proxy") for sl in LEVELS)

    def noise(self, speedlevel):
        energy = float(np.mean(abs(constellation(speedlevel)) ** 2))
        return (self.fitted_noise * self.inverse_energy + energy * self.residual_energy) / abs(self.gain) ** 2


# observe_body below deliberately mirrors the existing SL5 training/inverse fit.
# It preserves complex observations and does not alter p4long.body_scores.

def observe_body(baseband, *, first_training_start: int,
                frequency_offset: float = 0.0) -> BodyObservation | None:
    """Retain the legacy training/inverse fit as complex placed-body symbols."""
    z = np.asarray(baseband, dtype=complex)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("baseband must be a finite one-dimensional vector")
    if type(first_training_start) is not int or first_training_start < 0:
        raise ValueError("first_training_start must be a nonnegative integer")
    if (not np.isscalar(frequency_offset) or not np.isreal(frequency_offset)
            or not np.isfinite(frequency_offset)):
        raise ValueError("frequency_offset must be finite and real")
    start = first_training_start
    if len(z) < start + p4long.LAYOUT.total_symbols * p4rx.SAMPLES_PER_CHIP + 128:
        return None
    peak = np.max(abs(z))
    if peak == 0:
        return None
    z = z / peak * np.exp(-2j * np.pi * frequency_offset * np.arange(len(z)) / 28800)
    matched = p4rx.matched_filter(z)
    return observe_matched_body(matched, first_training_start=start,
                                frequency_offset=frequency_offset,
                                input_samples=len(baseband))


def observe_matched_body(matched, *, first_training_start, frequency_offset,
                         input_samples):
    """Apply the same all-training fit to retained causal matched windows."""
    start = first_training_start
    spacing = 32 + p4long.LAYOUT.block_symbols
    targets = np.arange(10, 22)
    trainings = len(p4long.LAYOUT.training_offsets)
    training = np.array([p4normal.training_sequence(k + 1) for k in range(trainings)])
    positions = start + 128 + 16 * (spacing * np.arange(trainings)[:, None] + targets)
    observed = matched[positions].ravel()
    if not np.any(observed):
        return None
    matrix = training[:, targets[:, None] + np.arange(-2, 3)].reshape(-1, 5)
    gram = matrix.conj().T @ matrix
    channel = np.linalg.solve(gram + 1e-6 * np.trace(gram).real / 5 * np.eye(5),
                              matrix.conj().T @ observed)
    variance = float(np.mean(abs(observed - matrix @ channel)**2))
    convolution = np.column_stack([np.convolve(channel, row) for row in np.eye(11)])
    target = np.zeros(15)
    target[7] = 1
    if variance == 0:
        inverse = np.linalg.lstsq(convolution, target, rcond=None)[0]
    else:
        inverse = np.linalg.solve(
            convolution.conj().T @ convolution + variance * np.eye(11),
            convolution.conj().T @ target)
    residual = np.convolve(channel, inverse)
    gain = residual[7]
    residual[7] = 0
    if not np.isfinite(gain) or abs(gain) == 0:
        return None
    body = (spacing * np.arange(p4long.LAYOUT.blocks)[:, None] + 32
            + np.arange(p4long.LAYOUT.block_symbols)).ravel()
    windows = start + 128 + 16 * (body[:, None] + np.arange(-5, 6))
    y = matched[windows] @ inverse
    symbols = np.frombuffer((y / gain).tobytes(), dtype=complex)
    return BodyObservation(symbols, complex(gain), variance,
                           float(np.vdot(inverse, inverse).real),
                           float(np.vdot(residual, residual).real),
                           SourceCoordinates("placed-baseband", int(first_training_start),
                                             float(frequency_offset), input_samples))


def decode_body(baseband, *, first_training_start, frequency_offset=0.0,
                speedlevels=LEVELS, iterations=8, max_work=None):
    """Placed-body diagnostic decode with shared-engine work accounting.

    Pilot/ISI variance is a diagnostic proxy. ``max_work`` has the complete-tier
    semantics documented by :func:`decode_fields`.
    """
    started = time.monotonic()
    modes = _levels(speedlevels)
    iterations, max_work = _iterations(iterations), _budget(max_work)
    _tables()
    observation_started = time.monotonic()
    observed = observe_body(baseband, first_training_start=first_training_start,
                            frequency_offset=frequency_offset)
    observation_seconds = time.monotonic() - observation_started
    coordinates = SourceCoordinates("placed-baseband", int(first_training_start),
                                    float(frequency_offset), len(baseband))
    domain = tuple(g.speedlevel for g in modes)
    def empty(complete, reason, noises=()):
        return DecodeResult(complete, reason, None,
            elapsed_seconds=time.monotonic()-started, declared_modes=domain,
            declared_iterations=iterations, coordinates=coordinates,
            observation=observed, noise_diagnostics=tuple(noises),
            stage_seconds=(("observe", observation_seconds),),
            observation_eligible=False)
    if observed is None:
        needed = first_training_start + p4long.LAYOUT.total_symbols * p4rx.SAMPLES_PER_CHIP + 128
        return empty(False, "incomplete-input") if len(baseband) < needed else empty(True, "no-observation")
    members, noises = [], []
    demap_started = time.monotonic()
    for g in modes:
        n0 = observed.noise(g.speedlevel)
        noises.append(NoiseDiagnostic(g.speedlevel,
            float(np.mean(np.abs(constellation(g.speedlevel))**2)), float(n0),
            "pilot-residual-isi-proxy"))
        if not np.isfinite(n0) or n0 <= 0:
            return empty(True, "invalid-noise", noises)
        members.append((g, demap(observed.symbols, n0, g.speedlevel)))
    demap_seconds = time.monotonic() - demap_started
    return _decode(members, iterations, max_work, started,
        symbols=observed.symbols, coordinates=coordinates,
        noise_diagnostics=noises, observation=observed,
        stage_seconds={"observe": observation_seconds, "demap": demap_seconds})
