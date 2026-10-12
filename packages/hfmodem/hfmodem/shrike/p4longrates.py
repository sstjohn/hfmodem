"""Standalone, placed normal-LONG SL6--10 diagnostics. No live receiver dispatch.

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

from . import p4c2, tablegen, p4coding, p4decode, p4frame, p4long, p4normal, p4rx

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


def demap(symbols, noise, speedlevel):
    """Return field-order, MSB label LLRs; positive values mean bit zero."""
    g = geometry(speedlevel)
    z, n0 = _symbols(symbols), _noise(noise)
    points = constellation(speedlevel)
    metrics = -(np.abs(z[:, None] - points) ** 2) / n0
    if not np.all(np.isfinite(metrics)):
        raise ValueError("demapper metric overflow")
    labels = np.arange(1 << g.width)
    llr = np.column_stack([
        logsumexp(metrics[:, ((labels >> bit) & 1) == 0], axis=1)
        - logsumexp(metrics[:, ((labels >> bit) & 1) == 1], axis=1)
        for bit in range(g.width - 1, -1, -1)
    ])
    return llr[symbol_permutation()].ravel()


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


@dataclass(frozen=True)
class _Iteration:
    bits: np.ndarray = field(repr=False)
    depuncture_seconds: float
    constituent_seconds: tuple


def _check_frame(bits, g):
    net = 8 * g.net_bytes
    padding_ok = not bool(np.any(bits[net:net + g.padding]))
    wire = np.packbits(bits[:net], bitorder="little").tobytes()
    decoded = p4frame.decode_field(wire, data_bytes=g.data_bytes)
    encoded = p4coding.rsc_encode(bits[:-3], initial_state=(0, 0, 0))
    termination = p4coding.rsc_terminate(encoded.final_state)
    flush_ok = bool(np.array_equal(bits[-3:], termination.systematic))
    return decoded, padding_ok, flush_ok


def _frame(bits, g):
    decoded, padding_ok, flush_ok = _check_frame(bits, g)
    return decoded if padding_ok and flush_ok else None


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


def _iterations_for(fields, g, iterations):
    started = time.monotonic()
    n = g.systematic
    systematic = fields[:n]
    p1, p2 = np.zeros(n), np.zeros(n)
    p1[::10] = fields[n:n + g.parity_each]
    p2[4::10] = fields[n + g.parity_each:]
    order = component_permutation(g.speedlevel)
    prior = np.zeros(n)
    depuncture = time.monotonic() - started
    for _ in range(iterations):
        started = time.monotonic()
        first = p4decode.component_decode(systematic, p1, prior,
                                         end_state=0, initial_state=0)
        first_seconds = time.monotonic() - started
        started = time.monotonic()
        second = p4decode.component_decode(systematic[order], p2,
                                          first.extrinsic[order], initial_state=0)
        second_seconds = time.monotonic() - started
        prior[order] = second.extrinsic
        bits = np.empty(n, np.uint8)
        bits[order] = second.posterior < 0
        yield _Iteration(bits, depuncture, (first_seconds, second_seconds))
        depuncture = 0.0


def _decode(members, iterations, max_work, started, *, declared_modes=None,
            symbols=None, coordinates=None, noise_diagnostics=(),
            observation=None, stage_seconds=None):
    found, completed, outcomes = {}, [], []
    costs = {name: 0.0 for name in ("observe", "demap", "depuncture",
             "constituent-c1", "constituent-c2", "crc", "reencode")}
    if stage_seconds:
        costs.update(stage_seconds)
    # Every declared member is retained even if its LLR vector is all zero.
    members = tuple(members)
    domain = tuple(g.speedlevel for g, _ in members) if declared_modes is None else tuple(declared_modes)
    work = checks = 0
    def result(complete, reason):
        frames = tuple(found.values())
        return DecodeResult(complete, reason,
            frames[0] if complete and len(frames) == 1 else None,
            len(frames), tuple(completed), work, checks, time.monotonic()-started,
            domain, iterations, frames, tuple(outcomes), tuple(costs.items()),
            coordinates, tuple(noise_diagnostics), observation)
    for g, fields in members:
        stream = _iterations_for(fields, g, iterations)
        for iteration in range(1, iterations + 1):
            if max_work is not None and work >= max_work:
                return result(False, "incomplete")
            step = next(stream)
            if isinstance(step, _Iteration):
                bits = step.bits
                costs["depuncture"] += step.depuncture_seconds
                costs["constituent-c1"] += step.constituent_seconds[0]
                costs["constituent-c2"] += step.constituent_seconds[1]
            else:
                bits = step  # controlled candidate-stream substitution in tests
            work += 1
            checked = time.monotonic()
            frame, padding_ok, flush_ok = _check_frame(bits, g)
            costs["crc"] += time.monotonic() - checked
            checks += 1
            identity_index = None
            diagnostic = None
            if frame is not None:
                encoded = time.monotonic()
                diagnostic = _reencoding(bits, g, fields, symbols)
                costs["reencode"] += time.monotonic() - encoded
                if padding_ok and flush_ok:
                    key = (frame.data, frame.status.byte, frame.crc)
                    if key not in found:
                        found[key] = frame
                    identity_index = tuple(found).index(key)
            outcomes.append(IterationOutcome(g.speedlevel, iteration,
                frame is not None, padding_ok, flush_ok, identity_index, diagnostic))
        completed.append(g.speedlevel)
    reason = "unique" if len(found) == 1 else "ambiguous" if found else "no-hit"
    return result(True, reason)


def decode_fields(fields, speedlevel, iterations=8, *, max_work=None):
    """Decode one declared mode, retaining every iteration's CRC identity."""
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
    """Decode a complete declared mode domain; supplied N0 is complex variance."""
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
    spacing = 32 + p4long.LAYOUT.block_symbols
    if len(z) < start + p4long.LAYOUT.total_symbols * p4rx.SAMPLES_PER_CHIP + 128:
        return None
    peak = np.max(abs(z))
    if peak == 0:
        return None
    z = z / peak * np.exp(-2j * np.pi * frequency_offset * np.arange(len(z)) / 28800)
    matched = p4rx.matched_filter(z)
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
                                             float(frequency_offset), len(baseband)))


def decode_body(baseband, *, first_training_start, frequency_offset=0.0,
                speedlevels=LEVELS, iterations=8, max_work=None):
    """Standalone placed-body decode. Pilot/ISI variance is a diagnostic proxy."""
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
