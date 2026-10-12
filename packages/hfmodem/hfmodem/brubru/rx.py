# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Offline and incremental Robust Packet reception with retained burst history."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert, resample_poly

from . import frame, waveform as w


@dataclass(frozen=True)
class Detection:
    ref: int
    centre: float
    inverted: bool
    score: complex


def acquire(samples: np.ndarray, centre: float = w.CENTRE) -> Detection | None:
    if len(samples) < 12 * w.SPS:
        return None
    best = None
    strength = 0.55
    for inverted in (False, True):
        for offset in (-12.5, 0.0, 12.5):
            frequency = centre + offset
            scores = w.preamble_correlation(w.matched(samples, frequency, inverted))
            ref = int(np.argmax(np.abs(scores)))
            if abs(scores[ref]) > strength:
                strength = abs(scores[ref])
                best = Detection(ref, frequency, inverted, scores[ref])
    if best is None:
        return None
    for _ in range(3):
        correction = np.angle(best.score) * 50 / (2 * np.pi)
        frequency = best.centre + (-correction if best.inverted else correction)
        scores = w.preamble_correlation(w.matched(samples, frequency, best.inverted))
        lo = max(0, best.ref - w.SPS // 2)
        hi = min(len(scores), best.ref + w.SPS // 2 + 1)
        ref = lo + int(np.argmax(np.abs(scores[lo:hi])))
        best = Detection(ref, frequency, best.inverted, scores[ref])
    return best


def soft_values(filtered: np.ndarray, ref: int, start: int, count: int,
                level: int = 0) -> np.ndarray:
    positions = ref + w.SPS * np.arange(start, start + count + 1)
    if positions[0] < 0 or positions[-1] >= filtered.shape[1]:
        raise ValueError("truncated burst")
    points = filtered[:, positions]
    steps = points[:, 1:] * points[:, :-1].conj()
    steps /= np.maximum(np.abs(steps), 1e-30)
    if level == 0:
        return (steps.T * w.ROTATION).real.ravel()
    return np.stack((steps.T.real, -steps.T.imag), axis=-1).ravel()


def _points(filtered, positions):
    low = np.floor(positions).astype(int)
    if low.min() < 0 or low.max() + 1 >= filtered.shape[1]:
        raise ValueError("truncated tracked burst")
    fraction = positions - low
    return filtered[:, low] * (1 - fraction) + filtered[:, low + 1] * fraction


def tracked_values(samples, detection, header, filtered):
    """Estimate timing and differential phase every 32 symbols on long frames.

    Decisions remove the constellation rotations before estimating common phase.
    Timing combines matched-symbol energy with an early/late decision residual;
    low-confidence sections retain the previous grid so a fade cannot steer it.
    """
    start = 24 if header.cls == 15 else 16
    count = header.symbols - start - 1
    indices = np.arange(start, start + count + 1)
    grid = detection.ref + w.SPS * indices.astype(float)
    positions = grid.copy()
    centres, phases = [], []
    offset = 0.0
    previous_phase = 0.0
    order = 2 if header.level == 0 else 4
    for lo in range(0, count, 32):
        hi = min(count, lo + 32)
        section = grid[lo:hi + 1]
        offsets = offset + np.array([-12.0, 0.0, 12.0])
        candidates = [_points(filtered, section + shift) for shift in offsets]
        powers = [float(np.mean(np.abs(candidate) ** 2)) for candidate in candidates]
        curvature = powers[0] - 2 * powers[1] + powers[2]
        if curvature < -1e-20:
            step = np.clip(6 * (powers[0] - powers[2]) / curvature, -6, 6)
        else:
            step = 0.0
        points = candidates[1]
        differences = points[:, 1:] * points[:, :-1].conj()
        differences /= np.maximum(np.abs(differences), 1e-30)
        decisions = np.mean((differences * w.ROTATION) ** order)
        phase = np.angle(decisions) / order
        phase += round((previous_phase - phase) * order / (2 * np.pi)) * 2 * np.pi / order
        if abs(decisions) >= 0.15:
            hard_phase = (np.rint((np.angle(differences) + np.pi / 4 - phase)
                                  * order / (2 * np.pi)) * (2 * np.pi / order)
                          - np.pi / 4 + phase)
            target = (points[:, :-1] / np.maximum(np.abs(points[:, :-1]), 1e-30)
                      * np.exp(1j * hard_phase) * np.abs(points[:, 1:]))
            derivative = (candidates[2][:, 1:] - candidates[0][:, 1:]) / 24
            residual = points[:, 1:] - target
            timing_error = -float(np.mean((residual * derivative.conj()).real)) / max(
                float(np.mean(np.abs(derivative) ** 2)), 1e-30)
            # The decision residual has a narrower loop bandwidth than the
            # energy estimate because individual differential decisions are noisy.
            offset += float(step) + 0.05 * float(np.clip(timing_error, -6, 6))
            previous_phase = phase
        positions[lo:hi + 1] = section + offset
        centres.append(float(np.mean(section)))
        phases.append(previous_phase)
    # Correct the changing carrier before filtering again, avoiding filter loss
    # at the end of a long burst whose carrier has moved across the passband.
    frequency = np.interp(np.arange(len(samples)), centres,
                          np.asarray(phases) * 50 / (2 * np.pi))
    time = np.arange(len(samples)) / w.RATE
    base = hilbert(samples) * np.exp(-2j * np.pi * detection.centre * time)
    if detection.inverted:
        base = base.conj()
    base *= np.exp(-2j * np.pi * np.cumsum(frequency) / w.RATE)
    corrected = w.matched_base(base)
    points = _points(corrected, positions)
    steps = points[:, 1:] * points[:, :-1].conj()
    steps /= np.maximum(np.abs(steps), 1e-30)
    if header.level == 0:
        return (steps.T * w.ROTATION).real.ravel()
    return np.stack((steps.T.real, -steps.T.imag), axis=-1).ravel()


@dataclass(frozen=True)
class Header:
    cls: int
    ext: int
    level: int
    correlation: float

    @property
    def symbols(self) -> int:
        return frame.symbol_count(self.cls, self.ext, self.level)


def read_header(filtered: np.ndarray, ref: int) -> Header:
    index, correlation = frame.decode_header(soft_values(filtered, ref, 8, 8))
    cls, level = divmod(index, 4)
    if level > 1:
        raise ValueError("unsupported header level")
    ext = 0
    if cls == 15:
        ext, quality = frame.decode_header(soft_values(filtered, ref, 16, 8))
        correlation = min(quality, correlation)
    frame.packet_bytes(cls, ext)
    return Header(cls, ext, level, correlation)


@dataclass(frozen=True)
class Decoded:
    ref: int
    centre: float
    inverted: bool
    header: Header
    packet: bytes
    suggestion: int
    frames: tuple[bytes, ...]
    agreement: float
    differentials: np.ndarray


def decode(samples: np.ndarray, detection: Detection, origin: int = 0, *, tracking=True) -> Decoded:
    filtered = w.matched(samples, detection.centre, detection.inverted)
    header = read_header(filtered, detection.ref)
    start = 24 if header.cls == 15 else 16
    count = header.symbols - start - 1
    soft = (tracked_values(samples, detection, header, filtered)
            if tracking and header.symbols >= 300 else
            soft_values(filtered, detection.ref, start, count, header.level))
    packet, agreement = frame.decode_packet(soft, frame.packet_bytes(header.cls, header.ext),
                                            header.level)
    suggestion, frames = frame.parse(packet)
    diffs = soft_values(filtered, detection.ref, 0, header.symbols - 1)
    return Decoded(origin + detection.ref, detection.centre, detection.inverted,
                   header, packet, suggestion, tuple(frames), agreement, diffs)


class Receiver:
    WINDOW = 48 * w.SPS
    STEP = 16 * w.SPS
    MAX_SAMPLES = (frame.symbol_count(15, frame.MAX_EXT, 0) + 64) * w.SPS

    def __init__(self, centre: float = w.CENTRE):
        self.centre = centre
        self.reset()

    def reset(self, start: int = 0) -> None:
        self._samples = np.empty(0)
        self._pending = np.empty(0)
        self._origin = start
        self._held: tuple[Detection, Header] | None = None
        self.events: deque[tuple[int, str]] = deque(maxlen=64)
        self.preambles: deque[int] = deque(maxlen=64)

    @property
    def held(self) -> bool:
        return self._held is not None

    @property
    def buffered(self) -> int:
        return len(self._samples) + len(self._pending)

    def _discard(self, count: int) -> None:
        self._samples = self._samples[count:]
        self._origin += count

    def _process(self, final: bool = False) -> list[Decoded]:
        result = []
        while True:
            if self._held is None:
                if len(self._samples) < (28 * w.SPS if final else self.WINDOW):
                    break
                window = self._samples[:self.WINDOW]
                found = acquire(window, self.centre)
                if found is None:
                    self._discard(min(self.STEP, len(self._samples)))
                    continue
                self.preambles.append(self._origin + found.ref)
                if found.ref + 27 * w.SPS >= len(window) and not final:
                    self._discard(max(0, found.ref - 3 * w.SPS))
                    break
                try:
                    header = read_header(w.matched(window, found.centre, found.inverted), found.ref)
                except ValueError as exc:
                    self.events.append((self._origin + found.ref, str(exc)))
                    self._discard(max(w.SPS, found.ref + w.SPS))
                    continue
                self._held = found, header
            found, header = self._held
            end = found.ref + (header.symbols + 3) * w.SPS
            if len(self._samples) < end and not final:
                break
            try:
                if len(self._samples) <= found.ref + (header.symbols - 1) * w.SPS:
                    raise ValueError("truncated burst")
                result.append(decode(self._samples[:end], found, self._origin))
            except ValueError as exc:
                self.events.append((self._origin + found.ref, str(exc)))
            self._held = None
            self._discard(min(len(self._samples), found.ref + header.symbols * w.SPS))
        return result

    def feed(self, samples: np.ndarray) -> list[Decoded]:
        samples = np.asarray(samples, float)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("finite mono audio required")
        result = []
        offset = 0
        while offset < len(samples):
            count = min(self.STEP - len(self._pending), len(samples) - offset)
            self._pending = np.concatenate((self._pending, samples[offset:offset + count]))
            offset += count
            if len(self._pending) == self.STEP:
                self._samples = np.concatenate((self._samples, self._pending))
                self._pending = np.empty(0)
                result.extend(self._process())
                if len(self._samples) > self.MAX_SAMPLES:
                    self.events.append((self._origin, "hold exceeded maximum frame"))
                    self._held = None
                    self._discard(len(self._samples) - self.WINDOW)
        return result

    def finish(self) -> list[Decoded]:
        self._samples = np.concatenate((self._samples, self._pending))
        self._pending = np.empty(0)
        result = self._process(final=True)
        self._origin += len(self._samples)
        self._samples = np.empty(0)
        self._held = None
        return result


def receive(samples: np.ndarray, rate: int = w.RATE, *, centre: float | None = None) -> list[Decoded]:
    samples = np.asarray(samples, float)
    if rate != w.RATE:
        samples = resample_poly(samples, w.RATE, rate)
    if len(samples) < 12 * w.SPS:
        return []
    receiver = Receiver(w.band_centre(samples) if centre is None else centre)
    return receiver.feed(samples) + receiver.finish()


def main() -> None:
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Decode Robust Packet audio files offline")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for path in args.files:
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                                  "-ac", "1", "-ar", str(w.RATE), "-"],
                                 capture_output=True, check=True)
        for burst in receive(np.frombuffer(decoded.stdout, "<f4")):
            for raw in burst.frames:
                print(f"{path} sample={burst.ref} level={burst.header.level} frame={raw.hex()}")


if __name__ == "__main__":
    main()
