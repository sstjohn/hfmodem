# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Offline Robust Packet bursts at 9600 Hz, without scheduled PTT lead."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging

import numpy as np

from hfmodem.core.levels import TX_DRIVE, at_drive
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.resample import for_card

from . import frame, waveform


def phase_steps(packet: frame.Packet, level: int) -> np.ndarray:
    prefix = np.unpackbits(np.frombuffer(bytes([frame.REFERENCE]) + frame.PREAMBLE, np.uint8),
                           bitorder="little").reshape(-1, 8)
    header = frame.header_bits(4 * packet.cls + level)
    if packet.cls == 15:
        header = np.concatenate((header, frame.header_bits(packet.ext)))
    encoded = frame.encode_packet(packet.data, level)
    prefix_steps = np.array([-2, -10])[np.concatenate((prefix, header))]
    if level == 0:
        data_steps = np.array([-2, -10])[encoded.reshape(-1, 8)]
    else:
        pairs = encoded.reshape(-1, 8, 2)
        data_steps = np.array([-2, -6, -14, -10])[pairs[..., 0] + 2 * pairs[..., 1]]
    return np.concatenate((prefix_steps, data_steps))


def render(packet: frame.Packet, level: int = 0, *, limiter: bool = True) -> np.ndarray:
    base = waveform.composite(phase_steps(packet, level), level, limiter)
    return np.concatenate((waveform.audio(base), np.zeros(waveform.SPS)))


@dataclass(frozen=True)
class Burst:
    packet: frame.Packet
    level: int
    limiter: bool
    audio: np.ndarray

    def card_audio(self, drive: float = TX_DRIVE) -> np.ndarray:
        return at_drive(for_card(self.audio, waveform.RATE), drive)

    def lead_s(self, txdelay: float = 100, settle_s: float = 0.04) -> float:
        """Scheduled lead from total TXDELAY in milliseconds and converted onset."""
        if txdelay < 0 or settle_s < 0:
            raise ValueError("negative transmit delay")
        return max(settle_s, txdelay / 1000 - onset_s(self.limiter))

    def keyed_s(self, txdelay: float = 100, settle_s: float = 0.04) -> float:
        return self.lead_s(txdelay, settle_s) + len(self.audio) / waveform.RATE + 0.1


@lru_cache(maxsize=2)
def onset_s(limiter: bool) -> float:
    packet = frame.Packet(bytes(15), 0, 0)
    converted = at_drive(for_card(render(packet, limiter=limiter), waveform.RATE))
    # Ignore filter residue below the first nonzero PCM16 sample at default drive.
    return int(np.flatnonzero(np.abs(converted) >= 0.5 / 32767)[0]) / CARD_RATE_HZ


def transmit(frames: list[bytes], level: int = 0, *, suggestion: int = 0,
             limiter: bool = True, max_key_s: float | None = None,
             txdelay: float = 100, settle_s: float = 0.04) -> list[Burst]:
    if level not in (0, 1) or not np.isfinite(txdelay) or txdelay < 0 or not np.isfinite(settle_s) or settle_s < 0:
        raise ValueError("invalid level or transmit delay")
    max_capacity = 825
    if max_key_s is not None:
        if not np.isfinite(max_key_s) or max_key_s <= 0:
            raise ValueError("invalid maximum keyed interval")
        lead = max(settle_s, txdelay / 1000 - onset_s(limiter))
        capacities = []
        for cls, ext in [(c, 0) for c in range(15)] + [(15, e) for e in range(17)]:
            duration = (frame.symbol_count(cls, ext, level) + waveform.RAMP_SYMBOLS + 1) * waveform.SPS / waveform.RATE
            if lead + duration + 0.1 <= max_key_s:
                capacities.append(frame.packet_bytes(cls, ext))
        max_capacity = max(capacities, default=0)
    try:
        packets = frame.pack_frames(frames, suggestion, max_capacity=max_capacity)
    except ValueError as exc:
        logging.getLogger(__name__).warning("cannot pack transmit frames: %s", exc)
        raise
    result = []
    for packet in packets:
        burst = Burst(packet, level, limiter, render(packet, level, limiter=limiter))
        if max_key_s is not None and burst.keyed_s(txdelay, settle_s) > max_key_s:
            raise ValueError("packet's complete keyed interval exceeds max_key_s")
        result.append(burst)
    return result


def main() -> None:
    import argparse
    import wave

    from hfmodem.ax25.frame import ui_frame

    parser = argparse.ArgumentParser(description="Render a Robust Packet UI frame offline")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("text")
    parser.add_argument("output")
    parser.add_argument("--level", type=int, choices=(0, 1), default=0)
    parser.add_argument("--no-limiter", action="store_true")
    args = parser.parse_args()
    burst, = transmit([ui_frame(args.source, args.destination, args.text.encode())],
                      args.level, limiter=not args.no_limiter)
    with open(args.output, "xb") as output:
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(waveform.RATE)
            writer.writeframes(np.round(at_drive(burst.audio) * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    main()
