# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Immutable receive coordinates on the transport input clock."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CaptureContext:
    tune_epoch: int
    decoder_epoch: int
    sample_start: int
    sample_rate: int
    signal_delay: int = 0

    def shifted(self, samples: int):
        return replace(self, sample_start=self.sample_start + samples)

    def origin(self, training: int, end: int, level: int, bandwidth_hz: int):
        base = self.sample_start - self.signal_delay
        return FrameOrigin(self.tune_epoch, self.decoder_epoch,
                           base + training, base + end, level, bandwidth_hz)


@dataclass(frozen=True)
class FrameOrigin:
    tune_epoch: int
    decoder_epoch: int
    training_origin: int
    end_sample: int
    level: int
    bandwidth_hz: int


@dataclass(frozen=True)
class LocatedFrame:
    ordinal: int
    origin: FrameOrigin
    body: bytes | None
