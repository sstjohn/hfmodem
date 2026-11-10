# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Payload-free UI exclusion geometry on the transport input clock."""
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite

from .broadcast import BODY_SAMPLES, PERIOD, deadline

TOLERANCE = {1: 16384, 2: 8192, 4: 2048}
MAX_HORIZON = deadline(0, 1, 18)


class Classification(str, Enum):
    CLEAR = 'CLEAR'
    CONTESTED = 'CONTESTED'
    STALE = 'STALE'


CLEAR = Classification.CLEAR
CONTESTED = Classification.CONTESTED
STALE = Classification.STALE


@dataclass(frozen=True)
class UISpan:
    id: int
    tune_epoch: int
    level: int
    header_origin: int
    possible_packet_count: int = 18
    ambiguous_2_or_18: bool = False
    resolution: str | None = None
    validated_end: int | None = None
    sample_rate: float = 48000
    first_training_origin: int = field(init=False)
    period: int = field(init=False)
    tolerance: int = field(init=False)
    classification_end: int = field(init=False)
    deadline: int = field(init=False)

    def __post_init__(self):
        if self.level not in PERIOD or not 1 <= self.possible_packet_count <= 18:
            raise ValueError('invalid UI span geometry')
        if not isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError('invalid UI span sample rate')
        def scale(n):
            return round(n * self.sample_rate / 48000)
        first = self.header_origin + scale(67072)
        end = self.header_origin + scale(67072 + (self.possible_packet_count - 1)
                                         * PERIOD[self.level] + BODY_SAMPLES[self.level])
        object.__setattr__(self, 'first_training_origin', first)
        object.__setattr__(self, 'period', scale(PERIOD[self.level]))
        object.__setattr__(self, 'tolerance', scale(TOLERANCE[self.level]))
        object.__setattr__(self, 'classification_end',
                           end if self.validated_end is None else self.validated_end)
        object.__setattr__(self, 'deadline', self.header_origin + scale(
            deadline(0, self.level, self.possible_packet_count)))


@dataclass(frozen=True)
class SpanPublished:
    span: UISpan


@dataclass(frozen=True)
class SpanResolved:
    span_id: int
    tune_epoch: int
    outcome: str
    validated_end: int | None = None
    count: int | None = None


class UISpanLedger:
    """Serial owner-thread ledger; origins below result_floor classify STALE.

    A route may retain results for at most route_retention input samples. The
    enforced floor makes a result older than that bound unusable even if its
    route has not advanced. No exclusion is evicted to satisfy a candidate cap.
    """
    def __init__(self, tune_epoch=0, *, route_retention=MAX_HORIZON):
        if route_retention < 0:
            raise ValueError('negative route retention')
        self.route_retention = route_retention
        self.retune(tune_epoch)

    @property
    def spans(self):
        return tuple(self._spans.values())

    @property
    def result_floor(self):
        return self._floor

    def publish(self, span):
        if span.tune_epoch != self.tune_epoch:
            return
        if span.id in self._retired:
            return
        if span.classification_end < self._floor:
            return
        previous = self._spans.get(span.id)
        if previous is not None:
            if (previous.level, previous.header_origin) != (span.level, span.header_origin):
                raise ValueError('span refinement changed its header')
            if previous.resolution in ('group', 'beacon', 'singleton'):
                return
        duplicate = next((s for s in self._spans.values()
                          if s.id != span.id and (s.level, s.header_origin)
                          == (span.level, span.header_origin)), None)
        if duplicate is not None:
            return
        self._spans[span.id] = span

    def resolve(self, span_id, outcome, *, validated_end=None, count=None,
                tune_epoch=None):
        if tune_epoch is not None and tune_epoch != self.tune_epoch:
            return
        span = self._spans.get(span_id)
        if span is None or span.resolution in ('group', 'beacon', 'singleton'):
            return
        if outcome in ('group', 'beacon', 'singleton'):
            if validated_end is None or count is None or not 1 <= count <= 18:
                raise ValueError('validated resolution needs body end and count')
            if outcome in ('beacon', 'singleton') and count != 1:
                raise ValueError('single body resolution needs count one')
            ambiguous = outcome == 'group' and count == 2 and span.ambiguous_2_or_18
            self._spans[span_id] = replace(span, resolution=outcome,
                possible_packet_count=18 if ambiguous else count,
                validated_end=None if ambiguous else validated_end)
        elif outcome in ('failed', 'truncated'):
            self._spans[span_id] = replace(span, resolution=outcome)
        else:
            raise ValueError('unknown UI span resolution')

    def apply(self, event):
        if isinstance(event, SpanPublished):
            self.publish(event.span)
        elif isinstance(event, SpanResolved):
            self.resolve(event.span_id, event.outcome, validated_end=event.validated_end,
                         count=event.count, tune_epoch=event.tune_epoch)
        else:
            raise TypeError('unknown UI span event')

    def classify(self, frame_origin):
        if frame_origin.tune_epoch != self.tune_epoch:
            return CLEAR
        origin = frame_origin.training_origin
        if origin < self._floor:
            return STALE
        if frame_origin.bandwidth_hz != 500 or frame_origin.level not in PERIOD:
            return CLEAR
        for span in self._spans.values():
            if span.level != frame_origin.level or origin >= span.classification_end:
                continue
            relative = origin - span.first_training_origin
            slot = min(span.possible_packet_count - 1, max(0, round(relative / span.period)))
            if abs(relative - slot * span.period) <= span.tolerance:
                return CONTESTED
        return CLEAR

    def set_frontier(self, route, sample):
        self._frontiers[route] = max(sample, self._frontiers.get(route, sample))
        self._floor = max(self._floor, min(self._frontiers.values()))

    def advance(self, input_clock, detector_watermark):
        self._input_clock = max(self._input_clock, input_clock)
        floor = min(input_clock, detector_watermark,
                    min(self._frontiers.values(), default=input_clock))
        self._floor = max(self._floor, floor, input_clock - self.route_retention)
        passed = min(self._floor, input_clock)
        for i, span in tuple(self._spans.items()):
            if span.classification_end < passed:
                self._retired[i] = span.header_origin + round(MAX_HORIZON * span.sample_rate / 48000)
                del self._spans[i]
        self._retired = {i: end for i, end in self._retired.items() if end >= self._floor}

    def retune(self, new_epoch):
        self.tune_epoch = new_epoch
        self._spans = {}
        self._frontiers = {}
        self._retired = {}
        self._floor = float('-inf')
        self._input_clock = 0
