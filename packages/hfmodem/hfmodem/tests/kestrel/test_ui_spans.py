# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Exclusion lifetime is independent of packet assembly and decoder resets."""
from dataclasses import replace
from math import ceil

import pytest

from hfmodem.kestrel.rx import broadcast as UI
from hfmodem.kestrel.rx.observation import FrameOrigin
from hfmodem.kestrel.rx.ui_spans import (
    CLEAR, CONTESTED, STALE, MAX_HORIZON, TOLERANCE, SpanPublished,
    SpanResolved, UISpan, UISpanLedger,
)


def observation(span, slot=0, delta=0, **changes):
    start = span.first_training_origin + slot * span.period + delta
    return replace(FrameOrigin(span.tune_epoch, 9, start,
                               start + UI.BODY_SAMPLES[span.level], span.level, 500),
                   **changes)


@pytest.mark.parametrize('level', [1, 2, 4])
@pytest.mark.parametrize('slot', [0, 1, 17])
@pytest.mark.parametrize('side', [-1, 1])
def test_exact_slot_and_tolerance_edges(level, slot, side):
    span = UISpan(1, 7, level, 1000000)
    ledger = UISpanLedger(7)
    ledger.publish(span)
    tolerance = TOLERANCE[level]
    for distance in (0, tolerance - 1, tolerance):
        assert ledger.classify(observation(span, slot, side * distance)) == CONTESTED
    for distance in (tolerance + 1, tolerance + 2):
        assert ledger.classify(observation(span, slot, side * distance)) == CLEAR
    assert ledger.classify(observation(span, -1)) == CLEAR
    assert ledger.classify(observation(span, 18)) == CLEAR


@pytest.mark.parametrize('changes', [
    {'bandwidth_hz': 2300}, {'bandwidth_hz': 2750}, {'level': 3},
    {'level': 5}, {'level': 6}, {'tune_epoch': 8},
])
def test_only_actual_narrow_geometry_in_the_same_tune_is_eligible(changes):
    span = UISpan(1, 7, 2, 1000000)
    ledger = UISpanLedger(7)
    ledger.publish(span)
    assert ledger.classify(observation(span, **changes)) == CLEAR
    assert ledger.classify(observation(span, decoder_epoch=100)) == CONTESTED


@pytest.mark.parametrize('level', [1, 2, 4])
def test_off_grid_data_is_clear_inside_the_horizon(level):
    span = UISpan(1, 0, level, 0)
    ledger = UISpanLedger()
    ledger.publish(span)
    assert ledger.classify(observation(span, 8, span.period // 2)) == CLEAR


@pytest.mark.parametrize('level', [1, 2, 4])
@pytest.mark.parametrize('meta,count', [(None, 18), (0x60, 2), (0x50, 18),
                                      (0x00, 2), (0x10, 18)]
                         + [(0xA0 | n, n + 2) for n in range(1, 16)])
def test_metadata_counts_and_exact_deadlines(level, meta, count):
    span = UISpan(1, 0, level, 1234, possible_packet_count=(
        18 if meta is None else UI.packet_count(meta)))
    assert span.possible_packet_count == count
    assert span.first_training_origin == 1234 + 67072
    assert span.period == UI.PERIOD[level]
    assert span.classification_end == 1234 + 67072 + (count - 1) * span.period + span.period - 1536
    assert span.deadline == UI.deadline(1234, level, count)


@pytest.mark.parametrize('outcome', ['failed', 'truncated'])
def test_failed_assembly_or_a_final_looking_packet_cannot_acquit(outcome):
    span = UISpan(1, 0, 4, 0)
    ledger = UISpanLedger()
    ledger.publish(span)
    ledger.resolve(span.id, outcome, count=1, validated_end=span.first_training_origin)
    assert ledger.spans[0].possible_packet_count == 18
    assert ledger.classify(observation(span, 17)) == CONTESTED


@pytest.mark.parametrize('meta,remaining', [(0x60, CLEAR), (0x50, CONTESTED)])
def test_valid_two_packet_group_distinguishes_0x60_from_0x50(meta, remaining):
    span = UISpan(1, 0, 1, 0, UI.packet_count(meta), bool(meta & 16))
    ledger = UISpanLedger()
    ledger.publish(span)
    end = span.first_training_origin + span.period + UI.BODY_SAMPLES[1]
    ledger.resolve(span.id, 'group', count=2, validated_end=end)
    assert ledger.classify(observation(span, 0)) == CONTESTED
    assert ledger.classify(observation(span, 1)) == CONTESTED
    assert ledger.classify(observation(span, 2)) == remaining
    assert ledger.classify(observation(span, 17)) == remaining
    assert ledger.spans[0].classification_end == (span.classification_end if meta & 16 else end)


@pytest.mark.parametrize('outcome,count', [('group', 3), ('beacon', 1), ('singleton', 1)])
def test_validation_closes_future_slots_but_keeps_history(outcome, count):
    span = UISpan(1, 0, 2, 0)
    ledger = UISpanLedger()
    ledger.set_frontier('slow', 0)
    ledger.publish(span)
    end = span.first_training_origin + (count - 1) * span.period + UI.BODY_SAMPLES[2]
    ledger.resolve(span.id, outcome, count=count, validated_end=end)
    ledger.advance(span.deadline + 1, span.deadline + 1)
    assert ledger.classify(observation(span, count - 1)) == CONTESTED
    assert ledger.classify(observation(span, count)) == CLEAR
    assert ledger.spans[0].classification_end == end
    ledger.set_frontier('slow', end)
    ledger.advance(span.deadline + 1, span.deadline + 1)
    assert ledger.spans
    ledger.set_frontier('slow', end + 1)
    ledger.advance(span.deadline + 1, span.deadline + 1)
    assert not ledger.spans


def test_overlapping_spans_are_independent():
    first = UISpan(1, 0, 4, 0)
    second = UISpan(2, 0, 4, first.period)
    ledger = UISpanLedger()
    ledger.publish(first)
    ledger.publish(second)
    ledger.resolve(first.id, 'singleton', count=1,
                   validated_end=first.first_training_origin + UI.BODY_SAMPLES[4])
    assert ledger.classify(observation(first, 1)) == CONTESTED
    assert ledger.classify(observation(second, 17)) == CONTESTED


def test_refinements_events_and_resolutions_are_idempotent():
    span = UISpan(1, 7, 4, 0)
    ledger = UISpanLedger(7)
    for _ in range(2):
        ledger.apply(SpanPublished(span))
        ledger.publish(replace(span, id=2))
    assert len(ledger.spans) == 1
    refined = replace(span, possible_packet_count=3)
    ledger.apply(SpanPublished(refined))
    end = refined.classification_end
    event = SpanResolved(span.id, 7, 'group', end, 3)
    ledger.apply(event)
    resolved = ledger.spans
    ledger.apply(event)
    ledger.resolve(span.id, 'failed')
    ledger.publish(span)
    ledger.apply(replace(event, tune_epoch=8, count=1))
    assert ledger.spans == resolved
    ledger.advance(end + 1, end + 1)
    assert not ledger.spans
    ledger.apply(event)
    ledger.publish(span)
    assert not ledger.spans


def test_tune_reset_ignores_old_events_and_decoder_reset_keeps_coverage():
    span = UISpan(1, 7, 1, 0)
    ledger = UISpanLedger(7)
    ledger.publish(span)
    for decoder_epoch in (0, 10, 100):
        assert ledger.classify(observation(span, 17, decoder_epoch=decoder_epoch)) == CONTESTED
    ledger.retune(8)
    ledger.publish(span)
    ledger.resolve(span.id, 'group', tune_epoch=7, count=1, validated_end=1)
    assert not ledger.spans
    assert ledger.classify(observation(span)) == CLEAR
    current = replace(span, tune_epoch=8)
    ledger.publish(current)
    ledger.apply(SpanResolved(span.id, 7, 'singleton', 1, 1))
    assert ledger.spans == (current,)


def test_all_routes_must_pass_history():
    span = UISpan(1, 0, 4, 0, 2)
    ledger = UISpanLedger()
    for route in ('streaming', 'segmented', 'fallback'):
        ledger.set_frontier(route, 0)
    ledger.publish(span)
    end = span.classification_end
    for route in ('streaming', 'segmented'):
        ledger.set_frontier(route, end + 1)
    ledger.advance(end + 1, end + 1)
    assert ledger.spans
    ledger.set_frontier('fallback', end + 1)
    ledger.advance(end + 1, end)
    assert not ledger.spans


def test_without_routes_detector_watermark_establishes_the_result_floor():
    span = UISpan(1, 0, 4, 0, 2)
    ledger = UISpanLedger()
    ledger.publish(span)
    end = span.classification_end
    ledger.advance(end + 1, end)
    assert ledger.spans
    ledger.advance(end + 1, end + 1)
    assert not ledger.spans


def test_inactive_then_active_virtual_origin_cannot_bypass_retired_span():
    span = UISpan(1, 0, 1, 0, 2)
    ledger = UISpanLedger()
    ledger.publish(span)
    clock = span.deadline + 1
    ledger.set_frontier('streaming', clock)
    ledger.set_frontier('inactive', clock)
    ledger.advance(clock, clock)
    assert not ledger.spans
    floor = ledger.result_floor
    ledger.set_frontier('new_fallback', clock - 200000)
    ledger.set_frontier('inactive', 0)
    assert ledger.result_floor == floor
    assert ledger.classify(observation(span, 1, decoder_epoch=20)) == STALE
    assert ledger.classify(replace(observation(span), training_origin=floor)) == CLEAR


@pytest.mark.parametrize('stalled_detector', [False, True])
def test_sustained_false_headers_have_bounded_metadata_and_route_retention(stalled_detector):
    stride, retention = 24000, 96000
    ledger = UISpanLedger(route_retention=retention)
    ledger.set_frontier('stalled', 0)
    maximum = ceil((MAX_HORIZON + retention) / stride) + 2
    for i in range(2000):
        clock = i * stride
        ledger.publish(UISpan(i, 0, 1, clock))
        ledger.advance(clock, 0 if stalled_detector else clock)
        assert len(ledger.spans) + len(ledger._retired) <= maximum
        assert ledger.result_floor >= clock - retention
    old = UISpan(3000, 0, 1, 0)
    ledger.publish(old)
    assert old not in ledger.spans
    assert ledger.classify(observation(old, 17)) == STALE
    ledger.set_frontier('stalled', clock + MAX_HORIZON + 1)
    ledger.advance(clock + MAX_HORIZON + 1, clock + MAX_HORIZON + 1)
    assert not ledger.spans and not ledger._retired
