# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""ADC loss and timestamp noise have independent sample-clock effects."""
import os
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core import recording
from hfmodem.tests.station.recording_support import TIMEOUT, bounded, flush, pcm, samples


@pytest.fixture
def adc_child(sessions):
    controls = []
    def attach(parent):
        child, _ = sessions('off', attached_children=False)
        tap = child.stream('tap', rate=48000, clock='tap')
        tap.row.update(monotonic_origin=1000., clock_uncertainty_s=0.)
        rd, wr = os.pipe()
        sender = recording.ChildControlSender(wr)
        sender.bind(parent, anchors={})
        receiver = recording.ChildControlReceiver(child, rd)
        controls.append((sender, rd))
        def delivered():
            sender.close(timeout=TIMEOUT)
            receiver.drain()
            flush(child)
            return child.index['contacts']['loss']
        return delivered
    yield attach
    for sender, rd in controls:
        sender.close(timeout=TIMEOUT)
        os.close(rd)


@pytest.mark.parametrize('ppm', (-200., 200.))
def test_adc_irregular_jitter_does_not_accumulate_rate_error(sessions, ppm):
    rec, _ = sessions('off', attached_children=False)
    lane = rec.stream('rx', rate=48000, clock='adc')
    measured = lane.rate * (1 + ppm / 1e6)
    rng = np.random.default_rng(0)
    for n in range(2000):
        adc = 100. + (n * 128 + rng.uniform(-8., 8.)) / measured
        lane.push_adc(pcm(0, 128), NS(inputBufferAdcTime=adc, currentTime=adc),
                      captured_at=900. + adc)
    assert lane.position == 2000 * 128
    assert lane.gaps == []
    assert 'uncertain' not in lane.row


@pytest.mark.parametrize('reset', (False, True))
def test_adc_small_loss_before_rate_fit_keeps_exact_gap_and_bounds(sessions, adc_child, reset):
    rec, root = sessions('everything', attached_children=False)
    delivered = adc_child(rec)
    lane = rec.stream('rx', rate=48000, clock='adc')
    at = 0
    if reset:
        for at in range(0, 128 * 40, 128):
            lane.push_adc(pcm(at, at + 128), NS(inputBufferAdcTime=at / lane.rate),
                          captured_at=1000. + at / lane.rate)
            flush(rec)
        at += 128
        lane.push_adc(pcm(at, at + 128), NS(), captured_at=1000. + at / lane.rate)
        flush(rec)
        assert 'measured_rate' not in lane.row
        at += 128
    for offset in (0, 160, 288, 416):
        first = at + offset
        lane.push_adc(pcm(first, first + 128), NS(inputBufferAdcTime=first / lane.rate),
                      captured_at=1000. + first / lane.rate)
        flush(rec)
        assert 'measured_rate' not in lane.row
    rec.begin('loss', {'adc': at + 160})
    rec.end('loss', {'adc': at + 288})
    child = delivered()
    assert child['positions'] == {'tap': at + 160}
    assert child['end_positions'] == {'tap': at + 288}
    bounded(rec.close)
    assert lane.gaps == [[at + 128, at + 160]]
    assert lane.position == at + 544
    row = rec.index['contacts']['loss']
    assert row['positions'] == {'adc': at + 160}
    assert row['end_positions'] == {'adc': at + 288}
    assert row['lanes']['rx']['actual'] == [0, at + 544]
    assert len(samples(root, row['lanes']['rx'])) == at + 512


@pytest.mark.parametrize('ppm', (-200., 0., 200.))
@pytest.mark.parametrize('loss', (False, True))
def test_adc_jitter_and_drift_preserve_pcm_child_and_contact_bounds(sessions, ppm, loss):
    parent, root = sessions('everything', attached_children=False)
    child, _ = sessions('off', attached_children=False)
    rate, size = 48000, 128
    measured = rate * (1 + ppm / 1e6)
    lane = parent.stream('rx', rate=rate, clock='adc')
    tap = child.stream('tap', rate=rate, clock='tap')
    tap.row.update(monotonic_origin=1000., measured_rate=measured, clock_uncertainty_s=0.)
    rd, wr = os.pipe()
    sender = recording.ChildControlSender(wr)
    sender.bind(parent, anchors={})
    receiver = recording.ChildControlReceiver(child, rd)
    expected = []
    try:
        for n in range(400):
            first = n * size + (64 if loss and n >= 320 else 0)
            jitter = (0, 2, -2, 8, -8, -2, 2, 0)[n % 8]
            if n == 300:
                jitter = 20
            adc = 100. + (first + jitter) / measured
            block = pcm(first, first + size)
            lane.push_adc(block, NS(inputBufferAdcTime=adc, currentTime=adc),
                          captured_at=900. + adc)
            expected.append(np.round(block * 32768).astype('<i2'))
            flush(parent)
        first, last = 350 * size + (64 if loss else 0), 375 * size + (64 if loss else 0)
        parent.begin('jitter', {'adc': first})
        parent.end('jitter', {'adc': last})
        sender.close(timeout=TIMEOUT)
        receiver.drain()
        flush(child)
        bounded(parent.close)
        assert lane.position == 400 * size + (64 if loss else 0)
        assert lane.gaps == ([[320 * size, 320 * size + 64]] if loss else [])
        assert 'uncertain' not in lane.row
        row = child.index['contacts']['jitter']
        assert row['positions'] == {'tap': first}
        assert row['end_positions'] == {'tap': last}
        ref = parent.index['contacts']['jitter']['lanes']['rx']
        assert ref['actual'] == [0, lane.position]
        np.testing.assert_array_equal(samples(root, ref), np.concatenate(expected))
    finally:
        sender.close(timeout=TIMEOUT)
        os.close(rd)


class _SnapshotGate:
    """Holds the drainer inside its copy of the index until the callback has run."""

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()

    def __deepcopy__(self, memo):
        if threading.current_thread().name == 'recording-pcm' and not self.entered.is_set():
            self.entered.set()
            self.release.wait(TIMEOUT)
        return None


def test_adc_row_written_by_the_callback_does_not_break_the_index_snapshot(sessions):
    reports = []
    rec, _ = sessions('everything', attached_children=False, report=reports.append)
    lane = rec.stream('rx', rate=48000, clock='adc')

    def callback(first, adc):
        lane.push_adc(pcm(first, first + 128), NS(inputBufferAdcTime=adc, currentTime=adc),
                      captured_at=1000. + adc)

    callback(0, 0.)
    flush(rec)
    gate = lane.row['gate'] = _SnapshotGate()
    callback(128, 128 / lane.rate)
    if not gate.entered.wait(TIMEOUT):
        raise TimeoutError('the drainer never copied the index')
    # A stamp behind the fit, as a late WASAPI callback reads, makes the
    # callback add 'uncertain' to the row the drainer is part way through.
    callback(256, 0.)
    gate.release.set()
    flush(rec)
    bounded(rec.close)
    assert 'uncertain' in lane.row
    assert not [r for r in reports if 'changed size during iteration' in r]
    assert lane.gaps == [[256, 257]]


def test_an_empty_push_publishes_timing_alone(sessions):
    rec, _ = sessions('everything', attached_children=False)
    lane = rec.stream('tx', rate=48000, clock='output')
    lane.push(0, pcm(0, 128))
    marker = dict(from_sample=0, to_sample=128, reason='output underrun')
    lane.push(lane.position, pcm(0, 0), timing=dict(uncertain=marker))
    assert lane.timing()['uncertain'] == marker
    assert (lane.origin, lane.position) == (0, 128)
    lane.push(128, pcm(128, 256))
    flush(rec)
    bounded(rec.close)
    assert lane.row['uncertain'] == marker
    assert lane.gaps == [] and lane.position == 256


def test_overflow_flag_the_stamps_account_for_adds_no_gap(sessions):
    """CoreAudio flags an overflow blocks after the stall its stamps already placed."""
    rec, _ = sessions('everything', attached_children=False)
    lane = rec.stream('rx', rate=48000, clock='adc')
    flagged = {1, 40, 43}
    for n in range(80):
        first = n * 128 + (184 if n >= 41 else 0)
        adc = first / lane.rate
        lane.push_adc(pcm(first, first + 128), NS(inputBufferAdcTime=adc, currentTime=adc),
                      overflow=n in flagged, captured_at=1000. + adc)
    flush(rec)
    assert lane.gaps == [[41 * 128, 41 * 128 + 184]]
    assert 'uncertain' not in lane.row


def test_overflow_flag_without_stamps_is_still_loss(sessions):
    rec, _ = sessions('everything', attached_children=False)
    lane = rec.stream('rx', rate=48000, clock='adc')
    for n in range(4):
        lane.push_adc(pcm(0, 128), None, overflow=n == 2, captured_at=1000. + n * 128 / 48000)
    flush(rec)
    assert lane.gaps == [[256, 257]]
    assert lane.row['uncertain']['reason'] == 'input loss length unknown'


def test_a_flagged_loss_below_the_jitter_tolerance_is_still_a_gap(sessions):
    rec, _ = sessions('off', attached_children=False)
    lane = rec.stream('rx', rate=48000, clock='adc')
    for n in range(1000):
        first = n * 128 + (8 if n >= 400 else 0)
        adc = first / lane.rate
        lane.push_adc(pcm(first, first + 128), NS(inputBufferAdcTime=adc, currentTime=adc),
                      overflow=n == 400, captured_at=1000. + adc)
    assert lane.gaps == [[400 * 128, 400 * 128 + 8]]
    assert lane.position == 128008
