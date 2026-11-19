# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Held robust words are diagnostic evidence on the original answer clock."""
from types import SimpleNamespace
import re
import time

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx
from hfmodem.shrike.arq import PactorArq
from hfmodem.tests.shrike.test_robust_answer_early_1114 import _burst
from hfmodem.tests.shrike.test_robust_first_answer_1117 import (
    FS, SLOT, _replay, _single_window, _tape,
)


def _observer():
    audio, start = _single_window(_tape('D'), 17)
    got = p1rx.sweep_robust_answer(audio, .02, .11)
    onset = start + round(got.at * FS)
    return onair._RobustHeldObserver(got, 17, onset, onset - 17 * SLOT)


def _window(slot):
    start = slot * SLOT + round(.88 * FS)
    return _tape('D')[start:(slot + 1) * SLOT - round(.04 * FS)], start


def test_d20_to_d28_are_nine_exact_held_observations(capsys):
    observer = _observer()
    tx = SimpleNamespace(robust_observer=observer)
    for slot in range(20, 29):
        audio, start = _window(slot)
        rows = onair._observe_robust_held(tx, audio, start)
        assert len(rows) == 1, (slot, rows)
        row = rows[0]
        assert (row['slot'], row['entry'], row['word'], row['errors']) == (slot, 3, 0x095AC5, 0)
        assert abs(row['onset'] - slot * SLOT - round(.985 * FS)) < round(.005 * FS)
        assert 16 <= row['bits'] <= 20
        if slot == 21:
            assert row['bits'] == 18 and row['tone'] == 1400
        assert not onair._observe_robust_held(tx, audio, start)
    log = capsys.readouterr().out
    assert log.count('[robust held]') == 9
    assert all(f'slot={slot} ' in log for slot in range(20, 29))
    assert onair._held_answers([], observer) == (
        'gateway held robust entry 3 (095AC5) in slots 20–28; no P3 control decoded')


def test_observer_local_cost():
    costs = []
    for slot in range(20, 29):
        audio, start = _window(slot)
        for _ in range(30):
            observer = _observer()
            began = time.perf_counter()
            assert observer.observe(audio, start)
            costs.append((time.perf_counter() - began) * 1000)
    print(f'held observer: median={np.median(costs):.3f} max={max(costs):.3f} ms')
    assert max(costs) < 5, costs


@pytest.mark.parametrize('wrong', ['phase', 'parity', 'entry', 'noise'])
def test_observer_rejects_wrong_anchor_and_noise(wrong):
    observer = _observer()
    audio, start = _window(21)
    if wrong == 'phase':
        observer.onset -= round(.03 * FS)
    elif wrong == 'parity':
        observer.sense ^= 1
    elif wrong == 'entry':
        observer.entry = 0
    elif wrong == 'noise':
        audio = np.random.default_rng(1114).normal(0, .05, len(audio))
    assert not observer.observe(audio, start)
    assert observer.summary() == 'held robust repeats unread'


def test_overlapping_windows_retry_only_with_more_search_slack(monkeypatch):
    observer = _observer()
    audio, start = _window(21)
    at = observer.onset + 4 * SLOT - start
    reads = []

    def unread(*args, **kwargs):
        reads.append(kwargs['bits'])
        return None

    monkeypatch.setattr(p1rx, 'read_held_answer', unread)
    for bits in range(16, 21):
        end = at + bits * 480
        for _ in range(3):
            assert not observer.observe(audio[:end], start)
    end = at + 20 * 480 + round(p1rx.HELD_ANSWER_SEARCH_S * FS)
    for _ in range(3):
        assert not observer.observe(audio[:end], start)
    assert reads == [16, 17, 16, 18, 17, 19, 18, 20, 19, 20]


def test_floor_report_matches_only_the_measured_burst():
    observer = _observer()
    audio, start = _window(20)
    row, = observer.observe(audio, start)
    tx = SimpleNamespace(robust_observer=observer)
    line = '1 burst; 1 UNDER THE FLOOR, nearest +11 ms -- a transmission where only a fast turnaround fits'
    report = onair._robust_grid_report(line, [row['onset']], tx)
    assert 'measured held robust entry 3 (095AC5)' in report
    assert onair._robust_grid_report(line, [row['onset'] - FS], tx) == line


def test_failed_observation_is_reported_as_unread(capsys):
    observer = _observer()
    audio, start = _window(21)
    rows = onair._observe_robust_held(SimpleNamespace(robust_observer=observer),
                                     np.zeros_like(audio), start)
    assert rows == [dict(slot=21, unread=True)]
    assert 'slot=21 unread' in capsys.readouterr().out
    assert not observer.observations


def test_observer_retires_after_four_consecutive_unread_slots():
    phase = round(.985 * FS)
    answer = p1rx.RobustAnswer(3, p1rx.ROBUST_CS_WORDS[3], 0, 20, 0, .985)
    observer = onair._RobustHeldObserver(answer, 0, phase, phase)
    quiet = np.zeros(SLOT)
    for slot in range(1, 4):
        assert not observer.observe(quiet, slot * SLOT)
        assert not observer.observe(quiet, slot * SLOT)
        assert not observer.retired
    heard = quiet.copy()
    heard[phase:phase + 20 * 480] = _burst()
    row, = observer.observe(heard, 4 * SLOT)
    assert row['slot'] == 4 and not observer.retired
    for slot in range(5, 9):
        assert not observer.observe(quiet, slot * SLOT)
        assert observer.retired == (slot == 8)
    checked = {slot: dict(bits) for slot, bits in observer.checked.items()}
    assert not observer.observe(heard, 10 * SLOT)
    assert observer.checked == checked
    assert list(observer.observations) == [4]


def test_observer_retries_prefix_when_more_alignments_fit():
    phase = round(.985 * FS)
    answer = p1rx.RobustAnswer(3, p1rx.ROBUST_CS_WORDS[3], 0, 20, 0, .985)
    observer = onair._RobustHeldObserver(answer, 0, phase, phase)
    audio = np.zeros(SLOT)
    burst = _burst(sense=1, flip=0xF0000)
    audio[phase + 192:phase + 192 + len(burst)] = burst
    for bits in (16, 17):
        observer.observe(audio[:phase + bits * 480], SLOT)
    observer.observe(audio, SLOT)
    fresh = onair._RobustHeldObserver(answer, 0, phase, phase)
    expected = fresh.observe(audio, SLOT)
    assert len(expected) == 1
    row, = observer.observations.values()
    assert (row['entry'], row['word'], row['errors']) == (3, 0x095AC5, 0)


@pytest.mark.parametrize('slots, bits', [((1, 3, 5, 7), 20), ((1, 2, 3, 4), 16)])
def test_observer_does_not_retire_after_nonconsecutive_misses(slots, bits):
    phase = round(.985 * FS)
    answer = p1rx.RobustAnswer(3, p1rx.ROBUST_CS_WORDS[3], 0, 20, 0, .985)
    observer = onair._RobustHeldObserver(answer, 0, phase, phase)
    quiet = np.zeros(SLOT)
    for slot in slots:
        assert not observer.observe(quiet[:phase + bits * 480], slot * SLOT)
    heard = quiet.copy()
    heard[phase:phase + 20 * 480] = _burst()
    row, = observer.observe(heard, 8 * SLOT)
    assert row['slot'] == 8 and row['entry'] == 3 and row['errors'] == 0
    assert not observer.retired


def test_linked_replay_observes_d_without_changing_the_link(tmp_path, monkeypatch):
    results = []
    for enabled in (True, False):
        directory = tmp_path / str(enabled)
        directory.mkdir()
        events, transitions, grids = [], [], []
        with monkeypatch.context() as patch:
            original_on, original_to = onair._SessionRx._on, PactorArq._to
            original_update = onair._MasterGrid.update

            def event(self, ev, *a, **kw):
                events.append(repr(ev))
                return original_on(self, ev, *a, **kw)

            def transition(self, state):
                transitions.append((self.state, state))
                return original_to(self, state)

            def update(self, *a, **kw):
                result = original_update(self, *a, **kw)
                grids.append({k: repr(v) for k, v in vars(self).items()
                              if k not in ('evidence', 'sessrx')})
                grids[-1]['evidence'] = repr(vars(self.evidence))
                return result

            patch.setattr(onair._SessionRx, '_on', event)
            patch.setattr(PactorArq, '_to', transition)
            patch.setattr(onair._MasterGrid, 'update', update)
            if not enabled:
                patch.setattr(onair, '_observe_robust_held', lambda *a: [])
            calls, opening, log, _ = _replay(directory, patch, 'D', 'p3', hold=12)
        results.append((calls, opening, events, transitions, grids))
        if enabled:
            observations = re.findall(r'\[robust held\] entry 3 \(095AC5\) slot=(\d+) '
                                      r'.*bits=(\d+) errors=0 tone=(\d+)', log)
            assert [int(s) for s, _, _ in observations if 20 <= int(s) <= 28] == list(range(20, 29)), log
            assert ('21', '18', '1400') in observations
            ended, = [line for line in log.splitlines() if line.startswith('session ended:')]
            assert 'gateway held robust entry 3 (095AC5) in slots 20–29' in ended, ended
            assert 'no P3 control decoded' in ended
            assert not any('held robust' in line for line in log.splitlines()
                           if line.startswith('verdict:'))
            assert 'measured held robust entry 3' in log
    differences = [(i, k, left[k], right[k])
                   for i, (left, right) in enumerate(zip(results[0][4], results[1][4]))
                   for k in left if left[k] != right[k]]
    assert not differences, differences[:10]
    assert results[0] == results[1]
