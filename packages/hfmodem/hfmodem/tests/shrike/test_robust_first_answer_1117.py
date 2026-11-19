# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Corroborated exact robust answers replace the next call in their first slot."""
import contextlib
from functools import lru_cache
import io
import sys
import time

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, p4probe, session
from hfmodem.shrike.arq import PactorArq
from hfmodem.tests.evidence import OPERATOR_DATA
from hfmodem.tests.shrike.test_robust_answer_early_1114 import (
    FS, SLOT, _grid, _single_window, _tape as _legacy_tape,
)

TAPES = {
    'A': ('onair-20261116T221745091177Z-e6ab9e8a-b882-4eff-87cb-de69544cddda', 5633),
    'B': ('onair-20261116T221907072730Z-78d033ab-a99b-4a03-bffb-6fcfa22be38e', 2795),
    'C': ('onair-20261117T043105948220Z-04ec851e-15cc-43c2-a59c-38d24696500f', 2800),
    'D': ('onair-20261117T043223037082Z-945bbd0f-6cd4-4e0a-95d6-a0df4a1de30f', 2774),
}


@lru_cache(maxsize=4)
def _tape(peer):
    if peer not in TAPES:
        return _legacy_tape(peer)
    capture, anchor = TAPES[peer]
    folder = OPERATOR_DATA / 'captures' / capture
    stream = next(folder.glob('stream*.flac'), None)
    if stream is None:
        pytest.skip(f'{peer} recording is not available')
    submitted = next(folder.glob('submitted-tx-*.flac'), None)
    if submitted is None:
        pytest.skip(f'{peer} submitted TX recording is not available')
    audio = session.load_wav(str(stream.resolve()), FS).astype(np.float32)
    own = session.load_wav(str(submitted.resolve()), FS)
    idx = np.flatnonzero(np.abs(own) > 1e-4)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 480) + 1):
        if run.size:
            audio[run[0] + 756:run[-1] + 757] = 0
    return audio[anchor:]


def _replay(tmp_path, monkeypatch, peer, after, *, hold=0):
    wav = tmp_path / 'peer.wav'
    session.write_wav(str(wav), _tape(peer))
    keyed, decision, elapsed = [], [], []
    key, answer = onair.RadioTx._tx, PactorArq.on_rx_robust_answer

    def accepted(self, entry):
        decision.append(time.perf_counter())
        return answer(self, entry)

    def tx(self, audio, what, *a, **kw):
        if decision and not elapsed and not what.startswith('connect'):
            elapsed.append((time.perf_counter() - decision[-1]) * 1000)
        before = len(self.slots_used)
        result = key(self, audio, what, *a, **kw)
        if (len(self.slots_used) > before
                or (not keyed and what.startswith('connect') and not self.refused)):
            keyed.append((self.slot, what))
        return result

    monkeypatch.setattr(PactorArq, 'on_rx_robust_answer', accepted)
    monkeypatch.setattr(onair.RadioTx, '_tx', tx)
    monkeypatch.setattr(onair, '_run_p4_link', lambda *a, **kw: 'opening handed to link')
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    monkeypatch.setattr(p4probe, 'code_revision', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', [
        'onair', '--replay', str(wav), '--mycall', 'W9SSJ',
        '--dxcall', 'W6IDS' if peer in ('C', 'D') else 'VE3KPG',
        '--dial', '3593500', '--hold', str(hold), '--connect', 'robust',
        '--after-robust', after, '--retries', '30',
        '--max-cycles', '40',
        '--outdir', str(tmp_path / 'out')])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    (tmp_path / 'replay.log').write_text(log.getvalue())
    calls = [s for s, label in keyed if label.startswith('connect')]
    opening = [(s, label) for s, label in keyed if not label.startswith('connect')]
    print(f'{peer} {after}: calls={calls}, opening={opening[:3]}, decision-to-tx={elapsed} ms')
    return calls, opening, log.getvalue(), elapsed


@pytest.mark.parametrize('peer,slot,sense', [('A', 0, 0), ('B', 17, 1)])
def test_exact_first_answer_window(peer, slot, sense):
    seg, start = _single_window(_tape(peer), slot)
    grid = _grid()
    keyed_at, boundary = slot * SLOT + round(.88 * FS), (slot + 1) * SLOT
    assert onair._robust_answer(seg, start, keyed_at, boundary, grid, grid.d_max_n, False) is None
    got, _ = onair._robust_single_answer(seg, start, keyed_at, boundary, grid, grid.d_max_n, False)
    assert (got.entry, got.bits, got.errors, got.tone, got.sense) == (0, 20, 0, 0, sense)


@pytest.mark.parametrize('peer,slot,after', [('A', 0, 'p4-robust'), ('B', 17, 'p2')])
def test_exact_first_answer_replaces_the_next_call(tmp_path, monkeypatch, peer, slot, after):
    calls, opening, log, elapsed = _replay(tmp_path, monkeypatch, peer, after)
    assert calls[-1] == slot and opening[0][0] == slot + 1, log
    assert elapsed and elapsed[0] < 49, elapsed


@pytest.mark.parametrize('peer,slot', [('C', 5), ('D', 17), *[('K4MSU', n) for n in range(6)]])
def test_single_gate_preserves_existing_answer_paths(peer, slot):
    seg, start = _single_window(_tape(peer), slot)
    grid = _grid()
    assert onair._robust_single_answer(seg, start, slot * SLOT + round(.88 * FS),
        (slot + 1) * SLOT, grid, grid.d_max_n, False) is None


@pytest.mark.parametrize('change', ['sweep-short', 'sweep-error', 'sweep-tone',
                                   'held-short', 'held-error', 'held-tone',
                                   'held-entry', 'held-sense', 'excess'])
def test_single_gate_requires_all_joint_conditions(monkeypatch, change):
    from dataclasses import replace
    candidate = p1rx.RobustAnswer(3, p1rx.ROBUST_CS_WORDS[3], 1, 20, 0, .105, 0, 82)
    got = replace(candidate)
    if change.startswith('sweep-'):
        candidate = replace(candidate, **{'short': {'bits': 19}, 'error': {'errors': 1},
                                          'tone': {'tone': 1400}}[change[6:]])
    elif change.startswith('held-'):
        got = replace(got, **{'short': {'bits': 19}, 'error': {'errors': 1},
                              'tone': {'tone': 1400}, 'entry': {'entry': 0},
                              'sense': {'sense': 0}}[change[5:]])
    reads = []

    def held(audio, at, **kwargs):
        reads.append((at, kwargs))
        return got

    monkeypatch.setattr(p1rx, 'sweep_robust_answer', lambda *a: candidate)
    monkeypatch.setattr(p1rx, 'read_held_answer', held)
    monkeypatch.setattr(onair, '_candidate_excess', lambda *a: 1.39 if change == 'excess' else 8)
    grid = _grid()
    assert onair._robust_single_answer(np.zeros(16000), round(.88 * FS),
        round(.88 * FS), SLOT, grid, grid.d_max_n, False) is None
    assert all(at == round(candidate.at * FS) and kwargs == dict(
        words={3: p1rx.ROBUST_CS_WORDS[3]}, sense=1, offset_hz=82, bits=20)
        for at, kwargs in reads)


@pytest.mark.parametrize('delta,sense', [(0, 1), (-.03, 0), (.03, 0)])
def test_single_gate_held_corroboration_rejects_wrong_phase_and_parity(monkeypatch, delta, sense):
    from dataclasses import replace
    seg, start = _single_window(_tape('A'), 0)
    candidate = p1rx.sweep_robust_answer(seg, .02, .11)
    monkeypatch.setattr(p1rx, 'sweep_robust_answer',
                        lambda *a: replace(candidate, at=candidate.at + delta, sense=sense))
    grid = _grid()
    assert onair._robust_single_answer(seg, start, round(.88 * FS), SLOT,
                                       grid, grid.d_max_n, False) is None

