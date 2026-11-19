# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Anchored repeats of onair-20261114 K4MSU and onair-0923-1636 VE3KPG."""
import contextlib
from functools import lru_cache
import io
import json
import sys
import time

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, p4probe, session
from hfmodem.shrike.arq import PactorArq
from hfmodem.shrike import pactor1
from hfmodem.tests.evidence import CAPTURES, OPERATOR_DATA

FS, SLOT, SPS = 48000, 60000, 480
K4MSU = OPERATOR_DATA / 'captures/onair-20261114T020853404736Z-6622e5fd-8875-4776-85c4-2a5a6d1b650d'
VE3KPG = CAPTURES / 'onair-0923-1636'


@lru_cache(maxsize=2)
def _tape(peer):
    folder, anchor = (K4MSU, 2772) if peer == 'K4MSU' else (VE3KPG, 3585)
    stream = next(folder.glob('stream*.flac'), None)
    if stream is None:
        pytest.skip(f'{peer} recording is not available')
    audio = session.load_wav(str(stream.resolve()), FS).astype(np.float32)
    if peer == 'K4MSU':
        submitted = next(folder.glob('submitted-tx-*.flac'))
        own = session.load_wav(str(submitted.resolve()), FS)
        idx = np.flatnonzero(np.abs(own) > 1e-4)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 480) + 1):
            if run.size:
                audio[max(0, run[0] + 84 - 240):run[-1] + 84 + 240] = 0
    else:
        keep = np.zeros(audio.size, bool)
        for path in folder.glob('rx_*.json'):
            d = json.loads(path.read_text())
            keep[d['end_stream_sample'] - d['samples']:d['end_stream_sample']] = True
        audio[~keep] = 0
    return audio[anchor:]


def _grid():
    return onair._MasterGrid(0, SLOT, 0, packet_n=46080, cs_n=5760,
                             d_max_n=onair._d_max_n(1.25, .04))


def _window(audio, slot):
    start = slot * SLOT + round(.88 * FS)
    return audio[start:(slot + 1) * SLOT - round(.089 * FS)], start


def _single_window(audio, slot):
    start = slot * SLOT + round(.88 * FS)
    return audio[start:(slot + 1) * SLOT - round(.043 * FS)], start


def _offer(anchor, audio, slot, grid):
    seg, start = _window(audio, slot)
    anchor.offer(seg, start, slot * SLOT + round(.88 * FS), (slot + 1) * SLOT,
                 grid, grid.d_max_n, False, slot)
    return seg, start


def _confirm(anchor, audio, slot, grid):
    seg, start = _window(audio, slot)
    return anchor.confirm(seg, start, slot * SLOT + round(.88 * FS), grid, slot)


def _replay(tmp_path, monkeypatch, peer, after):
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
        if len(self.slots_used) > before:
            keyed.append((self.slot, what))
        return result

    monkeypatch.setattr(PactorArq, 'on_rx_robust_answer', accepted)
    monkeypatch.setattr(onair.RadioTx, '_tx', tx)
    monkeypatch.setattr(onair, '_run_p4_link', lambda *a, **kw: 'opening handed to link')
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    monkeypatch.setattr(p4probe, 'code_revision', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', [
        'onair', '--replay', str(wav), '--mycall', 'W9SSJ', '--dxcall', peer,
        '--dial', '3593500', '--hold', '0', '--connect', 'robust',
        '--after-robust', after, '--retries', '20' if peer == 'K4MSU' else '30',
        '--max-cycles', '32' if peer == 'K4MSU' else '40',
        '--outdir', str(tmp_path / 'out'),
        *(['--rig', 'ftdx10', '--serial', '/dev/null', '--ptt-port', '/dev/null']
          if peer == 'K4MSU' else [])])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    (tmp_path / 'replay.log').write_text(log.getvalue())
    calls = [s for s, label in keyed if label.startswith('connect')]
    opening = [(s, label) for s, label in keyed if not label.startswith('connect')]
    print(f'{peer} {after}: calls={calls}, opening={opening[:3]}, decision-to-tx={elapsed} ms')
    return calls, opening, log.getvalue(), elapsed


@pytest.mark.parametrize('peer', ['K4MSU', 'VE3KPG'])
def test_recorded_answer_replaces_next_call_with_consecutive_shorts(tmp_path, monkeypatch, peer):
    calls, opening, log, elapsed = _replay(tmp_path, monkeypatch, peer, 'p4-robust')
    slots = [s for s, _ in opening[:3]]
    assert slots == list(range(calls[-1] + 1, calls[-1] + 4)), log
    # K4MSU repeats its entry through the opening, so v6/A is keyed again.
    values = (6, 6, 6) if peer == 'K4MSU' else (6, 9, 8)
    assert all(f'v={v} hdr=A' in label for (_, label), v in zip(opening, values))
    if peer == 'K4MSU':
        assert 4 <= slots[0] <= 6, log
        assert calls[-1] <= 5, log
    else:
        assert calls[-1] == 24 and slots[0] == 25, log
    assert elapsed and elapsed[0] < 49, elapsed


def test_k4msu_p2_bridge_replaces_next_call(tmp_path, monkeypatch):
    calls, opening, log, _ = _replay(tmp_path, monkeypatch, 'K4MSU', 'p2')
    assert opening[0][0] == calls[-1] + 1, log
    assert opening[0][1].startswith('P2 ROBUST OPENING'), opening
    assert calls[-1] <= 5, log


@pytest.mark.parametrize('peer,noise_slots,phase,entry', [
    ('K4MSU', 3, .985, 3), ('VE3KPG', 24, .956, 0)])
def test_noise_slots_cannot_confirm_any_offered_phase(peer, noise_slots, phase, entry):
    audio, grid = _tape(peer), _grid()
    offered, accepts, single = [], 0, 0
    for slot in range(noise_slots):
        anchor = onair._RobustAnswerAnchor()
        _offer(anchor, audio, slot, grid)
        seg, start = _single_window(audio, slot)
        single += onair._robust_single_answer(seg, start, slot * SLOT + round(.88 * FS),
            (slot + 1) * SLOT, grid, grid.d_max_n, False) is not None
        offered.extend(anchor.candidates)
    # Every offered phase is tried against every noise slot, with both parities;
    # this also exercises the one-error sweep win in 16:36 slot 7.
    for slot in range(noise_slots):
        for _, at, e, parity in [*offered, (0, round(phase * FS), entry, 0)]:
            for p in (parity, parity ^ 1):
                anchor = onair._RobustAnswerAnchor()
                anchor.candidates = [(slot - 1, at, e, p)]
                accepts += _confirm(anchor, audio, slot, grid) is not None
    print(f'{peer}: {len(offered)}/{noise_slots} candidate windows, {accepts} repeat accepts, '
          f'{single}/{noise_slots} recorded noise-slot single accepts')
    assert accepts == single == 0


@pytest.mark.parametrize('peer,slots,phase,entry', [
    ('K4MSU', range(3, 16), .985, 3), ('VE3KPG', range(24, 28), .956, 0)])
def test_answer_rejects_wrong_phase_and_parity(peer, slots, phase, entry):
    audio, grid = _tape(peer), _grid()
    for slot in slots:
        for delta, parity in ((-.4, 0), (-.03, 0), (.03, 0), (0, 1)):
            anchor = onair._RobustAnswerAnchor()
            anchor.candidates = [(slot - 1, round((phase + delta) * FS), entry, parity)]
            assert _confirm(anchor, audio, slot, grid) is None, (peer, slot, delta, parity)


def _burst(entry=3, bits=20, sense=0, faded=None, flip=0):
    word = p1rx.ROBUST_CS_WORDS[entry] ^ flip
    freqs = pactor1._fsk_freqs([(word >> j) & 1 for j in range(bits)], SPS, bool(sense))
    wave = .3 * np.cos(2 * np.pi * np.cumsum(freqs) / FS)
    return wave * np.where(freqs == faded, .03, 1)


@pytest.mark.parametrize('faded', [None, p1rx.MARK, p1rx.SPACE])
@pytest.mark.parametrize('sense', [0, 1])
def test_sweep_finds_faded_full_word_but_only_offers_a_candidate(faded, sense):
    audio = np.random.default_rng(11).normal(0, .005, SLOT)
    at = round(.985 * FS)
    audio[at:at + 20 * SPS] += _burst(sense=sense, faded=faded)
    got = p1rx.sweep_robust_answer(audio, .90, .11)
    assert (got.entry, got.sense, got.bits, got.errors) == (3, sense, 20, 0)
    assert abs(got.at - .985) < .003
    grid, anchor = _grid(), onair._RobustAnswerAnchor()
    _offer(anchor, audio, 0, grid)
    assert anchor.confirm(audio, 0, round(.88 * FS), grid, 0) is None
    assert anchor.confirm(audio, SLOT, SLOT + round(.88 * FS), grid, 1) is None


@pytest.mark.parametrize('faded', [None, p1rx.MARK, p1rx.SPACE])
def test_held_prefix_reads_sixteen_bits_exactly(faded):
    wave = _burst(bits=16, sense=1, faded=faded)
    got = p1rx.read_held_answer(wave, 0, sense=1, words={3: p1rx.ROBUST_CS_WORDS[3]}, bits=16)
    assert (got.entry, got.sense, got.bits, got.errors) == (3, 1, 16, 0)
    assert p1rx.read_held_answer(wave, 0, sense=1) is None
    assert p1rx.read_held_answer(wave, 0, sense=0, bits=16) is None
    wave = _burst(bits=16, sense=1, faded=faded, flip=1 << 8)
    assert p1rx.read_held_answer(wave, 0, sense=1, bits=16) is None


def test_seeded_noise_run_has_no_anchored_accepts():
    rng, grid, anchor = np.random.default_rng(1114), _grid(), onair._RobustAnswerAnchor()
    candidates = accepts = single = prekey = 0
    windows = 2000
    for slot in range(windows):
        window = rng.normal(0, .05, SLOT)
        start = slot * SLOT
        accepts += anchor.confirm(window, start, start + round(.88 * FS), grid, slot) is not None
        anchor.offer(window, start, start + round(.88 * FS), (slot + 1) * SLOT,
                     grid, grid.d_max_n, False, slot)
        candidates += sum(c[0] == slot for c in anchor.candidates)
        single += onair._robust_single_answer(window, start, start + round(.88 * FS),
            start + SLOT, grid, round(.13 * FS), False) is not None
        seg, origin = _single_window(window, 0)
        prekey += onair._robust_single_answer(seg, start + origin, start + round(.88 * FS),
            start + SLOT, grid, grid.d_max_n, False) is not None
    print(f'seeded noise: {accepts}/{windows} accepts, {candidates}/{windows} candidates '
          f'({candidates / windows:.1%})')
    print(f'single-slot gate: {single}/{windows} accepts; '
          f'pre-key geometry: {prekey}/{windows} accepts')
    assert accepts == single == prekey == 0
