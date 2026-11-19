# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Held answers at the callback-delivered end of the bridge's listen."""
from functools import lru_cache

import numpy as np
import pytest

from hfmodem.shrike import p1rx, robust, session
from hfmodem.tests.evidence import OPERATOR_DATA

FS, SLOT = 48000, 60000
TAPES = {
    'B': ('onair-20261116T221907072730Z-78d033ab-a99b-4a03-bffb-6fcfa22be38e',
          2795, round(.9865 * FS), 18),
    'C': ('onair-20261117T043105948220Z-04ec851e-15cc-43c2-a59c-38d24696500f',
          2800, round(.9880208333333333 * FS), 5),
}
CASES = [('B', slot) for slot in range(19, 29)] + [('C', 9)]


@lru_cache(maxsize=2)
def _tape(name):
    folder = OPERATOR_DATA / 'captures' / TAPES[name][0]
    stream = next(folder.glob('stream*.flac'), None)
    if stream is None:
        pytest.skip(f'{name} recording is not available')
    audio = session.load_wav(str(stream.resolve()), FS).astype(np.float32)
    submitted = next(folder.glob('submitted-tx-*.flac'), None)
    if submitted is None:
        pytest.skip(f'{name} submitted TX recording is not available')
    own = session.load_wav(str(submitted.resolve()), FS)
    idx = np.flatnonzero(np.abs(own) > 1e-4)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 480) + 1):
        if run.size:
            audio[run[0] + 756:run[-1] + 757] = 0
    return audio


def _window(name, slot, endpoint=None):
    _, anchor, phase, answer_slot = TAPES[name]
    boundary = anchor + slot * SLOT
    deadline = boundary + SLOT - 1920 - 916
    if endpoint is None:
        endpoint = 128 * ((deadline - 788) // 128 + 1)
    origin = boundary + round(.88 * FS)
    bridge = robust.Bridge('W9SSJ', phase, fsk_sense=0, fsk_slot=answer_slot)
    return bridge, _tape(name)[origin:endpoint], origin, boundary


def _assert_fsk(bridge, audio, origin, boundary, slot):
    bridge.silent = 1
    grant, report = bridge.read(audio, origin, boundary, slot)
    row = bridge.heard[-1]
    assert not grant and row['form'] == 'FSK', report
    assert row['entry'] == 0 and row['errors'] == 0
    assert 16 <= row['bits'] <= 20 and row['tone'] in (0, 1400, 1600)
    assert row['sense'] == bridge.fsk_sense ^ ((slot - bridge.fsk_slot) & 1)
    assert abs(row['onset'] - boundary - bridge.fsk_phase) <= round(.005 * FS)
    assert bridge.silent == 0
    return row


@pytest.mark.parametrize('name,slot', CASES)
@pytest.mark.parametrize('delta', [-128, 0, 128])
def test_callback_delivered_answers_are_not_silence(name, slot, delta):
    bridge, audio, origin, boundary = _window(name, slot)
    endpoint = origin + len(audio) + delta
    bridge, audio, origin, boundary = _window(name, slot, endpoint)
    at = boundary + bridge.fsk_phase - origin
    full = p1rx.read_held_answer(audio, at, sense=(slot - bridge.fsk_slot) & 1,
                                words={0: p1rx.ROBUST_CS_WORDS[0]})
    assert full is None
    _assert_fsk(bridge, audio, origin, boundary, slot)


def test_b21_retries_a_nominally_fitting_prefix():
    bridge, audio, origin, boundary = _window('B', 21, 1319296)
    at = boundary + bridge.fsk_phase - origin
    assert (len(audio) - at) // 480 == 19
    assert p1rx.read_held_answer(audio, at, sense=1, bits=19) is None
    row = _assert_fsk(bridge, audio, origin, boundary, 21)
    assert row['bits'] == 18


@pytest.mark.parametrize('name,slot,end_s',
                         [(name, slot, 1.191) for name, slot in CASES] + [('C', 9, 1.185)])
def test_longer_full_word_control(name, slot, end_s):
    anchor = TAPES[name][1]
    boundary = anchor + slot * SLOT
    bridge, audio, origin, boundary = _window(name, slot, boundary + round(end_s * FS))
    at = boundary + bridge.fsk_phase - origin
    full = p1rx.read_held_answer(audio, at, sense=(slot - bridge.fsk_slot) & 1)
    assert full is not None and full.errors == 0 and full.bits == 20
    expected_bits = min(20, (len(audio) - at) // 480)
    assert _assert_fsk(bridge, audio, origin, boundary, slot)['bits'] == expected_bits


@pytest.mark.parametrize('name,slot', CASES)
def test_1180_cut_needs_a_prefix(name, slot):
    boundary = TAPES[name][1] + slot * SLOT
    bridge, audio, origin, boundary = _window(name, slot, boundary + round(1.180 * FS))
    at = boundary + bridge.fsk_phase - origin
    sense = (slot - bridge.fsk_slot) & 1
    assert p1rx.read_held_answer(audio, at, sense=sense) is None
    got = p1rx.read_held_answer(audio, at, sense=sense, bits=19)
    assert got is not None and got.errors == 0
    assert _assert_fsk(bridge, audio, origin, boundary, slot)['bits'] == 19


@pytest.mark.parametrize('wrong', ['entry', 'parity', 'short'])
def test_bridge_rejects_wrong_or_short_held_answer(wrong):
    bridge, audio, origin, boundary = _window('B', 21)
    if wrong == 'entry':
        bridge.fsk_entry = 3
    elif wrong == 'parity':
        bridge.fsk_sense = 1
    else:
        audio = audio[:boundary + bridge.fsk_phase - origin + 15 * 480]
    grant, report = bridge.read(audio, origin, boundary, 21)
    assert not grant and report == 'silence'
    assert bridge.silent == 1


def test_prefix_rejects_errors_even_in_a_full_word(monkeypatch):
    reads = []

    def held(audio, at, fs, **kwargs):
        reads.append(kwargs)
        return p1rx.RobustAnswer(3, p1rx.ROBUST_CS_WORDS[3], 1,
                                kwargs['bits'], 1 if kwargs['bits'] == 20 else 0,
                                at / fs, 1400, kwargs['offset_hz'])

    monkeypatch.setattr(p1rx, 'read_held_answer', held)
    got = robust.read_held_prefix(np.zeros(10080), 480, entry=3, sense=1, offset_hz=82)
    assert got.bits == 19 and got.errors == 0 and got.offset_hz == 82
    assert [r['bits'] for r in reads] == [20, 19]
    assert all(r['words'] == {3: p1rx.ROBUST_CS_WORDS[3]} and r['sense'] == 1
               and r['offset_hz'] == 82 for r in reads)
    reads.clear()
    assert robust.read_held_prefix(np.zeros(10080), -1, entry=3, sense=1) is None
    assert robust.read_held_prefix(np.zeros(15 * 480), 0, entry=3, sense=1) is None
    assert not reads
