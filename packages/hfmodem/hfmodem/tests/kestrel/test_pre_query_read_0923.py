# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KD0SFY, 2026-09-23: five queries keyed over answers held whole and unread.

The stream's hard readers lost each to a symbol or three under the clearance.
A continue is scored whole in the stream once its span has ended; a query answer
is read whole by the query's own last look.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests.evidence import require_input
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from .test_data_over_gate import _connected

FS = MK.FS
FIXTURES = Path(__file__).with_name('fixtures') / 'pre-query-read-0923'
MISSED = ['over3-continue', 'over5-continue', 'over6-continue', 'over7-continue']


def clip(name):
    path = FIXTURES / f'{name}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == FS and x.dtype == np.int16
    return x.astype(float) / 32768


def station(monkeypatch, called=None):
    prov = json.loads(require_input(FIXTURES / 'provenance.json').read_text())
    clock = [1000.0]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: clock[0])
    hs, io = _connected(bw=prov['bw'])
    hs.called = called or prov['called']
    hs._peer_offset = prov['offset_hz'] / (FS / MK.NFFT)
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'end')
    return hs, io, clock


def drive(hs, clock, audio, span):
    """Feed 20 ms blocks from the stream cursor, calling the query whenever the
    driver would, until the pending over moves or a query keys."""
    old, start = hs._tx_pending, clock[0]
    audio = np.pad(audio, (0, max(0, int(span * FS) - len(audio))))
    for at in range(0, len(audio), 960):
        hs.on_rx_stream(audio[at:at + 960])
        clock[0] = start + (at + 960) / FS
        if hs._tx_pending is not old:
            return 'stream', clock[0] - start
        due = hs.intermediate_query_due_in()
        if due is not None and due <= 0:
            queried = hs.query_intermediate_answer()
            if hs._tx_pending is not old:
                return 'before query', clock[0] - start
            if queried:
                return 'query', clock[0] - start
    return None, None


@pytest.mark.parametrize('name', MISSED)
def test_a_continue_the_hard_readers_lost_is_scored_whole_in_the_stream(monkeypatch, name):
    hs, io, clock = station(monkeypatch)
    old = hs._tx_pending
    how, at = drive(hs, clock, clip(name), 2.0)
    assert how == 'stream' and at < 0.8, (how, at, io.msgs)
    assert hs._tx_pending[1] == old[1] + 1 and hs._intermediate_query_attempts == 0
    assert any('scored whole at frame 0' in m for m in io.msgs), io.msgs


def test_the_continue_read_live_is_still_read_in_the_stream(monkeypatch):
    hs, io, clock = station(monkeypatch)
    how, at = drive(hs, clock, clip('over4-continue-read-live'), 2.0)
    assert how == 'stream' and at < 1.0, (how, at)
    assert not any('scored whole' in m for m in io.msgs)


def test_a_query_answer_with_an_unclear_last_symbol_is_read_before_the_next(monkeypatch):
    hs, io, clock = station(monkeypatch)
    old = hs._tx_pending
    hs._retry_data_over()
    assert hs._intermediate_query_attempts == 1
    how, at = drive(hs, clock, clip('query1-answer'), 4.8)
    assert how == 'before query' and at >= 4.5 - 1e-6, (how, at, io.msgs)
    assert hs._tx_pending[1] == old[1] + 1
    assert not any(m.startswith('intermediate query 2/') for m in io.msgs), io.msgs
    assert any('session-intermediate-query-answer (288/1)' in m for m in io.msgs), io.msgs


def test_the_stream_alone_still_refuses_that_query_answer(monkeypatch):
    hs, _, _ = station(monkeypatch)
    hs._retry_data_over()
    x = clip('query1-answer')
    assert not hs._peer_intermediate_query_answer(x)
    assert hs._peer_intermediate_query_answer(x, whole=True)


@pytest.mark.parametrize('name', MISSED + ['query1-answer'])
def test_another_links_tails_read_nothing(monkeypatch, name):
    hs, io, clock = station(monkeypatch, called='N0XYZ')
    if name.startswith('query'):
        hs._retry_data_over()
    how, _ = drive(hs, clock, clip(name), 4.8 if name.startswith('query') else 2.0)
    assert how == 'query', io.msgs


def test_the_links_nak_is_not_read_as_its_continue(monkeypatch):
    hs, _, _ = station(monkeypatch)
    nak = MK.synth_tone_pairs(VF.nak(hs.called, hs.bw)[1])
    x = np.concatenate([np.zeros(FS // 10), nak, np.zeros(FS)])
    assert VA._burst_match(x, VF.nak(hs.called, hs.bw)[1], hs._band) >= VA._BURST_MATCH
    assert not hs._held_continue(x)


def test_a_final_over_is_never_continued(monkeypatch):
    hs, _, _ = station(monkeypatch)
    hs._txq.clear()
    assert not hs._held_continue(clip('over3-continue'))


@corpora.requires_regress_fixtures
def test_nothing_in_the_regression_corpus_is_read_as_the_continue(monkeypatch):
    """The false-accept side at this link's grid; the sweep behind the floor
    ran every link's tails at nine grids  [vara_arq, _held_continue]."""
    hs, _, _ = station(monkeypatch)
    held = []
    for path in sorted(corpora.REGRESS_FIXTURES.glob('*.wav')):
        x = corpora.wav_mono(path)
        if hs._held_continue(x / (np.abs(x).max() + 1e-12)):
            held.append(path.name)
    assert not held
