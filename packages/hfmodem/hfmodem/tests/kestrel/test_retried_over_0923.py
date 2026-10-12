# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VE3WLR, 2026-09-23: every answer to an over, a retry or a query, left unread.

Its transmitter loses the first two symbols of every burst, so the stream's hard
continue readers never read it: the continue is scored whole as soon as its span
has ended. A retried over is continued at frame 4 or 6, and the query behind it
answered at the same frame on the 62-draw lattice; the first query after a lost
over is answered with the 16-symbol over-NAK.
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
from .test_pre_query_read_0923 import drive

FS = MK.FS
FIXTURES = Path(__file__).with_name('fixtures') / 'retried-over-0923'
CONTINUES = [('over2-continue', False, 0), ('retry3-continue-frame4', True, 4),
             ('retry5-continue-frame6', True, 6)]


def clip(name):
    path = FIXTURES / f'{name}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == FS and x.dtype == np.int16
    return x.astype(float) / 32768


def station(monkeypatch, *, retried=False, called=None):
    """A caller with a two-over delivery pending; ``retried`` leaves it where the
    over-NAK to its first query does: the over re-sent at record 101."""
    prov = json.loads(require_input(FIXTURES / 'provenance.json').read_text())
    clock = [1000.0]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: clock[0])
    hs, io = _connected(bw=prov['bw'])
    hs.called = called or prov['called']
    hs._peer_offset = prov['offset_hz'] / (FS / MK.NFFT)
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'end')
    if retried:
        hs._retry_data_over()
        hs._retry_data_over(peer_asked=True, adapt=True)
        assert hs._tx_retries == 1 and hs._tx_pending[2] == 101
    return hs, io, clock


def continue_end(hs, x, frame):
    """Seconds into ``x`` at which the link's continue ``frame`` ends."""
    mag = VA._band_power(x, hs._band, hs._grid)
    _, at = VA._frame_fit(mag, hs._peer_continue_tails()[frame], hs._band)
    return (at + VF.OVER_CONTINUE_NSYM * MK.HOP) / FS


@pytest.mark.parametrize('name,retried,frame', CONTINUES)
def test_a_head_cut_continue_keys_the_next_over_behind_its_last_symbol(
        monkeypatch, name, retried, frame):
    hs, io, clock = station(monkeypatch, retried=retried)
    old, x = hs._tx_pending, clip(name)
    end = continue_end(hs, x, frame)
    how, at = drive(hs, clock, x, 2.0)
    assert how == 'stream' and 0 <= at - end <= 0.6, (how, at, end, io.msgs)
    assert hs._tx_pending[1] == old[1] + 1 and hs._intermediate_query_attempts == 0
    assert any(f'scored whole at frame {frame}' in m for m in io.msgs), io.msgs


@pytest.mark.parametrize('name', ['retry3-continue-frame4', 'retry5-continue-frame6'])
def test_a_level_step_frame_continues_any_over(monkeypatch, name):
    """Frames 4 and 6 are the receiver's level steps, keyed by stock pairs to
    overs never retried: 6 behind a level-2 over at BW500 and BW2300, and 4
    behind our record-1 over at BW500 on the cables  [vara_arq._LEVEL_STEP]."""
    hs, io, clock = station(monkeypatch)
    how, _ = drive(hs, clock, clip(name), 2.0)
    assert how == 'stream', io.msgs


def test_the_short_over_nak_to_the_first_query_resends_the_over(monkeypatch):
    hs, io, clock = station(monkeypatch)
    hs._retry_data_over()
    how, at = drive(hs, clock, clip('query1-answer-289-541'), 4.8)
    assert how == 'stream' and at < 1.0, (how, at, io.msgs)
    assert hs._tx_retries == 1 and hs._tx_pending[2] == 101
    assert any(m.startswith('rx responder NAK from VE3WLR') for m in io.msgs), io.msgs
    assert not any(m.startswith('intermediate query 2/') for m in io.msgs), io.msgs


def test_the_short_over_nak_is_read_only_as_a_query_answer(monkeypatch):
    hs, _, _ = station(monkeypatch)
    assert not hs._peer_responder_nak(clip('query1-answer-289-541'))


def test_the_retried_overs_query_answer_at_frame_6_is_read(monkeypatch):
    hs, io, clock = station(monkeypatch, retried=True)
    old = hs._tx_pending
    hs._retry_data_over()
    how, at = drive(hs, clock, clip('query3-answer-288-373'), 4.8)
    assert how == 'stream' and at < 1.6, (how, at, io.msgs)
    assert hs._tx_pending[1] == old[1] + 1
    assert any('session-retry-query-answer-b (288/373)' in m for m in io.msgs), io.msgs


@pytest.mark.parametrize('retried', [False, True])
def test_a_silent_peer_still_draws_the_query(monkeypatch, retried):
    hs, io, clock = station(monkeypatch, retried=retried)
    how, at = drive(hs, clock, clip('over3-silent'), 2.0)
    assert how == 'query' and at >= 1.8 - 1e-6, (how, at, io.msgs)


@pytest.mark.parametrize('name', [c[0] for c in CONTINUES]
                         + ['query1-answer-289-541', 'query3-answer-288-373'])
def test_another_links_tails_read_nothing(monkeypatch, name):
    hs, io, clock = station(monkeypatch, retried='retry' in name or 'query3' in name,
                            called='N0XYZ')
    if name.startswith('query'):
        hs._retry_data_over()
    how, _ = drive(hs, clock, clip(name), 4.8 if name.startswith('query') else 2.0)
    assert how == 'query', io.msgs


def test_a_continue_is_not_taken_before_its_last_symbol(monkeypatch):
    hs, _, _ = station(monkeypatch, retried=True)
    burst = MK.synth_tone_pairs(hs._peer_continue_tails()[6])
    lead = np.zeros(FS // 10)
    assert hs._held_continue(np.concatenate([lead, burst, np.zeros(MK.HOP)]))
    assert not hs._held_continue(np.concatenate([lead, burst[:-MK.HOP // 2]]))


def test_the_links_nak_is_not_read_as_a_retried_continue(monkeypatch):
    hs, _, _ = station(monkeypatch, retried=True)
    nak = MK.synth_tone_pairs(VF.nak(hs.called, hs.bw)[1])
    assert not hs._held_continue(np.concatenate([np.zeros(FS // 10), nak, np.zeros(FS)]))


@corpora.requires_regress_fixtures
def test_nothing_in_the_regression_corpus_is_read_as_a_retried_answer(monkeypatch):
    retried, _, _ = station(monkeypatch, retried=True)
    queried, _, _ = station(monkeypatch)
    queried._retry_data_over()
    held = []
    for path in sorted(corpora.REGRESS_FIXTURES.glob('*.wav')):
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() + 1e-12)
        if retried._held_continue(x) or queried._short_over_nak_at(x, None) is not None:
            held.append(path.name)
    assert not held
