# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.shrike import onair, p4oracle, p4robusttx
from hfmodem.shrike.p4receiver import Emission, Kind, P4Receiver, Role
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx
from hfmodem.tests.shrike.test_ws8eoc_p4_silent_0927 import Live

ROOT = Path(__file__).resolve().parents[5]


def capture():
    path = ROOT / 'captures/onair-0929-0119'
    if not (path / 'stream.flac').exists():
        pytest.skip('local WS8EOC September 29 0119 capture absent')
    return onair.session.load_wav(str(path / 'stream.flac'), FS), json.loads((path / 'p4-probe.json').read_text())


@pytest.mark.parametrize('index', [0, 1])
def test_recorded_early_answer_head_stays_unnamed(index):
    audio, meta = capture()
    emission = meta['emissions'][index]
    at, origin = emission['audio_start'], emission['audio_end']
    end = at+round(1.138*FS)
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.enable_hold()
    receiver.expected_words = {'X27', 'Y27', 'Z27'}
    receiver.note_emission(Emission(Kind.SHORT, Role.ISS, at, (origin-at)/FS))
    record, = receiver.read_partial(audio[origin:end], origin, deadline=end)
    assert record.responds_to == at
    word = record.observation
    # S5(b): retain the acquired head, but withhold its name until 300 ms.
    assert word.word is None and word.provisional and word.symbols_read == 20
    assert receiver.cfo_hz is None and receiver.clock.cfo_hz == 0.
    assert word.score > .97 and abs(word.cfo_hz+7.34) < 1
    assert abs(word.stream_sample-(1247970 if index == 0 else 1307958)) < 3


@pytest.mark.parametrize('cfo', [-18., -7.5, 7.5, 18.])
def test_unknown_carrier_acquisition_stays_in_the_placed_head(cfo):
    signal = p4robusttx.robust_word('Y27')
    shifted = np.real(hilbert(signal)*np.exp(2j*np.pi*cfo*np.arange(len(signal))/FS))
    onset = .2
    audio = np.r_[np.zeros(round(onset*FS)), shifted]
    end = round((onset+.215)*FS)
    word = p4oracle.word_at(audio[:end], FS, 0, onset, cfo_hz=None,
                           provisional=True, expected_words={'Y27'})
    assert word is not None and word.word == 'Y27' and word.provisional
    assert abs(word.cfo_hz-cfo) < 1
    assert p4oracle.word_at(audio[:end], FS, 0, onset-.10, cfo_hz=None,
                           provisional=True, expected_words={'Y27'}) is None


def test_unknown_carrier_never_names_noise_or_a_carrier_grid_edge():
    rng = np.random.default_rng(929)
    for _ in range(20):
        audio = rng.standard_normal(round(.415*FS))
        assert p4oracle.word_at(audio, FS, 0, .2, cfo_hz=None,
                               provisional=True, expected_words={'Y27'}) is None
    signal = p4robusttx.robust_word('Y27')
    for cfo in (-20., 20.):
        shifted = np.real(hilbert(signal)*np.exp(2j*np.pi*cfo*np.arange(len(signal))/FS))
        audio = np.r_[np.zeros(round(.2*FS)), shifted][:round(.415*FS)]
        assert p4oracle.word_at(audio, FS, 0, .2, cfo_hz=None,
                               provisional=True, expected_words={'Y27'}) is None


def test_three_thousand_noise_inputs_never_acquire_a_carrier(monkeypatch):
    score, maxima, durations = p4oracle.p4words.placed_scores, [], []
    def ranked(symbols, **kwargs):
        fitted, scores = score(symbols, **kwargs)
        if kwargs.get('acquisition'):
            maxima.append(float(scores.max()))
        return fitted, scores
    monkeypatch.setattr(p4oracle.p4words, 'placed_scores', ranked)
    rng = np.random.default_rng(9290119)
    for _ in range(3000):
        audio = rng.standard_normal(round(.415*FS))
        started = perf_counter()
        found = p4oracle.word_at(audio, FS, 0, .2, cfo_hz=None,
                                provisional=True, expected_words={'Y27', 'X27', 'Z27'})
        durations.append(1000*(perf_counter()-started))
        assert found is None
    assert maxima and max(maxima) < p4oracle.p4words.MATCH_KNEE
    print(dict(inputs=3000, acquired=0, maximum_score=max(maxima),
               ranked_inputs=len(maxima), maximum_ms=max(durations),
               p99_ms=float(np.percentile(durations, 99))))


def probe(monkeypatch, audio, *, start=1202736, defer=False):
    live = Live(audio)
    live.pos = start-round(.5*FS)
    tx = Tx(live)
    tx.slot = 20
    tx.sessrx.host = tx.p4_link_host
    tx.p4_probe.emissions.clear()
    tx.p4_probe.values, tx.p4_probe.headers = (7, 38), ('A', 'A')
    tx.p4_probe.cadence, tx.p4_probe.packet = 'consecutive', 'robust'
    tx.p4_link_options.p4_link_timeout = 3.
    raster = onair._MasterGrid(start-20*60000, 60000, 0, packet_n=46080,
                              cs_n=12000, d_max_n=4800)
    aim = tx.aim
    def aim_slot(grid, slot):
        aim(grid, slot)
        return grid.boundary(slot)
    monkeypatch.setattr(tx, 'aim', aim_slot)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    reads, read = [], tx.sessrx.p4.read_partial
    def partial(audio, origin, *, deadline):
        reads.append((origin, len(audio), deadline))
        return [] if defer and len(reads) == 1 else read(audio, origin, deadline=deadline)
    monkeypatch.setattr(tx.sessrx.p4, 'read_partial', partial)
    return tx, live, raster, reads


def test_recorded_probe_cancels_v38_and_acks_the_valid_repeat(monkeypatch):
    audio, _ = capture()
    tx, live, raster, _ = probe(monkeypatch, audio)
    onair._run_p4_probe(tx, live, raster, in_slot=True)
    entry = [e for e in tx.p4_probe.emissions if 'proposed_at' not in e]
    assert [e['value'] for e in entry] == [7]
    keys = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert keys and keys[0]['kind'] == Kind.WORD and keys[0]['word'] == 'X27'
    assert abs(keys[0]['at']-(1307958+round(.920*FS))) < 4
    assert not any(e['event'] == 'Refused deadline' for e in tx.p4_probe.link_events)


def test_probe_reads_the_captured_tail_before_admitting_v38(monkeypatch):
    start = 2*FS
    audio = np.zeros(6*FS)
    word = p4robusttx.robust_word('Y27')
    at = start+round(.923*FS)
    audio[at:at+len(word)] = word
    tx, live, raster, reads = probe(monkeypatch, audio, start=start, defer=True)
    handoffs = []
    def link(tx, live, raster):
        handoffs.append((live.pos, tx.p4_probe.handoff_audio))
        return 'handoff verified'
    monkeypatch.setattr(onair, '_run_p4_link', link)
    assert onair._run_p4_probe(tx, live, raster, in_slot=True) == 'handoff verified'
    assert [e['value'] for e in tx.p4_probe.emissions] == [7]
    assert len(reads) == 2 and reads[1][0] == reads[0][0]
    assert reads[1][1] > reads[0][1]
    assert len(handoffs) == 1 and handoffs[0][0] < start+60000
    origin, window = handoffs[0][1]
    assert origin == tx.p4_probe.emissions[0]['audio_end']
    assert len(window) == reads[1][1]
