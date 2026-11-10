# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Measure solicited query ACKs against off-air audio and other responder frames."""
from collections import Counter
import json

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_request_by_stream import _recordings


_CALLS = ('N0LCR-1', 'K7ABC', 'N0CALL')
_CORPUS_MAX = {'500': (8, 9, 8, 7, 8, 9),
               '2300': (6, 7, 7, 6, 6, 6),
               '2750': (5, 6, 6, 6, 6, 6)}
_CROSS_MAX = {'500': 8, '2300': 6, '2750': 5}
_CROSS_ANSWER_MAX = {'500': 8, '2300': 5, '2750': 5}
_ANSWERS = tuple(dict.fromkeys((*VA._QUERY_ANSWERS.values(),
                              VF.SESSION_INTERMEDIATE_QUERY_ANSWER,
                              VF.SESSION_RETRY_QUERY_ANSWER,
                              VF.SESSION_RETRY_QUERY_ANSWER_B,
                              VF.SESSION_RESPONDER_OVER_ANSWER)))
_OTHERS = tuple(dict.fromkeys(
    k for k in vars(VF).values() if isinstance(k, VF.BurstKind)
    and k.n_payload == 31 and k.seed_off in (288, 1078) and k not in _ANSWERS))


def _counts(heard, tones):
    clear = heard[1:] >= 0
    exact = int(((heard[1:] == tones[1:]) & clear).sum())
    return int(clear.sum()), exact, int(clear.sum()) - exact


def _distribution(label, hist):
    print(f'{label}: ' + json.dumps({','.join(map(str, k)): v
                                    for k, v in sorted(hist.items())}), flush=True)


@corpora.requires_onair_connect_attempt
@corpora.requires_regress_fixtures
@corpora.requires_carrier_only_channel
@corpora.requires_monitored_quiet
@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_off_air_query_answers_stay_below_delivery_credit_floor(bw):
    from hfmodem.tests.kestrel.test_response_by_stream import _load_48k

    # 4.5 s receive brackets, overlapping by 2.5 s: every complete 32-symbol
    # alignment is inside at least one bracket. Fit partial ends too, as live RX
    # does. All starts and bracket edges stay on the reader's 32-sample grid.
    width, step = int(4.5 * MK.FS), 2 * MK.FS
    hist = {k.preadv: Counter() for k in _ANSWERS}
    seconds, windows, alignments = 0.0, 0, 0
    best = 0
    for label, path, own in _recordings():
        x = _load_48k(path)
        seconds += len(x) / MK.FS
        track = VA._top3_track(x, band=MK.band_for(bw))
        local = 0
        for start in range(0, len(x), step):
            end = min(start + width, len(x))
            # Only FFT windows fully inside this bracket belong to its track.
            a = start // VA._ACK_GRID
            b = a + max(0, (end - start - MK.NFFT) // VA._ACK_GRID + 1)
            part = tuple(t[a:b] for t in track)
            for call in (c for c in _CALLS if c not in own):
                for template in _ANSWERS:
                    kind = VF.for_bw(template, bw)
                    tones = np.asarray(VF.handshake_tones(call, kind))
                    heard, _ = VA._payload_fit(x[start:end], kind, tones, part,
                                               MK.band_for(bw))
                    counts = _counts(heard, tones)
                    hist[template.preadv][counts] += 1
                    local = max(local, counts[1])
                    windows += 1
                    alignments += ((end - start - MK.STRIDE + 31 * MK.HOP)
                                   // VA._ACK_GRID + 1) * len(VA._RESP_SHIFTS)
        best = max(best, local)
        print(f'query ACK BW{bw}: {label}: best exact={local}', flush=True)
    for (preadv, h), measured in zip(hist.items(), _CORPUS_MAX[bw]):
        _distribution(f'query ACK BW{bw} 288/{preadv} (clear,exact,wrong)', h)
        assert max(c[1] for c in h) <= measured, (preadv, h)
    print(f'query ACK BW{bw}: {seconds:.3f} s, {windows} window-template fits, '
          f'{alignments} alignment-shifts, best exact={best}', flush=True)
    assert seconds > 2300, f'corpus shrank to {seconds} s'
    assert windows > 20000, windows
    assert alignments > 100_000_000, alignments
    # Any alignment with >=15 exact tones also clears _payload_fit's >=13
    # priority tier. Such an alignment cannot hide behind a weaker clean sweep
    # at the best fit, even when its last tone or tail is missing.
    assert best < 15


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_other_responder_frames_stay_below_query_answer_floor(bw):
    hist = Counter()
    worst = {}
    for call in _CALLS:
        for other in _OTHERS:
            kind = VF.for_bw(other, bw)
            # Caller-keyed turn frames are deliberately keyed to the same call:
            # the most dangerous case, without help from a different callsign.
            clean = MK.synth_burst(call, kind)
            for snr in (None, 10, 5, 0, -5, -10, -15):
                for seed in range(4 if snr is not None else 1):
                    x = np.pad(clean, (4800, 4800))
                    if snr is not None:
                        # SNR in the conventional 2500 Hz reference bandwidth.
                        sigma = np.sqrt(np.mean(clean ** 2) * MK.FS / 5000
                                        * 10 ** (-snr / 10))
                        x = x + np.random.default_rng(seed).normal(0, sigma, len(x))
                    track = VA._top3_track(x, band=MK.band_for(bw))
                    for template in _ANSWERS:
                        answer = VF.for_bw(template, bw)
                        tones = np.asarray(VF.handshake_tones(call, answer))
                        heard, _ = VA._payload_fit(x, answer, tones, track,
                                                   MK.band_for(bw))
                        counts = _counts(heard, tones)
                        hist[counts] += 1
                        key = f'{other.seed_off}/{other.preadv}->{template.preadv}'
                        worst[key] = max(worst.get(key, 0), counts[1])
    _distribution(f'cross-frame BW{bw} (clear,exact,wrong)', hist)
    print(f'cross-frame BW{bw} exact maxima: {json.dumps(worst, sort_keys=True)}',
          flush=True)
    assert sum(hist.values()) == 4050, 'cross-frame population changed'
    assert max(worst.values()) <= _CROSS_MAX[bw], worst
    assert max(worst.values()) < 15, worst


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_unequal_answer_kinds_stay_below_query_answer_floor(bw):
    hist = Counter()
    worst = {}
    for call in _CALLS:
        for other in _ANSWERS:
            kind = VF.for_bw(other, bw)
            clean = MK.synth_burst(call, kind)
            for snr in (None, 10, 5, 0, -5, -10, -15):
                for seed in range(4 if snr is not None else 1):
                    x = np.pad(clean, (4800, 4800))
                    if snr is not None:
                        sigma = np.sqrt(np.mean(clean ** 2) * MK.FS / 5000
                                        * 10 ** (-snr / 10))
                        x = x + np.random.default_rng(seed).normal(0, sigma, len(x))
                    track = VA._top3_track(x, band=MK.band_for(bw))
                    for template in _ANSWERS:
                        if template == other:
                            continue
                        answer = VF.for_bw(template, bw)
                        tones = np.asarray(VF.handshake_tones(call, answer))
                        heard, _ = VA._payload_fit(x, answer, tones, track,
                                                   MK.band_for(bw))
                        counts = _counts(heard, tones)
                        hist[counts] += 1
                        key = f'{other.preadv}->{template.preadv}'
                        worst[key] = max(worst.get(key, 0), counts[1])
    _distribution(f'cross-answer BW{bw} (clear,exact,wrong)', hist)
    print(f'cross-answer BW{bw} exact maxima: {json.dumps(worst, sort_keys=True)}',
          flush=True)
    assert sum(hist.values()) == 2250, 'cross-answer population changed'
    assert max(worst.values()) <= _CROSS_ANSWER_MAX[bw], worst
    assert max(worst.values()) < 15, worst
