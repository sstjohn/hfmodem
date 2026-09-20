# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW2300 base training preamble — bin law + byte-exact audio reproduction.

kestrel's preamble generator (spec/01 §2300 "gDC REFERENCE PREAMBLE": VB6 Rnd bin
law, one continuous stream over the session) must reproduce the spec's ground-truth
bin table, and, prepended to the base burst, must make kestrel's transmit audio
match the real captured VARA over
`analysis/caps/bw2300/AAAA1-BBBB2-bw2300-counter512__a2b.wav` — the last time-domain
residual of the base level (0.985 without it).
"""
from __future__ import annotations

import numpy as np

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.tests.kestrel.corpora import BW2300_CAPTURE, requires_bw2300_capture, wav_mono
from hfmodem.kestrel.tx import varahf2300_tx as tx

# spec/01 §2300 ground truth: the 12 preamble bins of the 7 overs of the staged
# captures (audio-measured, both captures identical).
_GROUND_TRUTH = [
    [14, 24, 10, 21, 23, 21, 22, 24, 15, 24, 23, 15],
    [17, 16, 9, 16, 12, 9, 12, 22, 18, 21, 19, 14],
    [12, 21, 19, 20, 20, 22, 22, 10, 18, 19, 13, 18],
    [9, 16, 11, 17, 10, 19, 10, 9, 14, 9, 9, 16],
    [12, 22, 23, 15, 10, 18, 11, 22, 20, 19, 9, 18],
    [10, 12, 17, 24, 20, 22, 23, 17, 12, 23, 19, 19],
    [18, 10, 24, 13, 16, 22, 18, 14, 14, 18, 22, 10],
]


def test_preamble_bins_match_ground_truth():
    for over, exp in enumerate(_GROUND_TRUTH):
        assert tx.over_preamble_bins(over) == exp, f"over {over} bins differ"


def test_preamble_bins_in_band():
    r = rx.RECORDS[rx.BASE_LEVEL]
    for over in range(32):
        bins = tx.over_preamble_bins(over)
        assert len(bins) == 12
        assert all(r.first_bin <= b < r.first_bin + r.span for b in bins)


def test_preamble_is_not_payload():
    """The RX decodes the frame from the data columns alone, with or without it."""
    pl = bytes(range(rx.payload_bytes(rx.BASE_LEVEL)))
    for over in (None, 0, 3):
        fr = rx.decode_burst(tx.synth_burst(pl, over=over), rx.BASE_LEVEL)
        assert fr.crc_ok and fr.payload == pl


def _over_alignment(audio, s, e):
    """(data-column-1 sample index, decoded frame) for one captured over."""
    for onset, g, mag in rx._alignments(np.asarray(audio[s:e], float),
                                        rx._GUARD_TRIES):
        fr = rx.check_frame(rx.onair_to_frame(rx._onair_llr(mag[g:], rx.BASE_LEVEL),
                                              rx.BASE_LEVEL), rx.BASE_LEVEL)
        if fr.crc_ok:
            return s + onset + g * 512, fr
    return None, None


def _best_corr(x, audio, base):
    """Normalised correlation of ``x`` against ``audio`` at the best sample lag
    (the capture's over onset is not sample-locked to the block grid)."""
    best = -1.0
    for lag in range(-64, 65):
        y = audio[base + lag:base + lag + len(x)]
        if len(y) < len(x):
            continue
        best = max(best, float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))))
    return best


@requires_bw2300_capture
def test_preamble_reproduces_real_capture():
    audio = wav_mono(BW2300_CAPTURE)
    segs = rx.detect_overs(audio)
    assert len(segs) == len(_GROUND_TRUTH)
    for over, (s, e) in enumerate(segs):
        c1, fr = _over_alignment(audio, s, e)
        assert fr is not None, f"over {over} did not decode"
        pre = tx.synth_preamble(tx.over_preamble_bins(over))[rx._BASE_LEADIN * 512:]
        data = tx.synth_frame(fr.frame_bytes, over=None)
        start = c1 - len(pre)
        with_pre = _best_corr(np.concatenate([pre, data]), audio, start)
        without = _best_corr(np.concatenate([np.zeros(len(pre)), data]), audio, start)
        assert without < 0.99, f"over {over}: preamble-free baseline unexpectedly high"
        assert with_pre > 0.9999, f"over {over}: correlation {with_pre:.5f}"


if __name__ == "__main__":
    audio = wav_mono(BW2300_CAPTURE)
    for over, (s, e) in enumerate(rx.detect_overs(audio)):
        c1, fr = _over_alignment(audio, s, e)
        pre = tx.synth_preamble(tx.over_preamble_bins(over))[rx._BASE_LEADIN * 512:]
        data = tx.synth_frame(fr.frame_bytes, over=None)
        st = c1 - len(pre)
        print(f"over {over}: bins_ok={tx.over_preamble_bins(over) == _GROUND_TRUTH[over]} "
              f"corr without={_best_corr(np.concatenate([np.zeros(len(pre)), data]), audio, st):.4f} "
              f"with={_best_corr(np.concatenate([pre, data]), audio, st):.6f}")
