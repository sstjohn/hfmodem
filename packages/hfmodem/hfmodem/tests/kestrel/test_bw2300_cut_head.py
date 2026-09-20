# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A wide over read from a window that opens after the burst started.

The station is deaf for 0.17-0.19 s after every keying — PTT is held 0.14-0.15 s
past the last sample and the rig's audio returns 0.03 s after PTT-off — and the
over-search buffer restarts empty at each keying [``vara_arq._key``,
``working/vara-evening-en63bc-0910/analysis/kc9`` §4]. A peer that keys into that
window is therefore first heard some columns into its burst, with nothing ahead of
it to search.

The alignment search reached no further back than the first sample it was given,
so the frame's column 0 was outside the window and the 24 reference columns fell
to chance: the 2026-09-11 KC9GHZ greetings, whole and CRC-clean off the tape,
score 5 and 6 of 24 from 128 ms and 85 ms in and read as an empty band. Standing
the record's own training columns in as silence ahead of the window is what
reaches them  [``analysis/wide-lock``].
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.tests.kestrel.corpora import (BW2300_CAPTURE, BW2750_CAPTURE,
                                           requires_bw2300_capture,
                                           requires_bw2750_capture)

#: Columns of burst the window opens past. 181 ms at the base symbol: past the
#: whole 14-column lead, and past the longest deaf window measured on air.
_CUT = 17
_SNR_DB = 20.0


def _payload(level: int) -> bytes:
    return bytes([0x81]) + bytes(range(1, rx.payload_bytes(level)))


def _from(audio: np.ndarray, at: int, level: int) -> rx.Frame2300:
    """Decode ``audio`` from ``at``, with nothing ahead of it and 0.3 s behind."""
    x = np.concatenate([np.asarray(audio[at:], float), np.zeros(int(0.3 * rx.FS))])
    return rx.decode_over(x, 0, len(x), level=level)


@pytest.mark.parametrize("bw", ["2300", "2750"])
def test_a_rendered_over_decodes_from_a_window_that_opens_mid_burst(bw):
    level = rx.BASE_LEVELS[bw]
    payload = _payload(level)
    audio = tx.synth_burst(payload, level)
    rng = np.random.default_rng(7)
    audio = audio + rng.normal(0.0, np.sqrt(np.mean(audio ** 2)
                                            / 10 ** (_SNR_DB / 10)), len(audio))
    fr = _from(audio, _CUT * rx.RECORDS[level].dw50, level)
    assert fr.crc_ok, f"BW{bw} over lost from {_CUT} columns in"
    assert fr.payload == payload


@requires_bw2300_capture
def test_a_captured_2300_over_decodes_from_a_window_that_opens_mid_burst():
    _captured(_wav(str(BW2300_CAPTURE)), "2300")


@requires_bw2750_capture
def test_a_captured_2750_over_decodes_from_a_window_that_opens_mid_burst():
    # The capture harness died before closing this file, so its RIFF data-size
    # field is 0 and a wav reader returns nothing; the samples are intact behind
    # the header  [see test_bw2750].
    a = np.fromfile(str(BW2750_CAPTURE), dtype=np.float32, offset=44).astype(float)
    _captured(a / np.abs(a).max(), "2750")


def _wav(path: str) -> np.ndarray:
    from scipy.io import wavfile
    _, a = wavfile.read(path)
    a = np.asarray(a, float)
    if a.ndim > 1:
        a = a[:, 0]
    return a / np.abs(a).max()


def _captured(a: np.ndarray, bw: str) -> None:
    """The first DATA over of a real VARA capture, read from ``_CUT`` columns in."""
    level = rx.BASE_LEVELS[bw]
    for s, e in rx.detect_overs(a):
        whole = rx.decode_over(a, s, e, level=level)
        if not whole.crc_ok:
            continue
        fr = _from(a[:e], s + _CUT * rx.RECORDS[level].dw50, level)
        assert fr.crc_ok, f"BW{bw} over at {s / rx.FS:.1f}s lost from {_CUT} in"
        assert fr.payload == whole.payload
        return
    pytest.fail(f"no CRC-clean BW{bw} over in the capture")
