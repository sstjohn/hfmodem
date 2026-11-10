# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Native VE3 changeover bodies must use the qualified head's measured CFO.

These are acquisition regressions, not callback/RadioTx deadline qualification.
The held WAV cuts contain only audio available to the corresponding live read.
"""

from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.shrike import p3acquire, rxfront


CAPTURE = Path(__file__).resolve().parents[5] / "captures/onair-0914-2101"


@pytest.fixture(scope="module")
def held_audio():
    if not CAPTURE.is_dir():
        pytest.skip(f"the arm's own session capture is not in this tree: {CAPTURE}")
    cuts = {}
    for hold, expected in ((7, 104438), (8, 59459)):
        with wave.open(str(CAPTURE / f"hold_{hold:02d}.wav"), "rb") as wav:
            assert wav.getframerate() == 48000
            assert wav.getsampwidth() == 2
            raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            mono = raw.reshape(-1, wav.getnchannels())[:, 0]
            cuts[hold] = mono.astype(np.float64) / 32768.0
        assert len(cuts[hold]) == expected
    return cuts


def test_native_h7_uses_refined_cfo_at_original_qualified_head(held_audio):
    candidate = p3acquire.changeover(held_audio[7])
    assert candidate is not None
    assert candidate.event.start == 4860
    assert candidate.coarse_hz == -25.0
    assert candidate.offset_hz == pytest.approx(-17.6, abs=0.05)
    assert candidate.quality >= 0.65
    assert candidate.event.packet == (1, 0, b"RMS", True)


def test_same_native_head_isolates_coarse_body_crc_failure(held_audio):
    audio = held_audio[7]
    candidate = p3acquire.changeover(audio)
    assert candidate is not None and candidate.event.packet is not None
    at = candidate.event.start
    coarse = rxfront._cs_event(
        p3acquire.compensate(audio, candidate.coarse_hz),
        2, 0, at, at / 48000, ", coarse counterfactual",
    )
    fine = rxfront._cs_event(
        p3acquire.compensate(audio, candidate.offset_hz),
        2, 0, at, at / 48000, ", measured CFO",
    )
    assert coarse.packet is None
    assert fine.packet == (1, 0, b"RMS", True)


def test_h8_retained_cfo_is_separate_from_default_cold_acquisition(held_audio):
    # The patch does not weaken cold header gates or silently invent retention.
    assert p3acquire.changeover(held_audio[8]) is None
    previous = p3acquire.changeover(held_audio[7])
    assert previous is not None and previous.event.packet is not None
    candidate = p3acquire.changeover(
        held_audio[8], offsets=(previous.offset_hz,),
    )
    assert candidate is not None
    assert candidate.event.packet == (1, 0, b"RMS", True)


def test_intact_control_with_erased_body_does_not_gain_crc(held_audio):
    audio = held_audio[7].copy()
    # Preserve the first complete CS3 head; erase its body and the next repeat.
    audio[4860 + round(0.24 * 48000):] = 0.0
    candidate = p3acquire.changeover(audio)
    assert candidate is None or candidate.event.packet is None


def test_truncated_control_cannot_promote_a_packet(held_audio):
    candidate = p3acquire.changeover(held_audio[7][:4860 + 4800])
    assert candidate is None or candidate.event.packet is None


def test_silence_has_no_changeover_candidate(held_audio):
    assert p3acquire.changeover(np.zeros_like(held_audio[8])) is None


def test_broadband_noise_has_no_changeover_candidate(held_audio):
    noise = np.random.default_rng(20260915).normal(0.0, 0.1, len(held_audio[8]))
    assert p3acquire.changeover(noise) is None
