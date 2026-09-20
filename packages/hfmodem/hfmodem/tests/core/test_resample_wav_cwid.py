# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The three extractions that landed without tests, and the facts each protects."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hfmodem.core import cwid, levels, rates, resample, wav

# --- resample: the edges only, and ARDOP's rate is normative ------------------


def test_the_card_is_48k_and_ardops_core_stays_12k():
    """besra's byte-exact cross-decode against ardopcf is the tightest constraint
    in this tree, and it holds at 12 kHz. Resampling happens at the sound card,
    never in the protocol."""
    n = 12000
    x = np.zeros(n, dtype=np.int16)
    up = resample.to_card(x, 12000)
    assert len(up) == n * 4
    assert len(resample.from_card(up, 12000)) == n


@pytest.mark.parametrize("rate", [12000, 48000])
def test_a_round_trip_preserves_length_exactly(rate):
    x = (np.random.default_rng(0).normal(0, 3000, rate)).astype(np.int16)
    back = resample.from_card(resample.to_card(x, rate), rate)
    assert len(back) == len(x)


def test_the_card_boundary_is_also_the_dtype_boundary():
    """int16 inside, float32 at the card. A 48 k lane changes representation but
    not rate, and picks up no resampler transient doing it."""
    x = (np.random.default_rng(1).normal(0, 3000, 4800)).astype(np.int16)
    out = resample.to_card(x, 48000)
    assert out.dtype == np.float32 and len(out) == len(x)
    assert np.allclose(out, x / 32768.0, atol=1e-4), (
        "a same-rate conversion introduced a filter transient")


def test_index_arithmetic_is_exact_not_approximate():
    """The card index is the single timebase; every lane's position must be an
    integer function of it, or the fan-out drifts against itself."""
    for n in (0, 1, 4799, 4800, 48000, 1_234_567):
        assert rates.to_native(n, 12000) == n * 12000 // 48000
        assert rates.to_native(n, 48000) == n
        assert isinstance(rates.to_native(n, 12000), int)


# --- wav: never normalise on write -------------------------------------------


def test_a_hot_capture_is_still_hot_after_a_round_trip(tmp_path):
    """`session.write_wav` scaled to 0.8 peak, and a capture with 9.60% of its
    samples railed off the codec then read 0.00% from the file. The measurement
    you most need, destroyed by the writer."""
    hot = np.clip(np.random.default_rng(2).normal(0, 1.4, 48000), -1, 1)
    before = levels.railed_ppm(hot)
    assert before > 0, "the fixture is not actually hot"

    p = tmp_path / "hot.wav"
    wav.write(p, hot, 48000)
    after = levels.railed_ppm(wav.read(p))
    assert after == pytest.approx(before, rel=0.02), (
        f"railed went {before:.0f} -> {after:.0f} ppm across the writer")


def test_the_level_sidecar_records_the_pre_transform_truth(tmp_path):
    p = tmp_path / "cap.wav"
    hot = np.clip(np.random.default_rng(3).normal(0, 1.4, 24000), -1, 1)
    wav.write(p, hot, 48000)
    side = p.with_suffix(".json")
    assert side.exists(), "no sidecar — the pre-transform levels are unrecoverable"
    meta = json.loads(side.read_text())
    assert meta["railed_pct"] > 0
    assert meta["peak_dbfs"] == pytest.approx(levels.peak_dbfs(hot), abs=0.1)
    assert meta["normalised_on_write"] is False


def test_digital_silence_records_null_rather_than_a_plausible_floor(tmp_path):
    """-inf dBFS is a real reading. A floor like -120 would be a plausible-looking
    number for a measurement that was not made, and JSON cannot carry the honest
    one — so the sidecar says null."""
    p = tmp_path / "silence.wav"
    wav.write(p, np.zeros(4800), 48000)
    meta = json.loads(p.with_suffix(".json").read_text())
    assert meta["peak_dbfs"] is None
    assert meta["rms_dbfs"] is None


def test_reading_a_missing_file_says_so(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        wav.read(tmp_path / "absent.wav")


# --- cwid: the speed cap is a refusal ----------------------------------------


def test_over_twenty_wpm_is_refused_not_clamped():
    """§97.119(b)(1). A caller asking for 25 has a wrong idea of what it may
    send; quietly sending 20 leaves it holding that idea."""
    cwid.keying("W1AW", wpm=20.0)
    for fast in (20.1, 25.0, 40.0):
        with pytest.raises(ValueError):
            cwid.keying("W1AW", wpm=fast)


def test_paris_is_the_timing_reference():
    """PARIS is 50 dit units; at 20 WPM one dit is 60/(50*20) = 60 ms."""
    assert cwid.dit_seconds(20.0) == pytest.approx(0.06, abs=1e-9)
    total = sum(d for _, d in cwid.keying("PARIS PARIS", wpm=20.0))
    assert total == pytest.approx(2 * 60 / 20, rel=0.15)


@pytest.mark.parametrize("text,code", [
    ("E", "."),
    ("SOS", "... --- ..."),
    ("W1AW", ".-- .---- .- .--"),
])
def test_a_callsign_renders_to_its_code(text, code):
    """An identification that is not the callsign is worse than none."""
    assert cwid.pattern(text) == code


def test_e_is_exactly_one_dit():
    keyed = [d for on, d in cwid.keying("E", wpm=20.0) if on]
    assert keyed == [pytest.approx(cwid.dit_seconds(20.0))]


def test_an_uncodeable_character_is_refused():
    """Better to fail than to transmit an identification that is not the
    callsign."""
    with pytest.raises((KeyError, ValueError)):
        cwid.keying("W1AW¡")


def test_keying_alternates_and_starts_with_a_mark():
    seq = cwid.keying("SOS", wpm=20.0)
    assert seq[0][0] is True
    assert all(a[0] != b[0] for a, b in zip(seq, seq[1:])), (
        "adjacent entries must alternate, or a gap has been merged into a mark")
    assert all(d > 0 for _, d in seq)
    assert math.isfinite(sum(d for _, d in seq))
