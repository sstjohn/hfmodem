# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the receive-level instrument must not get wrong before it runs at the radio.

This tree has twice paid for measurement code whose first run was in front of an
operator, so every number `tools/rx_levels.py` prints is pinned here against something
that knows the answer independently: a nonlinearity with a closed-form third-harmonic
level, a synthetic capture whose noise floor was set on purpose, and the station's own
recordings.

Four defects specifically:

  * **The quiet floor read as band noise.** The very quietest windows of a session
    recording are usually our own receiver muted by our own transmitter -- 16 of
    NS0A_2300's 108 seconds sit at a flat -45.0 dBFS, 22 dB below anything else in the
    file. A floor estimate that swallows those flatters the chain by the depth of the
    mute, and every dynamic-range figure computed from it is that much too generous.
  * **A measurement floor read as a clean result.** "No image found" and "no image
    above what this capture could have seen" are different findings, and comparing two
    floor readings compares nothing.
  * **A recommendation with no measurement behind it.** Short dynamic range with the
    floor in its window and nothing railing is a weak burst, not a chain fault; telling
    the operator to turn something down is the wrong answer to it.
  * **A harmonic attributed to the wrong end of the link.** The frequency test is the
    only thing that separates them, and it inverts if the offsets are compared the
    wrong way round.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora

RL = corpora.harness("rx_levels")
FS = 48000
_ROOT = corpora.ROOT


def _noise(seconds: float, dbfs: float, seed: int = 1) -> np.ndarray:
    """Passband-shaped noise at a known in-band level, over a small flat floor."""
    n = int(seconds * FS)
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1 / FS)
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * (((f > 400) & (f < 2700)) + 0.02), n)
    return x * (10 ** (dbfs / 20) / np.sqrt((x ** 2).mean()))


def _tone(seconds: float, hz: float, amp: float) -> np.ndarray:
    return amp * np.cos(2 * np.pi * hz * np.arange(int(seconds * FS)) / FS)


def _cubic_3f_dbc(amp: float, eps: float) -> float:
    """Third-harmonic level of ``x + eps*x**3`` on a tone of amplitude ``amp``.

    ``x**3`` of a cosine is ``(3/4)cos + (1/4)cos3``, so the harmonic is ``eps*A**3/4``
    against a fundamental grown to ``A*(1 + 3*eps*A**2/4)``.
    """
    return 20 * np.log10(eps * amp ** 2 / 4 / (1 + 3 * eps * amp ** 2 / 4))


# ------------------------------------------------------------------ harmonic images

def test_image_matches_the_closed_form_of_the_nonlinearity():
    x = _tone(8, 1500.0, 0.2) + _noise(8, -70.0)
    m = RL.image_dbc(x + 3.0 * x ** 3)
    assert m["harmonics"][3] == pytest.approx(_cubic_3f_dbc(0.2, 3.0), abs=1.0)


def test_a_pure_cubic_makes_no_fifth_harmonic():
    x = _tone(8, 1500.0, 0.2) + _noise(8, -70.0)
    assert RL.image_dbc(x + 3.0 * x ** 3)["harmonics"][5] is None


def test_third_order_products_fall_three_for_one():
    """A defining property of the measurement: 10 dB less drive, 20 dB better ratio."""
    def dbc(amp):
        x = _tone(8, 1500.0, amp) + _noise(8, -80.0)
        return RL.image_dbc(x + 3.0 * x ** 3)["harmonics"][3]

    assert dbc(0.2) - dbc(0.2 / np.sqrt(10)) == pytest.approx(20.0, abs=1.0)


def test_a_linear_chain_reports_a_floor_and_not_a_number():
    """"Nothing found" and "nothing above what this could see" are different findings."""
    m = RL.image_dbc(_tone(8, 1500.0, 0.2) + _noise(8, -60.0))
    assert m["image_dbc"] is None
    assert m["floor_dbc"] < -40.0


def test_band_noise_alone_is_not_a_carrier():
    m = RL.image_dbc(_noise(8, -35.0))
    assert m["tone_db"] < RL._MIN_TONE_DB
    assert m["image_dbc"] is None


def test_carrier_frequency_is_found_between_bins():
    c = RL.carrier(_tone(8, 1502.67, 0.2) + _noise(8, -60.0))
    assert c["hz"] == pytest.approx(1502.67, abs=0.5)
    assert c["excess_db"] > 30.0


# ------------------------------------------------------------------ harmonic origin

@pytest.mark.parametrize("offset", [2.67, -4.0])
def test_a_harmonic_made_after_our_translation_scales_the_offset(offset):
    x = _tone(10, 1500.0 + offset, 0.2)
    r = RL.harmonic_origin(x + 3.0 * x ** 3 + _noise(10, -70.0), tone_hz=1500.0)
    assert r["b_hz"] == pytest.approx(3 * offset, abs=0.6)
    assert r["sigma_near"] < 3.0 <= r["sigma_far"]
    assert "made HERE" in r["verdict"]


def test_a_harmonic_made_at_the_far_end_carries_the_offset_unchanged():
    x = _tone(10, 1502.67, 0.2) + 0.01 * _tone(10, 4502.67, 1.0)
    r = RL.harmonic_origin(x + _noise(10, -70.0), tone_hz=1500.0)
    assert r["b_hz"] == pytest.approx(2.67, abs=0.6)
    assert r["sigma_far"] < 3.0 <= r["sigma_near"]
    assert "FAR END" in r["verdict"]


def test_a_tone_on_nominal_cannot_separate_the_two():
    """With no offset both hypotheses predict the same frequency, and saying so is the
    only honest answer. Reported as a verdict, not as whichever fit happened to win."""
    x = _tone(10, 1500.0, 0.2)
    r = RL.harmonic_origin(x + 3.0 * x ** 3 + _noise(10, -70.0), tone_hz=1500.0)
    assert "inconclusive" in r["verdict"]


def test_no_harmonic_to_attribute_is_said_rather_than_guessed():
    r = RL.harmonic_origin(_tone(10, 1502.67, 0.2) + _noise(10, -60.0), tone_hz=1500.0)
    assert r["segments"] == 0
    assert "nothing to attribute" in r["verdict"]


def test_a_capture_too_short_to_measure_does_not_pretend():
    r = RL.harmonic_origin(_tone(0.1, 1502.67, 0.2), tone_hz=1500.0)
    assert r["a_hz"] is None and "too short" in r["verdict"]


# ------------------------------------------------------------------------- the floor

def test_band_noise_reads_the_level_it_was_built_at():
    lv = RL.levels(_noise(30, -35.0))
    assert lv["band_noise"] == pytest.approx(-35.0, abs=1.0)
    assert lv["muted_windows"] == 0


def test_a_muted_receiver_is_gated_out_of_the_floor_estimate():
    """The defect this whole distinction exists for: our own transmitter mutes our own
    receiver, and those windows are not the band's noise floor."""
    x = _noise(40, -30.0)
    for start in range(0, 40, 8):                      # 2 s of mute every 8 s
        x[start * FS:(start + 2) * FS] *= 10 ** (-40 / 20)
    lv = RL.levels(x)
    assert lv["band_noise"] == pytest.approx(-30.0, abs=1.5)
    assert lv["muted_windows"] >= 8
    assert lv["quiet_floor"] < lv["band_noise"] - RL.MUTE_DROP_DB


def test_rail_occupancy_is_counted():
    x = np.clip(_noise(10, -6.0) * 4, -1, 1)
    assert RL.levels(x)["railed_ppm"] > 0


@pytest.mark.skipif(not (_ROOT / "offair" / "NS0A_2300" / "rig_rx.wav").exists(),
                    reason="off-air gateway recording not present")
def test_the_reference_sessions_quiet_windows_are_not_its_band_noise():
    """The station's known-good session. Its quietest windows sit far below everything
    else in the file, which is a muted receiver; reading them as band noise is what
    makes an archive session look like it had 37 dB of range."""
    audio, fs = RL.read_wav(_ROOT / "offair" / "NS0A_2300" / "rig_rx.wav")
    lv = RL.levels(audio, fs)
    assert lv["muted_windows"] > 0
    assert lv["band_noise"] - lv["quiet_floor"] > 20.0


@pytest.mark.skipif(not corpora.CLEAR_CHANNEL.exists(),
                    reason="verified clear-channel capture not present")
def test_an_uninterrupted_capture_gates_nothing_out():
    audio, fs = RL.read_wav(corpora.CLEAR_CHANNEL)
    lv = RL.levels(audio, fs)
    assert lv["muted_windows"] == 0
    assert lv["band_noise"] == pytest.approx(lv["quiet_floor"], abs=1.0)


# -------------------------------------------------------------- verdicts and advice

def _dr(noise_dbfs, signal_peak_dbfs, seed=3):
    noise = _noise(20, noise_dbfs, seed)
    signal = _noise(20, noise_dbfs, seed + 1) + _tone(20, 1500.0, 10 ** (signal_peak_dbfs / 20))
    return RL.dynamic_range(noise, signal)


def test_a_good_working_point_is_left_alone():
    dr = _dr(-40.0, -8.0)
    assert dr["ok"]
    assert any("Working point is good" in f for f in RL.assess(dr))


def test_a_hot_floor_asks_for_gain_ahead_of_the_codec():
    findings = RL.assess(_dr(-20.0, -8.0))
    assert any("above the -30 dBFS window" in f and "DATA OUT" in f for f in findings)


def test_a_weak_burst_is_not_reported_as_a_chain_fault():
    """Short range with the floor in its window and nothing railing is a weak signal.
    Recommending a change here would be advice with no measurement behind it."""
    dr = _dr(-40.0, -30.0)
    assert not dr["ok"]
    findings = RL.assess(dr)
    assert any("this burst was simply weak" in f for f in findings)
    assert not any("DATA OUT" in f for f in findings)


def test_a_peak_inside_the_certified_settings_spread_is_noted_not_failed():
    """The working point in use holds real traffic at -4 to -8 dBFS peak with nothing
    railed. A criterion that calls -5 a fault fails the setting the station just
    certified, so it is reported as an observation and does not mask the pass."""
    dr = _dr(-40.0, -5.0)
    assert dr["ok"]
    findings = RL.assess(dr)
    assert any("Noted, not a fault" in f for f in findings)
    assert any("Working point is good" in f for f in findings)


def test_a_peak_with_no_headroom_left_does_ask_for_a_change():
    findings = RL.assess(_dr(-40.0, -1.0))
    assert any("from the rail" in f and "DATA OUT" in f for f in findings)


def test_railing_is_reported_before_anything_else_about_the_signal():
    noise = _noise(20, -40.0)
    signal = np.clip(_noise(20, -40.0, 4) + _tone(20, 1500.0, 1.4), -1, 1)
    findings = RL.assess(RL.dynamic_range(noise, signal))
    assert any("against the rail" in f for f in findings)


def test_dynamic_range_is_measured_against_the_noises_p99():
    """Not its median. On the 2026-07-26 recording band noise reached +6.7 dB over its
    own floor, which is what the gateway's reply had to be separated from."""
    dr = _dr(-40.0, -8.0)
    assert dr["noise"]["noise_p99"] > dr["noise"]["band_noise"]
    assert dr["range_db"] == pytest.approx(
        dr["signal"]["peak_dbfs"] - dr["noise"]["noise_p99"], abs=1e-9)


@pytest.mark.parametrize("band_noise,fires", [(-35.0, False), (-43.2, True)])
def test_the_deafness_guards_are_read_against_the_measured_floor(band_noise, fires):
    """shrike's guard sits at -40 dBFS, and the station's own known-good reference is
    quoted at -43.2. Whichever way tonight's measurement lands, the tool reports the
    consequence rather than leaving it to be discovered on the air."""
    shrike = [ln for ln in RL.guard_implications(band_noise) if "shrike" in ln][0]
    assert ("FIRES" in shrike) is fires


# ----------------------------------------------------------------------- the sweep

def test_the_sweep_recovers_the_same_image_at_every_frequency():
    plan = RL.sweep_plan(dwell=6.0, level_dbfs=-14.0)
    x = RL.sweep_signal(plan)
    report = RL.sweep_report(np.tanh(2.0 * x) / 2.0, plan)
    dbc = [r["image"]["image_dbc"] for r in report]
    assert all(v is not None for v in dbc)
    assert max(dbc) - min(dbc) < 1.0
    assert "flat with frequency" in RL.sweep_verdict(report)


def test_the_sweep_tolerates_a_recording_that_started_late():
    plan = RL.sweep_plan(dwell=6.0, level_dbfs=-14.0)
    x = RL.sweep_signal(plan)
    late = np.concatenate([np.zeros(int(1.3 * FS)), np.tanh(2.0 * x) / 2.0, np.zeros(FS)])
    report = RL.sweep_report(late, plan)
    assert all(r["seconds"] >= 3.0 for r in report)
    assert all(r["image"]["image_dbc"] is not None for r in report)


def test_a_linear_loopback_sweeps_clean():
    plan = RL.sweep_plan(dwell=6.0, level_dbfs=-14.0)
    x = RL.sweep_signal(plan) + _noise(len(RL.sweep_signal(plan)) / FS, -70.0)
    assert "linear at this level" in RL.sweep_verdict(RL.sweep_report(x, plan))


def test_the_sweep_holds_its_level_across_the_band():
    plan = RL.sweep_plan(dwell=2.0, level_dbfs=-20.0)
    x = RL.sweep_signal(plan)
    n = int(2.0 * FS)
    rms = [20 * np.log10(np.sqrt((x[i * n + FS // 2:(i + 1) * n - FS // 2] ** 2).mean()))
           for i in range(len(plan["hz"]))]
    assert max(rms) - min(rms) < 0.5


# -------------------------------------------------------------------------- the CLI

def test_read_wav_normalises_integer_and_float_files_alike(tmp_path):
    x = _noise(3, -20.0)
    RL.write_wav(tmp_path / "i.wav", x)
    got, fs = RL.read_wav(tmp_path / "i.wav")
    assert fs == FS
    assert RL.levels(got)["band_noise"] == pytest.approx(RL.levels(x)["band_noise"], abs=0.5)


def test_measure_runs_end_to_end_against_the_modelled_chain(tmp_path):
    """The guided flow itself, not just its parts: an operator who reaches this at the
    radio is the wrong place to find out that a step does not run."""
    rec = tmp_path / "rec.json"
    r = subprocess.run([sys.executable, str(corpora.TOOLS / "rx_levels.py"),
                        "--record", str(rec), "measure", "--simulate", "--skip-state",
                        "--seconds", "8"],
                       capture_output=True, text=True, timeout=300, cwd=corpora.REPO)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DYNAMIC RANGE" in r.stdout and "findings" in r.stdout
    assert "measure" in rec.read_text()


def test_sweep_writes_a_file_the_analyser_can_read_back(tmp_path):
    tool = str(corpora.TOOLS / "rx_levels.py")
    wav = tmp_path / "sweep.wav"
    w = subprocess.run([sys.executable, tool, "sweep", "--write", str(wav), "--dwell", "3"],
                       capture_output=True, text=True, timeout=300, cwd=corpora.REPO)
    assert w.returncode == 0, w.stderr[-2000:]
    x, fs = RL.read_wav(wav)
    RL.write_wav(wav, np.tanh(2.0 * x) / 2.0, fs)
    a = subprocess.run([sys.executable, tool, "sweep", "--analyse", str(wav), "--dwell", "3"],
                       capture_output=True, text=True, timeout=300, cwd=corpora.REPO)
    assert a.returncode == 0, a.stderr[-2000:]
    assert "Images present" in a.stdout
