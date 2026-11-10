# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The tracker on synthetic emissions whose answer is known by construction,
and on the four recordings the monitor design names as its done-when.

Two of those four are held to the design's words as strict xfails, because
what the design asked for is not what a power spectrum can deliver on them,
and the measured shortfall is written into each reason.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from hfmodem.monitor.tracker import BIN_HZ, FS, ROW_S, Tracker, cadence
from hfmodem.tests import evidence
from hfmodem.tests import monitor_evidence as E
from hfmodem.tests.kestrel import corpora

WS8EOC_TAP = (evidence.WORKING / "pactor-level-air-0916" / "arms"
              / "pactor-current-ws8eoc-80-force-20260917T143320Z" / "tap.wav")
#: Where the monitor design places WS8EOC's CW ID, from an ear and a spectrum.
WS8EOC_CW_HZ = 1605.5
K0SI_TAPES = E.K0SI
PACTOR3 = corpora.REGRESS_FIXTURES / "oracle_pactor3_dl6maa.wav"
LOCAL_P1 = corpora.REGRESS_FIXTURES / "pos_pactor1_local.wav"
CONTROLS = [corpora.REGRESS_FIXTURES / f for f in ("neg_noise_chatham.wav", "neg_ft8_local.wav")]


def _run(x: np.ndarray, lo_hz: float = 300.0, hi_hz: float = 2900.0, chunk: int = 4800):
    tr = Tracker(lo_hz, hi_hz)
    out = []
    for i in range(0, len(x), chunk):
        out += tr.push(i, x[i:i + chunk])
    return out + tr.flush()


def _ends(reps):
    return [r for r in reps if r.kind == "track-end"]


def _load(path) -> np.ndarray:
    if Path(path).parent == E.DIRECTORY:
        return E.samples(path) / 32768.0
    fs, a = wavfile.read(path)
    a = a.astype(float) / (32768.0 if a.dtype == np.int16 else 1.0)
    a = a[:, 0] if a.ndim > 1 else a
    if fs != FS:
        r = Fraction(FS, fs).limit_denominator(1000)
        a = resample_poly(a, r.numerator, r.denominator)
    return a


@lru_cache(maxsize=None)
def _tape(path: str):
    return _run(_load(path))


def _band_noise(n: int, lo: float, hi: float, rng) -> np.ndarray:
    x = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / FS)
    x[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(x, n)
    return x / x.std()


def _keyed(n: int, spans, ramp_s: float = 0.02) -> np.ndarray:
    g = np.zeros(n)
    for a, b in spans:
        g[int(a * FS):int(b * FS)] = 1.0
    k = np.hanning(int(ramp_s * FS))
    return np.convolve(g, k / k.sum(), "same")


# --------------------------------------------------------------------------- #
# Synthetic.
@pytest.mark.parametrize("tone_hz", [1512.0, 1505.9, 1517.0, 1500.3])
def test_a_tone_reports_its_centre_to_a_hertz_on_an_11_7_hz_grid(tone_hz):
    rng = np.random.default_rng(1)
    t = np.arange(20 * FS) / FS
    x = rng.normal(0, 0.01, len(t)) + 0.05 * np.sin(2 * np.pi * tone_hz * t) * (t > 2) * (t < 15)
    [end] = _ends(_run(x))
    assert abs(end.centre_hz - tone_hz) <= 1.0, end
    assert end.label == "CW-shaped" and end.bw_hz < 60


def test_a_minute_of_white_noise_raises_no_track():
    x = np.random.default_rng(1).normal(0, 0.01, 60 * FS)
    assert _run(x) == []


def test_two_interleaved_links_with_touching_skirts_stay_two_tracks():
    """Overs of 1 s, one link's 100 ms after the other's, 1000-1500 Hz and
    1500-2000 Hz: their -26 dB edges overlap by a bin."""
    rng = np.random.default_rng(2)
    n = 30 * FS
    a_on, b_on, t = [], [], 3.0
    while t < 26:
        a_on.append((t, t + 1.0))
        b_on.append((t + 1.1, t + 2.1))
        t += 2.2
    x = (rng.normal(0, 0.003, n)
         + 0.03 * _band_noise(n, 1000, 1500, rng) * _keyed(n, a_on)
         + 0.03 * _band_noise(n, 1500, 2000, rng) * _keyed(n, b_on))
    ends = sorted(_ends(_run(x)), key=lambda r: r.centre_hz)
    assert len(ends) == 2, ends
    low, high = ends
    assert abs(low.lo_hz - 1000) <= 2 * BIN_HZ and abs(low.hi_hz - 1500) <= 2 * BIN_HZ, low
    assert abs(high.lo_hz - 1500) <= 2 * BIN_HZ and abs(high.hi_hz - 2000) <= 2 * BIN_HZ, high
    assert low.hi_hz >= high.lo_hz, "the skirts were meant to touch"
    assert low.bursts == high.bursts == len(a_on)


def test_two_links_that_coincide_in_time_and_band_are_one_wide_unknown_track():
    rng = np.random.default_rng(2)
    n = 30 * FS
    on = _keyed(n, [(t, t + 1.0) for t in np.arange(3, 26, 2.2)])
    x = (rng.normal(0, 0.003, n) + 0.03 * on * _band_noise(n, 1000, 1500, rng)
         + 0.03 * on * _band_noise(n, 1300, 1800, rng))
    [end] = _ends(_run(x))
    assert end.label == "UNKNOWN"
    assert abs(end.lo_hz - 1000) <= 2 * BIN_HZ and abs(end.hi_hz - 1800) <= 2 * BIN_HZ, end


def test_cadence_is_the_arq_cycle_and_passes_the_null_test():
    rng = np.random.default_rng(3)
    t = np.arange(int(10 / ROW_S)) * ROW_S
    keyed = np.where((t % 1.25) < 0.96, 0.0, -20.0) + rng.normal(0, 2, len(t))
    assert cadence(keyed) == pytest.approx(1.25, abs=0.01)
    assert cadence(rng.normal(0, 2, len(t))) is None
    starts = np.cumsum(rng.exponential(1.25, 12))
    random = np.full(len(t), -20.0)
    for s in starts:
        random[(t >= s) & (t < s + 0.96)] = 0.0
    assert cadence(random + rng.normal(0, 2, len(t))) is None


def test_a_pactor_shaped_link_tracks_with_its_cycle():
    rng = np.random.default_rng(4)
    n = 30 * FS
    on = _keyed(n, [(t, t + 0.96) for t in np.arange(2, 29, 1.25)])
    x = rng.normal(0, 0.003, n) + 0.03 * on * _band_noise(n, 1300, 1700, rng)
    reps = _run(x)
    cads = [r.cadence_s for r in reps if r.kind == "track" and r.cadence_s]
    assert cads and all(abs(c - 1.25) <= 0.01 for c in cads), cads
    assert {r.label for r in reps if r.cadence_s} == {"FSK-shaped"}


def test_the_window_is_the_constructors_and_a_12_khz_span_finds_a_tone_at_8200_hz():
    rng = np.random.default_rng(5)
    t = np.arange(20 * FS) / FS
    x = rng.normal(0, 0.01, len(t)) + 0.05 * np.sin(2 * np.pi * 8203.1 * t) * (t > 2) * (t < 15)
    assert _ends(_run(x)) == []
    [end] = _ends(_run(x, 0.0, 12000.0))
    assert abs(end.centre_hz - 8203.1) <= 1.0, end


def test_a_receiver_mute_under_our_own_keying_raises_no_track():
    """2.1 s of every 8.3 s, 12 dB down with the PTT transitions at digital
    silence: the shape of the 2026-07-26 tape, and shallower than
    MUTE_DROP_DB. The band coming back must not read as an emission."""
    rng = np.random.default_rng(6)
    n = 60 * FS
    t = np.arange(n) / FS
    mute = (t % 8.3) > 6.2
    edge = np.convolve(np.diff(mute.astype(int), prepend=0) != 0, np.ones(7200), "same") > 0
    gain = np.where(edge, 0.0, np.where(mute, 10 ** (-12 / 20), 1.0))
    assert _run(rng.normal(0, 0.01, n) * gain) == []


def test_a_window_from_0_hz_leaves_the_dc_offset_out():
    """A Kiwi IQ span starts at 0 Hz. On dl6maa the input's DC offset and its
    Hann skirt outweighed the PACTOR link and the link's track collapsed to
    3 Hz."""
    rng = np.random.default_rng(8)
    t = np.arange(20 * FS) / FS
    x = 0.05 + rng.normal(0, 0.01, len(t)) + 0.05 * np.sin(2 * np.pi * 1512 * t) * (t > 2) * (t < 15)
    [end] = _ends(_run(x, 0.0, 12000.0))
    assert abs(end.centre_hz - 1512) <= 1.0, end


def test_a_window_to_24_khz_leaves_the_nyquist_bin_out():
    rng = np.random.default_rng(9)
    n = 20 * FS
    t = np.arange(n) / FS
    x = rng.normal(0, 0.001, n) + 0.05 * (-1.0) ** np.arange(n) * (t > 2) * (t < 15)
    assert [r for r in _run(x, 0.0, 24000.0) if r.hi_hz > 23900] == []


def test_level_is_rms_so_a_0_1_sine_reads_minus_23_dbfs():
    t = np.arange(20 * FS) / FS
    x = (0.1 * np.sin(2 * np.pi * 1512 * t) * (t > 2) * (t < 15)
         + np.random.default_rng(10).normal(0, 0.001, len(t)))
    [end] = _ends(_run(x))
    assert end.level_dbfs == pytest.approx(20 * np.log10(0.1 / np.sqrt(2)), abs=0.3)


def test_frames_are_the_stft_rows_on_the_grid():
    tr = Tracker()
    x = np.random.default_rng(7).normal(0, 0.01, 12 * FS)
    for i in range(0, len(x), 4800):
        tr.push(i, x[i:i + 4800])
    t, f, p = tr.frames()
    assert p.shape == (len(t), len(f)) and len(t) == int(10 / ROW_S)
    assert np.allclose(np.diff(t), ROW_S) and np.allclose(np.diff(f), BIN_HZ)
    assert 300 <= f[0] < 300 + BIN_HZ and 2900 - BIN_HZ < f[-1] <= 2900
    assert np.nanmedian(p) == pytest.approx(10 * np.log10(0.01 ** 2 / (FS / 2) * BIN_HZ), abs=3)


# --------------------------------------------------------------------------- #
# (a) The labelled 2026-07-26 VARA recording.
def _gate_bursts():
    monitor = pytest.importorskip("hfmodem.kestrel.monitor")
    x = _load(corpora.ONAIR_CONNECT_ATTEMPT)
    x = x / (np.abs(x).max() or 1.0)
    gate, out = monitor.MonitorGate(), []
    for i in range(0, len(x), 4800):
        for start, burst in gate.push(x[i:i + 4800]):
            ref = gate.reference()
            band = monitor.occupied_band_hz(burst, ref) if ref is not None else None
            if band is not None:
                out.append((start / FS, (start + len(burst)) / FS, band[0], band[1]))
    return out


@lru_cache(maxsize=1)
def _vara_tracks():
    return _ends(_tape(str(corpora.ONAIR_CONNECT_ATTEMPT))), _gate_bursts()


@corpora.requires_onair_connect_attempt
def test_every_burst_the_vara_gate_segments_lies_inside_a_track():
    tracks, bursts = _vara_tracks()
    assert len(bursts) >= 10
    for a, b, lo, hi in bursts:
        live = [r for r in tracks if r.t_start <= b and r.t >= a]
        cover = max((min(r.hi_hz, hi) - max(r.lo_hz, lo)) / (hi - lo) for r in live) if live else 0
        assert cover >= 0.8, f"gate burst {a:.2f}-{b:.2f} s, {lo:.0f}-{hi:.0f} Hz: {cover:.2f} covered"


@corpora.requires_onair_connect_attempt
@pytest.mark.parametrize("heard_from", [56.04, 64.32])
def test_kb9mmts_answer_as_our_mute_lifts_raises_a_track(heard_from):
    """KB9MMT answers our connect request as our receiver unmutes, some 9 dB
    under the band: one answer from 56.02 s, the other begun 0.34 s under our
    own mute and heard from its lifting at 64.32 s. Its fifteen tones hop
    across 797-2297 Hz, each standing a few dB over the floor, which is why an
    over of hops is gathered across the whole band."""
    tracks, _ = _vara_tracks()
    over = [r for r in tracks if abs(r.t_start - heard_from) <= 0.4
            and min(r.hi_hz, 2297) - max(r.lo_hz, 797) >= 500]
    assert over, [(r.t_start, r.lo_hz, r.hi_hz) for r in tracks if 50 < r.t_start < 70]


@corpora.requires_onair_connect_attempt
@pytest.mark.xfail(strict=True, reason=(
    "The gate reads -10 dB edges off one burst and a track holds the median of "
    "its overs' -26 dB edges; the BW500 station's single-tone MFSK overs reach "
    "different parts of 1113-1781 Hz over by over, and the 785-1102 Hz station "
    "keying beside it joins the gate's own edges (809-1699). The failure says "
    "how many share both edges."))
def test_every_burst_the_vara_gate_segments_has_its_edges_on_a_track_within_a_bin():
    tracks, bursts = _vara_tracks()
    missed = [(a, b, lo, hi) for a, b, lo, hi in bursts
              if not any(r.t_start <= b and r.t >= a and abs(r.lo_hz - lo) <= BIN_HZ + 0.1
                         and abs(r.hi_hz - hi) <= BIN_HZ + 0.1 for r in tracks)]
    assert not missed, (f"{len(bursts) - len(missed)} of {len(bursts)} gate bursts share "
                        f"both edges with a track to within a bin; missed {missed}")


@pytest.mark.skipif(not LOCAL_P1.exists(), reason=f"{LOCAL_P1} is not on this machine")
def test_a_tone_of_a_pactor_link_joins_the_links_track():
    """On pos_pactor1_local one tone, 1523-1640 Hz, rose above the rest of the
    1264-1733 Hz link for a while and held a track of its own inside the link's
    band, which kestrel then called a VARA long burst. It covers a quarter of
    the link's band, over PART_FRAC."""
    x = (np.clip(_load(LOCAL_P1), -1, 1) * 32767).astype(np.int16)
    ends = _ends(_run(x, 0.0, 24000.0))
    tone = [r for r in ends if r.t_start < 9 and r.lo_hz >= 1250 and r.hi_hz <= 1760
            and r.bw_hz < 200]
    assert tone == [], tone


# --------------------------------------------------------------------------- #
# (b) A PACTOR link's cycle, and none on the controls.
@pytest.mark.skipif(not PACTOR3.exists(), reason=f"{PACTOR3} is not on this machine")
def test_a_pactor_3_link_shows_its_1_25_s_cycle():
    """On the link's own band. Its skirts, 400-650 and 2370-2600 Hz, fold at
    2.5 s about as often, which is the cycle twice over."""
    reps = _tape(str(PACTOR3))
    link = [r for r in reps if r.lo_hz < 1300 and r.hi_hz > 1700 and r.cadence_s]
    cads = [r.cadence_s for r in link]
    assert len(cads) >= 10
    assert np.median(cads) == pytest.approx(1.25, abs=0.005)
    assert sum(abs(c - 1.25) <= 0.01 for c in cads) >= 0.8 * len(cads), sorted(cads)


@pytest.mark.skipif(not PACTOR3.exists(), reason=f"{PACTOR3} is not on this machine")
def test_the_upgraded_pactor_3_link_is_one_track_at_its_full_width():
    """DL6MAA never leaves a gap for 77 s, so a floor learned from its own band
    is the link; when it widened from about 600-2300 to 400-2500 Hz at 47 s
    only its new edges stood over that, as two narrow tracks at 410-633 and
    2367-2590 Hz, and nothing 2 kHz wide heard the upgraded link. The two key
    in step, and are one track at the link's full width until 65 s and more;
    the track the link opened on still ends at the upgrade."""
    ends = _ends(_tape(str(PACTOR3)))
    link = [r for r in ends if r.lo_hz < 700 and r.hi_hz > 2300 and r.t_start < 48.5
            and r.t > 65]
    assert link, [(r.t_start, r.t, r.lo_hz, r.hi_hz) for r in ends if r.active_s > 2]


@pytest.mark.skipif(not PACTOR3.exists(), reason=f"{PACTOR3} is not on this machine")
def test_no_track_is_narrower_than_two_bins():
    assert [r for r in _tape(str(PACTOR3)) if r.hi_hz <= r.lo_hz] == []


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_a_control_recording_shows_no_cadence(path):
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    assert [r for r in _tape(str(path)) if r.cadence_s] == []


# --------------------------------------------------------------------------- #
# (c) VARA-shaped channel overs on the K0SI-frequency tapes.
def _k0si_overs():
    return {Path(p): [r for r in _ends(_tape(p))
                           if r.label == "2k-multicarrier" and r.active_s >= 3]
            for p in K0SI_TAPES}


def test_every_2k_multicarrier_over_on_the_k0si_tapes_spans_varas_band():
    """The band, not the count: an over the tracker calls 2k-multicarrier on
    these tapes covers at least 80 % of 797-2297 Hz (measured 84-100 %). The
    eleven whole arm taps yield eight overs on five tapes. These are channel
    overs on K0SI's frequency, including occupants and some of our own TX;
    they are not eight attributable peer answers. The remaining six tapes
    still belong to the population, even though they yield no qualifying over.
    """
    overs = _k0si_overs()
    assert len(overs) == 11
    assert sum(bool(rs) for rs in overs.values()) == 5
    found = [r for rs in overs.values() for r in rs]
    assert len(found) == 8, len(found)
    assert all(min(r.hi_hz, 2297) - max(r.lo_hz, 797) >= 0.8 * 1500 for r in found), [
        (r.lo_hz, r.hi_hz) for r in found]


@pytest.mark.xfail(strict=True, reason=(
    "A 0.5 dB tilt across a 1500 Hz band moves a power-weighted centroid 14 Hz, "
    "and fading tilts every over. The half-bin offset is a property of the "
    "carrier grid, which kestrel's fractional-grid search measures and reports as "
    "offset_hz. The failure gives the centroids measured."))
def test_k0si_sits_12_hz_low_on_every_tape_to_3_hz():
    overs = _k0si_overs()
    centres = sorted(r.centre_hz for rs in overs.values() for r in rs)
    said = (f"{len(centres)} overs on {sum(bool(v) for v in overs.values())} tapes, "
            f"centroids {centres}")
    assert centres and max(centres) - min(centres) <= 3.0, said
    assert abs(np.median(centres) - 1488) <= 4, said


def test_arm_taps_with_the_same_filename_keep_each_tapes_overs(monkeypatch):
    from types import SimpleNamespace
    import sys

    tapes = [Path("first/tap.wav"), Path("second/tap.wav")]
    over = SimpleNamespace(kind="track-end", label="2k-multicarrier", active_s=4)
    monkeypatch.setattr(sys.modules[__name__], "K0SI_TAPES", tapes)
    monkeypatch.setattr(sys.modules[__name__], "_tape", lambda p: [over] if p == tapes[0] else [])
    overs = _k0si_overs()
    assert overs == {tapes[0]: [over], tapes[1]: []}


# --------------------------------------------------------------------------- #
# (d) WS8EOC's CW ID after our QRT on the 2026-09-17 80 m force arm.
@pytest.mark.skipif(not WS8EOC_TAP.exists(), reason=f"{WS8EOC_TAP} is not on this machine")
def test_ws8eoc_identifies_once_in_cw_on_a_narrow_track_within_2_hz_of_the_design():
    """One ID, at 62.54 s, about 0.26 s after our last keying drops; the arm
    log's nineteen unattributed bursts are its Morse elements. The design puts
    it at WS8EOC_CW_HZ and allows 2 Hz either way."""
    near = [r for r in _ends(_tape(str(WS8EOC_TAP)))
            if abs(r.centre_hz - WS8EOC_CW_HZ) <= 30]
    assert len(near) == 1, near
    [cw] = near
    assert cw.label == "CW-shaped"
    assert abs(cw.centre_hz - WS8EOC_CW_HZ) <= 2.0, cw
    assert abs(cw.t_start - 62.54) <= 0.2, cw
    assert 2.5 <= cw.active_s <= 5.0, cw
