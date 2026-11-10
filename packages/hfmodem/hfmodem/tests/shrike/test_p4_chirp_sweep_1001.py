# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""R2 S1–S3: calibrated witness fits, independent geometry, recorded decisions."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.shrike import p4chirp, p4chirprx
from hfmodem.tests import evidence
from hfmodem.tests.shrike import chirp_sweep_analyzer as cb

ROOT = Path(__file__).resolve().parents[5]
evidence.require_input(ROOT / 'tools/witness_align.py', allow_module_level=True)
sys.path.insert(0, str(ROOT / 'tools'))
import witness_align

FIXTURES = Path(__file__).with_name('fixtures')
evidence.require_input(FIXTURES / 'chirp-sweep-2226.json', allow_module_level=True)
INVENTORY = json.loads((FIXTURES / 'chirp-sweep-2226.json').read_text())
ROWS = INVENTORY['rows']
STOCK_MS = np.array([[-4.75, -3.75], [-5., -4.25], [-3.5, -3.5], [-4.25, -3.75]])
CALIBRATION_MS = {
    12.5: np.array([[0., 0.], [0., -.25], [0., .25], [0., .25]]),
    12.25: np.array([[-3.5, -3.25], [-3.5, -3.5], [-3.5, -3.], [-3.5, -3.]]),
}


@lru_cache(None)
def witness():
    path = ROOT / INVENTORY['witness']
    for filename in (path, path.with_suffix('.json')):
        evidence.require_input(filename)
        with filename.open('rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == INVENTORY[filename.name.replace('.', '_') + '_sha256']
    return witness_align._at_48k(path, witness_align.witness_rate(path, 11999.))


def interior(fit, coarse):
    for result, center in zip(fit, coarse):
        assert abs(result[1] - center[1]) < .003 - 1e-9
        assert abs(result[2] - center[2]) < .4 - 1e-9
    return np.array([r[1] * 1000 for r in fit])


def test_analyzer_baselines_match_spec_values():
    np.testing.assert_array_equal(cb.PULSE, p4chirp.QUASI_RRC)
    assert cb.PACKET_S == p4chirp.PACKET_S
    from hfmodem.shrike import pactor2
    for marker, references in cb.MARKER.items():
        for frozen, spec in zip(references, pactor2.marker_steps(marker)):
            np.testing.assert_allclose(frozen, spec, rtol=0, atol=0)


def render_fit(row):
    audio = p4chirp.entry_packet(bytes.fromhex(row['data_hex']), row['status'],
        repeat=row['marker'] == 9, rank0_upper=row['rank0_upper'])
    padded = np.r_[np.zeros(48000), audio, np.zeros(48000)]
    symbols = cb.symbols(cb.segment(padded, 48000, 48000), 0.)
    d = [s[1:] * s[:-1].conj() for s in symbols]
    coarse = [(1., (p4chirp.CHIRP_START_T - 12.5) * .015, 294., 0.)] * 2
    return interior(cb.fine_fit(d, row['marker'], coarse), coarse)


@lru_cache(None)
def stock_fits():
    assert cb.ORIGIN == (12.5 + 31 / 16 - .5) * .015
    assert cb.RATE == 294.
    assert cb.baseband.__defaults__ == (cb.ORIGIN, 294.)
    assert cb.symbols.__defaults__[:2] == (cb.ORIGIN, 294.)
    onsets = np.array([r['start_s'] - INVENTORY['lag'] for r in ROWS])
    np.testing.assert_allclose(onsets, [58.501833, 62.251917, 66.001708, 73.501979], atol=5e-7, rtol=0)
    fits = np.array([interior(cb.fine(witness(), 48000, r), r['sweep']) for r in ROWS])
    np.testing.assert_allclose(fits, STOCK_MS, atol=1e-9)
    return fits


@pytest.mark.parametrize('origin', [12.5, 12.25])
def test_s1_fixed_analyzer_calibrates_each_field(monkeypatch, origin):
    monkeypatch.setattr(p4chirp, 'CHIRP_START_T', origin)
    fits = np.array([render_fit(r) for r in ROWS])
    np.testing.assert_allclose(fits, CALIBRATION_MS[origin], atol=1e-9)
    residuals = stock_fits() - fits
    median = np.median(residuals.mean(axis=1))
    assert median == pytest.approx(-4.1875 if origin == 12.5 else -.8125)
    print(dict(origin=origin, calibration_ms=fits.tolist(), residual_ms=residuals.tolist(), pooled_ms=median))


def test_s1_production_origin_passes_calibrated_gate():
    residuals = stock_fits() - np.array([render_fit(r) for r in ROWS])
    q = residuals.mean(axis=1)
    omitted = [np.median(np.delete(q, i)) for i in range(4)]
    print(dict(residual_ms=residuals.tolist(), pooled_ms=np.median(q),
               max_ms=np.max(abs(residuals)), omitted_ms=omitted, copy_means_ms=q.tolist()))
    assert abs(np.median(q)) <= 1.
    assert np.max(abs(residuals)) <= 1.75
    assert max(abs(np.array(omitted))) <= 1.
    assert np.median(q) == pytest.approx(-.8125)


def test_s2_both_carriers_have_independent_sample_origin_and_geometry(monkeypatch):
    phases, stuffed = [], []
    exp, convolve = p4chirp.np.exp, p4chirp.np.convolve

    def capture_exp(arg):
        if np.size(arg) == 158760:
            phases.append(np.imag(arg) / (2 * np.pi))
        return exp(arg)

    def capture_convolve(arg, *args):
        stuffed.append(arg.copy())
        return convolve(arg, *args)

    monkeypatch.setattr(p4chirp.np, 'exp', capture_exp)
    monkeypatch.setattr(p4chirp.np, 'convolve', capture_convolve)
    audio = p4chirp.entry_packet(b'geometry', 0x61)
    assert len(audio) == 158760
    assert len(phases) == len(stuffed) == 2
    i = np.arange(158760)
    for phase, tone in zip(phases, (1530., 550.)):
        sweep = phase - tone * i / 48000
        assert np.flatnonzero(sweep > 1e-10)[0] == 9856
        np.testing.assert_allclose(sweep, 294. / 2 * (np.maximum(i - 9855, 0) / 48000)**2, atol=2e-12)
    assert 9855 > .200 * 48000
    assert p4chirp.PATH.n_symbols == 208
    assert p4chirp.QUASI_RRC.size == 32
    for pulse, delay in zip(stuffed, (0, 4)):
        assert len(pulse) == 1764
        np.testing.assert_array_equal(np.flatnonzero(pulse), delay + 8 * np.arange(217))
        np.testing.assert_allclose(abs(pulse[pulse != 0]), 1., atol=1e-15)


@lru_cache(None)
def stream(arm):
    return wav.read(evidence.require_input(ROOT / f'captures/onair-{arm}/stream.flac'), 48000)


def identity(result):
    assert result is not None
    return result.frame.status.byte, result.frame.data, result.marker, result.rank0_upper


@pytest.mark.parametrize('arm,at,cfo,expected', [
    ('0930-2206', 39.750, .44, (0x61, b'RMS Trimode 1.4.4.0 We', 8, False)),
    ('0930-2206', 43.502, .44, (0x61, b'RMS Trimode 1.4.4.0 We', 9, True)),
    ('0930-2206', 54.750, .44, (0x62, b'lcome to K4MSU Winlink', 9, False)),
    ('1001-0952', 81.000, 1.08, (0x61, b'RMS Trimode 1.4.4.0 We', 8, True)),
    ('1001-0952', 84.747, 1.08, (0x61, b'RMS Trimode 1.4.4.0 We', 9, False)),
    ('1001-0952', 88.500, 1.08, (0x62, b'lcome to K4MSU Winlink', 8, True)),
])
def test_s3_six_raw_recorded_fields(arm, at, cfo, expected):
    raw = stream(arm)
    header = p4chirprx.acquire_header(raw, nominal_sample=round(at * 48000), frequency_offset=cfo)
    assert header is not None
    assert identity(p4chirprx.decode_packet(raw, start_sample=header.start_sample, frequency_offset=cfo)) == expected


@pytest.mark.parametrize('row', ROWS)
def test_s3_four_fixed_witness_fields(row):
    result = p4chirprx.decode_packet(witness(), start_sample=round(row['start_s'] * 48000), frequency_offset=row['cfo'])
    assert identity(result) == (row['status'], bytes.fromhex(row['data_hex']), row['marker'], row['rank0_upper'])


@pytest.mark.parametrize('name', ['n9vc-0802-chirp-weak', 'n9vc-0802-chirp-crc-fail'])
def test_s3_recorded_weak_and_crc_failure(name):
    path = FIXTURES / f'{name}.wav'
    raw = evidence.checked_samples(path, path.with_suffix('.json'))
    row = json.loads(path.with_suffix('.json').read_text())['chirps'][0]
    result = p4chirprx.decode_packet(raw, start_sample=round(row['start_s'] * 48000), frequency_offset=row['offset_hz'])
    if row['crc']:
        assert identity(result) == (0x61, row['data'].encode(), 9, False)
        assert p4chirprx.acquire_header(raw, nominal_sample=round(row['start_s'] * 48000), frequency_offset=row['offset_hz']) is not None
    else:
        assert result is None
        assert p4chirprx.acquire_header(raw, nominal_sample=round(row['start_s'] * 48000), frequency_offset=row['offset_hz']) is None


def test_s2_rx_uses_the_same_numeric_origin(monkeypatch):
    audio = p4chirp.entry_packet(b'geometry', 0x61)
    phases = []
    exp = p4chirprx.np.exp

    def capture(arg):
        if np.size(arg) == 26460:
            phases.append(np.imag(arg) / (-2 * np.pi))
        return exp(arg)

    monkeypatch.setattr(p4chirprx.np, 'exp', capture)
    assert p4chirprx.decode_packet(audio) is not None
    assert len(phases) == 2
    t = np.arange(26460) / 8000
    for phase, tone in zip(phases, (1530., 550.)):
        np.testing.assert_allclose(phase - tone * t,
            .5 * 294. * np.maximum(t - 9855 / 48000, 0)**2, atol=2e-12)
