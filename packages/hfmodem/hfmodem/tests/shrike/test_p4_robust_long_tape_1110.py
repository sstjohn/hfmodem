# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock robust LONG re-encoding, independently placed by production readers.
Edge fitting replaces the design's -20 dB crossing because these tapes' noise
floor sits only 8-15 dB under the burst.
"""
from functools import lru_cache
from hashlib import sha256
import json
from io import BytesIO
from math import gcd
from pathlib import Path
import subprocess
import shutil
import wave

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve, resample_poly

from hfmodem.shrike import (p4coding, p4frame, p4oracle, p4robustlongtx as tx,
    p4robustrx, p4robusttx, p4rx, p4sf8rx, p4sig)
from .test_p4robustlongtx import quarter_turns, recover_audio_body

ROOT = Path(__file__).resolve().parents[5]
# Between prototype random/no-interleave maximum .242 and true minimum .712.
BODY_MIN = .50
CONTROL_MAX = .30  # Above the prototype's .242 maximum unrelated-body control.
HEADER_MIN = .60  # Reviewed whole-header floor; the faded idle head is excluded.
HEADER_MARGIN = .30  # Reviewed separation from each of the other three classes.
COHERENCE_RATIO_MIN = .90  # Reviewed strong-frame requirement, coherence >= .6.
LAG_CHIPS_MAX = 1.0  # Reviewed placement tolerance in TX impulse coordinates.
LENGTH_MEDIAN_TOLERANCE_MS = 1.0  # Median absolute error: opposing errors cannot cancel.
LENGTH_FRAME_TOLERANCE_MS = 5.0  # Bound supported by a 2 ms envelope at 8 dB SNR.
LEADING_TAU_TOLERANCE_MS = 1.5  # Edge fit must agree with the correlation placement.
C2_CONTROL_MAX = .50  # Wrong dibits have chance agreement .25; .50 rejects structure.

# path, file SHA256, onset sample at 48 kHz, level, status, data, CRC, role,
# reader CFO, station. Each physical copy appears once, including fixture copies.
CASES = [{'path': 'captures/onair-0929-0121/stream.flac',
  'sha256': '4b2cf6abf0ba354687c68488167c9cc6516e2c1ffd11f00516278c57f9176075',
  'onset': 2688057,
  'level': 3,
  'status': 97,
  'data': '524d53205472696d6f646520312e342e342e300d0a573953534a2068617320313039206461696c79206d696e757465732072656d61696e696e672077697468205645334b50',
  'crc': 55412,
  'role': 'REPEAT',
  'cfo': 1.54,
  'station': 'VE3KPG',
  'header_gate': True},
 {'path': 'captures/onair-0929-1024/stream.flac',
  'sha256': '3d4b8d1269ac9c24f63494330819bf753738fe8067673f09068d86ebe4b509f9',
  'onset': 2867952,
  'level': 2,
  'status': 98,
  'data': '206d696e757465732072656d61696e696e672077697468205645334b50472028464e30345645290d',
  'crc': 25565,
  'role': 'NEW',
  'cfo': 3.026985452564156,
  'station': 'VE3KPG',
  'header_gate': True},
 {'path': 'captures/onair-0929-1024/stream.flac',
  'sha256': '3d4b8d1269ac9c24f63494330819bf753738fe8067673f09068d86ebe4b509f9',
  'onset': 3047952,
  'level': 3,
  'status': 3,
  'data': '5b574c324b2d352e302d4232465749484a4d245d0d3b50513a2032353238363737390d434d5320766961205645334b5047203e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e',
  'crc': 26018,
  'role': 'NEW',
  'cfo': 3.026985452564156,
  'station': 'VE3KPG',
  'header_gate': True},
 {'path': 'captures/onair-1001-1059/stream.flac',
  'sha256': '531e5f314fd321d09adda22de30d60205c0e079700a96ff5d72d68cae1692ce5',
  'onset': 2628029,
  'level': 3,
  'status': 97,
  'data': '524d53205472696d6f646520312e342e342e300d0a573953534a20686173203934206461696c79206d696e757465732072656d61696e696e672077697468205645334b5047',
  'crc': 11032,
  'role': 'NEW',
  'cfo': 0.9343233763961152,
  'station': 'VE3KPG',
  'header_gate': True},
 {'path': 'captures/onair-1001-1059/stream.flac',
  'sha256': '531e5f314fd321d09adda22de30d60205c0e079700a96ff5d72d68cae1692ce5',
  'onset': 4427957,
  'level': 2,
  'status': 97,
  'data': '524d53205472696d6f646520312e342e342e300d0a573953534a20686173203934206461696c7920',
  'crc': 25059,
  'role': 'REPEAT',
  'cfo': 0.9343233763961152,
  'station': 'VE3KPG',
  'header_gate': True},
 {'path': 'captures/onair-1002-2142/stream.flac',
  'sha256': 'b07cf45c99369a552d26b53d5c217b32650c3a435bef1bafb6815b1ac230f56c',
  'onset': 7248554,
  'level': 2,
  'status': 64,
  'data': '1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e',
  'crc': 42690,
  'role': 'NEW',
  'cfo': -0.024659811961935674,
  'station': 'WS8EOC',
  'header_gate': False},
 {'path': 'captures/onair-1002-2150/stream.flac',
  'sha256': '5503b4fa6c1b38f82877cd2524db2394f9521e6a34707dc20ed6b4c71e62fd74',
  'onset': 1368444,
  'level': 3,
  'status': 98,
  'data': '556e69766572736974790d0a573953534a2068617320313136206461696c79206d696e757465732072656d61696e696e672077697468204b344d53552028454d3536554f29',
  'crc': 63549,
  'role': 'NEW',
  'cfo': 2.4340165827987224,
  'station': 'K4MSU',
  'header_gate': True},
 {'path': 'captures/onair-1002-2154/stream.flac',
  'sha256': '7d703ad3fda45619ed1a5a51641339f2664d6566a7e6d9d5e5025d104a1e233e',
  'onset': 3528586,
  'level': 4,
  'status': 3,
  'data': '313137206461696c79206d696e757465732072656d61696e696e672077697468205645334b50472028464e30345645290d5b574c324b2d352e302d4232465749484a4d245d0d3b50513a2031383333393639340d434d5320766961205645334b5047203e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e',
  'crc': 64173,
  'role': 'NEW',
  'cfo': 0.69,
  'station': 'VE3KPG',
  'header_gate': True},
 {'path': 'captures/onair-1002-2154/stream.flac',
  'sha256': '7d703ad3fda45619ed1a5a51641339f2664d6566a7e6d9d5e5025d104a1e233e',
  'onset': 4428586,
  'level': 4,
  'status': 3,
  'data': '313137206461696c79206d696e757465732072656d61696e696e672077697468205645334b50472028464e30345645290d5b574c324b2d352e302d4232465749484a4d245d0d3b50513a2031383333393639340d434d5320766961205645334b5047203e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e',
  'crc': 64173,
  'role': 'REPEAT',
  'cfo': 0.69,
  'station': 'VE3KPG',
  'header_gate': True},
 {'path': 'operator-data/captures/onair-20261027T064359603967Z-732431ae-9d5f-4908-b964-efb24ed92a98/stream.flac',
  'sha256': '7ae4edaacd229a4fa7a6759d6add3e0042d5bb0cd26f664c95e34bfd1a4332b0',
  'onset': 3948607,
  'level': 4,
  'status': 99,
  'data': '746865204b345350422048796272696420524d5320696e204d656d706869732c20544e0d0a573953534a2068617320313830206461696c79206d696e757465732072656d61696e696e672077697468204b345350422028454d35354344290d4d696e696d756d20616c6c6f776564207375737461696e6564207472616e73666572207370656564206973203235',
  'crc': 43668,
  'role': 'NEW',
  'cfo': 4.952527899547475,
  'station': 'K4SPB',
  'header_gate': True},
 {'path': 'operator-data/captures/onair-20261102T052617935711Z-3ac21505-0157-4f8c-a9cf-53d23a48a95b/stream.flac',
  'sha256': 'c3cca4848106dcf21650d337d66800d6929c37cdd28cf41d0d72c9ba1bc6ac34',
  'onset': 2868405,
  'level': 2,
  'status': 97,
  'data': '6d696e757465732072656d61696e696e672077697468204b394b444a2028454d36374857290d7b53',
  'crc': 53491,
  'role': 'NEW',
  'cfo': -2.2648775395228293,
  'station': 'K9KDJ',
  'header_gate': True},
 {'path': 'operator-data/captures/onair-20261102T052617935711Z-3ac21505-0157-4f8c-a9cf-53d23a48a95b/stream.flac',
  'sha256': 'c3cca4848106dcf21650d337d66800d6929c37cdd28cf41d0d72c9ba1bc6ac34',
  'onset': 3048405,
  'level': 2,
  'status': 98,
  'data': '4649203d203934204f6e20323032362d31312d30322030323a3030205554437d0d0a5b574c324b2d',
  'crc': 45091,
  'role': 'NEW',
  'cfo': -2.2648775395228293,
  'station': 'K9KDJ',
  'header_gate': True},
 {'path': 'operator-data/captures/onair-20261102T052617935711Z-3ac21505-0157-4f8c-a9cf-53d23a48a95b/stream.flac',
  'sha256': 'c3cca4848106dcf21650d337d66800d6929c37cdd28cf41d0d72c9ba1bc6ac34',
  'onset': 3588405,
  'level': 2,
  'status': 98,
  'data': '4649203d203934204f6e20323032362d31312d30322030323a3030205554437d0d0a5b574c324b2d',
  'crc': 45091,
  'role': 'REPEAT',
  'cfo': -2.2648775395228293,
  'station': 'K9KDJ',
  'header_gate': True}]

# K9KDJ's repeat has two C2 channel errors; it is not a zero-error gate.
CHANNEL_ERROR_CASE = CASES.pop()


@lru_cache(None)
def checked_path(relative, expected):
    path = ROOT/relative
    if not path.is_file():
        pytest.skip(f'required stock robust LONG recording absent: {relative}')
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    assert h.hexdigest() == expected
    return path


def window(case, *, lead=.1, duration=3.6):
    path = checked_path(case['path'], case['sha256'])
    # FLAC STREAMINFO gives the native rate without launching a metadata tool.
    with path.open('rb') as f:
        assert f.read(4) == b'fLaC'
        block = f.read(4)
        assert block[0] & 0x7f == 0 and int.from_bytes(block[1:], 'big') == 34
        info = f.read(34)
    fs = int.from_bytes(info[10:18], 'big') >> 44
    start = round((case['onset']/48000-lead)*fs)
    count = round(duration*fs)
    flac = shutil.which('flac')
    assert flac is not None, 'flac is required to read the pinned stock windows'
    raw = subprocess.check_output([flac, '-d', '-c', '--silent',
        f'--skip={start}', f'--until=+{count}', str(path)], stderr=subprocess.PIPE)
    with wave.open(BytesIO(raw), 'rb') as source:
        assert source.getsampwidth() == 2 and source.getframerate() == fs
        x = np.frombuffer(source.readframes(count), '<i2').reshape(-1, source.getnchannels())[:, 0]/32768.
    assert len(x) == count
    g = gcd(fs, 48000)
    return resample_poly(x, 48000//g, fs//g) if fs != 48000 else x


def rho(a, b):
    return float(abs(np.vdot(a, b))/np.sqrt(np.vdot(a, a).real*np.vdot(b, b).real))


def baseband(audio):
    return p4rx.audio_to_baseband(np.r_[np.zeros(4800), audio, np.zeros(4800)], 48000)[2880:-2880]


def body_rho(template, received):
    return float(np.mean([rho(template[i:i+2880], received[i:i+2880])
        for i in range(5120, len(template)-2880+1, 2880)]))


def acquire(rx, template):
    # Acquire once on the body. Every hypothesis/control uses this CFO and lag.
    n = np.arange(len(rx))
    body = template[5120:]
    best = (-1, None, None)
    for grid in (np.arange(-4, 4.001, .25), None):
        if grid is None:
            grid = np.arange(max(-4, best[2]-.25), min(4, best[2]+.25)+.001, .025)
        for cfo in grid:
            corrected = rx*np.exp(-2j*np.pi*cfo*n/28800)
            corr = fftconvolve(corrected, body[::-1].conj(), 'valid')
            k = int(np.argmax(abs(corr)))
            if abs(corr[k]) > best[0]:
                best = (abs(corr[k]), k-5120, float(cfo))
    _, lag, cfo = best
    assert 0 <= lag and lag+len(template) <= len(rx)
    return (rx*np.exp(-2j*np.pi*cfo*n/28800))[lag:lag+len(template)], lag, cfo


def control_audio(bits, level, header, *, order=None):
    g = tx.GEOMETRY[level]
    if order is None:
        body = tx.body_symbols(bits, level)
    else:
        pairs = bits.reshape(-1, 2)
        steps = np.empty(g.data_symbols)
        steps[order] = np.array([0, 1, -1, 2])[2*pairs[:, 0]+pairs[:, 1]]
        body = np.exp(.5j*np.pi*np.r_[0, np.cumsum(steps)])
    spread = p4rx.SPREAD8 if level == 4 else p4sig.SPREAD16
    chips = np.r_[(tx.header_symbols(header)[:, None]*p4sig.SPREAD16).ravel(),
                  (body[:, None]*spread).ravel()]
    return baseband(p4robusttx.shape_chips(chips, 48000))


def wrong_c2(bits, level, rng):
    g = tx.GEOMETRY[level]
    D = g.systematic
    period, phase = (2, 1) if level == 2 else (10, 4)
    parity = p4coding.rsc_encode(bits[:D][rng.permutation(D)], initial_state=(0, 0, 0)).parity
    bad = bits.copy()
    bad[D+D//period:] = parity[phase::period]
    return bad


def envelope(z):
    return np.sqrt(np.maximum(uniform_filter1d(abs(z)**2,
        size=round(.002*28800)), 0))


def edge_tau_ms(tape, render, edge):
    """Fit tape(t) = a * render(t - tau) + b; positive tau is later."""
    radius, search = round(.040*28800), round(.020*28800)
    indices = np.arange(edge-radius, edge+radius+1)
    assert indices[0]-search >= 0 and indices[-1]+search < len(render)
    target = tape[indices]
    best = (np.inf, None)
    for tau in range(-search, search+1):
        design = np.column_stack((render[indices-tau], np.ones(len(indices))))
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = np.sum((target-design @ coefficients)**2)
        if residual < best[0]:
            best = (residual, tau)
    return best[1]/28.8


def edge_fit(case, template, lag):
    tape = baseband(window(case, lead=.2, duration=3.75))
    # Acquisition used a 100 ms lead; this window has a 200 ms lead.
    first = lag+2880
    last = first+len(template)-1
    render = np.zeros(len(tape), complex)
    render[first:last+1] = template
    tape_env, render_env = envelope(tape), envelope(render)
    leading = edge_tau_ms(tape_env, render_env, first)
    trailing = edge_tau_ms(tape_env, render_env, last)
    return dict(tau_leading_ms=leading, tau_trailing_ms=trailing,
                length_difference_ms=trailing-leading)


def duration_controls(wave):
    """Insert/cut 2 ms (3.6 chips) before the final pulse, at unchanged gain.

    The final 20 ms, including the trailing pulse, moves intact; the leading
    edge and every preceding sample stay fixed. At 48 kHz, 2 ms is exact.
    """
    n = 96
    at = len(wave)-960
    return {
        'long_2ms': baseband(np.r_[wave[:at], wave[at-n:at], wave[at:]]),
        'short_2ms': baseband(np.r_[wave[:at-n], wave[at:]]),
    }


@pytest.fixture(scope='module', params=CASES,
    ids=lambda c: f"{c['station']}-{c['onset']}-SL{c['level']}-{c['role']}")
def measured(request):
    return measure_case(CASES.index(request.param))


@lru_cache(None)
def measure_case(index):
    case = CASES[index]
    audio = window(case)
    level, cfo = case['level'], case['cfo']
    z = p4rx.audio_to_baseband(audio, 48000)
    if level == 4:
        ob = p4rx.recover_robust(z, reference_start=8000, spread_factor=8,
            data_symbols=696, timing_offsets=range(-64, 65), cfo_hz=(cfo,))
        r = p4sf8rx.decode_symbols(ob.symbols)
        raw, coherence, reference = ob.symbols, ob.coherence, ob.reference_start
    else:
        coherence, symbols, first = p4oracle.placed_despread(audio, 48000, .1, cfo, count=373)
        r = p4robustrx.decode_symbols(symbols)
        assert r is not None
        raw = symbols[r.field_start:r.field_start+349]
        reference = round(first*.6)+256*r.field_start
    assert r is not None
    frame = r.frame
    assert (frame.data.hex(), frame.status.byte, frame.crc) == (case['data'], case['status'], case['crc'])
    bits = tx.coded_bits(p4frame.encode_field(frame.data, frame.status), level)
    repeat = case['role'] == 'REPEAT'
    header = 'RQ' if repeat else level
    wave = tx.robust_long(frame.data, frame.status.byte, level, repeat=repeat)
    rendered_reception, rendered_raw = recover_audio_body(wave, level)
    assert rendered_reception.frame == frame
    np.testing.assert_array_equal(rendered_reception.fields < 0, bits)
    template = baseband(wave)
    received, lag, cfo = acquire(z, template)
    heads = {str(key): rho(baseband(tx.robust_long(frame.data, frame.status.byte,
        level, header=key))[:5120], received[:5120]) for key in tx.HEADER_STEPS}
    rng = np.random.default_rng(7)
    random = tx.coded_bits(p4frame.encode_field(
        rng.bytes(tx.GEOMETRY[level].data_bytes), frame.status), level)
    bad = wrong_c2(bits, level, rng)
    controls = {
        'random': body_rho(control_audio(random, level, header), received),
        'no_interleave': body_rho(control_audio(bits, level, header,
            order=np.arange(len(bits)//2)), received),
        'wrong_c2': body_rho(control_audio(bad, level, header), received),
    }
    D = tx.GEOMETRY[level].systematic
    P = (len(bits)-D)//2
    order = np.concatenate([np.arange(j, len(bits)//2, 29) for j in range(29)])
    c2_indices = order[(D+P)//2:]
    actual = quarter_turns(raw)[c2_indices]
    expected = quarter_turns(rendered_raw)[c2_indices]
    wrong = quarter_turns(tx.body_symbols(bad, level))[c2_indices]
    metric = dict(station=case['station'], onset=case['onset'], level=level,
        role=case['role'], body=body_rho(template, received), controls=controls,
        header=heads, coherence=coherence, lag_chips=(lag+5120-reference)/16,
        cfo=cfo, **edge_fit(case, template, lag), c2=float(np.mean(actual == expected)),
        wrong_c2=float(np.mean(actual == wrong)),
        errors=[int(x.sum()) for x in np.split(bits != (r.fields < 0), [D, D+P])])
    metric['absolute_length_difference_ms'] = abs(metric['length_difference_ms'])
    metric['duration_controls'] = {name: edge_fit(case, control, lag)
        for name, control in duration_controls(wave).items()}
    metric['coherence_ratio'] = metric['body']/coherence
    print('\nTAPE_METRICS '+json.dumps(metric, sort_keys=True), flush=True)
    return case, metric


def test_stock_bits_body_headers_and_placement(measured):
    case, m = measured
    assert m['errors'] == [0, 0, 0]
    assert m['body'] >= BODY_MIN
    assert m['controls']['random'] <= CONTROL_MAX
    assert m['controls']['no_interleave'] <= CONTROL_MAX
    if m['coherence'] >= .6:
        assert m['coherence_ratio'] >= COHERENCE_RATIO_MIN
    if case['header_gate']:
        key = 'RQ' if case['role'] == 'REPEAT' else str(case['level'])
        assert m['header'][key] >= HEADER_MIN
        assert all(m['header'][key]-value >= HEADER_MARGIN
                   for other, value in m['header'].items() if other != key)
    assert abs(m['lag_chips']) <= LAG_CHIPS_MAX


def test_stock_burst_envelope(measured):
    case, m = measured
    assert abs(m['tau_leading_ms']) <= LEADING_TAU_TOLERANCE_MS, (case, m)
    assert abs(m['length_difference_ms']) <= LENGTH_FRAME_TOLERANCE_MS, (case, m)


def test_stock_burst_envelope_median():
    differences = [measure_case(i)[1]['absolute_length_difference_ms'] for i in range(len(CASES))]
    median = float(np.median(differences))
    print(f'\nTAPE_ABSOLUTE_LENGTH_MEDIAN_MS {median}', flush=True)
    assert median <= LENGTH_MEDIAN_TOLERANCE_MS, differences


@pytest.mark.parametrize('control', ['long_2ms', 'short_2ms'])
def test_stock_burst_envelope_rejects_duration_error(control):
    fits = [measure_case(i)[1]['duration_controls'][control] for i in range(len(CASES))]
    differences = [abs(fit['length_difference_ms']) for fit in fits]
    median = float(np.median(differences))
    print('\nTAPE_DURATION_CONTROL '+json.dumps(dict(control=control,
        median_absolute_length_difference_ms=median, differences_ms=differences)), flush=True)
    assert all(abs(fit['tau_leading_ms']) <= LEADING_TAU_TOLERANCE_MS for fit in fits), fits
    assert median >= 1.5, (control, median, differences)


def test_stock_c2_only_differentials(measured):
    _, m = measured
    assert m['c2'] == 1.0
    assert m['wrong_c2'] <= C2_CONTROL_MAX


def test_pinned_channel_errors_are_not_a_clean_gate():
    case = CHANNEL_ERROR_CASE
    audio = window(case)
    _, symbols, _ = p4oracle.placed_despread(audio, 48000, .1, case['cfo'], count=373)
    r = p4robustrx.decode_symbols(symbols)
    assert r is not None
    assert (r.frame.data.hex(), r.frame.status.byte, r.frame.crc) == (
        case['data'], case['status'], case['crc'])
    bits = tx.coded_bits(p4frame.encode_field(r.frame.data, r.frame.status), 2)
    assert [int(x.sum()) for x in np.split(bits != (r.fields < 0), [348, 522])] == [0, 0, 2]
