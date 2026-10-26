# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Two third-party N9VC PACTOR-4 links, read where they sit in the passband.

N9VC is a BPQ32 node. On the 08-02 K5DAT-13 recording its link runs 400 Hz up
the passband; on the 09-25 Kiwi recording 910 Hz up. Nothing here moves the
audio: a reader that cannot find a frame where it was keyed is the gap.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from hfmodem.shrike import p1rx, p4chirprx, p4frame, p4oracle, p4robusttx, p4words, rxfront

FIXTURES = Path(__file__).parent / 'fixtures'
SID = b'[BPQ-6.0.25.1-B2FWIHJM'

P1_FIXED = ('the PACTOR-1 readers listen at a fixed 1400/1600 Hz and this link '
            'keys 910 Hz above that')


def _load(name: str):
    meta = json.loads((FIXTURES / f'{name}.json').read_text())
    _, pcm = wavfile.read(FIXTURES / meta['file'])
    x = pcm.astype(np.float64) / 32768
    if meta['rate'] == 48000:
        return meta, x, 48000
    # The Kiwi's true rate is its sidecar's, about 11999 Hz, not the header's.
    return meta, resample_poly(x, 4, 1), round(4 * meta['rate'])


@pytest.mark.parametrize('name,index', [('n9vc-0802-chirps', 0), ('n9vc-0802-chirps', 1),
                                        ('n9vc-0802-chirps', 2), ('n9vc-0802-chirp-weak', 0)])
def test_a_bpq_chirp_copies_where_it_sits(name, index):
    meta, x, fs = _load(name)
    chirp = meta['chirps'][index]
    got = p4chirprx.decode_packet(x, fs=fs, start_sample=round(chirp['start_s'] * fs),
                                  frequency_offset=chirp['offset_hz'])
    assert got is not None
    assert got.frame.data == SID
    assert got.frame.status.counter == 1
    assert got.frame.status.protocol == p4frame.ProtocolStatus.SPEED_UP


def test_the_damaged_chirp_fails_its_crc_everywhere_near_it():
    meta, x, fs = _load('n9vc-0802-chirp-crc-fail')
    chirp, = meta['chirps']
    start = round(chirp['start_s'] * fs)
    assert not any(
        p4chirprx.decode_packet(x, fs=fs, start_sample=start + slide,
                                frequency_offset=chirp['offset_hz'] + hz)
        for slide in range(-480, 481, 48) for hz in (-8, 0, 8))


def _shorts():
    for name in ('n9vc-0925-kiwi-connect', 'n9vc-0925-kiwi-robust-long'):
        meta, x, fs = _load(name)
        for short in meta['shorts']:
            yield short, p4oracle.read_burst(x, short['onset_s'], 0.76, fs)


def test_the_kiwi_robust_shorts_read_where_they_sit():
    """About +900 Hz, where a carrier and its chip-rate alias meet. A +/-600 Hz
    search locked them on an alias: the values survived it and most headers
    read '?'."""
    read = list(_shorts())
    assert [(r['p4_hdr'], r['p4_value'], r['p4_walsh_peak']) for _, r in read] == [
        (s['header'], s['value'], 64) for s, _ in read]


def test_our_robust_headers_are_the_callers_own():
    """Keyed on quarter turns they matched these five at 0.83-0.95."""
    meta, x, fs = _load('n9vc-0802-robust-shorts')
    for short in meta['shorts']:
        at = round(short['onset_s'] * fs)
        heard = p4oracle.despread_word(x[at:at + round(0.76 * fs)], fs)
        ours = p4robusttx.phasors(p4robusttx.HEADER_STEPS[short['header']])
        assert p4words.correlate(ours, heard[:ours.size + 3])[0] >= 0.97, short


def test_the_kiwi_robust_long_packet_is_named_robust():
    meta, x, fs = _load('n9vc-0925-kiwi-robust-long')
    long = meta['long']
    row = p4oracle.read_burst(x, long['onset_s'], long['duration_s'], fs)
    assert row['reply_class'] == 'p4-robust+block'


@pytest.mark.xfail(strict=True, reason=P1_FIXED)
def test_the_kiwi_connect_names_n9vc():
    meta, x, fs = _load('n9vc-0925-kiwi-connect')
    at = round(meta['connect']['at_s'] * fs)
    got = p1rx.decode_connect(x[at:at + round(2.3 * fs)], fs)
    assert got and (got.variant, got.callsign) == ('Normal', 'N9VC')


@pytest.mark.xfail(strict=True, reason=P1_FIXED + (
    '; moved there, the stream reader still takes only the second CS4 and the '
    'CS1, and the first CS4 needs acquire_control_signal'))
def test_the_kiwi_answer_reads_cs4_cs4_cs1():
    meta, x, _ = _load('n9vc-0925-kiwi-connect')
    heard = [(ev.t, ev.text.split('/')[0]) for ev in
             rxfront.decode_events(x.astype(np.float32), p3_envelope=False)
             if ev.kind == 'cs' and ev.protocol == 'PACTOR-1']
    for control in meta['controls']:
        assert any(abs(t - control['at_s']) < 0.15 and cs == control['cs']
                   for t, cs in heard), (control, heard)


@pytest.mark.xfail(strict=True, reason=P1_FIXED)
def test_the_callers_packet_reads_between_the_answers():
    meta, x, fs = _load('n9vc-0925-kiwi-connect')
    packet = meta['packet']
    got = [(p.payload, p.status) for p in p1rx.decode_p1_packets(x, fs)]
    assert (packet['text'].encode(), packet['status']) in got
