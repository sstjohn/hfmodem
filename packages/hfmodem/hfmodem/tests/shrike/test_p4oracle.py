# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Every probe window reads back as the answer to the chirp before it."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from hfmodem.shrike import p4oracle, p4robusttx, p4sig, p4words, pactor1

FS = 48000


def _quiet(seconds, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 3e-4, int(seconds * FS)).astype(np.float32)


def _place(window, burst, at):
    start = int(at * FS)
    window[start:start + burst.size] += burst[:window.size - start]
    return window


def test_a_silent_window_names_no_reply():
    assert p4oracle.read_window(_quiet(4.0)) == []
    row = p4oracle.query(1, 0x61, 0, 48000, _quiet(4.0))
    assert row['reply_class'] == 'none' and row['counter'] == 1
    line, = p4oracle.lines(row)
    assert line.startswith('    [p4 query] 1 status=0x61 counter=1 field=0B -> none')
    assert 'dB floor' in line


def test_a_codeword_in_the_window_is_read_as_one():
    window = _place(_quiet(4.0), pactor1.control_signal(0) * 4, 0.6)
    replies = p4oracle.read_window(window)
    assert len(replies) == 1
    assert replies[0]['reply_class'] == 'p1-cs:CS1/ack'
    assert 590 <= replies[0]['onset_ms'] <= 640




def _robust(symbols, seed=2, cfo=0.0):
    """A robust-mode burst: `symbols` DQPSK symbols spread by the published SF-16."""
    from hfmodem.shrike import p4rx
    rng = np.random.default_rng(seed)
    phases = np.exp(2j * np.pi * rng.integers(0, 4, symbols) / 4)
    chips = (phases[:, None] * p4sig.SPREAD16[None, :]).ravel()
    impulses = np.zeros(p4rx.SAMPLES_PER_CHIP * (chips.size - 1) + 1, complex)
    impulses[::p4rx.SAMPLES_PER_CHIP] = chips
    baseband = np.convolve(impulses, p4rx.RRC_TAPS)
    t = np.arange(baseband.size) / p4rx.SAMPLE_RATE
    wave = (baseband * np.exp(2j * np.pi * (p4rx.CENTRE_HZ + cfo) * t)).real
    wave = resample_poly(wave, 5, 3)          # 28800 -> 48000
    return (wave / np.abs(wave).max() * .5).astype(np.float32)


def _short_packet():
    from hfmodem.shrike import p4shorttx
    return p4shorttx.short_packet(b'W9SSJ', 0x61).astype(np.float32) * .5


def test_the_spreading_is_what_divides_the_two_families():
    """The one question asked before any header: SF-16 or not."""
    from hfmodem.shrike import p4chirp
    robust = p4oracle.despread_fraction(_robust(85))
    normal = p4oracle.despread_fraction(_short_packet())
    chirp = p4oracle.despread_fraction(p4chirp.entry_packet(b'', 0x61))
    quiet = p4oracle.despread_fraction(_quiet(0.8))
    assert robust >= p4oracle.ROBUST_FRACTION_KNEE
    assert max(normal, chirp, quiet) < p4oracle.ROBUST_FRACTION_KNEE


def test_the_training_lattice_is_what_names_a_normal_short():
    from hfmodem.shrike import p4chirp
    assert p4oracle.training_lock(_short_packet()) >= p4oracle.NORMAL_LATTICE_KNEE
    # The negative controls: a robust burst, our own chirp render, and noise.
    for name, seg in (('robust', _robust(85)),
                      ('chirp', p4chirp.entry_packet(b'', 0x61)),
                      ('noise', _quiet(0.9, seed=21) + _quiet(0.9, seed=22))):
        assert p4oracle.training_lock(seg) < p4oracle.NORMAL_LATTICE_KNEE, name


def test_a_normal_short_is_named_off_its_lattice():
    reply, = p4oracle.read_window(_place(_quiet(4.0), _short_packet(), 0.5))
    assert reply['reply_class'] == 'p4-normal'
    assert reply['training_lock'] >= p4oracle.NORMAL_LATTICE_KNEE
    assert reply['despread_fraction'] < p4oracle.ROBUST_FRACTION_KNEE
    assert reply['root'] is None and reply['header_score'] is None
    assert 'training lattice' in p4oracle.lines(
        p4oracle.query(1, 0x63, 0, 0, _place(_quiet(4.0), _short_packet(), 0.5)))[0]


@pytest.mark.parametrize('symbols,expected', [
    (85, 'p4-robust-short'), (120, 'p4-robust+block'), (12, 'energy'),
])
def test_the_robust_family_is_cut_on_the_sixteen_chip_grid(symbols, expected):
    burst = _robust(symbols)
    reply, = p4oracle.read_window(_place(_quiet(5.0), burst, 0.5))
    assert reply['reply_class'] == expected, reply['duration_ms']
    assert reply['despread_fraction'] >= p4oracle.ROBUST_FRACTION_KNEE


@pytest.mark.parametrize('symbols', [21, 27])
def test_a_word_shaped_burst_the_bank_cannot_name_says_so(symbols):
    """A clean SF16 lock on a control-length burst is a word we do not hold."""
    reply, = p4oracle.read_window(_place(_quiet(5.0), _robust(symbols), 0.5))
    assert reply['reply_class'] == 'p4-word:?'
    assert reply['despread_fraction'] >= p4oracle.WORD_LOCK_KNEE
    assert reply['word'] is None and reply['word_score'] < p4words.MATCH_KNEE


@pytest.mark.parametrize('duration,expected', [
    (0.170, 'p4-header'), (0.245, 'p4-robust-cs'),
    (0.760, 'p4-robust-short'), (0.900, 'p4-robust+block'), (0.100, 'energy'),
])
def test_a_weak_lock_falls_back_to_the_length(duration, expected):
    """Below `WORD_LOCK_KNEE` the symbols are not worth a word, so length names it."""
    row = dict(p1_cs=None, despread_fraction=0.33, word=None,
               training_lock=0.0, occupied_bw_hz=2200.0)
    assert p4oracle._classify(row, duration) == expected


def test_a_fragment_is_never_named_for_what_it_is_a_fragment_of():
    """50-140 ms of a real robust emission is energy, not a header."""
    for symbols in (6, 10, 15):
        reply, = p4oracle.read_window(_place(_quiet(4.0), _robust(symbols), 0.5))
        assert reply['duration_ms'] < p4oracle.HEADER_MIN_S * 1e3
        assert reply['reply_class'] == 'energy'


def test_the_root_rides_beside_the_class_and_never_in_it():
    reply, = p4oracle.read_window(_place(_quiet(5.0), _robust(85), 0.5))
    assert reply['reply_class'] == 'p4-robust-short'
    assert reply['root'] is not None and reply['header_score'] is not None
    provisional, = p4oracle.read_window(_place(_quiet(5.0), _robust(85), 0.5),
                                        headers=False)
    assert provisional['reply_class'] == 'p4-robust?'
    assert provisional['root'] is None
    assert p4oracle.class_key('p4-header(14,13,0.464)') == 'p4-header'
    assert p4oracle.class_key('p4-robust-short') == 'p4-robust-short'
    assert p4oracle.class_key('p1-cs:CS1/ack') == 'p1-cs:CS1/ack'


def test_the_cheap_pass_still_reads_a_codeword():
    window = _place(_quiet(4.0), pactor1.control_signal(0) * 4, 0.6)
    reply, = p4oracle.read_window(window, headers=False)
    assert reply['reply_class'] == 'p1-cs:CS1/ack'


def test_a_filled_window_is_not_silence():
    rng = np.random.default_rng(13)
    filled = rng.normal(0, .05, 4 * FS).astype(np.float32)
    assert p4oracle.read_window(filled) == []
    level, tenth = p4oracle.inband_db(filled)
    quiet = p4oracle.query(1, 0x61, 0, 0, filled)
    assert quiet['reply_class'] == 'none' and quiet['window_rms_db'] == round(level, 1)
    named = p4oracle.query(1, 0x61, 0, 0, filled, floor_db=tenth - 20)
    assert named['reply_class'] == 'filled'
    assert 'filled' in p4oracle.lines(named)[0] and 'dB' in p4oracle.lines(named)[0]
    assert p4oracle.query(1, 0x61, 0, 0, _quiet(4.0))['reply_class'] == 'none'


def test_the_strongest_burst_names_the_window():
    window = _place(_quiet(6.0), pactor1.control_signal(0) * 4, 0.6)
    window = _place(window, _robust(85), 2.0)
    row = p4oracle.query(3, 0x60, 0, 96000, window)
    assert [r['reply_class'] for r in row['replies']] == ['p1-cs:CS1/ack',
                                                          'p4-robust-short']
    assert row['reply_class'] == 'p4-robust-short'
    assert len(p4oracle.lines(row)) == 2


@pytest.mark.parametrize('status,counter', [(0x60, 0), (0x61, 1), (0x63, 3)])
def test_the_counter_is_the_low_two_bits_of_the_query(status, counter):
    row = p4oracle.query(1, status, 5, 48000, _quiet(4.0))
    assert row['counter'] == counter and row['status_hex'] == f'0x{status:02X}'
    assert row['payload_len'] == 5 and row['chirp_end_s'] == 1.0


def test_a_locked_offset_scores_one_carrier_and_says_so():
    window = _place(_quiet(5.0), _robust(85, cfo=6.0), 0.5)
    free, = p4oracle.read_window(window)
    locked, = p4oracle.read_window(window, cfo_hz=6.0)
    assert not free['cfo_locked'] and locked['cfo_locked']
    assert locked['cfo_hz'] == 6.0
    assert p4oracle.query(1, 0x63, 0, 0, window, cfo_hz=6.0)['cfo_locked']
    assert not p4oracle.query(1, 0x63, 0, 0, window)['cfo_locked']


def test_the_reader_names_a_real_robust_recording():
    """Real PACTOR-4 off the air, where the recording is on this machine."""
    wav = (Path(__file__).resolve().parents[5] / 'working/pactor4-decode-0916'
           / 'normal-short/local-audio/PACTOR-IV_112-5_DQPSK.WAV')
    if not wav.exists():
        pytest.skip(f'{wav} absent -- the local PACTOR-4 recordings are not here')
    fs, audio = wavfile.read(wav)
    x = audio.mean(1) if audio.ndim > 1 else audio
    x = x.astype(np.float32)[:5 * fs] / (np.iinfo(audio.dtype).max + 1)
    read = p4oracle.read_window(x, fs)
    named = [r['reply_class'] for r in read]
    # NO WORD IS NAMED HERE. The four robust-shorts are two headers alternating
    # -- {0,2} against {1,3} -- and neither is in the bank; the 0.96 the third
    # once scored against CALLER20 sat on both search boundaries at once
    # (`p4words.interior`), which is the search running out of room.
    assert named == ['p4-robust-short'] * 4, named
    assert all(r['word'] is None for r in read)
    heads = [r['word_score'] for r in read[:4]]
    assert heads[0] == pytest.approx(heads[2], abs=.35), heads
    assert heads[1] == pytest.approx(heads[3], abs=.05), heads
    # THE BLOCK IS READ EVEN WHERE THE BANK CANNOT NAME THE HEAD. Two values,
    # each keyed twice, once under each header, straight off the recording.
    assert [(r['p4_value'], r['p4_hdr']) for r in read] == [
        (13, 'A'), (13, 'B'), (49, 'A'), (49, 'B')]
    assert [r['p4_walsh_peak'] for r in read] == [64] * 4
    assert 'header A value 13 peak 64/64' in p4oracle.word_note(read[0])


#: Header A is the bank's CALLER20, so a header-A short is named by its word.
_SHORT_CLASS = {'A': 'p4-word:CALLER20', 'B': 'p4-robust-short'}


@pytest.mark.parametrize('value,header', [(6, 'A'), (0, 'B'), (63, 'A')])
def test_a_keyed_robust_value_comes_back_as_itself(value, header):
    """The whole point of the reader: what we key is what a window reads."""
    burst = p4robusttx.robust_short(value, header).astype(np.float32) * .5
    reply, = p4oracle.read_window(_place(_quiet(4.0), burst, 0.6))
    assert reply['reply_class'] == _SHORT_CLASS[header]
    assert (reply['p4_value'], reply['p4_walsh_peak'], reply['p4_hdr']) \
        == (value, 64, header)
    assert f'header {header} value {value} peak 64/64' in p4oracle.word_note(reply)


def test_a_robust_burst_that_is_not_a_codeword_is_not_given_a_value():
    """The null the knee is set against: random DQPSK at the same length."""
    peaks = []
    for seed in range(12):
        reply, = p4oracle.read_window(_place(_quiet(4.0), _robust(85, seed), 0.6))
        assert reply['reply_class'] == 'p4-robust-short'
        assert reply['p4_value'] is None and reply['p4_hdr'] == '?'
        peaks.append(reply['p4_walsh_peak'])
    assert max(peaks) < p4oracle.ROBUST_BLOCK_KNEE, peaks


def _burst(steps):
    """One spread burst from an explicit dibit sequence, at window level."""
    return p4robusttx._render(p4robusttx.symbols(steps), FS).astype(np.float32) * .5


def test_a_control_burst_reports_its_six_steps_or_nothing():
    """No mask is known for the 27-symbol field, so it is read raw or not read.

    The tape's three answerer bursts each spend two or three of their six steps
    off the differential-BPSK grid, and open on neither header, so none of them
    is a six-bit field. A control burst that does open on a header is read.
    """
    for k in range(3):
        field = np.array(p4robusttx.CS_DIBITS[k])[p4robusttx.HEADER_SYMBOLS:]
        assert p4oracle._raw_field(field) is None
    # A clean control burst is named by the bank, not by its length, and the
    # field is read off the length all the same.
    burst = p4robusttx.robust_cs_clone(0).astype(np.float32) * .5
    reply, = p4oracle.read_window(_place(_quiet(4.0), burst, 0.6))
    assert reply['reply_class'] == 'p4-word:?'
    assert reply['p4_walsh_peak'] is None
    assert reply['p4_hdr'] == '?' and reply['p4_value'] is None


def test_a_control_burst_on_a_known_header_hands_over_its_six_bits():
    steps = (p4robusttx.HEADER_DIBITS['A'] + (p4robusttx.REFERENCE_STEP,)
             + (0, 2, 2, 0, 0, 2))
    assert len(steps) == p4robusttx.CS_SYMBOLS - 1
    reply, = p4oracle.read_window(_place(_quiet(4.0), _burst(steps), 0.6))
    assert reply['p4_hdr'] == 'A'
    assert reply['p4_value'] == 0b011001 == 25
    assert reply['p4_walsh_peak'] is None
    assert 'header A value 25' in p4oracle.word_note(reply)


def test_a_short_we_cannot_decode_still_names_its_header():
    """A Dragon answering with a block that is not a codeword said SOMETHING.

    The Walsh peak picks the origin only when it reaches one. Where it does not,
    the header is asked on its own, or the burst reads as a packet from nobody.
    """
    rng = np.random.default_rng(11)
    for _ in range(8):
        block = rng.integers(0, 2, p4robusttx.BLOCK_SYMBOLS)
        if p4robusttx.block_value(block)[1] >= p4oracle.ROBUST_BLOCK_KNEE:
            continue
        steps = (p4robusttx.HEADER_DIBITS['A'] + (p4robusttx.REFERENCE_STEP,)
                 + tuple(2 * int(b) for b in block))
        reply, = p4oracle.read_window(_place(_quiet(4.0), _burst(steps), 0.6))
        assert reply['reply_class'] in _SHORT_CLASS.values()
        assert reply['p4_hdr'] == 'A', reply
        assert reply['p4_value'] is None
        assert reply['p4_walsh_peak'] < p4oracle.ROBUST_BLOCK_KNEE


def test_the_block_reading_is_absent_from_every_other_class():
    window = _place(_quiet(4.0), pactor1.control_signal(0) * 4, 0.6)
    reply, = p4oracle.read_window(window)
    assert reply['p4_value'] is None and reply['p4_hdr'] is None
    assert p4oracle.word_note(reply) == ''


def test_a_link_four_hundred_hertz_up_the_passband_reads_where_it_sits():
    """N9VC's five robust shorts, 400 Hz high, read without retuning them.

    A +/-120 Hz carrier search locked them on an alias: the values came through
    and every header read '?'.
    """
    fixtures = Path(__file__).parent / 'fixtures'
    meta = json.loads((fixtures / 'n9vc-0802-robust-shorts.json').read_text())
    fs, audio = wavfile.read(fixtures / meta['file'])
    x = audio.astype(np.float32) / 32768
    read = [p4oracle.read_burst(x, s['onset_s'], 0.76, fs) for s in meta['shorts']]
    assert [(r['p4_hdr'], r['p4_value'], r['p4_walsh_peak']) for r in read] == [
        (s['header'], s['value'], 64) for s in meta['shorts']]
    assert [r['word'] for r in read] == [
        'CALLER20' if s['header'] == 'A' else None for s in meta['shorts']]
    assert all(370 < r['cfo_hz'] < 420 for r in read), [r['cfo_hz'] for r in read]
