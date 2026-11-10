# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""45.45 Bd Baudot RTTY round trips: clean, noisy, offset, and a broken stop."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.lanner import fsk, ita2, rtty

TEXT = "RYRYRY THE QUICK BROWN FOX 0123456789"


def test_clean_round_trip():
    assert rtty.demodulate(rtty.modulate(TEXT)) == TEXT


@pytest.mark.parametrize('usos,expected', [(True, '123456 UIOPQW ERTYUI\n'),
                                         (False, '123456 789012 345678\n')])
def test_receiving_space_policy_without_redundant_figures_shifts(usos, expected):
    codes = ita2.encode('123456 789012 345678\n')
    first_figs = codes.index(27)
    codes = codes[:first_figs + 1] + [code for code in codes[first_figs + 1:] if code != 27]
    schedule = [(10, True)]
    for code in codes:
        schedule.extend([(1, False)] + [(1, bool(code >> k & 1)) for k in range(5)]
                        + [(1.5, True)])
    schedule.append((10, True))
    fs = 6000
    bounds = np.rint(np.r_[0, np.cumsum([n for n, _ in schedule])] * fs / rtty.BAUD).astype(int)
    tones = np.repeat([1585 if mark else 1415 for _, mark in schedule], np.diff(bounds))
    samples = .5 * np.sin(2 * np.pi * np.cumsum(tones) / fs)
    assert rtty.demodulate(samples, fs=fs, unshift_on_space=usos) == expected
    stream = rtty.Stream(fs=fs, unshift_on_space=usos)
    heard = [stream.feed(samples[start:start + 997]) for start in range(0, len(samples), 997)]
    heard.append(stream.flush())
    assert ''.join(heard) == expected


def test_round_trip_with_noise_at_10_db_snr():
    audio = rtty.modulate(TEXT).astype(np.float64)
    rng = np.random.default_rng(7)
    spec = np.fft.rfft(rng.standard_normal(audio.size))
    spec[np.fft.rfftfreq(audio.size, 1 / fsk.FS) > 3000] = 0
    noise = np.fft.irfft(spec, audio.size)
    noise *= np.sqrt(np.mean(audio**2) / np.mean(noise**2) / 10)
    assert rtty.demodulate(audio + noise) == TEXT


def test_round_trip_with_tone_offset():
    for offset in (10.0, -10.0):
        audio = rtty.modulate(TEXT, mark_hz=rtty.MARK_HZ + offset, space_hz=rtty.SPACE_HZ + offset)
        assert rtty.demodulate(audio) == TEXT


def test_broken_stop_bit_drops_the_character():
    def mod(units):
        return fsk.modulate(units, baud=rtty.BAUD, f_hi=rtty.MARK_HZ, f_lo=rtty.SPACE_HZ)

    # ITA2 "A" (code 3, bits ZZAAA): lead-in, start space, five units, stop.
    units = [True] * 10 + [False, True, True, False, False, False, True]
    assert rtty.demodulate(mod(units)) == "A"
    assert rtty.demodulate(mod(units[:-1] + [False])) == ""


def test_characters_returns_codes_and_start_edges():
    text = "RY 123 ABC 456\nEND"
    # Leave a small timing margin: detected edges jitter by a few samples with
    # tone phase, even when physical frame starts are exactly 7.5 units apart.
    audio = rtty.modulate(text, stop_units=1.6)
    frames = rtty.characters(audio)
    positions, codes = zip(*frames)
    assert list(codes) == ita2.encode(text)
    assert np.all(np.diff(positions) >= 7.5 * fsk.FS / rtty.BAUD)
    assert abs(positions[0] - round(rtty.LEAD_TAIL_UNITS * fsk.FS / rtty.BAUD)) < fsk.FS / rtty.BAUD
    assert rtty.characters(audio, start=positions[3]) == frames[3:]
    assert rtty.characters(audio, start=positions[4]) == frames[4:]


@pytest.mark.parametrize("noisy", [False, True])
def test_stream_random_blocks_match_finished_buffer(noisy):
    text = "RYRY 0123456789 ABC!DEF 98.76\n" * 3
    audio = rtty.modulate(text).astype(np.float64)
    if noisy:
        rng = np.random.default_rng(17)
        spectrum = np.fft.rfft(rng.standard_normal(audio.size))
        spectrum[np.fft.rfftfreq(audio.size, 1 / fsk.FS) > 3000] = 0
        noise = np.fft.irfft(spectrum, audio.size)
        noise *= np.sqrt(np.mean(audio**2) / np.mean(noise**2) / 10**0.6)
        audio += noise
    expected = rtty.demodulate(audio)
    assert expected == text
    stream = rtty.Stream()
    rng = np.random.default_rng(91)
    out = []
    pos = 0
    while pos < audio.size:
        size = int(rng.integers(round(0.001 * fsk.FS), 3 * fsk.FS + 1))
        out.append(stream.feed(audio[pos:pos + size]))
        pos += size
    out.append(stream.flush())
    assert "".join(out) == expected
    assert stream.flush() == ""
    assert stream.feed([]) == ""


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("block_size", [1, 1500, 20000])
def test_stream_complex_baseband_matches_finished_buffer(dtype, block_size):
    from scipy.signal import hilbert

    fs = 6000
    text = "CQ TEST 12345\n"
    audio = rtty.modulate(text, fs=fs)
    baseband = (hilbert(audio) *
                np.exp(-2j * np.pi * 1500 * np.arange(audio.size) / fs)).astype(dtype)
    params = dict(fs=fs, mark_hz=85, space_hz=-85)
    expected = rtty.demodulate(baseband, **params)
    assert expected == text
    stream = rtty.Stream(**params)
    out = [stream.feed(baseband[pos:pos + block_size])
           for pos in range(0, baseband.size, block_size)]
    out.append(stream.flush())
    assert "".join(out) == expected


def test_stream_tiny_blocks_preserve_shifts_and_custom_parameters():
    params = dict(baud=50, mark_hz=1800, space_hz=1600, fs=8000)
    text = "A12B34C=5Z+6\n"
    audio = rtty.modulate(text, variant="itu", **params)
    stream = rtty.Stream(variant="itu", **params)
    out = [stream.feed(audio[pos:pos + 8]) for pos in range(0, audio.size, 8)]
    out.append(stream.flush())
    assert "".join(out) == rtty.demodulate(audio, variant="itu", **params) == text


def test_stream_events_report_each_frame_once_at_absolute_start_edges():
    text = "RY 123 ABC! 456\nEND" * 3
    audio = rtty.modulate(text)
    expected = rtty.characters(audio)
    decoder = ita2.Decoder()
    expected_events = [rtty.FrameEvent(sample, code, decoder.feed(code), True)
                       for sample, code in expected]
    stream = rtty.Stream()
    assert stream.events == []
    rng = np.random.default_rng(91)
    events = []
    pos = 0
    while pos < audio.size:
        size = int(rng.integers(1, fsk.FS + 1))
        value = stream.feed(audio[pos:pos + size])
        assert value == "".join(event.text for event in stream.events)
        events.extend(stream.events)
        pos += size
    value = stream.flush()
    assert value == "".join(event.text for event in stream.events)
    events.extend(stream.events)
    assert events == expected_events
    assert "".join(event.text for event in events) == text
    assert any(event.text == "" for event in events)  # Shift frames are events too.
    assert stream.flush() == ""
    assert stream.events == []
    assert stream.feed([]) == ""
    assert stream.events == []


def test_stream_events_report_broken_stop_without_changing_shift_state():
    codes = [27, 3]  # FIGS, A
    units = [True] * 10
    for code, accepted in zip(codes, [False, True]):
        units += [False] + [bool(code >> k & 1) for k in range(5)]
        units += [accepted, True]
    units += [True] * 10
    audio = fsk.modulate(units, baud=rtty.BAUD,
                         f_hi=rtty.MARK_HZ, f_lo=rtty.SPACE_HZ)
    stream = rtty.Stream()
    rng = np.random.default_rng(42)
    events = []
    pos = 0
    out = []
    while pos < audio.size:
        size = int(rng.integers(1, 2001))
        out.append(stream.feed(audio[pos:pos + size]))
        events.extend(stream.events)
        pos += size
    out.append(stream.flush())
    events.extend(stream.events)
    assert "".join(out) == rtty.demodulate(audio) == "A"
    assert [(event.code, event.text, event.accepted) for event in events] == [
        (27, "", False), (3, "A", True)]
    assert events[0].sample < events[1].sample
    assert abs(events[0].sample - round(10 * fsk.FS / rtty.BAUD)) < fsk.FS / rtty.BAUD
    assert [(event.sample, event.code) for event in events if event.accepted] == rtty.characters(audio)


def test_stream_flush_uses_finished_buffer_endpoint_decisions():
    fs = 8000
    audio = rtty.modulate("A", fs=fs)
    frames = rtty.characters(audio, fs=fs)
    # Stop centre is available, but its full centered window is not yet present.
    end = frames[-1][0] + round(6.75 * fs / rtty.BAUD)
    audio = audio[:end]
    stream = rtty.Stream(fs=fs)
    assert stream.feed(audio) == ""
    assert stream.flush() == rtty.demodulate(audio, fs=fs) == "A"
    assert stream.flush() == ""


def test_stream_memory_is_bounded_over_twenty_minutes():
    fs = 8000
    block = (0.5 * np.sin(2 * np.pi * rtty.MARK_HZ * np.arange(fs) / fs)).astype(np.float32)
    stream = rtty.Stream(fs=fs)
    for _ in range(20 * 60):
        assert stream.feed(block) == ""
        assert stream._buffer.size <= np.ceil(8.5 * fs / rtty.BAUD) + 3
        assert stream._buffer.size < 2 * fs
    assert stream.flush() == ""


def test_empty_and_short_buffers():
    for audio in ([], np.zeros(1), np.zeros(100)):
        assert rtty.characters(audio) == []
        assert rtty.demodulate(audio) == ""
    stream = rtty.Stream()
    assert stream.feed([]) == stream.flush() == ""


def test_stream_retune_preserves_figures_and_held_frame():
    text = "CQ\n12345678901234567890\n"
    samples = rtty.modulate(text)
    reader = rtty.Stream()
    cut = len(samples) // 2 + 137
    got = reader.feed(samples[:cut])
    before = (reader._next, reader._origin, reader._buffer.copy())
    reader.retune(rtty.MARK_HZ + 1, rtty.SPACE_HZ + 1)
    assert (reader._next, reader._origin) == before[:2]
    np.testing.assert_array_equal(reader._buffer, before[2])
    got += reader.feed(samples[cut:]) + reader.flush()
    assert got == text


def test_stream_reports_unknown_shift_until_a_valid_shift_frame():
    # Join in the middle of figures, then receive FIGS and LTRS explicitly.
    codes = ita2.encode('12')[2:] + [0b11011] + ita2.encode('34')[2:] + ita2.encode('AB')
    units = [True] * 10
    for code in codes:
        units += [False] + [bool(code >> k & 1) for k in range(5)] + [True, True]
    units += [True] * 10
    sent = fsk.modulate(units, baud=rtty.BAUD, f_hi=rtty.MARK_HZ, f_lo=rtty.SPACE_HZ)
    stream = rtty.Stream()
    stream.feed(sent)
    events = stream.events.copy()
    stream.flush()
    events += stream.events
    assert ''.join(e.text for e in events) == 'QW34AB'
    assert [e.shift_known for e in events if e.text] == [False, False, True, True, True, True]
    assert stream.shift_known


def test_erased_shift_frame_cannot_change_or_establish_shift():
    sent = rtty.modulate('12AB')
    frames = rtty.characters(sent)
    valid = np.ones(len(sent), dtype=bool)
    # Erase both leading shifts; otherwise clean frame/stop bits must not count.
    for edge, code in frames[:2]:
        valid[edge:edge + round(6.5 * fsk.FS / rtty.BAUD)] = False
    stream = rtty.Stream()
    events = []
    for start in range(0, len(sent), 997):
        stream.feed(sent[start:start + 997], valid=valid[start:start + 997])
        events.extend(stream.events)
    stream.flush()
    events.extend(stream.events)
    # The first data frame falls in the post-erasure framing guard.
    assert ''.join(e.text for e in events) == 'WAB'
    assert [e.shift_known for e in events if e.text] == [False, True, True]
    assert stream.rejected == 3


@pytest.mark.parametrize('prefix, suffix, restore, shift', [
    ('1234567890' * 3, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '123 RESTORED\n', 31),
    ('ABCDEF' * 3, '1234567890' * 4, 'RESTORED ABC\n', 27),
])
def test_erased_shift_revokes_certainty_until_an_accepted_shift(prefix, suffix, restore, shift):
    sent = rtty.modulate(prefix + suffix + '\n' + restore)
    index = next(i for i, code in enumerate(ita2.encode(prefix + suffix))
                 if i > 2 and code == shift)
    lo = round((10 + 7.5 * index) / rtty.BAUD * fsk.FS) - round(.04 * fsk.FS)
    hi = lo + round(.25 * fsk.FS)
    valid = np.ones(len(sent), dtype=bool)
    valid[lo:hi] = False
    sent[lo:hi] = 0
    stream = rtty.Stream()
    events = []
    for start in range(0, len(sent), fsk.FS // 4):
        stream.feed(sent[start:start + fsk.FS // 4], valid=valid[start:start + fsk.FS // 4])
        events.extend(stream.events)
    stream.flush()
    events.extend(stream.events)
    assert all(e.shift_known for e in events if e.text and e.sample < lo)
    resumed = next(i for i, e in enumerate(events)
                   if e.sample > hi and e.accepted and e.code in (27, 31))
    tentative = [e for e in events[:resumed] if e.text and e.sample > hi]
    assert tentative and all(not e.shift_known for e in tentative)
    assert all(e.shift_known for e in events[resumed:] if e.text)
    assert ''.join(e.text for e in events[resumed:]) == restore


def test_reframing_after_a_fade_never_takes_a_shift_from_tone_changes():
    """Resumed mid-character, a frame hunted from a data edge samples the next
    character, half a unit off this one's grid, on its tone changes. On shaped
    keying the near-zero values there read as a framed LTRS, and figures became
    letters."""
    sent = rtty.modulate('1234567890' * 5 + '\n')
    valid = np.ones(len(sent), dtype=bool)
    valid[3 * fsk.FS:round(3.6 * fsk.FS)] = False
    sent[~valid] = 0
    stream = rtty.Stream()
    text = ''
    for start in range(0, len(sent), fsk.FS // 4):
        text += stream.feed(sent[start:start + fsk.FS // 4], valid=valid[start:start + fsk.FS // 4])
    text += stream.flush()
    assert text.startswith('12345678901234') and text.endswith('67890\n')
    assert set(text) <= set('0123456789\n')



@pytest.mark.parametrize('data,code', [([.1] * 5, 31), ([.1, .1, 0, .1, .1], 27)])
@pytest.mark.parametrize('stop,classified', [(0, False), (.15, True), (1, True)])
def test_classifier_counts_stop_sign_without_committing_uncertain_shifts(data, code,
                                                                        stop, classified):
    unit = 10
    edge = unit
    soft = np.ones(8 * unit)
    soft[edge:edge + unit] = -.9
    centres = edge + np.round((np.arange(7) + .5) * unit).astype(int)
    soft[centres] = [-.9, *data, stop]
    accepted, rejected, _ = rtty._frames(soft, unit, 1, edge + 1, confidence=0)
    assert accepted == ([(edge, code)] if classified else [])
    assert rejected == ([] if classified else [(edge, code)])
    # The decoder still declines these ambiguous LTRS/FIGS units, even when
    # the stop itself is strong enough to establish statistical framing.
    accepted, rejected, _ = rtty._frames(soft, unit, 1, edge + 1)
    assert accepted == [] and rejected == [(edge, code)]


def test_unshift_on_space_changes_provisional_case_without_restoring_certainty():
    rate = 6000

    def framed(codes):
        units = [True] * 10
        for code in codes:
            units += [False] + [bool(code >> k & 1) for k in range(5)] + [True, True]
        return fsk.modulate(units, baud=rtty.BAUD, f_hi=1585, f_lo=1415, fs=rate)

    first = framed([27, 1])  # FIGS, E (3 in figures)
    second = framed([4, 1])  # SPACE, E
    gap = np.zeros(round(10 * rate / rtty.BAUD))
    samples = np.r_[first, gap, second]
    valid = np.ones(len(samples), dtype=bool)
    valid[len(first):len(first) + len(gap)] = False
    reader = rtty.Stream(fs=rate)
    events = []
    for start in range(0, len(samples), rate // 4):
        reader.feed(samples[start:start + rate // 4], valid=valid[start:start + rate // 4])
        events.extend(reader.events)
    reader.flush()
    events.extend(reader.events)
    text = [event for event in events if event.text]
    assert ''.join(event.text for event in text) == '3 E'
    assert [event.shift_known for event in text] == [True, False, False]


RATES = ['45.45/170', '50/170', '75/170', '50/425', '50/450', '50/850', '75/850']


def _typist(script, fs=fsk.FS, **kw):
    """A transmitter handed each (sample, text) of `script` once it has reached it."""
    tx = rtty.Transmitter(kw.pop('rate', rtty.AMATEUR), kw.pop('mark', 2295), kw.pop('space', 2125),
                          fs=fs, pull=lambda: (script.pop(0)[1] if script
                                               and tx.position >= script[0][0] else ''), **kw)
    return tx


@pytest.mark.parametrize('usos', [True, False])
@pytest.mark.parametrize('rate', RATES)
def test_transmitter_round_trip_with_typing_and_idle_diddle(rate, usos):
    fs = 8000
    rate = rtty.Rate.parse(rate)
    mark, space = 2210 + rate.shift / 2, 2210 - rate.shift / 2
    rng = np.random.default_rng(int(rate.baud * rate.shift))
    text = 'CQ DE W9SSJ 599 073 EN63 K\nR 5NN 12 ABC 34/5 73\n'
    cuts = sorted(rng.choice(np.arange(1, len(text)), 6, replace=False))
    pieces = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
    at = np.cumsum(rng.integers(0, round(1.5 * fs), len(pieces)))
    script = list(zip(at.tolist(), pieces))
    tx = _typist(script, fs, rate=rate, mark=mark, space=space)
    stream = rtty.Stream(baud=rate.baud, mark_hz=mark, space_hz=space, fs=fs,
                         unshift_on_space=usos)
    heard = []
    while tx.end is None or tx.position < tx.end:
        if not script:
            tx.finish()
        heard.append(stream.feed(tx.render(int(rng.integers(1, fs // 2)))))
    heard.append(stream.feed(tx.render(fs // 2)) + stream.flush())
    assert ''.join(heard) == text
    assert stream.rejected == 0


def test_idle_keying_is_letters_diddle():
    tx = rtty.Transmitter(rtty.AMATEUR, rtty.MARK_HZ, rtty.SPACE_HZ)
    codes = [code for _, code in rtty.characters(tx.render(3 * fsk.FS))]
    assert len(codes) >= 15 and set(codes) == {31}


def test_render_boundaries_are_phase_and_frequency_continuous():
    def run(sizes):
        pulls = ['RYRY 123 TEST', '', '', 'DE W9SSJ']
        tx = rtty.Transmitter(rtty.AMATEUR, 2295, 2125, pull=lambda: pulls.pop(0) if pulls else '')
        return np.concatenate([tx.render(n) for n in sizes])

    rng = np.random.default_rng(3)
    sizes = rng.integers(0, 3000, 300).tolist()
    whole = run([sum(sizes)])
    np.testing.assert_allclose(run(sizes), whole, atol=1e-4)
    step = 2 * np.pi * 2295 / fsk.FS * 0.5
    assert np.max(np.abs(np.diff(whole))) <= step * 1.001


def test_finish_sends_the_tail_ramps_down_and_falls_silent():
    pending = ['TEST']
    tx = rtty.Transmitter(rtty.AMATEUR, 2295, 2125, pull=lambda: pending.pop() if pending else '')
    tx.finish()
    audio = tx.render(3 * fsk.FS)
    assert tx.end is not None and tx.end < 3 * fsk.FS
    assert abs(audio[0]) < 1e-3
    assert np.all(audio[tx.end:] == 0)
    assert np.max(np.abs(audio[tx.end - 10:tx.end])) < .01
    unit = fsk.FS / rtty.BAUD
    assert tx.end == round((2 * rtty.LEAD_TAIL_UNITS + 5 * 7.5) * unit)
    assert rtty.demodulate(audio, mark_hz=2295, space_hz=2125) == 'TEST'


def _occupied(audio, fs, fraction=.99):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1 / fs)
    cumulative = np.cumsum(spectrum) / spectrum.sum()
    lo = freqs[np.searchsorted(cumulative, (1 - fraction) / 2)]
    hi = freqs[np.searchsorted(cumulative, 1 - (1 - fraction) / 2)]
    return hi - lo


def test_occupied_bandwidth_is_pinned_and_narrower_than_hard_keying():
    rng = np.random.default_rng(5)
    text = ''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789'), 300))
    shaped = rtty.modulate(text, mark_hz=2295, space_hz=2125)
    units = [True] * 10
    for code in ita2.encode(text):
        units += [False] + [bool(code >> k & 1) for k in range(5)] + [True]
    hard = fsk.modulate(units, baud=rtty.BAUD, f_hi=2295, f_lo=2125)
    # First measured 2026-10-10: 236 Hz at 99% and 338 Hz at 99.9%, against
    # 250 Hz and 492 Hz for the same characters keyed hard.
    assert _occupied(shaped, fsk.FS) < 240
    assert _occupied(shaped, fsk.FS, .999) < 345
    assert _occupied(hard, fsk.FS, .999) > 450


def test_rate_parses_baud_shift_and_stop():
    assert rtty.Rate.parse('45.45/170') == rtty.AMATEUR == (45.45, 170, 1.5)
    assert rtty.Rate.parse('50/850/1') == (50, 850, 1)
    assert str(rtty.Rate.parse('75/850')) == '75/850'
    for bad in ('45.45', 'fast', '50/0', '50/170/1/2'):
        with pytest.raises(ValueError):
            rtty.Rate.parse(bad)
