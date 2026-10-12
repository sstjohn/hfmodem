# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Mode B slot layout, round trips under offset and noise, and time diversity."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.lanner import ccir476, fsk, ita2, rtty, sitor

TEXT = "DE TEST 01234\nOK"

PH1 = ccir476.PHASING_1
PH2 = ccir476.PHASING_2


@pytest.mark.parametrize('fs', [6000, 48000])
@pytest.mark.parametrize('start', [2.24, 5, 10])
def test_midstream_traffic_signature_without_phasing(fs, start):
    samples = sitor.modulate('LONG BULLETIN 0123456789\n' * 12, fs=fs)
    samples = samples[round(start * fs):round((start + 3) * fs)]
    soft = fsk.discriminate(samples, f_hi=sitor.F_B, f_lo=sitor.F_Y, baud=100, fs=fs)
    assert sitor.traffic_signature(soft, fs=fs)


@pytest.mark.parametrize('text', ['RYRYRY' * 8, '1234567890' * 5,
                                  'LONG BULLETIN 0123456789\n' * 2])
def test_traffic_signature_does_not_claim_async_rtty(text):
    fs = 6000
    samples = rtty.modulate(text, fs=fs)
    for start in range(0, len(samples), fs):
        soft = fsk.discriminate(samples[start:start + 3 * fs], f_hi=1585, f_lo=1415,
                                baud=100, fs=fs)
        assert not sitor.traffic_signature(soft, fs=fs)


def test_constant_ratio_word_grid_is_evidence_even_with_jittered_transitions():
    words = sitor.words('LONG BULLETIN 0123456789\n' * 2, phasing_pairs=0)
    units = [unit for word in words for unit in ccir476.units(word)]
    soft = np.repeat(np.where(units, 1., -1.), 60)
    edges = np.flatnonzero(np.diff(soft > 0)) + 1
    for i, edge in enumerate(edges):
        # Alternating 1/3-unit displacement destroys the transition-clock test
        # while leaving every word's centre decisions intact on the known grid.
        if i % 2:
            soft[edge:edge + 20] = soft[edge - 1]
        else:
            soft[edge - 20:edge] = soft[edge]
    edges = np.flatnonzero(np.diff(soft > 0))
    assert abs(np.mean(np.exp(2j * np.pi * edges / 60))) < .85
    assert sitor.traffic_signature(soft, fs=6000, centres=np.arange(30, len(soft), 60))


def _tx(words):
    units = [u for w in words for u in ccir476.units(w)]
    return fsk.modulate(units, baud=sitor.BAUD, f_hi=sitor.F_B, f_lo=sitor.F_Y)


def test_slot_layout():
    w = sitor.words(TEXT, phasing_pairs=0)
    for k, d in enumerate([ccir476.from_ita2(c) for c in ita2.encode(TEXT)]):
        assert w[2 * k] == d
        assert w[2 * k + 5] == d
    w = sitor.words(TEXT, phasing_pairs=6)
    for j in range(12):
        assert w[j] == (PH2 if j % 2 == 0 else PH1)
    assert [w[j] for j in range(len(w) - 6, len(w) - 1, 2)] == [PH1, PH1, PH1]


def test_clean_round_trip():
    assert sitor.demodulate(sitor.modulate(TEXT)) == TEXT


def test_round_trip_sample_offset_and_tone_offset():
    rng = np.random.default_rng(11)
    pad = int(rng.integers(1, fsk.FS // sitor.BAUD))
    for off in (5.0, -5.0):
        units = [u for w in sitor.words(TEXT) for u in ccir476.units(w)]
        audio = fsk.modulate(units, baud=sitor.BAUD, f_hi=sitor.F_B + off, f_lo=sitor.F_Y + off)
        assert sitor.demodulate(np.concatenate((np.zeros(pad), audio))) == TEXT


def test_round_trip_noise_10db():
    audio = sitor.modulate(TEXT).astype(np.float64)
    rng = np.random.default_rng(7)
    spec = np.fft.rfft(rng.standard_normal(audio.size))
    spec[np.fft.rfftfreq(audio.size, 1 / sitor.FS) > 3000] = 0
    noise = np.fft.irfft(spec, audio.size)
    noise *= np.sqrt(np.mean(audio**2) / np.mean(noise**2) / 10)
    assert sitor.demodulate(audio + noise) == TEXT


def test_time_diversity_recovers_corrupted_dx():
    w = sitor.words(TEXT)
    n = len(ita2.encode(TEXT))
    for k in range(n):
        w[2 * (16 + k)] ^= 1
    assert sitor.demodulate(_tx(w)) == TEXT


def test_both_copies_mutilated_prints_underscore():
    w = sitor.words(TEXT)
    k = 16 + 1       # the traffic word for the "D" of "DE"
    w[2 * k] ^= 1
    w[2 * k + 5] ^= 1
    assert sitor.demodulate(_tx(w)) == "_E TEST 01234\nOK"


def test_no_phasing_returns_empty():
    assert sitor.demodulate(sitor.modulate(TEXT, phasing_pairs=0)) == ""


def _noise(rng, n, signal, snr_db):
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / sitor.FS) > 3000] = 0
    noise = np.fft.irfft(spec, n)
    return noise * np.sqrt(np.mean(signal ** 2) / np.mean(noise ** 2) / 10 ** (snr_db / 10))


def test_ten_minutes_decode_exact_under_clock_error():
    rng = np.random.default_rng(31)
    text = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")) for _ in range(2600))
    words = sitor.words(text)
    units = [u for w in words for u in ccir476.units(w)]
    audio = fsk.modulate(units, baud=sitor.BAUD, f_hi=sitor.F_B, f_lo=sitor.F_Y,
                         fs=sitor.FS * (1 + 40e-6))
    audio = np.concatenate((audio, np.zeros(sitor.FS * 600 - audio.size)))
    assert sitor.demodulate(audio) == text


def test_two_transmissions_separated_by_noise_both_decode():
    t1, t2 = "DE ALPHA STATION", "DE BRAVO STATION"
    rng = np.random.default_rng(32)
    gap = _noise(rng, round(20 * sitor.FS), 0.5 * np.ones(round(20 * sitor.FS)), 0)
    audio = np.concatenate((sitor.modulate(t1), gap, sitor.modulate(t2)))
    assert sitor.demodulate(audio) == t1 + "\n" + t2


def test_deep_fade_costs_at_most_the_characters_inside_it():
    text = "DE CHARLIE " * 80          # letters only: one word per character
    nchar = len(text)
    rng = np.random.default_rng(33)
    audio = sitor.modulate(text).astype(np.float64)
    noise = _noise(rng, audio.size, audio, 26)
    start, end = audio.size // 2 - round(1.5 * sitor.FS), audio.size // 2 + round(1.5 * sitor.FS)
    audio[start:end] *= 0.01
    got = sitor.demodulate(audio + noise)
    period = sitor.FS / sitor.BAUD
    dx_start = 2 * (17 + np.arange(nchar)) * 7 * period
    rx_end = dx_start + 6 * 7 * period
    affected = np.flatnonzero((dx_start < end + period) & (rx_end > start - period))
    first, last = affected[0], affected[-1] + 1
    assert got[:first] == text[:first]
    assert got[-(nchar - last):] == text[last:]
    assert nchar - (last - first) <= len(got) <= nchar
    assert got != text


def test_word_sync_maintenance_recovers_a_unit_slip(monkeypatch):
    text = "DELTA FOXTROT " * 60
    slip_word = int(len(sitor.words(text)) * 0.15)
    words = sitor.words(text)
    units = [u for w in words for u in ccir476.units(w)]
    audio = fsk.modulate(units, baud=sitor.BAUD, f_hi=sitor.F_B, f_lo=sitor.F_Y)
    cut = round(slip_word * 7 * sitor.FS / sitor.BAUD)
    assert cut > 32 * 7 * sitor.FS / sitor.BAUD
    slipped = np.concatenate((audio[:cut], audio[cut + int(sitor.FS // sitor.BAUD):]))
    got = sitor.demodulate(slipped)
    assert got[:slip_word // 2 - 20] == text[:slip_word // 2 - 20]
    assert got[-300:] == text[-300:]
    monkeypatch.setattr(sitor, "_realign", lambda hard, off, j: off)
    without_realign = sitor.demodulate(slipped)
    assert without_realign[-300:] != text[-300:]


def test_soft_combining_recovers_complementary_weak_bit_errors():
    for code in range(32):
        word = ccir476.from_ita2(code)
        sent = np.where(ccir476.units(word), 1.0, -1.0)
        decisions = np.zeros(42)
        decisions[:7] = sent
        decisions[35:42] = sent
        decisions[0] *= -0.1
        decisions[36] *= -0.1
        dx, rx = word ^ 1, word ^ 2
        assert not ccir476.valid(dx) and not ccir476.valid(rx)
        expected = ita2.Decoder().feed(code)
        assert sitor._combine(decisions, 0, 0, dx, rx, ita2.Decoder()) == expected


def test_word_sync_recovers_a_late_unit_slip(monkeypatch):
    rng = np.random.default_rng(61)
    text = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "), size=2550))
    words = sitor.words(text)
    units = np.array([u for w in words for u in ccir476.units(w)])
    assert 350 < units.size / sitor.BAUD < 370
    slip_word = 2 * (int(len(words) * 0.8) // 2)
    decisions = np.where(np.delete(units, slip_word * 7), 1.0, -1.0)
    monkeypatch.setattr(fsk, "discriminate", lambda *a, **kw: decisions)
    monkeypatch.setattr(fsk, "track", lambda *a, **kw: np.arange(decisions.size))
    got = sitor.demodulate(np.empty(0))
    slip_char = slip_word // 2 - 17  # preamble and initial letters shift
    assert got[:slip_char - 3] == text[:slip_char - 3]
    assert len(got) == len(text)
    assert got[slip_char + 20:] == text[slip_char + 20:]


def test_transmissions_phasing_scan_cost_is_bounded(monkeypatch):
    texts = [f"DE STATION {i} " + "ALPHA BRAVO " * 10 for i in range(5)]
    fs = 12000
    audio = np.concatenate([part for text in texts for part in
                            (sitor.modulate(text, fs=fs), np.zeros(fs))])
    examined = None
    original = sitor.find_phasing

    class CountedStream(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, slice):
                examined[key] += 1
            return super().__getitem__(key)

    def counted_find_phasing(hard, pos):
        nonlocal examined
        if examined is None:
            examined = np.zeros(hard.size, dtype=np.int32)
        return original(hard.view(CountedStream), pos)

    monkeypatch.setattr(sitor, "find_phasing", counted_find_phasing)
    assert [text for _, text in sitor.transmissions(audio, fs=fs)] == texts
    # Allow revisits from chunk overlap and lookahead into the next emission.
    assert examined.sum() <= 28 * examined.size
    assert examined.max() <= 42


def test_find_phasing_across_chunk_boundary_and_after_pos():
    units = np.zeros(400 * 7, dtype=bool)
    preamble = np.array([u for w in [PH2, PH1] * 4 for u in ccir476.units(w)])
    for start in (247 * 7, 249 * 7 + 3):
        hard = units.copy()
        hard[start:start + preamble.size] = preamble
        found = sitor.find_phasing(hard, 13)
        assert found is not None
        assert found[2] == start
        assert found[0] + 7 * found[1] == start
        assert sitor.find_phasing(hard, start + 1) is None


def test_resume_preserves_shifts_fec_and_adjacent_locks():
    rate = 6000
    first = "1234567890" * 25 + "\nFIRST END\n"
    second = "SECOND 9876543210\n" * 20
    audio = np.concatenate((sitor.modulate(first, fs=rate * (1 + 40e-6)),
                            sitor.modulate(second, fs=rate * (1 + 40e-6)), np.zeros(rate)))
    expected = sitor.transmissions(audio, fs=rate)
    resume = sitor.Resume()
    for end in range(10 * rate, len(audio), 10 * rate):
        origin = max(0, round(resume.sample) - 12 * rate)
        sitor.transmissions(audio[origin:end], fs=rate, resume=resume,
                            origin=origin, final=False)
    origin = max(0, round(resume.sample) - 12 * rate)
    actual = sitor.transmissions(audio[origin:], fs=rate, resume=resume, origin=origin)
    assert [text for _, text in actual] == [text for _, text in expected] == [first, second]
    assert [index + round(origin * sitor.BAUD / rate) for index, _ in actual] == [
        index for index, _ in expected]


@pytest.mark.parametrize("size", [0, 1, 59, 6001, 18000, 24000])
@pytest.mark.parametrize("final", [False, True])
def test_empty_and_silent_input_decodes_nothing(size, final):
    audio = np.zeros(size)
    with np.errstate(all="raise"):
        assert sitor.transmissions(audio, fs=6000, final=final) == []
        resume = sitor.Resume()
        assert sitor.transmissions(audio, fs=6000, resume=resume, final=final) == []
        assert sitor.transmissions([], fs=6000, resume=resume) == []


def test_short_recordings_at_every_symbol_sample_offset():
    audio = sitor.modulate("A", phasing_pairs=4, fs=6000)
    for offset in range(60):
        shifted = np.concatenate((np.zeros(offset), audio))
        assert sitor.demodulate(shifted, fs=6000) == "A"
        assert [text for _, text in sitor.transmissions(shifted[:6001], fs=6000)] == ["A"]
        assert sitor.transmissions(shifted[:4500], fs=6000, resume=sitor.Resume()) == []
    clipped = sitor.modulate("TEST", phasing_pairs=4, fs=6000)[:-31]
    assert sitor.demodulate(clipped, fs=6000) == "TEST"


@pytest.mark.parametrize("replacement", [ccir476.from_ita2(25),
    ccir476.word([c == "B" for c in ccir476.SERVICE_SIGNALS["idle_beta"]])])
def test_conflicting_valid_copies_are_mutilated_even_with_unequal_strengths(replacement):
    a = ccir476.from_ita2(3)
    for dx, rx in ((a, replacement), (replacement, a)):
        decisions = np.zeros(42)
        decisions[:7] = np.where(ccir476.units(dx), 0.9, -0.9)
        decisions[35:42] = np.where(ccir476.units(rx), 0.3, -0.3)
        assert sitor._combine(decisions, 0, 0, dx, rx, ita2.Decoder()) == "_"
    words = sitor.words("AB")
    words[2 * 17 + 5] = replacement
    assert sitor.demodulate(_tx(words)) == "_B"


@pytest.mark.parametrize("copies", [(True, True), (True, False), (False, True)])
def test_agreeing_or_single_valid_traffic_copy_decodes(copies):
    words = sitor.words("A")
    if not copies[0]:
        words[2 * 17] ^= 1
    if not copies[1]:
        words[2 * 17 + 5] ^= 1
    assert sitor.demodulate(_tx(words)) == "A"


def test_agreeing_idle_beta_copies_are_filler_between_traffic():
    beta = ccir476.word([c == "B" for c in ccir476.SERVICE_SIGNALS["idle_beta"]])
    dx = ([PH2] * 16 + [ccir476.from_ita2(c) for c in ita2.encode("A")]
          + [beta, ccir476.from_ita2(25)] + [PH1] * 3)
    slots = []
    for k, word in enumerate(dx):
        delayed = dx[k - 2] if k >= 2 else PH1
        slots.extend((word, PH1 if delayed == PH2 else delayed))
    assert sitor.demodulate(sitor._tx(slots, fs=6000), fs=6000) == "AB"


def test_finalization_decodes_received_dx_without_padding():
    rate = 6000
    audio = sitor.modulate("TEST", phasing_pairs=4, fs=rate)[:-28 * 60]
    resume = sitor.Resume()
    got = sitor.transmissions(audio, fs=rate, resume=resume, final=True)
    assert [text for _, text in got] == ["TEST"]
    assert max(resume.char_samples[0]) < len(audio)


@pytest.mark.parametrize('overlapping', [False, True])
def test_final_dx_lost_shift_revokes_certainty(overlapping):
    text = 'PREFIX ABCDEF123'
    slot = 2 * (16 + ita2.encode(text).index(27))
    words = sitor.words(text)
    words[slot] ^= 1
    # EOF precedes the damaged FIGS word's RX copy, after two more DX words.
    audio = sitor._tx(words, fs=6000)[:(slot + 5) * 7 * 60]
    resume = sitor.Resume()
    if overlapping:
        sitor.transmissions(audio[:-600], fs=6000, resume=resume, final=False)
    [(sample, decoded)] = sitor.transmissions(audio, fs=6000, resume=resume)
    assert decoded == 'PREFIX ABCDEFQW'
    assert resume.char_shift_known == [[True] * 13 + [False] * 2]
    assert len(resume.char_samples[0]) == len(decoded)


@pytest.mark.parametrize("received", range(8))
def test_final_tail_requires_a_complete_valid_traffic_dx(received):
    # Four phasing pairs and a letters shift establish the traffic grid.
    words = sitor.words("AB", phasing_pairs=4)
    rate = 6000
    # A is DX slot 10. No delayed RX copy is present in any of these clips.
    audio = sitor._tx(words, fs=rate)[:(10 * 7 + received) * 60]
    expected = ["A"] if received == 7 else []
    assert [text for _, text in sitor.transmissions(audio, fs=rate)] == expected
    if received == 7:
        for replacement in (words[10] ^ 1, PH2,
                ccir476.word([c == "B" for c in ccir476.SERVICE_SIGNALS["idle_beta"]])):
            words[10] = replacement
            audio = sitor._tx(words, fs=rate)[:(10 * 7 + received) * 60]
            assert sitor.transmissions(audio, fs=rate) == []


def test_sitor_figures_shift_survives_space_in_batch_and_resume():
    # An independent sender stays in FIGS: 1, 2, SPACE, 3, 4, with no reshifts.
    codes = (27,) + (23, 19) * 15 + (4,) + (1, 10) * 20
    dx = [PH2] * 4 + [ccir476.from_ita2(c) for c in codes] + [PH1] * 3
    slots = []
    for k, word in enumerate(dx):
        delayed = dx[k - 2] if k >= 2 else PH1
        slots.extend((word, PH1 if delayed == PH2 else delayed))
    audio = sitor._tx(slots, fs=6000)
    assert sitor.demodulate(audio, fs=6000) == "12" * 15 + " " + "34" * 20
    resume = sitor.Resume()
    # Pause after SPACE, retaining the live cursor’s three seconds of lookahead.
    end = (36 * 14 + 300) * 60
    sitor.transmissions(audio[:end], fs=6000, resume=resume, final=False)
    assert resume.texts[-1][1] == "12" * 15 + " "
    assert [text for _, text in sitor.transmissions(audio, fs=6000, resume=resume)] == [
        "12" * 15 + " " + "34" * 20]


@pytest.mark.parametrize('left', [0, 8])
def test_fitted_positions_locate_phasing_and_characters_in_overlapping_windows(left):
    rate = 6000
    at = 19.991
    audio = np.zeros(35 * rate)
    sent = sitor.modulate('ABC', phasing_pairs=4, fs=rate)
    start = round(at * rate)
    audio[start:start + len(sent)] = sent
    resume = sitor.Resume()
    result = sitor.transmissions(audio[left * rate:], fs=rate, origin=left * rate,
                                resume=resume, sample_positions=True)
    assert len(result) == 1
    sample, text = result[0]
    assert text == 'ABC'
    assert sample / rate == pytest.approx(at + 0.005, abs=0.002)
    assert sample < 20 * rate
    assert np.array(resume.char_samples[0]) / rate == pytest.approx(
        [at + 0.005 + (5 + i) * 0.14 for i in range(3)], abs=0.002)


@pytest.mark.parametrize('duration', [.25, 1, 2, 2.5])
def test_resume_holds_fec_and_shift_through_bounded_erasures(duration):
    rate = 6000
    text = '1234567890' * 25
    audio = sitor.modulate(text, fs=rate)
    audio[10 * rate:round((10 + duration) * rate)] = 0
    resume = sitor.Resume(max_fade_s=2)
    for end in range(4 * rate, len(audio), rate):
        origin = max(0, round(resume.sample) - 12 * rate)
        sitor.transmissions(audio[origin:end], fs=rate, resume=resume,
                            origin=origin, final=False)
    origin = max(0, round(resume.sample) - 12 * rate)
    result = sitor.transmissions(audio[origin:], fs=rate, resume=resume, origin=origin)
    assert len(result) == 1
    decoded = result[0][1]
    assert decoded.startswith('1234567890' * 4)
    if duration <= 2:
        assert decoded.endswith('1234567890' * 10)
        assert resume.emission_finished
        if duration == .25:
            assert decoded == text
    else:
        assert len(decoded) < len(text) // 2
        assert not resume.emission_finished


@pytest.mark.parametrize('dtype', [np.int16, np.int32])
def test_fade_detection_accepts_integer_pcm_without_overflow(dtype):
    text = 'INTEGER AUDIO TEST 1234'
    pcm = (sitor.modulate(text, fs=6000) * 32767).astype(dtype)
    assert [value for _, value in sitor.transmissions(
        pcm, fs=6000, resume=sitor.Resume(max_fade_s=2))] == [text]


@pytest.mark.parametrize('size', [0, 1, 20, 59, 60])
def test_fade_detection_accepts_short_fragments(size):
    assert sitor.transmissions(np.zeros(size, dtype=np.int16), fs=6000,
                               resume=sitor.Resume(max_fade_s=2)) == []


@pytest.mark.parametrize('overlapping', [False, True])
def test_fade_reference_belongs_to_each_emission(overlapping):
    from hfmodem.lanner import navtex

    rate = 6000
    first = navtex.Message('Q', 'A', 17, 'LOUD BULLETIN')
    second = navtex.Message('Q', 'A', 20, 'WEAK BULLETIN')
    audio = np.r_[navtex.modulate(first, fs=rate), np.zeros(rate),
                  .05 * navtex.modulate(second, fs=rate)]
    resume = sitor.Resume(max_fade_s=2)
    if overlapping:
        for end in range(4 * rate, len(audio), rate):
            origin = max(0, round(resume.sample) - 12 * rate)
            sitor.transmissions(audio[origin:end], fs=rate, resume=resume,
                                origin=origin, final=False)
    origin = max(0, round(resume.sample) - 12 * rate)
    decoded = navtex.demodulate(audio[origin:], fs=rate, resume=resume, origin=origin)
    assert decoded == [first, second]


@pytest.mark.parametrize('overlapping', [False, True])
def test_erased_diversity_shift_revokes_case_until_a_recovered_shift(overlapping):
    from hfmodem.lanner import navtex

    rate = 6000
    body = 'POSITION ' + 'ABCDEF' * 8 + '1234567890' * 4 + ' RESTORED TEXT'
    text = navtex.frame(navtex.Message('Q', 'A', 17, body))
    index = [i for i, code in enumerate(ita2.encode(text)) if code == 27][-1]
    at = (16 + index) * .14 - .04
    audio = sitor.modulate(text, fs=rate)
    audio[round(at * rate):round((at + .43) * rate)] = 0
    resume = sitor.Resume(max_fade_s=2)
    if overlapping:
        for end in range(4 * rate, len(audio), rate):
            origin = max(0, round(resume.sample) - 12 * rate)
            sitor.transmissions(audio[origin:end], fs=rate, resume=resume,
                                origin=origin, final=False)
    origin = max(0, round(resume.sample) - 12 * rate)
    [(sample, decoded)] = sitor.transmissions(audio[origin:], fs=rate,
                                             resume=resume, origin=origin)
    certainty = resume.char_shift_known[0]
    assert len(certainty) == len(decoded) == len(resume.char_samples[0])
    garble = decoded.index('_')
    restored = decoded.index('RESTORED TEXT')
    assert all(certainty[:garble])
    assert not any(certainty[garble:restored - 1])
    assert 'QWERTYUIOP' * 4 in decoded
    assert all(certainty[restored:])



def test_acquired_word_activity_requires_evidence_above_random_words():
    rows = np.array([np.where(ccir476.units(word), 1., -1.)
                     for word in sitor.words('CARRIER ACTIVITY')])
    assert sitor.word_activity(np.r_[np.zeros(28), rows.ravel()])
    assert not sitor.word_activity(np.ones(100))
    random = np.random.default_rng(37).choice([-1., 1.], 700)
    assert not sitor.word_activity(random)
