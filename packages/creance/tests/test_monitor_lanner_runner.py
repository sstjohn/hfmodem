# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Synthetic audio through the runner's actual stdin/JSON contract."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from creance.monitor.activity import Detection
from creance.monitor.runners import lanner_runner as runner
from hfmodem.lanner import fsk, ita2, navtex, rtty, sitor
from hfmodem.monitor import FS


class Reads(io.BytesIO):
    def __init__(self, raw, sizes):
        super().__init__(raw)
        self.sizes = iter(sizes)

    def read(self, size=-1):
        return super().read(min(size, next(self.sizes, size)))


def run(monkeypatch, audio, centre=None, sizes=(), *, status=False, args=()):
    raw = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    out = io.StringIO()
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=Reads(raw, sizes)))
    monkeypatch.setattr(runner.sys, "stdout", out)
    argv = [] if centre is None else ["--centre", str(centre), "--bw", "700"]
    assert runner.main([*argv, *args]) == 0
    return [json.loads(line) for line in out.getvalue().splitlines()
            if "heard" not in json.loads(line)
            and (status or json.loads(line).get("kind") != "emission-status")]


def audio(text, midpoint=1500):
    return rtty.modulate(text, mark_hz=midpoint + 85, space_hz=midpoint - 85)


def test_off_nominal_rtty_in_a_track(monkeypatch):
    sent = "CQ 73\nTEST\n"
    records = run(monkeypatch, audio(sent, 2240), 2200)
    assert "".join(r["text"] for r in records) == sent
    assert len(records) == 2
    for record in records:
        det = Detection.from_json(record)
        assert (det.protocol, det.kind, det.grade) == ("RTTY", "TEXT", "confirmed")
        assert det.detail == "45.45 Bd 170 Hz"
        # A short FSK spectrum's peak can lie several Hz off the carrier.
        assert det.offset_hz == pytest.approx(740, abs=10)
        assert det.centre_hz == pytest.approx(2240, abs=10)


def test_navtex_and_no_rtty_garbage(monkeypatch):
    message = navtex.Message("Q", "A", 17, "TEST")
    records = run(monkeypatch, navtex.modulate(message, phasing_pairs=4), 1700)
    assert len(records) == 1
    det = Detection.from_json(records[0])
    assert (det.protocol, det.kind, det.station, det.detail, det.text, det.grade) == (
        "NAVTEX", "MESSAGE", "Q", "A 17", "TEST", "confirmed")
    assert det.centre_hz == pytest.approx(1700, abs=10)


@pytest.mark.parametrize("centre", [None, 1500])
def test_noise_emits_nothing(monkeypatch, centre):
    noise = np.random.default_rng(1006).normal(0, 0.03, 3 * FS)
    assert run(monkeypatch, noise, centre) == []


def test_two_tracks_only_read_their_own_text(monkeypatch):
    one, two = audio("ONE 12\n", 1000), audio("TWO 34\n", 2300)
    mixed = np.zeros(max(len(one), len(two)))
    mixed[:len(one)] += one * 0.6
    mixed[:len(two)] += two * 0.6
    for centre, text in [(1000, "ONE 12\n"), (2300, "TWO 34\n")]:
        records = run(monkeypatch, mixed, centre)
        assert "".join(r["text"] for r in records) == text
        assert {r["protocol"] for r in records} == {"RTTY"}


def test_odd_reads_preserve_framing_shifts_and_sample_clock(monkeypatch):
    samples = audio("A12\nB34\n", 1947)
    expected = run(monkeypatch, samples, 1900)
    sizes = np.random.default_rng(8).integers(1, 9600, size=1000).tolist()
    assert run(monkeypatch, samples, 1900, sizes) == expected


def test_pause_emits_without_waiting_for_eof(monkeypatch):
    sent = audio("CQ")
    records = []
    monkeypatch.setattr(runner, "_emit", lambda r: records.append((engine.n / FS, r)))
    engine = runner._AcquisitionManager(None, 700)
    samples = np.concatenate((sent, np.zeros(4 * FS)))
    for start in range(0, len(samples), FS // 4):
        engine.push(samples[start:start + FS // 4])
    [(when, record)] = records
    assert record["text"] == "CQ"
    assert when <= len(sent) / FS + 2.25


def test_control_only_stretches_are_silent(monkeypatch):
    assert run(monkeypatch, audio(" \n\n ")) == []


def test_plain_sitor_text(monkeypatch):
    records = run(monkeypatch, sitor.modulate("CQ TEST", phasing_pairs=4), 1700)
    assert [(r["protocol"], r["text"]) for r in records] == [("SITOR-B", "CQ TEST")]


def test_final_sitor_dx_lost_shift_is_tentative(monkeypatch):
    from scipy.signal import resample_poly

    text = 'PREFIX ABCDEF123'
    slot = 2 * (16 + ita2.encode(text).index(27))
    words = sitor.words(text)
    words[slot] ^= 1
    low = sitor._tx(words, fs=6000)[:(slot + 5) * 7 * 60]
    records = run(monkeypatch, resample_poly(low, 8, 1), 1700)
    assert [(r['protocol'], r['text'], r['grade']) for r in records] == [
        ('SITOR-B', 'PREFIX ABCDEFQW', 'tentative')]


def test_rejected_stop_bits_are_counted_without_changing_stream_text():
    units = [True] * 10
    for code in ita2.encode("TEST"):
        units += [False] + [bool(code >> k & 1) for k in range(5)] + [False, False]
    units += [True] * 10
    samples = fsk.modulate(units, baud=rtty.BAUD, f_hi=1585, f_lo=1415)
    reader = rtty.Stream(mark_hz=1585, space_hz=1415, fs=FS)
    text = reader.feed(samples) + reader.flush()
    assert text == rtty.demodulate(samples)
    assert reader.rejected > 0


def test_navtex_survives_decode_windows_without_partial_messages(monkeypatch):
    monkeypatch.setattr(runner, "BLOCK_S", 2.0)
    monkeypatch.setattr(runner, "LOOKBACK_S", 1.0)
    message = navtex.Message("Q", "B", 5, "TEST")
    records = run(monkeypatch, navtex.modulate(message, phasing_pairs=4), 1700)
    assert [(r["protocol"], r["station"], r["detail"], r["text"]) for r in records] == [
        ("NAVTEX", "Q", "B 05", "TEST")]


def test_rtty_resumes_after_sitor_on_the_same_track(monkeypatch):
    samples = np.concatenate((sitor.modulate("TEST", phasing_pairs=4),
                              np.zeros(FS // 2), audio("CQ\n", 1700)))
    records = run(monkeypatch, samples, 1700)
    assert sorted((r["protocol"], r["text"]) for r in records) == [
        ("RTTY", "CQ\n"), ("SITOR-B", "TEST")]


def test_rejected_frame_after_newline_lowers_the_next_lines_grade(monkeypatch):
    engine = runner._DecoderLane(None, 700)
    engine.midpoint = 1500
    engine.reader = rtty.Stream(mark_hz=1585, space_hz=1415, fs=FS)
    engine.n = 3 * FS
    engine.reader.events = [rtty.FrameEvent(100, 3, "A", True),
                            rtty.FrameEvent(200, 2, "\n", True),
                            rtty.FrameEvent(300, 0, "", False),
                            rtty.FrameEvent(400, 25, "B", True),
                            rtty.FrameEvent(500, 2, "\n", True)]
    records = []
    monkeypatch.setattr(runner, "_emit", records.append)
    engine.consume()
    engine.release()
    assert [(r["text"], r["grade"]) for r in records] == [
        ("A\n", "confirmed"), ("B\n", "tentative")]


def test_an_accidental_shift_cannot_clear_rejected_framing_history(monkeypatch):
    engine = runner._DecoderLane(None, 700)
    engine.midpoint = 1500
    engine.reader = rtty.Stream()
    engine.n = 30 * FS
    attempts = [(0, '', False, False)] * 16
    attempts += [(31, '', True, True), (28, 'M', True, True), (2, '\n', True, True)]
    attempts += [(3, 'A', True, True)] * 32 + [(2, '\n', True, True)]
    engine.reader.events = [rtty.FrameEvent(i * 8000, *attempt)
                            for i, attempt in enumerate(attempts)]
    records = []
    monkeypatch.setattr(runner, '_emit', records.append)
    engine.consume()
    engine.release()
    assert [(r['text'], r['grade']) for r in records] == [
        ('M\n', 'tentative'), ('A' * 32 + '\n', 'confirmed')]
    assert len(engine.recent_framing) == 32


def test_line_latency_is_short_before_eof(monkeypatch):
    engine = runner._AcquisitionManager(None, 700)
    records = []
    monkeypatch.setattr(runner, "_emit", lambda r: records.append((engine.n / FS, r)))
    sent = audio("CQ\n")
    samples = np.concatenate((sent, np.zeros(2 * FS)))
    for start in range(0, len(samples), FS // 4):
        engine.push(samples[start:start + FS // 4])
    [(when, record)] = records
    assert record["text"] == "CQ\n"
    assert when <= len(sent) / FS + 1.25


def test_empty_input(monkeypatch):
    assert run(monkeypatch, np.zeros(0)) == []


def test_truncated_sitor_releases_suppression_after_silence(monkeypatch):
    short = sitor.modulate("TEST", phasing_pairs=4)[:-round(0.42 * FS)]
    samples = np.concatenate((short, np.zeros(2 * FS), audio("CQ\n", 1700)))
    records = run(monkeypatch, samples, 1700)
    assert [r["text"] for r in records if r["protocol"] == "RTTY"] == ["CQ\n"]


def test_corrupted_stop_bits_lower_the_emitted_grade(monkeypatch):
    units = [True] * 10
    for code in ita2.encode("BADBAD"):
        units += [False] + [bool(code >> k & 1) for k in range(5)] + [False, False]
    # Good framing after the damaged frames does not establish shift certainty.
    units += [True] * 10
    for code in ita2.encode("CQ\n")[1:]:
        units += [False] + [bool(code >> k & 1) for k in range(5)] + [True, True]
    units += [True] * 10
    damaged = fsk.modulate(units, baud=rtty.BAUD, f_hi=1585, f_lo=1415)
    records = run(monkeypatch, np.concatenate((audio("CQ"), damaged, audio("CQ\n"))))
    assert [(r["text"], r["grade"]) for r in records] == [
        ("CQ", "confirmed"), ("CQ\n", "tentative"), ("CQ\n", "confirmed")]


def test_adjacent_sitor_transmissions_each_own_their_phasing(monkeypatch):
    samples = np.concatenate((sitor.modulate("ONE", phasing_pairs=4),
                              sitor.modulate("TWO", phasing_pairs=4)))
    records = run(monkeypatch, samples, 1700)
    assert [(r["protocol"], r["text"]) for r in records] == [
        ("SITOR-B", "ONE"), ("SITOR-B", "TWO")]


@pytest.mark.slow
def test_three_minute_navtex_is_released_once_before_eof(monkeypatch):
    body = "LONG NAVTEX BULLETIN 0123456789\n" * 35
    message = navtex.Message("Q", "A", 17, body)
    sent = navtex.modulate(message, phasing_pairs=16)
    assert 170 < len(sent) / FS < 190
    engine = runner._AcquisitionManager(None, 700)
    records = []
    monkeypatch.setattr(runner, "_emit", records.append)
    # The bare runner expects the track at its nominal 1500 Hz.
    from hfmodem.monitor.channel import Channel
    mixer = Channel(1700, 1500, 350)
    sent = np.concatenate((mixer.push(sent), mixer.flush()))
    samples = np.concatenate((sent, np.zeros(25 * FS)))
    maximum = 0
    for start in range(0, len(samples), FS // 4):
        engine.push(samples[start:start + FS // 4])
        maximum = max(maximum, *(len(p.audio) for p in engine.decoder_lanes), 0)
    messages = [r for r in records if r.get("protocol") == "NAVTEX"]
    assert [(r["station"], r["detail"], r["text"]) for r in messages] == [
        ("Q", "A 17", body)]
    engine.push(np.empty(0), final=True)
    assert [r for r in records if r.get("protocol") == "NAVTEX"] == messages
    assert maximum <= (runner.EMISSION_S + runner.LOOKBACK_S + runner.BLOCK_S) * FS


@pytest.mark.parametrize("header", ["ZCZC QA20\n", "ZC_C QA20\n"])
def test_lock_loss_joins_rephased_text_or_releases_plain_text(monkeypatch, header):
    first = sitor.modulate(header + "BEFORE THE FADE\n\n\n", phasing_pairs=16)
    first = first[:-round(0.42 * FS)]  # Lose carrier before the idle-alpha endpoint.
    second = sitor.modulate("AFTER THE FADE\nNNNN\n", phasing_pairs=16)
    noise = np.random.default_rng(71).normal(0, 0.01, 3 * FS)
    samples = np.concatenate((first, noise, second))
    records = run(monkeypatch, samples, 1700)
    messages = [r for r in records if r["protocol"] == "NAVTEX"]
    if header.startswith("ZCZC"):
        assert len(messages) == 1
        assert messages[0]["detail"] == "A 20"
        assert "BEFORE THE FADE" in messages[0]["text"]
        assert "AFTER THE FADE" in messages[0]["text"]
    else:
        assert messages == []
        plain = "".join(r["text"] for r in records if r["protocol"] == "SITOR-B")
        assert plain.count("BEFORE THE FADE") == plain.count("AFTER THE FADE") == 1


@pytest.mark.parametrize("midpoint,start,level,text", [
    (1100, 2, 0.28, "CQ ONE 12\nTEST ONE\n"),
    (2100, 8, 0.18, "CQ TWO 34\nTEST TWO\n"),
])
def test_acquisition_lookback_does_not_prefix_rtty_with_noise_frame(
        monkeypatch, midpoint, start, level, text):
    samples = np.random.default_rng(1006).normal(0, 0.002, 16 * FS)
    sent = audio(text, midpoint)
    samples[start * FS:start * FS + len(sent)] += level * sent
    records = run(monkeypatch, samples, midpoint)
    lines = [r["text"] for r in records if r["protocol"] == "RTTY"]
    assert lines[:2] == text.splitlines(keepends=True)


def test_incomplete_navtex_wait_is_bounded(monkeypatch):
    monkeypatch.setattr(runner, "TEXT_WAIT_S", 5.0)
    sent = sitor.modulate("ZCZC QA05\nNO TERMINATOR", phasing_pairs=16)
    engine = runner._AcquisitionManager(None, 700)
    records = []
    monkeypatch.setattr(runner, "_emit", records.append)
    # The bare runner expects the track at its nominal 1500 Hz.
    from hfmodem.monitor.channel import Channel
    mixer = Channel(1700, 1500, 350)
    sent = np.concatenate((mixer.push(sent), mixer.flush()))
    samples = np.concatenate((sent, np.zeros(25 * FS)))
    for start in range(0, len(samples), FS // 4):
        engine.push(samples[start:start + FS // 4])
    plain = [r for r in records if r.get("protocol") == "SITOR-B"]
    assert len(plain) == 1 and "NO TERMINATOR" in plain[0]["text"]
    assert all(p.sitor.pending == "" for p in engine.decoder_lanes)


def test_unterminated_emission_cannot_pin_audio_forever(monkeypatch):
    monkeypatch.setattr(runner, "EMISSION_S", 5.0)
    monkeypatch.setattr(runner, "BLOCK_S", 2.0)
    monkeypatch.setattr(runner, "LOOKBACK_S", 1.0)
    sent = sitor.modulate("ZCZC QA17\n" + "NO END YET\n" * 20, phasing_pairs=4)
    from hfmodem.monitor.channel import Channel
    mixer = Channel(1700, 1500, 350)
    samples = np.concatenate((mixer.push(sent), mixer.flush()))
    engine = runner._AcquisitionManager(None, 700)
    monkeypatch.setattr(runner, "_emit", lambda record: None)
    for start in range(0, len(samples), FS // 4):
        engine.push(samples[start:start + FS // 4])
        assert all(len(p.audio) <= 8 * FS for p in engine.decoder_lanes)
    assert any(p.sitor.end > 0 for p in engine.decoder_lanes)


def test_complete_messages_are_released_during_a_continuing_emission(monkeypatch):
    message = navtex.Message("Q", "A", 20, "TEST")
    sent = sitor.modulate(navtex.frame(message) * 2 + "FILLER TEXT\n" * 25,
                          phasing_pairs=4)
    from hfmodem.monitor.channel import Channel
    mixer = Channel(1700, 1500, 350)
    samples = np.concatenate((mixer.push(sent), mixer.flush()))
    engine = runner._AcquisitionManager(None, 700)
    records = []
    monkeypatch.setattr(runner, "_emit", lambda r: records.append((engine.n / FS, r)))
    for start in range(0, len(samples), FS // 4):
        engine.push(samples[start:start + FS // 4])
    engine.push(np.empty(0), final=True)
    messages = [(when, r) for when, r in records if r.get("protocol") == "NAVTEX"]
    assert len(messages) == 2
    assert all(r["text"] == "TEST" and when < 21 for when, r in messages)


@pytest.mark.slow
def test_ten_minute_navtex_memory_and_decode_volume(monkeypatch):
    import tracemalloc

    body = "BULLETIN 0123456789\n" * 184
    text = navtex.frame(navtex.Message("Q", "A", 17, body))
    low = sitor.modulate(text, fs=FS // 8, phasing_pairs=16)
    assert 590 < len(low) / (FS // 8) < 610
    from scipy.signal import resample_poly
    from hfmodem.monitor.channel import Channel
    mixer = Channel(1700, 1500, 350)
    records = []
    monkeypatch.setattr(runner, "_emit", records.append)
    original = sitor.transmissions
    volume = 0

    def counted(audio, **kwargs):
        nonlocal volume
        volume += len(audio) / kwargs.get("fs", sitor.FS)
        return original(audio, **kwargs)

    monkeypatch.setattr(sitor, "transmissions", counted)
    # Warm imports and DSP caches before measuring runner allocations.
    engine = runner._AcquisitionManager(None, 700)
    engine.push(np.zeros(FS // 4))
    engine = runner._AcquisitionManager(None, 700)
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    try:
        for start in range(0, len(low), FS // 32):
            block = resample_poly(low[start:start + FS // 32], 8, 1)
            engine.push(mixer.push(block))
        engine.push(mixer.flush(), final=True)
        peak = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        tracemalloc.stop()
    assert peak < 40_000_000
    assert volume < 3 * len(low) / (FS // 8)
    assert [(r["detail"], r["text"]) for r in records
            if r.get("protocol") == "NAVTEX"] == [("A 17", body)]
    print(f"peak runner memory {peak / 1e6:.2f} MB; decoded {volume:.2f} s")


def test_streaming_sitor_audio_matches_batch():
    from hfmodem.monitor.channel import Channel
    from scipy.signal import resample_poly

    samples = np.random.default_rng(24).normal(0, 0.1, FS * 2 + 37)
    mixer = Channel(1500, 1700, 350)
    expected = resample_poly(np.concatenate((mixer.push(samples), mixer.flush())), 1, 8)
    stream = runner._SitorAudio(700)
    chunks = [stream.push(samples[i:i + 713]) for i in range(0, len(samples), 713)]
    chunks.append(stream.push(np.empty(0), final=True))
    np.testing.assert_allclose(np.concatenate(chunks), expected, atol=1e-13)


def test_refit_preserves_figures_and_framing(monkeypatch):
    text = "CQ\n" + "1234567890" * 24 + "\n"
    records = run(monkeypatch, audio(text), 1500)
    assert "".join(r["text"] for r in records) == text
    assert all(r["grade"] == "confirmed" for r in records)


def test_wrong_shift_is_not_confirmed(monkeypatch):
    samples = rtty.modulate("WRONG SHIFT\n", mark_hz=1712.5, space_hz=1287.5)
    assert run(monkeypatch, samples, 1500) == []


def test_adjacent_transmitters_do_not_make_a_phantom_pair(monkeypatch):
    one, two = audio("ONE 12\n" * 4, 1300), audio("TWO 34\n" * 4, 1640)
    samples = np.zeros(max(len(one), len(two)))
    samples[:len(one)] += 0.5 * one
    samples[:len(two)] += 0.5 * two
    records = run(monkeypatch, samples, 1300)
    assert records
    assert not any(abs(r["centre_hz"] - 1470) < 20 for r in records)
    assert "ONE 12\n" in "".join(r["text"] for r in records)


def test_continuous_rtty_releases_bounded_lines(monkeypatch):
    records = run(monkeypatch, audio("RY" * 150), 1500)
    assert "".join(r["text"] for r in records) == "RY" * 150
    assert max(len(r["text"]) for r in records) <= 80
    assert len(records) == 4


@pytest.mark.parametrize("damage", ["clock", "unit", "late-unit"])
def test_decoder_grid_ends_sitor_before_following_rtty(monkeypatch, damage):
    from scipy.signal import resample_poly

    rate = FS // 8
    body = "LONG NAVTEX BULLETIN 0123456789\n" * (28 if damage == "clock" else 4)
    sent = navtex.modulate(navtex.Message("Q", "A", 17, body),
                          fs=rate * (1 + (40e-6 if damage == "clock" else 0)))
    if damage in ("unit", "late-unit"):
        cut = round((23.12 if damage == "late-unit" else 7) * rate)
        sent = np.concatenate((sent[:cut], sent[cut + rate // 100:]))
    samples = resample_poly(np.concatenate((sent, np.zeros(rate // 2),
        rtty.modulate("AFTER\n", fs=rate, mark_hz=1785, space_hz=1615))), 8, 1)
    records = run(monkeypatch, samples, 1700)
    assert [r["text"] for r in records if r["protocol"] == "RTTY"] == ["AFTER\n"]
    assert len([r for r in records if r["protocol"] == "NAVTEX"]) == 1


def test_noisy_navtex_headers_each_emit_once_with_sender_role(monkeypatch):
    text = "ZCZC QA17XYZ\nONE\nNNNN\nZCZC QA20XYZ\nTWO\nNNNN\n"
    records = run(monkeypatch, sitor.modulate(text, phasing_pairs=4), 1700)
    assert [(r["detail"], r["text"], r["role"]) for r in records] == [
        ("A 17", "ONE", "sender"), ("A 20", "TWO", "sender")]


def test_header_split_across_emissions_is_joined(monkeypatch):
    first = sitor.modulate("ZCZC Q", phasing_pairs=16)
    second = sitor.modulate("A17\nBODY\nNNNN", phasing_pairs=16)
    records = run(monkeypatch, np.concatenate((first, np.zeros(FS), second)), 1700)
    assert [(r["protocol"], r["text"]) for r in records] == [("NAVTEX", "BODY")]


def test_acquisition_replay_after_a_long_pause_is_bounded(monkeypatch):
    maximum = 0
    original = runner._DecoderLane

    class Measured(original):
        def push(self, block, final=False):
            nonlocal maximum
            super().push(block, final=final)
            maximum = max(maximum, len(self.audio))

    monkeypatch.setattr(runner, "_DecoderLane", Measured)
    samples = np.concatenate((audio("FIRST\n"), np.zeros(45 * FS), audio("SECOND\n", 1580)))
    records = run(monkeypatch, samples, 1500)
    assert "".join(r["text"] for r in records) == "FIRST\nSECOND\n"
    assert maximum <= (runner.BLOCK_S + runner.LOOKBACK_S) * FS



@pytest.mark.parametrize("body", ["A" * 390, "TEST 12\n" * 37])
def test_two_navtex_headers_in_one_emission_have_their_own_air_times(monkeypatch, body):
    from scipy.signal import resample_poly

    text = navtex.frame(navtex.Message('Q', 'A', 17, body))
    text += navtex.frame(navtex.Message('Q', 'A', 20, 'SECOND'))
    decoder = ita2.Decoder('us', unshift_on_space=False)
    decoded = ''
    times = []
    for i, code in enumerate(ita2.encode(text)):
        char = decoder.feed(code)
        if char == 'Z' and text.startswith('ZCZC ', len(decoded)):
            times.append((16 + i) * 0.14)
        decoded += char
    samples = resample_poly(sitor.modulate(text, fs=FS // 8), 8, 1)
    records = [r for r in run(monkeypatch, samples, 1700) if r['protocol'] == 'NAVTEX']
    assert [r['detail'] for r in records] == ['A 17', 'A 20']
    assert [r['t'] for r in records] == pytest.approx(times, abs=2)
    assert records[1]['t'] - records[0]['t'] == pytest.approx(60, abs=3)


def test_plain_sitor_record_is_dated_by_its_first_character(monkeypatch):
    sent = sitor.modulate('FIRST CHARACTER', phasing_pairs=16)
    records = run(monkeypatch, sent, 1700)
    assert len(records) == 1 and records[0]['protocol'] == 'SITOR-B'
    assert records[0]['t'] == pytest.approx(17 * 0.14, abs=0.1)



def test_sitor_time_marks_follow_trimmed_partial_headers_and_plain_text(monkeypatch):
    records = []
    monkeypatch.setattr(runner, '_emit', records.append)
    engine = runner._DecoderLane(None, 700)
    engine.sitor.origin = (90, 1500)
    engine.sitor_text('NOISEZCZC Q', times=100 + np.arange(11) * .14,
                      certainty=[True] * 11, complete=True)
    assert engine.sitor.pending == 'ZCZC Q'
    engine.sitor_text('A17\nONE\nNNNN\n', times=160 + np.arange(13) * .14,
                      certainty=[True] * 13, complete=True)
    assert records[0]['t'] == pytest.approx(100 + 5 * 0.14, abs=0.001)
    assert engine.sitor.marks == []
    engine.sitor.origin = (170, 1500)
    engine.sitor_text('NEXT', times=180 + np.arange(4) * .14,
                      certainty=[True] * 4, complete=True)
    assert records[1]['protocol'] == 'SITOR-B'
    assert records[1]['t'] == 180


def test_reacquisition_refreshes_a_weaker_transmitters_level(monkeypatch):
    samples = np.concatenate((audio('FIRST\n'), np.zeros(2 * FS),
                              0.05 * audio('SECOND\n', 1580)))
    records = run(monkeypatch, samples, 1500)
    assert ''.join(r['text'] for r in records) == 'FIRST\nSECOND\n'


@pytest.mark.parametrize('gap,noise', [(.075, 0), (.1, 0), (.125, 0),
                                      (.1, .001), (.1, .002),
                                      (.125, .001), (.125, .002)])
def test_short_handoff_refreshes_a_weaker_transmitters_level(monkeypatch, gap, noise):
    samples = np.r_[audio('FIRST STATION 12345\n'), np.zeros(round(gap * FS)),
                    .05 * audio('SECOND STATION 67890\n'), np.zeros(3 * FS)]
    samples += np.random.default_rng(17).normal(0, noise, len(samples))
    records = run(monkeypatch, samples, 1500)
    assert [r['text'] for r in records] == ['FIRST STATION 12345\n',
                                           'SECOND STATION 67890\n']
    assert all(r['grade'] == 'confirmed' for r in records)


@pytest.mark.parametrize('gap', [1, 2])
@pytest.mark.parametrize('midpoint', [1500, 1530, 1545, 1550, 1580])
def test_weak_successor_at_eof_has_its_own_power_and_clock(monkeypatch, gap, midpoint):
    first = audio('FIRST\n')
    second = .05 * audio('SECOND\n', midpoint)
    records = run(monkeypatch, np.r_[first, np.zeros(gap * FS), second], 1500)
    assert [r['text'] for r in records] == ['FIRST\n', 'SECOND\n']
    assert all(r['grade'] == 'confirmed' for r in records)
    assert records[1]['centre_hz'] == pytest.approx(midpoint, abs=10)
    alone = run(monkeypatch, second, 1500)
    assert records[1]['t'] == pytest.approx(
        len(first) / FS + gap + alone[0]['t'], abs=.03)


@pytest.mark.parametrize('gain', [.2, .3])
def test_two_keyed_pairs_in_one_cut_have_independent_framing_and_shift(monkeypatch, gain):
    one, two = audio('ONE 12\n', 1300), audio('TWO 34\n', 1560)
    samples = np.zeros(max(len(one), len(two)))
    samples[:len(one)] += gain * one
    samples[:len(two)] += 0.8 * two
    records = run(monkeypatch, samples, 1500)
    assert sorted(r['text'] for r in records) == ['ONE 12\n', 'TWO 34\n']
    for record in records:
        centre = 1300 if record['text'].startswith('ONE') else 1560
        assert record['centre_hz'] == pytest.approx(centre, abs=10)
        assert record['offset_hz'] == pytest.approx(centre - 1500, abs=10)


def test_two_navtex_pairs_keep_independent_emissions_and_parsers(monkeypatch):
    from scipy.signal import hilbert, resample_poly

    signals = []
    for centre, serial, body, gain in [(1300, 17, 'ONE', .6), (1560, 20, 'TWO', .6)]:
        low = navtex.modulate(navtex.Message('Q', 'A', serial, body), fs=6000)
        shifted = np.real(hilbert(low) * np.exp(
            2j * np.pi * (centre - 1700) * np.arange(len(low)) / 6000))
        signals.append(gain * resample_poly(shifted, 8, 1))
    samples = np.zeros(max(map(len, signals)))
    for signal in signals:
        samples[:len(signal)] += signal
    records = run(monkeypatch, samples, 1500)
    assert sorted((r['protocol'], r['detail'], r['text']) for r in records) == [
        ('NAVTEX', 'A 17', 'ONE'), ('NAVTEX', 'A 20', 'TWO')]
    for record in records:
        assert record['centre_hz'] == pytest.approx(1300 if record['text'] == 'ONE' else 1560, abs=10)


def test_new_reader_uses_the_successors_level(monkeypatch):
    from hfmodem.monitor.channel import tuned

    records = []
    monkeypatch.setattr(runner, '_emit', records.append)
    chan = tuned(1500, 700)
    engine = runner._AcquisitionManager(chan, 700)
    samples = np.concatenate((audio('FIRST\n'), np.zeros(2 * FS),
                              .05 * audio('SECOND\n', 1530)))
    for start in range(0, len(samples), FS // 4):
        engine.push(chan.push(samples[start:start + FS // 4]))
    engine.push(chan.flush(), final=True)
    assert ''.join(r['text'] for r in records if 'text' in r) == 'FIRST\nSECOND\n'
    assert len(engine.decoder_lanes) == 1


@pytest.mark.parametrize("midpoint", [1530, 1545, 1550, 1580])
def test_reacquisition_finds_the_successors_frequency(monkeypatch, midpoint):
    first = audio('FIRST\n')
    second = audio('SECOND\n', midpoint)
    samples = np.concatenate((first, np.zeros(2 * FS), second))
    records = run(monkeypatch, samples, 1500)
    assert [r['text'] for r in records] == ['FIRST\n', 'SECOND\n']
    assert records[1]['centre_hz'] == pytest.approx(midpoint, abs=10)
    # The record dates its first decoded character, after key-on and LTRS.
    alone = run(monkeypatch, second, 1500)
    assert records[1]['t'] == pytest.approx(
        (len(first) + 2 * FS) / FS + alone[0]['t'], abs=.03)


def test_weak_new_pair_after_one_second_gap_at_eof(monkeypatch):
    samples = np.concatenate((audio('FIRST\n'), np.zeros(FS),
                              .05 * audio('SECOND\n', 1580)))
    records = run(monkeypatch, samples, 1500)
    assert [r['text'] for r in records] == ['FIRST\n', 'SECOND\n']


def test_new_pair_replay_excludes_a_previous_transmitter(monkeypatch):
    # More than the association tolerance: force a new lane and replay.
    first = audio('FIRST\n')
    second = audio('SECOND\n', 1580)
    records = run(monkeypatch, np.concatenate((first, np.zeros(FS), second)), 1500)
    assert [r['text'] for r in records] == ['FIRST\n', 'SECOND\n']
    alone = run(monkeypatch, second, 1500)
    assert records[1]['t'] == pytest.approx(len(first) / FS + 1 + alone[0]['t'], abs=.03)


def test_rtty_status_holds_open_for_keyed_activity_and_closes_after_a_pause(monkeypatch):
    # Spaces key both tones while emitting no printable record.
    sent = audio(' ' * 30)
    records = run(monkeypatch, np.concatenate((sent, np.zeros(4 * FS))), 1500, status=True)
    states = [r for r in records if r.get('kind') == 'emission-status']
    assert all(r['detail'] == 'open' for r in states if 2 < r['t'] < len(sent) / FS)
    assert any(r['detail'] == 'open' for r in states)
    assert states[-1]['detail'] == 'closed'
    assert not any('text' in r for r in records)


def test_stopped_carrier_is_not_replayed_or_reactivated(monkeypatch):
    sent = audio('GONE\n')
    records = run(monkeypatch, np.concatenate((sent, np.zeros(38 * FS))),
                  1500, status=True)
    assert [r['text'] for r in records if 'text' in r] == ['GONE\n']
    assert all(r['detail'] == 'closed' for r in records
               if r.get('kind') == 'emission-status' and r['t'] > len(sent) / FS + 2)


def test_sequential_carriers_release_acquisition_slots(monkeypatch):
    samples = []
    for i, centre in enumerate((1260, 1380, 1500, 1620, 1740)):
        samples.extend((audio(f'TEST {i}\n', centre), np.zeros(FS)))
    records = run(monkeypatch, np.concatenate(samples), 1500)
    assert [r['text'] for r in records] == [f'TEST {i}\n' for i in range(5)]


def test_sitor_power_does_not_gate_a_weaker_rtty_successor(monkeypatch):
    sent = navtex.modulate(navtex.Message('Q', 'A', 17, 'BULLETIN'), phasing_pairs=4)
    records = run(monkeypatch, np.concatenate((sent, np.zeros(FS // 2),
                  .05 * audio('AFTER\n', 1700))), 1700)
    assert [(r['protocol'], r['text']) for r in records] == [
        ('NAVTEX', 'BULLETIN'), ('RTTY', 'AFTER\n')]


def test_rejected_frame_does_not_disable_pause_release(monkeypatch):
    units = [True] * 10
    for code in ita2.encode('BADBADBADBAD'):
        units += [False] + [bool(code >> k & 1) for k in range(5)] + [False, False]
    damaged = fsk.modulate(units, baud=rtty.BAUD, f_hi=1585, f_lo=1415)
    records = run(monkeypatch, np.concatenate((audio('FIRST'), damaged,
                  np.zeros(2 * FS), audio('SECOND\n'))), 1500)
    assert len(records) == 2
    assert records[0]['text'].startswith('FIRST')
    reader = rtty.Stream()
    reader.feed(damaged)
    reader.flush()
    assert reader.rejected > 0
    assert records[1]['text'] == 'SECOND\n'


def test_zero_db_snr_rtty_through_runner(monkeypatch):
    text = 'RYRYRY THE QUICK BROWN FOX 0123456789\n'
    sent = audio(text)
    noise = np.random.default_rng(240).normal(size=len(sent))
    spectrum = np.fft.rfft(noise)
    spectrum[np.fft.rfftfreq(len(sent), 1 / FS) > 3000] = 0
    noise = np.fft.irfft(spectrum, len(sent))
    noise *= np.sqrt(np.mean(sent ** 2) / np.mean(noise ** 2))
    records = run(monkeypatch, .4 * (sent + noise), 1500)
    assert ''.join(r['text'] for r in records) == text


@pytest.mark.parametrize('tones', [(1500,), (1415, 1585)])
def test_unkeyed_carriers_emit_nothing(monkeypatch, tones):
    t = np.arange(4 * FS) / FS
    samples = sum(.2 * np.sin(2 * np.pi * tone * t) for tone in tones)
    assert run(monkeypatch, samples, 1500) == []


def test_navtex_tuning_trials_share_header_identity_even_when_it_is_garbled(monkeypatch):
    records = []
    monkeypatch.setattr(runner, '_emit', records.append)
    owner = runner._AcquisitionManager(None, 700)
    first = runner.normalize(100, 'NAVTEX', 'BULLETIN', 0,
                             station='Q', detail='A 15')
    alternate = runner.normalize(100.03, 'NAVTEX', 'BUL_ETIN', 10,
                                 station='Q', detail='D 15')
    repeated = dict(first, t=120)
    owner.publish(first)
    owner.publish(alternate)
    owner.publish(repeated)
    assert records == [first, repeated]


@pytest.mark.parametrize('duration', [.25, .5, .75, 1, 2, 2.5])
@pytest.mark.parametrize('keyed', ['hard', 'shaped'], indirect=True)
def test_rtty_fade_preserves_shift_until_continuity_expires(monkeypatch, keyed, duration):
    sent = keyed('1234567890' * 5 + '\n', mark_hz=1585, space_hz=1415)
    sent[3 * FS:round((3 + duration) * FS)] = 0
    records = run(monkeypatch, np.r_[sent, np.zeros(3 * FS)], 1500, status=True)
    text = [r for r in records if r.get('protocol') == 'RTTY']
    after = [r for r in text if r['t'] >= 3 + duration]
    assert ''.join(r['text'] for r in text).startswith('12345678901234')
    if duration <= 2:
        assert ''.join(r['text'] for r in text).endswith('67890\n')
        assert set(''.join(r['text'] for r in text)) <= set('0123456789\n')
    else:
        assert after
        assert all(r['grade'] == 'tentative' for r in after)
    states = [r for r in records if r.get('kind') == 'emission-status']
    if duration <= 2:
        assert all(r['detail'] == 'open' for r in states if 3 <= r['t'] <= 3 + duration)
    else:
        assert any(r['detail'] == 'closed' for r in states if 5 < r['t'] <= 5.5)
    assert states[-1]['detail'] == 'closed'


@pytest.mark.parametrize('duration', [0, .125, .25, .375, .5, 1, 2, 2.5])
def test_navtex_fade_keeps_the_synchronous_cursor(monkeypatch, duration):
    from scipy.signal import hilbert, resample_poly

    ended = []
    original = runner._AcquisitionManager.end_carrier

    def end_carrier(owner, lane):
        original(owner, lane)
        ended.append((owner.n / FS, lane))

    monkeypatch.setattr(runner._AcquisitionManager, 'end_carrier', end_carrier)
    rate = FS // 8
    body = 'TEST BULLETIN 0123456789\n' * 6
    low = sitor.modulate(navtex.frame(navtex.Message('Q', 'A', 17, body)), fs=rate)
    low = np.real(hilbert(low) * np.exp(-2j * np.pi * 200 * np.arange(len(low)) / rate))
    low[10 * rate:round((10 + duration) * rate)] = 0
    batch = sitor.transmissions(low, fs=rate, offset_hz=-200)
    samples = resample_poly(np.r_[low, np.zeros(3 * rate)], 8, 1)
    records = run(monkeypatch, samples, 1500)
    messages = [r for r in records if r['protocol'] == 'NAVTEX']
    if duration <= 2:
        assert len(ended) == 1 and ended[0][0] > len(low) / rate
        assert len(messages) == 1
        assert messages[0]['detail'] == 'A 17'
        assert messages[0]['text'].endswith('TEST BULLETIN 0123456789\n' * 3)
        assert not any(r['protocol'] == 'RTTY' for r in records)
        if duration == .25:
            assert any(body in text for _, text in batch)
            assert messages[0]['text'] == body
    else:
        assert ended[0][0] == pytest.approx(12.25, abs=.25)
        assert all(tr.decoded and tr.end / FS < 10.25
                   for tr in ended[0][1].sitor.transmissions)
        assert messages == []


@pytest.mark.parametrize('text', ['R', 'E', 'OK', '73', 'OK\n'])
@pytest.mark.parametrize('neighbour', [False, True])
def test_completed_burst_between_searches_is_decoded_once(monkeypatch, text, neighbour):
    first = .4 * audio('CONTINUING RTTY LINE\n' * 3, 1300)
    second = .4 * audio(text, 1560)
    samples = first if neighbour else np.zeros(len(first))
    start = round(1.8 * FS)
    samples[start:start + len(second)] += second
    records = run(monkeypatch, samples, 1500)
    assert [r['text'] for r in records if r['centre_hz'] > 1500] == [text]


def test_midstream_rtty_is_tentative_until_an_explicit_shift(monkeypatch):
    # The first two data frames have good stops but no preceding shift code.
    codes = ita2.encode('12')[2:] + ita2.encode('34AB\n')
    units = [True] * 10
    for code in codes:
        units += [False] + [bool(code >> k & 1) for k in range(5)] + [True, True]
    units += [True] * 10
    sent = fsk.modulate(units, baud=rtty.BAUD, f_hi=1585, f_lo=1415)
    records = run(monkeypatch, sent, 1500)
    assert [(r['text'], r['grade']) for r in records] == [
        ('QW', 'tentative'), ('34AB\n', 'confirmed')]


@pytest.mark.parametrize('prefix, suffix, restore, shift', [
    ('1234567890' * 3, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '123 RESTORED\n', 31),
    ('ABCDEF' * 3, '1234567890' * 4, 'RESTORED ABC\n', 27),
])
def test_faded_shift_splits_confirmed_tentative_and_restored_text(
        monkeypatch, prefix, suffix, restore, shift):
    sent = audio(prefix + suffix + '\n' + restore)
    index = next(i for i, code in enumerate(ita2.encode(prefix + suffix))
                 if i > 2 and code == shift)
    lo = round((10 + 7.5 * index) / rtty.BAUD * FS) - round(.04 * FS)
    hi = lo + round(.25 * FS)
    sent[lo:hi] = 0
    records = run(monkeypatch, np.r_[sent, np.zeros(3 * FS)], 1500)
    assert {r['protocol'] for r in records} == {'RTTY'}
    assert [r['grade'] for r in records] == ['confirmed', 'tentative', 'confirmed']
    assert prefix.startswith(records[0]['text'])
    assert records[1]['t'] > hi / FS
    assert records[2]['text'] == restore


@pytest.mark.parametrize('midpoint', [1260, 1740])
def test_navtex_fade_is_not_masked_by_a_continuous_rtty_neighbour(monkeypatch, midpoint):
    from scipy.signal import hilbert, resample_poly

    rate = FS // 8
    body = 'TEST BULLETIN 0123456789\n' * 6
    low = navtex.modulate(navtex.Message('Q', 'A', 17, body), fs=rate)
    low = np.real(hilbert(low) * np.exp(-2j * np.pi * 200 * np.arange(len(low)) / rate))
    low[10 * rate:12 * rate] = 0
    own = resample_poly(np.r_[low, np.zeros(3 * rate)], 8, 1)
    neighbour = .5 * audio('NEIGHBOUR RTTY LINE\n' * 15, midpoint)
    own += neighbour[:len(own)]
    records = run(monkeypatch, own, 1500)
    messages = [r for r in records if r['protocol'] == 'NAVTEX']
    assert len(messages) == 1
    assert messages[0]['station'] == 'Q' and messages[0]['detail'] == 'A 17'
    assert messages[0]['text'].endswith('TEST BULLETIN 0123456789\n' * 3)
    nearby = [r for r in records if r['protocol'] == 'RTTY']
    assert nearby and all(abs(r['centre_hz'] - midpoint) < 20 for r in nearby)
    assert ''.join(r['text'] for r in nearby).count('NEIGHBOUR RTTY LINE\n') >= 5


def test_dormant_navtex_parser_releases_audio_and_waits_without_dsp(monkeypatch):
    from hfmodem.monitor.channel import tuned

    records = []
    monkeypatch.setattr(runner, '_emit', records.append)
    chan = tuned(1700, 700)
    engine = runner._AcquisitionManager(chan, 700)
    first = sitor.modulate('ZCZC QA17\nUNFINISHED', phasing_pairs=16)
    later = sitor.modulate(' CONTINUED\nNNNN', phasing_pairs=16)
    samples = np.r_[first, np.zeros(25 * FS), later, np.zeros(3 * FS)]
    parked = []
    original = runner._DecoderLane.push

    def push(lane, block, final=False):
        assert lane in engine.active or any(not tr.decoded for tr in lane.sitor.transmissions)
        original(lane, block, final)

    monkeypatch.setattr(runner._DecoderLane, 'push', push)
    for start in range(0, len(samples), FS // 4):
        engine.push(chan.push(samples[start:start + FS // 4]))
        for lane in engine.decoder_lanes:
            if lane not in engine.active and lane.sitor.pending:
                parked.append(lane)
                assert lane.history.storage.nbytes == lane.low_history.storage.nbytes == 0
                assert lane.sitor_stream is None and lane.reader is None
    engine.push(chan.flush(), final=True)
    assert parked
    assert [(r['protocol'], r['text']) for r in records if 'text' in r] == [
        ('NAVTEX', 'UNFINISHED CONTINUED')]


@pytest.mark.parametrize('delta', [18, 35, 40, 45])
def test_material_move_replays_onset_before_the_periodic_refit(monkeypatch, delta):
    first = 'FIRST STATION 12345\n' * 2
    second = 'SECOND STATION 67890\n' * 3
    samples = np.r_[audio(first), audio(second, 1500 + delta), np.zeros(3 * FS)]
    records = run(monkeypatch, samples, 1500)
    assert [r['text'] for r in records] == first.splitlines(True) + second.splitlines(True)
    assert all(r['grade'] == 'confirmed' for r in records)
    assert all(abs(r['centre_hz'] - 1500 - delta) < 10 for r in records[2:])


def test_navtex_erased_figures_shift_is_tentative(monkeypatch):
    from scipy.signal import resample_poly

    rate = FS // 8
    body = 'POSITION ' + 'ABCDEF' * 8 + '1234567890' * 4 + ' RESTORED TEXT'
    message = navtex.Message('Q', 'A', 17, body)
    index = [i for i, code in enumerate(ita2.encode(navtex.frame(message))) if code == 27][-1]
    at = (16 + index) * .14 - .04
    low = navtex.modulate(message, fs=rate)
    low[round(at * rate):round((at + .43) * rate)] = 0
    records = run(monkeypatch, resample_poly(np.r_[low, np.zeros(3 * rate)], 8, 1), 1700)
    assert len(records) == 1
    record = records[0]
    assert (record['protocol'], record['station'], record['detail'], record['grade']) == (
        'NAVTEX', 'Q', 'A 17', 'tentative')
    assert 'QWERTYUIOP' * 4 in record['text']
    assert record['text'].endswith('RESTORED TEXT')


def _neighbouring_rtty(*, short=False):
    samples = np.zeros(22 * FS)
    stations = [(1380, 'SECOND 9876543210\n' * 5, 0),
                (1500, 'THIRD 0123456789\n' * 5, 0),
                (1740, 'FOURTH 1357924680\n' * 5, 0),
                (1620, 'REPLACEMENT 2468\n' * 3, 6 * FS)]
    if short:
        stations.append((1260, 'SHORT\n', 0))
    for midpoint, text, start in stations:
        sent = .25 * audio(text, midpoint)
        samples[start:start + len(sent)] += sent
    return samples + np.random.default_rng(19).normal(0, .001, len(samples))


def test_framing_rescues_a_readable_fourth_pair_among_neighbours(monkeypatch):
    records = run(monkeypatch, _neighbouring_rtty(), 1500)
    replacement = [r['text'] for r in records if abs(r['centre_hz'] - 1620) < 15]
    assert ''.join(replacement).count('REPLACEMENT 2468\n') >= 2
    assert all(r['protocol'] == 'RTTY' for r in records)


def test_neighbour_leakage_does_not_keep_a_stopped_lane(monkeypatch):
    ended = []
    original = runner._AcquisitionManager.end_carrier

    def end_carrier(manager, lane):
        ended.append((manager.n / FS, lane.midpoint))
        original(manager, lane)

    monkeypatch.setattr(runner._AcquisitionManager, 'end_carrier', end_carrier)
    records = run(monkeypatch, _neighbouring_rtty(short=True), 1500)
    short_ends = [t for t, midpoint in ended if abs(midpoint - 1260) < 15]
    assert len(short_ends) == 1 and short_ends[0] <= 5
    assert 'SHORT\n' in [r['text'] for r in records]
    assert any(abs(r['centre_hz'] - 1620) < 15 and 'REPLACEMENT' in r['text'] for r in records)


@pytest.mark.parametrize('duration', [1, 3, 5, 10])
def test_rtty_mark_idle_preserves_figures_and_carrier_lifetime(monkeypatch, duration):
    sent = '12345678901234567890\n'
    # Insert mark after six digits, with no new shift word after the idle.
    schedule = [(10, True)]
    for i, code in enumerate(ita2.encode(sent)):
        schedule.extend([(1, False)] + [(1, bool(code >> k & 1)) for k in range(5)]
                        + [(1.5, True)])
        if i == 7:
            schedule.append((duration * rtty.BAUD, True))
    schedule.append((10, True))
    bounds = np.rint(np.r_[0, np.cumsum([n for n, _ in schedule])] * FS / rtty.BAUD).astype(int)
    frequencies = np.repeat([1585 if mark else 1415 for _, mark in schedule], np.diff(bounds))
    samples = .5 * np.sin(2 * np.pi * np.cumsum(frequencies) / FS)
    records = run(monkeypatch, np.r_[samples, np.zeros(3 * FS)], 1500, status=True)
    text = [r for r in records if r['protocol'] == 'RTTY']
    assert ''.join(r['text'] for r in text) == sent
    assert all(r['grade'] == 'confirmed' for r in text)
    states = [r for r in records if r.get('kind') == 'emission-status']
    assert all(r['detail'] == 'open' for r in states if 2 < r['t'] < len(samples) / FS)
    assert states[-1]['detail'] == 'closed'


def test_live_rtty_shares_the_managers_block_power(monkeypatch):
    calls = []
    original = runner._AcquisitionManager.tone_energy

    def energy(samples, midpoint):
        if len(samples) == FS // 4:
            calls.append((midpoint, samples.tobytes()))
        return original(samples, midpoint)

    monkeypatch.setattr(runner._AcquisitionManager, 'tone_energy', staticmethod(energy))
    records = run(monkeypatch, np.r_[audio('TEST LINE 1234567890\n' * 3), np.zeros(3 * FS)], 1500)
    assert ''.join(r['text'] for r in records) == 'TEST LINE 1234567890\n' * 3
    # Identical live blocks share power; replay and changed tuning calculate it afresh.
    nonzero = [key for key in calls if any(key[1])]
    assert len(nonzero) == len(set(nonzero))


@pytest.mark.parametrize('gap', [.5, 1, 2, 3])
def test_keyed_successor_can_share_a_stopped_stations_mark(monkeypatch, gap):
    first, second = 'FIRST 12345\n', 'SECOND STATION 67890\n' * 3
    samples = np.r_[audio(first), np.zeros(round(gap * FS)),
                    .7 * audio(second, 1670), np.zeros(3 * FS)]
    records = run(monkeypatch, samples, 1500)
    assert [r['text'] for r in records] == first.splitlines(True) + second.splitlines(True)
    assert all(r['grade'] == runner.CONFIRMED for r in records)
    assert all(abs(r['centre_hz'] - 1670) < 10 for r in records[1:])


@pytest.mark.parametrize('at', [2.24, 5, 10])
def test_starting_mid_sitor_never_confirms_async_garbage(monkeypatch, at):
    sent = sitor.modulate('LONG BULLETIN 0123456789\n' * 12)
    records = run(monkeypatch, np.r_[sent[round(at * FS):], np.zeros(3 * FS)], 1700)
    assert all(r['protocol'] != 'RTTY' or r['grade'] == runner.TENTATIVE for r in records)


@pytest.mark.parametrize('gap', [0, 3])
def test_mid_sitor_grading_allows_next_phasing_and_later_rtty(monkeypatch, gap):
    unphased = sitor.modulate('LONG BULLETIN 0123456789\n' * 3)[5 * FS:]
    samples = np.r_[unphased, np.zeros(gap * FS), sitor.modulate('NEXT PHASING'),
                    np.zeros(3 * FS), audio('REAL RTTY 73\n', 1700)]
    records = run(monkeypatch, samples, 1700)
    confirmed = [r for r in records if r['grade'] == runner.CONFIRMED]
    assert [(r['protocol'], r['text']) for r in confirmed] == [
        ('SITOR-B', 'NEXT PHASING'), ('RTTY', 'REAL RTTY 73\n')]


@pytest.mark.parametrize('args,expected', [
    ((), '123456 UIOPQW ERTYUI\n'),
    (('--rtty-usos',), '123456 UIOPQW ERTYUI\n'),
    (('--no-rtty-usos',), '123456 789012 345678\n'),
])
def test_runner_receiving_space_policy(monkeypatch, args, expected):
    codes = ita2.encode('123456 789012 345678\n')
    first_figs = codes.index(27)
    codes = codes[:first_figs + 1] + [code for code in codes[first_figs + 1:] if code != 27]
    schedule = [(10, True)]
    for code in codes:
        schedule.extend([(1, False)] + [(1, bool(code >> k & 1)) for k in range(5)]
                        + [(1.5, True)])
    schedule.append((10, True))
    bounds = np.rint(np.r_[0, np.cumsum([n for n, _ in schedule])] * FS / rtty.BAUD).astype(int)
    tones = np.repeat([1585 if mark else 1415 for _, mark in schedule], np.diff(bounds))
    samples = .5 * np.sin(2 * np.pi * np.cumsum(tones) / FS)
    records = run(monkeypatch, samples, 1500, args=args)
    assert ''.join(r['text'] for r in records) == expected
    assert all(r['grade'] == runner.CONFIRMED for r in records)


def test_navtex_history_retains_only_bounded_emission_identity(monkeypatch):
    said = []
    monkeypatch.setattr(runner, '_emit', said.append)
    manager = runner._AcquisitionManager(None, 700)
    for i in range(1100):
        for dt, hz, protocol in ((0, 0, 'NAVTEX'), (.799, 59.999, 'NAVTEX'),
                                  (.8, 60, 'NAVTEX'), (.1, 0, 'RTTY'), (.01, 1, 'NAVTEX')):
            manager.publish(runner.normalize(10 * i + dt, protocol, 'X' * 4000, hz,
                                             station='Q', detail='A 17' if dt == 0 else 'D 15'))
    assert len(said) == 3300
    assert len(manager.said) == 1000
    assert all(set(old) == {'t', 'centre_hz'} for old in manager.said)
    assert manager.said[0]['t'] == 6000
    # The first emission was evicted; an identical body/header can recur.
    manager.publish(runner.normalize(0, 'NAVTEX', 'X' * 4000, 0, station='Q', detail='A 17'))
    assert len(said) == 3301
