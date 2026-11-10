# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Emission registration and the early-word / later-extent integration boundary."""

from dataclasses import replace
import numpy as np
import pytest
from hfmodem.shrike import p4breakintx, p4robusttx
from hfmodem.shrike.p4receiver import (
    P4Receiver,
    Emission,
    Role,
    Kind,
    Observation,
    Registered,
    Expect,
)

FS = 48000


@pytest.mark.parametrize("body", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_a_word_read_early_is_extended_once_by_its_whole_window(body, partial):
    rx = P4Receiver(FS, admitted=True)
    y = round(0.923 * FS)
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, 0, 0.758))
    signal = (
        p4breakintx.changeover_packet(b"hello", 0)
        if body
        else p4robusttx.robust_word("Y27")
    )
    audio = np.r_[np.zeros(y), signal, np.zeros(FS)]
    if partial:
        records = rx.read_partial(
            audio[: y + round(0.18 * FS)], 0, deadline=y + round(0.846 * FS)
        )
        assert len(records) == 1 and records[0].observation.provisional
    records = rx.read_window(audio[: y + round(0.300 * FS)], 0, whole=False)
    assert [r.outcome for r in records] == [
        Registered.CONFIRMED if partial else Registered.NEW
    ]
    assert records[0].observation.extent_s is None
    records = rx.read_window(audio[: y + round(0.845 * FS)], 0, whole=True)
    assert [r.outcome for r in records] == (
        [Registered.EXTENDED, Registered.ENRICHED] if body else [Registered.EXTENDED]
    )
    assert records[0].observation.extent_s == pytest.approx(
        0.80 if body else 0.245, abs=0.015
    )
    if body:
        assert records[1].observation.frame.data.startswith(b"hello")
    assert rx.read_window(audio[: y + round(0.845 * FS)], 0, whole=True) == []
    assert len(rx.observations) == (2 if body else 1)


def test_provisional_resolves_by_kind_and_remembers_key():
    rx = P4Receiver(admitted=True)
    old = Observation(Kind.LONG, FS, 0, 100, 100, 0.85, 0, 0.7, provisional=True)
    assert rx.register(old).outcome == Registered.NEW
    rx.note_emission(Emission(Kind.WORD, Role.IRS, 100000, 0.244))
    full = replace(old, kind=Kind.SHORT, provisional=False, header="A", value=13)
    reg = rx.register(full)
    assert reg.outcome == Registered.CONTRADICTED and reg.replaced == old
    assert reg.answered == 100000
    assert rx.register(full).outcome == Registered.DUPLICATE


def test_role_clock_and_refused_raster():
    rx = P4Receiver()
    rx.note_emission(Emission(Kind.WORD, Role.IRS, 1000, 0.244))
    assert rx.clock.expects == Expect.PACKET
    assert rx.clock.next_due == 1000 + round(0.33 * FS)
    for i in range(3):
        due = rx.clock.next_due
        rx.note_refusal(Kind.BODY, due)
        assert rx.clock.next_due == due
        rx.new_cycle()
        assert rx.clock.next_due == 1000 + round((0.33 + (i + 1) * 1.25) * FS)


def test_whole_window_truncated_inside_body_has_unknown_extent():
    rx = P4Receiver(admitted=True)
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, 0, 0.758))
    y = round(0.923 * FS)
    audio = np.r_[np.zeros(y), p4breakintx.changeover_packet(b"hello", 0)]
    (reg,) = rx.read_window(audio[: y + round(0.34 * FS)], 0, whole=True)
    assert reg.observation.extent_s is None


def test_acquisition_establishes_a_clock_and_counts_one_miss_per_slot():
    rx = P4Receiver(FS, admitted=True)
    audio = np.r_[np.zeros(FS // 10), p4robusttx.robust_word("X27"), np.zeros(FS // 10)]
    assert rx.read_window(audio, 0, whole=True)
    assert rx.clock is not None and rx.expected
    rx.new_cycle()
    due = rx.clock.next_due
    for _ in range(3):
        rx.read_window(np.zeros(FS), due - FS // 10)
    assert rx.clock.misses == 1


@pytest.mark.parametrize("value,header", [(13, "A"), (60, "B"), (49, "A"), (37, "B")])
def test_placed_short_value_and_provisional_resolution(value, header):
    rx = P4Receiver(FS, admitted=True)
    rx.note_emission(Emission(Kind.WORD, Role.IRS, 0, 0.244))
    onset = round(0.33 * FS)
    audio = np.r_[
        np.zeros(onset), p4robusttx.robust_short(value, header), np.zeros(FS // 10)
    ]
    early = onset + round(0.81 * FS)
    (partial,) = rx.read_partial(audio[:early], 0, deadline=early)
    assert partial.observation.kind == Kind.SHORT and partial.observation.provisional
    (full,) = rx.read_window(audio, 0, whole=True)
    assert full.outcome == Registered.CONFIRMED
    assert full.observation.value == value and full.observation.header == header
    assert abs(full.observation.stream_sample - onset) <= 2
    assert rx.read_window(audio, 0, whole=True) == []


def test_recorded_normal_is_named_at_its_lattice_without_claiming_bytes():
    from pathlib import Path
    from scipy.io import wavfile
    from hfmodem.shrike import p4short, p4rx

    fs, pcm = wavfile.read(
        Path(__file__).parent / "fixtures/p4-breakin/n9vc-normal.wav"
    )
    audio = np.r_[np.tile(pcm[:600], 20), pcm].astype(float) / 32768
    rx = P4Receiver(fs, admitted=True)
    rx.note_emission(Emission(Kind.WORD, Role.IRS, round(0.75 * fs), 0.244))
    (partial,) = rx.read_partial(
        audio[: round(1.89 * fs)], 0, deadline=round(1.89 * fs)
    )
    assert partial.observation.kind == Kind.NORMAL and partial.observation.provisional
    (full,) = rx.read_window(audio, 0, whole=True)
    obs = full.observation
    assert full.outcome == Registered.CONFIRMED and obs.kind == Kind.NORMAL
    assert obs.lattice >= 0.33 and obs.frame is None and obs.value is None
    assert obs.stream_sample / fs == pytest.approx(1.092, abs=0.003)
    z = p4rx.audio_to_baseband(audio, fs)
    assert (
        p4short.decode_body(
            z,
            first_training_start=round(obs.stream_sample * p4rx.SAMPLE_RATE / fs)
            + 140 * 16,
            frequency_offset=obs.cfo_hz,
        )
        is None
    )


@pytest.mark.parametrize("admitted", [False, True])
def test_passive_session_does_not_acquire_without_admission(admitted):
    rx = P4Receiver(admitted=admitted)
    audio = np.r_[np.zeros(FS // 10), p4robusttx.robust_word("Y27"), np.zeros(FS // 10)]
    assert bool(rx.read_window(audio, 0, whole=True)) == admitted


@pytest.mark.parametrize("name", ["x27", "v27", "u27", "t27"])
def test_recorded_bank_word_has_one_name(name):
    from pathlib import Path
    from scipy.io import wavfile
    from hfmodem.shrike import p4oracle, p4words

    fs, pcm = wavfile.read(
        Path(__file__).parent / "fixtures/p4-breakin" / f"n9vc-{name}.wav"
    )
    audio = pcm.astype(float) / 32768
    (obs,) = p4oracle.word_scan(audio, fs)
    assert obs.word == name.upper()
    placed = p4oracle.word_at(audio, fs, 0, obs.first_chip / fs, cfo_hz=obs.cfo_hz-p4words.CARRIER_HZ[obs.word])
    assert placed is not None and placed.word == obs.word
    assert abs(placed.stream_sample - obs.stream_sample) <= 2


def test_recorded_long_resolves_and_its_answer_uses_the_long_clock():
    from pathlib import Path
    from scipy.io import wavfile

    fs, pcm = wavfile.read(
        Path(__file__).parent / "fixtures/p4-breakin/n9vc-long-ack.wav"
    )
    audio = np.r_[np.tile(pcm[:600], 20), pcm].astype(float) / 32768
    rx = P4Receiver(fs, admitted=True)
    rx.note_emission(Emission(Kind.WORD, Role.IRS, round(0.75 * fs), 0.244))
    early = round(1.926 * fs)
    (partial,) = rx.read_partial(audio[:early], 0, deadline=early)
    assert partial.observation.kind == Kind.LONG
    (full,) = rx.read_window(audio[: round(4.40 * fs)], 0, whole=True)
    assert full.outcome == Registered.CONFIRMED and full.observation.kind == Kind.LONG
    assert rx.clock.cycle_s == 3.75
    # The same tape's X27 is addressed to the station that keyed this long.
    txrx = P4Receiver(fs, admitted=True, cfo_hz=4.8-3.65)
    txrx.note_emission(Emission(Kind.LONG, Role.ISS, round(1.08 * fs), 3.28))
    (word,) = txrx.read_long_answer(audio, 0)
    assert word.observation.word == "X27"
    assert (word.observation.stream_sample - round(1.08 * fs)) / fs == pytest.approx(
        3.414, abs=0.015
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_recorded_chirp_reader_is_opt_in_on_a_yielded_long_slot(enabled):
    from pathlib import Path
    from hfmodem.shrike import p4frame
    from hfmodem.shrike.p4receiver import Clock
    from hfmodem.tests.shrike.test_p4chirp import REAL_FIELDS

    path = Path(__file__).parent / "fixtures/sigidwiki/p4-chirps/audio.npz"
    if not path.exists():
        pytest.skip(f"{path} is absent")
    with np.load(path) as fixture:
        fs = int(fixture["fs"])
        for i, row in enumerate(fixture["audio"]):
            audio = np.r_[np.zeros(fs // 3), row, np.zeros(fs // 3)]
            rx = P4Receiver(fs, admitted=True, chirp=enabled)
            rx.expected = True
            rx.clock = Clock(Expect.PACKET, fs // 3, 0.33, 3.75, -242.4)
            records = rx.read_window(audio, 0, whole=True)
            if not enabled:
                assert records == []
            else:
                (obs,) = records
                assert obs.observation.kind == Kind.CHIRP
                frame = obs.observation.frame
                assert p4frame.encode_field(frame.data, frame.status) == REAL_FIELDS[i]


def test_local_1228_replay_names_each_captured_answer_in_cycle():
    import json
    from pathlib import Path
    from hfmodem.shrike import onair

    root = Path(__file__).resolve().parents[5] / "captures/onair-0926-1228"
    if not (root / "stream.flac").exists():
        root = Path(__file__).resolve().parents[5].parents[2] / "captures/onair-0926-1228"
    if not (root / "stream.flac").exists():
        pytest.skip("local onair-0926-1228 stream.flac recording is absent")
    meta = json.loads((root / "p4-probe.json").read_text())
    live = onair._ReplayInput(str(root / "stream.flac"))
    receiver = P4Receiver(FS, admitted=True)
    heard = []
    for sent in meta["emissions"]:
        receiver.note_emission(
            Emission(Kind.SHORT, Role.ISS, sent["audio_start"], 0.758)
        )
        live.take_until(sent["audio_end"])
        begin = live.pos
        audio = live.take_until(sent["audio_start"] + round(1.25 * FS))
        heard.extend(
            r.observation.word
            for r in receiver.read_window(audio, begin, whole=True)
            if r.observation.kind == Kind.WORD
        )
    assert heard == ["X27", "Y27", "Y27", "Y27"]


@pytest.mark.parametrize('name', ['T27', 'X27', 'V27'])
@pytest.mark.parametrize('reference', ['unknown', 'known-local', 'synthetic-peer'])
def test_rendered_words_acquire_and_reacquire_without_changing_peer_offsets(name, reference, monkeypatch):
    from scipy.signal import hilbert
    from hfmodem.shrike import p4oracle, p4words
    body_cfo = 0. if reference == 'known-local' else -5.
    wave = p4robusttx.robust_word(name)
    carrier = body_cfo if reference != 'unknown' else 0.
    if reference == 'synthetic-peer':
        carrier += p4words.CARRIER_HZ[name]
    wave = (hilbert(wave)*np.exp(2j*np.pi*carrier*np.arange(len(wave))/FS)).real
    rx = P4Receiver(FS, admitted=True, cfo_hz=None if reference == 'unknown' else body_cfo)
    rx.enable_hold()
    rx._body_carrier_known = reference != 'unknown'
    rx.note_emission(Emission(Kind.LONG, Role.ISS, 0, 3.28))
    assert rx.clock.expects == Expect.WORD and 'T27' in rx.expected_words
    onset = rx.clock.next_due
    audio = np.r_[np.zeros(onset), wave, np.zeros(FS//10)]
    calls = []
    word_at = p4oracle.word_at
    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return word_at(*args, **kwargs)
    monkeypatch.setattr(p4oracle, 'word_at', tracked)
    rows = rx.read_window(audio, 0, whole=True)
    final = [r.observation for r in rows if r.observation.kind == Kind.WORD and not r.observation.provisional]
    assert final and {o.word for o in final} == {name}
    obs = final[-1]
    assert obs.score > p4words.MATCH_KNEE and abs(obs.stream_sample-onset) <= 8
    assert obs.cfo_hz == pytest.approx(carrier, abs=.6)
    assert rx.cfo_hz == pytest.approx(carrier-p4words.CARRIER_HZ[name], abs=.6)
    if reference == 'known-local':
        assert any(c.get('reacquire') for c in calls)
        if name in ('T27', 'X27'):
            assert len(calls) > 1
    if reference == 'synthetic-peer':
        assert rx.cfo_hz == pytest.approx(body_cfo, abs=.6)
    assert all(c.get('carrier_offsets') is None for c in calls)
