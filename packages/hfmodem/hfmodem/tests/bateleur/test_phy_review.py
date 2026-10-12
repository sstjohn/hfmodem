# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Deterministic reproductions of the PHY import review failures."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import welch

from hfmodem.bateleur import fec, mil110, s4285, s4539, spec, stream, tone16, tone39
from hfmodem.bateleur.modem import BateleurModem, Mode


def _welch_span(audio, fs):
    n = round(1.5 * fs / 30)
    frequencies, power = welch(audio, fs=fs, window="hann", nperseg=n,
                               noverlap=n // 2, nfft=16384)
    occupied = frequencies[power >= power.max() * 10 ** (-26 / 10)]
    return occupied[0], occupied[-1]


@pytest.mark.parametrize("find", [mil110.find_eom, s4285.find_eom])
def test_payload_marker_requires_flush(find):
    marker = np.array(spec.EOM110_BITS, dtype=np.uint8)
    near = marker.copy()
    near[[2, 13, 27]] ^= 1
    for candidate in (marker, near):
        assert find(np.r_[candidate, np.ones(144, dtype=np.uint8)]) == -1
        assert find(np.r_[candidate, np.zeros(64, dtype=np.uint8)]) == 0
        assert find(np.r_[candidate, np.zeros(63, dtype=np.uint8)]) == -1


@pytest.mark.parametrize("find", [mil110.find_eom, s4285.find_eom])
@pytest.mark.parametrize("offset", range(1, 8))
def test_bit_stream_eom_at_each_offset(find, offset):
    bits = np.r_[np.ones(offset, dtype=np.uint8), spec.EOM110_BITS, np.zeros(144)]
    assert find(bits) == offset


def test_front_end_accepts_empty_and_sub_decimation_chunks():
    x = np.random.default_rng(19).normal(size=87)
    whole = stream.FrontEnd(48000).push(x)
    fe = stream.FrontEnd(48000)
    assert fe.push([]).size == 0 and fe.audio.size == 0
    pieces = [fe.push(x[i:i + 1]) for i in range(x.size)]
    np.testing.assert_allclose(np.concatenate(pieces), whole, atol=1e-14)


@pytest.mark.parametrize("standard,name,seconds", [
    ("110", "2400Z", 30), ("110", "4800N", 30),
    ("4285", "1200S", 12), ("4539", "6400US", 40),
])
def test_duration_includes_every_filter_sample(standard, name, seconds):
    mode = Mode.parse(standard, name)
    n = mode.most_bytes(seconds)
    for trim in (False, True):
        actual = mode.render(bytes(n), trim=trim).size / spec.FS
        assert mode.seconds(n) == pytest.approx(actual, abs=1 / spec.FS)
        assert actual <= seconds


def test_receiver_failure_does_not_drop_neighbours_chunk():
    log, got = [], []
    modem = BateleurModem(lambda *_: "", card_rate=spec.FS, max_key_s=30, log=log.append)
    broken = modem._rx110
    def fail(_):
        raise ValueError("injected receive failure")
    broken.push = fail
    modem._rx4539.push = lambda x: got.append(x.copy()) or []
    modem._rx39.push = lambda x: []
    modem._audio(np.zeros(1024))
    assert len(got) == 1 and got[0].size == 1024
    assert modem._rx110 is not broken
    assert any("110" in line and "ValueError" in line for line in log)


def test_somless_4285_keeps_its_first_decoded_bits():
    rx = stream.Receiver4285("600S")
    rx._bits = np.zeros(0, dtype=np.uint8)
    rx._decoded_n, rx._msg_at, rx._sent = 0, None, 0
    rx._bytes = stream.Bytes()
    rx._vit = fec.ViterbiStream()
    bits = np.random.default_rng(39).integers(0, 2, rx.mode.flush_bits + 500, dtype=np.uint8)
    events = []
    for i in range(0, bits.size, 64):
        assert not rx._place(bits[i:i + 64], events)
    out = b"".join(e.data for e in events if isinstance(e, stream.Data))
    assert len(out) >= 50
    assert out == mil110.bits_to_bytes(bits)[:len(out)]


def test_39_z_one_failed_word_keeps_later_words(monkeypatch):
    m = tone39.mode("2400Z")
    data = bytes(range(100))
    coded = tone39.coded_stream(m, data)
    original = tone39.decode_superblock
    calls = 0
    def one_failed(mode, bits):
        nonlocal calls
        calls += 1
        out, fixed, failed = original(mode, bits)
        return out, fixed, 1 if calls == 2 else failed
    monkeypatch.setattr(tone39, "decode_superblock", one_failed)
    got, eom, _, _, failed = tone39._decode_stream(1.0 - 2.0 * coded, m)
    assert got == data and eom and failed == 1


def test_39_async_stop_is_mark_and_seven_bit_parity_passes_through():
    for name in ("2400/288/A10.8", "2400/288/A10.7"):
        m = tone39.mode(name)
        chars = tone39.chars(bytes([0x00, 0x80, 0xff]), m).reshape(-1, m.char)
        assert np.all(chars[:, -1] == 1)
        assert np.all(chars[:, 0] == 0)
        if m.data_bits == 7:
            assert list(chars[:, 8]) == [0, 1, 1]


def test_39_default_receiver_searches_extended_preamble():
    m = replace(tone39.mode("600/33"), extended=True)
    msg = bytes(range(100))
    x = np.r_[np.zeros(1000), tone39.transmit(msg, m), np.zeros(2 * spec.FS)]
    rx = stream.Receiver39("600/33")
    events = []
    for i in range(0, x.size, 2400):
        events.extend(rx.push(x[i:i + 2400]))
    assert b"".join(e.data for e in events if isinstance(e, stream.Data)) == msg
    assert any(isinstance(e, stream.Ended) and e.complete for e in events)


def test_16_tone_75_spectrum_and_crest_factor():
    mode = Mode.parse("16", "75")
    msg = np.random.default_rng(31).integers(0, 256, mode.most_bytes(30), dtype=np.uint8).tobytes()
    x = tone16.transmit(msg, 75)
    f, p = welch(x, fs=spec.FS, nperseg=480, nfft=8192)
    occupied = f[p >= p.max() * 10 ** (-26 / 10)]
    assert occupied[0] >= 365 and occupied[-1] <= 2985
    assert 10 * np.log10(np.max(x*x) / np.mean(x*x)) < 10


@pytest.mark.parametrize("standard,name,receiver", [
    ("110", "2400S", stream.Receiver110), ("4285", "1200S", stream.Receiver4285),
    ("4539", "6400US", stream.Receiver4539), ("39", "600/33", stream.Receiver39),
    ("16", "1200", stream.Receiver16),
])
def test_batch_receive_uses_streaming_decoder(monkeypatch, standard, name, receiver):
    msg = b"same receiver, arbitrary chunk boundaries" * 3
    audio = Mode.parse(standard, name).render(msg)
    original = receiver.push
    calls = []
    def chunked(self, samples):
        calls.append(len(samples))
        return [event for i in range(0, len(samples), 731)
                for event in original(self, samples[i:i + 731])]
    monkeypatch.setattr(receiver, "push", chunked)
    receive = {"110": lambda: mil110.receive(audio),
               "4285": lambda: s4285.receive(audio, name),
               "4539": lambda: s4539.receive(audio),
               "39": lambda: tone39.receive(audio, name),
               "16": lambda: tone16.receive(audio, name)}
    result = receive[standard]()
    assert calls and result is not None
    assert result.data[:len(msg)] == msg


@pytest.mark.parametrize("name", ["2400S", "600S", "2400Z"])
def test_last_110_block_has_no_next_block_identification(name):
    m = spec.MODES110[name]
    symbols = mil110.data_symbols(m, b"last block")
    clear = (symbols - np.resize(spec.scramble110(), symbols.size)) % 8
    frames = clear.reshape(-1, m.unknown + m.known)
    assert np.all(frames[-2:, m.unknown:] == 0)


def test_rendered_oversize_piece_is_split_before_submission(monkeypatch):
    submitted = []
    modem = BateleurModem(lambda x, fs: submitted.append(len(x)) or "", card_rate=spec.FS,
                         max_key_s=1, log=lambda _: None)
    monkeypatch.setattr(Mode, "most_bytes", lambda *_: 20)
    monkeypatch.setattr(Mode, "render", lambda self, data, *args: np.zeros(1000 * len(data)))
    modem.transmit(b"x" * 25)
    assert submitted and max(submitted) <= spec.FS
    assert sum(submitted) == 25000


def test_ale_declared_span_uses_welch_power_density():
    from hfmodem.bateleur.ale import fsk, words
    from hfmodem.core.occupied import keyed_hz
    lo, hi = keyed_hz("ale")
    frames = [words.call("JOE", "SAM", scan_words=n) for n in (2, 20, 60)]
    frames += [np.random.default_rng(seed).integers(0, 1 << 24, n).tolist()
               for seed in (*range(32), 47, 48) for n in (1, 100)]
    for frame in frames:
        a, b = _welch_span(fsk.modulate(frame), fsk.FS)
        assert lo <= a and b <= hi


def test_39_short_burst_stays_in_declared_span_and_duration():
    from hfmodem.core.occupied import keyed_hz
    lo, hi = keyed_hz("bateleur", "3000")
    for name in ("1200/63/S", "300/17/SF"):
        m = Mode.parse("39", name)
        audio = m.render(b"\xff")
        a, b = _welch_span(audio, spec.FS)
        assert lo <= a and b <= hi
        assert audio.size / spec.FS == pytest.approx(m.seconds(1))


def test_measurement_counts_missing_zero_bits_as_errors(monkeypatch):
    from hfmodem.bateleur import measure
    monkeypatch.setattr(measure, "_waveform", lambda _: (
        lambda mode: 128, lambda data, mode: np.zeros(2400), lambda audio, mode: None))
    errors, bits, _, _ = measure.measure("110", "2400S", "awgn", 10,
                                       trials=1, seconds=1, workers=1)
    assert errors == bits == 128
    assert measure.error_count(b"\x00\xff", b"") == (16, 16)
    assert measure.error_count(b"\x01\x00", b"\x00") == (9, 8)
    assert measure.upper95(0, 96000) == pytest.approx(1 - .05 ** (1 / 96000))


def test_measurement_records_sweep_conditions_and_counts(monkeypatch, tmp_path):
    import json
    from hfmodem.bateleur import measure
    seen = []
    def trial(args):
        seen.append(args)
        return dict(bits=800, errors=0, missing_bits=0, realtime_factor=0,
                    seed=args[5], eom=True)
    monkeypatch.setattr(measure, "_one", trial)
    output = tmp_path / "sweep.json"
    measure.main(["--preset", "c6-4", "--trials", "1", "--output", str(output)])
    record = json.loads(output.read_text())
    assert [job[7] for job in seen] == [-3.5, 3.5]
    assert all(job[4] == 96 for job in seen)
    assert record["source_sha256"]["sim.py"]
    assert all(row["bits"] == 800 and row["errors"] == 0
               and row["independent_bit_upper95"] > 1e-5 for row in record["measurements"])


def test_39_pilot_phase_continues_from_part_three(monkeypatch):
    phases = []
    original = tone39._element
    def record(freqs, amps, phase, t_on):
        if len(freqs) == 40:
            phases.append(phase[0])
        return original(freqs, amps, phase, t_on)
    monkeypatch.setattr(tone39, "_element", record)
    tone39.transmit(b"phase", "600/33")
    expected = 2 * np.pi * tone39.DOPPLER_HZ * tone39.T / spec.FS
    assert np.angle(np.exp(1j * (phases[1] - phases[0] - expected))) == pytest.approx(0)


def test_serial_receiver_storage_stays_bounded_during_message(monkeypatch):
    monkeypatch.setattr(mil110, "find_eom", lambda *args, **kwargs: -1)
    msg = np.random.default_rng(42).bytes(18000)
    audio = mil110.transmit(msg, "4800N")
    rx, fe = stream.Receiver110(), stream.FrontEnd(spec.FS)
    data = bytearray()
    peak = 0
    for at in range(0, len(audio), spec.FS):
        for event in rx.push(fe.push(audio[at:at + spec.FS])):
            if isinstance(event, stream.Data):
                data.extend(event.data)
        if rx.receiving:
            peak = max(peak, np.asarray(rx.plan.kind).size)
            assert len(rx.buf) < 4 * spec.FS
            assert len(rx._bits) < 200
    assert len(data) > 16000 and bytes(data) == msg[:len(data)]
    assert peak < 20000


def test_somless_4285_audio_keeps_payload_from_the_start(monkeypatch):
    msg = np.random.default_rng(81).bytes(240)
    original = s4285._info_bits
    def without_som(mode, data, order="lsb"):
        framed = original(mode, data, order)
        return np.r_[framed[32:], np.zeros(32, dtype=np.uint8)]
    monkeypatch.setattr(s4285, "_info_bits", without_som)
    audio = np.r_[s4285.transmit(msg, "600S"), np.zeros(4 * spec.FS)]
    rx, fe, events = stream.Receiver4285("600S"), stream.FrontEnd(spec.FS), []
    for i in range(0, len(audio), 997):
        events.extend(rx.push(fe.push(audio[i:i + 997])))
    assert b"".join(e.data for e in events if isinstance(e, stream.Data)) == msg
    assert any(isinstance(e, stream.Ended) and e.complete for e in events)


@pytest.mark.parametrize("standard,name,transmit", [
    ("16", "600", tone16.transmit), ("39", "600/33", tone39.transmit),
])
def test_parallel_render_has_one_amplitude_convention(standard, name, transmit):
    audio = transmit(b"one drive setting", name)
    assert np.max(np.abs(audio)) == pytest.approx(1.0)
    np.testing.assert_array_equal(audio, Mode.parse(standard, name).render(b"one drive setting"))


def test_measured_39_framing_survives_a_split_insertion():
    m = tone39.Mode39(75, 4, char=0, seq=384)
    msg = bytes(range(24))
    coded = np.r_[tone39.coded_stream(m, msg[:12]), tone39.coded_stream(m, msg[12:])]
    rx = tone39.Decoder(m)
    soft = 1.0 - 2.0 * coded
    got = b"".join(rx.push(soft[i:i + 37]) for i in range(0, len(soft), 37))
    assert got == msg and rx.failed == 0
