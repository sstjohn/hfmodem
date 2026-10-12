# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import math
import shutil
import struct
import sys
import wave

import pytest

from hfhost import audio
from hfhost.audio import FS, FRAME_SAMPLES, WavReplaySource


def _write_wav(path, seconds, *, rate=FS, channels=1):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        # a quiet ramp, enough to be non-trivial without needing numpy
        frames = b"".join(
            struct.pack("<h", (i % 100 - 50) * 100) * channels
            for i in range(n))
        w.writeframes(frames)


def test_replay_yields_48k_mono_frames_on_a_common_clock(tmp_path):
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.5)
    src = WavReplaySource(wav, realtime=False)

    frames = list(src.frames())
    assert frames, "replay produced no audio"
    # timeline is contiguous in samples, first frame at 0
    assert frames[0][0] == 0
    total = 0
    for t0, pcm in frames:
        assert t0 == total
        assert len(pcm) % 2 == 0
        total += len(pcm) // 2
    assert total == int(0.5 * FS)
    # full frames are FRAME_SAMPLES; only the tail may be short
    assert all(len(p) == FRAME_SAMPLES * 2 for _, p in frames[:-1])


def test_replay_missing_file_is_a_clean_error(tmp_path):
    src = WavReplaySource(tmp_path / "nope.wav", realtime=False)
    with pytest.raises(audio.AudioError):
        list(src.frames())


@pytest.mark.realtime
def test_replay_realtime_paces_to_the_wall_clock(tmp_path):
    import time
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.3)
    src = WavReplaySource(wav, realtime=True, speed=10.0)   # 10x to keep it quick
    start = time.monotonic()
    list(src.frames())
    elapsed = time.monotonic() - start
    # ~0.03 s of paced replay; generously bounded, just proves it does pace
    assert 0.005 < elapsed < 0.5


def _stored(wav):
    """`hfmodem.core.wav.store`, by hand: this package cannot import it."""
    import subprocess
    subprocess.run(["flac", "-s", "--keep-foreign-metadata", "-o",
                    str(wav.with_suffix(".flac")), str(wav)], check=True)
    wav.unlink()


@pytest.mark.skipif(shutil.which("flac") is None, reason="no flac on PATH")
@pytest.mark.parametrize("name", ["cap.wav", "cap.flac"])
def test_a_recording_stored_as_flac_replays_as_it_did_before(tmp_path, name):
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.5)
    before = list(WavReplaySource(wav, realtime=False).frames())
    _stored(wav)

    assert list(WavReplaySource(tmp_path / name, realtime=False).frames()) == before


@pytest.mark.skipif(shutil.which("flac") is None, reason="no flac on PATH")
def test_a_stored_recording_keeps_its_rate_and_window(tmp_path):
    wav = tmp_path / "kiwi.wav"
    _write_wav(wav, 0.1, rate=12000)
    _stored(wav)

    assert audio.file_rate(wav) == 12000
    assert WavReplaySource(wav, realtime=False).window.hi_hz == 6000.0


def test_a_flac_with_no_flac_installed_says_so(tmp_path, monkeypatch):
    (tmp_path / "cap.flac").write_bytes(b"fLaC")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(audio.AudioError, match="brew install flac"):
        list(WavReplaySource(tmp_path / "cap.wav", realtime=False).frames())


def test_source_is_a_context_manager(tmp_path):
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.1)
    with WavReplaySource(wav, realtime=False) as src:
        assert list(src.frames())


# -- audio off another process -------------------------------------------------

def test_a_command_source_reads_s16le_off_its_stdout():
    """The station's capture runs in a process of its own — PortAudio needs a
    library hfhost may not import — so what this has to get right is the pipe,
    not the card."""
    n = 3 * FRAME_SAMPLES
    src = audio.CommandSource(
        [sys.executable, "-c",
         f"import sys; sys.stdout.buffer.write(bytes(2 * {n}))"])
    frames = list(src.frames())

    assert [t0 for t0, _ in frames] == [0, FRAME_SAMPLES, 2 * FRAME_SAMPLES]
    assert sum(len(p) for _, p in frames) == 2 * n


def test_a_command_that_produces_nothing_ends_the_stream_rather_than_hanging():
    src = audio.CommandSource([sys.executable, "-c", ""])
    assert list(src.frames()) == []


# -- windows -------------------------------------------------------------------

def test_a_wav_with_no_dial_has_its_own_span_and_no_rf_mapping(tmp_path):
    wav = tmp_path / "kiwi.wav"
    _write_wav(wav, 0.1, rate=12000)
    window = WavReplaySource(wav, realtime=False).window
    assert (window.lo_hz, window.hi_hz) == (0.0, 6000.0)
    assert window.rf_hz(1500.0) is None


def test_a_wav_given_a_dial_maps_by_its_sideband(tmp_path):
    wav = tmp_path / "rig.wav"
    _write_wav(wav, 0.1)
    assert WavReplaySource(wav, dial_hz=7_100_000).window.rf_hz(1500) == 7_101_500
    assert WavReplaySource(wav, dial_hz=7_100_000, mode="PKTLSB").window.rf_hz(1500) \
        == 7_098_500


def test_a_float_wav_reports_its_rate_without_the_wave_module(tmp_path):
    wav = tmp_path / "float.wav"
    fmt = struct.pack("<HHIIHH", 3, 1, 20250, 20250 * 4, 4, 32)
    data = struct.pack("<4f", 0.0, 0.1, -0.1, 0.0)
    wav.write_bytes(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data))
                    + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
                    + b"data" + struct.pack("<I", len(data)) + data)
    assert audio.file_rate(wav) == 20250
    assert WavReplaySource(wav).window.hi_hz == 10125.0


def test_a_command_source_carries_the_window_it_was_given():
    window = audio.Window.sideband(300, 2900, "rig", 7_100_000, "PKTUSB")
    src = audio.CommandSource([sys.executable, "-c", ""], window=window)
    assert src.window.rf_hz(src.window.lo_hz) == 7_100_300
    assert audio.CommandSource([sys.executable, "-c", ""]).window.rf_hz(0) is None


# -- the Kiwi, IQ and sideband, over a fake stream -------------------------------

KIWI_RATE = 12000
DIAL_KHZ = 7104.0


class _FakeKiwi:
    """What `KiwiStream.open` returns, playing back a synthetic span."""

    def __init__(self, payload: bytes, rate: float = KIWI_RATE, block: int = 2048):
        self.payload, self.sample_rate, self.block = payload, rate, block
        self.asked: dict = {}
        self.closed = False

    def __call__(self, host, port, dial_khz, **kw):
        self.asked = dict(host=host, port=port, dial_khz=dial_khz, **kw)
        return self

    def blocks(self):
        for i in range(0, len(self.payload), self.block):
            yield self.payload[i:i + self.block]

    def close(self):
        self.closed = True


def _iq_tones(tones, seconds, rate=KIWI_RATE, amp=8000):
    """Big-endian interleaved I/Q of complex tones at baseband offsets (Hz)."""
    out = bytearray()
    for n in range(int(seconds * rate)):
        i = q = 0.0
        for f in tones:
            ph = 2 * math.pi * f * n / rate
            i += amp * math.cos(ph)
            q += amp * math.sin(ph)
        out += struct.pack(">hh", round(i), round(q))
    return bytes(out)


def _samples(src):
    pcm = b"".join(p for _, p in src.frames())
    return struct.unpack(f"<{len(pcm) // 2}h", pcm)


def _power(x, f, fs=FS):
    """Goertzel power at f, normalised to a full-scale sinusoid's."""
    w = 2 * math.cos(2 * math.pi * f / fs)
    s1 = s2 = 0.0
    for v in x:
        s1, s2 = v + w * s1 - s2, s1
    return (s1 * s1 + s2 * s2 - w * s1 * s2) / (len(x) / 2) ** 2


def _frequency(x, fs=FS):
    """A clean tone's frequency, off its interpolated rising zero crossings."""
    ups = [i + x[i] / (x[i] - x[i + 1]) for i in range(len(x) - 1)
           if x[i] < 0 <= x[i + 1]]
    return (len(ups) - 1) * fs / (ups[-1] - ups[0])


def test_an_iq_span_comes_out_real_at_48k_with_the_dial_where_it_belongs():
    offset = 2234.5
    fake = _FakeKiwi(_iq_tones([offset], 1.5))
    src = audio.KiwiSource("kiwi.example", DIAL_KHZ, connect=fake)

    x = _samples(src)
    assert abs(len(x) - 1.5 * FS) < FS * 0.01

    steady = x[FS // 4: FS // 4 + FS]
    heard = _frequency(steady)
    assert abs(heard - (6000 + offset)) < 0.05
    assert src.window.lo_hz == 0 and src.window.hi_hz == 12000
    assert abs(src.window.rf_hz(heard) - (DIAL_KHZ * 1e3 + offset)) < 0.05

    image = 6000 - offset
    assert _power(steady, image) < 1e-4 * _power(steady, 6000 + offset)
    assert fake.asked["mode"] == "iq"
    assert (fake.asked["lo_hz"], fake.asked["hi_hz"]) == (-6000, 6000)
    assert fake.asked["wideband"] is False
    assert fake.closed


def test_a_tone_below_the_dial_lands_below_the_middle_of_the_window():
    fake = _FakeKiwi(_iq_tones([-3000.0], 1.0))
    src = audio.KiwiSource("kiwi.example", DIAL_KHZ, connect=fake)
    heard = _frequency(_samples(src)[FS // 4:])
    assert abs(heard - 3000.0) < 0.05
    assert abs(src.window.rf_hz(heard) - (DIAL_KHZ * 1e3 - 3000)) < 0.05


class _Rates(_FakeKiwi):
    """A Kiwi whose ordinary and wideband channels run at different rates."""

    def __init__(self, ordinary, wideband):
        super().__init__(_iq_tones([1000.0], 0.5, rate=wideband), rate=ordinary)
        self.rates, self.calls = (ordinary, wideband), []

    def __call__(self, host, port, dial_khz, **kw):
        self.calls.append(kw["wideband"])
        self.sample_rate = self.rates[kw["wideband"]]
        return super().__call__(host, port, dial_khz, **kw)


def test_a_wider_span_reaches_for_the_wideband_connection_only_when_it_must():
    fake = _Rates(12000, 24000)
    src = audio.KiwiSource("kiwi.example", DIAL_KHZ, span=20000, connect=fake)
    heard = _frequency(_samples(src)[FS // 4:])
    assert fake.calls == [False, True]
    assert (fake.asked["lo_hz"], fake.asked["hi_hz"]) == (-10000, 10000)
    assert abs(heard - 11000.0) < 0.05
    assert abs(src.window.rf_hz(heard) - (DIAL_KHZ * 1e3 + 1000)) < 0.05

    wide_already = _Rates(20250, 20250)
    list(audio.KiwiSource("kiwi.example", DIAL_KHZ, span=20000,
                          connect=wide_already).frames())
    assert wide_already.calls == [False]


def test_a_kiwi_that_cannot_serve_the_span_says_so():
    fake = _Rates(12000, 12000)
    src = audio.KiwiSource("kiwi.example", DIAL_KHZ, span=20000, connect=fake)
    with pytest.raises(audio.AudioError, match="short of the 20000 Hz span"):
        list(src.frames())
    assert fake.calls == [False, True]


def test_a_kiwi_clock_trimmed_rate_serves_its_nominal_span():
    fake = _Rates(11998.874085, 11998.874085)
    list(audio.KiwiSource("kiwi.example", DIAL_KHZ, span=12000, connect=fake).frames())
    assert fake.calls == [False]


def test_the_sideband_mode_is_what_a_rig_would_hear():
    rate, f = KIWI_RATE, 1500.0
    mono = b"".join(struct.pack(">h", round(8000 * math.sin(2 * math.pi * f * n / rate)))
                    for n in range(rate))
    fake = _FakeKiwi(mono)
    src = audio.KiwiSource("kiwi.example", DIAL_KHZ, mode="usb", span=3000,
                           connect=fake)
    heard = _frequency(_samples(src)[FS // 4:])
    assert abs(heard - f) < 0.05
    assert src.window.rf_hz(heard) == pytest.approx(DIAL_KHZ * 1e3 + f, abs=0.05)
    assert (fake.asked["lo_hz"], fake.asked["hi_hz"]) == (0, 3000)


def test_a_sideband_wider_than_six_kilohertz_is_refused():
    with pytest.raises(audio.AudioError, match="wants IQ"):
        audio.KiwiSource("kiwi.example", DIAL_KHZ, mode="usb", span=8000)


def test_a_48k_recording_is_windowed_to_the_rig_filter_not_the_codecs_span(tmp_path):
    wav = tmp_path / "rig.wav"
    _write_wav(wav, 0.1)
    window = WavReplaySource(wav).window
    assert (window.lo_hz, window.hi_hz) == audio.RIG_FILTER_HZ == (300.0, 2900.0)
    assert WavReplaySource(wav, filter_hz=(200, 3100)).window.hi_hz == 3100


def test_an_explicit_window_overrides_the_recordings_own(tmp_path):
    wav = tmp_path / "rig.wav"
    _write_wav(wav, 0.1)
    span = audio.Window(0.0, 12000.0, "synthetic 12 kHz", 7_098_000)
    assert WavReplaySource(wav, window=span).window is span
