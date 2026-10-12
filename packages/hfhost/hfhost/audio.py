# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Live receive audio, as a source you can fan to several decoders at once.

The whole point is that the primary path is a *stream*, not a file: one rig (or
one remote receiver) is producing audio right now, and several modem monitors
each want to hear it. So a source yields a continuous run of fixed-size PCM
frames on a common sample clock, and the same interface covers a live sound
card, a KiwiSDR over the network, and — only for tests and replay — a recorded
WAV played back at the pace it was captured.

Frames are raw ``s16le`` mono at 48 kHz. Bytes, not arrays, on purpose: this
layer stays stdlib-only, and every consumer here forwards the audio to a decoder
subprocess that owns its own numpy. The one place a sample rate other than 48 k
or a non-mono file appears, ffmpeg is asked to normalise it — the sources never
grow a DSP of their own.

Nothing here ever transmits. Every source opens a capture and only a capture.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from . import kiwi

FS = 48000
CHANNELS = 1
SAMPLE_WIDTH = 2                       # s16le
FRAME_SAMPLES = 4800                   # ~0.1 s, matches the modems' live read
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH
RIG_FILTER_HZ = (300.0, 2900.0)         # config.Rig.filter_hz's default, for a bare hfhost


class AudioError(RuntimeError):
    pass


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError("ffmpeg not found (macOS: brew install ffmpeg; "
                         "Debian/Ubuntu: apt install ffmpeg)")
    return exe


@dataclass(frozen=True, slots=True)
class Window:
    """Where a source's audio sits: the span worth analysing, and on the dial.

    ``rf_origin_hz`` is the dial frequency audio 0 Hz stands for, ``None`` when
    nobody knows it. ``inverted`` is a lower sideband, where audio climbs as RF
    falls.
    """
    lo_hz: float
    hi_hz: float
    label: str
    rf_origin_hz: float | None = None
    inverted: bool = False

    def rf_hz(self, audio_hz: float) -> float | None:
        if self.rf_origin_hz is None:
            return None
        return self.rf_origin_hz + (-audio_hz if self.inverted else audio_hz)

    @classmethod
    def sideband(cls, lo_hz: float, hi_hz: float, label: str,
                 dial_hz: float | None, mode: str = "USB") -> "Window":
        """A receiver's filter behind a dial read in ``mode`` (USB, PKTLSB, ...)."""
        return cls(lo_hz, hi_hz, label, dial_hz, "LSB" in mode.upper())


class AudioSource:
    """A continuous run of ``(t0_samples, pcm)`` frames, s16le mono @ 48 kHz.

    ``t0_samples`` is the sample index of the frame's first sample from the
    start of the stream, so every consumer shares one clock regardless of which
    source produced the audio. Subclasses implement :meth:`frames`; the base
    handles the context-manager and teardown contract. ``window`` says what the
    audio covers, and nothing above a source needs to know which kind it is.
    """

    window: Window = Window(0.0, FS / 2, "unlabelled")

    def frames(self) -> Iterator[tuple[int, bytes]]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "AudioSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _ProcSource(AudioSource):
    """Common machinery for sources that read s16le from a subprocess pipe."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def _spawn(self) -> subprocess.Popen:
        raise NotImplementedError

    def frames(self) -> Iterator[tuple[int, bytes]]:
        self._proc = self._spawn()
        consumed = 0
        assert self._proc.stdout is not None
        try:
            while True:
                raw = self._proc.stdout.read(FRAME_BYTES)
                if not raw:
                    break
                # A partial final read still carries whole samples; only pad an
                # odd trailing byte, which a killed pipe can leave.
                if len(raw) % SAMPLE_WIDTH:
                    raw = raw[:-(len(raw) % SAMPLE_WIDTH)]
                if raw:
                    yield consumed, raw
                    consumed += len(raw) // SAMPLE_WIDTH
        finally:
            self.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


class CommandSource(_ProcSource):
    """s16le mono @ 48 kHz off any command's stdout.

    What produces the samples is the caller's business, and it has to be: this
    station captures through PortAudio, which needs a library hfhost may not
    import. ffmpeg's own capture inputs are not a substitute. Measured on this
    station's codec 2026-08-14 and again over six live passes on 25 and 26
    August, avfoundation delivered 0.877-0.904 of real time in samples --
    unchanged by ``-drop_late_frames``, ``-thread_queue_size`` or ``-c:a copy``
    -- while PortAudio on the same card in the same hour kept 0.9979. So the
    missing audio was never recoverable by any flag, and the process that opens
    the card is named from outside; ``creance monitor`` points this at
    ``creance/monitor/runners/capture_runner.py``.

    A consumer still has to account for what actually arrived rather than assume
    a continuous stream -- ``creance monitor`` prints the kept fraction of every
    live pass -- because no capture library makes that question go away.
    """

    def __init__(self, cmd: list[str], window: Window | None = None) -> None:
        super().__init__()
        self.cmd = list(cmd)
        if window is not None:
            self.window = window

    def _spawn(self) -> subprocess.Popen:
        # Its own session: a terminal's ^C is the consumer's to act on.
        return subprocess.Popen(self.cmd, stdout=subprocess.PIPE, start_new_session=True)


class KiwiSource(_ProcSource):
    """Live audio from a remote KiwiSDR, no local rig.

    IQ mode by default, so the span is whatever the Kiwi serves: 12 kHz on any
    Kiwi, more over the wideband connection on one that offers it. The complex
    span is resampled to 48 kHz, mixed up by half its width and its real part
    taken, which puts the dial's lower edge at audio 0 Hz and holds any span to
    24 kHz. The window is then ``0..span`` and ``rf_hz`` is
    ``dial - span/2 + audio``.

    ``mode="usb"`` (or ``"lsb"``) at a span of 6000 Hz or less is what a rig
    would hear instead: that sideband, filtered to ``passband``, resampled.

    ffmpeg does the arithmetic so this layer stays free of numpy; the stream is
    :class:`hfhost.kiwi.KiwiStream`, and ``connect`` exists so a test can hand
    in a fake one without a socket.
    """

    SIDEBAND_SPAN = 6000
    #: How far under the span a served IQ rate may sit: a Kiwi's clock trim.
    CLOCK_SLACK = 1e-3

    def __init__(self, host: str, freq_khz: float, *, port: int = 8073,
                 span: int = 12000, mode: str = "iq",
                 passband: tuple[int, int] | None = None, gain: int = 50,
                 ident: str = "hfmodem",
                 connect: Callable[..., kiwi.KiwiStream] = kiwi.KiwiStream.open
                 ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode != "iq" and span > self.SIDEBAND_SPAN:
            raise AudioError(f"{mode} is for spans of {self.SIDEBAND_SPAN} Hz or less; "
                             f"{span} Hz wants IQ")
        if span > FS // 2:
            raise AudioError(f"a {span} Hz span does not fit 48 kHz audio; "
                             f"{FS // 2} Hz is the most")
        self.host, self.port = host, port
        self.freq_khz = freq_khz
        self.span = span
        self.passband = passband or (0, span)
        self.gain = gain
        self.ident = ident
        self._connect = connect
        self._stream: kiwi.KiwiStream | None = None
        self._pump: threading.Thread | None = None
        self.error: BaseException | None = None
        self._closing = False
        dial = freq_khz * 1e3
        where = f"kiwi {host}:{port} {freq_khz:g} kHz"
        if self.mode == "iq":
            self.window = Window(0.0, float(span), f"{where} iq {span} Hz",
                                 dial - span / 2)
        else:
            self.window = Window.sideband(*map(float, self.passband),
                                          f"{where} {self.mode}", dial, self.mode)

    def _open(self, wideband: bool = False) -> kiwi.KiwiStream:
        half = self.span // 2
        lo, hi = (-half, half) if self.mode == "iq" else self.passband
        try:
            return self._connect(self.host, self.port, self.freq_khz,
                                 mode=self.mode, lo_hz=lo, hi_hz=hi,
                                 gain=self.gain, ident=self.ident, wideband=wideband)
        except (kiwi.KiwiError, OSError) as exc:
            raise AudioError(f"{self.host}:{self.port}: {exc}") from exc

    def _opened(self) -> kiwi.KiwiStream:
        """The ordinary channel if it serves the span, else the wideband one.

        A Kiwi's ordinary IQ rate is 12 kHz or 20.25 kHz depending on how its
        owner configured it, so the rate is asked of the Kiwi, not assumed. It
        is served off the Kiwi's own clock, so a nominal 12 kHz arrives as
        11998.9 Hz, and that serves a 12000 Hz span.
        """
        stream = self._open()
        if self.mode == "iq" and not self._serves(stream.sample_rate):
            stream.close()
            stream = self._open(wideband=True)
            if not self._serves(stream.sample_rate):
                stream.close()
                raise AudioError(f"{self.host}:{self.port} serves {stream.sample_rate:g} Hz "
                                 f"of IQ, short of the {self.span} Hz span asked for")
        return stream

    def _serves(self, rate: float) -> bool:
        return rate >= self.span * (1 - self.CLOCK_SLACK)

    def _filters(self, rate: int) -> list[str]:
        if self.mode != "iq":
            return ["-f", "s16be", "-ar", str(rate), "-ch_layout", "mono",
                    "-i", "pipe:0", "-ar", str(FS)]
        f = self.span / 2
        return ["-f", "s16be", "-ar", str(rate), "-ch_layout", "stereo",
                "-i", "pipe:0",
                "-af", f"aresample={FS},"
                       f"aeval=val(0)*cos(2*PI*{f:g}*t)-val(1)*sin(2*PI*{f:g}*t):c=mono"]

    def _spawn(self) -> subprocess.Popen:
        self._stream = stream = self._opened()
        rate = stream.sample_rate
        # Inherited stderr: an ffmpeg that stops reading is a silent gap, and a
        # monitor cannot report a loss it is not told about.
        conv = subprocess.Popen(
            [ffmpeg(), "-hide_banner", "-loglevel", "warning",
             *self._filters(round(rate)), "-ac", str(CHANNELS),
             "-f", "s16le", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._pump = threading.Thread(target=self._feed, args=(stream, conv),
                                      daemon=True)
        self._pump.start()
        return conv

    def _feed(self, stream: kiwi.KiwiStream, conv: subprocess.Popen) -> None:
        assert conv.stdin is not None
        try:
            for block in stream.blocks():
                conv.stdin.write(block)
        except BrokenPipeError:
            pass
        except BaseException as exc:
            if not self._closing:
                self.error = exc
        finally:
            try:
                conv.stdin.close()
            except OSError:
                pass

    def frames(self) -> Iterator[tuple[int, bytes]]:
        yield from super().frames()
        if self.error is not None:
            raise AudioError(f"{self.host}:{self.port}: {self.error}") from self.error

    def close(self) -> None:
        self._closing = True
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        super().close()
        if self._pump is not None:
            self._pump.join(timeout=2)


def _stored(path: Path) -> Path:
    # `hfmodem.core.wav.stored`, which this stdlib-only package cannot import: a
    # recording named `x.wav` is `x.flac` once the station has stored it.
    for p in (path, path.with_suffix(".wav"), path.with_suffix(".flac")):
        if p.exists():
            return p
    return path


def file_rate(path: str | Path) -> int:
    """A recording's sample rate, PCM or float alike, off its RIFF header, or
    off its STREAMINFO once it is stored as FLAC."""
    path = _stored(Path(path))
    with open(path, "rb") as f:
        riff = f.read(12)
        if riff[:4] == b"fLaC":
            # STREAMINFO is always the first block; its rate is the 20 bits at byte 10.
            info = riff[8:12] + f.read(14)
            return int.from_bytes(info[10:13], "big") >> 4
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            return _probed_rate(path)
        while chunk := f.read(8):
            tag, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if tag == b"fmt ":
                return struct.unpack("<I", f.read(8)[4:8])[0]
            f.seek(size + (size & 1), 1)
    raise AudioError(f"{path}: a WAVE file with no fmt chunk")


def _probed_rate(path: str | Path) -> int:
    exe = shutil.which("ffprobe")
    if not exe:
        raise AudioError(f"{path} is not a WAVE file and ffprobe is not found to read it")
    out = subprocess.run([exe, "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=sample_rate", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True)
    try:
        return int(out.stdout.split()[0])
    except (IndexError, ValueError):
        raise AudioError(f"{path}: no audio stream ffprobe can read") from None


class WavReplaySource(AudioSource):
    """Replay a recording through the same pipeline, for tests and dry runs.

    The *only* place a WAV appears. A 48 kHz mono 16-bit file streams straight
    from the stdlib ``wave`` reader; anything else is normalised once through
    ffmpeg. A recording stored as FLAC is decoded back to its WAV first, so it
    replays exactly as it did before it was stored. ``realtime`` paces frames to
    the wall clock so a replay behaves like a live capture; tests set it False to
    drain as fast as possible.

    A 48 kHz recording is taken to be a rig's, so its window is the rig's
    filter, ``filter_hz``; 0 to 24 kHz of it is mostly the codec's own floor,
    and a tracker handed that raises tracks across the whole of it. Any other
    rate is a receiver that delivered its own span (a Kiwi's 12 kHz, say), and
    the window is that span. ``window`` overrides both. A recording carries no
    dial, so ``rf_hz`` maps only when ``dial_hz`` is given.
    """

    def __init__(self, path: str | Path, *, realtime: bool = True,
                 speed: float = 1.0, dial_hz: float | None = None,
                 mode: str = "USB", window: Window | None = None,
                 filter_hz: tuple[float, float] = RIG_FILTER_HZ) -> None:
        self.path = Path(path)
        self.realtime = realtime
        self.speed = speed
        self.dial_hz = dial_hz
        self.mode = mode
        self.filter_hz = filter_hz
        self._window = window
        self._proc: subprocess.Popen | None = None

    @property
    def window(self) -> Window:
        if self._window is not None:
            return self._window
        rate = file_rate(self.path)
        lo, hi = self.filter_hz if rate == FS else (0.0, min(FS, rate) / 2)
        return Window.sideband(float(lo), float(hi), f"wav {self.path.name}",
                               self.dial_hz, self.mode)

    def _native(self) -> Iterator[bytes]:
        path = _stored(self.path)
        if path.suffix != ".flac":
            yield from self._read(path)
            return
        exe = shutil.which("flac")
        if not exe:
            raise AudioError(f"{path} is stored as FLAC and flac is not installed "
                             "(macOS: brew install flac; Debian/Ubuntu: apt install flac)")
        # To a file rather than a pipe: flac restores the recorded header only
        # when it writes one, and a replay streamed from disk stays flat in
        # memory however long the recording.
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / path.with_suffix(".wav").name
            run = subprocess.run(
                [exe, "-s", "-d", "--keep-foreign-metadata-if-present", "-o", str(wav),
                 str(path)], stderr=subprocess.PIPE, text=True)
            if run.returncode:
                raise AudioError(f"flac could not decode {path}: {run.stderr.strip()}")
            yield from self._read(wav)

    def _read(self, path: Path) -> Iterator[bytes]:
        # `wave` handles PCM only and RAISES on anything else rather than
        # reporting a header this can compare -- so a float file never reached the
        # transcode path that exists for exactly this case, it crashed the caller.
        # Not a corner: 192 of the 196 recordings in offair/captures are
        # WAVE_FORMAT_IEEE_FLOAT, and every modem in the flock reads audio through
        # here, so all of them were blind to all of them.
        try:
            w = wave.open(str(path), "rb")
        except wave.Error:
            yield from self._transcoded(path)
            return
        with w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) \
                    != (FS, CHANNELS, SAMPLE_WIDTH):
                yield from self._transcoded(path)
                return
            while True:
                raw = w.readframes(FRAME_SAMPLES)
                if not raw:
                    return
                yield raw

    def _transcoded(self, path: Path) -> Iterator[bytes]:
        # Inherited stderr: a transcode that fails here yields zero frames, and
        # with the reason discarded that is indistinguishable from a recording
        # of silence.
        self._proc = subprocess.Popen(
            [ffmpeg(), "-hide_banner", "-loglevel", "warning", "-i", str(path),
             "-ac", str(CHANNELS), "-ar", str(FS), "-f", "s16le", "-"],
            stdout=subprocess.PIPE)
        assert self._proc.stdout is not None
        while True:
            raw = self._proc.stdout.read(FRAME_BYTES)
            if not raw:
                return
            yield raw

    def frames(self) -> Iterator[tuple[int, bytes]]:
        if not _stored(self.path).exists():
            raise AudioError(f"no such recording: {self.path}")
        consumed = 0
        period = FRAME_SAMPLES / FS / (self.speed or 1.0)
        clock = None
        try:
            for raw in self._native():
                # Paced from the first frame rather than from the call, or the
                # time a FLAC decode or an ffmpeg start took is paid back as a
                # burst of frames faster than real time.
                clock = clock or time.monotonic()
                yield consumed, raw
                consumed += len(raw) // SAMPLE_WIDTH
                if self.realtime:
                    clock += period
                    delay = clock - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
        finally:
            self.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
