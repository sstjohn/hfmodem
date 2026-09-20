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
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Iterator

FS = 48000
CHANNELS = 1
SAMPLE_WIDTH = 2                       # s16le
FRAME_SAMPLES = 4800                   # ~0.1 s, matches the modems' live read
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH


class AudioError(RuntimeError):
    pass


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError("ffmpeg not found (macOS: brew install ffmpeg; "
                         "Debian/Ubuntu: apt install ffmpeg)")
    return exe


class AudioSource:
    """A continuous run of ``(t0_samples, pcm)`` frames, s16le mono @ 48 kHz.

    ``t0_samples`` is the sample index of the frame's first sample from the
    start of the stream, so every consumer shares one clock regardless of which
    source produced the audio. Subclasses implement :meth:`frames`; the base
    handles the context-manager and teardown contract.
    """

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

    def __init__(self, cmd: list[str]) -> None:
        super().__init__()
        self.cmd = list(cmd)

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(self.cmd, stdout=subprocess.PIPE)


class KiwiSource(_ProcSource):
    """Live audio from a remote KiwiSDR, no local rig.

    kiwirecorder streams a WAV to stdout in netcat mode (``--nc --nc-wav``) at
    the Kiwi's native 12 kHz; ffmpeg reads that pipe and resamples to 48 kHz
    s16le mono. No files touch disk.

    ``--ncomp`` is not optional here. The Kiwi's default audio stream is
    IMA-ADPCM, which is lossy, and every waveform this feeds is read off symbol
    amplitude and phase.

    Two defaults are chosen rather than assumed. The recorder is kiwiclient's
    ``kiwirecorder.py``, looked up on ``PATH`` — not at a baked-in path, which
    is one machine's — and a missing one is an error naming how to supply it
    rather than a spawn failure. The interpreter defaults to the running one
    because kiwirecorder imports numpy, and a bare ``python3`` is whichever
    interpreter the PATH happens to offer.
    """

    def __init__(self, host: str, freq_khz: float, *, port: int = 8073,
                 mode: str = "usb", lp: int = 0, hp: int = 3000,
                 kiwirecorder: str | None = None,
                 python: str | None = None) -> None:
        super().__init__()
        self.host = host
        self.freq_khz = freq_khz
        self.port = port
        self.mode = mode
        self.lp = lp
        self.hp = hp
        self.kiwirecorder = kiwirecorder or shutil.which("kiwirecorder.py")
        self.python = python or sys.executable

    def _spawn(self) -> subprocess.Popen:
        if not self.kiwirecorder or not Path(self.kiwirecorder).exists():
            raise AudioError(
                "kiwirecorder.py not found "
                + (f"at {self.kiwirecorder}" if self.kiwirecorder
                   else "on PATH")
                + " — install kiwiclient (github.com/jks-prv/kiwiclient) and "
                  "put kiwirecorder.py on PATH, or pass kiwirecorder= with its "
                  "path")
        # Both stderrs are inherited: a Kiwi that drops the connection mid-pass
        # and an ffmpeg that stops reading are the same class of silent gap as
        # the local capture's, and a monitor cannot report a loss it is not told
        # about.
        kiwi = subprocess.Popen(
            [self.python, self.kiwirecorder, "-s", self.host,
             "-p", str(self.port), "-f", f"{self.freq_khz:g}", "-m", self.mode,
             "-L", str(self.lp), "-H", str(self.hp),
             "--ncomp", "--nc", "--nc-wav", "-q"],
            stdout=subprocess.PIPE)
        conv = subprocess.Popen(
            [ffmpeg(), "-hide_banner", "-loglevel", "warning",
             "-f", "wav", "-i", "pipe:0", "-ac", str(CHANNELS), "-ar", str(FS),
             "-f", "s16le", "-"],
            stdin=kiwi.stdout, stdout=subprocess.PIPE)
        if kiwi.stdout:
            kiwi.stdout.close()          # let ffmpeg own the read end for SIGPIPE
        self._kiwi = kiwi
        return conv

    def close(self) -> None:
        super().close()
        kiwi = getattr(self, "_kiwi", None)
        if kiwi and kiwi.poll() is None:
            kiwi.terminate()
            try:
                kiwi.wait(timeout=2)
            except subprocess.TimeoutExpired:
                kiwi.kill()


class WavReplaySource(AudioSource):
    """Replay a recording through the same pipeline, for tests and dry runs.

    The *only* place a WAV appears. A 48 kHz mono 16-bit file streams straight
    from the stdlib ``wave`` reader; anything else is normalised once through
    ffmpeg. ``realtime`` paces frames to the wall clock so a replay behaves like
    a live capture; tests set it False to drain as fast as possible.
    """

    def __init__(self, path: str | Path, *, realtime: bool = True,
                 speed: float = 1.0) -> None:
        self.path = Path(path)
        self.realtime = realtime
        self.speed = speed
        self._proc: subprocess.Popen | None = None

    def _native(self) -> Iterator[bytes]:
        # `wave` handles PCM only and RAISES on anything else rather than
        # reporting a header this can compare -- so a float file never reached the
        # transcode path that exists for exactly this case, it crashed the caller.
        # Not a corner: 192 of the 196 recordings in offair/captures are
        # WAVE_FORMAT_IEEE_FLOAT, and every modem in the flock reads audio through
        # here, so all of them were blind to all of them.
        try:
            w = wave.open(str(self.path), "rb")
        except wave.Error:
            yield from self._transcoded()
            return
        with w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) \
                    != (FS, CHANNELS, SAMPLE_WIDTH):
                yield from self._transcoded()
                return
            while True:
                raw = w.readframes(FRAME_SAMPLES)
                if not raw:
                    return
                yield raw

    def _transcoded(self) -> Iterator[bytes]:
        # Inherited stderr: a transcode that fails here yields zero frames, and
        # with the reason discarded that is indistinguishable from a recording
        # of silence.
        self._proc = subprocess.Popen(
            [ffmpeg(), "-hide_banner", "-loglevel", "warning", "-i", str(self.path),
             "-ac", str(CHANNELS), "-ar", str(FS), "-f", "s16le", "-"],
            stdout=subprocess.PIPE)
        assert self._proc.stdout is not None
        while True:
            raw = self._proc.stdout.read(FRAME_BYTES)
            if not raw:
                return
            yield raw

    def frames(self) -> Iterator[tuple[int, bytes]]:
        if not self.path.exists():
            raise AudioError(f"no such recording: {self.path}")
        consumed = 0
        period = FRAME_SAMPLES / FS / (self.speed or 1.0)
        clock = time.monotonic()
        try:
            for raw in self._native():
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
