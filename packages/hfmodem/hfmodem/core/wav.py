# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Captures on disk. **The writer never normalises.**

`shrike/session.py` used to scale every file it wrote to 0.8 peak. A capture that
came off the codec with 9.60% of its samples against the rail then measured 0.00%
railed when read back, so the one fault that makes a recording undecodable no
matter what the far end did was invisible in the file. Two readers measured saved
captures and concluded there had been no clipping, and were wrong both times. A
normalising writer does not lose accuracy, it loses *the* measurement — the level
a capture had is a property of the receive chain, and once divided out it is not
recoverable from anything.

So: samples go to disk as they arrive, clipped to int16 full scale and no more,
and a JSON sidecar lands beside every file before the WAV is written. The sidecar
is unconditional because the decision "this one does not need levels" is the
decision that produced the corpus of files nobody can grade. It carries
`normalised_on_write` as a positive statement rather than by omission: the corpus
holds files from both eras and a reader has to be able to tell which it has.

Railed is measured against full scale, never against the capture's own peak, and
`core.levels` owns that threshold. Reading is the same job in reverse: full scale
is 32768, so a sample written at the rail reads back at the rail.

`write` is for a capture already in hand; `CaptureRecorder` is the same
convention for one that arrives a block at a time and outlives the memory to hold
it. Both live here so a recording from any stack in this tree grades the same.

Everything here works in WAV; at rest a recording is FLAC. `store` turns a closed
WAV into a FLAC that decodes back to the same bytes, header and all, and deletes
the WAV only once that is proven. `source` is the other direction, and it is what
every reader opens: a recording named `x.wav` is found as `x.flac` once stored.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from . import levels

log = logging.getLogger(__name__)


def stored(path: str | Path) -> Path:
    """The file on disk holding the recording `path` names, WAV or FLAC.

    A path read out of a log or a sidecar was written while the recording was
    still a WAV, so either spelling finds it; a real `path` wins. A recording in
    neither form comes back as `path`, for the caller's own error to name.
    Only absent candidates are ignored; other filesystem faults propagate.
    """
    path = Path(path)
    for p in (path, path.with_suffix(".wav"), path.with_suffix(".flac")):
        try:
            p.stat()
        except FileNotFoundError as exc:
            if exc.filename != os.fspath(p):
                raise
        else:
            return p
    return path


def source(path: str | Path) -> str | io.BytesIO:
    """What to hand `wave.open` or `wavfile.read` for the recording `path` names.

    A WAV on disk comes back as its path, so a caller can still mmap it; a
    stored one as the original WAV's bytes, decoded in memory.
    """
    p = stored(path)
    if p.suffix in {".wav", ".flac"} and p.exists():
        from .capture_identity import validate_artifact
        validate_artifact(p)
    if p.suffix != ".flac":
        return str(p)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / p.with_suffix(".wav").name
        _flac("-d", "--keep-foreign-metadata-if-present", "-o", out, p)
        return io.BytesIO(out.read_bytes())


def store(path: str | Path, *, ownership=None) -> dict:
    """Keep the closed WAV at `path` as FLAC, byte for byte, and delete the WAV.

    Returns sidecar fields: `wav`, the name of the file now holding the
    recording, and `sha256`, the original WAV's, which the FLAC decodes back
    to. Anything short of a verified round trip leaves the WAV in place and
    says why under `kept`. Never raises: this runs as recorders close, and a
    recording is worth more than its compression.
    """
    path = Path(path)
    if ownership is not None:
        ownership.audio.check()
        if path.absolute() != ownership.audio.path:
            raise ValueError('Recording ownership path differs')
    flac, part = path.with_suffix(".flac"), path.with_suffix(".flac.part")
    try:
        facts = {"wav": path.name, "sha256": _sha256(path)}
    except OSError as exc:
        return _kept({"wav": path.name}, f"unreadable: {exc}")
    try:
        wave.open(str(path)).close()
    except (wave.Error, EOFError) as exc:
        return _kept(facts, f"not integer PCM, the only kind FLAC holds ({exc})")
    try:
        if ownership is not None:
            part, reserved = ownership.container.temporary()
            reserved.close()
        _flac("-8", "--keep-foreign-metadata", "-o", part, path)
        with tempfile.TemporaryDirectory() as tmp:
            back = Path(tmp) / path.name
            _flac("-d", "--keep-foreign-metadata", "-o", back, part)
            if _sha256(back) != facts["sha256"]:
                return _kept(facts, "the FLAC does not decode back to this WAV")
        if ownership is None:
            part.replace(flac)
        else:
            if _sha256(path) != facts['sha256']:
                return _kept(facts, 'the owned WAV changed during compression')
            ownership.container.publish_file(part)
            facts['container_sha256'] = _sha256(flac)
            facts['container_artifact_id'] = ownership.container.record['artifact_id']
        path.unlink()
    except OSError as exc:
        return _kept(facts, str(exc))
    finally:
        if ownership is None:
            part.unlink(missing_ok=True)
    return facts | {"wav": flac.name}


@functools.cache
def _flac_exe() -> str | None:
    return shutil.which("flac")


def _flac(*args) -> None:
    if not (exe := _flac_exe()):
        raise OSError("no `flac` on PATH to store or read FLAC recordings")
    # An absolute path and inherited fds are what let CPython posix_spawn rather
    # than fork a process that may be holding a live audio stream.
    run = subprocess.run([exe, "-s", "-f", *map(str, args)], close_fds=False,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if run.returncode:
        raise OSError(f"flac {' '.join(map(str, args))}: {run.stderr.strip()}")


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _kept(facts: dict, why: str) -> dict:
    log.warning("%s stays WAV: %s", facts["wav"], why)
    return facts | {"kept": why}


def read(path: str | Path, target_fs: int | None = None) -> np.ndarray:
    """First channel of a WAV as float in [-1, 1], resampled to `target_fs` if given.

    Read through scipy rather than the `wave` module, which handles PCM only and
    raises `unknown format: 3` on WAVE_FORMAT_IEEE_FLOAT. That is not an exotic
    corner: 192 of the 196 recordings in the PACTOR off-air captures are float, so
    every entry point that reached audio through `wave` silently saw four files
    where there were 196, and the corpus's longest run of real PACTOR-1 packets
    sat unread the whole time.
    """
    src = source(path)
    fs, raw = wavfile.read(src, mmap=isinstance(src, str))
    a = np.asarray(raw)
    if a.ndim > 1:
        a = a[:, 0]
    if np.issubdtype(a.dtype, np.unsignedinteger):
        # 8-bit WAV is unsigned with the midpoint as silence, so scaling without
        # removing it leaves a ~0.5 DC pedestal. A noncoherent tone detector
        # survives that; anything that squares the samples to estimate an SNR
        # reports the pedestal instead of the signal.
        half = (np.iinfo(a.dtype).max + 1) / 2
        x = (a.astype(np.float64) - half) / half
    elif np.issubdtype(a.dtype, np.integer):
        x = a.astype(np.float64) / -np.iinfo(a.dtype).min
    else:
        x = a.astype(np.float64)
    if target_fs and fs != target_fs:
        from scipy.signal import resample_poly
        g = gcd(int(fs), int(target_fs))
        x = resample_poly(x, target_fs // g, fs // g)
    return x


def write(path: str | Path, audio: np.ndarray, fs: int, *,
          channels: int = 1, ownership=None, **facts) -> dict:
    """Write mono float samples as int16 PCM at `fs`, and the sidecar beside them.

    `fs` is required. A WAV writer with a default rate is how a 12 kHz capture
    ends up labelled 48 kHz, and nothing downstream can tell.

    `facts` join the sidecar as they are given — `xruns`, `source`, whatever the
    caller knows and the samples do not say. The returned dict is what was
    written, for a caller that wants to log it.
    """
    path = Path(path)
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    side = _sidecar(path, x, fs, channels) | facts
    # Before the WAV, so a writer that dies mid-file still leaves the levels: the
    # audio without them is a recording nobody can grade, and the sidecar without
    # the audio at least says what was there.
    if ownership is None:
        path.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
    else:
        if path.absolute() != ownership.audio.path:
            raise ValueError('Recording ownership path differs')
        if path.exists() or path.is_symlink():
            raise FileExistsError('Owned recording already exists')
        ownership.sidecar.update(side | {'capture_state': 'partial'})

    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1).reshape(-1)
    temporary = None
    sink = None
    if ownership is not None:
        temporary, sink = ownership.audio.temporary()
    try:
        with wave.open(str(path) if sink is None else sink, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(fs)
            w.writeframes(pcm.tobytes())
    finally:
        if sink is not None:
            sink.close()
    if ownership is not None:
        ownership.audio.publish_file(temporary)
        return ownership.sidecar.update(side | {'capture_state': 'finalized', 'sha256': _sha256(path)})
    return side


def _sidecar(path: Path, x: np.ndarray, fs: int, channels: int) -> dict:
    peak_dbfs = levels.peak_dbfs(x)
    rms = float(np.sqrt(x @ x / x.size)) if x.size else 0.0
    return {
        "wav": path.name,
        "samplerate": int(fs),
        "channels": int(channels),
        "samples": int(x.size),
        "duration_s": round(x.size / fs, 3),
        "peak": round(float(np.abs(x).max()) if x.size else 0.0, 6),
        # `null` for digital silence, which is -inf dBFS and is a real reading. A
        # floor like -120 would be a plausible-looking number for a measurement
        # that was not made, and JSON cannot carry the honest one.
        "peak_dbfs": round(peak_dbfs, 2) if np.isfinite(peak_dbfs) else None,
        "rms_dbfs": round(20 * np.log10(rms), 2) if rms > 0 else None,
        "railed_pct": round(levels.railed_ppm(x) / 1e4, 4),
        "normalised_on_write": False,
    }


class CaptureRecorder:
    """Everything the sound card delivered, unbroken, to one WAV.

    Unbroken **includes the intervals we were keyed**, which every receive path
    here discards. The audio there is worthless — the receiver is muted — but the
    elapsed time is not, and neither is whatever a peer sent underneath it. A
    recording with the keyed stretches cut out has no timebase: a detection
    logged at a sample index cannot be located in it afterwards, and two receive
    windows either side of a transmission cannot be joined. It is also the
    specific confusion such a recording has to be able to settle. A run of
    undecoded "bursts" at a fixed offset after each of our transmissions looks
    like a station answering on a cadence, and reads instead as our own PTT cycle
    once the transmissions are visible in the same file at their true spacing.

    `push` is the sound-card callback and only queues; the bytes are written by
    `drain`, on a thread of the owner's choosing and one drainer at a time — an
    owner with a pump thread joins it before `close`, or the two interleave in
    the file. A file write inside a PortAudio callback is a capture dropout, and
    a dropout in the receive stream is a frame nobody can prove was missed.

    SAMPLE k OF THE FILE IS SAMPLE k OF WHAT WAS PUSHED, so a recorder armed
    before its stream starts carries the card's own index and every sample index
    the session logs is an offset into it. `facts` is what the owner knows and
    the samples do not — the index it started at, what the stream lost — and it
    joins the sidecar at `close`. Every such count is a one-way reading: non-zero
    says the indices moved off the air, zero says only that nothing reached the
    counter. The driver's xruns miss blocks a starved interpreter never accepted;
    `_LiveInput.lost` catches those and misses anything dropped upstream of the
    converter's own timestamps. `tools/clock_shortfall.py` reads the sessions
    recorded before either count existed.

    `rate` is required, for this module's reason: a writer with a default rate is
    how a 12 kHz capture ends up labelled 48 kHz. 48 kHz mono int16 costs 5.8 MB
    a minute, and the bound on a recording is the session it belongs to — there
    is deliberately no second one, because a cap that stops a recorder
    mid-session truncates exactly the thing it was taken to measure.

    A process killed outright leaves the WAV valid to its last `drain`, because
    `wave` patches the header behind every `writeframes` and seeking there
    flushes what is under it. What is missing then is the sidecar, and that
    absence is the signal: a file without one is a file nobody can grade, which
    is this module's rule, and here it says the recording stopped rather than
    ended.
    """

    def __init__(self, path, *, rate: int, source: str = "", ownership=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.facts: dict = {}
        self._rate = rate
        self._source = source
        self._ownership = ownership
        self._component = 'stream-' + ownership.audio.record['artifact_id'] if ownership else None
        if ownership is not None:
            ownership.audio.writer.update_status(state='ongoing', pending=1, component=self._component)
        self._owned_handle = None
        if ownership is not None:
            if self.path.absolute() != ownership.audio.path:
                raise ValueError('Recording ownership path differs')
            if self.path.exists() or self.path.is_symlink():
                raise FileExistsError('Owned continuous recording already exists')
            ownership.sidecar.update({'wav': self.path.name, 'samplerate': rate,
                                      'channels': 1, 'samples': 0, 'capture_state': 'ongoing'})
            self._owned_handle = ownership.audio.open()
        self._wav = wave.open(str(self.path) if ownership is None else self._owned_handle, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(rate)
        self._q: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._n = self._railed = 0
        self._peak = self._sumsq = 0.0

    @property
    def seconds(self) -> float:
        return self._n / self._rate

    def push(self, block: np.ndarray) -> None:
        """The sound-card callback. Queue only."""
        with self._lock:
            self._q.append(np.asarray(block, dtype=np.float32))

    def drain(self) -> None:
        with self._lock:
            blocks, self._q = self._q, []
        if self._wav is None:                    # a pump thread that outlived `close`
            return
        for b in blocks:
            a = np.abs(b.astype(np.float64))
            self._n += a.size
            self._peak = max(self._peak, float(a.max(initial=0.0)))
            self._sumsq += float(a @ a)
            self._railed += int(np.count_nonzero(a >= levels.RAIL))
            self._wav.writeframes(
                np.clip(np.round(b * 32768.0), -32768, 32767).astype("<i2").tobytes())

    def close(self) -> None:
        if self._wav is None:
            return
        try:
            self._close()
        except BaseException:
            if self._ownership is not None:
                writer = self._ownership.audio.writer
                writer._status['failed'] = True
                try:
                    writer.update_status(
                        state='partial', written=1 if self.path.exists() or
                        self.path.with_suffix('.flac').exists() else 0, partial=1,
                        component=self._component)
                except BaseException as trouble:
                    # Preserve the original recording failure; durable ongoing
                    # metadata remains conservative if the journal itself failed.
                    writer._status['journal_error'] = type(trouble).__name__
            raise
        else:
            if self._ownership is not None:
                self._ownership.audio.writer.update_status(state='finalized', written=1, component=self._component)


    def _close(self) -> None:
        """Flush, close the WAV, and leave the sidecar — even if the flush fails.

        The sidecar is what states how long the file is, so a final write that
        failed without one would leave a short recording nothing can tell is
        short. It goes in the `finally` for that reason, and the failure is
        raised over the top of it for the owner to report.

        A recording that closed cleanly is then stored as FLAC, and the sidecar
        rewritten to name the file it ended up in. The first write stands until
        then, so an owner killed while the FLAC is made still leaves a graded WAV.
        """
        if self._wav is None:                    # teardown can reach here twice
            return
        side = self.path.with_suffix(".json")
        completed = False
        try:
            self.drain()
            completed = True
        finally:
            self._wav.close()
            self._wav = None
            if self._owned_handle is not None:
                self._owned_handle.close()
            if self._ownership is None:
                side.write_text(json.dumps(self._sidecar(), indent=2) + "\n")
            else:
                self._ownership.sidecar.update(self._sidecar() | {
                    'capture_state': 'finalized' if completed else 'partial',
                    'sha256': _sha256(self.path)})
        facts = self._sidecar() | store(self.path, ownership=self._ownership)
        if self._ownership is None:
            side.write_text(json.dumps(facts, indent=2) + "\n")
        else:
            self._ownership.sidecar.update(facts | {'capture_state': 'finalized'})

    def _sidecar(self) -> dict:
        rms = (self._sumsq / self._n) ** 0.5 if self._n else 0.0
        return {
            "wav": self.path.name,
            "source": self._source,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._t0)),
            "samplerate": self._rate,
            "channels": 1,
            "samples": self._n,
            "duration_s": round(self.seconds, 3),
            "peak": round(self._peak, 6),
            # `null` for digital silence: -inf dBFS is the honest reading and JSON
            # cannot carry it, where a floor like -120 would look like a measurement.
            "peak_dbfs": round(20 * np.log10(self._peak), 2) if self._peak > 0 else None,
            "rms_dbfs": round(20 * np.log10(rms), 2) if rms > 0 else None,
            "railed_pct": round(100.0 * self._railed / self._n, 4) if self._n else 0.0,
            "normalised_on_write": False,
        } | self.facts
