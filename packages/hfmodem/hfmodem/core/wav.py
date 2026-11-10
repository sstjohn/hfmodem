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

Continuous recordings use one hour of native-rate samples per segment,
avoiding multi-GB RIFF files and bounding open writer state to one segment.
The base path names the recording; only its timestamped segments exist on disk.

Everything here works in WAV; at rest a recording is FLAC. `store` turns a closed
WAV into a FLAC that decodes back to the same bytes, header and all, and deletes
the WAV only once that is proven. `source` opens one file in either form;
`segments` resolves a recording and `read` concatenates it.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import functools
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from math import gcd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import wavfile

from . import levels
from .capture_identity import _absolute

log = logging.getLogger(__name__)
SEGMENT_SAMPLES = 3600 * 48000


def _write_json(path: Path, facts: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.partial', delete=False) as out:
            temporary = Path(out.name)
            json.dump(facts, out, indent=2)
            out.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def recording_base(path: str | Path) -> Path:
    from hfhost.recordings import base
    return base(path)


def segments(path: str | Path) -> list[Path]:
    """Stored segments on their original timebase; incomplete interiors raise."""
    from hfhost.recordings import segments as resolve
    return resolve(path, SEGMENT_SAMPLES)


def stored(path: str | Path) -> Path:
    from hfhost.recordings import stored as resolve
    return resolve(path)


@contextmanager
def opened_source(path):
    """One file in WAV form, held open across storage replacement."""
    from hfhost.recordings import open_segment
    from .capture_identity import validate_artifact
    with open_segment(_absolute(path)) as stream:
        validate_artifact(Path(stream.name))
        with decoded(stream) as out:
            yield out


def decoded(stream):
    """An open WAV or FLAC as the WAV it was written as."""
    from hfhost.recordings import wave_source
    return wave_source(stream, decode=lambda encoded, out:
                       _flac('-d', '--keep-foreign-metadata-if-present', '-o', out, encoded))


def source(path: str | Path):
    """What to hand `wave.open` or `wavfile.read` for one file; never concatenates segments.

    A WAV on disk comes back as its open file, so it is read in place and
    survives storage replacing it; a stored one as the original WAV's bytes.
    """
    from hfhost.recordings import open_segment
    from .capture_identity import validate_artifact
    stream = open_segment(_absolute(path))
    validate_artifact(Path(stream.name))
    if Path(stream.name).suffix != '.flac':
        return stream
    with stream, decoded(stream) as out:
        return io.BytesIO(out.read())


def read_pcm(path):
    from hfhost.recordings import require_frames
    with opened_source(path) as stream, wave.open(stream, 'rb') as audio:
        raw = audio.readframes(audio.getnframes())
        require_frames(len(raw) // (audio.getnchannels() * audio.getsampwidth()),
                       audio.getnframes(), stream.name)
        return audio.getparams(), raw


def pcm_frames(path, block):
    from hfhost.recordings import require_frames
    with opened_source(path) as stream, wave.open(stream, 'rb') as audio:
        received = 0
        while raw := audio.readframes(block):
            received += len(raw)
            yield audio.getparams(), raw
        require_frames(received // (audio.getnchannels() * audio.getsampwidth()),
                       audio.getnframes(), stream.name)


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
        if _absolute(path) != _absolute(ownership.audio.path):
            raise ValueError('Recording ownership path differs')
    flac = path.with_suffix(".flac")
    part = path.with_suffix(".flac.part") if ownership is None else None
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
        _flac("-5", "--keep-foreign-metadata", "-o", part, path)
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
            facts['container_sha256'] = _sha256(part)
            facts['container_artifact_id'] = ownership.container.record['artifact_id']
            from .capture_identity import audio_metadata
            prior = audio_metadata(path)
            ownership.sidecar.update({k: v for k, v in prior.items()
                                      if k not in ownership.audio.record and
                                      k not in ('metadata_version', 'previous_sha256')} |
                                     facts | {'wav': flac.name})
            ownership.container.publish_file(part)
        path.unlink()
    except OSError as exc:
        return _kept(facts, str(exc))
    finally:
        if part is not None:
            part.unlink(missing_ok=True)
    return facts | {"wav": flac.name}


def store_later(path: str | Path, log: str | Path) -> None:
    """`store` the closed WAV at `path` once this process exits, from a child that outlives it.

    A six-hour session's store held its exit for most of a minute (10-23), so the
    exit unkeys, reports and hands back the station first, and the store runs on
    in its own session with `log` its only output. The WAV goes only once the
    FLAC decodes back to it, as in `store`.
    """
    atexit.register(store_detached, Path(path), Path(log))


_STORE = ("import json, sys, time; from hfmodem.core.wav import store; "
          "print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sys.argv[1], flush=True); "
          "facts = store(sys.argv[1]); print(json.dumps(facts), flush=True); "
          "sys.exit('kept' in facts)")


def store_detached(path: Path, log: Path) -> None:
    package = str(Path(__file__).resolve().parents[2])
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, (package, os.environ.get("PYTHONPATH"))))}
    with Path(log).open("ab") as out:
        subprocess.Popen([sys.executable, "-c", _STORE, str(path)], stdin=subprocess.DEVNULL,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env=env)


@functools.cache
def _flac_exe() -> str | None:
    return shutil.which("flac")


def _flac(*args) -> None:
    if not (exe := _flac_exe()):
        raise OSError("no `flac` on PATH to store or read FLAC recordings")
    # On POSIX, the absolute path and close_fds=False enable posix_spawn
    # without forking a process holding a live audio stream.
    run = subprocess.run([exe, "-s", "-f", *map(str, args)], close_fds=os.name == "nt",
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if run.returncode:
        raise OSError(f"flac {' '.join(map(str, args))}: {run.stderr.strip()}")


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _kept(facts: dict, why: str) -> dict:
    log.warning("%s stays WAV: %s", facts["wav"], why)
    return facts | {"kept": why}


def analysis_bounds(path):
    """The absolute sample bounds a whole-source analysis reads, known before its PCM is allocated."""
    from hfhost.recordings import intervals
    spans = intervals(path)
    return spans[0][0][0], spans[-1][0][1]


def read(path: str | Path, target_fs: int | None = None, *, start=None, stop=None) -> np.ndarray:
    """First channel of a WAV as float in [-1, 1], resampled to `target_fs` if given.

    Read through scipy rather than the `wave` module, which handles PCM only and
    raises `unknown format: 3` on WAVE_FORMAT_IEEE_FLOAT. That is not an exotic
    corner: 192 of the 196 recordings in the PACTOR off-air captures are float, so
    every entry point that reached audio through `wave` silently saw four files
    where there were 196, and the corpus's longest run of real PACTOR-1 packets
    sat unread the whole time.
    """
    if start is None and stop is None:
        chunks = []
        rate = None
        for segment in segments(path):
            fs, x = _read_one(segment)
            if rate is not None and rate != fs:
                raise ValueError('Recording segments have different sample rates')
            rate = fs
            chunks.append(x)
        x = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if target_fs and rate != target_fs:
            from scipy.signal import resample_poly
            g = gcd(int(rate), int(target_fs))
            x = resample_poly(x, target_fs // g, rate // g)
        return x
    chunks = []
    rate = None
    from hfhost.recordings import intervals
    spans = intervals(path, start=start, stop=stop)
    if not spans:
        raise ValueError('Recording interval unavailable')
    first = spans[0][0][0] if start is None else start
    last = spans[-1][0][1] if stop is None else stop
    if last < first:
        raise ValueError('Invalid recording interval')
    end = first
    for (lo, hi), segment in spans:
        a, b = max(first, lo), min(last, hi)
        if a >= b:
            continue
        if a != end:
            raise ValueError('Recording gap or missing interval')
        fs, x = _read_one(segment, start=a - lo, stop=b - lo)
        if rate is not None and rate != fs:
            raise ValueError('Recording segments have different sample rates')
        rate = fs
        chunks.append(x)
        end = b
    if end != last:
        raise ValueError('Recording interval unavailable')
    x = np.concatenate(chunks) if len(chunks) > 1 else chunks[0] if chunks else np.empty(0)
    if target_fs and rate != target_fs:
        from scipy.signal import resample_poly
        g = gcd(int(rate), int(target_fs))
        x = resample_poly(x, target_fs // g, rate // g)
    return x


def _read_one(path, *, start=None, stop=None):
    from hfhost.recordings import require_frames, stream_info
    with opened_source(path) as stream:
        is_riff = stream.read(4) == b'RIFF'
        stream.seek(0)
        promised = stream_info(stream)[1] if is_riff else None
        if start is not None or stop is not None:
            raw_stream = _selected_wave(stream, start or 0, stop)
            fs, raw = wavfile.read(raw_stream)
        else:
            fs, raw = wavfile.read(stream, mmap=Path(stream.name).suffix == '.wav')
        if promised is not None and start is None and stop is None:
            require_frames(len(raw), promised, stream.name)
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
    return fs, x


def _selected_wave(stream, start, stop):
    """Select RIFF data bytes before scipy allocates or converts samples."""
    import struct
    header = stream.read(12)
    if header[:4] != b'RIFF' or header[8:] != b'WAVE':
        raise ValueError('Invalid WAV header')
    fmt = None
    while chunk := stream.read(8):
        tag, size = struct.unpack('<4sI', chunk)
        if tag == b'fmt ':
            fmt = stream.read(size)
            if size & 1:
                stream.read(1)
        elif tag == b'data' and fmt is not None:
            align = struct.unpack_from('<H', fmt, 12)[0]
            end = size // align if stop is None else stop
            if not 0 <= start <= end <= size // align:
                raise ValueError('Recording interval unavailable')
            stream.seek(start * align, 1)
            data = stream.read((end - start) * align)
            if len(data) != (end - start) * align:
                raise ValueError('Recording ended early')
            body = (b'WAVEfmt ' + struct.pack('<I', len(fmt)) + fmt + b'\0' * (len(fmt) & 1)
                    + b'data' + struct.pack('<I', len(data)) + data + b'\0' * (len(data) & 1))
            return io.BytesIO(b'RIFF' + struct.pack('<I', len(body)) + body)
        else:
            stream.seek(size + (size & 1), 1)
    raise ValueError('Incomplete WAV header')


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
        _write_json(path.with_suffix('.json'), side)
    else:
        if _absolute(path) != _absolute(ownership.audio.path):
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


class SegmentedWriter:
    """Mono int16 PCM on a session timebase; `closed` owns segment disposition."""

    def __init__(self, path, *, rate, closed, ownership=None, defer_finalize=False,
                 publication=None, started=None, report=None, sample_clock=False, sample_stem=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.segment_samples = max(1, SEGMENT_SAMPLES * rate // 48000)
        self._defer_finalize = defer_finalize
        self._publication = publication
        self._report = report
        self._staged = {}
        if rate <= 0 or self.segment_samples <= 0:
            raise ValueError('Recording rate and segment capacity must be positive')
        self.started = int(ownership.started if ownership is not None else
                           time.time() if started is None else started)
        self.samples = 0
        self._occupied = 0
        self._sample_index = 0
        self.sample_clock = sample_clock
        self.sample_stem = sample_stem or self.path.stem
        self.position = None
        self.error = None
        self.paths = []
        self.owners = {}
        self._ownership = ownership
        self._closed_segment = closed
        self._wav = self._handle = None
        self._index = None
        self._finished = False
        from hfhost.recordings import refuse_occupied
        refuse_occupied(self.path)
        self._failed_index = None
        self._tickets = {}
        self._admission_lock = threading.RLock()
        if not sample_clock:
            self._open(0)

    def sample_path(self, first):
        stamp = time.strftime('%Y%m%dT%H%MZ', time.gmtime(self.started))
        return self.path.with_name(f'{self.sample_stem}-{stamp}-s{first:020d}.wav')

    def set_position(self, first):
        if self.position is not None and first != self.position:
            self.seal()
        self.position = first

    def seal(self):
        """Close the current segment while preserving the next source position."""
        try:
            self._finish()
        finally:
            if self.sample_clock and self._occupied:
                self._sample_index += 1
                self._occupied = 0

    def layout(self, count):
        """Planned segments as (index, path, within, count) on the source clock."""
        within = self._occupied
        index = self._sample_index
        position = self.position
        path = self.paths[-1] if self._wav is not None else self.sample_path(position - within)
        while count > 0:
            size = min(count, self.segment_samples - within)
            yield index, path, within, size
            count -= size
            position += size
            index, within, path = index + 1, 0, self.sample_path(position)

    def _open(self, index):
        stamp = time.strftime('%Y%m%dT%H%MZ', time.gmtime(
            self.started + index * self.segment_samples / self.rate))
        path = self.path.with_name(f'{self.path.stem}-{stamp}.wav')
        if self.sample_clock:
            path = self.sample_path(self.position)
        owner = None
        physical = path
        if self._ownership is not None:
            if self._publication is None:
                owner = self._reserve_owner(index)
                self._owner_opened(owner, path)
            else:
                ticket = self._tickets.get(index)
                if index or self.sample_clock:
                    physical = path.with_name('.' + path.stem + '.pending.wav')
                    self._staged[path] = physical
                def prepare():
                    owner = self._reserve_owner(index, ticket)
                    self.owners[path] = owner
                    self._owner_opened(owner, path)
                self._publication(prepare)
        handle = physical.open('xb') if owner is None else owner.audio.open()
        self._segment_failed = False
        self._written = 0
        self._occupied = 0
        self.paths.append(path)
        self.owners.setdefault(path, owner)
        self._handle = handle
        self._wav = wave.open(handle, 'wb')
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.rate)
        self._wav.writeframes(b'')
        handle.flush()
        with self._admission_lock:
            self._index = index
            self._tickets.pop(index, None)
        self._publish_segments()

    def _reserve_owner(self, index, ticket=None):
        if index == 0 and not self.sample_clock:
            from .capture_identity import Recording
            self._ownership.audio.check()
            return Recording(self._ownership.audio, self._ownership.container,
                             self._ownership.sidecar.segment(self._ownership.audio))
        return (ticket or self._tickets[index]).reserve()

    def _owner_opened(self, owner, path):
        owner.sidecar.update({'wav': path.name, 'samplerate': self.rate,
                              'channels': 1, 'samples': 0, 'capture_state': 'ongoing',
                              'recording_artifact_id': self._ownership.audio.record['artifact_id']})

    def _publish_segments(self):
        facts = {
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.started)),
            'samplerate': self.rate, 'segment_samples': self.segment_samples,
            'samples': self.samples}
        paths = tuple(self.paths)
        def publish():
            segments = []
            for path in paths:
                try:
                    path = stored(path)
                except Exception:
                    pass
                segments.append(path.name)
            facts['segments'] = segments
            if self._ownership is None:
                _write_json(self.path.with_suffix('.json'), facts)
            else:
                self._ownership.sidecar.update(facts)
        if self._publication is None:
            publish()
        else:
            self._publication(publish)

    @property
    def stored_paths(self):
        paths = []
        for path in self.paths:
            try:
                path = stored(path)
            except Exception:
                pass
            paths.append(path)
        return paths

    def _finish(self):
        if self._wav is None:
            return
        path = self.paths[-1]
        writer, self._wav = self._wav, None
        error = None
        try:
            writer.close()
        except Exception as exc:
            error = exc
        finally:
            try:
                self._handle.close()
            except Exception as exc:
                error = error or exc
            self._handle = None
        samples, failed = self._written, bool(error or self._segment_failed)
        def publish():
            if self._ownership is None:
                return
            owner = self.owners[path]
            if owner is None:
                raise ValueError('Recording segment ownership is unavailable')
            if path in self._staged:
                owner.audio.publish_file(self._staged[path])
            owner.sidecar.update({'wav': path.name, 'samplerate': self.rate,
                                  'channels': 1, 'samples': samples,
                                  'capture_state': ('partial' if failed else
                                                    'storing' if self._defer_finalize else 'finalized'),
                                  'recording_artifact_id': self._ownership.audio.record['artifact_id'],
                                  **({} if self._defer_finalize else {'sha256': _sha256(path)})})
        try:
            if self._publication is None:
                publish()
            else:
                self._publication(publish)
        except Exception as exc:
            error = error or exc
        if error is not None:
            self._report_failure(error, self._index)
        try:
            self._closed_segment(path)
            self._publish_segments()
        except Exception as exc:
            error = error or exc
        if error is not None:
            self._report_failure(error, self._index)
            raise error

    def _report_failure(self, error, index):
        report = self._failed_index != index or self.error is None
        self._failed_index = index
        self.error = self.error or error
        self._segment_failed = True
        if report:
            try:
                message = f'Recording {self.path}: {type(error).__name__}: {error}'
                if self._report is None:
                    print(message, file=sys.stderr, flush=True)
                else:
                    self._report(message)
            except Exception:
                pass

    def write_converted(self, frame, convert):
        before = self.samples
        try:
            self.writeframes(convert(frame))
        except Exception as error:
            remaining = len(frame) - (self.samples - before)
            if remaining > 0:
                self.abandon(remaining, error)
            raise

    def abandon(self, samples, error):
        """A frame that could not become PCM still occupied session time."""
        while samples:
            index, within = divmod(self.samples, self.segment_samples)
            count = min(samples, self.segment_samples - within)
            self._report_failure(error, index)
            self.samples += count
            samples -= count
            if self.samples % self.segment_samples == 0:
                try:
                    self._finish()
                except Exception:
                    pass

    def writeframes(self, pcm):
        if self._finished:
            raise ValueError('Segmented writer is closed')
        pcm = memoryview(pcm)
        if len(pcm) % 2:
            raise ValueError('Mono int16 PCM requires whole samples')
        error = None
        offset = 0
        while offset < len(pcm):
            if self.sample_clock:
                index, _, within, _ = next(self.layout((len(pcm) - offset) // 2))
            else:
                index, within = divmod(self.samples, self.segment_samples)
            count = min((len(pcm) - offset) // 2, self.segment_samples - within)
            try:
                if self._failed_index != index:
                    if self._wav is None or self._index != index:
                        self._finish()
                        self._open(index)
                    self._wav.writeframes(pcm[offset:offset + count * 2])
                    self._written += count
            except Exception as exc:
                error = error or exc
                self._segment_failed = True
                self._report_failure(exc, index)
            finally:
                # Lost bytes still occupied session time, including when a disk
                # failure occurs halfway through a callback crossing a boundary.
                self.samples += count
                if self.sample_clock:
                    self.position += count
                    self._occupied += count
                offset += count * 2
            if within + count == self.segment_samples:
                try:
                    self.seal()
                except Exception as exc:
                    error = error or exc
        if error is not None:
            self.error = self.error or error
            raise error

    def queue_through(self, samples):
        with self._admission_lock:
            self._queue_through(samples)

    def _queue_through(self, samples):
        if self._ownership is None:
            return
        from .capture_identity import QueuedSegment
        if self.sample_clock:
            for index, path, _, _ in self.layout(samples - self.samples):
                if index not in self._tickets and path not in self.owners:
                    self._tickets[index] = QueuedSegment.queue(self._ownership, path)
            return
        for index in range(1, (samples - 1) // self.segment_samples + 1):
            if index in self._tickets or index <= (self._index or 0):
                continue
            stamp = time.strftime('%Y%m%dT%H%MZ', time.gmtime(
                self.started + index * self.segment_samples / self.rate))
            path = self.path.with_name(f'{self.path.stem}-{stamp}.wav')
            self._tickets[index] = QueuedSegment.queue(self._ownership, path)

    def flush(self):
        if self._handle is not None:
            self._handle.flush()

    def close(self):
        if not self._finished:
            self._finished = True
            self._finish()


class CaptureRecorder:
    """Everything the sound card delivered, unbroken, to segmented WAVs.

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
    owner with a pump thread stops it before `close`; close waits for any drain
    already in progress. A file write inside a PortAudio callback is a capture
    dropout, and a dropout in the receive stream is a frame nobody can prove was missed.

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

    A process killed outright leaves completed segments and the current WAV
    valid to its last `drain`, because
    `wave` patches the header behind every `writeframes` and seeking there
    flushes what is under it. What is missing then is the final sidecar, and its
    absent measurements (or an owned sidecar still marked ongoing) are the signal: a file
    without final measurements is a file nobody can grade, which
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
            if ownership.base is None:
                raise ValueError('Continuous recording requires real segment reservations')
            ownership.audio.writer.update_status(state='ongoing', pending=1, component=self._component)
        if ownership is not None:
            if _absolute(self.path) != _absolute(ownership.base or ownership.audio.path):
                raise ValueError('Recording ownership path differs')
            if self.path.exists() or self.path.is_symlink():
                raise FileExistsError('Owned continuous recording already exists')
        self._stores = ThreadPoolExecutor(max_workers=1, thread_name_prefix='capture-store')
        self._pending = []
        self._store_results = {}
        self._drain_lock = threading.RLock()
        self._closing = False
        self._closing_failed = False
        self._wav = SegmentedWriter(self.path, rate=rate, closed=self._store_segment,
                                    ownership=ownership)
        self._out = self._wav
        if ownership is not None:
            ownership.sidecar.update({'wav': self._out.paths[0].name, 'samplerate': rate,
                                      'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                                   time.gmtime(self._out.started)),
                                      'channels': 1, 'samples': 0,
                                      'segment_samples': self._out.segment_samples,
                                      'segments': [self._out.paths[0].name],
                                      'capture_state': 'ongoing'})
        self._q: list[np.ndarray] = []
        self._lock = self._out._admission_lock
        self._push_error = None
        self._t0 = self._out.started
        self._queued_samples = 0
        self._n = self._railed = 0
        self._peak = self._sumsq = 0.0

    @property
    def seconds(self) -> float:
        return self._n / self._rate

    def _store_segment(self, path):
        if not self._closing_failed:
            self._pending.append(self._stores.submit(self._store_one, path))

    def _store_one(self, path):
        owner = self._out.owners[path]
        facts = store(path, ownership=owner)
        if owner is not None and 'kept' in facts:
            from .capture_identity import audio_metadata
            prior = audio_metadata(path)
            owner.sidecar.update({k: v for k, v in prior.items()
                                  if k not in owner.audio.record and
                                  k not in ('metadata_version', 'previous_sha256')} | facts)
        self._store_results[path] = facts

    def push(self, block: np.ndarray) -> None:
        """The sound-card callback. Queue only."""
        try:
            with self._lock:
                if self._closing:
                    return
                try:
                    self._out.queue_through(self._queued_samples + block.size)
                except Exception as error:
                    self._push_error = self._push_error or error
                self._q.append(block)
                self._queued_samples += block.size
        except Exception as error:
            self._push_error = self._push_error or error

    def drain(self) -> None:
        with self._drain_lock:
            self._drain()

    def _drain(self):
        with self._lock:
            blocks, self._q = self._q, []
        if self._wav is None:                    # a pump thread that outlived `close`
            return
        error = self._push_error
        for b in blocks:
            self._n += b.size
            try:
                self._wav.write_converted(b, self._convert)
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def _convert(self, block):
        block = np.asarray(block, dtype=np.float32)
        a = np.abs(block.astype(np.float64))
        self._peak = max(self._peak, float(a.max(initial=0.0)))
        self._sumsq += float(a @ a)
        self._railed += int(np.count_nonzero(a >= levels.RAIL))
        return np.clip(np.round(block * 32768.0), -32768, 32767).astype('<i2').tobytes()

    def close(self) -> None:
        with self._drain_lock:
            with self._lock:
                self._closing = True
            self._close_locked()

    def _close_locked(self):
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
                        state='partial', written=sum(stored(p).is_file() for p in self._out.paths), partial=1,
                        component=self._component)
                except BaseException as trouble:
                    # Preserve the original recording failure; durable ongoing
                    # metadata remains conservative if the journal itself failed.
                    writer._status['journal_error'] = type(trouble).__name__
            raise
        else:
            if self._ownership is not None:
                self._ownership.audio.writer.update_status(state='finalized', written=len(self._out.paths), component=self._component)


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
        error = None
        try:
            self.drain()
            if self._out.error is not None:
                raise self._out.error
            self._write_sidecar(False)
        except BaseException as exc:
            error = exc
        self._closing_failed = error is not None and self._out.error is None
        if error is not None:
            self._out._segment_failed = True
        try:
            self._wav.close()
        except BaseException as exc:
            error = error or exc
        finally:
            self._wav = None
        self._stores.shutdown(wait=True)
        for pending in self._pending:
            try:
                pending.result()
            except BaseException as exc:
                error = error or exc
        try:
            self._write_sidecar(error is None)
        except BaseException as exc:
            error = error or exc
        if error is not None:
            raise error

    def _write_sidecar(self, completed):
        facts = self._sidecar()
        if len(self._out.paths) == 1 and self._ownership is None:
            facts |= self._store_results.get(self._out.paths[0], {})
        if self._ownership is None:
            _write_json(self.path.with_suffix('.json'), facts)
        else:
            self._ownership.sidecar.update(facts | {
                'capture_state': 'finalized' if completed else 'partial'})

    def _sidecar(self) -> dict:
        rms = (self._sumsq / self._n) ** 0.5 if self._n else 0.0
        return {
            "wav": stored(self._out.paths[0]).name,
            "segment_samples": self._out.segment_samples,
            "segments": [stored(p).name for p in self._out.paths],
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
