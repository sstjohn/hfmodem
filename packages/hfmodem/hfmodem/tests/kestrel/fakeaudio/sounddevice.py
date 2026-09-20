# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A stand-in for the PortAudio bindings, so an on-air session can be rehearsed.

`kestrel/tests/kestrel/test_onair_dress_rehearsal.py` puts this directory at the front of a
subprocess's ``PYTHONPATH`` and nothing else does, so the operational path never
sees it: run any tool normally and it gets the real package. Its whole job is to
give the session an audio path whose contents the test chooses.

Two things it deliberately is not. It is not a way to make an attempt succeed: a
connect completes here only if the tool really synthesises a connect-request, a
simulated station really demodulates it, and the tool really recovers that
station's answer through its own segmenter and state machine — the same code that
runs at the radio, over real synthesised VARA audio. And it is not silent about
itself: every read, every write and every stream goes into an event log the test
reads back, so "the session played the identification" is a thing checked against
the transport rather than against the session's own printout.

Configured from ``KESTREL_FAKE_AUDIO`` (JSON), because it has to be handed to a
process the test does not construct:

``bed``       what the receiver hears when nothing is transmitting: ``"noise"``
              (band-limited, reads clear), ``"carrier"`` (noise under a steady
              tone, which is what the occupancy detector's ``tone`` score is for),
              ``"quiet"`` (below the codec floor — a deaf receiver), or
              ``{"wav": path}`` to replay a real off-air recording.
``station``   ``{"call": ..., "bw": ..., "max_tx": n}`` to put a listening kestrel
              on the far end, or absent for a frequency with nobody on it.
              ``max_tx`` caps what that station will answer with, so a gateway that
              sends its connect-response and then goes quiet — the KB9MMT failure —
              is ``1``.
``rec_fails`` / ``input_fails``
              raise :class:`PortAudioError` from the session's read or from the
              connect tool's stream. The station runs a second modem against the same
              codec, so losing the device is a thing that happens rather than a thing
              that cannot.
``log``       path to a JSON-lines event log. Every transmission's samples are
              parked beside it as raw float32 and named in its ``play`` event, so a
              test can ask what the transmitter carried and not only how long it was.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time

import numpy as np

FS = 48000
_CFG = json.loads(os.environ.get("KESTREL_FAKE_AUDIO") or "{}")
_LOG = _CFG.get("log")
_BED_S = 30.0
_GAP_S = 0.4        # bed between our transmission ending and an answer starting
_BLOCK = 4800       # what a device hands a callback at a time

_lock = threading.Lock()


class PortAudioError(Exception):
    pass


def _event(kind: str, **fields) -> None:
    if not _LOG:
        return
    line = json.dumps({"event": kind, "pid": os.getpid(), "t": time.time(), **fields})
    with _lock, open(_LOG, "a") as fh:
        fh.write(line + "\n")


_tx_seq = itertools.count()


def _keep(mono: np.ndarray) -> str | None:
    """Park a transmission on disk and say where. A length and a level cannot tell
    a whole callsign from a clipped one; the samples can."""
    if not _LOG:
        return None
    path = f"{_LOG}.tx-{os.getpid()}-{next(_tx_seq)}.f32"
    mono.astype(np.float32).tofile(path)
    return path


def _dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


# -- what the receiver hears ------------------------------------------------------

def _make_bed() -> np.ndarray:
    from scipy.signal import butter, sosfilt

    spec = _CFG.get("bed", "noise")
    if isinstance(spec, dict):
        from scipy.io import wavfile

        a = np.asarray(wavfile.read(spec["wav"])[1], float)
        a = a[:, 0] if a.ndim > 1 else a
        start = int(spec.get("start", 0.0) * FS)
        end = int(spec["stop"] * FS) if "stop" in spec else len(a)
        a = a[start:end]
        return (a / (np.abs(a).max() or 1.0) * spec.get("gain", 0.5)).astype(np.float32)
    n = int(_BED_S * FS)
    if spec == "quiet":
        return np.zeros(n, np.float32)
    rng = np.random.default_rng(_CFG.get("seed", 7))
    sos = butter(4, [300 / (FS / 2), 2800 / (FS / 2)], btype="band", output="sos")
    y = sosfilt(sos, rng.standard_normal(n))
    y = y / (np.abs(y).max() or 1.0) * 0.15
    if spec == "carrier":
        y = y + 0.1 * np.sin(2 * np.pi * 1500 * np.arange(n) / FS)
    return y.astype(np.float32)


class _Bed:
    """The band, looping, with room for a station's answer to be laid over it."""

    def __init__(self) -> None:
        self._bed = _make_bed()
        self._at = 0
        self._pending = np.zeros(0, np.float32)
        self._lock = threading.Lock()

    def inject(self, samples: np.ndarray) -> None:
        with self._lock:
            gap = max(0, int(_GAP_S * FS) - len(self._pending))
            self._pending = np.concatenate(
                [self._pending, np.zeros(gap, np.float32), np.asarray(samples, np.float32)])

    def take(self, n: int) -> np.ndarray:
        idx = (self._at + np.arange(n)) % len(self._bed)
        self._at = (self._at + n) % len(self._bed)
        out = self._bed[idx].copy()
        with self._lock:
            if len(self._pending):
                k = min(n, len(self._pending))
                out[:k] += self._pending[:k]
                self._pending = self._pending[k:]
        return out


_bed = _Bed()


# -- the far end ------------------------------------------------------------------

class _Station:
    """A listening kestrel standing in for the gateway.

    It is the real :class:`~hfmodem.kestrel.vara.vara_arq.VaraStationHandshake`, so what it
    answers with is a real synthesised VARA burst and it answers only what it was
    really called with. Audio played by the tool is handed to it whole — a station
    on the far end of a channel does its own segmentation, and simulating ours twice
    would only test this file.
    """

    def __init__(self, call: str, bw: str, max_tx: int) -> None:
        from hfmodem.kestrel.vara.vara_arq import VaraStationHandshake

        self.sent = 0
        self.max_tx = max_tx
        self.outbox: list[np.ndarray] = []
        self.hs = VaraStationHandshake([call.upper()], self, bw=bw)
        self.hs.listen(True)

    # the VaraIO the handshake transmits through
    def key(self, on: bool) -> None: pass
    def pending(self) -> None: _event("station_pending")
    def connected(self, caller, called, bw) -> None: _event("station_connected",
                                                            caller=caller, called=called)

    def log(self, msg: str) -> None:
        if "not an MFSK handshake burst" not in msg:
            _event("station_log", msg=msg)

    def tx(self, samples) -> None:
        s = np.asarray(samples, float)
        self.outbox.append((s / (np.abs(s).max() or 1.0) * 0.5).astype(np.float32))

    def heard(self, mono: np.ndarray) -> None:
        try:
            self.hs.on_rx_audio(np.asarray(mono, float))
        except Exception as e:                       # noqa: BLE001 - a peer's problem
            _event("station_error", error=repr(e))

    def answer(self) -> None:
        """One burst per over, which is what half duplex allows it.

        The responder role renders its connected-ack as soon as it has answered the
        connect-request (its link-setup receive is a stub), so without this the ack
        would be on the channel while the caller was still transmitting the
        link-setup it is an answer to — and swallowed by the caller's own receive
        mute, which is not a channel any radio presents.
        """
        if not self.outbox:
            return
        burst = self.outbox.pop(0)
        if self.sent >= self.max_tx:
            _event("station_withheld", secs=len(burst) / FS)
            return
        self.sent += 1
        _event("station_answers", n=self.sent, secs=len(burst) / FS)
        _bed.inject(burst)


_station: _Station | None = None


def _ensure_station() -> None:
    """Built by the first input stream, so only the process that is listening to the
    channel puts a station on it — the session's own Morse identification goes into
    a transmitter, not into a gateway."""
    global _station
    spec = _CFG.get("station")
    if _station is None and spec:
        _station = _Station(spec["call"], spec.get("bw", "2300"), spec.get("max_tx", 9))
        _event("station_ready", call=spec["call"])


# -- the PortAudio surface --------------------------------------------------------

#: What this device still has to emit when a write returns — the buffer a real card
#: holds and a real ``Pa_StopStream`` waits out. A card with none of it could not
#: tell a transmit path that drains from one that guesses, and the whole point of
#: the on-air rehearsal is that it can.
OUTPUT_LATENCY_S = 0.20


def query_devices(device=None, kind=None):
    return {"name": str(device), "max_output_channels": 2, "max_input_channels": 1,
            "default_samplerate": float(FS),
            "default_low_output_latency": OUTPUT_LATENCY_S,
            "default_high_output_latency": OUTPUT_LATENCY_S}


def rec(frames, samplerate=FS, channels=1, device=None, dtype="float32", **kw):
    if _CFG.get("rec_fails"):
        raise PortAudioError("simulated: input device is gone")
    _event("rec", device=str(device), secs=frames / samplerate)
    return _bed.take(int(frames)).reshape(-1, 1).astype(np.float32)


def wait(*a, **kw):
    return None


def _emitted(mono, device, start: float, handed_s: float) -> None:
    """Log what the transmitter actually carried, and let the far end answer it.

    ``secs`` is what reached the air and ``handed_s`` what the tool handed over;
    they differ exactly when the end of a transmission was thrown away. The
    wall-clock length is real because the tool advances its receive cursor past
    everything recorded while it was transmitting, so a write that returned
    instantly would hand it a channel it never had; and the far end is given the
    over only once the over has finished, because a station that answered into the
    middle of our transmission would be answering on a channel no radio provides.
    """
    _event("play", t=start, device=str(device), secs=len(mono) / FS,
           handed_s=handed_s, dbfs=round(_dbfs(mono), 1), audio=_keep(mono))
    if _station is not None:
        _station.heard(mono)
        _station.answer()


def play(data, samplerate=FS, device=None, blocking=True, **kw):
    """The convenience call, and it throws the end of the transmission away.

    Not an embellishment: sounddevice's callback raises ``CallbackAbort`` when the
    array runs out, which is PortAudio's ``paAbort`` — terminate immediately, do
    not wait for pending buffers. So whatever the device had buffered is discarded
    rather than played, and a transmit path built on this emits everything except
    its last :data:`OUTPUT_LATENCY_S`. A modem is expected to use an explicit
    stream instead; this stays here so that going back to `play` shows up as the
    clipped transmission it is rather than as nothing at all.
    """
    x = np.asarray(data, float)
    mono = x[:, 0] if x.ndim > 1 else x
    handed = len(mono) / samplerate
    start = time.time()
    time.sleep(max(0.0, handed - OUTPUT_LATENCY_S))
    _emitted(mono[:len(mono) - int(OUTPUT_LATENCY_S * FS)], device, start, handed)


class OutputStream:
    """A device that is genuinely behind, so that draining it can be got wrong.

    Emission begins when :meth:`write` does and runs in real time, so when a write
    returns the last :data:`OUTPUT_LATENCY_S` is still on its way out.
    :meth:`stop` is ``Pa_StopStream`` and waits for it; closing without stopping is
    ``paAbort`` and loses it. That difference is the whole of what the transmit
    path has to get right, and a card with no latency at all could not tell a path
    that drains from one that guesses.
    """

    def __init__(self, samplerate=FS, device=None, channels=1, dtype="float32",
                 latency=None, **kw):
        if _CFG.get("output_fails"):
            raise PortAudioError("simulated: output device is gone")
        self._device, self._sr = device, samplerate
        self.latency = OUTPUT_LATENCY_S
        self._audio: np.ndarray | None = None
        self._start = 0.0
        _event("stream_open", device=str(device), direction="output")

    def start(self): ...

    def write(self, data):
        x = np.asarray(data, float)
        mono = x[:, 0] if x.ndim > 1 else x
        self._start, self._audio = time.time(), mono
        time.sleep(max(0.0, len(mono) / self._sr - self.latency))
        return False

    def _emit(self, keep: int) -> None:
        mono, self._audio = self._audio, None
        if mono is not None:
            _emitted(mono[:keep], self._device, self._start, len(mono) / self._sr)

    def stop(self):
        """Wait out what the device still holds — everything reaches the air."""
        if self._audio is not None:
            time.sleep(self.latency)
            self._emit(len(self._audio))

    def close(self):
        self._emit(len(self._audio) - int(self.latency * FS) if self._audio
                   is not None else 0)          # unstopped: this is the abort
        _event("stream_close", direction="output")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop()
        self.close()


class _Times:
    """PortAudio's `CallbackTime`. The converter's own instant advances by
    exactly one block, forever, which is the property the capture clock is
    measured against -- a fake that hands over `None` cannot be read for it."""

    def __init__(self, adc: float):
        self.currentTime = adc
        self.inputBufferAdcTime = adc
        self.outputBufferDacTime = adc


class _Flags:
    """PortAudio's `CallbackFlags`, all clear. Falsy as a whole, like the real
    one, so a caller testing `if status:` sees a clean callback."""

    input_overflow = input_underflow = output_underflow = False
    priming_output = False

    def __bool__(self) -> bool:
        return False


class InputStream:
    """Feeds the callback band noise, plus whatever the far end answered with."""

    def __init__(self, device=None, channels=1, samplerate=FS, dtype="float32",
                 callback=None, blocksize=None, **kw):
        if _CFG.get("input_fails"):
            raise PortAudioError("simulated: input device is gone")
        self._cb = callback
        self._n = int(blocksize or _BLOCK)
        self._run = threading.Event()
        self._thread: threading.Thread | None = None
        _ensure_station()
        _event("stream_open", device=str(device))

    def start(self):
        if self._thread is None:
            self._run.set()
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()

    def _pump(self):
        period = self._n / FS
        due = time.monotonic()
        n = 0
        while self._run.is_set():
            due += period
            block = _bed.take(self._n).reshape(-1, 1)
            if self._cb is not None:
                self._cb(block, self._n, _Times(n / FS), _Flags())
            n += self._n
            time.sleep(max(0.0, due - time.monotonic()))

    def stop(self):
        self._run.clear()

    def close(self):
        self._run.clear()
        _event("stream_close")
