# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import atexit
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from hfmodem.core import coexist


class PortAudioError(RuntimeError):
    pass


class CallbackStop(Exception):
    pass


class CallbackAbort(Exception):
    pass


class CallbackFlags:
    input_underflow = input_overflow = output_underflow = output_overflow = False
    priming_output = False

    def __bool__(self):
        return any((self.input_underflow, self.input_overflow,
                    self.output_underflow, self.output_overflow, self.priming_output))


default = SimpleNamespace(device=(0, 0), samplerate=48000, channels=(1, 1),
                          dtype=('float32', 'float32'), latency=('low', 'low'))
_streams = set()
_last = None


def _require_harness():
    if os.environ.get('HFMODEM_COEXIST_HARNESS') != '1':
        raise PortAudioError('coexist sounddevice requires HFMODEM_COEXIST_HARNESS=1')


def device_path():
    _require_harness()
    root = Path(os.environ['HFMODEM_STATE_DIR']).resolve()
    path = Path(os.environ['HFMODEM_COEXIST_AUDIO']).resolve()
    if root not in path.parents or not path.is_file():
        raise PortAudioError('fake audio must be an existing file under HFMODEM_STATE_DIR')
    return path


def query_devices(device=None, kind=None):
    _require_harness()
    info = dict(name='Coexist fake audio', hostapi=0, max_input_channels=2,
                max_output_channels=2, default_samplerate=48000,
                default_low_input_latency=.01, default_low_output_latency=.01,
                default_high_input_latency=.1, default_high_output_latency=.1)
    return [info] if device is None and kind is None else info


def query_hostapis(index=None):
    info = dict(name='Coexist (cooperative)', devices=[0],
                default_input_device=0, default_output_device=0)
    _require_harness()
    return [info] if index is None else info


def check_input_settings(device=None, channels=None, dtype=None, samplerate=None, **kwargs):
    _require_harness()
    if channels is not None and channels not in (1, 2):
        raise PortAudioError('fake device supports one or two channels')
    if samplerate is not None and samplerate <= 0:
        raise PortAudioError('invalid sample rate')
    if dtype is not None:
        np.dtype(dtype)


check_output_settings = check_input_settings


class Stream:
    _input = _output = True
    _raw = False

    def __init__(self, samplerate=None, blocksize=0, device=None, channels=None,
                 dtype=None, latency=None, callback=None, finished_callback=None, **kwargs):
        _require_harness()
        self.samplerate = float(samplerate or default.samplerate)
        self.blocksize = int(blocksize)
        self.device = default.device if device is None else device
        channels = channels or 1
        self.channels = channels
        self._channels = channels if isinstance(channels, (tuple, list)) else (channels, channels)
        dtype = dtype or 'float32'
        self.dtype = dtype
        self._dtype = dtype if isinstance(dtype, (tuple, list)) else (dtype, dtype)
        for ch, dt in zip(self._channels, self._dtype):
            check_input_settings(channels=ch, dtype=dt, samplerate=self.samplerate)
        self.latency = (.01, .01) if self._input and self._output else .01
        self.callback = callback
        self.finished_callback = finished_callback
        self.active = False
        self.closed = False
        self.error = None
        self._thread = None
        self._stop = threading.Event()
        self._done = threading.Event()
        self._fds = []
        try:
            for direction, enabled in (('capture', self._input), ('playback', self._output)):
                if enabled:
                    coexist.checkpoint('audio_before_open')
                    path = device_path()
                    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
                    self._fds.append(fd)
                    info = os.fstat(fd)
                    coexist.event('device_open', path=str(path), fd=fd, dev=info.st_dev,
                                  ino=info.st_ino, direction=direction, cooperative=True)
            _streams.add(self)
        except BaseException:
            self.close()
            raise

    @property
    def time(self):
        return time.monotonic()

    @property
    def stopped(self):
        return not self.active

    @property
    def cpu_load(self):
        return 0.0

    @property
    def read_available(self):
        return self.blocksize or 256

    @property
    def write_available(self):
        return self.blocksize or 256

    def _data(self, frames, index):
        data = np.zeros((frames, self._channels[index]), dtype=self._dtype[index])
        return bytearray(data.tobytes()) if self._raw else data

    def start(self):
        if self.closed:
            raise PortAudioError('stream is closed')
        if self.active:
            return self
        self._stop.clear()
        self._done.clear()
        self.active = True
        if self.callback:
            self._thread = threading.Thread(target=self._run, name='coexist-audio', daemon=True)
            self._thread.start()
        return self

    def _run(self):
        frames = self.blocksize or 256
        try:
            while not self._stop.is_set():
                now = self.time
                stamps = SimpleNamespace(inputBufferAdcTime=now - .01,
                                         currentTime=now, outputBufferDacTime=now + .01)
                buffers = []
                if self._input:
                    buffers.append(self._data(frames, 0))
                if self._output:
                    buffers.append(self._data(frames, 1))
                self.callback(*buffers, frames, stamps, CallbackFlags())
                self._stop.wait(frames / self.samplerate)
        except (CallbackStop, CallbackAbort):
            pass
        except BaseException as exc:
            self.error = exc
            coexist.event('audio_callback_failed', error=repr(exc))
        finally:
            self.active = False
            try:
                if self.finished_callback:
                    self.finished_callback()
            finally:
                self._done.set()

    def read(self, frames):
        if not self.active or not self._input:
            raise PortAudioError('input stream is not active')
        self._stop.wait(frames / self.samplerate)
        return self._data(frames, 0), False

    def write(self, data):
        if not self.active or not self._output:
            raise PortAudioError('output stream is not active')
        frames = len(data)
        if self._raw:
            frames //= self._channels[1] * np.dtype(self._dtype[1]).itemsize
        self._stop.wait(frames / self.samplerate)
        return False

    def stop(self, ignore_errors=True):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(2)
            if self._thread.is_alive():
                raise PortAudioError('fake audio callback did not stop')
        self.active = False
        self._done.set()

    abort = stop

    def close(self, ignore_errors=True):
        if self.closed:
            return
        self.stop(ignore_errors)
        for fd in self._fds:
            os.close(fd)
            coexist.event('device_close', fd=fd)
        self._fds.clear()
        self.closed = True
        _streams.discard(self)

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


class InputStream(Stream):
    _output = False


class OutputStream(Stream):
    _input = False


class RawInputStream(InputStream):
    _raw = True


class RawOutputStream(OutputStream):
    _raw = True


def rec(frames, samplerate=None, channels=None, dtype=None, blocking=False, **kwargs):
    global _last
    stop()
    data = np.zeros((frames, channels or 1), dtype=dtype or 'float32')
    remaining = frames

    def callback(indata, count, stamps, status):
        nonlocal remaining
        remaining -= count
        if remaining <= 0:
            raise CallbackStop

    _last = InputStream(samplerate=samplerate, channels=channels, dtype=dtype,
                        callback=callback, **kwargs)
    _last.start()
    if blocking:
        wait()
    return data


def play(data, samplerate=None, blocking=False, loop=False, **kwargs):
    global _last
    stop()
    data = np.asarray(data)
    remaining = len(data)

    def callback(outdata, count, stamps, status):
        nonlocal remaining
        outdata.fill(0)
        remaining -= count
        if remaining <= 0 and not loop:
            raise CallbackStop

    _last = OutputStream(samplerate=samplerate, channels=1 if data.ndim == 1 else data.shape[1],
                         callback=callback, **kwargs)
    _last.start()
    if blocking:
        wait()


def wait(ignore_errors=True):
    if _last is not None:
        try:
            _last._done.wait()
        finally:
            _last.close(ignore_errors)
        if _last.error:
            raise _last.error


def stop(ignore_errors=True):
    global _last
    if _last is not None:
        _last.close(ignore_errors)
        _last = None


def sleep(msec):
    time.sleep(msec / 1000)


def _exit_handler():
    for stream in list(_streams):
        stream.close()


atexit.register(_exit_handler)
