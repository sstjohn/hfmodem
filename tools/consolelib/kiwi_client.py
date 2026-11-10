# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Sideband audio adapter for the shipped KiwiSDR stream."""
from types import SimpleNamespace
import time
import socket
import threading
import numpy as np
from hfhost.kiwi import KiwiStream, WebSocket


class Busy(RuntimeError):
    pass


BUSY_ERRORS = (Busy,)
DEADLINES = True


class _SessionSocket:
    """Admit socket operations until stop; let the owner await their return."""

    def __init__(self, session, sock):
        self.session, self.sock = session, sock

    def _call(self, method, *args):
        s = self.session
        with s._io_done:
            if s.stopping:
                raise InterruptedError('witness stopped')
            s._active_io += 1
        try:
            return getattr(self.sock, method)(*args)
        finally:
            with s._io_done:
                s._active_io -= 1
                s._io_done.notify_all()

    def recv(self, size):
        return self._call('recv', size)

    def sendall(self, data):
        return self._call('sendall', data)

    def settimeout(self, seconds):
        return self._call('settimeout', seconds)


class KiwiSession:
    def __init__(self, host, port, dial_hz, *, low_cut=300, high_cut=2900, agc_gain=50, passband=None, mode='usb', deadline=None):
        self.host, self.port, self.dial_hz = host, port, dial_hz
        if passband is not None:
            low_cut, high_cut = passband
        if mode not in ('usb', 'lsb'):
            raise ValueError('Kiwi sideband must be usb or lsb')
        if mode == 'lsb':
            low_cut, high_cut = -high_cut, -low_cut
        self.options = dict(mode=mode, lo_hz=low_cut, hi_hz=high_cut, gain=agc_gain)
        self.stream = None
        self.deadline = deadline
        self.ws = None
        self.stopping = False
        self._stop_lock = threading.RLock()
        self._io_done = threading.Condition(self._stop_lock)
        self._active_io = 0

    def start(self):
        try:
            if self.stopping:
                raise InterruptedError('witness stopped during placement')
            ws = WebSocket.open(self.host, self.port, f'/{int(time.time())}/SND', deadline=self.deadline)
            with self._stop_lock:
                self._close_sock = self._shutdown_sock = ws.sock
                ws.sock = _SessionSocket(self, ws.sock)
                self.ws = ws
                if self.stopping:
                    self.stop()
                    raise InterruptedError('witness stopped during placement')
            self.stream = KiwiStream(self.ws, self.dial_hz / 1000, name=f'{self.ws.host}:{self.port}', **self.options)
            self.stream.start()
        except BaseException as error:
            self.close()
            if isinstance(error, RuntimeError) and 'channels are taken' in str(error):
                raise Busy(str(error)) from error
            raise
        self.sample_rate = self.stream.sample_rate
        return self

    def placed(self):
        with self._stop_lock:
            if not self.stopping:
                self.ws.deadline = None
                self.ws.sock.settimeout(20.)

    def blocks(self):
        try:
            blocks = self.stream.blocks()
            while not self.stopping:
                data = next(blocks, None)
                if data is None:
                    break
                yield SimpleNamespace(samples=np.frombuffer(data, dtype='>i2').astype(np.float32) / 32768)
        except (OSError, EOFError):
            if not self.stopping:
                raise

    def stop(self):
        with self._io_done:
            self.stopping = True
            # A signal handler can re-enter this thread; detach the socket atomically.
            sock = self.__dict__.pop('_shutdown_sock', None)
            if sock is not None:
                self._active_io += 1
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                with self._io_done:
                    self._active_io -= 1
                    self._io_done.notify_all()

    def close(self):
        """Owner cleanup, after stream I/O unwinds; never run in a stop callback."""
        self.stop()
        with self._io_done:
            self._io_done.wait_for(lambda: self._active_io == 0)
            sock = self.__dict__.pop('_close_sock', None)
        if sock is not None:
            sock.close()
