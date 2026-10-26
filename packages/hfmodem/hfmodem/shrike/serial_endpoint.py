# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Passive, bounded PTC serial I/O, serviced by the modem's existing owner.

This module does not create a thread or advance an ARQ clock. A production
sample-clock driver services it between receive/transmit opportunities; the
standalone demonstration supplies its own clock separately.
"""
from __future__ import annotations

import errno
import math
import os
from pathlib import Path
import select
import stat
import termios
import threading
import time
import tty


class PtcSerialEndpoint:
    """One trusted/exclusive Unix attachment, including its final retirement.

    Open, service and close run on the same owner as PtcHost and the ARQ. Even
    cleanup must not call the state machine from a serial-reader thread. A new
    attachment uses a new endpoint; no kernel bytes or parser retry cache may
    cross that boundary.
    """

    def __init__(self, host, *, endpoint='', output_limit=256*1024, drain_timeout=30.0):
        if (not isinstance(output_limit, int) or isinstance(output_limit, bool)
                or output_limit <= 0 or not math.isfinite(drain_timeout)
                or drain_timeout <= 0):
            raise ValueError("positive integer output limit and finite drain timeout required")
        self.host = host
        self.output_limit = output_limit
        self.drain_timeout = drain_timeout
        self.name = None
        self._alias = Path(endpoint) if endpoint else None
        self._alias_identity = self._slave_name = None
        self.alias_error = None
        self.hangup = False
        self._fd = self._replica = None
        self._owner = None
        self._used = self._attached = False
        self._pending = bytearray()
        self._last_progress = 0.0

    @property
    def closed(self):
        return self._fd is None

    @property
    def pending_bytes(self):
        return len(self._pending)

    def _check_owner(self):
        if self._owner is not None and self._owner is not threading.current_thread():
            raise RuntimeError("PTC serial I/O must run on its modem owner")

    def open(self, device=None):
        self._check_owner()
        if self._used:
            raise RuntimeError("a PTC serial endpoint serves only one attachment")
        self._used = True
        self._owner = threading.current_thread()
        try:
            if device and self._alias is not None:
                raise ValueError('PTC endpoint alias cannot also open a serial device')
            if device:
                self._fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
                self.name = device
            else:
                self._fd, self._replica = os.openpty()
                self._slave_name = os.ttyname(self._replica)
                self.name = self._slave_name
                tty.setraw(self._replica)
                if self._alias is not None:
                    # Exclusive creation preserves every existing operator path,
                    # including a dangling symlink. Each attachment has a fresh
                    # PTY; keeping a master open would retain old kernel bytes.
                    self._alias.symlink_to(self._slave_name)
                    info = self._alias.lstat()
                    if (not stat.S_ISLNK(info.st_mode)
                            or os.readlink(self._alias) != self._slave_name):
                        raise RuntimeError('PTC endpoint name changed during creation')
                    self._alias_identity = (info.st_dev, info.st_ino)
                    self.name = str(self._alias)
            tty.setraw(self._fd, termios.TCSANOW)
            os.set_blocking(self._fd, False)
            # Retain the replica only until the first client input. Keeping it
            # forever would hide the last application's hangup.
            self._attached = True
            self.host.set_listen(True)
            self._queue(self.host.open())
            return self.name
        except BaseException:
            self.close()
            raise

    def _queue(self, data):
        if len(self._pending) + len(data) > self.output_limit:
            raise BufferError("PTC SERIAL OUTPUT OVERFLOW")
        if data and not self._pending:
            self._last_progress = time.monotonic()
        self._pending += data

    def service(self, *, timeout=0.0):
        """Service at most one input/output chunk without ever ticking the ARQ.

        A sample-clock owner uses the default zero timeout. The optional wait
        is for the standalone loop; output no-progress bounds still apply.
        Hangup retires the attachment and returns False. I/O or host-processing
        faults retire it before raising, so callers cannot accidentally retain
        an active modem behind a failed endpoint.
        """
        self._check_owner()
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("serial service timeout must be finite and nonnegative")
        if self.closed:
            return False
        try:
            # Terminal RX/link output is asynchronous. A quiet application is
            # still a reader; no extra host command or invented ARQ tick is
            # needed to publish these bytes through the bounded serial spool.
            self._queue(self.host.drain_terminal_output())
            now = time.monotonic()
            if self._pending:
                timeout = min(timeout, max(0.0,
                    self._last_progress + self.drain_timeout - now))
            ready, writable, _ = select.select(
                [self._fd], [self._fd] if self._pending else [], [], timeout)
            if ready:
                try:
                    chunk = os.read(self._fd, 4096)
                except BlockingIOError:
                    chunk = None
                except OSError as exc:
                    if exc.errno not in (errno.EIO, errno.ENXIO):
                        raise
                    chunk = b""
                if chunk == b"":
                    self.hangup = True
                    self.close()
                    return False
                if chunk:
                    if self._replica is not None:
                        os.close(self._replica)
                        self._replica = None
                    self._queue(self.host.feed(chunk))
            if writable and self._pending:
                try:
                    written = os.write(self._fd, self._pending)
                except BlockingIOError:
                    written = 0
                except OSError as exc:
                    if exc.errno not in (errno.EIO, errno.ENXIO):
                        raise
                    self.hangup = True
                    self.close()
                    return False
                if written:
                    del self._pending[:written]
                    self._last_progress = time.monotonic()
            # Host output retirement is owner work, but does not advance the
            # protocol's sample/cycle clock.
            self.host.service_host_output()
            if self._pending and time.monotonic() - self._last_progress >= self.drain_timeout:
                raise TimeoutError("PTC SERIAL OUTPUT DRAIN TIMEOUT")
            if self.host.host_output_failure:
                raise ConnectionAbortedError(self.host.host_output_failure)
            return True
        except BaseException:
            self.close()
            raise

    def close(self):
        self._check_owner()
        fd, replica = self._fd, self._replica
        attached = self._attached
        self._fd = self._replica = None
        self._attached = False
        try:
            if attached:
                if self._pending:
                    self.host.log(
                        f"attachment retired {len(self._pending)} unread serial output bytes")
                self._pending.clear()
                self.host.attachment_lost()
        finally:
            try:
                if replica is not None:
                    os.close(replica)
            finally:
                try:
                    if fd is not None:
                        os.close(fd)
                finally:
                    self._remove_alias()

    def _remove_alias(self):
        identity, self._alias_identity = self._alias_identity, None
        if identity is None:
            return
        try:
            info = self._alias.lstat()
            if ((info.st_dev, info.st_ino) == identity and stat.S_ISLNK(info.st_mode)
                    and os.readlink(self._alias) == self._slave_name):
                self._alias.unlink()
            else:
                self.alias_error = 'PTC endpoint name replaced while active'
        except FileNotFoundError:
            pass
        except OSError as exc:
            # A namespace cleanup failure cannot prevent descriptor retirement
            # or replace an original RF/serial failure. The lifecycle owner can
            # report it; a future exclusive bind will also fail visibly.
            self.alias_error = f'PTC endpoint name could not be removed: {exc}'
