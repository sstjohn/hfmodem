# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A passive PTC byte stream serviced only by the native PACTOR owner."""
import math
import select
import threading
import time


class PtcStreamEndpoint:
    def __init__(self, host, stream, *, output_limit=256*1024, drain_timeout=30.):
        if (not isinstance(output_limit, int) or isinstance(output_limit, bool)
                or output_limit <= 0 or not math.isfinite(drain_timeout) or drain_timeout <= 0):
            raise ValueError('positive output limit and finite drain timeout required')
        self.host, self.stream = host, stream
        self.output_limit, self.drain_timeout = output_limit, drain_timeout
        self._pending = bytearray()
        self._owner = None
        self._used = self._attached = False
        self._last_progress = 0.
        self.hangup = False

    @property
    def closed(self):
        return self.stream is None

    @property
    def pending_bytes(self):
        return len(self._pending)

    def _check_owner(self):
        if self._owner is not None and self._owner is not threading.current_thread():
            raise RuntimeError('PTC stream I/O must run on its modem owner')

    def _queue(self, data):
        if len(self._pending) + len(data) > self.output_limit:
            raise BufferError('PTC STREAM OUTPUT OVERFLOW')
        if data and not self._pending:
            self._last_progress = time.monotonic()
        self._pending += data

    def open(self):
        self._check_owner()
        if self._used:
            raise RuntimeError('a PTC stream endpoint serves only one attachment')
        self._used = True
        self._owner = threading.current_thread()
        try:
            try:
                self.stream.setblocking(False)
            except OSError:
                if self.stream.fileno() >= 0:
                    raise
                # The lease ended while waiting for the owner to adopt it.
                # No host attachment was opened, so no owner reset is owed.
                self.hangup = True
                self.close()
                return False
            self._attached = True
            self.host.set_listen(True)
            self._queue(self.host.open())
            return True
        except BaseException:
            self.close()
            raise

    def service(self, *, timeout=0.):
        self._check_owner()
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('stream service timeout must be finite and nonnegative')
        if self.closed:
            return False
        try:
            # The authenticated broker can close the stream to revoke its lease.
            if self.stream.fileno() < 0:
                self.hangup = True
                self.close()
                return False
            self._queue(self.host.drain_terminal_output())
            if self._pending:
                timeout = min(timeout, max(0., self._last_progress + self.drain_timeout
                                           - time.monotonic()))
            try:
                data = None
                readable, writable, _ = select.select(
                    [self.stream], [self.stream] if self._pending else [], [], timeout)
                if readable:
                    try:
                        data = self.stream.recv(4096)
                    except BlockingIOError:
                        data = None
                    if data == b'':
                        self.hangup = True
                        self.close()
                        return False
                if writable and self._pending:
                    try:
                        count = self.stream.send(self._pending)
                    except BlockingIOError:
                        count = 0
                    if count:
                        del self._pending[:count]
                        self._last_progress = time.monotonic()
            except (OSError, ValueError):
                # Socket revocation/reset is an attachment end, never permission
                # to run a new modem or to preserve the old parser/cache.
                self.hangup = True
                self.close()
                return False
            if data:
                self._queue(self.host.feed(data))
            self.host.service_host_output()
            if self._pending and time.monotonic() - self._last_progress >= self.drain_timeout:
                raise TimeoutError('PTC STREAM OUTPUT DRAIN TIMEOUT')
            if self.host.host_output_failure:
                raise ConnectionAbortedError(self.host.host_output_failure)
            return True
        except BaseException:
            self.close()
            raise

    def close(self):
        self._check_owner()
        stream, self.stream = self.stream, None
        attached, self._attached = self._attached, False
        try:
            if attached:
                if self._pending:
                    self.host.log(f'attachment retired {len(self._pending)} unread stream output bytes')
                self._pending.clear()
                self.host.attachment_lost()
        finally:
            if stream is not None:
                stream.close()
