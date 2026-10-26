# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PTC host controls at the existing PACTOR engine's operation boundaries.

No thread, decoder, audio device or cycle clock lives here. The contact owner
can service serial I/O while its producer waits, without applying a disconnect
inside the suspended ARQ operation. Data admission remains the engine's FIFO.
"""
from collections import deque
from contextlib import contextmanager, nullcontext
import threading
import sys

from .arq import State

ACTIONS = frozenset(('connect', 'disconnect', 'abort', 'over', 'breakin', 'changeover'))


class PtcOwner:
    def __init__(self, host, *, limit=64, defer_connect=False):
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 64:
            raise ValueError('owner control limit must be 1..64')
        self.host, self.limit = host, limit
        self._thread = None
        self._depth = 0
        self._pending = deque()
        self._draining = False
        self._closing = self._abort_requested = False
        self._retire_requested = self._retirement_done = False
        self.retired = False
        self.cleanup_error = None
        self.defer_connect = defer_connect
        self._admitting_connect = False
        self._connect = None
        self._contact_active = False
        self._driver_active = False

    @property
    def contact_active(self):
        return self._contact_active

    @contextmanager
    def contact(self):
        """Retain acquisition/terminal ownership beyond logical disconnection."""
        self.check()
        if self._contact_active or self.in_operation or self._draining:
            raise RuntimeError('PTC contact driver is already active or inside an operation')
        self._contact_active = True
        try:
            yield
        finally:
            self._contact_active = False
            self._sync()

    @contextmanager
    def driver(self):
        if self._driver_active or self.in_operation:
            raise RuntimeError('PTC contact driver is already active or inside an operation')
        self.check()
        with nullcontext() if self.contact_active else self.contact():
            self._driver_active = True
            try:
                yield
            finally:
                failure = sys.exc_info()[1]
                try:
                    # A finite native bench can return with a call still in
                    # acquisition. A station cannot retain it after its sample
                    # stream closes. Abort preserves original unconfirmed bytes
                    # and final host output; it never retires the attachment.
                    self._connect = None
                    if self.host.arq.state in (State.CONNECTING, State.CONNECTED,
                                                State.DISCONNECTING):
                        self.host.arq.on_host_abort()
                except BaseException as exc:
                    self.cleanup_error = type(exc).__name__
                    self.retired = True
                    if failure is None:
                        raise
                finally:
                    self._driver_active = False

    @property
    def connect_pending(self):
        return self._connect is not None

    def defer_connect_burst(self, mycall, dxcall):
        """Capture only the initial serial C; native raster retries stay direct."""
        if not self.defer_connect or not self._admitting_connect:
            return False
        if self._connect is not None:
            raise RuntimeError('PTC initial connect already pending')
        self._connect = (mycall, dxcall)
        return True

    def start_connect(self, endpoint):
        """Emit the admitted call only after its serial response has been written.

        The sample-clock owner calls this outside serial service. False means
        there is no admitted call or output still needs servicing. A completed
        first burst supplies the native driver's anchor, without another C.
        """
        self.check()
        if self.in_operation or self._draining:
            raise RuntimeError('PTC initial transmit requires an outer owner boundary')
        if endpoint.host is not self.host:
            raise ValueError('PTC connect endpoint belongs to another host')
        if endpoint.closed:
            raise ConnectionAbortedError('PTC connect endpoint closed')
        if self.host.arq.state != State.CONNECTING:
            self._connect = None
        if self._connect is None or endpoint.pending_bytes:
            return False
        args, self._connect = self._connect, None
        with self.operation('initial_connect'):
            try:
                self.host.peer.connect_burst(*args)
            except BaseException:
                # No caller may resume a half-started acquisition after an
                # output failure. Preserve the original failure on cleanup.
                self._pending.clear()
                self._pending.append(('abort', (), {}))
                self._closing = self._abort_requested = True
                raise
        return True

    @property
    def in_operation(self):
        return self._depth > 0

    @property
    def pending_count(self):
        return len(self._pending)

    @property
    def abort_requested(self):
        return self._abort_requested or self.retired

    @property
    def admitting_data(self):
        return not self.retired and not self._closing

    def check(self, *, retired_ok=False):
        current = threading.current_thread()
        if self._thread is None:
            self._thread = current
        elif self._thread is not current:
            raise RuntimeError('PTC host and ARQ must run on their contact owner')
        if self.retired and not retired_ok:
            raise ConnectionAbortedError('PTC attachment retired')

    @contextmanager
    def operation(self, name):
        self.check(retired_ok=name == 'on_host_abort')
        self._depth += 1
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self._depth -= 1
            if not self._depth and not self._draining:
                if failed and self._pending:
                    # A failed RF operation cannot start graceful-close work.
                    # Preserve explicit attachment retirement, otherwise abort
                    # once the failed engine stack has unwound.
                    action = 'retire' if self._retire_requested else 'abort'
                    self._pending.clear()
                    self._pending.append((action, (), {}))
                    self._closing = self._abort_requested = True
                if failed:
                    try:
                        self._drain()
                    except BaseException as exc:
                        self.cleanup_error = type(exc).__name__
                        self.retired = True
                else:
                    self._drain()

    def request(self, name, *args, **kwargs):
        self.check()
        if name not in ACTIONS:
            raise ValueError('unknown PTC owner control')
        if name == 'connect' and (self.in_operation or self._draining or self._closing
                                  or self.contact_active):
            raise ValueError('CHANNEL OWNER BUSY')
        if self.in_operation or self._draining:
            if name == 'abort':
                self._pending.clear()
            elif len(self._pending) >= self.limit:
                raise ValueError('PTC OWNER CONTROL QUEUE FULL')
            self._pending.append((name, args, kwargs))
            if name in ('disconnect', 'abort'):
                self._closing = True
            self._abort_requested |= name == 'abort'
            return
        if name in ('disconnect', 'abort'):
            self._closing = True
        self._run(name, args, kwargs)
        self._sync()

    def retire(self):
        self.check(retired_ok=True)
        if self._retirement_done:
            return
        self.retired = self._retire_requested = self._closing = True
        self._connect = None
        self._pending.clear()
        self._pending.append(('retire', (), {}))
        if not self.in_operation and not self._draining:
            self._drain()

    def reset(self):
        self.check()
        if self.in_operation or self._draining:
            self.request('abort')
            raise ValueError('HOSTMODE REENTRY WAITS FOR ARQ OPERATION')
        self._closing = True
        self._connect = None
        try:
            self.host._attachment_lost_now()
        except BaseException:
            self.retired = True
            raise
        self._pending.clear()
        self._closing = self._abort_requested = False

    def _run(self, name, args, kwargs):
        if name == 'retire':
            self.host._attachment_lost_now()
            self._retirement_done = True
        elif name == 'connect':
            validate = getattr(self.host.peer, 'validate_host_connect', None)
            if validate is not None:
                validate(*args, **kwargs)
            self._admitting_connect = True
            try:
                self.host.arq.on_host_connect(*args, **kwargs)
            finally:
                self._admitting_connect = False
        else:
            getattr(self.host.arq, 'on_host_' + name)(*args, **kwargs)

    def _drain(self):
        self._draining = True
        try:
            while self._pending:
                self._run(*self._pending.popleft())
        except BaseException:
            self.retired = True
            self._pending.clear()
            raise
        finally:
            self._draining = False
            self._sync()

    def _sync(self):
        if self.host.arq.state != State.CONNECTING:
            self._connect = None
        if not self._pending and self.host.arq.state in (State.DISCONNECTED, State.LISTENING):
            self._closing = self._abort_requested = False
