# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Lifecycle-managed AGWPE TCP or Tracker POSIX serial endpoints."""
from collections import deque
import os
from pathlib import Path
import select
import threading
import time

from hfmodem.core.hostio import Connection, listen


class AgwServer:
    def __init__(self, runtime, *, host='127.0.0.1', port=8000):
        self.runtime, self.host, self.port = runtime, host, port
        self._stop = threading.Event()
        self._listener = self._client = self._thread = self._worker = None
        self._lock = threading.Lock()

    def start(self):
        listener = listen(self.host, self.port, backlog=4)
        try:
            self.runtime.start()
        except BaseException:
            listener.close()
            raise
        self._listener = listener
        self.port = listener.getsockname()[1]
        self._thread = threading.Thread(target=self._accept, name='rpr-agwpe-accept', daemon=True)
        self._thread.start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                sock, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with self._lock:
                if self._client is not None or self._stop.is_set():
                    sock.close()
                    continue
                self._client = Connection(sock)
                self._worker = threading.Thread(target=self._serve, args=(self._client,),
                                                name='rpr-agwpe-client', daemon=True)
                self._worker.start()

    def _serve(self, conn):
        attachment = None
        try:
            attachment = self.runtime.attach('agwpe', conn.send_bounded,
                                             failed=lambda _: conn.close(discard=True))
            while not self._stop.is_set() and attachment.alive:
                data = conn.recv(8192)
                if not data:
                    break
                self.runtime.feed(attachment, data)
        except (OSError, ValueError, BufferError, TimeoutError):
            pass
        finally:
            if attachment is not None:
                try:
                    self.runtime.detach(attachment)
                except (OSError, TimeoutError):
                    pass
            conn.close(discard=True)
            conn._writer.join(1)
            conn._delivery_thread.join(1)
            with self._lock:
                if self._client is conn:
                    self._client = None

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        if self._client is not None:
            self._client.close(discard=True)
        for worker in (self._thread, self._worker):
            if worker is not None:
                worker.join(3)
                if worker.is_alive():
                    raise RuntimeError('AGWPE endpoint did not retire')
        self.runtime.stop()
        return True


class TrackerServer:
    def __init__(self, runtime, *, endpoint=''):
        self.runtime = runtime
        self.endpoint = endpoint
        self._alias = Path(endpoint) if endpoint else None
        self._slave_path = None
        self._master = self._thread = None
        self._stop = threading.Event()
        self.failure = ''

    def start(self):
        if os.name != 'posix':
            raise NotImplementedError('Tracker requires a qualified COM bridge on Windows')
        import pty
        import tty
        master, slave = pty.openpty()
        try:
            tty.setraw(slave)
            self._slave_path = os.ttyname(slave)
            os.set_blocking(master, False)
            if self._alias is not None:
                # Creation is exclusive: never replace an operator-owned path.
                self._alias.symlink_to(self._slave_path)
            else:
                self.endpoint = self._slave_path
            self.runtime.start()
        except BaseException:
            os.close(master)
            if (self._alias is not None and self._alias.is_symlink()
                    and os.readlink(self._alias) == self._slave_path):
                self._alias.unlink()
            raise
        finally:
            os.close(slave)
        self._master = master
        self._thread = threading.Thread(target=self._run, name='rpr-tracker-pty', daemon=True)
        self._thread.start()

    def _run(self):
        attachment = None
        output, offset, started = deque(), 0, None
        try:
            while not self._stop.is_set():
                try:
                    readers, writers, _ = select.select([self._master],
                                                       [self._master] if output else [], [], .02)
                    if self._master in readers:
                        data = os.read(self._master, 8192)
                        if not data:
                            raise ConnectionResetError('Tracker serial closed')
                        if attachment is None:
                            import tty
                            tty.setraw(self._master)
                            attachment = self.runtime.attach('tracker')
                        reply = self.runtime.feed(attachment, data)
                        if reply:
                            if sum(map(len, output)) + len(reply) > 65536 or len(output) >= 64:
                                raise BufferError('Tracker serial output overflow')
                            output.append(reply)
                            if started is None:
                                started = time.monotonic()
                    if attachment is not None and not attachment.alive:
                        raise BufferError(attachment.failure)
                    if self._master in writers and output:
                        count = os.write(self._master, output[0][offset:])
                        if not count:
                            raise ConnectionResetError('Tracker serial write stopped')
                        offset += count
                        if offset == len(output[0]):
                            output.popleft()
                            offset = 0
                            started = time.monotonic() if output else None
                    if started is not None and time.monotonic() - started > 5:
                        raise BufferError('Tracker serial output timed out')
                except BlockingIOError:
                    continue
                except OSError:
                    # Closing the slave retires that attachment. Keeping only
                    # the master open lets POSIX report the next real reopen.
                    if attachment is not None:
                        self.runtime.detach(attachment)
                        attachment = None
                    import termios
                    # Already written responses can remain in the PTY kernel
                    # queue after close. They also belong to the old attachment.
                    termios.tcflush(self._master, termios.TCOFLUSH)
                    output.clear()
                    offset, started = 0, None
                    self._stop.wait(.02)
        except (ValueError, BufferError, TimeoutError) as exc:
            self.failure = str(exc)
        finally:
            if attachment is not None:
                self.runtime.detach(attachment)
            os.close(self._master)
            self._master = None

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3)
            if self._thread.is_alive():
                raise RuntimeError('Tracker endpoint did not retire')
        self.runtime.stop()
        if (self._alias is not None and self._alias.is_symlink()
                and os.readlink(self._alias) == self._slave_path):
            self._alias.unlink()
        return True
