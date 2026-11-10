# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

import codecs
import io
import os
import queue
import select
import threading
import time


class Reader:
    def __init__(self, stream=None, *, bounded=None, lines=False, encoding=None, errors=None):
        self.stream, self.bounded, self.lines = stream, bounded, lines
        self.encoding, self.errors = encoding, errors
        self.queue = queue.Queue()
        self.done = threading.Event()
        self.eof = threading.Event()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='pipe-reader', daemon=True)
        self.thread.start()

    def _blocks(self):
        if self.bounded is not None:
            while not self.stop.is_set():
                result = self.bounded.read(65536, deadline=time.monotonic() + .05)
                if not result.confirmed:
                    raise OSError('pipe read cancellation unresolved')
                if result.status == 'eof':
                    return
                if result.status == 'data':
                    yield result.data
            return
        try:
            fd = self.stream if isinstance(self.stream, int) else self.stream.fileno()
        except (AttributeError, OSError):
            yield from self.stream
            return
        while not self.stop.is_set():
            if os.name == 'nt':
                from . import winproc
                count, eof = winproc.pipe_state(fd)
                if eof:
                    return
                if not count:
                    self.stop.wait(.05)
                    continue
                # This is the sole reader of a synchronous anonymous pipe. Only
                # bytes already buffered are read, so no pending read can survive.
                block = os.read(fd, min(count, 65536))
            else:
                if not select.select([fd], [], [], .05)[0]:
                    continue
                block = os.read(fd, 65536)
            if not block:
                return
            yield block

    def _run(self):
        decoder = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder(self.encoding or getattr(self.stream, 'encoding', None) or 'utf-8')(
                errors=self.errors or getattr(self.stream, 'errors', None) or 'strict'), translate=True)
        pending = ''
        try:
            for block in self._blocks():
                if not self.lines:
                    self.queue.put(block)
                    continue
                pending += decoder.decode(block) if isinstance(block, bytes) else block
                while '\n' in pending:
                    line, pending = pending.split('\n', 1)
                    self.queue.put(line.rstrip('\r') + '\n')
            if self.lines:
                pending += decoder.decode(b'', final=True)
                if pending:
                    self.queue.put(pending)
            if not self.stop.is_set():
                self.eof.set()
        except BaseException as error:
            self.error = error
        finally:
            self.done.set()

    def close(self):
        self.stop.set()
        self.thread.join(1.5)
        if self.thread.is_alive():
            return False
        if self.bounded is not None:
            return self.bounded.close()
        return True


class Owner:
    def __init__(self, fd):
        self.reader = Reader(os.dup(fd))

    def gone(self):
        while True:
            try:
                self.reader.queue.get_nowait()
            except queue.Empty:
                return self.reader.done.is_set()

    def close(self):
        if self.reader.stream is None:
            return True
        if not self.reader.close():
            return False
        os.close(self.reader.stream)
        self.reader.stream = None
        return True
