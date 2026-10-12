# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Commands and a sample-bounded audio backlog owned by one modem thread."""
from collections import deque
import queue
import threading


class ModemWorker:
    worker_name = "bateleur"

    def _init_worker(self):
        self._q = queue.SimpleQueue()
        self._audio_buffer = deque()
        self._audio_samples = 0
        self._audio_limit = int(6.4 * self.card_rate)
        self._work = threading.Condition()
        self._thread = None
        self._running = False
        self._cleanup = queue.SimpleQueue()
        self._audio_gap = False
        self.audio_dropped = 0

    def mark_audio_gap(self):
        with self._work:
            self._audio_gap = True

    def _feed(self, item):
        with self._work:
            size = len(item[-1])
            if not size:
                return
            gap = self._audio_gap
            self._audio_gap = False
            if size > self._audio_limit:
                cut = size - self._audio_limit
                audio = item[-1][cut:].copy()
                item = (item[0] + cut, audio) if len(item) == 2 else (audio,)
                size = len(audio)
                gap = True
                self.audio_dropped += 1
            dropped = False
            while self._audio_samples + size > self._audio_limit:
                _, old = self._audio_buffer.popleft()
                self._audio_samples -= len(old[-1])
                self.audio_dropped += 1
                dropped = True
            if dropped:
                if self._audio_buffer:
                    _, first = self._audio_buffer[0]
                    self._audio_buffer[0] = (True, first)
                else:
                    gap = True
            self._audio_buffer.append((gap, item))
            self._audio_samples += size
            self._work.notify()

    def post(self, fn):
        with self._work:
            self._q.put(("call", fn))
            self._work.notify()

    def post_cleanup(self, fn):
        with self._work:
            self._cleanup.put(fn)
            self._work.notify()

    def _clean(self):
        while not self._cleanup.empty():
            self._cleanup.get_nowait()()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=self.worker_name, daemon=True)
        self._thread.start()

    def stop(self):
        with self._work:
            self._running = False
            self._work.notify()
        if self._thread:
            self._thread.join()
            self._thread = None
        self._clean()

    def _after_work(self):
        pass

    def _loop(self):
        while self._running:
            self._clean()
            with self._work:
                self._work.wait_for(lambda: not self._running or not self._q.empty()
                                    or self._audio_buffer or not self._cleanup.empty())
                if not self._running:
                    break
                if not self._q.empty():
                    kind, item = self._q.get_nowait()
                elif self._audio_buffer:
                    kind = "audio"
                    item = self._audio_buffer.popleft()
                    self._audio_samples -= len(item[1][-1])
                else:
                    continue
            try:
                if kind == "audio":
                    gap, args = item
                    if gap:
                        self.reset_receive()
                    self._audio(*args)
                else:
                    item()
            except Exception as exc:
                self._log(f"{self.worker_name}: {kind} failed ({type(exc).__name__}: {exc})")
            self._after_work()
