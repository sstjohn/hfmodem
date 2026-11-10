# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded pipe process adapter. No shell, guessed prompts, or runtime fetches."""
from __future__ import annotations

import codecs
import os
from pathlib import Path
import queue
import selectors
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

from .base import GameEvent


@dataclass(frozen=True)
class InputProfile:
    mode: str = "line"
    encoding: str = "utf-8"
    terminator: bytes = b"\n"
    max_input_bytes: int = 256
    max_output_bytes: int = 32768
    write_timeout: float = 5.0
    cleanup_seconds: float = 1.0
    structured: bool = False

    def __post_init__(self):
        if self.mode not in {"line", "key", "pause", "unknown"}:
            raise ValueError("invalid input profile")
        if not (1 <= self.max_input_bytes <= 4096 and 256 <= self.max_output_bytes <= 1048576):
            raise ValueError("invalid process bounds")
        if self.write_timeout <= 0 or self.cleanup_seconds <= 0:
            raise ValueError("invalid lifecycle bounds")
        codecs.lookup(self.encoding)


class ProcessAdapter:
    """All spawn/filesystem/I/O/reaping work runs in one worker thread.

    World lock inheritance protects against a surviving direct backend after
    owner death. A profile spawning children must preserve that descriptor;
    profiles that close it in descendants are not validated shared writers.
    """
    capabilities = frozenset()

    def __init__(self, argv: tuple[str, ...] | list[str], *, profile: InputProfile | None = None,
                 env: dict[str, str] | None = None):
        if not argv or not Path(argv[0]).is_absolute():
            raise ValueError("runtime executable must be an absolute installed path")
        self.argv = tuple(argv)
        self.profile = profile or InputProfile()
        self.env = dict(env or {})
        self._mutex = threading.Lock()
        self._events: list[GameEvent] = []
        self._event_bytes = 0
        self._commands: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._overflow = False
        self._thread: threading.Thread | None = None
        self._ready = False
        self._mode = self.profile.mode
        self._turn = 0
        self.directory: Path | None = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, player: str, directory: Path) -> None:
        if self._thread is not None:
            raise RuntimeError("adapter instances cannot be restarted")
        self.directory = Path(directory)
        self._thread = threading.Thread(target=self._run, name="bbs-game", daemon=True)
        self._thread.start()

    def _emit(self, event: GameEvent) -> None:
        with self._mutex:
            size = len(event.text.encode("utf-8")) + 64
            if self._event_bytes + size > self.profile.max_output_bytes:
                self._overflow = True
                self._stop.set()
                return
            self._events.append(event)
            self._event_bytes += size

    def poll(self) -> list[GameEvent]:
        with self._mutex:
            result, self._events = self._events, []
            self._event_bytes = 0
            return result

    def submit(self, text: str, mode: str = "default") -> bool:
        if not isinstance(text, str) or "\n" in text or "\r" in text or "\x00" in text:
            return False
        if any(ord(c) < 32 or ord(c) == 127 for c in text):
            return False
        actual = self._mode if mode == "default" else mode
        if actual == "unknown":
            actual = self.profile.mode
        if actual not in {"line", "key", "pause"}:
            return False
        try:
            data = text.encode(self.profile.encoding, "strict")
        except UnicodeError:
            return False
        if len(data) > self.profile.max_input_bytes or (actual == "key" and len(text) != 1):
            return False
        if actual == "pause" and text:
            return False
        if actual in {"line", "pause"}:
            data += self.profile.terminator
        with self._mutex:
            if not self._ready or self._stop.is_set():
                return False
            self._ready = False
            self._commands.put_nowait(data)
        return True

    def close(self, reason: str) -> None:
        self._stop.set()
        with self._mutex:
            self._ready = False

    def _prompt(self, mode: str) -> None:
        with self._mutex:
            self._ready = True
            self._mode = mode
            self._turn += 1
        self._emit(GameEvent("prompt", mode=mode, turn_id=str(self._turn)))

    def _prepare(self) -> None:
        pass

    def _control(self, line: str) -> None:
        # Opaque stderr is diagnostics, never caller-facing text or readiness.
        pass

    def _run(self) -> None:
        from ..store import OwnershipError, WorldLock
        process = None
        world_lock = None
        outcome = "closed"
        try:
            assert self.directory is not None
            self.directory.mkdir(parents=True, exist_ok=True)
            self.directory = self.directory.resolve()
            world_lock = WorldLock(self.directory / ".world.lock")
            self._prepare()
            environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(self.directory),
                           **self.env}
            if self._stop.is_set():
                return
            # No launch-time package runners or ambient environment inheritance.
            process = subprocess.Popen(self.argv, cwd=self.directory, env=environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True, pass_fds=(world_lock.fileno(),), bufsize=0)
            assert process.stdin and process.stdout and process.stderr
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "out")
            selector.register(process.stderr, selectors.EVENT_READ, "err")
            decoder = codecs.getincrementaldecoder(self.profile.encoding)(errors="replace")
            errors = bytearray()
            pending = b""
            write_started = 0.0
            if not self.profile.structured:
                self._prompt("unknown")
            while not self._stop.is_set():
                if not pending:
                    try:
                        pending = self._commands.get_nowait()
                        write_started = time.monotonic()
                    except queue.Empty:
                        pass
                if pending:
                    if time.monotonic() - write_started > self.profile.write_timeout:
                        raise TimeoutError("backend write timeout")
                    try:
                        count = os.write(process.stdin.fileno(), pending)
                        pending = pending[count:]
                        if not pending and not self.profile.structured:
                            self._prompt("unknown")
                    except BlockingIOError:
                        pass
                for key, _ in selector.select(.02):
                    try:
                        data = os.read(key.fileobj.fileno(), 4096)
                    except BlockingIOError:
                        continue
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "out":
                        text = decoder.decode(data)
                        if text:
                            self._emit(GameEvent("text", text=text))
                    else:
                        errors.extend(data)
                        if len(errors) > 8192:
                            raise RuntimeError("backend diagnostic flood")
                        while b"\n" in errors:
                            line, _, rest = errors.partition(b"\n")
                            errors[:] = rest
                            # Runtime flushes stdout before its control write.
                            # Drain those already-readable bytes before exposing
                            # the matching prompt, independent of select order.
                            while not self._stop.is_set():
                                try:
                                    preceding = os.read(process.stdout.fileno(), 4096)
                                except BlockingIOError:
                                    break
                                if not preceding:
                                    break
                                text = decoder.decode(preceding)
                                if text:
                                    self._emit(GameEvent("text", text=text))
                            if not self._stop.is_set():
                                self._control(line.decode("ascii", "replace"))
                if process.poll() is not None and not selector.get_map():
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        self._emit(GameEvent("text", text=tail))
                    outcome = "eof" if process.returncode == 0 else "crash"
                    break
            selector.close()
            if self._overflow:
                outcome = "output_limit"
        except Exception:
            outcome = "error"
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=self.profile.cleanup_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                # The leader may already have exited while descendants ignore
                # TERM; do not leave those writers running merely because wait
                # on the direct child returned immediately.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream:
                        stream.close()
            if world_lock:
                world_lock.close()
                if process is not None:
                    # KILL is asynchronous. Do not report successful cleanup
                    # while an inherited writer still owns the world lock.
                    deadline = time.monotonic() + self.profile.cleanup_seconds
                    while True:
                        try:
                            probe = WorldLock(self.directory / ".world.lock")
                            probe.close()
                            break
                        except OwnershipError:
                            if time.monotonic() >= deadline:
                                outcome = "cleanup_uncertain"
                                break
                            time.sleep(.005)
            with self._mutex:
                self._ready = False
                # Lifecycle delivery must survive a full display queue.
                self._events.append(GameEvent("exited", outcome=outcome))


DoorAdapter = ProcessAdapter
