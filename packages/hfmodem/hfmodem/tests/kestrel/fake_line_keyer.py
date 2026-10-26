# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keying-line doubles: a decision-only LineKeyer, and a cross-process line.

Neither is device, socket or physical-line evidence.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from pathlib import Path


class FakeLineKeyer:
    def __init__(self, device, log, *, delay=0.0, events: Path | None = None):
        self.device, self.log = device, log
        self.delay, self.events = delay, events
        self.calls = []
        self.high = False
        self.must_retire = False
        self.released_low: bool | None = None
        self.result = True
        self.key_started = threading.Event()
        self.key_release = None
        self.unkeyed = threading.Event()
        self.gone = False

    def _record(self, event):
        self.calls.append(event)
        if self.events is not None:
            with self.events.open('a') as out:
                out.write(f'{event}\n')

    def arm(self):
        self._record('arm')
        self.high = False

    def key(self, on, why=''):
        self._record(on)
        if on:
            self.key_started.set()
            if self.key_release is not None:
                if not self.key_release.wait(5):
                    raise AssertionError('test did not release blocked key-up')
            if self.delay:
                time.sleep(self.delay)
        if self.gone or not self.result:
            self.must_retire = True
            return False
        self.high = on
        if not on:
            self.unkeyed.set()
        return True

    def hand_back(self):
        self._record('hand_back')
        self.high = False
        self.released_low = True


#: Names the `FakeLine` file a child process keys through; see `install`.
LINE_ENV = "HFMODEM_FAKE_LINE"


class LineRecord:
    """One keying line shared by every process that opens it, as a JSONL file.

    Each row is an edge, ``{"t", "pid", "on"}`` in ``time.time()``, so a test
    can bracket audio against when the line moved. ``stick_after`` drops that
    many times and then refuses: the next deassert leaves the line high for good.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def configure(self, *, high: bool = False, stick_after: int | None = None) -> None:
        self.path.write_text(json.dumps({"stick_after": stick_after}) + "\n"
                             + (json.dumps({"t": time.time(), "pid": 0, "on": True}) + "\n"
                                if high else ""))

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]

    def edges(self, pid: int | None = None) -> list[tuple[float, int, bool]]:
        return [(r["t"], r["pid"], r["on"]) for r in self._rows()
                if "on" in r and (pid is None or r["pid"] == pid)]

    @property
    def high(self) -> bool:
        edges = self.edges()
        return bool(edges) and edges[-1][2]

    def raised(self, pid: int | None = None) -> list[float]:
        return [t for t, _, on in self.edges(pid) if on]

    def raised_before(self, t: float) -> float | None:
        return next((at for at in reversed(self.raised()) if at <= t), None)

    def dropped_after(self, t: float) -> float | None:
        return next((at for at, _, on in self.edges() if at >= t and not on), None)

    def wait_for_key(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.raised():
                return True
            time.sleep(0.01)
        return False

    def drive(self, on: bool) -> bool:
        """Move the line under an exclusive lock; True when it reads ``on`` after."""
        with self.path.open("a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            rows = self._rows()
            conf = next((r for r in rows if "stick_after" in r), {})
            edges = [r["on"] for r in rows if "on" in r]
            high = bool(edges) and edges[-1]
            stuck = any(r.get("stuck") for r in rows)
            if high == on:
                return True
            if not on and (stuck or (conf.get("stick_after") is not None
                                     and edges.count(False) >= conf["stick_after"])):
                if not stuck:
                    fh.write(json.dumps({"stuck": True}) + "\n")
                return False
            fh.write(json.dumps({"t": time.time(), "pid": os.getpid(), "on": on}) + "\n")
            return True


class FakeLine:
    """`core.ptt.RtsPtt` with the ioctls answered by a `LineRecord`."""

    released_low: bool | None = None

    def __init__(self, port: str, *, line: str = "rts") -> None:
        self.port = port
        self._line = LineRecord(Path(os.environ[LINE_ENV]))
        self._open = False

    def open(self) -> None:
        self._line.drive(False)
        self._open = True

    def assert_(self, on: bool) -> None:
        from hfmodem.core.ptt import PttError
        if not self._open:
            raise PttError(f"{self.port} is not open")
        self._line.drive(on)

    def sense(self) -> bool | None:
        return self._line.high if self._open else None

    def release(self) -> None:
        if self._open:
            self._open = False
            self.released_low = self._line.drive(False)


def install() -> None:
    """Swap the line under every keyer in this process, when a test names one."""
    if os.environ.get(LINE_ENV):
        from hfmodem.core import ptt
        ptt.RtsPtt = FakeLine


def pty_line():
    """A keying line to name: the far end of a pty held open for the fixture's life,
    a character device the arm gate accepts and the claim finds unheld (/dev/null is
    held by every process). Every open of it under `install` is a `FakeLine`."""
    near, far = os.openpty()
    path = os.ttyname(far)
    os.close(far)
    try:
        yield path
    finally:
        os.close(near)


#: Puts `FakeLine` under `core.ptt` before a child imports anything, then hands on
#: to the interpreter's own sitecustomize, which this one shadows.
_LINE_SHIM = """\
import sys
from pathlib import Path
from hfmodem.tests.kestrel.fake_line_keyer import install
install()
here = Path(__file__).resolve().parent
for d in sys.path:
    nxt = Path(d) / "sitecustomize.py"
    if Path(d).resolve() != here and nxt.exists():
        exec(compile(nxt.read_text(), str(nxt), "exec"))
        break
"""


def line_shim(root: Path) -> Path:
    """The directory to put first on a child's PYTHONPATH so it keys a `FakeLine`."""
    shim = root / "line-shim"
    shim.mkdir(exist_ok=True)
    (shim / "sitecustomize.py").write_text(_LINE_SHIM)
    return shim
