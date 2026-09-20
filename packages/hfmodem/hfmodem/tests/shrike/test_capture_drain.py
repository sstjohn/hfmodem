# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A session must not take its last captures with it when it exits.

`onair` defers every WAV write to a background thread on purpose: a disk stall
between a listening window closing and the next key-down would move a
transmission, and nothing gets to move a transmission. But the thread is a
daemon, so an interpreter shutting down does not wait for it -- whatever is
still queued is dropped, and the drop is silent.

The captures still in flight at exit are the LAST ones. A session that ends
because something interesting happened ends holding exactly the recordings that
would explain it, and the operator finds out the next day, off the analysis,
when the files are not there.

The race is constructed rather than hoped for: the writer is made slow enough
that the queue is certainly still full when the process ends. The counterexample
arm exits through `os._exit`, which skips every shutdown hook there is, and it
must lose captures -- if it ever stops losing them the check above is free.
"""

from __future__ import annotations

import pathlib
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from hfmodem.shrike import onair

# Five captures at 0.4 s a write. Nothing that queues these and exits can have
# emptied the queue by accident: the writer has 2 s of work and the process is
# gone in milliseconds.
_N = 5
_SLOW_S = 0.4

# Spliced in at column zero -- a stray indent makes the child an
# IndentationError that dies before it queues anything, which reads exactly like
# a capture being lost.
_QUEUES_AND_EXITS = """\
import pathlib, sys, time
sys.path.insert(0, {root!r})
import numpy as np
from hfmodem.shrike import onair

outdir = pathlib.Path({outdir!r})


def _slow_write(path, seg, *facts):
    time.sleep({slow!r})
    path.write_bytes(b"RIFF")
    path.with_suffix(".json").write_text("{{}}\\n")


onair._save_capture = _slow_write
{install}
for i in range({n}):
    onair._save_capture_async(outdir / ("rx_%02d.wav" % i), np.zeros(64, np.float32))
(outdir / "queued").write_text("all of them")
{tail}
"""

_INSTALL_SIGNALS = "onair._unkey_on_signal()"


def _child(tmp_path: pathlib.Path, *, install: str = "", tail: str = ""):
    root = str(pathlib.Path(__file__).resolve().parents[3])   # .../packages/hfmodem
    outdir = tmp_path / "cap"
    outdir.mkdir()
    src = tmp_path / "queues_captures.py"
    src.write_text(_QUEUES_AND_EXITS.format(
        root=root, outdir=str(outdir), slow=_SLOW_S, n=_N,
        install=install, tail=tail))
    return outdir, subprocess.Popen(
        [sys.executable, str(src)], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)


def _written(outdir: pathlib.Path) -> int:
    return len(list(outdir.glob("rx_*.wav")))


def test_a_session_that_exits_promptly_still_writes_every_capture(tmp_path) -> None:
    """THE DEFECT. Queue five slow captures, exit at once, count the files."""
    outdir, child = _child(tmp_path)
    child.wait(timeout=60)
    assert _written(outdir) == _N, (
        f"{_N - _written(outdir)} captures went in the queue and never reached "
        f"the disk:\n{child.stdout.read()}")


def test_without_a_drain_the_captures_are_lost(tmp_path) -> None:
    """The counterexample. `os._exit` is the shape the daemon thread already
    had -- no unwinding, no shutdown hooks, nobody waiting. If this ever finds
    all five files the race above was never constructed."""
    outdir, child = _child(tmp_path, tail="import os; os._exit(0)")
    child.wait(timeout=60)
    assert _written(outdir) < _N, (
        "every capture survived an exit that waits for nothing -- the writer is "
        "not slow enough for this file to be measuring anything")


def test_a_signalled_session_keeps_its_tail(tmp_path) -> None:
    """SIGTERM is how an unattended run is stopped, and a run stopped that way
    is one whose last cycles are worth the most. `_unkey_on_signal` turns the
    signal into an unwind; the drain has to be on the far side of it."""
    outdir, child = _child(tmp_path, install=_INSTALL_SIGNALS,
                           tail="time.sleep(30)")
    try:
        for _ in range(400):
            if (outdir / "queued").exists():
                break
            time.sleep(0.05)
        else:
            pytest.fail("the child never queued its captures")
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=5)
    assert _written(outdir) == _N, (
        f"a signalled session dropped {_N - _written(outdir)} of its last "
        f"captures:\n{child.stdout.read()}")


def test_a_wedged_writer_cannot_hold_a_finished_session(tmp_path, monkeypatch) -> None:
    """The wait is bounded. A writer stuck on a disk that never answers must
    cost the session a few seconds and its report, not the session."""
    stuck = threading.Event()
    monkeypatch.setattr(onair, "_save_capture", lambda *a, **kw: stuck.wait())
    writer = onair._CaptureWriter()
    try:
        for i in range(3):
            writer.put((tmp_path / f"rx_{i}.wav", np.zeros(4, np.float32), 0, None))
        t0 = time.monotonic()
        left = writer.drain(0.3)
        waited = time.monotonic() - t0
        assert left == 3 and 0.3 <= waited < 5.0, (left, waited)
        assert "3 still unwritten" in writer.report(), writer.report()
    finally:
        stuck.set()


def test_a_capture_that_cannot_be_written_is_not_lost_in_silence(
        tmp_path, monkeypatch, capsys) -> None:
    """`except Exception: pass` is right to refuse to stop a session over a
    recording and wrong to say nothing about it. The operator has to be able to
    tell a capture that failed from one that was never taken."""
    # `*facts` rather than the writer's parameter list: what is under test is
    # that a failed write is said out loud, and a stand-in that restates the
    # signature fails for the wrong reason every time a fact is added to it.
    def _no_room(path, seg, *facts):
        raise OSError("no space left on device")

    monkeypatch.setattr(onair, "_save_capture", _no_room)
    writer = onair._CaptureWriter()
    writer.put((tmp_path / "rx_07.wav", np.zeros(4, np.float32), 0, None))
    assert writer.drain(5.0) == 0
    assert "rx_07.wav" in capsys.readouterr().out
    report = writer.report()
    assert "1 LOST" in report and "rx_07.wav" in report, report


def test_the_drain_never_precedes_the_unkey() -> None:
    """Order, not presence. Getting the key down comes first, always -- a few
    hundred milliseconds of writing is fine after the transmitter is dropped and
    is not fine before it."""
    import inspect
    src = inspect.getsource(onair.run)
    assert src.index("rig.stop()") < src.index("_drain_captures("), (
        "the capture drain sits between the session ending and the rig being "
        "unkeyed -- nothing may delay dropping the transmitter")
