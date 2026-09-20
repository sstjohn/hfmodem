# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A terminated session must unwind, because that is where the unkey lives.

On 2026-08-04 a session holding the key was stopped from outside and kept
transmitting on 10.1 MHz until the operator powered the rig down. Nothing in the
modem was wrong: every unkey in `onair` hangs off a `finally`, and a `finally`
runs when Python unwinds, not when the process is killed by a signal. SIGTERM is
what `timeout(1)` sends, and what a supervisor or a cancelled tool call sends, so
the ordinary ways an unattended run is stopped were exactly the ways that left it
keyed.

The check runs a child that holds a flag across a `try`/`finally`, signals it,
and reads the flag back. The flag stands in for the key line: the modem's own
`finally` chain is what the fix restores, and this asserts the property that
chain depends on. Without `_unkey_on_signal` the child dies at the `try` and the
flag stays KEYED -- that arm is asserted too, because a guard that cannot fail
is the shape this repo has been caught by three times.
"""

from __future__ import annotations

import pathlib
import signal
import subprocess
import sys
import tempfile
import textwrap

import pytest

_HOLDS_THE_KEY = """\
import pathlib, signal, sys, time
# Take the launcher's signal mask out of the measurement. `nohup` sets SIGHUP to
# SIG_IGN and a child inherits it, so without this the unguarded arm measures how
# the SUITE was started rather than what a signal does -- it passes from a
# terminal and fails under nohup, which is exactly how it slipped through once.
signal.signal(signal.SIGHUP, signal.SIG_DFL)
{install}
mark = pathlib.Path({flag!r})
# Taking the key INSIDE the try: the parent signals the moment the flag file
# appears, and a signal landing between the write and the try would kill the
# guarded child before any finally protects it -- a flake that reads exactly
# like the guard failing, seen under suite load.
try:
    mark.write_text("KEYED")
    time.sleep(30)
finally:
    mark.write_text("UNKEYED")
"""

# No leading indent on the continuations: this is spliced in at column zero, and
# a stray four spaces makes the child an IndentationError that dies before it
# ever takes the key -- which looks exactly like the guard failing.
_INSTALL = ("sys.path.insert(0, {root!r})\n"
            "from hfmodem.shrike.onair import _unkey_on_signal\n"
            "_unkey_on_signal()")


def _run_and_signal(*, guarded: bool, sig: int) -> str:
    root = str(pathlib.Path(__file__).resolve().parents[3])   # .../packages/hfmodem
    with tempfile.TemporaryDirectory() as tmp:
        flag = str(pathlib.Path(tmp) / "key")
        src = pathlib.Path(tmp) / "holds_the_key.py"
        src.write_text(textwrap.dedent(_HOLDS_THE_KEY).format(
            install=_INSTALL.format(root=root) if guarded else "", flag=flag))
        child = subprocess.Popen([sys.executable, str(src)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(200):                      # wait until it holds the key
                # Content, not existence: write_text creates the file before the
                # bytes land, and signalling into that window reads back "".
                if (pathlib.Path(flag).exists()
                        and pathlib.Path(flag).read_text() == "KEYED"):
                    break
                import time as _t; _t.sleep(0.05)
            else:
                pytest.fail("the child never took the key")
            child.send_signal(sig)
            child.wait(timeout=15)
            return pathlib.Path(flag).read_text()
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=5)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP],
                         ids=["SIGTERM", "SIGHUP"])
def test_a_signalled_session_unkeys(sig: int) -> None:
    assert _run_and_signal(guarded=True, sig=sig) == "UNKEYED"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP],
                         ids=["SIGTERM", "SIGHUP"])
def test_without_the_guard_it_stays_keyed(sig: int) -> None:
    """The counterexample. If this ever reads UNKEYED the check above is free."""
    assert _run_and_signal(guarded=False, sig=sig) == "KEYED"
