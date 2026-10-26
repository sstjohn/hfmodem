# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The signal path, in a child process because it ends in `os._exit`.

`Rig.panic()` arms a deadman and then exits, which would take the test session
with it — so the child is the subject and the parent watches. kestrel's own unkey
tests do the same thing for the same reason; this is that pattern applied to the
shared rig.

What is being asserted is the thing an operator is entitled to: reach for Ctrl-C
and get a dead transmitter and a dead process in about a second, **whatever
rigctld is doing** — and that is exactly the moment rigctld is least likely to
answer. So the interesting case is not the tidy one; it is the wedged daemon.
"""
from __future__ import annotations

import select
import subprocess
import sys
import textwrap
import time

import pytest

from hfmodem.core.regulatory import Control, Unregulated
from hfmodem.core.rig import Cat, Rig, StuckTransmitter
from hfmodem.tests.core.fakerig import FakePtt, FakeRigctld

CHILD = textwrap.dedent("""
    import sys, time
    from hfmodem.core.regulatory import Control, Unregulated, centred
    from hfmodem.core.rig import Cat, Rig
    from hfmodem.tests.core.fakerig import FakePtt

    port = int(sys.argv[1])
    class MarkingPtt(FakePtt):
        def release(self):
            super().release()
            assert self.line is False and self.released_low is True
            print("RELEASED LOW", flush=True)

    ptt = MarkingPtt()
    r = Rig(model="ft891", cat=Cat("127.0.0.1", port), ptt=ptt,
            profile=Unregulated(because="child process test"),
            control=Control.LOCAL, mycall="N0CALL", transmit=True)
    r.arm(prove_ptt=False)
    r.key(centred(7_100_000, 500.0), why="child", duration_s=30.0)
    print("KEYED", flush=True)
    assert sys.stdin.readline() == "panic\\n"
    r.panic()
    print("UNREACHABLE", flush=True)
""")


def run_child(fake, timeout: float = 8.0, *, wedge: bool = False) -> tuple[int, float, str]:
    with subprocess.Popen([sys.executable, "-c", CHILD, str(fake.port)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as p:
        try:
            assert select.select([p.stdout], [], [], timeout)[0], "child never became ready"
            ready = p.stdout.readline()
            if ready != "KEYED\n":
                out, err = p.communicate(timeout=timeout)
                pytest.fail(f"child never keyed:\n{ready}{out}{err}")
            # Silence only the panic path, after real key() admission succeeded.
            fake.wedge = wedge
            t0 = time.monotonic()
            out, err = p.communicate("panic\n", timeout=timeout)
            return p.returncode, time.monotonic() - t0, ready + out + err
        finally:
            if p.poll() is None:
                p.kill()
            p.wait()


@pytest.mark.realtime
def test_panic_exits_promptly_and_unkeys_through_cat():
    fake = FakeRigctld()
    try:
        rc, elapsed, out = run_child(fake)
        assert "KEYED" in out, out
        assert "UNREACHABLE" not in out, "panic() returned; it must not"
        assert elapsed < 6.0, f"panic took {elapsed:.1f} s"
        assert rc == 0, out
        assert "+T 0" in fake.log, "the transmitter was never told to stop"
        # The emergency connection must disable caching before its readback.
        start = fake.log.index("+T 0")
        assert fake.log[start - 1:start + 2] == ["+\\set_cache 0", "+T 0", "+t"]
    finally:
        fake.close()


@pytest.mark.realtime
def test_panic_exits_even_when_rigctld_never_answers():
    """The case that matters. A wedged daemon accepts the connection and replies
    to nothing, so every CAT path blocks — and the process must still die, because
    the deadman is armed before the unkey is attempted rather than after."""
    fake = FakeRigctld()
    try:
        rc, elapsed, out = run_child(fake, timeout=10.0, wedge=True)
        assert "KEYED" in out, out
        assert "unkeying" in out, out
        assert "RELEASED LOW" in out, out
        assert fake.log.count("+\\set_cache 0") > 1, "the emergency CAT path was never attempted"
        assert elapsed < 8.0, f"a wedged rigctld held the process for {elapsed:.1f} s"
        assert "UNREACHABLE" not in out
        # A silent CAT path cannot erase the independent line's confirmation.
        assert rc == 0 and "confirmed down" in out, out
    except subprocess.TimeoutExpired:
        pytest.fail("panic() did not terminate against a wedged rigctld — the "
                    "deadman is not arming before the unkey")
    finally:
        fake.close()


def test_the_line_is_released_before_the_process_goes():
    """Ordering: the keying line comes down first, and the exit is what happens
    after. A child that exits before releasing would leave RTS to the driver's
    hangup behaviour, which is the one kill path still unproven."""
    fake = FakeRigctld()
    try:
        rc, elapsed, out = run_child(fake)
        assert "unkeying" in out, out          # panic_unkey writes this to fd 2
        assert "RELEASED LOW" in out, "process exited before confirmed line release"
        assert out.index("KEYED") < out.index("RELEASED LOW")
        assert rc == 0 and "UNREACHABLE" not in out, out
        assert "+T 0" in fake.log
    finally:
        fake.close()


def _panic_rig(daemon, line):
    r = Rig(model="ft891", cat=Cat("127.0.0.1", daemon.port), ptt=line,
            profile=Unregulated(because="panic test"), control=Control.LOCAL,
            mycall="N0CALL", transmit=True)
    r.cat.open()
    r._armed = True
    r._dial_hz = daemon.freq
    return r


def test_the_keying_line_confirms_the_unkey_when_the_daemon_cannot():
    """The loudest message in the program must not fire on every shutdown.

    `_check_ptt_owner` refuses to arm unless rigctld owns no PTT, and a daemon
    owning no PTT answers `t` with ENAVAIL. So on the only configuration this
    station accepts, the CAT leg could never confirm anything — and every
    watchdog and every Ctrl-C ended in "CHECK THE RADIO AND REMOVE POWER", which
    is how an operator learns to stop reading it.

    The keying line is the better witness in any case: it is what keys this
    transmitter, established by asserting each line in turn and watching the
    receiver go deaf on RTS and not on DTR.
    """
    daemon = FakeRigctld(ptt_type="None")
    line = FakePtt()
    rig = _panic_rig(daemon, line)
    try:
        line.assert_(True)
        rig.panic_unkey("watchdog")          # must not raise
        assert line.line is False, "the line was not brought down"
        assert rig.retired
    finally:
        rig.close(); daemon.close()


def test_a_line_that_will_not_read_back_still_raises():
    """The scream is kept for the case that earns it: a line whose state cannot
    be read, on a daemon that cannot answer either, is precisely when nobody
    knows whether the transmitter is down."""
    daemon = FakeRigctld(ptt_type="None")
    line = FakePtt(unreadable=True)
    rig = _panic_rig(daemon, line)
    try:
        line.assert_(True)
        with pytest.raises(StuckTransmitter):
            rig.panic_unkey("watchdog")
    finally:
        rig.close(); daemon.close()


def test_a_stuck_line_is_never_reported_as_confirmed():
    """A line that reads high after release is the failure this whole path
    exists for, and the readback must not launder it into a pass."""
    daemon = FakeRigctld(ptt_type="None")
    line = FakePtt(stick=True)
    rig = _panic_rig(daemon, line)
    try:
        line.assert_(True)
        with pytest.raises(StuckTransmitter):
            rig.panic_unkey("watchdog")
        assert line.line is True, "the fake did not model a stuck line"
    finally:
        rig.close(); daemon.close()
