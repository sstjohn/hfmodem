# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A session must not be able to report carrier it never produced.

On 2026-08-10 `tools/onair.sh pactor` ran a 27-cycle session whose summary read
"keyed 23 of 27 cycles, 22.1 s of carrier" -- against `--serial`'s placeholder
default, `/dev/cu.usbserial-XXXXB0`, a path that exists on no machine. The
operator was at the radio and heard nothing, because there was nothing: every
key-down was a pipe write into a freshly spawned rigctl already dying on a
serial port it could not open, and a pipe write succeeds whether or not anything
reads it. Nothing between the flag and the ioctl ever asked whether the device
was there, and the carrier count was appended where PTT was ASKED FOR rather
than anywhere the asking could be checked.

Three seams close it, and each is tested here the way it would have been caught:

  * `ota.ptt_device` refuses a keying path that is not a character device,
    naming what it derived and from what (`core.ptt.require_char_device` -- a
    stat, never an open, because opening a keying line asserts RTS).
  * `ota.Rig.ptt(True)` refuses a device that is not there, and
    `ota.Rig.key_failure` reports, after the burst, a key-down whose taker is
    gone -- the pipe-write lie, checked at the only time it can be.
  * `ota.Rig.stop` falls to `core.ptt.drop_rts` -- one local ioctl, no daemon,
    no reply -- when the one-shot unkey cannot confirm the rig is down.

The decisive tests drive `onair.main()` whole, with `--transmit` armed, over
the arithmetic-clock bench from `test_grid` and the REAL `ota.Rig` talking to a
stand-in rigctl. No serial port, no audio device and no transmitter is touched.

A fourth seam joined them on 2026-08-12, and it is the same failure one layer
down: not a keying line that is not there, but the PROGRAM that drives it. See
the last section -- those tests run the tool and the launcher as processes,
because a stand-in rigctl handed to the object under test is exactly what let a
machine with no rigctl at all go unnoticed.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hfmodem.core import ptt as core_ptt
from hfmodem.core.occupied import FILTER_HZ
from hfmodem.core.ptt import PttError
from hfmodem.shrike import onair, ota

from .test_grid import _Bench

PLACEHOLDER = "/dev/cu.usbserial-XXXXB0"

# rigctl, reduced to what these tests need of it: it opens (or fails to open)
# the -r port, answers `f` with the dial and `m` with the mode and width the
# startup readback refuses to key without, logs what crossed the wire, and
# otherwise reads commands until EOF. The real one exits when its serial port
# does not exist; so does this, and that exit -- after the caller's pipe write
# already "succeeded" -- is the whole incident. Both answers are one-shots and
# both carry the process status with them: a port that will not open, or an
# `f`/`m` this stand-in does not answer, reaches the caller as a refusal.
FAKE_RIGCTL = r"""#!/bin/sh
port=
while [ $# -gt 0 ]; do
  case "$1" in
    -r) port=$2; shift 2 ;;
    -m|-s|-P|-p) shift 2 ;;
    *) break ;;
  esac
done
if [ ! -e "$port" ]; then
  echo "rig_open: error opening $port" >&2
  exit 2
fi
if [ $# -gt 0 ]; then
  printf '%s\n' "oneshot: $*" >> "${FAKE_RIGCTL_LOG:-/dev/null}"
  if [ "$1" = f ] && [ -z "$FAKE_RIGCTL_MUTE_F" ]; then echo 7100000; fi
  if [ "$1" = m ]; then printf 'PKTUSB\n3000\n'; fi
  [ -n "$FAKE_RIGCTL_ONESHOT_FAILS" ] && exit 2
  exit 0
fi
while read -r line; do
  printf '%s\n' "$line" >> "${FAKE_RIGCTL_LOG:-/dev/null}"
  if [ -n "$FAKE_RIGCTL_REFUSES" ]; then
    printf 'set_ptt: error = Communication timed out while getting current VFO\n'
  fi
  if [ -n "$FAKE_RIGCTL_DIE_ON_KEY" ] && [ "$line" = "T 1" ]; then
    exit 1
  fi
done
"""


@pytest.fixture
def fake_rigctl(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    exe = d / "rigctl"
    exe.write_text(FAKE_RIGCTL)
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_RIGCTL_LOG", str(tmp_path / "rigctl.log"))
    return d


def _rig(rigctl_dir: Path, *, serial: str = "/dev/null", ptt_type: str = "RTS",
         ptt_port: str | None = None) -> ota.Rig:
    return ota.Rig(1036, serial, 38400, ptt_type=ptt_type,
                   ptt_port=ptt_port or serial, rigctl_dir=str(rigctl_dir))


# -- fix 1: the derived PTT path must exist ---------------------------------

def test_a_derived_ptt_path_that_does_not_exist_is_refused() -> None:
    """The arm gate the incident never had. The refusal must name BOTH ends of
    the derivation, because the operator's next move is to fix --serial, and a
    message naming only the B1 path sends them looking for a flag they never
    passed."""
    with pytest.raises(PttError) as e:
        ota.ptt_device("ft891", PLACEHOLDER)
    msg = str(e.value)
    assert "/dev/cu.usbserial-XXXXB1" in msg, msg   # what it derived
    assert PLACEHOLDER in msg, msg                  # ...and from what


def test_a_derived_ptt_path_that_exists_is_returned() -> None:
    # /dev/null has no trailing interface digit, so the derivation is the
    # identity -- which makes it a character device the check must accept
    # without inventing hardware for the test to depend on.
    assert ota.ptt_device("ft891", "/dev/null") == "/dev/null"


def test_a_cat_keyed_rig_checks_the_one_port_it_has() -> None:
    """No second interface to derive does not mean nothing to check: for a rig
    that keys over CAT, the CAT port IS the keying path."""
    with pytest.raises(PttError) as e:
        ota.ptt_device("g90", PLACEHOLDER)
    assert PLACEHOLDER in str(e.value)


def test_a_trailing_digit_on_a_foreign_adapter_is_not_advanced(tmp_path) -> None:
    """/dev/ttyUSB0 -> /dev/ttyUSB1 is not a derivation, it is a DIFFERENT
    adapter -- quite possibly another radio's CAT port. `core.ptt` carries a
    deliberately narrow rule (`derive_ptt_port`) written to refuse exactly
    this, and a loose last-digit substitution here bypassed it. Both nodes
    exist and both stat as character devices, so only the refusal stands
    between the keying line and hardware nobody named."""
    (tmp_path / "ttyUSB0").symlink_to("/dev/null")
    (tmp_path / "ttyUSB1").symlink_to("/dev/null")
    with pytest.raises(PttError) as e:
        ota.ptt_device("ft891", str(tmp_path / "ttyUSB0"))
    assert "explicitly" in str(e.value), str(e.value)


# -- fix 2: the last-resort unkey -------------------------------------------

def test_stop_takes_the_line_down_itself_when_rigctl_cannot(
        fake_rigctl, monkeypatch) -> None:
    """The bridge's escape hatch, on this path too: when the one-shot unkey
    cannot confirm the rig is down, the keying line is dropped with our own
    ioctl -- no daemon, no socket, no reply to wait for. The ladder lives in
    `core.ptt.Keyer` now, so that is where the drop is intercepted."""
    monkeypatch.setenv("FAKE_RIGCTL_ONESHOT_FAILS", "1")
    dropped: list[str] = []
    monkeypatch.setattr(core_ptt, "drop_rts",
                        lambda dev, log: dropped.append(dev) or True)
    rig = _rig(fake_rigctl)
    rig.stop()
    assert dropped == ["/dev/null"]


def test_stop_trusts_a_oneshot_that_answered(fake_rigctl, monkeypatch) -> None:
    """...and only then. `drop_rts` opens the port, which is what makes it work
    and also what makes it unfit as a routine unkey."""
    dropped: list[str] = []
    monkeypatch.setattr(core_ptt, "drop_rts",
                        lambda dev, log: dropped.append(dev) or True)
    rig = _rig(fake_rigctl)
    rig.stop()
    assert dropped == []


def test_stop_has_no_line_to_drop_on_a_cat_keyed_rig(
        fake_rigctl, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_RIGCTL_ONESHOT_FAILS", "1")
    monkeypatch.setattr(core_ptt, "UNKEY_BUDGET_S", 0.2)   # the ladder, hurried
    dropped: list[str] = []
    monkeypatch.setattr(core_ptt, "drop_rts",
                        lambda dev, log: dropped.append(dev) or True)
    rig = _rig(fake_rigctl, ptt_type="RIG")
    rig.stop()
    assert dropped == []


def test_nothing_respawns_rigctl_after_stop(fake_rigctl) -> None:
    """The watchdog's `stop()` and the main thread's `finally` both mutate
    `_proc`, from two threads, and until now only timing kept them apart. After
    a stop, an unkey must not spawn a fresh rigctl: it would re-open the CAT
    port `stop` just released and block the one-shot unkey on it for up to
    8 s, with the rig possibly keyed."""
    rig = _rig(fake_rigctl)
    assert rig.ptt(True) is True
    rig.stop()
    assert rig.ptt(False) is False, "an unkey after stop has nothing to write into"
    assert rig._proc is None, "the unkey respawned rigctl after stop()"
    assert rig.ptt(True) is False, "a key-down after stop must be refused"
    assert rig._proc is None, "the refused key-down still spawned rigctl"


def test_a_dead_child_is_not_replaced_once_stop_has_begun(fake_rigctl) -> None:
    """The window the test above cannot see. `_cmd`'s gate reads *(child dead)
    and stop begun*, so a thread whose child is still alive walks past it; the
    watchdog's `stop()` then sets `_down` and kills the child, and that
    thread's `_open()` meets a dead process with nothing left to stop it
    spawning a fresh rigctl onto the CAT port `stop` just released. The
    refusal has to live where the spawn does."""
    rig = _rig(fake_rigctl)
    assert rig.ptt(True) is True                 # a live rigctl is standing
    proc = rig._proc
    # The interleave, made flesh: stop() lands between the gate and _open.
    rig._down.set()
    proc.kill()
    proc.wait()
    assert rig._open() is None, (
        "a fresh rigctl was spawned onto the CAT port stop() released")
    assert rig._proc is proc, "the dead child was replaced behind stop()'s back"
    assert rig._cmd("T", "0") is False


def test_a_stop_landing_inside_the_spawn_leaves_no_stray_rigctl(
        fake_rigctl, monkeypatch) -> None:
    """The window the gate above narrowed but could not close: check-and-spawn
    is not atomic, and `stop()` deliberately takes no lock. A thread passes the
    `_down` check and enters `Popen`; the watchdog's `stop()` sets `_down`,
    closes the old child and runs its one-shot; the `Popen` completes and hands
    back a fresh rigctl nothing will ever kill — holding the CAT port every
    later one-shot blocks on for up to 8 s, with the ladder ending unconfirmed
    on a CAT-PTT rig. The spawn must re-check `_down` once `Popen` returns and
    kill what it made."""
    import threading
    rig = _rig(fake_rigctl)
    entered, resume = threading.Event(), threading.Event()
    spawned: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def slow_popen(*a, **kw):
        persistent = kw.get("stdin") is subprocess.PIPE
        if persistent:                    # the one-shots pass no stdin pipe
            entered.set()
            assert resume.wait(5.0), "stop() never released the spawn"
        p = real_popen(*a, **kw)
        if persistent:
            spawned.append(p)
        return p

    monkeypatch.setattr(ota.subprocess, "Popen", slow_popen)
    t = threading.Thread(target=rig.ptt, args=(True,))
    t.start()
    try:
        assert entered.wait(5.0), "the key-down never reached the spawn"
        rig.stop()                        # the watchdog's ladder, mid-spawn
    finally:
        resume.set()
        t.join(timeout=5.0)
    assert not t.is_alive()

    deadline = time.monotonic() + 5.0
    while any(p.poll() is None for p in spawned) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert spawned and all(p.poll() is not None for p in spawned), (
        "a rigctl spawned while stop() ran is still alive, holding the CAT "
        "port against the one-shot unkey")
    assert rig._proc is None or rig._proc.poll() is not None, (
        "the fresh child was installed behind stop()'s back")


def test_a_close_racing_a_put_is_a_refusal_not_a_traceback(fake_rigctl) -> None:
    """`_put` read `self._proc.stdin` after `_open` had returned it, and a
    concurrent `_close` nulling `_proc` in between raised AttributeError —
    which `except OSError` does not catch, so the session died with a
    traceback instead of a summary. The `finally` still unkeyed; the record
    did not survive. Sequential here, but the same interleave."""
    rig = _rig(fake_rigctl)
    assert rig.ptt(True) is True
    rig._close()                          # the concurrent teardown, made flesh
    assert rig._proc is None
    assert rig._put("T 0\n") is False, (
        "a put with no child behind it must answer False, not raise")
    rig.stop()


# -- the retune must leave the operator's filter alone -----------------------

def test_set_mode_never_asks_for_the_rigs_default_width(
        fake_rigctl, tmp_path) -> None:
    """``M <mode> 0`` does not mean "mode only". hamlib's third argument is the
    passband and 0 selects the rig's DEFAULT width for that mode, which on this
    FT-891 in PKTUSB is 1700 Hz — so every retune quietly narrowed the operator's
    filter to 1700, and the front panel read exactly that on 2026-08-15 while two
    callers believed they had asked for 3000 and 2700. The width asked for is
    `core.occupied.FILTER_HZ`, the one every path here asks for and the one the
    `core.busy` thresholds were calibrated through.
    """
    rig = _rig(fake_rigctl)
    assert rig.set_mode("PKTUSB") is True
    assert rig.set_mode("PKTUSB", 2400) is True
    log = tmp_path / "rigctl.log"
    deadline, lines = time.monotonic() + 5.0, []
    while time.monotonic() < deadline:
        lines = log.read_text().splitlines() if log.exists() else []
        if len(lines) >= 2:
            break
        time.sleep(0.02)
    rig.stop()
    assert lines[:2] == [f"M PKTUSB {FILTER_HZ}", "M PKTUSB 2400"], lines


# -- fix 3: keying that cannot be confirmed is not keying --------------------

def test_a_key_down_refuses_a_device_that_is_not_there(fake_rigctl) -> None:
    rig = _rig(fake_rigctl, serial=PLACEHOLDER, ptt_port=PLACEHOLDER)
    with pytest.raises(PttError) as e:
        rig.ptt(True)
    assert PLACEHOLDER in str(e.value)
    # The unkey checks nothing: it must stay possible in every state the
    # channel can be in, including this one.
    rig.ptt(False)
    # That unkey spawns rigctl, which exits at once against a port no machine
    # has, and `stop()` is the only path that reaps it. Without this the last
    # line of the test left a defunct child for however long it took the
    # collector to notice -- 21 tests, on the run that found it.
    rig.stop()


def test_a_key_that_landed_in_a_dead_pipe_is_reported(
        fake_rigctl, monkeypatch) -> None:
    """The incident's exact lie, at the seam it lives on: the write that keys
    succeeds into a child that is already dying, so the only honest answer is
    an after-the-burst one."""
    monkeypatch.setenv("FAKE_RIGCTL_DIE_ON_KEY", "1")
    rig = _rig(fake_rigctl)
    assert rig.ptt(True) is True          # the pipe write itself succeeds
    deadline = time.monotonic() + 5.0
    why = rig.key_failure()
    while why is None and time.monotonic() < deadline:
        time.sleep(0.02)
        why = rig.key_failure()
    assert why is not None and "exited" in why, why
    rig.stop()


def test_a_standing_channel_reports_no_failure(fake_rigctl) -> None:
    rig = _rig(fake_rigctl)
    assert rig.ptt(True) is True
    time.sleep(0.1)
    assert rig.key_failure() is None
    rig.ptt(False)
    rig.stop()


def test_a_rigctl_that_refuses_in_band_is_seen(fake_rigctl, monkeypatch) -> None:
    """Radio off, CAT cable out, USB adapter still enumerated: rigctl stays
    alive and answers `set_ptt: error = ...` to every command. Until now nothing
    read those answers -- stdout was pumped to nowhere and stderr to DEVNULL --
    so this passed all four of `key_failure`'s checks against a radio that
    never keyed, which is the 2026-08-10 false record one failure mode over."""
    monkeypatch.setenv("FAKE_RIGCTL_REFUSES", "1")
    rig = _rig(fake_rigctl)
    assert rig.ptt(True) is True          # the pipe write itself still succeeds
    deadline = time.monotonic() + 5.0
    why = rig.key_failure()
    while why is None and time.monotonic() < deadline:
        time.sleep(0.02)
        why = rig.key_failure()
    assert why is not None and "error" in why, why
    rig.stop()


def test_a_dial_that_cannot_be_read_back_is_not_reported_as_read(
        fake_rigctl, tmp_path, monkeypatch) -> None:
    """`f` answering nothing used to become `int("" or dial)` -- the INTENDED
    dial printed as though the rig had confirmed it, on the same CAT channel
    the session was about to trust for every key-down."""
    monkeypatch.setenv("FAKE_RIGCTL_MUTE_F", "1")
    real_rig = ota.Rig
    monkeypatch.setattr(ota, "Rig", lambda *a, **kw: real_rig(
        *a, rigctl_dir=str(fake_rigctl), **kw))
    monkeypatch.setattr(ota.session, "build_session", lambda *a, **kw: [0.0] * 480)
    monkeypatch.setattr(ota.session, "write_wav", lambda *a, **kw: None)
    monkeypatch.setattr(ota, "find_device", lambda name, kind: 0)
    played: list = []
    monkeypatch.setattr(ota, "_play",
                        lambda *a, **kw: played.append(a) or (0.01, 0.0))
    args = argparse.Namespace(
        outdir=str(tmp_path / "out"), dxcall="K7ABC", dial=7100000, center=None,
        band="40m", data=False, message="", sl=2, mycall="W9SSJ", rig="ft891",
        dry_run=False, audio_out=None, audio_in=None, serial="/dev/null",
        baud=0, attempts=1, listen=0.1, gap=0.0, max_key=40.0)
    with pytest.raises(SystemExit) as e:
        ota.run(args)
    assert "NOT KEYING" in str(e.value), str(e.value)
    assert not played, "audio went out on a CAT channel that answers nothing"


def test_run_claims_no_channel_check_it_does_not_make() -> None:
    """`run` printed "Listening 3s for a busy channel before TX..." above a bare
    sleep. Whether to transmit on a shared band is the operator's judgement,
    and no occupancy gate runs here: the tool may pause and say it is pausing,
    but must not claim a check that does not exist."""
    src = inspect.getsource(ota.run)
    assert "busy" not in src.lower(), "run() claims an occupancy check it does not make"


# -- the incident itself -----------------------------------------------------

def _run_onair(monkeypatch, tmp_path, fake_rigctl, serial: str, *,
               unplug_during: int = 0):
    """One whole `onair.main()` with --transmit armed and the REAL `ota.Rig`.

    Only the seams that would touch hardware are faked: the duplex stream is
    `test_grid`'s bench (an arithmetic clock, never a sound card), the device
    table answers 0, and rigctl is the stand-in above. `unplug_during` removes
    the serial device -- a symlink, standing in for the /dev node macOS drops
    on unplug -- while that burst is on the air, which is the incident's
    silence beginning mid-session instead of before it.
    """
    bursts = [0]

    def peer(bench, end):
        bursts[0] += 1
        if unplug_during and bursts[0] == unplug_during:
            Path(serial).unlink()

    real_rig = ota.Rig
    monkeypatch.setattr(ota, "Rig",
                        lambda *a, **kw: real_rig(
                            *a, rigctl_dir=str(fake_rigctl), **kw))
    # The vanished-device case ends in `stop()`'s ladder retrying a one-shot
    # that can never answer; the budget is real time, so it is shortened here.
    monkeypatch.setattr(core_ptt, "UNKEY_BUDGET_S", 0.2)
    monkeypatch.setattr(onair, "_LiveInput",
                        lambda *a, **kw: _Bench(peer=peer))
    monkeypatch.setattr(onair, "find_device",
                        lambda name, kind, required=False: 0)
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv",
                        ["onair", "--transmit", "--mycall", "W9SSJ",
                         "--dxcall", "K7ABC", "--dial", "7100000",
                         "--serial", serial, "--max-cycles", "4",
                         "--outdir", str(tmp_path / "cap")])
    log, code, exc = io.StringIO(), None, None
    try:
        with contextlib.redirect_stdout(log):
            code = onair.main()
    except (SystemExit, PttError) as e:
        exc = e
    return code, log.getvalue(), exc


def test_a_session_against_a_device_that_does_not_exist_cannot_report_carrier(
        monkeypatch, tmp_path, fake_rigctl) -> None:
    """THE INCIDENT. `--serial`'s placeholder default, a full session armed for
    transmit, and rigctl dying on every spawn exactly as the real one does.
    Today's code ran this to completion and summarised carrier; it must not be
    able to."""
    code, log, exc = _run_onair(monkeypatch, tmp_path, fake_rigctl, PLACEHOLDER)
    claimed = re.search(r"keyed [1-9]\d* of \d+ cycles.*carrier", log)
    assert claimed is None, (
        f"the summary claims carrier that never existed: {claimed.group(0)!r}")
    assert code != 0, "a session that never touched the radio reported success"
    # ...and the refusal is the arm gate's, naming the derivation.
    assert exc is not None and "usbserial-XXXXB1" in str(exc) \
        and PLACEHOLDER in str(exc), f"exc={exc!r}"


def test_a_device_that_vanishes_mid_session_is_a_truthful_summary(
        monkeypatch, tmp_path, fake_rigctl) -> None:
    """A session that refuses to start is not sufficient: the same silence can
    begin mid-session when the device is unplugged. Two bursts go out on a
    real keying channel, the device node vanishes under the third, and the
    summary must say exactly that -- the two confirmed bursts as carrier, the
    unconfirmable third refused, the failure as the reason the session ended,
    and a nonzero exit."""
    cat = tmp_path / "cat"
    cat.symlink_to("/dev/null")
    code, log, exc = _run_onair(monkeypatch, tmp_path, fake_rigctl, str(cat),
                                unplug_during=3)
    assert "KEYING FAILED" in log, log[-2000:]
    assert str(cat) in log, log[-2000:]
    assert re.search(r"keyed 2 of \d+ cycles", log), (
        "the summary must count exactly the two bursts that had a keying "
        f"channel behind them:\n{log[-2000:]}")
    assert code == 1 and exc is None, (code, exc)


# -- and the tool must find the tool that keys -------------------------------
#
# Every test above hands `ota.Rig` an explicit `rigctl_dir`, or fakes the
# subprocess outright, so none of them could ever ask the question the operator's
# machine asks: is `rigctl` where this will look for it? On 2026-08-12 the answer
# became no. A home-directory default was removed from `ota.Rig` -- rightly, it
# was one station's -- and the resolution became `shutil.which("rigctl") or
# "rigctl"`, whose second half hands `Popen` a bare name on any machine whose
# Hamlib is a private build. That is a FileNotFoundError raised out of a keying
# path with the audio device already open, at the radio, in the middle of a
# scheduled session.
#
# So these two run PROCESSES, not objects: the tool as the operator starts it,
# and the launcher that starts it. Nothing here opens a serial port, an audio
# device or a transmitter -- the first refuses before the rig object exists, and
# the second never gets past `command -v`.

_TOOLS = Path(__file__).resolve().parents[5] / "tools"
_PORTS_LIB = _TOOLS / "lib" / "ports.sh"

requires_launcher = pytest.mark.skipif(
    not _PORTS_LIB.exists(),
    reason=f"{_PORTS_LIB} is not present (installed-wheel run)")


def test_a_path_without_hamlib_refuses_instead_of_raising(tmp_path) -> None:
    """The transmit path, started exactly as the launcher starts it, on a PATH
    that carries no Hamlib. What must come back is a refusal an operator can act
    on -- what is missing, where it was looked for, how to supply it -- and not a
    traceback out of `subprocess`.

    `/dev/null` is the keying line: the arm gate stats it and never opens it, so
    the run gets as far as building the rig and no further.
    """
    empty = tmp_path / "bin"
    empty.mkdir()
    r = subprocess.run(
        [sys.executable, "-m", "hfmodem.shrike.onair", "--transmit",
         "--mycall", "W9SSJ", "--dxcall", "K7ABC", "--dial", "7100000",
         "--serial", "/dev/null", "--max-cycles", "1",
         "--outdir", str(tmp_path / "cap")],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PATH": f"{empty}:/usr/bin:/bin"})
    said = r.stdout + r.stderr
    assert r.returncode != 0, said
    assert "Traceback" not in said and "FileNotFoundError" not in said, said
    assert "NOT KEYING" in said and "rigctl" in said, said
    assert str(empty) in said, f"the refusal does not say where it looked:\n{said}"
    assert "HAMLIB_BIN" in said, f"the refusal does not say how to supply it:\n{said}"


@requires_launcher
def test_the_launcher_puts_hamlib_on_path_for_the_children_it_starts(tmp_path) -> None:
    """The other half, and the reason the fix is not a flag on each consumer:
    shrike, besra and sabir all resolve `rigctl` by name, so the launcher's
    shared environment answers all three at once. A build that is on no machine's
    PATH goes on this one's, for everything started from `tools/`.
    """
    ham = tmp_path / "hamlib" / "bin"
    ham.mkdir(parents=True)
    for name in ("rigctl", "rigctld"):
        exe = ham / name
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
    empty = tmp_path / "bin"
    empty.mkdir()
    # The child is asked the question `ota.Rig` asks, through the same call, so
    # this pins the launcher against the resolution rather than against PATH's
    # spelling.
    r = subprocess.run(
        ["/bin/bash", "-c", f'set -euo pipefail; source "{_PORTS_LIB}"; '
                            f'"{sys.executable}" -c '
                            "'import shutil; print(shutil.which(\"rigctl\"))'"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": f"{empty}:/usr/bin:/bin",
             "HAMLIB_BIN": str(ham)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(ham / "rigctl"), (r.stdout, r.stderr)


@requires_launcher
def test_the_launcher_refuses_a_machine_with_no_hamlib_at_all(tmp_path) -> None:
    """And when there is none to put there, the refusal is the launcher's own,
    before a modem is started -- not a Python traceback out of whichever tool
    reached for it first."""
    empty = tmp_path / "bin"
    empty.mkdir()
    r = subprocess.run(
        ["/bin/bash", "-c", f'source "{_PORTS_LIB}"; require_hamlib'],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": f"{empty}:/usr/bin:/bin",
             "HAMLIB_BIN": str(tmp_path / "nope")})
    said = r.stdout + r.stderr
    assert r.returncode != 0, said
    assert "hamlib is not on PATH" in said and "HAMLIB_BIN" in said, said


def test_a_rig_whose_unkey_ladder_ran_out_refuses_to_key_again():
    """The ladder's last rung, which shrike alone did not reach: `core.ptt`
    documents retiring the rig as the step after the alarm, and besra and sabir
    both poll `must_retire`, but `ota.Rig` held no such state — so an unkey that
    failed at every level left the rig keyable. Unkeys stay allowed regardless:
    they must be possible in every state the channel can be in."""
    import threading
    import types

    rig = ota.Rig.__new__(ota.Rig)
    rig._down, rig.retired, rig._sick = threading.Event(), False, False
    calls: list[tuple] = []
    rig._cmd = lambda *a: (calls.append(a), True)[1]
    rig._devices_present = lambda: None
    rig._errs = types.SimpleNamespace(clear=lambda: None)

    rig.retire()
    assert rig.retired
    assert rig.ptt(True) is False, "a retired rig keyed"
    assert not calls, "a retired rig reached the transport"
    rig.ptt(False)
    assert calls == [("T", "0")], "a retired rig refused an unkey"
