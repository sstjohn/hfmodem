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

P1 preserves those safety properties through a held keying line: arm reads
low before keying, each drive requires readback, and stop unkeys and reads the
line low before handing it back. CAT remains a separate tune-only child.

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
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hfmodem.core import native
from hfmodem.core import ptt as core_ptt
from hfmodem.core.occupied import FILTER_HZ
from hfmodem.core.ptt import PttError
from hfmodem.shrike import onair, ota
from hfmodem.tests import fakeclaim

from .test_grid import _Bench
from .test_p1_line_keyer import FakeLine

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


@pytest.fixture(autouse=True)
def cat_children(monkeypatch):
    """Every rigctl the rig starts, as (long_lived, child): admitted, and run as
    plain subprocesses of this one."""
    children = []

    def spawn(argv, **kw):
        child = fakeclaim.spawn(argv, **kw)
        children.append((kw.get("long_lived", False), child))
        return child

    fakeclaim.install(monkeypatch, spawner=spawn)
    yield children
    for _, child in children:
        if child.alive():
            child.kill()


@pytest.fixture(autouse=True)
def fake_lines(monkeypatch):
    lines = []

    def make(device):
        core_ptt.require_char_device(device)
        line = FakeLine(device)
        lines.append(line)
        return line

    monkeypatch.setattr(core_ptt, "RtsPtt", make)
    yield lines
    for line in lines:
        line.release()


def _rig(rigctl_dir: Path, *, serial: str = "/dev/null", ptt_type: str = "LINE",
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


def test_a_profile_without_a_line_never_uses_cat_as_ptt() -> None:
    assert ota.ptt_device("g90", PLACEHOLDER) is None
    assert ota.ptt_device("g90", "/dev/null") is None


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
        fake_rigctl, monkeypatch, fake_lines) -> None:
    """A dead CAT transport cannot prevent an independently confirmed unkey."""
    monkeypatch.setenv("FAKE_RIGCTL_ONESHOT_FAILS", "1")
    rig = _rig(fake_rigctl)
    assert rig.ptt(True)
    line = fake_lines[-1]
    rig.stop()
    assert line.calls[-4:] == ["sense", False, "sense", "release"]
    assert line.released_low is True and line.state is False
    assert rig.ptt(True) is False


def test_stop_trusts_only_line_readback(fake_rigctl, monkeypatch, fake_lines):
    """A successful drive without low readback must alarm and retire."""
    rig = _rig(fake_rigctl)
    assert rig.ptt(True)
    line = fake_lines[-1]
    line.readable = False
    rig.stop()
    assert line.state is False  # the write succeeded; confirmation did not
    assert line.released_low is False
    assert rig.retired and rig.key_failure() is not None
    assert rig.ptt(True) is False


def test_stop_has_no_line_to_drop_on_a_receive_only_rig(fake_rigctl, fake_lines):
    rig = ota.Rig(1036, "/dev/null", 38400, rigctl_dir=str(fake_rigctl))
    rig.stop()
    assert fake_lines == []
    assert rig.ptt(True) is False
    assert rig.key_failure() == ota.NO_KEYING_LINE


def test_nothing_respawns_rigctl_after_stop(fake_rigctl) -> None:
    """The watchdog's `stop()` and the main thread's `finally` both mutate
    `_proc`, from two threads, and until now only timing kept them apart. After
    a stop, an unkey must not spawn a fresh rigctl: it would re-open the CAT
    port `stop` just released. P1 keys separately, but CAT ownership must
    still end at stop."""
    rig = _rig(fake_rigctl)
    assert rig.ptt(True) is True
    rig.stop()
    assert rig.ptt(False) is True  # stop already confirmed low and released
    assert rig._proc is None, "the unkey respawned rigctl after stop()"
    assert rig.ptt(True) is False, "a key-down after stop must be refused"
    assert rig._proc is None, "the refused key-down still spawned rigctl"


def test_a_dead_child_is_not_replaced_once_stop_has_begun(fake_rigctl, cat_children) -> None:
    """The window the test above cannot see. `_cmd`'s gate reads *(child dead)
    and stop begun*, so a thread whose child is still alive walks past it; the
    watchdog's `stop()` then sets `_down` and kills the child, and that
    thread's `_open()` meets a dead process with nothing left to stop it
    spawning a fresh rigctl onto the CAT port `stop` just released. The
    refusal has to live where the spawn does."""
    rig = _rig(fake_rigctl)
    assert rig.set_freq(7100000) is True          # a live CAT child is standing
    proc = rig._proc
    # The interleave, made flesh: stop() lands between the gate and _open.
    rig._down.set()
    proc.kill()
    assert proc.wait_gone(5)
    spawned = len(cat_children)
    assert rig._open() is None, (
        "a fresh rigctl was spawned onto the CAT port stop() released")
    assert len(cat_children) == spawned and rig._proc in (None, proc), (
        "the dead child was replaced behind stop()'s back")
    assert rig.set_freq(7100000) is False
    rig.stop()


def test_a_stop_landing_inside_the_spawn_leaves_no_stray_rigctl(
        fake_rigctl, monkeypatch) -> None:
    """The window the gate above narrowed but could not close: check-and-spawn
    is not atomic, and `stop()` deliberately takes no lock. A thread passes the
    `_down` check and enters the spawn; the watchdog's `stop()` sets `_down`,
    closes the old child and releases the line; the spawn completes and hands
    back a fresh rigctl nothing will ever kill — holding the CAT port every
    later CAT readback needs. The spawn must re-check `_down` once it returns and
    kill what it made."""
    import threading
    rig = _rig(fake_rigctl)
    entered, resume = threading.Event(), threading.Event()
    spawned = []
    real_spawn = native.spawn_native

    def slow_spawn(argv, **kw):
        persistent = kw.get("long_lived", False)
        if persistent:
            entered.set()
            assert resume.wait(5.0), "stop() never released the spawn"
        child = real_spawn(argv, **kw)
        if persistent:
            spawned.append(child)
        return child

    monkeypatch.setattr(native, "spawn_native", slow_spawn)
    t = threading.Thread(target=rig.set_freq, args=(7100000,))
    t.start()
    try:
        assert entered.wait(5.0), "the tune never reached the spawn"
        rig.stop()                        # the watchdog's ladder, mid-spawn
    finally:
        resume.set()
        t.join(timeout=5.0)
    assert not t.is_alive()

    deadline = time.monotonic() + 5.0
    while any(c.alive() for c in spawned) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert spawned and not any(c.alive() for c in spawned), (
        "a rigctl spawned while stop() ran is still alive, holding the CAT "
        "port against later CAT access")
    assert rig._proc is None or not rig._proc.alive(), (
        "the fresh child was installed behind stop()'s back")


def test_a_close_racing_a_put_is_a_refusal_not_a_traceback(fake_rigctl) -> None:
    """`_put` read `self._proc.stdin` after `_open` had returned it, and a
    concurrent `_close` nulling `_proc` in between raised AttributeError —
    which `except OSError` does not catch, so the session died with a
    traceback instead of a summary. The `finally` still unkeyed; the record
    did not survive. Sequential here, but the same interleave."""
    rig = _rig(fake_rigctl)
    assert rig.set_freq(7100000) is True
    rig._close()                          # the concurrent teardown, made flesh
    assert rig._proc is None
    assert rig._put("F 7100000\n") is False, (
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


@pytest.mark.parametrize("rig_name, passband", [("ft891", FILTER_HZ), ("ftdx10", -1)])
def test_run_leaves_a_profile_width_to_tune(
        rig_name, passband, fake_rigctl, tmp_path, monkeypatch) -> None:
    """2026-10-02, FTDX10: `tune` set DATA width 4000 Hz and the 12 kHz roof by
    raw CAT and read both back, then the arm's `M PKTUSB 3000` had hamlib rewrite
    SH and RF and the panel's WIDTH went out. A profile that names its width gets
    hamlib's no-change passband; the FT-891 still asks for FILTER_HZ."""
    monkeypatch.setenv("FAKE_RIGCTL_MUTE_F", "1")
    real_rig = ota.Rig
    monkeypatch.setattr(ota, "Rig", lambda *a, **kw: real_rig(
        *a, rigctl_dir=str(fake_rigctl), **kw))
    monkeypatch.setattr(ota.session, "build_session", lambda *a, **kw: [0.0] * 480)
    monkeypatch.setattr(ota.session, "write_wav", lambda *a, **kw: None)
    monkeypatch.setattr(ota, "find_device", lambda name, kind: 0)
    args = argparse.Namespace(
        outdir=str(tmp_path / "out"), dxcall="K7ABC", dial=7100000, center=None,
        band="40m", data=False, message="", sl=2, mycall="W9SSJ", rig=rig_name,
        dry_run=False, audio_out=None, audio_in=None, serial="/dev/null",
        baud=0, attempts=1, listen=0.1, gap=0.0, max_key=40.0)
    with pytest.raises(SystemExit):
        ota.run(args)
    modes = [line for line in (tmp_path / "rigctl.log").read_text().splitlines()
             if line.startswith("M ")]
    assert modes == [f"M PKTUSB {passband}"], modes


# -- fix 3: keying that cannot be confirmed is not keying --------------------

def test_a_key_down_refuses_a_device_that_is_not_there(fake_rigctl) -> None:
    with pytest.raises(PttError, match=PLACEHOLDER):
        _rig(fake_rigctl, serial=PLACEHOLDER, ptt_port=PLACEHOLDER)


def test_a_key_without_readback_is_reported(fake_rigctl, fake_lines):
    """An accepted drive is not carrier evidence when readback disappears."""
    rig = _rig(fake_rigctl)
    rig.arm()
    line = fake_lines[-1]
    line.readable = False
    assert rig.ptt(True) is False
    assert rig.key_failure() is not None
    assert rig.retired
    rig.stop()


def test_a_standing_channel_reports_no_failure(fake_rigctl) -> None:
    rig = _rig(fake_rigctl)
    assert rig.ptt(True) is True
    time.sleep(0.1)
    assert rig.key_failure() is None
    rig.ptt(False)
    rig.stop()


def test_a_line_that_refuses_a_key_is_seen(fake_rigctl, fake_lines):
    """A present adapter refusing its assert cannot produce a carrier record."""
    rig = _rig(fake_rigctl)
    line = fake_lines[-1]
    line.fail_on = True
    assert rig.ptt(True) is False
    assert rig.key_failure() is not None
    assert line.state is False
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

    def drive(on):
        if on:
            bursts[0] += 1
        elif unplug_during and bursts[0] == unplug_during:
            if Path(serial).exists():
                Path(serial).unlink()
            for line in lines:
                line.readable = False

    real_rig = ota.Rig
    lines = []

    def make_rig(*a, **kw):
        rig = real_rig(*a, rigctl_dir=str(fake_rigctl), **kw)
        lines.append(rig._keyer.transport._ptt)
        lines[-1].drive_hook = drive
        return rig

    monkeypatch.setattr(ota, "Rig", make_rig)
    monkeypatch.setattr(onair, "_LiveInput",
                        lambda *a, **kw: _Bench())
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
    assert "keying line" in log and str(cat) in log, log[-2000:]
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
    assert all(shutil.which(name, path=str(empty)) is None
               for name in ("rigctl", "rigctld"))
    r = subprocess.run(
        [sys.executable, "-m", "hfmodem.shrike.onair", "--transmit",
         "--mycall", "W9SSJ", "--dxcall", "K7ABC", "--dial", "7100000",
         "--serial", "/dev/null", "--max-cycles", "1",
         "--outdir", str(tmp_path / "cap")],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PATH": str(empty),
             "HAMLIB_BIN": str(tmp_path / "nope")})
    said = r.stdout + r.stderr
    assert r.returncode != 0, said
    assert "Traceback" not in said and "FileNotFoundError" not in said, said
    assert "NOT KEYING" in said and "rigctl" in said, said
    assert str(empty) in said, f"the refusal does not say where it looked:\n{said}"
    assert "HAMLIB_BIN" in said, f"the refusal does not say how to supply it:\n{said}"


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    # A symlink under a scratch .venv has no pyvenv.cfg and loses editable installs.
    # Exercise the launcher with its supported explicit interpreter override.
    monkeypatch.setenv("HFMODEM_PYTHON", sys.executable)
    library = tmp_path / "tools/lib/ports.sh"
    library.parent.mkdir(parents=True)
    library.write_bytes(_PORTS_LIB.read_bytes())
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    return library


@requires_launcher
def test_the_launcher_puts_hamlib_on_path_for_the_children_it_starts(tmp_path, launcher) -> None:
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
        ["/bin/bash", "-c", f'set -euo pipefail; source "{launcher}"; '
                            f'"{sys.executable}" -c '
                            "'import shutil; print(shutil.which(\"rigctl\"))'"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": f"{empty}:/usr/bin:/bin",
             "HAMLIB_BIN": str(ham)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(ham / "rigctl"), (r.stdout, r.stderr)


@requires_launcher
def test_the_launcher_refuses_a_machine_with_no_hamlib_at_all(tmp_path, launcher) -> None:
    """And when there is none to put there, the refusal is the launcher's own,
    before a modem is started -- not a Python traceback out of whichever tool
    reached for it first."""
    empty = tmp_path / "bin"
    empty.mkdir()
    dirname = shutil.which("dirname")
    assert dirname is not None
    (empty / "dirname").symlink_to(Path(dirname).resolve())
    assert all(shutil.which(name, path=str(empty)) is None
               for name in ("rigctl", "rigctld"))
    r = subprocess.run(
        ["/bin/bash", "-c", f'source "{launcher}"; '
                            '! command -v rigctl && ! command -v rigctld || exit 97; require_hamlib'],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": str(empty), "HFMODEM_PYTHON": sys.executable,
             "HAMLIB_BIN": str(tmp_path / "nope")})
    said = r.stdout + r.stderr
    assert r.returncode != 0, said
    assert "hamlib is not on PATH" in said and "HAMLIB_BIN" in said, said


def test_a_rig_whose_unkey_ladder_ran_out_refuses_to_key_again(
        fake_rigctl, fake_lines):
    """Retirement refuses further keys but must still allow another unkey."""
    rig = _rig(fake_rigctl)
    assert rig.ptt(True)
    line = fake_lines[-1]
    line.fail_on = False
    assert rig.ptt(False) is False
    assert rig.retired
    before = list(line.calls)
    assert rig.ptt(True) is False
    assert line.calls == before
    line.fail_on = None
    assert rig.ptt(False) is True
    assert line.state is False and line.calls[-2:] == [False, "sense"]
    rig.stop()
