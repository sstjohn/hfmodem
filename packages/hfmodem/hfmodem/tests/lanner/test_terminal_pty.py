# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The RTTY terminal on a real pty, started through the launcher's own `rtty)` arm,
on a fake card and keying line (`qso_on_fakes`). The arm runs the terminal in the
background of a shell with no job control, which is where stdin, SIGINT and
SIGWINCH could be lost; this drives keys, a resize, turnover and each exit path."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import struct
import sys
import termios
import time

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
LAUNCHER = ROOT / 'tools' / 'onair.sh'

#: The forked child does nothing but exec bash, so the other threads this test
#: process may hold cannot leave it a lock to deadlock on.
pytestmark = [pytest.mark.skipif(not (LAUNCHER.is_file() and shutil.which('bash')),
                                 reason='needs the launcher and bash'),
              pytest.mark.filterwarnings('ignore:This process .* is multi-threaded, use of forkpty')]

ARM = r'''
set -euo pipefail
MYCALL=W9SSJ; AUDIO_IN=codec; AUDIO_OUT=codec; PTT_PORT=/dev/null; RIG_NAME=ftdx10
RIG_MODE=PKTUSB; VENV="$STUB"; REPO="$STUB"
die()              { printf '!! %s\n' "$*" >&2; exit 1; }
say()              { :; }
require_cat()      { :; }
eval "$(awk '/^redact_secrets\(\)/ {f = 1} f {print} f && /^}/ {exit}' "$PORTS")"
eval "$(awk '/^RIG_LOCK_FD=/ {f = 1} f {print} f && /^}/ {exit}' "$PORTS")"
require_hardware() { claim_rig; }
rigctld_start()    { trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP; }
rig_power()        { :; }
tune()             { :; }
nc()               { case "$(cat)" in *get_mode*) echo 'Mode: PKTUSB' ;; esac; }
unkey_and_exit()   { printf 'verdict %s %s\n' "$1" "$2"; exit "$1"; }
eval "$(awk '/^positional\(\)/ {f = 1} f {print} f && /^}/ {exit}' "$LAUNCHER")"
set -- rtty 7080000 --adif "$STUB/log.adi" --session "$STUB/session"
eval "case rtty in $(awk 'index($0, "rtty)") == 1 {f = 1} f && /^  ;;/ {exit} f' "$LAUNCHER" | sed 's/#.*//') ;; esac"
'''


class Terminal:
    def __init__(self, tmp_path, owner=None):
        (tmp_path / 'python').write_text(
            f'#!/bin/bash\n[ "$1" = -m ] || exec {sys.executable} "$@"\n'
            f'shift 2\nexec {sys.executable} {HERE / "qso_on_fakes.py"} "$@"\n')
        (tmp_path / 'python').chmod(0o755)
        (tmp_path / 'arm.sh').write_text(ARM)
        self.log = tmp_path / 'fake.log'
        self.log.touch()
        self.claim = Path(f'/tmp/hfmodem-rig-hfmodem-pty-test-{os.getpid()}.claim')
        env = dict(os.environ, STUB=str(tmp_path), LAUNCHER=str(LAUNCHER), TERM='xterm',
                   PORTS=str(LAUNCHER.parent / 'lib' / 'ports.sh'),
                   CAT_PORT=f'/dev/hfmodem-pty-test-{os.getpid()}',
                   FAKE_LOG=str(self.log), PYTHONPATH=os.pathsep.join(sys.path))
        # Inherited fixed dimensions override ncurses resizing. This child owns
        # a real pty; ncurses must follow its ioctl dimensions across resizes.
        for name in ('LINES', 'COLUMNS', 'HFMODEM_STATION', 'HFMODEM_OWNER_FD',
                     'HFMODEM_RIG', 'RIG_CLAIMED'):
            env.pop(name, None)
        if owner is not None:
            os.set_inheritable(owner, True)
            env['HFMODEM_OWNER_FD'] = str(owner)
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execvpe('bash', ['bash', str(tmp_path / 'arm.sh')], env)
        self.resize(24, 100)
        self.screen, self.status = b'', None

    def resize(self, rows, cols):
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))

    def pump(self, seconds=0.2):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if select.select([self.fd], [], [], 0.05)[0]:
                try:
                    self.screen += os.read(self.fd, 65536)
                except OSError:
                    return

    def send(self, keys, settle=0.3):
        os.write(self.fd, keys)
        self.pump(settle)

    def said(self):
        return self.log.read_text().splitlines()

    def until(self, line, seconds=10, after=0):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if line in self.said()[after:]:
                return True
            self.pump(0.1)
        return False

    def alive(self):
        if self.status is None and os.waitpid(self.pid, os.WNOHANG) != (0, 0):
            self.status = 'exited'
        return self.status is None

    def close(self):
        if self.alive():
            os.killpg(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
        os.close(self.fd)
        self.claim.unlink(missing_ok=True)


@pytest.fixture
def terminal(tmp_path):
    made = Terminal(tmp_path)
    yield made
    made.close()


@pytest.mark.parametrize('stop', ['ctrl-c', 'ctrl-q', 'group-int', 'launcher-int'])
def test_the_backgrounded_terminal_takes_keys_resizes_and_unkeys_before_exit(terminal, stop):
    assert terminal.until('size 24x100', 30), terminal.screen[-400:]
    assert b'RX' in terminal.screen
    terminal.resize(30, 60)
    assert terminal.until('size 30x60')

    terminal.send(b'\x1bOP')
    assert terminal.until('key up'), 'F1 did not key'
    keyed = len(terminal.said())
    terminal.send(b'\x1b', settle=0.5)
    assert terminal.until('key down', 30, after=keyed), 'Esc did not turn the over'

    keyed = len(terminal.said())
    terminal.send(b'AB\t')
    assert terminal.until('key up', after=keyed)
    keyed = len(terminal.said())
    if stop in ('ctrl-c', 'ctrl-q'):
        terminal.send(b'\x03' if stop == 'ctrl-c' else b'\x11')
    elif stop == 'group-int':
        os.killpg(os.tcgetpgrp(terminal.fd), signal.SIGINT)
    else:
        os.kill(terminal.pid, signal.SIGINT)
    assert terminal.until('key down', after=keyed)
    end = time.monotonic() + 15
    while terminal.alive() and time.monotonic() < end:
        terminal.pump(0.2)
    terminal.pump()
    assert not terminal.alive()
    code = 130 if stop.endswith('int') else 0
    assert f'verdict {code} 7080000'.encode() in terminal.screen
    assert 'hand back' in terminal.said()


def test_a_console_gone_mid_over_halts_the_terminal_through_the_launcher(tmp_path):
    heard, owner = os.pipe()
    terminal = Terminal(tmp_path, owner=heard)
    os.close(heard)
    try:
        assert terminal.until('size 24x100', 30)
        terminal.send(b'CQ\t')
        assert terminal.until('key up')
        keyed = len(terminal.said())
        os.close(owner)
        assert terminal.until('key down', after=keyed)
        keyed = len(terminal.said())
        terminal.send(b'\t', settle=1)
        assert 'key up' not in terminal.said()[keyed:] and terminal.alive()
    finally:
        terminal.close()


def _is_open(fd):
    try:
        fcntl.fcntl(fd, fcntl.F_GETFD)
    except OSError:
        return False
    return True


def pipe_on_nine(make):
    """`make()` as it runs when os.pipe() hands back descriptor 9 for the read end,
    whether 9 was open beforehand or not; what was there is put back."""
    real, held = os.pipe, os.dup(9) if _is_open(9) else None
    first, owner = real()
    if first != 9:
        os.dup2(first, 9)
        os.close(first)
    os.pipe = lambda: (9, owner)
    try:
        return make()
    finally:
        os.pipe = real
        if held is not None:
            os.dup2(held, 9)
            os.close(held)
        elif _is_open(9):
            os.close(9)


@pytest.mark.parametrize('nine', ['open', 'closed'])
def test_pipe_on_nine_puts_back_what_it_found(nine):
    """Closed, 9 is where a pipe made inside lands when the lower descriptors are
    busy; open, it is someone else's and must come back unchanged."""
    import stat
    held = os.dup(9) if _is_open(9) else None
    if nine == 'closed' and held is not None:
        os.close(9)
    if nine == 'open' and held is None:
        spare = os.open(os.devnull, os.O_RDONLY)
        os.dup2(spare, 9)
        os.close(spare)
    try:
        before = os.fstat(9).st_ino if nine == 'open' else None
        seen = []
        heard, owner = pipe_on_nine(lambda: (seen.append(os.fstat(9).st_mode), os.pipe())[1])
        assert heard == 9 and stat.S_ISFIFO(seen[0])
        os.close(owner)
        assert os.fstat(9).st_ino == before if nine == 'open' else not _is_open(9)
    finally:
        if held is not None:
            os.dup2(held, 9)
            os.close(held)
        elif _is_open(9):
            os.close(9)


def test_the_ownership_pipe_survives_the_launchers_rig_claim(tmp_path):
    """The rig claim takes descriptor 9 in the launcher, which is where a bare
    os.pipe() lands when the lower ones are busy."""
    rtty = pytest.importorskip('consolelib.rtty')
    heard, owner = pipe_on_nine(rtty.owner_pipe)
    terminal = Terminal(tmp_path, owner=heard)
    os.close(heard)
    try:
        assert terminal.until('size 24x100', 30), terminal.screen[-400:]
        assert terminal.claim.is_file()
        terminal.send(b'CQ\t')
        assert terminal.until('key up'), terminal.screen[-600:]
        keyed = len(terminal.said())
        os.close(owner)
        assert terminal.until('key down', after=keyed)
    finally:
        terminal.close()
