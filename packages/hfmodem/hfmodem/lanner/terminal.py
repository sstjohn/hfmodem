# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The RTTY terminal: curses over a `qso.Session`. Stdout stays the session's event
log, which the launcher records, so the screen binds the operator console."""
from __future__ import annotations

from contextlib import ExitStack
import curses
import os
import select
import sys

from hfmodem.lanner.qso import FS, MACROS
from hfmodem.core import posix

termios = posix.module("termios")

HISTORY = 2000

CTRL = {chr(ord(letter) - 64): letter for letter in 'CEFKQRU'}
KEYS = ('^C unkey+quit', 'Tab key+send/turn', 'Esc/^[ key,send,unkey', 'F1 key+CQ', f'F2-F{len(MACROS)} macros',
        '^K call', '^E log', '^R rev', '^U USOS', '^F find', '<> nudge', 'up/dn sq',
        '+- drive', '^Q quit')
#: Polls that found the terminal readable and yet no key: a hung-up tty reads EOF.
EOF_POLLS = 3

_console = None


def handle(session, key):
    """One key. Returns 'leave' for ^C/^Q, 'flash' for a character ITA2 cannot send."""
    if key == '\t':
        session.toggle()
    elif key == '\x1b':
        session.turn_over()
    elif key in CTRL:
        letter = CTRL[key]
        if letter in ('C', 'Q'):
            session.abort()
            return 'leave'
        {'K': session.set_call, 'E': session.commit,
         'R': session.toggle_reverse, 'U': session.toggle_usos,
         'F': session.reacquire}[letter]()
    elif isinstance(key, int) and curses.KEY_F1 <= key < curses.KEY_F1 + len(MACROS):
        session.macro(key - curses.KEY_F1)
    elif key in (curses.KEY_LEFT, curses.KEY_RIGHT):
        session.nudge(1 if key == curses.KEY_RIGHT else -1)
    elif key in (curses.KEY_UP, curses.KEY_DOWN):
        session.adjust_squelch(0.05 if key == curses.KEY_UP else -0.05)
    elif key in ('+', '-'):
        session.adjust_drive(1 if key == '+' else -1)
    elif key in (curses.KEY_BACKSPACE, '\x7f', '\b'):
        session.backspace()
    elif key in ('\r', '\n', curses.KEY_ENTER):
        session.type('\n')
    elif isinstance(key, str) and key.isprintable() and not session.type(key):
        return 'flash'
    return None


class Pane:
    """Lines of (text, kind, dim) runs, newest last."""

    def __init__(self):
        self.lines = [[]]

    def add(self, kind, text, dim=False):
        if kind in ('note', 'alarm'):
            if self.lines[-1]:
                self.lines.append([])
            self.lines[-1].append((f'{"!!" if kind == "alarm" else "--"} {text}', kind, False))
            self.lines.append([])
        else:
            first, *rest = text.split('\n')
            self.lines[-1].append((first, kind, dim))
            self.lines += [[(part, kind, dim)] for part in rest]
        del self.lines[:-HISTORY]


def status(session):
    rf = session.mark_rf_hz
    since = ('never' if session.id_at is None
             else f'{(session.audio.samples - session.id_at) / FS:.0f}s ago')
    current = session.log.current
    above = session.drive > session.drive_ceiling
    return '  '.join(part for part in (
        'PTT UNCONFIRMED' if session.ptt_unconfirmed else session.state.upper(),
        'REC TROUBLE' if session.recording_trouble else '',
        session.alarm_text and f'!! {session.alarm_text}',
        *(f'!! {warning}' for warning in session.warnings),
        above and f'DRIVE {session.drive:.3f} ABOVE tx_drive {session.drive_ceiling:.3f}',
        str(session.rate),
        f'mark {rf / 1e3:.3f} kHz dial {session.dial_hz / 1e3:.3f}' if rf else 'dial unknown',
        f'rx {session.rx_centre - session.tx_centre:+.0f} Hz' + (' REV' if session.reverse else '')
        + ('' if session.stream.unshift_on_space else ' noUSOS'),
        f'sq {session.squelch:.0%}',
        '' if above else f'drive {session.drive:.3f}',
        f'ID {since}',
        f'#{session.log.serial:03d}' + (f' {current.call}' if current else ''),
    ) if part)


def wrap(line, width):
    """One pane line of (text, kind, dim) runs as rows no wider than `width`."""
    rows, row, used = [], [], 0
    for text, kind, dim in line:
        while text:
            take = text[:width - used]
            row.append((take, kind, dim))
            used += len(take)
            text = text[len(take):]
            if used == width:
                rows.append(row)
                row, used = [], 0
    return rows + [row] if row or not rows else rows


def help_rows(width, count):
    rows = ['']
    for item in KEYS:
        if rows[-1] and len(rows[-1]) + 2 + len(item) > width:
            rows.append('')
        rows[-1] += ('  ' if rows[-1] else '') + item
    return [row[:width] for row in rows[:count]]


def draw(screen, session, pane, attrs):
    """The typeahead and the status line first, at the bottom, then the spectrum
    strip, the key help and the pane as the window allows."""
    height, width = screen.getmaxyx()
    width -= 1
    screen.erase()
    if width < 1 or height < 1:
        return
    keys = 2 if height >= 10 else 0
    strip = height >= 5
    status_row = height >= 2
    y = height - 1
    screen.addstr(y, 0, ('> ' + session.unsent.replace('\n', '|'))[-width:])
    if status_row:
        y -= 1 + strip
        screen.addstr(y, 0, status(session)[:width], attrs['status'])
    if strip:
        line, mark, space = session.spectrum(width)
        screen.addstr(y + 1, 0, line)
        for col, letter in ((mark, 'M'), (space, 'S')):
            screen.addstr(y + 1, col, letter, attrs['status'])
    help_lines = help_rows(width, keys)
    if (trouble := session.recording_trouble) and help_lines:
        help_lines[-1] = f'REC {trouble}'[:width]
    y -= len(help_lines)
    for row, line in enumerate(help_lines):
        screen.addstr(y + row, 0, line, attrs['note'])
    if y > 0:
        shown = [row for line in pane.lines[-y:] for row in wrap(line, width)][-y:]
        for row, runs in enumerate(shown):
            col = 0
            for text, kind, dim in runs:
                screen.addstr(row, col, text, attrs[kind] | (attrs['dim'] if dim else 0))
                col += len(text)
    screen.refresh()


def tty_lost(fd=0):
    """True for a terminal that is gone, not one that is merely quiet."""
    if os.name == 'nt':
        return _console is None or _console.lost()
    try:
        termios.tcgetattr(fd)
        return bool(select.select([fd], [], [], 0)[0])
    except (termios.error, OSError, ValueError):
        return True


def loop(screen, session, attrs, lost=tty_lost):
    pane = Pane()
    silent = 0
    while True:
        if session._owner is not None and session._owner.gone():
            session.abort()
            session.note('the console that owns this session is gone')
            return 1
        while session.pieces:
            pane.add(*session.pieces.popleft())
        try:
            draw(screen, session, pane, attrs)
        except curses.error:
            pass
        except Exception:
            session.abort()
            raise
        try:
            key = screen.get_wch()
        except curses.error:
            silent = silent + 1 if lost() else 0
            if silent < EOF_POLLS:
                continue
            session.abort()
            session.note('the terminal is gone: unkeyed and leaving')
            return 1
        silent = 0
        said = handle(session, key)
        if said == 'leave':
            return 0
        if said == 'flash':
            curses.flash()


def claim_tty(held=None):
    """Bind the screen to the console; Windows ownership belongs to `held`."""
    if os.name == 'nt':
        from hfmodem.core.winconsole import Console
        global _console
        console = Console(bind=True)
        try:
            held.callback(console.close)
            held.callback(setattr, sys.modules[__name__], '_console', _console)
            _console = console
        except BaseException:
            console.close()
            raise
        return sys.stdout
    log = os.fdopen(os.dup(1), 'w', buffering=1)
    tty = os.open('/dev/tty', os.O_RDWR)
    os.dup2(tty, 0)
    os.dup2(tty, 1)
    os.close(tty)
    sys.stdout = log
    return log


def run(session):
    os.environ.setdefault('ESCDELAY', '25')

    def main(screen):
        curses.raw()
        screen.keypad(True)
        screen.timeout(100)
        attrs = dict(rx=curses.A_NORMAL, tx=curses.A_BOLD, note=curses.A_DIM,
                     dim=curses.A_DIM, status=curses.A_REVERSE,
                     alarm=curses.A_REVERSE | curses.A_BOLD)
        if curses.has_colors():
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_RED)
            attrs.update(tx=curses.color_pair(1), note=curses.color_pair(2),
                         alarm=curses.color_pair(3) | curses.A_BOLD)
        session.start()
        session.on_screen = True
        try:
            return loop(screen, session, attrs)
        except BaseException as exc:
            # Before curses restores the terminal, which can itself block.
            session.halt(f'the screen failed ({type(exc).__name__}: {exc})')
            raise
        finally:
            session.on_screen = False

    with ExitStack() as held:
        if os.name == 'nt' and _console is None:
            claim_tty(held)
        return curses.wrapper(main)
