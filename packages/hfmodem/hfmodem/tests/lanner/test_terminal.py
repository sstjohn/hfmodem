# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The terminal's key map and screen, driven by a scripted key feed on the bench card."""
from __future__ import annotations

import curses
import io

import pytest

from hfmodem.lanner import qso, terminal
from hfmodem.station.qsolog import QsoLog

from .test_qso import Keyer, Loopback, emitted, run


@pytest.fixture
def session(tmp_path):
    loop = Loopback(20)
    keyer = Keyer(loop.sa)
    made = qso.Session(loop.sa, keyer, QsoLog(tmp_path / 'log.adi', 'W9SSJ', contest='TEST'),
                       mycall='W9SSJ', dial_hz=14_080_000, out=io.StringIO(),
                       shout=lambda line: None)
    made.loop, made.keyer = loop, keyer
    return made


class Screen:
    """A window that hands out scripted keys, letting the card run between them."""

    def __init__(self, session, script, size=(24, 100)):
        self.session, self.script, self.text, self.size = session, list(script), {}, size

    def getmaxyx(self):
        return self.size

    def erase(self):
        self.text = {}

    def addstr(self, row, col, text, attr=0):
        height, width = self.size
        if not (0 <= row < height and 0 <= col and col + len(text) < width):
            raise curses.error(f'addstr({row}, {col}, {len(text)} chars) outside {height}x{width}')
        line = self.text.get(row, '').ljust(col)
        self.text[row] = line[:col] + text + line[col + len(text):]

    def refresh(self):
        pass

    def get_wch(self):
        key, seconds = self.script.pop(0)
        run(self.session.loop, self.session, seconds,
            until=(lambda: self.session.state == 'rx') if seconds > 2 else None)
        return key


ATTRS = dict(rx=0, tx=0, note=0, dim=0, status=0, alarm=0)


def test_a_scripted_over_through_the_screen(session):
    script = ([(ch, 0) for ch in 'CQ DE W9SSJ'] + [('\t', 0.3)]
              + [(ch, 0) for ch in ' K'] + [('\x1b', 0), (curses.KEY_RESIZE, 8), ('\x11', 0)])
    screen = Screen(session, script)
    assert terminal.loop(screen, session, ATTRS) == 0
    assert emitted(session.loop) == 'CQ DE W9SSJ K\n'
    assert [on for on, _ in session.keyer.events] == [True, False]
    shown = '\n'.join(screen.text.values())
    assert 'CQ DE W9SSJ K' in shown and 'RX' in shown and 'M' in screen.text[22]


def test_keys_reach_the_session(session):
    assert terminal.handle(session, '@') == 'flash'
    for key in 'AB':
        terminal.handle(session, key)
    terminal.handle(session, curses.KEY_BACKSPACE)
    terminal.handle(session, '\r')
    assert session.typeahead == ['A', '\n']
    terminal.handle(session, curses.KEY_RIGHT)
    terminal.handle(session, '\x12')
    assert session.rx_centre == qso.CENTRE_HZ + qso.NUDGE_HZ and session.reverse
    terminal.handle(session, '\x15')
    assert not session.stream.unshift_on_space
    before = session.drive
    terminal.handle(session, '+')
    assert session.drive == pytest.approx(before * 10 ** (1 / 20))
    terminal.handle(session, curses.KEY_DOWN)
    assert session.squelch == pytest.approx(qso.SQUELCH_OPEN - 0.05)
    session._heard = 'W9SSJ DE K1ABC K'
    terminal.handle(session, '\x0b')
    assert session.log.current.call == 'K1ABC'
    session.typeahead.clear()
    terminal.handle(session, curses.KEY_F1 + 3)
    assert ''.join(session.typeahead) == '\nK1ABC 599 001 001\n'
    terminal.handle(session, '\x05')
    assert session.log.serial == 2
    terminal.handle(session, curses.KEY_F1)
    assert session.state == 'tx'
    assert terminal.handle(session, '\x03') == 'leave'
    assert session.state == 'rx' and not session.keyer.up
    assert terminal.handle(session, '\x11') == 'leave'


def test_the_pane_interleaves_both_sides_and_notes():
    pane = terminal.Pane()
    pane.add('rx', 'CQ DE K1ABC\nK')
    pane.add('tx', 'K1ABC DE W9SSJ')
    pane.add('note', 'logged')
    pane.add('rx', 'X', True)
    assert pane.lines == [[('CQ DE K1ABC', 'rx', False)],
                          [('K', 'rx', False), ('K1ABC DE W9SSJ', 'tx', False)],
                          [('-- logged', 'note', False)],
                          [('X', 'rx', True)]]


def test_the_status_line_says_where_we_are(session):
    line = terminal.status(session)
    assert line.startswith('RX  45.45/170  mark 14082.295 kHz dial 14080.000  rx +0 Hz')
    assert 'ID never' in line and '#001' in line
    session.adjust_drive(+3)
    assert 'ABOVE tx_drive' in terminal.status(session)


def test_a_long_line_wraps_rather_than_losing_its_end(session):
    pane = terminal.Pane()
    pane.add('rx', 'CQ CQ CQ CONTEST DE K1ABC K1ABC K1ABC ' * 2 + 'UR 599 NNNN N0XYZ')
    screen = Screen(session, [], size=(24, 40))
    terminal.draw(screen, session, pane, ATTRS)
    shown = ''.join(screen.text[row] for row in sorted(screen.text) if row < 19)
    assert shown.endswith('UR 599 NNNN N0XYZ')


@pytest.mark.parametrize('height', [1, 2, 3, 4, 5, 6, 9])
def test_a_small_window_keeps_the_typeahead_and_drops_the_rest_in_order(session, height):
    pane = terminal.Pane()
    pane.add('rx', 'K1ABC DE N0XYZ\n' * 5)
    session.type('Q')
    screen = Screen(session, [], size=(height, 30))
    terminal.draw(screen, session, pane, ATTRS)
    assert screen.text[height - 1].startswith('> Q')
    if height >= 2:
        assert screen.text[height - 2 - (height >= 5)].startswith('RX')


def test_a_drawing_error_does_not_end_the_session(session):
    class Flaky(Screen):
        failed = False

        def addstr(self, *args):
            if not Flaky.failed:
                Flaky.failed = True
                raise curses.error('drawn off the edge')
            return super().addstr(*args)

    screen = Flaky(session, [('A', 0), ('\x11', 0)])
    assert terminal.loop(screen, session, ATTRS) == 0
    assert Flaky.failed and session.typeahead == ['A']


def test_the_drive_warning_survives_an_80_column_status_line(session):
    session.adjust_drive(+3)
    assert 'ABOVE tx_drive' in terminal.status(session)[:79]


def test_the_key_help_says_which_keys_transmit(session):
    shown = '  '.join(terminal.help_rows(99, 2))
    for key in ('^C unkey+quit', 'Tab key+send', 'Esc key,send,unkey', 'F1 key+CQ',
                f'F2-F{len(qso.MACROS)} macros'):
        assert key in shown


@pytest.mark.parametrize('size', [(10, 30), (8, 24), (12, 10), (12, 40), (24, 80), (3, 6)])
def test_every_row_fits_and_status_and_typeahead_are_always_drawn(session, size):
    height, width = size
    session.type('Q')
    screen = Screen(session, [], size=size)
    terminal.draw(screen, session, terminal.Pane(), ATTRS)
    assert screen.text[height - 1].startswith('> Q'[:width - 1])
    assert screen.text[height - 2 - (height >= 5)].startswith('RX'[:width - 1])
    if width >= 40 and height >= 10:
        shown = screen.text[height - 5] + screen.text[height - 4]
        assert shown.startswith('^C unkey+quit') and 'F1 key+CQ' in shown


class Gone(Screen):
    """A terminal whose every read fails, as a hung-up tty's does."""

    def get_wch(self):
        self.reads = getattr(self, 'reads', 0) + 1
        if self.reads > 50:
            raise RuntimeError('the loop never noticed the terminal had gone')
        run(self.session.loop, self.session, 0.01)
        raise curses.error('no input')


def test_a_lost_terminal_unkeys_and_leaves(session):
    session.type('E')
    session.key()
    assert terminal.loop(Gone(session, []), session, ATTRS, lost=lambda: True) == 1
    assert not session.keyer.up and session.state == 'rx'


def test_a_quiet_terminal_is_not_a_lost_one(session):
    class Quiet(Screen):
        def get_wch(self):
            self.reads = getattr(self, 'reads', 0) + 1
            if self.reads < 20:
                raise curses.error('no input')
            self.keyed = session.keyer.up
            return '\x11'

    session.type('E')
    session.key()
    quiet = Quiet(session, [])
    assert terminal.loop(quiet, session, ATTRS, lost=lambda: False) == 0
    assert quiet.keyed and not session.keyer.up


def test_a_resolved_alarm_leaves_the_status_line_and_stays_in_the_pane(session, monkeypatch):
    from hfmodem.core import audio as card
    loop = session.loop
    monkeypatch.setattr(card.time, 'monotonic', lambda: loop.t)
    session.type('E')
    session.key()
    run(loop, session, 0.3)
    loop.t += 2
    session.watchdog()
    run(loop, session, 0.1)
    assert '!! WATCHDOG' in terminal.status(session)
    session.adjust_drive(+3)
    session.type('E')
    session.turn_over()
    line = terminal.status(session)
    assert line.startswith('TURNOVER') and 'WATCHDOG' not in line and 'ABOVE tx_drive' in line[:79]
    assert any(kind == 'alarm' and 'WATCHDOG' in text for kind, text, _ in session.pieces)


def test_an_unexpected_drawing_error_unkeys_and_leaves(session):
    class Broken(Screen):
        def addstr(self, *args):
            raise RuntimeError('a broken invariant')

    session.type('E')
    session.key()
    with pytest.raises(RuntimeError):
        terminal.loop(Broken(session, [('\x11', 0)]), session, ATTRS)
    assert not session.keyer.up and session.state == 'rx'


def test_a_failure_inside_the_screen_unkeys_before_curses_restores_the_terminal(session,
                                                                               monkeypatch):
    seen = []

    class Failing(Screen):
        def keypad(self, on):
            pass

        def timeout(self, ms):
            pass

        def get_wch(self):
            raise RuntimeError('get_wch failed')

    def wrapper(main):
        try:
            return main(Failing(session, []))
        except BaseException:
            seen.append(session.keyer.up)
            raise

    monkeypatch.setattr(curses, 'wrapper', wrapper)
    monkeypatch.setattr(curses, 'raw', lambda: None)
    monkeypatch.setattr(curses, 'has_colors', lambda: False)
    monkeypatch.setattr(session, 'start', lambda: None)
    session.type('E')
    session.key()
    assert session.keyer.up
    with pytest.raises(RuntimeError):
        terminal.run(session)
    assert seen == [False]


@pytest.mark.parametrize('key', ['\x03', '\x11'])
def test_exit_keys_end_the_screen_loop_and_unkey(session, key):
    screen = Screen(session, [('\t', 0), (key, 0.3)])
    assert terminal.loop(screen, session, ATTRS) == 0
    assert not session.keyer.up and session.state == 'rx'
    keys = [on for on, _ in session.keyer.events]
    assert keys[0] and keys[1:] and not any(keys[1:])
