# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Operator editing keeps text, cursor and the final spoken ending coherent."""
import curses
import json
import os
import pty
import select
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hfmodem.contact.terminal import Terminal, edit_layout
from hfmodem.macagua import script


@pytest.fixture
def terminal():
    drafts = []
    session = SimpleNamespace(simulation=None, contact=SimpleNamespace(own_call='W9SSJ', state='FREE'),
        cancel_draft=lambda: None, edits=[], event=lambda *args, **kwargs: None,
        make_draft=lambda kind, text: drafts.append((kind, text)), codec=script)
    terminal = Terminal(session)
    terminal.edit_draft_kind = 'free'
    terminal.edit('script', 'The antenna sounds good. Over.', 'Edit script')
    return terminal, drafts


def test_insert_delete_and_navigation_edit_at_cursor(terminal):
    ui, drafts = terminal
    ui.key(curses.KEY_HOME)
    for char in 'W9SSJ. ': ui.key(char)
    ui.key(curses.KEY_RIGHT)
    ui.key(curses.KEY_RIGHT)
    ui.key(curses.KEY_LEFT)
    ui.key(curses.KEY_DC)
    ui.key('h')
    ui.key(curses.KEY_END)
    ui.key(curses.KEY_LEFT)
    ui.key(curses.KEY_BACKSPACE)
    ui.key('r')
    assert ui.editor['value'] == 'W9SSJ. The antenna sounds good. Over.'
    ui.key('\n')
    assert drafts == [('free', 'W9SSJ. The antenna sounds good. Over.')]
    assert ui.editor is None


def test_up_down_keep_column_on_wrapped_lines_and_allow_insertion(terminal):
    ui, drafts = terminal
    ui.edit_width = 10
    ui.edit('script', '0123456789abcdefghijABCDEFGHIJ', 'Edit script')
    ui.key(curses.KEY_HOME)
    for _ in range(5): ui.key(curses.KEY_RIGHT)
    ui.key(curses.KEY_DOWN)
    assert ui.editor['cursor'] == 15
    ui.key(curses.KEY_DOWN)
    assert ui.editor['cursor'] == 25
    ui.key(curses.KEY_UP)
    ui.key('X')
    assert ui.editor['value'] == '0123456789abcdeXfghijABCDEFGHIJ'


def test_editor_is_visible_and_scrolls_cursor_into_view(terminal, monkeypatch):
    ui, drafts = terminal
    visible = []
    monkeypatch.setattr(curses, 'curs_set', visible.append)
    class Screen:
        def __init__(self): self.lines = {}; self.cursor = None
        def erase(self): self.lines.clear()
        def getmaxyx(self): return 12, 20
        def addnstr(self, y, x, text, count, attr=0): self.lines[y] = text[:count]
        def move(self, y, x): self.cursor = (y, x)
        def refresh(self): pass
    screen = Screen()
    value = 'Long editable antenna comparison. ' * 12
    ui.edit('script', value, 'Edit script')
    ui.draw(screen)
    assert visible[-1] == 1
    assert 5 <= screen.cursor[0] < 10 and 0 <= screen.cursor[1] < 18
    assert ui.editor['scroll'] > 0
    ui.key(curses.KEY_HOME)
    ui.draw(screen)
    assert ui.editor['scroll'] == 0 and screen.cursor == (5, 0)
    assert any('Long editable' in text for text in screen.lines.values())


def test_trim_controls_do_not_dispatch_qso_hotkeys(terminal):
    ui, drafts = terminal
    ui.key('\x01')
    for char in 'cq ': ui.key(char)
    ui.key('\x15')
    assert ui.editor['value'] == 'The antenna sounds good. Over.'
    ui.key('\x05')
    for _ in range(6): ui.key(curses.KEY_LEFT)
    ui.key('\x0b')
    assert ui.editor['value'] == 'The antenna sounds good.'
    assert not drafts


def test_layout_counts_wide_characters_for_screen_cursor():
    rows, positions = edit_layout('ab界cd', 4)
    assert rows == ['ab界', 'cd']
    assert positions[3] == (1, 0) and positions[-1] == (1, 2)


@pytest.mark.parametrize('text', [
    'W9SSJ. Your second antenna sounds just as strong. Over.',
    'Your antenna sounds good. Whiskey nine Sierra Sierra Juliett, over.',
    'Whiskey niner Sierra Sierra Juliet. Thanks and 73.',
])
def test_existing_station_call_and_operator_ending_are_preserved(text):
    assert script.identify(text, 'W9SSJ') == text


@pytest.mark.parametrize('ending', ['Over.', 'over', 'OVER!'])
def test_missing_id_goes_before_a_single_terminal_over(ending):
    text = script.identify('Your antenna sounds good. ' + ending, 'W9SSJ')
    assert text.count(script.phonetic('W9SSJ')) == 1
    assert text.lower().count('over') == 1 and text.endswith('Over.')
    assert script.identify(text, 'W9SSJ') == text


def test_real_curses_terminal_decodes_arrows_and_displays_cursor(tmp_path):
    # A real curses wrapper on a PTY: no keyboard, radio, sound card or models.
    result = tmp_path / 'edit.json'
    runner = tmp_path / 'edit.py'
    runner.write_text('''
import curses,json,queue,sys,threading
from pathlib import Path
from types import SimpleNamespace
from hfmodem.contact.terminal import Terminal
from hfmodem.contact.txmark import Tag
from hfmodem.macagua import script
session=SimpleNamespace(simulation=None,test=True,monitor=True,stop=threading.Event(),
    contact=SimpleNamespace(own_call='W9SSJ',state='FREE',heard=None),
    cancel_draft=lambda:None,edits=[],event=lambda *a,**k:None,codec=script,
    check_receive=lambda:None,receive_failure=None,output_done=threading.Event(),
    playing=None,pending=None,poll_draft=lambda:None,events=queue.Queue(),
    waveform=SimpleNamespace(name='voice'),cfg=SimpleNamespace(voice=SimpleNamespace(synthesis_backend='kokoro')),
    tx_mark=Tag(),watermark_key=None,dial_hz=None,draft=None,contacts=[],directory=Path('.'),
    finish_playback=lambda **k:None)
drafts=[]
session.make_draft=lambda kind,text:drafts.append((kind,text))
ui=Terminal(session)
ui.edit_draft_kind='free'
ui.edit('script','antenna sounds good. W9SSJ. Over.','EDIT TEST READY')
curses.wrapper(ui.run)
Path(sys.argv[1]).write_text(json.dumps(drafts))
''')
    master, slave = pty.openpty()
    process = subprocess.Popen([sys.executable, str(runner), str(result)],
        stdin=slave, stdout=slave, stderr=slave, env=dict(os.environ, TERM='xterm-256color'))
    os.close(slave)
    screen = bytearray()
    try:
        deadline = time.monotonic() + 5
        while b'EDIT TEST READY' not in screen:
            if time.monotonic() > deadline:
                pytest.fail('editor did not appear: ' + screen.decode(errors='replace'))
            if select.select([master], [], [], .1)[0]:
                screen.extend(os.read(master, 65536))
        # Ctrl-A, insertion, Ctrl-E, five left arrows, right then left, insert.
        os.write(master, b'\x01The \x05' + b'\x1bOD' * 5 + b'\x1bOC\x1bODback to you. \r')
        # Wait for Enter to finish the edit before pressing the quit hotkey.
        deadline = time.monotonic() + 5
        while b'Preparing' not in screen:
            if time.monotonic() > deadline:
                pytest.fail('edit did not submit: ' + screen.decode(errors='replace'))
            if select.select([master], [], [], .1)[0]:
                screen.extend(os.read(master, 65536))
        os.write(master, b'q')
        deadline = time.monotonic() + 5
        while process.poll() is None:
            if time.monotonic() > deadline:
                pytest.fail('terminal did not quit: ' + screen.decode(errors='replace'))
            if select.select([master], [], [], .1)[0]:
                try:
                    screen.extend(os.read(master, 65536))
                except OSError:
                    break
        process.wait(timeout=1)
        assert process.returncode == 0
        assert json.loads(result.read_text()) == [['free', 'The antenna sounds good. W9SSJ. back to you. Over.']]
        assert b'\x1b[?25h' in screen  # xterm visible-cursor sequence
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
