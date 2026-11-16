# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import builtins
from types import SimpleNamespace

import pytest

from hfmodem.ui import choose_chat_ui


FACTS = dict(textual_available=True, stdin_tty=True, controlling_tty=True,
             term='xterm-256color', inside_emacs=None, columns=80, rows=24)


@pytest.mark.parametrize('changes', [
    {}, {'columns': 120, 'rows': 40}, {'term': 'screen'}, {'term': 'tmux-256color'},
])
def test_auto_chooses_textual_on_a_rich_terminal(changes):
    assert choose_chat_ui('auto', **(FACTS | changes)) == ('textual', None)


@pytest.mark.parametrize('changes', [
    {'stdin_tty': False}, {'controlling_tty': False}, {'term': None}, {'term': ''},
    {'term': 'dumb'}, {'term': 'DUMB'}, {'term': 'unknown'}, {'term': 'UNKNOWN'},
    {'inside_emacs': 't'}, {'inside_emacs': ''}, {'columns': 79}, {'rows': 23},
    {'columns': 0, 'rows': 0},
])
def test_auto_falls_back_silently_for_each_nonrich_fact(changes):
    assert choose_chat_ui('auto', **(FACTS | changes)) == ('line', None)


def test_auto_missing_extra_explains_the_optional_install_once():
    selected, reason = choose_chat_ui('auto', **(FACTS | {'textual_available': False}))
    assert selected == 'line'
    assert "pip install 'hfmodem[tui]'" in reason
    assert '\n' not in reason


def test_explicit_line_is_independent_of_dependencies_and_terminal_facts():
    assert choose_chat_ui('line', textual_available=False, stdin_tty=False,
                          controlling_tty=False, term=None, inside_emacs='t',
                          columns=0, rows=0) == ('line', None)


@pytest.mark.parametrize('changes', [
    {'stdin_tty': False}, {'term': None}, {'term': 'dumb'}, {'term': 'unknown'}, {'inside_emacs': 't'},
    {'columns': 1, 'rows': 1}, {'term': '', 'inside_emacs': '', 'columns': 0, 'rows': 0},
])
def test_forced_textual_overrides_terminal_richness(changes):
    assert choose_chat_ui('textual', **(FACTS | changes)) == ('textual', None)


@pytest.mark.parametrize('changes, explanation', [
    ({'textual_available': False}, 'install'),
    ({'controlling_tty': False}, '/dev/tty'),
])
def test_forced_textual_falls_back_with_one_line_when_it_cannot_run(changes, explanation):
    selected, reason = choose_chat_ui('textual', **(FACTS | changes))
    assert selected == 'line' and explanation in reason
    assert '\n' not in reason


@pytest.fixture
def sensing(tools, monkeypatch):
    import varac_chat

    calls = []
    monkeypatch.setenv('TERM', 'xterm-256color')
    monkeypatch.delenv('INSIDE_EMACS', raising=False)
    monkeypatch.setattr(varac_chat, 'find_spec', lambda name: calls.append(('spec', name)) or object())
    monkeypatch.setattr(varac_chat, 'os', SimpleNamespace(
        environ=varac_chat.os.environ, O_RDWR=varac_chat.os.O_RDWR,
        isatty=lambda fd: calls.append(('isatty', fd)) or fd == 0,
        get_terminal_size=lambda fd: calls.append(('size', fd)) or (80, 24),
        open=lambda path, flags: calls.append(('open', path, flags)) or 43,
        close=lambda fd: calls.append(('close', fd))))
    return varac_chat, calls


def test_real_sensing_checks_stdin_and_controlling_tty_not_the_stdout_pipe(sensing, monkeypatch):
    launch, calls = sensing

    def forbidden(*args):
        pytest.fail('stdout is the evidence pipe, not a terminal-sensing input')

    monkeypatch.setattr(launch, 'sys', SimpleNamespace(stdout=SimpleNamespace(isatty=forbidden, fileno=forbidden)))
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        assert name != 'textual' and not name.startswith('textual.')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', importing)
    assert launch.sense_ui('auto') == ('textual', None)
    assert ('spec', 'textual') in calls
    assert ('open', '/dev/tty', launch.os.O_RDWR) in calls
    assert ('size', 43) in calls and ('close', 43) in calls
    assert all(call[1] == 0 for call in calls if call[0] == 'isatty')


def test_real_sensing_handles_a_missing_controlling_tty(sensing, monkeypatch):
    launch, _ = sensing

    def missing(*args):
        raise OSError('no controlling terminal')

    monkeypatch.setattr(launch.os, 'open', missing)
    assert launch.sense_ui('auto') == ('line', None)
    selected, reason = launch.sense_ui('textual')
    assert selected == 'line' and '/dev/tty' in reason


def test_real_sensing_nonterminal_stdin_never_opens_a_tty(sensing, monkeypatch):
    launch, calls = sensing
    monkeypatch.setattr(launch.os, 'isatty', lambda fd: False)
    assert launch.sense_ui('auto') == ('line', None)
    assert not any(call[0] == 'open' for call in calls)


def test_forced_sensing_uses_controlling_terminal_with_redirected_stdin(sensing, monkeypatch):
    launch, calls = sensing
    monkeypatch.setattr(launch.os, 'isatty', lambda fd: False)
    assert launch.sense_ui('textual') == ('textual', None)
    assert ('open', '/dev/tty', launch.os.O_RDWR) in calls
    assert ('size', 43) in calls and ('close', 43) in calls


def test_real_sensing_size_failure_is_silent_in_auto_and_force_can_proceed(sensing, monkeypatch):
    launch, _ = sensing

    def unavailable(fd):
        raise OSError('size unavailable')

    monkeypatch.setattr(launch.os, 'get_terminal_size', unavailable)
    assert launch.sense_ui('auto') == ('line', None)
    assert launch.sense_ui('textual') == ('textual', None)


def test_real_sensing_explicit_line_does_not_probe_textual_or_open_tty(sensing):
    launch, calls = sensing
    assert launch.sense_ui('line') == ('line', None)
    assert not any(call[0] in ('spec', 'open', 'size') for call in calls)


def test_real_sensing_missing_extra_returns_the_install_message(sensing, monkeypatch):
    launch, _ = sensing
    monkeypatch.setattr(launch, 'find_spec', lambda name: None)
    selected, reason = launch.sense_ui('auto')
    assert selected == 'line' and "pip install 'hfmodem[tui]'" in reason


@pytest.mark.parametrize('requested', ['auto', 'textual', 'line'])
def test_real_chat_parser_accepts_the_single_ui_flag(tools, tmp_path, requested):
    import varac_chat
    from hfmodem.tests.contact.test_varac_launcher import arguments

    assert arguments(varac_chat, tmp_path).ui == 'auto'
    assert arguments(varac_chat, tmp_path, '--ui', requested).ui == requested


@pytest.mark.parametrize('changes', [
    {'stdin_tty': False}, {'controlling_tty': False}, {'term': 'dumb'},
    {'inside_emacs': 't'}, {'columns': 79}, {'rows': 23},
])
def test_missing_textual_is_silent_for_nonrich_terminals(changes):
    assert choose_chat_ui('auto', **(FACTS | changes | {'textual_available': False})) == ('line', None)


def test_launcher_ui_hint_uses_stderr(tools, monkeypatch, capsys):
    from contextlib import contextmanager
    import varac_chat as launch
    @contextmanager
    def opened(*args):
        yield SimpleNamespace(monitor=lambda: None), object()
    monkeypatch.setattr(launch, 'sense_ui', lambda requested: ('line', 'install hint'))
    monkeypatch.setattr(launch, 'open_chat', opened)
    monkeypatch.setattr(launch, 'terminal', lambda *args: None)
    launch.run(SimpleNamespace(ui='auto'), None, None)
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err == 'install hint\n'
