# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from contextlib import ExitStack
import inspect
import io
import queue
import sys
from types import SimpleNamespace

import pytest

from hfmodem.tests.contact.parent_console import source


@pytest.mark.parametrize('stdin_tty', (False, True))
@pytest.mark.parametrize('available', (False, True))
def test_terminal_matches_parent_with_redirected_input_and_missing_tty(
        tools, monkeypatch, stdin_tty, available):
    import varac_chat as launch
    parent = source()
    parent = parent[parent.index('class Terminal:'):parent.index('\n\nclass Diagnostics:')]

    def exercise(source):
        trace = []
        inherited = io.StringIO('/quit\nenclosing command\nsecond enclosing command\n')
        output = SimpleNamespace(flush=lambda: trace.append('flush-log'))
        streams = SimpleNamespace(stdin=inherited, stdout=output)
        screen = io.StringIO()

        def open_tty(path, flags):
            trace.append(('open', path, flags))
            if not available:
                raise OSError('no controlling terminal')
            return 40

        class Reader:
            def __init__(self, *, target, daemon, args=()):
                trace.append(('reader', target.__name__, daemon, args))

            def start(self):
                trace.append('start-reader')

        readline = SimpleNamespace(
            backend='editline', parse_and_bind=lambda value: trace.append(('bind', value)),
            set_history_length=lambda value: trace.append(('history-length', value)),
            read_history_file=lambda value: trace.append(('history-read', value)))
        monkeypatch.setitem(sys.modules, 'readline', readline)
        namespace = dict(
            sys=streams, queue=queue, threading=SimpleNamespace(Thread=Reader),
            os=SimpleNamespace(O_RDWR=2, isatty=lambda fd: stdin_tty, open=open_tty,
                               close=lambda fd: trace.append(('close', fd)),
                               dup=lambda fd: trace.append(('dup', fd)) or fd + 100,
                               dup2=lambda a, b: trace.append(('dup2', a, b))),
            termios=SimpleNamespace(TCSANOW=0, tcgetattr=lambda fd: ('attrs', fd),
                                    tcsetattr=lambda *args: trace.append(('restore', args))),
            io=SimpleNamespace(FileIO=lambda fd, mode: (fd, mode)),
            Log=lambda raw, **kwargs: trace.append(('log', raw, kwargs)) or 'evidence-pipe',
            open=lambda fd, mode, **kwargs: screen)
        exec(source, namespace)
        with ExitStack() as held:
            try:
                terminal = namespace['Terminal'](held, 'history')
            except OSError as error:
                trace.append(('error', str(error)))
            else:
                terminal.write('operator screen\n')
                terminal.write_bbs('page')
                terminal.flush()
                trace.append(('screen', screen.getvalue()))
        return trace, streams.stdout is output, inherited.read()

    actual = exercise(inspect.getsource(launch.Terminal))
    assert actual == exercise(parent)
    assert actual[2] == '/quit\nenclosing command\nsecond enclosing command\n'
