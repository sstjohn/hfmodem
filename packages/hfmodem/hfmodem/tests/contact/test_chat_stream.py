# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('selected,reason,unavailable', [
    ('line', None, False),
    ('line', 'Textual unavailable; using line console.', False),
    ('textual', None, True),
])
def test_ui_fallback_runs_line_console_and_warns_once(tools, monkeypatch, capsys,
                                                     selected, reason, unavailable):
    import varac_chat as launch
    events = []
    session = SimpleNamespace(monitor=lambda: events.append('monitor'))
    tty = object()

    @contextmanager
    def opened(*args, **kwargs):
        yield session, tty

    @contextmanager
    def missing():
        raise OSError('terminal disappeared')
        yield

    monkeypatch.setattr(launch, 'sense_ui', lambda mode: (selected, reason))
    monkeypatch.setattr(launch, 'open_chat', opened)
    monkeypatch.setattr(launch, 'textual_tty', missing)
    monkeypatch.setattr(launch, 'terminal', lambda s, t: events.append((s, t)))
    assert launch.run(SimpleNamespace(ui='auto'), object(), object()) == 0
    assert events == ['monitor', (session, tty)]
    captured = capsys.readouterr()
    assert captured.out == ''
    lines = captured.err.splitlines()
    assert len(lines) == int(reason is not None or unavailable)
    if unavailable:
        assert 'terminal disappeared' in lines[0] and 'using line console' in lines[0]
