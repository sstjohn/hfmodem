# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""An unanswered call ends on its cycle count or the wall clock, whichever first.

The Windows PACTOR arm's cycles ran late and the call never gave up: a count
of cycles is a duration only while the cycles keep the grid.
"""
import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike import test_connect_tail
from hfmodem.tests.shrike.test_identify import Clock


def _session(tmp_path, monkeypatch, *, slowdown, cycles, retries):
    """`test_connect_tail.session` with a wall clock running `slowdown` times
    the capture clock: every cycle costs that much more than it should."""
    benches = []

    class _Clock(Clock):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            benches.append(self)

    monkeypatch.setattr(test_connect_tail, 'Clock', _Clock)
    monkeypatch.setattr(onair, '_call_clock',
                        lambda: benches[-1].now / onair.FS * slowdown if benches else 0.0,
                        raising=False)
    keys = test_connect_tail.session(tmp_path, monkeypatch, cycles=cycles,
                                     retries=retries)
    return [k for k in keys if k[0].startswith('connect->')], keys


def test_late_cycles_end_the_call_on_the_wall_clock(tmp_path, monkeypatch, capsys):
    calls, keys = _session(tmp_path, monkeypatch, slowdown=3, cycles=16, retries=16)
    # 16 cycles plus one is 21.25 s on the grid; at three times late that is
    # about 7 cycles of capture clock.
    assert 5 <= len(calls) <= 9, len(calls)
    out = capsys.readouterr().out
    assert 'CALL ENDED BY THE WALL CLOCK' in out
    assert 'CALL ENDED BY ITS CYCLE COUNT' not in out
    assert 'FINAL CALL LISTEN' in out
    assert keys[-1][0].startswith('ID ')


@pytest.mark.parametrize('cycles,retries', [(4, 20), (20, 3)])
def test_a_call_on_the_grid_ends_on_its_count(tmp_path, monkeypatch, capsys,
                                              cycles, retries):
    calls, _ = _session(tmp_path, monkeypatch, slowdown=1, cycles=cycles,
                        retries=retries)
    assert len(calls) == min(cycles, retries) + 1
    out = capsys.readouterr().out
    assert 'CALL ENDED BY ITS CYCLE COUNT' in out
    assert 'CALL ENDED BY THE WALL CLOCK' not in out


def test_maxerror_255_has_no_wall_clock_either(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(onair, '_call_clock', lambda: now[0])
    budget = onair._CallBudget(None, 1.25)
    now[0] = 1e6
    assert not budget.overdue()
    finite = onair._CallBudget(70, 1.25)
    now[0] += 71 * 1.25 - 1e-6
    assert not finite.overdue()
    now[0] += 2e-6
    assert finite.overdue()
