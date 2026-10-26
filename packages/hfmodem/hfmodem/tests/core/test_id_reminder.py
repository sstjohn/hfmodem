# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The identification reminder's clock and profile selection.

`Rig._id_last` once started at the Unix epoch, making the second key immediately
appear overdue and repeating the reminder on every later key. These tests check
elapsed time and that the selected `unregulated` profile disables the reminder.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from hfmodem.core import config
from hfmodem.tests.kestrel.fake_line_keyer import FakeLineKeyer

_REPO = Path(__file__).resolve().parents[5]
_TOOLS = _REPO / "tools"
_EXAMPLES = _REPO / "examples"
sys.path.insert(0, str(_TOOLS))

# `vara_rig_bridge` and the example station files ship from the repo rather than
# the wheel, so an installed-wheel run reaches neither. Named rather than skipped
# silently: this is the scheduler that sends the station's identification.
if not (_TOOLS / "vara_rig_bridge.py").exists():
    pytest.skip(f"{_TOOLS}/vara_rig_bridge.py is not in this tree",
                allow_module_level=True)

import vara_rig_bridge as bridge     # noqa: E402


@pytest.fixture
def clock(monkeypatch):
    """The bridge's `time`, with the hands under the test — a ten-minute session
    in no seconds at all."""

    class _Clock:
        now = 1_700_000_000.0

        def time(self) -> float:
            return self.now

        def __getattr__(self, name):
            return getattr(time, name)

    held = _Clock()
    monkeypatch.setattr(bridge, "time", held)
    return held


@pytest.fixture
def rig(monkeypatch):
    """An armed `Rig` with a fake line, under a named station file or
    none, keeping its log."""
    monkeypatch.setattr(bridge, "LineKeyer", FakeLineKeyer)
    made = []

    def make(station: Path | None = None):
        if station is None:
            monkeypatch.delenv(config.STATION_ENV, raising=False)
        else:
            monkeypatch.setenv(config.STATION_ENV, str(station))
        logged: list[str] = []
        r = bridge.Rig(None, armed=True, log=logged.append,
                       line_ptt=True, ptt_device="/dev/fake")
        r.logged = logged
        made.append(r)
        return r

    yield make
    for r in made:
        r.shutdown()


def _reminders(r) -> list[str]:
    return [m for m in r.logged if "station identification reminder is due" in m]


def test_the_reminder_waits_for_the_full_interval(rig, clock):
    """Five seconds into a session is not ten minutes into one."""
    r = rig()
    r._note_identification_due()
    clock.now += 5.0
    r._note_identification_due()
    assert _reminders(r) == []

    clock.now += bridge._ID_INTERVAL_S - 5.0
    r._note_identification_due()
    assert len(_reminders(r)) == 1
    assert "10 min" in _reminders(r)[0]


def test_the_reminder_does_not_repeat_until_the_next_interval(rig, clock):
    r = rig()
    r._note_identification_due()
    clock.now += bridge._ID_INTERVAL_S
    r._note_identification_due()
    clock.now += 30.0
    r._note_identification_due()
    assert len(_reminders(r)) == 1

    clock.now += bridge._ID_INTERVAL_S
    r._note_identification_due()
    assert len(_reminders(r)) == 2


def test_a_station_that_declared_unregulated_is_not_quoted_part_97(rig, clock):
    """The selected unregulated profile disables the identification reminder."""
    r = rig(_EXAMPLES / "replay.toml")
    r._note_identification_due()
    for _ in range(4):
        clock.now += bridge._ID_INTERVAL_S
        r._note_identification_due()
    assert _reminders(r) == []


def test_a_part_97_station_is_reminded(rig, clock):
    r = rig(_EXAMPLES / "station.toml")
    r._note_identification_due()
    clock.now += bridge._ID_INTERVAL_S
    r._note_identification_due()
    assert len(_reminders(r)) == 1


def test_an_unreadable_station_file_says_so_and_keeps_the_reminder(rig, clock,
                                                                  tmp_path):
    """Not knowing the selected profile is not a reason to go quiet about the clock."""
    r = rig(tmp_path / "nothing.toml")
    r._note_identification_due()
    clock.now += bridge._ID_INTERVAL_S
    r._note_identification_due()
    assert len(_reminders(r)) == 1
    assert any(config.STATION_ENV in m for m in r.logged)
