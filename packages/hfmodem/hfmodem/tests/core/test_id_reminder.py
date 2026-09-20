# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The §97.119 reminder every armed tool prints, and the two things it has to
get right: the clock, and whose rules it is quoting.

`Rig._id_last` started at the Unix epoch, so on the second key of a session the
comparison was `time.time() - 0 >= 600` — 1.8 billion seconds past due — and the
station was told "0 min of transmitting: station identification is due" within
seconds of its first over, then again on every key thereafter. A reminder that
arrives whatever the clock says is a reminder an operator learns to read past,
which is the one failure mode a ten-minute rule cannot afford.

The same line printed at a station whose file declares `regulatory =
"unregulated"` — a bench, a dummy load, or a regime hfmodem does not model. That
station is answerable for its emissions under rules that are not Part 97, and a
US section number is noise to it.

Off air entirely: no rigctld, no serial line, no sound card. The reminder is
driven through `Rig._note_identification_due` against a clock the test holds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from hfmodem.core import config

_REPO = Path(__file__).resolve().parents[5]
_TOOLS = _REPO / "tools"
_EXAMPLES = _REPO / "examples"
sys.path.insert(0, str(_TOOLS))

# `vara_rig_bridge` and the example station files ship from the repo rather than
# the wheel, so an installed-wheel run reaches neither. Named rather than skipped
# silently: this is the scheduler that speaks for the licensee.
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
    """An armed `Rig` with no daemon behind it, under a named station file or
    none, keeping its log."""
    made = []

    def make(station: Path | None = None):
        if station is None:
            monkeypatch.delenv(config.STATION_ENV, raising=False)
        else:
            monkeypatch.setenv(config.STATION_ENV, str(station))
        logged: list[str] = []
        r = bridge.Rig(None, armed=True, log=logged.append)
        r.logged = logged
        made.append(r)
        return r

    yield make
    for r in made:
        r.retire()


def _reminders(r) -> list[str]:
    return [m for m in r.logged if "97.119" in m]


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
    """`regulatory = "unregulated"` is the operator saying Part 97 is not the
    regime these emissions are made under."""
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
    """Not knowing the regime is not a reason to go quiet about the clock."""
    r = rig(tmp_path / "nothing.toml")
    r._note_identification_due()
    clock.now += bridge._ID_INTERVAL_S
    r._note_identification_due()
    assert len(_reminders(r)) == 1
    assert any(config.STATION_ENV in m for m in r.logged)
