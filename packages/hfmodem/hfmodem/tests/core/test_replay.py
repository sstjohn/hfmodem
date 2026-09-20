# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`examples/replay.toml` runs, and opens nothing.

It is the only way into this project for somebody without an HF rig, and it
advertises three things: no CAT port, no keying line, no sound card. For as long
as the recording had no way to be named it advertised them without running at
all — `hfmodem station examples/replay.toml` went looking for a sound card called
`replay` and stopped there.

So the checks are the three claims plus the one that was missing: the station
comes up on a recording that ships, and every device entry point is booby-trapped
for the duration.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hfhost.config import ConfigError

from hfmodem.core import config
from hfmodem.core.audio import ReplayAudio
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.station.process import Station, recording

REPO = Path(__file__).resolve().parents[5]
EXAMPLE = REPO / "examples" / "replay.toml"


@pytest.fixture
def no_devices(monkeypatch):
    """Anything that reaches for a card fails the test rather than the station.

    Both doors: `devices.find_device` is what `open_audio` would call, and
    `sounddevice` is what would be imported underneath it. A replay that opened a
    card and worked anyway would pass every other assertion here.
    """
    import sys

    from hfmodem.core import devices

    def refuse(*a, **kw):
        raise AssertionError("the replay station went looking for a sound card")

    monkeypatch.setattr(devices, "find_device", refuse)
    monkeypatch.setitem(sys.modules, "sounddevice", None)


def test_the_example_names_a_recording_that_ships():
    """A path into the tests directory, which the distribution carries.

    The capture corpus does not cross into the distribution, so a station file
    pointing at `captures/` would work here and nowhere else — which is the
    failure that reads as "the project is broken" to the one reader who has
    nothing else to run.
    """
    cfg = config.load(EXAMPLE)
    path = recording(cfg.audio)
    assert path is not None, f"{EXAMPLE} no longer replays anything"
    assert path.is_file(), f"{EXAMPLE} names {path}, which is not in the tree"
    assert not cfg.station.transmit


def test_the_station_comes_up_on_the_recording_with_no_device(no_devices):
    cfg = config.load(EXAMPLE)
    station = Station(cfg)
    audio = station.open_audio()
    assert isinstance(audio, ReplayAudio)
    assert len(station.replay) / CARD_RATE_HZ > 1.0, "the recording is empty"

    assert station.open_rig() is None, "a station with transmit = false opened the rig"
    assert list(station.build_lanes()) == ["vara"]
    assert audio.pump(8) == 8

    while not audio.exhausted():
        station._tick()
    assert audio.samples == len(station.replay)
    station.shutdown()


def test_a_recording_that_is_not_there_says_so(tmp_path):
    """Named, rather than an OSError from inside scipy four frames down."""
    missing = tmp_path / "nothing.wav"
    with pytest.raises(ConfigError, match=str(missing)):
        recording(config.Audio(input=f"replay:{missing}"))


def test_a_bare_replay_says_what_to_supply():
    """The form every station file used before the recording could be named."""
    with pytest.raises(ConfigError, match="replay:FILE"):
        recording(config.Audio(input="replay"))


def test_a_device_name_is_still_a_device_name():
    assert recording(config.Audio(input="USB Audio CODEC")) is None
