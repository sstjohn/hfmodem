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
from hfmodem.core.virtualrig import VirtualRig
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


def test_the_example_names_a_recording_that_ships(monkeypatch):
    """A path into the tests directory, which the distribution carries.

    The capture corpus does not cross into the distribution, so a station file
    pointing at `captures/` would work here and nowhere else — which is the
    failure that reads as "the project is broken" to the one reader who has
    nothing else to run.
    """
    monkeypatch.chdir(REPO)
    cfg = config.load(EXAMPLE)
    path = recording(cfg.audio)
    assert path is not None, f"{EXAMPLE} no longer replays anything"
    assert path.is_file(), f"{EXAMPLE} names {path}, which is not in the tree"
    assert not cfg.station.transmit


def test_the_station_comes_up_on_the_recording_with_no_device(no_devices, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = config.load(EXAMPLE)
    station = Station(cfg)
    audio = station.open_audio()
    assert isinstance(audio, ReplayAudio)
    assert len(station.replay) / CARD_RATE_HZ > 1.0, "the recording is empty"

    assert isinstance(station.open_rig(), VirtualRig)
    assert not station.rig.transmit and station.rig.cat is station.rig.ptt is None
    assert list(station.build_lanes()) == ["vara"]
    assert station.failed == []
    from hfmodem.station.link import KestrelLink
    assert isinstance(station.lanes["vara"], KestrelLink)
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


def test_replay_strict_placement_reservation_and_cancellation():
    import numpy as np
    from hfmodem.core.audio import BurstInvalidated
    audio = ReplayAudio(np.zeros(10), blocksize=4)
    audio.wait_until(7, healthy=True)
    wave = np.ones(3)
    with pytest.raises(BurstInvalidated, match='passed'):
        audio.arm_burst(wave, 6, strict_at=True)
    assert audio.arm_burst(wave, 6) == 7
    audio.cancel_burst()
    assert audio.burst_done() and audio.transmitted == []
    with pytest.raises(BurstInvalidated, match='deadline') as error:
        audio.reserve_burst(wave, 5, 4809)
    assert error.value.displacement == 2
    assert audio.reserve_burst(wave, 5, 4810) == 7
    assert not audio.burst_done()
    audio.wait_until_dac(10, healthy=True)
    assert audio.burst_done()
    assert audio.finish_burst() == 10
    np.testing.assert_array_equal(audio.transmitted[0][1], wave)


def test_replay_wait_cancellation_deadline_and_eof():
    import numpy as np
    from hfmodem.core.audio import BurstInvalidated
    audio = ReplayAudio(np.zeros(3), blocksize=2)
    with pytest.raises(BurstInvalidated, match='cancelled'):
        audio.wait_until_dac(20, cancelled=lambda: True)
    with pytest.raises(BurstInvalidated, match='deadline'):
        audio.wait_until(20, deadline=0)
    assert audio.samples == 0
    audio.wait_until_dac(20)
    assert audio.samples == 20 and audio.exhausted()
    assert audio.pump() == 0
    audio.check_callbacks()


def test_replay_pump_drains_stream_and_cancel_discards_recording():
    import numpy as np
    audio = ReplayAudio(np.zeros(20), blocksize=4)
    stream = audio.arm_stream(2, 10)
    stream.write(np.ones(6))
    stream.close()
    assert not audio.burst_done()
    assert audio.pump(2) == 2
    assert stream.emitted == 6 and stream.end == 8 and audio.burst_done()
    audio.cancel_burst()
    audio.finish_burst()
    assert audio.transmitted == []


@pytest.mark.parametrize('arm', ['burst', 'stream'])
def test_replay_tape_end_waits_for_pending_transmission(arm):
    import numpy as np
    audio = ReplayAudio(np.zeros(8), blocksize=4)
    if arm == 'burst':
        audio.arm_burst(np.ones(20), 0)
    else:
        audio.arm_stream(0, 20)
    audio.pump(4)
    assert audio.samples == 8 and not audio.exhausted()
    audio.wait_until(20)
    audio.finish_burst()
    assert audio.exhausted()


def test_replay_reserved_burst_cannot_report_early_completion():
    import numpy as np
    from hfmodem.core.audio import BurstInvalidated
    audio = ReplayAudio(np.zeros(100))
    audio.reserve_burst(np.ones(10), 20, 5000)
    audio.wait_until_dac(29)
    with pytest.raises(BurstInvalidated, match='not completed'):
        audio.finish_burst()
    assert audio.transmitted == [] and audio.burst_done()


def test_replay_starvation_keeps_first_dry_index_until_refilled():
    import numpy as np
    audio = ReplayAudio(np.zeros(40), blocksize=4)
    stream = audio.arm_stream(0, 20)
    stream.write(np.ones(2))
    audio.pump()
    assert stream.starved_at == 2
    audio.pump(3)
    assert stream.starved_at == 2
    stream.write(np.ones(4))
    audio.pump()
    assert stream.starved_at is None
    audio.pump()
    assert stream.starved_at == 20


def test_failed_reserved_replay_burst_flushes_lanes():
    import numpy as np
    from unittest.mock import Mock
    from hfmodem.core.audio import BurstInvalidated
    audio = ReplayAudio(np.zeros(100))
    lane = audio.subscribe(Mock())
    audio.reserve_burst(np.ones(10), 20, 5000)
    audio.wait_until(29)
    with pytest.raises(BurstInvalidated, match='not completed'):
        audio.finish_burst()
    lane.flush_to.assert_called_once_with(30)
    assert not audio.transmitted


@pytest.mark.parametrize('operation', ['pump', 'wait', 'cancel', 'finish', 'arm'])
def test_replay_mutations_wait_for_audio_lock(operation):
    import threading
    import numpy as np
    audio = ReplayAudio(np.zeros(100), blocksize=4)
    started, done = threading.Event(), threading.Event()
    actions = {'pump': audio.pump, 'wait': lambda: audio.wait_until(8),
               'cancel': audio.cancel_burst, 'finish': audio.finish_burst,
               'arm': lambda: audio.arm_burst(np.ones(4), 0)}
    def run():
        started.set()
        actions[operation]()
        done.set()
    with audio._lock:
        worker = threading.Thread(target=run)
        worker.start()
        assert started.wait(2)
        blocked = not done.wait(.1)
    worker.join(2)
    assert blocked, 'mutation bypassed the audio lock'
    assert done.is_set()


def test_replay_wait_releases_lock_between_blocks_for_pump_and_poll():
    import threading
    import numpy as np
    from unittest.mock import Mock
    audio = ReplayAudio(np.zeros(100), blocksize=4)
    lane = audio.subscribe(Mock())
    waiting, resume, pumped = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def cancelled():
        if audio.samples == 4:
            waiting.set()
            if not resume.wait(2):
                errors.append('pump was blocked by wait_until')
        return False
    waiter = threading.Thread(target=lambda: audio.wait_until(12, cancelled=cancelled))
    def poll():
        audio.pump()
        lane.poll()
        pumped.set()
    waiter.start()
    assert waiting.wait(2)
    poller = threading.Thread(target=poll)
    poller.start()
    try:
        progressed = pumped.wait(1)
    finally:
        resume.set()
        waiter.join(3)
        poller.join(3)
    assert progressed and not errors
    lane.poll.assert_called_once()
    assert audio.samples == 12
