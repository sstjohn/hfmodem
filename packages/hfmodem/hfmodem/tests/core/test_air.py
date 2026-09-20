# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""sabir's session over the station's audio path. No radio, no simulator.

Everything sabir's ARQ stack has ever run over is a simulated air: two endpoints
on one thread, a transmission handed straight from one outbox to the other's
`on_air`, virtual time. That arrangement cannot be wrong about where a burst
starts, because nothing has to find it.

These tests take the handoff out. A peer's burst is written into a recording, the
recording is replayed through the station exactly as a sound card would deliver
it, and the endpoint has to reach the same conclusion from a stream that is mostly
noise — through the lane, the segmenter and the burst bracket. That is the whole
of what M5 means.
"""
from __future__ import annotations

import time
import tomllib
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core import config
from hfmodem.sabir.arq.fsm import ArqConfig, SessionState
from hfmodem.sabir.arq.modem import LinkModem
from hfmodem.sabir.host import SabirModem
from hfmodem.sabir.offair import to_real
from hfmodem.station.process import Station

CALL = "W9SSJ"
PEER = "W1AW"
FS = 48000


def _repo() -> Path:
    return Path(__file__).resolve().parents[5]


def _peer_burst(*, connect_to: str = CALL) -> np.ndarray:
    """What another sabir station would put on the air, calling us."""
    a = LinkModem(ArqConfig(callsign=PEER), clock=lambda: 0.0)
    a.fsm.on_host_connect(connect_to)
    assert a.outbox, "the peer composed nothing to transmit"
    # The outbox is the analytic signal; a transmitter emits its real part,
    # and a recording is what a receiver would have captured.
    return to_real(np.concatenate(a.outbox))


def _recording(burst: np.ndarray, *, snr_scale: float = 0.25,
               noise: float = 0.01, lead_s: float = 2.0,
               tail_s: float = 2.0) -> np.ndarray:
    """A burst inside a channel, at card scale.

    The lead is not padding: the segmenter's floor is learned from idle frames,
    and a recording that opens mid-burst teaches it that a burst is the floor.
    """
    rng = np.random.default_rng(20260730)
    n = int(lead_s * FS) + burst.size + int(tail_s * FS)
    x = rng.normal(0.0, noise, n)
    at = int(lead_s * FS)
    x[at:at + burst.size] += burst * snr_scale
    return np.clip(x, -1.0, 1.0)


def _station(recording: np.ndarray) -> Station:
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    raw["station"]["mycall"] = CALL
    raw["protocols"] = {"sabir": dict(raw["protocols"]["sabir"], enabled=True)}
    st = Station(config.parse(raw), replay=recording)
    st.open_audio()
    st.build_lanes()
    return st


def _connected(air) -> bool:
    return (air._end is not None
            and air._end.link.fsm.state == SessionState.CONNECTED)


def _run(st: Station, *, seconds: float = 10.0, until=_connected) -> None:
    """Replay to exhaustion, then let the air thread finish with it.

    `until` is what the caller is actually waiting for. Returning on the session
    alone raced every consequence of it: the answer the endpoint composes is
    submitted on the air thread *after* the state changes, so a test asserting on
    the answer failed two runs in five.
    """
    while st.audio.pump(64):
        st.lanes["sabir"].poll()
    deadline = time.monotonic() + seconds
    air = st.lanes["sabir"].air
    while time.monotonic() < deadline:
        st.lanes["sabir"].poll()
        if until(air):
            return
        time.sleep(0.02)


@pytest.fixture
def sabir_station():
    st = _station(_recording(_peer_burst()))
    air = st.lanes["sabir"].air
    modem = SabirModem(air, ArqConfig(callsign=CALL))
    done = []
    air.post(lambda: (modem.link.fsm.on_host_listen(True), done.append(1)))
    for _ in range(200):
        if done:
            break
        time.sleep(0.01)
    assert done, "the air thread never ran the posted command"
    yield st, air, modem
    st.lanes["sabir"].stop()
    st.audio.close()


def test_the_segmenter_finds_a_burst_in_a_recording_of_one(sabir_station):
    """Before anything about decoding: was the burst even located?

    Under simulation this question does not exist, which is why it is the first
    thing M5 has to answer. A segmenter that never opens and one that never closes
    both look like a modem that heard nothing.
    """
    st, air, _ = sabir_station
    _run(st)
    assert air.bursts == 1, f"segmented {air.bursts} bursts, expected exactly one"


def test_a_peer_connects_through_the_station_audio_path(sabir_station):
    """The claim M5 exists to make: sabir's session runs on real audio.

    The peer's CONNECT is written into a recording and replayed through the card
    interface. Nothing hands the burst over — the lane delivers a stream, the
    segmenter brackets it, and the endpoint reaches CONNECTED or it does not.
    """
    st, air, modem = sabir_station
    _run(st)
    assert modem.link.fsm.state == SessionState.CONNECTED, (
        f"state {modem.link.fsm.state} after {air.bursts} burst(s)")
    assert modem.link.fsm.peer == PEER


def test_the_station_answers_and_the_arbiter_is_what_refuses(sabir_station):
    """A connected endpoint composes a reply, and a receive-only station must not
    put it on the air. The refusal belongs to the arbiter rather than to sabir:
    the protocol's response to a refused burst is its response to a lost one, so a
    modem that knew it could not transmit would take a different path than the one
    that runs on air."""
    st, air, modem = sabir_station
    _run(st, until=lambda a: _connected(a) and a.refused >= 1)
    assert modem.link.fsm.state == SessionState.CONNECTED
    assert air.refused >= 1, "the endpoint never tried to answer"
    assert air.sent == 0, "a station with transmit disabled sent something"


def test_a_recording_of_noise_produces_no_session(sabir_station):
    """The negative that makes the positive mean anything.

    A segmenter with a low enough threshold finds bursts in noise, and a decoder
    that accepts anything reports a peer that is not there. Both failures pass the
    test above.
    """
    st, air, modem = sabir_station
    st.audio.samples_in = _recording(np.zeros(0), noise=0.01, lead_s=6.0)
    _run(st, seconds=2.0, until=lambda a: False)   # nothing should happen
    assert modem.link.fsm.state == SessionState.LISTENING
    assert air.bursts == 0, f"the segmenter opened {air.bursts} times on noise"
