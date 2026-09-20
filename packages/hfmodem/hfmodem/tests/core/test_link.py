# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The adapter that puts a protocol on the station's radio. No radio involved."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.core import config
from hfmodem.core.audio import ReplayAudio
from hfmodem.core.occupied import keyed_hz
from hfmodem.core.regulatory import Control, Unregulated
from hfmodem.core.rig import Cat, Rig
from hfmodem.station.arbiter import Identity, TxArbiter
from hfmodem.station.link import ProtocolLink, Waveform
from hfmodem.tests.core.fakerig import FakePtt, FakeRigctld

BENCH = Unregulated(because="unit test, no radio")


class _Toy(ProtocolLink):
    waveform = Waveform("ardop", 12000)

    def __init__(self, station, **kw):
        self.heard = []
        super().__init__(station, **kw)

    def decode(self, samples):
        return ["frame"] if len(samples) > 1000 else []

    def on_frame(self, frame):
        self.heard.append(frame)


class _Station:
    """Only what the adapter uses, so the test is about the adapter."""

    def __init__(self, arbiter, rig, cfg):
        self.arbiter, self.rig, self.cfg = arbiter, rig, cfg


@pytest.fixture
def station():
    fake = FakeRigctld()
    ptt = FakePtt()
    fake.ptt_line = ptt
    rig = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=ptt,
              profile=BENCH, control=Control.LOCAL, mycall="N0CALL", transmit=True)
    rig.cat.open(); rig._armed = True; rig._dial_hz = fake.freq
    audio = ReplayAudio(np.zeros(48000 * 4, np.int16))
    arb = TxArbiter(audio, rig, identity=Identity("N0CALL", interval_s=600.0))
    cfg = config.load(_repo() / "examples" / "station.toml")
    yield _Station(arb, rig, cfg)
    rig.close(); fake.close()


def _repo():
    from pathlib import Path
    return Path(__file__).resolve().parents[5]


def _ephemeral(raw: dict) -> dict:
    """Every host port asked for as 0, so two suites at once cannot want the same
    one. What was bound is read back off the server, as `Served.ports`."""
    for p in raw["protocols"].values():
        p.update({k: 0 for k in ("cmd_port", "data_port", "port") if k in p})
    return raw


def test_a_protocol_receives_through_its_lane(station):
    link = _Toy(station)
    from hfmodem.core.audio import Block
    link.lane.push(Block(0.0, 0, np.zeros(48000, np.float32)))
    assert link.poll()
    assert link.heard == ["frame"]


def test_a_protocol_transmits_through_the_arbiter_and_never_keys(station):
    """Four modems each holding their own PTT is the arrangement that voided eight
    on-air sessions. A protocol submits; the arbiter decides."""
    link = _Toy(station)
    end = link.transmit(np.zeros(4800, np.float32), why="test")
    assert end is not None and link.sent == 1
    assert station.arbiter.sent == 1
    assert not hasattr(link, "key"), "the adapter must not expose a way to key"


def test_a_refusal_is_reported_rather_than_raised(station):
    """The station refuses for reasons a protocol has no view of, and the correct
    response to all of them is the same as to a lost frame."""
    station.rig.transmit = False
    link = _Toy(station)
    assert link.transmit(np.zeros(4800, np.float32), why="test") is None
    assert link.sent == 0 and link.refusals


def test_every_burst_describes_itself(station):
    """Rig.key takes an Emission, so a protocol that cannot say what it is about
    to put on the air cannot put anything on the air."""
    link = _Toy(station)
    e = link.waveform.emission(7_100_000)
    assert (e.audio_lo_hz, e.audio_hi_hz) == keyed_hz("ardop")
    assert e.technique == "ardop"


def test_the_passband_is_looked_up_not_restated():
    """Four pairs of literals sat beside these four classes and three were inside
    the emission they described. One table answers it now, so the band the station
    senses before it keys and the band the gate judges as it keys are the same
    numbers rather than two copies that drifted."""
    for name in ("pactor", "vara", "ardop", "sabir"):
        assert Waveform(name, 48000).emission(14_100_000, drift_hz=0).span == (
            (14_100_000 + keyed_hz(name)[0], 14_100_000 + keyed_hz(name)[1]))


def test_a_protocol_with_no_passband_on_file_cannot_describe_a_burst():
    """The gate's fallback is a refusal rather than a guess: an unlisted waveform
    has no band edge that can be checked, and `core.busy.FULL_BAND` -- what the
    sense falls back to -- sits inside the sabir row at both edges."""
    with pytest.raises(ValueError, match="no passband on file"):
        Waveform("toy", 48000).emission(7_100_000)


def test_the_passband_is_stated_not_derived():
    """The station uses Sabir's measured occupied edges, including its skirts."""
    lo, hi = Waveform("sabir", 48000).emission(14_100_000, drift_hz=0).span
    assert (lo, hi) == pytest.approx((14_100_140.0, 14_102_860.0))


def test_a_receive_only_station_refuses_without_an_arbiter(station):
    station.arbiter = None
    link = _Toy(station)
    assert link.transmit(np.zeros(100, np.float32)) is None
    assert "cannot transmit" in link.refusals[0]


def test_besra_binds_at_ardops_own_rate():
    from hfmodem.station.link import BesraLink
    assert BesraLink.waveform.rate == 12000, (
        "ARDOP's 12 kHz is normative; resampling it costs the ardopcf cross-decode")


def test_the_station_builds_a_lane_per_enabled_protocol():
    """Three protocols reading one capture at their own rates, from one card.
    This is the thing the merge was for, and it runs on a recording."""
    import numpy as np
    from hfmodem.station.process import Station
    cfg = config.load(_repo() / "examples" / "station.toml")
    tone = (np.sin(2 * np.pi * 1500 * np.arange(48000) / 48000) * 6000).astype(np.int16)
    st = Station(cfg, replay=tone)
    st.open_audio()
    lanes = st.build_lanes()
    try:
        assert set(lanes) == {"pactor", "vara", "sabir"}, sorted(lanes)
        while st.audio.pump(128):
            pass
        # every lane saw the same capture
        assert all(link.lane._seen > 0 for link in lanes.values())
    finally:
        st.shutdown()


def test_a_lane_that_cannot_be_built_does_not_stop_the_station():
    """besra's adapter needs its modem. A station without one runs the others
    rather than refusing to start — containment, not all-or-nothing."""
    import numpy as np
    from hfmodem.station.process import Station
    import tomllib
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    raw["protocols"]["ardop"]["enabled"] = True
    st = Station(config.parse(raw), replay=np.zeros(4800, np.int16))
    st.open_audio()
    try:
        lanes = st.build_lanes()
        assert "ardop" not in lanes and len(lanes) == 3
    finally:
        st.shutdown()


def test_the_station_serves_each_dialect_on_its_own_port():
    """Four modems present four control surfaces and that is deliberate — an
    application driving VARA expects VARA's sockets. What unifies them is where
    they run, so a station with four enabled is indistinguishable from four
    stations, from the application's side."""
    import socket
    import tomllib
    from hfmodem.station import hosts
    from hfmodem.station.process import Station
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    raw["protocols"]["ardop"]["enabled"] = True
    st = Station(config.parse(_ephemeral(raw)), replay=np.zeros(4800, np.int16))
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    try:
        assert {"vara", "ardop", "sabir"} <= set(served), sorted(served)
        ports = [port for sv in served.values() for port in sv.ports]
        assert len(set(ports)) == len(ports), ports
        for name, sv in served.items():
            for port in sv.ports:
                with socket.create_connection(("127.0.0.1", port), 2.0):
                    pass                    # it accepts, which is the claim
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            if hasattr(link, "stop"):
                link.stop()
        st.audio.close()


def test_a_dialect_that_will_not_start_does_not_stop_the_others():
    """The containment property four separate processes gave for free."""
    import tomllib
    from hfmodem.station import hosts
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    # pactor's dialect is a pty and the station does not own it yet
    served = hosts.build(config.parse(_ephemeral(raw)), log=lambda *_: None)
    try:
        assert "pactor" not in served and "vara" in served
    finally:
        hosts.stop_all(served)


def test_a_lane_that_could_not_bind_does_not_read_as_serving():
    """ardop is the lane that used to bind on a thread of its own, so a port
    already taken never reached `build` — it escaped as a warning and `served`
    went on claiming a dialect nothing was listening on."""
    import tomllib
    from hfmodem.station import hosts
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    raw["protocols"]["ardop"]["enabled"] = True
    holding = hosts.build(config.parse(_ephemeral(raw)), log=lambda *_: None)
    try:
        cmd, data = holding["ardop"].ports
        raw["protocols"]["ardop"].update(cmd_port=cmd, data_port=data)
        second = hosts.build(config.parse(raw), log=lambda *_: None)
        try:
            assert "ardop" not in second, sorted(second)
        finally:
            hosts.stop_all(second)
    finally:
        hosts.stop_all(holding)


def test_a_lane_that_cannot_key_says_so_on_the_line_it_comes_up_on():
    """A loopback is indistinguishable from a radio at the socket, so it has to
    be distinguishable at the log.

    `vara` and `ardop` are served over `LoopbackModem`: an application attaches,
    is answered `CONNECTED`, and reads back its own bytes with nothing keyed.
    Pat cannot tell that from a station on the air — nothing in either dialect
    carries the difference — and the station said nothing either, so the one
    place it can be said is the line the lane comes up on. `Served.keys` is the
    fact and the line is its report; wiring a real modem in has to move both.
    """
    import tomllib
    from hfmodem.station import hosts
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    raw["protocols"]["ardop"]["enabled"] = True
    lines: list[str] = []
    served = hosts.build(config.parse(_ephemeral(raw)), log=lines.append)
    try:
        for name, sv in served.items():
            line = next(ln for ln in lines if ln.startswith(f"{name}: "))
            assert sv.keys is ("NO RADIO" not in line), line
        for name in ("vara", "ardop"):
            assert not served[name].keys, f"{name} claims a transmitter"
    finally:
        hosts.stop_all(served)


def test_a_dialect_with_no_server_is_named_rather_than_skipped():
    """The `pactor` branch returned None into a bare `continue`: no log line, no
    warning, and a station whose SCS port simply was not there. Absent is a
    result, and one an operator waiting for a client to attach has to be told."""
    import tomllib
    from hfmodem.station import hosts
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    lines: list[str] = []
    served = hosts.build(config.parse(_ephemeral(raw)), log=lines.append)
    try:
        assert "pactor" not in served
        said = [ln for ln in lines if ln.startswith("pactor:")]
        assert said and "pty" in said[0], lines
    finally:
        hosts.stop_all(served)


def test_a_server_the_station_cannot_stop_is_one_it_will_not_start():
    """Stopping used to probe for three method names and do nothing when it found
    none, which is indistinguishable from having stopped."""
    from hfmodem.station import hosts

    class _Unstoppable:
        def start_background(self):
            raise AssertionError("started a server that could not be stopped")

    with pytest.raises(TypeError, match="stop"):
        hosts._start(_Unstoppable())


def test_sabir_mounts_a_real_endpoint_on_the_station_air():
    """M5, stated as a property rather than a mood.

    sabir is the one protocol whose session this station also *runs*: the other
    three decode into something driven elsewhere. Its ARQ stack was written
    against a simulated air — two endpoints, one thread, virtual time — so
    binding it here means supplying that same `clock`/`post`/`register` port over
    a real radio, and the proof is that the endpoint registers on it.
    """
    import tomllib
    from hfmodem.station import hosts
    from hfmodem.station.process import Station
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    raw["protocols"] = {"sabir": dict(raw["protocols"]["sabir"], port=0)}
    st = Station(config.parse(raw), replay=np.zeros(4800, np.int16))
    st.open_audio()
    st.build_lanes()
    try:
        air = st.lanes["sabir"].air
        served = hosts.build(st.cfg, station=st, log=lambda *_: None)
        try:
            modem = served["sabir"].server._factory()
            assert air._end is modem, "the endpoint did not register on the air"
            assert modem.air is air
        finally:
            hosts.stop_all(served)
    finally:
        st.lanes["sabir"].stop()
        st.audio.close()


def test_sabir_without_a_station_says_what_is_missing():
    """The factory runs inside a server thread, where a TypeError about a
    positional argument is indistinguishable from the connection simply failing."""
    import tomllib
    from hfmodem.station import hosts
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    served = hosts.build(config.parse(_ephemeral(raw)), log=lambda *_: None)
    try:
        with pytest.raises(RuntimeError, match="needs a station"):
            served["sabir"].server._factory()
    finally:
        hosts.stop_all(served)


def test_a_replay_delivers_what_a_card_delivers():
    """Two audio sources feed the same lanes, so they must agree about scale.

    A recording is 16-bit PCM and the card is float32 in [-1, 1]. Handing out the
    integers put every decoder 32768x hot, and `resample.from_card` scales by
    32768 again on the way to ARDOP's 12 kHz — so besra's lane saw nothing but
    rails, on every replay test that has ever run.
    """
    from hfmodem.core.audio import RollingLane
    tone = np.sin(2 * np.pi * 1500 * np.arange(48000) / 48000) * 0.5
    audio = ReplayAudio((tone * 32767).astype(np.int16))
    lane = audio.subscribe(RollingLane(48000, decode=lambda _: []))
    audio.pump(200)
    buf, _ = lane._gather()
    peak = float(np.abs(buf).max())
    assert 0.45 < peak < 0.55, f"card scale is [-1, 1]; this lane saw {peak}"


def test_a_replay_of_a_float_recording_is_not_silence():
    """`core.wav.read` returns floats, and the previous conversion cast them to
    int16 — so replaying a recording read the documented way produced silence,
    and every lane politely decoded nothing."""
    from hfmodem.core.audio import RollingLane
    tone = np.sin(2 * np.pi * 1500 * np.arange(48000) / 48000) * 0.5
    audio = ReplayAudio(tone)
    lane = audio.subscribe(RollingLane(48000, decode=lambda _: []))
    audio.pump(200)
    buf, _ = lane._gather()
    assert float(np.abs(buf).max()) > 0.4, "a float recording replayed as silence"
