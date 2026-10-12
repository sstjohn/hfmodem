# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ALE as a station lane: config, the host port, the rig it retunes."""
from __future__ import annotations

import socket
import time
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.bateleur.ale import fsk, words as W
from hfmodem.bateleur.ale.host import AleHost
from hfmodem.core import band, config
from hfmodem.core.audio import ReplayAudio
from hfmodem.core.config import ProtocolConfig
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.resample import for_card
from hfmodem.core.rig import Cat, Rig
from hfmodem.station import hosts
from hfmodem.station.arbiter import Identity, TxArbiter
from hfmodem.station.link import AleLink
from hfmodem.station.process import Station
from hfmodem.tests.core.fakerig import FakePtt, FakeRigctld
from hfmodem.tests.core.test_link import BENCH, _repo

CHANNELS = (7_102_000, 10_145_500, 14_109_000)


class _Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.s.settimeout(0.05)
        self.buf = b""
        self.seen: list[str] = []

    def lines(self):
        try:
            self.buf += self.s.recv(65536)
        except socket.timeout:
            pass
        out = []
        while b"\r\n" in self.buf:
            ln, self.buf = self.buf.split(b"\r\n", 1)
            out.append(ln.decode())
        self.seen += out
        return out

    def ask(self, line, timeout=10.0):
        self.s.sendall(line.encode() + b"\r\n")
        return self.wait(lambda ln: ln.startswith(("OK", "ERR")), timeout)

    def wait(self, pred, timeout=30.0, pump=None):
        t0 = time.monotonic()
        start = len(self.seen)
        while time.monotonic() - t0 < timeout:
            if pump:
                pump()
            self.lines()
            for ln in self.seen[start:]:
                if pred(ln):
                    return ln
        raise AssertionError(f"timed out; heard {self.seen}")

    def close(self):
        self.s.close()


def _ale(**over):
    base = dict(enabled=True, host="ale", port=0, addresses=("W9SSJ",), channels=CHANNELS[:1])
    base.update(over)
    return ProtocolConfig(**base)


def _cfg(**over):
    cfg = config.load(_repo() / "examples" / "station.toml")
    return replace(cfg, protocols={"ale": _ale(**over)},
                   station=replace(cfg.station, transmit=False))


def test_the_station_knows_it_and_checks_its_config():
    base = __import__("tomllib").loads((_repo() / "examples" / "station.toml").read_text())
    base["protocols"] = {"ale": {"enabled": True, "host": "ale", "port": 8330,
                                 "addresses": ["W9SSJ", "HFL"], "channels": list(CHANNELS)}}
    cfg = config.parse(base)
    assert cfg.protocols["ale"].addresses == ["W9SSJ", "HFL"]
    net = dict(base["protocols"]["ale"], nets=[{"address": "NET", "member": "HFL", "slot": 2}])
    assert config.parse(dict(base, protocols={"ale": net})).protocols["ale"].nets[0]["slot"] == 2
    for bad, why in (({"addresses": []}, "addresses"), ({"addresses": ["W9 SSJ"]}, "address"),
                     ({"channels": []}, "channels"), ({"channels": [7102]}, "HF dial"),
                     ({"scan_rate": 50}, "scan_rate"),
                     ({"nets": [{"address": "NET", "member": "JOE", "slot": 1}]}, "member"),
                     ({"nets": [{"address": "NET", "member": "W9SSJ", "slot": 0}]}, "slot"),
                     ({"nets": ["NET"]}, "nets"),
                     ({"lp_level": 3}, "lp_level"),
                     ({"lp_level": 2, "lp_key": "abc"}, "lp_key")):
        raw = dict(base["protocols"]["ale"], **bad)
        base2 = dict(base, protocols={"ale": raw})
        with pytest.raises(config.ConfigError, match=why):
            config.parse(base2)


def test_a_sound_heard_on_the_air_reaches_the_host_port():
    air = fsk.modulate(W.sound("SAM", repeats=3))
    rng = np.random.default_rng(1)
    air = np.concatenate([np.zeros(4000), air, np.zeros(8 * fsk.FS)])
    air = air + rng.normal(0, 0.05, air.size)
    card = for_card(air * 0.3, fsk.FS)
    st = Station(_cfg(), replay=card)
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    try:
        srv = served["ale"]
        assert srv.keys is False
        app = _Client(*srv.ports)
        assert app.ask("VERSION").startswith("OK bateleur-ale")

        def pump():
            if not st.replay_exhausted():
                st._tick()
        app.wait(lambda ln: ln.startswith("HEARD SAM"), pump=pump)
        assert app.ask("STATUS").startswith("OK STATUS AVAILABLE")
        app.close()
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            link.stop()


@pytest.fixture
def keyed():
    """An ALE lane on a fake rig and a real arbiter."""
    fake = FakeRigctld()
    ptt = FakePtt()
    fake.ptt_line = ptt
    rig = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=ptt,
              profile=BENCH, control=config.Control.LOCAL, mycall="N0CALL",
              transmit=True, max_key_s=30.0)
    rig.cat.open(); rig._armed = True; rig._dial_hz = fake.freq
    audio = ReplayAudio(np.zeros(CARD_RATE_HZ * 2, np.float32))
    arb = TxArbiter(audio, rig, identity=Identity("N0CALL", interval_s=600.0))

    class _St:
        pass
    st = _St()
    st.lanes = {}
    st.arbiter, st.rig, st.audio = arb, rig, audio
    st.cfg = _cfg(channels=CHANNELS)
    st.cfg = replace(st.cfg, rig=replace(st.cfg.rig, max_key_s=30.0))
    link = AleLink(st)
    link.start()
    yield link, audio, fake
    link.stop()
    rig.close(); fake.close()


def _decode(audio_card) -> list:
    from hfmodem.core.resample import from_card
    x = from_card(np.asarray(audio_card, np.float32), fsk.FS).astype(float)
    r = fsk.DualReceiver()
    out = []
    x = np.concatenate([x, np.zeros(2 * fsk.FS)])
    for i in range(0, x.size, 800):
        out += r.push(x[i:i + 800])
    return [o.frame for o in out if isinstance(o, fsk.Received)]


def test_a_call_from_the_host_port_keys_through_the_arbiter_on_a_listed_channel(keyed):
    link, audio, fake = keyed
    host = AleHost(link.modem, port=0)
    host.start_background()
    try:
        app = _Client(host.port)
        assert app.ask("CHANNEL 2").startswith("OK CHANNEL 2")
        # The rig is told the centre; the dial it reports is the listed channel.
        t0 = time.monotonic()
        while fake.freq != CHANNELS[2] and time.monotonic() - t0 < 5:
            link.modem.feed(int(audio.sample_now()), np.zeros(4800, np.float32))
            time.sleep(0.05)
        assert fake.freq == CHANNELS[2]
        assert app.ask("CALL JOE hello").startswith("OK CALL JOE")

        # A receiver's noise, not silence: the occupancy sense reads a muted
        # input as busy, since no evidence is not evidence of a clear channel.
        rng = np.random.default_rng(3)

        def pump():
            link.modem.feed(link.modem._now_index,
                            rng.normal(0, 0.02, 4800).astype(np.float32))
        app.wait(lambda ln: ln.startswith("CALL FAILED JOE"), timeout=60, pump=pump)
        app.close()
    finally:
        host.stop()
    assert audio.transmitted, f"nothing was keyed: {link.refusals} {app.seen}"
    frames = _decode(audio.transmitted[0][1])
    assert any(f.to == ["JOE"] and f.caller == "W9SSJ" and f.amd == ["HELLO"] for f in frames)
    assert band.DIAL_OFFSET_HZ == 1500


def test_the_host_dialect_makes_one_to_many_calls():
    from hfmodem.bateleur.ale.controller import Config
    from hfmodem.bateleur.ale.modem import AleModem
    sent = []
    m = AleModem(Config(["SAM"], ["A"], scanning=False, listen_s=0.0),
                 send=lambda audio, fs: sent.append(audio) or 0)
    m.start()
    host = AleHost(m, port=0)
    try:
        assert host.command("CALL joe,bob hello").startswith("OK CALL JOE,BOB")
        assert m.ctl._group == ["JOE", "BOB"] and m.ctl._slots == ["JOE", "BOB"]
        m.post(lambda: m.ctl._set_state(m.ctl.AVAILABLE))
        assert host.command("NET NET1 joe,bob").startswith("OK NET NET1")
        assert m.ctl._group == ["NET1"] and m.ctl._slots == ["JOE", "BOB"]
        m.post(lambda: m.ctl._set_state(m.ctl.AVAILABLE))
        assert host.command("ANYCALL E") == "OK ANYCALL E"
        assert m.ctl.peer == "@@E"
        m.post(lambda: m.ctl._set_state(m.ctl.AVAILABLE))
        assert host.command("ALLCALL /M net check") == "OK ALLCALL M"
        # Answered on the modem's thread after the AllCall's events went out.
        assert host.command("ANYCALL XYZ").startswith("ERR")
        assert any(ln.startswith("SENT @M@ CHANNEL A allcall: NET CHECK") for ln in host.log)
    finally:
        m.stop()


@pytest.mark.parametrize("occupied", [True, False])
def test_other_traffic_on_the_link_holds_off_twa(occupied):
    """A.5.5.3.5: a link ends after Twa without activity; a channel carrying
    someone's traffic (or another protocol's bursts) is activity."""
    from hfmodem.bateleur.ale.controller import Config
    from hfmodem.bateleur.ale.modem import AleModem
    m = AleModem(Config(["SAM"], ["A"], scanning=False, listen_s=0.0, activity_s=5.0),
                 send=lambda audio, fs: 0, busy=lambda x: occupied, card_rate=CARD_RATE_HZ)
    m.ctl.linked.add("JOE")
    m.ctl._set_state(m.ctl.LINKED)
    rng = np.random.default_rng(0)
    n = CARD_RATE_HZ // 10
    for i in range(100):                                   # 10 s
        m._audio(i * n, rng.normal(0, 0.02, n).astype(np.float32))
    assert (m.ctl.state == "linked") is occupied


def test_another_protocols_burst_is_link_activity(keyed):
    link, audio, fake = keyed
    link.modem.post(lambda: (link.modem.ctl.linked.add("JOE"),
                             link.modem.ctl._set_state("linked")))
    link.station.arbiter.last_sent["vara"] = 5 * CARD_RATE_HZ
    link.poll()
    box = []
    link.modem.post(lambda: box.append(link.modem.ctl._last_activity))
    t0 = time.monotonic()
    while not box and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert box == [5.0]


def test_linking_protection_reaches_the_controller_with_the_dial_frequencies():
    class _St:
        pass
    st = _St()
    st.lanes = {}
    st.cfg = _cfg(channels=CHANNELS, lp_level=2, lp_key="c2284a1ce7be2f")
    st.cfg = replace(st.cfg, station=replace(st.cfg.station, regulatory="unregulated",
                     regulatory_settings={"because": "bench"}))
    st.rig = st.audio = st.arbiter = None
    link = AleLink(st)
    c = link.modem.ctl
    assert c.lp is not None and c.lp.pi_s == 2 and c.lp.channel_hz == [float(f) for f in CHANNELS]
    host = AleHost(link.modem, port=0)
    link.modem.start()
    try:
        assert " LP AL-2 TIME Q3" in host.command("STATUS")
    finally:
        link.modem.stop()


def test_the_amateur_networks_conventions_reach_the_controller():
    class _St:
        pass
    st = _St()
    st.lanes = {}
    st.cfg = _cfg(channels=CHANNELS, conventions="hflink", base=True,
                  pilot_channels=(14_109_000,),
                  nets=({"address": "RED", "member": "W9SSJ", "slot": 2},))
    st.rig = st.audio = st.arbiter = None
    c = AleLink(st).modem.ctl
    assert c.cfg.activity_s == 360 and c.cfg.sound_terminator == W.TWAS
    assert c.cfg.nets["HFN"] == ("W9SSJ", 1) and c.cfg.nets["RED"] == ("W9SSJ", 2)
    assert c._sound_due[2] <= 3600 and c._sound_due[0] == float("inf")
    assert c.cfg.channel_hz == [float(f) for f in CHANNELS]


def test_a_gpr_from_the_host_port_is_a_twas_broadcast():
    class _St:
        pass
    st = _St()
    st.lanes = {}
    st.cfg = _cfg(conventions="hflink")
    st.rig = st.audio = st.arbiter = None
    link = AleLink(st)
    host = AleHost(link.modem, port=0)
    link.modem.start()
    try:
        reply = host.command("GPR 43.0731 -89.4012 270 all well")
    finally:
        link.modem.stop()
    assert reply.startswith("OK GPR GPR*W9SSJ*43N073100*089W401200*270M*") \
        and reply.endswith("*ALL WELL")
    assert link.modem.ctl._broadcast and link.modem.ctl.peer == "GPR"
    for bad in ("conventions", "pilot_channels"):
        base = __import__("tomllib").loads((_repo() / "examples" / "station.toml").read_text())
        raw = {"enabled": True, "host": "ale", "port": 8330, "addresses": ["W9SSJ"],
               "channels": list(CHANNELS)}
        raw[bad] = "arrl" if bad == "conventions" else [3_596_000]
        with pytest.raises(config.ConfigError, match=bad):
            config.parse(dict(base, protocols={"ale": raw}))


def test_response_turnaround_through_replay_arbiter():
    from types import SimpleNamespace
    from hfmodem.bateleur.ale.controller import Config, Tw
    from hfmodem.bateleur.ale.modem import AleModem
    from hfmodem.core.regulatory import centred
    from hfmodem.station.arbiter import LIVE, TxRequest

    incoming = for_card(fsk.modulate(W.call("SAM", "JOE", scan_words=2)), fsk.FS)
    card = np.r_[incoming, np.zeros(CARD_RATE_HZ)]
    audio = ReplayAudio(card)
    first_tones = []
    arm = audio.arm_burst
    def remember(wave, *, at):
        actual = arm(wave, at=at)
        first_tones.append(actual / CARD_RATE_HZ)
        return actual
    audio.arm_burst = remember
    rig = SimpleNamespace(max_key_s=30, key=lambda *a, **kw: 1,
                          unkey=lambda token: None, retired=False)
    arb = TxArbiter(audio, rig)
    def send(wave, rate):
        return arb.submit(TxRequest(LIVE, 0, "ale", wave, rate,
                                    centred(7_102_000, 2500), settle_s=.04, responding=True))
    modem = AleModem(Config(["SAM"], ["A"], scanning=False), send=send)
    for at in range(0, len(card), 960):
        chunk = card[at:at + 960]
        audio.samples = at + len(chunk)
        modem._audio(at, chunk)
        if first_tones:
            break
    assert first_tones
    delay = first_tones[0] - len(incoming) / CARD_RATE_HZ
    assert 0 <= delay <= 5 * Tw
    print(f"ALE replay turnaround: {delay:.6f} s; 5 Tw: {5 * Tw:.6f} s")
