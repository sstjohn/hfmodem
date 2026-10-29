# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""bateleur as a modem on the station: its lane, its host ports, the arbiter.

No radio: the station runs on replayed audio, and what it would have keyed is
recorded by the replay and decoded back. That covers both directions through
the same code an on-air station runs — the host port, the modem's thread, the
arbiter's emission check — with the transmitter the one thing missing.
"""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.tests.core.test_host_teardown import loopback_bind as loopback_bind

from hfmodem.bateleur import mil110, s4285, sim, tone16, tone39
from hfmodem.bateleur.host import BateleurHost
from hfmodem.bateleur.modem import BateleurModem, Mode
from hfmodem.core import config
from hfmodem.core.audio import ReplayAudio
from hfmodem.core.config import ProtocolConfig
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.resample import for_card
from hfmodem.core.rig import Cat, Rig
from hfmodem.station import hosts
from hfmodem.station.arbiter import Identity, TxArbiter
from hfmodem.station.link import BateleurLink
from hfmodem.station.process import Station
from hfmodem.tests.core.fakerig import FakePtt, FakeRigctld
from hfmodem.tests.core.test_link import BENCH, _repo

MSG = b"CQ CQ DE N0CALL bateleur on the station " * 4


class _Client:
    """An application on the two ports."""

    def __init__(self, cmd_port, data_port):
        self.cmd = socket.create_connection(("127.0.0.1", cmd_port), timeout=5)
        self.data = socket.create_connection(("127.0.0.1", data_port), timeout=5)
        self.cmd.settimeout(0.05)
        self.data.settimeout(0.05)
        self._lines = b""
        self.received = b""
        self._queue = []
        self.seen = []

    def ask(self, line):
        self.cmd.sendall(line.encode() + b"\r\n")
        return self.wait(lambda ln: ln.startswith(("OK", "ERR")))

    def lines(self):
        """Whole lines received so far and not yet handed out."""
        try:
            self._lines += self.cmd.recv(65536)
        except socket.timeout:
            pass
        out = []
        while b"\r\n" in self._lines:
            ln, self._lines = self._lines.split(b"\r\n", 1)
            out.append(ln.decode())
        self._queue = self._queue + out
        out, self._queue = self._queue, []
        return out

    def pull(self):
        try:
            self.received += self.data.recv(65536)
        except socket.timeout:
            pass

    def wait(self, pred, timeout=30.0, pump=None, received_bytes=None):
        t0 = time.monotonic()
        matched = None
        while time.monotonic() - t0 < timeout:
            if pump and matched is None:
                pump()
            self.pull()
            if matched is None:
                got = self.lines()
                for i, ln in enumerate(got):
                    self.seen.append(ln)
                    if pred(ln):
                        self._queue = got[i + 1:] + self._queue    # read, not yet waited on
                        matched = ln
                        break
            if matched is not None and (received_bytes is None
                                        or len(self.received) >= received_bytes):
                return matched
        raise AssertionError(f"timed out; heard {self.seen}")

    def settle(self, seconds=0.3):
        """Read whatever arrives for a moment, expecting nothing in particular."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            self.pull()
            self.seen = self.seen + self.lines()

    def close(self):
        self.cmd.close()
        self.data.close()


@pytest.mark.parametrize('stalled', [False, True])
def test_receive_end_waits_for_data_within_the_original_deadline(monkeypatch, stalled):
    from types import SimpleNamespace
    import sys
    clock = SimpleNamespace(now=0., pumps=0, command_reads=0)
    monkeypatch.setattr(sys.modules[__name__], 'time',
                        SimpleNamespace(monotonic=lambda: clock.now))
    app = _Client.__new__(_Client)
    app.received, app.seen, app._queue = b'', [], []

    def pull():
        clock.now += 1
        app.received = b'head' if clock.now == 1 or stalled else b'headtail'

    def lines():
        clock.command_reads += 1
        return ['RX END LOST']

    def pump():
        clock.pumps += 1

    app.pull, app.lines = pull, lines
    if stalled:
        with pytest.raises(AssertionError, match='timed out'):
            app.wait(lambda ln: ln == 'RX END LOST', timeout=3,
                     pump=pump, received_bytes=8)
        assert clock.now == 3  # END does not start a second timeout.
        assert app.received == b'head'
    else:
        assert app.wait(lambda ln: ln == 'RX END LOST', timeout=3,
                        pump=pump, received_bytes=8) == 'RX END LOST'
        assert clock.now == 2 and app.received == b'headtail'
    assert clock.pumps == clock.command_reads == 1
    assert app.seen == ['RX END LOST']


def _replay_pump(station):
    """Feed one replay tick only after its real decoder call has completed."""
    modem = station.lanes['bateleur'].modem
    assert station.audio.samples == 0
    lock = threading.Lock()
    completed, failures = 0, []
    decode = modem._audio

    def consume(samples):
        nonlocal completed
        try:
            decode(samples)
        except BaseException as exc:
            with lock:
                failures.append(exc)
            raise
        with lock:
            completed += len(samples)

    modem._audio = consume

    def pump():
        with lock:
            done, failed = completed, tuple(failures)
        assert not failed, f'replay decoder failed: {failed!r}'
        assert modem._thread.is_alive(), 'replay decoder stopped'
        assert modem.audio_dropped == 0, 'replay overran the decoder'
        assert done <= station.audio.samples
        if done == station.audio.samples and not station.replay_exhausted():
            station._tick()
    return pump


def _controlled_replay(decode):
    pending = []
    modem = SimpleNamespace(_audio=decode, audio_dropped=0,
                            _thread=SimpleNamespace(is_alive=lambda: True))
    station = SimpleNamespace(audio=SimpleNamespace(samples=0),
                              lanes={'bateleur': SimpleNamespace(modem=modem)},
                              replay_exhausted=lambda: False)

    def tick():
        station.audio.samples += 4096
        pending.append(np.zeros(4096))
    station._tick = tick
    return station, modem, pending


def test_replay_waits_for_completed_decode_not_dequeue():
    def decode(samples):
        pump()  # Still processing: dequeue alone must not admit the next tick.
        assert station.audio.samples == 4096
    station, modem, pending = _controlled_replay(decode)
    pump = _replay_pump(station)
    pump()
    samples = pending.pop()
    pump()
    assert station.audio.samples == 4096 and not pending
    modem._audio(samples)
    pump()
    assert station.audio.samples == 8192 and len(pending) == 1


@pytest.mark.parametrize('failure', ['exception', 'stopped', 'dropped'])
def test_replay_decoder_failures_are_not_acknowledged(failure):
    def decode(samples):
        raise ValueError('broken decoder')
    station, modem, pending = _controlled_replay(decode)
    pump = _replay_pump(station)
    pump()
    if failure == 'exception':
        with pytest.raises(ValueError, match='broken decoder'):
            modem._audio(pending.pop())
    elif failure == 'stopped':
        modem._thread.is_alive = lambda: False
    else:
        modem.audio_dropped = 1
    with pytest.raises(AssertionError, match='replay'):
        pump()
    assert station.audio.samples == 4096


def test_pending_replay_still_obeys_the_original_wait_deadline(monkeypatch):
    import sys
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(sys.modules[__name__], 'time',
                        SimpleNamespace(monotonic=lambda: clock.now))
    station, _, pending = _controlled_replay(lambda samples: None)
    app = _Client.__new__(_Client)
    app.received, app.seen, app._queue = b'', [], []
    def pull():
        clock.now += 1
    app.pull, app.lines = pull, lambda: []
    with pytest.raises(AssertionError, match='timed out'):
        app.wait(lambda line: False, timeout=3, pump=_replay_pump(station))
    assert clock.now == 3 and station.audio.samples == 4096 and len(pending) == 1


def _cfg(**over):
    cfg = config.load(_repo() / "examples" / "station.toml")
    protos = {"bateleur": ProtocolConfig(enabled=True, host="bateleur")}
    return replace(cfg, protocols=protos, station=replace(cfg.station, transmit=False), **over)


def test_the_station_knows_it():
    raw = {"protocols": {"bateleur": {"enabled": True, "host": "bateleur",
                                      "cmd_port": 8310, "data_port": 8311}}}
    base = __import__("tomllib").loads((_repo() / "examples" / "station.toml").read_text())
    base["protocols"] = raw["protocols"]
    cfg = config.parse(base)
    assert cfg.protocols["bateleur"].cmd_port == 8310


def test_the_rig_s_top_edge_picks_the_filter_and_the_declared_passband():
    base = __import__("tomllib").loads((_repo() / "examples" / "station.toml").read_text())
    for bw, trim, top in (("", True, 2995.0), ("3000", True, 2995.0), ("3200", False, 3140.0)):
        base["protocols"] = {"bateleur": {"enabled": True, "host": "bateleur",
                                          "bandwidth": bw}}

        class _St:
            pass
        st = _St()
        st.cfg = config.parse(base)
        link = BateleurLink(st)
        assert link.modem.trim is trim
        assert link.waveform.emission(14_100_000, drift_hz=0).span[1] == 14_100_000 + top
    base["protocols"] = {"bateleur": {"enabled": True, "host": "bateleur",
                                      "bandwidth": "2800"}}
    with pytest.raises(config.ConfigError, match="bandwidth"):
        config.parse(base)


@pytest.mark.realtime
def test_receive_through_the_station_to_the_host_port():
    """A 188-110 transmission in the replay arrives at an attached application,
    bytes on the data port and the events on the command port."""
    air = np.concatenate([np.zeros(9600), mil110.transmit(MSG, "1200S"), np.zeros(9600 * 4)])
    card = for_card(sim.channel(air, snr_db=20, freq_hz=12.0, seed=1) * 0.3, 9600)
    st = Station(_cfg(), replay=card)
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    try:
        srv = served["bateleur"]
        assert srv.keys is False                     # transmit = false: no arbiter
        app = _Client(*srv.ports)
        assert app.ask("VERSION").startswith("OK bateleur")

        def pump():
            if not st.replay_exhausted():
                st._tick()
        app.wait(lambda ln: ln.startswith("RX START 110 1200S"), pump=pump)
        app.wait(lambda ln: ln.startswith("RX END EOM"), pump=pump)
        app.settle()                                 # let the last bytes land
        assert app.received == MSG
        app.close()
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            link.stop()


@pytest.fixture
def keyed():
    """A link on a fake rig and a real arbiter, recording what it keys."""
    fake = FakeRigctld()
    ptt = FakePtt()
    fake.ptt_line = ptt
    rig = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=ptt,
              profile=BENCH, control=config.Control.LOCAL, mycall="N0CALL",
              transmit=True, max_key_s=20.0)
    rig.arm(prove_ptt=False)
    audio = ReplayAudio(np.zeros(CARD_RATE_HZ * 2, np.float32))
    arb = TxArbiter(audio, rig, identity=Identity("N0CALL", interval_s=600.0))

    class _St:
        pass
    st = _St()
    st.arbiter, st.rig, st.audio = arb, rig, audio
    st.cfg = _cfg()
    st.cfg = replace(st.cfg, rig=replace(st.cfg.rig, max_key_s=20.0))
    link = BateleurLink(st)
    link.start()
    yield link, audio
    link.stop()
    rig.close(); fake.close()


def _decode_bursts(audio, standard="110", mode=None):
    got = b""
    for _, burst in audio.transmitted:
        x = np.asarray(burst, dtype=np.float64)
        from hfmodem.core.resample import from_card
        at9600 = from_card(x.astype(np.float32), 9600).astype(float) / 32768
        r = (mil110.receive(at9600) if standard == "110"
             else tone39.receive(at9600, mode) if standard == "39"
             else tone16.receive(at9600, mode) if standard == "16"
             else s4285.receive(at9600, mode))
        got += r.data
    return got


@pytest.mark.realtime
def test_send_from_the_host_port_keys_through_the_arbiter(keyed):
    link, audio = keyed
    host = BateleurHost(link.modem, cmd_port=0, data_port=0)
    host.start_background()
    try:
        app = _Client(host.cmd_port, host.data_port)
        assert app.ask("MODE 110 2400S") == "OK MODE 110 2400S"
        app.data.sendall(MSG)
        app.settle()
        assert app.ask("SEND") == f"OK SEND {len(MSG)}"
        app.wait(lambda ln: ln.startswith("TX START 110 2400S"))
        app.wait(lambda ln: ln == "TX END")
        app.close()
    finally:
        host.stop()
    assert len(audio.transmitted) == 1
    assert _decode_bursts(audio) == MSG


@pytest.mark.realtime
def test_39_tone_through_the_station_to_the_host_port():
    """The 39-tone mode needs no MODE command to be heard: the receiver
    finds the mode from the data."""
    text = b"39 TONE ON THE STATION\r\n" * 3
    air = np.concatenate([np.zeros(9600), tone39.transmit(text, "600/33/A10.8"),
                          np.zeros(9600 * 10)])
    card = for_card(sim.channel(air, snr_db=20, freq_hz=-9.0, seed=1) * 0.3, 9600)
    st = Station(_cfg(), replay=card)
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    try:
        app = _Client(*served["bateleur"].ports)

        def pump():
            if not st.replay_exhausted():
                st._tick()
        app.wait(lambda ln: ln.startswith("RX START 39 600/33/A10.8"), pump=pump)
        app.wait(lambda ln: ln.startswith("RX END EOM"), pump=pump)
        app.settle()
        assert app.received == text
        app.close()
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            link.stop()


@pytest.mark.realtime
def test_39_tone_from_the_host_port(keyed):
    link, audio = keyed
    host = BateleurHost(link.modem, cmd_port=0, data_port=0)
    host.start_background()
    try:
        app = _Client(host.cmd_port, host.data_port)
        assert app.ask("MODE 39 1200/63/s") == "OK MODE 39 1200/63/S"
        app.data.sendall(MSG)
        app.settle()
        assert app.ask("SEND") == f"OK SEND {len(MSG)}"
        app.wait(lambda ln: ln.startswith("TX START 39 1200/63/S"))
        app.wait(lambda ln: ln == "TX END")
        app.close()
    finally:
        host.stop()
    assert _decode_bursts(audio, "39", "1200/63/S")[:len(MSG)] == MSG


@pytest.mark.realtime
def test_16_tone_from_the_host_port(keyed):
    link, audio = keyed
    host = BateleurHost(link.modem, cmd_port=0, data_port=0)
    host.start_background()
    try:
        app = _Client(host.cmd_port, host.data_port)
        assert app.ask("MODE 16 1200") == "OK MODE 16 1200"
        app.data.sendall(MSG)
        app.settle()
        assert app.ask("SEND") == f"OK SEND {len(MSG)}"
        app.wait(lambda ln: ln.startswith("TX START 16 1200"))
        app.wait(lambda ln: ln == "TX END")
        app.close()
    finally:
        host.stop()
    assert _decode_bursts(audio, "16", 1200)[:len(MSG)] == MSG


@pytest.mark.realtime
def test_16_tone_through_the_station_to_the_host_port():
    """The rate is not on the air: the host sets it, and the receiver
    listens at it."""
    air = np.concatenate([np.zeros(9600), tone16.transmit(MSG, 600), np.zeros(9600 * 3)])
    card = for_card(sim.channel(air, snr_db=20, freq_hz=6.0, seed=1) * 0.3, 9600)
    st = Station(_cfg(), replay=card)
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    try:
        app = _Client(*served["bateleur"].ports)
        assert app.ask("MODE 16 600") == "OK MODE 16 600"

        def pump():
            if not st.replay_exhausted():
                st._tick()
        app.wait(lambda ln: ln.startswith("RX START 16 600"), pump=pump)
        app.wait(lambda ln: ln.startswith("RX END LOST"), pump=pump)
        app.settle()
        assert app.received[:len(MSG)] == MSG
        app.close()
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            link.stop()


@pytest.mark.realtime
def test_fsk_from_the_host_port_and_back_through_the_station(keyed):
    link, audio = keyed
    host = BateleurHost(link.modem, cmd_port=0, data_port=0)
    host.start_background()
    try:
        app = _Client(host.cmd_port, host.data_port)
        assert app.ask("MODE FSK 75w") == "OK MODE FSK 75W"
        assert app.ask("MODE FSK 75V1").startswith("ERR FSK 75V1 keys")
        app.data.sendall(MSG)
        app.settle()
        assert app.ask("SEND") == f"OK SEND {len(MSG)}"
        # TX END reports one key-up; await both within one original send bound.
        deadline = time.monotonic() + 60.0
        for _ in range(2):
            app.wait(lambda ln: ln.startswith("TX START FSK 75W"),
                     timeout=max(0.0, deadline - time.monotonic()))
            app.wait(lambda ln: ln == "TX END",
                     timeout=max(0.0, deadline - time.monotonic()))
        app.close()
    finally:
        host.stop()
    from hfmodem.bateleur import fsk
    from hfmodem.core.resample import from_card
    # At 75 Bd it runs past the 20 s key-up: two whole transmissions.
    assert len(audio.transmitted) == 2
    got = b""
    for _, burst in audio.transmitted:
        at9600 = from_card(np.asarray(burst, dtype=np.float32), 9600).astype(float) / 32768
        got += fsk.receive(at9600, "75W").data
    assert got == MSG


@pytest.mark.realtime
def test_fsk_through_the_station_to_the_host_port():
    from hfmodem.bateleur import fsk
    air = np.concatenate([np.zeros(9600), fsk.transmit(MSG, "75W/1700"), np.zeros(9600 * 3)])
    card = for_card(sim.channel(air, snr_db=20, freq_hz=9.0, seed=1) * 0.3, 9600)
    st = Station(_cfg(), replay=card)
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    modem = st.lanes['bateleur'].modem
    pump = _replay_pump(st)
    try:
        app = _Client(*served["bateleur"].ports)
        assert app.ask("MODE FSK 75W/1700") == "OK MODE FSK 75W/1700"

        app.wait(lambda ln: ln.startswith("RX START FSK 75W/1700"), pump=pump)
        def ended(line):
            assert 'receiver reset' not in line, (line, modem.audio_dropped)
            return line.startswith("RX END LOST")
        # Command and data sockets have independent asynchronous writers.
        app.wait(ended, pump=pump, received_bytes=len(MSG))
        app.settle()  # Preserve the post-delivery window for surplus bytes.
        assert modem.audio_dropped == 0
        assert not any('receiver reset' in line for line in app.seen)
        assert app.received == MSG
        app.close()
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            link.stop()


@pytest.mark.realtime
def test_4529_from_the_host_port_and_back_through_the_station(keyed):
    from hfmodem.bateleur import s4529
    from hfmodem.core.resample import from_card
    link, audio = keyed
    host = BateleurHost(link.modem, cmd_port=0, data_port=0)
    host.start_background()
    try:
        app = _Client(host.cmd_port, host.data_port)
        assert app.ask("MODE 4529 600s") == "OK MODE 4529 600S"
        app.data.sendall(MSG)
        app.settle()
        assert app.ask("SEND") == f"OK SEND {len(MSG)}"
        app.wait(lambda ln: ln.startswith("TX START 4529 600S"))
        app.wait(lambda ln: ln == "TX END", timeout=60.0)
        app.close()
    finally:
        host.stop()
    got = b""
    for _, burst in audio.transmitted:
        at9600 = from_card(np.asarray(burst, dtype=np.float32), 9600).astype(float) / 32768
        got += s4529.receive(at9600, "600S").data
    assert got == MSG


@pytest.mark.realtime
def test_4529_through_the_station_to_the_host_port():
    from hfmodem.bateleur import s4529
    air = np.concatenate([np.zeros(9600), s4529.transmit(MSG, "600S/1500"),
                          np.zeros(9600 * 3)])
    card = for_card(sim.channel(air, snr_db=20, freq_hz=-8.0, seed=1) * 0.3, 9600)
    st = Station(_cfg(), replay=card)
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    try:
        app = _Client(*served["bateleur"].ports)
        assert app.ask("MODE 4529 600S/1500") == "OK MODE 4529 600S/1500"

        def pump():
            if not st.replay_exhausted():
                st._tick()
        app.wait(lambda ln: ln.startswith("RX START 4529 600S/1500"), pump=pump)
        app.wait(lambda ln: ln.startswith("RX END EOM"), pump=pump)
        app.settle()
        assert app.received == MSG
        app.close()
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            link.stop()


@pytest.mark.realtime
def test_appendix_d_from_the_host_port_and_back_through_the_station(keyed):
    from hfmodem.bateleur import wbhf
    from hfmodem.core.resample import from_card
    link, audio = keyed
    host = BateleurHost(link.modem, cmd_port=0, data_port=0)
    host.start_background()
    try:
        app = _Client(host.cmd_port, host.data_port)
        assert app.ask("MODE wbhf 4800s") == "OK MODE WBHF 4800S"
        app.data.sendall(MSG)
        app.settle()
        assert app.ask("SEND") == f"OK SEND {len(MSG)}"
        app.wait(lambda ln: ln.startswith("TX START WBHF 4800S"))
        app.wait(lambda ln: ln == "TX END", timeout=60.0)
        app.close()
    finally:
        host.stop()
    got = b""
    for _, burst in audio.transmitted:
        at9600 = from_card(np.asarray(burst, dtype=np.float32), 9600).astype(float) / 32768
        got += wbhf.receive(at9600).data
    assert got == MSG


@pytest.mark.realtime
def test_appendix_d_through_the_station_to_the_host_port():
    """Like 188-110 and 4539, heard whatever mode is set: the preamble
    names the waveform."""
    from hfmodem.bateleur import wbhf
    air = np.concatenate([np.zeros(9600), wbhf.transmit(MSG, "1200S"), np.zeros(9600 * 3)])
    card = for_card(sim.channel(air, snr_db=15, freq_hz=-8.0, seed=1) * 0.3, 9600)
    st = Station(_cfg(), replay=card)
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    try:
        app = _Client(*served["bateleur"].ports)

        def pump():
            if not st.replay_exhausted():
                st._tick()
        app.wait(lambda ln: ln.startswith("RX START WBHF 1200S"), pump=pump)
        app.wait(lambda ln: ln.startswith("RX END EOM"), pump=pump)
        app.settle()
        assert app.received == MSG
        app.close()
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            link.stop()


@pytest.mark.realtime
def test_a_39_tone_message_longer_than_a_key_up_is_split(keyed):
    """At 75 bit/s 20 s carries about 150 bytes; the modem's reckoning of
    how long a message keys is the waveform's, without rendering it."""
    link, audio = keyed
    data = b"0123456789ABCDEF" * 25
    link.modem.set_mode(Mode("39", "75/12/A10.8"))
    link.modem.transmit(data)
    assert len(audio.transmitted) >= 2
    for _, burst in audio.transmitted:
        assert burst.size / CARD_RATE_HZ <= 20.0
    assert _decode_bursts(audio, "39", "75/12/A10.8") == data


@pytest.mark.realtime
def test_a_message_longer_than_a_key_up_is_split(keyed):
    """max_key_s is 20 s here; 600 bit/s short carries about 1 KB in that, so
    3 KB goes as several whole transmissions that decode back to the message."""
    link, audio = keyed
    data = bytes(np.random.default_rng(3).integers(0, 256, 3000, dtype=np.uint8))
    link.modem.set_mode(Mode("110", "600S"))
    link.modem.transmit(data)
    assert len(audio.transmitted) >= 3
    for _, burst in audio.transmitted:
        assert burst.size / CARD_RATE_HZ <= 20.0
    assert _decode_bursts(audio) == data


def test_commands_answer_and_refuse():
    m = BateleurModem(lambda a, r: "", card_rate=CARD_RATE_HZ, max_key_s=60.0)
    m.start()
    h = BateleurHost(m, cmd_port=0, data_port=0)
    try:
        assert h.command("MODE") == "OK MODE 110 600S"
        assert h.command("MODE 4285 1200L") == "OK MODE 4285 1200L"
        assert m.mode == Mode("4285", "1200L")
        assert h.command("MODE 4285 9600X").startswith("ERR no 4285 mode")
        assert h.command("MODE 188 600S").startswith("ERR no standard")
        assert h.command("BITORDER MSB") == "OK BITORDER MSB"
        assert h.command("BITORDER middle").startswith("ERR")
        assert h.command("AUTOSEND 500") == "OK AUTOSEND 500"
        assert h.command("AUTOSEND OFF") == "OK AUTOSEND OFF"
        assert h.command("STATUS").startswith("OK STATUS MODE 4285 1200L RX idle TX idle")
        assert h.command("FROB").startswith("ERR unknown command")
        assert h.command("MODE 39 600/99") == "OK MODE 39 600/99/A10.8"
        assert h.command("MODE 39 2400/5/S").startswith("ERR no 39-tone mode")
        assert h.command("STATUS").startswith("OK STATUS MODE 39 600/99/A10.8 RX idle")
    finally:
        m.stop()


def _sis_client(port):
    from hfmodem.bateleur.s5066 import primitives as P
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    c.settimeout(0.05)
    c.sendall(P.encode(P.prim(P.BIND_REQUEST, sap=2, rank=0,
                              service=P.ServiceType(P.MODE_NON_ARQ, P.CONFIRM_NONE, True))))
    return c, P.Framer()


def _sis_wait(c, f, t, pump=None, timeout=60.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pump:
            pump()
        try:
            for p in f.push(c.recv(65536)):
                if p.type == t:
                    return p
        except socket.timeout:
            pass
    raise AssertionError(f"no primitive {t}")


@pytest.mark.realtime
def test_a_s5066_node_on_the_station_hears_a_broadcast():
    """[protocols.bateleur] host = "s5066": a far node's non-ARQ U_PDU, off
    the replayed air, reaches a client bound on the SIS port."""
    from hfmodem.bateleur.s5066 import primitives as P
    from hfmodem.tests.bateleur.s5066.ether import Client
    from hfmodem.bateleur.s5066.node import Node, Timing
    far = Node("0.0.0.9", Timing(1200, 0.6, 0.6))
    fc = Client(far)
    fc.bind(2, mode=P.MODE_NON_ARQ)
    fc.unidata(1, 2, MSG)
    data, _ = far.poll(0.0)
    air = np.concatenate([np.zeros(9600), mil110.transmit(data, "1200S"), np.zeros(9600 * 4)])
    card = for_card(sim.channel(air, snr_db=20, freq_hz=7.0, seed=3) * 0.3, 9600)
    cfg = _cfg()
    cfg = replace(cfg, protocols={"bateleur": ProtocolConfig(
        enabled=True, host="s5066", addresses=("0.0.0.1",), mode="110 1200S")})
    st = Station(cfg, replay=card)
    st.open_audio()
    st.build_lanes()
    served = hosts.build(st.cfg, station=st, log=lambda *_: None)
    try:
        c, f = _sis_client(served["bateleur"].ports[0])
        assert _sis_wait(c, f, P.BIND_ACCEPTED)["sap"] == 2

        def pump():
            if not st.replay_exhausted():
                st._tick()
        got = _sis_wait(c, f, P.UNIDATA_INDICATION, pump=pump)
        assert got["u_pdu"] == MSG and got["src"] == 9 and got["src_sap"] == 2
        c.close()
    finally:
        hosts.stop_all(served)
        for link in st.lanes.values():
            link.stop()


@pytest.mark.realtime
def test_a_s5066_client_keys_through_the_arbiter(keyed):
    from hfmodem.bateleur.s5066 import dpdu as D
    from hfmodem.bateleur.s5066 import primitives as P
    from hfmodem.bateleur.s5066.server import NodeService
    link, audio = keyed
    svc = NodeService(link.modem, "0.0.0.1", "110 2400S", port=0)
    svc.start_background()
    try:
        c, f = _sis_client(svc.port)
        _sis_wait(c, f, P.BIND_ACCEPTED)
        c.sendall(P.encode(P.prim(P.UNIDATA_REQUEST, priority=0, dest_sap=2, dest=0xFFFF,
                                  group=True, mode=P.ServiceType(P.MODE_DEFAULT), ttl=0,
                                  u_pdu=MSG)))
        t0 = time.monotonic()
        while not audio.transmitted and time.monotonic() - t0 < 30:
            time.sleep(0.05)
        time.sleep(0.5)
        c.close()
    finally:
        svc.stop()
    pdus = D.Parser().push(_decode_bursts(audio))
    assert b"".join(p.data for p in pdus)[6:] == MSG      # past the C_ and S_PDU headers
    assert all(p.type == D.NON_ARQ and p.group and p.src == 1 for p in pdus)
