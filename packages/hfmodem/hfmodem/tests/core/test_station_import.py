# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Station import regressions, with no radio attached."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core.rig import Rig, RigError
from hfmodem.core.regulatory import Control, centred
from hfmodem.tests.core.fakerig import FakePtt


def test_tune_cannot_move_between_check_and_key():
    checking, release, attempted = (threading.Event() for _ in range(3))
    checked, keyed, errors = [], [], []

    class Profile:
        name = "bench"

        def check(self, emission, **kw):
            checked.append(emission.dial_hz)
            checking.set()
            assert release.wait(2)

    fake = SimpleNamespace(freq=7_100_000)
    cat = SimpleNamespace(freq=lambda: fake.freq,
                          set_freq=lambda value: setattr(fake, "freq", value))
    r = Rig(model="ft891", cat=cat, ptt=FakePtt(), profile=Profile(),
            control=Control.LOCAL, mycall="N0CALL", transmit=True)
    r._armed = True
    r._dial_hz = fake.freq
    original = r.ptt.assert_

    def key_line(up):
        if up:
            keyed.append(fake.freq)
        original(up)

    r.ptt.assert_ = key_line

    def tune():
        attempted.set()
        try:
            r.tune(14_110_500)
        except RigError as exc:
            errors.append(str(exc))

    tokens = []
    keyer = threading.Thread(target=lambda: tokens.append(r.key(centred(7_100_000, 500), why="test", duration_s=1)))
    keyer.start()
    assert checking.wait(2)
    tuner = threading.Thread(target=tune)
    tuner.start()
    assert attempted.wait(2)
    tuner.join(0.1)
    release.set()
    keyer.join(2)
    tuner.join(2)
    try:
        assert checked == keyed
        assert errors == ["refusing to retune while keyed"]
    finally:
        if tokens:
            r.unkey(tokens[0])


@pytest.mark.parametrize("name", ["BateleurLink", "AleLink", "SelcallLink"])
def test_stream_lanes_fill_gaps(name):
    from hfmodem.station import link
    cls = getattr(link, name)
    lane = object.__new__(cls)
    got = []
    lane.lane = SimpleNamespace(poll=lambda: [(10, np.ones(3)), (16, np.ones(2))])
    lane.modem = SimpleNamespace(feed=lambda *args: got.append(args[-1]), post=lambda fn: fn())
    lane.station = SimpleNamespace(arbiter=None)
    lane._others_sent = 0
    lane._expect_audio = None
    lane.poll()
    np.testing.assert_array_equal(np.concatenate(got), [1, 1, 1, 0, 0, 0, 1, 1])


def test_broken_lane_constructor_stops_bringup(monkeypatch):
    from hfmodem.station.link import LINKS
    from hfmodem.station.process import Station

    def broken(station):
        raise TypeError("broken lane")

    monkeypatch.setitem(LINKS, "ale", broken)
    st = Station(SimpleNamespace(protocols={"ale": SimpleNamespace(enabled=True)}))
    with pytest.raises(TypeError, match="broken lane"):
        st.build_lanes()


def test_cat_rechecks_keyed_after_lock_wait():
    from hfmodem.core.rig import Cat
    reached = threading.Event()
    keyed = False
    sent, errors = [], []

    class Lock:
        def __enter__(self):
            nonlocal keyed
            reached.set()
            keyed = True
        def __exit__(self, *args): pass

    cat = Cat("unused", 1)
    cat.keyed_probe = lambda: keyed
    cat._lock = Lock()
    cat._sock = SimpleNamespace(sendall=sent.append)
    cat._read_reply = lambda: "RPRT 0"
    try:
        cat.command("F 14109000")
    except RigError as exc:
        errors.append(str(exc))
    assert reached.is_set() and errors and not sent


def test_active_link_reserves_transmitter():
    from hfmodem.station.arbiter import TxArbiter, TxRequest, Refused, ERRAND
    arb = TxArbiter(SimpleNamespace(), SimpleNamespace(max_key_s=30))
    arb.active_protocols = lambda: ("sabir",)
    sent = []
    arb._transmit = lambda req: sent.append(req) or 0
    with pytest.raises(Refused, match="sabir"):
        arb.submit(TxRequest(ERRAND, 0, "bateleur", np.zeros(1), 8000))
    assert not sent


def test_ale_scan_yields_to_other_active_lane():
    from hfmodem.station.link import AleLink
    lane = object.__new__(AleLink)
    tuned = []
    lane.dials = [7102000]
    lane.station = SimpleNamespace(lanes={"sabir": SimpleNamespace(active=True)},
                                   rig=SimpleNamespace(tune=tuned.append), audio=None, arbiter=None)
    assert lane._tune(0) is None
    assert not tuned


def test_ale_stop_restores_configured_centre():
    from hfmodem.station.link import AleLink
    lane = object.__new__(AleLink)
    calls = []
    lane.modem = SimpleNamespace(stop=lambda: calls.append("stop"))
    lane.station = SimpleNamespace(rig=SimpleNamespace(tune=calls.append),
                                   cfg=SimpleNamespace(rig=SimpleNamespace(centre_hz=7101500)))
    lane.stop()
    assert calls == ["stop", 7101500]


def test_ale_refuses_send_after_failed_readback():
    from hfmodem.station.link import AleLink
    lane = object.__new__(AleLink)
    lane.dials = [7102000]
    lane.modem = SimpleNamespace(ctl=SimpleNamespace(channel=0))
    lane.station = SimpleNamespace(rig=SimpleNamespace(dial=lambda: 14109000))
    lane.transmit = lambda audio: 42
    assert "dial" in lane._send(np.zeros(1), 8000)


@pytest.mark.parametrize("name", ["BateleurLink", "AleLink", "SelcallLink"])
def test_wrong_lane_rate_is_a_runtime_error(name):
    from hfmodem.station import link
    lane = object.__new__(getattr(link, name))
    lane.station = SimpleNamespace(rig=None)
    lane.transmit = lambda audio: 42
    with pytest.raises(ValueError, match="Hz"):
        lane._send(np.zeros(1), 123)


@pytest.mark.parametrize("name", ["BateleurLink", "AleLink", "SelcallLink"])
def test_long_gap_resets_receiver_before_new_audio(name):
    from hfmodem.station import link
    lane = object.__new__(getattr(link, name))
    seen = []
    lane.lane = SimpleNamespace(poll=lambda: [(0, np.ones(3)), (96000, np.ones(1))])
    lane.modem = SimpleNamespace(feed=lambda *a: seen.append(len(a[-1])),
                                 mark_audio_gap=lambda: seen.append("reset"), post=lambda fn: fn())
    lane.station = SimpleNamespace(arbiter=None)
    lane._others_sent = 0
    lane._expect_audio = None
    lane.poll()
    assert seen == [3, "reset", 1]


def test_errand_cannot_bypass_due_identification():
    from hfmodem.station.arbiter import ERRAND, Identity, Refused, TxArbiter, TxRequest
    identity = Identity("N0CALL", interval_s=0)
    arb = TxArbiter(None, SimpleNamespace(max_key_s=30), identity=identity)
    arb._transmit = lambda req: 0
    with pytest.raises(Refused, match="identification is due"):
        arb.submit(TxRequest(ERRAND, 0, "bateleur", np.zeros(1), 8000))


def test_other_errand_does_not_reset_identity_timer():
    from hfmodem.core.audio import ReplayAudio
    from hfmodem.station.arbiter import ERRAND, Identity, TxArbiter, TxRequest
    identity = Identity("N0CALL")
    before = identity._last
    rig = SimpleNamespace(max_key_s=30, key=lambda *a, **kw: 1, unkey=lambda t: None, retired=False)
    arb = TxArbiter(ReplayAudio(np.zeros(48000)), rig, identity=identity)
    arb.submit(TxRequest(ERRAND, 0, "bateleur", np.zeros(100), 8000,
                         emission=centred(7_100_000, 500)))
    assert arb.sent == 1
    assert arb.last_sent["bateleur"] > 0
    assert identity._last == before


def test_ale_poll_uses_snapshot_when_a_sender_is_added():
    from hfmodem.station.link import AleLink
    class Sender(str):
        __hash__ = str.__hash__
        def __eq__(self, other):
            sent["bateleur"] = 20
            return str.__eq__(self, other)
    sent = {Sender("sabir"): 10}
    lane = object.__new__(AleLink)
    lane.lane = SimpleNamespace(poll=lambda: [])
    lane.station = SimpleNamespace(arbiter=SimpleNamespace(last_sent=sent))
    lane._others_sent = 0
    lane._expect_audio = None
    lane.modem = SimpleNamespace(activity=lambda at: None)
    lane.poll()
    assert lane._others_sent == 10


def test_retune_failure_names_frequency_and_cause(capsys):
    from hfmodem.station.link import AleLink
    def tune(value):
        raise RigError("readback mismatch")
    lane = object.__new__(AleLink)
    lane.dials = [7102000]
    lane.station = SimpleNamespace(lanes={}, rig=SimpleNamespace(tune=tune), audio=None, arbiter=None)
    assert lane._tune(0) is None
    line = capsys.readouterr().err
    assert "7102.0 kHz" in line and "readback mismatch" in line


@pytest.mark.parametrize("drive", [-.1, 1.1, float("nan")])
def test_transmit_drive_stays_inside_card_range(drive):
    from hfhost.config import ConfigError
    from hfmodem.core import config
    from hfmodem.tests.core.test_config import base
    raw = base()
    raw["audio"]["tx_drive"] = drive
    with pytest.raises(ConfigError, match="tx_drive"):
        config.parse(raw)


@pytest.mark.parametrize("extra", [{"conventions": "hflink", "sound_s": 120}, {"base": True}])
def test_ale_settings_cannot_be_silently_ignored(extra):
    from hfhost.config import ConfigError
    from hfmodem.core import config
    from hfmodem.tests.core.test_config import base
    raw = base()
    raw["protocols"]["ale"].update(enabled=True, **extra)
    with pytest.raises(ConfigError, match="hflink"):
        config.parse(raw)


@pytest.mark.parametrize("override", [{"activity_s": 30.0}, {"sound_s": 0.0}])
def test_hflink_rejects_explicit_default_interval_overrides(override):
    from hfhost.config import ConfigError
    from hfmodem.core import config
    from hfmodem.tests.core.test_config import base
    raw = base()
    raw["protocols"]["ale"].update(enabled=True, conventions="hflink", **override)
    with pytest.raises(ConfigError, match="hflink"):
        config.parse(raw)


def test_ale_retune_uses_an_idle_arbiter_hold():
    from hfmodem.station.arbiter import TxArbiter
    from hfmodem.station.link import AleLink
    lane = object.__new__(AleLink)
    tuned = []
    arbiter = TxArbiter(None, None)
    lane.dials = [7102000]
    lane.station = SimpleNamespace(lanes={}, arbiter=arbiter, audio=None,
                                   rig=SimpleNamespace(tune=tuned.append))
    with arbiter.try_hold() as held:
        assert held
        assert lane._tune(0) is None
        assert tuned == []
    assert lane._tune(0) == 0
    assert tuned == [7103500]
    with arbiter.try_hold() as held:
        assert held
