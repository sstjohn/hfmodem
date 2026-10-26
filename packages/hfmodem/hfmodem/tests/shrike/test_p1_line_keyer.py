# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
import inspect
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import claim, ptt
from hfmodem.core.rigs import RIGS
from hfmodem.shrike import beacon, onair, ota, spec


class FakeLine:
    def __init__(self, device):
        self.device = device
        self.calls = []
        self.opened = False
        self.state = False
        self.readable = True
        self.fail_on = None
        self.open_error = False
        self.open_hook = None
        self.drive_hook = None
        self.released_low = None

    def open(self):
        self.calls.append("open")
        self.opened = True
        self.state = False
        if self.open_hook:
            self.open_hook()
        if self.open_error:
            raise ptt.PttError("partial open")

    def sense(self):
        self.calls.append("sense")
        return self.state if self.readable and self.opened else None

    def assert_(self, on):
        self.calls.append(on)
        if self.drive_hook:
            self.drive_hook(on)
        if not self.opened or self.fail_on is on:
            raise ptt.PttError("refused drive")
        self.state = on

    def release(self):
        self.calls.append("release")
        if self.opened:
            self.released_low = self.fail_on is not False and self.readable
            if self.fail_on is not False:
                self.state = False
            self.opened = False


@pytest.fixture
def line_rig(monkeypatch):
    lines = []

    def make(device):
        line = FakeLine(device)
        lines.append(line)
        return line

    def forbidden(*args, **kwargs):
        pytest.fail("PTT must not open CAT, a process, or a real device")

    monkeypatch.setattr(ptt, "RtsPtt", make)
    monkeypatch.setattr(ota, "find_rigctl", lambda directory: "fake-rigctl")
    monkeypatch.setattr(ota.subprocess, "Popen", forbidden)
    monkeypatch.setattr(ota.subprocess, "run", forbidden)
    monkeypatch.setattr(ota, "require_char_device", forbidden)
    rig = ota.Rig(1, "fake-cat", 38400, ptt_port="fake-line")
    yield rig, lines[0]
    rig.stop()


def test_first_and_later_keys_use_one_open_and_line_evidence(line_rig):
    rig, line = line_rig
    assert rig.key_failure() is not None
    rig.arm()
    assert line.calls == ["open", "sense"]
    for _ in range(2):
        assert rig.ptt(True)
        assert rig.key_failure() is None
        assert rig.ptt(False)
        assert rig.key_failure() is None
    assert line.calls == ["open", "sense", True, "sense", False, "sense",
                          True, "sense", False, "sense"]
    rig._sick = True
    rig._errs.append("CAT refused a command")
    rig._proc = SimpleNamespace(poll=lambda: 1)
    assert rig.key_failure() is None
    rig._proc = None
    rig.stop()
    assert not line.opened
    assert line.released_low is True
    assert not rig.ptt(True)
    assert line.calls.count("open") == 1


@pytest.mark.parametrize("readback", [True, None])
def test_arm_requires_positive_low_before_first_assert(line_rig, readback):
    rig, line = line_rig
    line.open_hook = lambda: setattr(line, "state", readback)
    with pytest.raises(ptt.PttError, match="not confirmed low"):
        rig.arm()
    assert True not in line.calls
    assert not line.opened
    assert rig.retired
    assert not rig.ptt(True)


@pytest.mark.parametrize("failure", ["refused", "unreadable", "high"])
def test_failed_unkey_retires_and_refuses_next_key(line_rig, failure):
    rig, line = line_rig
    assert rig.ptt(True)
    if failure == "refused":
        line.fail_on = False
    elif failure == "unreadable":
        line.readable = False
    else:
        line.sense = lambda: True
    assert not rig.ptt(False)
    assert rig.retired and rig._keyer.must_retire
    assert rig.key_failure() is not None
    before = list(line.calls)
    assert not rig.ptt(True)
    assert line.calls == before


@pytest.mark.parametrize("failure", ["refused", "unreadable"])
def test_failed_key_is_not_carrier_and_drops_line(line_rig, failure):
    rig, line = line_rig
    rig.arm()
    if failure == "refused":
        line.fail_on = True
    else:
        line.readable = False
    assert not rig.ptt(True)
    assert rig.key_failure() is not None
    assert False in line.calls
    assert line.state is False


@pytest.mark.parametrize("when", ["before_assert", "after_assert"])
def test_key_overtaken_by_stop(line_rig, when):
    rig, line = line_rig
    rig.arm()
    entered, proceed = threading.Event(), threading.Event()
    result = []
    if when == "before_assert":
        def hook(on):
            if on:
                entered.set()
                assert proceed.wait(2)
        line.drive_hook = hook
    else:
        sense = line.sense
        def delayed_sense():
            state = sense()
            if state:
                entered.set()
                assert proceed.wait(2)
            return state
        line.sense = delayed_sense
    worker = threading.Thread(target=lambda: result.append(rig.ptt(True)))
    worker.start()
    try:
        assert entered.wait(2)
        rig.stop()
    finally:
        proceed.set()
        worker.join(2)
    assert not worker.is_alive()
    assert result == [False]
    assert rig.retired
    assert not line.state and not line.opened
    assert not rig.ptt(True)
    assert rig.key_failure() is not None


def test_stop_overtakes_arm_and_releases_late_open(line_rig):
    rig, line = line_rig
    line.open_hook = rig.stop
    with pytest.raises(ptt.PttError):
        rig.arm()
    assert not line.opened
    assert rig.retired
    assert True not in line.calls


def test_partial_initialization_releases_without_opening_cat(line_rig):
    rig, line = line_rig
    line.open_error = True
    with pytest.raises(ptt.PttError, match="partial open"):
        rig.arm()
    rig.stop()
    assert line.calls.count("open") == 1
    assert not line.opened
    assert rig.retired
    assert True not in line.calls


def test_stop_before_arm_never_opens_line(line_rig):
    rig, line = line_rig
    rig.stop()
    rig.stop()
    assert "open" not in line.calls
    assert not rig.ptt(True)


def test_stop_releases_line_before_cat_close(line_rig, monkeypatch):
    rig, line = line_rig
    assert rig.ptt(True)
    def close():
        assert line.state is False
        assert not line.opened
    monkeypatch.setattr(rig, "_close", close)
    rig.stop()


def test_release_readback_failure_retires(line_rig, capsys):
    rig, line = line_rig
    assert rig.ptt(True)
    line.readable = False
    rig.stop()
    assert rig.retired
    assert "NOT CONFIRMED LOW" in capsys.readouterr().out


def test_line_keyer_honours_abort_and_retirement(line_rig):
    rig, line = line_rig
    rig.arm()
    rig._down.set()
    assert not rig._keyer.key(True)
    rig._down.clear()
    rig._keyer.must_retire = True
    assert not rig._keyer.key(True)
    assert True not in line.calls


def test_all_hamlib_argv_disable_ptt(line_rig, monkeypatch):
    rig, line = line_rig
    calls = []
    def run(argv, timeout, **_kw):
        calls.append(argv)
        if argv[-1] == "m":
            return SimpleNamespace(returncode=0, stdout="PKTUSB\n2400\n", stderr="")
        if "VS;" in argv:
            return SimpleNamespace(returncode=0,
                stdout="VS0; MD0C; MD1C; SH020; RF00;", stderr="")
        return SimpleNamespace(returncode=0, stdout="7100000\n", stderr="")
    monkeypatch.setattr(ota, "cat_once", run)
    calls.append(rig._argv())
    rig.get_freq()
    rig.get_power()
    rig.set_power_once(0.1)
    rig.get_mode()
    rig.get_frequency_readback()
    try:
        rig.get_receiver_filters()
    except ota.ReceiverReadbackError:
        pass
    assert len(calls) == 7
    for argv in calls:
        assert argv[argv.index("-P") + 1] == "NONE"
        assert "-p" not in argv and "T" not in argv
    assert not line.opened


@pytest.mark.parametrize("ptt_type", ["RIG", "RTS", "DTR", "NONE"])
def test_non_line_keying_is_not_selectable(ptt_type):
    with pytest.raises(ValueError, match="LINE"):
        ota.Rig(1, "fake-cat", 38400, ptt_type=ptt_type)


@pytest.mark.parametrize("ptt_type", ["LINE", "RIG", "RTS", "DTR", "NONE"])
def test_ptt_type_option_is_removed(ptt_type):
    with pytest.raises(SystemExit) as exc:
        onair.build_parser().parse_args(["--ptt-type", ptt_type])
    assert exc.value.code == 2


def test_cli_has_no_ptt_type():
    assert not hasattr(onair.build_parser().parse_args([]), "ptt_type")


@pytest.mark.parametrize("profile", ["g90", "x6100"])
def test_no_line_profile_has_no_cat_ptt_fallback(line_rig, profile):
    assert ota.ptt_device(profile, "must-not-stat-cat") is None
    rig = ota.Rig(1, "fake-cat", 38400)
    assert rig.ptt(True) is False
    assert rig.ptt(False) is True
    assert rig.key_failure() == ota.NO_KEYING_LINE
    rig.stop()


@pytest.mark.parametrize("listen, seconds", [
    ([], (8 + 2) * spec.CYCLE_SHORT_S),
    (["--max-cycles", "3"], (3 + 2) * spec.CYCLE_SHORT_S),
    (["--listen-only", "0.01"], 0.01),
    (["--answer", "0.02"], 0.02),
])
@pytest.mark.parametrize("profile", ["g90", "x6100"])
def test_no_line_profile_runs_receive_only(monkeypatch, tmp_path, capsys, profile, listen,
                                           seconds):
    events = []
    if "--answer" in listen:
        # Neither line-less profile settles inside the answer delay, which refuses
        # --answer before any listening; the budget is pinned on a row that does.
        monkeypatch.setitem(RIGS[profile], "settle", 0.02)

    class Admission:
        held = True

        def opened(self):
            assert self.held
            self.held = False
            events.append("opened")

        def failed(self):
            self.held = False
            events.append("failed")

    admission = Admission()

    def inherited(fd, *, launch):
        assert (fd, launch) == (9, True)
        events.append("inherited")
        return admission

    monkeypatch.setattr(claim, "inherited", inherited)
    args = onair.build_parser().parse_args([
        "--rig", profile, "--transmit", "--mycall", "W9SSJ", *listen,
        "--outdir", str(tmp_path / "receive"), "--ptt-port", "must-not-open",
    ])
    recordings = []
    def rec(n, **kwargs):
        assert admission.held
        events.append("capture")
        recordings.append(n)
        return np.zeros((n, 1), dtype=np.float32)
    def forbidden(*args, **kwargs):
        pytest.fail("receive-only profile tried to open a rig or output device")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(rec=rec, wait=lambda: None))
    monkeypatch.setattr(ota, "Rig", forbidden)
    def device(name, direction, **kwargs):
        assert direction == "in"
        return "fake-input"
    monkeypatch.setattr(onair, "find_device", device)
    monkeypatch.setattr(onair.rxfront, "decode_events", lambda audio: [])
    assert onair.run(args) == 0
    assert recordings == [int(seconds * onair.FS)]
    assert not args.transmit
    assert ota.NO_KEYING_LINE in capsys.readouterr().out
    assert events == ["inherited", "capture", "opened", "failed"]


def test_beacon_without_line_never_opens_devices(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["beacon", "--rig", "g90", "--serial", "fake",
                                      "--mycall", "W9SSJ", "--dxcall", "W9SSJ"])
    monkeypatch.setattr(beacon, "find_device", lambda *a: pytest.fail("audio open"))
    monkeypatch.setattr(beacon, "Rig", lambda *a, **k: pytest.fail("rig open"))
    beacon.main()
    assert ota.NO_KEYING_LINE in capsys.readouterr().out


def test_daemon_ptt_implementations_are_deleted():
    tree = ast.parse(inspect.getsource(ptt))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "OneShotRigctl" not in names
    assert "_RigctldPtt" not in names
    keyer = inspect.getsource(ptt.Keyer)
    for branch in ("transport.paced", "attempt_timeout", "confirm_silence", "drop_rts",
                   "taken_note", "time.sleep"):
        assert branch not in keyer
    assert "OneShotRigctl" not in inspect.getsource(ota)


@pytest.fixture
def watchdogs(monkeypatch):
    callbacks = []

    class Event:
        def wait(self, timeout=None):
            return False

        def set(self):
            pass

    class Thread:
        def __init__(self, target, **kwargs):
            callbacks.append(target)

        def start(self):
            pass

    monkeypatch.setattr(threading, "Thread", Thread)
    monkeypatch.setattr(threading, "Event", Event)
    return callbacks


def test_play_watchdog_stops_the_same_line_keyer(line_rig, watchdogs, monkeypatch):
    rig, line = line_rig
    monkeypatch.setattr(ota, "_warm_output", lambda device: None)
    def play(*args):
        assert line.state
        watchdogs[0]()
        assert not line.state and not line.opened
    monkeypatch.setattr(ota, "play_drained", play)
    ota._play(np.zeros(8), "fake-output", 1, rig, settle=0)
    assert rig._down.is_set()
    assert not rig.ptt(True)


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("low_confirmed", [False, True])
def test_duplex_watchdog_unkeys_without_ending_cat(
        line_rig, watchdogs, monkeypatch, guarded, low_confirmed):
    rig, line = line_rig
    cat = object()
    rig._proc = cat
    closes = []
    monkeypatch.setattr(rig, "_close", lambda: closes.append(True))
    live = onair._LiveInput.__new__(onair._LiveInput)
    live._duplex = True
    live.fs = onair.FS
    live.samples = 0
    live._blk = 128
    live.geometry_n = 0
    live._lat = 0
    live.tx_latency_n = 0
    live._dac_time = lambda index: 0
    live._sleep_until = lambda when: None
    live.sample_now = lambda: 0
    def drain(timeout):
        assert line.state
        line.readable = low_confirmed
        watchdogs[0]()
        assert line.calls[-2:] == [False, "sense"]
        assert not line.state and line.opened
        assert live._tx is None
        assert not rig._down.is_set()
        assert rig._proc is cat and closes == []
        assert rig.retired is (not low_confirmed)
        return False
    live._tx_done = SimpleNamespace(clear=lambda: None, wait=drain)
    key = rig.ptt
    if guarded:
        tx = onair.RadioTx.__new__(onair.RadioTx)
        tx._p3_tx_admit = lambda: None
        key = tx._p3_key_guard(key)
    with pytest.raises(SystemExit, match="output stream never sent"):
        live.transmit(np.zeros(8), at=1024, key=key)
    assert not rig._down.is_set()
    assert rig._proc is cat and closes == []
    assert rig.ptt(True) is low_confirmed


def test_ota_without_line_records_only(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(outdir=str(tmp_path), dry_run=False, rig="g90",
                           audio_in="fake-input", listen=0.01)
    recorded = []
    monkeypatch.setattr(ota, "find_device", lambda name, direction: "fake-input")
    monkeypatch.setattr(ota, "_record", lambda *args: recorded.append(args))
    monkeypatch.setattr(ota, "Rig", lambda *a, **k: pytest.fail("rig open"))
    ota.run(args)
    assert recorded == [(0.01, "fake-input", str(tmp_path / "listen.wav"))]
    assert ota.NO_KEYING_LINE in capsys.readouterr().out


def test_failed_arm_closes_cat_after_releasing_line(line_rig, monkeypatch):
    rig, line = line_rig
    line.open_error = True
    closed = []
    def close():
        assert not line.opened
        closed.append(True)
    monkeypatch.setattr(rig, "_close", close)
    with pytest.raises(ptt.PttError):
        rig.arm()
    assert closed == [True]


def test_interrupted_arm_releases_line(line_rig):
    rig, line = line_rig
    def interrupt():
        raise KeyboardInterrupt
    line.open_hook = interrupt
    with pytest.raises(KeyboardInterrupt):
        rig.arm()
    assert not line.opened
    assert rig.retired


def test_removed_daemon_keyers_have_no_remaining_callers(tools):
    from hfmodem.besra import radio
    assert "OneShotRigctl" not in inspect.getsource(radio)
    bridge = (tools / "vara_rig_bridge.py").read_text()
    assert "_RigctldPtt" not in bridge
    assert "self._daemon" not in bridge


def test_besra_without_a_line_cannot_key(monkeypatch):
    import atexit
    from hfmodem.besra import radio
    monkeypatch.setattr(radio, "find_rigctl", lambda name: "fake-rigctl")
    rig = radio.Rig(1036, "fake-cat", 38400)
    atexit.unregister(rig._atexit_unkey)
    monkeypatch.setattr(rig, "_write", lambda *a: pytest.fail("PTT reached CAT"))
    assert rig.ptt(True) is False
    assert rig.ptt(False) is True
    rig.stop()


def test_bridge_without_a_line_refuses_before_opening(tools, monkeypatch):
    import vara_rig_bridge
    monkeypatch.setattr(vara_rig_bridge, "LineKeyer",
                        lambda *a, **kw: pytest.fail("unexpected line open"))
    with pytest.raises(ValueError, match="requires --line-ptt"):
        vara_rig_bridge.Rig("fake-cat:4532", armed=True, log=lambda m: None)
