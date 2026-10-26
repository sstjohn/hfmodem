# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.core import ptt as core_ptt
from hfmodem.tests.kestrel import corpora

bridge = corpora.harness("vara_rig_bridge")


class FakeLineKeyer:
    def __init__(self, device, log):
        self.device = device
        self.calls = []
        self.must_retire = False
        self.result = True
        self.high = False
        self.closed = True
        self.transport = self
        self.released_low = None
        self.release_result = True

    def sense(self):
        return self.high

    def arm(self):
        self.calls.append("arm")
        self.closed = False

    def key(self, on, why=""):
        self.calls.append(on)
        if self.result:
            self.high = on
        return self.result

    def hand_back(self):
        self.calls.append("hand_back")
        self.high = False
        self.closed = True
        self.released_low = self.release_result


@pytest.fixture
def rigs(monkeypatch):
    monkeypatch.setattr(bridge, "LineKeyer", FakeLineKeyer)
    monkeypatch.setattr(bridge, "_id_reminders_owed", lambda log: False)
    monkeypatch.setattr(core_ptt, "drop_rts", lambda device, log: True)

    def no_cat(*args, **kwargs):
        raise AssertionError("PTT reached CAT")

    monkeypatch.setattr(bridge.Rig, "_cmd", no_cat)
    built = []

    def build(**kwargs):
        rig = bridge.Rig("127.0.0.1:1", log=lambda msg: None, **kwargs)
        built.append(rig)
        return rig

    yield build
    for rig in built:
        rig.shutdown()


@pytest.mark.parametrize("line,device", [(False, None), (False, "/dev/fake"),
                                         (True, None)])
def test_armed_rig_requires_a_line_before_opening_anything(rigs, line, device):
    with pytest.raises(ValueError, match="requires --line-ptt"):
        rigs(armed=True, line_ptt=line, ptt_device=device)


def test_line_keys_unkeys_and_hands_back_without_cat(rigs):
    rig = rigs(armed=True, line_ptt=True, ptt_device="/dev/fake")
    line = rig._keyer
    assert line.calls == ["arm"]
    assert rig.key(True)
    assert rig.keyed and rig.key_since
    assert rig.key(False)
    assert not rig.keyed and not rig.key_since
    assert rig.key(True)
    assert rig.unkey_hard("watchdog")
    assert not rig.keyed
    rig.shutdown()
    assert line.calls == ["arm", True, False, True, False, False, "hand_back"]
    assert not rig.key(True)


def test_failed_line_unkey_retires_without_cat(rigs):
    rig = rigs(armed=True, line_ptt=True, ptt_device="/dev/fake")
    assert rig.key(True)
    rig._keyer.result = False
    assert not rig.drop("test")
    assert rig.retired
    assert not rig.key(True)


@pytest.mark.parametrize("high", [False, True])
def test_line_state_reads_the_held_line_without_cat_or_open(rigs, monkeypatch, high):
    rig = rigs(armed=True, line_ptt=True, ptt_device="/dev/fake")
    line = rig._keyer
    line.high = high

    def no_open(*args, **kwargs):
        raise AssertionError("opened a line to read its state")

    monkeypatch.setattr(bridge, "LineKeyer", no_open)
    monkeypatch.setattr(line, "arm", no_open)
    assert rig.line_held
    assert rig.line_state() is high
    assert line.calls == ["arm"]


def test_line_state_without_a_held_line_is_unknown(rigs, monkeypatch):
    def no_open(*args, **kwargs):
        raise AssertionError("opened a line to read its state")

    monkeypatch.setattr(bridge, "LineKeyer", no_open)
    rig = rigs(armed=False, line_ptt=True, ptt_device="/dev/fake")
    assert not rig.line_held
    assert rig.line_state() is None
    assert rig.released_low is None
    rig.shutdown()
    assert rig.released_low is None


def test_line_state_read_error_is_unknown(rigs, monkeypatch):
    rig = rigs(armed=True, line_ptt=True, ptt_device="/dev/fake")

    def unreadable():
        raise OSError("line unavailable")

    monkeypatch.setattr(rig._keyer.transport, "sense", unreadable)
    assert rig.line_held
    assert rig.line_state() is None


@pytest.mark.parametrize("released_low", [True, False, None])
def test_shutdown_records_line_release_readback(rigs, released_low):
    rig = rigs(armed=True, line_ptt=True, ptt_device="/dev/fake")
    line = rig._keyer
    line.release_result = released_low
    assert rig.released_low is None
    rig.shutdown()
    assert line.calls[-1] == "hand_back"
    assert rig.released_low is released_low
    assert not rig.line_held
    assert rig.line_state() is None
    rig.shutdown()
    assert rig.released_low is released_low


def test_disarming_while_keyed_unkeys_then_closes_the_line(rigs, monkeypatch):
    rig = rigs(armed=True, line_ptt=True, ptt_device="/dev/fake")
    line = rig._keyer
    assert rig.key(True)
    assert line.high
    unkey = rig._unkey
    armed_at_unkey = []

    def unkey_before_disarming():
        armed_at_unkey.append(rig.armed)
        return unkey()

    monkeypatch.setattr(rig, "_unkey", unkey_before_disarming)
    rig.armed = False

    assert armed_at_unkey == [True]
    assert line.calls == ["arm", True, False, "hand_back"]
    assert not line.high and line.closed
    assert not rig.armed and not rig.keyed and not rig.key_since
    assert rig._keyer is None
    assert not rig.key(True)


@pytest.mark.parametrize("armed", [False, True])
def test_arming_after_shutdown_refuses_without_reopening_a_line(rigs, monkeypatch, armed):
    rig = rigs(armed=armed, line_ptt=True, ptt_device="/dev/fake")
    rig.shutdown()

    def no_line(*args, **kwargs):
        raise AssertionError("opened a line after shutdown")

    monkeypatch.setattr(bridge, "LineKeyer", no_line)
    with pytest.raises(ValueError, match="taken down"):
        rig.armed = True
    assert rig.retired
    assert rig._keyer is None
    assert not rig.key(True)


def test_failed_key_propagates_retirement(rigs):
    rig = rigs(armed=True, line_ptt=True, ptt_device="/dev/fake")
    rig._keyer.result = False
    rig._keyer.must_retire = True
    assert not rig.key(True)
    assert not rig.keyed
    assert rig.retired


def test_disarmed_cat_client_never_keys_or_opens_a_line(rigs):
    rig = rigs(armed=False, line_ptt=True, ptt_device="/dev/fake")
    assert rig._keyer is None
    assert not rig.key(True)
    assert rig.key(False)
    assert rig.drop("test")
    assert rig.unkey_hard("test")
    rig.shutdown()
    assert not hasattr(bridge, "_RigctldPtt")


@pytest.mark.parametrize("result", [True, False])
def test_lineless_disarmed_drop_uses_the_named_line(rigs, monkeypatch, result):
    calls = []
    monkeypatch.setattr(core_ptt, "drop_rts",
                        lambda device, log: (calls.append((device, log)), result)[1])
    rig = rigs(armed=False, ptt_device="/dev/fake")
    assert rig._keyer is None
    assert rig.drop("disarmed cleanup") is result
    assert calls == [("/dev/fake", rig.log)]


def test_lineless_drop_without_a_device_succeeds(rigs, monkeypatch):
    def no_device(*args):
        raise AssertionError("attempted a line drop without a device")

    monkeypatch.setattr(core_ptt, "drop_rts", no_device)
    rig = rigs(armed=False)
    assert rig.drop("no line")


def test_bridge_cli_refuses_armed_without_line_cleanly(monkeypatch, capsys, tmp_path):
    import sys

    def no_rig(*args, **kwargs):
        raise AssertionError("constructed a rig before refusing the flags")

    monkeypatch.setattr(bridge, "Rig", no_rig)
    monkeypatch.setattr(bridge, "arming_refusal", lambda device: None)
    monkeypatch.setattr(sys, "argv", ["vara_rig_bridge", "--mycall", "W9SSJ",
                                    "--arm", "--rigctld", "127.0.0.1:1",
                                    "--ptt-device", "/dev/fake",
                                    "--record", str(tmp_path)])
    assert bridge.main() == 2
    said = capsys.readouterr().out
    assert "--arm requires --line-ptt" in said
    assert "Traceback" not in said
