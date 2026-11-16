# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The X6100 base adapter against a scripted guard and a datagram bus; no hardware."""
from __future__ import annotations

import math
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from hfmodem.core import x6100_base as xb
from hfmodem.core.ptt import PttError
from hfmodem.core.regulatory import Control, Emission, Unregulated
from hfmodem.core.rig import Rig, RigError


class Bus:
    def __init__(self, refusals=0):
        self.frames, self.refusals = [], refusals

    def write(self, data):
        self.frames.append(data)
        if self.refusals:
            self.refusals -= 1
            return False
        return True


class Guard:
    """The guard's end of the socket: hello, then whatever reports a test sends."""

    def __init__(self, *, version=b"V1.1.9 Jan 1 2026,r12", rx_word=0):
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock = ours
        ours.send(xb.HELLO.pack(b"XGH1", version, rx_word, 5000, 500, 14_000_000, 14_350_000))
        self.link = xb.GuardLink(theirs, heartbeat_s=0.01)
        self.seq = 0
        self.beats = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        try:
            while data := self.sock.recv(64):
                self.beats.append(xb.HEARTBEAT.unpack(data))
        except OSError:
            pass

    def report(self, tx, *, power=1, swr=11, age_s=0.0):
        self.seq += 1
        self.sock.send(xb.TELEMETRY.pack(b"XGT1", self.seq, time.monotonic_ns() - int(age_s * 1e9),
                                         xb.FLAG_TX if tx else 0, power, swr, 0, 0, 1, 0,
                                         14_103_500))
        deadline = time.monotonic() + 1
        while self.link.latest is None or self.link.latest.seq != self.seq:
            assert time.monotonic() < deadline
            time.sleep(0.001)

    def close(self):
        self.link.close()
        self.sock.close()


@pytest.fixture
def guard():
    g = Guard()
    yield g
    g.close()


def test_frame_is_the_recovery_helpers_bytes():
    assert xb.frame(0x1000 | xb.MODEM) == bytes([0x00, 0x30, 0x00, 0x50, 0x00, 0x00])
    assert xb.TX_BITS == 0x0004E010 and xb.KNOWN_BITS == 0x0005F01A


@pytest.mark.parametrize("word", [xb.MODEM, xb.MIC_PTT, xb.POWER_OFF, 0x80000000, 0x4])
def test_a_receive_word_cannot_key_power_off_or_carry_unknown_bits(word):
    with pytest.raises(ValueError):
        xb.Register12(word, Bus())


def test_register_keys_with_modem_over_the_receive_word_and_retries_like_the_library():
    bus = Bus(refusals=2)
    reg = xb.Register12(xb.ATU_ENABLED, bus)
    assert reg.write(True)
    assert bus.frames == [xb.frame(xb.ATU_ENABLED | xb.MODEM)] * 3
    assert [w.accepted for w in reg.writes] == [True]
    assert reg.writes[0].end_ns >= reg.writes[0].begin_ns + 2 * xb.RETRY_S * 1e9
    assert reg.write(False) and bus.frames[-1] == xb.frame(xb.ATU_ENABLED)


def test_ptt_senses_only_an_accepted_write():
    bus = Bus()
    ptt = xb.X6100Ptt(xb.Register12(0, bus))
    assert ptt.sense() is None
    ptt.assert_(False)
    assert ptt.sense() is False
    bus.refusals = 3
    with pytest.raises(PttError):
        ptt.assert_(True)
    assert ptt.sense() is None
    ptt.release()
    assert ptt.released_low is None, "an accepted write is not RF off"
    assert ptt.sense() is False
    with pytest.raises(PttError, match="released"):
        ptt.assert_(True)
    assert bus.frames[-1] == xb.frame(0)


def test_datagram_socket_stands_in_for_the_i2c_device():
    with tempfile.TemporaryDirectory(dir="/tmp") as root:
        path = str(Path(root) / "i2c")
        sink = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sink.bind(path)
        bus = xb.I2cBus(path)
        try:
            assert xb.Register12(0, bus).write(True)
            assert sink.recv(16) == xb.frame(xb.MODEM)
        finally:
            bus.close()
            sink.close()


def test_link_reads_the_hello_reports_and_sends_heartbeats(guard):
    link = guard.link
    assert link.base_version.startswith("V1.1.9") and link.band == (14_000_000, 14_350_000)
    assert link.tx() is None
    guard.report(True, power=1, swr=12)
    assert link.tx() is True
    assert link.latest.power_w == pytest.approx(0.1) and link.latest.swr == pytest.approx(1.2)
    guard.report(False, age_s=1.0)
    assert link.tx() is None, "a stale report says nothing about now"
    link.expect_ms = 750
    time.sleep(0.05)
    beats = guard.beats
    assert beats and beats[-1][0] == b"XGB1" and beats[-1][2:] == (1, 750)


def test_wait_for_needs_a_report_newer_than_the_question(guard):
    guard.report(False)
    assert not guard.link.wait_for(False, 0.02)
    threading.Timer(0.02, guard.report, args=(False,)).start()
    assert guard.link.wait_for(False, 1.0)


def _rig(guard, bus):
    ptt = xb.X6100Ptt(xb.Register12(0, bus))
    cat = xb.X6100Cat(guard.link, xb.Register12(0, bus), dial_hz=14_103_000)
    rig = Rig(model="x6100", cat=cat, ptt=ptt, profile=Unregulated(because="dummy load bench test"),
              control=Control.LOCAL, transmit=True, max_key_s=5.0, identity="x6100")
    return rig, ptt


def test_rig_arms_keys_and_unkeys_through_register_12(guard):
    bus = Bus()
    rig, ptt = _rig(guard, bus)
    guard.report(False)
    with pytest.raises(RigError, match="not confirmed low"):
        rig.arm(prove_ptt=False)
    ptt.assert_(False)
    report = rig.arm(prove_ptt=False)
    assert "X6100" in report.model
    token = rig.key(Emission(14_103_000, 1000, 2000), why="test", duration_s=0.2)
    assert bus.frames[-1] == xb.frame(xb.MODEM)
    rig.unkey(token)
    assert bus.frames[-1] == xb.frame(0) and not rig.keyed


def test_rig_refuses_to_arm_over_a_transmitting_base(guard):
    rig, ptt = _rig(guard, Bus())
    ptt.assert_(False)
    guard.report(True)
    with pytest.raises(RigError, match="already transmitting"):
        rig.arm(prove_ptt=False)


def test_cat_refuses_to_tune_and_its_emergency_unkey_waits_for_the_base(guard):
    bus = Bus()
    cat = xb.X6100Cat(guard.link, xb.Register12(0, bus), dial_hz=14_103_000)
    with pytest.raises(RigError):
        cat.set_freq(7_100_000)
    assert cat.emergency_unkey() is None, "no fresh report: down is unconfirmed"
    threading.Timer(0.05, guard.report, args=(False,)).start()
    assert cat.emergency_unkey() is False
    assert bus.frames[-1] == xb.frame(0)


def test_every_write_is_register_12():
    # Register 21 bit 17 powered the base off; nothing here addresses any register but 12.
    bus = Bus()
    reg = xb.Register12(xb.ATU_ENABLED, bus)
    reg.write(True)
    reg.write(False)
    assert {frame[:2] for frame in bus.frames} == {b"\x00\x30"}


def test_open_writes_the_receive_word_so_the_line_reads_low():
    bus = Bus()
    ptt = xb.X6100Ptt(xb.Register12(xb.ATU_ENABLED, bus))
    ptt.open()
    assert bus.frames == [xb.frame(xb.ATU_ENABLED)] and ptt.sense() is False


def _beats_until(guard, want, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(beat[2:] == want for beat in list(guard.beats)):
            return True
        time.sleep(0.001)
    return False


def test_a_key_is_declared_to_the_guard_before_the_line_rises_and_cleared_after(guard):
    bus = Bus()
    rig, ptt = _rig(guard, bus)
    ptt.open()
    guard.report(False)
    rig.arm(prove_ptt=False)
    declared = []
    write = bus.write
    bus.write = lambda data: (declared.append(guard.link.expect_ms), write(data))[1]
    token = rig.key(Emission(14_103_000, 1000, 2000), why="test", duration_s=0.2)
    assert declared[-1] == math.ceil(rig._deadline(0.2, 0.0) * 1000)
    assert _beats_until(guard, (1, declared[-1]))
    rig.unkey(token)
    assert guard.link.expect_ms == 0 and _beats_until(guard, (0, 0))


def test_an_open_key_declares_the_guards_own_limit(guard):
    cat = xb.X6100Cat(guard.link, xb.Register12(0, Bus()), dial_hz=14_103_000)
    cat.protect(0, None)
    assert guard.link.expect_ms == guard.link.max_key_ms == 5000


def test_a_beat_never_carries_a_declaration_flag_without_its_length(guard):
    sent = []

    class Recorder:
        def __init__(self, sock):
            self.sock = sock

        def send(self, data):
            sent.append(xb.HEARTBEAT.unpack(data))
            return len(data)

        def __getattr__(self, name):
            return getattr(self.sock, name)

    guard.link.sock = Recorder(guard.link.sock)
    stop = threading.Event()

    def flip():
        while not stop.is_set():
            guard.link.declare(750)
            guard.link.declare(0)

    flipper = threading.Thread(target=flip)
    flipper.start()
    for _ in range(20000):
        guard.link.heartbeat()
    stop.set()
    flipper.join()
    guard.link.sock = guard.link.sock.sock
    assert len(sent) > 20000
    assert all(bool(expect) == bool(ms) for _, _, expect, ms in sent)


def test_a_key_inside_the_bases_tx_run_declares_the_whole_run(guard):
    cat = xb.X6100Cat(guard.link, xb.Register12(0, Bus()), dial_hz=14_103_000)
    guard.report(True, age_s=0.120)
    guard.report(True)
    declared = cat.protect(0, 48 * 1000)
    assert 1120 <= declared < 1300 and guard.link.expect_ms == declared
    cat.unprotect(declared)
    guard.report(False)
    assert cat.protect(0, 48 * 1000) == 1000


def test_the_ptt_proof_waits_for_the_bases_report(guard):
    bus = Bus()
    rig, ptt = _rig(guard, bus)
    ptt.open()
    guard.report(False)
    rig.arm(prove_ptt=False)
    keyed = threading.Event()
    write = bus.write
    bus.write = lambda data: (keyed.set() if data == xb.frame(xb.MODEM) else None, write(data))[1]

    def base():
        keyed.wait(2)
        time.sleep(0.08)  # the TX report lands after the proof's 50 ms read
        guard.report(True)

    threading.Thread(target=base, daemon=True).start()
    report = rig.arm(prove_ptt=True)
    assert report.ptt_proven, report.notes


def test_a_lost_guard_link_is_handed_to_the_owner(guard):
    cat = xb.X6100Cat(guard.link, xb.Register12(0, Bus()), dial_hz=14_103_000)
    lost = threading.Event()
    reasons = []
    cat.lost = lambda why: (reasons.append(why), lost.set())
    cat.open()
    guard.sock.close()
    assert lost.wait(2) and reasons[0].startswith("guard link lost")
    cat.close()


def test_without_the_guard_there_is_no_receive_word_and_no_rig(monkeypatch):
    monkeypatch.delenv("GUIFREE_GUARD_FD", raising=False)
    xb.guard_link.cache_clear()
    with pytest.raises(RigError, match="GUIFREE_GUARD_FD"):
        xb.onboard(14_103_000)


def test_onboard_keys_the_guards_receive_word_on_its_own_bus_handles(monkeypatch):
    word = xb.ATU_ENABLED | xb.SPLIT
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    ours.send(xb.HELLO.pack(b"XGH1", b"V1.1.8", word, 3000, 500, 0, 0))
    monkeypatch.setenv("GUIFREE_GUARD_FD", str(theirs.detach()))
    xb.guard_link.cache_clear()
    with tempfile.TemporaryDirectory(dir="/tmp") as root:
        path = str(Path(root) / "i2c")
        sink = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sink.bind(path)
        try:
            cat = xb.onboard(14_103_000, i2c=path)
            line = cat.keying_line()
            assert cat.register.rx_word == line.register.rx_word == word
            assert line.register.bus is not cat.register.bus
            line.assert_(True)
            assert sink.recv(16) == xb.frame(word | xb.MODEM)
            assert xb.onboard(14_103_000, i2c=path).link is cat.link
        finally:
            xb.guard_link().close()
            xb.guard_link.cache_clear()
            sink.close()
            ours.close()
