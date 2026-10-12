# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Bounded KI0BK polarity experiment, not proof of the gateway's RX phase.

KI0BK 0909-0855: coherent CONNECT slots 11..14 were inverted/normal/
inverted/normal; its CS4 replies used the opposite senses. The existing
alignment made first DATA slot 15 normal. Successful direct-CS4 KB5LZK
0908-1856 preserved call phase with the same lowercase announcement bytes.

Exercise real event delivery, grid aiming and TX rendering. The fixed-sense
peer reader uses an independent bitwise CRC residue, with the opposite-sense
read as a negative control. No devices, realtime waits or full-frame scans.
"""
from types import SimpleNamespace

import pytest

from hfmodem.shrike import onair, pactor1, ptc, rxfront, spec
from hfmodem.shrike.arq import IRS, ISS, State
from hfmodem.shrike.p4probe import EntryProbe
from hfmodem.tests.shrike.test_armdefaults import _armed, _parsed
from hfmodem.tests.shrike.test_p1peer import grid, peer_read


def setup(tmp_path, phase="reply", *, p1_only=True, p4=False):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.p1_setup_phase = phase
    if p4:
        tx.p4_probe = EntryProbe(2, 20.0, packet="robust", values=(7, 38),
                                 headers=("A", "A"), trigger_word="P1 ACK")
    emitted = []
    tx._tx = lambda audio, what, **kw: emitted.append((audio, what))
    host = ptc.PtcHost(peer=tx, mycall="W9SSJ")
    tx.attach(host)
    host.stay_in_pactor1 = p1_only
    if p4:
        # `--after-connect p1-robust`, the console's --p4-mail arm.
        host.p1_grant_only = host.p4_after_first_ack = True
    raster = grid()
    tx.aim(raster, 14)
    host.arq.on_host_connect("W9SSJ", "KI0BK")
    rx = onair._SessionRx(host)
    return SimpleNamespace(tx=tx, host=host, raster=raster, rx=rx, emitted=emitted)


def answer(s, slot, cs=pactor1.CS_SPEED, *, opposite=True):
    at = s.raster.rx_due(slot)
    # Like the live loop, aim the upcoming key before delivering RX events:
    # a request may render its retry immediately, before the following tick.
    s.tx.aim(s.raster, slot + 1)
    # The observed phase is measured against the original coherent call train,
    # not recomputed from whichever phase the current implementation adopted.
    sense = bool(slot & 1) ^ opposite
    ev = rxfront.Event(at / onair.FS, "cs", "fixture codeword",
                       protocol="PACTOR-1", cs=cs, sense=int(sense))
    s.rx._on(ev, anchored=True)
    onair._align_shift(s.raster, s.rx, s.tx, slot + 1, at)


def send(s, slot):
    s.tx.aim(s.raster, slot)
    s.host.tick()
    audio, what = s.emitted[-1]
    assert what.startswith("P1 pkt#"), what
    # RadioTx's renderer supplies 50 ms silence at either end; the fixed-phase
    # reader begins at the first symbol, exactly as its existing peer tests do.
    return audio[round(.05 * onair.FS):]


@pytest.mark.parametrize("phase,opposite,expected", [
    ("reply", True, False), ("call", True, True),
    ("reply", False, True), ("call", False, True),
])
@pytest.mark.parametrize("p4", [False, True])
def test_first_emission_phase_and_bytes(tmp_path, phase, opposite, expected, p4):
    s = setup(tmp_path, phase, p1_only=not p4, p4=p4)
    answer(s, 14, opposite=opposite)
    audio = send(s, 15)
    good = peer_read(audio, shift=int(expected))
    bad = peer_read(audio, shift=int(not expected))
    assert good["ok"], good
    assert good["bytes"] == bytes.fromhex("aa31773973736a0d1e01cd12")
    assert not bad["ok"], bad
    assert s.tx.boundary == 15 * 60000


@pytest.mark.parametrize("p4", [False, True])
def test_cs4_train_and_skipped_slot_keep_call_phase(tmp_path, p4):
    s = setup(tmp_path, "call", p1_only=not p4, p4=p4)
    for answered, keyed in [(14, 15), (15, 16), (17, 18), (18, 19)]:
        answer(s, answered)
        good = peer_read(send(s, keyed), shift=keyed & 1)
        assert good["ok"] and good["counter"] == 1, good
        assert not s.rx.p1_setup_finished
    assert s.tx.p1_packets == 1
    assert s.tx.p1_seq == [1] * 4


def test_real_first_ack_retires_override(tmp_path):
    s = setup(tmp_path, "call")
    answer(s, 14)
    send(s, 15)
    before = s.host.arq._buffer_raw
    answer(s, 15, pactor1.CS_ACK_A)
    assert s.host.arq._buffer_raw < before
    assert s.rx.p1_setup_finished
    # The ACK itself resumes the normal reply-derived alignment policy.
    assert s.raster.shift(15) is False
    assert s.tx.invert is True  # pending slot 16 was re-aimed
    # A later first-block-like state/counter cannot re-arm the experiment.
    s.host._p1_first_block = True
    answer(s, 16)
    assert s.rx.p1_setup_finished
    assert s.raster.shift(16) is True


def test_p4_arm_keeps_call_phase_until_the_ack_that_requests_the_probe(tmp_path):
    s = setup(tmp_path, "call", p1_only=False, p4=True)
    answer(s, 14)
    send(s, 15)
    assert not s.tx.p4_probe.requested
    assert s.raster.shift(15) is True
    answer(s, 15, pactor1.CS_ACK_A)
    assert s.tx.p4_probe.requested
    assert s.rx.p1_setup_finished
    assert s.raster.shift(15) is False
    assert s.tx.invert is True


def test_p4_stall_request_releases_call_phase_without_an_ack(tmp_path):
    s = setup(tmp_path, "call", p1_only=False, p4=True)
    answer(s, 14)
    send(s, 15)
    s.tx.p4_probe.request(0, b"", 0)
    answer(s, 15)
    assert not s.rx.p1_setup_finished
    assert s.raster.shift(15) is False


def test_changeover_retires_override_permanently(tmp_path):
    s = setup(tmp_path, "call")
    answer(s, 14)
    send(s, 15)
    answer(s, 15, pactor1.CS_CHANGEOVER)
    assert s.host.arq.role == IRS
    assert s.rx.p1_setup_finished
    s.host.arq.role = ISS
    answer(s, 16)
    assert s.rx.p1_setup_finished
    assert s.raster.shift(16) is True


def test_200_baud_repeated_answer_is_not_an_ack(tmp_path):
    s = setup(tmp_path, "call")
    for slot in (14, 15, 16):
        answer(s, slot, pactor1.CS_ACK_A)
        send(s, slot + 1)
        assert not s.rx.p1_setup_finished
    assert s.tx.p1_packets == 1


@pytest.mark.parametrize("protocol,p1_only", [
    (spec.Protocol.PACTOR3, True), (spec.Protocol.PACTOR1, False),
])
def test_p3_and_unrestricted_arms_cannot_use_override(tmp_path, protocol, p1_only):
    s = setup(tmp_path, "call", p1_only=p1_only)
    s.host.protocol = protocol
    s.host.arq.state = State.CONNECTED
    s.rx.cs_log.append(SimpleNamespace(sense=1, protocol=protocol))
    onair._align_shift(s.raster, s.rx, s.tx, 15, s.raster.rx_due(14))
    assert s.raster.shift(14) is True
    assert s.tx.invert is False


@pytest.mark.parametrize("flags", [(), ("--mail-fetch",), ("--p1-grant-only",)])
def test_cli_default_remains_reply(flags):
    assert _armed(*flags).p1_setup_phase == "reply"


@pytest.mark.parametrize("flags", [("--pactor1-only",), ("--mail-fetch",)])
def test_cli_explicit_experiment_in_p1_arm(flags):
    assert _armed(*flags, "--p1-setup-phase", "call").pactor1_only


@pytest.mark.parametrize("flags,fix", [
    ((), "add --pactor1-only"),
    (("--p1-grant-only",), "replace --p1-grant-only with --pactor1-only"),
    (("--pactor3-only",), "replace --pactor3-only with --pactor1-only"),
    (("--p3-uninvited", "--mail-fetch"), "replace --p3-uninvited with --pactor1-only"),
])
def test_cli_refuses_experiment_in_upgrade_arm(flags, fix):
    with pytest.raises(SystemExit, match=f"P4-entry arms: {fix}, or enter P4 with --p4-entry"):
        _armed(*flags, "--p1-setup-phase", "call")


def _resolved(*argv):
    args = _parsed(*argv)
    onair._connect_defaults(args)
    onair._p4_entry_defaults(args)
    onair._arm_defaults(args)
    return args


P4_MAIL = ("--p1-grant-only", "--p1-status-bits45", "2", "--p1-robust-stall", "12",
           "--p4-entry", "--p4-probe-packet", "robust", "--p4-probe-values", "7,38",
           "--p4-probe-headers", "A,A", "--p4-probe-cadence", "consecutive",
           "--p4-link", "--p4-ladder", "long", "--p4-data", "body",
           "--after-connect", "p1-robust")


@pytest.mark.parametrize("flags", [P4_MAIL, ("--p4-entry",), ("--p4-mail",)])
def test_cli_accepts_experiment_on_p4_entry_arm(flags):
    args = _resolved(*flags, "--p1-setup-phase", "call")
    assert args.p1_setup_phase == "call" and args.p4_entry and not args.pactor1_only


@pytest.mark.parametrize("flags,early", [
    (("--after-connect", "chirp"), "--after-connect chirp"),
    (("--connect", "robust", "--after-robust", "p3"), "--after-robust p3"),
    (("--connect", "robust", "--after-robust", "p4-robust"), "--after-robust p4-robust"),
])
def test_cli_refuses_experiment_that_leaves_p1_before_the_ack(flags, early):
    with pytest.raises(SystemExit, match=f"{early} leaves PACTOR-1 before one: drop {early}"):
        _resolved(*flags, "--p1-setup-phase", "call")


def test_unmeasured_sense_does_not_move_grid(tmp_path):
    s = setup(tmp_path, "call")
    s.rx.cs_log.append(SimpleNamespace(sense=None))
    onair._align_shift(s.raster, s.rx, s.tx, 15, s.raster.rx_due(14))
    assert s.raster.shift(15) is True
