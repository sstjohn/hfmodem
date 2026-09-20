# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Only emitted repeat ACKs count; a receive window settles after its readers."""
import pytest

from hfmodem.shrike import arq, spec
from hfmodem.shrike.ptc import PtcHost


class IO(arq.ArqIO):
    protocol = spec.Protocol.PACTOR3

    def __init__(self, deferred=True):
        self.deferred = deferred
        self.refuse = False
        self.pending = None
        self.controls, self.emissions, self.packets, self.logs = [], [], [], []
        self.delivered = bytearray()

    def send_cs(self, cs):
        physical = PtcHost._counter_cs_for(self, cs)
        self.controls.append(physical)
        if self.refuse:
            return arq.REFUSED
        if self.deferred:
            self.pending = physical
        else:
            self.emissions.append(physical)

    def emit(self):
        if self.pending is not None:
            self.emissions.append(self.pending)
            self.arq.on_cs_emitted(self.pending)
            self.pending = None

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((sl, payload, status, breakin))
        return arq.REFUSED if self.refuse else len(payload)

    def defer_rx_close(self):
        return self.deferred

    def deliver(self, payload):
        self.delivered.extend(payload)

    def log(self, message):
        self.logs.append(message)


def linked(*, deferred=True, repeat_gear=3):
    io = IO(deferred)
    a = io.arq = arq.PactorArq(io, arq.ArqConfig(
        repeat_gear=repeat_gear, long_cycle=False, speed_up_after=1000))
    a._enter_connected()
    a.role = arq.IRS
    a._reset_rx_seq()
    return a, io


def packet(a, seq=1, *, payload=b"repeat", crc=True, long=False):
    a.on_rx_packet(3, payload, seq, crc, protocol=spec.Protocol.PACTOR3,
                   cycle_long=long)


def emitted_repeats(a, io, count, seq=1):
    for _ in range(count):
        packet(a, seq)
        io.emit()


def test_many_deferred_frames_count_as_one_emitted_reply():
    a, io = linked()
    for _ in range(8):
        packet(a)
    assert a._repeat_run == 0
    assert io.pending == arq.CS_REQUEST  # Physical ACK for counter 1.
    assert io.emissions == []
    assert io.delivered == b"repeat"
    io.emit()
    assert a._repeat_run == 1
    assert io.emissions == [arq.CS_REQUEST]


@pytest.mark.parametrize("seq", [0, 1, 2, 3])
def test_fourth_actually_answered_repeat_selects_gear(seq):
    a, io = linked()
    emitted_repeats(a, io, 3, seq)
    assert a._repeat_run == 3
    packet(a, seq)
    assert io.pending == arq.CS_SPEED_UP
    assert a._repeat_run == 3  # Selection is not an emission.
    io.emit()
    assert a._repeat_run == 0
    assert io.emissions == [seq & 1] * 3 + [arq.CS_SPEED_UP]
    assert sum("identical packets answered" in line for line in io.logs) == 1


def test_refused_gear_does_not_consume_emitted_run():
    a, io = linked()
    emitted_repeats(a, io, 3)
    io.refuse = True
    packet(a)
    assert a._repeat_run == 3 and a._repeat_pending is None
    io.refuse = False
    packet(a)
    assert io.pending == arq.CS_SPEED_UP
    io.emit()
    assert a._repeat_run == 0


def test_superseded_gear_commits_only_new_counter_reply():
    a, io = linked()
    emitted_repeats(a, io, 3)
    packet(a)
    assert io.pending == arq.CS_SPEED_UP
    packet(a, 2, payload=b"new")
    assert io.pending == arq.CS_ACK
    io.emit()
    assert a.rx_seq == 2 and a._repeat_run == 1
    assert arq.CS_SPEED_UP not in io.emissions
    assert io.delivered == b"repeatnew"
    assert not any("identical packets answered" in line for line in io.logs)


def test_new_counter_replaces_older_deferred_parity_word():
    a, io = linked()
    packet(a, 1, payload=b"first")
    packet(a, 2, payload=b"second")
    a.on_cycle(elapsed_ticks=3)
    io.emit()
    assert io.emissions == [arq.CS_ACK]
    assert a._repeat_run == 1 and a.rx_seq == 2
    assert io.delivered == b"firstsecond"


def test_synchronous_legacy_reply_commits_without_driver_callback():
    a, io = linked(deferred=False)
    for _ in range(4):
        packet(a)
    assert io.emissions == [arq.CS_REQUEST] * 3 + [arq.CS_SPEED_UP]
    assert a._repeat_run == 0


def test_quiet_parity_replies_do_not_invent_decoded_repeats():
    a, io = linked()
    emitted_repeats(a, io, 1)
    a.on_cycle()
    for _ in range(3):
        a.on_cycle()
        io.emit()
    assert a._repeat_run == 1
    assert arq.CS_SPEED_UP not in io.emissions


def test_emitted_cycle_command_resets_stall_history():
    a, io = linked()
    emitted_repeats(a, io, 3)
    packet(a, 2, long=True)
    assert io.pending == arq.CS_CYCLE_TOG
    io.emit()
    assert a._repeat_run == 0 and a._repeat_pending is None


def test_bad_crc_cancels_pending_repeat_identity():
    a, io = linked()
    packet(a)
    packet(a, crc=False)
    io.emit()
    assert io.emissions == [arq.CS_REQUEST]  # First error: MAXDown not reached.
    assert a._repeat_run == 0 and not io.delivered.endswith(b"repeatrepeat")


def threshold_window():
    a, io = linked()
    a._silent_cycles = a.cfg.max_retries
    token = a.begin_receive_opportunity()
    a.on_cycle()
    io.emit()
    assert a._silent_cycles == a.cfg.max_retries and not a._qrt_pending
    return a, io, token


def test_late_crc_precedes_timeout_and_finish_emits_nothing():
    a, io, token = threshold_window()
    emitted = list(io.emissions)
    packet(a, 2, payload=b"late packet")
    a.finish_receive_opportunity(token)
    assert a._silent_cycles == 0 and not a._qrt_pending
    assert io.delivered == b"late packet"
    assert io.emissions == emitted and io.packets == []
    assert io.pending == arq.CS_ACK  # Driver owns when this later ACK emits.


def test_current_occupied_window_forgives_once_without_resetting_count():
    a, io, token = threshold_window()
    for _ in range(3):
        a.note_unreadable_answer()
    a.note_burst(-17, at_anchor=False)
    a.finish_receive_opportunity(token)
    assert a._silent_cycles == a.cfg.max_retries and not a._qrt_pending
    assert not a._burst_at_anchor
    assert sum("not counting this cycle" in line for line in io.logs) == 1
    next_token = a.begin_receive_opportunity()
    a.on_cycle()
    io.emit()
    a.finish_receive_opportunity(next_token)
    assert a._silent_cycles == a.cfg.max_retries + 1 and a._qrt_pending


@pytest.mark.parametrize("evidence", ["none", "off_anchor", "unpositioned"])
def test_nonoccupancy_evidence_does_not_hold_threshold(evidence):
    a, io, token = threshold_window()
    if evidence == "off_anchor":
        a.note_burst(-17, at_anchor=False)
    elif evidence == "unpositioned":
        a.note_peer_heard()
    a.finish_receive_opportunity(token)
    assert a._silent_cycles == a.cfg.max_retries + 1
    assert a._qrt_pending and io.packets == []


def test_legacy_callers_still_charge_timeout_immediately():
    a, _ = linked()
    a._silent_cycles = a.cfg.max_retries
    a.on_cycle()
    assert a._qrt_pending


def test_begin_drops_weak_evidence_from_previous_window():
    a, _ = linked()
    a.note_unreadable_answer()
    token = a.begin_receive_opportunity()
    a.on_cycle()
    a.finish_receive_opportunity(token)
    assert a._silent_cycles == 1


def test_opportunity_token_rejects_nesting_stale_and_duplicate_finish():
    a, _ = linked()
    token = a.begin_receive_opportunity()
    with pytest.raises(RuntimeError):
        a.begin_receive_opportunity()
    with pytest.raises(ValueError):
        a.finish_receive_opportunity(token + 1)
    with pytest.raises(ValueError):
        a.finish_receive_opportunity(True)
    a.finish_receive_opportunity(token)
    with pytest.raises(RuntimeError):
        a.finish_receive_opportunity(token)


def test_multiple_ticks_in_one_receive_window_charge_only_one_miss():
    a, _ = linked()
    token = a.begin_receive_opportunity()
    for _ in range(3):
        a.on_cycle()
    a.finish_receive_opportunity(token)
    assert a._silent_cycles == 1


def test_finishing_after_disconnect_does_not_restart_link_or_transmit():
    a, io, token = threshold_window()
    a.on_host_abort()
    a.finish_receive_opportunity(token)
    assert a.state == arq.State.DISCONNECTED
    assert not a._qrt_pending and io.packets == []
