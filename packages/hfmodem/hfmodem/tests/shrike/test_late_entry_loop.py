# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Full device-free session loop: grants, late recorded P3 entry, and replies."""
import contextlib
import io
import json
import sys

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3rx, pactor1, rx, rxfront, spec
from hfmodem.tests.shrike.archive import requires_ws8eoc_p3
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.test_p3_gateway_acquisition import audio

FS = onair.FS
SLOT = 60000
MESSAGE = b"pending late-entry application bytes"


class LatePeer(_Bench):
    def __init__(self, entry_number, lose_first):
        super().__init__(seconds=150)
        self.entry_number = entry_number
        self.lose_first = lose_first
        self.call_at = None
        self.entries = []
        self.responses = []
        self.controls = []
        self.p3_phase = None

    def transmit(self, samples, **kwargs):
        first, end = super().transmit(samples, **kwargs)
        if self.call_at is None:
            self.call_at = first
            for slot in range(45):
                word = (pactor1.CS_59A if slot >= 6 else
                        pactor1.CS_SPEED if slot == 0 else
                        pactor1.CS_ACK_A if slot % 2 else pactor1.CS_ACK_B)
                signal = onair._trim_silence(pactor1.control_signal(
                    word, invert=bool(slot % 2))).astype(np.float32)
                at = first + slot * SLOT + 46080 + round(.09 * FS)
                self.audio[at:at + len(signal)] = signal
        if 35000 < len(samples) < 45000:
            packets = p3rx.decode_p3_packets(np.pad(samples, (2400, 2400))).packets
            for packet in packets:
                if packet.sl == 1 and packet.status & 0xfc == 0x18 and not packet.payload:
                    self.entries.append(first)
                    if len(self.entries) == self.entry_number:
                        self.start_p3(first)
        if len(samples) < FS // 2:
            padded = np.pad(samples, (2400, 2400))
            pulse = rx._pulse(FS // 100)
            tones = {cn: rx._baseband(padded, cn, FS, pulse)
                     for cn in rxfront.HDR_TONES}
            ci, errors, _ = rxfront._best_cs(tones, (len(pulse) - 1) // 2, 7200)
            self.controls.append((first, end, ci, errors))
        return first, end

    def start_p3(self, entry_at):
        # Retain the recorded PCM and its frequency error. The first head phase
        # is 0.90 s after the entry boundary, in front of the old P1 answer.
        # Repeat on the peer's own short-cycle clock, irrespective of local TX.
        phase = entry_at + round(.90 * FS)
        self.p3_phase = phase
        self.audio[phase - 3840:] = 0
        # A fixed tape cannot wait for an ACK. Include retransmissions rather
        # than assuming each single copy is received before the next counter.
        names = (["rms"] * 6 + ["trim"] * 3 + ["ode"] * 3
                 + ["ode-repeat"] * 3 + ["banner"] * 3)
        for k, name in enumerate(names):
            if k == 0 and self.lose_first:
                continue
            signal = audio(f"ws8eoc-0909-p3-{name}.wav")
            at = phase + k * SLOT - 3840
            self.audio[at:at + len(signal)] = signal
            self.responses.append((at, at + len(signal)))


def run_late(monkeypatch, tmp_path, entry_number, lose_first, *, scan_cost=.006):
    peer = LatePeer(entry_number, lose_first)
    hosts = []
    receivers = []

    class Host(onair.PtcHost):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.queued = False
            self.entry_result = None
            self.payload_protocols = []
            hosts.append(self)

        def tick(self, **kwargs):
            if not self.queued and self.arq.state in onair.LINKED:
                self.arq.on_host_data(MESSAGE)
                self.queued = True
            return super().tick(**kwargs)

        def on_rx_event(self, event):
            pending = self.arq.entry_pending
            queued = bytes(self.arq._outbuf)
            result = super().on_rx_event(event)
            if event.packet is not None and event.packet[2] and event.packet[3]:
                self.payload_protocols.append(self.protocol)
            if pending and not self.arq.entry_pending and self.arq.role == arq.IRS:
                self.entry_result = (queued, bytes(self.arq._outbuf))
            return result

    class Receiver(onair._SessionRx):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            receivers.append(self)

        def deep_scan(self, samples):
            result = super().deep_scan(samples)
            # September 11 warmed tracked reader: max 5.06 ms; charge the
            # full 6 ms reserved by the driver. The older 16.2 ms cost
            # remains an explicit overloaded-receiver case below.
            peer.spend(round(scan_cost * FS))
            return result

    monkeypatch.setattr(onair, "PtcHost", Host)
    monkeypatch.setattr(onair, "_SessionRx", Receiver)
    monkeypatch.setattr(onair, "_LiveInput", lambda *a, **kw: peer)
    monkeypatch.setattr(onair, "find_device", lambda *a, **kw: 0)
    monkeypatch.setattr(onair.ota, "Rig", _Rig)
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", [
        "shrike.onair", "--transmit", "--hold", "12", "--max-cycles", "8",
        "--mycall", "W9SSJ", "--dxcall", "K7ABC", "--dial", "7100000",
        "--serial", "/dev/null", "--outdir", str(tmp_path),
        "--p1-grant-only", "--p1-status-bits45", "3", "--p3-entry", "template",
        # THIS PEER IS A HYBRID AND ONLY A HYBRID CAN BE ASKED THIS. Its
        # transmissions are recorded WS8EOC PCM, kept with the real -50 Hz
        # frequency error `start_p3` names; its reader is
        # `p3rx.decode_p3_packets` at nominal. No station is built that way, and
        # `--p3-follow-offset all` answers the transmitter it can hear rather
        # than the receiver this scene imagines. What these scenes measure --
        # entry budget, slot accounting, fallback, delivery -- is the same at
        # either setting, and the follow has its own file.
        "--p3-follow-offset", "none"])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        onair.main()
    log = output.getvalue()
    (tmp_path / "session.log").write_text(log)
    (tmp_path / "result.json").write_text(json.dumps(dict(
        respond_after_entries=entry_number, first_response_lost=lose_first,
        emitted_entries=peer.entries, peer_phase=peer.p3_phase,
        controls=peer.controls,
        delivered=bytes(hosts[0].channel(hosts[0].ptchn).rx).decode("ascii"),
        pending_at_changeover=None if hosts[0].entry_result is None else
        [part.hex() for part in hosts[0].entry_result]), indent=2) + "\n")
    return dict(peer=peer, host=hosts[0], rx=receivers[0], log=log)


@requires_ws8eoc_p3
@pytest.mark.parametrize("entry_number", [5, 8])
@pytest.mark.parametrize("lose_first", [False, True], ids=["reply", "first-lost"])
def test_late_recorded_changeover_through_complete_session_loop(
        monkeypatch, tmp_path, entry_number, lose_first):
    got = run_late(monkeypatch, tmp_path, entry_number, lose_first)
    assert len(got["peer"].entries) >= entry_number
    assert "changeover -> IRS" in got["log"]
    assert bytes(got["host"].channel(got["host"].ptchn).rx) == (
        b"RMS Trimode 1.4.3.0\r\nW9SSJ has 118 d")
    assert got["peer"].controls
    assert all(errors == 0 for _, _, _, errors in got["peer"].controls)
    assert any(ci == arq.CS_ACK for _, _, ci, _ in got["peer"].controls)
    assert any(ci == arq.CS_CYCLE_TOG for _, _, ci, _ in got["peer"].controls)
    assert max(got["peer"].entries) < got["peer"].controls[0][0]
    for first, end, _, _ in got["peer"].controls:
        phase = (first - got["peer"].p3_phase) % SLOT
        # Conservative envelope guard around the scripted 0.81 s P3 packets.
        assert phase - round(.04 * FS) >= round(.84 * FS)  # Include PTT settle.
        assert phase + end - first <= SLOT - round(.02 * FS)
    assert got["host"].entry_result is not None
    before, after = got["host"].entry_result
    assert before and before == after and MESSAGE.endswith(before)
    assert got["host"].payload_protocols
    assert all(p is spec.Protocol.PACTOR3 for p in got["host"].payload_protocols)
    # Session cleanup resets the next contact to P1 after the bounded goodbye.
    assert got["host"].arq.state is arq.State.DISCONNECTED
    assert got["host"].protocol is spec.Protocol.PACTOR1
    assert "entry retry budget exhausted" not in got["log"]


LOSS_MARKERS = ("IS GONE", "SHORT OF THE LISTEN FLOOR", "LATE TO THE KEY",
                "REGRID GAVE UP")


def before_the_last_delivery(log):
    """The part of a session transcript the peer was still being read in."""
    cut = log.rfind("HOLD RX (")
    assert cut > 0, "no delivery in this session at all"
    return log[:cut]


@requires_ws8eoc_p3
def test_old_reader_cost_costs_no_slot_and_no_delivery(monkeypatch, tmp_path):
    """September 11's 16.2 ms reader against the 6 ms one, same tape, same loop.

    THE LOOP IS THE ASSERTION AND NOT THE MARKER COUNT. An overloaded reader
    that loses a slot says so in one of the four lines below; one that loses the
    LOOP says nothing at all, and that is what the CS6 receive window did. Each
    `_regrid` try walked a whole new group of three slots and re-read the
    recovered window with a tracked decode the caller had just run over the same
    audio, so every try was late by exactly that decode and three tries cost
    twelve slots -- with `--hold` counted in cycles, the session never reached
    its own budget: 22 carriers in 150 s of bench clock against the 35 this tape
    gives in 85.

    WHAT THE OVERLOAD STILL COSTS IS IN THE CS6 PROBE TAIL AND NOWHERE ELSE. A
    reader spending 16.2 ms where `P3_DECODE_RESERVE_S` reserves 6 overruns its
    key, and the loop's answer is to hand the slot back and say so. Every one of
    those hand-backs falls after the peer's tape has run out, while this station
    is probing an empty channel with CS6; not one falls in the payload phase, at
    either reader cost, and the message is byte-identical.
    """
    slow = run_late(monkeypatch, tmp_path, 5, False, scan_cost=.0162)
    got = slow
    assert bytes(got["host"].channel(got["host"].ptchn).rx) == (
        b"RMS Trimode 1.4.3.0\r\nW9SSJ has 118 d")
    assert got["peer"].controls
    assert all(errors == 0 for _, _, _, errors in got["peer"].controls)
    for first, end, _, _ in got["peer"].controls:
        phase = (first - got["peer"].p3_phase) % SLOT
        assert phase - round(.04 * FS) >= round(.84 * FS)
        assert phase + end - first <= SLOT - round(.02 * FS)
    # NO SLOT LOST WHILE THE PEER IS BEING READ, in any of the four words the
    # loop has for losing one.
    carrying = before_the_last_delivery(got["log"])
    assert [m for m in LOSS_MARKERS if m in carrying] == []
    assert "SHORT OF THE LISTEN FLOOR" not in got["log"]
    # ...AND THE SESSION FINISHED. `_Bench` raises past its own 150 s limit, so
    # a run that reaches here ended on its hold budget rather than on the clock.
    # The unloaded reader is the control: same tape, same loop, and every
    # carrier of the payload phase in the same slot.
    quick = run_late(monkeypatch, tmp_path, 5, False, scan_cost=.006)
    assert [m for m in LOSS_MARKERS
            if m in before_the_last_delivery(quick["log"])] == []
    assert bytes(quick["host"].channel(quick["host"].ptchn).rx) == bytes(
        got["host"].channel(got["host"].ptchn).rx)
    slots = lambda run: [first // SLOT for first, _ in run["peer"].emissions]
    assert slots(slow)[:30] == slots(quick)[:30]
    assert len(slots(slow)) >= 30
