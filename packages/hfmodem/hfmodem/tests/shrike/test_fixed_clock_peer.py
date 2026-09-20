# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Decode actual P1 emissions against a peer clock fixed by the first call.

The five millisecond receive gate is a bench constraint, not an SCS tolerance.
The peer never reads the local ARQ state, packet arguments, or moving TX grid.
"""
import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, pactor1
from hfmodem.tests.shrike import test_grid as bench

FS = onair.FS
SLOT = 60000


class FixedPeer(bench._Bench):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.peer = None
        self.call_at = None
        self.received = []
        self.accepted = []
        self.last_count = None
        self.word = pactor1.CS_SPEED

    def answers(self, first_slot):
        # Stop after a bounded conversation so the production hold loop can end.
        for slot in range(first_slot, 24):
            at = self.call_at + slot * SLOT + 46080 + self.d_n
            signal = onair._trim_silence(pactor1.control_signal(
                self.word, invert=bool(slot % 2))).astype(np.float32)
            self.audio[at:at + len(signal)] = signal

    def transmit(self, audio, **kwargs):
        first, end = super().transmit(audio, **kwargs)
        if self.call_at is None:
            self.call_at = first
            # Delayed call answers arrive during the caller's setup hush.
            self.answers(6)
        for packet in p1rx.decode_p1_packets(np.pad(audio, (2400, 2400))):
            slot = round((first - self.call_at) / SLOT)
            error = first - (self.call_at + slot * SLOT)
            row = dict(slot=slot, error=error, count=packet.packet_count,
                       payload=packet.payload, inverted=packet.inverted)
            self.received.append(row)
            if abs(error) > round(.005 * FS) or packet.inverted != bool(slot % 2):
                continue
            if packet.packet_count != self.last_count:
                self.accepted.append(row)
                self.last_count = packet.packet_count
                self.word = (pactor1.CS_ACK_B if self.word == pactor1.CS_ACK_A
                             else pactor1.CS_ACK_A)
            self.answers(slot)
        return first, end


def run_peer(monkeypatch, turnaround, *, broken=False):
    monkeypatch.setattr(bench, "_Bench", FixedPeer)
    if broken:
        update = onair._MasterGrid.update

        def forget_link(self, *args, **kwargs):
            if kwargs.get("hushed"):
                kwargs["linked"] = False
            return update(self, *args, **kwargs)

        monkeypatch.setattr(onair._MasterGrid, "update", forget_link)
    return bench._session(cycles=16, hold=7, charge=0, peer=False,
                          seconds=200, d=turnaround)


@pytest.mark.parametrize("turnaround", [.075, .116125])
@pytest.mark.parametrize("broken", [False, True], ids=["fixed", "old-placement"])
def test_fixed_peer_acknowledges_only_decoded_packets_on_its_clock(
        monkeypatch, turnaround, broken):
    got = run_peer(monkeypatch, turnaround, broken=broken)
    peer = got["bench"]
    assert "HUSHED, not keying" in got["log"]
    assert "** CONNECTED to" in got["log"]
    assert peer.received, "No emitted DATA packet passed the peer's CRC decoder"
    if broken:
        assert not peer.accepted
        assert {row["count"] for row in peer.received} == {1}
        assert all(abs(row["error"]) > 240 for row in peer.received)
    else:
        assert len(peer.accepted) >= 3
        assert [row["count"] for row in peer.accepted[:3]] == [1, 2, 3]
        assert all(row["error"] == 0 for row in peer.accepted)
