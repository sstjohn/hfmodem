# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""WS8EOC M06: repeated changeovers are ACKs owed, not a clean traffic run.

The recorded input is seq0/type0/RMS every time (morning M06 transcript,
2026-09-10 14:59:43 UTC). Continued packets below are a specified peer script,
not a second instance of our ARQ: stop-and-wait repeats seq0 until its ACK,
then sends seq1,2,3,0. The recording cannot establish that continuation.
"""

import pytest

from hfmodem.shrike import arq, spec


class _Air(arq.ArqIO):
    def __init__(self):
        self.controls = []
        self.packets = []
        self.data = bytearray()
        self.lines = []

    def send_cs(self, cs_index):
        self.controls.append(cs_index)

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((sl, payload, status, breakin))

    def deliver(self, blob):
        self.data.extend(blob)

    def log(self, message):
        self.lines.append(message)


def _linked():
    air = _Air()
    link = arq.PactorArq(air, arq.ArqConfig(speed_up_after=3))
    link.role = arq.ISS
    link._enter_connected()
    return link, air


def test_repeated_changeovers_do_not_drive_speed_and_traffic_can_continue():
    link, air = _linked()
    for _ in range(14):
        link.on_rx_packet(1, b"RMS", 0, True, breakin=True)
        assert link.role == arq.IRS
        assert link.state == arq.State.CONNECTED
        assert air.controls[-1] == arq.CS_ACK
    assert air.data == b"RMS"
    assert link.rx_progress == 1
    assert link._clean_run == 0

    # Distinct ordinary traffic earns the receiver's gear request, and lost ACKs
    # between the packets END the run rather than merely failing to advance it:
    # a repeat is our own acknowledgement not arriving, which is a reading of
    # the channel and not a neutral event (`arq._gear_cs`, round 24). Counter
    # wrap stays valid either way.
    for seq, payload in [(1, b" greeting"), (2, b" and"), (3, b" SID"),
                         (0, b" prompt")]:
        link.on_rx_packet(1, payload, seq, True)
        assert air.controls[-1] == arq.CS_ACK
        assert link._clean_run == 1
        for _ in range(4):
            link.on_rx_packet(1, payload, seq, True)
            assert air.controls[-1] == arq.CS_ACK
            assert link._clean_run == 0
    assert arq.CS_SPEED_UP not in air.controls
    assert air.data == b"RMS greeting and SID prompt"
    assert link.rx_progress == 5


def test_uninterrupted_traffic_is_what_the_gear_request_is_left_for():
    """The positive control for the run above: nothing between the packets."""
    link, air = _linked()
    link.role = arq.IRS
    for seq, payload in [(1, b"a"), (2, b"b"), (3, b"c")]:
        link.on_rx_packet(1, payload, seq, True)
    assert air.controls == [arq.CS_ACK, arq.CS_ACK, arq.CS_SPEED_UP]


def test_changeover_cannot_spend_an_existing_clean_run():
    link, air = _linked()
    link.role = arq.IRS
    link._clean_run = link.cfg.speed_up_after - 1
    link.on_rx_packet(1, b"RMS", 0, True, breakin=True)
    assert air.controls == [arq.CS_ACK]
    # ...and it ends it: the shortened changeover field is no evidence for the
    # traffic ladder, so the run it interrupts is over rather than banked.
    assert link._clean_run == 0


@pytest.mark.parametrize("sl", [arq.P1_SPEED_LEVEL, 1])
def test_breakin_during_goodbye_keeps_teardown_and_receives_new_field(sl):
    link, air = _linked()
    link.on_host_disconnect()
    link.on_cycle()
    assert link.state == arq.State.DISCONNECTING
    assert link.said_goodbye

    link.on_rx_packet(sl, b"RMS", 0, True, breakin=True)
    assert link.state == arq.State.DISCONNECTING
    assert link.role == arq.IRS
    assert air.data == b"RMS"
    assert link._qrt_pending and link._breakin_pending
    assert not link.goodbye_acked  # CS3 is not a completed close handshake.
    link.on_host_data(b"must not reopen the link")
    assert not link._outbuf

    # Peer insists on retaining the channel. The QRT remains owed and the
    # existing goodbye cycle budget still closes; no CONNECTED re-entry.
    for _ in range(arq.GOODBYE_CYCLES + 1):
        link.on_cycle()
        if link.state == arq.State.DISCONNECTED:
            break
        assert link.state == arq.State.DISCONNECTING
        assert air.packets[-1][2] & spec.STATUS_QRT
        link.on_rx_packet(sl, b"RMS", 0, True, breakin=True)
        assert link.state == arq.State.DISCONNECTING
    assert link.state == arq.State.DISCONNECTED
    assert air.data == b"RMS"
    assert not any("DISCONNECTING -> CONNECTED" in line for line in air.lines)


@pytest.mark.parametrize("sl", [arq.P1_SPEED_LEVEL, 1])
def test_peer_qrt_during_local_teardown_is_received_and_answered(sl):
    link, air = _linked()
    link.on_host_disconnect()
    link.on_cycle()
    link.on_rx_packet(sl, b"bye", spec.STATUS_QRT, True, breakin=True)
    assert air.data == b"bye"
    assert air.controls == [arq.CS_ACK]
    assert link.state == arq.State.DISCONNECTED


def test_p1_repeated_payload_keeps_ack_and_delivers_once():
    link, air = _linked()
    link.role = arq.IRS
    for _ in range(8):
        link.on_rx_packet(arq.P1_SPEED_LEVEL, b"P1", 1, True)
    assert air.data == b"P1"
    assert air.controls == [arq.CS_ACK] * 8
    assert link._clean_run == 0
