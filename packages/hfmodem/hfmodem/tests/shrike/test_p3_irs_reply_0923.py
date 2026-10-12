import pytest
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Three IRS faults from the 80 m arms of 2026-09-23.

WS8EOC `onair-0923-1918` keyed 45 acknowledgements 886.6-891.1 ms past the
peer's phase and none was taken, and its repeat gear keyed CS4 into the
gateway's unacknowledged changeover packet. VE3KPG `onair-0923-1910` faded,
logged PEER'S PACKET UNREAD every cycle from hold 13, and at holds 28-33
stopped answering on `no fresh packet clock` while the gateway kept keying.
Coordinates are the transcripts'.
"""
import numpy as np

from hfmodem.shrike import arq, onair, rxfront
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p3_reply_placement import (
    CYCLE, IDENTITY, PACKET_N, irs_grid)
from hfmodem.tests.shrike.test_repeat_gear_stall import linked, status_at

FS = onair.FS

#: 1918's CRC changeover frames and the raster they sat on (TX[20], TX[21]).
WS8EOC_FRAMES = (2028792, 2088822, 2148822, 2208969, 2268909)

#: Where every accepted IRS ack sat past the packet's phase: VE3KPG 0923-1910
#: TX[18/20/21] and 0921-0001 TX[22].
ACCEPTED_MS = (908.7, 912.1)


def ms_past(g, slot, phase):
    return ((g.boundary(slot) - phase) % CYCLE) / FS * 1e3


def test_ws8eoc_1918_reply_moves_onto_the_accepted_instant():
    g = irs_grid()
    for at in WS8EOC_FRAMES[:3]:
        g.note_p3_packet(at, PACKET_N, CYCLE, identity=IDENTITY)
        g.cycles += 1
    raster = g.p3_reply_phase()
    slot = (raster - g.anchor) // CYCLE + 1
    line = g.p3_reply_shift(slot)
    assert "answered 910 ms past its phase reference" in line
    placed = ms_past(g, slot, raster)
    assert ACCEPTED_MS[0] <= placed <= ACCEPTED_MS[1]
    assert abs(placed - 910.0) < 0.05  # 1918 keyed 890.


def test_the_long_answer_stays_two_slots_on():
    assert round((onair.P3_LONG_REPLY_S - onair.P3_REPLY_S) * FS) == 2 * CYCLE


def changeover(host, *, seq=0):
    """The gateway's CS3-headed `RMS`, as 1918 decoded it 33 times."""
    host.on_rx_event(rxfront.Event(0.1, "packet", "p3", protocol=Protocol.PACTOR3,
                                   packet=(1, seq, b"RMS", True), breakin=True))


def test_ws8eoc_1918_changeover_repeats_draw_no_speed_up():
    host, seam = linked(repeat_gear=3)
    for _ in range(33):
        changeover(host)
    assert arq.CS_SPEED_UP not in seam.words


def test_the_gear_returns_once_the_peer_has_advanced():
    host, seam = linked(repeat_gear=3)
    changeover(host)
    for _ in range(4):
        host.on_rx_event(rxfront.Event(0.1, "packet", "p3", protocol=Protocol.PACTOR3,
                                       packet=(1, status_at(1), b" Trim", True)))
    assert seam.words[-1] == arq.CS_SPEED_UP


#: 1910's last CRC frame before the fade was hold 11's; the refusal fired at hold
#: 27, sixteen cycles on, and held through hold 33.
LAST_CRC_HOLD, FIRST_REFUSED, LAST_REFUSED = 11, 27, 33


def faded_grid():
    """A corroborated raster whose last CRC frame is `LAST_CRC_HOLD`'s."""
    g = irs_grid()
    peer0 = 1_500_000
    for k in range(LAST_CRC_HOLD - 2, LAST_CRC_HOLD + 1):
        g.note_p3_packet(peer0 + k * CYCLE, PACKET_N, CYCLE, identity=IDENTITY)
        g.cycles += 1
    assert g._p3_raster_corroborated
    slot0 = (peer0 + round(onair.P3_REPLY_S * FS) - g.anchor) // CYCLE
    return g, peer0, slot0


def unread_window(g, peer0, k):
    """Where 1910's answer window opened, d 55 ms after our CS1 ended."""
    return peer0 + k * CYCLE + round((onair.P3_REPLY_S + 0.236 + 0.055) * FS)


def test_ve3kpg_1910_without_peer_energy_the_corroborated_comb_still_keys():
    """The ARQ's silence budget, not the reply clock, ends a corroborated comb
    (1653d8a8, VE3KPG 0924-0033); `test_p3_peer_gone_0924` is what lets a peer
    with no carrier of its own go."""
    g, peer0, slot0 = faded_grid()
    for hold in range(FIRST_REFUSED - 1, LAST_REFUSED + 1):
        assert g.p3_control_refusal(slot0 + hold) is None


def test_ve3kpg_1910_an_unread_packet_every_cycle_keeps_the_reply_keying():
    g, peer0, slot0 = faded_grid()
    for hold in range(13, LAST_REFUSED + 8):
        assert g.p3_control_refusal(slot0 + hold) is None
        g.note_peer_unread(unread_window(g, peer0, hold))
    assert g._p3_heard_at == peer0 + (LAST_REFUSED + 8) * CYCLE


def test_the_unread_reading_reaches_the_raster_only_when_the_peer_owes_a_packet(
        monkeypatch):
    g, peer0, slot0 = faded_grid()
    monkeypatch.setattr(onair._AnswerBand, "sight",
                        lambda self, *a, **kw: onair._SlotReading("unread", True))

    host, a = _answer_band_host()
    token = a.begin_receive_opportunity()
    # Explicit current supported-reader unit seam; no waveform/CRC claim.
    assert a.qualify_peer(token, a._link_generation, "unit-short-header")
    assert a.peer_qualified

    hold = 20
    ours_end = peer0 + hold * CYCLE + round((onair.P3_REPLY_S + 0.236) * FS)
    seg_start = ours_end + round(0.020 * FS)
    seg = np.zeros(CYCLE // 2, np.float32)
    onair._report_answer_band(onair._AnswerBand(), host, g, seg,
                              seg_start - ours_end, 6240, read=False,
                              seg_start=seg_start)
    assert g._p3_heard_at == peer0 + (hold + 1) * CYCLE
    a.role = arq.ISS
    before = g._p3_heard_at
    onair._report_answer_band(onair._AnswerBand(), host, g, seg,
                              seg_start - ours_end + 5 * CYCLE, 6240, read=False,
                              seg_start=seg_start + 5 * CYCLE)
    assert g._p3_heard_at == before


def _answer_band_host():
    io = arq.ArqIO()
    io.protocol = Protocol.PACTOR3
    a = arq.PactorArq(io, arq.ArqConfig())
    a.state = arq.State.CONNECTED
    a.role = arq.IRS
    a.entry_pending = False

    class Host:
        protocol = Protocol.PACTOR3
        arq = a

    return Host(), a


@pytest.mark.parametrize("evidence", ["unqualified", "stale-generation", "old-opportunity"])
def test_prior_row_history_cannot_substitute_for_current_peer_qualification(monkeypatch, evidence):
    g, peer0, slot0 = faded_grid()
    monkeypatch.setattr(onair._AnswerBand, "sight",
                        lambda self, *a, **kw: onair._SlotReading("unread", True))
    host, a = _answer_band_host()
    a.on_host_data(bytes(range(8)))
    assert bytes(a._outbuf) == bytes(range(8))
    token = a.begin_receive_opportunity()
    if evidence != "unqualified":
        assert a.qualify_peer(token, a._link_generation, "unit-short-header")
        assert a.peer_qualified
        if evidence == "stale-generation":
            a._link_generation += 1
        else:
            a.finish_receive_opportunity(token)
            current = a.begin_receive_opportunity()
            assert current != token
            assert not a.qualify_peer(token, a._link_generation, "unit-short-header")
    assert not a.peer_qualified
    hold = 20
    ours_end = peer0 + hold * CYCLE + round((onair.P3_REPLY_S + 0.236) * FS)
    seg_start = ours_end + round(0.020 * FS)
    seg = np.zeros(CYCLE // 2, np.float32)
    before = g._p3_heard_at
    credit = (a.tx_acked_bytes, a.rx_seq, a.tx_seq, bytes(a._outbuf))
    onair._report_answer_band(onair._AnswerBand(), host, g, seg,
                              seg_start - ours_end, 6240, read=False,
                              seg_start=seg_start)
    assert g._p3_heard_at == before
    assert (a.tx_acked_bytes, a.rx_seq, a.tx_seq, bytes(a._outbuf)) == credit
