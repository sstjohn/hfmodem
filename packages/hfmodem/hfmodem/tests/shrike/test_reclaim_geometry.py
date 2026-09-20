# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Bare control signals must not acquire a data packet's duration during recovery."""
import pytest

from hfmodem.shrike import arq, onair, ptc, spec
from hfmodem.tests.shrike.test_grid import _Bench, _Rig


def _grid(protocol=spec.Protocol.PACTOR1):
    fs = onair.FS
    data, cs = ((0.960, 0.120) if protocol == spec.Protocol.PACTOR1
                else (0.810, 0.210))
    g = onair._MasterGrid(0, round(1.25 * fs), 0,
                          packet_n=round(data * fs), cs_n=round(cs * fs),
                          d_max_n=round(0.130 * fs))
    g.sending = False
    g.d_n = 0.075 * fs
    g.d_ref_n = g.cs_n
    return g


def test_recorded_cs2_does_not_trigger_a_phantom_789ms_collision():
    g = _grid()
    at = 3013244  # KB5LZK, 2026-09-08, session 62.78 seconds
    g.note_peer_codeword(at, g.cs_n, "CS2/ack", "KB5LZK")
    g.peer_onset = at
    assert g.key_refusal(at + round(0.171 * onair.FS),
                         round(0.160 * onair.FS)) is None


@pytest.mark.parametrize("protocol", [spec.Protocol.PACTOR1, spec.Protocol.PACTOR3])
@pytest.mark.parametrize("name", ["CS1/ack", "CS2/ack", "CS4/100Bd",
                                  "ACK", "REQ", "SPEED-UP", "NAK", "CYCLE-TOG"])
def test_reclaim_occupies_the_gap_after_a_bare_control_signal(protocol, name, tmp_path):
    g = _grid(protocol)
    slot = 10
    at = g.boundary(slot) - g.cs_n - g.peer_read_gap
    g.note_peer_codeword(at, g.cs_n, name, "KB5LZK")
    g.peer_onset = at
    tx = onair.RadioTx(None, transmit=False, settle=0.04, outdir=tmp_path)
    tx.aim(g, slot)
    assert g.peer_packet_end(slot).end == at + g.cs_n
    assert tx._breakin_key(g, slot) == g.boundary(slot)
    carrier = g.boundary(slot) - round(tx.settle * onair.FS)
    assert g.key_refusal(carrier, g.data_n + round(tx.settle * onair.FS),
                         changeover=True) is None


@pytest.mark.parametrize("evidence", ["CS3", "energy", "stale", "off_phase"])
def test_a_packet_head_or_unattributed_burst_keeps_the_full_packet_guard(evidence):
    g = _grid()
    at = 10 * g.cycle_n
    if evidence != "energy":
        g.note_peer_codeword(at, g.cs_n,
                             "CS3/break-in" if evidence == "CS3" else "CS2/ack",
                             "KB5LZK")
    if evidence == "stale":
        g.cycles += onair.ISS_GUARD_CYCLES + 1
        at += (onair.ISS_GUARD_CYCLES + 1) * g.cycle_n
    elif evidence == "off_phase":
        at += round(0.100 * onair.FS)
    g.peer_onset = at
    assert g.peer_packet_end(12).end == at + g.data_n
    assert g.key_refusal(at + round(0.171 * onair.FS),
                         round(0.160 * onair.FS), changeover=True) is not None


def test_three_p1_bare_words_reclaim_then_ack_on_the_existing_peer_raster(tmp_path):
    # This fixture derives the bare-word phase from a retained P1 turnaround.
    # It proves the P1 reclaim. It cannot assert a P3 clock was measured: that
    # needs independent packet/control references (test_p3_failed_turn_clock).
    protocol = spec.Protocol.PACTOR1
    g = _grid(protocol)
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path,
                      settle=0.04)
    host = ptc.PtcHost(peer=tx, mycall="W9SSJ")
    tx.host = host
    tx.live = bench = _Bench()
    host.protocol = protocol
    host.arq.role, host.arq.state = arq.IRS, arq.State.CONNECTED
    host.arq.mycall, host.arq.dxcall = "W9SSJ", "KB5LZK"
    host.arq._sl = arq.P1_SPEED_LEVEL if protocol == spec.Protocol.PACTOR1 else 1
    # The peer has already become IRS after a missed turn change. Its short
    # codewords fit the existing transmit comb; recovery must not add 840 ms
    # (P1) or 600 ms (P3) to them as if each were a data packet head.
    for slot in range(10, 13):
        g.cycles = slot
        at = g.boundary(slot) - g.cs_n - g.peer_read_gap
        name = "CS2/ack" if protocol == spec.Protocol.PACTOR1 else "REQ"
        g.note_peer_codeword(at, g.cs_n, name, "KB5LZK")
        g.peer_onset = at
        tx.aim(g, slot)
        bench.now = bench.pos = tx.key_instant(g, slot) - round(0.040 * onair.FS)
        host.arq.on_rx_cs(arq.CS_REQUEST)
        host.tick()
        onair._grid_reversal(g, host)
    assert host.arq.role == arq.ISS
    assert len(bench.emissions) == 3
    first, end = bench.emissions[-1]
    assert first == g.boundary(12)
    assert end <= at + g.cycle_n
    assert g.rx_due(12) == at + g.cycle_n
    assert host.arq._inflight.breakin
    # A decoded acknowledgement must advance this recovery packet, rather
    # than leave it in the repeated empty-break-in loop from the live arm.
    seq = host.arq._inflight.seq
    tx.aim(g, 13)
    bench.now = bench.pos = g.boundary(13) - round(0.040 * onair.FS)
    host.arq.on_rx_cs(arq.CS_ACK)
    assert host.arq._inflight is None or host.arq._inflight.seq != seq


def test_p3_bare_words_do_not_turn_retained_p1_turnaround_into_a_reply_clock(tmp_path):
    g = _grid(spec.Protocol.PACTOR3)
    g.protocol = spec.Protocol.PACTOR3
    g._p3_peer_confirmed = True
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path, settle=.04)
    host = ptc.PtcHost(peer=tx, mycall="W9SSJ")
    tx.host = host
    tx.live = bench = _Bench()
    host.protocol = spec.Protocol.PACTOR3
    host.arq.role, host.arq.state = arq.IRS, arq.State.CONNECTED
    host.arq.mycall, host.arq.dxcall, host.arq._sl = "W9SSJ", "KB5LZK", 1
    for slot in range(10, 13):
        g.cycles = slot
        # Deliberately preserve the old synthetic positioning; it does not
        # establish the independent P3 clock now required for either reply.
        at = g.boundary(slot) - g.cs_n - g.peer_read_gap
        g.note_peer_codeword(at, g.cs_n, "REQ", "KB5LZK",
                             protocol=spec.Protocol.PACTOR3)
        tx.aim(g, slot)
        bench.now = bench.pos = tx.key_instant(g, slot) - round(.040 * onair.FS)
        host.arq.on_rx_cs(arq.CS_REQUEST)
        host.tick()
        onair._grid_reversal(g, host)
    assert host.arq.role == arq.IRS
    assert bench.emissions == []
    assert host.arq._inflight is None
