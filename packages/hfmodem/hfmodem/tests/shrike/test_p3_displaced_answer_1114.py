# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The answer a refused changeover gives way to keys on its own comb.

Three long-cycle opening turns, each "CHANGEOVER NOT PLACED", the displaced
CS2 ACK keyed at the changeover's CS3 pulse reference (rot +31.8 ms against
the ordinary +14.1), then a packet/control pair from that ACK placing every
changeover 16-19 ms late, all unread; the arm's next pair put them at +104,
all read. Coordinates per arm: the ACK's tx_event audio start, the reply
clock's audio epoch for it, the emitted control phase and the peer packet
phase of the pair, and the forward gap the arm logged from it.
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_p3_breakin_d5 import (
    PAYLOAD, STATUS, _stale, changeover_lead, duplex)

FS = onair.FS
CYCLE, LONG, LONG_WIDTH, CS_N = 60000, 180000, 158160, 10080

ARMS = {
    #             ACK audio, epoch,   control,  packet,  logged forward ms
    "K4MSU 0522": (1472454, 1471692, 1473132, 1489236, 119.50),
    "WS8EOC 0542": (1412702, 1411752, 1413377, 1429296, 123.35),
    "WS8EOC 1553": (2192486, 2191754, 2193161, 2209298, 118.81),
}


def refused(tmp_path, late, mode):
    g = _stale()
    tx = duplex(g, tmp_path)
    tx.p3_displaced_answer = mode
    tx._flip = lambda: False
    tx.aim(g, 44)
    comb = tx.boundary
    # The arm's reply clock: its CS3 pulse reference `late` past the comb.
    clock = ReplyClock(entry_phase=1, pulse_epoch=comb + late - 10 * CYCLE)
    tx.reply_clock = g.reply_clock = clock
    tx.breakin_due = True
    tx.live.now = tx.live.pos = (tx._breakin_reference(g, 44) - changeover_lead()
                                 - 2000 - tx.live.key_notice)
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) == arq.REFUSED
    tx.send_cs(arq.CS_ACK)
    return tx.tx_audio_start - comb


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_displaced_ack_keys_on_its_comb(arm, tmp_path):
    ack, epoch, *_ = ARMS[arm]
    assert refused(tmp_path, ack - epoch, "comb") == 0


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_changeover_knob_keeps_the_arms_late_ack(arm, tmp_path):
    ack, epoch, *_ = ARMS[arm]
    assert refused(tmp_path, ack - epoch, "changeover") == ack - epoch


def read_gap(control, packet):
    g = onair._MasterGrid(0, CYCLE, 0, packet_n=38880, cs_n=CS_N, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.acquired = g.corroborated = True
    g.d_n, g.d_ref_n = 4800, 38880
    g.regear(True)
    g.note_p3_packet(packet - 2 * LONG, LONG_WIDTH, LONG)
    g.note_p3_control(control)
    g.note_p3_packet(packet, LONG_WIDTH, LONG)
    return g.peer_read_gap / FS * 1e3


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_pair_from_the_comb_puts_the_long_read_where_the_reads_land(arm):
    ack, epoch, control, packet, logged = ARMS[arm]
    assert abs(read_gap(control, packet) - logged) <= .05
    # The same pair with the ACK on its comb: +103.6 on all three, against the
    # +101.4 to +106.1 of the pairs whose changeovers were read.
    assert abs(read_gap(control - (ack - epoch), packet) - 103.6) <= .1


def test_the_displaced_answer_flag():
    assert onair.build_parser().parse_args([]).p3_displaced_answer == "comb"
    assert onair.build_parser().parse_args(
        ["--p3-displaced-answer", "changeover"]).p3_displaced_answer == "changeover"
    assert onair.RadioTx.p3_displaced_answer == "comb"
