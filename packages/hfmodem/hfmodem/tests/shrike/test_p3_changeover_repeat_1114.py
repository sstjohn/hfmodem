# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A changeover the peer keeps asking for again repeats on its own head.

WS8EOC 2026-11-14 05:42Z (arm d1fcd58a), turn 4, the FF. TX78 keyed at the
read, +88 ms past the gateway's short packet, on the comb our IRS answers rode
(TX77 + 1250.2 ms). The gateway answered CS2 REQ at 0 bit errors from the next
cycle on, 34 times, consistent with a peer that read the CS3 head, took the
receiving role and never read the body. TX79 to TX115 keyed on the held grid, every one 15.1 ms
in front of TX78's raster (TX78 -> TX79 1234.9 ms). Every coordinate is the arm
log's: TX77's pulse and audio start, the packet onset TX78 was placed behind,
the tx_event audio starts of TX78 and TX79, and the REQ's decoded sample.
"""

from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.onair_support import _Session
from hfmodem.tests.shrike.p3_reply_support import duplex

FS = onair.FS
CYCLE, WIDTH, CS_N, LEAD = 60000, 39120, 10080, 666
HALF_MS = round(.0005 * FS)

SLOT = 166
ACK_PULSE, ACK_AUDIO = 10018831, 10018156      # TX77 CS2 ACK seq=3
PEER = 10035507                                # the +815 ms packet onset
HEAD = 10078165                                # TX78, the first copy
HELD = 10137439                                # TX79 on the held grid
REQ = 10124314                                 # the REQ TX78 drew


def scene(tmp_path, phase):
    g = onair._MasterGrid(ACK_AUDIO + CYCLE - SLOT * CYCLE, CYCLE, 0,
                          packet_n=38880, cs_n=CS_N, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.acquired = g.corroborated = True
    g.d_n, g.d_ref_n = 4800, 38880
    g.cycles = 70
    # The reply clock the arm's held grid keyed TX79 on.
    clock = ReplyClock(entry_phase=1, pulse_epoch=HELD + LEAD - 2 * CYCLE)
    g.reply_clock = clock
    g.note_p3_packet(PEER - CYCLE, WIDTH, CYCLE)
    g.note_p3_control(ACK_PULSE)
    g.note_p3_packet(PEER, WIDTH, CYCLE)
    sess = _Session(role=arq.ISS)
    tx = duplex(g, tmp_path, seconds=260.0)
    tx.reply_clock = clock
    tx.p3_turn_phase = phase
    sess.host.peer = tx
    tx.attach(sess.host)
    packet = arq._Packet(status=0, payload=b"FF\r", sl=1, breakin=True)
    sess.host.arq._inflight = packet
    tx.live.now = tx.live.pos = PEER + 30000
    tx.aim(g, SLOT)
    tx.breakin_due = True
    tx.send_packet(1, b"FF\r", 0, breakin=True)
    assert not tx.refused and abs(tx.tx_audio_start - HEAD) <= HALF_MS
    head = tx.tx_audio_start + min(tx.tx_pulse_offsets)
    g.reverse(to_iss=True)
    g.turn_accepted = False
    return sess, tx, g, packet, head


def repeat(tx, g, n):
    g.cycles += 1
    g.note_peer_codeword(REQ + (n - 1) * CYCLE, CS_N, "REQ", "WS8EOC",
                         protocol=spec.Protocol.PACTOR3)
    tx.live.now = tx.live.pos = REQ + (n - 1) * CYCLE + CS_N
    tx.aim(g, SLOT + n)
    tx.breakin_due = True
    tx.send_packet(1, b"FF\r", 0, breakin=True)
    assert not tx.refused
    return tx.tx_audio_start, tx.tx_audio_start + min(tx.tx_pulse_offsets)


def test_repeats_ride_the_first_copys_head(tmp_path, capsys):
    sess, tx, g, packet, head = scene(tmp_path, "head")
    capsys.readouterr()
    for n in range(1, 4):
        start, pulse = repeat(tx, g, n)
        said = capsys.readouterr().out
        assert "ON THE HELD GRID" not in said
        assert "on the emitted CS3 clock, corroborated by the peer's repeat" in said
        assert pulse - head == n * CYCLE
        assert abs((start - HEAD) / FS * 1e3 - 1250.0 * n) <= .5
    assert sess.host.arq._inflight is packet
    # The turn's first packet still seats the head the repeats rode.
    assert abs(g._p3_turn_head - round(.0151 * FS)) <= HALF_MS


def test_the_held_knob_keeps_the_arms_held_grid(tmp_path, capsys):
    sess, tx, g, packet, head = scene(tmp_path, "held")
    capsys.readouterr()
    start, _ = repeat(tx, g, 1)
    assert "on the held grid" in capsys.readouterr().out
    assert abs(start - HELD) <= HALF_MS
    assert abs((start - HEAD) / FS * 1e3 - 1234.9) <= .1

