# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A turn we break into rides the changeover's head, not the comb it left.

WS8EOC 2026-11-14 04:36Z (arm 13cdc925), three changeovers keyed at the read
and each read at once. Turn 3's head sat 9.4 ms behind the reply clock our IRS
answers were keyed on; the data stayed on that clock, 1240.6 ms after the
changeover, and the gateway answered CS1 to it 38 times to max retries. Turns
1 and 2 sat 3.7 and 3.6 ms in front (1253.7 and 1253.6 ms). Every coordinate
below is the arm log's: the ACK keyed the cycle before each changeover (its
pulse and audio start), the peer packet the changeover was placed behind, and
the tx_event audio starts of the changeover and the first data packet.
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.p3_reply_support import duplex

FS = onair.FS
CYCLE, WIDTH, CS_N = 60000, 39120, 10080
# The changeover's own lead and SL1's are both 666 samples, and the log agrees:
# each changeover's audio start plus 666 is its logged read past the packet.
LEAD = 666
# Turn 3's CS1 decoded at 7935966, 45606 samples past the head it answered.
ANSWER = 45606
HALF_MS = round(.0005 * FS)

TURNS = {
    #    slot, ACK pulse, ACK audio, peer onset, label, changeover, data1
    1: (33, 1952534, 1951857, 1969266, " +86 ms", 2011866, 2072043),
    2: (70, 4201383, 4200708, 4218260, " +83 ms", 4260717, 4320892),
    3: (130, 7830360, 7829685, 7847070, " +87 ms", 7889694, 7949242),
}


def scene(tmp_path, turn, phase):
    slot, control, ack, peer, _, _, data1 = TURNS[turn]
    boundary = ack + CYCLE
    g = onair._MasterGrid(boundary - slot * CYCLE, CYCLE, 0, packet_n=38880,
                          cs_n=CS_N, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.acquired = g.corroborated = True
    g.d_n, g.d_ref_n = 4800, 38880
    # The clock the data rode on the arm: its first packet's pulse, a cycle back.
    clock = ReplyClock(entry_phase=1, pulse_epoch=data1 + LEAD - 2 * CYCLE)
    g.reply_clock = clock
    g.note_p3_packet(peer - CYCLE, WIDTH, CYCLE)
    g.note_p3_control(control)
    g.note_p3_packet(peer, WIDTH, CYCLE)
    tx = duplex(g, tmp_path)
    tx.reply_clock = clock
    tx.p3_turn_phase = phase
    tx.live.now = tx.live.pos = peer + 30000
    return tx, g, slot


def pulse(tx):
    return tx.tx_audio_start + min(tx.tx_pulse_offsets)


def changeover(tx, g, slot):
    tx.aim(g, slot)
    tx.breakin_due = True
    tx.send_packet(1, b"FS ", 0, breakin=True)
    assert not tx.refused and len(tx.live.emissions) == 1
    head = pulse(tx)
    g.reverse(to_iss=True)
    tx.host.arq.role = arq.ISS
    g.note_peer_codeword(head + ANSWER, CS_N, "ACK", "WS8EOC",
                         protocol=spec.Protocol.PACTOR3)
    return head


def data(tx, g, slot):
    tx.aim(g, slot)
    tx.send_packet(1, b"+\r", 0x01)
    assert not tx.refused
    return tx.tx_audio_start, pulse(tx)


@pytest.mark.parametrize("turn", sorted(TURNS))
def test_the_turn_rides_the_changeover_head(turn, tmp_path, capsys):
    tx, g, slot = scene(tmp_path, turn, "head")
    head = changeover(tx, g, slot)
    said = capsys.readouterr().out
    assert f"P3 BREAK-IN 3B -> SL1{TURNS[turn][4]} past the peer's packet end" in said
    start = tx.tx_audio_start
    assert abs(start - TURNS[turn][5]) <= HALF_MS  # the arm's own key
    data1, at1 = data(tx, g, slot + 1)
    assert "P3 TURN ON THE CHANGEOVER HEAD" in capsys.readouterr().out
    assert at1 - head == CYCLE
    assert abs((data1 - start) / FS * 1e3 - 1250.0) <= 0.5
    # It is the turn's phase, not one packet's: the retry rides it too.
    _, at2 = data(tx, g, slot + 2)
    assert at2 - head == 2 * CYCLE
    assert g._p3_turn_head is None


@pytest.mark.parametrize("turn", sorted(TURNS))
def test_the_held_knob_keeps_the_arms_offsets(turn, tmp_path):
    tx, g, slot = scene(tmp_path, turn, "held")
    changeover(tx, g, slot)
    start = tx.tx_audio_start
    data1, _ = data(tx, g, slot + 1)
    recorded = TURNS[turn][6] - TURNS[turn][5]
    assert abs((data1 - start) - recorded) <= HALF_MS
    assert g._p3_turn_head is None


def test_an_iss_turn_we_did_not_break_into_is_unchanged(tmp_path):
    placed = {}
    for phase in ("head", "held"):
        (tmp_path / phase).mkdir()
        tx, g, slot = scene(tmp_path / phase, 3, phase)
        g.reverse(to_iss=True)
        tx.host.arq.role = arq.ISS
        g.note_peer_codeword(7935966, CS_N, "ACK", "WS8EOC",
                             protocol=spec.Protocol.PACTOR3)
        anchor, epoch = g.anchor, g.reply_clock.pulse_epoch
        placed[phase] = data(tx, g, slot + 1)
        assert (g.anchor, g.reply_clock.pulse_epoch) == (anchor, epoch)
        assert (placed[phase][1] - epoch) % CYCLE == 0
    assert placed["head"] == placed["held"]


def test_an_unanswered_changeover_moves_nothing(tmp_path):
    tx, g, slot = scene(tmp_path, 3, "head")
    anchor, epoch = g.anchor, g.reply_clock.pulse_epoch
    changeover(tx, g, slot)
    *_, start, data1 = TURNS[3]
    assert abs(g._p3_turn_head - (start - (data1 - CYCLE))) <= HALF_MS  # +452
    # The peer carries on with its stint: the role comes back unturned.
    g.turn_accepted = False
    tx.host.arq.role = arq.IRS
    g.reverse(to_iss=False)
    assert g._p3_turn_head is None
    assert (g.anchor, g.reply_clock.pulse_epoch) == (anchor, epoch)


def test_the_turn_phase_flag():
    assert onair.build_parser().parse_args([]).p3_turn_phase == "head"
    assert onair.build_parser().parse_args(
        ["--p3-turn-phase", "held"]).p3_turn_phase == "held"
    assert onair.RadioTx.p3_turn_phase == "head"


# K4MSU 2026-11-14 05:22Z (arm 5b518ce9), turn 2, broken into a LONG cycle:
# CS1 seq=2 (TX24, slot 51) pulse 3092368 audio 3091692; the peer's long packet
# at 3289220, +3295 ms; the answered head TX26 (slot 60) at audio 3631702, CS1
# decoded at 3678083; data1 TX27 (slot 63) at audio 3811980, 3755.79 ms on, and
# every one of its 40 copies unread.
K4MSU = dict(ack=3092368, boundary=3451692 - 57 * CYCLE, peer=3289220,
             head=3631702, answer=3678083, data1=3811980)
LONG, LONG_WIDTH = 3 * CYCLE, 158160


@pytest.mark.parametrize("phase", ["head", "held"])
def test_a_long_cycle_turn_rides_its_head_too(phase, tmp_path, capsys):
    k = K4MSU
    g = onair._MasterGrid(k["boundary"], CYCLE, 0, packet_n=38880, cs_n=CS_N,
                          d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.acquired = g.corroborated = True
    g.d_n, g.d_ref_n = 4800, 38880
    g.regear(True)
    clock = ReplyClock(entry_phase=1, pulse_epoch=k["data1"] + LEAD - 20 * CYCLE)
    g.reply_clock = clock
    g.note_p3_packet(k["peer"] - LONG, LONG_WIDTH, LONG)
    g.note_p3_control(k["ack"])
    g.note_p3_packet(k["peer"], LONG_WIDTH, LONG)
    tx = duplex(g, tmp_path)
    tx.reply_clock = clock
    tx.p3_turn_phase = phase
    tx.live.now = tx.live.pos = k["peer"] + LONG + 140000
    tx.aim(g, 60)
    tx.breakin_due = True
    tx.send_packet(1, b";FW", 0, breakin=True)
    assert " +104 ms past the peer's packet end" in capsys.readouterr().out
    assert not tx.refused and tx.tx_audio_start == k["head"]
    head, start = pulse(tx), tx.tx_audio_start
    g.reverse(to_iss=True)
    tx.host.arq.role = arq.ISS
    tx.host.arq.cycle_long = False
    g.regear(False)
    g.note_peer_codeword(k["answer"], CS_N, "ACK", "K4MSU",
                         protocol=spec.Protocol.PACTOR3)
    data1, at1 = data(tx, g, 63)
    if phase == "held":
        assert abs(data1 - k["data1"]) <= HALF_MS  # the arm's +5.8 ms
        assert abs((data1 - start) / FS * 1e3 - 3755.79) <= 0.1
    else:
        assert at1 - head == 3 * CYCLE
        assert abs((data1 - start) / FS * 1e3 - 3750.0) <= 0.5
