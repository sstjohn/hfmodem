# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The IRS reply comb is placed from the peer's packet, every cycle it keys.

Coordinates are transcript readings from two recorded arms -- `onair-0912-2349`
(B3, profile B, 80 m WS8EOC) and `onair-0912-2321` (A5, profile A) -- driven
through the production grid and transmitter seams. No audio fixture: what these
scenes exercise is placement, and the arms' own frame phases are the input.
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

FS = onair.FS
CYCLE = 60000
PACKET_N = 38880
REPLY_N = round(onair.P3_REPLY_S * FS)
PULL_N = round(onair.MAX_PULL_S * FS)
IDENTITY = (True, 1, 0, b"RMS")


def irs_grid(anchor=2633):
    """The comb an entry-pending P3 caller holds when the peer takes the link."""
    g = onair._MasterGrid(anchor, CYCLE, 8880, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.d_n, g.d_ref_n = 4413, 46080
    g.keyed_slot = 29
    g.keying(spec.Protocol.PACTOR3, entry_pending=True, entry_variant="template")
    g.reverse(to_iss=False)
    return g


def driver(g, tmp_path):
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path,
                       settle=.04)
    tx.live = _Bench(seconds=120)
    tx.raster = g
    return tx


def key_ack(tx, g, slot):
    """Key one CS1 on `slot` through the real placement and admission path."""
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - round(.04 * FS)
    return tx._send_p3_control(arq.CS_ACK)


def test_b3_role_round_trip_answers_the_returning_stint_on_the_answer_slot(tmp_path):
    """B3 :213-288 -- the second reversal put every ACK inside the peer's packet."""
    g = irs_grid()
    assert g.anchor == 31433  # TX[27]'s logged comb, 2633 plus the 600 ms.
    tx = driver(g, tmp_path)
    g.note_p3_packet(1788773, PACKET_N, CYCLE, swapped=False, identity=IDENTITY)
    assert key_ack(tx, g, 31) is None
    assert g._p3_controls[-1][0] - 1788773 == CYCLE + REPLY_N
    placed = g.anchor

    g.turn_accepted = False  # TX[30], the changeover the peer never answered.
    g.reverse(to_iss=True)
    g.reverse(to_iss=False)  # ...and its stint comes back ten periods on.
    assert g.anchor == placed  # No turnaround, no move: not the 1200 ms B3 keyed on.
    g.note_p3_packet(2568924, PACKET_N, CYCLE, swapped=False, identity=IDENTITY)
    assert not g._p3_reply_phase_invalid
    assert key_ack(tx, g, 43) is None
    assert g._p3_controls[-1][0] - 2568924 == pytest.approx(REPLY_N, abs=PULL_N)
    carrier = tx.tx_key_up
    assert g.key_refusal(carrier, tx.tx_end - carrier) is None
    assert tx.n == 2 and not tx.refused


A5_PEER = 1_500_000
A5_SLOT = 30  # The comb's slot for the peer's k = 0; one slot to the cycle.


def a5_grid():
    """A5 :peer raster, three decodes, then the nine-cycle gap it advanced across."""
    g = irs_grid(anchor=A5_PEER + REPLY_N - A5_SLOT * CYCLE - 28800)
    for k in (0, 2, 5):
        g.note_p3_packet(A5_PEER + k * CYCLE, PACKET_N, CYCLE, identity=IDENTITY)
        g.cycles += 1
    return g


def test_raster_aligned_bursts_keep_the_reply_keying_through_a_decode_gap(tmp_path):
    """A5 :the peer kept a full packet in every cycle; seven of 38 decoded."""
    g = a5_grid()
    tx = driver(g, tmp_path)
    assert g._p3_raster_corroborated
    for k in range(6, 15):
        at = A5_PEER + k * CYCLE
        assert g.note_peer_bursts([(at, PACKET_N)]) is not None
        assert g.p3_control_refusal(A5_SLOT + k) is None
        assert key_ack(tx, g, A5_SLOT + k) is None
        g.cycles += 1
    assert tx.n == 9
    # ...and a burst off the peer's raster is not the peer still transmitting.
    assert g.note_peer_bursts([(A5_PEER + 15 * CYCLE + 3 * PULL_N, PACKET_N)]) is None


def test_a_silent_channel_still_gives_the_reply_up_at_eight_cycles():
    g = irs_grid()
    g.note_p3_packet(A5_PEER, PACKET_N, CYCLE, identity=IDENTITY)
    slot = (A5_PEER + REPLY_N - g.anchor) // CYCLE
    assert g.p3_control_refusal(slot + onair.ONSET_MAX_CYCLES - 1) is None
    assert "no fresh packet clock" in g.p3_control_refusal(
        slot + onair.ONSET_MAX_CYCLES)


def test_a_corroborated_raster_outlives_one_reading_of_it():
    g = a5_grid()
    slot = A5_SLOT + 5
    assert g._p3_raster_corroborated
    assert g.p3_control_refusal(slot + onair.ONSET_MAX_CYCLES) is None
    assert "no fresh packet clock" in g.p3_control_refusal(
        slot + onair.RASTER_PROJECT_CYCLES)


def late_comb(error_n):
    """An IRS holding one CRC packet, with its comb `error_n` off the answer slot."""
    g = irs_grid()
    g.note_p3_packet(A5_PEER, PACKET_N, CYCLE, swapped=False, identity=IDENTITY)
    due = A5_PEER + REPLY_N
    slot = round((due - g.anchor) / CYCLE)
    g.anchor += due - g.boundary(slot) + error_n
    assert g.boundary(slot) - due == error_n
    return g, slot, due


def settle_and_key(tx, g, slot):
    """The loop's own order: place the comb, size the cycle, then key.

    `_p3_place_reply` where `_regrid` and the hold loop call it, `_keyable_slot`
    on the comb that came out of it, and the clock put where the cycle's last
    read leaves it -- one settle in front of the key it is aiming at.
    """
    tx.aim(g, slot)
    onair._p3_place_reply(g, tx, slot)
    slot = onair._keyable_slot(tx.live, g, slot, round(.04 * FS))
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - round(.04 * FS)
    return slot, tx._send_p3_control(arq.CS_ACK)


@pytest.mark.parametrize("move_ms", [-600., -400., -200., 1.25])
def test_a_comb_off_the_answer_slot_still_keys_in_the_packets_own_slot(
        move_ms, tmp_path):
    """A backward re-place is half the residue space and it was never driven.

    Placed at the emit it cannot be made: the window has been sized and the
    clock is standing one settle in front of the OLD boundary, so anything
    further back than `settle + key_notice` -- about 48 ms -- names an instant
    already gone. Placed where the grid settles for the cycle it costs nothing,
    and the codeword lands in the answered packet's own slot at either sign.
    """
    error_n = -round(move_ms / 1e3 * FS)
    g, slot, due = late_comb(error_n)
    tx = driver(g, tmp_path)
    tx.live.now = tx.live.pos = due - round(.3 * FS)
    keyed, refused = settle_and_key(tx, g, slot)
    assert refused is None and tx.n == 1 and not tx.refused
    assert keyed == slot                      # the packet's own slot, not the next
    assert g.boundary(slot) == due
    assert g._p3_controls[-1][0] - A5_PEER == REPLY_N
    assert tx.tx_audio_start == due


@pytest.mark.parametrize("move_ms", [-600., -400., -200.])
def test_the_same_backward_re_place_at_the_emit_loses_the_slot(
        move_ms, tmp_path, capsys):
    """The negative control: the order this file exists to have changed."""
    error_n = -round(move_ms / 1e3 * FS)
    g, slot, due = late_comb(error_n)
    tx = driver(g, tmp_path)
    # The clock where a cycle that sized its window against the OLD comb leaves
    # it: one settle in front of the boundary the shift is about to abandon.
    assert key_ack(tx, g, slot) is None
    said = capsys.readouterr().out
    assert "would miss boundary" in said
    assert tx.slot > slot and tx.tx_audio_start >= due + CYCLE
