# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The five risks the round-1/2 critic listed, each decided on this tree.

One file because they are one question asked five times -- what carries between
cycles, and for how long -- and because four of the five answers are "bounded,
and here is the bound". The fifth, the rotation the comb owes a turnaround, was
a real defect twice over: the critic found a latch that could not tell one
protocol's rotation from another's, and `captures/onair-0916-2253` then found
that the latch had no business counting rotations at all. See
`_MasterGrid.reverse`.
"""
from types import SimpleNamespace

import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_p3_reply_placement import (
    A5_PEER, CYCLE, FS, IDENTITY, PACKET_N, PULL_N, REPLY_N, a5_grid, driver,
    irs_grid)

P1_DATA_N, P1_CS_N, P1_D_N = 46080, 5760, 4413
P1_ROT = P1_DATA_N - P1_CS_N                      # 840 ms
P3_ROT = onair.P3_PACKET_N - onair.P3_CS_N        # 600 ms


def p1_grid(anchor=0):
    """A plain PACTOR-1 link, the protocol `_grid_reversal` shares with P3."""
    g = onair._MasterGrid(anchor, CYCLE, 8880, packet_n=P1_DATA_N,
                          cs_n=P1_CS_N, d_max_n=6240)
    g.d_n, g.d_ref_n = P1_D_N, P1_DATA_N
    g.keyed_slot = 0
    return g


# -- risk 1: the rotation a turnaround owes, and the one it does not --------

PEER_PACKET = 1_000_000                       # where the peer's stint keys
PACKET_MS = P1_DATA_N / FS * 1e3              # 960: the peer is transmitting
OWED_MS = (P1_DATA_N + P1_D_N) / FS * 1e3     # 1051.9: its packet, then d


def comb_phase_ms(g, peer_at):
    """Where our comb keys in the peer's cycle, ms from its packet start.

    The mirror of `rx_ref_n`: an ISS reads its answer `packet_n + d` past its
    own boundary, so the IRS answering it owes a control signal at that same
    instant on the ISS's comb -- `OWED_MS`, and anything under `PACKET_MS` is
    inside the packet it was told to listen to.
    """
    return (g.anchor - peer_at) % CYCLE / FS * 1e3


def test_a_pactor1_comb_rotates_on_every_turnaround_the_peer_accepts():
    """Both combs move at a changeover, alternately, and ours owes every one.

    THE PREMISE THIS TEST CARRIED WAS REFUTED FROM THE AIR. It asserted that
    one rotation covers any number of round trips -- that the latch "keeps the
    rotation ON the anchor through the ISS stint, deliberately, so the comb is
    already right when the role comes back". A station's transmit grid rotates
    by `rot` when IT becomes the IRS and stands when it becomes the ISS, so the
    two ends move alternately and the PEER's comb moves across our ISS stint. A
    comb that stays put comes back a whole rotation behind.

    `captures/onair-0916-2253`, WS8EOC, PACTOR-1: the peer's comb moved +848 ms
    between the rasters either side of our stint and ours moved 0 ms, so for
    the last 50 s every acknowledgement keyed 147 ms into the 960 ms packet it
    was answering -- 30 codewords at zero bit errors and not one packet.

    So the peer's raster is carried here and moved by the same rule as ours.
    """
    peer = PEER_PACKET
    g = p1_grid(anchor=peer + P1_DATA_N + P1_D_N - P1_ROT)
    g.reverse(to_iss=False)              # the peer takes the link: OUR comb moves
    assert comb_phase_ms(g, peer) == OWED_MS
    for _ in range(3):
        g.reverse(to_iss=True)           # we take it back: ITS comb moves
        peer += P1_ROT
        g.reverse(to_iss=False)          # ...and yield again, owing another
        assert comb_phase_ms(g, peer) >= PACKET_MS   # in its silence, not its packet
        assert comb_phase_ms(g, peer) == OWED_MS


def test_a_changeover_the_peer_never_accepted_moves_nothing():
    """What the rotation used to be latched against, and it is a real link.

    A changeover packet makes this station the ISS the moment it is built, so
    the role runs a cycle or more ahead of the peer's answer to it. When that
    answer never comes the peer carries on its old stint on its old raster
    (`arq._resume_peer_stint`) and the role returns with no turnaround having
    happened at either end. 0912-2349 rotated on the way back out anyway and
    keyed every ACK inside the packet it was answering.
    """
    peer = PEER_PACKET
    g = p1_grid(anchor=peer + P1_DATA_N + P1_D_N - P1_ROT)
    g.reverse(to_iss=False)
    g.turn_accepted = False              # our changeover packet, still unanswered
    g.reverse(to_iss=True)
    g.reverse(to_iss=False)              # its stint, on the raster it never left
    assert comb_phase_ms(g, peer) == OWED_MS
    # ...and the refusal is spent on the stint it belonged to, not the next one.
    g.reverse(to_iss=True)
    peer += P1_ROT
    g.reverse(to_iss=False)
    assert comb_phase_ms(g, peer) == OWED_MS


def host_at(role, *, unconfirmed_breakin=False):
    """Just enough of a linked session for `_grid_reversal` to read a role off."""
    return SimpleNamespace(
        protocol=spec.Protocol.PACTOR1,
        arq=SimpleNamespace(state=arq.State.CONNECTED, role=role,
                            entry_pending=False,
                            unconfirmed_breakin=unconfirmed_breakin))


def test_the_driver_carries_the_peers_agreement_into_the_reversal():
    """The fact lives on the ARQ and is gone by the reversal that needs it.

    `unconfirmed_breakin` is true only while we hold an ISS role the peer has
    not answered, and `_resume_peer_stint` clears it and puts the role back in
    one call -- so a reversal that read it then would only ever see a stint that
    had already ended. The grid takes it every cycle the role stands instead.
    """
    peer = PEER_PACKET
    g = p1_grid(anchor=peer + P1_DATA_N + P1_D_N - P1_ROT)
    onair._grid_reversal(g, host_at(arq.IRS))
    assert not g.sending and comb_phase_ms(g, peer) == OWED_MS
    # A changeover of ours: the role is ours a cycle before the peer answers it.
    onair._grid_reversal(g, host_at(arq.ISS, unconfirmed_breakin=True))
    assert g.sending and not g.turn_accepted
    onair._grid_reversal(g, host_at(arq.ISS))     # ...and then it answers.
    assert g.turn_accepted
    peer += P1_ROT
    onair._grid_reversal(g, host_at(arq.IRS))
    assert comb_phase_ms(g, peer) == OWED_MS


def test_the_reversal_reports_the_move_it_made_and_not_the_one_it_owed():
    """An operator watching the transcript can see a rotation that was withheld.

    The line printed `+840 ms` off `rot` on every reversal, including the ones
    the latch suppressed, so 0916-2253's transcript reported a comb that had
    moved while the comb stood still. There is nothing else in the log that the
    fault would have shown up in.
    """
    g = p1_grid()
    assert "transmit anchor +840 ms;" in g.reverse(to_iss=False)
    g.turn_accepted = False
    g.reverse(to_iss=True)
    assert "transmit anchor +0 ms -- the changeover we keyed was never answered" \
        in g.reverse(to_iss=False)


def test_the_rotation_is_the_protocol_the_reversal_is_keyed_in():
    """A link that changes protocol between two turnarounds, and it is a real one.

    `rot` is `data_n - cs_n` and both terms are the protocol's -- 840 ms in
    PACTOR-1, 600 in PACTOR-3 -- so a station that was a PACTOR-1 IRS, yielded,
    upgraded and took the link back owes PACTOR-3's 600 on the next yield. Each
    turnaround moves the comb by the size of the protocol it is keyed in, and
    `_grid_reversal` restates `protocol` from the accepted changeover before it
    asks for the move.
    """
    g = p1_grid()
    base = g.anchor
    g.reverse(to_iss=False)
    assert g.anchor - base == P1_ROT
    g.reverse(to_iss=True)
    g.keying(spec.Protocol.PACTOR3)
    assert g.data_n - g.cs_n == P3_ROT
    g.reverse(to_iss=False)
    assert g.anchor - base == P1_ROT + P3_ROT
    # ...and back again, for the fallback that follows an entry budget running out.
    g.reverse(to_iss=True)
    g.keying(spec.Protocol.PACTOR1)
    g.reverse(to_iss=False)
    assert g.anchor - base == 2 * P1_ROT + P3_ROT


def test_a_recovered_turn_comes_back_to_the_comb_it_left():
    """`recover_p3_turn` restores a comb that is a PACTOR-3 IRS comb."""
    g = irs_grid()
    placed = g.anchor
    for k in (0, 1):
        g.note_p3_packet(A5_PEER + k * CYCLE, PACKET_N, CYCLE, swapped=False,
                         identity=IDENTITY)
        g.note_p3_control(A5_PEER + k * CYCLE + REPLY_N)
    g.remember_p3_turn(A5_PEER + CYCLE + REPLY_N)
    assert g._p3_turn is not None
    g.note_p3_packet(A5_PEER + 2 * CYCLE, PACKET_N, CYCLE, swapped=False,
                     identity=IDENTITY)
    g.turn_accepted = False
    g.reverse(to_iss=True)
    assert g.recover_p3_turn(A5_PEER + 2 * CYCLE, IDENTITY)
    assert g.anchor == placed and not g.sending and g.turn_accepted


# -- risk 2: `_p3_raster_run` across a role round trip ----------------------

def test_a_corroborated_raster_survives_a_role_round_trip_and_still_expires():
    """Reset only on a protocol change, and that is the right subject.

    The raster is the PEER's cycle grid. Our role does not move it, so a
    corroboration earned before an ISS stint is still a corroboration after one.
    What could be minutes old is the FRESHNESS, and that is not what the run
    carries: `p3_control_refusal` ages `_p3_heard_at` on every cycle either way,
    and the run only chooses which of the two bounds it ages against.
    """
    g = a5_grid()
    assert g._p3_raster_corroborated
    g.reverse(to_iss=True)
    g.reverse(to_iss=False)
    assert g._p3_raster_corroborated
    slot = round((A5_PEER + 5 * CYCLE + REPLY_N - g.anchor) / CYCLE)
    assert g.p3_control_refusal(slot + onair.ONSET_MAX_CYCLES) is None
    assert "no fresh packet clock" in g.p3_control_refusal(
        slot + onair.RASTER_PROJECT_CYCLES)
    # ...and a packet off the projection is a new raster, not a longer run.
    g.note_p3_packet(A5_PEER + 6 * CYCLE + 4 * PULL_N, PACKET_N, CYCLE,
                     identity=IDENTITY)
    assert g._p3_raster_run == 1 and not g._p3_raster_corroborated


def test_a_protocol_change_is_what_ends_the_run():
    g = a5_grid()
    assert g._p3_raster_corroborated
    g.keying(spec.Protocol.PACTOR1)
    assert g._p3_raster_run == 0 and g._p3_raster_origin is None


# -- risk 3: `note_peer_bursts` walking while the comb is held --------------

def test_an_undecoded_burst_cannot_walk_the_peers_raster_off_the_last_decode(
        tmp_path):
    """The burst reading is measured against the DECODE and never against itself.

    `note_peer_bursts` writes `peer_onset` and `_p3_heard_at`; on a PACTOR-3 link
    whose packet clock is confirmed, `peer_at` is `_p3_peer[0]`, so the window
    each burst is admitted into is +/-`MAX_PULL_S` of the last CRC packet's own
    projection and stays there. A peer walking 20 ms a cycle is therefore read
    once and refused from the second cycle on -- it cannot drag the origin along
    with it, and the reply comb, which answers `_p3_peer[0]` and not the burst,
    does not move at all. What a peer that has walked out of the window loses is
    the evidence it is still transmitting, and that ends the reply on
    `p3_control_refusal`'s own bound rather than on a collision.
    """
    g = a5_grid()
    tx = driver(g, tmp_path)
    last = A5_PEER + 5 * CYCLE
    first_slot = round((A5_PEER + REPLY_N - g.anchor) / CYCLE)
    took, placed = [], []
    for k in range(6, 15):
        at = A5_PEER + k * CYCLE + (k - 5) * PULL_N     # 20 ms a cycle
        took.append(g.note_peer_bursts([(at, PACKET_N)]) is not None)
        onair._p3_place_reply(g, tx, first_slot + k)
        placed.append(g.boundary(first_slot + k) - (A5_PEER + k * CYCLE))
        g.cycles += 1
    assert took == [True] + [False] * 8
    assert placed == [REPLY_N] * 9                  # the comb never followed
    assert g._p3_peer[0] == last and g._p3_raster_origin == A5_PEER
    assert g._p3_heard_at == A5_PEER + 6 * CYCLE + PULL_N
    # ...and with nothing refreshing it, the corroborated bound is what ends it.
    assert g.p3_control_refusal(first_slot + 6 + onair.RASTER_PROJECT_CYCLES) \
        is not None


# -- risk 4: `P3_LONG_REPLY_S` is inert, and right by construction ----------

def test_the_long_answer_slot_is_two_slots_on_and_needs_no_term_of_its_own():
    """3.390 - 0.890 is 2.500 s, which is the comb's own period twice over.

    So the long branch of `p3_reply_shift` reduces to the short one: same error,
    same move, same anchor, and one comb carrying both answers two slots apart.
    The constant states where the long answer sits rather than moving anything
    to it, and what actually keys it there is `regear` -- `ticks` becomes three
    and `_regear_next_slot` steps the comb by that.
    """
    assert round((onair.P3_LONG_REPLY_S - onair.P3_REPLY_S) * FS) == 2 * CYCLE
    moved = []
    for long in (False, True):
        g = irs_grid()
        g.regear(long)
        width = onair.P3_LONG_PACKET_N if long else PACKET_N
        g.note_p3_packet(A5_PEER, width, g.cycle_n, swapped=False,
                         identity=IDENTITY)
        slot = round((A5_PEER + REPLY_N - g.anchor) / g.slot_n)
        g.anchor += A5_PEER + REPLY_N - g.boundary(slot) + 1000
        before = g.anchor
        assert g.p3_reply_shift(slot) is not None
        moved.append((g.anchor - before, g.boundary(slot) - A5_PEER,
                      g.boundary(slot + 2) - A5_PEER))
    assert moved[0] == moved[1]
    assert moved[0] == (-1000, REPLY_N, round(onair.P3_LONG_REPLY_S * FS))


# -- risk 5: the QRT, connect and ISS paths ---------------------------------

@pytest.fixture
def sending_grid():
    g = irs_grid()
    g.note_p3_packet(A5_PEER, PACKET_N, CYCLE, swapped=False, identity=IDENTITY)
    g.reverse(to_iss=True)
    return g


def test_nothing_in_the_reply_clock_reaches_a_sending_grid(sending_grid, tmp_path):
    g = sending_grid
    slot = round((A5_PEER + REPLY_N - g.anchor) / CYCLE)
    tx = driver(g, tmp_path)
    before = g.anchor
    assert g.p3_reply_shift(slot) is None
    assert g.p3_control_refusal(slot) is None
    onair._p3_place_reply(g, tx, slot)
    assert g.anchor == before
    g.note_p3_control(A5_PEER + REPLY_N)
    assert not g._p3_controls


def test_nothing_in_the_reply_clock_reaches_pactor1_or_pactor2(tmp_path):
    for protocol in (spec.Protocol.PACTOR1, spec.Protocol.PACTOR2):
        g = irs_grid()
        g.note_p3_packet(A5_PEER, PACKET_N, CYCLE, swapped=False,
                         identity=IDENTITY)
        slot = round((A5_PEER + REPLY_N - g.anchor) / CYCLE)
        g.anchor += 1000
        before = g.anchor
        g.keying(protocol)
        tx = driver(g, tmp_path)
        assert g.p3_reply_shift(slot) is None
        assert g.p3_control_refusal(slot) is None
        onair._p3_place_reply(g, tx, slot)
        assert g.anchor == before


def test_a_queued_codeword_is_not_emitted_off_the_pactor3_irs_path(tmp_path):
    g = irs_grid()
    tx = driver(g, tmp_path)
    host = _Host(g)
    tx.attach(host)
    tx.live.now = tx.live.pos = A5_PEER
    for protocol, role in ((spec.Protocol.PACTOR1, arq.IRS),
                           (spec.Protocol.PACTOR3, arq.ISS)):
        host.protocol, host.arq.role = protocol, role
        tx._pending_p3_cs = arq.CS_ACK
        tx.emit_pending_cs()
        assert tx.n == 0 and tx._pending_p3_cs is None


class _Host:
    """The two fields `emit_pending_cs` asks its host for, and nothing else."""

    def __init__(self, raster):
        self.protocol = spec.Protocol.PACTOR3
        self.arq = type("Arq", (), {"role": arq.ISS})()
        self.raster = raster

    def on_cs_emitted(self, cs):
        raise AssertionError("emitted off the PACTOR-3 IRS path")
