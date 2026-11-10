# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The IRS reply cadence follows the peer's packets, never our own CS6.

WS8EOC, 2026-09-13 12:57 CDT on 40 m (`working/pactor3-header-0913/
arm-v8-A-40-ws8eoc.launch.log`). The greeting had arrived, the peer was at SL2
with status bit 5 up, and `--long-cycle` let the IRS grant it. From TX[75] on we
keyed CS6 nineteen times and read nineteen CRC-valid frames, every one of them
SHORT and every one on an unmoved 1.25 s raster -- 4553027 through 10973001, two
residues mod 60000, 26 samples apart. The log carries no `cycle length ->` line:
neither end ever went long. Our grid still keyed slot 80, 83, 86, 89 and gave up.

Nothing the peer did moved our comb. Our own command slot did, through a mask
that permitted only reply boundaries three slots from it -- 43 of the stint's 71
overruns took three slots instead of one, four of them off an overrun under
5 ms. These cases hold the cadence to the peer's packets through the real
`_MasterGrid`, `RadioTx` and `PactorArq`.
"""
import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_cs6_driver import (
    FIXTURES, _arm, _window, recorded, requires_recorded)
from hfmodem.tests.shrike.test_cs6_rx import ROWS
from hfmodem.tests.shrike.test_cs6_stalls import _duplex

FS = 48000
SETTLE_N = 1920
SHORT_N = round(spec.CYCLE_SHORT_S * FS)


def _raster_tape(seconds: int) -> np.ndarray:
    """The peer keying one short SL1 packet every 1.25 s, as WS8EOC did."""
    pcm = recorded(ROWS[0])
    tape = np.zeros(seconds * FS)
    for start in range(0, len(tape) - len(pcm), SHORT_N):
        tape[start:start + len(pcm)] = pcm
    return tape


def _buffered_listen(live, n, host, sessrx, limit, **kwargs):
    return live.take_until(live.pos + max(0, n))


def _cycle(clock, g, tx, s, slot):
    """One hold-loop cycle as `onair`'s own held-link loop spells it."""
    host = s.host
    tx.defer_p3_cs = True
    probing = (host.protocol == spec.Protocol.PACTOR3
               and host.arq.role == arq.IRS and host.arq.cycle_command_emitted)
    slot, _ = onair._regear_next_slot(
        g, slot, False if probing else host.arq.cycle_long)
    tx.aim(g, slot)
    onair._p3_place_reply(g, tx, slot)
    early = g.boundary(slot) - tx.key_instant(g, slot)
    slot = onair._keyable_slot(clock, g, slot, SETTLE_N + early)
    tx.aim(g, slot)
    key_at = tx.key_instant(g, slot)
    until = onair._p3_frame_ready(
        s.rx, onair._p3_decode_deadline(clock, key_at, SETTLE_N))
    audio, origin, _ = onair._collect(
        clock, s.rx, np.zeros(0, np.float32), clock.pos, until)
    if audio.size:
        onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    if probing:
        slot, audio, origin = onair._p3_transition_window(
            clock, g, tx, s.host, s.rx, slot, audio, origin, SETTLE_N)
    slot, audio, origin = onair._regrid(
        clock, g, tx, s.host, s.rx, slot, audio, origin, SETTLE_N)
    tx.emit_pending_cs()
    host.arq.on_cycle()
    return g.next_slot(slot)


@requires_recorded
def test_a_short_keying_peer_is_answered_every_cycle_after_cs6(
        tmp_path, monkeypatch):
    """Twelve peer packets, twelve answers, twelve consecutive slots.

    The arm's own channel: the peer is readable on every 1.25 s boundary and its
    status keeps bit 5 up. Only the first answer is CS6; valid short retries
    get their ordinary ACK without another cycle negotiation. The grid must
    preserve every reply slot through that transition.
    """
    tape = _raster_tape(30)
    s, tx, g, _ = _arm(tmp_path, tape, sl=1, expected_seq=1)
    clock = _duplex(tx, s, tape, 1536)
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    slot = 1
    for _ in range(12):
        slot = _cycle(clock, g, tx, s, slot)
    assert tx.slots_used == list(range(1, 13))
    assert len(s.packets) == 12
    assert [ev.cycle_long for ev in s.packets] == [False] * 12
    # Every packet gets its own reply slot, including the ordinary ACKs after
    # the cycle request. The surrounding slots are not the price of asking.
    assert [slot for slot, _ in tx.keyed] == list(range(1, 13))
    assert len(clock.emissions) == 12
    # ...and the comb never moved on the strength of any of them.
    assert not s.host.arq.cycle_long and g.ticks == 1
    assert s.host.arq.cycle_request is None
    assert not s.host.arq.cycle_command_emitted


@requires_recorded
def test_an_overrun_hands_back_one_slot_where_the_old_mask_took_three(
        tmp_path, monkeypatch):
    """The arm's `SLOT 80 IS GONE ... Keying on slot 83 instead`, both ways.

    The negative control is the removed rule itself, wrapped around the shipping
    search: a reply boundary was permitted only at a multiple of three slots from
    our own command. On the arm it fired at overruns of +0.2, +4.2 and +58.2 ms,
    none of which needs more than the next slot.

    PLANTED AT THE ARM'S LARGEST, because the two small ones no longer reach
    this search at all: `onair._clamp_forgives` leaves a sub-symbol overrun to
    the emission path, which keys the carrier INSIDE its own boundary rather
    than spending the slot. The +0.2 ms case is asserted below as the keeping it
    now is; +58.2 ms is three symbols late, which no reader forgives, and it is
    the overrun this search has to answer.
    """
    real = onair._keyable_slot

    def masked(live, raster, slot, lead_n, **kwargs):
        got = real(live, raster, slot, lead_n, **kwargs)
        command = raster._p3_command_slot
        while command is not None and (got - command) % 3:
            got = real(live, raster, raster.next_slot(got), lead_n, **kwargs)
        return got

    recovered = {}
    for name in ('fixed', 'old'):
        tape = _raster_tape(20)
        s, tx, g, _ = _arm(tmp_path, tape, sl=1, expected_seq=1)
        clock = _duplex(tx, s, tape, 1536)
        monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
        if name == 'old':
            monkeypatch.setattr(onair, '_keyable_slot', masked)
        tx.aim(g, 3)
        # The +58.2 ms overrun the arm printed: a key instant well gone, with
        # nothing else wrong.
        clock.samples = tx.key_instant(g, 3) - clock.key_notice + 2794
        recovered[name], _, _ = onair._regrid(
            clock, g, tx, s.host, s.rx, 3, np.zeros(0, np.float32), 0, SETTLE_N)
        monkeypatch.undo()
    assert recovered['old'] == 6      # the cadence the arm flew
    assert recovered['fixed'] == 4    # the next slot, and only it

    # ...AND THE ARM'S SMALL ONES COST NO SLOT AT ALL NOW. +0.2 ms is inside
    # `onair.KEY_CLAMP_TOL_S`, so the grid keeps the slot and the emission path
    # keys the carrier into its boundary. See `onair._clamp_forgives`.
    tape = _raster_tape(20)
    s, tx, g, _ = _arm(tmp_path, tape, sl=1, expected_seq=1)
    clock = _duplex(tx, s, tape, 1536)
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    tx.aim(g, 3)
    clock.samples = tx.key_instant(g, 3) - clock.key_notice + 10
    kept, _, _ = onair._regrid(
        clock, g, tx, s.host, s.rx, 3, np.zeros(0, np.float32), 0, SETTLE_N)
    assert kept == 3


@requires_recorded
def test_the_comb_moves_to_the_long_geometry_only_on_a_crc_valid_long_frame(
        tmp_path):
    """3.390 s past the packet's phase reference, and not one slot before it.

    `_arm` leaves an emitted CS6 outstanding with no frame behind it yet: the
    request alone leaves the link on the 1.25 s raster with a one-slot comb. The
    recorded long answer is what moves both, and where it puts the reply is the
    spec's own number (`docs/protocols/pactor/pactor3.md`; `P3_LONG_REPLY_S`).
    """
    _, audio = wavfile.read(FIXTURES / 'reference-long.wav')
    s, tx, g, clock = _arm(tmp_path, audio.astype(float) / 32768,
                           sl=3, expected_seq=0)
    assert not s.host.arq.cycle_long and g.ticks == 1
    assert s.host.arq.cycle_request is True and s.host.arq.cycle_command_emitted

    slot, _, _ = _window(s, tx, g, clock)
    assert [ev.cycle_long for ev in s.packets] == [True]
    assert s.host.arq.cycle_long and g.ticks == 3
    onair._p3_place_reply(g, tx, slot)
    at, _, cycle_n, _ = g._p3_peer
    assert cycle_n == round(spec.CYCLE_LONG_S * FS)
    assert (g.boundary(slot) - at) % g.slot_n == (
        round(onair.P3_LONG_REPLY_S * FS) % g.slot_n)


@requires_recorded
@pytest.mark.parametrize('slots', [1, 2])
def test_a_reply_boundary_is_never_refused_for_our_own_command(tmp_path, slots):
    """`_tx` carried its own copy of the mask and refused the burst outright.

    The arm never printed that refusal -- the search moved the slot before the
    transmitter saw it -- but it is the same rule, and a codeword held back for a
    long packet nobody sent is a cycle of the peer's channel spent on silence.
    """
    s, tx, g, _ = _arm(tmp_path, np.zeros(8 * FS))
    assert g._p3_command_slot == 0
    tx.aim(g, slots)
    tx._send_p3_control(arq.CS_REQUEST)
    assert not tx.refused and tx.slots_used == [slots]
