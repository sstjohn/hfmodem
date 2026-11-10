# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A stamp step keeps the reply on grid, and so does missing air.

2026-11-07 K4MSU and WS8EOC: converter re-anchors of 3.6 to 6 ms were taken as
missing air, so `note_capture_loss` forgot the peer's raster and
`p3_control_refusal` refused the next CS ("P3 REPLY CLOCK -- NOT KEYING")
until a fresh CRC frame re-seated it. The gateways' frames stayed exactly on
raster across those steps: nothing was lost, and nothing should move.

WS8EOC 05:01Z hold 34 was the same: its loss line read "the stream clock
stepped, not the air", and the peer's frames stayed exactly one long cycle
apart across it. A clock step moves nothing. Air that really is missing moves
our keys and the peer's raster back by its stamped size, and the reply keys on
the moved grid.
"""
from types import SimpleNamespace

import numpy as np

from hfmodem.shrike import onair, spec

FS, CYCLE = onair.FS, 60000
# onair-1003-0045: a measured reply pair and the 188-sample hole after it.
CONTROL, PEER, WIDTH, HOLE, LOST = 9932461, 10009288, 39120, 10138112, 188
IDENTITY = (False, 1, 2, bytes.fromhex('3e0d'))


def irs_grid():
    g = onair._MasterGrid(CONTROL % CYCLE, CYCLE, 8880, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.d_n, g.d_ref_n = 4413, 17280
    for at in (PEER - 2 * CYCLE, PEER - CYCLE):
        g.note_p3_packet(at, WIDTH, CYCLE, identity=IDENTITY, swapped=False)
        g.cycles += 1
    g.note_p3_control(CONTROL)
    g.note_p3_packet(PEER, WIDTH, CYCLE, identity=IDENTITY, swapped=False)
    assert g._p3_reply_timing() is not None
    return g


def reply_slot(g):
    """The first slot past the hole the reply clock keys before the loss."""
    slot = round((HOLE - g.anchor) / g.slot_n)
    while g.p3_control_refusal(slot) is not None or g.boundary(slot) <= HOLE:
        slot += 1
    return slot


def evidence(raster, step, silence):
    live = SimpleNamespace(lost=step, xruns=0, fs=FS, lost_at=[(HOLE, step, silence)])
    return onair._CycleEvidence(live, SimpleNamespace(host=None), raster, None)


def test_a_clock_step_moves_nothing():
    g = irs_grid()
    slot = reply_slot(g)
    anchor, boundary = g.anchor, g.boundary(slot)
    # 288 samples stamped between two callbacks 2.8 ms apart: no air is missing.
    evidence(g, 288, 0.0028).input_loss('frame-scan')
    assert g.anchor == anchor and g.boundary(slot) == boundary
    assert g.p3_control_refusal(slot) is None


def test_air_missing_moves_our_grid_and_the_peers_and_keeps_keying():
    g = irs_grid()
    slot = reply_slot(g)
    boundary = g.boundary(slot)
    # The driver went 8 ms without calling: the 240 stamped samples are air.
    evidence(g, 240, 0.008).input_loss('frame-scan')
    assert g.boundary(slot) == boundary - 240
    assert g.p3_control_refusal(slot) is None
    # The next CRC frame lands where the air put the peer: on the moved raster.
    g.cycles += 1
    g.note_p3_packet(PEER + 3 * CYCLE - 240, WIDTH, CYCLE, identity=IDENTITY, swapped=False)
    assert g._peer_raster_position(PEER + 3 * CYCLE - 240, CYCLE) == PEER + 3 * CYCLE - 240
    assert g.p3_control_refusal(slot + 3) is None
