# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A hole in the input moves the peer's raster back by its size, and keying goes on.

0921-1209 hold 84: 153904 samples (3.2 s) of air never reached Python, the
peer's packets were then 706 ms off every projection, and ten CS2 ACKs were
keyed against a raster corroborated before the hole. The 706 ms is exactly the
loss modulo the cycle: the projection was wrong because the raster stayed where
it was. Moved back by the stamped size, it puts the peer where the air did.
"""
import numpy as np

from hfmodem.core import rates
from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_p3_reply_placement import (
    CYCLE, FS, IDENTITY, PACKET_N, REPLY_N, irs_grid)

LOST = 153904


def corroborated():
    g = irs_grid()
    for k in range(3):
        g.note_p3_packet(1_500_000 + k * CYCLE, PACKET_N, CYCLE, identity=IDENTITY)
        g.cycles += 1
    assert g._p3_raster_corroborated and g.p3_control_refusal(32) is None
    return g


def test_a_known_lost_stretch_keeps_keying_on_the_raster_the_hole_moved():
    g = corroborated()
    boundary = g.boundary(32)
    g.note_capture_loss(LOST)
    assert g.p3_control_refusal(32) is None
    assert g.boundary(32) == boundary - LOST
    # The peer turns up 706 ms off the old raster, where the air put it, and
    # that is on the raster the hole moved.
    back = 1_500_000 + 13 * CYCLE - LOST % CYCLE
    assert g._peer_raster_position(back, CYCLE) == back
    g.note_p3_packet(back, PACKET_N, CYCLE, identity=IDENTITY)
    assert g.p3_control_refusal(44) is None
    g.p3_reply_shift(44)
    assert (g.boundary(44) - back - REPLY_N) % g.slot_n == 0


class _Live:
    fs = FS

    def __init__(self):
        self.lost = self.xruns = 0
        self.lost_at = []


def test_the_cycle_says_where_the_input_lost_air_before_it_keys(tmp_path, capsys):
    g, live = corroborated(), _Live()
    evidence = onair._CycleEvidence(live, None, g, tmp_path)
    evidence.input_loss("hold_83")
    assert g.p3_control_refusal(32) is None
    live.lost, live.lost_at = LOST, [(5_360_000, LOST, 3.21)]
    evidence.input_loss("hold_84")
    out = capsys.readouterr().out
    assert "the input lost 153904 samples (3206 ms) at hold_84" in out
    assert "the driver went 3210 ms without calling us: air missing" in out
    assert g.p3_control_refusal(32) is None
    evidence.input_loss("hold_85")
    assert "hold_85" not in capsys.readouterr().out
    live.lost, live.lost_at = LOST + 960, [(5_360_000, LOST, 3.21), (5_500_000, 960, 0.0027)]
    evidence.input_loss("hold_86")
    assert "between two callbacks 2.7 ms apart: the stream clock stepped" in capsys.readouterr().out


def test_the_clock_fit_survives_a_loss_step():
    t = np.arange(0, 120.0, 1.0)
    n = np.round(t * FS * (1 + 7e-6)).astype(int)
    points = list(zip(t, n))
    before, after = points[:80], [(s, k - LOST) for s, k in points[80:]]
    ppm, _ = rates.rate_fit(before + after, FS)
    assert ppm < -10_000
    line = rates.clock_report(samples=n[-1] - LOST, lost=LOST, xruns=0, underruns=0,
                              points=before, blocksize=128, fs=FS, breaks=1)
    assert "+7.0" in line and "longest of the 2 unbroken stretches, 79 s" in line
