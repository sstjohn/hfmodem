# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline reply admission with recorded input/PTT bounds and real CPU charge."""
import time
from types import SimpleNamespace

from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_p3_giveup_1002 import ReplaySeam
from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid


class ChargedReturn:
    def __init__(self, session, seam, masks, end, outdir):
        self.session, self.seam = session, seam
        self.deadline = min(b for b, e in masks if b >= end)-end
        self.radio = driver(irs_grid(), outdir)
        self.radio.host, self.radio.defer_p3_cs = session.host, True
        self.radio.live.limit = 1000*onair.FS
        self.started = time.perf_counter()
        seam.live = SimpleNamespace(transmit=True, lost=0,
            clamp_late=lambda at: (time.perf_counter()-self.started)*onair.FS > at)
        seam.key_instant = lambda *args: self.deadline
        self.marks = []

    def mark(self, name):
        self.marks.append((name, (time.perf_counter()-self.started)*1000))

    def emit(self, grid, hold):
        radio, seam = self.radio, self.seam
        assert seam.pending is not None
        cs, owner = seam.pending
        peer = grid._p3_peer
        actual = irs_grid(anchor=peer[0]+round(onair.P3_REPLY_S*onair.FS)
                          -hold*60000-28800)
        actual.note_p3_packet(*peer[:3], identity=grid._p3_peer_identity)
        radio.aim(actual, hold)
        key_at = radio.key_instant(actual, hold)
        radio.live.now = radio.live.pos = key_at-self.deadline
        radio.send_cs(cs)
        assert radio._pending_p3_cs_owner == owner
        original_tx = radio._tx_admitted
        def charged_tx(*args, **kwargs):
            self.elapsed = time.perf_counter()-self.started
            assert self.elapsed+onair.P3_DECODE_RESERVE_S <= self.deadline/onair.FS, self.marks
            radio.live.spend(round(self.elapsed*onair.FS))
            return original_tx(*args, **kwargs)
        radio._tx_admitted = charged_tx
        radio.emit_pending_cs()
        assert not radio.refused and radio.n == 1
        assert radio._pending_p3_cs is None
        seam.pending = None
        seam.sent.append(('cs', cs))
        del seam.live
        return dict(deadline_ms=self.deadline/48, work_ms=self.elapsed*1000,
                    margin_ms=self.deadline/48-self.elapsed*1000-6)


class DeferredReplay(ReplaySeam):
    """Mirror the live receive -> settle -> emit order without rendering."""
    defer_p3_cs = True

    def __init__(self, host):
        super().__init__()
        self.host, self.pending = host, None

    def send_cs(self, cs):
        self.pending = cs, self.host.arq.p3_tx_owner

    def cancel_pending_cs(self):
        self.pending = None

    def emit_pending_cs(self):
        pending, self.pending = self.pending, None
        if pending is not None and self.host.arq.p3_tx_allowed(pending[1]):
            super().send_cs(pending[0])
            self.host.on_cs_emitted(pending[0])
