# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A hole in the input moves the grid back by its size, so our keys stay on the air.

The WS8EOC-80 arm of 2026-09-23 20:45 lost 532 samples at hold_01, between the
connect answer and pkt#1. Every sample index after the hole sat 532 behind the
air, the grid kept its anchor, and all 28 pkt#1 keyed 11.3 ms late on the
raster the peer had latched from our calls; WS8EOC held its CS4 for the rest of
the session. On 09-20 the same peer trained up in 2 to 8.
"""
import numpy as np

from hfmodem.shrike import onair
from hfmodem.tests.shrike import test_call_cadence as cadence
from hfmodem.tests.shrike.test_ackplace import CS_N, DATA_N, FS, SLOT_N

LOST = 532


def test_a_capture_loss_moves_every_boundary_back_by_its_size():
    g = onair._MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=round(0.185 * FS),
                          packet_n=DATA_N, cs_n=CS_N, d_max_n=round(0.13 * FS))
    g.d_n = 0.0875 * FS
    b = g.boundary(7)
    g.note_capture_loss(LOST)
    assert g.boundary(7) == b - LOST
    assert g.d_n == 0.0875 * FS


class _Lossy(cadence._Clock):
    """Drops `LOST` samples of air once the reader is past the connect answer."""

    def __init__(self):
        super().__init__()
        self.hole_after = None

    def read(self, count):
        got = super().read(count)
        if self.hole_after is not None and self.pos >= self.hole_after:
            self.lost, self.hole_after = LOST, None
            self.lost_at = [(self.pos, LOST, LOST / FS)]
        return got


def test_pkt1_stays_on_the_connect_raster_across_a_capture_loss(
        tmp_path, monkeypatch, capsys):
    clock, keys = _Lossy(), []
    monkeypatch.setattr(cadence, '_Clock', lambda: clock)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    emit = onair.RadioTx._tx

    def transmit(tx, audio, what, *args, **kwargs):
        before = len(tx.keyed)
        emit(tx, audio, what, *args, **kwargs)
        if len(tx.keyed) > before:
            keys.append((what, tx.tx_audio_start + clock.lost))
            if what.startswith('connect->') and clock.hole_after is None and not clock.lost:
                clock.hole_after = tx.tx_end + round(.4 * FS)

    monkeypatch.setattr(onair.RadioTx, '_tx', transmit)
    cadence.session(tmp_path, monkeypatch, '--retries', '20', '--max-cycles', '32',
                    answer_from=1)
    out = capsys.readouterr().out
    assert 'CAPTURE LOSS: the input lost 532 samples' in out
    calls = [at for what, at in keys if what.startswith('connect->')]
    packets = [at for what, at in keys if what.startswith('P1 pkt#')]
    assert packets, 'no data packet keyed'
    phase = [((at - calls[0] + SLOT_N // 2) % SLOT_N - SLOT_N // 2) / FS * 1e3
             for at in packets]
    assert np.all(np.abs(phase) < 1.0), phase
    assert 'SESSION INVALID' not in out
