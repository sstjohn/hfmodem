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
import pytest

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
            # A real hole: its samples of silence past the block.
            self.lost_at = [(self.pos, LOST, (LOST + 128) / FS)]
        return got


def test_pkt1_stays_on_the_connect_raster_across_a_capture_loss(
        tmp_path, monkeypatch, capsys):
    clock, keys = _Lossy(), []
    monkeypatch.setattr(onair.ota, 'find_rigctl', lambda *_a: 'fake-rigctl')
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


@pytest.mark.parametrize("host", ["Windows WASAPI", "Core Audio", "ALSA"])
@pytest.mark.parametrize("owned", [False, True])
def test_window_persists_timestamp_uncertainty(tmp_path, monkeypatch, host, owned):
    import json
    from types import SimpleNamespace
    from hfmodem.core.audio import timestamp_limitation
    from hfmodem.core import capture_identity as ci

    limitation = timestamp_limitation(host)
    if owned:
        capture = ci.allocate_default(tmp_path)
        owner = capture.claim('modem')
        directory = owner.root
        monkeypatch.setattr(onair, '_CAPTURE_ROOTS', {directory: owner})
    else:
        directory = tmp_path
        monkeypatch.setattr(onair, '_CAPTURE_ROOTS', {})
    worker = onair._CaptureWriter()
    monkeypatch.setattr(onair, '_WRITER', worker)
    live = SimpleNamespace(lost=0 if limitation else 1920, xruns=0,
        estimated_lost=1920 if limitation else 0, timestamp_limitation=limitation,
        fs=48000, lost_at=[])
    # Keep the real enqueue, background write, sidecar, and owned journal path.
    monkeypatch.setattr(onair, '_peer_bursts', lambda *a: [])
    evidence = onair._CycleEvidence(live, None, None, directory)
    evidence.record('hold_01', np.full(128, .1, np.float32), 8064)
    assert worker.drain(5) == 0
    assert not worker.lost
    path = directory / 'hold_01.json'
    stats = json.loads(path.read_text())
    if owned:
        assert stats['capture_id'] == capture.capture_id
        assert stats['clock_id'] == capture.capture_id + ':local-stream'
    assert stats['end_stream_sample'] == 8192
    if limitation:
        assert stats['timestamp_limitation'] == limitation
        assert stats['estimated_lost_samples'] == 1920
        assert stats['continuous'] is None
        assert stats['lost_samples'] == 0 and stats['grid_loss_seen'] is False
        if owned:
            versions = list((directory / '.hold_01.json.versions').glob('*.json'))
            assert len(versions) == 2
            for version in versions:
                row = json.loads(version.read_text())
                assert row['timestamp_limitation'] == limitation
                assert row['estimated_lost_samples'] == 1920
                assert row['lost_samples'] == 0 and row['continuous'] is None
    else:
        assert stats['lost_samples'] == 1920 and stats['grid_loss_seen'] is True
        assert 'timestamp_limitation' not in stats
        assert 'estimated_lost_samples' not in stats and 'continuous' not in stats
