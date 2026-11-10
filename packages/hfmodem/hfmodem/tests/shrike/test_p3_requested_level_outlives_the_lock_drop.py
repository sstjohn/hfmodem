# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KB5LZK 30 m 2026-09-26: SL2 stays readable after our CS4 for SL3 and the lock's drop."""
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p3frame, placement, rxfront
from hfmodem.tests.shrike.test_p3_sl2_offset_band_0926 import GREETING, ROWS, recorded, receiver


def test_requested_level_outlives_the_lock_drop(tmp_path, monkeypatch):
    row = ROWS[31]
    s = armed(row['row0'] - 3 * 60000, tmp_path, monkeypatch)
    audio = recorded(row)
    quiet = np.random.default_rng(926).normal(0, .01, audio.size)
    for back in (2, 1):
        s.rx.new_cycle()
        onair._scan_frame(s.rx, quiet, row['start'] - back * 60000, tracked_only=True)
    assert not s.packets
    assert s.rx.sync.packet_level is None
    s.rx.new_cycle()
    onair._scan_frame(s.rx, audio, row['start'], tracked_only=True)
    assert [ev.packet[:3] for ev in s.packets] == [GREETING]
    assert s.rx.sync.packet_level == 2
    assert s.rx._p3_expected_levels == (2, 3)   # the peer declined; SL3 stays possible


def armed(row0, tmp_path, monkeypatch):
    s = receiver(ROWS[31])
    s.rx._p3_row0 = s.rx._p3_delivered_at = row0
    s.host.arq._p3_rx_sl = 2
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.attach(s.host)
    tx.sessrx = s.rx
    monkeypatch.setattr(tx, '_tx', lambda *a, **k: setattr(tx, 'n', tx.n + 1))
    tx._send_p3_control(3)
    return s


def test_the_pair_ends_with_the_links_silence_budget(tmp_path, monkeypatch):
    s = armed(ROWS[31]['row0'] - 60000, tmp_path, monkeypatch)
    s.rx.sync.packet_level = None
    for _ in range(s.host.arq.cfg.max_retries):
        s.rx.new_cycle()
    assert s.rx.sl2_prekey_expected()
    s.rx.new_cycle()
    assert not s.rx.sl2_prekey_expected() and not s.rx._p3_expected_levels


STREAM = Path(__file__).resolve().parents[5] / 'captures' / 'onair-0926-1920' / 'stream.flac'


@pytest.mark.skipif(not STREAM.is_file(), reason='needs captures/onair-0926-1920')
def test_arm_a_replay_reads_hold_23_and_hold_31(tmp_path, monkeypatch):
    """Arm A holds 22-31 after our CS4 for SL3 at hold 21, the peer staying at SL2.

    Hold 23 reads at nominal (fix A1), which is a frame at the level we asked
    from, so the pair stands through the lock's drop at holds 24-25 and hold 31
    reads at the followed -28.5 Hz. Hold 30 is not asked for: it decodes only at
    +2.5 and +5 Hz, and with the lock gone only the bounded reader is aimed at it.
    """
    stream = rxfront.load_wav(str(STREAM))
    lead = p3frame.DATA_OFFSET * rxfront.SPS
    span = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[2]))
    first = ROWS[23]['row0'] - 2 * 60000
    s = armed(first, tmp_path, monkeypatch)
    read = {}
    for hold in range(22, 32):
        row = first + (hold - 21) * 60000
        lo = row - lead - 2400
        n = len(s.packets)
        s.rx.new_cycle()
        onair._scan_frame(s.rx, stream[lo:row + span + 2400], lo, tracked_only=True)
        read.update({hold: ev.packet[:3] for ev in s.packets[n:]})
    assert read == {23: GREETING, 31: GREETING}
