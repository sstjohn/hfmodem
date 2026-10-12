# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KB5LZK 30 m 2026-09-26: a weak SL2 copy the followed offset misses and nominal reads."""
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, onair, p3acquire, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session

FIXTURE = Path(__file__).with_name('fixtures') / 'kb5lzk-0926-sl2'
if not (FIXTURE / 'metadata.json').is_file():
    pytest.skip('KB5LZK SL2 greeting recordings are not included in this distribution',
                allow_module_level=True)
ROWS = {row['hold']: row for row in json.loads((FIXTURE / 'metadata.json').read_text())}
FOLLOWED = -28.5
GREETING = (2, 0x03, b'1440 daily minutes rema')


def recorded(row):
    path = FIXTURE / row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    audio = rxfront.load_wav(str(path))
    assert audio.size == row['end'] - row['start']
    return audio


def receiver(row, *, misses=0):
    s = _Session(role=arq.IRS)
    s.rx.p3_wideband_prekey = True
    s.rx._p3_row0 = row['row0'] - 60000
    s.rx._p3_delivered_at = s.rx._p3_row0
    s.rx._p3_clock_role = arq.IRS
    s.rx.sync.packet_level = 2
    s.rx.sync.misses = misses
    s.rx.p3_receive_offset_hz = FOLLOWED
    s.rx._p3_offset_fixed = True
    s.host.arq.cfg.long_cycle = False
    s.host.arq._rx_seen = True
    s.host.arq._expected_seq = row['seq']
    return s


def decodes(audio, row, hz):
    sync = rxfront.SyncedRx()
    sync.packet_level = 2
    ev = sync.sl2_packet_at(p3acquire.compensate(audio, hz), row['row0'] - row['start'])
    return ev is not None and ev.packet[:3] == GREETING


def test_resampled_crops_hold_the_measured_band():
    hold23, hold31 = ROWS[23], ROWS[31]
    assert FOLLOWED not in hold23['decodes_hz'] and 0.0 in hold23['decodes_hz']
    a23, a31 = recorded(hold23), recorded(hold31)
    assert decodes(a23, hold23, 0.0) and not decodes(a23, hold23, FOLLOWED)
    assert decodes(a31, hold31, 0.0) and decodes(a31, hold31, FOLLOWED)


def test_bounded_sl2_read_retries_at_nominal():
    row = ROWS[23]
    s = receiver(row)
    onair._scan_frame(s.rx, recorded(row), row['start'], tracked_only=True)
    assert [ev.packet[:3] for ev in s.packets] == [GREETING]
    assert s.rx.sync.packet_level == 2 and s.rx.sync.misses == 0
    assert s.rx._p3_corrected[0][2] == FOLLOWED
    assert s.rx.p3_receive_offset_hz == FOLLOWED
    assert s.rx._p3_offset_candidate is None


def test_tracked_retry_at_nominal_keeps_the_lock(monkeypatch):
    row = ROWS[23]
    s = receiver(row, misses=1)
    monkeypatch.setattr(s.rx.sync, 'sl2_packet_at', lambda *a: None)
    onair._scan_frame(s.rx, recorded(row), row['start'], tracked_only=True)
    assert [ev.packet[:3] for ev in s.packets] == [GREETING]
    assert s.rx.sync.packet_level == 2 and s.rx.sync.misses == 0
    assert s.rx.sync.packet_at is not None
    assert s.rx._p3_corrected[0][2] == FOLLOWED
    assert s.rx._p3_offset_candidate is None
