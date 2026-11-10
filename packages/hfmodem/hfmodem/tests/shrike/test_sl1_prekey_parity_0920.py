# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Both physical carrier orders of a long SL1 frame decode inside the pre-key window.

VE3KPG's 80 m arm of 2026-09-20 22:24 alternated the leading carrier every
long cycle. Each crop runs from 13 symbols ahead of row 0 to the sample the
live loop actually stopped reading at, so a reader that passes here fits the
window the key was waiting on.
"""
import hashlib
import json
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.shrike import p3acquire, p3rx, rxfront

FIXTURES = Path(__file__).with_name('fixtures') / 'sl1-prekey-0920'
ROWS = json.loads((FIXTURES / 'metadata.json').read_text())


@pytest.fixture(params=ROWS, ids=lambda row: row['physical_lead'])
def frame(request):
    row = request.param
    fs, raw = wavfile.read(FIXTURES / row['file'])
    assert fs == rxfront.FS and raw.ndim == 1
    assert hashlib.sha256(raw.tobytes()).hexdigest() == row['pcm_sha256']
    return row, p3acquire.compensate(raw.astype(float) / 32768, row['offset_hz'])


@pytest.fixture
def trials(monkeypatch):
    starts = []
    decode_at = p3rx.decode_at
    def counted(audio, start, sl, **kw):
        starts.append(start)
        return decode_at(audio, start, sl, **kw)
    monkeypatch.setattr(p3rx, 'decode_at', counted)
    return starts


def one_trial_within_a_quarter_symbol(trials, row):
    # `row0_local` is where the live loop's lock grid passed the CRC; the
    # header block places the same frame a quarter symbol either side of it.
    return len(trials) == 1 and abs(trials[0] - row['row0_local']) <= rxfront.SPS // 4


def test_bounded_reader_takes_either_order_at_the_recorded_cutoff(frame, trials):
    row, audio = frame
    sync = rxfront.SyncedRx()
    sync.packet_level = 1
    ev = sync.sl1_packet_at(audio, row['row0_local'])
    assert ev is not None
    assert ev.packet == (1, row['status'], row['payload'].encode('latin-1'), True)
    assert ev.cycle_long and ev.carrier_swapped == row['carrier_swapped']
    assert one_trial_within_a_quarter_symbol(trials, row)


@pytest.mark.parametrize('shift', [-240, -120, 0, 120, 240])
def test_tracked_ladder_takes_either_order_from_a_lock_off_its_clock(frame, trials, shift):
    row, audio = frame
    sync = rxfront.SyncedRx()
    sync.packet_level = 1
    ev = sync._level_at_lock(audio, row['row0_local'] + shift, 1)
    assert ev is not None
    assert ev.packet == (1, row['status'], row['payload'].encode('latin-1'), True)
    assert ev.carrier_swapped == row['carrier_swapped']
    assert one_trial_within_a_quarter_symbol(trials, row)
