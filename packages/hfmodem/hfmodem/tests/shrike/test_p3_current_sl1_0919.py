# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The current ordinary field must reach the ACK before speculative SL3 work."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3rx, placement, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session

ROOT = Path(__file__).resolve().parents[5]
AUDIT = ROOT / 'working/pactor3-pattern-2029-0919'


def receiver(phase, level=None):
    s = _Session(role=arq.IRS)
    s.rx.p3_wideband_prekey = True
    s.rx.p3_receive_offset_hz = 1.8
    s.rx._seed_p3_changeover_clock(phase)
    s.rx.sync.packet_level = level
    s.host.arq._rx_seen = True
    s.host.arq._expected_seq = 1
    return s


def forbid(*args, **kwargs):
    pytest.fail('current SL1 read opened a speculative decoder')


@pytest.mark.parametrize('level', [None, 1])
def test_first_trim_reaches_its_own_ack_with_only_narrow_budget(monkeypatch, level):
    path = ROOT / 'captures/onair-0919-2005/hold_06.wav'
    if not path.exists():
        pytest.skip('saved WS8EOC first Trim window is unavailable')
    fs, raw = wavfile.read(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert fs == onair.FS
    pcm = raw[:, 0].astype(float) / 32768
    origin = meta['end_stream_sample'] - len(pcm)
    s = receiver(751468, level)
    s.rx.p3_receive_offset_hz = .6
    monkeypatch.setattr(s.rx, '_p3_acquisition_fits', lambda reserve: reserve <= .004)
    monkeypatch.setattr(s.rx.sync, 'wideband_packet_at', forbid)
    monkeypatch.setattr(s.rx, '_read_p3_packet', forbid)
    onair._scan_frame(s.rx, pcm, origin, tracked_only=True)
    assert [ev.packet[:3] for ev in s.packets] == [(1, 0x21, b' Trim')]
    assert s.host.arq.rx_seq == 1
    assert s.host.peer.sent[-1] == ('cs', arq.CS_REQUEST)
    assert not s.rx._p3_repeat_changeover


def test_six_omitted_copies_decode_without_wideband_or_alignment_ladders(monkeypatch):
    path = AUDIT / 'missed-sl1.npz'
    if not path.exists():
        pytest.skip('run working/pactor3-pattern-2029-0919/audit.py for saved fixture')
    meta = json.loads((AUDIT / 'measurements.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['fixture_sha256']
    monkeypatch.setattr(p3rx, 'levels_present', forbid)
    with np.load(path) as clips:
        for row in meta['frames']:
            if row['live_decoded']:
                continue
            s = receiver(row['phase'] - 60000, 1)
            monkeypatch.setattr(s.rx.sync, 'wideband_packet_at', forbid)
            monkeypatch.setattr(s.rx, '_read_p3_packet', forbid)
            onair._scan_frame(s.rx, clips[row['fixture_key']], row['crop_origin'], tracked_only=True)
            assert s.packets[-1].packet[2].hex() == row['payload_hex']
            assert s.packets[-1].carrier_swapped == row['carrier_swapped']


def test_recorded_turn_answers_every_counter_advance_in_its_current_cycle(monkeypatch):
    path = ROOT / 'captures/onair-0919-2029/stream.wav'
    if not path.exists():
        pytest.skip('saved complete WS8EOC turn is unavailable')
    fs, raw = wavfile.read(path)
    assert fs == onair.FS
    rows = list(csv.DictReader((AUDIT / 'peer-frames.csv').open()))
    s = receiver(571822)
    monkeypatch.setattr(s.rx.sync, 'wideband_packet_at', forbid)
    monkeypatch.setattr(s.rx, '_read_p3_packet', forbid)
    monkeypatch.setattr(p3rx, 'levels_present', forbid)
    for row in rows[2:]:
        phase = int(row['phase'])
        origin = phase - 4800
        pcm = raw[origin:phase+40500].astype(float) / 32768
        s.rx.new_cycle()
        onair._scan_frame(s.rx, pcm, origin, tracked_only=True)
        assert s.packets[-1].packet[2].hex() == row['payload_hex'], row['cycle']
        assert s.host.arq.rx_seq == int(row['seq'])
        assert s.host.peer.sent[-1] == ('cs', arq.CS_REQUEST if int(row['seq']) & 1 else arq.CS_ACK)
    assert len(s.packets) == 145


def test_no_budget_does_not_start_narrow_crc(monkeypatch):
    s = receiver(4800)
    monkeypatch.setattr(s.rx, '_p3_acquisition_fits', lambda _: False)
    monkeypatch.setattr(s.rx.sync, 'sl1_packet_at', forbid)
    s.rx._scan_origin, s.rx._tracked_only = 0, True
    assert s.rx._p3_packet(np.zeros(110000)) is None


@pytest.mark.parametrize('swapped', [False, True])
def test_short_window_and_noise_never_update_sl1_lock(swapped):
    sync = rxfront.SyncedRx()
    packet = placement.link_packet(1, b'fresh', 0x21, swapped=swapped)
    pcm = np.pad(packet, (4800, 4800))
    row = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + 4320
    span = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[1]))
    assert sync.sl1_packet_at(pcm[:row+span-1], row) is None
    assert sync.packet_level is None
    rng = np.random.default_rng(919)
    for _ in range(100):
        assert sync.sl1_packet_at(rng.normal(0, .1, len(pcm)), row) is None
    assert sync.packet_level is None
