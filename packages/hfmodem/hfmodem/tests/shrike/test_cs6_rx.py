# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Physical P3 carrier ordering must survive an unchanged retry/request header."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import modem, p3acquire, p3frame, p3rx, placement, rx, rxfront, spec

FIXTURES = Path(__file__).with_name('fixtures') / 'cs6_rx'
ROWS = (json.loads((FIXTURES / 'metadata.json').read_text())
        if (FIXTURES / 'metadata.json').is_file() else [])
FS = 48000


def recorded_cases(rows, metadata):
    """Keep absent packaged PCM visible as skips without hiding synthetic cases."""
    if not rows:
        return [pytest.param(None, id='missing-fixtures', marks=pytest.mark.skip(
            reason=f'CS6 recorded fixtures unavailable: {metadata}'))]
    return [pytest.param(row, id=row['file'], marks=(
        () if (FIXTURES / row['file']).is_file() else pytest.mark.skip(
            reason=f"CS6 recorded PCM unavailable: {row['file']}"))) for row in rows]


def recorded(row):
    fs, raw = wavfile.read(FIXTURES / row['file'])
    assert fs == FS and raw.ndim == 1
    assert hashlib.sha256(raw.tobytes()).hexdigest() == row['pcm_sha256']
    return p3acquire.compensate(raw.astype(float) / 32768, row['offset_hz'])


@pytest.mark.parametrize('row', recorded_cases(ROWS, 'metadata.json'))
def test_recorded_retry_order_reaches_tracked_event(row):
    audio = recorded(row)
    ev = rxfront.SyncedRx()._level_at_lock(audio, row['row0_local'], 1)
    assert ev is not None
    assert ev.packet == (1, row['status'], row['payload'].encode(), True)
    assert ev.cycle_long is False
    assert ev.carrier_swapped == row['carrier_swapped']


@pytest.mark.parametrize('swapped', [False, True, None])
def test_full_scan_preserves_packet_geometry_metadata(monkeypatch, swapped):
    packet = p3rx.P3Packet(3, 1, b'order', 7200, 1,
                           long_cycle=True, carrier_swapped=swapped)
    monkeypatch.setattr(p3rx, 'decode_p3_packets',
                        lambda *a, **kw: SimpleNamespace(packets=[packet]))
    event = next(rxfront.decode_events(np.zeros(9600)))
    assert event.packet == (3, 1, b'order', True)
    assert event.start == packet.start
    assert event.cycle_long is True
    assert event.carrier_swapped is swapped


@pytest.mark.parametrize('row', recorded_cases(ROWS[:2], 'metadata.json'))
def test_same_header_value_has_opposite_measured_orders(row):
    audio = recorded(row)
    pulse = rx._pulse(480)
    z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in spec.VH_CHANNELS}
    head = p3rx.header_of(z, range(6960, 7441, 60), placement.SPEED_PATHS[1])
    assert head is not None and head.fit >= p3rx.VH_FIT
    assert head.vh == 1
    assert head.carrier_swapped == row['carrier_swapped']
    assert head.swapped == row['carrier_swapped']
    anchors = p3rx.vh_anchors(z, len(audio))
    near = min(anchors, key=lambda h: abs(h.at + p3frame.DATA_OFFSET * 480 - 7200))
    assert near.vh == 1 and near.carrier_swapped == row['carrier_swapped']


@pytest.mark.parametrize('sl,long_cycle', [(1, False), (2, False), (2, True), (3, False), (6, True)])
@pytest.mark.parametrize('swapped', [False, True])
def test_generated_request_bit_does_not_choose_physical_order(monkeypatch, sl, long_cycle, swapped):
    make_header = p3frame.variable_header

    def opposite_request(level, **kwargs):
        kwargs['request_status'] = not swapped
        return make_header(level, **kwargs)

    monkeypatch.setattr(p3frame, 'variable_header', opposite_request)
    audio = np.pad(placement.link_packet(sl, b'order', 1, swapped=swapped,
                                         long_cycle=long_cycle), (4800, 4800))
    cfg = placement.protocol_config() if sl == 1 else modem.ModConfig()
    row0 = 4800 + (cfg.pulse().size - 1) // 2 + p3frame.DATA_OFFSET * 480
    ev = rxfront.SyncedRx()._level_at_lock(audio, row0, sl)
    assert ev is not None
    assert ev.packet == (sl, 1, b'order', True)
    assert ev.cycle_long == long_cycle
    assert ev.carrier_swapped == swapped


@pytest.mark.parametrize('vh', [0, 1, 8, 9])
@pytest.mark.parametrize('swapped', [False, True])
def test_header_reader_preserves_compatibility_and_measured_order(vh, swapped):
    values = p3frame.VARIABLE_TEMPLATES[vh].reshape(8, 2)
    order = p3frame.VH_ORDER[::-1] if swapped else p3frame.VH_ORDER
    diffs = {cn: values[:, i] for i, cn in enumerate(order)}
    measurement = p3frame.read_header_arrangement(diffs)
    assert measurement[0] == vh and measurement[3] == swapped
    assert p3frame.read_header(diffs) == measurement[:3]


REFERENCE_ROWS = (json.loads((FIXTURES / 'reference-metadata.json').read_text())
                  if (FIXTURES / 'reference-metadata.json').is_file() else [])


@pytest.mark.parametrize('row', recorded_cases(REFERENCE_ROWS, 'reference-metadata.json'))
def test_independent_reference_short_and_long_traffic_keeps_order(row):
    fs, raw = wavfile.read(FIXTURES / row['file'])
    assert fs == FS and hashlib.sha256(raw.tobytes()).hexdigest() == row['pcm_sha256']
    ev = rxfront.SyncedRx()._level_at_lock(raw.astype(float) / 32768,
                                         row['row0_local'], row['sl'])
    assert ev is not None
    assert ev.packet == (row['sl'], row['status'], bytes.fromhex(row['payload_hex']), True)
    assert ev.cycle_long == row['long_cycle']
    assert ev.carrier_swapped == row['carrier_swapped']


@pytest.mark.parametrize('seed', range(6))
def test_missing_header_does_not_turn_noise_into_a_tracked_sl1_packet(seed):
    audio = np.random.default_rng(seed).normal(0, .1, 49200)
    assert rxfront.SyncedRx()._level_at_lock(audio, 7200, 1) is None


def test_tracked_level_preference_keeps_speed_change_fallback(monkeypatch):
    receiver = rxfront.SyncedRx()
    receiver.packet_at = 7200
    receiver.packet_level = 1
    attempted = []
    actual = 1

    def decode(audio, at, sl):
        attempted.append(sl)
        if sl == actual:
            return rxfront.Event(0, 'packet', '', start=at,
                                 packet=(sl, 1, b'level', True))
        return None

    monkeypatch.setattr(receiver, '_level_at_lock', decode)
    monkeypatch.setattr(p3rx, 'channel_energy', lambda *a, **k: None)
    monkeypatch.setattr(p3rx, 'levels_present', lambda energy: (3, 4, 2, 1))
    audio = np.zeros(49200)
    assert receiver.packet(audio).packet[0] == 1
    assert attempted == [1]
    attempted.clear()
    actual = 2
    assert receiver.packet(audio).packet[0] == 2
    assert attempted == [1, 3, 4, 2]
    assert receiver.packet_level == 2
    attempted.clear()
    assert receiver.packet(audio).packet[0] == 2
    assert attempted == [2]
    actual = None
    for _ in range(receiver.MAX_MISSES):
        assert receiver.packet(audio) is None
    assert receiver.packet_level is None and not receiver.locked


def test_blind_packet_observation_seeds_level_preference():
    receiver = rxfront.SyncedRx()
    receiver.observe(rxfront.Event(0, 'packet', '', start=7200,
                                  packet=(3, 1, b'level', True)))
    assert receiver.packet_level == 3


@pytest.mark.parametrize('protocol,level', [('PACTOR-1', 0), ('PACTOR-1', 1), (None, 0)])
def test_p1_observation_cannot_seed_a_p3_packet_lock(monkeypatch, protocol, level):
    receiver = rxfront.SyncedRx()
    p3 = rxfront.Event(0, 'packet', '', protocol='PACTOR-3', start=7200,
                       packet=(3, 1, b'level', True))
    receiver.observe(p3)
    receiver.observe(rxfront.Event(0, 'packet', '', protocol=protocol, start=9600,
                                  packet=(level, 1, b'p1', True)))
    assert receiver.packet_level is None and receiver.packet_at is None
    attempted = []
    monkeypatch.setattr(receiver, '_packet_at_lock', lambda a, at: attempted.append(at))
    assert receiver.packet(np.zeros(1)) is None
    assert attempted == []
    receiver.observe(p3)
    assert receiver.packet_level == 3 and receiver.packet_at == 7200


def test_real_speed_changes_replace_preferred_level():
    receiver = rxfront.SyncedRx()
    for sl in (1, 3, 2):
        audio = np.pad(placement.link_packet(sl, b'speed', 1), (4800, 4800))
        cfg = placement.protocol_config() if sl == 1 else modem.ModConfig()
        receiver.packet_at = 4800 + (cfg.pulse().size - 1) // 2 + p3frame.DATA_OFFSET * 480
        ev = receiver.packet(audio)
        assert ev is not None and ev.packet == (sl, 1, b'speed', True)
        assert receiver.packet_level == sl


@pytest.mark.parametrize('swapped', [False, True])
@pytest.mark.parametrize('request_status', [False, True])
def test_explicit_request_status_is_independent_of_carrier_swap(swapped, request_status):
    value = p3frame.variable_header(3, swapped=swapped, long_cycle=True,
                                    request_status=request_status)
    assert value == 12 | int(request_status)
