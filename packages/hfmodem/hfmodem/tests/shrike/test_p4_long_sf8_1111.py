# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real SF8 placement regressions and complete-audio admission negatives."""
from functools import lru_cache
from hashlib import sha256

import numpy as np
import pytest

from hfmodem.shrike.p4receiver import Kind
from hfmodem.tests.shrike.test_p4_long_replay_1111 import (
    FRAMES, _capture_audio, _fresh,
)


FS = 48000
ROWS = {row['id']: row for row in FRAMES}
BASE = 'captures/onair-1003-0017'
WIDE_PLACEMENTS = [
    (f'{BASE}|3108633', 3108633, 3108735),
    (f'{BASE}|3648633', 3648633, 3648740),
    (f'{BASE}|3828633', 3828633, 3828738),
    (f'{BASE}|4008633', 4008633, 4008736),
    (f'{BASE}|4188633', 4188633, 4188735),
]
ENERGY_FAILURES = [row['id'] for row in FRAMES if row['form'] == 'SF8'
                   and 'energy' in (row.get('tip_live_replay') or {}).get('outcome', '')]
REFERENCE = f'{BASE}|2208633'


@pytest.fixture(scope='module')
def tape_audio():
    # Bound retained tape memory while avoiding repeated reads of adjacent rows.
    return lru_cache(maxsize=1)(_capture_audio)


def clip_for(row, audio, predicted):
    origin = predicted - FS // 2
    end = predicted + round(3.300 * FS)
    assert origin >= 0 and end <= len(audio), row['id']
    return audio[origin:end].copy(), origin


def receive(row, clip, origin, predicted, *, hint=None):
    receiver = _fresh(predicted, row['cfo_hz'], FS)
    receiver._long_hint = hint
    def through(seconds):
        return clip[:predicted - origin + round(seconds * FS)]
    prefix_end = predicted + round(.845 * FS)
    early = receiver.read_partial(through(.845), origin, deadline=prefix_end)
    prepared = receiver.prepare_sf8(through(3.230), origin)
    records = receiver.read_window(through(3.300), origin, whole=True)
    return receiver, prepared, [*early, *records]


def assert_identity(row, receiver, records):
    found = [record.observation for record in records
             if record.observation.kind == Kind.LONG
             and record.observation.frame is not None]
    assert any(observation.sf8 and observation.placed
               and not observation.provisional
               and observation.speedlevel == 'SL4-long'
               and observation.frame.status.byte == row['status']
               and sha256(observation.frame.data).hexdigest() == row['data_sha256']
               for observation in found), (row['id'], receiver.live_events)


def test_previous_sf16_hint_allows_same_slot_sf8_preparation_and_read(tape_audio):
    row = ROWS[REFERENCE]
    predicted = row['seed_predicted'][0]
    clip, origin = clip_for(row, tape_audio(row['tape']), predicted)
    receiver, prepared, records = receive(row, clip, origin, predicted, hint='sf16')
    assert prepared is not None and receiver.sf8_prepared(), receiver.live_events
    assert_identity(row, receiver, records)


@pytest.mark.parametrize('row_id,predicted,measured', WIDE_PLACEMENTS)
def test_measured_sf8_offset_outside_original_search_recovers(
        tape_audio, row_id, predicted, measured):
    row = ROWS[row_id]
    # Preserve the actual expected slot rather than granting the measured onset
    # to the receiver. These offsets are about +64 baseband samples.
    assert row['onset_sample'] == measured
    assert row['seed_predicted'] == [predicted]
    assert 61 <= (measured - predicted) * 28800 / FS <= 65
    assert 'below-coherence' in row['tip_live_replay']['outcome']
    clip, origin = clip_for(row, tape_audio(row['tape']), predicted)
    receiver, prepared, records = receive(row, clip, origin, predicted)
    assert prepared is not None, receiver.live_events
    assert_identity(row, receiver, records)


@pytest.mark.parametrize('row_id', ENERGY_FAILURES)
def test_recorded_energy_dip_does_not_veto_complete_crc_identity(tape_audio, row_id):
    row = ROWS[row_id]
    assert row['tip_live_replay']['frame'] is None
    assert 'sf8 prepare negative: energy' in row['tip_live_replay']['outcome']
    predicted = row['seed_predicted'][0]
    clip, origin = clip_for(row, tape_audio(row['tape']), predicted)
    receiver, prepared, records = receive(row, clip, origin, predicted)
    assert prepared is not None, receiver.live_events
    assert_identity(row, receiver, records)


@pytest.mark.parametrize('cut_s', [.9, 3.230, 3.275, 3.27925])
def test_zero_padded_truncated_sf8_never_registers_live_frame(tape_audio, cut_s):
    row = ROWS['captures/onair-1002-2154|3528586']
    predicted = row['seed_predicted'][0]
    clip, origin = clip_for(row, tape_audio(row['tape']), predicted)
    clip[predicted - origin + round(cut_s * FS):] = 0
    receiver, _, records = receive(row, clip, origin, predicted)
    assert not any(record.observation.frame is not None for record in records), receiver.live_events
    assert not any(record.observation.frame is not None
                   for record in receiver.registrations), receiver.live_events


def test_seeded_all_noise_sf8_window_never_registers_live_frame():
    row = ROWS[REFERENCE]
    predicted = row['seed_predicted'][0]
    origin = predicted - FS // 2
    clip = np.random.default_rng(111103).normal(0, .02, round(3.800 * FS))
    receiver, _, records = receive(row, clip, origin, predicted)
    assert not any(record.observation.frame is not None for record in records), receiver.live_events
    assert not any(record.observation.frame is not None
                   for record in receiver.registrations), receiver.live_events


def test_sf16_fallback_replaces_failed_spliced_members_and_placement(monkeypatch):
    from hfmodem.shrike import p4oracle, p4robustrx
    from hfmodem.tests.shrike.test_p4_long_decode_1111 import FX
    with np.load(FX / 'p4-long-0929/symbols.npz', allow_pickle=False) as fixture:
        symbols = fixture['n9vc_0']
    reception = p4robustrx.decode_symbols(symbols)
    expected = reception.frame
    at = FS
    rx = _fresh(at, 0., FS)
    rx._long_prepared[(at, 0.)] = dict(lo=0, onset=at)
    failed = np.random.default_rng(111105).normal(size=len(symbols)).astype(complex)
    monkeypatch.setattr(p4oracle, 'finish_despread_long',
                        lambda *a, **kw: (.8, failed, at, 'prepared'))
    monkeypatch.setattr(p4oracle, 'placed_despread',
                        lambda *a, **kw: (.9, symbols, at+7))
    audio = np.zeros(5*FS)
    assert rx._long(audio, 0)[0] is None
    original = rx._long_jobs[(at, 0., 'sf16')]['search']
    obs, _ = rx._long(audio, 0, fallback=True)
    assert obs is not None and obs.frame == expected, rx.live_events
    assert obs.first_chip == at+7+round(reception.packet_symbol*FS/112.5)
    assert rx._long_jobs[(at, 0., 'sf16')]['search'] is not original


def test_sf16_hint_prepares_both_forms_and_reads_sf16_first(tape_audio):
    row = ROWS['captures/onair-1002-2150|1188369']
    predicted = row['onset_sample']
    clip, origin = clip_for(row, tape_audio(row['tape']), predicted)
    receiver, prepared, records = receive(row, clip, origin, predicted, hint='sf16')
    assert prepared is not None and receiver.sf8_prepared()
    assert (predicted, row['cfo_hz']) in receiver._long_prepared
    frames = [record.observation for record in records if record.observation.frame is not None]
    assert any(obs.kind == Kind.LONG and not obs.sf8 and obs.placed
               and obs.frame.status.byte == row['status']
               and sha256(obs.frame.data).hexdigest() == row['data_sha256'] for obs in frames)
    assert not any(event['event'] == 'sf8-long' for event in receiver.live_events)
