# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded normal-short packets; no synthetic transmitter as the sole judge."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p4frame, p4rx, p4short


@pytest.fixture(scope='module')
def recordings():
    path = Path(__file__).parent / 'fixtures/sigidwiki/p4-short/audio.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        'a8d4bb98eff20c5fe75597d6759b0596f2f30039fbae0b95138e9242c5d102bf')
    with np.load(path, allow_pickle=False) as source:
        return [p4rx.audio_to_baseband(row.astype(float) / 32768, 28800)
                for row in source['audio']]


def test_permutation_matches_independently_archived_constant():
    q = p4short.component_permutation()
    assert np.array_equal(np.sort(q), np.arange(352))
    assert hashlib.sha256(q.astype('<u2').tobytes()).hexdigest() == (
        'a96675e46a0e83610c759bf188b1a1829f5ad1bea64c58b79eb7e599624ed0cb')


@pytest.mark.parametrize('index,crc,digest', [
    (0, 0x6bd1, 'a64c362bf48fd8d166f1ad41b324946619a22974c3243d409843e40aea4eb3ed'),
    (1, 0x2530, '27d73f734ba55584fbcb0aa751a0c32496b1dbf43e2d8b0319034c08ae85ea5e'),
])
def test_recorded_packets(recordings, index, crc, digest):
    frame = p4short.decode_body(recordings[index], first_training_start=2880)
    assert frame is not None
    assert frame.crc == crc
    assert frame.status == p4frame.Status(index + 1, protocol=p4frame.ProtocolStatus.SPEED_UP)
    packet = frame.data + bytes([frame.status.byte]) + frame.crc.to_bytes(2, 'little')
    assert hashlib.sha256(packet).hexdigest() == digest


@pytest.mark.parametrize('offset', [-13.25, 17.5])
def test_frequency_and_gain(recordings, offset):
    original = recordings[0]
    shifted = original * (0.003 - 0.008j) * np.exp(2j * np.pi * offset * np.arange(len(original)) / 28800)
    got = p4short.decode_body(shifted, first_training_start=2880, frequency_offset=offset)
    assert got is not None and got.crc == 0x6bd1


def test_rejects_wrong_geometry_and_missing_data(recordings):
    z = recordings[0]
    assert p4short.decode_body(z[:10000], first_training_start=2880) is None
    assert p4short.decode_body(z, first_training_start=2880 + 208 * 16) is None
    assert p4short.decode_body(z.conjugate(), first_training_start=2880) is None
    assert p4short.decode_body(z, first_training_start=2880, frequency_offset=100) is None


def test_noise_is_not_a_packet():
    random = np.random.default_rng(92)
    for z in (np.zeros(25000), random.normal(size=25000) + 1j * random.normal(size=25000)):
        assert p4short.decode_body(z, first_training_start=2880) is None


@pytest.mark.parametrize('data,start,offset', [
    ([float('nan')], 0, 0), ([[0]], 0, 0), ([0], -1, 0),
    ([0], True, 0), ([0], 0, float('inf')), ([0], 0, 1j),
])
def test_bad_arguments(data, start, offset):
    with pytest.raises(ValueError):
        p4short.decode_body(data, first_training_start=start, frequency_offset=offset)
