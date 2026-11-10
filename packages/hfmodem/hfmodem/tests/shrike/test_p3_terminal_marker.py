# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Measured stock post-QRT marker, independent of an ordinary packet CRC."""
import hashlib

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import p3frame, p3rx, placement, rx, spec
from hfmodem.tests.kestrel.corpora import RF_CORPUS

FS, SPS = 48000, 480
STOCK = RF_CORPUS/'PIII_Complete_1.wav'


def measure(audio, search):
    pulse = rx._pulse(SPS)
    delay = (len(pulse)-1)//2
    z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in (5, 12)}
    header = p3rx.header_of(z, search, placement.DETECT)
    assert header is not None
    path = p3rx.path_for(1, header)
    # Sample each physical carrier on the measured header's virtual clock.
    phases = {}
    for cn, offset in zip(path.tones, path.clock_offsets(SPS)):
        at = header.at+np.arange(8, 82)*SPS+offset+delay
        y = z[cn][at]
        phases[cn] = np.angle(y[1:]*y[:-1].conj())-header.rot
    soft = rx.case0_softs(
        z, header.at+8*SPS, fs=FS, delay=delay,
        order=header.tones(p3frame.VH_ORDER), rot=header.rot, path=path,
        lead=dict(zip(path.tones, path.clock_offsets(SPS))))
    chronological = soft[np.argsort(placement.case0_map(path))]
    return header, phases, chronological, z


def assert_marker(phases, chronological, swapped, *, tolerance):
    expected = np.tile([3*np.pi/4, -np.pi/4], 36)
    assert np.array_equal(chronological < 0, np.tile([False, False, True, True], 36))
    for home, trailer in ((5, -np.pi/4), (12, 3*np.pi/4)):
        physical = spec.CARRIER_SWAP[home] if swapped else home
        difference = np.angle(np.exp(1j*(phases[physical]-np.append(expected, trailer))))
        assert np.max(np.abs(np.rad2deg(difference))) < tolerance


@pytest.mark.parametrize('swapped', [False, True])
@pytest.mark.parametrize('header_bit', [0, 1])
def test_rendered_marker_keeps_header_body_stagger_and_virtual_trailer(
        swapped, header_bit):
    audio = np.pad(placement.terminal_packet(swapped=swapped, header_bit=header_bit),
                   (4800, 4800))
    header, phases, chronological, z = measure(audio, range(9420, 10620, 30))
    assert header.fit > .99 and header.vh == header_bit
    assert header.swapped == swapped and not header.long_cycle
    assert_marker(phases, chronological, swapped, tolerance=5)
    # This known-pattern marker must not accidentally become an ordinary data
    # packet accepted by the existing CRC path.
    for delta in (-120, 0, 120):
        assert p3rx.decode_at(audio, header.at+9*SPS+delta, 1,
                             Z=z, header=header) is None
    ordinary = placement.link_packet(1, b'', 0x19, swapped=swapped)
    assert len(audio)-9600 == len(ordinary)  # Same short packet symbol extent.


@pytest.mark.skipif(not STOCK.exists(), reason='stock P3 recording unavailable')
def test_stock_terminal_marker_has_the_same_uncoded_body_and_trailer():
    assert hashlib.sha256(STOCK.read_bytes()).hexdigest() == (
        '9adb8598734a1d4b61a277a27d6b9d6c035ab19d0a473003bda45b50061fa81b')
    fs, raw = wavfile.read(STOCK)
    assert fs == FS
    audio = raw[74*FS:, 0].astype(float)/32768
    header, phases, chronological, z = measure(audio, range(93600, 95000, 30))
    assert header.at+74*FS == 3641970
    assert header.vh == 1 and header.fit > .998 and not header.swapped
    assert_marker(phases, chronological, False, tolerance=5)
    for delta in (-120, 0, 120):
        assert p3rx.decode_at(audio, header.at+9*SPS+delta, 1,
                             Z=z, header=header) is None


@pytest.mark.parametrize('header_bit', [-1, 2])
def test_terminal_header_bit_rejects_other_values(header_bit):
    with pytest.raises(ValueError, match='header_bit'):
        placement.terminal_packet(header_bit=header_bit)
