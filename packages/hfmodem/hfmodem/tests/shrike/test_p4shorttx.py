# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The short-packet transmitter, read back by the receiver and the recording."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

from hfmodem.shrike import p4coding, p4frame, p4normal, p4rx, p4short, p4shorttx, placement

TRAINING_START = 2880
SILENCE = 4800


@pytest.fixture(scope='module')
def recordings():
    path = Path(__file__).parent / 'fixtures/sigidwiki/p4-short/audio.npz'
    if not path.exists():
        # The release manifest ships no npz, so the drop has no recordings.
        pytest.skip(f'{path.name} is not in this tree')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        'a8d4bb98eff20c5fe75597d6759b0596f2f30039fbae0b95138e9242c5d102bf')
    with np.load(path, allow_pickle=False) as source:
        return [p4rx.audio_to_baseband(row.astype(float) / 32768, 28800)
                for row in source['audio']]


def read_back(audio, *, offset=0):
    padded = np.concatenate([np.zeros(SILENCE), audio, np.zeros(SILENCE)])
    baseband = p4rx.audio_to_baseband(padded, 48000)
    start = SILENCE * p4rx.SAMPLE_RATE // 48000 + p4shorttx.HEADER_CHIPS * p4rx.SAMPLES_PER_CHIP
    return p4short.decode_body(baseband, first_training_start=start + offset)


def header_estimate(baseband):
    """The recorded header's chips, through the channel its trainings support."""
    z = np.asarray(baseband, complex)
    matched = p4rx.matched_filter(z / np.max(abs(z)))
    targets = np.arange(10, 22)
    training = np.array([p4normal.training_sequence(k + 1) for k in range(6)])
    received = matched[TRAINING_START + 128 + 16 * (208 * np.arange(6)[:, None] + targets)]
    design = training[:, targets[:, None] + np.arange(-2, 3)][[0, 1, 4, 5]].reshape(-1, 5)
    observed = received[[0, 1, 4, 5]].ravel()
    gram = design.conj().T @ design
    channel = np.linalg.solve(gram + 1e-6 * np.trace(gram).real / 5 * np.eye(5),
                              design.conj().T @ observed)
    variance = float(np.mean(abs(observed - design @ channel) ** 2))
    convolution = np.column_stack([np.convolve(channel, row) for row in np.eye(11)])
    target = np.zeros(15)
    target[7] = 1
    inverse = np.linalg.solve(convolution.conj().T @ convolution + variance * np.eye(11),
                              convolution.conj().T @ target)
    chips = TRAINING_START + 128 - 16 * np.arange(p4shorttx.HEADER_CHIPS, 0, -1)
    return (matched[chips[:, None] + 16 * np.arange(-5, 6)] @ inverse
            / np.convolve(channel, inverse)[7])


def coherence(a, b):
    return abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)


def test_coded_field_dimensions_and_tail():
    field = p4frame.encode_field(bytes(range(40)), p4frame.Status(1))
    coded = p4shorttx.coded_bits(field)
    assert coded.shape == (3 * p4shorttx.SYSTEMATIC_BITS,)
    systematic = coded[:p4shorttx.SYSTEMATIC_BITS]
    assert np.array_equal(systematic[:344],
                          np.unpackbits(np.frombuffer(field, np.uint8), bitorder='little'))
    assert not systematic[344:349].any()
    assert p4coding.rsc_encode(
        systematic, initial_state=p4shorttx.INITIAL_STATE).final_state == (0, 0, 0)


@pytest.mark.parametrize('index,crc', [(0, 0x6bd1), (1, 0x2530)])
def test_recorded_frames_survive_a_round_trip(recordings, index, crc):
    recorded = p4short.decode_body(recordings[index], first_training_start=TRAINING_START)
    assert recorded is not None and recorded.crc == crc
    rendered = read_back(p4shorttx.short_packet(recorded.data, recorded.status.byte))
    assert rendered == recorded


@pytest.mark.parametrize('offset', range(-4, 5))
def test_our_own_payload_reads_back(offset):
    frame = read_back(p4shorttx.short_packet(b'W9SSJ', 0x61), offset=offset)
    assert frame is not None
    assert frame.data == placement.field_info(b'W9SSJ', 40, 0x61)[:-1]
    assert frame.status == p4frame.Status(1, protocol=p4frame.ProtocolStatus.SPEED_UP)


@pytest.mark.parametrize('index', [0, 1])
def test_both_recorded_frames_close_with_a_seventh_training(recordings, index):
    """Figure 6.2's B+1 trainings, not the prose's B: the slot after the last
    176 data symbols carries training #7 as strongly as the six before it."""
    matched = p4rx.matched_filter(recordings[index] / np.max(abs(recordings[index])))
    targets = np.arange(10, 22)
    scores = [coherence(p4normal.training_sequence(k + 1)[targets],
                        matched[TRAINING_START + 128 + 16 * (208 * k + targets)])
              for k in range(7)]
    assert min(scores) > 0.4
    assert scores[6] >= 0.8 * np.mean(scores[:6])
    rendered = p4shorttx.body_symbols(np.zeros(1056, np.uint8))
    assert np.array_equal(rendered[-32:], p4normal.training_sequence(7))


def test_header_is_the_waveform_the_other_recorded_frame_sent(recordings):
    assert coherence(p4shorttx.HEADER, header_estimate(recordings[1])) >= 0.9


def test_occupied_bandwidth():
    """Section 10's "approximately 2400 Hz at -25 dB", on the 1500 Hz centre.

    A single burst's periodogram is too ragged to read a skirt off, so this is
    the averaged spectrum; the recorded frames measure 281-2625 Hz the same way.
    """
    frequencies, power = welch(p4shorttx.short_packet(b'', 0x61), 48000, nperseg=1024)
    occupied = frequencies[power > power.max() * 10 ** (-2.5)]
    assert 250 <= occupied.min() and occupied.max() <= 2750
    assert occupied.max() - occupied.min() == pytest.approx(2400, abs=100)
    assert (occupied.min() + occupied.max()) / 2 == pytest.approx(1500, abs=50)


@pytest.mark.parametrize('fs', [48000, 12000])
def test_emission_length(fs):
    assert p4shorttx.EMISSION_S == pytest.approx(0.7928, abs=1e-4)
    assert p4shorttx.short_packet(b'', 0x61, fs).size / fs == pytest.approx(0.7928, abs=1e-3)
