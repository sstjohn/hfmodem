# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The long-packet transmitter, read back by the receiver and the recording."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

from hfmodem.shrike import (p4coding, p4frame, p4long, p4longtx, p4normal, p4rx,
                            p4shorttx, placement, tablegen)

TRAINING_START = 2880
SILENCE = 4800
SPACING = 32 + p4long.LAYOUT.block_symbols


@pytest.fixture(scope='module')
def recordings():
    path = Path(__file__).parent / 'fixtures/sigidwiki/p4-long/audio.npz'
    if not path.exists():
        # The release manifest ships no npz, so the drop has no recordings.
        pytest.skip(f'{path.name} is not in this tree')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        'c641415075f315e839691230e186136601c06c74b64d95539cabc2a243858fd7')
    with np.load(path, allow_pickle=False) as source:
        return [p4rx.audio_to_baseband(row.astype(float) / 32768, 28800)
                for row in source['audio']]


def read_back(audio, *, offset=0):
    padded = np.concatenate([np.zeros(SILENCE), audio, np.zeros(SILENCE)])
    baseband = p4rx.audio_to_baseband(padded, 48000)
    start = SILENCE * p4rx.SAMPLE_RATE // 48000 + p4longtx.HEADER_CHIPS * p4rx.SAMPLES_PER_CHIP
    return p4long.decode_body(baseband, first_training_start=start + offset)


def equalizer(baseband):
    """The channel the long body's twenty-five trainings support, and its inverse."""
    matched = p4rx.matched_filter(baseband / np.max(abs(baseband)))
    targets = np.arange(10, 22)
    count = len(p4long.LAYOUT.training_offsets)
    training = np.array([p4normal.training_sequence(k + 1) for k in range(count)])
    observed = matched[TRAINING_START + 128
                       + 16 * (SPACING * np.arange(count)[:, None] + targets)].ravel()
    design = training[:, targets[:, None] + np.arange(-2, 3)].reshape(-1, 5)
    gram = design.conj().T @ design
    channel = np.linalg.solve(gram + 1e-6 * np.trace(gram).real / 5 * np.eye(5),
                              design.conj().T @ observed)
    variance = float(np.mean(abs(observed - design @ channel) ** 2))
    convolution = np.column_stack([np.convolve(channel, row) for row in np.eye(11)])
    target = np.zeros(15)
    target[7] = 1
    inverse = np.linalg.solve(convolution.conj().T @ convolution + variance * np.eye(11),
                              convolution.conj().T @ target)
    return matched, inverse, np.convolve(channel, inverse)[7]


def chips(baseband, offsets):
    matched, inverse, gain = equalizer(baseband)
    positions = TRAINING_START + 128 + 16 * np.asarray(offsets)
    return (matched[positions[:, None] + 16 * np.arange(-5, 6)] @ inverse) / gain


def header_estimate(baseband):
    """The recorded header's chips, through the channel its trainings support."""
    return chips(baseband, -np.arange(p4longtx.HEADER_CHIPS, 0, -1))


def coherence(a, b):
    return abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)


def test_coded_field_dimensions_and_tail():
    field = p4frame.encode_field(bytes(range(203)), p4frame.Status(1))
    coded = p4longtx.coded_bits(field)
    assert coded.shape == (3 * p4longtx.SYSTEMATIC_BITS,)
    systematic = coded[:p4longtx.SYSTEMATIC_BITS]
    assert np.array_equal(systematic[:1648],
                          np.unpackbits(np.frombuffer(field, np.uint8), bitorder='little'))
    assert not systematic[1648:1653].any()
    assert p4coding.rsc_encode(
        systematic, initial_state=p4longtx.INITIAL_STATE).final_state == (0, 0, 0)


@pytest.mark.parametrize('index,crc', [(0, 0x5a2c), (1, 0x1bf8)])
def test_recorded_frames_survive_a_round_trip(recordings, index, crc):
    recorded = p4long.decode_body(recordings[index], first_training_start=TRAINING_START)
    assert recorded is not None and recorded.crc == crc
    assert len(recorded.data) == 203
    rendered = read_back(p4longtx.long_packet(recorded.data, recorded.status.byte))
    assert rendered == recorded


@pytest.mark.parametrize('index', [0, 1])
def test_the_chain_runs_forward_on_a_recorded_frame(recordings, index):
    """Re-encoded parity sits as close to the raw decisions as the systematic
    field does, and every wrong reading of the same chain sits at chance."""
    recorded = p4long.decode_body(recordings[index], first_training_start=TRAINING_START)
    decisions = p4long.body_scores(
        recordings[index], first_training_start=TRAINING_START) < 0
    field = p4frame.encode_field(recorded.data, recorded.status)
    coded = p4longtx.coded_bits(field)
    systematic, c1, c2 = np.split(coded, 3)
    hard = np.split(decisions, 3)
    errors = [int((part != seen).sum()) for part, seen in zip((systematic, c1, c2), hard)]
    scattered = np.empty(coded.size, bool)
    scattered[tablegen.INTERLEAVERS['0xb81c0_4968']()] = decisions
    assert max(errors) < 1.6 * errors[0]
    assert max(errors) < 0.12 * p4longtx.SYSTEMATIC_BITS

    order = p4long.component_permutation()
    state = p4longtx.INITIAL_STATE
    unpermuted = np.array(p4coding.rsc_encode(systematic, initial_state=state).parity,
                          np.uint8)
    plain = np.concatenate([np.unpackbits(np.frombuffer(field, np.uint8), bitorder='little'),
                            np.zeros(8, np.uint8)])
    wrong = {
        'C2 without its input permutation': (unpermuted, hard[2]),
        'the components swapped': (c2, hard[1]),
        'the symbol interleaver omitted': (c1, np.split(scattered, 3)[1]),
        'the eight extra bits taken as plain zeros': (
            np.array(p4coding.rsc_encode(plain[order], initial_state=state).parity,
                     np.uint8), hard[2]),
    }
    for name, (part, seen) in wrong.items():
        assert int((part != seen).sum()) > 3 * max(errors), name


@pytest.mark.parametrize('offset', range(-4, 5))
def test_our_own_payload_reads_back(offset):
    frame = read_back(p4longtx.long_packet(b'W9SSJ', 0x63), offset=offset)
    assert frame is not None
    assert frame.data == placement.field_info(b'W9SSJ', 203, 0x63)[:-1]
    assert frame.status == p4frame.Status(3, protocol=p4frame.ProtocolStatus.SPEED_UP)


@pytest.mark.parametrize('index', [0, 1])
def test_both_recorded_frames_close_with_a_twenty_fifth_training(recordings, index):
    """Figure 6.2's B+1 trainings: the slot after the last 207 data symbols
    carries training #25 as strongly as the twenty-four before it."""
    matched, inverse, gain = equalizer(recordings[index])
    targets = np.arange(10, 22)
    scores = [coherence(p4normal.training_sequence(k + 1)[targets],
                        matched[TRAINING_START + 128 + 16 * (SPACING * k + targets)])
              for k in range(25)]
    assert min(scores) > 0.35
    assert scores[24] >= 0.8 * np.mean(scores[:24])
    rendered = p4longtx.body_symbols(np.zeros(4968, np.uint8))
    assert np.array_equal(rendered[-32:], p4normal.training_sequence(25))


def test_header_is_the_waveform_the_other_recorded_frame_sent(recordings):
    assert coherence(p4longtx.HEADER, header_estimate(recordings[1])) >= 0.9


def test_header_is_not_the_short_packets(recordings):
    """Both long frames send one header; it is not the short variant's.

    A residual frequency would hide an identity, so the short header gets the
    whole +/-900 Hz scan and still falls short of what the two long frames
    reach against each other at zero.
    """
    turn = 2j * np.pi * np.arange(p4longtx.HEADER_CHIPS) / p4rx.CHIP_RATE
    scanned = max(coherence(p4longtx.HEADER * np.exp(turn * hz), p4shorttx.HEADER)
                  for hz in np.arange(-900, 900, 0.5))
    assert coherence(p4longtx.HEADER, p4shorttx.HEADER) < 0.3
    assert scanned < 0.5
    assert coherence(header_estimate(recordings[0]), header_estimate(recordings[1])) > 0.9
    for index in (0, 1):
        body = chips(recordings[index], 32 + np.arange(p4longtx.HEADER_CHIPS))
        noise = chips(recordings[index],
                      -np.arange(2 * p4longtx.HEADER_CHIPS, p4longtx.HEADER_CHIPS, -1))
        assert coherence(p4longtx.HEADER, body) < 0.2
        assert coherence(p4longtx.HEADER, noise) < 0.2


def test_occupied_bandwidth():
    """Section 10's "approximately 2400 Hz at -25 dB", on the 1500 Hz centre."""
    frequencies, power = welch(p4longtx.long_packet(b'', 0x63), 48000, nperseg=1024)
    occupied = frequencies[power > power.max() * 10 ** (-2.5)]
    assert 250 <= occupied.min() and occupied.max() <= 2750
    assert occupied.max() - occupied.min() == pytest.approx(2400, abs=100)
    assert (occupied.min() + occupied.max()) / 2 == pytest.approx(1500, abs=50)


@pytest.mark.parametrize('fs', [48000, 12000])
def test_emission_length(fs):
    assert p4longtx.EMISSION_S == pytest.approx(3.2861, abs=1e-4)
    assert p4longtx.long_packet(b'', 0x63, fs).size / fs == pytest.approx(3.2861, abs=1e-3)
