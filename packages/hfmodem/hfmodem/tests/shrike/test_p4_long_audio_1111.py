# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Synthetic full normal-LONG audio through the live receiver, SL5--SL10.

SL6--10 fields, constellations and permutations come from the independent
scalar encoder. SL5 uses its scalar CRC, whitener and RSC, with the established
1656-entry C2 table. The measured SL5 LONG header, training implementation and
transmit pulse are shared scaffolding. In particular, reusing that header for
SL6--10 does not establish their over-air header meanings or interoperability.
No placed symbols, LLRs, decoder substitutions or injected observations enter
the receiver. These are semantic tests; host timing belongs in wall_clock tests.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

import hfmodem
from hfmodem.shrike import p4long, p4longtx, p4normal, p4rx
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, P4Receiver


ROOT = Path(__file__).resolve().parents[5]
FS = 48000
DATA_BYTES = {5: 203, 6: 514, 7: 1031, 8: 1549, 9: 2066, 10: 2584}


def _sl5_symbols(encoder, payload, status):
    covered = payload + bytes([status])
    crc = encoder.crc_x25(covered)
    wire = encoder.whiten(covered + crc.to_bytes(2, 'little'))
    systematic = [(byte >> bit) & 1 for byte in wire for bit in range(8)]
    systematic.extend([0] * 5)
    first, state = encoder._encode(systematic)
    for bit in encoder.flush_bits(state):
        systematic.append(bit)
        state, parity = encoder.rsc_step(state, bit)
        first.append(parity)
    second, _ = encoder._encode([systematic[int(i)]
                                for i in p4long.component_permutation()])
    coded = np.asarray(systematic + first + second, dtype=np.uint8)
    assert len(systematic) == 1656 and coded.size == 4968 and state == 0
    symbols = np.empty(4968, dtype=complex)
    # The independently preserved 4968-symbol order is common to SL5--10.
    symbols[encoder.constants(6).symbol_order] = 1 - 2.0 * coded
    return symbols


def render_long_audio(encoder, sl, payload, status, *, cfo, seed):
    """Render a complete header + 25 trainings + 24 data blocks at 48 kHz."""
    data = (_sl5_symbols(encoder, payload, status) if sl == 5 else
            np.asarray(encoder.generate(sl, payload, status).symbols))
    body = np.empty(25 * 32 + 24 * 207, dtype=complex)
    for block in range(25):
        at = block * 239
        body[at:at + 32] = p4normal.training_sequence(block + 1)
        if block < 24:
            body[at + 32:at + 239] = data[block * 207:(block + 1) * 207]
    chips = np.concatenate((p4longtx.HEADER, body))
    impulses = np.zeros(16 * (len(chips) - 1) + 1, dtype=complex)
    impulses[::16] = chips
    shaped = np.convolve(impulses, p4rx.RRC_TAPS)
    shaped = resample_poly(shaped[16 * p4longtx.HEADER_CHIPS - p4longtx.HEADER_SAMPLES:], 5, 3)
    phase = .37 + 2 * np.pi * (1500 + cfo) * np.arange(len(shaped)) / FS
    waveform = (shaped * np.exp(1j * phase)).real
    waveform /= np.max(abs(waveform))
    onset = FS // 4 + 137  # Deliberately not a chip or callback boundary.
    audio = np.pad(waveform, (onset, FS // 5))
    audio += np.random.default_rng(seed).normal(0, .0002, len(audio))
    return audio, onset


@pytest.mark.parametrize('sl', range(5, 11))
def test_full_audio_normal_long_recovers_explicit_level_and_bytes(long_encoder, sl):
    assert ROOT in Path(hfmodem.__file__).resolve().parents
    payload = np.random.default_rng(111100 + sl).bytes(DATA_BYTES[sl])
    status = 0x40 | (sl % 4)
    cfo = -2.7 if sl % 2 else 3.1
    audio, onset = render_long_audio(long_encoder, sl, payload, status,
                                    cfo=cfo, seed=111200 + sl)
    rx = P4Receiver(FS, admitted=True, cfo_hz=cfo - .12)
    rx.expected = True
    # Packet prediction is 3 ms early, carrier prediction has 0.12 Hz residual.
    # Placement must be established from the actual noisy audio trainings.
    rx.clock = Clock(Expect.PACKET, onset - 144, .33, 3.75, cfo - .12)
    prefix_end = onset + round(.90 * FS)
    early = rx.read_partial(audio[:prefix_end], 0, deadline=prefix_end)
    assert not any(r.observation.frame is not None for r in early)
    records = rx.read_window(audio, 0, whole=True)
    final = [r.observation for r in records
             if r.observation.kind == Kind.LONG and r.observation.frame is not None]
    assert len(final) == 1, rx.live_events
    observation = final[0]
    assert not observation.provisional and observation.placed
    assert observation.speedlevel == f'SL{sl}-long'
    assert observation.frame.data == payload
    assert observation.frame.status.byte == status
    assert abs(observation.stream_sample - onset) <= round(.003 * FS)
    assert abs(observation.cfo_hz - cfo) < .5


@pytest.mark.parametrize('offset', [-1800/239, 1800/239])
@pytest.mark.parametrize('partial_first', [False, True])
def test_high_coherence_carrier_alias_recovers(long_encoder, offset, partial_first):
    payload = np.random.default_rng(111105).bytes(203)
    audio, onset = render_long_audio(long_encoder, 5, payload, 0x41, cfo=0., seed=111205)
    rx = P4Receiver(FS, admitted=True, cfo_hz=offset)
    rx.clock = Clock(Expect.PACKET, onset, .33, 3.75, offset)
    if partial_first:
        rx._normal_long(audio[:onset+round(.9*FS)], 0, partial=True, alias=False)
    obs, coherence = rx._normal_long(audio, 0, alias=True)
    assert coherence > .9
    assert obs is not None and obs.frame is not None, rx.live_events
    assert obs.frame.data == payload and obs.frame.status.byte == 0x41
    assert abs(obs.cfo_hz) < .5
