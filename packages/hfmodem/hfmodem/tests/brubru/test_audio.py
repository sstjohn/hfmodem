# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest
from scipy.signal import hilbert, lfilter

from hfmodem.ax25.frame import ui_frame
from hfmodem.brubru import frame, rx, tx, waveform as w
from hfmodem.tests.brubru.vectors import CAPACITIES, frames_for


def impaired(audio, level, noise, offset, inverted, seed):
    analytic = hilbert(audio)
    t = np.arange(len(audio)) / w.RATE
    if inverted:
        analytic = analytic.conj() * np.exp(4j * np.pi * 1500 * t)
    result = (analytic * np.exp(2j * np.pi * offset * t)).real
    if noise:
        power = np.mean(audio[audio != 0] ** 2)
        variance = power / 10 ** ((7, 10)[level] / 10) * (w.RATE / 2) / 3000
        result += np.random.default_rng(seed).normal(0, np.sqrt(variance), len(result))
    return result


@pytest.mark.parametrize('capacity', CAPACITIES)
@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('noise', [False, True])
@pytest.mark.parametrize('offset,inverted', [(0, False), (-15, False), (15, False),
                                              (0, True), (-15, True), (15, True)])
def test_every_class_audio(capacity, level, noise, offset, inverted):
    frames = frames_for(capacity, capacity)
    burst, = tx.transmit(frames, level)
    audio = impaired(burst.audio, level, noise, offset, inverted, capacity + 1000 * level)
    got = rx.receive(audio, centre=1500)
    assert len(got) == 1
    assert list(got[0].frames) == frames
    assert got[0].header.symbols == frame.symbol_count(burst.packet.cls, burst.packet.ext, level)


@pytest.mark.parametrize('chunk', [256, 4096, 48000])
def test_retained_history_and_adjacent_short_bursts(chunk):
    frames = frames_for(15)
    burst, = tx.transmit(frames)
    audio = np.concatenate([burst.audio, burst.audio])
    receiver = rx.Receiver()
    got = []
    for i in range(0, len(audio), chunk):
        got.extend(receiver.feed(audio[i:i + chunk]))
    got.extend(receiver.finish())
    assert [list(d.frames) for d in got] == [frames, frames]
    baseline = rx.receive(audio, centre=1500)
    assert [(d.ref, d.frames, d.header) for d in got] == [(d.ref, d.frames, d.header) for d in baseline]
    assert got[1].ref - got[0].ref == pytest.approx(len(burst.audio), abs=2)


@pytest.mark.parametrize('chunk', [1, 256, 4096, 48000])
@pytest.mark.parametrize('start', [0, 3, 51])
def test_card_decimator_phase_and_filter_state(chunk, start):
    source = np.random.default_rng(9).normal(size=13007)
    decimator = w.Decimator(start)
    got = []
    positions = []
    for i in range(0, len(source), chunk):
        n, values = decimator.feed(start + i, source[i:i + chunk])
        got.extend(values)
        positions.extend(5 * (n + np.arange(len(values))) + decimator.DELAY)
    first = (decimator.DELAY - start) % 5
    expected = lfilter(decimator.TAPS, [1], source)[first::5]
    np.testing.assert_allclose(got, expected, atol=1e-14)
    np.testing.assert_array_equal(positions, start + np.arange(first, len(source), 5))
    with pytest.raises(ValueError, match='discontinuous'):
        decimator.feed(start, source[:5])
    decimator.reset(start)
    n, values = decimator.feed(start, source)
    np.testing.assert_allclose(values, expected, atol=1e-14)


def test_crc_failure_truncation_and_no_preamble_deliver_nothing():
    burst, = tx.transmit([ui_frame('N0CALL', 'CQ', b'validation')])
    assert rx.receive(burst.audio[:len(burst.audio) // 2], centre=1500) == []
    invalid = bytearray(burst.packet.data)
    invalid[15] ^= 0x40
    broken = frame.Packet(bytes(invalid), burst.packet.cls, burst.packet.ext)
    assert rx.receive(tx.render(broken), centre=1500) == []
    tone = np.cos(2 * np.pi * 1500 * np.arange(w.RATE) / w.RATE)
    assert rx.receive(tone, centre=1500) == []


@pytest.mark.parametrize('index,extension', [(2, None), (60, 17)])
def test_invalid_header_and_extension(index, extension):
    packet = frame.assemble(frames_for(345 if extension is not None else 75))
    steps = tx.phase_steps(packet, 0)
    steps[9:17] = np.array([-2, -10])[frame.header_bits(index)]
    if extension is not None:
        steps[17:25] = np.array([-2, -10])[frame.header_bits(extension)]
    audio = w.audio(w.composite(steps, 0))
    assert rx.receive(audio, centre=1500) == []


@pytest.mark.parametrize('limiter', [False, True])
def test_duration_lead_tail_and_class_bound(limiter):
    first, = tx.transmit(frames_for(75), limiter=limiter)
    second, = tx.transmit(frames_for(105), limiter=limiter)
    expected = (frame.symbol_count(3, 0, 0) + 8) * w.SPS
    assert len(first.audio) == expected
    assert np.any(first.audio[:w.SPS]) if not limiter else not np.any(first.audio[:120])
    assert not np.any(first.audio[-w.SPS:])
    limit = first.keyed_s()
    assert first.lead_s() >= 0.04
    assert len(tx.transmit(frames_for(75), limiter=limiter, max_key_s=limit)) == 1
    assert tx.transmit(frames_for(105), limiter=limiter, max_key_s=limit) == []
    assert second.keyed_s() > limit
    onset = np.flatnonzero(np.abs(first.card_audio()) >= 0.5 / 32767)[0] / 48000
    # Cached onset uses a reference packet; peak normalization of another
    # payload shifts its first PCM16 sample by less than a millisecond.
    assert first.lead_s() + onset == pytest.approx(0.1, abs=0.001)
    reference = tx.render(frame.Packet(bytes(15), 0, 0), limiter=limiter)
    from hfmodem.core.levels import at_drive
    from hfmodem.core.resample import for_card
    pcm = np.round(at_drive(for_card(reference, w.RATE)) * 32767).astype(np.int16)
    measured = np.flatnonzero(pcm)[0] / 48000
    assert tx.onset_s(limiter) == measured
    assert first.lead_s() == pytest.approx(max(0.04, 0.1 - measured))


@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('limiter', [False, True])
def test_card_conversion_and_receive_decimator(level, limiter):
    frames = frames_for(75)
    burst, = tx.transmit(frames, level, limiter=limiter)
    card = burst.card_audio()
    decimator = w.Decimator()
    receiver = rx.Receiver()
    receiver.reset(-decimator.DELAY // 5)
    result = []
    for at in range(0, len(card), 4096):
        _, native = decimator.feed(at, card[at:at + 4096])
        result.extend(receiver.feed(native))
    result.extend(receiver.finish())
    assert [list(d.frames) for d in result] == [frames]


def test_duration_bound_splits_containers_before_rendering():
    one = frames_for(75)[0]
    burst, = tx.transmit([one])
    results = tx.transmit([one, one, one], max_key_s=burst.keyed_s())
    assert len(results) == 3
    assert all(d.keyed_s() <= burst.keyed_s() for d in results)


def test_receiver_retains_only_bounded_history():
    receiver = rx.Receiver()
    for _ in range(12):
        assert receiver.feed(np.zeros(48000)) == []
        assert receiver.buffered <= receiver.WINDOW + receiver.STEP
    assert receiver.finish() == []


@pytest.mark.parametrize('prefix_symbols', [10, 20, 30, 40])
def test_preamble_near_acquisition_window_end_is_retained(prefix_symbols):
    frames = frames_for(345)
    burst, = tx.transmit(frames)
    audio = np.concatenate([np.zeros(prefix_symbols * w.SPS), burst.audio])
    got = rx.receive(audio, centre=1500)
    assert [list(d.frames) for d in got] == [frames]


@pytest.mark.parametrize('level', [0, 1])
def test_limiter_defaults_on(level):
    frames = frames_for(75)
    default, = tx.transmit(frames, level)
    limited, = tx.transmit(frames, level, limiter=True)
    unlimited, = tx.transmit(frames, level, limiter=False)
    assert default.limiter is True
    np.testing.assert_array_equal(default.audio, limited.audio)
    np.testing.assert_array_equal(tx.render(default.packet, level), limited.audio)
    assert not np.array_equal(default.audio, unlimited.audio)


def test_tx_pulse_shape():
    # Literal RRC samples at t=0, 1/2, 1, 3/2 and 2 symbols (alpha=0.33),
    # normalized by the centre so the finite-tap gain is immaterial.
    expected = [1.0, 0.560403725019, -0.074256971272, -0.130910616947, 0.052534294213]
    pulse = w.tx_pulse()
    np.testing.assert_allclose(pulse[[59, 71, 83, 95, 107]] / pulse[59], expected,
                               atol=1e-9, rtol=0)


def test_rrc_removable_singularities():
    alpha = 0.33
    edge = 1 / (4 * alpha)
    # Step outside the singularity branch's isclose tolerance on both sides.
    for at in (-edge, 0, edge):
        values = w.rrc(np.array([at - 1e-4, at, at + 1e-4]), alpha)
        assert np.isfinite(values).all()
        assert values[1] == pytest.approx((values[0] + values[2]) / 2, abs=1e-7)
