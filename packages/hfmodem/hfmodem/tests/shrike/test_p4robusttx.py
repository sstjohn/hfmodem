# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The robust-mode renderer: read back by the despreader, then by two recordings."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import get_window

from hfmodem.shrike import p4oracle, p4rx, p4robusttx, p4sig, p4words
from hfmodem.tests.evidence import CORPUS

SILENCE = 4800
ORIGIN = SILENCE * p4rx.SAMPLE_RATE // 48000
SPREAD_ENERGY = float(np.vdot(p4sig.SPREAD16, p4sig.SPREAD16).real)

TAPE = CORPUS / "sigidwiki/PACTORIVaudio1.mp3"
#: Where `SHORT_DIBITS` was lifted from, and how much of the tape to load.
BURST_ONSET_S = 4.863
BURST_SPAN_S = 0.95

DIBITS = Path(__file__).parent / "fixtures/p4-robust/dibits.npz"

#: Every robust short packet on file, by the value and header it carries. B4 is
#: `SHORT_DIBITS`; the other six are the fixture's, from the same tape and from
#: a second station. Provenance: the fixture's own README.
KNOWN = (("B4", 6, "A"), ("B5", 9, "A"), ("B6", 8, "A"),
         ("T0", 13, "A"), ("T1", 13, "B"), ("T2", 49, "A"), ("T3", 49, "B"))


@pytest.fixture(scope="module")
def recorded_dibits():
    """The six recorded packets that are not module literals."""
    if not DIBITS.exists():
        pytest.skip(f"{DIBITS} is absent")
    with np.load(DIBITS) as f:
        return {name: np.asarray(f[name], int) for name in f}


def despread(baseband, start, count, cfo_hz=0.0):
    """`count` SF-16 symbols from a chip impulse index, at a carrier offset."""
    z = np.asarray(baseband, complex)
    rotated = z * np.exp(-2j * np.pi * cfo_hz * np.arange(z.size) / p4rx.SAMPLE_RATE)
    matched = p4rx.matched_filter(rotated)
    chips = matched[start + p4rx.MATCHED_DELAY
                    + p4rx.SAMPLES_PER_CHIP * np.arange(count * 16)].reshape(count, 16)
    return chips @ p4sig.SPREAD16.conj() / SPREAD_ENERGY, chips


def spread_fraction(symbols, chips):
    """Share of the received chip energy lying on the published sequence."""
    return float(np.mean(abs(symbols) ** 2 * SPREAD_ENERGY
                         / np.maximum((abs(chips) ** 2).sum(1), 1e-30)))


def dibits(symbols):
    """Hard quarter-turn decisions, with the residual common rotation removed."""
    step = symbols[1:] * symbols[:-1].conj()
    step /= np.maximum(abs(step), 1e-30)
    rotation = np.angle(np.mean(step ** 4)) / 4
    return (np.round(np.angle(step * np.exp(-1j * rotation))
                     / (np.pi / 2)) % 4).astype(int)


def read_back(audio, count):
    padded = np.concatenate([np.zeros(SILENCE), audio, np.zeros(SILENCE)])
    return despread(p4rx.audio_to_baseband(padded, 48000), ORIGIN, count)


def occupied_bw(audio, floor_db=-25.0, fs=48000):
    magnitude = abs(np.fft.rfft(audio * get_window("hann", audio.size)))
    f = np.fft.rfftfreq(audio.size, 1 / fs)
    band = (f >= 300) & (f <= 2700)
    over = magnitude[band] > magnitude[band].max() * 10 ** (floor_db / 20)
    return float(f[band][over].max() - f[band][over].min())


@pytest.fixture(scope="module")
def recorded_burst():
    """B4 as 48 kHz mono, straight off the tape. The literals never need this."""
    if not TAPE.exists():
        pytest.skip(f"{TAPE} is absent")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is needed to decode the tape")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(TAPE), "-ac", "1", "-ar", "48000",
         "-f", "s16le", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, "<i2").astype(float) / 32768
    lead = int(0.05 * 48000)
    return audio[int(BURST_ONSET_S * 48000) - lead:
                 int((BURST_ONSET_S + BURST_SPAN_S) * 48000)], lead


def test_the_clone_is_one_robust_short_packet():
    assert p4robusttx.SHORT_SYMBOLS == p4robusttx.HEADER_SYMBOLS + 1 + 64
    assert len(p4robusttx.SHORT_DIBITS) == p4robusttx.SHORT_SYMBOLS - 1
    audio = p4robusttx.robust_short(6, "A")
    assert audio.size / 48000 == pytest.approx(p4robusttx.EMISSION_S, abs=1e-4)
    assert audio.size / 48000 * 1e3 == pytest.approx(760.0, abs=1.0)
    assert occupied_bw(audio) == pytest.approx(2400.0, abs=100.0)


def test_the_model_reads_as_the_tape_but_for_its_dropped_first_step():
    """`SHORT_DIBITS` is a measurement; value 6 under header A is the model."""
    model = p4robusttx.short_dibits(6, "A")
    assert model[1:] == p4robusttx.SHORT_DIBITS[1:]
    assert (model[0], p4robusttx.SHORT_DIBITS[0]) == (0, 2)


def test_the_header_is_keyed_at_its_measured_phases():
    """Quarter turns match a real header at 0.70-0.83; its own phases at 0.99."""
    symbols, chips = read_back(p4robusttx.robust_short(6, "A"),
                               p4robusttx.SHORT_SYMBOLS)
    assert spread_fraction(symbols, chips) >= 0.9
    step = symbols[1:] * symbols[:-1].conj()
    for keyed, measured in zip(np.degrees(np.angle(step)),
                               p4robusttx.short_steps(6, "A")):
        assert (keyed - measured + 180) % 360 - 180 == pytest.approx(0, abs=2)


def test_the_datablock_is_differential_bpsk():
    """§6.2 calls the robust payload DQPSK; this packet's 64 symbols are not."""
    header = np.array(p4robusttx.SHORT_DIBITS[:p4robusttx.HEADER_SYMBOLS - 1])
    body = np.array(p4robusttx.SHORT_DIBITS[p4robusttx.HEADER_SYMBOLS - 1:])
    assert set(header.tolist()) == {0, 1, 2, 3}
    assert body.size == 65 and not (body % 2).any()


def test_p4sig_names_the_render():
    """Cleanly keyed it scores where the clean taunus robust references do."""
    assert p4sig.spread_score(p4robusttx.robust_short(6, "A"),
                              48000) >= p4sig.SPREAD_KNEE


def test_header_a_is_the_callers_word_in_the_bank():
    assert p4words.correlate(p4words.BANK["CALLER20"],
                             p4robusttx.phasors(p4robusttx.HEADER_STEPS["A"]))[0] >= 0.98


def test_the_control_signal_clones_are_three_different_bursts():
    assert p4robusttx.CS_SYMBOLS == p4robusttx.HEADER_SYMBOLS + 1 + 6
    assert {len(k) for k in p4robusttx.CS_DIBITS} == {p4robusttx.CS_SYMBOLS - 1}
    assert len(set(p4robusttx.CS_DIBITS)) == 3
    for k, expected in enumerate(p4robusttx.CS_DIBITS):
        audio = p4robusttx.robust_cs_clone(k)
        assert audio.size / 48000 * 1e3 == pytest.approx(243.9, abs=1.0)
        symbols, chips = read_back(audio, p4robusttx.CS_SYMBOLS)
        assert spread_fraction(symbols, chips) >= 0.9
        assert np.array_equal(dibits(symbols), np.array(expected))


@pytest.mark.parametrize("bad", [(), (4,), (-1, 0), np.zeros((2, 2), int)])
def test_symbols_rejects_what_is_not_a_dibit_sequence(bad):
    with pytest.raises(ValueError):
        p4robusttx.symbols(bad)


@pytest.mark.parametrize("bad", [-1, 3, 1.0, True])
def test_robust_cs_clone_rejects_an_index_it_has_no_burst_for(bad):
    with pytest.raises(ValueError):
        p4robusttx.robust_cs_clone(bad)


def test_the_clone_carries_the_recorded_burst_symbol_for_symbol(recorded_burst):
    """The tape is faded; the agreement to look for is at the symbol layer.

    The residual carrier is fitted, not read off the quarter-turn lattice: the
    header is not on it, and a fourth-power estimate leans on its steps.
    """
    audio, lead = recorded_burst
    baseband = p4rx.audio_to_baseband(audio, 48000)
    origin = lead * p4rx.SAMPLE_RATE // 48000
    best = None
    for cfo in np.arange(-4.0, 4.01, 1.0):
        for start in range(origin - 40 * p4rx.SAMPLES_PER_CHIP,
                           origin + 8 * p4rx.SAMPLES_PER_CHIP):
            symbols, chips = despread(baseband, start, p4robusttx.SHORT_SYMBOLS, cfo)
            quality = spread_fraction(symbols, chips)
            if best is None or quality > best[0]:
                best = (quality, symbols)
    quality, recorded = best
    assert quality > 0.3, "no robust-mode lock on the recorded burst"
    rendered, _ = read_back(p4robusttx.robust_short(6, "A"),
                            p4robusttx.SHORT_SYMBOLS)
    turns = np.exp(-1j * np.radians(np.arange(-45, 45, 0.25))[:, None]
                   * np.arange(recorded.size))
    coherence = (abs((recorded * turns) @ rendered.conj()).max()
                 / np.sqrt(np.vdot(rendered, rendered).real
                           * np.vdot(recorded, recorded).real))
    assert coherence >= 0.95


def test_every_recorded_packet_is_a_codeword_under_one_mask(recorded_dibits):
    """Seven packets, two stations, one mask: the whole claim, in one assertion.

    Nothing was fitted to six of these. `BLOCK_MASK` is fixed by B4's coset
    alone, and each of the other six leaves zero residual against it -- one in
    2**58 apiece if the code were not there.
    """
    literals = {"B4": np.array(p4robusttx.SHORT_DIBITS)} | recorded_dibits
    read = []
    for name, value, header in KNOWN:
        steps = literals[name]
        assert steps.size == p4robusttx.SHORT_SYMBOLS - 1
        block = steps[p4robusttx.HEADER_SYMBOLS:]
        assert not (block % 2).any(), f"{name} is not differential BPSK"
        read.append((name, *p4robusttx.block_value((block == 2).astype(int)),
                     p4oracle.header_name(steps)))
    assert read == [(name, value, 64, header) for name, value, header in KNOWN]


def test_the_two_headers_are_the_only_ones_recorded(recorded_dibits):
    """A is the tape's three and taunus's odd bursts, B is taunus's even ones.

    The five clean packets carry their word exactly. B5 and B6 are the tape's
    weak second and third bursts and each miss a step -- which is what the
    knee, and not equality, is for.
    """
    literals = {"B4": np.array(p4robusttx.SHORT_DIBITS)} | recorded_dibits
    word = p4robusttx.HEADER_WORD
    missed = {}
    for name, _, header in KNOWN:
        want = np.array(p4robusttx.HEADER_DIBITS[header])[word]
        missed[name] = int(np.count_nonzero(literals[name][word] != want))
    assert missed == dict(B4=0, B5=2, B6=1, T0=0, T1=0, T2=0, T3=0)
    # The one step the two recordings disagree on at the same header, which is
    # why `HEADER_WORD` starts at one.
    assert {int(literals[n][0]) for n, _, h in KNOWN if h == "A"} == {0, 2}


@pytest.mark.parametrize("header", sorted(p4robusttx.HEADER_DIBITS))
def test_every_value_renders_and_reads_back_as_itself(header):
    for value in range(p4robusttx.BLOCK_SYMBOLS):
        steps = np.array(p4robusttx.short_dibits(value, header))
        block = steps[p4robusttx.HEADER_SYMBOLS:]
        assert not (block % 2).any()
        assert p4robusttx.block_value((block == 2).astype(int)) == (value, 64)
        assert p4oracle.header_name(steps) == header


@pytest.mark.parametrize("value,header", [(0, "A"), (31, "B"), (63, "A")])
def test_a_rendered_value_survives_the_despreader(value, header):
    audio = p4robusttx.robust_short(value, header)
    symbols, chips = read_back(audio, p4robusttx.SHORT_SYMBOLS)
    assert spread_fraction(symbols, chips) >= 0.9
    assert np.array_equal(dibits(symbols),
                          np.array(p4robusttx.short_dibits(value, header)))


def test_noise_is_not_a_codeword():
    """64 random bits reach nothing like a codeword's peak."""
    rng = np.random.default_rng(4)
    peaks = [p4robusttx.block_value(rng.integers(0, 2, 64))[1] for _ in range(400)]
    assert max(peaks) < p4oracle.ROBUST_BLOCK_KNEE


@pytest.mark.parametrize("bad", [-1, 64, 1.0, True, "6"])
def test_a_value_outside_the_field_is_refused(bad):
    with pytest.raises(ValueError):
        p4robusttx.block_bits(bad)


@pytest.mark.parametrize("bad", ["C", "a", "", None])
def test_a_header_that_was_never_recorded_is_refused(bad):
    with pytest.raises(ValueError):
        p4robusttx.short_dibits(6, bad)


@pytest.mark.parametrize("bad", [np.zeros(63, int), np.full(64, 2),
                                 np.zeros((8, 8), int)])
def test_block_value_refuses_what_is_not_a_block(bad):
    with pytest.raises(ValueError):
        p4robusttx.block_value(bad)
