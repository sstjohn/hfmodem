# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-2 transmitter: bits -> two-carrier differential-PSK audio.

Generates the PACTOR-2 physical waveform proven in docs/protocols/pactor/pactor2.md: two
DPSK carriers 200 Hz apart at 100 Bd, each carrying an independent differential
stream (DBPSK at the low speed levels, DQPSK higher). The channel-coding chain
runs on the same backend PACTOR-3 uses -- CRC-16/X-25, Viterbi, puncturing, block
interleaver (shrike.coding) -- per the shared-backend result (facts s1, s3): the
P2 decoder calls the same functions as P3. Which of that backend's two codes it
selects is published rather than inferred: `coding.CODE_K9` at every speed level,
punctured to 2/3 at SL3 and 7/8 at SL4.

Scope. The *waveform* (tone geometry, symbol rate, DPSK order) is MEASURED and is
what `modulate` emits. The *coding chain* below the waveform is no longer
self-referential either: a frame this file builds, handed to the reference
decoder's packet decoder at the point that decoder reads its channel softs, comes
back out of its trellis byte-exact, passes its CRC and is reported as a PACTOR-2
packet. Code, packing, interleaver, depth, geometry, CRC and the absence of a
whitener are all confirmed that way rather than against ourselves.

Acquisition is closed too, as of 2026-08-01. `frame_marker` emits the nine-pulse
header that carries a frame's parameters, and a conforming receiver detects it:
sixteen markers played in, thirteen frame descriptors formed, every one on the
right 25 Hz bin pair and carrying back the codeword index that was sent. The
three markers on the published FEC recording read the same way through `p2rx`,
which is what makes that a measurement rather than a round trip.

Transmit is read from outside as of 2026-08-03, at speed levels 1, 2 and 3. An
independent monitor prints the payload of a stream `data_burst` renders, and on
the 32 bursts of a real link re-rendered field for field it returns that link's
own payload text in order. What stood in the way was WHERE EACH LEVEL'S
DIFFERENTIAL ALPHABET STARTS, and nothing here could have found it: the receive
path enumerates every half-sector rotation, so it absorbed a constant turn of the
alphabet and the round trip stayed green while the waveform was wrong. All three
origins are measured through that monitor, SL3's off the air as well
(`cell_steps`); SL4's is the same rule and no measurement.

The emission is SHAPED as of 2026-09-02. [SCS] s1 and s3 publish raised-cosine
pulse shaping and a 450 Hz bandwidth at -50 dB, and every burst this file built
was rectangular until then -- 5.0 per cent of the power outside 1250-1750 Hz and
1899 Hz between the -26 dB edges, against 0.005 per cent and 360 Hz through
`tablegen.symbol_pulse`. The independent monitor reads the shaped waveform at
speed levels 1, 2 and 3 exactly as it read the unshaped one, replica arm
included.

`control_signal` is the one thing here that is a HYPOTHESIS rather than a
measurement, and it says so at length: nothing in the corpus carries a PACTOR-2
control signal, so its keying is PACTOR-3's -- the same six codewords, measured
on a stranger's tape -- put on PACTOR-2's two carriers. It closes with
`p2rx.control_signal_at`, and an independent monitor cannot judge it: a monitor
never answers.

Three things this file had wrong until 2026-08-01, all invisible to a round trip
in the same way. The DQPSK alphabet was on multiples of 90 degrees where `p2rx`'s
own soft demapper reads it off the 45-degree diagonals. The interleaver was
PACTOR-3's backward helical walk rather than the forward pointer walk Annex II
prints. And the field was whitened, which PACTOR-2 does not do -- that one came
from reading PACTOR-3's chain and was carried here by the shared-backend argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.signal import resample_poly, upfirdn

from . import coding, spec, tablegen
from .p2rx import FS_DEFAULT, NOMINAL_TONES_HZ, SYMBOL_RATE, cs_table

_DBPSK_STEP = {(0,): 3 * np.pi / 4, (1,): 7 * np.pi / 4}
"""Bit -> differential phase step. On the diagonals, and MEASURED there.

The diagonals are published: SCS, *The PACTOR-4 Protocol*, section 11.7,
describing the speed level that section 11 states is based on PACTOR-2 speed
level 1 -- "The utilized differential phases are PI/4 when a 0 is being
transferred, and 5*PI/4 when a 1 is being transferred - just like P2." That last
clause is what makes it a PACTOR-2 fact and not an analogy; sections 11.5 and
11.6 inherit the convolutional code and the interleaver from the same P2 level
outright.

WHICH diagonal carries the zero is the document's own phase reference, not ours,
and against a reference decoder the pair it prints is 90 degrees out. Eight
transmissions of the same frames at 45-degree spacing, one reference decoder run
each: it reads the field at 90, 135 and 180 degrees and reads nothing at 45, 225,
270, 315 or 0. A DBPSK demapper accepts everything within 90 degrees of its own
axis, so an open window of exactly (45, 225) puts that axis at 135 with no room
either side -- and puts the PI/4 this file used to emit exactly ON the boundary,
which is why that decoder's channel softs came back three-quarters zeroed rather
than wrong."""

_DQPSK_STEP = {(0, 0): 3 * np.pi / 4, (0, 1): 5 * np.pi / 4,
               (1, 1): 7 * np.pi / 4, (1, 0): np.pi / 4}
"""Dibit -> differential phase step: the diagonals, 90 degrees per Gray step.

Same experiment, same decoder, four transmissions at 90-degree spacing and then
four more at 22.5: it reads the field at 247.5, 270 and 292.5 degrees from where
this table used to sit, and reads nothing at 225 or 315. A DQPSK demapper accepts
within 45 degrees of a constellation point, so an open window of exactly
(225, 315) centres the alphabet at 270 with nothing to fit -- both edges are the
sector boundary, measured rather than interpolated.

That the labelling itself survived the sweep is the other half of it: a sweep
turns one Gray ladder and cannot repair a mirrored one, so a level that decodes
at every rotation inside its window has the right dibit on the right diagonal."""

CODE = coding.CODE_K9
"""K=9, rate 1/2, terminated, bits packed LSB-first, over the field as it stands.

Published: SCS, *The PACTOR-2 Protocol -- A Technical Description* (1996), Annex I
p. 6, "PACTOR-2 always uses a Convolutional Code with k=9 and R=1/2". The code is
read from the document; the LSB-first packing and the absence of a whitener
around it are measured through the reference decoder's own trellis, which returns
a frame built this way byte-exact and passes its CRC on it."""

TONES_PER_FRAME = 2

FLUSH_BITS = CODE.constraint_length - 1
"""Zeros that terminate the trellis, and part of the transmitted packet.

Annex II is explicit that the interleaver's input "includes the status byte, the
2 CRC bytes as well as the 8 flushing bits from the Convolutional Encoder", so
the flush is inside `Path.n_buf` and comes off the top of the field length."""


@dataclass(frozen=True)
class Path:
    """One PACTOR-2 speed level: the frame geometry that level transmits at.

    A level and a short/long frame flag fix coded bits per frame, symbols per
    frame and bits per symbol per tone; the interleaver depth and the code rate
    follow the level alone. The four levels are the PACTOR-2 ladder Table 1 (p. 5)
    prints -- DBPSK, DQPSK, 8-DPSK, 16-DPSK on each of two 100 Bd tones, at rates
    1/2, 1/2, 2/3, 7/8 -- and the coded-bit counts factor exactly as
    `symbols x 2 tones x bits_per_cell`, which is what ties the ladder to the frame
    sizes rather than leaving them two independent tables.

    The receiver does not guess the level: byte 1 of the frame descriptor carries
    it as `sl = (desc[1] >> 1) & 3` with `long = (desc[1] >> 3) & 1`. On real
    off-air audio (`PACTOR-II_FEC_c1500.wav`) the four frames read `desc[1]` =
    01, 00, 06, 01 -> SL0, SL0, SL3, SL0, short frames throughout."""

    name: str
    level: int
    n_buf: int
    n_symbols: int
    stride: int
    bits_per_cell: int
    puncture: coding.Puncture

    @property
    def n_pairs(self) -> int:
        """Trellis steps behind the `n_buf` bits that reach the air.

        Puncturing is the whole difference: the buffer holds what was transmitted,
        the trellis runs over what was encoded."""
        kept = sum(sum(row) for row in self.puncture.mask)
        return self.n_buf * self.puncture.period // kept

    @property
    def crc_bytes(self) -> int:
        """Bytes the CRC covers -- payload, status byte and the CRC itself.

        `n_pairs` steps less the flush, over 8. It divides exactly on all eight
        paths, and all eight then land on Table 1 (p. 5) once the status byte and
        the two CRC bytes are added back: 5, 14, 32, 59 short and 36, 76, 156, 276
        long. Two tables reached from opposite ends agreeing eight times over is
        what pins the rates; a geometry that needed rounding would not."""
        return (self.n_pairs - FLUSH_BITS) // 8


INTERLEAVER_DEPTHS = (16, 8, 4, 2)
"""Block-interleaver depth for SL1..SL4, the same four whatever the frame length.

Published, not measured: SCS, *The PACTOR-2 Protocol -- A Technical Description*
(1996), Annex II, "PACTOR-2 utilizes 4 different interleaver depths: SL1: 16
SL2: 8 SL3: 4 SL4: 2", one set of four with no short/long split. Annex II's
permutation loop steps a pointer forward by the depth and restarts one place
further in whenever it runs past the end; `channel_of_code` walks the same
lattice backwards, so it emits each column in reverse and is otherwise that same
permutation. All four depths divide all eight P2 packet sizes.

The file carried (8, 17, 35, 62) and (39, 79, 159, 279) until this was checked
against Annex II. Those are a reference decoder's per-level packet FIELD LENGTH
IN BYTES, taken for a stride -- `Path.crc_bytes` now derives the same eight
numbers from the geometry. Both tables permute every buffer size, so no round trip
against ourselves can tell them apart."""

PUNCTURES = (coding.RATE_1_2, coding.RATE_1_2,
             coding.PUNCTURE_2_3, coding.PUNCTURE_7_8)
"""Code rate per speed level: Annex I p. 6 gives 1/2, 1/2, 2/3, 7/8, with the
puncturing vector for each of the two levels that has one. Every one of the eight
packet sizes is a whole number of puncture periods, so no path ever transmits a
partial pattern."""


def _paths(long_frame: bool) -> tuple[Path, ...]:
    n_sym = 320 if long_frame else 72
    tag = "long" if long_frame else "short"
    return tuple(
        Path(f"SL{k}-{tag}", k - 1, n_sym * TONES_PER_FRAME * k, n_sym, s, k, p)
        for k, s, p in zip(range(1, 5), INTERLEAVER_DEPTHS, PUNCTURES))


PATHS = _paths(False)
PATHS_LONG = _paths(True)


def interleave_pointer(n_buf: int, depth: int) -> np.ndarray:
    """Annex II's permutation pointer walk: `air[I] = coded[P[I]]`.

    Transcribed from SCS, *The PACTOR-2 Protocol* (1996), Annex II, which prints
    the routine outright:

        S=1; P=0; M=INTERLEAVER_DEPTH;
        for (I=0; I<PACKET_SIZE; I++) {
            OUTPUT_PACKET(I)=INPUT_PACKET(P);
            P=P+M;
            if (P>PACKET_SIZE) P=S++;
        }

    with PACKET_SIZE the punctured coded-bit count, status byte, CRC and the 8
    flush bits included. The printed test is `P>PACKET_SIZE`, which for a size
    divisible by the depth lets P reach PACKET_SIZE itself and index one past the
    end; `>=` is what a 0-based pointer needs, and is what is implemented.

    The walk goes FORWARD from 0 and restarts one place further in. It is not the
    backward helical walk this file used to carry -- that one steps down from the
    end, so it emits each column reversed, and no start offset turns one into the
    other.
    """
    if depth <= 1:
        return np.arange(n_buf, dtype=np.int64)
    p, s = 0, 1
    out = np.empty(n_buf, dtype=np.int64)
    for i in range(n_buf):
        out[i] = p
        p += depth
        if p >= n_buf:
            p, s = s, s + 1
    return out


def channel_of_code(path: Path) -> np.ndarray:
    """`src[k]` = channel-order index holding code-order position k.

    The inverse of `interleave_pointer`: the transmitter reads code position
    `P[I]` into air position `I`, so the receiver's gather is P inverted.
    """
    fwd = interleave_pointer(path.n_buf, path.stride)
    src = np.empty(path.n_buf, dtype=np.int64)
    src[fwd] = np.arange(path.n_buf)
    return src


def soft_bits(diff: np.ndarray, bits_per_cell: int) -> np.ndarray:
    """Differential phasors -> (n_symbols, bits_per_cell) soft bits.

    Positive means a zero bit, the decoder's sign convention. Every level reads
    the alphabet `cell_steps` emits, in the CONJUGATE sense -- so this is the
    modulator's inverse at no rotation, and the axes below move whenever the
    measured alphabets do.

    A phasor's MAGNITUDE is its confidence, and it scales the softs on every
    level: the two-tone waveform fades one carrier at a time, and a faded
    symbol's phase handed to the trellis at full weight steers it wrong exactly
    where it most needs steering. Unit input keeps the old behaviour bit for
    bit. The dense levels take the max-log metric on the phasor itself --
    nearest zero-labelled point against nearest one-labelled, on the cosine --
    rather than a sector count, which is what a Viterbi's branch metric wants.
    """
    d = np.asarray(diff)
    if bits_per_cell == 1:
        return (d * np.exp(3j * np.pi / 4)).real[:, None]
    if bits_per_cell == 2:
        return np.stack([-d.real, -d.imag], axis=1)
    m = 1 << bits_per_cell
    gray = np.array([g ^ (g >> 1) for g in range(m)])
    # the sector labelled g = c ^ (c >> 1) sits at 2*pi*c/m of the input
    score = (np.abs(d)[:, None]
             * np.cos(np.angle(d)[:, None]
                      - (2 * np.pi * np.arange(m) / m)[None, :]))
    out = np.empty((d.size, bits_per_cell))
    for j in range(bits_per_cell):
        one = (gray >> (bits_per_cell - 1 - j)) & 1 == 1
        out[:, j] = score[:, ~one].max(1) - score[:, one].max(1)
    return out


def cell_steps(bits_per_cell: int) -> np.ndarray:
    """Cell value (first bit most significant) -> differential phase step.

    SL1 and SL2 take the tables above. The two denser levels place the same
    Gray-coded ladder on 2^k phases, in the sense `soft_bits` reads it: the sector
    that demapper labels `g = c ^ (c >> 1)` sits at `2*pi*c/m`, and the step a
    transmitter emits for that cell is its negation, because the demapper reads
    the constellation in the conjugate sense. Cell 0 therefore turns the carrier
    by nothing at all.

    THE ORIGIN IS MEASURED AT SL3, off air, against the frame marker riding the
    same carrier a few symbols earlier. Twenty-eight bursts of
    `hb9ak_055246_c1500.wav` decode byte-exact, so the cell value behind every
    data symbol of a real transmitter's frame is known; the marker's eight chips
    are known too, and they measure the rotation a carrier offset puts on every
    one-symbol differential, which is the only thing standing between a
    differential phase and its alphabet. De-rotate by it and each of the eight
    cell values lands on one phase at |r| = 0.995-0.997 over 183-364 symbols --
    a ladder one whole sector ahead of the `2*pi*(c+1)/m` this file used to emit.
    Nothing local could have caught that: the receive path enumerates every
    half-sector rotation, so it absorbs the error, and the marker calibration is
    what makes the measurement absolute rather than one more free rotation.

    SL4's origin rides on the same rule and no measurement -- no level-4 material
    exists here."""
    m = 1 << bits_per_cell
    if bits_per_cell <= 2:
        table = _DBPSK_STEP if bits_per_cell == 1 else _DQPSK_STEP
        out = np.empty(m)
        for cell, step in table.items():
            out[int("".join(str(b) for b in cell), 2)] = step
        return out
    gray = np.array([g ^ (g >> 1) for g in range(m)])
    out = np.empty(m)
    out[gray] = -2 * np.pi * np.arange(m) / m
    return out % (2 * np.pi)


def channel_buffer(lanes: list[np.ndarray], start: int, path: Path) -> np.ndarray:
    """Per-tone soft-bit lanes -> the decoder's channel-order buffer.

    `channel_index = (symbol * n_tones + tone_rank) * bits_per_cell + j`, and
    **`tone_rank` 0 is the upper carrier**. The reference-vector comparison
    on three frames is described in pactor2.md section 5.12.

    **`j` runs the cell's bits LEAST SIGNIFICANT FIRST**, against `soft_bits`,
    which returns them most significant first. Also measured: reversing this axis
    is what takes twelve speed-level-3 frames of `hb9ak_055246_c1500.wav` to the
    exact fields an independent decoder held for them, and the unreversed order
    returns none of them. It is the same LSB-first convention the field's bytes
    are packed in.
    """
    block = np.stack([lane[start:start + path.n_symbols, ::-1]
                      for lane in lanes], axis=1)
    return block.reshape(path.n_buf)


def _cell_values(bits: np.ndarray, bits_per_cell: int) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8)
    if b.size % bits_per_cell:
        b = np.append(b, np.zeros(-b.size % bits_per_cell, np.uint8))
    return b.reshape(-1, bits_per_cell).dot(
        1 << np.arange(bits_per_cell - 1, -1, -1))


@cache
def tx_pulse(fs: int = FS_DEFAULT) -> np.ndarray:
    """`tablegen.symbol_pulse` on this sample rate's grid.

    Published, and the whole reason PACTOR-2 fits a CW filter: [SCS] s1 calls it
    "a two-tone DPSK system with raised cosine pulse shaping, which reduces the
    required bandwidth to less than 500 Hz", and s3 gives the figure -- "around
    450 Hz at minus 50 dB". The kernel is the one PACTOR-3 already transmits
    through, at `tablegen.SPS` samples per symbol; 100 Bd at 48 kHz wants 480, so
    it is resampled rather than redesigned.

    Resampling by a whole ratio lands the 31-tap kernel's own samples on the new
    grid and appends a partial symbol beyond the last of them; TRIMMING THAT TAIL
    is what keeps the kernel symmetric. Left on, the peak sits 59 samples short of
    the centre, the pulse is no longer its own time reverse, and the cascade of
    transmit and matched filter carries 10 per cent of a neighbouring symbol into
    every decision -- 12 degrees of phase error at 48 kHz, which is half a
    16-DPSK sector. Trimmed, the worst inter-symbol tap is 0.9 per cent."""
    L = int(round(fs / SYMBOL_RATE))
    taps = tablegen.symbol_pulse()
    return resample_poly(taps, L, tablegen.SPS)[:(taps.size - 1) * L // tablegen.SPS + 1]


def pulse_lead(fs: int = FS_DEFAULT) -> int:
    """Samples a shaped burst runs before its own phase-reference pulse.

    A shaped burst opens with the leading tail of its first pulse, so sample 0 is
    not the instant Figure 1's raster is written in: the reference pulse sits this
    far in, and every later pulse a symbol period after that. Callers placing a
    burst on the 1.25 s grid -- or placing an answer slot against it -- are the
    ones that need it; a caller that only concatenates does not."""
    return (tx_pulse(fs).size - 1) // 2


def _render(walks: list[np.ndarray], tones: tuple[float, float], fs: int,
            leads: tuple[int, ...]) -> np.ndarray:
    """Per-carrier differential phase walks -> one summed real passband signal.

    The whole transmitter narrows to this: each carrier's walk becomes unit
    phasors one symbol apart, those impulses go through `tx_pulse`, and the
    shaped baseband rides its tone. `leads` is where each carrier's train starts,
    in samples, which is how the T/2 stagger is expressed.

    Rectangular symbols were what this emitted until 2026-09-02, and they cost
    5.0 per cent of the radiated power outside 1250-1750 Hz against 0.005 per
    cent shaped, with the -26 dB bandwidth at 1899 Hz rather than 360.
    `tests/shrike/test_p2.py::test_emitted_bandwidth` holds the figures."""
    L = int(round(fs / SYMBOL_RATE))
    pulse = tx_pulse(fs)
    n_sym = max(len(w) for w in walks) + 1
    out = np.zeros(n_sym * L + pulse.size - 1 + max(leads))
    for steps, f0, start in zip(walks, tones, leads):
        ph = np.concatenate([[0.0], np.cumsum(steps)])
        # Filter the symbol impulses without multiplying the intervening zeros.
        # Dense convolution cost ~30 ms per short entry, inside the keying
        # deadline. Keep its full tail/extent, including the final L-1 zeros.
        bb = np.pad(upfirdn(pulse, np.exp(1j * ph), up=L), (0, L - 1))
        t = np.arange(bb.size) + start
        out[start:start + bb.size] += (bb * np.exp(2j * np.pi * f0 * t / fs)).real
    return out / (np.abs(out).max() + 1e-30)


def modulate(bits_lo: np.ndarray, bits_hi: np.ndarray, bits_per_cell: int = 2,
             tones: tuple[float, float] = NOMINAL_TONES_HZ,
             fs: int = FS_DEFAULT) -> np.ndarray:
    """Two independent DPSK streams -> summed real audio at 100 Bd.

    `bits_per_cell` is the speed level's 1, 2, 3 or 4 -- DBPSK through 16-DPSK.
    The two tones carry independent data (throughput, not diversity; facts
    s2.2)."""
    steps = cell_steps(bits_per_cell)
    return _render([steps[_cell_values(b, bits_per_cell)]
                    for b in (bits_lo, bits_hi)], tones, fs, (0, 0))


MARKER_CHIPS = 8
"""Chips in a frame marker's codeword -- Figure 1's "Header consisting of 8 pulses"."""

MARKER_SYMBOLS = MARKER_CHIPS + 1
"""Pulses a marker occupies. The codeword is scored on DIFFERENTIAL phase, so
eight chips need nine pulses -- Figure 1's "single phase reference pulse" is the
extra one, and it is what the differential is taken against."""

MARKER_STAGGER = 0.5
"""Symbols by which the LOWER carrier leads the upper inside a marker.

Not a free parameter: an acquiring receiver scores the upper carrier against the
phasor row of the current analysis frame and the lower carrier against the row
four frames -- half a symbol at 100 Bd -- earlier. That is the same T/2 stagger
the data field carries (facts s1.4), read off the acquisition path instead."""


def _stagger(fs: int) -> tuple[int, int]:
    """Sample offsets of the two carriers' symbol trains: the T/2 stagger."""
    return 0, int(round(MARKER_STAGGER * fs / SYMBOL_RATE))


def marker_codewords() -> np.ndarray:
    """`(2, 16, 8)` complex codewords, `[carrier][k][chip]`, on the 45 deg diagonals.

    The acquisition correlator's codebook. Its 512 chips of +-1 are commonly read
    as 64 real codewords of 8, but the correlator pairs them: it scores chip row
    `2k` against the cosine of a carrier's differential phase and row `2k+1`
    against the sine, then sums, which is one complex codeword
    `c[k][t] = chip[2k][t] + 1j*chip[2k+1][t]` per k. Every chip is an odd
    multiple of 45 degrees, and that -- not a convention we chose -- is why the
    whole protocol's phase alphabet sits on the diagonals.

    Index 0 is the codebook the LOWER carrier is scored against, index 1 the
    upper. pactor2.md section 5.5."""
    return tablegen.marker_codes()


def marker_steps(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Codeword `k` -> the eight differential phase steps for (lower, upper).

    The correlator sums `e^{i d_t} * c[t]` over the eight one-symbol differentials
    `d`, tap `t` being `t` symbols back, so the steps that align every term are
    `-arg(c)` in reverse tap order. Any common rotation cancels, but the SAME one
    has to serve both carriers or their two sums stop adding in phase."""
    c = marker_codewords()[:, k]
    return -np.angle(c[0][::-1]), -np.angle(c[1][::-1])


def frame_marker(k: int, tones: tuple[float, float] = NOMINAL_TONES_HZ,
                 fs: int = FS_DEFAULT) -> np.ndarray:
    """The nine-pulse frame marker that carries frame parameter `k`, as audio.

    `k` is what an acquiring receiver reads back as the frame descriptor's
    parameter byte: it recovers the codeword index, so the marker is how a frame's
    parameters reach a receiver that has not yet demodulated anything.

    Without this a transmitter emits data symbols and no acquisition point at all:
    a conforming receiver's correlator scores 8-chip codewords against the
    per-symbol differential phase of each carrier, and unmodulated or arbitrarily
    modulated data scores at the noise floor."""
    return _render(list(marker_steps(k)), tones, fs, _stagger(fs))


def marker_index(level: int, long_frame: bool = False, flag: int = 0) -> int:
    """Speed level and frame length -> the codeword index a marker carries.

    A receiver reads the index straight back as the frame descriptor's parameter
    byte and takes `sl = (k >> 1) & 3` and `long = (k >> 3) & 1` off it (section 2).
    Bit 0 is `flag`: it is transmitted and read back, but what it means is not
    established -- real frames on the published FEC recording carry both 0 and 1
    at the same level and length. It is NOT the carrier swap, which was the
    obvious candidate: across the 30 markers `p2rx.find_markers` arms on in
    `hb9ak_055246_c1500.wav` both values occur in both arrangements."""
    return (flag & 1) | ((level & 3) << 1) | (int(bool(long_frame)) << 3)


def frame(bits_lo: np.ndarray, bits_hi: np.ndarray, k: int,
          bits_per_cell: int = 2,
          tones: tuple[float, float] = NOMINAL_TONES_HZ,
          fs: int = FS_DEFAULT) -> np.ndarray:
    """One complete frame: the marker, then the data symbols.

    This is the whole burst a receiver sees. Ahead of the marker it had nothing to
    acquire on and no way to learn the frame's level or length; behind it the
    symbols mean what `k` says they mean.

    ONE PHASE WALK PER CARRIER, marker and data together, and the T/2 stagger held
    across the seam. Neither is decoration: `p2rx.burst_window` places the data
    field AT the instant acquisition reports the marker at, and takes the leading
    lane four analysis frames ahead of the other -- so a burst whose data field
    starts its own phase reference, or whose two lanes are flush, sits half a
    symbol from where its own marker says it is. This used to concatenate the
    marker with `modulate`, which does both of those, and the field behind such a
    marker decodes at no offset the receiver offers. Built as one walk, a speed
    level 3 field goes out and comes back byte-exact through
    `p2rx.decode_expected_burst`, and does so across +-2.5 ms of anchor error.

    `bits_lo` rides the LOWER tone, which at home is the carrier that LEADS --
    rank 1 of `p2rx.lane_tones`. Hand it the tone pair reversed for a swapped ARQ
    cycle and the rest of the involution follows."""
    walks = [np.concatenate([msteps, cell_steps(bits_per_cell)[
                 _cell_values(bits, bits_per_cell)]])
             for msteps, bits in zip(marker_steps(k), (bits_lo, bits_hi))]
    return _render(walks, tones, fs, _stagger(fs))


def data_burst(field: bytes, path: Path, *, swapped: bool = False,
               tones: tuple[float, float] = NOMINAL_TONES_HZ,
               fs: int = FS_DEFAULT) -> np.ndarray:
    """One ARQ cycle's transmission of `field`: marker and staggered data symbols.

    The seam between the coding chain and the waveform, in the transmit
    direction, and the involution `swapped` names is the whole reason it has one
    home. Three conventions have to agree with the receive path at once, and each
    was measured there rather than chosen (`channel_buffer`, `lane_tones`):
    channel rank 0 is the UPPER tone at home and rank 1 the lower; the lower tone
    LEADS by T/2 at home; and a cell's bits leave the channel buffer least
    significant first. On a swapped cycle the two virtual carriers exchange
    tones, taking their marker codebooks, their channel rank AND the stagger with
    them -- which `frame` renders from nothing more than the tone pair reversed,
    because everything else is keyed on argument order.

    Byte-exact through `p2rx.decode_expected_burst` at speed levels 1-3, short
    and long frames, in BOTH arrangements -- and the arms that break the
    involution do not decode, so the round trip is sensitive to what this
    function is for. Speed level 4 renders but does not survive even our own
    front end: its 16-DPSK cells sit closer together than the analysis kernel's
    residual phase error on this unshaped waveform, so nothing at that level is
    validated. `tests/shrike/test_p2.py`.
    """
    channel = np.zeros(path.n_buf, np.uint8)
    channel[channel_of_code(path)] = encode_frame(field, path)
    lanes = [channel.reshape(path.n_symbols, TONES_PER_FRAME,
                             path.bits_per_cell)[:, r, ::-1].ravel()
             for r in range(TONES_PER_FRAME)]
    k = marker_index(path.level, path.n_symbols == 320)
    return frame(lanes[1], lanes[0], k, path.bits_per_cell,
                 tones[::-1] if swapped else tones, fs)


TURNAROUND_S = 0.070
"""Seconds between a packet's last pulse and the answering codeword's first.

PACTOR-3's, MEASURED: sixteen short cycles of `PIII_Complete_1` put a control
signal's phase reference 889.4-891.9 ms after the answered packet's, against a
0.820 s keying (pactor3.md s7). PACTOR-2 is not measured -- no recording in the
corpus carries a PACTOR-2 answer slot at all -- and this is the assumption that
the turnaround is the equipment's rather than the protocol's, which is what
[SCS] s2 says in as many words: "The requirements to operate PACTOR-2 regarding
the transmit delay and the receiver recovery time of the used equipment
therefore remain unchanged in comparison with Level I."
"""


def cs_slot(path: Path) -> float:
    """Seconds from a packet's phase-reference pulse to the answering codeword's.

    The packet's own length plus `TURNAROUND_S`, so the short cycle answers at
    0.880 s and the long one at 3.360 s. PACTOR-3 answers at 0.890 and 3.390 on a
    keying one pulse longer than ours, which is the whole difference between the
    two: [SCS] s2 shortens the standard packet "to 0.8 seconds in order not to
    shorten the maximum possible propagation delay, which is thus still 170
    milliseconds", and Figure 1's 8 header pulses plus 72 data pulses plus the
    reference pulse are the 0.81 s this counts."""
    return (MARKER_SYMBOLS + path.n_symbols) * spec.PULSE_SLOT_S + TURNAROUND_S


def control_signal(index: int, swapped: bool = False,
                   tones: tuple[float, float] = NOMINAL_TONES_HZ,
                   fs: int = FS_DEFAULT) -> np.ndarray:
    """One of the six control signals as audio: DBPSK on both PACTOR-2 carriers.

    HYPOTHESIS, not a measurement, and it is the one thing standing between this
    station and a PACTOR-2 link. What is published ([SCS] s2, Figure 1): six
    codewords of 40 bits at exactly the Plotkin bound, "All CS are always sent in
    DBPSK in order to obtain a maximum of robustness", 20 pulses each preceded by
    a single phase reference pulse, in a 1.25 s cycle whose packet is 0.8 s. What
    is not published: how the twenty bits map onto the twenty pulses, and where in
    the cycle they sit.

    What fills those two gaps is PACTOR-3, which carries the SAME six codewords
    (`spec.CONTROL_SIGNALS`) and whose keying of them is measured on a stranger's
    tape -- DBPSK, one reference pulse then the twenty bits at 100 Bd, on two
    carriers at a fixed instant in the same 1.25 s cycle (pactor3.md s7,
    `placement.control_signal`). Everything there is protocol rather than physical
    layer, so this is that keying on PACTOR-2's two carriers: the bit order
    `cs_table` fixes for the whole modem, this waveform's own measured DBPSK
    alphabet rather than PACTOR-3's on-axis one, and the T/2 stagger the two
    virtual carriers carry everywhere else in PACTOR-2. `swapped` exchanges the
    tones for the ARQ cycle's arrangement, as it does for a data burst.
    """
    steps = cell_steps(1)[cs_table()[index]]
    return _render([steps, steps], tones[::-1] if swapped else tones, fs,
                   _stagger(fs))


def build_field(info: bytes, path: Path) -> bytes:
    """Info bytes -> the field the decoder reconstructs, CRC last."""
    if len(info) != path.crc_bytes - 2:
        raise ValueError(f"{path.name} carries {path.crc_bytes - 2} info bytes")
    return bytes(info) + coding.crc16(info).to_bytes(2, "little")


def encode_frame(field: bytes, path: Path) -> np.ndarray:
    """A full field -> `path.n_buf` coded bits, in code order.

    The field goes down the trellis as it stands: bits packed LSB-first, zero
    flush, no whitening. PACTOR-2 has no whitener -- see `decode_frame`.

    Nothing is padded and nothing is truncated: the field, the flush and the
    puncture period fill `n_buf` exactly."""
    if len(field) != path.crc_bytes:
        raise ValueError(f"{path.name} field is {path.crc_bytes} bytes, got {len(field)}")
    bits = coding.bytes_to_bits(field, msb_first=False)
    return path.puncture.apply(CODE.encode(bits, terminate=True))


def decode_frame(soft: np.ndarray, path: Path) -> bytes:
    """`path.n_buf` code-order soft values -> the field, CRC last.

    The inverse of `encode_frame`: punctured positions come back as the neutral
    0.0, the trellis runs, and its bytes are the field. The CRC is left to the
    caller, because a search over candidate alignments wants the field either way.

    There is no whitening step. `test_accepted_vector` pins the encoded field
    and channel bits; pactor2.md section 5.10 describes the vector.

    **The trellis ends are PINNED at zero.** `encode_frame` starts its register
    at zero and the flush returns it there, so unlike PACTOR-3's high speed
    levels (see `coding.viterbi_decode`) a P2 field grants the decoder both
    boundary states -- and taking them is worth a real burst: leaving the ends
    free lets an impostor path outscore the true field on the faded bursts of
    `hb9ak_055246_c1500.wav`.
    """
    full = path.puncture.depuncture(np.asarray(soft, dtype=float), path.n_pairs)
    bits = coding.viterbi_decode(full, CODE, terminated=True, pinned=True)
    return coding.bits_to_bytes(bits, msb_first=False)


def demod_clean(audio: np.ndarray, bits_per_cell: int = 2,
                tones: tuple[float, float] = NOMINAL_TONES_HZ,
                fs: int = FS_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Coherent inverse of `modulate` for a clean (noiseless, aligned) signal --
    the round-trip check. Returns (bits_lo, bits_hi).

    Matched to `tx_pulse`, not a boxcar over the symbol: the transmitted pulse
    spans nearly four symbols, so integrating one of them reads three quarters of
    a neighbour's energy as this symbol's. Transmit filter and matched filter in
    cascade are a raised cosine, which is zero at every other symbol instant --
    that is what makes the sample below exact rather than approximate, and the
    kernel's odd length puts the cascade's peak on a whole sample."""
    L = int(round(fs / SYMBOL_RATE))
    pulse = tx_pulse(fs)
    delay = pulse.size - 1
    steps = cell_steps(bits_per_cell)
    shifts = np.arange(bits_per_cell - 1, -1, -1)

    def recover(f0):
        n = np.arange(len(audio))
        y = np.convolve(audio * np.exp(-2j * np.pi * f0 * n / fs), pulse)
        sym = y[delay::L]
        d = sym[1:] * np.conj(sym[:-1])
        gap = np.abs(np.mod(np.angle(d), 2 * np.pi)[:, None] - steps[None, :])
        v = np.argmin(np.minimum(gap, 2 * np.pi - gap), axis=1)
        return ((v[:, None] >> shifts) & 1).ravel().astype(np.uint8)

    return recover(tones[0]), recover(tones[1])
