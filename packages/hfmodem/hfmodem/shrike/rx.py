# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-III receiver: audio -> header bytes + control signals.

shrike has been transmit-only; a real ARQ handshake with a Winlink gateway needs
the reverse direction too. This module inverts the machinery in `p3frame`/
`placement`/`coding`, and independently decodes the six 20-bit control signals
carried DBPSK on tones 5 & 12 (nearest-codeword over `spec.CONTROL_SIGNALS`).

STATUS OF THE STAGES. Each is stated against an independent PACTOR-III decoder,
which is receive-only and so settles what a real receiver accepts but never what
one transmits:

  * SL>=2 header (case 1) ...... PROVEN end to end from audio.
    `decode_case1_header` takes six tones of lag-1 DBPSK -> the stride-35 helical
    de-interleave -> K=7 (0o171, 0o133) with the flush -> de-whiten -> CRC-16/X25
    over 24 bytes, and returns `placement.data_packet`'s field byte-exact, with
    noise and a case-0 packet as negatives. None of that chain is shrike's guess:
    every stage is one an independent PACTOR-III decoder accepts. Offered the
    frame as soft codewords, that decoder's trellis and CRC take exactly one
    convention and reject thirteen neighbours; given the audio this path decodes,
    it reads the same field back byte-exact, 5/5 against a 0/5 random-grid
    control. Its soft raster and this front end agree 0.935 by sign / r = 0.91 on
    that audio.
  * case-0 (SL0) header ........ PROVEN end to end on CLEAN audio.
    `decode_case0_header` recovers shrike's own case-0 TX header -- the one an
    independent decoder read back CRC-valid (###STATUS, field 814800008000c8e1)
    -- byte-exact.
  * raster gather .............. PROVEN. `gather` on that decoder's real soft
    raster (the `vhinF.bin` capture) reproduces its de-interleaved buffer 1.0.
  * control signals (ACK/NAK) .. PROVEN end to end on CLEAN audio, exact when
    clean and through heavy AWGN. This is the peer-ACK path ARQ listens on.
  * acquisition / timing ....... PROVEN on real traffic, and not from here.
    `p3rx.header_anchors` matches the published packet headers, which places a
    packet to the sample and names its speed level, cycle length and carrier
    swap. NOT the PATTERN_A correlator below: a packet carries ONE
    phase-reference symbol, not the eight-symbol pilot block that correlator
    matches, so on our own transmissions and on a real speed-level-3 packet alike
    it peaks in the middle of the data field rather than at its start. It
    survives here as what it measurably is -- a trigger that fires on multitone
    energy, which is all `rxfront` asks of it. The envelope search it feeds is
    now only for the two speed levels whose header block is too narrow to match:
    level 1 carries no constant header at all and level 2 four of the sixteen.

  * the occ15 soft-raster comparison remains an optional diagnostic for a
    different span from the field `p3rx` decodes. `tests/shrike/test_rx.py`
    reports the comparison without asserting it as a receiver acceptance test.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from . import spec, tablegen
from scipy.signal import oaconvolve, resample_poly

from . import coding, p3frame, placement

FS_DEFAULT = spec.SAMPLE_RATE
SPS_DEFAULT = 480                       # 48 kHz / 100 Bd
HEADER_TONES = (5, 12)
HEADER_CODE = coding.ConvCode(7, (0o171, 0o133))
"""K=7 rate-1/2 SL>=2 header FEC -- see placement.CONV_GENERATORS for the
measurement (an independent decoder's trellis and CRC accept this pair and no
other, against thirteen matched negatives)."""

CASE0_CODE = coding.ConvCode(9, (0o657, 0o435))
"""K=9 rate-1/2 SL0 (case-0) header FEC."""
CASE0_MAP = placement.case0_map(placement.DETECT)
"""case-0 code-order -> cell-order permutation: 16 passes of 9.

`placement` owns it because the transmitter lays the cells down through the same
permutation, and the depth follows the buffer -- `placement.CHANGEOVER` runs the
same sixteen columns seven rows deep."""
BIT_PHASE_CASE0 = 0.75 * np.pi
"""case-0 DBPSK: bit 0 sits at 135 deg, the inverse of the sign rule every other
speed level uses."""

RASTER_BLOCKS = (52, 615, 1178, 1741)
"""Byte offsets of the four 72-symbol phase-soft streams inside the header soft
raster (2 tones x {I,Q}), in the layout a reference decoder builds and `gather`
indexes."""


def _gather_map() -> np.ndarray:
    return tablegen.hdr_raster_src()


# ---------------------------------------------------------------------------
# Proven backend: soft raster / code-order softs -> header bytes
# ---------------------------------------------------------------------------

def gather(raster: np.ndarray) -> np.ndarray:
    """The stride-563 diagonal gather: soft raster -> de-interleaved order.

    `raster[gather_map[k]]` is de-interleaved soft k (288 values). The optional
    `vhinF.bin` and `deintF.bin` reference vectors check this mapping. The
    following 288-to-432 depuncture and cross-cycle soft accumulation construct
    the final Viterbi input.
    """
    idx = _gather_map()
    return np.asarray(raster)[idx]


def case0_accepts(audio: np.ndarray, *, fs: int = FS_DEFAULT,
                  search: range | None = None,
                  Z: dict | None = None) -> Iterator[tuple[int, bytes]]:
    """Every (start, field) the case-0 CRC accepts over `search`, in order.

    SL0 (case-0) header: 2 tones {5,12}, lag-1 DBPSK, K=9 -> whiten -> CRC-16/X25.
    Demodulates 1 pilot + 72 grid symbols per tone, lag-1 differential, maps
    (symbol, tone) -> cell -> code order (CASE0_MAP), K=9 Viterbi, de-whitens and
    checks the CRC.

    VALIDATED AGAINST A STATION THAT IS NOT US. This chain reads DL6MAA's
    speed-level-1 entry packet out of `PIII_Complete_1` CRC-valid, field
    0f8f87c7c31a6689, at five consecutive eighth-symbol alignments and nowhere
    else in the 79 s recording -- and those five payload bytes are the first five
    of the repeating pattern the same station's level 3 packets carry through a
    different code and a different interleave. Until then case 0 had only ever
    been checked against shrike's own transmitter (field 814800008000c8e1), which
    a self-consistent error would have passed just as happily.

    The blind scan reads the HOME carrier order, because a scan has no header
    block to read the swap out of. An anchored decode does -- `p3rx.decode_at`
    passes it, through `case0_softs`.

    It yields rather than returning the first hit because WHERE a scan accepted,
    and whether it accepted anywhere near there as well, is what tells a frame
    from an accident -- see `p3rx.confirmed`. The caller that only wants a
    yes/no is `decode_case0_header`.

    `Z` is the per-tone baseband, already filtered. It is passed for the same
    reason it is in `decode_at` -- the matched filter is 1860 taps over
    the whole buffer, and confirming an accept asks this function for one position
    twice more -- though it is not where a sweep's time goes: on a noisy 25 s web
    capture the whole receiver runs 34 s either way, in `bursts` and `bodies`.
    """
    sps = fs // 100
    pulse = _pulse(sps); delay = (len(pulse) - 1) // 2
    if Z is None:
        Z = {cn: _baseband(audio, cn, fs, pulse) for cn in HEADER_TONES}
    if search is None:
        search = range(0, len(audio) - 74 * sps, sps // 8)
    for start in search:
        code = case0_softs(Z, start, fs=fs, delay=delay)
        if code is None:
            break
        field, ok = decode_case0_softs(code)
        if ok:
            yield start, field


def case0_softs(Z: dict, start: int, *, fs: int = FS_DEFAULT, delay: int = 0,
                order: tuple[int, int] = p3frame.VH_ORDER,
                rot: float = 0.0,
                path: placement.Path = placement.DETECT,
                lead: dict[int, int] | None = None) -> np.ndarray | None:
    """One case-0 frame's code-order softs, its pilot at `start`.

    None where the frame runs off the end of the baseband.

    `order` is the two virtual carriers as THIS cycle carries them, cell 0 first.
    The swap alternates on every ARQ cycle, so a receiver holding the home order
    gathers both carriers into each other's cells on alternate cycles: a rendered
    swapped speed-level-1 packet decodes at no alignment until this argument
    carries the swap, and a real link would lose every second packet.

    `rot` is the residual carrier offset the packet's own header block measured,
    in radians of differential phase -- `cells_from_baseband`'s argument, on the
    axis case 0 slices against.

    `path` is the frame's geometry: 72 rows for a speed-level-1 packet and 56 for
    `placement.CHANGEOVER`, which is the same construction shortened by what its
    CS3 head costs.

    `lead` is each CHANNEL's symbol clock in samples past `start`, which case 0
    needs for the reason `cells_from_baseband` takes `Path.clock_offsets`: this
    comb is split half a symbol (`spec.SUBBAND_LEAD`). Read on one clock the
    frame's optimum sits at the midpoint of the two rather than at either, so a
    decoder anchored on the packet's own header block -- which reads the leading
    carrier -- missed grid row 0 by an eighth of a symbol and decoded nothing at
    the one alignment an anchored scan offers. A caller with no swap to go on
    passes nothing and keeps the single clock.
    """
    sps = fs // 100
    n_rows = path.n_buf // len(order)
    axis = np.exp(-1j * (BIT_PHASE_CASE0 + rot))
    chan = np.zeros(path.n_buf)
    idx = start + np.arange(n_rows + 1) * sps + delay
    for rank, cn in enumerate(order):
        j = idx if lead is None else idx + lead.get(cn, 0)
        if j[0] < 0 or j[-1] >= Z[cn].size:
            return None
        y = Z[cn][j]
        d = y[1:] * np.conj(y[:-1])
        chan[rank::len(order)] = (d * axis).real
    return chan[placement.case0_map(path)]


def decode_case0_softs(softs: np.ndarray, path: placement.Path = placement.DETECT
                       ) -> tuple[bytes, bool]:
    """Code-order case-0 softs -> (field, crc_ok).

    `decode_frame_softs`' opposite number: the K=9 trellis with the flush, bits
    unpacked LSB-first, de-whitened, CRC-16/X25 over the field less its last two
    bytes against the little-endian pair behind them.
    """
    n = path.crc_bytes
    bits = coding.viterbi_decode(softs, CASE0_CODE, terminated=True)
    info = coding.whiten(bytes(coding.bits_to_bytes(bits, msb_first=False)))
    if len(info) < n:
        return b"", False
    return info[:n], coding.crc16(info[:n - 2]) == int.from_bytes(info[n - 2:n],
                                                                  "little")


def decode_case0_header(audio: np.ndarray, *, fs: int = FS_DEFAULT,
                        search: range | None = None) -> tuple[bytes, bool]:
    """The first case-0 field `search` validates, or (b"", False).

    Timing is found by the CRC itself, which on clean audio is a hard 16-bit
    accept/reject and needs no separate acquisition. On a channel it is not:
    a long enough scan finds a passing CRC in anything, which is why every
    caller that has to decide whether a signal was there goes through
    `case0_accepts` and `p3rx.confirmed` instead.
    """
    for _, field in case0_accepts(audio, fs=fs, search=search):
        return field, True
    return b"", False


def decode_frame_softs(softs: np.ndarray, path=placement.HEADER) -> tuple[bytes, bool]:
    """Code-order soft coded bits (+ => bit 0) -> (field, crc_ok) for any path.

    The inverse of `placement.encode_frame`, and every stage of it is a convention
    an independent decoder was shown to accept rather than one assumed here: K=7
    (0o171, 0o133) with the trellis flushed, bits unpacked LSB-first, de-whitened,
    then CRC-16/X25 over the field against the little-endian pair that follows.
    Only the length differs between paths.
    """
    softs = np.asarray(softs, float)
    n = path.crc_bytes
    softs = path.puncture.depuncture(softs, path.n_pairs)
    bits = coding.viterbi_decode(softs, HEADER_CODE, terminated=True)
    if bits.size < 8 * n:
        return b"", False
    by = coding.whiten(bytes(coding.bits_to_bytes(bits[:8 * n], msb_first=False)))
    crc_ok = coding.crc16(by[:n - 2]) == (by[n - 2] | (by[n - 1] << 8))
    return by, crc_ok


def decode_header_softs(softs: np.ndarray) -> tuple[bytes, bool]:
    """The case-1 header, the one path most callers want."""
    return decode_frame_softs(softs, placement.HEADER)


def demod_cells(audio: np.ndarray, start: int, path=placement.HEADER, *,
                fs: int = FS_DEFAULT, n_rows: int = 72, lag: int = 1,
                rot: float = 0.0) -> np.ndarray:
    """Audio -> the CHANNEL-order softs of one frame, on any path.

    The grid is N tones laid down row-major, so cell (row, rank) occupies channel
    `(row * N + rank) * bits_per_cell`. `start` is the sample index of grid row 0,
    and each cell's step is measured from the symbol `lag` before it. The softs
    are the phase-only quantities a reference decoder forms, positive meaning bit
    0: one per cell off sin(dphi - 45 deg) for the DBPSK paths, two off sin and
    -cos for the DQPSK ones. Case 0 inverts the DBPSK sign rule and has its own
    decoder.
    """
    sps = int(fs) // 100
    pulse = _pulse(sps)
    delay = (len(pulse) - 1) // 2
    Z = {cn: _baseband(audio, cn, fs, pulse) for cn in path.tones}
    return cells_from_baseband(Z, start, path, fs=fs, delay=delay,
                               n_rows=n_rows, lag=lag, rot=rot)


def cells_from_baseband(Z: dict, start: int, path=placement.HEADER, *,
                        fs: int = FS_DEFAULT, delay: int = 0,
                        n_rows: int = 72, lag: int = 1,
                        rot: float = 0.0) -> np.ndarray:
    """`demod_cells` over ALREADY-FILTERED tones, so a timing search is not O(N^2).

    The per-tone baseband filter runs over the whole clip, and it does not depend
    on `start` -- but it sat inside the loop, so a CRC timing scan refiltered the
    entire window once per candidate position. Hoisting it turns a scan of a 3.2 s
    window from 4.6 s into a fraction of that, which is what makes a full scan
    affordable to a caller that knows a frame is due.

    `start` is grid row 0 of the EARLIEST carrier, which on every level but 2 is
    all of them; where a level staggers its comb, each carrier is read on its own
    clock (`Path.clock_offsets`). Reading a staggered level on one clock costs the
    misread half its margin and not its decode: measured on noiseless audio, the
    soft magnitudes there fall from 0.99 to 0.70 and the CRC still passes, so the
    price is paid on a channel rather than on the bench.

    `rot` is where the packet's own header block says the constellation sits --
    the residual carrier offset, in radians of differential phase. A receiver
    that ignores it slices the field on axes the signal is not using, and there
    is nothing gradual about the price: a station 39 deg off is six degrees
    inside a DQPSK decision boundary and reads at a 4% bit error rate on a
    channel whose phase noise is 12 deg RMS.
    """
    sps = int(fs) // 100
    tones = path.tones
    bpc = path.bits_per_cell
    offsets = path.clock_offsets(sps)
    out = np.zeros(n_rows * len(tones) * bpc)
    for rank, cn in enumerate(tones):
        z = Z[cn]
        idx = start + offsets[rank] + (np.arange(n_rows + lag) - lag) * sps + delay
        if idx[0] < 0 or idx[-1] >= z.size:
            raise ValueError(f"{path.name} grid runs past the end of the audio")
        y = z[idx]
        dphi = np.angle(y[lag:] * np.conj(y[:-lag])) - rot
        base = rank * bpc
        if bpc == 1:
            out[base::len(tones)] = -np.sin(dphi - 0.25 * np.pi)
        else:
            step = len(tones) * bpc
            out[base::step] = np.sin(dphi)
            out[base + 1::step] = -np.cos(dphi)
    return out


def decode_frame(audio: np.ndarray, path=placement.HEADER, *,
                 fs: int = FS_DEFAULT,
                 search: range | None = None) -> tuple[bytes, bool, int]:
    """One frame from audio: grid -> de-interleave -> K=7 -> CRC, on any path.

    Timing is found by the CRC, which is a hard 16-bit gate; `search` is the range
    of candidate row-0 sample indices (default: every symbol of a short cycle).
    Returns (field, crc_ok, start).
    """
    sps = int(fs) // 100
    if search is None:
        search = range(sps, len(audio) - 74 * sps, sps)
    pulse = _pulse(sps)
    delay = (len(pulse) - 1) // 2
    Z = {cn: _baseband(audio, cn, fs, pulse) for cn in path.tones}
    for start in search:
        try:
            chan = cells_from_baseband(Z, start, path, fs=fs, delay=delay)
        except ValueError:
            break
        field, ok = decode_frame_softs(placement.deinterleave(chan, path), path)
        if ok:
            return field, True, start
    return b"", False, -1


def decode_case1_header(audio: np.ndarray, *, fs: int = FS_DEFAULT,
                        search: range | None = None) -> tuple[bytes, bool, int]:
    """The SL>=2 header specifically. Returns (26-byte field, crc_ok, start)."""
    return decode_frame(audio, placement.HEADER, fs=fs, search=search)


def demod_case1(audio: np.ndarray, start: int, *, fs: int = FS_DEFAULT,
                n_rows: int = 72, lag: int = 1) -> np.ndarray:
    """The case-1 grid specifically. Returns its 432 channel-order softs."""
    return demod_cells(audio, start, placement.HEADER, fs=fs, n_rows=n_rows, lag=lag)


# ---------------------------------------------------------------------------
# Front end: audio -> timing, per-tone symbols, soft raster
# ---------------------------------------------------------------------------

def _pulse(sps: int) -> np.ndarray:
    return resample_poly(tablegen.symbol_pulse(), sps // 8, 1)


def carrier(cn: int, fs: int, start: int, count: int) -> np.ndarray:
    """The mixing exponential for tone `cn`, `count` samples from absolute `start`.

    TABULATED, not evaluated per sample. Every channel sits on the 120 Hz grid
    and every sample rate this runs at is a whole number of those, so the
    exponential repeats exactly every `fs / spec.TONE_SPACING_HZ` samples -- 400
    at 48 kHz -- and one period tiled from the right phase IS the sequence a
    per-sample `np.exp` computes. Measured over the eighteen channels of a 3.8 s
    window, 4 ms against 36 -- five sixths of what a tracked decode spends
    getting to baseband, and through `onair._SessionRx.deep_scan` the rest of
    the way from 23.4 ms to 16.3 at speed level 6 on the short cycle and 67.5
    to 45.3 on the long one.

    The table is the more accurate of the two rather than a shortcut: `np.exp`
    is handed an argument reaching 1.2e6 radians by the end of that window and
    pays for the range reduction, while a tile never leaves the first period.
    They agree to 4e-12 on a signal whose own amplitude is 0.1.

    A sample rate the tone grid does not divide has no period to tile, and the
    exponential is evaluated where it is asked.

    THE BLIND PATH TAKES IT TOO, now that a straddling alignment cannot be read.
    It could not before: `rxfront._best_cs` swept 22 symbols in front of the
    burst and took the first alignment of least Hamming distance, so half of what
    it read had ten of the codeword's twenty symbols off the end of the burst,
    and a twenty-bit word with no CRC out of an alphabet of six lands on one. A
    rendered CYCLE-TOG read BREAK-IN at zero bit errors 100 ms in front of its
    own first sample, and which of the two the sweep met first was decided by
    float noise -- 4e-12 of it was enough to swap them. That was never a property
    of the table; the table is what measured it. `rxfront.CS_EXTENT_FLOOR` is the
    repair, and it is the burst's extent: an alignment is read only if its 21
    symbols lie inside the burst the envelope describes.
    """
    period = fs / spec.TONE_SPACING_HZ
    n = start + np.arange(count)
    if period != int(period):
        return np.exp(-2j * np.pi * spec.channel_freq_hz(cn) * n / fs)
    period = int(period)
    turn = np.exp(-2j * np.pi * spec.channel_freq_hz(cn)
                  * np.arange(period) / fs)
    return np.resize(np.roll(turn, -start % period), count)


def _baseband(audio: np.ndarray, cn: int, fs: int, pulse: np.ndarray) -> np.ndarray:
    """Matched-filtered complex baseband for tone `cn`.

    Overlap-add rather than `np.convolve`: the matched filter is 1860 taps, so the
    direct form costs O(len(audio) * 1860) and was 55% of the whole receiver on a
    33 s capture. Same 'full' length and indexing, agreeing to 1e-15 relative.

    On the TABULATED carrier, which this path could not take while a straddling
    alignment was readable. Through `onair._SessionRx.deep_scan`, blind scan,
    short cycle: 72.5 ms to 63.4 at speed level 1 and 85.2 to 75.9 at level 6;
    long cycle 211.1 to 183.4 at level 2 and 267.4 to 240.2 at level 6. On
    `watch_pactor3_maryland.wav` through the stream receiver, 262.8 ms to 243.9
    over a 2.00 s window -- 0.131 of real time to 0.122.
    """
    bb = audio * carrier(cn, fs, 0, len(audio))
    return oaconvolve(bb, pulse)


def pilot_corr(Z: dict, starts: np.ndarray, delay: int, dA: np.ndarray) -> np.ndarray:
    """PATTERN_A pilot correlation over the header tones, one score per start.

    The acquisition metric, shared by `acquire` and the monitor's packet scan.
    Both sweep every eighth of a symbol across the whole capture, so `starts` is an
    array and the eight pilot taps are gathered for all of them at once; a start
    whose window runs past the end scores 0.
    """
    starts = np.atleast_1d(np.asarray(starts, dtype=np.int64))
    idx = starts[:, None] + np.arange(8) * SPS_DEFAULT + delay
    ok = idx[:, -1] < min(Z[cn].size for cn in HEADER_TONES)
    idx = np.where(ok[:, None], idx, 0)
    score = np.zeros(starts.size)
    for cn in HEADER_TONES:
        y = Z[cn][idx]
        d = y[:, 1:] * np.conj(y[:, :-1])
        d /= np.abs(d) + 1e-9
        score += np.abs((d * np.conj(dA)).sum(axis=1))
    return np.where(ok, score, 0.0)


def acquire(audio: np.ndarray, fs: int = FS_DEFAULT,
            search: tuple[float, float] = (0.0, None)) -> int:
    """Locate the packet body: correlate the fixed 8-symbol pilot PATTERN_A (sent
    absolute on the header tones at block 0, seed = 1) against the lag-1
    differential of the header tones. Returns the sample index of pilot symbol 0.
    """
    sps = fs // 100
    delay = (len(_pulse(sps)) - 1) // 2
    Z = {cn: _baseband(audio, cn, fs, _pulse(sps)) for cn in HEADER_TONES}
    dA = p3frame.PATTERN_A[1:] * np.conj(p3frame.PATTERN_A[:-1])
    dA = dA / np.abs(dA)
    lo = int(search[0] * fs)
    hi = int((search[1] if search[1] else len(audio) / fs) * fs) - 9 * sps
    starts = np.arange(max(lo, 0), hi, sps // 8)
    if starts.size == 0:
        return lo
    return int(starts[np.argmax(pilot_corr(Z, starts, delay, dA))])


def demod_raster(audio: np.ndarray, start: int, *, fs: int = FS_DEFAULT,
                 lag: int = 8, n_sym: int = 72) -> np.ndarray:
    """Best-effort reproduction of a reference decoder's header soft raster.

    Per tone in {5,12} and per symbol: phi = atan2(I, -Q) of the matched-filter
    sample, differential over `lag` (default 8 -- a real receiver's symbol history
    rotates mod 8, so each symbol is taken against the one eight back, not the one
    before it), then the phase-only soft pair
    (Isoft, Qsoft) = (127 sin dphi, 127 cos dphi). Returns a flat buffer whose
    RASTER_BLOCKS slots hold the four 72-symbol streams, matching the layout the
    `gather` map indexes.

    NB: a real receiver forms its per-symbol complex value through a down-convert
    and decimation followed by a biquad + FIR; this matched filter reduces the tone
    straight instead, so it only partially tracks one (see `raster_correlation`)
    and does not yet drive a full audio->CRC decode.
    """
    sps = fs // 100
    pulse = _pulse(sps)
    delay = (len(pulse) - 1) // 2
    raster = np.zeros(max(RASTER_BLOCKS) + n_sym, np.int64)
    for ti, cn in enumerate(HEADER_TONES):
        z = _baseband(audio, cn, fs, pulse)
        idx = start + np.arange(n_sym + lag) * sps + delay
        if idx[-1] >= z.size:
            # A frame shorter than the requested raster (a short cycle, or a
            # capture that ends mid-packet) is an ordinary input, not an error:
            # take the symbols that exist and leave the rest of the block zero,
            # which the soft metric already reads as "no information".
            idx = idx[idx < z.size]
            if idx.size <= lag:
                continue
        s = z[idx]
        phi = np.arctan2(s.real, -s.imag)
        dphi = phi[lag:] - phi[:-lag]
        isoft = np.clip(np.round(127 * np.sin(dphi)), -127, 127)
        qsoft = np.clip(np.round(127 * np.cos(dphi)), -127, 127)
        oi, oq = RASTER_BLOCKS[2 * ti], RASTER_BLOCKS[2 * ti + 1]
        n = min(n_sym, isoft.size)
        raster[oi:oi + n] = isoft[:n]
        raster[oq:oq + n] = qsoft[:n]
    return raster


def soft_accumulate(rasters: list[np.ndarray]) -> np.ndarray:
    """Memory-ARQ soft combine across identical-header retransmit cycles.

    A real decoder merges each cycle's soft raster into a running accumulator,
    magnitude-weighted and clipped to int8; the additive core is captured here.
    Accumulation is what lifts a marginal header over the Viterbi threshold (occ15
    needed three passes) -- but it only averages DOWN the *random* part of the
    per-cycle error. A demod with a *systematic* sign bias stays biased however
    many cycles are summed, so accumulation cannot rescue a low-fidelity front end
    (see tests/shrike/test_rx.py: the numeric demod sits at ~0.6 sign agreement to
    the CRC-valid raster while the FEC needs ~0.98).
    """
    acc = np.zeros_like(np.asarray(rasters[0], np.float64))
    for r in rasters:
        acc += np.asarray(r, np.float64)
    return np.clip(np.round(acc / len(rasters)), -127, 127).astype(np.int64)


def raster_correlation(raster: np.ndarray, reference: np.ndarray) -> float:
    """Mean |normalised correlation| of the four raster blocks against a reference
    raster (e.g. the CRC-valid `vhinF.bin` capture). 1.0 == bit-exact front end."""
    cs = []
    for o in RASTER_BLOCKS:
        a = raster[o:o + 72].astype(float)
        b = reference[o:o + 72].astype(float)
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        cs.append(abs(float(np.dot(a, b) / d)) if d else 0.0)
    return float(np.mean(cs))


# ---------------------------------------------------------------------------
# Control signals: DBPSK on tones 5 & 12, nearest-codeword
# ---------------------------------------------------------------------------

def _cs_bits(word: int) -> np.ndarray:
    return np.array([(word >> i) & 1 for i in range(spec.CS_BITS_PER_TONE)], np.uint8)


_CS_TABLE = np.array([_cs_bits(w) for w in spec.CONTROL_SIGNALS], np.uint8)


def nearest_control_signal(bits: np.ndarray) -> tuple[int, int]:
    """Nearest of the six 20-bit CS codewords. Returns (index, Hamming distance).

    The set is a distance-12 code, so up to 5 bit errors are corrected.
    """
    bits = np.asarray(bits, np.uint8)[:spec.CS_BITS_PER_TONE]
    d = (_CS_TABLE != bits).sum(1)
    i = int(np.argmin(d))
    return i, int(d[i])


def cs_bits(Z: dict, start: int, delay: int, sps: int = SPS_DEFAULT):
    """The 20 DBPSK bits of a control signal at `start`, from precomputed tone
    baseband -- the shared kernel behind `decode_control_signal` and the monitor's
    CS scan. Both header tones carry the same bits and are summed before slicing.
    Returns None when the 21 symbols run past the end of the baseband."""
    acc = np.zeros(spec.CS_BITS_PER_TONE)
    for cn in HEADER_TONES:
        idx = start + np.arange(spec.CS_BITS_PER_TONE + 1) * sps + delay
        if idx[-1] >= len(Z[cn]):
            return None
        y = Z[cn][idx]
        acc += (y[1:] * np.conj(y[:-1])).real        # DBPSK: Re>0 => bit 0
    return (acc < 0).astype(np.uint8)


def decode_control_signal(audio: np.ndarray, start: int, *,
                          fs: int = FS_DEFAULT) -> tuple[int, int]:
    """Decode one control signal starting at `start` (phase-reference symbol).

    Each CS is 20 DBPSK symbols on tones 5 and 12 carrying the same 20 bits;
    both tones are combined before the nearest-codeword decision.
    """
    sps = fs // 100
    pulse = _pulse(sps)
    delay = (len(pulse) - 1) // 2
    Z = {cn: _baseband(audio, cn, fs, pulse) for cn in HEADER_TONES}
    bits = cs_bits(Z, start, delay, sps)
    if bits is None:
        raise ValueError("control signal runs past end of audio")
    return nearest_control_signal(bits)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

@dataclass
class Decode:
    sl: int | None
    status: int | None
    header_bytes: bytes
    crc_ok: bool
    cs: tuple[int, int] | None


def decode_from_softs(softs432: np.ndarray) -> Decode:
    """432 code-order soft coded bits -> decoded SL>=2 header."""
    header, crc_ok = decode_header_softs(np.asarray(softs432, float))
    status = header[0] if crc_ok else None
    return Decode(sl=(status >> 2) & 3 if crc_ok else None,
                  status=status, header_bytes=header, crc_ok=crc_ok, cs=None)


def decode(audio: np.ndarray, *, fs: int = FS_DEFAULT) -> Decode:
    """Full audio decode of an SL>=2 header packet."""
    header, crc_ok, _ = decode_case1_header(audio, fs=fs)
    status = header[0] if crc_ok else None
    return Decode(sl=(status >> 2) & 3 if crc_ok else None,
                  status=status, header_bytes=header, crc_ok=crc_ok, cs=None)
