# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Open-loop BW500 receiver — records 3, 2, 1 and 0, built from spec/03 §3.6.

Decodes real captured transmit audio to CRC-passing frames and the exact
transmitted payload bytes. Round-trip-validated (spec/03 §3.6) cross-payload
byte-exact: counter / PRBS15 / cryptographic-random.

Record 3 (host ``BITRATE (4)``, off=3 SHORT) is the base level and the one every
BW500 link-setup rides. Records 2, 1 and 0 (host ``BITRATE (3)``, ``(2)``,
``(1)``) are the gear-down ladder below it, a different waveform family — see
"The index ladder" below. ``decode_burst_frames`` tells them apart before it
decodes any.

Base-level decode law (all constants fixed / payload-independent; no payload
knowledge anywhere in the path). Section references are to spec/03 §3.6:

    1. down-convert each sub-band (0 -> 1350 Hz, 1 -> 1650 Hz), H = 512 samples
       per symbol, 120 Hz low-pass, single-sample-per-symbol (WOLA Nyquist pulse). §3.6.1
    2. de-rotate by the fixed grid480 reference at index  (§3.6.2)
           471*sub + (462+col) % 471      (sub 1)
                     (460+col) % 469       (sub 0)
    3. alignment (grid epoch and sub-symbol phase together): scan the sample
       offset of emission column 0 over [-9, +2) columns of the detected burst
       start, maximise the differential collapse |E[(v_k conj v_{k-1})^2/|.|^2]|
       — payload-blind, and immune to a carrier offset (:func:`_lock_alignment`).
    4. burst detect: 1150-1850 Hz envelope; ~4.3 s bursts are data/link-setup frames.
       The link-setup frame (decoded body starts 04 10 41) is dropped.
    5. per data frame, frame-start column c0: scan 0..71, maximise turbo
       self-consistency (encode(sys)==coded) — payload-blind; locks c0 = 9. §3.6.2
    6. differential-BPSK across columns over the data cells  (§3.6.1/§3.6.2/§3.6.3)
           {(s,co): co in 1..392, USED[17*(128*co+s)+3] != 1}
       -> coded bits placed at P = IDX1[3::17][:748] (Stage-1 de-interleave)
       -> turbo (13,15) log-MAP BCJR (N=368) -> XOR pre-FEC PN whitener
       -> CRC-16/GENIBUS over frame[:44] vs frame[44:46].  §3.6.4
    7. frame carries 43 payload bytes (the final frame of a message is short).

One burst, up to two frames: a burst of 796 columns packs a second 394-column
frame behind the first (403-column bursts carry one). The frames are decoded by
the same law with one running advance per frame index ``fk`` (0-based within the
burst): the grid480 reference is read one entry later
(``grid480_index(sub, n) + fk``) and the Stage-1 placement one cell later
(``P[(cell + fk) % 748]``), the differential/turbo/whitener/CRC unchanged. Frame
0 is the classic SHORT frame; ``fk`` is the only difference for frame 1. A burst
truncated before its later frames still reports them via
``DecodeResult.frames_skipped``, which is why completeness is
``DecodeResult.complete`` and not the frame CRCs.

The index ladder — records 2, 1 and 0 (host ``BITRATE (3)`` 61 bps, ``(2)`` 41
bps, ``(1)`` 18 bps) — is not this waveform in lower gears: it is one-hot index
modulation, the same family as BW2300 records 3..0. A burst is a few training
columns and then the record's frame columns, each lighting exactly one bin of
the column's own DFT — data columns carrying three bits apiece at ``alloc +
stride*Gray(value) (mod span)``, and 24 reference columns lighting ``alloc +
stride*map2`` whatever the payload (:data:`INDEX_RECORDS`):

    record 2: 4 training + 226 columns of 1024 samples, bins 27..37, stride 1,
              202 data columns -> 606 coded bits, turbo rate 1/2 (N=297),
              34 payload bytes;
    record 1: 4 training + 228 columns of 1024 samples, the same comb,
              204 data columns -> 612 coded bits, turbo rate 1/3 (N=200),
              22 payload bytes;
    record 0: 2 training + 124 columns of 2048 samples, bins 54..74 (the same
              band at half the spacing), stride 2 modulo 21, 100 data columns
              -> 300 coded bits, turbo rate 1/3 (N=96), 9 payload bytes.

So the decode, per record, is:

    1. block grid: the column phase at which the columns read as a single lit
       bin (:func:`_l3_grid`) — payload-blind.
    2. frame start: the lead, of 0..8, at which the 24 reference columns light
       the bins ``alloc``/``map2`` predict (:func:`_l3_align`) — payload-blind,
       and the guard that says the burst is this record at all. It locks at the
       record's training count, and it is what tells the records apart: the
       right one scores 24 of 24, a wrong one no more than 8 (99 corpus bursts).
    3. per data column, joint soft bits over the band reading (:func:`_l3_decode`)
       -> coded bits at IDX1[rec::17][:coded] -> turbo (13,15) at the record's
       rate with its own il2 column -> XOR PN whitener
       -> CRC-16/GENIBUS over the frame ahead of its last two bytes.
    4. the frame is payload + marker + CRC.

One burst, N frames, at these records too: a stock sender packs a second frame
behind the first once a low-level delivery has had a run of successes, and the
frames butt up on the record's own column grid — no gap, no training of their
own, the same base bins and reference layout from the first column of each.
Measured on the two 1002-column level-1 overs of 2026-09-11 (stock 4.9.0, BW500
cables), whose second frames open at column 2 + 124, score 24 of 24 reference
columns there and read byte-exact (:func:`_l3_frames`). The frame count is the
burst's measured length on that grid, rounded, as at the base level.

The ``_l3_`` functions are named for the level they were first read at, host
level 3; ``level`` picks the record and defaults to it. The training columns
are not reversed. They are training the peer cannot check — a receiver has no
way to know the transmitter's Rnd stream position — so `tx/varahf500_tx` keys
these records with a stand-in and the body is exact.

The de-rotation grid, Stage-1 permutation, cell-usage map and PN whitener are
fixed session-independent tables carrying no payload data. They are regenerated
by :mod:`hfmodem.kestrel.rx.tablegen`; the
one genuinely measured input, ``grid480``'s 786 recovered lattice integers, is
written out in that module as literals, because nothing derives a measurement.
Records 2 and 1 need no measured table at all: their base bins and 24 reference
classes are the draws of their own interleaver gaps, which land exactly on the
next record's seed, and their reference columns are a closed form. Record 0's
124 base bins are measured (:func:`tablegen.bw500_alloc`).
The turbo codec is ``hfmodem.kestrel.coding.turbo``.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import cache, partial

import numpy as np
from scipy.signal import fftconvolve, firwin

from . import tablegen
from ..coding import turbo as _turbo
from ..coding.crc import crc16_genibus

# --------------------------------------------------------------------------- #
# fixed parameters (VARA HF 500, Level 4, off=3 SHORT frame)
FS = 48000
H = 512                       # samples per symbol
SUBBANDS = ((0, 1350.0), (1, 1650.0))
OFF = 3
CODED = 748                   # coded cells per frame
NSYM = 394                    # columns per frame
NCARR = 2
N_SRC = 368                   # turbo input bits
FRAME_PAYLOAD = 43            # payload bytes carried per frame
_FRAME_COLS = 470             # per-frame demod window (a frame + the c0 headroom)
_NCOL_DEMOD = NSYM + _FRAME_COLS   # 864: reaches a second frame's window in full
_C0_MAX = 72
_PREAMBLE = 9                 # columns ahead of the first frame (the c0 lock)
# A link-setup is told from a DATA over by its fixed structure bytes  [spec 04
# §4.2A], never by its leading bytes — those hold the 6-bit packed CALLER
# callsign. This test read "041041", which is not a signature at all but the
# packing of a call beginning "AAAA": every BW500 recording held until
# 2026-08-15 was the same AAAA1 -> BBBB2 loopback pair. Bench captures of five
# other callsigns that day put the structure bytes at these three offsets in all
# five, and the same three carry the BW2300 frame's  [spec 04 §4.2A].
_SETUP_STRUCTURE = ((7, 0x80), (8, 0x14), (42, 0x04))
_PAD = 1500                   # guard samples kept either side of the burst
#: Columns the alignment search reaches ahead of the detected burst start and
#: behind it. Ahead is bounded by the preamble: a burst that lost more than its
#: nine lead columns has lost data cells too. Behind covers the envelope edge
#: preceding the grid by a column (the 2026-09-11 stock terminal window).
_ALIGN_EARLY, _ALIGN_LATE = _PREAMBLE, 2
_LEAD = _PAD + _ALIGN_EARLY * H   # samples kept ahead of the burst start
#: The band a burst is detected in and the shortest run that is one: ~3 s, where a
#: burst holding one frame is ~4.3 s and one holding two is ~8.5 s.
BURST_BAND = (1150.0, 1850.0)
BURST_MIN = 150000
_SEGLEN = (_NCOL_DEMOD + 8) * H + _PAD + _LEAD    # samples a burst is demodulated over

_LP = firwin(801, 120 / (FS / 2))             # 120 Hz sub-band low-pass
# down-conversion table, one row per sub-band: a burst is always demodulated
# over the same sample span, so the two mixers are constants, not per-call work
_MIX = np.exp(-2j * np.pi * np.array([f0 for _, f0 in SUBBANDS])[:, None]
              * np.arange(_SEGLEN) / FS)


def _load():
    # spec/tables/bw500 constant data facts, regenerated (see tablegen). grid480
    # is the de-rotation reference (measured, 786 recovered cells); the rest are
    # computed from the VB6 Rnd stream or in closed form.
    grid480 = tablegen.grid480()
    idx1 = tablegen.interleaver("bw500", 1)
    used = tablegen.usedmap_plane()
    pn = (tablegen.whitener() & 1).astype(int)
    return grid480, idx1, used, pn


_GRID480, _IDX1, _USED, _PN = _load()
_CELLS = [(s, co) for s in range(NCARR) for co in range(NSYM - 1)
          if co != 0 and _USED[17 * (128 * co + s) + OFF] != 1]
_P = _IDX1[OFF::17][:CODED]


def grid480_index(sub: int, ncol: int) -> np.ndarray:
    """Fixed grid480 read index for a sub-band, per emission column 0..ncol-1."""
    if sub == 1:
        return np.array([471 + ((462 + co) % 471) for co in range(ncol)])
    return np.array([(460 + co) % 469 for co in range(ncol)])


# --------------------------------------------------------------------------- #
# fixed parameters (VARA HF 500, the index ladder: records 2, 1, 0)
@dataclass(frozen=True)
class IndexRecord:
    """One one-hot index-law record: its physical layout and coding lengths."""
    level: int          # host BITRATE number
    rec: int            # record index: the interleaver column, and the table key
    dw: int             # samples per emission column
    first_bin: int      # low bin of the comb in the column's own DFT
    span: int           # bins in the comb
    stride: int         # bins per Gray step
    ncols: int          # frame columns, data + 24 reference
    lead: int           # training columns a burst opens with
    n_src: int          # turbo input bits
    coded: int          # on-air bits: 2*n_src+12 at rate 1/2, 3*n_src+12 at 1/3
    payload: int        # payload bytes ahead of the marker and CRC

    @property
    def frame(self) -> int:
        return self.payload + 3

    @property
    def rate13(self) -> bool:
        return self.coded == 3 * self.n_src + 12


#: The ladder by host level. Record 2's numbers are the ``L3_*`` constants below.
INDEX_RECORDS = {3: IndexRecord(3, 2, 1024, 27, 11, 1, 226, 4, 297, 606, 34),
                 2: IndexRecord(2, 1, 1024, 27, 11, 1, 228, 4, 200, 612, 22),
                 1: IndexRecord(1, 0, 2048, 54, 21, 2, 124, 2, 96, 300, 9)}
_REC2 = INDEX_RECORDS[3]
L3_DW = _REC2.dw              # samples per emission column
L3_NCOL = _REC2.ncols         # emission columns a frame occupies
L3_LEAD = _REC2.lead          # training columns a burst opens with
L3_FIRST_BIN, L3_SPAN = _REC2.first_bin, _REC2.span
L3_BPC = 3                    # bits a data column carries, at every record
L3_CODED, L3_N_SRC = _REC2.coded, _REC2.n_src
L3_PAYLOAD = _REC2.payload    # payload bytes carried per frame
L3_FRAME = _REC2.frame        # payload + marker + 2-byte CRC
#: Reference columns an alignment has to match before a burst is read as an index
#: record. Every recorded index over scores 24 of 24 at one lead of its own record
#: and no more than 7 at any other lead or record; over 99 corpus bursts, one- and
#: two-frame record-3 included, no wrong record scores above 8.
_L3_GUARD_MIN = 12
_L3_LEAD_MAX = 8              # training columns the lead search covers
_L3_GRID_BLOCKS = 32          # columns the block-grid search scores over
_L3_BIN_FLOOR = 1e-3          # per-bin equalisation floor, as a share of the band peak
_L3_LLR_CAP = 8.0             # the most any one channel bit may claim

#: BW500 speed levels, in the host's own ``BITRATE`` numbering: 4 is record 3's
#: sub-band DBPSK, 3 is record 2's one-hot index law one gear below it, and 2 and
#: 1 are records 1 and 0 below that (:data:`INDEX_RECORDS`).
BASE_LEVEL, ROBUST_LEVEL = 4, 3

#: Payload bytes each level's frame carries ahead of its marker and CRC.
_PAYLOAD = {BASE_LEVEL: FRAME_PAYLOAD, **{lv: r.payload for lv, r in INDEX_RECORDS.items()}}


@cache
def _l3_tables(level: int = ROBUST_LEVEL):
    """``(ref_col, ref_bin, data_col, labels, valid, place, perm)`` for a record.

    A column lights one bin of ``span`` at ``alloc + offset (mod span)``: the
    offset is ``stride*map2`` on a reference column and ``stride*Gray(value)`` of
    the column's three coded bits on a data one. Only eight strides of the comb
    are reachable, so the other bins of a data column mean nothing at all —
    ``valid`` is what says so, and it is what makes a wrong alignment visible
    before any decode.
    """
    r = INDEX_RECORDS[level]
    alloc = tablegen.bw500_alloc(r.rec)
    ref = tablegen.bw500_ref_cols(r.rec)
    is_ref = np.zeros(r.ncols, bool)
    is_ref[ref] = True
    data = np.flatnonzero(~is_ref)
    ref_bin = (((alloc[ref] + r.stride * tablegen.bw500_map2(r.rec) - r.first_bin)
                % r.span) + r.first_bin)
    off = (np.arange(r.span)[None, :] - (alloc[data] - r.first_bin)[:, None]) % r.span
    npts = 2 ** L3_BPC
    valid = (off % r.stride == 0) & (off // r.stride < npts)
    row = tablegen.clsparm_gray()[(L3_BPC - 2) * 16:(L3_BPC - 2) * 16 + npts]
    val = np.argsort(row)[np.where(valid, off // r.stride, 0)]
    labels = (val[:, :, None] >> np.arange(L3_BPC - 1, -1, -1)) & 1
    perm = tablegen.interleaver("bw500", 2)[r.rec::17][:r.n_src]
    return ref, ref_bin, data, labels, valid, _IDX1[r.rec::17][:r.coded], perm


# --------------------------------------------------------------------------- #
def frames_carried(ncol: int) -> int:
    """Whole frames a burst of ``ncol`` columns holds — preamble + n * NSYM.

    Measured over the corpus, a data burst falls in one of two tight classes:
    403 columns (4.31 s) and 796 (8.50 s), against 9 + 394*n = 403 and 797.
    Nothing longer was seen, and the 462-, 466- and 506-column classes are the
    index ladder's records 2, 1 and 0, which :func:`decode_burst_frames`
    recognises before it asks this question at all. Rounding rather than flooring puts the
    one/two decision ~197 columns clear of either class.

    A burst of unmeasured length (``ncol`` 0, from :func:`decode_burst` called
    without a span) carries one frame: claim nothing that was not measured.
    """
    return max(1, round((ncol - _PREAMBLE) / NSYM))


@dataclass
class FrameResult:
    start: int
    onset: tuple                 # per sub-band: sample offset of emission column 0 from start
    c0: int
    self_consistency: float
    crc_ok: bool
    frame_bytes: bytes           # by[:46]  (43 payload + marker + 2 CRC)
    is_connect: bool = False
    level: int = BASE_LEVEL      # 4 = sub-band DBPSK; 3, 2, 1 = index records 2, 1, 0
    # what the burst this frame came out of held, as the detector measured it —
    # a property of the burst rather than of the frame, and the only place the
    # decoded span can be compared against the transmitted one
    burst_columns: int = 0
    frames_in_burst: int = 1

    @property
    def payload(self) -> bytes:  # the payload bytes this frame carries
        return self.frame_bytes[:_PAYLOAD[self.level]]

    @property
    def marker(self) -> int:
        return self.frame_bytes[_PAYLOAD[self.level]]


@dataclass
class DecodeResult:
    frames: list[FrameResult]
    data_frames: list[FrameResult] = field(default_factory=list)

    @property
    def payload(self) -> bytes:
        """Concatenated 43-byte payloads of the data frames (untrimmed).

        VARA signals the total message length out of band; trim to the known
        length to drop the final frame's short-frame padding (see module docs)."""
        return b"".join(f.payload for f in self.data_frames)

    @property
    def frames_skipped(self) -> int:
        """Frames the recording carried that were never demodulated.

        Per burst, not per frame: a burst says how many frames its measured
        length holds, and the decode says how many came back out of it."""
        got = Counter(f.start for f in self.frames)
        held = {f.start: f.frames_in_burst for f in self.frames}
        return sum(max(held[st] - n, 0) for st, n in got.items())

    @property
    def complete(self) -> bool:
        """Every frame the recording carried was demodulated, and passed CRC.

        This is the question "did the transfer arrive?" and it is not the
        question the CRCs answer. `all_crc_ok` used to stand here and asked only
        whether the frames that *were* decoded passed: over a 4096-byte transfer
        where 186 of 269 bursts carried a second frame the demodulator never
        reached, it read True with 46.5% of the payload missing. A caller has to
        be able to tell a clean transfer from a partial one, and the CRCs cannot
        tell it — nothing fails a CRC that was never demodulated.
        """
        return (bool(self.data_frames) and not self.frames_skipped
                and all(f.crc_ok for f in self.data_frames))


# --------------------------------------------------------------------------- #
def burst_spans(samples: np.ndarray, min_len: int = BURST_MIN) -> list[tuple[int, int]]:
    """Payload-blind burst detection: ``BURST_BAND`` envelope, ``(start, stop)``
    samples of every run longer than ``min_len``.

    The stop edge is as much a measurement as the start one: it is what says how
    much signal a burst held, and so how much of it the decode left behind."""
    h = firwin(401, [f / (FS / 2) for f in BURST_BAND], pass_zero=False)
    env = np.abs(fftconvolve(samples, h, "same"))
    sm = fftconvolve(env, np.ones(2048) / 2048, "same")
    on = sm > 0.15 * sm.max()
    # rising/falling edges of the gate; a run still open at the end is not a burst
    edge = np.diff(on.view(np.int8), prepend=np.int8(0))
    rise, fall = np.flatnonzero(edge > 0), np.flatnonzero(edge < 0)
    return [(int(a), int(b)) for a, b in zip(rise, fall) if b - a > min_len]


def detect_bursts(samples: np.ndarray, min_len: int = BURST_MIN) -> list[int]:
    """Start sample of each burst — :func:`burst_spans` with the length dropped."""
    return [a for a, _ in burst_spans(samples, min_len)]


def _lock_alignment(bb: np.ndarray, ref: np.ndarray, lead: int) -> int:
    """Sample offset of emission column 0 from the detected burst start, in
    ``[-_ALIGN_EARLY*H, _ALIGN_LATE*H)``: the alignment whose consecutive cells,
    de-rotated by the reference, agree hardest in differential phase under
    |E[(v_k conj v_{k-1})^2 / |v_k conj v_{k-1}|^2]| — payload-blind (§3.6.3).

    The reference is a pseudo-random lattice sequence, so the score is ~0.05 at
    every alignment but the one the burst was keyed on, where it is ~1: it locks
    the grid epoch and the sub-symbol phase in one search. Squaring drops the
    data sign; the differential drops the carrier phase, and that is what makes
    it an on-air lock. The coherent |E[v^2]| this replaces wound through a full
    turn over the frame at the +0.12 Hz offset of the 2026-09-11 K5FIT greeting
    — a strong, clean level-4 frame, three preamble columns of it lost to our
    own TX->RX changeover — and scored its true onset 0.25 against noise.

    Column block k reads ``bb[lead + k*H + o + n*H]`` for n in 0..NSYM-1, so each
    block is one contiguous span of NSYM*H samples viewed as (NSYM, H) and
    transposed: row o is that onset's cell set. A block starting before the
    segment, or an onset whose last cell falls past the end of a truncated
    segment, is not a candidate.
    """
    best, best_offset = -1.0, 0
    for k in range(-_ALIGN_EARLY, _ALIGN_LATE):
        base = lead + k * H
        ncand = min(H, len(bb) - base - (NSYM - 1) * H)
        if base < 0 or ncand <= 0:
            continue
        span = bb[base: base + NSYM * H]
        if len(span) < NSYM * H:
            span = np.concatenate([span, np.zeros(NSYM * H - len(span), span.dtype)])
        v = np.ascontiguousarray(span.reshape(NSYM, H).T[:ncand]) * ref
        a = np.abs(v)
        live = a > 0.3 * np.median(a, axis=1)[:, None]
        d = v[:, 1:] * np.conj(v[:, :-1])
        m = live[:, 1:] & live[:, :-1]
        w = np.zeros_like(d)
        np.divide(d * d, np.abs(d) ** 2, out=w, where=m)
        n = m.sum(1)
        score = np.full(ncand, -1.0)
        np.divide(np.abs(w.sum(1)), n, out=score, where=n > 0)
        o = int(score.argmax())
        if score[o] > best:
            best, best_offset = float(score[o]), k * H + o
    return best_offset


def _demod_burst(samples: np.ndarray, start: int):
    """Down-convert + alignment-lock -> ``({sub: per-column complex}, onsets)``.

    Returns the raw onset-sampled columns of the whole demodulated window; the
    grid480 de-rotation is applied per frame (:func:`_frame_gh`), because a
    burst's second frame reads the reference one entry advanced.
    """
    # A capture that begins mid-burst — a bracket cut at our own un-mute — has
    # fewer than _LEAD samples ahead of `start`. Silence stands in for them: the
    # columns it holds are dead to the lock and to the cells alike, and the
    # alignment can still be read off the columns that are there.
    lead = min(start, _LEAD)
    seg = samples[start - lead: start + (_NCOL_DEMOD + 8) * H + _PAD]
    if lead < _LEAD:
        seg = np.concatenate([np.zeros(_LEAD - lead, seg.dtype), seg])
    # both sub-bands mixed to baseband and low-passed in one batched transform
    bb = fftconvolve(seg * _MIX[:, :len(seg)], _LP[None, :], "same", axes=-1)
    cols, onsets = {}, {}
    for k, (sub, _) in enumerate(SUBBANDS):
        ref = _GRID480[grid480_index(sub, NSYM)]     # the lock reads frame 0's grid
        onset = _lock_alignment(bb[k], ref, _LEAD)
        c = _LEAD + onset + np.arange(_NCOL_DEMOD) * H
        cols[sub] = bb[k][c[c < bb.shape[1]]]
        onsets[sub] = onset
    return cols, onsets


def _frame_gh(cols, fk: int):
    """De-rotate frame ``fk`` of a burst: its columns start at ``NSYM*fk`` and read
    the grid480 reference advanced by ``fk`` entries. Returns ``{sub: cell array}``
    (a fresh c0=0 origin per frame) or ``None`` if the window has no room for it."""
    gh = {}
    for sub, _ in SUBBANDS:
        col = cols[sub]
        base = NSYM * fk
        avail = min(len(col) - base, _FRAME_COLS)
        if avail <= 0:
            return None
        idx = grid480_index(sub, avail) + fk
        r = col[base: base + avail] * _GRID480[idx]
        gp = np.sign(r.real) @ r
        if gp == 0:
            return None
        gh[sub] = r * np.conj(gp / abs(gp))
    return gh


def _dewhiten(bits: np.ndarray) -> bytes:
    """Turbo output bits -> frame bytes (de-whiten, pack). Both levels' lengths."""
    return np.packbits((bits ^ _PN[:len(bits)]).astype(np.uint8)).tobytes()


def _frame_crc_ok(bits: np.ndarray) -> bool:
    """The turbo decoder's convergence test.

    Passed to the decoder as `check=` so it can stop once the frame is good
    instead of always running its full 12-iteration schedule. The CRC lives up
    here rather than in the decoder, which is why it has to be handed down.

    Not because further iterations *cannot* change a frame that checks — a
    hard-decision vector can settle and then move again, which was measured —
    but because they are wasted work on one that already does. The decoder
    guards the difference by requiring two consecutive passes over identical
    bits.
    """
    by = _dewhiten(bits)
    c = crc16_genibus(by[:44])
    return by[44:46] == bytes(((c >> 8) & 0xFF, c & 0xFF))


def _cells_llr(gh, fk: int, c0: int):
    """Differential-BPSK soft cells for frame ``fk`` at frame-start column ``c0``.

    ``(self_consistency, Lc)`` — ``(−1.0, None)`` if the cell grid runs off the end
    at this ``c0``. The Stage-1 placement is advanced ``fk`` cells in lockstep with
    the grid advance :func:`_frame_gh` applies."""
    place = _P[(np.arange(CODED) + fk) % CODED]
    try:
        dv = np.array([gh[s][c0 + co] * np.conj(gh[s][c0 + co - 1])
                       for s, co in _CELLS])
    except IndexError:
        return -1.0, None
    emit = (dv.real < 0).astype(int)
    pf = np.empty(CODED, int)
    pf[place] = emit
    sc = (_turbo.encode(pf[0:2 * N_SRC:2].astype(int)) == pf).mean()
    Lc = np.zeros(CODED)
    Lc[place] = 6.0 * dv.real / np.median(np.abs(dv.real))
    return max(sc, 1 - sc), Lc


def _decode_cells(gh, fk: int = 0,
                  c0: int | None = None) -> tuple[float, int, bool, bytes, bool]:
    """diff-BPSK -> turbo -> de-whiten -> CRC.

    ``c0`` (frame-start column) is scanned blind when ``None`` — the classic
    payload-blind lock, which settles at 9. A burst's frames are rigidly 394
    columns apart, so once frame 0 has locked its ``c0`` the later frames are
    handed that value rather than re-scanning: their own self-consistency surface
    has spurious peaks the trailing silence of a truncated window can win.
    """
    if c0 is None:
        best = (-1.0, 0, None)
        for cand in range(_C0_MAX):
            sc, Lc = _cells_llr(gh, fk, cand)
            if sc > best[0]:
                best = (sc, cand, Lc)
        sc, c0, Lc = best
    else:
        sc, Lc = _cells_llr(gh, fk, c0)
    if Lc is None:
        # Every c0 ran off the end of the cell grid: the burst is shorter than one
        # frame. Reachable from ordinary input — detect_bursts accepts runs from
        # 150000 samples and _CELLS needs column c0+392 — so a recording that ends
        # mid-burst, or a fade that splits one, used to take decode_stream down with
        # a TypeError several frames later.
        return 0.0, c0, False, b"", False
    dec = _turbo.decode_from_coded_llr(Lc, N_SRC, iters=12, check=_frame_crc_ok)
    by = _dewhiten(dec)
    crc_ok = _frame_crc_ok(dec)
    is_connect = all(by[i] == v for i, v in _SETUP_STRUCTURE)
    return sc, c0, crc_ok, by[:46], is_connect


# --------------------------------------------------------------------------- #
# the index ladder (host BITRATE(3), (2), (1)): one-hot columns, three bits a column
def _l3_band(samples: np.ndarray, off: int, ncol: int,
             level: int = ROBUST_LEVEL) -> np.ndarray:
    """``ncol × span`` band readings from block ``off``, equalised per bin.

    One column lights one bin, so the median down a bin is what that bin reads
    while unlit — its share of the channel's response — and dividing it out puts
    every bin's lit and unlit readings on one scale before anything compares them.
    """
    r = INDEX_RECORDS[level]
    b = np.asarray(samples[off:off + ncol * r.dw], float).reshape(ncol, r.dw)
    m = np.abs(np.fft.rfft(b, axis=1)[:, r.first_bin:r.first_bin + r.span])
    return m / np.maximum(np.median(m, axis=0), _L3_BIN_FLOOR * m.max() + 1e-12)


def _l3_purity(band: np.ndarray) -> float:
    """Mean share of a column's band energy sitting in its strongest bin.

    The block grid's own figure of merit, and a level classifier on its own: the
    two recorded record-2 overs read 0.98 and 0.99 aligned and never below 0.30 at
    any offset, where every record-3 burst of that session reads 0.15-0.17.
    """
    s = band.sum(1)
    p = np.zeros(len(band))
    np.divide(band.max(1), s, out=p, where=s > 0)
    return float(p.mean())


def _l3_grid(samples: np.ndarray, start: int, stop: int | None,
             level: int = ROBUST_LEVEL) -> int:
    """Sample phase of the record's column grid — payload-blind.

    A column is one lit bin, so the phase the emitter used is the one whose columns
    read as a single bin; every other phase splits each column between the bins its
    two neighbours light. Coarse over one column period, then twice refined.
    """
    dw = INDEX_RECORDS[level].dw
    end = len(samples) if stop is None else min(stop, len(samples))
    ncol = min(_L3_GRID_BLOCKS, (end - start) // dw - 1)
    if ncol < 4:
        return start

    def score(o: int) -> float:
        if o < 0 or o + ncol * dw > len(samples):
            return -1.0
        return _l3_purity(_l3_band(samples, o, ncol, level))

    best = max(range(start, start + dw, 32), key=score)
    for span, step in ((32, 4), (4, 1)):
        best = max(range(best - span, best + span + 1, step), key=score)
    return best


def _l3_ref_hits(band: np.ndarray, col: int, level: int = ROBUST_LEVEL) -> int:
    """How many of the 24 reference columns of a frame opening at ``col`` light
    the bin the tables predict."""
    ref, ref_bin = _l3_tables(level)[:2]
    lit = band[col + ref].argmax(1) + INDEX_RECORDS[level].first_bin
    return int((lit == ref_bin).sum())


def _l3_align(samples: np.ndarray, start: int, stop: int | None,
              level: int = ROBUST_LEVEL):
    """``(offset, lead, matched, band)`` for an index-record burst at ``start``.

    ``matched`` is how many of the 24 reference columns light the bin the tables
    predict, at the lead that matches most — payload-blind, and the whole of what
    says a burst is this record rather than another, or the base one. ``band``
    covers the whole measured burst, every frame it holds; a burst of unmeasured
    length is read one frame deep.
    """
    r = INDEX_RECORDS[level]
    off = _l3_grid(samples, start, stop, level)
    end = len(samples) if stop is None else min(stop + r.dw, len(samples))
    ncol = (end - off) // r.dw
    if stop is None:
        ncol = min(ncol, r.ncols + _L3_LEAD_MAX)
    if ncol < r.ncols:
        return off, 0, 0, None
    band = _l3_band(samples, off, ncol, level)
    hits = [_l3_ref_hits(band, lead, level)
            for lead in range(min(ncol - r.ncols, _L3_LEAD_MAX) + 1)]
    lead = int(np.argmax(hits))
    return off, lead, hits[lead], band


def _l3_crc_ok(bits: np.ndarray, level: int = ROBUST_LEVEL) -> bool:
    """The index records' convergence test: CRC-16/GENIBUS over the frame ahead
    of its last two bytes."""
    n = INDEX_RECORDS[level].frame
    by = _dewhiten(bits)
    c = crc16_genibus(by[:n - 2])
    return by[n - 2:n] == bytes(((c >> 8) & 0xFF, c & 0xFF))


def _l3_decode(band: np.ndarray, lead: int,
               level: int = ROBUST_LEVEL) -> tuple[bytes, bool]:
    """Band readings -> ``(frame bytes, crc_ok)`` for the frame starting at ``lead``.

    A data column's three bits are one joint decision over its whole band reading
    rather than three independent ones: the LLR of a bit is the log-ratio of the
    energy in the bins carrying it as zero to the energy in those carrying it as
    one, over the eight offsets that mean anything at all.
    """
    r = INDEX_RECORDS[level]
    _, _, data, labels, valid, place, perm = _l3_tables(level)
    p = band[lead + data] ** 2
    ll = p / (np.median(p, axis=1, keepdims=True) + 1e-18)
    lik = (np.exp(ll - ll.max(axis=1, keepdims=True)) * valid)[:, :, None]
    bits = (np.log((lik * (labels == 0)).sum(1) + 1e-300)
            - np.log((lik * (labels == 1)).sum(1) + 1e-300)).ravel()
    Lc = np.zeros(r.coded)
    Lc[place] = np.clip(bits[:r.coded], -_L3_LLR_CAP, _L3_LLR_CAP)
    check = partial(_l3_crc_ok, level=level)
    if r.rate13:
        dec = _turbo.decode_r13(Lc, r.n_src, perm, iters=12, check=check)
    else:
        dec = _turbo.decode_from_coded_llr(Lc, r.n_src, perm=perm, iters=12,
                                           check=check)
    return _dewhiten(dec)[:r.frame], check(dec)


def _l3_frames(start: int, stop: int | None, off: int, lead: int,
               matched: int, band: np.ndarray | None,
               level: int = ROBUST_LEVEL) -> list[FrameResult]:
    """Every frame of an index-record burst from its alignment, in emission
    order. ``self_consistency`` is the share of the frame's 24 reference columns
    that matched, ``c0`` the column the frame opens at, and ``onset`` the block
    grid's phase inside the burst.

    Frames follow one another on the record's own grid with no gap and no
    training of their own, so frame ``k`` opens at ``lead + k * ncols`` and reads
    through the same tables as the first. The band says how many the burst
    holds, rounded on the frame length as :func:`frames_carried` rounds; one a
    truncated window cannot reach is counted, not returned.
    """
    r = INDEX_RECORDS[level]
    ncol = 0 if stop is None else (stop - start) // H
    if band is None:
        return [FrameResult(start=start, onset=(off - start,), c0=lead,
                            self_consistency=0.0, crc_ok=False, frame_bytes=b"",
                            burst_columns=ncol, frames_in_burst=1, level=level)]
    nf = max(1, round((len(band) - lead) / r.ncols))
    out = []
    for c0 in range(lead, len(band) - r.ncols + 1, r.ncols)[:nf]:
        by, crc_ok = _l3_decode(band, c0, level)
        hit = matched if c0 == lead else _l3_ref_hits(band, c0, level)
        out.append(FrameResult(start=start, onset=(off - start,), c0=c0,
                               self_consistency=hit / len(_l3_tables(level)[0]),
                               crc_ok=crc_ok, frame_bytes=by, burst_columns=ncol,
                               frames_in_burst=nf, level=level))
    return out


def decode_l3_burst(samples: np.ndarray, start: int, stop: int | None = None,
                    level: int = ROBUST_LEVEL) -> FrameResult:
    """The first frame the burst at ``start`` carries, read as the index record
    of host ``level``."""
    return _l3_frames(start, stop, *_l3_align(samples, start, stop, level), level)[0]


# --------------------------------------------------------------------------- #
def decode_burst_frames(samples: np.ndarray, start: int, stop: int | None = None,
                        level: int | None = None) -> list[FrameResult]:
    """Every frame the burst at ``start`` holds, in emission order.

    ``stop`` is the burst's end as :func:`burst_spans` measured it; it fixes how
    many frames the burst carries (:func:`frames_carried`). A window that runs out
    of columns before a later frame stops the list there — that frame is then
    counted by ``DecodeResult.frames_skipped`` rather than returned.

    ``level`` names the speed level to read the burst at. Left ``None`` the burst
    says which it is, by the payload-blind reference-column score of each index
    record (:data:`INDEX_RECORDS`): the record that scores the guard is read for
    every frame its length holds (:func:`_l3_frames`); none scoring it, the
    burst is base.
    """
    if level != BASE_LEVEL:
        levels = INDEX_RECORDS if level is None else (level,)
        aligned = {lv: _l3_align(samples, start, stop, lv) for lv in levels}
        lv = max(aligned, key=lambda lv: aligned[lv][2])
        if level is not None or aligned[lv][2] >= _L3_GUARD_MIN:
            return _l3_frames(start, stop, *aligned[lv], lv)
    return _decode_base_frames(samples, start, stop)


def _decode_base_frames(samples: np.ndarray, start: int, stop: int | None) -> list[FrameResult]:
    """Every base-level frame of the measured burst window, in emission order."""
    cols, onsets = _demod_burst(samples, start)
    ncol = 0 if stop is None else (stop - start) // H
    nf = frames_carried(ncol)
    out: list[FrameResult] = []
    locked_c0: int | None = None      # frame 0 locks it; later frames reuse it
    for fk in range(nf):
        gh = _frame_gh(cols, fk)
        if gh is None:
            break
        sc, c0, crc_ok, fb, is_connect = _decode_cells(gh, fk, locked_c0)
        locked_c0 = c0
        out.append(FrameResult(start=start, onset=(onsets[0], onsets[1]), c0=c0,
                               self_consistency=sc, crc_ok=crc_ok,
                               frame_bytes=fb, is_connect=is_connect,
                               burst_columns=ncol, frames_in_burst=nf))
    if not out:
        # A burst too short even for frame 0 (a fade, or a recording ending
        # mid-burst): return one failed frame so the shape stays one-per-burst.
        out.append(FrameResult(start=start, onset=(onsets[0], onsets[1]), c0=0,
                               self_consistency=0.0, crc_ok=False, frame_bytes=b"",
                               is_connect=False, burst_columns=ncol,
                               frames_in_burst=nf))
    return out


def decode_burst(samples: np.ndarray, start: int, stop: int | None = None) -> FrameResult:
    """The first frame of the burst at ``start`` (see :func:`decode_burst_frames`)."""
    return decode_burst_frames(samples, start, stop)[0]


def decode_stream(samples: np.ndarray) -> DecodeResult:
    """Decode a full VARA-A TX recording (mono, 48 kHz float)."""
    samples = np.asarray(samples, dtype=float)
    frames: list[FrameResult] = []
    for span in burst_spans(samples):
        group = decode_burst_frames(samples, *span)
        frames.extend(group)
        # An operator watching a transfer go past has to see a hole in it; a
        # silent skip here is what let half a session go missing unremarked. With
        # both frames now demodulated this fires only on a genuinely truncated
        # burst — the frames its length promised that the columns could not yield.
        if (lost := group[0].frames_in_burst - len(group)) > 0:
            print(f"[bw500] {span[0] / FS:7.2f} s burst: {group[0].burst_columns} "
                  f"columns hold {group[0].frames_in_burst} frames, {len(group)} "
                  f"demodulated — {lost} frame(s) skipped, "
                  f"{lost * _PAYLOAD[group[0].level]} payload bytes not read",
                  file=sys.stderr, flush=True)
    data = [f for f in frames if not f.is_connect]
    return DecodeResult(frames=frames, data_frames=data)
