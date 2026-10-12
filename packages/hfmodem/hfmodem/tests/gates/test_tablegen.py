# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every constant table, regenerated and held against a byte reference.

This checks `kestrel/rx/tablegen.py` and `shrike/tablegen.py` against optional
reference vectors under `working/vara/spec/tables/`, or `KESTREL_CORPUS`.
Comparisons skip when the reference corpus is absent.

The few tables that are not fully derivable are compared over exactly the extent
that is -- with the boundary asserted, not waved at:

  * the whiteners match over their 150000-bit live extent; the reference files
    carry a foreign tail past it that no receiver reads,
  * `usedmap` matches over the off=3 record the receiver consults; the other 16
    records' planes are out of scope,
  * `grid480` matches, on the 128-point phase lattice, the 786 of 939 cells a
    SHORT burst emits and inversion can recover.

Everything else is byte-for-byte, including the two BW500 interleavers whose
"garbage" tail is the BW2300 table written under them.

One check has no shipped subject either. The PACTOR-III interleaver tables are
archived under `working/pactor/spec/` rather than shipped -- 8 of the 21
regenerate, the other 13 have no published derivation, and nothing at runtime
reads either -- so the comparison against them runs on a source tree and skips,
by name, anywhere the archive is absent.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pathlib

import numpy as np
import pytest

from hfmodem.kestrel.rx import tablegen as vara
from hfmodem.kestrel.rx import varahf500
from hfmodem.shrike import tablegen as pactor

_REPO = Path(__file__).resolve().parents[5]

#: Optional reference vectors for P3 interleaver checks.
EXTRACTED_INTERLEAVERS = (_REPO / "working" / "pactor" / "spec"
                          / "interleaver_tables.npz")

requires_extracted_interleavers = pytest.mark.skipif(
    not EXTRACTED_INTERLEAVERS.is_file(),
    reason=f"the archived P3 interleaver tables are not present at {EXTRACTED_INTERLEAVERS}")

#: The byte reference for the regenerated VARA tables: the `spec/tables/` data
#: facts, which do not ship. A source tree keeps a copy under `working/vara/`;
#: `KESTREL_CORPUS` points at the vara tree for a run that has it elsewhere.
GOLDEN = Path(os.environ.get("KESTREL_CORPUS") or _REPO / "working" / "vara") / "spec" / "tables"

requires_golden = pytest.mark.skipif(
    not GOLDEN.is_dir(),
    reason=f"the spec/tables byte reference is not present at {GOLDEN} "
           "(set KESTREL_CORPUS, or run in a source tree with working/vara/)")


def _gbytes(*parts: str) -> bytes:
    return GOLDEN.joinpath(*parts).read_bytes()


def test_draw_is_the_binary32_multiply():
    # The witness the shift-and-truncate reading gets wrong (703 vs the true 704).
    assert vara.draw(5613669, 2104) == 704
    assert (5613669 * 2104) >> 24 == 703


@requires_golden
def test_turbo_interleaver_byte_identical():
    """`kestrel/coding/turbo.py`'s table, which it now builds rather than loads.

    The comparison moved here from a check against a shipped copy of the same
    bytes, which proved the generator agreed with itself. Against the reference it
    proves what the shipped copy was standing in for."""
    assert vara.interleaver("bw500", 2).tobytes() == _gbytes("bw500", "turbo_interleave.i32")


@requires_golden
def test_whitener_live_extent():
    pn = vara.whitener()
    assert len(pn) == 150000
    for parts, tail in ((("bw500", "whitener_pn.u8"), 512),
                        (("bw2300", "whitener_pn.u8"), 70000)):
        ref = np.frombuffer(_gbytes(*parts), dtype=np.uint8)
        assert len(ref) == 150000 + tail
        assert np.array_equal(pn, ref[:150000])


@requires_golden
def test_interleavers_byte_identical():
    for bw, stage, parts in (("bw500", 1, ("bw500", "interleave_stage1.i32")),
                             ("bw2300", 1, ("bw2300", "interleave_stage1.i32")),
                             ("bw2300", 2, ("bw2300", "interleave_stage2.i32"))):
        assert vara.interleaver(bw, stage).tobytes() == _gbytes(*parts)


@requires_golden
def test_bw2300_base_tables_byte_identical():
    for gen, parts in ((vara.alloc_col3, ("bw2300", "alloc_col3.i32")),
                       (vara.map1_col3, ("bw2300", "map1_col3.i32")),
                       (vara.map2_col3, ("bw2300", "map2_col3.i32")),
                       (vara.clsparm_gray, ("bw2300", "clsparm_gray.i32"))):
        assert gen().tobytes() == _gbytes(*parts)


@requires_golden
def test_constellation_lut_byte_identical():
    assert vara.to_c64(vara.constellation_lut()) == _gbytes("constellation_lut_64.c64")


@requires_golden
def test_usedmap_record3_plane():
    ref = np.frombuffer(_gbytes("bw500", "usedmap.u8"), dtype=np.uint8)
    plane = vara.usedmap_plane()
    assert np.array_equal(plane[3::17], ref[3::17])
    assert int((plane[3::17] != 0).sum()) == 36


def _fetched_indices() -> np.ndarray:
    """Every grid480 cell the receiver reads, off the receiver's own index map."""
    ncol = varahf500._NCOL_DEMOD
    return np.array(sorted({int(i) for sub in (0, 1)
                            for i in varahf500.grid480_index(sub, ncol)}))


def test_grid480_generator_is_786_cells_on_the_lattice():
    """The generator zero-fills every cell the corpus could not recover, so what it
    lights is exactly the 786 measured integers, each an exact 128-point phasor.

    The fetch set tracks the demod window, not a snapshot of it: sub0 and sub1
    read disjoint period-469 and period-471 grid480 blocks (the moduli in
    grid480_index), so a window of ncol columns reaches min(ncol, period) distinct
    entries of each. 470 columns stopped one short of sub1's block (939); the
    two-frame window reads both in full (940)."""
    gen = vara.grid480()
    recovered = np.flatnonzero(np.abs(gen) > 1e-6)
    fetched = _fetched_indices()
    assert len(recovered) == 786
    assert set(recovered.tolist()).issubset(set(fetched.tolist()))
    assert np.allclose(np.abs(gen[recovered]), 1.0)

    ncol = varahf500._NCOL_DEMOD
    assert len(fetched) == min(ncol, 469) + min(ncol, 471)


@requires_golden
def test_grid480_matches_the_reference_on_the_lattice():
    ref = np.frombuffer(_gbytes("bw500", "derotation_reference_grid480.c64"), dtype="<f4")
    shipped = ref[::2] + 1j * ref[1::2]
    gen = vara.grid480()
    recovered = np.flatnonzero(np.abs(gen) > 1e-6)

    def lattice_k(z):
        return np.mod(np.round(np.angle(z) * 128 / (2 * np.pi)), 128).astype(int)

    assert np.array_equal(lattice_k(gen[recovered]), lattice_k(shipped[recovered]))


def test_the_shrike_tables_are_the_shapes_their_readers_index():
    """The three tables `rx`, `pactor2` and `p3frame` used to load from `assets/`.

    Their byte references are the acquisition and header structure asserted right
    through this suite and the P2/P3 conformance modules, so what is pinned here is
    the dtype and extent a caller indexes into -- an int64 raster the gather reads
    positionally, complex64 chips the correlator multiplies, and the 512 int taps
    `p3frame` reshapes to 64 groups of 8. A generator that returned the right
    values in the wrong dtype would decode nothing and fail nowhere.
    """
    raster = pactor.hdr_raster_src()
    assert raster.shape == (288,) and raster.dtype == np.int64
    assert len(set(raster.tolist())) == 288      # a gather map has to be injective
    assert raster.min() == 52 and raster.max() == 52 + 563 * 3 + 71

    codes = pactor.marker_codes()
    assert codes.shape == (2, 16, 8) and codes.dtype == np.complex64
    assert set(np.unique(codes).tolist()) == {1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j}

    pre = pactor.pre800()
    assert pre.shape == (512,) and pre.dtype == np.int64
    assert set(np.unique(pre).tolist()) == {-800, 800}


def test_markers_are_real_orthogonal():
    codes = pactor.marker_codes()
    w = np.concatenate([codes[0], codes[1]], axis=1)
    gram = w @ w.conj().T
    assert np.array_equal(gram.real, 32 * np.eye(16))


@requires_extracted_interleavers
def test_pactor_interleavers_byte_identical():
    tables = np.load(EXTRACTED_INTERLEAVERS)
    for name, gen in pactor.INTERLEAVERS.items():
        assert np.array_equal(gen(), tables[name].astype(np.int64)), name


#: Every binary asset the publication manifest ships, by directory. `.py` and the
#: (non-shipping) `.md` notes are not constants; everything else here crosses.
#:
#: The glob is recursive. It used to be `**/assets/*`, one level, and a table one
#: directory further down -- `assets/bw2300/interleave_stage1.i32`, which is where
#: seven of them sat the last time this went wrong -- was invisible to it.
_SHIPPED_ASSETS = sorted(
    p for p in (_REPO / "packages" / "hfmodem" / "hfmodem").glob("**/assets/**/*")
    if p.is_file() and p.suffix not in (".py", ".md") and "__pycache__" not in p.parts)

_DERIVATIONS_DOC = _REPO / "docs" / "protocols" / "table-derivations.md"


#: The supported category for measured binary assets.
_CATEGORIES = ("measured", "replay swaps")

#: Source vocabulary for the measured-asset inventory.
_OWN_SOURCES = ("loopback", "off-air", "audio", "recording")


def _asset_rows() -> dict[str, dict[str, str]]:
    """The shipped-binaries table, keyed by filename.

    Found by its header rather than by its position or its contents, so the
    document's sections can be rewritten under it without silently emptying this
    gate -- a parser that locates the table by what is in it today reports every
    asset clean the moment the table moves.
    """
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for line in _DERIVATIONS_DOC.read_text().splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        keys = [re.sub(r"[^a-z]", "", c.lower()) for c in cells]
        if "category" in keys and "howknown" in keys:
            header = keys
            continue
        if header is None:
            continue
        named = re.search(r"`([^`]+)`", cells[0])
        if named:
            rows[named.group(1)] = dict(zip(header, cells))
    return rows


def test_every_shipped_binary_is_filed_and_sourced():
    """The document still has to account for what ships, file by file.

    This is bookkeeping rather than clearance now: what may ship at all is settled
    by `test_manifest.py`, which refuses any non-text file in the distribution that
    is not one of the named off-air recordings or the one measurement whose input
    recording does not ship. A row that reads well cannot clear a file past that.
    What this keeps
    is the account -- a shipped binary nobody wrote down is still a defect.
    """
    rows = _asset_rows()
    faults = []
    for shipped in _SHIPPED_ASSETS:
        row = rows.get(shipped.name)
        if row is None:
            faults.append(f"{shipped.name}: no row in the shipped-binaries table")
            continue
        category = row.get("category", "").lower()
        how = row.get("howknown", "").lower()
        if category not in _CATEGORIES:
            faults.append(f"{shipped.name}: filed as {row.get('category', '')!r}, "
                          f"which is not one of {list(_CATEGORIES)}")
        elif category == "measured" and not any(s in how for s in _OWN_SOURCES):
            faults.append(
                f"{shipped.name}: measured, and the How-known column names no "
                f"source we may distribute -- say one of {list(_OWN_SOURCES)}")
    assert not faults, (
        f"{_DERIVATIONS_DOC.name} does not clear these to ship:\n  "
        + "\n  ".join(faults))


def test_symbol_pulse_is_generated_and_only_named_c2_replay_data_ships():
    """Generated pulse plus the explicitly approved replay data inventory.

    Offline CML source and LGPL license are text; every binary other than the
    named C2 replay resources remains a regression. Export gates pin their bytes.
    """
    from hfmodem.shrike import tablegen

    mf = tablegen.symbol_pulse()
    assert mf.shape == (31,)
    assert np.array_equal(mf, mf[::-1])
    assert abs(float(mf.sum()) - 1.0) < 1e-6
    assert abs(float(mf.max()) - 0.146490) < 1e-6

    shrike_pkg = pathlib.Path(tablegen.__file__).parent
    data = sorted(p.relative_to(shrike_pkg).as_posix()
                  for p in shrike_pkg.rglob("*")
                  if p.is_file() and p.suffix not in (".py", ".md", ".c", ".h")
                  and p.name != "COPYING.LESSER" and "__pycache__" not in p.parts)
    expected = sorted(f"assets/c2-{n}.swaps.z" for n in
                      (280,348,352,704,1056,1160,1408,1656,4140,8280,12420,16560,20700))
    assert data == expected, f"unexpected shrike binary inventory: {data}"


def test_the_acquisition_window_is_not_the_symbol_pulse():
    """The two kernels run at the same rate and are not the same filter.

    Reusing the pulse here is the plausible wrong answer, and it is expensive in a
    way a correlation score does not report: it costs the P2 receive path every dB
    of its single-copy margin and stops burst copies summing at any noise level,
    while still recovering an off-air marker at 0.997. So the shape is pinned by
    what tells the two apart -- unit roll-off collapses the numerator to
    `4t*cos(2*pi*t)`, whose zeros are exact and fall at samples 6 and 10 either
    side of a peak that sits at 13 of the 32-deep ring, centred rather than
    left-aligned two samples late.
    """
    from hfmodem.shrike import tablegen

    w = tablegen.acq_window()
    assert w.shape == (32,)
    assert int(w.argmax()) == 13
    assert np.array_equal(w[:27], w[26::-1])
    assert not w[27:].any()
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert np.allclose(w[[3, 7, 19, 23]], 0.0, atol=1e-15)


def _draw_number(n: int, span: int, seed: int = 29318) -> int:
    s = seed
    for _ in range(n):
        s = vara.step(s)
    return vara.draw(s, span)


def test_scramble_is_the_whitener_stream_row_major_by_bin():
    s = vara.scramble()
    assert s.shape == (62, 409)
    assert s[8, 0] == _draw_number(3273, 129) == 75      # BW2300's first carrier
    assert s[27, 0] == _draw_number(11044, 129) == 89    # BW500's
    assert s[61, 408] == _draw_number(62 * 409, 129)
    assert s.min() == 0 and s.max() == 128


def _gap_length(bw: str, rec: int) -> int:
    """LCG steps from the record's gap to the next record's seed."""
    s, target, n = vara._gap(bw, rec), vara._next_seed(bw, rec), 0
    while s != target:
        s, n = vara.step(s), n + 1
        assert n < 100
    return n


#: Gap lengths in draws, as the closures measured them on the high speed capable tapes.
_OFDM_GAPS = {("bw2300", r): 49 for r in range(9, 16)} | {
    ("bw2750", r): 59 for r in range(10, 17)} | {
    ("bw500", 6): 46, ("bw500", 7): 44, ("bw500", 8): 44, ("bw500", 9): 45,
    ("bw500", 10): 44, ("bw500", 11): 46, ("bw500", 12): 49}


@pytest.mark.parametrize("bw, rec", sorted(_OFDM_GAPS))
def test_gap_cells_close_on_the_next_seed(bw, rec):
    cells = vara.gap_cells(bw, rec)
    carriers, per, qam = {"bw2300": (49, 1, (14, 15)), "bw2750": (59, 1, (15, 16)),
                          "bw500": (11, 4, (11, 12))}[bw]
    assert cells.shape == (carriers, per)
    assert _gap_length(bw, rec) == _OFDM_GAPS[bw, rec]
    qam = rec in qam
    lo, hi = (0, 18) if qam else (7, 148)
    assert cells.min() >= lo and cells.max() <= hi
    assert all(len(set(row)) == per for row in cells.tolist())


def test_gap_cell_values():
    assert vara.gap_cells("bw2300", 9)[:3, 0].tolist() == [57, 59, 61]
    assert vara.gap_cells("bw2300", 14)[:3, 0].tolist() == [2, 10, 0]
    assert vara.gap_cells("bw500", 6)[0].tolist() == [134, 146, 117, 31]
    with pytest.raises(ValueError):
        vara.gap_cells("bw2300", 8)


#: (draws, bands, pilots per band, columns) per tone record.
_TONE_GAPS = {("bw500", 3): (37, 2, 18, 392), ("bw500", 4): (38, 2, 18, 393),
              ("bw500", 5): (36, 3, 12, 374), ("bw2750", 9): (39, 13, 3, 347)} | {
    (bw, rec): row
    for rec, row in {4: (37, 3, 12, 374), 5: (36, 4, 9, 363), 6: (36, 6, 6, 353),
                     7: (40, 8, 5, 350), 8: (41, 10, 4, 347)}.items()
    for bw in ("bw2300", "bw2750")}


@pytest.mark.parametrize("bw, rec", sorted(_TONE_GAPS))
def test_tone_pilots_close_on_the_next_seed(bw, rec):
    draws, bands, npil, ncols = _TONE_GAPS[bw, rec]
    pilots = vara.tone_pilots(bw, rec)
    assert _gap_length(bw, rec) == draws
    assert len(pilots) == bands
    for band in pilots:
        assert len(band) == npil == len(set(band))
        assert list(band) == sorted(band)
        assert 5 <= band[0] and band[-1] <= ncols - 2


def test_tone_pilots_reproduce_the_bw500_level4_table():
    """The 36 (column, sub-band) pilots the BW500 L4 receiver was built on, when
    they were a literal read off the air."""
    measured = {
        (7, 1), (24, 0), (26, 1), (30, 1), (32, 1), (38, 0), (42, 0), (48, 0),
        (62, 1), (107, 0), (108, 1), (139, 1), (154, 0), (158, 0), (161, 1),
        (177, 0), (177, 1), (184, 0), (188, 1), (200, 0), (212, 1), (226, 0),
        (227, 0), (243, 1), (251, 1), (284, 1), (292, 1), (302, 0), (316, 1),
        (323, 0), (323, 1), (331, 0), (334, 0), (335, 0), (336, 0), (374, 1),
    }
    got = {(co, s) for s, cols in enumerate(vara.tone_pilots("bw500", 3)) for co in cols}
    assert got == measured
    assert vara.tone_pilots("bw2300", 4)[0][:3] == (30, 39, 90)


def test_grid_rows_are_lattice_integers_long_enough_for_their_levels():
    lengths = [len(vara.grid_row(b)) for b in range(13)]
    assert lengths == [394, 394, 375, 364, 354, 354, 351, 351, 348, 348, 348, 348, 348]
    for b in range(13):
        row = vara.grid_row(b)
        assert row.min() >= 0 and row.max() < 128
    cells = dict(vara._GRID480_CELLS)
    assert vara.grid_row(0).tolist() == [cells[i] for i in range(393)] + [109]
    assert vara.grid_row(1).tolist() == [cells[471 + i] for i in range(393)] + [82]
    assert vara.grid_row(10)[:3].tolist() == [81, 86, 18]
    assert vara.grid_row(12)[:3].tolist() == [96, 50, 102]
    # Level 7's last three columns, read off BW2750 L7's clean overs.
    assert vara.grid_row(4)[-3:].tolist() == [32, 93, 122]
    assert vara.grid_row(5)[-3:].tolist() == [100, 105, 5]
    # Every tone row reads columns 0..ncols of each band's row.
    for (bw, rec), (bands, _, ncols) in vara._TONE.items():
        assert all(len(vara.grid_row(b)) > ncols for b in range(bands)), (bw, rec)


@pytest.mark.parametrize("name, half, taps", [("P8", 1024, 2049), ("P9", 1024, 2049)])
def test_tone_pulses_are_the_measured_fits(name, half, taps):
    from hfmodem.kestrel.tx import tone_pulse

    p = np.array(getattr(tone_pulse, name))
    assert p.shape == (taps,) and p[half] == 1.0
    a = (p[half:] + p[half::-1]) / 2
    assert int(np.argmax(a < 0.5)) == 211
    assert int(np.argmax(a < 0)) == 385
    assert int(a.argmin()) == 481 and -0.071 < a.min() < -0.070
    tail = np.sum(p[np.abs(np.arange(taps) - half) > 512] ** 2) / np.sum(p ** 2)
    assert (1e-3 < tail < 3e-3) if name == "P8" else (5e-5 < tail < 1e-4)


def test_bw2750_chain_continues_bw2300_and_closes_on_every_seed():
    assert vara.BW2750[:9] == vara.BW2300[:9]
    assert vara.BW2750[9][0] == vara.BW2300[9][0] == 15908083
    assert _gap_length("bw2750", 9) == 39                # level 10's pilots, 13 x 3
    assert vara.tone_pilots("bw2750", 9)[0] == (218, 240, 284)
    assert vara.BW2750[10][0] == 11548315
    assert vara._next_seed("bw2750", 16) == 4061299
    assert [n for _, _, n in vara.BW2750[10:]] == [8667, 11556, 13864, 17338, 20804, 26145, 31376]


def test_bw2750_interleaver_holds_its_longest_permutation():
    one, two = vara.interleaver("bw2750", 1), vara.interleaver("bw2750", 2)
    assert len(one) == 39235 * 17 and len(two) == 31376 * 17
    col = one[16::17]
    assert sorted(col.tolist()) == list(range(39235))
    ref = vara.interleaver("bw2300", 1)
    for rec in range(9):
        n = vara.BW2300[rec][1]
        assert np.array_equal(one[rec::17][:n], ref[rec::17][:n])
