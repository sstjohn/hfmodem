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
_CATEGORIES = ("measured",)

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


def test_the_symbol_pulse_is_generated_and_shrike_ships_no_data_at_all():
    """The generated kernel used for transmit shaping and receive matching.

    The absence is asserted too, and it is now the whole of shrike rather than two
    filenames. Naming `tx_pulse.npz` and `p2_acquisition.npz` caught the two files
    that had gone and nothing that might arrive next to them; every table shrike
    needs is computed by `tablegen.py`, so any data file anywhere under the package
    is a regression whatever it is called."""
    from hfmodem.shrike import tablegen

    mf = tablegen.symbol_pulse()
    assert mf.shape == (31,)
    assert np.array_equal(mf, mf[::-1])
    assert abs(float(mf.sum()) - 1.0) < 1e-6
    assert abs(float(mf.max()) - 0.146490) < 1e-6

    shrike_pkg = pathlib.Path(tablegen.__file__).parent
    data = sorted(p.relative_to(shrike_pkg).as_posix()
                  for p in shrike_pkg.rglob("*")
                  if p.is_file() and p.suffix not in (".py", ".md")
                  and "__pycache__" not in p.parts)
    assert not data, f"shrike carries data files again: {data}"


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
