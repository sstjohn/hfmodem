# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the wheels actually contain — all of it, not just the parts we look for.

Three packaging defects reached a built wheel in this project, and none was
visible from the source tree, from `git ls-files`, or from a green suite:

  * `packages = ["kestrel"]` is an explicit list, not a prefix. The wheel held
    the top-level modules and no subpackage and no asset at all — no receiver
    constant tables. Fixed by `packages.find` plus `package-data`.
  * `setuptools` reuses `build/lib/`, so a stale package — deleted from disk and
    untracked in git — kept riding along in every wheel built in that tree,
    shadowing current code with a copy that predated several deletions.
  * A wheel inspection that counts the subpackages you expect will not notice a
    stranger. That is how the second one survived being looked at directly.

Hence the shape of this file: assert the **complete** contents and fail on
anything unexpected, rather than assert that the things we want are present. The
first defect is caught by checking what is missing, the second only by checking
what is extra, and the third is why both are spelled out rather than sampled.

`pip wheel` is invoked rather than `build.__main__` because it is what a consumer
runs. It takes a couple of seconds per distribution against a clean tree; if it
takes minutes, see `namespaces = false` in the hfmodem pyproject — discovery
otherwise walks the working corpora, which are tens of gigabytes.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]

pytestmark = [
    pytest.mark.wheel,
    # `tools/` and not the root `pyproject.toml`: the distribution ships that
    # config, for its testpaths and its markers, so it stopped telling the two
    # trees apart. Building wheels there would also leave `build/` behind in
    # someone else's checkout, which is the litter the second defect above hid in.
    pytest.mark.skipif(not (REPO / "tools").is_dir(),
                       reason="not a source tree (installed-wheel run)"),
]

# The three distributions this monorepo builds, and the one top-level package
# each is allowed to contain.
DISTS = {
    "hfmodem": "hfmodem",
    "hfhost": "hfhost",
    "creance": "creance",
}

# Every subpackage of hfmodem, by equality. Adding one is a deliberate act and has
# to be recorded here; `host` joins this set when the dialects are extracted.
TOP = {"core", "station", "besra", "kestrel", "sabir", "shrike", "winlink", "tests"}

# Per protocol, so an explicit-list regression cannot ship a modem's top-level
# modules with nothing under them.
SUB = {
    # shrike has no subpackages at all: `assets` was one until every table it held
    # became a call into `shrike/tablegen.py`. An empty set is an assertion here,
    # not a placeholder -- it is what fails if a data directory comes back.
    "shrike": set(),
    "kestrel": {"arq", "coding", "host", "rx", "tx", "vara"},
    "besra": {"arq", "dsp", "fec", "frame", "host", "phy", "sim"},
    "sabir": {"arq", "compress", "dsp", "fec", "floor", "frame", "host", "phy", "sim"},
}

# Every non-`.py` file the hfmodem wheel may contain, by equality. There is one:
# the BW500 synthesis pulse, a deconvolution of recorded audio. Its generator is
# not missing -- it re-runs in 3.4 s and returns the shipped bytes -- but the
# recording it reads does not ship. Everything else that was once package data is
# now computed at the point of use.
#
# By equality because that is the half that was missing. This file asserts module
# lists by equality and used to assert assets by existence -- "something is under
# each of these directories" -- and a stale `build/lib/` staging tree rode 6.25 MB
# of deleted tables into every wheel built here without disturbing that check.
# The wavs under `tests/besra/fixtures/` are deliberately absent: package-data does
# not name them, so they reach a reader through the source distribution and not
# through a wheel.
WHEEL_DATA = {"hfmodem/kestrel/tx/assets/prototype_pulse.npz"}


_built: dict[str, zipfile.ZipFile] = {}


@pytest.fixture(scope="session")
def wheels(tmp_path_factory):
    """Every distribution, built once. Each test then asks for the ones it is
    about, rather than being parameterised over all three and skipping two —
    a permanent skip reads exactly like a gate that stopped running."""
    def build(dist: str) -> zipfile.ZipFile:
        if dist not in _built:
            out = tmp_path_factory.mktemp(f"wheel-{dist}")
            r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q",
                                "-w", str(out), str(REPO / "packages" / dist)],
                               capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                pytest.fail(f"{dist} wheel build failed:\n"
                            f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            built = list(out.glob("*.whl"))
            assert len(built) == 1, f"expected one {dist} wheel, got {built}"
            _built[dist] = zipfile.ZipFile(built[0])
        return _built[dist]
    return build


@pytest.mark.parametrize("dist", sorted(DISTS))
def test_wheel_has_no_stowaways(dist, wheels):
    """The whole top level, exactly. A stale `build/lib/` shipped for several
    commits precisely because nothing asserted the absence of extras."""
    wheel = wheels(dist)
    top = {n.split("/")[0] for n in wheel.namelist()}
    info = {t for t in top if t.endswith(".dist-info")}
    assert len(info) == 1, f"expected exactly one dist-info in {dist}, got {sorted(info)}"
    pkg = DISTS[dist]
    extra = top - info - {pkg}
    assert not extra, (
        f"unexpected top-level entries in the {dist} wheel: {sorted(extra)}. "
        "A stale build/ is the usual cause — remove it and rebuild.")
    assert pkg in top, f"the package itself is missing from {dist}: {sorted(top)}"


def test_hfmodem_carries_every_subpackage(wheels):
    wheel = wheels("hfmodem")
    got = {n.split("/")[1] for n in wheel.namelist()
           if n.startswith("hfmodem/") and n.count("/") > 1}
    assert got == TOP, (
        f"missing: {sorted(TOP - got)}; unexpected: {sorted(got - TOP)}. "
        "Equality, not containment — a stowaway one level down is what survived last time.")
    for proto, expected in SUB.items():
        prefix = f"hfmodem/{proto}/"
        under = {n.split("/")[2] for n in wheel.namelist()
                 if n.startswith(prefix) and n.count("/") > 2}
        assert under == expected, (
            f"{proto}: missing {sorted(expected - under)}, unexpected {sorted(under - expected)}")


def test_the_hfmodem_wheel_carries_one_data_file_and_no_other(wheels):
    wheel = wheels("hfmodem")
    got = {n for n in wheel.namelist()
           if n.startswith("hfmodem/") and not n.endswith("/")
           and not n.endswith(".py")}
    assert got == WHEEL_DATA, (
        f"missing: {sorted(WHEEL_DATA - got)}; unexpected: {sorted(got - WHEEL_DATA)}. "
        "Missing means package-data stopped matching and the wheel installs a modem "
        "that raises on first transmit. Unexpected means something ships as data "
        "that should be computed — or a stale build/ is riding along; remove it and "
        "rebuild.")


@pytest.mark.parametrize("dist", sorted(DISTS))
def test_wheel_ships_no_provenance_notes(dist, wheels):
    """Globbing package-data by extension keeps the notes beside the tables in
    the source tree. A directory-wide glob would publish them.

    The tokens are split so this file, which ships, names them as data without
    carrying them — the same reason `test_static.py` brackets a letter."""
    wheel = wheels(dist)
    leaked = [n for n in wheel.namelist()
              if n.lower().endswith("provenance" ".md")
              or ("CLEAN" "ROOM") in n.upper()]
    assert not leaked, f"method notes in the {dist} wheel: {leaked}"


@pytest.mark.parametrize("dist", sorted(DISTS))
@pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
def test_wheel_carries_the_licence(dist, name, wheels):
    """AGPL-3.0-only, one text, three distributions. A wheel that declares
    the licence in its metadata and does not carry it is not distributable.

    NOTICE by the same rule and for a sharper reason: besra reproduces parts of
    ardopcf, whose MIT terms require the copyright and permission notice to travel
    with every copy, and NOTICE is the only place either text exists. By equality,
    so a per-package copy cannot drift from the root."""
    wheel = wheels(dist)
    canonical = (REPO / name).read_bytes()
    got = [n for n in wheel.namelist() if Path(n).name == name]
    assert got, f"no {name} in the {dist} wheel"
    for n in got:
        assert wheel.read(n) == canonical, f"{dist}: {n} differs from the root {name}"


def test_hfhost_wheel_is_stdlib_only(wheels):
    """hfhost has to run on a machine with a commercial modem and no numpy, and
    its independence from the modem implementations is what makes creance's
    grading mean anything. Enforced against the built artifact, not the tree."""
    wheel = wheels("hfhost")
    import ast

    banned = {"numpy", "scipy", "sounddevice", "hfmodem", "creance",
              "shrike", "kestrel", "besra", "sabir"}
    for name in wheel.namelist():
        if not name.endswith(".py"):
            continue
        tree = ast.parse(wheel.read(name).decode(), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [node.module.split(".")[0]] if node.module and node.level == 0 else []
            else:
                continue
            for r in roots:
                assert r not in banned, f"{name} imports {r}"
                assert r in sys.stdlib_module_names or r == "hfhost", (
                    f"{name} imports {r}, which is neither stdlib nor hfhost")
