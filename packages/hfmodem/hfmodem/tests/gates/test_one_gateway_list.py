# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The published gateway lists have one copy each, and every reader names it.

**What happened.** Three copies of Winlink's VARA gateway export lived in this
tree at once — `winlink-vara-gateways.csv`, `working/winlink-vara-gateways.csv`
and `working/vara/winlink-vara-gateways.csv` — fetched on 30 July, 4 August and
29 July. Same URL, same automated export, three dates. None had been edited by
hand; there was no correction in any of them to lose. They existed because
`--cache` defaulted to a bare relative name in `hfcapture.py` and `vara_monitor.py`,
so which file a run read or wrote was decided by the directory it was started in,
and each launcher then hardcoded whichever one its own working directory had left
behind. `.gitignore` matched the name unanchored, so the second and third never
appeared in `git status` and nothing said they were there.

**Why it is a gate and not a cleanup.** The list is what
`kestrel_connect.unspeakable` refuses an arm on — it decides whether a channel can
answer before the transmitter is keyed. Asked about the 1255 channels the three
copies name between them, they returned different verdicts on 58 at BW2300 and on
129 at BW500. WP4OH-13 on 7104.7 kHz is one, and it shows the dangerous direction:
that station is listed VARA 500 only, so a BW2300 call there cannot be answered
and two of the copies refuse it — but the copy `onair.sh` actually passed did not
list WP4OH-13 at all, and an unlisted station reads as unknown, which the guard
allows. The oldest snapshot did not disagree with the guard; it was silent, and
silence is a pass. Meanwhile the operator's briefing pointed at a fourth file
entirely, so the channel the runbook vetted and the channel the guard checked were
read out of different snapshots.

**Where the authoritative file comes from.** It is winlink.org's own public
GatewayChannels export, filtered to one mode and service PUBLIC, fetched over
HTTP from the same page a browser renders — two ASP.NET postbacks, in
`hfcapture.fetch_gateways`, one mode per call. Every column in it is published
by the gateway operators themselves. The header uses winlink's column names.

**Keyed on content, not on the filename.** Two gates in this tree have already
gone blind on a name: the stale-claims gate lost its reports when a filename
convention drifted under its glob, and the transmit-drain gate never walked the
directory the bench tools live in. A rename cannot hide a copy from this one —
every CSV in the tree is opened and matched on the export's own header row, which
is winlink.org's and not ours to drift.

**The path check is an identity, not a grep.** The tools are imported and their
defaults compared against the one constant, so a reader that resolves somewhere
else fails here even if it spells the path in a way no pattern anticipated.

**A session snapshot is not an exception, and it asked to be one four times.**
Three rig sessions each kept the roster they planned from — `roster-snapshot.csv`
twice and `channels-pactor.csv` once, byte-identical, one export between them —
and a fourth kept the VARA list of the same morning. Each was defended as what
that arm actually read, which is true and is not the question: the newest PACTOR
roster in the tree then lived in a session directory while the canonical one was
six weeks old, and the operator's briefing planned from the session copy. What an
arm read is fixed by the rows it worked, and those are in its own `queue.csv`;
what the export said on the day is in git. Neither needs a second roster on disk.

`working/winlink-channels-from-pat.csv` is deliberately not covered. It is a
different instrument from a different source — `pat rmslist`, carrying distance
and hours-since-status for all three protocols — and not a copy of this export.
Where the two disagree they are two opinions, which is the honest state; it is
covered here only to the extent that it, too, may exist exactly once.
"""
from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
TOOLS = REPO / "tools"

pytestmark = pytest.mark.skipif(not TOOLS.is_dir(),
                                reason="tools/ not present (installed-wheel run)")

#: The header winlink.org's GatewayChannels export writes, for every mode.
EXPORT_HEADER = ("Timestamp,Callsign,BaseCallsign,GridSquare,Frequency,Mode,"
                 "Hours,Sysop,QTH")

#: One canonical path per export. The VARA list sits at the root because that is
#: where every reader that was already right had it, and because it is a fetched
#: cache rather than a source artifact. The PACTOR list has only ever had one copy.
CANONICAL = {REPO / "winlink-vara-gateways.csv",
             REPO / "working" / "winlink-pactor-gateways.csv"}

#: Not searched: build spoil, and every hidden directory. The hidden ones hold
#: git's internals, the virtualenv and the per-agent worktrees, which are separate
#: checkouts carrying their own copies by design; none is a path a tool reads.
SKIP_DIRS = {"node_modules", "__pycache__"}


def _every_csv() -> list[Path]:
    found, stack = [], [REPO]
    while stack:
        for entry in stack.pop().iterdir():
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                    stack.append(entry)
            elif entry.suffix.lower() == ".csv":
                found.append(entry)
    return sorted(found)


@functools.cache
def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_gate", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(TOOLS))
    return mod


def test_each_published_export_exists_in_exactly_one_place():
    copies = [p for p in _every_csv()
              if p.read_text(errors="replace").split("\n", 1)[0].strip() == EXPORT_HEADER]
    stray = sorted(p.relative_to(REPO).as_posix() for p in copies if p not in CANONICAL)
    assert not stray, (
        "a second copy of a Winlink gateway export is in the tree: "
        + ", ".join(stray)
        + ". The guard refuses arms on one of these; two of them means the copy it "
          "read and the copy the runbook quoted can disagree, which they did on 58 "
          "channels at BW2300. Delete it, or re-fetch the canonical one with "
          "--refresh: " + ", ".join(sorted(p.relative_to(REPO).as_posix()
                                           for p in CANONICAL)))


#: Every tool that fetches, caches or guards on the VARA list. Each imports the
#: constant rather than spelling the path, so this reads the resolved value out
#: of the loaded module: a reader that lands somewhere else fails here however it
#: spells the path, and whether it spells one at all.
READERS = ("hfcapture", "vara_monitor", "kestrel_connect", "onair_session",
           "gwsurvey")


def test_the_vara_list_has_exactly_one_definition():
    canonical = REPO / "winlink-vara-gateways.csv"
    for name in READERS:
        assert _tool(name).GATEWAY_LIST == canonical, (
            f"{name}.py resolves the gateway list to "
            f"{_tool(name).GATEWAY_LIST}, not {canonical}")

    # pick_targets holds the path rather than the module: it is stdlib-only on
    # purpose, since it reaches the propagation feed and importing hfcapture would
    # pull numpy in behind it. Membership is the check that costs it nothing.
    assert canonical in _tool("pick_targets").LISTS


def test_a_fetch_lands_only_at_the_one_path(tmp_path, monkeypatch):
    """Pointing a reader elsewhere must not be able to CREATE a copy there.

    The three copies were not written by hand. `load_gateways` used to fetch into
    whatever `--cache` named when that file was missing, so a caller with a
    relative path and a working directory of its own left a fourth copy behind
    every time it ran — one appeared in `working/vara` twice while this
    consolidation was being written, minutes apart, from a caller that has not
    been identified. Locating that caller is not the fix; a fetch having one
    destination is. Reading from another path stays allowed, which is what an
    audit against a preserved snapshot needs.
    """
    vm = _tool("vara_monitor")
    landed, asked = tmp_path / "canonical.csv", tmp_path / "somewhere-else.csv"
    monkeypatch.setattr(vm, "GATEWAY_LIST", landed)
    monkeypatch.setattr(vm, "fetch_gateways",
                        lambda: EXPORT_HEADER + "\n8/27/2026 5:39:00 AM,N0AAA,"
                        'N0AAA,EN63,"7,103.500 KHz",VARA,00-23,-,"Here"\n')

    assert vm.load_gateways(asked, None, False) == ["N0AAA"]
    assert landed.exists(), "a fetch did not land at the canonical path"
    assert not asked.exists(), (
        f"a fetch wrote a second copy at {asked.name}. A reader pointed "
        f"elsewhere may read from there; it may never create it.")


def test_no_launcher_names_a_path_of_its_own():
    """The shell launchers pass `--gateways` explicitly, so the constant cannot
    reach them. They are checked as text — the only place in this tree where the
    path is spelled rather than imported."""
    scripts = [TOOLS / "onair.sh", TOOLS / "lib" / "attempts.sh"]
    wrong = []
    for script in scripts:
        if not script.exists():
            continue
        for n, line in enumerate(script.read_text().splitlines(), 1):
            if "winlink-vara-gateways.csv" not in line:
                continue
            if '"$REPO/winlink-vara-gateways.csv"' not in line:
                wrong.append(f"{script.relative_to(REPO).as_posix()}:{n}: {line.strip()}")
    assert not wrong, (
        "a launcher names a gateway list other than $REPO/winlink-vara-gateways.csv:\n"
        + "\n".join(wrong))
