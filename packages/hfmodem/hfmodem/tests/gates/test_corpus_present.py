# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The evidence the receive side rests on has to be reachable, or say so loudly.

Every test that reads a real off-air recording is decorated with a ``skipif``, and
that is right for a run with no way to get the material: it is tens of megabytes of
captured audio and it is not in the package. It is wrong wherever the material is
meant to be to hand, because a recording that has gone missing turns the tests
resting on it into a green run with a smaller number in it.

Both readings cannot live in a per-test decorator, so the distinction is drawn here
instead, once, for every body of evidence kept outside the package. What separates
them is whether the material is **repairable from where the tree already points**:

  * it is, and something is missing — a stale or damaged copy, so this FAILS and
    names the files and the command that restores them;
  * it is not — nobody here can conjure it, so this warns by name and skips.

The second half is deliberate. The corpora are not redistributed, and a gate that
failed on their absence would be a wall in front of everyone who lacks them rather
than a report to the one person who can act. What must not happen is the third
outcome, which is what this file used to produce: a green run that omits the
evidence and says nothing. A skip reason is invisible without ``-rs``, so each skip
carries a warning too — pytest prints its warnings summary by default, and that
line is the whole mechanism. Without it, 236 of besra's outcomes and 6 of kestrel's
BW500 results existed only on the machines that happened to hold the material,
while every other run reported success without them.

Where the tree points is `tests/evidence.py`, and in a worktree that is not the
worktree: almost none of this material is in the index, so a worktree gets the code
and none of the recordings. Every declaration used to resolve against the worktree's
own root, and the tests reading them skipped -- which is how an agent sent to
investigate the WW2MI stream ran `tests/winlink/` to `91 passed, 3 skipped` and did
not learn that the three skipped were the WW2MI tests.

Each body, and what "repairable" means for it:

  ``rf-corpus`` beside this checkout holds the off-air recordings that are not this
  repo's own captures, with their hashes in ``rf-corpus/CHECKSUMS``. Present means
  it should be whole.

  The logged VARA sessions the BW500 and BW2300 alphabets are read out of are 92 MB
  apiece and live in a private archive. Nothing here may name it — shipping files
  may not point at the repositories this was merged from — so the claim that they
  are reachable is `KESTREL_CORPUS`, and it is the claim, not a directory, that
  this holds to account. Its home is ``tools/arbiters.env``, beside `PMON_ORACLE`.

  ``working/ardop/reference/`` holds besra's ground-truth renders. Git-ignored per
  that tree's own reading discipline, but cloned from a public repository — so an
  incomplete clone is repairable, and the skip names the command.

  ``working/pactor`` is committed and denied at the publication boundary by
  ``publish/manifest.toml``, so it is in every clone and in no distribution. A file
  missing from it is a damaged checkout, not material nobody downloaded. The
  silent-gateway recordings under it are the exception and are held to the other
  rule: they were made here and are in no clone at all.

  ``working/winlink`` holds both an arbiter produced outside this project, which is
  committed, and WW2MI's own turn, which was cut out of a receive recording on this
  machine and is not. Same directory, opposite readings.

  This station's own session records under ``working/`` and ``logs/`` are written by
  a run and ignored by git, so nobody else can produce them. So are the KiwiSDR
  recordings of sabir's first transmission under ``captures/``, which are the only
  fading channel any sabir result is measured on.
"""
from __future__ import annotations

import warnings

import pytest

from hfmodem.tests import evidence
from hfmodem.tests.besra import groundtruth
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.sabir import clips
from hfmodem.tests.shrike import archive
from hfmodem.tests.winlink import corpora as winlink


def _toll(session, marks) -> str:
    """What this absence costs, in tests of this run rather than in files.

    A count is what acting on this needs: six tests and none call for opposite
    things, and "every test resting on one" separates them no better than
    silence does. The markers are the link between a body of evidence and the
    tests resting on it, and pytest has expanded parametrisation and
    module-level ``pytestmark`` by the time a test runs — so it is counted here
    rather than tallied by hand, which no hand could keep right:
    `requires_bw500_handshake` is two functions and four tests.

    A run that collected none of them is the one case with no number to give. A
    gate run on its own is not evidence that nothing rests on this.
    """
    if session is None:
        return "every test resting on one is skipping rather than running"
    # Only the markers actually firing. A body of evidence is guarded by several,
    # and the ones whose own files are present hold nothing back — counting every
    # test that merely carries one of these marks reported 9 where 6 were skipping.
    firing = [m.mark for m in marks if m.mark.args and m.mark.args[0]]
    n = sum(1 for item in session.items
            if any(mk in firing for mk in item.iter_markers("skipif")))
    if not n:
        return "no test in this run rests on one, so this run does not measure it"
    return (f"{n} test{'s' if n > 1 else ''} in this run "
            f"{'are' if n > 1 else 'is'} skipping rather than running")


def _reachable(declared, *, repairable: bool, at, what: str, how: str,
               session=None, marks=()) -> None:
    missing = [p for p in declared if not p.exists()]
    if not missing:
        return
    toll = _toll(session, marks)
    if not repairable:
        # The heading is a claim about how much is here, so it is read off
        # `missing` rather than off the declared count: material nobody has and a
        # partial copy call for different work, and this station's own tree is
        # the second — 3 of the 8 logged VARA sessions absent, which the declared
        # count reports as all 8.
        head = (f"NO {what.upper()}" if len(missing) == len(declared)
                else f"{what.upper()} INCOMPLETE")
        absent = (f"{head} — {len(missing)} of {len(declared)} are not under "
                  f"{at}: {toll}. {how}")
        warnings.warn(absent, stacklevel=2)
        pytest.skip(absent)
    pytest.fail(
        f"{len(missing)} of {len(declared)} {what} are not under {at}, which is "
        f"otherwise to hand — so this is a stale or damaged copy rather than "
        f"material nobody has: {toll}:\n  "
        + "\n  ".join(str(p) for p in missing) + f"\n{how}")


def test_the_corpus_the_receive_side_is_measured_on_is_reachable():
    _reachable(corpora.RF_CORPUS_RECORDINGS,
               repairable=corpora.RF_CORPUS.is_dir(),
               at=corpora.RF_CORPUS,
               what="corpus recordings",
               how="They are the shared archive kept beside this checkout; "
                   "HFMODEM_CORPUS points the search elsewhere.")


def test_the_logged_sessions_kestrels_alphabets_are_read_out_of_are_reachable(request):
    _reachable(corpora.VARA_TREE_RECORDINGS,
               repairable=corpora.ROOT_DECLARED,
               at=corpora.ROOT,
               what="logged VARA sessions",
               how="Point KESTREL_CORPUS at the archive holding them — its home is "
                   "tools/arbiters.env, which tools/gates.sh sources.",
               session=request.session, marks=corpora.VARA_TREE_MARKS)


def test_the_reference_renders_besra_is_measured_against_are_reachable():
    _reachable(groundtruth.reference_renders(),
               repairable=groundtruth.REFERENCE_TREE.is_dir(),
               at=groundtruth.REFERENCE_TREE,
               what="ARDOP reference renders",
               how=f"Clone them in place: {groundtruth.REFERENCE_CLONE} "
                   f"{groundtruth.REFERENCE_TREE}")


def test_the_working_record_shrike_is_measured_on_is_reachable():
    _reachable(archive.COMMITTED,
               repairable=archive.ARCHIVE.is_dir(),
               at=archive.ARCHIVE,
               what="committed working-record files",
               how="git holds them, so `git checkout -- working/pactor` restores "
                   "them on a source tree; from a distribution they are gone by "
                   "design.")


def test_the_gateway_packets_shrikes_p3_acquisition_is_read_from_are_reachable():
    _reachable(archive.ws8eoc_p3(),
               repairable=evidence.private_source_tree(evidence.TREE, evidence.RECORD) or
                          any(p.exists() for p in archive.WS8EOC_P3_INDEX),
               at=archive.P3_FIXTURES,
               what="WS8EOC PACTOR-3 cuts",
               how="git holds them, so `git checkout -- "
                   "packages/hfmodem/hfmodem/tests/shrike/fixtures` restores them "
                   "on a source tree; these indexes and their WAVs are excluded "
                   "from the distribution by design.")


def test_the_station_records_kestrels_live_results_rest_on_are_reachable():
    lost = {p: flight for p, flight in corpora.STATION_LOST.items() if not p.exists()}
    if lost:
        warnings.warn(
            f"LOST STATION RECORDINGS: {len(lost)} original recordings remain unavailable; "
            "substitutes do not restore their receiver-specific evidence:\n  "
            + "\n  ".join(f"{p}: restoring flight: {flight}" for p, flight in lost.items()),
            stacklevel=2)
    for manifest, _ in corpora.STATION_MANIFESTS:
        if evidence.private_source_tree(evidence.TREE, evidence.RECORD):
            assert manifest.is_file(), f"Missing station evidence manifest: {manifest}"
    from hfmodem.tests import d_evidence
    logs = d_evidence.FIXTURES / "logs.json"
    if logs.is_file():
        for row in evidence.manifest_rows(logs):
            d_evidence.log(row["file"])
    # Check tracked clips before external absence can skip the gate. Hash damage
    # must also fail even if a different external recording is absent.
    for path, manifest in corpora.STATION_CHECKED:
        if manifest.name != "external.json" or path.exists():
            evidence.checked_samples(path, manifest)
    _reachable(corpora.STATION_RECORDS,
               repairable=any(d.is_dir() for d in
                              (evidence.WORKING, evidence.LOGS, evidence.CAPTURES)),
               at=evidence.RECORD,
               what="this station's own session records",
               how="The surviving originals and checked substitutes are declared "
                   "in corpora.STATION_RECORDS and STATION_MANIFESTS. Install "
                   "external manifest files create-only under captures/test-evidence; "
                   "lost receiver-specific pins need the flights named by their skips. "
                   "Missing files: " + ", ".join(str(p) for p in corpora.STATION_RECORDS
                                                if not p.exists()))


def test_every_lost_original_is_present_or_has_a_restoring_flight():
    if not evidence.private_source_tree(evidence.TREE, evidence.RECORD):
        pytest.skip(f"Station original recordings omitted from distribution: {evidence.WORKING}")
    assert len(corpora.STATION_ORIGINALS) == 32
    for path in corpora.STATION_ORIGINALS:
        assert path.exists() or corpora.STATION_LOST.get(path), path
    assert all(flight.strip() for flight in corpora.STATION_LOST.values())


@pytest.mark.parametrize("repairable", [False, True], ids=["published", "station"])
def test_missing_external_station_evidence_obeys_repairability(tmp_path, monkeypatch, repairable):
    missing = tmp_path / "external.flac"
    monkeypatch.setattr(corpora, "STATION_MANIFESTS", ())
    monkeypatch.setattr(corpora, "STATION_CHECKED", ())
    monkeypatch.setattr(corpora, "STATION_LOST", {})
    monkeypatch.setattr(corpora, "STATION_RECORDS", (missing,))
    for name in ("WORKING", "LOGS", "CAPTURES"):
        monkeypatch.setattr(evidence, name, tmp_path / name)
    if repairable:
        evidence.CAPTURES.mkdir()
        with pytest.raises(pytest.fail.Exception, match="external.flac"):
            test_the_station_records_kestrels_live_results_rest_on_are_reachable()
    else:
        with pytest.warns(UserWarning, match="external.flac"), pytest.raises(pytest.skip.Exception):
            test_the_station_records_kestrels_live_results_rest_on_are_reachable()


@pytest.mark.parametrize("kind", ["source", "origin", "partial", "published"])
def test_ws8eoc_absence_distinguishes_a_checkout_from_a_distribution(tmp_path, monkeypatch, kind):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "unrelated.json").write_text("{}")
    indexes = (fixtures / "ws8eoc-0908-p3.json", fixtures / "ws8eoc-0909-p3.json")
    (tmp_path / ".git").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "working").mkdir()
    origin = tmp_path / "origin"
    if kind in {"source", "origin"}:
        marker = (tmp_path if kind == "source" else origin) / "publish" / "manifest.toml"
        marker.parent.mkdir(parents=True)
        marker.write_text("# development source")
    if kind == "partial":
        indexes[0].write_text("[]")
    monkeypatch.setattr(evidence, "TREE", tmp_path)
    monkeypatch.setattr(evidence, "RECORD", origin)
    monkeypatch.setattr(archive, "P3_FIXTURES", fixtures)
    monkeypatch.setattr(archive, "WS8EOC_P3_INDEX", indexes)
    outcome = pytest.skip.Exception if kind == "published" else pytest.fail.Exception
    with pytest.raises(outcome, match="(?i)WS8EOC PACTOR-3 cuts"):
        test_the_gateway_packets_shrikes_p3_acquisition_is_read_from_are_reachable()


def test_the_gateway_recordings_shrikes_schedules_are_run_against_are_reachable():
    _reachable(archive.SILENT_GATEWAYS,
               repairable=False,
               at=archive.ARCHIVE / "captures",
               what="silent-gateway recordings",
               how="They were recorded here by analysis/provoke_listen.py and are "
                   "in no clone, so only this station can produce more. Without "
                   "them the burst-lock and break-in schedules run on synthetic "
                   "audio alone.")


def test_the_winlink_specimen_the_codec_is_measured_against_is_reachable():
    _reachable(winlink.COMMITTED,
               repairable=winlink.FIXTURES.is_dir(),
               at=winlink.FIXTURES,
               what="winlink specimen files",
               how="git holds them, so `git checkout -- working/winlink` restores "
                   "them on a source tree; from a distribution they are gone by "
                   "design.")


def test_the_cms_turn_the_winlink_layer_is_measured_against_is_reachable():
    _reachable(winlink.WW2MI,
               repairable=False,
               at=winlink.FIXTURES,
               what="WW2MI's recovered turn",
               how="It was cut out of a receive recording on the machine that made "
                   "it -- see working/winlink/SOURCE.txt -- so a clone never has "
                   "it and nobody else can restore it.")


def test_the_recordings_sabir_was_first_heard_on_are_reachable():
    _reachable(clips.KEPT,
               repairable=False,
               at=clips.SLOT,
               what="sabir first-transmission clips",
               how="Two public KiwiSDRs recorded them on 2026-08-29 and the run "
                   "wrote them into gitignored captures/, so a clone never has "
                   "them and only this station can make more. Without them the "
                   "beacon rungs are measured on a channel with no fading in it, "
                   "which is what let beacon_short reach the air.")


def test_public_robust_packet_recordings_are_reachable():
    from hfmodem.tests.brubru import corpus

    _reachable(corpus.RECORDINGS,
               repairable=any(p.exists() for p in corpus.RECORDINGS),
               at=corpus.ROOT, what="public Robust Packet recordings",
               how="The three named public recordings remain external corpus inputs.")
