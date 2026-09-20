# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Standing negatives in the shipped documents, against the record that disproves them.

Three times a published document has asserted that something has never happened
while this station's own tooling, in this tree, had already written down that it
had. The README and `docs/ONAIR-READINESS.md` said "no data packet of shrike's has
ever been acknowledged by a real station" and "the peer has never alternated
CS1/CS2" while six verdict files — WS8EOC and KB5LZK, four sessions on four dates —
read `verdict: ACKNOWLEDGED`. That one shipped in the wheel's PyPI description and
survived a same-day edit of the paragraph containing it. The same documents said
shrike had no per-speed-level data frame for fourteen days after the frames landed.

**The mechanism is what this gate is built against: a stretch goal failed, and was
written down as a floor.** The bar for a session was three consecutive per-cycle
advances. That has still never happened, and it is still written down — correctly —
in both documents. What went wrong is that "three consecutive advances has never
happened" was restated as "no packet has ever been acknowledged", which is a
different sentence about a different thing, and false. A committed working note
recorded all three packets of one session accepted first try on the same day the
shipped text said none ever had.

So the ground truth here is the machine-written verdict, not a person's summary.
`shrike/onair.py` prints one at the end of every session, in a form it has held
across every record the globs below reach — 344 files and 123 verdict lines on
2026-08-19 — and `tools/rigtest.sh` keeps it beside the recording. This reads those
and refuses a shipped sentence that denies what they say.

WHAT IT KNOWS, deliberately five things and not fifty. A gate that is right about a
handful of claims is worth more than one that is vague about all of them, and every
claim below is one this project has actually got wrong in public:

  * a data packet of ours was acknowledged;
  * the peer alternated its control signal;
  * our packet counter advanced;
  * a payload crossed to a second station;
  * shrike has per-speed-level data frames.

The first four are read off `verdict: ACKNOWLEDGED`. The fifth is read off
`placement.SPEED_PATHS`, which ships, so that one bites in a clone too.

WHAT KEEPS IT OFF CORRECT WRITING. These documents are full of true negatives and
of history stated as history, and a gate that flagged those is a gate people switch
off. Four discriminators, all of them earned from the real text:

  * **Scope.** "No station has accepted a **PACTOR-3** packet" and "carry a payload
    above **speed level 1**" are true today. A sentence scoped to PACTOR-3, to SL3,
    or to a speed level above the first is left alone.
  * **Subject.** "It does not run ARQ, so it has never acknowledged a shrike packet"
    is about the PACTOR-III *monitor*, which reads and cannot answer. A sentence
    naming a monitor is left alone.
  * **Date and callsign.** "the signature WS8EOC produced on the air on 2026-07-27 —
    CS4 at zero errors, twenty cycles, no advance" is a record of one session, not a
    standing claim, and the whole tree is written that way.
  * **History markers**, checked over the sentence, its paragraph and the heading
    above it. `ONAIR-READINESS.md` corrected the data-frame claim by past-tensing it
    in place and dating the supersession rather than deleting it, which is the right
    repair and the model for the failure message below.

And one guard that is the mechanism itself rather than a discriminator: the advance
claim spares any sentence qualified by *three consecutive*, *run of three* or
*per-cycle*. The qualified sentence is true and belongs in the documents. Only the
unqualified floor is refused.

The evidence is kept out of the distribution rather than out of a clone, and those
are different absences. `publish/manifest.toml` denies `working/**` and `captures/**`
both, so a wheel has neither; `.gitignore` denies `captures/` alone, and `working/`
is in the index — 4335 files, the nine verdicts reading ACKNOWLEDGED among them. So a
clone and a worktree read this for real and an installed package cannot. Where the
material is missing anyway this follows `test_corpus_present.py`: only this station
can produce more, so the absence warns by name and skips rather than walling. The
warning is the point. A skip reason is invisible without `-rs`, and silence is the
exact failure mode being designed against.

The second half of this file, below the shipped-document claims, runs the same
mechanism the other way round: a working report set against the logs it quotes.
A shipped document denying what the record shows and a field report asserting what
no record holds are one failure seen from either side, and the second dispatches
work the night it is written.

AND WHAT THE FIVE CANNOT REACH. The list above is retrospective by construction:
each entry pairs an English negative with the machine token that refutes it, and a
person supplies that pairing after the error. So it caught none of the seven
findings of 2026-08-26, two of which were exactly its shape. The fourth section at
the foot of this file is the part of it that does generalise, and it is a
convention rather than a truth check: a standing negative in a published document
names the record that settles it. That fires on tomorrow's negative without
anybody having had to be wrong about it first, and run against `README.md` as it
stood that morning it flags the turn-law sentence -- the one that shipped in the
wheel's long description while seven byte-exact deliveries sat in the tree.
"""
from __future__ import annotations

import re
import tomllib
import warnings
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

import pytest

from hfmodem.tests import evidence

REPO = Path(__file__).resolve().parents[5]
PYPROJECT = REPO / "packages" / "hfmodem" / "pyproject.toml"
MANIFEST = REPO / "publish" / "manifest.toml"
PLACEMENT = REPO / "packages" / "hfmodem" / "hfmodem" / "shrike" / "placement.py"

#: Where shrike's own tooling leaves its adjudication of a session. Written by the
#: run, one line, machine phrasing unchanged across all 344 records these reach
#: (123 verdict lines, 14 of them ACKNOWLEDGED, on 2026-08-19). The
#: globs are shallow on purpose: `captures/` is 5 GB of recordings and rglob over it
#: costs more than reading every log under it.
VERDICTS = ("working/rig-session-*/verdict.txt", "captures/*.log", "captures/*.sum",
            "captures/*/*.log", "captures/*/*.sum")

_ACKNOWLEDGED = re.compile(r"^verdict: ACKNOWLEDGED\b.*", re.M)
_SPEED_PATHS = re.compile(r"^SPEED_PATHS\b.*", re.M)

#: The two bodies of evidence, keyed by what they establish, and what to say when
#: one of them is not here. The first is denied to the distribution and the second
#: ships, so their absences mean opposite things and must not read the same.
FROM_THE_AIR = "a data packet of ours acknowledged by a station"
IN_THE_SOURCE = "a data frame for each speed level"
ABSENT = {
    FROM_THE_AIR: "under working/ and captures/, so nothing here can tell a stale "
                  "denial of it from a true one. A run writes both trees; the index "
                  "holds working/ and not captures/, and the manifest ships neither, "
                  "so this is a distribution or a tree somebody pruned. Only this "
                  "station can produce more: keep the sessions or lose the gate "
                  "that reads them.",
    IN_THE_SOURCE: "in shrike/placement.py, which ships — so this is not a missing "
                   "input but a missing table, and the sentence denying it may well "
                   "be true again.",
}


def _where(path: Path) -> str:
    """A path as the reader will look for it, whichever tree it came out of."""
    for root in (REPO, evidence.RECORD):
        if path.is_relative_to(root):
            return str(path.relative_to(root))
    return str(path)


@dataclass(frozen=True)
class Fact:
    path: Path
    line: int
    text: str

    def cite(self) -> str:
        return f'{_where(self.path)}:{self.line}\n      "{self.text}"'


@dataclass(frozen=True)
class Claim:
    """A thing the record establishes, and the sentence shape that denies it.

    `denies` and `about` must both match, and `about` is what keeps a negation from
    being read as this claim because it happens to share a verb. `unless` is the
    claim's own exception, for the neighbouring sentence that is true.
    """
    holds: str
    rests_on: str
    denies: re.Pattern[str]
    about: re.Pattern[str]
    unless: re.Pattern[str] | None = None

    def stale(self, sentence: str) -> bool:
        if not (self.denies.search(sentence) and self.about.search(sentence)):
            return False
        return not (self.unless and self.unless.search(sentence))


# A negation, then the claim's own verb within a sentence's reach. Written in that
# order because every shape this has gone wrong in puts them that way round — "no
# data packet ... has ever been acknowledged", "nothing shrike sent has ever been
# accepted" — while the innocent sentences that share the verb put the negation
# after it ("An acknowledgement is a codeword toggle, though, not a receipt").
_NO = r"\b(?:no|not|never|nothing|none|nor)\b"

#: The stretch goal this project set itself and has still not met. A negative
#: qualified by it is the one sentence shape this file must never flag, whichever
#: rule is doing the reading -- restating it as an unqualified floor is the whole
#: failure being designed against, and the qualified form is the repair.
_THE_STRETCH_GOAL = re.compile(r"three consecutive|run of three|per-cycle"
                               r"|three cycles running", re.I)

CLAIMS = (
    Claim(
        holds="a station has acknowledged a data packet of shrike's",
        rests_on=FROM_THE_AIR,
        # Past participle only. `acknowledges` in the present is these documents
        # describing a mechanism — "after a correct packet it acknowledges and
        # forces 200 Bd" — and never a claim about what has happened on the air.
        denies=re.compile(
            rf"{_NO}[^.]{{0,160}}?\b(?:acknowledged|acknowledgement|accepted)", re.I),
        # `data` alone is far too common in these documents to mean the subject.
        about=re.compile(r"\b(?:packet|payload)\b", re.I),
    ),
    Claim(
        holds="the peer has alternated its control signal",
        rests_on=FROM_THE_AIR,
        denies=re.compile(rf"{_NO}[^.]{{0,160}}?\balternat", re.I),
        about=re.compile(r"\b(?:CS1|CS2|control signal|codeword|acknowledg)", re.I),
    ),
    Claim(
        holds="our packet counter has advanced",
        rests_on=FROM_THE_AIR,
        denies=re.compile(rf"{_NO}[^.]{{0,160}}?\b(?:advanc|moved)", re.I),
        about=re.compile(r"\b(?:counter|packet)\b", re.I),
        # The stretch goal, which is true and stays. This is the whole failure in one
        # line: "three consecutive advances have never happened" is not "no packet
        # has ever been acknowledged", and turning the first into the second is what
        # this file exists to stop.
        unless=_THE_STRETCH_GOAL,
    ),
    Claim(
        holds="a payload of shrike's has crossed to a station",
        rests_on=FROM_THE_AIR,
        # `carr` on its own would take `carrier`, which is on every timing page.
        denies=re.compile(
            rf"(?:{_NO}|\bcannot\b)[^.]{{0,160}}?"
            r"\b(?:cross\w*|carry|carries|carried|carrying|exchang\w*)\b"
            r"|\b(?:cross\w*|carry|carried|exchang\w*)\b[^.]{0,80}?\bimpossible\b",
            re.I),
        # Not `message`: "No complete message has crossed in either direction" is
        # true, is the standing gap both documents state exactly, and is a different
        # claim from the bytes that have crossed.
        about=re.compile(r"\b(?:payload|data packet)\b", re.I),
    ),
    Claim(
        holds="each speed level has its own data frame",
        rests_on=IN_THE_SOURCE,
        denies=re.compile(rf"(?:{_NO}|\blacks\b)[^.]{{0,80}}?\bdata frame", re.I),
        about=re.compile(r"per[- ]speed[- ]level|per-SL|speed level", re.I),
    ),
)

#: A sentence carrying one of these is not a standing claim about the record.
_SCOPED = re.compile(r"PACTOR-3|PACTOR-III|\bSL ?[234]\b|\bSL ?≥ ?2\b"
                     r"|speed level [234]|above speed level", re.I)
_A_MONITOR = re.compile(r"\bmonitor\b|\bPMON\b", re.I)
_CALLSIGN = re.compile(r"\b[A-Z]{1,2}\d[A-Z]{1,4}\b")
_DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
#: ...and these mark writing that is *about* an earlier state, which is correct and
#: is how the data-frame claim was repaired: past-tense it in place and date it.
_HISTORY = re.compile(r"\bsupersede[sd]\b|\bused to (?:say|read|call)\b|\bno longer\b"
                      r"|\ban earlier revision\b|\bproved false\b|\bhas since\b"
                      r"|\b(?:changed|corrected|superseded) 20\d\d-\d\d-\d\d",
                      re.I)


def _exempt(sentence: str, paragraph: str, heading: str) -> bool:
    if _SCOPED.search(sentence) or _A_MONITOR.search(sentence):
        return True
    if _CALLSIGN.search(sentence) and _DATE.search(sentence):
        return True
    return any(_HISTORY.search(t) for t in (sentence, paragraph, heading))


_FENCE = re.compile(r"^\s*(?:```|~~~)")
#: A blockquote marker mid-paragraph lands inside the joined sentence, where it
#: splits phrases the exemptions are matched on: `docs/RUNBOOK.md` opens on a quoted
#: briefing whose "carry a payload above speed level 1" wraps across the `>`.
_QUOTE = re.compile(r"^(?:>\s*)+")
#: A full stop followed by something that opens a sentence. Not a bare `[.!?]\s`:
#: these documents are full of `0.07 s` and `pactor1-timing.md`, and a split inside
#: one of those separates a negation from the thing it negates.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[*`_A-Z])")


def _sentences(text: str):
    """(line, sentence, paragraph, heading) over the prose of a markdown document.

    Fenced blocks and indented transcripts are skipped: a log excerpt quoted in the
    runbook is evidence being shown, not a claim being made.
    """
    heading, para, out, fenced = "", [], [], False

    def flush():
        if para:
            out.extend(_split(para, heading))
            para.clear()

    for n, raw in enumerate(text.splitlines(), 1):
        if _FENCE.match(raw):
            fenced = not fenced
            flush()
            continue
        if fenced:
            continue
        line = _QUOTE.sub("", raw.strip())
        if not line:
            flush()
        elif line.startswith("#"):
            flush()
            heading = line
        elif raw.startswith("    ") and not para:
            continue
        else:
            para.append((n, line))
    flush()
    return out


def _split(para: list[tuple[int, str]], heading: str):
    joined, marks = "", []
    for n, line in para:
        if joined:
            joined += " "
        marks.append((len(joined), n))
        joined += line
    cuts = [0, *(m.start() for m in _BOUNDARY.finditer(joined)), len(joined)]
    for start, end in zip(cuts, cuts[1:]):
        sentence = joined[start:end].strip()
        if sentence:
            line = max(n for off, n in marks if off <= start)
            yield line, sentence, joined, heading


def _read(paths, pattern: re.Pattern[str]) -> list[Fact]:
    facts = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        facts += [Fact(path, text.count("\n", 0, m.start()) + 1, m.group().strip())
                  for m in pattern.finditer(text)]
    return facts


@pytest.fixture(scope="module")
def record() -> dict[str, list[Fact]]:
    sessions = sorted({p for g in VERDICTS for p in evidence.RECORD.glob(g)})
    return {FROM_THE_AIR: _read(sessions, _ACKNOWLEDGED),
            IN_THE_SOURCE: _read([PLACEMENT], _SPEED_PATHS)}


def _published() -> list[Path]:
    """Root-level and `docs/` markdown, as a reader of the distribution gets it.

    Globbed rather than listed, so a new page in `docs/` is covered by writing it.
    Not recursive, which is the whole scoping decision: `docs/protocols/` records
    what *other* stations did in captured recordings — "a packet counter that never
    moved", "that packet was never acknowledged" — and every one of those sentences
    is true and about somebody else's modem.

    That glob is then narrowed to what `publish/manifest.toml` lets across, and
    only where the manifest is here. In the distribution it is not, and there the
    glob needs no narrowing: the manifest is an allowlist, so a `docs/` that has
    been through it holds exactly what crossed. In the tree the glob is the wrong
    answer — `docs/ARCHITECTURE.md` is a plan for work not yet begun and does not
    ship — and until this narrowing the gate reached one verdict here and another
    inside the wheel, where the ratchet below read that document's absence as a
    negative somebody had settled and failed to strike off. The document set has to
    agree in both places or the ratchet has nothing steady to count against.

    Measured before narrowing, because a gate that quietly stops looking is this
    family's recurring fault: the three pages it drops on 2026-08-27 —
    `docs/ARCHITECTURE.md`, `docs/AUTOPILOT.md`, `docs/TRANSMIT-MEASUREMENT.md` —
    hold no claim and no standing negative between them, so nothing under scrutiny
    today leaves it.

    Read as data, never imported. `publish/` does not cross its own boundary, and a
    gate that depended on it would be one more test that has to leave the
    distribution rather than run in it — and the distribution is where these
    documents are read.
    """
    docs = sorted([*REPO.glob("*.md"), *REPO.glob("docs/*.md")])
    if not MANIFEST.exists():
        return docs
    man = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    held = man.get("exclude", {}).get("paths", [])

    def crosses(rel: str) -> bool:
        return any(fnmatch(rel, rule["path"])
                   and not any(fnmatch(rel, e)
                               for e in (*rule.get("except", ()), *held))
                   for rule in man["include"])

    return [p for p in docs if crosses(p.relative_to(REPO).as_posix())]


@pytest.fixture(scope="module")
def shipped() -> list[tuple[Path, str, int]]:
    """This station's account of itself, with each document's line offset."""
    docs = [(p, p.read_text(encoding="utf-8"), 0) for p in _published()]
    # The wheel's long description is markdown living in a TOML string, and it is
    # where the acknowledgement claim reached PyPI.
    raw = PYPROJECT.read_text(encoding="utf-8") if PYPROJECT.exists() else ""
    blurb = tomllib.loads(raw).get("project", {}).get("readme") if raw else None
    if isinstance(blurb, dict):
        text = blurb["text"]
        n, first = next((n, ln) for n, ln in enumerate(text.splitlines(), 1)
                        if ln.strip())
        docs.append((PYPROJECT, text, raw.count("\n", 0, raw.index(first)) + 1 - n))
    return docs


def _stale(claim: Claim, docs) -> list[tuple[str, str]]:
    return [(f"{path.relative_to(REPO)}:{offset + line}", sentence)
            for path, text, offset in docs
            for line, sentence, para, heading in _sentences(text)
            if claim.stale(sentence) and not _exempt(sentence, para, heading)]


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.holds)
def test_no_shipped_document_denies_what_the_record_shows(claim, record, shipped):
    facts = record[claim.rests_on]
    if not facts:
        absent = (f"NO EVIDENCE OF {claim.rests_on.upper()} {ABSENT[claim.rests_on]} "
                  f"Nothing is checking whether a shipped document still denies "
                  f"that {claim.holds}.")
        warnings.warn(absent, stacklevel=2)
        pytest.skip(absent)
    offences = _stale(claim, shipped)
    assert not offences, (
        f"{len(offences)} shipped sentence(s) deny what this station's own record "
        f"shows — that {claim.holds}:\n\n"
        + "\n\n".join(f'  {where}\n      "{sentence}"\n\n'
                      f"  disproved by {facts[0].cite()}"
                      + "".join(f"\n      and by {_where(f.path)}:{f.line}"
                                for f in facts[1:])
                      for where, sentence in offences)
        + "\n\nIf the sentence was true once, repair it the way ONAIR-READINESS.md "
          "repaired the data-frame claim: past-tense it in place and date the "
          "supersession, so the record of having been wrong survives the fix.")


#: Passages from the shipped documents as they stand. Every one carries the exact
#: vocabulary of a claim above and every one is correct: true negatives scoped to a
#: speed level, to a monitor or to the stretch goal; the standing gap, which is
#: about a complete message and not about bytes; another modem's record; and one
#: dated session. A rule that fires here is a rule that gets switched off, so these
#: hold the narrow end.
#:
#: Passages rather than sentences, because two of the discriminators are not visible
#: in a sentence on its own — the last entry is a claim in the past tense whose
#: supersession is dated in the sentence after it, which is the repair this gate
#: asks for and so must be the one shape it never flags.
INNOCENT = (
    "What has *not* happened: no **PACTOR-3** data packet has been accepted — each "
    "of those sessions keyed one SL3 packet, all four were answered in PACTOR-1 and "
    "the link fell back — no run of three consecutive per-cycle advances, on our "
    "counter or the peer's, and no complete message either way.",
    "It does not run ARQ, so it has never acknowledged a shrike packet, and a frame "
    "a monitor prints is not proof an SCS modem in a session will accept it.",
    "**No station has accepted a PACTOR-3 packet.**",
    "What shrike cannot yet do — carry a payload above speed level 1 — begins after "
    "the link is up, and has been misread as a limit on which stations it can reach.",
    "Nor has any link sustained the advance: three consecutive per-cycle advances "
    "have never happened, on our counter or the peer's, and the exchange ends after "
    "pkt#3, whose break-in bit invites the changeover.",
    "No complete message has crossed in either direction, for any of the four.",
    "What crossed in those sessions was 132 and 239 bytes of RMS Trimode greeting "
    "banner; no message body has ever been carried.",
    "It is the signature WS8EOC produced on the air on 2026-07-27 — CS4 at zero "
    "errors, twenty cycles, no advance — which `tests/shrike/test_p1peer.py` reads "
    "as a peer saying the packet does not decode.",
    "No sabir signal has ever been exchanged with a second station, because a "
    "receiver that cannot transmit gives no round trip: nothing acknowledged us, "
    "nothing asked us to repeat, and no implementation other than this one has "
    "ever produced a sabir waveform.",
    "An acknowledgement is a codeword toggle, though, not a receipt — it does not "
    "show the bytes arrived intact.",
    "The data frame is not what stands in the way: every speed level has its own "
    "frame geometry, and an independent PACTOR-III monitor, running in a VM, reads "
    "what shrike renders and prints the payload back at the level it was sent at.",
    "The same run with the packet counter frozen, and the same run with the data "
    "field mangled under the CRC the good field computed, each brought the link up "
    "and then **sent CS4 forever without advancing** — nothing delivered.",
    "**Still true that evening, and not fixed by any of this:** shrike had no "
    "per-speed-level data frame, so a payload exchange with a real station was "
    "impossible. The realistic goal was a link that comes up and is *seen* to come "
    "up by both ends. (Superseded 2026-08-01, when the per-level frames landed — "
    "see \"What still blocks a data session\" below.)",
)


def test_the_claim_rules_do_not_fire_on_the_correct_writing():
    """A gate that flags correct work is a gate people switch off."""
    for passage in INNOCENT:
        for _, sentence, para, heading in _sentences(passage):
            assert not [c.holds for c in CLAIMS if c.stale(sentence)
                        and not _exempt(sentence, para, heading)], sentence


#: The sentences that actually shipped, recovered from the commits that removed
#: them. This is the counterweight: a rule narrowed until nothing innocent trips it
#: reports a clean tree and means nothing by it.
#:
#: The last is the exception and is written here rather than quoted, because the
#: advance claim shipped as one clause of a sentence whose other clauses are
#: correctly scoped to PACTOR-3 — "no **PACTOR-3** data packet has been accepted …
#: no sustained advance past a second packet". A sentence is the unit here, so that
#: clause is invisible, and it is the price of not firing on the true clause beside
#: it. What is caught is the claim standing on its own, which is how it has been
#: written every other time.
SHIPPED_AND_FALSE = (
    ("What has never happened is a payload crossing: there is no per-speed-level "
     "data frame yet, so shrike can bring a link up and cannot carry a payload to "
     "an SCS modem, and no data packet of shrike's has ever been acknowledged by a "
     "real station.",
     {"a station has acknowledged a data packet of shrike's",
      "a payload of shrike's has crossed to a station",
      "each speed level has its own data frame"}),
    ("What no session has produced is an acknowledged data packet: the peer has "
     "never alternated CS1/CS2, so nothing shrike sent has ever been accepted as "
     "data.",
     {"a station has acknowledged a data packet of shrike's",
      "the peer has alternated its control signal"}),
    ("A gateway answers; no payload has ever crossed.",
     {"a payload of shrike's has crossed to a station"}),
    ("shrike has no per-speed-level data frame, so a payload exchange with a real "
     "station remains impossible.",
     {"a payload of shrike's has crossed to a station",
      "each speed level has its own data frame"}),
    ("shrike has no per-speed-level data frame transmitter, so it borrows the fixed "
     "24-byte header frame as the carrier and adds a length byte, which no SCS "
     "modem will read.",
     {"each speed level has its own data frame"}),
    ("Until the real per-SL data frames exist in both directions, shrike can "
     "complete a handshake with a real station but not a payload exchange.",
     {"a payload of shrike's has crossed to a station"}),
    ("Our packet counter has never advanced against a real station.",
     {"our packet counter has advanced"}),
)


@pytest.mark.parametrize("sentence,expected", SHIPPED_AND_FALSE,
                         ids=range(len(SHIPPED_AND_FALSE)))
def test_the_claim_rules_fire_on_what_actually_shipped(sentence, expected):
    """Each of these was published, and the tree already held its disproof."""
    caught = {c.holds for c in CLAIMS
              if c.stale(sentence) and not _exempt(sentence, sentence, "")}
    assert caught == expected, sentence


# --------------------------------------------------------------------------- #
# The other direction: a working report, against the logs it quotes.
#
# The same failure the other way round. Above, a shipped document denies what the
# record shows; here a field report asserts what no record holds -- and the second
# does more damage, because work is dispatched on it the same night it is written.
#
# Five such claims were found in one report. The one that names the class is set in
# a fenced block as log output:
#
#     2 turnarounds are witnessed in this window and timing cannot say which is
#     answering us
#
# It appears in no log on file. It is `shrike/onair.py`'s string for `witnessed > 1`,
# and it has never printed, because no session has taken that branch. Beside it in
# the same report: "`rx data` is zero in every PACTOR log on file", which five lines
# in two logs disprove; a comparison column carrying another run's figures; and a
# "changeover: never" for a run that logged three.
#
# Which of those a mechanical check can hold is the whole design. It can hold the
# quotation: a span set as the program's own output either is in a log or is not.
# It cannot hold "this column's figures came from this run", because the columns are
# named by flag and not by file, and a gate that guesses is a gate that gets
# switched off. So this refuses a quotation no log holds, and refuses one attributed
# to a log that does not hold it, and leaves the rest to a reader -- who now has the
# report's own headings to read it under.
#
# The absence claims are the damaging half and are not checkable from a report at
# all: they stop the search, because nobody greps for what they have been told does
# not exist. `_stale` above is the mechanism for the ones that reach a shipped
# document, and it is the reason this file exists.

#: The field reports and the logs they are written off, both written by a run and
#: both in the index -- so a clone runs this and only a distribution, which the
#: manifest denies them to, cannot. The same bargain as the rest of the file.
#:
#: `*-REPORT.md` stood here until 2026-08-26 and reached five of the twenty reports
#: this tree holds: a trailing date defeated it, and the naming drifted to carry one
#: while nothing noticed. Widening it is one character and surfaced 258 quotations
#: no log held -- of which two were real. What the other 256 measured is where the
#: operator keeps the log, not what the report says, and that is answered by
#: `test_a_report_that_quotes_output_names_a_log_the_tree_holds` below.
REPORTS = "*-REPORT*.md"

#: Every log in the tree, not just the ones at the top of `working/`. A shallow glob
#: stood here and missed 34 quotations that a session directory or a capture holds
#: verbatim -- `captures/onair-0825-2101/` alone answers ten. 869 files and 6.5 MB,
#: enumerated and read in 0.1 s together, so the cost this was avoiding is not there.
LOG_TREES = ("working/**/*.log", "working/**/*.narr", "logs/**/*.log",
             "logs/**/*.narr", "captures/**/*.log", "captures/**/*.narr")

#: Where a phantom is looked for once no log holds it. A string the program CAN
#: print and never has is a different mistake from an invented one: somebody read
#: the source and wrote down what it would say. Different mistake, different repair,
#: so the failure separates them.
SOURCE = (REPO / "packages" / "hfmodem" / "hfmodem", REPO / "tools")

#: Below this a span in backticks is a flag, a filename or an identifier rather than
#: a line of output. `--ack-aim closed` and `working/…-REPORT.md` are the shapes
#: this keeps out; `#1, #2, #3, #0 x22` is the shortest real quotation on file.
_QUOTE_WORDS = 5
#: Fences holding something other than output. A language tag says so outright, and
#: a prompt or a command name says it where the tag is missing.
_LANGUAGES = {"sh", "bash", "zsh", "console", "shell", "python", "py", "diff",
              "json", "toml", "yaml", "md", "markdown", "text"}
#: A command line, which is an instruction rather than a transcript. The leading
#: `VAR=value` run matters: an operator sets a budget in front of the verb it
#: governs, and a gate that reads `POUNCE_WAIT=0 ./tools/onair.sh …` as output
#: demands a log hold the command that produced it.
_A_COMMAND = re.compile(r"^(?:[A-Z_][A-Z0-9_]*=\S*\s+)*"
                        r"(?:[$>#]|(?:\./|python|python3|pytest|git|grep|rg|sed|awk"
                        r"|cat|ls|curl|ssh|make|uv|pip|ruff|hfmodem)\b"
                        r"|\S+\.(?:sh|py)\b)")
_A_PATH = re.compile(r"^[\w./-]+$")
#: A span quoting the shell rather than a run of it: a parameter expansion, or a
#: placeholder standing where an argument goes. `[ "${5:-}" = force ]` is a line of
#: `onair.sh` and `rigtest.sh vara <gw> <centre> <bw> force` is its usage, and
#: demanding a log hold either demands the impossible of a report about a defect in
#: the script itself.
_A_TEMPLATE = re.compile(r"\$\{|<[a-z][a-z ]*>")
#: A run of spaces wide enough to be a column gutter, mid-line. A fenced block laid
#: out in columns is the report's own arrangement of figures rather than a
#: transcript -- the slot reports set their bandwidth and ack-placement summaries
#: that way -- and holding an arrangement to being verbatim is how a gate starts
#: firing on correct writing.
_A_COLUMN = re.compile(r"\S {3,}\S")
_LOG_NAMED = re.compile(r"`([\w][\w.-]*\.log)`")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s")
_INLINE_SPAN = re.compile(r"`([^`]+)`")
#: A report showing a line in order to say it did NOT appear. Narrow on purpose:
#: `_NO` over the whole sentence read a neighbouring "no orphaned workers", and a
#: table row's "never held" in the cell before, as denials of the span beside them.
_NOT_SEEN = re.compile(r"\b(?:did not|does not|never|no longer|has not|have not)\s+"
                       r"(?:\w+\s+){0,2}?"
                       r"(?:appear|appeared|print|printed|show|shown|come|came"
                       r"|arrive|arrived|fire|fired|occur|occurred)\b", re.I)


def _flat(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class Quotation:
    """A span a report sets as the program's own output.

    ``names`` is the log the quotation's own section attributes it to, and is None
    where the section names none or names several. Attribution by proximity is a
    guess everywhere else, and the report's log inventory names eleven at once.
    """
    report: Path
    line: int
    text: str
    names: str | None

    def cite(self) -> str:
        return f'{_where(self.report)}:{self.line}\n      "{self.text}"'


def _sections(text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """A markdown document as its headings divide it, lines numbered.

    Fences are tracked here as well as in `_spans`, because a `#` at the head of a
    quoted line is a comment and not a heading. The creance monitor's own banner is
    one -- `# creance monitor | kestrel, shrike, besra | window 20s` -- and cutting
    a section there leaves the opening fence in one half and the closing fence in
    the other, where it reads as an opening and inverts every fence after it. That
    turned the prose of two reports into quoted output and accounted for over a
    hundred of the offences the widened glob first reported.
    """
    out = [("", [])]
    fenced = False
    for n, raw in enumerate(text.splitlines(), 1):
        if _FENCE.match(raw):
            fenced = not fenced
        elif not fenced and _HEADING_LINE.match(raw):
            out.append((raw.strip(), []))
        out[-1][1].append((n, raw))
    return out


def _unwrapped(block: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """A fenced block's lines with the report's own wrapping undone.

    A log line wider than the page is re-wrapped where it is quoted, and the
    continuation is indented past the line it belongs to. Undoing that is what lets
    the comparison be exact, and exact is the point: fuzzy is how a compressed quote
    passes for a verbatim one.
    """
    lines = [(n, raw) for n, raw in block if raw.strip()]
    #: Only a line that reached the page could have been wrapped. Without this,
    #: any indented line reads as a continuation of the short one above it, and
    #: two consecutive log lines are quoted back as one that was never printed.
    wrapped_at = max((len(raw.rstrip()) for _, raw in lines), default=0) - 12
    out: list[list] = []
    for n, raw in lines:
        indent = len(raw) - len(raw.lstrip())
        if out and indent > out[-1][2] and len(out[-1][1]) >= wrapped_at:
            out[-1][1] += " " + raw.strip()
        else:
            out.append([n, raw.strip(), indent])
    return [(n, text) for n, text, _ in out if not _A_COLUMN.search(text)]


def _denied(paragraph: str, start: int, end: int) -> bool:
    """Whether the sentence holding this span says the line did NOT appear.

    "`*** Unknown client types are not allowed on production servers` -- the
    refusal that ended every prior session -- did not appear once" is a report
    doing precisely what this file asks of a document, and demanding that the line
    be in a log demands the opposite of what the sentence says. The span itself is
    blanked before the search, because half these lines are refusals and carry
    their own negation in the words being quoted.
    """
    masked = paragraph[:start] + " " * (end - start) + paragraph[end:]
    cuts = [0, *(b.start() for b in _BOUNDARY.finditer(masked)), len(masked)]
    return bool(_NOT_SEEN.search(
        masked[max(c for c in cuts if c <= start):min(c for c in cuts if c > start)]))


def _is_output(text: str) -> bool:
    return (len(text.split()) >= _QUOTE_WORDS
            and not _A_COMMAND.match(text)
            and not _A_PATH.match(text)
            and not _A_TEMPLATE.search(text)
            and any(c.islower() for c in text))


def _quotations(report: Path) -> list[Quotation]:
    """Every span a report sets as output, with the log its section names.

    ``_sentences`` above throws away exactly what this needs. There a fenced block
    is evidence being shown rather than a claim being made, and skipping it is
    right; here it is the claim.
    """
    out: list[Quotation] = []
    for _, lines in _sections(report.read_text(encoding="utf-8", errors="ignore")):
        named = {m.group(1) for _, raw in lines for m in _LOG_NAMED.finditer(raw)}
        log = next(iter(named)) if len(named) == 1 else None
        for at, text in _spans(lines):
            if _is_output(text):
                out.append(Quotation(report, at, text, log))
    return out


def _spans(lines: list[tuple[int, str]]):
    """(line, text) for every fenced line and every inline-code span in a section.

    Paragraphs are joined before the inline search because a quotation wraps across
    them: the phantom this was built for is one backticked span broken over two
    lines of prose, and per-line matching finds neither half.
    """
    fence: list[tuple[int, str]] | None = None
    skip, para = False, []

    def prose():
        joined, marks = "", []
        for n, line in para:
            if joined:
                joined += " "
            marks.append((len(joined), n))
            joined += line
        para.clear()
        for m in _INLINE_SPAN.finditer(joined):
            if _denied(joined, m.start(), m.end()):
                continue
            yield max(n for off, n in marks if off <= m.start()), _flat(m.group(1))

    for n, raw in lines:
        if _FENCE.match(raw):
            if fence is None:
                yield from prose()
                fence, skip = [], raw.strip("`~ \t").lower() in _LANGUAGES
            else:
                if not skip:
                    yield from ((at, _flat(t)) for at, t in _unwrapped(fence))
                fence = None
            continue
        if fence is not None:
            fence.append((n, raw))
        elif not raw.strip() or _HEADING_LINE.match(raw):
            yield from prose()
        else:
            para.append((n, _QUOTE.sub("", raw.strip())))
    yield from prose()
    if fence is not None and not skip:
        yield from ((at, _flat(t)) for at, t in _unwrapped(fence))


@pytest.fixture(scope="module")
def logs() -> dict[str, list[str]]:
    """Every log the reports are written off, a line at a time, whitespace flat.

    Keyed by the path from the repository root. Bare name stood here and collapsed
    901 files onto 333 keys, which is fine for "does any log hold this line" and
    wrong for "does the tree hold the log this report names": `--log NAME` reuses a
    name every slot, so a live arm answered a three-day-old report's citation of a
    log on the operator's desktop and turned seven unchecked quotations green.
    """
    return {p.relative_to(evidence.RECORD).as_posix():
            [_flat(ln) for ln in
             p.read_text(encoding="utf-8", errors="ignore").splitlines()]
            for p in sorted({p for g in LOG_TREES for p in evidence.RECORD.glob(g)})}


def _holds(lines: list[str], quote: str) -> bool:
    return any(quote in line or _abridged(quote, line) for line in lines)


def _abridged(quote: str, line: str) -> bool:
    """Every word of the quotation, in order, in one line of the log.

    Reports drop a log's date stamp and level from the middle of a line they are
    otherwise quoting faithfully, and a report that had to reproduce those to pass
    would just stop quoting. What this still refuses is a word that is not there and
    an order that was not the log's.
    """
    words, at = quote.split(), 0
    for word in line.split():
        if at < len(words) and words[at] == word:
            at += 1
    return at == len(words)


def _printable(quote: str) -> str:
    """Whether the package holds a format string that would produce this quotation.

    The tail, because the head is where the placeholder is: "2 turnarounds are
    witnessed" is `f"{witnessed} turnarounds are witnessed"` in the source and
    matches nothing whole.
    """
    tail = " ".join(quote.split()[-8:])
    for path in (p for tree in SOURCE for p in tree.rglob("*.py")):
        if tail in _flat(path.read_text(encoding="utf-8", errors="ignore")):
            return (f"{_where(path)} can print it. It is a line the program would "
                    f"say and has not said, so this is source read as record.")
    return "Nothing in the package can print it either."


#: A log a report names, however it was typed at the shell: bare, with the path it
#: was read from, or as the brace expansion that ran eight arms at once.
_A_LOG_NAMED = re.compile(r"`([^`\s]*?\.(?:log|narr))`")
_BRACE = re.compile(r"\{[^}]*\}")


def _kept(report: Path, logs: dict[str, list[str]]) -> set[str]:
    """The logs a report names that the tree actually holds.

    A citation carrying a directory is answered only from that directory. A bare
    name is still answered from anywhere, because that is how a log gets named at
    the shell -- but `~/pactor-day-02-ws8eoc.log` says the operator's desktop, and
    a same-named arm under `working/` three days later is not that file.
    """
    found = set()
    for m in _A_LOG_NAMED.finditer(report.read_text(encoding="utf-8",
                                                    errors="ignore")):
        cited = _BRACE.sub("*", m.group(1))
        found |= {w for w in logs
                  if fnmatch(w if "/" in cited else PurePosixPath(w).name, cited)}
    return found


@pytest.fixture(scope="module")
def reports() -> list[Path]:
    return sorted(evidence.WORKING.glob(REPORTS))


#: The reports whose logs never reached the tree, and where each one's are instead.
#: Named rather than counted, for the same reason as `KNOWN_PHANTOM`: a fourteenth
#: fails, and the entry comes out in the commit that keeps the log.
#:
#: This is what the widened glob really found. Of 374 quotations across twenty
#: reports, the seven that keep their logs here hold 106 and two of those are wrong.
#: These thirteen hold the other 268, and against them the quotation check has no
#: ground truth at all -- every line reads as a phantom because the log is somewhere
#: this tree cannot see. Calling those 268 offences would be a gate measuring the
#: operator's file layout and reporting it as the report's honesty.
KNOWN_UNKEPT = {
    # Sixteen logs, every one of them written to the home directory: the report
    # names them `~/day-00-listen-14108-1411z.log` and so on, and the run that
    # produced them left nothing under `working/`.
    "DAYTIME-REPORT-2026-08-22.md",
    "P3POWER-REPORT-2026-08-22.md",
    "PACTOR-REPORT-2026-08-26.md",
    "PACTOR-SLOT-REPORT-0826.md",
    "TONIGHT-6-REPORT.md",
    "VARA-MORNING-REPORT-0826.md",
    "VARA-REPORT-2026-08-22.md",
    "VARA-REPORT-2026-08-23.md",
    "VARA-REPORT-2026-08-26.md",
    # The 0826 slot reports. Untracked on the day this landed, which is a second
    # absence on top of the logs': a clone has neither the report nor its record.
    # The two ARDOP ones came off the list when their logs were kept, and the
    # PACTOR one when its keyed arm was found already in the tree under the dated
    # name `working/pactor-day-02-ws8eoc-2026-08-26.log`.
    "VARA-DAY-REPORT-0826.md",
}


def test_a_report_that_quotes_output_names_a_log_the_tree_holds(reports, logs):
    """The precondition the quotation checks below were resting on unstated.

    A report sets a line in backticks and says the program printed it. Whether that
    is true is decidable only where the log is here, and the check that follows
    cannot tell "this line was never printed" from "the log is on somebody's desktop"
    -- it reports both as a phantom. So the premise is checked first and by name.

    The repair is the one `tools/rigtest.sh` already makes: keep the log beside the
    recording, under `working/`, and name it in the report. Then the quotations get
    read, which is the whole point of writing them down.
    """
    if not (reports and logs):
        pytest.skip(_absent())
    unkept = [r for r in reports if _quotations(r) and not _kept(r, logs)]
    fresh = [r for r in unkept if r.name not in KNOWN_UNKEPT]
    assert not fresh, (
        f"{len(fresh)} report(s) quote the program's own output and name no log "
        f"this tree holds, so nothing can read the quotations back:\n\n"
        + "\n".join(f"  {_where(r)} — {len(_quotations(r))} quotation(s)"
                    for r in fresh)
        + "\n\nKeep the log where the report can point at it. A quotation whose log "
          "is on the operator's desktop is a figure the next reader has to take on "
          "trust, which is the thing this file exists to stop.\n\n"
          "Run the arm with `./tools/onair.sh --log NAME`, which writes "
          "`working/onair-<mmdd>-<hhmm>/NAME.log` itself, and name that FILE in the "
          "report -- `working/onair-0829-1254/vara-day-01-ke8lva.log`, not the "
          "directory it sits in and not the bare name. This matches on paths: a "
          "directory names no log, and `--log` reuses a name every slot, so a bare "
          "one answers with whichever run keyed last.")
    mended = KNOWN_UNKEPT - {r.name for r in unkept}
    if mended:
        pytest.fail(
            f"{len(mended)} report(s) KNOWN in this file for keeping no log now keep "
            f"one. Take each out of the list, in the commit that kept it:\n  "
            + "\n  ".join(sorted(mended)))


#: The offences on the night this landed, itemised rather than counted.
#:
#: A gate that fails on arrival is a gate that gets deleted, and these are somebody
#: else's field notes to repair. Itemised, because a count would let a new
#: misquotation take a repaired one's place -- and because the entry IS the finding:
#: each was checked by hand against the logs, and what it is wrong about is written
#: beside it. Repairing one fails this file too, which is the point: the line comes
#: out of the list in the commit that fixes the report.
KNOWN_PHANTOM = {
    # An elided traceback. `working/onair-0828-2057/ardop-night-03-kn4lqn.log`
    # carries the real one; the report cut the argv middle to `[...]` to fit a line.
    "subprocess.TimeoutExpired: Command '[... rigctl -m 2 -r 127.0.0.1:4532 "
    "\\dump_caps]' timed out after 8 seconds",
    # The line is real in `working/onair-0828-2058/ardop-night-04-kn4lqn.log`; the
    # report stripped the leading date, so the timestamp no longer matches a record.
    "21:03:33,363 INFO TX BREAK 0.47s",
    # In-slot analysis, not program output: burst arrivals differenced against the
    # tap's PTT edges by an ad-hoc script. Superseded -- the SETTLED block above it
    # records that this clock ran 0.6-0.7 s late, so the figures are the measurement
    # as taken and not the measurement as it stands.
    "arm5 N5WAJ fast: 46 bursts, 17 within 1.5 s of an unkey (37%) gaps (s): "
    "0.02, 0.36, 0.52, 0.81, 0.82, 0.83, 0.83, 0.86, 0.91, 0.93, 0.97, 0.99, "
    "1.05, 1.11, 1.14, 1.18, 1.28, 1.55",
    "arm4 N5WAJ slow: 6 bursts, 0 within 1.5 s (0%)",
    # The figures are the log's; the sentence is not. `clamped-force.log:245` reads
    # "TURNAROUND ACQUIRED @ sample 2552448: d = 92.2 ms after our data ends (edges
    # -0.8 ms), corroborated in cycle 43". No such format string exists.
    "TURNAROUND ACQUIRED d = 92.2 ms (edges -0.8 ms)",
    # A template set as a transcript: `tools/kestrel_connect.py` prints the attempt
    # number where the report put `n`. It also prints on the resend rather than on
    # the failure, so the seven lines mark CRs 2 through 8 going out.
    "no answer — resending CR (n/8)",
    # The separator alone, and the figure is the log's: "at 0.09 s and our counter
    # advanced #1 -> #2. Prediction met." is verbatim `clamped.log:285`, and 0.09 s
    # is the commonest reading on file -- 36 lines against 16 at 0.07 and one at
    # 0.10. The log writes `--` where the report set an em dash.
    "verdict: ACKNOWLEDGED — the peer alternated to CS2/ack at 0.09 s and our "
    "counter advanced #1 -> #2. Prediction met.",
    # No log holds this reading, `slot-20260816-autopilot.log` included.
    "receiver rms 0.09346 -> LIVE, not transmitting",
    # The 0828 and 0829 slot reports, reached for the first time by the commit that
    # kept their logs. Every one of these is an arrangement of lines that were
    # printed rather than a line that was, so the repair is in the report.
    # A range over three arms, set as one reading. The three logs of the slot read
    # "receiver -17.3 dBFS in band, rms 0.134 -> LIVE" at
    # `ardop-day-01-w6ids-check.log:14`, -17.7/0.200 at `-03-w6ids-forced.log:14` and
    # -17.8/0.134 at `-03-w6ids-pounce.log:19`.
    "−17.3 to −17.7 dBFS, rms 0.134–0.200 -> LIVE",
    # Two log lines joined by an arrow the log never printed.
    # `ardop-day-01-w6ids-check.log` holds the RX at 67 and the TX at 68, 71 and 72,
    # 75 and 78 — verbatim, one per line, each with the `sess=0x0d` the report drops.
    "18:56:17 RX 4PSK.200.100.E ok=False HEADER-ONLY q=53 -> TX DATANAK",
    "18:56:30 RX 4PSK.200.100.E ok=False HEADER-ONLY q=55 -> TX DATANAK",
    "18:56:43 RX 4PSK.200.100.E ok=False HEADER-ONLY q=53 -> TX DISC",
    # The same join with an ellipsis in place of `sess=0x0d ok=False`.
    # `ardop-day-03-w6ids-forced.log:52` holds the line whole.
    "19:07:00,589 INFO RX 4PSK.200.100.E ... HEADER-ONLY q=57",
    # A peer agent's command line and `tools/onair.sh:395`'s refusal. Both are real —
    # the second a `die` string in the shell, which `_printable` does not read — and
    # neither reached a log: the launch died before its arm log existed.
    "rigctld -m 1036 -r /dev/cu.no-rig-here-A",
    "rigctld did not answer on 4532 within 12 s",
    # The parenthetical compressed. `pactor-day-02-ws8eoc-bits2.log:204` reads
    # "connect candidates: 3 of 5 cycles searched, chance alone gives about 0.2
    # (cycle 2 at d = 95 ms, cycle 4 at d = 94 ms, cycle 5 at d = 96 ms) --
    # corroborated in cycle 5".
    "3 of 5 cycles searched (d = 95, 94, 96 ms) -- corroborated in cycle 5",
    # The report's own summary row, set inside the tap's burst table. The rows either
    # side of it are the table's and are exempt as columns; this one is prose.
    "... thirteen bursts on the 1.25 s raster, unbroken ...",
    # Two lines of `pactor-day-02-ws8eoc-bits2.log` joined at an ellipsis: the verdict
    # at 215 and the LINK DOWN clause at 192.
    "the peer never alternated its acknowledgement (7 CS1/CS2, 0 other) ... "
    "every one a repeat request",
    # A column row re-laid-out. `pactor-day-02-ws8eoc-bits2.log:59` reads
    # "RX 6.11 cs CS1/codeword search (0 bit errors, PACTOR-1, shift norma" — the log
    # truncates the word. The four rows below it keep a wide enough gutter to read as
    # columns and are exempt; this one does not.
    "6.11 CS1/codeword search shift normal",
    # Four separate lines, joined by `_unwrapped` because each is indented past the
    # one above. The gate's artefact rather than the report's: the lines are the
    # log's, one per line, and no single line was ever this long.
    "HOLD RX 22.35 cs CS3/at anchor (0 bit errors, PACTOR-1, shift inverted) "
    "TX[19] P1 CS1 LSB (0.1s) -- keying [host] rx CS BRK [host] changeover -> IRS",
    # A source citation, not a quotation: the sentence around it names
    # `arq.py:1200` and sets the expression it holds in the same marks a log line
    # gets. `_printable` reports it as source read as record, which is the right
    # reading of the span and the wrong one of the sentence.
    '_give_up("no decodable traffic from the peer")',
    # The same elision as the first entry, on one arm.
    # `pactor-day-01-kb5lzk.log:15` reads "receiver -9.8 dBFS in band, rms 0.276 -> LIVE".
    "-9.8 dBFS, rms 0.276 -> LIVE",
    # The 0826 ARDOP reports, reached by the same commit.
    # `LeaderDetects` abbreviated. `ardop-day-06-k5vp.log:54` reads "RX funnel over
    # 25.8 s: LeaderDetects=165 (+75 Hz off tune) Good Frame Type Decodes=0 ...".
    "LD=165 (+75 Hz off tune)",
    # The same abbreviation with the middle of the line cut out.
    # `ardop-day-03-k0rvw.log:52` reads LeaderDetects=0, then Good Frame Type
    # Decodes=0, then the 2563.
    "LD=0 Failed Frame Type Decodes=2563",
    # `tools/txwitness.py edges` printed this to the terminal and the slot kept no
    # log of it; the report also broke the one line across two. Re-running `edges`
    # against the tap and keeping the output is the repair.
    "median of 14: body from 142.0 ms after key-up (14.2 bits at 100 Bd) to "
    "1868.0 ms",
    "after it, receiver back 136.0 ms after that",
    # `ardop-night-01-kn4lqn.log:39-41`, verbatim and in order. `_unwrapped` joined
    # them because each is indented past the one above: the gate's artefact rather
    # than the report's.
    "2026-08-25 21:24:43,048 INFO RX DATAACK sess=0xac ok=UNVERIFIED theirq=80 "
    "CONNECTED KN4LQN @ 2000 Hz ARQ CONNECTION ESTABLISHED WITH KN4LQN, "
    "SESSION BW = 2000 HZ",
    # Three arms' readings collapsed into one line. Each is verbatim at line 12 of
    # its own log: arms 01, 02 and 03 of `ardop-night-0826/`.
    "channel sense: -3.2 / -3.1 / -1.9 dB shape -> clear",
    # Six readings collapsed. Each is its own arm's
    # "receiver -NN.N dBFS in band, rms ... -> LIVE".
    "-16.2, -14.4, -15.6, -15.1, -13.0, -12.5 dBFS in band",
    # ----------------------------------------------------------------------- #
    # Reached when the 0826-0830 slot reports were made to name their arm logs.
    # Twenty of the thirty-seven that surfaced were the report compressing a line
    # the log holds and are repaired in the same commit; these are the rest, and
    # every one of them is a run whose output nobody kept. That is the shape to
    # watch: not a report inventing a figure, a report quoting a terminal.
    #
    # `tools/rehear` against arm 05's own session log, at the terminal. The first
    # is `rehear/__init__.py:177` verbatim; the second joins that run's
    # "live 3, ungated 3" to a `discarded` figure only the `totals` line carries.
    # Re-running it with the output kept under `working/` is the repair.
    "reproduces 5 of the 6 frames this session logged",
    "live 3, ungated 3, discarded 0",
    # Arm 1 of the 0826 PACTOR slot -- the refused arm, 18 lines, no transmission.
    # Its log is `~/pactor-day-01-ws8eoc.log` and never reached the tree, so this
    # is real output of a run this repository cannot show. Arm 2's log is here and
    # the report's quotations off it all read back. It drops the line's trailing
    # " of live audio".
    "burst 10.28/6.6, shape 13.35/6.0, tone 5.39/5.0 at 2613 Hz over 7.9 s",
    # The propagation feed's re-derive and its census, both printed at the
    # terminal while the operator picked the target. No arm had started, so there
    # was no arm log for them to land in.
    "7101.5 kHz 0.33 h ago WS8EOC <- N9NIC Pactor 3 63 s",
    "looked back 24 h: 2027 records, 136 PACTOR, 116 in 3–11 MHz, 66 within "
    "2600 mi, 60 audible — freshest 0.04 h ago",
    "986 records, 111 PACTOR, 88 in 3-11 MHz, 37 within 2600 mi, 35 audible",
    # `tools/txwitness.py edges` at the terminal, the same absence as the 0828
    # entry above. Re-running `edges` against the tap and keeping the output is
    # the repair.
    "body from 51.0 ms after key-up (5.1 bits at 100 Bd) to 995.0 ms after it, "
    "receiver back 1.0 ms after that",
    # A bench run with the rig down, kept nowhere. `core/cbtrace.py:143` prints
    # "adc step samples: median 128.0  max 128.1" -- the report also dropped the
    # "samples:", so even against its own output this is not verbatim.
    "adc step median 128.0 max 128.1",
    # A row of the runbook's codec-gain table, not a line anything printed. The
    # report is arguing that the row is unsafe to carry across an evening, and
    # `working/PACTOR-DAY.md:179` has since been rewritten to say so.
    "0.209 | daylight, 30 m",
    # `kiwi_witness` choosing a receiver, at the terminal. The exception is real
    # -- `tools/kiwi_witness.py:65` records the same attempt -- and the selection
    # ran outside any arm, so no arm log holds it.
    "KiwiTooBusyError: all 2 client slots taken",
    # In-slot analysis: an ad-hoc alignment of the arm 9 recording against
    # `CONTROL_BURST_RESPONDER_2300`, laid out as two labelled rows. Nothing
    # prints it, and the finding it carries is the report's own arithmetic.
    "read tail: (34,80) (83,97) (50,66) (59,95) (64,96) (59,77) (57,72)",
    "reference: (34,79) (80,82) (50,66) (59,95) (64,96) (59,77) (57,72)",
    # The 2026-08-26 bench corpus measured offline over
    # `working/vara/oracle/logs/audio/` -- 8 sessions, 17 instances, differenced
    # by hand. The block is the measurement written out, not a transcript of one.
    "responder DATA over, rec3, 395 columns, 4.376 s",
    "+0.155-0.169 s CALLER keys the 11-symbol two-tone control burst",
    "+0.089-0.098 s RESPONDER keys SESSION_TURN_RELEASE_RESPONDER, 17/17 tones",
    "+0.075-0.126 s CALLER keys its next WIDEBAND rec3 DATA OVER",
    # A B2F continuation composed at the bench to drive `B2FSession._turn_line`,
    # showing that `;PM:` alone leaves the session at `their turn` while an
    # `FC`/`F>` over moves it to `FS +`. The article off the air reads
    # `mail: peer sent: FC EM JWKY65C2OZES 643 524 0` and carries other byte
    # counts, so this is the input to a bench run and not a line from one.
    "FC EM JWKY65C2OZES 524 400 0\\rF> xx\\r",
    # A `TIOCMGET` read at the terminal on a freshly re-plugged interface, which
    # is how the report establishes that opening the PTT port asserts RTS.
    # Nothing in the station prints the modem lines, so there is no run to keep
    # the output of; the repair is a tool that prints them.
    "modem bits: 0x6 RTS=True DTR=True",
}

#: The same, for a quotation that exists in a log other than the one its section
#: names.
KNOWN_MISPLACED = {
    # Read off `hold24-force.log:260` under a heading naming
    # `pactor-ws8eoc-40m-clamped-force.log`, whose own tally at line 505 is 10
    # PACTOR-1 and 0 PACTOR-3. The section it is arguing for is about the other
    # run.
    "24 PACTOR-1 at zero errors, 0 PACTOR-3",
}


def _ratchet(offences: list[Quotation], known: set[str], *, what: str, detail,
             how: str) -> None:
    """Refuse a new offence, and refuse a repaired one still on the list."""
    fresh = [q for q in offences if q.text not in known]
    assert not fresh, (
        f"{len(fresh)} line(s) {what}:\n\n"
        + "\n\n".join(detail(q) for q in fresh) + f"\n\n{how}")
    mended = known - {q.text for q in offences}
    if mended:
        pytest.fail(
            f"{len(mended)} of the quotations KNOWN in this file no longer offends. "
            f"Take it out of the list, in the commit that repaired the report:\n  "
            + "\n  ".join(sorted(mended)))


def _absent() -> str:
    return ("NO WORKING REPORTS OR NO LOGS under "
            f"{evidence.WORKING}, so nothing here is checking whether a report "
            "quotes output that was never printed. Both are written by a run and "
            "both are in the index, so this is a distribution -- the manifest "
            "denies them -- or a tree somebody pruned. Only this station can "
            "produce more.")


def test_every_line_a_report_quotes_as_log_output_is_in_a_log(reports, logs):
    """A quotation no log holds anywhere.

    This is the one that dispatched work: a line set as the program's own output,
    read as a measurement of the session, and never printed by anything.

    Over the reports whose logs are here, which is the precondition the check above
    holds the rest of them to. Not a narrowing to keep this green: against a report
    whose log is elsewhere this reads every line as a phantom, and 268 such lines
    would bury the two that are real.
    """
    if not (reports and logs):
        warnings.warn(_absent(), stacklevel=2)
        pytest.skip(_absent())
    _ratchet(
        [q for r in reports if _kept(r, logs) for q in _quotations(r)
         if not any(_holds(lines, q.text) for lines in logs.values())],
        KNOWN_PHANTOM,
        what=f"are quoted as log output and appear in none of the {len(logs)} logs "
             f"under {evidence.RECORD}'s working/, logs/ and captures/",
        detail=lambda q: f"  {q.cite()}\n\n  {_printable(q.text)}",
        how="A quotation is either verbatim or is not a quotation. Compressing a "
            "log line to fit a column, or writing down what the source would say, "
            "produces a sentence the next reader greps for and cannot find.\n\n"
            "Three repairs, in order of preference. QUOTE LESS: a verbatim "
            "fragment of the line passes and a summarised whole does not, so cut "
            "the ellipsis and keep the run of words the log actually printed. KEEP "
            "THE OUTPUT: if the figure came off a tool you ran at the terminal "
            "rather than off an arm, re-run it with the output under `working/` "
            "and cite that. Or SAY IT IS YOURS: analysis, arithmetic and a "
            "reconstruction are all fine in a report -- write them as your own "
            "measurement instead of setting them in the marks that claim a program "
            "said them, and add the line to KNOWN_PHANTOM above with what it is. "
            "Deleting the citation so this stops asking is the one move that is "
            "not a repair.")


def test_every_line_a_report_attributes_to_a_log_is_in_that_log(reports, logs):
    """A quotation that exists, under a heading naming the log it did not come from.

    Only where the section names exactly one log, which is the only place a report
    says which run a figure is from. Narrow on purpose: it is the difference between
    a gate and a guess, and the columns of a comparison table are named by flag.
    """
    if not (reports and logs):
        pytest.skip(_absent())

    #: A heading names a log bare, so the attribution is resolved bare. Where two
    #: runs left the same name the later one answers, which is the reading a
    #: heading naming no directory can support.
    named = {PurePosixPath(w).name: lines for w, lines in logs.items()}

    def elsewhere(q: Quotation) -> str:
        holders = [w for w, lines in logs.items() if _holds(lines, q.text)]
        return f"; {', '.join(holders)} does" if holders else ""

    _ratchet(
        [q for r in reports if _kept(r, logs) for q in _quotations(r)
         if q.names in named and not _holds(named[q.names], q.text)],
        KNOWN_MISPLACED,
        what="are quoted under a heading that names one log, and are not in it",
        detail=lambda q: (f"  {q.cite()}\n      attributed to {q.names}, which "
                          f"does not hold it{elsewhere(q)}"),
        how="A figure carried across from another run reads as a comparison and "
            "is not one.")


# --------------------------------------------------------------------------- #
# The third direction: an absence claim, against the command that would settle it.
#
# THE CONVENTION. An absence claim carries the command that establishes it.
# "Nothing in the tree does X" and "`grep -rn X packages/` returns nothing" are
# different statements, and only the second can be read a second time. The first
# is the half of this failure mode that does the damage, because it stops the
# search: nobody greps for what they have been told is not there, so a negative
# that has quietly stopped being true is the one thing in a document that nothing
# -- not a reader, not a later writer, not this file's other two checks -- will
# go and look at.
#
# `README.md` is the model, and the commit that finally got it right states the
# moral: in a paragraph a reader is invited to verify, any enumeration is a
# hostage, so write a property and the command that checks it rather than a
# count. Two riders. A date is not a command -- "verified 2026-08-11" says
# somebody looked, not what to type. And a command has to be one a reader can
# afford: `grep -rn hashlib` from the repository root does not finish inside
# 20 s on this station, because `captures/` is 5 GB of recordings, so a command
# that means the tree says the tree.
#
# WHAT IS ENFORCED IS ONE CLAUSE OF THAT, AND DELIBERATELY THE THIN ONE. Absence
# claims cannot be picked out of this tree's prose by machine, and the measuring
# was done before the rule was written rather than after. Over every comment and
# docstring in `packages/`, a rule keyed on a negation near a searchable body --
# tree, package, log, capture, corpus, record -- flags 49 passages, of which 42
# are correct writing: "nothing is gated", "no payload knowledge anywhere in the
# path", "nothing in any log would have". A rule keyed on the searching words
# flags 328, because in a modem a search is a codeword search and a sweep is a
# noise sweep. Four in five wrong is a gate somebody switches off inside a week,
# and then nothing is enforcing anything.
#
# What is decidable is the narrowest honest thing: a document that says it
# searched and does not say with what. `grepped` has no second sense here. Over
# the same corpus this fires five times; two carry their command and pass, and
# the three that did not are repaired in the commit that added this. No correct
# passage is flagged, which is the bar the rest of the file is held to.

#: A claim to have searched. Past tense or a result, never the imperative: "grep
#: `tools/` for it" is an instruction to a reader and has nothing to be right or
#: wrong about.
_A_SEARCH_REPORTED = re.compile(
    r"\bgrep(?:p)?ed\b"
    r"|\bgrep\b[^.;]{0,90}?"
    r"\b(?:finds?|found|returns?|returned|shows?|showed|comes? back|came back"
    r"|turns? up)\b[^.;]{0,40}?"
    r"\b(?:no|nothing|none|only|every|zero|clean|empty)\b",
    re.I)

#: ...and the command that would let a reader run it again. A tool with an
#: argument: `git ls-files` on its own names a tool, not a search.
_A_SEARCH_COMMAND = re.compile(
    r"^(?:[A-Z_][A-Z0-9_]*=\S*\s+)*"
    r"(?:grep|rg|ripgrep|git\s+(?:grep|ls-files|log)|find|ls|comm)\b"
    r"(?:\s+-{1,2}[\w-]+)*\s+\S")


def _carries_the_command(text: str) -> bool:
    return any(_A_SEARCH_COMMAND.match(m.group(1).strip())
               for m in _INLINE_SPAN.finditer(text))


@pytest.fixture(scope="module")
def documented() -> list[tuple[Path, str]]:
    """Every markdown document this station writes about itself in.

    Wider than `shipped`, which is scoped to what crosses the publication
    boundary. An absence claim does its damage where work is dispatched off it,
    and that is the field notes as much as the published page.
    """
    #: The scratch tree as THIS checkout has it. `evidence.WORKING` resolves
    #: against the checkout a worktree was made from, which is right for the
    #: recordings and wrong for the documents: those are in the index, so a
    #: worktree carries its own and it is the one under test.
    notes = REPO / evidence.WORKING.relative_to(evidence.RECORD)
    return [(p, p.read_text(encoding="utf-8", errors="ignore"))
            for p in sorted([*REPO.glob("*.md"), *REPO.glob("docs/*.md"),
                             *notes.glob("*.md")])]


def test_every_search_a_document_reports_carries_the_command(documented):
    """A document that says it searched, and does not say with what."""
    offences = [(f"{_where(path)}:{line}", sentence)
                for path, text in documented
                for line, sentence, para, heading in _sentences(text)
                if _A_SEARCH_REPORTED.search(sentence)
                and not _carries_the_command(para)]
    assert not offences, (
        f"{len(offences)} sentence(s) report a search and withhold it:\n\n"
        + "\n\n".join(f'  {where}\n      "{sentence}"' for where, sentence in
                      offences)
        + "\n\nAn absence claim carries the command that establishes it. "
          "\"I did not find X with `grep -rn X packages/`\" and \"X does not "
          "exist\" are different statements and only the first can be read a "
          "second time; the second stops the search, which is how a negative "
          "outlives the thing that made it true.")


#: Real sentences from this tree that report a search and carry it. The rule has
#: to pass these: it is the whole of what the convention asks for, and a rule
#: that flagged the writing it is trying to produce is worse than no rule.
CITED = (
    "Hash functions appear, and none of them secures traffic; `grep -rn "
    "hashlib` finds every use, in the test suites as well as the code they "
    "exercise.",
    "**Nothing in the tree links to it.** Not the README, not `ARCHITECTURE.md`, "
    "not `docs/STATION.md` — a `grep -r RUNBOOK` returns only its own title line.",
    "If `ls /dev/cu.usbserial*` finds nothing, the interface is still unplugged "
    "or the radio is off.",
)

#: ...and the ones that did not, as they stood. The counterweight: a rule
#: narrowed until nothing innocent trips it reports a clean tree and means
#: nothing by it. Each of these named a real search whose pattern, whose domain
#: or both went unwritten, so the only way to check any of them was to guess
#: what had been typed and where. The third is trimmed rather than quoted whole:
#: what it searched for is a publication marker, and `test_manifest.py` counts
#: those wherever they appear, a fixture included.
UNCITED = (
    "`varahf500._load()` and `varahf2300._load()` build every table from "
    "`tablegen.*`; grepped, nothing in the package `np.load`s any deleted asset.",
    "Grepped clean: no \"breakpoint\", \"emulation\", or \"lldb\" anywhere in "
    "the shipping file.",
    "The GOLDEN entries now explain what an ABSENT verdict means and what flips "
    "it, naming no prompt file (grepped: no such reference in the module; the "
    "`abort_prompt` probe is protocol vocabulary, not workflow).",
)

#: Correct writing that shares the vocabulary. The words this tree spends on a
#: codeword search and a noise sweep are why the rule is keyed on `grep` alone,
#: and these are the passages that settled that.
NOT_A_SEARCH = (
    "A capability probe that greps another repo's source is absence in disguise: "
    "when that path moves the probe returns False, the stage it gates is never "
    "recorded, and `require` has nothing to match.",
    "Before writing anything that feels general, grep `tools/` for it.",
    "That is the argument for the second reader, not for a longer grep list.",
    "Two causes, both now fixed: the search stopped for a whole window whenever "
    "neither of its two energy gates fired.",
    "Windows 14 and 20's single live detections do not survive: shrike named "
    "P1-BURSTs at +5.5 and +5.0 dB over guard; the deep pass on the same audio "
    "finds nothing in either.",
)


@pytest.mark.parametrize("passage", CITED, ids=range(len(CITED)))
def test_a_search_that_carries_its_command_passes(passage):
    assert _carries_the_command(passage)


@pytest.mark.parametrize("passage", UNCITED, ids=range(len(UNCITED)))
def test_a_search_that_withholds_its_command_is_caught(passage):
    assert _A_SEARCH_REPORTED.search(passage) and not _carries_the_command(passage)


@pytest.mark.parametrize("passage", NOT_A_SEARCH, ids=range(len(NOT_A_SEARCH)))
def test_the_rule_does_not_read_a_search_into_prose_that_reports_none(passage):
    assert not _A_SEARCH_REPORTED.search(passage), passage


# --------------------------------------------------------------------------- #
# The fourth direction: a standing negative, against the record that settles it.
#
# WHAT THE FIRST SECTION CANNOT DO. `CLAIMS` above knows five things and is right
# about all five, and it is retrospective by construction: each entry is a claim
# this project has already got wrong in public, paired by hand with the machine
# token that refutes it. There is no mechanical route from an English negative to
# the line in a log that disproves it -- somebody supplies that mapping, one claim
# at a time, after the error. So it would not have caught the day's own findings,
# and two of those were exactly its shape: `README.md` saying kestrel's turn law
# "has never been validated" against a real station while seven byte-exact
# deliveries out of a stock VARA's data port sat in the tree, and a slot report
# saying `16QAM.500.100` had never decoded when the rejects it read as failures
# were the demodulator's own sentinel.
#
# WHAT GENERALISES IS NOT THE TRUTH BUT THE CONVENTION, and it is the one already
# enforced above for searches: an absence carries what establishes it. A standing
# negative -- not "the peer did not answer on 2026-08-14", which is a session, but
# "nothing besra has transmitted has ever been decoded", which is asserted without
# end -- either names the record a reader can open or it does not, and that is
# decidable without knowing whether the claim is true. It is also prospective: this
# fires on tomorrow's negative without anybody having to be wrong about it first.
#
# THE MEASURING, done before the rule rather than after, and it is why this reaches
# only the published documents. Over `packages/` a rule of this shape was measured
# at 49 flags to 42 correct passages and abandoned, for good reason -- in a modem a
# negation near a searchable body is usually about a signal. The published corpus is
# a different population: nine standing negatives in all of `*.md` and `docs/*.md`,
# five of them uncited. `docs/RUNBOOK.md`'s `HUPCL` sentence passes because it names
# `docs/STATION.md` and the bench procedure in it; `docs/ONAIR-READINESS.md`'s
# advance sentence passes on the scope guard it already carried. Run against
# `README.md` as it stood the morning of 2026-08-26, this flags the turn-law
# sentence, which is the one that shipped in the wheel's long description.

#: Asserted without end, which is what separates this from the session records the
#: whole tree is written in. `never` in the past perfect, or a negative subject with
#: `has ever` inside a clause of it.
_A_STANDING_NEGATIVE = re.compile(
    r"\b(?:has|have|had)\s+never\b|\bnever\s+(?:been|yet)\b"
    r"|\b(?:no|nothing|none)\b[^.]{0,80}?\b(?:has|have)\s+ever\b", re.I)

#: ...and the record that settles it: a path into the tree a reader can open, or the
#: search command the section above already asks for. A bare filename is not one --
#: `verdict.txt` names a hundred files -- so a separator is required.
_A_RECORD_CITED = re.compile(r"^[\w][\w./-]*/[\w][\w./-]*\.(?:md|py|log|sum|txt|wav)$")

#: A defect written down with the date it was repaired beside it is not asserted
#: without end -- it is a bounded account of something over, and the sentence a
#: reader would go and check is answered two lines later. Narrower than `_HISTORY`
#: and separate from it on purpose: `_exempt` is shared with the five enumerated
#: claims above, and each of those is paired with a machine token in a log that a
#: fix note does not touch.
_ITS_OWN_FIX = re.compile(r"\b(?:fixed|repaired|closed)\s+(?:on|in)\s+20\d\d-\d\d-\d\d",
                          re.I)


def _carries_the_record(text: str) -> bool:
    return any(_A_RECORD_CITED.match(span) or _A_SEARCH_COMMAND.match(span)
               for span in (m.group(1).strip()
                            for m in _INLINE_SPAN.finditer(text)))


#: The standing negatives published without their record on the day this landed.
#: Itemised on the same terms as `KNOWN_PHANTOM`: a sixth fails on sight, and an
#: entry comes out in the commit that cites it. A distinctive clause of each, so
#: that reflowing a paragraph does not silently retire one.
#:
#: Empty since 2026-09-09, when the last three went. The FCC sentence left with the
#: regulatory section that carried it. `no message body has ever been carried` had
#: become false: `logs/mail/JWKY65C2OZES.b2f` is 643 bytes fetched over a live VARA
#: link. The sabir clause was narrowed rather than dropped -- two public receivers
#: decoded the beacon, so what survives is the *exchange*, and README's row now
#: cites `docs/protocols/sabir/ONAIR.md` for why a WebSDR cannot supply one.
KNOWN_UNCITED: set[str] = set()


def _uncited(docs) -> list[tuple[str, str]]:
    return [(f"{path.relative_to(REPO)}:{offset + line}", sentence)
            for path, text, offset in docs
            for line, sentence, para, heading in _sentences(text)
            if _A_STANDING_NEGATIVE.search(sentence)
            and not _THE_STRETCH_GOAL.search(sentence)
            and not _exempt(sentence, para, heading)
            and not _ITS_OWN_FIX.search(para)
            and not _carries_the_record(para)]


def test_every_standing_negative_a_shipped_document_makes_carries_its_record(shipped):
    """A published document asserting that something has never happened, and not
    saying where a reader would find out.

    The generalisation of the five enumerated claims at the top of this file, and
    the only part of them that generalises. It cannot tell a true negative from a
    stale one -- nothing mechanical can -- but it can refuse the form in which a
    stale one survives, which is the form with nothing to check it against.
    """
    offences = _uncited(shipped)
    fresh = [(where, sentence) for where, sentence in offences
             if not any(known in sentence for known in KNOWN_UNCITED)]
    assert not fresh, (
        f"{len(fresh)} shipped sentence(s) assert that something has never happened "
        f"and name nothing that settles it:\n\n"
        + "\n\n".join(f'  {where}\n      "{sentence}"' for where, sentence in fresh)
        + "\n\nA standing negative carries the record that establishes it: the "
          "report, the log, the verdict or the search. A negative with nothing "
          "beside it is the one sentence in a document that nothing goes back to "
          "look at, which is how it outlives the thing that made it true.")
    mended = {k for k in KNOWN_UNCITED
              if not any(k in sentence for _, sentence in offences)}
    if mended:
        pytest.fail(
            f"{len(mended)} of the negatives KNOWN in this file now carries its "
            f"record, or is gone. Take it out of the list, in the commit that "
            f"settled it:\n  " + "\n  ".join(sorted(mended)))


#: Standing negatives in this tree that carry their record, one scoped to the
#: stretch goal, and one that dates its own repair. The rule has to pass these: it
#: is the whole of what the convention asks for, the second is the sentence the
#: first section of this file exists to protect, and the third is a defect named in
#: the past tense — the form the data-frame claim was repaired into.
GROUNDED = (
    "`HUPCL` is set for it and this adapter has never been asked whether it honours "
    "it (`docs/STATION.md`, \"What is not proven\", which gives the bench "
    "procedure).",
    "Nor has any link sustained the advance: three consecutive per-cycle advances "
    "have never happened, on our counter or the peer's, and the exchange ends after "
    "pkt#3, whose break-in bit invites the changeover.",
    "It has already cost: `[audio] tx_drive` was declared in the config, "
    "documented in the example station file, wired to the arbiter — and reached "
    "nothing that has ever been transmitted. An operator could set it, see no "
    "change, and reasonably conclude drive was not their problem. Fixed on "
    "2026-08-16 by resolving the drive in each keying path, which closed the "
    "symptom and left the fork.",
)

#: ...and the one that shipped in the wheel's long description, as it stood on the
#: morning of 2026-08-26. Seven byte-exact deliveries out of a stock VARA HF 4.9.0's
#: data port were in the tree when this was published.
SHIPPED_AND_UNGROUNDED = (
    "The turn *scheduler* exists and is tested; what is undecoded is the turn law's "
    "**semantics** — the answer to a turn request is read and logged and never "
    "required, so acceptance is positional and has never been validated against a "
    "real station.",
)


@pytest.mark.parametrize("passage", GROUNDED, ids=range(len(GROUNDED)))
def test_a_standing_negative_that_carries_its_record_passes(passage):
    assert not _uncited([(REPO / "README.md", passage, 0)])


@pytest.mark.parametrize("passage", SHIPPED_AND_UNGROUNDED,
                         ids=range(len(SHIPPED_AND_UNGROUNDED)))
def test_the_standing_negative_that_shipped_is_caught(passage):
    assert _uncited([(REPO / "README.md", passage, 0)])
