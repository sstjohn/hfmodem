# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Replay a recording through our own receive path twice, and report the difference.

Once with the live gating and mute handling, once ungated. The recording is
ground truth; the live path is the unit under test, and what the second pass
hears and the first does not is what the live path destroyed.

Doing this by hand has contradicted the live log four separate times, each time
in our own disfavour — a post-transmit deafness overstated 6x, a 100 ms cursor
skip inside the answer window, seven recovered gateway answers, an 80 ms leader
clip. Every one of those was found because somebody re-decoded a recording.
Standing, it finds the next one before a rig slot is spent on it.

Three things the reader has to be told, and which every adapter here declares
rather than implies:

``zero``
    What every time in the report is measured from. Three numbers with three
    different zeros is how a 6x error survived for months, so no adapter reports
    a time without saying where it starts counting.

``recorder``
    Whether the recorder sits UPSTREAM of the live path's drops — in which case
    the recording holds what the decoder was denied, and the difference is a
    measurement — or DOWNSTREAM, in which case the recording *is* what the
    decoder read and no amount of replaying it can show a sample that never
    reached the file. Saying which is the difference between a measurement and a
    zero that means nothing.

``uncovered``
    The frame types this pass cannot see. An adapter that reports "no difference"
    while declining to look at three quarters of the frame types has reported
    nothing.

``verdicts``
    What each ungated hit IS. A difference between two decoders is a loss only
    where the second one is the live path shown more audio; where it is a freer
    search shown the same audio, the difference is mostly the search. Twice this
    week a `discarded` column was read as a receive-gating lead and twice it was
    withdrawn — once on fourteen hits of which four were in the session's own log
    at the same second and ten sat inside the peer's own packet — so an adapter
    that can place its capture against the station's record says which of the
    three each hit is, and `discarded` then carries only the third.

``deaf``
    Where the replay hears less than the station did, and by how much. A recording
    of somebody else's QSO comes with no session log and is replayed without the
    session id the live decoder was polling; that is the honest reading and it is
    half the frames short of one, so it is announced rather than assumed. A replay
    that carried everything the live path had says nothing here.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from consolelib import residency

from hfmodem.station import sessionlog

UPSTREAM = "upstream"
DOWNSTREAM = "downstream"


@dataclass(frozen=True, slots=True, order=True)
class Heard:
    """One thing a receive path decoded, at a time in the recording."""
    at: float
    what: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.at:8.2f}  {self.what}" + (f"  {self.detail}" if self.detail else "")


#: The three readings an ungated hit can get where the adapter can place the
#: capture against the session's own record of the same instant. `SEEN` is a hit
#: the station reported too, `IN_PACKET` one that sits inside a transmission of
#: the peer's that this capture holds whole — a false accept by construction, and
#: not a thing the peer keyed — and `UNSEEN` is what is left, which is the only
#: class that can be a loss and the only one worth a rig slot.
SEEN = "seen"
IN_PACKET = "in a packet"
UNSEEN = "unseen"


@dataclass(frozen=True, slots=True)
class Verdict:
    """What an ungated hit is, and the evidence for saying so."""
    what: str
    why: str


@dataclass(slots=True)
class Rehearing:
    recording: Path
    protocol: str
    seconds: float
    zero: str
    recorder: str
    live: list[Heard] = field(default_factory=list)
    ungated: list[Heard] = field(default_factory=list)
    #: What each ungated hit is against the station's own record of the same
    #: instant. `None` where the adapter cannot place the capture on the session's
    #: clock at all, and then `discarded` stands — which is a measurement only
    #: where the recorder is UPSTREAM of the drops. An empty dict is a different
    #: statement, and the two must not read alike: it says the capture WAS placed
    #: and held nothing to read.
    verdicts: dict[Heard, Verdict] | None = None
    #: Live hits from a reader that is AIMED at an instant rather than sweeping
    #: for one. `only_live` is a check on two runs of the same search and cannot
    #: hold these: a point read at the grid's own instant reaches bursts no
    #: alignment the sweep tries lands on, and 44 of `onair-0902-2342`'s
    #: changeover heads are exactly that.
    aimed: set[Heard] = field(default_factory=set)
    #: What this station transmitted, where the recorder is UPSTREAM of the
    #: half-duplex mute and the capture therefore holds our own keyed intervals.
    #: Neither arm's business — both exist to exclude it — and the only reading
    #: of our own transmitter that costs nothing to take.
    ours: list[Heard] = field(default_factory=list)
    slack_s: float = 0.5
    uncovered: str = ""
    #: What this replay could not hear that the station could — empty only where it
    #: was given everything the live path had.
    deaf: str = ""
    #: `(reproduced, logged)` against the frames the session's own log reports,
    #: where a log is to hand: the instrument's verdict on itself.
    against_log: tuple[int, int] | None = None
    note: str = ""
    capture_timing: dict = field(default_factory=dict)

    def append(self, other, offset):
        """Retain source-relative event times across independently decoded spans."""
        def shifted(h):
            return Heard(h.at + offset, h.what, h.detail)
        self.live.extend(shifted(h) for h in other.live)
        self.ungated.extend(shifted(h) for h in other.ungated)
        self.ours.extend(shifted(h) for h in other.ours)
        self.aimed.update(shifted(h) for h in other.aimed)
        if other.verdicts is not None:
            if self.verdicts is None:
                self.verdicts = {}
            self.verdicts.update({shifted(h): v for h, v in other.verdicts.items()})
        if other.against_log is not None:
            old = self.against_log or (0, 0)
            self.against_log = tuple(a+b for a,b in zip(old, other.against_log))
        self.deaf = other.deaf
        self.uncovered = other.uncovered
        self.note = other.note

    def discarded(self) -> list[Heard]:
        """In the audio and not in the live decode.

        A loss where the recorder is UPSTREAM of the drops. Where it is not, this
        is the difference between two decoders and `verdicts` is what says which
        of them is the deafer — so an adapter that fills `verdicts` empties this.
        """
        if self.verdicts is not None:
            return [h for h, v in self.verdicts.items() if v.what == UNSEEN]
        return [h for h in self.ungated if not _near(h, self.live, self.slack_s)]

    def by_verdict(self, what: str) -> list[Heard]:
        return sorted(h for h, v in (self.verdicts or {}).items() if v.what == what)

    def only_live(self) -> list[Heard]:
        """The other direction, which is never good news and is always reported.

        An ungated pass is allowed to be more permissive — that is the point —
        but it is not allowed to be *different*. Anything here is either a live
        accept the ungated pass rejects, in which case one of the two is wrong,
        or a replay that does not model the live path as closely as it claims.

        There is a third reading where the recorder sits upstream of a half-duplex
        mute, and `against_log` is what tells it apart: the ungated arm is shown our
        own carrier as well, and acquisition steps past a frame by its length, so a
        transmission of ours can eat the window a peer's frame sits in. Two of the
        51 frames of `20260826T133925Z` come back this way — an E data frame at
        145.65 s and the END at 265.13 — and the session's own log reports both, so
        it is the ungated arm falling short and not the gated one inventing.
        """
        return [h for h in self.live if h not in self.aimed
                and not _near(h, self.ungated, self.slack_s)]

    def as_dict(self) -> dict:
        return {
            **({'capture_timing': self.capture_timing} if self.capture_timing else {}),
            "recording": str(self.recording), "protocol": self.protocol,
            "seconds": round(self.seconds, 2), "zero": self.zero,
            "recorder": self.recorder, "uncovered": self.uncovered,
            "deaf": self.deaf, "against_log": self.against_log, "note": self.note,
            "live": [[h.at, h.what, h.detail] for h in self.live],
            "ungated": [[h.at, h.what, h.detail] for h in self.ungated],
            "discarded": [[h.at, h.what, h.detail] for h in self.discarded()],
            "only_live": [[h.at, h.what, h.detail] for h in self.only_live()],
            "ours": [[h.at, h.what, h.detail] for h in self.ours],
            "verdicts": None if self.verdicts is None else
                        [[h.at, h.what, v.what, v.why]
                         for h, v in sorted(self.verdicts.items())],
        }


def _near(h: Heard, among: Sequence[Heard], slack: float) -> bool:
    return any(o.what == h.what and abs(o.at - h.at) <= slack for o in among)


#: ``(protocol, does this path belong to it, replay it)``. Every adapter is offered
#: the session's own log, because what a station heard depends on things that are not
#: in the audio; the ones whose arms are a tone search rather than a session decoder
#: take it and have no use for it.
Adapter = tuple[str, Callable[[Path], bool],
                Callable[[Path, "sessionlog.SessionLog | None"], Rehearing]]


def adapters() -> list[Adapter]:
    from . import besra, kestrel, shrike
    return [(m.PROTOCOL, m.claims, m.rehear) for m in (kestrel, besra, shrike)]


def rehear(path: Path, logs: Iterable[Path] = (), bw: str | None = None, *, selection=None) -> Rehearing | None:
    """Replay one recording, or None if no adapter claims it.

    ``logs`` are extra places to look for the session's own log — a file, or a
    directory of them — ahead of `sessionlog._roots`. ``bw`` is kestrel's only:
    an override for the VARA bandwidth an arm ran, where the log has no
    ``originating:`` line and the name does not say either.
    """
    selection = selection if selection is not None else sessionlog.recording_for(path)
    selected = sessionlog.select(selection, path) if selection is not None else None
    log = selected.log if selected is not None else sessionlog.find(path, logs)
    protocol = (selection or {}).get('context', {}).get('protocol', '').upper()
    wanted = {'VARA': 'kestrel', 'ARDOP': 'besra', 'PACTOR': 'shrike'}.get(protocol)
    if selected is not None and (not selected.complete or wanted is None):
        return Rehearing(path, wanted or 'unknown', 0., 'source sample clock', UPSTREAM,
                         note='Context incomplete: ' + str(selected.reason or 'protocol unavailable'))
    for name, claims, run in adapters():
        claimed = name == wanted if selection is not None else claims(path)
        if claimed:
            args = (path, log, bw) if name == 'kestrel' else (path, log)
            return run(*args, selection=selection) if selection is not None else run(*args)
    return None


def recordings(roots: Iterable[Path]) -> list[Path]:
    """Every recording under `roots`, once each, stored as FLAC or not yet."""
    from hfmodem.core.wav import recording_base, segments, stored
    out: set[Path] = set()
    def add(path):
        try:
            base = recording_base(path.with_suffix('.wav'))
            out.add(base if segments(base) != [stored(base)] else stored(base))
        except (OSError, ValueError):
            pass
    for root in map(Path, roots):
        try:
            if not root.is_dir():
                add(root)
            else:
                for path in root.rglob('*'):
                    if path.suffix in ('.wav', '.flac'):
                        add(path)
        except (OSError, ValueError):
            continue
    return sorted(out)


def report(r: Rehearing, *, ours: bool = False) -> str:
    lines = [f"{r.recording.name}  [{r.protocol}] {r.seconds:.0f} s",
             f"    zero: {r.zero}",
             f"    recorder is {r.recorder} of the live path's drops"]
    if r.uncovered:
        lines.append(f"    not covered: {r.uncovered}")
    if r.note:
        lines.append(f"    {r.note}")
    lines += _wrapped("DEAF: ", r.deaf)
    if r.against_log is not None:
        got, logged = r.against_log
        short = "" if got == logged else f" — {logged - got} it heard and this did not"
        lines.append(f"    reproduces {got} of the {logged} frames this session "
                     f"logged{short}")
    lines.append(f"    live {len(r.live)}, ungated {len(r.ungated)}")
    if ours:
        lines += [f"    we sent {h}" for h in r.ours]
    if r.verdicts is not None:
        # Every ungated hit, in the window's own order, with what it is against
        # the station's own record of the same instant. The whole list rather
        # than the unexplained tail of it: a column that shows only what it
        # cannot explain is a column that only ever reports losses.
        lines += [f"    {v.what.upper():>10}  {h}  {v.why}"
                  for h, v in sorted(r.verdicts.items())]
    else:
        lines += [f"    DISCARDED {h}" for h in r.discarded()]
    lines += [f"    live only {h}" for h in r.only_live()]
    return "\n".join(lines)


def _wrapped(label: str, text: str) -> list[str]:
    if not text:
        return []
    wrapped = textwrap.wrap(f"{label}{text}.", 88, subsequent_indent=" " * len(label))
    return [f"    {line}" for line in wrapped]


def totals(done: Sequence[Rehearing]) -> str:
    """The line the reports quote, one per protocol.

    Two shapes, and which one is printed is a property of the adapter rather
    than of the run. Where nothing classified the ungated hits it ends
    `discarded N, live-only N` and `discarded` means what it has always meant.
    Where an adapter placed the capture against the station's own record it ends
    `seen N, in a packet N, unseen N, live-only N` instead, and only `unseen`
    gets a breakdown by codeword — the other two are explained, and a report
    that quotes them as losses is quoting the wrong number.
    """
    lines = []
    for protocol in sorted({r.protocol for r in done}):
        mine = [r for r in done if r.protocol == protocol]
        lost = [h for r in mine for h in r.discarded()]
        held = [h for r in mine for h in r.only_live()]
        by_kind: dict[str, int] = {}
        for h in lost:
            by_kind[h.what] = by_kind.get(h.what, 0) + 1
        judged = [r for r in mine if r.verdicts is not None]
        tail = (f"seen {sum(len(r.by_verdict(SEEN)) for r in judged)}, "
                f"in a packet {sum(len(r.by_verdict(IN_PACKET)) for r in judged)}, "
                f"unseen {len(lost)}" if judged else f"discarded {len(lost)}")
        lines.append(
            f"{protocol}: {len(mine)} recordings, {sum(r.seconds for r in mine) / 3600:.2f} h, "
            f"live {sum(len(r.live) for r in mine)}, ungated {sum(len(r.ungated) for r in mine)}, "
            f"{tail}, live-only {len(held)}")
        lines.append(f"    recorder {mine[0].recorder} of the drops; zero: {mine[0].zero}")
        deaf = [r for r in mine if r.deaf]
        if deaf:
            lines.append(f"    {len(deaf)} of {len(mine)} replayed deaf against the "
                         f"station that made them:")
            lines += _wrapped("      ", deaf[0].deaf)
        graded = [r.against_log for r in mine if r.against_log is not None]
        if graded:
            lines.append(f"    graded against {len(graded)} session logs: "
                         f"{sum(g for g, _ in graded)} of "
                         f"{sum(n for _, n in graded)} frames reproduced")
        if mine[0].uncovered:
            lines.append(f"    not covered: {mine[0].uncovered}")
        label = "unseen" if judged else "discarded"
        lines += [f"    {label} {n:4d}  {what}"
                  for what, n in sorted(by_kind.items(), key=lambda kv: -kv[1])]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rehear",
        description="what the live receive path threw away, measured against the recording")
    ap.add_argument("paths", nargs="+", type=Path, help="recordings or directories of them")
    ap.add_argument("--json", type=Path, help="write every rehearing here")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    ap.add_argument("--ours", action="store_true",
                    help="also print what this station transmitted, where the capture holds it")
    ap.add_argument("--log", type=Path, action="append", default=[],
                    help="a session log, or a directory of them, searched ahead of "
                         "$HFMODEM_SESSION_LOGS and ~ for the session id and the mute "
                         "edges a replay cannot read off the audio")
    ap.add_argument("--bw", choices=("500", "2300", "2750"),
                    help="the VARA bandwidth a kestrel arm ran, where its log has no "
                         "originating: line and its name does not say either")
    ap.add_argument('--selection', type=Path)
    args = ap.parse_args(argv)
    selection = json.loads(args.selection.read_text()) if args.selection else None

    done: list[Rehearing] = []
    for path in recordings(args.paths):
        try:
            r = rehear(path, args.log, args.bw, selection=selection) if selection is not None else rehear(path, args.log, args.bw)
        except Exception as exc:                                # noqa: BLE001
            print(f"{path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if r is None:
            continue
        done.append(r)
        if not args.quiet:
            print(report(r, ours=args.ours), flush=True)

    # This tool imports the candidate on purpose, so every line above is a
    # statement about one artifact and has to say which one. The reading is taken
    # from `sys.modules` rather than from the tree, because a file on disk is not
    # proof of an import and the venv's editable finder decides that, not us.
    loaded = residency.imported()
    print()
    print(totals(done))
    print(f'    replayed against {residency.label(loaded)}')
    if args.json:
        args.json.write_text(json.dumps(
            dict(loaded=loaded, rehearings=[r.as_dict() for r in done]), indent=1))
    return 0
