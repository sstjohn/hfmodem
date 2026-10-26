# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The mode profile: everything one radio mode does differently, as data.

`console <mode> <verb>` resolves a name here. A profile says where its roster
and its flown arms live, which level presets it offers and what the operator
authorised them to ask for, which launcher `tools/onair.sh` verb it drives, what
the launcher line carries between the shared head and the shared tail, and how
its transcript reads. Everything else -- finding a host, claiming the rig,
reading the rig back, flying an arm, running a campaign -- is the substrate and
is written once.

The launcher line is assembled in four parts, and only the second is the mode's:

    onair.sh <launcher> CALL CENTRE --log ARM --band M      the substrate's head
    ...                                                     mode.flags(...)
    [--witness] [--force]                                   the substrate's tail
    [--modem-arg words]                                     the operator's own,
                                                            last and unchecked

A verb that takes its channel differently owns its head too: `vara-mail` reads a
bare bandwidth where `--band` stands and refuses the flag, so `Mode.head` is
there for a mode whose launcher spells the channel its own way.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

KNOWN = ('pactor', 'vara', 'ardop')
#: Everything that holds the rig's card under a lock of its own. The RTTY
#: terminal is one and is not a campaign mode, which `multimode` takes every
#: KNOWN mode to be.
RIG_HOLDERS = KNOWN + ('rtty', 'voice')

# THE COMMON CORE. Every mode's arm record carries these fields, spelled the same
# way, because `console log` is one view over all of them and a second spelling
# would be a second logbook. The substrate writes all of them -- `arm.plan` at the
# start and `arm.fly` at teardown -- so a profile cannot forget one; what a mode
# has of its own goes under `adjustments`, which the logbook never reads.
#
# The last three are written at teardown, so an arm that is still flying, or one
# that was stopped, carries the rest and not those. They read as None.
CORE = ('arm', 'directory', 'call', 'band_m', 'centre_hz', 'dial_hz',
        'mode', 'level', 'started_utc', 'ended_utc', 'returncode', 'verdict')
AT_TEARDOWN = ('ended_utc', 'returncode', 'verdict')


# THE THREE OUTCOMES, AND THE WHOLE VOCABULARY A CAMPAIGN HAS. They are kept
# apart everywhere they are printed or recorded because they say different things
# about what to do next: `answered` is settled and the channel leaves the list,
# `no reply` may answer on another pass, and `not attempted` never had a chance
# because we did not key at all. Folding the last two together spends cycles
# calling stations that were never called.
ANSWERED, NO_REPLY, NOT_ATTEMPTED = 'answered', 'no reply', 'not attempted'
OUTCOMES = (ANSWERED, NO_REPLY, NOT_ATTEMPTED)

# THE CHANNEL WAS CLEAR, in every spelling the launcher's gate has used. A windowed
# sense line reads one window of several, so `(clear); ... clear windows 1/3` is not
# a clear channel; only the full count is, or a line that says it outright.
CLEARED = re.compile(r'occupancy decision complete|channel is free(?! -- one more window)|channel sense: [^\n]*'
                     r'(?:-> clear|\(clear\)(?:$|;[^\n]*clear windows (\d+)/\1\b))', re.M)


@dataclass(frozen=True)
class Assessment:
    """What is on a channel, and how much of it the instrument could actually tell.

    The question a campaign has is not "is this frequency loud" but "is the
    station I mean to call busy": a gateway already in a session will not answer,
    and a free one answers under other traffic. Only a demodulator can separate
    those, so the fields an energy meter cannot fill are None and `confidence`
    is 0.0, rather than a number that reads like knowledge.

    `inviting` is the one thing today's instrument can be sure of: a PACTOR Free
    Signal is the wanted station saying it has no link in progress.
    """
    occupied: bool | None
    occupant: str | None = None         # what is there, when it can be named
    is_target: bool | None = None       # whether that is the station we want
    inviting: bool = False              # the target, asking to be called
    confidence: float = 0.0
    instrument: str = 'none'
    spent_s: float = 0.0                # what the reading cost, which a scan pays per stop
    detail: str = ''

    def spell(self):
        said = ('nothing heard' if self.occupied is False else
                'occupied' if self.occupied else 'not assessed')
        who = (f', {self.occupant}' if self.occupant else
               '' if self.occupied is not True else ', occupant unidentified')
        return (f'  {said}{who} [{self.instrument}, confidence {self.confidence:.1f}]'
                + (f': {self.detail}' if self.detail else ''))


# WHAT THE FAR END SAID TO US, WHICH IS THE HIGHEST-GRADE EVIDENCE THIS STATION
# GETS. Everything else in a report is inference from waveforms; this is the
# gateway stating its reason in English. VE3KPG hung up on 09-17 with "This RMS
# channel (VE3KPG) is set to only accept Pactor levels above 2 - Disconnecting",
# and a report that showed the banner and dropped that line said nothing about
# why the session ended. So every word of it is carried, from the first, in the
# order it arrived.
#
# ONE PATTERN BECAUSE THERE IS ONE MAIL CLIENT. `onair.sh pactor` and
# `onair.sh vara-mail` both drive the same B2F client, and it prints the far
# end's words on this line in either. A mode whose peer speaks somewhere else
# gathers its own lines and hands them to `rendered`.
SPOKE = re.compile(r'^mail: peer said: (.*)$', re.M)

#: How much of it the report carries, in bytes. The operator's figure: enough
#: for a banner and the refusal behind it, short enough to stay one paragraph.
SPOKEN_BYTES = 512


def rendered(said):
    """Everything the far end said, in order, safe to print and safe to save.

    Truncated by BYTES rather than characters, because the limit is the
    operator's and a byte is what they asked for. Kept verbatim inside that
    limit: the refusal above names the CHANNEL rather than the station or the
    software, which is the evidence that the policy is a per-channel sysop
    setting, and a paraphrase loses exactly that.

    A peer's last words are often the tail of a compressed block arriving after
    the disconnect. Those bytes are not dropped -- where a session died is worth
    seeing -- but what comes back is pure ASCII, so neither a terminal nor the
    document this is saved into can be corrupted by them. A byte that is not
    text becomes its own value where the read preserved one, and the escape the
    transcript itself holds where it did not: `\\xef` is a byte that arrived and
    `\\ufffd` is a byte the arm had already given up on, and the report is not
    entitled to turn the second into the first.
    """
    joined = ' | '.join(part.strip() for part in said if part.strip())
    if not joined:
        return ''
    raw = joined.encode('utf-8', 'surrogateescape')
    kept = raw[:SPOKEN_BYTES].decode('utf-8', 'surrogateescape')
    shown = ''.join(letter if letter.isprintable() else f'\\x{ord(letter) & 0xff:02x}'
                    for letter in kept)
    return (shown.encode('ascii', 'backslashreplace').decode('ascii')
            + (f' ... +{len(raw) - SPOKEN_BYTES} byte(s)' if len(raw) > SPOKEN_BYTES else ''))


def spoken(text):
    """The far end's own words out of one transcript the shared mail client wrote."""
    return rendered(SPOKE.findall(text))


@dataclass(frozen=True)
class Readout:
    """What one arm actually did, past the three outcomes and the list of milestones.

    "Answered" is the bottom rung of a ladder, and two arms that share it can be
    entirely different facts: one gateway acknowledged sixty-seven times without
    ever alternating, another delivered eight bytes of its banner. So the mode
    reads its own transcript for the rest -- there is no common vocabulary for
    what a PACTOR link and a VARA lattice frame did -- and every figure it hands
    back carries what that figure cannot bear.

    `rung` is where on the mode's own `ladder` this arm stopped. A mode that does
    not fill it leaves the report to take the highest milestone reached, which is
    right until a milestone covers two different things: `answer` matches every
    VARA lattice frame, and frame 17 is an offer where 14 to 16 are the gateway
    saying it heard the request and could not read it.
    """
    rung: str = ''
    did: tuple = ()             # what it did, in phrases its own transcript supports
    crossed: str = ''           # whether traffic moved
    said: str = ''              # the far end's own words, from the first; see `rendered`
    anomalies: tuple = ()       # (how many, what) -- the observations findings come from
    caveats: tuple = ()         # (the figure, what that figure cannot bear)
    suggests: str = ''          # the configuration this arm's own evidence points at


UNASSESSED = Assessment(
    occupied=None, instrument='none',
    detail='this mode has no assessor; the launcher\'s own gate decides at key-down')


@dataclass(frozen=True)
class Mode:
    name: str                                   # what `console <name>` answers to
    launcher: str                               # the verb tools/onair.sh takes
    entry: Sequence[str]                        # what that verb's python imports; the
                                                # residency probe imports these and no more
    roster: Path                                # peers.json: channels, spacing, gain points
    arms: Path                                  # flown arms; it never moves once one is in it
    lock: Path                                  # the arm lock, shared with anything else flying
    bands: Sequence[int]
    levels: Mapping[str, dict]
    phases: Sequence[str]                       # the level keys that ask the rig for watts
    drives: Sequence[str]                       # the level keys that scale the transmitted audio
    authorised_watts: float
    flags: Callable                             # (peer, level, args, directory, send) -> words, adjustments
    arguments: Callable                         # (parser) -> None; the fly flags this mode takes
    milestones: Sequence[tuple[str, re.Pattern]]
    head: Callable | None = None                # (mode, peer, args, arm) -> the words before flags
    ready: Callable | None = None               # (mode, plan) -> a context held while it flies
    campaign_arguments: Callable | None = None  # (parser) -> None; where a campaign's differ
    channel_rank: Callable | None = None        # (peer, args) -> how well this channel fits, or None
    channel_arms: Callable | None = None        # (args, peer) -> (args, phrase) per arm at it
    verdict: Callable | None = None             # (mode, directory) -> dict; None reads the transcript
    disposition: Callable | None = None         # (code, record) -> ('continue'|'stop', why)
    assess: Callable | None = None              # (mode, peer, dwell_s) -> Assessment
    dwell_floor_s: float = 0.0                  # the shortest read its assessor can take
    outcome: Callable | None = None             # (mode, record, code) -> one of OUTCOMES
    records: Callable | None = None             # (mode) -> its arm records, in CORE's spelling
    readout: Callable | None = None             # (mode, record, text) -> Readout
    ladder: Sequence[str] = ()                  # the report's rungs, lowest first; see `rungs`
    refresh: Callable | None = None             # (mode) -> what moved, rewriting the roster live


def arms_at(mode, args, peer):
    """The arms one channel is worth this pass, each with what it asks for in a phrase.

    One, unless the mode varies something at a channel besides the channel
    itself: a VARA campaign names widths as it names bands, and a channel it
    finds open is called at each of the named widths that channel may take.
    """
    return mode.channel_arms(args, peer) if mode.channel_arms else [(args, '')]


def rungs(mode):
    """The ladder a report ranks this mode's arms on, lowest rung first.

    The milestones are it, until a mode needs a rung its milestones do not spell:
    VARA puts a below-offer signal report between keying and an answer, because
    the same `answer` pattern matches both and they are opposite facts.
    """
    return tuple(mode.ladder) or tuple(name for name, _ in mode.milestones)


def highest(mode, reached):
    """The top rung of this mode's ladder that these milestones reach, or none."""
    climbed = [rung for rung in rungs(mode) if rung in (reached or ())]
    return climbed[-1] if climbed else ''


def resolve(name):
    """The profile for a mode name. An unknown one is refused with the ones that fly."""
    if name not in KNOWN:
        raise ValueError(f'Unknown mode {name!r}; this console flies {", ".join(KNOWN)}.')
    profile = import_module(f'{__package__}.{name}').PROFILE
    if profile is None:
        raise ValueError(f'{name} is not yet adapted to this console. Its session still '
                         f'lives in working/{name}-level-air-0915/; consolelib/{name}.py '
                         'names which of its functions map to which substrate piece.')
    from .operator import mode_root
    root = mode_root(name)
    return replace(profile, roster=root / "peers.json", arms=root / "arms", lock=root / "arm.lock")


def holder(name):
    """What holds the rig under `name`: a flying mode's profile, or a session's lock and records."""
    return resolve(name) if name in KNOWN else import_module(f'{__package__}.{name}').HOLDS


def level_for(mode, name):
    """A preset, checked against what the operator authorised this mode to ask for."""
    if name not in mode.levels:
        raise ValueError(f'Unknown level {name!r}; presets are {", ".join(mode.levels)}.')
    level = mode.levels[name]
    for phase in mode.phases:
        watts = level[phase]
        if watts is not None and watts > mode.authorised_watts:
            raise ValueError(f'Preset {name} asks {watts:g} W on {phase[:2].upper()}; '
                             f'the operator authorised {mode.authorised_watts:g} W.')
    return dict(level, name=name)


def spell_levels(mode):
    for name, level in mode.levels.items():
        watts = ', '.join(f'{phase[:2].upper()} '
                          f'{"as the rig sits" if level[phase] is None else f"{level[phase]:g} W"}'
                          for phase in mode.phases)
        drives = ', '.join(f'{drive.replace("_", "-")} {level[drive]:g}' for drive in mode.drives)
        said = ', '.join(part for part in (watts, drives) if part)
        yield f'{name:12} {said} -- {level["note"]}'


def core(record, mode_name=None):
    """One arm record reduced to the fields every mode carries, and what is absent.

    The logbook reads this and nothing else. `level` is the preset's name and
    `reached`/`line` are the verdict unpacked, because those are what a reader
    of the log is after; everything else keeps the spelling it was written with.
    """
    verdict = record.get('verdict') or {}
    entry = {field: record.get(field) for field in CORE}
    entry.update(mode_name=mode_name, level=(record.get('level') or {}).get('name'),
                 reached=verdict.get('reached') or [], line=verdict.get('line'),
                 missing=[field for field in CORE if record.get(field) is None])
    return entry
