"""THE CAMPAIGN REPORT: how far the night got, and what to do next.

The tally this replaces printed "5 of 9 answered" and named the cycle each
answer landed on. The operator's reading of it was the whole requirement:
*this doesn't tell me if we got to PACTOR-3, if we got more than one data
packet, or if we moved mail*. "Answered" is the bottom rung of a ladder and
five stations on it are not five equivalent facts -- one gateway acknowledged
sixty-seven times without ever alternating, another delivered eight bytes of
its banner, and the tally called both of them the same word.

THE LADDER IS THE SPINE, AND IT IS THE MODE'S OWN. `modes.rungs` asks the
profile: PACTOR climbs connect, grant, entry read, packet acknowledged,
greeting, login, mail, and those three questions map straight onto it. VARA
climbs its own -- cleared, keyed, below offer, answer, connect, greeting, mail
-- and its `below offer` rung exists because the same milestone pattern matches
a gateway offering a link and a gateway saying it could not read us. A third
mode arriving later writes its milestones and its `readout` and needs nothing
here.

WHAT A RUNG DOES NOT SAY, THE MODE SAYS. `Mode.readout` reads one arm's own
transcript for the facts a rung cannot carry -- whether the peer alternated,
whether our packet counter advanced, what crossed -- and hands back every
figure with what that figure cannot bear. Nothing in this file reads a
transcript for meaning, and nothing in it reads an exit code at all: one VARA
code covers a launcher refusal, a silent call and a connected session alike.

AND IT IS WRITTEN, NOT ONLY PRINTED. The same text goes to the terminal and to
a document beside the campaign's JSON, because a report that lives in
scrollback is a report the next campaign cannot read.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import logbook, modes, roster
from .station import cat_fault      # `station` itself is a Station below

#: What the report says of an arm that reached no rung at all. They are not a
#: rung -- nothing was climbed -- but a station has to appear somewhere, and the
#: difference between them is the difference between propagation and congestion.
#:
#: INTERRUPTED IS THE THIRD BECAUSE THE FIRST TWO WERE BEING USED TO SAY SOMETHING
#: FALSE. An arm the radio's CAT interface died under keyed nothing and read
#: nothing, and reading it as `never keyed` had the report announce the channel
#: had been found in use -- a band result invented out of a rig fault. It sorts
#: below `never keyed`: a channel refused at the gate was at least measured.
KEYED_NOTHING = 'keyed, no rung reached'
NEVER_KEYED = 'never keyed'
INTERRUPTED = 'interrupted by a rig fault'

#: The readings a report carries forward from the arms' provenance, and what to
#: call each one. Read back off the rig rather than assumed: `RFPOWER` is what
#: the radio answered, not what a preset asked for.
STATE = (('level', lambda record: (record.get('level') or {}).get('name')),
         ('codec gain', lambda record: (record.get('readings') or {}).get('codec_gain')),
         ('rig power', lambda record: (record.get('readings') or {}).get('rfpower')),
         ('rig mode', lambda record: ((record.get('readings') or {})
                                      .get('rig') or {}).get('mode')),
         ('dial off by', lambda record: (record.get('readings') or {}).get('dial_off_by_hz')),
         ('runtime', lambda record: (record.get('git') or {}).get('head', '')[:8]))


@dataclass(frozen=True)
class Flight:
    """One arm of the campaign, with the evidence it left on disk beside it.

    `key` and `mode` are read off the campaign where the arm's own row predates
    them: a single-mode pass before multimode landed wrote neither per arm, and
    a report is a reader of records that were already written.
    """
    entry: dict                 # the campaign's own row: outcome, assessment, verdict
    key: str
    mode: object
    cycle: int
    record: dict                # provenance.json, as the arm wrote it
    read: modes.Readout

    @property
    def keyed(self):
        return self.entry['outcome'] != modes.NOT_ATTEMPTED


def evidence(entry):
    """One arm's own record and transcript. An arm that wrote neither reads empty.

    The campaign row carries a verdict of its own for an arm interrupted before
    teardown, so the record is completed from it rather than left without one.

    surrogateescape and not `replace`, because the last thing a peer says is
    often the tail of a compressed block and its bytes are worth keeping: a
    replacement character says something arrived and nothing about what.
    `modes.rendered` is what makes them safe to print.
    """
    directory = Path(entry.get('directory') or '')
    written = directory / 'provenance.json'
    record = roster.read_json(written) if written.is_file() else {}
    log = directory / 'arm.log'
    return (dict(record, verdict=record.get('verdict') or entry.get('verdict')),
            log.read_text(errors='surrogateescape') if log.is_file() else '')


def flights(campaign):
    for cycle in campaign.get('cycles') or []:
        for entry in cycle['arms']:
            mode = modes.resolve(entry.get('mode') or campaign['mode'])
            record, text = evidence(entry)
            # A ROW MAY PREDATE THE WORD FOR WHAT HAPPENED TO IT. Campaigns flown
            # before an interrupted arm had a disposition of its own wrote the arm
            # and not the interruption -- and the launcher's own line is still in
            # the transcript that arm left, which is where it is read from here.
            entry = dict(entry, interrupted=entry.get('interrupted')
                         or cat_fault(entry.get('directory')))
            read = (mode.readout(mode, record, text) if mode.readout else
                    modes.Readout(rung=modes.highest(
                        mode, (record.get('verdict') or {}).get('reached'))))
            yield Flight(entry, entry.get('key') or entry['peer_id'], mode,
                         cycle['cycle'], record, read)


def rung_of(flight):
    if flight.read.rung:
        return flight.read.rung
    if flight.entry.get('interrupted'):
        return INTERRUPTED
    return KEYED_NOTHING if flight.keyed else NEVER_KEYED


@dataclass(frozen=True)
class Station:
    """Every arm at one elimination key, and the best of them.

    The best arm is the one that climbed highest, and its readout is what the
    station's line says. Merging four arms' figures would report a night that
    never happened: the peer that alternated did so in one of them.
    """
    key: str
    flown: tuple
    answered_on: int | None

    @property
    def mode(self):
        return self.flown[0].mode if self.flown else None

    @property
    def best(self):
        ladder = [INTERRUPTED, NEVER_KEYED, KEYED_NOTHING, *modes.rungs(self.mode)]
        return max(self.flown, key=lambda flight: ladder.index(rung_of(flight)))

    @property
    def outcome(self):
        """The elimination state across every arm, not the state of the last one.

        A station three passes keyed at and a fourth stepped over is `no reply`;
        reading the last arm alone would file it as never attempted.
        """
        if self.answered_on:
            return f'answered on cycle {self.answered_on}'
        keyed = [flight for flight in self.flown if flight.keyed]
        return f'{modes.NO_REPLY} ({len(keyed)}x)' if keyed else modes.NOT_ATTEMPTED

    @property
    def rung(self):
        return rung_of(self.best) if self.flown else NEVER_KEYED

    @property
    def reached(self):
        return (self.best.record.get('verdict') or {}).get('reached') or []

    @property
    def short_of(self):
        rungs = modes.rungs(self.mode)
        climbed = rungs.index(self.rung) if self.rung in rungs else -1
        return rungs[climbed + 1:]


def stations(campaign, keys, flown):
    at = {}
    for flight in flown:
        at.setdefault(flight.key, []).append(flight)
    for key in list(keys) or list(at):
        yield Station(key, tuple(at.get(key, ())), campaign['answered_on'].get(key))


def tally(campaign, keys, flown):
    """Who answered, who never did, and who was never attempted. The elimination state.

    Counted by the elimination key, so a station a mode lists on four channels is
    one line and not four.
    """
    attempts = {key: [flight for flight in flown if flight.key == key and flight.keyed]
                for key in keys}
    answered = campaign['answered_on']
    silent = [key for key in keys if key not in answered and attempts[key]]
    untried = [key for key in keys if not attempts[key]]
    yield (f'\nCampaign ended: {campaign["stopped_because"]} '
           f'{len(answered)} of {len(keys)} answered.')
    at = {flight.key: flight.entry.get('asks') for flight in flown
          if flight.entry['outcome'] == modes.ANSWERED}
    if answered:
        yield '  answered       ' + ', '.join(
            f'{name} on cycle {cycle}' + (f' at {at[name]}' if at.get(name) else '')
            for name, cycle in answered.items())
    if silent:
        yield '  no reply       ' + ', '.join(
            f'{name} ({len(attempts[name])}x)' for name in silent)
    stopped = {key: [flight for flight in flown
                     if flight.key == key and flight.entry.get('interrupted')]
               for key in untried}
    if untried:
        yield ('  not attempted  ' + ', '.join(untried)
               + ' -- nothing was keyed at them and they had no chance to answer.')
        if busy := [key for key in untried if not stopped[key]]:
            yield ('    in use       ' + ', '.join(busy)
                   + ' -- every pass found the channel busy and moved on.')
        if faulted := [key for key in untried if stopped[key]]:
            yield ('    rig fault    ' + ', '.join(faulted)
                   + ' -- the radio stopped answering CAT before anything was keyed, so '
                     'the channel was never read. Neither in use nor silent.')
    if untried and not any(attempts.values()) and not any(stopped.values()):
        yield ('  NOTHING WAS ATTEMPTED AT ALL. Every channel read busy on every '
               'pass. Scanning finds an opening far more often than waiting on one '
               'dial, but it cannot invent one, and a threshold that is wrong just '
               'returns wrong answers faster. --force is the operator saying the '
               'channel is clear.')


def recoveries(campaign):
    """Every time the mains went off under this pass, and what it cost.

    A SECTION OF ITS OWN BECAUSE IT IS WEATHER DATA. The tuner hunts a match when
    the wire moves, and the hunt's RFI is what kills CAT -- so these instants say
    when the antenna moved, which nothing else at this station records. They are
    written to be read against a weather record months later, and the CAT return
    time is worth watching on its own: a radio taking longer to come back every
    time is a fact about the radio.
    """
    said = campaign.get('recoveries') or []
    if not said:
        return
    yield f'\n## Rig faults and mains recoveries ({len(said)})'
    yield ('The ATU hunts whenever the end-fed wire moves -- a breeze, ice -- and its RFI '
           'takes the radio\'s CAT interface down. Only mains brings it back. Nothing was '
           'keyed at any of these; none of them is a reading of the band.')
    for one in said:
        yield (f'  {one.get("at", "?")[:19]}Z  {one.get("peer_id", "?")}  '
               f'{one.get("centre_hz", "?")} Hz  '
               + (f'CAT back in {one["cat_back_s"]}s on dial {one["dial_hz"]}'
                  if one.get('recovered') else 'THE RADIO DID NOT COME BACK'))
        for line in one.get('power') or []:
            yield f'    {line.strip()}'
        if one.get('refused'):
            yield f'    the mains cycle did not run: {one["refused"]}'


def ladder(seen):
    """How far the night got, top rung first, with every station on the rung it reached.

    One ladder per mode in the pass, because they are not comparable: a VARA
    `connect` and a PACTOR `connect` are the same word for different distances
    up two different protocols.
    """
    yield '\n## How far it got'
    yield ('The rungs are each mode\'s own. A station sits on the highest rung any '
           'of its arms reached.')
    for name in dict.fromkeys(station.mode.name for station in seen if station.mode):
        here = [station for station in seen if station.mode and station.mode.name == name]
        rungs = [*reversed(modes.rungs(here[0].mode)), KEYED_NOTHING, NEVER_KEYED,
                 INTERRUPTED]
        width = max(len(rung) for rung in rungs)
        yield f'  {name}'
        for rung in rungs:
            on = [station for station in here if station.rung == rung]
            yield (f'    {rung:{width}}  ' + (', '.join(
                f'{station.key}'
                + (f' (cycle {station.answered_on})' if station.answered_on else '')
                for station in on) or '--'))


def wanted(station, worked):
    """What to do next at one station, out of its own arms and every mode's logbook.

    Three readings and nothing invented. The arm's own `suggests` is the mode
    saying its transcript points at a different configuration. The logbook says
    whether this station has ever replied in the OTHER mode, which is the one
    question a single-mode campaign cannot ask itself. Everything else is the
    elimination state, said plainly.
    """
    if not station.flown:
        return 'never reached -- the campaign ended before its turn came round.'
    keyed = sum(1 for flight in station.flown if flight.keyed)
    if station.rung == INTERRUPTED:
        said = [f'{len(station.flown)} pass(es) ended on a rig fault before anything was '
                'keyed, so the channel was never read and nothing here is a reading of the '
                'band -- it is worth calling again on the next pass']
    elif station.rung == NEVER_KEYED:
        said = [f'{len(station.flown)} pass(es) found the channel in use and moved on, so '
                'nothing was keyed and it had no chance to answer -- --force is the '
                'operator saying the channel is clear']
    elif station.rung == KEYED_NOTHING:
        said = [f'keyed on {keyed} pass(es) and nothing came back'
                + (f', and {len(station.flown) - keyed} more found the channel in use'
                   if keyed < len(station.flown) else '')
                + ' -- another hour or another band is the lever, not another flag']
    elif not station.short_of:
        said = ['it reached the top of the ladder and nothing is outstanding here']
    else:
        said = [f'it reaches {station.rung} and stops short of {station.short_of[0]}, '
                'which is the rung another arm here would be for']
    if suggests := station.best.read.suggests:
        said.append(suggests.rstrip('.'))
    call, band = station.best.record.get('call'), station.best.record.get('band_m')
    elsewhere = {name: seen for name, seen in worked.get((str(call).upper(), band), {}).items()
                 if name != station.mode.name and seen['milestones']}
    for name, seen in elsewhere.items():
        said.append(f'it has reached a milestone in {name} ({seen["milestones"]} of '
                    f'{seen["arms"]} arm(s), latest {seen["latest"][:10]}), so calling it '
                    'there is the pass with evidence behind it')
    return '; '.join(said) + '.'


def block(label, phrases):
    """A labelled block: the label beside the first line, the rest ranged under it."""
    for number, phrase in enumerate(phrases):
        yield f'  {label if not number else "":12} {phrase}'


def detail(seen):
    yield '\n## Station by station'
    for station in seen:
        if not station.flown:
            yield f'\n{station.key}  -- never reached'
            continue
        best = station.best
        yield (f'\n{station.key}  {station.mode.name}  {len(station.flown)} arm(s), '
               f'{station.outcome} -- {station.rung}'
               + (f' at {best.entry["asks"]}' if best.entry.get('asks') else ''))
        yield from block('reached', [', '.join(station.reached)] if station.reached else [])
        yield from block('short of', [', '.join(station.short_of)] if station.short_of else [])
        yield from block('best arm', [f'{best.entry.get("arm", "none")} (cycle {best.cycle})'])
        yield from block('did', best.read.did)
        yield from block('crossed', [best.read.crossed] if best.read.crossed else [])
        yield from block('peer said', [best.read.said] if best.read.said else [])
        yield from block('anomalies', [f'{many}x {what}'
                                       for many, what in gathered(station.flown)])


def gathered(flown):
    """Every anomaly these arms named, with how many times it was seen across them."""
    counted = {}
    for flight in flown:
        for many, what in flight.read.anomalies:
            counted[what] = counted.get(what, 0) + many
    return [(many, what) for what, many in sorted(counted.items(), key=lambda it: -it[1])]


def limits(flown):
    """What the numbers above cannot support, in the words of whatever measured them.

    A report that quotes a count without saying what it can bear is worse than no
    report: the same night has a session log reporting zero PACTOR-3 on an arm
    whose PACTOR-3 reader never ran, and a control-signal count the arm itself
    prints a null calibration beside.
    """
    said = {}
    for flight in flown:
        for figure, cannot in flight.read.caveats:
            said.setdefault((figure, cannot), []).append(flight.entry.get('arm', '?'))
    if not said:
        return
    yield '\n## What these figures cannot bear'
    for (figure, cannot), arms in said.items():
        yield f'  {figure} -- {cannot}'
        yield f'    on {len(arms)} arm(s), including {arms[0]}'


def next_pass(seen, worked):
    yield '\n## What to do next'
    for station in seen:
        yield f'  {station.key}  {wanted(station, worked)}'


def settings(flown):
    """The station's own state, read off the arms' provenance rather than assumed.

    A figure that moved during the pass is said as two, because a campaign flown
    at two codec gains is two campaigns and the report is the only place that
    would show it.
    """
    yield '\n## The station itself, as the arms read it back'
    for name, reading in STATE:
        counted = {}
        for flight in flown:
            value = reading(flight.record)
            if value is not None and str(value):
                counted[str(value)] = counted.get(str(value), 0) + 1
        if not counted:
            continue
        order = sorted(counted.items(), key=lambda it: -it[1])
        yield (f'  {name:12} ' + (order[0][0] if len(order) == 1 else
                                  ', '.join(f'{value} ({many} arm(s))' for value, many in order)))
    drives = {}
    for flight in flown:
        level = flight.record.get('level') or {}
        for drive in flight.mode.drives:
            if level.get(drive) is not None:
                drives.setdefault(f'{drive.replace("_", "-")} {level[drive]:g}', 0)
                drives[f'{drive.replace("_", "-")} {level[drive]:g}'] += 1
    if drives:
        yield '  drives       ' + ', '.join(drives)


def spell(campaign, keys=()):
    """The whole report, one line at a time, printed and written from the same text."""
    flown = list(flights(campaign))
    keys = list(keys) or list(dict.fromkeys(flight.key for flight in flown))
    seen = list(stations(campaign, keys, flown))
    worked = logbook.answered_at()
    yield from tally(campaign, keys, flown)
    if not flown:
        return
    yield from head(campaign)
    yield from recoveries(campaign)
    yield from ladder(seen)
    yield from detail(seen)
    yield from limits(flown)
    yield from next_pass(seen, worked)
    yield from settings(flown)


def head(campaign):
    """What was asked for, out of the header whoever planned the pass wrote.

    A single-mode pass writes `mode` and `level`, a multimode one writes `modes`
    and `levels`, and both are read here rather than in two places. Anything the
    header does not carry is left out rather than guessed at.
    """
    started, ended = campaign.get('started_utc'), campaign.get('ended_utc')
    spent = ''
    try:
        spent = (f', {(datetime.fromisoformat(ended) - datetime.fromisoformat(started))}'
                 .rsplit('.', 1)[0] + ' on the air')
    except (TypeError, ValueError):
        pass
    levels = campaign.get('levels') or {campaign.get('mode'): campaign.get('level')}
    named = [f'{name} at {levels[name]}' if levels.get(name) else str(name)
             for name in campaign.get('modes') or [campaign.get('mode')] if name]
    bands = '/'.join(f'{band} m' for band in campaign.get('bands_m') or ())
    asks = '/'.join(campaign.get('asks') or ())
    yield ('\n# Campaign report' + (' -- ' + ', '.join(named) if named else '')
           + (f' on {bands}' if bands else '') + (f' at {asks}' if asks else ''))
    cycles = campaign.get('cycles') or []
    yield ((f'{started[:10]} {started[11:16]}Z' if started else 'undated')
           + f'{spent}; {campaign.get("fly_mode", "fly mode unrecorded")}, '
           + f'{len(cycles)} of {campaign.get("cycles_asked", len(cycles))} cycle(s) asked, '
           + f'{sum(len(cycle["arms"]) for cycle in cycles)} arm(s) flown.')


def write(campaign, keys, path):
    """The report beside the record it reads, so it outlives the terminal.

    One text, printed and saved as it is produced. Streaming is the point: a
    section that cannot render must not take the elimination state above it with
    it, and the document keeps whatever reached the screen.
    """
    lines = []
    try:
        for line in spell(campaign, keys):
            print(line)
            lines.append(line)
    finally:
        path.write_text('\n'.join(lines) + '\n')
    return lines
