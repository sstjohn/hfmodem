"""A campaign: a pounce across frequency, run again until every station has answered.

One loop does both things the operator asked for, because they are the same loop.

A PASS IS A POUNCE, SPENT ACROSS THE BAND INSTEAD OF ON ONE DIAL. Classic pounce
sits on a channel and waits for it to clear. With twenty channels on the roster
there is no reason to wait on any of them: read one, call it if it is clear, and
if it is not, step to the next one now. First fit, never best fit -- a sense
reading is worth about a minute before it goes stale, so surveying the list and
then picking "the quietest" would pick on readings minutes old.

REPEATING THE PASS IS THE ELIMINATION. `--cycles 3` goes through three times; a
station that answers is dropped, because the question at it has been settled, and
the passes left are spent only where the answer is still unknown. A busy channel
is not a lost opportunity: it is a station the next pass comes back to, which is
pounce's retry spent across frequency.

Three outcomes, kept apart everywhere they are recorded:

    answered        we keyed and a reply decoded    dropped from the list
    no reply        we keyed and nothing decoded    stays; propagation moves
    not attempted   we never keyed at all           stays; we moved on

AND AN ARM A RIG FAULT ENDED IS NONE OF THE THREE, which is why it carries a word
of its own. The station's ATU hunts a match whenever its wire moves -- wind, ice
-- and the RFI off that hunt takes the radio's CAT interface down with it. The
arm dies before it reads anything, and a report that filed it under `not
attempted` announced the channel had been found in use: a band result invented
out of a rig fault. Such an arm is marked `interrupted`, the mains are cycled
under it, and the channel is flown again -- once per frequency, which is also
what re-tunes the whole list. See `Rescue`.

THOSE THREE WORDS END THE PASS AND DO NOT REPORT IT. `answered` covers a gateway
that acknowledged sixty-seven times without ever alternating and one that handed
back its banner, and the operator's question of the tally was the whole
requirement: did we get to PACTOR-3, did more than one data packet go out, did
mail move. `report.py` answers those off each mode's own ladder, and writes the
answer beside this record.

What counts as a reply is the mode's to say (`Mode.outcome`), and what is on a
channel is the mode's to assess (`Mode.assess`). Nothing in this file measures
anything itself: the question that decides a stop is not "is this frequency loud"
but "is the station I mean to call busy", and only a demodulator answers that.

WHAT THIS DOES NOT FIX. On a night when every channel is busy, scanning finds an
opening far more often than sitting on one dial does -- but it cannot invent one,
and if the threshold itself is wrong then twenty channels return twenty wrong
answers faster than one channel returned one. The fix for that is the decode
assessor behind the same `assess` seam, not more scanning.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from . import arm, modes, report, roster, station

RETUNE_S = 2.0          # what a within-band hop costs before the dwell starts
DWELL_S = 8.0           # a stop's default read; pactor's instrument reads this
STOPPED_BY_HAND = 'Interrupted; the cycles run so far are in this record.'



@dataclass(frozen=True)
class Call:
    """One channel a pass may call, and the mode that would transmit at it.

    The mode's own peer record travels whole, beside the arguments that mode's
    arm takes. Nothing is flattened: the roster's shape, the flags, the assessor
    and the level presets stay with the mode, and a pass that spans modes asks
    each channel's own mode rather than a common denominator of the two.

    `key` is what a station is eliminated by. A single-mode pass keys on the
    channel, which is what it has always done; a pass that spans modes keys on
    the station, because it calls each station in one mode and an answer settles
    the question wherever it came from. Widening that key -- to keep calling a
    station in the other mode, or to hold out for a link rather than a reply --
    is a change to whoever builds the calls, not to the loop below.
    """
    mode: object
    args: object
    peer: dict
    key: str
    why: str = ''       # why this mode, in one phrase; blank when there was no choice
    asks: str = ''      # what this arm asks for that another arm here does not (BW2300)


def step_args(args, peer):
    """The namespace `fly` builds one arm from, so a campaign step is one arm."""
    return argparse.Namespace(**dict(vars(args), peer=peer['id'], band=peer['band_m']))


def fly_mode(args):
    return 'force' if args.force else f'pounce {args.pounce}s' if args.pounce else 'sense'


def assessment(mode, args, peer):
    """Ask the mode what is on the channel. The campaign never measures it itself.

    Nothing is waited for here -- the dwell is one short read at this stop, and
    the launcher senses again at key-down. Under --force the operator has already
    overruled the channel question, so nothing is measured at all.
    """
    if args.force or not args.assess or mode.assess is None:
        return modes.UNASSESSED
    try:
        return mode.assess(mode, peer, args.dwell)
    except Exception as trouble:        # an assessor may never cost the pass an arm
        return replace(modes.UNASSESSED,
                       detail=f'the assessor failed ({type(trouble).__name__}: {trouble})')


def classify(mode, record, code):
    if mode.outcome is None:
        raise ValueError(f'{mode.name} has no outcome test, so a campaign cannot tell a '
                         'channel that said nothing from one it never keyed on. See '
                         'consolelib/modes.py for what a profile owes a campaign.')
    return mode.outcome(mode, record, code)


def disposition(mode, code, record):
    if mode.disposition:
        return mode.disposition(code, record)
    if record is None:
        return 'stop', 'The arm wrote no record; the launcher never started.'
    if record.get('verdict', {}).get('line') == 'no transcript':
        return 'stop', 'The arm wrote no transcript; the launcher did not run.'
    return 'continue', f'launcher exit {code}'


def fly_one(call):
    """One arm at one channel: ask what is there, key or not, and classify it.

    Everything the arm needs is the call's -- the mode that transmits, that
    mode's own arguments, and the peer as its own roster wrote it.

    Returns the record of the arm and, when the campaign should end here, why.
    """
    mode, args, peer = call.mode, call.args, call.peer
    entry = dict(peer_id=peer['id'], gateway=peer['call'], band_m=peer['band_m'],
                 centre_hz=peer['centre_hz'], mode=mode.name, key=call.key,
                 asks=call.asks, returncode=None)
    look = assessment(mode, args, peer)
    entry['assessment'] = asdict(look)
    print(look.spell(), flush=True)
    if look.occupied and not look.inviting:
        entry.update(outcome=modes.NOT_ATTEMPTED, moved_on=True,
                     note=(f'{peer["call"]} is in a session and would not answer'
                           if look.is_target else 'the channel is in use')
                          + '; moved on rather than waiting here.')
        return entry, None
    record = None
    try:
        plan = arm.plan(mode, step_args(args, peer), datetime.now(timezone.utc))
        while Path(plan['directory']).exists():
            time.sleep(0.1)     # an arm is named to the second and a pass can return inside one
            plan = arm.plan(mode, step_args(args, peer), datetime.now(timezone.utc))
        entry.update(arm=plan['arm'], command=arm.spelled(plan), directory=plan['directory'])
        try:
            entry['returncode'] = arm.fly(mode, plan, args.qualify)
        finally:
            # fly publishes the exact allocated root before starting the arm.
            # The same root owns any record left by an interrupted teardown.
            entry.update(arm=plan['arm'], command=arm.spelled(plan),
                         directory=plan['directory'])
            for key in ('requested_arm', 'capture_id', 'capture_root', 'transcript'):
                if key in plan:
                    entry[key] = plan[key]
        record = roster.read_json(Path(plan['directory']) / 'provenance.json')
    except KeyboardInterrupt:
        written = Path(entry.get('directory', '')) / 'provenance.json'
        record = roster.read_json(written) if written.is_file() else None
        entry.update(outcome=classify(mode, record, None) if record else modes.NOT_ATTEMPTED,
                     note='Interrupted during the arm.')
        return entry, STOPPED_BY_HAND
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f'REFUSED: {error}', file=sys.stderr, flush=True)
        entry.update(outcome=modes.NOT_ATTEMPTED, note=str(error))
        return entry, str(error)
    entry.update(verdict=record.get('verdict'),
                 outcome=classify(mode, record, entry['returncode']),
                 note=(record.get('verdict') or {}).get('line'))
    if fault := station.cat_fault(entry['directory']):
        entry.update(interrupted=fault, outcome=modes.NOT_ATTEMPTED,
                     note='the radio stopped answering CAT before anything was keyed, so '
                          'this channel was never read: a rig fault, not a band result')
        return entry, f'The radio stopped answering CAT at {peer["id"]}.'
    moves, why = disposition(mode, entry['returncode'], record)
    return entry, why if moves == 'stop' else None


@dataclass
class Rescue:
    """Which frequencies have had the rig's mains cycled under them, and when to stop.

    ONE RESET PER FREQUENCY PER CAMPAIGN, and the key is the frequency and not the
    station: the tuner stores a match per segment, so a station on three channels
    is three first visits.

    THAT IS THE WHOLE BUDGET BECAUSE THE CAMPAIGN IS ALSO THE TUNE PASS. The ATU
    hunts a match whenever the end-fed wire moves -- a breeze, ice -- and the RFI
    off the hunt takes the radio's CAT interface down with it, so a fault can
    arrive at any channel at any time, including one worked for years. It hunts at
    every channel when it has lost its stored matches outright, which this tuner
    does to its own RFI. Either way the shape is the same: the channel seeks, the
    reset brings CAT back, the arm that follows flies, and the match is stored.
    Walking the list re-tunes it.

    A SECOND RESET AT ONE FREQUENCY ENDS THE CAMPAIGN. The match is not being held
    there or the rig has a fault, and neither is something more cycling repairs.
    So does a cycle that does not bring CAT back at all: the reset is spent only
    when the radio returns AND the arm after it flies.
    """
    reset: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def run(self, call, entry):
        """Cycle the mains under one interrupted channel. Returns why to stop, or None.

        THE LINE IS AN ENVIRONMENTAL OBSERVATION as much as an error. What makes
        the tuner hunt is the wire moving, so these instants are the only record
        this station keeps of when that happened -- which is why the channel, the
        frequency, the wall clock, how long CAT took to come back and what the
        plug's meter read on the way through are all in it, ready to be read
        against a weather record later.
        """
        peer, at = call.peer, datetime.now(timezone.utc)
        print(f'\n!! RIG FAULT at {peer["id"]}, {peer["centre_hz"]} Hz, '
              f'{at:%Y-%m-%dT%H:%M:%SZ} -- {entry["interrupted"]}\n'
              '   Nothing was keyed here. The ATU hunts a match whenever the wire moves '
              'and the RFI off that hunt takes CAT down with it; mains is the way back.',
              flush=True)
        if peer['centre_hz'] in self.reset:
            return (f'{peer["id"]} has now asked for a second CAT reset at '
                    f'{peer["centre_hz"]} Hz in this campaign, having already had one. A '
                    'frequency that needs the mains twice is not holding a match at the '
                    'tuner, or the rig has a fault -- and neither is repaired by cycling '
                    'it again. The campaign stops here.')
        try:
            back = station.recover_cat(say=lambda line: print(line, flush=True))
        except (OSError, RuntimeError) as trouble:
            back = dict(power=[], dial_hz=None, cat_back_s=None, recovered=False,
                        refused=f'{type(trouble).__name__}: {trouble}')
            print(f'   THE MAINS CYCLE DID NOT RUN: {trouble}', flush=True)
        back.update(at=at.isoformat(), peer_id=peer['id'], key=call.key,
                    centre_hz=peer['centre_hz'], fault=entry['interrupted'])
        entry['recovery'] = back
        self.log.append(back)
        self.reset[peer['centre_hz']] = peer['id']
        print('   ' + (f'CAT answers again on dial {back["dial_hz"]} after '
                       f'{back["cat_back_s"]}s; flying {peer["id"]} again'
                       if back['recovered'] else
                       f'THE RADIO DID NOT ANSWER CAT {back["cat_back_s"]}s after the mains '
                       'came back'), flush=True)
        if not back['recovered']:
            return (f'The radio did not answer CAT after a mains cycle at {peer["id"]}. '
                    'Cycling it again will not fix what a cycle did not fix, so the '
                    'campaign stops rather than switching the mains all night.')
        return None


def execute(calls, campaign, records, now):
    """One arm at a time, through the same plan/fly path the `fly` verb uses.

    `campaign` is the record's header, which whoever planned the pass wrote; this
    adds the cycles to it and saves it after every arm.
    """
    records.mkdir(parents=True, exist_ok=True)
    path = records / f'{now:%Y%m%dT%H%M%SZ}.json'
    cycles_asked = campaign['cycles_asked']
    campaign.update(cycles=[], answered_on={}, stopped_because=None, recoveries=[])
    rescue = Rescue(log=campaign['recoveries'])

    def save():
        path.write_text(json.dumps(campaign, indent=2) + '\n')

    save()
    print(f'Campaign record: {path}', flush=True)
    remaining, stop = list(calls), None
    try:
        for number in range(1, cycles_asked + 1):
            if not remaining:
                break
            # DRAWN AGAIN EVERY CYCLE, not once. A station held at one place in the
            # pass meets the same point in the propagation curve and the same
            # elapsed time since the start at every pass, and a single shuffle
            # only moves that bias rather than removing it. The draw crosses
            # bands and modes for that same reason; `reshuffle` says why.
            if campaign.get('order') == 'random':
                remaining = roster.reshuffle(remaining, campaign['bands_m'])
            cycle = dict(cycle=number, arms=[],
                         called=list(dict.fromkeys(call.peer['id'] for call in remaining)))
            campaign['cycles'].append(cycle)
            print(f'\n=== cycle {number} of {cycles_asked}: '
                  f'{len(remaining)} channel(s) still unanswered ===', flush=True)
            for call in list(remaining):
                # A station can hold several channels, and the answer came on one of
                # them earlier in this same pass. Its question is settled; the rest of
                # its channels are not three more chances to ask it.
                if call.key in campaign['answered_on']:
                    continue
                # A CHANNEL A RIG FAULT ENDED IS FLOWN AGAIN, not scored. The arm
                # read nothing, so there is nothing to score; the mains go off and
                # on under it and the same channel gets its turn back.
                while True:
                    entry, stop = fly_one(call)
                    cycle['arms'].append(entry)
                    print(f'  {call.peer["id"]}{" " + call.asks if call.asks else ""}: '
                          + ('moved on' if entry.get('moved_on') else
                             'interrupted' if entry.get('interrupted') else entry['outcome'])
                          + (' -- dropped from the list'
                             if entry['outcome'] == modes.ANSWERED else ' -- stays in the list')
                          + (f' ({entry["note"]})' if entry.get('note') else ''), flush=True)
                    save()
                    if not entry.get('interrupted'):
                        break
                    stop = rescue.run(call, entry)
                    save()
                    if stop:
                        break
                if entry['outcome'] == modes.ANSWERED:
                    campaign['answered_on'][call.key] = number
                    remaining = [left for left in remaining if left.key != call.key]
                    save()
                if stop:
                    break
            if stop:
                break
    finally:
        campaign['stopped_because'] = (
            stop if stop else 'Every channel answered and the list emptied.'
            if not remaining else f'{len(campaign["cycles"])} cycle(s) run.')
        campaign['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    # WRITTEN AS WELL AS PRINTED, from one text. The terminal loses it; the
    # document beside the record is what the next campaign, and the operator
    # tomorrow, actually read. A report that fails to render must not take the
    # record with it -- the JSON above is already saved and is the evidence.
    keys = list(dict.fromkeys(call.key for call in calls))
    try:
        report.write(campaign, keys, path.with_suffix('.md'))
        print(f'Report: {path.with_suffix(".md")}')
    except Exception as trouble:
        print(f'The report could not be rendered ({type(trouble).__name__}: {trouble}); '
              f'the record it reads is intact at {path}.', file=sys.stderr, flush=True)
    print(f'Record: {path}')
    # The record and the report are written first, and only then does the pass
    # report itself interrupted: a hand stop ends the campaign, and the console
    # above turns that into a status that reads as interrupted rather than failed.
    if stop == STOPPED_BY_HAND:
        raise KeyboardInterrupt
    return 2 if stop else 0


def preview(calls, heading, notes, now):
    """The whole pass, said out loud, with nothing opened.

    `heading` is whoever planned the pass saying what it is; everything below it
    is read off the calls themselves, so a pass that spans modes prints its
    channels through each channel's own mode.
    """
    for line in heading:
        print(line)
    for note in notes:
        print(f'  NOTE: {note}')
    width = max(len(call.peer['id']) for call in calls)
    for number, call in enumerate(calls, 1):
        peer = call.peer
        plan = arm.plan(call.mode, step_args(call.args, peer), now)
        print(f'  {number:>2}  {peer["id"]:{width}}  {peer["band_m"]:>3} m  '
              f'centre {peer["centre_hz"]}{roster.offset(peer)}  '
              f'hours {peer.get("hours", "unstated")}'
              + roster.distance_label(peer, call.args)
              + (f'  {call.asks}' if call.asks else ''))
        if call.why:
            print(f'      {call.mode.name} -- {call.why}')
        print(f'      {arm.spelled(plan)}')
    bands = list(dict.fromkeys(call.peer['band_m'] for call in calls))
    keys = list(dict.fromkeys(call.key for call in calls))
    cycles = calls[0].args.cycles
    scanned = [f'{band} m' for band in bands]
    print(f'{len(calls)} candidate(s)'
          + (f' at {len(keys)} station(s)' if len(keys) != len(calls) else '')
          + f', at most {len(calls) * cycles} arms: a pass '
          + (f'draws {", ".join(scanned)} interleaved and calls'
             if len(scanned) > 1 and calls[0].args.order in roster.CROSSES_BANDS else
             f'scans {", ".join(scanned)} in that order, never interleaved, and calls'
             if len(scanned) > 1 else f'scans {scanned[0]} and calls')
          + ' the first channel it finds clear. A station that answers is dropped '
            'from the list.')
    if calls[0].args.order == 'random':
        print('The numbering above is one draw, not the running order: a random pass '
              'draws again at every cycle, so this list is the shape of the pass and '
              'not the sequence. Each cycle records the order it actually flew.')
    if len(asked(calls)) > 1:
        print(f'Each channel is called at every one of {", ".join(asked(calls))} it may '
              'take, in the order the line named them, before the pass moves on to the '
              'next -- and an answer at any of them settles the station, so the rest are '
              'never flown.')
    for line in spell_cost(calls):
        print(line)
    manifests = {call.args.qualify for call in calls if getattr(call.args, 'qualify', None)}
    if manifests:
        print('Every arm is checked against '
              + ', '.join(sorted(Path(m).name for m in manifests)) + ' before the lock, and '
              'the first one the interpreter does not resolve out of that snapshot ends the '
              'campaign. Nothing is imported in a preview.')
    print('Preview only: nothing opened a port, a radio or an audio device.')
    return 0


def asked(calls):
    """What the arms of this pass ask for beyond the channel, each phrase once."""
    return list(dict.fromkeys(call.asks for call in calls if call.asks))


def spell_cost(calls):
    """What a full pass costs in scanning, per the assessor each channel's mode has.

    A channel nothing will be asked about costs no dwell and is counted apart,
    rather than folded into a figure it did not contribute to.
    """
    read = [call for call in calls
            if not (call.args.force or not call.args.assess or call.mode.assess is None)]
    if not read:
        yield (f'Expects to attempt all {len(calls)}: nothing is assessed '
               + ('under --force.' if any(call.args.force for call in calls)
                  else 'with --no-assess.'))
        return
    stops = [max(call.args.dwell, call.mode.dwell_floor_s) for call in read]
    spread = sorted(set(stops))
    short = sorted({f'the {call.args.dwell:g}s dwell asked for is shorter than anything '
                    f"{call.mode.name}'s assessor can read" for call in read
                    if call.args.dwell < call.mode.dwell_floor_s})
    yield (f'Expects to attempt every channel that reads clear. One full pass costs '
           f'about {(sum(stops) + len(read) * RETUNE_S) / 60:.1f} min of scanning if '
           f'none of them does ({len(read)} stops at '
           + (f'{spread[0]:g}s' if len(spread) == 1 else f'{spread[0]:g}-{spread[-1]:g}s')
           + f' plus {RETUNE_S:g}s to retune'
           + ('; ' + '; '.join(short) if short else '')
           + '); each opening found adds an arm on top of that.')
    named = list(dict.fromkeys(call.mode.name for call in read))
    yield ('Before each stop the campaign asks '
           + (named[0] if len(named) == 1 else
              'the mode that would transmit at it (' + ', '.join(named) + ')')
           + ' what is on the channel; nothing is measured in a preview.')
    if blind := [call for call in calls if call not in read]:
        yield (f'{len(blind)} of them is assessed by nothing and attempted as it comes: '
               + ', '.join(sorted({call.mode.name for call in blind}))
               + ' has no assessor behind the `assess` seam.')


def run(mode, args, now):
    if not args.band:
        raise ValueError('--band is required: name the band(s) this campaign may call. '
                         'A campaign never calls a band by omission.')
    if args.cycles < 1:
        raise ValueError('--cycles takes a whole number of passes, at least one.')
    channels, notes = roster.channels(mode, args.band, now, skip=args.skip,
                                      limit=args.limit, order=args.order, args=args)
    if not channels:
        raise ValueError(f'No {mode.name} channel on '
                         f'{"/".join(str(band) for band in args.band)} m is advertised open '
                         f'at {now.hour:02d} UTC after the selection filters and --limit.')
    # A single-mode pass keys on the channel, which is what it has always done, and
    # a channel is worth one arm for every way its mode may call it.
    calls = [Call(mode, one, peer, key=peer['id'], asks=asks)
             for peer in channels for one, asks in modes.arms_at(mode, args, peer)]
    if args.execute:
        campaign = dict(started_utc=now.isoformat(), mode=mode.name, level=args.level,
                        fly_mode=fly_mode(args), bands_m=args.band, order=args.order,
                        skip=args.skip, limit=args.limit, cycles_asked=args.cycles,
                        asks=asked(calls), hours_notes=notes,
                        candidates=[peer['id'] for peer in channels])
        campaign.update(roster.selection(args))
        return execute(calls, campaign, mode.arms / 'campaigns', now)
    bands = '/'.join(f'{band} m' for band in args.band)
    return preview(calls, [
        f'Campaign: {mode.name} at level {args.level} on {bands}, ordered by {args.order}, '
        f'{fly_mode(args)}, {args.cycles} cycle(s).',
        f'Hours checked against {now:%Y-%m-%d %H:00} UTC; roster compiled '
        f'{roster.compiled_utc(mode)}.'], notes, now)
