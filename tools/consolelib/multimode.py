# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""ONE CAMPAIGN OVER A BAND, calling every station in the mode that is right for it.

"80 is open, run a campaign with everyone in the correct mode for them" is one
pass, not a PACTOR campaign followed by a VARA one. The loop is `campaign.py`'s
and is not repeated here. What this file decides is the three things that only
arise once a pass spans modes.

ONE VIEW, TWO ROSTERS, AND NEITHER IS FLATTENED. There is no third roster file.
`roster.channels` is asked once per mode, with that mode's own arguments, so
`channel_rank`, the advertised hours, the spacing figure and the codec-gain
points all keep the shape the mode wrote them in -- the VARA snapshot never has
to satisfy PACTOR's shape or the other way round. What comes back is indexed by
the one thing both rosters spell alike, the station: a callsign on a band. A
`campaign.Call` then carries the mode, that mode's arguments and its own peer
record whole, so the arm, the assessor and the launcher line are built out of the
same dict they have always been built out of.

WHICH MODE IS CORRECT FOR A STATION: THE STATION'S OWN HISTORY SAYS. Most
stations are listed by one roster and there is nothing to decide. For the overlap
there is evidence, and it is evidence rather than a proxy: KB5LZK's 7101600
publishes VARA-2750 and PACTOR alike, PACTOR works there, and VARA has never once
connected because KB5LZK's own PACTOR session holds the channel. Nothing about
the roster says that; the arms do. `logbook.py` is already one view over every
mode's arms, so that is where this reads, and a milestone past our own sense and
keying is what counts as an answer. Where the
arms say nothing, the rosters' own advertisement decides, and where they say
nothing either, the order the operator named the modes in does.

THE CHOICE IS MADE ONCE, BEFORE THE FIRST ARM. Re-deciding each cycle out of a
logbook the pass is itself writing would make the preview a description of only
the first pass. The preview names the mode and the reason for every station, and
that is what flies.

WITHIN A BAND, GROUP BY MODE. A band change already ends a group, because it
recalls a different ATU memory and a different propagation picture. A mode change
costs something of its own, and it is in `onair.sh` rather than in the physics:
the `pactor` verb opens with `rigctld_stop` because shrike opens the CAT port
itself, and `vara-mail` opens with `rigctld_start none`. So every boundary pays a
daemon teardown or a start -- and the start is not merely slow, it `die`s when
4532 is still listened on by something the stop could not clear, which costs the
arm. Grouping pays that at most once per band per mode instead of once per
channel.

WHAT IS ON THE CHANNEL IS ASKED OF THE MODE THAT WOULD TRANSMIT. That falls out
of `campaign.Call` and needed nothing here: PACTOR's assessor judges the band its
own emission fills and VARA's judges the width its channel is listed at, so the
same frequency is read differently depending on which mode is about to key on it,
which is the whole point of the `assess` seam.
"""
from __future__ import annotations

from . import campaign, logbook, modes, roster

#: Where a pass that belongs to no single mode writes its record. A mode's arms
#: directory is the mode's, and a multimode campaign has no business claiming one
#: of them or writing the same table into both.
from .operator import data_root
RECORDS = data_root() / 'campaigns'

#: The substrate's own campaign flags, which every mode in a pass must be given
#: alike. A mode's own section may carry `--level` and the flags that mode's
#: `arguments` adds, and nothing else: a pass where one mode was told
#: `--cycles 3` and another `--cycles 1` is two campaigns wearing one name.
SHARED = ('band', 'cycles', 'order', 'skip', 'limit', 'dwell', 'assess', 'sense', 'pounce',
          'force', 'execute', 'dry_run', 'witness', 'witness_seconds', 'kiwi', 'retune',
          'tap_tail', 'send', 'only', 'prefer', 'from_grid',
          'min_distance_km', 'max_distance_km')

MODE_SWITCH = (
    'Within a band the pass also finishes one mode before it starts the next, and that '
    "cost is in onair.sh rather than in the physics: the `pactor` verb opens by stopping "
    'rigctld, because shrike owns the CAT port itself, and `vara-mail` opens by starting '
    'it. onair.sh bounds the stop at about 13 s of waiting on the process, the socket and '
    'the CAT port, and gives the start twelve tries to answer -- and a start that finds '
    '4532 still held refuses the arm outright, which costs the arm and not the seconds. '
    'Grouping pays that once per mode per band, not once per channel.')


def advertised(listed, names):
    """With no arms at this station, what the rosters themselves say.

    A VARA listing carries the contacts the gateway logged in the last day; a
    PACTOR channel taken off an assessed-station manifest carries no such count
    and reads as none, which is an absence and is said as one.
    """
    lately = {name: max((peer.get('activity_count_24h') or 0) for peer in listed[name])
              for name in names}
    busiest = max(names, key=lambda name: lately[name])
    if lately[busiest] > min(lately.values()):
        return busiest, (f'no arms here; {busiest} advertises {lately[busiest]} contact(s) '
                         'in the last day and the other roster none')
    return names[0], ('no arms here and no roster advertises recent activity; '
                      f'{names[0]} because you named it first')


def decide(listed, seen, order, proven=()):
    """Which mode to call one station in, and the one phrase that says why.

    `proven` is the modes with an answer on record anywhere. A mode that has none
    is not the untried one to reach for while another listing here has answered
    somewhere: a new mode would otherwise take every station the old ones have
    not yet won.
    """
    names = [name for name in order if name in listed]
    if len(names) == 1:
        return names[0], f'only {names[0]} lists it'
    tally = {name: seen.get(name) or dict(arms=0, milestones=0, latest='') for name in names}
    said = ', '.join(f'{name} answered {tally[name]["milestones"]} of {tally[name]["arms"]}'
                     for name in names)
    if answered := [name for name in names if tally[name]['milestones']]:
        best = max(answered, key=lambda name: (tally[name]['milestones'] / tally[name]['arms'],
                                               tally[name]['latest']))
        return best, said
    reach = [name for name in names if name in proven] or names
    if untried := [name for name in names if not tally[name]['arms'] and name in reach]:
        if any(tally[name]['arms'] for name in names):
            return untried[0], f'{said}; {untried[0]} is the one still untried here'
        return advertised(listed, names)
    unproven = [name for name in names if not tally[name]['arms'] and name not in reach]
    return names[0], (f'{said}; {names[0]} because you named it first'
                      + (f'; {", ".join(unproven)} has answered nowhere yet' if unproven else ''))


def plan(chosen, now):
    """The calls a pass makes: one mode per station, grouped by band and then by mode.

    `--limit` caps the pass rather than each mode's share of it, so it is applied
    after the grouping; every other roster flag is the mode's own to apply.
    """
    shared = chosen[0][1]
    order = [mode.name for mode, _ in chosen]
    offered, notes = {}, []
    for mode, args in chosen:
        channels, said = roster.channels(mode, args.band, now, skip=args.skip,
                                         order=args.order, args=args)
        offered[mode.name] = channels
        notes += [f'{mode.name}: {note}' for note in said]
    listed = {}
    for name in order:
        for peer in offered[name]:
            listed.setdefault((peer['call'].upper(), peer['band_m']), {}) \
                  .setdefault(name, []).append(peer)
    seen = logbook.answered_at(order)
    proven = {name for tally in seen.values() for name, said in tally.items()
              if said['milestones']}
    picked = {at: decide(offer, seen.get(at, {}), order, proven)
              for at, offer in listed.items()}
    calls = []
    for band in shared.band:
        for mode, args in chosen:
            for peer in offered[mode.name]:
                at = (peer['call'].upper(), peer['band_m'])
                if peer['band_m'] != band or picked[at][0] != mode.name:
                    continue
                calls += [campaign.Call(mode, one, peer, key=f'{at[0]} {band} m',
                                        why=picked[at][1], asks=asks)
                          for one, asks in modes.arms_at(mode, args, peer)]
    if getattr(shared, 'prefer', None):
        place = {band: index for index, band in enumerate(shared.band)}
        calls.sort(key=lambda call: (place[call.peer['band_m']],
                                    roster.preference(call.peer, shared)))
    return (calls[:shared.limit] if shared.limit else calls), notes, picked


def agree(chosen):
    """Every mode in a pass takes the same substrate flags, or it is two campaigns."""
    first, rest = chosen[0], chosen[1:]
    for name in SHARED:
        wanted = getattr(first[1], name, None)
        for mode, args in rest:
            if getattr(args, name, None) != wanted:
                raise ValueError(
                    f'--{name.replace("_", "-")} was given to {first[0].name} as '
                    f'{wanted!r} and to {mode.name} as {getattr(args, name)!r}. The '
                    'substrate flags belong before the first --with, where every mode '
                    'reads the same ones; only --level and a mode\'s own flags go after it.')
    return first[1]


def heading(chosen, picked, now):
    shared = chosen[0][1]
    bands = '/'.join(f'{band} m' for band in shared.band)
    compiled = ', '.join(f'{mode.name} {roster.compiled_utc(mode)}' for mode, _ in chosen)
    return ['Campaign: '
            + ', '.join(f'{mode.name} at level {args.level}' for mode, args in chosen)
            + f' on {bands}, ordered by {shared.order}, {campaign.fly_mode(shared)}, '
              f'{shared.cycles} cycle(s).',
            f'Each of {len(picked)} station(s) is called in one mode, decided before the '
            'first arm by its own arms and then by what its roster advertises; a station '
            'that answers leaves the list whichever mode raised it.',
            f'Hours checked against {now:%Y-%m-%d %H:00} UTC; rosters compiled {compiled}.',
            ('Geographic preference orders stations across modes within each band; '
             'this can require extra modem switches.' if getattr(shared, 'prefer', None)
             else MODE_SWITCH)]


def run(chosen, now):
    shared = agree(chosen)
    if not shared.band:
        raise ValueError('--band is required: name the band(s) this campaign may call. '
                         'A campaign never calls a band by omission.')
    if shared.cycles < 1:
        raise ValueError('--cycles takes a whole number of passes, at least one.')
    calls, notes, picked = plan(chosen, now)
    if not calls:
        raise ValueError(
            f'No channel of {", ".join(mode.name for mode, _ in chosen)} on '
            f'{"/".join(str(band) for band in shared.band)} m is advertised open at '
            f'{now.hour:02d} UTC after the selection filters and --limit.')
    if not shared.execute:
        return campaign.preview(calls, heading(chosen, picked, now), notes, now)
    record = dict(started_utc=now.isoformat(), modes=[mode.name for mode, _ in chosen],
                  levels={mode.name: args.level for mode, args in chosen},
                  fly_mode=campaign.fly_mode(shared), bands_m=shared.band,
                  order=shared.order, skip=shared.skip, limit=shared.limit,
                  cycles_asked=shared.cycles, asks=campaign.asked(calls),
                  hours_notes=notes,
                  chosen_mode={f'{gateway} {band} m': dict(mode=name, why=why)
                               for (gateway, band), (name, why) in picked.items()},
                  candidates=[call.peer['id'] for call in calls])
    record.update(roster.selection(shared))
    return campaign.execute(calls, record, RECORDS, now)
