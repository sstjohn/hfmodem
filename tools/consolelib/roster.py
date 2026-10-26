# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""FIND A HOST: the channels this station has worked, and which one an arm means.

The roster is data on disk, one `peers.json` per mode: channel, band, centre and
dial, advertised hours, the spacing rule the mode has found works, and the codec
gain that worked on each band by day and by night. A dial is always its centre
minus `core.band.DIAL_OFFSET_HZ`, so a roster that says otherwise is refused
rather than tuned to.
"""
from __future__ import annotations

import json
import math
import random
import re

from hfmodem.core.band import DIAL_OFFSET_HZ

HOURS = re.compile(r'(\d{1,2})\s*-\s*(\d{1,2})')
GRID = re.compile(r'[A-R]{2}[0-9]{2}(?:[A-X]{2}(?:[0-9]{2})?)?', re.I)


def distance_km(peer):
    """One unit for ordering; zero is a real distance, missing is not."""
    for field, scale in (('distance_km', 1), ('distance_mi', 1.609344)):
        try:
            value = float(peer[field]) * scale
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            return value
    return float('inf')


def geography(args):
    return any(getattr(args, field, None) is not None for field in
               ('prefer', 'from_grid', 'min_distance_km', 'max_distance_km'))


def selection(args):
    """Keep the geographic request in the campaign record as well as its preview."""
    return {field: getattr(args, field, None) for field in
            ('only', 'prefer', 'from_grid', 'min_distance_km', 'max_distance_km')}


def distance_label(peer, args):
    if not geography(args):
        return ''
    distance = distance_km(peer)
    return f'  {distance:.0f} km' if math.isfinite(distance) else '  distance unknown'


def preference(peer, args):
    """Preferred distance tier first, then other known distances, then unknown."""
    distance = distance_km(peer)
    if not math.isfinite(distance):
        return (2, distance)
    tier = 'near' if distance < 1000 else 'far' if distance < 4000 else 'dx'
    return (0 if tier == args.prefer else 1, distance)


def locate(listing, args):
    """Recompute from grids when available; never rewrite the saved roster."""
    from hfcapture import distance_km as great_circle, grid_to_latlon
    from . import rigsource

    minimum, maximum = (getattr(args, key, None) for key in
                        ('min_distance_km', 'max_distance_km'))
    for value in (minimum, maximum):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError('Distance bounds must be finite, nonnegative kilometres.')
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError('--min-distance-km must not exceed --max-distance-km.')
    if getattr(args, 'prefer', None) and getattr(args, 'order', 'roster') == 'random':
        raise ValueError('--prefer cannot be combined with --order random.')
    config = rigsource.station_config() if not getattr(args, 'from_grid', None) else None
    origin = getattr(args, 'from_grid', None) or (config.station.grid if config else '')
    if not origin or not GRID.fullmatch(origin):
        raise ValueError('Distance selection needs a valid station grid or --from-grid GRID.')
    home = grid_to_latlon(origin)
    # Saved distances only apply when their recorded origin agrees with this one.
    saved_origin = listing.get('local_grid') or (listing.get('station') or '').split()[-1:]
    if isinstance(saved_origin, list):
        saved_origin = saved_origin[0] if saved_origin else ''
    result = []
    for original in listing['peers']:
        peer = dict(original)
        grid = peer.get('grid', '')
        if isinstance(grid, str) and GRID.fullmatch(grid):
            peer['distance_km'] = great_circle(home, grid_to_latlon(grid))
            peer.pop('distance_mi', None)
        elif saved_origin.upper() != origin.upper():
            peer.pop('distance_km', None)
            peer.pop('distance_mi', None)
        result.append(peer)
    return result, origin


def read_json(path):
    return json.loads(path.read_text())


def load(mode):
    return read_json(mode.roster) if mode.roster.is_file() else {'peers': []}


def peers(mode):
    return load(mode)['peers']


def spacing(mode):
    """How long this mode has found it needs between calls at one gateway.

    A roster taken straight off the gateway listing has no such figure, and a
    mode without one reports the gap and leaves the weighing to the operator.
    """
    return load(mode).get('spacing')


def compiled_utc(mode):
    """When this roster was taken off the gateway listing, however it spells it."""
    listing = load(mode)
    return listing.get('compiled_utc') or listing.get('retrieved_utc')


def rank(mode, peer, args):
    """How well one channel fits this arm, for a mode whose channels are not alike.

    A PACTOR channel is a frequency and nothing else, so every one of them fits
    equally and this is 0. A VARA channel also carries the bandwidth it is
    listed at, an arm asks for one, and a station listed at both is wanted at the
    one it advertises exactly -- which is a rank rather than a filter. None means
    this channel is not one the arm may use at all.
    """
    if mode.channel_rank is None or args is None:
        return 0
    return mode.channel_rank(peer, args)


def peer_for(mode, name, band, args=None):
    """One channel, named by call or by full ID, on a band or on the only band it has."""
    listed = peers(mode)
    if not listed:
        raise ValueError(f'No {mode.name} peers. Run console data init, then console data import-roster {mode.name} FILE.')
    wanted = name.upper()
    ranked = [(fit, p) for p in listed if wanted in (p['id'].upper(), p['call'].upper())
              and (band is None or p['band_m'] == band)
              and (fit := rank(mode, p, args)) is not None]
    best = min((fit for fit, _ in ranked), default=None)
    matches = [p for fit, p in ranked if fit == best]
    if len(matches) != 1:
        raise ValueError(f'Name one channel: {name!r} on {band or "any band"} matches '
                         f'{len(matches)}. Run `console {mode.name} peers` and use a full '
                         f'ID such as {listed[0]["id"]}.')
    peer = matches[0]
    if peer['dial_hz'] != peer['centre_hz'] - DIAL_OFFSET_HZ:
        raise ValueError(f'{peer["id"]} has a dial that is not its centre minus '
                         f'{DIAL_OFFSET_HZ} Hz.')
    return peer


def gain_hint(mode, band_m, now):
    """What the codec gain has read on this band at this hour. A reading, not a setting."""
    night = not 12 <= now.hour < 23
    points = load(mode).get('codec_gain_working_points') or {}
    return points.get(f'{band_m}-{"night" if night else "day"}')


def hours_open(advertised, hour):
    """The roster's "HH-HH" UTC window, inclusive and possibly wrapping midnight.

    Unreadable hours read as open and say so: a channel is never dropped in
    silence over a string nobody could parse.
    """
    match = HOURS.fullmatch(advertised.strip()) if isinstance(advertised, str) else None
    if not match or any(not 0 <= int(part) <= 23 for part in match.groups()):
        return True, f'unreadable advertised hours {advertised!r}; treated as open'
    start, end = (int(part) for part in match.groups())
    return (start <= hour <= end if start <= end else hour >= start or hour <= end), None


#: `channels` sorts on nothing but the key for these, so a pass interleaves both
#: bands and modes. `reshuffle` says why that is right rather than merely cheap.
CROSSES_BANDS = ('random',)

ORDERS = {
    'roster': None,
    # A channel whose distance the roster never recorded goes last, not first.
    'distance': distance_km,
    'activity': lambda peer: (-peer.get('activity_count_24h', 0),
                              peer.get('latest_activity_utc') or ''),
    # The one order that crosses bands; `channels` drops the band from its key.
    'random': lambda peer: random.random(),  # noqa: ARG005
}


def channels(mode, bands, now, skip=(), limit=0, order='roster', args=None):
    """The channels a campaign may call this hour, in the order it should scan them.

    Filtered by band, by the operator's --skip, and by the hours each channel
    advertises; ordered within a band as the roster lists them, by distance, or by
    how lately the channel was seen.

    BANDS ARE NOT INTERLEAVED, except under an order in `CROSSES_BANDS`. A
    retune inside a band is nearly free and the scan finishes one band before it
    starts the next, in the order the bands were named -- which keeps each band's
    arms together for a pass that is reading one. A pass drawing at random wants
    the opposite, and `reshuffle` says why.
    """
    if order not in ORDERS:
        raise ValueError(f'Unknown order {order!r}; choose {", ".join(ORDERS)}.')
    dropped = {name.upper() for name in skip}
    wanted = {name.upper() for name in getattr(args, 'only', ())}
    chosen, notes = [], []
    listing = load(mode)
    candidates = listing['peers']
    geographic = geography(args)
    minimum = getattr(args, 'min_distance_km', None)
    maximum = getattr(args, 'max_distance_km', None)
    if geographic:
        candidates, origin = locate(listing, args)
        notes.append(f'Distances from {origin}; near <1000 km, far 1000–4000 km, '
                     'dx >=4000 km. Geographic preference is not a propagation forecast.')
        if getattr(args, 'prefer', None):
            notes.append(f'Prefer {args.prefer}: that group first, nearest first within '
                         'each group; other known distances follow, unknown last.')
        if minimum is not None or maximum is not None:
            notes.append(f'Distance bounds: {minimum if minimum is not None else 0:g} to '
                         f'{str(maximum) if maximum is not None else "unbounded"} km.')
    unknown = 0
    for peer in candidates:
        if wanted and not wanted & {peer['call'].upper(), peer['id'].upper()}:
            continue
        if peer['band_m'] not in set(bands):
            continue
        if dropped & {peer['call'].upper(), peer['id'].upper()}:
            continue
        if rank(mode, peer, args) is None:
            continue
        open_now, note = hours_open(peer.get('hours', ''), now.hour)
        if note:
            notes.append(f'{peer["id"]}: {note}')
        if not open_now:
            continue
        distance = distance_km(peer)
        if geographic and not math.isfinite(distance):
            unknown += 1
        if minimum is not None or maximum is not None:
            if (not math.isfinite(distance) or
                    (minimum is not None and distance < minimum) or
                    (maximum is not None and distance > maximum)):
                continue
        chosen.append(peer)
    within = ORDERS[order]
    place = {band: index for index, band in enumerate(bands)}
    chosen.sort(key=within if order in CROSSES_BANDS else
                (lambda peer: (place[peer['band_m']], within(peer) if within else 0)))
    if getattr(args, 'prefer', None):
        chosen.sort(key=lambda peer: (place[peer['band_m']], preference(peer, args)))
        for band in dict.fromkeys(bands):
            on_band = [peer for peer in chosen if peer['band_m'] == band]
            if on_band and not any(preference(peer, args)[0] == 0 for peer in on_band):
                notes.append(f'{band} m: no eligible {args.prefer} stations with known '
                             'distance; falling back to the other stations. '
                             'Use distance bounds to exclude that fallback.')
    if unknown:
        notes.append(f'{unknown} eligible channel(s) have unknown distance; '
                     + ('excluded by distance bounds.' if minimum is not None or maximum is not None
                        else 'placed last by --prefer.' if getattr(args, 'prefer', None)
                        else 'kept in the requested order.'))
    return (chosen[:limit] if limit else chosen), notes


def reshuffle(calls, bands):
    """Draw the calling order again, across the bands and inside each mode block.

    A station called at the same place in every pass meets the same point in the
    propagation curve and the same elapsed time since the campaign started every
    time, and that is a bias the order itself put there, so a random campaign
    draws afresh each cycle rather than reusing one shuffle.

    THE DRAW CROSSES BANDS AND MODES, and must, for the same reason. A pass that
    finishes 40 m before it starts 80 m gives every 40 m arm the first half of the
    session and every 80 m arm the second, which confounds the band with the hour
    exactly as a fixed order confounds the station with it. Grouping by mode
    confounds the mode with it the same way. A band hop costs what a within-band
    hop costs -- `RETUNE_S`, two seconds -- because the station's tuner recalls a
    stored match on its own; a mode hop costs a rigctld stop or start, which the
    operator has said to spend.

    A channel still keeps its own arms together, in the order the line asked for
    them, because that is what `campaign.preview` promises a reader. What is
    drawn is where each channel falls in the pass, and nothing else.
    """
    at_channel = {}
    for call in calls:
        at_channel.setdefault(call.peer['id'], []).append(call)
    drawn = list(at_channel.values())
    random.shuffle(drawn)
    return [call for arms in drawn for call in arms]


def offset(peer):
    """A channel called off its listing says so wherever its listed centre is printed."""
    return f' {peer["offset_hz"]:+d} Hz' if peer.get('offset_hz') else ''


def spell_peers(mode):
    for peer in peers(mode):
        said = (peer.get('last_result') or peer.get('latest_activity_utc')
                or 'never called from here')
        yield (f'{peer["id"]:12} centre {peer["centre_hz"]}{offset(peer)} dial {peer["dial_hz"]}  '
               f'{said}')
