# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The ARDOP profile: one Winlink exchange over one held link, `onair.sh ardop-mail`.

besra's `run_server` owns the radio and the B2F session together, and `onair.sh`
gates the channel at the width the line names before it starts. The line is the
one every ARDOP arm on record was pasted as -- `ardop-mail GW CENTRE BW
--mail-fetch --mail-timeout 420` -- so the console builds that and adds only what
the substrate adds to every mode.

The roster is the live ARDOP listing for the gateways this station has called or
ranked, taken whole, so every channel carries the width it is listed at. An arm
calls at that width unless `--bw` names another, and a narrower call is one the
gateway takes: KN4LQN, listed at 2000, has connected six times of six at 500.

The reading of the channel, the three outcomes and when a pass stops are VARA's
unchanged: `ardop-mail` passes the same gate with its own emission and width, and
leaves by the same exit codes and the same refusals.

A channel may carry `offset_hz`, where the gateway was found to answer rather
than where it is listed: KN4LQN answered six of six 8 Hz above its listing. The
arm calls and senses there; the record keeps the listed centre, the offset and
the called centre side by side. `console ardop peers --refresh` rebuilds the
roster from the live listing and keeps `offset_hz` and `last_result` by channel.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re

from hfcapture import GATEWAYS_URL, distance_km, fetch_gateways, grid_to_latlon
from hfmodem.core.band import DIAL_OFFSET_HZ

from . import vara
from .modes import CLEARED, Mode, Readout, highest, spoken
from .roster import read_json
from .station import ONAIR, winlink_password

from .operator import mode_root
HERE = mode_root('ardop')
BANDWIDTHS = (200, 500, 1000, 2000)
# The data segments selected for the bands this profile uses.
SEGMENTS = {80: (3_500_000, 3_600_000), 40: (7_000_000, 7_125_000),
            30: (10_100_000, 10_150_000), 20: (14_000_000, 14_150_000)}
HAND_KEPT = ('last_result', 'offset_hz')
MAIL_TIMEOUT_S = 420

# The rig's own setting on every ARDOP arm on record. Nothing here asks the radio
# for watts, so it stands as the ceiling a preset may name.
AUTHORISED_WATTS = 50.0

LEVELS = {
    'current': dict(note='RFPOWER as the rig sits, read back into the arm rather than set by it'),
}

# A gateway that turns the request down has heard it, and a campaign that read
# that as silence would keep calling a station that already answered. The rest
# unanchored, where the modem's own logger stamps the line.
MILESTONES = (
    ('cleared', CLEARED),
    ('keyed', re.compile(r'PTT ON|INFO TX |\bTX ConReq')),
    ('answer', re.compile(r'INFO RX Con\w+|CONNECTION REJECTED BY ')),
    ('connect', re.compile(r'ARQ CONNECTION ESTABLISHED WITH '
                           r'|^\s*CONNECTED [A-Z0-9/-]+ @ \d+ Hz', re.M)),
    ('greeting', re.compile(r'\bmail: remote SID ')),
    ('mail', re.compile(r'\bmail: exchange complete')),
)
ENDED = re.compile(r'^session ended: (.+)$', re.M)
REFUSED = re.compile(r'^!! (.+)$', re.M)


def called(peer):
    return peer['centre_hz'] + peer.get('offset_hz', 0)


def width(args, peer):
    return args.bw or peer['bw_hz']


def widths(args):
    asked = args.bw if isinstance(args.bw, list) else [args.bw] if args.bw else []
    return list(dict.fromkeys(asked))


def width_rank(peer, bw):
    """The listed width fits best, a narrower one is taken, a wider one is not asked."""
    if peer['bw_hz'] == bw:
        return 0
    return 1 if bw < peer['bw_hz'] else None


def channel_rank(peer, args):
    fits = [fit for fit in (width_rank(peer, bw) for bw in widths(args)) if fit is not None]
    return min(fits, default=None) if widths(args) else 0


def channel_arms(args, peer):
    """One arm for each named width this channel takes, or one at the width it lists."""
    if not widths(args):
        return [(args, '')]
    at = [argparse.Namespace(**dict(vars(args), bw=bw)) for bw in widths(args)]
    return [(one, f'BW{one.bw}') for one in at if width_rank(peer, one.bw) is not None]


def arguments(air):
    air.add_argument('--bw', type=int, choices=BANDWIDTHS,
                     help='the width this arm calls at; default the width the channel '
                          'is listed at')
    varied(air)


def campaign_arguments(many):
    many.add_argument('--bw', type=int, nargs='+', choices=BANDWIDTHS,
                      help="the width(s) this campaign may call at; a channel is called at "
                           "each of them it takes before the pass moves on. Default each "
                           "channel's listed width")
    varied(many)


def varied(air):
    air.add_argument('--mail-timeout', type=int, default=MAIL_TIMEOUT_S, metavar='S',
                     help=f'seconds for the whole exchange, connect included '
                          f'(default {MAIL_TIMEOUT_S})')
    air.add_argument('--mail-password-file', metavar='PATH',
                     help='a credential to put on the line; otherwise it goes into the '
                          "arm's environment and never into the transcript")


def head(mode, peer, args, name):
    """`ardop-mail` reads its width as a bare word before the flags, as `vara-mail` does."""
    return [str(ONAIR), mode.launcher, peer['call'], str(called(peer)),
            str(width(args, peer)), '--log', name]


def flags(peer, level, args, directory, send):
    words = ['--mail-fetch', '--mail-timeout', str(args.mail_timeout)]
    if send:
        words += ['--mail-send', str(send)]
    password = (str(Path(args.mail_password_file).expanduser().resolve())
                if args.mail_password_file else None)
    if password:
        words += ['--mail-password-file', password]
    return words, dict(bw_hz=width(args, peer), listed_bw_hz=peer['bw_hz'],
                       called_centre_hz=called(peer), offset_hz=peer.get('offset_hz', 0),
                       mail_timeout_s=args.mail_timeout, mail_password_file=password,
                       advertised_hours_utc=peer.get('hours'),
                       roster_retrieved_utc=read_json(mode_root('ardop') / 'peers.json').get('retrieved_utc'))


def ready(mode, plan):
    return winlink_password(plan['adjustments']['mail_password_file'])


def assess(mode, peer, dwell_s=vara.WINDOW_S):
    return vara.sense('ardop', dict(peer, dial_hz=peer['dial_hz'] + peer.get('offset_hz', 0)),
                      dwell_s)


def verdict(mode, directory):
    """The milestones, the launcher's refusals, and the session's own last word.

    `session ended:` is run_server's, printed on its way out whatever became of
    the link; an arm the gate refused never started it, and its `!!` line is the
    verdict instead.
    """
    log = Path(directory) / 'arm.log'
    if not log.is_file():
        return dict(reached=[], line='no transcript', keyed=False, connected=False, stops=[])
    text = vara.ANSI.sub('', log.read_text(errors='replace'))
    reached = [name for name, pattern in MILESTONES if pattern.search(text)]
    said = ENDED.findall(text) or REFUSED.findall(text)
    return dict(reached=reached, line=said[-1].strip() if said else 'no session line',
                keyed='keyed' in reached, connected='connect' in reached,
                answered='answer' in reached or 'connect' in reached,
                mail='complete' if 'mail' in reached else 'unknown',
                stops=[line.strip() for line in text.splitlines()
                       if vara.LAUNCHER_STOP.search(line)])


def readout(mode, record, text):
    said = record.get('verdict') or {}
    return Readout(
        rung=highest(mode, said.get('reached')),
        did=(said['line'],) if said.get('line') else (),
        said=spoken(vara.ANSI.sub('', text)),
        crossed='mail exchange complete' if said.get('mail') == 'complete' else '',
        anomalies=tuple((1, f'the launcher refused this channel: {stop}')
                        for stop in said.get('stops') or []),
        caveats=(('the launcher exit code', vara.EXIT_CANNOT_SAY),))


def listed(text, calls, home):
    """The listing's channels for these gateways on this profile's bands, in roster order."""
    found = []
    for row in csv.DictReader(io.StringIO(text)):
        call = row['Callsign'].strip()
        if call not in calls or not row['Mode'].startswith('ARDOP '):
            continue
        khz, unit = row['Frequency'].replace(',', '').split()
        centre = round(float(khz) * (1000 if unit.lower() == 'khz' else 1_000_000))
        band = next((band for band, (low, high) in SEGMENTS.items()
                     if low <= centre <= high), None)
        if band is None:
            continue
        bw, grid = int(row['Mode'].split()[1]), row['GridSquare'].strip()
        found.append(dict(
            id=f'{call}-{band}m-{centre}-{bw}', call=call, bw_hz=bw,
            advertised_mode=row['Mode'], centre_hz=centre, dial_hz=centre - DIAL_OFFSET_HZ,
            band_m=band, grid=grid,
            distance_km=round(distance_km(home, grid_to_latlon(grid)), 1),
            hours=row['Hours'].strip(), qth=row['QTH'].strip(),
            status_timestamp=row['Timestamp']))
    order = list(SEGMENTS)
    return sorted(found, key=lambda peer: (order.index(peer['band_m']), peer['centre_hz'],
                                           peer['call']))


def moved(before, after):
    return [f'{field} {before.get(field)!r} -> {value!r}' for field, value in after.items()
            if field != 'status_timestamp' and before.get(field) != value]


def refresh(mode):
    """Rebuild the roster from the live listing, for the gateways it already names.

    A channel keeps its hand-kept fields by id. A channel that moved is a new id,
    so what was learned at the old one is reported with the drop, not carried.
    """
    listing = read_json(mode.roster)
    was = {peer['id']: peer for peer in listing['peers']}
    home = grid_to_latlon(listing['station'].split()[-1])
    calls = {peer['call'] for peer in was.values()}
    fresh = {peer['id']: dict(peer, **{field: was[peer['id']][field] for field in HAND_KEPT
                                       if field in was.get(peer['id'], {})})
             for peer in listed(fetch_gateways('ARDOP'), calls, home)}
    listing.update(retrieved_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   peers=list(fresh.values()))
    mode.roster.write_text(json.dumps(listing, indent=1, ensure_ascii=False) + '\n')
    said = [f'added   {name}' for name in fresh if name not in was]
    said += [f'dropped {name}' + ''.join(f'; its {field} was {was[name][field]!r}'
                                          for field in HAND_KEPT if field in was[name])
             for name in was if name not in fresh]
    said += [f'changed {name}: {", ".join(diff)}' for name, peer in fresh.items()
             if name in was and (diff := moved(was[name], peer))]
    return [f'{len(fresh)} channel(s) of {len(calls)} gateway(s) from {GATEWAYS_URL}, '
            f'written to {mode.roster}'] + (said or ['nothing added, dropped or changed'])


PROFILE = Mode(
    name='ardop',
    launcher='ardop-mail',
    entry=('hfmodem.besra.host.run_server',),
    roster=HERE / 'peers.json',
    arms=HERE / 'arms',
    lock=HERE / 'arm.lock',
    bands=(80, 40, 30, 20),
    levels=LEVELS,
    phases=(),
    drives=(),
    authorised_watts=AUTHORISED_WATTS,
    flags=flags,
    arguments=arguments,
    campaign_arguments=campaign_arguments,
    milestones=MILESTONES,
    head=head,
    ready=ready,
    channel_rank=channel_rank,
    channel_arms=channel_arms,
    verdict=verdict,
    disposition=vara.disposition,
    assess=assess,
    dwell_floor_s=vara.WINDOW_S,
    outcome=vara.outcome,
    readout=readout,
    refresh=refresh,
)
