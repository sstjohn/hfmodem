# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Robust Packet Winlink arms run the station's own mail client over AX.25."""
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import re

from hfcapture import GATEWAYS_URL, distance_km, fetch_gateways, grid_to_latlon, mode_name
from hfmodem.core.band import AMATEUR_HF_BANDS, DIAL_OFFSET_HZ
from hfmodem.core.regulatory.part97 import DATA_SEGMENTS, Licence
from hfmodem.core.tx_events import read_line

from . import ardop, operator, vara
from .modes import CLEARED, Mode
from .roster import load, read_json
from .station import launcher_command, ONAIR, winlink_password


def band_m(hz):
    return next((band for lo, hi, band in AMATEUR_HF_BANDS
                 if lo * 1e6 <= hz <= hi * 1e6), None)


HERE = operator.mode_root('rpr')
# Band order and General RTTY/data segments by band (§97.301(d), §97.305(c)).
SEGMENTS = {band_m(s.lo): (s.lo, s.hi) for s in DATA_SEGMENTS[Licence.GENERAL]}
# §97.303(h): discrete 60 m data channels require these exact centres.
# The 2K80J2D carrier is 1.5 kHz below the centre, matching the USB dial offset.
SIXTY_METRE_CHANNELS = (5_332_000, 5_348_000, 5_373_000, 5_405_000)
# §97.305(c): these rows list phone and image only; the three 10 m rows are merged.
PHONE_IMAGE = ((3_600_000, 4_000_000), (7_125_000, 7_300_000),
               (14_150_000, 14_350_000), (18_110_000, 18_168_000),
               (21_200_000, 21_450_000), (24_930_000, 24_990_000),
               (28_300_000, 29_700_000))
AUTHORISED_WATTS = 50.0
LEVELS = ardop.LEVELS
# Level 0 carries about 300 bit/s; allow 20 minutes for connection and mail.
MAIL_TIMEOUT_S = 1200
MILESTONES = (
    ('cleared', CLEARED),
    ('keyed', re.compile(r'PTT ON|tx_event .*"outcome"\s*:\s*"sent"')),
    ('answer', re.compile(r'\brpr RX from ([A-Z0-9-]+) control=0x[0-9a-f]+')),
    ('connect', re.compile(r'\bCONNECTED [A-Z0-9-]+')),
    ('greeting', re.compile(r'\bmail: remote SID ')),
    ('mail', re.compile(r'\bmail: exchange complete')),
)


def arguments(parser):
    parser.add_argument('--mail-timeout', type=int, default=MAIL_TIMEOUT_S, metavar='S')
    parser.add_argument('--mail-password-file', metavar='PATH')


def head(mode, peer, args, name):
    return launcher_command(mode.launcher) + [peer['call'], str(ardop.called(peer)), '--log', name]


def flags(peer, level, args, directory, send):
    words = ['--mail-fetch', '--mail-timeout', str(args.mail_timeout)]
    if send:
        words += ['--mail-send', str(send)]
    password = str(Path(args.mail_password_file).expanduser().resolve()) if args.mail_password_file else None
    if password:
        words += ['--mail-password-file', password]
    return words, dict(bw_hz=500, called_centre_hz=ardop.called(peer),
                       offset_hz=peer.get('offset_hz', 0), mail_timeout_s=args.mail_timeout,
                       mail_password_file=password, advertised_hours_utc=peer.get('hours'),
                       roster_retrieved_utc=(read_json(operator.mode_root('rpr')/'peers.json').get('retrieved_utc')
                                             if (operator.mode_root('rpr')/'peers.json').is_file() else None))


def ready(mode, plan):
    return winlink_password(plan['adjustments']['mail_password_file'])


def assess(mode, peer, dwell_s=vara.WINDOW_S):
    return vara.sense('rpr', dict(peer, dial_hz=peer['dial_hz'] + peer.get('offset_hz', 0)), dwell_s)


def verdict(mode, directory):
    log = Path(directory) / 'arm.log'
    if not log.is_file():
        return dict(reached=[], line='no transcript', keyed=False, connected=False, stops=[])
    text = vara.ANSI.sub('', log.read_text(encoding='utf-8', errors='replace'))
    reached = [name for name, pattern in MILESTONES if pattern.search(text)]
    # Closing identification can be the only completed transmission after the exchange fails.
    if 'keyed' in reached and 'PTT ON' not in text:
        if not any((event := read_line(line)) and event['protocol'] in {'rpr', 'ident'} and
                   event['event'] == 'completed' and event['outcome'] == 'sent'
                   for line in text.splitlines()):
            reached.remove('keyed')
    call = re.search(r'RPR mail \S+ <-> (\S+),', text)
    if 'answer' in reached and (not call or not re.search(
            rf'\brpr RX from {re.escape(call[1])} control=', text)):
        reached.remove('answer')
    ended = ardop.ENDED.findall(text)
    if 'mail' in reached and ('mail: exchange failed' in text or
            ended and ended[-1].strip() != 'the exchange finished and the link was closed'):
        reached.remove('mail')
    said = ended or ardop.REFUSED.findall(text)
    return dict(reached=reached, keyed='keyed' in reached, connected='connect' in reached,
                answered='answer' in reached or 'connect' in reached,
                mail='complete' if 'mail' in reached else 'unknown',
                line=said[-1].strip() if said else 'no session line',
                stops=[line.strip() for line in text.splitlines() if vara.LAUNCHER_STOP.search(line)])


def readout(mode, record, text):
    return ardop.readout(mode, record, text)


def mhz(hz):
    whole, fraction = divmod(hz, 1_000_000)
    decimals = f'{fraction:06d}'.rstrip('0').ljust(3, '0')
    return f'{whole}.{decimals}'


def warning(centre, licence):
    lo, hi = centre - 255, centre + 255
    segments = DATA_SEGMENTS[licence]
    if (any(s.contains(lo, hi) for s in segments)
            or (licence is not Licence.TECHNICIAN and centre in SIXTY_METRE_CHANNELS)):
        return None
    outside = next((s.hi + 1 for s in segments if s.lo <= lo <= s.hi), lo)
    band = band_m(outside)
    name = licence.value.title()
    # §97.301(b)–(e) gives privileges for Extra, Advanced, General and Technician/Novice.
    paragraph = {Licence.EXTRA: 'b', Licence.ADVANCED: 'c',
                 Licence.GENERAL: 'd', Licence.TECHNICIAN: 'e'}[licence]
    phone = next(((a, b) for a, b in PHONE_IMAGE if a <= outside <= b), None)
    if phone:
        a, b = phone
        reason = f'{mhz(a)}–{mhz(b)} MHz lists phone and image only (§97.305(c))'
    elif band is None:
        reason = f'{mhz(outside)} MHz is not in a US amateur band (§97.301)'
    elif licence is Licence.TECHNICIAN:
        citation = '§97.301(e), §97.307(f)(9)' if band in (80, 40, 15) else '§97.301(e)'
        reason = f'{band} m has no {name} data privileges ({citation})'
    elif band == 60:
        reason = ('60 m data is 5.3515–5.3665 MHz or centred on '
                  '5.332, 5.348, 5.373 or 5.405 MHz (§97.303(h))')
    else:
        start = next(s.lo for s in DATA_SEGMENTS[licence] if band_m(s.lo) == band)
        reason = f'{band} m {name} data starts at {mhz(start)} MHz (§97.301({paragraph}))'
    return f'outside the US {name} data privileges: {reason}'


def listed(text, home=None, *, licence=Licence.GENERAL):
    found = {}
    for row in csv.DictReader(io.StringIO(text.lstrip('\ufeff'))):
        if mode_name(row['Mode'].strip()) != 'Robust Packet':
            continue
        khz, unit = row['Frequency'].replace(',', '').split()
        centre = round(float(khz) * {'khz': 1000, 'mhz': 1_000_000, 'hz': 1}[unit.lower()])
        call = row['Callsign'].strip()
        band = band_m(centre)
        grid = row['GridSquare'].strip()
        peer = dict(id=f'{call}-{band}m-{centre}-500', call=call, band_m=band,
                    centre_hz=centre, dial_hz=centre-DIAL_OFFSET_HZ, bw_hz=500,
                    advertised_mode=row['Mode'], hours=row['Hours'].strip(), grid=grid,
                    qth=row['QTH'].strip(), status_timestamp=row['Timestamp'])
        if reason := warning(centre, licence):
            peer['warning'] = reason
        if home and grid:
            peer['distance_km'] = round(distance_km(home, grid_to_latlon(grid)), 1)
        found[peer['id']] = peer
    order = {band: index for index, band in enumerate(SEGMENTS)}
    return sorted(found.values(), key=lambda p: (order.get(p['band_m'], len(order)), p['centre_hz'], p['call']))


def rebuild(mode, text, source=None):
    listing = load(mode)
    was = {p['id']: p for p in listing['peers']}
    cfg = operator.identity()
    assumed = cfg.regulatory != 'part97'
    licence = Licence.GENERAL if assumed else Licence(cfg.regulatory_settings['licence'])
    peers = listed(text, grid_to_latlon(cfg.grid), licence=licence)
    for peer in peers:
        peer.update({f: was[peer['id']][f] for f in ardop.HAND_KEPT if f in was.get(peer['id'], {})})
    listing.update(peers=peers, station=f'{cfg.mycall} {cfg.grid}', local_grid=cfg.grid,
                   source=source or GATEWAYS_URL,
                   retrieved_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    mode.roster.parent.mkdir(parents=True, exist_ok=True)
    mode.roster.write_text(json.dumps(listing, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    fresh = {p['id']: p for p in peers}
    changes = [f'added   {key}' for key in fresh if key not in was]
    changes += [f'dropped {key}' + ''.join(f'; its {f} was {was[key][f]!r}' for f in ardop.HAND_KEPT if f in was[key])
                for key in was if key not in fresh]
    for key in fresh:
        if key not in was:
            continue
        diff = ardop.moved(was[key], fresh[key])
        if 'warning' in was[key] and 'warning' not in fresh[key]:
            diff.append(f'warning {was[key]["warning"]!r} -> None')
        if diff:
            changes.append(f'changed {key}: {", ".join(diff)}')
    marked = [f'marked {p["call"]} centre {p["centre_hz"]}: {p["warning"]}'
              for p in peers if 'warning' in p]
    assumption = ['station file records no licence class; assuming US General'] if assumed else []
    return ([f'{len(peers)} RPR channels from {source or GATEWAYS_URL}, written to {mode.roster}']
            + assumption + marked + (changes or ['nothing added, dropped or changed']))


def refresh(mode):
    return rebuild(mode, fetch_gateways('Robust Packet'))


PROFILE = Mode(name='rpr', launcher='rpr-mail', entry=('hfmodem.station.process', str(ONAIR.parent/'rpr_mail.py')),
               roster=HERE/'peers.json', arms=HERE/'arms', lock=HERE/'arm.lock',
               bands=tuple(SEGMENTS), levels=LEVELS, phases=(), drives=(),
               authorised_watts=AUTHORISED_WATTS, flags=flags, arguments=arguments,
               campaign_arguments=arguments, milestones=MILESTONES, head=head, ready=ready,
               verdict=verdict, readout=readout, assess=assess, dwell_floor_s=vara.WINDOW_S,
               disposition=vara.disposition, outcome=vara.outcome, refresh=refresh)
