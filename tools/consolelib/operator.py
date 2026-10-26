# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Portable operator storage and optional services; station.toml owns identity."""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def data_root():
    return Path(os.environ.get('HFMODEM_DATA_DIR', ROOT / 'operator-data')).expanduser().resolve()


def mode_root(name):
    return data_root() / 'working' / name


def settings():
    path = data_root() / 'operator.toml'
    return tomllib.loads(path.read_text()) if path.is_file() else {}


def station_file():
    return Path(os.environ.get('HFMODEM_STATION', ROOT / 'station.toml')).expanduser().resolve()


def identity():
    from hfmodem.core.config import load
    path = station_file()
    if not path.is_file():
        raise ValueError('Set HFMODEM_STATION to your station.toml; station identity is required.')
    station = load(path).station
    if not station.mycall:
        raise ValueError("Set station.mycall in your station file; no hardware was opened.")
    return station


def delivery_identity(*, station=None):
    from hfmodem.core import config
    sender = config.mycall(str(station or station_file()))
    recipient = os.environ.get('HFMODEM_DELIVERY_TO') or settings().get('delivery_to')
    if not recipient or not isinstance(recipient, str) or any(c.isspace() for c in recipient):
        raise ValueError('Delivery tests need delivery_to in operator.toml or HFMODEM_DELIVERY_TO.')
    return sender, recipient


def witness_inventory():
    named = os.environ.get('HFMODEM_WITNESS_CANDIDATES') or settings().get('witness_candidates')
    return Path(named).expanduser().resolve() if named else data_root() / 'kiwi-candidates.csv'


def run(argv):
    parser = argparse.ArgumentParser(prog='console data', description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('init', help='create empty per-mode rosters without replacing existing data')
    imp = sub.add_parser('import-roster', help='validate and import an explicitly named JSON roster')
    imp.add_argument('mode', choices=('pactor', 'vara', 'ardop'))
    imp.add_argument('path', type=Path)
    imp.add_argument('--replace', action='store_true', help='replace an existing nonempty roster')
    refresh = sub.add_parser('refresh', help='load current public Winlink gateway channels')
    refresh.add_argument('mode', choices=('pactor', 'vara', 'ardop'))
    refresh.add_argument('--csv', type=Path, help='use a downloaded Winlink channel CSV instead')
    field = sub.add_parser('field', help=argparse.SUPPRESS)
    field.add_argument('name', choices=('mycall', 'grid', 'data-root', 'witness-inventory'))
    args = parser.parse_args(argv)
    if args.action == 'refresh':
        from . import gateways
        gateways.refresh(args.mode, csv_path=args.csv)
        return 0
    if args.action == 'field':
        value = (data_root() if args.name == 'data-root' else witness_inventory()
                 if args.name == 'witness-inventory' else getattr(identity(), args.name))
        print(value)
        return 0
    if args.action == 'init':
        for name in ('pactor', 'vara', 'ardop'):
            directory = mode_root(name)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'peers.json'
            if not path.exists():
                path.write_text(json.dumps({'peers': []}, indent=2) + '\n')
        print(f'Operator data: {data_root()}')
        return 0
    from hfmodem.core.band import DIAL_OFFSET_HZ
    data = json.loads(args.path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('peers'), list):
        raise ValueError('Roster must be a JSON object with a peers list.')
    seen = set()
    for peer in data['peers']:
        if not isinstance(peer, dict) or not all(k in peer for k in ('id', 'call', 'band_m', 'centre_hz', 'dial_hz')):
            raise ValueError('Every peer needs id, call, band_m, centre_hz and dial_hz.')
        if (not isinstance(peer['id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', peer['id'])
                or peer['id'] in seen):
            raise ValueError('Peer IDs must be nonempty and unique.')
        seen.add(peer['id'])
        if args.mode in ('vara', 'ardop'):
            widths = (500, 2300, 2750) if args.mode == 'vara' else (200, 500, 1000, 2000)
            if peer.get('bw_hz') not in widths:
                raise ValueError(f'{args.mode} peers need bw_hz in {widths}.')
        if (not isinstance(peer['call'], str) or not peer['call']
                or not isinstance(peer['band_m'], int)
                or not isinstance(peer['centre_hz'], int) or not isinstance(peer['dial_hz'], int)
                or peer['dial_hz'] != peer['centre_hz'] - DIAL_OFFSET_HZ):
            raise ValueError('Invalid peer identity or frequencies; dial must be centre minus 1500 Hz.')
    target = mode_root(args.mode) / 'peers.json'
    if target.exists() and json.loads(target.read_text()).get('peers') and not args.replace:
        raise ValueError('Roster already contains peers; use --replace to replace it.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + '\n')
    print(f'Imported {len(seen)} peers into {target}')
    return 0


if __name__ == '__main__':
    import sys
    try:
        raise SystemExit(run(sys.argv[1:]))
    except (ValueError, OSError) as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        raise SystemExit(2)
