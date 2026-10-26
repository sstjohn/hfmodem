# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Current public Winlink channels, normalized into the console roster."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from . import operator

BANDS = {160: (1_800_000, 2_000_000), 80: (3_500_000, 4_000_000),
         60: (5_250_000, 5_450_000), 40: (7_000_000, 7_300_000),
         30: (10_100_000, 10_150_000), 20: (14_000_000, 14_350_000),
         17: (18_068_000, 18_168_000), 15: (21_000_000, 21_450_000),
         12: (24_890_000, 24_990_000), 10: (28_000_000, 29_700_000)}


def band_for(centre):
    return next((band for band, (lo, hi) in BANDS.items() if lo <= centre <= hi), None)


def normalize(text, mode):
    from hfmodem.core.band import DIAL_OFFSET_HZ
    rows = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    required = {'Callsign', 'Frequency', 'Mode'}
    if not required <= set(rows.fieldnames or []):
        raise ValueError('The Winlink listing is not a channel CSV (missing Callsign, Frequency or Mode).')
    peers, seen = [], set()
    for row in rows:
        advertised = row['Mode'].strip()
        if not advertised.upper().startswith({'pactor': 'PACTOR', 'vara': 'VARA', 'ardop': 'ARDOP'}[mode]):
            continue
        if mode == 'vara' and 'FM' in advertised.upper():
            continue
        call = row['Callsign'].strip().upper()
        if not re.fullmatch(r'[A-Z0-9]+(?:-[A-Z0-9]+)?', call):
            raise ValueError(f'Invalid gateway callsign in listing: {call!r}')
        match = re.fullmatch(r'([0-9.,]+)\s*(kHz|MHz|Hz)', row['Frequency'].strip(), re.I)
        if not match:
            raise ValueError(f'Invalid gateway frequency: {row["Frequency"]!r}')
        value = float(match[1].replace(',', ''))
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Gateway frequency must be positive and finite.')
        centre = round(value * {'hz': 1, 'khz': 1000, 'mhz': 1_000_000}[match[2].lower()])
        band = band_for(centre)
        if band is None:
            continue
        width = None
        if mode != 'pactor':
            match = re.search(r'\b(2750|2300|2000|1000|500|200)\b', advertised)
            if match:
                width = int(match[1])
            elif mode == 'vara' and advertised.upper() in ('VARA', 'VARA HF'):
                # The public Winlink CSV uses plain VARA for its standard HF service.
                width = 2750
            else:
                raise ValueError(f'Missing advertised bandwidth: {advertised!r}')
        ident = f'{call}-{band}m-{centre}' + (f'-{width}' if width else '')
        if ident in seen:
            continue
        seen.add(ident)
        peer = dict(id=ident, call=call, band_m=band, centre_hz=centre,
                    dial_hz=centre - DIAL_OFFSET_HZ, advertised_mode=advertised,
                    hours=(row.get('Hours') or '').strip(), grid=(row.get('GridSquare') or '').strip(),
                    qth=(row.get('QTH') or '').strip())
        if width:
            peer['bw_hz'] = width
        peers.append(peer)
    if not peers:
        raise ValueError(f'The listing contains no usable HF {mode.upper()} channels; saved peers were kept.')
    return sorted(peers, key=lambda p: (p['centre_hz'], p['call'], p.get('bw_hz', 0)))


def refresh(mode, *, csv_path=None, fetch=None):
    from hfcapture import GATEWAYS_URL, fetch_gateways
    from hfmodem.core import config
    target = operator.mode_root(mode) / 'peers.json'
    text = (Path(csv_path).read_text(encoding='utf-8-sig') if csv_path else
            (fetch or fetch_gateways)({'pactor': 'Pactor', 'vara': 'VARA', 'ardop': 'ARDOP'}[mode]))
    peers = normalize(text, mode)
    # Keep the validated raw listing beside the roster so the modem's channel
    # guard and handshake decoder use it without another network round trip.
    cache = operator.data_root() / 'cache' / f'winlink-{mode}-gateways.csv'
    cache.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.gateways-', dir=cache.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, cache)
    finally:
        Path(temporary).unlink(missing_ok=True)
    previous = json.loads(target.read_text()) if target.is_file() else {}
    old = {p['id']: p for p in previous.get('peers', [])}
    for peer in peers:
        for field in ('last_result', 'offset_hz'):
            if field in old.get(peer['id'], {}):
                peer[field] = old[peer['id']][field]
    station = config.load(operator.station_file()).station if operator.station_file().is_file() else None
    result = dict(previous, peers=peers, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  source=str(Path(csv_path).resolve()) if csv_path else GATEWAYS_URL,
                  local_grid=station.grid if station else '')
    if station:
        result['station'] = f'{station.mycall} {station.grid}'
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.peers-', dir=target.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    print(f'{len(peers)} {mode.upper()} gateway channels loaded.')
    return peers
