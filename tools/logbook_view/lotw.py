# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Sign the logbook's completed contacts for LoTW and upload them, without TQSL.

    .venv/bin/python tools/logbook_view/lotw.py --p12 PATH --locations locations.json [--root PATH] [--since YYYY-MM-DD]
        [--limit N] [--out FILE.tq8] [--upload]

Without --upload this is a dry run: it writes the signed .tq8 and counts what is
in it. The .p12 is the callsign certificate TQSL exports; it and its password
are read at run time and kept only in memory.

The .tq8 is what tqsllib 2.8.6 with config.xml 11.34 writes: a gzipped GABBI file
of a TQSL_IDENT line, a tCERT, and per station location a tSTATION followed by
its tCONTACTs. Each tCONTACT carries SIGNDATA, the station location's sigspec
fields then the QSO's, concatenated and upper-cased, and its SHA-1 RSA
signature (src/location.cpp make_sign_data, tqsl_getGABBItCONTACTData).
"""

from __future__ import annotations

import argparse
import base64
import getpass
import gzip
import json
import re
import sys
import urllib.error
import urllib.request
import uuid
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import Encoding, pkcs12

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.face import COMPLETE, Logbook, adif_band, adif_logs, band_m, default_root, qso_adif, qso_time, read_adif

UPLOAD_URL = 'https://lotw.arrl.org/lotw/upload'
LEDGER = default_root() / 'lotw-uploaded.txt'
IDENT = 'hfmodem-logbook Lib: V2.8.6 Config: V11.34 AllowDupes: false'
SIGSPEC = 'SIGN_LOTW_V2.0'

# ARRL's private arc, tqsllib.cpp TQSL_OID_BASE: the callsign is a subject attribute, the rest extensions.
ARO_CALLSIGN, QSO_NOT_BEFORE, QSO_NOT_AFTER, DXCC_ENTITY = (
    x509.ObjectIdentifier(f'1.3.6.1.4.1.12348.1.{n}') for n in '1234')

# config.xml <modes>: what LoTW accepts.
LOTW_MODES = frozenset(
    'AM AMTOR ARDOP ATV C4FM CHIP CLOVER CONTESTI CW DATA DIGITALVOICE DOMINO DSTAR FAX FM FSK31 FSK441 '
    'FST4 FT4 FT8 GTOR HELL HFSK IMAGE ISCAT JT4 JT65 JT6M JT9 MFSK16 MFSK8 MINIRTTY MSK144 MT63 OLIVIA '
    'OPERA PACKET PACTOR PAX PHONE PSK10 PSK125 PSK2K PSK31 PSK63 PSK63F PSKAM PSKFEC31 Q15 Q65 QRA64 ROS '
    'RTTY RTTYM SSB SSTV T10 THOR THROB VOI WINMOR WSPR'.split())
# config.xml <adifmap> rows whose LoTW mode is not the ADIF name, for the modes this station works.
# A submode missing from both tables falls back to its mode, as tqslconvert.cpp does:
# PAC3 to PAC to PACTOR, JS8 and MFSK32 to MFSK to DATA; FT4, Q65 and ARDOP (a DATA-group mode) as themselves.
# config.xml has no DYNAMIC (VARA) row, so TQSL refuses it; DATA is the group LoTW matches it in.
ADIF_TO_LOTW = {'PAC': 'PACTOR', 'MFSK': 'DATA', 'JS8': 'DATA', 'PKT': 'PACKET', 'DYNAMIC': 'DATA'}


def load_locations(path: Path) -> dict:
    """Read explicit station locations; signing must never infer country or zones."""
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or not values:
        raise ValueError('Locations must be a nonempty JSON object keyed by station grid.')
    for grid, fields in values.items():
        if not re.fullmatch(r'[A-Ra-r]{2}[0-9]{2}(?:[A-Xa-x]{2})?', grid):
            raise ValueError(f'Invalid station grid: {grid}')
        if (not isinstance(fields, dict) or set(fields) - {'DXCC', 'GRIDSQUARE', 'ITUZ', 'CQZ', 'US_STATE'}
                or not all(isinstance(v, str) and v and v.isascii() for v in fields.values())
                or not fields.get('DXCC', '').isdigit()
                or fields.get('GRIDSQUARE', '').upper() != grid.upper()):
            raise ValueError(f'{grid}: supply DXCC and matching GRIDSQUARE as strings; check location fields.')
        for name, maximum in (('CQZ', 40), ('ITUZ', 90)):
            if name in fields and (not fields[name].isdigit() or not 1 <= int(fields[name]) <= maximum):
                raise ValueError(f'{grid}: invalid {name}.')
    if len({g.upper() for g in values}) != len(values):
        raise ValueError('Station grids must be unique, ignoring case.')
    return values


# The location's share of SIGNDATA: config.xml's LOTW V2.0 sigspec tSTATION fields that are set, in its order.
STATION_SIGNED = ('CQZ', 'GRIDSQUARE', 'ITUZ', 'US_STATE')


def field(name: str, value: str, kind: str = '') -> str:
    return f"<{name}:{len(value)}{':' + kind if kind else ''}>{value}"


def load_cert(p12_path: Path, password: str):
    key, cert, chain = pkcs12.load_key_and_certificates(Path(p12_path).read_bytes(), password.encode() or None)
    if key is None or cert is None:
        raise ValueError(f'{p12_path} holds no private key and certificate')
    return key, cert, chain


def cert_call(cert: x509.Certificate) -> str:
    return cert.subject.get_attributes_for_oid(ARO_CALLSIGN)[0].value


def cert_ext(cert: x509.Certificate, oid) -> str:
    # tqsllib reads these as the extension's raw octets: plain ASCII, no DER inside.
    return cert.extensions.get_extension_for_oid(oid).value.value.decode()


def qso_window(cert: x509.Certificate) -> tuple[date, date]:
    return (date.fromisoformat(cert_ext(cert, QSO_NOT_BEFORE)[:10]),
            date.fromisoformat(cert_ext(cert, QSO_NOT_AFTER)[:10]))


def lotw_mode(mode: str, submode: str = '') -> str:
    for name in ((submode or '').upper(), (mode or '').upper()):
        if name in ADIF_TO_LOTW:
            return ADIF_TO_LOTW[name]
        if name in LOTW_MODES:
            return name
    raise ValueError(f'no LoTW mode for ADIF {mode}' + (f'/{submode}' if submode else ''))


def location(grid: str, locations: dict) -> str:
    """The key in locations for an entry's MY_GRIDSQUARE; a four-character grid names the one location inside it."""
    grid = (grid or '').upper()
    found = [g for g in locations if g.upper() == grid] or [g for g in locations if len(grid) == 4 and g.upper().startswith(grid)]
    if len(found) != 1:
        raise ValueError(f'no single station location for MY_GRIDSQUARE {grid or "(none)"}')
    return found[0]


def contact(r: dict, locations: dict) -> dict:
    """One ADIF record as the fields a tCONTACT signs, normalised as tqslconvert.cpp does."""
    when = qso_time(r, 'QSO_DATE', 'TIME_ON')
    band = adif_band(r)
    if not (r.get('CALL') and when and band):
        raise ValueError('needs CALL, QSO_DATE, TIME_ON and a band')
    head, point, tail = r.get('FREQ', '').strip().partition('.')
    freq = head + point + tail.replace('.', '')  # fix_freq: only the first point counts
    if freq and (not float(freq) or band_m(float(freq) * 1e6) != band):
        freq = ''  # tqsllib drops a FREQ outside the BAND
    return {'CALL': r['CALL'].strip().upper(), 'BAND': f'{band}M', 'MODE': lotw_mode(r.get('MODE'), r.get('SUBMODE')),
            'FREQ': freq, 'QSO_DATE': f'{when:%Y-%m-%d}', 'QSO_TIME': f'{when:%H:%M:%SZ}',
            'location': location(r.get('MY_GRIDSQUARE'), locations)}


def ledger_key(c: dict) -> str:
    return '\t'.join((c['CALL'], c['BAND'], c['MODE'], f"{c['QSO_DATE']}T{c['QSO_TIME']}"))


def signdata(station: dict, c: dict) -> str:
    # The QSO's share is sigspec tCONTACT order: BAND BAND_RX CALL FREQ FREQ_RX MODE PROP_MODE QSO_DATE QSO_TIME SAT_NAME.
    parts = [station[k] for k in STATION_SIGNED if station.get(k)]
    parts += [c[k] for k in ('BAND', 'CALL', 'FREQ', 'MODE', 'QSO_DATE', 'QSO_TIME') if c[k]]
    return ''.join(p.strip() for p in parts).upper()


def t_cert(cert: x509.Certificate) -> str:
    body = cert.public_bytes(Encoding.PEM).decode().split('\n', 1)[1].split('-----END CERTIFICATE-----')[0]
    return f"<Rec_Type:5>tCERT\n{field('CERT_UID', '1')}\n{field('CERTIFICATE', body)}<eor>\n"


def t_station(uid: int, call: str, station: dict) -> str:
    lines = ['<Rec_Type:8>tSTATION', field('STATION_UID', str(uid)), field('CERT_UID', '1'), field('CALL', call)]
    lines += [field(k, station[k]) for k in ('DXCC', 'GRIDSQUARE', 'ITUZ', 'CQZ', 'US_STATE') if station.get(k)]
    return '\n'.join(lines) + '\n<eor>\n'


def t_contact(uid: int, c: dict, station: dict, key) -> str:
    data = signdata(station, c)
    sig = base64.b64encode(key.sign(data.encode(), padding.PKCS1v15(), hashes.SHA1())).decode()
    sig = ''.join(sig[i:i + 64] + '\n' for i in range(0, len(sig), 64))  # OpenSSL's base64 BIO lines
    lines = ['<Rec_Type:8>tCONTACT', field('STATION_UID', str(uid))]
    lines += [field(k, c[k]) for k in ('CALL', 'BAND', 'MODE', 'FREQ', 'QSO_DATE', 'QSO_TIME') if c[k]]
    return '\n'.join(lines) + '\n' + field(SIGSPEC, sig, '6') + field('SIGNDATA', data) + '\n<eor>\n'


def tq8(records: list[dict], key, cert: x509.Certificate, locations: dict) -> bytes:
    """records are ADIF fields, MY_GRIDSQUARE naming the station location. Only the signer's certificate
    goes in, as tqsllib writes it. Records are newline-separated, as tqsl.cpp's gzprintf("%s\\n") writes them."""
    contacts = [contact(r, locations) for r in records]
    first, last = qso_window(cert)
    if late := [c for c in contacts if not first <= date.fromisoformat(c['QSO_DATE']) <= last]:
        raise ValueError(f"{len(late)} QSOs fall outside the certificate's QSO dates {first} to {last}")
    used = dict.fromkeys(c['location'] for c in contacts)
    if abroad := [g for g in used if locations[g]['DXCC'] != cert_ext(cert, DXCC_ENTITY)]:
        raise ValueError(f"the certificate is for DXCC {cert_ext(cert, DXCC_ENTITY)}, not {', '.join(abroad)}'s")
    out = [field('TQSL_IDENT', IDENT) + '\n', t_cert(cert)]
    for uid, grid in enumerate(used, 1):
        out.append(t_station(uid, cert_call(cert), locations[grid]))
        out += (t_contact(uid, c, locations[grid], key) for c in contacts if c['location'] == grid)
    return gzip.compress(''.join(r + '\n' for r in out).encode('ascii'), 9)


def upload(tq8_bytes: bytes, filename: str = 'hfmodem.tq8') -> tuple[bool, str]:
    """POST as TQSL does (tqsl_prefs.h DEFAULT_UPL_*): one multipart file field, upfile. LoTW answers
    with an HTML page whose <!-- .UPL. accepted --> comment is the verdict and .UPLMESSAGE. the reason."""
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="upfile"; filename="{filename}"\r\n'
            f'Content-Type: application/octet-stream\r\n\r\n').encode() + tq8_bytes + f'\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(UPLOAD_URL, data=body, method='POST',
                                     headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            page = reply.read().decode('utf-8', errors='replace')
    except (urllib.error.URLError, TimeoutError) as e:
        return False, f'upload failed: {e}'
    status = re.search(r'<!-- \.UPL\.\s*([^-]+?)\s*-->', page)
    message = re.search(r'<!-- \.UPLMESSAGE\.\s*(.+?)\s*-->', page, re.S)
    ok = bool(status) and status[1].lower() == 'accepted'
    return ok, re.sub(r'\n +', '\n', message[1]) if message else (status[1] if status else 'LoTW gave no upload status')


def uploaded(ledger: Path) -> set[str]:
    return set(ledger.read_text().splitlines()) if ledger.is_file() else set()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--p12', type=Path, required=True, help='callsign certificate exported from TQSL')
    ap.add_argument('--locations', type=Path, required=True, help='JSON station locations keyed by grid, with DXCC and GRIDSQUARE')
    ap.add_argument('--root', type=Path, default=None, help='operator data directory')
    ap.add_argument('--since', type=date.fromisoformat, help='only QSOs on or after this UTC date')
    ap.add_argument('--limit', type=int, help='sign at most this many (the oldest new ones)')
    ap.add_argument('--out', type=Path, default=Path(f'lotw-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.tq8'))
    ap.add_argument('--upload', action='store_true', help='send to LoTW after confirmation')
    args = ap.parse_args(argv)

    locations = load_locations(args.locations)
    key, cert, _ = load_cert(args.p12, getpass.getpass(f'password for {args.p12.name}: '))
    book = Logbook((args.root or default_root()).resolve(), adif_logs())
    call, (first, last) = cert_call(cert), qso_window(cert)
    print(f'certificate {call}, serial {cert.serial_number}, QSOs {first} to {last}, '
          f'valid until {cert.not_valid_after_utc:%Y-%m-%d}')
    if call.upper() != book.station['call'].upper():
        raise SystemExit(f"certificate is for {call}, station.toml is {book.station['call']}")

    done = uploaded(LEDGER)
    entries = sorted((e for e in book.entries()['entries'] if COMPLETE & set(e['reached'])), key=lambda e: e['started_utc'])
    records, contacts = [], []
    for e in entries:
        if args.since and e['started_utc'][:10] < f'{args.since}':
            continue
        try:
            r = read_adif(qso_adif(e, book.station))[0] | {'MY_GRIDSQUARE': e['my_grid']}
            c = contact(r, locations)
        except KeyError as err:
            print(f"  skipped {e['call']} {e['started_utc']}: face.qso_adif has no ADIF mode for {err}")
            continue
        except ValueError as err:
            print(f"  skipped {e['call']} {e['started_utc']}: {err}")
            continue
        if not first <= date.fromisoformat(c['QSO_DATE']) <= last:
            print(f"  skipped {c['CALL']} {e['started_utc']}: outside the certificate's QSO dates")
        elif ledger_key(c) not in done:
            if (where := book.my_grid(e['started_utc'])) != c['location']:
                print(f"  note {c['CALL']} {e['started_utc']}: logged at {e['my_grid']}, the station was at {where}")
            records.append(r)
            contacts.append(c)
    records, contacts = records[:args.limit], contacts[:args.limit]
    if not contacts:
        return print('nothing new to sign')

    args.out.write_bytes(tq8(records, key, cert, locations))
    print(f'{len(contacts)} QSOs signed to {args.out}')
    for (grid, mode, band), n in sorted(Counter((c['location'], c['MODE'], c['BAND']) for c in contacts).items()):
        print(f'  {grid}  {mode:<8} {band:>4}  {n}')
    if not args.upload:
        return
    if input(f'type UPLOAD to send {len(contacts)} QSOs to LoTW: ') != 'UPLOAD':
        return print('not sent')
    ok, message = upload(args.out.read_bytes(), args.out.name)
    print(('accepted: ' if ok else 'REJECTED: ') + message)
    if not ok:
        sys.exit(1)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a') as f:
        f.writelines(ledger_key(c) + '\n' for c in contacts)


if __name__ == '__main__':
    main()
