# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Which arm delivered each message the station sent over Winlink, as the recipient's inbox shows it.

    .venv/bin/python tools/logbook_view/delivered.py --imap ACCOUNT [MBOX ...] [--root PATH]

--imap reads the mailbox itself: every message from the station's Winlink
address in Gmail's All Mail, headers only, with the app password in
~/.imap_password. An MBOX, such as a Gmail Takeout export, is read as well. The CMS carries a message's MID as the local part of its Message-ID and
the B2F Date as its Date. Each MID is looked for among the arms that prepared or
proposed it. Of those, the arm whose proposal the gateway accepted last before
the mail arrived delivered it, since a CMS turns away a MID it already holds.
The matches go to the optional operator delivery-evidence file: MID, times, gateway and arm,
nothing from the messages themselves.
"""
from __future__ import annotations

import argparse
import email
import email.utils
import imaplib
import json
import mailbox
import re
from datetime import datetime
from pathlib import Path

from legacy import Converter

from desk.face import Logbook, default_root, stamp

MID = re.compile(r'<(\w{12})@winlink\.org>', re.I)
# A B2F proposal; a PACTOR TX payload's bytes repr puts a literal \r right before it.
PROPOSED = re.compile(r'FC EM (\w{12})\b')
# The mail client's closing line for each message it offered, once the gateway answered.
ANSWERED = re.compile(r'mail: (?:prepared|transmitted) (\w{12}) — accepted FS [^\n]*')
# How far an arm's own record carries a MID, weakest first.
HOW = ('prepared', 'proposed', 'accepted', 'accepted, far end FF received')


def headers(account: str, address: str):
    """Only the headers `sent` reads, of every message from `address`; PEEK leaves them unread."""
    with imaplib.IMAP4_SSL('imap.gmail.com') as imap:
        imap.login(account, (Path.home() / '.imap_password').read_text().strip())
        imap.select('"[Gmail]/All Mail"', readonly=True)
        _, found = imap.uid('SEARCH', None, 'FROM', f'"{address}"')
        for uid in found[0].split():
            _, data = imap.uid('FETCH', uid, '(BODY.PEEK[HEADER.FIELDS (FROM MESSAGE-ID DATE RECEIVED)])')
            yield email.message_from_bytes(data[0][1])


def sent(messages, address: str) -> list[dict]:
    """Every message from `address`: its MID, the B2F Date the CMS kept, and when the inbox took it in."""
    out = []
    for m in messages:
        if email.utils.parseaddr(m['From'] or '')[1].lower() != address.lower():
            continue
        if not (mid := MID.fullmatch((m['Message-ID'] or '').strip())):
            continue
        received = m.get_all('Received') or []
        arrived = received and stamp(email.utils.parsedate_to_datetime(received[0].rpartition(';')[2].strip()))
        out.append({'mid': mid[1].upper(), 'sent_utc': stamp(email.utils.parsedate_to_datetime(m['Date'])),
                    'delivered_utc': arrived or None})
    return sorted(out, key=lambda e: e['sent_utc'])


def carried(root: Path) -> dict[str, list[dict]]:
    """For each MID, every arm that prepared or proposed it and how far it got: the arm's own
    delivery, its transcript, a converted arm's source transcript, and onair.sh's session log.
    An attempt legacy.py has yet to write counts under the name it will be written with."""
    working = root / 'working'
    from hfmodem.core.capture_identity import associated_transcripts, metadata_record
    onair = {}
    for path in working.glob('onair-*/**/*.log'):
        onair.setdefault(path.stem, []).append(path)
    arms = [(prov, metadata_record(prov)) for prov in working.glob('*/arms/*/provenance.json')]
    arms += [(Path(r['directory']) / 'provenance.json', r) for r in Converter(root).pending().values()]
    found: dict[str, list[dict]] = {}
    for prov, p in arms:
        how = {}
        if mid := (p.get('delivery') or {}).get('mid'):
            how[mid] = 0
        for log in associated_transcripts(root, prov, p, onair.get(p['arm'], [])):
            if not (log and log.is_file()):
                continue
            text = log.read_text(errors='replace')
            for m in PROPOSED.finditer(text):
                how[m[1]] = max(how.get(m[1], 0), 1)
            for m in ANSWERED.finditer(text):
                how[m[1]] = max(how.get(m[1], 0), 3 if 'far end FF received' in m[0] else 2)
        for mid, rank in how.items():
            found.setdefault(mid.upper(), []).append(
                {'arm': p['arm'], 'gateway': p['call'], 'started_utc': stamp(datetime.fromisoformat(p['started_utc'])),
                 'how': rank})
    return {mid: sorted(arms, key=lambda a: a['started_utc']) for mid, arms in found.items()}


def deliverer(message: dict, arms: list[dict]) -> dict | None:
    before = [a for a in arms if not message['delivered_utc'] or a['started_utc'] <= message['delivered_utc']]
    return max(before, key=lambda a: (a['how'] >= 2, a['started_utc']), default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('mbox', type=Path, nargs='*', help='Gmail Takeout mboxes; a MID in several is counted once')
    ap.add_argument('--imap', metavar='ACCOUNT', help="read this Gmail account's mail over IMAP")
    ap.add_argument('--root', type=Path, default=None, help='repository holding working/ and station.toml')
    args = ap.parse_args()
    if not (args.mbox or args.imap):
        ap.error('give --imap ACCOUNT, an MBOX, or both')
    refresh((args.root or default_root()).resolve(), args.mbox, args.imap)


def refresh(root: Path, mboxes=(), imap: str | None = None, say=print):
    """Match every message found to the arm that delivered it, and write the matches."""
    call = Logbook(root).station['call']
    if not call:
        raise ValueError('station.mycall is required for inbox confirmation')
    address = f'{call}@winlink.org'
    boxes = [mailbox.mbox(box) for box in mboxes] + ([headers(imap, address)] if imap else [])
    messages = sorted({m['mid']: m for box in boxes for m in sent(box, address)}.values(), key=lambda m: m['sent_utc'])
    found = carried(root)
    first = min((a['started_utc'] for arms in found.values() for a in arms), default=None)
    delivered = []
    for msg in messages:
        arms = found.get(msg['mid'], [])
        if not (arm := deliverer(msg, arms)):
            why = 'sent before the first arm' if first and msg['sent_utc'] < first else 'no arm prepared or proposed it'
            say(f"{msg['mid']}  {msg['sent_utc']}  unmatched: {why}")
            continue
        delivered.append({'mid': msg['mid'], 'sent_utc': msg['sent_utc'], 'delivered_utc': msg['delivered_utc'],
                          'gateway': arm['gateway'], 'arm': arm['arm']})
        others = ', '.join(f"{a['arm']} ({HOW[a['how']]})" for a in arms if a is not arm)
        say(f"{msg['mid']}  {msg['sent_utc']}  {arm['arm']}  {HOW[arm['how']]}" + (f'; also in {others}' if others else ''))
    out = root / 'working' / 'winlink-delivered.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix('.part')
    part.write_text(json.dumps(delivered, indent=1) + '\n')
    part.replace(out)  # the viewer may be reading it
    say(f'{len(delivered)} of {len(messages)} messages matched to an arm; written to {out}')


if __name__ == '__main__':
    main()
