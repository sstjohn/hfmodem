# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Parked-mail notices and operator-initiated VarAC collection.

VarAC manual v16.1.8 p112: the relay forwards mail when the destination
connects; there is no separate download request. Preserve notices until the
operator dismisses them: one received VMail does not prove the relay is empty.
"""
import json
import os
from pathlib import Path
import tempfile


class MailCollection:
    def __init__(self, session):
        self.session = session
        self.path = session.mailbox.root / 'relay-notices.json'

    def _load(self):
        if not self.path.exists():
            return []
        rows = json.loads(self.path.read_text())
        if not isinstance(rows, list) or any(
                not isinstance(row, dict) or
                not all(isinstance(row.get(key), str) for key in ('source', 'destination', 'utc')) or
                type(row.get('dial_hz')) is not int or
                type(row.get('urgent', False)) is not bool or
                type(row.get('received', 0)) is not int for row in rows):
            raise ValueError('invalid relay-notices.json; existing file preserved')
        return rows

    def _save(self, rows):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent,
                                         prefix='.relay-notices-', delete=False) as out:
            temporary = Path(out.name)
            try:
                json.dump(rows, out, indent=1)
                out.write('\n')
                out.flush()
                os.fsync(out.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def notice(self, message, *, dial):
        from .varac import canonical_call, utc
        s = self.session
        if (not s.parking.store.settings['relay'] or message.command != 1
                or message.info not in (b'<SMP>', b'<SMU>')):
            return
        try:
            owner, source, destination = map(canonical_call,
                                             (s.client.mycall, message.source, message.destination))
        except ValueError:
            return
        if destination != owner or source == owner:
            return
        if not s.parking.store.available:
            s.say('VMail parking store unavailable; collection notices remain enabled')
        try:
            rows = self._load()
            rows = [row for row in rows if (row['source'], row['destination']) !=
                    (source, destination)]
            rows.append(dict(source=source, destination=destination,
                             utc=utc(), dial_hz=dial, received=0, urgent=message.info == b'<SMU>'))
            self._save(rows)
        except (OSError, ValueError) as error:
            s.say(f'Parked-mail notice could not be saved: {error}')
            return
        s.say(f'{"Urgent mail" if rows[-1]["urgent"] else "Mail"} waiting at {message.source}; '
              f'/vmail collect {message.source} connects to collect it. '
              '/vmail pending lists saved notices.')
        s.emit('vmail_notice', **rows[-1])

    def received(self):
        from .varac import canonical_call
        s = self.session
        try:
            owner, peer = map(canonical_call, (s.client.mycall, s.peer))
            rows = self._load()
            for row in rows:
                if row['destination'] == owner and row['source'] == peer:
                    row['received'] = row.get('received', 0) + 1
                    self._save(rows)
                    return
        except (OSError, ValueError) as error:
            s.say(f'VMail saved, but relay notice could not be updated: {error}')

    def command(self, words):
        from .varac import CALLING_HZ, HostError, base_call, canonical_call, slot_frequency
        s = self.session
        try:
            owner = canonical_call(s.client.mycall)
            rows = self._load()
            mine = [row for row in rows if row['destination'] == owner]
            if words == ['pending']:
                for row in mine:
                    s.say(f"{row['source']}: mail notice on {row['dial_hz'] / 1000:.3f} kHz, "
                          f"{row['utc']}; {'urgent; ' if row.get('urgent') else ''}"
                          f"{row.get('received', 0)} received since notice; "
                          f"/vmail collect {row['source']}")
                s.say('Notices remain until /vmail dismiss CALL; the relay may have more mail.'
                      if mine else 'No parked-mail notices saved for this callsign')
                return
            if len(words) != 2 or words[0] not in ('collect', 'dismiss'):
                raise ValueError('usage: /vmail pending | /vmail collect CALL | /vmail dismiss CALL')
            peer = canonical_call(words[1])
            notice = next((row for row in mine if row['source'] == peer), None)
            if words[0] == 'dismiss':
                if notice is None:
                    raise ValueError(f'No parked-mail notice from {peer}')
                self._save([row for row in rows if row is not notice])
                s.say(f'Parked-mail notice from {peer} dismissed; saved VMails retained')
                return
        except OSError as error:
            raise HostError(f'Cannot access parked-mail notices: {error}') from error

        if peer == owner:
            raise ValueError('Cannot collect mail from your own callsign')
        if s.client.closed:
            raise HostError('Host connection is closed')
        # Reuse normal connection, QSY readiness and cancellation. A notice is
        # an RF claim, never permission to tune or transmit on its own.
        target = notice['dial_hz'] if notice else s.calling_hz
        channels = {s.calling_hz: None, **{slot_frequency(s.flow.band, slot, s.flow.codec): slot
                                         for slot in s.flow.codec.slots}}
        if target not in channels:
            band = next((band for band, cf in CALLING_HZ.items()
                         if abs(cf - target) <= 3750), None)
            action = f'/band {band} first' if band is not None else 'use /call on a verified channel'
            raise ValueError(f'{peer} notice is on {target / 1000:.3f} kHz; {action}')
        if s.line_state() is not False:
            raise HostError('Mail collection requires a verified-unkeyed session')
        slot = channels[target]
        known = next((entry for entry in reversed(s.heard)
                      if entry.call == base_call(peer) and entry.band in (None, s.flow.band)), None)
        s._leave('call', slot, known.bw if known else s.bandwidth)
        s.peer = peer
        s.say(f'Collecting mail from {peer}; the peer forwards it after connection and QSY. '
              'Each saved VMail is acknowledged; /vmail lists it and /bye ends the contact.')
        s._run(self._connect(peer, slot, target))

    def _connect(self, peer, slot, target):
        s = self.session
        # Normal /call without a slot uses the current dial. Collection instead
        # returns to the saved notice channel, including after passive following.
        if slot is None and s.flow.dial != target:
            s.flow.tune(target)
        yield from s._connecting(peer, slot)
