# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Opt-in VarAC BBS directory and single-file sender."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import os
from pathlib import Path

from .bbs import ExportError
from .bbs_download import validate_offer, DOWNLOAD_SECONDS
from .filetransfer import parse_offer


def configured_directory(override=None, *, station_path=None):
    """CLI wins; TOML relative paths are relative to the station file."""
    if override is not None:
        return str(Path(override).expanduser().absolute()) if override else None
    from ..core import config
    station_path = station_path or os.environ.get(config.STATION_ENV)
    if not station_path:
        return None
    cfg = config.load(station_path)
    protocol = cfg.protocols.get('vara')
    value = protocol.bbs_directory if protocol is not None else ''
    if not isinstance(value, str):
        raise ValueError('[protocols.vara] bbs_directory must be a path string')
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else Path(station_path).absolute().parent / path)


class BBSServer:
    def __init__(self, session, catalog=None):
        self.session, self.catalog = session, catalog
        self.application_lease = None
        self._future = None
        self._executor = None
        self.reset()
        self.advertise()

    @property
    def active(self):
        return self.phase is not None

    def reset(self, outcome='aborted'):
        lease, self.application_lease = self.application_lease, None
        self.phase = None
        self.payload = None
        self.generation = self.catalog_generation = self.dial = self.deadline = None
        self.name = None
        if lease and getattr(self.session, 'bbs', None):
            self.session.bbs.transfer_done(lease, outcome)

    def advertise(self):
        s = self.session
        services = (s.flow.services or ('V' if self.catalog is not None else '')).replace('B', '')
        if self.catalog is not None and self.catalog.usable:
            services += 'B'
        s.flow.services = services or None

    def valid_link(self):
        s, c = self.session, self.session.client
        return (c.connected and not c.closed and not s._closing
                and self.generation == c.link_generation and self.dial == s.flow.dial
                and self.catalog is not None and self.catalog_generation == self.catalog.generation)

    def tick(self):
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            if self.phase == 'preparing' and self.valid_link():
                try:
                    kind, result = future.result()
                    if kind == 'directory':
                        self._directory(result)
                    else:
                        lease = self.application_lease
                        self.offer_snapshot(result, lease=lease)
                except (ValueError, OSError):
                    refusal = '<BLJ>' if self.name == '<BLR>' else '<BGJ>'
                    self.reset('failed')
                    self.session.client.send_text(refusal)
        if not self.active:
            return
        if not self.valid_link():
            self.reset()
        elif self.session.client.clock() >= self.deadline:
            self.session.say('BBS send timed out; receipt unconfirmed')
            leased = bool(self.application_lease)
            self.reset('failed')
            if not leased:
                self.session.files.stop('BBS send timeout')
        elif self.phase == 'directory':
            c = self.session.client
            after = c._sent_after
            if (not (c.data_writer and c.data_writer.pending) and after is not None
                    and c.buffer_sequence > after and c.buffer == 0 and not c.ptt and not c.pending):
                self.reset('complete')

    def _bind(self, generation):
        s, c = self.session, self.session.client
        self.generation, self.dial = c.link_generation, s.flow.dial
        self.catalog_generation = generation

    def _directory(self, entries):
        rows = []
        for entry in entries:
            parse_offer(f'<SF:{entry.name}.VarAC_gzip|0|{"0" * 32}>')
            date = datetime.fromtimestamp(entry.mtime_ns / 1e9, timezone.utc).date()
            rows.append(f'{entry.name}|{date}|{entry.size}')
        message = '<BL:' + '\r\n'.join(rows) + '\r\n>'
        if not rows or len(message.encode()) > 65536:
            raise ExportError('Empty or oversized directory')
        self.session.client.send_text(message)
        if self.application_lease:
            self.phase, self.deadline = 'directory', self.session.client.clock() + 60
        else:
            self.phase = None

    def offer_snapshot(self, snapshot, lease=None):
        """Server-side offer for both native BG and shared lobby GET.

        The caller supplies an access-checked immutable catalog snapshot; this
        never requests a file from the peer. SFOK remains the receipt boundary.
        """
        if self.catalog is None or snapshot.generation != self.catalog.generation:
            raise ExportError('Catalog changed before file offer')
        name = snapshot.entry.name
        compressed = gzip.compress(snapshot.data, mtime=0)
        offer = parse_offer(f'<SF:{name}.VarAC_gzip|{len(compressed)}|'
                            f'{hashlib.md5(compressed).hexdigest().upper()}>')
        validate_offer(offer)
        self.application_lease = lease
        self.payload = '<SFB>' + base64.b64encode(compressed).decode('ascii')
        self.name = name
        self._bind(snapshot.generation)
        self.deadline, self.phase = self.session.client.clock() + 60, 'offer'
        self.session.client.send_text(offer.raw)

    def _prepare(self, text, peer):
        if text == '<BLR>':
            return 'directory', self.catalog.list_files(peer)
        if not text.endswith('>'):
            raise ExportError('Invalid file request')
        return 'file', self.catalog.read_file(text[4:-1], peer)

    def message(self, text):
        s, c = self.session, self.session.client
        if text not in ('<BLR>', '<SFRD>', '<SFOK>', '<SFFA>', '<SFAB>') and not text.startswith('<BG:'):
            return False
        if not c.connected or c.closed or s._closing:
            self.reset()
            return True
        if text in ('<SFRD>', '<SFOK>', '<SFFA>', '<SFAB>'):
            if not self.active or not self.valid_link():
                self.reset()
                return False
            if text in ('<SFFA>', '<SFAB>'):
                self.reset('refused')
                if not getattr(s, 'bbs', None) or not s.bbs.active:
                    s.files.stop('peer refused BBS file')
            elif text == '<SFRD>' and self.phase == 'offer':
                self.phase = 'receipt'
                self.deadline = c.clock() + DOWNLOAD_SECONDS
                c.send_text(self.payload)
            elif text == '<SFOK>' and self.phase == 'receipt':
                s.say(f'BBS file receipt confirmed by {s.peer}: {self.name}')
                s.emit('bbs_served', peer=s.peer, filename=self.name)
                self.reset('complete')
            return True
        from .varac import CALLING_HZ, State
        refusal = '<BLJ>' if text == '<BLR>' else '<BGJ>'
        if (self.catalog is None or self.active or s.files.active or s.steps is not None
                or s._opening or s.email.waiting or s.mail_ack.busy or s.parking.active
                or s.flow.state is not State.CONNECTED or s.flow.dial in CALLING_HZ.values()):
            c.send_text(refusal)
            return True
        entrance = getattr(s, 'bbs', None)
        if self._future is not None:
            c.send_text(refusal)
            return True
        lease = entrance.native_file_lease() if entrance else False
        if lease is None:
            c.send_text(refusal)
            return True
        self.application_lease = lease or None
        try:
            if entrance and entrance.core:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='bbs-export')
                self._bind(self.catalog.generation)
                self.name, self.phase, self.deadline = text, 'preparing', c.clock() + 60
                self._future = self._executor.submit(self._prepare, text, s.peer)
            else:
                kind, result = self._prepare(text, s.peer)
                if kind == 'directory':
                    self._directory(result)
                else:
                    self.offer_snapshot(result)
        except (ValueError, OSError) as error:
            self.reset()
            s.say(f'BBS request refused: {error}')
            c.send_text(refusal)
        return True
