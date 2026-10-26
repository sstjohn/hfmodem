# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded evidence capture, not an implementation of unknown file packets.

The caller owns the sole transport consumer and all protocol replies. This
module never decompresses data, emits acknowledgements, or claims a download.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import time
import unicodedata

MAX_CAPTURE_BYTES = 65536
MAX_CAPTURE_SECONDS = 120.0


@dataclass(frozen=True)
class FileOffer:
    raw: str
    wire_name: str
    declared_integer: int
    digest_text: str


def parse_offer(text: str) -> FileOffer:
    """Parse the publicly observed header without assigning checksum semantics."""
    if len(text.encode('utf-8')) > 1024:
        raise ValueError('File offer exceeds header limit')
    match = re.fullmatch(r'<SF:([^|]+)\|([0-9]{1,10})\|([0-9A-Fa-f]{32})>', text)
    if match is None:
        raise ValueError('Invalid file offer header')
    name, number, digest = match.groups()
    if (len(name.encode('utf-8')) > 255 or name != name.strip()
            or name in ('.', '..') or name.endswith('.')
            or any(c in '/\\:<>|\x00' or unicodedata.category(c).startswith('C')
                   for c in name)):
        raise ValueError('Unsafe file offer name')
    value = int(number)
    if value > 2**31 - 1:
        raise ValueError('File offer integer exceeds limit')
    return FileOffer(text, name, value, digest)


@dataclass(frozen=True)
class CaptureResult:
    directory: Path
    reason: str
    received_bytes: int
    sent_bytes: int
    truncated: bool
    # Even an orderly capture does not prove a complete/valid file transfer.
    validated_download: bool = False


class CaptureStopped(RuntimeError):
    def __init__(self, result: CaptureResult):
        self.result = result
        super().__init__(f'File evidence capture stopped: {result.reason}')


class RawTransferCapture:
    """Store ordered raw chunks with bounded total bytes and a hard deadline.

    append('rx'|'tx', bytes) records exactly the bytes observed/accepted by the
    transport. On limits it retains the fitting prefix, closes, then raises
    CaptureStopped. Call check_deadline even when no transport data arrives.
    The caller must end the experiment; this object performs no link actions.
    """
    max_seconds = MAX_CAPTURE_SECONDS

    def __init__(self, root, *, peer: str, filename: str, link_generation: int,
                 clock=time.monotonic, byte_limit=MAX_CAPTURE_BYTES,
                 time_limit=MAX_CAPTURE_SECONDS):
        if type(byte_limit) is not int or not 0 < byte_limit <= MAX_CAPTURE_BYTES:
            raise ValueError('Capture byte limit must be within 1..65536')
        if not 0 < time_limit <= self.max_seconds:
            raise ValueError(f'Capture deadline must be within 0..{self.max_seconds:g} seconds')
        self.clock, self.started = clock, clock()
        self.deadline = self.started + time_limit
        self.byte_limit = byte_limit
        self.offsets = {'rx': 0, 'tx': 0}
        self.sequence = 0
        self.result = None
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='bbs-capture-', dir=root))
        self._raw = self._events = None
        try:
            self._raw = (self.directory / 'transport.bin').open('xb')
            self._events = (self.directory / 'chunks.jsonl').open('x', encoding='utf-8')
            self._write_event({'event': 'start', 'utc': datetime.now(timezone.utc).isoformat(),
                               'peer': peer, 'filename': filename,
                               'link_generation': link_generation,
                               'byte_limit': byte_limit, 'time_limit': time_limit})
        except BaseException:
            for stream in (self._raw, self._events):
                if stream is not None:
                    stream.close()
            raise

    def _write_event(self, value):
        self._events.write(json.dumps(value, ensure_ascii=True) + '\n')
        self._events.flush()

    def check_deadline(self):
        if self.result is not None:
            raise CaptureStopped(self.result)
        if self.clock() >= self.deadline:
            raise CaptureStopped(self.finish('deadline', truncated=True))

    def append(self, direction: str, data: bytes):
        if direction not in self.offsets:
            raise ValueError('Capture direction must be rx or tx')
        self.check_deadline()
        if not isinstance(data, bytes):
            raise TypeError('Raw capture requires bytes')
        if not data:
            return
        offset = sum(self.offsets.values())
        kept = data[:self.byte_limit - offset]
        event = {'event': 'chunk', 'sequence': self.sequence, 'direction': direction,
                 'elapsed_s': self.clock() - self.started, 'offset': offset,
                 'direction_offset': self.offsets[direction], 'length': len(kept),
                 'observed_length': len(data)}
        try:
            count = self._raw.write(kept)
            self._raw.flush()
            if count != len(kept):
                raise OSError('Short raw capture write')
            self.offsets[direction] += len(kept)
            self.sequence += 1
            self._write_event(event)
        except OSError as original:
            try:
                self.finish('write error', truncated=True)
            except OSError as cleanup:
                raise original from cleanup
            raise
        if sum(self.offsets.values()) >= self.byte_limit:
            raise CaptureStopped(self.finish('byte limit', truncated=True))
        return event

    def finish(self, reason='operator stopped', *, truncated=False):
        if self.result is not None:
            return self.result
        self.result = CaptureResult(self.directory, reason, self.offsets['rx'],
                                    self.offsets['tx'], truncated)
        try:
            self._write_event({'event': 'end', 'reason': reason,
                               'elapsed_s': self.clock() - self.started,
                               'received_bytes': self.offsets['rx'],
                               'sent_bytes': self.offsets['tx'],
                               'truncated': truncated, 'validated_download': False})
        finally:
            try:
                self._raw.close()
            finally:
                self._events.close()
        return self.result
