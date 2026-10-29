# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""UTF-8 keyboard contacts over the VARA host API, with an injected dial tuner.

Wire forms follow the P1 decoded messages; public tag names remain visible.
CQ suffix IDs are the production mapping of P1 suffix/physical-slot pairs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import re
import select
import socket
import tempfile
import time
import uuid
from typing import Callable

from ..ax25 import kiss
from .beacon_hop import BeaconHop
from . import mail_storage
from ..core.config import clamp_power


DD = 'https://k0rv.org/wp-content/uploads/2024/02/VarAC_slide_show.pdf'
REL = 'https://www.varac-hamradio.com/varac-releases'


@dataclass(frozen=True)
class Tag:
    wire: str
    source: str
    emit: bool = False


TAGS = {
    'NAME': Tag('<NAME>', DD),
    'QTH': Tag('<QTH>', DD),
    'LOC': Tag('<LOC>', DD),
    'RIG': Tag('<RIG>', DD),
    'ANT': Tag('<ANT>', DD),
    'CALL': Tag('<CALL>', DD),
    'INFO': Tag('<INFO>', DD),
    'VER': Tag('<VER>', DD),
    'LHR': Tag('<LHR>', DD),
    'LHE': Tag('<LHE>', DD),
    'FSR': Tag('<FSR>', DD),
    'QSYU': Tag('<QSYU>', DD),
    'QSYD': Tag('<QSYD>', DD),
    'QSF': Tag('<QSF>', DD, False),
    '/QSYF': Tag('</QSYF>', DD, False),
    'QJO': Tag('<QJO>', DD),
    #: Named on deck slide 19 and sent beside <AWQ> (REL V8.6.2); every measured bare tag is <NAME>.
    'AWAY': Tag('<AWAY>', DD + '; REL V8.6.2', True),
    'SND': Tag('<SND>', DD),
    'DISC': Tag('<DISC>', DD),
    'SNRR': Tag('<SNRR>', REL + ' (V4.1.6)'),
    'S:A': Tag('<S:A>', REL + ' (V13.2.7; alternatives transcribed as S:A/E/N)', False),
    'S:E': Tag('<S:E>', REL + ' (V13.2.7; alternatives transcribed as S:A/E/N)', False),
    'S:N': Tag('<S:N>', REL + ' (V13.2.7; alternatives transcribed as S:A/E/N)', False),
}


MEASURED = 'P1 transcripts/messages.json'
VALUE_TAGS = ('NAME', 'QTH', 'LOC', 'EM', 'FC', 'SBJ', 'MSG', 'Q', 'QF', 'S')
BARE_TAGS = ('AWQ', 'DISC', 'ER', 'INFO', 'QSYR', 'SM', 'SMR', 'TL', 'VSI', 'VW',
             'SMF', 'SMFP', 'VRPJ')
TAGS.update({name: Tag(f'<{name}:value>', MEASURED, True) for name in VALUE_TAGS})
TAGS.update({name: Tag(f'<{name}>', MEASURED, True) for name in BARE_TAGS})
TAGS['R'] = Tag('<R±NN>', MEASURED, True)
TAGS['&IT&'] = Tag('&IT&', MEASURED, True)
MANUAL_16 = 'https://download-files.wixmp.com/ugd/288b5d_124e2927d3324571a2016b69e780d159.pdf'
TAGS['QF'] = Tag('<QF:value>', MANUAL_16 + ' p126; broadcast frequency notification', True)
TAGS['QSYF'] = Tag('<QSYF>', MANUAL_16 + ' p126; v16.1.8 connected parser, 12 digits', True)
TAGS.update({name: Tag(f'<{name}>', MANUAL_16 + ' pp113,125', name in BARE_TAGS)
             for name in ('E', 'EA', 'EJ', 'SMF', 'SMFP')})
TAGS['LHR'] = Tag('<LHR>', MANUAL_16 + ' p125 request data', True)
TAGS['LHJ'] = Tag('<LHJ>', MANUAL_16 + ' p125 request data')
TAGS['BLR'] = Tag('<BLR>', MANUAL_16 + ' p126; p65 exchange', True)
TAGS['VRP'] = Tag('<VRP>', MANUAL_16 + ' p125; peeking request')
TAGS['VRPJ'] = Tag('<VRPJ>', MANUAL_16 + ' p125; peeking rejection', True)


@dataclass(frozen=True)
class Text:
    raw: str
    tag: str | None = None
    value: str | None = None


class TextParser:
    """Incremental, lossless text/token split; unknown markup stays visible."""

    def __init__(self):
        self.pending = ''

    def feed(self, text: str, *, final=False) -> list[Text]:
        self.pending += text
        result = []
        while self.pending:
            starts = [i for i in (self.pending.find('<'), self.pending.find('&')) if i >= 0]
            start = min(starts) if starts else len(self.pending)
            if start:
                result.append(Text(self.pending[:start]))
                self.pending = self.pending[start:]
                continue
            if self.pending.startswith('&'):
                if self.pending.startswith('&IT&'):
                    result.append(Text('&IT&', '&IT&'))
                    self.pending = self.pending[4:]
                elif '&IT&'.startswith(self.pending) and not final:
                    break
                else:
                    result.append(Text(self.pending[0]))
                    self.pending = self.pending[1:]
                continue
            end = self.pending.find('>')
            if end < 0:
                if final or len(self.pending) > 4096:
                    result.append(Text(self.pending))
                    self.pending = ''
                break
            raw, self.pending = self.pending[:end + 1], self.pending[end + 1:]
            name, sep, value = raw[1:-1].partition(':')
            if sep and name in VALUE_TAGS and value:
                result.append(Text(raw, name, value))
            elif re.fullmatch(r'<R[+-][0-9]{2}>', raw):
                result.append(Text(raw, 'R', raw[2:-1]))
            else:
                result.append(Text(raw, raw[1:-1] if raw[1:-1] in TAGS else None))
        return result


def emit_tag(name: str, value: str | int | None = None) -> str:
    # ON2AD V16.1.8 pp125-127: bare Q announces present/auto-QSY-ready;
    # AWQ additionally announces away and can trigger the peer's VMail UI.
    if name == 'QSYF':
        if (not re.fullmatch(r'[0-9]+', str(value))
                or not 0 < int(value) < 10**12):
            raise ValueError('QSYF requires a positive frequency of at most 12 digits')
        return f'<QSYF>{int(value):012d}</QSYF>'
    if name == 'Q' and value is None:
        return '<Q>'
    if name in VALUE_TAGS and value is not None:
        value = str(value)
        if not value or any(c in value for c in '<>\r\n'):
            raise ValueError('No evidenced escaping for tag values')
        if name == 'Q' and not re.fullmatch(r'-?[0-9]+', value):
            raise ValueError('Q requires a decimal offset in tens of Hz')
        if name == 'QF' and (not re.fullmatch(r'[0-9]+', value) or int(value) <= 0):
            raise ValueError('QF requires a positive decimal frequency in Hz')
        return f'<{name}:{value}>'
    if name == 'R' and value is not None:
        if isinstance(value, int):
            value = f'{value:+03d}'
        if re.fullmatch(r'[+-][0-9]{2}', str(value)):
            return f'<R{value}>'
    if value is None and name in (*BARE_TAGS, 'AWAY', '&IT&', 'BLR', 'LHR'):
        return TAGS[name].wire
    raise ValueError(f'No evidenced emission syntax for tag {name!r}')


def frame_message(text: str) -> bytes:
    content = text.encode('utf-8')
    return str(len(content)).encode('ascii') + b' ' + content


class MessageParser:
    """Application lengths count bytes, independently of modem/TCP frames."""

    def __init__(self):
        self.pending = bytearray()

    def feed(self, chunk: bytes, *, final=False) -> list[bytes]:
        self.pending.extend(chunk)
        result = []
        while self.pending:
            header, sep, content = self.pending.partition(b' ')
            if not header.isdigit() or len(header) > 9:
                raise ValueError('Invalid VarAC message length')
            if not sep:
                break
            size = int(header)
            if size > 1_000_000:
                raise ValueError('VarAC message exceeds receive limit')
            if len(content) < size:
                break
            # A VMail's length leaves out its closing '>', which follows as one more byte
            # (P1 session 021: 196 declared, 197 sent).
            if (b'<SM>' in content[:size] and content[size - 1:size] != b'>'
                    and decode_application(content[:size].decode('utf-8', errors='replace')).envelope is not None):
                if len(content) == size and not final:
                    break
                size += content[size:size + 1] == b'>'
            result.append(bytes(content[:size]))
            del self.pending[:len(header) + 1 + size]
        return result


CALLING_HZ = {80: 3_595_000, 40: 7_105_000, 30: 10_133_000,
              20: 14_105_000, 17: 18_107_000, 15: 21_105_000,
              12: 24_927_000, 10: 28_105_000, 6: 50_330_000}
#: Stock VarAC calls CQ twice on the calling frequency, the second frame starting
#: 10.0 s after the first: 20 of 27 same-station pairs in the P1 census
#: start 9.96–10.05 s apart, the rest
#: 11–17 s, and none runs to a third frame.
CQ_FRAMES = 2
CQ_PERIOD_S = 10.0
#: VarAC's three VARA HF widths, carried as CQ codes 1/2/3 (varac-p3 BENCH B1). VarAC's
#: code of behaviour keeps the CF and slots at 500 (deck slide 23), so 500 is the default.
BANDWIDTHS = (500, 2300, 2750)
DEFAULT_BW = 500
FAREWELL_IDLE_S = {500: 25, 2300: 10, 2750: 10}
#: The slot step follows the modem, not the width: VarAC V8.2.0 sets 750 Hz for VARA HF
#: (2550 Hz only for VARA SAT), so the ten offsets hold at every HF bandwidth.
SLOT_OFFSETS = (-3750, -3000, -2250, -1500, -750, 750, 1500, 2250, 3000, 3750)
SNIFF_S = 5
#: How long a CQ caller waits on its slot. In the P1 census the answering connect
#: request came 8–109 s after the CQ started, so a 60 s wait would have missed one of the seven linked pairs.
SLOT_WAIT_S = 120
#: How long a link still being set up when the wait ends may take to finish; stock
#: connects in about 8.5 s (chat-p0 CHAT-FINDINGS).
LINK_SETUP_S = 30
#: A heard CQ older than this is answered with a warning: its caller returned to the
#: calling frequency long ago.
HEARD_STALE_S = 15 * 60
#: Default 15-minute cadence (DECK 9, FAQ); operator may choose any positive
#: whole-minute interval per explicit user override on 2026-10-24. In the P1 census each
#: station's -9 frames follow 15 min (or a multiple) after its last one plus a slip of
#: seconds, and N1AMD-9 came 15.0 min apart on 10-22: the interval runs from the last beacon.
BEACON_MIN = 15
BEACON_CLEAR_S = 5
BEACON_LIFETIME_S = 24 * 60 * 60
#: Ordinary CQ retry quiet time; beacon timing is independently named above.
#: Release V7.0.8 (2023-03-17) specifies 5 s before beacon/relay, superseding
#: the undated FAQ's 60 s.
CLEAR_S = 5
#: Kestrel waits eight seconds after a request unkeys before retrying, so the last
#: request keyed before the deadline is heard out that long; a key that never drops
#: still ends the attempt CONNECT_GRACE_S past the deadline.
CONNECT_REPLY_S = 8
CONNECT_GRACE_S = 30
#: VarAC's advanced beacon, as 26 stations keyed it on 10-23 and stock's KISS port hands
#: it over: twenty characters, `2`, a digit, a
#: status letter, the six-character locator and the callsign, space padded. The letter is
#: '@' plus service bits, matched against stations' own <S:...> service tags: 1 is set by
#: every station (VMail, most likely), 2 an email gateway, 4 an AI gateway, 8 probably a
#: BBS. The digit after the 2 never changes for a station and is not yet understood.
#: KISS command 2 both ways; stock drops command 0 text and falls over on command 1.
#: The subsquare arrives in either case: VA2OPS keys `FN35fl`.
BEACON_TEXT = re.compile(r'(2[01][A-Z])([A-R]{2}[0-9]{2}[A-Xa-x]{2})([A-Z0-9]{3,7}) *')
BEACON_KISS = 2
SERVICE_BITS = ((1, 'VMail'), (2, 'email'), (4, 'AI'), (8, 'BBS'))


SERVICE_LETTERS = {'V': 1, 'E': 2, 'A': 4, 'B': 8}


def status_letter(offered):
    """The beacon letter for the services in an <S:...> tag; VMail, the one we take, by default."""
    return chr(ord('@') + sum(SERVICE_LETTERS.get(c, 0) for c in set(offered or 'V')))


def services(status):
    bits = ord(status[-1]) - ord('@')
    return ' '.join(name for bit, name in SERVICE_BITS if bits & bit) or 'none'
BEACON_CHARS = 20
#: VarAC limits an AWAY link on the calling frequency to 5 min (REL V5.1.8).
AWAY_LINK_S = 5 * 60
#: What /follow replays of a channel's QSO when it starts.
REPLAY_S = 120
REPLAY_LINES = 200
#: VarAC's VMail cap: 350 characters at 500 Hz (REL V6.4.15), 500 since V8.0.6.
VMAIL_MAX = 500
DEFAULT_AWAY = 'Away from the station; please leave a VMail.'


class SlotCodec:
    """CQ suffix IDs for the production mapping; 9 denotes a beacon on the CF."""

    source = 'P1 suffix/physical-slot pairs; production mapping'
    slots = (*range(1, 6), *range(11, 16))

    def offset_hz(self, slot: int) -> int:
        if slot not in self.slots:
            raise ValueError('Slot ID must be 1–5 (below CF) or 11–15 (above CF)')
        return (-slot if slot < 10 else slot - 10) * 750

    def cq_command(self, call: str, slot: int, bandwidth: int = DEFAULT_BW) -> str:
        self.offset_hz(slot)
        if bandwidth not in BANDWIDTHS:
            raise ValueError('VarAC CQs carry 500, 2300 or 2750 Hz')
        base = base_call(call)
        return f'CQFRAME {base}-{slot} {bandwidth}'

    def decode(self, call: str) -> tuple[str, int | None]:
        base, sep, suffix = callsign(call).rpartition('-')
        if not sep or not suffix.isdigit():
            raise ValueError('CQ has no slot suffix')
        slot = int(suffix)
        if slot == 9:
            return base, None
        self.offset_hz(slot)
        return base, slot


DEFAULT_CODEC = SlotCodec()


class EvidenceMissing(ValueError):
    pass


def slot_frequency(band: int, slot: int, codec: SlotCodec | None = DEFAULT_CODEC) -> int:
    if codec is None or not codec.source:
        raise EvidenceMissing('An evidenced slot codec is required')
    offset = codec.offset_hz(slot)
    if offset not in SLOT_OFFSETS:
        raise ValueError('A slot must be one of the ten 750 Hz offsets')
    return CALLING_HZ[band] + offset


def callsign(call: str) -> str:
    call = call.upper()
    if not re.fullmatch(r'[A-Z0-9/-]+', call):
        raise ValueError('Expected a callsign of letters, digits, / and -')
    return call


def canonical_call(call: str) -> str:
    call = callsign(call)
    match = re.fullmatch(r'(.+)-([0-9]{1,2})', call)
    if match:
        return match[1] + (f'-{int(match[2])}' if int(match[2]) else '')
    return call


def base_call(call: str) -> str:
    return re.sub(r'-[0-9]{1,2}$', '', callsign(call))


@dataclass(frozen=True)
class Event:
    kind: str
    raw: str
    sequence: int | None = field(default=None, compare=False)
    generation: int | None = field(default=None, compare=False)
    diagnostic: bool = field(default=False, compare=False)


class HostError(RuntimeError):
    pass


class Unsupported(HostError):
    pass


class Client:
    """Single-owner command/data pump. Sockets and readiness are injectable."""

    def __init__(self, command, data, kiss_socket=None, *, clock=time.monotonic,
                 ready=None, event: Callable[[Event], None] = lambda event: None):
        self.command_socket, self.data_socket, self.kiss_socket = command, data, kiss_socket
        self._kiss = kiss.Parser()
        self.clock, self.event = clock, event
        self.ready = ready or (lambda sockets, timeout: select.select(sockets, [], [], timeout)[0])
        self._command = bytearray()
        self._messages = MessageParser()
        self.messages = deque(maxlen=4096)
        self.message_sequence = 0
        self.link_message_start = 0
        self.link_generation = 0
        self._outbound_connect_pending = False
        self._connect_deadline = 0.0
        self._connect_keyed = False
        self._connect_listen_end = 0.0
        self._sent_after = None
        self._lines = deque(maxlen=4096)
        self.sequence = 0
        self.connected = False
        self.busy = None
        self.busy_events = 0
        self.calling = False
        self.ptt = False
        self.pending = False
        self.closed = False
        self._receive_quarantined = False
        self.raw_transfer = None
        self.snr = None
        self.bitrate = None
        self.buffer = 0
        self.buffer_sequence = 0
        self._discard_sequence = 0
        self.version = None
        self.bandwidth = None
        self.data_writer = None
        self._writes = {}
        self._stream_poisoned = False

    def enable_writer(self):
        """Serialize every data write once interactive application use starts."""
        if self._stream_poisoned:
            raise HostError('Data stream is poisoned; reconnect the host')
        if self.data_writer is None:
            from .data_writer import DataWriter
            self.data_writer = DataWriter(self.data_socket)

    def _poll_writer(self):
        if self.data_writer is None:
            return
        for completion in self.data_writer.poll():
            record = self._writes.pop(completion.write_id, None)
            if record is None:
                continue
            generation, payload = record
            if completion.disposition == 'unknown':
                self._stream_poisoned = True
            if generation != self.link_generation:
                continue
            if completion.disposition == 'complete':
                # Fresh BUFFER evidence must follow known socket completion.
                self._sent_after = self.sequence
                if self.raw_transfer is not None:
                    self.raw_transfer.raw('tx', payload)
            self.event(Event('write_complete', json.dumps(dict(
                write_id=completion.write_id, generation=generation,
                disposition=completion.disposition, after=self.sequence))))

    @classmethod
    def open(cls, host='127.0.0.1', command_port=8300, data_port=8301, kiss_port=None, *,
             connect=socket.create_connection, timeout=5, **kwargs):
        """The KISS port is stock's "Enable KISS" setting, off by default; without it the
        session beacons as a CQFRAME."""
        opened = []
        try:
            for port in (command_port, data_port, kiss_port):
                opened.append(connect((host, port), timeout=timeout) if port else None)
        except BaseException:
            for stream in opened:
                if stream is not None:
                    stream.close()
            raise
        return cls(*opened, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _text(self, chunk=b'', *, final=False):
        if self._receive_quarantined:
            return
        generation = self.link_generation
        try:
            messages = self._messages.feed(chunk, final=final)
        except ValueError as error:
            rejected = bytes(self._messages.pending)
            self._messages.pending.clear()
            self._receive_quarantined = True
            self.event(Event('peer_protocol_error', json.dumps({
                'reason': str(error), 'generation': self.link_generation, 'bytes': len(rejected),
                'hex': rejected[:4096].hex(), 'truncated': len(rejected) > 4096})))
            return
        for content in messages:
            if generation != self.link_generation:
                break
            text = content.decode('utf-8', errors='replace')
            self.message_sequence += 1
            sequence = self.message_sequence
            self.messages.append((self.message_sequence, text))
            self.event(Event('message', text, sequence, generation))
            for part in TextParser().feed(text, final=True):
                self.event(Event('tag' if part.tag else 'text', part.raw,
                                 sequence, generation,
                                 diagnostic='<SM>' in text))
        if final and self._messages.pending:
            self.event(Event('incomplete', bytes(self._messages.pending).decode('utf-8', errors='replace')))
            self._messages.pending.clear()

    def wait_message_steps(self, predicate, *, after, timeout=30):
        end = self.clock() + timeout
        while True:
            for seq, text in self.messages:
                if seq > after and predicate(text):
                    return text
            if not self.connected or self.closed:
                raise HostError('Link ended during application exchange')
            remaining = end - self.clock()
            if remaining <= 0:
                raise TimeoutError('peer application response timed out')
            yield min(remaining, .1)

    def wait_message(self, predicate, *, after, timeout=30):
        return self.finish(self.wait_message_steps(predicate, after=after, timeout=timeout))

    def drain_steps(self, *, timeout=30, cancelled=lambda: False):
        """Done once the last send is delivered and the key is up."""
        after, end = self._sent_after, self.clock() + timeout
        delivered = after is None and not (self.data_writer and self.data_writer.pending)
        while True:
            if cancelled():
                return
            if self._stream_poisoned:
                raise HostError('Data write has unknown extent; framed stream is poisoned')
            if self._sent_after != after:
                after = self._sent_after
                delivered = after is None
            writing = bool(self.data_writer and self.data_writer.pending)
            if after is not None and self._discard_sequence > after:
                raise HostError("Transmit queue cleared without delivery confirmation")
            if writing:
                delivered = False
            elif not delivered:
                lines = self.since(after)
                delivered = 'DISCONNECTED' in lines or ('BUFFER 0' in lines and self.buffer == 0)
            elif not (self.ptt or self.pending):
                break
            if self.clock() >= end:
                raise TimeoutError('Peer has not acknowledged the last over' if not delivered
                                   else 'Host did not become unkeyed')
            yield .1
        if not self.connected:
            raise HostError('Link ended before delivery')
        self._sent_after = None

    def drain(self, *, timeout=30):
        self.finish(self.drain_steps(timeout=timeout))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.data_writer is not None:
            self.data_writer.close()
        try:
            for stream in (self.command_socket, self.data_socket, self.kiss_socket):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        finally:
            self._text(final=True)

    def _retire_link_messages(self):
        """Retire complete and partial application evidence with its link."""
        self._text(final=True)
        self.messages.clear()
        self.link_message_start = self.message_sequence
        self.link_generation += 1
        if self.data_writer is not None and self.data_writer.pending:
            # Old framed writes cannot be allowed onto a replacement link.
            self._stream_poisoned = True
            self.data_writer.close()

    def _line(self, raw):
        self.sequence += 1
        self._lines.append((self.sequence, raw))
        words = raw.split()
        if raw.startswith('CONNECTED '):
            if not (self.connected or self.pending or self._outbound_connect_pending):
                self._receive_quarantined = False
            self.connected, self.pending = True, False
            self._outbound_connect_pending = False
        elif raw == 'DISCONNECTED':
            self.connected, self.pending = False, False
            self._retire_link_messages()
            self._outbound_connect_pending = False
            self._sent_after = None
            # This link's queue is retired, not acknowledged by its peer.
            self.buffer = 0
            self.buffer_sequence = self.sequence
        elif raw in ('CLEANTXBUFFER OK', 'CLEANTXBUFFER BUFFEREMPTY'):
            self._discard_sequence = self.sequence
            # Queue state evidence is not a synthetic BUFFER 0 / delivery ACK.
            self.buffer = 0
            self.buffer_sequence = self.sequence
        elif raw in ('BUSY ON', 'BUSY OFF'):
            self.busy = raw == 'BUSY ON'
            self.busy_events += int(self.busy)
        elif raw in ('PTT ON', 'PTT OFF'):
            self.ptt = raw == 'PTT ON'
            if self._outbound_connect_pending:
                now = self.clock()
                if self.ptt and now < self._connect_deadline:
                    self._connect_keyed = True
                elif not self.ptt and self._connect_keyed:
                    self._connect_keyed = False
                    self._connect_listen_end = now + CONNECT_REPLY_S
        elif raw in ('PENDING', 'CANCELPENDING'):
            if raw == 'PENDING' and not (self.connected or self.pending or self._outbound_connect_pending):
                self._retire_link_messages()
                self._receive_quarantined = False
            self.pending = raw == 'PENDING'
        elif words[:1] == ['SN'] and len(words) == 2:
            try:
                self.snr = None if words[1] == '--' else float(words[1])
            except ValueError:
                pass
        elif words[:1] == ['BUFFER'] and len(words) == 2:
            try:
                self.buffer = int(words[1])
                self.buffer_sequence = self.sequence
            except ValueError:
                pass
        elif words[:1] == ['BITRATE']:
            self.bitrate = raw
        self.event(Event('host', raw))

    def poll(self, timeout=0.1):
        if self.closed:
            raise HostError('Host connection is closed')
        self._poll_writer()
        streams = [self.command_socket, self.data_socket] + ([self.kiss_socket] if self.kiss_socket else [])
        for stream in self.ready(streams, timeout):
            chunk = stream.recv(65536)
            if not chunk:
                self.close()
                raise HostError('VARA host closed its command or data port')
            if stream is self.kiss_socket:
                for port, command, payload in self._kiss.feed(chunk):
                    if not port and command == BEACON_KISS:
                        self.event(Event('kiss', payload.decode('ascii', errors='replace')))
                    elif not port and command in (0, 1):
                        from .broadcast import decode_ui
                        try:
                            decode_ui(payload, command=command)
                        except ValueError:
                            self.event(Event('broadcast_error', 'invalid direct UI envelope'))
                        else:
                            self.event(Event('broadcast', json.dumps(dict(
                                command=command, hex=payload.hex()))))
            elif stream is self.data_socket:
                if self.raw_transfer is None or not self.raw_transfer.raw("rx", chunk):
                    self._text(chunk)
            else:
                self._command.extend(chunk)
                while b'\r' in self._command:
                    line, _, rest = self._command.partition(b'\r')
                    self._command[:] = rest
                    self._line(line.decode('ascii', errors='replace').lstrip('\n'))
                if len(self._command) > 65536:
                    raise HostError('Unterminated VARA host line')
        self._poll_writer()

    def wait(self, predicate, *, after, timeout=10):
        end = self.clock() + timeout
        while True:
            for seq, line in self._lines:
                if seq > after:
                    if predicate(line):
                        return line
                    if line.startswith('WRONG'):
                        raise Unsupported(line)
            remaining = end - self.clock()
            if remaining <= 0:
                raise TimeoutError('VARA host response timed out')
            self.poll(min(remaining, 0.1))

    def send_command(self, command):
        """Issue one host command and return its response cursor, without waiting."""
        if self.closed:
            raise HostError('Host connection is closed')
        if not command.isascii() or any(ord(c) < 32 for c in command):
            raise ValueError('Expected one ASCII host command')
        after = self.sequence
        self.command_socket.sendall(command.encode('ascii') + b'\r')
        self.event(Event('command', command))
        return after

    def request(self, command, *, timeout=10, accepts=None):
        after = self.send_command(command)
        match = accepts or (lambda line: line == 'OK')
        try:
            reply = self.wait(lambda line: line == 'WRONG' or match(line), after=after, timeout=timeout)
        except TimeoutError:
            self.close()
            raise
        if reply == 'WRONG':
            raise Unsupported(f'VARA rejected {command}')
        return reply

    def initialize(self, mycall, bandwidth=DEFAULT_BW):
        """One width for the whole session: stock answers a connect only at its own BW
        (docs/protocols/vara/04 §4.2.2), so the CQ's width is also the slot's."""
        if bandwidth not in BANDWIDTHS:
            raise ValueError('VarAC runs VARA HF at 500, 2300 or 2750 Hz')
        self.mycall = callsign(mycall)
        for command in ('LISTEN OFF', 'PUBLIC ON', 'CWID OFF', 'COMPRESSION TEXT',
                        f'MYCALL {self.mycall}'):
            self.request(command)
        self.set_bandwidth(bandwidth)
        self.version = self.request('VERSION', accepts=lambda line: line.startswith('VERSION '))
        self.request('CHAT ON')
        self.request('LISTEN OFF')

    def set_bandwidth(self, bandwidth):
        if bandwidth not in BANDWIDTHS:
            raise ValueError('VarAC runs VARA HF at 500, 2300 or 2750 Hz')
        if self.connected or self.pending or self.ptt:
            raise HostError('Bandwidth changes need an idle host')
        self.request(f'BW{bandwidth}')
        self.bandwidth = bandwidth

    def listen(self):
        try:
            self.request('LISTEN CQ')
        except Unsupported:
            self.request('LISTEN ON')

    def send_text(self, text):
        return self._send(frame_message(text), text)

    def send_vmail(self, vmail):
        """As session 021 sent its VMail: the length counts all but the closing '>'."""
        text = vmail.wire()
        self._send(frame_message(text[:-1]) + b'>', text)

    def _send(self, payload, text):
        if self.closed or not self.connected:
            raise HostError('Text requires a connected session')
        if self._stream_poisoned:
            raise HostError('Data stream is poisoned; no additional frames are safe')
        self._sent_after = self.sequence
        write_id = None
        if self.data_writer is not None:
            from .data_writer import WriterUnavailable
            try:
                write_id = self.data_writer.submit(payload, self.link_generation)
            except WriterUnavailable as error:
                raise HostError(str(error)) from error
            self._writes[write_id] = (self.link_generation, payload)
        else:
            self.data_socket.sendall(payload)
            if self.raw_transfer is not None:
                self.raw_transfer.raw("tx", payload)
        self.event(Event('sent', text))
        return write_id

    def send_tag(self, name, value=None):
        self.send_text(emit_tag(name, value))

    def clean_buffer(self):
        reply = self.request('CLEANTXBUFFER', accepts=lambda line: line.startswith('CLEANTXBUFFER '))
        return reply[len('CLEANTXBUFFER '):]

    # Long exchanges are generators: each yield hands the loop one poll, so the
    # terminal keeps reading while a CQ plays out. `finish` runs one to its end.

    def finish(self, steps):
        while True:
            try:
                hint = next(steps)
            except StopIteration as done:
                return done.value
            self.poll(min(.1, hint) if hint else .1)

    def since(self, after):
        return [line for seq, line in self._lines if seq > after]

    def cq_steps(self, command, *, timeout=20, retry=True):
        if not command.startswith('CQFRAME '):
            raise ValueError('Slot codec must produce a CQFRAME host command')
        yield from self._keyed_steps(lambda: self.request(command), 'CQFRAME', 'CQ', timeout=timeout, retry=retry)

    def beacon_steps(self, text, *, timeout=20):
        """VarAC's advanced beacon: the text as a command-2 KISS frame, keyed as a CQFRAME
        is, and like a CQFRAME dropped by the host while it hears the channel busy."""
        if self.kiss_socket is None:
            raise Unsupported('the host has no KISS port')
        if not BEACON_TEXT.fullmatch(text) or len(text) > BEACON_CHARS:
            raise ValueError('a beacon is 2, a digit, a status letter, the locator and the callsign')
        wire = kiss.encode(text.ljust(BEACON_CHARS).encode('ascii'), BEACON_KISS)

        def send():
            self.kiss_socket.sendall(wire)
            self.event(Event('kiss-sent', text))
        yield from self._keyed_steps(send, 'beacon', 'beacon', timeout=timeout, retry=False)

    def forced_beacon_steps(self, text=None, *, timeout=20):
        if text is not None:
            if not BEACON_TEXT.fullmatch(text) or len(text) > BEACON_CHARS:
                raise ValueError('invalid advanced beacon text')
            command = f'BEACONFORCE {text}'
        else:
            command = f'CQFRAMEFORCE {base_call(self.mycall)}-9 {self.bandwidth}'
        yield from self._keyed_steps(lambda: self.request(command), command.split()[0],
                                    'forced beacon', timeout=timeout, retry=False, force=True)

    def _keyed_steps(self, send, verb, noun, *, timeout, retry=True, force=False):
        if self.connected or self.pending or self.ptt:
            raise HostError(f'{noun} requires an idle host')
        after = sent = self.sequence
        deadline = self.clock() + timeout
        cleared = None
        seen = self.busy_events
        self.calling = True
        try:
            held, told = self.busy and not force, False
            send()
            while True:
                lines = self.since(after)
                if 'CANCELPENDING' in lines:
                    if 'PENDING' not in lines or lines.index('PENDING') > lines.index('CANCELPENDING'):
                        raise Unsupported('CQ completion has no matching PENDING')
                    if not self.ptt:
                        if force and 'WRONG' in lines:
                            raise Unsupported('forced beacon did not transmit')
                        if 'PTT ON' not in lines:
                            raise HostError(f'{noun} completed without a reported key-up; not counted as sent')
                        break
                # The host answers OK and drops a CQFRAME that arrives while it hears the
                # channel busy, as stock does, and says nothing; call again once it has
                # reported the channel clear for as long as VarAC wants before it keys.
                dropped = not force and 'PENDING' not in lines and 'BUSY OFF' in self.since(sent)
                if not told and (held or dropped):
                    self.event(Event('held', noun))
                    told = True
                if dropped and not retry:
                    raise Unsupported('beacon dropped while busy')
                if dropped and not self.busy:
                    if cleared is None or self.busy_events != seen:
                        cleared, seen = self.clock(), self.busy_events
                    elif self.clock() - cleared >= CLEAR_S:
                        sent, cleared = self.sequence, None
                        held, told = self.busy and not force, False
                        send()
                else:
                    cleared = None
                if self.clock() >= deadline:
                    raise TimeoutError
                yield
        except TimeoutError:
            if not force and (self.busy or cleared is not None):
                raise Unsupported(f'the calling frequency stayed busy; {noun} not sent') from None
            raise Unsupported(f'{verb} accepted but never keyed') from None
        except Unsupported:
            if force:
                raise Unsupported(f'{verb} unavailable or rejected by host; no ordinary beacon fallback') from None
            if not retry and (self.busy or 'BUSY OFF' in self.since(sent)):
                raise Unsupported('beacon dropped while busy') from None
            raise Unsupported(f'{verb} rejected (the host answered WRONG)') from None
        finally:
            self.calling = False

    def connect_steps(self, peer, *, timeout=90, cancelled=lambda: False):
        """CONNECTED, or a HostError; `cancelled` is read each step and aborts the attempt."""
        if self.connected or self.pending or self.ptt:
            raise HostError('Connect requires an idle host')
        after = self.sequence
        deadline = self.clock() + timeout
        self._retire_link_messages()
        self._receive_quarantined = False
        self._connect_deadline = deadline
        self._connect_keyed = False
        self._connect_listen_end = 0.0
        self._outbound_connect_pending = True
        try:
            self.request(f'CONNECT {self.mycall} {callsign(peer)}')
        except BaseException:
            self._outbound_connect_pending = False
            self._retire_link_messages()
            raise
        ended = None
        while not self.connected:
            if 'DISCONNECTED' in self.since(after):
                raise HostError(ended or 'Connection attempt ended without a link')
            now = self.clock()
            listening = ((self._connect_keyed or now < self._connect_listen_end)
                         and now < deadline + CONNECT_GRACE_S)
            if ended is None and (cancelled() or now >= deadline and not listening):
                ended = 'stopped' if cancelled() else 'timed out'
                # ABORT replaces the modem's station without a DISCONNECTED; the key drops
                # when the frame in the air ends.
                self.request('ABORT')
            elif ended and not self.ptt:
                self._outbound_connect_pending = False
                self._retire_link_messages()
                if ended == 'stopped':
                    return False
                raise HostError('Connection attempt timed out')
            yield
        return True

    def disconnect(self, *, timeout=30):
        after = self.sequence
        self.request('DISCONNECT')
        self.wait(lambda line: line == 'DISCONNECTED', after=after, timeout=timeout)


class State(str, Enum):
    CALLING = 'calling-frequency'
    MONITORING = 'monitoring'
    SNIFFING = 'sniffing-slot'
    CQ = 'calling-cq'
    WAITING = 'waiting-on-slot'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    RETURNED = 'returned-to-calling-frequency'
    FAILED = 'failed'


class Flow:
    def __init__(self, client: Client, tuner: Callable[[int], None], *, band=20,
                 codec: SlotCodec | None = DEFAULT_CODEC, occupied=None,
                 profile=None, services=None, received_retune_steps=None, mail_ready=lambda: None):
        self.client, self.tuner, self.band, self.codec = client, tuner, band, codec
        self.occupied = occupied or (lambda: client.busy)
        self.state = State.CALLING
        self.cancelled = False
        self.dial = None
        self.profile = profile or {}
        self.services = services
        self._handled = 0
        self.qsy_slot = self.qsy_phase = None
        self.qsy_receiving = False
        self.received_retune_steps = received_retune_steps
        self.mail_ready = mail_ready

    def tune(self, dial):
        if self.client.ptt or self.client.pending or self.client.connected:
            raise HostError('Cannot QSY while keyed, pending or connected')
        self.tuner(dial)
        self.client.busy = None
        self.dial = dial
        self.client.event(Event('tune', str(dial)))

    def _state(self, state):
        self.state = state
        self.client.event(Event('state', state.value))

    # The flows are generators stepped by one poll loop, so the operator's terminal
    # reads on through a sniff, both CQ frames and the slot wait. A yielded number
    # is how long that poll may block. `cancel()` is honoured at the next frame
    # boundary, never mid-frame; `Client.finish` runs a flow to its end.

    def cancel(self):
        self.cancelled = True

    def sniff_steps(self, dial, seconds):
        self.client.request('LISTEN OFF')
        while self.client.ready([self.client.command_socket, self.client.data_socket], 0):
            self.client.poll(0)
        busy_events = self.client.busy_events
        self.tune(dial)
        self._state(State.SNIFFING)
        end = self.client.clock() + seconds
        clear = True
        while self.client.clock() < end and not self.cancelled:
            yield end - self.client.clock()
            clear = clear and self.occupied() is not True and self.client.busy_events == busy_events
        return clear and not self.cancelled

    def return_to_calling(self):
        self.client.request('LISTEN OFF')
        self.tune(CALLING_HZ[self.band])
        self._state(State.RETURNED)

    def cq_steps(self, slot, *, sniff_seconds=SNIFF_S, wait_seconds=SLOT_WAIT_S):
        """True once a station has connected on the slot; False when the slot was busy,
        nobody answered or the operator cancelled, back on the calling frequency."""
        dial = slot_frequency(self.band, slot, self.codec)
        command = self.codec.cq_command(self.client.mycall, slot, self.client.bandwidth)
        if not command.startswith('CQFRAME ') or any(ord(c) < 32 for c in command):
            raise ValueError('Invalid CQ slot command')
        if sniff_seconds <= 0 or wait_seconds <= 0:
            raise ValueError('Sniff and answer windows must be positive')
        self.cancelled = False
        try:
            if not (yield from self.sniff_steps(dial, sniff_seconds)):
                self.return_to_calling()
                return False
            self.tune(CALLING_HZ[self.band])
            self._state(State.CQ)
            start = self.client.clock()
            for n in range(1, CQ_FRAMES + 1):
                due = start + CQ_PERIOD_S if n > 1 else start
                while self.client.clock() < due and not self.cancelled:
                    yield due - self.client.clock()
                if self.cancelled:
                    self.return_to_calling()
                    return False
                start = self.client.clock()
                self.client.event(Event('cq', f'{n}/{CQ_FRAMES}'))
                yield from self.client.cq_steps(command)
            if self.cancelled:
                self.return_to_calling()
                return False
            self.tune(dial)
            self._state(State.WAITING)
            self.client.listen()
            end = self.client.clock() + wait_seconds
            while not self.client.connected:
                if self.cancelled or self.client.clock() >= end:
                    # A link being set up is seen through, whichever way it ends.
                    if not (self.client.pending or self.client.ptt):
                        self.return_to_calling()
                        return False
                    if self.client.clock() >= end + LINK_SETUP_S:
                        raise HostError('Slot wait ended during link setup; staying on the slot')
                yield
            self._state(State.CONNECTED)
            return True
        except BaseException:
            self._state(State.FAILED)
            raise

    def call_steps(self, peer, *, slot=None, timeout=90):
        peer = callsign(peer)
        dial = slot_frequency(self.band, slot, self.codec) if slot is not None else None
        self.cancelled = False
        self.client.request('LISTEN OFF')
        if dial is not None:
            self.tune(dial)
        self._state(State.CONNECTING)
        try:
            if not (yield from self.client.connect_steps(peer, timeout=timeout,
                                                        cancelled=lambda: self.cancelled)):
                self.return_to_calling()
                return False
        except BaseException:
            self._state(State.FAILED)
            raise
        self._state(State.CONNECTED)
        return True

    def opening_steps(self, *, timeout=90):
        end = self.client.clock() + timeout
        while self.client.snr is None:
            if not self.client.connected or self.client.closed:
                raise HostError('Link ended while waiting for an opening signal report')
            if self.client.clock() >= end:
                raise TimeoutError('Opening signal report timed out')
            yield end - self.client.clock()
        # R has exactly two decimal digits on the observed wire. A noiseless
        # local modem path can estimate more than 99 dB; saturate only the
        # transmitted report, retaining the measured host value for diagnostics.
        report = emit_tag('R', max(-99, min(99, round(self.client.snr))))
        text = f'de {self.client.mycall} {report}'
        if self.services is not None:
            text += ' ' + emit_tag('S', self.services)
        profile = '\r\n'.join(emit_tag(name, self.profile[name])
                              for name in ('NAME', 'QTH', 'LOC') if self.profile.get(name))
        self.client.send_text(text)
        if profile:
            self.client.send_text(profile)

    def opening(self, **kwargs):
        return self.client.finish(self.opening_steps(**kwargs))

    def _qsy_steps(self, offset, *, timeout):
        if offset not in SLOT_OFFSETS:
            raise ValueError('QSY must select one of the ten 750 Hz slots')
        if self.dial != CALLING_HZ[self.band]:
            raise HostError('QSY exchange must start on the calling frequency')
        yield from self.client.drain_steps(timeout=timeout)
        self.tuner(CALLING_HZ[self.band] + offset)
        self.dial = CALLING_HZ[self.band] + offset
        self.client.busy = None
        self.client.event(Event('tune', str(self.dial)))

    def _qsy(self, offset, *, timeout):
        return self.client.finish(self._qsy_steps(offset, timeout=timeout))

    def request_qsy_steps(self, slot, *, timeout=90):
        offset = self.codec.offset_hz(slot)
        if self.dial != CALLING_HZ[self.band]:
            raise HostError('QSY currently supports calling-frequency to slot moves only')
        if not self.client.connected:
            raise HostError('QSY requires an established link')
        link_start = self.client.link_message_start
        generation = self.client.link_generation
        def current_link():
            if self.client.link_generation != generation or not self.client.connected or self.client.closed:
                raise HostError('Link changed during QSY readiness wait; no invitation sent')
            if self.dial != CALLING_HZ[self.band]:
                raise HostError('Frequency changed during QSY readiness wait; no invitation sent')
        self.qsy_slot, self.qsy_phase = slot, 'waiting for peer auto-QSY readiness (<AWQ> or <Q>)'
        try:
            # REL V11.0.7: an invitation before the peer's AWQ is not prompted.
            # ON2AD V16.1.8 p127 adds bare Q for present/auto-QSY-ready peers.
            # The boundary precedes CONNECT, so early readiness in the same poll
            # as CONNECTED is retained; prior contacts cannot authorize a move.
            try:
                readiness = self.client.wait_message_steps(
                    lambda text: self.client.connected and not self.client.closed
                                 and any(part.raw in ('<AWQ>', '<Q>')
                                     for part in decode_application(text).parts),
                    after=link_start, timeout=timeout)
                for hint in readiness:
                    current_link()
                    yield hint
                current_link()
            except TimeoutError:
                raise TimeoutError(f'QSY to slot {slot}: peer did not announce auto-QSY readiness; '
                                   'remaining on the calling frequency') from None
            try:
                # A manual QSY can start with local chat still in the modem.
                # Drain it before queuing the invitation, including after NAK
                # repacketization; modem delivery is not application acceptance.
                self.qsy_phase = 'waiting for local delivery'
                drain = self.client.drain_steps(timeout=timeout)
                try:
                    for hint in drain:
                        current_link()
                        yield hint
                finally:
                    drain.close()
                current_link()
                after = self.client.message_sequence
                self.qsy_phase = 'waiting for invitation delivery'
                # Relative Q assumes both ends know the current rig frequency
                # and support offset invitations. AWQ/Q promises neither.
                # v16.1.8's connected parser handles QSYF with exactly 12 digits.
                # QF is a broadcast notification, despite the manual's table.
                self.client.send_tag('QSYF', CALLING_HZ[self.band] + offset)
                yield from self._qsy_steps(offset, timeout=timeout)
                self.qsy_phase = 'waiting for peer acceptance (<QSYR>)'
                self.client.send_tag('TL')
                self.mail_ready()
                yield from self.client.wait_message_steps(
                    lambda text: any(part.raw == '<QSYR>' for part in decode_application(text).parts),
                    after=after, timeout=timeout)
                self.qsy_phase = 'waiting for opening signal report'
                yield from self.opening_steps(timeout=timeout)
                self._state(State.CONNECTED)
            except BaseException:
                self._state(State.FAILED)
                raise
        finally:
            self.qsy_slot = self.qsy_phase = None

    def request_qsy(self, slot, *, timeout=90):
        return self.client.finish(self.request_qsy_steps(slot, timeout=timeout))

    def received_tuned(self, target):
        """Publish a completed owner move, including completion racing cancellation."""
        if self.dial != target:
            self.dial = target
            self.client.busy = None
            self.client.event(Event('tune', str(target)))

    def received_qsy_target(self, request, origin):
        absolute = re.fullmatch(r'<QSYF>([0-9]{12})</QSYF>', request)
        match = re.fullmatch(r'<(Q|QF):(-?[0-9]+)>', request)
        if absolute is not None:
            value = target = int(absolute[1])
        elif match is not None:
            value = int(match[2])
            target = value if match[1] == 'QF' else origin + value * 10 if origin is not None else None
        else:
            raise ValueError('Expected a measured QSY request')
        if (not value or target == origin
                or target not in {CALLING_HZ[self.band] + n for n in SLOT_OFFSETS}):
            raise ValueError('QSY must select one of the ten 750 Hz slots')
        return target

    def answer_qsy_steps(self, request, *, after=None, timeout=90,
                         generation=None, origin=None):
        """Follow an absolute destination or an offset from the actual dial.

        The modem owner acknowledges received DATA and confirms physical unkey
        before tuning. Unrelated outbound text remains queued across the move.
        """
        origin = self.dial if origin is None else origin
        target = self.received_qsy_target(request, origin)
        generation = self.client.link_generation if generation is None else generation
        after = self.client.message_sequence if after is None else after
        def current_link():
            if (self.client.link_generation != generation or not self.client.connected
                    or self.client.closed):
                raise HostError('Link changed during received QSY')
        current_link()
        if self.received_retune_steps is None:
            raise HostError('Received QSY needs a modem-owner retune adapter')
        self.qsy_receiving = True
        self.qsy_slot = next(slot for slot in self.codec.slots
                             if self.codec.offset_hz(slot) == target - CALLING_HZ[self.band])
        self.qsy_phase = 'waiting for received DATA acknowledgement and physical unkey'
        def wait(steps):
            try:
                while True:
                    current_link()
                    try:
                        hint = next(steps)
                    except StopIteration:
                        break
                    yield hint
                current_link()
            finally:
                steps.close()
        try:
            if self.dial != origin:
                raise HostError('Frequency changed before received QSY')
            end = self.client.clock() + timeout
            transaction = wait(self.received_retune_steps(target, generation))
            try:
                for hint in transaction:
                    if self.client.clock() >= end:
                        raise TimeoutError('Received QSY retune boundary timed out')
                    yield hint
            finally:
                transaction.close()
            self.received_tuned(target)
            self.qsy_phase = 'waiting for peer arrival (<TL>)'
            yield from wait(self.client.wait_message_steps(
                lambda text: any(part.raw == '<TL>' for part in decode_application(text).parts),
                after=after, timeout=timeout))
            after = next(seq for seq, text in self.client.messages if seq > after
                         and any(part.raw == '<TL>' for part in decode_application(text).parts))
            self.client.send_tag('QSYR')
            self.mail_ready()
            self.qsy_phase = 'waiting for peer opening signal report'
            yield from wait(self.client.wait_message_steps(
                lambda text: text.startswith('de ') and any(
                    part.tag == 'R' for part in TextParser().feed(text, final=True)),
                after=after, timeout=timeout))
            yield from wait(self.opening_steps(timeout=timeout))
            self._state(State.CONNECTED)
        except GeneratorExit:
            # Cancellation does not undo a completed physical move.
            raise
        except BaseException:
            self._state(State.FAILED)
            raise
        finally:
            self.qsy_receiving = False
            self.qsy_slot = self.qsy_phase = None

    def answer_qsy(self, request, **kwargs):
        return self.client.finish(self.answer_qsy_steps(request, **kwargs))

    def receive(self):
        """Return one current-link invitation; never run a nested poll loop."""
        for seq, text in list(self.client.messages):
            if seq <= max(self._handled, self.client.link_message_start):
                continue
            self._handled = seq
            if re.fullmatch(r'<(?:Q:-?[0-9]+|QF:[0-9]+)>|<QSYF>[0-9]{12}</QSYF>', text):
                return text, seq, self.client.link_generation, self.dial
        return None

    def close_steps(self, *, timeout=10, cancelled=lambda: False):
        """Discard text, bound the close, then confirm ABORT and the last unkey."""
        client = self.client
        end = client.clock() + timeout
        # Do not wait for CLEANTXBUFFER's answer: FAILED is normal for an over
        # in flight, and a missing answer must never prevent DISCONNECT.
        failed = False
        # ABORT preserves LISTEN; close incoming admission before the unkey wait.
        for command in ('LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT'):
            try:
                client.send_command(command)
            except (HostError, OSError):
                # A broken host still gets every best-effort stop command. If
                # ABORT also fails, propagate so run() stops the local runtime.
                failed = True
        while failed or cancelled() or client.connected or client.ptt or client.pending:
            if failed or cancelled() or client.clock() >= end:
                after = client.sequence
                try:
                    client.send_command('VERSION')
                finally:
                    client.send_command('ABORT')
                end = client.clock() + timeout
                # Host commands/replies are ordered. VERSION fences off an
                # outstanding DISCONNECT OK; only an OK after this tagged
                # response can acknowledge ABORT. Both writes happen now, so
                # the fence never delays issuing the stop command.
                while True:
                    barrier = next((seq for seq, line in client._lines
                                    if seq > after and line.startswith('VERSION ')), None)
                    answers = client.since(barrier) if barrier is not None else []
                    if any(line.startswith('WRONG') for line in answers):
                        raise HostError('Host rejected ABORT')
                    if 'OK' in answers and not client.ptt:
                        client.connected = client.pending = False
                        client._retire_link_messages()
                        client._outbound_connect_pending = False
                        client.buffer = 0
                        client.buffer_sequence = client.sequence
                        client._sent_after = None
                        return
                    if client.clock() >= end:
                        raise TimeoutError('Host did not confirm ABORT and become unkeyed')
                    yield .1
            yield .1

    def farewell_steps(self, text='73', *, timeout=None, cancelled=lambda: False):
        client = self.client
        if timeout is None:
            timeout = FAREWELL_IDLE_S[client.bandwidth or DEFAULT_BW]
        start = client.clock()
        end, hard_end = start + timeout, start + 60
        cursor, messages = client.sequence, client.message_sequence
        buffered = client.buffer
        drained = peer_close = disconnected = False

        def expired():
            nonlocal end, cursor, messages, buffered, drained, peer_close, disconnected
            now = client.clock()
            if cancelled() or now >= min(end, hard_end):
                return True
            progress = False
            for seq, line in client._lines:
                if seq <= cursor:
                    continue
                words = line.split()
                if words[:1] == ['BUFFER'] and len(words) == 2:
                    try:
                        remaining = int(words[1])
                    except ValueError:
                        continue
                    if remaining < 0:
                        continue
                    fresh_zero = (remaining == 0 and not drained
                                  and client._sent_after is not None
                                  and not (client.data_writer and client.data_writer.pending)
                                  and seq > client._sent_after)
                    progress |= remaining < buffered or fresh_zero
                    drained |= fresh_zero
                    buffered = remaining
                elif line == 'DISCONNECTED' and not disconnected:
                    disconnected = progress = True
            for seq, message in client.messages:
                if seq > messages and not peer_close and any(
                        part.tag == 'DISC' for part in TextParser().feed(message, final=True)):
                    peer_close = progress = True
            cursor, messages = client.sequence, client.message_sequence
            if progress:
                end = now + timeout
            return False

        try:
            if not cancelled():
                client.send_text(text + emit_tag('DISC'))
                yield from client.drain_steps(timeout=60, cancelled=expired)
                if not expired():
                    client.send_command('LISTEN OFF')
                    client.send_command('DISCONNECT')
                    end = client.clock() + timeout
                    while True:
                        if expired():
                            break
                        if not (client.connected or client.ptt or client.pending):
                            return
                        yield .1
        except (HostError, TimeoutError, ValueError, OSError):
            # Delivery is best effort. A missing peer ACK must never leave ARQ
            # querying forever instead of issuing the host close commands.
            pass
        yield from self.close_steps(cancelled=lambda: True)

    def farewell(self, text='73', *, timeout=None):
        self.client.finish(self.farewell_steps(text, timeout=timeout))


def khz(hz):
    return f'{hz / 1e3:.3f}'


def age(seconds):
    if seconds < 1:
        return 'now'
    if seconds < 60:
        return f'{seconds:.0f} s'
    if seconds < 3600:
        return f'{seconds / 60:.0f} min'
    return f'{seconds // 3600:.0f} h {seconds % 3600 // 60:.0f} min'


COVERAGE_WINDOW_S = 60
COVERAGE_NOTICE_S = 300
COVERAGE_LOSS_SHARE = 0.02
#: The passive watch dropping a channel to keep up with the audio, and taking it back.
SHED = 'passive follower behind the audio: not reading this channel'
RESUMED = 'passive follower caught up: reading this channel again'


def channel_frequency(band, slot, codec=DEFAULT_CODEC):
    if isinstance(slot, str) and slot.startswith('hz:'):
        return int(slot[3:])
    return CALLING_HZ[band] if slot is None else slot_frequency(band, slot, codec)


def label(slot):
    if isinstance(slot, str) and slot.startswith('hz:'):
        return f'{khz(int(slot[3:]))} kHz'
    return 'CF' if slot is None else f'slot {slot}'


def slot_line(activity):
    """`slot 1 busy: KD5LEE? ↔ AI7EQ chatting, level 3, 2 min (one side heard)`,
    `slot 1 busy (VARA link; stations not yet heard)`, `slot 11 signal`, `CF quiet`."""
    a, name = activity, label(activity.slot)
    if a.state != 'busy':
        return f'{name} {a.state}'
    if not a.calls:
        return f'{name} busy (VARA link; stations not yet heard)'
    lasted = a.at - a.since if a.since is not None else 0
    said = ', '.join(filter(None, (f'{a.who()} {a.stage}'.rstrip(), a.level and f'level {a.level}',
                                   lasted >= 60 and age(lasted))))
    return f'{name} busy: {said}' + (' (one side heard)' if a.one_side else '')


def follow_line(said, *, frequency=None):
    """`[slot 1] AI7EQ: de AI7EQ <R-10>`, `[slot 1] idle or polling`."""
    channel = label(said.slot)
    if frequency is not None and not isinstance(said.slot, str):
        channel += f' {khz(frequency)} kHz'
    if said.kind == 'coverage':
        return f'[{channel}] coverage: {said.text}'
    station = f'{said.station} (signed): ' if said.station else 'station unverified: '
    return f'[{channel}] ' + station + said.text


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


@dataclass
class Heard:
    """One `CQFRAME CALL-SLOT BW` report, or one advanced beacon off the KISS port. A
    beacon (-9, or advanced) and a CQ with no slot suffix are answered on the calling
    frequency, where their caller sits; the advanced beacon is narrow."""
    n: int
    call: str
    slot: int | None
    bw: int
    sn: float | None
    at: float
    beacon: bool = False
    grid: str = ''
    status: str = ''
    via: str = ''
    band: int | None = None

    @classmethod
    def parse(cls, n, report, sn, at, codec=DEFAULT_CODEC):
        _, call, bw = report.split()
        try:
            base, slot = codec.decode(call)
        except ValueError:
            return cls(n, base_call(call), None, int(bw), sn, at)
        return cls(n, base, slot, int(bw), sn, at, beacon=slot is None)

    @classmethod
    def advanced(cls, n, text, sn, at):
        match = BEACON_TEXT.fullmatch(text)
        if match is None:
            raise ValueError(f'not a VarAC beacon: {text!r}')
        status, grid, call = match.groups()
        return cls(n, call, None, 500, sn, at, beacon=True, grid=grid.upper(), status=status)


VMAIL = re.compile(r'(?:(?:<QSYR>|<TL>)\r\n)?<SM>((?:<(?:TO|FRM|TME|SBJ):[^<>\r\n]*>)*)<MSG:([^<>]*)>', re.S)


@dataclass(frozen=True)
class ApplicationMessage:
    parts: tuple
    envelope: str | None = None


def decode_application(text):
    if text.startswith('<SM>'):
        return ApplicationMessage((), text)
    prefix, separator, envelope = text.partition('\r\n<SM>')
    if separator:
        parts = tuple(TextParser().feed(prefix, final=True))
        report = bool(parts and re.fullmatch(r'de [A-Za-z0-9/-]+\s*', parts[0].raw)
                      and any(part.tag == 'R' for part in parts))
        if any(part.tag for part in parts) and all(
                part.tag or not part.raw.strip() or report and i == 0
                for i, part in enumerate(parts)):
            return ApplicationMessage(parts, '<SM>' + envelope)
    if '<SM>' in text:
        return ApplicationMessage((Text(text),))
    return ApplicationMessage(tuple(TextParser().feed(text, final=True)))


@dataclass(frozen=True)
class VMail:
    """Direct and forwarded VMail, including optional TO/FRM/TME fields.

    P1 session 021 omits the closing '>' from its length. The captured 2026-10-24
    relay includes QSYR and the closing '>' in one counted application message.
    Manual v16.1.8 pp112,125 documents the optional compact-envelope fields.
    """
    subject: str
    body: str
    sender: str | None = None
    recipient: str | None = None
    composed: str | None = None

    @classmethod
    def parse(cls, text):
        match = VMAIL.fullmatch(text)
        if match is None:
            return None
        fields = {}
        for name, value in re.findall(r'<(TO|FRM|TME|SBJ):([^<>\r\n]*)>', match[1]):
            if name in fields or (name != 'SBJ' and not value):
                return None
            fields[name] = value
        # Address fields may contain an email address (manual p125). Keep the
        # original author distinct from the connected station carrying the mail.
        for name in ('TO', 'FRM'):
            if name in fields and (len(fields[name]) > 254 or
                                   any(c.isspace() or not c.isprintable() for c in fields[name])):
                return None
        if 'TME' in fields:
            try:
                datetime.strptime(fields['TME'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                return None
        return cls(fields.get('SBJ', ''), match[2], fields.get('FRM'),
                   fields.get('TO'), fields.get('TME'))

    def wire(self):
        text = '<SM>'
        for name, value in (('TME', self.composed), ('FRM', self.sender), ('TO', self.recipient)):
            if value is not None:
                text += f'<{name}:{value}>'
        text += f'<SBJ:{self.subject}><MSG:{self.body}>'
        if self.parse(text) != self:
            raise ValueError('Invalid VMail fields; no evidenced escaping for protocol brackets')
        return text


@dataclass(frozen=True)
class BBSFile:
    name: str
    created: str
    size: int


def bbs_files(text):
    """Public manual V16.1.8 p65: one BL envelope, newline-separated triples.

    Preserve raw replies in the session log; accept only the documented rows.
    Empty-list wire semantics and filename escaping are not yet established.
    """
    if len(text.encode('utf-8')) > 65536:
        raise ValueError('BBS directory exceeds 64 KiB limit; raw reply retained in session log')
    if not (text.startswith('<BL:') and text.endswith('>')):
        raise ValueError('Unrecognized BBS directory envelope; raw reply retained in session log')
    rows = text[4:-1].splitlines()
    if rows and not rows[-1]:
        rows.pop()  # Manual p65 puts the closing bracket on its own line.
    if not rows or len(rows) > 500:
        raise ValueError('Empty or oversized BBS directory; raw reply retained in session log')
    files = []
    for row in rows:
        parts = row.split('|')
        if (len(parts) != 3 or not parts[0] or any(ord(c) < 32 for c in parts[0])
                or any(c in parts[0] for c in '<>')
                or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', parts[1])
                or not re.fullmatch(r'[0-9]{1,12}', parts[2])):
            raise ValueError('Unrecognized BBS directory row; raw reply retained in session log')
        files.append(BBSFile(parts[0], parts[1], int(parts[2])))
    return files


class Mailbox:
    """One JSON file per VMail under `root`, listed by an append-only index.jsonl. A record
    names from, to and the station it crossed the air with (`via`), so a relayed VMail
    (REL V8.2.0: compose time and the relaying station) fits the same file."""

    def __init__(self, root):
        self.root = Path(root)
        self._records = {}
        self._dedup = {}

    @staticmethod
    def _entry(record):
        keys = ('id', 'utc', 'direction', 'from', 'to', 'subject')
        if (not isinstance(record, dict) or
                any(not isinstance(record.get(key), str) for key in keys) or
                record['direction'] not in ('in', 'out') or
                Path(record['id']).name != record['id'] or
                record['id'] in ('', '.', '..')):
            raise ValueError('invalid mailbox record; existing files preserved')
        return {key: record[key] for key in keys}

    def _append_index(self, records):
        with (self.root / 'index.jsonl').open('a', encoding='utf-8') as index:
            for record in records:
                index.write(json.dumps(self._entry(record), ensure_ascii=False) + '\n')
            index.flush()
            os.fsync(index.fileno())
        mail_storage.sync_directory(self.root)

    def store(self, direction, mycall, peer, vmail, **detail):
        from .mail_storage import durable_mkdir
        self.entries()
        now = datetime.now(timezone.utc)
        sender, to = (peer, mycall) if direction == 'in' else (mycall, peer)
        record = {'id': f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}-{sender.replace('/', '_')}",
                  'utc': now.isoformat(timespec='milliseconds'), 'direction': direction,
                  'from': sender, 'to': to, 'via': peer, 'subject': vmail.subject, 'body': vmail.body, **detail}
        self._entry(record)
        durable_mkdir(self.root)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root,
                                         prefix='.vmail-', delete=False) as out:
            temporary = Path(out.name)
            try:
                json.dump(record, out, ensure_ascii=False, indent=1)
                out.write('\n')
                out.flush()
                os.fsync(out.fileno())
                os.replace(temporary, self.root / (record['id'] + '.json'))
                mail_storage.sync_directory(self.root)
            finally:
                temporary.unlink(missing_ok=True)
        self._append_index([record])
        return record

    def entries(self):
        path = self.root / 'index.jsonl'
        if not self.root.exists():
            return []
        entries, offset, repair = [], 0, False
        if path.exists():
            data = path.read_bytes()
            lines = data.splitlines(keepends=True)
            for n, line in enumerate(lines):
                try:
                    entry = self._entry(json.loads(line))
                except (ValueError, UnicodeDecodeError):
                    if n != len(lines) - 1 or line.endswith(b'\n'):
                        raise ValueError('invalid mailbox index; existing file preserved') from None
                    repair = True
                    break
                entries.append(entry)
                offset += len(line)
            if repair or (data and not data.endswith(b'\n')):
                with path.open('r+b') as index:
                    index.truncate(offset)
                    index.seek(offset)
                    if entries and not data[:offset].endswith(b'\n'):
                        index.write(b'\n')
                    index.flush()
                    os.fsync(index.fileno())
        known = {entry['id'] for entry in entries}
        if len(known) != len(entries):
            raise ValueError('duplicate mailbox index IDs; existing file preserved')
        recovered, present = [], set()
        for record_path in self.root.glob('*.json'):
            if not re.fullmatch(r'\d{8}T\d{12}Z-.+\.json', record_path.name):
                continue
            stat = record_path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
            present.add(record_path.stem)
            cached = self._records.get(record_path.stem)
            if cached is not None and cached[0] == signature:
                record = cached[1]
            else:
                record = json.loads(record_path.read_text(encoding='utf-8'))
                self._entry(record)
                if (record_path.stem != record['id'] or
                        not isinstance(record.get('body'), str)):
                    raise ValueError('invalid mailbox record; existing file preserved')
                self._records[record['id']] = (signature, record)
            if record['id'] not in known:
                # A record replace can commit before its index append fails.
                with record_path.open('rb') as source:
                    os.fsync(source.fileno())
                recovered.append(record)
        if known - present:
            raise OSError('indexed mailbox record missing; custody cannot be verified')
        recovered.sort(key=lambda record: (record['utc'], record['id']))
        if recovered or path.exists():
            self._append_index(recovered)
        entries += [self._entry(record) for record in recovered]
        self._dedup = {}
        for n, entry in enumerate(entries, 1):
            cached = self._records.get(entry['id'])
            if cached is not None:
                row = cached[1]
                if row['direction'] == 'in' and row.get('composed_utc'):
                    key = (row['to'], row['from'], row['composed_utc'], row['subject'], row['body'])
                    self._dedup.setdefault(key, n)
        return entries

    def duplicate(self, owner, sender, mail):
        self.entries()
        return self._dedup.get((owner, sender, mail.composed, mail.subject, mail.body))

    def read(self, n):
        entries = self.entries()
        if not 1 <= n <= len(entries):
            raise ValueError('no VMails yet' if not entries else f'/vmail N: N is 1 to {len(entries)}')
        return json.loads((self.root / (entries[n - 1]['id'] + '.json')).read_text())


OUTGOING = {'cq': 'a CQ', 'call': 'a call', 'answer': 'an answer', 'beacon': 'a beacon'}


class Session:
    """One attended session: monitor the calling frequency, answer a heard CQ, call or
    call CQ, chat, and return. `say` is the operator's screen, `emit` the desk's
    events.jsonl, `arm` the keying line and `line_state` its readback. `caller` names
    the station behind an incoming attempt that never linked, when the host read it.
    `power` sets the rig's RF power in watts when given them and returns what it reads
    back, through the session's own CAT; `power_clamp_w` is the station's ceiling on it.
    `slots` is the receive-only channel watch (`contact.slots.SlotWatch`), if any.
    `band_tuner` takes the rig to another band's calling frequency and reads its setup
    back; the flow's own tuner, which moves the dial alone, by default."""

    COMMANDS = ('/status', '/help [bbs|email]', '/answer N', '/cq [SLOT] [repeat]', '/call CALL [SLOT|cf]', '/stop (or Enter)', '/heard', '/slots',
                '/follow all|cf|SLOT|KHZ|off',
                '/tag TAG [value]', '/qsy SLOT', '/bye [text]', '/clear', '/auto on|off',
                '/beacon [MINUTES|now|force|off]', '/beacon hop BANDS [every MINUTES]',
                '/beacon hop auto [BANDS] [every MINUTES]',
                '/beacon hop status', '/band N', '/band ready|unready', '/away [MESSAGE]', '/back', '/vmail [N | send SUBJECT | send CALL SUBJECT (offline) | park CALL SUBJECT | parked [N] | cancel N | parking on|off | relay on|off | notify CALL | pending | collect CALL | dismiss CALL]', '/email send TO_EMAIL REPLY_EMAIL SUBJECT',
                '/bbs list|get N|request N|capture', '/inquire heard', '/pskreporter [status|off]', '/all [cf|slot N|TEXT|status|off]', '/verbose [on|off]', '/power [WATTS]', '/quit')

    def __init__(self, client: Client, flow: Flow, *, say, mailbox: Mailbox, emit=lambda kind, **record: None,
                 arm=lambda on: None, line_state=lambda: None, caller=lambda: None,
                 bandwidth=DEFAULT_BW, wait_seconds=SLOT_WAIT_S, slots=None, band_tuner=None, reporter=None, bbs_catalog=None,
                 wall_clock=time.time, bbs_core=None, bbs_say=None, bbs_reload=None,
                 identify=lambda bandwidth, after_peer: None, idcall=None, ident_report=None,
                 power=None, power_clamp_w=None):
        self.client, self.flow, self.mailbox = client, flow, mailbox
        self.power, self.power_clamp_w = power, power_clamp_w
        self._power_owed = None
        self.band_tuner = band_tuner or flow.tuner
        self._band_fault = False
        self.band_setup = None
        self.slots = slots
        self.reporter = reporter
        self._shown = {}
        self._transcripts = {}
        self.following = None
        self._follow_tuned = False
        self._follow_hint_shown = False
        self.say, self.emit, self.arm, self.line_state = say, emit, arm, line_state
        self.ident_report = ident_report or say
        self.idcall = idcall or client.mycall
        self.keyer = identify
        self.owed = self._after_peer = False
        self._identifying = False
        self.caller = caller
        self.bandwidth, self.wait_seconds = bandwidth, wait_seconds
        self.heard: list[Heard] = []
        self.auto = True
        self.verbose = False
        self._audio_lost = {}
        self._coverage_noticed = float('-inf')
        self._bbs_waiting = False
        self._peer_parse_fault = None
        self._service_hints = set()
        self._opening = False
        self._notice_phase = None
        self._notice_data = None
        self.away = None
        self._away_owed = None
        self._bbs_list_queued = None
        self.new_vmails = 0
        self.compose = None
        self._compose_link = None
        from .email_gateway import EmailClient
        self.email = EmailClient(self)
        from .bbs_client import BBSClient
        self.files = BBSClient(self)
        self.client.raw_transfer = self.files
        from .bbs_server import BBSServer
        self.bbs_host = BBSServer(self, bbs_catalog)
        from .broadcast_client import BroadcastClient
        self.broadcast = BroadcastClient(self)
        from .mail_collection import MailCollection
        self.mail_collection = MailCollection(self)
        self._linked_at = None
        self.peer = None
        self.intent = None
        self.slot = None
        self.contact = None
        self.contacts = 0
        self.steps = None
        self.repeat = False
        self._hint = None
        self._called = False
        self.beacon = None
        self._beacon_due = None
        self._beacon_expires = None
        self._beacon_last = None
        self._beacon_now = False
        self._beacon_generation = 0
        self.hopping = BeaconHop(CALLING_HZ, client.clock, wall_clock)
        self._clear_since = None
        self._busy_seen = 0
        self._attempt_at = None
        self._closing = False
        self._quitting = False
        self._force_close = False
        self._close_failed = None
        from ..bbs.frontends.varac import VarACEntrance
        self.bbs = VarACEntrance(self, bbs_core, say=bbs_say, reload=bbs_reload)
        from .mail_ack import MailAckOwner
        from .mail_parking import MailParking
        self.mail_ack = MailAckOwner(self)
        self.parking = MailParking(self, wall_clock=wall_clock)
        self._mail_replies = []
        self.flow.mail_ready = self._mail_ready
        self._inbox_contact_keys = set()

    @property
    def calling_hz(self):
        return CALLING_HZ[self.flow.band]

    @property
    def _vmail_sent(self):
        return self.mail_ack.busy and self.mail_ack.kind in ('direct', 'parking')

    @property
    def _auto_answer(self):
        return (self.auto and not self._closing and not self._quitting
                and (not self.hopping.active or self.hopping.ready(self.flow.band)))

    def where(self, entry):
        band = entry.band or self.flow.band
        return channel_frequency(band, entry.slot, self.flow.codec) if entry.slot else CALLING_HZ[band]

    def listing(self, entry, *, received_unix=None):
        place = 'beacon  ' if entry.beacon else f'slot {entry.slot:<3}' if entry.slot else 'no slot '
        sn = f'SN {entry.sn:+5.1f}' if entry.sn is not None else 'SN   -- '
        when = (age(self.client.clock() - entry.at) if received_unix is None else
                datetime.fromtimestamp(received_unix, timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ'))
        return (f'{entry.n:>3}  {entry.call:<8} {place} {khz(self.where(entry))} kHz  '
                f'{entry.bw:>4} Hz  {sn}  {when}'
                + (f'  {entry.grid} {services(entry.status)}' if entry.grid else '')
                + (f'  via {entry.via}' if entry.via else '')
                + (f'  {entry.band} m' if entry.band not in (None, self.flow.band) else ''))

    # -- what the host says ------------------------------------------------------

    def handle(self, event: Event):
        kind, raw = event.kind, event.raw
        if self.verbose:
            self._verbose_event(event)
        if kind == 'peer_protocol_error':
            detail = json.loads(raw)
            self.emit('peer_protocol_error', peer=self.peer, **detail)
            self._peer_parse_fault = (detail['generation'], detail['reason'])
            return
        if kind == 'host':
            words = raw.split()
            if raw == 'PTT ON':
                self.owed = True
            if (self._closing or self._quitting) and words[:1] in (['PENDING'], ['CONNECTED']):
                self._force_close = True
                try:
                    self.client.send_command('LISTEN OFF')
                finally:
                    self.client.send_command('ABORT')
                return
            if words[:1] == ['CQFRAME'] and len(words) == 3:
                self._heard(raw)
            elif raw == 'PENDING' and not self.client.calling and self._attempt_at is None:
                self._attempt_at = self.client.clock()
            elif raw == 'CANCELPENDING' and self._attempt_at is not None:
                self._attempted()
            elif words[:1] == ['CONNECTED']:
                self.parking.invalidate_notices()
                self.broadcast.selected = None
                self._attempt_at = None
                mine = callsign(self.client.mycall)
                previous = self.peer if self.contact is not None else None
                self.peer = next((call for call in words[1:3] if callsign(call) != mine), previous)
                self._begin()
            elif raw == 'DISCONNECTED':
                self.bbs.ended()
        elif kind == 'kiss':
            self._heard(raw, advanced=True)
        elif kind == 'broadcast':
            from .broadcast import decode_ui
            envelope = json.loads(raw)
            message = decode_ui(bytes.fromhex(envelope['hex']), command=envelope['command'])
            self.broadcast.received(message, dial=self.flow.dial)
        elif kind == 'broadcast_error':
            self.emit('broadcast_error', detail=raw)
        elif kind == 'message':
            if self._closing or self._quitting:
                self.emit('rx', text=raw)
                return
            application = decode_application(raw)
            acknowledged = self.mail_ack.message(
                application, event.sequence if event.sequence is not None else self.client.message_sequence,
                event.generation if event.generation is not None else self.client.link_generation)
            if application.envelope is not None:
                reply = self.client.send_tag
                if (self.flow.qsy_receiving
                        and self.flow.qsy_phase == 'waiting for peer arrival (<TL>)'
                        and any(part.tag == 'TL' for part in application.parts)):
                    reply = lambda tag: self._mail_replies.append((self.client.link_generation, tag))
                self.emit('rx', text=raw)
                if (vmail := VMail.parse(application.envelope)) is not None:
                    if self.bbs.owned or self.files.active or self.bbs_host.active:
                        reply('SMF' if self.parking.parking_enabled else 'SMFP')
                    else:
                        self._take(vmail, reply=reply)
                else:
                    self.say('Unrecognized VMail envelope retained in session log; not acknowledged')
                return
            if any(part.tag == 'VRP' for part in application.parts):
                if self.client.connected:
                    self.client.send_tag('VRPJ')
                return
            if self.bbs.message(raw):
                return
            if self.bbs_host.message(raw):
                return
            if self.files.message(raw):
                return
            self.emit('rx', text=raw)
            lines = raw.replace('\r\n', '\n').rstrip('\n').split('\n')
            self.say(f'{"RX " if self.verbose else ""}{self.peer}: {lines[0]}')
            for line in lines[1:]:
                self.say(' ' * (len(self.peer) + 2) + line)
            if acknowledged and self.mail_ack.kind == 'direct' and self.mail_ack.result == 'SMR':
                self.say(f'{self.peer} acknowledged the VMail')
        elif kind == 'incomplete' and self._bbs_waiting:
            self.emit('bbs_incomplete', peer=self.peer, dial_hz=self.flow.dial, text=raw)
            self.say('Incomplete BBS reply retained in session log')
        elif kind == 'tag':
            self.emit('tag', text=raw)
            if not event.diagnostic:
                self._service_hint(raw)
        elif kind == 'sent':
            self.emit('tx', text=raw, status='queued')
        elif kind == 'write_complete':
            if json.loads(raw)['disposition'] != 'complete':
                self.mail_ack.interrupt('writer did not confirm complete admission')
            self.bbs.writer_completion(raw)
        elif kind == 'held':
            self.say(f'{"beacon" if self.intent == "beacon" else raw} held: channel busy')
        elif kind == 'cq':
            self._called = True
            self.say(f'calling CQ {raw} on {khz(self.calling_hz)} kHz, slot {self.slot} at '
                     f'{self.client.bandwidth} Hz')
        elif kind == 'state':
            self._state(State(raw))
        elif kind == 'tune':
            self.parking.invalidate_notices()
            if self.slots is not None:
                self.slots.retune(int(raw))

    def _verbose_event(self, event):
        kind, raw = event.kind, event.raw
        if kind == 'sent':
            preview = raw.replace('\r\n', '\n').replace('\n', ' ↵ ')
            if len(preview) > 160:
                preview = preview[:157] + '…'
            self.say(f'TX queued (not yet acknowledged): {preview}')
        elif kind == 'host':
            if raw == 'BUSY ON':
                self.say('RX: channel busy; no frame identified by BUSY alone')
            elif raw == 'BUSY OFF':
                self.say('RX: busy detector clear')
            elif raw == 'PTT OFF':
                self.say('TX ended: PTT OFF')
            elif raw.startswith('BUFFER '):
                self.say('TX buffer empty' if raw == 'BUFFER 0' else
                         f'TX buffer: {raw[7:]} bytes pending')
            elif raw.startswith('BITRATE '):
                self.say(f'Modem: {raw}')
            elif raw.startswith('SN '):
                self.say(f'RX signal report: {raw[3:]} dB')
            elif raw == 'PENDING':
                self.say('RX: incoming connection setup detected')
            elif raw == 'CANCELPENDING':
                self.say('Modem: pending operation ended')
        elif kind == 'command':
            if raw.startswith('CONNECT '):
                self.say(f'Link requested: {raw[8:]}')
            elif raw in ('CLEANTXBUFFER', 'DISCONNECT', 'ABORT'):
                self.say({'CLEANTXBUFFER': 'Close: asking modem to discard queued text',
                          'DISCONNECT': 'Close: requesting disconnect exchange',
                          'ABORT': 'Close: aborting; waiting for confirmation and unkey'}[raw])

    def diagnostic(self, kind, **details):
        """Actual runtime observations, delivered on the terminal thread."""
        if kind in ('link_recovery', 'capture_rate'):
            self.say(details['text'])
            self.emit(kind, **details)
            return
        if kind in ('acknowledged', 'discarded'):
            self.emit('tx', status=kind, bytes=details['bytes'])
        if not self.verbose:
            return
        if kind == 'rx_protocol':
            # These are the decoder's own accepted-frame logs, not names inferred
            # from carrier activity or a pending application write.
            self.say('RX: ' + details['text'])
        elif kind in ('tx_start', 'tx_end'):
            frame = ('CW identification' if details.get('identification') else
                     str(details.get('packet_kind', 'audio')).removeprefix('session-').replace('-', ' '))
            facts = []
            if details.get('over_id') is not None:
                facts.append(f'over {details["over_id"]}')
            if details.get('speed_level') is not None:
                facts.append(f'level {details["speed_level"]}')
            if details.get('retry'):
                facts.append('retry')
            suffix = f' ({", ".join(facts)})' if facts else ''
            if kind == 'tx_start':
                verb = 'sending' if details.get('keyed') else 'playing audio; key not confirmed'
            else:
                verb = {'sent': 'sent', 'refused': 'refused; not sent',
                        'error': 'failed; transmission may be incomplete',
                        'dry_run': 'audio finished; no armed transmitter',
                        'unconfirmed': 'finished; key not confirmed'}[details['outcome']]
            self.say(f'TX {verb}: {frame}{suffix}')
        elif kind == 'acknowledged':
            self.say(f'TX acknowledged by peer modem: {details["bytes"]} original bytes')
        elif kind == 'discarded':
            self.say(f'TX discarded without acknowledgement: {details["bytes"]} original bytes')
        elif kind == 'dropped':
            self.say(f'Verbose: {details["count"]} runtime observations omitted while the terminal was busy')

    def _attempted(self):
        """PENDING ended without CONNECTED: somebody's connect request was answered and
        the link never came up (10-23 11:13Z, a weak caller's setup unread)."""
        caller, seconds, self._attempt_at = self.caller(), self.client.clock() - self._attempt_at, None
        self.say(f'\a{caller or "a station"} called you; link setup not read')
        self.emit('attempt', caller=caller, called=self.client.mycall, dial_hz=self.flow.dial,
                  bandwidth_hz=self.client.bandwidth, seconds=round(seconds, 1))

    def _heard(self, report, advanced=False, *, sn=None, via='', slot=None):
        """`via` names the monitored channel a frame was heard on off our own; a
        beacon's caller sits on that channel, a CQ's on the slot its suffix names."""
        n, now = len(self.heard) + 1, self.client.clock()
        try:
            entry = (Heard.advanced(n, report, sn if via else self.client.snr, now) if advanced
                     else Heard.parse(n, report, sn if via else self.client.snr, now, self.flow.codec))
        except ValueError:
            return
        entry.via, entry.band = via, self.flow.band
        if entry.beacon and via:
            entry.slot = slot
        self.heard.append(entry)
        if self.slots is not None:
            self.slots.heard(entry.call, None if entry.beacon else entry.slot)
        received_unix = int(time.time())
        self.hopping.observe(entry.band, entry.call, 'beacon' if entry.beacon else 'cq')
        self.say(self.listing(entry, received_unix=received_unix))
        rx_dial = (CALLING_HZ[self.flow.band] if slot is None
                   else channel_frequency(self.flow.band, slot, self.flow.codec)) if via else self.flow.dial
        self.emit('heard', rf_source='beacon' if entry.beacon else 'cq',
                  rx_frequency_hz=rx_dial + 1500, received_unix=received_unix, n=entry.n, call=entry.call, slot=entry.slot, beacon=entry.beacon,
                  band_m=entry.band, dial_hz=self.where(entry), bandwidth_hz=entry.bw, sn=entry.sn,
                  **(dict(grid=entry.grid, status=entry.status) if entry.grid else {}))
        if entry.beacon:
            self.parking.heard(entry.call, rx_dial)

    def _state(self, state):
        self.emit('state', state=state.value, dial_hz=self.flow.dial)
        bw = self.client.bandwidth
        said = {
            State.SNIFFING: f'sniffing slot {self.slot} ({khz(self.flow.dial)} kHz) for {SNIFF_S} s',
            State.WAITING: f'waiting on slot {self.slot} ({khz(self.flow.dial)} kHz) at {bw} Hz '
                           f'for {self.wait_seconds} s',
            State.CONNECTING: f'connecting {self.peer} on {khz(self.flow.dial)} kHz at {bw} Hz',
            State.CONNECTED: f'connected {self.peer}; Enter sends, /bye ends',
            State.MONITORING: f'monitoring {khz(self.calling_hz)} kHz at {bw} Hz, {self._answering()}'
                              + (f', beacon every {self.beacon} min' if self.beacon else '')
                              + ''.join(f'; {slot_line(a)}' for a in self._watched() if a.state == 'busy'),
        }.get(state)
        if state is State.CONNECTED and self.intent not in ('answer', 'call') and not self.away:
            said = '\a' + said
        if said:
            self.say(said)

    def _begin(self):
        if self.contact is not None or self._closing or self._quitting:
            return
        if not self._auto_answer and self.intent is None:
            self._force_close = True
            self._run(self._farewell())
            return
        self.bbs_host.reset()
        self.bbs_host.advertise()
        self.contacts += 1
        self._service_hints.clear()
        self.contact = dict(n=self.contacts, peer=self.peer, how=self.intent or 'incoming',
                            dial_hz=self.flow.dial, bandwidth_hz=self.client.bandwidth,
                            started_utc=utc())
        self.emit('link', event='connected', peer=self.peer, how=self.contact['how'])
        self.emit('contact', **self.contact, ended_utc=None, line=None)
        self._linked_at = self.client.clock()
        self.mail_ack.linked()
        self._inbox_contact_keys.clear()
        self.bbs.linked()
        for seq, text in self.client.messages:
            if seq > self.client.link_message_start:
                for part in decode_application(text).parts:
                    if part.tag:
                        self._service_hint(part.raw)
        if self.intent is None:
            self.flow._state(State.CONNECTED)
            if self.away and self.flow.dial == self.calling_hz:
                # ON2AD V16.1.8 p126: with auto-QSY only <AWQ> or <Q> crosses the
                # calling frequency, as the measured `<AWQ>\r\n<VW>` does. The
                # away message waits for the caller's QSY.
                self.client.send_tag('AWQ')
                self._away_owed = self.client.link_generation
                self.say(f'away: <AWQ> sent to {self.peer}; the away message follows the QSY')
            elif self.away:
                self._greet_away()
            elif self.flow.dial == self.calling_hz:
                self.client.send_tag('Q')
            if self.flow.dial in CALLING_HZ.values():
                if self.parking.pending():
                    self.client.send_tag('VW')
            else:
                self.parking.start_delivery()

    def _greet_away(self):
        self._away_owed = None
        self.client.send_text(emit_tag('AWAY') + '\r\n' + self.away)
        self.say(f'away greeting sent to {self.peer}')

    def _opened_here(self):
        """The peer sent its opening report on the current frequency: a caller
        without auto-QSY exchanges it on the calling frequency (ON2AD V16.1.8 p126)."""
        return any(text.startswith('de ') and any(part.tag == 'R' for part in TextParser().feed(text, final=True))
                   for seq, text in self.client.messages if seq > self.client.link_message_start)

    def _settled_tick(self):
        if (self.contact is None or not self.client.connected or self.steps is not None or self._closing
                or self.files.active or self.bbs_host.active or self.bbs.active):
            return
        generation = self.client.link_generation
        if self._away_owed is not None:
            if self._away_owed != generation or not self.away:
                self._away_owed = None
            elif self.flow.dial != self.calling_hz or self._opened_here():
                self._greet_away()
        if self._bbs_list_queued is not None:
            if self._bbs_list_queued != generation:
                self._bbs_list_queued = None
            elif (self.flow.dial not in CALLING_HZ.values() and self.flow.state is State.CONNECTED
                  and not self.mail_ack.busy and not self.parking.active):
                self._bbs_list_queued = None
                self.list_bbs()

    def _mail_ready(self):
        # C4: QSYR precedes the receipt SMR for a TL/SM aggregate.
        replies, self._mail_replies = self._mail_replies, []
        for generation, tag in replies:
            if generation == self.client.link_generation:
                self.client.send_tag(tag)
        self.parking.start_delivery()

    def _take(self, vmail, *, reply=None):
        if not self.client.connected or self.peer is None:
            return
        reply = reply or self.client.send_tag
        if vmail.recipient is not None:
            try:
                recipient = canonical_call(vmail.recipient)
                mine = canonical_call(self.client.mycall)
            except ValueError:
                reply('SMF')
                return
            if recipient != mine:
                self.parking.deposit(vmail, reply=reply)
                return
        detail = {}
        if vmail.sender is not None:
            detail['from'] = vmail.sender
        if vmail.composed is not None:
            detail['composed_utc'] = vmail.composed
        try:
            # A relay can repeat a stored message after losing our SMR. Only
            # deduplicate envelopes with both original author and compose time.
            duplicate = None
            contact_key = (vmail.sender or self.peer, vmail.composed, vmail.subject, vmail.body)
            if contact_key in self._inbox_contact_keys:
                reply('SMR')
                return
            if vmail.composed:
                duplicate = self.mailbox.duplicate(self.client.mycall, vmail.sender or self.peer, vmail)
            if duplicate is not None:
                reply('SMR')
                self.say(f'VMail {duplicate} already saved; acknowledged repeated relay from {self.peer}')
                return
            record = self.mailbox.store('in', self.client.mycall, self.peer, vmail,
                                        dial_hz=self.flow.dial, snr=self.client.snr, **detail)
            n = len(self.mailbox.entries())
            self._inbox_contact_keys.add(contact_key)
        except (OSError, ValueError, KeyError) as error:
            self.say(f'VMail could not be saved; custody refused (SMF): {error}')
            reply('SMF')
            return
        via = f' via {self.peer}' if vmail.sender else ''
        self.say(f'VMail {n} from {record["from"]}{via}: {vmail.subject or "(no subject)"} '
                 f'({len(vmail.body)} characters); /vmail {n} reads it')
        self.emit('vmail', direction='in', peer=self.peer, id=record['id'], subject=vmail.subject)
        reply('SMR')
        self.mail_collection.received()
        if self.away:
            self.new_vmails += 1

    def _leaving(self):
        """The acknowledgement is delivered, then the link closes as the recorded farewells
        close it; a peer that hung up first has already closed it."""
        self._closing = True
        try:
            try:
                yield from self.client.drain_steps(timeout=15, cancelled=lambda: self._quitting)
            except TimeoutError:
                pass
            yield from self.flow.farewell_steps(cancelled=lambda: self._quitting)
        except HostError:
            if self.client.connected:
                raise

    def _end(self):
        self.mail_ack.interrupt('disconnected before acknowledgement')
        self.bbs.ended()
        contact, self.contact = self.contact, None
        self._after_peer = not self._closing
        verdict = {True: 'keying line reads HIGH; check the transmitter',
                   False: 'keying line reads LOW'}.get(self.line_state())
        if self.client.closed:
            self.say(f'Host connection lost; local contact with {contact["peer"]} ended; '
                     'peer disconnect and final unkey are unconfirmed')
            self.emit('link', event='host_lost', peer=contact['peer'])
            verdict = (verdict + ' (observed before runtime shutdown)') if verdict else None
        else:
            self.say(f'{contact["peer"]} disconnected' + (f'; {verdict}' if verdict else ''))
            self.emit('link', event='disconnected', peer=contact['peer'])
        self.emit('contact', **contact, ended_utc=utc(), line=verdict)

    # -- the loop ------------------------------------------------------------------

    @property
    def idle(self):
        return not (self.client.connected or self.client.pending or self.client.ptt)

    def tick(self, timeout=.1):
        self.client.poll(min(timeout, self._hint) if self.steps and self._hint else timeout)
        self.bbs.tick()
        self.files.tick()
        self.bbs_host.tick()
        self.mail_ack.tick()
        if (self.away and not self.bbs.owned and not self._closing and self.contact is not None and self.client.connected
                and self.client.clock() - self._linked_at >= AWAY_LINK_S):
            self.say(f'away link at {AWAY_LINK_S // 60} min; closing')
            self.mail_ack.interrupt('absolute away deadline')
            if self.steps is not None:
                self.steps.close()
                self.steps = None
            self._run(self._leaving())
        if self._peer_parse_fault is not None:
            (generation, reason), self._peer_parse_fault = self._peer_parse_fault, None
            if generation == self.client.link_generation:
                self._error(HostError(f'received application framing failed: {reason}; '
                                      'closing only this contact'))
                if self.steps is not None:
                    self.steps.close()
                    self.steps = None
                self._run(self._close_malformed_contact())
        try:
            if not (self._closing or self._quitting or self.files.active or self.bbs_host.active):
                invitation = self.flow.receive()
                if invitation is not None:
                    self._receive_qsy(invitation)
        except (ValueError, HostError, TimeoutError) as error:
            self._error(error)
        if self.steps is not None:
            self._step()
        try:
            self._settled_tick()
        except (ValueError, HostError, TimeoutError) as error:
            self._error(error)
        if self.contact is not None and not self.client.connected:
            self._end()
        # A link that drops with the key down returns once the frame in the air ends.
        if (self.steps is None and self.contact is None and self.idle
                and self.line_state() is not True
                and self.flow.state in (State.FAILED, State.RETURNED, State.CONNECTED)):
            self._return()
        self._slots_tick()
        self._beacon_tick()
        self.parking.tick()
        self._power_tick()
        self._operator_notice()

    def _receive_qsy(self, invitation):
        self.bbs.detach('peer QSY')
        text, after, generation, origin = invitation
        self.flow.received_qsy_target(text, origin)
        self.mail_ack.interrupt('QSY interrupted the mail exchange')
        if self.steps is not None:
            readiness = self.flow.qsy_phase in (
                'waiting for peer auto-QSY readiness (<AWQ> or <Q>)',
                'waiting for local delivery')
            acceptance = self.flow.qsy_phase == 'waiting for peer acceptance (<QSYR>)'
            sent_after = self.client._sent_after
            drained = (not self.client.ptt and not self.client.pending
                       and (sent_after is None or (
                           self.client.buffer == 0
                           and 'BUFFER 0' in self.client.since(sent_after))))
            if self.flow.qsy_receiving or not (readiness or (acceptance and drained)):
                raise HostError('Received QSY not started: another exchange is in progress')
            self.steps.close()
            self.steps = None
            self.flow._state(State.CONNECTED)
            self.say('Peer QSY supersedes the previous QSY wait')
        self._run(self.flow.answer_qsy_steps(
            text, after=after, generation=generation, origin=origin))

    def _close_malformed_contact(self):
        self._closing = True
        self.repeat = False
        self.flow._state(State.FAILED)
        yield from self.flow.close_steps()
        if self.line_state() is True:
            raise HostError('received framing failed and keying line remains HIGH; cannot resume monitoring')
        self.say('Contact with rejected application framing closed; '
                 'resuming monitoring with beacon schedule preserved')

    def _slots_tick(self):
        if self.slots is None:
            return
        for kind, slot, item in self.slots.poll():
            if kind == 'error':
                self.slots = None
                self._error(HostError(f'slot monitor stopped: {item}'))
                return
            if kind == 'state':
                self._slot_changed(item)
            elif kind == 'qso':
                self._slot_said(item)
            elif kind == 'passband':
                self.emit('passband', edges={str(level): list(edge) for level, edge in item.passband.edges.items()},
                          slots=list(item.slots), wide_offset=item.wide)
                self.say(item.line())
            elif kind == 'broadcast':
                dial = self.calling_hz if slot is None else channel_frequency(self.flow.band, slot, self.flow.codec)
                self.broadcast.received(item, dial=dial, via=label(slot))
            elif kind == 'cq':
                self._heard(f'CQFRAME {item.call} {item.bandwidth}', sn=item.sn, via=label(slot), slot=slot)
            else:
                self._heard(item.text, advanced=True, sn=item.sn, via=label(slot), slot=slot)

    def _slot_changed(self, activity):
        """Said only when what the operator sees changes: busy, a signal off our own
        channel (ours lists its CQs and beacons already), quiet, or a newly named call."""
        shown = 'quiet' if activity.home and activity.state == 'signal' else activity.state
        before = self._shown.get(activity.slot)
        self._shown[activity.slot] = (shown, activity.calls)
        self.emit('slot', slot=activity.slot, state=activity.state, audio_hz=activity.audio_hz,
                  calls=list(activity.calls), level=activity.level)
        if before is None and shown == 'quiet' or before == (shown, activity.calls):
            return
        if before and before[0] == shown and not set(activity.calls) - set(before[1]):
            return
        self.say(slot_line(activity))

    def _show_follow_hint(self):
        if self.slots is not None and not self._follow_hint_shown:
            self._follow_hint_shown = True
            self.say('Passive transcripts are logged; /follow cf shows calling-frequency activity, '
                     '/follow all shows the current passband, /follow off hides it. '
                     '/follow SLOT or KHZ retunes if needed; /follow cf returns to CF.')

    def _follow_line(self, said):
        frequency = (self.calling_hz if said.slot is None else
                     channel_frequency(self.flow.band, said.slot, self.flow.codec))
        return follow_line(said, frequency=frequency)

    def _follow_selected(self, slot):
        watched = {a.slot for a in self._watched()}
        return (slot in watched and self.following is not None
                and (self.following == 'all' or slot == (None if self.following == 'cf' else self.following)))

    def _slot_said(self, said):
        """Retain bounded current-band evidence; display only currently watched channels."""
        self.emit('slot', slot=said.slot, qso=said.kind, text=said.text, station=said.station or None)
        if said.kind == 'coverage':
            self._coverage_said(said)
            return
        self._show_follow_hint()
        now = self.client.clock()
        lines = self._transcripts.setdefault(said.slot, deque(maxlen=REPLAY_LINES))
        lines.append((now, said))
        while lines and now - lines[0][0] > REPLAY_S:
            lines.popleft()
        if self._follow_selected(said.slot):
            self.say(self._follow_line(said))

    def _coverage_said(self, said):
        """Coverage notes are diagnostics, logged but not traffic: the console gets one
        line, at most every COVERAGE_NOTICE_S, when the audio lost in the last minute is
        material (or under /verbose), and whenever the watch stops or resumes reading a
        channel to keep up with the audio."""
        if said.text in (SHED, RESUMED) and self.following is not None:
            self.say(self._follow_line(said))
            return
        gap = re.match(r'passive audio gap: ([\d.]+) seconds', said.text)
        if not gap or self.following is None:
            return
        now = self.client.clock()
        lost = self._audio_lost.setdefault(said.slot, deque())
        lost.append((now, float(gap[1])))
        while lost[0][0] <= now - COVERAGE_WINDOW_S:
            lost.popleft()
        share = max(sum(s for _, s in d) for d in self._audio_lost.values() if d) / COVERAGE_WINDOW_S
        if ((self.verbose or share > COVERAGE_LOSS_SHARE)
                and now - self._coverage_noticed >= COVERAGE_NOTICE_S):
            self._coverage_noticed = now
            self.say(f'passive follower is dropping audio ({share:.0%} over the last minute); '
                     'decodes may be missed')

    def _check_follow_retune(self):
        if (self.steps is not None or self.contact is not None or self._closing or self.broadcast.active
                or not self.idle or self.line_state() is True):
            raise HostError('cannot retune to follow during a link or transmission; /stop or /bye first')
        if self._band_fault:
            raise HostError('band setup unverified; /band N must succeed first')

    def follow(self, slot):
        if slot is None:
            if self._follow_tuned:
                self._check_follow_retune()
                self.monitor()
            self.following = None
            self.say('follow off')
            return
        if self.slots is None:
            raise ValueError('slot monitor off')
        channel = None if slot == 'cf' else slot
        if slot != 'all':
            if isinstance(slot, str) and slot != 'cf':
                try:
                    value = float(slot)
                    if not 100 <= value <= 1000000:
                        raise ValueError
                    dial = round(value * 1000)
                except ValueError:
                    raise ValueError('/follow: use a slot number or dial frequency in kHz') from None
                channel = next((n for n in self.flow.codec.slots
                                if slot_frequency(self.flow.band, n, self.flow.codec) == dial),
                               None if dial == self.calling_hz else f'hz:{dial}')
                slot = 'cf' if channel is None else channel
            else:
                dial = channel_frequency(self.flow.band, channel, self.flow.codec)
            watched = {a.slot for a in self._watched()}
            if channel not in watched or (slot == 'cf' and self._follow_tuned):
                self._check_follow_retune()
                self.arm(False)
                self.client.request('LISTEN OFF')
                if not self.idle or self.line_state() is True:
                    raise HostError('link or keying began while disabling listen; follow cancelled')
                self._follow_tuned = True
                self.flow.tune(dial)
                if slot == 'cf':
                    self.monitor()
                else:
                    self.say(f'Monitor tuned to {khz(dial)} kHz; auto-answer and beacons suspended. '
                             'CF activity outside the passband will be missed; /follow cf or /follow off returns.')
        elif not self._watched():
            raise ValueError('no channels inside the passband')
        self.following, now = slot, self.client.clock()
        recent = sorted(((at, said) for lines in self._transcripts.values() for at, said in lines
                         if now - at <= REPLAY_S and self._follow_selected(said.slot)),
                        key=lambda item: item[0])[-REPLAY_LINES:]
        selection = 'all current passband channels' if slot == 'all' else label(channel)
        self.say(f'following {selection}; passive decoded activity only, text may be incomplete'
                 + ('' if recent else f'; nothing read in {REPLAY_S // 60} min'))
        for _, said in recent:
            self.say(self._follow_line(said))

    def _beacon_tick(self):
        now = self.client.clock()
        self.hopping.sample(self.flow.band, self._hop_idle())
        if self.beacon and now >= self._beacon_expires:
            pending = self._beacon_now
            self.set_beacon(None)
            self._beacon_now = pending
            self.say('beacon expired after 24 hours')
        force = self._beacon_now == 'force'
        quiet = (self.flow.state is State.MONITORING and not self.client.busy
                 and self.client.busy_events == self._busy_seen)
        self._busy_seen = self.client.busy_events
        self._clear_since = (now if self._clear_since is None else self._clear_since) if quiet else None
        if self.hopping.active and not self._hop_tick():
            return
        if ((self._beacon_now or self.beacon) and not (self._closing or self._quitting)
                and not self._follow_tuned
                and self.flow.state is State.MONITORING and (quiet or force)
                and self.steps is None and self.contact is None and self.idle
                and self.line_state() is not True
                and (self._beacon_now or now >= self._beacon_due)
                and (force or now - self._clear_since >= BEACON_CLEAR_S)):
            try:
                self._leave('beacon', None, self.bandwidth)
            except (ValueError, HostError, TimeoutError) as error:
                self._beacon_now = False
                self._error(error)
                return
            once = self._beacon_now
            self._beacon_now = False
            self._run(self._beaconing(once=once))

    def _error(self, error):
        self.say(f'error: {error}')
        self.emit('error', text=str(error))

    def _step(self):
        try:
            self._hint = next(self.steps)
        except StopIteration:
            self.steps = None
        except (ValueError, HostError, TimeoutError) as error:
            self.steps = None
            if self.flow.state is State.FAILED:
                self.mail_ack.interrupt('opening or QSY exchange failed')
            if self._closing:
                # A failed stop is fatal: leave the terminal so Host.close stops
                # the runtime before rig.shutdown retires the keyer.
                self._close_failed = error
                raise
            self._error(error)
        except OSError as error:
            if self._closing:
                self._close_failed = error
            raise

    def _run(self, steps):
        self.steps, self._hint = steps, None
        self._step()
        self._operator_notice()

    def stop(self):
        self.mail_ack.interrupt('operator stopped exchange')
        self.bbs.hard_stop('operator stop')
        if self._bbs_list_queued is not None:
            self._bbs_list_queued = None
            self.say('queued BBS list cancelled')
            if self.steps is None:
                return
        if self.files.active or self.bbs_host.active:
            self._force_close = True
            self.files.stop()
            return
        if self.flow.qsy_receiving and self.steps is not None:
            self.steps.close()
            self.steps = None
            self.say('Received QSY wait cancelled; remaining on the current frequency')
            return
        if self.broadcast.cancel():
            return
        if self.email.waiting and self.steps is not None:
            self.steps.close()
            self.steps = None
            return
        if (self.steps is not None and self.flow.qsy_phase is not None
                and self.flow.qsy_phase.startswith('waiting for peer auto-QSY readiness')):
            self.steps.close()
            self.steps = None
            self.say('QSY readiness wait cancelled; remaining on the calling frequency')
            return
        if self._beacon_now:
            self._beacon_now = False
            self.say('pending one-shot beacon cancelled')
            if self.steps is None:
                return
        if self._bbs_waiting and self.steps is not None:
            self.steps.close()
            self.steps = None
            self.say('BBS wait cancelled; the queued request may still be delivered')
            return
        if self.client.connected:
            if self.steps is not None:
                self.steps.close()
            self._force_close = True
            self.say('Stopping link; discarding unconfirmed text and waiting for unkey')
            self._run(self._farewell())
            return
        if self.steps is None:
            self.say('nothing to stop')
            return
        self.flow.cancel()
        self.say('stopping after this frame' if self.client.ptt or self.client.pending else 'stopping')

    def _return(self):
        try:
            self.monitor()
        except (ValueError, HostError, TimeoutError) as error:
            if self._close_failed is not None:
                raise
            self.flow.state = State.CALLING
            self.say(f'error: {error}; not monitoring. /auto on tries again')
            self.emit('error', text=str(error))

    def identification_unsent(self):
        if self.owed:
            from ..core.cwid import unsent
            self.owed = False
            unsent(self.idcall, 'session ended before identification', self.ident_report)

    def identify(self):
        owed, after_peer = self.owed, self._after_peer
        self.owed = self._after_peer = False
        if owed:
            return self.keyer(self.client.bandwidth, after_peer)

    def _identify_steps(self):
        closing = self._closing
        self._closing = self._identifying = True
        try:
            steps = self.identify()
            if steps is not None:
                yield from steps
            if self.line_state() is True:
                raise HostError('keying line remains HIGH after identification; stopping runtime')
            self._closing = closing
        finally:
            self._identifying = False

    def _identified_return(self):
        yield from self._identify_steps()
        self.monitor()

    def monitor(self):
        if self.owed:
            self._run(self._identified_return())
            return
        if self._band_fault:
            raise HostError('band setup unverified; /band N must succeed before transmitting')
        self._closing = False
        self._force_close = False
        self.broadcast.selected = None
        self.intent = self.slot = None
        self._called = self.repeat = False
        if self.flow.dial != self.calling_hz:
            self.flow.return_to_calling()
        if self.client.bandwidth != self.bandwidth:
            self.client.request('LISTEN OFF')
            self.client.set_bandwidth(self.bandwidth)
        self._follow_tuned = False
        listening = self._auto_answer
        self.arm(listening)
        if listening:
            self.client.listen()
        else:
            self.client.request('LISTEN OFF')
        self.flow._state(State.MONITORING)
        self._show_follow_hint()

    def _farewell(self, text='73'):
        if self.idle and self.line_state() is not True:
            self.say('not linked; no farewell to send')
            return
        self._closing = True
        self.repeat = False
        self.bbs.hard_stop('operator ended link')
        if ((self.contact is not None or self.client.connected)
                and (self.beacon or self._beacon_now or self.hopping.active)):
            self.set_beacon(None)
        if self.files.active or self.bbs_host.active:
            yield from self.files.close_steps('operator ended capture')
            return
        # Also settle an unlinked but still-keyed host. Completion must return
        # to monitoring even when /bye began in MONITORING.
        if self.flow.state not in (State.FAILED, State.RETURNED, State.CONNECTED):
            self.flow._state(State.FAILED)
        if self.client.connected:
            yield from self.flow.farewell_steps(
                text, cancelled=lambda: self._quitting or self._force_close)
        else:
            yield from self.flow.close_steps(
                cancelled=lambda: self._quitting or self._force_close)
        if self.line_state() is True:
            raise HostError('keying line remains HIGH after close; cannot resume monitoring')

    def quit(self):
        self._quitting = True
        self._closing = True
        self.bbs.hard_stop('operator quit')
        if self.beacon or self._beacon_now or self.hopping.active:
            self.set_beacon(None)
        try:
            if self._close_failed is not None:
                # terminal's finally calls quit after tick fails. Do not spend
                # another close deadline while the runtime still owns the key.
                raise self._close_failed
            steps, self.steps = self.steps, None
            if self._identifying and steps is not None:
                steps.close()
                steps = None
            stopping = steps is not None or not self.idle or self.line_state() is True
            try:
                if stopping:
                    self.client.finish(self.flow.close_steps(timeout=5, cancelled=lambda: True))
                elif not self.client.closed:
                    self.client.send_command('LISTEN OFF')
            finally:
                if steps is not None:
                    steps.close()
            if self.line_state() is True:
                raise HostError('keying line remains HIGH after close; stopping runtime')
        finally:
            self.files.end("session ended")
            self.bbs_host.reset()
            if self.contact is not None:
                self._end()
            self.identification_unsent()

    # -- the operator's commands ---------------------------------------------------

    def _local(self, line, words):
        """Reads and settings that touch neither the link nor the transmitter. No busy
        state refuses them and no draft is dropped for them."""
        if self.parking.local_command(line, words):
            return True
        if words[:1] == ['/pskreporter']:
            if words[1:] not in ([], ['status'], ['off']):
                self.say('usage: /pskreporter [status|off]; enable at launch with --pskreporter --loc GRID')
            elif self.reporter is None:
                self.say('PSKReporter off; enable at launch with --pskreporter --loc GRID')
            else:
                if words[1:] == ['off']:
                    self.reporter.stop()
                self.say('PSKReporter: ' + json.dumps(self.reporter.status(), sort_keys=True))
        elif words == ['/status']:
            self.status()
        elif words in (['/help'], ['/help', 'bbs']):
            if words == ['/help']:
                self.say('commands: ' + ', '.join(self.COMMANDS))
                self.say('Slash commands control this client; other text is sent as chat when linked.')
                self.say('/bye sends a graceful goodbye; /stop aborts an ordinary link or an active close; '
                         '/quit closes and exits without waiting for delivery. '
                         'During a readiness wait, /stop cancels only that wait.')
                self.say('/status, /help, /heard, /slots, /follow, /vmail reads, /auto, /away, /back, '
                         '/power and /verbose work in any state and preserve a draft; other commands drop it.')
                self.broadcast.help()
            self.bbs_help()
        elif words == ['/help', 'all']:
            self.broadcast.help()
        elif words == ['/all', 'status']:
            self.broadcast.status()
        elif words == ['/help', 'email']:
            self.email.command([])
        elif words == ['/email', 'status']:
            self.email.command(['status'])
        elif words[:1] == ['/verbose']:
            if words[1:] not in ([], ['on'], ['off']):
                self.say('usage: /verbose [on|off]')
            else:
                self.verbose = not self.verbose if len(words) == 1 else words[1] == 'on'
                self.say('verbose on: queued text, transmitted frames and decoded replies'
                         if self.verbose else 'verbose off')
        elif words == ['/heard']:
            for entry in self.heard:
                self.say(self.listing(entry))
            if not self.heard:
                self.say('nothing heard yet')
        elif words == ['/slots']:
            self.list_slots()
        elif words[:1] == ['/follow'] and len(words) == 2:
            target = words[1].lower()
            if target.isdigit() and len(target) <= 2:
                target = int(target)
            self.follow(None if target == 'off' else target)
        elif words == ['/beacon', 'hop', 'status']:
            self.hop_status()
        elif words == ['/vmail']:
            self.list_vmails()
        elif words[:2] in (['/vmail', 'pending'], ['/vmail', 'dismiss']):
            self.mail_collection.command(words[1:])
        elif words[:1] == ['/vmail'] and len(words) == 2 and words[1].isdigit():
            self.read_vmail(int(words[1]))
        elif words[:1] == ['/auto'] and words[1:] in (['on'], ['off']):
            self.set_auto(words[1] == 'on')
        elif words[:1] == ['/away']:
            self.set_away(line[5:].strip() or DEFAULT_AWAY)
        elif words == ['/back']:
            self.set_back()
        elif words[:1] == ['/power']:
            self.set_power(words[1:])
        else:
            return False
        return True

    def command(self, line):
        """One operator line; False ends the session."""
        line = line.rstrip('\r\n')
        words = line.split()
        try:
            if self._local(line, words):
                return True
            if self.parking.draft is not None:
                if words[:1] in (['/quit'], ['/bye'], ['/stop']):
                    self.parking.draft = None
                    self.say('Local VMail draft dropped; nothing sent')
                else:
                    self.parking.line(line)
                    return True
        except (ValueError, HostError, TimeoutError, OSError) as error:
            self._error(error)
            return True
        if self.bbs.command(line):
            return True
        if line == '/quit':
            return False
        if self.broadcast.active:
            if words in (['/stop'], ['/bye']):
                self.broadcast.cancel()
            else:
                self.say('Broadcast operation pending; /status or /all status shows progress; /stop cancels')
            return True
        if self._closing:
            if words == ['/stop']:
                self._force_close = True
                self.say('Stopping link now; waiting for abort confirmation and unkey')
            else:
                self.say('closing; /stop aborts without waiting for delivery')
            return True
        try:
            if (self.files.active or self.bbs_host.active) and words[:1] not in (['/stop'], ['/bye']):
                raise HostError('File evidence operation active; /stop or /bye ends this contact')
            if self.email.draft is not None:
                if not line.startswith('/'):
                    self.email.line(line)
                    return True
                self.email.draft = None
                self.say('Email draft dropped; nothing sent')
            if self.compose is not None and not line.startswith('/'):
                if line.strip() == '.':
                    self.send_vmail()
                else:
                    self.compose[1].append(line)
                return True
            if self.compose is not None:
                self.compose = None
                self.say('VMail dropped')
            if not line.strip():
                if self.steps is not None:
                    self.stop()
            elif not line.startswith('/'):
                self.client.send_text(line + '\n')
            elif words == ['/stop']:
                self.stop()
            elif words[:1] == ['/all']:
                self.broadcast.command(line[4:].strip())
            elif words[0] == '/answer' and len(words) == 2:
                self.answer(int(words[1]))
            elif words[0] == '/cq' and len(words) <= 2 + (repeat := words[-1] == 'repeat'):
                self.cq(int(words[1]) if len(words) > 1 + repeat else None, repeat=repeat)
            elif words[0] == '/call' and len(words) in (2, 3):
                stay_cf = len(words) == 3 and words[2].lower() == 'cf'
                self.call(words[1], int(words[2]) if len(words) == 3 and not stay_cf else None,
                          auto_qsy=not stay_cf)
            elif words[:1] == ['/inquire']:
                if words == ['/inquire', 'heard']:
                    self.inquire_heard()
                else:
                    self.say('usage: /inquire heard (remote list; /heard is local)')
            elif words[0] == '/tag' and len(words) >= 2:
                name, sep, value = line[5:].partition(' ')
                if name == 'LHR':
                    if sep:
                        raise ValueError('LHR takes no value; use /inquire heard')
                    self.inquire_heard()
                elif name == 'BLR':
                    if sep:
                        raise ValueError('BLR takes no value; use /bbs list')
                    self.list_bbs()
                else:
                    self.client.send_tag(name, value if sep else None)
            elif words[0] == '/qsy' and len(words) == 2:
                if self.steps is not None:
                    if self.flow.qsy_phase:
                        raise HostError(f'QSY to slot {self.flow.qsy_slot} already in progress: '
                                        f'{self.flow.qsy_phase}; /bye closes the link')
                    raise HostError('connection/opening or another application exchange is in progress')
                self._run(self.flow.request_qsy_steps(int(words[1])))
            elif words[0] == '/bye':
                if self.steps is not None:
                    self.steps.close()
                self._run(self._farewell(line[5:] or '73'))
            elif words == ['/clear']:
                self.say(f'transmit buffer: {self.client.clean_buffer()}')
            elif words[:2] == ['/beacon', 'hop']:
                self.set_hopping(words[2:])
            elif words[0] == '/beacon' and len(words) <= 2:
                if words[1:] in (['now'], ['force']):
                    self.beacon_now(force=words[1] == 'force')
                else:
                    self.set_beacon(None if words[1:] == ['off'] else int(words[1]) if words[1:] else BEACON_MIN)
            elif words[0] == '/band' and len(words) == 2:
                if words[1] in ('ready', 'unready'):
                    self.band_ready(words[1] == 'ready')
                else:
                    self.change_band(int(words[1]))
            elif words[:1] == ['/bbs']:
                if words == ['/bbs', 'list']:
                    self.list_bbs()
                elif words[1:2] in (['offer'], ['request'], ['get']) and len(words) >= 3:
                    self.files.command(words[1], line.split(None, 2)[2])
                elif words == ['/bbs', 'capture']:
                    self.files.command('capture')
                elif words[1:2] == ['get']:
                    self.say('usage: /bbs get N|filename')
                else:
                    self.say('usage: /bbs list | /bbs get N|filename | /bbs request N|filename | /bbs capture')
            elif words[:1] == ['/email']:
                self.email.command(words[1:])
            elif words[:2] == ['/vmail', 'collect']:
                self.mail_collection.command(words[1:])
            elif words[:2] == ['/vmail', 'notify'] and len(words) == 3:
                self.parking.notify(words[2])
            elif words[:2] == ['/vmail', 'send'] and len(words) >= 3:
                self.compose_vmail(line.split(None, 2)[2].strip())
            else:
                self.say('commands: ' + ', '.join(self.COMMANDS) + '; anything else is sent when linked')
        except (ValueError, HostError, TimeoutError) as error:
            self._error(error)
            if self.intent is not None and self.contact is None and self.steps is None:
                self.flow.state = State.FAILED
        return True

    def _operator_phase(self):
        """Local operation state, independent of the peer's advertised services."""
        if self._closing or self._quitting:
            return ('closing', 'closing: waiting for farewell/drain/disconnect',
                    'No input needed; waiting for the close to settle.')
        if self.client.closed:
            return ('host-closed', 'host connection closed', 'Restart the session to continue.')
        if (not self.client.connected
                and (self.client.ptt or self.line_state() is True)
                and self.flow.state in (State.CONNECTED, State.RETURNED, State.FAILED)):
            return ('settling', 'link ended; waiting for the keying line to settle before monitoring/beacons',
                    'No input needed; monitoring remains paused until host and physical unkey settle.')
        if self.broadcast.active:
            return ('broadcast', self.broadcast.phase,
                    'No input needed; /stop cancels after any frame in flight. Reception is unconfirmed.')
        if self.broadcast.selected == self.flow.dial and self.broadcast.selected is not None:
            return ('broadcast-slot', 'unlinked broadcast slot selected; auto-answer and beacons paused',
                    'Operator action: /all TEXT sends once here; /all off returns to normal monitoring.')
        if self._band_fault:
            return ('band-fault', 'band setup unverified', 'Operator action: /band N must succeed.')
        if self.email.draft is not None:
            return ('email-compose', 'composing email; waiting for your input',
                    'Enter body lines; . sends. /stop drops draft; /status and /help preserve it.')
        if self.email.waiting:
            return (('email', self.email.phase), self.email.phase,
                    'No input needed; awaiting gateway response. /stop cancels only the wait; /bye closes.')
        if self.compose is not None:
            return ('composing', 'composing VMail; waiting for your input',
                    'Operator action: enter body lines; a line containing . sends the VMail. '
                    '/status and /help preserve the draft.')
        if self.flow.qsy_phase is not None:
            return (('qsy', self.flow.qsy_phase),
                    f'QSY to slot {self.flow.qsy_slot}: {self.flow.qsy_phase}',
                    'No input needed; waiting for the peer/protocol. '
                    '/stop cancels a readiness wait; /bye closes the link.')
        if self.bbs_host.active:
            return ('bbs-send', f'BBS sending {self.bbs_host.name}',
                    '/stop or /bye cancels and closes this contact')
        if self.files.active and self.files.downloading:
            return ('file-download', 'receiving and validating BBS file',
                    '/stop or /bye cancels and closes this contact')
        if self.files.active or self.bbs_host.active:
            return ('file-capture', 'recording BBS file evidence; download unvalidated',
                    '/stop or /bye ends capture and closes this contact')
        if self._bbs_waiting:
            return ('bbs', 'waiting for the peer BBS directory response',
                    'No input needed; /stop cancels the wait, /bye closes the link.')
        if self._opening:
            return ('opening', 'link established; opening signal report exchange pending',
                    'No input needed; waiting for the peer/protocol. /bye closes the link.')
        if self._vmail_sent and self.client.connected:
            return ('vmail-ack', 'waiting for the peer VMail acknowledgement (<SMR>)',
                    'No input needed; an empty modem buffer does not acknowledge the VMail application.')
        if self.steps is not None and self.intent in ('cq', 'beacon'):
            return (self.intent, f'{self.intent} operation in progress ({self.flow.state.value})',
                    'No input needed; /stop cancels after any current frame.')
        if not self.client.connected and (self.client.pending or self.flow.state is State.CONNECTING):
            return ('connecting', 'connecting: waiting for modem link establishment',
                    'No input needed; /stop cancels the attempt.')
        if self.client.connected:
            if self.steps is not None:
                return ('exchange', 'link established; application exchange in progress',
                        'No input needed; waiting for the protocol. /bye closes the link.')
            if self.flow.dial in CALLING_HZ.values():
                return ('ready-cf', 'link established on the calling frequency; no exchange in progress',
                        'Ready for optional chat or /bye. BBS listing needs an agreed move off CF.')
            return ('ready-slot', 'link established; no exchange in progress',
                    'Ready for optional chat, /bbs list, or /bye. No automatic BBS request.')
        if self._follow_tuned and self.steps is None:
            return ('monitoring', 'passive follow; auto-answer and beacons suspended',
                    '/follow cf or /follow off returns to the calling frequency.')
        return ('monitoring', f'not linked; {self.flow.state.value}; {self._answering()}',
                'Ready for an operator command; no outgoing contact starts automatically.')

    def _answering(self):
        if self._auto_answer:
            return 'auto-answer on'
        if not self.auto:
            return 'auto-answer off (/auto on to answer)'
        return 'auto-answer suspended on this unready band (/band ready to answer)'

    def _data_status(self):
        pending = self.client.buffer > 0 or bool(self.client.data_writer and self.client.data_writer.pending)
        # A local send can precede the first BUFFER report. An old BUFFER 0
        # cannot prove that this write has drained, nor prove peer reading.
        after = self.client._sent_after
        unreported = after is not None and self.client.buffer_sequence <= after
        if pending:
            text = f'{self.client.buffer} bytes queued or awaiting modem acknowledgement'
        elif unreported:
            text = 'text queued locally; waiting for a fresh modem buffer report'
        else:
            text = 'no pending text reported (not proof the peer read it)'
        if self.client.ptt:
            text = 'transmitting a modem frame; ' + text
        return pending or unreported, text

    def status(self):
        dial = self.flow.dial
        place = 'frequency unverified' if dial is None else f'{khz(dial)} kHz'
        if dial in CALLING_HZ.values():
            place += ' (calling frequency)'
        else:
            slot = next((n for n in self.flow.codec.slots
                         if slot_frequency(self.flow.band, n, self.flow.codec) == dial), None)
            if slot is not None:
                place += f' (slot {slot})'
        link = 'established' if self.client.connected else 'not established'
        self.say(f'Status: peer {self.peer if self.client.connected or self.steps else "none"}; '
                 f'{place}; {self.client.bandwidth} Hz; modem link {link}. {self.parking.summary()}')
        _, phase, action = self._operator_phase()
        self.say('Operation: ' + phase)
        self.say('Data: ' + self._data_status()[1])
        if (progress := self.files.progress_text()) is not None:
            self.say(progress)
        if self.hopping.active:
            self.hop_status()
        self.say(action)
        self._notice_phase = self._operator_phase()[0]

    def _operator_notice(self):
        key, phase, action = self._operator_phase()
        old = self._notice_phase
        self._notice_phase = key
        active = key not in ('monitoring', 'ready-cf', 'ready-slot')
        was_active = old is not None and old not in ('monitoring', 'ready-cf', 'ready-slot')
        if key != old and (active or was_active):
            self.say('Operation: ' + phase + '. ' + action)
        pending, text = self._data_status()
        data_key = ('transmitting' if self.client.ptt else 'pending') if pending else 'empty'
        if data_key != self._notice_data:
            if pending or self._notice_data not in (None, 'empty'):
                self.say('Data: ' + text)
            self._notice_data = data_key

    def _service_hint(self, raw):
        match = re.fullmatch(r'<S:([A-Z]+)>', raw)
        if not self.client.connected or self._closing or self._quitting or match is None:
            return
        offered = frozenset(match[1])
        if offered in self._service_hints:
            return
        self._service_hints.add(offered)
        names = [name for bit, name in SERVICE_BITS
                 if any(SERVICE_LETTERS.get(letter) == bit for letter in offered)]
        if names:
            self.say(f'{self.peer} advertises: {", ".join(names)} (availability not yet checked).')
        if 'B' in offered:
            self.bbs_help()

    def bbs_help(self):
        self.say('BBS offers a file directory; it is not a command-shell login. '
                 '/bbs list requests the directory; /bbs get N downloads a file (32 KiB compressed limit). '
                 '/bbs request N (alias /bbs offer N) records an offer without readiness. '
                 '/bbs capture explicitly records opaque transfer bytes, not a validated download.')
        if self.client.connected and self.flow.dial in CALLING_HZ.values():
            self.say('Still on the calling frequency: BBS listing needs an agreed QSY off it. '
                     'A service advertisement does not announce auto-QSY readiness.')
        self.say('No BBS request is sent automatically. /help bbs shows this guidance.')

    def inquire_heard(self):
        """Public LHR request; response formatting/completion is not assumed."""
        if self._closing or self.client.closed or not self.client.connected \
                or self.flow.state is not State.CONNECTED:
            raise HostError('Last-heard inquiry needs an established link')
        if self.flow.dial in CALLING_HZ.values():
            raise HostError('Last-heard inquiry requires QSY off the calling frequency first; '
                            + self.qsy_hint())
        if (self.steps is not None or self._opening or self._bbs_waiting
                or self.mail_ack.busy or self.parking.active):
            raise HostError('Last-heard inquiry not sent: another application exchange is in progress')
        if self.client.pending or self.client.ptt or self._data_status()[0]:
            raise HostError('Last-heard inquiry not sent: wait for pending transmission to settle')
        self.client.send_tag('LHR')
        self.emit('inquiry', kind_requested='heard', peer=self.peer,
                  link_generation=self.client.link_generation, dial_hz=self.flow.dial,
                  outcome='queued')
        self.say(f'Last-heard request queued for {self.peer}; the remote reply appears in contact. '
                 '<LHE> means empty; <LHJ> means rejected. A modem ACK is not a list response. '
                 '/heard remains your local observations; no automatic retry.')

    def qsy_hint(self):
        return ('/qsy SLOT moves this link' if self.intent is not None else
                f'{self.peer} called, so its station starts the QSY')

    def list_bbs(self):
        if self.mail_ack.busy or self.parking.active:
            raise HostError('BBS list requires the mail exchange to settle')
        if not self.client.connected or self._closing:
            raise HostError('BBS list needs an established link')
        moving = self.flow.qsy_phase is not None or self.flow.qsy_receiving
        if self.steps is not None and not moving and not self._opening:
            raise HostError('BBS list not sent: another application exchange is in progress; '
                            'wait for it to finish')
        on_cf = self.flow.dial in CALLING_HZ.values()
        if self.steps is not None or (on_cf and self.intent is None):
            self._bbs_list_queued = self.client.link_generation
            self.say('BBS list queued: it is requested once this link settles off the calling frequency'
                     + ('' if self.steps is not None else '; ' + self.qsy_hint()) + '. /stop cancels it')
            return
        if self.flow.state is not State.CONNECTED:
            raise HostError('BBS list needs an established link')
        if on_cf:
            raise HostError('BBS list requires QSY off the calling frequency first; /qsy SLOT moves this link')
        self.files.entries = ()
        self.files.offer = None
        self._run(self._bbs_list_steps())

    def _bbs_list_steps(self):
        self._bbs_waiting = True
        try:
            after = self.client.message_sequence
            self.client.send_tag('BLR')
            self.say(f'BBS directory requested from {self.peer}; /stop cancels the wait')
            text = yield from self.client.wait_message_steps(
                lambda text: text.strip() == '<BLJ>' or text.startswith('<BL:'),
                after=after, timeout=60)
            self.emit('bbs_directory', peer=self.peer, dial_hz=self.flow.dial, text=text)
            if text.strip() == '<BLJ>':
                self.say(f'{self.peer} rejected the BBS directory request')
                return
            entries = bbs_files(text)
            self.files.listed(entries)
            self.say(f'BBS directory from {self.peer}: {len(entries)} files (sizes in bytes)')
            for n, entry in enumerate(entries, 1):
                self.say(f'  {n:3d}  {entry.name}  {entry.created}  {entry.size} bytes')
            self.say('File downloads are not available yet')
        finally:
            self._bbs_waiting = False

    def answer(self, n):
        if not 1 <= n <= len(self.heard):
            raise ValueError('nothing heard yet' if not self.heard else f'/answer N: N is 1 to {len(self.heard)}')
        entry = self.heard[n - 1]
        if entry.band not in (None, self.flow.band):
            raise ValueError(f'{entry.call} was heard on {entry.band} m; /band {entry.band} first')
        since = self.client.clock() - entry.at
        if since > HEARD_STALE_S:
            self.say(f'{entry.call} called {age(since)} ago and has likely returned to the calling frequency')
        place = f'slot {entry.slot} ({khz(self.where(entry))} kHz)' if entry.slot else 'the calling frequency'
        self.say(f'answering {entry.call} on {place} at {entry.bw} Hz')
        self._connect('answer', entry.call, entry.slot, entry.bw)

    def call(self, call, slot=None, *, auto_qsy=True):
        call = callsign(call)
        if base_call(call) == base_call(self.client.mycall):
            self.say(f'warning: {call} is this station\'s own callsign; calling it anyway')
        known = next((entry for entry in reversed(self.heard)
                      if entry.call == base_call(call) and entry.band in (None, self.flow.band)), None)
        self._connect('call', call, slot, known.bw if known else self.bandwidth, auto_qsy=auto_qsy)

    def _connect(self, how, call, slot, bw, *, auto_qsy=True):
        self._leave(how, slot, bw)
        self.peer = call
        self._run(self._connecting(call, slot, auto_qsy=auto_qsy))

    def _connecting(self, call, slot, *, auto_qsy=True):
        if not (yield from self.flow.call_steps(call, slot=slot)):
            self.say(f'call to {call} stopped')
            return
        if slot is None:
            # Q/AWQ invites the receiving application to open its outgoing-QSY
            # selector. Stock VarAC blocks its DATA listener in that modal UI.
            # Only the answering station advertises readiness; the caller waits
            # for it and sends the invitation without a reciprocal announcement.
            if auto_qsy:
                # P1 R08: peer Q on CF, then TL/QSYR and opening on the slot.
                # REL V11.4 keeps the report and profile off the calling frequency.
                selected = self.free_slot()
                self.say(f'Waiting for {call} to announce auto-QSY readiness; no slot invitation sent yet. '
                         '/stop cancels the wait and keeps the link for chat; /bye ends it.')
                yield from self.flow.request_qsy_steps(selected)
                self.slot = selected
        else:
            self._opening = True
            try:
                yield from self.flow.opening_steps()
                self.parking.start_delivery()
            finally:
                self._opening = False

    def cq(self, slot=None, repeat=False):
        if slot is None:
            slot = self.free_slot()
            self.say(f'slot {slot}: no CQ heard on it in the last {HEARD_STALE_S // 60} min'
                     + (', quiet on the slot monitor' if slot in {a.slot for a in self._watched()} else ''))
        self._leave('cq', slot, self.bandwidth)
        self.repeat = repeat
        self._run(self._repeating(slot) if repeat else self._calling(slot))

    def _calling(self, slot):
        if (yield from self.flow.cq_steps(slot, sniff_seconds=SNIFF_S, wait_seconds=self.wait_seconds)):
            yield from self.flow.opening_steps()
            self.parking.start_delivery()
            return
        self.say(f'CQ on slot {slot} stopped' if self.flow.cancelled else
                 f'no answer on slot {slot}' if self._called else f'slot {slot} is busy; no CQ sent')

    def _repeating(self, slot):
        n = 1
        while True:
            self.slot, self._called = slot, False
            self.say(f'CQ {n} on slot {slot} (repeating; /stop ends)')
            began = self.client.clock()
            if (yield from self.flow.cq_steps(slot, sniff_seconds=SNIFF_S, wait_seconds=self.wait_seconds)):
                yield from self.flow.opening_steps()
                self.parking.start_delivery()
                return
            if self.flow.cancelled:
                self.say(f'repeating CQ stopped after CQ {n}')
                return
            if not self._called:
                busy, slot = slot, self.free_slot(after=slot)
                self.say(f'slot {busy} is busy; moving to slot {slot}')
                continue
            n += 1
            gap = self.wait_seconds
            share = gap / (self.client.clock() - began + gap)
            self.say(f'no answer on slot {slot}; monitoring {khz(self.calling_hz)} kHz for '
                     f'{gap} s, then CQ {n} (listening on the calling frequency {share:.0%} '
                     f'of each cycle)')
            if (yield from self._between()):
                return
            if self.flow.cancelled:
                self.say(f'repeating CQ stopped before CQ {n}')
                return

    def _between(self):
        """The calling frequency between CQ cycles, for as long as the slot wait: beacon
        replies and direct calls land here (one did on 10-23 at 11:13Z), and VarAC
        publishes no CQ repeat to follow instead. Heard CQs list; a connect is taken."""
        yield from self._identify_steps()
        self.client.listen()
        end = self.client.clock() + self.wait_seconds
        while not self.client.connected:
            if self.client.pending or self.client.ptt:
                yield
            elif self.flow.cancelled or self.client.clock() >= end:
                return False
            else:
                yield end - self.client.clock()
        self.flow._state(State.CONNECTED)
        self.client.send_tag('Q')
        return True

    def free_slot(self, after=None):
        """The first slot no heard CQ used lately and the slot monitor holds quiet,
        searching onward from `after`; a slot the monitor hears is preferred."""
        now = self.client.clock()
        taken = {entry.slot for entry in self.heard
                 if entry.slot and entry.band in (None, self.flow.band) and now - entry.at < HEARD_STALE_S}
        watched = {a.slot: a.state for a in self._watched()}
        slots = self.flow.codec.slots
        if after in slots:
            cut = slots.index(after) + 1
            slots = slots[cut:] + slots[:cut]
        free = [slot for slot in slots if slot not in taken and watched.get(slot, 'quiet') == 'quiet']
        return next((slot for slot in free if slot in watched), next(iter(free), slots[0]))

    def _watched(self):
        return self.slots.table()[0] if self.slots is not None else []

    def list_slots(self):
        if self.slots is None:
            self.say('slot monitor off')
            return
        table, now = self.slots.table()
        for a in table:
            heard = f'heard {age(now - a.heard)} ago' if a.heard is not None else 'nothing heard'
            self.say(f'{a.name:<8} {a.audio_hz:>4} Hz  {a.state:<6}  {heard}'
                     + (f'  {a.who()}' if a.calls else '')
                     + (f'  level {a.level}' if a.level and a.state == 'busy' else '')
                     + (f'  <{"> <".join(a.tags)}>' if a.tags and a.state == 'busy' else '')
                     + ('  (ours)' if a.home else ''))
        if not table:
            self.say('no VarAC channel inside the passband')

    def beacon_now(self, *, force=False):
        if (self._band_fault or self._closing or self._quitting or not self.idle
                or self.contact is not None or self.steps is not None
                or self.line_state() is True or self.flow.state is not State.MONITORING):
            raise HostError('beacon now requires idle monitoring with no transmission in progress')
        if self.hopping.active and (not self.hopping.ready(self.flow.band)
                                    or self.hopping.cooldown(self.flow.band)):
            raise HostError('hopping beacons require /band ready and 15 min since the last beacon on this band')
        self._beacon_now = 'force' if force else 'now'
        if force:
            self.say('one forced beacon pending; detected channel occupancy overridden')
            return
        self.say(f'one beacon pending; waiting for {BEACON_CLEAR_S} s clear on the calling frequency')

    def set_beacon(self, minutes):
        if minutes is not None and minutes <= 0:
            raise ValueError('/beacon MINUTES: a positive number of minutes')
        was_hopping = self.hopping.active
        self.hopping.disable()
        if was_hopping:
            self._beacon_generation += 1
        if minutes is None:
            self.beacon = None
            self._beacon_expires = None
            self._beacon_due = self._beacon_last = None
            self._beacon_now = False
            self._beacon_generation += 1
            self.say('beacon off')
            if was_hopping and self._hop_idle():
                self.monitor()
            return
        now = self.client.clock()
        if self.beacon is None:
            self._beacon_expires = now + BEACON_LIFETIME_S
        self.beacon = minutes
        self._beacon_due = now if self._beacon_last is None else self._beacon_last + minutes * 60
        self.say(f'beacon every {minutes} min on {khz(self.calling_hz)} kHz, each once it has been '
                 f'clear {BEACON_CLEAR_S} s while monitoring')
        if was_hopping and self._hop_idle():
            self.monitor()

    def _hop_idle(self):
        return (not (self._band_fault or self._closing or self._quitting or self._follow_tuned)
                and self.flow.state is State.MONITORING and self.steps is None
                and self.contact is None and self.idle and self.line_state() is not True
                and self.broadcast.selected is None)

    def set_hopping(self, words):
        if not self._hop_idle():
            raise HostError('enable hopping while idle on the calling frequency')
        self.hopping.configure(words)
        self._beacon_generation += 1
        self._beacon_now = False
        if self.beacon is None:
            self._beacon_expires = self.client.clock() + BEACON_LIFETIME_S
        self.beacon = self.hopping.minutes
        self._beacon_due = self.client.clock()
        self.say(f'beacon hopping {"auto" if self.hopping.auto else "rotation"}: '
                 f'{", ".join(str(b) for b in self.hopping.bands)} m, every {self.beacon} min; '
                 'one arrival beacon after a clear channel; /beacon off cancels')
        self.monitor()
        self.hop_status()

    def hop_status(self):
        hop = self.hopping
        state = ('paused: ' + hop.paused if hop.paused else
                 f'next opportunity in {max(0, hop.due - self.client.clock()):.0f} s') if hop.active else 'off'
        self.say(f'Beacon hopping: {state}'
                 + (f'; arrival beacon pending on {hop.pending} m' if hop.pending is not None else ''))
        self.say('Band readiness: ' + ', '.join(f'{b} m {hop.readiness.get(b, "unknown")}'
                                               for b in hop.bands or CALLING_HZ))
        if hop.active and not hop.ready(self.flow.band):
            self.say('Auto-answer suspended on this unready band; verify its match, then /band ready')

    def band_ready(self, ready):
        if not self._hop_idle():
            raise HostError('acknowledge band readiness while idle on its verified calling frequency')
        band, hop = self.flow.band, self.hopping
        hop.readiness[band] = 'ready' if ready else 'needs operator TUNE'
        self.say(f'{band} m: {hop.readiness[band]} (operator acknowledgement for this session)')
        if hop.active:
            if ready and hop.paused == 'readiness':
                hop.paused = None
                hop.arrived(band)
                self._beacon_due = self.client.clock() + hop.cooldown(band)
            self.monitor()

    def _hop_tick(self):
        hop, now = self.hopping, self.client.clock()
        if not self._hop_idle():
            if hop.pending is not None or now >= hop.due:
                hop.defer()
            return False
        if hop.paused:
            return False
        if self._beacon_now:
            return hop.ready(self.flow.band) and not hop.cooldown(self.flow.band)
        if now >= hop.due:
            band, reason = hop.choose()
            self.say((f'{band} m: ' if band is not None else '') + reason)
            if band is None:
                return False
            try:
                # change_band rechecks link/keying after LISTEN OFF as well.
                self.change_band(band)
                hop.arrived(band)
                self._clear_since = None
                self._beacon_due = self.client.clock() + hop.cooldown(band)
            except (ValueError, HostError, TimeoutError, OSError) as error:
                hop.paused, hop.pending = 'band transition failed; retry /band, then re-enable hopping', None
                self._band_fault = True
                self.arm(False)
                self._error(error)
                self.say('Beacon hopping paused after band transition failure')
                return False
            return False  # Fresh clear-channel observation after the completed move.
        if hop.pending is None:
            return False
        if not hop.ready(self.flow.band):
            hop.paused = 'readiness'
            self.say(f'hopping paused: verify the match on {self.flow.band} m, then /band ready')
            return False
        return self._hop_idle() and not hop.cooldown(self.flow.band)

    def change_band(self, band):
        """`/band N`: another band's calling frequency, taken only while nothing is linked or
        going out, with the keying line disarmed while the rig moves. The slot monitor starts
        the band afresh; the heard list keeps each station's band, and a beacon its schedule."""
        if band not in CALLING_HZ:
            raise ValueError(f'/band N: N is one of {", ".join(map(str, CALLING_HZ))}')
        if self.client.connected or self.contact is not None:
            raise HostError(f'linked with {self.peer}; /bye first, then /band')
        if self.steps is not None:
            raise HostError(f'{OUTGOING.get(self.intent, "a transmission")} is going out; '
                            '/stop first, then /band')
        if not self.idle or self.line_state() is True:
            raise HostError('a link is being set up or the transmitter is keyed; /band once it is over')
        if band == self.flow.band and not self._band_fault:
            self.say(f'already on {band} m')
            return
        before, dial = self.flow.band, CALLING_HZ[band]
        self.arm(False)
        self._band_fault = True
        self.band_setup = None
        try:
            self.client.request('LISTEN OFF')
            self.say(f'{before} m to {band} m: {khz(dial)} kHz; nothing keys while the rig moves')
            if not self.idle or self.contact is not None or self.line_state() is True:
                raise HostError('link or keying began while disabling listen; band change cancelled')
            self.band_setup = self.band_tuner(dial)
            self.flow.band, self.flow.dial, self.client.busy = band, dial, None
            self.client.event(Event('band', f'{band} {dial}'))
            self.following, self._shown, self._transcripts, self._clear_since = None, {}, {}, None
            if self.slots is not None:
                self.slots.retune(dial, dial)
            self.emit('band', band_m=band, dial_hz=dial, from_band_m=before)
            if self.hopping.active:
                self.hopping.arrived(band)
                self._beacon_due = self.client.clock() + self.hopping.cooldown(band)
                if not self.hopping.ready(band):
                    self.hopping.paused = 'readiness'
                    self.say('arrival beacon paused: verify the band match, then /band ready')
            self._band_fault = False
            self.monitor()
        except BaseException as error:
            # A dial-only return cannot verify the filters. Keep the transmitter
            # disarmed until an explicit /band retries the complete setup.
            self._band_fault = True
            self.arm(False)
            self.flow.dial, self.flow.state = None, State.CALLING
            if self.hopping.active:
                self.hopping.paused = 'band transition failed; retry /band, then re-enable hopping'
                self.hopping.pending = None
            if self.client.closed and isinstance(error, (HostError, TimeoutError)):
                raise HostError('host connection closed during band setup; exit and restart the session') from error
            raise

    def beacon_text(self):
        """The advanced beacon, when the host has a KISS port and the profile a locator;
        otherwise None and the beacon is the standard `CALL-9` CQFRAME."""
        self.bbs_host.advertise()
        grid = (self.flow.profile.get('LOC') or '').upper()
        if self.client.kiss_socket is None or not re.fullmatch(r'[A-R]{2}[0-9]{2}[A-X]{2}', grid):
            return None
        return f'20{status_letter(self.flow.services)}{grid}{base_call(self.client.mycall)}'

    def _beaconing(self, *, once=False):
        """One attempt; scheduled busy drops retry at the next clear spell.

        Manual attempts never move the repeating schedule; off invalidates an
        in-flight scheduled attempt's permission to update that schedule.
        """
        text = None
        generation = self._beacon_generation
        band = self.flow.band
        try:
            if self.hopping.active and (not self.idle or self.contact is not None
                    or self.line_state() is True or self._closing or self._quitting
                    or self._band_fault or not self.hopping.ready(band)
                    or self.hopping.cooldown(band) or (self.client.busy and once != 'force')):
                raise Unsupported('hopping eligibility changed before beacon transmission')
            call, bw = base_call(self.client.mycall), self.client.bandwidth
            text = self.beacon_text()
            if once == 'force':
                yield from self.client.forced_beacon_steps(text)
            elif text is None:
                yield from self.client.cq_steps(f'CQFRAME {call}-9 {bw}', retry=False)
            else:
                yield from self.client.beacon_steps(text)
        except Unsupported as error:
            self.say(f'beacon not sent: {error}')
            self._clear_since = None
            if (not once and generation == self._beacon_generation
                    and not self.client.busy and self.beacon and 'busy' not in str(error)):
                self._beacon_due = self.client.clock() + self.beacon * 60
        else:
            self.hopping.last_tx[band] = self.client.clock()
            if not once and self.beacon and generation == self._beacon_generation:
                self._beacon_last = self.client.clock()
            self.emit('beacon', dial_hz=self.flow.dial, bandwidth_hz=self.client.bandwidth,
                      text=text or f'{call}-9')
            if not once and self.beacon and generation == self._beacon_generation:
                self._beacon_due = self.client.clock() + self.beacon * 60
                self.say(f'beacon sent; next in {self.beacon} min')
            elif once:
                self.say('one-shot beacon sent')
            if self.hopping.active and generation == self._beacon_generation:
                self.hopping.defer()
        finally:
            self.intent = None
            if not (self._closing or self._quitting):
                listening = self._auto_answer
                self.arm(listening)
                if listening:
                    self.client.listen()

    def _leave(self, how, slot, bw):
        if self._band_fault:
            raise HostError('band setup unverified; /band N must succeed before transmitting')
        if self.steps is not None and (self.repeat or self.intent == 'beacon'):
            self.flow.cancel()
            self.client.finish(self.steps)
            self.steps = None
            if self.owed:
                self._return()
        if self.steps is not None:
            raise HostError(f'{self.intent} is in progress; /stop first')
        if self.client.connected:
            raise HostError(f'linked with {self.peer}; /bye first')
        if not self.idle:
            raise HostError('an incoming link is being set up or the transmitter is keyed; '
                            'wait for it to settle')
        if self.hopping.active and how != 'beacon':
            self.hopping.defer()
        self.intent, self.slot, self._called, self.repeat = how, slot, False, False
        if self.away and how != 'beacon':
            self.set_back()
        self.arm(True)
        self.client.request('LISTEN OFF')
        if bw != self.client.bandwidth:
            self.client.set_bandwidth(bw)

    # -- away and VMail ------------------------------------------------------------

    def set_away(self, text):
        if any(c in text for c in '<>\r\n'):
            raise ValueError('an away message is one line of plain text (VarAC allows no tags in it)')
        self.away, self.new_vmails = text, 0
        self.emit('away', on=True, text=text)
        self.say(f'away: a caller gets <AWAY> and "{text}", may leave VMails, and is closed at '
                 f'{AWAY_LINK_S // 60} min from connection; /back returns')
        if not self.auto:
            self.set_auto(True)

    def set_back(self):
        self.away = None
        self.emit('away', on=False, new_vmails=self.new_vmails)
        self.say(f'back; {self.new_vmails} new VMail{"" if self.new_vmails == 1 else "s"}')

    def list_vmails(self):
        entries = self.mailbox.entries()
        if not entries:
            self.say('no VMails')
        for n, entry in enumerate(entries, 1):
            who = f"from {entry['from']}" if entry['direction'] == 'in' else f"to {entry['to']}"
            self.say(f"{n:>3}  {entry['utc'][:16]}Z  {who:<14} {entry['subject']}")

    def read_vmail(self, n):
        record = self.mailbox.read(n)
        where = f", {khz(record['dial_hz'])} kHz" if record.get('dial_hz') else ''
        self.say(f"VMail {n} from {record['from']} to {record['to']}, {record['utc'][:19]}Z{where}")
        if record['direction'] == 'in' and record['via'] != record['from']:
            self.say(f"relayed by {record['via']}")
        if record.get('composed_utc'):
            self.say(f"composed: {record['composed_utc']} UTC (sender-declared)")
        self.say(f"subject: {record['subject']}")
        for line in record['body'].replace('\r\n', '\n').split('\n'):
            self.say('  ' + line)

    def compose_vmail(self, subject):
        if not self.client.connected:
            raise HostError('a VMail goes to the linked station; /call or /answer first')
        if self.flow.dial in CALLING_HZ.values():
            raise HostError('VMail requires QSY off the calling frequency first; ' + self.qsy_hint())
        from .mail_parking import validate_content
        validate_content(subject, '', local=True)
        VMail(subject, '').wire()
        self.compose = (subject, [])
        self._compose_link = (self.client.link_generation, self.peer,
                              datetime.fromtimestamp(self.parking.store.clock(), timezone.utc)
                              .strftime('%Y-%m-%d %H:%M:%S'))
        self.say(f'VMail to {self.peer}, subject "{subject}": type the body; a line of only . sends it, '
                 '/status and /help preserve it; other /commands drop it')

    def send_vmail(self):
        from .mail_parking import validate_content
        subject, lines = self.compose
        generation, peer, composed = self._compose_link
        if generation != self.client.link_generation or peer != self.peer:
            raise HostError('VMail draft belongs to an earlier contact; drop it and compose again')
        vmail = VMail(subject, '\r\n'.join(lines))
        validate_content(vmail.subject, vmail.body, local=True)
        try:
            self.parking.send_direct(vmail, composed=composed)
        except OSError as error:
            raise HostError(f'VMail could not be queued: {error}') from error
        self.compose = None
        try:
            record = self.mailbox.store('out', self.client.mycall, self.peer, vmail,
                                        dial_hz=self.flow.dial, snr=self.client.snr)
        except OSError as error:
            raise HostError(f'VMail retained in parked queue; sent index could not be saved: {error}') from error
        self.emit('vmail', direction='out', peer=self.peer, id=record['id'], subject=subject)
        self.say(f'VMail queued for {self.peer}; waiting for peer acknowledgement <SMR>')

    @property
    def _clamp_said(self):
        return 'no clamp' if self.power_clamp_w is None else f'clamp {self.power_clamp_w:g} W'

    def set_power(self, words):
        """`/power [WATTS]`: show or set the rig's RF power. It is CAT and not the key, so
        it works in any state, but it never moves under a transmission: asked while
        keyed, it is set as soon as the key is up."""
        if self.power is None:
            self.say('power control is not available in this session')
            return
        if not words:
            read = self.power(None)
            self.say(f'power {"unread" if read is None else f"{read:g} W"} ({self._clamp_said})')
            return
        try:
            asked = float(words[0])
        except ValueError:
            asked = None
        if len(words) != 1 or asked is None or not 0 < asked <= 100:
            self.say('usage: /power [WATTS], 0 < WATTS <= 100')
            return
        watts, warning = clamp_power(asked, self.power_clamp_w, '/power')
        if warning:
            self.say(warning)
        if self.client.ptt or self.line_state() is True:
            self._power_owed = (asked, watts)
            self.say(f'transmitting; power goes to {watts:g} W as soon as the key is up')
            return
        self._apply_power(asked, watts)

    def _power_tick(self):
        if self._power_owed is not None and not (self.client.ptt or self.line_state() is True):
            owed, self._power_owed = self._power_owed, None
            self._apply_power(*owed)

    def _apply_power(self, asked, watts):
        try:
            read = self.power(watts)
        except OSError as error:
            self.say(f'power not set: {error}')
            return
        if read is None:
            self.say(f'!! power asked {watts:g} W, RFPOWER did not read back ({self._clamp_said})')
        elif abs(read - watts) > 1:
            self.say(f'!! power asked {watts:g} W, the rig reads {read:g} W ({self._clamp_said})')
        else:
            self.say(f'power set to {read:g} W ({self._clamp_said})')
        self.emit('power', requested_w=asked, set_w=watts, read_w=read, clamp_w=self.power_clamp_w)

    def set_auto(self, on):
        self.auto = on
        if (self.contact is None and self.idle and self.steps is None
                and not (self._closing or self._quitting or self.broadcast.active)):
            self.monitor()
        else:
            self.say(f'auto-answer {"on" if on else "off"} from the next return to the calling frequency')
