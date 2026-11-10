# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
from enum import Enum, auto
import re

from ..broadcast import _address
from ..coding.crc import crc16_genibus
from ..rx.broadcast import parse_compact
from ..ui import UIFrame
from ..vara.vara_frames import _CS6


class KissTxDisposition(Enum):
    CONTENDED = auto()
    REFUSED_BEFORE_KEY = auto()
    PARTIAL = auto()
    SENT = auto()
    UNKEY_UNCONFIRMED = auto()


MAX_PACKETS = 18
_SIZE = {1: 9, 2: 22, 4: 43}
_SPECIAL = {'ALL': 42, 'CQ': 43}
_TABLE = {b'/': 0, b'\\': 1, b'I': 2, b'R': 3, b'V': 4, b'D': 6, b'S': 7}
_POSITION = re.compile(
    rb'[!=]([\d ]{4})\.([\d ]{2})([NS])(.)([\d ]{5})\.([\d ]{2})([EW])(?=.)', re.S)
_MESSAGE = re.compile(rb':([A-Za-z0-9 -]{9}):')


def _call(name, ssid, width):
    return (''.join(f'{_CS6[c]:06b}' for c in name)
            + ('000000' if len(name) < width else '') + f'{ssid:04b}')


def _destination(address, width):
    name, ssid = address.base, address.ssid
    if ssid == 0 and name in _SPECIAL:
        return f'{_SPECIAL[name]:06b}'
    if ssid == 0 and name.startswith('AP'):
        return ('110100' + ''.join(f'{_CS6[c]:06b}' for c in name[2:])
                + ('000000' if len(name) < width else ''))
    return _call(name, ssid, width)


def _digipeater(address, width, as_captured):
    name, ssid = address.base, address.ssid
    if name in ('WIDE1', 'WIDE2') and ssid <= (3 if as_captured else 1):
        return f'100101{(int(name[-1]) - 1) * 4 + ssid:03b}'
    return _call(name, ssid, width)


def _path(digis, width, ap, as_captured, full_path):
    if full_path:
        return ('0' if ap and digis else '') + ''.join(
            _call(a.base, a.ssid, width) for a in digis)
    if ap and 1 <= len(digis) <= 3 and all((a.base, a.ssid) == (f'WIDE{k}', 1)
                                        for k, a in enumerate(digis, 1)):
        return '1001'
    return ('0' if ap and digis else '') + ''.join(
        _digipeater(a, width, as_captured) for a in digis)


def _aprs(info, as_captured):
    if m := _POSITION.match(info):
        try:
            lat = int((m[1] + m[2]).replace(b' ', b''))
            lon = int((m[5] + m[6]).replace(b' ', b''))
        except ValueError:
            return '00', '', info
        plain = info[m.end():]
        expanded = (f'!{lat // 100:04d}.{lat % 100:02d}'.encode() + m[3] + m[4]
                    + f'{lon // 100:05d}.{lon % 100:02d}'.encode() + m[7] + plain)
        if as_captured or expanded == info and lat < 1 << 20 and lon < 1 << 21:
            table = f'1{_TABLE[m[4]]:03b}' if m[4] in _TABLE else f'0{m[4][0]:07b}'
            bits = f'{lat:020b}{int(m[3] == b"N")}{table}{lon:021b}{int(m[7] == b"W")}'
            return '01', bits, plain
    if m := _MESSAGE.match(info):
        name, _, suffix = m[1].decode().strip().upper().partition('-')
        if re.fullmatch(r'[A-Z0-9]{1,6}', name) and re.fullmatch(r'\d*', suffix):
            ssid = int(suffix or 0)
            addressee = name + (f'-{ssid}' if ssid else '')
            plain = info[m.end():]
            expanded = b':' + addressee.ljust(9).encode() + b':' + plain
            # VARA HF stations expand a leading 0x05 to "ack", even if it was literal input.
            if ssid <= 15 and (as_captured or expanded == info and not plain.startswith(b'\x05')):
                if plain.startswith(b'ack'):
                    plain = b'\x05' + plain[3:]
                return '10', _call(name, ssid, 6), plain
    return '00', '', info


def compact(message, *, as_captured=False, full_path=False):
    frame = message if isinstance(message, UIFrame) else UIFrame.from_ax25(
        _address(message.destination, 6 + message.command, command=True)
        + _address(message.source, 6 + message.command, last=True)
        + b'\x03\xf0' + message.info, message.command)
    if as_captured:
        # Capture reproduction includes the fabricated and truncated identities
        # VARA HF stations send.
        frame = replace(frame, addresses=tuple(replace(
            a, base=re.match(r'[A-Z0-9]*', a.base.strip().upper())[0])
            for a in frame.addresses))
    else:
        frame = frame.normalized()
    dst, src, *digis = frame.addresses
    width = 6 + frame.command
    general = (frame.control, frame.pid) != (3, 0xF0)
    ap = dst.base.startswith('AP')
    kind = ((0 if general else 0x40) | 16 * frame.command
            | ((0x08 | len(digis) - 1) if digis else 0))
    stream = (bytes([frame.control, frame.pid]) if general else b'') + frame.info
    w = 8 if any(c > 127 for c in (frame.info if as_captured else stream)) else 7
    bits = (f'{kind:08b}' + ''.join(str(a.h) for a in digis)
            + _destination(dst, width) + _call(src.base, src.ssid, width)
            + _path(digis, width, ap, as_captured, full_path))
    mode, payload, plain = (_aprs(frame.info, as_captured)
                            if ap and frame.command == 0 and w == 7 and not general
                            else ('00', '', frame.info))
    def serialize(mode, payload, plain):
        stream = (bytes([frame.control, frame.pid]) if general else b'') + plain
        encoded = bits + (mode if ap else '') + str(int(w == 8)) + payload
        encoded += ''.join(f'{c & ((1 << w) - 1):0{w}b}' for c in stream)
        if mode != '00':
            encoded += '0' * (-len(encoded) % 8)
        else:
            encoded += str(1 - int(encoded[-1])) * (
                8 - len(encoded) % 8 if w == 7 else -len(encoded) % 8)
        return bytes(int(encoded[i:i + 8], 2) for i in range(0, len(encoded), 8))

    core = serialize(mode, payload, plain)
    if not as_captured and mode != '00':
        decoded = parse_compact(core)
        if decoded is None or decoded.info != frame.info:
            mode = '00'
            core = serialize(mode, '', frame.info)
    return core, mode


def packets(core, mode='00', *, level=None):
    if not core:
        raise ValueError('empty UI core')
    if level is None:
        sized = len(core) - 8 * (mode == '01')
        level = 1 if sized <= 16 else 2 if sized <= 42 else 4
    if level not in _SIZE:
        raise ValueError('UI has no measured transmit row at this level')
    size = _SIZE[level]
    if len(core) > MAX_PACKETS * (size - 1):
        raise ValueError('UI exceeds 18 physical packets')
    if len(core) <= size:
        payloads = [core]
    else:
        body = core[1:]
        chunks = [body[:size - 2]] + [body[k:k + size - 1]
                    for k in range(size - 2, len(body), size - 1)]
        # At 18 packets the count overlaps checksum bit 4; this is a byte-wide OR.
        meta = (crc16_genibus(core) >> 8) & 0xF0 | len(chunks) - 2
        payloads = ([bytes([core[0] | 128, meta]) + chunks[0]]
                    + [bytes([core[0] | 128]) + c for c in chunks[1:-1]]
                    + [bytes([core[0]]) + chunks[-1]])
    frames = []
    for payload in payloads:
        full = len(payload) == size
        raw = (payload + (b'' if full else b'\x14\x00'))[:size].ljust(size, b'\x00')
        raw += bytes([(8 if level == 1 else 0) | full])
        frames.append(raw + crc16_genibus(raw).to_bytes(2, 'big'))
    return level, frames


def packetize(message, *, level=None, as_captured=False, full_path=False):
    return packets(*compact(message, as_captured=as_captured, full_path=full_path), level=level)
