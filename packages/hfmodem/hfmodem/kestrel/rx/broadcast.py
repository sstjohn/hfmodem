# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Measured KISS UI broadcasts; not connected DATA or advanced beacons.

The RF compact format is kept separate from its command-0/1 AX.25 host envelope.
Only CRC-clean, complete packet groups may enter this decoder.
"""

import numpy as np
from ..broadcast import Broadcast, MAX_INFO_BYTES, _split
from ..coding.crc import crc16_genibus
from ..ui import UIAddress, UIFrame
from ..vara.vara_frames import _CS6

HEADERS = {
    1: np.array(
        [
            74,
            64,
            60,
            60,
            76,
            60,
            60,
            54,
            72,
            66,
            66,
            52,
            60,
            70,
            64,
            70,
            60,
            50,
            50,
            72,
            52,
            72,
            50,
            52,
            50,
            56,
            62,
            76,
            70,
            64,
            58,
            58,
        ]
    ),
    4: np.array(
        [
            74,
            74,
            52,
            64,
            60,
            62,
            72,
            62,
            60,
            52,
            62,
            50,
            58,
            52,
            58,
            60,
            64,
            76,
            58,
            74,
            60,
            54,
            54,
            58,
            64,
            66,
            66,
            50,
            58,
            58,
            50,
            66,
        ]
    ),
}
_REVERSE = {v: k for k, v in _CS6.items()}
_SPECIAL = {42: "ALL", 43: "CQ"}


def _call_bits(call, command, *, destination=False):
    name, ssid = _split(call, 6 + command)
    if any(c not in _CS6 for c in name):
        raise ValueError(
            "the VARA RF encoding truncates internal-hyphen addresses; cannot transmit this identity intact"
        )
    special = next(
        (n for n, s in _SPECIAL.items() if destination and s == name and ssid == 0),
        None,
    )
    if special is not None:
        return f"{special:06b}"
    return (
        "".join(f"{_CS6[c]:06b}" for c in name)
        + ("" if len(name) == 6 + command else "000000")
        + f"{ssid:04b}"
    )


def compact(message: Broadcast) -> bytes:
    from ..tx.ui import compact as encode

    return encode(message)[0]


def _read_header(raw):
    if len(raw) < 2 or raw[0] not in (0x40, 0x50):
        raise ValueError("not a direct UI type")
    command = (raw[0] - 0x40) // 16
    bits = "".join(f"{byte:08b}" for byte in raw)
    at = 8

    def take(n):
        nonlocal at
        if at + n > len(bits):
            raise ValueError("short compact address")
        value = int(bits[at : at + n], 2)
        at += n
        return value

    def call(destination=False):
        chars = []
        for _ in range(6 + command):
            code = take(6)
            if destination and not chars and code in _SPECIAL:
                return _SPECIAL[code]
            if code == 0:
                break
            if code not in _REVERSE:
                raise ValueError("unknown compact callsign symbol")
            chars.append(_REVERSE[code])
        ssid = take(4)
        if not chars:
            raise ValueError("empty compact address")
        return "".join(chars) + (f"-{ssid}" if ssid else "")

    destination, source = call(True), call()
    width = 8 if take(1) else 7
    return command, source, destination, width, at, bits


def ui_type(kind):
    return kind & 0x7F in (0x00, 0x10, 0x40, 0x50) or kind & 0x68 in (0x08, 0x48)


class _Compact:
    def __init__(self, raw):
        self.bits = ''.join(f'{byte:08b}' for byte in raw)
        self.at = 0
        self.path_forms = []

    def take(self, n):
        if self.at + n > len(self.bits):
            raise ValueError('short compact field')
        value = int(self.bits[self.at:self.at + n], 2)
        self.at += n
        return value

    def call(self, width, *, destination=False, hop=False, source=False):
        code = self.take(6)
        if destination and code in _SPECIAL:
            return UIAddress(_SPECIAL[code])
        if hop and code == 37:
            self.path_forms.append('wide')
            value = self.take(3)
            return UIAddress(f'WIDE{(value >> 2) + 1}', value & 3)
        if code == 0 and not source:
            raise ValueError('empty compact address')
        ap = destination and code == 52
        chars = list('AP') if ap else [' '] if code == 0 else []
        if ap or code == 0:
            code = self.take(6)
        while code:
            if code not in _REVERSE:
                raise ValueError('unknown compact callsign symbol')
            chars.append(_REVERSE[code])
            if len(chars) == width:
                break
            code = self.take(6)
        if not chars:
            raise ValueError('empty compact address')
        if hop:
            self.path_forms.append('full')
        return UIAddress(''.join(chars), 0 if ap else self.take(4))

    def header(self):
        kind = self.take(8)
        if not ui_type(kind) or kind & 128:
            raise ValueError('not a UI core')
        self.command = kind >> 4 & 1
        self.general = not kind & 0x40
        count = (kind & 7) + 1 if kind & 8 else 0
        used = [self.take(1) for _ in range(count)]
        width = 6 + self.command
        dst, src = self.call(width, destination=True), self.call(width, source=True)
        ap = dst.base.startswith('AP')
        if ap and count and self.take(1):
            if count > 3 or self.take(3) != 1:
                raise ValueError('invalid standard path')
            self.path_forms.append('standard')
            path = [UIAddress(f'WIDE{k + 1}', 1) for k in range(count)]
        else:
            path = [self.call(width, hop=True) for _ in range(count)]
        self.addresses = [dst, src] + [
            UIAddress(a.base, a.ssid, 0x60 | h << 7) for a, h in zip(path, used)]
        last = self.addresses[-1]
        self.addresses[-1] = UIAddress(last.base, last.ssid, last.flags | 1)
        self.mode = self.take(2) if ap else 0
        self.width = 8 if self.take(1) else 7
        if self.mode == 3 or self.mode and (self.command or self.general or self.width != 7):
            raise ValueError('invalid compressed mode')
        self.control, self.pid = ((self.take(self.width), self.take(self.width))
                                  if self.general else (3, 0xF0))


def parse_compact(raw: bytes, *, sample=0, sn=0.0) -> UIFrame | None:
    try:
        fields = _Compact(raw)
        fields.header()
        prefix = b''
        if fields.mode == 1:
            lat, north = fields.take(20), fields.take(1)
            if fields.take(1):
                table = {0: '/', 1: '\\', 2: 'I', 3: 'R', 4: 'V', 6: 'D', 7: 'S'}[fields.take(3)]
            else:
                table = chr(fields.take(7))
                if not 32 <= ord(table) <= 126:
                    return None
            lon, west = fields.take(21), fields.take(1)
            prefix = (f'!{lat // 100:04d}.{lat % 100:02d}' + ('N' if north else 'S')
                      + table + f'{lon // 100:05d}.{lon % 100:02d}'
                      + ('W' if west else 'E')).encode('ascii')
        elif fields.mode == 2:
            addressee = fields.call(6).call
            prefix = b':' + addressee.ljust(9).encode('ascii') + b':'
        bits, at, width = fields.bits, fields.at, fields.width
        pads = range(7, -1, -1) if fields.mode else range(1 if width == 7 else 0, 9 if width == 7 else 8)
        for pad in pads:
            stop = len(bits) - pad
            if stop < at or (stop - at) % width:
                continue
            expected = '0' if fields.mode else str(1 - int(bits[stop - 1]))
            if bits[stop:] != expected * pad:
                continue
            plain = bytes(int(bits[i:i + width], 2) for i in range(at, stop, width))
            if fields.mode == 1 and not plain:
                continue
            if fields.mode == 2 and plain.startswith(b'\x05'):
                plain = b'ack' + plain[1:]
            return UIFrame(fields.command, tuple(fields.addresses), fields.control, fields.pid,
                           prefix + plain, int(sample), None if sn is None else float(sn))
        return None
    except (ValueError, KeyError, UnicodeError):
        return None


def packet_count(meta):
    return (meta & 15) + 2 if meta & 15 else 18 if meta & 16 else 2


def parse_packets(frames, *, level, sample=0, sn=0.0):
    """Deliver only a complete, CRC-clean group in its received encoding."""
    if level not in (1, 2, 3, 4) or not 1 <= len(frames) <= 18:
        return None
    size = {1: 9, 2: 22, 3: 34, 4: 43}[level]
    joined = b''
    kind = bytes(frames[0])[0] & 0x7F if len(frames[0]) else -1
    if not ui_type(kind):
        return None
    for index, frame in enumerate(frames):
        raw = bytes(frame)
        if len(raw) != size + 3 or crc16_genibus(raw[:-2]) != int.from_bytes(raw[-2:], 'big'):
            return None
        payload, marker = raw[:-3], raw[-3]
        if marker not in ((8, 9) if level == 1 else (0, 1)):
            return None
        if payload[0] & 0x7F != kind or bool(payload[0] & 128) != (index < len(frames) - 1):
            return None
        if index < len(frames) - 1 and not marker & 1:
            return None
        data = payload[2:] if index == 0 and len(frames) > 1 else payload[1:]
        if not marker & 1:
            ends = [i for i in range(len(data)) if data[i] == 0x14 and not any(data[i + 2:])]
            if not ends:
                return None
            data = data[:ends[-1]]
        joined += data
    core = bytes([kind]) + joined
    if len(frames) > 1 and bytes(frames[0])[1] != ((crc16_genibus(core) >> 8 & 0xF0) | (len(frames) - 2)):
        return None
    return parse_compact(core, sample=sample, sn=sn)


def packetize(message: Broadcast, *, level=None):
    from ..tx.ui import packetize as encode

    return encode(message, level=level)


PERIOD = {1: 259584, 2: 239104, 4: 207872}
BODY_SAMPLES = {level: period - 1536 for level, period in PERIOD.items()}


def deadline(header, level, n=18):
    return header + 67072 + (n - 1) * PERIOD[level] + BODY_SAMPLES[level] + 2048 + 24000


def decode_packet(samples, body_start, *, level, on_attempt=None, on_alignment=None):
    """One physical slot with the measured alignment search, at 48 kHz.

    Report a CRC-clean body's DATA-convention origin so the next slot can
    follow the measured grid, including the level-4 sub-band stagger.
    """
    from . import varahf500 as R

    x = np.asarray(samples)
    start = int(body_start)
    if level == 4:
        for delta in (-512, 1024):
            a = max(0, start + delta)
            b = min(len(x), a + 403 * 512 + 4096)
            if b - a < 403 * 512:
                continue
            if on_attempt:
                on_attempt()
            frame = R.decode_burst_frames(x, a, b, level=4)[0]
            if frame.crc_ok:
                if on_alignment:
                    on_alignment(frame.origin.training_origin)
                return bytes(frame.frame_bytes)
    elif level in (1, 2):
        r = R.INDEX_RECORDS[level]
        a = max(0, start - 2 * r.dw)
        b = min(len(x), a + (r.ncols + R._L3_LEAD_MAX + 2) * r.dw)
        if b - a < (r.ncols + r.lead) * r.dw:
            return None
        aligned = R._l3_align(x, a, b, level)
        if aligned[2] < R._L3_GUARD_MIN:
            return None
        if on_attempt:
            on_attempt()
        frame = R._l3_frames(a, b, *aligned, level)[0]
        if frame.crc_ok:
            if on_alignment:
                on_alignment(aligned[0] + (aligned[1] - r.lead) * r.dw)
            return bytes(frame.frame_bytes)
    return None


def decode_audio(samples, body_start, *, level, sample=0, sn=0.0, on_attempt=None):
    if level not in PERIOD:
        return None
    frames = []
    for index in range(18):
        raw = decode_packet(samples, body_start + index * PERIOD[level],
                            level=level, on_attempt=on_attempt)
        if raw is None or not ui_type(raw[0]):
            return None
        frames.append(raw)
        if not raw[0] & 128:
            return parse_packets(frames, level=level, sample=sample, sn=sn)
    return None


def _is_short_tail(raw, level):
    """The short last packet of a multipart UI group, which cannot parse alone.

    Its marker is the UI short form (8 at level 1, else 0) and its payload ends
    0x14, one CRC byte, then zeros. Session DATA never carries that marker with
    that end: a full body's per-frame field ends binary 01, a short close ends
    on the level byte and 0x82 [arq.phy.vara_body], and the one close whose
    field is the caller CRC puts 0x14 in the payload's last byte instead.
    """
    payload, marker = raw[:-3], raw[-3]
    if marker != (8 if level == 1 else 0) or payload[-1] == 0x14:
        return False
    end = payload.rfind(0x14)
    return 0 < end <= len(payload) - 2 and not any(payload[end + 2 :])


def _is_direct_single(raw, level):
    """Preserve the existing connected-DATA admission predicate."""
    if parse_packets([raw], level=level) is None:
        return False
    core = raw[:-3]
    if not raw[-3] & 1:
        core = core[:core.rfind(0x14)]
    try:
        command, source, destination, width, at, bits = _read_header(core)
        info = bytes(int(bits[i:i + width], 2)
                     for i in range(at, len(bits) - width + 1, width))
        alternatives = [info]
        if width == 7 and len(info) > 1 and info[-1] in (0, 127):
            alternatives.append(info[:-1])
        return any(1 <= len(candidate) <= MAX_INFO_BYTES
                   and compact(Broadcast(command, source, destination, candidate)) == core
                   for candidate in alternatives)
    except ValueError:
        return False


def is_broadcast_frame(frame):
    """Reject identified UI packets at connected-DATA admission.

    Complete single packets require canonical re-encoding; a short last packet
    heard without its first needs the UI short envelope. Multipart first
    packets require the qualified physical marker, checksum-nibble envelope
    and a complete compact address header. The group checksum is checked by
    parse_packets before UI delivery; a partial packet cannot deliver UI.
    As with advanced beacons, deliberately identical application bytes are
    indistinguishable without RF-header context.
    """
    raw = bytes(frame.frame_bytes)
    level = frame.level
    size = {1: 9, 2: 22, 4: 43}.get(level)
    if (
        not getattr(frame, "crc_ok", False)
        or size is None
        or len(raw) != size + 3
        or crc16_genibus(raw[:-2]) != int.from_bytes(raw[-2:], "big")
    ):
        return False
    if raw[0] in (0x40, 0x50):
        return _is_direct_single(raw, level) or _is_short_tail(raw, level)
    if raw[0] not in (0xC0, 0xD0) or raw[1] & 15 or raw[-3] != (9 if level == 1 else 1):
        return False
    core = bytes([raw[0] & 0x7F]) + raw[2:-3]
    try:
        command, source, destination, width, at, bits = _read_header(core)
        prefix = (
            f"{core[0]:08b}"
            + _call_bits(destination, command, destination=True)
            + _call_bits(source, command)
            + str(int(width == 8))
        )
        return at < len(bits) and bits.startswith(prefix)
    except ValueError:
        return False
