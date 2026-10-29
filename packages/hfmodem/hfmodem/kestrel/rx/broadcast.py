# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Measured direct KISS UI broadcasts; not connected DATA or advanced beacons.

The RF compact format is kept separate from its command-0/1 AX.25 host envelope.
Only CRC-clean, complete packet groups may enter this decoder.
"""

import numpy as np
from ...contact.broadcast import Broadcast, MAX_INFO_BYTES, _split
from ..coding.crc import crc16_genibus
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
            "stock RF truncates internal-hyphen addresses; cannot transmit this identity intact"
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
    """Pack the direct RF address and information fields (no packet envelope)."""
    message.encode()  # Validate host envelope and qualified information bound.
    width = 8 if any(c > 127 for c in message.info) else 7
    bits = (
        f"{0x40 + message.command * 0x10:08b}"
        + _call_bits(message.destination, message.command, destination=True)
        + _call_bits(message.source, message.command)
        + str(int(width == 8))
        + "".join(f"{c:0{width}b}" for c in message.info)
    )
    bits += str(1 - int(bits[-1])) * (8 - len(bits) % 8)
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


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


def parse_compact(raw: bytes, *, sample=0, sn=0.0) -> Broadcast | None:
    try:
        command, source, destination, width, at, bits = _read_header(raw)
        info = bytes(
            int(bits[i : i + width], 2) for i in range(at, len(bits) - width + 1, width)
        )
        # The packer always adds 1..8 complement bits. A whole ASCII padding
        # character can only be NUL/DEL; canonical re-encoding checks its bits.
        alternatives = [info]
        if width == 7 and len(info) > 1 and info[-1] in (0, 127):
            alternatives.append(info[:-1])
        for candidate in alternatives:
            if not 1 <= len(candidate) <= MAX_INFO_BYTES:
                continue
            result = Broadcast(
                command, source, destination, candidate, int(sample),
                None if sn is None else float(sn)
            )
            if compact(result) == raw:
                return result
        return None
    except (ValueError, UnicodeError):
        return None


def parse_packets(frames, *, level, sample=0, sn=0.0):
    """Reassemble at most two separately coded physical packets, failing closed."""
    if level not in (1, 2, 3, 4) or not 1 <= len(frames) <= 2:
        return None
    size = {1: 9, 2: 22, 3: 34, 4: 43}[level]
    joined = b""
    kind = None
    for index, frame in enumerate(frames):
        raw = bytes(frame)
        if len(raw) != size + 3 or crc16_genibus(raw[:-2]) != int.from_bytes(
            raw[-2:], "big"
        ):
            return None
        payload, marker = raw[:-3], raw[-3]
        if marker not in ((8, 9) if level == 1 else (0, 1)):
            return None
        if payload[0] & 0x7F not in (0x40, 0x50):
            return None
        if kind is None:
            kind = payload[0] & 0x7F
        if payload[0] & 0x7F != kind or bool(payload[0] & 128) != (
            index < len(frames) - 1
        ):
            return None
        data = payload[2:] if index == 0 and len(frames) > 1 else payload[1:]
        if not marker & 1:
            # The short trailer may contain the previous caller CRC. Its
            # terminator is the final 0x14 followed by one CRC byte and zeros.
            ends = [
                i
                for i in range(len(data))
                if data[i] == 0x14 and not any(data[i + 2 :])
            ]
            if not ends:
                return None
            data = data[: ends[-1]]
        joined += data
    core = bytes([kind]) + joined
    if len(frames) > 1 and bytes(frames[0])[1] != (crc16_genibus(core) >> 8) & 0xF0:
        return None
    return parse_compact(core, sample=sample, sn=sn)


def packetize(message: Broadcast, *, level=None):
    """Measured one/two-packet UI envelope; each packet has its own PHY CRC."""
    from ..tx.varahf500_tx import build_frame, build_l3_frame

    core = compact(message)
    if level is None:
        level = 1 if len(core) <= 16 else 2 if len(core) <= 42 else 4
    if level not in (1, 2, 4):
        raise ValueError("UI transmit row is not stock-qualified")
    size = {1: 9, 2: 22, 4: 43}[level]
    if len(core) <= size:
        payloads = [core]
    else:
        if len(core) > 2 * size - 2:
            raise ValueError("UI addresses and information exceed two physical packets")
        checksum = (crc16_genibus(core) >> 8) & 0xF0
        payloads = [
            bytes([core[0] | 128, checksum]) + core[1 : size - 1],
            bytes([core[0]]) + core[size - 1 :],
        ]
    frames = []
    for payload in payloads:
        full = len(payload) == size
        marker = (8 if level == 1 else 0) | int(full)
        payload = (payload + (b"\x14\0" if not full else b""))[:size].ljust(size, b"\0")
        frames.append(
            build_frame(payload, marker)
            if level == 4
            else build_l3_frame(payload, marker, level)
        )
    return level, frames


def decode_audio(samples, body_start, *, level, sample=0, sn=0.0, on_attempt=None):
    """Bounded header-gated UI decoder at 48 kHz/1500 Hz.

    Each physical packet restarts its coding grid, unlike connected paired DATA.
    At most two packets and two base-row alignment attempts per packet are read.
    """
    from . import varahf500 as R

    x = np.asarray(samples)
    if level not in (1, 2, 4):
        return None
    frames = []
    start = int(body_start)
    for index in range(2):
        if level == 4:
            frame = None
            for delta in (-512, 1024):
                a = max(0, start + delta)
                b = min(len(x), a + 403 * 512 + 4096)
                if b - a < 403 * 512:
                    continue
                if on_attempt:
                    on_attempt()
                candidate = R.decode_burst_frames(x, a, b, level=4)[0]
                if candidate.crc_ok:
                    frame = candidate
                    break
            if frame is None:
                return None
            step = 403 * 512 + 1536
        else:
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
            if not frame.crc_ok:
                return None
            # Use the measured data grid, not accumulated header quantization.
            start = aligned[0] + (aligned[1] - r.lead) * r.dw
            step = (r.lead + r.ncols) * r.dw + 1536
        raw = bytes(frame.frame_bytes)
        if raw[0] & 0x7F not in (0x40, 0x50):
            return None
        frames.append(raw)
        if not raw[0] & 128:
            return parse_packets(frames, level=level, sample=sample, sn=sn)
        start += step
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
        return parse_packets([raw], level=level) is not None or _is_short_tail(raw, level)
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
