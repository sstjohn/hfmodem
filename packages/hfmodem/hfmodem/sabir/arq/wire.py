# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""SABIR session and connectionless wire format; see docs/protocols/sabir/SPEC.md.

Session controls are 44 bytes: type, version, fresh uint64 session identifier,
32 type-specific bytes, CRC16. DATA carries a uint32 transmission generation,
absolute uint16 profile plus handover flag, bitmap, geometry and uint64 stream
offset. CONNECT explicitly names both peers. Connectionless types remain compact
22-byte blocks. The robust decoder's bounded length hypotheses are CRC screened.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hfmodem.sabir.frame.codec import crc16
from hfmodem.sabir.phy.modem import CONST_BPC, Gear, carrier_groups

BLOCK_BYTES = 44
CONNECTIONLESS_BYTES = 22
WIRE_VERSION = 2
CAPS_BYTES = 31
MAX_CAPS = 3
CAPS = 10
CFAIL = 0x80
CRIT = 0x80
PAD, GEARSET, IMPL = 0, 1, 5
IMPLEMENTED_TLV = {PAD, GEARSET, IMPL}
(CONNECT, CONNECT_ACK, DATA, ACK, DISC, DISC_ACK,
 TURN, TURN_REQ, ID) = range(1, 10)

# connectionless beacon/presence/CQ/sounding frame-class (§7.1, types 48-63)
BEACON_PRESENCE = 48
BEACON_LO, BEACON_HI = 48, 63
ALL_CALL = 0                                # addressee: presence/CQ to anyone
DATAGRAM = 11

STATION_BYTES = 12                          # §3.2 station identifier width

# -- DATA/ACK gear field (§4.4) -----------------------------------------------
GEAR_MASK = 0xFFFF                          # absolute profile ID
HANDOVER = 1 << 16                         # DATA profile/flags bit 16
ACK_TRAFFIC = 0x01                          # ACK: "I have traffic queued"
ACK_TOOK_ROLE = 0x02                        # ACK: piggybacked, sender is ISS

# Profile support occupies bits matching the absolute DATA profile IDs.
# Features occupy bits 24–28; unassigned bits are ignored on receive.
FASTCTL = 1 << 24
PBACK = 1 << 25
LOADING = 1 << 26
DEFLATE = 1 << 27
BEACON = 1 << 28

# -- gear registry (§7) -------------------------------------------------------
# DATA and DATAGRAM use the same absolute uint16 namespace.
# Bits for control waveform IDs 0–2 are not DATA permissions.
GEAR_ID = {"floor4": 0, "floor2": 1, "floor": 2, "robust": 3, "workhorse": 4,
           "workhorse34": 5, "fast": 6, "max": 7, "doppler": 16, "sparse34": 17,
           "narrow": 18, "narrow2": 19, "narrow4": 20, "wide256": 21}
GEAR_NAME = {v: k for k, v in GEAR_ID.items()}

DATA_IDS = frozenset(v for v in GEAR_ID.values() if v >= 3)

NIBBLE_BPC = {1: 0, 2: 2, 3: 4, 4: 6}      # 0 = gear default, handled apart


def station_bytes(name: str) -> bytes:
    raw = name.encode("ascii")
    if not 1 <= len(raw) <= STATION_BYTES or any(c < 32 or c > 126 for c in raw):
        raise ValueError("session station identifier must fit 12 ASCII bytes")
    return raw.ljust(STATION_BYTES)


def capabilities(profile_ids, features: int = 0) -> int:
    """Build an exact receive-profile bitmap plus optional feature bits."""
    word = features
    for profile_id in profile_ids:
        if profile_id not in DATA_IDS:
            raise ValueError("unregistered DATA profile")
        word |= 1 << profile_id
    return word


def supported_profiles(word: int) -> list[int]:
    """Implemented DATA IDs explicitly named by a peer; unknown bits do not grant permission."""
    return sorted(i for i in DATA_IDS if word & (1 << i))


@dataclass(frozen=True)
class Control:
    """A session control; seq is a transmission generation, offset a byte position."""

    type: int
    session: int
    seq: int = 0
    gear: int = 0
    mask: bytes = bytes(8)
    aux: bytes = bytes(8)
    offset: int = 0
    destination: str = ""

    def pack(self) -> bytes:
        if len(self.mask) != 8 or len(self.aux) != 8:
            raise ValueError("invalid control fields")
        prefix = bytes([self.type, WIRE_VERSION]) + self.session.to_bytes(8, "big")
        if self.type in (CONNECT, CONNECT_ACK):
            payload = (bytes([0, self.gear]) + self.capability_word.to_bytes(4, "big")
                       + self.station + station_bytes(self.destination) + bytes(2))
        else:
            payload = (self.seq.to_bytes(4, "big") + self.gear.to_bytes(3, "big")
                       + self.mask + self.aux + self.offset.to_bytes(8, "big") + b"\0")
        body = prefix + payload
        if len(body) != BLOCK_BYTES - 2:
            raise ValueError("invalid control width")
        body = body.ljust(BLOCK_BYTES - 2, b"\0")
        return body + crc16(body).to_bytes(2, "big")

    # -- connect / identity views (§3.2-3.3, §4.5.2) -----------------------
    @property
    def n_ext(self) -> int:
        return self.gear & 0x0F

    @property
    def capability_word(self) -> int:
        return int.from_bytes(self.mask[:4], "big")

    @property
    def station(self) -> bytes:
        return self.mask[4:8] + self.aux

    @property
    def call(self) -> str:
        return self.station.decode("ascii", "replace").strip()

    @classmethod
    def connect(cls, session: int, capability_word: int, station: str, *,
                ack: bool = False, xh: int = 0, destination: str = "") -> "Control":
        sid = station_bytes(station)
        return cls(CONNECT_ACK if ack else CONNECT, session, gear=xh,
                   mask=capability_word.to_bytes(4, "big") + sid[:4], aux=sid[4:],
                   destination=destination)

    @classmethod
    def ident(cls, type: int, session: int, station: str) -> "Control":
        sid = station_bytes(station)
        return cls(type, session, mask=bytes(4) + sid[:4], aux=sid[4:])

    @classmethod
    def unpack(cls, buf: bytes) -> "Control | Caps | Beacon | DatagramHeader | None":
        if len(buf) not in (BLOCK_BYTES, CONNECTIONLESS_BYTES):
            return None
        t = buf[0]
        if t == 0x00 or t == 0xFF:                      # §5.1 all-0/all-1
            return None
        if crc16(buf[:-2]) != int.from_bytes(buf[-2:], "big"):
            return None
        if BEACON_LO <= t <= BEACON_HI:                 # connectionless: no
            return Beacon.unpack(buf)                    # session field at [1]
        if t == DATAGRAM:
            return DatagramHeader.unpack(buf)
        if len(buf) != BLOCK_BYTES or buf[1] != WIRE_VERSION or not any(buf[2:10]):
            return None
        if t == CAPS:
            return Caps.unpack(buf)
        session = int.from_bytes(buf[2:10], "big")
        if t in (CONNECT, CONNECT_ACK):
            if (buf[10] or any(buf[40:42]) or (buf[11] & 15) > MAX_CAPS
                    or buf[11] & ~(15 | (CFAIL if t == CONNECT_ACK else 0))):
                return None
            try:
                c = cls.connect(session, int.from_bytes(buf[12:16], "big"),
                                buf[16:28].decode("ascii").rstrip(), ack=t == CONNECT_ACK,
                                xh=buf[11], destination=buf[28:40].decode("ascii").rstrip())
                if c.pack() != buf:
                    return None
            except (ValueError, UnicodeError):
                return None
        else:
            if buf[41]:
                return None
            c = cls(t, session, int.from_bytes(buf[10:14], "big"),
                    int.from_bytes(buf[14:17], "big"), buf[17:25], buf[25:33],
                    int.from_bytes(buf[33:41], "big"))
            if t == DATA and (c.gear & ~(GEAR_MASK | HANDOVER) or not 1 <= c.aux[2] <= 64 or c.aux[7]):
                return None
        return c


@dataclass(frozen=True)
class DatagramHeader:
    """Self-describing connectionless body; gear is an absolute uint16 ID.

    The 64-bit token binds the header to SHA-256(body bytes). It is never
    used as sufficient evidence to soft-combine two undecoded packets.
    """
    gear: int
    n_symbols: int
    n_cw: int
    size: int
    token: bytes
    type: int = DATAGRAM

    def pack(self):
        import struct
        if not 1 <= self.n_cw <= 64 or not 0 < self.size <= 65535 or len(self.token) != 8:
            raise ValueError("invalid datagram header")
        body = struct.pack(">BBHHBBH8sH", DATAGRAM, 1, self.gear, self.n_symbols,
                           self.n_cw, 0, self.size, self.token, 0)
        body = body.ljust(CONNECTIONLESS_BYTES - 2, b"\0")
        return body + crc16(body).to_bytes(2, "big")

    @classmethod
    def unpack(cls, raw):
        import struct
        if len(raw) != CONNECTIONLESS_BYTES or any(raw[20:-2]) or crc16(raw[:-2]) != int.from_bytes(raw[-2:], "big"):
            return None
        t, version, gear, ns, nc, flags, size, token, reserved = struct.unpack(">BBHHBBH8sH", raw[:20])
        if t != DATAGRAM or version != 1 or flags or reserved or not 1 <= nc <= 64 or not size or not ns:
            return None
        return cls(gear, ns, nc, size, token)


@dataclass(frozen=True)
class Caps:
    """Optional, position-tagged extension block; absent for current profiles."""
    session: int
    pos: int
    tlvs: bytes = bytes(CAPS_BYTES)
    type: int = CAPS

    @property
    def total(self):
        return self.pos >> 4

    @property
    def idx(self):
        return self.pos & 15

    def pack(self):
        if len(self.tlvs) > CAPS_BYTES or not 1 <= self.total <= MAX_CAPS or self.idx >= self.total:
            raise ValueError("invalid CAPS geometry")
        body = (bytes([CAPS, WIRE_VERSION]) + self.session.to_bytes(8, "big")
                + bytes([self.pos]) + self.tlvs.ljust(CAPS_BYTES, b"\0"))
        return body + crc16(body).to_bytes(2, "big")

    @classmethod
    def build(cls, session, tlvs, total, idx):
        if len(tlvs) > CAPS_BYTES or not 1 <= total <= MAX_CAPS or not 0 <= idx < total:
            raise ValueError("invalid CAPS geometry")
        return cls(session, (total << 4) | idx, tlvs.ljust(CAPS_BYTES, b"\0"))

    @classmethod
    def unpack(cls, raw):
        if (len(raw) != BLOCK_BYTES or raw[0] != CAPS or raw[1] != WIRE_VERSION
                or not any(raw[2:10]) or crc16(raw[:-2]) != int.from_bytes(raw[-2:], "big")):
            return None
        total, idx = raw[10] >> 4, raw[10] & 15
        if not 1 <= total <= MAX_CAPS or idx >= total or parse_tlvs(raw[11:-2]) is None:
            return None
        return cls(int.from_bytes(raw[2:10], "big"), raw[10], raw[11:-2])


def pack_tlvs(items):
    out = b"".join(bytes([PAD]) if t == PAD else bytes([t, len(v)]) + v for t, v in items)
    if len(out) > CAPS_BYTES:
        raise ValueError("TLV stream exceeds one CAPS block")
    return out.ljust(CAPS_BYTES, b"\0")


def parse_tlvs(data):
    out, i = [], 0
    while i < len(data):
        t = data[i]
        if t == PAD:
            i += 1
            continue
        if t == CRIT or i + 1 >= len(data):
            return None
        n = data[i + 1]
        if i + 2 + n > len(data):
            return None
        v = data[i + 2:i + 2 + n]
        if t & 0x7F == GEARSET:
            if n % 2 or any(int.from_bytes(v[j:j + 2], "big") < 24 for j in range(0, n, 2)):
                return None
        out.append((t, v))
        i += 2 + n
    return out


def gearset_ids(items):
    return sorted({int.from_bytes(v[j:j + 2], "big")
                   for t, v in items if t & 0x7F == GEARSET for j in range(0, len(v), 2)})


def crit_tlvs(items):
    return [t & 0x7F for t, _ in items if t & CRIT]


@dataclass(frozen=True)
class Beacon:
    """A connectionless presence/CQ/sounding frame (§7.1, types 48-63). It
    carries the station's advertised capability image, so a listener learns the
    peer *without* a session -- the negotiate-up rendezvous. Byte 1 is the
    addressee (all-call / group-net-id / unicast), never a session tag."""

    addressee: int
    capability_word: int
    station: bytes
    profile: int = 0
    type: int = BEACON_PRESENCE

    @property
    def call(self) -> str:
        return self.station.decode("ascii", "replace").strip()

    def pack(self) -> bytes:
        body = (bytes([self.type, self.addressee & 0xFF])
                + self.capability_word.to_bytes(4, "big")
                + self.station.ljust(STATION_BYTES, b" ")[:STATION_BYTES]
                + bytes([self.profile & 0xFF, 0]))
        body = body.ljust(CONNECTIONLESS_BYTES - 2, b"\0")
        return body + crc16(body).to_bytes(2, "big")

    @classmethod
    def build(cls, capability_word: int, station: str, *, addressee: int = ALL_CALL,
              profile: int = 0) -> "Beacon":
        return cls(addressee, capability_word, station_bytes(station), profile)

    @classmethod
    def unpack(cls, buf: bytes) -> "Beacon | None":
        if len(buf) != CONNECTIONLESS_BYTES or any(buf[20:-2]):
            return None
        word = int.from_bytes(buf[2:6], "big")
        return cls(addressee=buf[1], capability_word=word, station=buf[6:18],
                   profile=buf[18])


def decode_capabilities(word: int) -> dict:
    """Decode implemented permissions for host reports and passive listeners."""
    return {"profiles": supported_profiles(word),
            "fastctl": bool(word & FASTCTL), "pback": bool(word & PBACK),
            "loading": bool(word & LOADING), "deflate": bool(word & DEFLATE),
            "beacon": bool(word & BEACON)}


# -- codeword bitmaps ----------------------------------------------------------
def cw_mask(indices) -> bytes:
    v = 0
    for j in indices:
        v |= 1 << j
    return v.to_bytes(8, "little")


def mask_indices(mask: bytes, n_cw: int) -> list[int]:
    v = int.from_bytes(mask, "little")
    return [j for j in range(n_cw) if v >> j & 1]


# -- DATA aux ------------------------------------------------------------------
def data_aux(n_syms: int, n_cw: int, nibbles) -> bytes:
    packed = bytes(nibbles[2 * i] << 4 | nibbles[2 * i + 1] for i in range(4))
    return n_syms.to_bytes(2, "big") + bytes([n_cw]) + packed + b"\0"


def parse_data_aux(aux: bytes):
    n_syms = int.from_bytes(aux[:2], "big")
    nibbles = tuple(b >> s & 0xF for b in aux[3:7] for s in (4, 0))
    return n_syms, aux[2], nibbles


def expand_loading(nibbles, gear: Gear) -> np.ndarray | None:
    """Loading nibbles -> per-carrier bits-per-cell array (None = uniform)."""
    if not any(nibbles):
        return None
    default = CONST_BPC[gear.constellation]
    per_group = np.array([default if n == 0 else NIBBLE_BPC.get(n, default)
                          for n in nibbles], dtype=np.int64)
    return per_group[carrier_groups(gear.n_carriers)]


# -- ACK aux -------------------------------------------------------------------
def ack_aux(group_snr_db) -> bytes:
    if group_snr_db is None:
        return bytes([0xFF] * 8)
    return bytes(0xFF if db is None
                 else int(np.clip(round((db + 16.0) * 4), 0, 254))
                 for db in group_snr_db)


def parse_ack_aux(aux: bytes) -> list[float | None]:
    return [None if q == 0xFF else q / 4.0 - 16.0 for q in aux]
