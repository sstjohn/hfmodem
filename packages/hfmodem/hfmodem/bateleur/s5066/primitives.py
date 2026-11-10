# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""S_ primitives: what a client and the subnetwork say to each other (A.2).

On the wire (A.2.2.2) each is the Maury-Styles preamble (0x90, 0xEB), a
version byte (0), the primitive's size in two bytes (most significant first),
and the primitive, whose first byte is its type. Fields within the
primitive are as the figures of A.2.2 lay them out; node addresses take
four bytes, the address's size in half-bytes in the top three bits, a group
flag under them, and the 28-bit address in the rest (A.2.2.28.1).

A ``Primitive`` is its type and a dict of its arguments by name, the names
as the annex's tables give them in lower case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PREAMBLE = b"\x90\xeb"
VERSION = 0
MAX_TTL = 648000 * 2                       # A.2.2.20: two-second units

BIND_REQUEST = 1
UNBIND_REQUEST = 2
BIND_ACCEPTED = 3
BIND_REJECTED = 4
UNBIND_INDICATION = 5
HARD_LINK_ESTABLISH = 6
HARD_LINK_TERMINATE = 7
HARD_LINK_ESTABLISHED = 8
HARD_LINK_REJECTED = 9
HARD_LINK_TERMINATED = 10
HARD_LINK_INDICATION = 11
HARD_LINK_ACCEPT = 12
HARD_LINK_REJECT = 13
SUBNET_AVAILABILITY = 14
DATA_FLOW_ON = 15
DATA_FLOW_OFF = 16
KEEP_ALIVE = 17
MANAGEMENT_MSG_REQUEST = 18
MANAGEMENT_MSG_INDICATION = 19
UNIDATA_REQUEST = 20
UNIDATA_INDICATION = 21
UNIDATA_REQUEST_CONFIRM = 22
UNIDATA_REQUEST_REJECTED = 23
EXPEDITED_UNIDATA_REQUEST = 24
EXPEDITED_UNIDATA_INDICATION = 25
EXPEDITED_UNIDATA_REQUEST_CONFIRM = 26
EXPEDITED_UNIDATA_REQUEST_REJECTED = 27

NAMES = {v: k for k, v in globals().items() if k.isupper() and isinstance(v, int)
         and k not in ("VERSION",)}

# Transmission modes (A.2.2.3, A.2.2.28.2-3).
MODE_DEFAULT, MODE_ARQ, MODE_NON_ARQ, MODE_NON_ARQ_ERRORS = 0, 1, 2, 3
# Delivery confirmation.
CONFIRM_NONE, CONFIRM_NODE, CONFIRM_CLIENT = 0, 1, 2

# Reasons (A.2.1.4, A.2.1.5, A.2.1.8).
BIND_NO_RESOURCES, BIND_INVALID_SAP, BIND_SAP_TAKEN, BIND_BROADCAST_ONLY = 1, 2, 3, 4
UNBIND_PREEMPTED, UNBIND_INACTIVE, UNBIND_INVALID, UNBIND_TOO_MANY_EXPEDITED, \
    UNBIND_BROADCAST = 1, 2, 3, 4, 5
REJECT_TTL, REJECT_SAP_UNBOUND, REJECT_NOT_RESPONDING, REJECT_MTU, REJECT_NO_MODE = \
    1, 2, 3, 4, 5
# Hard links (A.3.1.6, A.3.1.7).
HL_REMOTE_BUSY, HL_HIGHER_PRIORITY, HL_NOT_RESPONDING, HL_SAP_UNBOUND = 1, 2, 3, 4
HL_TYPE0_EXISTS = 5            # A.3.2.2.2 names it; the table leaves 5 free
HL_TERM_CLIENT, HL_TERM_HIGHER, HL_TERM_SAP_UNBOUND, HL_TERM_NOT_RESPONDING, \
    HL_TERM_PHYSICAL = 1, 2, 4, 3, 5


@dataclass
class ServiceType:
    """The SERVICE TYPE of a bind, and the DELIVERY MODE of a U_PDU."""
    mode: int = MODE_ARQ
    confirm: int = CONFIRM_NONE
    in_order: bool = True
    extended: bool = False
    retransmissions: int = 0

    def bits(self) -> int:
        """12 bits: mode, confirmation, order, extended, retransmissions."""
        return ((self.mode & 0xF) << 8) | ((self.confirm & 3) << 6) \
            | ((0 if self.in_order else 1) << 5) | (int(self.extended) << 4) \
            | (self.retransmissions & 0xF)

    @classmethod
    def from_bits(cls, v: int) -> "ServiceType":
        return cls((v >> 8) & 0xF, (v >> 6) & 3, not (v >> 5) & 1, bool((v >> 4) & 1), v & 0xF)


@dataclass
class Primitive:
    type: int
    args: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return NAMES.get(self.type, f"type {self.type}")

    def __getitem__(self, k):
        return self.args[k]

    def get(self, k, default=None):
        return self.args.get(k, default)


def _addr(a: int, group: bool = False) -> bytes:
    n = max(1, (a.bit_length() + 3) // 4)
    return ((n << 29) | (int(group) << 28) | (a & 0x0FFFFFFF)).to_bytes(4, "big")


def _unaddr(b: bytes) -> tuple[int, bool]:
    v = int.from_bytes(b[:4], "big")
    return v & 0x0FFFFFFF, bool((v >> 28) & 1)


def _link(t, prio, sap) -> int:
    return ((t & 3) << 6) | ((prio & 3) << 4) | (sap & 0xF)


def _unlink(b: int) -> dict:
    return {"link_type": b >> 6, "link_priority": (b >> 4) & 3, "remote_sap": b & 0xF}


def encode_body(p: Primitive) -> bytes:
    """The primitive, without the preamble, version and size."""
    t, a = p.type, p.args
    out = bytes([t])
    if t == BIND_REQUEST:
        st = a.get("service", ServiceType()).bits()
        return out + bytes([((a["sap"] & 0xF) << 4) | (a.get("rank", 0) & 0xF),
                            st >> 4, (st & 0xF) << 4])
    if t == BIND_ACCEPTED:
        return out + bytes([(a["sap"] & 0xF) << 4]) + a.get("mtu", 2048).to_bytes(2, "big")
    if t in (BIND_REJECTED, UNBIND_INDICATION):
        return out + bytes([a["reason"] & 0xFF])
    if t in (UNBIND_REQUEST, DATA_FLOW_ON, DATA_FLOW_OFF, KEEP_ALIVE):
        return out
    if t == HARD_LINK_ESTABLISH or t == HARD_LINK_ACCEPT:
        return out + bytes([_link(a.get("link_type", 0), a.get("link_priority", 0),
                                  a.get("remote_sap", 0))]) + _addr(a["remote"])
    if t == HARD_LINK_TERMINATE:
        return out + _addr(a["remote"])
    if t in (HARD_LINK_ESTABLISHED, HARD_LINK_INDICATION, HARD_LINK_REJECTED,
             HARD_LINK_TERMINATED, HARD_LINK_REJECT):
        first = a.get("status", 0) if t in (HARD_LINK_ESTABLISHED, HARD_LINK_INDICATION) \
            else a.get("reason", 0)
        return out + bytes([first & 0xFF, _link(a.get("link_type", 0), a.get("link_priority", 0),
                                                a.get("remote_sap", 0))]) + _addr(a["remote"])
    if t == SUBNET_AVAILABILITY:
        return out + bytes([a.get("status", 0) & 0xFF, a.get("reason", 0) & 0xFF])
    if t in (MANAGEMENT_MSG_REQUEST, MANAGEMENT_MSG_INDICATION):
        return out + bytes([a.get("msg_type", 0) & 0xFF]) + a.get("body", b"")
    if t in (UNIDATA_REQUEST, EXPEDITED_UNIDATA_REQUEST):
        mode = a.get("mode", ServiceType(MODE_DEFAULT)).bits()
        ttl = max(0, min(int(a.get("ttl", 0) // 2), MAX_TTL // 2))
        u = a["u_pdu"]
        prio = a.get("priority", 0) if t == UNIDATA_REQUEST else 0
        return out + bytes([((prio & 0xF) << 4) | (a["dest_sap"] & 0xF)]) \
            + _addr(a["dest"], a.get("group", False)) \
            + bytes([mode >> 4, ((mode & 0xF) << 4) | (ttl >> 16), (ttl >> 8) & 0xFF,
                     ttl & 0xFF]) + len(u).to_bytes(2, "big") + u
    if t in (UNIDATA_INDICATION, EXPEDITED_UNIDATA_INDICATION):
        u = a["u_pdu"]
        mode = a.get("mode", MODE_ARQ)
        prio = a.get("priority", 0) if t == UNIDATA_INDICATION else 0
        body = out + bytes([((prio & 0xF) << 4) | (a["dest_sap"] & 0xF)]) \
            + _addr(a["dest"], a.get("group", False)) \
            + bytes([((mode & 0xF) << 4) | (a["src_sap"] & 0xF)]) + _addr(a["src"]) \
            + len(u).to_bytes(2, "big")
        if mode == MODE_NON_ARQ_ERRORS:
            for key in ("errored", "missing"):
                blocks = a.get(key, [])
                body += len(blocks).to_bytes(2, "big")
                for pos, size in blocks:
                    body += pos.to_bytes(2, "big") + size.to_bytes(2, "big")
        return body + u
    if t in (UNIDATA_REQUEST_CONFIRM, UNIDATA_REQUEST_REJECTED,
             EXPEDITED_UNIDATA_REQUEST_CONFIRM, EXPEDITED_UNIDATA_REQUEST_REJECTED):
        u = a.get("u_pdu", b"")
        hi = a.get("reason", 0) if t in (UNIDATA_REQUEST_REJECTED,
                                         EXPEDITED_UNIDATA_REQUEST_REJECTED) else 0
        return out + bytes([((hi & 0xF) << 4) | (a["dest_sap"] & 0xF)]) + _addr(a["dest"]) \
            + len(u).to_bytes(2, "big") + u
    raise ValueError(f"no S_ primitive type {t}")


def encode(p: Primitive) -> bytes:
    body = encode_body(p)
    return PREAMBLE + bytes([VERSION]) + len(body).to_bytes(2, "big") + body


class PrimitiveError(ValueError):
    pass


def decode_body(b: bytes) -> Primitive:
    if not b:
        raise PrimitiveError("empty primitive")
    t = b[0]
    need = {BIND_REQUEST: 4, BIND_ACCEPTED: 4, BIND_REJECTED: 2, UNBIND_INDICATION: 2,
            HARD_LINK_ESTABLISH: 6, HARD_LINK_ACCEPT: 6, HARD_LINK_TERMINATE: 5,
            HARD_LINK_ESTABLISHED: 7, HARD_LINK_INDICATION: 7, HARD_LINK_REJECTED: 7,
            HARD_LINK_TERMINATED: 7, HARD_LINK_REJECT: 7, SUBNET_AVAILABILITY: 3,
            MANAGEMENT_MSG_REQUEST: 2, MANAGEMENT_MSG_INDICATION: 2, UNIDATA_REQUEST: 12,
            EXPEDITED_UNIDATA_REQUEST: 12, UNIDATA_INDICATION: 13,
            EXPEDITED_UNIDATA_INDICATION: 13, UNIDATA_REQUEST_CONFIRM: 8,
            UNIDATA_REQUEST_REJECTED: 8, EXPEDITED_UNIDATA_REQUEST_CONFIRM: 8,
            EXPEDITED_UNIDATA_REQUEST_REJECTED: 8}.get(t, 1)
    if t not in NAMES:
        raise PrimitiveError(f"no S_ primitive type {t}")
    if len(b) < need:
        raise PrimitiveError(f"{NAMES[t]} is short")
    a: dict = {}
    if t == BIND_REQUEST:
        a["sap"], a["rank"] = b[1] >> 4, b[1] & 0xF
        a["service"] = ServiceType.from_bits((b[2] << 4) | (b[3] >> 4))
    elif t == BIND_ACCEPTED:
        a["sap"], a["mtu"] = b[1] >> 4, int.from_bytes(b[2:4], "big")
    elif t in (BIND_REJECTED, UNBIND_INDICATION):
        a["reason"] = b[1]
    elif t in (HARD_LINK_ESTABLISH, HARD_LINK_ACCEPT):
        a.update(_unlink(b[1]))
        a["remote"] = _unaddr(b[2:6])[0]
    elif t == HARD_LINK_TERMINATE:
        a["remote"] = _unaddr(b[1:5])[0]
    elif t in (HARD_LINK_ESTABLISHED, HARD_LINK_INDICATION, HARD_LINK_REJECTED,
               HARD_LINK_TERMINATED, HARD_LINK_REJECT):
        key = "status" if t in (HARD_LINK_ESTABLISHED, HARD_LINK_INDICATION) else "reason"
        a[key] = b[1]
        a.update(_unlink(b[2]))
        a["remote"] = _unaddr(b[3:7])[0]
    elif t == SUBNET_AVAILABILITY:
        a["status"], a["reason"] = b[1], b[2]
    elif t in (MANAGEMENT_MSG_REQUEST, MANAGEMENT_MSG_INDICATION):
        a["msg_type"], a["body"] = b[1], bytes(b[2:])
    elif t in (UNIDATA_REQUEST, EXPEDITED_UNIDATA_REQUEST):
        if t == UNIDATA_REQUEST:
            a["priority"] = b[1] >> 4
        a["dest_sap"] = b[1] & 0xF
        a["dest"], a["group"] = _unaddr(b[2:6])
        mode = (b[6] << 4) | (b[7] >> 4)
        a["mode"] = ServiceType.from_bits(mode)
        a["ttl"] = (((b[7] & 0xF) << 16) | (b[8] << 8) | b[9]) * 2
        n = int.from_bytes(b[10:12], "big")
        a["u_pdu"] = bytes(b[12:12 + n])
        if len(a["u_pdu"]) != n:
            raise PrimitiveError("U_PDU shorter than its size")
    elif t in (UNIDATA_INDICATION, EXPEDITED_UNIDATA_INDICATION):
        if t == UNIDATA_INDICATION:
            a["priority"] = b[1] >> 4
        a["dest_sap"] = b[1] & 0xF
        a["dest"], a["group"] = _unaddr(b[2:6])
        a["mode"], a["src_sap"] = b[6] >> 4, b[6] & 0xF
        a["src"] = _unaddr(b[7:11])[0]
        n = int.from_bytes(b[11:13], "big")
        at = 13
        if a["mode"] == MODE_NON_ARQ_ERRORS:
            for key in ("errored", "missing"):
                k = int.from_bytes(b[at:at + 2], "big")
                at += 2
                a[key] = [(int.from_bytes(b[at + 4 * i:at + 4 * i + 2], "big"),
                           int.from_bytes(b[at + 4 * i + 2:at + 4 * i + 4], "big"))
                          for i in range(k)]
                at += 4 * k
        a["u_pdu"] = bytes(b[at:at + n])
    elif t in (UNIDATA_REQUEST_CONFIRM, UNIDATA_REQUEST_REJECTED,
               EXPEDITED_UNIDATA_REQUEST_CONFIRM, EXPEDITED_UNIDATA_REQUEST_REJECTED):
        if t in (UNIDATA_REQUEST_REJECTED, EXPEDITED_UNIDATA_REQUEST_REJECTED):
            a["reason"] = b[1] >> 4
        a["dest_sap"] = b[1] & 0xF
        a["dest"] = _unaddr(b[2:6])[0]
        n = int.from_bytes(b[6:8], "big")
        a["u_pdu"] = bytes(b[8:8 + n])
    return Primitive(t, a)


class Framer:
    """Bytes from a client's stream in, whole primitives out."""

    MAX = 65535 + 5

    def __init__(self) -> None:
        self.buf = b""

    def push(self, data: bytes) -> list[Primitive]:
        self.buf += data
        out = []
        while True:
            i = self.buf.find(PREAMBLE)
            if i < 0:
                self.buf = self.buf[-1:] if self.buf.endswith(PREAMBLE[:1]) else b""
                return out
            if i:
                self.buf = self.buf[i:]
            if len(self.buf) < 5:
                return out
            if self.buf[2] != VERSION:
                self.buf = self.buf[1:]
                continue
            n = int.from_bytes(self.buf[3:5], "big")
            if len(self.buf) < 5 + n:
                return out
            body, self.buf = self.buf[5:5 + n], self.buf[5 + n:]
            try:
                out.append(decode_body(body))
            except PrimitiveError:
                continue


def prim(t: int, **args) -> Primitive:
    return Primitive(t, args)
