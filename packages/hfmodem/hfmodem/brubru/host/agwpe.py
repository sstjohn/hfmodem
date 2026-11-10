# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One RPR port in the AGWPE dialect used by Pat's pinned wl2k-go client.

Independent implementation from the public fixed-header interface and the
consumer contract. The legacy capability baud enumeration has no specified
R300 value: this profile returns 255 (unspecified) and identifies R300 in G.
That extension needs consumer qualification; it never advertises a 1200-baud
radio port. No raw-frame owner, second connection, or login service is offered.
"""
from dataclasses import dataclass
import struct

from hfmodem.ax25.connected import State
from hfmodem.ax25.frame import address

HEADER = struct.Struct('<B3xBxBx10s10sI4x')
MAX_DATA = 65536


def _call(raw):
    return raw.split(b'\0', 1)[0].decode('ascii').upper()


@dataclass(frozen=True)
class Packet:
    port: int
    kind: str
    pid: int = 0
    source: str = ''
    destination: str = ''
    data: bytes = b''

    def encode(self):
        if len(self.kind) != 1 or len(self.data) > MAX_DATA:
            raise ValueError('invalid AGWPE frame')
        source, destination = self.source.encode('ascii'), self.destination.encode('ascii')
        if len(source) > 10 or len(destination) > 10:
            raise ValueError('AGWPE callsign exceeds ten bytes')
        return HEADER.pack(self.port, ord(self.kind), self.pid, source, destination, len(self.data)) + self.data


class Decoder:
    def __init__(self):
        self.pending = bytearray()

    def feed(self, data):
        out = []
        for start in range(0, len(data), 4096):
            self.pending.extend(data[start:start + 4096])
            while len(self.pending) >= HEADER.size:
                port, kind, pid, source, destination, length = HEADER.unpack_from(self.pending)
                if length > MAX_DATA:
                    raise ValueError('AGWPE input exceeds length limit')
                if len(self.pending) < HEADER.size + length:
                    break
                body = bytes(self.pending[HEADER.size:HEADER.size + length])
                del self.pending[:HEADER.size + length]
                out.append(Packet(port, chr(kind), pid, _call(source), _call(destination), body))
        return out


class AgwHost:
    def __init__(self, engine, send, *, txdelay=25, persistence=32, slot=10):
        self.engine, self.send = engine, send
        self.txdelay, self.persistence, self.slot = txdelay, persistence, slot
        self.registered = ''
        self._originated = False
        self.alive = True
        self.decoder = Decoder()
        self.engine.state_changed = self._state
        self.engine.receive = self._receive
        self.engine.can_accept = lambda: self.alive and bool(self.registered)
        self.engine.listen = False

    def _send(self, packet):
        if self.alive and self.send(packet.encode()) is False:
            self.alive = False
            self.engine.listen = False
            self.engine.abort('AGWPE attachment output failed')
            raise ConnectionAbortedError('AGWPE host output failed')

    def _state(self, state, remote, reason):
        if state is State.CONNECTED:
            prefix = 'With' if self._originated else 'To'
            self._send(Packet(0, 'C', source=remote, destination=self.engine.mycall,
                              data=f'*** CONNECTED {prefix} {remote}\r'.encode('ascii')))
        elif state is State.DISCONNECTED:
            self._send(Packet(0, 'd', source=remote, destination=self.engine.mycall,
                              data=f'*** DISCONNECTED From {remote}\r'.encode('ascii')))
            self._originated = False

    def _receive(self, payload):
        # Pat's Conn.Read cannot accept a block larger than its caller's buffer.
        # 128 is this profile's declared block limit, independently of AX.25 N1.
        for start in range(0, len(payload), 128):
            self._send(Packet(0, 'D', 0xF0, self.engine.remote, self.engine.mycall,
                              payload[start:start + 128]))

    def attachment_lost(self):
        self.alive = False
        self.engine.listen = False
        self.engine.abort('AGWPE attachment lost')
        self.registered = ''
        self.decoder = Decoder()

    def feed(self, data):
        if not self.alive:
            raise ConnectionAbortedError('AGWPE attachment closed')
        for packet in self.decoder.feed(data):
            self.handle(packet)

    def _refuse(self, packet):
        self._send(Packet(0, 'd', source=packet.destination, destination=packet.source,
                          data=f'*** DISCONNECTED From {packet.destination}\r'.encode('ascii')))

    def handle(self, p):
        if p.port != 0:
            raise ValueError('AGWPE profile has only port zero')
        if p.kind == 'R':
            self._send(Packet(0, 'R', data=struct.pack('<HHHH', 0, 0, 1, 0)))
        elif p.kind == 'G':
            self._send(Packet(0, 'G', data=b'1;Port1 hfmodem R300 Robust Packet;\0'))
        elif p.kind == 'g':
            body = bytes((255, 255, min(self.txdelay, 255), 0, self.persistence,
                          self.slot, self.engine.window, int(self.engine.active))) + bytes(4)
            self._send(Packet(0, 'g', data=body))
        elif p.kind == 'X':
            try:
                address(p.source)
                okay = bool(p.source) and (not self.registered or self.registered == p.source)
                okay &= not self.engine.active or self.engine.mycall == p.source
            except ValueError:
                okay = False
            if okay:
                self.registered = self.engine.mycall = p.source
                self.engine.listen = True
            self._send(Packet(0, 'X', source=p.source, data=bytes([int(okay)])))
        elif p.kind == 'x':
            if p.source == self.registered:
                self.engine.listen = False
                self.engine.abort('AGWPE callsign unregistered')
                self.registered = ''
        elif p.kind in ('C', 'v'):
            try:
                if p.source != self.registered or not self.registered:
                    raise ValueError('unregistered source')
                if self.engine.active:
                    raise ValueError('one connection already active')
                via = ()
                if p.kind == 'v':
                    if not p.data or not 1 <= p.data[0] <= 8 or len(p.data) != 1 + 10 * p.data[0]:
                        raise ValueError('invalid repeater list')
                    via = tuple(_call(p.data[i:i + 10]) for i in range(1, len(p.data), 10))
                self._originated = True
                self.engine.connect(p.destination, via)
            except ValueError:
                self._originated = False
                self._refuse(p)
        elif p.kind == 'D':
            if (p.pid != 0xF0 or p.source != self.registered
                    or p.destination != self.engine.remote):
                raise ValueError('connected payload identity/PID mismatch')
            self.engine.write(p.data)
        elif p.kind == 'd':
            if {p.source, p.destination} != {self.registered, self.engine.remote}:
                self._refuse(p)
            elif self.engine.active:
                self.engine.disconnect()
            else:
                self._refuse(p)
        elif p.kind in ('Y', 'y'):
            count = (self.engine.outstanding_frames if p.kind == 'y'
                     or {p.source, p.destination} == {self.registered, self.engine.remote} else 0)
            self._send(Packet(0, p.kind, source=p.source, destination=p.destination,
                              data=struct.pack('<I', count)))
        elif p.kind == 'M':
            if p.source != self.registered or not self.registered:
                raise ValueError('unregistered UI source')
            self.engine.unproto(p.destination, p.data)
        else:
            raise ValueError(f'unsupported AGWPE command {p.kind!r}')
