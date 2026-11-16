# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from hfmodem.shrike import arq, pactor1, placement, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol


BODY = bytes(range(32, 127)) * 4


class IO(arq.ArqIO):
    def __init__(self, *, deferred=True):
        self.deferred = deferred
        self.refuse_cs = self.refuse_packet = False
        self.controls, self.packets, self.logs = [], [], []
        self.delivered = bytearray()
        self.closed = 0

    def send_cs(self, cs_index):
        self.controls.append(cs_index)
        return arq.REFUSED if self.refuse_cs else None

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((sl, bytes(payload), status, breakin))
        return arq.REFUSED if self.refuse_packet else len(payload)

    def defer_rx_close(self):
        return self.deferred

    def deliver(self, payload):
        self.delivered.extend(payload)

    def disconnected(self):
        self.closed += 1

    def log(self, text):
        self.logs.append(text)


def linked(*, role=arq.IRS, deferred=True, long_cycle=True):
    io = IO(deferred=deferred)
    fsm = arq.PactorArq(io, arq.ArqConfig(long_cycle=long_cycle,
                                        speed_up_after=1000))
    fsm._enter_connected()
    fsm.role = role
    return fsm, io


class Sink(arq.ArqIO):
    def __init__(self):
        self.sent = []
        self.delivered = []
        self.buffers = []
        self.refuse = False

    def send_packet(self, sl, payload, status, breakin=False):
        self.sent.append(("packet", bytes(payload[:3]), status, breakin))
        return arq.REFUSED if self.refuse else min(3, len(payload))

    def send_cs(self, index):
        self.sent.append(("cs", index))
        return arq.REFUSED if self.refuse else None

    def defer_rx_close(self):
        return True

    def deliver(self, data):
        self.delivered.append(data)

    def buffer(self, size):
        self.buffers.append(size)


class Seam:
    """A transmit seam that keys nothing and remembers every codeword.

    `send_p1_cs` exists so `PtcHost.send_cs` routes through `_counter_cs_for`:
    a seam without it takes the FSM's own alphabet unchanged, and the
    alternation is exactly what these scenes are reading.
    """

    # Recording a word is the synchronous emission in this fixture, not a
    # deferred queue requiring a later driver on_cs_emitted callback.
    defer_p3_cs = False

    def __init__(self) -> None:
        self.words: list[int] = []
        self.packets: list[tuple[int, int]] = []
        self.slots_used: list[int] = []

    def attach(self, host) -> None: ...

    def p3_slot_keyed(self) -> bool | None:
        return None

    def send_cs(self, index: int, *, p1_codeword=False, unread=False, maxdown=False,
                packet_phase=None, cycle_long=None) -> None:
        self.words.append(index)

    def send_p1_cs(self, index: int) -> None:
        self.words.append(index)

    def send_packet(self, sl, payload, status, breakin=False) -> int:
        self.packets.append((sl, status))
        return len(payload)

    def __getattr__(self, name):
        return lambda *a, **k: None


class Sender(Seam):
    def __init__(self):
        super().__init__()
        self.frames = []
        self.refuse = False

    def send_packet(self, sl, payload, status, breakin=False):
        if self.refuse:
            return arq.REFUSED
        n = (placement.CHANGEOVER if breakin else placement.SPEED_PATHS[sl]).crc_bytes-3
        self.frames.append((sl, status, payload[:n]))
        return n


def sender(enabled=True, *, payload=b'\x01/W' + BODY, hostmode=False):
    seam = Sender()
    host = PtcHost(peer=seam, mycall='W9SSJ')
    if hostmode:
        host.feed(b'\x1bJHOST4\r')
    host.arq.role, host.arq.dxcall = arq.ISS, 'K0NTS'
    host.arq._enter_connected()
    host.protocol = spec.Protocol.PACTOR3
    host.p3_tx_gear_hold = enabled
    a = host.arq
    a.cfg.long_cycle = False
    a.on_host_data(payload)
    wire = bytes(a._outbuf)
    a._inflight = arq._Packet(0,wire[:3],1,breakin=True)
    a._next_seq = 1
    del a._outbuf[:3]
    return host,seam


def control(host, cs):
    host.on_rx_event(rxfront.Event(0,'cs','test',protocol=spec.Protocol.PACTOR3,cs=cs-1))


class Keyed:
    """Everything the station puts on the air, in the protocol it went out in."""

    def __init__(self) -> None:
        self.packets: list[tuple[Protocol, bytes]] = []
        self.cs: list[tuple[Protocol, int]] = []
        self.host: PtcHost | None = None

    def attach(self, host: PtcHost) -> None:
        self.host = host

    def pump(self) -> None: ...
    def cycle(self) -> None: ...
    def connect_burst(self, mycall: str, dxcall: str) -> None: ...

    def _packet(self, payload: bytes) -> None:
        self.packets.append((self.host.protocol, bytes(payload)))

    def send_p1_packet(self, payload, baud, seq, **kw) -> None:
        self._packet(payload)

    def send_p1_breakin(self, payload, baud, seq, **kw) -> None:
        self._packet(payload)

    def send_packet(self, sl, payload, status, breakin=False) -> None:
        self._packet(payload)

    def send_p1_cs(self, index: int) -> None:
        self.cs.append((Protocol.PACTOR1, index))

    def p3_slot_keyed(self) -> bool | None:
        return None

    def send_cs(self, index: int, *, p1_codeword=False, unread=False, maxdown=False,
                packet_phase=None, cycle_long=None) -> None:
        self.cs.append((Protocol.PACTOR3, index))

    @property
    def p3(self) -> list[bytes]:
        return [pl for proto, pl in self.packets if proto is Protocol.PACTOR3]


def cs_event(cs: int, protocol: str = "PACTOR-1"):
    return rxfront.Event(0.1, "cs", f"CS{cs + 1}", protocol=protocol,
                         cs=cs, sense=0)


def calling_station() -> tuple[PtcHost, Keyed]:
    """A linked ISS at 100 Bd with its first packet -- the callsign announcement
    `_answer_link_setup` queues -- already on the air and unacknowledged."""
    keyed = Keyed()
    host = PtcHost(peer=keyed, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(cs_event(pactor1.CS_SPEED))     # CS4: the link runs 100 Bd
    host.tick()
    return host, keyed


def grant():
    return rxfront.Event(0.1, "unassigned", "0x59A", protocol=Protocol.PACTOR1,
                         spare=pactor1.CS_59A, sense=0)


def granted():
    host, keyed = calling_station()
    host.p1_grant_only = True
    host.arq.on_host_data(b"pending application bytes")
    host.on_rx_event(grant())
    host.tick()  # Finish the initial grant/entry cycle before its replies arrive.
    assert host.arq.entry_pending
    return host, keyed


def repeat(host):
    host.on_rx_event(grant())
    host.tick()


class Frames(Seam):
    def __init__(self):
        super().__init__()
        self.frames = []

    def send_packet(self, sl, payload, status, breakin=False):
        self.frames.append((sl, status & 3, bytes(payload)))
        return len(payload)

    def send_long_packet(self, sl, payload, status):
        return self.send_packet(sl, payload, status)



def pending(sl, seq, prefix, queue, *, long_cycle=False):
    seam = Frames()
    host = PtcHost(peer=seam, mycall='W9SSJ')
    host.arq._enter_connected()
    host.protocol = spec.Protocol.PACTOR3
    a = host.arq
    a.role = arq.ISS
    a.cfg.long_cycle = long_cycle
    a.cycle_long = long_cycle
    a._sl = a._peer_sl = a._established_sl = sl
    a._inflight = arq._Packet(seq, prefix, sl)
    a._next_seq = (seq + 1) % 4
    a._outbuf = bytearray(queue)
    a._buffer_raw = len(prefix) + len(queue)
    a.payload_ledger.queue(prefix + queue)
    seam.frames.append((sl, seq, prefix))
    return host, seam



def remaining(a):
    return (a._inflight.payload if a._inflight is not None else b'') + bytes(a._outbuf)
