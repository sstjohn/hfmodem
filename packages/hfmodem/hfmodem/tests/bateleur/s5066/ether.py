# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Nodes talking through a simulated channel, in simulated time.

The channel carries bytes, not audio: a transmission reaches every other
node that is not itself transmitting, ``detect_s`` after it begins (the
receiver finding the preamble), its bytes when it ends, each bit wrong
with probability ``ber``; two transmissions that overlap reach no one. It
runs as fast as the nodes compute, so ARQ exchanges of minutes take
milliseconds.

``AudioEther`` is the same arrangement with the waveform in it: each node
is an ``OnAir`` on its own ``BateleurModem``, what one keys is rendered,
passed through ``bateleur.sim.channel`` to each of the others and summed
with whatever else is on the air, and the receivers find it in noise as
they would off a radio. Simulated time still, stepped; a node's modem is
deaf while it keys.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from hfmodem.bateleur import sim
from hfmodem.bateleur.modem import BateleurModem, Mode
from hfmodem.bateleur.spec import FS
from hfmodem.bateleur.s5066 import primitives as P
from hfmodem.bateleur.s5066.node import Node
from hfmodem.bateleur.s5066.onair import OnAir


@dataclass
class _Tx:
    src: Node
    start: float
    end: float
    data: bytes
    heard_by: set = field(default_factory=set)
    spoiled: bool = False


class Client:
    """A client of one node: what the node sends it, kept."""

    def __init__(self, node: Node) -> None:
        self.got: list[P.Primitive] = []
        self.att = node.attach(self.got.append)

    def send(self, p: P.Primitive) -> None:
        self.att.handle(p)

    def bind(self, sap: int, *, mode: int = P.MODE_ARQ, confirm: int = P.CONFIRM_NONE,
             rank: int = 0, retransmissions: int = 0, in_order: bool = True) -> P.Primitive:
        self.send(P.prim(P.BIND_REQUEST, sap=sap, rank=rank,
                         service=P.ServiceType(mode, confirm, in_order,
                                               retransmissions=retransmissions)))
        return self.got[-1]

    def unidata(self, dest: int, dest_sap: int, data: bytes, *, priority: int = 0,
                ttl: float = 0, expedited: bool = False, mode: P.ServiceType | None = None,
                group: bool = False) -> None:
        t = P.EXPEDITED_UNIDATA_REQUEST if expedited else P.UNIDATA_REQUEST
        self.send(P.prim(t, priority=priority, dest_sap=dest_sap, dest=dest, group=group,
                         mode=mode or P.ServiceType(P.MODE_DEFAULT), ttl=ttl, u_pdu=data))

    def of(self, t: int) -> list[P.Primitive]:
        return [p for p in self.got if p.type == t]

    def received(self) -> bytes:
        return b"".join(p["u_pdu"] for p in self.got
                        if p.type in (P.UNIDATA_INDICATION, P.EXPEDITED_UNIDATA_INDICATION))


class Ether:
    def __init__(self, nodes: list[Node], *, ber: float = 0.0, seed: int = 0,
                 detect_s: float = 1.0, step_s: float = 0.05, deaf=()) -> None:
        self.nodes = nodes
        self.ber = ber
        self.rng = random.Random(seed)
        self.detect_s = detect_s
        self.step_s = step_s
        #: (a, b) pairs that cannot hear each other.
        self.deaf = {frozenset(p) for p in deaf}
        self.now = 0.0
        self.air: list[_Tx] = []
        self.log: list[tuple] = []

    def _corrupt(self, data: bytes) -> bytes:
        if self.ber <= 0:
            return data
        out = bytearray(data)
        n = len(out) * 8
        k = 0
        while True:
            # Geometric gaps between errors.
            k += int(self.rng.expovariate(self.ber)) + 1 if self.ber < 1 else 1
            if k > n:
                break
            out[(k - 1) // 8] ^= 1 << ((k - 1) % 8)
        return bytes(out)

    def step(self) -> None:
        now = self.now
        for node in self.nodes:
            if any(tx.src is node and tx.end > now for tx in self.air):
                continue
            got = node.poll(now)
            if got is None:
                continue
            data, secs = got
            tx = _Tx(node, now, now + secs, data)
            for other in self.air:
                if other.end > now and other.src is not node:
                    other.spoiled = tx.spoiled = True
            self.air.append(tx)
            self.log.append((now, node.address, secs, len(data)))
        for tx in list(self.air):
            for node in self.nodes:
                if node is tx.src or frozenset((node.address, tx.src.address)) in self.deaf:
                    continue
                key = id(node)
                if now >= tx.start + self.detect_s and key not in tx.heard_by \
                        and now < tx.end:
                    if not any(t.src is node and t.end > tx.start and t.start < tx.end
                               for t in self.air):
                        tx.heard_by.add(key)
                        node.rx_start(now)
                if now >= tx.end and key in tx.heard_by:
                    tx.heard_by.discard(key)
                    if not tx.spoiled:
                        node.rx_bytes(now, self._corrupt(tx.data))
                    node.rx_end(now)
            if now >= tx.end and not tx.heard_by:
                self.air.remove(tx)
        self.now = round(now + self.step_s, 6)

    def run(self, seconds: float, until=None) -> float:
        """Run until ``seconds`` pass or ``until()`` is true; -> the time."""
        stop = self.now + seconds
        while self.now < stop:
            self.step()
            if until is not None and until():
                break
        return self.now


@dataclass
class _Burst:
    src: int
    start: int                                  # sample
    heard: dict                                 # receiver index -> its audio


class AudioEther:
    """``snr_db`` in 3 kHz, ``preset`` a ``bateleur.sim`` channel; ``step_s``
    how much audio each step hands every receiver."""

    def __init__(self, *, snr_db: float = 20.0, preset: str = "awgn", seed: int = 0,
                 step_s: float = 0.1, max_key_s: float = 60.0) -> None:
        self.snr_db = snr_db
        self.preset = preset
        self.rng = np.random.default_rng(seed)
        self.step = int(round(step_s * FS))
        self.max_key_s = max_key_s
        self.stations: list[OnAir] = []
        self.n = 0                              # samples elapsed
        self.bursts: list[_Burst] = []
        self.keyed_until: list[int] = []
        self.sigma = 0.0
        self.log: list[tuple] = []

    @property
    def now(self) -> float:
        return self.n / FS

    def station(self, address, mode: Mode | str, **node_kw) -> OnAir:
        if isinstance(mode, str):
            mode = Mode.parse(*mode.split())
        i = len(self.stations)
        modem = BateleurModem(lambda audio, rate, i=i: self._send(i, audio),
                              card_rate=FS, max_key_s=self.max_key_s)
        on = OnAir(modem, address, mode, clock=lambda: self.now, **node_kw)
        # Driven here, not by the modem's thread: what it posted, do now.
        while not modem._q.empty():
            modem._q.get()[1]()
        self.stations.append(on)
        self.keyed_until.append(0)
        return on

    def _send(self, i: int, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float64)
        start = max(self.n, self.keyed_until[i])
        if not self.sigma:
            p_sig = np.mean(audio ** 2)
            self.sigma = float(np.sqrt(p_sig / 10 ** (self.snr_db / 10) / 3000.0 * FS / 2))
        heard = {}
        for j in range(len(self.stations)):
            if j != i:
                y = sim.channel(audio, snr_db=np.inf, preset=self.preset,
                                seed=int(self.rng.integers(1 << 31)))
                heard[j] = y
        self.bursts.append(_Burst(i, start, heard))
        self.keyed_until[i] = start + audio.size
        self.log.append((start / FS, i, audio.size / FS))
        return ""

    def run(self, seconds: float, until=None) -> float:
        stop = self.n + int(seconds * FS)
        while self.n < stop:
            self._step()
            if until is not None and until():
                break
        return self.now

    def _step(self) -> None:
        for i, on in enumerate(self.stations):
            if self.keyed_until[i] <= self.n:
                on.step()
        a, b = self.n, self.n + self.step
        for j, on in enumerate(self.stations):
            if self.keyed_until[j] > a:
                chunk = np.zeros(self.step)         # keying: the receiver is muted
            else:
                chunk = self.rng.normal(0, self.sigma, self.step) if self.sigma \
                    else np.zeros(self.step)
                for burst in self.bursts:
                    y = burst.heard.get(j)
                    if y is None:
                        continue
                    lo, hi = max(a, burst.start), min(b, burst.start + y.size)
                    if lo < hi:
                        chunk[lo - a:hi - a] += y[lo - burst.start:hi - burst.start]
            on.modem._audio(chunk.astype(np.float32))
        self.bursts = [x for x in self.bursts
                       if x.start + max((y.size for y in x.heard.values()), default=0) > b]
        self.n = b
