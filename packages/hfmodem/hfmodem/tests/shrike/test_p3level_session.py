# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The peer's speed level, through the receiver a live link actually runs.

`onair._SessionRx._p3_packet` is the only in-session PACTOR-3 data reader, and
both halves of it were pinned to one geometry: `rxfront.decode_expected_packet`
scanned `placement.HEADER` -- speed level 2, short cycle, home tone order -- and
`rxfront.SyncedRx._packet_at_lock` read the packet's header block for the swap
and the cycle length and then threw `header.levels` away. Measured through
`deep_scan` on a CONNECTED PACTOR-3 host, one rendered packet per row, that
reader delivered SL2 short and nothing else, while `p3rx.decode_p3_packets` read
all eleven off the same audio. Level 3 is `arq.ArqConfig.entry_sl` and the level
the reference session runs, so the receiving half of a mail exchange was closed
for the levels the peer is most likely to offer.

The entry point here is the production one for the same reason
`test_p1memory_session` uses it: `deep_scan` is what the cycle calls before its
key, `upgrade_scan` is what runs behind our own carrier for the protocol a peer
might be leading into, and a test that called `rxfront` directly would pass over
a session that never reaches either.

Six claims, each its own arm:

  * every level, both cycle lengths, both carrier arrangements, delivered to the
    host with the peer's own level reported;
  * a lock taken from one of those reads the next cycle at the same level, so
    the tracked path is not the pinned one wearing a new coat;
  * speed level 1 -- case 0, the entry answer's own level -- arrives on the live
    path, on a PACTOR-3 link and through `upgrade_scan` on a PACTOR-1 one;
  * the reported level is what `arq._gear_cs` climbs from, which is what makes
    it load-bearing rather than a log line;
  * the session's `p3rx.FieldMemory` spans cycles, so two copies that each fail
    alone deliver the field;
  * ...and it does not outlive the link.

Run:  python -m pytest hfmodem/tests/shrike/test_p3level_session.py
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.shrike import onair, placement, spec
from hfmodem.shrike.arq import IRS, State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol

FS = onair.FS
SNR_DB = 20.0
MEMORY_SNR_DB = -4.0
"""`test_p3memory`'s figure for the punctured levels: past the single-shot cliff
at speed level 5 and inside combining's reach."""


class _Seam:
    """A transmit seam that records instead of keying."""

    def __init__(self) -> None:
        self.sent: list = []

    def attach(self, host) -> None:
        pass

    def connect_burst(self, mycall, dxcall) -> None:
        self.sent.append(("connect", dxcall))

    def send_cs(self, i) -> None:
        self.sent.append(("cs", i))

    def send_p1_cs(self, i) -> None:
        self.sent.append(("p1cs", i))

    def send_p1_packet(self, payload, baud, packet_count, **kw) -> None:
        self.sent.append(("p1pkt", payload))

    def send_packet(self, sl, payload, status, breakin=False) -> None:
        self.sent.append(("pkt", sl, payload))

    def send_long_packet(self, sl, payload, status, **kw) -> None:
        self.sent.append(("long", sl, payload))

    def deliver(self, data) -> None:
        pass

    def buffer(self, n) -> None:
        pass

    def pump(self) -> None:
        pass

    def cycle(self) -> None:
        pass


def _linked(protocol: Protocol = Protocol.PACTOR3) -> PtcHost:
    host = PtcHost(peer=_Seam(), mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.role, host.arq.dxcall = IRS, "WS8EOC"
    host.arq._enter_connected()
    host.protocol = protocol
    host.peer.sent.clear()
    return host


class _Session:
    """A `_SessionRx` on a linked host, keeping the events it delivered."""

    def __init__(self, protocol: Protocol = Protocol.PACTOR3) -> None:
        self.host = _linked(protocol)
        self.rx = onair._SessionRx(self.host, tag="TEST")
        self.events: list = []
        deliver = self.host.on_rx_event

        def record(ev):
            self.events.append(ev)
            return deliver(ev)

        self.host.on_rx_event = record

    def cycle(self, audio: np.ndarray, upgrade: bool = False) -> None:
        self.rx.new_cycle()
        (self.rx.upgrade_scan if upgrade else self.rx.deep_scan)(audio)

    @property
    def received(self) -> bytes:
        return bytes(self.host.channel(self.host.ptchn).rx)

    @property
    def levels(self) -> list:
        return [ev.packet[0] for ev in self.events if ev.kind == "packet"]


def _payload(sl: int, long_cycle: bool) -> bytes:
    n = (spec.SPEED_LEVELS[sl].payload_long if long_cycle
         else spec.SPEED_LEVELS[sl].payload_short)
    return bytes((3 * i + sl) & 0x5F | 0x20 for i in range(n))


def _cycle_audio(audio: np.ndarray, snr_db: float = SNR_DB,
                 seed: int = 0) -> np.ndarray:
    """One receive window: the packet with a quiet lead and tail, in noise."""
    quiet = np.zeros(int(0.4 * FS))
    x = np.concatenate([quiet, np.asarray(audio, np.float64), quiet])
    sigma = float(np.sqrt(np.mean(np.asarray(audio, np.float64) ** 2))) \
        / 10 ** (snr_db / 20)
    return (x + np.random.default_rng(seed).normal(0, sigma, x.size)
            ).astype(np.float32)


GEOMETRIES = [(sl, long_cycle, swapped)
              for sl in sorted(placement.SPEED_PATHS)
              for long_cycle in (False, True)
              for swapped in (False, True)
              if not (long_cycle and sl == 1)]


@pytest.mark.parametrize("sl,long_cycle,swapped", GEOMETRIES)
def test_every_geometry_reaches_the_host(sl: int, long_cycle: bool,
                                         swapped: bool) -> None:
    """One rendered packet per row of the table the pinned reader failed."""
    payload = _payload(sl, long_cycle)
    sess = _Session()
    sess.cycle(_cycle_audio(placement.link_packet(
        sl, payload, 0x21, swapped=swapped, long_cycle=long_cycle)))
    assert sess.received == payload, \
        f"SL{sl} {'long' if long_cycle else 'short'} " \
        f"{'swapped' if swapped else 'home'}: {sess.received[:24]!r}"
    assert sess.levels == [sl], sess.levels


@pytest.mark.parametrize("sl", [1, 3, 5])
def test_a_lock_reads_the_next_cycle_at_the_same_level(sl: int) -> None:
    """The tracked path, which had the pin in its own line rather than a scan's.

    The first cycle is scanned and takes the lock; the second must come off it,
    at the peer's level and not at the home geometry's.
    """
    payload = _payload(sl, False)
    sess = _Session()
    for seed in (0, 1):
        sess.cycle(_cycle_audio(placement.link_packet(sl, payload, 0x21),
                                seed=seed))
    assert sess.levels == [sl, sl], sess.levels
    assert sess.rx.sync.tracked == 1, \
        f"{sess.rx.sync.tracked} tracked reads, {sess.rx.sync.locks} locks"


def test_the_peers_entry_answer_is_read_on_both_links() -> None:
    """Speed level 1 is case 0, and case 0 had no in-session reader at all.

    It is the level the upgrade enters at, so it arrives in two places: on a
    PACTOR-3 link, where `deep_scan` runs before the key, and on a PACTOR-1 one,
    where the PACTOR-3 reader is `upgrade_scan`'s and runs behind our carrier.
    """
    payload = _payload(1, False)
    audio = _cycle_audio(placement.link_packet(1, payload, 0x21))
    for protocol, upgrade in ((Protocol.PACTOR3, False),
                              (Protocol.PACTOR1, True)):
        sess = _Session(protocol)
        sess.cycle(audio, upgrade=upgrade)
        assert sess.received == payload and sess.levels == [1], \
            f"{protocol}: {sess.levels} {sess.received[:16]!r}"


def test_the_gear_request_climbs_from_the_peers_level() -> None:
    """What the level being right is FOR.

    `arq._gear_cs` counts clean cycles and asks for `sl + 1`; reported as 2
    whatever the peer sent, it asked a level 3 peer to move to level 3. The
    check is the log line, which names the level the request is against.
    """
    sess = _Session()
    payload = _payload(3, False)
    for seed in range(sess.host.arq.cfg.speed_up_after):
        sess.cycle(_cycle_audio(
            placement.link_packet(3, payload, spec.status_byte(seed % 4)),
            seed=seed))
    asked = [m for m in sess.host.log_lines if "clean run" in m]
    assert asked and "SL4" in asked[-1], (asked, sess.host.log_lines)


def _marginal(k: int) -> np.ndarray:
    """Copy `k` of one speed level 5 field, past the single-shot cliff.

    The carrier swap alternates with the cycle, which is what the memory has to
    unwind per copy before a sum means anything."""
    payload = _payload(5, False)
    return _cycle_audio(
        placement.link_packet(5, payload, 0x21, swapped=bool(k & 1)),
        snr_db=MEMORY_SNR_DB, seed=k)


def test_two_copies_combine_through_the_session_memory() -> None:
    sess = _Session()
    sess.cycle(_marginal(0))
    assert not sess.received and not sess.levels, \
        f"premise broken: a copy decodes alone -- {sess.levels}"
    sess.cycle(_marginal(1))
    assert sess.received == _payload(5, False), sess.received[:24]
    assert sess.levels == [5], sess.levels


def test_the_memory_does_not_outlive_the_link() -> None:
    """A disconnect ends the run of copies: nothing is repeating them any more."""
    sess = _Session()
    sess.cycle(_marginal(0))
    sess.host.arq._finish_disconnected()
    sess.rx.new_cycle()
    sess.host.arq.role, sess.host.arq.dxcall = IRS, "WS8EOC"
    sess.host.arq._enter_connected()
    sess.host.protocol = Protocol.PACTOR3
    sess.cycle(_marginal(1))
    assert not sess.received and not sess.levels, \
        f"the copy held before the link went down is gone: {sess.levels}"
    assert sess.host.arq.state == State.CONNECTED
