# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The called station grants the PACTOR-3 upgrade, as a stock called station does.

Stock PACTOR has the ANSWERING station command the upgrade (docs/protocols/
pactor/pactor3.md sec 17.1, `PIII_Complete_1.wav`): the caller's PACTOR-1
announcement sets status bits 4 and 5 (0x35 there, 0x31 at W4DNA), the called
station answers it with `0x59A` in the ordinary answer slot, the caller keys a
speed-level-1 PACTOR-3 entry packet in its next transmit slot, and the called
station answers that in PACTOR-3. Here both ends are ours, every burst crosses as
audio through the shipped receivers, and the caller announces the way
`onair.RadioTx` does by default (`onair.P1_STATUS_ANNOUNCE`).
"""
from __future__ import annotations

import dataclasses

import pytest

from hfmodem.shrike import onair, p2rx, pactor1, pactor2, placement, rxfront, spec
from hfmodem.shrike.arq import IRS, State
from hfmodem.shrike.ptc import GRANT_BUDGET, PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_qso import AudioLink, AudioSide, rx_of

pytestmark = pytest.mark.filterwarnings("ignore")


class AnnouncingSide(AudioSide):
    """`AudioSide` with what `onair.RadioTx` adds on the air: status bits 4-5 on
    PACTOR-1 data packets, the pinned entry-packet renders, and PACTOR-2."""

    bits45 = onair.P1_STATUS_ANNOUNCE

    def send_p1_packet(self, payload, baud, packet_count, *, header=None,
                       changeover_request=False, qrt=False, data_type=0):
        bits45 = 0 if qrt else self.host.p1_announcement_bits(self.bits45)
        flags = (" BK" if changeover_request else "") + (" QRT" if qrt else "")
        self._tx(f"P1 pkt#{packet_count} {len(payload)}B bits45={bits45}{flags}",
                 pactor1.packet_signal(payload, baud, packet_count=packet_count,
                                       header=header, bits45=bits45,
                                       changeover_request=changeover_request,
                                       qrt=qrt, invert=self._flip(),
                                       data_type=data_type))

    def send_entry_packet(self, sl, payload, status, *, acquire=False):
        self._flip()
        self._tx(f"SL{sl} ENTRY status=0x{status:02x}",
                 placement.link_packet(sl, payload, status, swapped=False,
                                       acquire=acquire,
                                       flush=placement.ENTRY_FLUSH))
        return placement.SPEED_PATHS[sl].crc_bytes - 3

    def send_p1_cs(self, index):
        name = {pactor1.CS_59A: "0x59A", pactor1.CS_6A9: "0x6A9"}.get(index)
        self._tx(name or f"P1 CS{index + 1}",
                 pactor1.control_signal(index, invert=self._flip()))

    # PACTOR-2, as `onair.RadioTx` renders it.
    def _p2(self, what, path, payload, status, swapped):
        n = path.crc_bytes - 3
        field = pactor2.build_field(placement.field_info(payload, n, status), path)
        k = pactor2.marker_index(path.level, path.n_symbols == 320, status & 1)
        self._tx(what, pactor2.data_burst(field, path, swapped=swapped, k=k))
        return n

    def send_p2_entry_packet(self, payload, status):
        self._flip()
        return self._p2(f"P2 SL1 ENTRY status=0x{status:02x}", pactor2.PATHS[0],
                        payload, status, False)

    def send_p2_packet(self, sl, payload, status):
        return self._p2(f"P2 SL{sl} pkt {len(payload)}B", pactor2.PATHS[sl - 1],
                        payload, status, self._flip())

    def send_p2_long_packet(self, sl, payload, status):
        return self._p2(f"P2 SL{sl} LONG pkt {len(payload)}B",
                        pactor2.PATHS_LONG[sl - 1], payload, status, self._flip())

    def send_p2_cs(self, index):
        self._tx(f"P2 CS{index + 1}",
                 pactor2.control_signal(index, swapped=self._flip()))


def _p2_packet(heard):
    """`onair._SessionRx._p2_packet`: the frame marker places the burst."""
    got = p2rx.decode_expected_burst(heard)
    if got is None:
        return None
    t, path, field, _ = got
    info = field[:path.crc_bytes - 2]
    return rxfront.Event(t, "packet", f"P2 {path.name}", protocol=Protocol.PACTOR2,
                         packet=(path.level + 1, info[-1],
                                 spec.field_payload(info[:-1]), True))


def _p2_cs(heard, quiet_n):
    """A PACTOR-2 codeword at the instant its burst's reference pulse lands,
    in either carrier arrangement, as a station holding the raster reads it."""
    at = quiet_n + pactor2.pulse_lead()
    for swapped in (False, True):
        got = p2rx.control_signal_at(heard, at, swapped=swapped)
        if got is not None and got[1] == 0:
            return rxfront.Event(0.0, "cs", f"P2 CS{got[0] + 1}",
                                 protocol=Protocol.PACTOR2, cs=got[0])
    return None


class GrantLink(AudioLink):
    """`AudioLink` with the answering station's receiver as onair runs it.

    A PACTOR-1 station holding a link reads only PACTOR-1 before its key; every
    other protocol is `onair._SessionRx.upgrade_scan`, run behind its own
    carrier, PACTOR-3 and then PACTOR-2. So the cycle carrying the entry packet
    is answered as a silent one -- the grant again -- and the entry is read
    after that answer has gone out, too late to be answered in its own cycle.
    One transmission is one frame (`_note_p3_frame`), where the blind scan
    behind `AudioLink._receive` can report a speed-level-1 packet twice.

    `deaf_behind_carrier` drops those late reads: an entry packet the answerer
    never acquires."""

    SIDE = AnnouncingSide
    deaf_behind_carrier = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.behind_carrier: list = []

    def _receive(self, heard, dst):
        if (dst is self.b and dst.arq.state == State.CONNECTED
                and dst.protocol is Protocol.PACTOR1 and dst.arq.role == IRS):
            ev = rxfront.decode_expected_p1_packet(heard)
            if ev is not None:
                return [ev]
            ev = rxfront.decode_expected_packet(heard)
            if ev is not None:
                self.sync[dst.mycall].observe(ev)
                self._note([ev])
            else:
                ev = _p2_packet(heard)
            if ev is not None and not self.deaf_behind_carrier:
                self.behind_carrier.append(ev)
            return []
        if dst.arq.state == State.CONNECTED and dst.protocol is Protocol.PACTOR2:
            ev = rxfront.decode_expected_p1_packet(heard)
            if ev is None:
                ev = (_p2_packet(heard) if dst.arq.role == IRS
                      else _p2_cs(heard, round(self.QUIET_S * rxfront.FS)))
            return [] if ev is None else [ev]
        return super()._receive(heard, dst)

    def exchange(self, cycles: int) -> None:
        for _ in range(cycles):
            self._carry(self.a_io, self.b, "A->B")
            self.b.tick()
            keyed = len(self.b_io.pending)
            for ev in self.behind_carrier:
                self.b.on_rx_event(ev)
            self.behind_carrier.clear()
            # An answer keyed by a late read has no slot left in this cycle.
            del self.b_io.pending[keyed:]
            self._carry(self.b_io, self.a, "B->A")
            self.a.tick()


MESSAGE = b"DE K7ABC -- the caller's traffic, long enough to want PACTOR-3 " * 12
SHORT = MESSAGE[:120]


def _link(*, grant_only: bool) -> GrantLink:
    link = GrantLink("K7ABC", "W9SSJ", verbose=False)
    link.a.p1_grant_only = grant_only
    link.a.arq.on_host_connect("K7ABC", "W9SSJ")
    link.exchange(2)
    assert link.a.arq.state == link.b.arq.state == State.CONNECTED
    assert link.b.arq.answering and link.b.arq.role == IRS
    return link


@pytest.mark.parametrize("grant_only", [True, False],
                         ids=["grant-only-caller", "default-caller"])
def test_our_answerer_grants_and_our_caller_reaches_pactor3(grant_only):
    link = _link(grant_only=grant_only)
    link.a.arq.on_host_data(MESSAGE)
    for _ in range(30):
        link.exchange(1)
        if not link.a.arq._outbuf and link.a.arq._inflight is None:
            break

    assert "0x59A" in link.b_io.sent, link.b_io.sent
    assert any("0x59A grant -> PACTOR-3 entry packet" in line
               for line in link.a.log_lines), link.a.log_lines[-20:]
    assert not any("uninvited" in line for line in link.a.log_lines
                   if "upgrade" in line), link.a.log_lines
    assert any("ENTRY" in w for w in link.a_io.sent), link.a_io.sent
    assert link.a.protocol is Protocol.PACTOR3, link.a.protocol
    assert link.b.protocol is Protocol.PACTOR3, link.b.protocol
    assert link.p3_rx, "no PACTOR-3 packet was demodulated"
    assert MESSAGE in rx_of(link.b)
    # Entered at speed level 1, the granted level, and climbed off it on the
    # answerer's own CS4s: level 1 is for acquisition, not traffic.
    assert link.p3_rx[0] == 1 and max(link.p3_rx) >= 3, link.p3_rx
    # The grant goes out twice: once for the announcement, and again in the
    # cycle carrying the entry, which a PACTOR-1 receiver cannot read before
    # its key. The caller re-keys the entry on that repeat and the answerer,
    # following by then, answers it in PACTOR-3 a cycle late. A property of
    # this station's receive order, not of a stock called station.
    assert link.b_io.sent.count("0x59A") == 2, link.b_io.sent


def test_an_unannounced_caller_is_not_granted():
    """bits 4-5 clear (0x01) draws an ordinary acknowledgement at every gateway
    on record (pactor1-data-packets.md, the KB5LZK 2x2), so it does here."""
    link = GrantLink("K7ABC", "W9SSJ", verbose=False)
    link.a_io.bits45 = 0
    link.a.p1_grant_only = True
    link.a.arq.on_host_connect("K7ABC", "W9SSJ")
    link.exchange(2)
    link.a.arq.on_host_data(SHORT)
    link.exchange(12)
    assert "0x59A" not in link.b_io.sent and "0x6A9" not in link.b_io.sent
    assert link.a.protocol is link.b.protocol is Protocol.PACTOR1
    assert SHORT in rx_of(link.b)


def test_a_pactor1_ceiling_answers_an_announcement_in_pactor1():
    link = GrantLink("K7ABC", "W9SSJ", verbose=False)
    link.b.stay_in_pactor1 = True
    link.a.p1_grant_only = True
    link.a.arq.on_host_connect("K7ABC", "W9SSJ")
    link.exchange(2)
    link.a.arq.on_host_data(SHORT)
    link.exchange(12)
    assert "0x59A" not in link.b_io.sent
    assert SHORT in rx_of(link.b)


class _Keys:
    def __init__(self):
        self.p1, self.p3 = [], []

    def send_p1_cs(self, index):
        self.p1.append(index)

    def send_cs(self, index):
        self.p3.append(index)

    def send_p1_packet(self, *a, **kw):
        pass

    def connect_burst(self, *a):
        pass

    def attach(self, host):
        pass

    def pump(self):
        pass

    cycle = pump


def _answering(mylevel=None, grant_pactor2=False):
    keys = _Keys()
    host = PtcHost(peer=keys, mycall="W9SSJ")
    if mylevel is not None:
        host.settings["MYLEVEL"] = mylevel
    host.grant_pactor2 = grant_pactor2
    host.arq.on_host_listen(True)
    host.arq.on_rx_connect("", "W9SSJ")
    assert host.arq.answering and host.arq.state == State.CONNECTED
    return host, keys


def _announce(host, status):
    host.on_rx_event(rxfront.Event(0.0, "packet", "", protocol="PACTOR-1",
                                   packet=(0, status, b"1k7abc\r", True)))


# The table in `PtcHost._read_announcement`: what the caller asks, capped by
# MYLevel, with the 0x6A9 row behind `grant_pactor2`.
@pytest.mark.parametrize("status,mylevel,p2,word", [
    (0x31, None, False, pactor1.CS_59A),
    (0x31, "3", True, pactor1.CS_59A),
    (0x31, "2", False, None),
    (0x31, "2", True, pactor1.CS_6A9),
    (0x11, None, False, None),
    (0x11, None, True, pactor1.CS_6A9),
    (0x11, "1", True, None),
    (0x21, None, True, None),
    (0x01, None, True, None),
    (0x71, None, True, None),   # a changeover request rides it: no grant
])
def test_the_grant_follows_what_the_caller_asks_and_our_ceiling(status, mylevel,
                                                                 p2, word):
    host, keys = _answering(mylevel, p2)
    _announce(host, status)
    if word is None:
        assert not {pactor1.CS_59A, pactor1.CS_6A9} & set(keys.p1), keys.p1
    else:
        assert keys.p1[-1] == word, keys.p1


def _announce_cycle(host, keys, status):
    """The announcement and its answer, then the tick closing that cycle --
    which keys nothing, the answer having gone out with the read."""
    _announce(host, status)
    n = len(keys.p1)
    host.tick()
    assert len(keys.p1) == n, keys.p1[n:]


def _tick(host, keys):
    """One cycle in which nothing was read: exactly one codeword goes out."""
    n = len(keys.p1)
    host.tick()
    assert len(keys.p1) == n + 1, keys.p1[n:]
    return keys.p1[-1]


def test_silence_behind_a_grant_is_the_entry_we_asked_for_up_to_the_budget():
    """A stock called station keys the word once a cycle while it waits for the
    entry packet (pactor3.md sec 17.1). A cycle that read nothing then is the
    caller keying a waveform we commanded, so it does not spend the link's
    silence budget until `GRANT_BUDGET` grants are out -- past our caller's
    `ENTRY_GRANT_CYCLES` campaign, so that campaign ends first. Then the
    ordinary held codeword, and the ordinary silence budget, take over."""
    host, keys = _answering()
    _announce_cycle(host, keys, 0x31)
    assert keys.p1 == [pactor1.CS_ACK_A, pactor1.CS_59A]
    for _ in range(GRANT_BUDGET - 1):
        assert _tick(host, keys) == pactor1.CS_59A
        assert host.arq.state == State.CONNECTED
    assert host._grants_keyed == GRANT_BUDGET
    held = _tick(host, keys)
    assert held in (pactor1.CS_ACK_A, pactor1.CS_ACK_B), keys.p1[-3:]
    n = len(keys.p1)
    for _ in range(host.arq.cfg.max_retries + 4):
        if host.arq.state != State.CONNECTED:
            break
        host.tick()
    assert set(keys.p1[n:]) == {held}, keys.p1[n:]
    assert host.arq.state != State.CONNECTED, "the silence budget never ran"


def test_a_grant_past_the_budget_is_not_keyed_even_without_a_new_packet():
    host, keys = _answering()
    _announce_cycle(host, keys, 0x31)
    host._grants_keyed = GRANT_BUDGET
    assert _tick(host, keys) in (pactor1.CS_ACK_A, pactor1.CS_ACK_B)
    assert host._grant_word is None


def _entry(counter, sl=1, field=bytes.fromhex("0f8f87c7c3")):
    return rxfront.Event(0.0, "packet", "", protocol="PACTOR-3",
                         packet=(sl, 0x18 | counter, field, True))


@pytest.mark.parametrize("announce,p2,counter,follows", [
    (0x31, False, 2, True),    # announcement counter 1 -> entry counter 2
    (0x32, False, 3, True),
    (0x31, False, 3, False),   # not the packet after the one we granted
    (0x31, False, 1, False),
    (0x11, True, 2, False),    # 0x6A9 granted PACTOR-2, not PACTOR-3
    (0x01, False, 2, False),   # nothing granted
])
def test_only_the_granted_entry_moves_the_link(announce, p2, counter, follows):
    host, keys = _answering(grant_pactor2=p2)
    _announce(host, announce)
    host.on_rx_event(_entry(counter))
    assert (host.protocol is Protocol.PACTOR3) == follows, host.log_lines[-4:]


def test_a_granted_speed_level_2_packet_still_follows():
    host, keys = _answering()
    _announce(host, 0x31)
    host.on_rx_event(_entry(2, sl=2, field=b""))
    assert host.protocol is Protocol.PACTOR3


def test_no_grant_once_pactor3_is_ruled_out_for_the_link():
    host, keys = _answering()
    _announce(host, 0x31)
    host.on_rx_event(_entry(2))
    assert host.protocol is Protocol.PACTOR3
    # The caller comes back in PACTOR-1 with a new packet: a verdict on P3.
    host.on_rx_event(rxfront.Event(0.0, "packet", "", protocol="PACTOR-1",
                                   packet=(0, 0x33, b"more", True)))
    assert host.protocol is Protocol.PACTOR1
    assert Protocol.PACTOR3 in host._ruled_out
    n = len(keys.p1)
    host.on_rx_event(rxfront.Event(0.0, "packet", "", protocol="PACTOR-1",
                                   packet=(0, 0x30, b"again", True)))
    assert pactor1.CS_59A not in keys.p1[n:], keys.p1[n:]


def test_a_recorded_stock_caller_is_granted_and_its_entry_followed():
    """DL6MAA's caller (`PIII_Complete_1.wav`), its own bursts through the
    shipped receiver into our answering station: the 0x35 announcement at
    3.40 s draws our `0x59A`, and its speed-level-1 entry packet at 4.76 s --
    status 0x1a, counter 2 -- moves the link to PACTOR-3 and is answered there.
    The recorded called station's own words (its 0x59A at 4.412 s, its
    greeting from 5.56 s) are left out: we are standing in for it."""
    from hfmodem.tests.shrike import test_p3_upgrade as rec
    if not rec.FULL.exists():
        pytest.skip(f"{rec.FULL} is not on this machine")
    keys = _Keys()
    host = PtcHost(peer=keys, mycall=rec.DXCALL)
    host.arq.on_host_listen(True)
    caller = [e for e in rec._events(rec.FULL) if e.t < 5.0
              and e.kind in ("connect", "packet")]
    # The receiver hands an empty template field up as b''. What the caller
    # keyed is the five template bytes (pactor3.md sec 17.1, `0f 8f 87 c7 c3
    # 1a 66 89`), so that is what reaches the link layer here.
    sl, status, field, ok = caller[-1].packet
    assert (sl, status, field, ok) == (1, 0x1a, b"", True)
    caller[-1] = dataclasses.replace(
        caller[-1], packet=(sl, status, bytes.fromhex("0f8f87c7c3"), ok))
    for ev in caller:
        host.on_rx_event(ev)
        if ev.kind == "packet" and ev.protocol == Protocol.PACTOR1:
            assert ev.packet[1] == 0x35
            assert keys.p1[-1] == pactor1.CS_59A, keys.p1
            assert host.protocol is Protocol.PACTOR1
    entry = caller[-1]
    assert entry.protocol == Protocol.PACTOR3 and entry.packet[:2] == (1, 0x1a)
    assert host.protocol is Protocol.PACTOR3
    assert keys.p3, "the entry packet was not answered in PACTOR-3"
    assert keys.p1.count(pactor1.CS_59A) == 1


def _p2_caller(link):
    link.a.pactor2_only = True
    link.a_io.bits45 = 1
    link.b.grant_pactor2 = True


@pytest.mark.parametrize("p2", [False, True], ids=["0x59A", "0x6A9"])
def test_a_granted_entry_the_answerer_never_reads_ends_in_a_working_pactor1(p2):
    """The caller takes our grant as the acknowledgement of its announcement,
    keys an entry this answerer never acquires, and comes back to PACTOR-1 with
    its next packet. That packet is the end of the attempt: no more grants for
    the protocol, and the PACTOR-1 link carries the traffic."""
    link = GrantLink("K7ABC", "W9SSJ", verbose=False)
    link.deaf_behind_carrier = True
    if p2:
        _p2_caller(link)
    link.a.arq.on_host_connect("K7ABC", "W9SSJ")
    link.exchange(2)
    link.a.arq.on_host_data(SHORT)
    link.exchange(40)
    word = "0x6A9" if p2 else "0x59A"
    assert word in link.b_io.sent
    assert any(w.startswith(("SL1 ENTRY", "P2 SL1")) for w in link.a_io.sent), \
        link.a_io.sent
    assert link.a.protocol is link.b.protocol is Protocol.PACTOR1
    assert SHORT in rx_of(link.b), (rx_of(link.b), link.a_io.sent[-6:],
                                    link.b_io.sent[-6:])
    last = max(i for i, w in enumerate(link.b_io.sent) if w == word)
    assert all(w.startswith("P1 CS") for w in link.b_io.sent[last + 1:])


def test_our_answerer_grants_pactor2_and_our_pactor2_caller_follows():
    """The 0x6A9 row (off by default) end to end: our PACTOR-2-only caller
    announces bit 4, the answerer under `grant_pactor2` answers 0x6A9, reads the
    PACTOR-2 speed-level-1 entry behind its carrier as `upgrade_scan` does, and
    the traffic crosses in PACTOR-2."""
    link = GrantLink("K7ABC", "W9SSJ", verbose=False)
    _p2_caller(link)
    link.a.arq.on_host_connect("K7ABC", "W9SSJ")
    link.exchange(2)
    link.a.arq.on_host_data(MESSAGE[:300])
    link.exchange(30)
    assert "0x6A9" in link.b_io.sent and "0x59A" not in link.b_io.sent
    assert link.a_io.sent[2].startswith("P2 SL1"), link.a_io.sent
    assert link.b.protocol is Protocol.PACTOR2, link.b.log_lines[-10:]
    assert MESSAGE[:300] in rx_of(link.b), (link.a_io.sent[-6:], link.b_io.sent[-6:])
