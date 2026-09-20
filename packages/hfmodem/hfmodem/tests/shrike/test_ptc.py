# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Drive the PTC-IIIusb emulation end to end, as a host program would.

The test *is* the master: it speaks CRC hostmode at shrike over a byte stream,
exactly the way ptc-go/Pat, Airmail or Winlink Express do, and checks the
answers. Sequence mirrors ptc-go's OpenModem: terminal-mode init, JHOST4, poll,
connect, carry data both directions, turn the link around with %O/%I/%Q,
disconnect.

    python3 tests/test_ptc.py
"""
from __future__ import annotations

import sys


from hfmodem.shrike import arq, hostmode
from hfmodem.shrike import pactor1
from hfmodem.shrike import rxfront
from hfmodem.shrike.arq import ArqIO, State
from hfmodem.shrike.ptc import (P1_HISPEED_RETRIES, TRANSMITTABLE, UPGRADE_TARGETS,
                                PtcHost, SimPeer)
from hfmodem.shrike.spec import Protocol

PACTOR_CH = 31


class Master:
    """The host half: writes CRC-hostmode packets, reads the modem's answers."""

    def __init__(self, host: PtcHost):
        self.host = host
        self.decoder = hostmode.Decoder("master")
        self.counter = 0

    def terminal(self, line: str) -> str:
        return self.host.feed((line + "\r").encode()).decode("latin-1")

    def send(self, frame: bytes) -> list:
        out = self.host.feed(frame)
        if out == hostmode.REQUEST:
            raise AssertionError("modem requested a repeat of a good packet")
        return list(self.decoder.feed(out))

    def cmd(self, channel: int, text: str) -> hostmode.Response:
        self.counter ^= 1
        (resp,) = self.send(hostmode.command(channel, text, counter=self.counter))
        return resp

    def write(self, channel: int, blob: bytes) -> hostmode.Response:
        self.counter ^= 1
        (resp,) = self.send(hostmode.data(channel, blob, counter=self.counter))
        return resp

    def poll_channels(self) -> list[int]:
        resp = self.cmd(255, "G")
        assert resp.code == hostmode.MSG, resp
        return [b - 1 for b in resp.data]

    def status(self, n: int = 3) -> bytes:
        resp = self.cmd(254, f"G{n}")
        assert resp.code == hostmode.DATA, resp
        return resp.data

    def link_status(self) -> list[int]:
        resp = self.cmd(PACTOR_CH, "L")
        assert resp.code == hostmode.MSG, resp
        return [int(x) for x in resp.data.decode().split()]

    def drain(self) -> tuple[list[str], bytes]:
        """Poll until the PACTOR channel is quiet; return link events and data."""
        events, data = [], bytearray()
        for _ in range(200):
            resp = self.cmd(PACTOR_CH, "G")
            if resp.code == hostmode.OK:
                return events, bytes(data)
            (events.append(resp.data.decode()) if resp.code == hostmode.LINK
             else data.extend(resp.data))
        raise AssertionError("channel never went quiet")


def _cs(index: int, protocol: str = Protocol.PACTOR1) -> rxfront.Event:
    """One decoded control signal, as the receiver hands it to `on_rx_event`."""
    return rxfront.Event(t=0.0, kind="cs", text=f"CS{index + 1}",
                         protocol=protocol, cs=index)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        raise AssertionError(label)


def main() -> int:
    print("PTC-IIIusb hostmode emulation\n")

    # A PACTOR link is a transparent byte stream: the far end is delivered whole
    # packets, which do not line up with the host's writes, so record what it
    # actually echoed rather than assuming write boundaries survive the link.
    echoes: list[bytes] = []

    def _echo(blob: bytes) -> bytes:
        echoes.append(b"; SHRIKE ECHO " + blob)
        return echoes[-1]

    peer = SimPeer(reply=_echo)
    host = PtcHost(peer, mycall="N0CALL")
    m = Master(host)

    # -- terminal mode: what ptc-go's OpenModem sends --------------------
    check("banner names the model", "PTC-IIIusb" in host.open().decode())
    check("Quit in the main menu is an error", "ERROR" in m.terminal("Quit"))
    init = ["MYcall N0CALL", f"PTCH {PACTOR_CH}", "MAXE 35", "REM 0", "CHOB 0",
            "TONES 4", "MARK 1600", "SPACE 1400", "CWID 0", "CONType 3", "MODE 0",
            "DATE 210726", "TIME 120000"]
    replies = [m.terminal(c) for c in init]
    check("init script accepted with no errors", not any("ERROR" in r for r in replies),
          f"{len(init)} commands")
    check("PTCH took effect", host.ptchn == PACTOR_CH, f"channel {host.ptchn}")
    check("MYCALL took effect", host.mycall == "N0CALL")
    check("%V reports firmware", "4.1" in m.terminal("VER"))

    # A stray hostmode frame arriving in terminal mode must not draw an error.
    check("stray hostmode frame in terminal mode is quiet",
          "ERROR" not in host.feed(hostmode.command(0, "JHOST0") + b"\r").decode("latin-1"))

    m.terminal("JHOST4")
    check("JHOST4 enters hostmode", host.hostmode)

    # -- hostmode: idle ---------------------------------------------------
    check("startup banner waits on the monitor channel", 0 in m.poll_channels())
    m.cmd(0, "G")
    check("standby status byte is 0x87", m.status(0) == b"\x87", hex(m.status(0)[0]))
    check("standby link state is 0", m.link_status() == [0, 0, 0, 0, 0, 0])

    # -- connect ----------------------------------------------------------
    check("connect accepted", m.cmd(PACTOR_CH, "C N0DX").code == hostmode.OK)
    check("link setup state reported", m.link_status()[5] == 1)
    host.tick()
    events, _ = m.drain()
    check("CONNECTED event delivered", events == ["(31) CONNECTED to N0DX"],
          str(events))
    check("link state is information transfer", m.link_status()[5] == 4)
    st = m.status(3)
    # THE LEVEL THE LINK IS ACTUALLY IN, which a freshly connected one is 1. This
    # asserted 3 and passed against a byte that was hard-coded to 3 whatever the
    # link was doing -- the one thing about the session a host cannot see for
    # itself, reported wrongly.
    #
    # Asked of a CALLED station, because the calling one no longer holds still in
    # PACTOR-1 long enough to be asked. An ISS with nothing to say sends an IDLE
    # packet rather than nothing, so it has an acknowledged packet -- which is the
    # evidence `arq._on_ack` offers an upgrade on -- in its very first cycle. A
    # called station never sends a data packet and so never offers one at all, and
    # a link that is up and still in PACTOR-1 is exactly what this is about. The
    # upgraded reading of the same byte is checked below.
    called = Master(PtcHost(SimPeer(), mycall="N0DX"))
    called.terminal("JHOST4")
    called.host.arq.on_host_listen(True)
    called.host.arq.on_rx_connect("N0CALL", "N0DX")
    check("status reports PACTOR-1 on a link that has not upgraded",
          called.status(3)[1] == 1, f"level {called.status(3)[1]}")
    check("status direction bit set (we are ISS)", bool(st[0] & 0x08), hex(st[0]))

    # -- data out ---------------------------------------------------------
    msg = b"FC EM TESTID 1234 567 890 0\r" + b"PACTOR-3 HOSTMODE PAYLOAD " * 12
    chunks = [msg[i:i + hostmode.MAX_DATA] for i in range(0, len(msg), hostmode.MAX_DATA)]
    for i, chunk in enumerate(chunks):
        check(f"host write {i} accepted", m.write(PACTOR_CH, chunk).code == hostmode.OK)
    check("buffer reported as unsent", m.link_status()[2] > 0)
    for _ in range(60):
        host.tick()
        if m.link_status()[2:4] == [0, 0]:
            break
    check("all frames transmitted and acknowledged", m.link_status()[2:4] == [0, 0])
    check("the link upgraded once PACTOR-1 had carried a packet",
          host.protocol is Protocol.PACTOR3, f"still {host.protocol}")
    check("...and the status byte says so", m.status(3)[1] == 3)
    check("peer received the payload byte-exact", bytes(peer.received) == msg,
          f"{len(peer.received)}/{len(msg)} bytes")
    check("%T counts confirmed bytes",
          m.cmd(PACTOR_CH, "%T").data == str(len(msg)).encode(),
          m.cmd(PACTOR_CH, "%T").data.decode())

    # -- data in ----------------------------------------------------------
    _, back = m.drain()
    check("peer's reply reaches the host", back == b"".join(echoes),
          f"{len(back)} bytes")

    # -- changeover: %O, %I, %Q ------------------------------------------
    check("changeover accepted", m.cmd(PACTOR_CH, "%O").code == hostmode.OK)
    for _ in range(6):
        host.tick()
        if peer.role == "iss":
            break
    check("%O hands the link over", (host.arq.role, peer.role) == ("irs", "iss"),
          f"we are {host.arq.role}")
    check("status direction bit clears", not m.status(0)[0] & 0x08, hex(m.status(0)[0]))

    inbound = b"; DE N0DX - SENT AS ISS AFTER THE TURNAROUND\r"
    peer.send(inbound)
    for _ in range(20):
        host.tick()
        events, got = m.drain()
        if got:
            break
    check("far end transmits as ISS", got == inbound, f"{len(got)} bytes")

    check("breakin accepted", m.cmd(PACTOR_CH, "%I").code == hostmode.OK)
    for _ in range(6):
        host.tick()
        if host.arq.role == "iss":
            break
    check("%I takes the link back", (host.arq.role, peer.role) == ("iss", "irs"))
    check("status direction bit set again", bool(m.status(0)[0] & 0x08))

    more = b"FF\r"
    m.write(PACTOR_CH, more)
    for _ in range(20):
        host.tick()
        if m.link_status()[2:4] == [0, 0]:
            break
    check("data flows again after the turnaround",
          bytes(peer.received) == msg + more, f"{len(peer.received)} bytes")

    check("over accepted", m.cmd(PACTOR_CH, "%Q").code == hostmode.OK)
    for _ in range(6):
        host.tick()
        if host.arq.role == "irs":
            break
    check("%Q hands over once the buffer is confirmed", host.arq.role == "irs")

    # -- disconnect (from the IRS: break in, then QRT) --------------------
    check("disconnect accepted", m.cmd(PACTOR_CH, "D").code == hostmode.OK)
    for _ in range(10):
        host.tick()
        if host.arq.state in (State.DISCONNECTED, State.LISTENING):
            break
    events, _ = m.drain()
    check("DISCONNECTED event delivered", events == ["(31) DISCONNECTED fm N0DX"],
          str(events))
    check("link state back to disconnected", m.link_status()[5] == 0)
    check("status byte back to standby", m.status(0) == b"\x87")

    # -- link-layer robustness -------------------------------------------
    frame = bytearray(hostmode.command(PACTOR_CH, "L", counter=m.counter ^ 1))
    frame[-2] ^= 0xFF
    check("corrupt packet draws a request", host.feed(bytes(frame)) == hostmode.REQUEST)
    good = hostmode.command(PACTOR_CH, "L", counter=m.counter ^ 1)
    first = host.feed(good)
    check("retransmitted packet draws the same reply", host.feed(good) == first)
    check("modem still answers afterwards", m.cmd(PACTOR_CH, "L").code == hostmode.MSG)

    # -- listen -----------------------------------------------------------
    m.cmd(PACTOR_CH, "%L 1")
    check("listen mode arms the FSM", host.arq.state == State.LISTENING)
    check("%W reports it is safe to retune", m.cmd(PACTOR_CH, "%W0").data == b"1")

    # -- a caller does not give up while the far end is still calling ------
    # Measured on the air: WS8EOC kept calling once a second long after shrike
    # had exhausted its retries and dropped the link. Its bursts were often too
    # poor to decode, so nothing the FSM recognised as an answer ever arrived --
    # but something was plainly transmitting, and quitting then guarantees no
    # contact. Presence buys another cycle and nothing more.
    from hfmodem.shrike.arq import ArqConfig, PactorArq  # noqa: E402

    class _Silent(ArqIO):
        def __init__(self):
            self.bursts = 0
            self.aborted = False

        def connect_burst(self, mycall, dxcall):
            self.bursts += 1

        def disconnected(self):
            self.aborted = True

        def log(self, msg):
            pass

    io = _Silent()
    a = PactorArq(io, ArqConfig(max_connect_retries=3))
    a.on_host_connect("W9SSJ", "WS8EOC")
    for _ in range(12):
        a.on_cycle()
    check("a silent frequency still gives up", a.state == State.DISCONNECTED,
          f"state {a.state} after 12 quiet cycles")
    quiet_bursts = io.bursts

    io2 = _Silent()
    b = PactorArq(io2, ArqConfig(max_connect_retries=3))
    b.on_host_connect("W9SSJ", "WS8EOC")
    for _ in range(12):
        b.note_peer_heard()          # something is keying, contents unreadable
        b.on_cycle()
    check("a peer that is still transmitting holds the attempt open",
          b.state == State.CONNECTING, f"state {b.state}")
    check("and it keeps calling rather than idling", io2.bursts > quiet_bursts,
          f"{io2.bursts} bursts vs {quiet_bursts} on a dead frequency")
    check("presence alone never connects anything", b.state != State.CONNECTED,
          f"state {b.state}")

    # -- the IRS answers every cycle, decode or no decode -------------------
    # The reverse channel IS one control signal per cycle. shrike only ever
    # answered a packet it had decoded, so a peer whose traffic did not resolve
    # heard nothing back -- indistinguishable from a dead station. WS8EOC took
    # the link with CS3, got silence, and broke in again for a whole session.
    class _Counting(ArqIO):
        def __init__(self):
            self.cs = []
            self.aborted = False

        def send_cs(self, i):
            self.cs.append(i)

        def connect_burst(self, mycall, dxcall):
            pass

        def connected(self, mycall, dxcall):
            pass

        def disconnected(self):
            self.aborted = True

        def log(self, msg):
            pass

    from hfmodem.shrike.arq import (CS_ACK, CS_BREAKIN,  # noqa: E402
                                    CS_NAK, CS_REQUEST, IRS, SEQ_MOD,
                                    UNREPAIRED_BUDGET)
    io3 = _Counting()
    c = PactorArq(io3, ArqConfig(max_retries=6))
    c.on_host_connect("W9SSJ", "WS8EOC")
    c.on_rx_cs(CS_BREAKIN)               # the peer takes the link, as WS8EOC did
    check("a break-in during setup yields the link", c.role == IRS,
          f"role {c.role}")
    io3.cs.clear()
    for _ in range(3):
        c.on_cycle()                     # nothing decodable arrives
    check("an IRS that decoded nothing still answers every cycle",
          io3.cs == [CS_REQUEST] * 3, f"sent {io3.cs}")

    # ...and exactly once per cycle when it DID decode, not twice.
    io4 = _Counting()
    d = PactorArq(io4, ArqConfig(max_retries=6))
    d.on_host_connect("W9SSJ", "WS8EOC")
    d.on_rx_cs(CS_BREAKIN)
    io4.cs.clear()
    d.on_rx_packet(1, b"hi", 0, True)    # a good frame -> ACK from on_rx_packet
    n_after_rx = len(io4.cs)
    d.on_cycle()
    check("a decoded packet draws exactly one control signal",
          len(io4.cs) == n_after_rx == 1, f"sent {io4.cs}")

    # A link that never decodes anything must still end rather than REQ forever.
    for _ in range(20):
        c.on_cycle()
    check("an undecodable link eventually gives up", io3.aborted,
          f"still {c.state} after 23 quiet cycles")

    # ...but "undecodable" has to mean the channel and not the demodulator.
    # MEASURED, WS8EOC 2026-08-18 (working/pactor-ws8eoc-40m-clamped-force.log):
    # the gateway took the link, keyed a burst of about a second every cycle for
    # the rest of the session, and this budget ran out under it -- nine cycles,
    # eight of them carrying a shape line, the operator hearing the gateway call
    # on for minutes after we unkeyed. The captures decide what it was: hold_08,
    # hold_12 and hold_17 of captures/onair-0818-2235 decode CRC-valid as
    # PACTOR-1 changeover packets, one read in six.
    io10 = _Counting()
    m = PactorArq(io10, ArqConfig(max_retries=4))
    m.on_host_connect("W9SSJ", "WS8EOC")
    m.on_rx_cs(CS_BREAKIN)
    for _ in range(20):
        m.note_burst(96.0, at_anchor=True)   # a burst where an answer is due
        m.on_cycle()
    check("a link the peer is audibly still transmitting on is not abandoned",
          not io10.aborted and len(io10.cs) == 20,
          f"{m.state}, {len(io10.cs)} control signals")
    # ...AND THE POSITION IS WHAT SAYS SO. `note_peer_heard` identifies nobody,
    # and read here it held a dead link open on a third station's energy ten
    # times over -- KB5LZK 2026-08-28, on cycles the grid reported as empty or
    # as bursts outside the band an answer has to fall in.
    io10b = _Counting()
    mb = PactorArq(io10b, ArqConfig(max_retries=4))
    mb.on_host_connect("W9SSJ", "WS8EOC")
    mb.on_rx_cs(CS_BREAKIN)
    for _ in range(20):
        mb.note_peer_heard()             # carrier at the peer's tones, no bytes
        mb.note_burst(-79.0, at_anchor=False)      # ...and nowhere near the slot
        mb.on_cycle()
    check("...and presence with no position on it does not hold one open",
          io10b.aborted, f"still {mb.state}")
    for _ in range(20):
        m.on_cycle()                     # ...and then the channel goes quiet
    check("...and the same budget still ends it when nothing is there",
          io10.aborted, f"still {m.state}")

    # A peer whose packets ARRIVE and will not decode is the third case, and it
    # was the one nothing could see. A bad CRC sets `_rx_this_cycle`, which zeroes
    # `_silent_cycles`, so a station whose every packet was unreadable read as the
    # healthiest link on the band for as long as it kept keying -- REQ, NAK, REQ,
    # forever, on a shared frequency and at a gateway's expense as much as ours.
    io11 = _Counting()
    n = PactorArq(io11, ArqConfig())
    n.on_host_connect("W9SSJ", "WS8EOC")
    n.on_rx_cs(CS_BREAKIN)                   # the peer takes the link
    before = len(io11.cs)
    for _ in range(UNREPAIRED_BUDGET + 1):
        n.on_rx_packet(1, b"", 0, False)     # arrived, would not decode
    naks = [i for i in io11.cs[before:] if i == CS_NAK]
    check(f"an unreadable packet draws a repeat request, {UNREPAIRED_BUDGET}x",
          len(naks) == UNREPAIRED_BUDGET, f"{len(naks)} of {io11.cs[before:]}")
    check("...and then the link is ended deliberately rather than traded on",
          n._qrt_pending, f"state {n.state}, qrt {n._qrt_pending}")

    # The count is CONSECUTIVE: a link that loses one packet to a fade and reads
    # the next is a working link and may not accumulate its way to a goodbye.
    io12 = _Counting()
    r = PactorArq(io12, ArqConfig())
    r.on_host_connect("W9SSJ", "WS8EOC")
    r.on_rx_cs(CS_BREAKIN)
    for k in range(8):
        r.on_rx_packet(1, b"", 0, False)
        r.on_rx_packet(1, b"hi", k % SEQ_MOD, True)
    check("a packet that reads clears the repeat budget",
          not r._qrt_pending and not io12.aborted,
          f"state {r.state}, qrt {r._qrt_pending}")

    # -- the yield-and-listen recovery, and what refutes its premise ---------
    # "We keep going into receive when it doesn't ask us to." Two of the four
    # sessions that reached a peer on 2026-08-14 changed over on nothing at all:
    # rig-session-20260814-211344 after nine straight `HOLD n RX (quiet)`
    # cycles, -212300 after nine of its own, both having repeated packet #1 nine
    # times, neither peer transmitting again afterwards. The recovery exists for
    # a lost turnaround -- both ends holding the ISS role, each deaf to the
    # other's carrier -- and a station in that state is on the air 0.96 s of
    # every 1.25. Silence is the refutation of the premise, not the evidence
    # for it, and a yield is a handover of the channel.
    class _Sending(_Counting):
        def __init__(self):
            super().__init__()
            self.lines = []

        def send_packet(self, sl, payload, status, breakin=False):
            pass

        def buffer(self, n):
            pass

        def log(self, msg):
            # The role is read off the log rather than off `role`, which the
            # teardown clears: what the check is about is whether a changeover
            # ever went out, not what is left standing afterwards.
            self.lines.append(msg)

    io5 = _Sending()
    e = PactorArq(io5, ArqConfig(max_retries=4))
    e.on_host_connect("W9SSJ", "KI0BK")
    e.on_rx_cs(CS_ACK)                   # the call is answered; we are the ISS
    e.on_host_data(b"hello")
    for _ in range(12):
        e.on_cycle()
    check("a link that has heard nothing since it came up is not handed over",
          "changeover -> IRS" not in io5.lines and io5.aborted,
          f"log {io5.lines[-2:]}")

    # ...and the deafness it does exist for. A stranded ISS is handed the other
    # ISS's packets and drops them on its role -- that discard is the only sign
    # of the far end there will ever be, so it is what arms the yield.
    io6 = _Sending()
    f = PactorArq(io6, ArqConfig(max_retries=4))
    f.on_host_connect("W9SSJ", "KI0BK")
    f.on_rx_cs(CS_ACK)
    f.on_host_data(b"hello")
    for _ in range(6):
        f.on_rx_packet(1, b"", 0, True)  # the other ISS, keying past us
        f.on_cycle()
    check("...and one that can hear the far end still yields to break the "
          "deafness", f.role == IRS, f"role {f.role}")

    # A burst heard while CALLING is not a burst heard on the link:
    # rig-session-20260814-212300 logged one `peer burst heard (not decoded)`
    # in cycle 6 and nothing after it, and that one line would have bought the
    # yield nine silent hold cycles later.
    io7 = _Sending()
    g = PactorArq(io7, ArqConfig(max_retries=4))
    g.on_host_connect("W9SSJ", "WS8EOC")
    g.note_peer_heard()                  # somebody keying, while we are calling
    g.on_rx_cs(CS_ACK)
    g.on_host_data(b"hello")
    for _ in range(12):
        g.on_cycle()
    check("presence heard while calling does not survive into the link",
          "changeover -> IRS" not in io7.lines and io7.aborted,
          f"log {io7.lines[-2:]}")

    # -- and what the recovery window leaves standing ------------------------
    # Presence goes on arriving while we listen, and nothing in the window
    # clears it, so the flag outlives the yield that spent it. It still cannot
    # buy a second handover, and the two reasons hold independently.
    #
    # ONE: the only exit from a recovery is a packet that decodes. Shape-only
    # energy is not one, so a window that hears nothing but carrier ends at the
    # IRS's own budget rather than back on the transmitting side.
    io8 = _Sending()
    h = PactorArq(io8, ArqConfig(max_retries=4))
    h.on_host_connect("W9SSJ", "KI0BK")
    h.on_rx_cs(CS_ACK)
    h.on_host_data(b"hello")
    for _ in range(8):
        if h.role == IRS:
            break                        # stop at the yield: a packet decoded
        h.on_rx_packet(1, b"", 0, True)  # after it would end the recovery
        h.on_cycle()
    check("the deafness yield fires", h.role == IRS, f"role {h.role}")
    for _ in range(16):
        h.note_peer_heard()              # carrier, and nothing that resolves
        h.on_cycle()
    check("presence alone never ends a recovery",
          io8.aborted and "no decodable traffic from the peer -> QRT"
          in io8.lines, f"log {io8.lines[-2:]}")

    # TWO: a station still recovering reaches the retry branch only behind a
    # host QRT, which re-takes the link because the goodbye rides a packet. The
    # channel has been handed over once already and the peer did not take it;
    # handing it over again on the same silence is how both ends end up
    # receiving. Presence every cycle, as loud as it gets, must not buy it.
    io9 = _Sending()
    k = PactorArq(io9, ArqConfig(max_retries=4))
    k.on_host_connect("W9SSJ", "KI0BK")
    k.on_rx_cs(CS_ACK)
    k.on_host_data(b"hello")
    for _ in range(8):
        if k.role == IRS:
            break
        k.on_rx_packet(1, b"", 0, True)
        k.on_cycle()
    k.on_host_disconnect()
    k.on_cycle()                         # re-takes the link to send the QRT
    for _ in range(8):
        k.note_peer_heard()
        k.on_cycle()
    check("a station in recovery does not hand the channel over twice",
          len([m for m in io9.lines if "yield the link" in m]) == 1
          and io9.aborted, f"log {io9.lines[-2:]}")

    # -- the ANSWER picks the link speed, and getting it wrong is silent ------
    # CS1 means the called station read our 200 Bd redundancy section cleanly and
    # the link starts at 200 Bd; CS4 means it did not and the link starts at 100.
    # A master that answers CS1 with a 100 Bd packet transmits into a receiver
    # listening at 200: the link is deaf and there is nothing on the air to break
    # the deadlock, so no amount of on-air listening would ever diagnose it.
    class _Speed:
        def __init__(self):
            self.sent = []                   # baud of each packet
            self.fields = []                 # (baud, payload) of each

        def attach(self, host):
            pass

        def send_cs(self, i):
            pass

        def send_p1_cs(self, i):
            pass

        def send_p1_packet(self, payload, baud, count, **kw):
            self.sent.append(baud)
            self.fields.append((baud, bytes(payload)))

        def send_packet(self, sl, payload, status, breakin=False):
            pass

        def connect_burst(self, mycall, dxcall):
            pass

        def pump(self):
            pass

        def cycle(self):
            pass

    class _CsEv:
        kind, protocol, t, text = "cs", "PACTOR-1", 0.0, ""

        def __init__(self, cs):
            self.cs = cs

    for label, answer, want in (("CS1", pactor1.CS_ACK_A, 200),
                                ("CS4", pactor1.CS_SPEED, 100)):
        sp = _Speed()
        hs = PtcHost(sp, mycall="W9SSJ")
        hs.arq.on_host_connect("W9SSJ", "WS8EOC")
        hs.on_rx_event(_CsEv(answer))
        hs.tick()
        check(f"a {label} answer runs the link at {want} Bd",
              hs.p1_baud == want and sp.sent[:1] == [want],
              f"link {hs.p1_baud} Bd, sent {sp.sent[:1]}")
        check("...and the field size follows it",
              hs.arq.payload_bytes_override == pactor1.DATA_FIELD[want],
              f"{hs.arq.payload_bytes_override} bytes")

    # -- only CS1 and CS4 answer a call --------------------------------------
    # This used to accept any of the four, on the reasoning that nothing else
    # would be transmitting in the peer's slot. CS2 is the other half of the
    # in-session acknowledge alternation and CS3 is never bare at all -- it is the
    # head of the break-in packet -- so either one arriving during setup is a
    # decode off a station that is not talking to us, or an exchange we have
    # walked in on. Both brought a link up. Both silent captures are a gateway
    # free-running one codeword at 1.25 s for a minute after its exchange died,
    # which is exactly the traffic that has to be refused here.
    for label, cs in (("CS2", pactor1.CS_ACK_B), ("CS3", pactor1.CS_CHANGEOVER)):
        sp = _Speed()
        hs = PtcHost(sp, mycall="W9SSJ")
        hs.arq.on_host_connect("W9SSJ", "WS8EOC")
        hs.on_rx_event(_CsEv(cs))
        hs.tick()
        check(f"a {label} does NOT bring a PACTOR-1 link up",
              hs.arq.state == State.CONNECTING and not sp.sent,
              f"state {hs.arq.state}, sent {sp.sent}")

    # -- CS4 on a 200 Bd link is a REJECT, and it names a speed ---------------
    # Measured in captures/witness.wav: WS8EOC answered our call CS1 four times
    # at zero errors, then sent CS4 and repeated it for 25 further cycles on the
    # 1.25 s raster -- 30 s of them after our transmitter had stopped -- and then
    # identified in CW. "Das gerade gesendete, nicht bestaetigte Paket wird
    # verworfen und die in ihm enthaltene Information erneut im 100-Baud-Modus
    # ausgesendet." It was asking for the same information at half the rate.
    #
    # `_logical_cs` mapped that to a plain repeat request and left the speed
    # alone, so shrike retransmitted at 200 Bd into a receiver listening at 100
    # and neither end had anything on the air to break the deadlock. The
    # information has to be RE-CHUNKED, not retransmitted: 20 bytes at 200 Bd
    # become three 8-byte packets at 100, at the same sequence number, so a peer
    # that did decode the 200 Bd packet throws the repeat away as a duplicate.
    sp = _Speed()
    hs = PtcHost(sp, mycall="W9SSJ")
    hs.arq.on_host_connect("W9SSJ", "WS8EOC")
    hs.on_rx_event(_CsEv(pactor1.CS_ACK_A))          # CS1 -> the link is 200 Bd
    hs.arq.on_host_data(bytes(range(40)))
    hs.tick()
    at200 = list(sp.fields)
    hs.on_rx_event(_CsEv(pactor1.CS_SPEED))          # CS4 -> fall back to 100
    hs.tick()
    after = sp.fields[len(at200):]
    check("a 200 Bd link that receives CS4 falls back to 100 Bd",
          hs.p1_baud == 100 and hs.arq.payload_bytes_override
          == pactor1.DATA_FIELD[100],
          f"{hs.p1_baud} Bd, {hs.arq.payload_bytes_override}-byte field")
    check("...it was really transmitting at 200 before it (the test can fail)",
          at200 and all(b == 200 and len(p) == 20 for b, p in at200),
          f"{[(b, len(p)) for b, p in at200]}")
    # The re-chunk, asserted on the BYTES: the first 100 Bd packet must carry the
    # front of the 20 the peer did not acknowledge, not the next 8 in the queue.
    # Checking the length alone passes for a packet that skipped the payload
    # entirely, which is the failure this exists to catch.
    check("...and the unacknowledged information is re-sent at the new size",
          after and after[0][0] == 100 and after[0][1] == at200[0][1][:8],
          f"{after[:1]} against {at200[0][1][:8]!r}")

    # A repeat of CS4 with no other codeword between is a plain Request and must
    # NOT drop the speed again -- "CS4 in Folge (ohne zwischenzeitliches richtiges
    # CS) werden als 'Request' interpretiert, also ignoriert." Nothing new may be
    # dequeued for it either: the same bytes go out again.
    before = len(sp.fields)
    hs.on_rx_event(_CsEv(pactor1.CS_SPEED))
    hs.tick()
    repeat = sp.fields[before:]
    check("a repeated CS4 is a plain Request -- the same packet, same speed",
          hs.p1_baud == 100 and repeat and set(repeat) == {after[0]},
          f"{[(b, len(p)) for b, p in repeat]}")

    # -- a link that came up at 200 on CS1 is on trial on the same figure -----
    # An unbroken run of the connect answer at zero bit errors says one thing per
    # cycle -- our 960 ms packet has not decoded (control-signals §5) -- and the
    # count used to be armed only by the CS4 speed-up, so this run was bounded by
    # nothing at all. The retry budget cannot see it: it counts silence, and a
    # codeword at zero errors is the reverse channel working. It is the run both
    # sessions that lost a gateway to an unreadable wideband emission spent at
    # full rate -- WS8EOC six CS1 on 2026-08-28, KB5LZK seven on 2026-08-29,
    # packet #1 at 200 Bd on every cycle of both, each gateway leaving PACTOR-1
    # on the cycle after its run.
    sp = _Speed()
    hs = PtcHost(sp, mycall="W9SSJ")
    hs.stay_in_pactor1 = True
    hs.arq.on_host_connect("W9SSJ", "WS8EOC")
    hs.on_rx_event(_CsEv(pactor1.CS_ACK_A))          # CS1 answer -> 200 Bd link
    hs.arq.on_host_data(bytes(range(40)))
    hs.tick()
    held200 = list(sp.fields)
    for n in range(1, P1_HISPEED_RETRIES + 1):
        before = len(sp.fields)
        hs.on_rx_event(_CsEv(pactor1.CS_ACK_A))      # held: "not yet", again
        hs.tick()
        if n < P1_HISPEED_RETRIES:
            check(f"a held CS1 keeps the link at 200 Bd through repeat {n}",
                  hs.p1_baud == 200, f"{hs.p1_baud} Bd after {n}")
    down = sp.fields[before:]
    check("a run of the CS1 connect answer falls the link back to 100 Bd",
          hs.p1_baud == 100 and hs.arq.payload_bytes_override
          == pactor1.DATA_FIELD[100],
          f"{hs.p1_baud} Bd, {hs.arq.payload_bytes_override}-byte field")
    check("...it was really transmitting at 200 before it (the test can fail)",
          held200 and all(b == 200 and len(p) == 20 for b, p in held200),
          f"{[(b, len(p)) for b, p in held200]}")
    check("...and the unacknowledged information is re-sent at the new size",
          down and down[0] == (100, held200[0][1][:8]),
          f"{down[:1]} against {held200[0][1][:8]!r}")

    # AND AN ALTERNATING PEER IS NOT A RUN. The count spends decoded codewords
    # only and any CHANGE of codeword disarms it, so a 200 Bd link the peer is
    # acknowledging cannot be geared down however long it lasts.
    sp = _Speed()
    hs = PtcHost(sp, mycall="W9SSJ")
    hs.stay_in_pactor1 = True
    hs.arq.on_host_connect("W9SSJ", "WS8EOC")
    hs.on_rx_event(_CsEv(pactor1.CS_ACK_A))
    hs.arq.on_host_data(bytes(range(200)))
    hs.tick()
    for cs in (pactor1.CS_ACK_B, pactor1.CS_ACK_A) * P1_HISPEED_RETRIES:
        hs.on_rx_event(_CsEv(cs))
        hs.tick()
    check("a peer that alternates keeps the link at 200 Bd", hs.p1_baud == 200,
          f"{hs.p1_baud} Bd")

    # -- and the other half of §4.3: CS4 on a 100 Bd link is the speed-up ------
    # "Kann nach jedem richtig empfangenen 100-Bd-Paket ... mit CS4 bestaetigen,
    # was den TX zur Umschaltung auf 200 Baud zwingt." Asserted on the BYTES that
    # went out, because this is what the speed is worth: 8 payload bytes a cycle
    # become 20, and a Winlink greeting of 132-239 bytes needs 7-12 clean cycles
    # instead of 17-30. Both halves have to hold at once -- the packet the CS4
    # answered is settled and the NEXT information goes out, at the new size.
    sp = _Speed()
    hs = PtcHost(sp, mycall="W9SSJ")
    # The 40 bytes below are what makes the speed worth measuring, and they are
    # also what the upgrade offer waits for -- so without this the CS1 two lines
    # down takes the link to PACTOR-3 and the CS4 after it is read by a station
    # that is no longer running PACTOR-1 at any rate at all.
    hs.stay_in_pactor1 = True
    hs.arq.on_host_connect("W9SSJ", "WS8EOC")
    hs.on_rx_event(_CsEv(pactor1.CS_SPEED))          # CS4 answer -> 100 Bd link
    hs.arq.on_host_data(bytes(range(40)))
    hs.tick()
    # CS1 first, because the speed-up is not reachable until it: "CS4 dient als
    # 'REQUEST'-CS fuer den ersten 100-Bd-Block, Bestaetigung erfolgt mittels CS1
    # oder CS3, danach kann bereits wieder ein CS4 als 'speedup'-Signal gesendet
    # werden." A CS4 here would still be the request -- the check below it.
    hs.on_rx_event(_CsEv(pactor1.CS_ACK_A))
    hs.tick()
    at100 = list(sp.fields)
    hs.on_rx_event(_CsEv(pactor1.CS_SPEED))          # ...and now the speed offer
    hs.tick()
    up = sp.fields[len(at100):]
    # `_answer_link_setup` queues the level-and-callsign packet ahead of the
    # host's own bytes, so this is the whole of what the link had to send.
    queued = b"1w9ssj\r" + bytes(range(40))
    check("a 100 Bd link that receives CS4 rises to 200 Bd",
          hs.p1_baud == 200 and hs.arq.payload_bytes_override
          == pactor1.DATA_FIELD[200],
          f"{hs.p1_baud} Bd, {hs.arq.payload_bytes_override}-byte field")
    check("...it was really sending 8 bytes at 100 before it (the test can fail)",
          at100 == [(100, queued[:8]), (100, queued[8:16])],
          f"{[(b, len(p)) for b, p in at100]}")
    # The acknowledgement, on the bytes: the 200 Bd packet carries the NEXT
    # information rather than repeating what the CS4 just settled. Asserting the
    # size alone passes for a link that changed gear and advanced nothing, which
    # is the half of CS4 that was missing.
    check("...and the next packet carries the NEXT 20 bytes at 200 Bd",
          up[:1] == [(200, queued[16:36])], f"{up[:1]}")

    # -- a control signal goes out in the protocol the link is actually in --
    # PACTOR-1 and PACTOR-3 control signals are different waveforms. A link that
    # has not been upgraded is still PACTOR-1, and answering it in PACTOR-3 is
    # the same as not answering.
    class _BothWays:
        def __init__(self):
            self.p1, self.p3 = [], []

        def attach(self, host):
            pass

        def send_cs(self, i):
            self.p3.append(i)

        def send_p1_cs(self, i):
            self.p1.append(i)

    # -- PACTOR-1 acknowledges by ALTERNATING, not by a distinct codeword ----
    # "CS1..3 have the same function as their AMTOR counterparts; CS4 serves as
    # the speed change control." In AMTOR the acknowledgement is the TOGGLE
    # between two control signals, and a repeat is requested by sending the same
    # one again. There is no NAK codeword. shrike sent a constant CS2 for a whole
    # on-air session, which reads as "send that again" forever -- and a real
    # gateway responded to it exactly that way, which is why sweeping all four
    # codewords changed nothing.
    from hfmodem.shrike import spec  # noqa: E402
    from hfmodem.shrike.arq import CS_ACK, CS_NAK, CS_SPEED_UP  # noqa: E402
    from hfmodem.shrike.arq import CS_BREAKIN, CS_CYCLE_TOG, CS_REQUEST  # noqa: E402
    from hfmodem.shrike.arq import SEQ_MOD  # noqa: E402
    from hfmodem.shrike.arq import _Packet  # noqa: E402

    class _Wire:
        def __init__(self):
            self.sent = []

        def attach(self, host):
            pass

        def send_cs(self, i):
            pass

        def send_p1_cs(self, i):
            self.sent.append(i)

    w = _Wire()
    h1 = PtcHost(w, mycall="W9SSJ")
    # Only CS1 and CS4 answer a connect, and the called station's first
    # acknowledgement IS that answer. It falls out of the counter with nothing
    # kept: "das erste normale Datenpaket mit Head=AA (HEX) und Paketzaehler=1",
    # so a link counts from one and the codeword before its first packet is
    # counter 0's, which is CS1.
    h1.arq.on_host_listen(True)
    h1.arq.on_rx_connect("WS8EOC", "W9SSJ")
    check("the first acknowledgement is CS1, a legal connect answer",
          w.sent == [pactor1.CS_ACK_A], f"sent {w.sent}")
    w.sent.clear()
    def accepted(host):
        """One more packet taken from the peer, which is what the codeword reads."""
        host.arq._expected_seq = (host.arq._expected_seq + 1) % SEQ_MOD

    for _ in range(3):
        accepted(h1)
        h1.send_cs(CS_ACK)
    check("an acknowledgement per packet accepted alternates",
          len(set(w.sent)) == 2 and w.sent[0] != w.sent[1] != w.sent[2],
          f"sent {w.sent}")
    w.sent.clear()
    # ...and a second acknowledgement of the SAME packet is the first one again.
    # A repeat is the peer saying it did not read the codeword, so keying its
    # alternate acknowledges a packet that never arrived and lands half the time
    # on the codeword the peer last read, which IS the request. See test_p1ack.
    for c in (CS_ACK, CS_ACK):
        h1.send_cs(c)
    check("acknowledging one packet twice keys one codeword twice",
          w.sent[0] == w.sent[1], f"sent {w.sent}")
    w.sent.clear()
    h1.send_cs(CS_NAK)
    h1.send_cs(CS_NAK)
    check("a repeat request re-sends the SAME codeword",
          w.sent[0] == w.sent[1], f"sent {w.sent}")
    w.sent.clear()
    h1.send_cs(CS_BREAKIN)
    check("changeover has its own codeword (CS3)",
          w.sent == [pactor1.CS_CHANGEOVER], f"sent {w.sent}")
    w.sent.clear()
    h1.send_cs(CS_SPEED_UP)
    check("speed change has its own codeword (CS4)",
          w.sent == [pactor1.CS_SPEED], f"sent {w.sent}")
    w.sent.clear()
    for _ in range(2):
        accepted(h1)
        h1.send_cs(CS_ACK)
    check("the alternation survives a changeover and a speed change",
          w.sent[0] != w.sent[1] and set(w.sent) <= {pactor1.CS_ACK_A, pactor1.CS_ACK_B},
          f"sent {w.sent}")

    # -- and the receive side, which is where a misread codeword costs a link ---
    # CS4 is the one a free-running gateway repeats at us, once per cycle, for as
    # long as it is calling. Read as a break-in it made shrike hand the channel to
    # a station that was asking it to retransmit, and both ends then waited for
    # each other. What it means is settled by the speed the link is already
    # running at -- pactor1-control-signals.md sec 4.3 -- so the sender never has
    # to guess: at 200 it is the REJECT, at 100 it acknowledges and forces 200.
    from hfmodem.shrike.rxfront import Event  # noqa: E402

    def logical(host, index):
        return host._logical_cs(Event(0.0, "cs", "", protocol="PACTOR-1", cs=index))

    h3 = PtcHost(_Wire(), mycall="W9SSJ")
    # ON A LINK, because "forces 200" is a statement about one. The same seam
    # runs on every decoded codeword, including the ones an arm hears while it
    # observes after teardown, and there a CS4 rises nothing.
    h3.arq.state = State.CONNECTED
    check("a received CS1 acknowledges", logical(h3, pactor1.CS_ACK_A) == CS_ACK)
    check("a received CS2 acknowledges too -- the ACK is the alternation",
          logical(h3, pactor1.CS_ACK_B) == CS_ACK)
    check("a repeat of the same codeword is a Request, not a NAK",
          logical(h3, pactor1.CS_ACK_B) == CS_REQUEST)
    check("a received CS3 is the break-in",
          logical(h3, pactor1.CS_CHANGEOVER) == CS_BREAKIN)
    # CS3 is not in the alternation. The CS2 above is
    # still the last acknowledgement, so the CS2 below is still its repeat; read
    # the other way, a peer whose acknowledgement is interrupted by a changeover
    # head has that acknowledgement counted twice and the sequence walks over a
    # packet the peer never received.
    check("a codeword either side of a CS3 is still a repeat of itself",
          logical(h3, pactor1.CS_ACK_B) == CS_REQUEST)
    got = logical(h3, pactor1.CS_SPEED)
    check("a received CS4 at 100 Bd acknowledges and takes the link to 200",
          got == CS_ACK and h3.p1_baud == 200
          and h3.arq.payload_bytes_override == pactor1.DATA_FIELD[200],
          f"mapped to {got}, {h3.p1_baud} Bd, "
          f"{h3.arq.payload_bytes_override}-byte field")
    check("consecutive CS4s stay a plain Request",
          logical(h3, pactor1.CS_SPEED) == CS_REQUEST and h3.p1_baud == 200)
    # CS4 consumed the ACK position of the 100 Bd block. The first 200 Bd ACK
    # is therefore the same wire word as BEFORE CS4 (hf-pactor
    # tx_100_200_cs1); the opposite word declines the trial without ACKing it.
    check("...and the speed-up consumes an acknowledgement position",
          logical(h3, pactor1.CS_ACK_B) == CS_ACK and h3.p1_baud == 200)
    # The speed-up is bounded, because nothing else bounds it: the retry budget
    # counts silence and a held CS4 is a codeword at zero errors, once a cycle,
    # asking for a packet at a rate the peer has stopped reading. Four of them
    # after the speed-up and the link puts itself back where it can be heard --
    # pactor1-timing.md sec 4. On its own host, because the count wants an
    # unbroken run and any other codeword between is the peer talking.
    h4 = PtcHost(_Wire(), mycall="W9SSJ")
    h4.arq.state = State.CONNECTED
    logical(h4, pactor1.CS_ACK_A)                     # an acknowledgement at 100
    logical(h4, pactor1.CS_SPEED)                     # ...then the speed offer
    for n in range(1, P1_HISPEED_RETRIES + 1):
        logical(h4, pactor1.CS_SPEED)                 # held: a repeat request
        if n < P1_HISPEED_RETRIES:
            check(f"the link holds 200 Bd through repeat {n}", h4.p1_baud == 200,
                  f"{h4.p1_baud} Bd after {n}")
    check("a speed-up nothing acknowledges falls back to 100 Bd",
          h4.p1_baud == 100
          and h4.arq.payload_bytes_override == pactor1.DATA_FIELD[100],
          f"{h4.p1_baud} Bd, {h4.arq.payload_bytes_override}-byte field")

    # The spec's own two internal checks, which the codewords must satisfy.
    cs = pactor1.CONTROL_SIGNALS
    d = {bin(a ^ b).count("1") for i, a in enumerate(cs) for b in cs[i + 1:]}
    check("mutual Hamming distance is 8, as the description states", d == {8}, str(d))
    rev = lambda v: int(format(v, "012b")[::-1], 2)  # noqa: E731
    check("CS1/CS2 and CS3/CS4 are bit-reverse pairs, as it states",
          rev(cs[0]) == cs[1] and rev(cs[2]) == cs[3],
          f"{rev(cs[0]):#05x} vs {cs[1]:#05x}, {rev(cs[2]):#05x} vs {cs[3]:#05x}")

    # -- PACTOR-3 alternates too, and it alternates on the COUNTER -------------
    # MEASURED off `rf-corpus/PIII_Complete_1.wav`, thirty consecutive
    # confirmations at zero bit errors: a packet whose status-byte counter is EVEN
    # is answered CS1 and an ODD one CS2, unbroken through the SL3-SL6 ladder,
    # both cycle lengths, and across every changeover, with CS4 and CS6
    # substituting in the odd slot. So the same rule PACTOR-1 runs on holds here,
    # with the phase carried on the air rather than kept as a local toggle -- and
    # a constant CS1 tells a real ISS, on every odd counter, to send it again.
    #
    # The FSM's alphabet IS the six codewords, so the codeword for CS2 is spelled
    # CS_REQUEST here; that ambiguity is the whole reason this translation exists.
    p3w = _BothWays()
    h5 = PtcHost(p3w, mycall="W9SSJ")
    h5.upgrade(payload_waiting=True)
    for seq in (1, 2, 3, 0, 1):
        h5.arq._expected_seq = (seq + 1) % 4          # ...that packet, accepted
        h5.send_cs(CS_ACK)
    check("a PACTOR-3 acknowledgement follows the answered packet's counter",
          p3w.p3 == [CS_REQUEST, CS_ACK, CS_REQUEST, CS_ACK, CS_REQUEST],
          f"sent {p3w.p3}")
    p3w.p3.clear()
    for _ in range(2):
        h5.send_cs(CS_ACK)                            # a duplicate, twice acked
    check("acknowledging one PACTOR-3 packet twice keys one codeword twice",
          p3w.p3[0] == p3w.p3[1], f"sent {p3w.p3}")
    p3w.p3.clear()
    # A cycle that decoded nothing still owes the ISS a codeword, and the one it
    # owes is the acknowledgement it last sent -- nothing was accepted, so nothing
    # moved. PACTOR-3 has no separate "again" word: CS2 is the answer to an odd
    # counter, and keying it against an even packet in flight would confirm a
    # packet that never arrived. CS5 is the NAK, and it drops a level with it.
    h5.arq._expected_seq = 3
    h5.send_cs(CS_ACK)
    h5.send_cs(CS_REQUEST)
    check("a PACTOR-3 repeat request holds the acknowledgement it last keyed",
          p3w.p3[0] == p3w.p3[1] == CS_ACK, f"sent {p3w.p3}")
    p3w.p3.clear()
    for c in (CS_NAK, CS_SPEED_UP, CS_BREAKIN, CS_CYCLE_TOG):
        h5.send_cs(c)
    check("the other four codewords carry their own meanings unchanged",
          p3w.p3 == [CS_NAK, CS_SPEED_UP, CS_BREAKIN, CS_CYCLE_TOG],
          f"sent {p3w.p3}")

    # ...and the receive side, where the deadlock was: a real peer confirms our
    # packet #1 with CS2, and `arq` maps CS2 to a repeat request. So we resent it,
    # the peer answered the duplicate with CS2 again, and neither end could ever
    # move. Nothing else in the session had to be wrong for that to happen.
    def p3_logical(host, index):
        return host._logical_cs(Event(0.0, "cs", "", protocol="PACTOR-3", cs=index))

    h6 = PtcHost(_Wire(), mycall="W9SSJ")
    h6.upgrade(payload_waiting=True)
    h6.arq._inflight = _Packet(status=spec.status_byte(1), payload=b"", sl=3)
    check("CS2 answering our odd packet ACKNOWLEDGES it",
          p3_logical(h6, CS_REQUEST) == CS_ACK)
    check("...and CS1 there is the peer still answering the packet before",
          p3_logical(h6, CS_ACK) == CS_REQUEST)
    h6.arq._inflight = _Packet(status=spec.status_byte(2), payload=b"", sl=3)
    check("CS1 answering our even packet acknowledges it",
          p3_logical(h6, CS_ACK) == CS_ACK)
    check("...and CS2 there asks for it again",
          p3_logical(h6, CS_REQUEST) == CS_REQUEST)
    check("CS3 to CS6 reach the FSM unread by the alternation",
          [p3_logical(h6, c) for c in (CS_BREAKIN, CS_SPEED_UP, CS_NAK,
                                       CS_CYCLE_TOG)]
          == [CS_BREAKIN, CS_SPEED_UP, CS_NAK, CS_CYCLE_TOG])
    h6.arq._inflight = None
    check("with nothing in flight there is no packet to answer",
          p3_logical(h6, CS_REQUEST) == CS_REQUEST)

    bw = _BothWays()
    h2 = PtcHost(bw, mycall="W9SSJ")
    check("a link starts out as PACTOR-1",
          h2.protocol is Protocol.PACTOR1)
    # Which PROTOCOL carries it, not which codeword -- the codeword now depends
    # on the alternation phase, and pinning it here would just re-assert the
    # lookup table this replaced.
    h2.send_cs(CS_ACK)
    check("an un-upgraded link answers in PACTOR-1",
          len(bw.p1) == 1 and bw.p3 == [], f"p1={bw.p1} p3={bw.p3}")
    h2.upgrade(payload_waiting=True)
    h2.arq._expected_seq = 3               # packet 2 taken, so the answer is CS1
    h2.send_cs(CS_ACK)
    check("an upgraded link answers in PACTOR-3",
          len(bw.p1) == 1 and bw.p3 == [CS_ACK], f"p1={bw.p1} p3={bw.p3}")
    # THE TARGET IS CHOSEN, not assumed, and the choice is what a peer can
    # contradict. A station that answers a PACTOR-3 packet in PACTOR-1 is telling
    # us the only way the protocol has of telling us anything.
    h2.arq.state = State.CONNECTED
    h2._follow_peer("PACTOR-1")
    check("a PACTOR-1 answer drops an upgraded link back",
          h2.protocol is Protocol.PACTOR1)
    # The ARQ's stored speed level survives the fallback (arq.PactorArq.speed_level
    # says so), so the status bytes must not read it on a PACTOR-1 link.
    check("...and the status bytes follow it down: level 1, speed level 0",
          h2._status_bytes()[1:3] == b"\x01\x00", h2._status_bytes().hex())
    h2.upgrade(payload_waiting=True)
    check("...and rules that target out, so it is not tried again",
          h2.protocol is Protocol.PACTOR1, f"went to {h2.protocol}")
    # PACTOR-2 IS KEYED AND IS NOT OFFERED, which is the one place the two facts
    # come apart. Following a peer that leads into it answers a station that has
    # already keyed the waveform; offering one uninvited keys a codeword nothing
    # outside this package has graded, at a station that has said nothing about
    # it -- and no recording anywhere holds a PACTOR-1 link making that
    # transition. So the rung is behind `offer_pactor2` and sits BEHIND
    # PACTOR-3, tried only once this link has been contradicted on it.
    check("PACTOR-2 is keyed, and is not an uninvited target by default",
          Protocol.PACTOR2 in TRANSMITTABLE
          and Protocol.PACTOR2 not in UPGRADE_TARGETS
          and h2._upgrade_targets() == UPGRADE_TARGETS)
    h2.offer_pactor2 = True
    check("...and behind the flag it is the rung after PACTOR-3",
          h2._upgrade_targets() == (Protocol.PACTOR3, Protocol.PACTOR2),
          str(h2._upgrade_targets()))
    # PACTOR-3 was ruled out three checks ago, so this offer takes the rung.
    h2.upgrade(payload_waiting=True)
    check("...which a link contradicted on PACTOR-3 then takes",
          h2.protocol is Protocol.PACTOR2
          and h2.arq.speed_level == arq.P2_LADDER.entry_sl
          and not h2.arq.entry_pending,
          f"{h2.protocol} at SL{h2.arq.speed_level}")
    h2.offer_pactor2 = False
    h2.protocol = Protocol.PACTOR1

    # -- ONE CONTACT'S BELIEFS DO NOT REACH THE NEXT ------------------------
    # A hostmode client works one modem down a channel list, so the second call
    # of a run is placed by the same PtcHost object the first one upgraded. Every
    # field below describes the link that ended; carried into CONNECTING they
    # describe the wrong station, and `protocol` is read by the frame scan that
    # is trying to hear this call's answer.
    r = PtcHost(peer=None, mycall="W9SSJ")
    r.arq.on_host_connect("W9SSJ", "KI0BK")
    r.on_rx_event(_cs(pactor1.CS_ACK_A))         # answered CS1: the link runs 200 Bd
    r.upgrade(payload_waiting=True)
    check("the first call comes up at the rate its answer named, and upgrades",
          r.arq.state == State.CONNECTED and r.p1_baud == 200
          and r.protocol is Protocol.PACTOR3, f"{r.protocol} at {r.p1_baud} Bd")
    r.arq.on_host_abort()
    r.arq.on_host_connect("W9SSJ", "WS8EOC")     # the next station on the list
    check("a new call starts in PACTOR-1, whatever the last link ended in",
          r.protocol is Protocol.PACTOR1, f"calling in {r.protocol}")
    check("...at 100 Bd, because this call has had no answer yet",
          r.p1_baud == 100, f"{r.p1_baud} Bd")
    check("...and with no memory of the last station's reverse channel",
          r._last_rx_cs is None and r._prev_rx_cs is None)
    # The sharp consequence, and it is silent: `_answer_link_setup` queues the
    # spec's mandatory first data packet only on a PACTOR-1 link, so a stale
    # upgrade meant a call that WAS answered then transmitted nothing at all and
    # the peer repeated its answer until it gave up.
    r.on_rx_event(_cs(pactor1.CS_SPEED))         # answered CS4: this one runs 100
    check("...so the answer to it queues the first data packet",
          r.arq._outbuf == b"1w9ssj\r", f"{bytes(r.arq._outbuf)!r}")
    # The upgrade ladder is per link for the same reason: a peer that cannot
    # follow says so by answering in PACTOR-1, and a station that never heard the
    # offer has said nothing at all.
    r.upgrade(payload_waiting=True)
    r._fall_back_to_pactor1("the peer never answered the upgrade")
    check("a link the peer could not follow rules that target out",
          Protocol.PACTOR3 in r._ruled_out, f"ruled out {sorted(r._ruled_out)}")
    r.arq.on_host_abort()
    r.arq.on_host_connect("W9SSJ", "KD9PTZ")
    check("...for that link and no other: the next call may offer it again",
          not r._ruled_out and r.upgrade(payload_waiting=True),
          f"ruled out {sorted(r._ruled_out)}")

    # -- A THIRD PARTY'S FRAME IS NOT OUR PEER, AND MUST NOT EXTEND THE CALL --
    # The connect budget is forgiven on `note_peer_heard` because undecoded
    # energy could be the station we called keying too poorly to read. A frame
    # that DECODED and was refused is the opposite: it names an exchange that is
    # not ours. Spending it on the budget makes an occupied frequency the reason
    # we go on calling over the occupant -- which is what rig-session-20260813-235522
    # did off one refused PACTOR-3 codeword, the only thing it decoded in 31 cycles.
    budget = ArqConfig(max_connect_retries=3)
    for label, ev in (("a PACTOR-3 frame -- somebody else's link, three cycles in",
                       _cs(pactor1.CS_ACK_A, Protocol.PACTOR3)),
                      ("a bare CS3 -- the head of somebody else's break-in",
                       _cs(pactor1.CS_CHANGEOVER)),
                      ("a bare CS2 -- the other half of somebody else's alternation",
                       _cs(pactor1.CS_ACK_B))):
        q = PtcHost(peer=None, mycall="W9SSJ")
        q.arq.cfg = budget
        q.arq.on_host_connect("W9SSJ", "KI0BK")
        for _ in range(8):
            q.on_rx_event(ev)
            q.arq.on_cycle()
        check(f"{label} does not buy a retry",
              q.arq.state == State.DISCONNECTED, f"state {q.arq.state}")
        check("...and the call is still refused, not answered",
              "is not an answer to a call" in "\n".join(q.log_lines))

    # The control: energy that identifies NOBODY still holds the budget open,
    # which is the whole reason the forgiveness exists (WS8EOC went on calling
    # long after shrike had given up on it).
    q2 = PtcHost(peer=None, mycall="W9SSJ")
    q2.arq.cfg = budget
    q2.arq.on_host_connect("W9SSJ", "KI0BK")
    for _ in range(8):
        q2.on_rx_event(rxfront.Event(t=0.0, kind="fsk", text="200 Bd FSK"))
        q2.arq.on_cycle()
    check("...while unidentified energy at the peer's tones still does",
          q2.arq.state == State.CONNECTING, f"state {q2.arq.state}")

    print("\nALL PASS")
    return 0


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
