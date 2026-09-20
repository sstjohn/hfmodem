# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What a PACTOR-1 PEER requires of our first data packet -- not what a monitor accepts.

A monitor decodes whatever it is shown. It tries both shift senses, has no session
state, and no opinion about which cycle a packet arrived in. Every packet shrike
has ever built passes that bar, and on 2026-07-27 a real gateway answered our call
with CS4 at zero errors twenty cycles running and never once advanced -- which,
per the reference's responder, is that station saying our packet does not decode. An
independent monitor read every one of those packets happily throughout. This file
is the bar a monitor cannot be.

What a peer requires, and what each assertion here holds:

  * the CRC region reduces to the reference's accept residue, checked against an
    independent bitwise implementation rather than against `coding.crc16`;
  * the packet is read in the shift the PEER'S CYCLE COUNT names -- one inversion
    per 1.25 s cycle since the call it locked to, whether or not we transmitted in
    that cycle -- and is refused in the other, with the refusal reducing to the
    reference's own INVERTED-FCS constant;
  * the shift phase survives SKIPPED SLOTS, which is what actually broke: shrike
    counted transmissions and its cycle guard was skipping every other slot;
  * the status byte, the mod-4 counter and the field contents are what a called
    station is told to expect of the first packet after a connect;
  * the acknowledgement we key once the link turns round carries the OPPOSITE
    shift to the packet it answers, which is what the changeover's 840 ms
    rotation leaves behind and the reverse of the rule that held before it.

The negative controls are the point. Each property is also asserted to FAIL for
the behaviour that shipped, because a test that only exercises the fixed path
cannot tell the fix from the bug -- and this whole failure was invisible to a
suite full of encoder/decoder round trips that shared the mistake.

Run: python -m hfmodem.tests.shrike.test_p1peer
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from hfmodem.shrike import onair, p1rx, pactor1, ptc

FS = 48000
CYCLE_N = 60000                 # 1.25 s at 48 kHz
ACCEPT_FCS = 0x0F47             # what the CRC region reduces to on a good packet
INVERTED_FCS_11 = 0xC438        # ...and what it gives when read in the wrong shift
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


# -- the peer, modelled from the reference's responder -----------------------
#
# Bitwise and table-free ON PURPOSE. `coding.crc16` is what built the packet, so
# reusing it here would assert only that a function agrees with itself. This is
# the reflected CCITT with preset 0xFFFF and a final complement, spelled out.
def _fcs(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return (~crc) & 0xFFFF


def peer_read(audio: np.ndarray, *, shift: int, baud: int = 100,
              expect_counter: int = 0) -> dict:
    """One cycle of a called station's packet receive, at a FIXED shift and offset.

    No search, no second sense, no sliding: the peer read is a hard decision at a
    point, which is exactly why a monitor's verdict says nothing about it.
    """
    nbytes = 12 if baud == 100 else 24
    sps = int(FS / baud)
    n = nbytes * 8
    seg = np.asarray(audio, float)[:n * sps]
    if seg.size < n * sps:
        return {"ok": False, "why": "burst shorter than the packet slot"}
    t = np.arange(sps) / FS
    mark = np.exp(-2j * np.pi * pactor1.MARK * t)
    space = np.exp(-2j * np.pi * pactor1.SPACE * t)
    bits = []
    for i in range(n):
        w = seg[i * sps:(i + 1) * sps]
        bits.append(int((abs(w @ mark) > abs(w @ space)) ^ shift))
    pkt = bytes(int("".join(str(b) for b in bits[i:i + 8][::-1]), 2)
                for i in range(0, n, 8))
    fcs = _fcs(pkt[1:])
    r = {"bytes": pkt, "header": pkt[0], "fcs": fcs, "ok": fcs == ACCEPT_FCS}
    if not r["ok"]:
        r["why"] = f"CRC region reduces to {fcs:#06x}, not {ACCEPT_FCS:#06x}"
        return r
    stat = pkt[-3]
    r.update(status=stat, counter=stat & 3, compression=(stat >> 2) & 3,
             brkin=bool(stat & 0x40), qrt=bool(stat & 0x80),
             new=(stat & 3) != expect_counter,
             field=pkt[1:-3].rstrip(bytes([pactor1.IDLE])))
    if r["compression"] not in (0, 1):
        r["ok"], r["why"] = False, f"unknown Datenmodus {r['compression']}"
    return r


# -- the reference's own registers, stepped as pactor.c steps them -----------
#
# `cycle_end` inverts `rxinv` and `txinv` together, once per cycle, in both roles
# (pactor.c:660). The changeover moves a TIME and not a polarity: the new ISS adds
# 840 ms to its receive anchor (`rx_tx_100`, pactor.c:1122) and the new IRS adds it
# to its transmit anchor (`tx_rx_100`, pactor.c:1214). Each station is stepped
# through its own states here and the two are put together only on the air, by
# time -- so what an emission and the listening it lands in have in common is
# measured rather than asserted.
CYCLE_US = 1_250_000
FEC_US = 960_000
CS_US = 120_000
ROT_US = FEC_US - CS_US


class _Station:
    def __init__(self, name: str) -> None:
        self.name, self.log = name, []
        self.txtime = self.rxtime = self.txinv = self.rxinv = 0

    def cycle_end(self) -> None:                            # :660
        self.rxtime += CYCLE_US
        self.txtime += CYCLE_US
        self.rxinv = 1 - self.rxinv
        self.txinv = 1 - self.txinv

    def send(self, what: str) -> None:
        self.log.append((self.txtime, "send", what, self.txinv))

    def receive(self, what: str) -> None:
        self.log.append((self.rxtime, "listen", what, self.rxinv))


def reference_air(cycles: int = 3) -> list[tuple[int, int, str, str, int, int]]:
    """(when, gap, emitter, what, shift sent, shift listened for) on the air.

    Every register assignment below is the pactor.c line beside it; the pairing
    is done afterwards, by which listening each emission falls in.
    """
    master, gateway = _Station("master"), _Station("gateway")

    master.rxtime = master.txtime + FEC_US                  # :1884
    master.send("CALL")
    gateway.rxinv = gateway.txinv = master.txinv            # :1744
    gateway.txtime = gateway.rxtime + FEC_US + 10_000       # :1745
    master.rxinv = gateway.txinv                            # :1893, the sense seen
    master.rxtime += CYCLE_US                               # :1897
    master.txtime += CYCLE_US                               # :1898
    master.txinv = 1 - master.txinv                         # :1901
    master.rxinv = 1 - master.rxinv                         # :1902
    master.send("CALL verify")                              # :1826
    master.receive("CS")                                    # :1829
    master.txtime += CYCLE_US                               # :1904
    master.txinv = 1 - master.txinv                         # :1905

    for _ in range(cycles + 1):                             # rx_100_csN, :1071
        gateway.send("CS")
        gateway.cycle_end()
        gateway.receive("packet")
    for k in range(cycles):                                 # tx_100_csN, :1156
        master.send(f"PKT{k}")
        master.cycle_end()
        master.receive("CS")

    master.txtime += ROT_US                                 # tx_rx_100, :1214
    gateway.rxtime += ROT_US                                # rx_tx_100, :1122
    for k in range(cycles):                                 # rx_tx_100, :1126
        gateway.send(f"ISS PKT{k}")
        gateway.cycle_end()
        gateway.receive("CS")
    for _ in range(cycles):                                 # rx_100_csN again
        master.send("CS")
        master.cycle_end()
        master.receive("packet")

    out = []
    for one, other in ((master, gateway), (gateway, master)):
        for t, kind, what, inv in one.log:
            if kind != "send":
                continue
            near = [(abs(t - u), u, heard) for u, k, _, heard in other.log
                    if k == "listen" and abs(t - u) < CYCLE_US // 4]
            if near:
                gap, _, heard = min(near)
                out.append((t, gap, one.name, what, inv, heard))
    return sorted(out)


# -- a session, as the peer counts it ---------------------------------------
def grid(anchor: int = 0) -> onair._MasterGrid:
    return onair._MasterGrid(anchor, CYCLE_N, 0, packet_n=round(0.96 * FS),
                             cs_n=round(0.12 * FS), d_max_n=round(0.13 * FS))


def session(*, cycle_locked: bool, slots_per_tx: int, calls: int = 3,
            packets: int = 4) -> list[tuple[int, bool]]:
    """(cycle offset from the call, shift we transmitted in) for each burst.

    Drives the real `RadioTx` shift logic over a real grid, so this measures
    shipped behaviour rather than a restatement of it. `cycle_locked=False`
    reproduces the per-transmission toggle that shipped.
    """
    tx = onair.RadioTx(None, transmit=False, outdir=Path("/tmp"))
    tx._tx = lambda audio, what, **kw: None
    raster = grid()
    out: list[tuple[int, bool]] = []
    slot = 0
    for _ in range(calls + packets):
        if cycle_locked:
            tx.aim(raster, slot)
        else:
            tx.boundary, tx.invert = raster.boundary(slot), None
        out.append((slot, tx._flip()))
        slot += slots_per_tx
    return out


def main() -> int:
    mycall = "W9SSJ"
    first = f"1{mycall.lower()}\r".encode()

    # 1. The packet a peer is told to expect after a CS4 answer, built by the real
    #    session layer rather than by hand -- so a change in ptc.py reaches here.
    tx_log: list[dict] = []
    real = pactor1.packet_signal

    def spy(payload=b"", baud=100, **kw):
        tx_log.append(dict(payload=payload, baud=baud, **kw))
        return real(payload, baud, **kw)

    pactor1.packet_signal = spy
    try:
        tx = onair.RadioTx(None, transmit=False, outdir=Path("/tmp"))
        tx._tx = lambda audio, what, **kw: None
        host = ptc.PtcHost(peer=tx, mycall=mycall)
        tx.attach(host)
        host.arq.on_host_connect(mycall, "WS8EOC")
        host.tick()

        class CS4:
            kind, cs, protocol = "cs", pactor1.CS_SPEED, "PACTOR-1"

        host.on_rx_event(CS4())
        host.tick()
    finally:
        pactor1.packet_signal = real

    check("a CS4 answer produces a data packet, not another control signal",
          len(tx_log) == 1, f"{len(tx_log)} packets")
    if not tx_log:
        print("\nFAILED")
        return 1
    sent = tx_log[0]
    check("CS4 selects 100 Bd", sent["baud"] == 100, f"{sent['baud']} Bd")
    check("the first packet carries the level and the master's call, CR-terminated",
          sent["payload"] == first, repr(sent["payload"]))

    audio = pactor1.packet_signal(sent["payload"], sent["baud"],
                                  packet_count=sent["packet_count"],
                                  invert=False, lead_s=0, tail_s=0)

    # 2. The peer's accept test, against an independent CRC. A packet is accepted
    #    on the RESIDUE of the whole 11-byte region, not on a recomputation of the
    #    trailer, so this also pins the trailer's byte order: swap it and the
    #    residue is not 0x0F47.
    good = peer_read(audio, shift=0)
    check("the CRC region reduces to the reference's accept residue",
          good["ok"], good.get("why", f"{good['fcs']:#06x}"))
    check("the peer sees packet counter 1 -- the value the first packet must carry",
          good.get("counter") == 1, str(good.get("counter")))
    check("...and it is NEW against a called station's initial counter of 0",
          bool(good.get("new")))
    check("Datenmodus is 8-bit ASCII, so the field is read as bytes",
          good.get("compression") == 0, str(good.get("compression")))
    check("no break-in and no QRT on the first packet",
          not good.get("brkin") and not good.get("qrt"))
    check("the field is the callsign line, padded with IDLE and nothing else",
          good.get("field") == first, repr(good.get("field")))

    # 3. The shift, and the constant that names the failure. A peer reading the
    #    other sense does not get noise -- it gets the reference's own
    #    INVERTED-FCS value, which is what makes this diagnosable from the far end
    #    and what makes the assertion sharp rather than "it failed somehow".
    wrong = peer_read(audio, shift=1)
    check("the same packet is REFUSED in the other shift",
          not wrong["ok"], wrong.get("why", ""))
    check("...and refused with the reference's INVERTED-FCS, so it is a shift "
          "error and not a byte error",
          wrong["fcs"] == INVERTED_FCS_11,
          f"{wrong['fcs']:#06x} vs {INVERTED_FCS_11:#06x}")

    # A monitor accepts both. Stated here so no future reader mistakes a monitor
    # decode for evidence about a peer.
    check("a both-senses reader accepts BOTH -- which is why a monitor read the "
          "packets happily through the whole failure",
          peer_read(audio, shift=0)["ok"] and peer_read(audio, shift=1)["fcs"]
          == INVERTED_FCS_11)

    # 4. THE REGRESSION. The peer expects one inversion per 1.25 s cycle since the
    #    call it locked to. Drive the shipped shift logic over a grid that SKIPS
    #    slots -- which is what the cycle guard was doing -- and every packet must
    #    still land in the sense that cycle names.
    for slots in (1, 2, 3):
        bursts = session(cycle_locked=True, slots_per_tx=slots)
        epoch = bursts[0][0]
        bad = [(s, inv) for s, inv in bursts if inv != bool((s - epoch) & 1)]
        check(f"shift follows the CYCLE at a {slots}-slot cadence",
              not bad, f"{len(bad)} of {len(bursts)} in the wrong sense: {bad}")

    # ...and the same check must FAIL for what shipped, or it proves nothing. At a
    # one-slot cadence the two rules coincide, which is exactly why this was
    # invisible until the guard started skipping slots.
    shipped2 = session(cycle_locked=False, slots_per_tx=2)
    off2 = [(s, inv) for s, inv in shipped2 if inv != bool((s - shipped2[0][0]) & 1)]
    check("NEGATIVE CONTROL: counting transmissions fails this at a 2-slot cadence",
          len(off2) >= len(shipped2) // 2,
          f"{len(off2)} of {len(shipped2)} wrong")
    shipped1 = session(cycle_locked=False, slots_per_tx=1)
    off1 = [(s, inv) for s, inv in shipped1 if inv != bool((s - shipped1[0][0]) & 1)]
    check("NEGATIVE CONTROL: ...and passes at a 1-slot cadence, which is why the "
          "defect hid",
          not off1, f"{len(off1)} wrong")

    # 5. Carry it through the modulator: the sense the grid names must be the sense
    #    the audio carries. The two are separate readings of one convention and
    #    nothing else makes them move together.
    for slot, inv in session(cycle_locked=True, slots_per_tx=2)[3:]:
        a = pactor1.packet_signal(first, 100, packet_count=1, invert=inv,
                                  lead_s=0, tail_s=0)
        r = peer_read(a, shift=int(bool(slot & 1)))
        check(f"cycle offset {slot}: the peer's own shift count reads it",
              r["ok"], r.get("why", ""))

    # 6. THE CALL TRAIN. The first call is transmitted before the grid exists --
    #    it is the burst the anchor is measured from -- so it takes the dry-run
    #    toggle and is slot 0. Every call after it is aimed. If the first two share
    #    a shift, a peer that locked to the first reads every later transmission
    #    one cycle out of phase, in EVERY cycle, for the rest of the link: twenty
    #    packets, the same codeword back every time, no advance.
    tx = onair.RadioTx(None, transmit=False, outdir=Path("/tmp"))
    tx._tx = lambda audio, what, **kw: None
    raster = grid()
    train = [tx._flip()]                       # the call the anchor comes from
    for slot in (1, 2, 3):
        tx.aim(raster, slot)
        train.append(tx._flip())
    check("the call train inverts every cycle, first call included",
          train == [False, True, False, True], f"{train}")

    # ...and the negative control: an epoch pinned on the first AIMED call, which
    # is what shipped. Slot 1 then reads as the epoch and the first two calls go
    # out in the same shift.
    pinned = grid()
    pinned.shift_slot = 1
    shipped = [False] + [pinned.shift(s) for s in (1, 2, 3)]
    check("NEGATIVE CONTROL: pinning the epoch on the second call collides them",
          shipped[0] == shipped[1], f"{shipped}")

    # 7. THE ANCHOR MOVES, AND THE SHIFT MUST NOT. Placing the grid after a hush
    #    slides the anchor by up to half a cycle, and each changeover slides it
    #    840 ms. A shift counted in SAMPLES since the call rotates with it,
    #    silently, in the middle of a live link; a shift counted in SLOTS cannot.
    moved = grid()
    before = [moved.shift(s) for s in range(8)]
    moved.anchor += 17_000                     # a placement's phase correction
    moved.reverse(to_iss=False)                # ...a changeover, which rotates
    moved.reverse(to_iss=True)                 # ...the way back, which does not
    moved.anchor += CYCLE_N // 2               # ...and a second placement
    check("an anchor move leaves every slot's shift where it was",
          [moved.shift(s) for s in range(8)] == before,
          f"{[moved.shift(s) for s in range(8)]} vs {before}")
    sample_rule = [bool(((moved.boundary(s) - 0) // CYCLE_N) & 1) for s in range(8)]
    check("NEGATIVE CONTROL: counting samples since the call does not survive it",
          sample_rule != before, f"{sample_rule} vs {before}")

    # 8. THE PEER'S PHASE WINS. Which of our calls the far end locked to is not
    #    observable from this end, so a count that free-runs from our own first
    #    call can be a whole cycle out and nothing in a round trip will say so.
    #    The peer's control signal does say so: within one cycle the two
    #    directions share the shift, so the sense its answer arrives in IS the
    #    sense it expects our packet in, that cycle (pactor1-data-packets.md §7).
    peer_phase = {s: bool((s + 1) & 1) for s in range(8)}       # one cycle out
    wrongly = grid()
    off = [s for s in range(8) if wrongly.shift(s) != peer_phase[s]]
    check("a phase one cycle out is wrong in EVERY cycle, not every other one",
          len(off) == 8, f"{len(off)} of 8")
    line = wrongly.align(3, int(peer_phase[3]))
    check("...and one control signal realigns it", line is not None, str(line))
    check("...to the peer's phase in every slot, not just the one heard",
          all(wrongly.shift(s) == peer_phase[s] for s in range(8)))
    check("...and a second reading of the same phase moves nothing",
          wrongly.align(4, int(peer_phase[4])) is None)

    # ...and the number that reaches `align` has to come off AUDIO, which is the
    # seam the whole failure lived in. Everything above hands `align` an integer
    # the test made up, so it passes for any convention the decoder happens to
    # report in -- and a control signal reported in a convention of its own puts
    # every packet one parity out. One shift, one cycle, both directions:
    # pactor1-control-signals.md §6.1, measured on three W4DNA cycles.
    quiet = np.zeros(int(0.3 * FS), np.float32)
    for s in (0, 1):
        heard = p1rx.decode_control_signal(
            np.concatenate([quiet,
                            pactor1.control_signal(pactor1.CS_ACK_A, invert=bool(s)),
                            quiet]), 0.3, 0.120)
        check(f"a control signal sent in shift {s} decodes to sense {s}",
              heard is not None and heard.errors == 0 and heard.sense == s,
              f"got {heard}")
        heeded = grid()
        heeded.align(2, heard.sense)
        for wrong in (False, True):
            a = pactor1.packet_signal(first, 100, packet_count=1,
                                      invert=heeded.shift(2) ^ wrong,
                                      lead_s=0, tail_s=0)
            r = peer_read(a, shift=s)
            if wrong:
                check(f"shift {s}: NEGATIVE CONTROL: one parity out is the "
                      f"inverted FCS, every cycle -- the gateway's whole answer",
                      r["fcs"] == INVERTED_FCS_11, f"{r['fcs']:#06x}")
            else:
                check(f"shift {s}: the packet the grid sends after that answer "
                      f"is the one the peer reads", r["ok"], r.get("why", ""))

    # Carry it to the air: the packet the realigned grid sends is the one the
    # peer's own hard-decision read accepts, and the un-realigned one is refused
    # with the inverted-FCS constant that names the fault.
    for slot in (4, 5):
        tx.aim(wrongly, slot)
        a = pactor1.packet_signal(first, 100, packet_count=1, invert=tx._flip(),
                                  lead_s=0, tail_s=0)
        r = peer_read(a, shift=int(peer_phase[slot]))
        check(f"slot {slot}: the realigned packet reads at the peer's shift",
              r["ok"], r.get("why", ""))
        bad = peer_read(pactor1.packet_signal(first, 100, packet_count=1,
                                              invert=not tx._flip(), lead_s=0,
                                              tail_s=0),
                        shift=int(peer_phase[slot]))
        check(f"slot {slot}: NEGATIVE CONTROL: the un-realigned one is refused",
              bad["fcs"] == INVERTED_FCS_11, f"{bad['fcs']:#06x}")

    # 9. ONE PACKET A CYCLE, against the train WS8EOC actually sent: CS1 five
    #    times running, then CS4 fifteen times. A repeated codeword is a repeat
    #    request, so this is a station saying "again" twenty times -- and each of
    #    those answers reaches the FSM inside the cycle it arrived in, where the
    #    cycle tick then follows it. Two 0.96 s packets do not fit in a 1.25 s
    #    cycle, and the second lands squarely in the 0.29 s the peer answers in.
    #
    #    Unreachable while no control signal decoded, which is every session so
    #    far. It is reachable now.
    sent_per_cycle, baud_at = [[]], []
    spy_tx = onair.RadioTx(None, transmit=False, outdir=Path("/tmp"))
    spy_tx._tx = lambda audio, what, **kw: sent_per_cycle[-1].append(what)
    raster = grid()
    host = ptc.PtcHost(peer=spy_tx, mycall=mycall)
    host.arq.on_host_connect(mycall, "WS8EOC")

    class Answer:
        kind, protocol = "cs", "PACTOR-1"

        def __init__(self, cs):
            self.cs = cs

    for slot, cs in enumerate([pactor1.CS_ACK_A] * 5 + [pactor1.CS_SPEED] * 15,
                              start=1):
        spy_tx.aim(raster, slot)
        sent_per_cycle.append([])
        host.on_rx_event(Answer(cs))
        host.tick()
        baud_at.append(host.p1_baud)
    del sent_per_cycle[0]                      # the call, sent before the grid
    over = [(i + 1, w) for i, w in enumerate(sent_per_cycle) if len(w) > 1]
    check("no cycle of the CS1/CS4 train transmits twice",
          not over, f"{len(over)} doubled: {over[:3]}")
    # AND THE RATE OVER THE WHOLE TRAIN, because the run in front of the CS4s is
    # the same run both lost gateways answered with. Four repeats of the connect
    # answer gear the link down on their own (`P1_HISPEED_RETRIES`), so the first
    # CS4 arrives at a link already at 100 Bd -- and it must NOT read as the
    # speed-up there, because the block it is asking for is the link's first at
    # 100 and §4.3 puts the speed-up behind that block's acknowledgement.
    # Without that the link bounces back to the rate it just left, every five
    # cycles, for as long as the station holds CS4.
    check("the CS1 run gears down at 200 Bd, and the CS4 train holds 100",
          baud_at == [200] * 4 + [100] * 16, f"{baud_at}")
    # ...and it must not go silent instead, which is the other way to have one
    # transmission a cycle and the way that ends a QSO just as dead.
    answered = sum(1 for w in sent_per_cycle if w)
    check("...and the station answers the repeat request rather than going quiet",
          answered >= 14, f"{answered} of {len(sent_per_cycle)} cycles keyed")

    # 10. A RUN OF THE CONNECT ANSWER IS "NOT YET", AND THE COUNTER MUST NOT MOVE
    #     ON IT.
    #
    # `#1 x9` against eight consecutive CS1 at zero errors reads like a deadlock:
    # perhaps the peer is acknowledging and this layer refuses to advance. It is
    # not, and the reason needs no air. "An accepting station sends its answer every
    # cycle until the caller's first data packet decodes" -- so were acceptance
    # signalled by the SAME codeword the connect was answered with, no caller could
    # ever tell "still waiting" from "got it". Acceptance is a codeword CHANGE.
    #
    # Both halves are pinned here because the tempting fix -- not seeding
    # `_last_rx_cs` from the connect answer -- makes the first CS1 an ACK and walks
    # the counter over a packet the peer never received.
    #
    # The exception is a CS4 in between, which is the one train the two rules
    # above cannot read together. A CS1 connect answer puts the link at 200 Bd; a
    # CS4 there is the REJECT, and the re-chunked block that follows it is the
    # link's FIRST 100 Bd block, whose acknowledgement the description names
    # absolutely rather than by the alternation -- "CS4 dient als 'REQUEST'-CS
    # fuer den ersten 100-Bd-Block, Bestaetigung erfolgt mittels CS1 oder CS3".
    # Held against the CS1 that answered the connect, that acknowledgement reads
    # as a repeat and the link cannot advance past packet #1 for as long as it
    # lasts. Measured against K4MSU on 3595 kHz, `captures/onair-0819-2048`: CS1
    # answer, three CS4, then eleven CS1 at zero bit errors, and `#1 x16` on the
    # air. The same gateway answering CS4 ninety minutes later
    # (`captures/onair-0819-2207`) advanced on its first CS1.
    print("\nThe counter, against the four things a peer can be doing")
    for label, train, want in (
            ("holding CS1 -- our packet has not decoded", [0] * 6, [1] * 6),
            ("holding CS4 -- the same, at 100 Bd", [3] * 6, [1] * 6),
            ("CS1 then ALTERNATING -- it is acknowledging",
             [0, 1, 0, 1, 0, 1], [1, 2, 3, 0, 1, 2]),
            ("CS1, a CS4 REJECT, then CS1 -- the first 100 Bd block is taken",
             [0] + [3] * 3 + [0] * 11, [1] * 4 + [2] * 11)):
        spy = onair.RadioTx(None, transmit=False, outdir=Path("/tmp"))
        spy._tx = lambda audio, what, **kw: None
        h = ptc.PtcHost(peer=spy, mycall=mycall)
        h.stay_in_pactor1 = True     # one variable at a time; see PtcHost
        h.arq.on_host_connect(mycall, "WS8EOC")
        for cs in train:
            h.on_rx_event(Answer(cs))
            h.tick()
        check(f"{label}: counters {want}", spy.p1_seq == want, f"{spy.p1_seq}")

    # 11. THE CHANGEOVER TURNS THE ACKNOWLEDGEMENT'S SHIFT OVER.
    #
    #     While we hold the channel, the answer shares our shift. The called
    #     station takes `rxinv = txinv` from the call it locked to and answers
    #     inside that same cycle (pactor.c:1744), so its control signal comes back
    #     in the shift the packet went out in. On the air, 20 of 20 readable
    #     cycles across five 2026-08-13..15 sessions, and 6 of 6 read off
    #     `stream.wav` -- our own emission and the answer on one sample clock --
    #     on 2026-08-19.
    #
    #     Once the link turns round it is the other way, and the alternation
    #     alone cannot say so: the rotation moves a TIME, not a polarity. The new
    #     ISS puts 840 ms on its receive anchor and the new IRS puts it on its
    #     transmit anchor, and `cycle_end` goes on inverting once a cycle in both
    #     of them. The acknowledgement therefore lands in the next cycle's shift
    #     while the packet it answers keeps the last one -- and both stations do
    #     it, so the ISS listens with exactly the polarity the IRS sends.
    #
    #     Reading it the other way makes an inverted codeword, which the reference
    #     accepts in no state: `receive_cs` returns 9-12 for a word that matched
    #     at distance twelve, and every transmit state tests i against 1-6
    #     (pactor.c:1131-1207).
    air = reference_air()
    check("both stations hear every burst in the shift it was sent in",
          all(sent == heard for _, _, _, _, sent, heard in air),
          f"{[(w, s, h) for _, _, _, w, s, h in air if s != h]}")
    turned = next(i for i, r in enumerate(air) if r[3].startswith("ISS"))
    for i, (_, _, who, _, shift, _) in enumerate(air):
        if not i or (who == "gateway") != (i < turned):
            continue
        answered = air[i - 1]
        if i < turned:
            check(f"as ISS, the answer to {answered[3]} carries its shift",
                  shift == answered[4], f"{shift} against {answered[4]}")
        elif i > turned:
            check(f"as IRS, our answer to {answered[3]} carries the OTHER shift",
                  shift != answered[4], f"{shift} against {answered[4]}")

    # ...and the grid does that, without being told about it. The rotation is the
    # whole of it: `reverse` moves the transmit anchor and `shift` goes on
    # counting slots, so the ack for the packet on slot k-1 keys on slot k.
    turn = grid()
    turn.d_n = round(0.09 * FS)
    for k in (4, 5, 6):
        check(f"slot {k}: as ISS the answer we wait for rides the slot we sent on",
              turn.boundary(k) < turn.rx_due(k) < turn.boundary(k + 1),
              f"{turn.boundary(k)} < {turn.rx_due(k)} < {turn.boundary(k + 1)}")
    turn.reverse(to_iss=False)
    for k in (7, 8, 9):
        ends = turn.rx_due(k - 1) + turn.data_n
        check(f"slot {k}: as IRS the ack keys behind the packet on slot {k - 1}",
              0 < turn.boundary(k) - ends <= turn.d_max_n,
              f"{(turn.boundary(k) - ends) / FS * 1e3:+.0f} ms after it ends")
        check(f"slot {k}: ...and in the shift that packet did NOT ride",
              turn.shift(k) != turn.shift(k - 1))

    # NEGATIVE CONTROL, at the codeword: an acknowledgement carried across the
    # reversal in the packet's own shift is the twelve-bit inverse of the one the
    # ISS reads, which is `receive_cs`'s 9-12 and no state's accept.
    for packet_shift in (False, True):
        quiet = np.zeros(int(0.3 * FS), np.float32)
        for held, want in ((packet_shift, "REFUSED"), (not packet_shift, "read")):
            got = p1rx.decode_control_signal(
                np.concatenate([quiet,
                                pactor1.control_signal(pactor1.CS_ACK_A,
                                                       invert=held),
                                quiet]), 0.3, 0.120)
            reads = got is not None and got.sense == int(not packet_shift)
            check(f"packet in shift {int(packet_shift)}: an ack held in shift "
                  f"{int(held)} is {want} by the ISS",
                  reads == (want == "read"), f"{got}")

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
