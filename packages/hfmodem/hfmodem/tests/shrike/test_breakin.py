# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A direction reversal, as the PEER experiences it -- not as our decoder accepts it.

PACTOR-1 reverses direction by break-in, and a break-in is not a burst. "In
contrast to AMTOR, CS3 is transmitted as head portion of a special changeover
packet": the receiving station sends its OWN first packet, whose first 120 ms are
the CS3 codeword, and the station yielding switches to receive on hearing that and
reads the remaining 840 ms of the same transmission before answering.

Everything about the cycle grid follows from that one sentence, and none of it is
optional:

  * the changeover packet is 960 ms like any other, so the field gives up a byte at
    100 Bd and two at 200 to make room for the head;
  * it occupies the CONTROL-SIGNAL slot, so both stations rotate their grid by
    960 - 120 = 840 ms in the same cycle -- the station becoming ISS on its receive
    anchor, the station becoming IRS on its transmit anchor;
  * and the turnaround inverts, `d -> 170 - d + 2p`, which is what makes the two
    grids stay mutually consistent for any `d` at all.

The budget is the sharp edge, and it comes out the same whichever way a station
recognises the reversal:

    cycle - settle - packet - d - CS   =   170 - d - settle

is BOTH the gap from the CS3 head ending to an un-rotated key AND the gap from the
whole packet ending to the rotated one. The 840 ms rotation buys back exactly the
keying settle, so reading the packet costs a station nothing over reading the head.
At the FT-891's 40 ms that is 130 - d: 90 ms at d = 40, 40 ms at d = 90.

This file models what the far end requires, the way `test_p1peer` does. Our own
decoder agreeing with our own encoder is worth nothing here -- both ends of the
audio exchange below share every assumption in this repository, so the frame is
also read by a hand-rolled demodulator with an independent CRC, and the grid is
solved as arithmetic rather than asserted from the code that implements it.

Run: python -m hfmodem.tests.shrike.test_breakin
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, pactor1, rxfront, spec
from hfmodem.shrike.arq import CS_BREAKIN, IRS, ISS, ArqConfig, State
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike import archive
from hfmodem.tests.shrike import test_qso as qso

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
PACKET_N = round(spec.P1_PACKET_S * FS)
CS_N = round(spec.P1_CS_S * FS)
ROT_N = PACKET_N - CS_N                  # 840 ms, the changeover constant
SETTLE_N = round(0.04 * FS)              # the FT-891's, and the only settle that closes
ACCEPT_FCS = 0x0F47                      # X.25's accept residue, complemented
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def budget_n(d_n: int, settle_n: int = SETTLE_N) -> int:
    """Samples between recognising the reversal and having to key."""
    return SLOT_N - settle_n - PACKET_N - d_n - CS_N


# --------------------------------------------------------------------------
# a peer's reader: no search, no second sense, no library of ours
# --------------------------------------------------------------------------
def _fcs(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return (~crc) & 0xFFFF


def peer_bits(audio: np.ndarray, baud: int, n: int) -> list[int]:
    """`n` hard bit decisions at `baud`, read at a point, MARK as a one.

    ONE tone sense, and this is where a changeover packet proves it: the head is
    compared against the codeword table and the bytes behind it against the frame,
    off the SAME twelve hundred bit decisions. A reader needing to complement one
    of the two halves would be reading a packet no station on the band sends.
    """
    sps = int(FS / baud)
    t = np.arange(sps) / FS
    mark = np.exp(-2j * np.pi * pactor1.MARK * t)
    space = np.exp(-2j * np.pi * pactor1.SPACE * t)
    out = []
    for i in range(n):
        w = np.asarray(audio, float)[i * sps:(i + 1) * sps]
        if w.size < sps:
            break
        out.append(int(abs(w @ mark) > abs(w @ space)))
    return out


def peer_read(audio: np.ndarray, baud: int) -> dict:
    """One changeover packet as the far end reads it: head, field, status, CRC."""
    nbytes, nhead = (12, 2) if baud == 100 else (24, 3)
    bits = peer_bits(audio, baud, nbytes * 8)
    if len(bits) < nbytes * 8:
        return {"ok": False, "why": "burst shorter than a packet slot"}
    pkt = bytes(int("".join(str(b) for b in bits[i:i + 8][::-1]), 2)
                for i in range(0, nbytes * 8, 8))
    fcs = _fcs(pkt[nhead:])
    r = {"bytes": pkt, "head_bits": bits[:pactor1.CS_BITS * (baud // 100)],
         "fcs": fcs, "ok": fcs == ACCEPT_FCS,
         "field": pkt[nhead:-3].rstrip(bytes([pactor1.IDLE])),
         "status": pkt[-3], "counter": pkt[-3] & 3}
    if not r["ok"]:
        r["why"] = f"CRC region reduces to {fcs:#06x}, not {ACCEPT_FCS:#06x}"
    return r


# --------------------------------------------------------------------------
# 1. the changeover packet
# --------------------------------------------------------------------------
def the_packet(payload: bytes = b"DE K7ABC") -> None:
    print("\nThe changeover packet, as a peer reads it")
    cs3 = [(pactor1.CONTROL_SIGNALS[pactor1.CS_CHANGEOVER] >> i) & 1
           for i in range(pactor1.CS_BITS)]
    for baud in (100, 200):
        a = pactor1.breakin_signal(payload, baud, lead_s=0, tail_s=0)
        check(f"{baud} Bd: the packet is 960 ms, the same slot a data packet fills",
              abs(a.size - PACKET_N) <= 1, f"{a.size / FS * 1e3:.0f} ms")
        r = peer_read(a, baud)
        check(f"{baud} Bd: the CRC region begins after the head and reduces to "
              f"the accept residue", r["ok"], r.get("why", ""))
        check(f"{baud} Bd: the field is what we put in it, IDLE-padded",
              r.get("field") == payload[:pactor1.BREAKIN_FIELD[baud]],
              repr(r.get("field")))
        check(f"{baud} Bd: the counter is 0 -- it resets across the reversal",
              r.get("counter") == 0, str(r.get("counter")))
        # The head, read at 100 Bd whatever the packet's rate: that is the whole
        # point of the doubling, and it is why a station looking for a control
        # signal does not have to know which rate the break-in came at.
        head = peer_bits(a, 100, pactor1.CS_BITS)
        check(f"{baud} Bd: the first 120 ms IS CS3, read at 100 Bd", head == cs3,
              f"{head} vs {cs3}")
        if baud == 200:
            doubled = r["head_bits"]
            check("200 Bd: the twelve bits are sent DOUBLED, filling three bytes",
                  len(doubled) == 24 and all(doubled[2 * i] == doubled[2 * i + 1]
                                             for i in range(12)))
    # A changeover packet has no header byte, so the gate that admits a data
    # packet must refuse it and vice versa. Both frames ride the same tones in the
    # same slot; only the first bytes tell them apart.
    bk = pactor1.breakin_signal(payload, 100)
    dat = pactor1.packet_signal(payload, 100, packet_count=1)
    check("a data packet's own scan does not accept a changeover packet",
          not p1rx.decode_p1_packets(bk))
    check("...nor the changeover scan a data packet",
          not p1rx.decode_p1_packets(dat, breakin=True))
    check("and the changeover scan does accept one, at 0 bit errors of head",
          [p.breakin for p in p1rx.decode_p1_packets(bk, breakin=True)] == [True])


# --------------------------------------------------------------------------
# 2. the grid, solved either side of the reversal
# --------------------------------------------------------------------------
def the_grid(d_ms: float = 90.0) -> None:
    """Two stations' instants across one reversal, as arithmetic.

    A is the master and the ISS; B answers `d` after A's data ends. B breaks in on
    cycle 1. Every instant below is derived from the protocol's own constants and
    then compared against what `_MasterGrid` produces, so the two are independent.
    """
    print(f"\nThe 840 ms rotation, both stations, one cycle (d = {d_ms:.0f} ms)")
    d_n = round(d_ms / 1000 * FS)
    a_d, b_d = d_n, SLOT_N - PACKET_N - CS_N - d_n     # 170 - d, in samples
    a = onair._MasterGrid(0, SLOT_N, 0, packet_n=PACKET_N, cs_n=CS_N,
                          d_max_n=budget_n(0) + CS_N)
    b = onair._MasterGrid(PACKET_N + d_n, SLOT_N, 0, packet_n=PACKET_N, cs_n=CS_N,
                          d_max_n=budget_n(0) + CS_N)
    a.d_n, b.d_n = float(a_d), float(b_d)
    b.sending = False                                  # B starts as the IRS

    check("before: B's control signal lands where A's window predicts it",
          b.boundary(0) == a.rx_due(0), f"{b.boundary(0)} vs {a.rx_due(0)}")
    check("before: A's next packet lands where B's window predicts it",
          a.boundary(1) == b.rx_due(0), f"{a.boundary(1)} vs {b.rx_due(0)}")

    # Cycle 1: B sends its changeover packet in the control-signal slot, and both
    # grids turn over in that same cycle.
    bk_start = b.boundary(1)
    bk_end = bk_start + PACKET_N
    unrotated_key = a.boundary(2) - SETTLE_N
    a.reverse(to_iss=False)
    b.reverse(to_iss=True)

    check("the station yielding moves its TRANSMIT anchor by exactly 840 ms",
          a.boundary(2) - unrotated_key - SETTLE_N == ROT_N,
          f"{(a.boundary(2) - SETTLE_N - unrotated_key) / FS * 1e3:.0f} ms")
    check("the station taking the link does NOT move its transmit anchor",
          b.boundary(2) == bk_start + SLOT_N)
    check("...it moves its RECEIVE anchor by 840 ms, which is its packet growing "
          "from 120 ms to 960",
          b.packet_n - CS_N == ROT_N and a.data_n - a.packet_n == ROT_N)

    check("after: A's acknowledgement lands in the peer's window and not in its "
          "packet", a.boundary(2) == b.rx_due(1) and a.boundary(2) >= bk_end,
          f"key @ {a.boundary(2)}, packet ends {bk_end}, B expects {b.rx_due(1)}")
    check("after: B's next packet lands where A's window now predicts it",
          b.boundary(2) == a.rx_due(2), f"{b.boundary(2)} vs {a.rx_due(2)}")
    check("`d` survives the reversal untouched at both ends -- it is the peer's "
          "property, and the involution returns it",
          (a.d_n, b.d_n) == (float(a_d), float(b_d)))
    check("the turnarounds invert: s + s' = 170 ms with no path delay",
          abs((a_d + b_d) / FS * 1e3 - 170.0) < 0.1,
          f"{(a_d + b_d) / FS * 1e3:.1f} ms")

    # THE REGRESSION, and the reason the rotation is not cosmetic.
    check("NEGATIVE CONTROL: without the rotation the yielding station keys "
          "inside the packet it was told to listen to",
          bk_start < unrotated_key < bk_end,
          f"key would be {(unrotated_key - bk_start) / FS * 1e3:.0f} ms into a "
          f"{PACKET_N / FS * 1e3:.0f} ms packet")

    # ...and the trigger the session loop hangs it on. It is asked twice a cycle
    # and must move nothing when the role has not changed, which is the only way
    # a grid survives the cycles between reversals.
    class _Host:
        protocol = Protocol.PACTOR1

        class arq:
            role = ISS
    g = onair._MasterGrid(0, SLOT_N, 0, packet_n=PACKET_N, cs_n=CS_N, d_max_n=0)
    check("the loop's trigger rotates nothing while the role holds",
          onair._grid_reversal(g, _Host) is None and g.boundary(1) == SLOT_N)
    _Host.arq.role = IRS
    check("...and rotates once when it changes",
          onair._grid_reversal(g, _Host) is not None
          and g.boundary(1) == SLOT_N + ROT_N)


class _Placed(onair.RadioTx):
    """A transmitter that keys nothing and records where each burst was aimed."""

    def __init__(self, settle: float = SETTLE_N / FS):
        super().__init__(None, transmit=False, outdir=Path("."), settle=settle)
        self.at: list[tuple[str, int]] = []

    def _tx(self, audio, what, drive=None, lead_n=0, *, strict_guard=False,
            pulse_offsets=None):
        self.at.append((what, self.boundary))
        # Record the shaped waveform's phase reference separately from its
        # audio onset. This seam models placement, without emitting any audio.
        self.tx_audio_start = self.boundary - lead_n
        self.tx_pulse_offsets = pulse_offsets
        self.placed = False


def the_changeover_placement(d_ms: float = 91.3) -> None:
    """Where the changeover packet keys, against where the peer's packet ended.

    Arm 13's grid: WS8EOC on 80 m, 2026-08-30, 200 Bd, holding d = 91.3 ms
    at hold 8.
    Keyed on our own boundary the changeover packet arrives `170 - d` past the
    peer's, so the margin the yielding station gets is whatever the turnaround
    leaves rather than something this station chose -- 47.4-51.1 ms measured
    over the two arms, against the 97.8-98.7 ms the same station left us. The
    reference modems key theirs 71.3-71.9 ms after our packet's audio ends, at
    three gateways and both speeds, and that instant is the whole of what
    the 2026-09-01 comparison found to fly.
    """
    print(f"\nThe changeover packet's placement (d = {d_ms:.1f} ms)")
    d_n = round(d_ms / 1000 * FS)
    slot = 7
    raster = onair._MasterGrid(0, SLOT_N, 0, packet_n=PACKET_N, cs_n=CS_N,
                               d_max_n=budget_n(0) + CS_N)
    raster.sending = False               # IRS: we key a codeword, the peer a packet
    raster.d_n, raster.d_ref_n = float(d_n), CS_N
    # WHAT THE RECEIVER MEASURED, and the only thing the placement may read:
    # the onset of the peer's packet in the cycle before the one we key in.
    raster.peer_onset = raster.rx_due(slot - 1)
    lead_n = round(onair.BREAKIN_LEAD_S * FS)
    end = raster.peer_packet_end(slot).at_slot

    check("the grid puts the peer's packet ending 170 ms less the turnaround "
          "before our boundary",
          raster.boundary(slot) - end == SLOT_N - PACKET_N - CS_N - d_n,
          f"{(raster.boundary(slot) - end) / FS * 1e3:.1f} ms")
    check("...and with nothing read there is nothing to place against, so the "
          "burst keys where it always did",
          onair._MasterGrid(0, SLOT_N, 0, packet_n=PACKET_N, cs_n=CS_N,
                            d_max_n=0).peer_packet_end(slot) is None)

    for baud in (100, 200):
        tx = _Placed()
        tx.aim(raster, slot)
        tx.breakin_due = True
        tx.send_p1_cs(0)
        tx.aim(raster, slot)
        tx.send_p1_breakin(b"\x1e" * pactor1.BREAKIN_FIELD[baud], baud, 0)
        cs_at, bk_at = tx.at[0][1], tx.at[1][1]
        gap = (bk_at - end) / FS * 1e3
        check(f"{baud} Bd: the changeover packet's first bit is 170 ms less "
              f"the turnaround past the peer's packet -- where this station's "
              f"codewords are acknowledged", bk_at == end + raster.peer_read_gap,
              f"{gap:.1f} ms")
        check(f"{baud} Bd: ...which is the codeword slot it takes, so it lands "
              f"on the instant a control signal would have",
              bk_at == cs_at, f"{(bk_at - cs_at) / FS * 1e3:+.1f} ms")
        check(f"{baud} Bd: the control signal is NOT moved -- it keys on the "
              f"boundary, where the peer acknowledges it",
              cs_at == raster.boundary(slot),
              f"{(cs_at - raster.boundary(slot)) / FS * 1e3:+.1f} ms")

    # The changeover packet is 960 ms at either speed -- the field gives up the
    # bytes, not the cycle -- so the two placements are one instant.
    both = []
    for baud in (100, 200):
        tx = _Placed()
        tx.aim(raster, slot)
        tx.send_p1_breakin(b"\x1e" * pactor1.BREAKIN_FIELD[baud], baud, 0)
        both.append(tx.at[0][1])
    check("both speeds key the changeover at the same instant", both[0] == both[1],
          f"{both[0]} vs {both[1]}")

    # THE CONTROL, which is what flew on 2026-09-04: 18 keyings at +68 to
    # +70 ms, against a reader whose acknowledged window opens at 88.5.
    tx = _Placed()
    tx.breakin_at_boundary = False
    tx.aim(raster, slot)
    tx.breakin_due = True
    tx.send_p1_breakin(b"\x1e" * 18, 200, 0)
    check("--breakin-lead-72 keys it where a reference modem keys its own at "
          "us instead, which is in front of the reader at every turnaround "
          "this gateway has held", tx.at[0][1] == end + lead_n
          and raster.peer_read_gap > lead_n,
          f"{(tx.at[0][1] - end) / FS * 1e3:.1f} ms past the peer's packet, "
          f"{(raster.peer_read_gap - lead_n) / FS * 1e3:.1f} ms short of the "
          f"rule at d = {d_ms:.1f}")

    # ...and the window in front of it, which is the half that has to be decided
    # before the packet licensing the break-in has decoded.
    tx = _Placed()
    check("nothing moves while no changeover is due",
          tx.key_instant(raster, slot) == raster.boundary(slot))
    tx.breakin_due = True
    check("a changeover due closes the window at its own key instant",
          tx.key_instant(raster, slot) == end + raster.peer_read_gap)

    # THE FLOOR. PTT is up a settle ahead of the audio and the frame scan the
    # break-in is licensed by runs in front of that, so an instant under their
    # sum is one the transmitter cannot make.
    for settle_s in (0.040, 0.050, 0.100):
        floor_n = round((settle_s + onair.PREKEY_RESERVE_S) * FS)
        want = end + max(raster.peer_read_gap, floor_n)
        slow = _Placed(settle=settle_s)
        slow.aim(raster, slot)
        slow.send_p1_breakin(b"\x1e" * 18, 200, 0)
        # The peer's next transmission is due a cycle less a packet past this
        # one's ending, and it cancels it on reading our head -- so the head has
        # to be whole before then, which is what bounds the floor.
        spare = end + SLOT_N - PACKET_N - (slow.at[0][1] + CS_N)
        check(f"a {settle_s * 1e3:.0f} ms settle keys the changeover at the "
              f"earliest instant it can reach, on the peer's transmission and "
              f"not on our boundary", slow.at[0][1] == want,
              f"{(slow.at[0][1] - end) / FS * 1e3:.1f} ms past the peer, "
              f"{(slow.at[0][1] - raster.boundary(slot)) / FS * 1e3:+.1f} ms "
              f"on the boundary")
        check("...and the CS3 head is still whole inside the peer's gap",
              spare >= 0, f"{spare / FS * 1e3:.1f} ms to spare")

    # P3 replaces the phase reference of an emitted control, corroborated by
    # two decoded peer packets. Its reverse-direction gap must be measured;
    # neither the old P1 turnaround nor the forward entry response supplies it.
    p3 = onair._MasterGrid(0, SLOT_N, 0, packet_n=PACKET_N, cs_n=CS_N,
                           d_max_n=budget_n(0) + CS_N)
    p3.protocol = Protocol.PACTOR3
    p3.sending = False
    p3.d_n, p3.d_ref_n = float(round(0.080 * FS)), onair.P3_CS_N
    p3.note_p3_packet(313920, round(.810 * FS), SLOT_N)
    tx = _Placed()
    tx.aim(p3, slot)
    check("PACTOR-3: one decoded packet and the inherited P1 turnaround do "
          "not license a changeover placement",
          tx._place_breakin() == "" and "no fresh corroborated" in tx.unplaceable)
    p3.note_p3_control(360000)
    p3.note_p3_packet(373920, round(.810 * FS), SLOT_N)
    # The explicit synthetic samples above measure an 80 ms reverse gap.
    # Changing the unrelated P1 estimate must not move this P3 placement.
    p3.d_n = round(.105 * FS)
    p3_end = p3.peer_packet_end(slot).at_slot
    tx = _Placed()
    tx.aim(p3, slot)
    tx.breakin_due = True
    tx.send_packet(3, b"\x1e" * 3, 0, breakin=True)
    gap = (tx.at[0][1] - p3_end) / FS * 1e3
    check("PACTOR-3: an emitted control and two CRC packet phases establish "
          "the 150 ms forward gap, independently of the old P1 turnaround",
          tx.at[0][1] == p3_end + p3.peer_read_gap
          and p3.peer_read_gap == SLOT_N - p3.data_n - onair.P3_CS_N
          - round(0.080 * FS) and abs(gap - 150.0) < 0.5, f"{gap:.1f} ms")
    check("PACTOR-3: the changeover's leading pulse, not its filter skirt, "
          "replaces that control phase",
          tx.tx_pulse_offsets is not None and min(tx.tx_pulse_offsets) > 0
          and tx.tx_audio_start + min(tx.tx_pulse_offsets) == tx.at[0][1])
    check("...and the FSM's standing intent is what says a changeover is due, "
          "in either protocol",
          _breakin_due(Protocol.PACTOR3) and _breakin_due(Protocol.PACTOR1))
    tx = _Placed()
    tx.breakin_at_boundary = False
    tx.aim(p3, slot)
    tx.breakin_due = True
    tx.send_packet(3, b"\x1e" * 3, 0, breakin=True)
    check("--breakin-lead-72 explicitly overrides the measured PACTOR-3 gap",
          tx.at[0][1] == p3_end + lead_n)


def _breakin_due(protocol) -> bool:
    """`ptc.PtcHost.breakin_due` for a host holding `protocol` and taking the link."""
    from hfmodem.shrike.ptc import PtcHost
    host = PtcHost(peer=None, mycall="W9SSJ")
    host.protocol = protocol
    host.arq._breakin_pending = True
    return host.breakin_due and host.arq.taking_link


# --------------------------------------------------------------------------
# 3. the reversal over audio, between two shrike stations
# --------------------------------------------------------------------------
class _Recording(qso.AudioSide):
    """The QSO harness's transmit seam, keeping a copy of every burst rendered."""

    def __init__(self, name: str):
        super().__init__(name)
        self.log: list[tuple[str, np.ndarray]] = []

    def _tx(self, what: str, audio: np.ndarray) -> None:
        self.log.append((what, np.asarray(audio, np.float32)))
        super()._tx(what, audio)


class _Link(qso.AudioLink):
    SIDE = _Recording


def over_audio() -> None:
    print("\nA full reversal between two shrike stations, every burst as audio")
    link = _Link("W9SSJ", "K7ABC", verbose=False)
    link.a.arq.on_host_connect("W9SSJ", "K7ABC")
    link.exchange(2)
    if not (link.a.arq.state == State.CONNECTED
            and link.b.arq.state == State.CONNECTED):
        check("the link comes up before anything can be reversed", False)
        return
    link.a.arq.on_host_data(b"DE W9SSJ QSL")
    link.exchange(4)
    check("A holds the link as the ISS before the break-in",
          (link.a.arq.role, link.b.arq.role) == (ISS, IRS))

    # Exactly the bytes a changeover packet's field holds -- three, on a link that
    # is in PACTOR-3 -- so "the rest of the packet arrived" is a byte-for-byte
    # claim rather than a prefix.
    payload = b"BK "
    link.b.arq.on_host_data(payload)
    link.b.arq.on_host_breakin()
    b_log0, a_log0 = len(link.b_io.log), len(link.a_io.log)
    link.exchange(1)

    sent = link.b_io.log[b_log0:]
    check("B takes the link by sending a PACKET, not a control signal",
          len(sent) == 1 and "BREAK-IN" in sent[0][0],
          ", ".join(w for w, _ in sent) or "nothing")
    if sent:
        rf = onair._trim_silence(sent[0][1])
        # Against a speed-level-1 packet from the same renderer rather than a
        # nominal duration: same two carriers, same shaping skirts, so the
        # comparison is exact instead of a tolerance on a figure -- less the one
        # trailer symbol the entry-level packet keys and this one does not yet.
        # The reference's changeover keys one too (82 symbols at its head
        # anchor), but its trailer bits vary packet to packet where the entry's
        # are constant, so nothing is keyed there on a guess.
        slot = onair._trim_silence(np.asarray(qso.placement.link_packet(1, b"", 0),
                                              np.float32)).size - FS // 100
        check("...and its carrier is up for a PACKET's slot rather than a control "
              "signal's, so the cycle arithmetic either side of the reversal "
              "still closes",
              abs(rf.size - slot) <= FS // 1000,
              f"{rf.size / FS * 1e3:.1f} ms against {slot / FS * 1e3:.1f}")
    check("A yields on that packet, in the cycle it arrived in",
          (link.a.arq.role, link.b.arq.role) == (IRS, ISS),
          f"A={link.a.arq.role} B={link.b.arq.role}")
    check("A read the REST of the packet -- the field behind the head is the "
          "peer's data and it is delivered",
          payload in bytes(link.a.channel(link.a.ptchn).rx),
          repr(bytes(link.a.channel(link.a.ptchn).rx)))
    answered = link.a_io.log[a_log0:]
    check("A answers in the peer's next window, and answers with a control signal",
          len(answered) == 1 and answered[0][0].startswith("CS"),
          ", ".join(w for w, _ in answered) or "nothing")

    # A BREAK-IN NO LONGER COSTS THE UPGRADE AT ALL. It used to cost one cycle:
    # with no PACTOR-3 changeover packet to send, `ptc.PtcHost.breakin_now` dropped
    # the link to PACTOR-1 to seize the channel and the FSM climbed back on the
    # acknowledgement. `placement.CHANGEOVER` is that packet, read off both
    # directions of a real session, so the reversal happens where the link already
    # is and the traffic behind it resumes at the level it was running at.
    link.b.arq.on_host_data(b"AND MORE")
    b_log1 = len(link.b_io.log)
    link.exchange(1)
    sent_after = [w for w, _ in link.b_io.log[b_log1:]]
    check("the link never left PACTOR-3 to change hands",
          link.b.protocol is Protocol.PACTOR3
          and any(w.startswith("SL") for w in sent_after),
          f"{link.b.protocol}, sent {', '.join(sent_after) or 'nothing'}")

    # ...and the 0x55 header rule is PACTOR-1's, so it is exercised on a link that
    # stays there. Ruled out through the real mechanism rather than by patching a
    # module constant: a peer that answers a PACTOR-3 packet in PACTOR-1 has told
    # us it cannot follow, and this is a station that has already been told.
    p1 = _Link("W9SSJ", "K7ABC", verbose=False)
    p1.a._ruled_out.add(Protocol.PACTOR3)
    p1.b._ruled_out.add(Protocol.PACTOR3)
    p1.a.arq.on_host_connect("W9SSJ", "K7ABC")
    p1.exchange(2)
    p1.a.arq.on_host_data(b"DE W9SSJ QSL")
    p1.exchange(4)
    p1.b.arq.on_host_data(b"BK DE K")
    p1.b.arq.on_host_breakin()
    p1.exchange(1)
    check("a link with PACTOR-3 ruled out stays in PACTOR-1",
          p1.b.protocol is Protocol.PACTOR1, str(p1.b.protocol))
    # The packet after a changeover packet carries header 0x55 -- the one place
    # the header does not follow our own counter's parity.
    p1.b.arq.on_host_data(b"AND MORE")
    b_log1 = len(p1.b_io.log)
    p1.exchange(1)
    after = [p1rx.decode_p1_packets(a) for w, a in p1.b_io.log[b_log1:]
             if w.startswith("P1 pkt")]
    heads = [p[0].header for p in after if p]
    check("the packet after a break-in carries header 0x55",
          heads[:1] == [pactor1.SYNC_HEADER],
          ", ".join(f"{h:#04x}" for h in heads) or "no data packet followed")


# --------------------------------------------------------------------------
# 4. the budget -- measured, on the path the state machine actually takes
# --------------------------------------------------------------------------
def the_budget() -> None:
    print("\nRecognising the reversal inside 170 - d - settle")
    peer = pactor1.breakin_signal(b"BK DE K7ABC", 100, lead_s=0.02, tail_s=0.02)

    class _Peer:
        def __getattr__(self, k):
            return lambda *a, **kw: None

    def one(seg: np.ndarray) -> tuple[float, str]:
        host = qso.PtcHost(peer=_Peer(), mycall="W9SSJ")
        host.arq.on_host_connect("W9SSJ", "K7ABC")
        host.arq._to(State.CONNECTED)
        host.arq.role = ISS
        host.arq._expected_seq = 0
        t = time.perf_counter()
        ev = rxfront.decode_expected_p1_packet(seg)
        if ev is not None:
            host.on_rx_event(ev)
        return (time.perf_counter() - t) * 1e3, host.arq.role

    # The window the yielding station holds: its own carrier has just dropped and
    # the rest of the cycle is the peer's. One trial to warm the code paths, then
    # the best of five -- a scheduler tail is not a property of the decoder.
    seg = np.concatenate([np.zeros(round(0.09 * FS), np.float32), peer])
    one(seg)
    trials = [one(seg) for _ in range(5)]
    cost = min(t for t, _ in trials)
    check("the changeover packet drives the yield, not a control signal",
          all(role == IRS for _, role in trials))
    for d_ms in (40, 90):
        limit = budget_n(round(d_ms / 1000 * FS)) / FS * 1e3
        check(f"d = {d_ms:3d} ms: the decision costs {cost:.1f} ms of a "
              f"{limit:.0f} ms budget", cost <= limit)

    # The sweeping path is BLIND to a break-in, which is why the in-session entry
    # point is the only door: the control-signal detector is looking for a run of
    # 60-260 ms and the head is the first eighth of a 960 ms one, and the P1 data
    # scan behind it is gated on a connect decoded in the same buffer.
    kinds = {ev.kind for ev in rxfront.decode_events(seg)}
    check("NEGATIVE CONTROL: the band-sweeping decoder reports no control signal "
          "and no frame in a changeover packet",
          not kinds & {"cs", "packet"}, str(sorted(kinds)))
    check("NEGATIVE CONTROL: ...because the control-signal detector finds no "
          "burst of a control signal's length in one",
          not rxfront._p1_cs_bursts(peer),
          str(rxfront._p1_cs_bursts(peer)))


# --------------------------------------------------------------------------
# 5. the live session loop -- where the reversal has to actually happen
# --------------------------------------------------------------------------
#
# Sections 1-4 prove the frame, the arithmetic and the state machine, and all
# three passed while a real session could not reverse at all: nothing delivered
# the changeover packet to any of them. A sending station's listening window is
# `cycle - packet - settle - block`, 165 ms, and the packet it has to recognise
# spans 1050 ms from its own carrier dropping. No window holds one, so the scan
# that needs all 960 ms is never given a whole frame, and the loop transmits over
# the packet it was being told to listen to -- every cycle, deterministically.
#
# What fits is the HEAD, and only if it is read where the grid says it is. So the
# loop is run here end to end, over a recording, through `shrike.onair`'s own
# argument parser: a scripted peer answers on the caller's grid for four cycles
# and then takes the link. The negative control is the same run with the head
# read switched off, which is the loop as it was.
D_MS = 90.0
BREAKIN_SLOT = 5
BK_PAYLOAD = b"BK DE K"                  # exactly the 7 bytes the field holds
BK_AT = BREAKIN_SLOT * SLOT_N + PACKET_N + round(D_MS / 1000 * FS)


def _peer_wav(path: Path, slots: int = 10) -> None:
    """A scripted PACTOR-1 peer, as a recording on the caller's own grid.

    The caller anchors its grid on its own connect and free-runs from there, so
    every instant the peer needs is known in advance: it answers `d` after each of
    our packets would end. CS4 first, which is a legal answer to a call and brings
    the link up at 100 Bd; then the acknowledgement alternation, because a repeat
    of one codeword means REQUEST and would stall the exchange; then, in one slot,
    the changeover packet instead. The shift inverts on every transmission, as a
    real station's does.
    """
    out = np.zeros((slots + 2) * SLOT_N, np.float32)
    for k in range(slots):
        if k == 0:
            burst = pactor1.control_signal(pactor1.CS_SPEED, invert=k % 2)
        elif k == BREAKIN_SLOT:
            burst = pactor1.breakin_signal(BK_PAYLOAD, 100, invert=k % 2,
                                           lead_s=0, tail_s=0)
        else:
            burst = pactor1.control_signal(
                pactor1.CS_ACK_A if k % 2 else pactor1.CS_ACK_B, invert=k % 2)
        rf = onair._trim_silence(np.asarray(burst, np.float32))
        at = k * SLOT_N + PACKET_N + round(D_MS / 1000 * FS)
        out[at:at + rf.size] += rf
    onair.session.write_wav(str(path), out)


def _run_loop(wav: Path, outdir: Path, *, head_read: bool = True,
              hold: int = BREAKIN_SLOT + 3, over: bool = False,
              cycles: int = 3, retries: int = 0) -> dict:
    """One whole `shrike.onair` session over `wav`, with the loop instrumented.

    Instrumented rather than read back out of the log: the claims below are about
    where the grid moved and what instant the acknowledgement was aimed at, and
    both are numbers the session holds rather than sentences it prints.
    """
    keyed: list[tuple[str, int | None]] = []
    keyed_roles = []
    made: list = []

    class _Tx(onair.RadioTx):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

        def _tx(self, audio, what, drive=None, lead_n=0, *, strict_guard=False,
                pulse_offsets=None):
            keyed.append((what, self.boundary))
            keyed_roles.append((what, self.host.arq.role))
            super()._tx(audio, what, drive=drive, lead_n=lead_n,
                        strict_guard=strict_guard, pulse_offsets=pulse_offsets)

    class _Grid(onair._MasterGrid):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.rotations: list[tuple[bool, int]] = []
            made.append(self)

        def reverse(self, *, to_iss):
            was = self.anchor
            line = super().reverse(to_iss=to_iss)
            self.rotations.append((to_iss, self.anchor - was))
            return line

    saved = (onair.RadioTx, onair._MasterGrid, onair._SessionRx.control_signal,
             onair._save_capture_async)
    onair.RadioTx, onair._MasterGrid = _Tx, _Grid
    # Captures go through a module-global daemon thread that outlives `main()`,
    # and `outdir` here lives in a TemporaryDirectory: under load the drain
    # falls behind and writes into the tree while the context manager is
    # deleting it (ENOTEMPTY). A replay is not real time, so write them here.
    onair._save_capture_async = onair._save_capture
    if not head_read:
        onair._SessionRx.control_signal = lambda self, *a, **kw: None
    argv, log = sys.argv, io.StringIO()
    sys.argv = ["shrike.onair", "--replay", str(wav), "--hold", str(hold),
                "--max-cycles", str(cycles), "--retries", str(retries),
                "--mycall", "W9SSJ", "--dxcall", "K7ABC",
                "--dial", "7100000", "--outdir", str(outdir)]
    if over:
        sys.argv.append("--over")
    try:
        with contextlib.redirect_stdout(log):
            onair.main()
    finally:
        (onair.RadioTx, onair._MasterGrid, onair._SessionRx.control_signal,
         onair._save_capture_async) = saved
        sys.argv = argv
    tx, grid = made[0], made[1]
    return {"host": tx.host, "grid": grid, "keyed": keyed, "keyed_roles": keyed_roles, "log": log.getvalue()}


def the_short_window() -> None:
    """What can decode in the window a one-slot keyed cycle actually leaves.

    The reversal is read at the anchor because nothing else in the receiver can
    reach it in time -- and the same is true of the ORDINARY acknowledgement,
    which is why the read is not a break-in special case. A sending station's
    window is `cycle - packet - settle` and the bridge's block of it is held back:
    250 ms, under every sweeping decoder's floor, and those floors are right.
    """
    print("\nThe 250 ms a one-slot keyed cycle leaves the sending station")
    d_n = round(D_MS / 1000 * FS)
    seg = np.zeros(SLOT_N - PACKET_N - SETTLE_N, np.float32)
    rf = onair._trim_silence(np.asarray(
        pactor1.control_signal(pactor1.CS_ACK_A), np.float32))
    seg[d_n:d_n + rf.size] += rf

    class _Peer:
        def __getattr__(self, k):
            return lambda *a, **kw: None

    host = qso.PtcHost(peer=_Peer(), mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "K7ABC")
    host.arq._to(State.CONNECTED)
    host.arq.role = ISS
    host.arq.on_host_data(b"DE W9SSJ")
    host.arq.on_cycle()                       # ...so there is a packet to acknowledge
    rx = onair._SessionRx(host)
    rx.new_cycle()
    rx.feed(seg)
    rx.flush()
    rx.deep_scan(seg)
    check("NEGATIVE CONTROL: the rolling decoder, the flush and the frame scan "
          "all decline a quarter second -- correctly, and between them that is "
          "every path the loop had", rx.count == 0, f"{rx.count} events")
    check("the read at the anchor delivers the peer's acknowledgement",
          rx.control_signal(seg, 0, d_n) == pactor1.CS_ACK_A)
    check("...and the state machine acts on it: the packet in flight is cleared",
          host.arq._inflight is None)
    check("...ONCE. A second path offering the same burst in the same cycle is "
          "refused", rx.control_signal(seg, 0, d_n) is None)
    rx.new_cycle()
    check("nothing is claimed where the peer said nothing",
          rx.control_signal(np.zeros_like(seg), 0, d_n) is None)


SILENT = archive.SILENT_GATEWAYS


def the_anchor_on_real_audio() -> None:
    """The read at the anchor, against stations that are not us.

    Everything about the changeover packet is measured against our own encoder,
    because no station has ever broken in on shrike and the corpus holds no
    changeover packet to check it with. What CAN be checked off-air is the thing
    the read changes for every cycle of every session: aiming the decoder from the
    grid instead of from the burst detector. Two recordings of a gateway calling
    with our transmitter off supply the bursts and, between them, the negatives.
    """
    print("\nThe read at the anchor, on two recordings of real stations")
    rng = np.random.default_rng(3)
    for path in SILENT:
        if not path.exists():
            check(f"{path.name}: recording present", False, str(path))
            continue
        a = onair.session.load_wav(str(path), FS)
        ons = rxfront.p1_burst_onsets(a)
        # The detector's own placement is what the loop uses today; the grid's
        # anchor is what it will use in a window too short for the detector. Both
        # are read here at the SAME instants, so the comparison is of the aiming.
        hit = sum(1 for t in ons
                  if (p1rx.decode_control_signal(a, t, spec.P1_CS_S) or (0, 9))[1] == 0)
        quiet: list[float] = []
        while len(quiet) < 150:
            t = float(rng.uniform(0.3, a.size / FS - 0.3))
            if all(abs(t - o) > 0.4 for o in ons):
                quiet.append(t)
        false_cs = sum(1 for t in quiet
                       if (p1rx.decode_control_signal(a, t, spec.P1_CS_S)
                           or (0, 9))[1] == 0)
        false_head = sum(1 for t in quiet
                         if (got := p1rx.cs_head(a, t)) is not None
                         and got.index == pactor1.CS_CHANGEOVER)
        check(f"{path.parent.name}: the anchor read decodes a third or more of "
              f"{len(ons)} real bursts at zero errors", hit >= len(ons) // 3,
              f"{hit} of {len(ons)}")
        check(f"{path.parent.name}: ...and manufactures at most one control "
              f"signal in {len(quiet)} quiet instants of the same recording",
              false_cs <= 1, f"{false_cs}")
        check(f"{path.parent.name}: ...and no break-in at all, which is the one "
              f"that would hand away the channel", false_head == 0,
              f"{false_head}")


BK_2210 = b"RMS Tri"        # what K4MSU's changeover packet carried
BK_2210_AT = 64.855         # ...and where its CS3 head sits on that stream
BK_2210_EDGE = 65.237       # ...and where the loop's window boundary cut it


def the_break_in_the_anchor_could_not_be_aimed_at() -> None:
    """A changeover the SWEEP read, through the whole session loop.

    Every other scene here has the head arriving where the grid says an answer is
    due, which is where the anchored read can be pointed. A station taking the
    channel is under no such obligation: K4MSU's head landed 237 ms from the
    predicted instant and 145 ms before our own packet was due to end, so both
    positioners were aimed at silence and the sweeping decoder is what read the
    codeword -- out of audio it had kept, a cycle after the window holding it had
    closed.

    The role reversed on that codeword all the same, and the field behind it was
    thrown away: the loop asked `control_signal` whether a changeover had
    happened, and that read declines the moment another path has already given
    the state machine a codeword. So the one recording in which a gateway ever
    handed this station a readable field ended with the field on disk and nothing
    on the host.

    The recording settles both halves, which is why the claim is made here and
    not against our own encoder: the packet is there, it is readable, and the
    session over it delivered nothing.
    """
    print("\nA gateway's changeover packet, off our grid, over the whole session")
    a = onair.session.load_wav(str(archive.K4MSU_BREAKIN), FS)
    head = round(BK_2210_AT * FS)
    got = rxfront.decode_expected_p1_packet(a[head:head + 2 * SLOT_N])
    check("the head at 64.855 s carries a field, and it is the gateway's name",
          got is not None and got.packet[2] == BK_2210,
          str(got.text)[:60] if got is not None else "nothing decoded")
    edge = round(BK_2210_EDGE * FS)
    check("NEGATIVE CONTROL: neither window the loop cut it into holds a frame, "
          "so nothing below can be reading it out of one",
          rxfront.decode_expected_p1_packet(a[edge - SLOT_N:edge]) is None
          and rxfront.decode_expected_p1_packet(a[edge:edge + SLOT_N]) is None)

    with tempfile.TemporaryDirectory() as tmp:
        # The connect took until cycle 36 of this recording and the hold runs to
        # the end of the audio; a session that stops short of either never
        # reaches the changeover at all.
        run = _run_loop(Path(archive.K4MSU_BREAKIN), Path(tmp) / "out",
                        hold=12, cycles=60, retries=60)

    log, host = run["log"], run["host"]
    check("the loop reads the codeword and yields the channel",
          "CS3/break-in" in log and run["grid"].rotations[:1] == [(False, ROT_N)],
          str(run["grid"].rotations))
    # THE ONE THAT USED TO FAIL: the rotation happened and the packet did not.
    check("...and delivers the field behind it to the host",
          BK_2210 in bytes(host.channel(host.ptchn).rx),
          repr(bytes(host.channel(host.ptchn).rx)[-24:]))


QSL_PAYLOAD = b"QSL 599"                 # the 7 bytes an 8-byte field carries


def _new_iss_wav(path: Path, slots: int = 10) -> None:
    """The peer AFTER it has taken the link: what WS8EOC actually sent.

    Same script as `_peer_wav` through the break-in, and then the part the
    2026-08-03 sessions received and could not read: the changeover packet again
    -- a peer that missed our acknowledgement repeats it, byte for byte -- and
    then its first data packets. Each sits where the new ISS's grid puts it, one
    slot after the last, ending about `d` before OUR rotated boundary.
    """
    out = np.zeros((slots + 2) * SLOT_N, np.float32)
    for k in range(slots):
        if k == 0:
            burst = pactor1.control_signal(pactor1.CS_SPEED, invert=k % 2)
        elif k < BREAKIN_SLOT:
            burst = pactor1.control_signal(
                pactor1.CS_ACK_A if k % 2 else pactor1.CS_ACK_B, invert=k % 2)
        elif k <= BREAKIN_SLOT + 1:          # the break-in, and its repeat
            burst = pactor1.breakin_signal(BK_PAYLOAD, 100, invert=k % 2,
                                           lead_s=0, tail_s=0)
        elif k == BREAKIN_SLOT + 2:
            burst = pactor1.packet_signal(QSL_PAYLOAD, 100, packet_count=1,
                                          header=pactor1.SYNC_HEADER,
                                          invert=k % 2, lead_s=0, tail_s=0)
        else:
            burst = pactor1.packet_signal(b"", 100, packet_count=2,
                                          invert=k % 2, lead_s=0, tail_s=0)
        rf = onair._trim_silence(np.asarray(burst, np.float32))
        at = k * SLOT_N + PACKET_N + round(D_MS / 1000 * FS)
        out[at:at + rf.size] += rf
    onair.session.write_wav(str(path), out)


def the_receiving_station() -> None:
    """The half of the reversal the 2026-08-03 sessions never completed.

    The witnessed failure (captures/onair-0803-225309-revwitness): WS8EOC took
    the link, sent its changeover packet bit-perfect in three consecutive
    receive windows, and the live session decoded none of them -- while the
    SAVED windows decode CRC-valid. The involution keeps each side's own
    turnaround, so the peer's packet ends about `d` before our boundary, which
    is the very sample the mid-cycle scan buffer stops at: the frame's last
    bits were never in the audio it was given. The gateway was asked for a
    repeat it had already sent, 34 times, and then signed off in CW.
    """
    print("\nThe receiving station reads the packets it is acknowledging")

    # The geometry first, on the decoder alone: a buffer that ends inside the
    # frame's last bit cannot decode it, whole cycles of signal notwithstanding.
    frame = onair._trim_silence(np.asarray(
        pactor1.breakin_signal(BK_PAYLOAD, 100, lead_s=0, tail_s=0), np.float32))
    window = np.concatenate([np.zeros(round(0.2 * FS), np.float32), frame,
                             np.zeros(round(0.057 * FS), np.float32)])
    bit_n = FS // 100
    cut = window[:round(0.2 * FS) + frame.size - bit_n]
    check("NEGATIVE CONTROL: the frame does not decode from a buffer cut at "
          "its final bit, which is what `boundary - d` hands the scan",
          rxfront.decode_expected_p1_packet(cut) is None)
    check("...and decodes whole from the same audio with the tail in hand",
          rxfront.decode_expected_p1_packet(window) is not None)

    # A file has no capture blocks, so a bare replay ends its mid-cycle buffer a
    # holdback later than the live loop does -- which is exactly the margin the
    # failure lived in. One live-sized block restores the geometry: without it
    # this scenario decodes mid-cycle by accident and guards nothing.
    saved_init = onair._ReplayInput.__init__

    def _live_blocks(self, *a, **kw):
        saved_init(self, *a, **kw)
        self.holdback = round(0.021 * FS)
    onair._ReplayInput.__init__ = _live_blocks
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "new_iss.wav"
            _new_iss_wav(wav)
            got = _run_loop(wav, Path(tmp) / "live")
    finally:
        onair._ReplayInput.__init__ = saved_init

    grid, host = got["grid"], got["host"]
    rx = bytes(host.channel(host.ptchn).rx)
    check("the loop yields once, on the first changeover packet",
          grid.rotations[:1] == [(False, ROT_N)], str(grid.rotations))
    check("the REPEATED changeover packet reaches the state machine and is "
          "delivered exactly once", rx.count(BK_PAYLOAD) == 1, repr(rx[-24:]))
    check("the peer's first data packet is read and delivered -- the exchange "
          "the 2026-08-03 reversal wedged on", QSL_PAYLOAD in rx, repr(rx[-24:]))
    acks = [w for w, _ in got["keyed"] if w.startswith("P1 CS")]
    check("...and every packet is answered with a control signal, one a cycle",
          len(acks) >= 3, str([w for w, _ in got["keyed"]]))


def the_live_loop() -> None:
    print("\nThe reversal through the live session loop, over a recording")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "peer.wav"
        _peer_wav(wav)
        deaf = _run_loop(wav, Path(tmp) / "deaf", head_read=False)
        got = _run_loop(wav, Path(tmp) / "live")

    # THE REGRESSION. Everything below passed as arithmetic while this was the
    # behaviour of the program that flies.
    check("NEGATIVE CONTROL: without the head read the loop never sees the "
          "changeover packet at all -- it straddles two windows and neither "
          "holds a frame", not deaf["grid"].rotations
          and BK_PAYLOAD not in bytes(deaf["host"].channel(deaf["host"].ptchn).rx),
          f"rotations {deaf['grid'].rotations}, role {deaf['host'].arq.role}")
    check("...and the station goes on believing it is the sender: its QRT goes "
          "out as an ordinary packet, not a break-in",
          any(w.startswith("P1 pkt") and "QRT" in w for w, _ in deaf["keyed"])
          and not any(w.startswith("P1 BREAK-IN") for w, _ in deaf["keyed"]),
          str([w for w, _ in deaf["keyed"]]))

    grid, host = got["grid"], got["host"]
    check("the loop yields on the CS3 head, once, and moves its TRANSMIT anchor "
          "by exactly 840 ms", grid.rotations[:1] == [(False, ROT_N)],
          str(grid.rotations))
    check("...and any later rotation is the QRT teardown taking the link back, "
          "which moves nothing",
          all(r == (True, 0) for r in grid.rotations[1:]), str(grid.rotations))
    # SAYING SO IS THE POINT, and which role says it depends on the peer. This
    # fixture's does not go quiet: it answers with a bare codeword every cycle
    # after its changeover packet, which is both ends holding the receiving
    # role. `arq.RECLAIM_CODEWORDS` ends that by taking the link back, so the
    # goodbye rides an ordinary packet from the sending role rather than a
    # break-in. The IRS break-in stays the path against a peer that is silent,
    # where there is nothing to reclaim from -- test_signoff pins it.
    check("the hold budget's end is a transmitted goodbye: the station takes "
          "the link back off a peer answering as a receiver, and says it",
          "take the link back" in got["log"]
          and any(w.startswith("P1 pkt") and "QRT" in w
                  for w, _ in got["keyed"]),
          str([w for w, _ in got["keyed"]]))
    check("the 840 ms behind the head is delivered to the host, byte-exact",
          BK_PAYLOAD in bytes(host.channel(host.ptchn).rx),
          repr(bytes(host.channel(host.ptchn).rx)[-16:]))
    check("...in the cycle the packet arrived in, not the one after",
          "GRID REVERSED -> IRS" in got["log"]
          and got["log"].index("GRID REVERSED -> IRS")
          < got["log"].index("P1 BREAK-IN"))

    ack = next((b for w, b in got["keyed"] if w.startswith("P1 CS")), None)
    check("the acknowledgement is a control signal, and there is one",
          ack is not None, str([w for w, _ in got["keyed"]]))
    if ack is not None:
        # The placement, measured on the loop rather than solved, and held to
        # the invariant both `--ack-aim` forms share rather than to either
        # one's arithmetic: the ack hangs off the peer's MEASURED burst, past
        # the guard, and its whole 120 ms fits in the air the peer's packet
        # leaves. Which aim inside that band flies is `ACK_FORMS`' business
        # and `test_ackplace`'s.
        gap = ack - (BK_AT + PACKET_N)
        check(f"...and it keys {gap / FS * 1e3:.0f} ms after the peer's packet "
              f"ends -- off the measured burst, clear of it, and whole "
              f"inside the {(SLOT_N - PACKET_N) / FS * 1e3:.0f} ms the packet "
              f"leaves",
              0 <= gap <= SLOT_N - PACKET_N - CS_N,
              f"{gap / FS * 1e3:.1f} ms")
        rots = [float(m.group(1)) for m in re.finditer(
            r"\[ack\] key-dataend\s*[+-][\d.]+ ms \| rot\s*([+-][\d.]+) ms",
            got["log"])]
        check("...and every keyed ack lands inside the instant the peer reads "
              "at, on the instrument the operator reads",
              bool(rots) and all(abs(r) <= p1rx.CS_ANCHOR_S * 1e3
                                 for r in rots),
              f"{len(rots)} line(s), rot {rots}")
        check("NEGATIVE CONTROL: un-rotated, that same key lands inside the "
              "packet it was told to listen to",
              BK_AT < ack - ROT_N < BK_AT + PACKET_N,
              f"{(ack - ROT_N - BK_AT) / FS * 1e3:.0f} ms into it")

    # The cost, in the slot it is spent in. The read is what buys the whole
    # reversal and it happens between the peer's head ending and our PTT.
    seg = np.concatenate([np.zeros(round(0.09 * FS), np.float32),
                          pactor1.breakin_signal(BK_PAYLOAD, 100,
                                                 lead_s=0, tail_s=0)])
    p1rx.cs_head(seg, 0.09)
    t = time.perf_counter()
    for _ in range(20):
        p1rx.cs_head(seg, 0.09)
    cost = (time.perf_counter() - t) / 20 * 1e3
    for d_ms in (40, 90):
        limit = budget_n(round(d_ms / 1000 * FS)) / FS * 1e3
        check(f"d = {d_ms:3d} ms: the head read costs {cost:.2f} ms of a "
              f"{limit:.0f} ms budget", cost <= limit)


# --------------------------------------------------------------------------
# 6. the changeover the peer never takes
# --------------------------------------------------------------------------
ASK_HOLD = ArqConfig().max_retries + 5
"""Held cycles: past the retry budget, with room to see what happens after it.

Read off the config rather than written down, because the number this scene is
about IS the budget -- a hold that stops short of it cannot tell the fix from the
fault."""


def _asking_peer_wav(path: Path, slots: int) -> None:
    """A station that answers every cycle and never takes the channel.

    CS4 answers the call and brings the link up at 100 Bd. CS1 answers the packet
    that carries the callsign, which is one CHANGE and the only acknowledgement
    this station ever gives; from there it is ONE codeword, CS2, repeated for the
    rest of the session, which PACTOR-1 spells REQUEST (`ptc._logical_cs`) --
    "send that again". No packet, ever, and no CS3: this station is the IRS and
    means to stay it.

    THE ACK IS ITS OWN CYCLE and was not always. This peer answers 5 ms inside the
    nominal turnaround, which the point read missed and the sweeping decoder then
    delivered a cycle late -- so the connect answer's own CS1 was credited to the
    first data packet, and a scene whose peer never acknowledges anything drained
    its buffer on a codeword the peer had transmitted before that packet existed.
    Widening the anchored read to `p1rx.CS_SEARCH_HALF_S` puts each answer in the
    cycle it arrived in, so the acknowledgement this scene needs has to be one the
    peer actually sends.

    Which is what WS8EOC did on 2026-08-09 to a caller that had asked for the
    changeover: 120-135 ms control signals, one a cycle, at zero errors, and never
    a 960 ms burst. The same shape the module header of `shrike.arq` records for
    2026-08-03.
    """
    out = np.zeros((slots + 2) * SLOT_N, np.float32)
    for k in range(slots):
        word = (pactor1.CS_SPEED if k == 0 else
                pactor1.CS_ACK_A if k == 1 else pactor1.CS_ACK_B)
        burst = pactor1.control_signal(word, invert=k % 2)
        rf = onair._trim_silence(np.asarray(burst, np.float32))
        at = k * SLOT_N + PACKET_N + round(D_MS / 1000 * FS)
        out[at:at + rf.size] += rf
    onair.session.write_wav(str(path), out)


def the_changeover_the_peer_never_takes() -> None:
    """A caller with nothing left to send HOLDS the link. It cannot yield alone.

    The break-in is the receiving station's move and the only changeover there
    is, so an ISS that wants to hand over sets bit 6 and waits -- for as long as
    the peer takes. The retry budget used to end that wait: the peer's repeated
    codeword is a request, nine of them spent the budget on a packet with an
    EMPTY FIELD, and the yield-and-listen recovery then flipped this end to IRS
    with nothing on the air to say so. The peer had never asked for the channel,
    so both ends held the receiving role and the session died with the caller
    emitting break-in packets at a station that was waiting for a data packet --
    measured against WS8EOC on 2026-08-09, twice.

    The budget is for a peer that has gone deaf, and a codeword decoded at zero
    errors every single cycle is the refutation of that. Asked off the air: what
    got keyed, and in which direction the link was running when the hold ended.
    """
    print("\nA caller that asked for the changeover, against a peer that waits")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "peer.wav"
        _asking_peer_wav(wav, slots=ASK_HOLD + onair.QRT_CYCLES + 3)
        got = _run_loop(wav, Path(tmp) / "out", hold=ASK_HOLD, over=True)

    host, grid = got["host"], got["grid"]
    keyed = [w for w, _ in got["keyed"]]
    pkts = [w for w in keyed if "pkt" in w or "BREAK-IN" in w]
    check("the link comes up and the caller is still the SENDING station when "
          "the hold ends",
          all(role == ISS for what, role in got["keyed_roles"]
              if what.startswith("P1 pkt"))
          and host.arq.state == State.DISCONNECTED
          and "goodbye went unanswered" in got["log"],
          f"packet roles {got['keyed_roles']}, final state {host.arq.state}")
    # THE ONE THAT USED TO FAIL. A yield rotates the grid, so an end that handed
    # the link over on its own leaves the rotation behind as evidence.
    check("...and the link never changed direction: nothing rotated the grid",
          not grid.rotations, str(grid.rotations))
    check("...over more held cycles than the retry budget, so the budget was "
          "given every chance to spend itself",
          len(pkts) > ArqConfig().max_retries,
          f"{len(pkts)} packets keyed, budget {ArqConfig().max_retries}")
    # The request STANDS on every packet: the peer holds the invitation up until
    # it decides to take it, and an invitation withdrawn after one packet is an
    # invitation the peer can miss.
    asking = [w for w in pkts if w.startswith("P1 pkt") and "QRT" not in w]
    check("...with the changeover request standing on every one of them, not "
          "only the first -- the invitation is held up until the peer takes it",
          bool(asking) and all(" BK" in w for w in asking), str(pkts))
    check("the caller never takes the channel it asked the peer to take",
          not any("BREAK-IN" in w for w in keyed), str(keyed))
    check("...and the goodbye rides an ordinary data packet, which is what an "
          "ISS has to say it with",
          any(w.startswith("P1 pkt") and "QRT" in w for w in keyed), str(pkts))


# --------------------------------------------------------------------------
# 7. the changeover packet that will not decode
# --------------------------------------------------------------------------
DEAF_HOLD = BREAKIN_SLOT + ArqConfig().max_retries + 4
"""Held cycles: past the IRS's budget with the peer transmitting throughout.

Read off the config for `ASK_HOLD`'s reason -- the number this scene is about IS
the budget, so a hold that stops short of it cannot tell the fix from the fault."""


def _damaged_breakin(k: int) -> np.ndarray:
    """A changeover packet whose head survives the channel and whose body does not.

    The body played backwards: the CS3 codeword is left as rendered and the
    840 ms behind it runs the other way, so the burst is one continuous 960 ms run
    of the same two tones -- what the envelope detector and the head reader both
    see -- carrying bits that cannot pass a CRC. That is the shape of every cycle
    WS8EOC's 200 Bd changeover packets arrived in on 2026-08-09: heads at zero bit
    errors, no frame anywhere in the saved window.
    """
    rf = onair._trim_silence(np.asarray(
        pactor1.breakin_signal(BK_PAYLOAD, 100, invert=k % 2,
                               lead_s=0, tail_s=0), np.float32))
    return np.concatenate([rf[:CS_N], rf[CS_N:][::-1]])


def _deaf_iss_wav(path: Path, slots: int) -> None:
    """The peer takes the link, and then every packet it sends is unreadable."""
    out = np.zeros((slots + 2) * SLOT_N, np.float32)
    for k in range(slots):
        if k == 0:
            rf = onair._trim_silence(np.asarray(
                pactor1.control_signal(pactor1.CS_SPEED, invert=k % 2), np.float32))
        elif k < BREAKIN_SLOT:
            rf = onair._trim_silence(np.asarray(pactor1.control_signal(
                pactor1.CS_ACK_A if k % 2 else pactor1.CS_ACK_B,
                invert=k % 2), np.float32))
        elif k == BREAKIN_SLOT:
            rf = onair._trim_silence(np.asarray(
                pactor1.breakin_signal(BK_PAYLOAD, 100, invert=k % 2,
                                       lead_s=0, tail_s=0), np.float32))
        else:
            rf = _damaged_breakin(k)
        at = k * SLOT_N + PACKET_N + round(D_MS / 1000 * FS)
        out[at:at + rf.size] += rf
    onair.session.write_wav(str(path), out)


def the_head_of_a_packet_we_could_not_read() -> None:
    """A CS3 decoded while we are ALREADY the IRS, and what it is evidence of.

    It is not a request for a channel we have already given up -- the peer holding
    the link cannot ask for it -- and it is not part of the acknowledge
    alternation, so the repeat rule that turns a second CS1 into "send that again"
    must not reach it. It is the first 120 ms of the peer's changeover packet,
    repeated because our acknowledgement never arrived, and it says the one thing
    the receiving station's link-dead budget has to know: the peer is transmitting
    a packet every cycle and we are failing to read it.

    MEASURED, WS8EOC 2026-08-09 (captures/onair-0809-2111, -2118 and -2108). Each
    session yielded correctly, read one packet, and then died on "no decodable
    traffic from the peer -> abort" while the gateway's heads went on decoding at
    zero bit errors -- `_silent_cycles` spent on a station the operator could hear.
    The budget is `_on_nak`'s rule read from the receiving side: it counts SILENCE.
    """
    print("\nThe peer's changeover packet, repeated, with a body we cannot read")
    lead = np.zeros(round(0.09 * FS), np.float32)
    damaged = np.concatenate([lead, _damaged_breakin(1)])
    check("NEGATIVE CONTROL: the damaged packet decodes as neither a data frame "
          "nor a changeover one -- so nothing below can be passing on the body",
          rxfront.decode_expected_p1_packet(damaged) is None)
    check("...and its CS3 head still reads at zero bit errors, which is the "
          "whole of what the cycle has to go on",
          p1rx.cs_head(damaged, 0.09)[:2] == (pactor1.CS_CHANGEOVER, 0),
          str(p1rx.cs_head(damaged, 0.09)))

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "deaf_iss.wav"
        _deaf_iss_wav(wav, slots=DEAF_HOLD + onair.QRT_CYCLES + 3)
        got = _run_loop(wav, Path(tmp) / "out", hold=DEAF_HOLD)

    host, grid, log = got["host"], got["grid"], got["log"]
    keyed = [w for w, _ in got["keyed"]]
    unread = DEAF_HOLD - BREAKIN_SLOT
    check("the loop yields on the first changeover packet, once",
          grid.rotations[:1] == [(False, ROT_N)], str(grid.rotations))
    check("...and the repeated heads reach the state machine as break-ins, not "
          "as repeat requests", "rx CS BRK" in log and log.count("rx CS BRK") > 1,
          f"BRK {log.count('rx CS BRK')}, REQ {log.count('rx CS REQ')}")
    # THE ONE THAT USED TO FAIL.
    check(f"...and the link is still up after {unread} cycles of unreadable "
          f"packets, more than the {ArqConfig().max_retries}-cycle budget",
          "no decodable traffic from the peer" not in log
          and "LINK DOWN" not in log,
          f"state {host.arq.state}, role {host.arq.role}")
    check("...and it stays the RECEIVING one for all of them: it answers, it "
          "does not take the channel back. The only break-in it keys is the "
          "goodbye the hold's end owes, which an IRS can only send as a packet",
          not any("BREAK-IN" in w and "QRT" not in w for w in keyed), str(keyed))
    acks = [w for w in keyed if w.startswith("P1 CS")]
    check("...and it owes the peer one control signal a cycle, and pays it",
          len(acks) >= unread, f"{len(acks)} against {unread} cycles")

    # The instrument the operator's report was read off. Every hold cycle here
    # has the peer transmitting in it, so every acknowledgement has a burst to be
    # measured against; the fallback below it counts from a stamp the hold loop
    # never set, and reported the age of the connect phase instead.
    stale = [ln for ln in log.splitlines() if "burst time unknown" in ln]
    against = [ln for ln in log.splitlines() if "after the tone pair came up" in ln]
    check("the acknowledgement's latency is measured against the burst that "
          "stopped the listen, not against a stamp from the connect phase",
          not stale and len(against) >= unread,
          f"{len(against)} measured, {len(stale)} unmeasurable")


def the_greeting_behind_the_answer() -> None:
    """The counter this end grades against has to reset on EVERY path that hands
    the link over, and the connect answered by a break-in is a path.

    A Winlink RMS hears the call, takes the channel and sends its greeting, so the
    first packet of the session arrives under the peer's own reset counter. This
    end had just come up counting from one, and one short of one is zero: the
    greeting read as a repeat of a packet nobody had sent, and was acknowledged
    with its payload thrown away. Two of the four yields carried the reset beside
    the call and this one carried none, which is why it lives in the yield now.
    """
    print("\nA gateway that answers the call by taking the channel")

    class _Peer:
        def __getattr__(self, k):
            return lambda *a, **kw: None

    host = qso.PtcHost(peer=_Peer(), mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WW2MI")
    host.arq.on_rx_cs(CS_BREAKIN)
    check("the break-in answers the call and hands the channel to the gateway",
          host.arq.state == State.CONNECTED and host.arq.role == IRS,
          f"{host.arq.state} / {host.arq.role}")

    greeting = b"RMS Trimode 1.4.2.0 WW2MI Winlink Gateway"
    host.arq.on_rx_packet(1, greeting, spec.status_byte(0), crc_ok=True)
    check("...and the greeting under its reset counter reaches the host, rather "
          "than an acknowledgement for a packet nobody sent",
          greeting in qso.rx_of(host), repr(qso.rx_of(host)[:32]))

    host.arq.on_rx_packet(1, greeting, spec.status_byte(0), crc_ok=True)
    check("...while the peer's own repeat of it is still only acknowledged",
          qso.rx_of(host).count(greeting) == 1, repr(qso.rx_of(host)[:64]))


# One stage reads the two off-air recordings; everything else here is synthesis
# and needs nothing outside the package. They shared a skipif, so an absent
# `silent.wav` -- which is every installed wheel, the recordings being
# uncommitted -- took the changeover packet, the grid arithmetic and the whole
# live loop down with it.
SYNTHETIC = (the_packet, partial(the_grid, d_ms=90), partial(the_grid, d_ms=40),
             the_changeover_placement, partial(the_changeover_placement, d_ms=48.5),
             over_audio, the_budget, the_short_window, the_live_loop,
             the_receiving_station, the_changeover_the_peer_never_takes,
             the_head_of_a_packet_we_could_not_read, the_greeting_behind_the_answer)
RECORDED = (the_anchor_on_real_audio,
            the_break_in_the_anchor_could_not_be_aimed_at)


def _run(stages) -> int:
    global ok
    ok = True
    for stage in stages:
        stage()
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def main() -> int:
    return _run(SYNTHETIC + RECORDED)


def test_main() -> None:
    assert _run(SYNTHETIC) == 0


@pytest.mark.skipif(
    not all(p.exists() for p in SILENT),
    reason=f"the off-air gateway recordings are not under {archive.ARCHIVE}/captures — "
           "they live in the working record, which does not cross the "
           "publication boundary")
def test_the_anchor_on_real_audio() -> None:
    assert _run((the_anchor_on_real_audio,)) == 0


@pytest.mark.skipif(
    not archive.K4MSU_BREAKIN.exists(),
    reason=f"{archive.K4MSU_BREAKIN} is a session recording, written by a run "
           "and under captures/, which is not in the index")
def test_the_break_in_the_anchor_could_not_be_aimed_at() -> None:
    assert _run((the_break_in_the_anchor_could_not_be_aimed_at,)) == 0


if __name__ == "__main__":
    sys.exit(main())
