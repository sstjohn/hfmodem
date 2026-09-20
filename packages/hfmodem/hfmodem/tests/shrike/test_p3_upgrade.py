# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The link follows a REAL station's PACTOR-1 -> PACTOR-3 upgrade.

The loopback in `test_qso.py` proves the two halves of shrike agree with each
other, which is worth having and is not evidence about the protocol: both ends
share every belief we hold, so a misreading is symmetric and invisible. This asks
the other question. A recorded off-air session does exactly what shrike is now
built to do -- a PACTOR-1 connect, a couple of PACTOR-1 cycles, and then PACTOR-3
data on the raster the connect set -- and the link layer either follows it or it
does not.

Nothing here is a fixture of our own making. The events come out of the real
audio through the same decoders the on-air loop runs, and the only thing under
test is what `ptc.PtcHost` and `arq.PactorArq` do when a peer stops transmitting
one protocol and starts transmitting another.

WHAT THE RECORDING SAYS, measured from the audio alone:

    1.0, 2.0 s   PACTOR-1 connect bursts naming DL6MAA
    3.16 s       PACTOR-1 CS1 at zero bit errors -- the connect answer
    4.41 s       one more control signal, on the 1.25 s raster
    6.60 s       PACTOR-3 CS5 at zero bit errors -- the first PACTOR-3 thing
                 on the air, and it is the reverse channel rather than a packet
    6.91 s       first PACTOR-3 packet
    5.56, 7.77 s PACTOR-3 CS3 at zero bit errors, and 0.81 s of unbroken
                 carrier behind each: the link changing hands, once in each
                 direction, through the packet the codeword heads
    9.11 s on    PACTOR-3 speed level 3, one packet per 1.25 s cycle,
                 status-byte counter 1, 2, 3, 0, 1, ... unbroken
    11.37 s      PACTOR-3 CS1 at zero bit errors, in the turnaround slot

The counter is the sharp part: shrike's own receiver expects 1 from a link it has
just brought up (`arq._enter_connected`), and a real station's first data packet
carries 1. So a session that follows this recording is agreeing with a stranger's
modem about where the sequence starts, not with itself.

AND THE 4.41 s BURST IS THE HANDSHAKE'S OTHER HALF. It reads `0x59A` at zero bit
errors -- the PACTOR-3 upgrade grant -- and the station that received it keyed its
first PACTOR-3 packet 0.198 s later, one turnaround, in the slot its next PACTOR-1
packet would have gone in. `pos_pactor1_local.wav` is an independent measurement
of the same exchange: grant at 12.443 s, PACTOR-3 tone energy at 12.60 s. So the
recordings answer a question the replay above cannot ask -- what a station does
when the peer COMMANDS the upgrade -- and `_grant_arm` drives this station's own
transmit path from the real codeword.

`ref_occ15_pactor3.wav` is the first fifteen seconds of that session and covers
the transition; `PIII_Complete_1.wav` is the whole of it and covers the ladder --
speed levels 3, 4, 5 and 6 and BOTH cycle lengths, which is the part a loopback
between two of our own instances can say nothing about, because shrike neither
transmits a long cycle nor climbs past what its own IRS asks for. Both are driven
where both are present.

Neither capture is in the repo -- between them they are 18 MB of audio -- so each
arm skips cleanly without its file and the module skips when neither is here.

Run:  python -m hfmodem.tests.shrike.test_p3_upgrade
"""
from __future__ import annotations

import dataclasses
import functools
import re
import sys
from pathlib import Path

import numpy as np

from hfmodem.shrike import (compress, onair, p1rx, p3frame, p3rx, pactor1,
                            placement, ptc, rx as _rx, rxfront, session, spec)
from hfmodem.shrike import arq as arq_mod
from hfmodem.shrike.arq import (CS_ACK, CS_BREAKIN, CS_REQUEST, IRS,
                                State)
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.shrike.test_p3_offer import (MESSAGE, Keyed, cs_event,
                                                calling_station,
                                                upgraded_station)

CAPTURE = corpora.RF_CORPUS / "ref_occ15_pactor3.wav"
FULL = corpora.RF_CORPUS / "PIII_Complete_1.wav"
LOCAL = corpora.RF_CORPUS / "regress" / "fixtures" / "pos_pactor1_local.wav"
"""A grant this station recorded off its own antenna, at a different gateway on a
different day from the DL6MAA session -- so `_grant_arm` does not rest on one
recording of one exchange."""

GRANTS = (FULL, LOCAL)
"""The recordings whose `0x59A` this receiver reads.

`ref_occ15_pactor3.wav` is the same exchange and is not one of them: no PACTOR-1
codeword survives in that copy -- neither the grant nor the CS1 connect answer
that precedes it -- so what it would measure is the transfer, not the grant."""
DXCALL = "DL6MAA"
"""The station the recording's connect burst names -- so, for a receiver
following that session, the callsign being called."""

SLOT_N = round(spec.CYCLE_SHORT_S * rxfront.FS)
P1_PACKET_N = round(spec.P1_PACKET_S * rxfront.FS)
P1_CS_N = round(spec.P1_CS_S * rxfront.FS)
D_MAX_N = onair._d_max_n(spec.CYCLE_SHORT_S, 0.04)
"""The grid a station calling this recording's peer would hold. Only `_window_arm`
reads them, and only to place the receive window the production loop places."""


@functools.cache
def _events(path: Path) -> list:
    """Every event the real receiver finds in the recording, in TIME order.

    `decode_events` yields the PACTOR-3 data pass ahead of the hop loop, because
    the two are separate passes over the buffer rather than one walk along it. A
    station on the air sees them in the order they arrive, and feeding a link its
    data packets before the connect that opened it tests nothing at all.
    """
    audio = session.load_wav(str(path), rxfront.FS)
    return sorted(rxfront.decode_events(audio), key=lambda e: e.t)


def _replay(path: Path) -> tuple[PtcHost, list, list, list]:
    """Drive one station through the whole recording, and report what it saw.

    A listening station, called by the recording's own connect, taking every
    event the receiver produces in the order the air produced it.
    """
    host = PtcHost(peer=None, mycall=DXCALL)
    host.arq.on_host_listen(True)
    followed: list = []
    delivered: list = []
    before_qrt: list = []
    rx = host.channel(host.ptchn).rx
    for ev in _events(path):
        was = (host.protocol, host.arq.state)
        before = len(rx)
        if (ev.protocol == Protocol.PACTOR3 and ev.packet is not None
                and ev.packet[3] and ev.packet[1] & spec.STATUS_QRT):
            before_qrt.append((ev, host.protocol, host.arq.state,
                               host.arq._breakin_pending))
        host.on_rx_event(ev)
        if host.protocol is not was[0] or host.arq.state is not was[1]:
            followed.append((ev, was, (host.protocol, host.arq.state)))
        if host.arq.state == State.CONNECTED and ev.kind == "packet":
            delivered.append((ev.t, ev.protocol, ev.packet[0],
                              len(ev.packet[2]), len(rx) - before,
                              ev.packet[1] & spec.STATUS_SEQ))
    return host, followed, delivered, before_qrt


def _reverse_channel(path: Path) -> list[tuple[float, int, int, int]]:
    """The IRS's answer to each of the recording's packets: (t, sl, counter, cs).

    `decode_events` does not report these, and that is a property of the receiver
    rather than of the recording: reading a control signal costs a search over
    alignments, so it is spent only where an answer is EXPECTED, and a file
    replay expects nothing. The packets say where to look -- the answer falls in
    the turnaround between one packet and the next -- so the search here is aimed
    the same way `rxfront.SyncedRx` aims it on a live link, and takes the same
    zero-of-twenty at mutual distance twelve as a decode.
    """
    from hfmodem.shrike import rx as _rx
    audio = session.load_wav(str(path), rxfront.FS)
    sps, fs = rxfront.SPS, rxfront.FS
    pulse = _rx._pulse(sps)
    delay = (len(pulse) - 1) // 2
    Z = {cn: _rx._baseband(audio, cn, fs, pulse) for cn in rxfront.HDR_TONES}
    packets = sorted(p3rx.decode_p3_packets(audio).packets, key=lambda p: p.start)
    out = []
    for i, p in enumerate(packets):
        stop = packets[i + 1].start if i + 1 < len(packets) else p.start + int(4.0 * fs)
        span = range(p.start + int(0.7 * fs), min(stop, len(audio)), sps // 8)
        ci, be, _ = rxfront._best_cs(Z, delay, 0, span)
        if be == 0:
            out.append((p.start / fs, p.sl, p.status & spec.STATUS_SEQ, ci))
    return out


def _ack_arm(path: Path, check) -> None:
    """FINDING 4: in PACTOR-3 the acknowledgement IS the alternation.

    Read straight off the audio, and then read back to the link layer. A real IRS
    answers an EVEN packet counter with CS1 and an ODD one with CS2, and shrike
    mapped CS2 to a repeat request -- so the first packet a real peer ever
    confirmed, which is packet #1, arrived as "send that again". We resent it, the
    peer answered the duplicate the same way, and the link could not move. The
    counter pinned at #1 is the K0NTS signature, 33 codewords at zero bit errors.
    """
    answers = _reverse_channel(path)
    print(f"\n  the reverse channel in {path.name}:")
    for t, sl, seq, ci in answers:
        print(f"    {t:6.2f}  SL{sl}  counter {seq}  answered "
              f"CS{ci + 1} {spec.CS_NAMES[ci]}")
    check("the reverse channel decoded at all", len(answers) >= 4,
          f"{len(answers)} answers")
    if len(answers) < 4:
        return

    # THE ENTRY PACKET IS NOT IN THE ALTERNATION, and what stands in its slot is
    # not an acknowledgement at all. Read in the band a live receiver aims at --
    # `_short_cycle_answers`, which `_window_arm` then shows the production grid
    # reaches -- so this is the codeword a station keying that packet would have
    # got back, not one found somewhere in the cycle.
    _, band = _short_cycle_answers(path)
    entry = [ci for pkt, _, ci in band if pkt.sl == 1]
    check("the recording's speed-level-1 entry packet is answered in the "
          "turnaround an acknowledgement would come back in",
          len(entry) == 1, f"{len(entry)} answered")
    check("...and the answer is a BREAK-IN -- the IRS takes the channel rather "
          "than confirming",
          entry == [CS_BREAKIN],
          ", ".join(f"CS{ci + 1} {spec.CS_NAMES[ci]}" for ci in entry))
    # So the steady-state train is what the alternation is counted over. One
    # entry packet is one observation, not a rule about entry packets.
    even = [a for a in answers if a[2] % 2 == 0 and a[1] > 1]
    odd = [a for a in answers if a[2] % 2]
    check("an EVEN packet counter is answered CS1, every time",
          all(a[3] == CS_ACK for a in even),
          f"{len(even)} even, exceptions "
          f"{[(a[0], a[2], a[3] + 1) for a in even if a[3] != CS_ACK]}")
    # CS4 and CS6 substitute in the odd slot -- they carry a gear command AND this
    # cycle's acknowledgement, because the IRS emits exactly one codeword a cycle.
    # What never appears there is CS1.
    check("...and an ODD one is answered CS2, or a codeword that is not CS1",
          all(a[3] != CS_ACK for a in odd),
          f"{len(odd)} odd, exceptions "
          f"{[(a[0], a[2], a[3] + 1) for a in odd if a[3] == CS_ACK]}")

    # THE DEADLOCK, counted. Every CS2 in that train is a confirmation, and the
    # unfixed reader turned each one into a retransmission of a delivered packet.
    plain = [a for a in answers if a[3] in (CS_ACK, CS_REQUEST)]
    check("...so a reader that takes CS2 for a repeat request misreads every "
          "odd counter in it",
          sum(1 for a in plain if a[3] == CS_REQUEST) >= 2,
          f"{sum(1 for a in plain if a[3] == CS_REQUEST)} of {len(plain)}")

    # And the train delivered to a station SENDING, which is the arm that says the
    # link would move. CS3 is left out because it is not an acknowledgement at all
    # -- it is the head of the IRS's own changeover packet, and a station that
    # honours it stops transmitting, so what follows would be a different test.
    #
    # The harness opens its link one packet ahead of the recording's, so the first
    # answer lands on the wrong parity and reads as "not yet" -- which is right,
    # and is the thing worth watching: the counter is the phase, so a station that
    # joins the alternation out of step FINDS it, in one cycle, and then advances
    # once per answer for the rest of the ladder. Nothing has to be seeded.
    train = [ci for _, _, _, ci in answers if ci != CS_BREAKIN]
    if len(train) < 8:
        return
    host, keyed = upgraded_station()
    seqs = []
    for ci in train:
        host.arq.on_host_data(b"x" * 64)
        if host.arq._inflight is None:
            host.tick()
        seqs.append(host.arq._inflight.seq)
        host.on_rx_event(rxfront.Event(0.0, "cs", "", protocol="PACTOR-3", cs=ci))
        host.tick()
    check("driven by that train, a sending station's counter advances once per "
          "answer, from the cycle it finds the phase",
          len(set(seqs)) == 4
          and seqs[1:] == [(seqs[1] + i) % 4 for i in range(len(train) - 1)],
          f"counters {seqs}")
    check("...and it keyed a packet for each of them",
          len(keyed.p3) >= len(train), f"{len(keyed.p3)} keyed for {len(train)}")

    # THE K0NTS SIGNATURE, reproduced -- and now it is a diagnosis rather than our
    # own defect. A peer whose answer never changes parity is answering the same
    # packet every time, which is what "send it again" IS, so the counter moves
    # once and then stops. Thirty-three codewords at zero bit errors with `#1` on
    # the air is that, and it was us: shrike-as-IRS sent a constant CS1.
    host, _ = upgraded_station()
    stuck = []
    for _ in range(4):
        stuck.append(host.arq._inflight.seq)
        host.on_rx_event(rxfront.Event(0.0, "cs", "", protocol="PACTOR-3",
                                       cs=CS_ACK))
        host.tick()
    check("...while a peer that never alternates pins it after one packet",
          len(set(stuck[1:])) == 1, f"counters {stuck}")


def _arm(path: Path, check) -> None:
    """The claims every real PACTOR-1 -> PACTOR-3 recording has to support."""
    host, followed, delivered, before_qrt = _replay(path)
    connect = next(((ev, was) for ev, was, _ in followed
                    if was[1] != State.CONNECTED), (None, None))
    upgrade = next(((ev, was) for ev, was, now in followed
                    if was[0] is Protocol.PACTOR1
                    and now[0] is not Protocol.PACTOR1), (None, None))

    check("the real PACTOR-1 connect brings the link up",
          any(was[1] != State.CONNECTED and now[1] == State.CONNECTED
              and now[0] is Protocol.PACTOR1 for _, was, now in followed),
          "recorded connection transition")
    check("a link opened by a PACTOR-1 connect starts in PACTOR-1",
          connect[1] is not None and connect[1][0] is Protocol.PACTOR1)
    # A CONTROL SIGNAL COUNTS, and in these recordings it is what arrives first.
    # The turnaround at t=6.60 reads CS5 at ZERO bit errors of twenty at mutual
    # distance twelve, a third of a second ahead of the first data packet, and the
    # one at 11.37 reads CS1 the same way. Requiring a packet would have been an
    # assertion about the recording rather than about the protocol: capability is
    # what a station transmits, and a station that transmits a PACTOR-3 codeword
    # has said what it is as plainly as one that transmits a PACTOR-3 frame.
    check("the peer's first PACTOR-3 frame upgrades the link",
          upgrade[0] is not None and upgrade[0].protocol == Protocol.PACTOR3,
          "nothing upgraded it" if upgrade[0] is None
          else f"on a {upgrade[0].protocol} {upgrade[0].kind} "
               f"at t={upgrade[0].t:.2f}")
    # The full recording ends in a CRC-valid status0x98 QRT at74.7175s.
    # The improved reader now reaches it; asserting CONNECTED after it would
    # require ignoring a real goodbye. Preserve the active-link assertions at
    # the event boundary, then require the exact terminal cleanup separately.
    if path == FULL:
        check("the recording ends in its measured SL5 counter0 QRT",
              len(before_qrt) == 1 and before_qrt[0][0].packet == (5, 0x98, b"", True)
              and abs(before_qrt[0][0].t - 74.7175) < rxfront.SPS / rxfront.FS)
        active_protocol, active_state, pending = (
            before_qrt[0][1:] if len(before_qrt) == 1 else (None, None, False))
        check("...and that QRT returns the station to LISTENING in PACTOR-1",
              host.arq.state == State.LISTENING
              and host.protocol is Protocol.PACTOR1
              and not host.arq._breakin_pending,
              f"{host.arq.state} {host.protocol}")
    else:
        check("the truncated recording contains no terminal QRT", not before_qrt)
        active_protocol, active_state, pending = (
            host.protocol, host.arq.state, host.arq._breakin_pending)
    check("the active link stays in PACTOR-3 through the recording's data",
          active_protocol is Protocol.PACTOR3, str(active_protocol))
    check("the link stays up through the last data, before any QRT",
          active_state == State.CONNECTED, str(active_state))

    p3 = [d for d in delivered if d[1] == Protocol.PACTOR3]
    check("the peer's PACTOR-3 packets reach the FSM as data", len(p3) >= 4,
          f"{len(p3)} PACTOR-3 packets")
    # AND THE ENTRY PACKET IS NOT THE FIRST PACKET OF THE TRAIN. Its counter is
    # the last of the numbering the PACTOR-1 phase was keeping -- 2, behind the
    # one data packet that phase sent -- and the train that follows opens at 1,
    # because the IRS broke in between the two and a changeover restarts the
    # count. A receiver that reads the entry packet as packet zero of the train
    # expects 3 next and grades the greeting a gap; the rule that stops it is in
    # `arq.PactorArq.on_rx_packet`, and this is the observation behind it.
    entry = next((d for d in p3 if d[2] == 1), None)
    train = [d for d in p3 if d[2] > 1]
    check("the entry packet's counter does not continue into the data train",
          entry is not None and bool(train)
          and train[0][5] != (entry[5] + 1) % 4,
          f"entry counter {entry[5] if entry else '--'}, then "
          f"{[d[5] for d in train[:5]]}")
    # WHAT A THIRD-PARTY RECORDING CAN AND CANNOT SUPPORT, said rather than left
    # for a byte count to imply. The first PACTOR-3 packet in these recordings
    # carries a CHANGEOVER REQUEST (status 0x5d, bit 6). A request is an
    # invitation, not a transfer: the link changes hands only when the IRS
    # transmits its CS3-headed break-in (measured against WS8EOC 2026-08-03 --
    # a real gateway acknowledges the bit-6 packet and remains the IRS -- and
    # this recording says the same from the other side: the requesting station
    # carries on transmitting, a whole speed ladder of packets, which it could
    # not do had the ack handed the link over). So the receiver keeps
    # acknowledging and delivering while the request stands, with its break-in
    # scheduled; a replay drives RX events with no cycle ticks, so the break-in
    # stays pending until the recorded QRT, or the truncated crop ends. This used to
    # assert delivery STOPPED at the request -- the ack-turnaround model, which
    # is the reading the on-air sessions refuted.
    # What the host is owed is the field's CHARACTERS, not its wire coding: the
    # link layer decodes each payload by its own status byte (shrike.compress),
    # so a 212-byte PMC field arrives as its 41 characters and an idle fill --
    # most of these recordings' short cycles -- arrives as nothing at all. This
    # used to assert wire bytes reached the port byte for byte, which the idle
    # frames satisfied with coded fill the host could do nothing with. The
    # arbiter now is the plaintext itself, which for this recording is known --
    # DL6MAA is transmitting the SCS PTC-II manual.
    handed = [d for d in delivered if d[4]]
    rx_bytes = bytes(host.channel(host.ptchn).rx)
    check("the payload of a packet received AS THE IRS reaches the host data "
          "port, decoded to the characters its status byte declares",
          bool(handed) and b"Maildrop QRV" in rx_bytes,
          f"{len(handed)} delivered, port opens {rx_bytes[:48]!r}")
    check("...and delivery continues past the changeover request, break-in "
          "scheduled",
          len(delivered) >= 4 and delivered[-1][0] > delivered[0][0]
          and pending,
          f"{len(delivered)} acknowledged through t={delivered[-1][0]:.2f}"
          f"{'' if pending else ', BREAK-IN NOT SCHEDULED'}"
          if delivered else "nothing delivered")

    # WHAT A LOOPBACK CANNOT ASK. shrike neither transmits a long cycle nor climbs
    # past what its own receiving end asks for, so the only place these two are
    # exercised is a real station's session: the ladder runs 3 -> 4 -> 5 -> 6 and
    # the fields go from 59 bytes to 1276, which is the long cycle. A receiver
    # that could not follow either would deliver the first few packets and then go
    # quiet, and the counter would break at the change.
    levels = sorted({d[2] for d in p3})
    long_cycle = [d for d in p3 if d[3] > spec.SPEED_LEVELS[d[2]].payload_short]
    if len(delivered) > 8:
        check("the link follows the speed-level ladder", len(levels) >= 3,
              f"levels {levels}")
        check("...and the long cycle, which is 4.5 times the grid",
              bool(long_cycle),
              f"{len(long_cycle)} long-cycle packets, "
              f"largest field {max((d[3] for d in p3), default=0)}B")
        # The escape and the run seam, exercised by the recording itself: this
        # sentence carries a codepage-437 umlaut and arrives split across two
        # CRC-validated fields, and it must land at the port in one piece.
        check("...as the manual's own prose, readable across packet boundaries",
              "wunsch verfügbar. Es sind keinerlei".encode("cp437") in rx_bytes,
              repr(rx_bytes[130:190]))

    print(f"\n  what the receiver followed in {path.name}:")
    for t, proto, sl, n, gave, seq in delivered:
        # A PACTOR-1 packet reports speed level 0, which is not a rung on the
        # PACTOR-III ladder, and the recording carries one: DL6MAA's 200 Bd
        # announcement, ahead of the upgrade.
        rung = spec.SPEED_LEVELS.get(sl) if proto is Protocol.PACTOR3 else None
        cyc = "long " if rung and n > rung.payload_short else "short"
        print(f"    {t:6.2f}  {proto}  SL{sl}  {cyc}  {n:5d}B  counter {seq}"
              f"{f'  -> host {gave}B' if gave else ''}")


def _announcement_arm(path: Path, check) -> None:
    """The PACTOR-1 announcement that opens the recording, as the host reads it.

    Audio to data port, through the shipped decoders and the shipped link layer.
    DL6MAA's 200 Bd announcement is the corpus's only PACTOR-1 field from another
    station that is compressed, and the two readings of its status byte -- 0x35 --
    give different characters: two bits make it plain Huffman and it says
    `1dl6maa`, which is the whole point of an announcement and what an independent
    decoder reads off this same audio, while three bits make it PMC German swapped
    and it says `1DIT)   WS DUNZ0 AN`. The link layer read three bits on both
    seams, so that is what opened the port for as long as the frame decoded at
    all.
    """
    host, _, delivered, _ = _replay(path)
    rx_bytes = bytes(host.channel(host.ptchn).rx)
    p1 = [d for d in delivered if d[1] == Protocol.PACTOR1]
    check("the recording's PACTOR-1 phase reaches the link layer at all",
          any(d[4] for d in p1),
          f"{len(p1)} of {len(delivered)} packets are PACTOR-1")
    check("...and the host data port opens with the caller's own callsign",
          rx_bytes.startswith(b"1dl6maa\r"), repr(rx_bytes[:40]))


def _codewords(audio: np.ndarray) -> list[tuple[int, int]]:
    """(alignment, codeword) for every control signal `audio` holds whole.

    The blind sweep, not the aimed read: nothing here knows where a codeword is
    due, so it has to be its own evidence -- which at mutual distance twelve over
    twenty bits, taken at zero errors, it is.
    """
    fs, sps = rxfront.FS, rxfront.SPS
    pulse = _rx._pulse(sps)
    delay = (len(pulse) - 1) // 2
    Z = {cn: _rx._baseband(audio, cn, fs, pulse) for cn in rxfront.HDR_TONES}
    out: list[tuple[int, int]] = []
    for st in range(0, max(0, len(audio) - 22 * sps), sps // 16):
        bits = _rx.cs_bits(Z, st, delay)
        if bits is None:
            break
        ci, be = _rx.nearest_control_signal(bits)
        if be == 0 and (not out or st - out[-1][0] > sps):
            out.append((st, ci))
    return out


def _cs3_heads(audio: np.ndarray) -> list[int]:
    """Where every changeover packet begins: the CS3s among `_codewords`."""
    return [at for at, ci in _codewords(audio) if ci == CS_BREAKIN]


def _changeover_arm(path: Path, check) -> None:
    """FINDING: a CS3 head and a PACTOR-3 frame are ONE keying, and this is it.

    `ptc.PtcHost.breakin_now` used to drop an upgraded link to PACTOR-1 to take
    the channel, because nothing had established how the two join and a plausible
    construction offered to the air is how a wrong waveform gets frozen in. This
    recording carries the construction twice, in opposite directions -- the IRS
    takes the link after the entry packet and the ISS takes it back after the
    greeting -- and everything asserted below is read off those two bursts.

    THE FIELD IS THE PIN. `placement.CHANGEOVER`'s geometry was fitted to these
    two bursts jointly, so a CRC on one of them proves nothing that a large enough
    search would not have found; what a search cannot arrange is that the three
    bytes it yields are the three characters missing from the front of the
    sentence the NEXT packet carries. They are.
    """
    audio = session.load_wav(str(path), rxfront.FS)
    fs, sps = rxfront.FS, rxfront.SPS
    pulse = _rx._pulse(sps)
    delay = (len(pulse) - 1) // 2
    heads = _cs3_heads(audio)
    check("the recording carries a CS3 at zero bit errors in each direction",
          len(heads) >= 2, f"{len(heads)} at "
          f"{', '.join(f'{h / fs:.4f}' for h in heads)}")
    if len(heads) < 2:
        return

    packets = sorted(p3rx.decode_p3_packets(audio).packets, key=lambda p: p.start)
    for head in heads:
        t = head / fs
        # ONE KEYING. The head and the frame behind it are 81 symbols of carrier
        # that never drops: the quietest symbol of the burst stands over half the
        # median, where a re-key would take it to the noise floor.
        n = round(placement.PACKET_S * fs)
        seg = audio[head:head + n].astype(float)
        rms = np.sqrt((seg[:seg.size // sps * sps].reshape(-1, sps) ** 2).mean(1))
        check(f"the CS3 at {t:.4f} heads one continuous {placement.PACKET_S:.2f} s "
              f"keying, not a codeword and then a packet",
              rms.min() > 0.5 * float(np.median(rms)),
              f"quietest symbol {rms.min() / np.median(rms):.2f} of the median")
        # ...on the two carriers a control signal already rides, and no others, so
        # it is not keyed at the speed level the traffic either side of it runs at.
        levels = p3rx.levels_present(p3rx.channel_energy(audio[head:head + n]))
        check("...on the two-carrier comb alone, whatever level the traffic runs",
              levels == (1,), f"levels {levels}")
        # ...and carrying no packet header block, which is what says the frame
        # behind the head is not an ordinary packet displaced.
        Zv = {cn: _rx._baseband(audio[head:head + n], cn, fs, pulse)
              for cn in spec.VH_CHANNELS}
        check("...and no packet header block anywhere in it",
              not p3rx.vh_anchors(Zv, n, fs=fs, step=sps // 8),
              f"{len(p3rx.vh_anchors(Zv, n, fs=fs, step=sps // 8))} anchors")

    fields = [p3rx.decode_changeover(audio, h) for h in heads]
    check("EVERY one of them heads a frame this geometry reads -- which is the "
          "claim, because a CS3 that headed nothing would decode to nothing",
          all(ok for _, ok in fields),
          ", ".join(f.hex() if ok else "--" for f, ok in fields))
    if not all(ok for _, ok in fields):
        return
    check("...and the first two are the fields DL6MAA's two stations exchanged "
          "the link with",
          [f.hex() for f, _ in fields[:2]] == ["0d5054202761", "0f8f87180155"],
          ", ".join(f.hex() for f, _ in fields[:2]))
    check("...every one under counter 0, which is the reset a changeover "
          "performs and `arq.PactorArq.on_cycle` already did",
          all(f[3] & spec.STATUS_SEQ == 0 for f, _ in fields),
          ", ".join(f"0x{f[3]:02x}" for f, _ in fields))
    # THE CORROBORATION A CRC CANNOT BE. The IRS's three bytes are ASCII by their
    # own status byte, and the 212-byte field it sends in the very next cycle
    # decompresses to a sentence that begins in the middle of a word.
    greeting = next((p for p in packets if p.start / fs > heads[0] / fs
                     and len(p.payload) > 100), None)
    text = (b"" if greeting is None else
            compress.decompress(greeting.payload, (greeting.status >> 2) & 7))
    check("...and the IRS's three bytes are the head of the sentence its NEXT "
          "packet carries, which is what no search could have arranged",
          text.startswith(b"C-II") and fields[0][0][:3] == b"\rPT",
          f"{fields[0][0][:3]!r} + {text[:16]!r}")

    # AND WHAT WE KEY IS WHAT THEY KEYED. Rendering the IRS's own payload and
    # status through `placement.changeover_packet` reproduces its field byte for
    # byte -- so the geometry is not merely one our own decoder agrees with.
    for swapped in (False, True):
        pkt = np.concatenate([np.zeros(sps),
                              placement.changeover_packet(b"\rPT", 0x20,
                                                          swapped=swapped),
                              np.zeros(4 * sps)]).astype(np.float32)
        Zp = {cn: _rx._baseband(pkt, cn, fs, pulse) for cn in rxfront.HDR_TONES}
        ci, be, at = rxfront._best_cs(Zp, delay, 6 * sps)
        check(f"our own changeover packet (swap {int(swapped)}) opens on a CS3 a "
              f"receiver reading for an acknowledgement finds",
              (ci, be) == (CS_BREAKIN, 0),
              f"CS{ci + 1} {spec.CS_NAMES[ci]} at {be} bit errors")
        field, ok = p3rx.decode_changeover(pkt, at)
        check("...and its field is DL6MAA's, byte for byte",
              ok and field == fields[0][0],
              field.hex() if ok else "no decode")


def _rotation_arm(path: Path, check) -> None:
    """FINDING: the changeover rotation has to move with the protocol too.

    `_MasterGrid.cs_n` is the other half of the defect `data_n` carried: PACTOR-1's
    120 ms codeword, assigned once at grid setup and held whatever the link was in.
    `reverse` spends it as `data_n - cs_n`, so an upgraded link rotated 690 ms
    where it owes 600, and `rx_due` as the IRS, `rx_due_in`'s window fit,
    `key_refusal`'s fit test and the listen-window floor all read the same 120.

    None of it was reachable before a changeover could be keyed in PACTOR-3 at
    all, and this station has never been the IRS on a PACTOR-3 link -- so the
    figures those paths ran on have no corroboration of ours. The recording has a
    real pair doing exactly this, twice, in opposite directions.
    """
    audio = session.load_wav(str(path), rxfront.FS)
    fs, sps = rxfront.FS, rxfront.SPS
    words = _codewords(audio)
    heads = [at for at, ci in words if ci == CS_BREAKIN][:2]
    refs = sorted(_phase_ref(p) for p in p3rx.decode_p3_packets(audio).packets)
    check("the recording hands the link over in both directions",
          len(heads) == 2 and len(refs) >= 4,
          f"{len(heads)} changeovers, {len(refs)} packets")
    if len(heads) != 2 or len(refs) < 4:
        return
    cycle_n = float(np.median([b - a for a, b in zip(refs, refs[1:])
                               if b - a < 2 * SLOT_N]))

    # WHAT THE STATION YIELDING THE CHANNEL DOES, driven through the production
    # grid: it holds its own packet's phase reference, reverses, and keys a
    # codeword one cycle plus one rotation later. Read back at that instant by
    # the production reader at the production tolerance.
    for head in heads:
        held = max(r for r in refs if r < head)
        grid = onair._MasterGrid(held, SLOT_N, 0, packet_n=P1_PACKET_N,
                                 cs_n=P1_CS_N, d_max_n=D_MAX_N)
        grid.protocol = Protocol.PACTOR3
        grid.reverse(to_iss=False)
        at = grid.boundary(1)
        anchored = held + SLOT_N + (onair.P3_PACKET_N - P1_CS_N)
        rx = rxfront.SyncedRx()
        check(f"the station yielding at {head / fs:.4f} s keys its codeword where "
              f"a grid rotating {(grid.data_n - grid.cs_n) / fs * 1e3:.0f} ms "
              f"predicts",
              rx.control_signal_at(audio, at) is not None,
              f"predicted {at / fs:.4f} s")
        check("...and nothing is there for the PACTOR-1 codeword length, which "
              "puts the rotation 90 ms out",
              rx.control_signal_at(audio, anchored) is None,
              f"predicted {anchored / fs:.4f} s")

    # THE ARBITER between 21 symbols and the 22 on the air, and it needs no
    # packet: `reverse`'s own involution says the turnarounds either side of a
    # reversal sum to the cycle less a packet and a codeword, and each of these
    # is one changeover packet's phase reference to the next codeword's -- so
    # what a codeword is worth falls out of four control-signal alignments.
    left = [next(at for at, _ in words if at > head + onair.P3_PACKET_N)
            - head - onair.P3_PACKET_N for head in heads]
    measured = cycle_n - onair.P3_PACKET_N - sum(left)
    print(f"\n  the codeword {path.name} rotates for: "
          f"{cycle_n / fs * 1e3:.1f} ms cycle less {onair.P3_PACKET_N / fs * 1e3:.0f} "
          f"of packet less turnarounds of "
          f"{' and '.join(f'{d / fs * 1e3:.1f}' for d in left)} ms "
          f"= {measured / fs * 1e3:.1f} ms")
    check("the two reversals leave a codeword of 21 symbols, not the 22 the "
          "recording keys",
          abs(measured - onair.P3_CS_N) < abs(measured - (onair.P3_CS_N + sps)),
          f"{measured / fs * 1e3:.1f} ms against {onair.P3_CS_N / fs * 1e3:.0f} "
          f"and {(onair.P3_CS_N + sps) / fs * 1e3:.0f}")
    check("...and nothing like the PACTOR-1 codeword the grid was holding",
          abs(measured - P1_CS_N) > 4 * sps,
          f"{measured / fs * 1e3:.1f} ms against {P1_CS_N / fs * 1e3:.0f}")

    # WHAT THE 22nd SYMBOL IS. Every bare codeword in the recording holds full
    # amplitude one symbol past the twenty bits, at no phase step from the last
    # of them -- a run-out carrying nothing a reader takes, the same construction
    # as the changeover packet's four run-in symbols. What follows it is the
    # shoulder the leading edge has too.
    bare = [at for at, ci in words if ci != CS_BREAKIN]
    pulse = _rx._pulse(sps)
    delay = (len(pulse) - 1) // 2
    Z = {cn: _rx._baseband(audio, cn, fs, pulse) for cn in rxfront.HDR_TONES}
    tail, step, gone = [], [], []
    for at in bare:
        idx = at + np.arange(24) * sps + delay
        mag = sum(np.abs(Z[cn][idx]) for cn in rxfront.HDR_TONES)
        d = sum((Z[cn][idx][1:] * np.conj(Z[cn][idx][:-1])).real
                for cn in rxfront.HDR_TONES)
        med = float(np.median(mag[:21]))
        tail.append(mag[21] / med)
        step.append(float(d[20]) / float(np.median(np.abs(d[:20]))))
        gone.append(mag[23] / med)
    check("every bare codeword carries a 22nd symbol at full amplitude",
          min(tail) > 0.7, f"quietest {min(tail):.2f} of the burst median "
          f"over {len(tail)} codewords")
    check("...at no phase step from the 21st, so it repeats it and carries no "
          "bit", min(step) > 0.5, f"weakest {min(step):+.2f}")
    check("...and the carrier is gone by the 24th, so the burst is 22 symbols "
          "and not more", max(gone) < 0.1, f"loudest {max(gone):.2f}")


def _phase_ref(p) -> int:
    """The packet's first symbol -- the grid row a station keying it is on.

    `p3rx.P3Packet.start` names grid row 0, nine symbols later."""
    return p.start - p3frame.DATA_OFFSET * rxfront.SPS


@functools.cache
def _short_cycle_answers(path: Path) -> tuple[np.ndarray, list]:
    """(packet, answer instant, codeword) for every short cycle of the link.

    The answer is searched over the turnaround band the grid itself will believe
    -- `onair.D_MIN_S` to `onair._d_max_n`, measured from `_phase_ref` -- so a
    codeword found here is one a live receiver could be aimed at, and the search
    can neither reach into the next cycle nor manufacture a gap outside the band.
    """
    audio = session.load_wav(str(path), rxfront.FS)
    sps, fs = rxfront.SPS, rxfront.FS
    pulse = _rx._pulse(sps)
    delay = (len(pulse) - 1) // 2
    Z = {cn: _rx._baseband(audio, cn, fs, pulse) for cn in rxfront.HDR_TONES}
    out = []
    for p in sorted(p3rx.decode_p3_packets(audio).packets, key=lambda p: p.start):
        ref = _phase_ref(p)
        lo = ref + onair.P3_PACKET_N + round(onair.D_MIN_S * fs)
        hi = min(ref + onair.P3_PACKET_N + D_MAX_N, len(audio) - 22 * sps)
        if hi <= lo:
            continue
        ci, be, at = rxfront._best_cs(Z, delay, 0, range(lo, hi, sps // 16))
        if be == 0:
            out.append((p, at, ci))
    return audio, out


@functools.cache
def _entry_head_n() -> int:
    """Samples our own key instant leads the entry packet's phase reference by.

    Read off the render through the production decoder rather than written down:
    the transmit filter puts 40 ms in front of the first symbol, `_trim_silence`
    keeps what clears 2% of the peak, and what is left is a property of those two
    together. 35.0 ms at speed level 1, which is the level an upgrade enters at.
    """
    a = onair._trim_silence(placement.link_packet(1, b"\x00" * 5, 0))
    pad = round(0.5 * rxfront.FS)
    room = np.concatenate([np.zeros(pad, np.float32), a.astype(np.float32),
                           np.zeros(pad, np.float32)])
    p = min(p3rx.decode_p3_packets(room).packets, key=lambda q: q.start)
    return _phase_ref(p) - pad


def _window_arm(path: Path, check) -> None:
    """FINDING: the receive window has to move with the protocol we are keying.

    `_MasterGrid.rx_due` is our own transmission's end plus the turnaround, and
    the transmission it measured was PACTOR-1's 960 ms packet whatever the link
    was in. A real PACTOR-3 answer arrives 810 ms of packet plus a turnaround
    after the phase reference, so the window opened 110 ms behind it and no `d`
    could bring the two together -- the answer lands BEFORE the earliest instant
    the grid will look at, not merely off centre.

    Read off the recording with the production reader, at the production
    tolerance, from ONE turnaround measured across the session -- which is what a
    link that acquired once and free-runs has to work with.
    """
    audio, answers = _short_cycle_answers(path)
    if len(answers) < 4:
        check(f"{path.name} carries short-cycle PACTOR-3 answers to read",
              False, f"{len(answers)} found")
        return
    d_n = float(np.median([at - _phase_ref(p) - onair.P3_PACKET_N
                           for p, at, _ in answers]))
    print(f"\n  the turnaround {path.name} answers in: "
          f"{d_n / rxfront.FS * 1e3:.1f} ms over {len(answers)} cycles")

    def reads(protocol) -> int:
        hits = 0
        for p, _, _ in answers:
            grid = onair._MasterGrid(_phase_ref(p), SLOT_N, 0, packet_n=P1_PACKET_N,
                                     cs_n=P1_CS_N, d_max_n=D_MAX_N)
            grid.protocol, grid.d_n = protocol, d_n
            hits += rxfront.SyncedRx().control_signal_at(
                audio, grid.rx_due(0)) is not None
        return hits

    upgraded, anchored = reads(Protocol.PACTOR3), reads(Protocol.PACTOR1)
    check("a grid keying PACTOR-3 reads every one of them at its own instant",
          upgraded == len(answers), f"{upgraded} of {len(answers)}")
    check("...and the PACTOR-1 packet length reads none",
          anchored == 0, f"{anchored} of {len(answers)}")

    # NOT A TOLERANCE, A SIGN. Anchored on PACTOR-1's packet the answers imply a
    # NEGATIVE turnaround, so widening the search would not have found them.
    implied = [(at - _phase_ref(p) - P1_PACKET_N) / rxfront.FS * 1e3
               for p, at, _ in answers]
    check("...because that anchor puts the answer before our own data ends, so "
          "no turnaround the grid will believe can reach it",
          max(implied) < onair.D_MIN_S * 1e3,
          f"implied turnaround {min(implied):.1f} to {max(implied):.1f} ms, "
          f"against a floor of {onair.D_MIN_S * 1e3:.0f}")

    # AND WHAT `rx_ref_n`'S CARRY COSTS WHEN THE PEER DOES FOLLOW. The answer
    # moves here, three misses release the carried reference, and the search
    # reopens against the packet we are keying -- measured from OUR OWN key
    # instant, which leads the phase reference by the transmit filter our
    # renderer puts in front of the first symbol.
    ours = [(at - _phase_ref(p) + _entry_head_n() - onair.P3_PACKET_N)
            / rxfront.FS * 1e3 for p, at, _ in answers]
    check("...and a grid keying PACTOR-3 finds these again in one cycle of "
          "search, so the release after an upgrade the peer DID follow costs "
          "three cycles and not the link",
          onair.D_MIN_S * 1e3 <= min(ours)
          and max(ours) <= D_MAX_N / rxfront.FS * 1e3,
          f"{min(ours):.0f}-{max(ours):.0f} ms from our key, against a band of "
          f"{onair.D_MIN_S * 1e3:.0f}-{D_MAX_N / rxfront.FS * 1e3:.0f}")


def _grant_event(spare: int = pactor1.CS_59A):
    """An unassigned word where the answer to our packet is due."""
    return rxfront.Event(0.2, "unassigned", f"word {spare}", protocol="PACTOR-1",
                         spare=spare, sense=0)


def _granted_station() -> tuple[PtcHost, list]:
    """A CALLING station one unacknowledged announcement into a PACTOR-1 link --
    where the recordings' caller stands when the grant arrives -- keying into
    `onair.RadioTx` with the rig taken out, so each burst below is the audio a
    transmitter would have put on the air rather than a description of one."""
    keyed: list[tuple[Protocol, np.ndarray]] = []
    tx = onair.RadioTx(None, transmit=False, outdir=Path("."))
    host = ptc.PtcHost(peer=tx, mycall="W9SSJ")
    tx._tx = lambda audio, what, **kw: keyed.append(
        (host.protocol, onair._trim_silence(np.asarray(audio, np.float32))))
    tx.attach(host)
    host.arq.on_host_connect("W9SSJ", DXCALL)
    host.on_rx_event(cs_event(pactor1.CS_SPEED))      # CS4: the link runs 100 Bd
    host.tick()
    host.arq.on_host_data(MESSAGE)
    return host, keyed


UPGRADED = ("onair-0825-2101", "onair-0825-2105",
            "onair-0825-2108", "onair-0825-2111")
"""This station's own four upgraded arms of 2026-08-26 -- two gateways, 40 m and
80 m -- kept beside the corpus rather than in it. Each drew `0x59A` at zero bit
errors, upgraded, and keyed four speed-level-1 entry packets that nothing
answered inside the window; what they hold is the peer's own audio through those
cycles, which is where `_raster_arm` reads the answer that was there."""


def _upgraded_cycles(name: str):
    """(anchor, d at the upgrade, [(slot, window start, audio)]), or None.

    A post-upgrade window is one that OPENS where an entry packet ended: the
    capture resumes on the last sample our own carrier put out, so the offset
    from the boundary is the length of what was keyed and tells the two waveforms
    apart without consulting the log for it.
    """
    import json

    from hfmodem.tests import evidence

    d = evidence.CAPTURES / name
    if not (d / "session.log").exists():
        return None
    log = (d / "session.log").read_text()
    first = re.search(r"slot 1 boundary @ sample (\d+)", log)
    entry_at = log.find("SL1 pkt")
    if first is None or entry_at < 0:
        return None
    anchor = int(first.group(1)) - SLOT_N
    # What the grid held when the upgrade was keyed: the last turnaround it
    # printed before the first entry packet, acquired or tracked.
    gaps = re.findall(r"\[grid\] (?:d |.*: d = )([\d.]+) ms", log[:entry_at])
    if not gaps:
        return None
    # These arms flew before the render keyed the trailer symbol, so the length
    # that places their windows is one symbol short of what it keys today.
    entry = onair._trim_silence(placement.link_packet(1, b"\x00" * 5, 0)).size \
        - rxfront.SPS
    out = []
    for j in sorted(d.glob("hold_*.json")):
        side = json.loads(j.read_text())
        end = side["end_stream_sample"]
        start = end - side["samples"]
        slot = (start - anchor) // SLOT_N
        if abs(start - anchor - slot * SLOT_N - entry) > round(0.008 * rxfront.FS):
            continue
        out.append((slot, start, session.load_wav(str(j.with_suffix(".wav")),
                                                  rxfront.FS)))
    return (anchor, float(gaps[-1]), out) if out else None


def _is_grant(cs) -> bool:
    """Is this the word the peer repeats while it waits for the entry packet?"""
    return cs is not None and cs.index == pactor1.CS_59A and cs.errors == 0


def _raster_arm(name: str, check) -> tuple[int, int, int]:
    """FINDING: the peer answers on the cycle, not a turnaround after our packet.

    `rx_due` was our own data end plus `d`, so shortening the packet by 150 ms
    walked the window 150 ms earlier -- and the peer, which holds a free-running
    grid of its own and had not acknowledged the entry packet, went on answering
    where PACTOR-1 put it. `_MasterGrid.rx_ref_n` is the fix and this is what it
    is measured on: the peer's own bursts, through the production onset detector
    and the production anchored reader, at the anchor the grid actually holds.

    Returns `(cycles, answers read, accepts at the anchor this replaces)`.
    """
    got = _upgraded_cycles(name)
    if got is None:
        print(f"  SKIP -- no upgraded cycles under {name}")
        return 0, 0, 0
    anchor, d_ms, cycles = got
    reach = rxfront.SyncedRx.TRACK_SYMBOLS * rxfront.SPS
    held = heard = clear = both = stale_true = stale_junk = 0
    off = []
    for slot, start, audio in cycles:
        g = onair._MasterGrid(anchor, SLOT_N, 0, packet_n=P1_PACKET_N,
                              cs_n=P1_CS_N, d_max_n=D_MAX_N)
        # The turnaround the session held when it keyed the entry packet, taken
        # where every other cycle takes it: in PACTOR-1, off our 960 ms packet.
        g.d_n, g.d_ref_n = d_ms / 1e3 * rxfront.FS, g.packet_n
        g.keying(Protocol.PACTOR3)
        g.keyed_slot = slot

        at = g.rx_due_in(start, start + audio.size)
        held += at is not None
        onsets = [start + round(t * rxfront.FS)
                  for t in rxfront.p1_burst_onsets(audio)]
        heard += bool(onsets)
        off += [abs(o - g.rx_due(slot)) for o in onsets]
        read = (at is not None
                and _is_grant(p1rx.cs_anchored(audio, (at - start) / rxfront.FS)))
        clear += read
        both += read and bool(onsets)
        # ...against the instant this replaces, which is the same grid asked for
        # our own PACTOR-3 data end.
        stale = g.boundary(slot) + onair.P3_PACKET_N + g.d
        cs = p1rx.cs_anchored(audio, (stale - start) / rxfront.FS)
        stale_true += _is_grant(cs)
        stale_junk += cs is not None and not _is_grant(cs)

    n = len(cycles)
    check(f"{name}: every upgraded cycle's window holds the peer's codeword",
          held == n, f"{held} of {n}")
    check("...no burst those cycles carried sits outside the reader's reach of "
          "the carried anchor", off and max(off) <= reach,
          f"{len(off)} burst(s), worst {max(off, default=0) / 48:.0f} ms of "
          f"{reach / 48:.0f}")
    check("...and every cycle that carried one reads `0x59A` at zero errors "
          "there", both == heard, f"{both} of {heard}")
    check("...where our own PACTOR-3 data end reads the answer in none of them",
          stale_true == 0,
          f"{stale_true} of {n}, and {stale_junk} accept(s) there that are not "
          f"the answer")
    return n, clear, stale_junk


ENTRY_KEYED_N = 40266
"""Samples from key-up to the entry packet's last symbol -- 838.9 ms.

13.9 ms of transmit filter the 2 % trim leaves in front of the first symbol, and
82.5 symbols of keyed comb: the 82 the entry keys, plus the half the sub-band
lead adds to the tail because channel 12 starts half a symbol after channel 5
(`spec.SUBBAND_LEAD`). Banked rather than recomputed, so a renderer change moves
this number or fails."""


def _entry_extent_arm(check) -> None:
    """Where the entry packet stops, off the buffer the transmitter hands `_tx`.

    It decides where the ENTRY position sits, so the whole answer-position
    reading is only as good as it. Read through the trim `_tx` applies, which is
    why the stagger moves it twice over: the tail grows half a symbol and the
    quieter leading half-symbol -- one carrier where there used to be two --
    puts key-up later, so the keyed extent comes down while the packet grows.

    `onair.ENTRY_END_N` is the same quantity and wants to be `ENTRY_KEYED_N`. It
    is checked here to half a symbol rather than exactly, because that constant
    is not this module's to move and half a symbol is nothing against the
    121.1 ms that separates the two answer positions -- but the render itself is
    pinned to the sample.
    """
    ent = placement.link_packet(1, b"", 0x1a, swapped=False,
                               flush=placement.ENTRY_FLUSH)
    env = np.abs(ent)
    front = int(np.argmax(env > 0.02 * env.max()))
    # The pulse is symmetric, so its taps past the peak are what rings out behind
    # the last symbol; how many there are is `placement.PROTOCOL_RISE`'s to say.
    taps = placement.protocol_config().pulse()
    keyed = ent.size - (taps.size - 1 - int(taps.argmax())) - front
    check("the entry packet's last symbol falls where the render puts it",
          keyed == ENTRY_KEYED_N,
          f"{keyed} vs {ENTRY_KEYED_N} samples "
          f"({keyed / rxfront.FS * 1e3:.1f} ms after key-up)")
    check("...and `onair.ENTRY_END_N` is within half a symbol of it",
          abs(onair.ENTRY_END_N - keyed) < rxfront.FS // 200,
          f"constant {onair.ENTRY_END_N}, render {keyed}, "
          f"{(onair.ENTRY_END_N - keyed) / rxfront.FS * 1e3:+.1f} ms")
    check("...and it is shorter than the PACTOR-1 packet, which is what "
          "separates the two answer positions",
          0 < P1_PACKET_N - onair.ENTRY_END_N < SLOT_N,
          f"{(P1_PACKET_N - onair.ENTRY_END_N) / rxfront.FS * 1e3:.1f} ms apart")


def _answer_arm(name: str, check) -> tuple[int, int]:
    """FINDING: the answer position is what says the entry packet was not read.

    Three slots have taken this reading by hand off the `[grid] d` series and the
    `at anchor` tags and then written the same sentence. `answer_position` is the
    same reading taken once, in the cycle it happens: the peer's own burst,
    through the production onset detector, measured from the slot boundary the
    grid keyed on and compared with where the grid saw the same peer answer
    before an entry packet ever went out.

    AND THE NULL IS THE SAME BURSTS DISPLACED. A reading nothing could ever move
    is not a reading, so every cycle is put through twice: once where the peer
    actually keyed it, and once shifted by exactly the two packets' difference,
    which is where a peer that read the entry would have keyed it. The second
    pass has to come back with the opposite verdict off the same audio and the
    same detector.

    Returns `(cycles read, cycles NOT at the entry position)`.
    """
    got = _upgraded_cycles(name)
    if got is None:
        print(f"  SKIP -- no upgraded cycles under {name}")
        return 0, 0
    anchor, d_ms, cycles = got
    d_n = round(d_ms / 1e3 * rxfront.FS)
    split = P1_PACKET_N - onair.ENTRY_END_N

    def replay(shift: int) -> tuple[onair._MasterGrid, list[str]]:
        g = onair._MasterGrid(anchor, SLOT_N, 0, packet_n=P1_PACKET_N,
                              cs_n=P1_CS_N, d_max_n=D_MAX_N)
        g.d_n, g.d_ref_n = float(d_n), g.packet_n
        g.acquired = g.corroborated = True
        # The turnaround the session's log printed in the last PACTOR-1 cycle
        # before the entry went out, put back where it was measured: our own
        # 960 ms packet, that gap. The two positions are placed from it.
        g.update([g.boundary(cycles[0][0] - 1) + g.packet_n + d_n])
        g.keying(Protocol.PACTOR3)
        out = []
        for _, start, audio in cycles:
            g.update([start + round(t * rxfront.FS) - shift
                      for t in rxfront.p1_burst_onsets(audio)], audio, start)
            line = g.answer_position()
            if line is not None:
                out.append(line)
        return g, out

    bare = onair._MasterGrid(anchor, SLOT_N, 0, packet_n=P1_PACKET_N,
                             cs_n=P1_CS_N, d_max_n=D_MAX_N)
    check(f"{name}: a grid that keyed no entry packet says so rather than "
          "reporting a position",
          "NO ENTRY PACKET WAS KEYED" in bare.entry_verdict())
    g, lines = replay(0)
    entry = sum("AT THE ENTRY POSITION" in ln for ln in lines)
    check("...and no cycle the peer answered in sits where reading the entry "
          "would have moved it", lines and entry == 0,
          f"{len(lines) - entry} of {len(lines)} cycle(s) elsewhere")
    check("...so the run's account never says the peer read one",
          "MOVED TO THE ENTRY POSITION" not in g.entry_verdict(),
          g.entry_verdict()[:104])
    # ...and the same audio, moved to where a peer that DID read one answers.
    moved, shifted = replay(split)
    was_p1 = sum("AT THE PACTOR-1 POSITION" in ln for ln in lines)
    now_entry = sum("AT THE ENTRY POSITION" in ln for ln in shifted)
    check("...while the same bursts displaced by the two packets' difference "
          "read as exactly that, off the same audio and cycle for cycle",
          was_p1 and now_entry == was_p1,
          f"{now_entry} at the entry position where {was_p1} were at the "
          f"PACTOR-1 one")
    return len(lines), len(lines) - entry


def _grant_arm(path: Path, check) -> None:
    """What this station does with the recording's own upgrade grant."""
    grant = next((ev for ev in _events(path)
                  if ev.kind == "unassigned" and ev.spare == pactor1.CS_59A), None)
    check("the recording's grant reaches the FSM as the word it is",
          grant is not None,
          f"0x59A at t={grant.t:.3f} s" if grant is not None else "none decoded")
    if grant is None:
        return

    # DECLINING IS WHAT TAKES A FLAG, and this is what it buys: the word is
    # named in the log and the cycle goes on in PACTOR-1. It was the default
    # until 2026-08-27, which is why every grant this station was ever sent went
    # unanswered.
    host, keyed = _granted_station()
    host.p1_act_on_grant = False
    n = len(keyed)
    host.on_rx_event(grant)
    host.tick()
    check("...and a station that declines it changes nothing at all",
          host.protocol is Protocol.PACTOR1
          and [p for p, _ in keyed[n:]] == [Protocol.PACTOR1],
          f"{host.protocol}, keyed {[p.value for p, _ in keyed[n:]]}")

    # ...and by default the SLOT is the claim. `RadioTx._tx` is where a burst
    # meets the grid, so a packet handed to it inside the grant's own dispatch
    # goes out at the next boundary -- the slot the next PACTOR-1 packet would
    # have had. One turnaround after the grant, which is what both recordings
    # measure, and NOT one cycle later.
    host, keyed = _granted_station()
    n = len(keyed)
    host.on_rx_event(grant)
    check("the grant upgrades the link and keys the entry packet in the same "
          "dispatch, with no cycle in between",
          host.protocol is Protocol.PACTOR3 and len(keyed) == n + 1
          and keyed[-1][0] is Protocol.PACTOR3,
          f"{host.protocol}, keyed {[p.value for p, _ in keyed[n:]]}")
    host.tick()
    check("...and the cycle tick that ends the slot adds nothing behind it",
          len(keyed) == n + 1, f"{len(keyed) - n} bursts in the slot")

    entry = keyed[n][1]
    scan = p3rx.decode_p3_packets(entry)
    got = scan.packets[0] if scan.packets else None
    check("our own receiver acquires that packet from a standing start and "
          "reads speed level 1",
          got is not None and got.sl == 1,
          got.report() if got is not None else "nothing decoded")
    check("...with the field the reference entry packet carries and no user "
          "data at all -- see `_entry_arm`, which puts the two side by side",
          got is not None and got.payload == b"" and got.data_type == 6,
          got.report() if got is not None else "")
    energy = p3rx.channel_energy(entry)
    lit = [c for c, e in enumerate(energy)
           if e > p3rx.MIN_TONE_EXCESS and e > p3rx.TONE_REL * energy.max()]
    # The two carriers CARRY it; the neighbours are the transmit pulse's own
    # skirt, and 120 Hz spacing under a 100 Bd pulse means there is always some.
    # The claim is a margin, not an empty slot: `placement.PROTOCOL_RISE` keys the
    # protocol's own pulse and lights 11 and 13 at -9.6 dB, where the reference
    # entry of `PIII_Complete_1` lights 4, 5, 6, 11, 12 and 13 with its own
    # neighbours only 2.1 to 4.3 dB down on this same instrument.
    rest = max((e for c, e in enumerate(energy) if c not in (5, 12)), default=0.0)
    check("...off tones 5 and 12, the pair every speed level shares and the "
          "reason a PACTOR-1 receiver can acquire it",
          set(lit) >= {5, 12} and rest < min(energy[5], energy[12]) / 4
          and p3rx.levels_present(energy)[0] == 1,
          f"channels {lit}, neighbours "
          f"{10 * np.log10(rest / min(energy[5], energy[12])):.1f} dB down, "
          f"levels {p3rx.levels_present(energy)}")

    # The entry packet is the entry and nothing more: once the peer has answered
    # it the waveform is acquired, and the traffic runs at `entry_sl`.
    ci = CS_ACK if host.arq.tx_seq % 2 == 0 else CS_REQUEST
    host.on_rx_event(rxfront.Event(0.0, "cs", "", protocol="PACTOR-3", cs=ci))
    check("an acknowledged entry packet takes the link to entry_sl",
          host.arq.speed_level == host.arq.cfg.entry_sl,
          f"SL{host.arq.speed_level}")

    # ...AND THE ANSWER THESE RECORDINGS CARRY IS A BREAK-IN, which `_ack_arm`
    # measures off the audio: the IRS reads the entry packet and takes the
    # channel to send its own greeting instead of keying a codeword. That is
    # still the entry packet's acknowledgement -- `arq.PactorArq._yield_link`
    # settles the packet in flight on one -- so everything the acknowledgement
    # above buys has to survive the handover. It did not: the traffic level was
    # spent only on the `_on_ack` path, and a granted link answered this way ran
    # its whole PACTOR-3 phase at speed level 1, five bytes a cycle.
    host, keyed = _granted_station()
    host.on_rx_event(grant)
    n = len(keyed)
    host.on_rx_event(rxfront.Event(0.0, "cs", "", protocol="PACTOR-3",
                                   cs=CS_BREAKIN))
    check("a break-in answering the entry packet hands the channel over, with "
          "the link up and still in PACTOR-3",
          host.arq.role == IRS and host.arq.state == State.CONNECTED
          and host.protocol is Protocol.PACTOR3,
          f"{host.protocol} {host.arq.role} {host.arq.state}")
    check("...and answers the upgrade, so nothing counts it unfollowed",
          not host.arq.upgrade_unanswered)
    check("...and takes the traffic to entry_sl, as an acknowledgement does",
          host.arq.speed_level == host.arq.cfg.entry_sl,
          f"SL{host.arq.speed_level}")
    host.tick()
    check("...and the cycle behind it keys nothing over the station we just "
          "gave the channel to",
          [pr for pr, _ in keyed[n:]] == [], f"{len(keyed) - n} bursts")

    # ...AND TAKING THE CHANNEL BACK NO LONGER ENDS THE PACTOR-3 PHASE. Under
    # this flag the grant is the ONLY door into PACTOR-3 and `_grant_taken` has
    # closed it, so the fall-back `breakin_now` used to make -- there being no
    # PACTOR-3 changeover packet to key -- left the session in PACTOR-1 from the
    # first reversal to the end of it. This is that reversal, driven through the
    # transmit path, and what comes out is read back by the same reader that
    # reads DL6MAA's.
    host.arq.on_host_breakin()
    # "The IRS sends it after a correctly received packet" -- the CS3 replaces
    # that packet's acknowledgement, so the break-in waits for one to decode.
    host.on_rx_event(rxfront.Event(0.0, "packet", "", protocol=Protocol.PACTOR3,
                                   packet=(3, spec.status_byte(
                                       host.arq._expected_seq), b"HI", True)))
    n = len(keyed)
    host.tick()
    check("this station takes the channel back without leaving PACTOR-3",
          host.protocol is Protocol.PACTOR3
          and [pr for pr, _ in keyed[n:]] == [Protocol.PACTOR3],
          f"{host.protocol}, keyed {[pr.value for pr, _ in keyed[n:]]}")
    if len(keyed) > n:
        rf = keyed[-1][1]
        head = next(iter(_cs3_heads(rf)), None)
        field, ok = ((b"", False) if head is None
                     else p3rx.decode_changeover(rf, head))
        check("...keying a changeover packet the reader that reads DL6MAA's "
              "reads, under the counter DL6MAA's carry",
              ok and field[3] & spec.STATUS_SEQ == 0,
              field.hex() if ok else "no CS3 head" if head is None
              else "head, but no field")


DL6MAA_ENTRY_FIELD = bytes.fromhex("0f8f87c7c31a6689")
"""The only entry packet in recorded history a peer is known to have read.

The walking template truncated to five bytes, a status byte at counter 2 and data
type 6, the CRC (pactor3.md §14, §17.1). It is written down here and READ BACK
below off `PIII_Complete_1`, which is where it comes from; n = 1, and there is no
second recording of a PACTOR-III entry anywhere in the corpus."""


def _cells_off_air(audio: np.ndarray, grid0: int, expect: dict) -> int:
    """Cells of the recording's entry packet that disagree with `expect`.

    The one measurement in this file that touches no decoder of ours. Case 0
    neither punctures nor interleaves, so each of its 144 coded bits is one
    differential step on one of two carriers and can be read straight off the
    audio. `expect` is what a render keys, per home channel and header block
    first; the residual carrier offset is a rotation common to every step, so it
    is estimated from the packet itself rather than searched for.

    The symbol clock IS searched, because the grid row a decode reports and the
    sample a matched filter peaks at answer to different indexing conventions and
    being one sample out would score a perfect render as a wrong one.
    """
    sps = rxfront.FS // 100
    pulse = _rx._pulse(sps)
    bb = {cn: _rx._baseband(audio, cn, rxfront.FS, pulse)
          for cn in placement.DETECT.tones}
    best = None
    for dt in range(-sps, sps + 1):
        bad = 0
        for cn, z in bb.items():
            at = grid0 + dt + (len(pulse) - 1) // 2 + (np.arange(81) - 9) * sps
            sym = z[at]
            step = np.angle(sym[1:] * sym[:-1].conj())
            res = np.exp(1j * (step - expect[cn]))
            res = res * np.exp(-1j * np.angle(res.sum()))
            bad += int((np.abs(np.angle(res)) > np.pi / 3).sum())
        best = bad if best is None else min(best, bad)
    return best


def _entry_steps(flush) -> dict:
    """Per-carrier differential steps of the entry packet this station keys."""
    info = placement.field_info(b"", placement.DETECT.crc_bytes - 3, 0x1A)
    cells = placement.case0_cells(info, placement.DETECT, flush)
    grid = placement.grid_steps(
        cells.reshape(placement.FRAME_SYMBOLS, 2, 1),
        dataclasses.replace(placement.DETECT, tones=p3frame.VH_ORDER))
    head = p3frame.header_steps(
        p3frame.variable_header(1, swapped=False, long_cycle=False),
        placement.DETECT.tones)
    return {cn: np.concatenate([head[cn], grid[cn]]) for cn in placement.DETECT.tones}


def _trailer_off_air(audio: np.ndarray, grid0: int) -> dict:
    """The symbols around the packet's end, read bare off the audio.

    Per home channel: the trailer symbol's level and the one after it, in dB
    against the packet body's median, and the differential step onto the
    trailer in degrees. Symbol 81 is one past the last data row and no part of
    the coded field; nothing here runs through a decoder of ours.

    EACH CARRIER ON ITS OWN CLOCK. Speed level 1 splits its two by half a symbol
    (`spec.SUBBAND_LEAD`), so a single clock reads the late carrier's symbol 82
    half inside its symbol 81 and reports the packet running one symbol longer
    than it does -- ch12 at -6 dB where it is gone, on the recording and on our
    own render alike.
    """
    sps = rxfront.FS // 100
    pulse = _rx._pulse(sps)
    offsets = dict(zip(placement.DETECT.tones,
                       placement.DETECT.clock_offsets(sps)))
    out = {}
    for cn in placement.DETECT.tones:
        z = _rx._baseband(audio, cn, rxfront.FS, pulse)
        at = (grid0 + offsets[cn] + (len(pulse) - 1) // 2
              + (np.arange(84) - 9) * sps)
        sym = z[np.clip(at, 0, len(z) - 1)]
        body = np.median(np.abs(sym[5:76]))
        db = lambda k: 20 * np.log10(np.abs(sym[k]) / body)
        step = np.degrees(np.angle(sym[81] * np.conj(sym[80])))
        out[cn] = (db(81), db(82), step)
    return out


def _entry_arm(path: Path, check) -> None:
    """FINDING: our entry packet was a different packet from the one that works.

    Through the 2026-08-26 slot this station answered `0x59A` with a valid speed
    level 1 packet whose field was five bytes of user text at data type 0 --
    `39 53 53 4a 20 02 74 f8` off the witness, `"9SSJ "` and a status byte. Four
    gateways across three bands granted the upgrade and then repeated the grant
    thirteen to seventeen times apiece, and §7 says that word is what a receiver
    sends both when nothing arrived AND when it will not take what did. So the
    air could not separate the two, and the packet went unexamined for a week.

    The recording can separate them, because it holds the entry packet a real
    IRS answered in the turnaround. It is not a data frame: it is the firmware's
    own idle field, the template, at the data type its traffic declares. An
    independent monitor reports such a field as `LEN: 0`.

    Read off the audio through the production decoder, and compared with what the
    production transmit path keys off a real grant -- not with a round trip
    through our own encoder, which is the failure mode this whole arm is about.
    """
    theirs = next((ev for ev in _events(path)
                   if ev.kind == "packet" and ev.packet[0] == 1), None)
    check("the recording carries the entry packet the session's IRS answered",
          theirs is not None,
          f"at t={theirs.t:.4f} s" if theirs is not None else "none decoded")
    if theirs is None:
        return
    sl, status, payload, ok = theirs.packet
    check("...and its field is the template and nothing else -- no user data, "
          "at the data type the same station's traffic declares",
          (sl, status, payload, ok) == (1, 0x1A, b"", True),
          f"SL{sl} status=0x{status:02x} type={(status >> 2) & 7} "
          f"payload={payload!r}")
    ours = placement.build_field(placement.field_info(b"", 5, status),
                                 placement.DETECT)
    check("...which is the field this station now builds for that counter, byte "
          "for byte", ours == DL6MAA_ENTRY_FIELD, ours.hex(" "))

    # AND THE FIELD IS NOT THE WHOLE PACKET. Everything above compares bytes;
    # what goes on the air is 144 coded cells, and the last sixteen of them are
    # the trellis flush, which no field byte and no CRC reaches. Read off the
    # audio against what each render keys -- `placement.ENTRY_FLUSH`.
    audio = session.load_wav(str(path), rxfront.FS)
    grid0 = round(theirs.t * rxfront.FS)
    keyed = _cells_off_air(audio, grid0, _entry_steps(placement.ENTRY_FLUSH))
    zeroed = _cells_off_air(audio, grid0, _entry_steps(None))
    check("the packet this station keys is the packet on the air, cell for cell "
          "-- all 144, both carriers, no decoder of ours in the way",
          keyed == 0, f"{keyed} of 144 differ")
    check("...and the zero flush every other path sends is not: it misses only "
          "the sixteen cells of the flush, and misses most of them",
          zeroed > keyed, f"zero flush {zeroed} of 144, flushed {keyed}")

    # AND THE CELLS ARE NOT THE WHOLE PACKET EITHER. The real station keys one
    # further symbol past the last data row -- 82 in all -- and the step onto
    # it is modulation, not a transmitter's key-down ramp, which would hold the
    # previous phase. `placement.ENTRY_TRAILER` is this measurement.
    theirs_tail = _trailer_off_air(audio, grid0)
    check("the air does not stop at the last data row: symbol 81 stands at "
          "body level on both carriers and symbol 82 is gone",
          all(t81 > -3.0 and t82 < -10.0 for t81, t82, _ in
              theirs_tail.values()),
          "; ".join(f"ch{cn} {t81:+.1f}/{t82:+.1f} dB re body"
                    for cn, (t81, t82, _) in theirs_tail.items()))
    want = np.degrees(placement.ENTRY_TRAILER)
    check("...and the step onto it is the -45 degrees the render keys, on both",
          all(abs(st - want) < 25.0 for _, _, st in theirs_tail.values()),
          "; ".join(f"ch{cn} {st:+.1f} deg"
                    for cn, (_, _, st) in theirs_tail.items()))

    grant = next((ev for ev in _events(path)
                  if ev.kind == "unassigned" and ev.spare == pactor1.CS_59A), None)
    if grant is None:
        return
    host, keyed = _granted_station()
    held = len(host.arq._outbuf)
    host.on_rx_event(grant)
    rf = keyed[-1][1]
    got = next(iter(p3rx.decode_p3_packets(rf).packets), None)
    check("and what the transmit path keys off that grant reads back as the "
          "same three things through the same decoder",
          got is not None and (got.sl, got.status, got.payload) == (1, 0x1A, b""),
          got.report() if got is not None else "nothing decoded")
    check("...with the user data still queued behind it and the announcement the "
          "grant acknowledged settled, so nothing is offered to a station that "
          "has not acquired the waveform yet",
          host.arq._outbuf.endswith(MESSAGE) and len(host.arq._outbuf) == held,
          f"{len(host.arq._outbuf)} B against {held} queued")
    if got is not None:
        ours_tail = _trailer_off_air(rf, got.start)
        check("and what it keys ends the way the recording ends: the same "
              "trailer symbol at body level, the same step, then silence",
              all(t81 > -3.0 and t82 < -10.0 and abs(st - want) < 25.0
                  for t81, t82, st in ours_tail.values()),
              "; ".join(f"ch{cn} {t81:+.1f}/{t82:+.1f} dB, {st:+.1f} deg"
                        for cn, (t81, t82, st) in ours_tail.items()))

    # THE ARRANGEMENT IS PINNED, and it was not. `RadioTx.send_packet` took the
    # carrier swap from the same per-cycle parity as the PACTOR-1 shift, so the
    # four entry packets of a grant alternated: variable header 0 in one cycle,
    # header 1 with the carriers exchanged in the next. DL6MAA keyed one entry,
    # unswapped, at header 0, read at 0.977 over its 32 bits.
    sps = rxfront.FS // 100
    pulse = _rx._pulse(sps)
    Z = {cn: _rx._baseband(rf, cn, rxfront.FS, pulse) for cn in spec.VH_CHANNELS}
    heads = p3rx.vh_anchors(Z, len(rf))
    check("...on variable header 0, unswapped, over all thirty-two of its bits "
          "and on both carriers",
          bool(heads) and heads[0].vh == 0 and not heads[0].swapped,
          f"vh {heads[0].vh} at {heads[0].fit:.3f}" if heads else "no anchor")

    # ...AND THE GRANT'S REMAINING CYCLES ARE SPENDABLE. A granted peer asks 13
    # to 17 times; one entry packet needs four of those to show it is not being
    # taken, and the rest are the only place an alternative can ever be tried.
    host, keyed = _granted_station()
    host.arq.cfg.entry_ladder = ("template", "data")
    host.on_rx_event(grant)
    n = len(keyed)
    for _ in range(8):
        host.on_rx_event(grant)
        host.tick()
    entries = [a for _, a in keyed[n - 1:] if len(a) > round(0.5 * rxfront.FS)]
    check("a peer that goes on asking gets the next entry on the ladder, not a "
          "fall-back at four cycles",
          len(entries) >= 8 and len({a.tobytes() for a in entries}) > 1,
          f"{len(entries)} entry packets, "
          f"{len({a.tobytes() for a in entries})} distinct")


@functools.cache
def _peer_answers(path: Path) -> tuple:
    """What the recording's IRS says back to the entry packet, in order.

    Both halves, and neither is written here. The first is the changeover packet
    whose head is the CS3 in the turnaround, built by the `rxfront._cs_event` a
    live receiver builds one with; the second is the greeting a cycle behind it,
    taken from the blind pass exactly as `_events` hands it to a station on the
    air. Every status byte below is DL6MAA's own.

    AND THE TWO ARE NOT EQUALLY EASY TO HOLD. `decode_events` finds the second
    and not the first: a changeover packet's twenty-symbol head has to be aimed
    at, and a file replay has nothing to aim with. That is why the arm drives the
    second on its own as well as the pair -- on the air it is the half a receiver
    is likelier to be left with.
    """
    audio = session.load_wav(str(path), rxfront.FS)
    entry = next(ev for ev in _events(path)
                 if ev.kind == "packet" and ev.packet[0] == 1)
    head = next((at for at in _cs3_heads(audio)
                 if at / rxfront.FS > entry.t), None)
    out = [] if head is None else [
        rxfront._cs_event(audio, CS_BREAKIN, 0, head, head / rxfront.FS, "")]
    out += [ev for ev in _events(path)
            if ev.kind == "packet" and entry.t < ev.t < entry.t + 3.0]
    return tuple(sorted((ev for ev in out if ev.kind == "packet"),
                        key=lambda e: e.t))


def _drive_grant(path: Path, answers) -> tuple:
    """A granted station, its entry packet keyed, then handed `answers`.

    Returns the host, the bursts it put on the air after the entry packet, and
    what reached the host data port.
    """
    grant = next(ev for ev in _events(path)
                 if ev.kind == "unassigned" and ev.spare == pactor1.CS_59A)
    host, keyed = _granted_station()
    host.on_rx_event(grant)
    n = len(keyed)
    rx = host.channel(host.ptchn).rx
    before = len(rx)
    for ev in answers:
        host.on_rx_event(ev)
    host.tick()
    return host, keyed[n:], len(rx) - before


def _sequence_arm(path: Path, check) -> None:
    """FINDING: answered the way the reference answers one, a good entry packet
    still read to this station as a link that had failed.

    The four things `PIII_Complete_1` does that shrike expected something else
    of, driven through the FSM off the recording's own bytes. The entry packet
    itself is `_entry_arm`'s, and it is the packet DL6MAA keyed cell for cell;
    what happens in the four seconds after it is here.

    n = 1 ON THE SEQUENCE, and the arm is built so that it says so. Every claim
    below is either read off the audio or is about what this station does with
    what was read -- none of them is a timing this station now imitates. The
    recording's caller spends 7.5 s on empty template packets before its first
    payload byte, and nothing here reproduces that: the same recording's IRS puts
    212 bytes in the FIRST packet it keys after taking the channel, so the quiet
    is a station with an empty buffer and not a thing the protocol asks for.
    """
    answers = _peer_answers(path)
    check("the recording answers the entry packet with a break-in and then a "
          "data packet of its own",
          [ev.breakin for ev in answers] == [True, False],
          ", ".join(f"{ev.t:.4f} "
                    f"{'CS3+frame' if ev.breakin else 'data'}" for ev in answers))
    if [ev.breakin for ev in answers] != [True, False]:
        return
    greeting = answers[1]
    sl, status, payload, _ = greeting.packet
    check("...which resumes at a WIDE level and carries traffic in that very "
          "first packet, so there is no quiet for this station to imitate",
          sl >= 5 and len(payload) > 200,
          f"SL{sl}, {len(payload)} B at t={greeting.t:.4f} s")
    check("...and asks for the channel back on it, with nothing behind it to "
          "ask again -- the request is answered in that cycle or not at all",
          bool(status & spec.STATUS_CHANGEOVER)
          and not any(ev.t > greeting.t for ev in _events(path)
                      if ev.kind == "packet" and ev.t < greeting.t + 2.0),
          f"status=0x{status:02x}")

    # THE PAIR. What a station keying that entry packet would have held if it
    # held everything, which is the case the model is written from.
    host, keyed, got = _drive_grant(path, answers)
    check("a station whose entry packet is answered that way ends the cycle as "
          "the SENDER again, in PACTOR-3, at entry_sl",
          host.protocol is Protocol.PACTOR3 and host.arq.role == arq_mod.ISS
          and host.arq.speed_level == host.arq.cfg.entry_sl
          and not host.arq.entry_pending,
          f"{host.protocol} {host.arq.role} SL{host.arq.speed_level} "
          f"entry_pending={host.arq.entry_pending}")
    check("...with the greeting delivered to the host rather than graded on a "
          "role or a counter", got > 0, f"{got} B")
    # TWO BURSTS, AND THE RECORDING'S CALLER KEYS THE SAME TWO. It answers the
    # changeover packet with a codeword of its own -- CS5 at 6.5194 s, having
    # failed the frame -- and then takes the channel at 7.7681 s with the CS3
    # that replaces the greeting's acknowledgement. What is NOT in between is a
    # second codeword: the request rides the packet it arrived on and is answered
    # in that cycle, not held for another packet to license it.
    check("...having keyed the two bursts the recording's caller keys: one "
          "codeword for the changeover packet, then the changeover packet back",
          len(keyed) == 2, f"{len(keyed)} bursts")
    if len(keyed) == 2:
        rf = keyed[-1][1]
        head = next(iter(_cs3_heads(rf)), None)
        field, ok = ((b"", False) if head is None
                     else p3rx.decode_changeover(rf, head))
        # DL6MAA's own two changeover packets carry counter 0 and are followed
        # by counter 1 -- `placement.CHANGEOVER`, measured in both directions.
        # So the numbering this station takes the link back under is already the
        # recording's, and nothing here moves it.
        check("...under the counter DL6MAA's changeover packets carry",
              ok and field[3] & spec.STATUS_SEQ == 0,
              field.hex(" ") if ok else "no changeover packet keyed")

    # THE HALF A RECEIVER IS LIKELIER TO HOLD, on its own. The head is twenty
    # symbols at the front of an 0.81 s burst; the greeting is a whole cycle.
    host, keyed, got = _drive_grant(path, answers[1:])
    check("and the greeting ALONE answers the entry packet, with the CS3 head "
          "missed: the peer transmitting a packet of its own is the answer",
          host.protocol is Protocol.PACTOR3 and host.arq.role == arq_mod.ISS
          and host.arq.speed_level == host.arq.cfg.entry_sl
          and not host.arq.entry_pending and got > 0,
          f"{host.protocol} {host.arq.role} SL{host.arq.speed_level} "
          f"entry_pending={host.arq.entry_pending} {got} B")
    check("...and nothing in that window counts the upgrade unfollowed",
          not host.arq.upgrade_unanswered)

    # THE EMPTY FIELD'S DATA TYPE, off the recording and then off this station.
    audio = session.load_wav(str(path), rxfront.FS)
    empty = [p for p in p3rx.decode_p3_packets(audio).packets if not p.payload]
    types = {(p.status >> 2) & 7 for p in empty}
    check("every field the reference writes the template into declares one data "
          "type, across every speed level it keys one at and in all three runs",
          len(empty) >= 15 and types == {spec.DataType.PMC_ENGLISH},
          f"{len(empty)} empty fields at SL{sorted({p.sl for p in empty})}, "
          f"types {sorted(types)}")
    host, _ = upgraded_station()
    host.arq._outbuf.clear()
    host.arq._inflight = None
    host.tick()
    idle = host.arq._inflight
    check("...and so does the one this station writes when it has nothing to "
          "say, which is the same field its entry packet carries",
          idle is not None and not idle.payload
          and (idle.status >> 2) & 7 == spec.DataType.PMC_ENGLISH,
          f"status=0x{idle.status:02x}" if idle is not None else "no idle packet")


def _inert_arm(check) -> None:
    """The grant is one word, one direction, one moment, and once."""
    host, keyed = calling_station()
    host.on_rx_event(_grant_event(pactor1.CS_6A9))
    host.tick()
    check("the other unassigned word drives nothing, armed or not",
          host.protocol is Protocol.PACTOR1 and keyed.p3 == [],
          f"{host.protocol}, {len(keyed.p3)} PACTOR-3 packets")

    answering = PtcHost(peer=Keyed(), mycall="W9SSJ")
    answering.arq.on_host_listen(True)
    answering.arq.on_rx_connect(DXCALL, "W9SSJ")
    answering.on_rx_event(_grant_event())
    check("a station that ANSWERED the call does not take a grant",
          answering.protocol is Protocol.PACTOR1, str(answering.protocol))

    host, keyed = upgraded_station()
    sl = host.arq.speed_level
    host.on_rx_event(_grant_event())
    check("a grant arriving on a link already in PACTOR-3 does not re-enter it",
          host.arq.speed_level == sl, f"SL{host.arq.speed_level} was SL{sl}")

    # ONE GRANT PER LINK, and the target being ruled out is not what stops the
    # second: a peer sends the grant once per cycle for as long as it is waiting,
    # and a link that re-entered PACTOR-3 on each of them would key an entry
    # packet a cycle for the rest of the session.
    host, keyed = calling_station()
    host.on_rx_event(_grant_event())
    host.fall_back("the arm ends the window here")
    host._ruled_out.clear()
    host.on_rx_event(_grant_event())
    check("and the link takes one grant, not one per cycle of them",
          host.protocol is Protocol.PACTOR1, str(host.protocol))


def _ladder_arm(check) -> None:
    """What the ladder may hold, which was prose while the prose armed it.

    `burst` carried "MUST NOT BE RUN AS IT STANDS" in capitals in the docstring
    of the field that had it in the default -- so every arm flown between the
    measurement and 2026-08-27 keyed the rung its own documentation forbade, and
    nothing here noticed. The bound is data now (`arq.ENTRY_RUNGS_GROUNDED`) and
    it is refused at both places a ladder is set: the config that carries it and
    the command line that overrides it.
    """
    default = arq_mod.ArqConfig().entry_ladder
    check("the shipped ladder holds nothing measured unflyable",
          not set(default) & set(arq_mod.ENTRY_RUNGS_GROUNDED), str(default))

    for build, why in ((lambda: arq_mod.ArqConfig(entry_ladder=("template", "burst")),
                        "the config"),
                       (lambda: onair._entry_ladder("template,burst"),
                        "the command line")):
        try:
            build()
        except Exception as exc:                        # noqa: BLE001
            named = "1.074" in str(exc) and "one slot in two" in str(exc)
        else:
            named = False
        check(f"...and {why} refuses one that does, with the arithmetic in the "
              "message", named)

    check("a rung that is merely untaken is still flyable -- the refusal is "
          "about a measurement, not about the vocabulary",
          onair._entry_ladder("template,data") == ("template", "data"))


def main() -> int:
    ok = True

    def check(claim: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {claim}"
              + (f" -- {detail}" if detail else ""))

    print("\nthe grant is one word, one direction, one moment, and once:")
    _inert_arm(check)
    print("\nwhat the entry ladder may hold:")
    _ladder_arm(check)

    for path in (CAPTURE, FULL):
        if not path.exists():
            print(f"  SKIP -- {path} is not here")
            continue
        print(f"\n{path.name}:")
        _arm(path, check)
        _ack_arm(path, check)
        _window_arm(path, check)
        _changeover_arm(path, check)
        _rotation_arm(path, check)
    print("\nwhere our own entry packet ends:")
    _entry_extent_arm(check)
    print("\nthe peer answers on the cycle, not after our packet:")
    tally = [_raster_arm(name, check) for name in UPGRADED]
    n, clear, junk = (sum(c) for c in zip(*tally)) if tally else (0, 0, 0)
    if n:
        check("over all four arms the carried anchor reads the answer, and our "
              "own PACTOR-3 data end reads nothing that is one",
              clear > junk,
              f"{clear} of {n} cycles against {junk} accept there, and that one "
              f"is a two-error CS3 -- the codeword the grid REVERSES on")

    print("\nand the answer POSITION says the entry packet was not read:")
    seen = [_answer_arm(name, check) for name in UPGRADED]
    read, elsewhere = (sum(c) for c in zip(*seen)) if seen else (0, 0)
    if read:
        check("over all four arms not one answer to an entry packet sits where "
              "reading one would have put it",
              elsewhere == read, f"{elsewhere} of {read} cycles")

    for path in GRANTS:
        if not path.exists():
            continue
        print(f"\nthe upgrade grant in {path.name}:")
        _grant_arm(path, check)
    if FULL.exists():
        print(f"\nthe announcement {FULL.name} opens with:")
        _announcement_arm(FULL, check)
        print(f"\nthe entry packet, beside the one in {FULL.name}:")
        _entry_arm(FULL, check)
        print(f"\nthe upgrade sequence, driven by {FULL.name}'s own answers:")
        _sequence_arm(FULL, check)

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
