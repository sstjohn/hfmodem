# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two shrike stations holding a QSO, with every burst crossing as audio.

`SimPeer` hands decoded events straight from one state machine to the other, so it
proves the ARQ logic and says nothing about the waveforms. This carries every
burst the way the air does: the sender renders it to samples, and the receiver
recovers it with `rxfront` and `p1rx` -- the same decoders the monitor and the
on-air loop run. A message only arrives if it was genuinely modulated and
demodulated.

The link UPGRADES, which is the thing this rehearsal now exists to catch. It
opens in PACTOR-1 because a connect burst is PACTOR-1, carries its callsign
announcement there, and moves to PACTOR-3 on the cycle that announcement is
acknowledged -- so both waveforms and the seam between them are exercised in one
session. That seam had never been run: the loopback used to render every
PACTOR-3 burst as PACTOR-1, or every PACTOR-1 burst as PACTOR-3, depending on
which side of it a defect sat, and passed either way.

That is the rehearsal for the thing shrike exists to do. A station that cannot
hold a multi-cycle exchange with itself over its own waveforms will not hold one
with an SCS modem, and this asks the question without a radio or a stranger's
gateway. What it cannot answer is whether an SCS modem agrees with our reading of
the protocol -- both ends here share our assumptions, and any misreading is
symmetric and therefore invisible. Only an independent decoder can answer that
one; this is the regression gate.

Run: python -m hfmodem.tests.shrike.test_qso
"""
from __future__ import annotations


import numpy as np

from hfmodem.shrike import pactor1, placement, rxfront  # noqa: E402
from hfmodem.shrike.arq import IRS, ISS, State  # noqa: E402
from hfmodem.shrike.ptc import PtcHost  # noqa: E402
from hfmodem.shrike.spec import Protocol  # noqa: E402

FS = rxfront.FS


def _p1_packet(heard: np.ndarray) -> "rxfront.Event | None":
    """The PACTOR-1 frame in this burst, as the event the FSM consumes.

    `rxfront.decode_events` reaches the same decoder, but only once a connect has
    opened a session in the SAME buffer -- the scan is far too costly to run on
    every burst in the band. Here each burst arrives alone, in the slot where a
    frame is due, which is the situation that gate exists to exclude, so the
    in-session entry point is asked directly. It is the receiver, not a stand-in.
    """
    return rxfront.decode_expected_p1_packet(heard)


class AudioSide:
    """One station's transmit seam: render each burst and hand it to the link."""

    def __init__(self, name: str):
        self.name = name
        self.pending: list[tuple[str, np.ndarray]] = []
        # ...and the label of every burst, kept after `_carry` has popped it.
        # What went out is the only place a cycle length is visible off-air.
        self.sent: list[str] = []
        self.host: PtcHost | None = None
        # "Mit jedem neuen Paket oder Kontrollsignal wird die Shiftlage
        # invertiert", the same alternation `onair.RadioTx` keys on the air, and
        # off-air it holds without exception. Carried here so the rehearsal makes
        # our own receiver read both shift positions rather than only the one.
        self.shift_inverted = False

    def attach(self, host: PtcHost) -> None:
        self.host = host

    def _flip(self) -> bool:
        inv = self.shift_inverted
        self.shift_inverted = not inv
        return inv

    def connect_burst(self, mycall: str, dxcall: str) -> None:
        # Every connect goes out in the same position, and then the toggle moves
        # on -- see RadioTx.connect_burst for why both halves of that matter.
        self.shift_inverted = False
        self._tx(f"connect->{dxcall}",
                 pactor1.connect_signal(dxcall, invert=self._flip()))

    def send_cs(self, cs_index: int) -> None:
        # THE PACTOR-3 CONTROL SIGNAL, because this seam is only reached once the
        # link has been upgraded -- `PtcHost.send_cs` routes a PACTOR-1 link to
        # `send_p1_cs` below. It rendered PACTOR-1 here, so an upgraded link
        # answered on 1400/1600 Hz and the far end, listening for twenty DBPSK
        # symbols on tones 5 and 12, heard nothing at all.
        self._tx(f"CS{cs_index + 1}", placement.control_signal(cs_index))

    def send_p1_cs(self, index: int) -> None:
        self._tx(f"P1 CS{index + 1}",
                 pactor1.control_signal(index, invert=self._flip()))

    def send_p1_packet(self, payload: bytes, baud: int, packet_count: int, *,
                       header: int | None = None,
                       changeover_request: bool = False,
                       qrt: bool = False) -> None:
        # THE PATH THE RADIO TAKES. A link that has not been upgraded is
        # PACTOR-1, so `PtcHost.send_packet` routes here -- but only if this seam
        # offers the method, and it did not. So the rehearsal built its data
        # packets out of `placement.data_packet`, a PACTOR-3 frame, and passed
        # while the waveform shrike actually transmits on the air went untested.
        flags = ("" if not changeover_request else " BK") + ("" if not qrt else " QRT")
        self._tx(f"P1 pkt#{packet_count} {len(payload)}B{flags}",
                 pactor1.packet_signal(payload, baud, packet_count=packet_count,
                                       header=header,
                                       changeover_request=changeover_request,
                                       qrt=qrt, invert=self._flip()))

    def send_p1_breakin(self, payload: bytes, baud: int, packet_count: int, *,
                        qrt: bool = False) -> None:
        self._tx(f"P1 BREAK-IN #{packet_count} {len(payload)}B",
                 pactor1.breakin_signal(payload, baud, packet_count=packet_count,
                                        qrt=qrt, invert=self._flip()))

    def send_packet(self, sl: int, payload: bytes, status: int,
                    breakin: bool = False) -> int:
        # An UPGRADED link's data frame, through `placement.link_packet` -- the
        # same call `onair.RadioTx` keys on the air, including the carrier swap
        # this ARQ cycle puts the virtual carriers on. It used to build its own
        # field at speed level 2's geometry whatever level was asked for, so the
        # rehearsal passed over a waveform no radio transmits.
        #
        # ...and `changeover_packet` for a break-in, for exactly that reason: this
        # branch used to ignore `breakin` and render an ordinary packet, which is
        # a station seizing a channel with a waveform that says nothing about
        # seizing it. The far end read it as traffic and went on transmitting.
        swapped = self._flip()
        if breakin:
            self._tx(f"P3 BREAK-IN {len(payload)}B -> SL{sl}",
                     placement.changeover_packet(payload, status, swapped=swapped))
            return placement.CHANGEOVER.crc_bytes - 3
        self._tx(f"SL{sl} pkt {len(payload)}B",
                 placement.link_packet(sl, payload, status, swapped=swapped))
        return placement.SPEED_PATHS[sl].crc_bytes - 3

    def send_long_packet(self, sl: int, payload: bytes, status: int) -> int:
        # `onair.RadioTx.send_long_packet`'s render, and the absence of this
        # method was a silent 217-byte hole: `PtcHost.send_packet` routes a long
        # cycle here only if the seam offers it, falls through to the short
        # renderer otherwise, and `link_packet` cuts a 276-byte field to 59 with
        # no error. The ARQ settled all 276. Reproduced 3 of 3 on a 522-byte
        # compressed message, and the B2F session died on framing several blocks
        # later.
        self._tx(f"SL{sl} LONG pkt {len(payload)}B",
                 placement.link_packet(sl, payload, status,
                                       swapped=self._flip(), long_cycle=True))
        return placement.LONG_PATHS[sl].crc_bytes - 3

    def pump(self) -> None:
        pass

    def cycle(self) -> None:
        pass

    def _tx(self, what: str, audio: np.ndarray) -> None:
        self.sent.append(what)
        self.pending.append((what, np.asarray(audio, np.float32)))


class AudioLink:
    """Move bursts between two stations, decoding each with the real receiver."""

    SIDE = AudioSide
    """The transmit seam to build each station on. `perf/shrike-rtf.py` swaps in a
    timed subclass; nothing else has any business changing it."""

    def __init__(self, a_call: str, b_call: str, *, snr_db: float | None = None,
                 verbose: bool = True):
        self.snr_db, self.verbose = snr_db, verbose
        self.a_io, self.b_io = self.SIDE(a_call), self.SIDE(b_call)
        self.a = PtcHost(peer=self.a_io, mycall=a_call)
        self.b = PtcHost(peer=self.b_io, mycall=b_call)
        self.a_io.attach(self.a)
        self.b_io.attach(self.b)
        self.b.arq.on_host_listen(True)
        # One tracking receiver per station: each burst arrives in its own window
        # with the same lead-in, which is what a station holding a cycle grid sees.
        self.sync = {a_call: rxfront.SyncedRx(), b_call: rxfront.SyncedRx()}
        self.delivered = 0
        # PACTOR-3 data packets that were genuinely demodulated, per receiver.
        # Counted where the decode happens rather than where the burst was
        # rendered: the claim is that a packet crossed, not that one was built.
        self.p3_rx: list[int] = []

    # Pad each burst with quiet either side. A receiver never sees a burst
    # wall-to-wall -- it sees one inside a rolling window with receive noise
    # around it -- and the pilot trigger measures each candidate against the
    # clip's own median, so a clip that is nothing but signal has no floor to
    # stand above. Handing it a bare burst tests a situation the air never
    # produces, and fails.
    QUIET_S = 1.0

    def _channel(self, audio: np.ndarray) -> np.ndarray:
        quiet = np.zeros(int(self.QUIET_S * FS), np.float32)
        audio = np.concatenate([quiet, audio, quiet]).astype(np.float32)
        if self.snr_db is None:
            return audio
        p = float(np.mean(audio ** 2))
        if p <= 0:
            return audio
        n = np.sqrt(p / (10 ** (self.snr_db / 10)))
        return audio + np.random.default_rng(0).normal(0, n, audio.shape).astype(np.float32)

    def _receive(self, heard: np.ndarray, dst: PtcHost) -> list:
        """Decode one burst into `dst` -- synced when in session, blind otherwise.

        A station in a session always has an answer due: it has either just sent a
        frame and is waiting to be acknowledged, or just acknowledged one and is
        waiting for the next. Its role says WHICH -- the IRS is owed a data packet,
        the ISS a control signal -- and after the first one it also knows where in
        the cycle it lands. That is the whole of the synced mode: ask the tracking
        receiver for the frame that is due, and only sweep when it comes up empty.

        Only a station that is not in a link is truly sweeping, and the blind path
        below is what it sweeps with -- unchanged, because that is the monitor.
        """
        sync = self.sync[dst.mycall]
        expect = (rxfront.CS_EXPECTED_MAX_ERRORS
                  if dst.arq.state in (State.CONNECTING, State.CONNECTED)
                  else rxfront.CS_MAX_ERRORS)
        p1 = dst.protocol is Protocol.PACTOR1
        # BOTH roles, on a PACTOR-1 link. The IRS is owed a data packet; the ISS is
        # owed a control signal but can be handed a CHANGEOVER packet instead, and
        # that is a frame -- CS3 as its head and 840 ms of the new sender's data
        # behind it. Gating the packet scan on IRS made a break-in undecodable by
        # construction: the only station it is ever addressed to never looked.
        #
        # AND BOTH PROTOCOLS, because the upgrade is one station deciding to key
        # PACTOR-3 and the other finding out from the packet. A receiver that reads
        # only what IT is transmitting cannot follow a peer that leads, and the
        # PACTOR-1 scan is CRC-gated and costs 5 ms, so it is asked either way.
        if dst.arq.state == State.CONNECTED:
            ev = _p1_packet(heard)
            if ev is None and not p1:
                ev = (sync.packet(heard) if dst.arq.role == IRS
                      else sync.control_signal(heard, expect))
            if ev is not None:
                return [ev]
        evs = list(rxfront.decode_events(heard, cs_max_errors=expect))
        # A station holding an established link knows a frame may be due, so
        # it pays for the scan that actually finds one. The pilot gate does
        # not: a packet decodes on a clean channel and at no SNR below it,
        # because the gate's ratio sits on its own threshold and the pilot
        # peak's offset from row 0 scatters once noise is present.
        if dst.arq.state == State.CONNECTED and not p1 \
                and not any(e.kind == "packet" for e in evs):
            ev = rxfront.decode_expected_packet(heard)
            if ev is not None:
                evs.append(ev)
        for ev in evs:
            sync.observe(ev)
        return evs

    def _note(self, evs: list) -> None:
        for ev in evs:
            if ev.kind == "packet" and ev.protocol == Protocol.PACTOR3:
                self.p3_rx.append(ev.packet[0])

    def _carry(self, src: AudioSide, dst: PtcHost, tag: str) -> int:
        """Transmit everything queued at `src`; decode it into `dst`."""
        moved = 0
        while src.pending:
            what, audio = src.pending.pop(0)
            heard = self._channel(audio)
            evs = self._receive(heard, dst)
            self._note(evs)
            kinds = ",".join(sorted({e.kind for e in evs})) or "nothing"
            if self.verbose:
                print(f"    {tag} {what:22s} {len(audio) / FS:4.2f}s -> {kinds}")
            for ev in evs:
                dst.on_rx_event(ev)
                self.delivered += 1
                moved += 1
        return moved

    def exchange(self, cycles: int) -> None:
        """Run `cycles` turnarounds: A transmits, B answers, both clock on."""
        for c in range(1, cycles + 1):
            if self.verbose:
                print(f"  cycle {c}:")
            self._carry(self.a_io, self.b, "A->B")
            self.b.tick()
            self._carry(self.b_io, self.a, "B->A")
            self.a.tick()
            if self.verbose:
                print(f"    states: A={self.a.arq.state} B={self.b.arq.state}")


def rx_of(host: PtcHost) -> bytes:
    return bytes(host.channel(host.ptchn).rx)


def main() -> int:
    print("Two shrike stations, every burst rendered and demodulated\n")
    link = AudioLink("W9SSJ", "K7ABC")
    print("  W9SSJ calls K7ABC")
    link.a.arq.on_host_connect("W9SSJ", "K7ABC")
    link.exchange(2)

    a_st, b_st = link.a.arq.state, link.b.arq.state
    up = a_st == State.CONNECTED and b_st == State.CONNECTED
    print(f"\n  link: A={a_st} B={b_st}  ({link.delivered} events carried as audio)")
    if not up:
        print("\nINCOMPLETE -- the link did not come up over the audio path.")
        return 1

    # The multi-cycle part. A connect can be carried by one burst that the far end
    # retries until it lands; a QSO is turnarounds that each carry their own
    # traffic, so the question is whether payload survives the round trip in both
    # directions and lands in the right order.
    # LONGER THAN ONE PACTOR-1 FIELD, which is 20 bytes at 200 Bd and 8 at 100.
    # The upgrade is offered on an acknowledged packet and taken only if the
    # buffer still has something in it (`ptc.PtcHost.upgrade`), so a message that
    # fits in the first packet is carried by a link that never climbs -- which is
    # the right answer for the message and no test of the climb.
    out_a = b"DE W9SSJ QSL -- 73 and thanks for the PACTOR"
    out_b = b"DE K7ABC RR"
    print(f"\n  A sends {out_a!r}")
    link.a.arq.on_host_data(out_a)
    link.exchange(4)
    # SAMPLED WHILE THE PAYLOAD IS CROSSING, because that is when the upgrade is
    # a fact about this link rather than about an idle one. A drained buffer no
    # longer climbs at all, so the instant to read this at is the one where there
    # was something to carry -- and B is in PACTOR-3 here only by having decoded
    # a PACTOR-3 frame A rendered.
    in_p3 = (link.a.protocol, link.b.protocol)
    upgraded = (link.a.protocol is Protocol.PACTOR3
                and link.b.protocol is Protocol.PACTOR3)
    # B is the IRS, so it cannot simply transmit: it has to take the channel
    # first. That is a host action in PACTOR (hostmode break-in), not something
    # queueing data does by itself, and it is the half of a QSO that a one-way
    # transfer never exercises.
    print(f"\n  B breaks in and sends {out_b!r}")
    link.b.arq.on_host_data(out_b)
    link.b.arq.on_host_breakin()
    link.exchange(8)

    # The other way to turn a link around: a changeover REQUEST is a bit in the
    # status byte, not a control signal, and it is an invitation rather than a
    # transfer -- the peer answers it by breaking in with its own CS3-headed
    # packet, and the link changes hands on THAT (measured against WS8EOC,
    # 2026-08-03: a real gateway acknowledges the bit-6 packet and remains the
    # IRS; treating the ack as the turnaround stranded both ends receiving).
    # The exchange therefore costs cycles the old ack-turnaround did not --
    # request, ack, break-in. It ends with both ends in PACTOR-1, where the
    # break-in put them, and there they stay: the buffers are empty by now and an
    # idle link has nothing to gain from climbing again.
    print("\n  B hands the link back (%O -- the changeover bit rides a packet)")
    link.b.arq.on_host_over()
    link.exchange(8)
    handed = (link.a.arq.role, link.b.arq.role) == (ISS, IRS)

    # The sign-off is protocol, not silence: QRT is bit 7 riding a packet, and
    # the link is down only when that packet is acknowledged. A station that
    # vanishes instead leaves the far end retrying into an empty channel -- for
    # a Winlink gateway, a stranded session. This is the only place the QRT
    # crosses as audio; everywhere else it is exercised at the logic layer.
    print("\n  A signs off (QRT -- bit 7 rides a packet, the ack ends the link)")
    link.a.arq.on_host_disconnect()
    link.exchange(6)
    closed = (link.a.arq.state != State.CONNECTED
              and link.b.arq.state != State.CONNECTED)
    print(f"  after QRT      : A={link.a.arq.state} B={link.b.arq.state}")

    got_b, got_a = rx_of(link.b), rx_of(link.a)
    print(f"\n  K7ABC received : {got_b!r}")
    print(f"  W9SSJ received : {got_a!r}")

    fwd = out_a in got_b
    rev = out_b in got_a
    # THE UPGRADE, and it is asserted on packets that were DEMODULATED rather than
    # on the state of the two link layers. Both ends agreeing they are in PACTOR-3
    # is worth nothing on its own -- they would agree just as readily about a
    # waveform neither could read. What is worth something is that a PACTOR-3 data
    # frame was rendered by one station and recovered by the other.
    p3 = link.p3_rx
    print(f"\n  A->B payload   : {'carried' if fwd else 'LOST'}")
    print(f"  B->A payload   : {'carried' if rev else 'LOST'}")
    print(f"  changeover     : {'A is the ISS again' if handed else 'NOT HANDED OVER'}")
    print(f"  upgrade        : both ends in {in_p3[0]}/{in_p3[1]} while the "
          f"payload crossed, {len(p3)} PACTOR-3 packets demodulated at "
          f"SL{sorted(set(p3))}")

    # A clean channel says nothing about a radio. The handshake used to fail at
    # EVERY signal-to-noise ratio, 40 dB included, because the far end's
    # acknowledgement was rejected for landing 2 bit errors from its codeword --
    # so this sweep is the regression that matters, not the clean run above.
    print("\n  handshake and payload against noise:")
    noisy = []
    for snr in (30, 20, 10, 6, 3):
        n = AudioLink("W9SSJ", "K7ABC", snr_db=snr, verbose=False)
        n.a.arq.on_host_connect("W9SSJ", "K7ABC")
        n.exchange(2)
        up = n.a.arq.state == State.CONNECTED and n.b.arq.state == State.CONNECTED
        carried = False
        if up:
            # ONE FIELD'S WORTH, so the sweep stays the measurement it is named
            # for: the PACTOR-1 handshake and a PACTOR-1 packet against noise.
            # A message that outlasts the first packet climbs the link halfway
            # down the ladder and the arm reads the PACTOR-3 floor instead.
            n.a.arq.on_host_data(out_b)
            n.exchange(6)
            carried = out_b in rx_of(n.b)
        noisy.append(up and carried)
        print(f"    {snr:2d} dB SNR -> {'link up' if up else 'NO LINK':8s}  "
              f"payload {'carried' if carried else 'LOST'}")

    print()
    if fwd and rev and handed and closed and upgraded and p3 and all(noisy):
        print("ALL PASS -- the link comes up in PACTOR-1, upgrades to PACTOR-3 "
              "once PACTOR-1 has carried a packet and there is more behind it, "
              "carries payload both ways, "
              "turns around in both directions, signs off with an acknowledged "
              "QRT, and comes up and carries traffic down to 3 dB SNR")
        return 0
    if fwd and rev and handed and not closed:
        print("INCOMPLETE -- the QSO works but the sign-off does not: the QRT "
              "never tears the link down over audio, and a gateway we leave "
              "mid-session is a gateway we strand.")
        return 1
    if fwd and rev and handed and not (upgraded and p3):
        print("INCOMPLETE -- the PACTOR-1 link works and nothing upgraded it, "
              "which is a link to 7% of the Winlink PACTOR network.")
        return 1
    if fwd and rev and not handed:
        print("INCOMPLETE -- payload survives, but a changeover REQUEST never "
              "reaches the far end, so shrike can only ever take a channel.")
        return 1
    if fwd and rev:
        print("INCOMPLETE -- it works on a clean channel but not against noise, "
              "which is not a link.")
        return 1
    print("INCOMPLETE -- the link comes up over real audio but payload does not")
    print("  survive a turnaround. That is the gate for a multi-cycle QSO: a")
    print("  handshake alone is a connect, not a conversation.")
    return 1


def test_main() -> None:
    assert main() == 0


def test_a_kilobyte_climbs_to_the_long_cycle_and_arrives_byte_exact() -> None:
    """The gate's largest payload was 44 bytes, which never reaches a speed
    level where a long cycle is asked for -- so `AudioSide` flew without
    `send_long_packet` and `PtcHost.send_packet` fell through to the short
    renderer, which cut a 276-byte long field to 59 in silence. The ARQ settled
    all 276. A kilobyte climbs the ladder, draws the ask, draws the CS6, and the
    bytes have to come back whole on the other side.
    """
    from hfmodem.shrike import spec

    link = AudioLink("W9SSJ", "K7ABC", verbose=False)
    link.a.arq.on_host_connect("W9SSJ", "K7ABC")
    link.exchange(2)
    assert link.a.arq.state == State.CONNECTED

    blob = bytes((37 * i + 11) & 0xFF for i in range(1024))
    link.a.arq.on_host_data(blob)
    for _ in range(60):
        link.exchange(1)
        if not link.a.arq._outbuf and link.a.arq._inflight is None:
            break
    # ...inside the announcement the connect phase already delivered.
    assert blob in rx_of(link.b)

    long_out = [w for w in link.a_io.sent if "LONG" in w]
    assert long_out, link.a_io.sent
    assert link.a.protocol is Protocol.PACTOR3
    # CHUNKED AT THE LONG FIELD OF ITS OWN LEVEL, which is the length the
    # truncation ate: 276 handed down, 59 on the air, and nothing said so. The
    # last long packet is the one the buffer could not fill -- status bit 5 rides
    # a loaded field with anything behind it, not a long field's worth, so the
    # tail of a transfer runs out inside a long cycle.
    carried = [(int(what.split()[0][2:]), int(what.split()[3][:-1]))
               for what in long_out]
    assert all(n <= spec.SPEED_LEVELS[sl].payload_long for sl, n in carried), \
        long_out
    assert all(n == spec.SPEED_LEVELS[sl].payload_long
               for sl, n in carried[:-1]), long_out


if __name__ == "__main__":
    raise SystemExit(main())
