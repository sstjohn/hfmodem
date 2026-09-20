# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A PACTOR-2 link end to end: two stations, every burst crossing as audio.

`test_p2link.py` renders one raster and grades what a reader takes off it, with a
synthetic peer answering on a script. This carries a whole SESSION: two real
`ptc.PtcHost` state machines, the production renderer for every burst and the
production readers for every one of them, from the PACTOR-1 connect through the
follow into PACTOR-2, a climb up the ladder, the long cycle, a changeover and an
acknowledged QRT -- with the payload graded byte for byte at both ends.

WHAT IT CANNOT SAY, and it is the same bound `pactor2.control_signal` states:
the codeword keying is a HYPOTHESIS. The six words are PACTOR-3's, measured on a
stranger's tape, and their placement on PACTOR-2's two carriers is PACTOR-3's
placement moved across. Both ends here share that reading, so a misreading is
symmetric and therefore invisible; only a station that ANSWERS could grade it,
and no such recording exists. What is graded from outside is the other half --
an independent monitor prints the payload of our data bursts at speed levels 1
through 3, and read the same link three ways with the codewords at 0.880 s, at
0.890 s and absent, and could not tell them apart.

The three arms that are not the session:

  * THE GRID. Where a PACTOR-2 burst keys and where its answer is read, against
    the same `_MasterGrid` PACTOR-3 flies on. A shaped burst opens with the
    leading tail of its own first pulse, so it goes out `pactor2.pulse_lead`
    early or its whole comb is two symbols late.
  * THE CONTROL. A PACTOR-1 link that never hears a PACTOR-2 key, with its
    numbers pinned: nothing above may change what the one protocol every PACTOR
    station has does.
  * THE FLAG. `ptc.PtcHost.offer_pactor2` is off, and what turns it on.

Run:  python -m pytest hfmodem/tests/shrike/test_p2qso.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hfmodem.shrike import onair, p2rx, pactor2, placement, rxfront, spec
from hfmodem.shrike.arq import IRS, ISS, LONG_TICKS, P2_LADDER, State
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_grid import _Bench
from hfmodem.tests.shrike.test_longcycle import _Placed
from hfmodem.tests.shrike.test_qso import AudioLink, AudioSide, rx_of

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)


# --------------------------------------------------------------------------- #
# The session: two stations over audio
# --------------------------------------------------------------------------- #

class P2Side(AudioSide):
    """`AudioSide` with PACTOR-2's three seams, keyed the way `RadioTx` keys them.

    `invert` is the ARQ CYCLE's carrier arrangement, and the two seams that read
    it read it differently for the reason `RadioTx._flip` and
    `RadioTx._cycle_swap` do: a packet opens a cycle and steps the alternation, a
    codeword answers one and must go out in the arrangement the packet it answers
    was keyed in. A fresh flip on the answer puts it on the arrangement the NEXT
    packet will use, which is the one the peer is not listening on.
    """

    invert = False

    def send_p2_cs(self, index: int) -> None:
        self._tx(f"P2 CS{index + 1}",
                 pactor2.control_signal(index, swapped=self.invert))

    def send_p2_packet(self, sl: int, payload: bytes, status: int) -> int:
        self.invert = not self.invert
        return self._p2_burst(pactor2.PATHS[sl - 1], payload, status)

    def send_p2_long_packet(self, sl: int, payload: bytes, status: int) -> int:
        self.invert = not self.invert
        return self._p2_burst(pactor2.PATHS_LONG[sl - 1], payload, status)

    def _p2_burst(self, path, payload: bytes, status: int) -> int:
        n = path.crc_bytes - 3
        self._tx(f"P2 {path.name} pkt {len(payload)}B",
                 pactor2.data_burst(
                     pactor2.build_field(
                         placement.field_info(payload, n, status), path),
                     path, swapped=self.invert))
        return n


class P2AudioLink(AudioLink):
    """`AudioLink` with PACTOR-2 in both directions, through the live readers.

    The receive side is `onair._SessionRx`'s own `_p2_packet` and `_p2_cs` --
    not a copy of them -- so what this grades is the code a session runs. One
    `_SessionRx` per station, held for its `p2_bin_pair`: the codeword read
    cannot find its own carrier pair and takes the one the last decoded frame
    armed on.

    BOTH READERS ON EVERY BURST, which a real cycle does not do -- there
    `deep_scan` runs the link's own protocol before the key and `upgrade_scan`
    the rest behind our own carrier. The difference is WHEN, not whether, and a
    session that followed a cycle late would need a cycle more of exchange to say
    the same things.
    """

    SIDE = P2Side

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.rx = {self.a.mycall: onair._SessionRx(self.a, "A"),
                   self.b.mycall: onair._SessionRx(self.b, "B")}

    def _p2(self, heard: np.ndarray, dst) -> list:
        rx = self.rx[dst.mycall]
        got = rx._p2_packet(heard)
        if got is not None:
            return [got[0]]
        # The codeword's phase reference, which the pad in front of the burst
        # puts at a known sample -- the grid's `rx_due` on the air.
        at = int(self.QUIET_S * FS) + pactor2.pulse_lead()
        ev = rx._p2_cs(heard, at)
        return [ev] if ev is not None else []

    def _receive(self, heard: np.ndarray, dst) -> list:
        evs = self._p2(heard, dst)
        if evs:
            return evs
        if dst.protocol is Protocol.PACTOR2:
            # In PACTOR-2 the production reader table is `_p2_cs` and then
            # `_p1_cs` and never PACTOR-3's (`_SessionRx.control_signal`), so
            # the harness may not offer one either: `SyncedRx` will read an
            # acknowledgement out of our own PACTOR-2 codeword, which on the air
            # nothing is listening on tones 5 and 12 to do.
            ev = rxfront.decode_expected_p1_packet(heard)
            return [ev] if ev is not None else []
        return super()._receive(heard, dst)

    def _carry(self, src, dst, tag: str) -> int:
        # ONE ARRANGEMENT PER ARQ CYCLE, AND IT IS THE SENDER'S. Both stations
        # read it off one raster on the air (`_MasterGrid.shift`); here it
        # travels with the audio, so the station about to answer answers in the
        # arrangement of the transmission it just decoded.
        dst.peer.invert = src.invert
        return super()._carry(src, dst, tag)


def _linked(**kw) -> P2AudioLink:
    link = P2AudioLink("W9SSJ", "K7ABC", verbose=False, **kw)
    link.a.arq.on_host_connect("W9SSJ", "K7ABC")
    link.exchange(2)
    assert (link.a.arq.state, link.b.arq.state) == (State.CONNECTED,
                                                    State.CONNECTED)
    return link


def _drain(link: P2AudioLink, cycles: int = 40) -> None:
    for _ in range(cycles):
        link.exchange(1)
        if not (link.a.arq._outbuf or link.b.arq._outbuf
                or link.a.arq._inflight or link.b.arq._inflight):
            return


def test_a_pactor2_qso_from_the_connect_to_the_goodbye() -> None:
    """The whole session, and the payload has to survive every seam in it.

    The CALLED station leads, which is the only way a PACTOR-2 link has ever
    been reachable: it is the IRS, so it takes the channel first, and the packet
    it takes it with is the first PACTOR-2 frame on the air. `_ruled_out` here
    stands in for the contradiction a PACTOR-2-only gateway supplies on its own
    -- it answered our PACTOR-3 in PACTOR-1 -- and `offer_pactor2` is what lets
    the second rung be tried behind it.

    The caller follows on the frame (`ptc.PtcHost._follow_peer`), and from there
    both ends are on `arq.P2_LADDER`: four rungs climbing to three, chunked at
    5/14/32 bytes short and 36/76/156 long.
    """
    link = _linked()
    link.a.arq.cfg.speed_up_after = 1
    link.b.arq.cfg.speed_up_after = 1
    link.b.offer_pactor2 = True
    link.b._ruled_out.add(Protocol.PACTOR3)

    # -- the called station takes the channel and leads into PACTOR-2 --------
    # Long enough for the climb to have the cycles it needs: status bit 5 rides
    # a loaded field with anything behind it, so this phase runs mostly on the
    # long cycle and a rung costs one packet rather than one short packet.
    from_b = (b"K7ABC HERE, PACTOR TWO -- ENOUGH TEXT TO CLIMB THE LADDER "
              b"TWICE. THE FLOOR CARRIES 5 BYTES SHORT AND 36 LONG AND THE "
              b"RUNG ABOVE IT 76, SO A CLIMB OF TWO RUNGS COSTS MORE THAN A "
              b"HUNDRED AND SEVENTEEN OF THEM BEFORE THE THIRD IS REACHED.")
    link.b.arq.on_host_data(from_b)
    link.b.arq.on_host_breakin()
    _drain(link, 24)

    assert link.b.protocol is Protocol.PACTOR2, link.b.protocol
    assert link.a.protocol is Protocol.PACTOR2, link.a.protocol
    assert link.a.arq.ladder is link.b.arq.ladder is P2_LADDER
    assert rx_of(link.a).endswith(from_b), rx_of(link.a)

    # -- the ladder: the phase opened at the floor and CS4 climbed it --------
    keyed = [w for w in link.b_io.sent if w.startswith("P2 SL")]
    levels = [int(w.split()[1].split("-")[0][2:]) for w in keyed]
    assert levels[0] == P2_LADDER.entry_sl == 1
    assert sorted(set(levels)) == [1, 2, 3], keyed
    assert levels == sorted(levels), keyed          # a climb, never a drop
    assert link.b.arq.speed_level == P2_LADDER.top == 3
    # ...and every field is the one its own rung holds.
    for what, sl in zip(keyed, levels):
        long = "-long" in what
        assert int(what.split()[-1][:-1]) <= P2_LADDER.payload(sl, long), what

    # -- the long cycle, which PACTOR-2 reaches at its floor as well ---------
    assert [w for w in keyed if "-long" in w], keyed
    blob = bytes((41 * i + 7) & 0xFF for i in range(900))
    link.b.arq.on_host_data(blob)
    _drain(link, 80)
    assert blob in rx_of(link.a)
    long_at_speed = [w for w in link.b_io.sent if "SL3-long" in w]
    assert long_at_speed, [w for w in link.b_io.sent if w.startswith("P2 SL")]
    carried = [int(w.split()[-1][:-1]) for w in long_at_speed]
    assert max(carried) == P2_LADDER.payload(3, True), long_at_speed

    # -- the changeover, which in PACTOR-2 is the bare CS3 -------------------
    # `ptc.PtcHost.breakin_rides_a_packet` is False here: PACTOR-2 has no
    # 51-pulse frame to hang a codeword head on, so the codeword IS the
    # break-in and the new sender's first packet goes out a cycle later.
    from_a = b"W9SSJ RR ALL COPY, THE LINK TURNED AROUND"
    before = len(link.a_io.sent)
    link.a.arq.on_host_data(from_a)
    link.a.arq.on_host_breakin()
    _drain(link, 40)
    assert (link.a.arq.role, link.b.arq.role) == (ISS, IRS)
    assert rx_of(link.b).endswith(from_a), rx_of(link.b)
    took = link.a_io.sent[before:]
    assert "P2 CS3" in took, took
    assert not [w for w in took if "BREAK-IN" in w], took

    # -- and the goodbye is protocol rather than silence ---------------------
    link.a.arq.on_host_disconnect()
    link.exchange(8)
    assert link.a.arq.state != State.CONNECTED
    assert link.b.arq.state != State.CONNECTED
    assert link.a.arq.goodbye_acked, "the QRT went unacknowledged"
    # Nothing fell back: every burst after the follow was PACTOR-2.
    tail = [w for w in link.a_io.sent[before:] + link.b_io.sent[before:]
            if not w.startswith("P2 ")]
    assert not tail, tail


# --------------------------------------------------------------------------- #
# The grid: where a PACTOR-2 burst keys, and where its answer is read
# --------------------------------------------------------------------------- #

def _grid(*, long: bool, sending: bool = True, d_ms: float = 80.0):
    """A settled PACTOR-2 grid, at PACTOR-3's own measured turnaround.

    80 ms, because that is the only turnaround anything here has measured
    (`_MasterGrid.data_n`) and [SCS] s2 says PACTOR-2 changes nothing about it:
    "the transmit delay and the receiver recovery time of the used equipment
    therefore remain unchanged in comparison with Level I". At that gap the
    answer lands exactly where `pactor2.cs_slot` puts it.
    """
    raster = onair._MasterGrid(0, SLOT_N, round(onair.TX_OFFSET_S * FS),
                               packet_n=round(spec.P1_PACKET_S * FS),
                               cs_n=round(spec.P1_CS_S * FS),
                               d_max_n=onair._d_max_n(spec.CYCLE_SHORT_S, 0.04))
    raster.protocol = Protocol.PACTOR2
    raster.sending = sending
    raster.d_n = float(round(d_ms / 1000 * FS))
    raster.d_ref_n = onair.P2_PACKET_N if sending else onair.P2_CS_N
    if long:
        raster.regear(True)
    return raster


def test_the_grid_answers_a_pactor2_packet_at_the_slot_it_renders() -> None:
    """0.880 s short and 3.360 s long -- `pactor2.cs_slot` from the other side.

    The two figures have to agree or the link is deaf in one direction with
    nothing on the air to say so: `cs_slot` is where our own transmitter puts
    the answer to a peer's packet, and `rx_due` is where our receiver looks for
    the answer to ours.
    """
    short, long = _grid(long=False), _grid(long=True)
    assert short.data_n == onair.P2_PACKET_N == round(0.800 * FS)
    assert long.data_n == onair.P2_LONG_PACKET_N == round(3.280 * FS)
    assert short.cs_n == onair.P2_CS_N == round(0.210 * FS)
    assert (short.ticks, long.ticks) == (1, LONG_TICKS)
    assert long.cycle_n == round(3.750 * FS)

    for raster, path in ((short, pactor2.PATHS[0]), (long, pactor2.PATHS_LONG[0])):
        due = raster.rx_due(0) - raster.boundary(0)
        assert due == round(pactor2.cs_slot(path) * FS), due
    assert short.rx_due(0) - short.boundary(0) == round(0.880 * FS)
    assert long.rx_due(0) - long.boundary(0) == round(3.360 * FS)
    # ...and 10 ms in front of PACTOR-3's at either length, which is the one
    # pulse the two combs differ by.
    assert (onair.P3_PACKET_N - onair.P2_PACKET_N
            == onair.P3_LONG_PACKET_N - onair.P2_LONG_PACKET_N
            == round(0.010 * FS))
    # ...and home again with every number back where it started.
    assert long.regear(False)
    assert (long.ticks, long.data_n, long.rx_due(0)) == (
        short.ticks, short.data_n, short.rx_due(0))


class _Rendered(_Placed):
    """`_Placed`, keeping the samples: the arrangement is IN the audio."""

    def __init__(self) -> None:
        super().__init__()
        self.audio: list = []

    def _tx(self, audio, what, drive=None, lead_n=0):
        super()._tx(audio, what, drive, lead_n)
        self.audio.append(audio)


def test_a_pactor2_burst_keys_its_pulse_lead_early() -> None:
    """A shaped burst does not start at its own phase reference.

    `pactor2.tx_pulse` spans nearly four symbols, so the first pulse's leading
    tail is on the air 18.75 ms before the instant Figure 1's raster is written
    in. Keyed ON the boundary the whole comb is that late -- nearly two symbols
    at 100 Bd -- and the peer's answer window, which counts from the phase
    reference, is aimed 18.75 ms wrong for the rest of the link.
    """
    lead = pactor2.pulse_lead()
    assert lead == 900 == round(0.01875 * FS)

    raster = _grid(long=False)
    tx = _Rendered()
    tx.live = _Bench()
    tx.aim(raster, 4)
    n = tx.send_p2_packet(1, b"", spec.status_byte(1))
    what, at, slot = tx.at[-1]
    assert n == P2_LADDER.payload(1, False) == 5
    assert (slot, at) == (4, raster.boundary(4) - lead)
    assert "SL1-short" in what, what

    # The long frame keys on the comb three slots on, the same lead early.
    assert raster.regear(True)
    tx.send_p2_long_packet(3, bytes(156), spec.status_byte(2))
    what, at, slot = tx.at[-1]
    assert "SL3-long" in what and "156B" in what, what
    assert (slot, at) == (4 + LONG_TICKS,
                          raster.boundary(4 + LONG_TICKS) - lead)

    # A codeword takes the same lead, and it takes the CYCLE's arrangement --
    # the one the packet it answers went out in. A fresh flip here would put our
    # acknowledgement on the arrangement the next packet will use, which is the
    # one the peer is not listening on.
    slot = 4 + 2 * LONG_TICKS
    tx.aim(raster, slot)
    tx.send_p2_cs(0)
    what, at, _slot = tx.at[-1]
    assert at == raster.boundary(slot) - lead
    assert np.array_equal(
        tx.audio[-1],
        pactor2.control_signal(0, swapped=bool(raster.shift(slot))))

    # And it spends no toggle. `_flip` is the alternation every PACKET steps;
    # where there is no grid to count cycles on, `_cycle_swap` reads that
    # standing toggle rather than advancing it.
    bare = onair.RadioTx(None, transmit=False, outdir=Path("."))
    assert bare._cycle_swap() is bare._cycle_swap() is False
    assert bare._flip() is False and bare._cycle_swap() is True


def test_the_answer_slot_reads_the_codeword_our_own_renderer_keys() -> None:
    """The grid and the waveform, closed on each other over real samples.

    `rx_due` names an instant and `p2rx.control_signal_at` reads twenty bits
    there at zero errors. Rendered at that instant the codeword comes back; a
    slot away it does not, which is what makes the placement load-bearing rather
    than incidental.
    """
    for path in (pactor2.PATHS[2], pactor2.PATHS_LONG[2]):
        for swapped in (False, True):
            slot = pactor2.cs_slot(path)
            lead = pactor2.pulse_lead()
            audio = np.zeros(int((slot + 1.0) * FS))
            cs = pactor2.control_signal(4, swapped=swapped)
            at = int(round(slot * FS))
            audio[at - lead:at - lead + cs.size] += 0.2 * cs
            got = p2rx.control_signal_at(audio, at, swapped=swapped)
            assert got == (4, 0), (path.name, swapped, got)
            # NEGATIVE: the home arrangement does not read a swapped cycle.
            assert p2rx.control_signal_at(audio, at, swapped=not swapped) is None
            # NEGATIVE: nor does the instant a PACTOR-3 grid would aim at.
            assert p2rx.control_signal_at(
                audio, at + round(0.010 * FS), swapped=swapped) is None


# --------------------------------------------------------------------------- #
# The control: a PACTOR-1 link that never hears a PACTOR-2 key
# --------------------------------------------------------------------------- #

def test_a_pactor1_link_decodes_exactly_as_it_did() -> None:
    """The half that must not move, with its numbers written down.

    Everything above adds a protocol to `ptc.TRANSMITTABLE`, a ladder to the ARQ
    layer, a reader to the control-signal table and a length to the grid. None of
    it may reach a link that never hears PACTOR-2 -- which is most links, and the
    one protocol every PACTOR station has.
    """
    link = AudioLink("W9SSJ", "K7ABC", verbose=False)
    link.a.stay_in_pactor1 = link.b.stay_in_pactor1 = True
    link.a.arq.on_host_connect("W9SSJ", "K7ABC")
    link.exchange(2)
    assert (link.a.arq.state, link.b.arq.state) == (State.CONNECTED,
                                                    State.CONNECTED)

    out = b"DE W9SSJ QSL -- 73 AND THANKS FOR THE PACTOR"
    link.a.arq.on_host_data(out)
    link.exchange(8)

    assert out in rx_of(link.b), rx_of(link.b)
    assert link.a.protocol is Protocol.PACTOR1
    assert link.b.protocol is Protocol.PACTOR1
    assert link.a.arq.ladder is not P2_LADDER
    # PINNED, every burst of it. The answer named 200 Bd, so the field is 20
    # bytes: the announcement, an idle packet, then 20 + 20 + 4 of the message
    # and idle packets after it. The counter runs mod 4 with no repeat anywhere,
    # and the reverse channel alternates strictly -- which is what an
    # acknowledgement IS in PACTOR-1.
    assert link.a_io.sent == [
        "connect->K7ABC",
        "P1 pkt#1 7B", "P1 pkt#2 0B", "P1 pkt#3 20B", "P1 pkt#0 20B",
        "P1 pkt#1 4B", "P1 pkt#2 0B", "P1 pkt#3 0B", "P1 pkt#0 0B",
        "P1 pkt#1 0B", "P1 pkt#2 0B"], link.a_io.sent
    assert link.b_io.sent == ["P1 CS1"] + ["P1 CS1", "P1 CS2"] * 5, \
        link.b_io.sent
    assert not [w for w in link.a_io.sent + link.b_io.sent if "P2" in w]
    assert rx_of(link.b) == b"1w9ssj\r" + out


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #

def test_the_pactor2_offer_is_off_until_it_is_asked_for() -> None:
    """`offer_pactor2` is the OFFER, not the follow, and the null is its reason.

    No recording anywhere holds a PACTOR-1 link transitioning into PACTOR-2, and
    both graded third-party completions go 1 -> 3 in one step. So an uninvited
    PACTOR-2 rung is an operator's arm rather than a session's own behaviour --
    where FOLLOWING a peer that has already keyed the waveform needs no flag at
    all.
    """
    from unittest import mock
    import sys

    from hfmodem.shrike.ptc import UPGRADE_TARGETS, PtcHost, SimPeer

    host = PtcHost(SimPeer(), mycall="W9SSJ")
    assert host.offer_pactor2 is False
    assert host._upgrade_targets() == UPGRADE_TARGETS == (Protocol.PACTOR3,)
    host.offer_pactor2 = True
    assert host._upgrade_targets() == (Protocol.PACTOR3, Protocol.PACTOR2)

    got = []
    argv = ["onair", "--dxcall", "WS8EOC", "--mycall", "W9SSJ"]
    with mock.patch.object(onair, "run", lambda a: got.append(a) or 0), \
            mock.patch.object(sys, "argv", argv):
        onair.main()
    assert got[0].offer_pactor2 is False
    got.clear()
    with mock.patch.object(onair, "run", lambda a: got.append(a) or 0), \
            mock.patch.object(sys, "argv", argv + ["--offer-pactor2"]):
        onair.main()
    assert got[0].offer_pactor2 is True
