# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The long cycle: suggested by the ISS, commanded by the IRS, kept by both.

The sustained phase of the one complete third-party PACTOR-3 session on record
runs mostly on the 3.75 s cycle, and the choreography is measured off it
(`rf-corpus/PIII_Complete_1`, DL6MAA -> PTC-II):

    16.61 s   SL3 SHORT packet, 59 B, status 0x33 -- first loaded field, and
              the first with STATUS bit 5 (the long-cycle request) raised
    17.41 s   CS6 in that packet's own answer slot -- the IRS grants the ask
    17.86 s   SL3 LONG packet, 276 B, seq advanced by one: not a repeat, and
              the raster does not move -- 3.750 s spacing on the same comb
    ...       the IRS climbs the ladder with CS4 (24.92, 32.42, 47.42 s);
              its CS4s come only once the fields carry content -- the six
              EMPTY SL3 fields at 9.11-15.37 s are answered CS1/CS2 only
    47.87 s   SL6 LONG packet with bit 5 DROPPED (0x10), the buffer near its
              end; answered CS1 -- the grant is the IRS's discretion
    54.92 s   CS6 -- and from 55.37 s the packets are short again

So the cycle length is link state: the sending station asks with a status bit,
the receiving station commands with a codeword that also acknowledges, and the
counter runs through the change unbroken.  Three arms below: the loopback
choreography, the transmit path against the receiver those ten long fields
validated, and the live stream against the recording itself.

Run:  python -m pytest hfmodem/tests/shrike/test_longcycle.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p3frame, p3rx, placement, rxfront, spec
from hfmodem.shrike.arq import (ArqConfig, ArqIO, CS_ACK, CS_CYCLE_TOG,
                                CS_REQUEST, CS_SPEED_UP, IRS, ISS, LONG_TICKS,
                                PactorArq, State)
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.shrike.test_grid import _Bench

RECORDING = corpora.RF_CORPUS / "PIII_Complete_1.wav"

FS = onair.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
SETTLE_S = 0.040


# --------------------------------------------------------------------------- #
# Loopback: the choreography end to end
# --------------------------------------------------------------------------- #

class _LoopIO(ArqIO):
    def __init__(self, name):
        self.name = name
        self.q: list = []
        self.delivered = bytearray()
        self.sent: list = []          # (sl, len(payload), status) as transmitted
        self.answers: list = []       # every codeword this end keyed
        self.fsm: PactorArq           # source of this packet's rendered cycle

    def connect_burst(self, mycall, dxcall):
        self.q.append(("connect", (mycall, dxcall)))

    def send_packet(self, sl, payload, status, breakin=False):
        self.sent.append((sl, len(payload), status))
        # Carry the physical frame header's cycle independently of status bit5,
        # which requests the NEXT cycle. Snapshot at emission, not queue drain.
        self.q.append(("packet", (sl, payload, status, True, breakin,
                                 None, self.fsm.cycle_long)))

    def send_cs(self, cs_index):
        self.answers.append(cs_index)
        self.q.append(("cs", (cs_index,)))

    def connected(self, mycall, dxcall): pass
    def disconnected(self): pass
    def deliver(self, blob): self.delivered += blob
    def buffer(self, n): pass
    def log(self, m): pass


def _link(*, long_cycle: bool = True):
    a_io, b_io = _LoopIO("A"), _LoopIO("B")
    # speed_up_after=1 so the ladder climbs within a short session, as the
    # reference IRS climbs within ten data cycles.
    cfg = ArqConfig(speed_up_after=1, long_cycle=long_cycle)
    A, B = PactorArq(a_io, cfg), PactorArq(b_io, ArqConfig(long_cycle=long_cycle))
    a_io.fsm, b_io.fsm = A, B

    def pump():
        for _ in range(200):
            moved = False
            for src, dst in ((a_io, B), (b_io, A)):
                while src.q:
                    moved = True
                    kind, args = src.q.pop(0)
                    getattr(dst, {"connect": "on_rx_connect",
                                  "packet": "on_rx_packet",
                                  "cs": "on_rx_cs"}[kind])(*args)
            if not moved:
                return
        raise RuntimeError("pump did not settle")

    def run(cycles=40, until=None):
        for _ in range(cycles):
            for fsm in (A, B):
                fsm.on_cycle()
                pump()
            if until is not None and until():
                return
        if until is not None:
            raise AssertionError("did not settle")

    B.on_host_listen(True)
    A.on_host_connect("N0CALL", "N0DX")
    pump()
    assert (A.state, B.state) == (State.CONNECTED, State.CONNECTED)
    assert (A.role, B.role) == (ISS, IRS)
    A.speed_level = 3            # the level a granted PACTOR-3 phase opens at
    return A, B, a_io, b_io, run, pump


def test_the_iss_asks_the_irs_commands_and_both_keep_the_length():
    A, B, a_io, b_io, run, pump = _link()

    # Empty fields first: an idle ISS draws neither a CS4 nor a CS6.
    run(cycles=4)
    assert all(s & spec.STATUS_LONG_CYCLE == 0 for _, _, s in a_io.sent)
    assert set(b_io.answers) <= {CS_ACK, CS_REQUEST}, b_io.answers

    idle_sent = len(a_io.sent)
    blob = bytes(i & 0xFF for i in range(3000))
    A.on_host_data(blob)
    run(until=lambda: not A._outbuf and A._inflight is None)

    loaded = a_io.sent[idle_sent:]
    statuses = [s for _, _, s in loaded]
    # The first loaded packet raises bit 5 -- the reference's 0x33 at 16.61 s.
    assert statuses[0] & spec.STATUS_LONG_CYCLE, f"0x{statuses[0]:02x}"
    # The IRS grants it with CS6, once each way and never more.
    assert b_io.answers.count(CS_CYCLE_TOG) == 2, b_io.answers
    # The ISS's next chunk is cut to the LONG field of the level it was at.
    first_long = loaded[1]
    assert first_long[1] == spec.SPEED_LEVELS[first_long[0]].payload_long
    # The counter runs through the change unbroken (no restart, no skip).
    seqs = [s & spec.STATUS_SEQ for _, _, s in a_io.sent]
    assert all(b == (a + 1) % 4 for a, b in zip(seqs, seqs[1:])), seqs
    # The ladder climbed on CS4 and never past speed level 6.
    assert CS_SPEED_UP in b_io.answers
    assert max(sl for sl, _, _ in loaded) <= 6
    # Bit 5 dropped on the packet that emptied the buffer, as the reference's
    # did (DL6MAA, 47.87 s, 0x10), and the link came home on the idle train
    # behind it -- the IRS confirms a length change on the peer's next header,
    # and there is no loaded packet left to carry one.
    assert statuses[-1] & spec.STATUS_LONG_CYCLE == 0
    run(cycles=4)
    assert not A.cycle_long and not B.cycle_long
    assert b_io.answers.count(CS_CYCLE_TOG) == 2, b_io.answers
    # Byte-exact delivery through the whole exchange.
    assert bytes(b_io.delivered) == blob


def test_a_long_cycle_is_three_grid_slots():
    """The driver keeps ticking on the 1.25 s raster; a long-cycle FSM acts on
    every third tick. Tick-driven paths only -- a packet chained straight off an
    acknowledgement goes in the slot the answer arrived in, as on a real link.
    The reference's long packets sit 3.750 s apart on the short raster's comb."""
    A, B, a_io, b_io, run, pump = _link()
    # An idle short-cycle ISS keys one packet per tick...
    for _ in range(3):
        A.on_cycle()
    assert len(a_io.sent) == 3
    # ...and the same ISS on the long cycle keys one packet per THREE ticks.
    A.cycle_long = True
    before = len(a_io.sent)
    for _ in range(6):
        A.on_cycle()
    assert len(a_io.sent) - before == 2, a_io.sent[before:]


def test_the_operator_can_decline_the_length_for_a_slot():
    """`--no-long-cycle`. Neither end asks and neither end grants, so the whole
    session runs on the raster every arm has flown -- and a peer that commands
    CS6 anyway is still followed, because a station holding the short raster
    through one desynchronises the link and there is no codeword for "I cannot".
    """
    A, B, a_io, b_io, run, pump = _link(long_cycle=False)
    A.on_host_data(bytes(i & 0xFF for i in range(3000)))
    run(until=lambda: not A._outbuf and A._inflight is None)
    assert all(s & spec.STATUS_LONG_CYCLE == 0 for _, _, s in a_io.sent)
    assert CS_CYCLE_TOG not in b_io.answers, b_io.answers
    assert not A.cycle_long and not B.cycle_long
    assert bytes(b_io.delivered) == bytes(i & 0xFF for i in range(3000))
    # The peer asks anyway: the ordinary gear codeword goes back, which is the
    # reference IRS's own way of holding a request (DL6MAA's bit-5 drop at
    # 47.87 s is answered CS1 and the CS6 comes one packet later).
    answered = len(b_io.answers)
    B.on_rx_packet(3, b"x" * 59,
                   spec.status_byte(1) | spec.STATUS_LONG_CYCLE, True)
    assert b_io.answers[answered:] and CS_CYCLE_TOG not in b_io.answers[answered:]
    assert not B.cycle_long
    # ...and a CS6 commanded at us is still obeyed.
    A.on_rx_cs(CS_CYCLE_TOG)
    assert A.cycle_long


def test_a_seam_that_renders_less_than_it_was_handed_loses_nothing():
    """`ptc.PtcHost.send_packet` routes a long cycle to a seam only if the seam
    has the renderer, falls through to the short one otherwise, and
    `placement.link_packet` cuts the field with no error. The ARQ settled all of
    it: 217 bytes out of the middle of a mail stream, no error in the link layer,
    and the B2F session died on framing several blocks later.
    """
    A, B, a_io, b_io, run, pump = _link()
    cap = 20                      # what this seam's renderer can actually build

    def truncating(sl, payload, status, breakin=False):
        a_io.sent.append((sl, len(payload), status))
        a_io.q.append(("packet", (sl, payload[:cap], status, True, breakin)))
        return min(len(payload), cap)

    a_io.send_packet = truncating
    blob = bytes((5 * i + 3) & 0xFF for i in range(200))
    A.on_host_data(blob)
    run(cycles=200, until=lambda: not A._outbuf and A._inflight is None)
    assert bytes(b_io.delivered) == blob
    # ...and the guard was reached: the FSM cut fields the seam could not build.
    assert any(n > cap for _, n, _ in a_io.sent), a_io.sent
    # The bytes went back at the FRONT of the buffer and in order -- the whole
    # message arrives once, not with a hole and not twice.
    assert bytes(b_io.delivered).count(blob[:64]) == 1


def test_the_irs_can_grant_long_sl1_but_waits_for_the_physical_header():
    A, B, a_io, b_io, run, pump = _link()
    ask = spec.status_byte(1) | spec.STATUS_LONG_CYCLE
    answered = len(b_io.answers)
    B.on_rx_packet(1, b"data", ask, True)
    assert CS_CYCLE_TOG in b_io.answers[answered:]
    assert B._cycle_request is True
    assert not B.cycle_long
    B.observe_peer_cycle(True)
    assert B.cycle_long


def test_a_nak_to_the_floor_keeps_the_negotiated_long_frame():
    A, B, a_io, b_io, run, pump = _link()
    A.cycle_long = True
    A.speed_level = 1
    A.on_host_data(bytes(100))
    for _ in range(3):
        A.on_cycle()
    sl, n, _ = a_io.sent[-1]
    assert (sl, n) == (1, spec.SPEED_LEVELS[1].payload_long)
    assert A.cycle_long


# --------------------------------------------------------------------------- #
# Transmit: the long frame against the receiver the reference validated
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sl", (1, 2, 3, 4, 5, 6))
@pytest.mark.parametrize("swapped", (False, True))
def test_a_long_packet_round_trips_through_the_validated_receiver(sl, swapped):
    """`link_packet(long_cycle=True)` decodes byte-exact through `p3rx` --
    the same paths that read all ten of DL6MAA's long-cycle fields CRC-valid,
    which is what makes this a check against a stranger's modem rather than
    against our own construction."""
    n = spec.SPEED_LEVELS[sl].payload_long
    payload = bytes((7 * i + sl) & 0xFF for i in range(n))
    audio = placement.link_packet(sl, payload, spec.status_byte(1),
                                  swapped=swapped, long_cycle=True)
    pad = np.zeros(spec.SAMPLE_RATE)
    scan = p3rx.decode_p3_packets(np.concatenate([pad, audio, pad]))
    assert [(p.sl, p.payload) for p in scan.packets] == [(sl, payload)]


# --------------------------------------------------------------------------- #
# The raster: where a long packet keys, and where its answer is read
# --------------------------------------------------------------------------- #

class _Placed(onair.RadioTx):
    """A transmitter that keys nothing and records where each burst was aimed.

    `slots_used` is kept the way `_tx` keeps it, because a spent slot is what
    `_advance_aim` reads: a burst chained off an acknowledgement belongs to the
    cycle AFTER the one the tick already keyed, and on the long cycle that is
    three grid slots on rather than one.
    """

    def __init__(self, settle: float = SETTLE_S):
        super().__init__(None, transmit=False, outdir=Path("."), settle=settle)
        self.at: list[tuple[str, int, int]] = []

    def _tx(self, audio, what, drive=None, lead_n=0, *, pulse_offsets=None):
        # `lead_n` is `RadioTx._tx`'s own adjustment, applied here for the same
        # reason: what is recorded has to be where the CARRIER comes up.
        self.at.append((what, self.boundary - lead_n, self.slot))
        self.slots_used.append(self.slot)


def _head_n(audio: np.ndarray) -> int:
    """Samples a rendered packet's audio leads its own phase reference by.

    Read off the render through the production decoder rather than written down:
    the transmit filter puts 40 ms in front of the first symbol and
    `_trim_silence` keeps what clears 2% of the peak. `rx_due` names the phase
    reference, so this is what separates it from where the burst opens.
    """
    pad = np.zeros(round(0.5 * FS), np.float32)
    room = np.concatenate([pad, audio.astype(np.float32), pad])
    p = min(p3rx.decode_p3_packets(room).packets, key=lambda q: q.start)
    return p.start - p3frame.DATA_OFFSET * rxfront.SPS - len(pad)


def _grid(*, long: bool, sending: bool = True, d_ms: float = 80.0):
    """A settled PACTOR-3 grid at the reference's own turnaround.

    `d = 80 ms` is `PIII_Complete_1`'s: over fifteen short cycles its IRS answers
    889.4-891.9 ms after the packet's phase reference, of which 810 is the packet.
    """
    raster = onair._MasterGrid(0, SLOT_N, round(onair.TX_OFFSET_S * FS),
                               packet_n=round(spec.P1_PACKET_S * FS),
                               cs_n=round(spec.P1_CS_S * FS),
                               d_max_n=onair._d_max_n(spec.CYCLE_SHORT_S, 0.04))
    raster.protocol = Protocol.PACTOR3
    raster.sending = sending
    raster.d_n = float(round(d_ms / 1000 * FS))
    raster.d_ref_n = onair.P3_PACKET_N if sending else onair.P3_CS_N
    if long:
        raster.regear(True)
    return raster


def test_the_short_cycle_keeps_every_number_it_flies_with():
    """The control, and it is the half that must not move. Every figure here is
    what the raster produced before it knew a long cycle existed."""
    raster = _grid(long=False)
    assert (raster.ticks, raster.cycle_n) == (1, SLOT_N)
    assert raster.data_n == onair.P3_PACKET_N == round(0.810 * FS)
    assert raster.rx_due(0) - raster.boundary(0) == round(0.890 * FS)
    assert raster.next_slot(4) == 5
    assert raster.boundary(5) - raster.boundary(4) == SLOT_N
    # The fit test, on the burst that flies: a 0.89 s packet behind a 40 ms
    # settle, against a peer codeword one cycle on.
    raster.note_peer_codeword(raster.rx_due(0), onair.P3_CS_N, "CS1", None)
    air_n = round((SETTLE_S + 0.89) * FS)
    assert raster.key_refusal(raster.boundary(1) - round(SETTLE_S * FS),
                              air_n) is None


def test_the_grid_places_a_long_packet_on_the_comb_and_not_off_the_grant():
    """A long packet is chained off the acknowledgement that granted it -- the
    FSM toggles inside `on_rx_cs` and the renderer runs in the flush that decoded
    the codeword, mid-slot. Where it KEYS is the grid's answer, and the grid used
    to have only one: the next 1.25 s slot, 1.25 s inside a 3.37 s packet's own
    cycle and off the phase comb the peer is counting on.
    """
    raster = _grid(long=False)
    tx = _Placed()
    tx.live = _Bench()

    tx.aim(raster, 4)
    tx.send_packet(3, bytes(59), spec.status_byte(1))
    assert tx.at[-1][1:] == (raster.boundary(4), 4)

    # ...the CS6 arrives in that packet's own answer slot and the next packet is
    # long: new counter, not a repeat (the reference, 17.41 -> 17.86 s).
    assert raster.regear(True)
    tx.send_long_packet(3, bytes(276), spec.status_byte(2))
    what, at, slot = tx.at[-1]
    assert "LONG" in what and "276B" in what
    assert (slot, at) == (4 + LONG_TICKS, raster.boundary(4 + LONG_TICKS))
    assert at - tx.at[0][1] == LONG_TICKS * SLOT_N == round(3.750 * FS)


def test_the_answer_to_a_long_packet_is_read_at_3_390_seconds():
    """`rx_due` aimed at 0.890 s whatever the cycle, and the reference's long
    cycles answer at 3390 -- 2.5 s of window aimed into our own transmission.

    The anchor moves by the CYCLE's growth and not the packet's: the packet grows
    2.480 s (810 -> 3290) where the answer moves 2.500, because both answers sit
    360 ms before the next boundary. `d` is the peer's turnaround and comes
    through untouched.
    """
    short = _grid(long=False)
    long = _grid(long=True)
    assert long.data_n == round(3.290 * FS)
    assert long.ticks == LONG_TICKS
    assert long.cycle_n == round(3.750 * FS)
    assert long.rx_due(0) - long.boundary(0) == round(3.390 * FS)
    assert (long.rx_due(0) - long.boundary(0)
            - (short.rx_due(0) - short.boundary(0))) == 2 * SLOT_N
    assert long.d == short.d
    # ...and home again, with every number back where it started.
    assert long.regear(False)
    assert (long.ticks, long.data_n, long.rx_due(0)) == (
        short.ticks, short.data_n, short.rx_due(0))


def test_the_qrm_guard_can_pass_a_long_packet_at_all():
    """`key_refusal`'s fit is `phase + air_n <= slot_n`, and a long keying is
    3.41 s against a 1.25 s slot -- unsatisfiable, so every cycle with anything in
    `_peer_air` was refused `GUARD_MAX_DROPS` times and then keyed anyway by the
    stand-down, over the peer. The projection belongs on the cycle: a station
    transmits once a turn of the link, not once a grid slot.
    """
    raster = _grid(long=True)
    raster.note_peer_codeword(raster.rx_due(0), onair.P3_CS_N, "CS1", None)
    air_n = round(SETTLE_S * FS) + len(
        placement.link_packet(3, bytes(276), spec.status_byte(1),
                              long_cycle=True))
    carrier = raster.boundary(LONG_TICKS) - round(SETTLE_S * FS)
    assert raster.key_refusal(carrier, air_n) is None
    # NEGATIVE CONTROL: the same burst against the slot modulus, which is what
    # the guard asked before, cannot fit by construction.
    assert air_n > SLOT_N


def test_the_peer_s_long_packet_lands_whole_in_our_listen_window():
    """As the IRS we key a 210 ms codeword and the peer keys 3.37 s of packet.
    The hold loop's window runs from our own audio ending to a settle, a holdback
    and `PREKEY_RESERVE_S` before the next key -- and the next key was one 1.25 s
    slot away, so the peer's packet was cut 2.5 s short of its CRC.
    """
    raster = _grid(long=True, sending=False)
    live = _Bench()
    settle_n = round(SETTLE_S * FS)
    slot = 4
    # The loop's own expression, from the hold branch of `onair.run`, and the
    # window opens where our own carrier drops -- the trimmed render, which is
    # what `_tx` keys and `flush_to` discards.
    key_at = raster.boundary(raster.next_slot(slot))
    win_hi = key_at - settle_n - live.holdback - round(
        onair.PREKEY_RESERVE_S * FS)
    win_lo = raster.boundary(slot) + len(
        onair._trim_silence(placement.control_signal(0)))

    peer = onair._trim_silence(
        placement.link_packet(3, bytes(276), spec.status_byte(1),
                              long_cycle=True))
    peer_lo = raster.rx_due(slot) - _head_n(peer)
    peer_hi = peer_lo + len(peer)

    assert win_hi - win_lo >= len(peer), (
        f"window {(win_hi - win_lo) / FS:.3f} s against a "
        f"{len(peer) / FS:.3f} s packet")
    assert win_lo <= peer_lo and peer_hi <= win_hi, (
        f"packet [{(peer_lo - win_lo) / FS:+.3f}, {(peer_hi - win_hi) / FS:+.3f}]"
        f" against the window")

    # NEGATIVE CONTROL: the same cycle stepped one slot, which is what the loop
    # did, leaves a window that cannot hold a third of it.
    short_hi = raster.boundary(slot + 1) - settle_n - live.holdback - round(
        onair.PREKEY_RESERVE_S * FS)
    assert short_hi - win_lo < len(peer) / 3


# --------------------------------------------------------------------------- #
# The stream against the recording
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not RECORDING.exists(), reason="rf-corpus recording not here")
def test_the_stream_carries_every_long_field_of_the_reference_once():
    """`live.RollingRx` used to drop five of the recording's ten long-cycle
    fields: a ~3.4 s body on a 1.5 s slide straddled most 4.0 s windows. It also
    emitted duplicates, because its dedupe key included a per-decode trials
    counter. Both against the file decode, and the count is TAKEN from that decode
    rather than frozen as a literal: the literal outlived the recording's PACTOR-1
    announcement becoming readable, and raising it would have hidden the duplicate
    that arrived in the same red run -- one frame at 3.4000 s reported under two
    hop labels, 2.5 and 3.0 s, which is what the spacing check below catches.
    """
    from hfmodem.shrike import live, rxfront, session

    audio = session.load_wav(str(RECORDING), rxfront.FS)
    got = []
    rx = live.RollingRx(lambda ev: got.append(ev))
    step = rxfront.FS // 10
    for i in range(0, len(audio), step):
        rx.push(audio[i:i + step])
    rx.flush()

    pk = sorted(e.t for e in got if e.kind == "packet")
    want_t = sorted(e.t for e in rxfront.decode_events(audio)
                    if e.kind == "packet")
    assert len(pk) == len(want_t), (pk, want_t)
    # Within a symbol of the file decode: the same frames, timestamped where the
    # audio puts them and not where either caller's search happened to start.
    assert all(abs(a - b) < 0.01 for a, b in zip(pk, want_t)), (pk, want_t)
    long_fields = (17.86, 21.61, 25.36, 29.12, 32.87, 36.62,
                   40.37, 44.12, 47.87, 51.62)
    for want in long_fields:
        assert any(abs(t - want) < 0.05 for t in pk), f"missing {want} s"
    # Emitted once each: no two packet events inside half a short cycle.
    assert all(b - a > 0.6 for a, b in zip(pk, pk[1:])), pk


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
