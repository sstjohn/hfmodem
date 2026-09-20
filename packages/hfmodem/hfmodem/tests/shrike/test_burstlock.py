# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Reading the peer's data phase, and transmitting in the gap after its bursts.

Two halves of the same thing -- talking to a station that is not us -- and both
were missing in ways that only a real QSO would have shown.

RECEIVE. `rxfront.decode_events` runs the PACTOR-1 data scan only after it has
decoded a connect in the SAME buffer. In a QSO the connect is ours, so nothing the
peer sends can set that flag, and a station that transmits a correct data phase
still cannot read one. `decode_expected_p1_packet` is the entry point for a caller
that knows a frame is due; what it must not become is a decoder that finds frames
in noise, so the header gate is measured here rather than asserted.

TRANSMIT. A caller is the MASTER, and a master's transmit instants are a local
1.25 s grid referenced to nothing the far end does -- the peer synchronises to
it, not the other way round. This file used to assert the opposite and pass: it
checked that the raster tracked the peer's bursts to within 5 ms over 24 cycles,
which the code did faithfully, and which is a positive feedback path the protocol
deliberately does not have. What replaces it is the harder claim, that a peer
which jitters and drifts moves our RECEIVE window and not one sample of our
transmit timing.

Run:  python -m hfmodem.tests.shrike.test_burstlock
"""
from __future__ import annotations

import contextlib
import io
import re

import numpy as np
import pytest

from hfmodem.shrike import (coding, onair, p1rx, pactor1, placement, rxfront,
                            session, spec)
from hfmodem.shrike.arq import IRS, State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike import archive

FS = rxfront.FS
SETTLE_N = round(0.04 * FS)     # the FT-891's, and the only settle that closes
# The peer in the schedule tests below is a recording of a real gateway calling,
# so the detector is in the loop and only the transmitter is modelled.
PEER_WAV = archive.SILENT_GATEWAYS[0]
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _in_noise(burst: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """A burst inside a quiet window, at a stated signal-to-noise ratio."""
    quiet = np.zeros(int(0.2 * FS), np.float32)
    x = np.concatenate([quiet, burst, quiet]).astype(np.float32)
    p = float(np.mean(x ** 2)) or 1.0
    n = np.sqrt(p / 10 ** (snr_db / 10))
    return x + np.random.default_rng(seed).normal(0, n, x.shape).astype(np.float32)


# --------------------------------------------------------------------------
# receive: the peer's data packet
# --------------------------------------------------------------------------

def expected_p1_packet() -> None:
    print("\na peer's PACTOR-1 data packet, in the cycle where one is due")
    payload = b"DE WS8EOC QSL"
    got = []
    for snr in (30, 20, 10, 6, 3):
        audio = _in_noise(pactor1.packet_signal(payload, 100, packet_count=1), snr)
        ev = rxfront.decode_expected_p1_packet(audio)
        got.append(ev is not None and ev.packet[2].startswith(payload[:8]))
    check("decoded from 30 dB down to 3 dB", all(got), f"{sum(got)} of 5")

    ev = rxfront.decode_expected_p1_packet(
        _in_noise(pactor1.packet_signal(payload, 100, packet_count=2), 20))
    check("the event carries the protocol it was read in",
          ev is not None and ev.protocol == "PACTOR-1",
          ev.protocol if ev else "no event")
    check("...and the status byte, which is where the sequence lives",
          ev is not None and (ev.packet[1] & 0b11) == 2,
          f"status 0x{ev.packet[1]:02x}" if ev else "no event")

    # Both shift positions: "mit jedem neuen Paket ... wird die Shiftlage
    # invertiert", so half of a peer's packets arrive as the complement.
    both = [rxfront.decode_expected_p1_packet(
                _in_noise(pactor1.packet_signal(payload, 100, packet_count=1,
                                                invert=inv), 20)) is not None
            for inv in (False, True)]
    check("both FSK shift positions read", all(both), f"{sum(both)} of 2")

    quiet = np.random.default_rng(1).normal(0, 0.05, int(4 * FS)).astype(np.float32)
    check("nothing claimed on band noise",
          rxfront.decode_expected_p1_packet(quiet) is None)


def header_gate() -> None:
    """The eight bits the CRC does not cover, and what they are worth."""
    print("\nthe header gate on a CRC-valid frame")
    payload = b"1W9SSJ\r"
    frame = pactor1.data_packet(payload, 100, packet_count=1)
    check("a genuine frame's header is one of the two the mode has",
          frame[0] in p1rx.P1_DATA_HEADERS, f"0x{frame[0]:02x}")
    # The header sits OUTSIDE the CRC region, so a frame carrying a foreign one
    # still validates -- which is exactly why the CRC alone cannot be the gate,
    # and why the gate has to be a separate test rather than a stronger checksum.
    # Asserted against the CRC directly: `decode_p1_packets` applies the gate
    # itself now, so asking IT would only measure the gate twice.
    bad = pactor1.data_packet(payload, 100, packet_count=1, header=0x3C)
    check("...and a foreign header does not break the CRC",
          coding.crc16(bad[1:-2], pactor1.DATA_CRC)
          == (bad[-1] << 8 | bad[-2]), f"header 0x{bad[0]:02x}")
    audio = _in_noise(pactor1.packet_signal(payload, 100, packet_count=1,
                                            header=0x3C), 30)
    check("but the gate rejects it", not p1rx.decode_p1_packets(audio))
    # ...while the same frame under a legal header is still read, or the gate is
    # simply a decoder that has stopped decoding.
    good = _in_noise(pactor1.packet_signal(payload, 100, packet_count=1), 30)
    check("...and the same frame under a legal header still decodes",
          any(p.payload == payload for p in p1rx.decode_p1_packets(good)))


def packet_reaches_the_arq_layer() -> None:
    """Gap 1 end to end: a peer's frame must be acknowledged and delivered."""
    print("\nthe frame the peer sent reaches PactorArq")
    sent: list = []

    class Seam:
        """A transmit seam that records instead of keying."""

        def attach(self, host):
            pass

        def connect_burst(self, mycall, dxcall):
            sent.append(("connect", dxcall))

        def send_cs(self, i):
            sent.append(("cs", i))

        def send_p1_cs(self, i):
            sent.append(("p1cs", i))

        def send_p1_packet(self, payload, baud, packet_count, **kw):
            sent.append(("p1pkt", payload))

        def send_packet(self, sl, payload, status, breakin=False):
            sent.append(("pkt", payload))

        def pump(self):
            pass

        def cycle(self):
            pass

    host = PtcHost(peer=Seam(), mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.role, host.arq.dxcall = IRS, "WS8EOC"
    host.arq._enter_connected()
    sent.clear()

    rx = onair._SessionRx(host, tag="TEST")
    # Counter 1: "das erste normale Datenpaket mit Head=AA (HEX) und
    # Paketzaehler=1", which is what the FSM expects on a freshly-made link.
    payload = b"1WS8EOC\r"
    audio = _in_noise(pactor1.packet_signal(payload, 100, packet_count=1), 20)
    rx.deep_scan(audio)

    check("the payload was delivered to the host",
          payload in bytes(host.channel(host.ptchn).rx),
          repr(bytes(host.channel(host.ptchn).rx)[:24]))
    check("...and answered with a PACTOR-1 control signal",
          any(k == "p1cs" for k, _ in sent), str(sent))
    # A PACTOR-3 frame decoder on a PACTOR-1 link is not a fallback, it is the
    # wrong demodulator on the wrong tones -- which is what deep_scan ran before.
    check("deep_scan did not have to fall back to the PACTOR-3 scan",
          host.protocol is Protocol.PACTOR1)

    # THE PROTOCOL THE PEER MIGHT LEAD INTO IS STILL READ, just not before the
    # key: the blind scan for it costs more than the gap between the peer's burst
    # and our PTT, so it runs behind our own carrier instead. Asked here from the
    # other side, with the same PACTOR-1 packet and a link that thinks it is in
    # PACTOR-3 -- the reader that finds it is then the speculative one, and if
    # `upgrade_scan` ever stops being called the packet reaches nobody.
    up = PtcHost(peer=Seam(), mycall="W9SSJ")
    up.arq.on_host_listen(True)
    up.arq.role, up.arq.dxcall = IRS, "WS8EOC"
    up.arq._enter_connected()
    up.protocol = Protocol.PACTOR3
    rx2 = onair._SessionRx(up, tag="TEST")
    rx2.deep_scan(audio)
    check("the scan before the key reads the link's own protocol and stops "
          "there", rx2.count == 0, f"{rx2.count} events")
    rx2.upgrade_scan(audio)
    check("...and the scan behind our carrier is what follows the peer into "
          "another one", payload in bytes(up.channel(up.ptchn).rx),
          repr(bytes(up.channel(up.ptchn).rx)[:24]))

    # A CALL IS ANSWERED BY A FRAME AS WELL AS BY A CODEWORD, and the frame is
    # the unforgeable one -- a CRC-16 against a 20-bit codeword read out of
    # noise. `arq.on_rx_packet` has carried the branch for it since PACTOR-1
    # links started coming up, for the case a Winlink RMS answers a connect by
    # sending its greeting rather than a control signal, and no audio could
    # reach it: `_SessionRx._scan` admitted CONNECTED only, so the CRC scan --
    # the one path a data frame arrives by at any real SNR -- was closed for
    # exactly the phase the branch exists for.
    calling = PtcHost(peer=Seam(), mycall="W9SSJ")
    calling.arq.on_host_connect("W9SSJ", "WS8EOC")
    sent.clear()
    onair._SessionRx(calling, tag="TEST").deep_scan(audio)
    check("a frame answering our call brings the link up while CONNECTING",
          calling.arq.state == State.CONNECTED and calling.arq.role == IRS,
          f"state {calling.arq.state}, role {calling.arq.role}")
    check("...and its payload is delivered rather than dropped with the phase",
          payload in bytes(calling.channel(calling.ptchn).rx),
          repr(bytes(calling.channel(calling.ptchn).rx)[:24]))
    check("...and it is answered", any(k == "p1cs" for k, _ in sent), str(sent))

    # ...AND THE FRAME HAS TO BE ONE A CALL COULD BE ANSWERED WITH. A data packet
    # carries no address in any PACTOR mode, so a caller cannot ask who sent one
    # -- what it can ask is whether the frame belongs to a link that is already
    # up, because ours is not. Two answers to that, and each of them used to bring
    # the link up off a station that had never heard our call.
    walked_in = PtcHost(peer=Seam(), mycall="W9SSJ")
    walked_in.arq.on_host_connect("W9SSJ", "WS8EOC")
    sent.clear()
    rx3 = onair._SessionRx(walked_in, tag="TEST")
    # A CHANGEOVER PACKET: CS3 as its head, and CS3 is never bare -- it is a link
    # changing hands between two stations that already have one.
    rx3.deep_scan(_in_noise(
        pactor1.breakin_signal(b"QSL\r", 100, packet_count=1), 20))
    check("somebody else's changeover packet is not an answer to our call",
          walked_in.arq.state == State.CONNECTING,
          f"state {walked_in.arq.state}, role {walked_in.arq.role}")
    # A PACTOR-3 FRAME: every session of one is entered through a PACTOR-1
    # connect and its phase opens about three cycles after the PACTOR-1 answer
    # (docs/protocols/pactor/pactor3.md sec 17.1), so a station keying one has a
    # link -- with somebody else.
    rx3.new_cycle()
    rx3.upgrade_scan(_in_noise(placement.link_packet(2, b"CQ DE ANOTHER QSO",
                                                     0x01), 20))
    check("...nor is a PACTOR-3 frame from an exchange already running",
          walked_in.arq.state == State.CONNECTING,
          f"state {walked_in.arq.state}, role {walked_in.arq.role}")
    check("...and both were read and refused, not simply missed",
          rx3.count == 2, f"{rx3.count} frames scanned")
    check("...so neither cost a transmission", not sent, str(sent))
    # AND NEITHER BOUGHT A RETRY. A refusal used to spend `note_peer_heard`, whose
    # premise is undecoded energy that might be the station we called; these frames
    # were decoded, and what they say is that somebody else has this frequency.
    # Measured cost of the old reading: working/rig-session-20260813-235522, where a
    # stranger's PACTOR-3 codeword at zero bit errors (line 96) was spent at line
    # 119 and held the call open seven grid cycles past its budget.
    walked_in.log_lines.clear()
    walked_in.arq.on_cycle()
    check("...nor did either buy a retry off a QSO we are not in",
          not any("not counting this retry" in m for m in walked_in.log_lines),
          str(walked_in.log_lines))

    # THE SAME QUESTION ASKED OF A CODEWORD, where the answer used to turn on an
    # index collision: PACTOR-3's first control signal and PACTOR-1's CS1 are both
    # index 0, so a stranger's PACTOR-3 acknowledgement read as an answer to our
    # call. Twenty bits carry no address either, and the protocol they were read
    # in is what the phase has.
    def _cs(index: int, protocol: str = "PACTOR-1"):
        return rxfront.Event(0.0, "cs", f"CS{index + 1}", protocol=protocol,
                             cs=index)

    stranger = PtcHost(peer=Seam(), mycall="W9SSJ")
    stranger.arq.on_host_connect("W9SSJ", "WS8EOC")
    sent.clear()
    stranger.on_rx_event(_cs(0, protocol="PACTOR-3"))
    check("a PACTOR-3 codeword does not answer a PACTOR-1 call",
          stranger.arq.state == State.CONNECTING, f"state {stranger.arq.state}")
    # ...and the codeword the protocol DOES answer a call with still lands,
    # whatever protocol a previous link left this station transmitting in.
    stranger.protocol = Protocol.PACTOR3
    stranger.on_rx_event(_cs(pactor1.CS_ACK_A))
    check("...while CS1 brings the link up from either protocol state",
          stranger.arq.state == State.CONNECTED and stranger.p1_baud == 200,
          f"state {stranger.arq.state}, {stranger.p1_baud} Bd")
    stranger.log_lines.clear()
    stranger.arq.on_cycle()
    check("...and a refused codeword bought no retry either",
          not any("not counting this retry" in m for m in stranger.log_lines),
          str(stranger.log_lines))

    # A CALL IS NOT A FRESH PROCESS, and everything a link left behind is about
    # that link. The hostmode `C` command -- how Winlink Express and Pat drive this
    # modem -- calls twice on one PtcHost, and a second call used to open in the
    # first contact's waveform: PACTOR-3 rendered at a station whose connect burst
    # is PACTOR-1, with the field size and the ruled-out upgrades to match.
    again = PtcHost(peer=Seam(), mycall="W9SSJ")
    again.arq.on_host_connect("W9SSJ", "WS8EOC")
    again.on_rx_event(_cs(pactor1.CS_ACK_A))
    again.protocol = Protocol.PACTOR3                    # the link upgraded
    again._ruled_out.add(Protocol.PACTOR2)
    again._p1_first_block = True
    again.arq.on_host_abort()                            # ...and then it ended
    again.arq.on_host_connect("W9SSJ", "KB5LZK")         # a different station
    check("a call placed after an upgraded link opens in PACTOR-1",
          again.protocol is Protocol.PACTOR1 and again.p1_baud == 100,
          f"{again.protocol} at {again.p1_baud} Bd")
    check("...with the field size, the first-block exception and the ruled-out "
          "upgrades to match",
          (again.arq.payload_bytes_override == pactor1.DATA_FIELD[100]
           and not again._p1_first_block and not again._ruled_out),
          f"{again.arq.payload_bytes_override} B, first block "
          f"{again._p1_first_block}, ruled out {again._ruled_out}")

    # A CONNECT BURST NAMES THE CALLED PARTY AND NOBODY ELSE, which is why the
    # receiver has no caller to pass on. That is enough to answer while
    # LISTENING; while we are calling it says only that somebody wants us, and
    # taking it dropped a call that was on the air for a station we could not
    # name -- `** CONNECTED to **`, with an empty dxcall.
    called_back = PtcHost(peer=Seam(), mycall="W9SSJ")
    called_back.arq.on_host_connect("W9SSJ", "WS8EOC")
    sent.clear()
    called_back.on_rx_event(rxfront.Event(
        0.0, "connect", "###CONNECT: [Normal Call: W9SSJ]",
        connect=p1rx.Connect("Normal", "W9SSJ", False)))
    check("a connect burst naming us does not take over a call in progress",
          called_back.arq.state == State.CONNECTING
          and called_back.arq.dxcall == "WS8EOC",
          f"state {called_back.arq.state}, dxcall {called_back.arq.dxcall!r}")
    check("...and nothing was sent to it", not sent, str(sent))
    # The station we called is the exception, and the only one: a peer that
    # answers a call by calling back is answering US.
    called_back.arq.on_rx_connect("WS8EOC", "W9SSJ")
    check("...while the station we called is still admitted",
          called_back.arq.state == State.CONNECTED
          and called_back.arq.dxcall == "WS8EOC",
          f"state {called_back.arq.state}, dxcall {called_back.arq.dxcall!r}")


# --------------------------------------------------------------------------
# transmit: the raster
# --------------------------------------------------------------------------

PACKET_N = round(spec.P1_PACKET_S * FS)
CS_N = round(spec.P1_CS_S * FS)
D_MAX_N = round((1.25 - 0.04 - spec.P1_PACKET_S - spec.P1_CS_S) * FS)


def _grid(anchor: int = 0, cycle: float = 1.25) -> "onair._MasterGrid":
    return onair._MasterGrid(anchor, round(cycle * FS),
                             round(onair.TX_OFFSET_S * FS),
                             packet_n=PACKET_N, cs_n=CS_N, d_max_n=D_MAX_N)


def the_master_holds_its_own_grid() -> None:
    """THE regression. A master's transmit instants answer to nothing but itself.

    This file used to assert the opposite -- that the raster tracks the peer's
    bursts to within 5 ms over 24 cycles -- and it passed, and the code was
    wrong. The responder is the follower, with about an eighth of a bit of authority
    per cycle: "Der SLAVE-Takt wird auf den MASTER-Takt synchronisiert". Two
    followers is not a link, and the failure is silent, so it is worth a test
    that fails the moment the loop is closed again.

    The peer here jitters AND drifts, which is exactly the input that would move
    a tracking anchor and must not move this one.
    """
    print("\nthe master's transmit grid against a peer that wanders")
    g = _grid()
    d_true = 0.085 * FS
    rng = np.random.default_rng(7)
    anchors, keys = [], []
    for k in range(1, 25):
        # Their answer to our packet in cycle k: the protocol's turnaround, plus
        # 3 ms rms of jitter, plus a clock 200 us a cycle away from ours -- the
        # 1.2498 against 1.2500 measured off the air.
        at = int(g.boundary(k) + PACKET_N + d_true
                 + rng.normal(0, 0.003 * FS) - k * 0.0002 * FS)
        g.update([at])
        anchors.append(g.anchor)
        keys.append(g.boundary(k + 1))

    check("the transmit anchor did not move, at all", len(set(anchors)) == 1,
          f"{len(set(anchors))} distinct anchors over 24 cycles")
    check("...and every transmission is exactly one cycle after the last",
          all(b - a == g.slot_n for a, b in zip(keys, keys[1:])),
          f"steps {sorted({b - a for a, b in zip(keys, keys[1:])})}")
    check("the receive window is what absorbed the wander", g.locked,
          f"d = {g.d_n / FS * 1e3:.1f} ms, from {d_true / FS * 1e3:.0f} nominal")
    # And what the tracking loop would have done with the same input: an eighth
    # of nothing is nothing, but 0.3 of a jittering error is a walking anchor.
    walk, err = 0.0, 0.0
    for k in range(24):
        err = rng.normal(0, 0.003 * FS) - 0.0002 * FS
        walk += 0.3 * err
    check("...where a peer-tracking anchor would have moved instead",
          abs(walk) > 0.002 * FS,
          f"{walk / FS * 1e3:+.1f} ms of our own transmit timing, imported from "
          f"the far end's jitter and then fed back to it")


def only_the_receive_window_is_steered() -> None:
    """`d` follows the peer at the protocol's gain; the anchor follows nothing."""
    print("\nthe turnaround gap, corrected at an eighth a cycle")
    g = _grid()
    # Two cycles, because the tracking loop does not start until the turnaround
    # is corroborated -- one control signal in the window is a candidate.
    for k in (0, 1):
        g.update([g.boundary(k) + PACKET_N + round(0.085 * FS)])
    check("the first control signal in the window sets the turnaround",
          g.locked and abs(g.d_n - 0.085 * FS) < 0.001 * FS,
          f"d = {g.d_n / FS * 1e3:.1f} ms")
    anchor = g.anchor

    # A STEP: the peer's turnaround moves 8 ms and stays there. A first-order
    # loop of gain 1/8 answers with 1 ms, then 1.9, then 2.6 ... and the point of
    # the test is that it is NOT 8, because a master that answers a step whole is
    # a master steering on one measurement.
    step = 0.008 * FS
    resp = []
    for k in range(2, 14):
        g.update([g.boundary(k) + PACKET_N + round(0.085 * FS + step)])
        resp.append(g.d_n - 0.085 * FS)
    check("the first cycle takes an eighth of the step, not the step",
          abs(resp[0] - step / 8) < 0.2e-3 * FS,
          f"{resp[0] / FS * 1e3:.2f} ms of an {step / FS * 1e3:.0f} ms step")
    check("...and it converges rather than ringing",
          all(a < b for a, b in zip(resp, resp[1:])) and resp[-1] < step,
          f"{resp[-1] / FS * 1e3:.2f} ms after 12 cycles")
    check("the anchor is untouched by any of it", g.anchor == anchor)

    # QRM in the window, outside the capture range: refused, and after
    # MAX_MISSES the window goes back to searching -- the anchor still does not
    # move, because nothing is allowed to move it.
    for _ in range(onair._MasterGrid.MAX_MISSES - 1):
        g.update([g.boundary(20) + PACKET_N + round(0.2 * FS)])
    check("a burst outside the capture range is refused, not followed",
          g.locked and g.anchor == anchor)
    g.update([g.boundary(23) + PACKET_N + round(0.2 * FS)])
    check("persistent disagreement releases the RECEIVE window only",
          not g.locked and g.anchor == anchor)

    # ...and the corroboration's CYCLE goes with it. The acquisition line names
    # the cycle the evidence closed in, and a re-acquisition that inherited the
    # first acquisition's read as a lock corroborated twenty cycles before it
    # happened -- which is how three sessions against a gateway whose raster was
    # walking read as one acquisition and one alias of it.
    released = g.cycles
    for k in (26, 27):
        line = g.update([g.boundary(k) + PACKET_N + round(0.085 * FS)])
    check("a re-acquisition names the cycle IT closed in, not the first one's",
          g.locked and g.evidence.at is not None and g.evidence.at > released,
          line or "")


def the_grid_is_placed_only_off_the_air() -> None:
    """Where the one anchor movement in the whole design is allowed to happen."""
    print("\nplacing the grid: the only move that touches the transmit anchor")
    g = _grid()
    cold = g.anchor
    # Heard while we are transmitting: this is our own receive window, so the
    # burst is an ANSWER. It sets `d` and nothing else, however far out it is.
    g.update([g.boundary(1) + PACKET_N + round(0.12 * FS)])
    check("a burst in our own window is an answer, and moves nothing",
          g.anchor == cold and g.locked,
          f"d = {g.d_n / FS * 1e3:.0f} ms")

    # Heard with our transmitter off, having never had an answer: this is a
    # station on its own raster and the only thing to do is decide where to call.
    g = _grid()
    at = round(0.44 * FS)
    line = g.update([at], hushed=True, since_tx=g.slot_n)
    check("a burst heard during a hush places the grid", g.anchor != cold,
          line or "")
    nxt = g.boundary_after(at)
    check("...so that our packet falls in the gap after one of theirs",
          abs((nxt - at) / FS - onair.TX_OFFSET_S) < 1 / FS,
          f"{(nxt - at) / FS * 1e3:.1f} ms after its onset, aiming for "
          f"{onair.TX_OFFSET_S * 1e3:.0f}")
    # ...and once there is a turnaround, a hush cannot re-place it either.
    for k in (0, 1):
        g.update([g.boundary_after(at) + k * g.slot_n
                  + PACKET_N + round(0.085 * FS)])
    placed = g.anchor
    g.update([round(0.9 * FS)], hushed=True, since_tx=g.slot_n)
    check("but a grid that has been answered is never re-placed",
          g.anchor == placed)

    # THE FIRST CYCLE OF A HUSH IS NOT OFF THE AIR YET. Its window opens where
    # our own carrier dropped, not a cycle later: `_acquisition_window` searches
    # it as the answer band and 7 of the 74 connect candidates on file were
    # accepted in one. So a burst in it is the peer answering our last call, and
    # placing on it is the master retiming towards the peer -- 10 of the 25
    # placements on record were made from a first hush cycle, six of them from a
    # burst 53-76 ms past our carrier.
    g = _grid()
    cold = g.anchor
    answer = g.boundary(1) + PACKET_N + round(0.090 * FS)
    line = g.update([answer], hushed=True, since_tx=round(0.010 * FS))
    check("a hush whose window opens in our own answer band places nothing",
          g.anchor == cold, line or "")
    check("...it reads the burst as the answer to our last call",
          g.acquired and abs(g.d_n / FS - 0.090) < 0.005,
          f"d = {g.d_n / FS * 1e3:.0f} ms" if g.d_n is not None else line or "")


def the_turnaround_search_stops_where_we_have_to_key() -> None:
    """The span `d` is searched over, and why it is that span and not wider."""
    print("\nthe turnaround search span")
    g = _grid()
    # 130 ms at the FT-891's 0.04 settle: 0.960 + d + 0.120 <= 1.250 - 0.040.
    # A reference implementation's own search loop stops at 139 ms with a 30 ms
    # lead-in, which is the same arithmetic to a millisecond -- past it, their
    # control signal is still going when our carrier comes up. Asked of
    # `_budget`'s own band rather than of this file's arithmetic, which would
    # only compare D_MAX_N against its own definition.
    _, budget = onair._budget(1.25, onair.TX_OFFSET_S, 0.04)
    latest = float(re.search(r"whole for d \d+-(\d+) ms", budget).group(1))
    check("the span is the budget's own latest-serviceable turnaround",
          abs(D_MAX_N / FS * 1e3 - latest) <= 0.5,
          f"{D_MAX_N / FS * 1e3:.0f} ms, budget says {latest:.0f}")
    g.update([g.boundary(1) + PACKET_N + D_MAX_N + round(0.02 * FS)])
    check("a turnaround we could not hear out is not acquired", not g.locked)
    g.update([g.boundary(2) + PACKET_N + D_MAX_N - round(0.005 * FS)])
    check("...and one just inside it is", g.locked,
          f"d = {g.d_n / FS * 1e3:.0f} ms")


def an_impossible_turnaround_is_not_acquired() -> None:
    """The floor under the search, and the night the search had none.

    2026-08-10, captures/onair-0810-2201: the grid latched d = 39.1 ms off one
    fresh, corroborated onset, while the same session's real answers from WS8EOC
    measured 94-100 ms. The receive window then opened on an echo and
    `key_refusal` measured every boundary against it, so a live link went down
    at hold cycle 20. Nothing in the acquisition had ever asked whether the gap
    it was latching could be a turnaround at all.
    """
    print("\nthe floor under the turnaround search")
    g = _grid()
    bad = round(0.0391 * FS)
    line = g.update([g.boundary(1) + PACKET_N + bad])
    check("a 39.1 ms gap is not latched as a turnaround", not g.locked,
          f"d_n {g.d_n}")
    check("...and the cycle is reported blind, naming the band an answer has to "
          "fall in", "NO CONTROL SIGNAL" in (line or "")
          and "between 40 and" in (line or ""), line or "")

    # The rejection must not end the search: a cycle that offers our own echo
    # first and the peer second still has the peer in it.
    g = _grid()
    good = round(0.096 * FS)
    g.update([g.boundary(1) + PACKET_N + bad, g.boundary(1) + PACKET_N + good])
    check("...and the real answer behind it in the same cycle is still acquired",
          g.locked and abs(g.d_n - good) < 1e-6,
          f"d = {g.d_n / FS * 1e3:.1f} ms" if g.locked else "nothing latched")

    # THE OTHER HALF. The floor is 40 ms because that is what a shipped responder
    # implementation transmits, so a peer that is merely fast is a peer -- and
    # it is answered where it reads, 170 ms - d past its own data end, like
    # any other turnaround the search accepts.
    g = _grid()
    fast = round(onair.D_MIN_S * FS)
    for k in (1, 2):
        line = g.update([g.boundary(k) + PACKET_N + fast])
    check("a peer at the shipped floor is acquired rather than thrown away",
          g.locked and abs(g.d_n - fast) < 1e-6, line or "")
    # Asked as the emission path asks it: on the IRS side, about the CARRIER a
    # settle in front of the boundary, and with the whole interval the rig is
    # keyed for. Sending, this grid has no decoded word and no opinion at all.
    g.reverse(to_iss=False)
    air_n = SETTLE_N + CS_N
    why = g.key_refusal(g.boundary(3) - SETTLE_N, air_n)
    check("...and the acknowledgement it gets is clear of its own packet",
          why is None, str(why))


def the_anchored_read_is_aimed_at_a_slot_we_keyed() -> None:
    """The receive instant is OUR OWN data end plus `d`, or there is not one.

    2026-08-14, the night the operator heard it first: "we acted like the flow
    direction reversed but nothing was going on on-channel". Two of the four
    sessions that reached a peer changed over on a zero-error codeword read at
    the receive instant of a slot they had spent HUSHED -- an answer to a packet
    that never went out. rig-session-20260814-211608 read `rx_due(5)` 1.06 s
    into the 2.045 s captures/onair-0814-2116/hold_01 and reversed on a CS3
    against a peer whose turnaround measured 81-88 ms; -212300 read `rx_due(7)`
    1.05 s into the 1.192 s onair-0814-2123/hold_01. Both figures were reported
    as turnarounds and neither was one: they are offsets into the segment, and
    they agree across two gateways because they are one cycle of the raster plus
    the usual 0.09.

    The instant was found from the grid's period -- the first one at or after
    the window's start, whatever slot it belonged to -- which is the right
    answer whenever the window opens just behind our own carrier and a different
    one whenever it does not. A hushed cycle collects to the boundary rather
    than to the key, so its window runs about 2 s and reaches back over a slot
    nothing of ours was in.
    """
    print("\nthe receive instant, against the slot our carrier came up in")
    g = _grid()
    g.d_n = round(0.09 * FS)
    # The geometry of captures/onair-0814-2116/hold_01, scaled to this grid: a
    # window that opens before slot 5's receive instant and closes after it,
    # with slot 6's still ahead of its end.
    lo = g.boundary(5) - round(0.05 * FS)
    hi = lo + round(2.045 * FS)
    check("a slot we never keyed offers no receive instant",
          g.rx_due_in(lo, hi) is None, f"{g.rx_due_in(lo, hi)}")

    g.keyed_slot = 5
    at = g.rx_due_in(lo, hi)
    check("...and the slot our carrier came up in offers its own",
          at == g.rx_due(5),
          f"{at} against rx_due(5) = {g.rx_due(5)}")

    # rig-session-20260814-210924, the one legitimate reversal of that night:
    # the read that answered TX[26] was 0.09 s into onair-0814-2109/hold_06 and
    # slot 49 is where that carrier came up. Nothing about it changes.
    g.keyed_slot = 6
    check("a window that has run past the keyed slot's instant holds none",
          g.rx_due_in(g.rx_due(6) + 1, hi) is None,
          f"{g.rx_due_in(g.rx_due(6) + 1, hi)}")
    check("...and one that cannot hold the whole codeword holds none either",
          g.rx_due_in(lo, g.rx_due(6) + g.cs_n - 1) is None)


def the_edge_statistic_reads_real_bursts() -> None:
    """The loop's error signal, on off-air audio rather than on synthesis.

    It is a residual against a prediction, which is what separates it from the
    transition-derived bit phase tried as a DECODE origin and rejected for
    measuring worse than the envelope onset. Nothing here has to be right first
    time: an eighth of it is applied per cycle.
    """
    print("\nthe sub-bit edge statistic, against a real gateway's bursts")
    audio = session.load_wav(str(PEER_WAV), FS)
    onsets = rxfront.p1_burst_onsets(audio)[:12]
    at = [p1rx.cs_time_dev(audio, t) for t in onsets]
    got = [d for d in at if d is not None]
    check("every burst the detector found yields a deviation",
          len(got) == len(onsets), f"{len(got)} of {len(onsets)}")
    check("...bounded to half a bit, as the wrap requires",
          all(abs(d) <= 0.005 + 1e-9 for d in got),
          f"worst {max(abs(d) for d in got) * 1e3:.2f} ms")
    # Predict the same burst 3 ms early and the residual must grow by 3 ms: the
    # measurement is of the PREDICTION, not of the burst.
    #
    # MODULO A BIT, and the wrap is not a caveat to be apologised for -- it is
    # what the estimator is. A burst already reading 4.8 ms late reads 7.2 ms
    # EARLY when the prediction moves 3 ms earlier, because a control signal is
    # twelve identical-length symbols and nothing inside it says which bit you
    # are looking at. That ambiguity is the protocol's: "+/-10 ms is a whole bit
    # slip and is unrecoverable", and a slip is the burst detector's business,
    # not this function's.
    #
    # 1.5 ms of tolerance, against a measured worst of 1.2 ms on the noisier of
    # the two silent recordings -- see the note in `cs_time_dev` for where that
    # comes from and what was tried to reduce it. An eighth of it reaches the
    # receive window.
    step = p1rx.CS_BIT_S
    shifted = [p1rx.cs_time_dev(audio, t - 0.003) for t in onsets]
    both = [(a, b) for a, b in zip(at, shifted) if a is not None and b is not None]
    worst = max(abs(((b - a) - 0.003 + step / 2) % step - step / 2) for a, b in both)
    check("a prediction 3 ms early reads 3 ms late, modulo a bit", worst < 1.5e-3,
          f"worst {worst * 1e3:.2f} ms from the 3.00 expected, over "
          f"{len(both)} bursts")
    check("nothing is claimed where there is no burst",
          p1rx.cs_time_dev(np.zeros(int(0.5 * FS), np.float32), 0.2) is None)


class _Tape:
    """The reader's position over a recording, wearing the production clamp.

    `_keyable_slot`'s docstring records four spellings of the keyable-slot rule
    disagreeing about the holdback, and the 2026-07-27 every-2.5-second session
    they caused; a fifth spelling lived in this file, so the scenes below ran a
    model of the scheduler and could not fail with the scheduler broken. The
    bench therefore supplies only what the production search asks of a stream --
    the read position, the notice, the clamp -- and the notice is `_LiveInput`'s
    own arithmetic at a file's figures: audio is delivered the instant it is
    read, so there is no latency and no blocks in flight, and placeable means
    simply "not already past".
    """
    _lat, _blk = 0, 0

    def __init__(self, pos: int):
        self.pos = pos

    @property
    def samples(self) -> int:
        return self.pos

    tx_latency_n = 0
    key_notice = onair._LiveInput.key_notice
    clamp_late = onair._LiveInput.clamp_late


def _schedule(audio: np.ndarray, anchor: int, cycles: int, *,
              escape: bool) -> dict:
    """The session loop's window arithmetic over real audio, with the blanking.

    `--replay` cannot show this and never will: a dry run's transmission costs
    the file no samples, so the receiver is never deaf and the peer's burst lands
    in the window whatever the phase. Being deaf is the entire failure. Here the
    recording is the channel and our own transmission cuts the hole in it that a
    keyed rig would -- carrier up on the boundary, down a packet later, and the
    reader picks up where it dropped.

    One slot a cycle throughout, which is the cadence a locked or connected
    session runs at. That is where the deadlock lives: 1.25 s less our 0.96 s
    packet and the keying settle leaves a 250 ms window, and if the peer's
    120 ms burst is not in it there is nothing to steer by and nothing that
    will move.

    The slot is picked by `onair._keyable_slot` -- the one place the rule
    lives -- not by an arithmetic of this file's own; see `_Tape`.
    """
    r = _grid(anchor)
    if not escape:
        r.BLIND_CYCLES = cycles + 1        # how it behaved before the escape
    live, slot = _Tape(anchor + PACKET_N), 1
    dropped = live.pos                     # where our carrier last came down
    out = {"hushed": 0, "heard": 0, "lock_cycle": None, "anchors": [],
           "windows": [], "locked_heard": 0, "locked_cycles": 0}
    for c in range(1, cycles + 1):
        hush = r.hush_left > 0
        slot = onair._keyable_slot(live, r, slot, SETTLE_N)
        boundary = r.boundary(slot)
        end = boundary if hush else boundary - SETTLE_N
        seg, seg_start = audio[live.pos:end], live.pos
        since_tx = seg_start - dropped
        out["windows"].append((end - live.pos) / FS)
        onsets = [at for at, _ in onair._peer_bursts(seg, seg_start)]
        out["heard"] += len(onsets)
        if r.locked and not hush:
            out["locked_cycles"] += 1
            out["locked_heard"] += bool(onsets)
        if hush:
            out["hushed"] += 1
            live.pos, slot = boundary, slot + 1
        else:
            out["anchors"].append(r.anchor)  # what each keyed cycle ran on
            live.pos = dropped = boundary + PACKET_N   # where our carrier drops
            slot += 1
        # Folded at the end of the cycle, where the run loop folds it and where
        # the audio the onsets came from is still in hand.
        r.update(onsets, seg, seg_start, hushed=hush, since_tx=since_tx)
        if r.locked and out["lock_cycle"] is None:
            out["lock_cycle"] = c
    return out


def deadlock_and_escape() -> None:
    """The failure the escape exists for, and the escape breaking it."""
    print("\nthe deadlock: our own transmission covering the burst the lock needs")
    audio = session.load_wav(str(PEER_WAV), FS)
    # Put the peer's burst squarely inside our own transmission. The anchor is
    # half a packet ahead of the first burst, so every one of them lands 0.5 s
    # into the 0.96 s our carrier is up -- and stays there, because the peer's
    # raster and ours are both steady to a fraction of a millisecond a cycle.
    # That stability is the whole trap: a phase that is wrong stays wrong.
    anchor = int((rxfront.p1_burst_onsets(audio)[0] - 0.5) * FS)
    # Eighteen cycles is 22 s, and it stops there because the recording does:
    # from 26 s the gateway stops calling on its 1.25 s raster and sends groups
    # 0.24 s apart, which lands a burst in any window at all. That is a real
    # thing a gateway does and it is not the thing under test.
    cycles = 18

    stuck = _schedule(audio, anchor, cycles, escape=False)
    check("without the escape the session never hears the peer at all",
          stuck["heard"] == 0 and stuck["lock_cycle"] is None,
          f"{stuck['heard']} onsets in {cycles} cycles, "
          f"every window {min(stuck['windows']):.3f} s wide and in the wrong "
          f"place")

    free = _schedule(audio, anchor, cycles, escape=True)
    check("the escape goes quiet and finds a raster to call on",
          free["lock_cycle"] is not None,
          f"acquired on cycle {free['lock_cycle']} after {free['hushed']} "
          f"cycles off the air")
    check("...and it did so by listening, not by transmitting differently",
          0 < free["hushed"] <= onair._MasterGrid.HUSH_CYCLES,
          f"{free['hushed']} hushed of {onair._MasterGrid.HUSH_CYCLES} allowed")
    check("...within a cycle or two of the threshold it is allowed",
          free["lock_cycle"] <= onair._MasterGrid.BLIND_CYCLES + 5,
          f"cycle {free['lock_cycle']}, threshold "
          f"{onair._MasterGrid.BLIND_CYCLES}")
    # Having placed the grid, the one-slot window is supposed to hold the peer's
    # burst every cycle -- that is what TX_OFFSET_S is for, and it is the claim
    # that makes a one-slot cadence viable at all.
    check("once acquired the one-slot window holds the peer's burst every cycle",
          free["locked_cycles"] and free["locked_heard"] == free["locked_cycles"],
          f"{free['locked_heard']} of {free['locked_cycles']} locked cycles")
    # THE POINT OF THE WHOLE EXERCISE. The escape is a DECISION, taken once with
    # the transmitter off, not a tracking loop: the anchor changes at the hush and
    # never again, however far the peer's bursts wander afterwards.
    moves = sum(a != b for a, b in zip(free["anchors"], free["anchors"][1:]))
    check("...and the transmit anchor changed exactly once, at the hush",
          moves == 1 and len(set(free["anchors"][-6:])) == 1,
          f"{moves} change(s) over {len(free['anchors'])} transmitting cycles")


def _blind_for(g: "onair._MasterGrid", cycles: int, **how) -> int:
    """Run `cycles` of hearing nothing through the real grid; return hushed cycles.

    Through `update`, the loop's own entry point, and counting the cycles the
    loop would have read as hushed -- `hush_left > 0` before the cycle is folded
    in, which is where the loop takes it. Nothing here re-implements the count.
    """
    hushed = 0
    for _ in range(cycles):
        hushed += g.hush_left > 0
        g.update([], **how)
    return hushed


def the_hush_is_only_for_a_grid_nothing_has_ever_answered() -> None:
    """Who may be taken off the air, and who may not.

    The hush buys one thing: a grid whose phase has never been checked against
    anything gets placed by what it hears. It cannot buy that twice. A grid that
    has measured a turnaround heard the peer inside its own receive window with
    our carrier up in the same cycle, so the phase is not one that hides it; a
    session holding a link has a peer timing 1.25 s cycles against it, and six of
    them off the air is several cycles past the point an IRS gives up. In both
    states a hush can only drop a station that IS answering.

    MEASURED, 2026-08-06: four sessions against three gateways, keyed 4 cycles in
    every 10 from the first blind cycle to the last, on a raster that never once
    acquired -- including every cycle of the held link.
    """
    print("\nthe hush: armed only while nothing has been heard on this grid")
    HUSH = onair._MasterGrid.HUSH_CYCLES
    n = onair._MasterGrid.BLIND_CYCLES + HUSH + 4

    cold = _blind_for(_grid(), n)
    check("a grid nothing has ever answered still goes quiet", cold > 0,
          f"{cold} of {n} cycles off the air")

    # A LINK IS UP. Same grid, same silence, and the only difference is the one
    # bit the loop reads off the FSM.
    linked = _grid()
    check("a session holding a link is never taken off the air",
          _blind_for(linked, n, linked=True) == 0,
          f"hush_left {linked.hush_left} after {n} blind cycles")

    # ...AND A TURNAROUND WAS ONCE MEASURED. `d` is released after MAX_MISSES and
    # `locked` goes back to False with it, which is why the latch is a separate
    # fact: the peer having been heard is not undone by losing it again.
    was = _grid()
    was.update([was.boundary(1) + PACKET_N + round(0.085 * FS)])
    check("...and one that has heard an answer before is not either",
          was.acquired and not _blind_for(was, n),
          f"acquired {was.acquired}, hush_left {was.hush_left}")
    check("...even though its receive window was long since released",
          not was.locked, f"d_n {was.d_n}")

    # THE COUNTEREXAMPLE, PLANTED. Both rules read one branch of `_blind`, off
    # two inputs; take those two inputs away and leave every other line of the
    # production arithmetic standing, and the same grids are hushed again --
    # which is what the sessions above flew.
    saved = onair._MasterGrid._blind

    def as_flown(self, why="nothing heard", *, linked=False):
        self.acquired = False
        return saved(self, why, linked=False)

    onair._MasterGrid._blind = as_flown
    try:
        again = _grid()
        again.update([again.boundary(1) + PACKET_N + round(0.085 * FS)])
        flown = (_blind_for(_grid(), n, linked=True), _blind_for(again, n))
    finally:
        onair._MasterGrid._blind = saved
    # AT LEAST A WHOLE HUSH EACH, not exactly one: `update` cancels an
    # in-flight hush the moment a link comes up (see its `linked` branch), and
    # the patch above leaves that cancellation standing while taking the branch
    # in `_blind` away -- so the linked grid re-arms every cycle rather than
    # draining once. What the control is for is that both go off the air, and
    # both spend a full hush doing it.
    check("NEGATIVE CONTROL: without the rule both are hushed, as they flew",
          all(h >= HUSH for h in flown),
          f"{flown[0]} cycles off the air while linked, {flown[1]} after an "
          f"acquisition, against {HUSH} a hush is worth")


def one_slot_cadence_is_reachable() -> None:
    """The floor on a listen window, which used to make one slot impossible.

    Asked of `onair._keyable_slot` itself, standing where a keyed cycle stands:
    our carrier just dropped, the next boundary one listen window away. What
    the window has to be long enough for is the thing it is listening for --
    the 120 ms control signal -- and gating it on the protocol's whole 290 ms
    answer window instead skipped a slot every cycle, so the peer's own rate
    was unreachable.
    """
    print("\nthe listen-window floor against the one-slot cadence")
    r = _grid()
    live = _Tape(r.boundary(1) + PACKET_N)     # our carrier just dropped
    got = onair._keyable_slot(live, r, 2, SETTLE_N)
    readable = (r.boundary(2) - SETTLE_N - live.pos) / FS
    check("the very next slot is keyable, so the peer's own rate is reachable",
          got == 2, f"slot {got}, {readable * 1e3:.0f} ms to listen in")
    check("...with less than the protocol's whole answer window to do it in",
          readable < spec.CS_WINDOW_S,
          f"{readable * 1e3:.0f} ms readable, CS_WINDOW_S is "
          f"{spec.CS_WINDOW_S * 1e3:.0f}")
    # The counterexample, through the same production search: put the old floor
    # back as the raster's own requirement and the slot is skipped.
    wide = _grid()
    wide.p1_cs_n = round(spec.CS_WINDOW_S * FS)
    check("NEGATIVE CONTROL: gating on the whole answer window skips the slot",
          onair._keyable_slot(_Tape(r.boundary(1) + PACKET_N), wide, 2,
                              SETTLE_N) > 2)


def blindness_is_reported() -> None:
    """A grid with nothing answering it must say so, every cycle."""
    print("\nsilence about having no control signal")
    r = _grid()
    lines = [r.update([]) for _ in range(3)]
    check("every blind cycle returns a line for the log", all(lines),
          repr(lines[-1]))
    check("...and it names the fault rather than the symptom",
          all("NO CONTROL SIGNAL" in ln for ln in lines))
    # Hearing something in the wrong place is still blindness, and saying "3
    # bursts heard" while calling it blind is the honest report: the fault is
    # that none of them can be an answer, not that the band is quiet.
    line = r.update([r.boundary(1) + PACKET_N + round(0.4 * FS)])
    check("a burst nowhere an answer could be counts as blind, and says why",
          line is not None and "NO CONTROL SIGNAL" in line and "burst" in line,
          line or "")
    # An answer clears the count: the escape is for a session that cannot get
    # answered, not for one that has and then lost a burst to fading.
    r.update([r.boundary(2) + PACKET_N + round(0.085 * FS)])
    check("acquiring resets the blind count", r.blind == 0 and r.hush_left == 0)


def _band(line: str, edge: int | None = None) -> float:
    """The `d` range a budget line says it holds an answer whole in, in ms.

    Its width, or one edge of it -- `edge=0` for the bottom, `edge=1` for the top.
    """
    lo, hi = (float(x) for x in
              re.search(r"whole for d (-?\d+)-(-?\d+) ms", line).groups())
    return (lo, hi)[edge] if edge is not None else hi - lo


def budget_arithmetic() -> None:
    print("\nthe cycle budget, stated rather than assumed")
    fits, line = onair._budget(1.25, onair.TX_OFFSET_S, 0.04)
    check("the default offset closes the cycle at the FT-891's 0.04 settle",
          fits and "OUTSIDE" not in line, line)
    # 0.10 was once said to make the cycle "impossible", from a budget that had us
    # waiting out the peer's burst before keying. A master keys on its own grid, so
    # what a longer settle really costs is keyed time and therefore the RANGE of
    # peer turnarounds that still fit -- retracted 2026-07-28.
    #
    # THE RETRACTION WENT ONE STEP TOO FAR, and this is the correction, measured
    # rather than argued. "It narrows the band, it does not shut it" was asserted
    # here for a fortnight and it is false at 0.100: the band closes. A codeword
    # has to be READ as well as heard out, which needs `ACQUIRE_TAIL_S` more audio
    # than the 55-70 ms a 0.100 settle leaves, so `_acquisition_window` hands that
    # schedule nothing. It narrows to zero, and the operator hears "RX (nothing
    # decoded)" every cycle of a call nobody could have answered.
    #
    # A measured peer answers at 87-134 ms (`onair.PEER_TURNAROUND_S`), so 55-70
    # would have held none of them even if it could be read. The 0.100 settle is
    # the g90; the 0.400 is the x6100. Neither can call a PACTOR gateway.
    fits, line = onair._budget(1.25, onair.TX_OFFSET_S, 0.10)
    wide = onair._budget(1.25, onair.TX_OFFSET_S, 0.04)[1]
    check("a 0.10 s settle does not close -- it shuts the band, not merely narrows it",
          not fits and "NONE of which is long enough to read" in line, line)
    check("...and the band it leaves would not have held a real answer anyway",
          _band(line) < _band(wide)
          and _band(line, 1) < onair.PEER_TURNAROUND_S[0] * 1e3,
          f"{_band(line):.0f} ms wide, topping out at {_band(line, 1):.0f} ms "
          f"against a peer at {onair.PEER_TURNAROUND_S[0] * 1e3:.0f} ms and later")
    # Out-of-band is a bootstrap worth announcing, not a reason to refuse to key:
    # the first measured answer replaces it. Refusing here is how a modem that
    # would have worked never transmits at all.
    _, line = onair._budget(1.25, 0.13, 0.04)
    check("an offset implying a turnaround past the window says so",
          "OUTSIDE" in line and "still running when we key" in line, line)


def collision_reporting_is_measured_not_inferred() -> None:
    """The peer's burst against OUR CARRIER, which is the interval that decides it.

    Three separate analyses of one on-air collision reached three different
    answers, two confidently wrong, because every figure available was relative to
    a slot boundary or a window edge -- where we meant to transmit -- and none was
    relative to the carrier. This pins the arithmetic against geometries whose
    answer is known by construction.
    """
    from hfmodem.shrike import pactor1, rxfront, spec
    from hfmodem.shrike.onair import (_MasterGrid, _forecast_next_key,
                                      _report_collision, FS)

    print("\nthe peer's burst against our carrier, measured rather than inferred")
    burst = round(spec.P1_CS_S * FS)
    slot = round(spec.CYCLE_SHORT_S * FS)

    class _Tx:
        settle = 0.04

        def __init__(self, start, end):
            self.tx_key_up, self.tx_end = start, end
            # Where the audio was aimed, so `boundary - tx_key_up` is the lead
            # the carrier actually took. The forecast reads that rather than the
            # nominal settle: measured on this station they differ by 12 ms, and
            # the one -11 ms `INTO IT` of 2026-08-26 is the whole of it.
            self.boundary = start + round(self.settle * FS)

    def _grid(offset, onset):
        # Placed against that burst -- the one path that sets the anchor -- so
        # the geometry the prediction is made against is the known one.
        r = _MasterGrid(0, slot, round(offset * FS), packet_n=PACKET_N,
                        cs_n=CS_N, d_max_n=D_MAX_N)
        r.update([onset], hushed=True, since_tx=slot)
        return r

    def run(onset, rf_start, rf_end, offset=0.185):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _report_collision([(onset, round(spec.P1_CS_S * FS))],
                              _Tx(rf_start, rf_end))
        return buf.getvalue()

    class _Rx:
        """Only what the forecast reads: the words the FSM was handed."""

        def __init__(self, *events):
            self.cs_log = list(events)

    def _word(at, cs=pactor1.CS_ACK_A):
        return rxfront.Event(at / FS, "cs", "CS1/anchor", protocol="PACTOR-1",
                             cs=cs)

    def forecast(onset, rf_start, rf_end, offset=0.185, rx=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _forecast_next_key(rx if rx is not None else _Rx(_word(onset)),
                               _Tx(rf_start, rf_end), _grid(offset, onset), 0)
        return buf.getvalue()

    # Their 122 ms burst starting 60 ms before our carrier: 62 ms of it under us.
    txt = run(onset=10 * FS, rf_start=10 * FS + burst - round(0.062 * FS),
              rf_end=11 * FS)
    check("a burst half under our carrier reads as overlap, not as clear",
          "collide" in txt and "62 ms" in txt, txt.strip().splitlines()[0])

    # Their burst wholly before we key: clear, and the distance is stated.
    txt = run(onset=10 * FS, rf_start=11 * FS, rf_end=12 * FS)
    check("a burst that finished before we keyed reads as clear",
          "[clear]" in txt and "before our carrier came up" in txt,
          txt.strip().splitlines()[0])

    # THE MEASUREMENT NO LONGER CARRIES A FORECAST. `[predict]` extrapolated
    # `max(onsets)` -- an energy shape the corpus has firing on VARA, on PACTOR-2
    # and on a 500 Hz-class ARQ station -- one cycle forward and printed a
    # verdict on it. Against the KiwiSDR witness of 2026-08-26 its one alarm in
    # 22 forecasts was wrong by 124 ms, and it was silent on the twenty
    # codewords that session did transmit over.
    txt = run(onset=10 * FS, rf_start=11 * FS, rf_end=12 * FS)
    check("an unattributed burst gets no forecast at all",
          "predict" not in txt, txt.strip())
    check("...and neither does a cycle whose readers found no codeword",
          not forecast(10 * FS, 11 * FS, 12 * FS, rx=_Rx()).strip())

    # With a codeword under it the forecast is a statement about a station: the
    # anchor is adopted from this very burst, so the next boundary sits offset
    # past it and keying a lead early still clears the 120 ms codeword.
    pred = forecast(10 * FS, 11 * FS, 12 * FS).strip()
    check("a codeword we read is forecast onto the next cycle", "clear" in pred,
          pred or "(none)")
    check("...and the line names the word it rests on", "CS1" in pred, pred)

    # A grid whose offset cannot clear their codeword must say INTO IT rather
    # than report a boundary it met. 0.10 s < their 0.120 s codeword.
    pred = forecast(10 * FS, 11 * FS, 12 * FS, offset=0.10).strip()
    check("an offset too small to clear their codeword is called out",
          "INTO IT" in pred, pred or "(none)")


# Which stages need the gateway recording, and which need nothing but this
# package. The two were one test behind one skipif, so an absent `silent.wav` --
# which is every installed wheel, the recording being uncommitted -- took the
# whole file down with it, the 30-to-3 dB sweep included. Only the two stages
# below read it.
SYNTHETIC = (expected_p1_packet, header_gate, packet_reaches_the_arq_layer,
             the_master_holds_its_own_grid, only_the_receive_window_is_steered,
             the_grid_is_placed_only_off_the_air,
             the_turnaround_search_stops_where_we_have_to_key,
             an_impossible_turnaround_is_not_acquired,
             the_anchored_read_is_aimed_at_a_slot_we_keyed, blindness_is_reported,
             the_hush_is_only_for_a_grid_nothing_has_ever_answered,
             one_slot_cadence_is_reachable, budget_arithmetic,
             collision_reporting_is_measured_not_inferred)
RECORDED = (the_edge_statistic_reads_real_bursts, deadlock_and_escape)


def _run(stages) -> int:
    global ok
    ok = True
    for stage in stages:
        stage()
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def main() -> int:
    return _run(SYNTHETIC + RECORDED)


def test_burstlock() -> None:
    assert _run(SYNTHETIC) == 0


@pytest.mark.skipif(
    not PEER_WAV.exists(),
    reason=f"the gateway recording is not at {PEER_WAV} — it lives in the working "
           "record, which does not cross the publication boundary")
def test_burstlock_against_a_real_gateway() -> None:
    assert _run(RECORDED) == 0


if __name__ == "__main__":
    raise SystemExit(main())
