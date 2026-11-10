# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The peer leads into PACTOR-2, and the session follows it as far as it can.

The gap this closes was predicted by the code that had it. `_SessionRx._readers`
carried two entries and a note saying why the third was missing -- `p2rx` reported
carriers and built no frame, and presence by this project's rule can never drive a
link -- and on 2026-08-03 an ARQ session got its data phase, handed the channel
over with a changeover, and heard the gateway answer in something it could measure
and not read. The frame decode landed the same night. This is the wiring.

FOUR CLAIMS, and they are separable on purpose:

  * the READER is where the speculative scan will reach it. A PACTOR-1 link runs
    the PACTOR-2 pass behind its own carrier, never in front of the key, and never
    ahead of the protocol the cycle's own answer depends on.
  * a PACTOR-2 frame in the window reaches `ptc.PtcHost`, carrying its payload and
    naming its protocol -- and the host FOLLOWS the peer into it, as of
    2026-09-02: `ptc.TRANSMITTABLE` holds PACTOR-2 now, so a peer that has keyed
    the waveform is answered in it rather than in the one protocol we could not
    leave. This asserted the opposite for as long as the codewords were
    unrendered.
  * once the link is in PACTOR-2 the frame moves to the head of `_readers`, where
    the cycle's own answer depends on it, and the ANSWER SLOT is read beside it:
    every rung, both frame lengths, both carrier arrangements.
  * the COST fits the cycle. This is the constraint the whole receive path is
    currently being rebuilt against: `rxfront.decode_events` stepped 27-fold at
    0.85 s of buffer and a session that misses its 1.25 s raster is dropped by the
    gateway. A reader that finds the peer's data and loses the cycle it arrived in
    has not helped.

The synthetic arm proves the WIRING and nothing about the waveform: its burst is
built by `pactor2.frame`, read by `p2rx`, and a shared misreading would be
invisible in it. The real arm is what carries the waveform, on captures whose
fields are byte-exact against the reference fields.

Run:  python -m hfmodem.tests.shrike.test_p2session
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import compress, onair, p2rx, pactor2, rxfront, session, spec
from hfmodem.shrike.arq import IRS, P2_LADDER
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.kestrel import corpora

FS = rxfront.FS
CYCLE_S = 1.25
WINDOW_S = 1.30
"""The window the cycle actually hands a scan, and the one every cost below is
measured over -- the same length `_SessionRx.deep_scan` and `upgrade_scan` are
budgeted at in `test_grid`."""

CAPTURES = corpora.RF_CORPUS / "p2hunt"
REAL = [CAPTURES / "hb9ak_055246_c1500.wav", CAPTURES / "hb9ak_055156_c1500.wav"]
FIRST_BURST_S = {"hb9ak_055246_c1500.wav": 0.650, "hb9ak_055156_c1500.wav": 0.659}
"""Where each recording's burst grid starts, from `p2rx.burst_grid` on the whole
file. Written down rather than recomputed so a window can be cut around a burst
without the thing under test choosing its own audio."""

PASS = 0
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


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
        self.sent.append(("pkt", payload))

    def send_p2_cs(self, i) -> None:
        self.sent.append(("p2cs", i))

    def send_p2_packet(self, sl, payload, status) -> int:
        self.sent.append(("p2pkt", payload))
        return pactor2.PATHS[sl - 1].crc_bytes - 3

    def send_p2_long_packet(self, sl, payload, status) -> int:
        self.sent.append(("p2pkt", payload))
        return pactor2.PATHS_LONG[sl - 1].crc_bytes - 3

    def pump(self) -> None:
        pass

    def cycle(self) -> None:
        pass


def _linked() -> tuple[PtcHost, _Seam]:
    """A station holding a PACTOR-1 link as the IRS, which is the case at hand."""
    seam = _Seam()
    host = PtcHost(peer=seam, mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.role, host.arq.dxcall = IRS, "WS8EOC"
    host.arq._enter_connected()
    seam.sent.clear()
    return host, seam


def _burst(level: int = 2, seed: int = 7,
           long_frame: bool = False) -> tuple[np.ndarray, bytes]:
    """A PACTOR-2 frame rendered to audio, and the field it carries.

    The interleaver is walked in the transmit direction -- code order into channel
    order, then channel order split into the two lanes `channel_buffer` reads them
    back out of -- so the bits reaching the air are the ones the receiver's own
    permutation expects to find. Rank 0 is the upper tone at home; the lower tone
    carries rank 1 and leads by half a symbol, which `pactor2.frame` holds.
    """
    path = (pactor2.PATHS_LONG if long_frame else pactor2.PATHS)[level]
    rng = np.random.default_rng(seed)
    info = bytearray(rng.integers(0, 256, path.crc_bytes - 2, dtype=np.uint8))
    # A real frame's status byte DECLARES its payload coding, and the link
    # layer decodes by that declaration -- random bits here once read as
    # "Huffman" and turned a plumbing test into a codec test. 8-bit mode, so
    # what is rendered is what the host is owed.
    info[-1] = spec.status_byte(int(info[-1]) & 0b11)
    field = pactor2.build_field(bytes(info), path)
    channel = np.zeros(path.n_buf, np.uint8)
    channel[pactor2.channel_of_code(path)] = pactor2.encode_frame(field, path)
    lanes = [channel.reshape(path.n_symbols, 2, path.bits_per_cell)[:, r, ::-1]
             .ravel() for r in range(2)]
    return pactor2.frame(lanes[1], lanes[0],
                         pactor2.marker_index(level, long_frame),
                         path.bits_per_cell, pactor2.NOMINAL_TONES_HZ, FS), field


def _recording(on_rx_event, into: list):
    """Tap the host's event port, so the EVENT can be asserted on and not only
    its effects."""
    def tapped(ev):
        into.append(ev)
        on_rx_event(ev)
    return tapped


def _window(burst: np.ndarray, at_s: float = 0.25,
            span_s: float = WINDOW_S) -> np.ndarray:
    """`burst` spliced into a cycle's worth of silence at `at_s`.

    `span_s` is the window the cycle hands the scan, which on the long cycle is
    the 3.75 s raster rather than the 1.25 s one -- a frame missing its last rows
    is one the receiver rejects without a word (`_SessionRx.deep_scan`)."""
    audio = np.zeros(int(span_s * FS))
    audio[int(at_s * FS):int(at_s * FS) + burst.size] = burst
    return audio


# --------------------------------------------------------------------------
# where the reader sits
# --------------------------------------------------------------------------

def reader_order() -> None:
    print("\nwhere the PACTOR-2 reader sits in the cycle")
    host, _ = _linked()
    rx = onair._SessionRx(host, tag="TEST")

    names = [r.__name__ for r in rx._readers()]
    check("a PACTOR-1 link scans PACTOR-1 first", names[0] == "_p1_packet",
          str(names))
    check("...and PACTOR-2 is in the list at all", "_p2_packet" in names,
          str(names))
    # THE WHOLE POINT OF THE SPLIT. `deep_scan` takes the head and runs before the
    # key, out of the reserve the cycle's own answer is paid from; `upgrade_scan`
    # takes the tail and runs behind our own carrier. A PACTOR-2 pass in front of
    # the key would spend the reserve on a protocol we are not in.
    check("...behind the head, so it runs in the transmit slot",
          names.index("_p2_packet") > 0, str(names))

    host.protocol = Protocol.PACTOR3
    check("an upgraded link still keeps its own protocol first",
          [r.__name__ for r in rx._readers()][0] == "_p3_packet",
          str([r.__name__ for r in rx._readers()]))


# --------------------------------------------------------------------------
# a rendered frame, end to end
# --------------------------------------------------------------------------

def synthetic_frame_reaches_the_host() -> None:
    print("\na rendered PACTOR-2 frame through the session receiver")
    burst, field = _burst()
    audio = _window(burst)
    path = pactor2.PATHS[2]
    payload = field[:path.crc_bytes - 3].rstrip(bytes([spec.IDLE]))
    status = field[path.crc_bytes - 3]

    host, seam = _linked()
    host.arq._expected_seq = status & 0b11
    rx = onair._SessionRx(host, tag="TEST")

    rx.deep_scan(audio)
    check("the scan before the key reads the link's own protocol and stops there",
          rx.count == 0 and host.protocol is Protocol.PACTOR1,
          f"{rx.count} events, link in {host.protocol}")

    rx.upgrade_scan(audio)
    check("the scan behind our own carrier finds the PACTOR-2 frame",
          rx.count == 1, f"{rx.count} events")
    check("...and its payload reached the host data port",
          payload in bytes(host.channel(host.ptchn).rx),
          repr(bytes(host.channel(host.ptchn).rx)[:24]))
    check("...and the ARQ layer answered it",
          any(k == "p2cs" for k, _ in seam.sent), str(seam.sent))
    # A PEER THAT LEADS INTO PACTOR-2 IS FOLLOWED, as of 2026-09-02. It has keyed
    # the waveform, which is the strongest statement PACTOR has about a far end,
    # and `ptc.TRANSMITTABLE` holds PACTOR-2 now -- so the answer goes back in the
    # protocol the peer asked for rather than in the one we could not leave.
    check("the link follows the peer into the protocol it is transmitting",
          host.protocol is Protocol.PACTOR2, str(host.protocol))


def a_pactor2_link_reads_every_rung_before_the_key() -> None:
    """The link is IN PACTOR-2, so `deep_scan` is the reader that has to find it.

    `upgrade_scan` above is the speculative pass, behind our own carrier, for a
    protocol we are not in. Once the link has followed the peer the PACTOR-2
    frame is the one this cycle's answer hangs on -- an IRS that has not read it
    by its own boundary has nothing to acknowledge -- so it moves to the head of
    `_readers` and runs out of `PREKEY_RESERVE_S`. Every rung and both frame
    lengths, because the marker names them and a station may be handed any of
    them without warning.

    THE CODEWORD IS THE OTHER HALF, and it is the half that makes us the peer's
    ISS rather than a station it can only talk at. It is read at one instant --
    `pactor2.cs_slot` past the packet's own phase reference -- at zero bit
    errors, in the arrangement the cycle is in.
    """
    print("\na PACTOR-2 link, every rung through the pre-key scan")
    for level in (0, 1, 2):
        for long_frame in (False, True):
            paths = pactor2.PATHS_LONG if long_frame else pactor2.PATHS
            path = paths[level]
            burst, field = _burst(level, seed=11 + level, long_frame=long_frame)
            host, seam = _linked()
            host.protocol = Protocol.PACTOR2
            host.arq._expected_seq = field[path.crc_bytes - 3] & 0b11
            heard: list = []
            host.on_rx_event = _recording(host.on_rx_event, heard)
            rx = onair._SessionRx(host, tag="TEST")
            names = [r.__name__ for r in rx._readers()]

            span = burst.size / FS + 0.5
            rx.deep_scan(_window(burst, 0.25, span))
            got = heard[0] if heard else None
            check(f"{path.name}: the link's own protocol reads it before the key",
                  names[0] == "_p2_packet" and got is not None
                  and got.protocol is Protocol.PACTOR2,
                  f"{names} / {got}")
            check(f"{path.name}: ...at the speed level its marker named",
                  got is not None and got.packet[0] == level + 1,
                  str(got.packet[0]) if got else "nothing")
            check(f"{path.name}: ...with the field the peer built",
                  got is not None
                  and got.packet[2] == spec.field_payload(
                      field[:path.crc_bytes - 3]),
                  repr(got.packet[2][:24]) if got else "nothing")

            # ...and the peer's answer to a packet of ours, at the grid's instant.
            for swapped in (False, True):
                rx.host.peer.invert = swapped
                cs = pactor2.control_signal(2, swapped=swapped)
                lead = pactor2.pulse_lead()
                at = int(round(pactor2.cs_slot(path) * FS))
                audio = np.zeros(at + cs.size + FS // 2)
                audio[at - lead:at - lead + cs.size] = 0.2 * cs
                ev = rx._p2_cs(audio, at)
                check(f"{path.name}: ...and the answer slot reads a codeword "
                      f"{'swapped' if swapped else 'at home'}",
                      ev is not None and ev.cs == 2 and "0 bit errors" in ev.text,
                      str(ev))


def a_link_with_no_peer_hears_nothing() -> None:
    """The same window with the burst taken out. A scan that fires on silence is
    not a reader, and the arming threshold is the only thing standing between a
    256-alignment rotation search and every empty cycle of a session."""
    print("\nan empty window")
    host, _ = _linked()
    rx = onair._SessionRx(host, tag="TEST")
    rng = np.random.default_rng(11)
    rx.upgrade_scan(rng.normal(0, 0.05, int(WINDOW_S * FS)))
    check("noise in the peer's slot raises nothing", rx.count == 0,
          f"{rx.count} events")


# --------------------------------------------------------------------------
# real off-air PACTOR-2
# --------------------------------------------------------------------------

def real_bursts_survive_the_one_cycle_path(path: Path) -> None:
    """Every burst of a real recording, read one cycle at a time.

    `decode_bursts` reads the whole file: it fits the burst grid's residue over
    all the markers at once and covers the anchors whose own marker faded. A
    session has no such recourse -- one window, one marker, one burst -- so the
    marker is the boundary of what it can be asked for: every burst whose own
    marker arms must come back identical, and a burst only the grid can place
    (the 4.40 s fade of `_055246`, whose marker reaches 0.70 in both
    arrangements) is the whole-file path's to read, not this one's.
    """
    print(f"\n{path.name}: one cycle at a time")
    audio = session.load_wav(str(path), FS)
    whole = {round(t, 2): f for t, f in p2rx.decode_bursts(audio, FS)}
    armed = {round(t, 2) for t, _b, _k, _s, sc, _sw in p2rx.find_markers(audio, FS, 0.90)
             if sc >= p2rx.MARKER_ARM}
    anchored = {t: f for t, f in whole.items() if t in armed}
    first = FIRST_BURST_S[path.name]

    read: dict[float, bytes] = {}
    costs: list[float] = []
    n = round(WINDOW_S * FS)
    for i in range(int((audio.size / FS - first) / CYCLE_S)):
        at = first + CYCLE_S * i
        seg = audio[round((at - 0.25) * FS):][:n]
        if seg.size < n:
            break
        t0 = time.perf_counter()
        got = p2rx.decode_expected_burst(seg, FS)
        costs.append(time.perf_counter() - t0)
        if got is not None:
            read[round(at, 2)] = got[2]

    check(f"the one-cycle path reads every burst whose marker arms "
          f"({len(anchored)} of {len(whole)})", set(read) >= set(anchored),
          f"{len(read)} read, missing {sorted(set(anchored) - set(read))}")
    check("...and reads them to the same bytes",
          all(read[t] == f for t, f in whole.items() if t in read),
          str([t for t, f in whole.items() if t in read and read[t] != f]))
    print(f"    {len(read)} of {int(audio.size / FS / CYCLE_S)} cycles carry a "
          f"CRC-valid field; per-window cost min {min(costs) * 1e3:.1f} / "
          f"median {np.median(costs) * 1e3:.1f} / max {max(costs) * 1e3:.1f} ms")


def a_real_burst_reaches_the_host(path: Path) -> None:
    print(f"\n{path.name}: one real burst through the session receiver")
    audio = session.load_wav(str(path), FS)
    at = FIRST_BURST_S[path.name] + CYCLE_S      # the second, which is clear of the
    seg = audio[round((at - 0.25) * FS):][:round(WINDOW_S * FS)]  # ragged start
    got = p2rx.decode_expected_burst(seg, FS)
    if got is None:
        check("the chosen window carries a CRC-valid burst", False, f"t={at}")
        return
    path_, field = got[1], got[2]
    status = field[path_.crc_bytes - 3]
    # Trailing IDLE is padding and the receiver drops it -- and the link layer
    # then decodes the field by its own status byte before the host sees a
    # byte, so what the host is owed is the field's CHARACTERS. On these
    # recordings that is PMC English ('in the ', on the 055156 burst) or a pure
    # idle fill, which decodes to nothing and must deliver none of it.
    payload = field[:path_.crc_bytes - 3].rstrip(bytes([spec.IDLE]))
    expected = compress.decompress(payload, (status >> 2) & 0x7)

    host, seam = _linked()
    host.arq._expected_seq = status & 0b11
    heard: list = []
    rx = onair._SessionRx(host, tag="TEST")
    host.on_rx_event = _recording(host.on_rx_event, heard)
    rx.upgrade_scan(seg)

    check("a real station's PACTOR-2 data reaches the host", rx.count == 1,
          f"{rx.count} events")
    check("...naming the protocol it arrived in",
          [ev.protocol for ev in heard] == [Protocol.PACTOR2],
          str([ev.protocol for ev in heard]))
    check("...with that field's characters, wire coding decoded",
          bytes(host.channel(host.ptchn).rx) == expected,
          f"host {bytes(host.channel(host.ptchn).rx)[:24]!r}, "
          f"owed {expected[:24]!r}")
    check("...and is answered in PACTOR-2", host.protocol is Protocol.PACTOR2
          and any(k == "p2cs" for k, _ in seam.sent), str(seam.sent))
    # THE GEAR IS ASKED FOR ON THE FRAME'S OWN LADDER. This reported
    # `arq.P1_SPEED_LEVEL` while PACTOR-2 was unkeyable, because `arq._gear_cs`
    # turns a clean run into CS4 -- the next level up -- and the codeword that
    # would have carried the request was PACTOR-1's, in which the same index
    # means Speedchange. The request goes out as `pactor2.control_signal` now,
    # so the level the marker named travels and `arq.P2_LADDER` is what it is
    # counted against.
    check("...reporting the speed level the frame's own marker named",
          all(ev.packet[0] == path_.level + 1 for ev in heard),
          str([ev.packet[0] for ev in heard]))
    check("...on PACTOR-2's ladder and not PACTOR-3's",
          host.arq.ladder is P2_LADDER and host.arq.ladder.top == 3,
          str(host.arq.ladder.protocol))


# --------------------------------------------------------------------------
# the cost
# --------------------------------------------------------------------------

def the_reader_fits_the_cycle() -> None:
    """What a PACTOR-2 pass costs the cycle it runs in.

    Beside `test_grid`'s budget for the same 1.30 s window on the same machine:
    the PACTOR-1 scan 5 ms, `deep_scan` 20.3 ms with nothing in the channel, the
    blind PACTOR-3 pass 48.0 ms, the flush 26.7 ms.

    The floor is what matters, because it is what every cycle pays. Acquisition is
    a correlation and nothing behind it runs until a marker arms at 0.94, so an
    empty window costs the correlator alone. The ceiling is the rotation search:
    256 trellis decodes when a marker armed and no alignment produced a CRC, which
    on real audio happened once in the 32 cycles of `hb9ak_055246_c1500.wav`.
    """
    print("\nwhat it costs the cycle")
    burst, _ = _burst()
    rng = np.random.default_rng(3)
    empty = rng.normal(0, 0.05, int(WINDOW_S * FS))
    carrying = _window(burst)

    def cost(audio) -> float:
        return min(_timed(audio) for _ in range(3))

    def _timed(audio) -> float:
        t0 = time.perf_counter()
        p2rx.decode_expected_burst(audio, FS)
        return time.perf_counter() - t0

    floor, loaded = cost(empty), cost(carrying)
    print(f"    empty window {floor * 1e3:.1f} ms, window carrying a frame "
          f"{loaded * 1e3:.1f} ms, over a {WINDOW_S:.2f} s window")
    # Deliberately loose. The number this guards is 1.25 s of cycle against a
    # measured 15 ms, and what it is here to catch is a twentyfold regression --
    # a reader that grew a per-alignment decode, or an acquisition that stopped
    # being a matrix product -- not a scheduler hiccup on a shared machine.
    check("an empty cycle pays the correlator and nothing else",
          floor < 0.25 * CYCLE_S, f"{floor * 1e3:.1f} ms of {CYCLE_S:.2f} s")
    check("...and a cycle carrying a frame still fits the transmit slot",
          loaded < 0.5 * CYCLE_S, f"{loaded * 1e3:.1f} ms")


def main() -> int:
    reader_order()
    synthetic_frame_reaches_the_host()
    a_pactor2_link_reads_every_rung_before_the_key()
    a_link_with_no_peer_hears_nothing()
    the_reader_fits_the_cycle()
    for path in REAL:
        if not path.exists():
            print(f"\n  SKIP -- {path} is not here")
            continue
        real_bursts_survive_the_one_cycle_path(path)
        a_real_burst_reaches_the_host(path)

    print(f"\n{PASS} passed, {len(FAILURES)} failed")
    return 0 if not FAILURES else 1


def test_main() -> None:
    if not any(p.exists() for p in REAL):
        pytest.skip("no off-air PACTOR-2 recording available")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
