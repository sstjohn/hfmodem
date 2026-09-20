# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Oracle gate for the PACTOR-1 DATA phase: PMON must print our payload.

The connect has been oracle-proven since 2026-07-14 (`###CONNECT: [Normal Call:
W1AW]` from fully self-generated audio). The data phase never has, and on the air
a gateway answers our connect and then asks for the packet again -- exactly what
PACTOR-1 does when it cannot take one. This closes that loop: render a whole link
setup out of `shrike.pactor1` alone, no recorded audio anywhere, and require the
reference monitor to read the payload back.

What is rendered, on one 1.25 s raster (`spec.CYCLE_SHORT_S`):

    slot 0   connect burst for the DX call
    slot 1   first data packet, Head=0xAA, counter=1 -- what the description says
             a caller sends "sobald das erste gueltige CS empfangen ... ist"
    slot 2   the identical packet again

The negative arm is the same audio with the data field corrupted in both copies,
its CRC trailer left at the value the good field computed. It is what makes a pass
mean anything: PMON reading our bytes, rather than PMON printing something it
would have printed regardless.

VERDICT, 2026-07-26 -- PASS. Five positive runs across both Lima instances, each
instance selftested first, every one of them printing:

    ###PLISTEN: Level: 1:
    ###STATUS: SL: 1, CYC: 0, RQ: 0, REV: 1, LSB: 0, dF:  0.0, FRNR: 1
    ###PAYLOAD1: LEN: 8, TYPE: 4
    ###PAYLOAD2:
    1W9SSJ

    ###PAYLOAD_END

-- once per copy, the second reported RQ: 1, FRNR: 2. That is the same shape as
the one real PACTOR-1 packet in the corpus, which reads `1w4dna` at LEN 8 TYPE 4.
The corrupted arm, four conclusive runs, prints the connect out of the shared
prefix and then nothing at all.

The repeat in slot 2 is NOT what makes the packet acceptable, which is worth
saying plainly because `shrike/pactor1.py` says the opposite: its 0x59390 note
claims PMON never recomputes the data CRC and takes a packet only on a matching
consecutive copy. Rendered with `--repeats 1`, a lone packet decodes -- PMON
printed the payload from a single burst. That agrees with `docs/protocols/pactor/pactor3.md`,
which already retracted the "0x59390 is the data CRC" reading (it returns 1 for a
NEW frame; the real CRC-16/X-25 is 0x5d314). The repeat is protocol-faithful
memory-ARQ and costs nothing, but the module comment overstates its role.

Transcripts: `oracle/vm/results/p1data_oracle_2026-07-26.txt`.

THE 200 Bd ARM, 2026-07-27 -- PASS, and it is the only validation this speed has.
`docs/protocols/pactor/pactor1-data-packets.md` records that 12.25 hours of corpus hold three PACTOR-1
data packets and all three are 100 Bd, so the link speed a gateway selects by
answering CS1 has never been checked against anything but our own decoder. One
rendered file carries four segments and settles four questions; see `render_200`
and `docs/protocols/pactor/pactor1-data-packets.md` §6 for the transcripts. Six runs of it returned
the same ten frames every time, FRNR 1-10, nothing dropped and nothing extra --
the 200 Bd path needed no change to pass, and the only failures this arm has had
were in how its transcript was read.

THE LIVE ARM, 2026-07-30 -- PASS, and it is the one that answers the question the
data phase actually poses. WS8EOC answered a connect and then never advanced past
`pkt#1`, and the two standing suspicions were both about the packet: that the
7-byte payload meant a rate-dependent field was being read wrong, and that the
first-packet header rule was off. Neither survives. `render_live` drives
`ptc.PtcHost` and `onair.RadioTx` -- the same objects a session on the air drives
-- and renders what the transmit path emits, invert flag and all. PMON reads it at
both speeds:

    ###CONNECT: [Normal Call: WS8EOC]
    ###STATUS: SL: 2, CYC: 0, RQ: 0, REV: 1, LSB: 0, dF:  0.0, FRNR: 1
    ###PAYLOAD1: LEN: 8, TYPE: 4
    ###PAYLOAD2:
    1W9SSJ

    ###PAYLOAD_END

-- FRNR 1 and 2 off the CS1/200 Bd segment at SL: 2, FRNR 3 and 4 off the CS4/100
Bd segment at SL: 1, payload byte-exact in all four. The transmitted packet is
spec-correct and an independent PACTOR-1 receiver takes it, so whatever stalls the
link is not the packet. The field is 20 bytes at 200 Bd and 8 at 100, as it should
be; the 7 is simply how long `1W9SSJ\\r` is.

THE SHIFT ARM, 2026-07-30 -- what it settles is that the oracle cannot settle it.
"Mit jedem neuen Paket oder Kontrollsignal wird die Shiftlage invertiert", so the
first data packet goes out in the opposite FSK sense from the connect, and no arm
above had ever rendered that. `render_shift` carries the 2x2 of (connect sense,
packet sense) and PMON read all four segments, reporting the packet's own sense in
its `REV` field and never referring it back to the connect. It auto-detects per
burst. So this gate is silent on whether our alternation is in the phase a locked
ARQ receiver expects -- a monitor that searches has no reason to care, and a
receiver holding a link does. Keep the arm: it is what retires the oracle as an
arbiter of the shift question, which is worth more than another pass.

The oracle guest reads its input through a Lima mount, so the staging directory
has to be one the guest can see. `PMON_STAGE` names it; without it there is no
verdict and the gate says so rather than rendering into a path the VM will find
empty.

Run:  PMON_STAGE=<dir the guest mounts> PMON_ORACLE=<oracle tree> \\
      .venv/bin/python -m hfmodem.tests.shrike.test_p1_oracle          (all arms)
      .venv/bin/python -m hfmodem.tests.shrike.test_p1_oracle --render-only
      .venv/bin/python -m hfmodem.tests.shrike.test_p1_oracle --repeats 1
      .venv/bin/python -m hfmodem.tests.shrike.test_p1_oracle --arm 200
PMON drops a decode now and then, so every stage runs twice and one success is
enough. A default run is up to twelve VM boots -- health, 100 Bd positive and
negative, 200 Bd, live, shift -- so `--arm` is how a single question gets asked.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import frame, onair, pactor1, ptc, spec
from hfmodem.tests.shrike import archive

UNAVAILABLE = "ORACLE UNAVAILABLE -- no verdict. Nothing below would mean anything."
NO_STAGE = ("set PMON_STAGE to a directory the oracle guest mounts -- without one "
            "the VM is fed a path it cannot read, which looks exactly like a "
            "signal that did not decode")
NO_ORACLE = "set PMON_ORACLE to the tree holding the oracle runner and its images"
# `PMON_ORACLE` names the tree the runner, its images and the selftest capture
# live in, exactly as the P2/P3 gates read it. It has no in-repo fallback: the
# guest mounts the original tree and nothing else, so a runner rooted in an
# in-repo copy of the oracle tree hands the guest paths it cannot read -- which
# looks exactly like a signal that did not decode, and did, from the day of the
# monorepo move to 2026-08-05.
_ORACLE_ROOT = os.environ.get("PMON_ORACLE")
ROOT = Path(_ORACLE_ROOT) if _ORACLE_ROOT else None
STAGE = os.environ.get("PMON_STAGE")
OUTDIR = (Path(STAGE) if STAGE else
          archive.ARCHIVE / "oracle" / "vm" / "p1data")
DXCALL = "W1AW"
PAYLOAD = b"1W9SSJ\r"
EXPECT = "1W9SSJ"      # what PMON should print; the CR is not part of the match
BAUD = 100
CYCLE = spec.CYCLE_SHORT_S
AMP = 0.11             # `connect_signal`/`packet_signal`/`control_signal` default
PHASE0 = 1.22
PACKET_S = 0.96        # both speeds: 96 bauds at 100 Bd, 192 at 200 Bd
TURNAROUND_S = 0.085   # d, pactor1-timing.md §0 -- 170 ms residual split 85/85


class Grid:
    """Absolute placement of bursts on the 1.25 s ARQ raster.

    `at()` takes seconds from the start of slot 0 -- the start of the connect's
    DATA BITS, not of the file -- so the geometry in pactor1-timing.md is
    transcribed rather than recomputed at each call site. A master's connect and
    its data packets share one cycle grid, and the peer's control signals sit in
    the same grid at packet-end plus `d`.
    """

    def __init__(self, lead_s: float = 0.5) -> None:
        self.lead = int(lead_s * spec.SAMPLE_RATE)
        self._items: list[tuple[int, np.ndarray]] = []

    def at(self, t: float, sig: np.ndarray) -> "Grid":
        self._items.append((self.lead + int(round(t * spec.SAMPLE_RATE)), sig))
        return self

    def connect(self, t: float, call: str = DXCALL) -> "Grid":
        return self.at(t, pactor1._dualrate_frame(call, AMP, PHASE0))

    def packet(self, t: float, payload: bytes, baud: int, count: int, *,
               corrupt: bool = False) -> "Grid":
        """One data packet, optionally with its field mangled under a good CRC.

        Corruption is re-modulated rather than edited in the sample domain, so
        nothing about the timing, the header, the status byte or the raster
        changes. Only the field bytes differ, and the CRC trailer keeps the value
        the good field computed -- the frame is well-formed in every respect
        except that it does not check out.
        """
        pkt = bytearray(pactor1.data_packet(payload, baud, count))
        if corrupt:
            for i in range(1, 1 + pactor1.DATA_FIELD[baud]):
                pkt[i] ^= 0x5A
        return self.at(t, pactor1._fsk_burst(bytes(pkt), baud, PHASE0, AMP))

    def cs(self, t: float, index: int) -> "Grid":
        return self.at(t, pactor1.control_signal(index, amp=AMP))

    def render(self, tail_s: float = 0.5) -> np.ndarray:
        end = max(s + len(x) for s, x in self._items)
        out = np.zeros(end + int(tail_s * spec.SAMPLE_RATE), dtype=np.float32)
        for s, x in self._items:
            out[s:s + len(x)] += x.astype(np.float32)
        return out


def render(*, corrupt: bool = False, repeats: int = 2,
           connect_slots: int = 1) -> np.ndarray:
    """A complete self-generated 100 Bd PACTOR-1 link setup as one float32 signal.

    `connect_slots` is how many raster slots the connect occupies before the data
    starts; the caller's packet follows the answering station's control signal,
    which we do not transmit, so the slot it would occupy is simply silent.
    """
    g = Grid().connect(0.0)
    for i in range(repeats):
        g.packet((connect_slots + i) * CYCLE, PAYLOAD, BAUD, 1, corrupt=corrupt)
    return g.render()


# --- 200 Bd, the speed with no off-air packet to check against ---------------
#
# Four segments in one file, each with its own connect and ~6.5 s of silence
# ahead of it so PMON has dropped the previous link. One VM boot per arm is what
# keeps the suite inside its timeout, and packing loses nothing: PMON reports
# frames in time order, so the segment a decode came from is never in doubt.
#
# THE CORRUPTED SEGMENT GOES FIRST, and that ordering is load-bearing. If the
# guest is killed before the file finishes -- boot plus 40 s of audio against the
# `--wait` cap -- a truncated run can only cost a positive arm. With the negative
# at the tail, truncation would have made it pass for having been shown nothing.
NEG_PAYLOAD = b"DELTA MUST NOT SHOW\r"
BARE_PAYLOAD = b"DE W9SSJ ALFA 200BD\r"
SESSION_CHUNKS = (b"BRAVO0 DE W9SSJ 200B", b"BRAVO1 SESSION LINK ",
                  b"BRAVO2 FOUR PACKETS ", b"BRAVO3 END OF TEST \r")
FALLBACK_20 = b"CHARLIE FALLBACK 200"
FALLBACK_8 = (b"CHARLIE ", b"FALLBACK", b" 200")
SEG_T = (0.0, 10.0, 20.0, 33.0)


def render_200() -> np.ndarray:
    """The whole 200 Bd case as one signal: four questions, four segments.

    1. NEGATIVE -- connect plus two copies of a 200 Bd packet whose 20-byte field
       is mangled under the CRC the good field computed. Nothing may be read out
       of it, and the segments after it are what prove PMON was listening.

    2. BARE -- the 100 Bd arm's shape at 200 Bd: a connect and two copies of one
       packet, with nothing from the peer anywhere on the air. This is the segment
       that shows PMON takes the speed level off the packet itself; no control
       signal tells it, and it still reports SL: 2.

    3. SESSION -- a QSO that never happened. PMON is a passive monitor, so both
       directions can be rendered: the peer's CS1 answer to the connect, four new
       packets carrying 20 bytes each, and the peer's acknowledgements alternating
       CS1/CS2 on the raster. Counter and header run 1/0xAA, 2/0x55, 3/0xAA,
       0/0x55, which `data_packet` derives from the counter.

    4. SPEED TRANSITION -- CS1 brings the link up at 200 Bd, one 20-byte packet
       goes out, and the peer answers CS4. On a link already at 200 that is a
       REJECT, not a repeat request: the packet is discarded and the same
       information is re-chunked into three 8-byte packets at 100 Bd, the first
       reusing sequence 1. That is what `shrike/ptc.py` does on CS4, and until
       2026-07-27 it had never been rendered for anything to read.
    """
    g = Grid()
    t = SEG_T[0]
    g.connect(t)
    for i in range(2):
        g.packet(t + (1 + i) * CYCLE, NEG_PAYLOAD, 200, 1, corrupt=True)

    t = SEG_T[1]
    g.connect(t)
    for i in range(2):
        g.packet(t + (1 + i) * CYCLE, BARE_PAYLOAD, 200, 1)

    t = SEG_T[2]
    g.connect(t)
    g.cs(t + PACKET_S + TURNAROUND_S, pactor1.CS_ACK_A)
    for i, chunk in enumerate(SESSION_CHUNKS):
        s = t + (1 + i) * CYCLE
        g.packet(s, chunk, 200, (1 + i) & 3)
        g.cs(s + PACKET_S + TURNAROUND_S,
             pactor1.CS_ACK_B if i % 2 == 0 else pactor1.CS_ACK_A)

    t = SEG_T[3]
    g.connect(t)
    g.cs(t + PACKET_S + TURNAROUND_S, pactor1.CS_ACK_A)
    g.packet(t + CYCLE, FALLBACK_20, 200, 1)
    g.cs(t + CYCLE + PACKET_S + TURNAROUND_S, pactor1.CS_SPEED)
    # The acknowledgement toggle is not reset by CS4, so it resumes on CS2 -- the
    # partner of the CS1 that answered the connect.
    for i, chunk in enumerate(FALLBACK_8):
        s = t + (2 + i) * CYCLE
        g.packet(s, chunk, 100, 1 + i)
        g.cs(s + PACKET_S + TURNAROUND_S,
             pactor1.CS_ACK_B if i % 2 == 0 else pactor1.CS_ACK_A)
    return g.render()


# --- the live path, and the shift position it transmits in -------------------

LIVE_SEG_T = (0.0, 12.0)
SHIFT_SEG_T = (0.0, 10.0, 20.0, 30.0)
SHIFT_SEGS = (("A", False, False, b"ALFA CONN0 PKT0 200B"),
              ("B", False, True,  b"BRAVO CONN0 PKT1 20B"),
              ("C", True,  False, b"CHARLIE CONN1 PKT0 2"),
              ("D", True,  True,  b"DELTA CONN1 PKT1 200"))


def _session_bursts(cs_index: int) -> tuple[np.ndarray, np.ndarray]:
    """(connect, first data packet) as the session layer emits them.

    Rendered by `ptc.PtcHost` and `onair.RadioTx` rather than by calling the
    renderers directly, so what reaches the oracle is what a station on the air
    would key -- the payload the FSM chose, the counter and header it derived, and
    the shift position `RadioTx._flip` put it in. Transcribing those into a call
    here would be a second implementation of the transmit path, and a gate that
    agrees with its own transcription proves nothing about the one that flies.

    Two calls go out before the answer -- one on the connect, one on the cycle
    that follows it -- so the packet is the LAST burst rather than the second.

    There is no sample grid here, so `RadioTx._flip` falls back to its
    per-transmission toggle and the packet goes out in the non-inverted sense. On
    the air the sense is whole cycles since the call, which makes the first data
    packet's shift position a property of how long the call ran rather than of the
    packet: a link answered on the first cycle sends it inverted. `render_shift`
    is where both positions are put in front of the oracle.
    """
    out: list[np.ndarray] = []
    tx = onair.RadioTx(None, transmit=False, outdir=Path("."))
    tx._tx = lambda audio, what, **kw: out.append(onair._trim_silence(audio))
    host = ptc.PtcHost(peer=tx, mycall="W9SSJ")
    tx.attach(host)
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.tick()

    class Answer:
        kind, cs, protocol = "cs", cs_index, "PACTOR-1"

    host.on_rx_event(Answer())
    host.tick()
    return out[0], out[-1]


def render_live() -> np.ndarray:
    """Both link speeds of the live transmit path, on the raster it keys them on.

    CS1 first (the answer that takes the link to 200 Bd), then CS4 (the answer
    that holds it at 100). Each segment is the connect the session sent and two
    copies of the data packet it sent after the answer.
    """
    g = Grid()
    for t, cs in zip(LIVE_SEG_T, (pactor1.CS_ACK_A, pactor1.CS_SPEED)):
        conn, pkt = _session_bursts(cs)
        g.at(t, conn / np.abs(conn).max() * AMP)
        for i in range(2):
            g.at(t + (1 + i) * CYCLE, pkt / np.abs(pkt).max() * AMP)
    return g.render()


def render_shift() -> np.ndarray:
    """The 2x2 of connect shift position against packet shift position.

    The live path emits segment B -- connect in the sense the link was fixed in,
    first data packet in the other one -- and until this existed no arm had ever
    put an inverted packet in front of the oracle.
    """
    g = Grid()
    for (_, cinv, pinv, payload), t in zip(SHIFT_SEGS, SHIFT_SEG_T):
        g.at(t, pactor1._dualrate_frame(DXCALL, AMP, PHASE0, invert=cinv))
        for i in range(2):
            g.at(t + (1 + i) * CYCLE,
                 pactor1._fsk_burst(pactor1.data_packet(payload, 200, 1), 200,
                                    PHASE0, AMP, invert=pinv))
    return g.render()


# PMON's console carries raw bytes that are not UTF-8 -- the stray 0xfa either
# side of every report -- so a strict decode raises partway through the health
# check and takes the whole gate down before it has run anything.
def _sh(*argv: str, timeout: int | None = None) -> str:
    r = subprocess.run(["bash", *argv], capture_output=True, text=True,
                       errors="replace", timeout=timeout)
    return r.stdout + r.stderr


def _oracle(wav: Path, runner: str, wait: int) -> str:
    return _sh(str(ROOT / "oracle" / "vm" / runner), str(wav), str(wait),
               timeout=wait + 600)


def _lines(out: str) -> list[str]:
    """PMON's console with its periodic amplitude report taken out.

    That report is not framed against anything and lands wherever the timer fires
    -- including between a payload and its ###PAYLOAD_END, measured three times in
    four runs. Anything reading the transcript has to drop it first.
    """
    return [ln for ln in out.splitlines() if "INPUT AMPLITUDE" not in ln]


def _pmon_section(out: str) -> str:
    _, _, tail = out.partition("===== PMON OUTPUT =====")
    return "\n".join(_lines(tail)).strip() or "(nothing after the marker)"


def _arm(wav: Path, label: str, runner: str, wait: int, attempts: int) -> list[str]:
    """Run one arm `attempts` times and print every transcript verbatim."""
    outs = []
    for i in range(attempts):
        print(f"\n== {label} arm, run {i + 1}/{attempts} ==")
        out = _oracle(wav, runner, wait)
        print(_pmon_section(out))
        outs.append(out)
    return outs


_STATUS = re.compile(r"SL:\s*(\d+).*FRNR:\s*(\d+)")


def _frames(out: str) -> list[tuple[int, int, str]]:
    """(speed level, frame number, payload text) for every frame PMON reported.

    Substring matching is not enough for the 200 Bd arm: the speed level is the
    whole point of two of its four segments, and only the ###STATUS line carries
    it. Frames come out in time order, which is what lets a packed file be read
    segment by segment.

    Payloads are compared stripped. PMON's LEN is neither the field size nor the
    payload size -- trailing 0x1E IDLE is dropped and a CR counts two, so the
    8-byte field `20 32 30 30 1e 1e 1e 1e` reports LEN: 4 -- and its rendering of
    a CR breaks the line, so a multi-line body is rejoined.

    It reads the transcript through `_lines`, and that is not tidiness. Parsing
    the raw console instead cost three of four runs: PMON's amplitude report fires
    on a timer and landed between the payload and its ###PAYLOAD_END, so a frame
    came back as `'BRAVO2 FOUR PACKETS \n###INPUT AMPLITUDE: 11.00 % ...'`. It
    failed the sequence checks and the negative control at once, and read exactly
    like PMON having accepted a field we mangled.
    """
    frames, lines = [], _lines(out)
    sl = frnr = None
    for i, ln in enumerate(lines):
        m = _STATUS.search(ln)
        if m:
            sl, frnr = int(m.group(1)), int(m.group(2))
        elif ln.startswith("###PAYLOAD2:") and sl is not None:
            body = []
            for nxt in lines[i + 1:]:
                if nxt.startswith("###PAYLOAD_END"):
                    break
                body.append(nxt)
            frames.append((sl, frnr, "\n".join(body).strip()))
    return frames


def _run_of(frames, want: list[tuple[int, str]]) -> list[tuple[int, int, str]] | None:
    """The first consecutive run of frames matching `want` as (speed level, text).

    Consecutive rather than a scattered subsequence, because both callers are
    asking whether PMON held the link across a sequence of packets -- a match with
    something else decoded in between is a different claim.
    """
    n = len(want)
    for i in range(len(frames) - n + 1):
        if all((frames[i + j][0], frames[i + j][2]) == want[j] for j in range(n)):
            return frames[i:i + n]
    return None


def _consecutive(run) -> bool:
    return all(b[1] == a[1] + 1 for a, b in zip(run, run[1:]))


def _txt(b: bytes) -> str:
    return b.decode("ascii").strip()


def _verdict_200(outs: list[str]) -> bool:
    """Four claims about the 200 Bd path, each read off PMON's own report."""
    runs = [_frames(o) for o in outs]
    reached = [f for f in runs if f]
    checks = []

    bare = [f for f in runs
            if any(sl == 2 and t == _txt(BARE_PAYLOAD) for sl, _, t in f)]
    checks.append(("PMON reads a 200 Bd packet at SL: 2 with no control signal "
                   "on the air", bool(bare), f"{len(bare)}/{len(outs)} runs"))

    want = [(2, _txt(c)) for c in SESSION_CHUNKS]
    sess = [r for r in (_run_of(f, want) for f in runs) if r and _consecutive(r)]
    checks.append(("a whole 200 Bd QSO decodes, four new packets, FRNR unbroken",
                   bool(sess), f"{len(sess)}/{len(outs)} runs"))

    want = [(2, _txt(FALLBACK_20))] + [(1, _txt(c)) for c in FALLBACK_8]
    fall = [r for r in (_run_of(f, want) for f in runs) if r and _consecutive(r)]
    checks.append(("CS4 takes the link 200 -> 100 Bd and PMON follows it down "
                   "without losing sync", bool(fall), f"{len(fall)}/{len(outs)} runs"))

    # Every frame in the file is accounted for, rather than just checking that the
    # payload we mangled is absent: a corrupted field that PMON accepted would
    # print the mangled bytes, not `NEG_PAYLOAD`. A run that decoded nothing at
    # all proves nothing here and is not counted either way -- but since the
    # corrupted segment is first, any decode at all shows PMON got past it.
    good = {_txt(p) for p in (BARE_PAYLOAD, FALLBACK_20, *SESSION_CHUNKS, *FALLBACK_8)}
    stray = [(sl, n, t) for f in reached for sl, n, t in f if t not in good]
    checks.append(("a mangled 200 Bd field is NOT read back", bool(reached)
                   and not stray, f"{len(reached)} conclusive runs"))

    if not reached:
        print("  [INCONCLUSIVE] no 200 Bd run decoded anything -- rerun")
        return False
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} ({detail})")
    # Naming the frame is the difference between a diagnosis and another VM run.
    # Every failure this arm has had was console noise inside a payload block, and
    # the verdict line alone could not tell that from PMON having accepted a
    # mangled field -- which is the one thing it exists to detect.
    for sl, n, t in stray:
        print(f"    unexpected frame: SL {sl}, FRNR {n}, payload {t!r}")
    return all(ok for _, ok, _ in checks)


def _verdict_live(outs: list[str]) -> bool:
    """The transmit path's own audio, at both speeds, read back byte-exact.

    Two claims and no negative arm, because the negative this arm needs already
    exists: the 100 Bd corrupted segment shows PMON does not print a payload it
    was not given. What is new here is only WHERE the audio came from.
    """
    runs = [_frames(o) for o in outs]
    ok = True
    for sl, what in ((2, "CS1 -> 200 Bd"), (1, "CS4 -> 100 Bd")):
        hit = [f for f in runs if any(s == sl and t == EXPECT for s, _, t in f)]
        ok &= bool(hit)
        print(f"  [{'PASS' if hit else 'FAIL'}] the session layer's own {what} "
              f"packet reads back as {EXPECT!r} at SL: {sl} "
              f"({len(hit)}/{len(outs)} runs)")
    return ok


def _verdict_shift(outs: list[str]) -> bool:
    """Whether the oracle can tell the four shift combinations apart. It cannot.

    A pass here is PMON reading ALL FOUR, which is the finding: it takes a data
    packet in either shift position whatever position the connect went out in, so
    it cannot arbitrate the alternation. Anything less than four would be the
    interesting result and would make this an arbiter again.
    """
    runs = [_frames(o) for o in outs]
    want = [_txt(p) for _, _, _, p in SHIFT_SEGS]
    read = [{t for _, _, t in f} for f in runs]
    missing = [set(want) - r for r in read]
    best = min(missing, key=len)
    ok = not best
    print(f"  [{'PASS' if ok else 'FAIL'}] PMON reads a data packet in either "
          f"shift position, whichever position the connect used "
          f"({len(want) - len(best)}/{len(want)} segments)")
    if best:
        print(f"    not read: {sorted(best)}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--runner", default="run.sh",
                    help="oracle/vm runner: run.sh (shrike) or p2-run.sh (shrike-p2)")
    ap.add_argument("--selftest", default="selftest.sh")
    ap.add_argument("--wait", type=int, default=240)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--connect-slots", type=int, default=1)
    ap.add_argument("--arm", choices=("100", "200", "live", "shift", "all"),
                    default="all")
    a = ap.parse_args(argv)

    def wants(name: str) -> bool:
        return a.arm in (name, "all")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pos = OUTDIR / "p1data_pos.wav"
    neg = OUTDIR / "p1data_neg.wav"
    p200 = OUTDIR / "p1data_200.wav"
    plive = OUTDIR / "p1data_live.wav"
    pshift = OUTDIR / "p1data_shift.wav"
    files = []
    if wants("100"):
        files += [(pos, render(repeats=a.repeats, connect_slots=a.connect_slots)),
                  (neg, render(corrupt=True, repeats=a.repeats,
                               connect_slots=a.connect_slots))]
    if wants("200"):
        files.append((p200, render_200()))
    if wants("live"):
        files.append((plive, render_live()))
    if wants("shift"):
        files.append((pshift, render_shift()))
    for path, sig in files:
        frame.write_wav(str(path), sig)
        print(f"rendered {path.name}: {len(sig) / spec.SAMPLE_RATE:.2f} s")
    if a.render_only:
        return 0
    if ROOT is None:
        print(NO_ORACLE)
        return 2

    # A silent oracle is only evidence when the oracle is known to be speaking;
    # it has failed silently three times and each looked like a clean negative.
    # The guest also just drops a decode now and then -- observed on a capture it
    # had decoded minutes earlier -- so the health check gets the same retries the
    # arms get. One success proves the oracle speaks; that is all it has to prove.
    healthy = False
    for i in range(a.attempts):
        print(f"\n== oracle health, run {i + 1}/{a.attempts} ==")
        st = _sh(str(ROOT / "oracle" / "vm" / a.selftest), str(a.wait))
        print(_pmon_section(st))
        if "ORACLE OK" in st:
            healthy = True
            break
    if not healthy:
        print("\n" + UNAVAILABLE)
        return 2

    ok = True
    if wants("100"):
        p = _arm(pos, "100 Bd positive", a.runner, a.wait, a.attempts)
        n = _arm(neg, "100 Bd negative (data field corrupted)", a.runner, a.wait,
                 a.attempts)

        # The guest is nondeterministic: it drops a whole decode now and then, and
        # a run where PMON said nothing at all is not a negative result -- it is no
        # result. The first 1.84 s of both files are the same samples, so a
        # negative run only counts once PMON has printed the connect out of that
        # shared prefix, proving it was listening and reached the data.
        pos_ok = any(EXPECT in o for o in p)
        conclusive = [o for o in n if "###CONNECT" in o]
        neg_clean = not any(EXPECT in o for o in n)
        print()
        print(f"  [{'PASS' if pos_ok else 'FAIL'}] PMON prints {EXPECT!r} from our"
              f" packet ({sum(EXPECT in o for o in p)}/{len(p)} runs)")
        if not conclusive:
            print("  [INCONCLUSIVE] no negative run reached the connect -- rerun; "
                  "the corrupted arm has not been shown anything")
            return 1
        print(f"  [{'PASS' if neg_clean else 'FAIL'}] corrupted field does NOT print"
              f" it ({len(conclusive)}/{len(n)} runs reached the connect)")
        if pos_ok and not neg_clean:
            print("\nTHE TEST IS BROKEN -- both arms 'pass', so the gate proves "
                  "nothing.")
            return 1
        ok &= pos_ok and neg_clean

    if wants("200"):
        outs = _arm(p200, "200 Bd", a.runner, a.wait, a.attempts)
        print()
        ok &= _verdict_200(outs)

    if wants("live"):
        outs = _arm(plive, "live transmit path", a.runner, a.wait, a.attempts)
        print()
        ok &= _verdict_live(outs)

    if wants("shift"):
        outs = _arm(pshift, "shift position", a.runner, a.wait, a.attempts)
        print()
        ok &= _verdict_shift(outs)

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    if not STAGE:
        pytest.skip(NO_STAGE)
    if ROOT is None:
        pytest.skip(NO_ORACLE)
    rc = main([])
    if rc == 2:
        pytest.skip(UNAVAILABLE)
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(main())
