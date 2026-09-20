# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Oracle gate for conformant PACTOR-III: an independent monitor must read a
packet with no acquisition burst in front of it.

Every PACTOR-III packet this project has had judged from outside was preamble-led
-- twenty symbols of the +-800 acquisition burst, then a phase reference, then 67
of the 72 grid rows, the other five given up to the burst. A monitor read those,
which was worth a great deal at the time and settled nothing about what a station
on a link transmits. A real packet, measured off the air, is `[one phase-reference
symbol][the eight-symbol header block][all 72 rows]` and carries no burst at all.

So this asks the one question the old gate could not: offered what shrike now
keys on an established link, does an independent decoder still read it, at the
speed level it was sent at, with the payload byte-exact?

WHAT EACH ARM ANSWERS

  conformant  Five speed levels in one file, each its own segment: a PACTOR-1
              connect and four ARQ cycles of packets rendered by `onair.RadioTx`
              -- the object a station on the air drives, not a transcription of
              it -- so the carrier swap under test is the one the transmit path
              chooses, alternating cycle by cycle.
  negative    The same shape at speed level 3 with the data field mangled under
              the CRC the good field computed. Nothing may be read out of it, and
              it goes FIRST so that a run truncated by the `--wait` cap can only
              cost a positive arm.
  acquire     The conformant packets again with the +-800 burst restored in front
              of the first packet of the link, which is the only place a real
              station could be said to need one. Its value is comparative: if the
              conformant arm reads and this one does too, the burst is not what
              was carrying the old result.

Segments are packed into one file per arm because a VM boot is minutes and PMON
reports frames in time order with the speed level on its own ###STATUS line, so
which segment a decode came from is never in doubt.

VERDICT, 2026-08-01 -- PASS at speed levels 3, 4, 5 and 6 with no acquisition
burst anywhere in the file and all 72 rows on the air. Levels 5 and 6 had never
been read by anything but shrike's own receiver. The monitor printed payloads of
59, 122, 212 and 284 bytes, each on its own ###STATUS line at the level it was
sent at, from a link setup and conformant packets alone. What is *asserted* is a
prefix of each -- 24 bytes cold, 20 locked -- because the monitor drops trailing
IDLE and rewrites CR, so a whole-payload comparison would fail on its rendering
rather than on ours. Read "byte-exact" nowhere in this file. The carrier
swap alternated under all of it and the monitor followed: its own REV field flips
cycle by cycle in step with the header we sent.

Speed levels 1 and 2 pass too, through the `locked` arm and only there, three
runs out of three each. Level 1 took counting the status byte's packet counter to
get, which is a defect this gate had rather than one the modem had; level 2 took
`spec.SUBBAND_LEAD`, which is one the modem had. Nothing is reported-and-not-
required here any more. `NARROW_LEVELS_NOTE` keeps the account, measured apart
from inferred.

The negative arm is what keeps all of that from being a decoder printing whatever
it is shown: offered the same link setup and four cycles whose field was mangled
under the CRC the good field computed, the monitor printed the connect out of the
shared prefix -- so it was listening and it reached the packets -- and no payload
at all.

The oracle guest reads its input through a Lima mount, so the staging directory
has to be one the guest can see. `PMON_STAGE` names it; without it there is no
verdict and the gate says so rather than rendering into a path the VM will find
empty. `PMON_ORACLE` names the tree the runner and its images live in.

Run:  PMON_STAGE=<dir the guest mounts> PMON_ORACLE=<oracle tree> \\
      .venv/bin/python -m hfmodem.tests.shrike.test_p3_oracle
      .venv/bin/python -m hfmodem.tests.shrike.test_p3_oracle --render-only
      .venv/bin/python -m hfmodem.tests.shrike.test_p3_oracle --arm conformant
      .venv/bin/python -m hfmodem.tests.shrike.test_p3_oracle --arm locked
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import frame, onair, pactor1, placement, spec

UNAVAILABLE = "ORACLE UNAVAILABLE -- no verdict. Nothing below would mean anything."
NO_STAGE = ("set PMON_STAGE to a directory the oracle guest mounts -- without one "
            "the VM is fed a path it cannot read, which looks exactly like a "
            "signal that did not decode")
NO_ORACLE = "set PMON_ORACLE to the tree holding the oracle runner and its images"
ORACLE_ROOT = os.environ.get("PMON_ORACLE")
ORACLE = Path(ORACLE_ROOT) / "oracle" / "vm" if ORACLE_ROOT else None
STAGE = os.environ.get("PMON_STAGE")
OUTDIR = Path(STAGE) if STAGE else None

DXCALL = "W1AW"
LEVELS = (2, 3, 4, 5, 6)
CYCLES = 4

OPEN_LEVELS: tuple[int, ...] = ()
"""Levels reported and not required. EMPTY, and it should stay that way.

It held level 2 for as long as level 2 could not be read cold, which was read as
a header too narrow to acquire on -- four of the sixteen constant headers. That
was the wrong explanation. Its comb was staggered wrong (`spec.SUBBAND_LEAD`),
and once that was fixed it acquires from a standing start like any other level,
2 of 2 runs in both cold arms. A level that reads does not get an exemption."""

LOCK_LEVEL = 3
"""The level a narrow one is reached THROUGH. Speed level 3 lights fourteen
channels, which the oracle acquires on from a standing start every time."""

NARROW_LEVELS = (2, 1)
"""The levels the `locked` arm reaches through a wider one.

Level 1 needs it -- its header block is the variable header alone, so nothing
anchors it cold. Level 2 no longer does, and is kept here because a link that
falls back to it is worth exercising, not because it cannot be read otherwise."""

ACQUIRE_OPEN_LEVELS = OPEN_LEVELS
"""The acquisition-led arm holds the same levels as the conformant one.

It briefly did not. A single run of it read levels 3 and 4 and missed 5 and 6,
which reads like the burst costing the punctured levels their margin -- exactly
the argument that took the burst out of the packet, so it was easy to believe.
The next single run read all four and the CONFORMANT arm missed level 6 instead.
The two punctured levels are marginal PER RUN in both arms, and nothing about
that is a property of the burst.

Which is what `--attempts` is for, and why its default is 2 rather than 1: a
level counts if any run reads it. Marking these open would have recorded a
coin toss as a finding.
"""

OPEN_LEVELS_LOCKED: tuple[int, ...] = ()
"""Nothing is reported-and-not-required here any more.

Both narrow levels read through a locked link, so the arm holds both to it. Level
2 was on this list for as long as the transmitter sent its six carriers on one
symbol clock, which is not what a real one does -- `NARROW_LEVELS_NOTE`."""

NARROW_LEVELS_NOTE = """What is known about speed levels 1 and 2, separated into
what was measured and what is inferred from it.

A conformant packet carries its header block on the speed level's own channels;
that is what five real speed-level-3 packets do, and at fourteen channels the
oracle reads it from a standing start. Level 2 has six channels, four carrying
constant headers, and level 1 has none at all -- its block is the variable header
alone. Offered either COLD, the oracle reads nothing, with the acquisition burst
in front of it or without. Four candidate blocks were put to it at level 2, one
segment each, four cycles apiece, and only a block widened to all sixteen non-VH
channels was read.

Widening it would have been the wrong fix. The only clean off-air sample of a
header block puts it on the level's own channels and nowhere else, and fitting a
transmitter to one monitor's acquisition against a direct measurement is how a
wrong waveform gets frozen in.

A real link never meets these levels cold: it negotiates upward from PACTOR-1 and
falls back under fading, so by the time such a packet arrives the decoder has been
tracking the raster for cycles. That is what the `locked` arm offers.

MEASURED, through the locked path: both levels read, and the arm requires both.
Level 1 needed the status byte's modulo-4 packet counter made to count -- its
silence was never the header block. Level 2 needed the transmitter to stop sending
its six carriers on one symbol clock, which is the rest of this note.

FIRST, MEASURE THE INSTRUMENT. The oracle has been seen to vary at level 2: one
file scored 5 or 7 accepted frames run to run, bimodally, while level 3 scored 12
of 12 every time. So a level 2 count is worth nothing without repeats, and every
figure here has them. That spread has not been seen since the stagger landed --
six runs, every one of them a full score -- which suggests the variability was a
marginal signal rather than a property of the decoder, and is not a reason to
start quoting single runs.

WHAT WAS WRONG. Level 2's six carriers are two three-tone clusters and they are
not transmitted together; shrike sent them together. `spec.SUBBAND_LEAD` states
the offset, its sign, and what is measured about it against what is inferred.
The recorded waveform supplies an independent timing measurement:

  * Two real off-air level 2 packets, with no decoder in the way -- each lit
    channel's own known header word gives that carrier's symbol clock directly.
    `spec.SUBBAND_LEAD` carries those figures.

WHAT IT COST AND WHAT IT BOUGHT. Eight level 2 cycles a file behind eight at level
3, accepted frames per run. The left column is a hand-rolled render holding the
home arrangement; the right is `onair.RadioTx` with the swap alternating, four
cycles in each arrangement, which is a link:

                              hand-rolled       transmit path
    one clock, as we sent      0, 0, 0            0, 0
    staggered                  4, 8, 8            8, 8, 8

Eight of eight is every home cycle and every swapped one. The gate's own `locked`
arm, four cycles at level 2, reads all four in three runs of three.

That single defect is also what the level looked like from every other angle. An
alternating link scored a stable 2 of 8 and holding the swap scored 3 to 5, which
read like two faults; the level was payload-dependent as well, the gate's own
"SL2 SHRIKE CONFORMANT PACKET" scoring 0 where "SL2 CYCLE {i} LOCKED PROBE"
scored 2. Both went with the stagger: the payload above is the gate's, unchanged.

ELIMINATED by direct test on the way, each with a control that behaved, and worth
keeping so nobody spends the time twice: amplitude from 0.10 to 0.95 (so it was
never signal-to-noise), the matched pulse, rolloff and filter span, strides
34/36/69/71, reversed tone order, five rotations of the tone tuple, the
header/data boundary at +1 and +2 idle symbols, no header block, zero and two
phase-reference symbols, reversed row order, negated differential steps, and the
variable-header carrier order reversed. Widening the header block to all sixteen
non-VH channels is within the noise and was never the fix.

Neither level blocks a session, which negotiates upward from PACTOR-1."""

SEGMENT_GAP_S = 4.0
"""Silence between segments. Long enough that PMON has dropped the previous link
before the next connect, so a segment cannot inherit the one before it."""


def payload(sl: int) -> bytes:
    """This level's payload, filled to the exact length its geometry carries.

    Distinct per level and legible in a transcript, which is what lets one packed
    file be read segment by segment: a decode that names the wrong level names
    itself.

    The whole of it: the field holds `payload_short` bytes and the transmit path
    now puts `payload_short` bytes in it. It used to hold one fewer, because a
    length byte of shrike's own took the last one, and this was capped to match.
    """
    n = spec.SPEED_LEVELS[sl].payload_short
    text = f"SL{sl} SHRIKE CONFORMANT PACKET ".encode()
    return (text * (n // len(text) + 1))[:n]


def _status(cycle: int) -> int:
    """The status byte cycle `cycle` carries, counter and all.

    Bits 0-1 are a modulo-4 packet counter, and a station that pins it sends every
    packet claiming to be a repetition of the last -- `arq.py` counts it properly
    and this gate used not to. The cost was not a failed run but a passing one:
    the monitor flagged `RQ: 1` on every cycle after the first, memory-ARQ-combined
    four packets it had been told were the same packet, and printed a payload
    anyway because they genuinely were identical. Change one of them, as a mixed
    speed-level session does, and the combination yields nothing -- which is a gate
    that only worked while it was being fed the one input its defect could survive.
    """
    return spec.status_byte(cycle % 4)


def _live_packets(sl: int, *, acquire: bool = False,
                  first_cycle: int = 0) -> list[np.ndarray]:
    """`CYCLES` packets as the transmit path keys them, in cycle order.

    Rendered through `onair.RadioTx.send_packet` rather than by calling the
    placement layer here, so what the oracle hears is what a station transmits --
    the field it assembles, the status byte it chooses, and above all the carrier
    swap it alternates. A transcription of that would be a second implementation,
    and a gate that agrees with its own transcription proves nothing.

    `first_cycle` continues the packet counter across a speed change, because the
    counter belongs to the link and not to the level.
    """
    out: list[np.ndarray] = []
    tx = onair.RadioTx(None, transmit=False, outdir=Path("."))
    tx._tx = lambda audio, what, **kw: out.append(onair._trim_silence(audio))
    for c in range(CYCLES):
        tx.send_packet(sl, payload(sl), _status(first_cycle + c))
    if acquire:
        out[0] = _acquire_led(sl)
    return out


def _acquire_led(sl: int) -> np.ndarray:
    """Cycle 0 with the +-800 burst in front of it, everything else unchanged."""
    return placement.data_packet(_info(sl), placement.SPEED_PATHS[sl], acquire=True)


def _info(sl: int) -> bytes:
    """The field a station transmits for this level's payload.

    From `placement.link_packet`'s own arithmetic rather than a second copy of it,
    so the acquisition-led and corrupted arms carry the same field the conformant
    arm does.
    """
    n = placement.SPEED_PATHS[sl].crc_bytes - 3
    return payload(sl)[:n].ljust(n, bytes([spec.IDLE])) + bytes([_status(0)])


def _corrupt_packets(sl: int) -> list[np.ndarray]:
    """The same packets with the field mangled under the CRC the good one computed.

    Well-formed in every respect except that it does not check out: the header
    block, the timing, the tone set and the carrier swap are all what the good
    packet carries, and only the information bytes differ. That is what makes the
    negative a statement about the CRC rather than about the geometry.
    """
    path = placement.SPEED_PATHS[sl]
    good = placement.build_field(_info(sl), path)
    field = bytes(b ^ 0x5A for b in good[:-2]) + good[-2:]
    grid = placement.interleave(placement.encode_frame(field, path), path).reshape(
        path.n_symbols, len(path.tones), path.bits_per_cell)
    steps = placement.grid_steps(grid, path)
    return [placement.assemble(steps, path, swapped=bool(i & 1))
            for i in range(CYCLES)]


def _segment(packets: list[np.ndarray]) -> np.ndarray:
    """A PACTOR-1 connect and `packets`, one per 1.25 s ARQ cycle.

    A monitor follows a link rather than sweeping for packets, so a segment that
    opens without a connect is a segment it reports nothing for.
    """
    fs = spec.SAMPLE_RATE
    step = int(round(spec.CYCLE_SHORT_S * fs))
    connect = pactor1.connect_signal(DXCALL)
    anchor = len(connect) + int(round(0.20 * fs))
    out = np.zeros(anchor + (len(packets) - 1) * step + max(len(p) for p in packets))
    out[:len(connect)] = connect
    for i, pkt in enumerate(packets):
        at = anchor + i * step
        out[at:at + len(pkt)] += pkt
    return out


def _render(segments: list[np.ndarray]) -> np.ndarray:
    gap = np.zeros(int(SEGMENT_GAP_S * spec.SAMPLE_RATE))
    return np.concatenate([gap] + [x for s in segments for x in (s, gap)]
                          ).astype(np.float32)


def render_conformant(*, acquire: bool = False) -> np.ndarray:
    return _render([_segment(_live_packets(sl, acquire=acquire)) for sl in LEVELS])


def render_negative() -> np.ndarray:
    return _render([_segment(_corrupt_packets(3))])


def render_locked() -> np.ndarray:
    """`CYCLES` cycles at `LOCK_LEVEL`, then `CYCLES` at a narrow one, per segment.

    One continuous grid, so the drop is a speed change inside a link rather than a
    new link -- which is the only way a real station ever reaches these levels.
    The carrier swap carries across the change because `CYCLES` is even, so the
    second half restarts at the parity the first half would have continued into.
    """
    return _render([_segment(_live_packets(LOCK_LEVEL)
                             + _live_packets(sl, first_cycle=CYCLES))
                    for sl in NARROW_LEVELS])


# --- running the guest -------------------------------------------------------

# PMON's console carries raw bytes that are not UTF-8 -- the stray 0xfa either
# side of every report -- so a strict decode raises partway through and takes the
# whole gate down before it has run anything.
def _sh(*argv: str, timeout: int | None = None) -> str:
    r = subprocess.run(["bash", *argv], capture_output=True, text=True,
                       errors="replace", timeout=timeout)
    return r.stdout + r.stderr


def _lines(out: str) -> list[str]:
    """PMON's console with its periodic amplitude report taken out.

    That report is not framed against anything and lands wherever the timer fires
    -- including between a payload and its ###PAYLOAD_END -- so anything reading
    the transcript has to drop it first.
    """
    return [ln for ln in out.splitlines() if "INPUT AMPLITUDE" not in ln]


def _pmon_section(out: str) -> str:
    _, _, tail = out.partition("===== PMON OUTPUT =====")
    return "\n".join(_lines(tail)).strip() or "(nothing after the marker)"


def _arm(wav: Path, label: str, wait: int, attempts: int) -> list[str]:
    outs = []
    for i in range(attempts):
        print(f"\n== {label} arm, run {i + 1}/{attempts} ==")
        out = _sh(str(ORACLE / "run.sh"), str(wav), str(wait), timeout=wait + 600)
        print(_pmon_section(out))
        outs.append(out)
    return outs


_STATUS = re.compile(r"SL:\s*(\d+).*FRNR:\s*(\d+)")
_HEXDUMP = re.compile(r"[0-9A-F]{2}(,[0-9A-F]{2})*$")


def _body(text: str) -> str:
    """PMON's rendering of a field, in whichever of its two forms it chose.

    A field of printable bytes prints as itself; one with anything else in it
    prints as a comma-separated hex dump instead. Which form a level gets is
    nothing to do with the signal -- speed levels 5 and 6 land on the dump because
    a field padded out with IDLE, or one carrying anything else unprintable, lands
    on the dump -- so the transcript is normalised here rather than the gate being
    made to care.
    """
    return (bytes.fromhex(text.replace(",", "")).decode("latin-1")
            if _HEXDUMP.fullmatch(text) else text)


def _frames(out: str) -> list[tuple[int, int, str]]:
    """(speed level, frame number, payload text) for every frame PMON reported.

    Substring matching is not enough here: the speed level is half the claim, and
    only the ###STATUS line carries it.
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
            frames.append((sl, frnr, _body("\n".join(body).strip())))
    return frames


def _verdict(outs: list[str], label: str,
             open_levels: tuple[int, ...] = OPEN_LEVELS) -> bool:
    """One claim per speed level: PMON read our payload, and called it ours.

    Matched on a PREFIX rather than on the whole payload, because PMON's own
    rendering of a field is not byte-for-byte what was sent -- it drops trailing
    IDLE and expands a CR -- and the prefix already names the level twice over,
    once in the text and once in ###STATUS.

    Speed level 2 is reported and not required HERE, because this arm offers it
    cold and nothing reaches it that way; the `locked` arm is where it is
    required. `NARROW_LEVELS_NOTE` is why.
    """
    runs = [_frames(o) for o in outs]
    ok = True
    for sl in LEVELS:
        want = payload(sl)[:24].decode()
        hit = [f for f in runs if any(s == sl and t.startswith(want) for s, _, t in f)]
        mark = "PASS" if hit else ("OPEN" if sl in open_levels else "FAIL")
        ok &= bool(hit) or sl in open_levels
        print(f"  [{mark}] {label}: SL{sl} reads back as "
              f"{want!r} at SL: {sl} ({len(hit)}/{len(outs)} runs)")
    return ok


def _verdict_locked(outs: list[str]) -> bool:
    """Reached through speed level 3, the narrow levels are read or they are not.

    Both are required here, and that is the whole point of the arm: cold they are
    silent, locked they must read, and the difference between those two is the
    claim.

    How much a match is worth is not the same at every level, and the arm says so
    rather than letting a uniform-looking table imply it is. Level 3 carries 58
    payload bytes and level 1 carries FOUR, so a level 1 verdict rests on `SL1 `
    and the level on the ###STATUS line, and on nothing else in the file being at
    that level. That is all a four-byte field can offer; it is not a stronger
    claim dressed as a weaker one.
    """
    runs = [_frames(o) for o in outs]
    ok = True
    for sl in NARROW_LEVELS:
        want = payload(sl)[:20].decode()
        hit = [f for f in runs if any(s == sl and t.startswith(want) for s, _, t in f)]
        required = sl not in OPEN_LEVELS_LOCKED
        mark = "PASS" if hit else ("FAIL" if required else "OPEN")
        ok &= bool(hit) or not required
        print(f"  [{mark}] locked via SL{LOCK_LEVEL}: SL{sl} reads back as "
              f"{want!r} at SL: {sl} ({len(hit)}/{len(outs)} runs)")
    return ok


def _verdict_negative(outs: list[str]) -> bool:
    """A mangled field is not read back -- and the run has to have been listening.

    A run that decoded nothing at all proves nothing here. What makes one
    conclusive is PMON printing the connect out of the segment's own link setup,
    which is the same audio either way: it says the guest reached the packets and
    declined them.
    """
    conclusive = [o for o in outs if "###CONNECT" in o]
    stray = [f for o in conclusive for f in _frames(o)]
    ok = bool(conclusive) and not stray
    if not conclusive:
        print("  [INCONCLUSIVE] no negative run reached the connect -- rerun; "
              "the corrupted arm has not been shown anything")
        return False
    print(f"  [{'PASS' if ok else 'FAIL'}] a mangled speed-level-3 field is NOT "
          f"read back ({len(conclusive)}/{len(outs)} runs reached the connect)")
    for sl, n, t in stray:
        print(f"    unexpected frame: SL {sl}, FRNR {n}, payload {t!r}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--selftest", default="selftest.sh")
    ap.add_argument("--wait", type=int, default=300)
    # Two, because the two punctured levels are marginal per run: at rate 3/4 and
    # 8/9 a single pass reads them or does not, and a level counts if any run
    # reads it. One attempt turns the gate into a coin toss and reports it as a
    # finding -- which it did, twice, in opposite directions.
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--arm",
                    choices=("conformant", "negative", "acquire", "locked", "all"),
                    default="all")
    a = ap.parse_args(argv)

    def wants(name: str) -> bool:
        return a.arm in (name, "all")

    if OUTDIR is None:
        print(NO_STAGE)
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    neg, pos, acq, lock = (OUTDIR / "p3_negative.wav", OUTDIR / "p3_conformant.wav",
                           OUTDIR / "p3_acquire.wav", OUTDIR / "p3_locked.wav")
    files = []
    if wants("negative"):
        files.append((neg, render_negative()))
    if wants("conformant"):
        files.append((pos, render_conformant()))
    if wants("acquire"):
        files.append((acq, render_conformant(acquire=True)))
    if wants("locked"):
        files.append((lock, render_locked()))
    for path, sig in files:
        frame.write_wav(str(path), sig)
        print(f"rendered {path.name}: {len(sig) / spec.SAMPLE_RATE:.2f} s")
    if a.render_only:
        return 0

    if ORACLE is None:
        print(NO_ORACLE)
        return 2

    # A silent oracle is only evidence when the oracle is known to be speaking; it
    # has failed silently three times and each looked like a clean negative.
    healthy = False
    for i in range(a.attempts):
        print(f"\n== oracle health, run {i + 1}/{a.attempts} ==")
        st = _sh(str(ORACLE / a.selftest), str(a.wait))
        print(_pmon_section(st))
        if "ORACLE OK" in st:
            healthy = True
            break
    if not healthy:
        print("\n" + UNAVAILABLE)
        return 2

    ok = True
    if wants("negative"):
        ok &= _verdict_negative(_arm(neg, "negative (data field mangled)",
                                     a.wait, a.attempts))
    if wants("conformant"):
        ok &= _verdict(_arm(pos, "conformant", a.wait, a.attempts),
                       "no acquisition burst")
    if wants("acquire"):
        ok &= _verdict(_arm(acq, "acquisition-led", a.wait, a.attempts),
                       "burst on cycle 0", ACQUIRE_OPEN_LEVELS)
    if wants("locked"):
        ok &= _verdict_locked(_arm(lock, "locked via SL3", a.wait, a.attempts))

    if wants("conformant") or wants("locked"):
        print("\n" + NARROW_LEVELS_NOTE)
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    if not STAGE:
        pytest.skip(NO_STAGE)
    if ORACLE is None:
        pytest.skip(NO_ORACLE)
    rc = main([])
    if rc == 2:
        pytest.skip(UNAVAILABLE)
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(main())
