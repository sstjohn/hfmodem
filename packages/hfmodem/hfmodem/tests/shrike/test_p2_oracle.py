# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Oracle gate for PACTOR-2 transmit: an independent monitor must read our air.

Every other convention in the PACTOR-2 chain is graded by a decode, and a
constellation's ORIGIN is the one that cannot be. `p2rx.decode_burst` enumerates
every half-sector rotation of each carrier, so it absorbs any constant turn of
the alphabet, and `p2rx.burst_window` measures the data grid from the marker time
our own transmitter put there. Transmit and receive cancel a shared error and the
round trip stays green while the waveform is wrong -- which is how this project
once carried two wrong parameters at once. Only an outside decoder settles it.

WHAT EACH ARM ANSWERS

  health    `hb9ak_055246_c1500.wav` itself, a real off-air PACTOR-2 link, ahead
            of every other arm and after the PACTOR-1 selftest. The selftest says
            PMON is running; only this says the guest has a level-2 decoder at
            all, and without it a silent arm cannot be told from a monitor that
            was never going to read one. When it comes back empty nothing else
            runs: the answer is CANNOT SAY, not FAIL.
  replica   The same 32 bursts re-rendered by our own transmitter, field for
            field, at the same anchors and in the same arrangement, so the two
            differ only in what our transmitter does and the arm passes on the
            payload TEXT rather than on frame counts.
  short     Speed levels 1 to 4 on the 1.25 s cycle, eight cycles each, keyed by
            `onair.RadioTx` inside one link that opens in PACTOR-1 -- the object
            a station drives, with the counter running and the carrier swap
            alternating across the level changes.
  long      The same four levels on the 3.75 s cycle, `send_p2_long_packet`,
            four cycles each.
  entry     `send_p2_entry_packet`, the `p2sl1` rung, twice: behind a PACTOR-1
            link setup and cold with nothing in front of it. The rung's whole
            job is to be acquired by a station that has heard no PACTOR-2, so
            the cold segment is the one that matches the air.
  controls  `test_p2link.render` -- a two-sided link with a peer's codewords in
            the answer slot. It grades OUR PACKETS under a peer transmitting at
            `cs_slot`; the codewords themselves cannot be graded here at all.
  negative  Eight cycles at level 2 with the field mangled under the CRC the
            good field computed. Nothing may be read out of it, and the run only
            counts if the PACTOR-1 connect in front of it came back -- otherwise
            the arm has not been shown anything.
  sl4       Sixteen turns of the level-4 alphabet in one link, four cycles each.
            Not part of `all`: it is a measurement rather than a gate, and it is
            the one lever left on the only level no outside decoder has read.

WHAT THIS GATE CANNOT SAY, and it is not a failure of the arm: PMON reports a
forward channel and has no output of any kind for a control signal. Its whole
vocabulary is `###CONNECT`, `###PLISTEN`, `###STATUS`, `###PAYLOAD1/2/_END` and
its amplitude report. `PIII_Complete_1` carries PACTOR-3 codewords this project
measured off it (pactor3.md s7) and PMON printed not one line about them. So a
codeword's verdict from here is CANNOT SAY, permanently, whatever it is keyed
like -- only a station that answers can grade a reverse channel.

VERDICT, 2026-08-03 -- PASS at speed levels 1, 2 and 3. The replica arm returns
the recording's own payload stream, "is a / pointless / exercise / as / they /
collect / every week / of the / year! / the / moment with a", in order, from our
transmitter -- and the recording is a keyboard QSO between two commercial modems.

HOW THE THREE ORIGINS WERE FOUND, since none of them is in a document. Each level
was transmitted at a sweep of alphabet origins, one oracle run per feed, and the
monitor's own verdict read off the segment:

  SL1  eight origins 45 degrees apart. Read at 90, 135 and 180 and silent at 45,
       225, 270, 315 and 0. A DBPSK demapper accepts within 90 degrees of its
       axis, so an open window of exactly (45, 225) puts that axis at 135 with no
       room either side. The PI/4 of [SCS-P4] s11.7 is the boundary itself, which
       is why an earlier feed's channel softs came back three-quarters ZEROED
       rather than wrong -- a symbol on the boundary decides nothing.
  SL2  four origins 90 degrees apart, then four more at 22.5. Read at 247.5, 270
       and 292.5, silent at 225 and 315: a window of exactly one sector, so the
       origin is its centre and not a fit.
  SL3  measured off air first -- `test_p2.py` reads the alphabet straight off the
       HB9AK recording against its frame marker -- and confirmed here. The old
       origin, one whole sector away, returns NOTHING through this gate: 0 of 32
       bursts where the corrected one returns all 32.

VERDICT, 2026-09-17 (D-1a) -- PASS at speed levels 1, 2 and 3 on BOTH cycle
lengths, and for the entry rung cold. Every cycle offered came back at
`Level: 2`, at the level it was sent at, carrying its own payload -- eight of
eight short cycles per level, four of four long ones, with `CYC` following the
frame length, and the `p2sl1` entry packet 8 of 8 behind a link setup and 8 of 8
with nothing at all in front of it. What is new is that the question is put to
the TRANSMIT PATH rather than to a render of it, at all four levels and both
lengths, that the entry rung is asked cold, and that a null is separated from a
dead oracle by controls at each end: the health arm before anything is scored,
the negative arm after.

Speed level 4 is SILENT at both lengths, and the verdict there is CANNOT SAY
rather than FAIL, because nothing can show this build has a level-4 decoder --
no PACTOR-2 level-4 material exists to try it on. The `sl4` arm is what narrows
it: sixteen turns of the 16-DPSK alphabet, four cycles each, read none, on
either reading of the rate-7/8 puncture.

Set PMON_ORACLE to the tree holding the runner and its images, and PMON_STAGE to
a directory the guest mounts.

Run:  PMON_STAGE=<dir the guest mounts> PMON_ORACLE=<oracle tree> \\
      .venv/bin/python -m hfmodem.tests.shrike.test_p2_oracle
      .venv/bin/python -m hfmodem.tests.shrike.test_p2_oracle --render-only
      .venv/bin/python -m hfmodem.tests.shrike.test_p2_oracle --arm short
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests import evidence
from hfmodem.tests.shrike import test_p2link
from hfmodem.shrike import coding, onair, p2rx, pactor2, spec

NO_STAGE = ("set PMON_STAGE to a directory the oracle guest mounts -- without one "
            "the VM is fed a path it cannot read, which looks exactly like a "
            "signal that did not decode")
NO_ORACLE = "set PMON_ORACLE to the tree holding the oracle runner and its images"
UNAVAILABLE = "the oracle guest did not run"
NO_LEVEL2 = ("this guest read nothing off real off-air PACTOR-2, so it cannot "
             "grade ours. Every arm would return CANNOT SAY rather than a "
             "verdict, and none of them was run.")

ORACLE_ROOT = os.environ.get("PMON_ORACLE")
ORACLE = Path(ORACLE_ROOT) / "oracle" / "vm" if ORACLE_ROOT else None
STAGE = os.environ.get("PMON_STAGE")
OUTDIR = Path(STAGE) if STAGE else None

RECORDING = evidence.CORPUS / "p2hunt" / "hb9ak_055246_c1500.wav"
FS = 48000
CYCLES = 8
LONG_CYCLES = 4
LEVELS = (1, 2, 3, 4)
AMP = test_p2link.AMP
CYCLE = spec.CYCLE_SHORT_S

def _payload(path: pactor2.Path) -> bytes:
    """This path's payload, filled to the exact length its geometry carries.

    Legible in a transcript and naming its own path, which is what lets one
    packed link be read level by level: a decode that names the wrong one names
    itself. `SL1-short` carries five bytes and gets what fits.
    """
    n = path.crc_bytes - 3
    text = f"P2 {path.name} SHRIKE 0917 ".encode()
    return (text * (n // len(text) + 1))[:n]


def _station():
    """A transmit path whose keyings land in a list instead of on the air.

    The carrier swap alternates ON THE OBJECT, so one segment has to be keyed by
    one station from its first cycle to its last or the arrangement it goes out
    in is not a link's.
    """
    keyed: list[np.ndarray] = []
    tx = onair.RadioTx(None, transmit=False, outdir=Path("."))
    tx._tx = lambda audio, what, **kw: keyed.append(audio)

    def key(method: str, *args) -> np.ndarray:
        getattr(tx, method)(*args)
        return keyed[-1]
    return key


def _lead() -> float:
    """Seconds by which a shaped burst precedes its own phase reference.

    Every PACTOR-2 keying goes out this much early, so a burst placed at a grid
    boundary has to start before it -- `test_p2link.render` places them the same
    way and so does the transmit path on the air.
    """
    return pactor2.pulse_lead(FS) / FS


def _mix(items: list[tuple[float, np.ndarray]], tail: float = 2.0) -> np.ndarray:
    out = np.zeros(int(max(t * FS + x.size for t, x in items) + tail * FS))
    for t, x in items:
        at = int(round(t * FS))
        out[at:at + x.size] += x
    return out


def _link(long_frames: bool) -> np.ndarray:
    """One link that opens in PACTOR-1 and walks all four speed levels.

    The counter runs on across the level changes because it belongs to the link,
    and the swap alternates for the same reason: what is being graded is what a
    station keys in a session, not four independent renders.
    """
    paths = pactor2.PATHS_LONG if long_frames else pactor2.PATHS
    count = CYCLES if not long_frames else LONG_CYCLES
    step = 3 * CYCLE if long_frames else CYCLE
    send = "send_p2_long_packet" if long_frames else "send_p2_packet"
    items, key, t, n = test_p2link.preamble(), _station(), 2 * CYCLE, 2
    for sl in LEVELS:
        for _ in range(count):
            items.append((t - _lead(), AMP * key(
                send, sl, _payload(paths[sl - 1]), spec.status_byte(n % 4))))
            t += step
            n += 1
    return _mix(items)


@contextlib.contextmanager
def _turned(sectors: int):
    """`pactor2.cell_steps` with the 16-DPSK alphabet turned `sectors` on.

    A constant added to every differential step is exactly what an alphabet
    origin is -- it turns the whole ladder and nothing else -- and it is the one
    parameter at speed level 4 that has never been measured against anything.
    SL1's and SL2's were found this way, one transmission per candidate, and
    SL3's was measured off the air and agreed.
    """
    base = pactor2.cell_steps

    def turned(bits_per_cell: int) -> np.ndarray:
        steps = base(bits_per_cell)
        if bits_per_cell == 4:
            steps = (steps + sectors * 2 * np.pi / 16) % (2 * np.pi)
        return steps

    pactor2.cell_steps = turned
    try:
        yield
    finally:
        pactor2.cell_steps = base


@contextlib.contextmanager
def _swapped_puncture(path: pactor2.Path, name: str):
    """This path with `coding.<name>` in place of the rate it ships with.

    Annex I's rate-7/8 vector reads two ways and only one of them can be on the
    air. `coding.PUNCTURE_7_8_INTERLEAVED` is the candidate the shipped one was
    chosen over, kept for exactly this.
    """
    want = getattr(coding, name)
    if want is path.puncture:
        yield
        return
    turned = dataclasses.replace(path, puncture=want)
    paths = tuple(turned if p is path else p for p in pactor2.PATHS)
    base, pactor2.PATHS = pactor2.PATHS, paths
    try:
        yield
    finally:
        pactor2.PATHS = base


def sl4_sweep_feed(puncture: str = "PUNCTURE_7_8") -> np.ndarray:
    """All sixteen turns of the level-4 alphabet, four cycles each, in one link.

    Speed level 4 is the one waveform in PACTOR-2 no outside decoder has ever
    read, and `cell_steps` says why it might not be readable: its origin is SL3's
    rule carried over, with no level-4 material anywhere to check it against. If
    the monitor reads exactly one of the sixteen, that turn IS the measurement;
    if it reads none, the origin is not what stands in the way and the next
    parameter is the rate-7/8 puncture, which this takes by name.

    Each segment says which turn it is in its own payload, so a decode names
    itself and the file can be read without counting seconds.
    """
    items, n = test_p2link.preamble(), 2
    with _swapped_puncture(pactor2.PATHS[3], puncture):
        fill = pactor2.PATHS[3].crc_bytes - 3
        for turn in range(16):
            key = _station()
            text = f"P2 SL4 TURN {turn:02d} SHRIKE 0917 ".encode()
            with _turned(turn):
                for c in range(LONG_CYCLES):
                    t = (2 + (turn * LONG_CYCLES + c)) * CYCLE
                    items.append((t - _lead(), AMP * key(
                        "send_p2_packet", 4,
                        (text * (fill // len(text) + 1))[:fill],
                        spec.status_byte(n % 4))))
                    n += 1
    return _mix(items)


def entry_feed() -> np.ndarray:
    """The `p2sl1` rung twice: inside a link setup, then cold.

    Cold is the arrangement the rung actually flies in -- a peer that granted an
    entry has heard PACTOR-1 and nothing else -- and it is the harder of the two
    to acquire. The two segments carry different payloads so a decode says which
    one it came out of without counting seconds.
    """
    items = test_p2link.preamble()
    key = _station()
    for c in range(CYCLES):
        items.append(((2 + c) * CYCLE - _lead(),
                      AMP * key("send_p2_entry_packet", b"LINKD",
                                spec.status_byte(c % 4))))
    t0 = (2 + CYCLES) * CYCLE + 6.0
    key = _station()
    for c in range(CYCLES):
        items.append((t0 + c * CYCLE - _lead(),
                      AMP * key("send_p2_entry_packet", b"COLD ",
                                spec.status_byte(c % 4))))
    return _mix(items)


def negative_feed() -> np.ndarray:
    """Level 2, field mangled under the CRC the good field computed.

    Bypasses the transmit path deliberately: what has to be corrupted is the
    field, and the transmit path builds its own. Everything else -- the link
    setup in front, the grid, the swap -- is what the `short` arm carries.
    """
    path = pactor2.PATHS[1]
    items = test_p2link.preamble()
    for c in range(CYCLES):
        good = pactor2.build_field(
            _payload(path) + bytes([spec.status_byte(c % 4)]), path)
        field = bytes(b ^ 0x5A for b in good[:-2]) + good[-2:]
        items.append(((2 + c) * CYCLE - _lead(),
                      AMP * pactor2.data_burst(field, path,
                                               swapped=bool(c & 1), fs=FS)))
    return _mix(items)


def replica_feed() -> tuple[np.ndarray, int]:
    """The recording's own bursts, re-rendered by our transmitter.

    Returns the audio and the number of grid slots our own receiver could not
    read a field out of. Those slots go out carrying the previous field, so a
    payload that rode only on one of them cannot come back, and the arm allows
    exactly that many misses and no more."""
    audio, fs = p2rx._read_wav_mono(str(RECORDING))
    grid, _bin_pair, arms = p2rx.burst_grid(audio, fs)
    decoded = dict(p2rx.decode_bursts(audio, fs))
    path = pactor2.PATHS[2]

    # Where our own marker lands inside a rendered burst, so each one can be put
    # where the recording's marker was rather than where its first sample was.
    lead = {}
    for swapped in (False, True):
        one = np.concatenate([np.zeros(FS // 2),
                              pactor2.data_burst(next(iter(decoded.values())),
                                                 path, swapped=swapped, fs=FS),
                              np.zeros(FS // 2)])
        lead[swapped] = max(p2rx.find_markers(one, FS, 0.90),
                            key=lambda h: h[4])[0] - 0.5

    bursts, last = [], None
    for t, swapped in zip(grid, arms):
        field = decoded.get(t, last)
        if field is None:
            continue
        last = field
        bursts.append((t - lead[swapped],
                       pactor2.data_burst(field, path, swapped=swapped, fs=FS)))
    return _mix(bursts, 3.0), len(grid) - len(decoded)


def _write(dest: Path, audio: np.ndarray) -> Path:
    pcm = (np.clip(audio / np.abs(audio).max() * 0.55, -1, 1) * 32767)
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(np.repeat(pcm.astype(np.int16), 2).tobytes())
    return dest


# --- running the guest -------------------------------------------------------

# PMON's console carries raw bytes that are not UTF-8, so a strict decode raises
# partway through and takes the gate down before it has run anything.
def _sh(*argv: str, timeout: int | None = None) -> str:
    r = subprocess.run(["bash", *argv], capture_output=True, text=True,
                       errors="replace", timeout=timeout)
    return r.stdout + r.stderr


def _lines(out: str) -> list[str]:
    """PMON's console with its periodic amplitude report taken out. That report
    is not framed against anything and lands wherever its timer fires, including
    inside a payload."""
    return [ln for ln in out.splitlines() if "INPUT AMPLITUDE" not in ln]


def _run(wav: Path, label: str, wait: int) -> str:
    print(f"\n== {label} arm ==")
    out = _sh(str(ORACLE / "p2-run.sh"), str(wav), str(wait), timeout=wait + 600)
    _, _, tail = out.partition("===== PMON OUTPUT =====")
    print("\n".join(_lines(tail)).strip() or "(nothing after the marker)")
    return out


_STATUS = re.compile(r"SL:\s*(\d+),\s*CYC:\s*(\d+)")
_LEVEL = re.compile(r"###PLISTEN: Level: (\d)")


def _frames(out: str) -> list[tuple[int, int, int, str]]:
    """(protocol level, speed level, cycle flag, payload text) per frame.

    The level is half the claim and only `###PLISTEN` carries it, the speed level
    and the cycle flag only `###STATUS`, and the payload only the lines between
    `###PAYLOAD2` and its end -- so a frame is read off three kinds of line and
    the state between them.
    """
    frames, lines = [], _lines(out)
    level = sl = cyc = None
    for i, ln in enumerate(lines):
        lvl = _LEVEL.search(ln)
        st = _STATUS.search(ln)
        if lvl:
            level = int(lvl.group(1))
        elif st:
            sl, cyc = int(st.group(1)), int(st.group(2))
        elif ln.startswith("###PAYLOAD2:") and sl is not None:
            body = []
            for nxt in lines[i + 1:]:
                if nxt.startswith("###PAYLOAD_END"):
                    break
                body.append(nxt)
            frames.append((level, sl, cyc, "\n".join(body).strip()))
    return frames


def _text(out: str) -> list[str]:
    """The non-empty payload strings PMON printed, in order."""
    return [t for _, _, _, t in _frames(out) if t]


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


def _verdict_levels(out: str, label: str, long_frames: bool) -> None:
    """One claim per speed level: read at level 2, at the level it was sent at,
    with its own payload back. The cycle flag is REPORTED, not required -- what
    PMON's CYC field means is its own business and nothing here establishes it."""
    paths = pactor2.PATHS_LONG if long_frames else pactor2.PATHS
    frames = _frames(out)
    for sl in LEVELS:
        want = _payload(paths[sl - 1])[:16].decode()
        hit = [f for f in frames if f[1] == sl and f[0] == 2 and f[3].startswith(want)]
        check(f"{label} SL{sl} reads back at Level 2 as {want!r}", bool(hit),
              f"PMON reported {sorted({(f[0], f[1], f[2]) for f in frames})}")
        if hit:
            print(f"       CYC: {sorted({f[2] for f in hit})}, {len(hit)} frames")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=400)
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--arm", default="all",
                    choices=("all", "replica", "short", "long", "entry",
                             "controls", "negative", "sl4"))
    ap.add_argument("--puncture", default="PUNCTURE_7_8",
                    choices=("PUNCTURE_7_8", "PUNCTURE_7_8_INTERLEAVED"),
                    help="the rate-7/8 reading the sl4 arm keys")
    a = ap.parse_args(argv)
    if OUTDIR is None:
        print(NO_STAGE)
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)

    def wants(name: str) -> bool:
        return a.arm in (name, "all")

    feeds = {}
    if wants("short"):
        feeds["p2_short"] = _link(False)
    if wants("long"):
        feeds["p2_long"] = _link(True)
    if wants("entry"):
        feeds["p2_entry"] = entry_feed()
    if wants("controls"):
        feeds["p2_controls"] = test_p2link.render()
    if wants("negative"):
        feeds["p2_negative"] = negative_feed()
    if a.arm == "sl4":
        feeds["p2_sl4_turns"] = sl4_sweep_feed(a.puncture)
    gaps = 0
    if RECORDING.exists():
        feeds["p2_source"] = p2rx._read_wav_mono(str(RECORDING))[0]
        if wants("replica"):
            feeds["p2_replica"], gaps = replica_feed()
    paths = {name: _write(OUTDIR / f"{name}.wav", audio)
             for name, audio in feeds.items()}
    for name, audio in feeds.items():
        print(f"rendered {name}.wav: {audio.size / FS:.2f} s")
    if a.render_only:
        return 0
    if ORACLE is None:
        print(NO_ORACLE)
        return 2

    # A silent oracle is only evidence when the oracle is known to be speaking.
    # It has failed silently more than once and each time looked like a clean
    # negative, so the run starts by feeding it a capture it is known to decode.
    health = _sh(str(ORACLE / "p2-selftest.sh"), str(a.wait), timeout=a.wait + 600)
    print("\n== oracle health ==\n" + "\n".join(_lines(health)).strip())
    if "SHRIKE-P2 ORACLE OK" not in health:
        print("\n" + UNAVAILABLE)
        return 2

    # The PACTOR-1 selftest above says PMON is running; it does not say this
    # guest has a level-2 decoder, and only a real off-air PACTOR-2 link does.
    # Without it a silent arm cannot be told from a monitor that was never going
    # to read one, so nothing below is run rather than scored.
    want = _text(_run(paths["p2_source"], "the recording itself", a.wait)) \
        if "p2_source" in paths else []
    if not want:
        print("\n" + NO_LEVEL2)
        return 2

    if "p2_replica" in paths:
        said = _text(_run(paths["p2_replica"], "replica", a.wait))
        missing = [w for w in want if w not in said]
        check("the recording's own payload comes back from our transmitter",
              len(missing) <= gaps,
              f"missing {missing} against {gaps} unreadable bursts; "
              f"it said {said}")
        shared = [w for w in want if w in said]
        check("...in the order the link sent it",
              [w for w in said if w in shared] == shared,
              f"it said {said}, the recording said {want}")

    if "p2_negative" in paths:
        out = _run(paths["p2_negative"], "negative (field mangled)", a.wait)
        # The link setup in front of the mangled cycles is the same audio the
        # other arms open on, and PMON reads its PACTOR-1 announcement: that is
        # what says the guest reached the PACTOR-2 bursts and declined them,
        # rather than having been fed nothing. A run without it proves nothing.
        stray = [f for f in _frames(out) if f[0] == 2]
        if "###CONNECT" not in out:
            print("  INCONCLUSIVE negative arm -- it never reached the connect, "
                  "so it has not been shown anything")
        else:
            check("a mangled level-2 field is NOT read back", not stray,
                  f"it read {stray}")

    if "p2_short" in paths:
        _verdict_levels(_run(paths["p2_short"], "short cycle, SL1-4", a.wait),
                        "short", False)
    if "p2_long" in paths:
        _verdict_levels(_run(paths["p2_long"], "long cycle, SL1-4", a.wait),
                        "long", True)
    if "p2_entry" in paths:
        out = _run(paths["p2_entry"], "entry rung", a.wait)
        for tag, label in ((b"LINKD", "behind a PACTOR-1 link setup"),
                           (b"COLD ", "cold, nothing in front of it")):
            check(f"the p2sl1 entry packet reads {label}",
                  any(f[0] == 2 and f[3].startswith(tag.decode().strip())
                      for f in _frames(out)),
                  f"PMON read {_frames(out)}")
    if "p2_sl4_turns" in paths:
        out = _run(paths["p2_sl4_turns"],
                   f"level 4, sixteen alphabet turns, {a.puncture}", a.wait)
        read = sorted({f[3][12:14] for f in _frames(out)
                       if f[0] == 2 and f[3].startswith("P2 SL4 TURN")})
        check(f"one turn of the level-4 alphabet is read ({a.puncture})",
              len(read) == 1, f"turns read: {read or 'none'}")

    if "p2_controls" in paths:
        out = _run(paths["p2_controls"], "two-sided link with codewords", a.wait)
        check("our packets still read with a peer keying the answer slot",
              len([f for f in _frames(out) if f[0] == 2]) >= 6,
              f"PMON read {_frames(out)}")
        print("  CANNOT SAY: the codewords themselves. PMON reports a forward "
              "channel and prints no line of any kind for a control signal.")

    print(f"\n{PASS} passed, {len(FAILURES)} failed")
    return 0 if not FAILURES else 1


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
    sys.exit(main())
