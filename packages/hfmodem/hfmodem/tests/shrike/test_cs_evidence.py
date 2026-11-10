# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What a listen window that decoded nothing is allowed to say.

On 2026-08-14, calling N5UXT on 14110.0 kHz, shrike printed `energy in the FSK
bins but not shaped like a control signal (QRM)` on 18 of 21 cycles. The operator
at the rig, asked what the channel sounded like, said "quiet -- no real QRM". The
draft reading of that session -- "the channel was dirty, so the sample was never
fair to PACTOR" -- was built out of those lines and nothing else, and only the
question to the operator kept it out of the record.

Two things are held here, and the second matters more:

  * a quiet channel reads as quiet, on the audio of that very run;
  * no line names a cause it cannot separate from noise. The middle state reports
    energy at the tones and refuses to say whose it is.

Going quiet by going deaf is not a fix, so the positive control runs first and it
is committed: the six sessions of 2026-07-30 in which WS8EOC answered every cycle,
136 receive windows through this station's own rig and codec.

Run: python -m hfmodem.tests.shrike.test_cs_evidence
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import rxfront
from hfmodem.tests.kestrel import corpora

FS = rxfront.FS
REPO = Path(__file__).resolve().parents[5]

#: The sessions of 2026-07-30 in which a real gateway answered every cycle.
#: Named rather than globbed: `captures/` is ignored and holds whatever the last
#: run wrote, so a glob measures the machine and these six measure the checkout.
ANSWERED = [REPO / "captures" / n for n in
            ("onair-0730-2036", "onair-0730-2037", "onair-0730-2039",
             "onair-0730-2047", "onair-0730-2049", "onair-0730-2050")]

#: The 21 listen windows of the run that produced the false QRM, exactly as the
#: session handed them to `cs_evidence`. Not committed -- `captures/` is ignored
#: and these were written by the run itself.
INCIDENT = REPO / "captures" / "onair-0814-1034"

#: Recorded band noise: five minutes of 80 m that the monitor logged zero
#: detections in, 6950 kHz outside the amateur allocation entirely, and the
#: shared corpus's noise fixture, which is neither this rig nor this continent.
QUIET = (REPO / "working" / "rx-20260813-231019-3595000" / "audio" / "rx-000.wav",
         REPO / "working" / "detector-evidence-20260811" / "control-6950-outofband.wav",
         corpora.REGRESS_FIXTURES / "neg_noise_chatham.wav")

#: How much of each quiet recording to slice. The whole 3595.0 kHz session is
#: 983 windows and was measured over in full for `P1_ENERGY_MIN_MS`; the suite
#: reads the first minute of it.
QUIET_S = 60.0
#: What a keyed cycle leaves to listen in -- the window the incident was decided
#: on, and shorter than the hushed cycle's 1.25 s.
LISTEN_S = 0.25

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _windows(path: Path, seg_s: float = LISTEN_S, limit_s: float = QUIET_S):
    a = rxfront.load_wav(str(path))[:int(limit_s * FS)]
    n = int(seg_s * FS)
    return [a[i:i + n] for i in range(0, len(a) - n, n)]


def _run_ms(seg: np.ndarray) -> float:
    return max((d for _, d in rxfront._p1_runs(rxfront._cs_profile(seg))),
               default=0.0) * 1000


def _quiet(line: str) -> bool:
    return "nothing at the FSK tones" in line


def _close(line: str) -> bool:
    return "CLOSE" in line


def main() -> int:
    if not any(d.is_dir() for d in ANSWERED):
        print(f"  [SKIP] no on-air sessions under {REPO / 'captures'}")
        return 2

    print("\nA gateway that answered is still heard")
    heard = []
    for sd in ANSWERED:
        for f in sorted(sd.glob("*.wav")):
            seg = rxfront.load_wav(str(f))
            heard.append((f"{sd.name}/{f.name}", seg, rxfront.cs_evidence(seg),
                          bool(rxfront._p1_cs_bursts(seg))))
    bursty = sum(x[3] for x in heard)
    close = sum(_close(x[2]) for x in heard)
    check("the committed sessions are all here", len(heard) == 136, f"{len(heard)} windows")
    check("the gateway's bursts still reach the CLOSE line", close == bursty >= 79,
          f"{close} of {len(heard)} windows, {bursty} with a burst")
    check("...and none of them reads as nothing there",
          not [n for n, _, line, b in heard if b and _quiet(line)])
    # The number in the line is the burst's own length now. It was the raw
    # threshold mask's longest run, which reported 18 ms for a window whose burst
    # the same function was calling control-signal length at 111.
    short = [(n, line) for n, _, line, _b in heard
             if _close(line) and int(line.split("longest run ")[1].split(" ")[0])
             < rxfront.CS_BURST_MS[0]]
    check("...reporting a run at least as long as the burst it found",
          not short, str(short[:2]))

    print("\nThe run the operator heard as quiet")
    if not INCIDENT.exists():
        print(f"  [SKIP] {INCIDENT} absent -- the run's own capture directory")
    else:
        segs = [rxfront.load_wav(str(f)) for f in sorted(INCIDENT.glob("rx_*.wav"))]
        lines = [rxfront.cs_evidence(s) for s in segs]
        check("all 21 listen windows are there", len(segs) == 21, f"{len(segs)}")
        check("every cycle reads as nothing at the tones",
              all(_quiet(x) for x in lines),
              next((x for x in lines if not _quiet(x)), ""))
        check("...and none claims interference",
              not [x for x in lines if "QRM" in x])
        runs = [_run_ms(s) for s in segs]
        check("...on a margin, not on the knee",
              max(runs) < rxfront.P1_ENERGY_MIN_MS,
              f"longest run {max(runs):.0f} ms against the {rxfront.P1_ENERGY_MIN_MS} ms "
              "floor -- 11 ms when the floor was set")

    print("\nRecorded band noise, three receivers")
    seen = 0
    for p in QUIET:
        if not p.exists():
            print(f"  [SKIP] {p} absent")
            continue
        seen += 1
        segs = _windows(p)
        lines = [rxfront.cs_evidence(s) for s in segs]
        loud = [x for x in lines if not _quiet(x)]
        check(f"{p.name}: every window reads as nothing at the tones",
              not loud, f"{len(loud)} of {len(segs)}: {loud[:1]}")
        runs = [_run_ms(s) for s in segs]
        check(f"{p.name}: the longest run stays under the floor",
              max(runs) < rxfront.P1_ENERGY_MIN_MS,
              f"{max(runs):.0f} ms against {rxfront.P1_ENERGY_MIN_MS}")
    if not seen:
        print("  [SKIP] no quiet recordings present")

    print("\nWhat the middle line is allowed to say")
    # A real quiet window off the air with 40 ms of the two tones laid into it:
    # past the floor, and a third of the 120 ms a codeword takes. The middle state
    # is the one that used to name a cause, so it is asserted on a signal built to
    # land in it rather than waited for.
    base = min((seg for _, seg, _line, b in heard if not b), key=_run_ms)
    at = len(base) // 3
    t = np.arange(min(int(0.040 * FS), len(base) - at)) / FS
    stub = base.copy()
    stub[at:at + len(t)] += 0.5 * float(np.abs(base).max()) * (
        np.sin(2 * np.pi * 1400 * t) + np.sin(2 * np.pi * 1600 * t))
    line = rxfront.cs_evidence(stub)
    check("a 40 ms two-tone stub is reported as energy", not _quiet(line), line)
    check("...not as a control signal", not _close(line), line)
    check("...and is not attributed", "not attributed" in line, line)
    check("...naming the two it cannot separate",
          "interference" in line and "could not read" in line, line)
    # The other half of the same defect, and the older half: the quiet line used
    # to end "nobody answered here", which rules out the answer that arrived in
    # another mode or too weak to profile. Neither line may say who was there.
    check("the quiet line claims only what it measured",
          not [w for w in ("nobody", "answer", "QRM")
               if w in rxfront.cs_evidence(base)],
          rxfront.cs_evidence(base))

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip(f"no on-air sessions under {REPO / 'captures'}")
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
