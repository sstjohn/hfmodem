# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The monitor must agree with an independent decoder on real off-air captures.

Positives and negatives here are labelled by a receive-only reference decoder run
over the overnight W9SSJ corpus. The point is to keep "confirmed" honest as the
decoders change: a capture the reference reads must still produce a decode, and a
capture with no PACTOR in it must never produce one.

CONFIRMABLE means a line shrike would act on -- a connect, a control signal, or a
CRC-valid frame. A `detect`/`fsk` line states evidence without claiming a decode,
so it is not a false positive; the monitor is allowed to say "something is here".

Captures live in ~/src/radio/rf-corpus and, for what the station recorded itself,
~/src/radio/offair/captures. Neither is in the repo -- between them they are
gigabytes of audio -- and the whole module skips cleanly when they are absent.

Run:  python -m hfmodem.tests.shrike.test_corpus
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hfmodem.tests import evidence
from hfmodem.shrike import rxfront

CORPUS = evidence.CORPUS

# A reference decoder reads this one end to end: KE5YTA connect -> PACTOR-3
# data. shrike must find the connect, and W4DNA's data packets behind it.
#
# The packet count is the sharper of the two and is the only third-party PACTOR-1
# traffic anywhere in the corpus that shrike can read, so it is worth an exact
# number rather than a floor. The station retransmits its announcement on six
# consecutive cycles at t=4.5, 5.5, 7.0, 8.0, 9.5 and 11.0, and the last of those
# was being lost to TWO guards that had no business in front of a CRC:
#
#   * `_fsk_present`, whose width test measures the whole 2 s hop rather than the
#     signal at the FSK tones -- so the 1000 Hz transmission that starts up beside
#     this packet reads the window 1000 Hz wide and vetoes it, while the PACTOR
#     tones themselves stand 20 and 30 times over the noise floor;
#   * the `last_fsk` debounce, which the `p1reply` PRESENCE line at t=10.50 armed.
#     That line explicitly does not claim to read the burst it saw, and it was
#     hiding one that could be read 0.5 s behind it.
#
# So this pins a rule as much as a count: a report that knows nothing must never
# suppress one that knows something.
POSITIVES = {"7101k_234600": ("KE5YTA", 6)}

# The reference is silent on these AND they cannot be PACTOR: FT8, and signals whose
# occupied width is impossible for the mode a looser monitor once claimed.
NEGATIVES = ["rig_rx", "7101k_001603", "7104k_001443", "14104k_030841",
             # Every PACTOR-3 claim withdrawn on 2026-07-31, kept so they cannot
             # come back. The first is FT8 on 14109 kHz, where the header path
             # reported a CRC-valid PACTOR-3 header and then, from a data path
             # added the same day, a five-byte payload found in 19,484 trials.
             "14109k_131254",
             # The other three are dwells on a Winlink PACTOR gateway that were
             # read as PACTOR-3 on the strength of a 120 Hz tone comb -- 13 to 15
             # tones of 17. The comb is the receiver site's own birdies, on a
             # ladder of 90 and 150 Hz whose autocorrelation reads 90, 120 or
             # 240 Hz depending which second is measured, and it is as strong in
             # the 1.96 s recorded before each burst as in the burst. In-band
             # power over that lead-in is +0.15, +0.04 and -0.32 dB, and an
             # independent decoder handed all three at full input amplitude
             # reports nothing. There is no PACTOR in these files.
             "20260731T153338Z_TI0BCR-14.113MHz-Pactor34",
             "20260731T153400Z_TI0BCR-14.113MHz-Pactor34",
             "20260731T153425Z_TI0BCR-14.113MHz-Pactor34"]

OFFAIR = Path.home() / "src/radio/offair/captures"
OFFAIR_NEGATIVES = ["20260731T224322Z_KL7RI-3.587MHz-Pactor3",
                    "20260731T224622Z_KL7RI-3.587MHz-Pactor3"]
"""Two dwells on an 80 m gateway that answered nothing, the fourth of the four
PACTOR-3 claims withdrawn on 2026-07-31. They live beside the captures rather
than in the corpus because that is where the station writes them, and a negative
is worth keeping wherever it was recorded."""

CONFIRMABLE = {"connect", "cs", "packet"}
ok = True
CHECKS = 0


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok, CHECKS
    CHECKS += 1
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def main() -> int:
    if not CORPUS.is_dir():
        print(f"  [SKIP] corpus not present at {CORPUS}")
        return 2

    for name, (callsign, packets) in POSITIVES.items():
        wav = CORPUS / f"{name}.wav"
        if not wav.exists():
            print(f"  [SKIP] {name} absent")
            continue
        evs = list(rxfront.decode_events(rxfront.load_wav(str(wav))))
        got = [e for e in evs if e.kind == "connect" and callsign in e.text]
        check(f"reference-positive {name} still decodes {callsign}", bool(got),
              f"{len(evs)} events, no {callsign}" if not got else "")
        p1 = [e for e in evs if e.kind == "packet" and e.protocol == "PACTOR-1"]
        check(f"reference-positive {name} reads all {packets} of its data packets",
              len(p1) == packets,
              f"{len(p1)} at " + ", ".join(f"{e.t:.2f}" for e in p1))
        # ...and the PACTOR-3 side of the same capture, counted SEPARATELY. Both
        # protocols print `[CRC-VALID]`, so a reader totalling those lines sees more
        # than the confirmation rule leaves and reads the difference as a lost
        # PACTOR-1 packet. It is not.
        #
        # Four packets: the upgrade and three repeats of its first traffic. The entry is
        # speed level 1, counter 2, data type 0, with a field of pure IDLE that the
        # receiver drops like any other fill -- so what identifies it is the status
        # byte and not the padding. Behind it is `Q-6.0`, which an independent
        # monitor reads off this same audio three times over. `p3rx.VH_FIT` stood
        # above that packet's anchor until 2026-09-01, on the reading that its
        # CRC-valid five bytes were the coincidence that fixed the threshold; the
        # five bytes are the payload, and this capture carries a PACTOR-3 phase.
        # Preserving measured carrier order independently of request-status bit 0
        # recovers all three repeats. They keep status 0x21 while the physical
        # order alternates home/swapped/home at 21.185, 22.435 and 23.6825 seconds.
        # The pre-existing independent monitor transcript records the same three
        # Q-6.0 payloads (FRNR 6-8).
        p3 = [e for e in evs if e.kind == "packet" and e.protocol != "PACTOR-1"]
        check(f"reference-positive {name} reads its entry packet and its traffic",
              [e.packet for e in p3] == [(1, 0x02, b'', True)]
              + [(1, 0x21, b'Q-6.0', True)] * 3
              and [e.carrier_swapped for e in p3[1:]] == [False, True, False]
              and all(e.cycle_long is False for e in p3),
              "; ".join(f"{e.t:.2f} {e.text[:48]}" for e in p3) or "none")

    for root, name in ([(CORPUS, n) for n in NEGATIVES]
                       + [(OFFAIR, n) for n in OFFAIR_NEGATIVES]):
        wav = root / f"{name}.wav"
        if not wav.exists():
            print(f"  [SKIP] {name} absent")
            continue
        evs = list(rxfront.decode_events(rxfront.load_wav(str(wav))))
        bad = [e for e in evs if e.kind in CONFIRMABLE]
        check(f"reference-negative {name} claims no decode", not bad,
              "; ".join(f"{e.kind}:{e.text[:34]}" for e in bad[:3]))

    print("\nALL PASS" if ok else "\nFAILED")
    # Nothing ran. Individual fixtures `continue` when absent, so a file
    # whose corpus directory exists but whose recordings do not would
    # otherwise report a green pass having asserted nothing — which is the
    # exact difference between "we checked" and "we could not".
    if CHECKS == 0:
        print("  [SKIP] nothing was checked — no fixture was present")
        return 2
    return 0 if ok else 1


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip(f"corpus not present at {CORPUS}")
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
