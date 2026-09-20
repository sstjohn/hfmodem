# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The two shrike lines that report presence, held to what their tests establish.

Both misled an operator on the air, and neither by being wrong about the signal:
`p1reply` ended "a peer answered" on 162 windows of a night when nobody answered,
and the PACTOR-2 line printed a spacing and no frequency for a stranger at
1704/1910 Hz, which was read as the gateway. The corpus regression guards what
they still find; this guards what they still refuse to say.

Most cases are a PLANTED COUNTEREXAMPLE -- a signal built to be the thing the
detector must not claim -- and those signals come from shrike's OWN transmitters,
because a hand-rolled one is a model of the mode rather than the mode. Written
by hand, an FSK burst alternating on every symbol reads 0.42 on
`_carrier_concurrency` where `pactor1.packet_signal` reads 0.0 and real off-air
PACTOR-1 reads 0.01-0.28; a test built on the first would have called the
carrier-pair gate broken and been wrong.

THE PACTOR-2 CASE IS OFF AIR, and that is the correction of 2026-09-02. It was
our own rendered frame, which reached the PACTOR-2 branch of `decode_events`
only because an unshaped render measured wider than `P1_MAX_BW_HZ` and so failed
the FSK width gate on its way past. Shaped, the same frame measures 326 Hz and
takes the `fsk` branch -- and so does HB9AK at 362 Hz, which is to say the branch
this file claimed to cover had never once fired on a real PACTOR-2 station. A
fixture that only our transmitter can satisfy is not a fixture; the recording is.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from hfmodem.shrike import p2rx, pactor1, pactor2, rxfront
from hfmodem.tests.kestrel import corpora

FS = rxfront.FS
#: HB9AK on 7.051 MHz, 12.1 s of a speed-level-3 keyboard QSO: nine data bursts
#: on the 1.25 s cycle, every one of them byte-exact through `p2rx` against an
#: independent decoder's fields (`rf-corpus/regress` grades that separately).
OFFAIR_P2 = corpora.REGRESS_FIXTURES / "pos_p2_sl3_hb9ak_055156.wav"


def _in_noise(sig: np.ndarray, secs: float = 2.0, snr: float = 0.6,
              seed: int = 20260803) -> np.ndarray:
    """A rendered burst dropped a quarter of the way into `secs` of band noise."""
    rng = np.random.default_rng(seed)
    n = int(secs * FS)
    x = rng.normal(0, 0.02, n)
    lo = n // 4
    m = min(len(sig), n - lo)
    x[lo:lo + m] += sig[:m] * snr / max(1e-9, float(np.abs(sig).max()))
    return x


def _pactor1(baud: int = 100) -> np.ndarray:
    return _in_noise(pactor1.packet_signal(b"HELLO WORLD", baud=baud, packet_count=1))


def _pactor2() -> np.ndarray:
    rng = np.random.default_rng(5)
    return _in_noise(pactor2.frame(rng.integers(0, 2, 400),
                                   rng.integers(0, 2, 400), k=0))


def _p2_carriers() -> np.ndarray:
    """The geometry and nothing else: two carriers keyed together, no frame
    marker, sharing the passband with a second station so the width gate lets it
    through. `pactor2.modulate` is `frame` without the header, so this is the
    same transmitter making the one signal the carrier-pair test can honestly
    claim."""
    rng = np.random.default_rng(7)
    x = _in_noise(pactor2.modulate(rng.integers(0, 2, 400),
                                   rng.integers(0, 2, 400)))
    return x + 0.05 * np.sin(2 * np.pi * 2200.0 * np.arange(len(x)) / FS)


def main() -> int:
    ok = True

    def check(name: str, cond: bool, note: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f": {note}" if note else ""))

    p1, p2 = _pactor1(), _pactor2()

    print("\nA lone carrier is not a two-tone burst")
    # The failure the both-tones gate exists for. One strong tone at 1400 Hz and
    # nothing at 1600 clears a threshold taken on the MEAN of the two tone
    # windows, because a mean is met by one of them at twice the figure. On a
    # night of passive listening most of the 162 accepted windows looked like this.
    t = np.arange(2 * FS) / FS
    lone = _in_noise(np.sin(2 * np.pi * 1400.0 * t[:FS // 2]))
    check("a 1400 Hz carrier alone raises no 1400/1600 burst",
          rxfront._p1_reply(lone) is None)

    print("\n...and shrike's own PACTOR-1 packet still does")
    got = rxfront._p1_reply(p1)
    check("a rendered PACTOR-1 packet is reported", got is not None)
    if got is not None:
        check("...on the WEAKER tone's excess, not the pair's mean",
              got.weaker_tone_db >= 20 * np.log10(rxfront.P1_REPLY_MIN_TONE_EXCESS),
              f"+{got.weaker_tone_db:.1f} dB")

    print("\nWhat that line is allowed to say")
    # A clean packet reads as `fsk` and never reaches this branch, which is right:
    # the more specific answer wins. Sharing the passband with a second station
    # puts the occupied width past `P1_MAX_BW_HZ`, `_fsk_present` declines, and
    # the burst lands here -- the working case, and the one the air produces.
    crowded = p1 + 0.1 * np.sin(2 * np.pi * 2200.0 * np.arange(len(p1)) / FS)
    evs = [e for e in rxfront.decode_events(crowded, 0.5) if e.kind == "p1reply"]
    check("the burst reaches decode_events", bool(evs))
    for e in evs[:1]:
        low = e.text.lower()
        check("the line claims no peer and no answer",
              "peer" not in low and "answer" not in low, e.text)
        check("...says the evidence is shape", "shape only" in low)
        check("...and carries the number it was measured on", "dB" in e.text)

    print("\nCarriers at the PACTOR-1 tones, keyed together")
    # The case the old gate could not report at all: it discarded any pair within
    # 25 Hz of 1400/1600 outright, which is exactly where an on-frequency PACTOR-2
    # station transmits. What keeps PACTOR-1 out is simultaneity, measured.
    got = rxfront._p2_present(rxfront._Spectrum(p2))
    check("a rendered PACTOR-2 frame is reported", got is not None)
    if got is not None:
        check("...as simultaneous, measured rather than assumed",
              got.concurrency >= rxfront.P2_MIN_CONCURRENCY, f"{got.concurrency:.2f}")
    check("...while a PACTOR-1 packet on the SAME tones is not",
          rxfront._p2_present(rxfront._Spectrum(p1)) is None)

    # ...at 100 Bd. AND NOT AT 200, which is the limit of what simultaneity can
    # buy: 5 ms a tone puts a transition inside most analysis windows, and the
    # same rendered packet comes back a pair at concurrency 0.48-0.50 against the
    # 0.35 gate. That is not a corner case -- it is a working Winlink gateway's
    # data phase. WS8EOC keyed one every cycle on 2026-08-18 and the live line
    # read 0.43, 0.47 and 0.50 (working/pactor-ws8eoc-40m-clamped-force.log),
    # while the captures behind those three readings decode CRC-valid as PACTOR-1
    # off disk. Nothing above the decode can separate the two, so the gate keeps
    # its threshold and the LINE gives up the protocol name, below.
    hi_speed = rxfront._p2_present(rxfront._Spectrum(_pactor1(200)))
    check("a 200 Bd PACTOR-1 packet clears the same pair gate",
          hi_speed is not None and hi_speed.concurrency >= rxfront.P2_MIN_CONCURRENCY,
          "" if hi_speed is None else f"{hi_speed.concurrency:.2f}")

    print("\nWhat the carrier-pair line is allowed to say")
    evs = [e for e in rxfront.decode_events(_p2_carriers(), 0.5)
           if e.kind == "detect"]
    check("the pair reaches decode_events", bool(evs))
    for e in evs[:1]:
        check("the line names the frequencies it found",
              "Hz" in e.text and "/" in e.text, e.text)
        check("...says the evidence is shape", "shape only" in e.text.lower())
        check("...and the TEXT names no protocol, because three of them read "
              "alike here", "no protocol named" in e.text.lower())
        check("...and the event names its own protocol, so no consumer guesses",
              e.protocol == "PACTOR-2", str(e.protocol))

    if not OFFAIR_P2.exists():
        print(f"\n  [SKIP] off-air PACTOR-2 absent at {OFFAIR_P2}")
        return 1 if not ok else 2

    print("\nA real PACTOR-2 station, and what it used to be called")
    offair = rxfront.load_wav(str(OFFAIR_P2))
    sp = rxfront._Spectrum(offair[int(0.7 * FS):int(2.7 * FS)])
    width = rxfront._occupied_bw(sp)
    # The finding this section exists for. PACTOR-1 is the NARROWER mode, so a
    # width gate written to keep FT8 and VARA out of the FSK branch cannot keep
    # PACTOR-2 out of it: HB9AK measures inside the limit and lights both FSK
    # tones, and every burst of this QSO was reported as PACTOR-1 shape.
    check("a real PACTOR-2 burst measures NARROWER than PACTOR-1's own limit",
          width <= rxfront.P1_MAX_BW_HZ,
          f"{width:.0f} Hz against {rxfront.P1_MAX_BW_HZ} Hz")
    check("...and lights both PACTOR-1 tones, so width cannot settle this",
          rxfront._fsk_present(sp))

    evs = list(rxfront.decode_events(offair, 0.5))
    kinds = Counter(e.kind for e in evs)
    det = [e for e in evs if e.kind == "detect"]
    check("every data burst of the recording reaches the PACTOR-2 branch",
          len(det) >= 9, f"{len(det)} lines for 9 bursts")
    # The half that is not a count. Before the marker took precedence this
    # recording came out as 20 `fsk` and 3 `detect`, none of the three on a
    # burst; the 40 s companion added a PACTOR-1 CONNECT naming a callsign read
    # out of 8-DPSK. A PACTOR-2 station may not be reported as PACTOR-1 at all.
    check("...and nothing in it is reported as PACTOR-1",
          not (kinds["fsk"] or kinds["connect"] or kinds["p1reply"]),
          str(dict(sorted(kinds.items()))))
    for e in det[:1]:
        check("the line names PACTOR-2, because a codeword is not a geometry",
              "PACTOR-2 frame marker" in e.text, e.text)
        check("...and carries the correlation it armed on",
              "correlation" in e.text and str(p2rx.MARKER_ARM) in e.text)
        check("...and says the field is read elsewhere, not here",
              "read by p2rx" in e.text)
        check("...and the event names its own protocol, so no consumer guesses",
              e.protocol == "PACTOR-2", str(e.protocol))

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip(f"off-air PACTOR-2 not present at {OFFAIR_P2}")
    assert rc == 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
