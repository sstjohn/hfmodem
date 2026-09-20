# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Whether a window that got louder was carrying PACTOR-III, or was band noise.

On 2026-09-11 this station keyed nine consecutive PACTOR-III entry packets at
WS8EOC on 80 m, the gateway granted the upgrade sixteen times, and two answer
slots printed

    ANSWER SLOT OCCUPIED -- ... across 2379 Hz ... centred 1258 Hz ...
    Somebody is transmitting where our answer is due in something that is not
    PACTOR-1

Read as a 2.4 kHz emission, that line sent an hour of receiver investigation
after a PACTOR-III reception which was never on the tape. The width was noise.
`occupied_bw` is the span of passband bins within 14 dB of the peak, and noise
fills a passband, so on the same session:

    the window the line was printed for           2346 and 2385 Hz
    a window nothing was ever claimed in          2396 Hz
    the morning's CRC-decoded PACTOR-III frame    1005 .. 1471 Hz

The real reception is the NARROWEST of the three. Width grades nothing, which
`occupied_bw` says in its own docstring and the line did not.

`rxfront.p3_comb_db` is what the line was missing. Channels 5 and 12 -- 1080 and
1920 Hz -- are in every PACTOR-III speed level's channel set, carry the variable
header and every control signal, and are the acquisition burst's own two bands,
so a window either stands both of them over the channel comb or it has no
PACTOR-III in it at any speed level. The control pair needs no second capture and
no rendered signal: `captures/onair-0911-2332` alternates our own keyed entry
packet with the listen window that followed it, every 1.25 s, eight times over,
on one tape and one sample clock.

Run: python -m hfmodem.tests.shrike.test_p3_comb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import rxfront

REPO = Path(__file__).resolve().parents[5]
CAPTURES = REPO / "captures"

#: The evening arm. Named rather than globbed: `captures/` is ignored and holds
#: whatever the last run wrote, so a glob measures the machine and this measures
#: the checkout. Log:
#: `working/onair-0911-2332/evening-E08-ws8eoc-80m-p3-cs6.log`.
EVENING = CAPTURES / "onair-0911-2332"

#: The morning arm at the SAME gateway, where the receiver decoded PACTOR-III
#: repeatedly -- `HOLD RX (P3 changeover acquired) ... b'RMS'`. The positive
#: control, and the one reception on file this station has read end to end.
MORNING = CAPTURES / "onair-0911-1022"

#: ...and the arm whose log reports `the window carried 1 burst(s) at a
#: turnaround of 127 ms that no reader took`. `rx_05` is the listen window that
#: burst falls in. It is a 1393.8/1594.5 Hz pair at a 200.7 Hz shift -- the
#: peer's PACTOR-1 tones, not a comb -- so nothing here may read it as PACTOR-III.
UNREAD = (CAPTURES / "onair-0911-2319", "rx_05")

#: Where the morning session reported `(CRC frame @ N)`, read out of its log.
#: Stream samples, and a short-cycle data field is 0.81 s, so the frame is the
#: 42000 samples from each.
MORNING_FRAMES = (588766, 708694, 828658, 948814,
                  1068598, 1428598, 1788598, 2148598)
FRAME_N = 42_000

#: The nine cycles of `TX[22]`..`TX[30]`, every one of them
#: `SL1 ENTRY 0B P3 status=0x1a` keyed at 0.843 s. What separates them from the
#: windows below is not the receiver and not the band: they are the same tape,
#: 1.25 s apart, through the same capture mute.
ENTRY_HOLDS = range(10, 19)

#: What the evening arm's answer slots are allowed to reach. Measured -0.8 to
#: +2.4 dB over all nine, the two the session flagged included; the margin is
#: to the knee and not to the measurement, so a drift toward it is visible
#: before it is a false positive.
EVENING_MAX_DB = 4.0

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _sidecar(d: Path, stem: str) -> dict:
    return json.loads((d / f"{stem}.json").read_text())


def _window(d: Path, stream: np.ndarray, stem: str) -> np.ndarray:
    """A listen window, cut out of the stream the session recorded it from.

    Out of the stream rather than off the window's own file because the entry
    packets between them have no file of their own, and a control pair measured
    two different ways is not a control pair.
    """
    j = _sidecar(d, stem)
    return stream[j["end_stream_sample"] - j["samples"]:j["end_stream_sample"]]


def _keyed(d: Path, stream: np.ndarray, n: int) -> np.ndarray:
    """Our own transmission: what lies between hold `n` and the next window."""
    a = _sidecar(d, f"hold_{n:02d}")["end_stream_sample"]
    j = _sidecar(d, f"hold_{n + 1:02d}")
    return stream[a:j["end_stream_sample"] - j["samples"]]


def _stems(d: Path) -> list[str]:
    return (sorted((f.stem for f in d.glob("rx_*.wav")), key=lambda s: int(s[3:]))
            + sorted((f.stem for f in d.glob("hold_*.wav")), key=lambda s: int(s[5:])))


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    if not (EVENING / "stream.wav").exists() or not (MORNING / "stream.wav").exists():
        print(f"  no {EVENING.name} / {MORNING.name} under {CAPTURES}")
        return 2

    evening = rxfront.load_wav(str(EVENING / "stream.wav"))

    keyed = [rxfront.p3_comb_db(_keyed(EVENING, evening, n))
             for n in ENTRY_HOLDS[:-1]]
    check("our own keyed PACTOR-III entry packets stand over the knee",
          min(keyed) >= rxfront.P3_COMB_KNEE_DB,
          f"{min(keyed):.1f} .. {max(keyed):.1f} dB over {len(keyed)} packets")

    slots = {n: rxfront.p3_comb_db(_window(EVENING, evening, f"hold_{n:02d}"))
             for n in ENTRY_HOLDS}
    worst = max(slots, key=slots.get)
    check("...and the answer slots between them do not",
          slots[worst] < EVENING_MAX_DB,
          f"worst hold_{worst:02d} at {slots[worst]:+.1f} dB, "
          f"knee {rxfront.P3_COMB_KNEE_DB:.1f}")

    every = [rxfront.p3_comb_db(_window(EVENING, evening, s))
             for s in _stems(EVENING)]
    check("no listen window of the whole evening arm reaches the knee",
          max(every) < rxfront.P3_COMB_KNEE_DB,
          f"{max(every):+.1f} dB worst of {len(every)} windows")

    morning = rxfront.load_wav(str(MORNING / "stream.wav"))
    frames = [rxfront.p3_comb_db(morning[c:c + FRAME_N]) for c in MORNING_FRAMES]
    check("every CRC-valid PACTOR-III frame of the morning arm clears the knee",
          min(frames) >= rxfront.P3_COMB_KNEE_DB,
          f"{min(frames):.1f} .. {max(frames):.1f} dB over {len(frames)} frames")

    d, stem = UNREAD
    if (d / f"{stem}.wav").exists():
        burst = rxfront.p3_comb_db(rxfront.load_wav(str(d / f"{stem}.wav")))
        check("the 127 ms burst no reader took is not read as PACTOR-III",
              burst < rxfront.P3_COMB_KNEE_DB, f"{burst:+.1f} dB")
    else:
        print(f"  [skip] {d.name}/{stem} absent")

    # The measurement this replaces, kept as the reason it was replaced: a real
    # reception is NARROWER than the noise the session called wideband.
    wide = max(rxfront.occupied_bw(_window(EVENING, evening, f"hold_{n:02d}"))
               for n in ENTRY_HOLDS)
    narrow = max(rxfront.occupied_bw(morning[c:c + FRAME_N]) for c in MORNING_FRAMES)
    check("width still grades nothing, which is why it is not the test",
          narrow < wide,
          f"decoded frames reach {narrow:.0f} Hz, empty slots {wide:.0f} Hz")

    return 0 if ok else 1


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip(f"no onair-0911-1022 / onair-0911-2332 under {CAPTURES}")
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
