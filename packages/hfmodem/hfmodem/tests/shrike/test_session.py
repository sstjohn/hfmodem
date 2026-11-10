# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""How late does an event reach the ARQ state machine?

This is the number that decides whether shrike can hold a multi-cycle QSO. A
connect can be answered late and still work -- a gateway retries its reply until
it gives up -- but every turnaround inside a session carries its own answer that
is sent once, so an event delivered after the cycle it belongs to is an event
missed, and the link stalls.

The quantity is delay from the END of a peer's burst to the decoder emitting its
event, because that is when the turnaround clock starts: the peer transmits for
most of the cycle and leaves ~0.29 s of the 1.25 s short cycle for us to decode
and key up. Measuring from the burst's START instead charges the decoder for the
burst's own duration and makes a 3.75 s data cycle look 3.75 s late when it is
not late at all -- worth stating because that is exactly the error this file was
first written with.

So a burst of known length is spliced into silence at a known offset and the
delivery time is compared against its end. Nothing is inferred from a recording's
unknown timing.

Run: python -m hfmodem.tests.shrike.test_session
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import live, onair, rxfront  # noqa: E402
from hfmodem.tests.shrike import archive  # noqa: E402

FS = rxfront.FS
# The short cycle is 1.25 s, of which the peer's transmission takes ~0.96 s. What
# is left is the whole budget for hearing the end of it, decoding, and keying.
TURNAROUND_S = 0.29
BURSTS = [
    ("P1 connect", "captures/connect_W1AW.wav"),
    ("real off-air P3", "captures/offair_KE5YTA_p3.wav"),
]


def deliver_delay(burst: np.ndarray, at_s: float, window_s: float,
                  keep_s: float) -> float | None:
    """Splice `burst` into silence starting at `at_s`; delay after it ends."""
    pad = np.zeros(int(at_s * FS), np.float32)
    tail = np.zeros(int(6.0 * FS), np.float32)
    audio = np.concatenate([pad, burst, tail])
    end_s = at_s + len(burst) / FS

    first: list[float] = []
    fed = 0.0

    def on(ev):
        # Only the events that CARRY something count. A bare presence detect
        # fires on the first fragment of a burst and would flatter the number
        # without telling the state machine anything it can answer.
        if ev.kind in ("detect", "fsk"):
            return
        if not first:
            first.append(fed)

    rx = live.RollingRx(on, window_s=window_s, keep_s=keep_s)
    step = int(0.05 * FS)
    for i in range(0, len(audio), step):
        rx.push(audio[i:i + step])
        fed = min(i + step, len(audio)) / FS
        if first:
            break
    return first[0] - end_s if first else None


class _StalledInput:
    """A capture stream that opens, delivers two blocks and then dies.

    Exactly what the C-Media dongle did when the input stream held a non-default
    buffer size and something else opened the same device for output: PortAudio
    stopped calling back, `samples` froze, and every read returned nothing. The
    session could not tell that from a quiet band -- it kept computing perfectly
    self-consistent boundaries and kept transmitting, eleven times, into a
    receiver that was not listening.
    """

    def __init__(self):
        self.samples = self.pos = 960
        self.xruns = self.holdback = 0

    def read(self, count):
        return np.zeros(0, np.float32)

    def wait_until(self, index):
        return 0.0

    def sample_now(self):
        return float(self.samples)


def deaf_receiver_stops_the_session() -> bool:
    """A window that captures nothing from a frozen stream must abort, loudly."""
    print("\ncapture stream stalls mid-session")
    dead = _StalledInput()
    try:
        onair._assert_capturing(np.zeros(0, np.float32), dead, dead.samples)
        print("  [FAIL] a dead capture stream was accepted as a quiet band")
        return False
    except SystemExit as e:
        print(f"  [PASS] session stops: {str(e).splitlines()[0][:64]}...")
    # ...and a genuinely quiet band, where samples ARE still arriving, must not
    # trip it: the guard is about a dead stream, not about silence.
    dead.samples += 48000
    try:
        onair._assert_capturing(np.zeros(0, np.float32), dead, dead.samples - 48000)
        print("  [PASS] a quiet band with a live stream keeps running")
        return True
    except SystemExit:
        print("  [FAIL] a quiet band was mistaken for a dead stream")
        return False


def grid_holds() -> bool:
    """The replay stream must hold the session's read position to the grid.

    The on-air failure looked like grid arithmetic and was not: after an early
    break truncated a window, the read position stayed where the window ended
    while the boundaries marched on, and the difference walked -1210, -3710,
    -6210, -8710 ms. The walk itself belongs to `onair`'s session loop -- its
    on-air measurement and reproduction live in `test_grid` -- and this used to
    restate the loop's boundary arithmetic around `read` and assert an equality
    `read`'s own return guarantees, which held whatever the loop did. What this
    file CAN honestly assert is the stream contract that loop stands on, driven
    through the production `_ReplayInput`:

      * `wait_until(boundary)` lands the read position exactly ON the boundary
        however short the read before it was cut -- `wait_until` doing nothing
        is precisely how the walk began (-1.1 s at the first early break on
        captures/witness.wav, per its own docstring);
      * the samples the wait skips over come back through `read_ready`, held
        rather than dropped -- they are most of the cycle, and the part a
        link-setup answer arrives in;
      * `clamp_late` reads every boundary not yet reached as placeable.
    """
    print("\nreplay stream against the cycle grid")
    fs, cycle, settle = onair.FS, 1.25, 0.10
    slot_n, settle_n = round(cycle * fs), round(settle * fs)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "quiet.wav"
        onair.session.write_wav(str(wav), np.zeros(int(30 * fs), np.float32))
        src = onair._ReplayInput(str(wav))
        anchor, seen = src.pos, 0
        for slot in range(1, 21):
            boundary = anchor + slot * slot_n
            if src.clamp_late(boundary - settle_n):
                print(f"  [FAIL] slot {slot}: boundary {boundary} reads as "
                      f"already late at pos {src.pos} -- the grid has run away")
                return False
            # An early break: the window is cut 0.43 s in, most of the slot
            # unread -- the exact shape the on-air walk started from.
            seen += src.read(round(0.43 * fs)).size
            src.wait_until(boundary - settle_n)
            if src.pos != boundary - settle_n:
                print(f"  [FAIL] slot {slot}: wait_until left pos {src.pos}, "
                      f"the grid wanted {boundary - settle_n}")
                return False
            seen += src.read_ready().size
            if seen != src.pos - anchor:
                print(f"  [FAIL] slot {slot}: {src.pos - anchor - seen} samples "
                      f"skipped without being handed back")
                return False
    print(f"  [PASS] 20 slots: every boundary reached exactly after an early "
          f"break, every skipped sample handed back ({seen / fs:.2f} s in all)")
    return True


def main() -> int:
    root = archive.ARCHIVE
    shapes = [
        ("band monitor (1.50 s slide)", live.WINDOW_S, live.KEEP_S),
        ("ARQ session  (0.25 s slide)",
         onair.RX_WINDOW_S, onair.RX_WINDOW_S - onair.RX_SLIDE_S),
    ]
    ok, tested = True, 0
    for label, fx in BURSTS:
        path = root / fx
        if not path.exists():
            print(f"  SKIP {label} (absent)")
            continue
        # Trim to the signal so the "burst end" is the signal's end, not a
        # recording's trailing silence.
        burst = onair._trim_silence(rxfront.load_wav(str(path)))
        if burst.size > int(3.75 * FS):
            burst = burst[:int(3.75 * FS)]          # one long data cycle at most
        print(f"\n{label}  ({burst.size / FS:.2f} s burst, from {fx})")
        for shape, win, keep in shapes:
            got = [deliver_delay(burst, at, win, keep) for at in (0.4, 1.1, 2.3)]
            got = [g for g in got if g is not None]
            if not got:
                print(f"  {shape}: never decoded")
                continue
            worst = max(got)
            verdict = "in budget" if worst <= TURNAROUND_S else "TOO LATE"
            print(f"  {shape}: decoded {min(got):+.2f} to {worst:+.2f} s "
                  f"relative to the burst end (negative = before it ends) "
                  f"-- {verdict}")
            if "ARQ" in shape:
                tested += 1
                if worst > TURNAROUND_S:
                    ok = False

    ok &= grid_holds()
    ok &= deaf_receiver_stops_the_session()

    print()
    if not ok:
        print("FAIL -- a burst was delivered too late to answer inside its cycle")
        return 1
    if not tested:
        print("  [SKIP] no burst fixture present -- nothing was timed")
        return 2
    print(f"ALL PASS -- every burst delivered within {TURNAROUND_S:.2f} s "
          f"of ending, the PACTOR turnaround budget")
    return 0


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip("no burst fixture present -- "
                    + ", ".join(fx for _, fx in BURSTS) + " absent")
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(main())
