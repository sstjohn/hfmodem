# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where the capture loses its blocks, on a bench with no sound card in it.

`test_shortfall` establishes the fact off 66 recorded sessions: 871216 samples
gone, every one of them in a listening cycle, none of the 1381 keyed cycles ever
short a sample. It cannot say WHY, because the recordings hold no timing. This
does, and it reproduces both halves -- a listening cycle that loses about one
percent of itself and a keyed cycle that loses nothing -- out of the readers, a
tick, and a thread.

THE DEVICE IS A PIPE AND AN IDLE PROCESS. A thread that sleeps to the block grid
and stamps its own wake measures the OS timer and the GIL together and cannot
separate them, so the grid is kept by a helper process -- idle, therefore crisp --
which writes one `time.monotonic()` per 2.67 ms block down a pipe. The thread
under test blocks in `os.read`, which holds no GIL; its lag is kernel delivery
plus GIL acquisition, and kernel delivery measures microseconds at idle. A block
still in the pipe when the converter's own ring has turned over is a block
CoreAudio overwrote before the callback could take it, and that is the loss.
`rates.lost_step` then reads it back off the `ahead` series the delivered blocks
leave behind -- the same arithmetic, on the same series, that the sidecars in
`captures/` were written from -- and the two agree.

WHAT THE READING TURNS ON. The ring is the only free parameter, and the answer is
not: unbreathed, the worst stall is 19-57 ms and the loss runs from 1.9-2.7% of
a listening cycle at the codec's own 3.42 ms of input latency, through 0.4-1.0%
at 11 ms, to 0.00-0.01% at 43 ms -- which is why an identical session can read
0.56% one day and 1.10% the next with nothing about the machine changed. The
2026-09-02 and 09-03 arms read 0.003% and 0.000% against 08-30's 1.05% on the
same cadence, the same rig and an untouched capture path, and their readers are
DEARER, not cheaper -- the blind PACTOR-3 scan measures 48.7 ms at the 08-30
revision and 77.2 ms today. Nothing was fixed between those days; the tail
landed inside the ring, and the exposure that put it past the ring on 08-30 is
still there and bigger. Breathed, the worst stall is one `gil.PERIOD` and the
loss is zero at every ring depth there is. That is the result worth having: not
a smaller number, an insensitive one.

AND WHY THE SESSION'S FIX HAS NOT REACHED THE AIR. `gil.breathing()` went into
`live.RollingRx._decode` on 2026-08-26 and is still the only one in the package.
The rolling decoder is not where a listening cycle spends itself: measured here,
per 1.25 s cycle, the rolling flush costs 40 ms and the readers `onair` calls
straight through to cost another 105 -- the blind PACTOR-3 scan alone is 67-89 --
and every one of those runs outside the breather. Attributing each lost block to
the stage that was running when it went puts 23552 of 23808 in the blind scan.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from hfmodem.core import gil, rates
from hfmodem.shrike import live, p3rx, pactor1, rxfront

FS = rates.CARD_RATE_HZ
BLOCK = 128
BLOCK_S = BLOCK / FS

#: What the device holds before a block is gone. `core.audio.play_drained`
#: records this station's codec at 3.42 ms of input latency on `latency="low"`,
#: and a stall past it overwrites.
RING_S = 0.00342

#: PACTOR's cycle, and the 0.24 s a keyed one hands the reader once our own
#: carrier has been flushed out of it.
CYCLE_S = 1.25
KEYED_S = 0.24

#: Enough cycles that a stall has to happen in most of them to pass the floor,
#: and few enough that three arms fit in a dozen seconds.
CYCLES = 12

_TICKER = f"""
import os, struct, time
due = time.monotonic() + {BLOCK_S!r}
while True:
    rest = due - time.monotonic()
    if rest > 0:
        time.sleep(rest)
    try:
        os.write(1, struct.pack("d", time.monotonic()))
    except (BrokenPipeError, OSError):
        break
    due += {BLOCK_S!r}
    if due < time.monotonic():
        due = time.monotonic() + {BLOCK_S!r}
"""


class _Capture:
    """A capture device whose callback is a Python callable, and its accounting."""

    def __init__(self, ring_s: float = RING_S) -> None:
        self.ring_s = ring_s
        self.adc: list[float] = []
        self.got: list[float] = []
        self.base = 0
        self._stop = threading.Event()
        self._proc = subprocess.Popen([sys.executable, "-c", _TICKER],
                                      stdout=subprocess.PIPE)
        self._thread = threading.Thread(target=self._callback, daemon=True)

    def __enter__(self) -> "_Capture":
        self._thread.start()
        time.sleep(0.3)
        self.base = len(self.adc)
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._proc.kill()
        self._proc.stdout.close()

    def _callback(self) -> None:
        out = self._proc.stdout
        while not self._stop.is_set():
            b = out.read(8)
            if len(b) < 8:
                break
            now = time.monotonic()
            self.adc.append(struct.unpack("d", b)[0])
            self.got.append(now)

    def lost(self) -> tuple[int, int, float]:
        """(samples the device overwrote, the same off `ahead`, worst stall)."""
        adc = np.array(self.adc[self.base:])
        lag = np.array(self.got[self.base:]) - adc
        kept = lag <= self.ring_s
        delivered = adc[kept]
        ahead = delivered - np.arange(delivered.size) * BLOCK_S
        step, prev = 0, None
        for a in ahead:
            step += rates.lost_step(float(a), prev, BLOCK, FS)
            prev = float(a)
        return int((~kept).sum()) * BLOCK, step, float(lag.max())


class _Null:
    def __enter__(self): return self
    def __exit__(self, *_exc): return False


def _cycle_audio(seed: int) -> np.ndarray:
    """One cycle of a link: a PACTOR-1 burst on an ordinary noise floor.

    Synthesised rather than read off a capture, because `captures/` is not in the
    checkout and the readers cost what they cost on either -- the blind PACTOR-3
    scan measures 67 ms here against 78-89 ms on `onair-0830-1215`, and the burst
    detector 17 against 19.
    """
    rng = np.random.default_rng(seed)
    burst = np.asarray(pactor1.packet_signal(b"CAPTURELOSS", 100,
                                             lead_s=0.02, tail_s=0.02), float)
    x = np.zeros(int(CYCLE_S * FS))
    x[:burst.size] = burst[:x.size]
    return (x + 0.02 * rng.standard_normal(x.size)).astype(np.float32)


class _Session:
    """`onair`'s receive path for one cycle, at the shape the field flies.

    A listening cycle bridges five 0.25 s slices, scans for the link's own frame
    before the key, and behind the carrier flushes, scans for the protocols the
    peer might be leading into, finds the peer's burst edges and reads the
    channel. A keyed cycle is the same tail over the 0.24 s our own carrier left.

    `breathe` is where the argument is. False is what flies: the flush is inside
    `RollingRx._decode`'s breather and every other reader is not.
    """

    def __init__(self, *, breathe: bool) -> None:
        self.breathe = breathe
        self.audio = _cycle_audio(1)
        self.short = self.audio[:int(KEYED_S * FS)]
        self.memory = p3rx.FieldMemory()
        self.sync = rxfront.SyncedRx()
        self.rx = live.RollingRx(lambda _ev: None, window_s=4.0, keep_s=3.75)
        step = int(0.25 * FS)
        for i in range(16):
            self.rx.push(self.audio[i * step:(i + 1) * step])
        self._full = self.rx.buf.copy()

    def _breath(self):
        return gil.breathing() if self.breathe else _Null()

    def listening(self) -> None:
        step = int(0.25 * FS)
        for i in range(5):
            slice_ = self.audio[i * step:(i + 1) * step]
            self.rx.hold(slice_)
            rxfront.p1_reply_starting(slice_)
        with self._breath():
            if self.sync.packet(self.audio) is None:
                rxfront.decode_expected_p1_packet(self.audio, None)
        self._behind_the_carrier(self.audio)

    def keyed(self) -> None:
        self.rx.hold(self.short)
        rxfront.p1_reply_starting(self.short)
        self._behind_the_carrier(self.short)

    def _behind_the_carrier(self, whole: np.ndarray) -> None:
        self.rx.buf = self._full[-int(0.75 * FS):].copy()
        self.rx.pending = 0
        self.rx.flush()
        with self._breath():
            rxfront.decode_expected_packet(whole, self.memory)
            rxfront.p1_burst_onsets(whole)
            rxfront.cs_evidence(whole)


def _arm(*, breathe: bool, keyed: bool) -> tuple[int, int, float, float]:
    """(lost, the same off `ahead`, worst stall, seconds of decode)."""
    session = _Session(breathe=breathe)
    run = session.keyed if keyed else session.listening
    run()
    with _Capture() as device:
        t0 = time.monotonic()
        for _ in range(CYCLES):
            run()
        busy = time.monotonic() - t0
    lost, step, worst = device.lost()
    return lost, step, worst, busy


def _per_cycle(lost: int) -> float:
    """Samples lost per cycle, which is the currency the record keeps.

    Not a fraction of the audio: the arms here run their cycles back to back
    with none of the 1.1 s a real cycle spends waiting on the card, so a rate
    against the audio would flatter whichever arm carries less of it. The air's
    listening cycles lose 400-1400 samples apiece and its keyed cycles lose
    none, and that is a count.
    """
    return lost / CYCLES


@pytest.mark.realtime
def test_a_listening_cycle_loses_capture_and_a_keyed_cycle_does_not():
    """The record's own shape, off the readers rather than off the recordings.

    The floor is 150 samples a cycle against the 1300-2300 this measures and the
    400-1400 the air lost, so what fails here is the mechanism going away, not
    the machine having a good afternoon. The keyed arm runs the same tail over
    0.24 s instead of 1.25 and is asserted only against the listening one: the
    claim is the phase, and a bench that put them equal would have refuted it.
    """
    heard, heard_step, worst, busy = _arm(breathe=False, keyed=False)
    keyed, _, _, _ = _arm(breathe=False, keyed=True)

    assert _per_cycle(heard) > 150, (
        f"a listening cycle costing {1e3 * busy / CYCLES:.0f} ms lost "
        f"{_per_cycle(heard):.0f} samples, and the air's listening cycles lose "
        "400-1400 apiece")
    assert heard_step == pytest.approx(heard, rel=0.05), (
        f"the loss the device took ({heard}) and the loss `rates.lost_step` "
        f"reads off the delivered stream ({heard_step}) are the same samples")
    assert worst > 4 * gil.PERIOD, (
        f"the worst stall was {1e3 * worst:.1f} ms, which no capture would "
        "notice -- the load is no longer holding the interpreter")
    assert _per_cycle(keyed) < _per_cycle(heard) / 4, (
        f"a keyed cycle lost {_per_cycle(keyed):.0f} samples against the "
        f"listening cycle's {_per_cycle(heard):.0f}, and the record has 1381 "
        "keyed cycles that never lost one")


@pytest.mark.realtime
def test_breathing_the_whole_receive_path_takes_the_loss_to_zero():
    """Not less loss -- none, and a stall bounded by the breather's own period.

    Breathing only `RollingRx._decode`, which is what flies, is the arm above:
    the flush is inside it and the readers `onair` calls straight through are not,
    and the loss is unmoved. What erases it is the breather held over the whole
    cycle.
    """
    lost, step, worst, busy = _arm(breathe=True, keyed=False)
    reference, _, _, plain = _arm(breathe=False, keyed=False)

    assert worst < 5 * gil.PERIOD, (
        f"breathing left the capture out for {1e3 * worst:.1f} ms against a "
        f"{1e3 * gil.PERIOD:.0f} ms period")
    assert lost == 0 and step == 0, (
        f"{lost} samples still went while the whole path was breathing "
        f"({step} off the delivered stream)")
    assert reference > 0, "the unbreathed arm lost nothing, so this proves nothing"
    assert busy < 1.5 * plain, (
        f"breathing cost {100 * (busy / plain - 1):.0f}% of the decode; a "
        "listening cycle has 1.25 s and the readers want 150 ms of it")


@pytest.mark.realtime
def test_the_reader_in_front_of_the_key_still_fits_its_reserve():
    """The pre-key scan is the one decode a cycle cannot be late out of.

    `onair` gives it the holdback plus `PREKEY_RESERVE_S` -- 43 ms measured on
    this station -- and a cycle that overruns hands its slot back to the grid.
    Breathing costs the scan about a seventh, which is 3 ms of that budget and
    not the whole of it, so the reader is timed here rather than argued about.
    """
    audio = _cycle_audio(2)
    rxfront.decode_expected_p1_packet(audio, None)

    def cost(breathe: bool) -> float:
        runs = []
        for _ in range(5):
            t = time.perf_counter()
            with (gil.breathing() if breathe else _Null()):
                rxfront.decode_expected_p1_packet(audio, None)
            runs.append(time.perf_counter() - t)
        return float(np.median(runs))

    plain, breathed = cost(False), cost(True)
    assert breathed < 0.043, (
        f"the pre-key scan takes {1e3 * breathed:.1f} ms breathing, against the "
        f"43 ms it has in front of the key ({1e3 * plain:.1f} ms without)")
