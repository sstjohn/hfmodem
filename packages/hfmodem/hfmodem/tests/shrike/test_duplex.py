# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate the duplex transmit path -- audio AND keying -- on one timebase.

The transmit timing is the one part of shrike that could only ever be measured
on the air, and the numbers that came back said the key was up 143.7 ms longer
than the audio: 40.3 ms of stream startup before the first sample and 103.4 ms
of CoreAudio stream teardown after the last, per burst. Against a 1.25 s PACTOR
cycle -- of which a commercial modem keys 961.4 ms and leaves 288.5 ms of clear
air -- that left us 86 ms, so the far end's 120 ms answer could not fit in the
gap even in principle, and sometimes we keyed onto its tail.

`_LiveInput` now owns one duplex stream and schedules the burst on the output
frame counter, so this can be checked without a radio: a loopback device returns
our own emission on the capture stream, and every claim the transmit path makes
about it -- which capture sample the carrier came up on, which one it dropped
on -- is checked against where the audio actually landed.

PTT USED TO BE A STUB HERE, and that is why this file passed while the rig did
something else entirely. On 2026-07-28 the operator watched the key stay down
across five bursts, then hold a dead carrier for seconds, then flash five times
in quick succession -- while every cycle of the log read `PTT keyed 0.998 s for
0.960 s of audio`. Nothing in the tree could have caught that: `keyed_s` is the
interval between two writes to a pipe, and a stub rig records the same two
writes. So the keying is now measured where a rig would act on it. `rigctl` is
given a pseudo-terminal in place of /dev/cu.usbserial-*, with the model number,
baud and arguments the on-air path uses, and the FT-891 backend's `TX1;`/`TX0;`
are timestamped as they reach the wire (tests/shrike/ptyrig.py). Same clock as the
audio, so "was the rig keyed before the first sample, and for how long" is a
subtraction rather than an assumption.

Three stages, because each fault only shows in one of them:

  * `rigctl` with both its pipes full, which is the unkey asked for nothing but
    to still happen. No audio and no radio in it -- it is about whether a
    command can leave this program at all.
  * a stand-in that answers CAT promptly, which is the transmit path working;
  * a stand-in that is slow to answer `TX;` after being told to transmit, which
    is a rig with something better to do while its relays close. Keying is on
    RTS by then, so the audio and the key stay together while the CAT wire is
    left waiting -- the fault this stage was written for, from the other side.

What is still NOT checked is the radio: a pty is not a T/R relay and it is not a
codec that shares a clock with a transmitter. This says the software's keying is
where it claims to be, and what becomes of it when the far end is slow.

Needs a loopback device -- one of the BlackHoles, whichever of them is bit-exact
today -- the local Hamlib build, and a compiler for the pty's modem lines
(tests/ptyrts.c). With any of them missing there is no verdict and it says so.

Run:  python -m hfmodem.tests.shrike.test_duplex
"""
from __future__ import annotations

import array
import fcntl
import os
import sys
import termios
import threading

import pytest
import time
from pathlib import Path

import numpy as np

from hfmodem.shrike import onair, ota, spec
from hfmodem.tests.shrike import ptyrig

FS = spec.SAMPLE_RATE
CYCLE_S = spec.CYCLE_SHORT_S            # 1.25 s, the PACTOR short cycle
BURST_N = round(spec.P1_PACKET_S * FS)  # 46080 -- a PACTOR-1 data packet
SETTLE = ota.RIGS["ft891"]["settle"]
CYCLES = 22
STALL_CYCLES = 5                        # what the operator counted before the flashes
# How long the stand-in withholds its `TX;` answer in the second stage. Longer
# than the 1.25 s cycle, so a rig that is slow for a single turnaround is
# already more than the transmit path can absorb.
STALL_S = 2.5
# What a keyed window may cost above the audio it carries. The budget is the
# cycle minus the peer's control signal and both turnarounds; 10 ms over the
# 0.960 + settle the transmission itself needs is the whole allowance.
KEY_BUDGET_S = SETTLE + BURST_N / FS + 0.010
LOOPBACK = "blackhole"
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _loopback() -> int | None:
    """A device that returns our own samples unaltered, PROVED rather than named.

    Which BlackHole is bit-exact is a property of this machine's audio settings
    and not of the driver: 16ch was the one for months and on 2026-07-29 came
    back at 0.648 of unity, constant over every sample, because its device volume
    had moved. A gain is indistinguishable here from the transmit path mangling
    the burst, and picking the device by name would have charged that to shrike.
    So each candidate is asked to loop a short burst back and the first one that
    returns it exactly is used.
    """
    import sounddevice as sd
    probe = _burst(round(0.05 * FS), seed=3)
    for i, d in enumerate(sd.query_devices()):
        if not (LOOPBACK in d["name"].lower() and d["max_input_channels"] > 0
                and d["max_output_channels"] > 0):
            continue
        live = onair._LiveInput(i, i)
        try:
            time.sleep(0.3)
            first, end = live.transmit(probe, settle=0.0)
            seg = live.take_until(end + round(0.01 * FS))
            got = seg[first - (live.pos - seg.size):][:probe.size]
            err = (float(np.abs(got - probe).max()) if got.size == probe.size
                   else 1.0)
        finally:
            live.close()
        print(f"  loopback candidate {i} {d['name']!r}: worst sample error "
              f"{err:.3e}")
        if err < 1e-6:
            return i
    return None


def _burst(n: int, seed: int) -> np.ndarray:
    """Random signs at a constant magnitude.

    Not a tone: alignment against a periodic signal is ambiguous by a period, and
    the whole measurement here is an alignment. Constant magnitude so the edges
    are the burst's edges -- an envelope has nothing to decide -- and so
    `RadioTx._tx`'s silence trim cannot take a sample off either end.
    """
    rng = np.random.default_rng(seed)
    return (0.5 * rng.choice([-1.0, 1.0], n)).astype(np.float32)


def _onset(seg: np.ndarray) -> int:
    """First sample of the burst in `seg`. Exact: the loopback is sample-exact
    and the burst has a hard edge, so this is a threshold and not an estimator."""
    on = np.flatnonzero(np.abs(seg) > 0.25 * np.abs(seg).max())
    return int(on[0]) if on.size else -1


def _align(seg: np.ndarray, at: int, burst: np.ndarray) -> tuple[int, float]:
    """(lag, worst sample error) of `burst` inside `seg` near index `at`.

    The lag is searched on a prefix and the error is then taken over the whole
    burst, so a sample dropped in the middle of a transmission shows up as a
    residual rather than being averaged away.
    """
    span = 256
    lags = range(-96, 97)
    head = burst[:span]
    lag = min(lags, key=lambda k: np.abs(seg[at + k:at + k + span] - head).max())
    got = seg[at + lag:at + lag + burst.size]
    return lag, float(np.abs(got - burst).max()) if got.size == burst.size else 1.0


class WireRig:
    """`ota.Rig` driving the real FT-891 backend, with the wire watched.

    The whole on-air PTT stack except the radio: the same `rigctl` held open
    across the session, the same one line per change down its stdin, the same
    backend with its 50 ms post-write delay and its read-back of every PTT
    change. Only the far end is a pty.

    `ptt_type` picks which keying is under test. "RIG" is a CAT command, which is
    how the Xiegus key; "RTS" is the FT-891's, a line on a SECOND port -- so the
    stand-in grows a second pty, exactly as the rig presents a second port, and
    the keying is watched there instead. `pty` stays the CAT wire either way.
    """

    def __init__(self, *, answer_delay: float = 0.0, ptt_type: str = "RIG"):
        self.pty = ptyrig.PtyRig(answer_delay=answer_delay)
        self.ptt_pty = (ptyrig.PtyRig(ptt_line="RTS") if ptt_type != "RIG"
                        else self.pty)
        self.rig = ota.Rig(ptyrig.FT891_MODEL, self.pty.path, ptyrig.FT891_BAUD,
                           ptt_type=ptt_type, ptt_port=self.ptt_pty.path,
                           rigctl_dir=os.path.dirname(ptyrig.RIGCTL))
        self.rig._open()
        self.issued: list[tuple[float, bool]] = []
        time.sleep(2.0)               # rig_open's own CAT exchange, out of the way

    def ptt(self, on: bool) -> bool:
        """What the transmit path asked for, kept beside what the wire got."""
        self.issued.append((time.monotonic(), bool(on)))
        return self.rig.ptt(on)

    def edges(self) -> list[tuple[float, bool]]:
        return list(self.ptt_pty.edges)

    def alive(self) -> bool:
        return self.rig._proc is not None and self.rig._proc.poll() is None

    def close(self) -> None:
        try:
            self.rig.stop()
        except Exception:
            pass
        if self.rig._proc is not None:
            self.rig._proc.kill()
        for p in {id(self.pty): self.pty, id(self.ptt_pty): self.ptt_pty}.values():
            p.close()


def _windows(edges: list[tuple[float, bool]]) -> list[tuple[float, float]]:
    """(down, up) for each keyed span at the wire.

    A release with no assertion before it is dropped -- the transmit path unkeys
    twice on purpose -- and a span still open at the end is reported against
    +inf, because a rig left transmitting is exactly what this is looking for.
    """
    out, down = [], None
    for t, on in edges:
        if on and down is None:
            down = t
        elif not on and down is not None:
            out.append((down, t))
            down = None
    if down is not None:
        out.append((down, float("inf")))
    return out


def _keying(tag: str, edges: list[tuple[float, bool]], keyed_at: list[float],
            cycles: int, audio_s: float, issued: int = 0,
            budget: bool = True) -> None:
    """The keying pattern at the wire, against the audio it was meant to carry.

    `keyed_at` is the system-clock instant each burst's first sample reached the
    DAC, so key-down to `keyed_at` is the settle the rig actually got.
    """
    win = _windows(edges)
    if issued:
        check(f"{tag}: every PTT change shrike asked for reached the wire",
              len(edges) >= issued, f"{len(edges)} of {issued} arrived")
    check(f"{tag}: PTT asserted once per burst", len(win) == cycles,
          f"{len(win)} keyed windows for {cycles} bursts")
    spans = np.array([d - u for u, d in win]) if win else np.zeros(0)
    check(f"{tag}: every assertion was released",
          bool(spans.size) and bool(np.all(np.isfinite(spans))),
          f"{int(np.sum(~np.isfinite(spans)))} still keyed at the end")
    spans = spans[np.isfinite(spans)]
    if not spans.size:
        return
    if budget:
        check(f"{tag}: keyed window under {KEY_BUDGET_S:.3f} s",
              bool(spans.max() < KEY_BUDGET_S),
              f"{spans.mean():.4f} s mean, {spans.max():.4f} s worst "
              f"(audio {audio_s:.3f} + settle {SETTLE:.3f})")
        check(f"{tag}: keyed window jitter under 5 ms",
              float(spans.max() - spans.min()) < 0.005,
              f"{(spans.max() - spans.min()) * 1e3:.2f} ms spread")
    # No keyed window may carry more than the burst it was opened for. A
    # backlogged pipe shows up here first: the key goes down for one cycle and
    # does not come up again until several bursts have gone out under it.
    held = max(sum(1 for t in keyed_at if u <= t <= d) for u, d in win)
    check(f"{tag}: no keyed window carries more than its own burst", held <= 1,
          f"one window carried {held} bursts")
    n = min(len(win), len(keyed_at))
    settled = np.array([keyed_at[i] - win[i][0] for i in range(n)])
    check(f"{tag}: the rig was keyed {SETTLE * 1e3:.0f} ms before the first sample",
          bool(np.all(np.abs(settled - SETTLE) < 0.010)),
          f"{settled.min() * 1e3:+.1f} to {settled.max() * 1e3:+.1f} ms, "
          f"worst error {np.abs(settled - SETTLE).max() * 1e3:.1f} ms")


def _stage_one(dev: int) -> None:
    """The transmit path against a stand-in that answers CAT promptly.

    Keyed over CAT, which is how the Xiegus key and what `--ptt-type RIG`
    selects. Nothing is wrong with that path against a rig that answers, and it
    is the one stage that still measures it end to end.
    """
    import sounddevice as sd
    print(f"\n== responsive rig, keying over CAT ==\nloopback: device {dev} "
          f"{sd.query_devices(dev)['name']!r}, {CYCLES} cycles of {CYCLE_S:.3f} s, "
          f"burst {BURST_N} samples ({BURST_N / FS:.3f} s), settle {SETTLE:.3f} s")
    burst = _burst(BURST_N, seed=7)
    wire = WireRig()
    live = onair._LiveInput(dev, dev)
    slot_n, settle_n = round(CYCLE_S * FS), round(SETTLE * FS)
    starts, ends, lags, errs, keyed_at = [], [], [], [], []
    try:
        time.sleep(0.4)                   # let the stream settle before it matters
        anchor = int(live.sample_now()) + FS // 2
        for c in range(CYCLES):
            boundary = anchor + c * slot_n
            live.take_until(boundary - settle_n - live.holdback)
            live.wait_until(boundary - settle_n)
            first, end = live.transmit(burst, at=boundary, settle=SETTLE,
                                       key=wire.ptt)
            keyed_at.append(live._dac_time(first - live._lat))
            # Read our own emission back, a little past the reported end so the
            # tail is in hand too.
            seg = live.take_until(end + round(0.03 * FS))
            seg_start = live.pos - seg.size
            lag, err = _align(seg, first - seg_start, burst)
            starts.append(first + lag)
            ends.append(first + lag + burst.size)
            lags.append(lag)
            errs.append(err)
            print(f"  cycle {c + 1:2d}: reported {first}..{end}, measured "
                  f"{first + lag}..{first + lag + burst.size} "
                  f"({lag / FS * 1e3:+.2f} ms), residual {err:.2e}",
                  flush=True)
        # ...and one burst through the caller that keys it in a session, so what
        # is checked is the path the modem takes and not just the stream under
        # it: `RadioTx._tx` renders, trims, normalises, transmits, and then
        # discards the capture its own carrier covered. The marker goes straight
        # through the stream so it survives that discard -- if the discard left
        # the read position anywhere but the sample it claims, the marker would
        # turn up somewhere else in the window.
        tx = onair.RadioTx(wire.rig, transmit=True, out_dev=None, outdir=Path("."),
                           settle=SETTLE)
        tx.live, tx.boundary = live, anchor + CYCLES * slot_n
        live.take_until(tx.boundary - settle_n - live.holdback)
        tx._tx(_burst(round(0.1 * FS), seed=11), "bench")
        tx_err = (tx.tx_audio_start - tx.boundary) / FS
        pos_kept = live.pos == tx.tx_end
        marker = _burst(round(0.05 * FS), seed=13)
        f2, end2 = live.transmit(marker, at=tx.tx_end + round(0.2 * FS),
                                 settle=SETTLE, key=wire.ptt)
        seg = live.take_until(end2 + round(0.03 * FS))
        mark_err = (live.pos - seg.size + _onset(seg) - f2) / FS
        lost, underruns = onair._loss_seen(live), live.underruns
    finally:
        live.close()
        time.sleep(0.5)                   # let the last unkey reach the wire
        edges = wire.edges()
        wire.close()

    starts, ends = np.array(starts), np.array(ends)
    lags, errs = np.array(lags), np.array(errs)

    print()
    check("no capture loss", lost is None, lost or "nothing seen to go missing")
    check("no output underruns", underruns == 0, f"{underruns}")
    check("every burst looped back sample-exactly",
          float(errs.max()) < 1e-6,
          f"worst residual {errs.max():.2e} over {CYCLES} x {BURST_N} samples")
    check("captured burst length is the transmitted length",
          bool(np.all(ends - starts == BURST_N)),
          f"{sorted(set(ends - starts))} samples")
    check("reported carrier start is where the burst is",
          float(np.abs(lags).max()) / FS < 0.002,
          f"worst {np.abs(lags).max() / FS * 1e3:.2f} ms, "
          f"mean {lags.mean() / FS * 1e3:+.3f}")
    check("reported carrier end is where the burst ends",
          float(np.abs(lags).max()) / FS < 0.002,
          f"same alignment, worst {np.abs(lags).max() / FS * 1e3:.2f} ms")
    check("the session's own caller puts its carrier on the slot boundary",
          abs(tx_err) < 0.002, f"{tx_err * 1e3:+.2f} ms")
    check("the discard leaves the read position on the sample it claims",
          pos_kept and abs(mark_err) < 0.002, f"{mark_err * 1e3:+.2f} ms")

    resid = (starts - starts[0]) - np.arange(CYCLES) * slot_n
    rms = float(np.sqrt(np.mean(resid.astype(float) ** 2))) / FS
    check(f"the grid holds over {CYCLES} consecutive {CYCLE_S:.2f} s cycles",
          rms < 0.002, f"{rms * 1e3:.3f} ms rms, worst "
                       f"{np.abs(resid).max() / FS * 1e3:.3f} ms")

    # The two bursts after the loop key as well; the keying assertions are about
    # the cycles that were measured against the grid.
    _keying("wire", edges[:2 * CYCLES], keyed_at, CYCLES, BURST_N / FS,
            issued=2 * CYCLES)
    spans = [d - u for u, d in _windows(edges[:2 * CYCLES])]
    if spans:
        print(f"\n  cycle budget: keyed {np.mean(spans):.4f} s of {CYCLE_S:.3f} "
              f"leaves {(CYCLE_S - np.mean(spans)) * 1e3:.1f} ms of clear air "
              f"(a commercial modem leaves 288.5 +/- 2.9)")


def _stage_two(dev: int) -> None:
    """The same path against a rig that is slow to answer, keying on RTS.

    A CAT PTT change is validated by the backend writing `TX;` and blocking on
    the answer, and on an empty read `newcat_set_cmd_validate` jumps back ABOVE
    its own retry counter -- so a rig that does not answer holds `rigctl`'s
    command loop, and with it every PTT change still sitting in the pipe. That is
    what this stage was written to catch, and it caught it: five bursts went out
    under one keyed window, the sixth key-down landing 10.8 s late.

    Keying no longer goes that way. It is RTS on the rig's second port -- one
    ioctl, dispatched in hamlib's frontend, which never reaches the backend that
    can spin. So the rig here is as slow as it ever was, and the assertions below
    are the same ones; what has changed is that nothing the CAT wire does can
    reach the key. The proof of that is `commands`: not one `TX1;` or `TX0;`.
    """
    print(f"\n== rig {STALL_S:.1f} s slow to answer TX;, keying on RTS ==\n"
          f"{STALL_CYCLES} cycles, with the transmit path's own account of its "
          f"keying printed beside what reached the wire")
    burst = _burst(BURST_N, seed=7)
    wire = WireRig(answer_delay=STALL_S, ptt_type="RTS")
    live = onair._LiveInput(dev, dev)
    slot_n, settle_n = round(CYCLE_S * FS), round(SETTLE * FS)
    keyed_at = []
    try:
        time.sleep(0.4)
        anchor = int(live.sample_now()) + FS // 2
        for c in range(STALL_CYCLES):
            boundary = anchor + c * slot_n
            live.take_until(boundary - settle_n - live.holdback)
            live.wait_until(boundary - settle_n)
            first, end = live.transmit(burst, at=boundary, settle=SETTLE,
                                       key=wire.ptt)
            keyed_at.append(live._dac_time(first - live._lat))
            live.flush_to(end)
            print(f"  cycle {c + 1:2d}: the log says PTT keyed "
                  f"{live.keyed_s:.3f} s for {BURST_N / FS:.3f} s of audio",
                  flush=True)
    finally:
        live.close()
        # Long enough for a backlogged pipe to give itself away: the old fault
        # kept delivering PTT changes for seconds after the audio stopped, at the
        # backend's own pace, which is the run of rapid toggles the operator
        # counted. Anything arriving in this window is late keying.
        wire.pty.answer_delay = 0.0
        time.sleep(3.0)
        edges, issued = wire.edges(), len(wire.issued)
        alive, cat = wire.alive(), list(wire.pty.commands)
        wire.close()

    t0 = keyed_at[0] - SETTLE
    during = [c for t, c in cat if keyed_at[0] <= t <= keyed_at[-1]]
    print(f"\n  {issued} PTT changes written to rigctl's stdin; it is "
          f"{'still running' if alive else 'GONE -- it exited under the fault'}. "
          f"{len(during)} CAT commands crossed the wire while the five bursts "
          f"went out.")
    print("\n  keying at the wire, against the first burst's carrier:")
    for t, on in edges:
        print(f"    {t - t0:8.4f}  {'KEY DOWN' if on else 'key up  '}")
    print("  bursts emitted at: "
          + ", ".join(f"{t - t0:.4f}" for t in keyed_at))
    _keying("wire, slow rig", edges, keyed_at, STALL_CYCLES, BURST_N / FS,
            issued=issued, budget=False)
    # The direct proof that the CAT path is bypassed. The RTS transition itself
    # is invisible to a pty without the interposer, and even with it the thing
    # worth asserting is this absence: a keyed window measured at the RTS line
    # while the rig's CAT wire carries no PTT command at all.
    ptt_cat = [c for _, c in cat if c.startswith("TX") and c[2:-1]]
    check("wire, slow rig: not one PTT command crossed the CAT wire",
          not ptt_cat, f"{len(ptt_cat)} of {len(cat)} CAT commands were TX1;/TX0;")


def _bounded(what, limit: float):
    """Run `what` on a thread and report (returned, seconds).

    The fault this exists for is a write that never returns, so it cannot be
    called and timed: it would take the suite with it. A thread that is still
    alive at `limit` IS the failure, and the run carries on to say so.
    """
    out = []
    th = threading.Thread(target=lambda: out.append(what()), daemon=True)
    t0 = time.monotonic()
    th.start()
    th.join(limit)
    return (out[0] if out else None) if not th.is_alive() else None, \
        time.monotonic() - t0


def _stage_pipe() -> None:
    """The unkey must not wait on rigctl reading its own stdout.

    `Rig` gives the child a stdout PIPE. Nothing read it, and a pipe nobody reads
    holds about 64 KiB: past that, rigctl blocks writing, stops reading its
    stdin, and a write into that pipe blocks FOREVER WITHOUT RAISING. That
    defeats, in order, the unkey in `transmit`'s finally, its watchdog's, the
    `q` in `_close`, and therefore the belt-and-braces one-shot at the end of
    `stop`, which only runs if the first two return. An independent route to a
    permanently keyed transmitter, and one no amount of PTT rework would touch.

    So both pipes are filled here on purpose, with the drain thread taken away,
    and what is asked is only this: does the unkey still get out. `dump_caps`
    wedges the child in a dozen commands and touches no CAT; after that it is no
    longer reading, and the 64 KiB the kernel holds for its stdin is the entire
    margin -- about sixteen thousand commands, which is a long session and not a
    hypothetical.

    Keyed over CAT, because `TX0;` arriving at the pty is what an unkey looks
    like from the far end. Nothing here is about which line does the keying.
    """
    print("\n== unkey with rigctl's pipes full ==")
    pty = ptyrig.PtyRig()
    saved, ota.drain = ota.drain, lambda proc, errors=None: None
    rig = ota.Rig(ptyrig.FT891_MODEL, pty.path, ptyrig.FT891_BAUD,
                  rigctl_dir=os.path.dirname(ptyrig.RIGCTL))
    try:
        rig._open()
        time.sleep(2.0)
        for _ in range(24):
            if not rig._cmd("1"):             # dump_caps: kilobytes, no CAT
                break
            time.sleep(0.05)
        pending = array.array("i", [0])
        for _ in range(50):     # dump_caps is slower to write than we are to ask
            fcntl.ioctl(rig._proc.stdout.fileno(), termios.FIONREAD, pending, True)
            if pending[0] >= 60000:
                break
            time.sleep(0.1)
        check("the child's stdout pipe is full", pending[0] >= 60000,
              f"{pending[0]} bytes unread, and rigctl is "
              f"{'blocked on it' if rig._proc.poll() is None else 'gone'}")
        sent, took = _bounded(lambda: sum(rig._cmd("T", "0") for _ in range(20000)),
                              20.0)
        check("filling the child's stdin returns", took < 20.0,
              f"{took:.3f} s, {sent} of 20000 commands taken before it "
              f"{'refused' if sent else 'blocked'}")
        went, took = _bounded(lambda: rig.ptt(False), 3.0)
        check("an unkey into the wedged pipe returns", took < 3.0,
              f"{took:.3f} s, reported {went!r}")
        _, took = _bounded(rig.stop, 20.0)
        check("Rig.stop() completes", took < 20.0, f"{took:.3f} s")
        time.sleep(0.3)
        unkeyed = [t for t, on in pty.edges if not on]
        check("the fail-safe unkey reached the wire anyway", bool(unkeyed),
              f"{len(pty.edges)} PTT edges at the rig, "
              f"{len([1 for _, on in pty.edges if on])} of them key-down")
    finally:
        ota.drain = saved
        if rig._proc is not None:
            rig._proc.kill()
        pty.close()


def main() -> int:
    shim = ptyrig.rts_shim()
    if not ptyrig.have_rigctl():
        print(f"\nNO rigctl AT {ptyrig.RIGCTL} -- no verdict. The keying is "
              f"measured through the real Hamlib backend or not at all.")
        return 2
    if shim is None:
        print("\nNO PTY MODEM LINES (tests/ptyrts.c did not build) -- no "
              "verdict. This station keys on RTS, Darwin will not carry RTS on "
              "a pty, and measuring the CAT path instead would be measuring "
              "something that is no longer the keying.")
        return 2
    os.environ["DYLD_INSERT_LIBRARIES"] = shim
    _stage_pipe()
    dev = _loopback()
    if dev is None:
        print(f"\nNO SAMPLE-EXACT LOOPBACK ({LOOPBACK!r}) -- no verdict. This "
              f"measures the transmit path against its own emission, and without "
              f"a device that returns it unaltered nothing below would mean "
              f"anything. Check the device volumes.")
        return 2
    _stage_one(dev)
    _stage_two(dev)
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


@pytest.mark.hardware
def test_duplex() -> None:
    """Needs a live audio loopback AND the machine to itself: the PTT jitter
    budget is 5 ms, and under a full-suite run this fails on scheduling noise
    rather than on anything in the code. Run it with HFMODEM_AUDIO=1."""
    assert main() in (0, 2)


if __name__ == "__main__":
    sys.exit(main())
