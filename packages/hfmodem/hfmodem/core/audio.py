# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The sound card: one stream, one clock, and four lanes reading off it.

Derived from shrike's `_LiveInput`, which is the only audio front end in this
project with measurements behind it, and freed of PACTOR's 1.25 s cycle so the
other three protocols can share it.

## One duplex stream, held open

Two PortAudio streams on one CoreAudio device is the cause of both audio incidents
here: capture RMS fell 0.13 → 0.008 with a second stream open, and a separate
attempt returned err −50. A duplex stream cannot contend with itself.

Holding it open also removes stream-startup latency from the turnaround. A PACTOR
peer answers about 0.29 s after our burst ends; that window was being spent opening
the device, so an answer audible on the speaker never reached the capture at all.

`play_drained` is the same fact seen from the transmit-only tools, which hold no
capture stream to contend with and so open one output stream per burst.

## The grid is made of sample indices, not wall clock

The codec emits 48000 samples a second whatever the scheduler is doing, so a
boundary named by sample index is immune to it. Measured on this machine under
load: sleeping to an absolute deadline lands within 2 ms at the median and **70 ms
at the tail** — and 70 ms is a quarter of the window a peer's answer has to fit
inside. Waiting for sample N asks the OS only to keep a buffer fed.

`holdback` is how far short of a deadline a reader must stop for the final short
sleep to have anything to bridge: a block **plus the input latency**, because the
grid is anchored on the converter's timestamps and a sample exists one latency
before the driver hands it over. Measured before the latency term was there: PTT
went up 0 to 8 ms late, walking cycle to cycle because 1.25 s is 468.75 blocks and
the phase rotates.

## The ADC↔DAC offset is measured, and not from one callback

One frame counter serves both directions, so the capture index our own carrier
came up on is arithmetic rather than an estimate. The offset was measured at 1152
samples with a standard deviation of zero over 22208 callbacks.

That stability is why it is safe to *use*, and it is not a reason to read it once.
A transient in a single callback would make the station either eat the opening
symbols of every reply or feed its own carrier to four decoders — forever, and
self-consistently, because nothing downstream can tell. So it is the median of the
first `_LAT_MEDIAN_N` non-priming callbacks, and the spread is asserted.

## Lanes, and why the callback fans out

Four protocols read the same capture at up to three different rates. The callback
appends to a bounded `deque` per lane: O(1), atomic under the GIL, no lock, and it
drops the oldest for free — which is the policy we want. `queue.Queue` would raise
`Full` instead of dropping and takes a lock a decode thread can hold, and blocking
on that inside a 2.7 ms callback is an xrun.

## Level-blind, everywhere

No RMS gate in the callback or in any lane. besra gated its receive path at 200
int16 while this station's *quiet* channel measures 3935; the gate opened on the
first block and never closed, and a perfect answer would have been discarded ahead
of the demodulator.
"""
from __future__ import annotations

import contextlib
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from hfmodem.core import gil, levels, rates, resample

CARD_RATE_HZ = rates.CARD_RATE_HZ
BLOCKSIZE = 128
LATENCY = "low"

#: Callbacks to median the ADC↔DAC offset over before trusting it.
_LAT_MEDIAN_N = 64

#: A stream that starts and delivers nothing is the silent-total failure this
#: module exists to make loud. It takes about 15 ms in practice.
STARTUP_DEADLINE_S = 2.0

#: The final hop onto a sample boundary, in short sleeps. A long `time.sleep`
#: lands up to 9 ms late, which was two thirds of the observed PTT jitter.
_HOP_S = 0.0005


class AudioError(Exception):
    """The card could not be opened, or started and delivered nothing."""


class BurstInvalidated(Exception):
    """An output underrun happened while a burst was in flight.

    A dropout mid-burst is a hole in what we transmitted. The emission is not
    what the protocol composed, so it is abandoned rather than reported as sent —
    ARQ exists to retry.
    """


#: Output streams left running between bursts, by (device, channels), and empty
#: until a caller asks  [see hold_output].
_HELD: dict[tuple[str, int], object] = {}


def play_drained(samples: np.ndarray, samplerate: float, device) -> None:
    """Play one transmission into `device`, returning once it has been emitted.

    The one call in this project that hands audio to a sound card to transmit, so
    that what `tools/ptt_tail_check.py` measures on the radio is the path the
    modems transmit through rather than a second copy of it. That is a checked
    claim rather than an intention:
    `tests/core/test_audio.py::test_only_core_audio_hands_a_transmission_to_a_sound_card`
    walks every module for a PortAudio output stream opened without a callback —
    which is a stream that exists to be written to, and so is a transmission —
    and this is the only one it allows. `samples` is mono or (frames, channels) —
    a mono stream is inaudible on a multi-channel virtual device, so the callers
    that must fill a channel count shape it themselves.

    ``sd.play(..., blocking=True)`` is not this path and cannot be. Its callback
    raises ``CallbackAbort`` when the array runs out (sounddevice's
    ``_CallbackContext.callback_exit``), which is PortAudio's ``paAbort``:
    *terminate immediately, do not wait for pending buffers to complete*. The
    frames PortAudio was still holding are discarded rather than played, so the
    end of the transmission is thrown away before the key is even considered. How
    much depends on the stream's negotiated output latency — this station's USB
    codec reports 3.42 ms low / 12.75 ms high — which is small, and nothing like
    the quarter second the PTT hold that stood in for this was set to.

    An explicit stream has the drain built in and needs no constant for it:
    ``Stream.stop()`` is ``Pa_StopStream``, which waits until all pending audio
    buffers have been played before it returns. Nothing here estimates how far
    behind the device is, which is the whole point — a fixed hold had to, and
    could only be wrong in one of two directions. Too short cuts the transmission
    off; too long holds an unmodulated carrier on a shared band, and at the 0.25 s
    it once was this station's input measured dead for 0.42 s after every burst,
    which is a peer's whole preamble.

    Opened AND CLOSED per burst unless the caller has said otherwise, and the
    default is not negotiable on a shared device. ``sd.play`` leaves
    sounddevice's module-global output stream open between calls, and a rig whose
    PTT follows audio then stays keyed after CAT has unkeyed. Holding a stream
    open on purpose is worse there: tried on the air 2026-07-28, capture fell from
    ~0.13 RMS to 0.008 — 16x down, "receiver may be DEAF" every cycle, 0 control
    signals in 21 — because two PortAudio streams on one CoreAudio device contend
    for it, and a second one drew err −50 outright. `StationAudio` is the other
    answer to that same fact, one duplex stream that cannot contend with itself;
    this is for the transmit-only paths, which hold no capture stream at all.
    :func:`hold_output` is the third, and it is for the case that measurement is
    about: a transmit device that is not the device the capture holds.

    What remains uncovered is everything below PortAudio: the USB codec's own
    buffering and the rig's audio path, which report nothing. An idle hold past
    the last sample is what covers that, and it is the caller's to hold — a modem
    has work to do between the drain and the unkey, and where that work happens
    is not this function's business.
    """
    import sounddevice as sd

    block = np.ascontiguousarray(np.asarray(samples, dtype=np.float32))
    if block.ndim == 1:
        block = block.reshape(-1, 1)
    if (held := _HELD.get((str(device), block.shape[1]))) is not None:
        # The drain a held stream cannot get from `stop`, in the same currency
        # and not as an estimate of the device's lag: the ring the write fills is
        # what stands between the last sample and the air, and it is empty again
        # exactly when the callback has taken all of it.
        room = held.write_available
        held.write(block)
        while held.write_available < room:
            time.sleep(0.001)
        return
    stream = sd.OutputStream(samplerate=int(samplerate), device=device,
                             channels=block.shape[1], dtype="float32")
    try:
        stream.start()
        stream.write(block)
        stream.stop()              # returns once the device has played it all
    finally:
        stream.close()


#: How much silence a warm-up plays: long enough that the device is running and
#: not merely opened, short enough to sit in front of a key-down unfelt. Paid once
#: per device per process, so it is in no turnaround.
WARM_S = 0.2

#: Which device-and-shape pairs this process has already warmed. The cost the
#: warm-up pays is the FIRST open, so a second one buys nothing and an ARQ link
#: that paid it per burst would put `WARM_S` into every turnaround — against the
#: 0.29 s a PACTOR peer takes to answer.
_WARMED: set[tuple[str, int]] = set()


def warm_output(device, channels: int = 1) -> None:
    """Open `device` and run silence through it, BEFORE the key. Once per process.

    Two things this buys, and both are the caller's reason for calling it before
    PTT rather than after. A device that will not open refuses here, with the
    transmitter still down. And the first open of a session is the expensive one:
    on the air, six 1 s tones sent in one order and then in reverse had the FIRST
    one truncated both times, whichever settle it carried, because the stream
    startup landed inside the keyed period — and the first burst of a session is
    the connect request, which is all a station being called has to lock onto.
    Pre-opening the device, the first tone came up full length.

    What it does not buy is a burst whose own stream was opened before the key:
    `play_drained` opens per burst and must, so that open is still paid under the
    carrier. `tools/ptt_tail_check.py` sizes it — its key-to-first-sample lead has
    no settle in it at all, so the 0.040 s it measured on this station's FT-891 is
    the whole of the CAT acknowledgement, a COLD stream open and the codec's
    latency together, and a warmed one is inside that.

    `channels` is the shape the caller's own bursts carry, so that what this opens
    is the stream they will open: kestrel fills a multi-channel device and the
    rest of the tree sends mono. Nothing measured separates the device's start
    from the format negotiated on top of it, so the warm-up matches the burst
    rather than picking which half to believe costs.
    """
    key = (str(device), channels)
    if key in _WARMED:
        return
    play_drained(np.zeros((int(WARM_S * CARD_RATE_HZ), channels), np.float32),
                 CARD_RATE_HZ, device)
    _WARMED.add(key)


def hold_output(device, samplerate: float, channels: int = 1) -> None:
    """Keep `device`'s output stream running between bursts, from here on.

    What the warm-up cannot buy. Warming pays the process's first open and no
    later one, and on a virtual cable every later open still costs 0.15 s of
    CoreAudio starting the device before a sample leaves — measured from a second
    process on the cable, 0.150-0.192 s per burst, against 0.016-0.021 s once the
    stream has been running. That 0.15 s stands in front of every answer this
    station keys, and a stock VARA HF 4.9.0 responder tolerates about one symbol
    of a late answer's tail.

    ONLY WHERE NOTHING ELSE HOLDS THE DEVICE. Two PortAudio streams on one
    CoreAudio device contend, and the caller's capture stream is one of them
    [see :func:`play_drained`] — so this is for a transmit device that is not the
    receive device, which is what a two-cable bench has and a station with one
    codec does not. The caller owns that distinction; nothing here can see it.

    Idempotent, and :func:`drop_output` puts the device back.
    """
    import sounddevice as sd

    key = (str(device), channels)
    if key in _HELD:
        return
    stream = sd.OutputStream(samplerate=int(samplerate), device=device,
                             channels=channels, dtype="float32")
    stream.start()
    _HELD[key] = stream


def drop_output() -> None:
    """Close every held output stream. A transmitter left keyed by a stream that
    outlives the session is the failure `play_drained` opens and closes to avoid,
    so whoever holds one is the one who has to let it go."""
    while _HELD:
        _, stream = _HELD.popitem()
        stream.stop()
        stream.close()


@dataclass(frozen=True, slots=True)
class Block:
    """One callback's worth of capture, with where it sits on the card clock."""

    adc_time: float
    n0: int
    samples: np.ndarray


class Lane(Protocol):
    """A decoder's view of the capture, at its own rate."""

    rate: int

    def push(self, block: Block) -> None: ...

    def flush_to(self, card_index: int) -> None:
        """Discard everything up to `card_index` — our own transmission."""
        ...

    def resync(self, card_index: int) -> None:
        """A discontinuity happened at `card_index`. Recover, and say so."""
        ...


class _BaseLane:
    """Shared queue and index bookkeeping.

    The callback only ever *appends* a block or *publishes an index*. Everything
    that reads or trims the buffer runs on the consumer's thread. A
    read-modify-write from the callback races the decode thread doing the same, and
    a turnaround is exactly when both happen at once — so the half-duplex flush
    could be silently undone by the very thing it exists to prevent.
    """

    def __init__(self, rate: int, *, depth: int = 4096) -> None:
        self.rate = rate
        self._q: deque[Block] = deque(maxlen=depth)
        self.dropped = 0
        self.resyncs = 0
        self._seen = 0
        #: Published by the callback and the arbiter; consumed on the decode thread.
        self._flush_before = 0
        self._resync_at: int | None = None
        #: Card index of the next sample this lane has not yet accounted for.
        #: None until the first block, because a lane subscribed after the stream
        #: started does not begin at zero — and assuming it does shifts a
        #: sample-locked grid permanently, with nothing to say so.
        self.pos: int | None = None

    def push(self, block: Block) -> None:
        if len(self._q) == self._q.maxlen:
            self.dropped += 1     # oldest goes; a slow decoder cannot stall the card
        self._q.append(block)
        self._seen += 1

    def _drain(self) -> list[Block]:
        """popleft until empty. `list()` then `clear()` silently discards anything
        the callback appended between the two — measured at 12,896 of 60,000 blocks
        under contention, with `dropped` reporting zero. popleft is atomic per item
        under the GIL and under free-threaded CPython."""
        out = []
        while True:
            try:
                out.append(self._q.popleft())
            except IndexError:
                return out

    def _gather(self) -> tuple[np.ndarray, int]:
        """Card-rate float32 from contiguous blocks, and the index it starts at.

        Blocks are assembled by `n0`, so a gap — a deque drop, or a driver
        discontinuity — is detected here rather than spliced silently into the
        buffer. A splice is what makes a lost block invisible to a grid that is
        made of the sample count.
        """
        blocks = self._drain()
        if self._resync_at is not None:
            at, self._resync_at = self._resync_at, None
            blocks = [b for b in blocks if b.n0 >= at]
            self.pos = at
        blocks = [b for b in blocks if b.n0 + len(b.samples) > self._flush_before]
        if not blocks:
            return np.zeros(0, np.float32), self.pos or 0
        if self.pos is None:
            self.pos = blocks[0].n0
        out, at = [], blocks[0].n0
        expect = at
        for b in blocks:
            if b.n0 != expect:          # a hole: start again at this block
                out, at, expect = [], b.n0, b.n0
            out.append(b.samples)
            expect = b.n0 + len(b.samples)
        buf = np.concatenate(out)
        # Trim our own carrier, which is still in flight when finish_burst runs:
        # sample_now leads the delivered count by one input latency, so the last
        # blocks of a burst arrive after the flush index was published.
        if self._flush_before > at:
            skip = min(self._flush_before - at, len(buf))
            buf, at = buf[skip:], at + skip
        return buf, at


class RollingLane(_BaseLane):
    """Overlapping windows, decoded as they fill. besra's measured policy.

    `STEP_S` is how far the stream must advance before decoding again — a
    length gate alone would re-decode the whole buffer on every station tick,
    which measured five times besra's cadence. `OVERLAP_S` is longer than the
    longest frame, so nothing straddles an edge and the resampler's transient
    lands in a neighbouring window's interior. Deduplication is by stream
    position rather than by wall clock, because a frame that sits in a six-second
    buffer is otherwise delivered again on every later poll.
    """

    STEP_S = 0.25
    OVERLAP_S = 6.0

    def __init__(self, rate: int, decode: Callable[[np.ndarray], object],
                 on_frame: Callable[[object], None] | None = None, **kw) -> None:
        super().__init__(rate, **kw)
        self.decode, self.on_frame = decode, on_frame
        self._buf = np.zeros(0, np.float32)
        self._buf_at = 0
        self._decoded_to = 0

    def poll(self) -> list[tuple[int, object]]:
        """Decode what has arrived. Level-blind: nothing is gated.

        Returns (card index, frame) so a position reported by a lane stays an
        exact function of the card's sample count.
        """
        buf, at = self._gather()
        if len(buf):
            if self._buf_at + len(self._buf) != at:
                self._buf, self._buf_at = buf, at       # discontinuity: start over
            else:
                self._buf = np.concatenate([self._buf, buf])
        if not len(self._buf):
            return []
        end = self._buf_at + len(self._buf)
        if end - self._decoded_to < int(self.STEP_S * CARD_RATE_HZ):
            return []
        window = (self._buf if self.rate == CARD_RATE_HZ
                  else resample.from_card(self._buf, self.rate))
        found = [(self._buf_at, f) for f in _iterable(self.decode(window))]
        self._decoded_to = end
        for _, f in found:
            if self.on_frame:
                self.on_frame(f)
        keep = int(self.OVERLAP_S * CARD_RATE_HZ)
        if len(self._buf) > keep:
            self._buf_at += len(self._buf) - keep
            self._buf = self._buf[-keep:]
        return found

    def flush_to(self, card_index: int) -> None:
        self._flush_before = card_index

    def resync(self, card_index: int) -> None:
        self.resyncs += 1
        self._resync_at = card_index


class StreamLane(_BaseLane):
    """Every sample once, in order — for a detector that carries state.

    An adaptive noise floor, a hysteresis gate, a burst bracket: each of those
    learns from the stream, so what it is shown has to be the stream. Overlapping
    windows re-teach it the same audio on every poll, and a raster moves its
    boundaries whenever a block is lost. Neither failure is visible from inside
    the detector, which is why the lane makes the choice instead.

    Resampling is refused rather than done. A stateless resample of each chunk
    puts a transient at every chunk edge, and an energy gate reads those as
    onsets — so a lane that quietly resampled would manufacture the bursts it
    exists to find.
    """

    def __init__(self, rate: int, **kw) -> None:
        if rate != CARD_RATE_HZ:
            raise AudioError(
                f"a stream lane runs at the card rate ({CARD_RATE_HZ} Hz), not "
                f"{rate}: chunk-wise resampling would put an edge transient in "
                "front of a stateful detector")
        super().__init__(rate, **kw)

    def poll(self) -> list[tuple[int, np.ndarray]]:
        """What arrived since the last call, and the card index it starts at.

        A list of at most one, so a caller can treat every lane the same way.
        """
        buf, at = self._gather()
        if not len(buf):
            return []
        self.pos = at + len(buf)
        return [(at, buf)]

    def flush_to(self, card_index: int) -> None:
        self._flush_before = card_index

    def resync(self, card_index: int) -> None:
        self.resyncs += 1
        self._resync_at = card_index


class CycleLane(_BaseLane):
    """A reader locked to absolute sample positions, for a protocol with a raster.

    PACTOR's cycle is 1.25 s of sample indices, so this hands out samples by
    position rather than by availability. A discontinuity is not survivable the
    way it is for a rolling window: the grid is *made of* the sample count, so a
    lost block moves every later boundary. It re-acquires and says so.
    """

    def __init__(self, rate: int, **kw) -> None:
        super().__init__(rate, **kw)
        self.grid_valid = True
        self._buf = np.zeros(0, np.float32)
        self._buf_at = 0

    def take_until(self, card_index: int) -> np.ndarray:
        """Card-rate samples from `pos` up to `card_index`."""
        buf, at = self._gather()
        if len(buf):
            if not len(self._buf):
                self._buf, self._buf_at = buf, at
            elif self._buf_at + len(self._buf) == at:
                self._buf = np.concatenate([self._buf, buf])
            else:
                self.grid_valid = False
                self._buf, self._buf_at = buf, at
        want = card_index - self._buf_at
        if want <= 0 or not len(self._buf):
            return np.zeros(0, np.float32)
        take = self._buf[:want]
        self._buf = self._buf[len(take):]
        self._buf_at += len(take)
        self.pos = self._buf_at
        return take

    def flush_to(self, card_index: int) -> None:
        self._flush_before = card_index
        self.pos = max(self.pos or 0, card_index)

    def resync(self, card_index: int) -> None:
        self.resyncs += 1
        self.grid_valid = False
        self._resync_at = card_index


def _to_card_scale(samples: np.ndarray) -> np.ndarray:
    """Whatever a recording was stored as, in the units the card delivers.

    The divisor is the negative rail rather than the positive one, matching how
    `core.wav` reads and writes: two's complement is asymmetric, so a sample
    written at full scale reads back at full scale only if both ends agree which
    rail full scale is.
    """
    a = np.asarray(samples)
    if np.issubdtype(a.dtype, np.integer):
        return (a.astype(np.float32) / -float(np.iinfo(a.dtype).min))
    return a.astype(np.float32)


def _iterable(x):
    if x is None:
        return ()
    return x if isinstance(x, (list, tuple)) else (x,)


class StationAudio:
    """The card. One stream, one clock, N lanes.

    `open()` blocks until the stream has delivered a block and the ADC↔DAC offset
    has settled, or fails inside `STARTUP_DEADLINE_S`.
    """

    def __init__(self, *, input_device, output_device, gain: float | None = None,
                 blocksize: int = BLOCKSIZE, tx_latency_n: int = 0,
                 trace=None) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.gain = levels.WORKING_POINT if gain is None else gain
        self.blocksize = blocksize
        #: The station's own, from `[audio] tx_latency_ms`. See `loop_lat`.
        self.tx_latency_n = int(tx_latency_n)
        #: `core.cbtrace.CallbackTrace`, or None. Two `time.monotonic()` calls
        #: and eight indexed stores when it is on, and a null test when it is not.
        self.trace = trace

        self.samples = 0
        self.holdback = 0
        self.underruns = 0
        self.overflows = 0
        #: Captured and timestamped by the converter, never handed to Python.
        #: `overflows` is the driver's opinion and it does not cover this.
        self.lost = 0
        self.lat: int | None = None

        self._lanes: list[Lane] = []
        self._stream = None
        self._lat_samples: list[int] = []
        self._last: tuple[float, int] | None = None
        self._ahead: float | None = None
        self._fit: list[tuple[float, int]] = []
        self._burst: tuple[np.ndarray, int] | None = None
        self._emitted = 0
        self._tx_invalid = False
        #: Set by the callback instead of raising. See `_callback`.
        self.failure: BaseException | None = None
        self._lock = threading.Lock()
        self._breathing = contextlib.ExitStack()

    @property
    def loop_lat(self) -> int:
        """Capture samples from a sample's ADC to an armed sample's emission.

        `lat` is what the driver reports -- `outputBufferDacTime` against
        `inputBufferAdcTime` -- and it is not the whole of the path. The codec's
        own in-and-out delay is not in it, and until 2026-09-14 nothing carried
        it, so every burst this station keyed reached the air that much later
        than the index it was armed at. Measured twice on the same arm and
        agreeing to 1 ms: 19.8 ms from where the peer's PACTOR-1 answer sat
        against our 960 ms packet, and ~20 ms from what the PACTOR-3 witness
        chain needs to close. `tx_latency_n` is the station's own measurement of
        the residue and this is the sum the scheduler actually owes.
        """
        return (self.lat or 0) + self.tx_latency_n

    # -- lanes -------------------------------------------------------------

    def subscribe(self, lane: Lane) -> Lane:
        self._lanes.append(lane)
        return lane

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        import sounddevice as sd

        self._stream = sd.Stream(
            samplerate=CARD_RATE_HZ, blocksize=self.blocksize, dtype="float32",
            channels=1, latency=LATENCY,
            device=(self.input_device, self.output_device),
            callback=self._callback)
        self._stream.start()

        deadline = time.monotonic() + STARTUP_DEADLINE_S
        while self.samples == 0 or self.lat is None:
            if self.failure is not None:
                self.close()
                raise AudioError(f"the audio callback failed: {self.failure}")
            if time.monotonic() > deadline:
                self.close()
                raise AudioError(
                    f"the stream on {self.input_device!r}/{self.output_device!r} "
                    f"started but delivered nothing usable in {STARTUP_DEADLINE_S} s. "
                    "Transmitting now would be calling into a channel we cannot hear.")
            time.sleep(0.002)

        lat = self._stream.latency
        in_lat = lat[0] if isinstance(lat, tuple) else lat
        self.holdback = self.blocksize + round(in_lat * CARD_RATE_HZ)

    def breathe(self) -> None:
        """Hand the callback the interpreter on a clock, until `close()`.

        Called from the thread that polls the lanes and not inside `open()`:
        `gil.breathing` naps only the thread that claims it, and the stream may be
        opened from any thread. Scoped to one decode it left the readers around it
        uncovered and the capture lost about a thousand samples a listening cycle;
        held for the life of the stream it loses none at any ring depth
        [see core.gil, tests/core/test_capture_loss].
        """
        self._breathing.enter_context(gil.breathing())

    def close(self) -> None:
        self._breathing.close()
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:      # noqa: BLE001 — teardown must not mask a failure
                pass

    # -- the callback ------------------------------------------------------

    def _callback(self, indata, outdata, frames, t, status) -> None:
        """Never raises.

        sounddevice wraps this with `error=paAbort`, so an exception stops the
        stream permanently — no further callbacks — while `sample_now()` keeps
        extrapolating from the last timestamp against the wall clock. The arbiter
        would go on keying into a channel the station can no longer hear, which is
        precisely the silent-and-total failure this module exists to make loud. So
        a failure is stored and the stream keeps running; `open()` and every
        pre-key path check it.
        """
        try:
            self._body(indata, outdata, frames, t, status)
        except BaseException as exc:            # noqa: BLE001 — nothing may escape
            outdata[:] = 0.0
            self.failure = exc

    def alive(self) -> None:
        """Raise if the card has stopped being trustworthy.

        Checked before a burst is armed, because a dead stream and a quiet band
        look identical from above.
        """
        if self.failure is not None:
            raise AudioError(f"the audio callback failed: {self.failure}")
        if self._stream is not None and not self._stream.active:
            raise AudioError("the audio stream is no longer active")

    def _body(self, indata, outdata, frames, t, status) -> None:
        t_in = time.monotonic() if self.trace is not None else 0.0
        n0 = self.samples
        self.samples += frames

        if status:
            if getattr(status, "input_overflow", False):
                self.overflows += 1
                for lane in self._lanes:
                    lane.resync(n0)
            if getattr(status, "output_underflow", False):
                self.underruns += 1
                burst = self._burst
                if burst is not None and 0 < self._emitted < len(burst[0]):
                    # Only while samples are actually going out. The lead before
                    # `at` and the tail after the last sample are not part of the
                    # emission, and invalidating a burst that went out intact is
                    # the wrong direction to be wrong.
                    self._tx_invalid = True

        ahead = t.inputBufferAdcTime - n0 / CARD_RATE_HZ
        self.lost += rates.lost_step(ahead, self._ahead, self.blocksize,
                                     CARD_RATE_HZ)
        self._ahead = ahead

        self._last = (t.inputBufferAdcTime, n0)
        if self.lat is None and not getattr(status, "priming_output", False):
            self._lat_samples.append(
                round((t.outputBufferDacTime - t.inputBufferAdcTime) * CARD_RATE_HZ))
            if len(self._lat_samples) >= _LAT_MEDIAN_N:
                self.lat = self._settle_lat()

        # The fit point is the anchor pair above -- the converter's own
        # timestamp against its sample index -- not callback-entry time, whose
        # 3 ms rms delivery jitter is load-dependent and at 50 s of stream fits
        # tens of ppm of fiction onto a healthy crystal.
        if not self._fit or t.inputBufferAdcTime - self._fit[-1][0] >= 1.0:
            self._fit.append(self._last)

        # float32 at card rate, copied out of the driver buffer. Converting to
        # int16 here was a 32768x scale error against `resample.from_card`, which
        # takes float in +/-1.0 — measured at 75% of samples on the rail, i.e. a
        # square wave, for every lane not running at the card rate. And scaling by
        # 32767 wraps above full scale: 1.02 became -32114.
        block = Block(t.inputBufferAdcTime, n0, np.array(indata[:, 0], np.float32))
        for lane in self._lanes:
            lane.push(block)

        self._fill(outdata, frames, n0)

        if self.trace is not None:
            self.trace.stamp(t_in, time.monotonic(), t, frames, n0, status)

    def _settle_lat(self) -> int:
        """The median, with the spread checked.

        A spread here is not noise to average away — it means the offset is not a
        constant on this hardware, and everything downstream assumes it is.
        """
        med = int(statistics.median(self._lat_samples))
        spread = max(self._lat_samples) - min(self._lat_samples)
        if spread > self.blocksize:
            raise AudioError(
                f"the ADC/DAC offset varies by {spread} samples over "
                f"{len(self._lat_samples)} callbacks (median {med}). It has been "
                "measured stable to zero on this station's hardware; a varying one "
                "makes every transmit index an estimate.")
        return med

    def _fill(self, outdata, frames: int, n0: int) -> None:
        """Zero unless a burst is armed for this range.

        Zero-filling is the whole reason one held-open stream is safe here. besra
        closes its output stream per burst because `sd.play` leaves a module-global
        stream open and a rig whose PTT follows audio then stays keyed after CAT
        has unkeyed. The invariant it protects is *nothing emitted between
        bursts*, not *no stream* — and this satisfies it while keeping the
        transmit index arithmetic.
        """
        outdata[:] = 0.0
        burst = self._burst
        if burst is None:
            return
        tx, at = burst
        # The DAC range this block covers, in capture indices.
        start = n0 + self.loop_lat
        lo, hi = max(start, at), min(start + frames, at + len(tx))
        if hi <= lo:
            return              # the burst is not in this block, early or late
        # Exact intersection, so a burst that begins mid-block begins mid-block
        # rather than at the next boundary. A cursor instead of an index also
        # stretches a burst across a skipped callback rather than skipping ahead.
        outdata[lo - start:hi - start, 0] = tx[lo - at:hi - at]
        self._emitted = hi - at

    # -- the clock ---------------------------------------------------------

    def sample_now(self) -> float:
        """Where the capture is, interpolated from the last block's ADC timestamp.

        Not from callback entry: delivery jitter is 3 ms rms — a whole PACTOR bit —
        while the converter's own timestamps step by exactly one block, forever.
        """
        if self._last is None:
            return 0.0
        t0, n0 = self._last
        return n0 + (time.monotonic() - t0) * CARD_RATE_HZ

    def clock_ppm(self) -> tuple[float, float]:
        """(ppm, 1 sigma) of the capture clock against the system clock.

        Read beside `lost`, which is what `preflight` does: a slope past a few
        tens of ppm is a shortfall this fit cannot tell from a crystal.
        """
        return rates.rate_fit(self._fit, CARD_RATE_HZ)

    def wait_until(self, card_index: int) -> None:
        """Block until the card has delivered `card_index`, in short hops."""
        while self.sample_now() < card_index:
            time.sleep(_HOP_S)

    # -- transmit ----------------------------------------------------------

    def arm_burst(self, audio: np.ndarray, at: int) -> int:
        """Schedule `audio` so its first sample reaches the DAC as capture index
        `at` reaches the ADC. Returns the index actually used.

        The lead is the ADC->DAC offset plus a few blocks of notice — not
        `holdback`, which is a *reader's* constant and on this machine is 10986
        samples against an offset of 1152. A request that cannot be met is clamped
        forward and the real index returned, because a caller that records where it
        asked for the carrier rather than where it went has recorded nothing.

        One immutable store, not three fields: a callback landing between two of
        them would emit the new burst against the previous index.
        """
        buf = np.asarray(audio, np.float32)
        earliest = int(self.samples) + self.loop_lat + 3 * self.blocksize
        at = max(at, earliest)
        self._tx_invalid = False
        self._burst = (buf, at)                 # single atomic publication
        return at

    def burst_done(self) -> bool:
        burst = self._burst
        return burst is None or self._emitted >= len(burst[0])

    def cancel_burst(self) -> None:
        """Drop an armed burst that never keyed. Emits nothing and records nothing."""
        self._burst = None
        self._emitted = 0
        self._tx_invalid = False

    def finish_burst(self) -> int:
        """Stop emitting, flush every lane past our own carrier, and report where.

        Advancing the lanes here is the one place that knows the sample index, so
        half-duplex is enforced once — replacing besra's `muted` flag and shrike's
        per-session flush with one mechanism.
        """
        burst, self._burst = self._burst, None
        invalid, self._tx_invalid = self._tx_invalid, False
        end = (burst[1] + len(burst[0])) if burst is not None else int(self.samples)
        for lane in self._lanes:
            lane.flush_to(end)
        if invalid:
            raise BurstInvalidated(
                f"{self.underruns} output underrun(s) while the burst was in "
                "flight — what went out is not what was composed.")
        return end


class ReplayAudio:
    """The same lane interface, fed from a recording.

    This is how the transmit-adjacent code is exercised without a transmitter:
    the whole station runs, four lanes decode, four host ports accept, and no
    device is opened.

    A recording is whatever it was stored as — usually 16-bit PCM — and the card
    delivers float32 in [-1, 1]. Converting here rather than at each lane is what
    makes the substitution honest: a replay that handed out integer PCM would put
    audio 32768× hot into every decoder, and `resample.from_card` scales by 32768
    again on the way to ARDOP's 12 kHz, so besra's lane saw nothing but rails. A
    decoder tuned against that is tuned against a signal no card produces.
    """

    def __init__(self, samples: np.ndarray, *, blocksize: int = BLOCKSIZE) -> None:
        self.samples_in = _to_card_scale(samples)
        self.blocksize = blocksize
        self.samples = 0
        self.holdback = 0        # a file has no blocks to hold back for
        self.lat = 0
        self.tx_latency_n = 0
        self.underruns = self.overflows = 0
        self._lanes: list[Lane] = []
        self._i = 0
        self._pending: tuple[np.ndarray, int] | None = None
        #: (card index, samples) for every burst the station would have sent.
        self.transmitted: list[tuple[int, np.ndarray]] = []

    def subscribe(self, lane: Lane) -> Lane:
        self._lanes.append(lane)
        return lane

    def open(self) -> None:
        return None

    def breathe(self) -> None:
        return None

    def close(self) -> None:
        return None

    def pump(self, blocks: int = 1) -> int:
        """Hand the next `blocks` to every lane. Returns how many were delivered."""
        sent = 0
        for _ in range(blocks):
            chunk = self.samples_in[self._i:self._i + self.blocksize]
            if not len(chunk):
                break
            block = Block(self._i / CARD_RATE_HZ, self.samples,
                          np.asarray(chunk, np.float32))
            for lane in self._lanes:
                lane.push(block)
            self._i += len(chunk)
            self.samples += len(chunk)
            sent += 1
        return sent

    def exhausted(self) -> bool:
        return self._i >= len(self.samples_in)

    def sample_now(self) -> float:
        return float(self.samples)

    def wait_until(self, card_index: int) -> None:
        while self.samples < card_index and self.pump(8):
            pass

    # -- transmit, recorded rather than emitted ----------------------------
    #
    # A replay has no transmitter, but it does have an answer to "what would this
    # station have sent?" — which lets a whole ARQ
    # exchange be checked against a recording with no radio in the room. So the
    # burst is kept rather than dropped, and the lanes are still advanced past it,
    # because half-duplex is a property of the protocol rather than of the card.

    def arm_burst(self, audio: np.ndarray, at: int) -> int:
        at = max(at, int(self.samples))
        self._pending = (np.asarray(audio), at)
        return at

    def alive(self) -> None:
        return None

    def burst_done(self) -> bool:
        return getattr(self, "_pending", None) is None

    def cancel_burst(self) -> None:
        self._pending = None

    def finish_burst(self) -> int:
        pending = getattr(self, "_pending", None)
        self._pending = None
        if pending is None:
            return self.samples
        buf, at = pending
        self.transmitted.append((at, buf))
        end = at + len(buf)
        for lane in self._lanes:
            lane.flush_to(end)
        return end
