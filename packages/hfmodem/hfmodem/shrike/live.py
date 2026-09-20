# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Live receive off the sound card: the same rxfront demod, fed continuously from
an audio device instead of a WAV.

`RollingRx` slides a decode window over a stream of audio chunks and emits each
decoded event exactly once -- so a burst that straddles a window boundary is
decoded, not doubled, and the ARQ FSM is never driven twice by the same signal.
It is source-agnostic: a PortAudio callback and a file reader both just call
`push()`, which keeps it testable with no hardware.

`listen()` wires a real input stream to it. With a PtcHost it drives the modem's
receiver (via PtcHost.on_rx_event -- the exact path feed_audio and the monitor
use); without one it prints, a live band monitor. The audio callback only
enqueues; decoding runs on the main thread so a slow decode never underruns the
capture.

    python -m hfmodem.shrike.live --list-devices
    python -m hfmodem.shrike.live --device "USB Audio"                 # live monitor
    python -m hfmodem.shrike.live --device "USB Audio" --mycall W9SSJ  # listen as the modem
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np

from ..core import devices, gil
from . import rxfront

# A window must hold the longest burst we decode whole; keep enough overlap that
# such a burst always lands intact in some window. The longest is a long-cycle
# PACTOR-3 packet: ~3.39 s of air on the 3.75 s cycle. At 4.0/2.5 the slide
# (1.5 s) exceeded window minus body (0.6 s), so a long body straddled most
# windows -- five of the ten long-cycle fields in PIII_Complete_1 were dropped by
# the stream that the same decoder reads whole from a file. 6.0/4.5 keeps the
# slide (cadence and latency) and leaves 2.6 s of placement slack per window.
WINDOW_S = 6.0
KEEP_S = 4.5
# The shortest thing worth decoding: a PACTOR-1 control-signal reply is ~350 ms
# (three copies of a 12-bit codeword at 100 Bd) and the reply test wants a 0.40 s
# span to compare against its guard bands.
MIN_DECODE_S = 0.5


class RollingRx:
    """Decode a stream of audio chunks through rxfront, each event emitted once."""

    def __init__(self, on_event, *, fs: int = rxfront.FS,
                 window_s: float = WINDOW_S, keep_s: float = KEEP_S,
                 min_decode_s: float = MIN_DECODE_S, p3_packets=None):
        self.on_event = on_event
        # Optional live-context callback; None preserves ordinary P3 scanning.
        # Evaluate at each decode, not at construction or the previous feed.
        self.p3_packets = p3_packets
        self._p3_packets_after = None
        # Raised by a caller that is waiting on an answer; see rxfront.
        self.cs_max_errors = rxfront.CS_MAX_ERRORS
        # Set by a caller that does not have the cycle grid yet, and cleared the
        # moment it does; see rxfront.decode_events.
        self.acquiring = False
        self.fs = fs
        self.win = int(window_s * fs)
        self.slide = int((window_s - keep_s) * fs)
        self.min_n = int(min_decode_s * fs)
        self.buf = np.zeros(0, np.float32)
        self.pending = 0                    # new samples since the last decode
        self.t0 = 0.0                       # absolute time of buf[0]
        self.held = 0                       # leading run of samples only `hold` put here
        self._seen: set = set()

    def hold(self, chunk: np.ndarray) -> None:
        """Buffer audio and decode NOTHING -- for a caller that cannot afford it.

        A decode here costs whatever the buffer is long, and the ARQ loop feeds
        its last block of the cycle inside the 25 ms between the peer's control
        signal ending and its own PTT going up. That feed used to be a `push`,
        which crosses the slide as often as not and then decodes the whole cycle:
        67 ms measured, against a 40 ms budget. Held instead, the audio is still
        in the stream for the flush a moment later, which is the decode the slot
        was budgeted for and is capped to 0.75 s.
        """
        self._append(chunk, held=True)

    def forget_held(self) -> None:
        """Drop audio held in FRONT of what is about to be decoded.

        Held audio rides the same buffer `push` decodes, so anything in front of
        the fed tail used to be decoded again on every 0.25 s slide -- and the
        listen window grows by a whole slot every time a slot is handed back.
        Measured on the 2026-09-13 turn, over the arm's own audio: 1026 ms of
        rolling feed on the 2.507 s window a lost slot builds, against 290 ms on
        the 1.250 s one, in a loop bound in samples rather than in wall clock --
        so the overspend lands past the key and the lost slot loses the next.

        NOTHING IS LOST, which is what makes this the whole of the fix. A caller
        that holds rather than feeds does so because some other reader has the
        window: `onair._listen_until_answer`'s frame scan reads it entire, the
        anchored control read is aimed inside it, and `_regrid`'s recovered
        window is scanned and flushed in its own cycle. Read back through the
        arm's 27 hold windows the event count is unchanged and one event moves
        from the rolling decoder to the deep scan. The arithmetic is `skip`'s,
        without its "we were transmitting" meaning.
        """
        if not self.held:
            return
        self.t0 += self.held / self.fs
        self.buf = self.buf[self.held:]
        self.pending = min(self.pending, len(self.buf))
        self.held = 0

    def _append(self, chunk: np.ndarray, *, held: bool) -> None:
        chunk = np.asarray(chunk, np.float32)
        if held and self.held == len(self.buf):
            self.held += len(chunk)
        self.buf = np.concatenate([self.buf, chunk])
        self.pending += len(chunk)
        if len(self.buf) > self.win:
            drop = len(self.buf) - self.win
            self.buf = self.buf[drop:]
            self.t0 += drop / self.fs
            self.held = max(0, self.held - drop)

    def push(self, chunk: np.ndarray) -> None:
        """Buffer audio; decode every slide, over at most one window of history.

        The window is a CAP on how much history each decode sees, not a threshold
        that must be reached before decoding starts. It used to be the threshold,
        which is right for a monitor and unusable for ARQ: a session empties this
        buffer after every transmission, and a 1.25 s cycle never accumulates the
        4 s that a decode then required, so the state machine would have seen
        nothing at all for as long as the link was up.

        AUDIO FED FOR DECODING REPLACES HELD AUDIO IN FRONT OF IT. See
        `forget_held`: what a caller held, it held because it could not afford to
        decode it here, and carrying it into every later slide is how one lost
        slot loses the next. Held audio that arrives BEHIND fed audio -- the
        bridge to the key instant -- is not in front of anything and stays.
        """
        self.forget_held()
        self._append(chunk, held=False)
        if self.pending < self.slide or len(self.buf) < self.min_n:
            return
        self.pending = 0
        self._decode(self.buf)

    def flush(self) -> None:
        """Decode a final short tail (below one window) once the stream ends."""
        if len(self.buf) >= self.fs // 2:
            self._decode(self.buf)
            self.buf = self.buf[:0]
            self.held = 0

    def skip(self, seconds: float) -> None:
        """Drop buffered audio and advance the clock -- we were transmitting.

        Without this the audio either side of our own burst is concatenated into
        one window, so the decoder sees a splice that never went over the air and
        timestamps everything after it early by the length of the transmission.
        """
        self.t0 += len(self.buf) / self.fs + seconds
        self.buf = self.buf[:0]
        self.pending = 0
        self.held = 0

    def enable_p3_packets_after(self, at: float) -> None:
        """Retain an actual enabling control's absolute time for this epoch.

        Session dispatch calls this even when the control was read outside the
        rolling iterator. Keeping it across later flushes prevents an unseen old
        field from merely being postponed until the next now-enabled decode.
        """
        self._p3_packets_after = at

    def _decode(self, seg: np.ndarray) -> None:
        # NO ENVELOPE-ANCHORED PACTOR-3 SEARCH ON A STREAM. Measured on
        # watch_pactor3_maryland.wav, one decode costs 31.8 ms over 0.75 s of
        # audio and 846.7 ms over 0.85 s -- a step, at the length where the
        # buffer first holds a whole PACTOR-3 body, that every slide from there
        # on pays and that no event is ever returned for. A decode dearer than
        # the audio it covers can only fall further behind, and on the air it
        # did: a station over the step stopped keying once a cycle and the
        # gateway dropped it. `hfmodem.tests.shrike.test_rxcost` holds the curve.
        #
        # WHAT COMES OFF IS THE FALLBACK, NOT PACTOR-3. The header-anchored pass
        # still runs, and it is what finds the real thing -- every packet in the
        # two PACTOR-III recordings the corpus holds is anchored on its published
        # header block, none on an envelope edge. The fallback covers the narrow
        # levels, which carry no such block, and stays on for the callers reading
        # files rather than a clock.
        # A decode is compute-bound for twice the audio it covers, and the audio
        # callback is a Python callable that cannot run until this thread lets go.
        # See `core.gil`: it will not let go on its own, whatever the switch
        # interval says.
        def deliver(ev):
            at = self.t0 + ev.t
            # Preserve packet identity, absolute timestamp and overlap dedup.
            ident = repr(ev.packet) if getattr(ev, "packet", None) is not None \
                else ev.text
            key = (ev.kind, round(at, 1), ident)
            if key in self._seen:
                return
            self._seen.add(key)
            was_suppressed = self.p3_packets is not None and not self.p3_packets()
            self.on_event(replace(ev, t=at))
            if was_suppressed and self.p3_packets is not None and self.p3_packets():
                self.enable_p3_packets_after(at)

        read_p3 = self.p3_packets is None or self.p3_packets()
        with gil.breathing():
            for ev in rxfront.decode_events(seg, cs_max_errors=self.cs_max_errors,
                                            acquiring=self.acquiring,
                                            p3_envelope=False,
                                            p3_packets=read_p3,
                                            p3_packets_after=(None if self._p3_packets_after is None
                                                              else self._p3_packets_after - self.t0)):
                deliver(ev)
            # A P1 answer can change the protocol context during this iterator.
            # Read the first upgraded field on this same retained buffer rather
            # than depending on another feed/flush. Do not repeat P1/P2 scans.
            if not read_p3 and self.p3_packets is not None and self.p3_packets():
                for ev in rxfront.p3_packet_events(
                        seg, envelope=False,
                        not_before=(None if self._p3_packets_after is None
                                    else self._p3_packets_after - self.t0)):
                    deliver(ev)
        self._seen = {k for k in self._seen if k[1] > self.t0 - 10}   # bounded


def _print_event(ev) -> None:
    # `.get`, as `monitor` does it: this map has been short a kind since `p1reply`
    # landed, and a live monitor that raises KeyError on an event stops decoding
    # the session it was watching.
    tag = {"connect": "CONNECT", "cs": "CS", "packet": "HEADER",
           "detect": "DETECT", "fsk": "P1-FSK", "p1reply": "P1-BURST",
           "unassigned": "SPARE-CS"}.get(ev.kind, ev.kind.upper())
    print(f"{ev.t:8.2f}  {tag:8s} {ev.text}", flush=True)


def listen(on_event, *, device=None, fs: int = rxfront.FS,
           window_s: float = WINDOW_S, keep_s: float = KEEP_S, stop=None) -> None:
    """Stream the audio device through RollingRx until `stop` is set (Ctrl-C)."""
    import queue
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    rx = RollingRx(on_event, fs=fs, window_s=window_s, keep_s=keep_s)

    def cb(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(device=device, channels=1, samplerate=fs,
                        dtype="float32", callback=cb):
        while stop is None or not stop.is_set():
            try:
                rx.push(q.get(timeout=0.5))
            except queue.Empty:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description="shrike live receive off the sound card")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", help="input device name substring or index")
    ap.add_argument("--mycall", help="listen as the modem with this callsign "
                                     "(drives the ARQ FSM); omit for a band monitor")
    args = ap.parse_args()

    if args.list_devices:
        devices.list_devices()
        return 0

    if args.mycall:
        from .ptc import PtcHost
        host = PtcHost(mycall=args.mycall)
        host.arq.on_host_listen(True)
        print(f"# shrike listening as {host.mycall} (PTCHN {host.ptchn})", flush=True)

        def on_event(ev):
            _print_event(ev)
            host.on_rx_event(ev)
        on_ev = on_event
    else:
        print("# shrike live monitor", flush=True)
        on_ev = _print_event

    try:
        listen(on_ev, device=args.device)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
