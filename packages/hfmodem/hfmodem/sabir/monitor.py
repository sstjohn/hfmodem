# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""sabir monitor: receive-only. Audio in -- a WAV, a live rig input, or a raw
stream on stdin -- and a timestamped log of every sabir burst it decodes: who is
calling whom, the advertised capability images, the ARQ traffic, the beacons.
**It never transmits.**

A text renderer over the same decode path the live receiver runs
(`arq.modem.LinkModem`'s control-block decode plus the beacon decoders), driven
by a streaming energy-gate segmenter whose adaptive noise floor works the same
on a file and on a live stream. sabir's control blocks are self-identifying
(CONNECT/CONNECT_ACK/ID/DISC carry the station id, beacons carry the callsign +
capability image), so stations are named straight from the decode -- no external
gateway list needed.

    python monitor.py capture.wav                 # decode a recording
    python monitor.py --device "USB Audio Device" # live off a rig's RX audio
    kiwirecorder ... --wf | python monitor.py --stdin   # any s16le@48k source

The universal live input is ``--stdin``: pipe raw signed-16-bit mono 48 kHz from
anything -- a KiwiSDR recorder, ffmpeg, sox, arecord. ``--device`` is a
convenience that spawns ffmpeg on a named audio input. Both are INPUT only.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from hfmodem.sabir import offair
from hfmodem.sabir.arq import wire
from hfmodem.sabir.arq.fsm import ArqConfig
from hfmodem.sabir.arq.modem import HEADER_SAMPLES, PREROLL, LinkModem
from hfmodem.sabir.floor import beacon as wspr
from hfmodem.sabir.phy.modem import GUARD_HEAD

CHUNK = 4800                       # live read granularity (~0.1 s at 48 kHz)

FS = offair.FS

# streaming energy-gate segmenter: adaptive floor, hysteresis, hang and pre-roll
SEG_FRAME = 1024                   # energy-analysis frame
BAND_LO, BAND_HI = 700.0, 2700.0   # the floor and the 24-carrier tiers whole;
                                   # the wider 56-carrier tiers clipped
                                   # at both ends
# thresholds are IN-BAND energy ratios over the noise floor: the floor signal is
# ~330 Hz, so measuring energy over the whole audio bandwidth (a wide SSB
# passband) understated in-band SNR by tens of dB. FFT band-power fixes that and
# rejects out-of-band QRM/DC.
SEG_ENTER = 3.0                    # open a burst above this x the in-band floor
SEG_EXIT = 1.8                     # ...close it below this x
SEG_HANG = 6                       # ...after this many trailing quiet frames
SEG_PAD = 4                        # quiet frames of pre-roll kept before a burst
SEG_MIN_S = 0.30                   # shortest burst worth decoding
SEG_MAX_S = (1 + wire.MAX_CAPS) * HEADER_SAMPLES / FS + 1.0  # complete bounded offer
_BAND_MASK: Optional[np.ndarray] = None


def _band_energy(frame: np.ndarray) -> float:
    """Energy of a frame within the sabir band -- the SNR the decoder sees, not
    the full-bandwidth power an SSB passband's noise dominates."""
    global _BAND_MASK
    if _BAND_MASK is None:
        f = np.fft.rfftfreq(SEG_FRAME, 1 / FS)
        _BAND_MASK = (f >= BAND_LO) & (f <= BAND_HI)
    x = np.fft.rfft(frame)[_BAND_MASK]
    return float(np.sum((x * x.conj()).real)) / SEG_FRAME + 1e-12


class Segmenter:
    """Bracket a stream into keyed bursts incrementally -- ``push`` a chunk, get
    back the bursts that completed within it; ``flush`` at end of stream. The
    noise floor is a rolling low percentile of *idle* in-band energy (frozen
    while a burst is open): a handful of silent frames -- squelch tail, T/R
    mute, a dropout -- cannot collapse it the way a running minimum would, and a
    stuck-open burst re-seeds the floor on its force-close. Identical on a file
    and on a live device."""

    def __init__(self):
        import collections
        self.buf = np.zeros(0)
        self.t0 = 0                    # global sample index of buf[0]
        self.pos = 0                   # frame cursor within buf
        self.idle = collections.deque(maxlen=int(5 * FS / SEG_FRAME))
        self.inb = False
        self.start = 0
        self.quiet = 0
        self._e: list[float] = []      # in-burst energies, for force-close re-seed

    def _floor(self) -> float:
        if len(self.idle) >= 8:
            return float(np.percentile(self.idle, 30)) + 1e-12
        return (float(np.median(self.idle)) if self.idle else 0.0) + 1e-12

    def push(self, x: np.ndarray) -> list[tuple[int, np.ndarray]]:
        self.buf = np.concatenate([self.buf, np.asarray(x, float)])
        out: list[tuple[int, np.ndarray]] = []
        n = SEG_FRAME
        while len(self.buf) - self.pos >= n:
            fr = self.buf[self.pos:self.pos + n]
            e = _band_energy(fr)
            floor = self._floor()
            gs = self.t0 + self.pos
            if not self.inb:
                self.idle.append(e)
                if e > floor * SEG_ENTER:
                    self.inb, self.quiet, self._e = True, 0, [e]
                    pre = min(SEG_PAD, self.pos // n)
                    self.start = gs - pre * n
            else:
                self._e.append(e)
                if e < floor * SEG_EXIT:
                    self.quiet += 1
                elif e > floor * SEG_ENTER:
                    self.quiet = 0
                over = (gs + n - self.start) >= SEG_MAX_S * FS
                if self.quiet >= SEG_HANG or over:
                    if over:                          # a carrier/false-open never
                        self.idle.extend(self._e)     # fell silent -- re-learn from it
                    self._emit(out, gs + n)
            self.pos += n
        if not self.inb and self.pos > 5 * FS:        # compact the idle backlog
            self.buf, self.t0, self.pos = self.buf[self.pos:], self.t0 + self.pos, 0
        return out

    def _emit(self, out: list, end: int) -> None:
        lo = self.start - self.t0
        burst = self.buf[max(0, lo):end - self.t0]
        if burst.size >= SEG_MIN_S * FS:
            out.append((self.start, burst))
        self.inb, self.quiet = False, 0

    def flush(self) -> list[tuple[int, np.ndarray]]:
        out: list[tuple[int, np.ndarray]] = []
        if self.inb:
            self._emit(out, self.t0 + len(self.buf))
        return out


def segment(audio: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Batch wrapper: every keyed burst in a whole capture, via the streaming
    Segmenter (one implementation for file and live)."""
    seg = Segmenter()
    for i in range(0, len(audio), CHUNK):
        yield from seg.push(audio[i:i + CHUNK])
    yield from seg.flush()


@dataclass
class Event:
    t: float                       # seconds into the capture
    kind: str
    text: str


_TYPE = {wire.CONNECT: "CONNECT", wire.CONNECT_ACK: "CONNECT_ACK",
         wire.DATA: "DATA", wire.ACK: "ACK", wire.DISC: "DISC",
         wire.DISC_ACK: "DISC_ACK", wire.TURN: "TURN",
         wire.TURN_REQ: "TURN_REQ", wire.ID: "ID", wire.CAPS: "CAPS"}


def _features(word: int) -> str:
    f = wire.decode_capabilities(word)
    return f"profiles {f['profiles']} {[k for k, v in f.items() if v is True]}"


def _describe(blk) -> tuple[str, str]:
    if isinstance(blk, wire.DatagramHeader):
        return "DATAGRAM", f"profile {blk.gear} bytes {blk.size}"
    if isinstance(blk, wire.Beacon):
        return "BEACON", f"presence '{blk.call}' {_features(blk.capability_word)}"
    if isinstance(blk, wire.Caps):
        return "CAPS", f"session {blk.session:#04x} block {blk.idx + 1}/{blk.total}"
    t = blk.type
    name = _TYPE.get(t, f"type{t}")
    if t in (wire.CONNECT, wire.CONNECT_ACK):
        return name, f"'{blk.call}' {_features(blk.capability_word)}"
    if t in (wire.ID, wire.DISC, wire.DISC_ACK):
        return name, f"'{blk.call}'"
    if t == wire.DATA:
        g = wire.GEAR_NAME.get(blk.gear & wire.GEAR_MASK, f"gear{blk.gear & 0xF}")
        return name, f"session {blk.session:#04x} seq {blk.seq} gear {g}"
    if t == wire.ACK:
        n = bin(int.from_bytes(blk.mask, "big")).count("1")
        return name, f"session {blk.session:#04x} seq {blk.seq} {n} codewords acked"
    return name, f"session {blk.session:#04x}"


def _decode_burst(modem: LinkModem, z: np.ndarray) -> tuple[list, bool]:
    """Passive control-block decode of a whole burst: ``LinkModem._header``
    with the body dropped. Nothing here has to place what follows the header,
    so the floor tier reads the burst end to end rather than a bounded
    window."""
    for carrier, n in ((modem.fast, 1), (modem.fast, 2), (modem.short_fast, 1)):
        need = carrier.n_samples(n)
        if z.size < need:
            continue
        raw, _ = carrier.receive(z[: need + PREROLL], n, dd=modem.dd)
        if raw is not None:
            blocks = [wire.Control.unpack(raw[i: i + carrier.block_bytes])
                      for i in range(0, len(raw), carrier.block_bytes)]
            if all(b is not None for b in blocks):
                return blocks, True
    raw, st = modem.floor.receive(z, wire.BLOCK_BYTES)  # floor tier: scans z
    ctrl = wire.Control.unpack(raw) if raw is not None else None
    if ctrl is None:
        raw, st = modem.floor.receive(z, wire.CONNECTIONLESS_BYTES)
        ctrl = wire.Control.unpack(raw) if raw is not None else None
        if ctrl is None:
            return [], False
    blocks = [ctrl]
    if ctrl.type in (wire.CONNECT, wire.CONNECT_ACK):
        base = max(0, int(st.get("start", 0)) - GUARD_HEAD)
        for idx in range(ctrl.n_ext):
            extension, _ = modem._floor_block(z, base + (idx + 1) * HEADER_SAMPLES)
            if extension is not None and extension.type == wire.CAPS:
                blocks.append(extension)
    return blocks, False


def decode_burst(modem: LinkModem, s0: int, burst: np.ndarray) -> Iterator[Event]:
    """One segmented burst -> its decoded events. Every burst runs the modem's
    own control-block decode across both tiers; a non-control burst is tried
    against the narrow-tone beacon gears; anything else is a bare detection."""
    t = s0 / FS
    z = offair.to_analytic(burst)
    blocks, fast = _decode_burst(modem, z)
    if blocks:
        tier = "fast " if fast else ""
        for blk in blocks:
            kind, text = _describe(blk)
            yield Event(t, kind, tier + text)
        return
    for g in offair._WSPR_GEARS:
        pl, _ = wspr.recv_beacon(z, wspr.BEACON_GEARS[g])
        if pl is not None:
            yield Event(t, "WSPR", f"{pl.callsign} {pl.grid} ({g})")
            return
    yield Event(t, "DETECT", f"burst {burst.size / FS:.1f} s, no decode")


def decode_events(audio: np.ndarray,
                  modem: Optional[LinkModem] = None) -> Iterator[Event]:
    """Batch receive core: a whole capture in, decoded sabir events out."""
    modem = modem or LinkModem(ArqConfig())
    for s0, burst in segment(audio):
        yield from decode_burst(modem, s0, burst)


def decode_stream(chunks: Iterator[np.ndarray],
                  modem: Optional[LinkModem] = None) -> Iterator[Event]:
    """Live receive core: a stream of audio chunks in, events out as each burst
    completes -- the same Segmenter and decode as the batch path."""
    modem = modem or LinkModem(ArqConfig())
    seg = Segmenter()
    for chunk in chunks:
        for s0, burst in seg.push(chunk):
            yield from decode_burst(modem, s0, burst)
    for s0, burst in seg.flush():
        yield from decode_burst(modem, s0, burst)


# -- audio sources (INPUT only) -----------------------------------------------
def wav_chunks(path: str) -> Iterator[np.ndarray]:
    audio = offair.wav_read(path)
    for i in range(0, len(audio), CHUNK):
        yield audio[i:i + CHUNK]


def raw_chunks(stream, dtype="<i2", scale=1 / 32768.0) -> Iterator[np.ndarray]:
    """Signed-16-bit mono 48 kHz frames off a binary stream (default stdin). A
    truncated final sample (producer killed mid-write) is dropped, not crashed
    on, so the segmenter still flushes its in-flight burst."""
    width = np.dtype(dtype).itemsize
    rem = b""
    while True:
        raw = rem + stream.read(CHUNK * width)
        if not raw:
            return
        usable = len(raw) - len(raw) % width
        rem = raw[usable:]
        if usable:
            yield np.frombuffer(raw[:usable], dtype).astype(float) * scale
        elif not stream.read(1):                      # nothing more coming
            return


def _ffmpeg_input(device: str) -> list[str]:
    s = platform.system()
    if s == "Darwin":
        return ["-f", "avfoundation", "-i", f":{device}"]
    if s == "Windows":
        return ["-f", "dshow", "-i", f"audio={device}"]
    return ["-f", "alsa", "-i", device]


def device_chunks(device: str) -> Iterator[np.ndarray]:
    """Live audio off a named INPUT device via ffmpeg (s16le mono 48 kHz).
    No output stream, no PTT -- receive only."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *_ffmpeg_input(device),
           "-ac", "1", "-ar", str(FS), "-f", "s16le", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found -- install it, or pipe audio to "
                         "--stdin instead")
    any_audio = False
    try:
        for chunk in raw_chunks(proc.stdout):
            any_audio = True
            yield chunk
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not any_audio:                                  # ffmpeg opened nothing
        err = (proc.stderr.read() if proc.stderr else b"").decode(
            "utf-8", "replace").strip()
        raise SystemExit(f"ffmpeg could not open {device!r}"
                         + (f": {err.splitlines()[-1]}" if err else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sabir receive-only monitor")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("wav", nargs="?", help="decode a recording")
    src.add_argument("--device", help="live INPUT device carrying receiver audio")
    src.add_argument("--stdin", action="store_true",
                     help="live raw s16le mono 48 kHz on stdin (any source)")
    a = ap.parse_args(argv)

    if a.wav:
        label, events = a.wav, decode_events(offair.wav_read(a.wav))
    elif a.device:
        label, events = f"device {a.device!r}", decode_stream(device_chunks(a.device))
    else:
        label, events = "stdin", decode_stream(raw_chunks(sys.stdin.buffer))

    print(f"# sabir monitor | {label} @ {FS} Hz | receive-only, never transmits",
          flush=True)
    print(f"#   t(s)  {'kind':11s} detail", flush=True)
    seen = False
    try:
        for ev in events:
            seen = True
            print(f"{ev.t:8.2f}  {ev.kind:11s} {ev.text}", flush=True)
    except KeyboardInterrupt:
        pass
    if a.wav and not seen:
        print("#  (no sabir signal above the noise floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
