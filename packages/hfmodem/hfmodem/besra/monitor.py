# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Monitor mode: audio in, a timestamped log of everything besra decodes.

A text renderer over `besra.phy.demodulator` — the exact decode path the live
modem receiver runs — so the monitor is a faithful dry run of the receiver. Every
line names the frame from its decode and marks integrity: a data/ConReq/ID frame
shows its recovered content only when its CRC/RS validates (`ok`), otherwise it is
flagged. besra never transmits in this mode: there is no rig, no PTT and no output
device anywhere in this module.

Live off a receiver it also **records**, because a Winlink gateway answering us is
the receive-side fixture besra does not have and an unrecorded window produces
nothing. The recording is one unbroken WAV of everything the sound card delivered,
written unnormalised with a JSON level sidecar beside it.

    python -m hfmodem.besra.monitor <capture.wav>
    <int16 stream> | python -m hfmodem.besra.monitor --stdin
    python -m hfmodem.besra.monitor --device "USB Audio"      # live off a receiver, recorded
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
import wave
from pathlib import Path

import numpy as np

from ..core.rates import CARD_RATE_HZ, to_native
from .frame import frame as F
from .phy.demodulator import SAMPLE_RATE, DecodedFrame, Demodulator


def _load_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
            raise SystemExit(f"{path}: need 12 kHz mono, got "
                             f"{w.getframerate()} Hz / {w.getnchannels()} ch")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def _render(frame: DecodedFrame) -> str:
    fd = F.FRAMES.get(frame.type)
    name = fd.name if fd else f"0x{frame.type:02X}"
    if frame.header_only:
        flag = "  !HEADER ONLY (type unconfirmed)"
    elif frame.unverified:
        flag = "  !UNVERIFIED (no body to check)"
    else:
        flag = "" if frame.ok else "  !CRC/RS FAIL"
    detail = ""
    if name.startswith("ConReq") or name == "Ping":
        detail = f"  {frame.caller} > {frame.target}"
    elif name == "IDFrame":
        detail = f"  {frame.caller}  {frame.grid or ''}".rstrip()
    elif name.startswith("ConAck"):
        detail = f"  leader {frame.conack_timing_ms} ms"
    elif name == "PingAck":
        detail = f"  S/N {frame.pingack_sn_db} dB  Q {frame.pingack_quality}"
    elif frame.payload:
        head = frame.payload[:16].hex()
        detail = f"  {len(frame.payload)} B  {head}{'…' if len(frame.payload) > 16 else ''}"
    return f"0x{frame.type:02X} {name:<16} sess 0x{frame.session_id:02X}{detail}{flag}"


_TAIL = np.zeros(4800, dtype="<i2")


def _decode_capture(demod: Demodulator, samples: np.ndarray, base_s: float) -> None:
    """Decode the whole capture in one pass — the demodulator locates every frame
    by its leader — and print one line per frame, timestamped at the frame's own
    position in the stream so nothing is dropped or clipped by pre-segmentation."""
    au = np.concatenate([samples, _TAIL]) if samples.size else samples
    for f in demod.decode(au):
        print(f"{base_s + f.offset / SAMPLE_RATE:8.2f}  {_render(f)}", flush=True)


_HEARTBEAT_S = 10.0
_FULL_SCALE = 32767 / 32768      # a float sample at or past this railed the converter


def _dbfs(v: float) -> float:
    # A dead input floors instead of reading -inf, so the heartbeat line stays readable.
    return 20 * math.log10(v) if v > 0 else -120.0


class _Level:
    """Peak, RMS and railed fraction of a float32 capture, against FULL SCALE.

    Against full scale, never against the capture's own peak — that reads 100%
    railed by construction and would report a healthy quiet band as a clipped one."""

    def __init__(self) -> None:
        self.n = self.railed = 0
        self.peak = self.sumsq = 0.0

    def add(self, x: np.ndarray) -> None:
        a = np.abs(x.astype(np.float64))
        self.n += a.size
        self.peak = max(self.peak, float(a.max(initial=0.0)))
        self.sumsq += float(a @ a)
        self.railed += int(np.count_nonzero(a >= _FULL_SCALE))

    @property
    def rms(self) -> float:
        return math.sqrt(self.sumsq / self.n) if self.n else 0.0

    @property
    def railed_pct(self) -> float:
        return 100.0 * self.railed / self.n if self.n else 0.0


class LiveMonitor:
    """The live receive path: record everything, decode continuously, report levels.

    The recording is unbroken by construction — it is written from the raw capture
    before anything looks at it, so no energy gate or decode decision can shape it.
    That is deliberate: a capture chopped into per-burst files does not merely lose
    the audio between bursts, it can decode to *different well-formed* content. The
    sibling PACTOR work measured one off-air signal reading callsign `WS8EOC` from
    continuous audio and `WE8ON` / `W8MC` from the same audio cut into windows.

    The decode is `radio.RollingDecoder`, the same rolling window the ARQ receive
    path runs — one receive design, gated on nothing, so what the monitor hears is
    what a live session would have heard.

    Audio arrives on the sound card's thread and is only queued there; `pump` does
    the recording, decoding and reporting on the caller's thread. Feeding `push` and
    `pump` by hand is how the live path is exercised without a sound card.

    `close` prints the session's levels and its acquisition funnel
    (`demodulator.RxFunnel`). The funnel's other two surfaces are a host STATUS
    line, which needs a client attached to read it, and the session log, which is
    read afterwards; this is the one an operator is watching while there is still
    time to retune."""

    def __init__(self, demod: Demodulator, wav_path, *, source: str = "") -> None:
        # scipy, which the WAV and stdin paths would pay ~0.3 s of import for and
        # never use.
        from .radio import RollingDecoder
        self._rate = CARD_RATE_HZ
        self._rolling = RollingDecoder(demod, self._report)
        self._source = source
        self._path = Path(wav_path)
        self._wav = wave.open(str(self._path), "w")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self._rate)

        self._t0 = time.time()
        self._n = 0                              # capture samples recorded
        self.level = _Level()
        self._since_hb = _Level()

    @property
    def _frames(self) -> int:
        return self._rolling.funnel.frames

    def push(self, block: np.ndarray) -> None:
        """The sound-card callback. Queue only — never decode on the audio thread."""
        self._rolling.push(block)

    def pump(self) -> None:
        au = self._rolling.pump()
        if au.size:
            self.level.add(au)
            self._since_hb.add(au)
            self._wav.writeframes(
                np.clip(np.round(au * 32768.0), -32768, 32767).astype("<i2").tobytes())
            self._n += au.size
        if self._since_hb.n >= _HEARTBEAT_S * self._rate:
            self._heartbeat()

    def close(self) -> None:
        self.pump()                              # whatever the last cadence left queued
        self._rolling.flush()                    # the tail the step cadence never reached
        self._wav.close()
        self._path.with_suffix(".json").write_text(
            json.dumps(self._sidecar(), indent=2) + "\n")
        print(f"# recorded {self._n / self._rate:.1f} s to {self._path.name}  "
              f"peak {_dbfs(self.level.peak):.1f} dBFS  rms {_dbfs(self.level.rms):.1f} dBFS  "
              f"railed {self.level.railed_pct:.3f}%  frames {self._frames}\n"
              f"# {self._rolling.funnel}", flush=True)

    def _report(self, pos: int, frame: DecodedFrame) -> None:
        print(f"{self._stamp(pos)}  {_render(frame)}", flush=True)

    def _heartbeat(self) -> None:
        """Levels since the last heartbeat, so a silent monitor is distinguishable
        from a dead audio path: a quiet band still reports a noise floor."""
        lv, self._since_hb = self._since_hb, _Level()
        clip = f"CLIPPING {lv.railed_pct:.2f}%" if lv.railed else "no clip"
        print(f"{self._stamp(to_native(self._n, SAMPLE_RATE))}  -- peak "
              f"{_dbfs(lv.peak):6.1f} dBFS  rms {_dbfs(lv.rms):6.1f} dBFS  {clip}  "
              f"frames {self._frames}", flush=True)

    def _stamp(self, pos: int) -> str:
        """Wall clock and elapsed, both taken from the frame's own position in the
        stream rather than from when its decode finished — so a decode correlates
        with operator action, and inter-frame cadence is measurable off the log."""
        el = pos / SAMPLE_RATE
        return f"{time.strftime('%H:%M:%S', time.gmtime(self._t0 + el))}Z {el:9.2f}"

    def _sidecar(self) -> dict:
        # Levels are as captured. Nothing normalises the recording, so these stay the
        # truth about what the receiver actually handed over.
        return {
            "wav": self._path.name,
            "source": self._source,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._t0)),
            "samplerate": self._rate,
            "channels": 1,
            "samples": self._n,
            "duration_s": round(self._n / self._rate, 3),
            "peak": round(self.level.peak, 6),
            "peak_dbfs": round(_dbfs(self.level.peak), 2),
            "rms_dbfs": round(_dbfs(self.level.rms), 2),
            "railed_pct": round(self.level.railed_pct, 4),
            "frames_decoded": self._frames,
        }


def _interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def _live(demod: Demodulator, device, record: Path) -> None:
    """Decode a receiver's audio live off a sound card, recording it unbroken."""
    from . import radio
    mon = LiveMonitor(demod, record, source=str(device))
    print(f"# recording to {record}", flush=True)
    # A window ends however it ends — Ctrl-C, `timeout`, a session teardown. The
    # sidecar is written on close, and a capture without one cannot be shown to
    # have been un-railed, which is the only reason to keep the level record at
    # all. Measured: SIGTERM used to take the audio and drop the sidecar.
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        with radio.capture_stream(device, mon.push):
            while True:
                time.sleep(0.25)
                mon.pump()
    except KeyboardInterrupt:
        pass
    finally:
        mon.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="besra ARDOP monitor (receive only)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("wav", nargs="?", help="12 kHz mono WAV capture")
    src.add_argument("--stdin", action="store_true",
                     help="read a raw little-endian int16 mono stream at 12 kHz")
    src.add_argument("--device", help="live INPUT device (name substring or index) "
                                      "carrying a receiver's audio")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--chunk", type=float, default=8.0,
                    help="stream decode window in seconds (--stdin)")
    ap.add_argument("--record", type=Path,
                    help="WAV to record the live capture to (--device; defaults to a "
                         "timestamped file in the working directory)")
    args = ap.parse_args()

    if args.list_devices:      # a diagnostic that needs no source
        from ..core.devices import list_devices
        list_devices()
        return 0

    if not (args.stdin or args.device or args.wav):
        ap.error("one of the arguments wav --stdin --device is required")
    if args.record and not args.device:
        ap.error("--record applies to --device (a WAV or stdin source is already a recording)")

    src_name = "stdin" if args.stdin else args.device or Path(args.wav).name
    print(f"# besra monitor  |  {src_name}  |  12 kHz", flush=True)

    demod = Demodulator()
    if args.device:
        # Always recorded: the reason to sit on a frequency is to come away with the
        # audio, and a default nobody has to remember is the only kind that survives
        # a live window.
        _live(demod, args.device, args.record or Path(
            time.strftime("besra-monitor-%Y%m%d-%H%M%SZ.wav", time.gmtime())))
    elif args.stdin:
        win = int(args.chunk * SAMPLE_RATE)
        base = 0.0
        buf = np.empty(0, dtype="<i2")
        while True:
            raw = sys.stdin.buffer.read(win * 2)
            if not raw:
                break
            buf = np.concatenate([buf, np.frombuffer(raw, dtype="<i2")])
            if len(buf) >= win:
                _decode_capture(demod, buf, base)
                base += len(buf) / SAMPLE_RATE
                buf = np.empty(0, dtype="<i2")
        if len(buf):
            _decode_capture(demod, buf, base)
    else:
        _decode_capture(demod, _load_wav(args.wav), 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
