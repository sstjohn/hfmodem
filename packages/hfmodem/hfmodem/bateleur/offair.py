# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Render a bateleur waveform transmission to a WAV, or decode one from a WAV.

``render`` writes audio at the selected sample rate and drive, with optional
Morse identification, and prints the declared audio span. ``decode`` reads
a WAV. The operator chooses how to use the resulting files.

    python -m hfmodem.bateleur.offair modes
    python -m hfmodem.bateleur.offair render 110 --mode 600L --callsign N0CALL \\
        --text "hello" out.wav
    python -m hfmodem.bateleur.offair render 110 --mode 2400S --bandwidth 3200 \\
        --text "hello" wide.wav
    python -m hfmodem.bateleur.offair decode 110 capture.wav
    python -m hfmodem.bateleur.offair decode 4285 --mode 1200L capture.wav
    python -m hfmodem.bateleur.offair decode 4539 capture.wav
    python -m hfmodem.bateleur.offair render WBHF --mode 9600L --text "hello" out.wav
    python -m hfmodem.bateleur.offair decode WBHF capture.wav
    python -m hfmodem.bateleur.offair decode MOD --all --carrier 1814 capture.wav
    python -m hfmodem.bateleur.offair render 39 --mode 600/99 --text "hello" out.wav
    python -m hfmodem.bateleur.offair decode 39 capture.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hfmodem.core import config, cwid, levels, wav
from hfmodem.core.occupied import keyed_hz
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.regulatory.part97 import MAX_BANDWIDTH_HZ
from hfmodem.core.resample import for_card

from . import charframe, fsk, mod110, s4529, s4539, tone16, tone39, wbhf
from .modem import Mode
from .spec import BAUD, CARRIER_HZ, FS, MODES110, MODES4285

#: The audio span every mode here fills, -26 dB edges, per rig filter width:
#: the station's rows, which ``tests/bateleur/test_phy.py`` measures every mode
#: against. The Morse identification at 700 Hz sits inside both.
OCCUPIED_AUDIO_HZ = {bw: keyed_hz("bateleur", bw) for bw in ("3000", "3200")}

#: Between the message and the identification: one key-up, two signals.
_ID_GAP_S = 0.5


def _payload(a) -> bytes:
    if a.text is not None:
        return a.text.encode("utf-8")
    if a.input == "-":
        return sys.stdin.buffer.read()
    return Path(a.input).read_bytes()


def render(standard: str, mode: str, data: bytes, callsign: str | None,
           bandwidth: str = "3000") -> np.ndarray:
    """The transmission at ``spec.FS``, peak 1.0, identification behind it.
    ``bandwidth`` is the rig's top audio edge: 3000 trims ours to fit under it."""
    x = Mode.parse(standard, mode).render(data, trim=bandwidth == "3000")
    if callsign:
        ident = cwid.audio(callsign.upper(), fs=FS)
        ident = ident / (float(np.abs(ident).max()) or 1.0)
        x = np.concatenate([x, np.zeros(int(_ID_GAP_S * FS)), ident])
    return x


def cmd_modes(a) -> None:
    spans = ", ".join(f"{lo:.0f}-{hi:.0f} Hz for a {bw} Hz filter"
                      for bw, (lo, hi) in OCCUPIED_AUDIO_HZ.items())
    print(f"serial (110, 4285, 4539): {BAUD} Bd on {CARRIER_HZ} Hz, audio {spans}")
    print("\nMIL-STD-188-110 serial (the receiver reads the mode off the preamble)")
    for m in MODES110.values():
        il = {"none": "uncoded", "zero": "zero interleave (decode with --zero)"}.get(
            m.interleave, f"{m.interleave} interleave")
        print(f"  {m.name:6s} {m.rate:5d} bit/s  {il}")
    print("\nSTANAG 4539 / MIL-STD-188-110B appendix C (the receiver reads the mode "
          "off the preamble)")
    for name, m in s4539.MODES4539.items():
        il = "uncoded" if not m.coded else f"{m.frames}-frame interleave"
        print(f"  {name:7s} {m.rate:5d} bit/s  {il}")
    print("\nMIL-STD-188-110C appendix D, 3 kHz (the receiver reads the waveform off the "
          "preamble)")
    for name, m in wbhf.MODES_WBHF.items():
        mod = "Walsh" if m.walsh else {1: "BPSK", 2: "QPSK", 3: "8-PSK"}.get(
            m.w.bits, f"{1 << m.w.bits}-QAM")
        print(f"  {name:7s} {m.w.rate:5d} bit/s  waveform {m.waveform}, {mod}, "
              f"{m.frames * m.frame_symbols / BAUD:.2f} s interleave")
    print("\nMIL-STD-188-110 appendix B, 39 tones (the receiver finds the mode from the data)")
    print("  <rate>/<degree>[/S | /A<char bits>.<data bits>][F][X], e.g. 600/99 (ten-bit "
          "characters of eight data bits), 2400/288/S (synchronous), 75/36/SF")
    for rate, degs in tone39.DEGREES.items():
        print(f"  {rate:5d} bit/s  degrees {', '.join(map(str, degs))}")
    print("\nMIL-STD-188-110 appendix A, 16 tones (both ends must be set to the same rate)")
    print(f"  {', '.join(map(str, tone16.RATES))} bit/s, uncoded")
    print("\nBinary FSK: STANAG 4481 broadcast and MIL-STD-188-110B 5.1 (both ends set "
          "the same mode)")
    print("  <baud>W: 850 Hz shift about 2000 Hz (W/1700: about 1700 Hz); <baud>N: 170 Hz")
    print("  <baud>V<n>: multi-channel sub-carrier n (255 + 170n Hz, 85 Hz shift); "
          "V<n>D: twinned with n+8")
    print("  an S after any: the bare synchronous stream, no character framing")
    print(f"  {', '.join(map(str, fsk.RATES))} Bd; e.g. 75W, 75W/1700, 50N, 75V3, 75V3D, 75WS")
    print("\nSTANAG 4529: 4285 at 1200 Bd in 1240 Hz (both ends set the same mode)")
    print(f"  {', '.join(s4529.MODES4529)}; /<Hz> after it for a sub-carrier other "
          f"than 1700 (800-2400 in 100 Hz steps)")
    print("\nSTANAG 4285 (both ends must be set to the same mode)")
    for name, m in MODES4285.items():
        il = "uncoded" if not m.coded else f"{m.interleave} interleave"
        print(f"  {name:6s} {m.rate:5d} bit/s  {il}")


def cmd_render(a) -> None:
    try:
        a.mode = Mode.parse(a.standard, a.mode).name
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    data = _payload(a)
    sent = data
    if a.async_fmt:
        sent = charframe.pack(charframe.frame(data, _char_format(a)))
    x = render(a.standard, a.mode, sent, a.callsign, a.bandwidth)
    drive = config.tx_drive(a.drive)
    card = levels.at_drive(for_card(x, FS), drive)
    wav.write(a.out, card, CARD_RATE_HZ, source="hfmodem.bateleur.offair",
              standard=a.standard, mode=a.mode, payload_bytes=len(data),
              callsign=a.callsign or "", bandwidth=a.bandwidth)
    lo, hi = OCCUPIED_AUDIO_HZ[a.bandwidth]
    title = {"110": "MIL-STD-188-110 serial", "4285": "STANAG 4285",
             "4539": "STANAG 4539", "WBHF": "MIL-STD-188-110C appendix D",
             "39": "MIL-STD-188-110 39-tone",
             "16": "MIL-STD-188-110 16-tone", "FSK": "FSK", "4529": "STANAG 4529"}[a.standard]
    print(f"  standard    {title} {a.mode}")
    print(f"  payload     {len(data)} bytes")
    print(f"  audio       {lo:.0f} - {hi:.0f} Hz ({hi - lo:.0f} Hz at -26 dB; "
          f"the ceiling is {MAX_BANDWIDTH_HZ})")
    print(f"  duration    {x.size / FS:.1f} s, peak {drive:.2f} of full scale, "
          f"{CARD_RATE_HZ} Hz")
    if not a.callsign:
        print("  identify    none rendered: --callsign puts it in the file")
    print(f"  wrote       {a.out}")


def _retune(x: np.ndarray, carrier: float | None, invert: bool) -> np.ndarray:
    """The recording moved so that its serial carrier is at 1800 Hz, and its
    spectrum mirrored first if it was made on the other sideband."""
    if carrier is None and not invert:
        return x
    from scipy.signal import hilbert
    an = hilbert(x)
    fc = CARRIER_HZ if carrier is None else carrier
    if invert:
        an, fc = np.conj(an), -fc
    return np.real(an * np.exp(2j * np.pi * (CARRIER_HZ - fc) * np.arange(x.size) / FS))


#: The standards ``--async`` applies to: those carrying a plain bit stream.
_ASYNC = ("110", "4285", "4529", "4539", "WBHF", "16")


def _char_format(a):
    if a.standard not in _ASYNC:
        raise SystemExit(f"--async is for {', '.join(_ASYNC)}: FSK and 39-tone frame "
                         "characters their own way")
    try:
        return charframe.CharFormat.parse(a.async_fmt)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def cmd_decode(a) -> None:
    x = wav.read(a.wav, target_fs=FS)
    x = _retune(x, a.carrier, a.invert)
    if a.standard == "MOD":
        _decode_mod(a, x)
        return
    out = []
    pos = 0
    while True:
        if a.standard in ("4285", "4529", "16", "FSK") and not a.mode:
            raise SystemExit(f"{a.standard} carries no mode on the air: give --mode")
        if a.standard == "39" and a.late:
            if not a.mode:
                raise SystemExit("late entry needs the mode: give --mode")
            r = tone39.late_entry(x, a.mode, start=pos)
        else:
            kw = {"zero_interleave": a.zero} if a.standard == "110" else {}
            r = Mode.receive(a.standard, x, a.mode, start=pos, **kw)
        if r is None:
            break
        raw = r.data
        if a.async_fmt:
            raw = charframe.Deframer(_char_format(a)).push(charframe.unpack(raw))
        data = raw.rstrip(b"\0") if (a.standard in ("4285", "4529", "16") and a.trim) \
            else raw
        t = r.start_sample / FS
        if a.standard == "39":
            state = ("EOM" if r.eom else "no EOM: signal lost") if r.mode.char else "synchronous"
            clock = f"  clock {r.clock_ppm:+.0f} ppm" if r.clock_ppm else ""
            print(f"  {t:7.2f} s  39-tone {r.mode.name}  {len(data)} bytes  {state}"
                  f"  offset {r.freq_hz:+.1f} Hz{clock}  SNR ~{r.snr_db:.0f} dB")
        elif a.standard == "4529":
            print(f"  {t:7.2f} s  4529 {r.mode.full_name}  {r.frames} frames  "
                  f"{len(data)} bytes  offset {r.freq_hz:+.1f} Hz  SNR ~{r.snr_db:.0f} dB")
        elif a.standard == "FSK":
            print(f"  {t:7.2f} s  FSK {r.mode.name}  {len(data)} bytes"
                  f"  offset {r.freq_hz:+.1f} Hz  SNR ~{r.snr_db:.0f} dB")
        elif a.standard == "16":
            print(f"  {t:7.2f} s  16-tone {r.rate} bit/s  {len(data)} bytes"
                  f"  offset {r.freq_hz:+.1f} Hz  SNR ~{r.snr_db:.0f} dB")
        elif a.standard in ("110", "4539", "WBHF"):
            state = "EOM" if r.eom else "no EOM: signal lost"
            print(f"  {t:7.2f} s  {a.standard} {r.mode.name}  {len(data)} bytes  {state}"
                  f"  offset {r.freq_hz:+.1f} Hz  SNR ~{r.snr_db:.0f} dB")
        else:
            print(f"  {t:7.2f} s  4285 {r.mode.name}  {r.frames} frames  "
                  f"{len(data)} bytes  offset {r.freq_hz:+.1f} Hz  SNR ~{r.snr_db:.0f} dB")
        out.append(data)
        if not a.all or r.end_sample <= pos:
            break
        pos = r.end_sample
    if not out:
        print("  nothing found")
        raise SystemExit(1)
    blob = b"".join(out)
    if a.output:
        Path(a.output).write_bytes(blob)
        print(f"  wrote       {a.output}")
    else:
        print(blob.decode("utf-8", errors="replace"))


def _decode_mod(a, x: np.ndarray) -> None:
    """The 188-110A mod: each burst's six columns, as hex (nothing is known
    of the payload's coding; see ``mod110``)."""
    pos, out = 0, []
    while True:
        r = mod110.receive(x, start=pos)
        if r is None:
            break
        fmt = "every bit twice, combined" if r.repeat == 2 else "every bit once"
        end = "end marker" if r.end_marker else "signal ended"
        print(f"  {r.start_sample / FS:7.2f} s  mod  {r.payload.size} bits ({fmt})  {end}"
              f"  offset {r.freq_hz:+.1f} Hz  SNR ~{r.snr_db:.0f} dB")
        out.append(r)
        if not a.all:
            break
        pos = max(r.end_sample, r.start_sample + FS)
    if not out:
        print("  nothing found")
        raise SystemExit(1)
    if a.output:
        Path(a.output).write_bytes(b"".join(b"".join(r.columns) for r in out))
        print(f"  wrote       {a.output}")
        return
    for r in out:
        print(f"  {r.start_sample / FS:7.2f} s: {r.blocks} block(s); each column, its bytes"
              " in order")
        for c, col in enumerate(r.columns):
            print(f"    column {c}")
            for i in range(0, len(col), 32):
                print("      " + col[i:i + 32].hex())
        print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="MIL-STD-188-110 serial, appendix D, 39-tone and 16-tone / FSK / STANAG 4285 / 4529 / 4539 to and "
                    "from WAV files; "
                    "never keys a radio")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("modes", help="list the modes")
    m.set_defaults(fn=cmd_modes)

    r = sub.add_parser("render", help="render a message to a WAV")
    r.add_argument("standard", choices=("110", "4285", "4529", "4539", "WBHF", "39", "16", "FSK"))
    r.add_argument("out")
    r.add_argument("--mode", required=True, help="e.g. 600L, 2400S, 600/99; see `modes`")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--input", help="a file, or - for stdin")
    r.add_argument("--callsign", help="append Morse identification")
    r.add_argument("--drive", type=float, default=None,
                   help="peak level; default the station file's tx_drive")
    r.add_argument("--async", dest="async_fmt", metavar="FORMAT",
                   help="send start-stop characters (e.g. 8N1, 7E1) as an asynchronous "
                        "DTE port would, 110B C.5.4.1.1")
    r.add_argument("--bandwidth", choices=("3000", "3200"), default="3000",
                   help="the highest audio Hz the rig transmits: 3000 trims the "
                        "top edge to fit; 3200 sends the whole pulse (default 3000)")
    r.set_defaults(fn=cmd_render)

    d = sub.add_parser("decode", help="decode a captured WAV")
    d.add_argument("standard", choices=("110", "4285", "4529", "4539", "WBHF", "39", "16", "FSK",
                                        "MOD"))
    d.add_argument("wav")
    d.add_argument("--mode", help="4285, 4529 and FSK: the mode it was sent in; 16: the rate; 39: "
                                   "the mode, if not to be found from the data")
    d.add_argument("--all", action="store_true", help="every transmission in the file")
    d.add_argument("--late", action="store_true",
                   help="39 only: join a transmission whose preamble was missed, at "
                        "its next framing insertion (needs --mode)")
    d.add_argument("--zero", action="store_true",
                   help="110 only: short preambles mean the interleaver is bypassed")
    d.add_argument("--trim", action="store_true",
                   help="4285 uncoded and 16-tone: drop trailing NUL bytes (padding)")
    d.add_argument("--output", help="write the bytes here instead of printing")
    d.add_argument("--carrier", type=float,
                   help="where the serial carrier is in the recording, Hz (default 1800); "
                        "for MOD, the header's third tone of four, fourth of seven")
    d.add_argument("--async", dest="async_fmt", metavar="FORMAT",
                   help="the sender's start-stop characters (e.g. 8N1): deframe them")
    d.add_argument("--invert", action="store_true",
                   help="the recording was made on the other sideband")
    d.set_defaults(fn=cmd_decode)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
