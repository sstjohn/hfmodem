# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Assemble a full synthetic PACTOR-3 session: connect -> data packets.

A PACTOR decoder is stateful: it takes the PACTOR-1 FSK connect (shrike.pactor1)
as the start of a session, then tracks the 1.25 s ARQ cycle and decodes the data
packets that follow. A monitor follows a LINK rather than sweeping for packets --
offered a single packet, or packets off the raster, it reports nothing -- so the
session has to be laid out on a continuous cycle grid: connect, then one data
packet per cycle with only the natural inter-cycle spacing between them. This is
the shape every PACTOR-III packet judged from outside has been offered in.

This module owns SESSION SEQUENCING/TIMING only; the connect waveform comes from
pactor1.connect_signal and each P3 packet from placement.data_packet.

A real PACTOR-III session opens its first data burst ~0.16 s after the connect
ends, one burst per 1.25 s cycle from there; `connect_to_data_s` and `cycle_s`
default into that layout.
"""
from __future__ import annotations

import wave

import numpy as np
from scipy.io import wavfile

from . import modem, pactor1, placement, spec

FS = spec.SAMPLE_RATE


def occupied_hz(*, sl: int, data: bool) -> tuple[float, float]:
    """The audio passband a `build_session` burst really occupies.

    Computed from the waveform rather than quoted from the designator, because
    what goes out is not one waveform: the connect is PACTOR-1 FSK at 1400/1600
    Hz and the data packets are PACTOR-3 on whatever channels the speed level
    uses -- 1080 and 1920 Hz at SL1, the whole 480..2520 Hz plan at SL6. A
    session declared at 2K20 when it emits an SL1 connect over-declares by more
    than a kilohertz, and at a segment edge an over-declaration is refused where
    the real emission fits.

    Each carrier is charged its own symbol rate either side, which is where the
    main lobes end; the connect's is the faster of its two sections.
    """
    lo = pactor1.MARK - pactor1.BAUD_RED
    hi = pactor1.SPACE + pactor1.BAUD_RED
    if data:
        channels = spec.SPEED_LEVELS[sl].channels
        lo = min(lo, spec.channel_freq_hz(min(channels)) - spec.SYMBOL_RATE_BD)
        hi = max(hi, spec.channel_freq_hz(max(channels)) + spec.SYMBOL_RATE_BD)
    return lo, hi


def build_session(callsign: str, payloads: list[bytes], *, sl: int = 2,
                  cycle_s: float = spec.CYCLE_SHORT_S,
                  connect_to_data_s: float = 0.20,
                  acquire: bool = False,
                  cfg: modem.ModConfig | None = None) -> np.ndarray:
    """connect(callsign) + one P3 data packet per ARQ cycle -> mono float audio.

    payloads: one ASCII payload per data cycle. cycle_s: packet-to-packet spacing
    on the ARQ grid. connect_to_data_s: gap from the connect burst's end to the
    first packet. `acquire` puts the +-800 burst in front of that first packet
    and no other: acquisition happens once per link, and every cycle after it is
    an established one.

    The carrier swap alternates cycle by cycle, which is what the specification
    asks of a transmitter and what a real link does, so no two consecutive packets
    put a virtual carrier on the same tone.
    """
    cfg = cfg or modem.ModConfig()
    cyc = int(round(cycle_s * FS))

    path = placement.SPEED_PATHS[sl]
    n_info = path.crc_bytes - 2
    connect = pactor1.connect_signal(callsign)
    out = list(connect)

    anchor = len(out) + int(round(connect_to_data_s * FS))
    for i, pl in enumerate(payloads):
        info = pl[:n_info].ljust(n_info, b"\x00")
        kw = dict(cfg=cfg, swapped=bool(i & 1), acquire=acquire and i == 0)
        pkt = (placement.case0_packet(info, **kw) if path.case == 0 else
               placement.data_packet(info, path, **kw))
        start = anchor + i * cyc
        if start + len(pkt) > len(out):
            out.extend([0.0] * (start + len(pkt) - len(out)))
        out[start:start + len(pkt)] = (np.asarray(out[start:start + len(pkt)])
                                       + pkt).tolist()
    # trailing silence so a decoder can finalise the last cycle
    out.extend([0.0] * int(0.5 * FS))
    return np.asarray(out, dtype=np.float32)


def load_wav(path: str, target_fs: int | None = None) -> np.ndarray:
    """Left channel of a WAV as float [-1, 1]; resampled to `target_fs` if given.

    Read through scipy rather than the `wave` module, which handles PCM only and
    raises `unknown format: 3` on WAVE_FORMAT_IEEE_FLOAT. That is not an exotic
    corner: 192 of the 196 recordings in offair/captures are float, so every entry
    point reaching audio through here -- the monitor included -- silently saw four
    files where there are 196, and the corpus's longest run of real PACTOR-1
    packets sat unread the whole time.
    """
    fs, raw = wavfile.read(path, mmap=True)
    a = np.asarray(raw)
    if a.ndim > 1:
        a = a[:, 0]
    # Integer formats carry their own full scale; float files are already [-1, 1].
    a = (a.astype(np.float64) / -np.iinfo(a.dtype).min if np.issubdtype(a.dtype, np.integer)
         else a.astype(np.float64))
    if target_fs and fs != target_fs:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(fs, target_fs)
        a = resample_poly(a, target_fs // g, fs // g)
    return a


def write_wav(path: str, mono: np.ndarray, *, amplitude: float = 0.8) -> None:
    """Write mono float [-1,1] as stereo 48 kHz int16, the capture format PACTOR
    decoders expect from a soundcard."""
    m = np.asarray(mono, dtype=np.float64)
    peak = np.max(np.abs(m)) or 1.0
    s = np.clip(m / peak * amplitude, -1, 1)
    i16 = (s * 32767).astype("<i2")
    stereo = np.column_stack([i16, i16]).ravel()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(stereo.tobytes())


if __name__ == "__main__":
    import sys
    call = sys.argv[1] if len(sys.argv) > 1 else "W1AW"
    sl = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    out = sys.argv[3] if len(sys.argv) > 3 else "captures/session_test.wav"
    msg = b"TEST DE " + call.upper().encode()
    audio = build_session(call, [msg] * 4, sl=sl)
    write_wav(out, audio)
    print(f"wrote {out}: {len(audio)/FS:.2f}s, connect({call}) + 4x SL{sl} '{msg.decode()}'")
