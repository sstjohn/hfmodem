# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""sabir off-air: render a beacon to a WAV to transmit, and decode a captured
WAV back. The one-way validation of the TX -> SSB -> WebSDR/KiwiSDR -> decode
chain, before a second station exists.

The receive side is solved by the shared ``../offair/`` KiwiSDR tool (it emits
48 kHz mono float WAVs -- exactly sabir's audio format). This adds the two
sabir-specific ends: rendering a transmittable beacon, and decoding a capture.

Because it is one-way (you transmit, a remote WebSDR hears you, you decode the
recording -- you cannot answer back through a WebSDR), the test signal is a
**beacon**: connectionless, self-identifying, no ARQ round trip. Two flavours:

- ``presence`` -- the M9 presence beacon on the floor MFSK waveform, carrying
  the station id + advertised capability image; decodes at the floor's ~-15 dB.
- ``wspr`` -- the M6b narrow-tone weak-signal beacon (callsign + grid), for a
  marginal path; decodes far below the noise (-22..-30 dB by gear).

Frequency convention matches the shared tool: sabir sits at 1500 Hz in the USB
audio passband, so **USB dial = intended RF centre - 1500 Hz**.

    python offair.py presence  beacon.wav --callsign W1AW
    python offair.py wspr       wspr.wav   --callsign W1AW --grid FN31 --gear beacon_deep
    python offair.py decode     capture.wav

Nothing here keys anything, and that is the point of the name. This module once
carried a ``transmit`` verb that opened a rig by serial port, keyed it over CAT
and played a WAV -- no arm gate, no keying line, no dial readback and no listen
before it, beside a decoder whose whole job is to be run at a desk. The
transmission belongs to `sabir.onair`, which has all four.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from hfmodem.sabir.arq import wire
from hfmodem.sabir.floor import beacon as wspr
from hfmodem.sabir.floor.mfsk import FloorModem

from hfmodem.sabir.phy.modem import analytic
from hfmodem.sabir.phy.rate import FS                         # sabir's audio rate
_WSPR_GEARS = ("beacon_short", "beacon_med", "beacon_deep", "beacon_deep2")


# -- WAV + audio-domain conversion --------------------------------------------
def wav_write(path: str, x: np.ndarray, headroom: float = 0.9) -> None:
    """48 kHz mono float32, peak-normalised -- the offair corpus format."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.abs(x).max()) or 1.0
    wavfile.write(path, FS, (headroom * x / peak).astype(np.float32))


def wav_read(path: str) -> np.ndarray:
    """Any WAV -> real 48 kHz float. Handles int16/float and resamples a Kiwi's
    ~8-12 kHz rate up to 48 kHz. A Kiwi's ~90 ppm sample-clock offset is a ~2%
    symbol slip over a burst -- tolerated because the beacon bursts are short,
    not because the floor/beacon receivers track sample-clock (they don't; only
    the OFDM sync does)."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    x = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.unsignedinteger):
        # 8-bit WAV is unsigned with the midpoint as silence, so scaling
        # without removing it leaves a ~0.5 DC pedestal. The floor decoder
        # survives that -- its tones are at 1359-1641 Hz and detection is
        # noncoherent -- but `_snr_estimate` squares the samples, so the
        # pedestal dominates and the reported SNR is meaningless.
        info = np.iinfo(data.dtype)
        x = (x - (info.max + 1) / 2) / ((info.max + 1) / 2)
    elif np.issubdtype(data.dtype, np.integer):
        x /= np.iinfo(data.dtype).max
    if rate != FS:
        x = resample_poly(x, FS, rate)
    return x


def to_real(analytic: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0) * np.asarray(analytic).real


def to_analytic(real: np.ndarray) -> np.ndarray:
    """Real capture -> analytic signal. See `phy.modem.analytic` for the edge
    treatment; this is the same conversion the PHY ingress uses, deliberately."""
    return analytic(real)


def _lead_silence(burst: np.ndarray, seconds: float = 0.5) -> np.ndarray:
    pad = np.zeros(int(seconds * FS))
    return np.concatenate([pad, burst, pad])


# -- render -------------------------------------------------------------------
#
# One renderer per beacon, returning audio rather than writing a file, because
# `sabir.onair` needs the samples and a WAV is only what this CLI does with them.

def render_presence(callsign: str, profile: int = 1) -> np.ndarray:
    cap0 = wire.capabilities(range(3, 7 + 1), wire.FASTCTL | wire.PBACK | wire.LOADING | wire.DEFLATE)
    bc = wire.Beacon.build(cap0, callsign.upper(), profile=profile)
    return _lead_silence(to_real(FloorModem().transmit(bc.pack())))


def render_wspr(callsign: str, grid: str, gear: str,
                status: int = 0) -> np.ndarray:
    pl = wspr.BeaconPayload(callsign=callsign.upper(), grid=grid.upper(),
                            status=status)
    return _lead_silence(to_real(wspr.send_beacon(pl, wspr.BEACON_GEARS[gear])))


def cmd_presence(a: argparse.Namespace) -> None:
    audio = render_presence(a.callsign, a.profile)
    wav_write(a.out, audio)
    print(f"presence beacon '{a.callsign.upper()}' -> {a.out} "
          f"({audio.size / FS:.1f} s on the floor waveform, ~500 Hz occupied)")


def cmd_wspr(a: argparse.Namespace) -> None:
    from hfmodem.sabir.frame.reporting import beacon_power_status
    status = beacon_power_status(a.power_dbm) if a.power_dbm is not None else a.status
    audio = render_wspr(a.callsign, a.grid, a.gear, status)
    wav_write(a.out, audio)
    print(f"WSPR beacon '{a.callsign.upper()}/{a.grid.upper()}' gear {a.gear} "
          f"-> {a.out} ({audio.size / FS:.1f} s)")


# -- decode -------------------------------------------------------------------
def _snr_estimate(real: np.ndarray) -> float:
    """Rough peak-window vs quiet-window level ratio in dB (broadband, 0.2 s
    windows) -- an operator hint for a real capture where the burst is a small
    part of a longer recording, NOT a calibrated SNR and meaningless on a
    capture that is mostly signal. Cumsum sliding window, O(N)."""
    p = real ** 2
    win = int(0.2 * FS)
    if p.size < 3 * win:
        return float("nan")
    c = np.concatenate([[0.0], np.cumsum(p)])
    energy = (c[win:] - c[:-win]) / win               # sliding-window mean
    sig, noise = energy.max(), np.percentile(energy, 10)
    return 10.0 * np.log10(sig / (noise + 1e-12))


def cmd_decode(a: argparse.Namespace) -> None:
    real = wav_read(a.wav)
    z = to_analytic(real)
    print(f"capture: {real.size / FS:.1f} s, ~{_snr_estimate(real):.0f} dB peak SNR")

    raw, st = FloorModem().receive(z, wire.CONNECTIONLESS_BYTES)
    blk = wire.Control.unpack(raw) if raw is not None else None
    if isinstance(blk, wire.Beacon):
        f = wire.decode_capabilities(blk.capability_word)
        print(f"  DECODED presence beacon: station '{blk.call}', "
              f"profile {blk.profile}, profiles {f['profiles']}, "
              f"features {[k for k, v in f.items() if v is True]}")
        return
    if blk is not None:
        print(f"  decoded a floor block, type {blk.type} (not a beacon)")

    # Deepest first: a `beacon_deep2` burst opens with a bit-exact `beacon_deep`
    # burst, so trying deep first always decodes it and always names it deep.
    # The outermost rung that fits is the rung that was actually transmitted.
    for g in reversed(_WSPR_GEARS):
        pl, _ = wspr.recv_beacon(z, wspr.BEACON_GEARS[g])
        if pl is not None:
            print(f"  DECODED WSPR beacon ({g}): {pl.callsign} {pl.grid} "
                  f"status {pl.status}")
            return

    print("  no sabir beacon decoded (check dial = centre - 1500 Hz and "
          "ADPCM/AGC off on the Kiwi; a wspr beacon below the noise decodes "
          "here but is not energy-visible on the waterfall)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sabir off-air beacon TX/decode")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("presence", help="render the M9 presence beacon to a WAV")
    p.add_argument("out")
    p.add_argument("--callsign", required=True)
    p.add_argument("--profile", type=int, default=1)
    p.set_defaults(fn=cmd_presence)

    w = sub.add_parser("wspr", help="render the M6b weak-signal beacon to a WAV")
    w.add_argument("out")
    w.add_argument("--callsign", required=True)
    w.add_argument("--grid", required=True)
    w.add_argument("--gear", choices=_WSPR_GEARS, default="beacon_deep")
    status = w.add_mutually_exclusive_group()
    status.add_argument("--status", type=int, default=0)
    status.add_argument("--power-dbm", type=int, help="propagation-v1 compact power (-30..60 dBm)")
    w.set_defaults(fn=cmd_wspr)

    d = sub.add_parser("decode", help="decode a captured WAV")
    d.add_argument("wav")
    d.set_defaults(fn=cmd_decode)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
