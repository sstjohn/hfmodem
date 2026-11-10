# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Bit error rates through ``bateleur.sim``, for the bateleur waveforms.

    python -m hfmodem.bateleur.measure 4285 600L poor 10 --output result.json
    python -m hfmodem.bateleur.measure 39tone 1200/567/S 2ms:2hz 20 --trials 8 --output result.json
    python -m hfmodem.bateleur.measure 4539 9600L flat 30 --seconds 20 --output result.json

The channel is a ``sim`` preset, or ``<delay>ms:<spread>hz`` for two equal
Rayleigh paths (the channel every standard's table sweeps), or ``1p:<spread>hz``
for one, or ``static:<delay>,<delay>,...ms`` for equal fixed paths
(appendix D table D-LIII). The SNR is in 3 kHz; ``inf`` is none. Each trial sends random data
for ``--seconds`` of payload time, on its own fading and noise, with a carrier
offset; the rate is the bits wrong over the bits sent, a message lost
counting every missing bit as wrong.
"""
from __future__ import annotations

import argparse
import time
import hashlib
import json
import sys
from pathlib import Path

import scipy
from scipy.signal import welch
from scipy.stats import beta

import numpy as np

from . import sim, s4539
from .spec import FS
from .modem import Mode


def _waveform(name: str):
    if name.lower() == "wbhf":
        from . import wbhf
        return (lambda m: wbhf.MODES_WBHF[m].w.rate,
                lambda d, m: wbhf.transmit(d, m, superframes=20),
                lambda a, m: wbhf.receive(a))
    standard = "39" if name == "39tone" else name
    return (lambda m: Mode.parse(standard, m).rate,
            lambda d, m: Mode.parse(standard, m).render(d),
            lambda a, m: Mode.receive(standard, a, m,
                **({"zero_interleave": m.endswith("Z")} if standard == "110" else {})))


def channel_paths(spec: str):
    """A channel name -> (preset, paths) for ``sim.channel``."""
    s = spec.lower()
    if s.startswith("static:"):
        delays = [float(d) for d in s[7:].rstrip("ms").split(",")]
        return "awgn", tuple((d / 1000, 0.0) for d in delays)
    if s.startswith("1p:"):
        return "awgn", ((0.0, float(s[3:].rstrip("hz"))),)
    if "ms:" in s:
        d, f = s.split("ms:")
        return "awgn", sim.two_paths(float(d), float(f.rstrip("hz")))
    if s not in sim.PRESETS:
        raise ValueError(f"no channel {spec!r}: a preset ({', '.join(sim.PRESETS)}), "
                         "<delay>ms:<spread>hz, 1p:<spread>hz or static:<delay>,...ms")
    return s, None


def error_count(sent: bytes, received: bytes) -> tuple[int, int]:
    """Wrong bits, counting every missing bit as wrong; and missing bits."""
    n = min(len(sent), len(received))
    xor = np.frombuffer(sent[:n], dtype=np.uint8) ^ np.frombuffer(received[:n], dtype=np.uint8)
    missing = 8 * (len(sent) - n)
    return int(np.unpackbits(xor).sum()) + missing, missing


def upper95(errors, bits):
    """One-sided binomial upper bound, conditional on independent bit errors."""
    return float(beta.ppf(0.95, errors + 1, bits - errors)) if errors < bits else 1.0


def spectrum(audio, fs=FS, rbw=30.0):
    """-26 dB from peak Welch PSD, Hann window, fixed equivalent noise bandwidth."""
    n = min(len(audio), round(1.5 * fs / rbw))
    f, power = welch(audio, fs=fs, window="hann", nperseg=n, noverlap=n // 2,
                     nfft=max(16384, n), detrend="constant", scaling="density")
    occupied = f[power >= power.max() * 10 ** (-26 / 10)]
    return float(occupied[0]), float(occupied[-1])


def _one(args):
    wave, mode, chan, snr, seconds, seed, freq_hz, sweep, clock_ppm = args
    rate, tx, rx = _waveform(wave)
    nbytes = max(16, int(rate(mode) * seconds / 8))
    msg = np.random.default_rng(1000 + seed).integers(0, 256, nbytes, dtype=np.uint8).tobytes()
    transmitted = tx(msg, mode)
    audio = np.r_[np.zeros(FS // 2), transmitted, np.zeros(FS // 2)]
    preset, paths = channel_paths(chan)
    offset = freq_hz
    if sweep:
        offset = sim.sweep(audio.size, rate_hz_s=abs(sweep), limit_hz=75) * np.sign(sweep)
    y = sim.channel(audio, snr_db=snr, preset=preset, paths=paths, freq_hz=offset,
                    clock_ppm=clock_ppm, seed=seed)
    begin = time.monotonic()
    got = rx(y, mode)
    elapsed = time.monotonic() - begin
    received = got.data if got else b""
    errors, missing = error_count(msg, received)
    return dict(seed=seed, message_seed=1000 + seed, bits=8 * nbytes, errors=errors,
                payload_seconds=8 * nbytes / rate(mode), transmit_samples=transmitted.size,
                channel_samples=y.size, sample_rate=FS,
                missing_bits=missing, received_bits=8 * len(received), acquired=got is not None,
                eom=bool(getattr(got, "eom", False)),
                realtime_factor=elapsed / (y.size / FS))


def run(wave, mode, chan, snr, *, trials=8, seconds=25.0, freq_hz=23.0,
        sweep=0.0, clock_ppm=0.0):
    results = [_one((wave, mode, chan, snr, seconds, seed, freq_hz, sweep, clock_ppm))
               for seed in range(trials)]
    errors, bits = sum(r["errors"] for r in results), sum(r["bits"] for r in results)
    return dict(waveform=wave, mode=mode, channel=chan, snr_db=str(snr),
                seconds=seconds, trials=trials, offset_hz=freq_hz, sweep_hz_s=sweep,
                clock_ppm=clock_ppm, trim=True, snr_bandwidth_hz=3000,
                errors=errors, bits=bits, ber=errors / bits,
                independent_bit_upper95=upper95(errors, bits), results=results)


def measure(wave, mode, chan, snr, *, trials=8, seconds=25.0, freq_hz=23.0, workers=1):
    if workers != 1:
        raise ValueError("measurement runs use one worker")
    result = run(wave, mode, chan, snr, trials=trials, seconds=seconds, freq_hz=freq_hz)
    return (result["errors"], result["bits"], [r["errors"] for r in result["results"]],
            max(r["realtime_factor"] for r in result["results"]))


def presets(name):
    """Reconstructed table conditions; new runs record exact counts and source hashes."""
    if name == "table-xx":
        rows = [("4800N", "awgn", 17, 2.5), ("4800N", "2ms-0.5hz", 27, 2.5),
                ("2400L", "awgn", 10, 5), ("2400L", "poor", 18, 5),
                ("2400L", "5ms-1hz", 30, 5), ("1200L", "poor", 11, 10),
                ("600L", "poor", 7, 12), ("2400L", "2ms-5hz", 30, 5),
                ("300L", "5ms-5hz", 7, 16), ("150L", "5ms-5hz", 5, 16),
                ("75L", "5ms-5hz", 2, 16)]
        return [dict(wave="110", mode=m, chan=c, snr=db, seconds=t,
                     trials=16 if m == "75L" else 8) for m, c, db, t in rows]
    if name == "annex-b":
        rows = [(2400, "awgn", 12), (2400, "flat", 15), (2400, "poor", 15),
                (2400, "poor", 20), (2400, "moderate", 15), (2400, "moderate", 20),
                (2400, "1ms:4hz", float("inf")), (2400, "1ms:8hz", float("inf")),
                (2400, "5ms:0.5hz", float("inf")), (2400, "6ms:0.5hz", float("inf")),
                (1200, "awgn", 4), (1200, "flat", 10), (1200, "poor", 10),
                (1200, "moderate", 10), (1200, "1ms:8hz", float("inf")),
                (600, "awgn", 2), (600, "flat", 5), (600, "poor", 5),
                (600, "moderate", 5), (600, "6ms:0.5hz", float("inf")),
                (600, "5ms-5hz", 10)]
        return [dict(wave="4285", mode=f"{rate}L", chan=c, snr=db, seconds=25, trials=8)
                for rate, c, db in rows]
    if name == "c-xvii":
        rows = [(3200, 9, 15), (4800, 13, 20), (6400, 16, 24), (8000, 19, 28), (9600, 21, 33)]
        return [dict(wave="4539", mode=f"{rate}VL", chan=c, snr=db,
                     seconds=(2 * s4539.MODES4539[f"{rate}VL"].block_bits - 32) / rate, trials=8)
                for rate, awgn, poor in rows for c, db in (("awgn", awgn), ("poor", poor))] + [
                    dict(wave="4539", mode="12800US", chan="awgn", snr=27,
                         seconds=(60 * s4539.MODES4539["12800US"].block_bits - 32) / 12800, trials=8)]
    if name == "b-xii":
        return [dict(wave="39tone", mode=f"{rate}L", chan="2ms:2hz", snr=db, seconds=20, trials=16)
                for rate, db in ((2400, 20), (2400, 30), (1200, 15), (1200, 20),
                                  (300, 6), (300, 8), (75, 4))]
    if name == "c6-3":
        return [dict(wave="4539", mode="9600L", chan="poor", snr=30, seconds=312,
                     trials=2, freq_hz=offset) for offset in (-75, 75)]
    if name == "c6-4":
        return [dict(wave="4539", mode="9600L", chan="awgn", snr=24, seconds=96,
                     trials=2, freq_hz=0, sweep=slope) for slope in (-3.5, 3.5)]
    raise ValueError(f"no measurement preset {name!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("waveform", nargs="?")
    parser.add_argument("mode", nargs="?")
    parser.add_argument("channel", nargs="?")
    parser.add_argument("snr", nargs="*", type=float)
    parser.add_argument("--preset", choices=("table-xx", "annex-b", "c-xvii", "b-xii", "c6-3", "c6-4"))
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--offset", type=float)
    parser.add_argument("--sweep", type=float, help="signed triangular sweep rate in Hz/s, limits +/-75 Hz")
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.preset:
        jobs = presets(args.preset)
    elif args.waveform and args.mode and args.channel and args.snr:
        jobs = [dict(wave=args.waveform, mode=args.mode, chan=args.channel, snr=db) for db in args.snr]
    else:
        parser.error("give a preset or waveform, mode, channel and SNR")
    for job in jobs:
        for key, value in (("trials", args.trials), ("seconds", args.seconds),
                           ("freq_hz", args.offset), ("sweep", args.sweep)):
            if value is not None:
                job[key] = value
    if any(job.get("seconds", 25) <= 0 or job.get("trials", 8) < 1 for job in jobs):
        parser.error("seconds and trials must be positive")
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.glob("*.py"))}
    record = dict(source_sha256=sources, numpy=np.__version__, scipy=scipy.__version__,
                  arguments=list(argv) if argv is not None else sys.argv[1:],
                  error_count="wrong delivered bits plus every missing bit",
                  bound="one-sided 95% binomial; independent bits assumed; fading errors are correlated",
                  measurements=[])
    for job in jobs:
        result = run(**job)
        record["measurements"].append(result)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{result['waveform']} {result['mode']} {result['channel']}: "
              f"{result['errors']}/{result['bits']} bits, upper95 "
              f"{result['independent_bit_upper95']:.3g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
