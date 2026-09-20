# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate the PACTOR-2 receiver front end.

Two kinds of check, kept distinct on purpose:
  * SYNTHETIC -- exercises the DSP *code* (carrier lock, modulation-order
    detector, CS codeword hunt) against signals we built, so a null result on
    real audio is a true null, not a broken detector.
  * REAL -- regression-locks the waveform *facts* measured off real off-air
    PACTOR-2 (sigidwiki), the only ground truth for the physical layer we have
    until a full P2 session capture exists that an independent decoder will
    acquire.
"""
import os
from pathlib import Path

import numpy as np


from hfmodem.shrike import p2rx, spec

ARCHIVE = Path(__file__).resolve().parents[5] / "working" / "pactor"
FS = 48000
PASS = 0
FAILURES: list[str] = []


def check(name, ok, detail=""):
    global PASS
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


def synth(bits_lo, bits_hi, order, tones=(1400.0, 1600.0), fs=FS, snr_db=25):
    """Two-tone differential-PSK at 100 Bd. bits_* are info bits per tone
    (1/sym for DBPSK, 2/sym for DQPSK)."""
    L = int(fs / p2rx.SYMBOL_RATE)
    step = 1 if order == "DBPSK" else 2
    np.zeros(0)
    np.zeros(0)

    def stream(bits, f0):
        k = step
        nsym = len(bits) // k + 1
        phase = 0.0
        phases = [0.0]
        for s in range(1, nsym):
            grp = bits[(s - 1) * k:(s - 1) * k + k]
            if len(grp) < k:
                grp = np.zeros(k, int)
            val = int("".join(map(str, grp)), 2)
            phase = (phase + val * (np.pi if step == 1 else np.pi / 2)) % (2 * np.pi)
            phases.append(phase)
        t = np.arange(nsym * L) / fs
        car = np.zeros(nsym * L)
        for s in range(nsym):
            seg = slice(s * L, (s + 1) * L)
            car[seg] = np.cos(2 * np.pi * f0 * t[seg] + phases[s])
        return car

    a = stream(bits_lo, tones[0])
    b = stream(bits_hi, tones[1])
    m = min(len(a), len(b))
    x = a[:m] + b[:m]
    x /= np.abs(x).max()
    rng = np.random.default_rng(0)
    x += rng.normal(0, 10 ** (-snr_db / 20), len(x))
    return x


def test_dbpsk_and_cs():
    # plant CS1 (index 0) as a run of DBPSK info bits on the low tone
    cs = p2rx.cs_table()[0].astype(int)
    rng = np.random.default_rng(1)
    lo = np.concatenate([rng.integers(0, 2, 40), cs, rng.integers(0, 2, 40)])
    hi = rng.integers(0, 2, len(lo))
    x = synth(lo, hi, "DBPSK")
    r = p2rx.analyze(x, FS)
    (f_lo, f_hi) = r["carriers_measured"]
    check("DBPSK carriers ~1400/1600", abs(f_lo - 1400) < 25 and abs(f_hi - 1600) < 25,
          f"got {f_lo:.1f}/{f_hi:.1f}")
    orders = [t["order"] for t in r["tones"].values()]
    check("DBPSK modulation-order detected", orders.count("DBPSK") == 2, str(orders))
    # CS hunt must find the planted CS1 at distance 0 on the low tone
    hits = r["cs_hits"]["tone_lo"]
    found = any(j == 0 and d == 0 for _, j, d, _ in hits)
    check("CS hunt finds planted CS1 (d=0)", found,
          f"best={sorted(hits, key=lambda h: h[2])[:3]}")


def test_dqpsk():
    rng = np.random.default_rng(2)
    lo = rng.integers(0, 2, 200)
    hi = rng.integers(0, 2, 200)
    x = synth(lo, hi, "DQPSK")
    r = p2rx.analyze(x, FS)
    orders = [t["order"] for t in r["tones"].values()]
    check("DQPSK modulation-order detected", orders.count("DQPSK") == 2, str(orders))


def test_real_audio():
    fec = str(ARCHIVE / "captures/PACTOR-II_FEC.wav")
    if not os.path.exists(fec):
        print(f"  skip real-audio ({fec} absent)")
        return
    audio, fs = p2rx._read_wav_mono(fec)
    r = p2rx.analyze(audio, fs)
    check("real FEC spacing ~200 Hz", 180 < r["spacing"] < 215, f"{r['spacing']:.1f}")
    orders = [t["order"] for t in r["tones"].values()]
    check("real FEC both carriers DQPSK", orders.count("DQPSK") == 2, str(orders))


REFERENCE_SOFTS = bytes.fromhex(
    "fbc1c1003f00003f02c1fec1fe00fc3fc1c13f0000fcc1053fc1c1fdfb3fc107"
    "f4c1fec100c1f5c13f063ff700fdfdf4fc07c1f4fa3dc3c107f1f307c1efc13f"
    "c13fc1e9c403edc100c5e23f3de7c1c2c6f907c83f3f3f3f033e0bd73fd0291e"
    "ecd4d2fc3117c420c3dc3f3926d7c6cdc1fd3f22fe32c2c1efdcd7d5cfd600d8"
    "c1da03c5d4ce10c1c7c1fde700c9001e")
"""Reference channel-order softs for one frame on this recording.

144 int8 values at level 0: 72 symbols on each of two carriers, interleaved
one symbol at a time. See pactor2.md section 5.12 for the comparison."""


def test_matches_an_independent_front_end():
    """Our demodulator against an independent one, on the same real audio.

    The statistic is rotation-invariant -- |sum d*s| over a 72-symbol window,
    which is the correlation of Re(d e^{i theta}) with the reference softs
    maximised over an unknown constellation rotation -- so it assumes no polarity
    or labelling convention. Every symbol start in the recording is offered on
    both carriers, and the null is the same search against shuffles of the same
    soft vector, measured here rather than taken on trust.

    It also pins the carrier assignment: the reference decoder's EVEN channel
    positions come off the UPPER carrier. pactor2.md section 5.12."""
    fec = str(ARCHIVE / "captures/PACTOR-II_FEC.wav")
    if not os.path.exists(fec):
        print(f"  skip front-end comparison ({fec} absent)")
        return
    audio, fs = p2rx._read_wav_mono(fec)
    f_lo, f_hi = p2rx.measure_carriers(audio, fs)
    streams = {}
    for name, f0 in (("lo", f_lo), ("hi", f_hi)):
        period, phase = p2rx.measure_symbol_period(audio, fs, f0)
        sym = p2rx.symbol_stream(audio, fs, f0, period, phase)
        d = sym[1:] * np.conj(sym[:-1])
        streams[name] = (d / (np.abs(d) + 1e-30), period, phase)

    softs = np.frombuffer(REFERENCE_SOFTS, dtype=np.int8).astype(float)
    rng = np.random.default_rng(512)

    def scan(d, s):
        c = np.abs(np.correlate(d, s, "valid")) / (np.sqrt(s.size) * np.linalg.norm(s))
        return float(c.max()), int(np.argmax(c))

    when = {}
    for lane, want in (("even", "hi"), ("odd", "lo")):
        s = softs[0 if lane == "even" else 1::2]
        best = {n: scan(d, s) for n, (d, _, _) in streams.items()}
        other = "lo" if want == "hi" else "hi"
        null = max(scan(streams[n][0], rng.permutation(s))[0]
                   for n in streams for _ in range(20))
        d, period, phase = streams[want]
        when[lane] = (phase + best[want][1] * period) / fs
        check(f"{lane} channel positions correlate with the {want} carrier",
              best[want][0] > null > best[other][0],
              f"{want} {best[want][0]:.3f}, {other} {best[other][0]:.3f}, "
              f"null {null:.3f}")
    check("both lanes land on the same frame",
          abs(when["even"] - when["odd"]) * spec.SYMBOL_RATE_BD < 1.0,
          f"{when['even']:.4f}s vs {when['odd']:.4f}s")


def test_frame_markers_on_real_audio():
    """Acquisition, against real off-air PACTOR-2 and an outside verdict on it.

    This is the one check here that is not self-referential in either direction:
    the marker times, the pair bin and the codeword index all come out of an
    INDEPENDENT decoder's own correlator on this same recording, and our
    correlator has to reproduce them without having been shown them.

    That decoder reports three frames on this clip, at 2.52, 6.11 and 9.71 s,
    carrying codeword indices 1, 0 and 1 -- and the two gaps are 3.592 s, which is
    the 362-symbol long cycle measured a completely different way in
    pactor2.md section 5.9.
    """
    fec = str(ARCHIVE / "captures/PACTOR-II_FEC.wav")
    if not os.path.exists(fec):
        print(f"  skip marker acquisition ({fec} absent)")
        return
    audio, fs = p2rx._read_wav_mono(fec)
    hits = [h for h in p2rx.find_markers(audio, fs, threshold=0.95)]
    check("three frame markers found on real audio", len(hits) == 3,
          f"{len(hits)}: {[(round(h[0], 3), h[2]) for h in hits]}")
    if len(hits) != 3:
        return
    times = [h[0] for h in hits]
    codes = [h[2] for h in hits]
    check("marker times match the reference detector",
          all(abs(t - w) < 0.05 for t, w in zip(times, (2.485, 6.077, 9.670))),
          str([round(t, 3) for t in times]))
    check("codeword indices match the reference detector", codes == [1, 0, 1],
          str(codes))
    check("marker spacing is the 362-symbol long cycle",
          all(abs(b - a - 3.592) < 0.02 for a, b in zip(times, times[1:])),
          str([round(b - a, 3) for a, b in zip(times, times[1:])]))
    check("markers score near the noiseless ceiling",
          all(h[4] > 0.95 for h in hits), str([round(h[4], 3) for h in hits]))


def test_marker_round_trip_and_null():
    """Our own marker through our own correlator, and what noise scores.

    A round trip alone proves nothing -- the carrier-to-codebook assignment was
    wrong in transmitter and correlator together for a while, and this check
    passed throughout. It earns its place only next to the real-audio check
    above, which that fault failed.
    """
    from hfmodem.shrike import pactor2

    for k in (0, 6, 15):
        sig = pactor2.frame_marker(k)
        audio = np.concatenate([np.zeros(FS // 10), sig, np.zeros(FS // 10)])
        hits = p2rx.find_markers(audio, FS, threshold=0.95)
        check(f"own marker k={k} recovered", len(hits) == 1 and hits[0][2] == k,
              str([(round(h[0], 3), h[2], round(h[4], 3)) for h in hits]))

    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.1, FS * 12)
    false = p2rx.find_markers(noise, FS, threshold=0.95)
    check("noise raises no marker at the arming threshold", not false,
          f"{len(false)} in 12 s")

    k = pactor2.marker_index(level=0)
    bits = rng.integers(0, 2, 144).astype(np.uint8)
    burst = pactor2.frame(bits[:72], bits[72:], k)
    hits = p2rx.find_markers(np.concatenate([np.zeros(FS // 10), burst]), FS,
                             threshold=0.95)
    check("a whole frame is acquired at its marker",
          len(hits) == 1 and hits[0][2] == k and hits[0][0] < 0.25,
          str([(round(h[0], 3), h[2], round(h[4], 3)) for h in hits]))


def test_control_signals_round_trip():
    """Our own codewords through the anchored reader, and what it refuses.

    A round trip cannot say the keying is the one SCS transmits -- nothing in the
    corpus carries a PACTOR-2 control signal, and `pactor2.control_signal` says
    what it is a hypothesis about. What it does say is that the renderer and the
    reader agree end to end at the instant the cycle grid predicts, in both
    arrangements and at both sample rates, and that the accept is narrow enough
    to be worth having: at zero bit errors nothing else in the protocol reaches
    it.
    """
    from hfmodem.shrike import pactor2

    clean = 0
    for index in range(len(spec.CONTROL_SIGNALS)):
        for swapped in (False, True):
            for fs in (48000, 12000):
                at = int(0.3 * fs)
                audio = np.zeros(fs)
                sig = pactor2.control_signal(index, swapped=swapped, fs=fs)
                start = at - pactor2.pulse_lead(fs)
                audio[start:start + sig.size] += sig
                clean += p2rx.control_signal_at(
                    audio, at, fs, swapped=swapped) == (index, 0)
    check("every codeword reads at zero bit errors, both arrangements, both rates",
          clean == 24, f"{clean} of 24")

    rng = np.random.default_rng(5)
    false = sum(p2rx.control_signal_at(rng.normal(0, 0.2, FS // 2), FS // 4, FS)
                is not None for _ in range(400))
    check("noise at the anchor is refused", false == 0, f"{false} of 400 windows")

    path = pactor2.PATHS[2]
    over = 0
    for seed in range(40):
        field = pactor2.build_field(bytes(np.random.default_rng(seed).integers(
            0, 256, path.crc_bytes - 2, dtype=np.uint8)), path)
        audio = np.concatenate([np.zeros(FS // 4),
                                pactor2.data_burst(field, path, fs=FS),
                                np.zeros(FS // 4)])
        over += p2rx.control_signal_at(
            audio, FS // 4 + pactor2.pulse_lead(FS), FS) is not None
    check("a data field at the anchor is refused", over == 0, f"{over} of 40 bursts")


def test_long_cycle():
    """Data mode: 320-pulse frames on the 3.75 s grid, acquired and decoded.

    [SCS] s2 -- the mode a mailbox listing arrives in, and the one this receiver
    could not reach until 2026-09-02: `burst_grid` filtered every long-frame
    marker out and `decode_bursts` only ever built a short path, so a peer that
    set the long-cycle flag went silent as far as this station was concerned.

    The window the SINGLE-CYCLE path needs is measured here rather than assumed,
    because it is the number a session has to hand it: the long burst is 3.3325 s
    of audio and nothing decodes until the whole of it is in the buffer.
    """
    from hfmodem.shrike import pactor2

    rng = np.random.default_rng(11)
    for level in range(3):
        path = pactor2.PATHS_LONG[level]
        fields = [pactor2.build_field(bytes(rng.integers(
            0, 256, path.crc_bytes - 2, dtype=np.uint8)), path) for _ in range(3)]
        grid = [2.0 + k * p2rx.CYCLE_LONG_S for k in range(3)]
        audio = np.zeros(int((grid[-1] + 5.0) * FS))
        for k, (t, field) in enumerate(zip(grid, fields)):
            burst = pactor2.data_burst(field, path, swapped=bool(k & 1), fs=FS)
            at = int(round(t * FS)) - pactor2.pulse_lead(FS)
            audio[at:at + burst.size] += burst
        got = [f for _t, f in p2rx.decode_bursts(audio, FS, level, long_frame=True)]
        check(f"{path.name}: three data-mode bursts decode byte-exact",
              got == fields, f"{len(got)} of 3")
        check(f"{path.name}: the short-cycle grid finds nothing there",
              not p2rx.decode_bursts(audio, FS, level), "a short path decoded")

    path = pactor2.PATHS_LONG[2]
    field = pactor2.build_field(bytes(rng.integers(
        0, 256, path.crc_bytes - 2, dtype=np.uint8)), path)
    burst = pactor2.data_burst(field, path, fs=FS)
    audio = np.zeros(int(6.0 * FS))
    audio[FS // 5:FS // 5 + burst.size] += burst
    short = p2rx.decode_expected_burst(audio[:FS // 5 + int(3.20 * FS)], FS)
    full = p2rx.decode_expected_burst(audio[:FS // 5 + int(3.40 * FS)], FS)
    check("a single long cycle decodes once the whole burst is in the window",
          full is not None and full[2] == field and full[1] is path,
          f"{full}")
    check("...and not before it: 3.20 s of a 3.33 s burst returns nothing",
          short is None, f"{short}")


def test_verdict() -> None:
    """Every check above only accumulates, so the verdict is drawn here, last."""
    print(f"\n{PASS} passed, {len(FAILURES)} failed")
    assert not FAILURES, (f"{len(FAILURES)} of {PASS + len(FAILURES)} FAILED: "
                          + "; ".join(FAILURES))
