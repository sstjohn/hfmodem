# Taper and window, settled by rendering tape overs from their bytes

`render_vs_tape.py` prints every number below (about 4 min, nothing written). Each over is rendered
from the transcript's bytes (payload checked against the transcript; the control byte and the lead Rnd
state come from one KPA decode) and fitted back against the tape with one real amplitude and one delay.
The tapes are 16-bit PCM in float32 wavs (every sample sits on the 2^-15 grid), so a perfect render
leaves the quantisation floor, (2^-15)²/12 over the signal power: -84 to -85 dB on these overs.

## Constants

**OFDM, every bandwidth (BW500, BW2300, BW2750 alike).** Let `o` be symbol s's IFFT origin, the
first sample of the 1024-sample FFT window that reads the cells phase-exact (the KPA window, raster
start + 112). Then, with symbols 1152 samples apart:

| constant | value |
|---|---|
| symbol support | cyclic extension of the IFFT from `o − 128` to `o + 1056`: a 128-sample CP in front, a 32-sample cyclic suffix behind |
| taper | **raised cosine, 32 samples**, `w[i] = sin²(π(i + 0.5)/64)`, i = 0..31, rising over the CP's first 32 samples and falling over the suffix |
| blend | overlap-add: the suffix of symbol s and the first 32 CP samples of symbol s+1 share `[o₊ − 128, o₊ − 96)` with complementary ramps |
| ISI-free CP | 96 samples, `[o − 96, o)` |
| burst onset | the first symbol's own ramp: the burst starts at `o₀ − 128` and rises over 32 samples |
| burst tail | the burst ends at `o_last + 1024`; its last 32 samples fall by the same ramp, which lands inside the last symbol's FFT window. There is no suffix on the last symbol |
| burst length | exactly `nsym × 1152` samples (176256 at 153 symbols, 183168 at 159) |

Residual of the full render, onset and tail included, against the tape: **−84.1 dB (BW2300 L10),
−84.1 (BW2300 L16), −84.9 (BW500 L8), −85.2 (BW500 L13), −84.2 (BW2750 L11), −84.2 (BW2750 L17)**,
each equal to its own 16-bit floor. The fitted delay is 0.0000 samples on all six. The model is exact.

**The 16-sample question: a ramp, not a CP position.** The CP is a plain 128-sample prefix in front of
the origin. The raster's grid start (the CP-metric fold) sits at `o − 112`, the midpoint of the
32-sample blend, 16 samples after the CP really starts. So "window 16 samples inside the CP's end"
is half the blend. RASTER's per-bandwidth blends of ~10 / 16 / 29 samples and "99 of 128 CP samples
ISI-free" were measurement-method estimates of this one 32-sample blend and its 96 clean samples.
The same rule explains why the KPA saw only the burst's last symbol deviate: its fade-out sits inside
its FFT window.

**Tone family (BW2300 L5-L9).** The sub-band model renders to **−64.6 dB at L5** (BW500's 513-tap
pulse, `tx/bw500_pulse.PULSE`) and **−73.4 dB at L9** (the 2049-tap L9 pulse fitted in
`kpa_bw2300_subband`), but only once each band carries its own small carrier offset. At the nominal
centres the same render is −50.7 / −52.1 dB. The offsets are ±0.01 to ±0.7 mHz (relative 1e-8 to
5e-7, sign varying by band, not proportional to frequency). They reproduce to ±0.01 mHz across five
clean L8 overs from four sessions, so they are a property of VARA's oscillator for each centre and
not of the channel. That is consistent with a single-precision phase accumulator, which is a
hypothesis, not measured. No receiver can see this: 0.7 mHz over a 4 s over is 0.7°. It matters
only to a sample-exact render gate, which has to include the offsets or accept −50 dB. Every
tone-family residual here uses the tape's own lead cells (t 0..7, least squares), because the lead's
generator is unknown. The other cells come from the bytes and the grid rows, with one complex gain
per band, which absorbs the rows' per-band phase constant.

| L | pulse | nominal centres | with offsets | free-cell LS (the KPA figure) | floor | offsets, mHz, band 0.. |
|---|---|---|---|---|---|---|
| 5 | BW500 513 taps | −50.7 dB | **−64.6** | −64.6 | −87.2 | 0.273, 0.341, 0.529 |
| 9 | L9 fit, 2049 taps | −52.1 | **−73.4** | −77.0 | −85.0 | 0.254, −0.329, 0.387, −0.324, 0.393, −0.18, 0.52, 0.01, −0.696, 0.134 |

The KPA's −64.6 / −77 dB were free-cell residuals. At L5 the free cells absorb nothing but the
offsets. At L9 they also absorb 3.6 dB of pulse error. Refitting the pulse with every cell anchored
gains nothing (−50.9 / −52.1 dB at the nominal centres), so the pulse is not what limits the
nominal render.

Tone onset and tail are the pulse alone. There is no extra ramp. L5's burst runs from 256 samples
before band 0's first pulse centre to 256 after the last band's last centre (196432 = 383·512 + 336,
the ±256 pulse plus the 336 stagger). The residual is flat across the burst: onset −65.4, interior
−64.6, tail −65.1 dB. L9 runs from 527 before to 513 after (183056 = 356·512 + 784): onset −78.0,
interior −73.4, tail −75.3 dB.

## For the transmitter

OFDM: IFFT, take `[−128, 1056)` cyclically, multiply both 32-sample ends by `sin²(π(i + 0.5)/64)`,
overlap-add at a 1152 stride, then truncate the burst to `nsym × 1152` samples and apply the falling
ramp to its last 32 samples. Nothing here differs by bandwidth or level. A render of this form is
sample-exact against VARA to the 16-bit floor, and the gate for OFDM levels can be that floor
(about −84 dB). The gate for tone levels is −50 dB at the nominal centres.

## Grids

Delay held at the value fitted at the winner (0.0000 on every over), amplitude free. Body = the burst
less 1408 samples at each end, so edges are scored separately. `pre` = samples from the nominal
boundary (blend midpoint) to the IFFT origin. `in` shapes ramp inside the 1152-sample symbol with no
overlap; `ola` shapes extend L/2 past each boundary and overlap-add. Grid: shape × L 0..40 × pre
96..144, 8036 cells per over.

| over | best | body | runner-up | best lin-in | best rc-in | best lin-ola |
|---|---|---|---|---|---|---|
| BW2300 L10 @1138938 | rc-ola 32 / 112 | −84.1 | rc-ola 34/112 −50.1 | −26.2 (6/112) | −25.6 (5/112) | −46.4 (24/112) |
| BW2300 L16 @3044455 | rc-ola 32 / 112 | −84.2 | rc-ola 34/112 −50.8 | −26.6 (6/112) | −26.0 (5/112) | −46.8 (24/112) |
| BW500 L8 @2181689 | rc-ola 32 / 112 | −84.9 | rc-ola 34/112 −51.0 | −26.5 (5/112) | −26.0 (4/112) | −47.1 (24/112) |
| BW500 L13 @2813230 | rc-ola 32 / 112 | −85.2 | rc-ola 34/112 −50.5 | −25.8 (5/113) | −25.5 (1/112) | −46.6 (24/112) |
| BW2750 L11 @1117460 | rc-ola 32 / 112 | −84.2 | rc-ola 34/112 −50.7 | −26.7 (6/113) | −26.1 (6/113) | −46.7 (24/112) |
| BW2750 L17 @3047432 | rc-ola 32 / 112 | −84.2 | rc-ola 34/112 −51.0 | −25.9 (4/111) | −25.6 (1/112) | −47.1 (24/112) |

rc-ola along L at pre 112 (BW2300 L10; the others agree within 1 dB):

| L | 0 | 8 | 16 | 24 | 28 | 30 | 31 | **32** | 33 | 34 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dB | −25 | −27 | −31 | −37 | −44 | −50 | −47 | **−84** | −47 | −50 | −44 | −39 |

rc-ola along pre at L 32 (BW2300 L10):

| pre | 96 | 104 | 108 | 110 | 111 | **112** | 113 | 114 | 116 | 120 | 128 | 144 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dB | −18 | −23 | −29 | −35 | −41 | **−84** | −41 | −35 | −29 | −24 | −18 | −13 |

A plain front CP with the boundary at the origin − 128 (pre 128) scores −18 dB. The ramp's sample
phase matters too: `sin²(πi/64)` or `sin²(π(i + 1)/64)` in place of the half-sample form scores −48 dB.

Burst edges, fitted over the first / last 1408 + 640 samples with the interior fixed at rc-ola
32/112 (ramp shape lin/rc, length 0..256, zero-end offset −64..160 from the first boundary, or from
the last boundary + 1152):

| over | head | tail | whole burst | edges left as interior OLA |
|---|---|---|---|---|
| BW2300 L10 | rc 32 at −16: −83.6 | rc 32 at −16: −84.4 | −84.1 | −39.1 |
| BW2300 L16 | rc 32 at −16: −84.1 | rc 32 at −16: −83.8 | −84.1 | −38.8 |
| BW500 L8 | rc 32 at −16: −84.1 | rc 32 at −16: −84.3 | −84.9 | −50.5 |
| BW500 L13 | rc 32 at −16: −84.9 | rc 32 at −16: −84.7 | −85.2 | −43.3 |
| BW2750 L11 | rc 32 at −16: −84.2 | rc 32 at −16: −83.7 | −84.2 | −44.0 |
| BW2750 L17 | rc 32 at −16: −84.0 | rc 32 at −16: −84.0 | −84.2 | −41.2 |

The nearest coarse neighbours (28 or 36 samples) sit at −47 to −64 dB. The tape's first nonzero
sample is 16 before the first boundary (`o₀ − 128`) and its last is at `o_last + 1024`, on every
over (±1 sample, where an end sample quantises to 0).
