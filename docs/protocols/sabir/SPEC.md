# SPEC.md — the sabir protocol

**Status: experimental, unpublished.** This specification defines the waveform,
FEC, framing and session behavior. [NEGOTIATION.md](NEGOTIATION.md) defines
capability advertisements; [HOST-API.md](HOST-API.md) defines the host interface.
Session controls use wire discriminator 2. The reference implementation and
known-answer tests are the executable companion to this specification.

## Conformance language

MUST / MUST NOT / SHOULD / MAY per RFC 2119. An implementation is **conformant**
if it reproduces the §KAT byte vectors, demodulates the reference waveform
(byte-exact in loopback), and honours the framing and ARQ rules below. Unknown capability bits and optional extensions are ignored; unsupported
critical extensions refuse a connection. A transmitter uses only explicitly
advertised receive profiles and mutually supported optional features
(`NEGOTIATION.md`).

## The name

Sabir was the lingua franca of the Mediterranean ports — the tongue sailors of
every nation used to find the words they shared; the word itself descends from
*saber*, "to know."

---

## 1. Overview

sabir carries a reliable byte stream between two stations over a single
nominal 2.8 kHz HF audio channel, with a tunable operating point spanning long-range/low-rate
to high-throughput. It ships **two waveform families** under one adaptive shell:

- a **pilot-aided coherent windowed-CP OFDM ladder** — the workhorse and
  throughput gears;
- a **noncoherent MFSK floor burst** — robust link control, the
  500 Hz bandwidth class, connect/ACK, and the connectionless
  beacon.

A transmission ("over") is one control burst optionally followed by an OFDM data
body the header describes. The link layer is a HARQ ARQ with carrier-group
selective acknowledgement and a transmitter-driven gearshift.

---

## 2. Numerology (the fixed grid)

All timing derives from one grid. These constants are normative **for the eight
ladder gears**; an off-ladder profile may override FFT size and CP, and two do —
`doppler` (ID 16) runs FFT 512 / CP 320 / window 32, giving 93.75 Hz spacing and
an 832-sample, 17.333 ms symbol, and `wide256` (ID 21) keeps FFT 1024 with CP 128
for a 1152-sample, 24 ms symbol (§5.1).

| Parameter | Value |
|---|---|
| Sample rate | 48 000 Hz (complex baseband) |
| FFT size | 1024 |
| Subcarrier spacing | 46.875 Hz (48000 / 1024) |
| Useful symbol | 21.333 ms (1024 samples) |
| Cyclic prefix | 256 samples, 5.333 ms |
| CP window (raised-cosine edge) | 64 samples |
| OFDM symbol total | 1280 samples, 26.667 ms (~37.5 baud) |
| Floor MFSK symbol | 1024 samples, 21.333 ms |
| Floor base frequency | 1359.375 Hz; tone grid 46.875 Hz |
| Carrier groups (selective-ACK unit) | 8 |
| Burst guard | 512 samples lead, 256 samples tail |

The standard prefix lasts 5.333 ms; its 1.333 ms overlap window leaves 4 ms
of untapered prefix for multipath and timing error. The receiver's early timing
bias consumes part of that margin. A 1 Hz Doppler spread is about 2% of the
standard carrier spacing. Windowing and post-clipping filtering reduce spectral
skirts; a carrier allocation alone does not establish occupied-power bandwidth.

All waveform families use a nominal **1500 Hz audio center**. For an even
contiguous OFDM carrier set, the center lies halfway between the middle pair.
The carrier frequencies are

```
f[r] = 1500 + (r - (n_carriers - 1)/2) * 48000/n_fft
```

The implementation uses centered integer FFT bins `-n_carriers/2` through
`n_carriers/2-1` and translates the entire windowed body by
`1500 + (48000/n_fft)/2` Hz. Translation uses a continuous sample index across
symbols and cyclic prefixes. This half-bin correction preserves carrier spacing
and makes the actual carrier midpoint exactly 1500 Hz.

### 2.1 Design tradeoffs

The 48 kHz interface fits the audio path without requiring a device-specific
sample rate. FFT 1024 gives 46.875 Hz spacing: longer symbols would spend less
on a fixed-duration multipath prefix but become more sensitive to Doppler and
frequency error. The standard prefix costs 20% of each symbol before pilots;
its overlap window and receiver timing bias reduce the usable delay margin.
The 512-point Doppler profile increases spacing and pilot update rate at the
cost of a larger prefix fraction and 50% pilot overhead. The short-prefix
256-QAM profile makes the opposite trade and requires explicit advertisement.

Carrier counts 24 and 56 divide into eight groups of three and seven carriers,
respectively, without fractional group boundaries. The narrower allocation
reduces bandwidth and the wider one increases coded-cell capacity. Both use the
same acquisition preamble, avoiding per-profile bootstrap detection. Keeping
the preamble requires a wider filter than the 24-carrier allocation alone.
The centered placement adds a half-bin frequency translation; the receiver
removes that known translation before FFT processing.

The pilot lattice trades payload cells against time and frequency sampling of
the channel. Standard even-carrier tracks repeat every 80 ms and the interleaved
tracks are 93.75 Hz apart. The sparse lattice doubles that time interval; it
assumes slower channel variation. These intervals describe sampling geometry,
not guaranteed channel limits. Pilot amplitude equals constellation RMS;
constant pilots simplify channel estimation but contribute to crest factor.

The code rates use separate QC-LDPC matrices because retransmission uses Chase
combining, not incremental parity. Quasi-cyclic structure permits shift/XOR
encoding and a shared decoder; the finite codeword sizes bound decoding work
and selective-retransmission granularity. Tail-biting convolutional coding
avoids termination bits in short control blocks, at the cost of a 2048-state
decoder and CRC-screened candidate search. Neither choice is a claim of optimal
coding performance against all alternatives.

PN9 whitening reduces dependence on repeated source patterns without changing
the code rate. The rectangular interleavers are deterministic, invertible and
require no transmitted permutation. Striped frames distribute each codeword
across frequency and time; grouped frames sacrifice some of that diversity so
a failed frequency group need not force the whole frame to be retransmitted.
Clipping thresholds and receiver smoothing constants remain empirical operating
points; their suitability depends on constellation, channel and PA behavior.

---

## 3. The OFDM waveform

### 3.1 Acquisition — paired conjugate-root Zadoff-Chu

Burst detection, symbol timing, and carrier-frequency offset come from a
**paired conjugate-root ZC preamble**: two segments carrying a chirp and its
conjugate. A single ZC root has a delay–Doppler ambiguity (CFO shifts the
correlation peak in time); transmitting the conjugate pair separates them — the
**sum** of the two peak positions gives timing, the **difference** gives CFO.

The sequence is the even-length centred chirp at root *u* = 1, and its
conjugate is the *N−u* root:

```
z[m] = exp(j·π·(m - N/2)² / N),   m = 0 … N-1,   N = 128
segment A = z,  segment B = conj(z)
```

The 256-chip pair is generated at a 1200 Hz chip rate, polyphase-resampled ×40
to 48 kHz (**5120 samples per segment, 10 240 total, 213.333 ms**) and mixed to
a **1500 Hz** centre. Its nominal chip-rate band is 900–2100 Hz, with
resampling and finite-burst skirts beyond those endpoints. The resampler runs over the *concatenation*, so the anti-alias
response straddles the A/B junction; the matched references are the two halves
of the same resampled waveform. The preamble is scaled to the RMS of the body
that follows it. A burst is `[512 zeros][preamble][body][256 zeros]`.

Timing and CFO come from the envelope cross-correlation of the two matched
filter outputs, not from two independent peak searches — pairing per-tap
argmaxes reads a 2 ms echo as ~11 Hz of phantom CFO. With
`K = N·D²/f_s = 128·1600/48000 = 4.2667 samples/Hz` and `flag` the
(interpolated) offset of the B-peak from the A-peak less one segment length:

```
tau = peak_A + flag/2                         # sum      -> timing
cfo = flag / (2 * K)                          # difference -> CFO
```

Peaks are parabolically interpolated over three taps. The search half-span is
`ceil(K · 90) = 384` samples, defining a **±90 Hz** CFO search interval. The
detector fires when `peak_A / median(|corr_A|) ≥ 5.0`, or on a fallback pair
prominence `(max − median)/(median − min) ≥ 2.5`. The frame start is the
detected `tau` biased **early** by `(CP − window)/2` samples on scattered gears
(96 at CP 256, 144 on `doppler`), and the CP absorbs the bias.

Acquisition hands off to an integer-bin pilot-coherence search. Everything past the ZC pair — the CP /
Schmidl-Cox / pilot-coherence refinement — is receiver-local and constrains
nothing on the wire.

The ZC path estimates no frequency drift. The floor drift tracker clips its
estimate at ±6.0 Hz/s and the narrow-tone beacon at ±0.25 Hz/s. Search intervals
and clipping thresholds do not guarantee acquisition or decoding at their edges;
channel measurements must state offset, drift, SNR and payload geometry.

### 3.2 Coherent detection and equalization

Detection is **pilot-aided coherent throughout — never differential**. A
scattered pilot lattice tracks the 500 Hz-scale selectivity of a 2 ms channel.
No ladder gear carries continual pilots.

A cell is a pilot iff, with `r` the carrier index **relative to the gear's first
carrier** and `s` the symbol index,

```
(r - stag * s) mod df == 0
```

`(df, stag)` = **(6, 2)** on all five OFDM ladder gears, repeating on each even-indexed
carrier every **3** symbols; odd-indexed carriers use frequency interpolation; `doppler` uses (2, 1) period 2 and `sparse34`
(12, 2) period 6. Every pilot cell carries the constant **1 + 0j** — there is no
PN or BPSK pilot sequence — at the constellation's RMS amplitude, i.e. **0 dB
boost**.

The reference receiver uses one-tap zero-forcing per carrier and weights LLRs
by **|H|²/σ²** before decoding. The weighting reduces confidence in estimates
from deeply faded carriers, where zero-forcing amplifies noise. Noise variance is
estimated from second differences along the pilot tracks (`E|d²|² = 6σ²`). A
carrier whose estimated SNR falls below `min(0.25, 0.15 × median SNR)` is
**masked**: weight 0, and `H` substituted with 1.0 so no NaN survives into HARQ
combining. Equalization, noise estimation, weighting, smoothing and masking
are receiver implementation choices; the pilot geometry and values define
the transmitted waveform.

### 3.3 PAPR

Each gear clips its envelope relative to RMS and then band-limits. The clip
thresholds are QPSK 4 dB, 16-QAM 6 dB, 64-QAM 8 dB and 256-QAM 12 dB.
Filtering can regrow peaks, so these thresholds are not guarantees of the final
waveform PAPR. Deeper clipping reduces the requested peak amplitude but adds
in-band constellation error; the higher QAM orders use shallower clipping.

### 3.4 Constellations

Uniform **Gray-mapped** QPSK / 16-QAM / 64-QAM / 256-QAM. No constellation
shaping. Bits per carrier: QPSK 2, 16-QAM 4, 64-QAM 6, 256-QAM 8.

Bits are packed **MSB first**: a cell's `k` bits `b₀ … b_{k−1}` index the point
table as `Σ b_i · 2^{k−1−i}`. QPSK is

```
points = [ -1-1j, -1+1j, +1-1j, +1+1j ] / sqrt(2)
```

so `b₀` selects the I sign and `b₁` the Q sign, bit 0 → negative. For square QAM
the **first half of the bits drives I, the second half Q**, each
axis an independently reflected-Gray PAM: amplitude
`2·gray_to_binary(axis_bits) − (side − 1)`, then the whole set scaled to unit
average energy — **1/√10** for 16-QAM, **1/√42** for 64-QAM and **1/√170**
for 256-QAM. The 256-QAM axis has 16 levels, the odd integers from −15 to +15.

```
16-QAM axis:   00    01    11    10
               -3    -1    +1    +3      (x 1/sqrt(10))

64-QAM axis:  000   001   011   010   110   111   101   100
               -7    -5    -3    -1    +1    +3    +5    +7   (x 1/sqrt(42))
```

**LLR convention: LLR > 0 ⟺ bit 0**, everywhere in sabir. With `w = |H|²/σ²`
the per-cell weight and `d(x) = |z − x|²`, the max-log soft output is

```
L_b = w * ( min_{x : bit_b = 1} d(x)  -  min_{x : bit_b = 0} d(x) )
```

which for QPSK reduces to `L₀ = −2√2·w·Re z`, `L₁ = −2√2·w·Im z`. An
implementation that inverts the sign inverts every decoder input.

---

## 4. The floor waveform

A **noncoherent MFSK burst**: CA-TBCC-coded bits (§7.2), PN9-whitened and
block-interleaved 16 × n/16 (§6.3.3, §6.3.4), mapped to 4-ary Gray tones, with
embedded Costas sync blocks. The floor carries every control block,
connect/ACK, the ACDS tier, the presence beacon, and FEC-broadcast.

**Tone grid.** Everything sits on a 46.875 Hz grid unit from a
**1359.375 Hz = 29 × 46.875** base. The four data tones are grid units
0, 2, 4, 6 — **93.75 Hz apart**:

| 2-bit value (first bit is MSB) | tone | grid unit | frequency |
|---|---|---|---|
| 0 | 0 | 0 | 1359.375 Hz |
| 1 | 1 | 2 | 1453.125 Hz |
| 3 | 2 | 4 | 1546.875 Hz |
| 2 | 3 | 6 | 1640.625 Hz |

In ascending frequency the labels read 00, 01, 11, 10 — Gray in frequency. The
odd units 1, 3, 5 carry no signal and are the receiver's **noise reference**.
Symbol length is 1024 samples (21.333 ms, 46.875 baud); the occupied span is
281.25 Hz plus CPFSK skirts.

**Sync.** A 7-symbol Costas block `(3, 1, 4, 0, 6, 5, 2)` — grid units, so
1500.0, 1406.25, 1546.875, 1359.375, 1640.625, 1593.75, 1453.125 Hz. Data runs
are at most **56** symbols: with `n_data` data symbols, `n_seg =
ceil(n_data/56)` runs are balanced as `divmod(n_data, n_seg)` with the first
`n_data mod n_seg` runs one symbol longer, and a Costas block **precedes every
run and one trails the last** (`n_seg + 1` blocks). A 22-byte block on the
`floor` gear is 176 data symbols → 4 runs of 44 → 5 Costas blocks at symbol
indices 0, 51, 102, 153, 204 → **211 symbols**. A 44-byte session block is 352
data symbols → 7 runs (two of 51, five of 50) → 8 Costas blocks → **408
symbols**.

**Repeat gears** (×2, ×4) tile the whitened, interleaved bit stream and
square-law combine the per-tone energies before the LLR — classical noncoherent
diversity.

**Shaping.** The per-sample instantaneous-frequency staircase is smoothed by a
**64-sample boxcar** before integration, producing a phase-continuous waveform
with constant envelope away from the burst ramps and reduced frequency-step
sidelobes. The tone wave is amplitude-ramped over **128 samples** at
each end with a raised cosine, inside the 512/256-sample guards.

The narrow-tone beacon sub-mode (§8) uses longer symbols and narrower tone
spacing for non-interactive reporting, with tighter Doppler and drift constraints.

---

## 5. The gear ladder

The baseline gear order is a transmitter selection policy with non-decreasing
carrier allocation. It is not a universal robustness ranking: Doppler and
multipath can favor different waveform families. Every accepted DATA profile
is advertised by its own bit in one 32-bit capability word (`NEGOTIATION.md`).
Accepting a higher-rate profile does not imply accepting any lower profile.

| ID | Gear | Waveform | Carriers | First FFT bin | Allocation | Constellation | FEC (n,k) | Rate | Repeat | Layout |
|---|---|---|---|---|---|---|---|---|---|---|
| — | floor4 | MFSK ×4 | — | — | ≤500 Hz | 4-FSK | CA-TBCC | — | 4 | — |
| — | floor2 | MFSK ×2 | — | — | ≤500 Hz | 4-FSK | CA-TBCC | — | 2 | — |
| — | floor | MFSK ×1 | — | — | ≤500 Hz | 4-FSK | CA-TBCC | — | 1 | — |
| 3 | robust | OFDM | 24 | −12 | 1125 Hz | QPSK | (1536,512) | 1/3 | **2** | striped |
| 4 | workhorse | OFDM | 24 | −12 | 1125 Hz | QPSK | (1024,512) | 1/2 | 1 | striped |
| 5 | workhorse34 | OFDM | 24 | −12 | 1125 Hz | QPSK | (1024,768) | 3/4 | 1 | striped |
| 6 | fast | OFDM | 56 | −28 | 2625 Hz | 16-QAM | (1024,768) | 3/4 | 1 | grouped |
| 7 | max | OFDM | 56 | −28 | 2625 Hz | 64-QAM | (1536,1280) | 5/6 | 1 | grouped |

The 24-carrier centers span **960.9375–2039.0625 Hz**; the 56-carrier
centers span **210.9375–2789.0625 Hz**. Their allocation widths are respectively
1125 and 2625 Hz: one full spacing per carrier, with a half-spacing margin
outside each outer center. Allocation width is not occupied-power bandwidth.

The clipping filter has a symmetric flat region with half-width
`max(600, n_carriers × spacing / 2)` Hz about 1500 Hz. Its raised-cosine
transition is `min(100, 1400 - half_width)` Hz on each side. Thus the 24-carrier
filter design is flat at 900–2100 Hz with transition endpoints 800/2200 Hz;
the 56-carrier design is flat at 187.5–2812.5 Hz with endpoints 100/2900 Hz.
The reference realizes this response with an 8193-tap Hann-windowed FIR, applied
by delay-compensated linear convolution on both clipping passes. The finite
filter has sidelobes, and its tails are truncated to burst length. Finite bursts,
the preamble and subsequent radio filters must therefore be included when
measuring spectral containment.

A common audio center is not optimal for every asymmetric radio filter. Wide
profiles need a passband centered near 1500 Hz with enough low-frequency
response for their outer carriers. A receiver with a 300 Hz low-cut can favor
an upward-shifted allocation, while a symmetric receiver passband can favor the
centered allocation under frequency offset. Qualify the complete radio and
receiver filter path before selecting 256-QAM; a nominal filter-width setting
is not a measured transfer response.
Frames are 4 / 12 / 16 / 48 / 60 codewords at rungs robust … max.

Floor, floor2 and floor4 are control waveforms, not DATA profile IDs. The
narrow DATA profiles 18–20 carry payload using those waveforms. Every station
implements floor control so it can receive a handshake; its DATA advertisement
contains only the profiles it can receive within its configured bandwidth.

The receive profile set may be sparse. A transmitter selects only profiles in
its peer's advertised set and within its own configured bandwidth. Receiver
capabilities are directional, so the two directions may use different sets.
A station with no mutually usable DATA profile may exchange controls; queued
payload that cannot be sent causes the reference to end the link with a
link-failed result rather than leave the queue stalled indefinitely.

The gearshift is transmitter-driven: up on a clean-frame streak once the
measured 3 kHz SNR clears the next rung's entry, down on a stall or a rebuild.
Per-carrier bit/power loading and dead-carrier masking ride the DATA header
(§6.3) and the SNR reported in every ACK.

### 5.1 The DATA profile registry

A DATA profile ID names one complete, immutable tuple: waveform family,
numerology, carrier set, constellation, pilot lattice, FEC matrix, codeword
geometry and frame layout. The same uint16 namespace addresses profiles in DATA and DATAGRAM headers;
bit `id` of the capability word advertises a currently assigned DATA profile.
Nothing about a receiver's advertised order enters the encoding. IDs 3–7 are the ladder rungs
above; 16 and 17 promote the two off-ladder OFDM gears to selectable profiles;
18–20 carry user data on the MFSK floor; 21 is the short-CP 256-QAM profile.

| Profile | ID | Modulation | Carriers | Symbol rate | Code rate | Allocation / tone span | Class | Bytes/cw | Cw/frame |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| robust | 3 | QPSK, repeat 2 | 24 | 37.5 Bd | 1/3 | 1125 Hz | 1500 | 61 | 4 |
| workhorse | 4 | QPSK | 24 | 37.5 Bd | 1/2 | 1125 Hz | 1500 | 61 | 12 |
| workhorse34 | 5 | QPSK | 24 | 37.5 Bd | 3/4 | 1125 Hz | 1500 | 93 | 16 |
| fast | 6 | 16-QAM | 56 | 37.5 Bd | 3/4 | 2625 Hz | 2750 | 93 | 48 |
| max | 7 | 64-QAM | 56 | 37.5 Bd | 5/6 | 2625 Hz | 2750 | 157 | 60 |
| doppler | 16 | QPSK | 24 | 57.692 Bd | 1/2 | 2250 Hz | 2750 | 61 | 8 |
| sparse34 | 17 | QPSK | 24 | 37.5 Bd | 3/4 | 1125 Hz | 1500 | 93 | 16 |
| narrow | 18 | 4-FSK | — | 46.875 Bd | CA-TBCC 1/2 | ~281 Hz | 500 | 19 | 1 |
| narrow2 | 19 | 4-FSK, repeat 2 | — | 46.875 Bd | CA-TBCC 1/2 | ~281 Hz | 500 | 19 | 1 |
| narrow4 | 20 | 4-FSK, repeat 4 | — | 46.875 Bd | CA-TBCC 1/2 | ~281 Hz | 500 | 19 | 1 |
| wide256 | 21 | 256-QAM | 56 | 41.667 Bd | 5/6 | 2625 Hz | 2750 | 157 | 64 |

**Class** is the bandwidth budget a profile is admitted under: a station whose
configured limit is narrower neither transmits nor advertises the profile.
Codewords per frame is a transmitter default, not a receiver maximum — every
DATA receiver accepts 1–64 codewords of any profile it advertised, so airtime
policy changes without a new ID. Symbol rate is `48000 / (n_fft + cp)`;
carrier allocation width is `n_carriers × 48000 / n_fft`; MFSK entries show
the outer tone-center span, excluding skirts.

`doppler` (ID 16) uses FFT 512, CP 320, a 32-sample window and 93.75 Hz
spacing. Its 24 centers span 421.875–2578.125 Hz,
and a (2, 1) lattice putting a pilot on every carrier every second symbol.
`sparse34` (17) keeps the default numerology with a (12, 2) period-6 lattice —
1/12 pilot overhead against `workhorse34`'s 1/6.

`wide256` (21) is the default numerology with **CP 128** and a (12, 2) period-6
lattice: FFT 1024, 56 carriers centered at 1500 Hz, the r5/6 (1536, 1280) matrix, 12 dB
envelope clip, no ACE. Its 16-PAM Gray axes take amplitudes −15, −13, … 15
divided by `sqrt(170)`; bits 7–4 select I, bits 3–0 select Q, through the same
Gray-to-binary map as every other constellation. The short CP deliberately
trades multipath margin for symbol rate, so a sender uses it only when the peer explicitly advertises ID 21.

Profiles 16–21 require an all-zero loading map. Adaptive loading above 64-QAM
needs a new interpretation of the nibble alphabet and therefore a new
assignment.

A narrow (18–20) codeword is a 22-byte information block —
`[payload_length:1][payload and zero pad:19][crc16:2]` — through the floor's
(352, 176) CA-TBCC, PN9 whitener and 16-row interleaver. Each present codeword
is a complete floor burst with its own guards; the DATA header's `n_symbols` is
the sum of the floor symbol counts, guards excluded, and codeword order is the
ascending bitmap order. Retransmissions retain the coded LLRs, so Chase
combining works exactly as it does on an OFDM body. No OFDM is required to move
payload on these profiles.

Profiles 16–21 are validated in the simulator and in loopback only; none has
been transmitted over a radio path.

---

## 6. Framing

### 6.1 Two block widths

Session controls are **44 bytes**; connectionless blocks — the presence beacon
and the DATAGRAM header — are **22 bytes**. Both end in a big-endian
**CRC-16/CCITT-FALSE** over every preceding byte. This is a closed two-length
grammar: the receive path tries each width's protected decode and checks type,
version, structure and CRC before dispatch, so no length is ever inferred from
undecoded audio.

| Type | Name | Type | Name |
|---|---|---|---|
| 1 | CONNECT | 6 | DISC_ACK |
| 2 | CONNECT_ACK | 7 | TURN |
| 3 | DATA | 8 | TURN_REQ |
| 4 | ACK | 9 | ID |
| 5 | DISC | 10 | CAPS |

Type 11 is DATAGRAM (§8), connectionless and 22 bytes. Type 48 is the presence
beacon. Other unassigned types are ignored; the block-type registry is
`NEGOTIATION.md`. Types 0x00 and 0xFF are invalid on receive, as
is any session block whose byte 1 is not wire version 2 or whose 64-bit session
identifier is zero.

CONNECT and CONNECT_ACK use bytes 0–9 for type, wire discriminator and session
ID; byte 10 is reserved zero. Byte 11 contains the count of trailing CAPS
blocks (0–3) in its low nibble; bit 7 is CFAIL on CONNECT_ACK only, and all
other bits are zero. The uint32 capability word occupies bytes
12–15, source identity bytes 16–27, destination identity bytes 28–39, reserved
zeros bytes 40–41 and CRC bytes 42–43. Identities are ASCII space-padded to 12
bytes. The receiver checks the destination; the initiator checks the expected
reply source. Integer fields are big-endian except the codeword bitmap.

A capability word names all current receive DATA profiles directly: bits 3–7
and 16–21 have the same numbers as their profile IDs. Optional feature bits are
24 FASTCTL, 25 PBACK, 26 LOADING, 27 DEFLATE and 28 BEACON. All other bits are
unassigned and ignored. A received profile bit authorizes that profile only;
features that change session behavior require support by both stations. The
current profile and feature advertisement fits in one CRC-protected CONNECT
or CONNECT_ACK. Profile and feature assignments are defined in `NEGOTIATION.md`.

Optional extensions use zero to three CAPS blocks. Each is 44 bytes with the
same type/version/session prefix, total/index at byte 10, a 31-byte TLV area
at bytes 11–41, and CRC at bytes 42–43. Unknown optional TLVs are skipped;
unknown critical TLVs refuse the connection. GEARSET (type 1) advertises
uint16 profile IDs at least 24 and cannot change bitmap permissions. IMPL
(type 5) carries opaque implementation identification. A connection takes
effect only after every declared extension position arrives; duplicate
positions cannot replace a missing position, and conflicting blocks invalidate
the offer. All current profiles fit the bitmap and require no CAPS blocks.

### 6.2 DATA and ACK

Both share the session envelope: `[type:1][version:1][session:8][seq:4]
[profile:3][mask:8][aux:8][offset:8][rsv:1][crc16:2]`. `seq` is a uint32
transmission generation and does not wrap; `offset` is a uint64 byte position in
that direction's logical stream. `rsv` (byte 41) is zero on send and rejected
nonzero on receive.

- **DATA** — bytes 14–16 are a uint24: the low 16 bits are the absolute profile
  ID (§5.1), bit 16 is **HANDOVER**, and any higher bit rejects the block.
  `mask` (64 bits, codeword *j* at bit *j* of a little-endian integer) marks the
  codewords present in this body. `aux` = `[n_syms:2][n_cw:1][loading:4][rsv:1]`
  — `n_cw` is 1–64 and `rsv` (byte 32) must be zero. The four loading bytes hold
  eight per-group nibbles, group `2i` in the high nibble of byte `3+i`: 0 profile
  default, 1 off, 2/3/4 QPSK / 16-QAM / 64-QAM. Values 5–15 are unassigned and a
  DATA body carrying one is rejected rather than demodulated; the nibble
  expansion itself falls back to the profile default, so an unassigned value can
  only ever cost loading precision, never produce a wrong bit width.
- **ACK** — bytes 14–16 carry the feedback flags: bit 0 (ACK_TRAFFIC) requests
  turnaround, bit 1 (ACK_TOOK_ROLE) marks a piggybacked ACK whose sender has
  taken the send role. `mask` is the codewords decoded so far, cumulative within
  the generation — an all-clear short of the frame's codeword count is the
  selective NAK. `aux` = eight per-group SNR bytes, 0.25 dB/LSB from −16 dB,
  0xFF = no measurement.

ID, DISC and DISC_ACK keep the identity view: four zero bytes then the first four
station bytes in the bitmap field, the remaining eight in `aux`. Their `seq`,
profile field and `offset` are zero on transmit.

### 6.3 The payload path — records, codewords, whitening, interleaving, layout

Five stages between an application's bytes and the modulator's bit stream. Each
is stated here in full; nothing in this section is deferred to a module.

#### 6.3.1 Record layer

The application byte stream is carried as a sequence of **records**, one per
`Send`:

```
[flags:1][length:3, big-endian][body:length-32][sha256:32]
```

`flags` is **0x10** (raw) or **0x11** (raw DEFLATE); no other value is accepted.
`length` counts the encoded body plus the 32-byte trailer, and the digest covers
the 4-byte header followed by the encoded body. Verification precedes
decompression and precedes delivery, so nothing under an unverified digest ever
reaches the host. DEFLATE is **raw DEFLATE (RFC 1951) with no zlib or gzip
wrapper** — `zlib` window bits −15; a zlib-wrapped stream is not interoperable.
The reference compresses at level 9 and emits raw whenever the compressed body
is not shorter than the input, so a host asking for compression may still see a
raw record on the wire. Maximum decoded record is 8 MiB, enforced on both
sides, and decompression is bounded to that. Records are a byte stream across
codeword and frame boundaries — a receiver buffers partial records.

Integrity is unconditional: it applies to records queued before negotiation
finishes as much as after it, and there is no configuration that omits the
trailer. The digest detects corruption; stable stream offsets suppress duplicate
delivery. Neither authenticates the sender.

A receiver MUST NOT deliver the body of a record whose flag it does not
recognize — the flag names the body's encoding, so delivery would hand the host
bytes under an unknown transformation — and MUST fail loud: error the record to
the host, or drop the session (the reference drops the session;
`NEGOTIATION.md`). `length` is defined independently of `flags`, so every
record stays self-delimiting whatever its flag and an unknown one never costs
stream framing.

#### 6.3.2 Codeword block

The record stream is chunked into `k/8 − 3` byte pieces, one per LDPC codeword
(61 bytes at k = 512, 93 at k = 768, 157 at k = 1280). Each piece becomes a
`k/8`-byte block:

```
[len:1][data:k/8-3, zero-padded][CRC-16:2, big-endian]
```

`len` is the number of valid data bytes (≤ `k/8 − 3`); the CRC-16/CCITT-FALSE
covers the first `k/8 − 2` bytes. A receiver rejects the codeword if the CRC
fails or `len` exceeds `k/8 − 3`. The block's bits, **MSB-first within each
byte**, are the LDPC information vector of §7.1.

#### 6.3.3 Whitening — PN9

The `n` coded bits of each codeword are XORed with the first `n` outputs of a
9-bit LFSR, **restarted at the head of every codeword**:

```
state = 0x1FF
for i in range(n):
    out[i] = state & 1                                   # LSB out
    state  = (state >> 1) | (((state ^ (state >> 5)) & 1) << 8)
```

That is `x⁹ + x⁵ + 1` in Fibonacci form, seed all-ones. First 32 outputs:
`11111111100001111011100001011001`; the first 64 packed MSB-first are
`ff87b859b7a1cc24`. De-whitening on soft values is a sign flip, so it costs no
soft information.

The same PN9 whitens the CA-TBCC output of the floor waveform (§4) and of the
fast control tier, and fills unused cells (§6.3.5).

#### 6.3.4 Interleaving

Each whitened codeword is block-interleaved over `R` rows: write the `n` bits
row-wise into an `R × (n/R)` array, read column-wise.

```
out[i] = in[(i mod R) * (n/R) + (i div R)]
```

`R = 32` for LDPC payload codewords; `R = 16` for a control block on the floor
waveform and on the fast control tier, so the array is 16 rows by `n/16` columns
— 16 × 44 for a session block (`n = 704`), 16 × 22 for a connectionless one
(`n = 352`). De-interleaving is the inverse scatter.

#### 6.3.5 Frame layout — two disciplines

`n_symbols_for(x)` throughout this section means the smallest number of OFDM
symbol rows whose cumulative data-cell bit capacity reaches `x`, under the
frame's loading map.

`n_cw` interleaved codewords of `n` bits each become one flat bit stream. Which
discipline applies is a property of the profile, carried nowhere on the wire, so
both ends derive it from the profile ID alone:

| Discipline | Profiles |
|---|---|
| Striped | robust 3, workhorse 4, workhorse34 5, doppler 16, sparse34 17 |
| Grouped | fast 6, max 7, wide256 21 |
| Neither | narrow 18, narrow2 19, narrow4 20 |

The narrow profiles have no frame layout at all: each codeword is its own
complete floor burst, guards included, laid end to end in ascending bitmap order
(§5.1). Nothing is striped or grouped because there is no shared OFDM frame to
lay codewords across.

**Striped (low gears).** Codeword bits are round-robined so every codeword
spans the whole frame's fade cycles and coherence bands:

```
stream[i * n_cw + j] = codeword[j][i]        i = 0..n-1,  j = 0..n_cw-1
```

A gear with `repeat` R > 1 (robust, R = 2) transmits the whole stream R times
back to back; the receiver sums the LLRs of the copies before decoding. The
body is `n_symbols_for(R * n * n_cw)` OFDM symbols; the PHY pads the tail to a
whole symbol with PN9 filler.

**Grouped (high gears).** Each codeword is confined to one of the 8 carrier
groups, so a notched or faded group kills only its own codewords — that is what
makes the carrier-group selective ACK (§7.3) a salvage unit rather than a
statistic. Repeat is not defined for grouped gears.

*Cell order.* For a body of `n_syms` symbols the data cells are enumerated
row-major — symbol index outer, relative carrier index inner — keeping only
cells that are neither pilots (§3.2) nor switched off by the DATA loading map
(§6.2). Cell *i* carries `b_i` bits, `b_i` being its group's constellation
width, at flat offset `off_i = Σ_{j<i} b_j`. Group *g*'s **slot list** is the
concatenation, in cell order, of `[off_i, off_i + b_i)` over every cell whose
carrier group is *g*; loading is uniform within a group, so `b` is constant
along a slot list. Carrier group of relative carrier *r* is `r * 8 //
n_carriers` (§5).

*Assignment.* Both ends recompute it from the DATA header alone — no assignment
map is transmitted:

1. `rate[g]` = (number of slots of group *g* at `n_syms = 12`) / 12. Twelve is
   an integer number of lattice periods (four at period 3, two at period 6),
   so the probe is unbiased by pilot phase. If
   every rate is 0 the frame cannot be sent.
2. `load[g] = 0`. For each codeword *j* = 0…`n_cw`−1 **in order**, assign it to
   the group minimizing `(load[g] + n) / rate[g]` over groups with
   `rate[g] > 0` — the group that would finish transmitting earliest — ties to
   the lowest group index; then `load[g] += n`.

*Sizing.* With `need[g]` = `n` × (codewords assigned to *g*), start from
`rows = n_symbols_for(Σ need)` and repeat: if any group has fewer slots than
its need at `rows`, double `rows`; otherwise `n_syms` is one plus the largest
symbol index among the `need[g]`-th slot of each non-empty group. The result
rides the DATA header's `n_syms` field. The receiver re-derives and validates
that geometry before allocating demodulation arrays. Both ends MUST compute
it this way to agree on the codeword slots.

*Placement.* The codewords assigned to group *g*, concatenated in increasing
*j*, fill that group's slot list in order. Every remaining slot takes the PN9
value **at its own flat index** — `fill = pn9(total_slots)`, `bits[s] =
fill[s]` — not the next value of a running sequence. Filler keeps the padded
cells statistically like data.

Frame integrity is the FEC plus the per-codeword CRC-16; DEFLATE is the only
compression and is explicitly signaled per record.

---

## 7. FEC

### 7.1 QC-LDPC ladder (OFDM gears)

An independent quasi-cyclic LDPC code per rate — no protograph-extended nested
family, no shared mother code. Registered codes:

| Code | (n, k) | Rate | mb × nb | Info column weight | Used by |
|---|---|---|---|---|---|
| r13 | (1536, 512) | 1/3 | 16 × 24 | 5 | robust |
| r12 | (1024, 512) | 1/2 | 8 × 16 | 3 | workhorse, doppler |
| r34 | (1024, 768) | 3/4 | 4 × 16 | 3 | workhorse34, fast, sparse34 |
| r56 | (1536, 1280) | 5/6 | 4 × 24 | 3 | max, wide256 |

The encoder uses QC lifting and dual-diagonal structure; the reference decoder
uses normalized min-sum.

#### 7.1.1 Lifting

Lifting size **Z = 64** for all four codes. Each code is an `mb × nb` **base
matrix** `B` of circulant shifts, with `(n, k) = (nb·Z, (nb − mb)·Z)` and
`kb = nb − mb` information block-columns.

`B[i][j] ∈ {0 … Z−1}` expands to the Z × Z circulant permutation `P_s`;
`B[i][j] = ·` (absent) expands to the Z × Z zero block. The convention is

```
(P_s v)[r] = v[(r - s) mod Z]                    i.e. P_s v == roll(v, s)
H[i*Z + r][j*Z + c] = 1  iff  c = (r - s) mod Z
```

A codeword is `c = [ m_0 … m_{kb-1} | p_0 … p_{mb-1} ]`, each block Z bits,
block-major: information block *j* occupies bit positions `j·Z … j·Z + Z − 1`,
parity block *i* occupies `(kb + i)·Z … (kb + i)·Z + Z − 1`. Bit order within a
block is the LFSR/byte order of §6.3.2 — MSB-first.

#### 7.1.2 Structure of the base matrix

Normative, and the reason the encoder is a shift-and-XOR recursion rather than
a matrix solve:

- **Columns 0 … kb−1** — information. Column weight is the per-code constant in
  the table above; row weights are near-uniform but not exactly balanced.
- **Column kb** — the weight-3 **anchor**. Entries at rows `0`, `mb//2`,
  `mb−1`, with shifts `s0`, `0`, `s0` respectively. `s0` is per-code (r13 57,
  r12 38, r34 37, r56 52) and readable off the matrix.
- **Columns kb+1+j, j = 0 … mb−2** — the **dual diagonal**. Entries at rows `j`
  and `j+1`, both with shift **0**.

Summing all `mb` block-rows kills every dual-diagonal column (each appears
twice at shift 0) and reduces the anchor column to the identity (`P_{s0}` twice
plus `P_0` once), which is what makes `p_0` solvable in closed form.

#### 7.1.3 Encoder

With `m_j` the information blocks and `⊕` bitwise XOR:

```
u_i  = ⊕_{j < kb, B[i][j] present}  roll(m_j, B[i][j])          i = 0 … mb-1

p_0  = ⊕_i u_i                                     # anchor: the shifts cancel
p_1  = u_0 ⊕ roll(p_0, s0)
for i = 1 … mb-2:
    p_{i+1} = u_i ⊕ p_i
    if i == mb//2:  p_{i+1} ^= p_0
```

Verified: for all four codes this reproduces the reference encoder bit for bit,
and `H·cᵀ = 0` over GF(2) for the codewords it produces.

#### 7.1.4 The matrices are data, not a recipe

The base-matrix tables below are normative. Reimplementations use these entries
rather than reproduce a pseudorandom search: matching a seed alone does not
specify a generator's permutation and bounded-integer draw algorithms. Matrix
and encoder hashes in §KAT verify the resulting codes independently of those
implementation choices.

The construction the search enforced, recorded so a future rung can be built to
the same standard: information columns are `layers` superimposed random
permutations (kb ≤ mb) or `layers` distinct rows drawn per column (kb > mb);
shifts are uniform on `[0, Z)`; and shifts are resampled until the lifted graph
has **no 4-cycles** — for any two block-columns c₁, c₂ sharing rows r₁ ≠ r₂,
`(B[r₁][c₁] − B[r₁][c₂] + B[r₂][c₂] − B[r₂][c₁]) mod Z ≠ 0`. All four shipped
matrices satisfy this; girth ≥ 6.

#### 7.1.5 Base matrices (normative)

`·` = absent (zero block). Rows are block-rows 0…mb−1, columns block-columns
0…nb−1.

**r13** — 16 × 24, kb = 8, s0 = 57:

```
      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
  0   ·   ·  54   ·   ·   ·   ·  18  57   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
  1   4   ·   ·   ·   ·  42   ·   9   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
  2   ·   8   ·   ·  62  27   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
  3   ·   ·   ·  52  34   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
  4   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
  5   3   ·  18   ·   2   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·
  6   ·   ·   ·   ·   ·  51   0  62   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·
  7   ·   ·  50   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·
  8   ·  52   2   ·   ·   ·   6   ·   0   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·
  9   ·   6   ·  58  25   ·   ·  21   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·
 10   ·  51   ·  32   ·   ·  51   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·
 11   ·   ·   ·  24   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·
 12  46  24  46   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·
 13  22   ·   ·   ·  59  39   ·   8   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·
 14  38   ·   ·   ·   ·   ·  18   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0
 15   ·   ·   ·   6   ·  50  12   ·  57   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0
```

**r12** — 8 × 16, kb = 8, s0 = 38:

```
      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  0   ·  48   ·   ·   ·  12  20   ·  38   0   ·   ·   ·   ·   ·   ·
  1   ·   ·   9   ·   ·  37   ·   1   ·   0   0   ·   ·   ·   ·   ·
  2   9   ·  30   ·   ·   ·  40   ·   ·   ·   0   0   ·   ·   ·   ·
  3   5  46   ·   ·   ·   ·   ·  14   ·   ·   ·   0   0   ·   ·   ·
  4  32   ·   ·   5   ·   ·   ·  58   0   ·   ·   ·   0   0   ·   ·
  5   ·  50  23   ·   5   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·
  6   ·   ·   ·  39  50  49   ·   ·   ·   ·   ·   ·   ·   ·   0   0
  7   ·   ·   ·  18  59   ·  34   ·  38   ·   ·   ·   ·   ·   ·   0
```

**r34** — 4 × 16, kb = 12, s0 = 37:

```
      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  0  39  44  27  32  20   ·   8   3  35   5   ·  41  37   0   ·   ·
  1   ·  30  59  42   ·  15   ·   ·   1  42  58  17   ·   0   0   ·
  2   6  13   ·   ·  39  28   1  24  55  51  12  53   0   ·   0   0
  3  57   ·   0  48  13  55  41   6   ·   ·  43   ·  37   ·   ·   0
```

**r56** — 4 × 24, kb = 20, s0 = 52:

```
      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
  0  28   9  25  51  62  49  42   ·   ·  58  56   ·  45  28  19  34  39  14  36   ·  52   0   ·   ·
  1  25   3  17  13   4   ·   ·  33  23  12   ·  42  47  42   9   ·  12  61   ·  50   ·   0   0   ·
  2  37   ·   ·   ·  11   4   0  20  51  36  27  16  28  27  24  15   ·   ·  43  49   0   ·   0   0
  3   ·   8  39  28   ·  38  57   2  48   ·  35  49   ·   ·   ·   2   4  49   5  56  52   ·   ·   0
```

#### 7.1.6 Decoder

Receiver-local — nothing here affects interoperability, and an implementation
is free to do better. The reference is **normalized min-sum**, scale factor
**0.8**, at most **50** iterations, checking the syndrome after every iteration
and stopping the moment it is zero. **LLR > 0 ⟺ bit 0**, throughout sabir.

### 7.2 CA-TBCC (floor)

The floor and all control blocks use a **CRC-aided tail-biting convolutional
code**: rate-1/2, constraint length K = 12 (2048 states), generators
(4335, 5723) octal. Octal
convention: the MSB of each generator taps the current input bit. Output bits
interleave `c₀[0], c₁[0], c₀[1], …`; the shift register starts loaded with the
block's own last K−1 bits, so the trellis path is circular. A 22-byte block
gives 352 coded bits, a 44-byte session block 704.

Decoding is receiver-local: the reference uses CRC-list wrap-around Viterbi,
`wraps` = 3 (two boundary-converging passes then one list pass), list 24,
survivors per state **4** on the floor and the beacon and **1** on the fast
control tier.
The screening predicate is the block CRC-16 the frame already carries, and for
a multi-block burst it requires **every** block CRC-clean. The short-block code is shared by floor, beacon and fast-control waveforms.

### 7.3 HARQ

A NAK triggers **Chase soft-combining** (retransmit + soft-combine, receiver-
local, no rate-compatibility needed). Carrier-group **selective ACK** retransmits
and soft-combines only the failed codewords, not whole frames — the selective
granularity equals the codeword, which is why high gears use carrier-group-
localized codewords and low gears full-band interleave.

### 7.4 Session identity, delivery and recovery

The initiator chooses a fresh random nonzero uint64 session ID for each new
connection attempt; retries preserve it. DATA, ACK, role exchange and disconnect
are scoped to that ID. Each direction maintains independent transmission
generations and byte offsets. Generations are uint32 and do not wrap: exhaustion
ends the session. The identifier provides accidental session isolation, not
authentication or durable replay protection across restarts.

A generation identifies one physical frame encoding. Its offset identifies the
position of its payload in that direction's logical byte stream:

- A retry preserves generation, offset, profile and codeword geometry. Loading
  and handover flags may change under their defined rules.
- A rebuild preserves the logical offset and queued bytes but selects a fresh
  generation. Profile and codeword boundaries may change.
- A smaller rebuilt frame covers a prefix of the original range; its successor
  starts at the actual next stream offset.

The receiver tracks its committed byte offset. It rejects gaps, combines soft
evidence only within the same generation/profile/count/offset, and delivers only
bytes beyond the committed prefix after the frame decodes completely. Fully
duplicate frames are acknowledged without another delivery. An exact duplicate
of the last complete frame receives its full ACK even when a retry carries only
selected codewords.

ACK reports physical-generation completion. Partial acknowledgement does not
commit logical bytes at the sender or reduce host pending-byte accounting. The
stop-and-wait sender commits a generation after its complete bitmap arrives.
Stable receive offsets make rebuilds safe when all earlier full ACKs were lost.
After disconnect, an application can remain uncertain whether its final
unconfirmed message arrived; cross-session transactions need application IDs and
acknowledgements.

Each CONNECT or CONNECT_ACK carries current profile and feature capabilities
in one protected block. If optional CAPS blocks are declared, all must arrive
with matching session, count and distinct positions before the offer takes
effect. Missing extensions never imply reduced capability. An active session
accepts only identical reoffers of its accepted capabilities. A delayed DISC for the immediately previous session may
receive a courtesy DISC_ACK but cannot terminate the active session; the
immediately retired CONNECT is refused while listening.

A caller that enables fast control, supports its bandwidth and has no optional
extensions sends its first CONNECT on the fast-control waveform. A listener
with matching support and no reply extensions answers on that waveform.
This is a bounded acquisition probe, not an assumption that fast control will
work in both directions. Its timeout is twice the fast-control duration plus
two configured turnarounds and the response margin. A timeout causes a floor
CONNECT retry with the same session ID. Narrow or floor-only configurations
and offers with extensions start on the floor; listeners unable to accept the
probe wait for that floor retry. Subsequent control selection follows the
accepted capabilities and channel policy.

With 250 ms modeled turnarounds, ten independent AWGN trials per condition
measured 1.921 s establishment at +20 dB SNR in 3 kHz, including 1.421 s of
control airtime. At −10 dB, an asymmetric +20/−10 dB link, or an erased first
fast ACK, each trial established through the floor retry in 21.362 s. Floor-only
establishment takes 17.940 s under the same timing assumptions. The fast probe
therefore saves 16.019 s when it succeeds and costs 3.422 s before fallback in
these cases. These virtual-time results exclude real audio and host scheduling.

An established session has a receiver-local inactivity deadline, 180 seconds
by default. Valid peer traffic refreshes it. Validated long DATA reception and
local transmission protect their active interval from premature expiry.
Expired sessions close without requiring a received disconnect. This bounds
the passive side's wait after a complete one-direction outage; it does not
resolve whether unacknowledged application bytes arrived.

---

## 8. Extension layers

- **Capability negotiation** — each CONNECT/CONNECT_ACK carries one complete
  profile-and-feature bitmap, defined in `NEGOTIATION.md`. Current profiles
  need no extension exchange. Optional CAPS blocks carry further profile IDs
  or typed metadata; an incomplete declared offer is ignored. The advertisement
  is directional receive capability, and unknown bits never authorize an
  unimplemented profile or feature.
- **Presence beacon / CQ / sounding** — the connectionless frame-class in the
  type range. The presence beacon is **type 48**, 22 bytes on
  the floor waveform:
  `[type:1][addressee:1][capabilities:4, big-endian][station_id:12][profile:1]
  [rsv:1][crc16:2]`, CRC-16/CCITT-FALSE big-endian over bytes 0–19. `rsv`
  (byte 19) is sent zero and is **not** checked on receive, so it is not a
  canary; the CRC is what protects it. The DATAGRAM header's reserved bytes
  18–19 *are* checked (§8.1).
  Byte 1 is the addressee (all-call 0 / group-net-id / unicast), never a session
  tag, so it is routed before the session-0 canary. It carries the station's
  advertised capability bitmap so a listener learns a peer without a
  session (`HOST-API.md` §8).

- **Narrow-tone beacon** — a *different object* from the presence beacon above,
  and the two must not be conflated. A narrow-tone sub-mode carrying a **9-byte**
  payload (6-character callsign base-38, 4-character Maidenhead locator, 8-bit
  status, then CRC-16) — not a 22-byte control block. It reuses the floor's
  Costas array, Gray map, 16-row interleaver, PN9 whitener and CA-TBCC, but on
  its own numerology, anchored on 1500 Hz rather than the 1359.375 Hz floor
  base, with data tones two grid units apart and a maximum data run of 24
  symbols:

  | Sub-gear | Symbol | Tone spacing | Repeat | Duration (9-byte payload) |
  |---|---|---|---|---|
  | beacon_short | 8192 samples | 11.719 Hz | 1 | 17.07 s |
  | beacon_med | 16384 | 5.859 Hz | 1 | 34.13 s |
  | beacon_deep | 32768 | 2.930 Hz | 1 | 68.27 s |
  | beacon_deep2 | 32768 | 2.930 Hz | 2 | 131.75 s |

  Amplitude ramp 1024 samples; frequency smoothing `symbol/16` samples. The
  receiver channelizes to 375 Hz around 1500 Hz (decimation 128), searches
  approximately ±75 Hz CFO and clips estimated drift at **±0.25 Hz/s**.
  Separately-synced captures may be LLR-combined across repetitions.
- **Host interface** — a single framed-CBOR connection with a symmetric HELLO,
  structured commands and events, and link telemetry: normative in `HOST-API.md`.

### 8.1 DATAGRAM — one packet, no session

Block type **11**, 22 bytes, all multibyte fields big-endian:

| Bytes | Meaning |
|---|---|
| 0 | Type 11 |
| 1 | Header version 1 |
| 2–3 | Absolute DATA profile ID (§5.1) |
| 4–5 | Body symbol count |
| 6 | Codeword count, 1–64 |
| 7 | Flags, zero in version 1 |
| 8–9 | Decoded packet byte count, 1–65535 |
| 10–17 | First eight bytes of SHA-256 over the decoded packet |
| 18–19 | Reserved zero |
| 20–21 | CRC-16/CCITT-FALSE over bytes 0–19 |

The body is the named profile's ordinary codeword codec under the same
striped / grouped / floor layout a session DATA body uses, so a datagram's
codeword count may exceed that profile's default frame size, up to the format's
64. The header rides the floor waveform by default; an application or net
profile may select the fast control carrier instead. There is no acknowledgement
and no negotiated claim about who is listening: a receiver decodes the profiles
it implements within its configured bandwidth and drops silently on an unknown
profile or version, invalid geometry, an incomplete codeword set, a token
mismatch or a nonzero flag byte. Geometry is validated before any DSP array is
allocated. The 64-bit token binds the header to its own body; it never
authorizes soft-combining two undecoded packets — repetition repeats whole
packets and combines at the object layer, after each packet's integrity holds.

### 8.2 Object envelope, version 1

A datagram body carries an object fragment. The fixed prefix is 72 bytes,
`>BBHH16sIIQ32sBB`: version 1, flags, header length, uint16 service, 16-byte
message ID, uint32 fragment index, uint32 fragment count, uint64 object length,
SHA-256 of the whole object, source length, destination length — then the two
identities as ASCII, then the fragment data. Header length is 72 plus both
identity lengths. An identity is 1–63 printable ASCII bytes, uppercased;
destination `*` is all-call and anything else is an exact match against the
receiver's configured identity. Group membership is application policy.

Flags 0 marks a data fragment, index `0 … count−1`. Flags 1 marks one XOR parity
fragment whose index equals `count`: the XOR of the equal-width data fragments,
the last zero-padded to that width. Count excludes the parity fragment, which
repairs at most one erasure. Object length fixes the final fragment's length.
Every completion, repaired or not, requires the exact length and a matching
SHA-256. Unknown flags and unknown versions are rejected. A future outer code
takes its own flag or envelope version; existing bytes are never reinterpreted.

The reference receiver bounds an object to 8 MiB, fragment data to 8192 bytes,
fragment count to 65535, pending objects to 32, and retention to 600 seconds; it
charges per-fragment metadata against that budget and evicts the oldest
incomplete object when it must. Completion deduplication is bounded and expires.
Applications may set smaller limits. Conflicting metadata, or a duplicate
fragment with different contents, cannot contribute to a completion. An 8 MiB
format limit is not a claim that any profile can broadcast an object that large
inside the retention window.

Service 0 is opaque application bytes; service 1 is the propagation record
below. Every other value is an application service with an opaque payload: no
implicit decompression, no executable content, no security interpretation.

### 8.3 Propagation reporting — service 1

`>BQQhB`: version 1, UTC milliseconds since the Unix epoch, frequency in Hz,
signed transmitter power in hundredths of a dBm, locator length — then an
uppercase 4-, 6- or 8-character Maidenhead locator. The source identity is the
enclosing object's source. Power is the transmitter's own statement; no received
SNR calibration and no antenna EIRP is implied. A receiver should export its own
identity, receive time, frequency, SNR reference bandwidth, uncertainty and
decoder version alongside the record.

For the narrow-tone beacon's single status byte, the opt-in **propagation-v1
application profile** reads `0x80 | (power_dbm + 30)` for integer power −30 … +60
dBm. Apply that reading only when the application profile selects it; an
arbitrary status byte keeps its own meaning. The waveform and its sensitivity are
unchanged. Longer identities and richer reporting use the object service and pay
for it in airtime.

---

## 9. Station profile settings

Local settings determine advertised bandwidth, identification cadence and
compression policy. The reference ARQ defaults to a 2750 Hz profile class and
an identification interval of 540 seconds. A configured width below 1500 Hz
excludes the wider DATA profiles; narrow DATA profiles must be explicitly
advertised. Configuration is subject to the operator's applicable
operating rules.

The host's `SetProfile` command sets the presence-beacon profile byte only. It
does not apply identification, power or bandwidth policy. `Configure` settings
and the local station configuration control those behaviors. Session capability
selection reads the actual advertisement, not the beacon profile byte.

---

## KAT — normative known-answer vectors

The reference vectors. `tests/sabir/test_kat.py` pins the six session and beacon
images below, the FEC and record vectors, and the floor waveform;
`tests/sabir/test_release_profiles.py` pins the DATAGRAM image, and only its
first 20 bytes — the CRC is checked by round trip rather than by literal. The
payload-path vector is pinned by `test_kat_payload_codeword`. The
vectors use session `0x81` and stations `W1AW` / `K6XYZ`; on the air a session
identifier is a fresh random nonzero uint64, never derived from the
identities. Profiles 3–7 plus FASTCTL/PBACK/LOADING/DEFLATE give capability
word `0x0F0000F8`, carried big-endian at CONNECT bytes 12–15 and beacon
bytes 2–5. All current DATA profiles plus those four features give
`0x0F3F00F8`.

| Block | Image (hex) |
|---|---|
| CONNECT | `0102000000000000008100000f0000f85731415720202020202020204b3658595a202020202020200000b908` |
| CONNECT_ACK | `0202000000000000008100000f0000f84b3658595a202020202020205731415720202020202020200000cc2d` |
| CAPS (GEARSET profiles 256, 65535) | `0a0200000000000000811001040100ffff00000000000000000000000000000000000000000000000000b9fa` |
| DATA (gen 7, workhorse, cw 0–2) | `0302000000000000008100000007000004070000000000000000780300000000000000000000000000007fdf` |
| ACK (gen 7, cw 0–2, no SNR) | `04020000000000000081000000070000000700000000000000ffffffffffffffff000000000000000000f0d8` |
| BEACON (all-call, profile 1) | `30000f0000f85731415720202020202020200100f1cc` |
| DATAGRAM (profile 21, 250 symbols, 64 cw, 10048 bytes, token `0102030405060708`) | `0b01001500fa40002740010203040506070800009ea1` |

The DATA image reads `03` type, `02` wire version, session `…0081`, generation
`00000007`, profile field `000004`, bitmap `07` in byte 17, `n_syms` 120 and
`n_cw` 3 in the aux at bytes 25–27, offset zero. Setting HANDOVER on profile 21
gives the profile field `010015`.

**FEC vectors.** Each base matrix of §7.1.5, serialized row-major as `mb·nb`
big-endian int16 with `−1` for an absent entry:

| Code | SHA-256 of the serialized base matrix |
|---|---|
| r13 | `210ced6a426aa0f1bc6ec764d6ab937851f7fe17fb5ecce08f7fd7154e15c996` |
| r12 | `171607af29cb523f77a155e0fc4628474de4c56f15c26c9e7171b6390b8cbf59` |
| r34 | `10fb0b6eb57329ed60c20bc45752d1a7233bd900f651469a9d72b7faafb290cc` |
| r56 | `c7d59fb5be3b1c6807bfc1a0dd0ecefc00cb030613ed3ff86b609f22a89f9b38` |

Encoder vector, exercising §7.1.3 end to end: information vector = the first
`k` outputs of the PN9 sequence of §6.3.3 (bit *i* = output *i*). The codeword
is packed MSB-first into `n/8` bytes.

| Code | k | SHA-256 of the codeword | First 8 parity bytes |
|---|---|---|---|
| r13 | 512 | `06051825430b9dc0be1610ceca910c961294e7538ab834ce36c97987128021f3` | `5d0b03c5021036fd` |
| r12 | 512 | `1f0584c9d3597093f27d8a877b0026c4dabfb84f0aa0fcf81b50d838e2ff5f87` | `0eadfb52116d5204` |
| r34 | 768 | `c674868b36e7f768d7a6e21a4d2bf95efadd838559279234dc3e234c97583d13` | `2c5ca989216a5c4e` |
| r56 | 1280 | `ee50383886eb74ac71d366ab898ddd9dec2ad7f50d9d770be3b0481d3e99a1ad` | `3d0c51c9bdf792be` |

**Payload-path vector** (`test_kat_payload_codeword`). One r12 codeword carrying the chunk `b"sabir"`
through §6.3.2 → §6.3.3 → §6.3.4: block = `05 73 61 62 69 72` + 56 zero bytes +
CRC-16 `35ba` (64 bytes total); the whitened, interleaved 1024-bit codeword packed MSB-first
begins `c7125d58e367f069` and has SHA-256
`26c9d300077a29e620eb4596f2019b04fc36066aba5b2cc5bc62dab2f9684399`.

**Record vectors** (§6.3.1). The raw record for `b"sabir"` is
`1000002573616269726fbd6b834be756614d476c69400d4e27e26f73a31272aa24b18fd0401100a50e`
— flag `0x10`, length `0x25` = 5 body bytes plus the 32-byte digest. DEFLATE is
pinned in the decode direction, since any RFC 1951 encoder is conformant on the
wire: the record
`1100002a2b4e4cca2c5228462701bb26622cc6b52f31e5d9ad8daab9d9755417818f78e2f92bea58120a259c9de6`
delivers `b"sabir sabir sabir sabir"`. The record `020000057361626972` MUST NOT
deliver its body; the reference errors and drops the session.

**Waveform conformance.** A 44-byte session block on the **`floor` gear (×1)** is
418 560 samples (512 lead + 408 × 1024 + 256 tail) and demodulates byte-exact in
loopback; `floor2` is 822 016 samples and `floor4` 1 636 096. A 22-byte
connectionless block on `floor` is 216 832 samples (211 symbols). Each OFDM
profile has byte-exact loopback coverage. Current placement and channel
measurements are generated by `python -m hfmodem.sabir.sim.design`.

---

## Verification

- **Loopback:** every gear encodes→decodes byte-exact; the KAT vectors pin the
  wire and the reference waveform.
- **Channel simulation:** `sim/` implements the Watterson model (ITU-R F.1487) at
  Good / Moderate / Poor / NVIS-disturbed / polar profiles and sweeps throughput
  and FER vs SNR. Offset, drift and channel results are reported with their
  explicit test conditions in `measurements.json`.
- **End-to-end:** application-to-application transfer over the host interface
  through the impaired channel.
- **On-air:** the narrow-tone beacon sub-mode (§8) was transmitted on 40 m on
  2026-08-29 and decoded from two public receivers 103 and 302 miles away, by our
  own decoder reading their recordings. That establishes remote reception of that
  one frame family. Nothing else here has been transmitted: not the session
  layer, not any DATA profile, not the connectionless frames. A two-station
  exchange over a sound card and Hamlib PTT has not been performed.

The full suite (`python -m pytest`) is the executable conformance check.
`python -m hfmodem.sabir.sim.design` generates reproducible current-waveform
measurements; `python -m hfmodem.sabir.sim.release` reports complete-session
virtual-airtime goodput. Arithmetic cycle rates include body waveform, DATA
control, ACK and two modeled 250 ms turnarounds, but exclude setup, teardown,
retries and host record overhead. Session measurements must state whether their
numerator counts link bytes or application bytes and which overheads are included.
