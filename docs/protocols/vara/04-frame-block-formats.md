# 04 — Frame & block formats

The frame types VARA HF puts on the air, the callsign-keyed handshake tone
generators, the byte layouts of the link-setup and DATA containers, and block
sizing against speed level.

## 4.1 Frame taxonomy

The distinct waveform/frame types. The CR and the connect-response are a single
**MFSK PRNG-tone family** fully specified in §4.2; the connected-ack is a
**two-tone-per-symbol** burst on the same symbol grid (§4.2C); the link-setup and
payload overs are the **OFDM DATA waveform**
([`01 §2300`](01-physical-layer.md), [`03`](03-coding.md)).

| Frame type (label)     | Purpose | Waveform | Keyed by | Direction |
|------------------------|---------|----------|----------|-----------|
| connect-request (CR)   | session establishment; elicits `PENDING`/answer | MFSK, 10 preamble + 31 payload tones, ~1.74 s | **called** (destination) callsign | initiator→responder |
| connect-response       | responder accepts the CR | MFSK, 8 preamble + 15 payload tones, ~1.005 s | **called** callsign | responder→initiator |
| link-setup (caller-ID) | initiator sends the **caller** callsign + session params | **BW2300 rec3 wideband OFDM** — the ordinary DATA waveform: 371 data emission columns of 512 samples, one-hot over a 16-bin span, turbo-coded (§4.2A). Reported as `BITRATE (4) 175 bps TX` | **caller** callsign | initiator→responder |
| connected-ack          | responder confirms link up | **two tones per symbol**, 4 preamble + 7 state symbols, 11 symbols / 0.479 s (§4.2C) | nothing — carries no callsign | responder→initiator |
| DATA over              | payload transport (ARQ ISS→IRS) | OFDM DATA (500/2300/2750), 1–2 blocks/over | — | ISS→IRS |
| ACK over               | stop-and-wait ACK (ARQ IRS→ISS) | short CONTROL burst (~0.34–0.68 s) | — | IRS→ISS |
| keepalive              | idle link maintenance (~10–12 s) | short + ~1.4 s burst pair | — | both |
| turn release           | the holder hands the channel back; the peer answers by transmitting | MFSK, 2 preamble + 15 payload tones, 17 symbols / ~0.725 s | **called** callsign | both |
| disconnect             | ends the session; keyed once, unanswered | MFSK, 32 symbols / ~1.365 s | **called** callsign | the closing station |

## 4.2 MFSK handshake-tone generators (CR / connect-response)

The callsign-keyed handshake bursts share **one waveform family and one
payload-tone generator**, differing only by three scalars: `(preamble, N_payload,
SEED_OFF, PREADV)`. A builder can synthesize each burst, and a recognizer can
accept or reject one, from the callsign alone. All are keyed to the
**called/destination** callsign only and are caller-invariant; the caller's own
callsign rides the link-setup burst (§4.2A), which is an ordinary BW2300 rec3
wideband OFDM burst, not one of these bursts.

> **The connected-ack is not a member of this family** — see
> [§4.2C](#42c-connected-ack-step-5--two-tones-per-symbol-no-callsign).

### 4.2.1 Shared MFSK burst waveform (physical)

| Attribute | Value |
|-----------|-------|
| Modulation | single-tone-per-symbol MFSK; each symbol is one `−j` tone `= sin(2π·carrier·n/2048)` |
| Tone → frequency | `f = carrier · 48000/2048` Hz (the emitted 2048-pt @48 kHz FFT bin **equals** the tone/carrier index); LS-fit `f = 23.4361·carrier + 0.15`, residual ≤ 3 Hz |
| Sample rate | 48000 Hz |
| Per-symbol advance | **2048** samples @48 kHz, symbol start to symbol start (§4.4.1a). It is exactly `48000/23.4375`, the reciprocal of the tone spacing, so the tones are orthogonal over one symbol and the burst carries no clock of its own |
| Inter-symbol shaping | A raised-cosine WOLA cross-fade of length N: a symbol spans advance + N samples and adjacent symbols overlap by N. Rise `w[i]=(cos((i−0.5)π/32+π)+1)/2`, fall `w[i]=(cos((i−0.5)π/32)+1)/2`, i=0..31; flat 1.0 between. **This document does not fix N**; N = 32 is the default here, giving a 2080-sample symbol span. Whole-burst correlation is 0.9926 / 0.9932 / 0.9935 for N = 32 / 16 / 0, i.e. indistinguishable (§4.4.1a). Any N leaves the advance and the tone sequence unchanged, so a receiver never depends on it |
| Bandwidth mode | **BW2300** (the 31/15 payload tone values are the BW2300 tone set; BW2750/BW500 keep the tone counts and change the values) |
| Retry cadence | burst retried ≈ every 2 s until answered |

### 4.2.2 Per-burst structure

| Burst | Fixed preamble tones (callsign-independent) | Payload tones `N` | Total | Duration | `(SEED_OFF, PREADV)` |
|-------|---------------------------------------------|-------------------|-------|----------|----------------------|
| connect-request (CR) | `[74,68,70,60,60,77,50,76,78,74]` (10) | 31 | 41 | ~1.74 s | `(50, 1)` |
| connect-response     | `[62,67,55,66,59,72,68,55]` (8) | 15 | 23 | ~1.005 s | `(289, 511)` |
| connect-request, BW500  | same 10 as the CR | 31 | 41 | ~1.74 s | `(57, 1)` |
| connect-response, BW500 | same 8 as the response | 15 | 23 | ~1.005 s | `(289, 721)` |
| connect-request, BW2750  | same 10 as the CR | 31 | 41 | ~1.74 s | `(58, 1)` — on the **BW2300** alphabet |
| connect-response, BW2750 | same 8 as the response | 15 | 23 | ~1.005 s | `(579, 91)` |

**The BW500 handshake is the same waveform on a narrower alphabet.** Same preambles,
same symbol grid, same generator; the payload tones come from §4.2.3 with **base 50
instead of 29, `P ∈ {0,1}` instead of `{0..4}`, and no parity term** — 14 carriers on
the even bins 50..76 (1171.9–1781.3 Hz) against BW2300's 70 carriers on 23.4 Hz
spacing over 727–2273 Hz. No BW500 handshake burst can emit an odd carrier. A
responder armed at BW500 does not answer the BW2300 request, and the reverse.

Measured on five real VARA HF 4.9.0 BW500 connects (called W1AW, N0DX, W2XY,
KC2OUR). Each burst's payload pins one 24-bit generator state out of 2²⁴ and
regenerates it 31/31 and 15/15; because the seeding map is many-to-one, the
descriptors above are the **intersection over the four callsigns** — a singleton in
both cases, where any single callsign leaves 9–18 candidate pairs. A real VARA
raised `PENDING` on the synthesised BW500 request and declined one naming a
different station.

**The BW2750 request is the BW2300 waveform in a state of its own.** Same
preamble, same 70-carrier alphabet, `SEED_OFF` 58 where BW2300's is 50 and BW500's
57 — so a station listening at any of the three reads it, and the state behind the
preamble is what names the bandwidth asked for. A responder armed at BW2750
answers the BW2300 request as well and brings that session up at 2300. **The
response is already on BW2750's own alphabet**: §4.2.3 with **base 22 instead of
29 and `P ∈ {0..5}` instead of `{0..4}`**, parity kept — 84 carriers on bins
22..105 (515.6–2460.9 Hz). Every BW2750 carrier is a BW2300 carrier seven bins up
or down, up exactly when `Int(Rnd·6)` exceeds `Int(Rnd·5)` on the same draw.
Measured on the one BW2750 loopback (AAAA1 → BBBB2, 2026-07-21), both cables,
each burst pinning one 2²⁴ state under its own alphabet and none under the other;
the BW2300 session the same modems keyed 50 s later on the same tape is the
control. One session and one callsign pair.

The post-connect session bursts of [`05 §5.3.3`](05-arq-session-state-machine.md)
are further members with their own `(SEED_OFF, PREADV)`. The connected-ack is not a
member at all — §4.2C.

### 4.2.3 Payload-tone generator (closed form)

Keyed to the destination/called callsign (ASCII, uppercase, incl. any SSID chars).
All arithmetic is exact integer. `Rnd()` is the public Visual Basic 6 runtime PRNG.

```
# --- public VB6 Rnd LCG (msvbvm60), period 2^24 ---
LCG(s)      = (s · 0x43FD43FD + 0xC39EC3) & 0xFFFFFF          # Rnd value = s / 2^24
# --- public VB6 Randomize(seed) seeding map ---
hi32(x)     = high 32 bits of the little-endian IEEE-754 double of x
fold(hi)    = (((hi & 0xFFFF) << 8) ^ ((hi >> 8) & 0xFFFF00)) & 0xFFFFFF
start(seed) = (fold(hi32(seed)) & 0xFFFF00) | 0x86           # 24-bit LCG start state

# --- CRC & callsign hash ---
crc  = CRC-16/GENIBUS(callsign_ascii)          # poly 0x1021, init/xorout 0xFFFF, no reflection (spec/03 §3.4)
seed = (crc + SEED_OFF) & 0x7FFF
mult = G(callsign) + ((crc + 50) >> 15)        # NB: the "+50" carry is fixed for ALL three bursts

# --- seed the tone stream (PREADV pre-advance draws) ---
s = start(seed)
repeat (mult + PREADV) times:  s = LCG(s)

# --- per payload column k = 0 .. N-1 ---
for k in 0..N-1:
    s = LCG(s);  P = floor( (s / 2^24) · 5 )    # coarse tone,  P ∈ {0..4}
    s = LCG(s);  D = floor( (s / 2^24) · 7 )    # dither,       D ∈ {0..6}
    parity = 1 if k is even else 0
    bin[k] = (29 + parity + 14·P + 2·D) & 0xFF
```

At **BW500** the same loop runs with `P = floor((s / 2^24) · 2)`, no `parity` term,
and base 50 — `bin[k] = (50 + 14·P + 2·D) & 0xFF` — giving the 14-carrier alphabet
of §4.2.2. At **BW2750** it runs with `P = floor((s / 2^24) · 6)` and base 22 —
`bin[k] = (22 + parity + 14·P + 2·D) & 0xFF`, 84 carriers. Everything above it,
the LCG, the seeding map, the `+50` carry and `G`, is unchanged at both.

`G` is the GF(2)-linear callsign hash (only the **last three** characters matter;
earlier characters affect `mult` only through the CRC carry):

```
G(cs)      = 254 XOR T[-3](cs[-3]) XOR T[-2](cs[-2]) XOR T[-1](cs[-1])   # skip offsets past the string start
T[off](ch) = XOR over set bits i (i=0..5) of (ord(ch) XOR 0x41) of GBASIS[off][i]
GBASIS[-3] = [110, 220, 440, 338, 134, 310]     # bit index 0..5; bits 6..7 unused for A–Z/0–9
GBASIS[-2] = [102, 204, 408, 274,   6,  22]
GBASIS[-1] = [ 32,  64, 128, 258,  36, 216]
```

The full burst carrier sequence is `preamble ++ [bin[0..N-1]]`, rendered by §4.2.1.

**Callsign-string / SSID rule (input to `CRC-16/GENIBUS` and `G`).** The key fed to
both the CRC and the `G` hash is the **entire dialed callsign string, ASCII,
uppercase, byte-for-byte** — including embedded digits and any trailing SSID
character(s). There is no stripping, no base-36 recoding, and no separator handling
in this MFSK-tone path (the base-36 6-bit packing seen in the OFDM *connect frame* is
a different representation and does not apply here). Consequences:
- `G` depends only on the **last three** characters; earlier characters affect the
  tones solely through the CRC (hence `seed15` and the `mult` carry).
- Although `GBASIS[off]` lists only bits 0..5, the generator is **total over any
  ASCII character**: for the printable set (`A–Z`, `0–9`, `-`, `/`) bits 6..7 of
  `ord(ch) XOR 0x41` are a constant/ignored contribution (`GBASIS[off][6..7] = 0`),
  so digits and punctuation map cleanly through bits 0..5. This is why digit-bearing
  and trailing-digit callsigns work with the same six-element basis.
- **Digits are part of the hashed string, embedded or trailing.** The KAT set
  `K7ABC, W2XYZ, BBBB2, CCCC3, AAAA1` exercises both positions, and a single
  trailing-digit change (`BBBB2 → BBBB1`) changes the tones.
- **Hyphenated SSIDs: the literal string, hyphen included, is hashed.** A CR dialed
  to destination `BBBB2-7` carries the tones the generator produces for the
  **literal `"BBBB2-7"`** — all 31 columns. Stripped `"BBBB2"` agrees on 0 of the 31
  and de-hyphenated `"BBBB27"` on 1 of the 31. No stripping or reformatting: the
  `'-7'` (both the hyphen and the digit) is part of the hashed string. Consistent with
  the OFDM link-setup, where byte 9 is the CRC over the full `"W9SSJ-10"` literal.

### 4.2.4 Recognizer

An initiator that dialed `<called>` regenerates the expected payload bins (same
closed form) and matches them against the demodulated response to confirm step 2.
The match is exact: the correct callsign reproduces all 15 response bins (and all 31
CR bins), while a different callsign agrees on at most 2 of the 15, so an
exact-match test admits no false accepts. The alignment at which the payload bins
are read is the one the burst's fixed preamble locks.

#### 4.2.4a Known-answer test (KAT) vectors — payload tone bins

Concrete numeric outputs of the §4.2.3 generator, for cross-checking any independent
implementation. Each row is the **payload** tone-bin sequence (the callsign-independent
preamble of §4.2.2 is *prepended* to form the full carrier sequence; preambles are
CR `[74,68,70,60,60,77,50,76,78,74]` and connect-response `[62,67,55,66,59,72,68,55]`).
Values are FFT-bin indices (tone→frequency `f = carrier·48000/2048`).

| callsign | crc16/genibus | seed15 | mult | CR payload (31) |
|----------|---------------|--------|------|-----------------|
| K7ABC | 0x5734 | 22374 | 20 | `[36,53,86,69,52,85,76,61,98,49,64,71,68,73,52,61,74,85,90,45,46,89,44,37,82,83,36,61,38,43,32]` |
| W2XYZ | 0xE563 | 26005 | 279 | `[44,33,64,49,62,33,70,53,72,35,94,89,80,77,60,89,88,81,40,73,78,61,72,95,78,77,70,61,78,93,78]` |
| BBBB2 | 0x369E | 14032 | 122 | `[44,97,68,93,50,63,66,97,36,73,78,71,52,95,80,71,62,41,46,87,34,47,90,67,40,89,94,89,32,91,44]` |
| AAAA1 | 0x7FF0 | 34 | 3 | `[80,85,90,65,88,43,78,69,52,85,90,39,94,73,98,71,68,59,62,95,78,77,78,43,94,65,72,41,56,81,58]` |

| callsign | connect-response payload (15) |
|----------|-------------------------------|
| K7ABC | `[80,71,84,77,70,33,56,79,56,95,98,59,50,29,76]` |
| W2XYZ | `[46,67,88,83,54,69,58,47,30,83,40,29,84,75,32]` |
| BBBB2 | `[36,43,40,63,92,63,70,59,80,59,58,29,74,81,58]` |
| CCCC3 | `[66,53,58,53,84,97,52,81,90,35,76,69,44,77,76]` |

Full carrier sequence (preamble ++ payload) for one anchor:
- CR K7ABC (41): `[74,68,70,60,60,77,50,76,78,74, 36,53,86,69,52,85,76,61,98,49,64,71,68,73,52,61,74,85,90,45,46,89,44,37,82,83,36,61,38,43,32]`

## 4.2A Caller-ID link-setup frame (BW2300 rec3 OFDM)

The link-setup frame rides the **ordinary BW2300 rec3 wideband OFDM burst** — the
same waveform, coding chain and 92-byte container as a DATA over. There is no
separate narrowband burst. The burst is caller-**dependent**: it is where the caller
identity travels. In a BW2300 session the initiator's first wideband burst runs
4.406 s and decodes through the rec3 DATA chain to a CRC-clean 92-byte frame whose
first four bytes are `04 10 41 70` — the 6-bit packed caller callsign `AAAA1`.

rec3 is **one-hot**: one lit bin per emission column out of a 16-bin span
(`first_bin=9`, `span=16`, `bpc=4`). A view that does not span all 65 carriers sees
one lit bin per column and reads as narrowband single-tone 16-FSK with 1024-sample
symbols; the burst is in fact wideband, its symbols are 512 samples, and its 371
data emission columns are the data columns of a BW2300 base burst (395 total = 371
data + 24 reference, [`01 §2300`](01-physical-layer.md)) carrying 371 × 4 = 1484
coded bits exactly. Every initiator transmission occupies 735–2350 Hz; no narrowband
emission appears anywhere in a session.

**Frame recovery chain (RX)** — identical to the DATA path ([`03 §3.6.4`](03-coding.md)):
```
371 data emission columns → one-hot bin index → Gray (4 bits/column, MSB-first) → 1484 on-air bits
   → CHANNEL de-interleave:  coded[π[k]] = onair[k], π = interleave_stage1 col-3 [:1484]
   → rate-1/2 (13,15) turbo log-MAP decode  (N = 736 info bits; coded = 2N+12 = 1484;
        internal turbo interleaver = interleave_stage2 col-3 [:736], native — NOT pruned)
   → de-whiten:  info_bits XOR the BW2300 whitener PN  (MSB-first)  → 92-byte frame
   → CRC-16/GENIBUS over body[0:90] == frame[90:92] (big-endian)  → caller frame
```
The inner code is the **same (13,15) turbo family as the OFDM DATA path** ([`03 §3.6.4`](03-coding.md)),
here at **rate 1/2 with N = 736** (vs N = 368 for BW500 L4). It is selected by the
per-segment FEC descriptor, not a distinct encoder.

**Interleavers.** Between turbo encode and the σ symbol map VARA applies a **channel
interleave** `onair[k] = coded[π[k]]`, `π = interleave_stage1` column 3, first 1484
entries (a clean `0..1483` bijection). The turbo encoder's **internal** N=736
interleaver is the **native `interleave_stage2` column 3**
(`perm[i] = IL2[3+17·i][:736]`). Full details, the N-parametric generation rule, and
the end-to-end round trip through the complete σ⁻¹→de-interleave→turbo-decode→CRC
receiver are in [`03 §3.5.5`](03-coding.md).

**Frame byte layout** (90-byte body + 2-byte CRC). The `AAAA1` frame in full:

```
off  0.. 3   04 10 41 70   caller BASE callsign, 6-bit packed (SSID excluded)
off  4       00            constant
off  5       00            caller SSID as an integer, 0 if bare (W9SSJ-10 → 0x0a, W9SSJ-5 → 0x05)
off  6       00            constant
off  7       80            constant
off  8       14            constant
off  9       7f            high byte of CRC-16/GENIBUS(full callsign ASCII incl. "-SSID")  (0x7ff0 for AAAA1)
off 10..87   00 × 78       zero
off 88       04            constant
off 89       82            LIVE — varies per burst (block/trailer byte)
off 90..91   53 87         CRC-16/GENIBUS(body[0:90]), big-endian
```

The tail is not optional padding: CRC-16/GENIBUS over a body with bytes 10..89 all zero is
`0x2e89`, and with `88=04, 89=00` it is `0xe24d`. Only `88=04, 89=0x82` reproduces the
frame's `0x5387`, and byte 89 is the unique solution over all 256 values.

The same `14 <crc_hi> … 04 <block>` tail appears in a **short DATA over**: an over
carrying 67 payload bytes is followed by `14 7f`, 19 zero bytes, `04 82`. So bytes
8..89 are a structured, zero-filled trailer that VARA regenerates whenever the
payload does not fill the container — not an inert unused region, and
**clean zeros rather than stale buffer content**.

**At BW500** the same body fills one level-4 frame instead ([`03 §3.6.4`](03-coding.md)):
**44 body bytes + CRC-16/GENIBUS(body[0:44]) big-endian**, 46 bytes, on a 403-column
/ 4.31 s burst. Every offset above keeps its position — from the front for bytes
0..9, from the back for `04 82` — and only the zero run shortens, 78 bytes to 32:

```
off  0.. 9   as above (packed base call, SSID at 5, 80 14, callsign CRC high byte)
off 10..41   00 × 32       zero
off 42..43   04 82         constant, then the block/trailer byte
off 44..45                 CRC-16/GENIBUS(body[0:44]), big-endian
```

Measured on five real VARA HF 4.9.0 BW500 connects (W9SSJ → `5e44d328 0000 00 80 14 f6
… 04 82 1dee`; also K5ABC, KB1A, W9SSJ-10, VE7QRP), each regenerated byte-exact. The
SSID rule is unchanged across the two bandwidths: W9SSJ-10 gives byte 5 = `0x0a` and
byte 9 = `0x37` at BW500 exactly as at BW2300. A real VARA armed at BW500 answered
the synthesised frame with `CONNECTED <caller> <called> 500`, including for a caller
appearing in no recording.

### 4.2B DATA over frame layout

The 92-byte container is **90 body bytes + CRC-16/GENIBUS(body[0:90]) big-endian**,
as for the link-setup frame, but the body is laid out differently:

```
FULL over  : body[0..88] = 89 payload bytes; body[89] is a per-frame FIELD
SHORT over : body[0..n-1] = n payload bytes
             body[n]      = 0x14                 trailer marker
             body[n+1]    = high byte of CRC-16/GENIBUS(own callsign ASCII)
             body[n+2..87]= 0x00
             body[88..89] = two small fields (observed 01 02 and 03 02;
                            the link-setup frame carries 04 82 here)
```

In a session called by `W9SSJ`, a 61-byte and a 43-byte short over from the gateway
both carry `0x14 0xf6`, where `crc16_genibus(b"W9SSJ") >> 8 == 0xf6` — the trailer
carries the **caller's** identity, not the gateway's, so it identifies the session
rather than the sender of the frame. A 90-byte over carries payload in every body
byte including 88/89, so those two positions are trailer fields only when a trailer
is present. This is the `14 <crc_hi> … 04 <block>` tail of §4.2A with the
`<crc_hi>` field identified.

**The layout is the level's, not the bandwidth's.** The container is always the
speed level's own frame and the body is that frame less its CRC, so one law covers
every level of every bandwidth: payload from offset 0, the per-frame field in the
last body byte, and the trailer laid out from the END of the body when the over is
short.

| bandwidth | level | body | payload | field | trailer at |
|---|---|---|---|---|---|
| BW2300 | rec3 / `BITRATE (4)` | 90 | 89 | body[89] | body[88..89] |
| BW2300 | rec2 / `BITRATE (3)` | 48 | 47 | body[47] | body[46..47] |
| BW500 | L4 / `BITRATE (4)` | **44** | **43** | body[43] | body[42..43] |
| BW500 | rec2 / `BITRATE (3)` | **35** | **34** | body[34] | body[33..34] |

Measured on a stock VARA HF 4.9.0 BW500 pair, `W9SSJ` calling `W1AW`, 2026-08-30,
one virtual cable per direction: the responder was handed 200 bytes and delivered
them in five overs, four full level-4 bodies of 43 payload bytes apiece — per-frame
fields `95`, `91`, `8d`, `81` — and a closing record-2 body of 28 payload bytes
ending `14 f6 00 00 00 04 82`. `0xf6` is again the CALLER `W9SSJ`, keyed by the
responder. The delivery steps its host 43/86/129/172/200 and every byte is
`00`–`c7` in order, so nothing is inferred about where a payload boundary fell.

**The over that closes a delivery drops a record, and the field byte announces
it.** Eight whole deliveries a stock 4.9.0 keyed on its own cable at BW2300 — six
sessions, both stations, both directions, the record read off the waveform by the
reference-column score and each over then decoded at it — pair the closing over's
record with the per-frame field of the full over in front:

| field on the last full over | closes at | deliveries |
|---|---|---|
| `0x81` | **record 2** / `BITRATE (3)` | 6 |
| `0x89` | base level / `BITRATE (4)` | 2 |
| `0x81`, keyed to this station | base level / `BITRATE (4)` | 3 |

The score is never close — 24 of 24 reference columns at the over's own record
and 7 or fewer at the other, against 9 of 24 for noise — and no delivery mixes
the two. The body is the same at either record: an empty close is `14 f6 00 … 00
04 82` laid out over 90 bytes and over 48, so the record is the only thing that
moves. A delivery that fits one short over closes at the base level, with no
full over in front of it to announce anything. The BW500 delivery above is the
same rule at that bandwidth: four level-4 bodies behind `95 91 8d 81`, then the
record-2 close. The pair `0x81` followed by a base-level close is one no
recording holds, and a stock responder handed it answers with the continue burst
and goes on waiting for a message that has already ended — 93 bytes as one full
over and a 4-byte close read 89 of 93 with the close at the base level and 93 of
93 at record 2, and 182 and 534 read byte-exact the same way.

The third row is the limit of the rule. Three deliveries a stock responder keyed
to this station on 2026-09-03 — the same 93-byte reply in three sessions — carried
`0x81` on the full over and closed at the base level, and our receiver read all
three 93 of 93. So the byte is not the announcement on its own, and what every
delivery on tape agrees on is narrower: a receiver reads the close at whichever
record it was keyed, and a close this station keys at record 2 behind `0x81` is
read where a base-level one behind the same byte was not.

**Implementation note — kestrel diverges here.** `kestrel/arq/phy.py` reserves
body[0] for its own control marker and therefore offers 89 payload bytes per over.
VARA uses all 90 and puts payload at offset 0. This is not what VARA specifies; a
kestrel DATA over will not parse at a real gateway until the marker is dropped and
the short-over trailer above is emitted instead.

**Receiver behaviour of the zero region.** The 78-byte zero region is not content the
responder checks. Varying the frame and watching the responder:

| body bytes 10..87 | responder |
|---|---|
| unmodified (control) | `CONNECTED` |
| all 78 bytes random | `CONNECTED` |
| 8-byte magic, 16/20/24/32-byte blocks, `0x00`–`0xff` singles | `CONNECTED` |
| **any single byte equal to `0x14`** | `PENDING`, then nothing |
| byte 88 `04` → `05` | `CONNECTED` |
| byte 8 `14` → `15` | `PENDING`, then nothing |

So the 78-byte region is **fully writable — 624 bits — with one constraint: the byte value
`0x14` must not appear in it.** `0x14` is the marker that opens the trailer structure at
offset 8 (and reappears at the payload end of a short DATA over), and a stray one in the
region mis-frames the parser. Neighbouring values `0x13`/`0x15` are accepted, as are
`0x04`, `0x7f`, `0x80`, `0x28`, `0xff` — only `0x14` rejects. Rejection is loud and
immediate: the responder answers the connect-request and then declines, never completing.

| Field | Value |
|-------|-------|
| Caller callsign | **6-bit packed, MSB-first, from bit 0**: `A–Z → 1–26`, `0–9 → 27–36`, `0` = terminator (e.g. `AAAA1` = `1,1,1,1,28`). BASE call only — the SSID is **not** packed here |
| Caller SSID | **body[5], plain integer** (0 if bare). The 6-bit field carries the base call unchanged; byte 9 is the CRC of the *full* `CALL-SSID` string (hyphen included). KAT: `W9SSJ-10` → byte5 `0x0a`, byte9 `0x37`; `W9SSJ-5` → byte5 `0x05`, byte9 `0x43` |
| Session params | remaining body bytes (bandwidth / session config), zero in the frames given here |
| Trailer | **CRC-16/GENIBUS**(body[0:90]), 2 bytes, **big-endian**, init=`0xFFFF` xorout=`0xFFFF` |
| Whitener | the additive BW2300 whitener PN, XORed over the 736 info bits before turbo (MSB-first bit order) |
| Frame CRC (KAT) | `AAAA1` → `0x5387`, `K7ABC` → `0x9aca`, `W2XYZ` → `0x9de4` (the frames are caller-dependent) |

## 4.2C Connected-ack (step 5) — two tones per symbol, no callsign

The burst a responder keys at the instant it emits `CONNECTED <caller> <called> <bw>`.
It rides the §4.2.1 symbol grid — 2048-sample advance, 48 kHz, the same 23.4375 Hz
carrier lattice — and nothing else about it belongs to §4.2: **every symbol lights two
equal-amplitude carriers**, it is 11 symbols long, and no part of it depends on either
callsign.

| Attribute | Value |
|-----------|-------|
| Modulation | **two tones per symbol**, equal amplitude; per-symbol 2048-pt FFT shows two peaks of ratio 1.00 each with its own ±1-bin Hann skirt and nothing else |
| Length | **11 symbols** = 22560 samples = **0.479 s** of audio; PTT window 0.446–0.492 s |
| Fixed preamble | the first **four** symbols, invariant: `{64,67} {56,74} {64,69} {68,78}` (1500/1570.3, 1312.5/1734.4, 1500/1617.2, 1593.8/1828.1 Hz) |
| Remaining 7 symbols | two tones each, varying by session and by bandwidth. They carry session state, and **this document does not specify their encoding.** They are not drawn from the §4.2.3 generator: an exhaustive search of all 2²⁴ VB6 `Rnd` states, over every payload start index 0..5 and both parities, returns **no** solution for any ack, where the same solver recovers the connect-response's state uniquely |
| Bandwidth dependence | **none in the preamble.** BW500, BW2300 and BW2750 all open on the same four pairs. Only the state symbols differ, and at BW500 they stay inside 52..74 where the wide modes spread 31..98 |
| Callsign dependence | **none.** The same four pairs open the burst at every bandwidth, off air and in loopback; nothing in the burst is a function of either callsign |

**Whole frames** (tone pairs, `(lo,hi)`). Because the state field's encoding is not
specified here, a transmitter needs concrete values to put there:

| frame | symbols 0..3 (preamble) | symbols 4..10 (state) |
|---|---|---|
| BW2300, loopback | `(64,67) (56,74) (64,69) (68,78)` | `(52,86) (83,87) (46,82) (59,77) (56,58) (69,75) (52,66)` |
| BW2300, off air | same | `(34,80) (83,97) (50,66) (59,95) (64,96) (59,77) (56,72)` |
| BW2750, loopback | same | `(45,93) (76,94) (39,75) (52,84) (49,51) (54,76) (59,75)` — keyed twice, at the link-setup and at the session-confirm |
| BW2750, loopback, answering the first keepalive | same | `(31,49) (44,86) (45,57) (54,102) (37,51) (66,98) (69,83)` |
| BW2750, off air | same | `(41,87) (35,90) (43,73) (44,66) (60,71) (66,84) (63,65)` |
| BW500, loopback | same | `(58,64) (52,72) (54,74) (56,68) (58,72) (54,68) (58,68)` |

The off-air rows are read through a fading HF path and their state symbols are the
least trustworthy part of this table; the preamble is what every path agrees on.

**At BW500 the state symbols are the link's, not the bandwidth's.** Two stock 4.9.0
BW500 sessions with different callers, every burst attributed by which one-way
cable holds it — 71 keyings, and re-synthesised from the symbols below each
correlates 0.990–0.999 with the audio it was read from:

| link | who keys it | occasion | symbols 4..10 (state) |
|---|---|---|---|
| `AAAA1` → `BBBB2`, 2026-07-13 | responder | the link-setup, the confirm, the LAST data over | `(58,64) (52,72) (54,74) (56,68) (58,72) (54,68) (58,68)` |
| | responder | each idle keepalive, and the disconnect request | `(58,72) (56,76) (54,68) (56,70) (58,68) (58,70) (52,68)` |
| | responder | the initiator's turn-request — the grant, and nothing else in 478 s | `(56,68) (60,68) (50,72) (54,68) (60,76) (60,68) (62,76)` |
| `W9SSJ` → `W1AW`, 2026-08-30 | responder | the link-setup | `(54,72) (56,70) (56,76) (54,66) (58,72) (54,76) (50,72)` |
| | **caller** | the responder's LAST data over | `(52,74) (56,76) (52,64) (56,68) (56,74) (58,68) (60,68)` |
| | **caller** | the idle cadence, ×8 at 12.06 s | `(58,76) (58,70) (52,76) (50,68) (54,64) (52,74) (54,66)` |

The two responders are in the same role answering the same occasion and share one
symbol of seven, so the state field is a function of the link and the tails of one
caller do not serve another — the same key the BW2300 burst was shown to carry
across eight sessions. Role is real as well: a caller and a responder of one link
share nothing.

**The intermediate over draws a shorter two-tone burst.** Only the LAST over of a
delivery is answered with the 11-symbol frame; every over with another behind it
draws an **8-symbol** one, 0.341 s, two tones throughout, whose FIRST symbol is the
11-symbol burst's own opening pair and whose remaining seven are on the bandwidth's
carriers:

| link | who keys it | copies | symbols |
|---|---|---|---|
| `AAAA1` → `BBBB2`, 2026-07-13 | responder | 50, bit-identical | `(64,67) (60,64) (52,68) (56,64) (58,70) (54,68) (60,74) (50,70)` |
| `W9SSJ` → `W1AW`, 2026-08-30 | caller | 3 | `(64,67) (50,66) (60,64) (50,68) (58,76) (52,76) (58,76) (60,68)` |
| | caller | 1, the over before the delivery's last | `(64,67) (62,76) (58,76) (50,68) (50,72) (60,74) (50,68) (52,76)` |

The seven behind the lead carry something this document does not pin: nothing that
varied across the 0830 delivery separates its two forms.

**Timing.** The responder's PTT window opens on the `CONNECTED` line and closes
0.446–0.492 s later, and the burst inside it is this one. Off air the same burst
appears 0.2–0.4 s after the initiator's link-setup over ends, and nowhere else.

**RX use.** The ack cannot be checked against the station dialed, because nothing
in it depends on a callsign. What identifies it is the four preamble pairs — eight
carriers — plus the fact that it arrived in the turnaround window that answers the
receiver's own link-setup. The test is specific: a genuine ack matches over 46–63
consecutive 32-sample alignments, while real off-air HF holding no ack produces no
preamble match at any alignment.

**The responder also keys this same 11-symbol frame later in the session** — answering
the initiator's step-6 confirm and its per-over responses — with the same preamble and
different state symbols. So the preamble identifies the frame, not the step; the step
comes from where in the exchange it arrives.

## 4.3 Block sizing vs speed level

Net host bytes cleared per ACKed block at 500 Hz, as reported by the host `BUFFER`
ledger. This document does not specify levels 0–2, nor the registered levels above
4. See [`06`](06-speed-gearshift.md).

| Speed level | Net host bytes / block | Blocks / over |
|-------------|------------------------|---------------|
| 3 (`61 bps`) | **34** | 1–2 (grows on success) |
| 4 (`88 bps`, free cap) | **43** | 1–2 (over 4.31 s → 8.51 s, ×1.94) |

## 4.4 Provisional facts, cautions & open questions

The implementer-ready facts are in §4.1–§4.3 and §4.2A. This section carries what
this document leaves unspecified, and the readings of the waveform that mislead.

### 4.4.0 500 Hz DATA byte layout

The **link-setup** is specified, at §4.2A: a 44-byte body plus the big-endian
CRC-16/GENIBUS trailer, filling one level-4 frame, and identical to the BW2300
body of §4.2A but for the length of the zero run between byte 9 and the `04 82`.
Five real VARA HF 4.9.0 BW500 connects (callsigns W9SSJ, K5ABC, KB1A, W9SSJ-10,
VE7QRP) regenerate byte-exact including the CRC, and a real VARA answered the
synthesised frame with `CONNECTED <caller> <called> 500`.

**Payload** overs are specified at §4.2B: the body is 44 bytes at level 4 and 35
at record 2, laid out by the wide bandwidth's own law, and a stock pair's 200-byte
delivery reassembles from it byte for byte. What is still unspecified is the
per-frame field the last body byte carries — it moves per over — and any length or
sequence field inside the trailer's zero run. A session running `COMPRESSION TEXT`
([`08`](08-compression.md)) also puts a compressed and framed transform of the host
payload on the air, so two overs carrying the same payload need not be
bit-identical.

An earlier revision declared the whole layout unestablished because "the 500 Hz
soft symbols carry no readable constellation", citing `03 §3.5`. That premise is
retracted — §3.5's single-band 256-point model is superseded by §3.6, which is
round-trip-validated and byte-exact on real VARA audio.

### 4.4.1 The CR is MFSK, not OFDM

Read through a clipped analysis window, the CR looks first like a set of 9 discrete
carriers and then like a contiguous twiddle-scrambled OFDM band. It is the plain
**single-tone-per-symbol MFSK** of §4.2: 10 fixed preamble tones
`[74,68,70,60,60,77,50,76,78,74]` + 31 callsign-keyed payload tones, one tone every
2048 samples, BW2300. The tone→frequency map `f = carrier·48000/2048` fits 93
(carrier,freq) pairs with residual ≤ 3 Hz.

### 4.4.1a Per-symbol advance — 2048 exactly

The advance is exactly 2048 samples, symbol start to symbol start. Three things
corroborate it structurally, and each fails at 2042:

- **Orthogonality.** 2048 = 48000/23.4375 is the exact reciprocal of the tone spacing,
  so one symbol spans a whole number of cycles of every tone in the set. This is
  ordinary orthogonal MFSK; 2042 is not a special number for any tone.
- **Start phase.** Under a global phase reference, real symbol start phases are
  constant across a burst (vector strength ≥ 0.99), which can only happen when the
  advance is an exact multiple of 2048.
- **Whole-burst correlation** against real gateway audio, jointly optimised over start
  and frequency offset, rises from 0.9596 to 0.9926 for a connect-request and from
  0.9244 to 0.9368 for a connect-response (per-symbol projection 0.9675 → 0.9830)
  when the advance is corrected from 2042 to 2048. A 41-symbol CR at 2042 ends 246
  samples early and its last symbol sits 11.7 % of a symbol off.

An advance of 2042 arises from taking a 2074-sample stride and subtracting an assumed
cross-fade length, which no observation of the waveform constrains. Being clock-free,
the advance cannot be explained away as a sound-card rate error: a receiver that
mis-times it loses matched-filter output but not the tone identities, which is why a
2042 advance demodulates real bursts and stays invisible.

**The cross-fade length is not specified here.** Whole-burst correlation is 0.9926 /
0.9932 / 0.9935 for N = 32 / 16 / 0 — nothing distinguishes them, and nothing can,
because the overlap is 0.7 ms of a 1.75 s burst. §4.2.1 carries N=32 as its default;
the advance does not depend on it.

### 4.4.2 CR generator — properties of the callsign map

The payload law is `bin[k] = (29 + parity[k] + 14·P + 2·D) & 0xFF` with `P` a coarse
tone `{0..4}` and `D` a dither `{0..6}`, both drawn from the public VB6 `Rnd` LCG
(2 draws/column). The seed derives from `CRC-16/GENIBUS(called)` via the VB6
`Randomize` seeding map (`fold`/`start` in §4.2.3). Two properties:
- **Tone stream:** there is **no callsign FEC** in the tones — `P` is a plain `Rnd`
  draw interleaved with the dither; the only callsign payload is `seed15` (15 bits of
  the destination CRC) + `mult`.
- **`mult` (RNG pre-advance):** a **GF(2)-linear hash `G` of the last three callsign
  characters** (basis in §4.2.3), XOR-linear per character.

### 4.4.3 Responder reaction to a CR

A station whose `MYCALL` matches the CR's called callsign answers the burst with the
host sequence `BUSY ON → PENDING → PTT ON`, once per retried burst
([`07`](07-host-api-mapping.md)). Nothing in the CR beyond the called callsign
affects this, so a CR built entirely from §4.2.3 elicits it.

### 4.4.4 Connect-response (step 2)

A member of the same MFSK family: 8 preamble tones `[62,67,55,66,59,72,68,55]` + 15
payload tones, ~1.005 s, same column law, `(SEED_OFF, PREADV) = (289, 511)`. It is
keyed to the **called** callsign alone — different callers dialing the same
destination elicit byte-identical responses.

### 4.4.5 Burst classification, and the two readings that mislead

A per-symbol native-2048-point FFT separates the burst kinds by the ratio of the
2nd-strongest to the strongest carrier: ≈0.50 for single-tone MFSK, ≈0.85–0.97 for
OFDM, 1.00 for the two-tone connected-ack. Burst inventory of a BW2300 connect:

| # | Dur | 2nd/1st | Type | Keyed by | Role |
|---|-----|---------|------|----------|------|
| 1 | 1.78 s | 0.51 | MFSK | called | CR (§4.2) |
| 2 | 1.01 s | 0.51 | MFSK | called | connect-response (§4.2) |
| 3 | 4.37 s | **0.90** | **OFDM** | **caller** | **link-setup / caller-identity DATA frame** |
| 4 | 0.48 s | 1.00 | **two-tone** | nothing | connected-ack (§4.2C) |

- **The connected-ack is not single-tone MFSK.** Read as single-tone, each of its
  symbols returns one of its two carriers, and which one depends on the alignment —
  which is how a single-tone tone-map model can be fitted to it and look clean. The
  ratio test above separates them: the ack's 2nd/1st carrier ratio is **1.00**, not
  the ≈0.50 of MFSK, because both carriers are lit. Its specification is §4.2C.
- **The link-setup is the OFDM DATA waveform, not a 4th MFSK burst.** The
  `BITRATE (N) … bps TX` burst is multi-carrier OFDM (2nd/1st ≈ 0.9; **65 contiguous
  carriers @23.4 Hz across 797–2297 Hz**, 100 symbols / 4.37 s) and is **keyed to the
  caller callsign** (identical for one caller across different destinations; differs
  by caller). This is the **only** burst carrying the caller identity — the CR and
  response are keyed to the called callsign and the connected-ack names nobody at
  all. **Consequence:** a responder cannot learn the caller from the MFSK handshake;
  it must demodulate and FEC-decode this OFDM data frame (§4.2A,
  [`03 §3.5.3`](03-coding.md)).

### 4.4.6 Link-setup frame recovery

The caller frame is recovered from ordinary received audio through the rec3 DATA
chain, CRC-clean, with the caller callsign 6-bit-packed at the frame start and any
SSID in `body[5]` ([§4.2A](#42a-caller-id-link-setup-frame-bw2300-rec3-ofdm)). No
access to the pre-WOLA cell grid is needed for it. That grid is a sub-0.1 ms
transient and the audio is WOLA-non-invertible — both true, and neither of them
stands between the received audio and the frame.

### 4.4.7 Open items

- **500 Hz DATA payload overs** (§4.4.0) — the link-setup is specified; the
  length and sequence fields of a payload over are not.
- **Connected-ack state symbols** (§4.2C) — the encoding of symbols 4..10 is not
  specified.
- **ARQ seq/ACK/NAK field encoding** (§4.1 ACK over) — not specified; the fields ride
  the CONTROL-burst waveform (see [`05 §5.7`](05-arq-session-state-machine.md)).
