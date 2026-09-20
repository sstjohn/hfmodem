# ARDOP Physical Layer — Waveform, Framing, FEC (functional record)

Compiled from public documents only: the **ARDOP Specification, Rev 2.0, 2017‑11‑27**
(Rick Muething KN6KB; its Appendix B/C frame
worksheets are the numeric backbone here) and two open reference implementations —
**ardopcf** (git `a7c9228`, MIT, © 2014‑2024 Muething/Wiseman/LaRue) and the
**M0LTE.Ardop** managed port (AGPL‑3.0‑or‑later, itself a cited port of ardopcf).
No reverse‑engineering of captured signals was performed; every fact below is read
directly from the spec PDF or the published source. Provenance is marked per fact as
`[spec-pdf]` or `[code]` with a `file:line` citation. **Where spec and code disagree,
the code is the operative wire format** and both values are recorded. This record is
sufficient to build an interoperable modem; it deliberately contains no source code.

Citations use these roots:
- `spec` = `ardopcf/docs/refs/ARDOP_Specification_20171127.pdf`
- ardopcf sources = `ardopcf/src/common/*` and `ardopcf/lib/rockliff/rrs.c`
- M0LTE = `M0LTE.Ardop/src/M0LTE.Ardop/*`

---

## 1. Global parameters

| Parameter | Value | Provenance |
|---|---|---|
| Audio sample rate | **12000 samples/s** (mono) | `[spec-pdf]` spec §4.2 "as low as possible (currently 12000 samples/sec)"; `[code]` `Modulate.c:311` `12000/intBaud`, `ARDOPCommon.c:536` rejects any WAV whose rate ≠ 12000 |
| Sample format | signed 16‑bit PCM, one real channel | `[code]` `Modulate.c` `SampleSink(short)`; templates are `short` |
| Sample‑rate tolerance | ±1000 ppm accommodated, ±100 ppm preferred | `[spec-pdf]` §4.2 |
| Audio centre frequency | **1500 Hz** for all control frames and single‑carrier data; passband filtered around 1500 Hz | `[code]` `Modulate.c:182‑184,533‑539` `initFilter(BW,1500)` |
| Occupied bandwidths | **200 / 500 / 1000 / 2000 Hz**, measured at the −26 dB points | `[spec-pdf]` §2.2, §6 |
| Max symbol rate (HF/SSB) | ≤ 300 baud (50 & 100 used on HF; 600 baud is VHF/UHF‑FM only) | `[spec-pdf]` §2.5 |
| Crest factor (PAPR) target | ~1.5 to 3.5 (pure sine = 1.41) | `[spec-pdf]` §2.4; empirical per‑mode scaling in `Modulate.c:505‑526` |
| Default leader length | **240 ms** (12 × 20 ms symbols); auto‑timing may shorten to 100 ms (5 symbols) or lengthen to 1000 ms (50 symbols) | `[code]` `ARDOPC.c:98` `LeaderLength = 240`; `[spec-pdf]` App. B "Leader" (5–50 symbols) |
| Default trailer length | **20 ms** | `[code]` `ARDOPC.c:99` `TrailerLength = 20` |
| Leader/sync symbol length | 20 ms (50 baud) | `[spec-pdf]` App. B; `[code]` `Modulate.c:82` |
| ARQ round‑trip timing budget | ARQ throughput figures assume 160 ms leader + 200 ms ACK + ~400 ms total timing/guard gap per turn | `[spec-pdf]` App. C notes 2 & 3 |
| VOX/keying latency tolerated | up to ~300 ms | `[spec-pdf]` §4.4 |

All modulation is generated from precomputed 16‑bit sample **templates** (one symbol
each). A per‑symbol raised‑cosine window (`sin(π·k/119)`, the spec's "Hamming"
cyclic‑prefix/guard entry) is applied to the PSK/QAM templates; 4FSK templates use no
envelope shaping beyond a 1.1× amplitude factor. `[code]` `CalcTemplates.c:409`, `:212`.

---

## 2. Frame‑type catalog and the frame‑type‑byte encoding

### 2.1 How a frame identifies itself on the air

Every frame begins with a **leader**, one **leader‑sync** symbol, then a **frame‑type
header** carried in 4FSK at 50 baud. The header is **10 symbols = 2 bytes + 2 parity
symbols** `[spec-pdf]` §7.1.3, App. B. Precisely, the header is emitted as two
5‑symbol groups `[code]` `Modulate.c:90‑112`:

```
[4 symbols: frame-type byte, 2 bits/symbol, MSB-pair first]  [1 symbol: parity]
[4 symbols: (frame-type XOR SessionID)]                      [1 symbol: parity]
```

- **First byte** = raw frame type. **Second byte** = frame type **XOR the 8‑bit
  Session ID** `[code]` `Modulate.c:97`, `ARDOPC.c:1145`. This binds the frame to a
  session so adjacent sessions on the same frequency don't cross‑contaminate.
  Unconnected / FEC / ConReq / Ping frames force **Session ID = 0xFF** (so the XOR is a
  bitwise complement) `[spec-pdf]` §7.1.3, §7.2.1; `[code]` `ARDOPC.c:1376,1412`.
- **Session ID derivation:** 8‑bit CRC of the ASCII concatenation
  `caller_callsign + target_callsign` (canonical `CALL` / `CALL-SSID` forms), with a
  result of `0xFF` remapped to `0x00` (0xFF is reserved). CRC‑8 poly constant `0xC6`,
  init `0xFF`, MSB‑first with LSB data injection `[code]` M0LTE `ArdopCrc.cs:83‑121`
  (ports ardopcf `ARQ.c:200,507`). `[spec-pdf]` §7.2.1 describes it as "an 8‑bit CRC
  hash of both call signs."

### 2.2 The parity quirk (precise scheme)

The two parity symbols are **not** a per‑byte parity — **both parity symbols carry the
same value**, computed **only from the raw frame‑type byte** (`bytEncodedBytes[0]`),
even the one appended to the XOR'd second byte `[code]` `Modulate.c:98‑99`.

`ComputeTypeParity(b)` `[code]` `ARDOPC.c:1640‑1655`:

1. Split byte `b` into its four 2‑bit symbols (`s0 s1 s2 s3`, MSB pair first).
2. `parity = 1 XOR s0 XOR s1 XOR s2 XOR s3`.
3. Return `parity & 0x3` — a single 2‑bit 4FSK symbol.

So a receiver recovers the frame type from 4 tone symbols, re‑derives the same 2‑bit
parity, and uses **minimum‑distance soft decoding** across the redundant copies for
robustness `[spec-pdf]` §7.1.3.

The frame‑type LSB is separately the **Even/Odd bit** (`blnOdd = frameType & 1`)
`[code]` `ARDOPC.c:749`. Data and some control frames are allocated in E/O pairs
(e.g. `0x48`/`0x49`) so a receiver can distinguish a *repeat* of a frame from *new*
data of the same mode.

### 2.3 Frame‑type number map

Up to 256 frame types; ACK and NAK each consume a 32‑code block (5 quality bits)
`[spec-pdf]` §7.1. `[code]` `ARDOPC.h:330‑367`:

| Range (hex) | Group | Frame(s) | Purpose |
|---|---|---|---|
| `00`–`1F` | Control (ACK/NAK) | **DATANAK** (32 codes, low 5 bits = decode quality 0–31 → Q 38…100) | IRS: frame type/ID decoded but data CRC failed after all FEC/averaging `[spec-pdf]` §7.2.3.1 |
| `23` | Control | **BREAK** | IRS signals intent to become ISS (take the link) `[spec-pdf]` §7.2.3.3 |
| `24` | Control | **IDLE** | ISS has no data; IRS answers ACK (keep idling) or BREAK `[spec-pdf]` §7.2.3.4 |
| `29` | Control | **DISC** | Disconnect request; answered by END `[spec-pdf]` §7.2.3.12 |
| `2C` | Control | **END** | Confirms session end `[spec-pdf]` §7.2.3.5 |
| `2D` | Control | **ConRejBusy** | Reject connect — channel busy (BUSYBLOCK) `[spec-pdf]` §7.2.3.11 |
| `2E` | Control | **ConRejBW** | Reject connect — incompatible bandwidth `[spec-pdf]` §7.2.3.11 |
| `30` | ID | **IDFRAME** | Callsign + grid‑square ID (no ACK); auto‑sent ≥ every 10 min `[spec-pdf]` §7.2.3.13 |
| `31`–`34` | Connect | **ConReq200M/500M/1000M/2000M** | Connect request, MAX bandwidth 200/500/1000/2000 Hz `[code]` `ARDOPC.h:344‑347` |
| `35`–`38` | Connect | **ConReq200F/500F/1000F/2000F** | Connect request, FORCED bandwidth `[code]` `ARDOPC.h:348‑351` |
| `39`–`3C` | Control | **ConAck200/500/1000/2000** | Connect acknowledge for the negotiated BW; carries received‑leader timing `[spec-pdf]` §7.2.3.6‑9 |
| `3D` | Control | **PINGACK** | Answers PING with S/N + constellation quality `[spec-pdf]` §7.2.3.10 |
| `3E` | Special | **PING** | Path/propagation probe (200 Hz BW, ~1.9 s), DISC state only `[spec-pdf]` §2.11, §7.2.3 |
| `40`–`7D` | Data | **Data frames** (see §4) | Payload transport; ARQ or FEC. Some codes in the range are unused. |
| `E0`–`FF` | Control (ACK) | **DATAACK** (32 codes, low 5 bits = decode quality) | IRS: frame type/ID **and all data** decoded OK `[spec-pdf]` §7.2.3.2 |

Quality encoding for DATAACK/DATANAK: `Q = 38 + 2·(code & 0x1F)`, range 38…100 in
steps of 2; ≤ 38 all map to 38 `[code]` `Modulate.c:165`. Q ≥ 60 is normally required
for reliable decoding `[spec-pdf]` §7.2.3.1‑2.

---

## 3. The 4FSK control frames

All control, connect, ID, ACK/NAK, ConAck, Ping and Ping‑Ack frames are **single
carrier, 4FSK, 50 baud** `[code]` `ARDOPC.c:747‑832`, `FrameInfo()`.

| Attribute | Value | Provenance |
|---|---|---|
| Carriers | 1 | `[code]` `ARDOPC.c:750` |
| Modulation | 4FSK (2 bits/symbol) | `[code]` `ARDOPC.c:753` |
| Symbol / baud rate | 50 baud → 20 ms/symbol → **240 samples/symbol** | `[code]` `Modulate.c:198` |
| Tone set (200 Hz BW) | **1425, 1475, 1525, 1575 Hz** — symbols 0,1,2,3 | `[code]` `CalcTemplates.c:111` |
| Tone spacing | **50 Hz**, centred on 1500 Hz | derived from tone set |
| Byte→tone mapping | each byte → 4 symbols, 2 bits each, **MSB pair first** (mask `0xC0`, shift right) → tone index 0–3 | `[code]` `Modulate.c:222‑224` |
| Symbol phase | template sign alternates every symbol (`(5j+k)&1`) to avoid phase discontinuities at symbol boundaries | `[code]` `Modulate.c:104‑107` |
| Quality threshold | 40–60 depending on frame class | `[code]` `ARDOPC.c:755,772,805` |

**Frame‑type header length:** 10 symbols × 20 ms = 200 ms `[spec-pdf]` App. B "Frame Len 200".

Content payloads (4FSK 50 baud) `[code]` `FrameInfo` + encoders:
- **DATANAK/BREAK/IDLE/DISC/END/ConRejBusy/ConRejBW/DATAACK:** 0 data bytes — the
  frame type itself is the message (2‑byte header) `[code]` `ARDOPC.c:751,768,827`.
- **ConAck / PingAck:** 3 data bytes `[code]` `ARDOPC.c:816`. ConAck's byte is the
  received leader length **in tens of ms (0–2550)**, repeated 3× for redundancy
  `[spec-pdf]` §7.2.3.6. PingAck packs S/N (5 bits, `((b>>3)&0x1F)−10` dB, −10…+21) and
  quality (3 bits, `(b&7)·10+30`, 30…100) `[code]` `Modulate.c:171‑172`.
- **IDFRAME / ConReq / PING:** 12 data bytes + **4 RS parity, no CRC** (see §5.2)
  `[code]` `ARDOPC.c:801‑802`, `ARDOPC.c:1387,1421`. Frame length ~1600 ms
  `[spec-pdf]` App. B.

The **leader** (all bandwidths, all frames): a 50‑baud two‑tone signal of **1475 and
1525 Hz** (i.e. 1500 ± 25 Hz), phase alternating 0/180° each 20 ms symbol
`[spec-pdf]` §7.1.1, App. B; `[code]` `Modulate.c:42`, `CalcTemplates.c:63‑64`. The
two‑tone design lets the DSP reject single‑tone carriers and tune to ~1 Hz via
envelope correlation. The **leader‑sync** is the final leader symbol emitted **without**
phase inversion (two adjacent symbols share phase) — this phase reversal marks the
symbol‑timing reference `[spec-pdf]` §7.1.2, App. B; `[code]` `Modulate.c:55‑58`.

---

## 4. DATA frame families across the four bandwidths

Data frames are 4FSK, 4PSK (QPSK), 8PSK or 16QAM, all at **100 baud** on HF (600 baud
4FSK is FM‑only). PSK/QAM carriers are spaced **200 Hz**; multi‑carrier layouts are
symmetric about 1500 Hz. Single carrier sits at 1500 Hz `[code]` `CalcTemplates.c:360`,
`Modulate.c:423‑460`.

**PSK/QAM carrier frequency table** (9 template slots) `[code]` `CalcTemplates.c:360,368‑370`:

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Hz | 800 | 1000 | 1200 | 1400 | **1500** | 1600 | 1800 | 2000 | 2200 |
| 1 car |  |  |  |  | ✓ |  |  |  |  |
| 2 car |  |  |  | ✓ |  | ✓ |  |  |  |
| 4 car |  |  | ✓ | ✓ |  | ✓ | ✓ |  |  |
| 8 car | ✓ | ✓ | ✓ | ✓ | *(skip)* | ✓ | ✓ | ✓ | ✓ |

Multi‑carrier modes always use the even‑hundred‑Hz carriers and skip 1500 Hz
`[code]` `Modulate.c:458‑459`.

**4FSK data tone sets** `[code]` `CalcTemplates.c:111,192‑195,274‑277`:
- 50 baud (200 Hz BW): 1425/1475/1525/1575 Hz (50 Hz spacing)
- 100 baud (500 Hz BW): 1350/1450/1550/1650 Hz (100 Hz spacing)
- 600 baud (2000 Hz BW, FM only): 600/1200/1800/2400 Hz (600 Hz spacing)

### 4.1 PSK/QAM symbol constellation

Per carrier a differential phase is accumulated; magnitude (16QAM only) is absolute
`[code]` `Modulate.c:354‑408`:
- Low 3 bits of a symbol = phase index 0–7 → **0, 45, 90, 135, 180, 225, 270, 315°**.
- 4PSK uses `SymSet = 2` (steps of 90° → only phases 0/2/4/6 as deltas, full 0–7 range
  used cumulatively); 8PSK uses all 8 phases; bit `0x08` set = **half amplitude**
  (16QAM's outer/inner ring).
- **Phase is differential** (`Symbols[n] = Symbols[n−1] + Δ`) with the **reference
  symbol phase forced to 0** `[code]` `Modulate.c:392‑399`; `[spec-pdf]` App. A rev
  0.1.13 "force 0 as reference." 16QAM magnitude is absolute per symbol, **not**
  differential `[code]` `Modulate.c:402‑405`.
- Bits are packed MSB‑first: 2/3/4 bits per symbol for 4PSK/8PSK/16QAM `[code]`
  `Modulate.c:370‑386`.

Each PSK/QAM carrier begins with **one reference symbol** (phase 0) — no reference is
needed for 4FSK `[spec-pdf]` App. B "FrameData."

### 4.2 Data‑frame table (code‑authoritative)

Per‑carrier on‑air block = **1 byte count + `k` data bytes + 2 byte CRC + `r` RS parity
bytes**. Below, `k` = payload bytes/carrier (`intDataLen`), `r` = RS parity
bytes/carrier (`intRSLen`); **net payload = k × carriers**. Modes are named
`MOD.BW.baud[S].E/O` where `S` = short variant, `E`/`O` = even/odd type pair. Values
from `[code]` `ARDOPC.c:837‑1072` (`FrameInfo`); frame lengths and mode names from
`[spec-pdf]` App. B/C.

| Mode name | Hex E/O | Cars | Mod | Baud | k (data/car) | r (RS/car) | Net payload (B) | Frame len (ms) |
|---|---|---|---|---|---|---|---|---|
| **200 Hz BW** |  |  |  |  |  |  |  |  |
| 4FSK.200.50S | 48/49 | 1 | 4FSK | 50 | 16 | 4 | 16 | 2180 |
| 4PSK.200.100S | 42/43 | 1 | 4PSK | 100 | 16 | 8 | 16 | 1280 |
| 4PSK.200.100 | 40/41 | 1 | 4PSK | 100 | 64 | 32 | 64 | 4160 |
| 8PSK.200.100 | 44/45 | 1 | 8PSK | 100 | 108 | 36 | 108 | 4120 |
| 16QAM.200.100 | 46/47 | 1 | 16QAM | 100 | 128 | 64 | 128 | 4150 |
| **500 Hz BW** |  |  |  |  |  |  |  |  |
| 4FSK.500.100S | 4C/4D | 1 | 4FSK | 100 | 32 | 8 | 32 | 1920 |
| 4FSK.500.100 | 4A/4B | 1 | 4FSK | 100 | 64 | 16 | 64 | 3520 |
| 4PSK.500.100 | 50/51 | 2 | 4PSK | 100 | 64 | 32 | 128 | 4330 |
| 8PSK.500.100 | 52/53 | 2 | 8PSK | 100 | 108 | 36 | 216 | 4120 |
| 16QAM.500.100 | 54/55 | 2 | 16QAM | 100 | 128 | 64 | 256 | 4150 |
| **1000 Hz BW** |  |  |  |  |  |  |  |  |
| 4PSK.1000.100 | 60/61 | 4 | 4PSK | 100 | 64 | 32 | 256 | 4170 |
| 8PSK.1000.100 | 62/63 | 4 | 8PSK | 100 | 108 | 36 | 432 | 4120 |
| 16QAM.1000.100 | 64/65 | 4 | 16QAM | 100 | 128 | 64 | 512 | 4150 |
| **2000 Hz BW** |  |  |  |  |  |  |  |  |
| 4PSK.2000.100 | 70/71 | 8 | 4PSK | 100 | 64 | 32 | 512 | 4120 |
| 8PSK.2000.100 | 72/73 | 8 | 8PSK | 100 | 108 | 36 | 864 | 4120 |
| 16QAM.2000.100 | 74/75 | 8 | 16QAM | 100 | 128 | 64 | 1024 | 4150 |
| **2000 Hz FM only** |  |  |  |  |  |  |  |  |
| 4FSK.2000.600S | 7C/7D | 1 | 4FSK | 600 | 200 | 50 | 200 | 1727 |
| 4FSK.2000.600 | 7A/7B | 1 | 4FSK | 600 | 600 | 150 | 600 | 5100 |

Notes:
- The 200 Hz single‑carrier modes are usable inside **any** wider bandwidth session for
  robustness/agility `[spec-pdf]` §6.2, App. C Fig C‑1. Likewise the 500 Hz 4FSK/4PSK
  modes are reused inside 1000/2000 Hz sessions `[spec-pdf]` App. C note 7.
- **600‑baud FM modes** concatenate three sequential sub‑packets, each with its own
  byte‑count + CRC + RS: the encoder calls RS with `k/3+3` data and `r/3` parity per
  sub‑packet `[code]` `ARDOPC.c:1324`. These are disabled unless `Use600Modes` is set
  and are intended for FM > 29 MHz `[spec-pdf]` §2.5.

### 4.3 Representative max throughput (raw, before ARQ overhead)

From `[spec-pdf]` App. C worksheets (bytes/min, and bits/s/Hz):

| Mode | Max thruput (B/min) | raw bits/s/Hz |
|---|---|---|
| 16QAM.200.100 | 1512 | 2.00 |
| 8PSK.200.100 | 1286 | 1.50 |
| 16QAM.500.100 | 3024 | 1.60 |
| 16QAM.1000.100 | 6036 | 1.60 |
| 16QAM.2000.100 | **12072** | 1.60 |
| 4FSK.200.50S (most robust) | 310 | 0.50 |

The spec states the net (post‑FEC) ARQ throughput range is ~38:1 fastest‑to‑slowest
`[spec-pdf]` §2.3.

---

## 5. Forward error correction — Reed‑Solomon

### 5.1 Field and code

- Symbol size **m = 8** → GF(2⁸), block length **n = 255** `[code]` `rrs.c:113` (`nn`).
- **Primitive polynomial** `pp = {1,0,1,1,1,0,0,0,1}` = **x⁸ + x⁴ + x³ + x² + 1 = 0x11D**
  (the RFC 5510 m=8 polynomial; the source comment's "0x171" is an alternate written
  form) `[code]` `rrs.c:124‑130`.
- Primitive element **α = 2**; generator built from `(x + αⁱ)`, i = 1…2t `[code]`
  `rrs.c:196‑199,121`.
- Codes are **shortened** RS(n, k): a frame carrier block has `d` data bytes and `r`
  parity bytes, so `t = r/2` correctable byte errors and the effective code is
  RS(d+r, d) drawn from the length‑255 field `[code]` `rrs.c` `rs_append(data, d, r)`.

### 5.2 How RS is applied per carrier

For each carrier of a data frame the encoder builds the RS input as
**[1 byte length][data bytes][2 byte CRC]** = `intDataLen + 3` bytes, then appends
`intRSLen` parity bytes `[code]` `ARDOPC.c:1161‑1191`:

```
per-carrier block = count(1) ‖ data(k) ‖ CRC16(2) ‖ RS_parity(r)
```

RS parity covers the length byte + data + CRC. Unused data bytes are zero‑filled
`[spec-pdf]` App. B. Multi‑carrier frames repeat this block once per carrier; the
2‑byte frame‑type header precedes all carrier blocks `[code]` `ARDOPC.c:1144‑1191`.

**Exception — ConReq / IDFRAME / PING:** these carry **12 data bytes + 4 RS parity and
no CRC** (RS replaces the CRC), giving an 18‑byte frame (2 header + 12 + 4) `[code]`
`ARDOPC.c:1381‑1392` (ConReq), `:1414‑1426` (Ping), `:802` (`intRSLen = 4`). The
12 data bytes are two 6‑byte compressed callsigns (caller + target); for IDFRAME they
are callsign + grid square (see §8). *Spec/code discrepancy:* the App. B worksheet
lists these frames as "RS 2 / net 14"; the shipped code uses **4** RS bytes and no
CRC. Code wins.

Memory‑ARQ combining of repeated copies (averaging + per‑carrier OK flags) is used on
receive but does not change the wire format `[code]` `FEC.c:332‑393`.

---

## 6. CRC — the nonstandard frame CRC‑16

The spec calls it "CRC16 polynomial x¹⁶ + x¹² + x⁵ + 1" `[spec-pdf]` App. B, but the
shipped implementation is **not** table‑standard CRC‑16/CCITT. `GenCRC16()` `[code]`
`ARDOPC.c:1673‑1718`, and M0LTE's port `ArdopCrc.cs:30‑59`, define:

| Field | Value |
|---|---|
| Width | 16 bit |
| Polynomial constant | **`0x8810`** (not `0x1021`) |
| Init | **`0xFFFF`** |
| Bit order | MSB‑first over each data byte |
| Data injection | the data bit is shifted **into the register LSB before** the polynomial XOR — the ordering that makes it nonstandard |
| RefIn / RefOut | none (no reflection) |
| XorOut | none |

Because the data bit enters the LSB inside the shift‑and‑divide step rather than being
XORed at the top, neither CRC‑16/X‑25 nor the standard CCITT constant reproduces it —
M0LTE explicitly flags this as the deployed wire format `[code]` `ArdopCrc.cs:11‑24`.

**Storage & frame‑type binding** `[code]` `GenCRC16FrameType`, `ARDOPC.c:1722‑1730`:
the two CRC bytes are appended as `[CRC>>8]` (high byte, verbatim) then
`[(CRC & 0xFF) XOR frameType]` (low byte XOR'd with the frame‑type byte). This ties
each carrier block's integrity check to the frame type that carried it. Receiver check
is the inverse `[code]` `ARDOPC.c:1734‑1748`.

**Coverage:** the CRC is computed over `intDataLen + 1` bytes = the length byte + the
carrier's data bytes, **before** RS parity is added `[code]` `ARDOPC.c:1179` ("does NOT
include any FEC" `[spec-pdf]` App. B).

---

## 7. Sync / acquisition and busy detection

Receive state machine `[code]` `ARDOPC.h:241‑250`:
`SearchingForLeader → AcquireSymbolSync → AcquireFrameSync → AcquireFrameType →
DecodeFrameType → AcquireFrame → DecodeFrame`.

- **Leader detection & tuning:** the 50‑baud two‑tone leader (1475/1525 Hz) is detected
  by envelope correlation of the two‑tone waveform, which rejects single‑tone carriers
  and yields frequency tuning to ~1 Hz `[spec-pdf]` §7.1.1.
- **Frame sync:** the phase‑reversed final leader symbol (two adjacent same‑phase
  symbols) sets symbol timing `[spec-pdf]` §7.1.2.
- **Tone/carrier detection uses a Sliding DFT (SDFT) variant**, not a plain DFT: the
  bins are offset by ½N so the SDFT lands exactly on ARDOP's off‑grid FSK tones (a
  240‑sample SDFT hits 1425/1475/1525/1575 Hz; a 120‑sample SDFT hits
  1350/1450/1550/1650 Hz) — bins a normal 50‑Hz‑grid DFT cannot produce `[code]`
  `sdft.c:14‑25`. Recurrence: `S_k(n) = e^{j2π(k+½)/N} · (S_k(n−1) + x(n) + x(n−N))`
  `[code]` `sdft.c:28‑30`. A **Goertzel** routine is retained for single‑block
  reference tone measurement `[code]` `sdft.c:9,32‑33`. The SDFT produces overlapping
  per‑sample results so symbol timing can be refined within a symbol period.
- **Busy detection / BUSYBLOCK:** a 1024‑point FFT at 12000 Sa/s (bin ≈ 11.72 Hz) over
  the passband feeds `BusyDetect3()`; a station may reject or defer a connection when
  prior signals are present (listen‑before‑transmit) `[code]` `BusyDetect.c:54`,
  `ARDOPC.c:2084‑2085`; `[spec-pdf]` §2.7, §5, App. D §1.5.
- **Frequency tolerance:** connect requests accepted with up to ±200 Hz client/server
  offset; short‑term stability < 1 Hz/s for SSB `[spec-pdf]` §4.1.

---

## 8. Scrambling / whitening and callsign encoding

### 8.1 Scrambling

There is **no additive scrambler / whitening** applied to ARDOP data. Robustness comes
from RS FEC, the redundant frame‑type coding, multi‑carrier spreading, and (for PSK/QAM)
the per‑symbol envelope taper. The only per‑symbol sign manipulation is the alternating
template sign used to keep 4FSK phase continuous across symbol boundaries `[code]`
`Modulate.c:104‑107` — a modulation detail, not a data scrambler.

### 8.2 Packed6 callsign / grid‑square compression

Callsigns and grid squares are carried as **6‑byte** fields using DEC SIXBIT
compression `[code]` `Packed6.c`, `StationId.c`:

- An 8‑character field is split into two 4‑character halves; each half → **3 bytes**
  (4 chars × 6 bits = 24 bits) `[code]` `Packed6.c:106‑107`. `COMP_SIZE = 6`
  `[code]` `ARDOPC.h:379`.
- **Alphabet:** ASCII 32 (space) … 63 (underscore) stored verbatim as value−32;
  lowercase a–z folded to uppercase; anything else → space (value 0). 6 bits per
  character `[code]` `Packed6.c:26‑44`.
- Packing is big‑endian within each 3‑byte group (first char occupies the most
  significant 6 bits) `[code]` `Packed6.c:43‑52`; decode is the exact inverse
  `[code]` `Packed6.c:62‑84`.
- A `StationId` renders its canonical `CALL` / `CALL-SSID` string (callsign up to 7
  chars + optional SSID `-1`…`-15` or `-A`…`-Z`) into the 8‑char work buffer before
  compression `[code]` `StationId.c:100‑123`; `[spec-pdf]` §7.2.1, App. A rev 0.1.9.

These 6‑byte fields feed the 12‑byte payload of ConReq (caller‖target), PING
(caller‖target) and IDFRAME (callsign‖grid square) — see §5.2 `[code]`
`ARDOPC.c:1381‑1382,1415‑1416`.

### 8.3 CW identification

When CW ID is enabled, an IDFRAME may be followed by a Morse callsign sent as two‑tone
FSK keying — mark **1609.375 Hz**, space **1390.625 Hz** — at 20 wpm or less `[code]`
`Modulate.c:949‑950`; `[spec-pdf]` §7.2.3.13. (The reference dot length of 768 samples
is ~12 wpm.)

---

## Open items / values not pinned down

- **Exact per‑mode frame lengths** are taken from the spec App. B/C worksheets
  (`[spec-pdf]`); the code computes them dynamically from symbol counts + leader, so a
  handful of App. B cells may be ±a few ms versus a live render. Frame‑type header =
  200 ms and 100‑baud symbol = 10 ms are exact `[code]`.
- **600‑baud FM sub‑packet geometry:** the `k/3`, `r/3` split is exact from `[code]`
  `ARDOPC.c:1324`, but the short 600‑baud variant (`k=200`) is not an exact multiple of
  3; the precise zero‑fill/rounding of its final sub‑packet was not traced and should be
  confirmed against a live 600‑baud render before relying on FM interop.
- **RS decode side** (error‑locator / Chien search parameters) lives in `rrs.c`; only
  the *encode* field constants were required for a compatible transmitter and are given
  above.
- The App. B "RS 2 / net 14" figures for ConReq/ID/Ping contradict the code's 4‑RS,
  no‑CRC construction; treat the **code** (`rs_append(...,12,4)`) as authoritative.
