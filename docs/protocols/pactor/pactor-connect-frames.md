# PACTOR connect frames — Normal, Longpath, Robust, Free Signal

The PACTOR connect is classified by the receiver into one of five kinds:
`Normal Call`, `Longpath Call`, `Robust Call`, `Free Signal Normal` and
`Free Signal Encrypted`. This document specifies what distinguishes them and
what a transmitter sends to select each. It is a companion to
`docs/protocols/pactor/pactor3.md`, which covers the rest of the PACTOR-1 link.

The dual-rate connect structure this document builds on — a 100 Bd address
section carried alongside a 200 Bd Memory-ARQ copy — is described in the 1990
PACTOR-1 description by DL6MAA and DF4KV and implemented in `hfkernel/fsk/pactor.c`
of hf-pactor (Sailer, HB9JNX).

Evidence tags are defined in `EVIDENCE.md`. The connect carries no capability
field, which is §1's subject and is also where `pactor-capability.md` starts.

---

## 1. The kind is a decode classification, not a capability field

The receiver records the kind as a small integer written by the connect parser
from how the frame decoded. There is no "which generations do you support"
capability field anywhere in the connect: the five labels are five *decode
outcomes*, a receiver-side classification rather than a negotiated capability
declaration.

| code | classification |
|---|---|
| 1, 2 | `Normal Call` |
| 3, 4 | unreachable (§2) |
| 5, 6 | `Longpath Call` |
| 7, 8 | unreachable (§2) |
| 9, 10 | `Robust Call` |
| 11, 12 | `Free Signal Normal` |
| 13, 14 | `Free Signal Encrypted` |
| 0, ≥15 (e.g. 126, 127) | no connect report |

## 2. Every code is a (kind, polarity) pair — hence the holes

Each label owns exactly **two** adjacent codes. In every case the second member
is the same frame with the whole waveform inverted, resolving the FSK polarity
ambiguity. The two holes at 3,4 and 7,8 are combinations the parser cannot
construct.

## 3. Branch A — `Normal` and `Longpath`

Branch A is entered when the rotated primary ring byte 0 is the sync byte `0x55`
(or `0xAA`). The classification is:

```
code  = 1 if sync == 0x55 else 2                  # 0xAA == whole frame inverted
code |= 4 if secondary[0] == ~primary[1]          # tested at index 0 only
```

- `primary[i]` = post-rotation image of the 9-byte 100-Bd address section.
- `secondary[i]` = post-rotation image of the 6-byte 200-Bd Memory-ARQ copy;
  the receiver requires `secondary[i] == primary[i+1]` for i = 0..5.
- At i = 0 **only**, a mismatch is tolerated if the two are bitwise complements.
  That sets bit 2, and the receiver then repairs the byte
  (`if (code & 4) payload[0] = ~payload[0]`) before the terminator scan.
- Codes 1,2 → `Normal Call`; codes 5,6 → `Longpath Call`.

**`Longpath` is signalled by transmitting the first callsign character of the
100-Bd address section bit-inverted, with the 200-Bd redundancy copy carrying the
true character.** One byte, in the address section, at a fixed position — nothing
to do with pad bytes and nothing to do with a capability list.

Two connect frames for `W1AW` whose address images begin `aa ae 62 …` and
`aa 50 63 …`, and whose 200-Bd redundancy sections are byte-identical, classify
as `Normal Call: W1AW` and `Longpath Call: W1AW` respectively.

## 4. Branch B — `Robust` and `Free Signal` (a different frame entirely)

Branch B is entered when branch A does not apply or fails. It is a **completely
different connect frame**: no `0x55` sync, no ASCII, no 200-Bd redundancy section.

Gates, all on the rotated 12-byte primary slot `a[]`:

```
a[6] == ~a[8]      a[7] == ~a[9]      a[6] == a[10]
```

then a 10-byte XOR-fold producing `b[]`:

```
b[0]=a[0]^a[6]  b[1]=a[1]^a[7]  b[2]=a[2]^a[8]  b[3]=a[3]^a[9]
b[4]=a[4]^a[6]  b[5]=a[5]^a[7]  b[6]=~a[6]      b[7]=~a[7]
```

then a **CRC-16/X-25** over `b[0..7]`, using the reflected CRC-CCITT table with
init `0xFFFF`, passing when the running value equals the residue `0xF0B8`.
`b[0..5]` is the payload, `b[6..7]` the CRC.

Six **whitening masks** are tried in order on the raw slot before folding, and
*the mask that succeeds is the kind*:

| mask | code | label |
|---|---|---|
| identity | 9 | `Robust Call` |
| complement | 10 | `Robust Call` |
| `^0x55` | 11 | `Free Signal Normal` |
| `^0xAA` | 12 | `Free Signal Normal` |
| `^0x0F` | 13 | `Free Signal Encrypted` |
| `^0xF0` | 14 | `Free Signal Encrypted` |

The address in this branch is **packed 6-bit**, not ASCII: eight characters are
read LSB-first out of `b[0..5]` as 6-bit fields, each `+0x20`
(`chars[k] = ((b as LE48) >> 6k & 0x3f) + 0x20`, giving ASCII `0x20..0x5F`).

The pairing appears again: identity/complement, `0x55`/`0xAA`, `0x0F`/`0xF0` —
each pair is one mask and its bit-inverse, i.e. the two FSK polarities of one
mask. Branch B therefore carries **three** frame types, and `Free Signal
Encrypted` differs from `Free Signal Normal` only in the whitening constant used
as its marker.

The branch-B frame carries a frame CRC-16. The `Normal` / `Longpath` connect
frame of §3 has no frame CRC. Both statements hold; they describe different
frames.

### Branch B's on-air framing  [P+S: 8 recordings, 3 bands, 2 months]

Eleven bytes, 88 symbols at 100 Bd, 0.88 s, one rate throughout, 200 Hz shift,
keyed on the connect raster, polarity alternating burst to burst. The 200-Bd
secondary slot is never used, so branch B shares none of the dual-rate FSK
timing of the `Normal` / `Longpath` connect.

| frame | rate | length | sync | address | CRC |
|---|---|---|---|---|---|
| `Normal` / `Longpath` | 100 Bd then 200 Bd | 0.96 s | `0x55` | ASCII | none |
| `Robust` / `Free Signal` | 100 Bd throughout | 0.88 s | none, three gates | 6-bit packed | CRC-16/X-25 |

SCS's own monitor, version 1.0, over the public sigidwiki Free Signal sample
(`rf-corpus/sigidwiki/PACTOR-FS.mp3`) prints
`###CONNECT: [Free Signal Encrypted: DAO8]` five times, and prints it again
after an 11 Hz retune, so what it reads is the whitening constant and not a
tuning artefact [P+S: 1 recording]. That sample measures six bursts of 0.895 to
0.898 s,
tones 1411.3 and 1610.9 Hz (shift 199.6 Hz), four on a 2.50 s raster then two at
1.25 s, consecutive bursts bit-identical in identity/complement pairs
[P: 1 recording, 6 bursts]. A sixth burst closes the CRC on a garbage ident, so
one CRC-valid
burst does not settle a callsign.

Seven off-air `Robust Call` trains carry the same framing from real SCS modems,
read byte-clean by that same monitor across three bands, two months and four
signal levels [P+S: 7 recordings, 7 destinations, 3 bands]:

| destination | file | reports |
|---|---|---|
| `K6SDR` | `rf-corpus/10145k_055704.wav` | 10 |
| `AJ7C` | `rf-corpus/10145k_055923.wav` | 5 |
| `W5STX` | `rf-corpus/14109k_131254.wav` | 10 |
| `KY4RY` | `offair/captures/20260722T025051Z_KB8AY-7.101MHz-VARA2750.wav` | 12 |
| `WM4RB` | `offair/captures/20260722T024936Z_KB8AY-7.101MHz-VARA2750.wav` | 12 |
| `KD4JWF` | `rf-corpus/7101k_192259.wav` | 5 |
| `K4MSU` | `captures/onair-0803-201543-passive/listen.wav` | 2 |

Bursts there run 0.863 to 0.915 s on the 1.25 s raster. The address the monitor
prints is the **destination**, not the caller: on the branch-A control
(`rf-corpus/sigidwiki/PACTOR_SELCALLaudio.mp3`) it calls `OL1A` and the
answering station's first packet is `1ol1b`. Plain 100 Bd FSK demodulation
reaches every byte the parser reads, which puts SCS's "Distance Spreading"
inside the 11 bytes rather than in a waveform of its own.

### What SCS calls these frames

SCS names two things here and never connects them in print, but they are one
frame family with one coding and one length, separated only by the whitening
constant that marks the kind. Every quotation below is from SCS's own
published manuals, each named by its filename at the point it is cited.

**Robust Connect** [A]. *Manual for Professional Firmware 3.2*
(`profi32_eng.pdf`) §1.5, §1.4 in the 3.1d edition: link initiation is normally
"based, for compatibility, on the so-called 'Level-I connect'", which works
"only down to around -15 dB @ 4 kHz" while PACTOR-II data reaches -18 dB,
leaving a connect "step" of about 3 dB. The replacement "uses a complex coding,
developed by SCS, called Distance Spreading. It can produce a fast, reliable
link, down to around -20 dB @ 4 kHz", and makes incorrect links from similar
callsigns or from calling into an occupied channel "impossible". §1.5.2:
`C %DL6MAA` starts one, `%` being the "Robust Connect Operator"; robust connects
are "only allowed within the normal PACTOR time frame", so `C !%CALL` and
`C %!CALL` are refused. §1.5.1 gives `CONType` 0 to 3, default 3, selecting
which connect kinds the station accepts: 0 none, 1 PACTOR-I only, 2 robust only,
3 all. The same table is in `SCS_Manual_PTC-II_4.0.pdf`,
`SCS_Manual_PTC-IIusb_4.0.pdf` and `SCS_Manual_PTC-IIIusb_4.1.pdf` at §6.23.

`SCS_Manual_PTC-II_4.0.pdf` §6.22.2 adds that "Up to firmware version 3.1, the
Robust Connect has only been available as a part of the Professional PTC-II
firmware. With version 3.1 Robust Connect will also be available as a part of
the 'normal' firmware, but restricted to outgoing calls/connects", that this
"is also valid for WA8DED hostmode", and that "The Robust Connect uses 'normal'
PACTOR timing." **That acceptance restriction is absent in the USB generation**:
the PTC-IIusb 4.0 and PTC-IIIusb 4.1 §6.22.2 keep only the `C %CALL` paragraph,
PTC-IIIusb §1.6 lists Robust Connect among the former Professional features
installed in that model, and the licence-gated list at PTC-IIusb §6.46 does not
name it. Dragon firmware 2.40 (`Update_Info_DR7X00_Version_2_40_English.pdf`)
adds `CONIntegrity`, 0 or 1, default 0, where 1 makes "the Robust Connect (SLAVE
side) … searching for incoming calls with less error tolerance", raising the
required SNR "by roughly 2 dB". A gateway on any of these at default settings is
therefore expected to accept a robust call.

**The PACTOR Free Signal Protocol** [A]. `profi32_eng.pdf` §1.4. A host with no
link in progress repeats a short signal on the link frequency; clients that hear
it may call, computing an access time from the host loading it advertises and
randomising the call so collisions stay "less than 1 percent". §1.4.3 gives the
framing this document's §4 measures: "The PACTOR FS is 0.88 sec long and is sent
in 100 Bd FSK with a shift of 200 Hz. It sounds similar to the usual PACTOR
connect packet. … The client PTC can read the PACTOR FS by means of a new, SCS
developed, coding method known as 'Distance Spreading', in combination with
'Memory-ARQ in progress' down to approximately -20 dB SNR in a 4 kHz noise
bandwidth." §1.4.5 makes the FS the link initiation rather than an
advertisement: a client in `FRee MOde 1` connects with a bare `C`, its callsign
argument ignored, and the PTC "acknowledges" an FS with its own transmission
after the randomised delay, `C +` acknowledging the very next FS heard. Once up,
"the link is now a completely normal PACTOR link, client and host exchange their
actual callsigns (MYcall)"; the FS ident (`FRee IDent`, 3 to 8 characters,
default `MAILHOST`) "has absolutely NOTHING to do with" the host's MYcall.

What the client's acknowledgement is on the air, and which of the six masks it
carries, is not published anywhere and is not on tape here [C].

## 5. Frame constructions and the classification each produces

Each of these frames recovers the intended callsign:

| construction | code | classification |
|---|---|---|
| stock `Normal` connect | 1 | Normal |
| stock, address image byte 1 complemented | 5 | Longpath |
| stock, whole frame complemented | 2 | Normal |
| complemented, image byte 1 flipped back | 6 | Longpath |
| 6-bit-packed `DL6MAA`, CRC-16, fold inverted | 9 | Robust |
| …complemented | 10 | Robust |
| …`^0x55` / `^0xAA` | 11 / 12 | Free Signal Normal |
| …`^0x0F` / `^0xF0` | 13 / 14 | Free Signal Encrypted |

## 6. Which bytes of the address image are load-bearing

The 9-byte address image is `T = [0x55] + ASCII callsign + 0x0F fill`. Setting
each image byte `T[0..8]` to all 256 values, for each callsign length, gives:

| callsign | `T[1]` | `T[2..6]` | `T[7]` | `T[8]` |
|---|---|---|---|---|
| `W1AW` (4) | 254×0, 1×code1, 1×code5 | one accepted value each | 256×code1 | 256×code1 |
| `K7ABC` (5) | 254×0, 1×code1, 1×code5 | one accepted value each | 256×code1 | 256×code1 |
| `N0CAL` (5) | 254×0, 1×code1, 1×code5 | one accepted value each | 256×code1 | 256×code1 |
| `DL6MAA` (6) | 254×0, 1×code1, 1×code5 | one accepted value each | 209×0, 47×code1 (terminator) | 256×code1 |

- **The only image byte that can produce a non-`Normal` code is `T[1]`**, the
  first callsign character, and only at its single complement value. Byte
  position 1 is never a pad byte.
- **Trailing pad bytes are inert.** `T[8]` is unconditionally free for every
  callsign length: all 256 values give the identical `Normal` report with the
  identical callsign. For callsigns of ≤5 characters, `T[7]` is free too.
- For a 6-character callsign `T[7]` is the terminator and is load-bearing, but
  even so it can only produce code 0 or code 1 — never a non-`Normal` kind.
- The bytes strictly *between* the last character and index 6 are **not** free:
  branch A's comparison loop covers `primary[1..6]` and rejects the frame if a
  non-`0x0F` byte follows the terminator inside that window. The free region is
  `T[max(len+2, 7) .. 8]`, i.e. 1 byte for a 6-character call, 2 bytes for ≤5.

**Consequence for using the connect as a capability channel.** Writing the
trailing pad bytes cannot make a stock peer classify a connect as `Longpath` or
`Robust`, because the kind is decided by (a) the address/redundancy complement
relationship at index 0, (b) global waveform polarity, and (c) which of six
whitening masks makes a *different frame's* CRC-16 close. None of those is
reachable from a pad byte. Equally, there is no spare code point to borrow: the
kind is a decode classification, not an advertised capability, and its two free
values (3,4 and 7,8) are arithmetically unreachable rather than reserved.

One residual coupling: branch B reads slot bytes 6..10, which for a 9-byte
address image includes `T[7]`, `T[8]` and stale ring bytes. A pad-byte value can
in principle make branch B's complement gates *and* its CRC-16 close by accident;
the probability is bounded by 6 masks × 2⁻¹⁶ ≈ 1e-4 per frame. No value of any
single image byte produces one.
