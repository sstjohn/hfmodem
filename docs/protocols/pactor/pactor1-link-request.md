# PACTOR-1 link-request frame

The link request ("Synchronisationspaket") is the frame a master sends to call a
responder. This document specifies its encoding: which bytes leave the transmitter,
in which order, and where the 200 Bd section starts. For the Normal / Longpath /
Robust classification a receiver applies to the frame once it has it, see
`pactor-connect-frames.md`; for the burst's place on the 1.25 s grid, see
`pactor1-timing.md` §3.

The normative description is the PACTOR Level-1 protocol description (Helfert
DL6MAA and Strate DF4KV, 5 November 1990), §Aufbau einer PACTOR-Verbindung.
Sailer HB9JNX's hf-pactor (`hfkernel/fsk/pactor.c`) is a complete PACTOR-1
implementation that interoperated with commercial controllers and is cited where
it fixes something the description leaves implicit.

## 1. The bytes

    T = [0x55] + uppercase-ASCII callsign + 0x0F fill to 9 bytes

    100 Bd section (9 bytes, 720 ms)   =  T
    200 Bd section (6 bytes, 240 ms)   =  T[1:7]

The callsign is at most 8 bytes, ASCII: "Diese Pakete enthalten das
SLAVE-Rufzeichen (max. 8 Byte, **ASCII-Codiert**) als 100 Baud- und
200-Baud-Bitmuster. Bei kuerzeren Rufzeichen werden Leerbytes mit 0F(HEX)
aufgefuellt." A receiver validates the address characters as uppercase letters
and digits.

**Neither section is transformed.** Both are LSB-first, continuous phase, with
MARK(1) = 1400 Hz and SPACE(0) = 1600 Hz, and the 200 Bd section begins at the
instant the 100 Bd section's last bit ends — no gap, no inserted bit, no
re-encoding. hf-pactor builds one buffer, `txbuf[0] = 0x55` followed by the
callsign, and hands it to layer 1 twice without touching it: `9*8` bits at 100 Bd
from `txbuf`, then `6*8` bits at 200 Bd from `txbuf+1`, the second scheduled at
exactly `txtime + (1e6/100)*9*8`. The only whole-frame transform layer 1 applies
is an XOR — the shift-polarity toggle — which is not a bit rotation.

The 200 Bd copy is the callsign a second time and nothing else. It exists
"lediglich zur Ueberpruefung der Kanalqualitaet": the responder answers CS1 if it
arrives clean and CS4 if it does not. The whole frame is 960 ms.

    DL6MAA:  55 44 4c 36 4d 41 41 0f 0f  |  44 4c 36 4d 41 41
    KE5YTA:  55 4b 45 35 59 54 41 0f 0f  |  4b 45 35 59 54 41

Because the second section is `T[1:7]`, a receiver that has located the sync byte
can cross-check the two halves byte for byte: `secondary[i] == primary[i+1]` for
all six bytes.

## 2. Locating the frame

Two framing hazards, neither visible at the byte layer.

**The sync byte does not fix the bit phase.** Inserting a leading zero and
shifting the whole stream one bit —
`raw[0]=(T[0]&0x7F)<<1; raw[i]=((T[i]&0x7F)<<1)|(T[i-1]>>7)` — produces a
bitstream that still contains the sync-byte image, one bit along. A receiver
hunting for that image finds it under either phase, so every byte-level check
passes on both and the address decodes identically. What separates them is below
the byte layer: the tone of the symbol immediately before the sync, and where the
200 Bd section falls relative to the located image.

**A one-bit shift of the address section moves the redundancy section by two
symbols**, because one 100 Bd bit is two bit periods at 200 Bd. Read back through
the address section's own timing, a correctly framed transmission puts its
redundancy at offset 0, and a transmission whose halves are shifted relative to
one another does not:

| what is transmitted | 200 Bd tail reads back `T[1:7]` at | symbol before the sync |
|---|---|---|
| **address and redundancy both plain** | **0** | whatever preceded the burst |
| both shifted one bit | 0 | SPACE, full power |
| address shifted, redundancy plain | **−2** | SPACE, full power |
| address plain, redundancy shifted | **+2** | whatever preceded the burst |

A receiver cross-checking `secondary[i] == primary[i+1]` fails on all six bytes
for either of the last two, and passes for either of the first two. The offset
test alone therefore does not distinguish a correctly framed frame from one whose
halves are shifted together; the tone of the symbol before the sync does, and in
a correctly framed transmission that symbol is not part of the frame at all.

**Envelope onset does not locate the first bit.** Stations key up on an idle MARK
carrier — one measured transmitter ramps about 1.4 bit periods before the frame —
so the onset measures the transmitter's keying rather than the frame boundary,
and splits roughly evenly across recorded traffic.
