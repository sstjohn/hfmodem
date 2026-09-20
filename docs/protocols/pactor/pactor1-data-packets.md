# PACTOR-1 data packets

A PACTOR-1 data packet is a header byte, a fixed-length data field, a status byte
and a CRC — the structure every later PACTOR mode inherits [M.1798 §4]. This
document specifies the field conventions as stations actually send them: which
fields a receiver may gate on, which it may not, and how the 200 Bd frame differs.

The 1990 Level-1 description by Helfert DL6MAA and Strate DF4KV is normative for
intent. Where transmissions on the air contradict it, both are recorded: the
description says what was meant, the air says what a receiver must accept.

Of the PACTOR-1 data packets available off the air, there are exactly **three**,
from **two** stations, in 12.25 hours of audio. All three are 100 Bd. No off-air
200 Bd data packet exists.

## 1. Two stations, byte for byte

**W4DNA**, `regress/fixtures/pos_pactor1_local.wav` t=7.62, retransmitted on five
consecutive cycles. The packet the CRC variant and byte order are settled against.

    on air   aa 31 77 34 64 6e 61 0d 1e | 31 | 41 5b
             hdr  <------ field ------>  status  CRC

**JN36lf**, `regress/fixtures/pos_p1_data_jn36lf.wav` — a European station heard on
14110 kHz through the Vérossaz WebSDR, three packets back to back at t=14.28, 15.25
and 16.23 with ~15 ms between them. The third repeats the second with the shift
inverted.

    55 | 7c 31 4f 7c 4b 4d 31 37 | 03 | 83 57     count 3, type 0 (8-bit ASCII)
    aa | 63 a0 32 f6 a6 9f ff e1 | 04 | fc cf     count 0, type 1 (Huffman)

The type field selects the on-line compression applied to the data field — plain
8-bit ASCII, Huffman, or run-length [M.1798 §5]. The CRC is CRC-16/X-25 over
bytes 1..9, i.e. the data field and the status byte, excluding the header.

JN36lf's tones measure 1378.6/1578.9 Hz — a 200.3 Hz shift sitting 21 Hz low,
decoded without correction because the 100 Bd integration window is 100 Hz wide.

## 2. Status bits 4-5 are the data type's third bit, and must go out clear

The 1990 description calls bits 4-5 "noch nicht belegt". W4DNA **sets** them
(0x31); JN36lf leaves them **clear** (0x03, 0x04). Both conventions are on the air,
so a receiver that *requires* either value rejects the other station for every
cycle of a whole session. **Nothing may gate on bits 4-5.**

**A transmitter is a different question, and the two were run together for a
month.** `pactor1.status_byte` used to force them SET, on the ground that this
reproduced a real station's packet byte for byte. Only one of the two stations here
is being acknowledged:

| | status | bits 4-5 | counter | peer |
|---|---|---|---|---|
| JN36lf | 0x03, 0x04 | clear | advances 3 → 0 | link running |
| W4DNA | 0x31 | set | stuck at 1, five cycles | same CS4 every cycle |

A repeat of the same control signal is a repeat request (`pactor1-control-signals.md` §4),
so W4DNA is being refused in every one of those cycles. "Recovered with a valid CRC
six times over" was six recoveries of *one* rejected frame.

What the bits mean to anything built after 1990 is measurable. Given identical
field bytes `31 57 39 53 53 4a 0d 1e`, an independent PACTOR monitor reports

    status 0x31   ###PAYLOAD1: LEN: 8, TYPE: 4
    status 0x01   ###PAYLOAD1: LEN: 8, TYPE: 0

so it reads bits 2-**4** as the data type, as every later PACTOR does, and 4 in the
SCS table is *PMC German* compression. Bit 5 is the long-cycle request. Status 0x31
therefore tells a modern gateway that eight bytes of plain ASCII are Markov-
compressed German and asks for a 3.75 s cycle. The frame is well formed and the CRC
is valid.

**That misdeclaration is not inert, and reading it as inert was the error.** PMC is
absent from PACTOR-1, whose own compression is Huffman, so a type of 4 declares an
encoding a PACTOR-1 modem cannot produce — a statement about the generation of the
transmitter, carried in the field that was already there rather than in the two bits
1990 left unassigned.

Clearing them is correct under **both** readings — 0x01 is counter 1, data type 0
whether the type field is two bits or three — which is what settles it. Setting them
is safe only against the 1990 text.

**Corrected in scope, 2026-08-30: the transmit rule above is a MID-LINK rule.**
The corpus splits on packet role, not on station: both stations recorded being
granted PACTOR-3 set bits 4–5 on the **first packet of the link** — the
announcement of the 1990 *Levelfestlegung*, `<level digit><call><CR>` — and no
station on record sets them mid-link. W4DNA's 0x31 read here as "being refused"
was resolved one word later by the `0x59A` grant, which acknowledged the
announcement and commanded the upgrade; the P1-level repeat-requests before it
were never a verdict on the byte. Every `0x59A` grant in this station's own
record was drawn by a bits-4–5 announcement (0x31/0x35) and none by 0x21 or by
clear bits. So: receivers still must not gate on them; transmitters clear them
mid-link; **a station announcing the PACTOR-3 upgrade sets both on the
announcement**, which is the shipped arm default (`onair.P1_STATUS_ANNOUNCE`).
**Bit 4 elicits an unassigned codeword; bit 5 selects WHICH ONE.** The full 2x2 has
been flown on **one** gateway, KB5LZK 10146400 on 2026-08-30, inside nineteen
minutes:

| status | bit 4 | bit 5 | codeword returned | our counter |
|---|---|---|---|---|
| `0x01` | — | — | none | **advances; the only value ever acknowledged** |
| `0x21` | — | yes | **none at all** | frozen |
| `0x11` | yes | — | **`0x6A9`** | frozen |
| `0x31` | yes | yes | **`0x59A`**, the grant | frozen |

So bit 4 without bit 5 draws a real, repeated, zero-error codeword that is not the
grant — **15 consecutive cycles with no gap**, still arriving when the link went
down — and only both bits together draw `0x59A`. **Neither bit is a passenger, and a
reading that credits the grant to bit 4 alone is over-claimed.**

`0x6A9` is otherwise indistinguishable from `0x59A`: the same answer slot, onset
1028–1041 ms after the caller's slot boundary against `0x59A`'s 1030–1033, the same
117.5–125.0 ms length that 12 bits at 100 Bd gives, and the same cycle-by-cycle shift
inversion. **What it means is unknown**, and nothing here acts on one.

**One gateway, one session.** An earlier `x18` attributed to a second gateway has no
surviving transcript, and of the six sessions from that date whose captures do
survive, none contains a `0x6A9` — they decode to `0x59A` trains.

**Every value except the untouched default freezes the packet counter**, so the two
intermediate cells buy neither a grant nor a working data path.

## 3. Header/counter parity is per-station

The header alternates 0x55 ↔ 0xAA on every packet carrying new information, and the
status byte's mod-4 counter increments on the same event, so the two stay locked
together — but the *phase* of that lock is per-station:

    W4DNA    count 1 (odd)   under 0xAA
    JN36lf   count 3 (odd)   under 0x55  and  count 0 (even) under 0xAA

Opposite parities. The description pins the phase only at link setup ("das erste
normale Datenpaket mit Head=AA (HEX) und Paketzaehler=1"), and a station that has
been running for a while need not still be on it. **The header's value is a gate;
its agreement with the counter is not.**

## 4. What the header is a gate for

The header byte lies outside the CRC-protected region, which makes it eight bits of
free check on a scan that has to trial thousands of alignments. It is needed.

`decode_p1_packets` reads two bauds × two polarities × 17 sub-symbol offsets per
envelope rising edge: ~9500 candidate alignments per second of signal and ~28000
per second of noise. At 2⁻¹⁶ per trial that is one CRC-valid frame every 1.8 s of
silence — 6341 accepts across 2659 files and 419,740,806 candidate alignments,
against 6405 predicted. That is combinatorics working correctly, not a defect in
the CRC.

The two valid header values are complements, so the test costs nothing to a mode
whose shift sense inverts on every transmission. Combined with the eye gate
(`p1rx.EYE_MIN`) it takes 6341 accepts to 5, and all 5 are the real packets above.

## 5. Packets need not sit on the 1.25 s ARQ raster

The three JN36lf packets are 0.975 s apart — the 0.96 s packet plus a ~15 ms gap,
back to back, with no room for a peer's control signal between them. That is not
the ARQ cycle of `pactor1-timing.md`.

A raster test is therefore not available as a validity gate: requiring one would
reject every data packet not sent inside an ARQ cycle. No recording of a data phase
*on* the raster is available; a decode of a link established locally would supply
one.

## 6. 200 Bd data packets

Nothing in the frame changes with the bit rate. A 200 Bd data packet is a 20-byte
field, a 5.00 ms bit period, CRC-16/X-25 over the 21 protected bytes, and the status
byte of §2 and header of §3 unaltered. Read back by an independent implementation, a
200 Bd packet yields the exact payload bytes at speed level 2:

    ###STATUS: SL: 2, CYC: 0, RQ: 0, REV: 1, LSB: 0, dF:  0.0, FRNR: 3
    ###PAYLOAD1: LEN: 20, TYPE: 4
    ###PAYLOAD2:
    BRAVO0 DE W9SSJ 200B
    ###PAYLOAD_END

**The speed level is carried by the packet, not by the link setup.** A connect
followed by two copies of one 200 Bd packet with nothing from the peer anywhere on
the air — no CS1, no acknowledgement — still reads as `SL: 2`.
`pactor1-timing.md` §7 says the connect answer selects the first data packet's
bit rate; that governs what the two stations must agree on, not how a listener
decides what it is hearing. The second copy reads `RQ: 1`, so a repeat is taken as
memory-ARQ rather than as new information — the header/counter lock of §3 survives
the repeat.

**A whole 200 Bd exchange decodes unbroken.** A connect, the peer's CS1 answer at
packet-end plus `d`, four new 20-byte packets on the 1.25 s raster, and the peer's
acknowledgements alternating CS1/CS2 between them: all four packets read `SL: 2`,
`RQ: 0`, `FRNR` 3 → 6.

**CS4 takes the link down to 100 Bd without a re-acquisition.** On CS4 after a link
that came up at 200, the 20-byte packet is discarded and the same information goes
out again as three 8-byte packets, the first reusing sequence 1. Frame numbering
runs continuously across the speed change:

    ###STATUS: SL: 2, ... FRNR: 7     LEN: 20   CHARLIE FALLBACK 200
    ###STATUS: SL: 1, ... FRNR: 8     LEN:  8   CHARLIE
    ###STATUS: SL: 1, ... FRNR: 9     LEN:  8   FALLBACK
    ###STATUS: SL: 1, ... FRNR: 10    LEN:  4    200

A 200 Bd packet whose field is mangled under the CRC the good field computed yields
no frame at all — the connect is read, the data is reached, and nothing comes out of
it. Ten frames is the complete output of the four segments above, in that order,
identically over six repeated runs.

**`LEN` is neither the field size nor the payload size.** Trailing 0x1E IDLE is
dropped and a CR counts two — the one rule that fits all ten observations across
both speeds. The 8-byte field `20 32 30 30 1e 1e 1e 1e` reports `LEN: 4`, and a full
20-byte field ending in CR reports `LEN: 21`. That the IDLE padding is dropped by an
implementation that is not ours is a second line on §3's `IDLE = 0x1E`.

The 200 Bd frame has not been checked against another station: no off-air 200 Bd
data packet exists in the corpus, and none has been received from a real station.

## 7. The shift phase is a property of the cycle, not of the transmission

"Die Shiftlage der FSK-Aussendung wird einmalig beim Verbindungsaufbau fixiert.
Mit jedem neuen Paket oder Kontrollsignal wird die Shiftlage invertiert." The
description reads like a per-transmission rule and it is not one. The per-cycle
epilogue inverts BOTH senses -- the one the station transmits in and the one it
reads in -- unconditionally, at the same place it advances both anchors by one
cycle, and it runs whether or not that station transmitted or received anything in
that cycle. There is no path around it: a failed receive, an empty slot, a retry, a
cycle spent listening, all advance the shift.

So a station counts cycles since the call it locked to, not things it has sent:

    shift(cycle N+k) = shift(call) XOR (k AND 1)

and within any one cycle the two directions share it -- the sense a called station
sends its control signal in is the sense it expects the caller's packet in, in that
same cycle. That is the observable form, and the one to test against.

**Measured on the air, transmitter off.** In an off-air recording of the WS8EOC
gateway the station repeats CS4 on its own raster with nothing to answer. Eight
bursts decode at zero errors over 20 s; taking the first as the reference phase and
folding the rest onto a 1250.32 ms period, the shift sense matches cycle parity on
all eight -- non-inverted on even cycles, inverted on odd -- with no exception and
no cycle where a repeat held its sense. A repeated control signal inverts like
everything else.

**A round trip cannot detect a wrong shift rule.** A listener that tries both senses
passes every packet, and a decoder that shares its own transmitter's rule agrees
with it at zero errors. The failure appears only against a station that counts
cycles. A packet in the wrong shift does not arrive as noise: its CRC region reduces
to the tabulated inverted-FCS constant for that length (0xC438 at 11 bytes, 0x77FF
at 23, 0x14A5 at 10, 0x1D9C at 21), which is the far end's own way of saying "shift,
not bytes". `tests/shrike/test_p1peer.py` asserts exactly that residue.

Two implementation requirements follow. A station advances the shift **once per
cycle**, including cycles it does not transmit in; advancing per transmission while
skipping alternate slots halves the rate and the phase never re-synchronises. And a
control-signal decoder must **report which sense matched** rather than trying both
and discarding the answer, or the station cannot observe the peer's phase even in
principle — the failure mode is twenty packets, CS4 every cycle, no advance.
