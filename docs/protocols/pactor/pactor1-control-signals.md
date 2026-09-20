# PACTOR-1 control signals

A control signal is a twelve-bit word the receiving station transmits in the
turnaround slot of each 1.25 s ARQ cycle. It carries the whole of the link-layer
response: acknowledgement, repeat request, speed change, and the handover of the
transmit direction. Twelve bits at 100 Bd is a 120 ms burst.

The normative description of the layer is the PACTOR Level-1 protocol
description — Helfert DL6MAA and Strate DF4KV, "PACTOR - Einfuehrende
Protokollbeschreibung (Level 1)", 5 November 1990, distributed as
`doc/pactor.txt` in the GPL `hf`/hfterm package. Sailer HB9JNX's hf-pactor
(`hfkernel/fsk/pactor.c` in the same package) is a complete PACTOR-1 ARQ
implementation that interoperated with commercial controllers; it is cited where
it fixes something the description leaves implicit, and where the two disagree
the disagreement is stated rather than resolved.

## 1. The codewords

    CS1: 0x4D5   CS2: 0xAB2   CS3: 0x34B   CS4: 0xD2C     (12 bits, LSB right)

The Level-1 description tabulates them with their names:

    CS1  Bestaetigung   4D5
    CS2  Bestaetigung   AB2
    CS3  Break-In       34B
    CS4  Speedchange    D2C

Two structural properties hold across the set:

  * the mutual Hamming distance is exactly 8, over all six pairs;
  * CS1/CS2 and CS3/CS4 are bit-reverse pairs — reverse(0x4D5) = 0xAB2,
    reverse(0x34B) = 0xD2C.

All four words have Hamming weight 6. A twelve-bit weight-6 word is DC-free on
the air: equal time on each tone.

The reversal property has a consequence for receivers. The set is closed under
bit reversal, so a decoder reading the word in the wrong order returns **zero
errors on the wrong codeword** — the error count is identical under both orders
and can never establish which order is in use.

## 2. Bit order on the air

The first bit transmitted is **bit 0** of the value as tabulated above: LSB
first. Every other PACTOR-1 field is LSB-first too — the address section, the
redundancy section, and the data packets — and the control signal is not an
exception.

hf-pactor lays the twelve bits into a byte buffer low byte first and hands that
buffer to the same FSK layer the data packets use; that layer takes stream bit
*n* as bit *n mod 8* of byte *n div 8*, so bit 0 of the tabulated value is the
first bit on the air. Its receiver is the exact mirror — the first received bit
is deposited in bit 0 of the first byte, and the assembled word is compared
against `0x4d5` and the rest as tabulated. Nothing in either path reverses
anything. The same file builds the CS3 packet head from the identical byte
layout, so a control signal and a changeover head carry the same twelve bits in
the same order.

## 3. Three further words in the same table

Implementations tabulate three more twelve-bit words alongside the four, as seven
little-endian 16-bit values and a terminator:

    d5 04  b2 0a  4b 03  2c 0d  a9 06  9a 05  2a 0b  00 00

    0x06A9  0x059A  0x0B2A

PACTOR-1 assigns them no meaning, and this document does not assign them one.
What can be said about them:

  * **all seven words have Hamming weight 6**, so the three extras are shaped
    like control signals — DC-free on the air — rather than like scalars that
    happen to share the table;
  * 0x06A9 and 0x059A sit at distance 6 from all four control signals and from
    each other, making {CS1..CS4, 0x06A9, 0x059A} a six-word code of minimum
    distance 6. PACTOR-2 has exactly six control signals. That correspondence is
    a shape and not an identification: 88 weight-6 words are equidistant from
    the four, so the pair is consistent with the reading rather than picked out
    by it. Checked against SCS's own specification, it does not hold: *The
    PACTOR-2 Protocol* (SCS, 1996) §2 gives the six CS as 40 bits on the air —
    20 DBPSK pulses on each of the two carriers — at mutual distance 24, exactly
    on the Plotkin bound, so the two 12-bit words cannot be PACTOR-2 control
    signals;
  * 0x0B2A is the exact **bit-complement of CS1** and does not fit that story —
    adding it drops the set's minimum distance to 4. A receiver that matches both
    shift senses, as every one here does, already reads 0x0B2A as a CS1 in the
    inverted shift; tabulating it as a word of its own renames half of every
    peer's acknowledgements. It is not in shrike's table for that reason;
  * **0x059A is on the air.** Scanning every control-signal-shaped burst in the
    corpus and in this station's own captures — 11227 candidates over 13.5 hours
    of audio, one trial each at zero errors — returns 55 readings of 0x059A, 49
    of them after discounting the recordings the corpus holds under more than one
    name, and none of the other two. They are not scattered: they arrive on the 1.25 s raster, in the answer
    slot, with the shift inverting cycle by cycle, in runs of up to eight cycles.
    Two are in third-party recordings — `PIII_Complete_1` at 4.412 s, where the
    caller's next burst is PACTOR-3, and `pos_pactor1_local` at 12.443 s, one
    cycle behind two clean CS4s from the station that owns the raster — and the
    rest are answers to this station's own packets, on 2026-07-30 and 2026-08-02.
    What it asks for is the PACTOR-3 upgrade. It arrives where the codeword
    answering an announcement packet would, it acknowledges that packet, and the
    station it answers keys PACTOR-3 in its very next burst — one turnaround
    later, not one cycle — with neither station moving its slot timing. A station
    still waiting for the entry packet repeats the word once a cycle for as long
    as it waits, which is why runs of it appear. `pactor3.md` §17.1 carries the
    measurement and what the granted station then transmits;
  * **0x06A9 is on the air too, and only ever as the other half of that answer.**
    A Winlink gateway on 30 m returned it fifteen times in fifteen consecutive
    cycles on 2026-08-30, at zero bit errors, to a PACTOR-1 announcement whose
    status byte set bit 4 and cleared bit 5; the same gateway four minutes later,
    answering the same announcement with bit 5 set as well, returned 0x059A
    instead and never 0x06A9. The two words arrive identically — the same answer
    slot 1030.8 ms after the caller's slot boundary, the same 120 ms burst, the
    same per-cycle shift inversion, the same repetition until the caller stops —
    so **bit 5 of the status byte selects which of the two comes back**, and
    nothing else measured separates them. What 0x06A9 asks for is not known: no
    recording of two commercial stations contains one, so it has only ever been
    drawn by a status byte no observed commercial announcement carries;
  * 0x059A is a bit-reversal palindrome and reads as itself in either bit order,
    so none of those readings is evidence about bit order.

## 4. What the codewords mean

The Level-1 description §3:

> CS1..3 have the same function as their AMTOR counterparts; CS4 serves as the
> speed change control. In contrast to AMTOR, CS3 is transmitted as head portion
> of a special changeover packet.

### 4.1 CS1 and CS2 — acknowledgement by alternation

CS1 and CS2 are both "Bestaetigung", and they are **not** two distinct meanings.
As in AMTOR/SITOR-A, the receiving station acknowledges by ALTERNATING between
the two: the acknowledgement is carried by the toggle, and repeating the previous
codeword is how a retransmission is requested.

    CS1 / CS2   acknowledge, alternating
                toggle  = that packet was good, send the next
                repeat  = send that packet again
    CS3         changeover
    CS4         speed change control

**There is no NAK codeword.** The description states the rule normatively:

> Die Kontrollsignale (CS) werden vom RX als Empfangsbestaetigung ausgesendet.
> **Wiederholung des gleichen CS bedeutet 'REQUEST'**, d. h. die Anforderung
> einer Blockwiederholung.

A receiver that waits for a distinct negative acknowledgement waits forever: a
constant CS2 reads as "send that again", indefinitely.

**The alternation is the packet counter, not a toggle of its own.** Bits 0-1 of
every status byte number the sending station's packets mod 4 (§7), and the
receiving station's codeword is a function of that number: **even → CS1, odd →
CS2**, held on the air while no new counter arrives. The two statements are the
same statement — a counter that advances is a codeword that changes, a counter
that stands is the codeword repeated, which is the Request — but only the counter
survives a changeover, because a changeover restarts it. §4.2 is where that
matters.

Nothing in the 1990 text says this in one sentence; it is what the two halves it
does say come to, and every observation agrees. A gateway on 30 m answered thirty
consecutive PACTOR-3 packets on the parity of their counters at zero bit errors
(`pactor3.md` §17), and PACTOR-1 carries the same two bits in the same place.
VE1YZ demonstrated it in PACTOR-1 against this station on 2026-09-02, both
parities inside twelve seconds: our packet #1 acknowledged CS2, our #2
acknowledged CS1.

### 4.2 CS3 — changeover, and never bare

CS3 is not a bare control signal. It *is* the first 2 bytes (100 Bd) or 3 bytes
(200 Bd) of the new ISS's first packet: the IRS sends it after a correctly
received packet, the old ISS switches to receive on seeing it and reads the rest
of that packet, then answers CS1. A repeat of the break-in packet is requested
with CS2.

Two bytes at 100 Bd is 160 ms, so a changeover head is longer on its own than a
control signal is, and is followed immediately by the remainder of the packet
rather than by clear air.

**The changeover restarts the counter, and the alternation with it.** "Der
Paketzaehler wird auf 0 gesetzt" — the new ISS numbers its first packet 0, so by
§4.1 that packet is acknowledged CS1, its repeats hold CS1, and the counter-1
packet behind them draws CS2. A codeword phase carried over from the stint before
lands on either face, and one of the two faces is CS2: the request to send the
break-in packet again.

**And CS2 is what the waiting cycles carry.** Between the CS3 head decoding and
the changeover packet decoding, the old ISS owes a codeword and has accepted
nothing under the new numbering, so it keys CS2 — the request — until the packet
arrives, and CS1 on the cycle it does. hf-pactor's `tx_rx_100` is that loop
literally: `send_cs(2)` at the foot of every iteration, with the CS1 branch
reached only once `receive_packet` returns. Its mirror `rx_tx_100`, the station
repeating the break-in packet, tests for an exact CS1 and nothing else — not even
a one-bit-error CS1, which the steady-state loops do accept — so a CS1 keyed
early is not merely premature: it is the word that says the packet arrived.

MEASURED, and the two faces of the coin are the same seven bytes. KB5LZK's
break-in packet on 2026-08-22 — `status=0x00 (cnt=0) 7B 'RMS Tri'` — was answered
CS1 and the gateway walked its whole banner through, counters 1, 2, 3, 0, our
codewords CS2, CS1, CS2, CS1 behind them (KB5LZK, 2026-08-22, the 40 m night
arm). The byte-identical
packet from VE1YZ on 2026-09-02 was answered CS2, held for 96 cycles, and the
gateway sent those seven bytes 45 more times and nothing else until the session
was torn down (VE1YZ, 2026-09-02, the mail arm).
Same packet, opposite codeword, opposite outcome.

**What separated the two arms was position in the link, not anything in the
packet.** The old alternation stepped once per codeword it chose fresh — the seed
keyed before anything had been accepted, or the acknowledgement of a newly
accepted packet — and a codeword held, or a changeover packet keyed in an
acknowledgement's place, stepped nothing. So the face a counter-0 packet drew was
the parity of that count: even at KB5LZK, where the first changeover of the link
had already spent two of them, odd at VE1YZ, where it had spent one. **The
corpus's empty changeover packets were answered correctly by the same arithmetic
and not for a better reason.** All four on file — WS8EOC twice on 2026-08-29,
VE1YZ twice on 2026-09-02 — have the gateway breaking in before this station had
accepted anything, so the answer came off the CS1 seed, and the zero-byte
`cnt=1 BK` packet behind each of them was answered by our own changeover packet
rather than by a codeword. Under the counter the empty packet and the banner are
one statement — counter 0 is even — which is what KB5LZK flew on 2026-09-03: CS1
against the empty counter-0 packet, the changeover packet against the counter-1
invitation behind it.

### 4.3 CS4 — speed change, and the ambiguity in it

CS4 is context-dependent rather than simply "speed change":

  * after a **bad** packet it is interpreted by the sender as **REJECT** —
    discard and repeat at 100 Bd;
  * after a **good** 100 Bd packet it is an acknowledgement that forces 200 Bd;
  * consecutive CS4s with no valid CS between them are read as a plain Request
    and ignored: "CS4 in Folge (ohne zwischenzeitliches richtiges CS) werden als
    'Request' interpretiert, also ignoriert."

Both branches are normative. §Geschwindigkeitsverminderung: the receiving station
"kann **immer** nach einem fehlerhaften Paket ein CS4 senden, was auf der TX-Seite
**immer** als 'REJECT' interpretiert wird". §Geschwindigkeitserhoehung: it "kann
nach jedem richtig empfangenen 100-Bd-Paket ... mit CS4 bestaetigen, was den TX
zur Umschaltung auf 200 Baud zwingt".

The distinguishing fact — whether the packet arrived — is the one the sender does
not have. The protocol resolves this by the speed the link is already running at,
which is context the sender does have:

  * an ISS at **200 Bd** that reads CS4 drops to 100 Bd and does *not*
    acknowledge the packet: the REJECT branch;
  * an ISS at **100 Bd** that reads CS4 acknowledges the packet and sends the
    next one at 200 Bd: the speed-up branch.

There is no state in which the sender has to guess. On the REJECT branch the
unacknowledged packet is re-sent at the lower rate: "Das gerade gesendete, nicht
bestaetigte Paket wird verworfen und die in ihm enthaltene Information erneut im
100-Baud-Modus ausgesendet." One 20-byte 200 Bd field re-chunks into three 8-byte
100 Bd packets carrying the same sequence number.

The speed-up branch has a timeout of its own. After a preselected number of
cycles with no clean 200 Bd packet, the receiving station sends "einem OK-CS
(verschieden vom letzten CS vor dem CS4)" — a fresh codeword, distinct from the
last CS before the CS4 — and the link drops back to 100 Bd.

One further rule constrains the first block of a session:

> CS4 dient als 'REQUEST'-CS fuer den ersten 100-Bd-Block, Bestaetigung erfolgt
> mittels CS1 oder CS3, danach kann bereits wieder ein CS4 als 'speedup'-Signal
> gesendet werden.

After a CS4 connect answer, the acknowledgement of the **first** 100 Bd block
must be CS1 or CS3; a CS4 in that position is still the request. hf-pactor
implements exactly that — its master, in the state a CS4 connect answer puts it
in, tests for CS1 and CS3 and for nothing else, and repeats the packet on a CS4.
The 100 Bd speed-up branch only becomes reachable once that first block has been
acknowledged.

## 5. The connect answer

The 200 Bd redundancy section of the link-request frame is a channel-quality
measurement, and the answer reports the verdict on the channel rather than on the
caller:

> Das 200 Bd-Muster dient lediglich zur Ueberpruefung der Kanalqualitaet. Wird es
> fehlerfrei empfangen, wird mit CS1 geantwortet, ansonsten mit CS4, was zur
> sofortigen Geschwindigkeitsreduzierung auf 100 Baud fuehrt.

    CS1   the 200 Bd redundancy section arrived error-free   link runs at 200 Bd
    CS4   it did not                                         link runs at 100 Bd

The English translation at ecjones.org reverses this, giving CS4 for a good
section and CS1 for a bad one. The German original, the ARRL description (Karty
N5SK) and hf-pactor all agree against it.

**Only CS1 and CS4 are connect answers.** A calling master compares the received
word against CS1 and CS4 only — and against their complements, which are the
other shift polarity — and keeps calling on anything else; CS2 and CS3 are not
tested. The answer's whole content is the speed flag `cs == CS1`.

The answer is a rate selection, not merely a capability probe. A master that read
CS1 sets the link to 200 Bd and its first data packet is 24 bytes at 200 Bd; a
master that read CS4 sets 100 Bd and sends 12 bytes. The data field is 20 bytes
against 8, and either packet is 960 ms on the air — 192 bits at 200 Bd, 96 at
100 Bd (`pactor1-timing.md` §0). The figures 23 and 11 name a different
region: the packet minus its header byte, i.e. field + status + CRC, the span
whose inverted-FCS residue is tabulated in `pactor1-data-packets.md` §7. An earlier
revision of this paragraph used them as the packet sizes; the on-air packet is
header + field + status + CRC = 24/12.

**The protocol has no refusal** — neither a codeword nor a state. hf-pactor's
responder acts only on an exact match against `mycall`; a near miss and a call for
another station are both dropped without transmitting, and every site that can
send a control signal lies inside an already-established or already-terminating
QSO. The Level-1 description likewise describes only the successful connect. A
station that declines a call says nothing at all; a station that repeats a
codeword at the caller is engaged with it, not refusing it.

An accepting station sends its answer **every cycle** until the caller's first data
packet decodes, up to a 30-cycle retry budget, so a long unbroken run of CS4 is
what a healthy station waiting on a caller looks like. hf-pactor hard-codes its
200 Bd channel score to zero — "currently 100 baud only monitor, therefore no 200
baud connects" — and so answers every call it accepts with CS4 and can never
answer CS1.

**Corollary: acceptance of the first data packet is a CHANGE of codeword, and a
run of the answer means "not yet".** If a station answering CS1 also acknowledged
with CS1, no caller could distinguish "still waiting for your packet" from "got
it" and no link could ever start. So an unbroken run of the connect answer at zero
errors says exactly one thing — the caller's first data packet has not decoded,
once per cycle for the length of the run.

**The counter says the same thing here**, which is why the two rules are one:
"das erste normale Datenpaket mit Head=AA (HEX) und Paketzaehler=1", so the
caller's first packet is counter 1 and §4.1 answers it CS2. Before it decodes the
called station has accepted nothing and holds the codeword of counter 0, which is
CS1 — the answer. A link therefore opens on CS1 held and a changeover on CS2 held
(§4.2), and neither is a seed anyone has to keep: they are the two numberings,
one counting from one and the other from zero.

**§4.3's first block is the one position outside the counter.** After a CS4
answer the first 100 Bd block is acknowledged CS1 or CS3 whatever counter it
carries — hf-pactor's master, in the state that answer puts it in, tests for
those two and nothing else, and its responder answers that block CS1 at counter 1.
Nothing here grades the redundancy section, so this station answers every connect
CS1 and never keys into that branch; it reads one, in `§4.3`'s REJECT.

The opposite reading is the natural one and it is wrong in a costly direction: it
makes a refusal look like an acknowledgement the caller is deaf to, and the repair
it suggests — not comparing the first in-session codeword against the answer — walks
the packet counter over a packet the peer never received. Measured against WS8EOC on
2026-08-02: eight consecutive CS1 at zero errors, `#1 x9`, in four sessions. That is
eight decode failures, not eight acknowledgements.

**UNBROKEN is load-bearing, and a CS4 breaks it.** A CS4 arriving on a 200 Bd link
is the REJECT, and the block re-chunked down behind it is the first 100 Bd block,
whose acknowledgement §4.3 names outright as CS1 or CS3. A CS1 there is that
acknowledgement and not a repeat of the answer, however many cycles ago the answer
was. §9.3 is the measurement.

### 5.1 Commit on first answer, or verify

The two references disagree on when the master may commit. The Level-1
description sends the first data packet "sobald das erste gueltige CS empfangen
und synchronisiert ist" — on the first valid control signal. hf-pactor instead
runs a verify cycle: it retransmits the call, requires the same codeword a second
time in the correctly toggled shift polarity, and only then connects, rejecting
an inverted repeat outright.

The verify cycle does not add the guard it appears to add. A station free-running
CS4 on the 1.25 s raster passes a same-codeword-twice test exactly as easily as a
single-answer test. The guard that bites is restricting the answer alphabet to
CS1 and CS4.

## 6. Cycle, placement and shift polarity

The ARQ cycle is 1.25 s. A control signal occupies about 0.12 s of it, leaving
1.13 s clear — enough for a 0.96 s packet with 170 ms to spare. The answer falls
at packet end + 105 ms.

The receiving station's grid is the link's master clock: it does not lock to the
caller. A measured station's first answer sat 63 ms off the grid it then kept,
re-anchored once, and never resynchronised while the caller's transmissions
drifted ~16 ms per cycle past it. A caller therefore synchronises to the peer's
bursts rather than to its own clock: detect a burst at T and transmit from about
T + 0.15. The clear window runs from R + 0.122 to R + 1.25, and a 0.96 s packet
fits inside it with 153 ms to spare. Transmitting once per 1.25 s cycle is
required; a 2.5 s cadence leaves the peer an empty cycle and walks its read
anchor.

The answering station uses 154-230 ms of the 0.29 s turnaround window, leaving
60-136 ms of margin.

**The shift polarity inverts with every new transmission** — "Mit jedem neuen
Paket oder Kontrollsignal wird die Shiftlage invertiert". The inverted form of a
control signal is the bit-complement of its codeword, so a receiver must accept
both. Successive bursts alternating in polarity is what identifies them as one
station's consecutive transmissions.

Keying is a lead-in of **steady tone**, not silence, and the carrier drops at the
END of the data. A burst therefore runs longer than its word: 145 ms and 115 ms
have been measured at two stations for the same 120 ms word, with the true bit
origin a consistent 19 ms into the longer one.

### 6.1 A codeword's one and a data byte's one ride the same tone

The codewords of §1 are tabulated in the **same bit sense as a data field**. That
is a measurement, and it is not one a round trip can make: a transmitter and a
receiver that both complement the codeword agree at zero errors on a signal one
cycle out of phase with the band.

What settles it is a frame and the control signal that ANSWERS it, inside the same
cycle, where the two directions share the shift (pactor1-data-packets.md §7). W4DNA
retransmits its link-setup packet on three consecutive cycles in
`pos_pactor1_local.wav` and the peer answers each 110 ms later. Off the raw
1400/1600 Hz magnitudes, with no decoder involved:

    packet t= 7.616 + CS4 at +110.0 ms    frame 1 = 1600    codeword 1 = 1600
    packet t= 8.865 + CS4 at +108.8 ms    frame 1 = 1400    codeword 1 = 1400
    packet t=10.114 + CS4 at +115.0 ms    frame 1 = 1600    codeword 1 = 1600

96 frame bits and 12 codeword bits per cycle, agreeing at 0% or 100% and never in
between, over three cycles of alternating polarity. A DL6MAA connect answered by
CS1 at +42 ms in `PIII_Complete_1.wav` gives the same answer through the two
decoders. No counterexample.

**The consequence is on transmit, not on receive.** A receiver tries both senses
and decodes either way, so a station can read every control signal on the band and
still be wrong about this. But the peer's control signal is the only reading of the
phase it counts on, so a station takes its own next shift from it — and a
control-signal sense one parity out of step with a frame's puts every DATA PACKET
in the wrong shift, from the first answer onward. The packets are not noise at the
far end: their CRC region collapses to the inverted-FCS constant, the peer repeats
its codeword rather than alternating, and both stations behave correctly forever.
That is WS8EOC on 2026-08-02 — connect accepted, ten cycles, five control signals
at zero errors, CS1 CS1 CS1 CS4 CS4, and a packet counter that never moved.

## 7. The data packet's status byte

**The PACTOR-1 data mode field is 2 bits, at positions 2-3.** The 3-bit field at
positions 2-4 belongs to PACTOR-2 and PACTOR-3; a receiver dispatches on protocol
level before extracting it.

The 2-bit table extends the 1990 document, which lists only 00 and 01 and calls
the rest "nicht belegt":

    00  8-bit ASCII        01  Huffman, normal
    10  Huffman, swapped   11  reserved -- no decoder

Packet acceptance checks CRC-16/X-25 and the expected counter in `status & 3`.

This document does not specify what bits 4-5 meant to the 1990 authors ("noch
nicht belegt"), and why a station sets them is not settled — one possibility, on
a single observation, is a maximum-system-level indication (the description's
"Levelfestlegung"): the one recorded session in which they are set goes on to
PACTOR-3 while the ASCII level digit stays "1" for compatibility. They do **not**
carry the 100/200 Bd speed — a receiver takes that from an internal per-packet
length code and reports a packet with bits 4-5 set as speed level 1 regardless.
But setting them is not inert: no receiver gates *acceptance* on them, yet an
independent monitor built after 1990 reads bit 4 as the third bit of the data
type and bit 5 as the long-cycle request, so status 0x31 declares PMC-German
compression and asks for a 3.75 s cycle. A transmitter sends them clear —
`pactor1-data-packets.md` §2, which is the normative statement of both rules
(receivers must not gate on them; transmitters must not set them).

Two further packet-level rules bear directly on the control-signal semantics:

  * "aufeinanderfolgende Pakete mit gleichem Zaehlerwert werden vom RX als
    REQUEST-Pakete erkannt" — consecutive packets carrying the same counter value
    are recognised as REQUEST packets;
  * the header is inverted only "bei jedem Paket, das neue Information enthaelt"
    — on each packet carrying new information.

## 8. Receiving a control signal

A control signal arrives as a bare 120 ms burst of 1400/1600 Hz two-tone, twelve
bit decisions at 10 ms each, with no framing of its own. Two properties of the
signal shape how it must be read.

**Burst envelope does not locate the bit grid.** An energy threshold on a 5 ms
hop reports the run start 5 to 36 ms early — up to three and a half bit periods —
because the keying lead-in is steady tone and is indistinguishable from a data
bit to anything that only asks how open the eye is. The unkey is what separates
them: sliding the window and keeping the alignment where the eye is widest AND
the two bit slots after the word are quietest converges on the true origin.
Measured against the alignments that decode, over 41 bursts from four stations:
with no tail term, mean -5.3 ms and sd 9.4 ms; with one slot, -1.1 and 5.7; with
two slots, **+0.5 and 2.9**, 41 of 42 inside half a bit.

**Three to five bit errors is the signature of sampling on the transitions.** A
control signal is a balanced word, so a grid a half-bit out puts most decisions
on an edge and the errors land where the word changes tone. The result reads as
too many errors for noise and too few for garbage because it is neither: it is a
correct signal read at the wrong instants. Every clean burst has an alignment at
which it reads a codeword exactly, and holds it over a plateau 5 to 9 ms wide — a
genuine eye. Bursts with no zero-error origin anywhere are a different frame type
and stay refused at every offset; the two populations do not overlap.

The origin search must not consult the codeword table. One alignment reaches the
comparison, so the distance-8 margin is spent on a single trial; across 123
candidate bursts raised on signals that are definitively not PACTOR-1, such a
search accepts none.

Two stations on the same frequency are separable by tone offset rather than by
duration: a measured pair sat at 1393.65/1593.70 Hz and 1389.98/1590.00 Hz, a
3.69 Hz difference resolved to ±0.4 Hz.

## 9. Observed practice

The raster is the protocol's and not one station's habit. Two stations on
different frequencies, recorded with no transmission from the observer:

    WS8EOC   7100.0 kHz   31 bursts in 77 s
             intervals, ms: 1250 1250 1245 1255 1245 1255 1245 1255 1250 1265
    W6IDS    7102.3 kHz   62 s
             intervals, ms: 1255 1245 1250 1250 2500 1250 1250

Exactly 1.25 s, ±10 ms, with one missed slot showing up as a doubled interval.
Bursts fall on `2.23 + n * 1.25` relative to the observer's final unkey. A
third-party recording of a full session measures the same cycle as **1249.97 ms**,
residual rms 2.7 ms across 34 control signals over 41 s, identical in the first
and second halves, one control signal per cycle at 0.122 s in all 34 cycles,
never doubled and never sped up, with the tone sense inverting on every single
cycle across 35 bursts.

That session's codeword train is CS1 for four cycles, then CS4 repeated 31 times:
a valid connect answer followed by a reject held to timeout. The station answered
the call CS1 twice, matched the caller's 2.5 s cadence for as long as the caller
transmitted, and returned to the 1.25 s raster 1.49 s after the caller's last
packet ended — CS1 at t=23.07 and 25.58, CS4 every 2.499 s from t=28.07 to 40.57,
then CS4 every 1.250 s from t=41.82 to 64.32, 24.0 s of it after the caller
stopped. It closed by identifying in CW at 1600.28 Hz, 20.7 WPM: WS8EOC.

Zero-error control signals across all available recordings:

    7101 kHz W4DNA/KE5YTA exchange            2 x CS4   (4 with a per-burst origin)
    20 m sweep, 2026-07-25                    1 x CS1, 5 x CS4
    third-party session recording             5 x CS1, 28 x CS4  (two stations)
    WS8EOC, observer silent                            16 x CS4
    W6IDS, observer silent                             18 x CS4

The two observer-silent recordings begin at the moment the observer stopped
calling, so every control signal in them is an in-session repeat and none of them
is a connect answer.

Across every station in every recording: not one CS2, not one bare CS3, and not
one alternating pair. The pattern is always the same — CS1 at most once at the
top of a session, then CS4 held, one per 1.25 s cycle, until the far end gives
up. An alternating pair is what an acknowledged link looks like, and no recording
holds one.

### 9.1 A link that does not advance

A 7101 kHz exchange — W4DNA calling the Winlink station KE5YTA — is audible and
decodable on both sides for its PACTOR-1 phase, 0-13 s, after which the pair
upshift and finish in PACTOR-3:

    0.505  1.755  3.008   963-969 ms   connect packets, address KE5YTA
    4.082               109 ms         answer, zero errors, 0xD2C
    5.122  6.371  7.623  8.869  10.116  11.373   963-971 ms   data packets
    6.199  7.446  8.693  9.946  11.190  12.260   answers, one per cycle
    12.442              121 ms         last answer before the PACTOR-3 pilot at 13.44

Both rasters are the protocol's: data starts 1.25 s apart to within 7 ms over
five cycles, answers 1.25 s apart to within 9 ms, and each answer sits at packet
end + 105 ms.

The answering station sent one codeword and only that codeword. Four of its seven
bursts decode at zero errors and all four are `0xD2C` — CS4; the other three are
refused at 1, 1 and 3 errors. The shift sense inverts between consecutive answers
(t=8.693 one sense, 9.946 the other, 11.190 back again), which is what identifies
them as one station's successive transmissions rather than four coincidences.

The calling station's five decodable data packets are byte-identical: payload
`1w4dna\r`, status `0x31`, header `0xAA`, 100 Bd, five cycles running — packet
counter 1 every time, and the header never inverting. Both are defined behaviour
for a repeat. A commercial modem, hearing a repeated codeword, retransmitted a
REQUEST packet on every cycle for five cycles and never advanced. `Q-6.0` and
`Welcome W4DNA  QTC` are PACTOR-3 payloads carried after t=13.4 s; nothing in the
PACTOR-1 phase was ever acknowledged.

### 9.2 A connect answered CS1

One recording carries a complete PACTOR-1 connect exchange with both sides
audible, ahead of its PACTOR-3 data:

    2.120 + 2.400   one 960 ms call packet   (255 ms + 710 ms runs, adjoining)
    3.140           120 ms burst, 30 ms after the call ends
    3.370 + 3.600   990 ms, the same shape — not the same call
    4.395           120 ms burst — 1.255 s after the first

Two answers on the 1.25 s raster, each in the turnaround slot. The burst at
t=3.140 reads **CS1 at zero errors** over a 9.7 ms span of grid offsets: six ones
and six zeros, 1400/1600 Hz, transitions at 10 and 20 ms. CS1 is what a connect
is answered with when the 200 Bd redundancy section arrives clean. The burst at
t=4.395 has the same tones and the same 10/20 ms transitions but reaches no
codeword below one error at any alignment within ±3 bits.

**This document does not identify the second 990 ms burst.** Its median tone
transition interval per quarter-burst is the first burst's profile in reverse —
200 Bd leading, 100 Bd trailing — which no shift inversion produces:

    t=2.120  990 ms   10.08 10.10 29.79  5.08   the call: 100 Bd body, 200 Bd tail
    t=3.140  140 ms                             CS1, zero errors
    t=3.370  990 ms    5.00 15.10 19.93 10.04   unidentified
    t=4.395  135 ms                             no zero-error codeword

No address decodes out of it at any of 40 window origins swept over 3.20-3.60 s
at three window lengths, and no CRC-valid packet is present at either speed. It
is not a repeated call.

### 9.3 The same gateway, ninety minutes apart, on either side of §4.3

K4MSU on 3595 kHz, 2026-08-19, two sessions to one Winlink gateway whose
turnaround held at 92-96 ms across the evening. Every window of both is on one
sample clock in `captures/onair-0819-2048` and `captures/onair-0819-2207`, and
every codeword below reads at zero bit errors.

    20:48   CS1  CS4 CS4 CS4  CS1 ×11              #1 ×16, nothing acknowledged
    22:07   CS4 ×6            CS1 CS1 CS2          #1 ×10, #2 ×5

Both trains are one gateway answering, and they differ in the codeword it answered
the connect with. The CS1 answer put the link at 200 Bd, its CS4 was the REJECT,
and the eleven CS1 behind it are the first 100 Bd block being acknowledged and
then requested — §4.3, not the alternation. Read as alternation against a CS1
answer they are eleven repeats and the counter cannot leave #1, which is what the
20:48 session transmitted. The CS4 answer at 22:07 put the same rule the other way
up: the reference is a CS4, so the same acknowledgement is a change of codeword
and the counter advanced on the cycle it arrived.

The 22:07 CS2 is the alternation proper, in `hold_14`, and it never reached that
session — the grid-anchored read is refused once the state leaves `CONNECTED`, and
the QRT had gone out two cycles earlier.

## 10. Where shrike diverges from PACTOR

These are implementation choices in shrike, not properties of the protocol.

**Both halves of CS4 are implemented** (`ptc._logical_cs`), by the speed rule of
§4.3 and not by a guess: at 200 Bd it is the REJECT, which drops the link to 100
and re-chunks the unacknowledged field; at 100 Bd it acknowledges the packet and
the next one goes out at 200 with the 20-byte field. Consecutive CS4s are a plain
Request either way, and because the connect answer seeds that comparison, a CS4
train behind a CS4 answer is the first-block request of §4.3 rather than a speed
offer. The REJECT arms the other half of the same clause (`_p1_first_block`): the
codeword that takes the re-chunked block is read as CS1 outright, so the seed from
a CS1 answer cannot veto it (§9.3).

**A speed-up CS4 consumes an acknowledgement position.** hf-pactor makes the
two following codewords distinct: `tx_100_cs1` takes CS4 to
`tx_100_200_cs2`, where CS1 acknowledges the first 200 Bd packet and CS2 declines
the speed-up, repeating its information at 100 Bd. The opposite phase is the
mirror (`tx_100_cs2` → `tx_100_200_cs1`). The refusal is therefore the OK-CS
"verschieden vom letzten CS vor dem CS4" specified in §4.3; the successful
200 Bd acknowledgement has the same wire value as that earlier CS. The implicit
acknowledgement carried by CS4 is what separates them.

Shrike now follows those states. Previously it advanced the packet counter on
CS4 but left its acknowledgement reference unchanged. K0NTS and KB5LZK exposed
the result on 2026-09-10: both sent CS1 after accepting packet 2 at 200 Bd, and
shrike repeated packet 2 because it compared that CS1 to the one before CS4.
The recorded controls and host/ARQ regressions are indexed in
`tests/shrike/fixtures/p1-speedup.json` and `tests/shrike/test_p1_speedup_seam.py`
under `packages/hfmodem/hfmodem/`.

One additional bound remains: four consecutive repeat requests during an
unacknowledged speed trial put shrike back at 100 Bd
(`ptc.P1_HISPEED_RETRIES`). A decoded refusal acts immediately and does not
wait for that count. The ordinary retry budget counts silence, so a held CS4
alone would otherwise leave the trial open indefinitely.

Transmitting at 200 Bd is implemented: `pactor1.packet_signal(baud=200)` renders
the 20-byte field, and the connect burst's own 200 Bd redundancy tail is copied
clean by real stations.

**shrike commits on the first valid connect answer**, following the Level-1
description rather than hf-pactor's verify cycle (§5.1).

**shrike clears status bits 4-5** (`pactor1.status_byte`). An earlier revision
set them, on the ground that this reproduced one recorded packet (W4DNA, status
0x31) byte for byte — but that packet was never acknowledged, and clearing them
is correct under both the 1990 two-bit reading and the later three-bit one.
`pactor1-data-packets.md` §2 carries the evidence.

Interoperating details that follow from the sections above, recorded so they are
not re-derived: `ptc._answer_link_setup` sets the link to 200 Bd on CS1 and 100 Bd
on CS4, with the field size at 20 bytes against 8; `ptc._p1_cs_for` starts the
CS1/CS2 alternation on CS1, since the first acknowledgement a called station sends
is its answer to the connect and only CS1 and CS4 are answers there;
`ptc._logical_cs` drops to 100 Bd and requeues the unacknowledged packet on a CS4
received at 200 Bd, re-chunking its 20 bytes into three 8-byte packets at the
reused sequence number, and raises the link to 200 Bd on one received at 100.
shrike's connect burst is 720 ms at 100 Bd followed by a 240 ms
tail at 5.00 ms/bit, with no postamble — structurally identical to a real call
packet, which measures 10.08, 10.10 and then 5.08 ms across quarters of its
length.
