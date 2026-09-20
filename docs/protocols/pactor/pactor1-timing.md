# PACTOR-1 link timing

This document specifies PACTOR-1 link timing from the master's point of view: the
1.25 s ARQ grid, what each station's transmit and receive instants are referenced
to, how the grid is acquired at link setup, and the tolerances a receiver has.

Two descriptions of the protocol are cited.

The **1990 Level-1 description** by Helfert DL6MAA and Strate DF4KV (5 November
1990), shipped as `doc/pactor.txt` in the GPL `hf`/hfterm package, is the protocol
authors' own and is normative here.

**hf-pactor** — `hfkernel/fsk/pactor.c` (Sailer HB9JNX, 1997, GPL-2), with its
layer-1 scheduler `fskl1.c` and `fskutil.c` — is a complete PACTOR-1 ARQ
implementation that interoperated with SCS stations. It is cited for behaviour the
1990 description leaves unstated. Where the two disagree, the difference is called
out: the implementation shows what real stations interoperated with, the document
shows what was meant.

ITU-R M.1798 restates the 1.25 s ARQ cycle as common to every PACTOR mode
[M.1798 §3].

## The timing budget

PACTOR-1 was specified in 1990 and ran, interoperably and for decades, on 8-bit
microcontrollers clocked in the single-digit megahertz with a few kilobytes of RAM,
driving 1980s transceivers over relay T/R switching. Every timing requirement in
this document was met by that hardware with margin to spare. A host sampling at
48 kHz with CAT keying has three to four orders of magnitude more in every
dimension that matters:

| requirement | protocol | at 48 kHz with CAT PTT | margin |
|---|---|---|---|
| bit period, 100 Bd | 10 ms | 480 samples at 48 kHz, 20.8 us each | 480x resolution |
| bit period, 200 Bd | 5 ms | 240 samples | 240x |
| turnaround `d` | 40-180 ms, measured per peer (§0) | CAT PTT 0.0-0.2 ms through persistent rigctl | >200x |
| ARQ cycle | 1250 ms | whole-cycle decode in the low hundreds of ms | ample |

Every constant in this document is a protocol value. Padding one for safety changes
the protocol: it places a transmission where the peer is not reading, and the peer
has no mechanism to go looking for it.

## 0. The cycle, stated as one grid

Everything below is one 1.25 s grid, and every time is relative to the start of
the **data bits** of the information packet. Keying preamble is not data and does
not move the grid.

    t = 0        ISS packet begins   (96 bits @100 Bd, or 192 bits @200 Bd)
    t = 960 ms   ISS packet ends
    t = 960+d    IRS control signal begins   (always 12 bits @100 Bd)
    t = 1080+d   control signal ends
    t = 1250 ms  next cycle: ISS packet begins again

`d` is the turnaround gap. The 1990 description does not name it; it gives the four
durations and leaves a residual of 170 ms "for switching operations and signal
propagation", which is exactly `1250 - 960 - 120`.

    Gesamt-Zyklusdauer                 : 1.25 sec
    Paketdauer                         : 0.96 sec
    Fenster fuer Kontrollsignalempfang : 0.29 sec
    Kontrollsignaldauer                : 0.12 sec
    Es verbleibt eine Restzeit von 170 ms

**`d` is a property of the peer and of the path. It is measured; the protocol
specifies no nominal turnaround and no station may hardcode one.**

### The changeover constant

At a direction changeover both stations rotate the grid by a single fixed constant
of **packet duration minus control-signal duration, 960 − 120 = 840 ms**, applied
to the receive anchor by the station becoming ISS and to the transmit anchor by the
station becoming IRS. The description states it — the answering control signals
begin *"eine CS-Laenge (0.12 sec) vor Ende des alten eigenen TX-Blocks"* — and both
its terms are speed-invariant: every control signal goes at 100 Bd whatever the
packet's speed, and `Paketdauer` is 0.96 s at both. hf-pactor writes the same
constant as "packet length in bits minus control-signal length in bits", which is
the right number only at 100 Bd; the 200 Bd packet's own bits would say `192 − 12`
= 900 ms, and nothing on the air or in the description says 900. Reading it as
durations is also what makes the PACTOR-3 form sound: 810 − 210 = 600 ms.

Solving the grid either side of a changeover, that constant maps the turnaround gap

    d  ->  170 - d + 2p       (p = one-way path delay)

which is an involution, so the pair stays mutually consistent for **any** `d`. The
involution selects no particular value: its fixed point, `d = 85 + p`, moves with
the path, and nothing requires a turnaround to survive a changeover unchanged.

### Observed turnarounds

Three figures, none of them the same:

  * hf-pactor's responder answers at `packet end + txdelay + 10 ms`, which with its
    shipped `txdelay = 30 ms` (`hfterm/src/params.c`) is **`d = 40 ms`** — from an
    implementation that interoperated with real hardware;
  * the 1990 description mandates no offset at all. It gives the master a 290 ms
    window and has it SEARCH — see §3, which is what the protocol does instead of
    agreeing a turnaround;
  * WS8EOC, a live SCS gateway, answered at **172-219 ms, median 181**, taken off
    the recording. shrike's own tracker reported 82.9-83.1 ms for the same gateway
    in the same session; the two figures are referenced to different instants and
    this document does not reconcile them.

The observed range is therefore 40-180 ms, two full control-signal lengths wide.

**A master measures `d` per peer and holds it.** hf-pactor's master does — see §2.
An implementation that needs a starting value opens the whole 290 ms window rather
than guessing a point inside it.

## 1. What the master's transmit timing references: its own free-running clock

**The master's transmit instants are a free-running local 1.25 s grid, anchored
once at connect from the local clock and never referenced to anything the far end
does.**

The 1990 description says so directly, in the systemclock section:

> Der SLAVE-Takt wird auf den MASTER-Takt synchronisiert (Auswertung der
> Flankenwechsel).

The answering station's clock is synchronised to the master's, by evaluating edge
transitions. Nothing synchronises the master.

hf-pactor's timing-recovery function measures a sub-bit deviation from the received
signal's edge transitions, low-pass filters it, and applies the correction to the
**receive** anchor unconditionally and to the **transmit** anchor **only when the
station is not the master**. The identical asymmetry appears in frequency tracking:
the master corrects only its receive frequency offset, the responder corrects its
transmit frequency too and therefore comes to the master's frequency.

The master's grid origin is the local clock plus a 200 ms lead-in, and thereafter
every advance is exactly one nominal cycle constant added to the transmit anchor —
during link setup, and for the whole established link by the per-cycle epilogue,
which adds the cycle constant to receive and transmit anchors together and inverts
both shift polarities.

**The far end's clock is a follower with about 12% gain per cycle.** The filter is
an eight-deep ring of past deviations; each cycle it applies the mean and
subtracts that mean from every stored entry, which is a first-order loop of gain
~1/8 per cycle, time constant ~10 s. The ring is zeroed at connect, so the first
eight cycles are further attenuated. The per-measurement input is itself bounded:
the edge-transition estimator searches only ±1 bit period around each expected
transition (`fskutil.c`, where `spacing` is the oversampling factor, one bit), so a
single measurement can never report more than about ±10 ms at 100 Bd.

The consequence is the one that matters: **a responder can trim a static offset of up
to about one bit, slowly. It cannot chase a moving grid.** A master whose cycle is
wrong by more than a millisecond or two per cycle walks out of the responder's
reader, and the responder has no mechanism to follow.

This is the behaviour a capture shows from the outside: a far end that re-anchored
once and then held 1249.97 ms with 2.7 ms rms for 41 s while the transmissions it
was answering drifted ~16 ms/cycle past it.

## 2. When the master keys relative to the received CS: never directly

**There is no delay from the control signal to the master's transmission. The
master's transmission is one cycle constant after its own previous transmission,
full stop.** The received control signal moves only the master's *receive* window,
and only by sub-bit amounts.

What the control signal is used for, once, is to learn `d`:

  * during link setup the master opens a receive window at **its own transmission
    start + 960 ms**, i.e. the instant its own data ends, and **searches** that
    window for a control signal;
  * when it finds one, the receive anchor is set to the **measured start of that
    control signal**, refined by the sub-bit edge estimate, plus one cycle. The
    transmit anchor is advanced by one whole cycle and by nothing else;
  * from then on the offset `d` between the master's own transmission and its
    receive window is whatever the search found, and it is held.

So the reference point is the control signal's **START**; nothing in the protocol
is referenced to its end. It is consumed once, at acquisition, not every cycle.

Restated as the rule a master implements:

    tx_next   = tx_prev + 1250 ms                       (exactly, always)
    rx_window = tx_this + 960 ms + d                    (d learned at connect)
    d         = corrected only by sub-bit edge statistics thereafter

## 3. Acquisition during link setup

**The master creates the grid; it does not acquire one.** Setup and the established
link run the same 1.25 s raster from the first cycle.

Per cycle, the master:

  1. transmits the sync/call packet, which is **exactly 960 ms** and occupies the
     information-packet slot: 9 bytes at 100 Bd (header 0x55 followed by the
     8-byte destination call, 0x0F-padded) = 720 ms, immediately followed by
     6 bytes at 200 Bd carrying the **first six bytes of the same call** = 240 ms.
     The 1990 description's diagram agrees: `/Header/S L A V C A L L/SLAVCA/`,
     first part 100 Bd, tail 200 Bd;
  2. keys a lead-in of `txdelay` before the data bits and drops the carrier at the
     end of the data. hf-pactor's lead-in is a steady tone, not silence, and its
     default length is 30 ms.

     The lead-in is on the air and it costs a receiver a codeword if ignored.
     W6IDS's control-signal bursts measure 145 ms for a 120 ms word against
     WS8EOC's 115 ms, and the bit grid starts a consistent 19 ms into the run — a
     steady lead-in tone at the head of the burst, from a station that is not
     running hf-pactor. A receiver aligning on the burst's leading edge therefore
     reads two bits of lead-in and drops two bits of data; aligning on where the
     carrier DROPS separates them, because the unkey is at the end of the data and
     nothing precedes the lead-in. See the alignment section of
     `pactor1-control-signals.md`;
  3. **listens for the entire remainder of the cycle**: the window opens the
     instant its own data ends and closes the instant the next cycle's keying
     lead-in begins, i.e. `1250 - 960 - txdelay` = **260 ms** with the default;
  4. **inverts the shift polarity every cycle**, setup included, per `Mit jedem
     neuen Paket oder Kontrollsignal wird die Shiftlage invertiert`;
  5. repeats, with no gap and no doubled cycle, up to the configured retry count
     (hf-pactor's default is 30 cycles = 37.5 s; its `PACTOR_RETRY_CALL` and
     `PACTOR_RETRY_QSO` constants are referenced nowhere).

The search inside the window is an exhaustive slide of a 12-bit correlator at the
oversampling step — **1.25 ms resolution** — across the window, testing at each
position for an **exact, zero-bit-error** match against CS1 or CS4 in **either
shift polarity**. The polarity that matches establishes the polarity reference for
the rest of the link.

**Usable search span: 0 to 138.75 ms.** hf-pactor's loop runs 120 correlator
positions at the 1.25 ms step, so it offers starts from 0 to `119 x 1.25` =
148.75 ms; the last eight of those read past the end of the sample buffer, leaving
112 fully-sampled positions. A responder that turns around later than 138.75 ms past
the end of the master's own transmission is never heard. (The loop bound subtracts
11 bit-times where it needs 12; the eight unusable positions are that off-by-one
and are not to be reproduced.)

**Separately, and to within one correlator step of the same answer, 140 ms is the
collision limit.** The master keys its next lead-in at `cycle - txdelay`, so a
control signal beginning at `960 + d` has to be finished by then:

    d  <=  1250 - txdelay - 960 - 120  =  140 ms   (txdelay = 30)

and the buffer is exactly `(1250 - 960 - txdelay) / 10 * 8` = 208 samples long
(`PACTOR_CYCLE_SP` 1250000, `PACTOR_CYCLE_STREAM_FEC` 960000, `RXOVERSAMPLING` 8),
of which the last 96 are the correlator's own length. So the master searches
precisely as late as it can still hear a control signal out before its own carrier
comes up, and not one sample later. **An implementation derives the span from its
own settle rather than copying 140 ms**: with shrike's 40 ms the same arithmetic
gives 130 ms.

The 1990 description has the master transmit data immediately on the first valid
control signal:

> Sobald das erste gueltige CS empfangen und synchronisiert ist, wird das erste
> normale Datenpaket mit Head=AA (HEX) und Paketzaehler=1 ausgesendet.

**hf-pactor does not.** On a first hit it transmits **one more identical call
packet** and requires a **second control signal**, read at the predicted position
with no search at all, exact match only, against the *specific* codeword the first
hit gave (CS1 if CS1, CS4 if CS4), and with the shift polarity required to have
inverted. Only then does data start, in the very next cycle:

    cycle n     call packet
    cycle n+1   call packet   (verify cycle)
    cycle n+2   first data packet
    cycle n+3   ...

Every cycle 1.25 s apart, transmitting in all of them. The extra cycle is insurance
against a false positive out of a 139 ms exact-match search and is harmless against
a real responder, which keeps sending its control signal until answered. Either
behaviour interoperates; the single-CS rule is the normative one.

Two framing facts fall out of the same path:

  * **the master's first data packet carries packet counter 1**, which both sources
    agree on;
  * the 1990 description specifies header **0xAA** for it; hf-pactor sends header
    **0x55**. The header is not part of the CRC, and hf-pactor's receiver reduces it
    to a single bit used only to bucket memory-ARQ accumulations, so the
    disagreement is tolerated on both sides rather than resolved. Prefer the
    document's 0xAA.

## 4. A missed or undecodable control signal: hold the grid

**Neither station ever re-acquires timing after connect.** The cycle advance is
unconditional: one routine adds the cycle constant to both anchors and inverts both
polarities at the end of every cycle, before the receive is even attempted, and it
is called on every arm of every state.

On a failed receive the master re-sends **the same packet** in the next cycle and
decrements a retry counter. Retry exhaustion never resynchronises, and what it
does instead **splits by role, not by rate** — an earlier revision of this
paragraph had it splitting by rate, which is wrong in the direction that matters
to a sending station.

hf-pactor enters every **ISS** state at **both** rates with the user's retry
parameter — `params.pactor.retry = 30`, floored at 4 (`hfterm/src/params.c`) —
and on exhaustion **drops the link**. That includes `tx_200_cs1` and
`tx_200_cs2`, so the reference's sending station never gears *itself* down; the
only way out of 200 Bd for it is reading a CS4.

`PACTOR_RETRY_HISPEED` = 4 is entered by the **IRS** states at 200 Bd
(`rx_200_cs1`, `rx_200_cs2`, `rx_100_200_cs1/2`, `tx_rx_200`), and on exhaustion
they go to `rx_200_100` — where the receiver keys **CS4**, which the sender reads
as the REJECT and drops to 100 on. So the four-cycle speed fallback is the
**receiver's**, and it reaches the sender as a codeword rather than as a count.

The 1990 description agrees on ownership: CS4 is the IRS's word, the speed-up
branch's own give-up after a preselected number of cycles is likewise the
receiving station's, and M.1798 §4 gives the sending station no gear command at
all. **A sending station that cannot hear the CS4 has, in the reference, no
protocol mechanism to leave 200 Bd** — see `shrike.ptc.P1_HISPEED_RETRIES` for
what this station does with the same figure from the other end.

One detail to implement deliberately: **the timing-recovery estimator is fed
on every receive attempt, including failed ones.** It runs before the codeword
comparison in the control-signal path and before the CRC check in every packet
path. The loop is driven by raw edge statistics, not by successful decodes, so it
keeps trimming through a fade instead of freezing.

The only path that re-derives a grid after connect is the disconnect courtesy: on
hearing a QRT packet addressed to itself, a station extrapolates the old grid by
whole cycles to the packet's arrival time, flips the shift polarity if the
whole-cycle count is odd, and answers with the last acknowledging codeword — but
gives up if more than **30 s** of extrapolation would be needed. That bounds how
stale a PACTOR-1 grid is ever considered usable.

## 5. Tolerances

**Codeword tolerance, link setup.** Zero bit errors, both polarities, CS1 or CS4
only. The verify cycle is likewise zero errors, one specific codeword, and a
polarity that must have alternated.

**Codeword tolerance, established link.** Zero or **one** bit error against each of
the four codewords in turn, polarity fixed by the link. The minimum distance
between codewords is 8, so one-error correction is unambiguous. A whole-word
inversion (distance 12) is detected and **reported as an error**, not accepted as
an alternative decode — no state-machine arm acts on it. A one-error CS1 or CS2 *is*
acted on, in the four data-transmitting states only; a one-error CS3 or CS4 is
never acted on, because a mis-taken changeover or speed change is unrecoverable
where a mis-taken acknowledgement is not.

**Timing tolerance, established link.** **This is one implementation's and not the
description's, and the difference matters.** hf-pactor reads the control signal as
exactly 12 bits starting at the predicted instant, with no scan (`receive_cs`,
`hfkernel/fsk/pactor.c:745`). The 1990 description says nothing about how the
established-link read is performed; what it does say is that the receive window is
0.29 s wide for a 0.12 s signal, and it has the master *search* it during setup
(§3). Treat the point read as what one station is known to do, not as a
requirement, and note that no recording we hold has ever placed a codeword outside
a reader's anchor, so the narrow reading has never been tested against a peer.

The tolerance a point read leaves is the matched filter's: at 100 Bd a bit is 10 ms,
decisions are taken at the bit centres, and error grows from about a quarter-bit.
**Treat ±2.5 ms as clean and ±5 ms as the edge; ±10 ms is a whole bit slip and is
unrecoverable**, because the correction loop's pull-in is at most one bit per
measurement and it applies about an eighth of that per cycle. Where the peer's
codeword actually lands is a separate, measured question: over 755 tracked cycles
of this station's own sessions it sat further than half a bit from the predicted
instant in 23% of them and never further than 21.4 ms, which is why shrike's own
reader searches ±15 ms rather than reading at a point
(`hfmodem.shrike.p1rx.CS_SEARCH_HALF_S`).

**Timing tolerance, acquisition.** Turnaround from 0 to 138.75 ms, at 1.25 ms
resolution, as in §3 — one correlator step short of the 140 ms collision limit,
which is as late as a control signal can be heard out before the master keys. That
is the whole of the master's flexibility about `d`, and it is spent once.

**Nothing anywhere drops a link on a timing criterion.** There is no "CS too far
from expected" test. A control signal outside the reader is simply not decoded, and
it costs a retry like any other failure.

## 6. Setup regime versus established regime

**There is no separate setup rate. Both are 1.25 s, every cycle, transmitting in
every cycle.** The differences between the two regimes are exactly three, and none
of them is a period:

  1. what is transmitted in the 960 ms slot — a call packet (720 ms @100 Bd
     + 240 ms @200 Bd) rather than an information packet;
  2. the control signal is found by a **search** over the receive window rather
     than read at a fixed offset — the description states the search only for
     setup, and states no method at all for the established link (§5);
  3. hf-pactor's master requires the acknowledgement **twice** before it starts
     data (implementation only — see §3).

The receive window differs only because it is being searched: 260 ms wide during
setup, versus hf-pactor's 12-bit read at a point in the established link.

## 7. What the 1990 description leaves open

**The turnaround gap.** The description gives 170 ms of residual and does not split
it, and no split is forced: the 84-bit changeover constant maps `d` to
`170 - d + 2p` for any `d` (§0). A station measures its peer's turnaround.

**Which end owns the clock, and what that means for frequency too.** The responder
follows the master in *both* time and frequency — its transmit frequency offset
tracks the master's. A master that drifts in frequency drags the responder with it; a
master that expects the responder to come to a nominal frequency is wrong.

**Speed after a CS1 connect answer.** A CS1 answer means the responder decoded the
200 Bd redundancy section, and **the link starts at 200 Bd**: the master's first
data packet in that case is 24 bytes at 200 Bd. A CS4 answer starts the link at
100 Bd with a 12-byte packet. Both packets are 960 ms; only the bit rate differs.
`pactor1-control-signals.md` §5 states the same rate selection.

The corollary: **a master that answers CS1 with a 100 Bd
packet is transmitting into a receiver listening at 200 Bd, and the link is deaf
with nothing on the air to break it.** hf-pactor's own responder never provokes
this, because its 200 Bd check is compiled out and it always answers CS4 — which
is also evidence that CS4/100 Bd is the well-travelled interoperable path.

**The changeover packet's placement** is not open, and was listed here as if it
were. CS3 is the head of the new sender's first packet and that packet occupies
the **control-signal slot**, not the packet slot, so the grid rotates by `960 + d`
at every direction change and both stations apply the 840 ms adjustment in the
same cycle. The description states that rotation outright, under
*Senderichtungswechsel*: *"Die Antwort-CS auf ein BK-Paket beginnen im Zeitraster
eine CS-Laenge (0.12 sec) vor Ende des alten eigenen TX-Blocks."* Against
`Paketdauer : 0.96 sec` that is the 840 ms, from the protocol's authors (§0).

**CS3 is always at an effective 100 Bd.** In a 200 Bd packet the 12 control bits
are transmitted **bit-doubled** into 3 bytes so the symbol duration is unchanged;
the receiver correspondingly skips 120 ms at 200 Bd and 160 ms at 100 Bd before the
data. The description's `4 Bit unbenutzt` belongs to the 100 Bd case, where 12 bits
sit in 2 bytes; at 200 Bd 12 doubled bits fill 3 bytes exactly.

**Packet counters at connect.** Master starts its transmit counter at 1, responder
starts its expected-receive counter at 0, so the master's first packet reads as new.

**Retry exhaustion at 200 Bd is a speed fallback, not a disconnect** (§4).

**hf-pactor's recovery from a failed verify cycle is not protocol and is not to be
reproduced.** On a failed verify the scan resumes inside a stale sample buffer while
both anchors have already been advanced; a subsequent hit computes a meaningless
receive anchor, and the grid then advances by two cycles instead of one. The
behaviour is ill-defined.

## 8. Caller conformance

Requirements on a station in the caller/MASTER role, against §§1-7, with shrike's
current state noted.

**1. One transmission every 1.25 s during setup, from the first cycle.** There is
no slow setup regime. The call packet goes out in every cycle, with the shift
polarity inverted between them, and the master listens in the 260 ms gap. A skipped
cycle presents the responder with an empty packet slot; its reader is at a fixed
offset and does not look elsewhere for it. The document, hf-pactor and the on-air
captures in `pactor1-control-signals.md` all agree.

**2. The transmit instant is not derived from the peer.** Hold a free-running
1.25 s grid, place the receive window at `own_tx_start + 960 ms + d`, learn `d`
once at connect by searching that window, and correct only `d` — never the transmit
anchor — from sub-bit edge statistics. Closing the transmit loop around the peer's
bursts imports the peer's jitter into the local grid and feeds it back, a positive
feedback path the protocol deliberately does not have. Referencing the peer is how
a station acquires; holding its own clock is how a master runs. (For a
*responder* the opposite is correct: it locks to the master's bursts.)

Referencing the next packet's data bits to the control signal's start, the correct
offset is `1250 - 960 - d = 290 - d`:

    d =  40 ms  (hf-pactor's shipped responder)  ->  250 ms
    d = 181 ms  (WS8EOC, measured)               ->  109 ms
    185 ms                                       ->  implies d = 105 ms

A fixed 185 ms is therefore 76 ms — seven whole bits at 100 Bd — from the correct
instant against a peer at `d = 181 ms`, into a receiver that reads at a fixed offset
with a correction loop whose pull-in is one bit at an eighth gain per cycle. That
alone fails every packet, indefinitely, while the peer's control signals continue to
sound perfectly healthy. Any fixed number here is wrong against some peer;
`TX_OFFSET_S = 0.205` and `D_NOMINAL_S = 0.085` in `onair.py` are not protocol
values.

Both halves are implemented in `shrike/onair.py` (`_MasterGrid`): one slot always,
an anchor advanced by exactly one cycle and moved by nothing once a control signal
has been heard, `d` searched over 0-130 ms and thereafter corrected at gain 1/8 from
`p1rx.cs_time_dev`. `tests/shrike/test_burstlock.py` fails if the transmit anchor ever
moves during a transmitting cycle.

**Two divergences from hf-pactor are deliberate, are marked as such in the code, and
are not what PACTOR specifies**: `d` is released back to searching after three
silent cycles, where §4 says a station never re-acquires; and the control signal is
captured over a 20 ms range around the predicted instant, where the protocol reads
at a point. Both are stopgaps that follow from acquiring on an envelope detector
rather than on a zero-error 12-bit correlation, and both should go when a
correlator-based acquisition exists.

**3. The connect answer selects the first data packet's bit rate.** CS1 → 24 bytes
at 200 Bd; CS4 → 12 bytes at 100 Bd (§7). Sending 100 Bd after either answer is a
silent failure independent of any timing fault. `shrike/ptc.py` selects the rate
from the answer, and the 200 Bd path decodes end to end against an independent
monitor (`pactor1-data-packets.md` §6): 200 Bd packets read at speed level 2, a whole
200 Bd session follows, and a CS4 fallback back down to 100 Bd follows without loss
of sync. What is untested is on the air: no 200 Bd link has been attempted with a
real station, and no off-air 200 Bd data packet exists in the corpus.

**4. The receive window opens at the end of your own data and stays open across the
whole 290 ms.** Nothing blanks the receiver at the end of a transmission; there is
no T/R mute in the protocol and none in `onair.py`, where `TR_SWITCH_S` appears
only in cycle-budget arithmetic and in the printed banner. The window has to span
the full residual because the peer's position inside it is unknown until measured.

**5. First data packet header.** The document says 0xAA with counter 1; hf-pactor
sends 0x55 with counter 1, and hf-pactor's receiver tolerates both. Prefer 0xAA.

**Confirmed against both sources and requiring no change:** the 1.25 s cycle
constant, the per-cycle shift inversion, and the LSB-first codeword order.

## 9. Clock ownership on an established link, and what a capture can show

**The master is the clock.** The timing correction is applied to the receive anchor
always and to the transmit anchor only when the station is not the master; the same
rule governs the frequency correction. The master's transmit instant is moved by
nothing the far end does; the responder's is dragged onto the master's.

The responder acquires that clock **once**, in the call-packet detector: its receive
anchor becomes the call frame's own start refined to sub-bit by the edge statistic,
and its transmit anchor that plus a constant 960 ms plus its keying delay plus
10 ms. Thereafter the per-cycle epilogue advances both by exactly one cycle,
forever. It never re-acquires from a later call packet — once it is in the ARQ loop,
a call frame is just a packet that fails CRC.

Two consequences bear on what a recording can be read to mean:

  * **A free-running raster is what a locked responder looks like.** It locked
    once, to a call, and has free-run since; a capture taken after that shows one
    clean
    period no matter whose clock it is. Three captures of one gateway fit a single
    period of 1249.69-1250.32 ms with 4.7-6.7 ms residual, which is consistent with
    either station owning the grid.
  * **The 960 ms is a constant, not the received frame's length**, so shortening a
    call frame by 260 ms (dropping the 200 Bd redundancy tail) cannot move the
    reply; the observed movement was 11 ms. A frame-start-locked responder and a
    free-running one make the same prediction here. Separating them requires the
    call phase to be walked, not held at a whole number of the peer's cycles.

**The measurement that does settle it is `d` on an established link.** The responder's
control signal is due at its own cycle start + 960 ms + txdelay + 10 ms. If the
caller's packet slot were displaced from the responder's cycle start by Δ, the caller
would read `d = txdelay + 10 - Δ`; a peer free-running independently would put its
burst at an arbitrary point in the caller's cycle, so `d` would be anywhere in
0-1250 ms and would not repeat. A repeatable **d = 16-19 ms** across a session is
consistent with a keying delay of a few milliseconds and Δ ≈ 0 — the two grids
coincide. On a two-slot cadence the burst heard is the one from the cycle the caller
did not transmit in, and the modulo-cycle gap folds it to the same figure:
`(1250 + 977 - 960) mod 1250 = 17 ms`.

## 10. Listen-window budget

The window a caller must be able to save and decode is `boundary - settle - pos`,
where `pos` is the first capture block after the carrier drops. A guard that
requires a whole control signal to fit must be measured on the audio that is
actually decoded, not on the window one call returns: on a one-slot cadence
`1250 - 960 - settle 100 - block 85 = 105 ms` against a 120 ms control signal,
while the concatenated audio that reaches the frame scan is 190 ms and clears.

**Capture window length is a direct read of the transmit cadence**: 105-190 ms on a
one-slot cadence, 1.27-1.44 s on a two-slot cadence. Check it on any session
where the cadence is in doubt.
