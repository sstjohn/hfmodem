# How a PACTOR station learns what the other end can demodulate

Every PACTOR link opens in FSK PACTOR-1 and may end up in PACTOR-2, PACTOR-3 or
PACTOR-4. Something has to stop the faster station keying a waveform the slower
one cannot read. This document records what that something is, and the shape of
the record is the finding: **the only complete description of the mechanism was
published in 1990, and every specification since states the property and
withholds the mechanism.**

Evidence tags are `EVIDENCE.md`. Companions: `pactor-connect-frames.md` for the
connect itself, `pactor1-data-packets.md` for the status byte, and
`pactor1-control-signals.md` for the codewords.

---

## 1. The 1990 mechanism, in full  [S]

Helfert DL6MAA and Strate DF4KV, *PACTOR — Einführende Protokollbeschreibung
(Level 1)*, 5 November 1990, section *Levelfestlegung beim Verbindungsaufbau*.
The copy read here is the one distributed inside Sailer HB9JNX's GPL `hf`
package.

- At connect the far end's maximum level is unknown, so **every link starts at
  level 1**.
- **The master declares, upward, in payload text.** The first characters
  transmitted are the maximum available software level number, then the master's
  callsign, then CR — the document's own example is `1DF4KV<CR>`. This is user
  data on the PACTOR-1 data channel, not a protocol field.
- **The slave refuses, downward, and only if it was over-asked.** A slave whose
  own system level is lower than the declared one says so immediately, by a CS3
  break-in plus supervisor information.
- **Absent a refusal the link continues at the declared master level.** Stated
  outright.
- The supervisor channel is defined in the same document: SB = ASCII FS =
  `0x1C`, a one-character function code, parameters, terminated by SPACE.
  Function code `A` is "level number follows". It travels as ordinary text,
  deliberately, so that a speed change cannot break it.

Three properties follow, and they are why the arrangement works:

1. **It is settled before the first non-FSK symbol.** The declaration is in the
   master's first data packet and the refusal window closes with the CR.
2. **It cannot deadlock on an undecodable waveform**, because the whole exchange
   is carried in Level 1, which every PACTOR station has by definition. That is
   the same reason PACTOR-2 and PACTOR-3 still open in FSK.
3. **It expresses one scalar ceiling per link**, declared once, by the caller.
   There is no per-mode capability list and no room for one.

## 2. Declaration, not probe  [S]

The 1990 arrangement is a **declaration with a binding default and an explicit
refusal channel**. The higher-capability station is stopped from keying a mode
the other cannot read because it never had permission: absent a declaration the
link stays at Level 1, and a declaration that over-asks draws an immediate
break-in.

PACTOR does contain a genuine probe-with-fallback, and it is a different shape
at a different layer — the 100→200 Bd speed-up of `pactor1-control-signals.md`.
There the receiving station commands CS4 after a correctly received packet and
switches its own receiver to 200 Bd; if no clean 200 Bd packet arrives within a
preselected number of cycles it answers with a **fresh** OK codeword, different
from the last one before the CS4, and steps back down.

Two rules generalise from it and are worth stating on their own:

- **The idiom for "my offer went unanswered" is a changed codeword, never a
  repeat.** A repeated codeword means REQUEST and is ignored.
- **Escalation is pull.** CS4 and CS6 belong to the receiving station at every
  level, so a level is capped by the receiving end never asking past it.

Merging the two layers over-generalises. The speed ladder is pull; the
protocol-level ceiling is a push declaration by the caller.

## 3. What every specification since says, which is nothing  [S]

*The PACTOR-2 Protocol* (SCS, 1996): "fully backwards compatible with the
current PACTOR ('PACTOR-1') standard, as the initial link setup is still done in
FSK. If both stations are capable of Level II, an automatic switching is
performed."

*The PACTOR-III Protocol* (SCS, 2004), reproduced verbatim in **ITU-R M.1798-0,
-1 and -2**: "the initial link setup is still performed using the FSK (PACTOR-I)
protocol, in order to achieve compatibility to the previous systems. If both
stations are capable of PACTOR-III, automatic switching to this highest protocol
level is performed."

That sentence, across three editions of an ITU Recommendation and two vendor
papers, is the entire published treatment. *The PACTOR-4 Protocol* has no
link-establishment section at all.

The PTC-IIIusb manual's `MYLevel` parameter takes 1, 2 or 3; the modem "will
switch to PACTOR-2/-3 when the other station is so fitted", and the default 3
"leads to a very reliable and automatic level choice procedure during the link
initialization". It is a **ceiling** setting, not a mode selector. So the vendor
confirms the *shape* the 1990 document specifies — one scalar ceiling per
station, resolved at link initialisation, invisible to the operator — and
declines to say what carries it above Level 1.

## 4. The published field is on the air and is frozen at "1"  [P: 2 stations]

Both third-party stations on record here that went on to PACTOR-3 announce the
digit **1**: `1dl6maa\r` and `1w4dna\r`, each in a session that reaches Level 3.
The field is present, well-formed and not carrying the capability at this era's
stations.

This station has never emitted a level digit at all, and it has reached PACTOR-3
from a link that declared nothing. Under the 1990 rule an undeclared link stays
at Level 1, so **the declaration is not what gates the level at these
stations** — whatever it may still do where it is present, it is not required.

## 5. What correlates instead, and what that is worth  [T: 2 gateways]

The PACTOR-1 data packet status byte, as the 1990 document assigns it:

    bit 0-1  packet counter          bit 2-3  data mode: 00 ASCII, 01 Huffman,
    bit 4    not yet assigned                            10 and 11 unassigned
    bit 5    not yet assigned        bit 6    BK request     bit 7  QRT

Bits 4 and 5 are the only reserved space in the byte, and PACTOR-2 and PACTOR-3
later spend bit 4 as the top of a three-bit data type and bit 5 as the long-cycle
suggestion (`pactor3.md` §14).

Crossing the two bits on this station's own announcement packet, at two Winlink
gateways, gives a complete 2×2:

| announced | as a PACTOR-2/-3 data type | bit 5 | what came back |
|---|---|---|---|
| `0x01` | 0, ASCII | 0 | ordinary acknowledgement; counter advances; stays PACTOR-1 |
| `0x11` | 4 | 0 | `0x6A9`, fifteen consecutive cycles at zero errors; counter frozen |
| `0x21` | 0, ASCII | 1 | nothing at all; counter frozen |
| `0x31` | 4 | 1 | `0x59A`; at the DL6MAA reference station, PACTOR-3 one turnaround later |

Two things can be said from this without straining.

**"Bit 4 is only the data type" cannot account for it.** `0x11` and `0x31`
declare the same data type and differ only in bit 5, yet draw different
codewords. `0x01` and `0x21` also differ only in bit 5, and one is acknowledged
while the other draws silence. Neither bit is a passenger.

**What the three announcements share is better described negatively.** Under the
1990 status byte, `0x11`, `0x21` and `0x31` are all *malformed*: bit 4 set means
data mode 10 or 11, unassigned in PACTOR-1, and bit 5 set is an unassigned bit
requesting a long cycle PACTOR-1 does not have. Only `0x01` is a legal PACTOR-1
packet, and only `0x01` draws an ordinary PACTOR-1 answer. So the announcement
may not *state* a level at all — it may simply fail to be a PACTOR-1 packet in a
specific, recognisable way, and the gateway may be answering the failure.

The reading that the bits are a two-bit capability extension — `00` Level 1,
`01` Level 2, `11` Level 3, with `0x6A9` and `0x59A` the two commands — accounts
for all four cells with one rule, including the silent one. **It is not known to
be true.** No document assigns `0x6A9` or `0x59A` any meaning, and the 1990
control-signal table has exactly four words (CS1 `0x4D5`, CS2 `0xAB2`, CS3
`0x34B`, CS4 `0xD2C`), neither of these among them. Both gateways are Winlink
infrastructure, so two stations is not two implementations.

There is a specific, cheap thing that would settle it, and it has not been done:
`0x6A9` has been drawn fifteen times and answered zero times.

## 6. A station that declares nothing is never over-asked  [S]

Sailer HB9JNX's hf-pactor (`hfkernel/fsk/pactor.c`, GPL, PACTOR-1 only) is the
one independent implementation available to read. It `#define`s the supervisor
byte and the level and callsign function codes **and never references them
again**. It neither sends nor parses a level digit and never sets status bits 4
or 5.

Its posture is its ceiling. It declares nothing, so nothing is commanded of it,
so it is never over-asked — and it interoperated with commercial controllers on
that basis. That is the cleanest confirmation available that the mechanism is
opt-in declaration rather than probe: a station with no capability machinery at
all is handled correctly, by accident.

The same reasoning places PACTOR-2. A PACTOR-2-only modem is `MYLevel 2` and
opens every link in FSK PACTOR-1. If a PACTOR-3 caller declares 3, the 1990
mechanism has the answerer break in at once and say "2"; if the PACTOR-2 box is
the caller it declares 2 and the PACTOR-3 station never asks past Level 2.
Either direction, the link settles at the lower of the two ceilings and no
waveform is keyed speculatively.

## 7. A gateway's level policy is a channel setting, not a software version

Where a Winlink gateway refuses a level, it refuses in application text after
the link is already up, and the text names the **channel**:

    This RMS channel (WS8EOC) is set to only accept Pactor levels above 2 - Disconnecting

Two channels on record refuse in those words. Another station running the same
gateway software serves PACTOR-1 mail in full, end to end — four messages
crossed at KB5LZK on 2026-09-11 over a link that never upgraded. **So the
refusal is a per-channel operator setting and not a property of a software
version**, and any account that attributes it to a release number is wrong about
where the decision is made.

It also sits at the wrong layer to be capability negotiation. It arrives in B2F
text, after connect, after the level is already established. Whatever it
inspects, it inspects the level the link is running at — not a capability
anybody declared.

## 8. What this implementation does

Neither half of the published mechanism. This station emits no level digit and
parses none; it sends no supervisor refusal and would not recognise one. On
receive it infers nothing about a peer's capability from the peer's status byte:
bit 4 on a PACTOR-1 packet is masked off and bit 5 is read only as the long-cycle
flag, so `0x6A9` is decoded, named, and drives nothing (`shrike/ptc.py`,
`shrike/arq.py`).

The consequence is worth stating plainly, because it is the reverse of the
usual: **the channel this station transmits on is a channel it does not receive
on.** If the status bits do carry a declaration, every peer's declaration is
being discarded here — including a PACTOR-2-only peer's, which is the one case
where discarding it has a cost.

## 9. Not settled here

- Whether `0x6A9` and `0x59A` mean levels, or mean "your status byte is not a
  PACTOR-1 packet", or mean something about the gateway's own configured
  ceiling. All three fit the 2×2.
- Whether a repeated `0x59A` is a permission held open or a demand. The 1990
  convention that a repeated codeword is a REQUEST argues for the second, with a
  word the document does not define.
- What a stock SCS modem puts in its announcement status byte in general. Two
  samples exist. If every stock station announces the same bits regardless of
  the level its session reaches, the field is not a graded ceiling.
- Whether the published digit is read anywhere at all in current equipment. No
  station on record has been sent one from here.
- What a PACTOR-2-only station actually does when called. Both ends' ceilings
  would be known, which is the one arrangement that could demonstrate the
  mechanism as the vendor describes it; it needs a cooperating operator with an
  SCS controller and has not been arranged.
