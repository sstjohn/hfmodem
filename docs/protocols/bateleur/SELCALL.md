<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# HF selcall (CCIR 493-4 land mobile, the UN open 6-digit selcall)

`hfmodem.bateleur.selcall` sends and receives the selective calling that HF
land-mobile radios (Codan, Barrett, Q-MAC, Vertex and others) carry in the
voice channel: a 4-digit selcall on the CCIR 493-4 pattern, and the UN's open
6-digit selcall. HFLINK's HFS and HFR amateur nets use it.

## Sources

- ITU-R M.493 (the maritime DSC recommendation the land-mobile system grew
  from): the ten-bit character code, phasing recognition, DX/RX time diversity
  and service symbols. The transmitted phasing layout here is the
  land-mobile variant described below.
- The UN open 6-digit selcall specification, as HFLINK publishes it: the
  layout of each kind of call, and a capture of a real call printed with it.
- HFLINK's description of the 4-digit layout, and three recordings it
  publishes (two of a Vertex VX1210, one of unknown origin). They were used
  as a black-box partner to check this implementation against, and are not
  in the tree.
- The Barrett interoperability note HFLINK publishes: which features are
  common to 4-digit radios, and that 4 and 6 digits are told apart
  automatically.


## The waveform

| Part | Where |
|---|---|
| FSK, 1700 Hz for 0 and 1870 Hz for 1, 100 bits a second, phase continuous | `fsk.modulate` |
| Characters of ten bits: seven of information, least significant first, then the count of 0s among them, most significant first | `code.char_bits` |
| A dot pattern (2 s for a radio sitting on the channel, 6 s or more for scanning ones), then phasing: six pairs of 125 and 109 down to 104 | `fsk.modulate`, `code.air_chars` |
| Each character in the DX position, and again four characters later in the RX position | `code.air_chars` |
| Format 120, the called address in digit pairs (2 for four digits, 3 for six), category 100, the caller's address, the call's content, the end character three times (117: acknowledgment asked) | `code.encode` |

**Which tone is which bit.** M.493 gives the maritime service's higher tone
to 0. The land-mobile diagrams label 1870 Hz as 1, and all three recordings
agree: their known characters' 0 bits are at 1700 Hz. `invert` swaps them,
and the receiver tries both, so an inverted sideband somewhere on the path
costs nothing.

**The 6-digit format character.** The UN diagrams draw 126 in the format
position of a 6-digit call. The capture printed with them sends 120, as the
4-digit calls do. We send 120 and accept either, and tell 4 digits from 6 by
where the category character falls.

## Calls

| Kind | Content after the caller's address | Host command |
|---|---|---|
| Selcall | nothing | `CALL <address>` |
| All, group, sub-group | a selcall to X000, XX00, XXX0 (XXX000 ... in six digits); all zeros is every station | `CALL 5600` |
| Beacon | format 125 in place of 120 | `BEACON <address>` |
| Hang-up | 104 105 | `HANGUP <address>` |
| Telcall | 102, the number of digits, 8 characters of digits, a parity character | `TELCALL <address> <number>` |
| Pagecall | 116 and up to 32 characters of text | `PAGE <address> <text>` |
| GPS position | 121, a status byte, the position in 8 characters, a parity character | `GPS <address> <lat> <lon>` |
| GPS request | 125 125 | `GPSREQ <address>` |
| Status request | 103 103 | `STATUSREQ <address>` |
| Emergency | 119, ending 119; with a position, 122 and the GPS content | `EMERGENCY [address] [lat lon]` |
| Acknowledgment | nothing, ending 122 | sent automatically if `acknowledge` is on |

A position is latitude and longitude as degrees (three digits), minutes and
thousandths of a minute, paired into characters; the status byte's low bits
give the hemispheres. The capture's parity character is the exclusive-or of
the status byte and the eight position characters (the legend says only the
eight); we send the capture's and accept either.

## Receive layouts

Transmit uses six DX/RX pairs of 125 against 109 through 104, followed by
one format specifier. This is the land-mobile layout; it differs from the
M.493 layout of six 125s against 111 through 106, then 105 and 104 under two
format specifiers. The receiver now finds the first format specifier and
accepts either layout, including the repeated format specifier.

The M.493 path is pinned by a synthetic symbol sequence and waveform. It
is **unverified against a foreign implementation**. A foreign recording
with the full phasing sequence, repeated format specifier, known caller
and destination, and end characters would settle it. The three cited
land-mobile recordings do not establish maritime M.493 interoperability.

The receiver caches tone windows across audio chunks and tries both tone
polarities. It emits calls to its observer without keeping an unread
history list. This implementation remains in `bateleur.selcall`; moving
its FSK code into `lanner` is a separate change.

## Measured

The earlier call-success table retained no measurement command, seeds or
raw output. Its fading results used the old per-burst-normalized channel;
those figures are withdrawn. The corrected-channel PHY harness does not
measure selcall call-success rates.

A minute of noise alone makes no call. Calls survive a clock 2% out and 40 Hz
of offset. All three published recordings decode: 3000 (all stations
beginning 3) called by 3332, 3332 calling itself, and 1225 called by 1221,
the last 40 Hz off with its clock 1% fast.

## On a station

    [protocols.selcall]
    enabled     = true
    host        = "selcall"
    port        = 8340
    addresses   = ["1234", "123456"]   # one of four digits and/or one of six
    dots_s      = 6.0
    acknowledge = false
    emergency   = false
    invert      = false

Host connections use the [normal host opening](../../implementation/bateleur/HOST-API.md#host-connections).
`EMERGENCY` is refused unless `emergency = true` is configured.

The lane listens on whatever channel the radio is on and never moves it. A
network that scans for selcall uses the radio's own scanning or the ALE
lane's. Every call heard is reported to the host port as a line:

    CALL PAGE FROM 123456 TO 567890 MINE TEXT MEET AT THE BRIDGE
    CALL GPS FROM 1234 TO 5678 POSITION -32.09520 115.80060

`MINE` marks a call to one of the station's addresses or a group it falls
in. `PARITY-BAD` marks a telcall or GPS call whose parity disagreed, and
`INVERTED` one read with the tones swapped.

## Not verified, and not implemented

- **Not heard against a radio.** Nothing here has been heard by a radio, nor
  has any radio's answer been heard by this implementation. The recordings
  are all a radio calling.
- **The revertive.** Codan's radios answer a call with a "revertive". Its
  form is not in any document we could read, so `acknowledge` sends an
  M.493-style acknowledgment call (ending 122) and is off by default.
- **4-digit calls other than selcall.** They are built by analogy with the
  6-digit layouts, with two-character addresses. The Barrett note lists
  4-digit pagecall, GPS and emergency calls as manufacturers' options, not
  common ground.
- **Codan's proprietary 6-digit protocol.** Not implemented, and not
  documented.
- **Transceiver lock ("stun").** A PIN-protected call that disables a remote
  radio. Deliberately left out.
- **The telcall's parity character.** Taken over the digit count and the
  digits, by analogy with the GPS call's. Not confirmed.
