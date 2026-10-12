# PACTOR-4 packet framing

The net packet is the source-coded data field, one status byte, then little-endian
CRC-16/X-25. Channel coding operates on the **whitened complete packet**; the
channel-code flush/tail is outside that byte transform. `shrike/p4frame.py`
implements this boundary without interpreting headers or advancing ARQ.

## Sources and coverage

- [SCS PACTOR-4 protocol](https://www.scs-pactor.com/assets/downloads/PACTOR-4%20Protocol.pdf),
  sections 5 and 11: field order, status layout, chirp dimensions and CRC.
- Independently traced whitening: state resets to `0x00B770CD` per packet,
  eight right-shift Galois steps per byte, feedback `0x48000000`, low-byte
  mask before advancing.
- Normal-short and normal-long source review: 43- and 206-byte fields, four
  CRC-valid frames from one recording.
- Four retained 25-byte Dragon chirp fields in `tests/shrike/test_p4chirp.py`,
  from `Pactor_IV_chirps.ogg`: the fixed normal-mode transform also yields
  standard X-25 CRCs on all four. This was checked September 20, 2026, without
  changing the seed, packet boundary, bit order or CRC convention.

## Chirp correction

The old chirp renderer appended `X25(data + status) XOR 0x53E1` while leaving
data and status unwhitened. The constant is exactly the affine CRC effect of
the 25-byte whitening mask:

```
cd70b7506acf508d436b7dfe1563d9a0bb27616aa6819653c9
```

For any 23-byte input, append ordinary X-25 and whiten all 25 bytes: checking
the still-whitened bytes gives the old `0x53E1` discrepancy. Thus the discrepancy
does not require an alternative CRC initialization. The old renderer created
a codeword that could pass the receiver's CRC after dewhitening, but its data
and status would differ from the intended values. CRC agreement alone could
not detect that transmitter error.

The four recorded status bytes `f5 f6 f7 f5` were still whitened. They become
`63 60 61 63`: transparent source mode, SPUG protocol status, counters 3/0/1/3.
The corrected renderer appends ordinary X-25 and whitens the entire net field.
The historical constant remains available for analysis, never for encoding.
Tests independently check the uncomplemented X-25 residue `0xf0b8`, exact
re-encoding of all four recorded fields and single-bit error rejection.

This establishes a consistent chirp framing rule against the retained recording,
not remote acceptance of our entry waveform or a successful P4 mail session.
Other stations and successful setup exchanges remain necessary interoperability
evidence. Existing empty-field fill remains an explicitly inherited assumption.

## Status handling

The layout is `E / S1 S0 / M2 M1 M0 / C1 C0`. Bits 5–6 are one enumeration:
0 normal, 1 CLOSE (0x20), 2 CHANGEOVER (0x40), 3 speed-up (unobserved).
The close and changeover names follow CRC-valid N9VC session bodies (0926);
these are an enumeration, not P3 independent control bits.
`Status` represents the enumeration explicitly. An extended-status bit is
reported but does not authorize stripping an unknown prefix. A CRC-valid
`Frame.data` is a source-coded field, not automatically mail or application data.
The WS8EOC 09-28 recording supplies one bounded exception: E=1, source mode 0,
protocol 0, and the exact prefix `01 01 01`. The host removes those three
bytes while preserving the raw frame and logging the prefix. Other extensions
remain refused; their field meanings and general grammar remain unknown.

Header-to-control mapping, entry negotiation, robust short controls, connected
ARQ and full transmit/receive coverage remain separate work for `--p4-mail`.

## Located normal-short reception

`p4short.decode_body` joins training-based channel estimation, finite equalization,
the measured component/symbol permutations, turbo decoding and this framing
boundary for normal-short BPSK (shared by SL5 and SL6). It requires 28800-Hz
baseband, a supplied first-training impulse coordinate and frequency offset.
It returns a frame only if exactly one of the two polarity interpretations
passes CRC; it does not classify headers or deliver source-coded data as mail.

The two Sigidwiki audio2 regression crops reproduce the independently established
43-byte packet hashes and CRCs `6bd1` and `2530`. The receiver erases the last
eight bits of each parity field. This evidence is limited to two packets from
one station and one recording.

## Normal-short transmission

`p4shorttx.short_packet` renders the same variant: both recorded frames
re-encode from their own decoded bytes and read back through
`p4short.decode_body` byte-exact, so the systematic tail both frames carry --
five zero bits and three that terminate the first component, from zero initial
states -- is a transmit rule and no longer an open dimension. The header is
cloned chip for chip from the first recorded frame; the two frames send the same
header waveform, so the clone is not one frame's accident. Both recorded frames
also send figure 6.2's seventh training after the sixth data block and stop
there, so the renderer sends it too and the emission is 792.8 ms.

Three items stay open and the module records them. The fill behind a short
payload is PACTOR-3's measured convention, not a P4 source-framing rule. The
header's semantics are unread, so what a cloned header asserts to a Dragon is
unknown, and rendering it is not evidence of acceptance. The receiver's window
guard erases the last fifteen body symbols, so fifteen of the 1056 coded bits --
and the sixteen erased parity-tail positions -- are rendered but never read back.

## Normal-long reception and transmission

`p4long` and `p4longtx` are the same pair for the 3.75 s cycle -- the layout
packet mail rides on. The field is 206 bytes (203 payload, status, CRC), the
systematic field 1656 bits, the body 24 data blocks of 207 symbols behind 24
trainings with a 25th to close, and the emission 3.286 s.

The two Sigidwiki audio2 long regression crops, 3.7500 s apart, reproduce the
independently established CRCs `5a2c` and `1bf8` and status bytes `63` and `60`.
Re-encoding each from its own decoded bytes puts C1 and C2 129 and 147 bits
(the second frame 144 and 139) from that recording's raw channel decisions,
where the systematic field the same decisions carry is already 104 and 147 out;
every wrong reading of the chain sits at chance, 790 to 919 of 1656. Both then
read back byte-exact through `p4long.decode_body`, as does our own payload at
nine transmit timing offsets.

The eight bits the dimension table charges over the field are spent exactly as
the short packet's eight are: five zeros, then the three `rsc_terminate`
returns for the state those 1653 leave the first component in, both components
from zero. Taking them as plain zeros instead costs C1 nothing but puts C2 659
and 639 bits out, so C2 is what settles it. C2 reads the 1656-entry table
`p4long.component_permutation` returns; the body rides the published 4968-entry
symbol interleaver.

The long header is its own waveform, cloned chip for chip the same way. The two
long frames send the same one (0.970), and it is NOT the short packet's: 0.245
against it, 0.412 at the best of a +/-900 Hz frequency scan, where permuting the
short header's chips already reaches 0.215 over that scan. What either header
asserts to a Dragon is still unread, and rendering one is not evidence of
acceptance.

### Level-3 field sizes across all ten speed levels

SCS's level-3 (PACTOR-4) receiver dispatch carries a ten-entry speed-level
table, path-1 sizes 25, 43, 72, 144, 206, 517, 1034, 1552, 2069, 2587 -- among
which are the two this document establishes independently, 43 (normal-short)
and 206 (normal-long). The other eight sizes are from that table only;
"bytes per packet" is this project's reading of what the table's numbers are,
not a labeled field in the binary.

## Robust-mode transmission

`p4robusttx` renders a robust-mode short packet by the value it carries.
Section 6.2's geometry is 20 header symbols, one phase reference and a
datablock, all spread by `p4sig.SPREAD16` to 1800 chips/s; the sequences are
not published, so the two measured headers are literals, and the datablock is
the one part of this that is decoded rather than cloned.

The tape is `rf-corpus/sigidwiki/PACTORIVaudio1.mp3`. Three caller bursts sit on
the 1.25 s raster at 4.863/6.113/7.364 s and measure 756-760 ms; three answerer
bursts follow at 9.214/10.464/11.714 s and measure 244-246 ms. 85 spread symbols
render as 759.4 ms and 27 as 243.9 ms, so the two lengths read as 20 + 1 + 64 --
section 6.2's short datablock, the variant with no user data field -- and
20 + 1 + 6, a field length the table does not list. Both are 2.3 kHz wide, carry
the 112.5 Hz comb at z 10-21 against 4.1 for the quiet channel, and hold the
SF-16 despread fraction at 0.31-0.39 end to end, where the same measurement
reads 0.118 on the CRC-valid normal-short packets of `PACTORIV_audio2.mp3` and
0.91 on clean local robust references. Correlated chip by chip the three caller
bursts agree 0.955-0.957 over exactly the first 320 chips and fall away after:
one 20-symbol header carried through the fade three times. Hard-decided it
survives intact on the strongest burst and misses two steps and one on the other
two, which is why a reader matches a header rather than equalling it.

The second recording is signals.taunus.de's `PACTOR-IV_112-5_DQPSK.WAV`, a
different station: four short packets on the same raster at
0.257/1.507/2.757/4.007 s, 757-759 ms, spending 0.905-0.912 of their energy on
the published spreading sequence, and unchanged across four analysis windows.
Seven packets in all, and they are the whole population everything below rests
on. Their dibits are in `tests/shrike/fixtures/p4-robust/`.

The carrier is the branch, not a residual. A differential quaternary alphabet
leaves a tone at four times the offset, and it repeats every 28.125 Hz; each
repeat relabels every dibit. The branch kept sits on the nominal 1500 Hz to
within 0.01 Hz and beats its neighbours on despread quality, 0.353 against 0.310
and 0.281, and on decision concentration, 0.708 against 0.372 and 0.505. The 84
dibits are unchanged across four analysis windows.

Section 6.2 calls the robust payload DQPSK. All 65 post-header differentials of
all seven packets are 0 or 2 -- differential BPSK -- while each header's 19 use
all four phases. No wrong carrier branch and no wrong chip phase produces that.

### The datablock is a [64,6,32] Walsh code

Read 0 -> 0 and 2 -> 1 in transmission order, the 64 datablock bits are
`p4robusttx.BLOCK_MASK` XOR one row of the 64-point Sylvester Hadamard matrix.
Any two of the seven blocks differ in exactly 32 places, and every one of them
leaves zero residual against the one shared mask -- a one-in-2**58 coincidence
apiece, and the mask was fixed by the first packet alone, so six of the seven
are out-of-sample. The datablock therefore carries **six bits**, spread 64 ways.
The tape's caller sent 6, then 9, then 8; the taunus station sent 13, 13, 49,
49. `p4robusttx.block_value` reads a block back and `p4oracle` reports it as
`p4_value` beside the Walsh peak that earned it.

What this refutes: the 64 bits are not an eight-byte block carrying a 6-bit
packed callsign and a CRC-16/X-25 in the Robust Call convention. Swept over
every window, block reversal, polarity, bit and byte order, all 256 constant XOR
masks and the P4 field whitener at nine offsets -- some 85,000 trials across the
five packets then in hand -- exactly one CRC closed, on one packet, at a window
straddling the header, spelling `'1#Z($$)%'`. That is the chance rate, and it
reproduced on none of the others. A status byte plus a CRC at any coded rate is
excluded by the code itself: a [64,6,32] difference structure leaves room for
six bits, not for twenty-four.

### Two headers, alternating

`p4robusttx.HEADER_DIBITS` holds both. The tape sends A three times; taunus
alternates A, B, A, B while its block repeats across each pair, so the header is
a packet parity and the value does not depend on it. Only the last eighteen of
the nineteen steps are station-invariant -- the first is 2 on the tape and 0 at
taunus -- so `HEADER_WORD` is what a reader matches on, and the five clean
packets carry their word exactly while the tape's two weak bursts miss one step
and two.

What stays unread: what a value MEANS, and where the mask comes from.
`BLOCK_MASK` is not the P4 field whitener at any bit offset in either bit order,
not its raw LFSR state stream, not PACTOR-3's crctable whitening, not any
degree-6 m-sequence and not periodic at 8 or 16. So a value is pinned only up to
a fixed 6-bit XOR and the mask only up to a global complement; relative values
are exact. There is still no CRC and no counter. The Chu root is not pinned
either -- `p4sig.header_score` returns 5, 11 and 7 for the tape's three bursts,
which carry one header, and 16 for the clean render of the first of them, one
root over ten copies at 10 dB. Keying a value asserts something to a Dragon
whose meaning this station cannot yet read.

The three control-signal clones are provisional. 26 differentials do not carry
the averaging 84 do: decision concentration 0.19-0.26 against the caller
packets' 0.71-0.81, and the runner-up carrier branch trailing by a tenth. Two
or three of each burst's six post-reference steps are not 0 or 2 at all, so the
six-bit field they are exactly the right width for cannot be read off them, and
no mask is known for them in any case. Their heads correlate 0.50-0.62 with each
other where the caller's three correlate 0.955, so they are three different
bursts rather than one answer repeated.

## Experimental changeover and turn handling

`p4breakintx` renders a 27-symbol Y27 head followed immediately by the
31-byte changeover field. `p4breakin` reads that field at 432 chips from the
head, returning bytes only after a unique CRC pass. The recorded KB5LZK and
VE3KPG greetings, and N9VC's SID, idle-changeover and close packets, are
regression fixtures. Carrier refinement uses the five known trainings within
±3 Hz; the returned likelihoods use decoder order.

`P4Receiver` owns the per-session emission registry. A provisional head can be
confirmed, contradicted or expired; a later complete envelope can extend a
word record once, and a decoded body enriches the same physical emission.
Its sample clock expects a packet 0.330 seconds after our answer word, or an
answer 0.923/3.403 seconds after our short/long packet. Normal packets are
identified by their training lattice without claiming payload bytes.

`P4Turn` keeps proposals separate from transmissions. Only an admitted key
spends a transmit attempt, and only a full accepted answer commits outgoing
bytes. Each received data copy must pass CRC and source-stream acceptance
before its own ACK deadline. Late bytes may be delivered, but cannot authorize
an ACK to another copy. Transparent escaping and supervisor state survive
packet boundaries; unsupported source modes terminate an attached application.

The optional `--p4-link` driver starts after the robust entry shorts.
`--p4-ladder long|chirp` selects the v13/v60 or v13/v49 experiment;
`--p4-data none|body|chirp` selects data behavior. The robust-long renderer is
unimplemented, so the long ladder enters a silent hold for
`--p4-hold-cycles` (default 6). `--p4-link-timeout` bounds the link session in seconds (default 600).
With `--no-mail`, `--p4-text` queues a SID and the requested Latin-1 text.
The normal radio admission and late-key checks remain authoritative.

These are experimental link policies. Recorded decoding and simulated-peer
traces establish offline behavior; device-loopback timing and the staged
gateway experiments are required before an interoperability claim. The placed
chirp receiver is separately opt-in and is not enabled by the link driver.
