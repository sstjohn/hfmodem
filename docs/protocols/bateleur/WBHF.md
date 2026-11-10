<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# MIL-STD-188-110C appendix D, 3 kHz

`hfmodem.bateleur.wbhf` sends and receives the 3 kHz waveforms of
MIL-STD-188-110C appendix D, the wideband HF family: waveforms 0 to 13, 75
to 16000 bit/s. It was written from MIL-STD-188-110C with Change 1,
appendix D. No other implementation was read, and nothing has been tested
against another sender.

The appendix covers channels of 3 to 24 kHz. Only 3 kHz is here: the rest
need wider channels than an amateur SSB transceiver passes. At 3 kHz it is
2400 symbols a second on 1800 Hz, the same as 188-110's serial tone and
STANAG 4539, so it goes through the same rig unmodified.

## The waveform

| Part | Where | Paragraphs |
|---|---|---|
| Waveforms 0-13: Walsh, BPSK, QPSK, 8-PSK, 16-, 32-, 64- and 256-QAM; their rates, frame lengths and mini-probes | `WAVEFORMS` | tables D-II, D-XI, D-XII |
| PSK numbers scrambled by adding the last 3 stages of x^9 + x^4 + 1, QAM by exclusive-or with its last 4-8; the register reset at every frame | `data_frame`, `s4539.scramble_values` | D.5.1.3 |
| The 256-QAM constellation | `QAM256` | table D-X |
| Waveform 0: a dibit a 32-chip Walsh symbol, an interleaver block's last from the alternate set, chips scrambled by the (159, 31) trinomial, 2048 values, reset at each block | `walsh_block`, `TRI` | D.5.1.2.3, D.5.1.4 |
| Preamble: 0-255 level-control blocks; M superframes of 9 fixed, 4 count and 5 waveform-identifier Walsh symbols | `preamble`, `superframe` | D.5.2.1 |
| Mini-probes: a base sequence extended (24, 32 or 48 symbols); the one after each long interleaver block's second-to-last frame shifted | `probe`, `shifted_after` | D.5.2.2, table D-XXI |
| K=7 code, full tail-biting, repeated then punctured (1/8 to 9/16) | `encode_block` | D.5.3.2, table D-XLII |
| Block interleaver, ultra short to long, loaded by increment and read in order | `encode_block` | D.5.3.3, table D-XLIII |
| End of message 4B65A5B2, and zero fill to whole blocks | `symbols`, `find_eom` | D.5.4.3 |

The test suite checks the appendix's worked examples:

- the trinomial's first 32 values;
- the dibit 01 scrambled with them;
- the 24-symbol mini-probe and its shifted form (table D-XXII);
- the 1/3 code as 2/3 applied after repeating;
- the waveform identifier and count codings.

### Where the text left a choice

- **Preamble scrambling.** Each section's table is 256 long, which is one
  symbol at 24 kHz. A 3 kHz Walsh symbol is 32 chips, and the appendix does
  not say which 32 of the 256 it takes. Here every symbol takes the first 32,
  as the level-control blocks do.
- **The checksum bit.** The appendix asks for d2, the identifier's checksum
  bit, and also says w1's least significant bit, which is d2, shall be 0.
  The checksum is sent here, and either is accepted.
- **The filter.** The appendix recommends a 35% root-raised cosine. Its
  requirement is only that the signal stay inside 3 kHz. bateleur sends its
  own pulse, the one all its serial waveforms share, which fits the
  station's filter.

## Modes

Modes are named by rate and interleaver: `US`, `S`, `M` or `L`, for example
`9600L`, `150US` or `75S`. Waveform 0 has no ultra short interleaver. Each
rate belongs to one waveform:

| Waveform | Rate | Modulation | Code | Interleaver, US / S / M / L |
|---|---|---|---|---|
| 0 | 75 | Walsh | 1/2 | – / 0.53 / 1.92 / 7.68 s |
| 1, 2 | 150, 300 | BPSK | 1/8, 1/4 | 0.16 / 0.64 / 2.56 / 10.24 s |
| 3, 4 | 600, 1200 | BPSK | 1/3, 2/3 | 0.11 / 0.43 / 1.71 / 6.83 s |
| 5-10 | 1600-9600 | BPSK to 64-QAM | 3/4 | 0.12 / 0.48 / 1.92 / 7.68 s |
| 11, 12 | 12000, 16000 | 64-, 256-QAM | 8/9 | 0.16 / 0.64 / 2.56 / 10.24 s |
| 13 | 2400 | QPSK | 9/16 | 0.12 / 0.48 / 1.92 / 7.68 s |

The receiver reads the waveform and interleaver from the preamble, so only
the sender sets the mode: `MODE WBHF <mode>` on the host port, or `offair
render WBHF --mode`. The modem always listens for it, as it does for 188-110
and 4539. A STANAG 5066 node set to it changes rate to any of its rates.

## The receiver

**Acquisition** looks for the preamble's 288 fixed symbols. Waveform 0 is to
work at -6 dB, where the fixed symbols score about 0.2 against the 0.35 the
other serial waveforms use. This one takes 0.13. Noise alone, searched over
minutes at three offsets, never went above 0.075. Nor did a 188-110, 4285 or
4539 transmission. A preamble is taken only once twice its fixed section has
arrived. The fixed section has sidelobes a Walsh symbol either side, and
these clear 0.13. Taken any sooner, the search had once settled on one of
them, and the header read was wrong.

**The header** is read from a symbol-spaced channel fitted to the fixed
section. Each count and identifier symbol's four values are scored against
what arrived, one symbol at a time, the chosen one then counted as known
for the next. The count and identifier taken are the valid codewords those
scores favour most. A longer preamble's count says how many superframes are
still to come.

**Equalization** is the 4539 receiver's: the block path equalizer, its gains
from the mini-probes, beside the RLS one. It takes the mini-probes' fifth
roots of unity (the 48-symbol probe of waveforms 1 and 2) as known points.
Waveform 0 is decided a Walsh symbol at a time by the path model alone, as
188-110's 75 bit/s is.

**Losing the signal.** These waveforms work below 0 dB, where the known
symbols' error never drops below the other receivers' "signal gone" line:
at -6 dB it is about 4 with the signal there. So reception ends when the
error is both past that line and six times what it was in the preamble,
sustained over a second or two. With the signal gone the error is past 30.

**Streaming** (`stream.ReceiverWbhf`) releases each interleaver block's
bytes as soon as it is decoded, less the few that might begin the end of
message.

## Not here

- The 6-24 kHz bandwidths.
- The K=9 code. It is optional, and the receiver does not accept it.
- The single-symbol fixed section. A sender may shorten the fixed section to
  one Walsh symbol when it sends one superframe. Only the nine-symbol form is
  sent or looked for.
- Late entry at a shifted mini-probe. It needs the waveform agreed
  beforehand, which is not how the modem is set up.
- The optional end-of-transmission marker. It is not sent, and not looked
  for.

## Measured performance

Table D-LII asks for a coded BER of 1e-5 with the long interleaver and a
20-superframe preamble, on white noise and on the 2 ms / 1 Hz poor channel.
Each figure below is from eight transmissions of about 25 s each, through
`bateleur.sim` with a 23 Hz offset, at the SNR the table gives
(`python -m hfmodem.bateleur.measure wbhf <mode> <channel> <snr>`). That is
15 kbit to 3.2 Mbit a point, so the rates are shown to the bits counted, not
to 1e-5 with any confidence.

| Rate | White noise | at | Poor (2 ms, 1 Hz) | at |
|---|---|---|---|---|
| 75 | 0 errors in 15 kbit | -6 dB | 0 in 15 kbit | -1 dB |
| 150 | 0 in 30 kbit | -3 dB | 0 in 30 kbit | 3 dB |
| 300 | 0 in 60 kbit | 0 dB | 0 in 60 kbit | 5 dB |
| 600 | 0 in 120 kbit | 3 dB | **1.4e-3** (one block of one transmission) | 7 dB |
| 1200 | 0 in 240 kbit | 5 dB | 0 in 240 kbit | 10 dB |
| 1600 | 0 in 320 kbit | 6 dB | 0 in 320 kbit | 11 dB |
| 2400 | 0 in 480 kbit | 6 dB | 0 in 480 kbit | 11 dB |
| 3200 | 0 in 640 kbit | 9 dB | 0 in 640 kbit | 14 dB |
| 4800 | 0 in 960 kbit | 13 dB | **1.4e-5** (13 in 960 kbit) | 19 dB |
| 6400 | 0 in 1.3 Mbit | 16 dB | 0 in 1.3 Mbit | 23 dB |
| 8000 | 0 in 1.6 Mbit | 19 dB | 0 in 1.6 Mbit | 27 dB |
| 9600 | 0 in 1.9 Mbit | 21 dB | 0 in 1.9 Mbit | 31 dB |
| 12000 | **9.0e-5** (216 in 2.4 Mbit) | 24 dB | – | |
| 16000 | 5.0e-6 (16 in 3.2 Mbit) | 30 dB | – | |

Three points miss, each by about a decibel:

- **600 bit/s on the poor channel.** The one transmission that failed lost
  a single interleaver block in a fade at 7 dB, and decoded whole at 8 dB.
  The failure at 4800 bit/s was in the same fading run.
- **12000 bit/s on white noise.** It had no errors in eight transmissions at
  25 dB. The equalizer's error is steady across every frame, so this is
  margin, not a fault: the 8/9 code leaves little room.

The other tests in D.6, each four transmissions:

- **Static multipath, D.6.2, at 50 dB.** No errors on any of the three
  channels: waveform 2 on 0, 3 and 9 ms (48 kbit), waveform 10 on 0, 2 and
  4.5 ms (960 kbit), and waveform 12 on 0 and 1.5 ms (1.6 Mbit).
- **Offset, D.6.4: waveform 10, long interleaver, poor channel, 30 dB.**
  At +75 Hz, no errors in 960 kbit. At -75 Hz, 19 errors in 960 kbit.
  This test asks only that synchronization be kept.

  At +75 Hz a transmission was first lost whole, because acquisition took
  the search from 0 Hz. That search read the offset 150 Hz wrong (a
  16-symbol chunk cannot tell the two apart). The same preamble found from
  +50 Hz now wins on its far higher score.
- **Doppler sweep, D.6.5.** Swept at 3.5 Hz/s through the -75 Hz corner
  at 24 dB: decoded whole, in the test suite.
