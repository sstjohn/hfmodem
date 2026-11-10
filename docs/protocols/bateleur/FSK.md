<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Binary FSK: the naval broadcast (STANAG 4481) and MIL-STD-188-110B 5.1

`hfmodem.bateleur.fsk` sends and receives binary FSK as STANAG 4481
(Edition 1) annex C gives it for NATO naval shore-to-ship broadcasts, with the
characteristic frequencies and conventions of MIL-STD-188-110B 5.1 and 4.2. It
was written from those texts. No other implementation was read, and no
recording of a broadcast has been tested against it.

## What the texts give

| Part | Mode names | Source |
|---|---|---|
| Single channel, 850 Hz shift (±425 Hz), space the higher; the mark's offset ends in 75 Hz so that the carrier sits on 100 Hz; the pair about 2000 Hz (110B table IV, HF: mark 1575, space 2425) or about 1700 Hz | `75W`, `75W/1700` | 4481 C.2; 110B 5.1.2, table IV |
| Single channel, 170 Hz shift about 2000 Hz | `75N`, `50N` | 110B 5.1.1, table IV |
| Multi-channel: sub-carriers at 255 + 170n Hz, n 1 to 16, each ±42.5 Hz; channel n twinned with n + 8 in diversity; channel 3 the one ships take as a single channel | `75V3`, `75V3D` | 4481 C.3 |
| Mark is binary one and the lower frequency | all | 110B 4.2.2, table II |
| Rates: 50 Bd, or 75 × 2^m | 50–1200 Bd | 110B 4.2.1 |
| PSK broadcasts: STANAG 4285 with its annex E, 75 bit/s and up | `MODE 4285 75L` and the rest | 4481 C.4, C.5 (`s4285`) |

## What neither says, and what is chosen here

- **The keying rate** of the FSK broadcast. 75 Bd, the rate 4481 annex D
  keys its FSK test interferer at, is the one tested most here; every rate
  in the list works.
- **Bytes.** A broadcast carries a synchronous stream, with no framing in
  either text. By default a message is sent as start-stop characters: a
  space start bit, eight data bits, and a mark stop bit, as teleprinters
  frame them, so that the far end finds the bytes. An `S` after the mode
  (`75WS`) sends the bare stream instead. A receiver joining a stream
  cannot tell where its bytes begin, so it starts at the first space after
  the idle line.
- **Around a message**, half a second of mark (the idle line) goes ahead and
  two characters' worth follow. The keying is phase-continuous.

On the station, a mode is keyed only if its tones and keying sidebands lie in
the span the station declares for bateleur (365–2985 Hz). So multi-channel
sub-carriers 1, 15 and 16, and any pair twinned with them, are for offline
work (`offair`) only.

## The receiver

`stream.ReceiverFsk` streams, and `fsk.receive` runs it over a recording. The
mode is set beforehand at both ends: nothing on the air says it.

- **Finding it.** Every quarter second it looks at the last half second.
  Where the tones stand in its spectrum gives the carrier offset, within a
  quarter of the shift (at most 60 Hz) either way. Whether they stand over
  the places beside them, where nothing is sent, says if a signal is there.
- **Soft bits.** At every sample, each tone's energy over the bit-long
  window from there. The soft bit is the mark's share less the space's;
  twinned channels' are added. The noise reference is averaged over half a
  second.
- **Timing.** Where a window lies across a bit's edge the tones mix and their
  difference shrinks; a bit is read half a bit from where it shrinks most,
  over the last 96 bits, weighted by the tones' energy so that a stretch
  where one tone has faded counts for little. On the idle line there are no
  edges and the timing goes where it likes; once edges are seen it moves a
  little at a time. Bits are decided 16 bits behind what has arrived.
- **Characters.** Nothing is taken until eight marks in a row, the idle line,
  have been read: the bits read as the signal comes up are not bits. After a
  stop bit that is not mark, characters sent back to back would let a wrong
  framing pass by chance. So the receiver hunts: it takes the framing under
  which the next six characters all start with a space and stop with a mark.
- **The end.** The signal has gone when no bit has stood over the noise for
  a second; a fade does not end it. Bits are released once a later one has
  stood clearly over the noise, halfway in decibels between the noise and
  the signal's own level, so the noise after the end does not become
  characters.

## How it is tested

Against itself only, through `bateleur.sim`'s channels, scored by character
error rate (edit distance over the message):

Three runs each, of a 159-character message:

| Mode | Channel | Offset | Character errors |
|---|---|---|---|
| 75W | white noise, 6 dB | +25 Hz | none |
| 75W/1700 | white noise, 10 dB | −30 Hz | none |
| 50N | white noise, 10 dB | +12 Hz | 0–2.5% |
| 75W | CCIR good, 10 dB | 0 | 0–4% |
| 75W, 75W/1700 | CCIR poor, 10 dB | 0 or −30 Hz | 2–11% |
| 75W | CCIR poor, 20 dB | 0 | 2–6% |
| 75V3 | CCIR poor, 6 dB | 0 | 4–7.5% |
| 75V3D | CCIR poor, 6 dB | 0 | 0–0.6% |

The poor channel's two paths, 2 ms apart, fade the mark and space tones
independently, and nothing here is coded. Even read at the true timing, 75W
there makes about 0.3% bit errors at 20 dB, which is several percent of
ten-bit characters. Diversity (`V…D`) is the broadcast's own answer to that.
Two minutes of noise start nothing. Fed in odd-sized pieces, back-to-back
messages come out whole. On the station, a message from the host port keys
through the arbiter, split across key-ups at 75 Bd, and decodes back; a
replayed transmission reaches the host port.
