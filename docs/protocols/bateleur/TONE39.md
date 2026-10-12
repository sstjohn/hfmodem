# The 39-tone parallel mode (MIL-STD-188-110 appendix B)

`hfmodem.bateleur.tone39` sends and receives the 39-tone parallel waveform of
MIL-STD-188-110B appendix B (the same text as 110A's appendix B), 75 to 2400
bit/s. It was written from the appendix. No other implementation was read.
Four published recordings of other senders were used as test partners only:
they are not in the tree, and nothing was taken from them but which way the
text's open choices go.

## The waveform (appendix B)

| Part | Where | Paragraphs |
|---|---|---|
| 39 data tones, 675–2812.5 Hz, 56.25 Hz apart; QDPSK each 22.5 ms element (216 samples at 9600, a 45-sample guard); an unmodulated Doppler tone at 393.75 Hz, 6 dB up | `transmit`, `_element` | B.5.4, B.5.5 |
| Table B-V: (0,1) +45°, (0,0) +135°, (1,0) +225°, (1,1) +315°; the earlier bit is the odd bit location | `_DPHI`, `element_dibits` | B.5.5 |
| Reed-Solomon (15,11) over GF(16), x⁴+x+1, g with roots α¹–α⁴; shortened to (14,10) at 2400 bit/s and (7,3) below | `rs16` | B.5.2 |
| Block interleaver, a byte a row, code words filled in pairs; degrees per table B-I | `encode_superblock`, `_load_order` | B.5.3 |
| Framing: the last *n* bits of the nine-stage sequence ending at the SPACE after nine MARKs, before each group of super blocks (tables B-III, B-IX to B-XI) | `framing_bits`, `coded_stream` | B.5.4.3 |
| In-band diversity below 1200 bit/s: time and frequency (a word's copies on later elements), or frequency only; tones 33–39 repeat 1–7 | `_groups`, `_combine` | B.5.6 |
| Asynchronous characters of 7 to 12 bits, start SPACE, stop MARK, caller-supplied parity after 7-bit data; EOM all MARK; the null character | `chars`, `null_char`, `_unchar` | B.5.7 |
| Preamble: part one's four tones, part two's three turning 180° each element, part three's 39 phases; the extended preamble for the longer interleavers | `preamble` | B.5.4.2 |

Where the text leaves a choice open, the published recordings settled it. Each
tone's phase is reckoned from the start of its own element. Figure B-5's
register is read with its labels. A symbol's first bit sent is its most
significant. The module's docstring says which recording settled which.

## The receiver

**Acquisition** looks for part one's four tones, searched on a grid ±80 Hz
wide. It takes them when they hold a quarter of the energy on average over
half of part one. Noise gives a few hundredths and data under a tenth. The
tones' phase advance over part one then gives the carrier offset to a
fraction of a hertz.

The element timing comes from which tones each element holds: part one's
four, part two's three, then part three's and the data's forty. Part one's
and two's tones are data tones as well, so they are left out of that last
set, and each boundary reads in both directions. Energy is used, not phase:
across two paths the tones' phases part company, and a match that summed
them lost the timing by an element or two even at 30 dB.

That timing is good to within the guard. `settle` then centres it. The
changes from element to element don't care where in the guard the window
starts, so how cleanly they fall on the QDPSK points is flat across every
timing the paths allow. The middle of that plateau is taken, when the
curve stands out from the noise.

**Where the data starts** is settled by the framing that opens it, not by
counting the preamble's parts. Senders lay those parts out differently:
one recording has an extra all-tone element before its data, another one
between parts one and two. A start one element late still reads as framing,
because the framing runs for several elements. But the framing it measures is
then short of every tabulated length, so the start whose framing is as long
as a table says is taken.

**Offsets and clocks.**

- **Carrier offset.** Each element is reckoned from its own start, so an
  offset turns every change by 2π·df·T, about 8° a hertz. That is known,
  and past 5.5 Hz it is beyond what the fourth power can resolve, so it is
  taken off before.
- **Clock error.** The sender's clock is measured from the whole message:
  a clock error turns each tone's changes in proportion to its frequency.
  The fourth power removes the data, and the slope across the tones gives
  the clock to about 20 ppm through fading. Each element is then read
  where it falls, the sub-sample remainder corrected in phase. Without
  this, 300 ppm (a poor sound card) loses a 30 s message. With it, 800 ppm
  through the 2 ms / 2 Hz channel decodes. Two guards constrain the estimate.
  It is measured only over the elements the transmission is on (the data
  tones' median power against the empty places), not the Morse
  identification after it: one steady tone read as a clock of −89 000 ppm.
  And it is taken only when the slope stands three standard errors clear
  of none: a short message at 4 dB through fading showed −538 ppm that was
  not there, and lost its last 20 bytes to it.
- **Gross clock errors.** One recording is 0.76% fast. Part one's four
  tones, spread over 2 kHz, show that, and the audio is resampled when the
  data then reads better. Fading scatters that estimate by several hundred
  ppm, so it is used only past 1500.

**Decoding.** A (7,3) word is the code word nearest the soft bits, all 4096
of them weighed at once. A (14,10) word that doesn't check is searched for
among erasures of its least reliable nibbles, two at a time with an error or
four at a time, and the nearest candidate found is taken. Against hard
decisions and two erasures this took 1200 bit/s at 15 dB from 4e-4 to no
errors.

**The mode** isn't carried in the preamble, so it is detected:

1. The framing sequence's length and the tones it arrives on give the rate
   and diversity.
2. The first super block's code words give the interleaving degree.
3. How the characters are framed decides between synchronous and
   asynchronous: per B.5.7, or as some senders leave them (start SPACE,
   stop MARK).
4. Text decides the character width.

A sender working from another edition's tables: one recording frames 75
bit/s with 384 bits, a length no table here has. Its framing length is
measured from the sequence's end, and the degree is found from the code
words (`Mode39(..., seq=384)`, named `75/4/S:384`).

## Late entry

Appendix B defines no late-entry procedure as such. It makes one possible:
B.5.4.3 puts the framing sequence back into the data every few super blocks
(table B-III), and from each insertion the data runs on exactly as it did
from the first. `tone39.late_entry` (and `offair decode 39 --late --mode
<mode>`) joins a transmission whose preamble was missed:

- the offset comes from the Doppler tone, a line well over its empty
  neighbourhood below the first data tone;
- the element timing comes from the data. A window across an element's edge
  mixes two elements' phases, so its changes sit less tightly on the
  constellation. The receiver fits the run of windows that lie wholly
  inside an element;
- the framing sequence is looked for at every bit of the demodulated stream,
  since in some asynchronous 2400 bit/s modes an insertion begins inside an
  element. Decoding then starts there, as it would after the preamble.

The mode must be known. How long joining can take is the mode's insertion
interval: 2.4 s at 2400 bit/s with degree 9, 9.4 s at 1200 bit/s with degree
63, 20 s at 600 bit/s with degree 33, and a minute and a half to two and a
half minutes at 75 bit/s.

Tested against itself at 1200, 600 (CCIR poor) and 2400 bit/s, synchronous
and asynchronous. It is offline only: the streaming receiver still needs the
preamble.

## Simulation measurements

The recorded short harness was rerun in simulation on 2026-09-30 with
corrected Gaussian-spectrum Rayleigh fading. Each row uses one trial,
one second of requested payload (minimum 16 bytes), message seed 1000 and
channel seed 0. Preambles, interleaver fill and flush extend the waveform
beyond the requested payload duration. SNR is measured in 3 kHz; the
carrier offset is +23 Hz unless shown otherwise. Two-path channels have
independent equal-power taps. Spread is twice the Gaussian spectral standard
deviation. Every missing bit counts as an error.

Bounds below are one-sided 95% binomial upper bounds assuming independent
bit errors. Fading errors are correlated, so these are conditional bounds.
The short trials do not establish a standards performance target.

The earlier long-run B-XII summaries lacked per-trial records and a source
revision; their fading figures are withdrawn. In the corrected short run,
1200L at 15 dB changed from 0 to 2 errors in 1,200 bits: BER 0.00166667,
conditional upper bound 0.00523711 instead of 0.00249333. Both errors were
in delivered data; no bits were missing. The other six rows retained zero
errors and unchanged conditional bounds.

| Mode | Channel: delay / spread | SNR dB | Bits | Errors | Upper 95% |
|---|---|---:|---:|---:|---:|
| 2400L | 2 paths, 2 ms / 2 Hz | 20 | 2,400 | 0 | 0.00124744 |
| 2400L | 2 paths, 2 ms / 2 Hz | 30 | 2,400 | 0 | 0.00124744 |
| 1200L | 2 paths, 2 ms / 2 Hz | 15 | 1,200 | 2 | 0.00523711 |
| 1200L | 2 paths, 2 ms / 2 Hz | 20 | 1,200 | 0 | 0.00249333 |
| 300L | 2 paths, 2 ms / 2 Hz | 6 | 296 | 0 | 0.0100697 |
| 300L | 2 paths, 2 ms / 2 Hz | 8 | 296 | 0 | 0.0100697 |
| 75L | 2 paths, 2 ms / 2 Hz | 4 | 128 | 0 | 0.0231324 |

A separate existing clock regression also changed in simulation. At 600/33/S,
two equal paths 2 ms apart with 2 Hz spread, 10 dB SNR in 3 kHz, +7 Hz
offset, −300 ppm clock error and channel seed 1, a 13,440-bit repeated payload
changed from 0 to 2 errors, with no missing bits. Its conditional upper bound
rose from 0.000222872 to 0.000468362. The clock estimate changed from −261.01
to −245.06 ppm, exceeding the test's 40 ppm tolerance. Neither run detected
EOM. This regression remains open; the test assertion is unchanged.

Reproduce these simulation rows:

```sh
python -m hfmodem.bateleur.measure --preset b-xii --trials 1 --seconds 1 --output b-xii-short.json
```

Omitting the short-run overrides requests sixteen 20-second payloads per
row; the counts above do not describe that longer run. See
[README.md](README.md#simulation-measurements) for the channel correction
and the recorded fields.

## Recordings

| Recording | Found | Result |
|---|---|---|
| A 600 bit/s test message | 600/99, ten-bit characters, extra all-tone element before the data | Every character, to the EOM |
| A 75 bit/s transmission, 0.76% fast and 10 Hz off | 75/4 with a 384-bit framing | Every code word checks; a test pattern, not text |
| Two 2400 bit/s transmissions | 2400/288 and 2400/144; start SPACE, stop MARK | About 2% of bits wrong in the recordings themselves; the text ("ZCZC ... 39 Tone Mode:") partly legible |

## On the station

The modem always listens for it (`stream.Receiver39`), on the front end's
real audio at 9600 Hz: the 1800 Hz matched filter the serial waveforms use
would cut the Doppler tone. With `MODE 39 <mode>` set, a transmission is
decoded in that mode as soon as its timing is found; with none, the mode is
found from the data once about eight seconds of it are in (`detect_elements`).
Elements are read as they arrive, each where the sender's clock puts it: the
clock is measured every 128 elements over the last 384, and the reading moves
half way to it. `tests/bateleur/test_stream.py` holds the streaming receiver
to the offline one, 400 ppm included. Transmitting, the modem reckons how long
a message keys from `element_count`, without rendering it, and splits one too
long for `max_key_s`.

The SNR reported is the data tones' power against the four empty places
between the Doppler tone and the first data tone (450–619 Hz), over the
elements the message was decoded from: within a dB or two of the channel's.

## Not done

- Late entry on the station: the streaming receiver needs the preamble;
  joining under way is offline only (`late_entry`).
- Gross clock errors (past 1500 ppm, a recording played at the wrong speed)
  are put right offline only; the streaming receiver follows what a sound
  card does.
- The extended preamble is sent and received but was tested only in
  loopback.

The streaming receiver searches both preamble lengths without an external
extended-preamble flag. Asynchronous transmission uses MARK stop bits, as
in the cited 2400 bit/s recordings. A caller sending seven data bits can
supply parity in the high bit of each input byte; the receiver still accepts
both observed stop conventions. The continuous Doppler pilot starts the
data at the phase reached by the end of preamble part three. A bandpass
filter limits short-burst skirts; its 400-sample tail is included in duration.
