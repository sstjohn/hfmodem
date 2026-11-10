<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# bateleur: MIL-STD-188-110 serial, STANAG 4285 and STANAG 4539

This page covers the 2400-symbol/s serial waveforms: MIL-STD-188-110,
STANAG 4285 and STANAG 4539. See
[S4539.md](../../protocols/bateleur/S4539.md),
[TONE39.md](../../protocols/bateleur/TONE39.md) and
[TONE16.md](../../protocols/bateleur/TONE16.md) for the high-rate and parallel
waveforms. The family also includes the [STANAG 5066 link layer](S5066.md),
[2G ALE operation and monitoring](ALE.md), and
[HF selcall](../../protocols/bateleur/SELCALL.md).

bateleur is a modem in the station process, like the other four. The operator
configures the rig and enables each lane. bateleur listens on a stream lane,
submits transmissions through the arbiter and emission check, and serves
an application on a command port and a data port ([HOST-API.md](HOST-API.md)).
Enable it in the station file:

    [protocols.bateleur]
    enabled   = true
    host      = "bateleur"
    cmd_port  = 8310
    data_port = 8311

This is a physical layer. A transmission carries one message one way. There is
no ARQ in the physical layer; [S5066.md](S5066.md) describes the link layer.
A message longer than the rig's `max_key_s` goes out as several complete
transmissions. 2G ALE is described in [ALE.md](ALE.md),
and HF selcall (CCIR 493-4 and the UN open 6-digit selcall) in
[SELCALL.md](../../protocols/bateleur/SELCALL.md).

An offline tool renders the same transmissions to WAV files and decodes them,
for testing without a station:

    python -m hfmodem.bateleur.offair modes
    python -m hfmodem.bateleur.offair render 110 --mode 600L --callsign N0CALL --text "hello" out.wav
    python -m hfmodem.bateleur.offair render 110 --mode 2400S --bandwidth 3200 --text "hello" wide.wav
    python -m hfmodem.bateleur.offair decode 110 capture.wav
    python -m hfmodem.bateleur.offair render 4285 --mode 1200L --input msg.bin out.wav
    python -m hfmodem.bateleur.offair decode 4285 --mode 1200L --trim capture.wav

The offline tool never keys a radio.

## Interoperability status

**188-110 serial has decoded another sender; 4285 has not.** The receiver
has read published recordings of other 188-110A senders: the sigidwiki
collection for that mode (fetched for testing and not kept in this tree).
It read the mode from each preamble and found the end-of-message marker in
75S, 150S, 150L, 300S, 300L, 600S, 2400S and 4800N. The message in 150S,
150L and 4800N is 8N1 asynchronous characters ("THE QUICK BROWN FOX JUMPS
OVER THE LAZY DOG 0123456789."), and it came out whole. The others (1200S,
1200L, 600L, 2400L, 75L and the long off-air recordings) decoded without
reaching a marker; why has not been looked into. No recording
of another 4285 sender has been decoded, and no other implementation has
been shown to decode bateleur. So read "STANAG 4285" here as "built to the
published text".

The 39-tone receiver has decoded published recordings
([TONE39.md](../../protocols/bateleur/TONE39.md)). No foreign receiver has been
shown to decode these waveform transmitters.
[SELCALL.md](../../protocols/bateleur/SELCALL.md) records three published
selcall decodes. [ALE.md](ALE.md#interoperability-evidence) distinguishes
local checks from an earlier LinuxALE decode whose artifacts were not retained.

The 4285 Annex A tables were cross-checked against hcab14's published decoder.
The following readings still need retained, targeted interoperability fixtures:

| Reading implemented | Recording needed to settle it |
|---|---|
| 110 data randomiser: seed `0xBAD`, shift left with feedback mask `0x053` eight times before taking the low three bits; reset every 160 symbols | A known-payload serial-110 recording, with rate and interleave specified, spanning a data-randomiser reset |
| Count fields: send `4 + field`, hence short counts `(4,4,6)`, `(4,4,5)`, `(4,4,4)`; receive also accepts exchanged 01/10 rows when consecutive segments count down | Foreign short and long preambles with their rate and interleave known; the claimed table erratum has not been confirmed |
| 4285 Annex E: SOM `03873C3C`, searched on its last 26 bits; EOM `4B65A5B2`; flush `J*1024/(2*repeat)+102` bits; pair repetition before interleaving at 300/150/75 bit/s | Known-payload coded 4285 bursts at short and long interleave, including low rates and their complete leading/trailing flush; also a SOM-less stream with a known start |

Annex E is informative. A SOM-less stream is delivered from its first
decoded bit once acquisition has selected a deinterleaver alignment. Coded
110/4285 require the EOM plus a 64-bit flush window with at least 60 zeros.
The search checks every bit position. Uncoded 4800 and 4539 use exact EOM
matches because their framing does not guarantee that flush. The byte API
delivers complete bytes preceding the EOM; a final partial byte is omitted.

D1/D2 precede each new 110 interleaver block. The final two probes of the
transmission are zeros, since no new block follows. Reception treats these
probes as BPSK decisions.

## What is implemented

**MIL-STD-188-110 serial, fixed frequency**

- 4800 bit/s uncoded.
- 2400, 1200, 600, 300, 150 and 75 bit/s, each with short (0.6 s), long
  (4.8 s) or zero interleave.
  - The receiver reads the rate and interleave from the preamble.
  - Zero interleave carries the short setting's D1/D2, so the standard leaves
    it to be agreed with the far end. The receiver is told (`ZEROINTERLEAVE`
    on the host port, `--zero` offline).
- 75 bit/s sends each coded dibit as one of four 32-symbol Walsh sets, with no
  probes.
- A message ends with the EOM `4B65A5B2`, 144 flush bits, and zero fill to the
  end of the interleaver block.
- Several transmissions in one recording decode in turn.

**STANAG 4285**

- 2400, 1200, 600, 300, 150 and 75 bit/s coded, with short or long
  interleaving.
- 3600, 2400 and 1200 bit/s uncoded (Annex A's channel rates).
- Nothing on the air names the mode, so both ends must be set to the same one.
- Coded messages are framed per Annex E:
  - start-of-message `03873C3C`, the message, then EOM `4B65A5B2`;
  - then the tabulated flush zeros (at 600 bit/s long, 6246 of them, about
    10 s);
  - the receiver returns the message and nothing around it.
- Uncoded modes carry a bare bit stream with no framing. `--trim` drops
  trailing NUL padding.

Both standards carry a bit stream and say nothing about bytes. bateleur sends
bytes least-significant bit first by default, the asynchronous serial-port
convention; `BITORDER MSB` reverses it.

Many stations send start-stop characters instead: hardware modems and MARS
stations set to the asynchronous interface of 110B C.5.4.1.1 put a start
bit, the character and its stop bits on the air. `FRAMING ASYNC 8N1` (or
`offair --async 8N1`) does the same, both ways. The published 188-110A test
recordings are such a stream: the 150 bit/s one reads "THE QUICK BROWN FOX
JUMPS OVER THE LAZY DOG 0123456789" this way.

**STANAG 4539** (MIL-STD-188-110B appendix C): 3200-9600 bit/s and 12800
uncoded, the mode read off the preamble or a reinserted one. Described, with
its measured performance, in [S4539.md](../../protocols/bateleur/S4539.md).

**MIL-STD-188-110C appendix D in 3 kHz**: waveforms 0-13, 75-16000 bit/s,
Walsh to 256-QAM, the waveform and interleaver read off the preamble
(`MODE WBHF <mode>`). Described, with its measured performance, in
[WBHF.md](../../protocols/bateleur/WBHF.md).

**The 188-110A "mod" waveform** (receive only): a tone header, a parallel
section and a 188-110A data phase, with no published description. It was
worked out from off-air recordings. Its channel bits are received
(`offair decode MOD`); how they are coded is not known. Described in
[MOD110.md](../../protocols/bateleur/MOD110.md).

**The 39-tone parallel mode** (MIL-STD-188-110 appendix B): 75-2400 bit/s,
every interleaving degree, both diversity modes, synchronous and
asynchronous, the mode detected from the data; on the station (`MODE 39
<mode>`) and in `offair`. Described, with its measured performance and the
recordings it has decoded, in [TONE39.md](../../protocols/bateleur/TONE39.md).

**The 16-tone DPSK mode** (MIL-STD-188-110 appendix A): 75-2400 bit/s,
uncoded, with in-band diversity below 2400. Both ends set the rate
beforehand (`MODE 16 <rate>`); it is in `offair` too. Described in
[TONE16.md](../../protocols/bateleur/TONE16.md).

**STANAG 4529**: STANAG 4285 at 1200 Bd in 1240 Hz for maritime links, on
a selectable sub-carrier (`MODE 4529 <mode>`). Described in
[S4529.md](../../protocols/bateleur/S4529.md).

**Binary FSK** (STANAG 4481 annex C, MIL-STD-188-110B 5.1): the naval
broadcast's 850 Hz shift, the 170 Hz narrow shift, and its multi-channel
sub-carriers with diversity, 50-1200 Bd (`MODE FSK <mode>`). A 4481 PSK
broadcast is STANAG 4285 with its annex E, already here. Described in
[FSK.md](../../protocols/bateleur/FSK.md).

**STANAG 5066** (edition 1.2, annexes A-C): the HF subnetwork's link layer
on any of these waveforms. It covers ARQ, non-ARQ and expedited delivery, soft
and hard links, and the data rate change, with clients on TCP port 5066
(`host = "s5066"`). Described in [S5066.md](S5066.md).

## What is not implemented, and why

| Part | Fits 3 kHz? | Status |
|---|---|---|
| 188-110 2400 bit/s digital voice (D1/D2 7, 7) | yes | Carries vocoder frames rather than data, so the receiver ignores it, as described in 5.3.2.3.7.2.1. |
| 188-110 frequency-hopping variants | no | They need a hopping radio. |
| 188-110 two-channel ISB modes | no | They need two sidebands. |
| 188-110C appendix D above 3 kHz (6-24 kHz) | no | They need wider channels. Its 3 kHz waveforms are here ([WBHF.md](../../protocols/bateleur/WBHF.md)). |

## On the air

- **Occupied bandwidth.** The declared rows are 350–2995 Hz for `"3000"`
  and 350–3140 Hz for `"3200"`. Serial modulation uses a root-raised-cosine
  pulse with roll-off 0.1; `"3000"` adds a 2950 Hz low-pass. The 39-tone
  transmitter filters short-burst skirts in either setting. The tests use
  Welch peak power density, a Hann window with 30 Hz equivalent noise
  bandwidth and a −26 dB threshold, on short and 30-second payloads.
  `docs/CONFIG.md` explains the operator's bandwidth setting.
- **The rig's filter.** Many rigs' default SSB transmit filter is narrower
  than this. A steep 300–2700 Hz filter cuts the top 300 Hz, and the equalizer
  still decodes 2400 bit/s cleanly (`test_a_rig_filter_narrower_than_the_signal`).
  A wider data filter is still better if the rig has one.
- **Tuning.** Acquisition tolerates about ±75 Hz of mistuning. The loops track
  sound-card clock errors of at least 150 ppm.
- **The randomisers.** The serial waveforms add the standard's fixed
  pseudo-random sequence to the symbols to flatten the spectrum.
- **Identification.** `render --callsign` appends Morse identification at
  700 Hz in the same key-up. The identification sits inside the declared span.

## Receiver

The receiver:

1. Mix down, apply a matched filter, and keep four samples per symbol.
2. Find the known sequence (188-110's preamble sync or 4285's frame sync) by
   differential chunked correlation. This step estimates the carrier offset.
   For 188-110, read the mode and the segment count from up to three preamble
   segments at once, with the energy of every path collected.
3. Equalize. Two equalizers run on the same symbols, and each frame is taken
   from the one whose probes around it came out nearer right
   (`phy.Selector`):
   - **The path model** (`hfmodem/bateleur/patheq.py`), which does the work.
     It finds the channel's paths (their delays, to a sixteenth of a sample)
     from the known symbols, tracks each path's gain by RLS from symbols
     already decided, and computes the MMSE decision-feedback equalizer for
     that channel every few symbols. Frequency and timing loops follow the
     gains' common rotation and the fit's error against the pulse's slope.
     The paths are searched for again every half second, so a path that
     was faded at the start is still taken in.
   - **An RLS decision-feedback equalizer** (`phy.Equalizer`), which assumes
     nothing about the channel's shape and is used for a frame only when it
     is clearly better.
4. Compute max-log soft decisions, deinterleave, and run a soft Viterbi decoder.

`patheq.py` and `phy.py` describe the equalizer choices.

## Simulation measurements

The recorded short harness ran in simulation on 2026-09-30 with
Gaussian-spectrum Rayleigh fading. Each row uses one trial,
one second of requested payload (minimum 16 bytes), message seed 1000 and
channel seed 0. Preambles, interleaver fill and flush extend the waveform
beyond the requested payload duration. SNR is measured in 3 kHz; the
carrier offset is +23 Hz unless shown otherwise. Two-path channels have
independent equal-power taps. Spread is twice the Gaussian spectral standard
deviation. Every missing bit counts as an error.

Bounds below are one-sided 95% binomial upper bounds assuming independent
bit errors. Fading errors are correlated, so these are conditional bounds.
The short trials do not establish a standards performance target.

The channel process resolves the spread with a duration of at least
20/spread seconds and a period longer than twice the excerpt. It scales by
expected ensemble power, preserving the Rayleigh amplitudes of short
excerpts, and interpolates at the audio sample rate.

The recorded short trials below have zero errors in all eleven 110 and
twenty-one 4285 rows. Bounds are shown to three significant figures.

### 188-110: short table-xx run

| Mode | Channel: delay / spread | SNR dB | Bits | Errors | Upper 95% |
|---|---|---:|---:|---:|---:|
| 4800N | AWGN | 17 | 4,800 | 0 | 0.000624 |
| 4800N | 2 paths, 2 ms / 0.5 Hz | 27 | 4,800 | 0 | 0.000624 |
| 2400L | AWGN | 10 | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 2 ms / 1 Hz | 18 | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 5 ms / 1 Hz | 30 | 2,400 | 0 | 0.00125 |
| 1200L | 2 paths, 2 ms / 1 Hz | 11 | 1,200 | 0 | 0.00249 |
| 600L | 2 paths, 2 ms / 1 Hz | 7 | 600 | 0 | 0.00498 |
| 2400L | 2 paths, 2 ms / 5 Hz | 30 | 2,400 | 0 | 0.00125 |
| 300L | 2 paths, 5 ms / 5 Hz | 7 | 296 | 0 | 0.0101 |
| 150L | 2 paths, 5 ms / 5 Hz | 5 | 144 | 0 | 0.0206 |
| 75L | 2 paths, 5 ms / 5 Hz | 2 | 128 | 0 | 0.0231 |

### 4285: short annex-b run

| Mode | Channel: delay / spread | SNR dB | Bits | Errors | Upper 95% |
|---|---|---:|---:|---:|---:|
| 2400L | AWGN | 12 | 2,400 | 0 | 0.00125 |
| 2400L | 1 path, 1 Hz | 15 | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 2 ms / 1 Hz | 15 | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 2 ms / 1 Hz | 20 | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 1 ms / 0.5 Hz | 15 | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 1 ms / 0.5 Hz | 20 | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 1 ms / 4 Hz | inf | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 1 ms / 8 Hz | inf | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 5 ms / 0.5 Hz | inf | 2,400 | 0 | 0.00125 |
| 2400L | 2 paths, 6 ms / 0.5 Hz | inf | 2,400 | 0 | 0.00125 |
| 1200L | AWGN | 4 | 1,200 | 0 | 0.00249 |
| 1200L | 1 path, 1 Hz | 10 | 1,200 | 0 | 0.00249 |
| 1200L | 2 paths, 2 ms / 1 Hz | 10 | 1,200 | 0 | 0.00249 |
| 1200L | 2 paths, 1 ms / 0.5 Hz | 10 | 1,200 | 0 | 0.00249 |
| 1200L | 2 paths, 1 ms / 8 Hz | inf | 1,200 | 0 | 0.00249 |
| 600L | AWGN | 2 | 600 | 0 | 0.00498 |
| 600L | 1 path, 1 Hz | 5 | 600 | 0 | 0.00498 |
| 600L | 2 paths, 2 ms / 1 Hz | 5 | 600 | 0 | 0.00498 |
| 600L | 2 paths, 1 ms / 0.5 Hz | 5 | 600 | 0 | 0.00498 |
| 600L | 2 paths, 6 ms / 0.5 Hz | inf | 600 | 0 | 0.00498 |
| 600L | 2 paths, 5 ms / 5 Hz | 10 | 600 | 0 | 0.00498 |

Reproduce these simulation rows serially:

```sh
python -m hfmodem.bateleur.measure --preset table-xx --trials 1 --seconds 1 --output table-xx-short.json
python -m hfmodem.bateleur.measure --preset annex-b --trials 1 --seconds 1 --output annex-b-short.json
```

`measure.py` records source hashes, conditions, library versions, seeds,
bits, errors, missing bits, acquisition, EOM and conditional bounds. Omitting
the short-run overrides requests the longer preset; the counts above do
not describe those longer runs.
