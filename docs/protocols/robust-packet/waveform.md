<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Robust Packet waveform

brubru renders and receives Robust Packet at 9600 samples/s. The
[station attachment](../../implementation/robust-packet/station.md) converts the card stream, assesses channel
occupancy, and serves KISS clients.

Eight carriers run at 50 symbols/s, spaced 60 Hz apart, from 1290 to 1710 Hz
about a 1500 Hz centre. The receiver also handles spectral inversion and fits
frequency and timing from the eight-symbol preamble. [P: 3 stations, 4 frames]

The first symbol supplies the differential reference, followed by eight preamble
symbols and eight header symbols. Level 0 uses differential binary phase steps
of −45° and −225°. The pulse has root-raised-cosine roll-off 0.33.
[P: 3 stations, 4 frames]

The crest limiter, enabled by default, reproduces the characteristic phase changes in
`RPR_Frame.wav`: regeneration agrees on at least 1327 of 1328 differential
bits. [P: 1 station] With limiting disabled, regeneration agrees on all 1328
bits in `Robust-packet.mp3` and the first radioscanner frame.
[P: 2 stations] These counts compare received differentials, including the
limiter's effects, rather than just the intended encoded bits.

Level 1 maps two bits to differential phase steps −45°, −135°, −315° and
−225°, with a rate-3/4 punctured code. The 32 length classes at each level
are exercised in generated audio, both clean and with noise at 7 dB for level 0
and 10 dB for level 1, referenced to a 3 kHz noise bandwidth, with inversion
and offsets of ±15 Hz. [T: 1 implementation, synthetic frames, 2026-09-30]

The transmit buffer contains seven zero-symbol slots for the shaping filter's
ramp-down and another 20 ms of silence. TXDELAY is scheduling metadata in
milliseconds; no scheduled lead is inserted into the buffer. The card conversion
is `core.resample.for_card`, with peak drive set by `core.levels.at_drive`.
Onset is the first converted sample reaching half a PCM16 LSB at default
drive, excluding sub-quantisation filter residue. These are implementation choices.

The unwindowed finite-burst −26 dB span after card conversion, swept over both
levels, both limiter settings, four capacities and three payload seeds, rounds
outward to 1247–1755 Hz. `core.occupied` records that envelope. [T: 1 implementation, 48 synthetic bursts, 2026-09-30]

The stateful card-input decimator retains its FIR history across blocks. Its
101-tap symmetric FIR has delay D = 50 card samples. Output index n is labelled
with the card index 5n + D. A discontinuity requires an explicit reset.

Frames of at least 300 symbols use decision-directed timing and phase estimates
every 32 symbols. Timing also uses matched-symbol energy; low-confidence sections
retain the previous grid. The changing carrier is removed before a second
matched-filter pass. Short frames retain the preamble grid. These are receiver
design choices.
[T: generated clock error, drift, and fade tests]
