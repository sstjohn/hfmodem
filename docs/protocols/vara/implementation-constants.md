# VARA HF implementation constants

VARA is an **alpha** implementation with successful live Winlink mail transfers,
including full-speed transfers. That milestone is broader than the retained
example set: the public send example reaches acknowledged **L9**, and the public
receive example is speed-limited **L4**. No high-speed receive recording survives
for publication. Bench validation across every codec row remains separate from
RF validation of a named session at a named rate.

The [illustrated guide](../../viz/vara-explained.html) explains the waveform
families and the high-speed pipeline. This map supplies the exact constants
behind that explanation via public source and the detailed protocol chapters.
Built-site source links are local and line-numbered. No private installation,
extracted table directory or unpublished Git history is needed to follow them.

## A shared chain, three waveform families

Every DATA frame is payload, one control byte and a big-endian
CRC-16/GENIBUS. Bits are MSB-first, zero-padded to the record's information size,
whitened from PN index zero, turbo encoded on the record's stage-2 permutation,
punctured to its coded size, then gathered by stage 1. Index modulation selects
one lit bin; tone modulation keys differential phase increments; OFDM maps
bit groups to full constellation columns and scrambles each cell's phase.
MFSK calling/answer bursts use their own callsign-derived tone sequence, with no
DATA whitening, interleaving or FEC.

The two turbo constituents have the same eight-state recurrence:
`a=u XOR r1 XOR r2`, `p=a XOR r0 XOR r2`, next `(a,r0,r1)`.
Both start in zero state and terminate with three steps each. The wire tail is
`[s1,p1,s2,p2]` repeated three times. Rate 1/3 retains all three streams; rate
1/2 alternates the constituents' parity. Higher rates group systematic bits
with one parity, alternating C1/C2 by group; the exact record size determines
any final zero padding. These rules define VARA’s tail placement and parity multiplexing.

## High-speed OFDM geometry

At 48 kHz, each symbol is a 1024-point IFFT plus a 128-sample cyclic prefix:
1152 samples, 41⅔ symbols/s, with 46.875 Hz bin spacing. Transmit synthesis uses
a 32-sample edge blend. Rows are filled bin-major, skipping references, pilots
and generator-defined gap cells. PSK is differential in time; QAM is coherent
on pilots, not differential.

| Bandwidth | OFDM levels | Active FFT bins | Carriers |
|---|---|---|---|
| 500 | L7–13 | 27–37 | 11 |
| 2300 | L10–16 | 8–56 | 49 |
| 2750 | L11–17 | 3–61 | 59 |

Each bandwidth follows seven OFDM rows in this order:

| Row | Constellation | Nominal code rate | Group size k |
|---|---|---|---|
| 1 | differential QPSK | 1/2 | 1 |
| 2 | differential QPSK | 2/3 | 2 |
| 3 | differential QPSK | 4/5 | 4 |
| 4 | differential 8PSK | 2/3 | 2 |
| 5 | differential 8PSK | 4/5 | 4 |
| 6 | coherent 16QAM | 5/6 | 5 |
| 7 | coherent 32-cross QAM | 4/5 | 4 |

PSK has four lead rows, a reference and 148 data rows (153 total). QAM has five
lead rows followed by 154 rows per frame; pilots appear at `8*j` on even FFT
bins and `8*j+1` on odd bins relative to the frame start. The highest row can
carry two frames: 313 rows instead of 159. BW2750 L17 carries 3922 net bytes
(3919 payload) per frame. The nominal top rate is approximately 8489 bps;
this is a speed descriptor, not a promise of delivered application throughput.
The [complete level tables](06-speed-gearshift.md#61c-levels-by-bandwidth) supply
information bits, coded bits, field bytes and timing for every row, not only
the top level.

## Every table input has a public definition

| Implementation input | Public definition and coverage |
|---|---|
| All bandwidth/level seeds and sizes | [tablegen.BW500](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L141), [tablegen.BW2300](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L148), [tablegen.BW2750](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L160); 17 `(seed,coded,N)` records per bandwidth. BW2750 shares the first nine BW2300 records, then its own chain. |
| PRNG and permutation rounding | [tablegen.step](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L60), [tablegen.draw](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L65), [tablegen.permutation](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L83); 24-bit VB6 recurrence, exact float32 multiplication/rounding and unique-value rejection. An integer floor shortcut is not equivalent. Stage 2 continues stage 1 state; each record’s remaining draws produce its gap/pilots and close on the next seed. |
| Whitening and phase scramble | [tablegen.whitener](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L102) uses seed 29318 and the high state bit. [tablegen.scramble](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L120) separately starts seed 29318, generating a 62×409 row-major table of `draw(129)` values; multiply cells by `exp(-j*2*pi*S/128)`. |
| Full constellation geometry and labels | [tablegen._column](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L491) / [tablegen._cross32](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L475) construct complete columns. Raw bit groups index them directly at OFDM levels, without an extra Gray pass. 32-cross is a 6×6 odd-coordinate grid with four corners removed, axis unit `0.3/sqrt(2)`. The historic 64-entry interleaved LUT is a truncated prefix, not all 32 labels. |
| FEC, dual tails and puncturing | [turbo._step](../../../packages/hfmodem/hfmodem/kestrel/coding/turbo.py#L49), [turbo.encode](../../../packages/hfmodem/hfmodem/kestrel/coding/turbo.py#L70), [turbo.encode_r13](../../../packages/hfmodem/hfmodem/kestrel/coding/turbo.py#L206) and [turbo._groups](../../../packages/hfmodem/hfmodem/kestrel/coding/turbo.py#L237) / [turbo.encode_punctured](../../../packages/hfmodem/hfmodem/kestrel/coding/turbo.py#L253); full trellis, tail order, group law, pad and soft-decode conventions. |
| CRC | [crc](../../../packages/hfmodem/hfmodem/kestrel/coding/crc.py); GENIBUS polynomial 0x1021, init/xorout 0xFFFF, no reflection, big-endian trailer. |
| OFDM rows, pilots, fill and receiver | [varaofdm.OfdmRecord](../../../packages/hfmodem/hfmodem/kestrel/rx/varaofdm.py#L64) / [varaofdm.ROWS](../../../packages/hfmodem/hfmodem/kestrel/rx/varaofdm.py#L157) supply all 21 rows’ geometry, full labels, channel/turbo permutations, pilot/gap layout and scramble. [tablegen.gap_cells](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L656) specifies exact missing/repeated cells. |
| OFDM transmitter and edge blend | [varaofdm_tx](../../../packages/hfmodem/hfmodem/kestrel/tx/varaofdm_tx.py) assembles frames, maps cells, carries differential state and renders CP/blend. Its output RMS is a local convention; its header/tail bytes are explicit. |
| Tone rows and shaping | [varatone.ROWS](../../../packages/hfmodem/hfmodem/kestrel/rx/varatone.py#L114) and [varatone_tx](../../../packages/hfmodem/hfmodem/kestrel/tx/varatone_tx.py) cover per-band geometry, labels and reference state; [tablegen.grid_row](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L1193) / [tablegen.tone_pilots](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py#L694) define reference grids and pilots. Exact pulse coefficients: [tone_pulse](../../../packages/hfmodem/hfmodem/kestrel/tx/tone_pulse.py) and [bw500_pulse](../../../packages/hfmodem/hfmodem/kestrel/tx/bw500_pulse.py). |
| Index rows, measured allocation constants | [varahf2300](../../../packages/hfmodem/hfmodem/kestrel/rx/varahf2300.py) / [varahf2300_tx](../../../packages/hfmodem/hfmodem/kestrel/tx/varahf2300_tx.py) and [varahf500](../../../packages/hfmodem/hfmodem/kestrel/rx/varahf500.py) / [varahf500_tx](../../../packages/hfmodem/hfmodem/kestrel/tx/varahf500_tx.py) define emission columns, strides, reference rows and coding. [tablegen](../../../packages/hfmodem/hfmodem/kestrel/rx/tablegen.py) contains the complete base-bin literals and constructors; [bw2750_allocations](../../../packages/hfmodem/hfmodem/kestrel/rx/bw2750_allocations.py) gives the widest base allocations. |
| MFSK calling, native answers and tokens | [vara_frames](../../../packages/hfmodem/hfmodem/kestrel/vara/vara_frames.py) contains alphabets, preambles, callsign hash/seed derivation, packed callsigns, answer/control tails and session tokens. [vara_mfsk](../../../packages/hfmodem/hfmodem/kestrel/vara/vara_mfsk.py) and [vara_control](../../../packages/hfmodem/hfmodem/kestrel/vara/vara_control.py) specify tone placement, pulse overlap and token waveforms. |
| Session lead state and draw cost | [vara_frames.lead_state](../../../packages/hfmodem/hfmodem/kestrel/vara/vara_frames.py#L603) seeds the caller/responder stream. [varaofdm_tx.cells](../../../packages/hfmodem/hfmodem/kestrel/tx/varaofdm_tx.py#L71) advances it by `lead*carriers` per frame, in bin-major order; a two-frame over consumes two lead blocks and transmits the second. [varatone_tx](../../../packages/hfmodem/hfmodem/kestrel/tx/varatone_tx.py) and [varahf2300_tx](../../../packages/hfmodem/hfmodem/kestrel/tx/varahf2300_tx.py) supply tone/index lead laws. |
| Frame sizing and ARQ use of levels | [phy](../../../packages/hfmodem/hfmodem/kestrel/arq/phy.py) supplies all native frame capacities, host/internal level mapping, caller-ID framing, control byte and over scheduling; [fsm](../../../packages/hfmodem/hfmodem/kestrel/arq/fsm.py) defines adaptation and timed control policy. These thresholds/budgets are implementation policy, not all proven universal peer behavior. |
| Source compression | [compression](../../../packages/hfmodem/hfmodem/kestrel/vara/compression.py) has Huffman symbols, escapes, mode bits and file framing. [Compression evidence and limits](08-compression.md) distinguish observed code assignments from unclosed cases. |

## Precision and evidence

The PRNG inputs and table constructors are sufficient to reproduce active DATA
permutations and phase tables without an extracted asset dump. Measured bin and
pulse literals remain public code constants, with their provenance explained
in the relevant source/chapter. A constructor's inactive compatibility padding
is not another active speed mode. Full physical detail is in
[physical layer](01-physical-layer.md), [coding](03-coding.md),
[frame formats](04-frame-block-formats.md) and
[speed adaptation](06-speed-gearshift.md). Their detailed bench evidence remains
available beside the descriptions. Every row being bench-validated does not
establish successful RF behavior at every bandwidth and speed.
