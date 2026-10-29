<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR implementation constants

PACTOR is an **alpha** implementation. PACTOR-3 has successful mail exchanges.
PACTOR-4 is an extension still under development, with narrower recorded and
live protocol evidence.
A recovered waveform or CRC-clean frame does not, by itself,
prove a complete interoperable mail session.

The [illustrated guide](../../viz/pactor-explained.html) explains the signal and
ARQ exchange. This companion maps the complete implementation inputs to public
source: literal codewords, table constructors, compact resources, bit conventions
and framing. Links in the built site open local, line-numbered copies from this
release. The source links also work in the source distribution. A protocol's
wire constants and hfmodem's receiver thresholds or sample-rate choices are
identified separately; copying a threshold is not a protocol requirement.

## PACTOR-3: packet to audio

The ordinary packet travels through source coding, fixed-length fill, status,
CRC, whitening, convolutional coding, puncturing, full-frame interleaving and
carrier placement. Link entry adds acquisition. An established packet starts
with a phase-reference row and the constant/variable header rows before its
coded body. Net field bytes enter the convolutional encoder **LSB-first** after
whitening; the CRC trailer is little-endian. The receiver reverses those steps and checks the CRC before delivery.

The carrier law is 18 tones at `480 + 120*c` Hz, 100 Bd, centred at 1500 Hz.
SHORT uses 72 body rows; LONG uses 320. A phase-reference row and eight header
rows are separate from that body. Speed level selects the active carrier set,
DBPSK/DQPSK width, convolutional code, puncture mask and field size. Both the
transmit path's tap convention and parity order matter: an octal pair alone is
not enough to reproduce a packet.

| Implementation input | Public definition and exact convention |
|---|---|
| Carrier sets, speed levels and capacities | [spec.SPEED_LEVELS](../../../packages/hfmodem/hfmodem/shrike/spec.py#L114); all 6 rows, rate, K, short/long payload bytes, active channels and nominal throughput. [spec](../../../packages/hfmodem/hfmodem/shrike/spec.py) also defines carrier frequencies and cycles. |
| Acquisition, headers and carrier swap | [spec.VARIABLE_HEADERS](../../../packages/hfmodem/hfmodem/shrike/spec.py#L207), [spec.CONSTANT_HEADERS](../../../packages/hfmodem/hfmodem/shrike/spec.py#L227) and [p3frame](../../../packages/hfmodem/hfmodem/shrike/p3frame.py); all 16 variable and constant codewords, middle-nibble transmit transform, header tone ordering, acquisition reference and dibit labels. |
| Status, fill and control words | [spec.status_byte](../../../packages/hfmodem/hfmodem/shrike/spec.py#L412); [spec.field_fill](../../../packages/hfmodem/hfmodem/shrike/spec.py#L386) and [spec.CONTROL_SIGNALS](../../../packages/hfmodem/hfmodem/shrike/spec.py#L455) cover status bit positions, the 15-byte repeating fill, 6 control words and CS timing. |
| CRC and whitening | [crc.crc16](../../../packages/hfmodem/hfmodem/core/crc.py#L57) defines X.25 init/reflection/xorout (default), and [coding.whiten](../../../packages/hfmodem/hfmodem/shrike/coding.py#L37) XORs the reflected CRC table in big-endian words, resetting per packet. Packet CRC bytes are little-endian in [placement.build_field](../../../packages/hfmodem/hfmodem/shrike/placement.py#L456). |
| Convolutional trellis and puncturing | [fec.ConvCode](../../../packages/hfmodem/hfmodem/core/fec.py#L14) defines register/tap ordering; [coding.CODE_K9](../../../packages/hfmodem/hfmodem/shrike/coding.py#L88) explains reciprocal K=9 polynomials; [placement.CONV_GENERATORS](../../../packages/hfmodem/hfmodem/shrike/placement.py#L49) pins actual K=7 TX parity order. [coding.PUNCTURE_3_4](../../../packages/hfmodem/hfmodem/shrike/coding.py#L125) and [coding.PUNCTURE_8_9](../../../packages/hfmodem/hfmodem/shrike/coding.py#L126) supply the complete SL5/6 masks. |
| Field sizing, tail bits and interleaving | [placement](../../../packages/hfmodem/hfmodem/shrike/placement.py) defines each Path, short/long strides, CRC coverage, flush bits, physical carrier-order walk and differential phase labels. Use [placement.link_packet](../../../packages/hfmodem/hfmodem/shrike/placement.py#L863) as the implemented packet entry point rather than a generic candidate interleaver. |
| Pulses, phases and CS placement | [modem](../../../packages/hfmodem/hfmodem/shrike/modem.py) and [placement.assemble](../../../packages/hfmodem/hfmodem/shrike/placement.py#L589) contain pulse generation, filter law, initial carrier phases, stagger and gain conventions; [placement.control_burst](../../../packages/hfmodem/hfmodem/shrike/placement.py#L958) covers 20-bit two-tone controls. 48 kHz is this implementation’s audio convention. |
| Source coding and link policy | [compress](../../../packages/hfmodem/hfmodem/shrike/compress.py) contains the full Huffman/reference tables, escapes and run coding; [arq](../../../packages/hfmodem/hfmodem/shrike/arq.py) contains P3 ladder, cycle changes, repeat budgets, changeover and disconnect state machine. Receiver acquisition and decoding are in [p3rx](../../../packages/hfmodem/hfmodem/shrike/p3rx.py). |

## PACTOR-4: two convolutional components, two different permutations

The **convolutional code** is an eight-state recursive systematic constituent,
used twice in parallel. With newest delay first, state `(s1,s2,s3)` and input `x`:

```
u = x XOR s2 XOR s3
parity = u XOR s1 XOR s3
next state = (u,s1,s2)
```

Its zero-state transfer is `(1+D+D³)/(1+D²+D³)`. C1 reads the systematic bits
in order; C2 reads `systematic[pi2[j]]`. This C2 input permutation is not the
symbol permutation applied after parity packing. The primitive takes explicit
states and permutation; packet paths supply zero starts and their specific
padding, flush and packing rules. Three flush inputs terminate C1 where those
paths specify it. Do not import VARA's 12-bit dual-tail format into PACTOR-4.

Systematic bits are retained. Nominal rate 1/3 keeps both parity streams; rate
1/2 keeps `C1[0::2]` and `C2[1::2]`; rate 5/6 keeps `C1[0::10]` and `C2[4::10]`.
Those patterns are defined from index zero of the component stream; tail packing
is a separate packet-path rule. LLRs use positive values for bit zero.

| Implementation input | Public definition |
|---|---|
| Constituent recurrence, termination and packing | [p4coding.rsc_encode](../../../packages/hfmodem/hfmodem/shrike/p4coding.py#L65), [p4coding.rsc_terminate](../../../packages/hfmodem/hfmodem/shrike/p4coding.py#L77), [p4coding.puncture](../../../packages/hfmodem/hfmodem/shrike/p4coding.py#L110); soft trellis and component exchange in [p4decode](../../../packages/hfmodem/hfmodem/shrike/p4decode.py). |
| Status, CRC and packet-reset whitener | [p4frame](../../../packages/hfmodem/hfmodem/shrike/p4frame.py); status E/S1S0/M2M1M0/C1C0, payload + status + X.25 CRC little-endian, XOR register seed `0x00B770CD` with feedback `0x48000000` over all net bytes. |
| C2 tables and integrity | [p4c2](../../../packages/hfmodem/hfmodem/shrike/p4c2.py); all 13 supported sizes, compressed-resource digests, canonical u16-LE permutation digests, base construction and replay order. [Full derivation and resource downloads](pactor4-c2-derivation.md). |
| Original stochastic constructor and recipes | [__main__](../../../packages/hfmodem/hfmodem/shrike/c2proof/__main__.py) and [cml_generate.c](../../../packages/hfmodem/hfmodem/shrike/c2proof/cml/cml_generate.c) / [original ran2 header](../../../packages/hfmodem/hfmodem/shrike/c2proof/cml/upstream/srandom.h). Twelve exact seed/history recipes; original 1160 seed/history unresolved. [Attribution](../../../packages/hfmodem/hfmodem/shrike/c2proof/NOTICE.md). |
| Symbol permutation and constellation labels | [tablegen.INTERLEAVERS](../../../packages/hfmodem/hfmodem/shrike/tablegen.py#L126), [tablegen.p4_symbol_permutation](../../../packages/hfmodem/hfmodem/shrike/tablegen.py#L146) and [tablegen.p4_constellation_points](../../../packages/hfmodem/hfmodem/shrike/tablegen.py#L156); arithmetic construction, Gray PSK ordering and the special 16/32-point geometries with scale/quantisation convention. |
| Normal SHORT coding | [p4shorttx](../../../packages/hfmodem/hfmodem/shrike/p4shorttx.py) (352 systematic bits, 43 net bytes); [p4shortlevels](../../../packages/hfmodem/hfmodem/shrike/p4shortlevels.py) for widths 1–3 and [p4qam16](../../../packages/hfmodem/hfmodem/shrike/p4qam16.py) for width 4. The SHORT symbol permutation covers 1056 body symbols. |
| Normal LONG coding | [p4longtx](../../../packages/hfmodem/hfmodem/shrike/p4longtx.py) / [p4long](../../../packages/hfmodem/hfmodem/shrike/p4long.py) for the BPSK path; [p4longrates](../../../packages/hfmodem/hfmodem/shrike/p4longrates.py) specifies SL6–10 widths 1–5, 4968 symbols, systematic sizes 4140/8280/12420/16560/20700 and net bytes 517/1034/1552/2069/2587. This higher-order module is a standalone diagnostic path, not a claim of live dispatch at every level. |
| Training, pulse and spread factors | [p4normal](../../../packages/hfmodem/hfmodem/shrike/p4normal.py) gives 32-symbol CAZAC training and block layouts; [p4rx](../../../packages/hfmodem/hfmodem/shrike/p4rx.py) has the exact 129-tap pulse and SF8 chips; [p4sig](../../../packages/hfmodem/hfmodem/shrike/p4sig.py) has 1800 chip/s, 1500 Hz centre, SF16 and Chu header templates. Packet callers select the layout; this primitive does not infer header meanings. |
| Robust words, packets and entry | [p4robusttx](../../../packages/hfmodem/hfmodem/shrike/p4robusttx.py), [p4robustc2](../../../packages/hfmodem/hfmodem/shrike/p4robustc2.py), [p4robustrx](../../../packages/hfmodem/hfmodem/shrike/p4robustrx.py) and [p4words](../../../packages/hfmodem/hfmodem/shrike/p4words.py) define differential words, masks, spreading, robust-body geometry and placed receive rules. [p4sf8rx](../../../packages/hfmodem/hfmodem/shrike/p4sf8rx.py) defines SF8 geometry. [p4chirp](../../../packages/hfmodem/hfmodem/shrike/p4chirp.py) specifies chirp entry, shaping and CRC offset, using the marker/pulse laws in [pactor2](../../../packages/hfmodem/hfmodem/shrike/pactor2.py) and [p2rx](../../../packages/hfmodem/hfmodem/shrike/p2rx.py); [p4breakin](../../../packages/hfmodem/hfmodem/shrike/p4breakin.py) / [p4breakintx](../../../packages/hfmodem/hfmodem/shrike/p4breakintx.py) define break-in packets. |
| Header interpretation and session timing | [p4short](../../../packages/hfmodem/hfmodem/shrike/p4short.py), [p4receiver](../../../packages/hfmodem/hfmodem/shrike/p4receiver.py), [p4turn](../../../packages/hfmodem/hfmodem/shrike/p4turn.py) and [p4host](../../../packages/hfmodem/hfmodem/shrike/p4host.py) separate waveform identification, ACK eligibility, timed turn-taking, source framing and host delivery. Use their documented eligibility rules rather than treating every CRC pass as an ACK. |

## Compact tables and provenance

Runtime reconstruction is deterministic: concatenate stride-`isqrt(N)` walks,
then replay the published ordered index swaps. Validate the resulting bijection
and its canonical hash before using it as a direct gather. All thirteen tables,
including **1160**, are exactly reconstructible this way. The swap records
correct the arithmetic base; they are not the accepted-swap history of the
original random constructor. The 1160 resource has 22 corrections. It is not
presented as a recovered original seed or a documented generator-state doping
recipe.

The original CML constructor is a separate reproducibility route for twelve
tables. Seed, prior construction stages, continuous ran2 state, draw rounding,
guard windows and accepted-swap counts are all necessary. The
[derivation](pactor4-c2-derivation.md) gives those recipes and known limits.
Knowing the deployed permutation and knowing its original generation history
are distinct evidence claims.

## Reading the remaining limits

The source deliberately includes generic primitives, diagnostic decoders and
measured waveform clones alongside live packet paths. In particular,
`p4shorttx` and `p4longtx` document the provenance and semantic limits of their
measured headers. Those values enable reproduction of that waveform; they do
not fully specify every header meaning or prove peer acceptance of an arbitrary
new transmission. [P4 framing](pactor4-framing.md) and the
[implementation status](../../STATUS.md) retain those limits. This map provides
the constants used by this implementation, rather than asserting a fully closed
specification of every proprietary PACTOR-4 behavior.
