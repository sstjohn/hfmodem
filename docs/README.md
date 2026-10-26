<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Documentation

Operator guides, protocol references and recorded results. VarAC-compatible
chat and supervised voice/CW contacts are included in 0.3.0.

Project contact: [hfmodem@ssj.lol](mailto:hfmodem@ssj.lol).

## Modes

| Mode | Release status | Implementation | Guide |
|---|---|---|---|
| PACTOR | Alpha (1–3); experimental (4) | shrike | [PACTOR guide](viz/pactor-explained.html), [PACTOR-4 protocol and PRNG](viz/pactor4-explained.html), [PACTOR-4 limits](STATUS.md#pactor-4) |
| VARA | Alpha | kestrel | [VARA](viz/vara-explained.html) |
| ARDOP | Alpha | besra | [ARDOP](viz/ardop-explained.html) |
| Robust Packet | Experimental | brubru | [Illustrated guide](viz/robust-packet-explained.html), [waveform](protocols/robust-packet/waveform.md), [framing](protocols/robust-packet/framing.md), [link boundary](implementation/robust-packet/link.md), [station](implementation/robust-packet/station.md) |
| RTTY, SITOR-B and NAVTEX | Experimental | lanner | [Illustrated text-mode guide](viz/rtty-explained.html), [implementation limits](implementation/sitor/README.md) |
| MIL-STD-188-110, STANAG waveforms, 2G ALE and HF selcall | Experimental | bateleur | [Illustrated STANAG and military-waveform guide](viz/stanag-explained.html), [host interfaces](implementation/bateleur/README.md), [ALE and selcall console](console-hflink.md) |
| sabir | Experimental | sabir | [Experimental mode](viz/sabir-explained.html) |
| Voice and CW | Experimental | macagua, merlin | [Supervised contacts](CONSOLE.md#voice-and-cw) |

[Compare the modes](viz/flock-compared.html) by channel occupancy, framing,
coding and link control. [Recorded evidence](../evidence/README.md) and [implementation evidence](STATUS.md) record completed
transfers, independent decodes and simulation results.

## Running hfmodem

Start with the [Winlink quickstart](CONSOLE.md#install-and-receive-winlink-mail)
or [VARA chat quickstart](CONSOLE.md#vara-chat-quickstart)
for an FTDX10 or FT-891 on macOS or Linux.

| Subject | Reference |
|---|---|
| Console setup, operator data and logbook | [Operator guide](CONSOLE.md) |
| Station setup and operation | [Station setup](STATION.md), [service and ownership](README-station.md) |
| Separate CW and digital callsigns | [Identification options](IDENTIFICATION.md) |
| Kestrel chat, passive QSO following and VMail | [Quickstart and commands](CONSOLE.md#vara-chat-quickstart) |
| Supervised voice and CW | [Contact commands](CONSOLE.md#voice-and-cw) |
| VARA transmit calibration | [Calibration](CONSOLE.md#vara-transmit-calibration) |
| Single-mode receive monitoring | [Console monitor](console-monitor.md) |
| Automatic ALE, calls and native text; selcall and paging | [HFLINK console](console-hflink.md) |
| Application and modem hosts | [Host compatibility](HOST-COMPATIBILITY.md) |
| Station files | [Configuration](CONFIG.md) |
| Shared components and mode implementations | [Architecture](../ARCHITECTURE.md) |

## Technical references

| Subject | Reference |
|---|---|
| PACTOR-4 framing, coding and PRNG | [Protocol guide](viz/pactor4-explained.html), [packet framing](protocols/pactor/pactor4-framing.md), [PRNG](protocols/pactor/pactor4-prng.md), [C2 recipes](protocols/pactor/pactor4-c2-derivation.md) |
| PACTOR generation selection | [Capability negotiation](protocols/pactor/pactor-capability.md) |
| PACTOR implementation inputs | [Constants, coding and permutation map](protocols/pactor/implementation-constants.md) |
| VARA implementation inputs | [Constants and high-speed waveform map](protocols/vara/implementation-constants.md) |
| Sources and strength of evidence | [Evidence tiers](protocols/EVIDENCE.md) |

The comparison, PACTOR, VARA and ARDOP figures are checked against their source
data by `docs/viz/tools/build.py --check`. Sabir illustrations and recorded simulation
results are maintained separately. Robust Packet, STANAG and text-mode guides use implementation-derived schematic data checked by the site tests. The PACTOR-4 guide includes offline waveform renders and recovered permutation recipes. All nine illustrated guides can be opened offline.

Regenerating the comparison's recorded figures also requires the `flac`
command-line decoder and the retained recordings under `site/examples/` and
`site/recordings/`.
Viewing the built figures does not require a decoder or an audio download.
