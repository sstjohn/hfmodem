# Documentation

The guides cover the data, linking and receive-only modes in hfmodem: their
waveforms, link behavior, monitoring and implementation evidence.

Project contact: [hfmodem@ssj.lol](mailto:hfmodem@ssj.lol).

## Modes

| Mode | Release status | Implementation | Guide |
|---|---|---|---|
| PACTOR | Alpha | shrike | [PACTOR-1, PACTOR-2 and PACTOR-3](viz/pactor-explained.html) |
| VARA | Alpha | kestrel | [VARA](viz/vara-explained.html) |
| ARDOP | Alpha | besra | [ARDOP](viz/ardop-explained.html) |
| Robust Packet | Experimental | brubru | [Illustrated guide](viz/robust-packet-explained.html), [waveform](protocols/robust-packet/waveform.md), [framing](protocols/robust-packet/framing.md), [link boundary](protocols/robust-packet/link.md), [station](protocols/robust-packet/station.md) |
| RTTY, SITOR-B and NAVTEX | Experimental | lanner | [Illustrated text-mode guide](viz/rtty-explained.html), [implementation limits](protocols/sitor/README.md) |
| MIL-STD-188-110, STANAG waveforms, 2G ALE and HF selcall | Experimental | bateleur | [Illustrated STANAG and military-waveform guide](viz/stanag-explained.html), [host interfaces](protocols/bateleur/README.md), [ALE and selcall console](console-hflink.md) |
| sabir | Experimental | sabir | [Experimental mode](viz/sabir-explained.html) |

[Compare the modes](viz/flock-compared.html) by channel occupancy, framing,
coding and link control. [Implementation evidence](STATUS.md) records completed
transfers, independent decodes and simulation results.

## Running hfmodem

Start with the [Winlink quickstart](CONSOLE.md#install-and-receive-winlink-mail)
for a plugged-in FTDX10 or FT-891 on macOS or Linux.

| Subject | Reference |
|---|---|
| Console setup, operator data and logbook | [Operator guide](CONSOLE.md) |
| Station setup and operation | [Station guide](STATION.md) |
| Separate CW and digital callsigns | [Identification options](IDENTIFICATION.md) |
| Single-mode receive monitoring | [Console monitor](console-monitor.md) |
| Automatic ALE, calls and native text; selcall and paging | [HFLINK console](console-hflink.md) |
| Station files | [Configuration](CONFIG.md) |
| Shared components and mode implementations | [Architecture](../ARCHITECTURE.md) |

## Technical references

| Subject | Reference |
|---|---|
| Frame formats and protocol details | [Evidence and protocol sources](protocols/EVIDENCE.md) |
| PACTOR generation selection | [Capability negotiation](protocols/pactor/pactor-capability.md) |
| PACTOR implementation inputs | [Constants, coding and permutation map](protocols/pactor/implementation-constants.md) |
| VARA implementation inputs | [Constants and high-speed waveform map](protocols/vara/implementation-constants.md) |
| Sources and strength of evidence | [Evidence tiers](protocols/EVIDENCE.md) |

The comparison, PACTOR, VARA and ARDOP figures are checked against their source
data by `docs/viz/build.py --check`. Sabir illustrations and recorded simulation
results are maintained separately. Robust Packet, STANAG and text-mode guides use implementation-derived schematic data checked by the site tests. All eight illustrated pages can be opened offline.

Regenerating the comparison's recorded figures also requires the `flac`
command-line decoder and the retained recordings under `site/examples/` and
`site/recordings/`.
Viewing the built figures does not require a decoder or an audio download.
