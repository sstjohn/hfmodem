# Documentation

hfmodem supports four modes: PACTOR, VARA, ARDOP and sabir. The guides describe
their waveforms, link behavior and implementation in hfmodem.

## Modes

| Mode | Implementation | Guide |
|---|---|---|
| PACTOR | shrike | [PACTOR-1, PACTOR-2 and PACTOR-3](viz/pactor-explained.html) |
| VARA | kestrel | [VARA](viz/vara-explained.html) |
| ARDOP | besra | [ARDOP](viz/ardop-explained.html) |
| sabir | sabir | [Experimental mode](viz/sabir-explained.html) |

[Compare the modes](viz/flock-compared.html) by channel occupancy, framing,
coding and link control. [Implementation evidence](STATUS.md) records completed
transfers, independent decodes and simulation results.

## Running hfmodem

| Subject | Reference |
|---|---|
| Station setup and operation | [Station guide](STATION.md) |
| Station files | [Configuration](CONFIG.md) |
| Shared components and mode implementations | [Architecture](../ARCHITECTURE.md) |

## Technical references

| Subject | Reference |
|---|---|
| Frame formats and protocol details | [Protocol descriptions](protocols/) |
| PACTOR generation selection | [Capability negotiation](protocols/pactor/pactor-capability.md) |
| Sources and strength of evidence | [Evidence tiers](protocols/EVIDENCE.md) |

The comparison, PACTOR, VARA and ARDOP figures are checked against their source
data by `docs/viz/build.py --check`. Sabir illustrations and recorded simulation
results are maintained separately. All five guides can be opened offline.
