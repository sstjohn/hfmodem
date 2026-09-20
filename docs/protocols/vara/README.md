# VARA HF — functional specification

This folder specifies the VARA HF waveform, coding, framing, ARQ link layer and
host API in the detail an implementer needs. The **constant data tables** the
waveform depends on — reference sequences, interleave maps, whitener PN,
constellation LUT — are not shipped as data files: the companion implementation
produces them in code, in `hfmodem/kestrel/rx/tablegen.py`, which derives each
table under a byte-exactness gate and records how it is known (PRNG stream,
closed form, or measurement). Where this folder cites a table by a file name,
the name identifies the table; the generator is where its values come from.

Coverage is uneven, and the table below says where. Areas marked **partial** are
partial in this document, not approximate in the protocol: where a value is not
fixed here, the text says so at the point it would otherwise appear.

## Coverage

| Area | Status | Where | Notes |
|------|--------|-------|-------|
| Connect-request (CR) generator and synthesis | **Specified** | [`04 §4.2`](04-frame-block-formats.md) | Closed-form `cr_tones(called)` |
| Connect-response generator and recognizer | **Specified** | [`04 §4.2`](04-frame-block-formats.md) | |
| Connected-ack waveform and recognizer | **Structure specified; state field open** | [`04 §4.2C`](04-frame-block-formats.md) | 11 two-tone symbols on a fixed 4-symbol preamble, carrying no callsign; recognised by that preamble |
| MFSK burst physical layer (tone→freq, stride, WOLA, preambles) | **Specified** | [`04 §4.2.1`](04-frame-block-formats.md) | One grid at all three bandwidths; the payload alphabet is the bandwidth's [`04 §4.2.3`](04-frame-block-formats.md) |
| CRC-16/GENIBUS (seed and frame CRC) | **Specified** | [`03 §3.4`](03-coding.md) | poly 0x1021, init/xorout 0xFFFF, no reflection |
| Connect handshake sequence | **Specified** | [`05 §5.3`](05-arq-session-state-machine.md) | CR → response → link-setup (OFDM) → connected-ack |
| BW2300 OFDM physical layer | **Specified** | [`01 §2300`](01-physical-layer.md) | 512-sample one-hot index modulation, bins 9..24 |
| BW2300 OFDM transmit coding chain | **Specified** | [`03 §3.5.3`](03-coding.md), [`03 §3.5.4`](03-coding.md) | CRC → turbo(13,15) → whiten → interleave → LUT → twiddle → grid; constant tables per [`03 §3.5.4`](03-coding.md) |
| Handshake tone vectors (CR / response) | **Specified** | [`04 §4.2.4a`](04-frame-block-formats.md) | Numeric payload bins for K7ABC/W2XYZ/BBBB2/AAAA1/CCCC3 |
| SSID / trailing-digit callsign rule (MFSK hash input) | **Specified**; hyphenated form open | [`04 §4.2.3`](04-frame-block-formats.md) | Full ASCII string hashed; the literal `-N` SSID form is not fixed here |
| BW500 (L4) data-path coding (whitener / interleave / turbo / CRC) | **Specified** | [`03 §3.6`](03-coding.md) | |
| BW2300 OFDM receive demodulation | **Specified** | [`01 §2300`](01-physical-layer.md), [`03 §3.5.3`](03-coding.md), [`04 §4.2A`](04-frame-block-formats.md) | Recovers the caller callsign through the rec3 DATA chain |
| BW2300 caller-frame byte layout | **Specified** | [`04 §4.2A`](04-frame-block-formats.md) | 92-byte container; caller call 6-bit packed at the frame start, SSID in `body[5]`, 2-byte BE CRC |
| Host-API command and notification map | **Specified** | [`07`](07-host-api-mapping.md) | |
| ARQ state machine and timing (500 Hz) | **Specified** | [`05`](05-arq-session-state-machine.md) | Stop-and-wait; ~85 ms turnaround; gear-shift direction |
| 500 Hz DATA coding and constellation | **Specified (L4)**; partial at other levels | [`03 §3.6`](03-coding.md) | L4 SHORT via the 2-subband (1350/1650 Hz, H=512) model; levels 0–3 rate-1/3 structure noted, per-level byte layouts partial |
| ARQ thresholds, timeouts, max retries | **Partial** | [`05 §5.4–5.7`](05-arq-session-state-machine.md), [`06`](06-speed-gearshift.md) | Shift direction is specified; the trigger values are not fixed here |
| Speed levels above the base rate | **Partial** | [`01 §2300`](01-physical-layer.md), [`06`](06-speed-gearshift.md) | Derived, not validated against VARA — see the implementation note in [`01`](01-physical-layer.md) |
| 2750 Hz base level | **Specified** | [`01 §2750`](01-physical-layer.md) | BW2300 record 3's index law on a 20-bin comb — the same 395 draws, coding chain and 92-byte frame |
| 2750 Hz MFSK alphabet and handshake pair | **Specified** | [`04 §4.2.2`](04-frame-block-formats.md), [`05 §5.3.6`](05-arq-session-state-machine.md) | 84 carriers on bins 22..105, base 22 with six values of the first draw; the request rides the BW2300 alphabet in a state of its own. One loopback session |
| 2750 Hz mode above the base level | **Partial** | [`01 §2750`](01-physical-layer.md) | Occupied bandwidth and the presence of a cyclic prefix are fixed; the symbol/CP split is not, and no gear-down record is held |

## Document map

| Document | Covers |
|----------|--------|
| [`01-physical-layer.md`](01-physical-layer.md) | Waveform geometry, carrier layout, cell values, burst structure, per bandwidth mode |
| [`02-preamble-sync.md`](02-preamble-sync.md) | Burst detection, preamble structure, timing and frequency acquisition |
| [`03-coding.md`](03-coding.md) | Whitening, FEC, interleaving, CRC — the bits between frame and cell |
| [`04-frame-block-formats.md`](04-frame-block-formats.md) | Frame containers, field layouts, callsign encoding, handshake vectors |
| [`05-arq-session-state-machine.md`](05-arq-session-state-machine.md) | Link states, transitions, turnaround and session timing |
| [`06-speed-gearshift.md`](06-speed-gearshift.md) | The speed ladder and the rules for moving along it |
| [`07-host-api-mapping.md`](07-host-api-mapping.md) | The TCP host interface: commands, notifications, transport |
| [`08-compression.md`](08-compression.md) | The payload compression codec |
| [`21-data-answer-recovery.md`](21-data-answer-recovery.md) | Measured DATA ACK/NAK detection and state-query recovery after missing feedback |
| [`22-turnaround-capture.md`](22-turnaround-capture.md) | Monitored transmit-end detection without waiting for codec DC settling |
| [`23-nak-repacketization.md`](23-nak-repacketization.md) | Solicited NAK recovery with smaller frames and exact byte preservation |
| [`24-lower-record-recovery.md`](24-lower-record-recovery.md) | Lost intermediate and final answers after qualified rate changes |
| [`25-recovery-diagnostics.md`](25-recovery-diagnostics.md) | Query phases, matched replies and pending-byte ownership in recovery logs |
| [`26-data-rejection-naks.md`](26-data-rejection-naks.md) | Direct and queried DATA NACKs requiring a one-record-lower retry |
| [`27-nak-scan-deadline.md`](27-nak-scan-deadline.md) | Scheduling the next receive scan at a partially received NACK tail |
| [`28-short-final-nak-retry.md`](28-short-final-nak-retry.md) | Lower-rate retry after a missing short-final DATA frame |
| [`29-low-level-data-nacks.md`](29-low-level-data-nacks.md) | Distinguishing eight-symbol ACK/NACK tails and recovery through host levels 1–4 |
| `hfmodem/kestrel/rx/tablegen.py` (companion implementation) | The constant data tables the waveform depends on — regenerated in code, byte-exact, with how each is known |
