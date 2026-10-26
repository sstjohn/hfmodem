<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Robust Packet link boundary

The frame layer supplies an AX.25 library and an incremental KISS codec.
`ax25.frame` encodes and validates addresses, control, PID and FCS;
`ax25.kiss` frames bytes and returns port, command and payload separately.
Both libraries are independent of the station and of individual protocols.

The modem transports the supplied frame unchanged, including non-UI frames
used as offline vectors. The KISS client owns link control when the station
selects the KISS host. Tracker and AGWPE hosts provide connected AX.25 sessions,
acknowledgements and retries; the attached application provides mail transport.

KISS payloads exclude FCS. The parser handles escaped FEND and FESC across
arbitrary input chunks, bounds payloads at 2048 bytes, and resynchronizes at
FEND after malformed escapes or overflow. The [station attachment](station.md)
applies commands and maintains exclusive client ownership.
[T: 1 implementation, synthetic KISS streams, 2026-09-30]

The offline entry points are `python -m hfmodem.brubru.rx FILE...` and
`python -m hfmodem.brubru.tx SOURCE DESTINATION TEXT OUTPUT.wav`.
The transmitter accepts `--level 0|1` and `--no-limiter`; its output is a
9600 Hz mono WAV and it refuses to replace an existing output file.
