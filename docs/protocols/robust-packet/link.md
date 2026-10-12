# Robust Packet link boundary

The frame layer supplies an AX.25 library and an incremental KISS codec.
`ax25.frame` encodes and validates addresses, control, PID and FCS;
`ax25.kiss` frames bytes and returns port, command and payload separately.
Both libraries are independent of the station and of individual protocols.

The modem transports the supplied frame unchanged, including non-UI frames
used as offline vectors. Connected-mode state, acknowledgements, retries and
mail transport are deferred. The KISS client owns everything above
the frame when the station selects the KISS host.

KISS payloads exclude FCS. The parser handles escaped FEND and FESC across
arbitrary input chunks, bounds payloads at 2048 bytes, and resynchronizes at
FEND after malformed escapes or overflow. The [station attachment](station.md)
applies commands and maintains exclusive client ownership.
[T: 1 implementation, synthetic KISS streams, 2026-09-30]

The offline entry points are `python -m hfmodem.brubru.rx FILE...` and
`python -m hfmodem.brubru.tx SOURCE DESTINATION TEXT OUTPUT.wav`.
The transmitter accepts `--level 0|1` and `--no-limiter`; its output is a
9600 Hz mono WAV and it refuses to replace an existing output file.
