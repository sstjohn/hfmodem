# Robust Packet station attachment

The station name is `rpr`; the waveform package is `brubru`. The operator controls
the station and enables transmission in the station configuration. The adapter
receives card-rate float samples, retains decimator and acquisition history, and
passes validated AX.25 frames without FCS to its KISS client. [T: software loopback]

Enable `[protocols.rpr]` with `host = "kiss"` and a TCP `port`. The listener binds
to `127.0.0.1`, accepts one client, and reports its actual bound port. The client
sends ordinary KISS frames immediately. A connection opening with an HTTP request
is closed. A station without a rig and arbiter reports that its host cannot
transmit. Start it with `hfmodem station FILE` or `python -m hfmodem station FILE`.

The optional `[protocols.rpr.options]` table accepts:

| Key | Default | Meaning |
|---|---|---|
| `limiter` | `true` | Waveform limiter |
| `level` | `0` | Transmit level, 0 or 1 |
| `txdelay` | `100` | Total PTT-to-data time in milliseconds |
| `persistence` | `63` | Inclusive threshold for a uniform byte draw |
| `slot_ms` | `100` | Contention slot in milliseconds |
| `tx_events` | absent | Operator-selected transmit transcript path |

KISS TXDELAY and SlotTime use 10 ms units. Persistence uses the byte as sent.
FullDuplex and SetHardware are accepted without changing station behavior. Other
KISS ports are ignored. Frames over 2048 bytes and malformed escape sequences are
dropped until the next frame delimiter. The outbound queue holds 32 frames; it
drops the oldest unsent frame when full. A send that cannot finish within two
seconds drops the client. [T: host transport tests]

The inbound queue holds at most 16 outstanding requests. One worker submits them
in order. Disconnect drops requests still queued; a request already handed to the
worker completes. Station shutdown cancels waiting requests, disables the protocol,
and waits for a committed burst. Each outcome carries its request identifier,
attempt count, and deferral count. With `tx_events` set, the adapter appends the
existing `tx_event` transcript lines after every outcome; otherwise it retains no
journal. [T: station tests]

DCD compares a 480 Hz band with its flanking guards and a qualified noise floor.
Startup requires two seconds of qualifying windows. Detectable in-band occupancy
is excluded from the floor. Broadband occupancy already present at startup,
raising the band and guards equally, qualifies as the floor: these measurements
cannot distinguish it from a noisy quiet channel. A held frame keeps DCD busy
through fades. Published receive faults invalidate eligibility immediately and
require one second of contiguous audio after the affected block. [T: access tests]

PTT assertion supplies the card index used to schedule TXDELAY. Audio is published
only if its scheduled end and unkey margin fit the keyed deadline. Callback health
and actual buffer consumption are checked separately, including the silent lead
and playback tail. The stall allowance includes reported output latency and
measured delivery jitter. Resource close waits for admitted operations and rejects later
calls. A worker or resource that cannot drain triggers fatal shutdown: a five-second
deadman, an attempted unkey, an explicit confirmation outcome, and nonzero process
termination. A worker arms the kernel timer before signaling the main thread.
Once installed, the timer handler needs interpreter progress; software tests establish no physical unkey guarantee. [T: fake resources
and subprocess tests]

Connected-mode ownership and the `mail` host are reserved for subsequent work.
Independent receiver interoperability and sensitivity need their own bench
measurements; software loopback does not establish them.
