# Sabir host interface

Sabir uses one framed CBOR connection for commands, payloads and events. The
reference TCP server binds loopback port 8400 by default and serves one
application at a time. `host/run_server.py` runs two modems over simulated
audio on ports 8400 and 8410. StationAir can bind the modem to station audio;
that integration has not been validated by a two-station Sabir radio session.

The implementation is in `host/messages.py`, `host/hostlink.py` and
`host/modem_core.py`. `host/client.py` is a threaded application client.
`tests/sabir/test_hostapi.py` and `test_m4.py` cover the framing, telemetry and
host-to-host simulated exchanges.

## 1. Framing and attachment

Each frame is a four-byte big-endian body length followed by one CBOR map.
The maximum body is 16 MiB. An oversized prefix receives MALFORMED and closes
the connection without allocating its body. Other malformed frames receive
MALFORMED and leave the established connection open.

Top-level keys are unsigned integers; key 0 is the unsigned message type `m`.
The CBOR subset comprises 64-bit integers, byte/text strings, arrays, maps,
booleans, null and float64. Indefinite lengths, tags and shorter floating-point
encodings are unsupported. Map keys must be integers or text and cannot repeat.
The encoder uses shortest integer arguments and sorts keys by their encoded
bytes. The decoder accepts nonminimal integer arguments; no canonical byte
image is needed for host framing.

The first message must be `Hello{m:0, proto:"1.0"}`. The modem answers with
`Hello{proto, features, modem, profiles, identity_required:true}`. `features`
is `['data-profiles', 'objects']`; `profiles` is `[1,2]`. The modem sends
INCOMPATIBLE and closes on a protocol major mismatch. Invalid initial frames
receive MALFORMED and close. Minor versions may add optional fields and events.
Unknown message types and unknown integer field keys are ignored.

Commands execute in connection order on the modem's air thread. Events may
arrive while a command is pending. There is no periodic keepalive, application
flow control or radio binding command. Disconnecting the host connection aborts
its modem session. TCP buffers supply backpressure.

## 2. Commands

Notation: `u` unsigned integer, `s` text, `b` bytes, `f` float; `?` optional.
`ref:u?` correlates command errors and may accompany any command.

| m | Command | Fields | Behavior |
|---|---|---|---|
| 0 | Hello | `proto:s`, `features:[s]?`, `client:s?` | Attach handshake |
| 1 | SetIdentity | `station_id:s` | Set a 1–12 byte printable ASCII identifier, uppercased |
| 2 | SetProfile | `profile:u` | Select the presence-beacon profile label: 1 Amateur, 2 Unrestricted |
| 3 | Listen | `on:bool`, `station_id:s?` | Enable/disable incoming connects; optional identity replaces the modem's listening identity |
| 4 | Connect | `peer_id:s`, `station_id:s?` | Connect, using the optional local identity for this session |
| 5 | Send | `data:b`, `id:u?`, `deflate:bool?` | Queue one integrity-protected record |
| 6 | Disconnect | — | Drain queued session data and close |
| 7 | Abort | — | Abandon queued transmissions and close |
| 8 | Subscribe | `events:[u]` | Replace the optional-event subscription |
| 9 | Configure | DATA options below | Configure an idle modem |
| 10 | Beacon | `addressee:u?` | Send a presence beacon; default addressee 0 |
| 11 | SendObject | Object options below | Send a connectionless object |

Listen, Connect, Beacon and SendObject require SetIdentity first. SetProfile
is a beacon metadata label; it does not enforce bandwidth, identification or
regulatory policy. The station's radio configuration controls that policy.
Aliases, per-Connect profiles, deadlines, nonzero streams and nonzero priority
are unsupported and rejected. `stream:0` and `priority:0` have ordinary Send
semantics. `Subscribe.stats_period` is ignored; telemetry has no periodic timer.

Configure accepts `data_profile:s?`, `receive_profiles:[s]?`,
`feedback_iters:u?`, `impulse_blank:f?`, `bandwidth_hz:u?` and
`inactivity_timeout_s:f?`. The option set is
validated before applying any change and only while DISCONNECTED or LISTENING.
`receive_profiles` contains at most eight distinct extended profile names;
`data_profile` may prefer any implemented DATA profile, including baseline
profiles. The extended receive profiles are `doppler`,
`sparse34`, `narrow`, `narrow2`, `narrow4` and `wide256`. `feedback_iters` is
0–3, `impulse_blank` 0–20, and `bandwidth_hz` is 500, 1500, 2300 or 2750.
The inactivity timeout is finite and positive (default 180 seconds); validated
peer traffic renews it and an active transmission or validated receive body
is protected until completion. Radio and PTT configuration submaps are rejected. The profile registry and
waveform widths are specified in [SPEC.md](SPEC.md).

SendObject accepts `data:b`, `destination:s?`, `service:u?`,
`data_profile:s?`, `parity:bool?`, `repeats:u?` and `message_id:b?`.
It requires an idle link with no previous object draining. Defaults are
all-call destination `*`, service 0, profile `workhorse`, parity true and one
repeat. The profile must fit the configured width. Repeats are 1–8.
The identifier is 16 bytes, randomly generated when omitted. An object with
multiple data fragments may carry one XOR repair fragment.

## 3. Events

| m | Event | Fields |
|---|---|---|
| 32 | StateChanged | `state:u`, `peer_id:s?`, `reason:u?` |
| 33 | CapabilitiesNegotiated | `peer_id:s`, `peer_capabilities:u`, `usable:{…}`, `peer_profiles:[u]` |
| 34 | LinkStats | `gear:s`, `rung:u`, `snr3k_db:f?`, `group_snr_db:[f?]`, `control_tier:u`, `throughput_bps:f`, `queue_bytes:u`, `harq_rounds:u`, `rebuilds:u`, `compression_ratio:f?` |
| 35 | DataReceived | `data:b`, `stream:0` |
| 36 | SendProgress | `id:u?`, `message_id:b?`, `sent:u?`, `total:u?`, `delivered:bool`, `deflated:bool?` |
| 37 | IdSent | `station_id:s`, `t:f` |
| 38 | PeerObserved | `peer_id:s`, `profile:u?`, `capabilities:{…}` |
| 39 | PhysicalState | `ptt:bool?`, `busy:bool?` |
| 40 | Error | `code:u`, `ref:u?` |
| 41 | ObjectReceived | `peer_id:s`, `destination:s`, `service:u`, `message_id:b`, `data:b`, `t:f` |

LinkStats, IdSent, PeerObserved and PhysicalState are subscription gated;
all other events are always delivered. State codes are 0 DISCONNECTED,
1 LISTENING, 2 CONNECTING, 3 CONNECTED and 4 DISCONNECTING. A listening
responder returns to LISTENING after its disconnect event.

Disconnect reasons are 1 remote, 2 local, 3 link failed and 4 negotiation
refused; the field is omitted when unset. Error codes are 1 INCOMPATIBLE,
2 BAD_STATE, 3 NO_IDENTITY and 4 MALFORMED. The reference currently reports
invalid command fields and rejected configurations as MALFORMED.

CapabilitiesNegotiated reports the completed connection's capability
intersection. `peer_profiles` lists the exact supported DATA registry IDs;
`peer_capabilities` is the received bitmap. `usable` has
boolean `fastctl`, `pback`, `loading` and `deflate` entries. Mandatory record
integrity does not have a negotiated capability bit.

LinkStats is emitted when the gear, control tier, queue depth or rebuild count
changes. SNR or throughput changes alone do not trigger an event. The group
SNR vector has eight entries, with null for groups not yet measured.
`control_tier` is 0 for floor or 1 for fast control. `queue_bytes` estimates
unacknowledged application bytes by mapping committed record bytes across the
submitted records; a partially acknowledged compressed record is interpolated.
`throughput_bps` is received application bits divided by elapsed time since
connection. `compression_ratio` includes record headers and integrity trailers,
so it can exceed one. Neither metric is a prediction of future throughput.

PeerObserved reports a decoded presence beacon. No automatic modem switching
or gateway discovery policy is implemented by this interface. PhysicalState
is modem telemetry; its presence does not prove an RF transmission.

## 4. Record and object delivery

Send queues one record even when its payload is empty. With `id`, acceptance
emits SendProgress with `sent=total=payload length`, `delivered:false`, and
`deflated` reflecting the encoding actually selected. `sent` here means queued,
not transmitted. DEFLATE is used only when negotiated and when it shrinks the
body. Delivery emits `delivered:true` after every record byte is acknowledged;
a session failure leaves unacknowledged messages without that event. Host
applications must not infer remote application processing from a link ACK.

DataReceived carries one complete, verified and decompressed record. Concatenate
its payloads to obtain an ordered byte stream. There is one stream and no
priority scheduler or independent stream multiplexing.

The record is `[flags:1][length:3][encoded body][SHA-256:32]`.
Flags are 0x10 raw or 0x11 raw DEFLATE. Length counts body plus digest; the
digest covers header and encoded body. Verification precedes decompression and
delivery. Invalid flags, invalid compression and integrity failures abort the
session. Decoded payloads are limited to 8 MiB. SHA-256 detects corruption;
it does not authenticate a peer.

SendObject acceptance echoes its message_id with `delivered:false`. It never
reports remote delivery because objects have no ACK. ObjectReceived carries
a complete verified object addressed to the receiver or all-call. `t` is the
receiver's local clock at completion, not necessarily UTC.

## 5. Field keys

Assigned keys identify fields; a key has meaning only in its message schema.
Unused assignments do not imply support for a feature. Future changes may add
optional keys; changing an existing field's interpretation requires a major
host version change.

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| 0 `m` | 1 `proto` | 2 `features` | 3 `client` | 4 `modem` | 5 `profiles` | 6 `identity_required` | 7 `station_id` | 8 `aliases` | 9 `profile` |
| 10 `on` | 11 `peer_id` | 12 `deadline` | 13 `ref` | 14 `data` | 15 `stream` | 16 `priority` | 17 `deflate` | 18 `id` | 19 `events` |
| 20 `stats_period` | 21 `radio` | 22 `ptt` | 23 `state` | 24 `reason` | 25 `peer_capabilities` | 26 `usable` | 27 unused | 28 `peer_profiles` | 29 `gear` |
| 30 `rung` | 31 `snr3k_db` | 32 `group_snr_db` | 33 `control_tier` | 34 `throughput_bps` | 35 `queue_bytes` | 36 `eta_s` | 37 `harq_rounds` | 38 `rebuilds` | 39 `compression_ratio` |
| 40 `sent` | 41 `total` | 42 `delivered` | 43 `deflated` | 44 `t` | 45 `capabilities` | 46 `busy` | 47 `code` | 48 `detail` | 49 `addressee` |
| 50 `data_profile` | 51 `receive_profiles` | 52 `feedback_iters` | 53 `impulse_blank` | 54 `bandwidth_hz` | 55 `destination` | 56 `service` | 57 `message_id` | 58 `parity` | 59 `repeats` |

Key 60 is `inactivity_timeout_s`.
