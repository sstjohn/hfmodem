# 07 — Host-API mapping (TCP command / data interface)

VARA presents its host interface as two TCP sockets: a **command channel** on port
8300 carrying settings, session control and asynchronous notifications, and a
**data channel** on port 8301 carrying payload bytes. A host application — VarAC,
Pat, Winlink Express — drives the modem entirely through these two sockets. This
section specifies the framing, the command set, the notifications, and how each
maps onto the ARQ session states of [`05`](05-arq-session-state-machine.md).

The implementation of this interface in this repository is
`hfmodem/kestrel/host/protocol.py` (message framing and command parsing) and
`hfmodem/kestrel/host/server.py` (the two sockets).

## 7.1 Transport

| Attribute                  | Value |
|----------------------------|-------|
| Command port               | **8300** (TCP) |
| Data port                  | **8301** (TCP) |
| Command framing            | ASCII text; each message terminated by a single `\r` (0x0D, CR). Several messages may share one TCP segment → split on `\r` |
| Data framing               | **Raw, un-framed bytes** — no length prefix, no delimiter. Bytes written are transmitted; bytes read are what the far end sent. Flow control is out-of-band via `BUFFER` on 8300 |
| Encoding                   | Command channel ASCII; data channel binary |

## 7.2 Commands (client → modem)

| Command | Arguments | Semantics | Expected response |
|---------|-----------|-----------|-------------------|
| `MYCALL` | `c1 [c2..c5]` | Set up to 5 station callsigns | `OK` |
| `LISTEN` | `ON` / `OFF` | Arm/disarm inbound connect (mid-session `OFF` ⇒ disconnect) | `OK` |
| `CONNECT` | `src dst` (HF/SAT); FM adds `via d1 d2` | Start an ARQ session to `dst` | `OK`; then `PENDING`/`CONNECTED`/`DISCONNECTED` async |
| `DISCONNECT` | — | Graceful close (flushes TX buffer first) | `OK`; then `DISCONNECTED` |
| `ABORT` | — | Immediate "dirty" disconnect | `OK` |
| `BW500` / `BW2300` / `BW2750` | — | Select HF bandwidth (narrow / standard / tactical) | `OK` |
| `COMPRESSION` | `OFF` / `TEXT` / `FILES` | Payload compression mode | `OK` |
| `CHAT` | `ON` / `OFF` | Keyboard-chat timing; `CHAT ON` also implies `LISTEN ON` and enables `SN` reports | `OK` |
| `CQFRAME` | `src bw` | Send a CQ frame (HF) | `OK`; peers may emit `CQFRAME …` |
| `VERSION` | — | Query modem version; a later addition, outside the base command set | `VERSION x` (or `WRONG` on old builds) |

Every command is echoed by exactly one synchronous `OK` (accepted) or `WRONG` (bad
command). Bandwidth/compression/`MYCALL`/`LISTEN` are settings; `CONNECT`/`DISCONNECT`/
`ABORT`/`CQFRAME` drive the ARQ state machine ([`05`](05-arq-session-state-machine.md)).

## 7.3 Async notifications (modem → client)

| Notification | Meaning | When emitted |
|--------------|---------|--------------|
| `OK` / `WRONG` | command accepted / rejected | synchronously after each command |
| `CONNECTED src dst [bw]` | ARQ link established (`bw` present on HF/FM, absent on SAT) | on handshake completion |
| `DISCONNECTED` | session closed by either end | on graceful/abort close |
| `PTT ON` / `PTT OFF` | key/unkey the radio (host drives PTT) | around every TX burst |
| `BUFFER n` | bytes remaining in the TX queue | on enqueue and on each ack-drain |
| `PENDING` / `CANCELPENDING` | inbound connect detected / aborted (pause scanning) | on CR detect / give-up |
| `BUSY ON` / `BUSY OFF` | channel busy-detector state | on busy-detector edges |
| `REGISTERED call` / `LINK REGISTERED` / `LINK UNREGISTERED` | registration status (gates full speed) | at connect |
| `BITRATE (N) x bps TX` / `… RX` | current speed level of my next over / the frame I decoded | per over |
| `IAMALIVE` | host-facing keepalive | ~every 60 s while connected |
| `SN xx` | signal-report (only with `CHAT ON`) | on decode |
| `CQFRAME src bw` | a CQ frame was decoded | on CQ decode |
| `ENCRYPTION …` / `ENCRYPTED/UNENCRYPTED LINK` | link encryption status (newer builds) | at connect |

## 7.4 Data-path semantics

| Attribute                              | Value |
|----------------------------------------|-------|
| Buffering / backpressure               | Host writes raw bytes to port 8301; the modem queues them and reports the queue depth as `BUFFER n` on 8300, decrementing per ACKed block. The host throttles on `BUFFER` |
| `BUFFER` units                          | **raw host bytes still queued** (independent of `COMPRESSION`); each ACKed block clears 43 B at level 4, 34 B at level 3 |
| **The data port is not opaque** | A 4.9.0 reads what the host writes: `FQ\r` handed to the data port ends the link by itself, `Disconnecting by FQ` in the modem's own log, with nothing keyed and no `DISCONNECT` command sent. Measured 2026-08-26, and it cost a bench a second handover before it was found. A B2F layer that means `FQ` for its peer has to answer the far end's `FQ` with something else — the benches here answer `;OK\r` |
| Relationship of data port to ARQ state | Data port bytes are only transported over the air while `CONNECTED`. Bytes the host writes **before** the link comes up are buffered against the TX queue (and reported via `BUFFER n`), then flushed into the session the moment `CONNECTED` fires — they are **not** dropped. A graceful `DISCONNECT` flushes the queue first; `ABORT` discards it. In this repository, `hfmodem/kestrel/host/modem_core.py`'s `LoopbackModem` is where that pre-connect buffering and flush-on-`CONNECT` behaviour lives |

## 7.5 Mapping to internal state

Ties host-API events to the state machine in [`05`](05-arq-session-state-machine.md).

| Host-API event | Internal transition ([05](05-arq-session-state-machine.md)) |
|----------------|-------------------------------------------------------------|
| cmd `LISTEN ON` | DISCONNECTED → LISTENING |
| cmd `CONNECT src dst` | DISCONNECTED → CONNECTING (keys CR burst; `BUSY ON`) |
| notif `PENDING` | LISTENING → CONNECTING (CR decoded) |
| notif `CONNECTED …` | CONNECTING → CONNECTED |
| notif `BUFFER n` (n↓) | CONNECTED: a block was ACKed |
| cmd `DISCONNECT` | CONNECTED → DISCONNECTING |
| notif `DISCONNECTED` | DISCONNECTING → DISCONNECTED (`BUSY OFF`) |

## 7.6 Behaviour not fixed here

- `VERSION`, `ENCRYPTION*`, `SN`, `CHAT`, `CQFRAME` and `ABORT` are named and their
  syntax given above, but this document does not fix their full behaviour: each rests
  on a single description, and the edge cases are not specified here.
- The `BITRATE (N)` level → (modulation, FEC-rate) mapping is a physical/coding fact
  and belongs to [`01`](01-physical-layer.md)/[`03`](03-coding.md), not to the host API.
