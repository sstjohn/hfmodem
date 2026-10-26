<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# 07 — Host-API mapping (TCP command / data interface)

VARA presents its host interface as two TCP sockets: a **command channel** on port
8300 carrying settings, session control and asynchronous notifications, and a
**data channel** on port 8301 carrying payload bytes. A host application — Pat,
Winlink Express — drives the modem entirely through these two sockets. This
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
| `LISTEN` | `ON` / `CQ` / `OFF` | Arm/disarm inbound connect; `CQ` also enables CQ notifications | `OK` |
| `CONNECT` | `src dst` (HF/SAT); FM adds `via d1 d2` | Start an ARQ session to `dst` | `OK`; then `PENDING`/`CONNECTED`/`DISCONNECTED` async |
| `DISCONNECT` | — | Graceful close (flushes TX buffer first) | `OK`; then `DISCONNECTED` |
| `ABORT` | — | Immediate "dirty" disconnect; an accepted CQ finishes | `OK` |
| `BW500` / `BW2300` / `BW2750` | — | Select HF bandwidth (narrow / standard / tactical) | `OK` |
| `COMPRESSION` | `OFF` / `TEXT` / `FILES` / `ON` | Payload compression; legacy `ON` selects TEXT | `OK` |
| `CHAT` | `ON` / `OFF` | Keep an idle, answering link open; `ON` enables listening and CQ reports; `OFF` preserves listening and suppresses CQ reports | `OK` |
| `CLEANTXBUFFER` | — | Clear queued data only when not holding the sending turn | `CLEANTXBUFFER BUFFEREMPTY`, `CLEANTXBUFFER FAILED ` (trailing space), or `CLEANTXBUFFER OK` |
| `CQFRAME` | `src bw` | Send a CQ frame (HF) | `OK`; peers may emit `CQFRAME …` |
| `VERSION` | — | Query modem version; a later addition, outside the base command set | `VERSION x` (or `WRONG` on old builds) |

Most commands reply with `OK` (accepted) or `WRONG` (bad command);
`VERSION` and `CLEANTXBUFFER` have the specific replies above. Bandwidth/compression/`MYCALL`/`LISTEN` are settings; `CONNECT`/`DISCONNECT`/
`ABORT`/`CQFRAME` drive the ARQ state machine ([`05`](05-arq-session-state-machine.md)).

The replacement supports CHAT timing and CQ transmission/reception. CQ reports
are enabled by `LISTEN CQ` or `CHAT ON`. `LISTEN ON`,
`LISTEN OFF` and `CHAT OFF` suppress the entire CQ notification sequence.
`CQFRAME` accepts exactly one space between its three fields. Calls are
uppercased, allow seven base characters and optional SSIDs 1–15, and may be
all digits. Explicit `-0`, suffixes above 15, `/P` and underscores return
`WRONG`. Arguments 500, 2300 and 2750 select codes 1, 2 and 3 independently
of the configured modem bandwidth; other bandwidth arguments return `OK`
without transmitting. A pending CQ or pending/active link returns `WRONG`.
ABORT leaves an accepted CQ intact; PTT OFF follows actual audio completion.
Command edges are pinned to B1/B2/B4 in `cq-fix1-host.json` and the CQ replay
ledger; `cq-fix2-modes.json` records the later listen/chat notification checks. Stock's SSID overflow and early PTT OFF are deliberately not reproduced.
The owner-serviced heartbeat, receive BUSY events and registration mapping
are implemented by the live producers.

Kestrel also provides `CQFRAMEFORCE`, `BEACONFORCE` and bounded `BROADCAST`
requests. These are extensions; an admission reply is not a transmit result.
Its BUFFER count uses original bytes for OFF/TEXT and container bytes for FILES.

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
| High speed capability (per callsign or link), or speed limited link status | high speed capability status (gates full speed) | at connect |
| `BITRATE (N)  x bps TX` / `… RX` | current speed level of my next over / the frame I decoded (two spaces after `)`) | RX immediately after `SN` for a decoded data over, once even for two frames; CQ uses level 1 at 18 bps; no RX bitrate for connect or short controls |
| `IAMALIVE` | host-facing keepalive | ~every 60 s while connected |
| `SN x.x` | signal report in dB, independent of chat mode | once per received data over, connect frame or CQ frame; no short-control reports |
| `CQFRAME src bw` | a CQ frame was decoded | with CQ reports enabled, after `BITRATE (1)  18 bps RX` and `SN` |
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

BW500 levels 1–4 use the spectral estimator fitted to the P0 noise ladder and
P2 gate-2 paired level-4 readings (2026-10-18). Its clean ceiling is 21.4 dB.
The common low-level estimator is not separately validated on paired L1–3 data.
The P0-only fit missed the held-out gate by up to 3.0 dB; the final pooled fit
has no independent gate holdout (only one observation at each noisy gate setting).

For higher levels, only the host `SN` is mapped; the decoder quality used for
burst detection and rate selection is unchanged. The paired scalar measurements
and transcript times are pinned in `tests/kestrel/fixtures/chat-sn-pairs.json`.

| BW500 level | Decoder raw dB observed | Host correction | Maximum paired error |
|---|---|---|---|
| 5 | 34.63–34.64 | −8.485 dB | 0.95 dB |
| 6 | 39.40–39.64 | −13.1685 dB | 1.63 dB |
| 8 | 61.83–63.01 | −32.34 dB | 0.31 dB |
| 10 | 63.01 | −34.26 dB | 0.65 dB |
| 11 | 61.88–63.01 | −34.6345 dB | 0.72 dB |
| 13 | 55.63–59.55 | 27.7 dB plateau | 1.70 dB |

These are sparse, clean-cable calibrations, not noisy-channel validation.
Beyond each observed range the nearest endpoint's additive correction is used,
except L13 stays at 27.7 dB above raw 59.546675. Additive extrapolation is
unvalidated. No stock paired SN exists for the noisy level-6 overs: the gate's 39.4–39.9/22.8/12.7/4.1 values are kestrel's own
reports, not stock reference values.

BW500 L7, L9 and L12 now carry one additive point each, so a raw metric near
the OFDM ceiling no longer reaches the host as 58-63 dB (bench 3 gate 3 reported
58.2 at L12 and 62.8 at L9). L7 is 30.7 at raw 63.01 and L9 29.7 at raw 63.01,
the stock-to-stock medians on these cables; L12 is 28.7 at raw 58.21, its three
paired readings. L12 is not monotone in the raw metric (stock 31.4 dB at raw
56.7858, 26.5 dB at raw 58.3450), so this point misses those by up to 4.1 dB,
inside stock's own 20.3-31.5 dB spread at that level. All BW2300/BW2750 SN
remain uncalibrated and are reported unchanged.
The unavailable noisy captures and empty gate WAV directions cannot supply
additional calibration pairs.

RX bitrate uses the bandwidth's nominal level ladder (physical table in
`evidence/RECORD-SEMANTICS.md` §6; low levels in `01`), independently of SN
calibration. When decoder quality is unavailable, neither SN nor BITRATE is
emitted for the data over. P0 and gate transcripts pin BW500 L1, L4–8 and
L10–13, plus BW2300 L6; remaining nominal rates lack a paired host transcript in these runs.

- P0 (2026-10-18) pins the listen modes, chat idle policy, buffer-clear replies
  and SN cadence above. B1/B2/B4 (2026-10-20) refine CQ host edges; P3
  implements the narrow CQ waveform at all three bandwidth arguments.
  The optional KISS port now carries advanced beacons and direct UI broadcasts.
- `VERSION`, `ENCRYPTION*` and `ABORT` edge cases are not fully specified here.
- The `BITRATE (N)` level → (modulation, FEC-rate) mapping is a physical/coding fact
  and belongs to [`01`](01-physical-layer.md)/[`03`](03-coding.md), not to the host API.
