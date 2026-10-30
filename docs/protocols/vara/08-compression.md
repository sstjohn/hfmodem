<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# 08 — Payload compression (host bytes → pre-FEC frame)

VARA HF 4.9.0 has two distinct link codecs: TEXT uses static prefix codes;
FILES uses raw deflate blocks. These findings were measured on 2026-09-27
with a stock pair and independently decoded CRC-clean overs. The earlier
version of this document described the 2021 HE0/HE3 reference implementation;
its containers, transmitted tables and LSB packing do **not** describe the
measured 4.9.0 wire format.

## Mode and scope

The caller's link-setup body byte 7 selects the mode for both directions:

| High-speed caller | Mode |
|---|---|
| `81` | OFF |
| `83` | TEXT |
| `85` | FILES |

Bit 0 denotes high-speed capability, bit 1 TEXT, bit 2 FILES. Speed-limited
callers send plain data regardless of their compression preference. The called
station follows the caller even when its own preference differs. OFF sends
host bytes directly, without a container. The compression setting is frozen
for the connection; host changes require an idle contact.

## FILES

A delivery is a sequence of blocks, which may span over boundaries. Each
block starts with a two-byte big-endian header:

- Bit 15: raw deflate (`1`) or stored bytes (`0`).
- Bits 0–14: length of the body following the header, excluding the header.

Input blocks contain at most 32,500 host bytes. Compress each independently
with raw RFC 1951 deflate (`wbits=-15`); store the input when compression does
not reduce its length. Different valid deflate encoders need not produce the
same compressed bytes. FILES has no per-over compression flag.

On receive, consume exactly the advertised number of bytes. For compressed
blocks, require deflate EOF, no unused input or unconsumed tail, and at most
32,500 output bytes. A truncated stream can produce all its apparent output
without reaching EOF. Stored blocks obey the same output limit. A closing
over must leave no partial header or block buffered.

Current transmission limits each delivery to 50,000 host bytes: one stock
64-KiB run lost the bytes after 50,000 despite a drained transmit buffer.
That observation has not been replicated, so the receiver-limit explanation
remains provisional. FILES BUFFER accounting uses container bytes.

## TEXT

In a TEXT session, field bit `20` marks a coded over; with it clear, payload
is verbatim. OFF sessions always pass bytes through, regardless of this bit. Each
coded over is independently decoded, with MSB-first bits and a three-bit
header:

| Header | Meaning supported today |
|---|---|
| `000` | A line prefix follows |
| `001` | Prose table, coded |
| `010` | Prose table, continuing a raw escape run |
| `011` | SFI/date table |
| `100` | SID/version table |
| `101` | Uppercase/word table |
| `110` | Numeric table |
| `111` | FW symbol table, continuing mid-line |

The receive dictionaries cover prose, SFI/date, SID/version, uppercase/word,
numeric and FW forms. Line prefixes select a dictionary and may insert a
literal or a complete line. The tables are in
`hfmodem.kestrel.vara.compression`.

ESC introduces raw eight-bit bytes until and including the first byte having
a code in the table. CR introduces framing: optional `10000` for LF, then a
variable line prefix unless the over ends there. This applies to raw CR too.
Padding is 1–8 copies of the complement of the last data bit. All reads,
including the optional LF, must stay before the pad-stripped end.

Unsupported headers, unknown code prefixes and unfinished symbols/raw bytes/
line prefixes stop decoding. The verified prefix can be delivered; nothing
from or after the unreadable portion can be passed as intact host data.

Line prefixes include `;FW: `, `; `, `FC EM `, `F> `, SID/version forms and
complete lines such as `FF\r`. Continuation headers resume the selected table
without repeating the line prefix. Escapes and table transitions preserve
partial lines across overs.

## Implementation and host integration

`VaraStationHandshake` applies compression at its host-data seams. FILES
encoding precedes over framing, and parsing follows ARQ duplicate suppression.
TEXT retries are compared as decoded bytes because re-encoding at another
over size changes their wire representation.

Both modes latch a damaged receive direction until the connection ends.
Already verified bytes remain delivered; later bytes in that direction are
withheld. ARQ still acknowledges CRC-clean overs. One
`WRONG COMPRESSED DATA offset=...` warning is emitted, with details in the log.
Offsets identify wire bytes, including FILES headers; TEXT errors also identify
a bit position within the failing over. A new session clears the latch.

`tools/kestrel_host.py` is the audio-backed TCP host entry point. It supplies
`VaraHostModem` to `VaraServer` and drives the existing audio transport plus
connection/recovery/idle cadences on one owner thread. Host threads enqueue
commands; receive faults reach the command socket. The adapter defaults to
high-speed capability; `--low-speed-only` explicitly disables it. The separate
package host CLI still defaults to loopback.

The launcher defaults to BlackHole cables and reports PTT ON/OFF to the host;
it does not configure or key CAT. It can be started with:

```sh
python tools/kestrel_host.py --tx-device "BlackHole 16ch" --rx-device "BlackHole 64ch"
```

A defensive DISCONNECT while idle or connecting cancels that attempt, without
arming a close on the next session. A connected caller drains its queue before
closing. The caller and answering station both use their measured close
descriptors; transport refusal leaves the close pending for retry. LISTEN preserves
pre-connect host writes until the caller's mode is known. Public driver methods
provide queue accounting and pre-connect data; the adapter does not inspect
private queue fields.

Direct callers select OFF, TEXT or FILES with `compression=...`, or through
`tools/kestrel_connect.py --compression`. OFF remains that CLI's default.
The live TCP host defaults to TEXT; legacy `COMPRESSION ON` selects TEXT.
High-speed callers advertise their selected mode. Speed-limited callers use OFF.

TEXT transmission codes intermediate prose overs and uses plain fallback for
terminal overs and unsuitable input. Retry and BUFFER accounting retain the
original host bytes. Independent virtual-audio checks cover TEXT reception,
prose transmission, binary replies and closing against stock VARA.

## Verification

Generated tests cover stored/deflated block boundaries and limits, truncation,
trailing input, damage latching, mode negotiation, retries, host command/warning
routing, and the prose code table. Two malformed TEXT regressions are pinned:
`29 b8 ff` returns only `a` with an error; `28 86 c0` returns only `aa\r` with
an error, never an LF fabricated from padding.
