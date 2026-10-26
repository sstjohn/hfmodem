<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# bateleur host interface

An application drives bateleur over two TCP ports on the station, set in the
station file:

    [protocols.bateleur]
    enabled   = true
    host      = "bateleur"
    cmd_port  = 8310
    data_port = 8311

This dialect is bateleur's own. MIL-STD-188-110 and STANAG 4285 define the
waveform, not how an application talks to a modem, and this interface is not a
copy of any other modem's. An application written for another 188-110 modem's
TCP interface will not attach here unchanged.

One application at a time. A new connection on either port replaces the
previous one, which is closed.

## Host connections

Listeners bind to loopback by default. Clients send their normal protocol
opening directly, including on data ports and the STANAG 5066 SIS port.
There is no extra handshake or key file. A connection opening with an HTTP
request line is closed before any command or data in its body is handled.
The check preserves the first command's bytes and framing; binary openings
are released as soon as they cannot be an HTTP method prefix.

Bateleur, ALE and selcall command lines are limited to 4096 bytes. Socket
writers wait at 64 queued writes or 256 KiB; a slow reader stays connected.
Modem callbacks and SIS indications hand off to a delivery thread for each
connection, so socket backpressure never waits on reception or station ticks.
Undelivered output remains in memory until the client reads it or closes.
Explicit close drains output for up to 250 ms before shutting down.

## Data port

- **Bytes written** are queued for transmission. They go out on `SEND`, or on
  their own after `AUTOSEND` milliseconds without new data.
- **Bytes read** are what the modem decoded off the air, delivered as soon as
  the decoder has committed to them. A long message arrives while it is still
  being sent. The receiver that produced the current `RX START` owns
  this stream until `RX END`; competing detections cannot insert bytes.

## Command port

ASCII lines, terminated by CR, LF or both. Each command gets one answer line,
`OK ...` or `ERR <reason>`. The modem also writes unsolicited event lines, so
read the port as a stream of lines and match answers to commands in order.

| Command | Answer | Effect |
|---|---|---|
| `VERSION` | `OK bateleur 1` | |
| `MODE` | `OK MODE <std> <mode>` | Report the current transmit mode. |
| `MODE 110 <mode>` | `OK MODE 110 <mode>` | Transmit 188-110 serial in `<mode>`: `4800N`, or `2400`, `1200`, `600`, `300`, `150` or `75` with `S` (short), `L` (long) or `Z` (zero interleave). |
| `MODE 4539 <mode>` | `OK MODE 4539 <mode>` | Transmit STANAG 4539's high-rate waveform (MIL-STD-188-110B appendix C): `3200`, `4800`, `6400`, `8000` or `9600` with `US`, `VS`, `S`, `M`, `L` or `VL` (1, 3, 9, 18, 36 or 72 frame interleaver, e.g. `9600L`), or `12800US` uncoded. It is always listened for: its preamble names the rate and interleaver. |
| `MODE WBHF <mode>` | `OK MODE WBHF <mode>` | Transmit MIL-STD-188-110C appendix D in 3 kHz: `<rate>` with `US`, `S`, `M` or `L` (ultra short to long interleaver), rates `75` (Walsh; no `US`), `150`, `300`, `600`, `1200`, `1600`, `2400`, `3200`, `4800`, `6400`, `8000`, `9600`, `12000`, `16000`; e.g. `16000L`. It is always listened for: its preamble names the waveform and interleaver. See [WBHF.md](../../protocols/bateleur/WBHF.md). |
| `MODE 39 <mode>` | `OK MODE 39 <mode>` | Transmit the 39-tone parallel mode (MIL-STD-188-110 appendix B), and decode the next 39-tone transmission heard in it: `<rate>/<degree>`, then `/S` for synchronous or `/A<bits>.<data bits>` for asynchronous characters (ten-bit characters of eight data bits if left out), `F` after it for frequency-only diversity, `X` for the extended preamble; e.g. `600/99`, `2400/288/S`, `75/36/A7.5`. The answer gives the full name (`600/99/A10.8`). `offair modes` lists the rates and degrees. |
| `MODE 16 <rate>` | `OK MODE 16 <rate>` | Transmit, and also listen for, the 16-tone DPSK mode (MIL-STD-188-110 appendix A) at `2400`, `1200`, `600`, `300`, `150` or `75` bit/s. |
| `MODE 4529 <mode>` | `OK MODE 4529 <mode>` | Transmit, and also listen for, STANAG 4529 (4285 at 1200 Bd in 1240 Hz): coded `1200`, `600`, `300`, `150` or `75` with `L` or `S`, or uncoded `1800U`, `1200U`, `600U`; `/<Hz>` after it for a sub-carrier other than 1700 (1100-2300 Hz on the station). See [S4529.md](../../protocols/bateleur/S4529.md). |
| `MODE FSK <mode>` | `OK MODE FSK <mode>` | Transmit, and also listen for, binary FSK (STANAG 4481 annex C, MIL-STD-188-110B 5.1): `<baud>W` (850 Hz shift about 2000 Hz; `W/1700` about 1700 Hz), `<baud>N` (170 Hz), `<baud>V<n>` (multi-channel sub-carrier n, `D` after it with its twin), at 50, 75, 150, 300, 600 or 1200 Bd; an `S` after any sends the bare synchronous stream instead of start-stop characters. Modes whose tones leave the station's declared span are refused. See [FSK.md](../../protocols/bateleur/FSK.md). |
| `MODE 4285 <mode>` | `OK MODE 4285 <mode>` | Transmit, and also listen for, 4285 in `<mode>`: coded `2400`, `1200`, `600`, `300`, `150` or `75` with `L` or `S` (e.g. `600L`), or uncoded `3600U`, `2400U`, `1200U`. |
| `FRAMING ASYNC <format>` / `FRAMING SYNC` | `OK FRAMING ...` | Start-stop characters on the air, both ways (MIL-STD-188-110B C.5.4.1.1): each byte written goes out as a start bit, its low data bits least significant first, any parity bit and the stop bits; received words are taken apart again. `<format>` is data bits, parity (`N`, `E`, `O`, `M`, `S`) and stop bits: `8N1`, `7E1`, `5N2` and the like, 5 to 8 bits with any parity. It applies to 188-110, 4539, WBHF, 4285, 4529 and 16-tone. FSK and the 39-tone mode frame characters their own way. The default, `SYNC`, sends the bytes themselves (C.5.4.1.2). Set it to match the far station: many hardware modems and MARS stations run 188-110 asynchronous. |
| `BITORDER LSB` / `BITORDER MSB` | `OK BITORDER ...` | Which end of each byte goes on the air first, both directions. Default `LSB`, the asynchronous serial convention. |
| `SEND` | `OK SEND <n>` | Transmit the `<n>` queued bytes as one message. |
| `AUTOSEND <ms>` / `AUTOSEND OFF` | `OK AUTOSEND ...` | Transmit the queue after `<ms>` without new data. Default off. |
| `ABORT` | `OK ABORT <n>` | Discard the `<n>` buffered bytes and cancel already posted messages before their next piece. A piece already keying finishes. |
| `BUFFER` | `OK BUFFER <n>` | Queued bytes. |
| `LISTEN ON` / `LISTEN OFF` | `OK LISTEN ...` | Receive, or not. Default on. |
| `ZEROINTERLEAVE ON` / `OFF` | `OK ZEROINTERLEAVE ...` | Receive a short-interleave 188-110 preamble as zero interleave. The standard gives zero interleave no preamble of its own and leaves it to be agreed with the far end. Default off. |
| `STATUS` | `OK STATUS MODE <std> <mode> RX <idle\|receiving> TX <idle\|sending> BUFFER <n>` | |

Commands, including FRAMING, posted to the modem wait for completion, including any transmission
already in progress. Audio buffering does not consume command queue space.
A connection closing cancels its commands that have not started; a command
already running finishes with its actual result.

The data buffer and posted transmit data each have a 65,536-byte limit.
A refused `SEND` leaves the buffered bytes available. `AUTOSEND` stops and
reports an error if posting fails.

### Events

| Line | Meaning |
|---|---|
| `RX START <std> <mode> OFFSET <hz>` | A transmission was found. For 188-110, 4539 and WBHF the mode is the one its preamble named; the offset is the carrier error. |
| `RX END EOM SNR <db> <detail>` | A message ended with its end-of-message marker: 188-110, 4539, WBHF, or 4285 in a coded mode (STANAG 4285 Annex E framing). |
| `RX END LOST SNR <db> <detail>` | The signal went away before an end-of-message marker. Uncoded 4285, the 16-tone mode and FSK have no marker, so they always end this way. |
| `TX START <std> <mode> <bytes> <seconds>` | One transmission is keying. |
| `TX END` | It went out. |
| `TX REFUSED <reason>` | The station would not key it, and why. |

`RX END` retains the receiver's detail, such as `signal lost before EOM`,
`signal lost`, `signal ended`, or `receiver reset`.

## Receiving

- **188-110, 4539 and WBHF.** The modem always listens for them, whatever
  `MODE` is set to. The preamble names the rate and interleave.
- **4285.** Nothing on the air names the mode. The modem listens for 4285 only
  after `MODE 4285 <mode>`, and only in that mode.
- **16-tone.** Like 4285, nothing on the air names the rate. The modem listens
  for it only after `MODE 16 <rate>`, and only at that rate. The last element
  is filled out with zero bits, so up to three NUL bytes may follow a message.
- **4529.** As for 4285: the modem listens only after `MODE 4529 <mode>`, in
  that mode and on that sub-carrier.
- **FSK.** Nothing on the air names the mode either: the modem listens for it
  only after `MODE FSK <mode>`, and only in that mode.
- **39-tone.** Always listened for; its preamble is its own but does not name
  the mode. After `MODE 39 <mode>` a transmission is decoded in that mode
  as soon as its timing is found. Otherwise the modem finds the mode from the
  data, which takes about eight seconds of it before `RX START 39 <mode>` and
  the first bytes. A synchronous 39-tone message has no end marker: it ends
  with `RX END LOST` when the signal goes.

## Long messages

The station refuses any single transmission longer than `[rig] max_key_s`. A
message too long for one key-up is sent as several complete transmissions, each
with its own preamble and end marker, and each reported by its own
`TX START` / `TX END`. A bateleur receiver hands their bytes on in order, so the
far end's application reads one stream. Nothing re-sends a lost piece: these
waveforms are one-way, and retransmission belongs to the layer above.
