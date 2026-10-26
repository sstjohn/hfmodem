<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Listening with the console

For installation and guided radio setup, start with the [operator guide](CONSOLE.md).

`hfmodem console <mode> monitor` opens the receive audio and decodes just that
protocol. It does not call a station, answer a call, acknowledge a packet,
scan channels or transmit. With no target it uses the rig's current dial and
mode. Set the radio's sideband and filter for the signal you want to hear.
A voice passband up to 4 kHz, such as the FTDX10's wide SSB filter, is supported:
RTTY and SITOR/NAVTEX acquire tone pairs throughout that audio range. The audio
input is captured at 48 kHz without cropping it to a 3 kHz data filter. Other
waveforms use their standard audio placement; the wider radio filter does not
change the waveform's bandwidth. The monitor preserves your radio's filter.
Set `filter_hz` under `[rig]` in the station file to match your receive passband
for accurate recording metadata.

For active ALE scanning, sounding, calls and native text, or selcall calling
and paging, use the [HFLINK console](console-hflink.md).

```sh
hfmodem console pactor monitor
hfmodem console rtty monitor --freq 14080
hfmodem console vara monitor NS0A --band 40
hfmodem console ardop monitor --peer W6IDS-40m-7061500-500
hfmodem console stanag4285 monitor --mode 1200L
hfmodem console mil110 monitor --framing 7E1
hfmodem console cw monitor --audio-centre 700
hfmodem console monitor --list-modes
```

Frequencies after `--freq` are **dial frequencies in kHz**: `7104` means
7.104 MHz. An optional peer is a saved console callsign or full channel ID;
`--band` resolves callsigns listed on several bands. Multiple entries at the
same dial frequency are one listening target. If a callsign names different
frequencies, the command lists the matching IDs and asks you to choose.
The peer is a frequency shortcut: traffic from other stations on that channel
is still decoded. PACTOR, VARA and ARDOP use their own saved peer lists. Other
monitor commands can use a named channel from any of those lists. No peer
list is needed to monitor the current dial or an explicit frequency.

The console tunes **once**, before opening audio, when you supply a frequency
or peer. You can then retune with the radio's controls and keep listening.
The console never writes the frequency again or restores the starting dial
on exit. The periodic channel line shows signal level and the latest CAT
dial read. Decoded events from live rig monitoring do not attach an RF
frequency calculated from the opening dial: a receiver can still be decoding
buffered audio when you move the knob. The recording and events remain one
continuous run. Give `--dial HZ` only to label a run when CAT is unavailable;
it does not tune the radio.

## Available receivers

| Command mode | What it decodes | Receive selection |
|---|---|---|
| `pactor` | PACTOR family receivers | Automatic within the PACTOR family |
| `vara` | VARA | Automatic |
| `ardop` | ARDOP | Automatic |
| `rtty` | ITA2 RTTY, 45.45 Bd, 170 Hz shift | Automatic tone-pair acquisition |
| `sitor-b` | SITOR-B text, including NAVTEX bulletins | 100 Bd, 170 Hz shift |
| `navtex` | Completed NAVTEX bulletins over SITOR-B | Bulletin headers and text |
| `sabir` | Sabir beacons and control bursts | Existing passive monitor decode path |
| `rpr` | Robust Packet AX.25 frames | Both supported levels; CRC-checked frames |
| `ale` | 2G ALE words, calls and messages | Receive primitives only; no link controller |
| `selcall` | HF selcall, pages and other supported call types | No automatic acknowledgment |
| `mil110` | MIL-STD-188-110 serial | Rate and interleaver from preamble |
| `stanag4285` | STANAG 4285 | `--mode 1200L` by default |
| `stanag4529` | STANAG 4529 | `--mode 600L` by default; optional carrier suffix supported by Bateleur |
| `stanag4539` | STANAG 4539 | Rate and interleaver from preamble |
| `wbhf` | MIL-STD-188-110C appendix D, 3 kHz | Waveform and interleaver from preamble |
| `tone39` | 39-tone parallel waveform | `--mode 600/33/A10.8` by default |
| `tone16` | 16-tone waveform | `--mode 1200` by default |
| `fsk` | Bateleur binary FSK | `--mode 75W` by default; e.g. `50N`, `75V3D` |
| `mod110` | 188-110A MOD channel bits | Payload coding remains unknown; output is packed channel bits |
| `cw` | Morse text | `--audio-centre 700` Hz by default |

<!-- site:omit:start -->
The dedicated `pactor4` receiver provides experimental PACTOR-4 chirp and
burst recognition. It selects PACTOR-4 only.
<!-- site:omit:end -->

`stanag` (also `stang` or `bateleur`) is a convenience spelling for selecting
one Bateleur waveform. It defaults to 4285 at 1200L; for example,
`hfmodem console stanag monitor --mode '4529 600L'` runs only 4529. The individual
commands are clearer when you know the standard. `--list-modes` also lists
aliases, including `110`, `4285`, `4529`, `4539`, `sitor`, `robust-packet`
and the modem bird names.

The waveforms with an explicit receive selection do not identify that
selection on air. Choose the correct rate, interleaver and carrier rather
than assuming the default matches the transmission. For serial waveforms,
`--framing 8N1`, `7E1` and the other supported character formats decode
start-stop characters. The default `SYNC` preserves synchronous bytes;
`--bit-order lsb|msb` selects their bit order. FSK and 39-tone have their own
character framing. Receive sub-carriers are not restricted by the station's
transmit passband; for example `--mode 75V16D` and `--mode 600L/2400`
remain available when listening through a wider filter. Binary output includes a hex representation, and terminal
control bytes are escaped. A successful waveform decode alone does not prove
an application message passed its own checksum.

PACTOR monitoring is passive observation of the existing receiver outputs;
it does not establish an ARQ session or promise to reconstruct arbitrary
third-party mail.
<!-- site:omit:start -->
PACTOR-4 recognition is experimental.
<!-- site:omit:end -->
Sabir's passive
monitor recognizes its existing beacon and control formats. MOD monitoring
does not claim to decode the unknown payload coding.

## Sources, recordings and stopping

The monitor uses the selected station audio input; override it with
`--device NAME`. It takes the console's arm locks while reading the local
card so a console flight cannot contend for that stream. It never changes
receive gain, sideband, filter or PTT.

By default a live run saves audio segments, `events.jsonl` and provenance
under the station's configured `[audio] capture_dir`. `--no-record` disables
audio recording while keeping events and provenance. Ctrl-C stops the source
and lets the receiver drain; a second Ctrl-C stops waiting. `--minutes N`
bounds the audio duration. These commands also accept the existing monitor
sources and recording options:

```sh
hfmodem console rtty monitor --minutes 30 --no-record
hfmodem console pactor monitor --wav capture.wav
hfmodem console vara monitor --arm SAVED-ARM
hfmodem console navtex monitor --kiwi HOST --freq 518 --span 6000
hfmodem console rtty monitor --no-rtty-usos
```

`--no-rtty-usos` keeps RTTY in figures across spaces. A recording cannot be
retuned with `--freq` or a peer; use `--dial HZ` for its frequency label.
The selected receiver runs directly over the input audio, without the
multimode tracker or per-track dispatch. Use the original `hfmodem console
monitor` when you want all available monitor decoders and track discovery.
