<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# ALE and selcall with the console

For installation and guided radio setup, start with the [operator guide](CONSOLE.md).

`hfmodem console hflink station` runs a foreground 2G ALE station: scan selected
channels, answer calls, hold a link, exchange native text, and optionally sound
hourly on selected pilot channels. `hfmodem console ale station` is the same
command. `hfmodem console selcall station` runs held-channel HF selcall, including
native text pages. Incoming messages appear directly in the terminal; no
Winlink connection or separate messaging application is needed.

The existing `ale monitor` and `selcall monitor` commands remain passive.

## Automatic HFLINK station

Use your station file with the correct rig, rigctld endpoint, separate RTS PTT
port, callsign, audio devices and drive. `HFMODEM_STATION` selects that file;
`--config FILE` overrides its path. Select the FTDX10 with `HFMODEM_RIG=ftdx10`
or `--rig ftdx10`. The console uses the existing shared transmitter, callsign
identification, emission checks and shutdown path. It claims the same locks as
console flights and monitors. It does not start rigctld for you.

Set USB or PKTUSB on the radio before starting. The console checks the sideband,
preserves the filter, and accepts a 4 kHz receive passband. This does not widen
the ALE or selcall transmitted waveform. Frequencies below are **USB dial
frequencies in kHz**, including below 10 MHz.

Preview a three-channel HFN base station:

```sh
hfmodem console hflink station --channels 7102 10145.5 14109 \
  --pilot 7102 10145.5 14109 --dry-run
```

Replace `--dry-run` with `--execute` to run. The file must already have
`[station] transmit = true`; the execute flag cannot override it. A dry run
opens no hardware, sockets or session files. The launch applies **automatic
control** to the selected emission profile because this station answers and
sounds on its own; the source file is unchanged. Only the ALE lane is enabled
for this session, regardless of the file's other enabled protocols.

The channels and pilots are explicit so an old saved list cannot unexpectedly
choose where you sound. Verify them against the
[HFLINK channel plan](https://hflink.com/channels/) before operating. The
published HFN data channels include 7102, 10145.5 and 14109 kHz. The plan is
not a report of present activity. HFL calling/voice channels and HFN
mobile/portable stations use `--no-sound` instead of `--pilot`.

Pilots must be members of the scan list. The HFLINK preset uses 15-second
scanning call preambles, TWAS sounds, hourly sounding and a six-minute link
idle timeout. The first scheduled sounds are staggered through the first hour,
then each pilot is due hourly. A busy channel defers a scheduled sound. Link
activity holds the channel; scanning resumes after hangup or idle expiry.
See [HFLINK sounding](https://hflink.com/sounding/).

The foreground terminal remains the control point. `quit` closes the station;
Ctrl-C uses the existing rig panic path. Closing standard input also stops the
station. Do not run it with stdin redirected from `/dev/null`. ALE shutdown
returns to the first channel selected for this session. While scanning, the
console owns tuning; use `scan off` or `channel` before manually moving the
radio. ALE refuses transmission if the actual dial differs from its selected
channel. A passive monitor remains the appropriate command for free manual
retuning without an ALE controller.

## Calls and native messages

At the `hflink>` prompt:

```text
status
heard
channels
call KQ6XA
msg KQ6XA HELLO FROM W9SSJ
hangup
net HFN
channel 14109.0
sound
scan on
```

`call ADDRESS [text]` searches the configured channels using measured link
quality, or sends within an existing link. Incoming individual and supported
net/group calls are answered by the existing ALE controller. `net HFN` and
`net HFL` use its HFLINK response-slot convention. `heard` lists recent stations,
channels and quality; `status` shows link and scan state.

`channel` accepts the printed channel index or exact name (for example
`14109.0`); it tunes and stops scanning. A linked or busy controller refuses a
channel change: `hangup` a link, or wait for the current call to finish.
`scan off` holds the current channel. With
scanning off, new calls and scheduled sounds stay on that channel, even if
other pilots are overdue. `scan on` restores the scan and full pilot schedule.

To run a manual held-channel ALE session without scheduled sounds:

```sh
hfmodem console ale station --freq 14109 --no-sound --execute
```

`msg ADDRESS text` sends ALE **AMD**, within a link or carried by a new call.
AMD permits up to 90 characters from the expanded 64-character alphabet
(ASCII space through underscore); lowercase is transmitted as uppercase.
Incoming AMD messages print as `AMD ADDRESS text`. Text needs no shell-style
quotes at the interactive prompt; quotes entered there become message content.
Unsupported characters and oversized messages are refused rather than cut off.
The configured key-up watchdog may impose a smaller practical AMD limit,
especially when a long message accompanies a scanning call. A refusal is shown.

For a peer supporting ALE data messages, `dtm ADDRESS text` and
`dbm ADDRESS text` use the existing native DTM/DBM ARQ implementation. They
split text into protocol blocks and acknowledge/retry them. Incoming text
prints as `DTM` or `DBM`; `DTM SENT`/`DBM SENT` report completed ARQ. These
paths remain self-tested, with external-radio interoperability limits described
in [ALE](implementation/bateleur/ALE.md). Both ends need matching support; AMD is
the first practical HFLINK messaging test.

`OK` means the modem accepted the command. `TX complete` means the burst
finished. Neither proves the peer received an AMD or page: request a text
reply. A completed ALE link proves the handshake, while DTM/DBM ARQ has its
own completion/failure events. All commands and terminal messages use printable
ASCII on one line.

## Selcall and paging

Run selcall on the radio's current USB dial, or supply a dial once:

```sh
hfmodem console selcall station --address 1234 123456 --dry-run
hfmodem console selcall station --address 1234 123456 --freq 14094 --execute
```

Replace the numeric IDs with your assigned IDs. Four- and six-digit IDs are
supported, at most one of each. The originating ID must have the same length
as the called ID. The default dot preamble is six seconds (`--dots SECONDS`).
This lane never retunes the radio after the requested initial tune, so you can
retune the rig yourself while listening. Finish a call/page before moving it.

At the prompt:

```text
selcall 5678
page 5678 HELLO FROM W9SSJ
ack 5678
beacon 5678
status
quit
```

`call 5678` also works in a selcall-only session. Pages permit **1–32 printable
ASCII characters**, uppercased on transmission. Received pages show sender,
recipient and text. Calls to one of your IDs are marked `MINE`; calls to
other IDs still appear. `ack ADDRESS` sends the implemented acknowledgment
call. Add `--acknowledge` to opt into automatic acknowledgments of individual
calls requesting them. This is the implemented acknowledgment-call format,
not a claim of compatibility with every radio's proprietary revertive tone.
See [selcall formats and evidence](protocols/bateleur/SELCALL.md).

You can enable both lanes on **one held ALE channel** with, for example,
`console hflink station --freq 7102 --no-sound --selcall 1234 --execute`.
Selcall alongside a multichannel ALE scan is refused: its receiver cannot yet
hold an ALE scan throughout a selcall preamble/page. Use a separate
held-channel selcall session for the published SEL channels. This also keeps
ALE sounding on the intended ALE channels.

## Output and retained evidence

Normal output shows heard stations, state changes, incoming text, calls,
responses, completed links, sounds, transmit starts/completions and refusals.
Repeated heard reports are thinned to once per station/channel per 30 seconds;
messages and transmit failures are always printed. Repeated identical tuning
failures are also thinned, with the first failure visible. Routine retunes are summarized by
station status every 30 seconds. `--verbose` prints every retune and repeated
event; `--summary-seconds 60` slows the heartbeat and `0` disables it.

The console prints its session directory under `[audio] capture_dir`. Every
event, including hidden repeated reports and retunes, is flushed to
`events.jsonl`; `plan.json` records the launch settings. `--log-dir PATH`
selects a new directory explicitly. These files retain exchanged text. This
station console journals events; it does not record receive audio. Use an
independent receiver/recording when you need on-air waveform evidence.
