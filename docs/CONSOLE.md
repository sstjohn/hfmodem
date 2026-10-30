<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Console and operator tools

The console provides per-mode monitoring, bounded PACTOR/VARA/ARDOP Winlink
attempts and campaigns, Kestrel chat with VarAC peers, supervised voice and CW,
ALE/HFLINK, HF selcall, and a logbook with audio replay. The project remains an experimental prerelease for developers only. Successful live-radio
mail exchanges are established; experimental waveform and host interoperability
limits are described in the [status guide](STATUS.md).

## Install and receive Winlink mail

Use Python 3.13 on macOS or Linux. Plug in the FTDX10's USB CAT and
audio connection, or the FT-891's CAT connection and external audio interface.
The FT-891's USB CAT connection alone does not carry audio.

Open Terminal and paste this block:

```sh
python3.13 -m venv ~/hfmodem-env
. ~/hfmodem-env/bin/activate
python3 -m pip install --pre hfmodem
hfmodem winlink
```

A virtual environment also works on Linux distributions that protect their
system Python from pip installs. Live operation uses Hamlib (`rigctl` and
`rigctld`), `flac` for recordings, and `nc` for launcher connections; Linux
also needs the PortAudio system library. Setup checks the command-line tools and
can offer to install missing tools through Homebrew or apt, with your consent.
Homebrew supplies `hamlib` and `flac`, plus `netcat` if `nc` is missing. The apt
packages are `libhamlib-utils`, `libportaudio2`, `flac` and `netcat-openbsd`.
If Linux audio discovery reports missing PortAudio, install `libportaudio2` and rerun.
Downloads and system dependencies determine installation time; a live exchange depends on the
channel, gateway and propagation.

When no station is saved, `hfmodem winlink` starts guided setup. Choose the
FTDX10 or FT-891, enter your callsign, select your regulatory profile and US
licence class where applicable, and check the CAT/PTT and audio devices. Grid
is optional; press Enter to skip it.
A single detected USB audio card can be selected automatically; a laptop's
built-in microphone is never chosen for you. Confirm whether to enable later
calls. Setup only writes configuration and does not access the radio.
Use the radio's Enhanced serial interface for CAT and its separate Standard
interface for RTS PTT. Set CAT speed to 38400 baud, which the current launcher expects.

Next choose PACTOR-3 or VARA HF and the band your antenna is ready for; the
band menu uses your configured profile. The console fetches the current public
gateway CSV, filters channels by the configured licence and UTC hours, and
ranks them by distance when you provide a grid. Choose a gateway and channel,
then a mail task: **Receive mail** (the default) or
**Send a text message and receive mail**. Sending asks for **To** (a callsign or email address), **Subject** and
**Message text**. Review the final **Start this mail call?** prompt. The Winlink
password prompt is hidden; the password is held for this session only, not
saved in the station file.

After confirmation and the password prompt, the guided call prepares and
verifies the known Yaesu DATA USB mode, data audio route, RTS PTT and saved RF
power target, initially 20 W. The startup summary shows what will be set.
The launcher holds the station lock during preparation and the call; the radio
settings persist afterward. Audio drive remains provisional: watch the radio's
ALC during transmission and adjust drive for your own audio chain. Guided
FT-891 calls do not start an auxiliary witness recording by default.

## Subsequent calls and previews

In each new Terminal, activate the environment saved during installation.
Then choose a call or preview:

```sh
. ~/hfmodem-env/bin/activate
hfmodem winlink pactor
hfmodem winlink vara
hfmodem winlink vara NS0A --band 40
hfmodem winlink pactor --refresh
hfmodem winlink vara --dry-run
```

The 40 m example assumes a US General or Extra licence. Use bands permitted by
your configured station profile. A callsign narrows the gateway choices,
`--band` narrows the band, and `--refresh` downloads a fresh listing. With saved setup, `--dry-run`
previews without opening radio hardware or requiring Hamlib. Normal calls
retain the final start prompt.

`Ctrl-C` stops a call and lets the launcher release PTT and finish recording.
Read the transfer and teardown results: connecting to a gateway, transferring
mail and confirmed destination delivery are different outcomes.

```sh
hfmodem console setup --replace
hfmodem console data refresh pactor
hfmodem console data refresh vara
hfmodem console pactor peers
hfmodem console log
```

Rerun setup to change the station. The saved defaults are
`~/.config/hfmodem/station.toml` and `~/.local/share/hfmodem`, respecting
`XDG_CONFIG_HOME` and `XDG_DATA_HOME`. `HFMODEM_STATION` and
`HFMODEM_DATA_DIR` select explicit alternatives. The station file is the authority
for identity, rig, audio and transmit permission; `transmit = false` disables
station transmission. Keep credentials and recordings out of public commits.
See the [station guide](STATION.md) and [configuration reference](CONFIG.md)
for manual settings and audio calibration.

## Send a message and read received mail

The guided mail task menu can send ordinary text; no B2F file is needed.
You can also supply the message directly:

```sh
hfmodem winlink vara --to CALL --subject 'Hello' --body 'Message'
hfmodem winlink --inbox
```

Replace `CALL` with the recipient's callsign or email address. The send command
still offers gateway selection and the final start confirmation, and receives
mail in the same call. `--inbox` reads saved received messages entirely offline.

## Advanced setup and checkout tools

The installed `hfmodem console` command is also available as `tools/console`
in a source checkout. Development installation uses:

```sh
python3 -m pip install -e ./packages/hfhost -e './packages/hfmodem[dev,radio,operator]' -e ./packages/creance
```

`dev` provides development checks, and `operator` optional LoTW certificate
signing. Sound-card and serial access are installed by default; `radio` is
retained as a compatibility extra. Live launchers use Bash. Optional LoTW
signing uses a callsign certificate exported from TQSL. Supply
`--locations locations.json`, keyed by station grid, with string fields `DXCC`
and matching `GRIDSQUARE`, plus applicable `CQZ`, `ITUZ` and `US_STATE`.
No country or zone is inferred. Upload requires `--upload` and its explicit
confirmation.

For legacy FT-891 launchers, export `CAT_PORT` and `PTT_PORT` to the actual
serial devices. Set `HAMLIB_BIN` only if Hamlib is outside `PATH`. Operator data
holds per-mode rosters, arm records, leases, campaigns and derived evidence.
`hfmodem console data init` creates empty rosters without replacing existing
files; gateway refresh supplies real PACTOR and VARA channels for first use.

## Peers and dry runs

A roster is a JSON object with a `peers` list. Each peer needs a unique `id`,
`call`, `band_m`, `centre_hz` and `dial_hz`. Frequencies are integer Hz: the USB
dial is the listed RF centre minus 1500 Hz. VARA and ARDOP peers also require `bw_hz` (advertised modem bandwidth).
For the examples below, add `"bw_hz": 2300` to the VARA roster and
`"bw_hz": 500` to the ARDOP roster. Optional fields include `grid`,
`hours` (UTC, for example `00-23`) and mode-specific bandwidth information.
A minimal synthetic roster for trying the planning interface is:

```json
{
  "peers": [
    {
      "id": "N1TEST-40",
      "call": "N1TEST",
      "band_m": 40,
      "centre_hz": 7101500,
      "dial_hz": 7100000,
      "hours": "00-23"
    }
  ]
}
```

These are demonstration values for manual roster import. Use `data refresh`
for current PACTOR and VARA gateways, and verify the selected service and
availability before operating.

```sh
hfmodem console data import-roster pactor peers.json
hfmodem console data import-roster vara vara-peers.json
hfmodem console data import-roster ardop ardop-peers.json
hfmodem console pactor peers
hfmodem console pactor fly N1TEST --band 40 --sense --no-witness --dry-run
hfmodem console vara fly N1TEST --band 40 --bw 2300 --direction rx --sense --no-witness --dry-run
hfmodem console ardop fly N1TEST --band 40 --bw 500 --sense --no-witness --dry-run
hfmodem console pactor campaign --band 40 --cycles 1 --sense --no-witness --dry-run
hfmodem console log
```

Use `--replace` on `data import-roster` to replace a nonempty roster deliberately.
Empty peer listings and logbooks work; a flight without a matching peer refuses
before opening hardware. Run each command's `--help` for supported mode options.
`--sense` checks occupancy once. `--pounce SECONDS` waits for a clear channel.
`--force` bypasses the occupancy decision and leaves the other checks in place.

A flight or campaign requires exactly one of `--dry-run` or `--execute`.
Review the printed frequency, rig, drive, source identity and complete command
before using `--execute`. Named level presets describe launch parameters;
verify actual RF output and audio linearity on your own station. An accepted
command or a local decode does not establish delivery to the far end.

`--power WATTS` sets the radio's RF power at the start of a flight in place of
the preset's, and every flight and chat session lowers a radio found above
`[rig] power_clamp_w` to it; see [the station file](CONFIG.md#transmit-power).

## Station ownership

```sh
hfmodem console attach station.toml
hfmodem console own station.toml --receive
hfmodem console release station.toml
```

Attach leaves hardware unopened. Omit `--receive` to request transmit ownership,
including its brief keying proof. See [station service](README-station.md).

## Receive-only operation

```sh
hfmodem console pactor monitor
hfmodem console rtty monitor --freq 14080
hfmodem console vara monitor N1TEST --band 40
hfmodem console stanag4285 monitor --mode 1200L
hfmodem console pactor monitor --wav recording.wav --no-record
hfmodem console monitor --list-modes
```

Without a frequency or peer, the monitor uses the current rig dial. A supplied
frequency is in kHz and tunes once at startup. You can retune on the radio while
listening; the monitor keeps running. The configured audio passband should
match the receiver, including a 4 kHz voice filter if selected. New events avoid
reusing a stale CAT frequency after manual retuning. See the
[monitor guide](console-monitor.md) for waveform parameters and recordings.

For ALE scanning, sounding, manual calls and native text, follow the
[HFLINK guide](console-hflink.md). ALE AMD text, DTM/DBM data and numeric selcall
pages have different encoding, length and acknowledgement rules; the console
reports them explicitly. External DTM/DBM and automatic selcall acknowledgement
interoperability still require a cooperative rig test. The passive `ale monitor`
and `selcall monitor` commands never transmit.

## Kestrel chat

### VARA chat quickstart

Chat with VarAC peers using hfmodem's Kestrel modem, included in 0.4.0.
Use Python 3.13 on macOS or Linux and an
FTDX10 with USB audio, or an FT-891 with an external audio interface.

From the checkout root, install and configure the station:

```sh
python3.13 -m venv ~/hfmodem-env
. ~/hfmodem-env/bin/activate
python3 -m pip install -e packages/hfhost -e packages/creance -e packages/hfmodem
hfmodem console setup --transmit
```

Setup asks for your radio, callsign, licence, CAT/PTT ports and audio devices.
Enter your grid for beacons. It saves configuration without transmitting and
checks the required system tools. If Winlink setup is already saved, reuse it;
add `--replace` only to change it. A saved receive-only station needs setup
with `--replace --transmit` before chat can transmit.

Choose a band your licence and antenna support. Preview the 20 m plan, then
start the session:

```sh
hfmodem console vara chat --band 20 --bw 500 --dry-run
hfmodem console vara chat --band 20 --bw 500
```

The second command opens the radio and answers incoming calls.
Check the displayed calling frequency and watch ALC during transmission.

1. Type `/heard` to list heard stations and CQs.
2. Answer a listed CQ with `/answer N`, using its displayed number. To call
   CQ yourself, type `/cq`; hfmodem checks a slot and waits there for an answer.
3. Once connected, type text and press Enter to send it.
4. Type `/bye` to end the contact and return to monitoring, or `/quit` to exit.
   `/stop` cancels an operation or aborts a link, discarding unconfirmed text.

Use `/status` for progress and `/help` for commands. `/beacon` enables repeating
beacons; `/beacon off` stops them. For passive listening only,
use `hfmodem console vara monitor`.

In a new Terminal, run `. ~/hfmodem-env/bin/activate` again before starting
chat. Change `--band 20` to your chosen band; 500 Hz is the default chat bandwidth.

### Chat commands

The session monitors the calling frequency, makes or answers contacts, and
returns to monitoring when a contact ends. Bandwidths are 500, 2300 and 2750 Hz.
`/answer N` uses the bandwidth advertised by the heard CQ.

| Command | Action |
|---|---|
| `/heard`, `/slots` | Show heard stations and channel activity |
| `/answer N` | Answer a heard CQ |
| `/cq [SLOT] [repeat]` | Call CQ on a free slot, optionally repeating |
| `/call CALL [SLOT]`, `/call CALL cf` | Call a station on a slot or the calling frequency |
| `/qsy SLOT`, `/band N` | Request a slot change or change bands while idle |
| `/follow TARGET` | Follow `all`, `cf`, a slot or dial kHz; `off` hides transcripts |
| `/beacon [MINUTES]` | Set cadence; `now`, `force` and `off` control one-shot or repeating beacons |
| `/beacon hop BANDS [every MINUTES]` | Rotate through a list of bands' calling frequencies, beaconing on arrival |
| `/beacon hop auto [BANDS] [every MINUTES]` | Choose ready bands using local heard activity and periodic exploration |
| `/beacon hop status`, `/band ready`, `/band unready` | Inspect hopping, acknowledge the current band's match, or withdraw readiness |
| `/away [MESSAGE]`, `/back` | Set or clear the away reply |
| `/vmail [N]`, `/vmail send SUBJECT` | Read or send VMail |
| `/vmail pending`, `/vmail collect CALL`, `/vmail dismiss CALL` | List parked-mail notices, connect to collect mail, or dismiss a notice |
| `/bbs list`, `/bbs get N` | List files and download a selected file with checksum and format validation |
| `/all` | All-stations broadcast; choose `cf` or `slot N`, enter text, inspect `status` or stop with `off` |
| `/inquire heard` | Request the connected peer’s last-heard list |
| `/auto on`, `/auto off` | Enable or disable automatic answers |
| `/verbose on`, `/verbose off` | Show or hide transmit and receive diagnostics |
| `/power [WATTS]` | Show or set RF power, held to the station clamp; set after the key drops if transmitting |
| `/status`, `/help` | Show the current operation or commands |
| `/bye [text]`, `/quit` | Close the contact or leave the session |
| `/stop` | Cancel the current operation; abort a connected link and discard unconfirmed text |

Plain text goes to the connected peer. Passive following covers the current
receive passband; activity outside it is missed. `/follow cf` returns to the
calling frequency. Passive transcripts are logged.

See [BBS files](VARAC-BBS.md) for download limits, directory hosting and the
distinction between validated downloads and raw research capture.

Receive retry requests, successful retry reception, held outgoing bytes and local
stall closure are shown even with `/verbose off`. The closing notice reports
elapsed time without protocol progress, recovery requests sent and unacknowledged
outgoing bytes. A transmitted request does not prove the peer received it;
undecoded activity does not prove that a retry arrived.

In a BW500 chat initiated here, a pending receive retry can solicit up to two
additional NAKs after at least 12 seconds since the previous request and sufficient
fresh receive audio. Recovery waits for a clear channel and completed receive
candidates. Complete CRC-clean replies at an earlier requested base rate remain
acceptable if the peer missed a later request. Recovery consumes the existing
stall budget; it does not acknowledge unread data or keep a broken link alive
indefinitely.

### Beacon band hopping

Hopping is optional and starts only when requested. Visit each intended band with
`/band N`, verify its antenna match (use the rig's TUNE control if needed), then
type `/band ready` on that band. Readiness lasts for this session; `/band unready`
withdraws it. ATU on/off and filter readback alone never establish a usable match.

```
/beacon hop 80 40 20 every 15
/beacon hop auto 80 40 20 every 20
/beacon hop status
/beacon off
```

The first form cycles through the listed calling channels in order. The second
chooses automatically within the list; `/beacon hop auto` considers all supported
bands. Lists use band numbers, not arbitrary dial frequencies or QSO slots.
Hop dwell defaults to 15 minutes and can be as short as 5 minutes, for example
`/beacon hop 80 40 20 every 5`. Use `every MINUTES` to
distinguish an interval from a band number such as 15, 20 or 30.

Each move requests one beacon on arrival, after five seconds of clear channel.
A manual `/band N` also requests an arrival beacon while hopping is enabled.
Each band has a 15-minute minimum since its last confirmed beacon, including
manual beacons; turning hopping off and on does not erase that history. A busy
arrival waits until clear or misses that opportunity. Missed opportunities never
produce catch-up bursts. Contacts, calling, close exchanges, keying and off-channel
following suspend hopping. Failed band transitions pause it until an operator
retries `/band N` and re-enables hopping. Fixed rotation pauses at an unready band;
automatic selection skips it and reports why.

Automatic choices use unique stations heard in this session near the current UTC
hour (including adjacent hours), normalized by listening time. Evidence is bounded
to seven days and is separate from slot occupancy. Every third choice, or every
choice without activity evidence, explores the least recently monitored eligible
band. The reason is printed before each move. This is a local activity heuristic;
it does not query PSKReporter or a propagation forecast, and evidence is not saved
across restarts.

`/beacon MINUTES` returns to single-band beaconing. `/beacon off` and `/quit` cancel
hopping; repeating beacons also expire after 24 hours. Existing single-band and
one-shot behavior is unchanged when hopping is off.

An addressed `<SMP>` broadcast means the sender has parked VMail for you.
`/vmail pending` lists notices saved in your mailbox across restarts.
For example, `/vmail collect KQ4QWH` calls that station on its saved notice
channel and uses the normal QSY exchange. If the notice is on another band,
use `/band N` first. Without a saved notice, collection calls on the current
band's calling frequency. Reception of a notice never starts a call by itself.

The relay forwards its mail after connection and QSY; there is no separate
download request. Each complete VMail is saved before `<SMR>` acknowledges it.
Forwarded mail retains its original sender and compose time separately from
the connected relay. `/vmail` lists saved mail, `/vmail N` reads it, and
`/bye` ends the contact. `/stop` uses the normal call/QSY cancellation behavior.
Failed calls and interrupted messages leave the notice available for retry.
A repeated forwarded message with the same author, compose time, recipient,
subject and body is acknowledged without adding another inbox entry.

Notices remain until `/vmail dismiss CALL`: receiving one message does not
prove that the relay has no more mail. Dismissing a notice keeps downloaded
VMails. Only complete supported text envelopes addressed to your station (or
with the compact envelope's omitted destination) are accepted. Unsupported
extensions, including urgent/attachment envelopes, remain in the session log
without a success acknowledgement.

`/bbs list` requests a peer's directory. `/bbs offer N` selects an offer;
`/bbs capture` explicitly records opaque transfer bytes. It does not produce a
validated download. Use `/help bbs` for peer-service requirements.

`/email send TO_EMAIL REPLY_EMAIL SUBJECT` starts an outbound email-gateway
submission. Use `/email status` for its result and `/help email` for the workflow.
Replies go to the external reply address; submission does not confirm delivery.

Add `--pskreporter --loc GRID` at launch to publish validated CQ and beacon
reception reports. Reporting is off by default. `/pskreporter status` shows its
state; `/pskreporter off` stops it.

## Voice and CW

Macagua provides speech recognition and synthesized voice; merlin encodes and
decodes CW. The contact terminal prepares reply drafts for operator review.
Both modes are experimental. Preview a contact without opening the radio:

```sh
hfmodem console voice fly --dial 14.285 --cq --dry-run
hfmodem console voice fly --waveform cw --dial 14.050 --cq --dry-run
```

Dial frequencies here are MHz. Replace `--dry-run` with `--execute` for a
supervised contact; Enter approves each transmitted over. Voice requires local
speech models and a speech runtime. `hfmodem console voice test` uses the
computer's microphone and speakers without the radio.

## VARA transmit calibration

`hfmodem console vara txcal --band 40 --execute` transmits stepped audio drives
at the current dial and reports RF power and ALC. `--band` checks the band; it
does not retune. Use the recommendation for the current radio and audio chain;
`--set` records the recommended drive and codec volume. Transmit-level monitoring warns
when the codec volume changes.

## Advanced messages, witness and power options

Winlink credentials come from `--mail-password-file PATH`, `WINLINK_PASSWORD`,
`WINLINK_PASSWORD_FILE`, or an interactive hidden prompt. Unattended execution
without a credential refuses before the radio opens. The password is kept out
of command-line values and saved transcripts. Never commit credential files.

For a delivery test or an existing B2F file, use the advanced console flight
options. A VARA TX flight defaults to a delivery-test message. Set its recipient in
`operator.toml` under the selected data directory, or export
`HFMODEM_DELIVERY_TO`. `--send delivery` selects the same feature for other
mail-capable flight paths; `--send FILE` sends an explicitly supplied B2F file.
The arm freezes sender, recipient, MID and message hash so later configuration
changes cannot alter the recorded message. Missing delivery configuration
refuses during planning.

An optional `operator.toml` example is:

```toml
delivery_to = "your-address@example.net"
witness_candidates = "/absolute/path/to/kiwi-candidates.csv"

# Enable this only after measuring your own smartplug/radio combination.
# [power]
# mac = "00:11:22:33:44:55"
# alias = "my-radio"
# zero_w = 0.5
# idle_w = 21.0
# carrier_w = 40.0
```

The witness inventory is CSV with `hostname`, `port`, `name_location`,
`lat_lon` (latitude,longitude) and `free_channels` fields; `tools/kiwi_witness.py --help` shows selection options.
`HFMODEM_WITNESS_CANDIDATES` overrides the inventory path. Witness geography
comes from your station grid; an explicit `--grid` can select another location.
An absent inventory reports no witness and a regular flight may continue.
The Sabir one-way proof command requires a placed witness and refuses without
one. The shipped Kiwi client requests uncompressed USB audio with fixed gain.

Power control is optional and disabled until an explicit smartplug MAC and
measured limits are configured. It supports the legacy TP-Link local protocol
on port 9999. Every cached address is checked against the device MAC and,
if configured, its alias. A carrier above the measured threshold refuses a
power cycle; a quiet meter cannot prove PTT is released. Stop the transmitter
before using a relay. These sample limits do not qualify your hardware.

## Stop, records and replay

`Ctrl-C` stops a flight and lets its launcher release PTT and finish recording.
A foreground HFLINK station also accepts `quit`. Read the teardown result:
it reports whether cleanup succeeded. `tools/onair.sh down` is the launcher's
emergency cleanup command and remains usable without station identity settings.
Complete diagnostic transcripts stay in arm records; the terminal normally
shows milestones and refusals. `hfmodem console MODE status` and `hfmodem console log`
read saved evidence without touching the radio.

The station service mounts the watch-only logbook and replay face at
`http://127.0.0.1:8737/` (or the station's configured port). It uses the loaded
station identity, including empty defaults. Audio access is restricted to the
selected evidence trees. Expand a QSO and select **Inspect rehear** to decode
saved audio; playback and decode events share the recording clock.

To include additional contacts in the live logbook and `/api/log.adi`, set an
optional directory in the station file. Relative paths resolve beside that file:

```toml
[desk]
adif_dir = "imported-logs"
```

The face reads that directory's `.adi` files on each log request. `/api/log.adi`
is the LoTW ADIF export. Inbox confirmations are refreshed only on explicit request:

```sh
hfmodem console logbook --imap ACCOUNT
hfmodem console logbook --root PATH --adif FILE --export DIR
```

`--imap` performs one refresh using the app password in `~/.imap_password`;
the service never polls the inbox. `--adif FILE` imports contacts into an offline
snapshot and is repeatable. Static export accepts optional `--audio-base URL`
and `--hostname HOST`; it excludes transcripts and requires recordings to be
uploaded separately. With no action, `console logbook` prints usage.
See the [replay guide](../tools/logbook_view/README.md) for packet, byte and timing
limits. The [site examples](https://hfmodem.ssj.lol/examples/) show successful
PACTOR-3, VARA and ARDOP mail transfers, with the recorded outcomes and limits
stated for each exchange.

For migration from a prior private checkout, import each explicitly selected
roster with `console data import-roster MODE PATH`. Keep historical recordings
and arms in their original tree and use the logbook's explicit `--root` option.
A source baseline comparison is optional through `HFMODEM_SOURCE_BASELINE`; snapshot
qualification through `--qualify` remains a separate execution check.
