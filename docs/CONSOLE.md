# Console and operator tools

The console provides per-mode monitoring, bounded PACTOR/VARA/ARDOP Winlink
attempts and campaigns, a foreground ALE/HFLINK station, HF selcall, and a
logbook with audio replay. The project remains an experimental prerelease for developers only. Successful live-radio
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

```sh
.venv/bin/python tools/logbook_view/serve.py --port 8738
```

Open `http://127.0.0.1:8738/`. The logbook reads your selected data directory and
station identity, and can start empty. Supply `--root PATH` to inspect an
explicit historical data tree, `--adif FILE` to import contacts, or
`--viewer-root PATH` to choose derived replay storage. Audio access is restricted
to the selected evidence trees. Expand an existing QSO and select **Inspect
rehear** to decode its saved audio. TX and peer RX use distinct waterfall colours;
recording playback and decode events share the recording clock.

Inbox polling is off unless `--imap ACCOUNT` is supplied. Static export uses
`--export DIR`, optional `--audio-base URL` and optional `--hostname HOST`;
it excludes local transcripts and requires recordings to be uploaded separately.
See the [replay guide](../tools/logbook_view/README.md) for packet, byte and timing
limits. The [site examples](https://hfmodem.ssj.lol/examples/) show successful
PACTOR-3, VARA and ARDOP mail transfers, with the recorded outcomes and limits
stated for each exchange.

For migration from a prior private checkout, import each explicitly selected
roster with `console data import-roster MODE PATH`. Keep historical recordings
and arms in their original tree and use the logbook's explicit `--root` option.
A source baseline comparison is optional through `HFMODEM_SOURCE_BASELINE`; snapshot
qualification through `--qualify` remains a separate execution check.
