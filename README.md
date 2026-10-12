# hfmodem

[![Release workflow status](https://img.shields.io/github/actions/workflow/status/sstjohn/hfmodem/release.yml?event=push&style=flat&label=release)](https://github.com/sstjohn/hfmodem/actions/workflows/release.yml?query=event%3Apush)
[![hfmodem version on PyPI](https://img.shields.io/pypi/v/hfmodem?style=flat&label=hfmodem)](https://pypi.org/project/hfmodem/)
[![hfhost version on PyPI](https://img.shields.io/pypi/v/hfhost?style=flat&label=hfhost)](https://pypi.org/project/hfhost/)
[![creance version on PyPI](https://img.shields.io/pypi/v/creance?style=flat&label=creance)](https://pypi.org/project/creance/)

**[Website, protocol guides and recorded examples → hfmodem.ssj.lol](https://hfmodem.ssj.lol/)**

Project contact: [hfmodem@ssj.lol](mailto:hfmodem@ssj.lol).

HF modems and shared station tools. The operator controls the station.
PACTOR, VARA and ARDOP are alpha; all other modes are experimental.

**Experimental prerelease for developers only.** This keys a transmitter. Every interop claim below is bounded by
the dated bench and on-air observations, and most of them are narrower than
you would want — read them before pointing this at a gateway. Report anything
that keys unbidden or holds PTT the way `SECURITY.md` asks.

- **shrike** — PACTOR-1/2/3, the waveform published as ITU-R M.1798
- **kestrel** — VARA
- **besra** — ARDOP
- **sabir** — a clean-sheet protocol
- **lanner** — RTTY, SITOR-B and NAVTEX
- **brubru** — [Robust Packet](docs/protocols/robust-packet/station.md)

with **hfhost** (host-protocol clients) and **creance** (the conformance bench).

**bateleur** provides MIL-STD-188-110 serial, 39-tone and
16-tone waveforms, STANAG 4285 and 4539, the STANAG 5066 link layer, HF
selcall and 2G ALE. The operator can enable ALE scanning to retune the rig
through a channel list. ALE directory lookup, DTM/DBM reception and linking
protection support monitoring as well as operation. These components have
station host ports; [their documentation](docs/protocols/bateleur/README.md)
records the tests and external recordings used. On-air operation is untested.

The original four — shrike, kestrel, besra and sabir — were developed in separate repositories. They are unified here because they
do the same thing — modulate a signal on the air through a PTT-driven transceiver and
a sound card — and had begun solving the same problems four different ways while
contending for one radio.

For a first PACTOR-3 or VARA Winlink call, follow the
[radio quickstart](#winlink-with-your-radio).

## Layout

    packages/hfmodem     modems, the shared core, and the station
    packages/hfhost      host-protocol clients — stdlib only, deliberately
    packages/creance     the conformance bench
    docs/                running a station, the station file, the on-air record
    docs/protocols/      normative descriptions of each protocol
    docs/viz/            the four protocols compared, drawn from the tree
    examples/            two station files: one for a radio, one for a recording

`ARCHITECTURE.md` describes the package boundaries. `docs/STATION.md` explains
station operation and `docs/CONFIG.md` describes the station file.
See the [changelog](CHANGELOG.md) for changes between public releases.

Explore recorded Winlink mail transfers in the [curated logbook](https://hfmodem.ssj.lol/examples/logbook/):
[VARA send](https://hfmodem.ssj.lol/examples/vara-send/) and
[receive](https://hfmodem.ssj.lol/examples/vara-receive/),
[PACTOR-3 send](https://hfmodem.ssj.lol/examples/pactor3-send/) and
[receive](https://hfmodem.ssj.lol/examples/pactor3-receive/), plus
[ARDOP send](https://hfmodem.ssj.lol/examples/ardop-send/) and
[receive](https://hfmodem.ssj.lol/examples/ardop-receive/).
The retained examples show VARA L9 send/L4 receive and PACTOR-3 SL6 send/SL3 receive.

A single data-loss incident cost significant development work and logbook material,
including the high-speed VARA receive recording and these ARDOP sessions'
original rig receive recordings. The examples use surviving recordings and
session records.

The ARDOP examples use remote witness recordings: one message delivered through
WW2MI and three received through W6IDS. The WW2MI session's subsequent receive
failed, and END was not confirmed.

The station uses a Yaesu FTDX10. The FT-891 remains supported through rig
profiles. X6100 and G90 support is planned through the same profiles; their
profile entries are not qualified station support.

The on-air procedure this station flies — pre-flight, dry run, and a connect
attempt — is its own operating log and is not published.

`docs/viz/flock-compared.html` puts the four side by side — carrier geometry, symbol,
coding, ARQ timing, rate ladder — and opens in a browser with nothing installed. Its
figures are not drawings: `docs/viz/build.py` regenerates each one from two of the off-air
recordings the distribution carries and from the constants the modems import, and
`--check` fails if the page has drifted from either.

## Winlink with your radio

Use Python 3.13 on macOS or Linux. Connect the FTDX10's USB CAT and
audio, or the FT-891's CAT connection and an external audio interface. Live
operation uses Hamlib and the `flac` and `nc` helpers; Linux also needs the
PortAudio system library. Setup checks the command-line tools and can offer to
install missing tools through Homebrew or apt with your consent.

Open Terminal and paste this block:

```sh
python3.13 -m venv ~/hfmodem-env
. ~/hfmodem-env/bin/activate
python3 -m pip install --pre hfmodem
hfmodem winlink
```

The first run guides you through the radio, callsign, regulatory profile,
licence class, CAT/PTT and audio settings. Grid is optional: press Enter to skip
it. Setup only saves configuration. Set the radio's CAT speed to 38400 baud.
Choose PACTOR-3 or VARA HF and the band your antenna is ready for. hfmodem
downloads the current public gateway list, filters it for your configured
licence and UTC operating hours, and ranks channels by distance when a grid
is provided. Choose a gateway, then receive mail or send a text message. Receiving is the default;
sending asks for a recipient, subject and message. Review the final **Start
this mail call?** prompt. Your Winlink password is entered at a hidden prompt
and held for this session only.

For a guided call, hfmodem sets and checks Yaesu DATA USB, the data audio route,
RTS PTT and your saved RF power target (initially 20 W). Read the startup
summary; these radio settings persist. Audio drive is provisional: watch ALC
and adjust the drive for your station.

In each new Terminal, activate the saved environment before running hfmodem:

```sh
. ~/hfmodem-env/bin/activate
hfmodem winlink
```

After setup, start directly with `hfmodem winlink pactor` or
`hfmodem winlink vara`. Add `--band 40` to select 40 m where your licence permits
it, or `--refresh` to update the gateway list. A saved-setup preview such as
`hfmodem winlink vara --dry-run` opens no radio hardware. Ctrl-C stops the call;
read the transfer and teardown results. Use `hfmodem console setup --replace`
to change the saved station settings. Read received messages offline
with `hfmodem winlink --inbox`.

The [operator guide](docs/CONSOLE.md) covers named gateways, sending mail,
monitoring and the logbook. Installation time depends on downloads and system
dependencies; making a contact also depends on the channel and propagation.

## Developing from a checkout

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e packages/hfhost -e "packages/hfmodem[dev,radio,operator]" -e packages/creance
```

The `dev` extra supplies pytest and ruff for development; tests are not a
prerequisite for an operator's first call. Sound-card and serial access are
installed by default; `radio` remains a compatibility extra. `operator` adds
certificate signing for optional LoTW uploads. The installed
`hfmodem console` command is also available as `tools/console` in the checkout.
Recordings use lossless FLAC. Setup checks for the `flac` helper as part of
the operator dependency installation.

## Without a radio

    ./.venv/bin/hfmodem station examples/replay.toml

A station that cannot transmit, running on a recording: `transmit = false` is a hard
interlock, so no CAT port is opened and no keying line is held, and the `[audio]`
input reads `replay:FILE`, so no sound card is touched either. It is the
configuration to develop against.

What it proves is that the station comes up — config, audio, lanes, host ports, in
that order — and then stops when the recording runs out. It does not decode anything:
the recording it points at is six seconds of an ARDOP gateway, and the lane the
example enables is VARA's. Point it at a recording of your own, at any sample rate,
and that changes.

## Where each modem stands

Reviewed **2026-10-10**. The table records dated sessions and operator receipt
confirmations, with their evidence limits:

| Modem | Date (2026) | Medium | Established milestone | Bound / still open |
|---|---|---|---|---|
| besra / ARDOP | Aug 19–26 | Live gateways | Mail received and sent in separate sessions; outbound message `2KX2HWM919HR` through WW2MI confirmed delivered | Reliable sessions across gateways |
| kestrel / VARA | Oct 2 review | Codec and ARQ implementation | High speed capable and speed limited links; transmit and receive ceilings span each bandwidth’s level table | Named RF sessions below establish transfers on their recorded paths |
| kestrel / VARA | Sep 9 | Live K0SI | All four offered messages pulled in one BW2300 session, clean close | Three full files and one partial; not repeated at another gateway |
| kestrel / VARA | Sep 17 | Live KB8AY | CMS accepted our secure login, took an outbound message, closed on `FQ` | Delivery into Winlink is not confirmable from this end |
| kestrel / VARA | Sep 18 | Live KB8AY | **Mail delivered.** Message `NL03TXWQ8SGL` crossed on an HF link and arrived at its internet address: `FS Y` accepted, 441 compressed bytes sent in two blocks, `FF` returned by the far end, and the operator has it in their inbox | One message, one gateway, one path; the CMS never confirmed receipt to us |
| kestrel / VARA | Reported Sep 20 | Operator confirmation | Several VARA test emails received in Gmail | Individual subjects, gateways and session dates not yet catalogued |
| shrike / PACTOR-1 | Sep 11 | Live KB5LZK | Four messages pulled in one exchange, over a link that never upgraded | One station, one evening |
| shrike / PACTOR-3 | Oct 2 | Live WS8EOC | Winlink mail sent and received over hfmodem’s PACTOR-3 | Mail transfer established; confirmed link shutdown remains separate |
| shrike / PACTOR-3 | Sep 20 | Live WS8EOC | **Mail delivered.** Message `9WQNVWBDLVOY` sent over PACTOR-3 through WS8EOC, with receipt confirmed by the operator | One named delivery; cross-gateway reliability remains open |
| brubru / Robust Packet | Sep 30 | Public recordings and generated audio | Four frames decoded from three stations; three regeneration comparisons; all length classes at both levels in loopback | Station and KISS attachment tested in software loopback; independent station bench and connected mode remain |
| sabir | Aug 29 | Two public RF receivers, our decoder | Presence beacon decoded | No exchange with a second transmitting station — a WebSDR cannot transmit (`docs/protocols/sabir/ONAIR.md`) |

[Development status](docs/STATUS.md) gives the dates, evidence and limits behind
these milestones. [The documentation index](docs/README.md) maps the guides,
protocol descriptions and historical notes. The project remains an experimental prerelease for developers only.

## Licensing

The code is AGPL-3.0-only; see `LICENSE`. The documentation, the images and the
site are CC BY 4.0; see `LICENSE-CC-BY-4.0`, and `docs/protocols/LICENSE.md` for
what that does and does not reach in a protocol description.

besra is derived from ardopcf in the places `NOTICE` names, and `NOTICE` carries the
MIT notice and the Reed-Solomon notice that come with it.

ALE linking protection encrypts ALE words when configured with a key. It is
off unless `[protocols.ale] lp_level` is set to 1 or 2; see
[the configuration guide](docs/CONFIG.md).

Hash functions appear, and none of them secures traffic; `grep -rn hashlib` finds
uses in the test suites as well as the code they exercise.
`hfmodem/winlink/session.py` answers a Winlink gateway's `;PQ:`
secure-login challenge with the MD5 challenge/response that Winlink's B2F protocol
prescribes for a client logging in with a password (`hfmodem mail` takes it from
the `WINLINK_PASSWORD` environment variable or from `--password-file`, both of
which keep it out of argv, and from `--password`, which does not) — it
authenticates the operator to the gateway and encrypts nothing. Other uses are
bookkeeping: hfhost records a SHA-1 of each data blob in its session transcripts,
and the conformance bench sums both ends of a test transfer with SHA-256 to prove
the same bytes arrived. Sabir's proof tooling generates reproducible payloads
and hashes payloads, waveforms, recordings and source files to identify the
artifacts being compared. Tests also seed fixtures and check that bytes survived
a round trip.

The [console operator guide](docs/CONSOLE.md) covers installation, single-mode
monitoring, ALE/HFLINK, peer rosters, flight planning and logbook replay.
For tactical digital callsigns, [separate CW identification](docs/IDENTIFICATION.md)
explains `--idcall` and the station configuration fallback.
