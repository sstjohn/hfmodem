# hfmodem

Four HF softmodems, one station.

**Pre-alpha.** This keys a transmitter. Every interop claim below is bounded by
the dated bench and on-air observations, and most of them are narrower than
you would want — read them before pointing this at a gateway. Report anything
that keys unbidden or holds PTT the way `SECURITY.md` asks.

- **shrike** — PACTOR-1/2/3, the waveform published as ITU-R M.1798
- **kestrel** — VARA
- **besra** — ARDOP
- **sabir** — a clean-sheet protocol

with **hfhost** (host-protocol clients) and **creance** (the conformance bench).

The four were developed in separate repositories. They are unified here because they
do the same thing — modulate a signal on the air through a PTT-driven transceiver and
a sound card — and had begun solving the same problems four different ways while
contending for one radio.

## Layout

    packages/hfmodem     the four protocols, the shared core, and the station
    packages/hfhost      host-protocol clients — stdlib only, deliberately
    packages/creance     the conformance bench
    docs/                running a station, the station file, the on-air record
    docs/protocols/      normative descriptions of each protocol
    docs/viz/            the four protocols compared, drawn from the tree
    examples/            two station files: one for a radio, one for a recording

`ARCHITECTURE.md` explains why things are where they are, including the five things
that are deliberately *not* shared. `docs/STATION.md` is how to run it,
`docs/CONFIG.md` is what the station file refuses. The on-air procedure this
station flies — pre-flight, dry run, and a connect attempt — is its own operating
log and is not published.

`docs/viz/flock-compared.html` puts the four side by side — carrier geometry, symbol,
coding, ARQ timing, rate ladder — and opens in a browser with nothing installed. Its
figures are not drawings: `docs/viz/build.py` regenerates each one from two of the off-air
recordings the distribution carries and from the constants the modems import, and
`--check` fails if the page has drifted from either.

## Getting started

    python -m venv .venv
    ./.venv/bin/pip install -e packages/hfhost -e "packages/hfmodem[dev]" -e packages/creance
    ./.venv/bin/python -m pytest
    ./.venv/bin/hfmodem config examples/station.toml

Two extras, neither of them implied. **`dev`** is pytest and ruff, which is what the
line above installs and what the suite needs. **`radio`** is `sounddevice`, and it
buys exactly one thing: opening a sound card. `hfmodem devices` needs it; nothing
else in this file does.

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

As of **2026-09-20**. The table records dated sessions and operator receipt
confirmations, with their evidence limits:

| Modem | Date (2026) | Medium | Established milestone | Bound / still open |
|---|---|---|---|---|
| besra / ARDOP | Aug 19–26 | Live gateways | Mail received and sent in separate sessions; outbound message `2KX2HWM919HR` through WW2MI confirmed delivered | Reliable sessions across gateways |
| kestrel / VARA | Sep 9 | Live K0SI | All four offered messages pulled in one BW2300 session, clean close | Three full files and one partial; not repeated at another gateway |
| kestrel / VARA | Sep 17 | Live KB8AY | CMS accepted our secure login, took an outbound message, closed on `FQ` | Delivery into Winlink is not confirmable from this end, and the SID announced was not hfmodem's own |
| kestrel / VARA | Sep 18 | Live KB8AY | **Mail delivered.** Message `NL03TXWQ8SGL` crossed on an HF link and arrived at its internet address: `FS Y` accepted, 441 compressed bytes sent in two blocks, `FF` returned by the far end, and the operator has it in their inbox | One message, one gateway, one path; the SID announced was not hfmodem's own, and the CMS never confirmed receipt to us |
| kestrel / VARA | Reported Sep 20 | Operator confirmation | Several VARA test emails received in Gmail | Individual subjects, gateways and session dates not yet catalogued |
| shrike / PACTOR-1 | Sep 11 | Live KB5LZK | Four messages pulled in one exchange, over a link that never upgraded | One station, one evening |
| shrike / PACTOR-3 | Sep 20 | Live WS8EOC | **Mail delivered.** Message `9WQNVWBDLVOY` sent over PACTOR-3 through WS8EOC, with receipt confirmed by the operator | One named delivery; all-speed and cross-gateway reliability remain open |
| sabir | Aug 29 | Two public RF receivers, our decoder | Presence beacon decoded | No exchange with a second transmitting station — a WebSDR cannot transmit (`docs/protocols/sabir/ONAIR.md`) |

[Development status](docs/STATUS.md) gives the dates, evidence and limits behind
these milestones. [The documentation index](docs/README.md) maps the guides,
protocol descriptions and historical notes. All four remain pre-alpha.

## Licensing

The code is AGPL-3.0-only; see `LICENSE`. The documentation, the images and the
site are CC BY 4.0; see `LICENSE-CC-BY-4.0`, and `docs/protocols/LICENSE.md` for
what that does and does not reach in a protocol description.

besra is derived from ardopcf in the places `NOTICE` names, and `NOTICE` carries the
MIT notice and the Reed-Solomon notice that come with it.

hfmodem implements no spread spectrum, no frequency hopping and no automatic link
establishment (ALE). It encrypts nothing: every waveform here goes on the air in the
clear, and nothing in the tree hides, scrambles or obscures a transmission.

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
