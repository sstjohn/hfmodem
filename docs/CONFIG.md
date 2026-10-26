<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# The station file

For initial FTDX10 or FT-891 setup, run `hfmodem winlink`; it starts the guided
setup when no station is saved. Use `hfmodem console setup --replace` to change
the saved settings. The [operator quickstart](CONSOLE.md#install-and-receive-winlink-mail)
covers installation and the first call. This page describes manual configuration.

Select the radio with `HFMODEM_RIG=ftdx10` or `--rig ftdx10` where offered;
use `ft891` for the FT-891. `HFMODEM_STATION` selects the station TOML. Explicit
CAT/PTT and device options take precedence over the selected profile.

The FTDX10 profile uses Hamlib model 1042 at 38400 baud and its built-in USB
audio codec. The FT-891 profile uses model 1036 with an external audio interface.
Configure your own serial endpoints rather than copying another station's
adapter IDs. Put FTDX10 endpoints and timing in `[rig.profiles.ftdx10]`, and
input/output devices, drive and measurements in `[audio.chains.ftdx10_usb]`.
The flat `[audio]` remains the FT-891 chain. Profile entries for X6100 and G90
are planned support, with incomplete timing and readback qualification.

FTDX10 operation was confirmed by the operator on 2026-10-02, including two
PTT tail checks that recorded the radio identity and clean unkey. Those
observations apply to that station's configured ports and audio chain; they
are not calibration measurements for a different installation. Missing
calibration can use provisional settings and warns accordingly. Existing
transmit authorization, identity, occupancy, clipping and unkey checks apply.
Help and console dry runs resolve configuration without opening devices.

For console installation, data directories and launch examples, see the
[operator guide](CONSOLE.md).

One file, the whole station. Unknown keys are an error at load: a typo'd key is a
setting that silently did not apply, and on a transmitter that is the difference
between a hot input and a working one.

Everything is validated eagerly. Validating on first use means a station comes up,
listens for an hour, answers a call, and only then discovers it cannot key.

See `examples/station.toml` for a station template requiring your PTT
configuration and `examples/replay.toml` for one that cannot transmit.

## Rig and audio selection

The station now uses a Yaesu FTDX10. The FT-891 remains supported through rig
profiles; X6100 and G90 support is planned through that system. Profile entries
alone do not establish qualified operation on a rig.

The shared tool resolver uses an explicit `--rig` selection, then `HFMODEM_RIG`,
then a legacy model selection, with `ft891` as its fallback. Select `ftdx10`
explicitly for the current station. Profile defaults and station-file overlays
resolve CAT/PTT endpoints, audio devices and timing; explicit tool options take
precedence. Inspect resolved settings rather than copying the old rig’s gains
or timing values to a different audio chain.

## Keys that are refused, and why

**`dial_hz`** is rejected by name. A gateway list publishes the channel *centre*;
the dial is derived as centre − 1500 Hz. Every modem in this project has had that
backwards at some point, and it is silent when wrong — the radio keys, the waveform
is correct, and nothing is heard in either direction.

**`input_gain`** outside 0.005–0.10 is refused. The measured working point for this
station's chain is 0.040; it has been 0.51, 0.50, 0.18 and 0.118 elsewhere, and a
hot input does not present as an error. It presents as a working radio that decodes
nothing, which is how it survived so long. The bound is coarse and cannot catch a
plausible wrong value for a different chain — that is what preflight measurement is
for.

**A port used by two protocols** is refused. Kestrel and bateleur use separate
command and data ports (bateleur with `host = "s5066"` uses one `port`);
Sabir uses one native host-interface port. Assign distinct listening ports
to every enabled protocol.

**Host ports.** Listeners bind to loopback by default and accept each
protocol's normal opening directly. An HTTP request line closes the
connection before its body can reach the modem. See
[host connections](implementation/bateleur/HOST-API.md#host-connections).
`tx_drive` must be between zero and one, inclusive.

The station's VARA and ARDOP hosts use their real modem owners and the same
receive stream and transmit arbiter as the other enabled protocols. A
receive-only station refuses their transmission requests. Their standalone
host-server defaults still provide loopback for protocol development.

PACTOR with `host = "ptc"` mounts the native session driver on a POSIX PTY.
`pty` optionally names an exclusively created symlink; an existing path is
preserved and startup fails. Each sequential attachment gets a fresh PTY under
that name. Without `pty`, the endpoint is reported under a private temporary
directory. This initial station profile requires the transmit arbiter, CW
identification, Normal calls, MYLevel 1 or 3 and PTT settle plus 10 ms fitting
the native 90 ms call-answer turnaround. Longer TXDelay and Robust/longpath
outgoing selections refuse. MODE defaults to 2 and MAXError to 70, from the SCS
manual. Full stock SCS compatibility remains unqualified.
`options.drive_reference_mv` optionally sets the positive calibration reference
(default 3000 mVpp). Unconfirmed wire bytes and exact original host bytes are
retained separately under the reported runtime recovery root; sixteen retained
contact directories stop further calls rather than discard recovery evidence.
PTC CW identification, scanner holds, terminal controls and delivery recovery
are implemented. This is not a qualified unattended station profile.
See [host compatibility](HOST-COMPATIBILITY.md) for client checks and limits.

Robust Packet also accepts `host = "tracker"` on a POSIX PTY and
`host = "agwpe"` on a TCP `port`. These hosts provide connected AX.25 sessions.
The KISS host leaves link control to its client.

Managed hosts are `vara-managed`, `ardop-managed`, `ptc-managed` and
`tracker-managed`. Each uses a `port` and an `auth_token_file` containing a
32-byte hex secret. Supervised Windows adapters attach to these hosts.
Legacy command/data pairs require an exclusively supervised endpoint; shared
localhost ports cannot identify racing applications.

VARA's optional `kiss_port` enables its KISS beacon and UI broadcast interface.
The live VARA host defaults to TEXT compression; legacy `COMPRESSION ON`
selects TEXT. OFF and FILES remain available.

**`[protocols.selcall]`** takes one four-digit address and/or one six-digit
address, with `dots_s` between 0.2 and 20 seconds. `acknowledge` and
`emergency` default to false; `EMERGENCY` requires the latter to be enabled.

**`[protocols.ale]`** is refused without `addresses` (1 to 15 of A-Z and 0-9
each, MIL-STD-188-141B A.5.2.4.4) or without `channels` (dial frequencies in
Hz, 1.5 to 30 MHz), or with a `scan_rate` outside 0.5 to 10 channels a second.
Each of `nets` is `{address, member, slot}`: the net's address, the one of
`addresses` this station answers it with, and its response slot, 1 to 32.
`lp_level` 1 or 2 turns on linking protection (AL-1 or AL-2), which encrypts
ALE words using `lp_key`, 14 hex digits. It is off by default. Protected
reception uses the configured key and clock (to the minute for AL-1, a
second or so for AL-2); `lp_accept_clear` also permits clear reception.
Choosing `regulatory = "part97"` excludes linking protection: enabled ALE
lanes with `lp_level` set are refused at configuration load and lane creation,
and the choice is checked again before transmission. With `unregulated`, a
protected lane warns at startup for configured amateur band-plan frequencies
and at each transmit attempt on one of those frequencies. These are limits
of the operator's chosen profile.
Directory lookup, DTM/DBM reception and protected reception support
monitoring as well as operation; see [ALE.md](implementation/bateleur/ALE.md).
`sound_s` must be zero or at least 60 seconds; `activity_s` must be finite
and nonnegative. The `hflink` preset supplies those intervals and rejects
explicit `sound_s` or `activity_s` entries, including entries equal to defaults. `base` and `pilot_channels` require `hflink`.
The channels are dial frequencies because ALE channel lists publish them that
way; the lane converts to the centre that `[rig] centre_hz` is written in.

**`bandwidth` in `[protocols.bateleur]`** other than `"3000"` or `"3200"` is
refused. It is the highest audio frequency the rig transmits: `"3000"`, the
default, trims the signal's top edge to fit under it; `"3200"` sends the
untrimmed pulse. The station uses the selected bandwidth's declared audio
span for its emission check; measured spans are in `core/occupied.py`.

**`rig.ptt.backend` other than `"rts"`** is refused, and `"rigctld"` is refused by
name because it is the plausible wrong answer: it is not implemented, and this
station would key RTS anyway — a transmitter keyed by a route other than the
configured one.

**Missing `control` or `regulatory`** is refused. Neither has a default.
`regulatory = "part97"` applies the limits the operator chose: a 2.8 kHz
bandwidth ceiling, data segments and power ceilings selected by `licence`,
and additional placement and response-width checks for automatic control.
`regulatory = "unregulated"` applies no emission limits and requires a stated
`because` reason. Responsibility for the station is stated in the README.

The bandwidth check uses the supplied passband width; placement adds the drift
allowance on both edges. `core.occupied` supplies outward-rounded -26 dB edges,
which include modulation skirts that carrier span or 99-percent power bandwidth
can omit. The 60 m power comparison assumes a 0 dBd dipole; the software does not
measure antenna gain.

## What the loader cannot check

Use separate CAT and PTT endpoints. Configuration alone does not prove the
wiring. `hfmodem rig station.toml --arm` checks the selected keying line;
this briefly transmits. A rigctld backend must leave PTT disabled.

## A station on a recording

**`[audio] input = "replay:PATH"`** is not a device name. The station reads PATH and
opens no card. With `transmit = false` beside it — as in `examples/replay.toml` —
that is the whole station on a machine with no sound card, no CAT port and no keying
line: every enabled lane decodes and every host port accepts, and nothing that could
key a radio is opened at all.

PATH is relative to the directory the station is started from, so an absolute path is
the safe form for anything started from elsewhere. `~` expands. A bare
`input = "replay"` is refused: it says to replay without saying what, and the error
says what to write instead.

Any WAV does. The file is read at whatever rate it was written, first channel, and
resampled to the card rate on the way in, so a 12 kHz capture and a 48 kHz one reach
the lanes as the same thing a card would have delivered. `input_gain` is a property of the
codec chain and is not applied to a recording — a capture's level is the level it was
recorded at, and dividing that out is how a corpus stops being gradable.

**`output`** is never opened; `replay.toml` writes `replay` there to say so.
A station on a recording plays nothing back, and the bursts it would have sent are
kept rather than emitted. The run ends when the recording does.

## What this station calls itself to Winlink

**`[station] client_sid`** is the name and version this station announces in its
B2F SID — `HFM-0.1` by default, on the air as `[HFM-0.1-B2FHM$]`. `--mail-sid`
overrides it for one run on every mail-capable tool, and `hfmodem config` prints
the whole SID as it will be sent.

It is here because Winlink's production servers refuse a client type they do not
recognise. KX8U's CMS answered ours with *Unknown client types are not allowed on
production servers*, ahead of the `;PQ:` challenge and ahead of any mail. What a
station announces is a claim its operator makes about their own station, so the
value is the operator's; hfmodem ships its own name and no one else's.

The refusal's suggestion to use `cms-z.winlink.org` refers to Winlink's
development CMS, which skips the client-type allowlist. ARSFI's reference
client selects `cms-z.winlink.org` in debug builds and `cms.winlink.org` in
release builds, only on its telnet path. The relay path carries no CMS hostname:
an RMS gateway chooses which CMS it relays to, and an over-air client cannot
select one in the B2F exchange.

The Winlink Development Team's client-onboarding instructions require a
temporary RMS with its callsign whitelisted to test against `cms-z` over RF,
because existing RMS gateways do not usually have access to the development
CMS. Transmitting to a production gateway does not change its CMS destination.
Production acceptance of hfmodem's own SID requires client registration. The
operator controls the announced identity; inspect it in `hfmodem config` and
the session report.

The capability letters — the `B2FHM$` — are **not** settable. Each is a promise
the code keeps (B2 forwarding, F compression, H hierarchical addresses, M message
IDs), so they are derived from `winlink/session.py` and change when it does. A
declared capability the code does not have breaks a session further in and far
more obscurely than being refused at the greeting does. The loader refuses a
`client_sid` containing brackets, `$`, or anything outside printable ASCII: a CR
in that field would not be a malformed SID, it would be a second protocol line
injected into the handshake.

## The two facts with one home

`core/band.py` holds the dial offset. `core/levels.py` holds the codec working
point. Both are ratcheted by `tests/gates/test_singleton_facts.py`, keyed on the
setting's *name* rather than its value — 0.04 is also the FT-891's PTT settle and a
filter constant, and 1500 is also five modulators' audio passband centre. Different
facts that share a number, and a gate that conflates them is a gate people switch
off.
