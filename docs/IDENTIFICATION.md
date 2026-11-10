<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Separate CW and digital callsigns

PACTOR, VARA and ARDOP calling attempts can use a digital tactical callsign
while closing with a different station callsign in CW:

```toml
[station]
mycall = "TACTIC"
idcall = "N0CALL"  # replace with your station callsign
```

`mycall` remains the digital caller, including link setup, Winlink login and
message identity. Protocol callsign format restrictions still apply. This
feature does not create a separate Winlink account identity.

The CW call resolves in this order: `--idcall CALL`, `IDCALL` environment,
`[station] idcall`, then the attempt's effective digital `mycall`. Omitting
`idcall` preserves the existing identity. An empty TOML `idcall` means omitted;
an explicitly empty CLI or environment value is an error. CW calls use ASCII
letters, digits and `/`, normalized to uppercase with outer whitespace removed.
Internal whitespace, Unicode and SSIDs such as `-1` are rejected; provide an
explicit CW callsign when the digital call contains an SSID. No suffix is
silently removed. A selected station file needed to resolve identity must load
successfully; an explicit CLI or environment CW override bypasses that lookup.

The console accepts the option for both flights and campaigns:

```sh
MYCALL=TACTIC console pactor fly PEER --sense --dry-run --idcall N0CALL
MYCALL=TACTIC console vara fly PEER --bw 2300 --sense --dry-run --idcall N0CALL
MYCALL=TACTIC console ardop fly PEER --sense --dry-run --idcall N0CALL
```

Replace the example calls and peer with your actual station and destination.
The same `--idcall` reaches `tools/onair.sh pactor`, `pactor-answer`, `vara`,
`vara-mail`, `ardop-mail`, and their calling Python entry points. Identity
options must be spelled completely (`--mycall`, `--idcall`); these entry points
do not abbreviate options. Final passthrough identity options take precedence
over console-native values. Console plans record both identities and the CW
source, and freeze both `MYCALL` and `IDCALL` in the child environment. Final
passthrough identity options keep their precedence and their position in argv.
An absent conventional `station.toml` remains absent; an explicitly selected
missing configuration is an error.

Closing CW is required by default. `--cwid-optional` retains existing behavior
when digital and CW calls match (ignoring ASCII case and outer whitespace):
PACTOR suppresses CW only after a link came up; VARA and ARDOP suppress their
closing CW. A distinct CW identity always requires closing CW, including when
`--cwid-optional` is present, because the digital frames carry a different call.
Startup output states both calls and explains this override.

Identity is validated before modem devices or arming. The scoped shell
launchers prevalidate before their CAT/audio setup and freeze the calls before
any channel wait. Normal completion uses the existing CW transmission path,
including tone, drive, maximum key duration and PACTOR peer timing guards.
On refusal or shutdown unwinding, the manual-identification warning names the
CW call; unwinding never starts another transmission. A longer CW call can
still exceed the configured maximum key duration and require manual ID.

An older modem snapshot without separate-identity support is refused by the
shell before CAT/audio setup. A qualified console plan with distinct calls also
checks that its snapshot modem imports the identity module before claiming or
preparing the rig. VARA validation follows its script's existing choice of its
own package tree; PACTOR and ARDOP validation follows their child import path.

The configuration-driven station CW interlock also uses `station.idcall`,
falling back to `station.mycall`, with validation before audio/PTT startup.
That process uses its configuration only, without `IDCALL`/`MYCALL` overrides;
its existing cadence and `id_mode = "none"` behavior are preserved.

Bare `tools/onair.sh ardop` host sessions, generic host `MYCALL` commands,
Sabir/Bateleur beacons, and waveform preview tools retain their existing
identity behavior. This change does not add closing CW to those paths.

## VarAC chat identification schedule

VarAC chat identifies `[station] id_interval_s` seconds after the first
transmission since the station last identified, at the first quiet point after
that, whatever the regulatory profile, and once at the end of a QSO (a link
that reached CONNECTED). The interval defaults to 540, inside §97.119's ten
minutes; 0 schedules no periodic ID. The closing ID follows `closing_cwid`
whatever the profile.

Any transmission that carries the station's call decodably identifies it once
it completes: CQ frames (including the plain beacon), advanced beacon text,
`/all` frames by their AX.25 source address, and the setup frame of a link we
call. Chat text naming the call, such as our opening `de W9SSJ <R+06>`, restarts
the clock once the host has it: VARA delivers queued text in order or
the link fails, but BUFFER names no write, so no report can say when the text
left. Text never stands in for the closing ID; the QSO's transmissions still owe
it. Connect requests and responses carry only a hash of the call, and ARQ data,
ACK, keepalive and disconnect frames carry none, so they do not identify the
station. With a distinct CW call, only frames and text naming that call count.

Inside a link the periodic ID is never CW, which would key over the peer's
over. It is our opening report sent again as chat text, `de CALL <R±NN>`, as
stock VarAC sends it: P1 session 6 holds N2KBE's `de N2KBE <R+06>` twice, 601 s
apart. It goes out in our turn, as all queued text does, behind any text still
queued, and waits for a signal report to carry. It is queued two minutes before
the interval ends, or half the interval when that is shorter, so that it is on
the air inside it; if the modem still holds data that long after it was queued, the operator is told to identify by
hand should it not have gone out. It also waits for a file, BBS or mail
exchange in progress to finish, so a transfer longer than the interval delays
it. Outside a link the ID is CW, at a quiet point on the frequency of the
latest transmission between CQ cycles. Retunes, band and slot changes, QSY, a
new CQ, the end of a beacon or of a CQ, and a call that never connected do not
trigger an ID. Quit sends a final CW ID when the host is open and the key reads
LOW. Otherwise, as on error unwinding, it reports the unidentified
transmissions for the operator to identify by hand.

The closing ID does not key over the peer's. When we close, ours goes out at
once on a clear channel. When the peer closes, ours waits for its ID, as stock
VarAC's receiving side does. Once reception is heard, ours waits for 0.5 s of
clear channel and goes out anyway at 8 s.

`[station] closing_cwid = false` turns off the end-of-QSO CW ID when the
digital and CW calls match; a distinct CW call keeps it on. The QSO's
transmissions stay on the `id_interval_s` clock and the quit ID still applies.
`[station] id_mode` does not apply to VarAC chat.
