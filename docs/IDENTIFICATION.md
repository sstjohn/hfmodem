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
