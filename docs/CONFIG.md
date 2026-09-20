# The station file

One file, the whole station. Unknown keys are an error at load: a typo'd key is a
setting that silently did not apply, and on a transmitter that is the difference
between a hot input and a working one.

Everything is validated eagerly. Validating on first use means a station comes up,
listens for an hour, answers a call, and only then discovers it cannot key.

See `examples/station.toml` for a station template requiring your PTT
configuration and `examples/replay.toml` for one that cannot transmit.

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

**A port used by two protocols** is refused. Kestrel uses separate command
and data ports; Sabir uses one native host-interface port. Assign distinct
listening ports to every enabled protocol.

**`rig.ptt.backend` other than `"rts"`** is refused, and `"rigctld"` is refused by
name because it is the plausible wrong answer: it is not implemented, and this
station would key RTS anyway — a transmitter keyed by a route other than the
configured one.

**Missing `control` or `regulatory`** is refused. Neither has a default.

## What the loader cannot check

`rig.ptt.port` equal to the CAT port is **not** refused. There is nothing to compare
it against: rigctld holds the CAT device and does not say which one, so the guard
that used to sit here was testing every port against an empty string. It read as
protection and was not.

Two owners of one serial port is still a mistake, and it is caught by being said out
loud instead. `hfmodem rig --preflight` prints the line it is about to key and by
what method — `RTS on /dev/cu.usbserial-XXXXB1` — and the arm gate refuses a daemon
that owns PTT itself. Read that line; it is the check.

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

**The whole login path beyond that refusal has only ever been demonstrated under
an overridden SID.** The September 17 sessions that reached a `;PQ:` challenge,
answered it and had the answer judged — refused by name at one gateway, accepted
at another, which then took an outbound message — announced a registered
third-party client type, not `HFM-0.1`. So what is demonstrated is that this
code's `;PR:` response is correct and that its B2F exchange is well formed. What
is **not** demonstrated is that a production CMS will admit hfmodem's own SID,
and nothing short of registering it will settle that.

**The rest of that gateway's sentence is not advice you can take.** It ends
*"— use cms-z.winlink.org"*, which reads as a fix and is not one over the air.
`cms-z` is Winlink's development CMS, the one that skips the client-type
allowlist: ARSFI's own reference client picks it in debug builds and
`cms.winlink.org` in release builds, and only on its telnet path — the relay
path carries no CMS hostname at all. An RMS gateway chooses which CMS it relays
to, and a client over the air has no field in which to ask for one. The Winlink
Development Team's own client-onboarding instructions say as much from the other
side: testing against `cms-z` over RF means standing up a temporary RMS and
having its callsign whitelisted, *because there are not usually existing RMS with
access to cms-z*. A production gateway heard on the air cannot get there, and
nothing we transmit to it can change that. The message is telling a client author
to register the SID; it is telling an operator nothing.

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
