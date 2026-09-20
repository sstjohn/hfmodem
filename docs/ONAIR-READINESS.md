# On-air readiness — shrike, confirmed back

> This is the historical July–August 2026 experiment record. Present-tense
> claims below describe those sessions, not current readiness. For the September
> mail-delivery milestones and remaining limits, read [Development status](STATUS.md).
> PACTOR-3 mail sent through WS8EOC was confirmed received in Gmail by the
> operator on 2026-09-20; the unsuccessful early attempts below remain historical.

Everything below was checked on this machine, not asserted from memory.

## What was checked

**1. Exclusive CAT — confirmed required.** `ota.Rig` shells out to one-shot
`rigctl -r <serial>` and opens the serial port directly, bypassing rigctld. Two
processes cannot both own it. Measured right now, these hold the port and must be
stopped before handover:

```
rigctld -m 1036 -r /dev/cu.usbserial-XXXXB0        <- the FT-891
python /tmp/scanner.py                             <- this station's corpus
                                                      scanner, same device
```

(The scanner is a station-local script, in no tree at all; the record is here
because anything holding the port — whatever it is — takes the key with it.)

shrike will not start a rigctld of its own.

*The contradiction is settled, and both rules were right — they are about
different directions:*

| operation | path | why |
|---|---|---|
| SET / latching — `T 1`, `T 0`, `F`, `M` | one-shot `rigctl` | latches at the rig, survives process exit; the daemon's startup race is what silently dropped PTT at 38400 |
| READ-back — `f`, `m` | one-shot, after closing the interactive process | *Retested 2026-07-24 and this row is corrected: five consecutive one-shot reads returned 7100000 identically. The "CAT race" was contention with a concurrently running rigctld plus a shell quoting bug.* `get_freq` still tolerates `""`, because contention remains possible if something else opens the port. |

Recorded in `ota.Rig`'s docstring so it does not have to be relearned.

**1b. Keying is not on the CAT wire — changed 2026-07-28.** `T 1`/`T 0` no longer
become `TX1;`/`TX0;` on the FT-891. They set **RTS on the Standard port**
`/dev/cu.usbserial-XXXXB1`, the second port of the same CP2105, which is the
rig's PTT input; hamlib dispatches on the PTT type in its frontend, so the Yaesu
backend is never reached and keying is one ioctl. A CAT PTT change is a
transaction — write, wait 50 ms, read back, re-send while the rig withholds its
answer — and on 2026-07-28 the key stayed down across five bursts while the log
read `PTT keyed 0.998 s` every cycle.

That is a flag, not a rewrite: `--ptt-type {RTS,DTR,RIG}` and `--ptt-port`, both
defaulting from the rig's own entry (`RIGS`), with the PTT port derived from
`--serial` so one flag still names the station. The Xiegus keep `RIG`.

The row above needs one qualification for it, and it is smaller than it used to
read here. RTS is held by the **open port** rather than latched at the rig, so
where a CAT PTT certainly survives the process, RTS *can* fall with it. Whether
it does is a driver question this station has never answered — `HUPCL` is set for
it and the CP2105's behaviour on last close is unmeasured (`docs/STATION.md` has
the procedure that would settle it, and it needs no transmitter). What this file
said instead — "RTS drops when it exits or is killed, the direction is the safe
one" — was contradicted a week later by the station's own record: on 2026-08-04 a
session keying RTS through `ota.Rig` was stopped from outside and went on
transmitting on 10.1 MHz until the operator powered the radio down. `ota.Rig`
keys through a child `rigctl`, so the process holding that port was not the one
that died, and hamlib clears `HUPCL` on the ports it opens — so that is not the
experiment either. It is what the unqualified sentence was worth on the air.

The same change closed a second way to leave a transmitter keyed, independent of
the first: `rigctl`'s stdout was a pipe nobody read, and a full pipe stops it
reading its stdin, at which point every unkey in the program — the `finally`, the
watchdog, `Rig.stop` and its one-shot — was one blocking write that never
returned and never raised. Stdout is drained now, and no write outlives 0.25 s.

Both are measured without a radio, by `python -m hfmodem.tests.shrike.test_duplex`: one
stage keys against a rig 2.5 s slow to answer CAT and asserts that not one PTT
command crossed the CAT wire, another fills both of `rigctl`'s pipes and asserts
the unkey gets out regardless.

Three things still want the hardware, and none blocks: that `…B1` is the Standard
port (the CP2105 interface strings are in `ioreg`), that the rig's menu has PTT
enabled on it, and what the session's FIRST key-down costs — hamlib opens the PTT
port there rather than at startup. `--preflight` answers all three: it prints the
device and the method it is about to key with, and measures actuation against
this rig's 40 ms settle.

**2. Defaults that named a station — removed.** `--serial` was
`/dev/tty.usbserial`, then this station's own CAT port. `beacon` and `ota` now
require it outright (`ota` excepts `--list-devices` and `--dry-run`, which never
open a port); `onair` requires it too under `--transmit`, and its dry-run and
`--replay` paths ask for nothing because they never open a port. The launchers
take `CAT_PORT` from the environment and derive the keying line from it, with no
default anywhere — a shipped default is right on one machine and silently wrong
at the next rig. Find yours with `ls /dev/cu.*` on macOS,
`ls /dev/serial/by-id/*` on Linux. The other half of the old fix stands: `cu.*`
(callout) rather than `tty.*`, which blocks waiting on carrier, so a copy-pasted
invocation no longer hangs. `--dxcall` has no default at all, and each CLI's
exceptions are its own: `beacon` always requires it, `ota` excepts only
`--list-devices`, and `onair` excepts `--listen-only`, `--preflight` and
`--tune`, which call nobody. `onair` likewise resolves a frequency only under
`--transmit` — the QSY before keying is its sole consumer — so a bare
`--preflight` demands neither a callsign nor a dial, and a dry-run connect
demands only the callsign it renders.

**3. Codec gain — measured at 0.099**, confirming the report (~20 dB down, set
deliberately for corpus capture). Rather than rely on remembering to raise it,
`onair.py` now measures each received window and, below `RX_DEAF_RMS`, prints:

```
!! RX level 0.0021 RMS -- receiver may be DEAF (raise the codec input gain);
   a silent window here is NOT evidence the gateway stayed quiet
```

That is the "we were deaf" vs "no reply" distinction, made structural. **The gain
still needs raising before a real run** — it was left alone so the running scanner
keeps its calibration. Read/set it with `python tools/codec_gain.py --set <0..1>`
— a station tool: `tools/` does not cross into the public distribution, so from
a distribution set the codec's input level with the OS mixer instead.

**3a. VERIFIED AUDIO WORKING POINT (2026-07-25).** Everything before this line was
measured against a saturated input and is superseded. With the rig's **DATA OUT at
45/100** and **codec input gain 0.040**, the capture sits at **−18.1 dB rms, 0.00%
railed, 7.2 dB headroom** — matching kestrel's −17.8 dB / 0.00% reference on this
rig. Earlier runs were 13–31% railed at gain 0.510, which carries no recoverable
phase: those captures could never have decoded, whatever the far end did.

**4. WIP committed**, so an on-air run is reproducible and citable. *`shrike/ota_run.py` and `shrike/wspr_tx.py` have since been deleted:
nothing drove either, both shelled out to `rigctl` per command — the ~900 ms
pattern `ota.Rig` abandoned — and `wspr_tx.py` hardcoded the X6100's port and model
while the station moved to the FT-891. A dead script carrying a known-bad rig path
is a trap, not a record.*

*The watchdog trap is fixed at the root:* `ota._play` now **refuses** audio longer
than `--max-key` instead of keying and silently truncating it, and sizes the
watchdog to the actual audio length. Silent truncation is what produced the
phantom dead-antenna diagnosis; a clipped 110.6 s WSPR frame against a 40 s
default would have done it again.

## Dry-run log (no rig keyed)

```
$ python -m hfmodem.shrike.onair --mycall W9SSJ --dxcall W1AW --dial 7101000 \
      --reply-wav captures/reply_ack.wav --max-cycles 3
dry run -- rig not keyed; TX rendered to WAV, RX from --reply-wav

connecting W9SSJ -> W1AW ...
  TX[1] connect->W1AW  (2.0s)  -- dry run -> captures/onair/tx_01.wav
    RX  0.42  cs       ACK  (0 bit errors)
  cycle 1: state CONNECTED
** CONNECTED to W1AW **
```

The rendered TX decodes back through shrike's own receiver as
`###CONNECT: [Normal Call: W1AW]` — the waveform is proven without keying.

## Station checklist

| item | value |
|---|---|
| Callsign | **W9SSJ** (`--mycall W9SSJ`) |
| Rig / CAT | FT-891, `/dev/cu.usbserial-XXXXB0` @ 38400, PKTUSB |
| PTT | RTS on `/dev/cu.usbserial-XXXXB1` (same USB, Standard port) — derived from `--serial`, printed by `--preflight` |
| Audio | USB-DigiMode 3 ("USB Audio Device") |
| Power | **≤ 50 W** — never above without the operator present |
| Band | 40 m |
| Target | **WS8EOC** (EN72QQ, 274 mi, az 97°, Pactor 3/4) — dial **7.100.00**, centre 7.101.50 |
| Alternate | **W6IDS** (EM79NV, 432 mi, az 142°) — Pactor **2** on dial 7.060.00, Pactor 3 on 7.102.30 |

Frequencies are from `pat rmslist --mode pactor --band 40m --sort-distance`, and
they independently confirm dial = centre − 1500 Hz.

**Operator actions before handover:** tune the antenna/ATU for 40 m, raise the
codec input gain, stop the scanner and the FT-891 `rigctld`, and enable PTT on
the Standard port's RTS in the rig's menu. Then, reading the `keying` line of the
pre-flight before anything is armed:

```
python -m hfmodem.shrike.onair --transmit --rig ft891 \
    --serial /dev/cu.usbserial-XXXXB0 \
    --audio-out "USB Audio" --audio-in "USB Audio" \
    --mycall W9SSJ --dxcall N0CALL --dial 7100000
```

(`--dxcall N0CALL` is a placeholder that calls nobody; the checklist's target
goes there, with the dial that belongs to it.)

## What this run can and cannot show

It can show whether a real gateway **answers** shrike's connect, and it records
every reply window to `rx_NN.wav` — plus the whole session to `stream.wav`, keyed
stretches and all, on the same sample clock — with every decoded event logged, so
a failure is analysable rather than opaque. That answer is the one thing no
amount of offline work can produce.

One caution on those recordings, measured 2026-08-19 and superseding the earlier
reading of `0 xruns` as a clean capture: a starved interpreter drops blocks the
driver never flags, and the recorder writes from the same callback the clock fit
counts, so a session's audio can be short of the air while every index inside it
stays self-consistent. A station tool that does not cross into the public
distribution ranks the record by how much each recording is missing — 31 of 68
readings short, 51.8 s in total, with an xrun count of zero on all 68. It is a
floor, not a total; only a second receiver sees the whole hole.

A connect and a reply is the honest goal.

## Why the 2026-07-25 session produced no handshake — and what changed

Three faults, all on our side, all found afterwards from recordings rather than
from the air.

**1. We transmitted every control signal inverted.** `control_signal` emitted
1400 Hz for a one where a real station emits 1600 Hz, so shrike's acknowledgements
went out as the exact complement of the intended codeword. The connect survives
that mapping — it carries a sync byte and the parser accepts either polarity — but
a bare 12-bit codeword does not. A gateway would answer our call and never see a
reply it could act on, which is what was observed all evening.

**2. We could not read theirs.** The reply path reported presence only, floored at
120 ms (long enough to miss a real 115 ms signal), and compared mark/space
magnitudes over a full symbol, which cannot recover the bits.

**3. The relaxed matching radius then invented acknowledgements from noise.**
`CS_EXPECTED_MAX_ERRORS = 2` on a distance-12 code reported ACK, BREAK-IN,
CYCLE-TOG and SPEED-UP from a gateway while the operator heard nothing, and
declared the link up on the strength of it. A 30 s recording with the transmitter
idle produced two "control signals" at radius 2 and none at radius 0. **Every
CONNECTED reported in that session was false.**

All three are fixed and the fixes are anchored to real off-air recordings, not to
our own encoder: two bursts 2.5 s apart in one of them decode to 0x34B at zero
errors with the runner-up eight away. Measured end to end, the control
signal now reads correct 20/20 from 30 dB down to 3 dB SNR and **zero wrong words at
any SNR** — it reads the right word or stays silent.

**The acceptance criterion for the next session, agreed in advance:** a connect is
confirmed when the reply decodes at **zero** bit errors. Nothing else counts, and
the code's distance-8 structure is what makes that a reasonable bar rather than a
strict one.

**Still true that evening, and not fixed by any of this:** shrike had no
per-speed-level data frame, so a payload exchange with a real station was
impossible. The realistic goal was a link that comes up and is *seen* to come up
by both ends. (Superseded 2026-08-01, when the per-level frames landed — see
"What still blocks a data session" below.)

## The gateway answered — 2026-07-26 and since

That goal was met. WS8EOC answered a shrike call on 2026-07-26 — CS1 twice, then
CS4 held to its timeout, then its CW identification — with both sides caught by a
KiwiSDR 52 km out, so the record does not depend on this station's own T/R path.
Replayed through the current receiver, the reply decodes at zero bit errors,
which is the criterion above, and this station holds that replay. Sessions of
2026-08-03/04 repeated the connect against the same gateway, control signals at
zero bit errors in every one.

A data packet has since been acknowledged, in four sessions against two
gateways. KB5LZK on 2026-08-14, at 00:54Z and 00:57Z; WS8EOC on 2026-08-03 and
again at 02:09Z on 2026-08-15. In every one the peer alternated its codeword at
zero bit errors and our packet counter advanced two steps, `#1 -> #2` and
`#2 -> #3`. On a PACTOR-1 link that counter moves only when a decoded peer
codeword changes, so each step
is one acceptance; the 00:54Z run sent counters `#1, #2, #3` and repeated none
of them, so both of its acceptances were of a packet sent once. Two steps are
two acceptances and not three: the third packet's answer was the CS3 break-in
that reversed the grid, which is a changeover and not an acknowledgement. Only
the first of those three carried payload — the 7-byte greeting — and the two
behind it were idle frames.

Both steps are ack-caused. The verdict's `(b - a) % 4 == 1` test is also
satisfied by the `#3 -> #0` break-in reset, which is not an acknowledgement at
all, and the 2026-08-03 counters — `#1 x6, #2, #3 x2, #0 x4` — do contain one,
which is why that session's verdict reads "our counter reached #0" rather than
naming a step. Neither step counted here is that transition. The peer's own
transmissions corroborate the reading independently of anything we keyed:
`-005758` decodes the gateway's PACTOR-1 DATA frames carrying 8 bytes of
ASCII, `'RMS Trim'`.

What has not happened is a PACTOR-3 packet accepted. Each of those four sessions
keyed one SL3 packet — the upgrade offer `arq._on_ack` makes on the first
acknowledgement — and each was answered by a PACTOR-1 control signal, with the
peer's next transmission in PACTOR-1 too, which ruled the target out and dropped
the link back on the spot. Both halves of that have since changed and neither has
been on the air yet: the offer waits for payload to carry, so the one SL3 packet
those sessions keyed — an idle frame, the greeting having drained — would not go
out at all now, and the answer to an upgrade is given `arq.UPGRADE_SILENCE_CYCLES`
cycles before a PACTOR-1 frame is read as a verdict on it
(`ptc.PtcHost._stale_pactor1`). No session has decoded a PACTOR-3 control signal or frame from a
station at all. Nor has any link sustained the advance: three consecutive
per-cycle advances have never happened, on our counter or the peer's, and the
exchange ends after pkt#3, whose break-in bit invites the changeover. No message
has completed **over shrike** in either direction — shrike's buffer is a 7-byte
greeting that drains after three packets. Two figures in the paragraph above have
since been joined by a third, on 2026-08-19 at 22:10 against K4MSU: the gateway
broke in with `RMS Tri` in two windows two cycles apart, each decoding to
`Packet(payload=b'RMS Tri', status=0, breakin=True)`. Seven bytes is the whole
of it and not a fragment —
`pactor1.BREAKIN_FIELD[100]` is 7 — and the repeat is an ISS re-sending what its
IRS never acknowledged. `tests/shrike/test_holdbudget.py` holds the reading.

**The peer was not refusing the upgrade — 2026-08-26.** That reading was on offer
and it is wrong. Calling WS8EOC on 40 m this station drew the grant and then fell
back, and the gateway went on sending `0x59A` — a PACTOR-1 codeword in the answer
slot, which is the upgrade request — on **thirty-six consecutive cycles at zero
bit errors**, 48.352 s to 92.102 s of its own 1.24999 s raster, shift sense
alternating on every one. Twenty-one of those are after our fallback and sixteen
after the log printed that the peer never took the entry packet. It then signed
off in CW and left. A station asking on every cycle for forty-five seconds is not
refusing. All thirty-six read through this station's own control-signal decoder
out of a KiwiSDR recording made 103 miles away, so the reader is now validated
against a real gateway in a file this station's transmitter never touched.

Three faults of ours, none of them the decoder, kept them out. Our own carrier sat
on **twenty of the forty-one** codewords the gateway sent while we were on the
channel. The turnaround band is measured from our transmission's end as the
transmit grid counted it, in *delivered* samples, against onsets in *captured*
samples — and the capture lost 298 ms, so the answer walked out of a band that had
not moved. And the end-of-cycle flush keeps the last 750 ms of a window, which a
held window is mostly in front of. The session presented **ten** of the
thirty-six to the state machine on the air and now presents **sixteen**, which is
every one our own recordings hold: a codeword is read where the burst detector
already found a burst rather than where a window happened to end
(`tests/shrike/test_codeword_reach.py`). None of this disturbs the negative above
— WS8EOC never keyed PACTOR-3, at any speed level, at any point.

(One message has completed over **ARDOP**, in the other direction and on another
transport — WW2MI's CMS on 2026-08-19, message `LJ2AJE2IHO9B`, the session log
kept with the rest of that slot's record at this station. Nothing in this section
is about that path.)

## A peer took the payload, not just the packet — 2026-08-15

The stations above acknowledged. What their control signal cannot show is whether
the bytes arrived intact, because an acknowledgement is a codeword toggle and not
a receipt. An independent *peer* now shows that half.

Sailer HB9JNX's hf-pactor (`hfkernel/fsk/pactor.c` in the GPL `hf`/hfterm
package, cited throughout `docs/protocols/pactor/`) is a complete PACTOR-1 ARQ
implementation, and until now had only ever been read. Built from the Debian
snapshot archive's `hf` 0.8 and run as a PACTOR-1 responder over a lock-step
virtual channel, it took shrike's data packets and **handed the payload up to its
own host byte-exact** — hfkernel's own `ARQ_SLAVE_CONNECT` state message, then
`DATA_RECEIVE` carrying `1W9SSJ\r`, four times, with shrike's receiver out of the
loop entirely.

Two negative arms say it means something. The same run with the packet counter
frozen, and the same run with the data field mangled under the CRC the good field
computed, each brought the link up and then **sent CS4 forever without advancing**
— nothing delivered. Every control signal in every arm reads at zero bit errors
on two decoders sharing no code.

That refusal is the more useful half. It is the signature WS8EOC produced on the
air on 2026-07-27 — CS4 at zero errors, twenty cycles, no advance — which
`tests/shrike/test_p1peer.py` reads as a peer saying the packet does not decode.
It now reproduces on a bench in twenty seconds, and what converts it to an
acceptance is the thing that file names: the **per-cycle shift**. The peer
inverts once per 1.25 s cycle from the call it locked to, and reads each packet
in the shift its own cycle count names, so a transmission that never inverts has
exactly its even cycles taken — measured as four of eight, carrying counters
2, 0, 2, 0. Tracking the peer's parity is unfinished; its phase is readable from
the sense field the bench already records and simply is not fed back.

Two facts fell out, both about sustaining a link rather than starting one:

  * **CS4 is what a good 100 Bd packet earns**, not a complaint — after a correct
    packet it acknowledges and forces 200 Bd. A master that does not follow the
    speed change stalls a few packets in.
  * **The connect is marginal.** hfkernel's responder requires a Hamming-distance-0
    callsign match and intermittently read `W1AW` as `W1EW`. Whether that is
    shrike's connect frame or the bench's levels is unsettled and worth settling,
    because an SCS modem applies the same exact-match rule.

The peer runs outside this tree as a separate GPL program driven over a socket.
It is a virtual channel with no loss, no noise and no frequency offset: it settles
what the link layer does with a well-formed packet, and says nothing about the air.

## Cycle timing — superseded 2026-07-25

This section used to say the loop could not hold an ARQ cycle, because it
recorded for `--listen` seconds and only then decoded, costing ~7 s an iteration
against PACTOR's 1.25 s. That was true of the loop, never of the decoder, and the
decode figure quoted here (2.86 s for a 4 s window of real off-air PACTOR-3)
predated the receiver performance work by a wide margin.

Measured now: a 3.75 s off-air PACTOR-3 data cycle decodes in **77 ms**, 0.021×
real time. The session receiver is a single rolling decoder sliding every 0.25 s,
about a third of one core, and `packages/hfmodem/hfmodem/tests/shrike/test_session.py` is the standing
measurement — a connect is decoded **0.06–0.16 s before its burst ends**, against
0.09–1.19 s *after* under the old monitor-shaped window.

`--replay <wav>` drives the whole loop and FSM from a recorded capture, so the
session path is exercised on the corpus rather than by transmitting at strangers.
A replayed peer cannot react, so it answers whether an event reaches the state
machine and how late — not whether a handshake completes.

## What still blocks a data session

`packages/hfmodem/hfmodem/tests/shrike/test_qso.py` runs two shrike stations through rendered audio and the
real receiver: connect, acknowledge, data one way, changeover, data back, byte-exact
both directions. That is the rehearsal, and it is not interop — both ends share our
reading of the protocol, so a symmetric misreading is invisible. An independent
decoder is what settles that.

**The data frame is no longer the gap — 2026-08-01.** Each speed level has its own
frame geometry (`placement.SPEED_PATHS`, and `LONG_PATHS` for the 3.75 s cycle),
the field is the fixed size that level carries with no length byte of shrike's in
it, and `placement.link_packet` is what `onair.RadioTx` keys on a link. The
borrowed 24-byte header frame is gone.

Two kinds of evidence, kept apart:

* **Our own receiver** decodes every level byte-exact and decodes each one at its
  own level and no other (`tests/shrike/test_p3rx.py`). That says the decoder
  inverts the encoder and nothing more — both ends read one table, so a symmetric
  misreading is invisible to it.
* **An independent PACTOR-III monitor** reads what the transmit path renders:
  levels 3, 4, 5 and 6 from a standing start and levels 1 and 2 through a link
  already locked to a wider one, each read back at the level it was sent at,
  over four ARQ cycles apiece with the carrier swap alternating — against a
  negative arm, the same packets with the field mangled under a good CRC, from
  which it prints nothing (`tests/shrike/test_p3_oracle.py`). Two limits on how
  that is read. The match is on a **prefix**, not the whole field — 24 bytes in
  the cold arms (`_verdict`) and 20 in the locked one (`_verdict_locked`) —
  because the monitor drops trailing IDLE and expands a CR, so its rendering of a
  field is not byte-for-byte what was sent. And there is no link under any of it:
  the runner plays the rendered WAV into an ALSA `snd-aloop` cable and points the
  monitor's capture at the other end, with no rig, no PTT and no channel.

That second one is the load-bearing evidence, and its limit is that a monitor
*reads*. It does not run ARQ, so it has never acknowledged a shrike packet, and a
frame a monitor prints is not proof an SCS modem in a session will accept it. Only
the air settles that.

What is left, then, is not the frame:

* **No station has accepted a PACTOR-3 packet.** PACTOR-1 data packets have been
  acknowledged — four sessions against two gateways, above — but every SL3 offer
  was answered in PACTOR-1 and the link fell back, and no session has decoded a
  PACTOR-3 control signal or frame from a station. That is an on-air result
  nothing offline can produce.
* **A link cannot choose the 3.75 s cycle.** Its geometry renders and decodes
  — `placement.LONG_PATHS` through `data_packet` — but `link_packet` indexes the
  short-cycle table alone, so no session can select one; `arq.py` acknowledges
  CS6 and logs that the cycle is not built. That is the right answer for now:
  the request rides the ISS's own status bit 5 and shrike never sets it, and
  nothing in the corpus carries a long-cycle field any decoder reads, so the
  long geometry rests on arithmetic rather than on a signal.
