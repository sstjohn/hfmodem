<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Running the station

The station service takes the shared rig claim before opening the radio's CAT,
keying line and sound devices. It releases the claim after checked teardown.
See the [station service guide](README-station.md) for startup, ownership,
STOP and handoff to the legacy tools.

For a first PACTOR-3 or VARA Winlink call, use `hfmodem winlink` and the
[operator quickstart](CONSOLE.md#install-and-receive-winlink-mail). It guides
radio and identity setup without transmitting. The sections below describe
manual station operation and its interlocks.

## Rig profiles

The station uses a Yaesu FTDX10 with its built-in USB audio codec. The FT-891
remains supported with an external audio interface through the same rig-profile
system. X6100 and G90 support is planned; their profile entries have unmeasured
timing and incomplete readback qualification.

Select the current profile with `HFMODEM_RIG=ftdx10`
or a per-modem tool’s `--rig ftdx10` option. Explicit options take precedence over the
environment. The shared resolver retains `ft891` as its fallback, so select the
FTDX10 explicitly. Station files supply rig-profile overlays and audio settings;
inspect them with `hfmodem config` before starting the station.

The operator chooses the station settings and when to transmit. The software
provides the modem, checks and controls used to run that station.

For the operator console, passive monitoring, ALE scanning and selcall, start
with the [console guide](CONSOLE.md), [monitor guide](console-monitor.md) and
[HFLINK guide](console-hflink.md). Passive single-mode monitoring permits
manual radio retuning and preserves the receive filter, including a 4 kHz
voice passband. Active ALE scanning owns tuning until you stop its scan.

## Before anything transmits

    hfmodem config station.toml     what the file actually means
    hfmodem devices                 what sound cards this machine has
    hfmodem rig station.toml        talk to the radio, without keying it

`config` prints the derived dial, whether transmit is enabled, the SID announced
to a Winlink gateway, and the automatic-control setting derived from
`listen = true`. See `[station] client_sid` in CONFIG.md for the SID format.

The station drives the supported Yaesu radios through native serial CAT
(`[rig] cat = "yaesu"`); no Hamlib daemon is needed for that path. The
`rigctld` backend remains available for other rigs. PTT uses the separate
keying line, not CAT commands. Legacy shell tools still use Hamlib.

## The interlocks, and why each exists

**`transmit = false`** prevents keying. Receive ownership opens capture audio
and, when configured, CAT for readback, but never the PTT line or playback
device. Attaching does not change ownership; a newly launched detached service
starts observing, with no devices open.
For development without hardware, use the replay configuration in
`examples/replay.toml`.

**`control`** has no default. `local` means an operator is at the control point;
`automatic` means the station answers on its own, which is what `listen = true`
amounts to. The profile checks use this setting.

**`regulatory`** selects the software's emission-checking profile and has no
default. `part97` requires a licence-class setting; `unregulated` bypasses the
profile checks and requires a stated reason. These are limits the operator
chooses to apply.

**Identification is an interlock rather than a request.** When one is due, the
station refuses to carry traffic until the callsign has gone out. A priority scheme
would starve exactly the station that is busiest, which is the one most likely to
owe a callsign.

## Bringing it up

    hfmodem station station.toml

With protocols enabled, foreground startup takes ownership in the mode allowed
by `transmit`. Transmit ownership includes a brief keying proof and preflight
before lanes and host servers start. With no protocols enabled, the live
service stays observing. Use `--detach` to start observing regardless of the
protocol settings; see the [service guide](README-station.md).

## Arming

`hfmodem rig station.toml --arm` proves the keying line reaches the radio. This
transmits briefly, so it requires `transmit = true` and refuses without it.

The arm gate checks the configured radio identity, reads the dial and checks
whether the radio is already transmitting before proving the keying line.
Station ownership and the standalone `hfmodem rig` command both drive the
configured CAT backend: the radio's own serial CAT for `cat = "yaesu"`, and
rigctld for `rigctld` and `civ`, which needs that daemon running and checks its
configuration as well.

`PTT UNPROVEN` in the report means the line was not tested, not that it failed.

## What is not proven

The keying line drops when the process dies **if** the driver honours `HUPCL` on
last close. That is set correctly and has not been measured on this hardware, and
it cannot be measured with a pseudo-terminal — Darwin returns `ENOTTY` for
`TIOCMGET` on both ends of one. Proving it needs a second USB-serial adapter with
its CTS jumpered to the keying line's RTS, read from another process.

Until then, treat process death as a hope rather than a guarantee. The watchdog,
the `finally`, and the panic path are all measured; that one is not. Two files
used to say otherwise — `docs/ONAIR-READINESS.md` and this station's operating
log, with `shrike/ota.py` and `shrike/onair.py` behind them — and the tree
therefore asserted both answers at once. They now say what this section says.

The one observation on file points the wrong way. On 2026-08-04 a shrike session
keying RTS was stopped from outside and kept transmitting on 10.1 MHz until the
operator powered the radio down. That was not the experiment — that version keyed
through a child `rigctl`, so the process holding the port was not the one that
died, and Hamlib cleared `HUPCL` on the ports it opened — but it is the only time
the question has been put to this station in anger, and the answer was no.

### The measurement, which needs no transmitter

A serial line's state after a process dies is observable with a second adapter
and a jumper. Do it with the radio switched off or its PTT lead unplugged;
nothing below puts anything on the air.

1. Wire a second USB-serial adapter as the witness. Its **CTS** (an input) to the
   keying port's **RTS**, and their grounds together — DE-9 pins 8, 7 and 5, or
   the labelled pads on a bare CP210x breakout. `PTT_PORT` keeps naming the
   keying line; call the witness `$WITNESS`.

2. Record the witness at **1 kHz**, retaining every CTS sample with its
   monotonic timestamp. A gap over **2 ms** invalidates pulse assertions;
   printing only changes or polling every 10 ms cannot establish them.
   Observe a LOW baseline before opening the keying port.

   Before each run, validate the witness with commanded 20, 10, 5 and 2 ms
   pulses from `RtsPtt`. Use a second witness channel on a source with known
   timing, or an oscilloscope capture, to establish physical edge times;
   the driving ioctl's timestamp is not a physical edge reference. Retain
   the reference capture with the samples. Measure rising and falling
   reporting delays separately, the shortest reported pulse, and blindness
   to shorter pulses.

   Bound a detected pulse by the span from the sample before its first HIGH
   to the sample after its last HIGH, plus the measured delay asymmetry.
   For an undetected pulse, use the largest gap between surrounding LOW
   samples plus blindness and reporting delay. Compare that upper bound
   with the 20 ms initial-open pulse limit. Unknown bounds, or insufficient
   witness resolution for the limit, mean **unresolved**, not a pass.

3. Key the line from a process you can kill, and note the pid it prints:

   ```
   ./.venv/bin/python - "$PTT_PORT" <<'EOF'
   import os, sys, time
   from hfmodem.core.ptt import RtsPtt
   p = RtsPtt(sys.argv[1])
   p.open()
   p.assert_(True)
   print("keyed, pid", os.getpid(), flush=True)
   time.sleep(600)
   EOF
   ```

   **The witness must go HIGH here.** If it does not, the jumper is on the wrong
   pin and every result after this is a false negative — which is the failure
   this whole section is about, so establish the positive first.

4. `kill -9 <pid>` from a third shell, and read the witness.

   * CTS goes low: record the measured delay from the kill and the sampling
     uncertainty; this is evidence for this adapter, line and scenario.
   * CTS stays HIGH: record that process death did not lower the line.
   * Missing samples or unvalidated witness timing: record unresolved.

Three runs, because they are three different mechanisms and only the first is
`HUPCL`: as above; again with a child of the holder also holding the port open,
which is the "last close" case `O_CLOEXEC` exists to prevent; and once more by
pulling the adapter's USB plug while keyed, which `HUPCL` cannot reach at all.

Jumper the witness to **DTR** and repeat if a rig here ever keys on that line.
`core.ptt.drop_rts` already takes both down and reads both back for the same
reason: which line keys a given rig is a measurement, not an assumption.

Repeat on macOS and Linux for each adapter and keying line. Publish the date,
platform, adapter, line, scenario and measured death-unkeys result (`true` or
`false`), with timing evidence. A measured `false` is still a result; an
unperformed or unresolved measurement does not complete the hardware gate.
No such measurement is recorded here yet.
