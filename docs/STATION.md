# Running the station

One process owns the radio: the CAT port, the keying line, and the sound card. It
hosts whichever protocols are enabled, each still serving its own host interface on
its own port. Contention is impossible by construction rather than by discipline —
which matters, because a transmitting radio mutes its own receiver, and one modem
transmitting while another recorded has already invalidated eight sessions that
were reported as a silent band.

## Before anything transmits

    hfmodem config station.toml     what the file actually means
    hfmodem devices                 what sound cards this machine has
    hfmodem rig station.toml        talk to the radio, without keying it

`config` prints the derived dial, whether transmit is enabled, the SID announced
to a Winlink gateway, and the automatic-control setting derived from
`listen = true`. See `[station] client_sid` in CONFIG.md for the SID format.

The station's CAT is a socket to a `rigctld` you start, so Hamlib has to be on
`PATH` — `command -v rigctl rigctld` before anything else. A build of your own
goes on `PATH` via `HAMLIB_BIN`.

## The interlocks, and why each exists

**`transmit = false`** is a hard interlock, not a preference. With it set, nothing
keys, no CAT port is opened and no keying line is held — a receive-only station has
no business touching either. It is the default in `examples/replay.toml`, which is
the configuration to develop against.

**`control`** has no default. `local` means an operator is at the control point;
`automatic` means the station answers on its own, which is what `listen = true`
amounts to. The profile checks use this setting.

**`regulatory`** selects the software's emission-checking profile and has no
default. `part97` requires a licence-class setting; `unregulated` bypasses the
profile checks and requires a stated reason.

**Identification is an interlock rather than a request.** When one is due, the
station refuses to carry traffic until the callsign has gone out. A priority scheme
would starve exactly the station that is busiest, which is the one most likely to
owe a callsign.

## Bringing it up

    hfmodem station station.toml

In order: config, devices, audio, channel sense, arm, lanes, host servers. Lanes
before servers is not cosmetic — a client that connects to a modem whose receiver
is not running gets a station transmitting into a channel it cannot hear.

## Arming

`hfmodem rig station.toml --arm` proves the keying line reaches the radio. This
transmits briefly, so it requires `transmit = true` and refuses without it.

Two things the arm gate does. It asks rigctld how it is
configured to key (`ptt_type`) rather than testing by keying — on a daemon set up
for CAT PTT, finding out the other way genuinely puts an unidentified carrier on
the air. And it re-reads the dial, because the dial does not stay where you left
it.

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
operator powered the radio down. That is not the experiment — `ota.Rig` keys
through a child `rigctl`, so the process holding the port was not the one that
died, and hamlib clears `HUPCL` on the ports it opens — but it is the only time
the question has been put to this station in anger, and the answer was no.

### The measurement, which needs no transmitter

A serial line's state after a process dies is observable with a second adapter
and a jumper. Do it with the radio switched off or its PTT lead unplugged;
nothing below puts anything on the air.

1. Wire a second USB-serial adapter as the witness. Its **CTS** (an input) to the
   keying port's **RTS**, and their grounds together — DE-9 pins 8, 7 and 5, or
   the labelled pads on a bare CP210x breakout. `PTT_PORT` keeps naming the
   keying line; call the witness `$WITNESS`.

2. Watch it. This only ever reads:

   ```
   ./.venv/bin/python - "$WITNESS" <<'EOF'
   import fcntl, os, struct, sys, termios, time
   fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
   was = None
   while True:
       bits, = struct.unpack("i", fcntl.ioctl(fd, termios.TIOCMGET, struct.pack("i", 0)))
       cts = bool(bits & termios.TIOCM_CTS)
       if cts != was:
           print(f"{time.time():.3f} CTS {'HIGH' if cts else 'low'}", flush=True)
           was = cts
       time.sleep(0.01)
   EOF
   ```

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

   * CTS low within a few milliseconds of the kill: the driver honours `HUPCL`
     on last close, the fail-safe is real, and this section can be rewritten as
     a measurement with the date on it.
   * CTS still HIGH: process death does not lower this line, and every path that
     leans on it — the SIGKILL in `hfhost.supervisor`, the last line of
     `shrike/onair._unkey_on_signal`, and whatever reap a launcher does — is
     leaning on nothing.

Three runs, because they are three different mechanisms and only the first is
`HUPCL`: as above; again with a child of the holder also holding the port open,
which is the "last close" case `O_CLOEXEC` exists to prevent; and once more by
pulling the adapter's USB plug while keyed, which `HUPCL` cannot reach at all.

Jumper the witness to **DTR** and repeat if a rig here ever keys on that line.
`core.ptt.drop_rts` already takes both down and reads both back for the same
reason: which line keys a given rig is a measurement, not an assumption.
