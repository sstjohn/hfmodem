<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Station service and ownership

The operator chooses when the station takes the radio and when to hand it
back. Attaching to the service does not take hardware. Ownership takes the
shared rig claim before opening devices; another holder must release it first.
For radio setup and the keying-line measurement, see [STATION.md](STATION.md).

## Start or attach

```sh
hfmodem station station.toml --detach
```

This serves in `observing` mode on `127.0.0.1`, using `[station] port`
(default 8737). Run by hand, `--detach` stays in the foreground; `attach`,
below, is what starts it in a separate session, with output in `station/`
under `HFMODEM_DATA_DIR` (default `operator-data`). A relative data directory
resolves against the directory `attach` is called from.

In a checkout that has `tools/desk`, the service also serves the desk's
logbook face on the same port.

Without `--detach`, a live station with enabled protocols immediately takes
ownership: transmit ownership if `[station] transmit = true`, receive-only
otherwise. With no protocols enabled it stays observing. Transmit ownership
briefly keys the radio for its proof and preflight measurements.
`hfmodem station station.toml --preflight` owns, measures, releases and exits;
it cannot be combined with `--detach`. Measurement needs `transmit = true` on a
live radio. With `transmit = false`, or with replay audio, it takes receive
ownership, measures nothing and exits 0.

The console provides ownership commands:

```sh
hfmodem console attach station.toml
hfmodem console own station.toml --receive
hfmodem console release station.toml
```

Omit `--receive` to request transmit ownership. The Python client provides
the same operations:


```python
from hfmodem.station.client import attach

station = attach("station.toml")
print(station.state())
```

`attach` connects to the service for that canonical station filename, or
launches `hfmodem station <file> --detach` from `PATH` if the connection is
refused. It waits up to 20 seconds by default. A different station file or
foreign service on the port is a conflict. Attaching to a `failed` station
still succeeds so you can inspect its diagnostics; it does not restore it.

The client returns an object on success; failures raise `ClientError`
subclasses with an `exit_code`. These are the client's codes, which
`hfmodem station stop` also returns:

| Code | Meaning |
|---|---|
| 0 | Attached; no hardware was taken by attaching. |
| 1 | Other client or command failure; read the diagnostic. |
| 3 | Server startup found the same station already running. `attach` connects to that server and reports success instead. |
| 4 | Port held by a foreign service or a different station, or its holder could not be validated. |
| 5 | An ownership command was refused because another holder has the rig claim. |
| 6 | Client launch failed or timed out; the diagnostic names the launch log. |

The server process (`hfmodem station FILE`) exits 0 on a requested shutdown,
1 when the radio or audio device is not usable or teardown could not finish,
2 for a configuration error or a replay with no protocols enabled, 3 or 4 when
the port is already taken as above, 6 for any other startup error (a foreground
start refused by another holder of the rig claim exits 6), and 128 plus the
signal number after SIGINT or SIGTERM.

## Take and release the radio

Use the attached client for these operations:

```python
station.own(receive=True)
print(station.state())
station.release()
```

Receive ownership takes the claim, opens capture audio and configured CAT
readback, and never opens the keying line or audio output. To take transmit
ownership, first release receive ownership, then explicitly call:

```python
station.own()
```

This briefly keys for the proof and preflight when `transmit = true`.
With `transmit = false`, even `own()` takes receive-only ownership. Changing
ownership mode requires a release first. A competing transition is refused
with `transitioning`; inspect the state and retry after it finishes.

| `service.state` | What it means |
|---|---|
| `starting` | The listener is up; startup has not finished. |
| `observing` | The service is available and owns no hardware. |
| `owning` | Transmit ownership is established. Check `service.ready` before using the enabled hosts. |
| `owning-receive` | Receive-only ownership is established. |
| `releasing` | Devices, lanes and hosts are being shut down. |
| `failed` | A failure needs attention; inspect `service.reason`, `service.failed` and `rig.retired`. Do not treat this as a handoff. |

A keying line that cannot be confirmed low at release leaves the service
`failed` and running, with the claim retained. If a host, lane or device does
not stop, the service unkeys and exits instead.

`service.ready` is true only when all requested lanes and hosts are ready.
Station lanes cover PACTOR, VARA, ARDOP, sabir, bateleur, ALE, selcall and
Robust Packet. Failed lanes or hosts appear in `service.failed` and leave
readiness false. Host choices and limits are in the
[compatibility guide](HOST-COMPATIBILITY.md).

`release()` stops activity, checks host and lane termination, closes the
radio and audio, and releases the claim last. A held keying line must be
read low before the service returns to `observing`. The listener stays available
through a successful release, so another tool can then take the radio.

## Control and records

Observers can read state and events without taking control. A client may request
an exclusive control session with `station.command("control", wait=True)`.
Subsequent commands carry its `control_id` and `generation`; heartbeats renew
its 30-second lease. STOP and shutdown remain available to other clients.
With no active control session, ordinary ownership commands remain available.

The service retains session state and records under the operator data directory.
The logbook shares the station's HTTP port. See the
[console guide](CONSOLE.md#stop-records-and-replay) for replay and static export.

## STOP and shutdown

```python
station.command("stop", wait=True)
```

STOP bypasses the ordinary command queue and requests that transmission stop.
During acquisition it cancels ownership. Its acknowledgement means the stop
was requested, not that the line has already been confirmed low. STOP does
not release normal ownership or exit the service; use `release()` for a
checked handoff.

To release and end the service:

```sh
hfmodem station stop station.toml
```

`station.command("shutdown", wait=True, timeout=45)` requests the same
shutdown through the client. Keep the station filename explicit when more
than one configuration is in use.

## Legacy shell cleanup

From the checkout, with the same station file selected:

```sh
HFMODEM_STATION=station.toml tools/onair.sh down
```

`down` asks a responding station to release and leaves its listener running.
It never signals the station. An unresponsive station, an unidentified device
holder or incomplete process information blocks cleanup and is reported.
A live cooperating claim holder is left alone: interrupt that holder, then
run `down` again. Legacy launchers also run cleanup under their inherited
claim when they finish.

| Exit | Meaning and next step |
|---|---|
| 0 | Require `CLEANUP COMPLETE` as well before treating cleanup as complete. |
| 2 | Cleanup mutex could not be acquired. Read the diagnostic identifying cleanup or admission still in progress; resolve that holder before retrying. |
| 7 | A live cooperating holder owns the claim. Interrupt the named holder and retry. |
| Other nonzero | Cleanup failed or was interrupted; retain the diagnostics. |

`PTT LINE low` is the line readback, not a cleanup verdict. `high` or
`unreadable` blocks completion; `none` means no PTT endpoint is configured.
`CAT ok`, `CAT failed` or `CAT none` reports the CAT check, with `none` when
CAT is disabled. `CLEANUP COMPLETE` appears only after the configured checks
pass and the evidence is written. A low line followed by a failed CAT check,
or a marker followed by a nonzero exit, is not completed cleanup.
