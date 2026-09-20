# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Serving each protocol's host dialect on its own port.

The four modems present four different control surfaces, and that is deliberate:
an application that drives VARA expects VARA's two-socket ASCII, and one that
drives an SCS controller expects WA8DED hostmode inside CRC framing. Normalising
them would mean no existing application could talk to this station at all.

So the dialects stay four, and what unifies them is *where they run* — one
process, one radio underneath, each on the port its callers already use. That is
what makes a station with four modems enabled indistinguishable, from an
application's point of view, from four stations.

The servers also disagree about how they are named into life — one offers
`start_background()`, another a `start()` — and a station should not have to know
which. What they may not disagree about is that binding happens before the call
returns and that there is a way to stop them again: a port this station could not
take, or a lane it could not put down afterwards, has to be visible here rather
than in a warning on a run that otherwise looks well.

A dialect that will not start is not fatal. Its protocol keeps receiving, the
other three keep serving, and the failure is reported rather than taking the
station down with it — the containment property four separate processes gave away
for free.

What a lane may NOT do is look like a radio it has not got. Two of these servers
default to a loopback modem — a link fabricated to itself, so an application
attaches, is told CONNECTED, and reads back its own bytes with nothing keyed —
and an application cannot tell that from a station on the air, because from the
socket it is not different. So `Served.keys` records whether an attached
application can reach a transmitter, every lane says which it is on the line it
comes up on, and a dialect with no server here is named as absent rather than
disappearing into a `continue`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Served:
    name: str
    server: object
    port_attrs: tuple[str, ...]
    keys: bool
    """Can an application attached here put anything on the air?

    False is the loopback: the dialect is whole, the ports are real, and the far
    end of every connection is this process. It travels with the server because
    the moment a real modem is wired in it has to move with it — a lane that
    says it keys and does not is the failure this field exists to prevent.
    """

    @property
    def ports(self) -> tuple[int, ...]:
        """Read off the server, so a config that asked for 0 reports what it got."""
        return tuple(getattr(self.server, a) for a in self.port_attrs)


def _start(server) -> None:
    """Bind and accept, on whichever of the two names this server uses.

    Both bind before returning, so a port already taken raises here and the lane
    never reaches `served`. A server the station cannot stop is one it does not
    start: the alternative is a lane that outlives every reference to it.
    """
    if not callable(getattr(server, "stop", None)):
        raise TypeError(f"{type(server).__name__} offers no way to stop it")
    for method in ("start_background", "start"):
        fn = getattr(server, method, None)
        if callable(fn):
            fn()
            return
    raise TypeError(f"{type(server).__name__} offers no way to start it")


def build(cfg, *, station=None, log=print) -> dict[str, Served]:
    """Start a host server for every enabled protocol that has one.

    Returns what actually came up, which is not necessarily what was asked for.

    `station` is optional because the ports are worth testing on their own, and a
    dialect that can be served without one says so by not needing it. sabir needs
    one: its session runs here rather than elsewhere.
    """
    out: dict[str, Served] = {}
    for name, p in cfg.protocols.items():
        if not p.enabled:
            continue
        try:
            served = _build_one(name, p, station)
        except Exception as exc:            # noqa: BLE001
            log(f"{name}: host dialect not served ({exc})")
            continue
        try:
            _start(served.server)
        except Exception as exc:            # noqa: BLE001
            log(f"{name}: host dialect would not start ({exc})")
            continue
        out[name] = served
        radio = "" if served.keys else \
            "  -- NO RADIO: an application attaching here does not reach the air"
        log(f"{name}: {p.host} on "
            f"{', '.join(str(x) for x in served.ports)}{radio}")
    return out


def _build_one(name: str, p, station=None) -> Served:
    if name == "vara":
        # LOOPBACK. `VaraServer` takes a `modem_factory` and there is nothing to
        # hand it: kestrel's real core has no constructor that binds it to this
        # station's air and arbiter, and `run_server.py` has no --modem either.
        from hfmodem.kestrel.host.server import VaraServer
        s = VaraServer(cmd_port=p.cmd_port, data_port=p.data_port)
        return Served(name, s, ("cmd_port", "data_port"), keys=False)
    if name == "ardop":
        # LOOPBACK, and this is the one with a real modem already written:
        # `besra.host.run_server --radio` builds a `BesraModem` on its own rig.
        # It owns that rig, which is the rig this station owns, so the wiring is
        # a shared-radio question rather than a missing argument.
        from hfmodem.besra.host.server import HostServer
        s = HostServer(control_port=p.cmd_port, data_port=p.data_port, quiet=True)
        return Served(name, s, ("control_port", "data_port"), keys=False)
    if name == "sabir":
        # A fresh session core attaches to the station's one Sabir audio lane.
        air = getattr(station.lanes.get("sabir"), "air", None) if station else None

        def _factory(*_a, **_kw):
            if air is None:
                raise RuntimeError(
                    "sabir needs a station: its session runs on the air rather "
                    "than being fed frames, so there is nothing to mount without "
                    "a lane and an arbiter.")
            from hfmodem.sabir.host import SabirModem
            return SabirModem(air)

        from hfmodem.sabir.host.hostlink import HostLinkServer
        s = HostLinkServer(_factory, port=p.port)
        return Served(name, s, ("port",), keys=air is not None)
    if name == "pactor":
        # An SCS controller is a serial device, so this one is a pty rather than a
        # socket: the application opens it exactly as it would open a PTC-IIIusb.
        # `PtcHost` is the command surface; putting it behind a pty is what
        # `shrike.ptc.main` does, and the station does not own that yet.
        raise NotImplementedError(
            "the SCS dialect is a pty and the station does not own one yet: "
            "shrike.ptc.main joins PtcHost to a pty with a simulated peer, "
            "shrike.onair joins PtcHost to the radio with no pty, and nothing "
            "joins the pty to the radio")
    raise LookupError(f"no host dialect is implemented for {name}")


def stop_all(served: dict[str, Served]) -> None:
    for s in served.values():
        s.server.stop()
