# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""External host endpoints for the station's protocol lanes.

Binding completes before a host is reported as served. A failed host is
reported while its lane continues receiving. ``Served.keys`` distinguishes
ports attached to a radio lane from standalone loopback modems.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
import threading
import time

from hfmodem.core.operations import LeaseTimeout


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
    endpoint_attrs: tuple[str, ...] = ()

    @property
    def ports(self) -> tuple[int, ...]:
        """Read off the server, so a config that asked for 0 reports what it got."""
        return tuple(getattr(self.server, a) for a in self.port_attrs)

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(str(port) for port in self.ports) + tuple(
            str(getattr(self.server, attr)) for attr in self.endpoint_attrs)


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


def build(cfg, *, station=None, log=print, failed=None, retain=None) -> dict[str, Served]:
    """Start a host server for every enabled protocol that has one.

    Returns what actually came up, which is not necessarily what was asked for.

    `station` is optional because the ports are worth testing on their own, and a
    dialect that can be served without one says so by not needing it. sabir needs
    one: its session runs here rather than elsewhere.
    """
    out: dict[str, Served] = {} if retain is None else retain
    for name, p in cfg.protocols.items():
        if not p.enabled:
            continue
        if station is not None and name not in station.enabled:
            continue
        try:
            served = _build_one(name, p, station)
        except Exception as exc:            # noqa: BLE001
            log(f"{name}: host dialect not served ({exc})")
            if failed is not None:
                failed.append({"name": name, "why": str(exc)})
            continue
        out[name] = served
        try:
            _start(served.server)
        except Exception as exc:            # noqa: BLE001
            log(f"{name}: host dialect would not start ({exc})")
            stop_all({name: served})
            del out[name]
            if failed is not None:
                failed.append({"name": name, "why": str(exc)})
            continue
        out[name] = served
        radio = "" if served.keys else \
            "  -- NO RADIO: an application attaching here does not reach the air"
        log(f"{name}: {p.host} on "
            f"{', '.join(served.endpoints)}{radio}")
    return out


def _build_one(name: str, p, station=None) -> Served:
    if name == "rpr":
        if p.host not in ("kiss", "tracker", "tracker-managed", "agwpe"):
            raise NotImplementedError("rpr mail host is not implemented")
        from hfmodem.brubru.host.server import KissServer
        link = station.lanes.get("rpr") if station is not None else None
        if link is None:
            raise ValueError("rpr KISS needs a station receive lane")
        if p.host in ("tracker", "tracker-managed", "agwpe"):
            from hfmodem.brubru.host.runtime import PacketRuntime
            from hfmodem.brubru.host.connected_server import TrackerServer, AgwServer
            from hfmodem.station.contact import ContactOwner
            arbiter = getattr(getattr(link, 'station', None), 'arbiter', None)
            runtime = PacketRuntime(
                link, mycall=station.cfg.station.mycall or 'NOCALL',
                full_scale_mv=p.options.get('drive_reference_mv', 3000),
                contact=ContactOwner(arbiter, 'rpr') if arbiter is not None else None)
            if p.host == 'tracker-managed':
                from hfmodem.brubru.host.managed_tracker import tracker_broker
                server = tracker_broker(runtime, secret=_managed_secret(p), port=p.port)
                return Served(name, server, ('port',),
                              keys=station.rig is not None and station.arbiter is not None)
            if p.host == 'tracker':
                server = TrackerServer(runtime, endpoint=p.pty)
                return Served(name, server, (),
                              keys=station.rig is not None and station.arbiter is not None,
                              endpoint_attrs=('endpoint',))
            server = AgwServer(runtime, port=p.port)
        else:
            server = KissServer(link, port=p.port)
        return Served(name, server, ("port",),
                      keys=station.rig is not None and station.arbiter is not None)
    if name == "vara":
        from hfmodem.kestrel.host.server import HostSession, VaraServer
        from hfmodem.kestrel.host.modem_core import LoopbackModem
        link = station.lanes.get('vara') if station is not None else None
        kwargs = {'modem_factory': link.runtime.modem_factory} if link is not None else {}
        if p.host == 'vara-managed':
            from hfmodem.core.managed_host import ManagedHost
            factory = kwargs.get('modem_factory', LoopbackModem)
            def serve_pair(command, data):
                session = HostSession(command, data, factory())
                try:
                    session.run()
                finally:
                    session.close()
                    session.ensure_retired()
            s = ManagedHost(serve_pair, secret=_managed_secret(p), port=p.port)
            return Served(name, s, ('port',), keys=link is not None
                          and station.rig is not None and station.arbiter is not None)
        s = VaraServer(cmd_port=p.cmd_port, data_port=p.data_port,
                       kiss_port=getattr(p, 'kiss_port', 0) or None, **kwargs)
        return Served(name, s, ("cmd_port", "data_port"),
                      keys=link is not None and station.rig is not None
                      and station.arbiter is not None)
    if name == "ardop":
        from hfmodem.besra.host.server import HostServer
        link = station.lanes.get('ardop') if station is not None else None
        s = HostServer(control_port=p.cmd_port, data_port=p.data_port, quiet=True,
                       modem=link.modem if link is not None else None)
        if link is not None:
            # The production core begins with listening disabled. Report that
            # effective value rather than the standalone loopback's default.
            s._cfg['LISTEN'] = 'FALSE'
        if p.host == 'ardop-managed':
            from hfmodem.core.managed_host import ManagedHost
            broker = ManagedHost(s.serve_attached, secret=_managed_secret(p), port=p.port,
                                 start_owner=s.start_attached, stop_owner=s.stop)
            return Served(name, broker, ('port',), keys=link is not None
                          and station.rig is not None and station.arbiter is not None)
        return Served(name, s, ("control_port", "data_port"),
                      keys=link is not None and station.rig is not None
                      and station.arbiter is not None)
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
    if name in ("bateleur", "ale", "selcall"):
        link = station.lanes.get(name) if station else None
        if link is None:
            raise RuntimeError(f"{name} needs its lane: the modem runs there")
        keys = getattr(station, "arbiter", None) is not None
        ports = ("port",)
        if name == "bateleur" and p.host == "s5066":
            from hfmodem.bateleur.s5066.server import NodeService
            from hfmodem.bateleur.config import DEFAULT_MODE
            server = NodeService(link.modem, p.addresses[0], p.mode or DEFAULT_MODE,
                                 port=p.port, log=lambda m: print(m, file=sys.stderr))
        elif name == "bateleur":
            from hfmodem.bateleur.host import BateleurHost
            server = BateleurHost(link.modem, cmd_port=p.cmd_port, data_port=p.data_port)
            ports = ("cmd_port", "data_port")
        elif name == "ale":
            from hfmodem.bateleur.ale.host import AleHost
            server = AleHost(link.modem, port=p.port)
        else:
            from hfmodem.bateleur.selcall.host import SelcallHost
            server = SelcallHost(link.modem, port=p.port, emergency=p.emergency)
        return Served(name, server, ports, keys=keys)
    if name == "pactor":
        if p.host not in ('ptc', 'ptc-managed'):
            raise ValueError('PACTOR station host must be ptc')
        from hfmodem.shrike.station_runtime import PtcServer
        link = station.lanes.get('pactor') if station is not None else None
        runtime = getattr(link, 'runtime', None)
        if runtime is None:
            raise ValueError('PTC requires the station native PACTOR owner')
        if p.host == 'ptc-managed':
            from hfmodem.core.managed_host import ManagedHost
            broker = ManagedHost(runtime.serve_stream, secret=_managed_secret(p), port=p.port,
                                 stream_roles=(3,),
                                 start_owner=lambda: runtime.start_host(managed=True),
                                 stop_owner=runtime.stop)
            return Served(name, broker, ('port',),
                          keys=station.rig is not None and station.arbiter is not None)
        return Served(name, PtcServer(runtime, endpoint=p.pty), (),
                      keys=station.rig is not None and station.arbiter is not None,
                      endpoint_attrs=('endpoint',))
    raise LookupError(f"no host dialect is implemented for {name}")


def _managed_secret(p):
    from pathlib import Path
    try:
        encoded = Path(p.auth_token_file).read_text(encoding='ascii').strip()
        secret = bytes.fromhex(encoded)
    except (OSError, ValueError, UnicodeError):
        raise ValueError('managed host token file must contain a 32-byte hex secret') from None
    if len(encoded) != 64 or len(secret) != 32:
        raise ValueError('managed host token file must contain a 32-byte hex secret')
    return secret


def stop_all(served: dict[str, Served], timeout=5.0) -> None:
    pending = []

    def stop(server, result):
        try:
            result.append(server.stop(timeout=timeout))
        except BaseException as exc:
            result.append(exc)

    for name, item in served.items():
        result = []
        thread = threading.Thread(target=stop, args=(item.server, result),
                                  name=f"stop-{name}", daemon=True)
        deadline = time.monotonic() + timeout + 0.25
        thread.start()
        pending.append((name, thread, deadline, result))
    failures = []
    for name, thread, deadline, result in pending:
        thread.join(max(0, deadline - time.monotonic()))
        if thread.is_alive() or not result or result[0] is not True:
            failures.append(name)
    if failures:
        raise LeaseTimeout(", ".join(failures))
