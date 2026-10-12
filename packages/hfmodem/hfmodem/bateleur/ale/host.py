# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ALE host dialect: one command port, ASCII lines.

MIL-STD-188-141B specifies the ALE controller, not how an application drives
one, and the radios that implement it each have their own remote control. This
dialect is small and written down in ``docs/protocols/bateleur/ALE-HOST.md``;
it is not a copy of any radio's.

Commands are answered with one line, ``OK ...`` or ``ERR <reason>``. Between
answers the controller writes what happens as unsolicited lines — ``LINKED``,
``CALLED``, ``HEARD``, ``AMD`` and the rest — so a client reads the port as a
stream of lines and matches answers to its commands in order.

One application at a time: a new connection replaces the old one.
"""
from __future__ import annotations

import socket
import threading
from collections import deque

from hfmodem.core import hostio
from datetime import datetime, timezone

from . import gpr
from .controller import Event
from .modem import AleModem, Observer

DEFAULT_PORT = 8330
VERSION = "bateleur-ale 1"


class AleHost(Observer):
    def __init__(self, modem: AleModem, *, port: int = DEFAULT_PORT,
                 host: str = "127.0.0.1", channel_names: list[str] | None = None) -> None:
        self.modem = modem
        modem.observer = self
        self._bind = (host, port)
        self.port = 0
        self.names = channel_names if channel_names is not None else modem.ctl.cfg.channels
        self._srv: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._lock = threading.Lock()
        self._running = False
        self._threads: list[threading.Thread] = []
        self.log = deque(maxlen=1024)                     # every event line, for tests
        self._told: set[str] = set()                 # stations already named

    # -- lifecycle -------------------------------------------------------------

    def start_background(self) -> None:
        s = hostio.listen(*self._bind)
        self._srv = s
        self.port = s.getsockname()[1]
        self._running = True
        t = threading.Thread(target=self._accept, daemon=True, name="ale-host")
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._running = False
        for s in (self._srv, self._conn):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        for t in self._threads:
            t.join(timeout=2.0)

    def _accept(self) -> None:
        while self._running:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if not self._running:
                conn.close()
                continue
            conn = hostio.Connection(conn, line=True)
            with self._lock:
                old, self._conn = self._conn, conn
            if old is not None:
                old.close()
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        hostio.serve_lines(conn, lambda: self._running, self.command, self._send_line)

    def _send_line(self, conn, text: str) -> None:
        try:
            conn.sendall(text.encode("ascii", "replace") + b"\r\n")
        except OSError:
            pass

    # -- commands ----------------------------------------------------------------

    def command(self, line: str) -> str:
        """One command, answered. Public so the dialect is testable without a
        socket."""
        word, _, rest = line.partition(" ")
        word = word.upper()
        args = rest.split()
        m = self.modem
        try:
            if word == "VERSION":
                return f"OK {VERSION}"
            if word == "WHO":
                # WHO <address>: what the directory says of a station.
                d = getattr(m, "directory", None)
                if d is None:
                    raise ValueError("no directory loaded ([protocols.ale] directory)")
                if len(args) != 1:
                    raise ValueError("WHO <address>")
                hits = d.who(args[0])
                return f"OK WHO {args[0].upper()} " + (
                    " | ".join(_who(e) for e in hits) if hits else "not listed")
            if word == "CALL":
                # One address, a wildcard ("?"), or a group: A,B,C.
                if not args:
                    raise ValueError("CALL <address>[,<address>...] [message]")
                peer, text = args[0], rest.partition(" ")[2].strip()
                called = peer.split(",") if "," in peer else peer
                self._on_modem(lambda: m.ctl.call(called, m.now, amd=text))
                return f"OK CALL {peer.upper()}"
            if word == "NET":
                if not args:
                    raise ValueError("NET <net address> <member>,<member>... [message]")
                net = args[0]
                if len(args) == 1:
                    from .hflink import NET_SLOTS, net_members
                    if net.upper() not in NET_SLOTS:
                        raise ValueError("NET needs members in response-slot order")
                    members = net_members(net, m.ctl.me[0])
                else:
                    members = args[1].split(",")
                text = rest.split(None, 2)[2].strip() if len(args) > 2 else ""
                self._on_modem(lambda: m.ctl.call(net, m.now, amd=text, members=members))
                return f"OK NET {net.upper()}"
            if word == "ANYCALL":
                sel = args[0] if args else ""
                self._on_modem(lambda: m.ctl.anycall(m.now, selective=sel))
                return f"OK ANYCALL {sel.upper()}".rstrip()
            if word == "ALLCALL":
                # ALLCALL [/X] [message]: /X for the stations ending in X.
                sel = args[0][1:] if args and args[0].startswith("/") else ""
                text = rest.split(None, 1)[1] if sel and len(args) > 1 else \
                    ("" if sel else rest.strip())
                if not self._on_modem(lambda: m.ctl.allcall(m.now, selective=sel, amd=text)):
                    raise ValueError("not sent: the controller is not available, "
                                     "or the channel is occupied")
                return f"OK ALLCALL {sel.upper()}".rstrip()
            if word == "BROADCAST":
                # BROADCAST <address> <message>: an AMD ended TWAS, on this
                # channel; no one answers (the GPR and RPT nets' way).
                if len(args) < 2:
                    raise ValueError("BROADCAST <address> <message>")
                to, text = args[0], rest.partition(" ")[2].strip()
                self._on_modem(lambda: m.ctl.broadcast(to, text, m.now))
                return f"OK BROADCAST {to.upper()}"
            if word == "GPR":
                # GPR <lat> <lon> [<altitude m> [comment]]: this station's
                # position, now, to the GPR net (ale.gpr).
                if len(args) < 2:
                    raise ValueError("GPR <lat> <lon> [<altitude m> [comment]]")
                alt = float(args[2]) if len(args) > 2 else None
                text = gpr.format(gpr.Report(
                    m.ctl.me[0], float(args[0]), float(args[1]), alt,
                    datetime.now(timezone.utc), " ".join(args[3:])))
                self._on_modem(lambda: m.ctl.broadcast("GPR", text, m.now))
                return f"OK GPR {text}"
            if word == "AMD":
                if len(args) < 2:
                    raise ValueError("AMD <address> <message>")
                peer, text = args[0], rest.partition(" ")[2].strip()
                self._on_modem(lambda: m.ctl.call(peer, m.now, amd=text))
                return f"OK AMD {peer.upper()}"
            if word in ("DTM", "DTMBIN", "DBM", "DBMBIN"):
                # DTM <address> [NOARQ] <text>; DTMBIN <address> [NOARQ] <hex>;
                # DBM and DBMBIN the same.
                text_mode = not word.endswith("BIN")
                if len(args) < 2:
                    raise ValueError(f"{word} <address> [NOARQ] <{'text' if text_mode else 'hex'}>")
                peer, body = args[0], rest.partition(" ")[2].strip()
                arq = True
                if body.split(None, 1)[0].upper() == "NOARQ":
                    arq, body = False, (body.split(None, 1) + [""])[1]
                data = body if text_mode else bytes.fromhex(body)
                send = m.ctl.send_dtm if word.startswith("DTM") else m.ctl.send_dbm
                self._on_modem(lambda: send(peer, data, m.now, arq=arq))
                return f"OK {word} {peer.upper()} {len(data)}"
            if word == "TIME":
                # TIME <address>: ask it the time (linking protection's
                # time exchange, B.5.5).
                if len(args) != 1:
                    raise ValueError("TIME <address>")
                self._on_modem(lambda: m.ctl.time_request(args[0], m.now))
                return f"OK TIME {args[0].upper()}"
            if word == "TIMECAST":
                to = args[0] if args else "@?@"
                if not self._on_modem(lambda: m.ctl.time_broadcast(m.now, to.upper())):
                    raise ValueError("not sent: no linking protection, or not available")
                return f"OK TIMECAST {to.upper()}"
            if word == "QUERY":
                what = {a.upper() for a in args[1:]}
                if not args or not what or what - {"LQA", "VERSION", "CAPABILITIES"}:
                    raise ValueError("QUERY <address> LQA|VERSION|CAPABILITIES ...")
                peer = args[0]
                self._on_modem(lambda: m.ctl.call(peer, m.now, lqa="LQA" in what,
                                                  version="VERSION" in what,
                                                  capabilities="CAPABILITIES" in what))
                return f"OK QUERY {peer.upper()} {' '.join(sorted(what))}"
            if word == "SOUND":
                ok = self._on_modem(lambda: m.ctl.sound(m.now))
                if not ok:
                    raise ValueError("not sounding: the controller is not available, "
                                     "or the channel is occupied")
                return "OK SOUND"
            if word == "HANGUP":
                self._on_modem(lambda: m.ctl.hang_up(m.now))
                return "OK HANGUP"
            if word == "SCAN":
                if len(args) != 1 or args[0].upper() not in ("ON", "OFF"):
                    raise ValueError("SCAN <ON|OFF>")
                on = args[0].upper() == "ON"
                self._on_modem(lambda: setattr(m.ctl.cfg, "scanning", on))
                return f"OK SCAN {args[0].upper()}"
            if word == "CHANNEL":
                if len(args) != 1:
                    raise ValueError("CHANNEL <index or name>")
                ch = self._channel(args[0])
                self._on_modem(lambda: m.ctl.stay(ch, m.now))
                return f"OK CHANNEL {ch} {self.names[ch]}"
            if word == "CHANNELS":
                return "OK CHANNELS " + " ".join(f"{i}={n}" for i, n in enumerate(self.names))
            if word == "STATUS":
                c = m.ctl
                linked = ",".join(sorted(c.linked)) or "-"
                lp = f" LP AL-{c.cfg.lp_level} TIME Q{c.time_quality(m.now)}" if c.lp else ""
                return (f"OK STATUS {c.state.upper()} CHANNEL {c.channel} {self.names[c.channel]} "
                        f"LINKED {linked} SCAN {'ON' if c.cfg.scanning else 'OFF'}{lp}")
            if word == "HEARD":
                rows = sorted(m.ctl.lqa.items(), key=lambda kv: -kv[1].when)[:20]
                return "OK HEARD " + " ".join(
                    f"{st}@{self.names[ch]}:ber={q.ber:.0f},sinad={q.sinad:.0f}"
                    for (st, ch), q in rows)
            raise ValueError(f"unknown command {word!r}")
        except Exception as exc:
            return f"ERR {exc}"

    def _channel(self, arg: str) -> int:
        if arg.isdigit() and int(arg) < len(self.names):
            return int(arg)
        if arg in self.names:
            return self.names.index(arg)
        raise ValueError(f"no channel {arg!r}; CHANNELS lists them")

    def _on_modem(self, fn, timeout: float | None = None):
        """Run ``fn`` on the modem's thread and wait, so the answer reports
        what the controller did."""
        return hostio.posted(self.modem.post, fn, timeout)

    def event(self, ev: Event) -> None:
        name = self.names[ev.channel] if 0 <= ev.channel < len(self.names) else "-"
        lqa = "".join(f" {k.upper()} {v}" for k, v in ev.lqa.items())
        if ev.kind == "STATE":
            line = f"STATE {ev.text.upper()}"
        elif ev.kind == "AMD":
            line = f"AMD {ev.peer} {ev.text}"
        elif ev.kind == "CAPABILITIES":
            line = f"CAPABILITIES {ev.peer} {ev.text}"
        elif ev.kind in ("DTM", "DBM"):
            line = (f"{ev.kind} {ev.peer} {ev.data}" if isinstance(ev.data, str)
                    else f"{ev.kind}BIN {ev.peer} {ev.data.hex()}")
        elif ev.kind in ("DTM_SENT", "DBM_SENT"):
            line = f"{ev.kind.replace('_', ' ')} {ev.peer} {len(ev.data)}"
        elif ev.kind in ("TIME", "TIME_FAILED", "TIME_SERVED", "TIME_REQUESTED"):
            line = f"{ev.kind.replace('_', ' ')} {ev.peer} {ev.text}"
        elif ev.kind in ("DTM_FAILED", "DTM_DAMAGED", "DBM_FAILED", "DBM_DAMAGED"):
            line = f"{ev.kind.replace('_', ' ')} {ev.peer} {ev.text}"
        elif ev.kind in ("TUNE_FAILED", "TX_REFUSED"):
            line = f"{ev.kind} CHANNEL {name} {ev.text}"
        elif ev.kind == "SOUNDED":
            line = f"SOUNDED CHANNEL {name}"
        elif ev.kind == "SENT":
            line = f"SENT {ev.peer} CHANNEL {name} {ev.text}".rstrip()
        elif ev.kind in ("CALL_FAILED", "LINK_ENDED"):
            line = f"{ev.kind.replace('_', ' ')} {ev.peer} {ev.text}".rstrip()
        else:
            line = f"{ev.kind} {ev.peer} CHANNEL {name}{lqa}"
            if ev.kind in ("HEARD", "CALLED", "LINKED") and ev.text:
                line += f" ({ev.text})"
        lines = [line]
        # The first time a station is heard, what the directory says of it.
        d = getattr(self.modem, "directory", None)
        if d is not None and ev.peer and ev.kind in ("HEARD", "CALLED", "LINKED") \
                and ev.peer not in self._told:
            self._told.add(ev.peer)
            hits = d.who(ev.peer)
            if hits:
                lines.append(f"WHO {ev.peer} " + " | ".join(_who(e) for e in hits))
        self.log.extend(lines)
        with self._lock:
            conn = self._conn
        if conn is not None:
            for ln in lines:
                conn.send_later(ln.encode("ascii", "replace") + b"\r\n")


def _who(e) -> str:
    return f"{e.pattern}={e.agency}" + (f": {e.description}" if e.description else "")
