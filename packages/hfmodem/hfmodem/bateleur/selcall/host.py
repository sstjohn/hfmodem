# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The selcall host dialect: one command port, ASCII lines, as the ALE
dialect is (docs/protocols/bateleur/SELCALL.md).

Commands are answered ``OK ...`` or ``ERR <reason>``; calls heard are
written between as ``CALL`` lines."""
from __future__ import annotations

from ..ale.host import AleHost
from .modem import Event, SelcallModem

DEFAULT_PORT = 8340
VERSION = "hfmodem-selcall 1"


class SelcallHost(AleHost):
    def __init__(self, modem: SelcallModem, *, port: int = DEFAULT_PORT,
                 host: str = "127.0.0.1", emergency: bool = False) -> None:
        super().__init__(modem, port=port, host=host, channel_names=[])
        self.emergency = emergency

    def _send_call(self, kind: str, to: str, **fields) -> str:
        self._on_modem(lambda: self.modem.call(kind, to, **fields))
        return f"OK {kind.upper()} {to}"

    def command(self, line: str) -> str:
        word, _, rest = line.partition(" ")
        word = word.upper()
        a = rest.split()
        try:
            if word == "VERSION":
                return f"OK {VERSION}"
            if word == "STATUS":
                m = self.modem
                return (f"OK STATUS ADDRESS {','.join(m.addresses)} ACK {'ON' if m.acknowledge else 'OFF'} "
                        f"DOTS {m.dots_s:g} {'INVERTED' if m.invert else 'NORMAL'}")
            if word in ("CALL", "BEACON", "HANGUP", "GPSREQ", "STATUSREQ"):
                if len(a) != 1:
                    raise ValueError(f"{word} <address>")
                kind = {"CALL": "selcall", "BEACON": "beacon", "HANGUP": "hangup",
                        "GPSREQ": "gps_request", "STATUSREQ": "status_request"}[word]
                return self._send_call(kind, a[0])
            if word == "TELCALL":
                if len(a) != 2:
                    raise ValueError("TELCALL <address> <number>")
                return self._send_call("telcall", a[0], number=a[1])
            if word == "PAGE":
                if len(a) < 2:
                    raise ValueError("PAGE <address> <text>")
                return self._send_call("page", a[0], text=rest.partition(" ")[2].strip())
            if word == "GPS":
                if len(a) != 3:
                    raise ValueError("GPS <address> <lat> <lon>")
                return self._send_call("gps", a[0], position=(float(a[1]), float(a[2])))
            if word == "EMERGENCY":
                if not self.emergency:
                    raise ValueError("EMERGENCY is disabled in station config")
                to = a[0] if a and len(a) != 2 else "0" * len(self.modem.address)
                pos = a[-2:] if len(a) in (2, 3) else None
                return self._send_call("emergency", to,
                                       position=(float(pos[0]), float(pos[1])) if pos else None)
            raise ValueError(f"unknown command {word!r}")
        except Exception as exc:
            return f"ERR {exc}"

    def event(self, ev: Event) -> None:
        c = ev.call
        if ev.kind in ("HEARD", "CALLED"):
            what = {"telcall": f" NUMBER {c.number}", "page": f" TEXT {c.text}"}.get(c.kind, "")
            if c.position is not None:
                what = f" POSITION {c.position[0]:.5f} {c.position[1]:.5f}"
            flags = ("" if c.parity_ok else " PARITY-BAD") + (" INVERTED" if ev.inverted else "")
            line = (f"CALL {c.kind.upper()} FROM {c.frm} TO {c.to}"
                    f"{' MINE' if ev.kind == 'CALLED' else ''}{what}{flags}")
        elif ev.kind == "SENT":
            line = f"SENT {c.kind.upper()} TO {c.to}"
        else:
            line = f"{ev.kind.replace('_', ' ')} {c.kind.upper() if c else ''} {ev.text}".strip()
        self.log.append(line)
        with self._lock:
            conn = self._conn
        if conn is not None:
            conn.send_later(line.encode("ascii", "replace") + b"\r\n")
