# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""besra's ARDOP host-interface TCP server — the modem side of the dialect.

Speaks ARDOP's two-socket ASCII/binary host interface to a real client (Pat,
Winlink Express) and delegates all link behaviour to a `ModemCore` — the real
`BesraModem`, or the `LoopbackModem` for a radioless run of the dialect (a client
connects, runs an ARQ session, transfers data and disconnects with no radio).

Responsibility split (see `modem_core.py` for the other half):

  * This server owns TCP framing, command syntax, the ``<CMD> now <value>`` echo
    grammar, host-side configuration state, and the length-prefixed data socket.
  * The `ModemCore` owns the NEWSTATE machine, keying, buffering and the
    connect/disconnect edges, reported back through `ModemObserver`.

The command surface, reply grammar and the byte-exact quirks reproduced here
(the misspelled ``not recoginized`` fault, the ``NEWSTATE`` trailing space, the
``DISCONNECT NOW TRUE`` reply, ``CONSOLELOG`` echoing without ``now``, the
``FORCED`` suffix) are documented in `docs/protocols/ardop/10-HOST-API.md`. besra
reproduces the incumbent's bytes for compatibility; it does not adopt the quirks
into any interface of its own.
"""

from __future__ import annotations

import socket
import struct
import threading
from typing import Callable

from .. import __version__
from . import protocol as P
from .modem_core import LoopbackModem, ModemCore, ModemObserver

VERSION_STRING = f"besra-{__version__}"

# Value sets, verbatim from docs/protocols/ardop/10-HOST-API.md §3.3.
_BW_SET = (
    "200FORCED", "500FORCED", "1000FORCED", "2000FORCED",
    "200MAX", "500MAX", "1000MAX", "2000MAX",
)
ARQBW_VALUES = _BW_SET                       # ARQBW: the 8 tokens
CALLBW_VALUES = _BW_SET + ("UNDEFINED",)     # CALLBW additionally accepts UNDEFINED
FECMODE_VALUES = (
    "4FSK.200.50S", "4PSK.200.100S", "4PSK.200.100", "8PSK.200.100",
    "16QAM.200.100", "4FSK.500.100S", "4FSK.500.100", "4PSK.500.100",
    "8PSK.500.100", "16QAM.500.100", "4PSK.1000.100", "8PSK.1000.100",
    "16QAM.1000.100", "4PSK.2000.100", "8PSK.2000.100", "16QAM.2000.100",
    "4FSK.2000.600", "4FSK.2000.600S",
)

DATA_TAGS = ("ARQ", "FEC", "ERR", "IDF")


class Fault(Exception):
    """Raised by a command handler; the server renders it ``FAULT <text>``."""


def _valid_call(call: str) -> bool:
    """3–7 alphanumerics, an optional ``-SSID`` (0–15 or a single A–Z).
    docs/protocols/ardop/10-HOST-API.md §3.1 MYCALL. The ASCII guards matter: ``str.isdigit`` is
    true for characters ``int`` rejects (superscripts), so an unguarded parse of
    a byte like 0xB2 would raise rather than reject the callsign."""
    base, _, ssid = call.partition("-")
    if not (3 <= len(base) <= 7 and base.isascii() and base.isalnum()):
        return False
    if "-" in call:
        if ssid.isascii() and ssid.isdigit():
            return 0 <= int(ssid) <= 15
        return len(ssid) == 1 and ssid.isascii() and ssid.isalpha()
    return True


def _valid_grid(grid: str) -> bool:
    """Maidenhead 2/4/6/8 chars; loose field/square/subsquare check."""
    g = grid.upper()
    n = len(g)
    if n not in (2, 4, 6, 8):
        return False
    ok = g[0:2].isalpha()
    if n >= 4:
        ok = ok and g[2:4].isdigit()
    if n >= 6:
        ok = ok and g[4:6].isalpha()
    if n >= 8:
        ok = ok and g[6:8].isdigit()
    return ok


class HostServer(ModemObserver):
    """The two-port ARDOP host server over a pluggable `ModemCore`."""

    def __init__(self, modem: ModemCore | None = None, *,
                 host: str = "127.0.0.1", control_port: int = P.DEFAULT_CONTROL_PORT,
                 data_port: int | None = None, quiet: bool = False) -> None:
        self.modem = modem or LoopbackModem()
        self.host = host                       # loopback by default; ardopcf binds 0.0.0.0
        self.control_port = control_port
        # Port 0 means "any free port", so the +1 derivation would ask for
        # port 1; both sockets go ephemeral instead, and `start()` writes the
        # ports actually bound back into these attributes.
        if data_port is not None:
            self.data_port = data_port
        else:
            self.data_port = 0 if control_port == 0 else P.data_port_for(control_port)
        self.quiet = quiet

        # Host-side config state (the server owns config; the modem owns the link).
        self._cfg: dict[str, str] = {
            "ARQTIMEOUT": "90", "ARQBW": "500MAX", "CALLBW": "UNDEFINED",
            "PROTOCOLMODE": "ARQ", "LISTEN": "TRUE", "GRIDSQUARE": "",
            "LEADER": "240", "TRAILER": "20", "DRIVELEVEL": "100",
            "SQUELCH": "5", "BUSYDET": "5", "AUTOBREAK": "TRUE",
            "BUSYBLOCK": "FALSE", "ENABLEPINGACK": "TRUE", "CWID": "FALSE",
            "FECMODE": "4FSK.500.100", "FECREPEATS": "0", "FECID": "FALSE",
            "FSKONLY": "FALSE", "MONITOR": "TRUE", "USE600MODES": "FALSE",
            "CONSOLELOG": "6", "LOGLEVEL": "6", "EXTRADELAY": "0",
            "TUNINGRANGE": "100", "FASTSTART": "TRUE", "CMDTRACE": "FALSE",
            "DEBUGLOG": "FALSE", "INPUTNOISE": "0", "RXLEVEL": "100",
            "TXLEVEL": "100", "MYAUX": "",
        }
        self._mycall = ""
        self._initializing = False

        self._cmd_lock = threading.Lock()      # serialises writes to the command socket
        self._cmd_sock: socket.socket | None = None
        self._data_lock = threading.Lock()
        self._data_sock: socket.socket | None = None
        self._running = False
        self._listeners: list[socket.socket] = []
        self._accepting: list[threading.Thread] = []
        self._table = self._build_table()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> tuple[int, int]:
        """Bind both listeners and start accepting; returns the ports actually
        bound, which is the whole point of asking for port 0."""
        cmd_l = self._listen(self.control_port)
        try:
            data_l = self._listen(self.data_port)
        except OSError:
            cmd_l.close()
            raise
        self._running = True
        self.modem.start(self)
        self.control_port = cmd_l.getsockname()[1]
        self.data_port = data_l.getsockname()[1]
        self._log(f"besra host: control :{self.control_port}  data :{self.data_port}")
        self._listeners = [cmd_l, data_l]
        self._accepting = [
            threading.Thread(target=self._accept_loop, args=(cmd_l, self._serve_command),
                             name="accept-cmd", daemon=True),
            threading.Thread(target=self._accept_loop, args=(data_l, self._serve_data),
                             name="accept-data", daemon=True),
        ]
        for t in self._accepting:
            t.start()
        return self.control_port, self.data_port

    def stop(self) -> None:
        """Undoes `start`: the listeners close, the accept loops unwind on them,
        and any attached host loses both sockets."""
        self._running = False
        with self._cmd_lock:
            cmd, self._cmd_sock = self._cmd_sock, None
        with self._data_lock:
            data, self._data_sock = self._data_sock, None
        for sock in (*self._listeners, cmd, data):
            if sock is not None:
                sock.close()
        for t in self._accepting:
            t.join(timeout=1.0)
        self._listeners, self._accepting = [], []
        self.modem.stop()

    def serve_forever(self) -> None:
        self.start()
        try:
            while self._running:
                threading.Event().wait(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _listen(self, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, port))
        s.listen(1)
        return s

    def _accept_loop(self, listener: socket.socket, handler: Callable) -> None:
        while self._running:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            # One host per socket: the handler installs itself, displacing any
            # incumbent (spec §1 socket lifecycle).
            threading.Thread(target=handler, args=(conn,), daemon=True).start()

    # -- command socket ----------------------------------------------------

    def _serve_command(self, conn: socket.socket) -> None:
        with self._cmd_lock:
            displaced, self._cmd_sock = self._cmd_sock, conn
        if displaced is not None:
            displaced.close()          # its reader unwinds on the closed socket
        buf = bytearray()
        try:
            while self._running:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                # Reassemble CR-terminated lines across segments; any number per
                # segment (spec §7 "Server tolerance").
                while b"\r" in buf:
                    line, _, rest = buf.partition(b"\r")
                    del buf[:]
                    buf.extend(rest)
                    self._dispatch(line.decode("latin-1", "replace"))
        except OSError:
            pass
        finally:
            with self._cmd_lock:
                live = self._cmd_sock is conn
                if live:
                    self._cmd_sock = None
            conn.close()
            self._host_link_lost(live)

    def _dispatch(self, raw: str) -> None:
        line = raw.strip("\n").rstrip()
        if not line or line.upper() == "RDY":   # blank lines and a leading RDY ACK are swallowed
            return
        parts = line.split()
        cmd = parts[0].upper()                   # the modem upper-cases before dispatch
        args = parts[1:]
        handler = self._table.get(cmd)
        if handler is None:
            self._send_cmd(f"FAULT CMD {cmd} not recoginized")   # misspelled, verbatim
            return
        try:
            reply = handler(args, line)
        except Fault as f:
            self._send_cmd(f"FAULT {f}")
            return
        except Exception as e:                # a handler bug must not kill the reader
            self._send_cmd(f"FAULT {cmd} {e}")
            return
        if reply is not None:
            self._send_cmd(reply)

    def _send_cmd(self, text: str) -> None:
        with self._cmd_lock:
            if self._cmd_sock is not None:
                try:
                    self._cmd_sock.sendall(text.encode("latin-1") + P.CR)
                except OSError:
                    self._cmd_sock = None

    # -- data socket -------------------------------------------------------

    def _host_link_lost(self, live: bool) -> None:
        """Either socket dropping mid-session ends the session and reverts to
        receive (spec §1). Only the *current* host counts: a connection that has
        already been displaced must not tear down its successor's link."""
        if live and self.modem.connected:
            self.modem.disconnect()

    def _serve_data(self, conn: socket.socket) -> None:
        with self._data_lock:
            displaced, self._data_sock = self._data_sock, conn
        if displaced is not None:
            displaced.close()
        try:
            while self._running:
                hdr = self._recv_exactly(conn, 2)
                if hdr is None:
                    break
                (length,) = struct.unpack(">H", hdr)
                payload = self._recv_exactly(conn, length)
                if payload is None:
                    break
                self.modem.transmit(bytes(payload))
        except OSError:
            pass
        finally:
            with self._data_lock:
                live = self._data_sock is conn
                if live:
                    self._data_sock = None
            conn.close()
            self._host_link_lost(live)

    @staticmethod
    def _recv_exactly(conn: socket.socket, n: int) -> bytearray | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return buf

    def _send_data(self, tag: str, blob: bytes) -> None:
        # modem→host block: <2-byte BE length><3-char tag><payload>; length
        # counts the tag + payload (spec §5 "Data path", §6).
        body = tag.encode("ascii") + blob
        with self._data_lock:
            if self._data_sock is not None and not self._initializing:
                try:
                    self._data_sock.sendall(struct.pack(">H", len(body)) + body)
                except OSError:
                    self._data_sock = None

    # -- ModemObserver: modem→host notifications on the command socket ------

    def modem_newstate(self, state: str) -> None:
        self._send_cmd(f"NEWSTATE {state} ")     # trailing space reproduces ARQ.c:338

    def modem_connected(self, remote: str, bw: int) -> None:
        self._send_cmd(f"CONNECTED {remote} {bw}")

    def modem_disconnected(self) -> None:
        self._send_cmd("DISCONNECTED")

    def modem_ptt(self, on: bool) -> None:
        self._send_cmd(f"PTT {'TRUE' if on else 'FALSE'}")

    def modem_buffer(self, nbytes: int) -> None:
        self._send_cmd(f"BUFFER {nbytes}")

    def modem_data_received(self, kind: str, blob: bytes) -> None:
        self._send_data(kind, blob)

    def modem_busy(self, on: bool) -> None:
        self._send_cmd(f"BUSY {'TRUE' if on else 'FALSE'}")

    def modem_pending(self, cancel: bool = False) -> None:
        self._send_cmd("CANCELPENDING" if cancel else "PENDING")

    def modem_target(self, call: str) -> None:
        self._send_cmd(f"TARGET {call}")

    def modem_status(self, text: str) -> None:
        self._send_cmd(f"STATUS {text}")

    def modem_fault(self, text: str) -> None:
        self._send_cmd(f"FAULT {text}")

    # -- command table -----------------------------------------------------

    def _build_table(self) -> dict[str, Callable[[list[str], str], str | None]]:
        t: dict[str, Callable[[list[str], str], str | None]] = {}

        def bool_knob(name: str, *, extra: tuple[str, ...] = ()):
            def h(args, _line):
                if not args:
                    return f"{name} {self._cfg[name]}"
                v = args[0].upper()
                if v not in ("TRUE", "FALSE") + extra:
                    raise Fault(f"Syntax Err: {name} {args[0]}")
                self._cfg[name] = v
                self._on_set(name, v)
                return f"{name} now {v}"
            return h

        def int_knob(name: str, lo: int, hi: int, round_to: int = 1):
            def h(args, _line):
                if not args:
                    return f"{name} {self._cfg[name]}"
                try:
                    n = int(args[0])
                except ValueError:
                    raise Fault(f"Syntax Err: {name} {args[0]}")
                if not lo <= n <= hi:
                    raise Fault(f"Syntax Err: {name} {args[0]}")
                if round_to > 1:
                    n = ((n + round_to - 1) // round_to) * round_to   # round up, as ardopcf does
                self._cfg[name] = str(n)
                return f"{name} now {n}"
            return h

        def enum_knob(name: str, values: tuple[str, ...], *, in_session_fault=False):
            def h(args, _line):
                if not args:
                    return f"{name} {self._cfg[name]}"
                v = args[0].upper()
                if v not in values:
                    raise Fault(f"Syntax Err: {name} {args[0]}")
                if in_session_fault and self.modem.connected:
                    raise Fault(f"Not from state {self.modem.state}")
                self._cfg[name] = v
                self._on_set(name, v)
                return f"{name} now {v}"
            return h

        # Boolean knobs.
        for name in ("AUTOBREAK", "BUSYBLOCK", "ENABLEPINGACK", "FECID",
                     "FSKONLY", "MONITOR", "USE600MODES", "LISTEN",
                     "FASTSTART", "CMDTRACE", "DEBUGLOG"):
            t[name] = bool_knob(name)
        t["CWID"] = bool_knob("CWID", extra=("ONOFF",))

        # Integer knobs (range, optional rounding) from the §3.1 table.
        t["ARQTIMEOUT"] = int_knob("ARQTIMEOUT", 30, 240)
        t["BUSYDET"] = int_knob("BUSYDET", 0, 10)
        t["DRIVELEVEL"] = int_knob("DRIVELEVEL", 0, 100)   # ardopcf accepts 0
        t["EXTRADELAY"] = int_knob("EXTRADELAY", 0, 100000)
        t["FECREPEATS"] = int_knob("FECREPEATS", 0, 5)
        t["LEADER"] = int_knob("LEADER", 120, 2500, round_to=10)
        t["LOGLEVEL"] = int_knob("LOGLEVEL", 1, 6)
        t["SQUELCH"] = int_knob("SQUELCH", 1, 10)
        t["TRAILER"] = int_knob("TRAILER", 0, 200, round_to=10)
        t["TUNINGRANGE"] = int_knob("TUNINGRANGE", 0, 200)
        t["INPUTNOISE"] = int_knob("INPUTNOISE", 0, 100000)
        t["RXLEVEL"] = int_knob("RXLEVEL", 0, 100)
        t["TXLEVEL"] = int_knob("TXLEVEL", 0, 100)

        # CONSOLELOG echoes WITHOUT `now` (spec §7 quirk).
        def consolelog(args, _line):
            if not args:
                return f"CONSOLELOG {self._cfg['CONSOLELOG']}"
            try:
                n = int(args[0])
                assert 1 <= n <= 6
            except (ValueError, AssertionError):
                raise Fault(f"Syntax Err: CONSOLELOG {args[0]}")
            self._cfg["CONSOLELOG"] = str(n)
            return f"CONSOLELOG {n}"
        t["CONSOLELOG"] = consolelog

        # Enums.
        t["ARQBW"] = enum_knob("ARQBW", ARQBW_VALUES, in_session_fault=True)
        t["CALLBW"] = enum_knob("CALLBW", CALLBW_VALUES)
        t["FECMODE"] = enum_knob("FECMODE", FECMODE_VALUES)

        # PROTOCOLMODE accepts anything (dead-code validation), forces DISC,
        # echoes the raw parameter (spec §7 quirk).
        def protocolmode(args, _line):
            if not args:
                return f"PROTOCOLMODE {self._cfg['PROTOCOLMODE']}"
            raw = args[0]
            mode = raw.upper() if raw.upper() in ("RXO", "FEC") else "ARQ"
            self._cfg["PROTOCOLMODE"] = mode
            self.modem.set_protocolmode(mode)
            if self.modem.connected:
                self.modem.abort()
            return f"PROTOCOLMODE now {raw}"
        t["PROTOCOLMODE"] = protocolmode

        # Identity.
        def mycall(args, _line):
            if not args:
                return f"MYCALL {self._mycall}" if self._mycall else "MYCALL"
            call = args[0].upper()
            if not _valid_call(call):
                raise Fault(f"Syntax Err: MYCALL {args[0]}")
            self._mycall = call
            self.modem.set_mycall(call)
            return f"MYCALL now {call}"
        t["MYCALL"] = mycall

        def myaux(args, _line):
            if not args:
                return f"MYAUX {self._cfg['MYAUX']}"
            calls = " ".join(args).replace(",", " ").split()
            if len(calls) > 10 or any(not _valid_call(c.upper()) for c in calls):
                self._cfg["MYAUX"] = ""          # an illegal call clears the whole list
                raise Fault(f"Syntax Err: MYAUX {' '.join(args)}")
            val = ",".join(c.upper() for c in calls)
            self._cfg["MYAUX"] = val
            return f"MYAUX now {val}"
        t["MYAUX"] = myaux

        def gridsquare(args, _line):
            if not args:
                return f"GRIDSQUARE {self._cfg['GRIDSQUARE']}"
            if not _valid_grid(args[0]):
                raise Fault(f"Syntax Err: GRIDSQUARE {args[0]}")
            g = args[0][:2].upper() + args[0][2:]
            self._cfg["GRIDSQUARE"] = g
            self.modem.set_gridsquare(g)
            return f"GRIDSQUARE now {g}"
        t["GRIDSQUARE"] = gridsquare

        # String device knobs (stored + echoed; audio not actually rerouted).
        for name in ("CAPTURE", "PLAYBACK"):
            def strk(args, _line, _n=name):
                if not args:
                    return f"{_n} {self._cfg.get(_n, '')}"
                self._cfg[_n] = args[0]
                return f"{_n} now {args[0]}"
            t[name] = strk
        t["CAPTUREDEVICES"] = lambda a, l: f"CAPTUREDEVICES {self._cfg.get('CAPTURE', '')}"
        t["PLAYBACKDEVICES"] = lambda a, l: f"PLAYBACKDEVICES {self._cfg.get('PLAYBACK', '')}"

        # Queries.
        t["VERSION"] = lambda a, l: f"VERSION {VERSION_STRING}"
        t["STATE"] = lambda a, l: f"STATE {self.modem.state}"
        t["BUFFER"] = lambda a, l: f"BUFFER {self.modem.queued}"

        # Actions / session verbs.
        t["INITIALIZE"] = self._cmd_initialize
        t["ARQCALL"] = self._cmd_arqcall
        t["PING"] = self._cmd_ping
        t["DISCONNECT"] = self._cmd_disconnect
        t["ABORT"] = self._cmd_abort
        t["DD"] = self._cmd_abort                        # alias
        t["SENDID"] = self._cmd_sendid
        t["FECSEND"] = self._cmd_fecsend
        t["PURGEBUFFER"] = self._cmd_purgebuffer
        t["CL"] = self._cmd_cl                            # PTC-emulator PURGEBUFFER alias
        t["DATATOSEND"] = self._cmd_datatosend
        t["TWOTONETEST"] = self._cmd_twotonetest
        t["CLOSE"] = self._cmd_close
        t["BREAK"] = lambda a, l: None                   # manual turnover, no reply

        # Radio-control stubs (no CAT layer on a desktop besra).
        t["RADIOFREQ"] = lambda a, l: (None if a else self._raise("RADIOFREQ command string missing"))
        t["RADIOHEX"] = lambda a, l: None                # silently ignored, no CAT
        t["RADIOPTTON"] = lambda a, l: self._raise("RADIOPTTON CAT Port not defined")
        t["RADIOPTTOFF"] = lambda a, l: self._raise("RADIOPTTOFF CAT Port not defined")
        return t

    # -- helpers used by handlers -----------------------------------------

    @staticmethod
    def _raise(text: str):
        raise Fault(text)

    def _on_set(self, name: str, value: str) -> None:
        """Propagate the config knobs the modem cares about."""
        if name == "LISTEN":
            self.modem.set_listen(value == "TRUE")
        elif name == "ARQBW":
            forced = value.endswith("FORCED")
            self.modem.set_bandwidth(int(value[:-6] if forced else value[:-3]), forced)

    # -- session-verb handlers --------------------------------------------

    def _cmd_initialize(self, args, line):
        self._initializing = True
        if self.modem.connected:
            self.modem.abort()
        self._initializing = False
        return "INITIALIZE"

    def _cmd_arqcall(self, args, line):
        if len(args) != 2:
            raise Fault(f"Syntax Err: {line}")
        target, rep = args[0], args[1]
        if not self._mycall:
            raise Fault("MYCALL not set")
        if self._cfg["PROTOCOLMODE"] != "ARQ":
            raise Fault("Not from mode " + self._cfg["PROTOCOLMODE"])
        try:
            repeats = int(rep)
            assert repeats >= 1
        except (ValueError, AssertionError):
            raise Fault(f"Syntax Err: {line}")
        self.modem.connect(target.upper(), repeats)
        return line                                       # echo original, case-preserved

    def _cmd_ping(self, args, line):
        if len(args) != 2 or not self._mycall:
            raise Fault("MYCALL not set" if not self._mycall else f"Syntax Err: {line}")
        if self.modem.state != P.ArdopState.DISC:
            raise Fault(f"No PING from state {self.modem.state}")
        return line

    def _cmd_disconnect(self, args, line):
        if self.modem.connected:
            self.modem.disconnect()
            return "DISCONNECT NOW TRUE"
        return "DISCONNECT IGNORED"

    def _cmd_abort(self, args, line):
        self.modem.abort()
        return "ABORT"

    def _cmd_sendid(self, args, line):
        if not self._mycall:
            raise Fault("MYCALL not set")
        if self.modem.state != P.ArdopState.DISC:
            raise Fault(f"Not from State {self.modem.state}")
        self.modem.send_id()
        return "SENDID"

    def _cmd_fecsend(self, args, line):
        if not args or args[0].upper() not in ("TRUE", "FALSE"):
            raise Fault(f"Syntax Err: {line}")
        if not self._mycall:
            raise Fault("MYCALL not set")
        return f"FECSEND now {args[0].upper()}"

    def _cmd_purgebuffer(self, args, line):
        # Empties the outbound buffer and nothing else: a host clearing a stalled
        # queue keeps its ARQ session (spec §3.1). The modem reports the new depth
        # as the async BUFFER.
        self.modem.purge_buffer()
        return "PURGEBUFFER"

    def _cmd_cl(self, args, line):
        self.modem.purge_buffer()                         # async BUFFER only, no echo
        return None

    def _cmd_datatosend(self, args, line):
        if not args:
            return f"DATATOSEND {self.modem.queued}"
        if args[0] == "0":
            self.modem.purge_buffer()
            return "DATATOSEND now 0"
        raise Fault(f"Syntax Err: {line}")

    def _cmd_twotonetest(self, args, line):
        if self.modem.state != P.ArdopState.DISC:
            raise Fault(f"Not from state {self.modem.state}")
        return "TWOTONETEST"

    def _cmd_close(self, args, line):
        self._running = False
        return None

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, flush=True)
