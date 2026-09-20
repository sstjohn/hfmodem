# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A rigctld that behaves like the real one, including badly.

A double is evidence about the code only if it is at least as strict as the thing
it stands for. This one used to be more generous in exactly one way — it appended
an `RPRT 0` to every reply, including the plain ones a real daemon terminates with
nothing at all — and that generosity concealed a transport which could not read a
single reply out of Hamlib 5.

The wire format below is the measured one, from a 5.0.0 daemon driving an FT-891
(model 1036):

    +f                    get_freq:\\nFrequency: 7097500\\nRPRT 0
    +m                    get_mode:\\nMode: PKTUSB\\nPassband: 1700\\nRPRT 0
    +t                    get_ptt:\\nRPRT -11
    +F 7097500            set_freq: 7097500\\nRPRT 0
    +M PKTUSB 0           set_mode: PKTUSB 0\\nRPRT 0
    +\\get_conf ptt_type   get_conf: ptt_type\\nptt_type=None\\nRPRT 0
    +\\set_cache 0         set_cache: 0\\nRPRT 0
    +\\dump_caps           dump_caps:\\nCaps dump for model: 1036\\n...

`+` is what asks for that framing. Without it a *get* answers with the bare value
and no terminator at all, which is the plain form this fake also speaks — and the
reason a reader waiting for `RPRT ` hangs against a daemon that ignores the `+`.

Three further properties are deliberate:

  * `\\dump_caps` is large and goes out in **small chunks**, so a reader that
    assumes one segment per reply fails here rather than on the air;
  * `t` reports a *transmitter* state the keying line can drive, not rigctld's
    memory of its own last `T`. A fake whose `t` answers from a variable only `T`
    moves cannot express the one state a panic confirmation exists to catch —
    line stuck up, rig keyed;
  * `\\set_conf extended_resp 1` is answered `RPRT -1` and the connection is then
    closed, which is what Hamlib 5 does with it. Nothing sends that command now;
    the fake keeps answering it the way the daemon does so that sending it again
    costs the connection here rather than at the radio.

`ptt_type` is the knob with teeth. Hamlib answers `t` and `T` with `RPRT -11` —
ENAVAIL — whenever the rig it drives is configured with no PTT of its own, and
that is the configuration `Rig._check_ptt_owner` demands, because this station
keys through its own line. One daemon cannot both leave the keying line to us and
report PTT back, so a test wanting CAT readback has to ask for a daemon the arm
gate would refuse.
"""
from __future__ import annotations

import socket
import threading

#: Long enough that one recv cannot hold it, as on a real rig.
_DUMP_CAPS_PADDING = "\n".join(
    f"Level {i}: RFPOWER(0.000000..1.000000/0.0) AF(0.000000..1.000000/0.0)"
    for i in range(220))

#: Hamlib's ENAVAIL. What a rig with no PTT of its own says to `t` and `T`.
_ENAVAIL = -11


class FakeRigctld:
    """Each option is named for the failure it reproduces."""

    def __init__(self, *, model="Yaesu FT-891", freq=7_100_000, wedge=False,
                 refuse=False, delay=0.0, lie_ptt=False, chunk=64, ptt_line=None,
                 ptt_type="RIG", no_extended=False, no_cache=False,
                 no_terminator=False):
        self.model, self.freq = model, freq
        self.mode, self.passband = "PKTUSB", 1700
        self.wedge, self.refuse, self.delay, self.lie_ptt = wedge, refuse, delay, lie_ptt
        #: Bytes per write. Small on purpose.
        self.chunk = chunk
        #: A `FakePtt`, when `t` should answer from the keying line rather than
        #: from rigctld's own bookkeeping.
        self.ptt_line = ptt_line
        #: How the daemon keys, and therefore whether it can answer `t` at all.
        #: "None" is what this station requires and what the bench daemon reports.
        self.ptt_type = ptt_type
        #: A daemon that refuses `\\set_cache` and hangs up on it, as Hamlib 5 does
        #: with the setup command this station used to send.
        self.no_extended = no_extended
        #: A daemon that refuses `\\set_cache` and stays up: an older rigctld that
        #: does not know the command.
        self.no_cache = no_cache
        #: A daemon that ignores the `+`, so its gets carry no terminator.
        self.no_terminator = no_terminator
        self.ptt = False
        self.cache = True
        #: Every line as it arrived, `+` and all.
        self.log: list[str] = []
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self.port = self._srv.getsockname()[1]
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def transmitting(self) -> bool:
        """What a radio would actually be doing.

        The keying line wins when there is one: a rig whose PTT input is held is
        transmitting whatever CAT was last told.
        """
        if self.lie_ptt:
            return True
        if self.ptt_line is not None:
            return bool(getattr(self.ptt_line, "line", False)) or self.ptt
        return self.ptt

    @property
    def ptt_readable(self) -> bool:
        """Whether this daemon can answer `t` and `T` at all."""
        return self.ptt_type.strip().lower() not in ("", "none")

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._session, args=(conn,), daemon=True).start()

    def _session(self, conn):
        with conn:
            conn.settimeout(5.0)
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except OSError:
                    return
                if not data:
                    return
                for raw in data.decode(errors="replace").splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    self.log.append(line)
                    if self.wedge:
                        continue            # accepted, and never answered
                    if self.delay:
                        self._stop.wait(self.delay)
                    if not self._send(conn, self._reply(line)):
                        return
                    if self._hangs_up(line):
                        return

    def _hangs_up(self, line: str) -> bool:
        """Refusing a setup command and dropping the connection are one event."""
        cmd = line.lstrip("+")
        return (cmd.startswith("\\set_conf extended_resp")
                or (self.no_extended and cmd.startswith("\\set_cache")))

    def _send(self, conn, payload: str) -> bool:
        raw = payload.encode()
        for i in range(0, len(raw), self.chunk):
            try:
                conn.sendall(raw[i:i + self.chunk])
            except OSError:
                return False
        return True

    def _reply(self, line: str) -> str:
        # A daemon that does not know the `+` answers plain whatever it is asked.
        framed = line.startswith("+") and not self.no_terminator
        name, args, fields, plain, code = self._answer(line.lstrip("+"))
        if self.refuse:
            code, fields, plain = -1, [], []
        if framed:
            head = f"{name}:{' ' + args if args else ''}\n"
            body = "".join(f"{f}\n" for f in fields) if code == 0 else ""
            return f"{head}{body}RPRT {code}\n"
        # Plain: a get is its values and nothing else; a set and every failure are
        # the terminator and nothing else. This is the shape that makes a reader
        # waiting for `RPRT ` wait forever.
        if code == 0 and plain:
            return "".join(f"{p}\n" for p in plain)
        return f"RPRT {code}\n"

    def _answer(self, cmd: str) -> tuple[str, str, list[str], list[str], int]:
        """One command, as (header name, echoed args, framed fields, plain values,
        return code)."""
        head, _, rest = cmd.partition(" ")
        arg = rest.strip()
        if head == "\\set_conf":
            return "set_conf", arg, [], [], -1
        if head == "\\set_cache":
            if self.no_extended or self.no_cache:
                return "set_cache", arg, [], [], -1
            self.cache = False
            return "set_cache", arg, [], [], 0
        if head == "\\get_conf":
            value = self.ptt_type if arg == "ptt_type" else ""
            return "get_conf", arg, [f"{arg}={value}"], [value], 0
        if head == "\\dump_caps":
            caps = ["Caps dump for model: 1036", f"Model name:\t{self.model}",
                    _DUMP_CAPS_PADDING]
            return "dump_caps", "", caps, caps, 0
        if cmd == "f":
            return "get_freq", "", [f"Frequency: {self.freq}"], [str(self.freq)], 0
        if head == "F":
            self.freq = int(arg)
            return "set_freq", arg, [], [], 0
        if cmd == "m":
            return ("get_mode", "",
                    [f"Mode: {self.mode}", f"Passband: {self.passband}"],
                    [self.mode, str(self.passband)], 0)
        if head == "M":
            self.mode = arg.split()[0] if arg else self.mode
            return "set_mode", arg, [], [], 0
        if cmd == "t":
            if not self.ptt_readable:
                return "get_ptt", "", [], [], _ENAVAIL
            state = "1" if self.transmitting else "0"
            return "get_ptt", "", [f"PTT: {state}"], [state], 0
        if head == "T":
            if not self.ptt_readable:
                # Same unavailability as `t`: a rig with no PTT of its own can
                # neither report one nor be told to key through CAT.
                return "set_ptt", arg, [], [], _ENAVAIL
            self.ptt = arg == "1"
            return "set_ptt", arg, [], [], 0
        return head or cmd, arg, [], [], -1

    def close(self):
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass


class FakePtt:
    """A keying line that records, and can refuse or stick."""

    def __init__(self, *, refuse_assert=False, stick=False, unreadable=False):
        self.refuse_assert, self.stick = refuse_assert, stick
        #: A driver that will not answer TIOCMGET. `sense()` returns None there,
        #: and the panic path must treat that as "nobody knows" rather than as
        #: confirmation.
        self.unreadable = unreadable
        self.line = False
        self.released = False
        #: What the line read back after `release()` — the real class reads it
        #: between the clear and the close, and the panic path asks it first.
        self.released_low: bool | None = None
        self.calls: list[str] = []

    def assert_(self, on: bool) -> None:
        from hfmodem.core.ptt import PttError
        self.calls.append(f"assert {'up' if on else 'down'}")
        if self.released:
            # What the real class does. A fake that quietly re-raises the line
            # after release cannot produce the "asserted after release" failure the
            # panic tests exist to guard, and would report a keyed line as a pass.
            raise PttError("not open")
        if self.refuse_assert:
            raise PttError("the fake refuses")
        if not (self.stick and not on):
            self.line = on

    def sense(self):
        return None if self.released else self.line

    def release(self) -> None:
        self.calls.append("release")
        self.released = True
        if not self.stick:
            self.line = False
        # The readback the real class takes between the clear and the close, and
        # its exact arithmetic: `sense() is False`, so a stuck line and a driver
        # that will not answer both come back False and neither reads as a
        # confirmed unkey. This fake returned None for the unreadable case for a
        # while — a state `RtsPtt.release` cannot produce, so anything written
        # against the fake carried a branch the real driver never takes, on the
        # one path where being wrong leaves a transmitter up.
        self.released_low = not self.unreadable and self.line is False
