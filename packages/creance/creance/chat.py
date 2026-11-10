# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Interactive terminal-to-terminal chat over a selected modem.

creance is a test harness, but the pieces it already has — the dialect-agnostic
`Link` seam, the supervisor, the transcript — make a real chat almost free, and
a chat is the most direct way for a human to feel a link work. So this is a
thin interactive layer, not a new subsystem: connect (or listen), then pump the
keyboard one way and the modem the other, until either end hangs up.

What it is, precisely: a **raw UTF-8 line pipe between two creance stations**
over whichever dialect the modem speaks — VARA, the structured dialect, PTC.
Each line typed is sent as bytes; each byte received is printed. It carries no
application framing of its own, so it talks to another creance `chat` (or to
anything that pipes text), but it does **not** speak VarAC's or any other
client's chat protocol — interop with a stranger's VarAC is a separate, larger
thing and is gated on the modem holding a real connection to that station.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

from hfhost.client import EpochFenced
from hfhost.supervisor import Supervisor
from hfhost.transcript import Transcript

from . import link as linkmod
from .config import Config
from .link import Link, open_link


class _Peer:
    """The link plus the two pumps that make it a conversation."""

    def __init__(self, link: Link, mycall: str, transcript: Transcript,
                 out=sys.stdout, inp=sys.stdin) -> None:
        self.link = link
        self.mycall = mycall
        self.transcript = transcript
        self.out = out
        self.inp = inp
        self.events: queue.Queue = queue.Queue()
        link.subscribe(self.events)
        self._stop = threading.Event()
        self._epoch = link.epoch

    # -- the two directions ------------------------------------------------

    def _pump_out(self) -> None:
        """Keyboard -> modem. A blank EOF (Ctrl-D) ends the chat cleanly."""
        for raw in self.inp:
            if self._stop.is_set():
                return
            text = raw.rstrip("\n")
            if text == "/quit":
                break
            data = (text + "\n").encode("utf-8")
            try:
                self.link.send(data, label="chat")
            except Exception as exc:
                self._emit(f"[send failed: {exc}]")
                break
            self.transcript.data(self.mycall, "tx", data, label="chat")
        self._stop.set()

    def _pump_in(self) -> None:
        """Modem -> screen. Prints partial lines as they arrive; the far end's
        newlines do the framing, exactly as they were typed."""
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.link.recv(None, timeout=0.4, epoch=self._epoch)
            except EpochFenced:
                # A recv fenced to a past epoch is this pump's own retirement;
                # anything else is transient and the pump outlives it. The
                # `hasattr(linkmod, ...)` probe this replaced named the wrong
                # module -- `EpochFenced` is hfhost's -- so it fell through to a
                # bare `except Exception: return` and the handler below was
                # unreachable: one timeout ended the receive side in silence.
                return
            except Exception:
                chunk = b""
            if chunk:
                buf += chunk
                self.transcript.data(self.mycall, "rx", chunk, label="chat")
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._show(line.decode("utf-8", "replace"))
            self._drain_events()
        if buf:
            self._show(buf.decode("utf-8", "replace"))

    # -- link lifecycle events --------------------------------------------

    def _drain_events(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                if ev.kind == linkmod.DISCONNECTED:
                    self._emit("[peer disconnected]")
                    self._stop.set()
                elif ev.kind == linkmod.CONNECTED:
                    self._emit(f"[connected: {ev.peer or '?'}]")
        except queue.Empty:
            pass

    # -- output ------------------------------------------------------------

    def _show(self, text: str) -> None:
        self.out.write(f"\r<< {text}\n")
        self.out.flush()

    def _emit(self, note: str) -> None:
        self.out.write(f"\r{note}\n")
        self.out.flush()

    # -- run ---------------------------------------------------------------

    def run(self) -> None:
        self._emit(f"[chat as {self.mycall} — type to send, /quit or Ctrl-D to end]")
        rx = threading.Thread(target=self._pump_in, name="chat-rx", daemon=True)
        rx.start()
        try:
            self._pump_out()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            rx.join(timeout=2.0)


def _bring_up(config: Config, modem_name: str, host: str,
              transcript: Transcript, supervisor: Supervisor,
              connect_timeout: float) -> Link:
    if not supervisor.ensure(modem_name):
        raise RuntimeError(f"{modem_name}: not reachable")
    supervisor.acquire(modem_name)
    link = open_link(config.modem(modem_name), host, transcript,
                     attach_timeout_s=connect_timeout)
    link.configure(config.site.mycall)
    link.attach()
    return link


def chat(config: Config, modem_name: str, *, dst: str | None = None,
         host: str = "127.0.0.1", results_root: str | Path | None = None,
         connect_timeout: float = 90.0, out=sys.stdout, inp=sys.stdin) -> str:
    """Run one interactive chat. `dst` set → initiator (connect); `dst` None →
    responder (listen for one inbound connect, then chat).

    Returns the outcome: 'ok', 'no_connect', or 'failed:<reason>'.
    """
    root = Path(results_root) if results_root is not None else Path(config.site.results_dir)
    day = time.strftime("%Y-%m-%d")
    tag = "chat-" + time.strftime("%Y%m%dT%H%M%S")
    sid_dir = root / config.site.name / day / tag
    sid_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(str(sid_dir / "transcript.jsonl"), tag)
    supervisor = Supervisor(config, str(sid_dir), host=host)

    link: Link | None = None
    try:
        link = _bring_up(config, modem_name, host, transcript, supervisor,
                         connect_timeout)
        events: queue.Queue = queue.Queue()
        link.subscribe(events)

        if dst is not None:
            out.write(f"calling {dst} on {modem_name}…\n")
            out.flush()
            link.connect(dst)
        else:
            link.set_listen(True)
            out.write(f"listening on {modem_name} as {config.site.mycall}"
                      f" — waiting for a call…\n")
            out.flush()

        if not _wait_connected(link, events, connect_timeout):
            return "no_connect"
        if dst is None:
            link.set_listen(False)

        _Peer(link, config.site.mycall, transcript, out=out, inp=inp).run()
        return "ok"
    except Exception as exc:
        transcript.error(modem_name, f"chat error: {exc!r}")
        return f"failed:{type(exc).__name__}"
    finally:
        if link is not None:
            try:
                if link.connected:
                    link.disconnect()
            except Exception:
                pass
            link.close()
        supervisor.release(modem_name)
        supervisor.stop_all()
        transcript.close()


def _wait_connected(link: Link, events: queue.Queue, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if link.connected:
            return True
        try:
            ev = events.get(timeout=0.25)
        except queue.Empty:
            continue
        if ev.kind == linkmod.CONNECTED:
            return True
        if ev.kind == linkmod.DISCONNECTED:
            return False
    return link.connected
