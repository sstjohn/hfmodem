# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The modem-core seam for besra's ARDOP host server.

`server.py` speaks ARDOP's host dialect to an application; everything below —
link establishment, over-the-air transfer, flow control — is delegated to a
`ModemCore`. `besra.arq.modem.BesraModem` is the real one (PHY + ARQ); the
`LoopbackModem` here fabricates a link to itself, so the dialect is exercisable
against a real client (Pat) with no radio and no waveform.

Split of responsibilities (see `server.py` for the transport half):

  * The SERVER owns TCP framing, command syntax, the ``<CMD> now <value>`` echo,
    configuration state, and the raw data port.
  * The MODEM owns everything a real radio makes link-dependent: the NEWSTATE
    machine, keying (PTT), TX-queue depth (BUFFER), channel activity
    (BUSY / PENDING), and the connect/disconnect edges. It reports these by
    calling back into a `ModemObserver`.

Unlike VARA's server, which infers state from side effects, ARDOP's modem
*publishes* its state: the core drives NEWSTATE transitions explicitly and the
server relays them verbatim.
"""

from __future__ import annotations

import abc
import queue
import threading
import time

from . import protocol as P


class ModemObserver(abc.ABC):
    """Callbacks a `ModemCore` uses to report link events to the server.

    The server implements this; each method maps to one ARDOP modem→host line.
    Methods may be called from arbitrary modem threads and must be thread-safe.
    """

    @abc.abstractmethod
    def modem_newstate(self, state: str) -> None:
        """State-machine transition -> ``NEWSTATE <state>`` (a ``P.ArdopState``)."""

    @abc.abstractmethod
    def modem_connected(self, remote: str, bw: int) -> None:
        """ARQ link up -> ``CONNECTED <remote> <bw>``."""

    @abc.abstractmethod
    def modem_disconnected(self) -> None:
        """ARQ link down -> ``DISCONNECTED``."""

    @abc.abstractmethod
    def modem_ptt(self, on: bool) -> None:
        """Key/unkey -> ``PTT TRUE`` / ``PTT FALSE``."""

    @abc.abstractmethod
    def modem_buffer(self, nbytes: int) -> None:
        """TX-queue depth changed -> ``BUFFER <n>``."""

    @abc.abstractmethod
    def modem_data_received(self, kind: str, blob: bytes) -> None:
        """Payload arrived from the air -> framed onto the app's data port.
        ``kind`` is one of ``ARQ`` / ``FEC`` / ``ERR`` / ``IDF``."""

    # -- optional / channel-state events (default: no-op) ------------------

    def modem_busy(self, on: bool) -> None:
        """Channel busy detector -> ``BUSY TRUE`` / ``BUSY FALSE``."""

    def modem_pending(self, cancel: bool = False) -> None:
        """Inbound connect detected / withdrawn -> ``PENDING`` / ``CANCELPENDING``."""

    def modem_target(self, call: str) -> None:
        """The callsign a heard connable frame is addressed to -> ``TARGET <call>``."""

    def modem_status(self, text: str) -> None:
        """Human-readable status -> ``STATUS <text>``."""

    def modem_fault(self, text: str) -> None:
        """Recoverable fault -> ``FAULT <text>``."""


class ModemCore(abc.ABC):
    """Interface besra's real modem (and the loopback stand-in) implement.

    Everything is asynchronous: methods return at once and results arrive via the
    observer, matching a real half-duplex radio and keeping the server's command
    handlers non-blocking.
    """

    @abc.abstractmethod
    def start(self, observer: ModemObserver) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    # -- configuration (host set-commands tune the modem) ------------------

    def set_mycall(self, call: str) -> None:
        self._mycall = call

    def set_gridsquare(self, grid: str) -> None:
        self._gridsquare = grid

    def set_bandwidth(self, hz: int, forced: bool) -> None:
        """The ARQBW: one of 200/500/1000/2000, MAX (negotiate down) or FORCED."""
        self._bandwidth = hz
        self._bw_forced = forced

    def set_listen(self, on: bool) -> None:
        self._listen = on

    def set_protocolmode(self, mode: str) -> None:
        """``ARQ`` or ``FEC``."""
        self._protocolmode = mode

    # -- session verbs -----------------------------------------------------

    @abc.abstractmethod
    def connect(self, target: str, repeats: int = 5) -> None:
        """Initiate an ARQ connection (become ISS). Async -> NEWSTATE/CONNECTED
        or, on failure, FAULT then NEWSTATE DISC. ``repeats`` is the connect-
        request retry budget (ARDOP's ``ARQCALL <target> <repeat>``)."""

    @abc.abstractmethod
    def transmit(self, blob: bytes) -> None:
        """Queue payload for over-the-air transmission (host wrote to the data
        port). Enqueue is reported via BUFFER."""

    @abc.abstractmethod
    def purge_buffer(self) -> None:
        """Discard queued payload without touching the link (host
        ``PURGEBUFFER`` / ``DATATOSEND 0`` / ``CL``). The resulting depth is
        reported through the observer, as any other queue change is."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Graceful close: flush the TX queue, then drop the link."""

    @abc.abstractmethod
    def abort(self) -> None:
        """Immediate ('dirty') disconnect; discards any queued TX."""

    def send_id(self) -> None:
        """Transmit an ID frame now (host ``SENDID``). Default: no-op."""

    @property
    @abc.abstractmethod
    def state(self) -> str: ...

    @property
    @abc.abstractmethod
    def queued(self) -> int:
        """Payload bytes still waiting to go out — the ``BUFFER`` /
        ``DATATOSEND`` value."""

    @property
    def connected(self) -> bool:
        return self.state in P.CONNECTED_STATES


class LoopbackModem(ModemCore):
    """A radio-less modem that connects to itself and echoes payload back.

    Not a model of any ARDOP waveform — it has none. Its job is to make the
    server exercisable end to end against a real client: a connect succeeds after
    a short simulated handshake with the full NEWSTATE choreography, and every
    byte written to the data port loops straight back after a small simulated
    over-the-air delay, wrapped in the documented PTT/BUFFER accounting. A single
    TX-worker thread drains the queue serially, mirroring a half-duplex modem.
    """

    OTA_DELAY = 0.02          # simulated per-burst airtime (s)
    HANDSHAKE_DELAY = 0.05    # simulated connect handshake (s)

    def __init__(self, bandwidth: int = 500) -> None:
        self._observer: ModemObserver | None = None
        self._mycall = ""
        self._gridsquare = ""
        self._bandwidth = bandwidth
        self._bw_forced = False
        self._listen = False
        self._protocolmode = "ARQ"

        self._lock = threading.Lock()
        self._state = P.ArdopState.DISC
        self._remote = ""
        self._buffer = 0
        self._preconnect: list[bytes] = []
        self._tx_q: "queue.Queue[bytes | None]" = queue.Queue()
        self._running = False
        self._worker: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, observer: ModemObserver) -> None:
        self._observer = observer
        self._running = True
        self._worker = threading.Thread(
            target=self._tx_worker, name="loopback-tx", daemon=True)
        self._worker.start()
        self._to_state(P.ArdopState.DISC)

    def stop(self) -> None:
        self._running = False
        self._tx_q.put(None)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def queued(self) -> int:
        with self._lock:
            return self._buffer

    def _to_state(self, state: str) -> None:
        with self._lock:
            self._state = state
        assert self._observer is not None
        self._observer.modem_newstate(state)

    # -- session verbs -----------------------------------------------------

    def connect(self, target: str, repeats: int = 5) -> None:
        threading.Thread(target=self._do_connect, args=(target,),
                         daemon=True).start()

    def _do_connect(self, target: str) -> None:
        obs = self._observer
        assert obs is not None
        # Simulated ARQ handshake: key up, send the connect request, key down.
        obs.modem_ptt(True)
        time.sleep(self.HANDSHAKE_DELAY)
        obs.modem_ptt(False)
        with self._lock:
            self._remote = target
            held, self._preconnect = self._preconnect, []
        # The caller is the Information Sending Station once the link is up.
        self._to_state(P.ArdopState.ISS)
        obs.modem_connected(target, self._bandwidth)
        for blob in held:
            self._tx_q.put(blob)

    def transmit(self, blob: bytes) -> None:
        if not blob:
            return
        with self._lock:
            self._buffer += len(blob)
            depth = self._buffer
            preconnect = self._state == P.ArdopState.DISC
            if preconnect:
                self._preconnect.append(blob)
        assert self._observer is not None
        self._observer.modem_buffer(depth)
        if not preconnect:
            self._tx_q.put(blob)

    def _tx_worker(self) -> None:
        obs = self._observer
        assert obs is not None
        while self._running:
            item = self._tx_q.get()
            if item is None:
                continue
            burst = [item]
            try:
                while True:
                    nxt = self._tx_q.get_nowait()
                    if nxt is None:
                        break
                    burst.append(nxt)
            except queue.Empty:
                pass

            obs.modem_ptt(True)
            time.sleep(self.OTA_DELAY)
            with self._lock:
                connected = self._state != P.ArdopState.DISC
            for b in burst:
                if connected:
                    obs.modem_data_received("ARQ", b)   # loopback: echo to host
                with self._lock:
                    self._buffer = max(0, self._buffer - len(b))
                    depth = self._buffer
                obs.modem_buffer(depth)
            obs.modem_ptt(False)

    def disconnect(self) -> None:
        threading.Thread(target=self._do_disconnect, daemon=True).start()

    def _do_disconnect(self) -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._buffer == 0:
                    break
            time.sleep(0.01)
        self._finish_disconnect()

    def purge_buffer(self) -> None:
        try:
            while True:
                self._tx_q.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._buffer = 0
            self._preconnect = []
        if self._observer is not None:
            self._observer.modem_buffer(0)

    def abort(self) -> None:
        self.purge_buffer()
        self._finish_disconnect()

    def _finish_disconnect(self) -> None:
        obs = self._observer
        with self._lock:
            self._remote = ""
            self._buffer = 0
            self._preconnect = []
        if obs is not None:
            obs.modem_disconnected()
        self._to_state(P.ArdopState.DISC)
