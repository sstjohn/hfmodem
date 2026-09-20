# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 'modem core' seam for the kestrel host-side VARA-protocol server.

The server in ``server.py`` speaks VARA's *documented TCP host interface* to an
application (VarAC, Pat, Winlink Express, or any conforming host-API client).
Everything below that -- the actual link establishment, the over-the-air
transfer, the flow-control cadence -- is delegated to a :class:`ModemCore`.

This is deliberately the ONE place the real kestrel modem
will plug in later. The server knows nothing about waveforms; it only knows the
narrow event vocabulary defined here. Today we ship a :class:`LoopbackModem`
that fabricates a connection to itself and echoes transmitted payload straight
back to the application's data port, so the whole server is testable end-to-end
against a real client with no radio present.

Split of responsibilities (see server.py for the transport half):

  * The SERVER owns TCP framing, command syntax validation (OK / WRONG),
    configuration state (MYCALL / bandwidth / compression / listen), the
    IAMALIVE heartbeat, and the plumbing of the raw data port.
  * The MODEM owns everything that in a real radio depends on the link: when a
    connection comes up or down (CONNECTED / DISCONNECTED), keying (PTT), the
    TX-queue depth (BUFFER), channel activity (BUSY / PENDING) and registration
    state. It reports these by calling back into a :class:`ModemObserver`.

The server implements :class:`ModemObserver` and turns each callback into the
exact wire message VARA would emit.
"""

from __future__ import annotations

import abc
import queue
import threading
import time
from collections.abc import Sequence


class ModemObserver(abc.ABC):
    """Callbacks a :class:`ModemCore` uses to report link events to the server.

    The server implements this; every method maps to one documented VARA
    modem->host message. All methods may be called from arbitrary modem
    threads, so implementations must be thread-safe.
    """

    @abc.abstractmethod
    def modem_connected(self, src: str, dst: str, bw: str | None) -> None:
        """Link established -> ``CONNECTED src dst [bw]``."""

    @abc.abstractmethod
    def modem_disconnected(self) -> None:
        """Link closed by either end -> ``DISCONNECTED``."""

    @abc.abstractmethod
    def modem_ptt(self, on: bool) -> None:
        """Key/unkey the radio -> ``PTT ON`` / ``PTT OFF``."""

    @abc.abstractmethod
    def modem_buffer(self, nbytes: int) -> None:
        """TX-queue depth changed -> ``BUFFER n``."""

    @abc.abstractmethod
    def modem_data_received(self, blob: bytes) -> None:
        """Payload arrived from the air -> write it to the app's data port."""

    # -- optional / channel-state events (default: no-op) ------------------

    def modem_busy(self, on: bool) -> None:
        """Channel busy detector -> ``BUSY ON`` / ``BUSY OFF``."""

    def modem_pending(self, cancel: bool = False) -> None:
        """Inbound connect detected / aborted -> ``PENDING`` / ``CANCELPENDING``."""

    def modem_registered(self, text: str = "LINK REGISTERED") -> None:
        """Registration status -> e.g. ``LINK REGISTERED``."""

    def modem_bitrate(self, level: int, bps: int, tx: bool) -> None:
        """Speed level of my next over (``tx``) or the frame I just decoded
        (``not tx``) -> ``BITRATE (N) x bps TX`` / ``... RX``. One per over."""

    def modem_snr(self, sn: int) -> None:
        """Decoded signal report. The session renders it as ``SN xx`` ONLY
        while chat mode is on (see :class:`HostSession`), so the modem may call
        it unconditionally on every decode and let the session gate it."""


class ModemCore(abc.ABC):
    """Interface the real kestrel modem (and the loopback stand-in) implement.

    Lifecycle::

        m = SomeModem()
        m.start(observer)      # begin; may emit registration status
        m.set_mycall([...]); m.set_bandwidth("500"); ...
        m.connect("N0CALL", "N0DX")     # async -> observer.modem_connected/…
        m.transmit(b"...")              # async -> BUFFER/PTT, delivery, BUFFER 0
        m.disconnect()                  # graceful: flush TX then DISCONNECTED
        m.stop()

    Everything is asynchronous: the methods return immediately and results are
    reported via the observer. That matches how a real radio behaves and keeps
    the server's command handlers non-blocking.
    """

    @abc.abstractmethod
    def start(self, observer: ModemObserver) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    # -- configuration (host commands that tune the modem) -----------------

    def set_mycall(self, calls: Sequence[str]) -> None:
        self._mycall = list(calls)

    def set_bandwidth(self, bw: str) -> None:
        self._bandwidth = bw

    def set_compression(self, mode: str) -> None:
        self._compression = mode

    def set_listen(self, on: bool) -> None:
        self._listen = on

    # -- session verbs -----------------------------------------------------

    @abc.abstractmethod
    def connect(self, src: str, dst: str,
                vias: Sequence[str] | None = None) -> None:
        """Initiate an ARQ connection. Async -> CONNECTED or DISCONNECTED."""

    @abc.abstractmethod
    def transmit(self, blob: bytes) -> None:
        """Queue payload for over-the-air transmission (host wrote to 8301)."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Graceful close: flush the TX queue, then drop the link."""

    @abc.abstractmethod
    def abort(self) -> None:
        """Immediate ('dirty') disconnect; discards any queued TX."""

    @property
    @abc.abstractmethod
    def connected(self) -> bool: ...


class LoopbackModem(ModemCore):
    """A radio-less modem that connects to itself and echoes payload back.

    NOT a model of VARA's waveform -- it has none. Its only job is to make the
    server exercisable end-to-end: a CONNECT succeeds after a short simulated
    handshake, and every byte written to the data port is looped straight back
    to the same data port after a small simulated over-the-air delay, with the
    documented PTT and BUFFER choreography around it.

    A single background 'TX worker' thread drains the transmit queue serially,
    which mirrors how a real half-duplex modem keys up, sends a burst, and
    reports the queue draining -- and gives deterministic BUFFER accounting.
    """

    #: simulated per-burst over-the-air delay (seconds); small to keep tests fast
    OTA_DELAY = 0.02
    #: simulated connect-handshake airtime (seconds)
    HANDSHAKE_DELAY = 0.05

    def __init__(self, bandwidth: str = "500") -> None:
        self._observer: ModemObserver | None = None
        self._mycall: list[str] = []
        self._bandwidth = bandwidth
        self._compression = "TEXT"
        self._listen = False

        self._connected = False
        self._link = None  # (src, dst) of the current link

        self._lock = threading.Lock()
        self._buffer = 0                    # bytes queued, not yet 'acked'
        self._preconnect: list[bytes] = []  # host writes before CONNECTED (spec §7.4)
        self._tx_q: queue.Queue[bytes | None] = queue.Queue()
        self._running = False
        self._draining_for_disconnect = False
        self._worker: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, observer: ModemObserver) -> None:
        self._observer = observer
        self._running = True
        self._worker = threading.Thread(
            target=self._tx_worker, name="loopback-tx", daemon=True)
        self._worker.start()
        # A real modem reports its registration state shortly after start; the
        # open kestrel core is (conceptually) unlicensed but fully functional.
        observer.modem_registered("LINK REGISTERED")

    def stop(self) -> None:
        self._running = False
        self._tx_q.put(None)  # wake the worker so it can exit

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    # -- session verbs -----------------------------------------------------

    def connect(self, src: str, dst: str,
                vias: Sequence[str] | None = None) -> None:
        threading.Thread(target=self._do_connect, args=(src, dst),
                         daemon=True).start()

    def _do_connect(self, src: str, dst: str) -> None:
        obs = self._observer
        assert obs is not None
        # Simulated ARQ handshake: key up, exchange, key down.
        obs.modem_ptt(True)
        time.sleep(self.HANDSHAKE_DELAY)
        obs.modem_ptt(False)
        # The channel is occupied for the duration of the session. PENDING is
        # deliberately *not* emitted here: the real core raises it only once it
        # has taken the responder role (`fsm.py` `_rx_connect_request`), and a
        # modem connecting outbound never sees its own request arrive.
        obs.modem_busy(True)
        with self._lock:
            self._connected = True
            self._link = (src, dst)
            held = self._preconnect
            self._preconnect = []
            # _buffer already reflects any pre-connect bytes; do NOT zero it.
        obs.modem_connected(src, dst, self._bandwidth)
        # Flush data the host wrote before the session came up (spec §7.4): now
        # that we are CONNECTED it can finally go over the air (loopback echo).
        for blob in held:
            self._tx_q.put(blob)

    def transmit(self, blob: bytes) -> None:
        if not blob:
            return
        with self._lock:
            self._buffer += len(blob)
            depth = self._buffer
            preconnect = not self._connected
            if preconnect:
                # Pre-session write: buffer it against the queue (spec §7.4) and
                # flush it once CONNECTED, rather than dropping it on the floor.
                self._preconnect.append(blob)
        # Report the enqueue immediately (VARA emits BUFFER on enqueue), whether
        # or not the session is up yet -- the bytes are really queued.
        assert self._observer is not None
        self._observer.modem_buffer(depth)
        if not preconnect:
            self._tx_q.put(blob)

    def _tx_worker(self) -> None:
        """Serially drain the TX queue. Keys PTT for each burst, delivers the
        payload to the far end (== ourselves, looped back), then reports the
        queue depth shrinking as the far end 'acks'."""
        obs = self._observer
        assert obs is not None
        while self._running:
            item = self._tx_q.get()
            if item is None:
                continue  # wake-up (stop() or disconnect drain probe)
            burst = [item]
            # Coalesce anything already waiting into one keyed burst.
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
                connected = self._connected
            for b in burst:
                if connected:
                    obs.modem_data_received(b)  # loopback: echo to host
                with self._lock:
                    self._buffer = max(0, self._buffer - len(b))
                    depth = self._buffer
                obs.modem_buffer(depth)
            obs.modem_ptt(False)

    def disconnect(self) -> None:
        threading.Thread(target=self._do_disconnect, daemon=True).start()

    def _do_disconnect(self) -> None:
        # VARA promises DISCONNECT flushes the TX queue first.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._buffer == 0:
                    break
            time.sleep(0.01)
        self._finish_disconnect()

    def abort(self) -> None:
        # Immediate: discard queued TX and drop.
        try:
            while True:
                self._tx_q.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._buffer = 0
            self._preconnect = []
        self._finish_disconnect()

    def _finish_disconnect(self) -> None:
        obs = self._observer
        with self._lock:
            was = self._connected
            self._connected = False
            self._link = None
            self._buffer = 0
            self._preconnect = []
        if obs is not None:
            obs.modem_disconnected()
            if was:
                obs.modem_busy(False)   # cleared at teardown, as `fsm.py` does
