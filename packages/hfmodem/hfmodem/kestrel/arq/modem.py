# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""KestrelModem: the real ARQ modem behind the host-API server's ModemCore seam.

Wires together:
    host API (host_api/ModemCore + ModemObserver)  <->  ArqFsm  <->  PHY (proven
    VARA HF 500 data burst codec)  <->  AudioChannel (a half-duplex medium).

Two KestrelModems sharing one :class:`AudioChannel` form a kestrel<->kestrel
link: connect handshake, stop-and-wait data transfer with ACK/retransmit/
gear-shift, and 3-burst disconnect — every frame a real synthesised burst
decoded by the frozen receiver.

Concurrency: every event (host command, decoded RX frame, timer tick) is pushed
onto one queue and processed by a single worker thread, so the FSM never needs
locking. Decoding (~0.7 s/burst) happens on the RX thread, off the worker.
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence

import numpy as np

from ..host.modem_core import ModemCore, ModemObserver
from . import frames as F
from . import phy
from .fsm import ArqConfig, ArqFsm, ArqIO, State


# --------------------------------------------------------------------------- #
class _Endpoint:
    """One side of the half-duplex medium: transmit -> peer's inbox."""
    def __init__(self, name: str):
        self.name = name
        self.inbox: queue.Queue = queue.Queue()
        self._peer: _Endpoint | None = None

    def transmit(self, samples: np.ndarray) -> None:
        if self._peer is not None:
            self._peer.inbox.put(np.asarray(samples, float))

    def recv(self, timeout: float | None = None):
        return self.inbox.get(timeout=timeout)


class AudioChannel:
    """A shared half-duplex audio medium with two endpoints (A, B).

    Bursts are real 48 kHz sample arrays; stop-and-wait guarantees only one side
    keys at a time, so no summation/collision model is needed. An optional
    ``loss`` predicate can drop bursts to exercise the retransmit path.
    """
    def __init__(self, loss=None):
        self.a = _Endpoint("A")
        self.b = _Endpoint("B")
        self.a._peer = self.b
        self.b._peer = self.a
        if loss is not None:
            self._install_loss(loss)

    def _install_loss(self, loss):
        counter = {"n": 0}
        def wrap(ep):
            orig = ep.transmit
            def t(samples):
                i = counter["n"]; counter["n"] += 1
                if loss(i):
                    return
                orig(samples)
            ep.transmit = t  # type: ignore
        wrap(self.a); wrap(self.b)


# --------------------------------------------------------------------------- #
class KestrelModem(ModemCore, ArqIO):
    """ModemCore that runs the ARQ FSM over an AudioChannel endpoint."""

    def __init__(self, endpoint: _Endpoint, config: ArqConfig | None = None):
        self._ep = endpoint
        self._cfg = config or ArqConfig()
        self._observer: ModemObserver | None = None
        self._bandwidth = "500"
        self._mycall = []
        self._compression = "TEXT"
        self._listen = False

        self._fsm = ArqFsm(self, self._cfg)
        self._events: queue.Queue = queue.Queue()
        self._running = False
        self._threads = []
        self._busy_state: bool | None = None
        self._log_fn = None  # set by run_server / tests if desired

    # ---- ModemCore lifecycle --------------------------------------------
    def start(self, observer) -> None:
        self._observer = observer
        self._running = True
        self._threads = [
            threading.Thread(target=self._worker, name="arq-worker", daemon=True),
            threading.Thread(target=self._rx_loop, name="arq-rx", daemon=True),
            threading.Thread(target=self._timer_loop, name="arq-timer", daemon=True),
        ]
        for t in self._threads:
            t.start()
        # registration status shortly after start (open core = unregistered but
        # fully functional; the loopback core reports REGISTERED).
        observer.modem_registered("LINK REGISTERED")

    def stop(self) -> None:
        self._running = False
        self._events.put(("stop",))
        self._ep.inbox.put(None)  # unblock the rx thread

    @property
    def connected(self) -> bool:
        return self._fsm.state == State.CONNECTED

    # ---- ModemCore config / verbs (push events; never touch FSM directly) --
    def set_bandwidth(self, bw: str) -> None:
        self._bandwidth = bw

    def set_listen(self, on: bool) -> None:
        self._listen = on
        self._events.put(("host_listen", on))

    def connect(self, src: str, dst: str, vias: Sequence[str] | None = None) -> None:
        self._events.put(("host_connect", src, dst, self._bandwidth))

    def transmit(self, blob: bytes) -> None:
        self._events.put(("host_data", bytes(blob)))

    def disconnect(self) -> None:
        self._events.put(("host_disconnect",))

    def abort(self) -> None:
        self._events.put(("host_abort",))

    # ---- worker / rx / timer threads ------------------------------------
    def _worker(self) -> None:
        while self._running:
            ev = self._events.get()
            if ev is None or ev[0] == "stop":
                break
            try:
                self._dispatch(ev)
            except Exception as e:  # keep the modem alive; log and continue
                self.log(f"worker error: {e!r}")

    def _dispatch(self, ev) -> None:
        kind = ev[0]
        if kind == "host_listen":
            self._fsm.on_host_listen(ev[1])
        elif kind == "host_connect":
            self._fsm.on_host_connect(ev[1], ev[2], ev[3])
        elif kind == "host_data":
            self._fsm.on_host_data(ev[1])
        elif kind == "host_disconnect":
            self._fsm.on_host_disconnect()
        elif kind == "host_abort":
            self._fsm.on_host_abort()
        elif kind == "rx_frame":
            self._fsm.on_rx_frame(ev[1])
        elif kind == "rx_token":
            self._fsm.on_rx_token(ev[1])
        elif kind == "timer":
            self._fsm.on_timer()

    def _rx_loop(self) -> None:
        while self._running:
            try:
                samples = self._ep.recv(timeout=0.5)
            except queue.Empty:
                continue
            if samples is None:
                break
            # A short burst is one of VARA's control bursts, not a DATA or
            # native-control over — length alone separates them. Which reader
            # names it is the bandwidth's: BW500's four DBPSK tokens
            # [spec 02 §2.6], BW2300's single index-modulated answer
            # [spec 04 §4.2C]. Handing a BW2300 burst to the DBPSK table names
            # nothing that has ever been on the air.
            if len(samples) < phy.CONTROL_TOKEN_MAX_SAMPLES:
                tok = phy.detect_token(samples, self._bandwidth)
                if tok is not None:
                    self._events.put(("rx_token", tok.name))
                continue
            try:
                fr = phy.decode(samples, self._bandwidth)
                rx = F.classify(fr)
            except Exception as e:
                self.log(f"decode error: {e!r}")
                continue
            self._events.put(("rx_frame", rx))

    def _timer_loop(self) -> None:
        while self._running:
            time.sleep(0.05)
            self._events.put(("timer",))

    # ---- ArqIO (FSM -> host API + PHY) ----------------------------------
    def key(self, on: bool) -> None:
        if on:
            time.sleep(self._cfg.turnaround_s)   # RX->TX turnaround [spec 05 §5.5]
        if self._observer is not None:
            self._observer.modem_ptt(on)

    def tx(self, payload: bytes, marker: int, bw: str = "500", level=None) -> None:
        self._ep.transmit(phy.render(payload, marker, bw, level))

    def tx_token(self, name: str, bw: str = "500") -> None:
        self._ep.transmit(phy.render_token(name, bw))

    def on_busy(self, on: bool) -> None:
        if on == self._busy_state:
            return
        self._busy_state = on
        if self._observer is not None:
            self._observer.modem_busy(on)

    def on_pending(self, cancel: bool = False) -> None:
        if self._observer is not None:
            self._observer.modem_pending(cancel)

    def on_connected(self, src: str, dst: str, bw: str) -> None:
        if self._observer is not None:
            self._observer.modem_connected(src, dst, bw)

    def on_disconnected(self) -> None:
        self._busy_state = None
        if self._observer is not None:
            self._observer.modem_disconnected()

    def on_buffer(self, nbytes: int) -> None:
        if self._observer is not None:
            self._observer.modem_buffer(nbytes)

    def on_deliver(self, blob: bytes) -> None:
        if self._observer is not None:
            self._observer.modem_data_received(blob)

    def log(self, msg: str) -> None:
        if self._log_fn:
            self._log_fn(f"[{self._ep.name}] {msg}")
