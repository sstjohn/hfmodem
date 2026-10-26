# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`BesraModem` — the real ARDOP modem: PHY + ARQ behind the host `ModemCore` seam.

This binds the three validated layers into one modem:

  * the ARQ session (`besra.arq.session`) drives the NEWSTATE protocol,
  * the modulator (`besra.phy.modulator`) renders each frame the session sends
    to 12 kHz audio, and
  * the demodulator (`besra.phy.demodulator`) turns received audio back into the
    decoded frames the session consumes.

The modem renders into and decodes out of an injected audio link (`audio_out`
for transmit, `receive_audio` for receive). Time and I/O are driven one of two
ways over the same core:

  * **passive** — a synchronous pump (`besra.sim.air.StepAir`) calls
    `receive_audio`/`tick` directly, for deterministic radioless tests; or
  * **threaded** — `start(observer, threaded=True)` runs a background pump that
    drains delivered audio and advances timers off the wall clock, so the host
    server can drive it live. A single lock serialises the ARQ session across the
    host thread and the pump thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import numpy as np

from ..frame import callsign, frame as F
from ..host import protocol as P
from ..host.modem_core import ModemCore, ModemObserver
from ..phy import modulator
from ..phy.cwid import render_cwid
from ..phy.demodulator import SAMPLE_RATE, DecodedFrame, Demodulator
from .session import ArqSession, peer_quality

log = logging.getLogger(__name__)


class _ObserverAdapter:
    """Maps the ARQ session's `Observer` calls onto the host `ModemObserver`.

    The two vocabularies differ only in spelling (`newstate` vs `modem_newstate`);
    this is the thin seam that keeps the session ignorant of the host dialect."""

    def __init__(self, obs: ModemObserver) -> None:
        self._obs = obs

    def newstate(self, state: str) -> None:
        self._obs.modem_newstate(state)

    def connected(self, remote: str, bw: int) -> None:
        self._obs.modem_connected(remote, bw)

    def disconnected(self) -> None:
        self._obs.modem_disconnected()

    def data_received(self, kind: str, blob: bytes) -> None:
        self._obs.modem_data_received(kind, blob)

    def buffer(self, nbytes: int) -> None:
        self._obs.modem_buffer(nbytes)

    def pending(self, cancel: bool = False) -> None:
        self._obs.modem_pending(cancel)

    def target(self, call: str) -> None:
        self._obs.modem_target(call)

    def status(self, text: str) -> None:
        self._obs.modem_status(text)


class BesraModem(ModemCore):
    """A radio-capable ARDOP modem. Transmits by rendering frames into its audio
    link; receives by decoding audio the link delivers via `receive_audio`."""

    #: silence padded around a delivered burst so the leader detector has room to
    #: acquire and the decoder to flush (the demod's own convention).
    _PAD_LEAD = 2400
    _PAD_TAIL = 4800

    def __init__(self, *, bandwidth: int = 500, timeout_s: int = 90,
                 leader_ms: int = modulator.DEFAULT_LEADER_MS,
                 threaded: bool = False) -> None:
        self._start_threaded = threaded    # host server drives live; sim drives passively
        self._mycall = ""
        self._gridsquare = ""
        self._bandwidth = bandwidth
        self._bw_forced = False
        self._listen = False
        self._protocolmode = "ARQ"
        self._timeout_s = timeout_s
        self._leader_ms = leader_ms
        self._trailer_ms = modulator.DEFAULT_TRAILER_MS
        self._codec_online = True
        self._drivelevel = 100
        self._autobreak = True
        self._cwid = 'FALSE'
        self._closing_id = None
        self._tx_bandwidth = None
        self._last_id_at = 0.0

        self._observer: ModemObserver | None = None
        self.retired = deque(maxlen=16)
        self.last_payload_status = None
        self._session: ArqSession | None = None
        self._demod = Demodulator(expect_session=self.expected_session,
                                  rx_epoch=self.rx_epoch)
        self._clock = 0.0
        self.audio_out = None            # set by the air / device: called with int16 samples

        self._lock = threading.RLock()
        self._rx_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._pump: threading.Thread | None = None
        self._running = False
        self._generation = 0

    # -- ModemCore lifecycle ----------------------------------------------

    def start(self, observer: ModemObserver, *, threaded: bool | None = None) -> None:
        if self._session is not None or self._pump is not None and self._pump.is_alive():
            raise RuntimeError('ARDOP core already started or still retiring')
        self._generation += 1
        self._rx_q = queue.Queue()
        self._observer = observer
        self._session = ArqSession(
            self._mycall, transport=self, observer=_ObserverAdapter(observer),
            bandwidth=self._bandwidth, timeout_s=self._timeout_s,
            listen=self._listen, identify=self._identify_on_close)
        self._session.tick(self._clock)      # publishes the initial NEWSTATE DISC
        self._session.autobreak = self._autobreak
        if self._start_threaded if threaded is None else threaded:
            self._t0 = time.monotonic()
            self._running = True
            self._pump = threading.Thread(target=self._run, name="besra-pump",
                                          daemon=True)
            self._pump.start()

    def stop(self) -> None:
        self._running = False
        self._generation += 1
        with self._lock:
            self._closing_id = None
            if self._session is not None:
                self._session.abort()
                session = self._session
                self.retired.extend(dict(record) for record in session.retired)
                self.last_payload_status = dict(accepted=session.accepted_total,
                    confirmed=session.confirmed_total, discarded=session.discarded_total,
                    unconfirmed=session.last_unconfirmed)
            self._session = None
        pump = self._pump
        if pump is not None and pump is not threading.current_thread():
            pump.join(30)
            if pump.is_alive():
                raise RuntimeError('ARDOP core pump did not retire')
            self._pump = None

    def status(self, text: str) -> None:
        """Say something to the host on the STATUS line, from outside the session —
        the radio backend's receive funnel is the one thing here the ARQ layer
        cannot see. Silent before `start`, when there is no host to say it to."""
        if self._observer is not None:
            self._observer.modem_status(text)

    @property
    def state(self) -> str:
        if not self._codec_online:
            return P.ArdopState.OFFLINE
        return self._session.state if self._session else P.ArdopState.OFFLINE

    @property
    def connected(self) -> bool:
        # `ModemCore.connected` reads `state` alone, which is ISS from the first
        # keyed ConReq — before any ConAck. Defer to the session's own notion,
        # which knows the connect handshake from a link a peer actually granted.
        return self._session.connected if self._session else False

    @property
    def queued(self) -> int:
        with self._lock:
            return self._session.queued if self._session else 0

    def expected_session(self) -> int | None:
        """The ARQ session id the demodulator should treat as corroborating a bare
        control frame right now (`ArqSession.expected_session`), or None. A method
        rather than a property so it can be handed to a `Demodulator` as its
        `expect_session` callable — the radio backend's decoder polls it live."""
        s = self._session
        return s.expected_session if s is not None else None

    def rx_epoch(self) -> int:
        """`ArqSession.rx_epoch` for the demodulator's memory ARQ, polled the same
        way and for the same reason as `expected_session`."""
        s = self._session
        return s.rx_epoch if s is not None else 0

    # -- config (the host server pushes these before start) ----------------

    def set_mycall(self, call: str) -> None:
        # The host sets MYCALL after start(), so push it into the live session.
        with self._lock:
            if self._closing_id is not None:
                raise ValueError('closing identification is pending')
            self._mycall = call
            if self._session is not None:
                self._session.mycall = call

    def set_listen(self, on: bool) -> None:
        self._listen = on
        with self._lock:
            if self._session is not None:
                self._session.listen = on and self._protocolmode == "ARQ"

    def set_protocolmode(self, mode: str) -> None:
        with self._lock:
            self._protocolmode = mode
            if self._session is not None:
                self._session.listen = self._listen and mode == "ARQ"

    def set_codec(self, on: bool) -> None:
        with self._lock:
            if (not on and (self._closing_id is not None or self._session is not None
                    and (self._session.state != P.ArdopState.DISC or self._session.queued))):
                raise ValueError("CODEC stop requires an idle lane with an empty queue")
            self._codec_online = on
            if self._observer is not None:
                self._observer.modem_newstate(self.state)

    def set_drivelevel(self, level: int) -> None:
        if not 0 <= level <= 100:
            raise ValueError("drive must be 0..100")
        with self._lock:
            self._drivelevel = level

    def set_autobreak(self, on: bool) -> None:
        with self._lock:
            self._autobreak = on
            if self._session is not None:
                self._session.autobreak = on

    def set_arqtimeout(self, seconds: int) -> None:
        with self._lock:
            self._timeout_s = seconds
            if self._session is not None:
                self._session._timeout_s = seconds

    def set_leader(self, milliseconds: int) -> None:
        if not 120 <= milliseconds <= 2500:
            raise ValueError('leader must be 120..2500 milliseconds')
        with self._lock:
            self._leader_ms = milliseconds

    def set_trailer(self, milliseconds: int) -> None:
        if not 0 <= milliseconds <= 200:
            raise ValueError('trailer must be 0..200 milliseconds')
        with self._lock:
            self._trailer_ms = milliseconds

    def set_cwid(self, mode: bool | str) -> None:
        if isinstance(mode, bool):
            mode = 'TRUE' if mode else 'FALSE'
        if mode not in ('TRUE', 'FALSE', 'ONOFF'):
            raise ValueError('CWID must be TRUE, FALSE or ONOFF')
        with self._lock:
            self._cwid = mode

    @property
    def identification_pending(self):
        return self._closing_id is not None

    @property
    def emission_bandwidth(self):
        return self._tx_bandwidth or self._bandwidth

    def _require_auxiliary_idle(self):
        if (not self._codec_online or self.state != P.ArdopState.DISC
                or self.identification_pending):
            raise ValueError('unconnected transmission requires an idle online lane')

    def send_id(self):
        with self._lock:
            self._require_auxiliary_idle()
            self._send_identification(self._mycall, self._gridsquare, self._cwid)

    def _send_identification(self, call, grid, cwid):
        # ARDOP_Win uses the explicit NO-GS marker when no locator is set.
        grid = grid.upper() if grid else 'NO-GS'
        payload = callsign.pack_callsign(call) + callsign.packed6_encode(grid)
        samples = modulator.render_frame(0x30, payload=payload, session_id=0xff,
                                        leader_ms=self._leader_ms,
                                        trailer_ms=self._trailer_ms)
        if cwid != 'FALSE':
            samples = np.concatenate((samples, np.zeros(4800, dtype='<i2'),
                                      render_cwid(call, on_off=cwid == 'ONOFF')))
        # Morse's two tones and transitions need the 500 Hz profile even if
        # the ARQ bandwidth is 200. Declare it before the station sink is called.
        self._tx_bandwidth = max(self._bandwidth, 500 if cwid != 'FALSE' else 200)
        try:
            duration = self._emit_samples(samples, label='IDFrame' +
                                          (' + CWID' if cwid != 'FALSE' else ''))
        finally:
            self._tx_bandwidth = None
        self._last_id_at = self._clock + duration
        if self._observer is not None:
            self._observer.modem_data_received('IDF', f' {call}:[{grid}] '.encode('ascii'))
        return duration

    def _identify_on_close(self, reason):
        identity = (self._mycall, self._gridsquare, self._cwid)
        if reason == 'peer_disc':
            # END precedes the caller's immediate ID. The supplied Windows
            # implementation schedules its own final ID five seconds later.
            # Physical channel/timing qualification remains a separate gate;
            # the station arbiter enforces ownership, not channel occupancy.
            self._closing_id = (self._clock + 5, identity)
        else:
            self._send_identification(*identity)

    def two_tone_test(self):
        with self._lock:
            self._require_auxiliary_idle()
            # 250 alternating 20 ms leader symbols, with the sync repetition.
            signs = (-1) ** np.arange(250)
            signs[-1] *= -1
            self._emit_samples((signs[:, None]*modulator.LEADER_50BD).ravel(), label='two-tone test')

    def break_link(self) -> None:
        with self._lock:
            if self._session is not None and self._codec_online:
                self._session.request_break()

    @property
    def bandwidth(self) -> int:
        """The session bandwidth in force, which is also the passband a burst
        occupies — `radio.RadioLink` states the emission from it, and a host
        `ARQBW` moves both at once."""
        return self._bandwidth

    def set_bandwidth(self, hz: int, forced: bool) -> None:
        # ARQBW arrives after start() too, so push it into the live session —
        # otherwise the next ConReq goes out at whatever the modem was built
        # with, and the host's setting is a stored number that governs nothing.
        self._bandwidth = hz
        self._bw_forced = forced
        with self._lock:
            if self._session is not None:
                self._session.bandwidth = hz

    # -- ModemCore session verbs ------------------------------------------

    def connect(self, target: str, repeats: int = 5) -> None:
        with self._lock:
            if not self._codec_online:
                raise ValueError("codec is offline")
            if self._protocolmode != "ARQ":
                raise ValueError("protocol mode is not ARQ")
            if self.identification_pending:
                raise ValueError('closing identification is pending')
            if self._session is None:
                raise ValueError('modem session is not running')
            self._session.connect(target, repeats)

    def transmit(self, blob: bytes) -> None:
        with self._lock:
            if not self._codec_online:
                raise ValueError("codec is offline")
            if self._session is None:
                return
            self._session.queue_data(blob)

    def purge_buffer(self) -> None:
        with self._lock:
            if self._session is None:
                return
            self._session.purge()

    def disconnect(self) -> None:
        with self._lock:
            if self._session is None:
                return
            self._session.disconnect()

    def abort(self) -> None:
        with self._lock:
            self._closing_id = None
            if self._session is None:
                return
            self._session.abort()

    # -- the ARQ session's Transport: render a frame into the audio link ---

    def send(self, frame_type: int, payload: bytes, session_id: int) -> float:
        identification_duration = 0.0
        if (self.connected and (frame_type == 0x24 or F.FRAMES[frame_type].k >= 16)
                and self._clock - self._last_id_at >= 600):
            # Insert periodic identification only on an ISS-owned transmit
            # turn, never between a received data frame and its timely ACK.
            identification_duration = self._send_identification(
                self._mycall, self._gridsquare, self._cwid)
        # ConReq/Ping/ID (unconnected frames) go out with Session ID 0xFF; both
        # ends re-derive the real session from the callsigns (spec §2.1).
        wire_session = 0xFF if F.FRAMES[frame_type].forces_session else session_id
        samples = modulator.render_frame(frame_type, payload=payload,
                                         session_id=wire_session,
                                         leader_ms=self._leader_ms,
                                         trailer_ms=self._trailer_ms)
        return identification_duration + self._emit_samples(samples, label=F.FRAMES[frame_type].name)

    def _emit_samples(self, samples, *, label=''):
        if self._drivelevel != 100:
            samples = np.rint(samples.astype(float) * self._drivelevel / 100).astype("<i2")
        else:
            samples = np.asarray(samples, dtype='<i2')
        log.info("TX %s %.2fs", label, len(samples) / SAMPLE_RATE)
        obs = self._observer
        generation = self._generation
        try:
            if obs is not None:
                obs.modem_ptt(True)
            if generation != self._generation:
                raise ConnectionAbortedError('ARDOP transmission cancelled before submission')
            if self.audio_out is not None:
                self.audio_out(samples)
            if generation != self._generation:
                raise ConnectionAbortedError('ARDOP transmission retired during submission')
        finally:
            if obs is not None:
                obs.modem_ptt(False)
        # The transmit-frame duration, so the session can schedule its repeat from
        # when the frame *ends* (the reference anchors dttNextPlay after playback),
        # not from when it starts — otherwise a 1.75 s ConReq eats the listen window.
        return len(samples) / SAMPLE_RATE

    # -- receive / timing --------------------------------------------------

    def deliver_rx(self, samples: np.ndarray) -> None:
        """Hand received audio to the modem. In threaded mode it is queued for
        the pump; passively it decodes inline."""
        if self._running:
            self._rx_q.put(samples)
        else:
            self.receive_audio(samples)

    def receive_audio(self, samples: np.ndarray) -> None:
        if not self._codec_online:
            return
        padded = np.concatenate([
            np.zeros(self._PAD_LEAD, dtype="<i2"), samples,
            np.zeros(self._PAD_TAIL, dtype="<i2")])
        self.receive_frames(self._demod.decode(padded))

    def receive_frames(self, frames: list[DecodedFrame]) -> None:
        """Feed already-decoded frames to the session. The radio backend decodes a
        rolling window on its own thread and reports each frame once it has decoded,
        so it enters here rather than through `receive_audio` — its windows overlap,
        and the same frame decoded a second time would replay its DATAACK and clear a
        data frame the peer never acknowledged. The one repeat it does hand up is a
        position it had only failed at before (`RollingDecoder.DEDUP_S`), which
        arrives as a NAK this end now has the payload to correct."""
        for frame in frames:
            fd = F.FRAMES.get(frame.type)
            # A DATAACK/DATANAK's own type carries the peer's grade of the frame it
            # answers — the whole of our transmit-rate feedback, and invisible while
            # all 32 codes logged under the one frame name.
            graded = peer_quality(frame.type)
            log.info("RX %s sess=%#04x ok=%s%s%s%s%s%s", fd.name if fd else hex(frame.type),
                     frame.session_id,
                     "UNVERIFIED" if frame.unverified else frame.ok,
                     " HEADER-ONLY" if frame.header_only else "",
                     "" if graded is None else f" theirq={graded}",
                     "" if frame.quality is None else f" q={frame.quality}",
                     # q=0 is a legal grade and also what is left when the body ran
                     # off the end of the capture (`DecodedFrame.truncated`), and
                     # five of the second kind were carried for days as the first.
                     " TRUNCATED" if frame.truncated else "",
                     f" {frame.caller}>{frame.target}" if frame.caller else "")
        with self._lock:
            if self._session is None or not self._codec_online:
                return
            # The timer pump is held off this lock for a whole transmission, so the
            # session's clock can be a frame old here, and a repeat scheduled from it
            # keys with no gap left to hear the answer in.
            if self._running:
                self._clock = self._session._now = time.monotonic() - self._t0
            for frame in frames:
                if self.identification_pending:
                    # Retain the previous session's lost-END recovery without
                    # accepting another call during its closing identification.
                    if frame.type != 0x29:
                        continue
                self._session.on_receive(frame.type, self._frame_payload(frame),
                                         frame.session_id, frame.ok, frame.quality,
                                         header_only=frame.header_only)

    def tick(self, now: float) -> None:
        self._clock = now
        with self._lock:
            if self._session is not None and self._codec_online:
                if self._closing_id is not None:
                    due, identity = self._closing_id
                    if now < due:
                        return
                    try:
                        self._send_identification(*identity)
                    finally:
                        self._closing_id = None
                    return
                self._session.tick(now)

    def _run(self) -> None:
        """Threaded pump: decode delivered audio and advance timers off the wall
        clock until stopped."""
        while self._running:
            try:
                samples = self._rx_q.get(timeout=0.1)
            except queue.Empty:
                samples = None
            if samples is not None:
                try:
                    self.receive_audio(samples)
                except Exception as exc:  # a malformed burst must not kill the pump
                    obs = self._observer
                    if obs is not None:
                        obs.modem_fault(f"receive error: {exc}")
            if not self._running:
                break
            try:
                self.tick(time.monotonic() - self._t0)
            except Exception as exc:
                # Failed timed/closing submissions retire the contact, but
                # must not silently kill the standalone core's only pump.
                self.abort()
                obs = self._observer
                if obs is not None:
                    obs.modem_fault(f'timing/transmit error: {exc}')

    @staticmethod
    def _frame_payload(frame: DecodedFrame) -> bytes:
        """Reconstruct the byte payload the ARQ session expects, since the demod
        surfaces some frames as parsed fields rather than raw bytes: the packed
        caller‖target for ConReq/Ping, the packed callsign‖grid for an ID frame,
        the timing triple for a ConAck. Everything else carries its bytes."""
        fd = F.FRAMES.get(frame.type)
        name = fd.name if fd else ""
        if (name.startswith("ConReq") or name == "Ping") and frame.caller and frame.target:
            return callsign.pack_callsign(frame.caller) + callsign.pack_callsign(frame.target)
        if name == "IDFrame" and frame.caller:
            # The ARQ session never consumes an ID's grid, so a missing/short one
            # is zero-filled rather than raising out of the pump (pack_grid is
            # strict about 2/4/6/8-char grids).
            grid = frame.grid or ""
            packed_grid = callsign.pack_grid(grid) if len(grid) in (2, 4, 6, 8) else b"\x00" * 6
            return callsign.pack_callsign(frame.caller) + packed_grid
        if name.startswith("ConAck"):
            t = min(255, (frame.conack_timing_ms or 0) // 10)
            return bytes([t, t, t])
        return frame.payload or b""
