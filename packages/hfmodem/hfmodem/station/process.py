# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One radio shared by live receiving lanes, then their host ports.

Lanes must precede hosts: a client must never transmit into a channel whose
receiver has not started. Release fences every host and lane before devices.
"""
from __future__ import annotations

import signal
from contextlib import nullcontext
import hashlib
import os
import logging
import sys
import threading
import time
from pathlib import Path

from hfhost.config import ConfigError

from hfmodem.core import claim, devices, regulatory, wav
from hfmodem.core.cat import RigctldCat, YaesuCat
from hfmodem.core.audio import ReplayAudio, StationAudio
from hfmodem.core.config import ChainAudio, Audio, StationConfig
from hfmodem.core.ptt import RtsPtt
from hfmodem.core.operations import LeaseTimeout
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.rigs import CAT_BACKENDS
from hfmodem.core.rig import FATAL_SIGNALS, Rig, RigError, arm_deadman, fatal_write
from hfmodem.core.virtualrig import VirtualRig
from hfmodem.station.arbiter import Identity, TxArbiter
from hfmodem.station.qsolog import SessionRecords
from hfmodem.station.sessions import Registry

#: An `[audio] input` of `replay:FILE` is not a device name: the station reads
#: FILE and opens no card. The recording is named in the station file rather than
#: on the command line because which audio a station hears is a fact about that
#: station, like its callsign and its dial.
REPLAY = "replay"


def recording(audio: Audio) -> Path | None:
    """The file an `[audio] input` of `replay:FILE` names, or None for a card."""
    kind, _, name = audio.input.partition(":")
    if kind != REPLAY:
        return None
    if not name:
        raise ConfigError(
            '[audio]: input = "replay" does not say what to replay. Write '
            "replay:FILE — any WAV, at any rate. This tree ships "
            "packages/hfmodem/hfmodem/tests/besra/fixtures/"
            "offair_ke8lva_greeting.wav.")
    path = wav.stored(Path(name).expanduser())
    if not path.is_file():
        raise ConfigError(
            f"[audio]: there is no recording at {path}. The path is relative to "
            "the directory the station is started from; supply a WAV there, or "
            "write an absolute path.")
    return path


def cat_backend(cat: str) -> str:
    """How the `[rig] cat` driver reaches the radio: serial, rigctld or none."""
    if cat not in CAT_BACKENDS:
        raise ConfigError(f"[rig]: cat = {cat!r} is not a CAT driver; "
                          f"choose one of {', '.join(CAT_BACKENDS)}")
    return CAT_BACKENDS[cat]


def cat_driver(cfg: StationConfig):
    """The station's CAT driver, unopened; None when `[rig] cat = "none"`."""
    match cat_backend(cfg.rig.cat):
        case "rigctld":
            return RigctldCat(cfg.rig.host, cfg.rig.port,
                              profile=cfg.resolved_profile, max_watts=cfg.rig.max_watts)
        case "serial":
            return YaesuCat(cfg.resolved_profile.get("cat_port"), cfg.resolved_profile)
    return None


class Station:
    """Everything, wired together, in the order above."""

    def __init__(self, cfg: StationConfig, *, replay=None, station_file=None, session_records=None) -> None:
        self.cfg = cfg
        self.station_file = str(Path(station_file).resolve()) if station_file else None
        self.service = None
        self.state = "observing"
        self.is_ready = False
        self.failed = self._unavailable()
        self.reason = ""
        self.receive = not getattr(getattr(cfg, "station", None), "transmit", False)
        self.claim = None
        self._line = self._driver = None
        self._lifecycle = threading.Lock()
        self._poll_stop = threading.Event()
        self._poll_thread = None
        self._last_inode_check = time.monotonic()
        from hfmodem.core import identification
        st = getattr(cfg, "station", None)
        # Freeze before audio lookup, PTT open or Rig.arm, which may itself key.
        self.cw_identity = (identification.from_station(st)
                            if st is not None and st.transmit and st.id_mode == "cw"
                            else None)
        self.replay = replay
        self.replay_path: Path | None = None
        self.audio = None
        self.rig: Rig | None = None
        self.arbiter: TxArbiter | None = None
        self.lanes: dict[str, object] = {}
        self.hosts: dict = {}
        self._stop = threading.Event()
        self._fatal_reason = None
        self._fatal_entered = False
        self._signals = 0
        self._signal_seen = 0
        self._signal_number = None
        self._panic_thread = None
        self._owning = False
        self._own_cancel = threading.Event()
        root = Path(os.environ.get("HFMODEM_DATA_DIR", "operator-data")).expanduser().resolve()
        key = hashlib.sha256((self.station_file or "default").encode()).hexdigest()
        self.sessions = Registry(
            records=session_records if session_records is not None else
                    SessionRecords(root / "station" / key / "sessions"),
            authority=self._session_authority,
            available=lambda: self.state in ("owning", "owning-receive")
                              and not self.transitioning and not self._stop.is_set())

    def _session_authority(self, id):
        return self.arbiter.authority(id) if self.arbiter is not None else (id, 0)

    # -- bring-up ----------------------------------------------------------

    def validate_options(self):
        from hfmodem.station.link import LINKS

        for name in self.enabled:
            LINKS[name].validate_options(self.cfg.protocols[name].options)

    def open_audio(self):
        self.validate_options()
        cfg = self.cfg
        may_transmit = cfg.station.transmit and not self.receive
        if cfg.selected_rig != "ft891" and self.replay is None:
            cfg.require_audio("station audio", input=True, output=may_transmit,
                              drive=may_transmit)
        if self.replay is None:
            self.replay_path = recording(cfg.audio)
            if self.replay_path is not None:
                self.replay = wav.read(self.replay_path, CARD_RATE_HZ)
        if self.replay is not None:
            self.audio = ReplayAudio(self.replay)
        else:
            inp = devices.find_device(cfg.audio.input, "in", required=True)
            out = (devices.find_device(cfg.audio.output, "out", required=True)
                   if may_transmit else None)
            self.audio = StationAudio(
                input_device=inp, output_device=out,
                gain=cfg.audio.input_gain, blocksize=cfg.audio.blocksize,
                tx_latency_n=round((cfg.audio.tx_latency_ms if cfg.audio.tx_latency_ms is not None
                                    else cfg.resolved_profile.get("tx_latency_ms", 0))
                                   * CARD_RATE_HZ / 1000))
            self.audio.close_budget_s = cfg.rig.max_key_s + cfg.rig.ptt.settle_s + 5
        self.audio.open()
        return self.audio

    def open_rig(self, *, prove_ptt: bool = False):
        """Open CAT and, for transmit ownership, the line before lanes and hosts."""
        cfg = self.cfg
        if self.replay is not None:
            if self.audio is None:
                self.open_audio()
            self.rig = VirtualRig(self.audio, transmit=not self.receive,
                                  max_key_s=cfg.rig.max_key_s, centre_hz=cfg.rig.centre_hz)
            self.rig.arm(prove_ptt=False)
            return self.rig
        self._driver = cat_driver(cfg)
        if self._driver is None:
            return None
        try:
            if not self.receive:
                if not cfg.rig.ptt.port:
                    raise RigError("[rig.ptt] port is unset and transmit = true")
                self._line = RtsPtt(cfg.rig.ptt.port)
                self._line.open()
            self.rig = Rig(model=cfg.rig.model, cat=self._driver, ptt=self._line,
                           profile=cfg.profile, control=cfg.control,
                           mycall=cfg.station.mycall, transmit=not self.receive,
                           max_key_s=cfg.rig.max_key_s, identity=cfg.rig.identity,
                           dial_max_age_s=cfg.rig.dial_max_age_s)
            self.rig.fatal = self.fatal_shutdown
            self.rig.admission_cancel = self._own_cancel
            self._driver.open()
            self._check_own_cancelled()
            if not self.receive:
                print(self.rig.arm(prove_ptt=prove_ptt))
        except BaseException:
            if self._line is not None:
                self._line.release()
            self._driver.close()
            raise
        return self.rig

    def build_arbiter(self):
        if self.receive or self.rig is None:
            return
        cfg = self.cfg
        ident = Identity(self.cw_identity.cw_call if self.cw_identity is not None
                         else cfg.station.mycall, interval_s=cfg.station.id_interval_s,
                         mode=cfg.station.id_mode)
        self.arbiter = TxArbiter(self.audio, self.rig, identity=ident,
                                 drive=(cfg.audio.drive_for() if isinstance(cfg.audio, ChainAudio)
                                        else cfg.audio.tx_drive), settle_s=cfg.rig.ptt.settle_s)

    def preflight_reference(self):
        from hfmodem.station.preflight import Preflight
        self._preflight = Preflight(self.cfg)
        self._preflight.cancelled = self._own_cancel.is_set
        self._preflight.cancel_event = self._own_cancel
        reference = self._preflight._reference()
        if reference is None:
            raise RigError("capture-only preflight failed")
        return reference

    def preflight_measure(self, reference):
        from hfmodem.core.audio import StreamLane
        from hfmodem.station.preflight import DWELL_S
        check = self._preflight
        check.tap = self.audio.subscribe(StreamLane(CARD_RATE_HZ))
        try:
            if self._stop.wait(DWELL_S):
                raise RigError("shutdown requested")
            check._duplex(check._read()[0], reference)
            check._xruns(self.audio)
            check._offset(self.audio)
            check._keying(self.audio, self.rig)
            self._check_own_cancelled()
            check._clock(self.audio)
            if check.verdict():
                if getattr(self.rig, "line_state", None) == "unconfirmed":
                    raise RigError("duplex preflight failed: the keying line is unconfirmed")
                raise RigError("duplex preflight failed")
        finally:
            self.audio._lanes.remove(check.tap)

    def build_lanes(self) -> dict:
        """One lane per enabled protocol, each subscribed to the card.

        Built after audio and before the host servers: a client that connects to a
        modem whose receiver is not running gets a station transmitting into a
        channel it cannot hear.
        """
        from hfmodem.station.link import LINKS
        for name in self.enabled:
            cls = LINKS.get(name)
            if cls is None:
                continue
            link = cls(self)
            self.lanes[name] = link
            self.audio.subscribe(link.lane)
            # Session owners begin before external hosts can admit commands.
            if hasattr(link, "start"):
                link.start()
        if self.arbiter is not None:
            self.arbiter.active_protocols = lambda: tuple(
                name for name, lane in self.lanes.items() if getattr(lane, "active", False))
        return self.lanes

    def _unavailable(self):
        return [{"name": name, "why": "served from cutover " +
                 {"ardop": "6.4", "vara": "6.5", "sabir": "6.6"}.get(name, "6.7")}
                for name, protocol in self.cfg.protocols.items()
                if protocol.enabled and name not in self.enabled]

    @property
    def enabled(self) -> list[str]:
        return [n for n, p in self.cfg.protocols.items()
                if p.enabled and n in {"bateleur", "ale", "selcall", "rpr", "ardop", "vara", "pactor"}]

    def describe(self) -> str:
        cfg = self.cfg
        lines = [
            f"station   {cfg.station.mycall or '(no callsign)'} "
            f"[{cfg.control.value} control, {cfg.profile.name}]",
            f"transmit  {'enabled' if cfg.station.transmit else 'DISABLED'}",
            f"protocols {', '.join(self.enabled) or '(none)'}",
        ]
        if self.cw_identity is not None:
            lines.append(f"CW ID     {self.cw_identity.cw_call} ({self.cw_identity.source})")
        if self.replay is not None:
            lines.append(f"audio     replay {self.replay_path or '(samples)'}, "
                         f"{len(self.replay) / CARD_RATE_HZ:.1f} s")
        elif self.audio is not None:
            lines.append(f"audio     card, offset {self.audio.lat} samples")
        if cfg.listens and cfg.control is regulatory.Control.AUTOMATIC:
            lines.append("NOTE      this station answers unattended; the selected emission "
                         "profile uses automatic control")
        return "\n".join(lines)

    # -- run ---------------------------------------------------------------

    @property
    def transitioning(self):
        return self._lifecycle.locked()

    def snapshot(self):
        rig = self.rig
        return {
            **self.sessions.snapshot(),
            "service": {"state": self.state, "ready": self.is_ready,
                        "failed": list(self.failed), "reason": self.reason,
                        "claim_holder": self._claim_holder()},
            "station": {"mycall": self.cfg.station.mycall,
                        "transmit": self.cfg.station.transmit,
                        "station_file": self.station_file},
            "rig": {"source": "poisoned" if rig and rig.retired else
                               "virtual" if isinstance(rig, VirtualRig) else
                               "cat" if rig is not None else "none",
                    "retired": bool(rig and rig.retired),
                    "retired_why": rig.retired_why if rig else "",
                    "released_low": getattr(rig, "released_low", None),
                    "panic_confirmed": getattr(rig, "_panic_confirmed", None)},
        }

    def _claim_holder(self):
        try:
            return claim.claim_path().read_text()[:16384].strip() or None
        except OSError:
            return None

    def _state(self, state, reason=""):
        from hfmodem.station.wire import SERVICE_STATES
        if state not in SERVICE_STATES:
            raise ValueError(state)
        with self.sessions.lock, (self.service.events.condition
                                 if self.service is not None else nullcontext()):
            self.state, self.reason = state, reason
            if state not in ("owning", "owning-receive"):
                self.is_ready = False
            if self.service is not None:
                self.sessions._changed()

    def own(self, *, receive=False):
        if type(receive) is not bool:
            raise ValueError("receive must be boolean")
        if self._stop.is_set():
            raise RuntimeError("shutting-down")
        # Cleared before the lifecycle is taken, so a STOP after it is kept.
        # Shutdown sets _stop before _own_cancel, so a clear that undid its
        # cancel is seen once the lifecycle is held.
        self._own_cancel.clear()
        if not self._lifecycle.acquire(blocking=False):
            raise RuntimeError("transitioning")
        try:
            if self._stop.is_set():
                raise RuntimeError("shutting-down")
            receive = receive or not self.cfg.station.transmit
            if self.state in ("owning", "owning-receive"):
                if receive != self.receive:
                    raise RigError("release before changing ownership mode")
                return {"state": self.state}
            if self.state != "observing":
                raise RigError(f"cannot own while {self.state}")
            self.receive = receive
            self.is_ready = False
            self._owning = True
            self.failed = self._unavailable()
            deadline = threading.Timer(40, self.fatal_shutdown,
                                       args=("own exceeded its 40 s admission budget",))
            deadline.daemon = True
            deadline.start()
            try:
                if self.replay is None:
                    self.replay_path = recording(self.cfg.audio)
                    if self.replay_path is not None:
                        self.replay = wav.read(self.replay_path, CARD_RATE_HZ)
                if self.replay is None:
                    cfg = self.cfg
                    self.claim = claim.acquire("station", "receive" if receive else "own",
                        devices=claim.Devices(
                            cat_port=cfg.resolved_profile.get("cat_port"),
                            ptt_port=cfg.rig.ptt.port if not receive else None,
                            capture=cfg.audio.input,
                            playback=cfg.audio.output if not receive else None,
                            cat=cat_backend(cfg.rig.cat)))
                reference = self.preflight_reference() if not receive and self.replay is None else None
                self._check_own_cancelled()
                self.open_audio()
                self._check_own_cancelled()
                self.open_rig(prove_ptt=not receive)
                self._check_own_cancelled()
                if not receive and self.replay is None:
                    self.preflight_measure(reference)
                self._check_own_cancelled()
                self.build_arbiter()
                if self._stop.is_set():
                    raise RigError("shutdown requested")
                self.build_lanes()
                self._check_own_cancelled()
                from hfmodem.station import hosts
                self.hosts = hosts.build(self.cfg, station=self, failed=self.failed,
                                         retain=self.hosts)
                self._check_own_cancelled()
                self._poll_stop.clear()
                self._poll_thread = threading.Thread(target=self._poll, name="station-lanes", daemon=True)
                self._last_inode_check = time.monotonic()
                self._state("owning-receive" if receive else "owning")
                self._poll_thread.start()
                self.is_ready = not self.failed and self.state != "failed"
                self.sessions.admit()
                return {"state": self.state, "ready": self.is_ready}
            except BaseException as exc:
                deadline.cancel()
                deadline.join()
                self._teardown(reason=str(exc))
                raise
            finally:
                self._owning = False
                self._own_cancel.clear()
                deadline.cancel()
                deadline.join()
        finally:
            self._own_cancel.clear()
            self._lifecycle.release()

    def _check_own_cancelled(self):
        if self._own_cancel.is_set() or self._stop.is_set():
            raise RigError("ownership cancelled by STOP or shutdown")

    def stop_transmit(self):
        arbiter = self.arbiter
        if self._owning or self.transitioning:
            self._own_cancel.set()
        if self._owning:
            # HTTP only records cancellation; the owning thread unkeys normally.
            return
        if arbiter is not None:
            arbiter.stop(None)

    def release(self, *, reason=""):
        if not self._lifecycle.acquire(blocking=False):
            raise RuntimeError("transitioning")
        try:
            return self._release(reason)
        finally:
            self._own_cancel.clear()
            self._lifecycle.release()

    def _release(self, reason):
        if (self.state == "observing" and not (self.audio or self.rig or self.lanes or self.hosts)
                and all(s['state'] == 'ended' for s in self.sessions.snapshot()['sessions'])):
            return {"state": self.state}
        warning_start = len(self.sessions.warnings)
        self._teardown(reason=reason)
        if self.state == "failed":
            raise RigError(self.reason)
        return {"state": self.state, "warnings": self.sessions.warnings[warning_start:]}

    def _teardown(self, *, reason=""):
        self.is_ready = False
        with self.sessions.lock:
            self.sessions.accepting = False
        self._state("releasing", reason)
        failures = []

        def close(name, fn, *, thread_report=True):
            try:
                result = fn()
                if thread_report and result is False:
                    raise LeaseTimeout(name)
            except BaseException as exc:
                failures.append(f"{name}: {exc}")

        # Rollback must not panic an already-cancelled ownership a second time.
        self._owning = False
        close("STOP", self.stop_transmit)
        self.sessions.suspend(persist=False)
        self._poll_stop.set()
        from hfmodem.station import hosts
        close("hosts", lambda: hosts.stop_all(self.hosts))
        if self._poll_thread is not None:
            self._poll_thread.join(self.cfg.rig.max_key_s + self.cfg.rig.ptt.settle_s + 5)
            if self._poll_thread.is_alive():
                failures.append("station-lanes")
        for name, lane in self.lanes.items():
            if hasattr(lane, "stop"):
                close(name, lane.stop)
        if self.arbiter is not None:
            close("arbiter", lambda: self.arbiter.close(
                timeout=self.cfg.rig.max_key_s + self.cfg.rig.ptt.settle_s + 5))
        line = self._line or getattr(self.rig, "ptt", None)
        if self.rig is not None:
            close("rig", self.rig.close, thread_report=False)
        else:
            if line is not None:
                close("line", line.release)
            if self._driver is not None:
                close("CAT", self._driver.close)
        if failures and self.rig is not None:
            if line is not None and getattr(line, "released_low", None) is not True:
                worker = threading.Thread(target=lambda: close("line", line.release),
                                          name="station-line-close", daemon=True)
                worker.start()
                worker.join(.25)
                if worker.is_alive():
                    failures.append("keying line still closing")
            driver = self._driver or getattr(self.rig, "cat", None)
            if driver is not None:
                close("CAT", driver.close)
        if self.audio is not None:
            close("audio", self.audio.close)
        self.sessions.persist_stopping()
        if failures:
            why = "; ".join(failures)
            self._state("failed", why)
            print(why, file=sys.stderr)
            self.fatal_shutdown(why)
            raise LeaseTimeout(why)
        if line is not None and getattr(line, "released_low", None) is not True:
            self.sessions.finish_all(outcome="unconfirmed", returncode=1,
                                     verdict={"line": "unconfirmed"}, hand_stopped=True)
            self._state("failed", "keying line unconfirmed at close")
            return
        self.sessions.finish_all(outcome="stopped", hand_stopped=True,
                                 verdict={"line": "down" if line is not None else None,
                                          "reason": reason})
        if self.claim is not None:
            claim.release(self.claim)
            self.claim = None
        self.audio = self.rig = self.arbiter = self._line = self._driver = None
        self.lanes, self.hosts = {}, {}
        self._poll_thread = None
        self._state("observing", reason)

    def check_claim(self, *, now=None):
        now = time.monotonic() if now is None else now
        if (self.claim is None or self.state == "failed" or self.transitioning
                or now - self._last_inode_check < 60):
            return
        self._last_inode_check = now
        try:
            claim.check_inode(self.claim)
        except claim.ClaimError as exc:
            try:
                self.release(reason=str(exc))
            except RuntimeError as race:
                if str(race) != "transitioning":
                    raise
                self._last_inode_check = now - 60
            except RigError:
                pass

    def _poll(self):
        try:
            self.audio.breathe()
            while not self._poll_stop.is_set():
                self._tick()
                if (self.rig is not None and self.rig.retired
                        and getattr(self.rig, "_panic_confirmed", None) is False
                        and self.state != "failed"):
                    self.is_ready = False
                    self._state("failed", self.rig.retired_why)
                if self.replay is not None and self.replay_exhausted():
                    self._stop.set()
                    return
        except BaseException as exc:
            self._state("failed", str(exc))
            self.stop_transmit()
            self._stop.set()

    def _main_work(self, fn):
        result, errors = [], []

        def work():
            try:
                result.append(fn())
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=work, name="station-lifecycle", daemon=True)
        thread.start()
        while True:
            self._handle_signals()
            if not thread.is_alive() and self._panic_thread is None:
                break
            if thread.is_alive():
                thread.join(.02)
            else:
                time.sleep(.02)
        if errors:
            raise errors[0]
        return result[0]

    def _start(self):
        return self.own()

    def run(self, *, on_ready=None, detach=False, preflight=False) -> int:
        self._install_signals()
        try:
            if not detach and (any(p.enabled for p in self.cfg.protocols.values()) or preflight):
                self._main_work(self._start)
            elif not detach and (self.replay is not None or recording(self.cfg.audio) is not None):
                print("no protocols enabled", file=sys.stderr)
                return 2
            if on_ready is not None:
                self._main_work(lambda: on_ready(self))
            if preflight:
                return 0
            while True:
                self._handle_signals()
                if self._panic_thread is not None:
                    time.sleep(.02)
                    continue
                if self._stop.wait(.02):
                    break
                if (self.claim is not None and self.state != "failed"
                        and not self.transitioning
                        and time.monotonic() - self._last_inode_check >= 60):
                    self._main_work(self.check_claim)
        finally:
            self._poll_stop.set()
            try:
                self._main_work(self.shutdown)
            finally:
                if self.service is not None:
                    try:
                        self._main_work(self.service.close)
                    except LeaseTimeout as exc:
                        self.fatal_shutdown(str(exc))
                        raise
        return 128 + self._signal_number if self._signal_number else 0

    def replay_exhausted(self) -> bool:
        if not getattr(self.audio, "exhausted", lambda: False)():
            return False
        # The audio finishes a burst before the arbiter accounts for it.
        if self.arbiter is None:
            return True
        with self.arbiter.try_hold() as idle:
            return idle

    def _identify_if_due(self) -> None:
        """Discharge the identification interlock.

        Without this the interlock has no producer: at `id_interval_s` it closes,
        every live request is refused, and no callsign is ever sent — so the
        station goes mute without sending its scheduled identification. The
        interlock is the right shape; it needs something to satisfy it.
        """
        arb = self.arbiter
        if arb is None or arb.identity is None or not arb.identity.due:
            return
        from hfmodem.core import cwid
        from hfmodem.core.regulatory import centred
        try:
            # The identification is an emission like any other, and on a band
            # with a power limit an undeclared one is refused — which would
            # mute the station rather than only the callsign.
            arb.identify(centred(self.rig.dial(), cwid.BANDWIDTH_HZ,
                                 centre_hz=cwid.TONE_HZ,
                                 power_w=self.cfg.declared_power_w))
        except Exception as exc:            # noqa: BLE001
            print(f"identification failed: {exc}", file=sys.stderr)

    def _tick(self) -> None:
        self._identify_if_due()
        if self.replay is not None:
            self.audio.pump(32)
        else:
            self._poll_stop.wait(0.05)
        for lane in self.lanes.values():
            poll = getattr(lane, "poll", None)
            if poll is not None:
                poll()

    def _signal(self, signum, frame):
        self._signal_number = signum
        self._signals += 1

    def _handle_signals(self):
        if self._signals == self._signal_seen:
            return
        first = not self._signal_seen
        self._signal_seen = self._signals
        if self._fatal_reason is not None and not self._fatal_entered:
            self.fatal_shutdown(self._fatal_reason)
        if self._panic_thread is not None:
            fatal_write(b"!! asked again during a panic: unkey unconfirmed\n")
            os._exit(1)
        if self.rig is not None:
            arm_deadman()
            self._panic_thread = threading.Thread(target=self.rig.panic,
                                                 name="station-panic", daemon=True)
            self._panic_thread.start()
        else:
            self._stop.set()
            self._own_cancel.set()
            if first:
                arm_deadman(5.0)

    def _install_signals(self):
        for sig in FATAL_SIGNALS:
            signal.signal(sig, self._signal)

    def fatal_shutdown(self, why):
        """Bounded process termination; unkey confirmation remains an observation.

        The bound assumes the interpreter can still execute its timer handler.
        A worker hands off through the station's signal path without a new thread.
        """
        self._fatal_reason = why
        if threading.current_thread() is not threading.main_thread():
            signal.setitimer(signal.ITIMER_REAL, 5.0)
            os.kill(os.getpid(), signal.SIGTERM)
            while True:
                time.sleep(0.1)
        if self._fatal_entered:
            try:
                fatal_write(b"fatal_shutdown confirmed=false reason=reentered fatal shutdown\n")
            finally:
                os._exit(1)
        self._fatal_entered = True
        arm_deadman(5.0, message=b"fatal_shutdown confirmed=false reason=deadman budget exhausted\n")
        self._panic_thread = threading.Thread(target=self._finish_fatal, args=(why,),
                                             name="station-fatal", daemon=True)
        self._panic_thread.start()
        while True:
            self._handle_signals()
            time.sleep(.02)

    def _finish_fatal(self, why):
        confirmed = False
        try:
            if self.rig is not None:
                self.rig._fatal_handoff = True
                self.rig.panic_unkey(why)
                confirmed = getattr(self.rig, "_panic_confirmed", None) is True
        except BaseException:
            confirmed = False
        message = f"fatal_shutdown confirmed={str(confirmed).lower()} reason={why}"
        try:
            fatal_write((message + "\n").encode())
            logger = logging.getLogger(__name__)
            record = logger.makeRecord(logger.name, logging.CRITICAL, __file__, 0,
                                       message, (), None)
            current = logger
            while current is not None:
                for handler in current.handlers:
                    if handler.lock.acquire(timeout=0.02):
                        try:
                            handler.emit(record)
                        finally:
                            handler.lock.release()
                current = current.parent if current.propagate else None
            for link in self.lanes.values():
                if hasattr(link, "flush_events"):
                    link.flush_events(timeout=0.02)
        except BaseException:
            pass
        finally:
            os._exit(1)

    def shutdown(self):
        self._stop.set()
        self._own_cancel.set()
        # An HTTP own may still be before open_rig. Let its cancellation unwind
        # under the signal deadman, and keep the lifecycle through release so a
        # queued own cannot start in between.
        with self._lifecycle:
            try:
                self._release("")
            except Exception as exc:
                self.fatal_shutdown(f"shutdown release failed: {exc}")
