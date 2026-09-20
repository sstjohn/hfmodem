# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The station: one process, one radio, whichever protocols are enabled.

Order matters and is not cosmetic:

    1  config          fail closed, before anything is opened
    2  devices         required=True if any protocol may transmit
    3  audio           wait for a block and a settled ADC/DAC offset, or die loudly
    4  rig.arm()       a proven PTT keys the radio (no modulation, so no RF), and
                       the muting receiver is what confirms the line arrived
    5  lanes           the receiver is live BEFORE any client can connect
    6  host servers    each protocol's own dialect on its own port

Step 5 before step 6 is the one worth defending: a client that connects to a modem
whose receiver is not running gets a station that transmits into a channel it
cannot hear. Eight kestrel on-air sessions were reported as a silent band and
the cause was the neighbouring one -- two processes on one transceiver, shrike
transmitting while kestrel recorded its own station's mute -- so this ordering
closes a second door onto the same room rather than that one.

A protocol that wedges is contained rather than fatal — its lane's deque bounds
itself, its host port begins refusing, and the other three keep running. That is
the one property four separate processes gave away for free, and it has to be
built here.
"""
from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path

from hfhost.config import ConfigError

from hfmodem.core import devices, regulatory, wav
from hfmodem.core.audio import ReplayAudio, StationAudio
from hfmodem.core.config import Audio, StationConfig
from hfmodem.core.ptt import RtsPtt
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.rig import FATAL_SIGNALS, Cat, Rig, RigError
from hfmodem.station.arbiter import Identity, TxArbiter

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
    path = Path(name).expanduser()
    if not path.is_file():
        raise ConfigError(
            f"[audio]: there is no recording at {path}. The path is relative to "
            "the directory the station is started from; supply a WAV there, or "
            "write an absolute path.")
    return path


class Station:
    """Everything, wired together, in the order above."""

    def __init__(self, cfg: StationConfig, *, replay=None) -> None:
        self.cfg = cfg
        self.replay = replay
        self.replay_path: Path | None = None
        self.audio = None
        self.rig: Rig | None = None
        self.arbiter: TxArbiter | None = None
        self.lanes: dict[str, object] = {}
        self.hosts: dict = {}
        self._stop = threading.Event()

    # -- bring-up ----------------------------------------------------------

    def open_audio(self):
        cfg = self.cfg
        if self.replay is None:
            self.replay_path = recording(cfg.audio)
            if self.replay_path is not None:
                self.replay = wav.read(self.replay_path, CARD_RATE_HZ)
        if self.replay is not None:
            self.audio = ReplayAudio(self.replay)
        else:
            may_transmit = cfg.station.transmit and bool(self.enabled)
            inp = devices.find_device(cfg.audio.input, "in", required=may_transmit)
            out = devices.find_device(cfg.audio.output, "out", required=may_transmit)
            self.audio = StationAudio(
                input_device=inp, output_device=out,
                gain=cfg.audio.input_gain, blocksize=cfg.audio.blocksize,
                tx_latency_n=round(cfg.audio.tx_latency_ms
                                   * CARD_RATE_HZ / 1000))
        self.audio.open()
        return self.audio

    def open_rig(self, *, prove_ptt: bool = False):
        """The radio. Skipped entirely when the station cannot transmit.

        A receive-only station has no business opening a CAT port or holding a
        keying line, and saying so here is cheaper than a transmit interlock that
        has to be right in five places.
        """
        cfg = self.cfg
        if not cfg.station.transmit:
            return None
        if not cfg.rig.ptt.port:
            raise RigError("[rig.ptt] port is unset and transmit = true")
        line = RtsPtt(cfg.rig.ptt.port)
        line.open()
        self.rig = Rig(model=cfg.rig.model, cat=Cat(cfg.rig.host, cfg.rig.port),
                       ptt=line, profile=cfg.profile, control=cfg.control,
                       mycall=cfg.station.mycall, transmit=True,
                       max_key_s=cfg.rig.max_key_s)
        report = self.rig.arm(prove_ptt=prove_ptt)
        print(report)
        ident = Identity(cfg.station.mycall, interval_s=cfg.station.id_interval_s,
                         mode=cfg.station.id_mode)
        self.arbiter = TxArbiter(self.audio, self.rig, identity=ident,
                                 drive=self.cfg.audio.tx_drive,
                                 settle_s=self.cfg.rig.ptt.settle_s)
        return self.rig

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
            try:
                link = cls(self)
            except TypeError:
                # besra's adapter needs its modem; a station without one runs the
                # others rather than refusing to start.
                print(f"{name}: no modem instance, lane not built", file=sys.stderr)
                continue
            self.audio.subscribe(link.lane)
            # sabir runs its own session on this station rather than handing frames
            # to something else, so its adapter owns a thread. The others do not.
            if hasattr(link, "start"):
                link.start()
            self.lanes[name] = link
        return self.lanes

    @property
    def enabled(self) -> list[str]:
        return [n for n, p in self.cfg.protocols.items() if p.enabled]

    def describe(self) -> str:
        cfg = self.cfg
        lines = [
            f"station   {cfg.station.mycall or '(no callsign)'} "
            f"[{cfg.control.value} control, {cfg.profile.name}]",
            f"transmit  {'enabled' if cfg.station.transmit else 'DISABLED'}",
            f"protocols {', '.join(self.enabled) or '(none)'}",
        ]
        if self.replay is not None:
            lines.append(f"audio     replay {self.replay_path or '(samples)'}, "
                         f"{len(self.replay) / CARD_RATE_HZ:.1f} s")
        elif self.audio is not None:
            lines.append(f"audio     card, offset {self.audio.lat} samples")
        if cfg.listens and cfg.control is regulatory.Control.AUTOMATIC:
            lines.append("NOTE      this station answers unattended, so §97.221 "
                         "confines where it may transmit")
        return "\n".join(lines)

    # -- run ---------------------------------------------------------------

    def run(self) -> int:
        if not self.enabled:
            print("no protocols enabled — nothing to run.", file=sys.stderr)
            return 2

        # Before the card and before the radio: `open_rig` arms, arming can key,
        # and a host port that is bound is a port a client can drive to key
        # through. Neither may happen while SIGTERM is still SIG_DFL.
        self._install_signals()
        self.open_audio()
        try:
            self.open_rig()
        except RigError as exc:
            print(f"the radio is not usable: {exc}", file=sys.stderr)
            self.audio.close()
            return 1

        try:
            self.build_lanes()
            print(self.describe())
            print(f"receiver live on {len(self.lanes)} lane(s): "
                  f"{', '.join(self.lanes) or '(none)'}")
            # Last, and only now: a client that connects before the receiver is
            # running gets a station transmitting into a channel it cannot hear.
            from hfmodem.station import hosts
            self.hosts = hosts.build(self.cfg, station=self)

            breathe = getattr(self.audio, "breathe", None)
            if breathe is not None:
                breathe()

            while not self._stop.is_set():
                if self.replay is not None and self.replay_exhausted():
                    print("replay ended after "
                          f"{self.audio.samples / CARD_RATE_HZ:.1f} s")
                    break
                self._tick()
        finally:
            self.shutdown()
        return 0

    def replay_exhausted(self) -> bool:
        return getattr(self.audio, "exhausted", lambda: False)()

    def _identify_if_due(self) -> None:
        """Discharge the identification interlock.

        Without this the interlock has no producer: at `id_interval_s` it closes,
        every live request is refused, and no callsign is ever sent — so the
        station goes mute *and* §97.119 is still unsatisfied. The interlock is the
        right shape; it needs something to satisfy it.
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
                                 power_w=self.cfg.rig.power_w))
        except Exception as exc:            # noqa: BLE001
            print(f"identification failed: {exc}", file=sys.stderr)

    def _tick(self) -> None:
        self._identify_if_due()
        if self.replay is not None:
            if not self.audio.pump(32):
                self._stop.set()
        else:
            time.sleep(0.05)
        for lane in self.lanes.values():
            poll = getattr(lane, "poll", None)
            if poll is not None:
                poll()

    def _install_signals(self) -> None:
        """The rig owns the signal path.

        An operator who reached for Ctrl-C is entitled to a dead transmitter and a
        dead process in about a second, whatever rigctld is doing — so the handler
        goes to `Rig.panic()`, which arms a deadman before it tries to unkey. SIGHUP
        belongs here with the other two: a station is run over ssh and from a
        terminal, and a dropped one used to skip both this and `run()`'s `finally`.
        """
        def handler(signum, frame):
            if self.rig is not None:
                self.rig.panic()        # does not return
            # Nothing is keyed, but a stop that only sets a flag is not a stop
            # while the bring-up is still inside a device open.
            self._stop.set()
            raise SystemExit(128 + signum)

        for sig in FATAL_SIGNALS:
            signal.signal(sig, handler)

    def shutdown(self) -> None:
        try:
            if getattr(self, "hosts", None):
                from hfmodem.station import hosts
                hosts.stop_all(self.hosts)
        finally:
            # Before the radio: an adapter with a thread of its own may still be
            # holding a transmission, and the arbiter is what ends one cleanly.
            # A dialect that will not come down is reported after the radio has.
            stubborn = []
            for name, link in self.lanes.items():
                if hasattr(link, "stop"):
                    try:
                        link.stop()
                    except Exception as exc:            # noqa: BLE001
                        stubborn.append((name, exc))
            try:
                if self.rig is not None:
                    self.rig.close()
            finally:
                if self.audio is not None:
                    self.audio.close()
            for name, exc in stubborn:
                print(f"{name}: the lane would not come down ({exc})", file=sys.stderr)
