# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ALE modem on a station: a receiver, a controller, a thread.

Like ``bateleur.modem`` it knows nothing of the station. It is handed card-rate
audio, a ``send(audio, rate)`` that puts a burst on the air and returns the
card sample index it ended at (or a refusal reason), and a ``tune(channel)``
that moves the radio and returns the card index from which audio is from the
new channel (or None if it could not). What it reports goes to an observer.

**One clock.** Every time the controller sees is the card's sample count:
the audio it is handed is stamped with it, and so is the end of every burst
the station sends. A burst blocks this thread while it plays; the audio that
arrived meanwhile is then read in order, and the controller's clock walks
through the burst as it would have in real time.

**Half duplex.** Audio from inside a burst of its own is not listened to.

**Scanning** retunes the radio. Audio from before a retune completed belongs to
the old channel and is dropped, and the receiver starts afresh: two channels'
audio stitched together would be decoded as one.
"""
from __future__ import annotations

import time

from ..worker import ModemWorker

import numpy as np
from hfmodem.core.resample import FirDecimator as _Decimate

from . import fsk
from .controller import Config, Controller, Event

#: Before a call or a sound, at least this much of what was heard is kept
#: to judge occupancy by (Twt, annex B); more if the listen period is longer.
_KEEP_S = 2.0


class Observer:
    """What the modem reports, on its own thread."""

    def event(self, ev: Event) -> None: ...


class AleModem(ModemWorker):
    worker_name = "ale"

    def __init__(self, config: Config, *, send, tune=None, busy=None, card_rate: int = 48000,
                 observer: Observer | None = None, log=None, hold=lambda: False) -> None:
        self._config = config
        self._hold = hold
        self.card_rate = card_rate
        self._send_audio = send
        self._tune_radio = tune
        self._busy_audio = busy
        self.observer = observer or Observer()
        #: Who is on the air (``directory.Directory``), if the station read one.
        self.directory = None
        self._log = log or (lambda *_: None)
        self.rx = fsk.DualReceiver(max_dtm_words=config.dtm_words, max_dbm_depth=config.dbm_depth)
        self._dec = _Decimate(card_rate)
        self._expected: list[int] = []
        self._valid_from = 0                     # card index: audio before is another channel's
        self._quiet_until = 0                    # card index: our own burst ends here
        self._now_index = 0
        self._traffic_at = 0
        self._recent: list[np.ndarray] = []      # card-rate audio for the busy judgment
        self._init_worker()
        #: UTC at card index 0, fixed by the first audio: the controller's
        #: clock is the card's, and linking protection needs the time of day.
        self._epoch: float | None = None
        self.ctl = Controller(config, tune=self._tune, send=self._send, busy=self._busy,
                              expect=self._expect, utc=self._utc)
        self.rx.set_lp(self.ctl.lp_view())
        self._drain()

    # -- from any thread ------------------------------------------------------

    def feed(self, at: int, audio: np.ndarray) -> None:
        """Card-rate samples starting at card index ``at``, every one once."""
        self._feed((at, np.asarray(audio, dtype=np.float32)))

    # -- on the modem's thread ---------------------------------------------

    @property
    def now(self) -> float:
        return self._now_index / self.card_rate

    def _tune(self, channel: int) -> bool:
        if self._hold():
            return False
        if self._tune_radio is None:
            return True
        at = self._tune_radio(channel)
        if at is None:
            return False
        self._valid_from = max(self._valid_from, int(at))
        self.reset_receive()
        return True

    def reset_receive(self):
        cfg = self._config
        self.rx = fsk.DualReceiver(max_dtm_words=cfg.dtm_words if cfg else 60,
                                   max_dbm_depth=cfg.dbm_depth if cfg else 196)
        self.rx.expect(self._expected)
        if getattr(self, "ctl", None) is not None:
            self.rx.set_lp(self.ctl.lp_view())
        self._dec = _Decimate(self.card_rate)
        self._recent = []

    def _send(self, words: list[int]) -> float | str:
        audio = fsk.modulate(words)
        end = self._send_audio(audio, fsk.FS)
        if end is None or isinstance(end, str):
            return end or "transmission refused"
        self._quiet_until = max(self._quiet_until, int(end))
        return audio.size / fsk.FS

    def _busy(self) -> bool:
        if self._hold() or self.rx.locked:
            return True
        if self._busy_audio is None or not self._recent:
            return False
        return bool(self._busy_audio(np.concatenate(self._recent)))

    def _expect(self, ws: list[int]) -> None:
        self._expected = list(ws)
        self.rx.expect(self._expected)

    def _utc(self, now: float) -> float:
        if self._epoch is None:
            return time.time()
        return self._epoch + now

    def _audio(self, at: int, x: np.ndarray) -> None:
        end = at + x.size
        if self._epoch is None:
            self._epoch = time.time() - end / self.card_rate
        self._now_index = max(self._now_index, end)
        if end <= self._valid_from:
            return
        if at < self._valid_from:
            x = x[self._valid_from - at:]
            at = self._valid_from
        if at < self._quiet_until:
            # Our own burst: heard as silence, so the receiver's clock runs on.
            x = x.copy()
            x[:min(x.size, self._quiet_until - at)] = 0.0
        self._recent.append(x)
        keep = int(max(_KEEP_S, self.ctl.cfg.listen_s) * self.card_rate)
        while sum(r.size for r in self._recent) > keep + self._recent[0].size:
            self._recent.pop(0)
        for ev in self.rx.push(self._dec.push(x)):
            self.ctl.heard(ev, self.now, age=(self.rx.samples - ev.end) / fsk.FS)
        if self.rx.signal:
            self.ctl.signal(self.now)
        self._traffic(end)
        if not self._hold():
            self.ctl.tick(self.now)

    def _traffic(self, end: int) -> None:
        """While linked, anything occupying the channel is the link's traffic
        — another protocol's, most likely — and holds off Twa. Judged once a
        second on audio heard since our own last burst (which reads as
        silence, and a silent input reads as occupied)."""
        if self.ctl.state != self.ctl.LINKED or self._busy_audio is None \
                or end < self._traffic_at:
            return
        self._traffic_at = end + self.card_rate
        heard = min(sum(r.size for r in self._recent), end - self._quiet_until)
        if heard < int(_KEEP_S * self.card_rate):
            return
        x = np.concatenate(self._recent)[-heard:]
        if self._busy_audio(x):
            self.ctl.activity(self.now)

    def activity(self, card_index: int) -> None:
        """Another protocol transmitted on the link, ending at ``card_index``
        (from any thread)."""
        self.post(lambda: self.ctl.activity(card_index / self.card_rate))

    def _drain(self) -> None:
        evs, self.ctl.events = self.ctl.events, []
        for ev in evs:
            try:
                self.observer.event(ev)
            except Exception as exc:                # noqa: BLE001
                self._log(f"ale: observer: {exc}")

    def _after_work(self):
        self._drain()
