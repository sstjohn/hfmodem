# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import threading
import time
from contextlib import nullcontext

from hfmodem.core import band
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.rig import (ArmReport, KeyToken, RigError,
                              _WATCHDOG_FRACTION, _WATCHDOG_SLACK_S)


class VirtualRig:
    def __init__(self, audio, *, centre_hz, transmit=True, max_key_s=30.0):
        self.audio = audio
        self._dial_hz = band.dial_hz(centre_hz)
        self.transmit = transmit
        self.max_key_s = max_key_s
        self.cat = self.ptt = None
        self.retired = self.must_retire = False
        self.retired_why = ""
        self.released_low = True
        self.line_state = "low"
        self._armed = False
        self._token = None
        self._seq = 0
        self._lock = threading.RLock()

    @property
    def keyed(self):
        return self._token is not None

    def dial(self):
        with self._lock:
            return self._dial_hz

    def tune(self, centre_hz):
        with self._lock:
            if self.keyed:
                raise RigError("refusing to retune while keyed")
            self._dial_hz = band.dial_hz(centre_hz)

    def arm(self, *, prove_ptt=True):
        with self._lock:
            if self.retired:
                raise RigError("virtual rig is retired")
            self._armed = True
        return ArmReport("replay", 0, False, False, ("no CAT or keying line",))

    def hold(self, deadline_index=None):
        return nullcontext(True)

    def key(self, emission, *, why, duration_s=None, settle_s=0.0,
            responding=False, sample_now=None, deadline_index=None,
            attended=False, open=False):
        if duration_s is None and not open:
            raise TypeError("finite key requires duration_s")
        with self._lock:
            if not self.transmit:
                raise RigError("transmit is disabled for this station")
            if self.retired or self.must_retire:
                raise RigError("virtual rig is retired")
            if not self._armed:
                raise RigError("arm() has not passed")
            if self.keyed:
                raise RigError("already keyed")
            index = int(self.audio.sample_now())
            duration = None if open else min(
                self.max_key_s, settle_s + duration_s + max(
                    _WATCHDOG_SLACK_S, _WATCHDOG_FRACTION * duration_s))
            end = None if open else index + int(duration * CARD_RATE_HZ)
            self._seq += 1
            self._token = KeyToken(self._seq, time.monotonic(), duration, why, index, end)
            self.line_state = "high"
            self.released_low = False
            return self._token

    def unkey(self, token=None):
        with self._lock:
            if token is not None and token.seq != self._seq:
                return
            self._token = None
            self.line_state = "low"
            self.released_low = True

    def panic_unkey(self, why):
        with self._lock:
            self.unkey()
            self.retired = self.must_retire = True
            self.retired_why = why

    def panic(self):
        self.panic_unkey("panic")

    def close(self):
        with self._lock:
            self.unkey()
            self._armed = False
        return self.released_low
