# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The selcall modem on a station: a receiver on its own thread, and calls
sent on command.

Like the ALE modem it knows nothing of the station: it is handed card-rate
audio stamped with the card's sample index, and a ``send(audio, rate)``
that puts a burst on the air and returns the card index it ended at (or
None if refused). Audio from inside its own bursts is not listened to.

Every call heard is reported, for whomever it was; a call for one of this
station's addresses — its own, or an all, group or sub-group call it falls
in — is reported as such. It answers one that asks for an acknowledgment
only if ``acknowledge`` is set, with an acknowledgment call (ending 122).
The radios' own answer (their "revertive") is not documented where we could
read it, so this is off until it has been heard against one.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..worker import ModemWorker

import numpy as np

from hfmodem.core.resample import FirDecimator as _Decimate
from . import code as C, fsk


@dataclass
class Event:
    kind: str                         # HEARD, CALLED, SENT, SEND_FAILED
    call: C.Call | None = None
    text: str = ""
    at: int = 0                       # card index
    inverted: bool = False
    freq_hz: float = 0.0


class Observer:
    def event(self, ev: Event) -> None: ...


class SelcallModem(ModemWorker):
    worker_name = "selcall"

    def __init__(self, addresses, *, send, card_rate: int = 48000, dots_s: float = 6.0,
                 invert: bool = False, acknowledge: bool = False,
                 observer: Observer | None = None, log=None) -> None:
        #: Its own addresses: one of four digits, one of six, or both; a
        #: call goes from the one as long as the address called.
        self.addresses = [addresses] if isinstance(addresses, str) else list(addresses)
        for a in self.addresses:
            C.address_chars(a)
        if len({len(a) for a in self.addresses}) != len(self.addresses):
            raise ValueError("at most one address of four digits and one of six")
        self.address = self.addresses[0]
        self.card_rate = card_rate
        self.dots_s = dots_s
        self.invert = invert
        self.acknowledge = acknowledge
        self._send_audio = send
        self.observer = observer or Observer()
        self._log = log or (lambda *_: None)
        self.rx = fsk.Receiver()
        self._dec = _Decimate(card_rate)
        self._quiet_until = 0
        self._init_worker()

    # -- from any thread ---------------------------------------------------------

    def feed(self, at: int, audio: np.ndarray) -> None:
        self._feed((at, np.asarray(audio, dtype=np.float32)))

    # -- calls ---------------------------------------------------------------------

    def reset_receive(self):
        self.rx = fsk.Receiver()
        self._dec = _Decimate(self.card_rate)

    def call(self, kind: str, to: str, **fields) -> C.Call:
        """Build and send a call from this station (on the modem's thread)."""
        frm = next((a for a in self.addresses if len(a) == len(to)), None)
        if frm is None:
            raise ValueError(f"no own address of {len(to)} digits to call {to} from")
        c = C.Call(kind, to, frm, **fields)
        audio = fsk.call_audio(c, dots_s=self.dots_s, invert=self.invert)
        end = self._send_audio(audio, fsk.FS)
        if end is None or isinstance(end, str):
            self._emit(Event("SEND_FAILED", c, f"transmission refused: {end or 'refused'}"))
        else:
            self._quiet_until = max(self._quiet_until, int(end))
            self._emit(Event("SENT", c, at=int(end)))
        return c

    def _emit(self, ev: Event) -> None:
        try:
            self.observer.event(ev)
        except Exception as exc:                    # noqa: BLE001
            self._log(f"selcall: observer: {exc}")

    # -- the receiver ------------------------------------------------------------------

    def _audio(self, at: int, x: np.ndarray) -> None:
        if at < self._quiet_until:
            x = x.copy()
            x[:min(x.size, self._quiet_until - at)] = 0.0
        for h in self.rx.push(self._dec.push(x)):
            if h.call is None:
                continue
            c = h.call
            mine = any(C.reaches(c.to, a) for a in self.addresses)
            self._emit(Event("CALLED" if mine else "HEARD", c, at=at + x.size,
                             inverted=h.inverted, freq_hz=h.freq_hz))
            if mine and self.acknowledge and c.end == C.ACK_RQ and c.kind != "ack" \
                    and c.to.rstrip("0") == c.to:            # individual calls only
                self.call("ack", c.frm)

