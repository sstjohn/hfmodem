# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Attached-mail progress after transport ACK, independent of peer absence."""
import math
import time
from typing import NamedTuple
from ..winlink.session import _GREETING, _WAIT_FS, _THEIR_TURN

DEFAULT_PROGRESS_SECONDS = 120.


class ProgressSnapshot(NamedTuple):
    deadline: float | None
    expired: bool
    failure_reason: str | None


class MailProgress:
    def __init__(self, client, transport, seconds=DEFAULT_PROGRESS_SECONDS, *, now=time.monotonic):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('P3 mail progress seconds must be finite and positive')
        self.client, self.transport, self.seconds, self.now = client, transport, seconds, now
        self.generation = transport._link_generation
        self.phase = None
        self.watermark = None
        self.deadline = None
        self.expired = False
        self.close_requested = False
        self.failure_reason = None
        self._p3_effects_applied = False
        self._logged = None

    def update(self, *, source=False, sent=False):
        a, session = self.transport, self.client.session
        if a._link_generation != self.generation:
            self.generation = a._link_generation
            self.phase = self.watermark = self.deadline = None
            self.expired = self.close_requested = False
            self.failure_reason = None
            self._p3_effects_applied = False
        if self.expired:
            return
        state = session._state
        banner = state == _GREETING and (source or self.phase == _GREETING)
        phase = state if state in (_WAIT_FS, _THEIR_TURN) or banner else None
        if phase is None:
            self.phase = self.watermark = self.deadline = None
            return
        if phase != self.phase or sent:
            self.phase, self.watermark, self.deadline = phase, a.tx_queued_bytes, None
        if source and (self.deadline is not None or phase == _GREETING):
            self.deadline = self.now() + self.seconds

    def check(self):
        a, session = self.transport, self.client.session
        if a.state not in ('CONNECTED', 'DISCONNECTING'):
            self.phase = self.watermark = self.deadline = None
            return ProgressSnapshot(None, self.expired, self.failure_reason)
        self.update()
        if self.expired:
            return ProgressSnapshot(self.deadline, True, self.failure_reason)
        if self.phase is None:
            return ProgressSnapshot(None, False, None)
        # The retired prefix is only unsent link-setup material before this
        # application's watermark. It is tracked separately from ACK credit.
        confirmed = a.tx_acked_bytes + a.tx_retired_setup_bytes
        if self.deadline is None and confirmed >= self.watermark:
            self.deadline = self.now() + self.seconds
        state = (session.stage, self.generation, self.watermark, a.tx_acked_bytes,
                 a.tx_retired_setup_bytes, self.deadline)
        if state != self._logged:
            self._logged = state
            remaining = None if self.deadline is None else max(0., self.deadline-self.now())
            a.io.log(f'mail progress phase={session.stage} generation={self.generation} '
                     f'watermark={self.watermark} acknowledged={a.tx_acked_bytes} '
                     f'retired_setup={a.tx_retired_setup_bytes} remaining={remaining}')
        if self.deadline is None or self.now() < self.deadline:
            return ProgressSnapshot(self.deadline, False, None)
        # The reason is decided now: a close below can end the link, and its
        # link_down must find the session already failed.
        self.fail(f'application progress timeout while {session.stage}; '
                  'transfer unconfirmed')
        return ProgressSnapshot(self.deadline, True, self.failure_reason)

    def fail(self, reason):
        session = self.client.session
        self.expired = True
        if self.phase == _GREETING:
            session.greeting_stalled()
        else:
            session._fail(reason)
        self.client._watch()
        self.failure_reason = session.failure

    def poll(self, *, qualified):
        a = self.transport
        if a.state not in ('CONNECTED', 'DISCONNECTING'):
            self.phase = self.watermark = self.deadline = None
            return False
        if not self.check().expired:
            return False
        if self._p3_effects_applied:
            if qualified and not self.close_requested:
                a._cancel_p3_proposals()
                self.close_requested = True
                a.on_host_disconnect(local=False)
            return True
        a._p3_app_failed = True
        self._p3_effects_applied = True
        a._cancel_p3_proposals()
        if not qualified:
            # Terminal application failure ends active retry permission even
            # before the peer-absence allowance is exhausted. Snapshot physical
            # receive ownership while the unconfirmed turn still exists, so a
            # later qualified peer can permit only the bounded goodbye.
            a._enter_receive_only()
        a.retain_unconfirmed('application progress timeout')
        # The retained copy is terminal history, never a pending retry. A
        # later peer return permits only the existing bounded goodbye.
        a._outbuf.clear()
        a._inflight = None
        a._breakin_pending = a._over_pending = False
        a._budget_goodbye = True
        if qualified:
            self.close_requested = True
            a.on_host_disconnect(local=False)
        return True

    def link_down(self):
        self.phase = self.watermark = self.deadline = None
        if not self.client.session.done:
            self.client.session._fail('P3 link ended; transfer unconfirmed')
            self.client._watch()
