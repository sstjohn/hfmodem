# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Server-owned, attachment-independent KISS transmit requests."""
from collections import deque
from dataclasses import dataclass
import threading
import time

from ..tx.ui import KissTxDisposition


@dataclass
class KissTxRequest:
    kind: str
    value: object
    arrived: float
    completed: object = None
    disposition: KissTxDisposition | None = None


class KissTxQueue:
    limit = 64
    holdoff = 2.0
    max_age = 600.0

    def __init__(self, *, clock=time.monotonic, log=None, diagnostic=None):
        self.clock = clock
        self._log = log or (lambda *args: None)
        self._diagnostic = diagnostic
        self._lock = threading.Lock()
        self._requests = deque()
        self._claimed = None
        self._closed = False
        self.busy = False
        self.quiet_at = self.unkey_at = None

    def __len__(self):
        with self._lock:
            return len(self._requests)

    def broadcast(self, command, raw, *, completed=None):
        from ..ui import UIFrame
        from ..tx.ui import packetize
        try:
            parsed = UIFrame.from_ax25(raw, command)
            message = parsed.normalized()
            _, frames = packetize(message)
        except ValueError as error:
            reason = str(error)
            self._log('kiss', f'UI frame ignored: {reason}')
            if self._diagnostic is not None:
                try:
                    self._diagnostic('kiss_refused', reason=reason)
                except Exception:
                    pass
            return False
        if parsed.addresses != message.addresses:
            self._log('kiss', 'warning: lowercase UI identifiers uppercased')
        if any(c > 127 for c in message.info):
            self._log('kiss', 'note: UI information is not 7-bit text; sent at 8-bit width')
        if len(raw) > 751 or len(frames) == 18:
            self._log('kiss', 'warning: UI exceeds the receive edge measured on VARA HF '
                      'stations; they may drop it')
        return self._enqueue('broadcast', message, completed)

    def _enqueue(self, kind, value, completed=None):
        with self._lock:
            if self._closed or len(self._requests) >= self.limit:
                self._log('kiss', 'transmit queue closed or full; new frame dropped')
                return False
            self._requests.append(KissTxRequest(kind, value, self.clock(), completed))
            return True

    def activity(self, busy):
        with self._lock:
            if self.busy and not busy:
                self.quiet_at = self.clock()
            self.busy = bool(busy)

    def unkey(self):
        with self._lock:
            self.unkey_at = self.clock()

    def claim(self):
        """Called under the current modem's lock, before transport admission."""
        with self._lock:
            if self._closed or self._claimed is not None or not self._requests or self.busy:
                return None
            head = self._requests[0]
            anchors = [head.arrived]
            anchors.extend(t for t in (self.quiet_at, self.unkey_at) if t is not None)
            if self.clock() < max(anchors) + self.holdoff:
                return None
            self._claimed = head
            return head

    def finish(self, head, disposition):
        with self._lock:
            if self._claimed is not head:
                return
            self._claimed = None
            previous, head.disposition = head.disposition, disposition
            expired = (disposition is KissTxDisposition.REFUSED_BEFORE_KEY
                       and self.clock() - head.arrived >= self.max_age)
            terminal = expired or disposition in (KissTxDisposition.SENT,
                KissTxDisposition.PARTIAL, KissTxDisposition.UNKEY_UNCONFIRMED)
            if terminal and self._requests and self._requests[0] is head:
                self._requests.popleft()
        if expired:
            self._log('kiss', 'transmit head refused for 10 minutes; dropped')
        elif disposition in (KissTxDisposition.PARTIAL, KissTxDisposition.UNKEY_UNCONFIRMED):
            self._log('kiss', f'transmit {disposition.name}; frame dropped')
        elif previous is not disposition and disposition is KissTxDisposition.CONTENDED:
            self._log('kiss', 'transmit contended; frame held')
        if terminal and head.completed is not None:
            head.completed(disposition is KissTxDisposition.SENT,
                           '' if disposition is KissTxDisposition.SENT else disposition.name.lower())

    def stop(self):
        with self._lock:
            if self._closed:
                return
            count = len(self._requests)
            self._requests.clear()
            self._closed = True
        self._log('kiss', f'server stopped; discarded {count} queued transmit frames')
