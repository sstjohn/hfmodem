# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One station drainer routes bounded observation copies by captured authority."""
from copy import deepcopy
import queue
import threading


class Diagnostics:
    def __init__(self, *, capacity=1024):
        self._pending = queue.Queue(capacity)
        self._lock = threading.Lock()
        self._subscribers = {}
        self._dropped = {}

    def subscribe(self, authority, callback):
        token = object()
        with self._lock:
            self._subscribers[token] = (tuple(authority), callback)
        def unsubscribe():
            with self._lock:
                self._subscribers.pop(token, None)
                if not any(a == tuple(authority) for a, _ in self._subscribers.values()):
                    self._dropped.pop(tuple(authority), None)
        return unsubscribe

    def put(self, kind, *, authority=None, **fields):
        authority = tuple(authority) if authority is not None else None
        try:
            self._pending.put_nowait((kind, authority, deepcopy(fields)))
        except queue.Full:
            with self._lock:
                # Session count does not create an unbounded loss map.
                interested = any(a == authority for a, _ in self._subscribers.values())
                if interested:
                    self._dropped[authority] = self._dropped.get(authority, 0) + 1

    def _route(self, kind, authority, fields):
        with self._lock:
            callbacks = [callback for captured, callback in self._subscribers.values()
                         if captured == authority]
        for callback in callbacks:
            try:
                callback(kind, **deepcopy(fields))
            except Exception:
                pass  # An observation consumer cannot terminate the station owner.

    def drain(self, *, arbiter=None, rig=None, publish=None, limit=256):
        """Called only by the station polling owner, never a chat controller.

        Meter producers must attach their captured authority. An untagged
        report may be published station-wide but cannot be assigned to chat.
        """
        for source, kind in ((getattr(arbiter, 'records', None), 'arbiter'),
                             (getattr(rig, 'meter_reports', None), 'tx_meters')):
            if source is None:
                continue
            for _ in range(limit):
                try:
                    fields = deepcopy(source.get_nowait())
                except queue.Empty:
                    break
                authority = fields.pop('authority', None)
                authority = tuple(authority) if authority is not None else None
                if publish is not None:
                    try:
                        publish('tx', dict(kind=kind, authority=authority, **deepcopy(fields)))
                    except Exception:
                        pass
                self._route(kind, authority, fields)
        for _ in range(limit):
            try:
                kind, authority, fields = self._pending.get_nowait()
            except queue.Empty:
                break
            self._route(kind, authority, fields)
        with self._lock:
            dropped, self._dropped = self._dropped, {}
        for authority, count in dropped.items():
            self._route('dropped', authority, dict(count=count))
