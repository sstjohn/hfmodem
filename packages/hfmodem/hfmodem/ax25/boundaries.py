# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded observation of local raw KISS link handshakes."""
from collections import OrderedDict
import threading
from types import SimpleNamespace
from uuid import uuid4

from hfmodem.core.recording import inherit_boundaries, contact_established, contact_ended
from .connected import callsign, SABM, UA, DISC, DM, FRMR
from .frame import parse


class RawBoundaries:
    candidate_limit = 32
    active_limit = 16

    def __init__(self, source, local):
        self.source, self.local = source, local
        self.candidates = OrderedDict()
        self.active = {}
        self.generation = self.epoch = 0
        self._endpoint = uuid4().hex
        self._lock = threading.RLock()

    def _report(self, message):
        boundaries = getattr(self.source, 'recording_boundaries', None)
        if boundaries is not None:
            boundaries.report(message)

    def close(self, reason):
        with self._lock:
            self.epoch += 1
            for source in self.active.values():
                contact_ended(source, reason)
            self.active.clear()
            self.candidates.clear()

    def observe(self, raw, *, transmitted=False, epoch=None):
        if getattr(self.source, 'recording_boundaries', None) is None:
            return
        try:
            with self._lock:
                if epoch is None or epoch == self.epoch:
                    self._observe(raw, transmitted)
        except Exception as exc:
            self._report(f'Recording boundary: {exc}')

    def _observe(self, raw, transmitted):
        try:
            frame = parse(raw)
        except ValueError:
            return
        dest, src = map(callsign, frame.addresses[:2])
        local, peer = (src, dest) if transmitted else (dest, src)
        if local != self.local:
            return
        path = frame.addresses[2:]
        if not transmitted and any(not a[6] & 0x80 for a in path):
            return
        command = bool(frame.addresses[0][6] & 0x80)
        response = bool(frame.addresses[1][6] & 0x80)
        if command == response:
            return
        route = tuple(callsign(a) for a in (path if transmitted else reversed(path)))
        slot = (local, peer)
        kind, pf = frame.control & 0xEF, bool(frame.control & 0x10)
        active = self.active.get(slot)
        if kind in (SABM, 0x6F, DISC) and command:
            if slot not in self.candidates and len(self.candidates) >= self.candidate_limit:
                self.candidates.popitem(last=False)
                self._report('Recording raw KISS candidate overflow')
            self.candidates[slot] = (kind, transmitted, route, pf)
        elif kind == UA and response:
            candidate = self.candidates.get(slot)
            if candidate is None:
                return
            setup, outgoing, expected_route, poll = candidate
            if outgoing == transmitted or route != expected_route or pf != poll:
                return
            del self.candidates[slot]
            if setup == DISC:
                if active is not None:
                    contact_ended(active)
                    del self.active[slot]
                return
            if active is not None and not active.traffic:
                return
            if active is not None:
                contact_ended(active, 'raw KISS reset after traffic')
                del self.active[slot]
            if len(self.active) >= self.active_limit:
                self._report('Recording raw KISS active overflow')
                return
            self.generation += 1
            source = inherit_boundaries(SimpleNamespace(
                traffic=False, route=route, _recording_endpoint=self._endpoint), self.source)
            self.active[slot] = source
            contact_established(source, 'rpr', local, peer, slot=peer,
                                generation=self.generation)
            cid = getattr(source, '_recording_id', None)
            if cid is not None:
                source.recording_boundaries.session.enrich(
                    cid, uncertainty='Raw KISS client timeouts are not observable')
        elif kind in (DM, FRMR) and response:
            candidate = self.candidates.get(slot)
            if candidate is not None and candidate[1] != transmitted and candidate[2] == route:
                self.candidates.pop(slot)
            if active is not None and route == active.route:
                contact_ended(active, 'raw KISS peer refused or reset connection')
                del self.active[slot]
        elif active is not None and route == active.route and frame.control & 3 != 3:
            active.traffic = True
