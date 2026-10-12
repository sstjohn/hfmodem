# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fixed normal-short candidates, with single-copy deadline-aware decoding.

SL7/8 use fixed component permutations.
Compact index swaps preserve the exact permutations.
Independent verification matched the component permutations' int64 little-endian
SHA-256 hashes: 837e4997d27088d7f30052c9d9ea8fa4fbf060b81e1a4b7ed3cf2432356d8159
(704 entries) and c67d4f7da1ad3b758b0e84672bd5e0abf80b8a15ffe4f9fab531e83df0339c4b
(1056 entries). Four recorded fields passed CRC; one-bit corruptions and wrong
PSK orders were rejected.
"""
from time import monotonic

import numpy as np

from . import p4c2, p4coding, p4decode, p4frame, p4qam16, p4short, tablegen

_SYMBOL_ORDER = tablegen.INTERLEAVERS['0xbaf20_1056']()
# Below this, C1 hard decisions do not establish a level change. Retain the
# legacy BPSK attempt first; all fixed levels remain eligible after its failure.
LEVEL_CHANGE_AGREEMENT = .55

_COMPONENT_704 = p4c2.permutation(704)
_COMPONENT_1056 = p4c2.permutation(1056)


def component_permutation(bits):
    if bits == 4:
        return p4qam16.component_permutation()
    if bits not in (1, 2, 3):
        raise ValueError('normal short requires 1, 2, 3 or 4 label bits')
    return (p4short.component_permutation() if bits == 1 else
            (_COMPONENT_704 if bits == 2 else _COMPONENT_1056).copy())


def demap(symbols, noise, bits):
    """MSB-first Gray labels, +I = all zero, positive LLR = bit zero.

    Noise is complex post-equalizer variance referred to unit signal gain.
    Gather symbols before flattening their labels into systematic/C1/C2.
    """
    symbols = np.asarray(symbols, complex)
    if (symbols.shape != (1056,) or not np.all(np.isfinite(symbols))
            or not np.isfinite(noise) or noise <= 0 or bits not in (1, 2, 3)):
        raise ValueError('1056 finite symbols, positive noise and 1..3 label bits required')
    phase = np.arange(1 << bits)
    points = np.exp(2j * np.pi * phase / len(phase))
    labels = ((phase ^ (phase >> 1))[:, None] >> np.arange(bits-1, -1, -1)) & 1
    distance = abs(symbols[:, None] - points)**2
    scores = np.stack([distance[:, labels[:, k] == 1].min(axis=1)
                       - distance[:, labels[:, k] == 0].min(axis=1)
                       for k in range(bits)], axis=1) / noise
    return scores[_SYMBOL_ORDER].ravel()


def agreement(fields):
    """C1 recurrence p[t]^p[t-2]^p[t-3] = u[t]^u[t-1]^u[t-3]."""
    systematic, parity, _ = np.split(np.asarray(fields) < 0, 3)
    t = np.arange(3, len(systematic)-8)
    errors = (systematic[t] ^ systematic[t-1] ^ systematic[t-3]
              ^ parity[t] ^ parity[t-2] ^ parity[t-3])
    return float(np.mean(~errors))


def candidates(symbols, noise, bpsk, preferred=None, *, qam=False):
    fields = {1: bpsk, 2: demap(symbols, noise, 2), 3: demap(symbols, noise, 3)}
    if qam and not p4qam16._lower_order_body(symbols):
        fields[4] = p4qam16.demap(symbols, noise)
    return sorted(fields.items(), key=lambda pair: (agreement(pair[1]), pair[0] == preferred),
                  reverse=True)


def _frame(bits, width, cache=None):
    size = 44 * width - 4
    wire = np.packbits(bits[:8*(size+3)], bitorder='little').tobytes()
    if cache is not None:
        key = width, wire
        if key not in cache:
            cache[key] = p4frame.decode_field(wire, data_bytes=size)
        return cache[key]
    return p4frame.decode_field(wire, data_bytes=size)


def codeword(frame, width):
    if width == 4:
        return p4qam16.codeword(frame)
    systematic = np.unpackbits(np.frombuffer(p4frame.encode_field(frame.data, frame.status),
                                             dtype=np.uint8), bitorder='little').tolist() + [0]*5
    state = p4coding.rsc_encode(systematic, initial_state=(0, 0, 0)).final_state
    systematic += list(p4coding.rsc_terminate(state).systematic)
    c1, c2 = p4coding.encode_components(systematic, permutation=component_permutation(width),
                                       initial_states=((0, 0, 0), (0, 0, 0)))
    return np.r_[systematic, c1.parity, c2.parity]


class QamShortDecode:
    """Complete the declared credible direct modes before publishing a field."""
    def __init__(self, ranked, *, costs, events, at, status_ok, now=monotonic):
        self.costs, self.events, self.at = costs, events, at
        self.status_ok, self.now = status_ok, now
        self.pending = []
        self.valid = {}
        self.references = {}
        self.fields = {}
        self.frame = self.width = None
        self.done = self.deferred = False
        self.deadline = None
        self.fallback = None
        self.remaining = []
        for width, fields in ranked:
            self.costs.setdefault(width, [])
            if agreement(fields) < LEVEL_CHANGE_AGREEMENT:
                if width != 4:
                    self.remaining.append((width, fields))
                continue
            systematic, c1, c2 = np.split(fields, 3)
            c1, c2 = c1.copy(), c2.copy()
            c1[-8:] = c2[-8:] = 0
            decoder = p4decode.turbo_iterations(systematic, c1, c2,
                                                component_permutation(width), iterations=8)
            self.pending.append([width, fields, decoder, 0])

    def _expired(self):
        if self.deadline is None or self.now() < self.deadline:
            return False
        self.frame = self.width = None
        self.valid.clear()
        self.pending.clear()
        self.done = True
        return True

    def discard(self):
        self.frame = self.width = None
        self.valid.clear()
        self.pending.clear()
        self.done = True

    def decode(self, deadline=None):
        if deadline is not None:
            self.deadline = deadline if self.deadline is None else min(self.deadline, deadline)
        if self._expired():
            return None
        if self.done:
            return self.frame
        while self.pending:
            width, fields, decoder, iteration = self.pending[0]
            times = self.costs.setdefault(width, [])
            budget = max((t for t in times[-32:] if t > 0),
                         default=.007 if width == 4 else .004*width) * 1.2
            start = self.now()
            allowance = None if self.deadline is None else self.deadline-start
            event = dict(event='normal-decode', at=self.at, stage='single', copies=1,
                         level='SL9/10-short' if width == 4 else 6 if width == 1 else width+5,
                         iterations=iteration+1, budget_ms=1000*budget, elapsed_ms=0.,
                         allowance_ms=None if allowance is None else 1000*allowance,
                         crc_valid=None)
            if allowance is not None and allowance < budget:
                if not self.deferred:
                    self.events.append(dict(event, reason='budget'))
                self.deferred = True
                return None
            self.deferred = False
            result = next(decoder)
            self.pending[0][3] += 1
            frame = _frame(result.bits, width, self.fields)
            event.update(crc_valid=frame is not None, reason='CRC')
            if frame is not None:
                key = frame.status.byte, frame.data, frame.crc
                if not self.status_ok(frame, combined=False):
                    event['reason'] = 'status'
                else:
                    reference_key = width, key
                    if reference_key not in self.references:
                        self.references[reference_key] = codeword(frame, width)
                    reference = self.references[reference_key]
                    weight = np.sum(abs(fields))
                    mismatch = np.sum(abs(fields)[(fields < 0) != reference])
                    limit = p4qam16._MAX_REENCODED_DISTANCE if width == 4 else .40
                    if not weight or mismatch > limit*weight:
                        event['reason'] = 'distance'
                    else:
                        self.valid[key] = frame, width
                        event['reason'] = 'candidate'
            elapsed = max(0., self.now()-start)
            times.append(elapsed)
            del times[:-32]
            event['elapsed_ms'] = 1000*elapsed
            if self._expired():
                self.events.append(dict(event, reason='late'))
                return None
            if self.done:
                self.events.append(dict(event, reason='ineligible'))
                return None
            self.events.append(event)
            if iteration+1 == 8:
                self.pending.pop(0)
        if len(self.valid) > 1:
            self.events.append(dict(event='normal-ambiguous', at=self.at))
            self.done = True
        elif self.valid:
            frame, width = next(iter(self.valid.values()))
            if self.status_ok(frame, combined=False) and not self._expired():
                self.frame, self.width = frame, width
                self.events.append(dict(event='normal-decode', at=self.at, stage='complete',
                    level='SL9/10-short' if width == 4 else 6 if width == 1 else width+5,
                    reason='accepted', crc_valid=True))
            self.done = True
        elif self.remaining:
            if self.fallback is None:
                self.fallback = ShortDecode(self.remaining, costs=self.costs, events=self.events,
                    at=self.at, status_ok=self.status_ok, now=self.now)
            frame = self.fallback.decode(self.deadline)
            if self._expired():
                if frame is not None:
                    self.events[-1]['reason'] = 'late'
            else:
                self.frame, self.width = frame, self.fallback.width
                self.done = self.fallback.done
        else:
            self.done = True
        return self.frame


class ShortDecode:
    """Bounded candidate search; no combining or held payload across emissions."""
    def __init__(self, ranked, *, costs, events, at, status_ok, now=monotonic):
        self.ranked = ranked
        self.costs, self.events, self.at = costs, events, at
        self.status_ok, self.now = status_ok, now
        self.frame = self.width = None
        self.pending = []
        self.done = False
        self.valid = {}
        self.crc_valid = set()
        self.deferred = False

    def _group(self):
        if not self.ranked:
            self.done = True
            return
        best = agreement(self.ranked[0][1])
        group = [pair for pair in self.ranked if agreement(pair[1]) == best]
        self.ranked = [pair for pair in self.ranked if agreement(pair[1]) != best]
        for width, fields in group:
            systematic, c1, c2 = np.split(fields, 3)
            c1, c2 = c1.copy(), c2.copy()
            c1[-8:] = c2[-8:] = 0
            self.pending.append([width, fields, p4decode.turbo_iterations(
                systematic, c1, c2, component_permutation(width)), 0])

    def decode(self, deadline=None):
        while not self.done:
            if not self.pending:
                if self.valid:
                    self.done = True
                    if len(self.crc_valid) == 1:
                        self.frame, self.width = next(iter(self.valid.values()))
                    else:
                        self.events.append(dict(event='normal-ambiguous', at=self.at))
                    break
                self._group()
                if self.done:
                    break
            width, fields, decoder, iteration = self.pending[0]
            times = self.costs[width]
            budget = max((t for t in times[-32:] if t > 0), default=.004 * width) * 1.2
            start = self.now()
            allowance = None if deadline is None else deadline-start
            event = dict(event='normal-decode', at=self.at, stage='single', copies=1,
                         level=6 if width == 1 else width+5, iterations=iteration+1,
                         allowance_ms=None if allowance is None else 1000*allowance,
                         budget_ms=1000*budget, elapsed_ms=0., crc_valid=None)
            if allowance is not None and allowance < budget:
                if not self.deferred:
                    self.events.append(dict(event, reason='budget'))
                    self.deferred = True
                return None
            self.deferred = False
            result = next(decoder)
            iteration += 1
            self.pending[0][3] = iteration
            frame = _frame(result.bits, width)
            event.update(crc_valid=frame is not None, reason='CRC')
            if frame is not None:
                self.crc_valid.add((frame.status.byte, frame.data, frame.crc))
                # Training fixes polarity. Require codeword agreement and the
                # same status admission as the existing single-copy reader.
                weight = np.sum(abs(fields))
                mismatch = np.sum(abs(fields)[(fields < 0) != codeword(frame, width)])
                if not self.status_ok(frame, combined=False):
                    event['reason'] = 'status'
                elif weight == 0 or mismatch > .40 * weight:
                    event['reason'] = 'distance'
                else:
                    event['reason'] = 'accepted'
                    self.valid[(frame.status.byte, frame.data, frame.crc)] = (frame, width)
            elapsed = max(0., self.now()-start)
            times.append(elapsed)
            del times[:-32]
            event['elapsed_ms'] = 1000*elapsed
            self.events.append(event)
            if event['reason'] == 'accepted' or iteration == 8:
                self.pending.pop(0)
        return self.frame
