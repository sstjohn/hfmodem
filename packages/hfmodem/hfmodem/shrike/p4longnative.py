# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Optional whole-grant P4 LONG backend; never builds code on import."""
import ctypes as ct
from functools import lru_cache
import logging
from pathlib import Path
import sys
import time

import numpy as np

_ABI = 2
_WIDTH = 20700


class _Buffer(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('length', ct.c_size_t)]


class _Request(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ('members', 'llr_width', 'state_width')]
    _fields_ += [(name, ct.c_uint64) for name in ('last_tier', 'live', 'budget', 'deadline_ns')]
    _fields_ += [(name, _Buffer) for name in (
        'llrs', 'geometries', 'orders', 'whitening', 'priors', 'first',
        'state', 'bits', 'flags', 'timings')]


@lru_cache(maxsize=1)
def _tables():
    from . import p4longdecode as engine, p4frame
    orders = np.zeros((9, _WIDTH), dtype=np.uint32)
    for sl in range(2, 11):
        order = engine.geometry(sl).order
        orders[sl - 2, :len(order)] = order
    whitening = np.frombuffer(p4frame.whiten(bytes(2587)), np.uint8)
    orders.flags.writeable = False
    return orders, whitening


class _State:
    def __init__(self, job):
        m, w = job.priors.shape
        orders, whitening = _tables()
        self.arrays = dict(llrs=job.llrs, geometries=np.array(
            [member.geometry.speedlevel for member in job.members], dtype=np.uint32),
            orders=orders, whitening=whitening, priors=job.priors, first=job.first,
            state=np.zeros(6, dtype=np.uint64), bits=np.zeros((10, m, w), dtype=np.uint8),
            flags=np.zeros((10, m, 3), dtype=np.uint8), timings=np.zeros((10, 2)))
        self.request = _Request(m, job.llrs.shape[1], w)
        for name, array in self.arrays.items():
            assert array.flags.c_contiguous and array.flags.aligned
            setattr(self.request, name, _Buffer(array.ctypes.data, array.size))
        self.crossings = 0


class NativeBackend:
    def __init__(self, library):
        self.library = library
        self.entry = library.p4long_advance_v2
        self.entry.argtypes = [ct.c_uint32, ct.POINTER(_Request), ct.c_size_t]
        self.entry.restype = ct.c_int
        if self.entry(0, None, 0) != _ABI:
            raise ValueError('P4 LONG ABI mismatch')

    def __call__(self, job, deadline):
        from . import p4longdecode as engine
        if not hasattr(job, '_native'):
            job._native = _State(job)
        native = job._native
        request = native.request
        request.last_tier = job.last_tier
        request.live = job.policy == 'live'
        request.budget = min(job.component_budget, 2**64 - 1) if job.component_budget is not None else 2**64 - 1
        request.deadline_ns = 0 if deadline is None else max(1, min(int(deadline * 1e9), 2**64 - 1))
        state = native.arrays['state']
        state[:5] = job.tier, job.done, job.calls, job.positions, job.member_index
        start = job.tier * len(job.members) + job.member_index
        previous_timings = native.arrays['timings'].copy()
        code = self.entry(_ABI, ct.byref(request), ct.sizeof(request))
        native.crossings += 1
        if code:
            raise RuntimeError(f'P4 LONG native request failed ({code})')
        job.tier, done, job.calls, job.positions, job.member_index = map(int, state[:5])
        job.done = bool(done)
        end = job.tier * len(job.members) + job.member_index
        python_checks = np.zeros(10)
        for index in range(start, end):
            tier, i = divmod(index, len(job.members))
            checked = time.monotonic()
            member = job.members[i]
            crc, pad, flush = map(bool, native.arrays['flags'][tier, i])
            if crc:
                bits = native.arrays['bits'][tier, i, :member.geometry.systematic]
                outcome = engine._check(bits, member, i, tier, job.found)
                if (outcome.crc_valid, outcome.padding_valid, outcome.flush_valid) != (crc, pad, flush):
                    raise RuntimeError('P4 LONG native check mismatch')
            else:
                outcome = engine.CandidateOutcome(i,
                    'systematic' if tier == 0 else 'c1-only' if tier == 1 else 'turbo',
                    max(0, tier - 1), False, pad, False, None)
            job.outcomes.append(outcome)
            python_checks[tier] += time.monotonic() - checked
        for tier in sorted({index // len(job.members) for index in range(start, end)}):
            decode, check = native.arrays['timings'][tier] - previous_timings[tier]
            job.costs.append((f'tier-{tier}-decode', float(decode)))
            job.costs.append((f'tier-{tier}-check', float(check) + python_checks[tier]))
        if job.done and job.tier <= job.last_tier and not job.found:
            raise RuntimeError('P4 LONG native stop failed Python acceptance')

    def self_test(self):
        from . import p4longdecode as engine
        g = engine.geometry(2)
        fields = np.random.default_rng(1111).normal(size=g.serialized_size)
        member = engine.LongMember(g, fields)
        expected = engine.LongSearch([member], iterations=1, policy='exhaustive')
        engine._python_advance(expected, None)
        actual = engine.LongSearch([member], iterations=1, policy='exhaustive')
        self(actual, None)
        if actual.tier != expected.tier or actual.outcomes != expected.outcomes:
            raise ValueError('P4 LONG self-test decisions')
        np.testing.assert_allclose(actual.priors, expected.priors, atol=1e-12, rtol=1e-12)
        prior, first = np.zeros(g.systematic), np.zeros(g.systematic)
        for tier in range(3):
            bits, prior, first, _ = engine._member_step(g, fields, prior, first, tier)
            np.testing.assert_array_equal(actual._native.arrays['bits'][tier, 0], bits)


def _library_path():
    package = Path(__file__).resolve().parent
    suffix = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
    installed = package / ('_p4long_native' + suffix)
    if installed.is_file():
        return installed
    # A developer's explicit offline cargo build is usable without copying it.
    name = ('hfmodem_p4long' if sys.platform == 'win32' else 'libhfmodem_p4long') + suffix
    return package.parents[1] / 'native/p4long/target/release' / name


def _load(path=None):
    path = _library_path() if path is None else Path(path)
    if not path.is_file():
        logging.getLogger(__name__).info('P4 LONG native library absent; using Python')
        return None
    try:
        candidate = NativeBackend(ct.CDLL(str(path)))
        candidate.self_test()
        return candidate
    except (OSError, AttributeError, ValueError, RuntimeError, AssertionError) as error:
        logging.getLogger(__name__).warning('P4 LONG native backend unavailable: %s; using Python', error)
        return None


backend = _load()

