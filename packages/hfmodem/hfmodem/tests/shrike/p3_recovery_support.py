# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import contextlib
import hashlib
import math
from unittest.mock import patch
import time

import numpy as np

from hfmodem.shrike import onair, p3acquire, rxfront
from hfmodem.tests.shrike import p3_deadline_support as deadline


class _ScaledClock:
    def __init__(self, scale):
        self.scale, self.origin = scale, time.perf_counter()

    def perf_counter(self):
        return self.origin + (time.perf_counter() - self.origin) * self.scale

    def __getattr__(self, name):
        return getattr(time, name)


class _SchedulingClock:
    def __init__(self, bench):
        self.bench = bench

    def perf_counter(self):
        return self.bench.sample_now() / onair.FS

    def __getattr__(self, name):
        return getattr(time, name)


class _MeasuredCosts:
    def __init__(self):
        self.values = {}

    @staticmethod
    def _identity(value):
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape,
                    hashlib.sha256(value.tobytes()).hexdigest())
        if isinstance(value, (tuple, list)):
            return tuple(_MeasuredCosts._identity(item) for item in value)
        if isinstance(value, dict):
            return tuple((key, _MeasuredCosts._identity(item))
                         for key, item in sorted(value.items()) if key != "can_decode")
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return value
        return type(value).__qualname__

    def seconds(self, stage, args, kwargs, elapsed):
        identity = (stage, self._identity(args), self._identity(kwargs))
        key = hashlib.sha256(repr(identity).encode()).hexdigest()
        return self.values.setdefault(key, elapsed)


@contextlib.contextmanager
def _charged_stages(replay, scale, recovery):
    stack = []
    # Recovery admission must see the packet searches already completed;
    # charging them at upgrade_scan's return lends it time already spent.
    stages = (*deadline.CHARGED, *deadline.MEASURED,
              ("packet", replay.sessrx.sync, "packet"),
              ("blind", rxfront, "decode_expected_packet"),
              ("search", p3acquire, "_search"),
              ("body", rxfront, "_cs_event"),
              ("extent", p3acquire, "_inside_burst"),
              ("p2", onair._SessionRx, "_p2_packet"),
              ("recovery", replay.sessrx, "_p3_recovery_changeover"),
              ("tail", replay, "_postkey_tail"))
    with contextlib.ExitStack() as patches:
        clock = (_SchedulingClock(replay.bench) if replay.measured_costs is not None
                 else _ScaledClock(scale))
        patches.enter_context(patch.object(onair, "time", clock))
        for stage, owner, name in stages:
            original = getattr(owner, name)

            def charged(*args, _original=original, _stage=stage, **kwargs):
                postkey = _stage == "scan" and kwargs.get("upgrade")
                previous_recovery = replay.sessrx._p3_recovery
                if postkey:
                    replay.sessrx._p3_recovery = recovery
                identity = dict(kwargs)
                if _stage in ("scan", "control", "flush", "upgrade", "tail", "recovery",
                              "packet", "blind"):
                    identity["replay_state"] = (len(replay.cycles), replay.tx.slot,
                        replay.host.protocol.name, replay.host.arq.role, replay.sessrx.count,
                        replay.sessrx._p3_row0)
                if _stage in ("packet", "blind"):
                    sync = replay.sessrx.sync
                    memory = (sync.memory if _stage == "packet" else
                              args[1] if len(args) > 1 else kwargs.get("memory"))
                    identity["memory_state"] = _MeasuredCosts._identity(
                        vars(memory) if memory is not None else None)
                    if _stage == "packet":
                        identity["tracker_state"] = (
                            sync.packet_at, sync.packet_level, sync.rotation,
                            sync.misses, sync.track_symbols)
                frame = [time.perf_counter(), 0.0, _stage]
                stack.append(frame)
                at = replay.bench.sample_now()
                tail_started = onair.time.perf_counter() if _stage == "tail" else None
                try:
                    result = _original(*args, **kwargs)
                    if postkey:
                        replay.sessrx._p3_recovery = previous_recovery
                        replay._postkey_tail(args[1], args[2])
                    return result
                finally:
                    if postkey:
                        replay.sessrx._p3_recovery = previous_recovery
                    elapsed = time.perf_counter() - frame[0]
                    stack.pop()
                    replay.cost[_stage].append(elapsed * 1e3)
                    exclusive = max(0.0, elapsed - frame[1])
                    if replay.measured_costs is not None:
                        exclusive = replay.measured_costs.seconds(
                            _stage, args, identity, exclusive)
                    charge = math.ceil(exclusive * scale * onair.FS)
                    replay.bench.spend(charge)
                    if _stage == "tail":
                        replay.sessrx._note_p3_recovery_cost("tail", args[0].size,
                                                           tail_started)
                    replay.work.append(dict(cycle=len(replay.cycles), stage=_stage,
                                            recovery_work=_stage == "recovery" or any(
                                                parent[2] == "recovery" for parent in stack),
                                            at=at, elapsed_ms=elapsed * 1e3,
                                            exclusive_ms=exclusive * 1e3,
                                            charged_samples=charge))
                    if stack:
                        stack[-1][1] += time.perf_counter() - frame[0]

            patches.enter_context(patch.object(owner, name, charged))
        accept = replay.sessrx._note_p3_frame

        def accepted(event, absolute):
            result = accept(event, absolute)
            if result:
                replay.deliveries.append(dict(
                    cycle=len(replay.cycles), at=replay.bench.sample_now(),
                    slot=replay.tx.slot,
                    identity=(absolute, event.breakin, repr(event.packet)),
                    emissions=len(replay.bench.emissions)))
            return result

        patches.enter_context(patch.object(replay.sessrx, "_note_p3_frame", accepted))
        fits = replay.sessrx._p3_recovery_fits

        def admitted(reserve):
            result = fits(reserve)
            replay.admissions.append(dict(cycle=len(replay.cycles),
                slot=replay.tx.slot,
                next_key=replay.tx.key_instant(
                    replay.grid, replay.grid.next_slot(replay.tx.slot)),
                at=replay.bench.sample_now(), reserve_s=reserve, admitted=result))
            return result

        patches.enter_context(patch.object(replay.sessrx, "_p3_recovery_fits", admitted))
        transmit = replay.bench.transmit

        def emitted(audio, **kwargs):
            raster, tx = replay.grid, replay.tx
            record = dict(cycle=len(replay.cycles), requested=kwargs.get("at"),
                          slot=tx.slot, boundary=raster.boundary(tx.slot),
                          earliest=tx.key_instant(raster, tx.slot),
                          accepted_crc_count=len(replay.deliveries))
            result = transmit(audio, **kwargs)
            record.update(actual=result[0], end=result[1])
            replay.placements.append(record)
            return result

        patches.enter_context(patch.object(replay.bench, "transmit", emitted))
        yield


class _RecoveryReplay(deadline.Replay):
    def __init__(self, *args, scale, recovery, measured_costs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.measured_costs = measured_costs
        self.scale, self.recovery = scale, recovery
        self.cost.update({name: [] for name in
                          ("packet", "blind", "search", "body", "extent",
                           "p2", "tail", "recovery")})
        self.work, self.admissions, self.deliveries, self.placements = [], [], [], []
        self.tx.record_audio = False
        self.evidence = onair._CycleEvidence(self.bench, self.tx, self.grid, self.tx.outdir)
        self.spectrum = onair._PeerSpectrum()
        self.held_at = 0

    def _charging(self):
        return _charged_stages(self, self.scale, self.recovery)

    def _postkey_tail(self, window, origin):
        receiver, grid, host = self.sessrx, self.grid, self.host
        bursts = self.evidence.record("recovery-replay", window, origin)
        onsets = [at for at, _ in bursts]
        onair._read_codeword_at_bursts(receiver, window, origin, onsets)
        if (grid.sending and host.protocol is onair.Protocol.PACTOR1
                and not receiver.words_at):
            onair._grid_answer(receiver, window, origin, origin - self.held_at)
        onair._grid_reversal(grid, host)
        self.spectrum.hear(window[-grid.cycle_n:])
        onair._forecast_next_key(receiver, self.tx, grid, origin)

    def _cycle(self, slot):
        self.held_at = self.tx.tx_end or 0
        result = super()._cycle(slot)
        self.cycles[-1]["accepted_crc_count"] = len(self.deliveries)
        return result
