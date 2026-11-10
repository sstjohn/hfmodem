# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Opt-in scalar observer. No scheduling decisions or replacement test oracle."""
import json
import math
import time


def _callback_coverage(rows, epoch, blocksize, offset, i0, key_entry, ptt_up):
    """Require contiguous real callback records around keying and first output."""
    if not rows:
        return False
    try:
        previous = None
        physical = []
        for row in rows:
            number, due, adc, dac, entered, exited, recorded_epoch = row
            if (not isinstance(number, int) or number < 0
                    or not all(math.isfinite(x) for x in row)
                    or recorded_epoch != epoch or exited < entered):
                return False
            expected = epoch + number * blocksize / 48000
            if (abs(due - expected) > 1e-8 or abs(adc - expected) > 1e-8
                    or abs(dac - (expected + offset / 48000)) > 1e-8):
                return False
            if previous is not None and (number != previous[0] + 1 or entered < previous[5]):
                return False
            if number * blocksize <= i0 < (number + 1) * blocksize:
                physical.append(row)
            previous = row
        if len(physical) != 1 or not all(math.isfinite(x) for x in (key_entry, ptt_up)):
            return False
        first = physical[0]
        first_air = epoch + (i0 + offset) / 48000
        lo = min(key_entry, ptt_up, first[1], first_air)
        hi = max(key_entry, ptt_up, first[1], first_air)
        actual_lo = min(key_entry, ptt_up, first[4], first_air)
        actual_hi = max(key_entry, ptt_up, first[5], first_air)
        return (rows[0][1] <= lo and rows[-1][1] + blocksize / 48000 >= hi
                and rows[0][4] <= actual_lo and rows[-1][5] >= actual_hi)
    except (TypeError, ValueError, IndexError):
        return False


class Witness:
    def __init__(self, enabled, card, audio):
        self.enabled, self.card, self.audio = enabled, card, audio
        self.events = [None] * 128
        self.callbacks = [None] * 256
        self.costs = [None] * 512
        self.callback_costs = [None] * 256
        self.ne = self.nc = self.ncost = 0
        self.overflow = 0
        self.window = None

    def _cost(self, start):
        end = time.monotonic()
        if self.ncost < len(self.costs):
            self.costs[self.ncost] = (start, end)
            self.ncost += 1
        else:
            self.overflow += 1

    def wrap(self, original, kind):
        def forward(*args, **kwargs):
            stamp = time.monotonic()
            anchor = self.audio._last
            epoch = self.card.t0
            slot = self.ne
            self.ne += 1
            if slot >= len(self.events):
                self.overflow += 1
            target = None
            if kind == 'arm':
                target = args[1] if len(args) > 1 else kwargs.get('at')
                self.window = target
            elif kind == 'wait':
                target = args[0] if args else kwargs.get('card_index')
            elif kind == 'key':
                target = kwargs.get('settle_s', 0.0)
            elif kind == 'cat':
                line = args[0] if args else kwargs.get('line', '')
                target = 'frequency' if line == 'f' else 'other'
            self._cost(stamp)
            entered = time.monotonic()
            returned = None
            ok = False
            try:
                value = original(*args, **kwargs)
                returned = time.monotonic()
                ok = True
                if kind == 'arm':
                    self.window = value
                    scalar = value
                else:
                    scalar = None
                return value
            finally:
                stopped = time.monotonic()
                if returned is None:
                    returned = stopped
                    scalar = None
                if slot < len(self.events):
                    self.events[slot] = (kind, target, scalar, entered, returned,
                                         anchor, self.audio._last, epoch, self.card.t0, ok)
                self._cost(stopped)
        return forward

    def callback(self, original):
        def forward(indata, outdata, frames, times, status):
            target = self.window
            adc = times.inputBufferAdcTime
            # A fixed 200 ms grid window spans key and first output; never move
            # the card epoch or inspect/alter sample arrays on this path.
            observe = target is not None and abs(adc - (self.card.t0 + target / 48000)) <= .1
            if not observe:
                return original(indata, outdata, frames, times, status)
            entered = time.monotonic()
            try:
                return original(indata, outdata, frames, times, status)
            finally:
                exited = time.monotonic()
                if self.nc < len(self.callbacks):
                    slot = self.nc
                    self.callbacks[slot] = (len(self.card.out), adc, adc,
                        times.outputBufferDacTime, entered, exited, self.card.t0)
                    self.nc += 1
                    self.callback_costs[slot] = (exited, time.monotonic())
                else:
                    self.overflow += 1
        return forward

    def flush(self, path, bench):
        """Called after test assertions and card shutdown, including failing tests."""
        events = [event for event in self.events if event is not None]
        data = dict(schema_version=1, enabled=self.enabled, events=events,
                    callbacks=self.callbacks[:self.nc], overflow=self.overflow,
                    observer_storage_spans=self.costs[:self.ncost],
                    callback_storage_spans=self.callback_costs[:self.nc],
                    observer_limits='Measured spans exclude scheduler interference; no proof of zero observer effect.',
                    ptt_edges=bench.ptt.edges, card_epoch=bench.card.t0,
                    conclusive=False, decompositions=[],
                    event_fields=['kind', 'target_or_command_class_or_settle_s', 'returned_at',
                        'entry', 'return', 'entry_anchor', 'return_anchor', 'entry_epoch', 'return_epoch', 'success'],
                    callback_fields=['number', 'due', 'adc', 'dac', 'entry', 'exit', 'epoch'],
                    callback_lateness=[c[4]-c[1] for c in self.callbacks[:self.nc]])
        if self.enabled:
            i0, carried = bench.card.carried()
            data.update(first_output=i0, carried_samples=len(carried))
            arms = [e for e in events if e[0] == 'arm' and e[9]]
            keys = [e for e in events if e[0] == 'key' and e[9]]
            ups = [t for t, on in bench.ptt.edges if on]
            for arm, key, up in zip(arms, keys, ups):
                a = arm[2]
                settle = int(key[1] * 48000)
                waits = [e for e in events if e[0] == 'wait' and e[1] == a - settle
                         and arm[4] <= e[3] <= key[3] and e[9]]
                if len(waits) != 1:
                    continue
                wait = waits[0]
                anchors = [e[j] for e in (arm, wait, key) for j in (5, 6)]
                if any(anchor is None for anchor in anchors):
                    continue
                residuals = [t - n / 48000 - bench.card.t0 for t, n in anchors]
                E = wait[6][0] - wait[6][1] / 48000 - bench.card.t0
                Tg = bench.card.t0 + (a - settle) / 48000
                O, G, C = wait[4] - Tg, key[3] - wait[4], up - key[3]
                D = bench.card.t_air(i0) - (bench.card.t0 + a / 48000)
                lead = bench.card.t_air(i0) - up
                cat = sum(e[4] - e[3] for e in events if e[0] == 'cat'
                          and key[3] <= e[3] <= e[4] <= up)
                error = lead - (settle / 48000 - O - G - C + D)
                stable = max(residuals) - min(residuals) <= 1e-8
                epoch_stable = all(e[7] == e[8] == bench.card.t0 for e in (arm, wait, key))
                coverage = _callback_coverage(self.callbacks[:self.nc], bench.card.t0,
                    bench.card.blocksize, bench.card.offset, i0, key[3], up)
                ordering = arm[4] <= wait[3] <= wait[4] <= key[3] <= up <= key[4]
                data['decompositions'].append(dict(a=a, settle_samples=settle,
                    wait_target=a-settle, Tg=Tg, R=wait[4], K=key[3], U=up,
                    O=O, G=G, C=C, D=D, E=E, wait_without_epoch=O-E,
                    cat_duration=cat, key_residual=C-cat, lead_air=lead,
                    identity_error=error, anchor_residuals=residuals, ordering_valid=ordering,
                    callback_coverage=coverage,
                    armed_minus_physical=a-(i0+bench.card.offset),
                    conclusive=stable and epoch_stable and abs(error) <= 1e-8
                               and not self.overflow and coverage and ordering
                               and all(c[6] == bench.card.t0 for c in self.callbacks[:self.nc])))
            data['conclusive'] = len(arms) == len(keys) == len(ups) == len(data['decompositions']) == 1 and all(
                d['conclusive'] for d in data['decompositions'])
        path.write_text(json.dumps(data, indent=2) + '\n')
