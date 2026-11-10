# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The required SCS audio-only CW policy, on the native capture sample clock.

Only CWID 0 and CWID 4 2 are implemented. The latter requests identification
after either side closes and at approximately seven-minute intervals. An interval
request waits for our sending turn and defers DATA without confirming it. Actual
output completion, not the request, renews the interval.
"""
from hfmodem.core.identification import validate_call

INTERVAL_S = 420
CENTER_HZ = 1400


def parse(value):
    words = value.split()
    if len(words) in (1, 2) and words[0] == '0':
        if len(words) == 1:
            return '0 0'
        if words[1] in ('0', '1', '2', '3'):
            return ' '.join(words)
    if words == ['4', '2']:
        return '4 2'
    raise ValueError('supported CWID profiles are 0 and audio-only 4 2')


class Schedule:
    def __init__(self, call, clock, *, rate=48000):
        self.call = validate_call(call, 'PTC CWID MYCALL')
        self.clock = clock
        self.interval = INTERVAL_S * rate
        self.deadline = None
        self.waiting = False
        self.transmitted = False
        self.completed = 0

    def note_transmission(self, end):
        if self.deadline is None:
            self.deadline = end + self.interval
        self.transmitted = True

    @property
    def due(self):
        return self.deadline is not None and self.clock() >= self.deadline

    def defer_data(self):
        if not self.due:
            return False
        self.waiting = True
        return True

    def identified(self, end):
        self.completed += 1
        self.deadline = end + self.interval
        self.waiting = False
