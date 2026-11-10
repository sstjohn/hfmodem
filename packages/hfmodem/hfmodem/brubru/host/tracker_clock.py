# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Private Tracker calendar, advancing on its owner's monotonic clock.

SCS 1.7 K supplies the date/time syntax. TheFirmware-derived presentation uses
US dates until a European date is selected. No command changes the system clock
or the AX.25 retransmission clock. The facade starts with host UTC.
"""
from datetime import datetime, timedelta, timezone
import re
import time


class TrackerClock:
    def __init__(self, clock):
        self.clock = clock
        self._base = datetime.fromtimestamp(time.time(), timezone.utc)
        self._at = clock()
        self.european = False

    def now(self):
        return self._base + timedelta(seconds=self.clock() - self._at)

    def format(self, stamp):
        date = stamp.strftime('%d.%m.%y' if self.european else '%m/%d/%y')
        return date + stamp.strftime(' %H:%M:%S')

    def set(self, argument):
        current = self.now()
        european = self.european
        date = re.fullmatch(r'(\d{2})([./])(\d{2})\2(\d{2})', argument, re.ASCII)
        clock = re.fullmatch(r'(\d{2}):(\d{2}):(\d{2})', argument, re.ASCII)
        if date:
            first, _, second, year = date.groups()
            european = date[2] == '.'
            day, month = (first, second) if european else (second, first)
            current = current.replace(year=2000 + int(year), month=int(month), day=int(day))
        elif clock:
            hour, minute, second = map(int, clock.groups())
            current = current.replace(hour=hour, minute=minute, second=second, microsecond=0)
        else:
            raise ValueError('K expects 0..2, DD.MM.YY, MM/DD/YY or HH:MM:SS')
        # Calendar validation happens before either the anchor or format changes.
        self._base, self._at, self.european = current, self.clock(), european
