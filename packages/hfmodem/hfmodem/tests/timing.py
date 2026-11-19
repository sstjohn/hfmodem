# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real-time bounds, raised as their own failure so qualification can tell a late
result on a slow host from a wrong one."""
import math


class TimingMiss(AssertionError):
    pass


def within(seconds, what, *, at_least=-math.inf, at_most=math.inf):
    if seconds > at_most:
        raise TimingMiss(f'{what}: {seconds:.4g} s, more than {at_most:.4g} s')
    if seconds < at_least:
        raise TimingMiss(f'{what}: {seconds:.4g} s, less than {at_least:.4g} s')
