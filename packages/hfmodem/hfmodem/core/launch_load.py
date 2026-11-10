# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A single startup observation; these values never grant timing permission."""
import math
from numbers import Integral, Real
import os
import sys


def advisory_line(*, load_sampler=None, cpu_sampler=None):
    """Sample the independent fields, returning only the fixed closed line."""
    load = "unavailable"
    try:
        sample = (load_sampler if load_sampler is not None else os.getloadavg)()
        if (isinstance(sample, (tuple, list)) and len(sample) == 3
                and all(isinstance(value, Real) and not isinstance(value, bool)
                        for value in sample)):
            values = tuple(float(value) for value in sample)
            if all(math.isfinite(value) and value >= 0 for value in values):
                load = "/".join(f"{value:.2f}" for value in values)
    except Exception:
        pass
    cpus = "unknown"
    try:
        sample = (cpu_sampler if cpu_sampler is not None else os.cpu_count)()
        if isinstance(sample, Integral) and not isinstance(sample, bool):
            value = int(sample)
            if value > 0:
                cpus = str(value)
    except Exception:
        pass
    return (f"LOAD ADVISORY: load average (1/5/15 min)={load}; "
            f"logical CPUs={cpus}; advisory only, no timing guarantee.")


def emit_load_advisory(*, load_sampler=None, cpu_sampler=None, stream=None):
    """Best-effort output. Ordinary sampling failures do not refuse startup."""
    line = advisory_line(load_sampler=load_sampler, cpu_sampler=cpu_sampler)
    target = sys.stdout if stream is None else stream
    try:
        target.write(line + "\n")
        target.flush()
    except OSError:
        pass
