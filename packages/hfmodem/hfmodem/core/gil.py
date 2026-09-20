# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Hand the interpreter to the audio callback on a clock, while a decode runs.

WHY A COMPUTE-BOUND DECODE STARVES THE CALLBACK, WHICH IS NOT THE OBVIOUS REASON.
CPython only forces a thread off the GIL once a waiter has sat through a whole
switch interval *with the GIL never changing hands*. A thread running nothing but
bytecode satisfies that and is preempted inside ~7 ms. A thread that mostly runs
bytecode but drops the GIL now and then -- which is every numpy or scipy call over
a few hundred elements, and a decode makes a hundred thousand of them -- resets
that condition on every drop and rearms nothing, while the window it leaves open
is a microsecond wide and the waiter almost never wins it. Measured against a load
of one small `np.abs` per thousand bytecode instructions, a thread waiting on the
GIL is kept out for 357 ms at the median and 1.43 s at the worst, against 10.8 ms
for the same load with the numpy call taken out. It is the release that does the
damage, and shortening the switch interval cannot touch it.

SO THE FIX IS A REAL SLEEP. `time.sleep(0)` is another microsecond window and
measures no better than nothing (246 ms median). `time.sleep(50 us)` every 2 ms
takes the thread out of the race long enough that the waiter is certain to be
scheduled: the same load holds a waiter to 2.3 ms at the worst, and the sleeping
costs 2.5% of a duty cycle that the thread was going to lose to preemption anyway.

WHY `sys.monitoring` RATHER THAN A CALL IN THE LOOPS. The starvation is spread
over every long loop in the decoder, and a `breathe()` sprinkled through the DSP
covers the loops that were profiled on the day it was written and silently stops
covering anything added afterwards. PY_START fires on every Python call, which is
every loop the decoder has and every loop it will grow, for about 5% on the decode.

WHAT IT CANNOT REACH. PY_START is the only hook cheap enough to leave on, and a
kernel that calls no Python at all -- a bytecode loop over ufuncs and builtins --
raises none of it and goes unbreathed. That is not the decoder: it raises about
seven hundred thousand Python calls a second, and with this on, a thread waiting
on the GIL through one gets in within 2.1 ms at the worst over twelve seconds. It
is the thing to check first if a future kernel starves the callback anyway.

AND IT REACHES ONLY WHAT IT IS HELD OVER, which is the failure that actually
happened. Wrapped around one decode this covers one decode, and a session runs
more than its decoder: measured on a 1.25 s PACTOR listening cycle, the rolling
decoder's flush costs 40 ms and the readers the session calls straight through
cost another 105 -- the blind PACTOR-3 scan alone is 67-89 -- so the window this
was opened for held a quarter of the load, and the air went on losing 0.56-1.10%
of every listening cycle for four days after it landed. Attributing each lost
block to what was running when it went puts 23552 of 23808 in that one scan. So
it belongs to the CAPTURE and not to a decode: opened where the stream is opened,
on the thread that decodes, and held until the stream closes. It costs nothing
while that thread is blocked on the card, because a thread that raises no Python
calls raises no hook.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager

PERIOD = 0.002
"""How long a waiting callback may be kept out. One PortAudio block is 2.67 ms."""

NAP = 5e-5
"""Long enough that the waiter is scheduled, short enough to be free."""

_TOOL_IDS = (3, 4)
"""The ids `sys.monitoring` leaves unclaimed; 0, 1, 2 and 5 are spoken for."""

_lock = threading.Lock()
_due = 0.0
_owner = 0
_tool: int | None = None


def _on_call(_code, _offset) -> None:
    # PY_START is process-wide, and the audio callback is Python too. It has no
    # need to nap and every reason not to, so only the thread that opened the
    # window breathes in it.
    global _due
    if threading.get_ident() != _owner:
        return
    now = time.monotonic()
    if now >= _due:
        time.sleep(NAP)
        _due = time.monotonic() + PERIOD


def _claim() -> int | None:
    global _due, _owner, _tool
    for tool in _TOOL_IDS:
        try:
            sys.monitoring.use_tool_id(tool, "hfmodem.breathing")
        except ValueError:
            continue
        _tool, _owner = tool, threading.get_ident()
        _due = time.monotonic() + PERIOD
        sys.monitoring.register_callback(
            tool, sys.monitoring.events.PY_START, _on_call)
        sys.monitoring.set_events(tool, sys.monitoring.events.PY_START)
        return tool
    return None


@contextmanager
def breathing():
    """For the duration, let every other thread in at least every `PERIOD`.

    One window at a time, process-wide: a nested call, a second decoding thread,
    or a `sys.monitoring` slot already spoken for all decode without breathing
    rather than fighting over the hook. The station has one live receiver.
    """
    global _tool
    with _lock:
        mine = _tool is None and _claim() is not None
    try:
        yield
    finally:
        if mine:
            with _lock:
                sys.monitoring.set_events(_tool, 0)
                sys.monitoring.free_tool_id(_tool)
                _tool = None
