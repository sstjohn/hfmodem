# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One scoped clock for synchronous replays with explicitly charged work."""
from contextlib import contextmanager, ExitStack
import time
from unittest.mock import patch

from hfmodem.shrike import p4hold, p4longdecode, p4longnative, p4receiver, p4shortlevels, p4turn
from hfmodem.shrike.p4shortlevels import ShortDecode, QamShortDecode
from hfmodem.shrike.p4turn import P4Turn


@contextmanager
def declared_time(sample_now, *, fs=48000, elapsed_clock=None):
    """Preserve every deadline while charging only the declared sample clock.

    Enter before installing a turn factory. Constructor classes are retained
    here so the default clock bindings remain scoped even if a test replaces
    their module names. An elapsed-clock override is for explicit clock tests.
    No within-call CPU profile or physical performance is claimed.
    """
    now = lambda: sample_now()/fs
    turn_init = P4Turn.__init__
    short_init = ShortDecode.__init__
    qam_init = QamShortDecode.__init__

    def turn(self, *args, **kwargs):
        kwargs['wall_clock'] = now
        turn_init(self, *args, **kwargs)

    def short(self, *args, **kwargs):
        kwargs['now'] = now
        short_init(self, *args, **kwargs)

    def qam(self, *args, **kwargs):
        kwargs['now'] = now
        qam_init(self, *args, **kwargs)

    # Rust uses physical monotonic timestamps. The reference
    # engine supplies arithmetic under this synthetic grant; native timing is
    # independently tested by the physical wall_clock gate.
    with ExitStack() as stack:
        stack.enter_context(patch.object(p4longnative, 'backend', None))
        # These semantic replays charge no decoder CPU work. Physical Python
        # admission costs belong to the injected-cost and wall-clock tests.
        stack.enter_context(patch.object(p4longdecode, '_python_step_seconds', lambda *args: 0.))
        stack.enter_context(patch.object(p4longdecode, '_python_check_seconds', lambda _: 0.))
        stack.enter_context(patch.object(time, 'monotonic', now))
        stack.enter_context(patch.object(time, 'perf_counter',
                                         now if elapsed_clock is None else elapsed_clock))
        for module in (p4receiver, p4hold, p4shortlevels, p4turn):
            stack.enter_context(patch.object(module, 'monotonic', now))
        stack.enter_context(patch.object(P4Turn, '__init__', turn))
        stack.enter_context(patch.object(ShortDecode, '__init__', short))
        stack.enter_context(patch.object(QamShortDecode, '__init__', qam))
        yield
