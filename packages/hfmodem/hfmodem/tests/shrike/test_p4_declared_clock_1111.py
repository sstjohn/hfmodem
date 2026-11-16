# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Declared replay clocks restore every global binding, including on errors."""
import time

import pytest

from hfmodem.shrike import p4hold, p4longdecode, p4longnative, p4receiver, p4shortlevels, p4turn
from hfmodem.tests.shrike.p4_declared_clock_1111 import declared_time


@pytest.mark.parametrize('raise_inside', [False, True])
@pytest.mark.parametrize('elapsed_override', [False, True])
def test_declared_clock_restores_bindings_and_backend_identity(monkeypatch, raise_inside, elapsed_override):
    native = object()
    monkeypatch.setattr(p4longnative, 'backend', native)
    modules = (p4receiver, p4hold, p4shortlevels, p4turn)
    classes = (p4turn.P4Turn, p4shortlevels.ShortDecode, p4shortlevels.QamShortDecode)

    def bindings():
        return (time.monotonic, time.perf_counter,
                *(module.monotonic for module in modules),
                *(cls.__init__ for cls in classes), p4longnative.backend,
                p4longdecode._python_step_seconds, p4longdecode._python_check_seconds)

    original = bindings()
    samples = [48000]
    def exercise():
        with declared_time(lambda: samples[0], elapsed_clock=(lambda: 0.) if elapsed_override else None):
            assert p4longnative.backend is None
            assert p4longdecode._python_step_seconds(p4longdecode.geometry(10), 3, 24) == 0.
            assert p4longdecode._python_check_seconds(p4longdecode.geometry(10)) == 0.
            turn = p4turn.P4Turn(None, None, role='irs')
            kwargs = dict(costs={}, events=[], at=0, status_ok=lambda status: True)
            short = p4shortlevels.ShortDecode([], **kwargs)
            qam = p4shortlevels.QamShortDecode([], **kwargs)
            hold = p4hold.BreakinHold(48000)
            for expected in (1., 2.):
                assert time.monotonic() == expected
                assert time.perf_counter() == (0. if elapsed_override else expected)
                assert all(module.monotonic() == expected for module in modules)
                assert turn._wall_clock() == short.now() == qam.now() == hold.now() == expected
                samples[0] += 48000
            if raise_inside:
                raise RuntimeError('declared replay failed')
    if raise_inside:
        with pytest.raises(RuntimeError, match='declared replay failed'):
            exercise()
    else:
        exercise()
    assert all(after is before for after, before in zip(bindings(), original))
    assert p4longnative.backend is native


def test_measured_replay_rejects_elapsed_clock_override():
    from hfmodem.tests.shrike.p4_chirp_deadline_support import replay
    with pytest.raises(ValueError, match='physical elapsed clock'):
        replay('unused', measured_work=True, elapsed_clock=lambda: 0.)
