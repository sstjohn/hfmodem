# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Passive CPU worker shutdown cannot hold the operator's quit behind decoding."""
import multiprocessing
import time
from types import SimpleNamespace

import pytest

from hfmodem.contact import slots as S


@pytest.mark.parametrize('exit_at', ['grace', 'terminate', 'kill', 'never'])
def test_close_bounds_every_wait_reaps_and_closes_queues(exit_at):
    calls = []
    state = SimpleNamespace(alive=True, signal='grace')
    def join(timeout):
        calls.append(('join', timeout))
        if state.signal == exit_at:
            state.alive = False
    def signal(name):
        calls.append((name,))
        state.signal = name
    worker = SimpleNamespace(join=join, is_alive=lambda: state.alive,
                             terminate=lambda: signal('terminate'), kill=lambda: signal('kill'))
    def channel(name):
        return SimpleNamespace(put_nowait=lambda item: calls.append((name, 'put', item)),
            cancel_join_thread=lambda: calls.append((name, 'cancel_join_thread')),
            close=lambda: calls.append((name, 'close')))
    watch = S.SlotWatch.__new__(S.SlotWatch)
    watch._process, watch._in, watch._out = worker, channel('in'), channel('out')
    if exit_at == 'never':
        with pytest.raises(RuntimeError, match='did not stop after kill'):
            watch.close()
    else:
        watch.close()
        assert not worker.is_alive()
        before = list(calls)
        watch.close()
        assert calls == before  # Already reaped; repeated close is harmless.
    assert sum(call[1] for call in calls if call[0] == 'join') <= .5
    assert ('in', 'close') in calls and ('out', 'close') in calls
    assert ('in', 'cancel_join_thread') in calls and ('out', 'cancel_join_thread') in calls
    assert (('terminate',) in calls) == (exit_at != 'grace')
    assert (('kill',) in calls) == (exit_at in ('kill', 'never'))


def test_full_queue_and_unresponsive_starting_worker_are_stopped_and_reaped():
    context = multiprocessing.get_context('spawn')
    watch = S.SlotWatch.__new__(S.SlotWatch)
    watch._in, watch._out = context.Queue(maxsize=1), context.Queue()
    watch._in.put('full')
    watch._process = context.Process(target=time.sleep, args=(60,), daemon=True)
    watch._process.start()
    real_join, waits = watch._process.join, []
    def observed_join(timeout):
        waits.append(timeout)
        real_join(timeout)
    watch._process.join = observed_join
    try:
        watch.close()
        assert not watch._process.is_alive() and watch._process.exitcode is not None
        # A structural wait budget, not a claim about OS scheduling latency.
        assert waits and sum(waits) <= .5
        watch.close()
    finally:
        if watch._process.is_alive():
            watch._process.kill()
            real_join(timeout=5)
        watch._process.close()
