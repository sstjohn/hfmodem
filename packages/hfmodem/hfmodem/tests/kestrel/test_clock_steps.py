# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CAT retune waits keep their monotonic deadlines through a forward wall-clock step."""
import queue
import threading
import time
from types import SimpleNamespace

from hfmodem.kestrel.host.audio_runtime import RetuneRequest
from hfmodem.kestrel.host.station_runtime import StationVaraRuntime
from hfmodem.tests.clock_steps import SteppedLock


def test_a_forward_step_does_not_time_out_a_retune_cancellation():
    request = RetuneRequest(None, None, None, None, None, None)
    lock = threading.Lock()
    request._lock = SteppedLock(lock)
    lock.acquire()
    threading.Thread(target=lambda: (time.sleep(0.03), lock.release())).start()
    assert request.cancel(timeout=0.1) is True


def test_a_forward_step_does_not_time_out_a_running_retune(forward_step):
    runtime = StationVaraRuntime.__new__(StationVaraRuntime)
    runtime.transport = SimpleNamespace(stop=threading.Event())
    runtime._calls = queue.Queue()
    runtime._thread = threading.current_thread()

    def owner():
        call = runtime._calls.get()
        with call['lock']:
            time.sleep(0.1)
            call['result'] = 'tuned'
            call['done'].set()
    threading.Thread(target=owner, daemon=True).start()
    assert runtime.retune(lambda: None, lambda: True, timeout=10) == 'tuned'
