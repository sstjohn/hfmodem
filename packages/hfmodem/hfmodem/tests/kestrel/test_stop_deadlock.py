# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A burst whose stream stop never returns, as KB2PCN-5's sixth request did on
2026-09-27: the arm sat keyed inside `Pa_StopStream` until the bridge watchdog
dropped PTT 25 s later, and then sat there for good."""
import atexit
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from . import corpora


class _Hal:
    """One device's IO context and one AUHAL unit, taken in opposite orders by
    the IO thread (PortAudio's IsRunning listener) and by `AudioOutputUnitStop`."""

    def __init__(self):
        self.context = threading.Lock()
        self.unit = threading.Lock()
        self.stopping = threading.Event()

    def io_thread(self, holding: threading.Event):
        with self.context:
            holding.set()
            self.stopping.wait()
            with self.unit:
                pass


class _Out:
    def __init__(self, hal):
        self.hal = hal

    def start(self):
        holding = threading.Event()
        threading.Thread(target=self.hal.io_thread, args=(holding,), daemon=True).start()
        holding.wait()

    def write(self, block):
        pass

    def stop(self):
        with self.hal.unit:
            self.hal.stopping.set()
            with self.hal.context:
                pass

    def close(self):
        pass


class _Rig:
    armed, retired = True, False

    def __init__(self):
        self.calls = []
        self.line = False

    def key(self, on):
        self.calls.append(on)
        self.line = on
        return True


def test_a_stop_that_never_returns_ends_the_arm_with_ptt_down(monkeypatch):
    kc = corpora.harness("kestrel_connect")
    budget = 0.2
    monkeypatch.setattr(kc, "STOP_BUDGET_S", budget, raising=False)
    monkeypatch.setattr(kc, "_wedged", None, raising=False)
    unregistered = []
    monkeypatch.setattr(atexit, "unregister", unregistered.append)
    hal = _Hal()
    fake = SimpleNamespace(
        query_devices=lambda _dev, _kind=None: {"max_output_channels": 2},
        InputStream=lambda **kw: SimpleNamespace(stop=lambda: None, close=lambda: None),
        OutputStream=lambda **kw: _Out(hal),
        _exit_handler=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    rig = _Rig()
    io = kc.AudioVaraIO("codec", "codec", rig=rig, tx_tail=0.0)
    outcome = []

    def arm():
        try:
            io.key(True)
            try:
                io.tx(np.zeros(4800, np.float32))
            finally:
                io.key(False)
        except BaseException as exc:
            outcome.append(exc)
        io.stop()

    started = time.monotonic()
    worker = threading.Thread(target=arm, daemon=True)
    worker.start()
    worker.join(budget + 5.0)

    assert not worker.is_alive(), "the arm is still inside the stream's stop()"
    assert time.monotonic() - started < budget + 2.0
    assert [type(e).__name__ for e in outcome] == ["AudioWedged"]
    assert rig.calls == [True, False] and rig.line is False
    assert unregistered == [fake._exit_handler]
