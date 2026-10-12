# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Independent TCP streams do not acknowledge each other's processing order."""
import queue
import threading

import pytest

from creance.config import ModemConfig
from creance.conformance import PASS, ProbeContext, probe_disconnect_flush_order
from hfhost.client import ModemClient
from hfhost.transcript import Transcript


def test_disconnect_probe_waits_for_the_data_socket_enqueue(tmp_path):
    core_module = pytest.importorskip('hfmodem.kestrel.host.modem_core')
    server_module = pytest.importorskip('hfmodem.kestrel.host.server')
    events, delivered = [], []
    read, accept, closed = (threading.Event() for _ in range(3))

    class DelayedEnqueue(core_module.LoopbackModem):
        def start(self, observer):
            receive = observer.modem_data_received

            def record(blob):
                delivered.append(blob)
                receive(blob)

            observer.modem_data_received = record
            super().start(observer)

        def transmit(self, blob):
            events.append('data-read')
            read.set()
            assert accept.wait(3), 'probe never observed the data enqueue'
            events.append('enqueue')
            super().transmit(blob)

        def disconnect(self):
            events.append('disconnect')
            super().disconnect()

        def _finish_disconnect(self):
            super()._finish_disconnect()
            closed.set()

    class ObservedQueue(queue.Queue):
        def get(self, *args, **kwargs):
            if read.is_set():
                events.append('observe')
                accept.set()
            return super().get(*args, **kwargs)

    class Client(ModemClient):
        _in_command = False

        def subscribe(self, q=None):
            if q is None and not self._in_command:
                q = ObservedQueue()
            return super().subscribe(q)

        def send_data(self, blob, label=''):
            super().send_data(blob, label)
            assert read.wait(3), 'data socket was not read'

        def command(self, text, timeout=5):
            self._in_command = True
            try:
                reply = super().command(text, timeout)
            finally:
                self._in_command = False
            if text == 'DISCONNECT':
                # Force completion before the caller observes more events. The
                # old probe therefore closes an empty queue before accepting data.
                assert closed.wait(3), 'disconnect did not finish'
            return reply

    core = DelayedEnqueue()
    server = server_module.VaraServer(cmd_port=0, data_port=0, modem_factory=lambda: core)
    server.start_background()
    transcript = Transcript(str(tmp_path / 'wire.jsonl'), 'enqueue-order')
    client = Client(ModemConfig(name='loopback', cmd_port=server.cmd_port,
                               data_port=server.data_port), '127.0.0.1', transcript)
    try:
        client.attach()
        result = probe_disconnect_flush_order(ProbeContext(
            client, payload_n=64, connect_timeout_s=3, drain_timeout_s=3,
            reply_timeout_s=3, abort_timeout_s=3))
        assert result.verdict == PASS, result
        assert result.evidence == ('BUFFER 64 -> BUFFER 0 -> DISCONNECTED',)
        assert events.index('enqueue') < events.index('disconnect'), events
        assert sum(map(len, delivered)) == 64
    finally:
        accept.set()
        client.close()
        server.stop()
        transcript.close()
        if core._worker is not None:
            core._worker.join(3)
            assert not core._worker.is_alive()
