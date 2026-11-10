# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Actual Tracker owner behind the private serial supervisor protocol."""
import time
from types import SimpleNamespace

from hfmodem.core import hostmode as hm
from hfmodem.brubru.host.runtime import PacketRuntime
from hfmodem.tests.brubru.test_connected_hosts import Link, wait
from hfmodem.tests.core.test_managed_host import Client, SECRET, header, read
from hfmodem.station.hosts import _build_one


def test_managed_tracker_retains_its_lease_until_cancelled_rf_work_completes(tmp_path):
    link = Link()
    token = tmp_path/'token'
    token.write_text(SECRET.hex())
    cfg = SimpleNamespace(host='tracker-managed', port=0, auth_token_file=str(token), options={})
    station = SimpleNamespace(lanes={'rpr': link}, rig=object(), arbiter=object(),
                              cfg=SimpleNamespace(station=SimpleNamespace(mycall='W9SSJ')))
    served = _build_one('rpr', cfg, station)
    assert served.keys and served.endpoint_attrs == ()
    broker = served.server
    broker.start()
    runtime = link.packet_engine
    assert isinstance(runtime, PacketRuntime)
    client = Client(broker)
    try:
        control, lease = client.register()
        stream = client.stream(3, lease)
        stream.sendall(b'\x1bS1\r\x1bJHOST1\r')
        data = bytearray()
        while b'CHANNEL 1' not in data:
            data += stream.recv(4096)
        stream.sendall(hm.unwrap(hm.command(1, 'C N0CALL')))
        assert read(stream, 2) == b'\x01\x00'
        wait(lambda: bool(link.requests))
        old = runtime._attachment
        control.close()
        wait(lambda: link.requests[0].state.value == 'cancelled')
        assert not old.alive
        assert broker._lease is not None
        assert read(client.connect(header(0)), 1) == b'\1'
        # Cancellation is accepted, but the lane is still physically busy.
        time.sleep(.03)
        assert broker._lease is not None
        link.complete(okay=False)
        wait(lambda: broker._lease is None)
        assert not runtime.active and runtime._attachment is None
        _, fresh = client.register()
        new = client.stream(3, fresh)
        new.sendall(b'\x1bS1\r\x1bJHOST1\r')
        data = bytearray()
        while b'CHANNEL 1' not in data:
            data += new.recv(4096)
        new.sendall(hm.unwrap(hm.command(1, 'L')))
        decoder = hm.Decoder('master', crc=False)
        messages = []
        while not messages:
            messages.extend(decoder.feed(new.recv(4096)))
        assert messages[0].code == hm.MSG
        assert runtime._attachment is not old
        assert not broker.failure
    finally:
        if link.requests:
            link.complete(okay=False)
        client.close()
        broker.stop()
    assert link.host is link.packet_engine is None
