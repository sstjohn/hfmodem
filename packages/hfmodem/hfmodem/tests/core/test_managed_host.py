# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Independent wire clients for supervised attachment ownership and retirement."""
import socket
import threading
import time

import pytest

from hfmodem.core.managed_host import ManagedHost

SECRET = bytes(range(32))


def wait(predicate):
    until = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < until
        time.sleep(.005)


def read(conn, count):
    data = bytearray()
    while len(data) < count:
        more = conn.recv(count-len(data))
        if not more:
            break
        data.extend(more)
    return bytes(data)


def header(role, lease=b'c'*16, attachment=0, *, secret=SECRET, magic=b'HFC1'):
    return magic + secret + bytes([role]) + lease + attachment.to_bytes(16, 'big')


class Client:
    def __init__(self, broker):
        self.broker = broker
        self.sockets = []

    def connect(self, blob):
        conn = socket.create_connection(('127.0.0.1', self.broker.port), timeout=1)
        self.sockets.append(conn)
        conn.sendall(blob)
        return conn

    def register(self):
        control = self.connect(header(0))
        response = read(control, 17)
        assert len(response) == 17 and response[0] == 0
        return control, response[1:]

    def stream(self, role, identity, sequence=1):
        conn = self.connect(header(role, identity, sequence))
        assert read(conn, 1) == b'\0'
        return conn

    def close(self):
        for conn in self.sockets:
            conn.close()


@pytest.fixture
def endpoint():
    entered = []
    def echo(cmd, data):
        entered.append((cmd, data))
        while blob := cmd.recv(8192):
            data.sendall(blob)
    broker = ManagedHost(echo, secret=SECRET, heartbeat_timeout=2.)
    broker.start()
    client = Client(broker)
    yield broker, client, entered
    client.close()
    broker.stop()


def test_data_first_pair_transfers_all_binary_values_without_protocol_prefixes(endpoint):
    broker, client, entered = endpoint
    control, lease = client.register()
    data = client.stream(2, lease)
    cmd = client.stream(1, lease)
    original = bytes(range(256))*2
    for at in range(0, len(original), 7):
        cmd.sendall(original[at:at+7])
    assert read(data, len(original)) == original
    assert len(entered) == 1
    assert control.send(b'\0') == 1


@pytest.mark.parametrize('blob', [header(0, secret=bytes(32)), header(3),
                                      header(0, magic=b'HFC2'), header(0, lease=bytes(16))])
def test_invalid_authentication_never_admits_an_attachment(endpoint, blob):
    broker, client, entered = endpoint
    conn = client.connect(blob)
    assert conn.recv(1) == b''
    assert broker._lease is None and not entered


def test_streams_require_the_server_issued_live_lease(endpoint):
    broker, client, entered = endpoint
    _, lease = client.register()
    assert lease != b'c'*16
    assert read(client.connect(header(1, b'c'*16, 1)), 1) == b'\1'
    assert read(client.connect(header(2, b'x'*16, 1)), 1) == b'\1'
    assert not entered


def test_control_loss_retires_both_streams_and_old_headers_cannot_enter_a_new_lease(endpoint):
    broker, client, entered = endpoint
    control, old = client.register()
    cmd, data = client.stream(1, old), client.stream(2, old)
    wait(lambda: len(entered) == 1)
    control.close()
    assert cmd.recv(1) == data.recv(1) == b''
    wait(lambda: broker._lease is None)
    _, new = client.register()
    assert new != old
    assert read(client.connect(header(1, old, 2)), 1) == b'\1'
    assert read(client.connect(header(2, old, 2)), 1) == b'\1'
    fresh_cmd, fresh_data = client.stream(1, new), client.stream(2, new)
    fresh_cmd.sendall(b'new application')
    assert read(fresh_data, 15) == b'new application'


def test_readmission_joins_the_previous_attachment_before_releasing_its_lease():
    entered, release = threading.Event(), threading.Event()
    def hold(cmd, data):
        entered.set()
        assert release.wait(3)
    broker = ManagedHost(hold, secret=SECRET)
    broker.start()
    client = Client(broker)
    try:
        control, lease = client.register()
        client.stream(1, lease)
        client.stream(2, lease)
        assert entered.wait(1)
        control.close()
        wait(lambda: broker._lease.retiring)
        assert read(client.connect(header(0)), 1) == b'\1'
        release.set()
        wait(lambda: broker._lease is None)
        client.register()
    finally:
        release.set()
        client.close()
        broker.stop()


def test_completed_attachment_counter_cannot_be_replayed_within_the_same_lease(endpoint):
    broker, client, entered = endpoint
    _, lease = client.register()
    cmd, data = client.stream(1, lease), client.stream(2, lease)
    wait(lambda: len(entered) == 1)
    cmd.close()
    assert data.recv(1) == b''
    wait(lambda: broker._lease.worker is not None and not broker._lease.worker.is_alive())
    assert read(client.connect(header(2, lease, 1)), 1) == b'\1'
    next_data, next_cmd = client.stream(2, lease, 2), client.stream(1, lease, 2)
    next_cmd.sendall(b'next')
    assert read(next_data, 4) == b'next'


def test_half_open_expiry_retires_the_counter_before_a_late_other_half():
    broker = ManagedHost(lambda *_: None, secret=SECRET, half_open_timeout=.05)
    broker.start()
    client = Client(broker)
    try:
        _, lease = client.register()
        half = client.stream(1, lease)
        assert half.recv(1) == b''
        assert read(client.connect(header(2, lease, 1)), 1) == b'\1'
        assert broker._lease.retired_sequence == 1
    finally:
        client.close()
        broker.stop()


def test_stalled_supervisor_loses_its_lease_and_pending_streams():
    broker = ManagedHost(lambda *_: None, secret=SECRET, heartbeat_timeout=.05)
    broker.start()
    client = Client(broker)
    try:
        control, lease = client.register()
        half = client.stream(1, lease)
        assert control.recv(1) == half.recv(1) == b''
        wait(lambda: broker._lease is None)
    finally:
        client.close()
        broker.stop()


def test_overlapping_half_pairs_fail_without_splicing_applications(endpoint):
    broker, client, entered = endpoint
    _, lease = client.register()
    old = client.stream(1, lease)
    assert read(client.connect(header(2, lease, 2)), 1) == b'\1'
    assert old.recv(1) == b''
    assert not entered
    data, cmd = client.stream(2, lease, 3), client.stream(1, lease, 3)
    cmd.sendall(b'new pair')
    assert read(data, 8) == b'new pair'


def test_authentication_readers_are_bounded_and_stop_joins_them(endpoint):
    broker, client, _ = endpoint
    for _ in range(8):
        client.connect(b'')
    wait(lambda: len(broker._workers) == 8)
    rejected = client.connect(b'')
    assert rejected.recv(1) == b''
    broker.stop()
    assert not broker._thread.is_alive()
    assert not any(worker.is_alive() for worker in broker._workers)


def test_non_loopback_and_wrong_secret_length_are_refused():
    with pytest.raises(ValueError, match='loopback'):
        ManagedHost(lambda *_: None, secret=SECRET, host='0.0.0.0')
    with pytest.raises(ValueError, match='32-byte'):
        ManagedHost(lambda *_: None, secret=b'short')


def test_serial_stream_has_one_role_and_joins_before_releasing_the_lease():
    entered, release = threading.Event(), threading.Event()
    def serial(stream):
        entered.set()
        try:
            while data := stream.recv(4096):
                stream.sendall(data)
        finally:
            assert release.wait(2)
    broker = ManagedHost(serial, secret=SECRET, stream_roles=(3,))
    broker.start()
    client = Client(broker)
    try:
        control, lease = client.register()
        # A TCP command/data role cannot enter a serial personality.
        assert client.connect(header(1, lease, 1)).recv(1) == b''
        stream = client.stream(3, lease)
        assert entered.wait(1)
        original = bytes(range(256))*2
        stream.sendall(original)
        assert read(stream, len(original)) == original
        control.close()
        assert stream.recv(1) == b''
        wait(lambda: broker._lease.retiring)
        assert read(client.connect(header(0)), 1) == b'\1'
        release.set()
        wait(lambda: broker._lease is None)
        _, new = client.register()
        assert new != lease
        assert read(client.connect(header(3, lease, 2)), 1) == b'\1'
        fresh = client.stream(3, new)
        fresh.sendall(b'fresh')
        assert read(fresh, 5) == b'fresh'
    finally:
        release.set()
        client.close()
        broker.stop()


@pytest.mark.parametrize('roles', [(), (1,), (2, 3), (1, 2, 3)])
def test_managed_stream_profiles_are_explicit(roles):
    with pytest.raises(ValueError, match='one serial stream'):
        ManagedHost(lambda *_: None, secret=SECRET, stream_roles=roles)


def test_backend_defect_closes_the_lease_and_prevents_replacement():
    def fail(*_):
        raise ValueError('backend defect')
    broker = ManagedHost(fail, secret=SECRET)
    broker.start()
    client = Client(broker)
    try:
        control, lease = client.register()
        client.stream(1, lease)
        client.stream(2, lease)
        assert control.recv(1) == b''
        assert broker.failure == 'managed attachment backend failed'
        # A failed broker closes an unauthenticated socket before consuming
        # its header. TCP may report either orderly EOF or a reset; neither
        # outcome may acknowledge or issue a replacement supervisor lease.
        try:
            refused = client.connect(header(0)).recv(1)
        except ConnectionResetError:
            refused = b''
        assert refused == b''
        assert broker.failure == 'managed attachment backend failed'
    finally:
        client.close()
        with pytest.raises(RuntimeError, match='backend failed'):
            broker.stop()
    assert broker._lease is None


@pytest.mark.parametrize('kind', ['vara', 'ardop'])
def test_station_managed_factory_binds_only_a_private_broker_and_reopens(kind, tmp_path):
    from types import SimpleNamespace
    from hfmodem.station.hosts import _build_one
    token = tmp_path/'token'
    token.write_text(SECRET.hex()+'\n')
    options = SimpleNamespace(host=f'{kind}-managed', cmd_port=0, data_port=0,
                              port=0, auth_token_file=str(token))
    served = _build_one(kind, options)
    assert not served.keys  # explicitly the no-radio host fixture
    broker = served.server
    broker.start()
    client = Client(broker)
    try:
        for attempt in range(2):
            control, lease = client.register()
            data, cmd = client.stream(2, lease), client.stream(1, lease)
            cmd.sendall(b'MYCALL W9SSJ\r')
            response = bytearray()
            wanted = b'OK\r' if kind == 'vara' else b'MYCALL now W9SSJ\r'
            while wanted not in response:
                response.extend(cmd.recv(4096))
            if kind == 'ardop' and attempt == 0:
                cmd.sendall(b'CLOSE\r')
                assert data.recv(1) == b''
            control.close()
            wait(lambda: broker._lease is None)
            assert data.recv(1) == b''
            assert not broker.failure
        assert len(served.ports) == 1
    finally:
        client.close()
        broker.stop()
    assert not broker._thread.is_alive()


def test_managed_configuration_requires_a_token_and_one_private_port():
    from pathlib import Path
    import tomllib
    from hfhost.config import ConfigError
    from hfmodem.core import config
    raw = tomllib.loads((Path(__file__).resolve().parents[5]/'examples/replay.toml').read_text())
    raw['protocols']['vara'].update(enabled=True, host='vara-managed')
    with pytest.raises(ConfigError, match='auth_token_file'):
        config.parse(raw)

    raw['protocols']['vara']['auth_token_file'] = '/unused/token'
    with pytest.raises(ConfigError, match='uses port'):
        config.parse(raw)
    raw['protocols']['vara'].update(cmd_port=0, data_port=0, port=18400)
    assert config.parse(raw).protocols['vara'].host == 'vara-managed'
    raw['protocols']['vara']['host'] = 'vara'
    with pytest.raises(ConfigError, match='requires a managed host'):
        config.parse(raw)


@pytest.mark.parametrize('family,host', [('pactor', 'ptc-managed'), ('rpr', 'tracker-managed')])
def test_managed_serial_config_exposes_only_an_authenticated_broker(family, host):
    from pathlib import Path
    import tomllib
    from hfhost.config import ConfigError
    from hfmodem.core import config
    raw = tomllib.loads((Path(__file__).resolve().parents[5]/'examples/replay.toml').read_text())
    raw['protocols'][family] = dict(enabled=True, host=host, port=18401,
                                     auth_token_file='/unused/token')
    assert config.parse(raw).protocols[family].host == host
    raw['protocols'][family]['pty'] = '/unused/operator'
    with pytest.raises(ConfigError, match='cannot expose a pty'):
        config.parse(raw)


def test_failed_vara_adapter_retirement_cannot_release_a_healthy_lease():
    from hfmodem.kestrel.host.modem_core import LoopbackModem
    from hfmodem.kestrel.host.server import HostSession
    class FailingStop(LoopbackModem):
        def stop(self):
            super().stop()
            raise RuntimeError('adapter failed retirement')
    def serve(command, data):
        session = HostSession(command, data, FailingStop())
        try:
            session.run()
        finally:
            session.close()
            session.ensure_retired()
    broker = ManagedHost(serve, secret=SECRET)
    broker.start()
    client = Client(broker)
    try:
        control, lease = client.register()
        command = client.stream(1, lease)
        client.stream(2, lease)
        command.close()
        assert control.recv(1) == b''
        wait(lambda: bool(broker.failure))
        rejected = client.connect(header(0))
        try:
            assert rejected.recv(1) == b''
        except ConnectionResetError:
            pass  # Closing a refused socket with unread auth can produce RST.
    finally:
        client.close()
        with pytest.raises(RuntimeError, match='backend failed'):
            broker.stop()


def test_delayed_authentication_cannot_replace_a_failed_backend(monkeypatch):
    entered, fail_backend = threading.Event(), threading.Event()
    gated, release_auth = threading.Event(), threading.Event()

    def backend(*_):
        entered.set()
        fail_backend.wait()
        raise ValueError('backend defect')

    broker = ManagedHost(backend, secret=SECRET)
    broker.start()
    client = Client(broker)
    try:
        control, lease = client.register()
        client.stream(1, lease)
        client.stream(2, lease)
        assert entered.wait(3)
        authenticate = broker._authenticate

        def delayed(conn):
            gated.set()
            release_auth.wait()
            return authenticate(conn)

        # The acceptor has checked failure=False before this worker is gated.
        monkeypatch.setattr(broker, '_authenticate', delayed)
        pending = client.connect(header(0))
        assert gated.wait(3)
        # Close from the peer so retirement does not depend on a server-side
        # shutdown interrupting its own blocked recv on every platform.
        old_lease = broker._lease
        control.close()
        wait(lambda: old_lease.retiring)
        fail_backend.set()
        wait(lambda: broker._lease is None)
        assert broker.failure == 'managed attachment backend failed'
        release_auth.set()
        try:
            response = pending.recv(17)
        except ConnectionResetError:
            response = b''
        assert response in (b'', b'\x01'), response
        assert broker._lease is None
        assert broker.failure == 'managed attachment backend failed'
    finally:
        fail_backend.set()
        release_auth.set()
        client.close()
        with pytest.raises(RuntimeError, match='backend failed'):
            broker.stop()


def test_delayed_stream_cannot_enter_a_failed_lease(monkeypatch):
    from hfmodem.core import managed_host

    entered, fail_backend = threading.Event(), threading.Event()
    gated, release_auth = threading.Event(), threading.Event()
    close_attempted = threading.Event()

    def backend(*_):
        entered.set()
        fail_backend.wait()
        raise ValueError('backend defect')

    broker = ManagedHost(backend, secret=SECRET)
    broker.start()
    client = Client(broker)
    hang_up = managed_host.hang_up
    try:
        _, identity = client.register()
        client.stream(1, identity)
        client.stream(2, identity)
        assert entered.wait(3)
        lease = broker._lease
        authenticate = broker._authenticate

        def delayed(conn):
            gated.set()
            release_auth.wait()
            return authenticate(conn)

        def defer_control_close(conn):
            if conn is lease.control:
                close_attempted.set()
                return
            return hang_up(conn)

        monkeypatch.setattr(broker, '_authenticate', delayed)
        monkeypatch.setattr(managed_host, 'hang_up', defer_control_close)
        pending = client.connect(header(1, identity, 2))
        assert gated.wait(3)
        fail_backend.set()
        assert close_attempted.wait(3)
        lease.worker.join(3)
        assert not lease.worker.is_alive() and lease.active is None
        assert broker._lease is lease and not lease.retiring
        assert broker.failure == 'managed attachment backend failed'
        release_auth.set()
        try:
            response = pending.recv(1)
        except ConnectionResetError:
            response = b''
        assert response in (b'', b'\x01'), response
        assert not lease.pending and lease.active is None
    finally:
        fail_backend.set()
        release_auth.set()
        monkeypatch.setattr(managed_host, 'hang_up', hang_up)
        client.close()
        with pytest.raises(RuntimeError, match='backend failed'):
            broker.stop()
