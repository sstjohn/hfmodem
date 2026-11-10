# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace

import pytest

from hfmodem.core import serialline


@pytest.fixture
def platform(monkeypatch):
    def switch(name):
        monkeypatch.setattr(serialline, 'os', SimpleNamespace(**{**vars(serialline.os), 'name': name}))
    return switch


@pytest.mark.parametrize('name', ['COM1', 'com9', 'COM41', r'\\.\COM41', r'\\.\com41'])
def test_windows_device_name(platform, name):
    platform('nt')
    assert serialline.device_name(name) == '\\\\.\\' + name.rsplit('\\', 1)[-1].upper()


@pytest.mark.parametrize('name', ['/dev/ttyUSB0', 'COMx', r'\\server\serial'])
def test_other_device_names(platform, name):
    platform('nt')
    assert serialline.device_name(name) == name


@pytest.mark.parametrize('name', ['COM41', r'\\.\COM41', '/dev/cu.fake'])
def test_posix_device_name(platform, name):
    platform('posix')
    assert serialline.device_name(name) == name


@pytest.mark.parametrize('a,b,equal', [('COM41', r'\\.\com41', True),
    ('com9', 'COM9', True), ('COM41', 'COM42', False)])
def test_windows_same_port(platform, a, b, equal):
    platform('nt')
    assert serialline.same_port(a, b) is equal


def test_posix_same_port(platform, monkeypatch):
    platform('posix')
    seen = []

    def realpath(path):
        seen.append(path)
        return {'alias': 'device', 'device': 'device', 'other': 'other'}[path]

    monkeypatch.setattr(serialline.os, 'path', SimpleNamespace(realpath=realpath))
    assert serialline.same_port('alias', 'device')
    assert not serialline.same_port('alias', 'other')
    assert seen == ['alias', 'device', 'alias', 'other']


@pytest.mark.parametrize('name', ['posix', 'nt'])
@pytest.mark.parametrize('options,stopbits', [({}, 1), ({'stopbits': 2}, 2)])
def test_open_order(platform, monkeypatch, name, options, stopbits):
    platform(name)
    from hfmodem.core import winproc
    monkeypatch.setattr(winproc, 'port_refusal', lambda *a, **k: pytest.fail('successful open looked up holders'))
    monkeypatch.setattr(winproc, 'record_port', lambda port: None)
    events = []

    class Port:
        def __setattr__(self, key, value):
            events.append((key, value))
            object.__setattr__(self, key, value)

        def open(self):
            events.append(('open', self.port))

        def fileno(self):
            return 17

        def close(self):
            events.append(('close',))

    def factory(**kwargs):
        events.append(('factory', kwargs))
        return Port()

    monkeypatch.setattr(serialline, 'fcntl', SimpleNamespace(ioctl=lambda *a: events.append(('ioctl', *a))))
    monkeypatch.setattr(serialline, 'termios', SimpleNamespace(TIOCEXCL=99))
    monkeypatch.setattr(serialline.os, 'set_blocking', lambda *a: events.append(('blocking', *a)), raising=False)
    ser = serialline.open_port(factory, 'COM41', baudrate=38400, rts=False, dtr=True,
                               write_timeout=.25, **options)
    port = r'\\.\COM41' if name == 'nt' else 'COM41'
    assert events == [('factory', dict(port=None, baudrate=38400, bytesize=8, parity='N',
        stopbits=stopbits, timeout=0, write_timeout=.25,
        **(dict(rtscts=False, dsrdtr=False) if name == 'nt' else {}))), ('rts', False), ('dtr', True),
        ('port', port), ('open', port)] + ([('ioctl', 17, 99), ('blocking', 17, False)] if name == 'posix' else [])
    assert ser.port == port


def test_exclusive_failure_closes(platform, monkeypatch):
    platform('posix')
    events = []
    ser = SimpleNamespace(open=lambda: events.append('open'), close=lambda: events.append('close'), fileno=lambda: 7)

    def refuse(*args):
        raise OSError('exclusive refused')

    monkeypatch.setattr(serialline, 'fcntl', SimpleNamespace(ioctl=refuse))
    monkeypatch.setattr(serialline, 'termios', SimpleNamespace(TIOCEXCL=99))
    monkeypatch.setattr(serialline.os, 'set_blocking', lambda *a: pytest.fail('set_blocking after ioctl failure'), raising=False)
    with pytest.raises(OSError, match='exclusive refused'):
        serialline.open_port(lambda **k: ser, 'fake', baudrate=38400, rts=False, dtr=False, write_timeout=.25)
    assert events == ['open', 'close']


class WindowsSerial:
    def __init__(self, raw=b'', *, timeout=0, write_timeout=.25):
        self.raw = raw
        self.events = []
        self._timeout, self._write_timeout = timeout, write_timeout
        self.error = None
        self.count = None

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self.events.append(('timeout', value))
        self._timeout = value

    @property
    def write_timeout(self):
        return self._write_timeout

    @write_timeout.setter
    def write_timeout(self, value):
        self.events.append(('write_timeout', value))
        self._write_timeout = value

    @property
    def in_waiting(self):
        return len(self.raw)

    def read(self, count):
        self.events.append(('read', count, self.timeout))
        if self.error:
            raise self.error
        result, self.raw = self.raw[:count], self.raw[count:]
        return result

    def write(self, raw):
        self.events.append(('write', raw))
        if self.error:
            raise self.error
        return len(raw) if self.count is None else self.count


def test_windows_read_drains_without_blocking(platform):
    platform('nt')
    ser = WindowsSerial(b'FA007100000;')
    assert serialline.read(ser, .1) == b'FA007100000;'
    assert ser.events == [('timeout', .1), ('read', 1, .1), ('timeout', 0), ('read', 11, 0)]


def test_windows_quiet_read_only_changes_timeout_once(platform):
    platform('nt')
    ser = WindowsSerial()
    assert serialline.read(ser, .1) == b''
    assert serialline.read(ser, .1) == b''
    assert ser.events == [('timeout', .1), ('read', 1, .1), ('read', 1, .1)]


def test_windows_nonblocking_drain_keeps_timeout(platform):
    platform('nt')
    ser = WindowsSerial(b'ID0650;')
    assert serialline.read(ser, 0) == b'ID0650;'
    assert ser.events == [('read', 1, 0), ('read', 6, 0)]


def test_windows_read_disconnect_propagates(platform):
    platform('nt')
    ser = WindowsSerial()
    ser.error = OSError('disconnected')
    with pytest.raises(OSError, match='disconnected'):
        serialline.read(ser, .1)


def test_windows_write_timeout_only_changes_when_needed(platform, monkeypatch):
    platform('nt')
    monkeypatch.setattr(serialline.time, 'monotonic', lambda: 10)
    ser = WindowsSerial()
    serialline.write(ser, b'AI0;', 10.25)
    serialline.write(ser, b'ID;', 10.5)
    serialline.write(ser, b'FA;', 10.5)
    assert ser.events == [('write', b'AI0;'), ('write_timeout', .5), ('write', b'ID;'), ('write', b'FA;')]


@pytest.mark.parametrize('remaining', [0, -1])
def test_windows_expired_write_does_not_write(platform, monkeypatch, remaining):
    platform('nt')
    monkeypatch.setattr(serialline.time, 'monotonic', lambda: 10)
    ser = WindowsSerial()
    with pytest.raises(TimeoutError):
        serialline.write(ser, b'ID;', 10 + remaining)
    assert not ser.events


@pytest.mark.parametrize('failure', ['short', 'timeout'])
def test_windows_write_failure(platform, monkeypatch, failure):
    platform('nt')
    from serial import SerialTimeoutException
    monkeypatch.setattr(serialline.time, 'monotonic', lambda: 10)
    ser = WindowsSerial()
    if failure == 'short':
        ser.count = 1
    else:
        ser.error = SerialTimeoutException('write timed out')
    with pytest.raises(OSError, match='short write|timed out'):
        serialline.write(ser, b'ID;', 10.25)


@pytest.mark.parametrize('readable', [True, False])
def test_posix_disconnect_or_quiet(platform, monkeypatch, readable):
    platform('posix')
    ser = SimpleNamespace(fileno=lambda: 17)
    monkeypatch.setattr(serialline.select, 'select', lambda *a: ([17] if readable else [], [], []))
    monkeypatch.setattr(serialline.os, 'read', lambda *a: b'')
    if readable:
        with pytest.raises(EOFError, match='disconnected'):
            serialline.read(ser, .1)
    else:
        assert serialline.read(ser, .1) == b''


def test_posix_write_retries_and_partial_writes(platform, monkeypatch):
    platform('posix')
    monkeypatch.setattr(serialline.time, 'monotonic', lambda: 10)
    calls = []
    counts = iter([BlockingIOError(), 1, 2])

    def write(fd, raw):
        calls.append((fd, raw))
        count = next(counts)
        if isinstance(count, Exception):
            raise count
        return count

    monkeypatch.setattr(serialline.select, 'select', lambda *a: ([], [17], []))
    monkeypatch.setattr(serialline.os, 'write', write)
    serialline.write(SimpleNamespace(fileno=lambda: 17), b'ID;', 10.25)
    assert calls == [(17, b'ID;'), (17, b'ID;'), (17, b'D;')]


@pytest.mark.parametrize('count', [0, -1])
def test_posix_write_no_progress(platform, monkeypatch, count):
    platform('posix')
    monkeypatch.setattr(serialline.time, 'monotonic', lambda: 10)
    monkeypatch.setattr(serialline.select, 'select', lambda *a: ([], [17], []))
    monkeypatch.setattr(serialline.os, 'write', lambda *a: count)
    with pytest.raises(OSError, match='made no progress'):
        serialline.write(SimpleNamespace(fileno=lambda: 17), b'ID;', 10.25)


def test_windows_cat_exchange_and_ptt_alias(platform, monkeypatch):
    from hfmodem.core import winproc
    monkeypatch.setattr(winproc, 'record_port', lambda port: None)
    import time
    from hfmodem.core.cat import YaesuCat, RigError
    from hfmodem.core.rigs import RIGS
    from hfmodem.tests.core.fakeyaesu import READBACKS

    platform('nt')
    with pytest.raises(RigError, match='PTT port'):
        YaesuCat('com41', {**RIGS['ft891'], 'ptt_port': r'\\.\COM41'}).open()

    class Peer(WindowsSerial):
        def __init__(self, **kwargs):
            super().__init__()
            self.__dict__.update(kwargs)
            self.replies = dict(READBACKS['ft891'])
            self.closed = True

        def open(self):
            assert not self.rts and not self.dtr
            assert self.port == r'\\.\COM41'
            self.closed = False

        def close(self):
            self.closed = True

        def read(self, count):
            if not self.raw:
                time.sleep(.001)
            return super().read(count)

        def write(self, raw):
            count = super().write(raw)
            command = raw.decode('ascii')
            if command in self.replies:
                self.raw += self.replies[command].encode('ascii')
            elif command.startswith('FA'):
                self.replies['FA;'] = command
            else:
                assert command == 'AI0;'
            return count

    driver = YaesuCat('COM41', serial_factory=Peer)
    ser = None
    try:
        driver.open()
        ser = driver._serial
        assert driver.identify() == RIGS['ft891']['name']
        assert driver.freq() == 7100000
        assert driver.mode().mode == 'PKTUSB'
        driver.set_freq(7102500)
        assert driver.freq() == 7102500
        assert not ser.closed
    finally:
        driver.close()
    assert ser is not None and ser.closed


@pytest.mark.parametrize('failure', ['short', 'timeout'])
def test_windows_cat_write_failure_poison(platform, failure):
    import time
    from serial import SerialTimeoutException
    from hfmodem.core.cat import YaesuCat, RigError

    platform('nt')
    ser = WindowsSerial()
    if failure == 'short':
        ser.count = 0
    else:
        ser.error = SerialTimeoutException('write timed out')
    driver = YaesuCat('COM41')
    driver._serial = ser
    with pytest.raises(RigError, match='CAT write failed'):
        driver._write('ID;', time.monotonic() + 1)
    assert driver.wedged and driver.poisoned


def test_serial_fake_serves_hamlib_and_unknown_queries(platform, monkeypatch):
    import time
    from hfmodem.tests.core.fakeyaesu import SerialFakeYaesu

    platform('nt')
    commands = ['AI;', 'ID;', 'FB;', 'EX0507;', 'IF;', 'OI;', 'PS;', 'ZZ;']

    class Port(WindowsSerial):
        def close(self):
            self.closed = True

        def read(self, count):
            if not self.raw:
                time.sleep(.001)
            return super().read(count)

    ser = Port(''.join(commands).encode('ascii'))
    ser.closed = False
    monkeypatch.setattr(serialline, 'open_port', lambda *a, **k: ser)
    peer = SerialFakeYaesu('COM42')
    try:
        peer.wait_for('ZZ;')
        deadline = time.monotonic() + 1
        while len([event for event in ser.events if event[0] == 'write']) < len(commands):
            assert time.monotonic() < deadline, ser.events
            time.sleep(.001)
        assert [event[1] for event in ser.events if event[0] == 'write'] == [
            b'AI0;', b'ID0650;', b'FB007100000;', b'EX05071;',
            b'IF001007100000+000000C00000;', b'OI001007100000+000000C00000;', b'PS1;', b'?;']
        assert peer.log == commands
        assert len(peer.replies['IF;']) == 28 and peer.replies['IF;'][22] == '0'
        assert len(peer.replies['OI;']) == 28 and peer.replies['OI;'][22] == '0'
        assert peer.error is None
    finally:
        peer.close()
    assert ser.closed


@pytest.mark.parametrize('exception_class', [OSError, PermissionError, 'SerialException'])
@pytest.mark.parametrize('wrapped', [False, True])
def test_windows_access_denied_names_foreign_opener(platform, monkeypatch, exception_class, wrapped):
    from hfmodem.core import winproc
    monkeypatch.setattr(winproc, 'port_refusal', lambda port, **k: f'{port} is open in another process')
    if exception_class == 'SerialException':
        from serial import SerialException
        exception_class = SerialException
    if wrapped:
        failure = exception_class(
            r"could not open port '\\.\COM41': PermissionError(13, 'Access is denied.', None, 5)")
    else:
        failure = exception_class('Access is denied.')
        failure.winerror = 5
    platform('nt')
    closed = []
    def refuse():
        raise failure
    ser = SimpleNamespace(open=refuse, close=lambda: closed.append(True))
    with pytest.raises(exception_class, match='^COM41 is open in another process$') as caught:
        serialline.open_port(lambda **k: ser, 'COM41', baudrate=38400,
                             rts=False, dtr=False, write_timeout=.25)
    assert type(caught.value) is exception_class
    assert caught.value.__cause__ is failure
    assert closed == [True]


@pytest.mark.parametrize('failure', [
    PermissionError(13, 'Access is denied.'),
    PermissionError(1, 'Operation not permitted'),
    OSError('Access is denied.'),
    OSError("could not open port: PermissionError(5, 'I/O error')"),
    FileNotFoundError(2, 'port not found'),
])
def test_windows_other_open_errors_are_unchanged(platform, failure):
    platform('nt')
    closed = []
    def refuse():
        raise failure
    ser = SimpleNamespace(open=refuse, close=lambda: closed.append(True))
    with pytest.raises(type(failure)) as caught:
        serialline.open_port(lambda **k: ser, 'COM41', baudrate=38400,
                             rts=False, dtr=False, write_timeout=.25)
    assert caught.value is failure
    assert closed == [True]


def test_posix_access_denied_is_unchanged(platform):
    platform('posix')
    failure = PermissionError(13, 'original refusal')
    def refuse():
        raise failure
    ser = SimpleNamespace(open=refuse, close=lambda: None)
    with pytest.raises(PermissionError) as caught:
        serialline.open_port(lambda **k: ser, 'port', baudrate=38400,
                             rts=False, dtr=False, write_timeout=.25)
    assert caught.value is failure


def test_refusal_lookup_receives_deadline_and_preserves_exception(platform, monkeypatch):
    from hfmodem.core import winproc
    platform('nt')
    monkeypatch.setattr(serialline.time, 'monotonic', lambda: 100)
    seen, closed = [], []
    def name(port, *, deadline):
        seen.append((port, deadline))
        return f'{port} is open in pid 5120 (modem.exe)'
    monkeypatch.setattr(winproc, 'port_refusal', name)
    failure = OSError('denied')
    failure.winerror = 5
    def refuse():
        raise failure
    ser = SimpleNamespace(open=refuse, close=lambda: closed.append(True))
    with pytest.raises(OSError, match=r'^COM41 is open in pid 5120 \(modem.exe\)$') as caught:
        serialline.open_port(lambda **k: ser, 'COM41', baudrate=9600,
                             rts=False, dtr=False, write_timeout=.25)
    assert caught.value.__cause__ is failure
    assert seen == [('COM41', 101)]
    assert closed == [True]


@pytest.mark.parametrize('result', [None, 'COM41 is held by your hfmodem station (pid 4120)',
                                   OSError('unavailable')])
def test_port_refusal_result_or_fallback(monkeypatch, result):
    import time
    from hfmodem.core import winproc
    def lookup(port, deadline):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(winproc, '_port_holder', lookup)
    assert winproc.port_refusal('COM41', deadline=time.monotonic() + 1) == (
        result if isinstance(result, str) else 'COM41 is open in another process')


@pytest.mark.wall_clock
@pytest.mark.parametrize('stage', ['live_launches', 'device_holders'])
def test_port_refusal_bounds_all_lookup_stages(monkeypatch, stage):
    import threading
    import time
    from hfmodem.core import winproc
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        entered.set()
        release.wait(5)
        return [], set()
    if stage == 'live_launches':
        from hfmodem.core import claim
        monkeypatch.setattr(claim, 'holder', lambda *a, **k: None)
        monkeypatch.setattr(winproc, 'live_launches', slow)
        lookup = winproc._port_holder
    else:
        monkeypatch.setattr(winproc, 'device_holders', slow)
        lookup = lambda port, deadline: winproc.device_holders([], {port}, deadline=deadline)
    def tracked(port, deadline):
        try:
            return lookup(port, deadline)
        finally:
            finished.set()
    monkeypatch.setattr(winproc, '_port_holder', tracked)
    start = time.monotonic()
    try:
        assert winproc.port_refusal('COM41', deadline=start + 1) == 'COM41 is open in another process'
        assert entered.is_set()
        assert .9 <= time.monotonic() - start < 1.5
    finally:
        release.set()
        assert finished.wait(5)


def test_expired_refusal_does_not_start_lookup(monkeypatch):
    import time
    from hfmodem.core import winproc
    monkeypatch.setattr(winproc, '_port_holder', lambda *a: pytest.fail('expired lookup started'))
    assert winproc.port_refusal('COM41', deadline=time.monotonic() - 1) == 'COM41 is open in another process'


@pytest.mark.parametrize('failure', [False, True])
def test_windows_open_records_until_close(platform, monkeypatch, failure):
    from hfmodem.core import winproc
    platform('nt')
    events = []
    def record(port):
        events.append(('held', port))
        return lambda: events.append(('released', port))
    monkeypatch.setattr(winproc, 'record_port', record)

    class Port:
        def open(self):
            events.append('open')

        def close(self):
            events.append('close')
            if failure:
                raise OSError('close failed')

    ser = serialline.open_port(lambda **k: Port(), 'COM41', baudrate=9600,
                             rts=False, dtr=False, write_timeout=.25)
    assert events == ['open', ('held', r'\\.\COM41')]
    if failure:
        with pytest.raises(OSError, match='close failed'):
            ser.close()
        assert events[-1] == 'close'
    else:
        ser.close()
        ser.close()
        assert events == ['open', ('held', r'\\.\COM41'), 'close',
                          ('released', r'\\.\COM41'), 'close']


def test_windows_evidence_failure_closes_and_reports_io_error(platform, monkeypatch):
    from hfmodem.core import winproc
    platform('nt')
    events = []
    def fail(port):
        raise winproc.ClaimError('record unavailable')
    monkeypatch.setattr(winproc, 'record_port', fail)
    ser = SimpleNamespace(open=lambda: events.append('open'), close=lambda: events.append('close'))
    with pytest.raises(OSError, match='cannot record COM41 holder: record unavailable'):
        serialline.open_port(lambda **k: ser, 'COM41', baudrate=9600,
                             rts=False, dtr=False, write_timeout=.25)
    assert events == ['open', 'close']


def test_nested_port_refusals_restore_after_exception():
    from hfmodem.core import winproc
    assert winproc._port_refusals.get() is None
    with winproc.port_refusals({'COM41': 'outer'}):
        with pytest.raises(ValueError):
            with winproc.port_refusals({'COM41': 'inner'}):
                assert winproc.port_refusal('COM41', deadline=0) == 'inner'
                raise ValueError
        assert winproc.port_refusal('COM41', deadline=0) == 'outer'
    assert winproc._port_refusals.get() is None
    assert winproc.port_refusal('COM41', deadline=0) == 'COM41 is open in another process'


def test_overlapping_port_refusals_are_local_and_restore():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from hfmodem.core import winproc
    a_entered, b_entered = threading.Event(), threading.Event()
    a_exit, a_exited, b_exited = threading.Event(), threading.Event(), threading.Event()
    def refuse():
        return winproc.port_refusal('COM41', deadline=0)
    def first():
        with winproc.port_refusals({'COM41': 'A'}):
            a_entered.set()
            assert b_entered.wait(5)
            during = refuse()
            assert a_exit.wait(5)
        a_exited.set()
        assert b_exited.wait(5)
        return during, refuse(), winproc._port_refusals.get()
    def second():
        assert a_entered.wait(5)
        with winproc.port_refusals({'COM41': 'B'}):
            b_entered.set()
            assert a_exited.wait(5)
            during = refuse()
        b_exited.set()
        return during, refuse(), winproc._port_refusals.get()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(first), pool.submit(second)
        try:
            assert b_entered.wait(5)
            assert refuse() == 'COM41 is open in another process'
        finally:
            a_exit.set()
        assert a.result(timeout=10) == ('A', 'COM41 is open in another process', None)
        assert b.result(timeout=10) == ('B', 'COM41 is open in another process', None)
    assert winproc._port_refusals.get() is None


@pytest.fixture
def port_evidence(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import json
    from hfmodem.core import claim, winproc
    own = winproc.Pinned(None, 42, 'win:123', 'python.exe')
    own.wait = lambda: False
    monkeypatch.setattr(winproc, '_ports', {})
    monkeypatch.setattr(winproc, '_own', lambda: nullcontext(own))
    monkeypatch.setattr(winproc, 'open', lambda pid: own)
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    monkeypatch.setattr(winproc, '_record_lock', lambda path, *, create=False: nullcontext(True))
    monkeypatch.setattr(winproc, '_prune_record', lambda path: path.unlink(missing_ok=True))
    def write(path, record):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record))
    monkeypatch.setattr(winproc, '_write_json', write)
    monkeypatch.setattr(winproc, '_read_json', lambda path:
        json.loads(path.read_text()) if path.exists() else None)
    return own, winproc._path('ports', own)


def test_port_evidence_never_accesses_claim(port_evidence, monkeypatch):
    from hfmodem.core import claim, winclaim, winproc
    own, path = port_evidence
    def forbidden(*a, **k):
        pytest.fail('port publication accessed the claim')
    for name in ('holder', 'inherited'):
        monkeypatch.setattr(claim, name, forbidden)
    for name in ('entries', 'publish', 'read_record', 'offset'):
        monkeypatch.setattr(winclaim, name, forbidden)
    first = winproc.record_port('COM41')
    second = winproc.record_port('COM42')
    assert path.parent.name == 'rig.claim.ports.d'
    assert list(winproc.recorded_devices([], None)) == [(own.identity, 'process', ['COM41', 'COM42'])]
    first()
    first()
    assert list(winproc.recorded_devices([], None)) == [(own.identity, 'process', ['COM42'])]
    second()
    assert list(winproc.recorded_devices([], None)) == [(own.identity, 'process', [])]


@pytest.mark.parametrize('condition', ['live', 'exited', 'reused', 'missing', 'denied'])
def test_port_reader_checks_identity_and_prunes(port_evidence, monkeypatch, condition):
    from hfmodem.core import claim, winproc
    own, path = port_evidence
    retire = winproc.record_port('COM41')
    publisher = claim.Holder(42, 'win:123', 456, 'station')
    if condition == 'exited':
        own.wait = lambda: True
    elif condition == 'reused':
        own.started = 'win:124'
    elif condition == 'missing':
        monkeypatch.setattr(winproc, 'open', lambda pid: None)
    elif condition == 'denied':
        def denied(pid):
            raise winproc.ProcessAccessError(pid, winproc.IDENTIFY)
        monkeypatch.setattr(winproc, 'open', denied)
    evidence = list(winproc.recorded_devices([], publisher))
    assert evidence == ([(own.identity, 'station', ['COM41'])] if condition == 'live' else [])
    assert path.exists() == (condition in ('live', 'denied'))
    retire()


def test_old_close_preserves_replacement_registration(platform, port_evidence):
    import threading
    from hfmodem.core import winproc
    platform('nt')
    own, path = port_evidence
    released, resume = threading.Event(), threading.Event()
    physical, errors = set(), []

    class Port:
        pause = False

        def open(self):
            assert self.port not in physical
            physical.add(self.port)

        def close(self):
            physical.discard(self.port)
            if self.pause:
                self.pause = False
                released.set()
                assert resume.wait(5)

    def open_port():
        return serialline.open_port(lambda **k: Port(), 'COM41', baudrate=9600,
                                    rts=False, dtr=False, write_timeout=.25)

    old = open_port()
    old.pause = True
    def close():
        try:
            old.close()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=close)
    replacement = None
    thread.start()
    try:
        assert released.wait(5)
        replacement = open_port()
        resume.set()
        thread.join(5)
        assert not thread.is_alive() and errors == []
        assert physical == {r'\\.\COM41'}
        assert winproc._read_json(path)['ports'] == [r'\\.\COM41']
        assert list(winproc.recorded_devices([], None)) == [(own.identity, 'process', [r'\\.\COM41'])]
    finally:
        resume.set()
        thread.join(5)
        if replacement is not None:
            replacement.close()
    assert not physical
    assert winproc._read_json(path)['ports'] == []
