# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Windows decisions with recording fakes; no serial handle is opened."""
from contextlib import nullcontext
import ctypes
import os
import sys
import time
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import serial

from hfmodem.core import cat, config, ptt, rig, serialline, windeadman, winproc
from hfmodem.core.regulatory import centred
from hfmodem.tests.core import test_rig

READ_OUTPUTS = serialline.read_outputs
rig_peer = test_rig.fake


@pytest.fixture
def windows(monkeypatch):
    platform = SimpleNamespace(**{**vars(os), 'name': 'nt', 'devnull': 'nul'})
    monkeypatch.setattr(winproc, 'record_port', lambda port: None)
    monkeypatch.setattr(ptt, 'os', platform)
    monkeypatch.setattr(serialline, 'os', platform)
    monkeypatch.setattr(windeadman, '_handles', ())
    monkeypatch.setattr(serialline, 'read_outputs', lambda ser, wait_s=.25: 0)
    monkeypatch.setattr(ptt, '_registered_com', lambda name: name.upper() == 'COM41')
    def drive(ser, line, on):
        assert any(handle == ser._port_handle for _, handle in windeadman._handles)
        setattr(ser, line, on)
    monkeypatch.setattr(serialline, 'drive', drive)


@pytest.fixture
def port(windows, monkeypatch):
    events = []

    class Port:
        _port_handle = 123
        fail_open = False
        fail_line = None
        fail_close = None

        def __init__(self, **kwargs):
            events.append(('factory', kwargs))

        def __setattr__(self, key, value):
            events.append((key, value))
            if key == self.fail_line:
                raise serial.SerialException('line failed')
            object.__setattr__(self, key, value)

        def open(self):
            events.append(('open', self.port))
            if self.fail_open:
                raise serial.SerialException('open failed')

        def close(self):
            registered = any(handle == self._port_handle for _, handle in windeadman._handles)
            assert registered is not self.fail_open
            events.append(('close',))
            if self.fail_close == 'before':
                raise OSError('close failed')
            object.__setattr__(self, '_port_handle', None)
            if self.fail_close == 'after':
                raise OSError('close failed')

    monkeypatch.setattr(serial, 'Serial', Port)
    return Port, events


def test_open_order_and_idempotence(port):
    _, events = port
    line = ptt.RtsPtt('COM41')
    line.open()
    line.open()
    assert events == [
        ('factory', dict(port=None, baudrate=9600, bytesize=8, parity='N', stopbits=1,
                         timeout=0, write_timeout=.25, rtscts=False, dsrdtr=False)),
        ('rts', False), ('dtr', False), ('port', r'\\.\COM41'),
        ('open', r'\\.\COM41'), ('rts', False), ('dtr', False)]
    assert line.sense() is False
    assert 'down' in repr(line)
    assert windeadman._handles == ((line._deadman_token, 123),)
    line.release()


@pytest.mark.parametrize('output,bit', [('rts', 2), ('dtr', 1)])
@pytest.mark.parametrize('state', range(4))
def test_sense_reads_configured_output(port, monkeypatch, output, bit, state):
    _, events = port
    monkeypatch.setattr(serialline, 'read_outputs', lambda ser, wait_s=.25: state)
    line = ptt.RtsPtt('COM41', line=output)
    assert line.sense() is None
    with pytest.raises(ptt.PttError, match='not open'):
        line.assert_(True)
    line.open()
    events.clear()
    line.assert_(True)
    assert line.sense() is bool(state & bit)
    assert ('asserted' if state & bit else 'down') in repr(line)
    line.assert_(False)
    assert events == [(output, True), (output, False)]
    line.release()
    assert line.released_low is (state == 0)
    assert line.sense() is None
    assert 'closed' in repr(line)


@pytest.mark.parametrize('failure', [None, 'rts', 'dtr'])
@pytest.mark.parametrize('close_failure', [None, 'before', 'after'])
def test_release_attempts_both_outputs_and_readback_before_close(port, monkeypatch,
                                                               failure, close_failure):
    factory, events = port
    def readback(ser, wait_s=.25):
        events.append(('readback',))
        assert windeadman._handles == ((token, ser._port_handle),)
        return 0
    monkeypatch.setattr(serialline, 'read_outputs', readback)
    line = ptt.RtsPtt('COM41')
    line.open()
    ser, token = line._serial, line._deadman_token
    factory.fail_line, factory.fail_close = failure, close_failure
    events.clear()
    line.release()
    line.release()
    assert events == [('rts', False), ('dtr', False), ('readback',), ('close',)]
    assert line.released_low is (failure is None)
    assert line.sense() is None
    retained = failure is not None and close_failure is not None
    assert windeadman._handles == (((token, 123),) if retained else ())
    assert ser._port_handle == (123 if close_failure == 'before' else None)
    with pytest.raises(ptt.PttError, match='not open'):
        line.assert_(True)


@pytest.mark.parametrize('stage', ['open', 'clear'])
@pytest.mark.parametrize('close_failure', [None, 'before', 'after'])
def test_failed_open_closes_without_masking_error(port, stage, close_failure):
    factory, events = port
    if stage == 'open':
        factory.fail_open = True
    else:
        original = factory.open
        def open_and_fail_clear(self):
            original(self)
            factory.fail_line = 'rts'
        factory.open = open_and_fail_clear
    factory.fail_close = close_failure
    line = ptt.RtsPtt('COM41')
    with pytest.raises(ptt.PttError, match='open failed|line failed'):
        line.open()
    assert events[-1] == ('close',)
    assert line._serial is None
    assert line._deadman_token is None
    retained = stage == 'clear' and close_failure is not None
    assert tuple(handle for _, handle in windeadman._handles) == ((123,) if retained else ())


def test_failed_open_cleanup_preserves_later_open_token(port, monkeypatch):
    factory, _ = port
    line = ptt.RtsPtt('COM41')
    later_ser = SimpleNamespace(_port_handle=456)
    close = factory.close

    def close_and_open_later(ser):
        close(ser)
        line._serial = later_ser
        line._deadman_token = windeadman.register_handle(later_ser._port_handle)

    monkeypatch.setattr(factory, 'close', close_and_open_later)
    monkeypatch.setattr(serialline, 'drive', Mock(side_effect=OSError('line failed')))
    with pytest.raises(ptt.PttError, match='line failed'):
        line.open()
    assert line._serial is later_ser
    assert line._deadman_token is not None
    assert windeadman._handles == ((line._deadman_token, 456),)


@pytest.mark.parametrize('state', [1, None])
def test_release_keeps_unconfirmed_port_when_close_raises(port, monkeypatch, state):
    factory, _ = port
    line = ptt.RtsPtt('COM41')
    line.open()
    ser, token = line._serial, line._deadman_token
    factory.fail_close = 'before'
    monkeypatch.setattr(serialline, 'read_outputs', lambda ser, wait_s=.25: state)
    line.release()
    assert line.released_low is False
    assert ser._port_handle == 123
    assert windeadman._handles == ((token, 123),)


def test_drive_error_is_ptt_error(port):
    factory, _ = port
    line = ptt.RtsPtt('COM41')
    line.open()
    factory.fail_line = 'rts'
    with pytest.raises(ptt.PttError, match='line failed'):
        line.assert_(True)
    start = time.monotonic()
    assert line.sense() is False
    assert time.monotonic() - start < .05
    line.release()


@pytest.mark.parametrize('name', ['COM41', 'com41', r'\\.\COM41', r'\\.\com41'])
def test_present_com_names_never_stat_or_open(windows, monkeypatch, name):
    monkeypatch.setattr(ptt.os, 'stat', lambda _: pytest.fail('COM ports cannot be statted'))
    monkeypatch.setattr(serial, 'Serial', lambda **_: pytest.fail('preflight opened a port'))
    ptt.require_char_device(name)
    assert ptt.arming_refusal(name) is None


@pytest.mark.parametrize('name', ['COM42', r'\\.\COM42', 'COMx', 'COM41junk', '/dev/ttyUSB0', 'file'])
def test_invalid_or_missing_com_names_refused(port, name):
    _, events = port
    with pytest.raises(ptt.PttError, match='PTT port'):
        ptt.RtsPtt(name).open()
    assert 'REFUSING TO ARM' in ptt.arming_refusal(name)
    assert events == []


def test_nul_still_uses_character_device_stat(windows, monkeypatch):
    import stat
    calls = []
    def stat_null(path):
        calls.append(path)
        return SimpleNamespace(st_mode=stat.S_IFCHR)
    monkeypatch.setattr(ptt.os, 'stat', stat_null)
    ptt.require_char_device('nul')
    assert calls == ['nul']


def test_a_com_port_the_namespace_lacks_is_refused(windows):
    with pytest.raises(ptt.PttError, match='PTT port COM42: COM port does not exist'):
        ptt.require_char_device('COM42')


@pytest.mark.parametrize('state', [0, 1, 2, 3, None])
def test_drop_reports_only_confirmed_low(port, monkeypatch, state):
    _, events = port
    monkeypatch.setattr(serialline, 'read_outputs', lambda ser, wait_s=.25: state)
    logs = []
    assert ptt.drop_rts('COM41', logs.append) is (state == 0)
    assert ('taken down directly' if state == 0 else 'NOT CONFIRMED LOW') in logs[0]
    assert events[-3:] == [('rts', False), ('dtr', False), ('close',)]


@pytest.fixture
def ioctl(monkeypatch):
    if sys.platform == 'win32':
        # The fake kernel below replaces ctypes.WinDLL and get_last_error process-wide,
        # which the stop channels a registered keying line installs would also meet.
        from hfmodem.core import winproc
        monkeypatch.setattr(winproc, 'install_channels', lambda *args, **kwargs: None)

    class Overlapped(ctypes.Structure):
        _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                    ('Offset', ctypes.c_uint32), ('OffsetHigh', ctypes.c_uint32),
                    ('hEvent', ctypes.c_void_p)]

    kernel = SimpleNamespace(state=0, count=4, error=0, wait=0, result=True)
    events = []
    def create(security, manual, initial, name):
        assert (security, manual, initial, name) == (None, True, False, None)
        events.append('event')
        return 456
    def control(handle, code, source, size, target, capacity, returned, overlapped):
        assert handle == 123 and code == 0x1B0078
        assert source is None and size == 0 and capacity == 4
        assert ctypes.cast(overlapped, ctypes.POINTER(Overlapped)).contents.hEvent == 456
        assert any(request.overlapped is overlapped._obj and request.state is target._obj
                   and request.returned is returned._obj and not request.abandoned
                   for request in serialline._pending_readbacks)
        kernel.buffer, kernel.overlapped = target._obj, overlapped._obj
        overlapped._obj.Internal = 0x103 if kernel.error == 997 else 0
        target._obj.value = kernel.state
        returned._obj.value = kernel.count
        events.append('ioctl')
        return not kernel.error
    def wait(event, milliseconds):
        assert event == 456 and 0 <= milliseconds <= 250
        events.append('wait')
        return kernel.wait
    def result(handle, overlapped, returned, wait):
        assert handle == 123 and wait is False
        assert any(request.overlapped is overlapped._obj and request.returned is returned._obj
                   and not request.abandoned
                   for request in serialline._pending_readbacks)
        returned._obj.value = kernel.count
        events.append('result')
        return kernel.result
    def cancel(handle, overlapped):
        assert handle == 123 and overlapped._obj is kernel.overlapped
        events.append('cancel')
        return False
    def close(event):
        assert event == 456
        events.append('close')
        return True
    for name, function in [('CreateEventW', create), ('DeviceIoControl', control),
                           ('WaitForSingleObject', wait), ('GetOverlappedResult', result),
                           ('CancelIoEx', cancel), ('CloseHandle', close)]:
        setattr(kernel, name, Mock(side_effect=function))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **kw: kernel, raising=False)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: kernel.error, raising=False)
    monkeypatch.setitem(sys.modules, 'serial.win32', SimpleNamespace(OVERLAPPED=Overlapped))
    monkeypatch.setattr(serialline, '_pending_readbacks', [])
    monkeypatch.setattr(serialline, 'read_outputs', READ_OUTPUTS)
    return kernel, events


@pytest.mark.parametrize('state', range(4))
@pytest.mark.parametrize('pending', [False, True])
def test_ioctl_reads_driver_mask(ioctl, pending, state):
    kernel, events = ioctl
    kernel.state, kernel.error = state, 997 if pending else 0
    ser = SimpleNamespace(_port_handle=123, _rts_state=not state & 2, _dtr_state=not state & 1)
    assert serialline.read_outputs(ser) == state
    assert events == ['event', 'ioctl', *(['wait'] if pending else []), 'result', 'close']
    assert not serialline._pending_readbacks
    assert kernel.CreateEventW.restype is ctypes.c_void_p
    assert kernel.DeviceIoControl.argtypes[0] is ctypes.c_void_p


@pytest.mark.parametrize('pending', [False, True])
def test_ioctl_completion_confirms_arm_release_and_drop(port, ioctl, pending):
    kernel, _ = ioctl
    kernel.error = 997 if pending else 0
    logs = []
    keyer = ptt.LineKeyer('COM41', logs.append)
    try:
        keyer.arm()
        assert keyer.last_drive.state is False and keyer.last_drive.state_read
    finally:
        keyer.hand_back()
    assert keyer.released_low is True
    assert ptt.drop_rts('COM41', logs.append) is True
    assert 'taken down directly' in logs[-1]


@pytest.mark.parametrize('fault', ['invalid', 'unsupported', 'failed', 'short', 'result',
                                  'timeout', 'wait_failed', 'incomplete'])
def test_ioctl_failure_refuses_sense_release_and_admission(port, ioctl, rig_peer, fault):
    from hfmodem.core.rig import RigError

    _, events = port
    kernel, io_events = ioctl
    if fault in ('invalid', 'unsupported', 'failed'):
        kernel.error = {'invalid': 1, 'unsupported': 50, 'failed': 31}[fault]
    elif fault == 'short':
        kernel.count = 3
    elif fault in ('result', 'incomplete'):
        kernel.error = 997 if fault == 'incomplete' else 0
        kernel.result = False
    else:
        kernel.error = 997
        kernel.wait = 258 if fault == 'timeout' else 0xffffffff
    line = ptt.RtsPtt('COM41')
    line.open()
    assert line.sense() is None
    assert repr(line) == "RtsPtt('COM41', unknown)"
    line.release()
    assert line.released_low is False
    logs = []
    assert ptt.drop_rts('COM41', logs.append) is False
    assert 'NOT CONFIRMED LOW' in logs[-1]
    keyer = ptt.LineKeyer('COM41', logs.append)
    with pytest.raises(ptt.PttError, match='not confirmed low'):
        keyer.arm()
    assert keyer.must_retire and keyer.released_low is False
    line.open()
    rig = test_rig.make(rig_peer, ptt=line, armed=False)
    try:
        with pytest.raises(RigError, match='not confirmed low'):
            rig._prove_ptt([])
        assert rig.line_state == 'unconfirmed'
        assert ('rts', True) not in events and ('dtr', True) not in events
    finally:
        rig.close()
    if fault in ('timeout', 'wait_failed'):
        assert io_events.count('cancel') == io_events.count('ioctl')
        assert len(serialline._pending_readbacks) == io_events.count('ioctl')
        request = serialline._pending_readbacks[-1]
        assert request.abandoned
        assert request.overlapped is kernel.overlapped and request.state is kernel.buffer
        assert ctypes.sizeof(request.state) == ctypes.sizeof(request.returned) == 4
    else:
        assert 'cancel' not in io_events and not serialline._pending_readbacks


@pytest.mark.parametrize('handle', [None, 0, -1])
def test_ioctl_without_handle_does_not_call_kernel(ioctl, handle):
    _, events = ioctl
    assert serialline.read_outputs(SimpleNamespace(_port_handle=handle)) is None
    assert serialline.read_outputs(object()) is None
    assert not events


def test_ioctl_event_creation_failure(ioctl):
    kernel, events = ioctl
    kernel.CreateEventW.side_effect = None
    kernel.CreateEventW.return_value = None
    assert serialline.read_outputs(SimpleNamespace(_port_handle=123)) is None
    assert not events


@pytest.mark.parametrize('completion_status', [0, 0xC0000120])
def test_ioctl_reaps_completed_timeouts_in_place(ioctl, completion_status):
    kernel, events = ioctl
    kernel.error, kernel.wait = 997, 258
    ser = SimpleNamespace(_port_handle=123)
    requests = serialline._pending_readbacks
    assert serialline.read_outputs(ser) is None
    completed = requests[0]
    assert serialline.read_outputs(ser) is None
    pending = requests[1]
    assert completed.abandoned and pending.abandoned
    assert completed.overlapped.Internal == pending.overlapped.Internal == 0x103
    completed.overlapped.Internal = completion_status
    kernel.error = 0
    events.clear()
    assert serialline.read_outputs(ser) == 0
    assert serialline._pending_readbacks is requests
    assert requests == [pending]
    assert events == ['event', 'ioctl', 'result', 'close']


@pytest.mark.parametrize('status', [0, 0xC0000120])
def test_ioctl_reap_keeps_live_request(ioctl, status):
    request = serialline._ReadbackRequest(SimpleNamespace(Internal=status),
                                         ctypes.c_uint32(), ctypes.c_uint32())
    serialline._pending_readbacks.append(request)
    assert not request.abandoned
    assert serialline.read_outputs(SimpleNamespace(_port_handle=123)) == 0
    assert serialline._pending_readbacks == [request]


def test_ioctl_reap_tolerates_concurrent_removal_and_keeps_append(ioctl):
    requests = serialline._pending_readbacks
    live = serialline._ReadbackRequest(SimpleNamespace(Internal=0),
                                      ctypes.c_uint32(), ctypes.c_uint32())

    class Completed:
        @property
        def Internal(self):
            requests.remove(abandoned)
            requests.append(live)
            return 0

    abandoned = serialline._ReadbackRequest(Completed(), ctypes.c_uint32(), ctypes.c_uint32())
    abandoned.abandoned = True
    requests.append(abandoned)
    assert serialline.read_outputs(SimpleNamespace(_port_handle=123)) == 0
    assert requests == [live]


@pytest.fixture
def inflight_readback(port, monkeypatch):
    pending = {}
    drive = serialline.drive

    def inflight_drive(ser, line, on):
        before = bool(getattr(ser, line))
        drive(ser, line, on)
        pending[ser, line] = [before, 3]

    def read_outputs(ser, wait_s=.25):
        state = int(ser.rts) * 2 | int(ser.dtr)
        # Clearing DTR must not complete an RTS drive still in flight.
        for line, bit in [('rts', 2), ('dtr', 1)]:
            old, remaining = pending.get((ser, line), (False, 0))
            if remaining:
                state = state | bit if old else state & ~bit
                pending[ser, line][1] -= 1
        return state

    monkeypatch.setattr(serialline, 'drive', inflight_drive)
    monkeypatch.setattr(serialline, 'read_outputs', read_outputs)


@pytest.mark.parametrize('output', ['rts', 'dtr'])
def test_inflight_sense_follows_each_command_without_redrive(inflight_readback, port, output):
    _, events = port
    line = ptt.RtsPtt('COM41', line=output)
    try:
        line.open()
        events.clear()
        line.assert_(True)
        assert line.sense() is True
        line.assert_(False)
        assert line.sense() is False
        assert events == [(output, True), (output, False)]
    finally:
        line.release()


@pytest.mark.parametrize('output', ['rts', 'dtr'])
def test_inflight_release_confirms_both_outputs_low(inflight_readback, port, output):
    _, events = port
    line = ptt.RtsPtt('COM41', line=output)
    line.open()
    line.assert_(True)
    events.clear()
    line.release()
    assert line.released_low is True
    assert events == [('rts', False), ('dtr', False), ('close',)]


def test_stuck_readback_waits_but_unreadable_returns_immediately(port, monkeypatch):
    line = ptt.RtsPtt('COM41')
    line.open()
    monkeypatch.setattr(serialline, 'read_outputs', lambda ser, wait_s=.25: 2)
    settle = getattr(ptt, '_READBACK_SETTLE_S', .2)
    try:
        start = time.monotonic()
        assert line.sense() is True
        assert settle <= time.monotonic() - start < .8
        start = time.monotonic()
        line.release()
        assert line.released_low is False
        assert settle <= time.monotonic() - start < .8
        unreadable = Mock(return_value=None)
        monkeypatch.setattr(serialline, 'read_outputs', unreadable)
        line.open()
        start = time.monotonic()
        assert line.sense() is None
        assert time.monotonic() - start < .05
        assert unreadable.call_count == 1
        start = time.monotonic()
        line.release()
        assert line.released_low is False
        assert time.monotonic() - start < .05
        assert unreadable.call_count == 2
    finally:
        line.release()


class MemoryCat:
    cache_free = True
    reads_ptt = False

    def open(self):
        pass

    def close(self):
        pass

    def identify(self):
        return 'Yaesu FT-891 (ID0650;)'

    def freq(self):
        return 7_105_000

    def ptt_state(self, **kwargs):
        return False

    def hold(self, deadline_index=None):
        return nullcontext(True)

    def clock(self):
        return time.monotonic() * 48000

    def protect(self, start, end):
        return object()

    def unprotect(self, token):
        pass

    def emergency_unkey(self):
        return None

    def sole_ptt_owner(self, notes):
        return True


#: The FT-891 keying acceptance station, less its ports and audio.
STATION = '''
schema = 1

[station]
mycall     = "W9SSJ"
grid       = "EN63bc"
control    = "local"
regulatory = "part97"
licence    = "extra"
transmit   = true
id_mode    = "none"

[rig]
model     = "ft891"
cat       = "yaesu"
mode      = "PKTUSB"
centre_hz = 7106500

[rig.profiles.ft891]
cat_rts = true
'''


def following(ser, wait_s=.25):
    return int(ser.rts) * 2 | int(ser.dtr)


def windows_rig(line, cat_driver, *, max_key_s):
    cfg = config.parse(tomllib.loads(STATION), environ={})
    return rig.Rig(model='ft891', cat=cat_driver, ptt=line, profile=cfg.profile,
                   control=cfg.control, mycall=cfg.station.mycall,
                   transmit=True, max_key_s=max_key_s)


@pytest.mark.parametrize('watchdog', [False, True])
def test_inflight_rig_unkey_is_confirmed(inflight_readback, capfd, watchdog):
    line = ptt.RtsPtt('COM41')
    made = windows_rig(line, MemoryCat(), max_key_s=.2)
    timer = None
    try:
        line.open()
        made.arm(prove_ptt=False)
        token = made.key(centred(7_105_000, 2300), why='in-flight drive', duration_s=1)
        timer = made._timer
        assert token.deadline == .2
        if watchdog:
            made._watchdog_fired('in-flight drive', token.deadline)
            assert made.panic_confirmed is True
            assert line.released_low is made.released_low is True
            text = capfd.readouterr().err
            assert 'keying line reads low' in text
            assert 'UNKEY NOT CONFIRMED' not in text
        else:
            made.unkey(token)
            assert not made.retired
            assert capfd.readouterr().err == ''
        assert not made.keyed and made.line_state == 'low'
    finally:
        made.close()
        if timer is not None:
            timer.join(1)
            assert not timer.is_alive()


def test_high_then_unreadable_after_unkey_takes_the_emergency_path(port, monkeypatch, capfd):
    line = ptt.RtsPtt('COM41')
    cat = MemoryCat()
    cat.emergency_unkey = Mock(return_value=None)
    made = windows_rig(line, cat, max_key_s=.2)
    timer = None
    try:
        line.open()
        made.arm(prove_ptt=False)
        monkeypatch.setattr(serialline, 'read_outputs', following)
        token = made.key(centred(7_105_000, 2300), why='stuck line', duration_s=1)
        timer = made._timer
        readings = iter([2, None])
        monkeypatch.setattr(serialline, 'read_outputs', lambda ser, wait_s=.25: next(readings, None))
        with pytest.raises(rig.StuckTransmitter):
            made.unkey(token)
        assert made.retired and line._serial is None
        assert line.released_low is False and made.line_state == 'unconfirmed'
        assert cat.emergency_unkey.called
        assert 'UNKEY NOT CONFIRMED' in capfd.readouterr().err
    finally:
        made.close()
        if timer is not None:
            timer.join(1)


def test_a_low_read_that_completes_after_the_bound_does_not_confirm(port, monkeypatch):
    line = ptt.RtsPtt('COM41')
    line.open()
    clock = [0.0]
    monkeypatch.setattr(ptt.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(ptt.time, 'sleep', lambda seconds: None)

    def read(ser, wait_s=.25):
        clock[0] += .15
        return 2 if clock[0] < .2 else 0

    monkeypatch.setattr(serialline, 'read_outputs', read)
    try:
        assert line.sense() is True
    finally:
        monkeypatch.setattr(serialline, 'read_outputs', lambda ser, wait_s=.25: 0)
        line.release()


def test_a_first_read_past_the_bound_confirms_nothing(port, monkeypatch):
    line = ptt.RtsPtt('COM41')
    line.open()
    clock = [0.0]
    waits = []
    monkeypatch.setattr(ptt.time, 'monotonic', lambda: clock[0])

    def read(ser, wait_s=.25):
        waits.append(wait_s)
        clock[0] += .21
        return 0

    monkeypatch.setattr(serialline, 'read_outputs', read)
    try:
        assert line.sense() is None
        assert waits == [pytest.approx(ptt._READBACK_SETTLE_S)]
    finally:
        monkeypatch.setattr(serialline, 'read_outputs', lambda ser, wait_s=.25: 0)
        line.release()


def test_a_late_first_high_read_after_unkey_takes_the_emergency_path(port, monkeypatch, capfd):
    line = ptt.RtsPtt('COM41')
    cat = MemoryCat()
    cat.emergency_unkey = Mock(return_value=None)
    made = windows_rig(line, cat, max_key_s=5)
    timer = None
    try:
        line.open()
        made.arm(prove_ptt=False)
        monkeypatch.setattr(serialline, 'read_outputs', following)
        token = made.key(centred(7_105_000, 2300), why='stuck line', duration_s=1)
        timer = made._timer
        late = iter([True])

        def read(ser, wait_s=.25):
            if next(late, False):
                time.sleep(ptt._READBACK_SETTLE_S + .01)
            return 2

        monkeypatch.setattr(serialline, 'read_outputs', read)
        with pytest.raises(rig.StuckTransmitter):
            made.unkey(token)
        assert made.retired and line._serial is None
        assert cat.emergency_unkey.called
    finally:
        made.close()
        if timer is not None:
            timer.join(1)


def test_inflight_line_transport_confirms_both_edges(inflight_readback):
    transport = ptt.LineTransport('COM41')
    try:
        transport.arm()
        for on in (True, False):
            result = transport.drive(on)
            assert result.verdict == 'confirmed'
            assert result.state is on and result.state_read
    finally:
        transport.release()


def test_inflight_k1_proof_keeps_real_yaesu_cat_readable(inflight_readback, monkeypatch):
    line = ptt.RtsPtt('COM41')
    commands = []

    class Ft891:
        _port_handle = 456

        def __init__(self, **kwargs):
            self.timeout = kwargs['timeout']
            self.write_timeout = kwargs['write_timeout']
            self.buffer = b''

        def open(self):
            pass

        def close(self):
            self._port_handle = None

        @property
        def in_waiting(self):
            return len(self.buffer)

        def read(self, size):
            if not self.buffer:
                time.sleep(self.timeout)
            result, self.buffer = self.buffer[:size], self.buffer[size:]
            return result

        def write(self, raw):
            commands.append(raw)
            assert raw.endswith(b';')
            for command in raw.split(b';')[:-1]:
                if command == b'ID':
                    self.buffer += b'ID0650;'
                elif command == b'FA':
                    self.buffer += b'FA007105000;'
                elif command == b'TX':
                    self.buffer += b'TX2;' if ptt_port.rts else b'TX0;'
                else:
                    assert command in (b'AI0', b'TX0')
            return len(raw)

    driver = cat.YaesuCat('COM42', serial_factory=Ft891)
    made = windows_rig(line, driver, max_key_s=1)
    original = made.unkey
    observed = []
    timers = []

    def unkey(*args, **kwargs):
        timers.append(made._timer)
        stamp = time.monotonic()
        result = original(*args, **kwargs)
        while time.monotonic() - stamp < 1:
            if made.cat.ptt_state(_arming=True) is False:
                observed.append(False)
                return result
            time.sleep(.001)
        pytest.fail('K1 CAT did not report TX0 within 1 s of unkey')

    monkeypatch.setattr(made, 'unkey', unkey)
    try:
        line.open()
        ptt_port = line._serial
        report = made.arm(prove_ptt=True)
        assert report.ptt_proven is True
        assert not made.retired and made.line_state == 'low'
        assert observed == [False] and not driver.poisoned
        assert commands[0] == b'AI0;'
        assert b'ID;' in commands and b'FA;' in commands and b'TX;' in commands
    finally:
        made.close()
        for timer in timers:
            if timer is not None:
                timer.join(1)
                assert not timer.is_alive()
