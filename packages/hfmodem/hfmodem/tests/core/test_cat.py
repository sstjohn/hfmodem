# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from contextlib import contextmanager
import errno
import os
import sys
import threading
import time
from unittest.mock import Mock

import pytest

from hfmodem.core import cat as module
from hfmodem.core.cat import RigctldCat, RigError, YaesuCat
from hfmodem.core.rigs import RIGS, resolve_profile, resolve_rig
from hfmodem.tests.core.fakerig import FakeRigctld
from hfmodem.tests.core.fakeyaesu import FakeYaesu, PtySerial


@contextmanager
def yaesu(model='ft891', **kwargs):
    fake = FakeYaesu(model, **kwargs)
    driver = YaesuCat(fake.port, model, serial_factory=PtySerial)
    try:
        driver.open()
        yield driver, fake
    finally:
        driver.close()
        fake.close()


@contextmanager
def rigctld(profile='ft891', **kwargs):
    fake = FakeRigctld(**kwargs)
    driver = RigctldCat(port=fake.port, profile=profile)
    try:
        driver.open()
        yield driver, fake
    finally:
        driver.close()
        fake.close()


def background(fn):
    result = []

    def run():
        try:
            result.append(fn())
        except Exception as exc:
            result.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def join(thread):
    thread.join(2)
    assert not thread.is_alive()


@pytest.mark.parametrize('model,width,roof', [('ft891', 3000, None), ('ftdx10', 4000, 12000)])
def test_native_profile_readbacks_and_setters(model, width, roof):
    with yaesu(model, chunk=1) as (driver, fake):
        assert driver.identify() == RIGS[model]['name']
        assert driver.sole_ptt_owner([])
        assert driver.freq() == 7100000
        mode, actual_roof = driver.receiver_filters()
        assert (mode.mode, mode.passband_hz, actual_roof) == ('PKTUSB', width, roof)
        assert driver.set_mode('PKTUSB', 2400)
        assert driver.mode().passband_hz == 2400
        driver.set_freq(7102500)
        assert driver.freq() == 7102500
        driver.set_power(25)
        assert driver.read('PC;', 'PC') == '025'
        assert fake.log[0] == 'AI0;'
        assert driver.cache_free and driver.reads_ptt


@pytest.mark.parametrize('model', ['ft891', 'ftdx10'])
@pytest.mark.parametrize('command,prefix', [('VS;', 'VS'), ('IS0;', 'IS0'), ('RT;', 'RT'),
    ('XT;', 'XT'), ('FT;', 'FT'), ('GT0;', 'GT0'), ('PA0;', 'PA0'), ('RA0;', 'RA0'),
    ('RG0;', 'RG0'), ('NR0;', 'NR0'), ('BC0;', 'BC0'), ('NB0;', 'NB0'), ('CO00;', 'CO00')])
def test_raw_readback_surface(model, command, prefix):
    with yaesu(model) as (driver, fake):
        assert driver.read(command, prefix) == fake.replies[command][len(prefix):-1]
        assert fake.log[-1] == command


@pytest.mark.parametrize('model,command,reply', [('ft891', 'EX0811;', 'EX0811045;'),
                                               ('ftdx10', 'EX010413;', 'EX010413050;')])
def test_menu_readbacks(model, command, reply):
    with yaesu(model) as (driver, fake):
        assert driver.read(command, command[:-1]) == reply[len(command) - 1:-1]


@pytest.mark.parametrize('reply', ['ID0000;', 'ID650;', 'ID0761;'])
def test_identity_is_exact(reply):
    with yaesu(replies={'ID;': reply}) as (driver, fake):
        with pytest.raises(RigError, match='identity'):
            driver.identify()


@pytest.mark.parametrize('model,replies', [
    ('ft891', {'SH0;': 'SH0099;'}), ('ft891', {'SH0;': 'SH0100;'}),
    ('ft891', {'MD0;': 'MD0Z;'}), ('ft891', {'VS;': 'VS2;'}),
    ('ft891', {'NA0;': 'NA02;'}), ('ft891', {'MD0;': 'MD02;'}),
    ('ftdx10', {'RF0;': 'RF08;'}), ('ftdx10', {'SH0;': 'SH0023;'})])
def test_unknown_readbacks_are_not_guessed(model, replies):
    with yaesu(model, replies=replies) as (driver, fake):
        with pytest.raises(RigError):
            driver.receiver_filters()


@pytest.mark.parametrize('model', ['ft891', 'ftdx10'])
def test_narrow_caps_width_and_vfo_b_is_read(model):
    with yaesu(model, replies={'NA0;': 'NA01;', 'VS;': 'VS1;', 'MD1;': 'MD18;'}) as (driver, fake):
        mode = driver.mode()
        assert (mode.mode, mode.passband_hz) == ('PKTLSB', 500)
        assert 'MD0;' not in fake.log


@pytest.mark.parametrize('model', ['ft891', 'ftdx10'])
@pytest.mark.parametrize('operation', [
    lambda driver: driver.set_mode('PKTUSB', 2400),
    lambda driver: driver.set_freq(7102500),
    lambda driver: driver.write('MD0C;'),
])
def test_vfo_b_selected_refuses_writes(model, operation):
    with yaesu(model, replies={'VS;': 'VS1;'}) as (driver, fake):
        with pytest.raises(RigError, match='VFO-B selected; CAT writes VFO-A only'):
            operation(driver)
        assert fake.log == ['AI0;', 'VS;']


@pytest.mark.parametrize('model', ['ft891', 'ftdx10'])
@pytest.mark.parametrize('command', ['MD1C;', 'FB007102500;', 'SH1016;',
                                     'VS1;', 'FT1;', 'TX1;', 'AI1;'])
def test_unsafe_raw_setters_are_never_sent(model, command):
    with yaesu(model) as (driver, fake):
        fake.wait_for('AI0;')
        with pytest.raises(RigError, match='CAT writes VFO-A only'):
            driver.write(command)
        assert fake.log == ['AI0;']


@pytest.mark.parametrize('method', ['exchange', 'read'])
@pytest.mark.parametrize('command,prefix', [
    ('MD1C;', 'MD1'), ('FB007102500;', 'FB'), ('SH1016;', 'SH1'),
    ('FA007102500;', 'FA'), ('VS1;', 'VS'), ('FT1;', 'FT'),
    ('TX1;', 'TX'), ('AI1;', 'AI'),
])
def test_public_queries_cannot_send_setters(method, command, prefix):
    with yaesu() as (driver, fake):
        fake.wait_for('AI0;')
        for reply_prefix in (prefix, command[:-1], None):
            with pytest.raises(RigError, match='expected a known CAT query'):
                getattr(driver, method)(command, reply_prefix)
        assert fake.log == ['AI0;']


@pytest.mark.parametrize('model', ['ft891', 'ftdx10'])
def test_each_native_setter_is_followed_by_its_matching_read(model):
    with yaesu(model) as (driver, fake):
        driver.set_mode('PKTUSB', 2400)
        driver.set_freq(7102500)
        driver.set_power(25)
        driver.write('RT0;')
        driver.write('XT0;')
        menu = 'EX0811' if model == 'ft891' else 'EX010413'
        driver.write(menu + '050;')
        width = 'SH0116;' if model == 'ft891' else 'SH0017;'
        assert fake.log == ['AI0;', 'VS;', 'MD0C;', 'MD0;',
                            'VS;', 'NA00;', 'NA0;', 'VS;', width, 'SH0;',
                            'VS;', 'FA007102500;', 'FA;',
                            'VS;', 'PC025;', 'PC;',
                            'VS;', 'RT0;', 'RT;', 'VS;', 'XT0;', 'XT;',
                            'VS;', menu + '050;', menu + ';']


@pytest.mark.parametrize('model', ['ft891', 'ftdx10'])
@pytest.mark.parametrize('command,query,operation', [
    ('FA007102500;', 'FA;', lambda driver: driver.set_freq(7102500)),
    ('MD0C;', 'MD0;', lambda driver: driver.set_mode('PKTUSB', 2400)),
    ('NA00;', 'NA0;', lambda driver: driver.set_mode('PKTUSB', 2400)),
    ('PC025;', 'PC;', lambda driver: driver.set_power(25)),
    ('RT0;', 'RT;', lambda driver: driver.write('RT0;')),
])
def test_setter_refusal_is_reported_to_setter_and_readback_is_drained(model, command, query, operation):
    with yaesu(model, replies={command: '?;'}, chunk=1) as (driver, fake):
        with pytest.raises(RigError, match='refusal after setter'):
            operation(driver)
        assert fake.log[-2:] == [command, query]
        assert not driver._owed and not driver.poisoned
        assert driver.identify() == RIGS[model]['name']


def test_coalesced_setter_refusal_and_readback_do_not_poison():
    with yaesu(replies={'PC025;': None, 'PC;': '?;PC005;'}) as (driver, fake):
        with pytest.raises(RigError, match='refusal after setter'):
            driver.set_power(25)
        assert not driver._owed and not driver.poisoned
        assert driver.identify() == RIGS['ft891']['name']


@pytest.mark.parametrize('model,command', [('ft891', 'SH0116;'), ('ftdx10', 'SH0017;')])
def test_width_setter_refusal_is_reported_and_drained(model, command):
    with yaesu(model, replies={command: '?;'}) as (driver, fake):
        with pytest.raises(RigError, match='refusal after setter'):
            driver.set_mode('PKTUSB', 2400)
        assert fake.log[-2:] == [command, 'SH0;']
        assert not driver._owed and not driver.poisoned
        assert driver.identify() == RIGS[model]['name']


@pytest.mark.parametrize('replies', [{'PC025;': '?;', 'PC;': None},
                                   {'PC025;': None, 'PC;': '?;'}])
def test_setter_refusal_with_missing_readback_keeps_tombstone(replies):
    with yaesu(replies=replies) as (driver, fake):
        with pytest.raises(RigError, match='refusal after setter'):
            driver._exchange('PC025;', 'PC', timeout=0.1, _read_after='PC;')
        assert driver.uncertain and list(driver._owed) == ['PC']
        assert driver.identify() == RIGS['ft891']['name']
        assert not driver._owed and driver.source == 'live'


def test_setter_guard_and_readback_do_not_interleave_with_another_query():
    entered, release = threading.Event(), threading.Event()

    def vfo(command):
        entered.set()
        assert release.wait(1)
        return 'VS0;'

    with yaesu(replies={'VS;': vfo}) as (driver, fake):
        setter, set_result = background(lambda: driver.set_power(25))
        try:
            assert entered.wait(1)
            reader, read_result = background(driver.identify)
        finally:
            release.set()
        join(setter)
        join(reader)
        assert set_result == [None] and read_result == [RIGS['ft891']['name']]
        assert fake.log == ['AI0;', 'VS;', 'PC025;', 'PC;', 'ID;']


@pytest.mark.parametrize('watts', [0, 4, 101, float('nan'), float('inf'), 5.5, True])
def test_invalid_native_power_is_not_sent(watts):
    with yaesu() as (driver, fake):
        with pytest.raises(RigError):
            driver.set_power(watts)
        assert not any(command.startswith('PC') for command in fake.log)


def test_invalid_width_does_not_partially_change_mode():
    with yaesu() as (driver, fake):
        with pytest.raises(RigError):
            driver.set_mode('PKTUSB', 4000)
        assert not any(command.startswith('MD') for command in fake.log)


def test_serial_configuration_and_single_thread(monkeypatch):
    fake = FakeYaesu()
    events = []

    class RecordingSerial(PtySerial):
        def open(self):
            events.append((self.rts, self.dtr, self.baudrate, self.bytesize,
                           self.parity, self.stopbits, threading.get_ident()))
            super().open()

    driver = YaesuCat(fake.port, {**RIGS['ft891'], 'cat_rts': True}, serial_factory=RecordingSerial)
    write, read, ioctl = module.os.write, module.os.read, module.fcntl.ioctl
    workers = set()
    exclusive = []

    def spy_ioctl(fd, request, *args):
        result = ioctl(fd, request, *args)
        if request == module.termios.TIOCEXCL:
            exclusive.append((fd, threading.get_ident()))
        return result

    def spy_write(fd, raw):
        if driver._serial is not None and fd == driver._serial.fileno():
            assert exclusive == [(fd, threading.get_ident())]
            workers.add(threading.get_ident())
        return write(fd, raw)

    def spy_read(fd, size):
        if driver._serial is not None and fd == driver._serial.fileno():
            workers.add(threading.get_ident())
        return read(fd, size)

    monkeypatch.setattr(module.os, 'write', spy_write)
    monkeypatch.setattr(module.os, 'read', spy_read)
    monkeypatch.setattr(module.fcntl, 'ioctl', spy_ioctl)
    try:
        driver.open()
        threads = [background(driver.freq) for _ in range(4)]
        for thread, result in threads:
            join(thread)
            assert result == [7100000]
        assert events[0][:6] == (True, False, 38400, 8, 'N', 1)
        assert workers == {events[0][-1]}
        assert threading.get_ident() not in workers
    finally:
        driver.close()
        fake.close()


@pytest.mark.parametrize('model', ['ft891', 'ftdx10'])
def test_open_without_factory_uses_pyserial(model):
    pytest.importorskip('serial', reason='optional production-path test requires pyserial')
    fake = FakeYaesu(model)
    driver = YaesuCat(fake.port, model)
    try:
        driver.open()
        assert driver.identify() == RIGS[model]['name']
        assert fake.log[:2] == ['AI0;', 'ID;']
        assert driver._serial.rts is False and driver._serial.dtr is False
    finally:
        driver.close()
        fake.close()


def test_idle_loop_error_is_separate_from_open_error():
    with yaesu() as (driver, fake):
        fake.send('VS0;')  # An unsolicited reply fails the idle parser.
        join(driver._thread)
        assert driver._open_error is None
        assert isinstance(driver._runtime_error, RigError)
        assert 'unsolicited' in str(driver._runtime_error)
        assert driver.poisoned and driver._serial is None
    assert driver._runtime_error is None


@pytest.mark.skipif(sys.platform == 'darwin', reason=(
    'macOS PTY ptsopen permits non-root opens despite TIOCEXCL; '
    'USB serial exclusion requires the hardware gate'))
@pytest.mark.skipif(os.geteuid() == 0, reason='privileged opens can bypass TIOCEXCL')
def test_serial_exclusive_open_rejects_later_open():
    with yaesu() as (driver, fake):
        with pytest.raises(OSError) as error:
            fd = os.open(fake.port, os.O_RDWR | os.O_NOCTTY)
            os.close(fd)
        assert error.value.errno == errno.EBUSY


def test_serial_exclusive_ioctl_failure_does_not_return_open(monkeypatch):
    fake = FakeYaesu()
    driver = YaesuCat(fake.port, serial_factory=PtySerial)
    opened = []

    def refuse(fd, request):
        assert request == module.termios.TIOCEXCL
        opened.append(driver._serial)
        raise OSError(errno.ENOTTY, 'exclusive ioctl refused')

    monkeypatch.setattr(module.fcntl, 'ioctl', refuse)
    try:
        with pytest.raises(RigError, match='exclusive ioctl refused'):
            driver.open()
        join(driver._thread)
        assert len(opened) == 1 and opened[0].fd is None
        assert driver.poisoned and not fake.log
        assert driver._open_error is not None and driver._runtime_error is None
    finally:
        driver.close()
        fake.close()


def test_never_opens_ptt_port():
    called = []
    driver = YaesuCat('/tmp/fake-ptt', {**RIGS['ft891'], 'ptt_port': '/tmp/fake-ptt'},
                      serial_factory=lambda **kwargs: called.append(kwargs))
    with pytest.raises(RigError, match='PTT port'):
        driver.open()
    assert not called


def test_timeout_keeps_tombstone_until_sentinel():
    with yaesu(replies={'TX;': None, 'ID;': (0.06, 'ID0650;')}) as (driver, fake):
        with pytest.raises(RigError, match='timed out'):
            driver.exchange('TX;', 'TX', timeout=0.03)
        fake.send('TX1;', delay=0.03)
        assert driver.source == 'uncertain'
        assert driver.read('VS;', 'VS') == '0'
        assert fake.log[-2:] == ['ID;', 'VS;']
        assert driver.source == 'live'
        assert not driver._owed
        fake.replies['TX;'] = 'TX0;'
        assert driver.emergency_unkey() is False
        assert driver.source == 'poisoned'


def test_sentinel_proves_dropped_reply_without_silence_assumption():
    with yaesu(replies={'FA;': None}) as (driver, fake):
        with pytest.raises(RigError):
            driver.exchange('FA;', 'FA', timeout=0.02)
        assert driver.read('VS;', 'VS') == '0'
        assert fake.log[-2:] == ['ID;', 'VS;']


def test_no_available_sentinel_stays_uncertain():
    with yaesu(replies={'ID;': None, 'VS;': None, 'FA;': None}) as (driver, fake):
        for _ in range(3):
            with pytest.raises(RigError):
                driver.exchange('FA;', 'FA', timeout=0.03)
        assert set(driver._owed) == {'ID', 'VS', 'FA'}
        before = len(fake.log)
        with pytest.raises(RigError, match='no distinct'):
            driver.read('TX;', 'TX')
        assert len(fake.log) == before
        assert driver.source == 'uncertain'


@pytest.mark.parametrize('reply', ['VS0;', 'FA007100000;FA007100001;'])
def test_desync_or_duplicate_poison(reply):
    with yaesu(replies={'FA;': reply}) as (driver, fake):
        with pytest.raises(RigError):
            driver.freq()
        assert driver.poisoned
        with pytest.raises(RigError, match='poisoned'):
            driver.identify()


def test_refusal_is_head_reply_and_does_not_poison():
    with yaesu(replies={'FA;': '?;'}) as (driver, fake):
        with pytest.raises(RigError, match='refused'):
            driver.freq()
        assert driver.identify() == RIGS['ft891']['name']


@pytest.mark.parametrize('timed_out', [False, True])
# The fake wire replies after 150ms; unkey must beat that real deadline.
@pytest.mark.wall_clock
def test_panic_does_not_consume_earlier_tx_reply(timed_out):
    with yaesu(replies={'TX;': None}) as (driver, fake):
        thread, result = background(lambda: driver.exchange('TX;', 'TX', timeout=0.04 if timed_out else 1))
        fake.wait_for('TX;')
        if timed_out:
            join(thread)
        fake.send('TX0;', delay=0.15)
        started = time.monotonic()
        assert driver.emergency_unkey() is None
        assert time.monotonic() - started < 0.15
        join(thread)
        assert isinstance(result[0], RigError)
        assert 'TX0;' not in fake.log
        assert driver.source == 'poisoned'


@pytest.mark.parametrize('reply,expected', [('TX0;', False), ('TX1;', True), ('TX2;', True),
                                            ('TX9;', None), ('VS0;', None), (None, None)])
@pytest.mark.wall_clock
def test_emergency_evidence_and_absolute_bound(reply, expected):
    with yaesu(replies={'TX;': reply}) as (driver, fake):
        start = time.monotonic()
        assert driver.emergency_unkey() is expected
        assert time.monotonic() - start < 0.55
        fake.wait_for('TX;')
        assert fake.log[-2:] == ['TX0;', 'TX;']
        with pytest.raises(RigError, match='poisoned'):
            driver.freq()


def test_panic_releases_queued_callers_and_bypasses_hold():
    with yaesu() as (driver, fake):
        with driver.hold() as held:
            assert held
            thread, result = background(driver.freq)
            assert driver.emergency_unkey() is False
            join(thread)
            assert isinstance(result[0], RigError)
        assert 'FA;' not in fake.log


def test_partial_write_wedges_and_emergency_sends_nothing(monkeypatch):
    with yaesu() as (driver, fake):
        original = module.os.write
        count = 0

        def partial(fd, raw):
            nonlocal count
            if fd == driver._serial.fileno():
                count += 1
                if count == 1:
                    return original(fd, raw[:2])
                raise BlockingIOError()
            return original(fd, raw)

        monkeypatch.setattr(module.os, 'write', partial)
        with pytest.raises(RigError, match='write'):
            driver.read('FA;', 'FA')
        assert driver.wedged and driver.poisoned
        assert driver.emergency_unkey() is None
        assert 'TX0;' not in fake.log


@pytest.mark.wall_clock
def test_panic_waiting_for_boundary_has_one_deadline(monkeypatch):
    with yaesu() as (driver, fake):
        original = driver._write
        entered, release = threading.Event(), threading.Event()

        def stalled(command, deadline):
            if command == 'PC005;':
                entered.set()
                release.wait(2)
            return original(command, deadline)

        monkeypatch.setattr(driver, '_write', stalled)
        thread, result = background(lambda: driver.write('PC005;'))
        assert entered.wait(1)
        started = time.monotonic()
        assert driver.emergency_unkey() is None
        assert time.monotonic() - started < 0.55
        release.set()
        join(thread)
        assert isinstance(result[0], RigError)
        assert 'TX0;' not in fake.log


@pytest.mark.parametrize('backend', ['yaesu', 'rigctld'])
def test_protection_and_nested_holds(backend):
    context = yaesu() if backend == 'yaesu' else rigctld()
    with context as (driver, fake):
        now = [100]
        driver.clock = lambda: now[0]
        token = driver.protect(100, None)
        with pytest.raises(RigError, match='protected'):
            driver.freq()
        driver.unprotect(token)
        driver.protect(100, 200)
        with pytest.raises(RigError, match='protected'):
            driver.freq()
        now[0] = 200
        assert driver.freq() == 7100000
        with driver.hold(100) as held:
            assert not held
        with driver.hold() as outer:
            with driver.hold(300) as inner:
                assert outer and inner
                assert driver.freq() == 7100000
        driver.keyed_probe = lambda: True
        with pytest.raises(RigError, match='protected'):
            driver.freq()
        assert driver.ptt_state(_arming=True) is False


def test_queued_exchange_rechecks_protection_at_execution():
    with yaesu() as (driver, fake):
        driver.clock = lambda: 100
        with driver.hold():
            thread, result = background(driver.freq)
            driver.protect(100, 100000)
        join(thread)
        assert isinstance(result[0], RigError)
        assert 'FA;' not in fake.log


# Refusal must finish within 50ms while a 150ms wire exchange is pending.
@pytest.mark.wall_clock
def test_hold_refuses_slot_before_active_exchange_deadline():
    with yaesu(replies={'ID;': (0.15, 'ID0650;')}) as (driver, fake):
        thread, result = background(driver.identify)
        fake.wait_for('ID;')
        started = time.monotonic()
        with driver.hold(driver.clock() + 480) as held:
            assert not held
        assert time.monotonic() - started < 0.05
        with driver.hold(driver.clock() + 96000) as held:
            assert held
        join(thread)
        assert result == [RIGS['ft891']['name']]


def test_rigctld_framing_and_readbacks():
    with rigctld(chunk=7, ptt_type='None') as (driver, fake):
        assert driver.identify() == 'Yaesu FT-891'
        assert driver.sole_ptt_owner([])
        assert driver.freq() == 7100000
        assert driver.mode().passband_hz == 1700
        assert driver.receiver_filters()[1] is None
        assert driver.set_mode('PKTUSB', 2400)
        assert driver.ptt_state() is None and not driver.reads_ptt
        assert not fake.cache and driver.cache_free
        assert all(command.startswith('+') for command in fake.log)


@pytest.mark.parametrize('owner,expected', [('None', True), ('', False), ('RTS', 'raises')])
def test_rigctld_owner_is_required(owner, expected):
    with rigctld(ptt_type=owner) as (driver, fake):
        notes = []
        if expected == 'raises':
            with pytest.raises(RigError, match='key the radio'):
                driver.sole_ptt_owner(notes)
        else:
            assert driver.sole_ptt_owner(notes) is expected
            assert bool(notes) is (not expected)


@pytest.mark.parametrize('value', ['', '   ', '\t'])
def test_empty_ptt_owner_is_unverified_even_when_field_is_present(monkeypatch, value):
    driver = RigctldCat()
    monkeypatch.setattr(driver, 'get_conf', lambda *args, **kwargs:
                        f'get_conf: ptt_type\nptt_type={value}\nRPRT 0')
    notes = []
    assert driver.sole_ptt_owner(notes) is False
    assert notes == ['rigctld did not name its PTT owner']


def test_get_conf_hangup_cannot_destroy_persistent_connection(monkeypatch):
    with rigctld(ptt_type='None') as (driver, fake):
        monkeypatch.setattr(fake, '_hangs_up', lambda line: 'get_conf' in line)
        persistent = driver._sock
        assert driver.sole_ptt_owner([])
        assert driver._sock is persistent
        assert driver.freq() == 7100000


def test_old_daemon_cache_refusal_is_not_evidence():
    with rigctld(no_cache=True) as (driver, fake):
        assert not driver.cache_free
        assert driver.freq() == 7100000
        fake.ptt = True
        assert driver.emergency_unkey() is None
        assert fake.log[-2:] == ['+\\set_cache 0', '+T 0']
        assert fake.ptt is False


@pytest.mark.parametrize('cache_accepted', [False, True])
def test_emergency_unkey_always_sends_t0_but_requires_cache_evidence(monkeypatch, cache_accepted):
    sock = Mock()
    sock.__enter__ = Mock(return_value=sock)
    sock.__exit__ = Mock(return_value=False)
    reply = (b'set_cache: 0\nRPRT 0\n' if cache_accepted else
             b'set_cache: 0\nRPRT -1\n')
    sock.recv.side_effect = [reply, b'set_ptt: 0\nRPRT 0\n',
                             b'get_ptt:\nPTT: 0\nRPRT 0\n']
    driver = RigctldCat()
    monkeypatch.setattr(driver, '_connect', lambda deadline: sock)
    assert driver.emergency_unkey() is (False if cache_accepted else None)
    sent = [call.args[0] for call in sock.sendall.call_args_list]
    assert sent[:2] == [b'+\\set_cache 0\n', b'+T 0\n']
    assert sent[2:] == ([b'+t\n'] if cache_accepted else [])
    assert driver.poisoned


@pytest.mark.parametrize('kwargs', [{'no_extended': True}, {'no_terminator': True}])
def test_bad_setup_does_not_return_open(kwargs):
    fake = FakeRigctld(**kwargs)
    driver = RigctldCat(port=fake.port)
    try:
        with pytest.raises(RigError):
            driver.open()
        assert driver._sock is None
    finally:
        driver.close()
        fake.close()


@pytest.mark.parametrize('reply,cache_free', [
    (b'set_cache: 0\nRPRT 0\n', True),
    (b'RPRT 0\n', None),
    (b'RPRT -1\nget_freq:\nFrequency: 7100000\nRPRT 0\n', False),
    (b'RPRT -1\n7100000\n', None),
])
def test_setup_requires_extended_framing(reply, cache_free, monkeypatch):
    # Byte-at-a-time scripted peer: exercise real framing and open cleanup
    # without requiring a listening TCP socket.
    sock = Mock()
    sock.recv.side_effect = [bytes([byte]) for byte in reply] + [b'']
    driver = RigctldCat()
    monkeypatch.setattr(driver, '_connect', lambda deadline: sock)
    try:
        if cache_free is None:
            with pytest.raises(RigError):
                driver.open()
            assert driver._sock is None
            sock.close.assert_called_once()
        else:
            driver.open()
            assert driver._sock is sock
            assert driver.cache_free is cache_free
            sock.close.assert_not_called()
    finally:
        driver.close()


def test_rigctld_panic_uses_independent_connection(monkeypatch):
    with rigctld() as (driver, fake):
        entered, release = threading.Event(), threading.Event()
        answer = fake._answer

        def stalled(command):
            if command == 'f':
                entered.set()
                release.wait(2)
            return answer(command)

        monkeypatch.setattr(fake, '_answer', stalled)
        thread, result = background(driver.freq)
        assert entered.wait(1)
        try:
            assert driver.emergency_unkey() is False
        finally:
            release.set()
        join(thread)
        assert isinstance(result[0], RigError)


def test_rigctld_raw_roof_power_and_refused_setter(monkeypatch):
    with rigctld(profile='ftdx10') as (driver, fake):
        original = fake._answer

        def answer(command):
            if command == '\\send_raw ; RF0;':
                return 'send_cmd', 'RF0;', ['Reply: RF06;'], [], 0
            if command.startswith('L RFPOWER'):
                return 'set_level', command[2:], [], [], 0
            return original(command)

        monkeypatch.setattr(fake, '_answer', answer)
        assert driver.receiver_filters()[1] == 12000
        driver.set_power(25)
        assert fake.log[-1] == '+L RFPOWER 0.25'
        fake.refuse = True
        assert driver.set_mode('PKTUSB', 2400) is False


@pytest.mark.parametrize('payload', ['FA007100000;FA007100001;', 'VS0;', ''])
def test_rigctld_raw_is_exactly_one_reply(monkeypatch, payload):
    with rigctld() as (driver, fake):
        monkeypatch.setattr(fake, '_answer', lambda command: ('send_cmd', 'FA;', [payload], [], 0))
        with pytest.raises(RigError, match='ambiguous'):
            driver.read('FA;', 'FA')


def test_generic_power_needs_explicit_scaling():
    driver = RigctldCat(profile='generic')
    with pytest.raises(RigError, match='power scaling unknown'):
        driver.set_power(5)
    driver = RigctldCat(profile='generic', max_watts=20)
    sent = []
    driver.command = sent.append
    driver.set_power(5)
    assert sent == ['L RFPOWER 0.25']
    assert resolve_rig('generic') == 'generic'
    row = RIGS['generic']
    assert row['model'] == 0 and row['cat'] == 'rigctld'
    assert row['audio_chain'] == 'generic' and row['settle'] == 0.1
    assert not row['data_widths'] and not row['readback']['raw_reads']
    assert 'max_watts' not in row and 'ptt_port' not in row


def test_profile_line_states_require_booleans():
    _, row, _ = resolve_profile('ft891', environ={}, profiles={'ft891': {'cat_rts': True, 'cat_dtr': False}})
    assert row['cat_rts'] is True and row['cat_dtr'] is False
    _, row, _ = resolve_profile('ft891', environ={}, profiles={'ft891': {'cat_rts': 'true', 'cat_dtr': 1}})
    assert row['cat_rts'] is False and row['cat_dtr'] is False


@pytest.mark.hardware
@pytest.mark.skip(reason='requires separate authorized RTS/TX bench on both Yaesus; never run in this unit')
@pytest.mark.parametrize('model', ['ft891', 'ftdx10'])
def test_tx_readback_with_external_rts_bench(model):
    pass


@pytest.fixture
def deadline_budget(monkeypatch):
    class Clock:
        now = 10.0

        def monotonic(self):
            return self.now

    clock = Clock()
    timeouts, commands, connections = [], [], []

    class BudgetSocket:
        def __init__(self):
            self.timeout = None
            self.command = ''
            self.closed = False

        def settimeout(self, timeout):
            self.timeout = timeout
            timeouts.append((clock.now, timeout))

        def advance(self):
            remaining = self.timeout
            clock.now += min(0.22, remaining)
            if remaining < 0.22:
                raise TimeoutError('budget exhausted')

        def sendall(self, raw):
            self.advance()
            self.command = raw.decode()
            commands.append(self.command)

        def recv(self, size):
            self.advance()
            if 'get_conf' in self.command:
                return b'get_conf: ptt_type\nptt_type=None\nRPRT 0\n'
            if self.command == '+f\n':
                return b'get_freq:\nFrequency: 7100000\nRPRT 0\n'
            return b'set_cache: 0\nRPRT 0\n'

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def connect(address, timeout):
        timeouts.append((clock.now, timeout))
        clock.now += min(timeout, 0.22)
        if timeout < 0.22:
            raise TimeoutError('connect budget exhausted')
        sock = BudgetSocket()
        connections.append(sock)
        return sock

    monkeypatch.setattr(module, 'time', clock)
    monkeypatch.setattr(module.socket, 'create_connection', connect)
    return clock, timeouts, commands, connections


@pytest.mark.parametrize('operation', ['freq', 'get_conf', 'open_refused'])
def test_rigctld_one_deadline_includes_connection_setup_send_and_receive(deadline_budget, operation, monkeypatch):
    clock, timeouts, commands, connections = deadline_budget
    driver = RigctldCat()
    if operation == 'open_refused':
        wire = driver._wire

        def refuse(sock, command, deadline, **kwargs):
            result = wire(sock, command, deadline, **kwargs)
            if 'set_cache' in command:
                raise module._Refused('old daemon')
            return result

        monkeypatch.setattr(driver, '_wire', refuse)
    try:
        with pytest.raises(RigError):
            if operation == 'freq':
                driver.freq()
            elif operation == 'get_conf':
                driver.get_conf('ptt_type')
            else:
                driver.open()
        assert clock.now == pytest.approx(11)
        assert all(now + timeout == pytest.approx(11) for now, timeout in timeouts)
        assert all(sock.closed for sock in connections[1:])
        assert connections[0].closed is (operation != 'get_conf')
        assert len(connections) == (2 if operation == 'get_conf' else 1)
    finally:
        driver.close()


def test_disposable_configuration_has_one_deadline_after_open(deadline_budget):
    clock, timeouts, commands, connections = deadline_budget
    driver = RigctldCat()
    driver.open()
    timeouts.clear()
    started = clock.now
    try:
        assert not driver.sole_ptt_owner([])
        assert clock.now == pytest.approx(started + 1)
        assert all(now + timeout == pytest.approx(started + 1) for now, timeout in timeouts)
        assert len(connections) == 2
        assert connections[1].closed and not connections[0].closed
    finally:
        driver.close()


def test_raw_passthrough_uses_read_terminator_and_no_reply_setter(monkeypatch):
    with rigctld() as (driver, fake):
        def answer(command):
            if command == '\\send_raw ; RT;':
                return 'send_raw', '; RT;', ['Reply: RT0;'], [], 0
            if command == '\\send_raw 0 RT0;':
                return 'send_raw', '0 RT0;', [], [], 0
            return 'unknown', command, [], [], -1

        monkeypatch.setattr(fake, '_answer', answer)
        driver.write('RT0;')
        assert driver.read('RT;', 'RT') == '0'
        assert fake.log[-2:] == ['+\\send_raw 0 RT0;', '+\\send_raw ; RT;']


def test_configuration_query_rechecks_keyed_guard():
    with rigctld() as (driver, fake):
        driver.keyed_probe = lambda: True
        with pytest.raises(RigError, match='protected'):
            driver.get_conf('ptt_type')
        assert not any('get_conf' in command for command in fake.log)


def test_unknown_serial_dependency_is_reported(monkeypatch):
    import builtins

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == 'serial':
            raise ImportError('no serial in this environment')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing)
    driver = YaesuCat('/tmp/fake-unused-port')
    try:
        with pytest.raises(RigError, match='no serial'):
            driver.open()
    finally:
        driver.close()


def test_frequency_cadence_backoff_caps_at_ten_seconds(monkeypatch):
    class Clock:
        now = 100.0

        def monotonic(self):
            return self.now

    clock = Clock()
    monkeypatch.setattr(module, 'time', clock)
    driver = YaesuCat('/tmp/fake-unused-port')
    requests = []

    def failed(command, prefix, **kwargs):
        requests.append((command, prefix, kwargs['timeout']))
        raise RigError('radio silent')

    monkeypatch.setattr(driver, 'exchange', failed)
    for delay in (0.6, 1.2, 2.4, 4.8, 9.6, 10.0, 10.0):
        with pytest.raises(RigError, match='silent'):
            driver.freq()
        assert driver.freq_retry_at - clock.now == pytest.approx(delay)
        count = len(requests)
        with pytest.raises(RigError, match='backoff'):
            driver.freq()
        assert len(requests) == count
        clock.now = driver.freq_retry_at
    assert requests == [('FA;', 'FA', 0.3)] * 7
    monkeypatch.setattr(driver, 'exchange', lambda *args, **kwargs: 'FA007100000;')
    assert driver.freq() == 7100000
    assert driver.freq_retry_at == 0 and driver._freq_backoff == 0.3


def test_malformed_sentinel_does_not_resume_normal_traffic():
    with yaesu(replies={'FA;': None, 'ID;': 'ID;'}) as (driver, fake):
        with pytest.raises(RigError):
            driver.exchange('FA;', 'FA', timeout=0.03)
        with pytest.raises(RigError, match='sentinel'):
            driver.read('VS;', 'VS')
        assert 'VS;' not in fake.log
        assert driver.poisoned
