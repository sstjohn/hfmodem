# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from contextlib import contextmanager
from importlib.util import find_spec
import io
import os
from pathlib import Path
import queue
import subprocess
import sys
import textwrap
from threading import get_ident
from types import SimpleNamespace

import pytest

from hfmodem.contact import chat_state as s
from hfmodem.tests.contact.test_varac import Host as WireHost

pytestmark = pytest.mark.skipif(find_spec('textual') is None, reason='requires hfmodem[tui]')


def assert_terminal_restored(fd, attrs):
    import termios
    restored = termios.tcgetattr(fd)
    # macOS sets PENDIN when re-entering canonical mode, even with no queued input.
    restored[3] &= ~getattr(termios, 'PENDIN', 0)
    expected = list(attrs)
    expected[3] &= ~getattr(termios, 'PENDIN', 0)
    assert restored == expected


@contextmanager
def pty_screen(master, select_fn=None):
    import select
    from threading import Event, Thread

    select_fn = select_fn or select.select
    finished = Event()
    screen = bytearray()

    def receive():
        while not finished.is_set() or select_fn([master], [], [], 0)[0]:
            if select_fn([master], [], [], .01)[0]:
                screen.extend(os.read(master, 65536))

    reader = Thread(target=receive)
    reader.start()
    try:
        yield screen
    finally:
        finished.set()
        reader.join(3)
        assert not reader.is_alive()


@pytest.mark.parametrize('exit_path', ('ordered', 'interrupt', 'startup', 'second-panic', 'deadman'))
def test_terminal_ctrl_c_panics_repeatedly_while_the_owner_is_stalled(exit_path):
    import fcntl
    import pty
    import select
    import termios
    import time

    script = r'''
        import os
        import termios
        import threading
        from types import SimpleNamespace
        from hfmodem.contact.chat_controller import Controller
        from hfmodem.contact.chat_fake import FakeSession
        from hfmodem.ui.textual_chat import ChatApp
        from vara_rig_bridge import Rig, _install_unkey_handlers

        stall, stalled, release = (threading.Event() for _ in range(3))
        class Fake(FakeSession):
            def poll(self, timeout_s):
                if stall.is_set():
                    stalled.set()
                    release.wait()
                return super().poll(timeout_s)

            def quit(self, *, unwinding=None):
                super().quit(unwinding=unwinding)
                print('OWNER QUIT', flush=True)

        rig = SimpleNamespace(panic=lambda why: print('PANIC ' + why, flush=True),
                              shutdown=lambda: None)
        if HARD_EXIT:
            rig._panicked = False
            rig._no_tx = threading.Event()
            rig._arm_deadman = lambda: Rig._arm_deadman(rig)
            rig.unkey_hard = lambda why: print('UNKEY ' + why, flush=True)
            def panic(why):
                print('PANIC ' + why, flush=True)
                Rig.panic(rig, why)
            rig.panic = panic
        _install_unkey_handlers(rig)
        def watch():
            stalled.wait()
            print('STALLED', flush=True)
            os.read(RELEASE_FD, 1)
            release.set()
        threading.Thread(target=watch, daemon=True).start()
        class App(ChatApp):
            async def on_mount(self):
                await super().on_mount()
                stall.set()

            def action_ordered_exit(self):
                super().action_ordered_exit()
                print('ORDERED EXIT', flush=True)

        if STARTUP_FAILURE:
            original_start = threading.Thread.start
            def start(thread):
                if thread.name == 'textual-input':
                    assert termios.tcgetattr(0)[3] & termios.ISIG
                    stall.set()
                    stalled.wait()
                    print('RAW START FAILED', flush=True)
                    raise RuntimeError('input thread never started')
                return original_start(thread)
            threading.Thread.start = start

        with open(os.dup(0), 'w', buffering=1) as tty:
            import signal
            app = App(Controller(Fake()), terminal=tty, stdout=__import__('sys').stdout,
                      exit_on_failure=True)
            app.interrupt = lambda: os.kill(os.getpid(), signal.SIGINT)
            attrs = termios.tcgetattr(0)
            try:
                app.run(size=(80, 24))
            finally:
                restored = termios.tcgetattr(0)
                restored[3] &= ~getattr(termios, 'PENDIN', 0)
                attrs[3] &= ~getattr(termios, 'PENDIN', 0)
                assert restored == attrs, (restored, attrs)
                print('RESTORED', flush=True)
    '''
    master, slave = pty.openpty()
    attrs = termios.tcgetattr(slave)
    attrs[6][termios.VINTR] = b'\x18'
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    startup_failure = exit_path == 'startup'
    hard_exit = exit_path in ('second-panic', 'deadman')
    release_r, release_w = os.pipe()

    def controlling_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    child = subprocess.Popen(
        [sys.executable, '-c', textwrap.dedent(script).replace('RELEASE_FD', str(release_r))
         .replace('STARTUP_FAILURE', repr(startup_failure)).replace('HARD_EXIT', repr(hard_exit))],
        stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(release_r,),
        preexec_fn=controlling_terminal, env=os.environ | {'TERM': 'xterm-256color'})
    os.close(release_r)
    evidence = b''
    screen = b''
    errors = b''

    def read_output(timeout):
        nonlocal evidence, screen, errors
        for fd in select.select([master, child.stdout.fileno(), child.stderr.fileno()], [], [], timeout)[0]:
            try:
                data = os.read(fd, 65536)
            except OSError:
                data = b''
            if fd == child.stdout.fileno():
                evidence += data
            elif fd == child.stderr.fileno():
                errors += data
            else:
                screen += data

    def expect(marker, timeout=5, count=1):
        deadline = time.monotonic() + timeout
        while evidence.count(marker) < count and time.monotonic() < deadline:
            read_output(.05)
            if child.poll() is not None:
                read_output(0)
                break
        assert evidence.count(marker) >= count, (evidence, child.poll())

    try:
        expect(b'STALLED\n')
        if startup_failure:
            expect(b'RAW START FAILED\n')
        elif exit_path == 'ordered':
            os.write(master, b'\x11')
            expect(b'ORDERED EXIT\n')
        active = termios.tcgetattr(slave)
        assert active[3] & termios.ISIG
        disabled = bytes([os.fpathconf(slave, 'PC_VDISABLE')])
        assert active[6][termios.VQUIT] == disabled
        assert active[6][termios.VSUSP] == disabled
        assert active[6][termios.VINTR] == b'\x03'
        assert active[3] & termios.NOFLSH
        os.write(master, b'\x1c\x1a')
        read_output(.15)
        assert child.poll() is None
        assert b'PANIC' not in evidence
        for count in ((1,) if exit_path == 'deadman' else (1, 2)):
            os.write(master, b'\x03')
            expect(b'PANIC SIGINT\n', timeout=1, count=count)
            if hard_exit and count == 1:
                expect(b'UNKEY SIGINT\n', timeout=1)
        if not hard_exit:
            assert child.poll() is None
        assert b'OWNER QUIT' not in evidence
    finally:
        if not hard_exit:
            os.write(release_w, b'x')
        deadline = time.monotonic() + 8
        while child.poll() is None and time.monotonic() < deadline:
            read_output(.05)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
        os.close(release_w)
        read_output(0)
        while select.select([master], [], [], 0)[0]:
            data = os.read(master, 65536)
            if not data:
                break
            screen += data
        os.close(slave)
        os.close(master)
    assert child.returncode == 1, (evidence, errors.decode())
    assert (b'OWNER QUIT\n' in evidence) is not hard_exit
    assert (b'RESTORED\n' in evidence) is not hard_exit
    assert b'\x1b[>25u' not in screen
    assert b'\x1b[<u' not in screen
    assert (b'\x1b[?1049l\x1b[?25h\x1b[?1004l' in screen) is not hard_exit
    if hard_exit:
        assert b'UNKEY SIGINT\n' in evidence


@pytest.mark.parametrize('stage', ('stale-read', 'reset-write', 'reset-drain'))
def test_vintr_preserves_queues_and_stale_readiness_cannot_hold_teardown(stage):
    import fcntl
    import select
    import termios
    import time

    script = r'''
        import asyncio
        import os
        import sys
        import termios
        import threading
        from types import SimpleNamespace
        from hfmodem.contact.chat_controller import Controller
        from hfmodem.contact.chat_fake import FakeSession
        from hfmodem.ui.textual_chat import ChatApp, TTYDriver, defer_interrupts
        from vara_rig_bridge import _install_unkey_handlers

        panics = []
        def panic(why):
            panics.append(why)
            print('PANIC ' + why, flush=True)
        _install_unkey_handlers(SimpleNamespace(panic=panic, shutdown=lambda: None))
        read, drain = os.read, termios.tcdrain
        paused = False
        def input_read(fd, size):
            global paused
            if threading.current_thread().name == 'textual-input' and not paused:
                paused = True
                print('READ READY', flush=True)
                read(RELEASE_FD, 1)
            return read(fd, size)

        async def check():
            with open(os.dup(0), 'w', buffering=1) as tty:
                attrs = termios.tcgetattr(0)
                controller = Controller(FakeSession())
                controller.ready.result(3)
                app = ChatApp(controller, terminal=tty)
                driver = TTYDriver(app)
                write = tty.write
                def reset_write(data):
                    if data.startswith('\x1b[?2004l\x1b[?7h'):
                        assert controller.join(0)
                        assert termios.tcgetattr(0)[3] & termios.NOFLSH
                        result = write(data)
                        tty.flush()
                        print('RESET WRITE', flush=True)
                        read(RELEASE_FD, 1)
                        return result
                    return write(data)
                def reset_drain(fd):
                    assert controller.join(0)
                    assert termios.tcgetattr(fd)[3] & termios.NOFLSH
                    print('RESET DRAIN', flush=True)
                    read(RELEASE_FD, 1)
                    return drain(fd)
                if STAGE == 'stale-read':
                    os.read = input_read
                elif STAGE == 'reset-write':
                    tty.write = reset_write
                else:
                    termios.tcdrain = reset_drain
                driver.start_application_mode()
                print('STARTED', flush=True)
                if STAGE == 'stale-read':
                    while not panics:
                        await asyncio.sleep(.01)
                driver.stop_application_mode()
                assert not driver._key_thread.is_alive()
                restored = termios.tcgetattr(0)
                restored[3] &= ~getattr(termios, 'PENDIN', 0)
                attrs[3] &= ~getattr(termios, 'PENDIN', 0)
                assert restored == attrs, (restored, attrs)
                print('RESTORED', flush=True)
        with defer_interrupts():
            asyncio.run(check())
    '''
    master, slave = os.openpty()
    attrs = termios.tcgetattr(slave)
    attrs[3] &= ~termios.NOFLSH
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    release_r, release_w = os.pipe()

    def controlling_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    child = subprocess.Popen(
        [sys.executable, '-c', textwrap.dedent(script).replace('RELEASE_FD', str(release_r))
         .replace('STAGE', repr(stage))], stdin=slave, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, pass_fds=(release_r,), preexec_fn=controlling_terminal,
        env=os.environ | {'TERM': 'xterm-256color'})
    os.close(release_r)
    evidence, screen, errors = bytearray(), bytearray(), bytearray()

    def receive(timeout):
        for fd in select.select([master, child.stdout.fileno(), child.stderr.fileno()], [], [], timeout)[0]:
            data = os.read(fd, 65536)
            target = evidence if fd == child.stdout.fileno() else errors if fd == child.stderr.fileno() else screen
            target.extend(data)

    def expect(marker, timeout=5):
        deadline = time.monotonic() + timeout
        while marker not in evidence and time.monotonic() < deadline:
            receive(.05)
            if child.poll() is not None:
                receive(0)
                break
        assert marker in evidence, (evidence, errors, child.poll())

    try:
        expect(b'STARTED\n')
        if stage == 'stale-read':
            os.write(master, b'x')
            expect(b'READ READY\n')
        else:
            expect(b'RESET WRITE\n' if stage == 'reset-write' else b'RESET DRAIN\n')
        os.write(master, b'\x03')
        expect(b'PANIC SIGINT\n')
        if stage == 'stale-read':
            assert select.select([slave], [], [], 0)[0], 'VINTR flushed readable input'
            assert os.read(slave, 1) == b'x'
        os.write(release_w, b'x')
        expect(b'RESTORED\n', timeout=1.5)
        deadline = time.monotonic() + 3
        while child.poll() is None and time.monotonic() < deadline:
            receive(.05)
        assert child.poll() == 1, (evidence, errors)
        receive(0)
        assert b'\x1b[?1049l\x1b[?25h\x1b[?1004l' in screen
        assert b'\x1b[<u' not in screen
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)
        os.close(release_w)
        os.close(slave)
        os.close(master)


@pytest.fixture
def runtime(tools, monkeypatch, tmp_path):
    import varac_chat as launch
    import kestrel_connect
    from hfmodem.contact import slots
    from hfmodem.core import config
    from hfmodem.station import pskreporter
    from hfmodem.ui import textual_chat

    trace, sessions, views = [], [], []
    state = SimpleNamespace(failure=None, snapshot=None, app_options=None, released_low=True)
    app_class = textual_chat.ChatApp

    def note(kind):
        trace.append((kind, get_ident()))

    def forbidden(*args, **kwargs):
        raise AssertionError('line terminal/readline constructed in Textual mode')

    monkeypatch.setattr(launch, 'Terminal', forbidden)
    monkeypatch.setattr(launch, 'sense_ui', lambda requested: ('textual', None))
    monkeypatch.setattr(config, 'power_limits', lambda: (1., 10., None))
    monkeypatch.setattr(config, 'tx_drive', lambda: 0.)
    monkeypatch.delenv('HFMODEM_OWNER_FD', raising=False)
    wire = WireHost(replies={'VERSION': ['VERSION fake'], 'DISCONNECT': ['OK', 'DISCONNECTED']})

    class Recorder:
        position = 0
        session = SimpleNamespace(policy='off', directory=tmp_path)

    class Audio:
        def __init__(self, *args, **kwargs):
            note('audio-create')
            self.taps = []
            self.rec = Recorder()

    class Watch:
        def __init__(self, *args, **kwargs):
            note('slots-create')

        def table(self):
            return (), None

        def poll(self):
            return ()

        def close(self):
            note('slots-close')

    class Host:
        def __init__(self, audio):
            note('host-create')
            self.server = SimpleNamespace(cmd_port=1, data_port=2)
            self.runtime = SimpleNamespace(modem=None)

        def __enter__(self):
            note('host-open')
            return self

        def __exit__(self, *exc):
            note('host-close')

        def arm(self, on):
            rig.armed = on

        def check(self):
            if state.failure == 'host':
                state.failure = None
                raise launch.HostError('fake Host failure')

    class Reporter:
        def __init__(self, config):
            note('reporter-create')

        def status(self):
            return dict(submitted=0, enabled=True)

        def close(self):
            note('reporter-close')
            return dict(submitted=1, unsent_at_shutdown=2, inflight_at_shutdown=3)

    class ContactLog:
        def __init__(self, directory, *, mycall, recording):
            note('contact-log-create')

        def update(self, *args, **kwargs):
            return SimpleNamespace()

        def finish(self, contact):
            note('save-finish')

    @contextmanager
    def client_open(**options):
        note('client-open')
        wire.client.ready = options['ready']
        wire.client.event = options['event']
        try:
            yield wire.client
        finally:
            note('client-close')
            wire.client.close()

    def select(readers, writers, exceptions, timeout):
        wire.now += timeout
        if state.failure == 'owner':
            state.failure = None
            return [91], [], []
        return [reader for reader in readers if getattr(reader, 'incoming', ())], [], []

    real_select = launch.select.select
    monkeypatch.setattr(launch.select, 'select', select)
    monkeypatch.setattr(launch.Client, 'open', client_open)
    monkeypatch.setattr(kestrel_connect, 'AudioVaraIO', Audio)
    monkeypatch.setattr(slots, 'SlotWatch', Watch)
    monkeypatch.setattr(launch, 'Host', Host)
    monkeypatch.setattr(launch, 'ContactLog', ContactLog)
    monkeypatch.setattr(pskreporter, 'Config', lambda *args, **kwargs: None)
    monkeypatch.setattr(pskreporter, 'Reporter', Reporter)

    original_session = launch.Session

    def session(*args, **kwargs):
        note('session-create')
        value = original_session(*args, **kwargs)
        sessions.append(value)
        original_quit, original_unsent = value.quit, value.identification_unsent
        original_close = value.parking.store.close
        original_monitor = value.monitor

        def quit(*, unwinding=None):
            note('session-quit')
            return original_quit(unwinding=unwinding)

        def unsent():
            note('identification-unsent')
            return original_unsent()

        def close():
            note('parking-close')
            return original_close()

        def monitor():
            note('session-monitor')
            original_monitor()
            if state.monitor_error is not None:
                raise state.monitor_error
            value.say('semantic session notice')
            value.bbs.remote_view = True
            value.bbs.message('remote BBS page\r\n')
            value.bbs.remote_view = False
            value.emit('band', band_m=20, dial_hz=14105000)
            value.ident_report('IDENT fake report\n')
            value.emit('contact', ended_utc='2026-10-29T00:00:00+00:00')

        value.quit, value.identification_unsent = quit, unsent
        value.parking.store.close, value.monitor = close, monitor
        return value

    session.COMMANDS = original_session.COMMANDS
    monkeypatch.setattr(launch, 'Session', session)

    original_open = Path.open

    class Stream:
        def __init__(self, stream, name):
            self.stream, self.name = stream, name

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *exc):
            note(self.name + '-close')
            return self.stream.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def opened(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return Stream(stream, path.name) if path.name in ('events.jsonl', 'contacts.jsonl', 'host.jsonl') else stream

    monkeypatch.setattr(Path, 'open', opened)
    rig = SimpleNamespace(keyed=False, retired=False, line_held=True, released_low=None,
                          meter_reports=queue.Queue(), power_w=lambda **kwargs: None)
    rig.line_state = lambda: False if rig.line_held else None

    def shutdown():
        note('rig-shutdown')
        rig.retired = True
        if rig.line_held:
            rig.released_low = state.released_low
            rig.line_held = False

    rig.shutdown = shutdown
    admission = SimpleNamespace(opened=lambda: note('admission-opened'))
    directory = tmp_path / 'operator' / 'sessions' / 'one'
    directory.mkdir(parents=True)
    args = launch.parser().parse_args([
        '--mycall', 'W9SSJ', '--band', '20', '--session', str(directory),
        '--expect-model', 'fake', '--ptt-device', 'unused', '--tx-device', 'unused',
        '--rx-device', 'unused', '--no-bbs', '--pskreporter', '--loc', 'EN51', '--ui', 'textual'])
    terminal = io.StringIO()

    @contextmanager
    def tty():
        note('tty-open')
        try:
            yield terminal
        finally:
            note('tty-restore')

    monkeypatch.setattr(launch, 'textual_tty', tty)

    class App:
        failure = None

        def __init__(self, controller, **kwargs):
            self.controller = controller
            state.app_options = kwargs
            assert kwargs['terminal'] is terminal
            assert kwargs['title'] == 'Kestrel chat'
            assert kwargs['exit_on_failure'] is True

        def run(self):
            self.controller.ready.result(3)
            self.controller.check(s.ChatCancelWait()).result(3)
            state.snapshot = self.controller.snapshot()
            if state.app_error is not None:
                raise state.app_error
            if state.failure_after_ready:
                if state.failure_after_ready == 'retired':
                    rig.retired = True
                else:
                    state.failure = state.failure_after_ready
                self.controller.check(s.ChatCancelWait())
                assert self.controller.join(3)
            self.controller.shutdown().result(3)
            assert self.controller.join(3)
            views.append(self.controller.snapshot())

    state.failure_after_ready = None
    state.app_error = None
    state.monitor_error = None
    monkeypatch.setattr(textual_chat, 'ChatApp', App)
    return SimpleNamespace(launch=launch, args=args, rig=rig, admission=admission, trace=trace,
                           state=state, sessions=sessions, views=views, tty=terminal, wire=wire,
                           app_class=app_class, real_select=real_select)


def test_real_textual_owns_session_and_finalizes_before_restoring_terminal(runtime, capsys):
    assert runtime.launch.run(runtime.args, runtime.rig, runtime.admission) == 0
    names = [name for name, _ in runtime.trace]
    ordered = ['session-quit', 'parking-close', 'client-close', 'host-close', 'reporter-close',
               'contacts.jsonl-close', 'events.jsonl-close',
               'host.jsonl-close', 'slots-close', 'rig-shutdown', 'tty-restore']
    assert [name for name in names if name in ordered] == ordered
    owner_calls = [(name, ident) for name, ident in runtime.trace if name not in ('tty-open', 'tty-restore')]
    assert {ident for _, ident in owner_calls} != {get_ident()}
    assert len({ident for _, ident in owner_calls}) == 1
    assert runtime.tty.getvalue() == ''
    snapshot = runtime.views[-1]
    payloads = [record.payload for record in snapshot.presentation.records]
    assert sum(isinstance(payload, s.Notice) and payload.text == 'semantic session notice'
               for payload in payloads) == 1
    assert any(isinstance(payload, s.BBSPage) and 'remote BBS page' in ''.join(
        span.text for span in payload.spans) for payload in payloads)
    assert any(isinstance(payload, s.Notice) and payload.text.startswith('PSKReporter shutdown:')
               for payload in payloads)
    assert runtime.rig.line_state() is None and runtime.rig.released_low is True
    assert snapshot.tx.physical_line is False
    assert capsys.readouterr().out == (
        'BAND 20 m 14105000 Hz\nIDENT fake report\n\n')


@pytest.mark.parametrize('released_low', (False, None))
def test_textual_closed_keyer_without_low_handback_proof_still_fails(runtime, released_low):
    runtime.state.released_low = released_low
    with pytest.raises(RuntimeError, match='low-line proof'):
        runtime.launch.run(runtime.args, runtime.rig, runtime.admission)
    assert runtime.rig.retired and not runtime.rig.line_held
    assert runtime.rig.line_state() is None
    assert [name for name, _ in runtime.trace][-2:] == ['rig-shutdown', 'tty-restore']


@pytest.mark.parametrize('failure,message', [('host', 'fake Host failure'),
                                           ('owner', 'Console owner exited'),
                                           ('retired', 'Station session was retired')])
def test_real_textual_runtime_failure_reports_unsent_and_restores_every_resource(
        runtime, monkeypatch, failure, message):
    runtime.state.failure_after_ready = failure
    if failure == 'owner':
        monkeypatch.setenv('HFMODEM_OWNER_FD', '91')
        from hfmodem.core import pipeio
        monkeypatch.setattr(pipeio, 'Owner', lambda fd: SimpleNamespace(
            gone=lambda: runtime.state.failure_after_ready == 'owner' and runtime.state.snapshot is not None,
            close=lambda: None))
    with pytest.raises(runtime.launch.HostError, match=message):
        runtime.launch.run(runtime.args, runtime.rig, runtime.admission)
    names = [name for name, _ in runtime.trace]
    assert 'identification-unsent' in names
    assert names.count('session-quit') == 1
    assert names.count('parking-close') == 1
    for name in ('client-close', 'host-close', 'slots-close', 'reporter-close',
                 'contacts.jsonl-close', 'events.jsonl-close', 'host.jsonl-close'):
        assert names.count(name) == 1
    assert names[-1] == 'tty-restore'
    assert names.count('rig-shutdown') == (2 if failure == 'owner' else 1)
    assert runtime.tty.getvalue() == ''


@pytest.mark.parametrize('error', [None, KeyboardInterrupt(), RuntimeError('driver failed')])
def test_textual_tty_restores_modes_after_success_interrupt_or_failure(tools, monkeypatch, error):
    import varac_chat as launch
    calls = []

    class Stream(io.StringIO):
        def fileno(self):
            return 83

        def close(self):
            calls.append('close')
            super().close()

    stream = Stream()
    monkeypatch.setattr(launch, 'os', SimpleNamespace(
        O_RDWR=launch.os.O_RDWR,
        open=lambda path, flags: calls.append(('open', path, flags)) or 83))
    monkeypatch.setattr(launch, 'open', lambda fd, *args, **kwargs: stream, raising=False)
    attrs = [1, 2, 3]
    monkeypatch.setattr(launch.termios, 'tcgetattr', lambda fd: calls.append(('get', fd)) or attrs)
    monkeypatch.setattr(launch.termios, 'tcsetattr',
                        lambda fd, when, value: calls.append(('set', fd, when, value)))

    def use():
        with launch.textual_tty() as output:
            assert output is stream
            if error is not None:
                raise error

    if error is None:
        use()
    else:
        with pytest.raises(type(error)) as caught:
            use()
        assert caught.value is error
    assert calls == [('open', '/dev/tty', launch.os.O_RDWR), ('get', 83),
                     ('set', 83, launch.termios.TCSANOW, attrs), 'close']


@pytest.mark.parametrize('error', [KeyboardInterrupt(), SystemExit(1), RuntimeError('screen failed')])
def test_real_textual_interrupt_and_screen_error_join_owner_before_terminal_restore(runtime, error):
    runtime.state.app_error = error
    with pytest.raises(type(error)) as caught:
        runtime.launch.run(runtime.args, runtime.rig, runtime.admission)
    assert caught.value is error
    names = [name for name, _ in runtime.trace]
    assert names.count('session-quit') == 1
    assert names.count('parking-close') == 1
    assert names.index('session-quit') < names.index('client-close') < names.index('host-close')
    assert names[-2:] == ['rig-shutdown', 'tty-restore']


def test_real_textual_monitor_startup_error_closes_the_partial_owned_session(runtime):
    original = runtime.launch.HostError('monitor startup failed')
    runtime.state.monitor_error = original
    with pytest.raises(runtime.launch.HostError, match='monitor startup failed') as caught:
        runtime.launch.run(runtime.args, runtime.rig, runtime.admission)
    assert caught.value is original
    names = [name for name, _ in runtime.trace]
    assert names.count('session-quit') == names.count('parking-close') == 1
    assert 'identification-unsent' in names
    assert names.index('session-quit') < names.index('client-close') < names.index('host-close')
    assert names[-2:] == ['rig-shutdown', 'tty-restore']


@pytest.fixture
def driver_terminal(runtime, monkeypatch):
    from hfmodem.ui import textual_chat
    master, slave = os.openpty()
    stream = open(slave, 'w', buffering=1)
    original_stop = textual_chat.LinuxDriver.stop_application_mode

    def stop(driver):
        runtime.trace.append(('driver-stop', get_ident()))
        return original_stop(driver)

    @contextmanager
    def tty():
        try:
            yield stream
        finally:
            runtime.trace.append(('tty-restore', get_ident()))

    monkeypatch.setattr(textual_chat.LinuxDriver, 'stop_application_mode', stop)
    monkeypatch.setattr(runtime.launch, 'textual_tty', tty)
    try:
        with pty_screen(master, runtime.real_select) as screen:
            yield screen
    finally:
        stream.close()
        os.close(master)


def test_textual_handler_failure_reaches_unsent_handling_before_owner_cleanup(
        runtime, monkeypatch, driver_terminal):
    from hfmodem.ui import textual_chat
    original = RuntimeError('screen handler failed')

    class App(runtime.app_class):
        async def on_mount(self):
            await super().on_mount()
            self.call_later(self.break_screen)

        def break_screen(self):
            raise original

        def run(self):
            return super().run(size=(80, 24))

    monkeypatch.setattr(textual_chat, 'ChatApp', App)
    with pytest.raises(RuntimeError, match='screen handler failed') as caught:
        runtime.launch.run(runtime.args, runtime.rig, runtime.admission)
    assert caught.value is original
    names = [name for name, _ in runtime.trace]
    assert [name for name in names if name in ('session-quit', 'identification-unsent', 'parking-close')] == [
        'session-quit', 'identification-unsent', 'identification-unsent', 'parking-close']
    assert names[-3:] == ['rig-shutdown', 'driver-stop', 'tty-restore']


@pytest.mark.parametrize('started', (False, True))
def test_driver_start_failure_finishes_owner_before_restoring_modes(
        runtime, monkeypatch, driver_terminal, started):
    from hfmodem.ui import textual_chat
    original_start = textual_chat.LinuxDriver.start_application_mode
    original = RuntimeError('partial driver startup')

    def start(driver):
        driver._app.controller.ready.result(3)
        if started:
            original_start(driver)
        raise original

    class App(runtime.app_class):
        def bell(self):
            pass

        def run(self):
            return super().run(size=(80, 24))

    monkeypatch.setattr(textual_chat, 'ChatApp', App)
    monkeypatch.setattr(textual_chat.LinuxDriver, 'start_application_mode', start)
    with pytest.raises(RuntimeError, match='partial driver startup'):
        runtime.launch.run(runtime.args, runtime.rig, runtime.admission)
    names = [name for name, _ in runtime.trace]
    assert 'identification-unsent' in names
    assert names.index('session-quit') < names.index('rig-shutdown') < names.index('tty-restore')
    if started:
        assert names.index('rig-shutdown') < names.index('driver-stop') < names.index('tty-restore')


def test_real_textual_exit_keys_finish_owner_teardown_before_terminal_restore(
        runtime, monkeypatch, driver_terminal):
    from hfmodem.ui import textual_chat

    async def exit_chat(pilot):
        await pilot.press('ctrl+q')

    class App(runtime.app_class):
        def run(self):
            return super().run(size=(80, 24), auto_pilot=exit_chat)

    monkeypatch.setattr(textual_chat, 'ChatApp', App)
    assert runtime.launch.run(runtime.args, runtime.rig, runtime.admission) == 0
    names = [name for name, _ in runtime.trace]
    assert names.count('session-quit') == names.count('parking-close') == 1
    assert names.index('session-quit') < names.index('client-close') < names.index('host-close')
    assert names[-3:] == ['rig-shutdown', 'driver-stop', 'tty-restore']


def test_ctrl_c_key_reaches_sigint_even_after_ordered_exit_starts(runtime, monkeypatch):
    from hfmodem.ui import textual_chat
    calls = []
    monkeypatch.setattr(runtime.launch.os, 'kill', lambda pid, sig: calls.append((pid, sig)))

    async def exit_chat(pilot):
        pilot.app._exiting_chat = True
        await pilot.press('ctrl+c', 'ctrl+c')
        assert calls == [(os.getpid(), runtime.launch.signal.SIGINT)] * 2
        pilot.app._exiting_chat = False
        await pilot.press('ctrl+q')

    class App(runtime.app_class):
        def run(self):
            return super().run(headless=True, size=(80, 24), auto_pilot=exit_chat)

    monkeypatch.setattr(textual_chat, 'ChatApp', App)
    runtime.launch.run(runtime.args, runtime.rig, runtime.admission)


def test_real_textual_print_capture_preserves_the_stdout_pipe():
    import asyncio
    import sys
    from hfmodem.contact.chat_controller import Controller
    from hfmodem.contact.chat_fake import FakeSession
    from hfmodem.ui.textual_chat import ChatApp

    async def check():
        fake = FakeSession()
        controller = Controller(fake, clock=fake.clock)
        pipe, errors, terminal = io.StringIO(), io.StringIO(), io.StringIO()
        app = ChatApp(controller, terminal=terminal, stdout=pipe, stderr=errors, title='Kestrel chat')
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            print('BAND 20 m 14105000 Hz', flush=True)
            print('IDENT report', flush=True)
            print('Contact recording save failed: failed', flush=True)
            print('diagnostic', file=sys.stderr, flush=True)
        assert pipe.getvalue() == ('BAND 20 m 14105000 Hz\nIDENT report\n'
                                   'Contact recording save failed: failed\n')
        assert errors.getvalue() == 'diagnostic\n'
        assert terminal.getvalue() == ''
        assert controller.join(0)
    asyncio.run(check())


def test_ordinary_runtime_print_is_line_buffered_before_the_owner_starts(runtime, monkeypatch):
    from hfmodem.ui import textual_chat

    read_fd, write_fd = os.pipe()
    original_session = runtime.launch.Session

    def session(*args, **kwargs):
        value = original_session(*args, **kwargs)
        monitor = value.monitor

        def start():
            monitor()
            print('ordinary runtime message')

        value.monitor = start
        return value

    class App(runtime.app_class):
        async def on_mount(self):
            await super().on_mount()
            assert runtime.real_select([read_fd], [], [], 1)[0]
            received = os.read(read_fd, 65536)
            assert received.endswith(b'ordinary runtime message\n'), received
            print('ordinary print during capture')
            assert runtime.real_select([read_fd], [], [], 1)[0]
            assert os.read(read_fd, 65536) == b'ordinary print during capture\n'
            self.action_ordered_exit()

        def run(self):
            return super().run(headless=True, size=(80, 24))

    session.COMMANDS = original_session.COMMANDS
    monkeypatch.setattr(runtime.launch, 'Session', session)
    monkeypatch.setattr(textual_chat, 'ChatApp', App)
    try:
        with os.fdopen(write_fd, 'w') as evidence:
            assert not evidence.line_buffering
            with monkeypatch.context() as patch:
                patch.setattr(sys, 'stdout', evidence)
                runtime.launch.run(runtime.args, runtime.rig, runtime.admission)
    finally:
        os.close(read_fd)


def test_textual_driver_draws_and_senses_size_on_the_injected_tty(monkeypatch):
    from hfmodem.ui import textual_chat
    calls = []
    terminal = SimpleNamespace(fileno=lambda: 97)
    monkeypatch.setattr(textual_chat.LinuxDriver, '__init__', lambda self, app, **kwargs: None)
    driver = textual_chat.TTYDriver(SimpleNamespace(chat_terminal=terminal, interrupt=None))
    monkeypatch.setattr(textual_chat, 'os', SimpleNamespace(
        get_terminal_size=lambda fd: calls.append(fd) or (80, 24)))
    assert driver._file is terminal
    assert driver.fileno == 97 and driver.input_tty
    assert driver._get_terminal_size() == (80, 24)
    assert calls == [97]


@pytest.mark.parametrize('stage', ('shutdown', 'wait', 'join', 'input-join', 'driver-stop',
                                 'termios-reset', 'reset-output'))
@pytest.mark.parametrize('exit_type', (SystemExit, KeyboardInterrupt))
@pytest.mark.parametrize('signal_name', ('SIGINT', 'SIGTERM', 'SIGHUP'))
def test_panic_signals_cannot_unwind_teardown_or_terminal_reset(
        monkeypatch, stage, exit_type, signal_name):
    import asyncio
    import concurrent.futures._base as futures
    import signal
    import termios
    from threading import Event
    from hfmodem.contact.chat_controller import Controller
    from hfmodem.contact.chat_fake import FakeSession
    from hfmodem.ui import textual_chat

    master, slave = os.openpty()
    attrs = termios.tcgetattr(slave)
    stream = open(slave, 'w', buffering=1, closefd=False)
    controller = Controller(FakeSession())
    controller.ready.result(3)
    app = textual_chat.ChatApp(controller, terminal=stream)
    calls = []
    original = exit_type(1)

    def panic(signum, frame):
        calls.append('panic')
        raise original

    signum = getattr(signal, signal_name)
    previous = signal.signal(signum, panic)
    shutdown, join = controller.shutdown, controller.join
    stop = textual_chat.LinuxDriver.stop_application_mode
    setattrs = termios.tcsetattr
    write = stream.write
    injected = False
    stopping = False

    def inject(at):
        nonlocal injected
        if stage == at and not injected:
            injected = True
            signal.raise_signal(signum)
            assert calls == ['panic']

    def begin():
        with controller._condition:
            inject('shutdown')
            completion = shutdown()
        done = completion.done

        def poll():
            with completion._condition:
                inject('wait')
                return done()

        monkeypatch.setattr(completion, 'done', poll)
        return completion

    def joined(timeout):
        assert timeout == 0
        inject('join')
        return join(timeout)

    def restore(driver):
        nonlocal stopping
        assert join(0)
        stopping = True
        inject('driver-stop')
        return stop(driver)

    def reset(fd, when, value):
        if stopping:
            inject('termios-reset')
        return setattrs(fd, when, value)

    def forbidden_wait(self):
        raise AssertionError('teardown must not acquire wait() completion locks')

    def reset_output(data):
        if data.startswith('\x1b[?2004l\x1b[?7h'):
            inject('reset-output')
        return write(data)

    monkeypatch.setattr(controller, 'shutdown', begin)
    monkeypatch.setattr(controller, 'join', joined)
    monkeypatch.setattr(futures._AcquireFutures, '__enter__', forbidden_wait)
    monkeypatch.setattr(textual_chat.LinuxDriver, 'stop_application_mode', restore)
    monkeypatch.setattr(termios, 'tcsetattr', reset)
    monkeypatch.setattr(stream, 'write', reset_output)

    async def check():
        driver = textual_chat.TTYDriver(app)
        release_input = Event()
        if stage == 'input-join':
            run_input = driver._run_input_thread

            def input_thread():
                run_input()
                release_input.wait(3)

            monkeypatch.setattr(driver, '_run_input_thread', input_thread)
        driver.start_application_mode()
        if stage == 'input-join':
            input_join = driver._key_thread.join

            def join_input(timeout):
                assert timeout == 0
                inject('input-join')
                release_input.set()
                return input_join(timeout)

            monkeypatch.setattr(driver._key_thread, 'join', join_input)
        assert termios.tcgetattr(slave)[3] & termios.ISIG
        with pytest.raises(exit_type) as caught:
            driver.stop_application_mode()
        assert caught.value is original
        assert injected and calls == ['panic']
        assert join(0)
        assert_terminal_restored(slave, attrs)
        driver.stop_application_mode()
        driver.close()

    try:
        with pty_screen(master) as screen:
            asyncio.run(check())
        reset_bytes = b'\x1b[?1049l\x1b[?25h\x1b[?1004l'
        assert screen.count(reset_bytes) == 1
        assert screen.endswith(b'\x1b[?25h\x1b[?1004l')
        assert b'\x1b[<u' not in screen
        assert b'\x1b[>25u' not in screen
    finally:
        signal.signal(signum, previous)
        shutdown()
        assert join(3)
        termios.tcsetattr(slave, termios.TCSANOW, attrs)
        stream.close()
        os.close(slave)
        os.close(master)


def test_real_app_exits_after_owner_completion_without_a_lifecycle_event():
    import asyncio
    from hfmodem.contact.chat_controller import Controller
    from hfmodem.contact.chat_fake import FakeSession
    from hfmodem.ui.textual_chat import ChatApp

    async def check():
        fake = FakeSession()
        controller = Controller(fake, clock=fake.clock)
        app = ChatApp(controller, exit_on_failure=True)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._subscription.close()
            await asyncio.wrap_future(controller.shutdown())
            assert controller.join(1)
            app.collect_updates()
            assert app._exit
    asyncio.run(check())


@pytest.mark.parametrize('signal_name', ('SIGINT', 'SIGTERM', 'SIGHUP'))
def test_panic_signal_before_app_start_is_delivered_to_the_exit_protocol(signal_name):
    import signal
    from hfmodem.contact.chat_controller import Controller
    from hfmodem.contact.chat_fake import FakeSession
    from hfmodem.ui.textual_chat import ChatApp, defer_interrupts

    controller = Controller(FakeSession())
    signum = getattr(signal, signal_name)
    app = ChatApp(controller, interrupt=lambda: signal.raise_signal(signum))
    original = SystemExit(1)
    calls = []

    def panic(signum, frame):
        calls.append('panic')
        raise original

    previous = signal.signal(signum, panic)
    try:
        with pytest.raises(SystemExit) as caught:
            with defer_interrupts():
                signal.raise_signal(signum)
                assert calls == ['panic']
                app.run(headless=True, size=(80, 24))
                assert app._exit and app.failure is original
                assert controller.join(0)
        assert caught.value is original
        assert signal.getsignal(signum) is panic
    finally:
        signal.signal(signum, previous)
        controller.shutdown().result(3)
        assert controller.join(3)


def test_textual_session_say_keeps_console_log_without_duplicate_notices():
    from hfmodem.contact.chat_session import SessionAdapter
    from hfmodem.tests.contact.test_varac import Desk
    from varac_chat import TextualOutput, logging_say
    desk = Desk()
    output, console = TextualOutput(), io.StringIO()
    desk.session.say = logging_say(output, console)
    port = SessionAdapter(desk.session)
    output.attach(port)
    desk.session.say('session line')
    assert console.getvalue().endswith(' session line\n')
    assert [payload.text for payload, _, _ in port._queued] == ['session line']
