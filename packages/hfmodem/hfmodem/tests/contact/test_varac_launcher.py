# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import re
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.contact.varac import Event, HostError, Identification, Session
from hfmodem.core.capture_identity import _absolute


@pytest.fixture
def launch(tools):
    import varac_chat
    return varac_chat


def arguments(launch, tmp_path, *extra):
    return launch.parser().parse_args(['--mycall', 'W9SSJ', '--band', '20', '--session', str(tmp_path),
        '--expect-model', 'fake', '--ptt-device', 'unused', '--tx-device', 'unused',
        '--rx-device', 'unused', *extra])


def test_the_launcher_takes_a_band_a_bandwidth_and_a_profile(launch, tmp_path):
    args = arguments(launch, tmp_path)
    assert args.band == 20 and args.bw == 500
    assert arguments(launch, tmp_path, '--dial', '14106000').dial == 14_106_000
    launch.validate(arguments(launch, tmp_path, '--bw', '2750', '--name', 'Saul', '--services', 'VE'))
    with pytest.raises(ValueError):
        launch.validate(arguments(launch, tmp_path, '--name', 'Saul><DISC>'))
    for words in (['--band', '99'], ['--bw', '1000'], ['--action', 'listen'], ['--slot', '1'],
                  ['--execute']):
        with pytest.raises(SystemExit):
            arguments(launch, tmp_path, *words)


@pytest.mark.parametrize('state', [dict(frequency_hz=None, mode='PKTUSB', passband_hz=3000),
    dict(frequency_hz=14_106_000, mode='PKTUSB', passband_hz=3000),
    dict(frequency_hz=14_105_000, mode='LSB', passband_hz=3000),
    dict(frequency_hz=14_105_000, mode='PKTUSB', passband_hz=200)])
def test_tuning_requires_readback(launch, state):
    rig = SimpleNamespace(keyed=False, set_freq=lambda hz: None, set_mode=lambda *a: None,
                          state=lambda: state)
    with pytest.raises(HostError, match='QSY did not take'):
        launch.tune(rig, 14_105_000, 500)


@contextmanager
def fake_rigctld(passband):
    """A rigctld that takes every set and answers the dial and mode it was last given."""
    server = socket.create_server(('127.0.0.1', 0))
    heard, state = [], dict(freq='7105000')

    def serve():
        while True:
            try:
                client, _ = server.accept()
            except OSError:
                return
            with client:
                line = client.recv(4096).decode()
                heard.append(line.strip())
                if line.startswith('F '):
                    state['freq'] = line.split()[1]
                reply = (f'get_freq:;Frequency: {state["freq"]};RPRT 0\n' if line == ';f\n' else
                         f'get_mode:;Mode: PKTUSB;Passband: {passband};RPRT 0\n' if line == ';m\n'
                         else 'RPRT 0\n')
                client.sendall(reply.encode())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f'127.0.0.1:{server.getsockname()[1]}', heard
    finally:
        server.close()


@pytest.mark.parametrize('rig,mode', [('ftdx10', 'M PKTUSB -1'), ('ft891', 'M PKTUSB 3000')])
def test_a_qsy_leaves_a_profile_owned_width_and_roofing_alone(launch, monkeypatch, rig, mode):
    # 10-22: `M PKTUSB 3000` on every QSY left the FTDX10 at WIDTH 018 and R. FIL 3 kHz
    # after tune had verified 4000 and 12 kHz.
    from rig_client import Rig
    monkeypatch.setenv('HFMODEM_RIG', rig)
    with fake_rigctld(4000) as (address, heard):
        cat = Rig(address, False, lambda message: None)
        try:
            launch.tune(cat, 7_104_650, 500)
        finally:
            cat.shutdown()
    assert heard[:2] == ['F 7104650', mode]


def test_keyed_rig_cannot_be_tuned(launch):
    with pytest.raises(HostError, match='keyed'):
        launch.tune(SimpleNamespace(keyed=True), 14_105_000, 500)


@pytest.mark.parametrize('error', [ConnectionResetError('reset twice'), ConnectionRefusedError(),
                                   BrokenPipeError()])
def test_a_cat_failure_is_a_host_error_the_session_reports(launch, error):
    def fail(hz):
        raise error
    rig = SimpleNamespace(keyed=False, set_freq=fail)
    with pytest.raises(HostError, match='CAT failed') as raised:
        launch.tune(rig, 14_105_000, 500)
    assert raised.value.__cause__ is error


@pytest.mark.parametrize('passband,warned', [(2400, True), (3000, False)])
def test_a_passband_narrower_than_the_session_warns_and_tunes(launch, passband, warned):
    said = []
    rig = SimpleNamespace(keyed=False, set_freq=lambda hz: None, set_mode=lambda *a: None,
                          state=lambda: dict(frequency_hz=14_105_000, mode='PKTUSB', passband_hz=passband))
    launch.tune(rig, 14_105_000, 2750, said.append)
    assert said == ([f'warning: the rig passband reads {passband} Hz, narrower than the 2750 Hz session']
                    if warned else [])


def test_host_start_failure_releases_transport(launch):
    stopped = []
    transport = SimpleNamespace(stop=lambda: stopped.append('audio'))
    runtime = SimpleNamespace(modem=None, modem_factory=lambda: None)
    def fail(**kwargs):
        assert kwargs['cmd_port'] == kwargs['data_port'] == 0
        assert kwargs['host'] == '127.0.0.1'
        raise OSError('port unavailable')
    with pytest.raises(OSError, match='unavailable'):
        with launch.Host(transport, runtime_factory=lambda _: runtime, server_factory=fail):
            pytest.fail('started')
    assert stopped == ['audio']


def test_host_worker_failure_reaches_client(launch):
    def fail():
        raise RuntimeError('decoder stopped')
    runtime = SimpleNamespace(modem=None, step=fail)
    host = launch.Host(None, runtime_factory=lambda _: runtime)
    host._run()
    with pytest.raises(HostError, match='decoder stopped'):
        host.check()


def test_chat_host_arms_only_between_runtime_steps(launch):
    rig = SimpleNamespace(armed=False)
    host = chat_host(launch, SimpleNamespace(rig=rig))
    arming = threading.Thread(target=host.arm, args=(True,))
    with host.lock:
        arming.start()
        arming.join(.2)
        assert arming.is_alive() and rig.armed is False
    arming.join(2)
    assert rig.armed is True


def test_chat_attach_before_the_first_arm_keeps_the_host_running(launch, tools, monkeypatch):
    # Windows rig proof 2026-11-06: the standing modem retired at attach, before
    # the session first armed, and the host stopped on an unconfirmed unkey.
    import kestrel_connect
    import rig_client
    from hfmodem.contact.varac import Client
    from hfmodem.tests.kestrel.test_kiss_hostless import Transport
    from hfmodem.tests.kestrel.test_unkey_fence import Keyer
    monkeypatch.setattr(rig_client, 'LineKeyer', Keyer)
    rig = rig_client.Rig(None, False, lambda _: None, ptt_device='/dev/fake', line_ptt=True)

    class Audio(Transport):
        unkey_confirmed = kestrel_connect.AudioVaraIO.unkey_confirmed
        _timing_key_active = False
        start = stop = lambda self: None

    audio = Audio()
    audio.rig = rig
    try:
        with launch.Host(audio) as host:
            def ready(sockets, timeout):
                host.check()
                return select.select(sockets, [], [], timeout)[0] if sockets else []
            with Client.open(command_port=host.server.cmd_port, data_port=host.server.data_port,
                             event=lambda item: None, ready=ready) as client:
                client.initialize('W9SSJ', 500)
                for _ in range(100):
                    if host.runtime._standing is None:
                        break
                    time.sleep(.01)
                assert host.runtime._standing is None
                host.arm(True)
                client.listen()
                host.check()
    finally:
        rig.retire()


def test_chat_host_reports_cq_frames_while_not_listening(launch):
    from hfmodem.kestrel.host.server import VaraServer
    server = launch.chat_server()
    assert issubclass(server, VaraServer)
    sent = []
    session = server.session_class(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    session._write_event = lambda kind, value: sent.append((kind, value))
    assert not session.listen_cq and not session.listening
    session.modem_cqframe('K7QB-2', '500', 6.04)
    assert sent == [('command', 'BITRATE (1)  18 bps RX'), ('command', 'SN 6.0'), ('command', 'CQFRAME K7QB-2 500')]


@pytest.mark.parametrize('bw', [None, 500, 2300, 2750])
@pytest.mark.parametrize('station_settings', [
    None, ('unregulated', False, 540), ('unregulated', True, 0), ('part97', True, None)])
def test_session_on_fakes_stops_host_before_retiring_keyer(launch, monkeypatch, tmp_path, bw, station_settings):
    import kestrel_connect
    from hfmodem.contact import varac
    original_open = Path.open
    def locale_open(path, mode='r', buffering=-1, encoding=None, *args, **kwargs):
        if 'b' not in mode and encoding in (None, 'locale'):
            encoding = 'cp1252'
        return original_open(path, mode, buffering, encoding, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', locale_open)
    monkeypatch.setattr(varac.time, "time", lambda: 1700000000)
    from hfmodem.core import config
    order, events = [], []
    if station_settings is not None:
        example = Path(__file__).resolve().parents[5] / 'examples' / 'station.toml'
        text = example.read_text(encoding='utf-8')
        if station_settings[0] != 'part97':
            text = text.replace('regulatory = "part97"', 'regulatory = "unregulated"\nbecause = "fixture"')
            text = text.replace('licence    = "technician"', '')
        text = text.replace('closing_cwid  = true',
                            f'closing_cwid  = {str(station_settings[1]).lower()}')
        interval = station_settings[2]
        text = re.sub(r'(?m)^id_interval_s = 540 .*$',
                      '' if interval is None else f'id_interval_s = {interval}', text)
        station_path = tmp_path / 'station.toml'
        station_path.write_text(text, encoding='utf-8')
        monkeypatch.setenv(config.STATION_ENV, str(station_path))
    args = arguments(launch, tmp_path, '--mycall', 'X-W9SSJ', '--idcall', 'W9SSJ',
                     *(['--bw', str(bw)] if bw else []))
    launch.validate(args)
    monkeypatch.setattr(launch, 'tune', lambda rig, hz, *_: order.append('tune ' + str(hz)))
    rig = SimpleNamespace(retired=False, shutdown=lambda: order.append('unkey'), armed=False, power_w=lambda w=None: None,
                          line_state=lambda: False)
    admission = SimpleNamespace(opened=lambda: order.append('admitted'))
    class Tty(io.StringIO):
        def close(self):
            pass
    tty = Tty()
    monkeypatch.setattr(launch, 'line_terminal', lambda held, history, ui: tty)
    monkeypatch.setattr(config, 'tx_drive', lambda: .4)
    audio = SimpleNamespace(taps=[])
    def make_audio(tx, rx, amplitude, rig_arg, **kwargs):
        assert amplitude == .4 and rig_arg is rig
        assert kwargs['record'] == tmp_path / 'receive.wav'
        assert kwargs['offset_hz'] == 0
        return audio
    monkeypatch.setattr(kestrel_connect, 'AudioVaraIO', make_audio)

    @contextmanager
    def host(transport):
        assert transport is audio
        try:
            yield SimpleNamespace(server=SimpleNamespace(cmd_port=100, data_port=101),
                                  runtime=SimpleNamespace(), arm=lambda on: setattr(rig, 'armed', on),
                                  identify_steps=lambda call, bw, after_peer, say:
                                      order.append(f'identify {call} BW{bw} after_peer={after_peer}')
                                      or Identification(True, 1))
        finally:
            order.append('host closed')
    monkeypatch.setattr(launch, 'Host', host)

    class Client:
        transmission_count = 0
        sequence = 0
        link_generation = 0
        mycall = 'W9SSJ'
        ptt = pending = connected = closed = False
        snr = 11
        busy = None
        clock = staticmethod(lambda: 0.0)
        def initialize(self, call, bandwidth):
            self.bandwidth = bandwidth
            order.append('initialize')
            order.append(f'BW{bandwidth}')
            events[0](Event('text', '<UNKNOWN>café'))
        def poll(self, seconds):
            raise KeyboardInterrupt
        def request(self, command):
            order.append(command)
        def listen(self):
            order.append('LISTEN CQ')
            self.event(Event('host', 'CQFRAME WI9YZA-15 500'))
    client = Client()
    @contextmanager
    def open_client(**kwargs):
        assert kwargs['command_port'] == 100 and kwargs['data_port'] == 101
        events.append(kwargs['event'])
        client.event = kwargs['event']
        try:
            yield client
        finally:
            order.append('client closed')
    monkeypatch.setattr(launch.Client, 'open', open_client)
    def terminal(session, tty):
        order.append('terminal')
        assert session.auto and rig.armed is True
        # An unset interval is the station default, 540 s, under any profile.
        interval, closing = (540, True) if station_settings is None else (
            540 if station_settings[2] is None else station_settings[2], station_settings[1])
        assert (session.id_interval_s, session.closing_cwid) == (interval, closing)
        client.transmission_count += 1
        client.event(Event('host', 'PTT ON'))
        list(session._identify_steps())
        raise KeyboardInterrupt
    monkeypatch.setattr(launch, 'terminal', terminal)
    with pytest.raises(KeyboardInterrupt):
        launch.run(args, rig, admission)
    assert order == ['admitted', 'initialize', f'BW{bw or 500}', 'LISTEN CQ', 'terminal',
                     f'identify W9SSJ BW{bw or 500} after_peer=False',
                     'client closed', 'host closed', 'unkey']
    transcript = (tmp_path / 'host.jsonl').read_text(encoding='utf-8')
    assert '<UNKNOWN>café' in transcript and 'CQFRAME WI9YZA-15 500' in transcript
    screen = tty.getvalue().splitlines()
    assert screen.pop(0) == f'session records: {tmp_path}'
    assert screen.pop(0) == ('VMail parking on; relay notices on; 0 pending; expiry 30 days; '
                             'limits 100/destination, 1000 total; '
                             'notices deferred: /band ready required each session')
    assert screen.pop(0) == 'PSKReporter off'
    assert screen[0].startswith('commands: /status, /help [bbs|email], /answer N, /cq [SLOT] [repeat], /call CALL [SLOT|cf]')
    assert screen[1:] == ['  1  WI9YZA   slot 15  14108.750 kHz   500 Hz  SN +11.0  2023-11-14 22:13:20Z',
                          f'monitoring 14105.000 kHz at {bw or 500} Hz, auto-answer on',
                          'Passive transcripts are logged; /follow cf shows calling-frequency activity, '
                          '/follow all shows the current passband, /follow off hides it. '
                          '/follow SLOT or KHZ retunes if needed; /follow cf returns to CF.']
    recorded = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [(r['kind'], r.get('call') or r.get('state')) for r in recorded] == [
        ('heard', 'WI9YZA'), ('state', 'monitoring')]
    assert all(r['utc'].endswith('+00:00') for r in recorded)
    assert (tmp_path / 'contacts.jsonl').read_text(encoding='utf-8') == ''


def test_kestrel_entry_dispatches_chat_without_opening_audio(launch, monkeypatch):
    import kestrel_host
    called = []
    monkeypatch.setattr(launch, 'main', lambda argv: called.append(argv) or 12)
    assert kestrel_host.main(['--chat', '--help']) == 12
    assert called == [['--help']]


def test_terminal_feeds_lines_between_ticks_and_says_goodbye_at_the_end(launch, monkeypatch):
    order = []
    lines = iter(['/answer 2\n', None, 'hello\n', '/quit\n'])
    class Terminal:
        def readline(self):
            return next(lines)
    class Fake:
        def tick(self):
            order.append('tick')
        def command(self, line):
            order.append(line)
            return line.rstrip('\n') != '/quit'
        def quit(self):
            order.append('quit')
    launch.terminal(Fake(), Terminal())
    assert order == ['tick', '/answer 2\n', 'tick', 'tick', 'hello\n', 'tick', '/quit\n', 'quit']


def test_terminal_eof_ends_the_session(launch, monkeypatch):
    order = []
    fake = SimpleNamespace(tick=lambda: None, command=lambda line: pytest.fail('EOF is not a command'),
                           quit=lambda: order.append('quit'))
    launch.terminal(fake, SimpleNamespace(readline=lambda: ''))
    assert order == ['quit']


PTY_CHILD = '''
import os, sys
from unittest.mock import patch
assert os.ttyname(0) == sys.argv[2]
open_file = os.open
def open_terminal(path, *args, **kwargs):
    return open_file(sys.argv[2] if path == '/dev/tty' else path, *args, **kwargs)
patch('varac_chat.os.open', open_terminal).start()
'''


TERMINAL_CHILD = PTY_CHILD + '''
import sys, time
from contextlib import ExitStack
import varac_chat
with ExitStack() as held:
    tty = varac_chat.Terminal(held, sys.argv[1])
    print('to arm.log')
    while (line := tty.readline()) != '':
        if line is None:
            time.sleep(.01)
        elif line == 'later\\n':
            time.sleep(.5)
            tty.write('heard\\n')
        else:
            tty.write(f'got {line!r}\\n')
    tty.write('EOF ends it\\n')
'''


def _terminal_env(tools):
    path = [os.environ.get('PYTHONPATH'), *sys.path, str(tools)]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, path)))


def test_terminal_children_keep_the_guard(tools, tmp_path, monkeypatch):
    work = tmp_path / 'child-guard'
    work.mkdir()
    monkeypatch.setenv('HERMETIC_WORK', str(work))
    inherited = tmp_path / 'inherited'
    inherited.mkdir()
    (inherited / 'sitecustomize.py').write_text('', encoding='utf-8')
    monkeypatch.syspath_prepend(str(inherited))
    code = '''
import sys, varac_chat
assert sys.modules['hermetic_guard'].armed
try:
    sys.audit('open', 'COM987654', 'r', 0)
except PermissionError as exc:
    assert 'serial device is not owned by this test' in str(exc), str(exc)
else:
    raise AssertionError('device audit event was not refused')
print(varac_chat.__file__)
'''
    result = subprocess.run([sys.executable, '-c', code], env=_terminal_env(tools),
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (tools / 'varac_chat.py').resolve()
    assert 'serial device is not owned by this test' in (work / 'trips').read_text(encoding='utf-8')


@pytest.mark.posix_only('requires a controlling POSIX pty with fcntl and termios')
@pytest.mark.realtime
def test_the_terminal_edits_recalls_and_keeps_the_typed_line_under_screen_lines(tools, tmp_path):
    import fcntl, os, pty, select, subprocess, sys, termios, time
    history = tmp_path / 'history'

    def session(*steps):
        master, slave = pty.openpty()
        def own_the_tty():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        child = subprocess.Popen([sys.executable, '-c', TERMINAL_CHILD, str(history), os.ttyname(slave)], stdin=slave,
                                 stdout=subprocess.PIPE, preexec_fn=own_the_tty,
                                 env=_terminal_env(tools))
        os.close(slave)
        screen, seen = b'', 0
        def until(text):
            nonlocal screen, seen
            alternatives = text if isinstance(text, tuple) else (text,)
            deadline = time.monotonic() + 10
            while True:
                found = [(screen.find(candidate, seen), candidate) for candidate in alternatives]
                found = [(index, candidate) for index, candidate in found if index >= 0]
                if found:
                    index, candidate = min(found)
                    seen = index + len(candidate)
                    return
                assert time.monotonic() < deadline, screen
                if select.select([master], [], [], .1)[0]:
                    screen += os.read(master, 4096)
        for wait, keys in steps:
            until(wait)
            time.sleep(.2)  # a ^D typed ahead between lines meets the cooked tty and is lost
            os.write(master, keys)
        until(b'EOF ends it')
        assert child.wait(10) == 0
        os.close(master)
        return screen, child.stdout.read()

    screen, log = session((b'> ', b'later\r'), (b'later\r\n', b'abc'), (b'heard\r\n> abc', b'\x01X\r'),
                          (b"got 'Xabc\\n'", b'\x1b[A\r'), (b"got 'Xabc\\n'", b'\x04'))
    assert log == b'to arm.log\n'
    assert history.read_text(encoding='utf-8').splitlines()[-2:] == ['later', 'Xabc']
    screen, _ = session((b'> ', b'\x1b[A\r'), (b"got 'Xabc\\n'", b'\x12'),
                        ((b'bck:', b'(reverse-i-search)'), b'lat'),
                        (b'later', b'\x05\r'), (b'heard', b'\x04'))
    assert history.read_text(encoding='utf-8').splitlines()[-1] == 'later'


HELD_BUSY_CHILD = PTY_CHILD + '''
import os, queue, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import kestrel_connect, varac_chat
from hfmodem.core import config

session = Path(sys.argv[1]) / 'sessions' / 's'
session.mkdir(parents=True)
os.environ['HFMODEM_RECORDING'] = 'everything'
rec = kestrel_connect.Recorder(session / 'receive.wav')
rec.write(np.zeros(kestrel_connect.FS, np.float32))
rec.after_pending(lambda: None).result(timeout=5)

class Channel(kestrel_connect._DryIO):
    """A channel the host hears busy throughout, over that recording."""
    receiving = True
    def __init__(self, rig):
        super().__init__()
        self.rig = rig
        self.rec = rec
        self.taps = []
    def log(self, msg): pass
    def stop(self): self.rec.close()

kestrel_connect.AudioVaraIO = lambda tx, rx, drive, rig, **k: Channel(rig)
config.tx_drive = lambda: .4
varac_chat.tune = lambda *a, **k: None
args = varac_chat.parser().parse_args(['--mycall', 'W9SSJ', '--band', '40', '--session', str(session),
    '--expect-model', 'fake', '--ptt-device', 'unused', '--tx-device', 'unused', '--rx-device', 'unused'])
rig = SimpleNamespace(retired=False, shutdown=lambda: None, armed=False, line_state=lambda: None, power_w=lambda w=None: None,
                      meter_reports=queue.SimpleQueue(), keyed=False)
varac_chat.run(args, rig, SimpleNamespace(opened=lambda: None))
'''


@pytest.fixture
def held_busy(tools, tmp_path):
    """varac_chat on a real host and terminal, its channel busy from the start."""
    import fcntl, os, pty, select, subprocess, sys, termios, time
    master, slave = pty.openpty()
    def own_the_tty():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    child = subprocess.Popen([sys.executable, '-c', HELD_BUSY_CHILD, str(tmp_path), os.ttyname(slave)], stdin=slave,
                             stdout=subprocess.DEVNULL, preexec_fn=own_the_tty,
                             env=_terminal_env(tools))
    os.close(slave)
    screen = bytearray()
    def until(text, seconds):
        deadline = time.monotonic() + seconds
        while text not in screen:
            assert time.monotonic() < deadline, bytes(screen)
            if select.select([master], [], [], .05)[0]:
                screen.extend(os.read(master, 4096))
    def busy():
        path = tmp_path / 'sessions' / 's' / 'host.jsonl'
        return path.exists() and '"BUSY ON"' in path.read_text(encoding='utf-8')
    until(b'monitoring', 10)
    deadline = time.monotonic() + 5
    while not busy():
        assert time.monotonic() < deadline
        time.sleep(.05)
    def typed(keys):
        screen.clear()
        os.write(master, keys)
    try:
        yield SimpleNamespace(typed=typed, until=until, child=child, session=tmp_path / 'sessions' / 's')
    finally:
        child.kill()
        # Release the controlling PTY before reaping: on macOS a killed reader
        # can remain in terminal teardown while the master stays open.
        os.close(master)
        child.wait(timeout=10)


@pytest.mark.posix_only('requires a controlling POSIX pty with fcntl and termios')
@pytest.mark.realtime
def test_the_line_answers_within_a_second_while_the_host_holds_busy(held_busy):
    held_busy.typed(b'/help\r')
    held_busy.until(b'commands: /status, /help [bbs|email], /answer N', 1)


@pytest.mark.posix_only('requires a controlling POSIX pty with fcntl and termios')
@pytest.mark.realtime
def test_quit_returns_at_once_and_the_recording_is_stored_after_it(held_busy):
    # 20261023T155947666857Z: /quit after six hours, then /help and /? unanswered while the
    # 2.2 GB receive.wav went to FLAC, until a Ctrl-C left it a WAV.
    import time
    held_busy.typed(b'/quit\r')
    quit_at = time.monotonic()
    held_busy.until(b'recording everything: ', 1)
    assert held_busy.child.wait(2) == 0 and time.monotonic() - quit_at < 2
    [index] = held_busy.session.glob('receive-*/recording.json')
    [stored] = json.loads(index.read_text(encoding='utf-8'))['segments']
    assert stored['state'] == 'finalized' and stored['samples'] == 48000
    received = index.parent / stored['source']
    assert received.suffix == '.flac' and received.exists()
    assert not received.with_suffix('.wav').exists()


def test_the_session_commands_are_the_terminals_help_line():
    assert Session.COMMANDS == (
        '/status', '/help [bbs|email]', '/answer N', '/cq [SLOT] [repeat]',
        '/call CALL [SLOT|cf]', '/stop (or Enter)', '/heard', '/slots',
        '/follow all|cf|SLOT|KHZ|off', '/tag TAG [value]', '/qsy SLOT', '/bye [text]',
        '/clear', '/auto on|off', '/beacon [MINUTES|now|force|off]',
        '/beacon hop BANDS [every MINUTES]', '/beacon hop auto [BANDS] [every MINUTES]',
        '/beacon hop status', '/band N', '/band ready|unready',
        '/away [MESSAGE]', '/back', '/vmail [N | send SUBJECT | send CALL SUBJECT (offline) | park CALL SUBJECT | parked [N] | cancel N | parking on|off | relay on|off | notify CALL | pending | collect CALL | dismiss CALL]',
        '/email send TO_EMAIL REPLY_EMAIL SUBJECT', '/bbs list|get N|request N|capture',
        '/inquire heard', '/pskreporter [status|off]', '/all [cf|slot N|TEXT|status|off|clear]',
        '/verbose [on|off]', '/power [WATTS]', '/quit')


def rigctld(script):
    """A rigctld on loopback that answers each connection from ``script``; a None
    reply resets that connection unanswered, as rigctld did on 2026-10-22."""
    server = socket.create_server(('127.0.0.1', 0))
    heard = []
    def serve():
        for reply in script:
            peer, _ = server.accept()
            with peer:
                heard.append(peer.recv(4096).decode())
                if reply is None:
                    peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                else:
                    peer.sendall(reply.encode())
        server.close()
    threading.Thread(target=serve, daemon=True).start()
    return f'127.0.0.1:{server.getsockname()[1]}', heard


def test_slot_qsy_survives_rigctld_resetting_the_mode_set(launch, tools):
    from rig_client import Rig
    address, heard = rigctld(['RPRT 0\n', None, 'RPRT 0\n',
                              'get_freq:;Frequency: 3593500;RPRT 0\n',
                              'get_mode:;Mode: PKTUSB;Passband: 3000;RPRT 0\n'])
    said = []
    rig = Rig(address, False, said.append)
    try:
        launch.tune(rig, 3_593_500, 500)
    finally:
        rig.retire()
    assert heard == ['F 3593500\n', 'M PKTUSB 3000\n', 'M PKTUSB 3000\n', ';f\n', ';m\n']
    assert said == ["rigctld reset the connection on 'M PKTUSB 3000'; asking once more"]


def test_a_second_reset_still_reaches_the_caller(tools):
    from rig_client import Rig
    address, heard = rigctld([None, None])
    rig = Rig(address, False, lambda line: None)
    try:
        with pytest.raises(ConnectionResetError):
            rig.set_freq(3_593_500)
    finally:
        rig.retire()
    assert heard == ['F 3593500\n'] * 2


def test_disarming_records_the_lines_hand_back(tools, monkeypatch):
    import rig_client
    class Keyer:
        released_low = None
        def __init__(self, device, log):
            pass
        def arm(self):
            pass
        def key(self, on):
            return True
        def hand_back(self):
            self.released_low = True
    monkeypatch.setattr(rig_client, 'LineKeyer', Keyer)
    rig = rig_client.Rig(None, False, lambda line: None, ptt_device='fake', line_ptt=True)
    try:
        rig.armed = True
        assert rig.released_low is None
        rig.armed = False
        assert rig.released_low is True
    finally:
        rig.retire()


def keyable(rig_client, monkeypatch, address, said):
    class Keyer:
        released_low = None
        def __init__(self, device, log):
            pass
        def arm(self):
            pass
        def key(self, on, why=''):
            return True
        def hand_back(self):
            self.released_low = True
    monkeypatch.setattr(rig_client, 'LineKeyer', Keyer)
    rig = rig_client.Rig(address, False, said.append, ptt_device='fake', line_ptt=True)
    rig.armed = True
    return rig


def test_a_transmission_logs_its_peak_power_and_alc(tools, monkeypatch):
    import time
    import rig_client
    monkeypatch.setattr(rig_client, 'METER_PERIOD_S', 30.0)
    monkeypatch.setattr(rig_client, 'MIN_METERED_KEY_S', 0.0)
    address, heard = rigctld(['42.5\n', '0.31\n'])
    said = []
    rig = keyable(rig_client, monkeypatch, address, said)
    try:
        assert rig.key(True)
        deadline = time.monotonic() + 3
        while len(heard) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert rig.key(False)
        report = rig.meter_reports.get(timeout=3)
    finally:
        rig.shutdown()
    assert heard == ['l RFPOWER_METER_WATTS\n', 'l ALC\n']
    assert (report['peak_watts'], report['peak_alc'], report['reads'], report['unread']) == (42.5, 0.31, 1, None)
    assert [line for line in said if line.startswith('TX METERS')] == [
        f"TX METERS: peak 42.5 W, ALC 0.31 over 1 reads in {report['seconds']:.1f} s"]


# Real key edges must return within 50ms despite an asynchronous meter failure.
@pytest.mark.wall_clock
def test_an_unread_meter_is_logged_and_never_holds_the_key(tools, monkeypatch):
    import time
    import rig_client
    closed = socket.create_server(('127.0.0.1', 0))
    address = f'127.0.0.1:{closed.getsockname()[1]}'
    closed.close()
    said = []
    rig = keyable(rig_client, monkeypatch, address, said)
    try:
        started = time.monotonic()
        assert rig.key(True)
        assert time.monotonic() - started < 0.05
        time.sleep(0.2)
        started = time.monotonic()
        assert rig.key(False)
        assert time.monotonic() - started < 0.05
        report = rig.meter_reports.get(timeout=3)
    finally:
        rig.shutdown()
    assert report['peak_watts'] is None and report['peak_alc'] is None
    assert set(report['unread']) == {'RFPOWER_METER_WATTS', 'ALC'}
    assert any(line.startswith('TX METERS: peak power unread, ALC unread') for line in said)


@pytest.mark.parametrize('timeout', [0, .25])
def test_the_session_writes_each_transmissions_meters_to_its_events(launch, tmp_path, monkeypatch, timeout):
    import queue
    import kestrel_connect
    from hfmodem.core import config
    args = arguments(launch, tmp_path)
    reports = queue.SimpleQueue()
    reports.put(dict(peak_watts=38.0, peak_alc=0.4, reads=20, seconds=5.4, unread=None))
    rig = SimpleNamespace(retired=False, shutdown=lambda: None, armed=False, line_state=lambda: False, power_w=lambda w=None: None,
                          meter_reports=reports)
    monkeypatch.setattr(launch, 'tune', lambda *a: None)
    monkeypatch.setattr(launch, 'line_terminal', lambda held, history, ui: io.StringIO())
    waited = []
    monkeypatch.setattr(launch, 'time', SimpleNamespace(sleep=waited.append))
    monkeypatch.setattr(config, 'tx_drive', lambda: .4)
    monkeypatch.setattr(kestrel_connect, 'AudioVaraIO', lambda *a, **kw: SimpleNamespace(taps=[]))

    @contextmanager
    def host(transport):
        yield SimpleNamespace(server=SimpleNamespace(cmd_port=100, data_port=101), check=lambda: None,
                              arm=lambda on: setattr(rig, 'armed', on))
    monkeypatch.setattr(launch, 'Host', host)

    @contextmanager
    def open_client(**kwargs):
        assert kwargs['ready']([], timeout) == []
        raise KeyboardInterrupt
        yield
    monkeypatch.setattr(launch.Client, 'open', open_client)
    with pytest.raises(KeyboardInterrupt):
        launch.run(args, rig, SimpleNamespace(opened=lambda: None))
    recorded = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [(r['kind'], r['peak_watts'], r['peak_alc']) for r in recorded] == [('tx_meters', 38.0, 0.4)]
    assert waited == [timeout]


def test_a_closed_receive_recording_is_stored_when_close_returns(tools, tmp_path, monkeypatch):
    import numpy as np
    import kestrel_connect
    from hfmodem.core import wav
    monkeypatch.setenv('HFMODEM_RECORDING', 'everything')
    def unavailable(*args):
        raise OSError('encoder unavailable')
    monkeypatch.setattr(wav, '_flac', unavailable)
    rec = kestrel_connect.Recorder(tmp_path / 'receive.wav')
    rec.write(np.zeros(480, np.float32))
    rec.close()
    [segment] = wav.segments(rec.path)
    [stored] = json.loads((rec.session.directory / 'recording.json').read_text())['segments']
    assert rec.session.directory / stored['source'] == segment
    assert stored['state'] == 'finalized' and stored['samples'] == 480
    assert stored['sha256'] == wav._sha256(segment)
    assert segment.stat().st_size == 44 + 960


@pytest.mark.parametrize('where', ['tick', 'readline', 'command'])
def test_terminal_interrupt_runs_the_same_bounded_quit(launch, where):
    order = []
    def action(name, result=None):
        def run(*args):
            order.append(name)
            if name == where:
                raise KeyboardInterrupt
            return result
        return run
    session = SimpleNamespace(tick=action('tick'), command=action('command'),
                              quit=action('quit'))
    launch.terminal(session, SimpleNamespace(readline=action('readline', 'hello')))
    assert order[-1] == 'quit' and order.count('quit') == 1


def test_bbs_operator_commands_cross_terminal_and_scripted_peer(launch):
    """The actual terminal loop drives Session through fake transport sockets."""
    from hfmodem.contact import varac as v
    from hfmodem.tests.contact.test_varac_bbs import linked, LIST
    desk = linked()
    previous = desk.host.data.respond
    def peer(payload):
        previous(payload)
        if payload == b'5 <BLR>':
            reply = v.frame_message(LIST)
            desk.host.data.incoming.extend([reply[:7], reply[7:35], reply[35:]])
    desk.host.data.respond = peer
    class Operator:
        stage = 0
        reads = 0
        def readline(self):
            self.reads += 1
            assert self.reads < 100, 'terminal did not finish the scripted BBS exchange'
            if self.stage == 0:
                self.stage = 1
                return '/verbose on\n'
            if self.stage == 1:
                self.stage = 2
                return '/bbs list\n'
            if self.stage == 2 and desk.kinds('bbs_directory'):
                self.stage = 3
                return '/bye\n'
            if self.stage == 3 and not desk.host.client.connected:
                self.stage = 4
                return '/quit\n'
            return None
    launch.terminal(desk.session, Operator())
    assert desk.kinds('bbs_directory')[-1]['text'] == LIST
    assert any('BBS directory from OH8XAT: 2 files' in line for line in desk.said)
    assert any('TX queued (not yet acknowledged): <BLR>' in line for line in desk.said)
    assert any('RX OH8XAT: <BL:' in line for line in desk.said)
    assert desk.texts == ['5 <BLR>', '8 73<DISC>']
    assert 'DISCONNECT' in desk.sent
    assert not desk.host.client.connected


def test_terminal_help_identifies_kestrel_and_protocol_separately(launch):
    help_text = launch.parser().format_help()
    assert 'Kestrel chat terminal using the VarAC protocol' in help_text
    assert 'The VarAC session terminal' not in help_text


def test_host_command_rejection_reaches_existing_runtime_log(launch):
    logged = []
    host = launch.Host(SimpleNamespace(log=logged.append),
                       runtime_factory=lambda _: SimpleNamespace(modem=None))
    host._host_log('rejected', 'BEACONFORCE: contact close admission pending')
    host._host_log('command', 'unrelated wire logging')
    assert logged == ['host command rejected: BEACONFORCE: contact close admission pending']


def test_pskreporter_requires_explicit_flag_and_valid_locator(launch, tmp_path):
    assert not arguments(launch, tmp_path).pskreporter
    with pytest.raises(ValueError, match='callsign and Maidenhead'):
        launch.validate(arguments(launch, tmp_path, '--pskreporter'))
    args = arguments(launch, tmp_path, '--pskreporter', '--loc', 'EN61ab')
    launch.validate(args)
    assert args.pskreporter


def test_normal_exit_reports_final_udp_flush_without_using_explicit_off(launch):
    calls, lines, events = [], [], []
    class Reporter:
        def status(self):
            return {'submitted': 2}
        def close(self):
            calls.append('close')
            return dict(submitted=7, unsent_at_shutdown=1, inflight_at_shutdown=0)
        def stop(self):
            pytest.fail('normal shutdown used explicit off/discard')
    launch.close_reporter(Reporter(), lines.append,
                          lambda kind, **fields: events.append((kind, fields)))
    assert calls == ['close']
    assert '5 final UDP reports submitted' in lines[-1] and '1 unsent' in lines[-1]
    assert 'Server receipt is unconfirmed' in lines[-1]
    assert events[0][0] == 'pskreporter_shutdown'


def test_bbs_directory_and_disable_flags(launch, tmp_path):
    args = arguments(launch, tmp_path, '--bbs-directory', str(tmp_path / 'public'))
    assert args.bbs_directory == str(tmp_path / 'public') and not args.no_bbs
    assert arguments(launch, tmp_path, '--no-bbs').no_bbs


def test_interactive_bbs_flags_are_independent_of_native_files(launch, tmp_path):
    args = arguments(launch, tmp_path, '--bbs-config', str(tmp_path / 'bbs.toml'), '--no-bbs')
    assert args.bbs_config == tmp_path / 'bbs.toml'
    assert args.no_bbs and not args.no_bbs_service
    args = arguments(launch, tmp_path, '--no-bbs-service', '--bbs-directory', str(tmp_path))
    assert args.no_bbs_service and not args.no_bbs


def test_invalid_interactive_bbs_stops_before_claimed_radio_main(launch, tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'validate', lambda args: None)
    monkeypatch.setattr(launch, 'claimed_main', lambda *args: pytest.fail('radio path entered'))
    argv = ['--mycall', 'W9SSJ', '--band', '20', '--session', str(tmp_path),
            '--expect-model', 'fake', '--ptt-device', 'unused', '--tx-device', 'unused',
            '--rx-device', 'unused', '--bbs-config', str(tmp_path / 'missing.toml')]
    with pytest.raises(FileNotFoundError):
        launch.main(argv)


def test_native_bbs_page_resets_color_before_restoring_edited_line(launch):
    tty = object.__new__(launch.Terminal)
    tty.screen = io.StringIO()
    tty.editing = True
    tty.edit = SimpleNamespace(get_line_buffer=lambda: 'look')
    tty.write_bbs('\x1b[31mWest of House\x1b[0m')
    result = tty.screen.getvalue()
    assert result.startswith('\r\x1b[K\x1b[0m')
    assert result.endswith('\x1b[0m\n> look')
    assert 'CALL:' not in result


def test_station_bbs_is_opened_before_radio_and_disable_wins(launch, tmp_path, monkeypatch):
    from hfmodem.bbs import config as board_config, service
    from hfmodem.contact import bbs_server
    from contextlib import contextmanager
    monkeypatch.setattr(launch, 'validate', lambda args: None)
    board = tmp_path / 'board.toml'
    monkeypatch.setattr(board_config, 'configured_path', lambda override: board)
    monkeypatch.setattr(bbs_server, 'configured_directory', lambda _: None)
    calls = []
    sentinel = object()

    @contextmanager
    def opened(path, **kwargs):
        calls.append(path)
        yield sentinel

    monkeypatch.setattr(service, 'open_service', opened)
    monkeypatch.setattr(launch, 'claimed_main', lambda args, active: calls.append(active) or 0)
    argv = ['--mycall', 'W9SSJ', '--band', '20', '--session', str(tmp_path),
            '--expect-model', 'fake', '--ptt-device', 'unused', '--tx-device', 'unused',
            '--rx-device', 'unused']
    assert launch.main(argv) == 0
    assert calls == [board, sentinel]
    calls.clear()
    assert launch.main(argv + ['--no-bbs-service']) == 0
    assert calls == [None]


class Keyed:
    """An audio transport that keys and plays what it is given, after one data burst."""

    def __init__(self, receiving=()):
        self.burst_level = (.1, .5)
        self.bursts_sent = 0
        self.played, self.keyed = [], []
        self._receiving = iter(receiving)

    @property
    def receiving(self):
        return next(self._receiving, False)

    def key(self, on):
        self.keyed.append(on)

    def unkey_confirmed(self):
        return not self.keyed or self.keyed[-1] is False

    def tx(self, audio, drive, *, identification=False, cancelled=None):
        assert identification
        self.played.append(audio)
        self.bursts_sent += 1


def chat_host(launch, transport, step=lambda: None):
    return launch.Host(transport, runtime_factory=lambda _: SimpleNamespace(
        modem=SimpleNamespace(observer=SimpleNamespace(transmission_count=0)), step=step))


@pytest.mark.parametrize('bw', [500, 2300, 2750])
@pytest.mark.parametrize('stepped', [False, True])
def test_the_chat_host_keys_the_idcall_in_audio_cw_at_the_channel_centre(launch, tools, bw, stepped):
    import numpy as np
    import time
    from hfmodem.core import cwid
    from hfmodem.core.occupied import occupied_hz
    args = arguments(launch, None, '--mycall', 'X-W9SSJ', '--idcall', 'W9SSJ')
    launch.validate(args)
    assert args.identity.cw_call == 'W9SSJ'
    transport = Keyed()
    said = []
    host = chat_host(launch, transport)
    if stepped:
        steps = host.identify_steps(args.identity.cw_call, bw, say=said.append)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                result = done.value
                break
            time.sleep(.001)
    else:
        result = host.identify(args.identity.cw_call, bw, say=said.append)
    assert result == Identification(True, 0)
    assert said == [] and transport.keyed == [True, False]
    expected = cwid.audio('W9SSJ', fs=48000, tone_hz=sum(occupied_hz('vara', bw)) / 2)
    assert np.array_equal(transport.played[0], expected)


def test_an_identification_that_cannot_go_out_says_send_it_by_hand(launch, tools):
    transport = Keyed()
    transport.burst_level = None
    said = []
    assert chat_host(launch, transport).identify('W9SSJ', 500, say=said.append) == Identification(
        True, 0, succeeded=False)
    assert transport.played == [] and len(said) == 1
    assert said[0].startswith('  !! IDENTIFICATION NOT CONFIRMED (')
    assert said[0].endswith(') -- send W9SSJ by hand')


def test_closing_id_wait_is_bounded_and_a_clear_channel_keys_ours_at_once(launch, tools, monkeypatch):
    import time
    monkeypatch.setattr(launch, 'PEER_IDENT_GAP_S', 0.0)
    transport = Keyed(receiving=[True] * 3)
    chat_host(launch, transport).identify('W9SSJ', 500, after_peer=True, say=pytest.fail)
    assert next(transport._receiving, None) is None and transport.played
    monkeypatch.setattr(launch, 'PEER_IDENT_S', 0.2)
    transport = Keyed(receiving=iter(lambda: True, None))
    start = time.monotonic()
    chat_host(launch, transport).identify('W9SSJ', 500, after_peer=True, say=pytest.fail)
    assert time.monotonic() - start < 1 and transport.played
    monkeypatch.setattr(launch, 'PEER_IDENT_GAP_S', 5.0)
    transport = Keyed()
    start = time.monotonic()
    chat_host(launch, transport).identify('W9SSJ', 500, say=pytest.fail)
    assert time.monotonic() - start < 1 and transport.played


def test_the_modem_does_not_step_while_the_id_keys(launch, tools, monkeypatch):
    import kestrel_connect
    import time
    steps = []
    host = chat_host(launch, Keyed(), step=lambda: steps.append(time.sleep(.001)))
    host.thread.start()
    try:
        while len(steps) < 3:
            time.sleep(.001)
        def identify(io, call, bw, say, **kwargs):
            before = len(steps)
            time.sleep(.05)
            assert len(steps) == before
            io.key(False)
            return True
        monkeypatch.setattr(kestrel_connect, 'identify', identify)
        host.identify('W9SSJ', 500)
        ran = len(steps)
        while len(steps) == ran:
            time.sleep(.001)
    finally:
        host.stop.set()
        host.thread.join(5)
    assert host.error is None


def test_quit_during_ident_unkeys_within_quit_bound(launch, tools):
    import time
    from hfmodem.tests.contact.test_varac import answered

    playing = threading.Event()
    transport = Keyed()
    def tx(audio, drive, *, identification=False, cancelled=None):
        assert identification and cancelled is not None
        playing.set()
        assert cancelled.wait(5), 'quit did not cancel identification'
        raise InterruptedError('identification cancelled')
    transport.tx = tx
    host = chat_host(launch, transport)
    desk = answered()
    desk.session.keyer = lambda bw, peer: host.identify_steps('W9SSJ', bw, peer, desk.said.append)
    desk.hear('PTT ON', 'PTT OFF')
    desk.session.command('/bye')
    end = time.monotonic() + 2
    while not playing.is_set() and time.monotonic() < end:
        desk.tick()
        time.sleep(.001)
    assert playing.is_set() and transport.keyed == [True]
    start = time.monotonic()
    assert desk.session.command('/quit') is False
    desk.session.quit()
    assert time.monotonic() - start < 1
    assert transport.keyed == [True, False]
    assert any('send W9SSJ by hand' in line for line in desk.said)
    assert desk.sent[-1] == 'LISTEN OFF'
    assert desk.session.steps is None


def test_ident_has_a_deadline_even_when_audio_never_finishes(launch, tools, monkeypatch):
    import time
    monkeypatch.setattr(launch, 'IDENT_TIMEOUT_S', .05)
    transport = Keyed()
    said = []
    def tx(audio, drive, *, identification=False, cancelled=None):
        assert cancelled.wait(2)
        raise InterruptedError('identification cancelled')
    transport.tx = tx
    host = chat_host(launch, transport)
    start = time.monotonic()
    steps = host.identify_steps('W9SSJ', 500, say=said.append)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            assert done.value == Identification(True, 0, succeeded=False)
            break
        time.sleep(.001)
    assert time.monotonic() - start < 1
    assert transport.keyed == [True, False]
    assert len(said) == 1 and 'send W9SSJ by hand' in said[0]


def test_cancelled_host_identification_returns_failure(launch, tools):
    cancelled = threading.Event()
    cancelled.set()
    transport, said = Keyed(), []
    result = chat_host(launch, transport).identify('W9SSJ', 500, say=said.append, cancelled=cancelled)
    assert result == Identification(True, 0, succeeded=False)
    assert not transport.keyed and not transport.played
    assert len(said) == 1 and 'send W9SSJ by hand' in said[0]


def test_identification_deadline_rejects_a_late_cooperative_success(launch, tools, monkeypatch):
    import time
    monkeypatch.setattr(launch, 'IDENT_TIMEOUT_S', .01)
    host = chat_host(launch, Keyed())
    def identify(*args, cancelled, **kwargs):
        assert cancelled.wait(2)
        return Identification(True, 0)
    monkeypatch.setattr(host, 'identify', identify)
    said = []
    steps = host.identify_steps('W9SSJ', 500, say=said.append)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            assert done.value == Identification(True, 0, succeeded=False)
            break
        time.sleep(.001)
    assert len(said) == 1 and 'send W9SSJ by hand' in said[0]


@pytest.mark.parametrize('ending', ['by-hand', 'deadline'])
def test_session_keeps_debt_after_production_keyer_reports_failure(launch, tools, monkeypatch, ending):
    import time
    from hfmodem.tests.contact.test_varac import Desk

    transport = Keyed()
    if ending == 'by-hand':
        transport.burst_level = None
    else:
        monkeypatch.setattr(launch, 'IDENT_TIMEOUT_S', .01)
        def tx(audio, drive, *, identification=False, cancelled=None):
            assert identification and cancelled.wait(2)
            raise InterruptedError('identification cancelled')
        transport.tx = tx
    host = chat_host(launch, transport)
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    d.session.keyer = lambda bw, peer: host.identify_steps('W9SSJ', bw, peer, d.said.append)
    d.host.now = first_tx + 600
    d.tick()
    for _ in range(1000):
        if d.session.steps is None:
            break
        time.sleep(.001)
        d.tick()
    assert d.session.steps is None
    assert d.session.owed and d.session._first_unidentified_tx == first_tx
    assert sum('send W9SSJ by hand' in line for line in d.said) == 1
    assert transport.keyed == ([] if ending == 'by-hand' else [True, False])
    for _ in range(10):
        d.tick()
    assert d.session.owed
    assert sum('send W9SSJ by hand' in line for line in d.said) == 1


def test_cancelled_audio_wait_is_bounded_even_if_the_driver_stalls(tools, monkeypatch):
    import sys
    import time
    import kestrel_connect
    cancelled, release = threading.Event(), threading.Event()
    cancelled.set()
    monkeypatch.setattr(kestrel_connect, '_wedged', None)
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(_exit_handler=lambda: None))
    start = time.monotonic()
    try:
        with pytest.raises(kestrel_connect.AudioWedged):
            kestrel_connect._bounded(30, release.wait, cancelled=cancelled)
        assert time.monotonic() - start < 1
    finally:
        release.set()



def _segment_capacity(kestrel_connect, monkeypatch, segment):
    from hfmodem.core import wav
    monkeypatch.setenv('HFMODEM_RECORDING', 'everything')
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', segment * 48000 // kestrel_connect.FS)


def _segmented_recorder(kestrel_connect, monkeypatch, path, segment):
    """A recorder whose storage jobs are listed, not run: at rollover, or at close."""
    _segment_capacity(kestrel_connect, monkeypatch, segment)
    rec = kestrel_connect.Recorder(path)
    handed, later = [], []
    def store(job):
        (later if rec.session._closing else handed).append(job[1])
    monkeypatch.setattr(rec.session, '_store', store)
    return rec, handed, later


def _feed(rec, count, value=0.):
    # The lane queue holds two seconds, so each second drains before the next.
    import numpy as np
    for start in range(0, count, rec.rx.rate):
        rec.write(np.full(min(rec.rx.rate, count - start), value, np.float32))
        rec.after_pending(lambda: None).result(timeout=3)


def _index(rec):
    return json.loads((rec.session.directory / 'recording.json').read_text())


def _fails_with_data(error):
    def fail(self, data):
        if data:
            raise error
    return fail


def test_a_segment_is_an_hour_and_far_below_the_riff_limit():
    from hfmodem.core import wav
    assert wav.SEGMENT_SAMPLES == 3600 * 48000
    assert 36 + 2 * wav.SEGMENT_SAMPLES < 2**31


def test_segments_are_named_by_their_start_and_found_in_order(tools, tmp_path, monkeypatch):
    import re
    import kestrel_connect
    from hfmodem.core import wav
    minute = 60 * kestrel_connect.FS
    rec, _, _ = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav', minute)
    _feed(rec, 150 * kestrel_connect.FS)
    rec.close()
    assert rec.path.name == 'receive-l' + 'rx'.encode().hex() + '.wav'
    names = [p.name for p in wav.segments(rec.path)]
    found = [re.fullmatch(rf'{rec.path.stem}-(\d{{8}}T\d{{4}}Z)-s(\d{{20}})\.wav', n) for n in names]
    assert len(names) == 3 and all(found)
    assert len({m[1] for m in found}) == 1
    assert [int(m[2]) for m in found] == [0, minute, 2 * minute]
    assert [Path(e['source']).name for e in _index(rec)['lanes']['rx']['expected']] == names
    assert not rec.path.exists()


def test_the_recorder_rolls_over_before_the_riff_size_limit(tools, tmp_path, monkeypatch):
    # The 2026-11-01 vara-chat session died 12.4 h in, when one receive.wav
    # reached 2**31 samples and wave could no longer patch its header.
    import numpy as np
    import kestrel_connect
    from hfmodem.core import wav
    rec, handed, later = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav',
                                             wav.SEGMENT_SAMPLES)
    rec.write(np.zeros(10, np.float32))
    rec.after_pending(lambda: None).result(timeout=3)
    writer = rec.rx.writer
    writer._written = writer._occupied = wav.SEGMENT_SAMPLES - 500
    writer._wav._datawritten = 2 * (wav.SEGMENT_SAMPLES - 500)
    rec.write(np.zeros(4800, np.float32))
    assert rec.after_pending(lambda: rec.samples).result(timeout=3) == 4810
    rec.close()
    # Segment 0's header claims a full hour over 510 real frames.
    [first], [second] = handed, later
    assert first.name.startswith(rec.path.stem + '-') and first.name < second.name
    assert second.name.endswith(f'-s{510:020d}.wav')
    assert [s['samples'] for s in _index(rec)['segments']] == [wav.SEGMENT_SAMPLES, 4300]
    assert second.stat().st_size == 44 + 2 * 4300


def test_each_finished_segment_is_stored_at_rollover_and_the_last_at_close(tools, tmp_path, monkeypatch):
    import kestrel_connect
    from hfmodem.core import wav
    minute = 60 * kestrel_connect.FS
    rec, handed, later = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav', minute)
    _feed(rec, minute + minute // 2)
    assert len(rec.after_pending(lambda: list(handed)).result(timeout=3)) == 1
    _feed(rec, minute)
    rec.close()
    segments = [_absolute(p) for p in wav.segments(rec.path)]
    assert [_absolute(p) for p in handed] == segments[:2]
    assert [_absolute(p) for p in later] == [segments[2]]
    assert rec.samples == 2 * minute + minute // 2
    assert [p.stat().st_size for p in segments] == [44 + 2 * minute] * 2 + [44 + minute]


def test_a_failed_write_is_reported_and_the_recorder_keeps_draining(tools, tmp_path, monkeypatch, capsys):
    import errno
    import wave
    import numpy as np
    import kestrel_connect
    rec, handed, later = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav',
                                             60 * kestrel_connect.FS)
    monkeypatch.setattr(wave.Wave_write, 'writeframes',
                        _fails_with_data(OSError(errno.ENOSPC, 'No space left on device')))
    rec.write(np.zeros(480, np.float32))
    assert rec.after_pending(lambda: 'ran').result(timeout=3) == 'ran'
    assert rec.session._pcm_thread.is_alive() and rec.session._storage_thread.is_alive()
    rec.write(np.zeros(480, np.float32))
    assert rec.after_pending(lambda: rec.samples).result(timeout=3) == rec.position == 960
    rec.close()
    assert capsys.readouterr().err.count('No space left on device') == 1
    lane = _index(rec)['lanes']['rx']
    assert lane['state'] == 'disabled' and lane['gaps'] == [[0, 960]]
    [segment] = _index(rec)['segments']
    assert _absolute(rec.session.directory / segment['source']) == _absolute(rec.rx.writer.paths[0])
    assert segment['state'] == 'partial' and segment['samples'] == 0
    # A partial segment stays the WAV it is; only complete ones are stored.
    assert handed == later == []


def test_a_lost_stretch_keeps_session_positions_and_cannot_shift_a_cutout(tools, tmp_path, monkeypatch):
    import wave
    import numpy as np
    import kestrel_connect
    from hfmodem.station.contactlog import cut_audio
    monkeypatch.setattr(kestrel_connect, 'FS', 8)
    rec, _, _ = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav', 480)
    original = wave.Wave_write.writeframes
    failed = False
    def fail_once(out, pcm):
        nonlocal failed
        if pcm and not failed:
            failed = True
            raise OSError('synthetic lost stretch')
        original(out, pcm)
    monkeypatch.setattr(wave.Wave_write, 'writeframes', fail_once)
    try:
        rec.write(np.zeros(3, np.float32))
        _feed(rec, 480)
        assert rec.after_pending(lambda: rec.samples).result(timeout=3) == 483
        with pytest.raises(ValueError, match='not yet completely written'):
            rec.after_pending(lambda: cut_audio(rec.path, tmp_path / 'clip.wav', 477, 483)).result(timeout=3)
        with pytest.raises(ValueError, match='not yet completely written'):
            rec.after_pending(lambda: cut_audio(rec.path, tmp_path / 'clip.wav', 1, 3)).result(timeout=3)
        assert rec.after_pending(lambda: rec.position).result(timeout=3) == 483
    finally:
        rec.close()
    counts = []
    for part in rec.rx.writer.paths:
        with wave.open(str(part)) as src:
            counts.append(src.getnframes())
    assert counts == [0, 3]
    assert rec.rx.writer.paths[1].name.endswith(f'-s{480:020d}.wav')
    assert not (tmp_path / 'clip.wav').exists()


def test_a_header_failure_at_close_still_indexes_the_last_segment(tools, tmp_path, monkeypatch, capsys):
    import numpy as np
    import kestrel_connect
    rec, handed, later = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav',
                                             60 * kestrel_connect.FS)
    rec.write(np.zeros(10, np.float32))
    rec.after_pending(lambda: None).result(timeout=3)
    def fail():
        raise OSError('synthetic header failure')
    monkeypatch.setattr(rec.rx.writer._wav, '_patchheader', fail)
    rec.write(np.zeros(10, np.float32))
    assert rec.after_pending(lambda: rec.samples).result(timeout=3) == 20
    rec.close()
    rec.close()
    [segment] = _index(rec)['segments']
    assert _absolute(rec.session.directory / segment['source']) == _absolute(rec.rx.writer.paths[0])
    assert segment['state'] == 'partial' and rec.rx.writer.paths[0].exists()
    assert handed == later == []
    assert capsys.readouterr().err.count('synthetic header failure') == 1


def test_storage_failure_is_reported_and_the_worker_survives(tools, tmp_path, monkeypatch, capsys):
    import kestrel_connect
    from hfmodem.core import wav
    monkeypatch.setattr(kestrel_connect, 'FS', 8)
    _segment_capacity(kestrel_connect, monkeypatch, 480)
    rec = kestrel_connect.Recorder(tmp_path / 'receive.wav')
    store, calls = wav.store, []
    def fail_once(path, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            raise OSError('synthetic handoff failure')
        return store(path, **kwargs)
    monkeypatch.setattr(wav, 'store', fail_once)
    _feed(rec, 481)
    assert rec.after_pending(lambda: rec.samples).result(timeout=3) == 481
    assert rec.session._pcm_thread.is_alive() and rec.session._storage_thread.is_alive()
    rec.close()
    assert capsys.readouterr().err.count('synthetic handoff failure') == 1
    assert [s['state'] for s in _index(rec)['segments']] == ['partial', 'finalized']
    assert calls == rec.rx.writer.paths


def test_a_block_that_cannot_become_pcm_is_a_gap_on_the_session_clock(tools, tmp_path, monkeypatch, capsys):
    import wave
    import numpy as np
    import kestrel_connect
    from hfmodem.station.contactlog import cut_audio
    monkeypatch.setattr(kestrel_connect, 'FS', 8)
    rec, handed, later = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav', 480)
    def failed(*args, **kwargs):
        raise MemoryError('synthetic PCM allocation failure')
    with monkeypatch.context() as patch:
        patch.setattr(np, 'clip', failed)
        rec.write(np.zeros(2, np.float32))
        assert rec.after_pending(lambda: rec.samples).result(3) == 2
    _feed(rec, 481, .5)
    assert rec.after_pending(lambda: rec.samples).result(3) == 483
    rec.close()
    assert 'synthetic PCM allocation failure' in capsys.readouterr().err
    assert _index(rec)['lanes']['rx']['gaps'] == [[0, 2]]
    first, second = rec.rx.writer.paths
    assert first.name.endswith(f'-s{2:020d}.wav') and second.name.endswith(f'-s{482:020d}.wav')
    with wave.open(str(first)) as src:
        assert src.getnframes() == 480
        assert set(np.frombuffer(src.readframes(480), '<i2')) == {16384}
    with wave.open(str(second)) as src:
        assert src.getnframes() == 1
    with pytest.raises(ValueError, match='not yet completely written'):
        cut_audio(rec.path, tmp_path / 'bad.wav', 0, 3)
    assert handed == [first] and later == [second]


@pytest.mark.parametrize('failure', ['write', 'action', 'close'])
def test_failed_stderr_cannot_kill_the_receive_drain(tools, tmp_path, monkeypatch, failure):
    import wave
    import numpy as np
    import kestrel_connect
    rec, handed, later = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav', 480)
    class BrokenLog:
        def __init__(self):
            self.attempted = []
        def write(self, text):
            self.attempted.append(text)
            raise OSError('synthetic log disk full')
        def flush(self):
            raise OSError('synthetic log disk full')
    log = BrokenLog()
    error = OSError('synthetic audio disk full')
    def fail(*args):
        raise error
    monkeypatch.setattr(kestrel_connect.sys, 'stderr', log)
    if failure == 'write':
        monkeypatch.setattr(wave.Wave_write, 'writeframes', _fails_with_data(error))
    elif failure == 'action':
        rec.session.after_pending(fail)
    else:
        monkeypatch.setattr(wave.Wave_write, '_patchheader', fail)
    rec.write(np.zeros(3, np.float32))
    assert rec.after_pending(lambda: rec.samples).result(3) == 3
    assert rec.session._pcm_thread.is_alive() and rec.session._storage_thread.is_alive()
    rec.close()
    assert 'synthetic audio disk full' in ''.join(log.attempted)
    [segment] = _index(rec)['segments']
    assert _absolute(rec.session.directory / segment['source']) == _absolute(rec.rx.writer.paths[0])
    if failure == 'action':
        assert segment['state'] == 'storing' and later == rec.rx.writer.paths
    else:
        assert rec.rx.writer.error is error
        assert segment['state'] == 'partial' and later == []


def test_shutdown_reports_known_paths_after_a_short_interior_segment(tools, tmp_path, monkeypatch, capsys):
    import threading
    import wave
    import kestrel_connect as kc
    from types import SimpleNamespace
    monkeypatch.setattr(kc, 'FS', 8)
    rec, _, _ = _segmented_recorder(kc, monkeypatch, tmp_path / 'receive.wav', 480)
    monkeypatch.setattr(wave.Wave_write, 'writeframes', _fails_with_data(OSError('synthetic write failure')))
    _feed(rec, 481)
    audio = object.__new__(kc.AudioVaraIO)
    audio.rec, audio._run, audio.warm_lost = rec, threading.Event(), 0
    audio.timestamp_limitation = None
    audio._stream = SimpleNamespace(stop=lambda: None, close=lambda: None)
    monkeypatch.setattr(kc, '_bounded', lambda *args, **kwargs: None)
    monkeypatch.setattr(kc.rates, 'host_report', lambda: 'host')
    monkeypatch.setattr(audio, 'clock_report', lambda: 'clock')
    audio.stop()
    report = capsys.readouterr().out
    assert str(rec.session.directory) in report
    paths = rec.rx.writer.paths
    assert len(paths) == 2 and all(_absolute(rec.session.directory) in _absolute(p).parents for p in paths)
    assert [_absolute(rec.session.directory / s['source']) for s in _index(rec)['segments']] == [_absolute(p) for p in paths]


@pytest.mark.parametrize('limitation', [None, 'capture timestamps are nominal'])
@pytest.mark.parametrize('count', [3, 1000])
def test_shutdown_keeps_timestamp_facts_in_the_rx_lane(tools, tmp_path, monkeypatch,
                                                      limitation, count):
    import numpy as np
    import kestrel_connect as kc
    # One live write is admitted up to two seconds of audio; the rate keeps 1000 samples inside it.
    monkeypatch.setattr(kc, 'FS', 1000)
    rec, handed, later = _segmented_recorder(kc, monkeypatch, tmp_path / 'receive.wav', 480)
    rec.write(np.zeros(count, np.float32))
    audio = object.__new__(kc.AudioVaraIO)
    audio.rec, audio._run, audio.warm_lost = rec, threading.Event(), 0
    audio._stream = SimpleNamespace(stop=lambda: None, close=lambda: None)
    audio.timestamp_limitation = limitation
    audio.lost, audio.estimated_lost, audio.xruns, audio._fit = 0, 17, 2, []
    monkeypatch.setattr(kc, '_bounded', lambda *args, **kwargs: None)
    monkeypatch.setattr(kc.rates, 'host_report', lambda: 'host')
    monkeypatch.setattr(kc.rates, 'rate_fit', lambda *args: (12.5, 0.25))
    monkeypatch.setattr(audio, 'clock_report', lambda: 'clock')
    audio.stop()
    expected = dict(timestamp_limitation=limitation, lost_samples=0, estimated_lost_samples=17,
                    estimated_clock_ppm=12.5, estimated_clock_sigma=0.25,
                    continuous=None, xruns=2)
    index = json.loads((rec.session.directory / 'recording.json').read_text(encoding='utf-8'))
    row = index['lanes']['rx']
    assert sum(segment['samples'] for segment in index['segments']) == count
    assert len(index['segments']) == (1 if count == 3 else 3)
    # Every indexed segment reaches storage once, and in order.
    assert [_absolute(p) for p in handed + later] == [_absolute(rec.session.directory / segment['source']) for segment in index['segments']]
    if limitation:
        assert {key: row[key] for key in expected} == expected
    else:
        assert not expected.keys() & row.keys()


def test_late_audio_callback_after_recorder_close_is_dropped(tools, tmp_path, monkeypatch):
    import numpy as np
    import kestrel_connect
    rec, handed, later = _segmented_recorder(kestrel_connect, monkeypatch, tmp_path / 'receive.wav', 480)
    rec.write(np.zeros(3, np.float32))
    rec.close()
    rec.write(np.zeros(5, np.float32))
    assert rec.samples == rec.position == 3
    assert not rec.session._pcm.queue
    assert not rec.session._pcm_thread.is_alive() and not rec.session._storage_thread.is_alive()
    assert handed == [] and len(later) == 1

# The observed closer IDs first and the other station waits through its
# identification (3.25 s at 28 WPM). Ours keys after 0.5 s of clear channel once a peer has been
# heard or when the peer closed, at once when we closed onto a clear channel,
# and at the 8 s bound when the channel never clears.
@pytest.mark.parametrize('after_peer,peer_ends,expected', [
    (True, 0.0, 0.5), (True, 0.25, 0.75), (True, 3.25, 3.75), (True, 20.0, 8.0),
    (False, 0.0, 0.0), (False, 3.25, 3.75), (False, 20.0, 8.0)])
def test_closing_identification_listens_until_clear_or_bound(launch, tools, monkeypatch, after_peer,
                                                            peer_ends, expected):
    import kestrel_connect
    now, keyed = [0.0], []
    class Cancelled:
        def wait(self, seconds):
            now[0] = round(now[0] + seconds, 8)
            return False
        def is_set(self):
            return False
    class Transport:
        def unkey_confirmed(self):
            return True
        @property
        def receiving(self):
            return now[0] < peer_ends
    monkeypatch.setattr(launch.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(kestrel_connect, 'identify', lambda *a, **kw: keyed.append(now[0]) or True)
    monkeypatch.setattr(kestrel_connect, '_wedged', None)
    chat_host(launch, Transport()).identify('W9SSJ', 500, after_peer=after_peer, cancelled=Cancelled())
    assert keyed == [pytest.approx(expected)]
    if peer_ends <= 8:
        assert keyed[0] >= peer_ends


@pytest.mark.parametrize('completion', ['keyed', 'finished'])
@pytest.mark.parametrize('outcome', ['success', 'failed', 'by-hand', 'cancelled', 'deadline-rejected'])
def test_cw_handoff_preserves_broadcast_keying_and_debt(launch, tools, monkeypatch, completion,
                                                       outcome):
    import time
    from hfmodem.tests.contact.test_broadcast_client import desk, select, expected
    from hfmodem.contact import varac as v

    d = desk()
    select(d)
    reporting = launch.chat_server().session_class
    observer = reporting.__new__(reporting)
    observer._event = lambda kind, raw: d.host.lines(raw)
    observer.modem_ptt(True)
    observer.modem_ptt(False)
    d.tick()
    first_tx = d.session._first_unidentified_tx
    d.host.now = first_tx + 599
    d.session.command('/all hello everyone')
    d.tick()
    written = d.host.now
    transport = Keyed()
    host = chat_host(launch, transport)
    host.runtime.modem = SimpleNamespace(observer=observer)
    if outcome == 'by-hand':
        transport.burst_level = None
    elif outcome in ('failed', 'cancelled'):
        def tx(audio, drive, *, identification=False, cancelled=None):
            assert identification
            if outcome == 'cancelled':
                assert cancelled.wait(2)
                raise InterruptedError('identification cancelled')
            raise OSError('identification playback failed')
        transport.tx = tx
    if outcome in ('cancelled', 'deadline-rejected'):
        monkeypatch.setattr(launch, 'IDENT_TIMEOUT_S', .01)
    if outcome == 'deadline-rejected':
        keyer = host.identify
        def late_identify(*args, cancelled, **kwargs):
            result = keyer(*args, cancelled=cancelled, **kwargs)
            assert cancelled.wait(2)
            return result
        monkeypatch.setattr(host, 'identify', late_identify)
    confirmations = []
    def unkey_confirmed():
        confirmations.append(host.lock.locked())
        return not d.line and (not transport.keyed or transport.keyed[-1] is False)
    transport.unkey_confirmed = unkey_confirmed
    ready, consume = [], []
    def identify(bw, peer):
        result = yield from host.identify_steps('W9SSJ', bw, peer, d.said.append)
        ready.append(result)
        while not consume:
            yield .1
        return result
    d.session.keyer = identify
    d.host.now = written + v.KISS_SUBMISSION_TIMEOUT_S
    for _ in range(1000):
        d.tick()
        if ready:
            break
        time.sleep(.001)
    assert len(ready) == 1
    assert ready[0] == Identification(True, 1, succeeded=outcome == 'success')
    assert host.stepping.is_set() and not host.lock.locked()
    before = len(d.sent)
    later_tx = d.host.now + 1
    starts = []
    event = d.host.client.event
    def timestamped(item):
        if item.kind == 'host' and item.raw == 'PTT ON':
            starts.append(d.host.now)
        event(item)
    d.host.client.event = timestamped
    d.host.now = later_tx
    with host.lock:
        observer._event('command', 'PENDING')
        observer.modem_ptt(True)
        d.line = True
        if completion != 'keyed':
            for _ in range(expected(d)):
                observer._event('command', 'BITRATE (1) 18 bps TX')
            observer.modem_ptt(False)
            observer._event('command', 'CANCELPENDING')
            d.line = False
    d.tick()
    consume.append(True)
    d.tick()
    assert d.session._close_failed is None
    # The /all frame names us: once it completes it identifies the station.
    if completion == 'keyed':
        assert d.session.owed
        assert d.session._first_unidentified_tx == (starts[0] if outcome == 'success' else first_tx)
    else:
        assert not d.session.owed
    assert confirmations == [True]
    assert len(d.ids) == 0
    assert len(transport.played) == (1 if outcome in ('success', 'deadline-rejected') else 0)
    assert sum('send W9SSJ by hand' in line for line in d.said) == (outcome != 'success')
    assert len(d.host.kiss.sent) == 1
    assert 'ABORT' not in d.sent[before:]
    if completion == 'keyed':
        d.line = False
        for _ in range(expected(d)):
            observer._event('command', 'BITRATE (1) 18 bps TX')
        observer.modem_ptt(False)
        observer._event('command', 'CANCELPENDING')
        d.tick()
    assert not d.session.broadcast.active and not d.session.owed


def test_closing_cw_generator_preserves_unkey_before_broadcast_handoff(launch, tools):
    from hfmodem.contact import varac as v
    from hfmodem.tests.contact.test_broadcast_client import desk, select

    d = desk()
    select(d)
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    d.host.now = first_tx + 599
    d.session.command('/all hello everyone')
    d.tick()
    written = d.host.now
    playing = threading.Event()
    transport = Keyed()
    def tx(audio, drive, *, identification=False, cancelled=None):
        playing.set()
        assert cancelled.wait(2)
        raise InterruptedError('identification cancelled')
    transport.tx = tx
    host = chat_host(launch, transport)
    confirmations = []
    def unkey_confirmed():
        confirmations.append(host.lock.locked())
        return transport.keyed[-1] is False
    transport.unkey_confirmed = unkey_confirmed
    resume = host.stepping.set
    def resume_with_broadcast():
        resume()
        with host.lock:
            d.line = True
            d.host.lines('PENDING', 'PTT ON')
    host.stepping.set = resume_with_broadcast
    d.session.keyer = lambda bw, peer: host.identify_steps('W9SSJ', bw, peer, d.said.append)
    d.host.now = written + v.KISS_SUBMISSION_TIMEOUT_S
    d.tick()
    assert playing.wait(2)
    steps, d.session.steps = d.session.steps, None
    steps.close()
    d.tick()
    assert d.line and not d.session._identifying
    assert d.session._close_failed is None and not host.stop.is_set()
    assert confirmations == [True] and transport.keyed == [True, False]
    assert d.session.owed and d.session._first_unidentified_tx == first_tx
    assert len(d.host.kiss.sent) == 1
    assert sum('send W9SSJ by hand' in line for line in d.said) == 1


@pytest.mark.parametrize('confirmation', [None, False])
def test_successful_cw_without_confirmed_unkey_stops_runtime(launch, tools, confirmation):
    transport = Keyed()
    transport.unkey_confirmed = None if confirmation is None else lambda: False
    host = chat_host(launch, transport)
    said = []
    with pytest.raises(HostError, match='CW unkey not confirmed'):
        host.identify('W9SSJ', 500, say=said.append)
    assert host.stop.is_set() and transport.keyed == [True, False]
    assert len(said) == 1 and 'send W9SSJ by hand' in said[0]
