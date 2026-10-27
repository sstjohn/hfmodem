# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from contextlib import contextmanager
import io
import json
import socket
import struct
import threading
from types import SimpleNamespace

import pytest

from hfmodem.contact.varac import Event, HostError, Session


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
    launch.validate(arguments(launch, tmp_path, '--bw', '2750', '--name', 'Saul', '--services', 'VE'))
    with pytest.raises(ValueError):
        launch.validate(arguments(launch, tmp_path, '--name', 'Saul><DISC>'))
    for words in (['--band', '99'], ['--bw', '1000'], ['--action', 'listen'], ['--slot', '1'],
                  ['--execute'], ['--dial', '14105000']):
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
    from vara_rig_bridge import Rig
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
def test_session_on_fakes_stops_host_before_retiring_keyer(launch, monkeypatch, tmp_path, bw):
    import kestrel_connect
    from hfmodem.contact import varac
    monkeypatch.setattr(varac.time, "time", lambda: 1700000000)
    from hfmodem.core import config
    order, events = [], []
    args = arguments(launch, tmp_path, *(['--bw', str(bw)] if bw else []))
    monkeypatch.setattr(launch, 'tune', lambda rig, hz, *_: order.append('tune ' + str(hz)))
    rig = SimpleNamespace(retired=False, shutdown=lambda: order.append('unkey'), armed=False,
                          line_state=lambda: False)
    admission = SimpleNamespace(opened=lambda: order.append('admitted'))
    class Tty(io.StringIO):
        def close(self):
            pass
    tty = Tty()
    monkeypatch.setattr(launch, 'Terminal', lambda held, history: tty)
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
                                  runtime=SimpleNamespace())
        finally:
            order.append('host closed')
    monkeypatch.setattr(launch, 'Host', host)

    class Client:
        sequence = 0
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
        raise KeyboardInterrupt
    monkeypatch.setattr(launch, 'terminal', terminal)
    with pytest.raises(KeyboardInterrupt):
        launch.run(args, rig, admission)
    assert order == ['admitted', 'initialize', f'BW{bw or 500}', 'LISTEN CQ', 'terminal',
                     'client closed', 'host closed', 'unkey']
    transcript = (tmp_path / 'host.jsonl').read_text()
    assert '<UNKNOWN>café' in transcript and 'CQFRAME WI9YZA-15 500' in transcript
    screen = tty.getvalue().splitlines()
    assert screen.pop(0) == 'PSKReporter off'
    assert screen[0].startswith('commands: /status, /help [bbs|email], /answer N, /cq [SLOT] [repeat], /call CALL [SLOT|cf]')
    assert screen[1:] == ['  1  WI9YZA   slot 15  14108.750 kHz   500 Hz  SN +11.0  2023-11-14 22:13:20Z',
                          f'monitoring 14105.000 kHz at {bw or 500} Hz, auto-answer on',
                          'Passive transcripts are logged; /follow cf shows calling-frequency activity, '
                          '/follow all shows the current passband, /follow off hides it. '
                          '/follow SLOT or KHZ retunes if needed; /follow cf returns to CF.']
    recorded = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert [(r['kind'], r.get('call') or r.get('state')) for r in recorded] == [
        ('heard', 'WI9YZA'), ('state', 'monitoring')]
    assert all(r['utc'].endswith('+00:00') for r in recorded)
    assert (tmp_path / 'contacts.jsonl').read_text() == ''


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


TERMINAL_CHILD = '''
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


def test_the_terminal_edits_recalls_and_keeps_the_typed_line_under_screen_lines(tools, tmp_path):
    import fcntl, os, pty, select, subprocess, sys, termios, time
    history = tmp_path / 'history'

    def session(*steps):
        master, slave = pty.openpty()
        def own_the_tty():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        child = subprocess.Popen([sys.executable, '-c', TERMINAL_CHILD, str(history)], stdin=slave,
                                 stdout=subprocess.PIPE, preexec_fn=own_the_tty,
                                 env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path + [str(tools)])))
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
    assert history.read_text().splitlines()[-2:] == ['later', 'Xabc']
    screen, _ = session((b'> ', b'\x1b[A\r'), (b"got 'Xabc\\n'", b'\x12'),
                        ((b'bck:', b'(reverse-i-search)'), b'lat'),
                        ((b'bck:la', b'lat'), b'\r'), (b'> ', b'\r'),
                        (b'heard', b'\x04'))


HELD_BUSY_CHILD = '''
import queue, struct, sys
from pathlib import Path
from types import SimpleNamespace
import kestrel_connect, varac_chat
from hfmodem.core import config, wav

session = Path(sys.argv[1]) / 'sessions' / 's'
session.mkdir(parents=True)
# The 10-23 20 m session's 6 h 19 min of 48 kHz mono, sparse.
received, size = session / 'receive.wav', kestrel_connect.FS * (6 * 3600 + 19 * 60) * 2
with received.open('wb') as out:
    out.write(b'RIFF' + struct.pack('<I', 36 + size) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 1, 1, 48000, 96000, 2, 16)
              + b'data' + struct.pack('<I', size))
    out.truncate(44 + size)
rec = SimpleNamespace(path=received, store_log=received.with_suffix('.store.log'))
rec.close = lambda: wav.store_later(rec.path, rec.store_log)

class Channel(kestrel_connect._DryIO):
    """A channel the host hears busy throughout, over that recording."""
    receiving = True
    def __init__(self):
        super().__init__()
        self.rec = rec
        self.taps = []
    def log(self, msg): pass
    def stop(self): self.rec.close()

kestrel_connect.AudioVaraIO = lambda *a, **k: Channel()
config.tx_drive = lambda: .4
varac_chat.tune = lambda *a, **k: None
args = varac_chat.parser().parse_args(['--mycall', 'W9SSJ', '--band', '40', '--session', str(session),
    '--expect-model', 'fake', '--ptt-device', 'unused', '--tx-device', 'unused', '--rx-device', 'unused'])
rig = SimpleNamespace(retired=False, shutdown=lambda: None, armed=False, line_state=lambda: None,
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
    child = subprocess.Popen([sys.executable, '-c', HELD_BUSY_CHILD, str(tmp_path)], stdin=slave,
                             stdout=subprocess.DEVNULL, preexec_fn=own_the_tty,
                             env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path + [str(tools)])))
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
        return path.exists() and '"BUSY ON"' in path.read_text()
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


def test_the_line_answers_within_a_second_while_the_host_holds_busy(held_busy):
    held_busy.typed(b'/help\r')
    held_busy.until(b'commands: /status, /help [bbs|email], /answer N', 1)


def test_quit_returns_at_once_and_the_recording_is_stored_after_it(held_busy):
    # 20261023T155947666857Z: /quit after six hours, then /help and /? unanswered while the
    # 2.2 GB receive.wav went to FLAC, until a Ctrl-C left it a WAV.
    import time
    held_busy.typed(b'/quit\r')
    quit_at = time.monotonic()
    held_busy.until(b'receive audio is being stored in the background (log: ', 1)
    assert held_busy.child.wait(2) == 0 and time.monotonic() - quit_at < 2
    received = held_busy.session / 'receive.wav'
    stored_log = received.with_suffix('.store.log')
    # store() removes the WAV before its caller writes the final JSON report.
    # Wait for both completion observations within the original storage bound.
    while received.exists() or '"wav": "receive.flac"' not in stored_log.read_text():
        assert time.monotonic() - quit_at < 120, stored_log.read_text()
        time.sleep(.2)
    assert received.with_suffix('.flac').exists()
    assert '"wav": "receive.flac"' in stored_log.read_text()


def test_the_session_commands_are_the_terminals_help_line():
    assert Session.COMMANDS == (
        '/status', '/help [bbs|email]', '/answer N', '/cq [SLOT] [repeat]',
        '/call CALL [SLOT|cf]', '/stop (or Enter)', '/heard', '/slots',
        '/follow all|cf|SLOT|KHZ|off', '/tag TAG [value]', '/qsy SLOT', '/bye [text]',
        '/clear', '/auto on|off', '/beacon [MINUTES|now|force|off]',
        '/beacon hop BANDS [every MINUTES]', '/beacon hop auto [BANDS] [every MINUTES]',
        '/beacon hop status', '/band N', '/band ready|unready',
        '/away [MESSAGE]', '/back', '/vmail [N | send SUBJECT | pending | collect CALL | dismiss CALL]',
        '/email send TO_EMAIL REPLY_EMAIL SUBJECT', '/bbs list|get N|request N|capture',
        '/inquire heard', '/pskreporter [status|off]', '/all [cf|slot N|TEXT|status|off]',
        '/verbose [on|off]', '/quit')


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
    from vara_rig_bridge import Rig
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
    from vara_rig_bridge import Rig
    address, heard = rigctld([None, None])
    rig = Rig(address, False, lambda line: None)
    try:
        with pytest.raises(ConnectionResetError):
            rig.set_freq(3_593_500)
    finally:
        rig.retire()
    assert heard == ['F 3593500\n'] * 2


def test_disarming_records_the_lines_hand_back(tools, monkeypatch):
    import vara_rig_bridge
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
    monkeypatch.setattr(vara_rig_bridge, 'LineKeyer', Keyer)
    rig = vara_rig_bridge.Rig(None, False, lambda line: None, ptt_device='fake', line_ptt=True)
    try:
        rig.armed = True
        assert rig.released_low is None
        rig.armed = False
        assert rig.released_low is True
    finally:
        rig.retire()


def keyable(vara_rig_bridge, monkeypatch, address, said):
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
    monkeypatch.setattr(vara_rig_bridge, 'LineKeyer', Keyer)
    rig = vara_rig_bridge.Rig(address, False, said.append, ptt_device='fake', line_ptt=True)
    rig.armed = True
    return rig


def test_a_transmission_logs_its_peak_power_and_alc(tools, monkeypatch):
    import time
    import vara_rig_bridge
    monkeypatch.setattr(vara_rig_bridge, 'METER_PERIOD_S', 30.0)
    address, heard = rigctld(['42.5\n', '0.31\n'])
    said = []
    rig = keyable(vara_rig_bridge, monkeypatch, address, said)
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
    import vara_rig_bridge
    closed = socket.create_server(('127.0.0.1', 0))
    address = f'127.0.0.1:{closed.getsockname()[1]}'
    closed.close()
    said = []
    rig = keyable(vara_rig_bridge, monkeypatch, address, said)
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


def test_the_session_writes_each_transmissions_meters_to_its_events(launch, tmp_path, monkeypatch):
    import queue
    import kestrel_connect
    from hfmodem.core import config
    args = arguments(launch, tmp_path)
    reports = queue.SimpleQueue()
    reports.put(dict(peak_watts=38.0, peak_alc=0.4, reads=20, seconds=5.4, unread=None))
    rig = SimpleNamespace(retired=False, shutdown=lambda: None, armed=False, line_state=lambda: False,
                          meter_reports=reports)
    monkeypatch.setattr(launch, 'tune', lambda *a: None)
    monkeypatch.setattr(launch, 'Terminal', lambda held, history: io.StringIO())
    monkeypatch.setattr(config, 'tx_drive', lambda: .4)
    monkeypatch.setattr(kestrel_connect, 'AudioVaraIO', lambda *a, **kw: SimpleNamespace(taps=[]))

    @contextmanager
    def host(transport):
        yield SimpleNamespace(server=SimpleNamespace(cmd_port=100, data_port=101), check=lambda: None)
    monkeypatch.setattr(launch, 'Host', host)

    @contextmanager
    def open_client(**kwargs):
        kwargs['ready']([], 0)
        raise KeyboardInterrupt
        yield
    monkeypatch.setattr(launch.Client, 'open', open_client)
    with pytest.raises(KeyboardInterrupt):
        launch.run(args, rig, SimpleNamespace(opened=lambda: None))
    recorded = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert [(r['kind'], r['peak_watts'], r['peak_alc']) for r in recorded] == [('tx_meters', 38.0, 0.4)]


def test_a_closed_receive_recording_is_left_to_store_after_exit(tools, tmp_path, monkeypatch):
    import numpy as np
    import kestrel_connect
    later = []
    monkeypatch.setattr(kestrel_connect, 'store_later', lambda *paths: later.append(paths))
    rec = kestrel_connect.Recorder(tmp_path / 'receive.wav')
    rec.write(np.zeros(480, np.float32))
    rec.close()
    assert later == [(tmp_path / 'receive.wav', tmp_path / 'receive.store.log')]
    assert (tmp_path / 'receive.wav').stat().st_size == 44 + 960


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
