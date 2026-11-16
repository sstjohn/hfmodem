# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import io
import json
import os
import subprocess
import sys
import textwrap
from threading import get_ident
from types import SimpleNamespace

import pytest

from hfmodem.contact import varac as v
from hfmodem.contact.chat_controller import LineAdapter
from hfmodem.contact.chat_session import SessionAdapter
from hfmodem.tests.contact.parent_console import symbols
from hfmodem.tests.contact.test_varac import Desk
from hfmodem.tests.contact.test_varac_bbs import LIST


SCRIPTS = {
    'chat': (('line', '/verbose on\n'), ('line', 'café [red]\n'), ('rx', 'reply\nsecond line'),
             ('line', '/status\n'), ('line', '/help\n'), ('line', '/tag SNR 4\n')),
    'draft': (('line', '/vmail subject\n'), ('line', 'first\n'), ('line', '/status\n'),
              ('line', '/auto off\n'), ('line', '/heard\n'), ('line', '.\n'), ('rx', '<SMR>')),
    'email': (('rx', '<S:E>'), ('line', '/email send a@example.org b@example.org subject\n'),
              ('line', 'body\n'), ('line', '/email status\n'), ('line', '.\n'),
              ('rx', '<SMR>'), ('rx', '<ER>')),
    'bbs': (('line', '/bbs list\n'), ('rx', LIST), ('line', '/bbs enter\n'),
            ('line', '/bbs term ansi16 80 24\n'), ('rx', 'BBS TERM OK ANSI16 80 24'),
            ('rx', '\x1b[31;1mMenu\x1b[0m\n\x1b]52;c;attack\x07next'),
            ('line', '\n'), ('rx', 'BBS EXIT OK')),
    'refusal': (('line', '/call INVALID!\n'), ('line', '/qsy 999\n'), ('line', '/all hello\n'),
                ('line', '/vmail draft\n'), ('line', 'keep\n'), ('line', '/not-a-command\n')),
    'band': (('host', 'DISCONNECTED'), ('line', '/band 40\n'), ('line', '/verbose on\n'),
             ('line', '/status\n')),
    'diagnostics': (('line', '/verbose on\n'), ('diagnostic', ('acknowledged', {'bytes': 17})),
                    ('diagnostic', ('discarded', {'bytes': 9})),
                    ('diagnostic', ('rx_protocol', {'text': 'accepted frame'})),
                    ('diagnostic', ('link_recovery', {'text': 'recovery evidence'})),
                    ('diagnostic', ('capture_rate', {'text': 'capture rate evidence'}))),
    'power': (('line', '/power\n'), ('line', '/power 80\n')),
    'parking': (('line', '/vmail park K1ABC parked subject\n'), ('line', 'body\n'),
                ('line', '.\n'), ('line', '/vmail parked\n'),
                ('line', '/vmail parking off\n'), ('line', '/vmail relay off\n')),
}


def replay(script, adapted, root):
    desk = Desk(mailbox=root)
    desk.session.parking.store.clock = lambda: 1793232000.
    desk.flow.dial = 14106500
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    desk.host.client.enable_writer = lambda: None
    stdout, screen, bbs = io.StringIO(), io.StringIO(), io.StringIO()
    terminal = SimpleNamespace(write=screen.write)
    import varac_chat as launch
    reference = symbols(vars(launch))
    owner_threads = set()
    Terminal = launch.Terminal
    def say(text):
        owner_threads.add(get_ident())
        screen.write(text + '\n')
    def bbs_say(text):
        bbs.write(text)
        (Terminal if adapted else reference['Terminal']).write_bbs(terminal, text)
    def emit(kind, **details):
        stdout.write(json.dumps(dict(kind=kind, **details), ensure_ascii=False) + '\n')
        if kind == 'band':
            stdout.write(f'BAND {details["band_m"]} m {details["dial_hz"]} Hz\n')
    desk.session.say, desk.session.emit = say, emit
    desk.session.bbs.say = bbs_say
    diagnostics = launch.Diagnostics()
    ready = desk.host.client.ready
    def drain(sockets, timeout):
        diagnostics.drain(desk.session)
        return ready(sockets, timeout)
    desk.host.client.ready = drain
    owner = LineAdapter(lambda: SessionAdapter(desk.session, legacy=True)) if adapted else desk.session
    try:
        for kind, value in script:
            if kind == 'rx':
                desk.host.data.incoming.append(v.frame_message(value))
            elif kind == 'host':
                desk.host.lines(value)
            elif kind == 'diagnostic':
                diagnostics.put(value[0], **value[1])
            owner.tick()
            if kind == 'line':
                assert owner.command(value)
        for _ in range(3):
            owner.tick()
    finally:
        owner.quit()
    if adapted:
        assert owner.controller.join(1)
    assert owner_threads == {get_ident()}
    return (stdout.getvalue().encode(), screen.getvalue().encode(), bbs.getvalue().encode(),
            tuple(desk.host.command.sent), tuple(desk.host.data.sent))


@pytest.mark.parametrize('scenario', SCRIPTS)
def test_paired_legacy_replay_exact_stdout_screen_bbs_and_host_writes(tools, monkeypatch, tmp_path,
                                                                     scenario):
    monkeypatch.setattr(v, 'utc', lambda: '2026-10-28T00:00:00Z')
    direct = replay(SCRIPTS[scenario], False, tmp_path / 'direct')
    adapted = replay(SCRIPTS[scenario], True, tmp_path / 'adapted')
    assert adapted == direct
    assert direct[0] and direct[1]
    if scenario == 'bbs':
        assert b'\x1b[0;1;31mMenu' in direct[2] and b'\x1b[0m' in direct[1]
    if scenario == 'band':
        assert b'BAND 40 m 7105000 Hz\n' in direct[0]


def test_private_line_preserves_quit_return_and_is_not_a_wire_action():
    desk = Desk()
    line = LineAdapter(lambda: SessionAdapter(desk.session, legacy=True))
    assert line.command('/quit\n') is False
    line.quit()
    assert line.controller.join(1)
    from hfmodem.contact import chat_state as s
    assert all('legacy' not in kind for kind in s.ACTIONS)


def test_line_owner_failure_joins_and_preserves_cleanup_error(monkeypatch):
    desk = Desk()
    initial, cleanup = ValueError('poll failed'), OSError('quit failed')
    def fail_tick(*args):
        raise initial
    def fail_quit(*, unwinding=None):
        raise cleanup
    monkeypatch.setattr(desk.session, 'tick', fail_tick)
    monkeypatch.setattr(desk.session, 'quit', fail_quit)
    line = LineAdapter(lambda: SessionAdapter(desk.session, legacy=True))
    with pytest.raises(ValueError) as caught:
        line.tick()
    assert caught.value is initial and caught.value.__cause__ is cleanup
    with pytest.raises(ValueError):
        line.quit()
    assert line.controller.join(1)


def test_paired_launcher_stdout_screen_and_canonical_logs(tools, monkeypatch, tmp_path):
    from contextlib import contextmanager, redirect_stdout
    import queue
    import kestrel_connect
    import varac_chat as launch
    from hfmodem.contact import slots
    from hfmodem.core import config
    from hfmodem.tests.contact.test_varac import Host
    from hfmodem.tests.contact.test_varac_launcher import arguments

    monkeypatch.setattr(v, 'utc', lambda: '2026-10-28T00:00:00Z')
    monkeypatch.setattr(launch, 'utc', v.utc)
    monkeypatch.setattr(config, 'tx_drive', lambda: .4)
    monkeypatch.setattr(launch, 'tune', lambda *args: None)
    monkeypatch.setattr(launch, 'change_band', lambda *args: {'mode': 'USB'})
    monkeypatch.setattr(slots, 'SlotWatch', lambda *a, **k: SimpleNamespace(
        put=lambda frame: None, poll=lambda: [], table=lambda: ([], 0),
        retune=lambda *args: None, close=lambda: None))
    monkeypatch.setattr(kestrel_connect, 'AudioVaraIO', lambda *a, **k: SimpleNamespace(taps=[]))
    @contextmanager
    def host(audio):
        yield SimpleNamespace(server=SimpleNamespace(cmd_port=100, data_port=101),
                              runtime=SimpleNamespace(), arm=lambda on: None, identify_steps=lambda *a: iter(()))
    monkeypatch.setattr(launch, 'Host', host)
    terminal = launch.terminal
    reference = symbols(vars(launch))
    Terminal = launch.Terminal
    logging_say = launch.logging_say
    def run(adapted, directory):
        directory.mkdir()
        fake = Host(replies={'VERSION': ['VERSION test-host'], 'DISCONNECT': ['OK', 'DISCONNECTED']})
        fake.data.respond = lambda payload: fake.lines('BUFFER 0')
        class Tty(io.StringIO):
            count = 0
            def readline(self):
                self.count += 1
                if self.count == 2:
                    fake.lines('CONNECTED W9SSJ N0XYZ 500')
                    return None
                return {1: '/band 40\n', 3: 'hello\n', 4: '/verbose on\n',
                        5: '/status\n', 6: '/bye\n', 10: '/quit\n'}.get(self.count)
            def write_bbs(self, text):
                Terminal.write_bbs(self, text)
        tty = Tty()
        monkeypatch.setattr(launch, 'Terminal', lambda *args: tty)
        @contextmanager
        def client(**kwargs):
            fake.client.event = kwargs['event']
            try:
                yield fake.client
            finally:
                fake.client.close()
        monkeypatch.setattr(launch.Client, 'open', client)
        def drive(session, stream):
            (terminal if adapted else reference['terminal'])(session, stream)
        monkeypatch.setattr(launch, 'logging_say', logging_say if adapted else reference['logging_say'])
        monkeypatch.setattr(launch, 'terminal', drive)
        rig = SimpleNamespace(retired=False, keyed=False, armed=False, line_state=lambda: False,
                              shutdown=lambda: None, meter_reports=queue.Queue(),
                              power_w=lambda watts=None, scale_w=100: 10 if watts is None else watts)
        stdout = io.StringIO()
        args = arguments(launch, directory, '--no-bbs')
        with redirect_stdout(stdout):
            assert launch.run(args, rig, SimpleNamespace(opened=lambda: None)) == 0
        return (stdout.getvalue(), tty.getvalue().replace(str(directory), '<session>'), (directory / 'events.jsonl').read_bytes(),
                (directory / 'host.jsonl').read_bytes(), (directory / 'contacts.jsonl').read_bytes(),
                (directory / 'console.log').read_bytes().replace(str(directory).encode(), b'<session>'),
                fake.command.sent, fake.data.sent)
    direct = run(False, tmp_path / 'direct')
    adapted = run(True, tmp_path / 'adapted')
    assert adapted == direct
    assert direct[0] == 'BAND 40 m 7105000 Hz\n'
    assert b'"kind": "contact"' in direct[4]


@pytest.mark.parametrize('physical_line', (None, False, True))
def test_disarmed_quit_through_line_adapter(physical_line):
    desk = Desk()
    desk.session.line_state = lambda: physical_line
    line = LineAdapter(lambda: SessionAdapter(desk.session, legacy=True))
    assert line.command('/quit\n') is False
    if physical_line is True:
        with pytest.raises(v.HostError, match='HIGH'):
            line.quit()
    else:
        line.quit()
    assert line.controller.join(0)


@pytest.mark.parametrize('panic', (False, True))
@pytest.mark.parametrize('operation', ('request', 'close'))
def test_sigint_unwinds_line_wait_on_main_thread(tools, panic, operation):
    if not (tools / 'vara_rig_bridge.py').is_file():
        pytest.skip('tools/vara_rig_bridge.py is a development tool that does not ship')
    script = r"""
        import atexit
        import io
        import os
        import signal
        import threading
        import time
        from types import SimpleNamespace
        from hfmodem.tests.contact.test_varac import Desk
        from varac_chat import terminal
        from vara_rig_bridge import _install_unkey_handlers

        desk = Desk()
        client = desk.host.client
        client.enable_writer = lambda: None
        client.clock = time.monotonic
        desk.host.replies['LISTEN CQ'] = []
        calls = []
        rig = SimpleNamespace(panic=lambda why: calls.append((why, threading.get_ident())),
                              shutdown=lambda: None)
        if PANIC:
            _install_unkey_handlers(rig)
        if OPERATION == 'close':
            desk.hear('CONNECTED W9SSJ N0XYZ 500')
        ready = client.ready
        blocked = False
        def wait(sockets, timeout):
            global blocked
            if not blocked and (desk.session._quitting if OPERATION == 'close' else 'LISTEN CQ' in desk.host.sent):
                blocked = True
                threading.Timer(.05, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
                time.sleep(2)
                raise AssertionError('blocking wait was not interrupted')
            return ready(sockets, timeout)
        client.ready = wait
        if OPERATION == 'close':
            tty = io.StringIO('/quit\n')
        else:
            tty = io.StringIO('/auto on\n')
        start = time.monotonic()
        try:
            terminal(desk.session, tty)
        except SystemExit as error:
            assert PANIC and error.code == 1
        except KeyboardInterrupt:
            assert not PANIC and OPERATION == 'close'
        elapsed = time.monotonic() - start
        assert blocked
        assert elapsed < 1, elapsed
        assert calls == ([('SIGINT', threading.get_ident())] if PANIC else [])
        assert not any(t.name == 'chat-owner' for t in threading.enumerate())
        atexit.unregister(rig.shutdown)
    """
    script = textwrap.dedent(script).replace('PANIC', repr(panic)).replace('OPERATION', repr(operation))
    path = os.pathsep.join(filter(None, (str(tools), os.environ.get('PYTHONPATH'))))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10,
                            env=dict(os.environ, PYTHONPATH=path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_line_configuration_is_owned_by_adapter():
    from inspect import signature
    from hfmodem.contact.chat_controller import Controller
    assert '_line_mode' not in signature(Controller).parameters


@pytest.mark.parametrize('typed_exit', (False, True))
def test_typed_controller_still_requires_low_line_proof(typed_exit):
    from hfmodem.contact import chat_state as s
    from hfmodem.contact.chat_controller import Controller
    desk = Desk()
    desk.session.line_state = lambda: None
    controller = Controller(lambda: SessionAdapter(desk.session, legacy=True))
    try:
        controller.ready.result(1)
        if typed_exit:
            request = s.Request('exit', controller.controller_id, 0, 'test', None, None, s.ChatExit())
            assert controller.submit(request).queued
            assert controller.join(1)
        with pytest.raises(RuntimeError, match='low-line proof'):
            controller.shutdown().result(1)
    finally:
        controller.shutdown()
        assert controller.join(1)
    if typed_exit:
        assert controller.outcome('exit').state.state == 'failed'


@pytest.mark.parametrize('fault', (RuntimeError, KeyboardInterrupt))
def test_paired_terminal_tick_fault_preserves_main_line_close(tools, monkeypatch, fault):
    import varac_chat as launch
    reference = symbols(vars(launch))['terminal']
    def run(terminal):
        desk = Desk()
        desk.hear('CONNECTED W9SSJ N0XYZ 500')
        desk.session.owed = True
        desk.session._tx_dial = desk.flow.dial
        def tick(*args):
            raise fault('tick fault')
        monkeypatch.setattr(desk.session, 'tick', tick)
        if fault is KeyboardInterrupt:
            terminal(desk.session, SimpleNamespace(readline=lambda: None))
        else:
            with pytest.raises(fault, match='tick fault'):
                terminal(desk.session, SimpleNamespace(readline=lambda: None))
        return desk.sent, desk.ids
    assert run(launch.terminal) == run(reference)



@pytest.mark.parametrize('fault', [OSError, KeyboardInterrupt])
def test_paired_terminal_reader_fault_preserves_main_line_close(tools, fault):
    import varac_chat as launch
    reference = symbols(vars(launch))['terminal']
    def readline():
        raise fault('reader fault')
    def run(terminal):
        desk = Desk()
        desk.hear('CONNECTED W9SSJ N0XYZ 500')
        desk.session.owed = True
        desk.session._tx_dial = desk.flow.dial
        if fault is KeyboardInterrupt:
            terminal(desk.session, SimpleNamespace(readline=readline))
        else:
            with pytest.raises(fault, match='reader fault'):
                terminal(desk.session, SimpleNamespace(readline=readline))
        return desk.sent, desk.ids
    assert run(launch.terminal) == run(reference)

def test_line_ctrl_c_during_projection_quits_quietly(tools, monkeypatch):
    import varac_chat as launch
    project = SessionAdapter.project
    count = 0
    def interrupted(port):
        nonlocal count
        count += 1
        if count == 3:
            raise KeyboardInterrupt
        return project(port)
    monkeypatch.setattr(SessionAdapter, 'project', interrupted)
    launch.terminal(Desk().session, SimpleNamespace(readline=lambda: None))


def test_paired_mail_collection_keeps_main_line_host_writes(tools, monkeypatch, tmp_path):
    from hfmodem.tests.contact.test_mail_collection import notice, WIRE
    import varac_chat as launch
    monkeypatch.setattr(v, 'utc', lambda: '2026-10-28T00:00:00Z')
    def run(adapted):
        desk = Desk(mailbox=tmp_path / str(adapted))
        notice(desk)
        def connect():
            desk.host.data.incoming.append(v.frame_message('<AWQ>'))
            return ['OK', 'CONNECTED W9SSJ N0BBS 500']
        desk.host.replies['CONNECT W9SSJ N0BBS'] = connect
        def reply(data):
            desk.host.lines('BUFFER 0')
            if data == v.frame_message('<TL>'):
                desk.host.data.incoming.append(WIRE)
        desk.host.data.respond = reply
        lines = iter(['/vmail collect N0BBS\n'] + [None] * 30 + ['/quit\n'])
        terminal = launch.terminal if adapted else symbols(vars(launch))['terminal']
        terminal(desk.session, SimpleNamespace(readline=lambda: next(lines)))
        return desk.sent, desk.texts, desk.said, desk.host.client.link_generation
    assert run(True) == run(False)


@pytest.mark.parametrize('during_command', (False, True))
def test_legacy_host_hooks_never_retire_client_state(during_command):
    desk = Desk()
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    port = SessionAdapter(desk.session, legacy=True)
    before = desk.host.client.link_generation
    desk.host.lines('CONNECTED W9SSJ N0XYZ 500')
    if during_command:
        port.legacy_line('/status')
    port.poll(.1)
    assert desk.host.client.link_generation == before
