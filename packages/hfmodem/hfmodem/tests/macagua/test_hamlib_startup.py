# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Installed Hamlib FTDX10 backend against a PTY, never a physical radio."""
from pathlib import Path
from types import SimpleNamespace
import os
import hashlib
import json
import re
import shutil
import socket
import subprocess
import time
import sys

import pytest

from hfmodem.core import config, ptt
from hfmodem.core.rig import Cat, RigError
from hfmodem.tests.macagua.test_radio_ready import FakePtt
from hfmodem.tests.shrike.ptyrig import PtyRig


class Ftdx10(PtyRig):
    def __init__(self):
        super().__init__()
        self.regs.update(ID='0761', MD0='C', GT0='1', SH0='018', RF0='7',
                         NA0='0', RT='1', XT='1', PC='020',
                         MD1='C', SH1='018', IF='001007100000+000000C00000',
                         OI='001007100000+000000C00000',
                         EX010415='1', EX010416='1', EX010418='1', EX030109='1')
        self.invalid_commands = []

    def _handle(self, command, now):
        payload = command.rstrip(';')
        # Independent syntax from Yaesu FTDX10 CAT manual (2308-F), pp. 9,
        # 13, 15, 17, 19-21 and 23. Do not infer query/set shapes from values.
        if payload[:2] in ('RT', 'XT'):
            self.commands.append((now, command))
            if payload in ('RT', 'XT'):
                self._write(payload + self.regs[payload] + ';')
            elif re.fullmatch(r'(RT|XT)[01]', payload):
                self.regs[payload[:2]] = payload[2]
            else:
                self.invalid_commands.append(command)
                self._write('?;')
            return
        shapes = {'MD': r'MD[01][1-9A-F]?', 'GT': r'GT0[0-4]?',
                  'SH': r'SH[01](?:0(?:0[0-9]|1[0-9]|2[0-3]))?',
                  'RF': r'RF0[1-5]?', 'NA': r'NA0[01]?',
                  'PC': r'PC(?:[0-9]{3})?',
                  'EX': r'EX(?:0104(?:15|16|18)[0-2]?|030109[0-3]?)'}
        shape = shapes.get(payload[:2])
        if shape and not re.fullmatch(shape, payload):
            self.commands.append((now, command))
            self.invalid_commands.append(command)
            self._write('?;')
            return
        prefix = next((p for p in sorted(self.regs, key=len, reverse=True)
                       if payload.startswith(p) and len(p) > 2), None)
        if prefix:
            self.commands.append((now, command))
            value = payload[len(prefix):]
            if value:
                self.regs[prefix] = {'1': '6', '2': '7', '4': '9', '5': 'A'}[value] if prefix == 'RF0' else value
            else:
                self._write(prefix + self.regs[prefix] + ';')
            return
        super()._handle(command, now)


@pytest.fixture
def hamlib(tmp_path, monkeypatch, request):
    required = os.environ.get("HFMODEM_QUALIFICATION_REQUIRED_HAMLIB") == "1"
    executable = (shutil.which('rigctld') or str(Path.home() / 'src/radio/hamlib/_install/bin/rigctld'))
    if required and sys.platform == 'linux':
        executable = shutil.which('rigctld')
        if not executable:
            pytest.fail('required selected PATH Hamlib backend unavailable')
    if not os.path.isfile(executable):
        if required:
            pytest.fail('required installed Hamlib backend unavailable')
        pytest.skip('installed Hamlib required for PTY integration')
    selected = Path(executable).resolve(strict=True)
    expected = os.environ.get('HFMODEM_QUALIFICATION_HAMLIB')
    if expected and selected != Path(expected).resolve(strict=True):
        pytest.fail('selected Hamlib escaped private prefix')
    destination = os.environ.get('HFMODEM_QUALIFICATION_OBSERVATIONS')
    name = hashlib.sha256(request.node.nodeid.encode()).hexdigest()
    records = []
    overflow = [0]
    metadata = dict(selected_path=str(selected),
        binary_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        version=subprocess.check_output([str(selected), '--version'], stderr=subprocess.STDOUT,
                                        text=True, timeout=30),
        requests=records, request_overflow=0)
    if sys.platform == 'linux':
        metadata['library_resolution'] = subprocess.check_output(
            ['ldd', str(selected)], stderr=subprocess.STDOUT, text=True, timeout=30)
    if destination:
        (Path(destination) / ('hamlib-' + name + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    original = Cat.command
    def command(cat, *args, **kwargs):
        # Observe the disposable get_conf connection as well as the main one.
        # Same original call, arguments, return and exception; no proof synthesis.
        line = args[0] if args else kwargs.get('line', '')
        observe = line == '\\get_conf ptt_type' and cat.port == port
        entered = time.monotonic() if observe else None
        reply = None
        error = None
        try:
            reply = original(cat, *args, **kwargs)
            return reply
        except Exception as exc:
            if observe:
                error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            if observe:
                stopped = time.monotonic()
                if len(records) < 8:
                    records.append(dict(request=line, reply=reply, error=error,
                                        entry=entered, exit=stopped))
                else:
                    overflow[0] += 1

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    monkeypatch.setattr(Cat, "command", command)
    radio = Ftdx10()
    with (tmp_path / 'rigctld.log').open('w') as log:
        daemon = subprocess.Popen([executable, '-m', '1042', '-r', radio.path,
            '-s', '38400', '--set-conf=ptt_type=None', '-T', '127.0.0.1',
            '-t', str(port), '-vvvvv'], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 12
            while True:
                if daemon.poll() is not None:
                    pytest.fail((tmp_path / 'rigctld.log').read_text())
                try:
                    with socket.create_connection(('127.0.0.1', port), .1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        pytest.fail('PTY Hamlib startup timed out')
                    time.sleep(.05)
            yield radio, port
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=2)
            radio.close()
            if destination:
                metadata['request_overflow'] = overflow[0]
                metadata['daemon_exit_code'] = daemon.returncode
                (Path(destination) / ('hamlib-' + name + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
                shutil.copyfile(tmp_path / 'rigctld.log', Path(destination) / ('rigctld-' + name + '.log'))


@pytest.mark.parametrize('command', ['\\send_cmd MD08;', '\\send_raw ; RT0;', '\\send_raw ; XT0;'])
def test_old_raw_setter_waits_for_answer_the_radio_never_sends(hamlib, command):
    radio, port = hamlib
    cat = Cat(port=port)
    try:
        cat.open()
        with pytest.raises(RigError, match='failed: timed out'):
            cat.command(command)
        # These all set state without answering; none is a valid query.
        register, value = ('MD0', '8') if 'MD08' in command else (command.split()[-1][:2], '0')
        assert radio.regs[register] == value
    finally:
        cat.close()


@pytest.mark.parametrize('dial', [7_208_000, 14_285_000])
def test_full_voice_startup_through_installed_backend_without_keying(hamlib, monkeypatch, dial):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / 'tools'))
    from consolelib.voice import radio as prepare
    from hfmodem.contact.radio import verify
    wire, port = hamlib
    cfg = config.parse(dict(schema=1, station=dict(mycall='W9SSJ', grid='EN63bb',
        transmit=True, control='local', regulatory='unregulated', because='PTY test'),
        rig=dict(model='ftdx10', port=port, ptt=dict(port='fake', settle_s=0))), environ={})
    line = FakePtt()
    monkeypatch.setattr(ptt, 'RtsPtt', lambda *args: line)
    session = SimpleNamespace(cfg=cfg, dial_hz=dial, rig=None, event=lambda *args, **kwargs: None)
    try:
        prepare(session, dial)
        state = verify(session)
        assert state['dial_hz'] == dial
        assert state['mode'] == ('DATA-L' if dial < 10_000_000 else 'DATA-U')
        assert True not in line.calls and not wire.edges
        assert line.calls == ['open']
        assert wire.regs['RT'] == wire.regs['XT'] == '0'
        assert not wire.invalid_commands
    finally:
        if session.rig:
            session.rig.close()
