"""Operator station planning and native message exchanges, without hardware."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import hflink
from hfmodem.bateleur.ale import words as W
from hfmodem.bateleur.ale.controller import Config, Controller, Event
from hfmodem.bateleur.ale.host import AleHost
from hfmodem.bateleur.ale.modem import AleModem
from hfmodem.bateleur.selcall import fsk
from hfmodem.bateleur.selcall.host import SelcallHost
from hfmodem.bateleur.selcall.modem import SelcallModem
from hfmodem.core.rig import Cat as RigCat


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'station.toml'
    path.write_text('''schema = 1
[station]
mycall = "W9SSJ"
control = "local"
regulatory = "part97"
licence = "extra"
transmit = false
[rig]
model = "ft891"
centre_hz = 7101500
power_w = 100
[rig.ptt]
port = "/fake/ptt"
[audio]
input = "fake input"
output = "fake output"
capture_dir = "/fake/captures"
[protocols.vara]
enabled = true
host = "vara"
cmd_port = 8300
data_port = 8301
''')
    return path


def args(source, *words, family='hflink'):
    return hflink.parser(family).parse_args(['station', '--config', str(source), *words, '--dry-run'])


def test_plan_is_automatic_and_sounds_only_explicit_pilots(source):
    before = source.read_bytes()
    a = args(source, '--channels', '7102', '10145.5', '14109', '--pilot', '7102')
    cfg, dials = hflink.configuration(a, 'hflink')
    assert cfg.control.value == 'automatic'
    assert not cfg.station.transmit
    assert set(cfg.protocols) == {'ale'}
    assert cfg.protocols['ale'].pilot_channels == [7102000]
    assert dials == [7102000, 10145500, 14109000]
    assert cfg.rig.dial_hz == 7102000
    assert source.read_bytes() == before


def test_dry_run_opens_no_station_or_log(source, monkeypatch, capsys):
    def forbidden(*a, **k):
        pytest.fail('dry run opened a station')
    monkeypatch.setattr(hflink.OperatorStation, 'run', forbidden)
    assert hflink.run(['station', '--config', str(source), '--freq', '7102', '--no-sound', '--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['sounding'] == 'off'


def test_execute_does_not_override_transmit_false(source):
    a = args(source, '--freq', '7102', '--no-sound')
    a.execute = True
    with pytest.raises(ValueError, match='transmit=false'):
        hflink.configuration(a, 'hflink')


@pytest.mark.parametrize('words,why', [
    (('--channels', '7102', '7102', '--no-sound'), 'duplicate'),
    (('--freq', '7102', '--pilot', '14109'), 'pilot'),
    (('--freq', '7102.123', '--no-sound'), 'steps'),
    (('--freq', 'nan', '--no-sound'), 'HF USB'),
    (('--channels', '7102', '14109', '--no-sound', '--selcall', '1234'), 'held channel'),
    (('--freq', '7102', '--no-sound', '--acknowledge'), 'own selcall'),
    (('--freq', '7102', '--no-sound', '--summary-seconds', 'nan'), 'finite'),
])
def test_bad_plans_refuse_before_hardware(source, words, why):
    with pytest.raises(ValueError, match=why):
        hflink.configuration(args(source, *words), 'hflink')


def test_selcall_station_keeps_current_dial_if_not_requested(source):
    a = args(source, '--address', '1234', '123456', '--acknowledge', family='selcall')
    cfg, dials = hflink.configuration(a, 'selcall')
    assert dials == []
    assert set(cfg.protocols) == {'selcall'}
    assert cfg.protocols['selcall'].acknowledge
    assert hflink.plan(cfg, dials, a)['dial_khz'] == 'current radio dial'


def test_all_events_logged_but_repeated_heard_and_tunes_quiet():
    journal, screen = io.StringIO(), io.StringIO()
    out = hflink.Output(journal, stream=screen, clock=lambda: 0)
    out.tuned(7102)
    out.line('ale', 'HEARD JOE CHANNEL 7102.0 BER 1 SINAD 10', repeated=True)
    out.line('ale', 'HEARD JOE CHANNEL 7102.0 BER 2 SINAD 11', repeated=True)
    out.line('ale', 'AMD JOE HELLO')
    out.line('ale', 'AMD JOE HELLO')
    out.line('ale', 'TX refused: busy')
    assert len(journal.getvalue().splitlines()) == 6
    assert 'TUNE' not in screen.getvalue()
    assert screen.getvalue().count('HEARD') == 1
    assert screen.getvalue().count('AMD') == 2
    assert 'TX refused: busy' in screen.getvalue()


def test_verbose_shows_every_tune():
    screen = io.StringIO()
    out = hflink.Output(io.StringIO(), stream=screen, verbose=True)
    out.tuned(7102)
    assert 'TUNE 7102 kHz USB' in screen.getvalue()


def test_commands_preserve_native_text_and_refuse_injection():
    assert hflink.command('msg JOE hello there', {'ale'}) == ('ale', 'AMD JOE hello there')
    assert hflink.command('page 5678 hello there', {'selcall'}) == ('selcall', 'PAGE 5678 hello there')
    assert hflink.command('call 5678', {'selcall'}) == ('selcall', 'CALL 5678')
    for text in ('msg JOE hello\nSOUND', 'msg JOE café', 'page 5678 \x1b[0m'):
        with pytest.raises(ValueError, match='printable ASCII'):
            hflink.command(text, {'ale', 'selcall'})
    with pytest.raises(ValueError, match='not enabled'):
        hflink.command('page 5678 hello', {'ale'})


def test_native_amd_and_data_receive_are_printed_and_forwarded():
    modem = AleModem(Config(['SAM'], ['7102.0']), send=lambda a, r: 0)
    host = AleHost(modem, port=0)
    journal, screen = io.StringIO(), io.StringIO()
    observer = hflink.Events(host, 'ale', hflink.Output(journal, stream=screen))
    observer.event(Event('AMD', peer='JOE', channel=0, text='QSY 14109'))
    observer.event(Event('DTM', peer='JOE', channel=0, data='longer native message'))
    assert 'AMD JOE QSY 14109' in screen.getvalue()
    assert 'DTM JOE longer native message' in screen.getvalue()
    assert len(host.log) == 2


def test_console_page_crosses_audio_and_ack_returns(monkeypatch, source):
    wire = []
    def sender(which):
        def send(audio, rate):
            wire.append((which, audio))
            return 0
        return send
    a = SelcallModem('1234', send=sender('a'), card_rate=fsk.FS, dots_s=0.5)
    b = SelcallModem('5678', send=sender('b'), card_rate=fsk.FS, dots_s=0.5, acknowledge=True)
    ah, bh = SelcallHost(a, port=0), SelcallHost(b, port=0)
    journal, screen = io.StringIO(), io.StringIO()
    out = hflink.Output(journal, stream=screen)
    a.observer, b.observer = hflink.Events(ah, 'selcall', out), hflink.Events(bh, 'selcall', out)
    options = args(source, '--address', '1234', family='selcall')
    cfg, dials = hflink.configuration(options, 'selcall')
    operator = hflink.OperatorStation(cfg, output=out, args=options, dials=dials)
    operator.hosts = {'selcall': SimpleNamespace(server=ah)}
    a.start()
    try:
        monkeypatch.setattr(hflink.sys, 'stdin', io.StringIO('page 5678 hello from console\nquit\n'))
        operator.inputs()
    finally:
        a.stop()
    def deliver(to):
        queued = list(wire)
        wire.clear()
        for _, audio in queued:
            x = np.r_[np.zeros(4000), audio, np.zeros(8000)]
            for i in range(0, len(x), 4000):
                to._audio(10000 + i, x[i:i + 4000])
    deliver(b)
    assert 'TEXT HELLO FROM CONSOLE' in screen.getvalue()
    assert wire and wire[0][0] == 'b'
    deliver(a)
    assert 'CALL ACK FROM 5678 TO 1234 MINE' in screen.getvalue()


def test_protocol_message_limits_are_reported_not_truncated(source, monkeypatch):
    m = SelcallModem('1234', send=lambda a, r: pytest.fail('invalid page transmitted'))
    host = SelcallHost(m, port=0)
    m.start()
    try:
        assert host.command('PAGE 5678 ' + 'A' * 33).startswith('ERR ')
    finally:
        m.stop()
    with pytest.raises(ValueError, match='90 characters'):
        W.amd_words('A' * 91)


def test_held_channel_cannot_be_changed_by_calls_or_sounding():
    tuned = []
    ctl = Controller(Config(['SAM'], ['A', 'B'], scanning=True, sound_s=60, listen_s=0),
                     tune=lambda ch: tuned.append(ch) or True, send=lambda w: len(w) * 0.392)
    ctl.stay(1, 0)
    tuned.clear()
    ctl.call('JOE', 1)
    assert ctl._call_plan == []  # selected channel already consumed
    assert ctl.channel == 1
    ctl._set_state(ctl.AVAILABLE)
    ctl._sound_due = [0, 1000]
    ctl.tick(100)
    assert ctl.channel == 1 and tuned == []
    ctl.cfg.scanning = True
    ctl.tick(101)
    assert ctl.channel == 0 and tuned == [0]


def test_bad_retune_and_linked_channel_change_refuse():
    ctl = Controller(Config(['SAM'], ['A', 'B']), tune=lambda ch: ch == 0, send=lambda w: 1.0)
    with pytest.raises(RuntimeError, match='could not tune'):
        ctl.stay(1, 0)
    ctl._set_state(ctl.LINKED)
    with pytest.raises(RuntimeError, match='hang up a link'):
        ctl.stay(0, 0)


def test_operator_checks_usb_before_tuning_and_closes_bad_rig(source, monkeypatch):
    a = args(source, '--freq', '7102', '--no-sound')
    cfg, dials = hflink.configuration(a, 'hflink')
    calls = []
    class Cat:
        def command(self, word):
            assert word == 'm'
            return 'Mode: LSB\nPassband: 4000'
        _field = staticmethod(RigCat._field)
    rig = SimpleNamespace(cat=Cat(), tune=lambda hz: calls.append(('tune', hz)), close=lambda: calls.append('close'))
    op = hflink.OperatorStation(cfg, output=hflink.Output(io.StringIO(), stream=io.StringIO()), args=a, dials=dials)
    def open_rig(self, **kw):
        self.rig = rig
    monkeypatch.setattr(hflink.Station, 'open_rig', open_rig)
    with pytest.raises(hflink.RigError, match='Set USB'):
        op.open_rig()
    assert calls == ['close']


def test_ready_failure_still_shuts_down_station(source, monkeypatch):
    from hfmodem.station import hosts
    a = args(source, '--freq', '7102', '--no-sound')
    cfg, _ = hflink.configuration(a, 'hflink')
    s = hflink.Station(cfg)
    closed = []
    monkeypatch.setattr(s, '_install_signals', lambda: None)
    monkeypatch.setattr(s, 'open_audio', lambda: setattr(s, 'audio', SimpleNamespace(lat=0)))
    monkeypatch.setattr(s, 'open_rig', lambda: None)
    monkeypatch.setattr(s, 'build_lanes', lambda: {})
    monkeypatch.setattr(s, 'shutdown', lambda: closed.append(True))
    monkeypatch.setattr(hosts, 'build', lambda *a, **k: {})
    def failed(station):
        assert station.audio is not None
        raise ValueError('failed operator setup')
    with pytest.raises(ValueError, match='operator setup'):
        s.run(on_ready=failed)
    assert closed == [True]


def test_console_native_amd_round_trip_over_simulated_audio():
    from hfmodem.tests.bateleur.ale import ether as E
    eth = E.Ether(snr_db=15, seed=1)
    a = eth.add(Config(['SAM'], ['7102.0'], scanning=False, listen_s=0))
    b = eth.add(Config(['JOE'], ['7102.0'], scanning=False, listen_s=0))
    eth.run(1)
    modem = SimpleNamespace(ctl=a.ctl, now=eth.now, post=lambda fn: fn())
    host = AleHost(modem, port=0)
    protocol, native = hflink.command('msg JOE hello from console', {'ale'})
    assert protocol == 'ale' and host.command(native) == 'OK AMD JOE'
    eth.run(20, until=lambda: a.ctl.state == b.ctl.state == 'linked')
    assert ('SAM', 'HELLO FROM CONSOLE') in b.ctl.amd_log
    b.ctl.call('SAM', eth.now, amd='reply from peer')
    eth.run(15, until=lambda: bool(a.ctl.amd_log))
    journal, screen = io.StringIO(), io.StringIO()
    observer = hflink.Events(host, 'ale', hflink.Output(journal, stream=screen))
    for event in a.ctl.events:
        observer.event(event)
    assert 'AMD JOE REPLY FROM PEER' in screen.getvalue()


def test_ready_attaches_ui_and_transmit_observation(source, monkeypatch):
    a = args(source, '--freq', '7102', '--no-sound')
    cfg, dials = hflink.configuration(a, 'hflink')
    journal, screen = io.StringIO(), io.StringIO()
    out = hflink.Output(journal, stream=screen)
    op = hflink.OperatorStation(cfg, output=out, args=a, dials=dials)
    modem = AleModem(Config(['SAM'], ['7102.0'], scanning=False), send=lambda a, r: 0,
                     tune=lambda ch: 0)
    host = AleHost(modem, port=0)
    op.lanes = {'ale': SimpleNamespace(modem=modem)}
    op.hosts = {'ale': SimpleNamespace(server=host)}
    monkeypatch.setattr(hflink.sys, 'stdin', io.StringIO('sound\nquit\n'))
    modem.start()
    try:
        op.ready(op)
        # Join the real input worker, not a wall-clock sleep.
        worker = op.operator_thread
        worker.join(timeout=5)
        assert not worker.is_alive()
        modem._tune_radio(0)
    finally:
        modem.stop()
    text = screen.getvalue()
    assert 'TX start' in text and 'TX complete' in text
    assert 'SOUNDED CHANNEL 7102.0' in text
    assert 'Stopping station' in text
    assert out.tunes == 1


def test_station_claims_all_console_arm_locks(mode):
    with hflink.claimed():
        for name in hflink.modes.KNOWN:
            with pytest.raises(RuntimeError, match='holds the lock'):
                with hflink.claim.lease(hflink.modes.resolve(name)):
                    pytest.fail('second operator claimed this station')
    with hflink.claim.lease(mode):
        pass
