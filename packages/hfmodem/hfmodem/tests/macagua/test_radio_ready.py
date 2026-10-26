# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""First-flight failure paths through real Rig control, with no hardware."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from hfmodem.contact.runtime import Session
from hfmodem.contact.radio import raw, verify
from hfmodem.contact.terminal import Terminal
from hfmodem.core.rig import Cat as RealCat, Rig
from hfmodem.core import config


class FakeCat(RealCat):
    def __init__(self, host=None, port=None):
        super().__init__()
        self.operations_log = []
        self.hz = 14285000
        self.fields = dict(VFO='VFOA', Split='0')
        self.values = dict(VS='0', MD0='C', RT='0', XT='0', GT0='1', SH0='018',
                           RF0='7', NA0='0', PC='020', EX010415='1', EX010416='1', EX010418='1')

    def open(self):
        self.operations_log.append('cat-open')
        self.poisoned = False

    def close(self):
        self.operations_log.append('cat-close')
        super().close()

    def identify(self): return 'FTDX10'
    def emergency_unkey(self): return False
    def freq(self): return self.hz
    def set_freq(self, hz): self.hz = hz
    def ptt_state(self, **kwargs): return False
    def get_conf(self, name, **kwargs): return self.command('\\get_conf ' + name, **kwargs)

    def command(self, command, **kwargs):
        self.operations_log.append(command)
        if command == '\\get_conf ptt_type':
            return 'ptt_type=None\nRPRT 0'
        if command in ('v', 's'):
            key = 'VFO' if command == 'v' else 'Split'
            return f'{key}: {self.fields[key]}\nRPRT 0'
        if command == '\\set_vfo VFOA':
            self.fields['VFO'] = 'VFOA'
            self.values['VS'] = '0'
        elif command.startswith('L AGC '):
            self.values['GT0'] = {2: '1', 3: '3', 5: '2'}[int(command.split()[-1])]
        elif command.startswith('L RFPOWER '):
            self.values['PC'] = f'{round(float(command.split()[-1])*100):03d}'
        elif command.startswith('\\send_raw '):
            mode = command.split()[1]
            assert mode in ('0', ';')
            payload = command.split()[-1].rstrip(';')
            if payload in self.values:
                assert mode == ';', 'readbacks must request a terminated reply'
                return f'send_raw: ; {payload};\nSend raw answer: {payload}{self.values[payload]};\nRPRT 0'
            assert mode == '0', 'setters must not wait for a reply'
            prefix = next((p for p in sorted(self.values, key=len, reverse=True) if payload.startswith(p)), None)
            if prefix:
                value = payload[len(prefix):]
                self.values[prefix] = {'1': '6', '2': '7', '4': '9', '5': 'A'}[value] if prefix == 'RF0' else value
            else:
                raise AssertionError(command)
        else:
            raise AssertionError(command)
        return 'RPRT 0'


class FakePtt:
    def __init__(self, port=None):
        self.calls = []
        self.high = False
    def open(self): self.calls.append('open')
    def assert_(self, on): self.calls.append(on); self.high = on
    def sense(self): return self.high
    def release(self): self.assert_(False)


class Audio:
    class CallbackStop(Exception): pass
    class CallbackAbort(Exception): pass
    default = SimpleNamespace(device=(0, 1))
    def __init__(self): self.streams = []; self.checks = []
    def query_devices(self):
        return [dict(name='Rig codec', max_input_channels=1, max_output_channels=1),
                dict(name='Local speakers', max_input_channels=0, max_output_channels=2)]
    def check_input_settings(self, **settings): self.checks.append(('input', settings))
    def check_output_settings(self, **settings): self.checks.append(('output', settings))
    def OutputStream(self, **settings):
        audio = self
        class Stream:
            def __init__(self): self.closed = False; self.settings = settings
            def start(self): pass
            def stop(self): pass
            def abort(self): pass
            def close(self): self.closed = True
            def frame(self, status=None):
                try:
                    settings['callback'](np.zeros((1536, 1), np.float32), 1536, None, status)
                except (audio.CallbackStop, audio.CallbackAbort):
                    settings['finished_callback']()
        stream = Stream()
        self.streams.append(stream)
        return stream


@pytest.fixture
def live(tmp_path, monkeypatch):
    cfg = config.parse(dict(schema=1, station=dict(mycall='W9SSJ', grid='EN63bb',
        transmit=True, control='local', regulatory='unregulated', because='hardware-free test'),
        rig=dict(model='ftdx10', power_w=20, ptt=dict(port='fake', settle_s=0))), environ={})
    audio = Audio()
    monkeypatch.setitem(sys.modules, 'sounddevice', audio)
    session = Session(cfg, tmp_path, dial_hz=14285000, input_device='Rig codec', output_device='Rig codec')
    ptt, cat = FakePtt(), FakeCat()
    session.rig = Rig(model='ftdx10', cat=cat, ptt=ptt, profile=cfg.profile,
                      control=cfg.station.control, transmit=True, max_key_s=30)
    session.rig.arm(prove_ptt=False)
    session.prepare_audio()
    session.radio_check = lambda: verify(session)
    session.draft = dict(kind='cq', text='CQ', audio=np.ones(1536, np.float32)*.1,
                         duration_s=.032, wav='fake.wav')
    yield session, cat, ptt, audio
    session.close()


@pytest.mark.parametrize('prefix,reply', [
    ('MD0', 'send_cmd: MD0;\nReply: ?;'), ('MD0', 'MD0C; MD08;'),
    ('MD0', 'send_cmd: MD0C;\nRPRT 0'), ('EX010415', 'EX0104161;'),
    ('MD0', 'send_raw: ; MD0C;\nSend raw answer: No answer\nRPRT 0'),
])
def test_echo_missing_and_ambiguous_readback_is_not_verification(prefix, reply):
    with pytest.raises(ValueError, match='readback'):
        raw(SimpleNamespace(command=lambda _: reply), prefix+';', prefix)


@pytest.mark.parametrize('field,value', [('VS', '1'), ('MD0', '2'), ('RT', '1'), ('XT', '1'),
    ('EX010415', '0'), ('EX010416', '0'), ('EX010418', '0'), ('GT0', '0'),
    ('SH0', '011'), ('RF0', '9'), ('NA0', '1'), ('PC', '100')])
def test_radio_state_change_prevents_keying(live, field, value):
    session, cat, ptt, audio = live
    cat.values[field] = value
    with pytest.raises(ValueError):
        session.play()
    assert True not in ptt.calls and not audio.streams


@pytest.mark.parametrize('change', ['dial', 'vfo', 'split'])
def test_dial_vfo_or_split_change_prevents_keying(live, change):
    session, cat, ptt, audio = live
    if change == 'dial': cat.hz += 1000
    elif change == 'vfo': cat.fields['VFO'] = 'VFOB'
    else: cat.fields['Split'] = '1'
    with pytest.raises(ValueError): session.play()
    assert True not in ptt.calls and not audio.streams


def test_complete_over_keys_only_on_enter_then_unkeys_and_advances(live):
    session, cat, ptt, audio = live
    terminal = Terminal(session)
    assert True not in ptt.calls
    terminal.key('\n')
    assert audio.streams[-1].settings['device'] == 0
    assert ptt.high and session.contact.state == 'CQ'
    audio.streams[-1].frame()
    terminal.step()
    assert not ptt.high and session.contact.state == 'LISTEN' and session.draft['kind'] == 'cq'


def test_enter_repeats_exact_cq_and_an_incoming_caller_invalidates_it(live):
    from hfmodem.contact.frame import Over
    session, cat, ptt, audio = live
    terminal = Terminal(session)
    draft = session.draft
    for _ in range(2):
        terminal.key('\n')
        assert session.playing['draft'] is draft and ptt.high
        audio.streams[-1].frame()
        terminal.step()
        assert session.draft is draft and not ptt.high
    assert len(audio.streams) == 2 and session.pending is None
    terminal.heard_event(dict(kind='heard', recognition=dict(text='KE4QJW'),
        over=Over(de='KE4QJW').record()))
    assert session.draft is None and session.contact.state == 'CONFIRM'
    terminal.key('\n')
    assert len(audio.streams) == 2 and not ptt.high


def test_underrun_aborts_unkeys_and_retains_draft(live):
    session, cat, ptt, audio = live
    session.play()
    audio.streams[-1].frame('output underflow')
    Terminal(session).step()
    assert not ptt.high and session.contact.state == 'CQ' and session.draft


def test_cancelled_stream_callback_cannot_finish_or_modify_new_over(live):
    session, cat, ptt, audio = live
    session.play()
    old = audio.streams[-1]
    session.finish_playback(cancelled=True)
    session.play()
    new_playing = session.playing
    old.frame('stale underflow')
    assert not session.output_done.is_set() and 'audio_status' not in new_playing
    assert ptt.high
    audio.streams[-1].frame()
    Terminal(session).step()
    assert not ptt.high


@pytest.mark.parametrize('preview', [None, 'Rig codec', 0])
def test_preview_cannot_follow_default_or_alias_to_rig(live, preview):
    session, cat, ptt, audio = live
    audio.default = SimpleNamespace(device=(0, 0))
    session.preview_device = preview
    with pytest.raises(ValueError, match='local speakers'): session.play(preview=True)
    assert True not in ptt.calls and not audio.streams


def test_local_preview_never_keys(live):
    session, cat, ptt, audio = live
    session.preview_device = 'Local speakers'
    session.play(preview=True)
    assert audio.streams[-1].settings['device'] == 1 and True not in ptt.calls
    audio.streams[-1].frame()
    Terminal(session).step()
    assert session.draft and session.contact.state == 'CQ'


def test_receive_failure_blocks_send_even_before_terminal_drains_event(live):
    session, cat, ptt, audio = live
    session._receive_failed(RuntimeError('worker lost'))
    with pytest.raises(ValueError, match='Receive failed'): session.play()
    assert True not in ptt.calls and not audio.streams


def test_capture_device_stops_during_over_and_unkeys_at_next_step(live):
    session, cat, ptt, audio = live
    session.capture = SimpleNamespace(active=True, abort=lambda: None, close=lambda: None)
    session.play()
    session.capture.active = False
    terminal = Terminal(session)
    terminal.step()
    assert not ptt.high and not terminal.receive_ok
    assert session.playing is None and session.pending is None


def test_audio_close_failure_cannot_log_a_completed_contact(live):
    session, cat, ptt, audio = live
    session.play()
    audio.streams[-1].frame()
    def broken(): raise OSError('codec disconnected')
    audio.streams[-1].stop = broken
    with pytest.raises(OSError): session.finish_playback()
    assert not ptt.high and audio.streams[-1].closed
    assert session.contact.state == 'CQ' and session.draft


def test_shutdown_cleans_models_even_when_capture_abort_fails(live):
    session, cat, ptt, audio = live
    cleaned = []
    def broken(): raise OSError('capture disconnected')
    session.capture = SimpleNamespace(abort=broken, close=lambda: cleaned.append('capture'))
    session.speech = SimpleNamespace(close=lambda: cleaned.append('speech'))
    session.language = SimpleNamespace(close=lambda: cleaned.append('language'))
    with pytest.raises(OSError): session.close()
    assert cleaned == ['capture', 'speech', 'language'] and session.rig is None
    assert not ptt.high


def test_radio_startup_prepares_and_reads_back_without_keying(live, monkeypatch):
    # Run the actual console setup against an actual Rig and fake transports.
    from consolelib import voice
    from hfmodem.core import rig as core_rig, ptt as core_ptt
    session, cat, line, audio = live
    session.rig.close()
    session.rig = None
    monkeypatch.setattr(core_rig, 'Cat', lambda *args: cat)
    monkeypatch.setattr(core_ptt, 'RtsPtt', lambda *args: line)
    voice.radio(session, 14285000)
    assert True not in line.calls and session.rig.dial() == 14285000
    assert line.calls[-1] == 'open'
    assert verify(session)['power_w'] == 20


def test_lower_sideband_startup_keeps_exact_requested_phone_dial(live, monkeypatch):
    from consolelib import voice
    from hfmodem.core import rig as core_rig, ptt as core_ptt
    session, cat, line, audio = live
    session.rig.close()
    session.rig = None
    session.dial_hz = 7208000
    monkeypatch.setattr(core_rig, 'Cat', lambda *args: cat)
    monkeypatch.setattr(core_ptt, 'RtsPtt', lambda *args: line)
    voice.radio(session, 7208000)
    assert True not in line.calls
    assert verify(session)['mode'] == 'DATA-L' and cat.hz == 7208000


def test_missing_daemon_ptt_type_refuses_startup_before_line_open(live, monkeypatch):
    from consolelib import voice
    from hfmodem.core import rig as core_rig, ptt as core_ptt
    session, cat, line, audio = live
    session.rig.close()
    session.rig = None
    line.calls.clear()
    monkeypatch.setattr(core_rig, 'Cat', lambda *args: cat)
    monkeypatch.setattr(core_ptt, 'RtsPtt', lambda *args: line)
    monkeypatch.setattr(cat, 'get_conf', lambda *args, **kwargs: 'RPRT 0')
    with pytest.raises(ValueError, match='did not name its PTT owner'):
        voice.radio(session, 7208000)
    assert not line.calls


def test_band_recall_happens_before_voice_settings_are_applied(live, monkeypatch):
    from consolelib import voice
    from hfmodem.core import rig as core_rig, ptt as core_ptt
    session, cat, line, audio = live
    session.rig.close()
    session.rig = None
    session.dial_hz = 7208000
    def recall(hz):
        cat.hz = hz
        cat.values.update(MD0='2', GT0='0', SH0='011', RF0='9', NA0='1')
    monkeypatch.setattr(cat, 'set_freq', recall)
    monkeypatch.setattr(core_rig, 'Cat', lambda *args: cat)
    monkeypatch.setattr(core_ptt, 'RtsPtt', lambda *args: line)
    voice.radio(session, 7208000)
    assert verify(session)['mode'] == 'DATA-L' and True not in line.calls


def test_unsupported_sample_format_fails_before_radio(live, monkeypatch):
    session, cat, line, audio = live
    def unsupported(**kwargs): raise ValueError('48 kHz unsupported')
    monkeypatch.setattr(audio, 'check_output_settings', unsupported)
    with pytest.raises(ValueError, match='48 kHz'): session.prepare_audio()
    assert True not in line.calls
