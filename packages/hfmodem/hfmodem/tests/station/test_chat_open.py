# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Successful attachment composition using only injected, in-memory facades."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from hfmodem.station import chat
from hfmodem.bbs import config as bbs_config, service as bbs_service
from hfmodem.contact import slots
from hfmodem.core import wav
from hfmodem.kestrel.host import audio_runtime, station_runtime
from hfmodem.station import chat_capture, contactlog


@pytest.mark.parametrize('with_cat', [False, True])
def test_successful_open_borrows_owned_runtime_and_registers_authority_before_listen(tmp_path, monkeypatch, with_cat):
    events, seen = [], {}
    authority, evidence = object(), object()
    core, catalog, mailbox = object(), object(), object()
    gate = chat.AttachmentGate()

    def forbidden(*args, **kwargs):
        pytest.fail('attachment attempted a second runtime, device owner, socket or rig arm')

    class Fence:
        identity = ('operator', 'exact-lease', 4)

        def check(self):
            events.append('fence')

        def __call__(self):
            return True

    class Stream:
        def __init__(self):
            self.closed = False

        def settimeout(self, value):
            assert value == 1

        def close(self):
            self.closed = True

    pairs = []
    def socketpair():
        pair = (Stream(), Stream())
        pairs.append(pair)
        return pair

    class Thread:
        def __init__(self, *, target, name):
            assert name == 'station-chat-adapter'
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            events.append('host-joined')

        def is_alive(self):
            return False

    class Host:
        def __init__(self, command, data, modem, *, kiss, fence):
            assert command is pairs[0][1] and data is pairs[1][1]
            assert modem is seen['modem'] and fence is owner.fence
            self._modem = modem
            self.delivery_boundary = object()
            self.submitter_live = True
            seen['host'] = self

        def run(self):
            events.append('host-running')

        def beacon_submit(self, text):
            forbidden()

        def close(self, *, deadline=None):
            events.append('host-closed')

        def ensure_retired(self, *, deadline):
            events.append('host-retired')

    class Client:
        def __init__(self, command, data, *, ready, beacon_submit):
            assert command is pairs[0][0] and data is pairs[1][0]
            assert ready.__self__ is attachment
            assert beacon_submit.__self__ is seen['host']
            self.ptt = False
            seen['client'] = self

        def initialize(self, mycall):
            assert mycall == 'W9SSJ'
            assert seen['modem'] is not None
            events.append('initialized')

        def enable_writer(self):
            self.data_writer = SimpleNamespace(before_write=None,
                _thread=SimpleNamespace(join=lambda timeout: events.append('writer-joined'),
                                        is_alive=lambda: False))
            events.append('writer-enabled')

        def close(self, *, deadline=None):
            events.append('client-closed')

    class Flow:
        def __init__(self, client, tune, *, band, profile):
            assert client is seen['client'] and tune.__self__ is attachment
            assert band == 20
            assert profile == {'NAME': 'Saul', 'QTH': 'Chicago', 'LOC': 'EN63'}

        def received_tuned(self, *args):
            pass

    class Session:
        def __init__(self, client, flow, **kwargs):
            assert kwargs['bbs_core'] is core
            assert kwargs['bbs_catalog'] is catalog
            assert kwargs['bbs_reload'] is reload_bbs
            assert kwargs['mailbox'] is mailbox
            assert kwargs['slots'] is seen['slots']
            assert kwargs['idcall'] == 'W9SSJ'
            assert kwargs['arm'].__self__ is attachment
            assert kwargs['identify'].__self__ is attachment
            assert kwargs['band_tuner'].__self__ is attachment
            if with_cat:
                assert kwargs['power'].__self__ is attachment
            else:
                assert kwargs['power'] is None
            assert kwargs['power_clamp_w'] == 25
            from hfmodem.contact.mail_parking import ParkedStore
            self.parking = SimpleNamespace(store=ParkedStore(tmp_path / 'vmail'))
            self.options = kwargs
            self.client = client
            self.flow = flow
            seen['session'] = self

        def _apply_power(self, asked, watts):
            self.options['power'](watts)

        def monitor(self):
            if with_cat:
                assert watts == [25]
            assert not attachment.allowed()
            self.options['arm'](True)
            assert attachment.allowed()
            events.append('listen')

        def quit(self, *, unwinding=None):
            assert self.parking.store.available
            self.options['arm'](False)
            events.append('quit')

    class Capture:
        def __init__(self, audio, recorder, watch, *, keyed, contact_log):
            assert audio is station.audio
            assert recorder is seen['recorder'] and watch is seen['slots']
            assert keyed() is False
            self.audio = audio
            seen['capture'] = self

        def close(self, *, deadline):
            events.append('capture-closed')

    @contextmanager
    def subscribe_payload_evidence():
        events.append('evidence-subscribed')
        yield evidence
        events.append('evidence-unsubscribed')

    def modem_factory(*, authority, fence):
        assert authority is registered.authority
        assert events.index('registered') < len(events)
        assert not fence()  # Private arm is still disarmed before Session.monitor.
        seen['modem'] = SimpleNamespace(_retired_attachment=True)
        events.append('modem-created')
        return seen['modem']

    def register(**kwargs):
        assert kwargs['kind'] == 'chat' and kwargs['lane'] == 'vara'
        assert kwargs['attended'] is True and kwargs['holds_rig'] is True
        assert kwargs['client'] == 'operator'
        assert kwargs['recording_origin_index'] == station.audio.samples
        assert kwargs['directory'] == tmp_path / 'chat/incarnation/1'
        events.append('registered')
        return registered

    def diagnostics_subscribe(submitted_authority, callback):
        assert submitted_authority is authority
        assert callback.__self__ is attachment
        events.append('diagnostics-subscribed')
        return lambda: events.append('diagnostics-unsubscribed')

    def make_slots(*args, **kwargs):
        assert args == (14_105_000, chat.CALLING_HZ[20])
        assert kwargs == {'mycall': 'W9SSJ', 'passband': 3000}
        seen['slots'] = object()
        return seen['slots']

    def make_recorder(path, **kwargs):
        assert path == tmp_path / 'chat/incarnation/1/receive.wav'
        assert kwargs == {'rate': 48000, 'source': 'station chat'}
        seen['recorder'] = object()
        return seen['recorder']

    def reload_bbs():
        pass

    registered = SimpleNamespace(id='registered-chat', authority=authority)
    bbs = SimpleNamespace(core=core, catalog=catalog, reload=reload_bbs,
                          close=lambda **kwargs: events.append('bbs-closed'))
    runtime = SimpleNamespace(attachments=gate, failure=None,
        modem_factory=modem_factory, owner=SimpleNamespace(subscribe_payload_evidence=subscribe_payload_evidence))
    watts = [100.]
    cat = SimpleNamespace(clock=lambda: 0, read=lambda *args: f'{int(watts[0]):03d}',
                          set_power=lambda value: watts.__setitem__(0, value)) if with_cat else None
    station = SimpleNamespace(lanes={'vara': SimpleNamespace(runtime=runtime)},
        audio=SimpleNamespace(samples=96000, close=forbidden),
        rig=SimpleNamespace(line_state='low', dial=lambda: 14_105_000, arm=forbidden, cat=cat,
            operations=SimpleNamespace(lease=nullcontext), hold=lambda deadline: nullcontext(True)),
        arbiter=SimpleNamespace(stop=lambda id: events.append('authority-stopped')),
        diagnostics=SimpleNamespace(subscribe=diagnostics_subscribe),
        is_ready=True, station_file=str(tmp_path / 'station.toml'),
        cfg=SimpleNamespace(bbs=SimpleNamespace(config='bbs.toml'),
            station=SimpleNamespace(mycall='W9SSJ', idcall='', name='Saul', qth='Chicago', grid='EN63'),
            rig=SimpleNamespace(filter_hz=3000, power_clamp_w=25, cat='yaesu')),
        sessions=SimpleNamespace(start=register,
            running=lambda id: events.append('running'),
            end=lambda *args, **kwargs: events.append('session-ended')))
    owner = SimpleNamespace(station=station, fence=Fence(), session=None,
                            service=SimpleNamespace(incarnation='incarnation'), generation=1)
    attachment = chat.Attachment(owner)

    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(audio_runtime, 'AudioHostRuntime', forbidden)
    monkeypatch.setattr(station_runtime, 'StationVaraRuntime', forbidden)
    monkeypatch.setattr(chat.socket, 'socket', forbidden)
    monkeypatch.setattr(chat.socket, 'socketpair', socketpair)
    monkeypatch.setattr(chat, 'threading', SimpleNamespace(Thread=Thread))
    monkeypatch.setattr(chat, 'PrivateSession', Host)
    monkeypatch.setattr(chat, 'Client', Client)
    monkeypatch.setattr(chat, 'Flow', Flow)
    monkeypatch.setattr(chat, 'Session', Session)
    monkeypatch.setattr(chat, 'Mailbox', lambda path: mailbox)
    monkeypatch.setattr(chat, 'AttachedSession', lambda instance, **kwargs: SimpleNamespace(attachment=instance, **kwargs))
    monkeypatch.setattr(slots, 'SlotWatch', make_slots)
    monkeypatch.setattr(wav, 'CaptureRecorder', make_recorder)
    monkeypatch.setattr(chat_capture, 'ChatCapture', Capture)
    monkeypatch.setattr(contactlog, 'ContactLog', lambda path: object())
    monkeypatch.setattr(bbs_config, 'load', lambda path: None)
    monkeypatch.setattr(bbs_service, 'open_service', lambda path: bbs)

    try:
        adapter = attachment.open()
        assert attachment.runtime is station.lanes['vara'].runtime is runtime
        assert seen['capture'].audio is station.audio
        assert attachment._authority is registered.authority is authority
        assert owner.session is registered
        assert adapter.attachment is attachment
        assert adapter.delivery_evidence is evidence
        assert adapter.delivery_boundary is seen['host'].delivery_boundary
        assert seen['client'].data_writer.before_write == owner.fence.check
        assert events.index('registered') < events.index('modem-created') < events.index('initialized') < events.index('listen') < events.index('running')
        assert len(pairs) == 2
        assert not attachment._initializing
    finally:
        attachment.close()
        attachment.release()
    assert not seen['session'].parking.store.available
    from hfmodem.contact.mail_parking import ParkedStore
    reopened = ParkedStore(tmp_path / 'vmail')
    assert reopened.available
    reopened.close()
    assert all(stream.closed for pair in pairs for stream in pair)
    assert events.index('capture-closed') < events.index('session-ended')
    assert 'diagnostics-unsubscribed' in events and 'evidence-unsubscribed' in events
    # Successful retirement releases the same borrowed runtime's attachment gate.
    gate.reserve(object())
