# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.ax25 import kiss
from hfmodem.contact import varac as v
from hfmodem.contact.broadcast import Broadcast, encode_ui
from hfmodem.kestrel.host.vara_modem import _HostIO
from hfmodem.kestrel.rx.cq_stream import CQDetector
from hfmodem.kestrel.tx.cq import synth_cq
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
from hfmodem.kestrel.vara import vara_arq as arq, vara_frames as frames
from hfmodem.tests.contact.test_varac import Desk
from hfmodem.tests.kestrel.test_cq import session_mock


class Endpoint:
    def __init__(self, pair, name, root):
        self.pair, self.name = pair, name
        self.desk = Desk(mailbox=root, kiss=True)
        self.server, _, self.modem = session_mock()
        self.server._send_cmd = self.desk.host.lines
        self.server._send_data = self._received
        self.server._kiss = SimpleNamespace(deliver_frame=lambda command, raw:
            self.desk.host.kiss.incoming.append(kiss.encode(raw, command)))
        self.modem._transport.tx = self._audio
        self.desk.host.command.respond = self._command
        self.desk.host.data.respond = self.modem.transmit
        self.waves = []
        self.ui_audio = deque()
        self.rename(name)

    @property
    def session(self):
        return self.desk.session

    def rename(self, name):
        assert not self.desk.host.client.connected
        self.name = name
        self.desk.host.client.mycall = name
        self.server._dispatch('MYCALL ' + name)
        self.server._dispatch('COMPRESSION OFF')
        self.server._dispatch('CHAT ON')
        self.server._dispatch('LISTEN CQ')
        self.modem.poll()

    def _audio(self, samples):
        self.waves.append(np.array(samples, copy=True))
        if self.modem._broadcast_pending is not None:
            self.ui_audio.append(np.array(samples, copy=True))

    def _received(self, data):
        for start in range(0, len(data), 7):
            self.desk.host.data.incoming.append(data[start:start + 7])

    def _command(self, raw):
        text = raw.decode().rstrip('\r')
        if text.startswith('CONNECT '):
            _, source, destination = text.split()
            assert source == self.name
            self.desk.host.lines('OK')
            self.pair.connect(self, destination)
        elif text in ('DISCONNECT', 'ABORT'):
            self.desk.host.lines('OK')
            self.pair.disconnect()
        else:
            self.server._dispatch(text)
            self.modem.poll()


class Cable:
    def __init__(self, root):
        self.events = []
        self.drop_smr = False
        self.a = Endpoint(self, 'K1ABC', root / 'a')
        self.b = Endpoint(self, 'W9SSJ', root / 'b')
        self.ends = (self.a, self.b)
        self.linked = False
        self.calling = None

    def other(self, endpoint):
        return self.b if endpoint is self.a else self.a

    def connect(self, caller, destination):
        called = self.other(caller)
        assert called.name == destination and not self.linked
        if not called.server.listening:
            self.calling = (caller, destination)
            return
        self.calling = None
        self.linked = True
        for end in self.ends:
            end.modem._replace_station()
            station = end.modem.station
            station.role = 'initiator' if end is caller else 'responder'
            station.caller, station.called = caller.name, called.name
            station.state = arq.VaraState.CONNECTED
            station.step = arq._I_CONNECTED if end is caller else arq._R_RESP_SENT
            station.turn = arq._TURN_PEER
            station._link_high_speed, station._rec500 = True, 7
            station._lead = VB6Rnd(frames.lead_state(caller.name, False, '500'))
            station._start_compression('OFF')
            station.io.connected(caller.name, called.name, '500')
            station.io.snr(-3)
        self.events.append(('connected', caller.name, called.name))

    def disconnect(self):
        self.calling = None
        if not self.linked:
            return
        self.linked = False
        for end in self.ends:
            end.modem._connected = end.modem._intent_active = False
            end.modem._replace_station()
            end.server.modem_disconnected()
        self.events.append(('disconnected',))

    def _data(self, end):
        station = end.modem.station
        if not self.linked or not station.pending_bytes:
            return
        station._turn_granted(np.zeros(0), said='simulation peer granted turn')
        for _ in range(64):
            pending = station._tx_pending
            if pending is None:
                break
            from hfmodem.kestrel.arq.phy import vara_payload, over_is_last
            body = pending[0]
            bodies = ([station._tx_pair[1], body] if station._tx_pair else [body])
            payload = b''.join(vara_payload(item, caller=station.caller,
                                            body_len=len(item)) for item in bodies)
            self.events.append(('data', end.name, self.other(end).name,
                                end.desk.flow.dial, payload))
            if not (self.drop_smr and b'<SMR>' in payload):
                self.other(end).modem.station._deliver(bodies)
            station._took_control_burst(continue_reply=not over_is_last(body, station.caller))
        else:
            pytest.fail('connected simulation did not drain its bounded transmit queue')
        end.modem.poll()

    def pump(self, until=lambda: False, count=1000):
        for _ in range(count):
            if self.calling:
                self.connect(*self.calling)
            for end in self.ends:
                end.modem.poll()
                self._data(end)
                while end.ui_audio:
                    wave = end.ui_audio.popleft()
                    detector = CQDetector()
                    got = []
                    audio = np.r_[np.zeros(24000), wave, np.zeros(6 * 48000)]
                    for start in range(0, len(audio), 4800):
                        got.extend(detector.feed(audio[start:start + 4800]))
                    assert len(got) == 1 and isinstance(got[0], Broadcast)
                    self.events.append(('ui', got[0]))
                    other = self.other(end)
                    _HostIO(other.modem, other.modem._transport).cq(got[0])
                end.session.tick(.02)
            if any(end.desk.host.client.data_writer is not None for end in self.ends):
                time.sleep(.001)
            if until():
                return
        pytest.fail('two-station simulation did not reach its application boundary: ' +
                    repr([(end.name, end.desk.said[-5:], end.desk.flow.state,
                           end.modem.station.pending_bytes) for end in self.ends]))

    def idle(self):
        self.pump(until=lambda: all(end.session.steps is None and
                  not end.desk.host.command.incoming and not end.desk.host.data.incoming
                  and end.modem._events.empty() and not end.modem.station.pending_bytes
                  and not (end.desk.host.client.data_writer and end.desk.host.client.data_writer.pending)
                  and not end.desk.host.client._writes
                  for end in self.ends))

    def deposit(self, text):
        self.a.session.command('/call W9SSJ cf')
        self.pump(until=lambda: all(end.desk.host.client.connected for end in self.ends))
        self.idle()
        self.a.desk.host.client.send_text(text)
        self.idle()

    def recipient(self):
        self.disconnect()
        self.idle()
        self.a.rename('W9SSJ-1')
        self.idle()

    def restart(self):
        assert not self.linked
        old_a, old_b = self.ends
        for end in self.ends:
            end.session.parking.store.close()
            end.desk.host.client.close()
        self.a = Endpoint(self, old_a.name, old_a.desk.mailbox.root)
        self.b = Endpoint(self, old_b.name, old_b.desk.mailbox.root)
        self.ends = (self.a, self.b)
        self.events.append(('restarted',))
        self.idle()

    def beacon(self):
        self.b.session.band_ready(True)
        self.b.server.modem_busy(False)
        self.b.desk.host.now += v.BEACON_CLEAR_S + 1
        detector = CQDetector()
        got = []
        audio = np.r_[synth_cq('W9SSJ-9', 500), np.zeros(48000)]
        for start in range(0, len(audio), 4800):
            got.extend(detector.feed(audio[start:start + 4800]))
        assert len(got) == 1 and got[0].call == 'W9SSJ-9'
        _HostIO(self.b.modem, self.b.modem._transport).cq(got[0])
        self.idle()

    def collect(self):
        self.a.session.command('/vmail collect W9SSJ')
        self.pump(until=lambda: bool(self.a.desk.mailbox.entries()) and
                  all(row['state'] == 'delivered' for row in self.b.session.parking.store.records))
        self.idle()


@pytest.fixture
def cable(tmp_path):
    pair = Cable(tmp_path)
    yield pair
    for end in pair.ends:
        end.session.parking.store.close()
        end.desk.host.client.close()


DEPOSIT = ('<SM><TME:2026-10-28 12:00:00><FRM:K1ABC><TO:W9SSJ-1>'
           '<SBJ:Through the relay><MSG:café\r\n73>')


def test_injected_third_party_custody_beacon_notice_qsy_forward_and_smr(cable):
    cable.deposit(DEPOSIT)
    record, = cable.b.session.parking.store.records
    assert (record['from'], record['to'], record['via'], record['state']) == (
        'K1ABC', 'W9SSJ-1', 'K1ABC', 'queued')
    assert json.loads(cable.b.session.parking.store.path.read_text())['records'] == [record]
    assert not cable.b.desk.mailbox.entries()
    assert any(event[0] == 'data' and event[-1] == v.frame_message('<SMR>')
               for event in cable.events)
    cable.recipient()
    cable.beacon()
    cable.pump(until=lambda: bool(cable.a.session.mail_collection._load()))
    notice, = [event[1] for event in cable.events if event[0] == 'ui']
    assert (notice.command, notice.source, notice.destination, notice.info) == (
        1, 'W9SSJ', 'W9SSJ-1', b'<SMP>')
    assert notice.encode() == encode_ui('W9SSJ', 'W9SSJ-1', b'<SMP>', command=1)
    cable.collect()
    received = cable.a.desk.mailbox.read(1)
    assert (received['from'], received['to'], received['via']) == ('K1ABC', 'W9SSJ-1', 'W9SSJ')
    assert received['body'] == 'café\r\n73'
    assert received['composed_utc'] == '2026-10-28 12:00:00'
    record, = cable.b.session.parking.store.records
    assert record['state'] == 'delivered' and record['attempts'] == 1
    assert json.loads(cable.b.session.parking.store.path.read_text())['records'] == [record]
    forwarded = [event for event in cable.events if event[0] == 'data'
                 and event[1] == 'W9SSJ' and b'<SM>' in event[-1]]
    assert forwarded and all(event[3] not in v.CALLING_HZ.values() for event in forwarded)
    assert any(event[0] == 'data' and b'<TL>' in event[-1] for event in cable.events)
    assert any(event[0] == 'data' and b'<QSYR>' in event[-1] for event in cable.events)


def test_recipient_call_waits_until_notice_restores_listening(cable):
    cable.deposit(DEPOSIT)
    cable.recipient()
    cable.beacon()
    cable.pump(until=lambda: bool(cable.a.session.mail_collection._load()))
    assert cable.b.session.broadcast.active and not cable.b.server.listening
    cable.a.session.command('/vmail collect W9SSJ')
    assert not cable.linked and cable.calling
    cable.pump(until=lambda: cable.linked)
    assert cable.b.server.listening and not cable.b.session.broadcast.active


def test_offline_local_mail_uses_same_two_endpoint_delivery(cable):
    before = len(cable.events)
    cable.b.session.command('/vmail park W9SSJ-1 Local note')
    cable.b.session.command('Locally queued without a link')
    cable.b.session.command('.')
    assert len(cable.events) == before and not cable.b.waves
    row, = cable.b.session.parking.store.records
    assert row['origin'] == 'local' and row['composed_utc']
    cable.a.rename('W9SSJ-1')
    cable.collect()
    received = cable.a.desk.mailbox.read(1)
    assert received['from'] == 'W9SSJ' and received['body'] == 'Locally queued without a link'


@pytest.mark.parametrize('origin', ['remote', 'local'])
@pytest.mark.parametrize('restart', [False, True], ids=['reconnect', 'process-restart'])
def test_lost_smr_reconnect_repeats_forwarding_but_not_inbox(cable, origin, restart):
    if origin == 'remote':
        cable.deposit(DEPOSIT)
        cable.recipient()
    else:
        for line in ('/vmail park W9SSJ-1 Local original', 'Local original body', '.'):
            cable.b.session.command(line)
        cable.a.rename('W9SSJ-1')
    cable.drop_smr = True
    cable.a.session.command('/vmail collect W9SSJ')
    cable.pump(until=lambda: len(cable.a.desk.mailbox.entries()) == 1)
    cable.idle()
    assert cable.b.session.parking.store.records[0]['state'] == 'in_flight'
    cable.disconnect()
    if restart:
        # Stop before the old application handles DISCONNECTED and saves uncertainty.
        assert json.loads(cable.b.session.parking.store.path.read_text())['records'][0]['state'] == 'in_flight'
        cable.restart()
    else:
        cable.idle()
    assert cable.b.session.parking.store.records[0]['state'] == 'uncertain'
    cable.drop_smr = False
    cable.collect()
    assert len(cable.a.desk.mailbox.entries()) == 1
    assert cable.b.session.parking.store.records[0]['attempts'] == 2


@pytest.mark.parametrize('refusal', ['off', 'quota', 'content', 'disk'])
def test_simulated_deposit_refusal_never_claims_custody(cable, monkeypatch, refusal):
    store = cable.b.session.parking.store
    text = DEPOSIT
    if refusal == 'off':
        store.configure(parking=False)
    elif refusal == 'quota':
        store.configure(per_destination=1)
        store.accept(owner='W9SSJ', sender='K1ABC', recipient='W9SSJ-1', via='K1ABC',
                     subject='existing', body='mail', composed='2026-10-28 11:00:00')
    elif refusal == 'content':
        text = DEPOSIT.replace('café\r\n73', 'x' * 501)
    else:
        def full(_):
            raise OSError('simulated full disk')
        monkeypatch.setattr(store, '_commit', full)
    cable.deposit(text)
    replies = [event[-1] for event in cable.events if event[0] == 'data' and event[1] == 'W9SSJ']
    expected = 'SMFP' if refusal == 'off' else 'SMF'
    assert v.frame_message('<' + expected + '>') in replies
    assert v.frame_message('<SMR>') not in replies
    assert len(store.records) == int(refusal == 'quota')


def test_disabled_notices_keep_custody_and_allow_connected_collection(cable):
    cable.deposit(DEPOSIT)
    cable.recipient()
    cable.b.session.parking.store.configure(relay=False)
    cable.beacon()
    assert not any(event[0] == 'ui' for event in cable.events)
    assert cable.b.session.parking.store.records[0]['state'] == 'queued'
    cable.collect()
    assert len(cable.a.desk.mailbox.entries()) == 1


def test_bbs_enabled_relay_delivers_before_optional_lobby_entry(cable, tmp_path):
    from hfmodem.bbs.config import BBSConfig
    from hfmodem.bbs.core import BBSCore
    from hfmodem.bbs.frontends.varac import VarACEntrance
    from hfmodem.bbs.store import Store
    from hfmodem.tests.bbs.test_core import settle
    config = BBSConfig(state_root=tmp_path / 'bbs', title='Mail relay BBS')
    store = Store(config.state_root)
    clock = lambda: cable.b.desk.host.now
    core = BBSCore(config, store, games={}, clock=clock,
                   utcnow=lambda: datetime(2026, 10, 28, tzinfo=timezone.utc) +
                   timedelta(seconds=clock()))
    cable.b.session.bbs = VarACEntrance(cable.b.session, core)
    try:
        cable.deposit(DEPOSIT)
        cable.recipient()
        cable.collect()
        assert core.session is None
        cable.a.desk.host.client.send_text('BBS')
        cable.pump(until=lambda: cable.b.session.bbs.active)
        assert cable.b.session.parking.store.records[0]['state'] == 'delivered'
        assert len(cable.a.desk.mailbox.entries()) == 1
    finally:
        cable.b.session.bbs.ended()
        core.shutdown()
        settle(core)
        core.shutdown()
        store.close()
