# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Station and host construction share one boundary owner and one publisher."""
from contextlib import nullcontext
import json
from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest

from hfmodem.core import config
from hfmodem.station import capture, hosts, link, process
from hfmodem.station.contactlog import ContactLog
from hfmodem.core.recording import ContactBoundaries, ResolvedRecording
from hfmodem.tests.station.recording_live_support import LOCAL, PEER, Clock
from hfmodem.tests.station.recording_support import bounded


def cfg(protocols=None):
    return config.StationConfig(
        station=config.Station(mycall=LOCAL, transmit=False, id_mode='voice',
                               control='local', regulatory='unregulated',
                               regulatory_settings={'because': 'synthetic transport'}),
        rig=config.Rig(cat='none'), audio=config.Audio(input='fake-input', output='fake-output'),
        protocols=protocols or {})


def test_station_installs_shared_boundary_before_capture_subscription(monkeypatch, tmp_path):
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(process.devices, 'find_device', lambda name, *a, **kw: name)
    observed = []

    class Audio:
        def __init__(self, **kwargs):
            pass

        def subscribe(self, recorder):
            observed.append(getattr(recorder, 'boundaries', None))

        def open(self):
            pass

    monkeypatch.setattr(process, 'StationAudio', Audio)
    station = process.Station(cfg(), recording_selection=ResolvedRecording('off', 'run'))
    station.open_audio()
    try:
        assert len(observed) == 1
        assert isinstance(observed[0], ContactBoundaries), 'install before samples can arrive'
        assert observed[0].session is station.capture.session
        assert station.recording_boundaries is observed[0]
        assert station.capture.tx.clock != station.capture.rx.clock
        assert station.capture.session.publish == station.capture.contact_log.publish
    finally:
        bounded(station.capture.close)


@pytest.mark.parametrize('protocol,host', [
    ('vara', 'vara'), ('ardop', 'ardop'), ('pactor', 'ptc'), ('ale', 'ale'),
    ('rpr', 'kiss'), ('vara', 'vara-managed'), ('ardop', 'ardop-managed'),
    ('pactor', 'ptc-managed'),
])
def test_station_link_construction_inherits_owner_hook_without_replacing_observers(
        live, monkeypatch, protocol, host):
    owner = live('off')
    p = config.ProtocolConfig(enabled=True, host=host, addresses=(LOCAL,), channels=(7100000,))
    station = NS(cfg=cfg({protocol: p}), capture=owner.capture, lanes={},
                 recording_boundaries=owner.edges, recording_positions=owner.positions,
                 audio=NS(samples=0, sample_now=lambda: 0, flush_tx=lambda: 0),
                 rig=NS(max_key_s=30, tune=lambda dial: None),
                 arbiter=NS(identity=NS(mode='cw'), receive_boundary=(0, 0),
                            try_hold=lambda: nullcontext(True), scan_held=False))
    if protocol == 'rpr':
        from hfmodem.brubru import rx, tx, waveform
        monkeypatch.setattr(rx, 'Receiver', lambda: NS())
        monkeypatch.setattr(waveform, 'Decimator', lambda: NS())
        monkeypatch.setattr(tx, 'onset_s', lambda *a: 0)
    lane = link.LINKS[protocol](station)
    station.lanes[protocol] = lane
    if protocol == 'vara':
        targets = [lane.runtime, lane.runtime.transport]
    elif protocol == 'ardop':
        targets = [lane.modem, lane.modem.core]
    elif protocol == 'pactor':
        targets = [lane.runtime]
    elif protocol == 'ale':
        targets = [lane.modem, lane.modem.ctl]
    else:
        targets = [lane]
    assert all(getattr(target, 'recording_boundaries', None) is owner.edges for target in targets)
    assert not owner.rows(), 'construction, lane activation and admission are not contacts'


@pytest.mark.parametrize('host', ['kiss', 'tracker', 'tracker-managed', 'agwpe', 'mail'])
def test_rpr_host_branches_inherit_shared_boundary_before_start(live, monkeypatch, host, tmp_path):
    from hfmodem.brubru.host import connected_server, managed_tracker, server

    owner = live('off')
    secret = tmp_path / 'broker-token'
    secret.write_text('11' * 32)
    p = config.ProtocolConfig(enabled=True, host=host, auth_token_file=str(secret))
    station = NS(cfg=cfg({'rpr': p}), capture=owner.capture, rig=None, arbiter=None,
                 recording_boundaries=owner.edges, recording_positions=owner.positions)
    packet = NS(station=station)
    station.lanes = {'rpr': packet}
    captured = []

    def facade(runtime, *args, **kwargs):
        captured.append(runtime)
        return NS(runtime=runtime)

    monkeypatch.setattr(connected_server, 'TrackerServer', facade)
    monkeypatch.setattr(connected_server, 'AgwServer', facade)
    monkeypatch.setattr(managed_tracker, 'tracker_broker', facade)
    monkeypatch.setattr(server, 'KissServer', facade)
    served = hosts._build_one('rpr', p, station)
    runtime = served.server if host == 'mail' else captured[0]
    assert getattr(runtime, 'recording_boundaries', None) is owner.edges
    if host != 'kiss':
        assert getattr(runtime.engine, 'recording_boundaries', None) is owner.edges
    assert not owner.rows()


def test_station_5066_service_threads_owner_hook_through_all_layers(live):
    owner = live('off')
    p = config.ProtocolConfig(enabled=True, host='s5066', addresses=('0.0.0.1',), mode='110 1200S')
    modem = NS(max_key_s=30, post=lambda fn: fn(), set_mode=lambda mode: None, observer=None)
    station = NS(lanes={'bateleur': NS(modem=modem)}, arbiter=None,
                 capture=owner.capture, recording_boundaries=owner.edges,
                 recording_positions=owner.positions)
    service = hosts._build_one('bateleur', p, station).server
    assert all(getattr(target, 'recording_boundaries', None) is owner.edges for target in
               (service, service.onair, service.onair.node, service.onair.node.cas))
    assert not owner.rows()


def test_live_boundary_result_reaches_existing_contactlog_publisher(live, tmp_path):
    from hfmodem.bateleur.s5066.node import Node, Timing
    from hfmodem.bateleur.s5066 import cas as C

    owner = live('off')
    journal = ContactLog(tmp_path / 'sessions' / 'live-run', protocol='s5066',
                         application='SIS', recording=owner.session)
    publisher = owner.session.publish
    n = owner.bind(Node(1, Timing(1200, .01, .01)))
    owner.bind(n.cas)
    n.cas.cpdu_received(2, 1, bytes([C.LINK_REQUEST << 4]), expedited=True, arq=False)
    n.cas.connection_lost(2)
    owner.finish()
    assert owner.session.publish == publisher == journal.publish
    assert len(journal.records) == 1, 'source contacts must reach the stage 2 ContactLog'
    paths = list((tmp_path / 'arms').glob('*/provenance.json'))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record['recording']['policy'] == 'off'
    assert record['recording']['state'] == 'partial'
    assert record['recording']['key'][1] == 's5066'
    assert not list(tmp_path.rglob('*.wav'))


def test_capture_close_retires_all_active_protocols_once(monkeypatch, tmp_path):
    recorder = capture.StationCapture(tmp_path, 'fake-input',
                                     resolved=ResolvedRecording('off', 'run'))
    edges = getattr(recorder, 'boundaries', None)
    try:
        assert isinstance(edges, ContactBoundaries), 'capture owns the shared boundary lifetime'
        run = edges.run
        for protocol in ('ale', 'vara', 's5066'):
            edges.established((run, protocol, 'fake-endpoint', LOCAL, PEER, 1), PEER,
                              {'card': 0, 'dac': 0})
        bounded(recorder.close)
        bounded(recorder.close)
        rows = recorder.session.index['contacts']
        assert len(rows) == 3
        assert all(row['state'] == 'partial' and row['end_positions'] == {'card': 0, 'dac': 0}
                   for row in rows.values())
    finally:
        bounded(recorder.close)


def test_source_edges_do_not_wait_for_blocked_publication(live):
    from hfmodem.bateleur.s5066.node import Node, Timing
    from hfmodem.bateleur.s5066 import cas as C

    owner = live()
    owner.advance(65)
    n = owner.bind(Node(1, Timing(1200, .01, .01)))
    owner.bind(n.cas)
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5), 'synthetic publication was not released'

    owner.session.after_pending(blocked)
    assert entered.wait(2), 'storage barrier did not enter'
    try:
        bounded(lambda: n.cas.cpdu_received(2, 1, bytes([C.LINK_REQUEST << 4]),
                                            expedited=True, arq=False))
        assert len(owner.starts) == 1, 'source stamping must not wait for storage publication'
        assert owner.starts[0][1] == {'card': 520, 'dac': 325}
    finally:
        release.set()
    assert owner.one()['positions'] == {'card': 520, 'dac': 325}


def test_raw_bateleur_and_selcall_activity_do_not_establish_contacts(live, monkeypatch):
    from hfmodem.bateleur.host import BateleurHost
    from hfmodem.bateleur.modem import Mode
    from hfmodem.bateleur.selcall import fsk
    from hfmodem.bateleur.selcall.modem import SelcallModem, Event
    from hfmodem.bateleur.selcall.host import SelcallHost
    from hfmodem.bateleur.s5066.node import Node, Timing
    from hfmodem.bateleur.s5066 import cas as C

    owner = live('off')
    raw_modem = owner.bind(NS(observer=None))
    raw_host = BateleurHost(raw_modem)
    raw_host.rx_started('110', '1200S', 0)
    raw_host.rx_data(b'unconnected bytes')
    raw_host.rx_ended(True, 20, '')
    raw_host.tx_started(Mode('110', '1200S'), 4, .01)
    raw_host.tx_ended(True, '')
    monkeypatch.setattr(fsk, 'call_audio', lambda *a, **kw: np.zeros(1, np.float32))
    selcall = owner.bind(SelcallModem(['1234'], send=lambda *a: 1, acknowledge=True))
    host = SelcallHost(selcall)
    call = selcall.call('selcall', '5678')
    selcall._emit(Event('HEARD', call))
    selcall._emit(Event('CALLED', call))
    assert host.log
    assert not owner.rows(), 'bursts, selcall acknowledgment and host attachment are not links'
    n = owner.bind(Node(1, Timing(1200, .01, .01)))
    owner.bind(n.cas)
    n.cas.cpdu_received(2, 1, bytes([C.LINK_REQUEST << 4]), expedited=True, arq=False)
    assert owner.one()['key'][1] == 's5066'


def test_disabled_sabir_factory_propagates_hook_without_enabling_station(live, monkeypatch, tmp_path):
    from hfmodem.sabir.host import hostlink

    owner = live('off')
    p = config.ProtocolConfig(enabled=True, host='hostlink')
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path))
    station = process.Station(cfg({'sabir': p}))
    assert 'sabir' not in station.enabled
    station.capture = owner.capture
    station.recording_boundaries, station.recording_positions = owner.edges, owner.positions
    station.lanes['sabir'] = NS(air=NS(clock=Clock(), register=lambda modem: None))
    factories = []

    def server(factory, **kwargs):
        factories.append(factory)
        return NS(port=0)

    monkeypatch.setattr(hostlink, 'HostLinkServer', server)
    hosts._build_one('sabir', p, station)
    modem = factories[0]()
    assert getattr(modem, 'recording_boundaries', None) is owner.edges
    assert getattr(modem.fsm, 'recording_boundaries', None) is owner.edges
    assert 'sabir' not in station.enabled


def test_replay_with_capture_destination_does_not_prove_ale_participation(monkeypatch, tmp_path):
    from dataclasses import replace
    from hfmodem.bateleur.ale import words as W
    from hfmodem.bateleur.ale.fsk import Received

    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path))
    p = config.ProtocolConfig(enabled=True, host='ale', addresses=(LOCAL,), channels=(7100000,))
    configured = cfg({'ale': p})
    configured = replace(configured, audio=replace(configured.audio, capture_dir=str(tmp_path / 'captures')))
    station = process.Station(configured, replay=np.zeros(48000, np.float32),
                              recording_selection=ResolvedRecording('contacts', 'run'))
    station.open_audio()
    try:
        lane = link.AleLink(station)
        frame = W.Frame(to=[W.ALLCALL], caller=PEER, terminator=W.TIS)
        lane.modem.ctl.heard(Received(frame, [], 0, 0, 0., 20., 0.), 1)
        assert PEER in lane.modem.ctl.linked
        station.audio.pump()
        bounded(station.capture.close)
        assert not station.capture.session.index['contacts']
        assert not list(tmp_path.rglob('*.wav'))
        assert getattr(lane.modem, 'recording_boundaries', None) is None
    finally:
        bounded(station.capture.close)
