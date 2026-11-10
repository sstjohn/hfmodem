# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Controller-established ALE peers select shared audio before host queues."""
import pytest
import threading

from hfmodem.bateleur.ale import words as W
from hfmodem.bateleur.ale.controller import Config, Controller
from hfmodem.bateleur.ale.fsk import Received
from hfmodem.bateleur.ale.modem import AleModem
from hfmodem.bateleur.ale.host import AleHost
from hfmodem.tests.station.recording_live_support import LOCAL, PEER, OTHER
from hfmodem.tests.station.recording_support import audio_files, samples


def controller(owner, **kwargs):
    cfg = Config([LOCAL], ['C0', 'C1'], scanning=False, **kwargs)
    return owner.bind(Controller(cfg, tune=lambda channel: True, send=lambda words: .01))


def heard(ctl, peer=PEER, *, to=None, terminator=W.TIS, now=65):
    frame = W.Frame(to=[LOCAL] if to is None else to, caller=peer, terminator=terminator)
    ctl.heard(Received(frame, [], 0, 0, 0.0, 20.0, 0.0), now)


@pytest.mark.parametrize('kind', ['individual-out', 'individual-in', 'group', 'net', 'anycall',
                                  'allcall-in', 'allcall-out'])
def test_every_linked_insertion_precedes_emit_including_tis_allcall(live, kind):
    owner = live()
    owner.advance(65)
    ctl = controller(owner)
    delivered = []
    emit = ctl._emit

    def publish(event, **kwargs):
        if event == 'LINKED':
            delivered.append(len(owner.starts))
        emit(event, **kwargs)

    ctl._emit = publish
    if kind == 'allcall-out':
        assert ctl.allcall(65)
        peers = {W.ALLCALL}
        assert not [event for event in ctl.events if event.kind == 'LINKED']
    elif kind == 'allcall-in':
        heard(ctl, to=[W.ALLCALL])
        peers = {PEER}
    elif kind == 'individual-out':
        ctl.peer, ctl.state = PEER, ctl.AWAIT_RESPONSE
        heard(ctl)
        peers = {PEER}
    elif kind == 'individual-in':
        ctl.peer, ctl.state = PEER, ctl.AWAIT_ACK
        heard(ctl)
        peers = {PEER}
    else:
        if kind == 'group':
            ctl.call([PEER, OTHER], 65, channel=0)
        elif kind == 'net':
            ctl.call('N0NET', 65, members=[PEER, OTHER], channel=0)
        else:
            ctl.anycall(65, channel=0)
        ctl._responders = [PEER, OTHER, PEER]
        ctl._acknowledge(65)
        peers = {PEER, OTHER}
    rows = owner.rows()
    assert {row['peer'] for row in rows.values()} == peers
    assert len(rows) == len(peers) == len(owner.starts)
    assert all(row['positions'] == {'card': 520, 'dac': 325} for row in rows.values())
    assert all(row['key'][1] == 'ale' for row in rows.values())
    assert all(count > 0 for count in delivered)


@pytest.mark.parametrize('end', ['remote-twas', 'hang-up', 'inactivity'])
def test_per_peer_end_and_timeout_use_card_positions(live, end):
    owner = live()
    owner.advance(65)
    ctl = controller(owner, activity_s=1)
    ctl._responders = [PEER, OTHER]
    ctl._acknowledge(65)
    owner.advance(67)
    if end == 'remote-twas':
        heard(ctl, terminator=W.TWAS, now=67)
        rows = owner.rows()
        assert len(rows) == 2
        assert next(r for r in rows.values() if r['peer'] == OTHER).get('end_positions') is None
        heard(ctl, peer=OTHER, terminator=W.TWAS, now=67)
    elif end == 'hang-up':
        ctl.hang_up(67)
        ctl.hang_up(67)
        assert all(event.t == 67 for event in ctl.events if event.kind == 'LINK_ENDED')
    else:
        ctl.tick(67)
    rows = owner.rows()
    assert len(rows) == 2, 'both established peer generations must be retained'
    assert len(owner.ends) == 2
    assert all(row['end_positions'] == {'card': 536, 'dac': 335} for row in rows.values())


@pytest.mark.parametrize('policy', ['off', 'contacts', 'everything'])
def test_tis_allcall_exact_padding_and_continuous_owner_lanes(live, policy):
    owner = live(policy)
    owner.advance(65)
    ctl = controller(owner)
    assert ctl.allcall(65)
    owner.advance(67)
    ctl.hang_up(67)
    owner.advance(72)
    rows = owner.finish()
    assert len(rows) == 1, 'TIS broadcast address is an established ALE endpoint'
    row = next(iter(rows.values()))
    assert row['peer'] == W.ALLCALL
    assert row['positions'] == {'card': 520, 'dac': 325}
    assert row['end_positions'] == {'card': 536, 'dac': 335}
    if policy == 'off':
        assert not audio_files(owner.root)
        assert not row['lanes']
    else:
        for name, bounds in [('rx', [40, 576]), ('submitted-tx', [25, 360])]:
            ref = row['lanes'][name]
            assert ref['requested'] == ref['actual'] == bounds
            assert not ref['gaps']
            assert ref['kind'] == ('interval' if policy == 'everything' else 'cutout')
            if policy == 'contacts':
                assert len(samples(owner.root, ref)) == bounds[1] - bounds[0]
        if policy == 'everything':
            assert owner.session.index['lanes']['rx']['position'] == 576
            assert owner.session.index['lanes']['submitted-tx']['position'] == 360


def test_heard_twas_sounding_failed_call_are_not_participation(live):
    owner = live()
    ctl = controller(owner)
    heard(ctl, to=[])
    heard(ctl, to=[OTHER])
    heard(ctl, to=[W.ALLCALL], terminator=W.TWAS)
    assert ctl.sound(66)
    ctl.state = ctl.AVAILABLE
    assert ctl.allcall(67, terminator=W.TWAS)
    ctl.call(PEER, 68)
    assert not owner.rows()
    assert not audio_files(owner.root)
    ctl.state = ctl.AVAILABLE
    assert ctl.allcall(69)
    assert owner.one()['peer'] == W.ALLCALL


def test_replaceable_modem_observer_cannot_delay_or_lose_boundary(live, monkeypatch):
    owner = live('off')
    modem = owner.bind(AleModem(Config([LOCAL], ['C0'], scanning=False),
                               send=lambda *a: 0, tune=lambda channel: 0))
    owner.bind(modem.ctl)
    monkeypatch.setattr(modem.ctl, '_send', lambda words: .01)
    first = AleHost(modem)
    modem.ctl.peer, modem.ctl.state = PEER, modem.ctl.AWAIT_RESPONSE
    heard(modem.ctl)
    assert len(owner.starts) == 1, 'source edge must precede the host event drain'
    second = AleHost(modem)
    modem._drain()
    assert not first.log
    assert any(line.startswith('LINKED ') for line in second.log)
    assert modem.observer is second
    modem.stop(timeout=1)
    assert owner.one()['state'] == 'partial'


def test_channel_change_preserves_absolute_card_clock(live):
    owner = live('off')
    owner.advance(65)
    ctl = controller(owner)
    assert ctl.allcall(65)
    owner.advance(67)
    ctl.hang_up(67)
    owner.advance(70)
    ctl.stay(1, 70)
    assert ctl.allcall(70)
    rows = owner.rows()
    assert len(rows) == 2
    assert sorted(row['positions']['card'] for row in rows.values()) == [520, 560]
    assert sorted(row['key'][-1] for row in rows.values())[0] < max(
        row['key'][-1] for row in rows.values())


def test_outgoing_tis_allcall_stamps_before_sent_event_delivery(live):
    owner = live('off')
    ctl = controller(owner)
    delivered = []
    emit = ctl._emit

    def publish(kind, **kwargs):
        if kind == 'SENT':
            delivered.append(len(owner.starts))
        emit(kind, **kwargs)

    ctl._emit = publish
    assert ctl.allcall(65)
    assert delivered == [1]


@pytest.mark.parametrize('path', ['decoder-backlog', 'blocking-send'])
def test_ale_event_card_clock_maps_lanes_behind_capture(live, monkeypatch, path):
    import numpy as np
    from types import SimpleNamespace as NS

    owner = live('off')
    owner.feed(owner.rx, 0, 640)
    owner.tx.row['source_anchor'] = dict(clock='card', sample=400, position=100)
    owner.feed(owner.tx, 100, 125)
    entered, release = threading.Event(), threading.Event()

    def send(audio, rate):
        entered.set()
        assert release.wait(2)
        return 100 * 48000

    modem = AleModem(Config([LOCAL], ['C0'], scanning=False), send=send,
                     recording_boundaries=owner.edges, recording_positions=owner.positions)
    if path == 'decoder-backlog':
        frame = W.Frame(to=[W.ALLCALL], caller=PEER, terminator=W.TIS)
        received = Received(frame, [], 0, 0, 0.0, 20.0, 0.0)
        modem.rx = NS(push=lambda x: [received], samples=0, signal=False, expect=lambda words: None)
        modem._dec = NS(push=lambda x: x)
        modem._audio(65 * 48000 - 1, np.zeros(1, np.float32))
    else:
        errors = []
        def call():
            try:
                assert modem.ctl.allcall(65)
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        assert entered.wait(2)
        owner.feed(owner.rx, 640, 800)
        release.set()
        worker.join(2)
        assert not worker.is_alive() and not errors
    row = owner.one()
    assert row['positions'] == {'card': 520, 'dac': 175}
    owner.feed(owner.rx, owner.rx.position, 880)
    release.set()
    modem.ctl.hang_up(67)
    assert owner.one()['end_positions'] == {'card': 536, 'dac': 185}
    assert owner.tx.position == 125
    modem.stop(timeout=1)


@pytest.mark.parametrize('path', ['decoder-backlog', 'blocking-send'])
def test_station_capture_maps_ale_tx_lane_to_the_event_card_time(tmp_path, path):
    import numpy as np
    from types import SimpleNamespace as NS
    from hfmodem.core.audio import Block
    from hfmodem.core.rates import CARD_RATE_HZ as R
    from hfmodem.core.recording import ResolvedRecording
    from hfmodem.station.capture import StationCapture
    from hfmodem.tests.station.recording_support import bounded, flush

    recorder = StationCapture(tmp_path, 'fake-input', resolved=ResolvedRecording('off', 'run'))
    one = np.zeros(1, np.float32)

    def callback(stop):
        recorder.push(Block(stop / R, stop - 1, one))
        recorder.push_tx(stop - 1, one)

    def rows():
        flush(recorder.session)
        return list(recorder.session.index['contacts'].values())

    callback(100 * R)
    entered, release = threading.Event(), threading.Event()

    def send(audio, rate):
        entered.set()
        assert release.wait(2)
        return 100 * R

    modem = AleModem(Config([LOCAL], ['C0'], scanning=False), send=send,
                     recording_boundaries=recorder.boundaries,
                     recording_positions=recorder.positions)
    try:
        if path == 'decoder-backlog':
            frame = W.Frame(to=[W.ALLCALL], caller=PEER, terminator=W.TIS)
            received = Received(frame, [], 0, 0, 0.0, 20.0, 0.0)
            modem.rx = NS(push=lambda x: [received], samples=0, signal=False,
                          expect=lambda words: None)
            modem._dec = NS(push=lambda x: x)
            modem._audio(65 * R - 1, one)
        else:
            errors = []

            def call():
                try:
                    assert modem.ctl.allcall(65)
                except BaseException as exc:
                    errors.append(exc)
            worker = threading.Thread(target=call, daemon=True)
            worker.start()
            assert entered.wait(2)
            callback(110 * R)
            release.set()
            worker.join(2)
            assert not worker.is_alive() and not errors
        [row] = rows()
        assert row['positions'] == {'card': 65 * R, 'dac': 65 * R}
        callback(120 * R)
        release.set()
        modem.ctl.hang_up(67)
        [row] = rows()
        assert row['end_positions'] == {'card': 67 * R, 'dac': 67 * R}
        modem.stop(timeout=1)
    finally:
        release.set()
        bounded(recorder.close)


def test_relinking_a_still_linked_peer_keeps_its_contact(live):
    owner = live('off')
    owner.advance(65)
    ctl = controller(owner)

    def link(peer, now):
        owner.advance(now)
        ctl.peer, ctl.state = peer, ctl.AWAIT_RESPONSE
        heard(ctl, peer, now=now)

    link(PEER, 65)
    link(OTHER, 66)
    link(PEER, 67)
    rows = owner.rows()
    assert len(rows) == 2, 'a peer that never retired keeps its one contact'
    assert not owner.ends
    assert {row['peer']: row['positions']['card'] for row in rows.values()} == {
        PEER: 520, OTHER: 528}

    owner.advance(68)
    heard(ctl, terminator=W.TWAS, now=68)
    link(PEER, 69)
    rows = owner.rows()
    assert len(rows) == 3, 'a retired peer that links again is a new contact'
    mine = sorted((row['key'][-1], row['positions']['card'], row.get('end_positions'))
                  for row in rows.values() if row['peer'] == PEER)
    assert [card for _, card, _ in mine] == [520, 552]
    assert mine[0][0] < mine[1][0]
    assert mine[0][2] == {'card': 544, 'dac': 340}
    assert mine[1][2] is None
