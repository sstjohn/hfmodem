# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""5066 physical incarnations select audio before independent SIS delivery."""
from types import SimpleNamespace as NS

import pytest

from hfmodem.bateleur.s5066 import cas as C
from hfmodem.bateleur.s5066.node import Node, Timing
from hfmodem.bateleur.s5066.server import NodeService
from hfmodem.tests.station.recording_live_support import Clock
from hfmodem.tests.station.recording_support import audio_files

LOCAL, PEER, OTHER = 1, 2, 3


def node(owner, local=LOCAL):
    result = owner.bind(Node(local, Timing(1200, .01, .01)))
    owner.bind(result.cas)
    return result


def receive(cas, remote, kind):
    cas.cpdu_received(remote, cas.address, bytes([kind << 4]), expedited=True, arq=False)


def establish(cas, *, incoming=False, remote=PEER):
    if incoming:
        receive(cas, remote, C.LINK_REQUEST)
    else:
        cas.make(remote)
        receive(cas, remote, C.LINK_ACCEPTED)


@pytest.mark.parametrize('incoming', [False, True])
def test_physical_establishment_without_sis_client_or_data(live, incoming):
    owner = live()
    owner.advance(65)
    n = node(owner)
    deliveries = []
    n.cas.upper = NS(link_made=lambda *a: deliveries.append((a, len(owner.starts))))
    establish(n.cas, incoming=incoming)
    n.cas.make(PEER)
    row = owner.one()
    assert row['key'][1] == 's5066'
    assert row['key'][3] == LOCAL and row['key'][4] == PEER
    assert row['positions'] == {'card': 520, 'dac': 325}
    assert len(owner.starts) == 1
    assert len(deliveries) == 2 and all(count == 1 for _, count in deliveries)


@pytest.mark.parametrize('end', ['remote-break', 'confirmed-break', 'first-data-timeout',
                                 'break-timeout', 'dts-loss'])
def test_each_physical_end_seam_ends_one_contact(live, end):
    owner = live()
    owner.advance(65)
    n = node(owner)
    establish(n.cas, incoming=end == 'first-data-timeout')
    owner.advance(67)
    if end == 'remote-break':
        receive(n.cas, PEER, C.LINK_BREAK)
        receive(n.cas, PEER, C.LINK_BREAK)
    elif end == 'confirmed-break':
        n.cas.break_link(PEER)
        receive(n.cas, PEER, C.LINK_BREAK_CONFIRM)
    elif end == 'first-data-timeout':
        n.cas.tick(n.cas.first_data_s + 1)
    elif end == 'break-timeout':
        n.cas.break_link(PEER)
        n.cas.links[PEER].tries = n.cas.tries
        n.cas.links[PEER].deadline = 1
        n.cas.tick(2)
    else:
        n.cas.connection_lost(PEER)
    owner.advance(72)
    row = owner.one()
    assert len(owner.starts) == len(owner.ends) == 1
    assert row['end_positions'] == {'card': 536, 'dac': 335}
    assert row['lanes']['rx']['actual'] == [40, 576]
    assert row['lanes']['submitted-tx']['actual'] == [25, 360]
    if end in ('first-data-timeout', 'break-timeout', 'dts-loss'):
        assert row['state'] == 'partial'


def test_repeated_make_and_in_place_dts_reset_preserve_incarnation(live):
    owner = live('off')
    n = node(owner)
    establish(n.cas, incoming=True)
    original = n.cas.links[PEER]
    original.first_data = True
    receive(n.cas, PEER, C.LINK_REQUEST)
    n.cas.make(PEER)
    assert n.cas.links[PEER] is original
    row = owner.one()
    cid = row['id']
    assert len(owner.starts) == 1
    receive(n.cas, PEER, C.LINK_BREAK)
    establish(n.cas, incoming=True)
    rows = owner.rows()
    assert len(rows) == 2 and cid in rows
    assert rows[cid].get('end_positions') is not None
    assert len(owner.starts) == 2


def test_sis_detach_and_peer_break_preserve_other_physical_links(live):
    owner = live('off')
    n = node(owner)
    attachment = n.attach(lambda primitive: None)
    establish(n.cas, incoming=True)
    establish(n.cas, incoming=True, remote=OTHER)
    attachment.close()
    rows = owner.rows()
    assert len(rows) == 2 and not owner.ends
    receive(n.cas, PEER, C.LINK_BREAK)
    rows = owner.rows()
    assert sum(row.get('end_positions') is None for row in rows.values()) == 1
    assert n.cas.made(OTHER)


@pytest.mark.parametrize('policy', ['off', 'contacts', 'everything'])
def test_service_threads_shared_hook_to_cas_and_closes_at_air_owner_stop(live, policy):
    owner = live(policy)
    clock = Clock(1000)
    modem = NS(max_key_s=30, observer=None, post=lambda fn: fn(), set_mode=lambda mode: None)
    service = NodeService(modem, LOCAL, '110 1200S', clock=clock)
    owner.bind(service.onair)
    owner.bind(service.onair.node)
    owner.bind(service.onair.node.cas)
    owner.advance(65)
    service.onair.node.tick(clock())
    establish(service.onair.node.cas)
    assert len(owner.starts) == 1, 'CAS must stamp before any SIS poll'
    owner.advance(67)
    clock.now = 9000
    assert service.onair.stop(timeout=1)
    row = owner.one()
    assert row['positions'] == {'card': 520, 'dac': 325}
    assert row['end_positions'] == {'card': 536, 'dac': 335}
    assert row['state'] == 'partial'
    owner.finish()
    if policy == 'off':
        assert not audio_files(owner.root)
    elif policy == 'everything':
        assert row['lanes']['rx']['kind'] == 'interval'
    else:
        assert row['lanes']['rx']['missing_post_roll'] == 40


def test_failed_make_is_not_a_contact_and_local_addresses_are_distinct(live):
    owner = live('off')
    a, b = node(owner, LOCAL), node(owner, 4)
    a.cas.make(PEER)
    receive(a.cas, PEER, C.LINK_REJECTED)
    assert not owner.rows()
    establish(a.cas)
    establish(b.cas)
    rows = owner.rows()
    assert len(rows) == 2
    assert {row['key'][3] for row in rows.values()} == {LOCAL, 4}


def test_ale_and_5066_overlap_without_ending_each_other(live):
    from hfmodem.bateleur.ale.controller import Config, Controller

    owner = live('off')
    ctl = owner.bind(Controller(Config(['K9ZZT'], ['C0']),
                                tune=lambda channel: True, send=lambda words: .01))
    n = node(owner)
    ctl.allcall(0)
    establish(n.cas)
    receive(n.cas, PEER, C.LINK_BREAK)
    rows = owner.rows()
    assert len(rows) == 2
    ale = next(row for row in rows.values() if row['key'][1] == 'ale')
    assert ale.get('end_positions') is None
    ctl.hang_up(1)
    assert len(owner.ends) == 2


def test_5066_selects_independent_witness_clock_with_exact_padding(live):
    owner = live()
    witness = owner.session.stream('witness', rate=7, clock='witness-card')
    witness.row['source_anchor'] = {'clock': 'card', 'sample': 0, 'position': 0,
                                  'uncertainty_samples': 1}
    owner.advance(65)
    owner.feed(witness, 0, 455)
    n = node(owner)
    establish(n.cas)
    owner.advance(67)
    owner.feed(witness, 455, 469)
    receive(n.cas, PEER, C.LINK_BREAK)
    owner.advance(72)
    owner.feed(witness, 469, 504)
    row = owner.one()
    assert row['positions'] == {'card': 520, 'dac': 325, 'witness-card': 455}
    assert row['end_positions'] == {'card': 536, 'dac': 335, 'witness-card': 469}
    assert row['lanes']['witness']['requested'] == row['lanes']['witness']['actual'] == [35, 504]
    assert not row['lanes']['witness']['gaps']


def test_recording_failure_does_not_replace_or_interrupt_sis_upper(live, monkeypatch):
    reports, events = [], []
    owner = live('off', report=reports.append)
    n = node(owner)
    upper = NS(link_made=lambda *a: events.append('made'), link_broken=lambda *a: events.append('broken'))
    n.cas.upper = upper

    def fail(*args, **kwargs):
        raise OSError('synthetic recording unavailable')

    monkeypatch.setattr(owner.session, 'begin', fail)
    monkeypatch.setattr(owner.session, 'end', fail)
    establish(n.cas)
    n.cas.connection_lost(PEER)
    assert events == ['made', 'broken'] and n.cas.upper is upper
    owner.rows()
    assert any('synthetic recording unavailable' in str(report) for report in reports)


def test_card_gaps_and_native_tx_clock_are_not_compressed_at_cas_edges(live):
    owner = live()
    owner.feed(owner.rx, 100, 180)
    owner.feed(owner.tx, 50, 100)
    n = node(owner)
    establish(n.cas)
    owner.feed(owner.rx, 200, 240)
    owner.feed(owner.tx, 100, 125)
    receive(n.cas, PEER, C.LINK_BREAK)
    owner.feed(owner.rx, 240, 280)
    owner.feed(owner.tx, 125, 150)
    row = owner.one()
    assert row['positions'] == {'card': 180, 'dac': 100}
    assert row['end_positions'] == {'card': 240, 'dac': 125}
    assert row['lanes']['rx']['actual'] == [100, 280]
    assert [180, 200] in row['lanes']['rx']['gaps']
    assert row['lanes']['submitted-tx']['actual'] == [50, 150]


@pytest.mark.parametrize('incoming', [False, True])
def test_replacing_breaking_physical_link_retires_old_contact(live, incoming):
    owner = live('off')
    n = node(owner)
    establish(n.cas)
    first = owner.one()['id']
    n.cas.break_link(PEER)
    establish(n.cas, incoming=incoming)
    rows = owner.rows()
    assert len(rows) == 2
    assert rows[first]['state'] == 'partial'
    assert rows[first].get('end_positions') is not None
    assert next(row for cid, row in rows.items() if cid != first).get('end_positions') is None
