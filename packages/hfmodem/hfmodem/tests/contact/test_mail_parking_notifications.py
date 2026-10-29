# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from datetime import datetime
import time

import pytest

from hfmodem.ax25 import kiss
from hfmodem.contact.broadcast import decode_ui
from hfmodem.contact.mail_parking import MailParking
from hfmodem.contact.varac import BEACON_CLEAR_S, BEACON_KISS, State
from hfmodem.tests.contact.test_broadcast_client import answer, wire
from hfmodem.tests.contact.test_varac import Desk


def ready(tmp_path=None, *, wall=None, caps=True):
    d = Desk(mailbox=tmp_path, kiss=True,
             **{'BROADCAST CAPS': ['BROADCAST CAPS 0 1'] if caps else []})
    d.wall = [time.time() if wall is None else wall]
    d.session.parking.store.close()
    d.session.parking = MailParking(d.session, wall_clock=lambda: d.wall[0] + d.host.now)
    d.session.band_ready(True)
    d.hear('BUSY OFF')
    return d


def deposit(d, recipient='K1ABC', *, owner='W9SSJ', body='hello'):
    return d.session.parking.store.accept(owner=owner, sender=owner, recipient=recipient,
                                         via=owner, subject='waiting', body=body, origin='local')[0]


def beacon(d, call='K1ABC', *, dial=None):
    d.session.parking.heard(call, d.flow.dial if dial is None else dial)
    d.tick()
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    d.tick()


def notice(d, recipient='K1ABC'):
    return d.session.parking.store.notice(d.host.client.mycall, recipient)


def test_beacon_callback_only_enqueues_then_tick_sends_exact_destination():
    d = ready()
    deposit(d, 'K1ABC-1')
    before = list(d.sent), list(d.texts), list(d.tunes)
    d.session.parking.heard('K1ABC', d.flow.dial)
    assert (d.sent, d.texts, d.tunes) == before
    d.tick()
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    d.tick()
    decoded = decode_ui(bytes.fromhex(wire(d)[0].split()[3]), command=1)
    assert (decoded.source, decoded.destination, decoded.info) == ('W9SSJ', 'K1ABC-1', b'<SMP>')
    assert notice(d, 'K1ABC-1')['last_sent'] is None
    answer(d, 'QUEUED', 'SENT')
    assert notice(d, 'K1ABC-1')['last_sent'] is not None
    assert d.session.parking.store.records[0]['state'] == 'queued'
    assert not d.texts and d.tunes == before[2]


@pytest.mark.parametrize('blocked', ['busy', 'unknown-busy', 'high', 'unknown-line', 'linked',
                                     'pending', 'ptt', 'contact', 'steps', 'closing', 'quitting',
                                     'band-fault', 'unready', 'hopping-cooldown', 'follow',
                                     'broadcast-hold', 'buffer', 'writer', 'not-monitoring'])
def test_notice_scheduling_defers_every_unsafe_state_without_reserving(blocked):
    d = ready()
    deposit(d)
    s, c = d.session, d.host.client
    if blocked in ('busy', 'unknown-busy'):
        c.busy = True if blocked == 'busy' else None
    elif blocked in ('high', 'unknown-line'):
        d.line = True if blocked == 'high' else None
    elif blocked in ('linked', 'pending', 'ptt'):
        setattr(c, 'connected' if blocked == 'linked' else blocked, True)
    elif blocked == 'contact':
        s.contact = {'peer': 'K2ABC'}
    elif blocked == 'steps':
        s.steps = iter([None] * 100)
    elif blocked in ('closing', 'quitting', 'band-fault'):
        setattr(s, '_' + blocked.replace('-', '_'), True)
    elif blocked == 'unready':
        s.hopping.readiness.clear()
    elif blocked == 'hopping-cooldown':
        s.hopping.last_tx[d.flow.band] = d.host.now
    elif blocked == 'follow':
        s._follow_tuned = True
    elif blocked == 'broadcast-hold':
        s.broadcast.selected = d.flow.dial
    elif blocked == 'buffer':
        c.buffer = 10
    elif blocked == 'writer':
        c._writes[1] = (c.link_generation, b'pending')
    else:
        d.flow.state = State.SNIFFING
    if blocked == 'contact':
        s.parking.heard('K1ABC', d.flow.dial)
        s.parking.tick()
        d.host.now += BEACON_CLEAR_S + 1
        s.parking.tick()
    else:
        beacon(d)
    assert not wire(d) and not notice(d)
    assert not d.texts


def test_off_channel_beacon_never_retunes_or_creates_a_notice():
    d = ready()
    deposit(d)
    before = list(d.tunes)
    beacon(d, dial=d.flow.dial + 750)
    assert not wire(d) and not notice(d) and d.tunes == before


def test_base_beacon_wakes_exact_ssid_but_not_another_owners_queue():
    d = ready()
    deposit(d, 'W9SSJ-1')
    deposit(d, 'W9SSJ', owner='K2ABC')
    beacon(d, 'W9SSJ')
    messages = [decode_ui(bytes.fromhex(line.split()[3]), command=1) for line in wire(d)]
    assert [message.destination for message in messages] == ['W9SSJ-1']
    answer(d, 'QUEUED', 'SENT')
    assert not notice(d, 'W9SSJ')


@pytest.mark.parametrize('recipient', ['ALL', 'CQ', 'W9SSJ', 'W9SSJ-0', 'w9ssj-00'])
def test_group_and_canonical_self_addresses_cannot_create_notices(recipient):
    d = ready()
    with pytest.raises(ValueError):
        deposit(d, recipient)
    d.session.command('/vmail notify ' + recipient)
    d.tick()
    assert not wire(d) and not d.texts


def test_busy_candidate_expires_and_requires_a_fresh_beacon():
    d = ready()
    deposit(d)
    d.host.client.busy = True
    beacon(d)
    d.host.now += 301
    d.tick()
    d.hear('BUSY OFF')
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    assert not wire(d) and not notice(d)
    beacon(d)
    assert len(wire(d)) == 1


@pytest.mark.parametrize('transition', ['relay-off', 'queue-empty', 'link', 'tune'])
def test_pre_admission_transition_cancels_without_cooldown(transition):
    d = ready(caps=False)
    row = deposit(d)
    beacon(d)
    assert d.session.broadcast.active and not wire(d)
    if transition == 'relay-off':
        d.session.command('/vmail relay off')
    elif transition == 'queue-empty':
        d.session.command(f'/vmail cancel {row["id"]}')
    elif transition == 'link':
        d.hear('CONNECTED K2ABC W9SSJ 500')
    else:
        d.flow.dial += 750
    d.hear('BROADCAST CAPS 0 1')
    assert not wire(d) and not notice(d)
    assert not d.session.broadcast.active


def test_relay_off_blocks_manual_and_automatic_without_deleting_queue():
    d = ready()
    deposit(d)
    d.do('/vmail relay off')
    beacon(d)
    d.do('/vmail notify K1ABC')
    assert not wire(d) and not notice(d)
    assert len(d.session.parking.store.pending('W9SSJ', 'K1ABC')) == 1


def test_manual_notify_obeys_cooldown_and_band_guard():
    d = ready()
    deposit(d)
    d.session.hopping.readiness.clear()
    d.session.command('/vmail notify K1ABC')
    d.tick()
    assert not wire(d)
    d.session.band_ready(True)
    d.session.command('/vmail notify K1ABC')
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    d.tick()
    assert len(wire(d)) == 1
    answer(d, 'QUEUED', 'SENT')
    d.session.command('/vmail notify K1ABC')
    d.tick()
    assert len(wire(d)) == 1


def test_notice_cooldown_survives_band_change_restart_and_clock_rollback(tmp_path):
    d = ready(tmp_path)
    deposit(d)
    beacon(d)
    answer(d, 'QUEUED', 'SENT')
    first = notice(d)
    d.session.change_band(40)
    d.session.band_ready(True)
    d.hear('BUSY OFF')
    beacon(d)
    assert len(wire(d)) == 1 and notice(d) == first
    d.session.parking.store.close()
    restarted = ready(tmp_path, wall=d.wall[0] - 3600)
    beacon(restarted)
    assert not wire(restarted) and notice(restarted) == first
    restarted.host.now += 1801
    beacon(restarted)
    assert not wire(restarted)
    restarted.host.now += 3601
    beacon(restarted)
    assert len(wire(restarted)) == 1


def test_uncertain_admission_reserves_cooldown_without_last_sent():
    d = ready()
    deposit(d)
    beacon(d)
    reserved = notice(d)
    assert reserved['last_sent'] is None
    d.drive()
    assert not d.session.broadcast.active and 'ABORT' in d.sent
    beacon(d)
    assert len(wire(d)) == 1 and notice(d) == reserved


def test_c2_beacon_timeline_has_bounded_cadence():
    d = ready()
    deposit(d, 'W8RDG')
    times = ['14:29:42.805', '14:44:43.336', '15:00:00.685',
             '15:15:01.105', '15:30:01.597', '15:45:01.887']
    seconds = [datetime.fromisoformat('2026-10-28T' + value).timestamp() for value in times]
    attempts = []
    for at in seconds:
        d.host.now = 100 + at - seconds[0]
        before = len(wire(d))
        beacon(d, 'W8RDG')
        if len(wire(d)) > before:
            attempts.append(d.host.now)
            answer(d, 'QUEUED', 'SENT')
    assert len(attempts) == 3
    assert all(b - a >= 1800 for a, b in zip(attempts, attempts[1:]))
    assert all(decode_ui(bytes.fromhex(line.split()[3])).info == b'<SMP>' for line in wire(d))


@pytest.mark.parametrize('state', ['cancelled', 'delivered', 'expired', 'empty'])
def test_no_notice_without_live_pending_mail(state):
    d = ready()
    if state != 'empty':
        row = deposit(d)
        store = d.session.parking.store
        if state == 'cancelled':
            store.cancel(row['id'])
        elif state == 'delivered':
            store.begin(row['id'])
            store.resolve(row['id'], 'SMR')
        else:
            d.wall[0] += 31 * 86400
    beacon(d)
    assert not wire(d) and not notice(d)


def test_busy_pulse_while_awaiting_caps_invalidates_clear_proof():
    d = ready(caps=False)
    deposit(d)
    beacon(d)
    assert d.session.broadcast.active and not wire(d)
    d.hear('BUSY ON', 'BUSY OFF', 'BROADCAST CAPS 0 1')
    assert not wire(d) and not notice(d)


def test_only_valid_beacons_wake_queue_and_known_local_tx_echo_is_ignored():
    d = ready()
    deposit(d, 'W9SSJ-1')
    d.host.client.ptt = True
    d.hear('CQFRAME W9SSJ-9 500')
    d.host.client.ptt = False
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    assert not wire(d)
    deposit(d, 'K1ABC')
    d.hear('CQFRAME K1ABC 500')
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    assert not wire(d)
    d.hear('CQFRAME K1ABC-9 500')
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    assert len(wire(d)) == 1
    assert decode_ui(bytes.fromhex(wire(d)[0].split()[3])).destination == 'K1ABC'


def test_valid_advanced_beacon_from_kiss_wakes_full_ssid_destination():
    d = ready()
    deposit(d, 'K1ABC-1')
    d.host.kiss.incoming.append(kiss.encode(b'20VFN31AAK1ABC', BEACON_KISS))
    d.tick()
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    d.tick()
    assert len(wire(d)) == 1
    assert decode_ui(bytes.fromhex(wire(d)[0].split()[3])).destination == 'K1ABC-1'


@pytest.mark.parametrize('transition', ['link', 'band', 'queue-empty'])
def test_deferred_notice_transition_requires_new_beacon(transition):
    d = ready()
    row = deposit(d)
    d.host.client.busy = True
    beacon(d)
    assert d.session.parking.candidates and not d.session.broadcast.active
    if transition == 'link':
        d.hear('CONNECTED K2ABC W9SSJ 500')
        d.hear('DISCONNECTED')
    elif transition == 'band':
        d.session.change_band(40)
        d.session.band_ready(True)
        d.tick()
        d.session.change_band(20)
    else:
        d.session.parking.store.cancel(row['id'])
        d.tick()
        deposit(d, body='new mail after the old candidate was cancelled')
    d.hear('BUSY OFF')
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    d.tick()
    assert not wire(d) and not notice(d)
    beacon(d)
    assert len(wire(d)) == 1


def test_clear_channel_proof_restarts_on_new_band_even_if_busy_off_arrives_before_tick():
    d = ready()
    deposit(d)
    d.tick()
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    assert d.session.parking.clear_since is not None
    d.session.change_band(40)
    d.session.band_ready(True)
    d.hear('BUSY OFF')
    d.session.parking.heard('K1ABC', d.flow.dial)
    d.tick()
    d.tick()
    assert not wire(d) and not notice(d)
    d.host.now += BEACON_CLEAR_S + 1
    d.tick()
    d.tick()
    assert len(wire(d)) == 1


def test_unsupported_broadcast_capability_does_not_retry_without_a_new_beacon():
    d = ready()
    d.host.replies['BROADCAST CAPS'] = ['WRONG']
    deposit(d)
    beacon(d)
    for _ in range(20):
        d.host.now += 1
        d.tick()
    assert d.sent.count('BROADCAST CAPS') == 1
    assert not wire(d) and not notice(d)
    beacon(d)
    assert d.sent.count('BROADCAST CAPS') == 2


def test_physical_transition_during_reservation_retains_cooldown_without_submission(monkeypatch):
    d = ready()
    deposit(d)
    store = d.session.parking.store
    reserve = store.reserve_notice
    def changed(owner, recipient):
        reserve(owner, recipient)
        d.line = True
    monkeypatch.setattr(store, 'reserve_notice', changed)
    beacon(d)
    assert not wire(d)
    assert notice(d)['last_attempt'] is not None and notice(d)['last_sent'] is None
    assert not store.eligible('W9SSJ', 'K1ABC')
