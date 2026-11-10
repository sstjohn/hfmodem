# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace

import pytest

from hfmodem.contact.beacon_hop import BeaconHop, HISTORY_S
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk, BEACON, CQ_DONE, wait


def hopping_desk(command='/beacon hop 40 20', **replies):
    desk = Desk(**{BEACON: CQ_DONE, **replies})
    desk.session.hopping.readiness.update({b: 'ready' for b in (80, 40, 20)})
    desk.do(command)
    return desk


def opportunity(desk):
    desk.host.now = desk.session.hopping.due
    desk.tick()
    wait(desk, v.BEACON_CLEAR_S + 1)
    desk.drive()


def test_rotation_beacons_on_arrival_resets_monitor_and_keeps_interval():
    desk = hopping_desk('/beacon hop 40 20 every 20')
    moves = []
    desk.session.slots = SimpleNamespace(retune=lambda *a: moves.append(a),
                                         poll=lambda: [], table=lambda: ([], 0))
    desk.session._transcripts = {1: ['old']}
    desk.tick()
    assert desk.flow.band == 40 and BEACON not in desk.sent
    assert moves == [(7_105_000, 7_105_000)] and not desk.session._transcripts
    wait(desk, 6)
    assert [e['dial_hz'] for e in desk.kinds('beacon')] == [7_105_000]
    last = desk.session.hopping.last_tx[40]
    assert desk.session.hopping.due == last + 1200
    opportunity(desk)
    assert [e['dial_hz'] for e in desk.kinds('beacon')] == [7_105_000, 14_105_000]
    opportunity(desk)
    assert desk.flow.band == 40 and len(desk.kinds('beacon')) == 3
    assert [e['band_m'] for e in desk.kinds('band')] == [40, 20, 40]


@pytest.mark.parametrize('suffix', ['', '40 40', '99 20', '40 every 0', '40 every 4',
                                     '40 every', '40 every 15 20', 'auto every -1'])
def test_invalid_configuration_preserves_existing_schedule(suffix):
    desk = hopping_desk()
    before = (desk.session.hopping.bands, desk.session.hopping.due, desk.session._beacon_expires)
    desk.do('/beacon hop ' + suffix)
    assert (desk.session.hopping.bands, desk.session.hopping.due, desk.session._beacon_expires) == before
    assert desk.said[-1].startswith('error:')


def test_band_numbers_are_not_ambiguous_intervals():
    desk = hopping_desk('/beacon hop 40 20 15')
    assert desk.session.hopping.bands == (40, 20, 15)
    desk.do('/beacon hop 40 20 25')
    assert desk.session.hopping.bands == (40, 20) and desk.session.beacon == 25
    desk.do('/beacon hop auto 40 20 every 30')
    assert desk.session.hopping.bands == (40, 20) and desk.session.beacon == 30


@pytest.mark.parametrize('mode', ['', 'auto '])
@pytest.mark.parametrize('minutes', [5, 14])
def test_shorter_dwell_is_allowed_without_shortening_beacon_cooldown(mode, minutes):
    desk = hopping_desk(f'/beacon hop {mode}40 20 every {minutes}')
    hop = desk.session.hopping
    assert hop.minutes == minutes
    hop.last_tx[40] = desk.host.now
    hop.defer()
    assert hop.due == desk.host.now + minutes * 60
    desk.host.now += minutes * 60
    assert hop.cooldown(40) == (15 - minutes) * 60


@pytest.mark.parametrize('mode', ['', 'auto '])
def test_omitted_dwell_still_defaults_to_fifteen_minutes(mode):
    desk = hopping_desk(f'/beacon hop {mode}40 20')
    assert desk.session.hopping.minutes == 15


def test_five_minute_rotation_beacons_on_each_arrival():
    desk = hopping_desk('/beacon hop 40 20 80 every 5')
    wait(desk, 6)
    first = desk.session.hopping.last_tx[40]
    for band in (20, 80, 40):
        assert desk.session.hopping.due == max(desk.session.hopping.last_tx.values()) + 300
        opportunity(desk)
        assert desk.flow.band == band
    assert [event['dial_hz'] for event in desk.kinds('beacon')] == [
        7_105_000, 14_105_000, 3_595_000, 7_105_000]
    assert desk.session.hopping.last_tx[40] - first >= 900


def test_unknown_readiness_pauses_without_moving_and_acknowledgement_resumes():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon hop 40 20')
    desk.tick()
    assert desk.tunes == [] and desk.session.hopping.paused == 'readiness'
    assert BEACON not in desk.sent
    desk.do('/band 40')
    assert desk.armed[-1] is False and desk.sent[-1] == 'LISTEN OFF'
    wait(desk, 10)
    assert BEACON not in desk.sent
    desk.do('/band ready')
    wait(desk, 6)
    assert desk.session.hopping.ready(40) and len(desk.kinds('beacon')) == 1
    assert not any('TUNE' in command or 'PTT' in command for command in desk.sent)


def test_atu_on_and_verified_filters_do_not_imply_readiness():
    desk = Desk()
    desk.session.band_tuner = lambda dial: dict(verified=True, tuner_on=True, atu_match_known=False)
    desk.do('/band 40')
    assert not desk.session.hopping.ready(40)
    desk.do('/band ready')
    assert desk.session.hopping.ready(40)
    desk.do('/band unready')
    assert desk.session.hopping.readiness[40] == 'needs operator TUNE'


@pytest.mark.parametrize('guard', ['connected', 'pending', 'ptt', 'line', 'steps', '_closing', '_quitting',
                                  '_follow_tuned', '_band_fault'])
def test_no_retune_or_beacon_during_contact_keying_close_or_follow(guard):
    desk = hopping_desk()
    if guard in ('connected', 'pending', 'ptt'):
        setattr(desk.host.client, guard, True)
    elif guard == 'line':
        desk.line = True
    else:
        setattr(desk.session, guard, iter([None]) if guard == 'steps' else True)
    desk.host.now += 4000
    desk.session._beacon_tick()
    assert desk.tunes == [] and BEACON not in desk.sent
    assert desk.session.hopping.due > desk.host.now


def test_resume_after_missed_opportunities_has_no_catch_up():
    desk = hopping_desk()
    desk.host.client.pending = True
    desk.host.now += 4000
    desk.session._beacon_tick()
    desk.host.client.pending = False
    wait(desk, 10)
    assert desk.tunes == [] and BEACON not in desk.sent
    opportunity(desk)
    assert desk.sent.count(BEACON) == 1


def test_busy_arrival_waits_and_does_not_sweep_other_bands():
    desk = hopping_desk()
    desk.tick()
    desk.hear('BUSY ON')
    wait(desk, 10)
    assert desk.flow.band == 40 and BEACON not in desk.sent
    assert len(desk.kinds('band')) == 1
    desk.hear('BUSY OFF')
    wait(desk, 6)
    assert desk.sent.count(BEACON) == 1


def test_busy_band_misses_opportunity_without_an_immediate_multi_band_attempt():
    desk = hopping_desk()
    desk.tick()
    desk.hear('BUSY ON')
    due = desk.session.hopping.due
    desk.host.now = due - 1
    desk.session._beacon_tick()
    assert len(desk.kinds('band')) == 1 and BEACON not in desk.sent
    desk.host.now = due
    desk.session._beacon_tick()
    assert [e['band_m'] for e in desk.kinds('band')] == [40, 20]
    assert BEACON not in desk.sent


def test_manual_band_change_queues_arrival_and_respects_last_actual_beacon():
    desk = hopping_desk()
    wait(desk, 6)
    desk.do('/band 80')
    wait(desk, 6)
    assert [e['dial_hz'] for e in desk.kinds('beacon')] == [7_105_000, 3_595_000]
    desk.do('/band 40')
    wait(desk, 6)
    assert len(desk.kinds('beacon')) == 2
    desk.host.now = desk.session.hopping.last_tx[40] + 900
    desk.tick()
    desk.drive()
    assert len(desk.kinds('beacon')) == 3


def test_manual_beacons_record_cooldown_across_off_and_mode_changes():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon now')
    wait(desk, 6)
    last = desk.session.hopping.last_tx[20]
    desk.do('/beacon off')
    desk.do('/band ready')
    desk.do('/beacon hop 20')
    wait(desk, 6)
    assert desk.session.hopping.last_tx[20] == last
    assert desk.sent.count(BEACON) == 1
    desk.do('/beacon force')
    assert not desk.session._beacon_now
    opportunity(desk)
    assert desk.sent.count(BEACON) == 2


@pytest.mark.parametrize('command', ['/beacon off', '/beacon 20'])
def test_cancel_or_single_band_mode_cancels_pending_hop(command):
    desk = hopping_desk()
    desk.tick()
    desk.do(command)
    assert not desk.session.hopping.active
    desk.host.now += 2000
    wait(desk, 6)
    assert desk.flow.band == 40
    if command.endswith('off'):
        assert BEACON not in desk.sent


def test_expiry_and_quit_cancel_hopping():
    desk = hopping_desk()
    desk.host.now = desk.session._beacon_expires
    desk.session._beacon_tick()
    assert not desk.session.hopping.active and not desk.tunes
    desk = hopping_desk()
    desk.session.quit()
    assert not desk.session.hopping.active and BEACON not in desk.sent


def test_failed_retune_disarms_and_pauses_without_retry_loop():
    desk = hopping_desk()
    def fail(dial):
        raise v.HostError('readback failed')
    desk.session.band_tuner = fail
    wait(desk, 10)
    assert desk.armed[-1] is False and desk.session._band_fault
    assert desk.session.hopping.paused.startswith('band transition failed')
    assert len(desk.kinds('error')) == 1 and BEACON not in desk.sent


def test_incoming_setup_during_listen_off_aborts_hop():
    desk = hopping_desk()
    desk.host.replies['LISTEN OFF'] = ['OK', 'PENDING']
    desk.tick()
    assert desk.tunes == [] and desk.armed[-1] is False and BEACON not in desk.sent
    assert desk.session.hopping.paused


@pytest.mark.parametrize('event', ['PENDING', 'BUSY ON', 'physical'])
def test_arrival_beacon_rechecks_eligibility_after_disabling_listen(event):
    desk = hopping_desk()
    desk.tick()  # Complete the retune first; change eligibility at beacon admission.
    def respond():
        if event == 'physical':
            desk.line = True
            return ['OK']
        return ['OK', event]
    desk.host.replies['LISTEN OFF'] = respond
    wait(desk, 7)
    assert BEACON not in desk.sent and not desk.session.hopping.last_tx


@pytest.mark.parametrize('manual', [False, True])
def test_slot_monitor_failure_pauses_the_whole_band_transition(manual):
    desk = hopping_desk()
    def fail(*args):
        raise v.HostError('slot monitor reset failed')
    desk.session.slots = SimpleNamespace(retune=fail, poll=lambda: [], table=lambda: ([], 0))
    if manual:
        desk.do('/band 80')
    else:
        desk.tick()
    assert desk.session._band_fault and desk.armed[-1] is False
    assert desk.session.hopping.pending is None and desk.session.hopping.paused
    wait(desk, 10)
    assert BEACON not in desk.sent


@pytest.mark.parametrize('reply', [['OK', 'PENDING', 'CANCELPENDING'], ['WRONG'],
                                   ['OK', 'BUSY ON', 'BUSY OFF']])
def test_failed_or_unkeyed_beacon_is_not_recorded_as_delivery(reply):
    desk = hopping_desk(**{BEACON: reply})
    wait(desk, 7)
    assert not desk.kinds('beacon') and not desk.session.hopping.last_tx
    assert desk.flow.band == 40 and len(desk.kinds('band')) == 1


def test_off_during_hop_beacon_preserves_actual_completion_without_restarting():
    desk = hopping_desk(**{BEACON: ['OK', 'PENDING', 'PTT ON']})
    for _ in range(100):
        desk.tick()
        if desk.host.client.ptt:
            break
    assert desk.host.client.ptt
    desk.session.command('/beacon off')
    desk.hear('PTT OFF', 'CANCELPENDING')
    desk.drive()
    assert not desk.session.hopping.active and desk.session.hopping.last_tx[40]
    assert desk.session._beacon_due is None


def test_disabling_hops_restores_preexisting_auto_answer_setting():
    desk = Desk()
    desk.do('/beacon hop auto')
    assert not desk.armed[-1] and desk.sent[-1] == 'LISTEN OFF'
    desk.do('/beacon off')
    assert desk.armed[-1] and desk.sent[-1] == 'LISTEN CQ'


def test_contact_after_arrival_cancels_pending_announcement():
    desk = hopping_desk()
    desk.tick()
    desk.host.client.pending = True
    desk.session._beacon_tick()
    assert desk.session.hopping.pending is None
    desk.host.client.pending = False
    wait(desk, 10)
    assert BEACON not in desk.sent


def chooser():
    clock = SimpleNamespace(now=1000, utc=172800 + 23 * 3600)
    hop = BeaconHop((80, 40, 20), lambda: clock.now, lambda: clock.utc)
    hop.readiness = {b: 'ready' for b in hop.channels}
    hop.configure(['auto'])
    return hop, clock


def test_auto_uses_unique_calls_adjacent_utc_hours_and_bounded_history():
    hop, clock = chooser()
    for _ in range(20):
        hop.observe(20, 'W1AAA', 'beacon')
    hop.observe(40, 'W2AAA', 'cq')
    hop.observe(40, 'W3AAA', 'beacon')
    assert hop.choose()[0] == 40
    clock.utc += 3600  # Across UTC midnight, 23 UTC remains adjacent.
    assert hop.evidence(40)[0] == 2
    clock.utc += HISTORY_S
    assert hop.evidence(40)[0] == 0
    assert hop.choose()[0] == 80  # No evidence: deterministic exploration.


def test_auto_explores_every_third_choice_and_skips_unready_and_cooldown():
    hop, clock = chooser()
    hop.observe(20, 'W1AAA', 'beacon')
    hop.last_observed.update({80: 2, 40: 1, 20: 3})
    assert [hop.choose()[0] for _ in range(3)] == [20, 20, 40]
    hop.readiness[20] = 'needs operator TUNE'
    hop.last_tx[40] = clock.now
    assert hop.choose()[0] == 80
    hop.readiness[80] = 'unknown'
    band, reason = hop.choose()
    assert band is None and 'needs operator TUNE' in reason and 'cooldown' in reason


def test_listening_time_survives_band_changes_but_excludes_gaps_and_contacts():
    hop, clock = chooser()
    hop.sample(20, True)
    clock.now += 1
    hop.sample(20, True)
    clock.now += 100
    hop.sample(20, False)
    hop.sample(40, True)
    clock.now += 1
    hop.sample(40, True)
    assert hop.evidence(20) == hop.evidence(40) == (0, 1)
    clock.utc += HISTORY_S + 3600
    hop.sample(40, False)
    assert not hop.listening


def test_auto_session_uses_heard_events_and_announces_choice():
    desk = hopping_desk('/beacon hop auto 40 20')
    desk.hear('CQFRAME K7QB-9 500')
    wait(desk, 6)
    assert desk.session.hopping.evidence(20)[0] == 1
    assert desk.flow.band == 20 and desk.sent.count(BEACON) == 1
    assert any('unique stations heard' in line for line in desk.said)
