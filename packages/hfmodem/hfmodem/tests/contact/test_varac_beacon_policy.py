# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk, BEACON, CQ_DONE, wait


def test_reconfigure_preserves_expiry_but_off_on_starts_fresh():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    expires = desk.session._beacon_expires
    wait(desk, v.BEACON_CLEAR_S + 1)
    last = desk.session._beacon_last
    desk.do('/beacon 20')
    assert desk.session._beacon_due == last + 1200
    assert desk.session._beacon_expires == expires
    desk.do('/beacon off')
    desk.host.now += 10
    desk.do('/beacon')
    assert desk.session._beacon_expires == desk.host.now + v.BEACON_LIFETIME_S
    assert desk.session._beacon_due == desk.host.now
    wait(desk, 61)
    assert desk.sent.count(BEACON) == 2


def test_beacon_expires_during_contact_and_does_not_resume():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    expires = desk.session._beacon_expires
    desk.hear('CONNECTED KQ4QWH W9SSJ 500')
    desk.host.now = expires
    desk.session._beacon_tick()
    assert desk.session.beacon is None
    assert desk.said[-1] == 'beacon expired after 24 hours'
    desk.hear('DISCONNECTED')
    wait(desk, 61)
    assert BEACON not in desk.sent


def test_band_change_keeps_expiry_but_requires_fresh_clear_period():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    expires = desk.session._beacon_expires
    desk.session.change_band(40)
    assert desk.session._beacon_expires == expires
    desk.host.now = expires
    desk.session._beacon_tick()
    assert desk.session.beacon is None
    assert not desk.kinds('beacon')


def test_invalid_interval_does_not_enable_or_renew():
    desk = Desk()
    desk.do('/beacon 0')
    assert desk.session.beacon is None and desk.session._beacon_expires is None
    desk.do('/beacon')
    expires = desk.session._beacon_expires
    desk.host.now += 100
    desk.do('/beacon -1')
    assert desk.session.beacon == 15 and desk.session._beacon_expires == expires


def test_one_minute_interval_is_allowed():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon 1')
    wait(desk, v.BEACON_CLEAR_S + 1)
    assert desk.sent.count(BEACON) == 1
    wait(desk, 61)
    assert desk.sent.count(BEACON) == 2


def test_now_when_disabled_and_pending_off():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.hear('BUSY ON')
    desk.do('/beacon now')
    wait(desk, 10)
    assert BEACON not in desk.sent
    desk.do('/beacon off')
    desk.hear('BUSY OFF')
    wait(desk, 6)
    assert BEACON not in desk.sent
    desk.do('/beacon now')
    wait(desk, 6)
    assert desk.sent.count(BEACON) == 1
    assert desk.session.beacon is None
    assert desk.session._beacon_last is None
    assert desk.session._beacon_due is None


def test_now_does_not_shift_repeating_schedule():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    wait(desk, 6)
    before = (desk.session._beacon_due, desk.session._beacon_last, desk.session._beacon_expires)
    desk.do('/beacon now')
    wait(desk, 6)
    assert desk.sent.count(BEACON) == 2
    assert before == (desk.session._beacon_due, desk.session._beacon_last, desk.session._beacon_expires)


def test_now_rejects_link_and_keying():
    for attr in ('connected', 'pending', 'ptt'):
        desk = Desk()
        setattr(desk.host.client, attr, True)
        desk.do('/beacon now')
        assert not desk.session._beacon_now
        assert 'requires idle' in desk.said[-1]


def test_off_during_frame_does_not_unkey_or_repopulate_timer():
    desk = Desk(**{BEACON: ['OK', 'PENDING', 'PTT ON']})
    desk.session.command('/beacon')
    for _ in range(100):
        desk.tick()
        if desk.host.client.ptt:
            break
    assert desk.host.client.ptt
    desk.session.command('/beacon off')
    assert desk.host.client.ptt and desk.session.steps is not None
    assert desk.session._beacon_last is desk.session._beacon_due is None
    desk.hear('PTT OFF', 'CANCELPENDING')
    desk.drive()
    assert desk.session._beacon_last is desk.session._beacon_due is None
    assert not any(x in desk.sent for x in ('ABORT', 'DISCONNECT'))


def test_now_hardware_and_operation_guards():
    for guard in ('line', '_closing', '_quitting', '_band_fault'):
        desk = Desk()
        setattr(desk if guard == 'line' else desk.session, guard, True)
        desk.do('/beacon now')
        assert not desk.session._beacon_now
    desk = Desk()
    desk.session.steps = iter([None])
    desk.session.command('/beacon now')
    assert not desk.session._beacon_now


def test_repeat_expiry_does_not_cancel_manual_request():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    desk.hear('BUSY ON')
    desk.do('/beacon now')
    desk.host.now = desk.session._beacon_expires
    desk.session._beacon_tick()
    assert desk.session.beacon is None and desk.session._beacon_now
    desk.hear('BUSY OFF')
    wait(desk, 6)
    assert desk.sent.count(BEACON) == 1


def test_force_bypasses_busy_and_clear_wait_without_changing_timer():
    command = 'CQFRAMEFORCE W9SSJ-9 500'
    desk = Desk(**{command: CQ_DONE})
    desk.do('/beacon')
    before = (desk.session._beacon_due, desk.session._beacon_expires, desk.session._beacon_last)
    desk.hear('BUSY ON')
    desk.do('/beacon force')
    desk.tick()
    desk.drive()
    assert desk.sent.count(command) == 1 and BEACON not in desk.sent
    assert before == (desk.session._beacon_due, desk.session._beacon_expires, desk.session._beacon_last)


def test_force_advanced_beacon_uses_scoped_host_command():
    command = 'BEACONFORCE 20AEN63bbW9SSJ'.upper()
    desk = Desk(kiss=True, loc='EN63bb', **{command: CQ_DONE})
    desk.hear('BUSY ON')
    desk.do('/beacon force')
    desk.tick()
    desk.drive()
    assert command in desk.sent
    assert not desk.host.kiss.sent


def test_force_unsupported_host_reports_failure_without_fallback():
    command = 'CQFRAMEFORCE W9SSJ-9 500'
    desk = Desk(**{command: ['WRONG']})
    desk.hear('BUSY ON')
    desk.do('/beacon force')
    desk.tick()
    desk.drive()
    assert any('unavailable or rejected' in x for x in desk.said)
    assert BEACON not in desk.sent and not desk.kinds('beacon')


def test_force_retains_connected_and_hardware_guards():
    for guard in ('connected', 'pending', 'ptt'):
        desk = Desk()
        setattr(desk.host.client, guard, True)
        desk.do('/beacon force')
        assert not desk.session._beacon_now
    desk = Desk()
    desk.line = True
    desk.do('/beacon force')
    assert not desk.session._beacon_now


def test_stop_cancels_manual_wait_without_disabling_repetition():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    desk.session._beacon_due = 10000
    desk.hear('BUSY ON')
    desk.do('/beacon now')
    desk.do('/stop')
    assert not desk.session._beacon_now and desk.session.beacon == 15
    desk.hear('BUSY OFF')
    wait(desk, 6)
    assert BEACON not in desk.sent


def test_manual_admission_failure_is_not_retried_each_tick():
    desk = Desk()
    desk.do('/beacon now')
    desk.session._band_fault = True
    wait(desk, 6)
    assert not desk.session._beacon_now
    assert len(desk.kinds('error')) == 1


def test_force_async_transport_refusal_never_reports_sent():
    command = 'CQFRAMEFORCE W9SSJ-9 500'
    desk = Desk(**{command: ['OK', 'PENDING', 'PTT ON', 'PTT OFF', 'WRONG', 'CANCELPENDING']})
    desk.do('/beacon force')
    desk.tick()
    desk.drive()
    assert not desk.kinds('beacon')
    assert any('unavailable or rejected' in x for x in desk.said)
    assert not any(x == 'one-shot beacon sent' for x in desk.said)


def test_pending_manual_follows_band_change_after_new_clear_period():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.hear('BUSY ON')
    desk.do('/beacon now')
    desk.session.change_band(40)
    wait(desk, 6)
    assert desk.sent.count(BEACON) == 1
    assert desk.kinds('beacon')[0]['dial_hz'] == 7105000


def test_manual_beacon_completion_without_keying_is_not_sent():
    for mode, command in [('now', BEACON), ('force', 'CQFRAMEFORCE W9SSJ-9 500')]:
        desk = Desk(**{command: ['OK', 'PENDING', 'CANCELPENDING']})
        desk.do('/beacon ' + mode)
        wait(desk, 6)
        assert not desk.kinds('beacon')
        assert not any(x == 'one-shot beacon sent' for x in desk.said)


def test_force_failure_waits_for_unkey_completion():
    command = 'CQFRAMEFORCE W9SSJ-9 500'
    desk = Desk(**{command: ['OK', 'PENDING', 'PTT ON', 'WRONG']})
    desk.session.command('/beacon force')
    desk.tick()
    assert desk.session.steps is not None and desk.host.client.ptt
    desk.hear('PTT OFF', 'CANCELPENDING')
    desk.drive()
    assert not desk.kinds('beacon')
