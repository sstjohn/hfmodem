# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace

import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import CQ_DONE, FRAME, Desk, answered
from hfmodem.tests.contact.test_broadcast_client import bracket, desk, expected, select, start, observed


def beacon(d):
    d.host.replies[f'CQFRAME W9SSJ-9 {d.session.bandwidth}'] = CQ_DONE
    d.session.command('/beacon now')
    d.drive(until=lambda: any(c.startswith('CQFRAME W9SSJ-9') for c in d.sent))
    for _ in range(70):
        d.tick()
    d.drive()


def in_vara(d):
    """Our in-link identifications: the opening report sent again as text."""
    return [t for t in d.texts if t.partition(' ')[2].startswith('de W9SSJ')]


def on_air(d):
    """The modem takes queued text and reports its buffer empty."""
    d.hear('BUFFER 16', 'BUFFER 0')


def clock_restarted(d):
    return d.session._periodic_from() is None


def linked(**kwargs):
    """An answered link whose opening report has gone out."""
    d = answered(**kwargs)
    on_air(d)
    return d


def timed(d):
    records = []
    def identify(bw, after_peer):
        records.append((d.host.now, d.flow.dial, d.host.client.ptt, d.line, after_peer))
        return d.identify(bw, after_peer)
    d.session.keyer = identify
    return records


def test_operator_band_beacon_cq_slot_and_return_do_not_identify():
    d = Desk(**{FRAME: CQ_DONE})
    d.do('/band 40')
    beacon(d)
    d.session.command('/cq 1')
    d.drive(until=lambda: d.flow.state is v.State.WAITING)
    assert not d.ids and d.flow.dial == 7104250
    assert d.sent.count(FRAME) == v.CQ_FRAMES == 2
    d.drive()
    d.tick()
    assert not d.ids and d.flow.dial == 7105000
    d.host.now += 600
    d.tick()
    assert len(d.ids) <= 1


def test_beacon_then_band_change_does_not_identify():
    d = Desk()
    beacon(d)
    d.do('/band 40')
    assert not d.ids and d.flow.band == 40


@pytest.mark.parametrize('command', ['/answer 1', '/call K1ABC 2'])
def test_beacon_then_qso_identifies_only_at_qso_end(command):
    d = Desk(**{'CONNECT W9SSJ K1ABC': ['OK', 'CONNECTED W9SSJ K1ABC 500']})
    beacon(d)
    d.hear('SN 2', 'CQFRAME K1ABC-2 500')
    d.do(command)
    assert not d.ids
    d.hear('PTT ON', 'PTT OFF')
    d.do('/bye')
    d.tick()
    assert [(i['dial'], i['after_peer']) for i in d.ids] == [(14103500, False)]


def test_cq_answered_on_slot_has_no_pre_retune_id_and_one_end_id():
    d = Desk(**{FRAME: CQ_DONE})
    d.session.command('/cq 1')
    d.drive(until=lambda: d.flow.state is v.State.WAITING)
    assert not d.ids
    d.hear('CONNECTED W9SSJ K1ABC 500', 'SN 2')
    d.drive()
    d.hear('PTT ON', 'PTT OFF', 'DISCONNECTED')
    d.drive()
    assert [i['dial'] for i in d.ids] == [14104250]


def test_repeating_cq_over_25_minutes_identifies_itself_and_keys_no_cw():
    """Every CQ frame names us, so each one resets the ten-minute clock."""
    d = Desk(**{FRAME: CQ_DONE})
    records = timed(d)
    d.session.wait_seconds = 15
    d.session.command('/cq 1 repeat')
    while d.host.now < 1600:
        d.tick()
    d.session.command('/stop')
    d.drive()
    assert records == [] and not d.session.owed


def test_long_qso_identifies_in_vara_then_with_the_closing_cw():
    """No CW is keyed inside a link: the ten-minute ID is our report as text."""
    d = linked()
    records = timed(d)
    opened = len(in_vara(d))
    d.hear('PTT ON', 'PTT OFF')
    began = d.host.now
    due = v.ID_INTERVAL_S - v.ID_TEXT_LEAD_S
    d.host.now = began + due - 1
    d.tick()
    assert not records and len(in_vara(d)) == opened
    d.host.now = began + due
    d.tick()
    assert not records and in_vara(d)[opened:] == ['15 de W9SSJ <R+06>']
    assert clock_restarted(d) and d.session.owed, 'text leaves the QSO to the closing ID'
    on_air(d)
    assert not records
    d.hear('PTT ON', 'PTT OFF')
    d.do('/bye')
    d.tick()
    assert len(records) == 1 and not records[0][-1] and records[0][1] == 14106500


def test_no_cw_id_is_keyed_while_a_link_is_up():
    d = linked()
    records = timed(d)
    for _ in range(4):
        d.hear('PTT ON', 'PTT OFF')
        d.host.now += 700
        for _ in range(5):
            d.tick()
    assert records == [] and d.host.client.connected
    assert len(in_vara(d)) >= 4


def test_the_in_link_id_goes_once_and_the_clock_restarts():
    d = linked()
    opened = len(in_vara(d))
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += 600
    for _ in range(20):
        d.tick()
    assert len(in_vara(d)) == opened + 1 and clock_restarted(d)
    on_air(d)
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += v.ID_INTERVAL_S - v.ID_TEXT_LEAD_S - 1
    d.tick()
    assert len(in_vara(d)) == opened + 1
    d.host.now += 1
    d.tick()
    assert len(in_vara(d)) == opened + 2


def test_the_in_link_id_is_queued_early_enough_to_be_on_the_air_inside_ten_minutes():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += v.ID_INTERVAL_S - v.ID_TEXT_LEAD_S
    d.tick()
    assert in_vara(d)[-1] == '15 de W9SSJ <R+06>'


@pytest.mark.parametrize('left', [True, False])
def test_an_id_text_still_queued_at_the_end_of_its_lead_tells_the_operator(left):
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    d.host.data.respond = lambda _: None
    d.host.now += v.ID_INTERVAL_S - v.ID_TEXT_LEAD_S
    d.tick()
    d.hear('BUFFER 15')
    if left:
        d.hear('BUFFER 0')
    d.host.now += v.ID_TEXT_LEAD_S
    d.tick()
    warned = [line for line in d.said if 'W9SSJ is still queued' in line]
    assert len(warned) == (0 if left else 1)
    assert not d.ids
    if not left:
        assert warned[0].endswith('send W9SSJ by hand')


@pytest.mark.parametrize('interval', [540, 0])
def test_the_in_link_id_follows_the_station_interval(interval):
    d = linked(session_options={'id_interval_s': interval})
    records = timed(d)
    opened = len(in_vara(d))
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += 540 - v.ID_TEXT_LEAD_S - 1
    d.tick()
    assert len(in_vara(d)) == opened
    d.host.now += 1
    d.tick()
    assert len(in_vara(d)) == opened + bool(interval) and not records
    d.host.now += 3600
    d.tick()
    assert len(in_vara(d)) == opened + bool(interval) and not records
    d.hear('PTT ON', 'PTT OFF', 'DISCONNECTED')
    d.drive()
    assert len(records) == 1, 'the closing ID follows its own option'


@pytest.mark.parametrize('interval', [60, 120])
def test_a_short_interval_leads_by_half_and_its_own_text_is_not_due_at_once(interval):
    d = linked(session_options={'id_interval_s': interval})
    d.host.data.respond = lambda _: None
    opened = len(in_vara(d))
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += interval / 2 - 1
    d.tick()
    assert len(in_vara(d)) == opened
    d.host.now += 1
    d.tick()
    assert len(in_vara(d)) == opened + 1
    d.hear('BUFFER 15', 'PTT ON', 'PTT OFF')
    d.host.now += interval / 2 - 1
    d.tick()
    assert len(in_vara(d)) == opened + 1
    assert not [line for line in d.said if 'W9SSJ is still queued' in line]
    d.host.now += 1
    d.tick()
    assert [line for line in d.said if 'W9SSJ is still queued' in line] == [
        f'!! text naming W9SSJ is still queued after {interval / 2:.0f} s; '
        'if it has not gone out, send W9SSJ by hand']


@pytest.mark.parametrize('interval', [540, 0])
def test_without_a_link_the_periodic_id_is_cw_at_the_station_interval(interval):
    d = Desk(session_options={'id_interval_s': interval})
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += 539
    d.tick()
    assert not d.ids
    d.host.now += 1
    d.tick()
    assert len(d.ids) == bool(interval)
    d.host.now += 3600
    d.tick()
    assert len(d.ids) == bool(interval)


@pytest.mark.parametrize('text, credits', [
    ('de W9SSJ <R+06>', True), ('W9SSJ here', True), ('tnx w9ssj/p', True),
    ('de KW9SSJ', False), ('W9SSJX', False), ('nothing of ours', False)])
def test_sent_text_naming_the_station_identifies_it(text, credits):
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    assert d.session.owed
    d.do(text)
    on_air(d)
    assert clock_restarted(d) is credits and d.session.owed


@pytest.mark.parametrize('ending', ['sent', 'discarded', 'link-lost'])
def test_text_never_stands_in_for_the_closing_id(ending):
    """No host report says when queued text left, so whatever became of it,
    the QSO's transmissions are still the closing ID's."""
    d = linked()
    records = timed(d)
    d.hear('PTT ON', 'PTT OFF')
    d.host.data.respond = lambda _: None
    d.do('de W9SSJ')
    d.hear('BUFFER 9')
    if ending == 'sent':
        d.hear('BUFFER 0')
    elif ending == 'discarded':
        d.host.client.send_command('CLEANTXBUFFER')
    d.hear('DISCONNECTED')
    d.drive()
    assert len(records) == 1


def test_a_distinct_cw_call_is_what_identifies_in_vara():
    d = linked(session_options={'idcall': 'N0CALL'})
    d.hear('PTT ON', 'PTT OFF')
    d.do('de W9SSJ')
    on_air(d)
    assert not clock_restarted(d)
    d.host.now += 600
    d.tick()
    assert d.texts[-1] == '16 de N0CALL <R+06>'
    assert clock_restarted(d)


@pytest.mark.parametrize('closing', ['local', 'peer', 'stop'])
def test_qso_end_identifies_once_on_its_frequency_before_return(closing):
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    if closing == 'peer':
        d.hear('DISCONNECTED')
    else:
        d.do('/bye' if closing == 'local' else '/stop')
    d.drive()
    d.tick()
    # The closer IDs first, as VARA HF stations do; only a peer close waits for the peer's ID.
    assert [(i['dial'], i['after_peer']) for i in d.ids] == [(14106500, closing == 'peer')]
    assert d.flow.dial == 14105000
    d.tick()
    assert len(d.ids) == 1


@pytest.mark.parametrize('reason', ['pending', 'key', 'physical', 'wrong-frequency'])
def test_periodic_waits_for_quiet_on_last_transmit_frequency(reason):
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += 600
    if reason == 'pending':
        d.host.client.pending = True
    elif reason == 'key':
        d.host.client.ptt = True
    elif reason == 'physical':
        d.line = True
    else:
        d.flow.dial += 750
    d.tick()
    assert not d.ids
    d.host.client.pending = d.host.client.ptt = d.line = False
    d.flow.dial = 14105000
    d.tick()
    assert len(d.ids) == 1


def test_quit_keys_final_identification_after_unidentified_transmission():
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.session.quit()
    assert len(d.ids) == 1
    assert not any('by hand' in s for s in d.said)


def test_quit_with_closed_host_reports_by_hand_without_keying():
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.host.client.closed = True
    d.session.quit()
    assert not d.ids
    assert sum('send W9SSJ by hand' in s for s in d.said) == 1


@pytest.mark.parametrize('command', ['/band 40', '/all slot 12'])
def test_unlinked_retunes_under_ten_minutes_do_not_identify(command):
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.do(command)
    assert not d.ids


def test_a_callsign_cq_identifies_so_quit_keys_no_cw():
    d = Desk(**{FRAME: CQ_DONE})
    d.do('/cq 1')
    d.tick()
    d.session.quit()
    assert d.ids == []
    assert not any('by hand' in s for s in d.said)


def test_none_mode_completed_call_frame_resets_clock_but_answered_link_data_owes_again():
    d = Desk(**{FRAME: CQ_DONE})
    d.do('/cq 1')
    assert not d.session.owed
    d.hear('CONNECTED W9SSJ K1ABC 500', 'SN 2', 'PTT ON', 'PTT OFF')
    d.host.now += 600
    d.tick()
    assert not d.ids and len(in_vara(d)) == 1 and clock_restarted(d)


@pytest.mark.parametrize('advanced', [False, True])
def test_none_mode_beacon_identifies_only_after_completion(advanced):
    d = Desk(kiss=advanced, loc='EN63BC')
    beacon(d)
    assert not d.session.owed and not d.ids
    d.session.quit()
    assert not d.ids


def test_none_mode_all_frame_identifies_only_after_matching_bracket():
    d = desk()
    start(d)
    d.hear('PENDING', 'PTT ON')
    assert d.session.owed
    observed(d)
    assert not d.session.owed
    d.session.quit()
    assert not d.ids


@pytest.mark.parametrize('closing_cwid,separate,count', [(False, False, 0), (True, False, 1), (False, True, 1)])
def test_closing_cwid_setting_preserves_distinct_cw_identity(closing_cwid, separate, count):
    d = linked(session_options={'closing_cwid': closing_cwid,
                                 'idcall': 'N0CALL' if separate else 'W9SSJ'})
    d.hear('PTT ON', 'PTT OFF', 'DISCONNECTED')
    d.drive()
    assert len(d.ids) == count
    if not count:
        assert d.session.owed
        d.host.now += 600
        d.flow._tune(14106500)
        d.tick()
        assert len(d.ids) == 1


@pytest.mark.parametrize('stopped', [False, True])
def test_call_never_connected_has_no_end_identification(stopped):
    reply = ['OK', 'PTT ON'] if stopped else ['OK', 'PTT ON', 'PTT OFF', 'DISCONNECTED']
    d = Desk(**{'CONNECT W9SSJ K1ABC': reply})
    d.session.command('/call K1ABC 2')
    if stopped:
        d.drive(until=lambda: d.flow.state is v.State.CONNECTING)
        d.session.command('/stop')
        d.drive(until=lambda: 'ABORT' in d.sent)
        d.hear('PTT OFF')
    d.drive()
    d.tick()
    assert not d.ids


def test_cq_repeat_can_be_replaced_by_answer_without_identifying_before_retune():
    d = Desk(**{FRAME: CQ_DONE,
                'CONNECT W9SSJ K1ABC': ['OK', 'CONNECTED W9SSJ K1ABC 500']})
    d.session.command('/cq 1 repeat')
    d.drive(until=lambda: d.flow.state is v.State.WAITING)
    d.hear('SN 2', 'CQFRAME K1ABC-2 500')
    d.do('/answer 1')
    assert not d.ids and d.flow.dial == 14103500
    d.hear('PTT ON', 'PTT OFF', 'DISCONNECTED')
    d.drive()
    assert [i['dial'] for i in d.ids] == [14103500]


def test_overdue_periodic_id_follows_a_new_transmission_on_the_current_band():
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.do('/band 40')
    d.host.now += 600
    d.tick()
    assert not d.ids
    d.hear('PTT ON')
    assert not d.ids
    d.hear('PTT OFF')
    assert [i['dial'] for i in d.ids] == [7105000]


def test_periodic_clock_restarts_at_next_transmission_after_identification():
    d = Desk()
    records = timed(d)
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += 600
    d.tick()
    assert len(records) == 1
    d.host.now += 300
    d.hear('PTT ON', 'PTT OFF')
    next_tx = d.host.now
    d.host.now = next_tx + 599
    d.tick()
    assert len(records) == 1
    d.host.now = next_tx + 600
    d.tick()
    assert len(records) == 2 and records[1][0] - records[0][0] >= 900


@pytest.mark.parametrize('frame', ['cq', 'advanced', 'forced'])
def test_none_mode_callsign_frame_waits_for_confirmed_completion(frame):
    d = Desk(kiss=frame == 'advanced', loc='EN63BC')
    d.host.replies[FRAME] = ['OK', 'PENDING', 'PTT ON', 'PTT OFF']
    if frame == 'advanced':
        d.host.kiss.respond = lambda _: d.host.lines('PENDING', 'PTT ON', 'PTT OFF')
        d.session.intent = 'beacon'
        d.session._run(d.session._beaconing(once=True))
    elif frame == 'forced':
        d.host.replies['CQFRAMEFORCE W9SSJ-9 500'] = ['OK', 'PENDING', 'PTT ON', 'PTT OFF']
        d.session.intent = 'beacon'
        d.session._run(d.session._beaconing(once='force'))
    else:
        d.session.command('/cq 1')
        d.drive(until=lambda: FRAME in d.sent)
    d.tick()
    assert d.session.owed and not d.ids
    d.hear('CANCELPENDING')
    assert not d.session.owed and not d.ids
    d.session.command('/stop')
    d.drive()


def test_none_mode_outbound_setup_identifies_then_arq_data_owes_end_cw():
    d = Desk(**{
        'CONNECT W9SSJ K1ABC': ['OK', 'PENDING', 'PTT ON', 'CONNECTED W9SSJ K1ABC 500', 'SN 2']})
    d.session.command('/call K1ABC 2')
    d.tick()
    assert d.session.owed and d.host.client.ptt
    d.hear('PTT OFF')
    d.drive()
    assert not d.session.owed and not d.ids
    d.do('de W9SSJ')
    d.hear('PTT ON', 'PTT OFF')
    assert d.session.owed
    d.do('/bye')
    d.tick()
    assert [i['dial'] for i in d.ids] == [14103500]


def test_none_mode_answered_link_setup_does_not_identify_us():
    d = Desk()
    d.hear('PENDING', 'PTT ON', 'PTT OFF', 'CONNECTED W9SSJ K1ABC 500')
    assert d.session.owed
    d.hear('DISCONNECTED')
    d.drive()
    assert len(d.ids) == 1


def test_disabled_closing_cw_keeps_quit_obligation():
    d = Desk(session_options={'closing_cwid': False})
    d.hear('CONNECTED W9SSJ K1ABC 500', 'PTT ON', 'PTT OFF', 'DISCONNECTED')
    d.drive()
    assert d.session.owed and not d.ids
    d.session.quit()
    assert len(d.ids) == 1


def test_no_transmission_starts_while_quit_unwinds_an_error():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    before = len(d.sent)
    texts = list(d.texts)
    with pytest.raises(OSError, match='host stopped'):
        try:
            raise OSError('host stopped')
        finally:
            d.session.quit()
    assert 'ABORT' in d.sent[before:]
    assert 'DISCONNECT' not in d.sent[before:]
    assert 'CLEANTXBUFFER' not in d.sent[before:]
    assert d.texts == texts
    assert not d.ids
    assert sum('send W9SSJ by hand' in s for s in d.said) == 1


def test_quit_with_unknown_key_state_reports_by_hand():
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.session.line_state = lambda: None
    d.session.quit()
    assert not d.ids
    assert sum('send W9SSJ by hand' in s for s in d.said) == 1


def test_the_in_link_id_text_leaves_the_end_of_the_qso_to_the_closing_id():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    d.host.now += 600
    d.tick()
    on_air(d)
    assert clock_restarted(d)
    d.hear('DISCONNECTED')
    d.drive()
    assert len(d.ids) == 1


def test_periodic_identification_waits_for_a_running_in_link_operation():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    resumed, cancelled = [], []
    def waiting():
        try:
            while True:
                resumed.append(d.host.now)
                yield .1
        finally:
            cancelled.append(True)
    d.session._run(waiting())
    d.host.now += 600
    d.tick()
    assert not d.ids and d.host.client.connected
    assert d.session.steps is not None and len(resumed) >= 2
    opened = len(in_vara(d))
    d.session.steps.close()
    d.session.steps = None
    assert cancelled == [True]
    d.tick()
    assert not d.ids and len(in_vara(d)) == opened + 1


def test_none_mode_another_callsign_in_a_frame_does_not_identify_us():
    d = Desk(**{'CQFRAME K1ABC-9 500': CQ_DONE})
    d.host.client.finish(d.host.client.cq_steps('CQFRAME K1ABC-9 500'))
    assert d.session.owed and not d.ids


def test_quit_after_retuning_cannot_identify_on_the_wrong_frequency():
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.do('/band 40')
    d.session.quit()
    assert not d.ids
    assert sum('send W9SSJ by hand' in s for s in d.said) == 1


@pytest.mark.parametrize('flow', ['cq', 'between', 'beacon'])
def test_brief_qso_stops_repeating_flow_and_identifies_before_retune(flow):
    d = Desk(**{FRAME: CQ_DONE})
    if flow == 'beacon':
        d.host.replies['CQFRAME W9SSJ-9 500'] = ['OK', 'PENDING', 'PTT ON']
        d.session.command('/beacon now')
        for _ in range(100):
            d.tick()
            if 'CQFRAME W9SSJ-9 500' in d.sent:
                break
        assert 'CQFRAME W9SSJ-9 500' in d.sent and d.session.steps is not None
    else:
        d.session.command('/cq 1 repeat')
        d.drive(until=lambda: d.flow.state is v.State.WAITING)
        if flow == 'between':
            d.drive(until=lambda: d.flow.state is v.State.RETURNED)
    qso_dial = d.flow.dial
    before = len(d.sent)
    def return_after_id(dial):
        assert [i['dial'] for i in d.ids] == [qso_dial]
        d.tunes.append(dial)
    d.flow.tuner = return_after_id
    d.hear('CONNECTED W9SSJ K1ABC 500', 'PTT ON', 'PTT OFF', 'DISCONNECTED')
    for _ in range(10):
        d.tick()
    assert [(i['dial'], i['after_peer']) for i in d.ids] == [(qso_dial, True)]
    assert not any(c.startswith('CQFRAME') for c in d.sent[before:])
    assert not d.session.repeat and d.session.steps is None
    assert not any('frequency unavailable' in line for line in d.said)
    d.host.now += 650
    d.tick()
    assert len(d.ids) == 1


@pytest.mark.parametrize('closing', [False, True])
def test_unknown_key_state_defers_periodic_and_closing_ids(closing):
    d = linked() if closing else Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.line = None
    original = d.flow.dial
    if closing:
        d.hear('DISCONNECTED')
    else:
        d.host.now += 600
        d.tick()
    assert not d.ids and d.session.owed
    assert d.flow.dial == original
    d.line = False
    d.tick()
    d.drive()
    assert [i['dial'] for i in d.ids] == [original]


def test_the_in_link_id_queues_behind_text_the_modem_has_not_reported():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    d.host.data.respond = lambda _: None
    d.do('still queued')
    assert not d.host.client._writes and d.host.client.data_writer is None
    assert d.session._data_status() == (
        True, 'text queued locally; waiting for a fresh modem buffer report')
    opened = len(in_vara(d))
    d.host.now += 600
    d.tick()
    assert len(in_vara(d)) == opened + 1 and not d.ids, 'ID text queues behind text'
    d.hear('BUFFER 5', 'BUFFER 0')
    assert len(in_vara(d)) == opened + 1


def test_periodic_id_waits_for_received_qsy_owner_to_finish():
    d = Desk()
    d.hear('CONNECTED W9SSJ K1ABC 500', 'SN 2.0', 'PTT ON', 'PTT OFF')
    d.host.now += 600
    moved = []
    release = [False]
    def owner_move(target, generation):
        while not release[0]:
            yield .1
        moved.append(target)
        d.flow.tuner(target)
    d.flow.received_retune_steps = owner_move
    d.receive('<QSYF>000014103500</QSYF>')
    assert d.flow.qsy_receiving
    assert d.flow.qsy_phase == 'waiting for received DATA acknowledgement and physical unkey'
    assert not moved and not d.ids and not in_vara(d)
    release[0] = True
    d.tick()
    assert moved == [14103500] and not d.ids and not in_vara(d)
    assert d.flow.qsy_phase == 'waiting for peer arrival (<TL>)'
    d.receive('<TL>')
    d.receive('de K1ABC <R+02>')
    d.drive()
    assert in_vara(d) == ['15 de W9SSJ <R+02>'] and not d.ids, 'our report after the QSY'


def test_periodic_id_waits_for_mail_collection_and_outbound_qsy_readiness():
    d = Desk(**{'CONNECT W9SSJ K1ABC': ['OK', 'CONNECTED W9SSJ K1ABC 500']})
    d.hear('SN 2', 'PTT ON', 'PTT OFF')
    d.host.now += 599
    d.session.command('/vmail collect K1ABC')
    d.drive(until=lambda: d.flow.qsy_phase is not None)
    assert d.session.steps is not None
    assert d.flow.qsy_phase == 'waiting for peer auto-QSY readiness (<AWQ> or <Q>)'
    d.host.now += 1
    d.tick()
    assert not d.ids and not in_vara(d)
    d.session.command('/stop')
    d.tick()
    assert not d.ids and len(in_vara(d)) == 1 and d.host.client.connected


def test_pending_closing_id_finishes_before_a_deferred_retune():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    qso_dial = d.flow.dial
    d.line = None
    d.hear('DISCONNECTED')
    d.flow.tune(14103500)
    assert d.session.steps is not None and not d.ids
    d.host.now += 650
    d.tick()
    assert d.flow.dial == qso_dial and not d.ids
    def after_id(target):
        assert [i['dial'] for i in d.ids] == [qso_dial]
        d.tunes.append(target)
    d.flow.tuner = after_id
    d.line = False
    d.tick()
    assert d.flow.dial == 14103500 and len(d.ids) == 1
    assert d.session._closing_dial is None and not d.session.owed


def test_closing_id_uses_owner_qsy_completion_published_during_cancellation(tools):
    import varac_chat
    d = Desk()
    d.hear('CONNECTED W9SSJ K1ABC 500', 'PTT ON', 'PTT OFF')
    request = SimpleNamespace(done=False, error=None, cancelled=False, cancel=lambda **_: None)
    moves = []
    def queued(callback, **kwargs):
        moves.append(callback)
        return request
    runtime = SimpleNamespace(modem=SimpleNamespace(station=object(), received_cursor=0),
                              request_retune=queued)
    d.flow.received_retune_steps = varac_chat.received_retune_adapter(
        runtime, d.host.client, d.flow.tuner, lambda: True, completed=d.flow.received_tuned)
    d.receive('<QSYF>000014103500</QSYF>')
    assert d.flow.qsy_receiving and len(moves) == 1
    moves[0]()
    request.done = True
    assert d.flow.dial == 14105000 and d.tunes[-1] == 14103500
    d.hear('DISCONNECTED')
    d.drive()
    assert [i['dial'] for i in d.ids] == [14103500]
    assert not d.session.owed
    assert not any('frequency unavailable' in line for line in d.said)


@pytest.mark.parametrize('state', ['files', 'hosting', 'bbs-output', 'email', 'bbs-wait',
                                  'broadcast', 'opening', 'qsy', 'mail'])
def test_periodic_id_waits_for_application_owners_even_without_a_flow(state, monkeypatch):
    d = linked()
    s = d.session
    d.hear('PTT ON', 'PTT OFF')
    owned = [True]
    if state == 'files':
        s.files.opaque = True
        s.files.generation, s.files.dial = s.client.link_generation, s.flow.dial
        release = lambda: setattr(s.files, 'opaque', False)
    elif state == 'hosting':
        monkeypatch.setattr(type(s.bbs_host), 'active', property(lambda _: owned[0]))
        monkeypatch.setattr(s.bbs_host, 'tick', lambda: None)
        release = lambda: owned.clear() or owned.append(False)
    elif state == 'bbs-output':
        s.bbs.remote_view = True
        s.bbs.generation, s.bbs.dial = s.client.link_generation, s.flow.dial
        s.bbs.inflight = dict(after=None)
        release = lambda: setattr(s.bbs, 'inflight', None)
    elif state == 'email':
        s.email.waiting = True
        s.email.phase = 'waiting for gateway response'
        release = lambda: setattr(s.email, 'waiting', False)
    elif state == 'bbs-wait':
        s._bbs_waiting = True
        release = lambda: setattr(s, '_bbs_waiting', False)
    elif state == 'broadcast':
        s.broadcast.active = True
        s.broadcast.phase = 'waiting for matching host frame'
        release = lambda: setattr(s.broadcast, 'active', False)
    elif state == 'opening':
        s._opening = True
        release = lambda: setattr(s, '_opening', False)
    elif state == 'mail':
        monkeypatch.setattr(type(s.parking), 'active', property(lambda _: owned[0]))
        release = lambda: owned.clear() or owned.append(False)
    else:
        s.flow.qsy_phase = 'waiting for peer acceptance (<QSYR>)'
        release = lambda: setattr(s.flow, 'qsy_phase', None)
    assert s.steps is None
    opened = len(in_vara(d))
    d.host.now += 600
    d.tick()
    assert not d.ids and s.owed
    assert len(in_vara(d)) == opened, 'no ID text goes into an exchange in progress'
    release()
    d.tick()
    assert not d.ids and len(in_vara(d)) == opened + 1, 'the ID follows the exchange'


@pytest.mark.parametrize('frame', ['setup', 'cq', 'beacon', 'forced', 'broadcast'])
def test_none_mode_delayed_frame_credit_preserves_later_transmission_debt(frame):
    d = desk() if frame == 'broadcast' else Desk(kiss=frame == 'beacon')
    starts = []
    event = d.host.client.event
    def timestamped(item):
        if item.kind == 'host' and item.raw == 'PTT ON':
            d.host.now += 1
            starts.append(d.host.now)
        event(item)
    d.host.client.event = timestamped
    if frame == 'setup':
        d.host.replies['CONNECT W9SSJ K1ABC'] = [
            'OK', 'PENDING', 'PTT ON', 'PTT OFF', 'CONNECTED W9SSJ K1ABC 500',
            'PTT ON', 'PTT OFF', 'SN 2']
        d.do('/call K1ABC 2')
    elif frame == 'broadcast':
        start(d)
        # Both the completed UI bracket and later ARQ bracket reach one poll.
        from hfmodem.tests.contact.test_broadcast_client import expected
        d.hear('PENDING', 'PTT ON', *(['BITRATE (1) 18 bps TX'] * expected(d)),
               'PTT OFF', 'CANCELPENDING', 'PTT ON', 'PTT OFF')
        d.do('/all clear')
    elif frame == 'beacon':
        d.host.kiss.respond = lambda _: d.host.lines(*CQ_DONE[1:], 'PTT ON', 'PTT OFF')
        d.host.client.finish(d.host.client.beacon_steps('20AEN63BCW9SSJ'))
    elif frame == 'forced':
        d.host.replies['CQFRAMEFORCE W9SSJ-9 500'] = CQ_DONE + ['PTT ON', 'PTT OFF']
        d.host.client.finish(d.host.client.forced_beacon_steps())
    else:
        command = FRAME if frame == 'cq' else 'CQFRAME W9SSJ-9 500'
        d.host.replies[command] = CQ_DONE + ['PTT ON', 'PTT OFF']
        d.host.client.finish(d.host.client.cq_steps(command))
    assert d.session.owed and not d.ids
    first_after_frame = d.session._first_unidentified_tx
    assert first_after_frame == starts[-1]
    if frame == 'setup':
        d.hear('DISCONNECTED')
        d.drive()
    else:
        d.host.now = first_after_frame + 600
        d.tick()
    assert len(d.ids) == 1 and not d.session.owed


@pytest.mark.parametrize('ending', ['quit', 'quit-unknown-key'])
def test_cancelled_periodic_id_retains_debt_and_original_clock(ending):
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    started, completed, cancelled_debt = [], [], []

    def identify(bw, after_peer):
        started.append((d.flow.dial, after_peer, d.session._first_unidentified_tx))
        if len(started) == 1:
            try:
                while True:
                    yield .1
            finally:
                cancelled_debt.append((d.session.owed, d.session._first_unidentified_tx))
        completed.append(d.flow.dial)
        return v.Identification(d.line is False, d.host.client.transmission_count)

    d.session.keyer = identify
    d.host.now = first_tx + 600
    d.tick()
    assert len(started) == 1 and not completed
    if ending == 'quit-unknown-key':
        d.line = None
    d.session.quit()
    assert cancelled_debt == [(True, first_tx)]
    if ending == 'quit-unknown-key':
        assert not completed
        assert sum('send W9SSJ by hand' in s for s in d.said) == 1
    else:
        assert completed == [d.flow.dial] and len(started) == 2
        assert not any('by hand' in s for s in d.said)
    assert all(clock == first_tx for _, _, clock in started)


@pytest.mark.parametrize('failure', [False, True])
def test_periodic_id_credits_only_successful_keyer_completion(failure):
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    finish = [False]

    def identify(bw, after_peer):
        while not finish[0]:
            yield .1
        if failure:
            raise OSError('keyer failed')
        return v.Identification(d.line is False, d.host.client.transmission_count)

    d.session.keyer = identify
    d.host.now = first_tx + 600
    d.tick()
    assert d.session.owed and d.session._first_unidentified_tx == first_tx
    finish[0] = True
    if failure:
        with pytest.raises(OSError, match='keyer failed'):
            d.tick()
        assert d.session.owed and d.session._first_unidentified_tx == first_tx
    else:
        d.tick()
        assert not d.session.owed and d.session._first_unidentified_tx is None


def test_completed_all_acknowledgement_does_not_block_periodic_id():
    d = desk()
    select(d)
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    d.host.now = first_tx + 599
    d.session.command('/all hello everyone')
    d.tick()
    # An unrelated completed bracket cannot settle this frame's KISS write.
    bracket(d, expected(d) + 1)
    d.host.now = first_tx + 600
    d.tick()
    assert not d.host.client.ptt and not d.host.client.pending
    assert d.session.broadcast.active and not d.ids
    observed(d)
    assert not d.session.broadcast.active
    assert d.session.broadcast.possibly_outstanding == 1
    assert not d.ids and not d.session.owed, 'the /all frame names us'


@pytest.mark.parametrize('waiting', ['menu', 'input'])
def test_interactive_bbs_identifies_between_exchanges_without_detaching(request, waiting):
    d = linked()
    request.addfinalizer(d.host.client.close)
    d.hear('PTT ON', 'PTT OFF')
    d.session.command('/bbs enter')
    def settle_writer():
        for _ in range(100):
            d.tick()
            if not d.host.client.data_writer.pending and not d.host.client._writes:
                return
            v.time.sleep(.001)
        pytest.fail('BBS writer did not settle')

    settle_writer()
    d.hear('BUFFER 5', 'BUFFER 0')
    assert d.session.bbs.owned and not d.session._data_status()[0]
    d.hear('PTT ON', 'PTT OFF')
    d.receive('BBS menu: 1. Read messages')
    d.host.now = d.session._first_unidentified_tx + 600
    if waiting == 'input':
        d.host.data.respond = lambda _: None
        d.session.command('1')
        settle_writer()
        assert d.session._data_status()[0] and not d.ids
        d.hear('BUFFER 1', 'BUFFER 0')
    else:
        d.tick()
    assert not d.ids and in_vara(d)
    settle_writer()
    on_air(d)
    assert d.session.bbs.owned and d.host.client.connected
    assert clock_restarted(d)


@pytest.mark.parametrize('result, asynchronous', [(None, False), (False, True), (None, True)])
def test_unconfirmed_keyer_return_keeps_debt_without_retrying_each_tick(result, asynchronous):
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    attempts = []

    def identify(bw, after_peer):
        attempts.append(d.host.now)
        if asynchronous:
            def steps():
                yield .1
                yield .1
                return result
            return steps()
        return result

    d.session.keyer = identify
    d.host.now = first_tx + 600
    d.tick()
    d.drive()
    assert d.session.owed and d.session._first_unidentified_tx == first_tx
    for _ in range(10):
        d.tick()
    assert len(attempts) == 1
    d.host.now += 600
    d.tick()
    d.drive()
    assert len(attempts) == 2 and d.session.owed
    for _ in range(10):
        d.tick()
    assert len(attempts) == 2
    d.session.quit()
    assert len(attempts) == 3 and d.session.owed and d.session._first_unidentified_tx == first_tx
    d.session.identification_unsent()
    assert sum('session ended before identification' in line for line in d.said) == 1


def successful_id(d):
    def identify(bw, after_peer):
        yield from ()
        return d.identify(bw, after_peer)
    d.session.keyer = identify


@pytest.mark.parametrize('ending', ['cancel', 'uncertain-write'])
@pytest.mark.parametrize('acknowledge', [False, True])
def test_unresolved_all_survives_its_wait_and_operator_acknowledgement(ending, acknowledge):
    d = desk()
    successful_id(d)
    select(d)
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    d.host.now = first_tx + 599
    if ending == 'uncertain-write':
        socket = d.host.kiss
        def fail(wire):
            socket.sent.append(wire)
            raise OSError('write extent unknown')
        socket.sendall = fail
    d.session.command('/all hello everyone')
    d.tick()
    if ending == 'cancel':
        d.do('/stop')
    assert not d.session.broadcast.active and d.session.steps is None
    if acknowledge:
        d.do('/all clear')
    d.host.now = first_tx + 601
    d.tick()
    assert not d.ids and d.session.owed
    bracket(d, expected(d) + 1)
    assert not d.ids and d.session.owed
    observed(d)
    assert not d.ids and not d.session.owed, 'the /all frame names us'
    assert d.session.broadcast.possibly_outstanding == (0 if acknowledge else 1)


@pytest.mark.parametrize('ending', ['cancel', 'uncertain-write', 'waiting'])
def test_unresolved_all_has_a_bounded_id_hold_without_operator_action(ending):
    d = desk()
    successful_id(d)
    select(d)
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    d.host.now = first_tx + 599
    written = []
    def submit(wire):
        written.append(d.host.now)
        if ending == 'uncertain-write':
            raise OSError('write extent unknown')
    d.host.kiss.respond = submit
    d.session.command('/all hello everyone')
    d.tick()
    if ending == 'cancel':
        d.do('/stop')
    assert len(written) == 1 and d.session.broadcast.active == (ending == 'waiting')
    d.host.now = written[0] + v.KISS_SUBMISSION_TIMEOUT_S - 1
    d.tick()
    assert not d.ids and d.session.owed
    d.host.now = written[0] + v.KISS_SUBMISSION_TIMEOUT_S
    d.tick()
    assert len(d.ids) == 1 and not d.session.owed
    assert d.session.broadcast.possibly_outstanding == 1
    if ending == 'waiting':
        # The short ID hold does not end the operator's completion wait.
        assert d.session.broadcast.active and d.session.steps is not None
        assert len(d.host.kiss.sent) == 1
        observed(d)
        assert len(d.host.kiss.sent) == 1 and len(d.ids) == 1
    assert not d.session.broadcast.active and d.session.steps is None


@pytest.mark.parametrize('ending', ['complete', 'stop'])
def test_broadcast_wait_pauses_for_yielding_id_without_resending(ending):
    d = desk()
    select(d)
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    d.host.now = first_tx + 599
    d.session.command('/all hello everyone')
    d.tick()
    written = d.host.now
    finish, unkeyed = [False], []

    def identify(bw, after_peer):
        d.line = True
        try:
            while not finish[0]:
                yield .1
        finally:
            d.line = False
            unkeyed.append(d.host.now)
        return d.identify(bw, after_peer)

    d.session.keyer = identify
    d.host.now = written + v.KISS_SUBMISSION_TIMEOUT_S
    for _ in range(3):
        d.tick()
        assert d.session._identifying and d.line and d.session.owed
        assert d.session.broadcast.active and not d.ids
        assert len(d.host.kiss.sent) == 1
    if ending == 'stop':
        d.session.command('/stop')
        d.tick()
        assert not d.session.broadcast.active and d.session.steps is None
        assert d.session.owed and d.session._first_unidentified_tx == first_tx
        assert not d.ids
    else:
        finish[0] = True
        d.tick()
        assert len(d.ids) == 1 and not d.session.owed
        assert d.session.broadcast.active and d.session.steps is not None
        d.host.now += 10000
        d.tick()
        assert d.session.broadcast.active and len(d.ids) == 1
        observed(d)
        assert not d.session.broadcast.active and d.session.steps is None
    assert len(unkeyed) == 1 and not d.line and not d.session._identifying
    assert len(d.host.kiss.sent) == 1


def yielding_id(d):
    finish, turns = [False], []
    def identify(bw, after_peer):
        for _ in range(3):
            turns.append(d.host.now)
            yield .1
        while not finish[0]:
            yield .1
        return d.identify(bw, after_peer)
    d.session.keyer = identify
    d.session._run(d.session._identify_steps())
    return finish, turns


def test_hosted_bbs_next_page_waits_for_yielding_identification():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    bbs, c = d.session.bbs, d.host.client
    first = SimpleNamespace(output_id='first', kind='text', text='first page')
    second = SimpleNamespace(output_id='second', kind='text', text='second page')
    done, polls = [], []
    def poll(sid):
        polls.append(sid)
        return [second]
    bbs.core = SimpleNamespace(tick=lambda: None, session=SimpleNamespace(id='bbs', state='lobby'),
                               poll=poll, output_done=lambda *args: done.append(args))
    bbs.sid = 'bbs'
    bbs.generation, bbs.dial = c.link_generation, d.flow.dial
    bbs.inflight = dict(event=first, after=c.sequence - 1)
    c.buffer_sequence = c.sequence
    bbs.tick()
    assert done == [('bbs', 'first', 'complete')] and bbs.inflight is None
    finish, turns = yielding_id(d)
    before = list(d.texts)
    for _ in range(4):
        d.tick()
        assert d.session._identifying and d.texts == before and not polls
    assert len(turns) == 3
    finish[0] = True
    d.tick()
    d.tick()
    assert not d.session._identifying and len(d.ids) == 1
    assert d.texts == before + ['11 second page'] and polls == ['bbs']


@pytest.mark.parametrize('line, text', [('1', '1'), ('', 'BBS MORE'),
                                       ('/bbs term plain', 'BBS TERM PLAIN 80 24')])
def test_remote_bbs_input_waits_for_yielding_identification(line, text):
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    bbs = d.session.bbs
    bbs.remote_view = True
    bbs.generation, bbs.dial = d.host.client.link_generation, d.flow.dial
    finish, turns = yielding_id(d)
    before = list(d.texts)
    d.session.command(line)
    assert d.texts == before
    for _ in range(4):
        d.tick()
        assert d.session._identifying and d.texts == before
    assert len(turns) == 3
    finish[0] = True
    d.tick()
    d.tick()
    assert not d.session._identifying and len(d.ids) == 1
    assert d.texts == before + [f'{len(text.encode())} {text}']
    d.tick()
    assert d.texts == before + [f'{len(text.encode())} {text}']


def test_cancelled_all_completion_after_id_hold_expiry_still_identifies():
    d = desk()
    select(d)
    d.hear('PTT ON', 'PTT OFF')
    d.host.now = d.session._first_unidentified_tx + 599
    d.session.command('/all hello everyone')
    d.tick()
    written = d.host.now
    d.do('/stop')
    d.host.now = written + v.KISS_SUBMISSION_TIMEOUT_S
    d.tick()
    assert len(d.ids) == 1 and not d.session.owed
    assert not d.host.client.unresolved_kiss
    observed(d)
    assert not d.session.owed
    assert not d.host.client._kiss_submissions
    d.session.quit()
    assert len(d.ids) == 1


@pytest.mark.parametrize('completion', ['after-expiry', 'spans-expiry'])
def test_cancelled_all_after_tracking_expiry_costs_at_most_one_extra_cw_id(completion):
    d = desk()
    select(d)
    d.hear('PTT ON', 'PTT OFF')
    d.host.now = d.session._first_unidentified_tx + 599
    d.session.command('/all hello everyone')
    d.tick()
    submission = d.host.client._kiss_submissions[0]
    d.do('/stop')
    d.host.now = submission['deadline']
    d.tick()
    assert len(d.ids) == 1 and not d.session.owed
    if completion == 'after-expiry':
        d.host.now = submission['expires'] + 1
        d.tick()
        observed(d)
    else:
        d.host.now = submission['expires'] - .5
        for line in ('PENDING', 'PTT ON'):
            d.host.client._line(line)
        d.host.now = submission['expires'] + 1
        for line in (*(['BITRATE (1) 18 bps TX'] * expected(d)), 'PTT OFF', 'CANCELPENDING'):
            d.host.client._line(line)
    d.session.quit()
    assert 1 <= len(d.ids) <= 2
    assert not d.session.owed
    assert not any('send W9SSJ by hand' in line for line in d.said)


def test_cw_receipt_covers_unpolled_earlier_transmissions_only():
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    result = []
    def identify(bw, peer):
        result.append(v.Identification(True, d.host.client.transmission_count + 1))
        yield .1
        return result[0]
    d.session.keyer = identify
    d.session._run(d.session._identify_steps())
    # The host's count includes a send whose lines are still on the command socket.
    d.session._step()
    assert not d.session.owed
    d.host.lines('PTT ON', 'PTT OFF')
    d.tick()
    assert not d.session.owed
    d.hear('PTT ON', 'PTT OFF')
    assert d.session.owed


@pytest.mark.parametrize('result', [True, v.Identification(False, 1)])
def test_keyer_without_unkey_evidence_keeps_debt_and_requires_hand_id(result):
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    d.session.keyer = lambda bw, peer: result
    with pytest.raises(v.HostError, match='CW unkey not confirmed'):
        d.session._run(d.session._identify_steps())
    assert d.session.owed and d.session._identification_failed
    d.session.identification_unsent()
    assert sum('send W9SSJ by hand' in line for line in d.said) == 1


@pytest.mark.parametrize('result', [False, None])
def test_keyer_without_unkey_evidence_stops_on_live_high(result):
    d = Desk()
    d.hear('PTT ON', 'PTT OFF')
    first_tx = d.session._first_unidentified_tx
    def identify(bw, peer):
        yield .1
        d.line = True
        return result
    d.session.keyer = identify
    d.session._run(d.session._identify_steps())
    with pytest.raises(v.HostError, match='keying line remains HIGH after identification'):
        d.session._step()
    assert d.session._identification_failed
    assert d.session.owed and d.session._first_unidentified_tx == first_tx


def test_the_in_link_id_waits_for_a_signal_report_to_carry():
    d = Desk()
    d.hear('CONNECTED W9SSJ K1ABC 500', 'PTT ON', 'PTT OFF')
    d.host.now += 600
    for _ in range(5):
        d.tick()
    assert not in_vara(d) and not d.ids
    d.hear('SN -3.4')
    d.tick()
    assert in_vara(d) == ['15 de W9SSJ <R-03>']


def test_text_written_through_the_writer_identifies_once_the_host_has_it(request):
    d = linked()
    request.addfinalizer(d.host.client.close)
    d.host.client.enable_writer()
    d.hear('PTT ON', 'PTT OFF')
    d.do('de W9SSJ here')
    for _ in range(100):
        d.tick()
        if not d.host.client._writes:
            break
        v.time.sleep(.001)
    assert clock_restarted(d)


def test_a_stale_empty_report_does_not_end_the_watch_on_queued_id_text():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    d.host.data.respond = lambda _: None
    d.host.now += v.ID_INTERVAL_S - v.ID_TEXT_LEAD_S
    d.tick()
    d.tick()
    d.hear('BUFFER 15')
    d.host.now += v.ID_TEXT_LEAD_S
    d.tick()
    assert sum('W9SSJ is still queued' in line for line in d.said) == 1


def test_chat_text_naming_the_station_keeps_the_original_deadline_watched():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    d.host.data.respond = lambda _: None
    d.host.now += 470
    d.do('W9SSJ here, still on')
    d.hear('BUFFER 20', 'PTT ON', 'PTT OFF')
    d.host.now += v.ID_TEXT_LEAD_S
    d.tick()
    assert sum('W9SSJ is still queued' in line for line in d.said) == 1


def test_a_zero_read_before_the_writer_completes_does_not_end_the_watch(request):
    d = linked()
    request.addfinalizer(d.host.client.close)
    d.host.client.enable_writer()
    d.hear('PTT ON', 'PTT OFF')
    d.host.data.respond = lambda _: None
    writer, held = d.host.client.data_writer, []
    poll = writer.poll
    writer.poll = lambda: []
    d.host.now += v.ID_INTERVAL_S - v.ID_TEXT_LEAD_S
    d.tick()
    for _ in range(100):
        held.extend(poll())
        if held:
            break
        v.time.sleep(.001)
    assert held
    d.hear('BUFFER 0')
    writer.poll = lambda: [held.pop()] if held else []
    d.tick()
    assert clock_restarted(d)
    d.hear('BUFFER 15')
    d.host.now += v.ID_TEXT_LEAD_S
    d.tick()
    assert sum('W9SSJ is still queued' in line for line in d.said) == 1


def test_discarding_queued_id_text_takes_its_clock_restart_back():
    d = linked()
    d.hear('PTT ON', 'PTT OFF')
    first = d.session._periodic_from()
    d.host.data.respond = lambda _: None
    d.do('de W9SSJ')
    assert clock_restarted(d)
    d.do('/clear')
    assert d.session._periodic_from() == first


def test_text_still_in_the_writer_when_the_queue_is_discarded_earns_nothing(request):
    d = linked()
    request.addfinalizer(d.host.client.close)
    d.host.client.enable_writer()
    d.hear('PTT ON', 'PTT OFF')
    first = d.session._periodic_from()
    writer, held = d.host.client.data_writer, []
    poll = writer.poll
    writer.poll = lambda: []
    d.session.command('de W9SSJ')
    for _ in range(100):
        held.extend(poll())
        if held:
            break
        v.time.sleep(.001)
    assert held
    d.host.client.send_command('CLEANTXBUFFER')
    writer.poll = lambda: [held.pop()] if held else []
    d.tick()
    assert d.session._periodic_from() == first


def test_a_text_after_an_emptied_queue_gets_a_watch_of_its_own(request):
    d = linked()
    request.addfinalizer(d.host.client.close)
    d.hear('PTT ON', 'PTT OFF')
    d.host.data.respond = lambda _: None
    d.do('W9SSJ first')
    d.host.client.enable_writer()
    writer, held = d.host.client.data_writer, []
    poll = writer.poll
    writer.poll = lambda: []
    d.session.command('W9SSJ second')
    for _ in range(100):
        held.extend(poll())
        if held:
            break
        v.time.sleep(.001)
    # The zero and the completion come in one poll, the zero first.
    calls = []
    writer.poll = lambda: (calls.append(1) or ([held.pop()] if len(calls) == 2 and held else []))
    d.hear('BUFFER 0')
    d.hear('BUFFER 12')
    d.host.now += v.ID_TEXT_LEAD_S
    d.tick()
    assert sum('W9SSJ is still queued' in line for line in d.said) == 1
