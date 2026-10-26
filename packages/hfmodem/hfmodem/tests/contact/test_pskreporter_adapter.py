# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hfmodem.tests.contact.test_varac import Desk
from hfmodem.station.pskreporter import Config, Reporter


def test_received_cq_frequency_not_advertised_destination():
    desk = Desk()
    desk.flow.dial = 14105000
    desk.session._heard('CQFRAME K1ABC-1 500')
    event = desk.kinds('heard')[-1]
    assert event['rf_source'] == 'cq'
    assert event['rx_frequency_hz'] == 14106500
    assert event['received_unix'] > 1700000000
    assert event['dial_hz'] != event['rx_frequency_hz']


def test_passive_cq_received_channel_not_advertised_slot():
    desk = Desk()
    desk.session._heard('CQFRAME K1ABC-1 500', via='slot 11', slot=11)
    event = desk.kinds('heard')[-1]
    assert event['rx_frequency_hz'] == 14107250
    assert event['rf_source'] == 'cq'


def test_status_off_never_send_chat_or_enable_network():
    desk = Desk()
    before = list(desk.texts)
    desk.do('/pskreporter status')
    assert 'PSKReporter off' in desk.said[-1]
    reporter = Reporter(Config('W9SSJ', 'EN61', enabled=True), start_worker=False)
    desk.session.reporter = reporter
    desk.do('/pskreporter status')
    assert 'server receipt unconfirmed' in desk.said[-1]
    desk.do('/pskreporter off')
    assert not reporter.status()['enabled']
    desk.do('/pskreporter on')
    assert 'enable at launch' in desk.said[-1]
    assert desk.texts == before
