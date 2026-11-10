# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest
from hfmodem.ax25 import kiss
from hfmodem.kestrel.broadcast import decode_ui, encode_ui
from hfmodem.kestrel.ui import UIFrame
from hfmodem.kestrel.tx.ui import packetize
from hfmodem.contact.varac import Identification, State, VMail, frame_message
from hfmodem.tests.contact.test_varac import CQ_DONE, FRAME, Desk, Socket


def desk():
    d = Desk(kiss=True)
    d.host.kiss.respond = lambda _: None
    return d


def select(d):
    d.do('/all slot 11')
    assert d.flow.dial == 14105750
    assert d.session.broadcast.selected == d.flow.dial


def wire(d):
    return [payload for port, command, payload in kiss.Parser().feed(
        b''.join(d.host.kiss.sent)) if port == 0 and command == 1] if d.host.kiss else []


def expected(d):
    return len(packetize(UIFrame.from_ax25(wire(d)[-1], command=1))[1])


def bracket(d, count, level=1):
    d.hear('PENDING', 'PTT ON', 'BUSY ON', 'REGISTERED',
           *([f'BITRATE ({level}) 18 bps TX'] * count),
           'PTT OFF', 'CANCELPENDING', 'BUSY OFF')


def observed(d):
    bracket(d, expected(d))


def start(d, directed=False, text='hello everyone'):
    if directed:
        d.hear('BUSY OFF')
        d.session.broadcast.directed('K1ABC')
    else:
        select(d)
        d.session.command('/all ' + text)
    d.tick()
    assert len(wire(d)) == 1
    assert d.session.broadcast.active


def test_explicit_selection_and_one_kiss_write_without_a_delivery_claim():
    d = desk()
    d.do('/all hello')
    assert not wire(d)
    before = list(d.texts)
    start(d)
    assert len(d.host.kiss.sent) == 1
    assert d.host.kiss.sent[0] == kiss.encode(encode_ui('W9SSJ', 'ALL', b'hello everyone'), 1)
    message = decode_ui(wire(d)[0], command=1)
    assert (message.source, message.destination, message.info) == ('W9SSJ', 'ALL', b'hello everyone')
    assert d.texts == before
    assert d.session.broadcast.phase == 'written; no matching transmission observed yet'
    observed(d)
    assert not d.session.broadcast.active
    assert d.session.broadcast.result == 'a host transmission matching this frame followed'
    assert d.session.broadcast.possibly_outstanding == 1
    assert not d.kinds('broadcast_tx')
    assert not any(command.startswith('BROADCAST') for command in d.sent)
    assert d.flow.dial == 14105750
    d.do('/all clear')
    d.do('/all off')
    assert d.flow.dial == 14105000 and d.session.broadcast.selected is None


def test_explicit_calling_frequency_selection_sends_and_restores_monitoring():
    d = desk()
    d.session.set_beacon(15)
    schedule = d.session._beacon_due, d.session._beacon_expires
    d.do('/all cf')
    assert d.flow.dial == 14105000
    assert d.session.broadcast.selected == 14105000
    assert not wire(d) and not d.texts
    assert d.sent[-1] == 'LISTEN OFF'
    d.session.command('/all hello on CF')
    d.tick()
    assert len(wire(d)) == 1
    observed(d)
    assert not d.session.broadcast.active
    assert d.session.broadcast.result == 'a host transmission matching this frame followed'
    d.do('/all clear')
    d.do('/all off')
    assert d.session.broadcast.selected is None and d.flow.dial == 14105000
    assert (d.session._beacon_due, d.session._beacon_expires) == schedule


def test_quit_cancels_admitted_frame_before_finishing():
    d = desk()
    start(d)
    d.session.quit()
    assert 'ABORT' in d.sent and not d.session.broadcast.active
    assert d.session.broadcast.possibly_outstanding == 1


def test_no_kiss_port_has_no_fallback_or_write():
    d = Desk()
    select(d)
    before = list(d.sent)
    d.do('/all hello')
    assert d.sent == before and not d.session.broadcast.active
    assert d.session.broadcast.possibly_outstanding == 0
    assert any('the host has no KISS port; enable KISS in VARA' in line for line in d.said)


@pytest.mark.parametrize('channel', ['slot 11', 'cf'])
@pytest.mark.parametrize('guard', ['linked', 'pending', 'ptt', 'physical', 'busy', 'steps'])
def test_unsafe_state_refuses_without_sending(guard, channel):
    d = desk()
    d.do(f'/all {channel}')
    if guard == 'linked':
        d.host.client.connected = True
    elif guard == 'pending':
        d.host.client.pending = True
    elif guard == 'ptt':
        d.host.client.ptt = True
    elif guard == 'physical':
        d.line = True
    elif guard == 'busy':
        d.host.client.busy = True
    else:
        d.session.steps = iter([None])
    d.session.command('/all hello')
    assert not wire(d)


@pytest.mark.parametrize('directed', [False, True], ids=['all', 'notice'])
@pytest.mark.parametrize('level', [1, 2, 4])
def test_p_console_bracket_releases_held_messages_without_retiring_the_link(directed, level):
    d = desk()
    start(d, directed)
    c = d.host.client
    generation = c.link_generation
    d.hear('PENDING', 'PTT ON', 'BUSY ON', 'REGISTERED')
    d.receive('held during classification')
    assert not c.messages and not any(e.kind == 'message' for e in d.host.events)
    assert c.link_generation == generation and d.session.broadcast.active
    d.hear(*([f'BITRATE ({level}) 18 bps TX'] * expected(d)))
    assert [text for _, text in c.messages] == ['held during classification']
    assert d.session.broadcast.active
    d.hear('PTT OFF', 'CANCELPENDING', 'BUSY OFF')
    assert c.link_generation == generation and not c.pending and not c.connected
    assert not d.session.broadcast.active
    assert d.session.broadcast.possibly_outstanding == 1
    assert not d.kinds('attempt') and not d.kinds('pending_unknown')


@pytest.mark.parametrize('directed', [False, True], ids=['all', 'notice'])
@pytest.mark.parametrize('message', ['held for the incoming link', 'hello\nsecond line', '<VRP>'])
def test_p_console_bracket_incoming_call_is_classified_at_connected(directed, message):
    d = desk()
    start(d, directed)
    c = d.host.client
    c._text(b'40 old partial')
    generation = c.link_generation
    d.hear('PENDING', 'PTT ON', 'REGISTERED')
    d.receive(message)
    d.host.now += 1
    d.hear('PTT OFF')
    assert c.link_generation == generation and d.session.broadcast.active
    assert not any(e.kind == 'message' for e in d.host.events)
    d.host.now += 4.5
    d.hear('CONNECTED W9SSJ K2ABC 500')
    assert c.link_generation == generation + 1
    assert [text for _, text in c.messages] == [message]
    assert not c._messages.pending
    assert any(e.kind == 'message' and e.raw == message
               and e.generation == generation + 1 for e in d.host.events)
    assert c.connected and d.session.peer == 'K2ABC' and d.session.contact is not None
    if message == '<VRP>':
        assert d.texts.count('6 <VRPJ>') == 1
    else:
        assert f'K2ABC: {message.splitlines()[0]}' in d.said
        if '\n' in message:
            assert '       second line' in d.said
    assert not d.session.broadcast.active and d.session.broadcast.selected is None
    assert d.session.broadcast.possibly_outstanding == 1
    assert 'ABORT' not in d.sent and not d.kinds('attempt')
    d.receive('new link')
    assert any(e.kind == 'message' and e.raw == 'new link' for e in d.host.events)


@pytest.mark.parametrize('directed', [False, True], ids=['all', 'notice'])
@pytest.mark.parametrize('ending', ['CANCELPENDING', 'DISCONNECTED'])
def test_p_console_bracket_failed_handshake_or_header_is_unknown(directed, ending):
    d = desk()
    start(d, directed)
    c = d.host.client
    generation = c.link_generation
    d.hear('PENDING', 'PTT ON', 'REGISTERED')
    d.receive('retired')
    d.hear('PTT OFF', ending, 'BUSY OFF')
    assert c.link_generation == generation + 1 and not c.messages
    assert not d.session.broadcast.active and d.session.broadcast.possibly_outstanding == 1
    assert any('the host keyed briefly without a link (failed call or failed transmission)' in x
               for x in d.said)
    assert not d.kinds('attempt') and not any('called you' in x for x in d.said)
    assert 'ABORT' not in d.sent


@pytest.mark.parametrize('directed', [False, True], ids=['all', 'notice'])
def test_named_incoming_call_keeps_data_held_until_session_is_connected(directed):
    d = desk()
    start(d, directed)
    c = d.host.client
    generation = c.link_generation
    d.hear('PENDING', 'PTT ON')
    d.receive('hello\nsecond line')
    d.session.caller = lambda: 'K2ABC'
    d.tick()
    assert c.link_generation == generation + 1 and c.link_pending
    assert not c.messages and not any(e.kind == 'message' for e in d.host.events)
    d.receive('<VRP>')
    assert not c.messages and not d.texts
    d.hear('PTT OFF', 'CONNECTED W9SSJ K2ABC 500')
    assert c.link_generation == generation + 1 and c.connected
    assert [text for _, text in c.messages] == ['hello\nsecond line', '<VRP>']
    assert 'K2ABC: hello' in d.said and '       second line' in d.said
    assert d.texts.count('6 <VRPJ>') == 1


@pytest.mark.parametrize('directed', [False, True], ids=['all', 'notice'])
def test_failed_named_call_does_not_classify_later_kiss_brackets(monkeypatch, directed):
    import numpy as np
    from hfmodem.kestrel.tx import broadcast
    from hfmodem.kestrel.host.kiss_tx import KissTxQueue
    from hfmodem.tests.kestrel.test_cq import session_mock
    monkeypatch.setattr(broadcast, 'synth_broadcast', lambda _: np.zeros(8))
    host, _, modem = session_mock()
    d = desk()
    host._send_cmd = lambda line: d.host.lines(line)
    d.session.caller = lambda: getattr(modem, 'attempt_caller', None)
    modem.station.io.pending()
    modem.station.caller = 'K1ABC'
    modem.abort()
    modem.poll()
    d.tick()
    assert modem.attempt_caller == 'K1ABC'
    assert [event['caller'] for event in d.kinds('attempt')] == ['K1ABC']
    generation = d.host.client.link_generation
    queue = modem.kiss_tx_queue = KissTxQueue(clock=lambda: d.host.now)
    modem._transport.tx = lambda _: None
    start(d, directed)
    # Include another client's frame ahead of ours, with a different size.
    for raw in (encode_ui('K9XYZ', 'ALL', b'x' * 400), wire(d)[-1]):
        assert queue.broadcast(1, raw)
        d.host.now += 3
        modem.service_kiss_queue()
        assert modem.attempt_caller is None
        count = len(packetize(UIFrame.from_ax25(raw, command=1))[1])
        bracket(d, count)
        assert d.host.client.link_generation == generation
        assert d.session.broadcast.active == (count != expected(d))
    assert len(d.kinds('attempt')) == 1 and not d.kinds('pending_unknown')
    assert not d.session.broadcast.active and len(wire(d)) == 1
    assert d.session.broadcast.possibly_outstanding == 1


def test_p_console_bracket_other_clients_before_first_all_and_after_last():
    d = desk()
    c = d.host.client
    generation = c.link_generation
    for _ in range(2):
        bracket(d, 3, 4)
        assert c.link_generation == generation and not c.pending
        assert not d.kinds('attempt') and not d.kinds('pending_unknown')
        if not wire(d):
            start(d)
            observed(d)
            d.do('/all clear')
            d.do('/all off')
    assert len(wire(d)) == 1 and d.session.broadcast.possibly_outstanding == 0


@pytest.mark.parametrize('directed', [False, True], ids=['all', 'notice'])
def test_two_unrelated_brackets_are_counted_separately_while_ours_is_queued(directed):
    d = desk()
    start(d, directed)
    n = expected(d)
    generation = d.host.client.link_generation
    for count in (n + 1, n + 2):
        bracket(d, count, 4)
        assert d.session.broadcast.active and d.session.broadcast.possibly_outstanding == 1
    d.hear('BUSY ON', 'BUSY OFF')
    d.host.now += 3600
    d.tick()
    assert d.session.broadcast.phase == 'written; no matching transmission observed yet'
    assert len(d.host.kiss.sent) == 1 and d.host.client.link_generation == generation
    bracket(d, n)
    assert not d.session.broadcast.active and d.session.broadcast.possibly_outstanding == 1


@pytest.mark.parametrize('command', ['/all off', '/all slot 12', '/all cf', '/band 40',
                                     '/auto on', '/auto off', '/cq 1', '/cq 1 repeat',
                                     '/call K1ABC 2', '/call K1ABC cf', '/answer 1',
                                     '/follow 2', '/follow cf', '/follow 14107.25', '/follow off'])
def test_stop_then_retune_warns_once_and_acknowledges(command):
    d = desk()
    if command.startswith('/follow'):
        from hfmodem.tests.contact.test_slots import Watch
        d.session.slots = Watch()
        # Force the selected target outside the monitor's current passband.
        d.session.slots.rows = []
        d.session._follow_tuned = command == '/follow off'
    if command == '/answer 1':
        d.hear('CQFRAME K1ABC-2 500')
    start(d)
    d.do('/stop')
    assert not d.session.broadcast.active and 'ABORT' not in d.sent
    assert any('the host may still key this frame' in x for x in d.said)
    dial = d.flow.dial
    before = d.tunes[:], d.sent[:], d.session.broadcast.selected
    d.session.command(command)
    d.session.command(command)
    warning = '1 frames written to the host may still key on the new frequency'
    assert sum(warning in x for x in d.said) == 1
    assert d.flow.dial == dial and d.session.broadcast.possibly_outstanding == 1
    assert (d.tunes, d.sent, d.session.broadcast.selected) == before
    d.session.command('/all yes')
    assert d.session.broadcast.possibly_outstanding == 0
    assert d.tunes != before[0] or d.session.steps is not None
    if d.session.steps is not None:
        d.session.command('/stop')
        d.drive()


@pytest.mark.parametrize('command', ['/auto on', '/cq 1', '/call K1ABC cf', '/follow 2'])
def test_declining_existing_command_keeps_broadcast_slot_and_warning(command):
    d = desk()
    if command.startswith('/follow'):
        from hfmodem.tests.contact.test_slots import Watch
        d.session.slots = Watch()
    start(d)
    d.do('/stop')
    before = d.tunes[:], d.sent[:], d.session.broadcast.selected
    d.do(command)
    d.do('/all no')
    assert (d.tunes, d.sent, d.session.broadcast.selected) == before
    assert d.session.broadcast.possibly_outstanding == 1
    d.do(command)
    assert sum('may still key on the new frequency' in line for line in d.said) == 2


@pytest.mark.parametrize('saved_slot', [False, True])
def test_mail_collection_warns_once_and_waits_for_acknowledgement(saved_slot):
    d = desk()
    target = 14106500 if saved_slot else 14105000
    if saved_slot:
        from hfmodem.tests.contact.test_mail_collection import notice
        notice(d, source='K1ABC', dial=target)
    d.host.replies['CONNECT W9SSJ K1ABC'] = ['OK', 'CONNECTED W9SSJ K1ABC 500']
    start(d)
    d.do('/stop')
    before = d.tunes[:], d.sent[:], d.session.broadcast.selected
    d.do('/vmail collect k1abc')
    d.do('/vmail collect K1ABC')
    assert (d.tunes, d.sent, d.session.broadcast.selected) == before
    assert d.flow.dial == 14105750 and d.session.steps is None
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command == '/vmail collect K1ABC'
    assert sum('may still key on the new frequency' in line for line in d.said) == 1
    d.session.command('/all yes')
    d.drive(until=lambda: d.host.client.connected)
    assert d.flow.dial == target and d.session.broadcast.selected is None
    assert 'CONNECT W9SSJ K1ABC' in d.sent and d.host.client.connected
    assert d.session.broadcast.possibly_outstanding == 0
    assert d.session.broadcast._retune_command is None
    assert not d.kinds('error')


@pytest.mark.parametrize('saved_slot', [False, True])
def test_mail_collection_refusal_preserves_slot_and_outstanding_count(saved_slot):
    d = desk()
    if saved_slot:
        from hfmodem.tests.contact.test_mail_collection import notice
        notice(d, source='K1ABC', dial=14106500)
    start(d)
    d.do('/stop')
    notices = d.session.mail_collection._load()
    before = d.tunes[:], d.sent[:], d.session.broadcast.selected
    d.do('/vmail collect K1ABC')
    d.do('/all no')
    assert (d.tunes, d.sent, d.session.broadcast.selected) == before
    assert d.flow.dial == 14105750 and d.session.steps is None
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command is None
    assert d.session.mail_collection._load() == notices
    d.do('/vmail collect K1ABC')
    assert sum('may still key on the new frequency' in line for line in d.said) == 2
    assert (d.tunes, d.sent, d.session.broadcast.selected) == before
    assert not d.kinds('error')


def test_automatic_band_hop_warns_and_preserves_outstanding_notice_count():
    d = desk()
    start(d, directed=True)
    d.do('/stop')
    d.session.hopping.readiness.update({40: 'ready', 20: 'ready'})
    d.do('/beacon hop 40 20')
    d.tick()
    assert d.flow.band == 40 and d.flow.dial == 7105000
    assert d.flow.state is State.MONITORING
    assert d.session.hopping.pending == 40 and d.session.hopping.paused is None
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command is None
    assert sum('1 frames written to the host may still key on the new frequency' in line
               for line in d.said) == 1
    assert not any('Proceed?' in line for line in d.said)
    assert len(wire(d)) == 1 and not d.kinds('error')
    d.do('/all clear')
    assert d.session.broadcast.possibly_outstanding == 0


def completed_broadcast_with_keyer(*, hopping=False, directed=False):
    d = desk()
    if hopping:
        d.session.hopping.readiness.update({40: 'ready', 20: 'ready'})
        d.do('/beacon hop 40 20')
        d.session.hopping.due = d.session._beacon_due = float('inf')
    start(d, directed=hopping or directed)
    observed(d)
    # The frame named us; a transmission that does not owes the next ID.
    d.hear('PTT ON', 'PTT OFF')
    assert not d.session.broadcast.active and d.session.steps is None
    assert d.session.owed and not d.ids
    assert d.session.broadcast.possibly_outstanding == 1
    finished = [False]
    def identify(bandwidth, after_peer):
        d.identify(bandwidth, after_peer)
        d.line = True
        try:
            while not finished[0]:
                yield .1
        finally:
            d.line = False
        return Identification(d.line is False, d.host.client.transmission_count)
    d.session.keyer = identify
    return d, finished


def request_completed_retune(d, command=None):
    dial, tunes = d.flow.dial, d.tunes[:]
    if command is None:
        d.session.hopping.due = d.host.client.clock()
        d.tick()
    else:
        d.session.command(command)
    assert not d.session._identifying and not d.line and not d.ids
    return dial, tunes


def test_completed_all_band_acceptance_preserves_retune_prompt():
    d, _ = completed_broadcast_with_keyer()
    dial, tunes = request_completed_retune(d, '/band 40')
    assert dial == 14105750 and d.flow.dial == dial and d.tunes == tunes
    assert sum('Proceed?' in line for line in d.said) == 1
    assert d.session.broadcast.possibly_outstanding == 1
    d.do('/all yes')
    d.tick()
    assert d.flow.band == 40 and d.flow.dial == 7105000
    assert d.session.broadcast.possibly_outstanding == 0
    assert not d.ids
    assert not d.kinds('error')


def test_completed_all_band_refusal_preserves_later_acknowledged_retune():
    d, _ = completed_broadcast_with_keyer()
    dial, tunes = request_completed_retune(d, '/band 40')
    d.do('/all no')
    d.tick()
    assert dial == 14105750 and d.flow.dial == dial and d.tunes == tunes
    assert d.session.broadcast.possibly_outstanding == 1
    assert not d.ids
    d.do('/band 40')
    assert d.flow.dial == dial and d.tunes == tunes
    assert sum('Proceed?' in line for line in d.said) == 2
    d.do('/all yes')
    d.tick()
    assert d.flow.band == 40 and d.flow.dial == 7105000
    assert d.session.broadcast.possibly_outstanding == 0
    assert not d.ids
    assert not d.kinds('error')


def test_completed_all_slot_change_preserves_retune_prompt():
    d, _ = completed_broadcast_with_keyer()
    dial, tunes = request_completed_retune(d, '/all slot 12')
    assert dial == 14105750 and d.flow.dial == dial and d.tunes == tunes
    assert sum('Proceed?' in line for line in d.said) == 1
    assert d.session.broadcast.possibly_outstanding == 1
    d.do('/all yes')
    d.tick()
    assert d.flow.dial == d.session.broadcast.selected == 14106500
    assert d.session.broadcast.possibly_outstanding == 0
    assert not d.ids
    assert not d.kinds('error')


def test_completed_notice_automatic_hop_preserves_band_change():
    d, _ = completed_broadcast_with_keyer(hopping=True)
    dial, _ = request_completed_retune(d)
    d.tick()
    assert dial == 14105000 and d.flow.band == 40 and d.flow.dial == 7105000
    assert d.flow.state is State.MONITORING
    assert d.session.hopping.pending == 40 and d.session.hopping.paused is None
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command is None
    assert not any('Proceed?' in line for line in d.said)
    assert sum('may still key on the new frequency' in line for line in d.said) == 1
    assert not d.ids
    assert not d.kinds('error')


def acknowledge_completed_retune(d, dial, tunes, command, accepted):
    assert d.flow.dial == dial and d.tunes == tunes
    assert sum('Proceed?' in line for line in d.said) == 1
    assert d.session.broadcast.possibly_outstanding == 1
    if not accepted:
        d.do('/all no')
        assert d.flow.dial == dial and d.tunes == tunes
        assert d.session.broadcast.possibly_outstanding == 1
        d.session.command(command)
        assert d.flow.dial == dial and d.tunes == tunes
        assert sum('Proceed?' in line for line in d.said) == 2
    d.session.command('/all yes')
    assert d.session.broadcast.possibly_outstanding == 0
    assert not d.ids
    assert not d.kinds('error')


@pytest.mark.parametrize('accepted', [True, False], ids=['accept', 'refuse-then-accept'])
def test_completed_all_call_preserves_retune_acknowledgement(accepted):
    d, _ = completed_broadcast_with_keyer()
    d.host.replies['CONNECT W9SSJ K1ABC'] = ['OK', 'CONNECTED W9SSJ K1ABC 500']
    command = '/call K1ABC 2'
    dial, tunes = request_completed_retune(d, command)
    acknowledge_completed_retune(d, dial, tunes, command, accepted)
    d.drive(until=lambda: d.host.client.connected)
    assert d.flow.dial == 14103500 and d.host.client.connected
    assert not d.ids
    assert not d.kinds('error')


@pytest.mark.parametrize('accepted', [True, False], ids=['accept', 'refuse-then-accept'])
@pytest.mark.parametrize('saved_slot', [False, True], ids=['cf', 'saved-slot'])
def test_completed_all_mail_collection_preserves_retune_acknowledgement(accepted, saved_slot):
    d, _ = completed_broadcast_with_keyer()
    target = 14106500 if saved_slot else 14105000
    if saved_slot:
        from hfmodem.tests.contact.test_mail_collection import notice
        notice(d, source='K1ABC', dial=target)
    notices = d.session.mail_collection._load()
    d.host.replies['CONNECT W9SSJ K1ABC'] = ['OK', 'CONNECTED W9SSJ K1ABC 500']
    command = '/vmail collect K1ABC'
    dial, tunes = request_completed_retune(d, command)
    acknowledge_completed_retune(d, dial, tunes, command, accepted)
    d.drive(until=lambda: d.host.client.connected)
    assert d.flow.dial == target and d.host.client.connected
    assert d.session.mail_collection._load() == notices
    assert not d.ids
    assert not d.kinds('error')


def test_completed_all_answer_preserves_retune_acknowledgement():
    d, _ = completed_broadcast_with_keyer()
    d.hear('CQFRAME K1ABC-2 500')
    d.host.replies['CONNECT W9SSJ K1ABC'] = ['OK', 'CONNECTED W9SSJ K1ABC 500']
    command = '/answer 1'
    dial, tunes = request_completed_retune(d, command)
    acknowledge_completed_retune(d, dial, tunes, command, True)
    d.drive(until=lambda: d.host.client.connected)
    assert d.flow.dial == 14103500 and d.host.client.connected
    assert not d.ids
    assert not d.kinds('error')


def test_completed_all_cq_preserves_retune_acknowledgement():
    d, _ = completed_broadcast_with_keyer()
    command = '/cq 2'
    dial, tunes = request_completed_retune(d, command)
    acknowledge_completed_retune(d, dial, tunes, command, True)
    assert d.flow.dial == 14103500 and d.flow.state is State.SNIFFING
    assert not d.ids


def test_completed_all_follow_preserves_retune_acknowledgement():
    from hfmodem.tests.contact.test_slots import Watch
    d, _ = completed_broadcast_with_keyer()
    d.session.slots = Watch()
    d.session.slots.rows = []
    command = '/follow 2'
    dial, tunes = request_completed_retune(d, command)
    acknowledge_completed_retune(d, dial, tunes, command, True)
    assert d.flow.dial == 14103500 and d.session.following == 2
    assert not d.ids


def test_completed_all_auto_on_preserves_retune_acknowledgement():
    d, _ = completed_broadcast_with_keyer()
    d.session.auto = False
    command = '/auto on'
    dial, tunes = request_completed_retune(d, command)
    assert not d.session.auto
    acknowledge_completed_retune(d, dial, tunes, command, True)
    assert d.flow.dial == 14105000 and d.flow.state is State.MONITORING
    assert d.session.auto and not d.ids


def notice_link_with_yielding_identification():
    d, finished = completed_broadcast_with_keyer(directed=True)
    d.hear('CONNECTED W9SSJ K2ABC 500', 'SN 2')
    assert d.session.contact is not None and d.flow.dial == 14105000
    assert d.session.owed and not d.ids
    return d, finished


def test_completed_notice_automatic_qsy_moves_without_identifying_inside_link():
    d, _ = notice_link_with_yielding_identification()
    d.receive('<QSYF>000014103500</QSYF>')
    assert d.flow.dial == 14103500 and d.host.client.connected
    assert not d.ids and not d.session._identifying
    assert d.session.broadcast.possibly_outstanding == 1
    assert not any('Proceed?' in line for line in d.said)
    assert sum('may still key on the new frequency' in line for line in d.said) == 1
    d.receive('<TL>')
    d.receive('de K2ABC <R+02>')
    assert d.flow.state is State.CONNECTED and d.session.steps is None
    assert not d.kinds('error')


@pytest.mark.parametrize('keyed', [False, True], ids=['unkeyed', 'still-keyed'])
def test_handshake_ptt_does_not_identify_or_delay_link_retune(keyed):
    d = desk()
    d.hear('PENDING', 'PTT ON', *([] if keyed else ['PTT OFF']),
           'CONNECTED W9SSJ K2ABC 500', 'SN 2')
    generation = d.host.client.link_generation
    requests = []
    def owner(target, link):
        requests.append((target, link, d.host.client.ptt))
        yield .1
    d.flow.received_retune_steps = owner
    d.receive('<QSYF>000014103500</QSYF>')
    assert requests == [(14103500, generation, keyed)]
    assert d.session.owed
    assert not d.ids and not d.session._identifying
    assert d.host.client.connected and d.session.contact is not None
    assert not d.kinds('error')
    d.session.steps.close()
    d.session.steps = None


def test_handshake_ptt_does_not_make_an_earlier_notice_identify_before_qsy():
    d, _ = completed_broadcast_with_keyer(directed=True)
    d.hear('PENDING', 'PTT ON', 'PTT OFF', 'CONNECTED W9SSJ K2ABC 500', 'SN 2')
    d.receive('<QSYF>000014103500</QSYF>')
    assert d.flow.dial == 14103500 and d.host.client.connected
    assert not d.ids and d.session.owed
    assert not d.kinds('error')


def test_the_due_id_inside_a_link_is_text_and_messages_go_on_being_answered():
    d, _ = notice_link_with_yielding_identification()
    d.do('/all clear')
    d.host.now += 600
    d.tick()
    assert not d.ids and not d.session._identifying and not d.line
    assert d.texts[-1] == frame_message("de W9SSJ <R+02>").decode()
    d.receive(VMail('During identification', 'Please keep this message').wire())
    d.receive('<VRP>')
    assert len(d.mailbox.entries()) == 1
    assert d.texts.count('5 <SMR>') == 1 and d.texts.count('6 <VRPJ>') == 1
    assert d.host.client.connected and not d.kinds('error')


def test_quit_during_periodic_id_propagates_identification_cancellation_failure(monkeypatch):
    # A due periodic ID exercises keyer cancellation on an idle channel.
    import sys
    d, _ = completed_broadcast_with_keyer()
    failure = TimeoutError('identification cancellation timed out')
    cancelled, unraisable = [], []
    monkeypatch.setattr(sys, 'unraisablehook', unraisable.append)
    def identify(bandwidth, after_peer):
        d.identify(bandwidth, after_peer)
        d.line = True
        try:
            yield .1
        finally:
            d.line = False
            cancelled.append(True)
            raise failure
    d.session.keyer = identify
    d.do('/all clear')
    d.host.now += 600
    d.tick()
    assert d.session._identifying and d.line
    assert len(d.ids) == 1 and d.session.steps is not None
    with pytest.raises(TimeoutError, match='identification cancellation') as caught:
        d.session.quit()
    assert caught.value is failure
    assert cancelled == [True] and not unraisable
    assert not d.session._identifying and not d.line
    assert d.session.steps is None and d.flow.band == 20
    assert d.session.broadcast._retune_command is None


@pytest.mark.parametrize('outgoing', ['repeat', 'beacon'])
@pytest.mark.parametrize('outstanding', [0, 1], ids=['clear', 'confirmation-required'])
def test_call_replaces_outgoing_after_confirmation_without_identifying(outgoing, outstanding):
    d = desk()
    d.host.replies['CONNECT W9SSJ K1ABC'] = ['OK', 'CONNECTED W9SSJ K1ABC 500', 'SN 2']
    if outgoing == 'repeat':
        d.host.replies[FRAME] = CQ_DONE
        d.session.command('/cq 1 repeat')
        d.drive(until=lambda: d.sent.count(FRAME) == 1 and not d.host.client.pending)
    else:
        d.host.replies['CQFRAME W9SSJ-9 500'] = ['OK', 'PENDING', 'PTT ON']
        d.session.intent = 'beacon'
        d.session._run(d.session._beaconing(once=True))
        d.host.lines('PTT OFF', 'CANCELPENDING')
        d.host.client.poll(0)
    assert d.session.steps is not None and not d.ids
    assert not d.host.client.ptt and not d.host.client.pending
    d.session.broadcast.possibly_outstanding = outstanding
    confirmation = []
    confirm = d.session.broadcast._confirm_retune
    def confirmed(command):
        confirmation.append((command, len(d.ids), d.session.repeat, len(d.sent)))
        return confirm(command)
    d.session.broadcast._confirm_retune = confirmed
    d.session.command('/call K1ABC 2')
    assert not d.kinds('error')
    if outstanding:
        assert d.session.broadcast._retune_command == '/call K1ABC 2'
        assert not d.host.client.connected and not d.session.repeat
        assert not d.ids
        assert sum('Proceed?' in line for line in d.said) == 1
        d.session.command('/all yes')
    d.drive()
    assert not d.ids
    assert len(confirmation) == 1 + outstanding
    assert all(row[:3] == ('/call K1ABC 2', 0, False) for row in confirmation)
    assert confirmation[0][3] <= d.sent.index('CONNECT W9SSJ K1ABC')
    assert not d.session.repeat and d.session.intent == 'call'
    assert d.host.client.connected and d.session.peer == 'K1ABC'
    assert d.flow.dial == 14103500 and d.flow.state is State.CONNECTED
    if outgoing == 'repeat':
        assert d.sent.count(FRAME) == 1
    assert not d.kinds('error')


@pytest.mark.parametrize('accepted', [True, False], ids=['accept', 'refuse-then-accept'])
def test_completed_notice_operator_qsy_preserves_retune_acknowledgement(accepted):
    d, _ = notice_link_with_yielding_identification()
    command = '/qsy 2'
    dial, tunes = request_completed_retune(d, command)
    acknowledge_completed_retune(d, dial, tunes, command, accepted)
    d.receive('<Q>')
    d.drive(until=lambda: d.flow.dial == 14103500)
    assert d.host.client.connected and d.flow.dial == 14103500
    d.receive('<QSYR>')
    assert d.flow.state is State.CONNECTED and d.session.steps is None
    assert not d.ids and not d.kinds('error')


def test_automatic_hop_and_later_peer_return_warn_without_prompting_or_pre_hop_id():
    d, finished = completed_broadcast_with_keyer(hopping=True)
    finished[0] = True
    d.session.hopping.due = d.host.client.clock()
    d.tick()
    assert not d.ids and not d.session._identifying
    assert d.flow.band == 40 and d.flow.dial == 7105000
    assert not d.session._operator_return_pending
    d.hear('CONNECTED W9SSJ K2ABC 500', 'SN 2')
    d.receive('<QSYF>000007103500</QSYF>')
    d.receive('<TL>')
    d.receive('de K2ABC <R+02>')
    assert d.flow.dial == 7103500 and d.session.steps is None
    d.hear('PTT ON', 'PTT OFF', 'DISCONNECTED')
    d.drive()
    d.tick()
    assert d.flow.dial == 7105000 and d.flow.state is State.MONITORING
    assert [ident['dial'] for ident in d.ids] == [7103500]
    assert not d.session._operator_return_pending
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command is None
    assert not any('Proceed?' in line for line in d.said)
    assert not d.kinds('error')




def test_completed_notice_direct_tune_warns_without_identifying():
    d, _ = completed_broadcast_with_keyer(directed=True)
    d.flow.tune(14103500)
    assert d.flow.dial == 14103500 and not d.line and not d.ids
    assert d.session.broadcast.possibly_outstanding == 1
    assert sum('may still key on the new frequency' in line for line in d.said) == 1
    assert not any('Proceed?' in line for line in d.said)
    assert not d.kinds('error')


def test_all_off_refusal_does_not_prompt_on_later_peer_return_or_identify_early():
    d, finished = completed_broadcast_with_keyer()
    finished[0] = True
    dial, tunes = d.flow.dial, d.tunes[:]
    d.session.command('/all off')
    assert not d.ids and not d.session._identifying
    d.do('/all no')
    assert d.flow.dial == dial and d.tunes == tunes
    assert d.session.broadcast.possibly_outstanding == 1
    assert sum('Proceed?' in line for line in d.said) == 1
    d.hear('CONNECTED W9SSJ K2ABC 500', 'PTT ON', 'PTT OFF', 'DISCONNECTED')
    d.drive()
    d.tick()
    assert d.flow.dial == 14105000 and d.flow.state is State.MONITORING
    assert [ident['dial'] for ident in d.ids] == [dial]
    assert not d.session._operator_return_pending
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command is None
    assert sum('Proceed?' in line for line in d.said) == 1
    assert not d.kinds('error')


def test_completed_notice_retune_preserves_an_active_operation():
    d, _ = completed_broadcast_with_keyer(directed=True)
    active = iter([.1, .1])
    d.session.steps = active
    dial, tunes = d.flow.dial, d.tunes[:]
    d.session.command('/call K1ABC 2')
    assert d.session.steps is active
    assert d.flow.dial == dial and d.tunes == tunes
    assert not d.ids and d.session.owed
    assert not d.session._identifying and not d.line
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command is None
    assert any('cannot retune during another operation' in row['text'] for row in d.kinds('error'))


def test_automatic_monitor_return_warns_and_preserves_outstanding_count():
    d = desk()
    start(d)
    d.do('/stop')
    d.session.monitor()
    assert d.flow.dial == 14105000 and d.session.broadcast.selected is None
    assert d.flow.state is State.MONITORING
    assert d.session.broadcast.possibly_outstanding == 1
    assert any('1 frames written to the host may still key on the new frequency' in line
               for line in d.said)
    assert d.session.broadcast._retune_command is None
    d.do('/all clear')
    assert d.session.broadcast.possibly_outstanding == 0


def linked_after_notice():
    d = desk()
    start(d, directed=True)
    d.do('/stop')
    d.hear('CONNECTED W9SSJ K2ABC 500', 'SN 2')
    d.receive('<QSYF>000014103500</QSYF>')
    d.receive('<TL>')
    d.receive('de K2ABC <R+02>')
    assert d.session.steps is None and d.flow.state is State.CONNECTED
    assert d.flow.dial == 14103500
    return d


def check_operator_link_return(d, command, accepted):
    before = d.tunes[:]
    warnings = sum('Proceed?' in line for line in d.said)
    closing = d.session._closing
    d.session.command(command)
    if command in ('/stop', '') and not closing:
        assert 'DISCONNECT' in d.sent and 'ABORT' in d.sent
        assert sum('Proceed?' in line for line in d.said) == warnings
    d.drive()
    d.tick()
    assert not d.host.client.connected and d.session.contact is None
    assert d.tunes == before and d.flow.dial == 14103500
    assert d.session.broadcast.possibly_outstanding == 1
    assert sum('Proceed?' in line for line in d.said) == warnings + 1
    for _ in range(3):
        d.tick()
    assert d.tunes == before
    assert sum('Proceed?' in line for line in d.said) == warnings + 1
    d.do('/all yes' if accepted else '/all no')
    for _ in range(3):
        d.tick()
    assert d.session.broadcast._retune_command is None
    if accepted:
        assert d.flow.dial == 14105000 and d.flow.state is State.MONITORING
        assert d.session.broadcast.possibly_outstanding == 0
    else:
        assert d.tunes == before and d.flow.dial == 14103500
        assert d.session.broadcast.possibly_outstanding == 1
    assert sum('Proceed?' in line for line in d.said) == warnings + 1
    assert not d.kinds('error')


def test_bye_return_waits_for_retune_acknowledgement():
    check_operator_link_return(linked_after_notice(), '/bye custom farewell', True)


def test_bye_return_refusal_holds_dial_and_outstanding_count():
    check_operator_link_return(linked_after_notice(), '/bye', False)


def test_stop_disconnects_before_retune_acknowledgement():
    check_operator_link_return(linked_after_notice(), '/stop', True)


def test_stop_return_refusal_holds_dial_and_outstanding_count():
    check_operator_link_return(linked_after_notice(), '/stop', False)


def check_keyed_operator_link_return(command, accepted):
    d = linked_after_notice()
    d.hear('PTT ON', 'PTT OFF')
    assert d.session.owed and not d.ids
    finished = [False]
    def identify(bandwidth, after_peer):
        d.identify(bandwidth, after_peer)
        while not finished[0]:
            yield .1
        return Identification(d.line is False, d.host.client.transmission_count)
    d.session.keyer = identify
    before = d.tunes[:]
    d.session.command(command)
    d.drive(until=lambda: d.session._identifying)
    assert not d.host.client.connected and d.session.contact is None
    assert d.session._identifying and d.session._operator_return_pending
    assert d.session.owed  # Identification credit waits for keyer completion.
    assert [ident['dial'] for ident in d.ids] == [14103500]
    assert not d.ids[0]['ptt']
    for _ in range(3):
        d.tick()
    assert d.tunes == before and d.flow.dial == 14103500
    assert d.session._operator_return_pending
    assert d.session.broadcast._retune_command is None
    assert not any('Proceed?' in line for line in d.said)
    finished[0] = True
    d.drive()
    d.tick()
    assert not d.session._identifying and not d.session._operator_return_pending
    assert d.tunes == before and d.flow.dial == 14103500
    assert d.session.broadcast.possibly_outstanding == 1
    assert sum('Proceed?' in line for line in d.said) == 1
    d.do('/all yes' if accepted else '/all no')
    if accepted:
        assert d.flow.dial == 14105000 and d.flow.state is State.MONITORING
    else:
        assert d.tunes == before and d.flow.dial == 14103500
        assert d.session.broadcast.possibly_outstanding == 1
        assert not d.session.owed
        d.do('/band 40')
        assert d.tunes == before and d.flow.dial == 14103500
        assert d.session.broadcast.possibly_outstanding == 1
        assert sum('Proceed?' in line for line in d.said) == 2
        d.do('/all yes')
        assert d.flow.dial == 7105000 and d.flow.band == 40
    for _ in range(3):
        d.tick()
    assert d.session.broadcast._retune_command is None
    assert d.session.broadcast.possibly_outstanding == 0
    assert not d.session.owed
    assert [ident['dial'] for ident in d.ids] == [14103500]
    assert sum('Proceed?' in line for line in d.said) == (1 if accepted else 2)
    assert not d.kinds('error')


def test_keyed_bye_identifies_before_retune_acknowledgement():
    check_keyed_operator_link_return('/bye', True)


def test_keyed_bye_refusal_identifies_before_later_acknowledged_retune():
    check_keyed_operator_link_return('/bye', False)


def test_keyed_stop_identifies_before_retune_acknowledgement():
    check_keyed_operator_link_return('/stop', True)


def test_keyed_stop_refusal_identifies_before_later_acknowledged_retune():
    check_keyed_operator_link_return('/stop', False)


@pytest.mark.parametrize('accepted', [False, True])
def test_blank_line_link_stop_requires_retune_acknowledgement(accepted):
    d = linked_after_notice()
    def waiting():
        while True:
            yield .1
    d.session._run(waiting())
    check_operator_link_return(d, '', accepted)


@pytest.mark.parametrize('accepted', [False, True])
def test_stop_during_automatic_close_requires_retune_acknowledgement(accepted):
    d = linked_after_notice()
    d.session._run(d.session._leaving())
    assert d.session._closing and d.session.steps is not None
    check_operator_link_return(d, '/stop', accepted)


def yielding_final_identification_after_notice():
    d = linked_after_notice()
    finished = [False]
    def identify(bandwidth, after_peer):
        d.identify(bandwidth, after_peer)
        while not finished[0]:
            yield .1
        return Identification(d.line is False, d.host.client.transmission_count)
    d.session.keyer = identify
    d.hear('PTT ON', 'PTT OFF', 'DISCONNECTED')
    assert not d.host.client.connected and d.session.contact is None
    assert d.ids and d.session._identifying and d.session._closing
    assert d.session.steps is not None and d.flow.dial == 14103500
    assert not d.session._operator_return_pending
    assert not any('Proceed?' in line for line in d.said)
    return d, finished


def test_stop_during_final_identification_requires_retune_acknowledgement():
    d, finished = yielding_final_identification_after_notice()
    finished[0] = True
    check_operator_link_return(d, '/stop', True)
    assert not d.session._operator_return_pending and not d.session._identifying


def test_stop_during_final_identification_refusal_holds_dial_and_outstanding_count():
    d, finished = yielding_final_identification_after_notice()
    finished[0] = True
    check_operator_link_return(d, '/stop', False)
    assert not d.session._operator_return_pending and not d.session._identifying


def test_automatic_return_after_final_identification_stop_warns_without_acknowledgement():
    d, finished = yielding_final_identification_after_notice()
    finished[0] = True
    check_operator_link_return(d, '/stop', False)
    before = len(d.said)
    d.hear('CONNECTED W9SSJ K2ABC 500', 'PTT ON', 'PTT OFF', 'DISCONNECTED')
    d.drive()
    d.tick()
    assert d.flow.dial == 14105000 and d.flow.state is State.MONITORING
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command is None
    assert not d.session._operator_return_pending and not d.session._identifying
    assert any('1 frames written to the host may still key on the new frequency' in line
               for line in d.said[before:])
    assert not any('Proceed?' in line for line in d.said[before:])
    assert not d.kinds('error')


@pytest.mark.parametrize('accepted', [False, True])
@pytest.mark.parametrize('command', ['/stop', '/bye'])
def test_file_capture_exit_requires_retune_acknowledgement(command, accepted):
    d = linked_after_notice()
    d.session.files.opaque = True
    d.session.files.generation = d.host.client.link_generation
    d.session.files.dial = d.flow.dial
    assert d.session.files.active
    check_operator_link_return(d, command, accepted)


def test_peer_link_exit_warns_and_returns_without_acknowledgement():
    d = linked_after_notice()
    d.hear('DISCONNECTED')
    assert d.flow.dial == 14105000 and d.flow.state is State.MONITORING
    assert d.session.broadcast.possibly_outstanding == 1
    assert d.session.broadcast._retune_command is None
    assert not any('Proceed?' in line for line in d.said)
    assert not d.kinds('error')


def test_linked_qsy_also_requires_acknowledgement_of_an_earlier_kiss_write():
    d = desk()
    start(d, directed=True)
    d.do('/stop')
    d.hear('CONNECTED W9SSJ K2ABC 500')
    before = d.tunes[:], d.texts[:]
    d.do('/qsy 2')
    assert (d.tunes, d.texts) == before and d.session.steps is None
    assert d.session.broadcast.possibly_outstanding == 1
    d.session.command('/all yes')
    assert d.session.steps is not None and d.session.broadcast.possibly_outstanding == 0
    d.do('/stop')
    assert d.host.client.connected and d.tunes == before[0]


def test_received_qsy_warns_and_proceeds_without_acknowledging_an_earlier_kiss_write():
    d = desk()
    start(d, directed=True)
    d.do('/stop')
    d.hear('CONNECTED W9SSJ K2ABC 500')
    d.receive('<QSYF>000014103500</QSYF>')
    assert d.tunes[-1] == 14103500 and d.session.flow.qsy_receiving
    assert d.session.steps is not None
    assert d.session.broadcast.possibly_outstanding == 1
    assert any('1 frames written to the host may still key on the new frequency' in line
               for line in d.said)
    assert d.session.broadcast._retune_command is None
    assert not any('Proceed?' in line for line in d.said)
    d.do('/stop')
    assert d.session.broadcast.possibly_outstanding == 1


def test_declined_retune_and_explicit_clear():
    d = desk()
    start(d)
    d.do('/stop')
    d.do('/all off')
    d.do('/all no')
    assert d.flow.dial == 14105750 and d.session.broadcast.possibly_outstanding == 1
    d.do('/all clear')
    assert d.session.broadcast.possibly_outstanding == 0
    d.do('/all off')
    assert d.flow.dial == 14105000


def test_two_same_size_writes_and_first_bracket_do_not_clear_the_retune_warning():
    d = desk()
    start(d)
    d.do('/stop')
    d.session.command('/all hello everyone')
    d.tick()
    assert len(wire(d)) == 2 and d.session.broadcast.possibly_outstanding == 2
    bracket(d, expected(d))
    assert not d.session.broadcast.active and d.session.broadcast.possibly_outstanding == 2
    d.do('/all off')
    assert any('2 frames written to the host may still key on the new frequency' in x for x in d.said)
    assert d.flow.dial == 14105750


def test_another_clients_same_size_frame_does_not_remove_our_warning():
    d = desk()
    d.host.kiss.sendall(kiss.encode(encode_ui('K2ABC', 'ALL', b'hello everyone'), 1))
    select(d)
    d.session.command('/all hello everyone')
    d.tick()
    bracket(d, expected(d))
    assert d.session.broadcast.possibly_outstanding == 1
    d.do('/all off')
    assert any('1 frames written to the host may still key on the new frequency' in x for x in d.said)


def test_stop_preserves_beacon_schedule_and_does_not_wait_for_unkey():
    d = desk()
    d.session.set_beacon(15)
    schedule = d.session._beacon_due, d.session._beacon_expires
    start(d)
    d.line = True
    d.do('/stop')
    assert not d.session.broadcast.active and not d.session._closing
    assert 'ABORT' not in d.sent and len(d.host.kiss.sent) == 1
    assert (d.session._beacon_due, d.session._beacon_expires) == schedule


def test_reconnect_keeps_and_repeats_possibly_outstanding_count():
    from hfmodem.contact.varac import HostError
    d = desk()
    start(d)
    d.host.command.incoming.append(b'')
    with pytest.raises(HostError, match='closed'):
        d.tick()
    assert d.session.broadcast.possibly_outstanding == 1
    d.session.steps.close()
    d.session.steps = None
    c = d.host.client
    c.command_socket = d.host.command = Socket(d.host.respond)
    c.data_socket = d.host.data = Socket()
    c.kiss_socket = d.host.kiss = Socket()
    c.closed = False
    c.initialize('W9SSJ')
    assert d.session.broadcast.possibly_outstanding == 1
    assert any('1 frames written to the host may still key after reconnect' in x for x in d.said)
    d.do('/all off')
    assert d.flow.dial == 14105750


@pytest.mark.parametrize('failure', ['partial', 'failed'])
def test_bad_kiss_write_closes_immediately_and_next_send_reopens(failure):
    d = desk()
    old = d.host.kiss
    def fail(data):
        assert d.session.broadcast.possibly_outstanding == 1
        if failure == 'partial':
            old.sent.append(data[:5])
        raise OSError('write interrupted')
    old.sendall = fail
    reopened = Socket()
    opens = []
    def connect():
        assert old.closed
        opens.append(True)
        return reopened
    d.host.client._kiss_connect = connect
    select(d)
    d.session.command('/all hello')
    d.tick()
    assert old.closed and d.host.client.kiss_socket is None
    assert d.session.broadcast.possibly_outstanding == 1 and not d.session.broadcast.active
    assert not opens
    d.session.command('/all hello again')
    d.tick()
    assert opens == [True] and d.host.client.kiss_socket is reopened
    assert len(reopened.sent) == 1 and d.session.broadcast.possibly_outstanding == 2
    assert reopened.sent[0] == kiss.encode(encode_ui('W9SSJ', 'ALL', b'hello again'), 1)
    d.do('/stop')


def test_broadcast_rx_is_separate_and_control_bytes_are_escaped():
    d = desk()
    before = list(d.texts)
    raw = encode_ui('K1ABC', 'ALL', b'hi\x1b[31m<DISC>', command=1)
    d.host.kiss.incoming.append(kiss.encode(raw, 1))
    d.tick()
    assert any('[broadcast ' in x and '\\x1b' in x for x in d.said)
    assert not any('\x1b' in x for x in d.said)
    assert d.texts == before and not d.kinds('rx') and not d.kinds('heard')
    assert d.kinds('broadcast_rx')[-1]['payload_hex'] == b'hi\x1b[31m<DISC>'.hex()


def rig_line(d):
    d.session.line_state = lambda: d.line if d.armed and d.armed[-1] else None


@pytest.mark.parametrize('channel', ['cf', 'slot 11'])
def test_selection_after_operator_close_arms_and_verifies_the_line(channel):
    d = desk()
    rig_line(d)
    d.hear('CONNECTED W9SSJ K7QB 500')
    d.do('/bye')
    d.do('/auto off')
    assert d.session.flow.state is State.MONITORING and not d.armed[-1]
    d.do(f'/all {channel}')
    assert d.session.broadcast.selected == d.flow.dial and d.armed[-1]
    d.session.command('/all hello')
    d.tick()
    assert len(wire(d)) == 1


def test_unreadable_line_after_arming_refuses_and_restores_the_arm():
    d = desk()
    d.hear('CONNECTED W9SSJ K7QB 500')
    d.do('/bye')
    d.do('/auto off')
    d.session.line_state = lambda: None
    d.do('/all cf')
    assert d.session.broadcast.selected is None and not d.armed[-1]
    assert any('keying line is unverified' in line for line in d.said)


@pytest.mark.parametrize('guard, reason', [
    ('linked', 'a link is up'), ('pending', 'connection pending'),
    ('ptt', 'PTT on'), ('physical', 'keying line is HIGH'), ('steps', 'in progress')])
def test_refusal_names_the_blocking_condition(guard, reason):
    d = desk()
    if guard == 'linked':
        d.host.client.connected = True
    elif guard == 'pending':
        d.host.client.pending = True
    elif guard == 'ptt':
        d.host.client.ptt = True
    elif guard == 'physical':
        d.line = True
    else:
        d.session.steps = iter([None])
    d.session.command('/all cf')
    assert d.session.broadcast.selected is None
    assert any(line.startswith('error: Broadcast refused: ') and reason in line for line in d.said)


def test_directed_notice_reserves_before_one_write_without_selection_or_retune():
    d = desk()
    d.hear('BUSY OFF')
    order = []
    before = list(d.tunes), list(d.texts)
    d.host.kiss.respond = lambda _: order.append('written')
    d.session.broadcast.directed('W9SSJ-1', before_admit=lambda: order.append('reserved'))
    assert not order
    d.tick()
    assert order == ['reserved', 'written']
    assert wire(d) == [encode_ui('W9SSJ', 'W9SSJ-1', b'<SMP>')]
    assert d.session.broadcast.selected is None
    observed(d)
    assert order == ['reserved', 'written']
    assert (d.tunes, d.texts) == before


@pytest.mark.parametrize('invalid', ['eligibility', 'busy', 'unknown', 'ptt', 'high', 'unknown-line',
                                     'dial', 'generation', 'writer', 'buffer'])
def test_directed_notice_rechecks_before_admission(invalid):
    d = desk()
    d.hear('BUSY OFF')
    allowed, reserved = [True], []
    d.session.broadcast.directed('K1ABC', valid=lambda: allowed[0],
                                 before_admit=lambda: reserved.append(True))
    if invalid == 'eligibility':
        allowed[0] = False
    elif invalid in ('busy', 'unknown'):
        d.host.client.busy = True if invalid == 'busy' else None
    elif invalid == 'ptt':
        d.host.client.ptt = True
    elif invalid in ('high', 'unknown-line'):
        d.line = True if invalid == 'high' else None
    elif invalid == 'dial':
        d.flow.dial += 750
    elif invalid == 'generation':
        d.host.client.link_generation += 1
    elif invalid == 'writer':
        d.host.client._writes[1] = (0, b'pending')
    else:
        d.host.client.buffer = 1
    d.tick()
    assert not wire(d) and not reserved and not d.session.broadcast.active


def test_directed_notice_failed_cooldown_commit_never_submits():
    d = desk()
    d.hear('BUSY OFF')
    def fail():
        raise OSError('disk full')
    d.session.broadcast.directed('K1ABC', before_admit=fail)
    d.drive()
    assert not wire(d) and 'ABORT' not in d.sent
    assert 'disk full' in d.session.broadcast.result


@pytest.mark.parametrize('change', ['high', 'unknown-line', 'busy', 'unknown-busy', 'dial',
                                   'generation', 'connected', 'pending', 'ptt', 'writer', 'buffer'])
def test_directed_notice_rechecks_state_after_durable_reservation(change):
    d = desk()
    d.hear('BUSY OFF')
    reserved = []
    def reserve():
        reserved.append(True)
        c = d.host.client
        if change in ('high', 'unknown-line'):
            d.line = True if change == 'high' else None
        elif change in ('busy', 'unknown-busy'):
            c.busy = True if change == 'busy' else None
        elif change == 'dial':
            d.flow.dial += 750
        elif change == 'generation':
            c.link_generation += 1
        elif change in ('connected', 'pending', 'ptt'):
            setattr(c, change, True)
        elif change == 'writer':
            c._writes[1] = (c.link_generation, b'pending')
        else:
            c.buffer = 1
    d.session.broadcast.directed('K1ABC', before_admit=reserve)
    d.drive()
    assert reserved == [True] and not wire(d)
    assert not d.session.broadcast.active and d.session.broadcast.possibly_outstanding == 0


def test_directed_notice_restores_auto_off_arm_after_completion():
    d = desk()
    d.do('/auto off')
    rig_line(d)
    d.hear('BUSY OFF')
    assert not d.armed[-1]
    d.session.broadcast.directed('K1ABC')
    assert d.armed[-1]
    d.tick()
    observed(d)
    assert not d.armed[-1] and not d.session.broadcast.active


def test_stopped_directed_notice_restores_auto_off_only_after_bracket_settles():
    d = desk()
    d.do('/auto off')
    rig_line(d)
    start(d, directed=True)
    d.hear('PENDING', 'PTT ON', 'BITRATE (1) 18 bps TX')
    d.do('/stop')
    assert d.armed[-1] and d.session.broadcast._arm_restore_pending
    assert not d.session.broadcast.active and d.session.steps is None
    d.hear('CANCELPENDING')
    assert d.armed[-1], 'PTT is still on; cleanup must continue to defer'
    d.hear('PTT OFF', 'BUSY OFF')
    assert not d.armed[-1] and d.sent[-1] == 'LISTEN OFF'
    assert not d.session.broadcast._arm_restore_pending and 'ABORT' not in d.sent


def test_deferred_notice_cleanup_does_not_disarm_an_incoming_contact():
    d = desk()
    start(d, directed=True)
    d.hear('PENDING', 'PTT ON')
    d.do('/stop')
    assert d.session.broadcast._arm_restore_pending
    d.hear('PTT OFF', 'CONNECTED W9SSJ K2ABC 500')
    assert d.host.client.connected and d.session.contact is not None
    assert d.armed[-1] and not d.session.broadcast._arm_restore_pending
    assert d.sent[-1] == 'LISTEN OFF' and 'ABORT' not in d.sent


@pytest.mark.parametrize('change', ['connected', 'pending'])
def test_directed_notice_does_not_disarm_new_contact(change):
    d = desk()
    d.do('/auto off')
    rig_line(d)
    d.hear('BUSY OFF')
    d.session.broadcast.directed('K1ABC')
    assert d.armed[-1]
    setattr(d.host.client, change, True)
    d.tick()
    assert d.armed[-1] and not wire(d)


def test_directed_notice_preflight_failure_restores_auto_off_arm():
    from hfmodem.contact.varac import HostError
    d = desk()
    d.do('/auto off')
    rig_line(d)
    d.host.client.busy = None
    with pytest.raises(HostError, match='deferred'):
        d.session.broadcast.directed('K1ABC')
    assert not d.armed[-1] and not wire(d)


def test_long_frame_uses_the_shared_packet_count():
    d = desk()
    start(d, text='x' * 400)
    assert expected(d) == 9
    bracket(d, 2, 4)
    assert d.session.broadcast.active
    bracket(d, 9, 4)
    assert not d.session.broadcast.active and len(d.host.kiss.sent) == 1


def test_operator_command_crosses_actual_kiss_queue_without_rf(monkeypatch):
    import numpy as np
    from hfmodem.kestrel.tx import broadcast
    from hfmodem.kestrel.host.kiss_tx import KissTxQueue
    from hfmodem.tests.kestrel.test_cq import session_mock
    monkeypatch.setattr(broadcast, 'synth_broadcast', lambda _: np.zeros(8))
    host, _, modem = session_mock()
    d = desk()
    queue = modem.kiss_tx_queue = KissTxQueue(clock=lambda: d.host.now)
    host._send_cmd = lambda line: d.host.lines(line)
    sent = []
    modem._transport.tx = sent.append
    def submit(wire):
        for port, command, raw in kiss.Parser().feed(wire):
            assert port == 0
            assert queue.broadcast(command, raw)
    d.host.kiss.respond = submit
    start(d)
    assert len(queue) == 1 and not sent
    d.host.now += 2
    modem.service_kiss_queue()
    assert len(sent) == 1 and not queue and len(d.host.kiss.sent) == 1
    observed(d)
    assert not d.session.broadcast.active and d.session.broadcast.possibly_outstanding == 1
    assert not d.texts


def test_stop_before_write_sends_nothing():
    d = desk()
    select(d)
    d.session.command('/all hello')
    d.do('/stop')
    assert not wire(d) and d.session.broadcast.possibly_outstanding == 0
    assert 'ABORT' not in d.sent


@pytest.mark.parametrize('directed', [False, True], ids=['all', 'notice'])
def test_packet_counts_do_not_accumulate_across_unrelated_brackets(directed):
    d = desk()
    start(d, directed, text='hello')
    assert expected(d) == 2
    for _ in range(2):
        bracket(d, 1, 1)
        assert d.session.broadcast.active
    bracket(d, 2, 1)
    assert not d.session.broadcast.active and d.session.broadcast.possibly_outstanding == 1


def test_two_writes_after_stop_still_warn_with_count_two():
    d = desk()
    select(d)
    for _ in range(2):
        d.session.command('/all hello')
        d.tick()
        d.do('/stop')
    assert d.session.broadcast.possibly_outstanding == 2 and len(d.host.kiss.sent) == 2
    d.do('/all off')
    assert any('2 frames written to the host may still key on the new frequency' in x for x in d.said)
    assert d.flow.dial == 14105750
