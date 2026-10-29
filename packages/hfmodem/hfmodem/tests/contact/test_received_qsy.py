# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real Session loop, fake owner fence; no RF or nested application polling."""
import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk


def linked():
    d = Desk()
    d.hear('CONNECTED KQ4QWH W9SSJ 500', 'SN 2')
    return d


def test_kq_queued_awq_and_chat_do_not_block_receiver_move():
    d = linked()
    d.host.data.respond = lambda data: None
    d.session.command('hello?')
    d.host.client._line('BUFFER 16')
    pending = d.host.client._sent_after
    before = list(d.host.data.sent)
    gate = {'ack': False, 'low': False}
    closed = []
    def owner(target, generation):
        try:
            while not (gate['ack'] and gate['low']):
                yield .05
            d.tunes.append(target)
        finally:
            closed.append(True)
    d.flow.received_retune_steps = owner
    began = d.host.now
    d.receive('<Q:225>')
    assert d.host.now - began < 1
    assert d.session.steps is not None and not d.tunes
    d.session.command('/status')
    assert any('physical unkey' in line for line in d.said)
    gate['ack'] = True
    d.tick()
    assert not d.tunes
    gate['low'] = True
    d.tick()
    assert d.tunes == [14_107_250]
    assert d.host.client.buffer == 16 and d.host.client._sent_after == pending
    assert d.host.data.sent == before and closed == [True]
    assert not any(x in d.sent for x in ('ABORT', 'CLEANTXBUFFER', 'DISCONNECT'))
    d.receive('<TL>')
    assert d.host.data.sent[-1] == v.frame_message('<QSYR>')
    d.receive('de KQ4QWH <R+02>')
    assert d.session.steps is None and d.flow.state is v.State.CONNECTED


def test_kb_second_invitation_is_relative_to_actual_slot_and_supersedes_old_wait():
    d = linked()
    d.host.client._text(v.frame_message('<Q>'))
    d.session.command('/qsy 1')
    for _ in range(5):
        d.tick()
    assert d.flow.dial == 14_104_250
    assert d.flow.qsy_phase == 'waiting for peer acceptance (<QSYR>)'
    d.receive('<Q:150>')
    assert d.tunes == [14_104_250, 14_105_750]
    assert d.flow.qsy_receiving
    assert d.flow.qsy_phase == 'waiting for peer arrival (<TL>)'
    d.receive('<TL>')
    d.receive('de KQ4QWH <R+02>')
    assert d.session.steps is None
    d.host.now += 100
    d.tick()
    assert not d.kinds('error')


@pytest.mark.parametrize('action', ['stop', 'bye', 'quit', 'new_link'])
def test_pending_owner_transaction_is_cancelled_without_late_tune(action):
    d = linked()
    cancelled = []
    def owner(target, generation):
        try:
            while True:
                yield .05
        finally:
            cancelled.append(True)
    d.flow.received_retune_steps = owner
    d.receive('<Q:75>')
    if action == 'new_link':
        d.host.client._line('DISCONNECTED')
        d.host.client._line('PENDING')
        d.host.client._line('CONNECTED N0XYZ W9SSJ 500')
        d.tick()
    elif action == 'quit':
        d.session.quit()
    else:
        d.session.command('/' + action)
        if action == 'bye':
            d.drive()
    assert cancelled == [True] and not d.tunes
    assert not d.flow.qsy_receiving


def test_other_exchange_and_undrained_tl_are_not_superseded():
    d = linked()
    d.host.client._text(v.frame_message('<Q>'))
    def reply(data):
        if data != v.frame_message('<TL>'):
            d.host.lines('BUFFER 0')
        else:
            d.host.lines('BUFFER 6')
    d.host.data.respond = reply
    d.session.command('/qsy 1')
    for _ in range(5):
        d.tick()
    old = d.session.steps
    d.receive('<Q:150>')
    assert d.session.steps is old
    assert d.tunes == [14_104_250]
    assert 'another exchange' in d.kinds('error')[-1]['text']


def test_received_boundary_timeout_closes_request_without_discarding_text():
    d = linked()
    closed = []
    def owner(target, generation):
        try:
            while True:
                yield .05
        finally:
            closed.append(True)
    d.flow.received_retune_steps = owner
    d.receive('<Q:75>')
    d.host.now += 91
    d.tick()
    assert closed == [True] and not d.tunes and d.session.steps is None
    assert 'retune boundary timed out' in d.kinds('error')[-1]['text']


def test_old_link_invitation_not_replayed_and_missing_adapter_never_tunes():
    d = linked()
    d.host.client._text(v.frame_message('<Q:75>'))
    d.host.client._line('DISCONNECTED')
    d.host.client._line('PENDING')
    d.host.client._line('CONNECTED N0XYZ W9SSJ 500')
    d.tick()
    assert not d.tunes and d.session.steps is None
    d.flow.received_retune_steps = None
    d.receive('<Q:75>')
    assert not d.tunes
    assert 'modem-owner' in d.kinds('error')[-1]['text']


def test_cancel_after_owner_completed_keeps_actual_dial_and_notifies_once(tools):
    from types import SimpleNamespace as NS
    from hfmodem.kestrel.host.audio_runtime import RetuneRequest
    from varac_chat import received_retune_adapter
    d = linked()
    requests = []
    def request(callback, **kwargs):
        item = RetuneRequest(callback, **kwargs)
        requests.append(item)
        return item
    runtime = NS(modem=NS(station=object(), received_cursor=10), request_retune=request)
    d.flow.received_retune_steps = received_retune_adapter(
        runtime, d.host.client, d.tunes.append, lambda: True,
        completed=d.flow.received_tuned)
    d.receive('<Q:75>')
    item, = requests
    item.callback()
    item.done = True
    d.session.command('/stop')
    assert d.tunes == [14_105_750] and d.flow.dial == 14_105_750
    assert sum(e.kind == 'tune' and e.raw == '14105750' for e in d.host.events) == 1
    assert d.session.steps is None


def test_invalid_new_invitation_does_not_cancel_existing_acceptance_wait():
    d = linked()
    d.host.client._text(v.frame_message('<Q>'))
    d.session.command('/qsy 1')
    for _ in range(5):
        d.tick()
    old = d.session.steps
    d.receive('<Q:99999>')
    assert d.session.steps is old
    assert d.flow.qsy_phase == 'waiting for peer acceptance (<QSYR>)'
    assert d.tunes == [14_104_250]
