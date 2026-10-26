# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Explicit QSYF destinations avoid assuming the peer's CAT frequency is known."""
import pytest
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac_cf_qsy import caller
from hfmodem.tests.contact.test_received_qsy import linked


@pytest.mark.parametrize('band', [20, 40, 80])
@pytest.mark.parametrize('slot', [1, 2, 15])
def test_outgoing_invitation_names_exact_destination_and_completes_opening(band, slot):
    d = caller()
    d.flow.band, d.flow.dial = band, v.CALLING_HZ[band]
    def reply(data):
        d.host.lines('BUFFER 0')
        if data == v.frame_message('<TL>'):
            d.host.data.incoming.append(v.frame_message('<QSYR>'))
    d.host.data.respond = reply
    d.do('/call KE8PLM cf')
    d.do(f'/qsy {slot}')
    target = v.slot_frequency(band, slot)
    assert d.texts[0] == v.frame_message(f'<QSYF>{target:012d}</QSYF>').decode()
    assert not any('<Q:' in s for s in d.texts)
    assert d.tunes == [target] and d.flow.state is v.State.CONNECTED
    assert not d.kinds('error')


def test_documented_full_frequency_tag_round_trips():
    assert v.emit_tag('QF', 14105750) == '<QF:14105750>'
    parts = v.TextParser().feed('<QF:14105750>', final=True)
    assert [(p.tag,p.value) for p in parts] == [('QF','14105750')]


@pytest.mark.parametrize('value', [0, -75, '14.105750', '+14105750', '1e7', 'x'])
def test_invalid_full_frequency_is_not_emitted(value):
    with pytest.raises(ValueError):
        v.emit_tag('QF', value)


def test_absolute_received_invitation_uses_owner_boundary_and_tl_qsyr_exchange():
    d = linked()
    d.receive('<QF:14105750>')
    assert d.tunes == [14105750] and d.flow.qsy_receiving
    assert d.flow.qsy_phase == 'waiting for peer arrival (<TL>)'
    d.receive('<TL>')
    assert d.host.data.sent[-1] == v.frame_message('<QSYR>')
    d.receive('de KQ4QWH <R+02>')
    assert d.session.steps is None and d.flow.state is v.State.CONNECTED


def test_absolute_destination_does_not_depend_on_an_origin_frequency():
    d = linked()
    assert d.flow.received_qsy_target('<QF:14105750>', None) == 14105750
    assert d.flow.received_qsy_target('<Q:75>', 14105000) == 14105750
    with pytest.raises(ValueError):
        d.flow.received_qsy_target('<Q:75>', None)


def test_peer_absolute_invitation_can_supersede_pending_local_chat():
    d = caller()
    d.host.data.respond = lambda data: None
    d.do('/call KE8PLM cf')
    d.host.client.send_text('hello')
    d.session.command('/qsy 1')
    for _ in range(5):
        d.tick()
    assert d.flow.qsy_phase == 'waiting for local delivery'
    d.receive('<QF:14105750>')
    assert d.tunes == [14105750] and d.flow.qsy_receiving
    assert not any('<QF:' in text for text in d.texts)
    assert not d.kinds('error')


@pytest.mark.parametrize('tag', ['<QF:7104250>', '<QF:14105000>', '<QF:14105001>',
                                 '<QF:-75>', '<QF:0>', '<QF:999999999>'])
def test_full_frequency_cannot_bypass_same_band_slot_policy(tag):
    d = linked()
    with pytest.raises(ValueError):
        d.flow.received_qsy_target(tag, d.flow.dial)
    assert not d.tunes
