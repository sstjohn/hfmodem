# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Connected QSY invitation contract verified against VarAC v16.1.8 IL.

See working/varac-on-air/qsyf-contract-1025/check_binary.py for an independent
binary guard evaluator. This peer models only that guard, not a real UI/rig.
"""
import re
import pytest
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac_cf_qsy import caller
from hfmodem.tests.contact.test_received_qsy import linked


def destination(message):
    match = re.fullmatch(r'<QSYF>([0-9]{12})</QSYF>', message)
    return int(match[1]) if match else None


@pytest.mark.parametrize('frequency', [7104250, 14104250, 3574250])
def test_absolute_invitation_has_connected_tag_and_twelve_frequency_digits(frequency):
    invitation = v.emit_tag('QSYF', frequency)
    assert destination(invitation) == frequency
    assert len(invitation) == 25
    assert v.frame_message(invitation) == b'25 ' + invitation.encode()


@pytest.mark.parametrize('value', [None, 0, -1, 10**12, '+14104250', '14.104250', 'x'])
def test_invalid_absolute_frequency_cannot_emit_invitation(value):
    with pytest.raises(ValueError):
        v.emit_tag('QSYF', value)


def test_w1hpc_wire_delivery_does_not_imply_application_invitation():
    # CRC-verified DATA from the 17:30 UTC W1HPC contact, e3c6dd26e dirty.
    recorded = b'13 <QF:14104250>'
    assert destination(recorded.decode().split(' ', 1)[1]) is None
    assert destination('<QF:000014104250>') is None
    assert destination('<QSYF>14104250</QSYF>') is None


@pytest.mark.parametrize('band', [20, 40, 80])
def test_peer_moves_only_when_it_recognizes_the_actual_invitation(band):
    d = caller()
    d.flow.band, d.flow.dial = band, v.CALLING_HZ[band]
    peer_dial = v.CALLING_HZ[band]
    def peer(data):
        nonlocal peer_dial
        # Delivery is acknowledged even for an unrecognized application tag.
        d.host.lines('BUFFER 0')
        message = data.decode().split(' ', 1)[1]
        accepted = destination(message)
        if accepted is not None:
            peer_dial = accepted
        elif message == '<TL>' and d.flow.dial == peer_dial:
            d.host.data.incoming.append(v.frame_message('<QSYR>'))
    d.host.data.respond = peer
    d.do('/call KE8PLM cf')
    d.do('/qsy 1')
    assert peer_dial == d.flow.dial == v.slot_frequency(band, 1)
    assert d.flow.state is v.State.CONNECTED
    assert not d.kinds('error')


def test_received_qsyf_uses_existing_owner_fence_and_arrival_exchange():
    d = linked()
    d.receive('<QSYF>000014105750</QSYF>')
    assert d.tunes == [14105750] and d.flow.qsy_receiving
    d.receive('<TL>')
    assert d.host.data.sent[-1] == v.frame_message('<QSYR>')
    d.receive('de KQ4QWH <R+02>')
    assert d.flow.state is v.State.CONNECTED


@pytest.mark.parametrize('invitation', ['<QSYF>14105750</QSYF>',
    '<QSYF>000007104250</QSYF>', '<QSYF>000014105001</QSYF>',
    '<QSYF>000014105000</QSYF>', '<QSYF>000000000000</QSYF>'])
def test_qsyf_does_not_bypass_slot_and_band_guards(invitation):
    d = linked()
    with pytest.raises(ValueError):
        d.flow.received_qsy_target(invitation, d.flow.dial)
    assert not d.tunes
