# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The caller must not prompt the callee's modal outgoing-QSY selector.

The actual VarAC app bench in qsy-modal-1025 establishes this blocking behavior;
this peer reproduces that application boundary rather than replying to every TL.
"""
import re
import pytest
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac_cf_qsy import caller


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('readiness', ['<Q>', '<AWQ>'])
def test_inviting_caller_does_not_block_peer_receive_thread(manual, readiness):
    d = caller(readiness=readiness)
    blocked = False
    peer_dial = v.CALLING_HZ[20]
    def peer(data):
        nonlocal blocked, peer_dial
        d.host.lines('BUFFER 0')  # Modem ACK does not imply application processing.
        message = data.decode().split(' ', 1)[1]
        if blocked:
            return
        if message in ('<Q>', '<AWQ>'):
            blocked = True  # Actual VarAC opens a modal outgoing invitation UI.
        match = re.fullmatch(r'<QSYF>([0-9]{12})</QSYF>', message)
        if match:
            peer_dial = int(match[1])
        if message == '<TL>' and peer_dial == d.flow.dial:
            d.host.data.incoming.append(v.frame_message('<QSYR>'))
    d.host.data.respond = peer
    d.do('/call KE8PLM' + (' cf' if manual else ''))
    if manual:
        assert d.texts == []
        d.do('/qsy 1')
    assert not blocked
    assert peer_dial == d.flow.dial == v.slot_frequency(20, 1)
    assert d.flow.state is v.State.CONNECTED
    assert d.texts[0] == '25 <QSYF>000014104250</QSYF>'
    assert not d.kinds('error')
