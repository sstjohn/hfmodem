# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Energy and failed expected reads cannot identify the station; fake I/O only."""
from types import SimpleNamespace

import pytest

from .test_owed_ask import _connected


@pytest.mark.parametrize('role, peer', [('initiator', 'KC9GHZ'), ('responder', 'W9SSJ')])
def test_busy_notice_names_expected_peer_without_attributing_signal(role, peer):
    station, io = _connected()
    station.role = role
    io.receiving = True
    station._txq = [b'hello']
    assert not station._ask_for_turn()
    assert io.keys == 0 and not io.sent
    assert io.msgs == [f'channel activity (expected peer {peer}; identity unverified) '
                       '— 1 block(s) queued, asking after the activity']


@pytest.mark.parametrize('role, peer', [('initiator', 'KC9GHZ'), ('responder', 'W9SSJ')])
def test_failed_expected_data_reports_nak_without_claiming_sender(monkeypatch, role, peer):
    station, io = _connected()
    station.role = role
    station._rx_failed = True
    monkeypatch.setattr(station, '_resend_matches', lambda level: True)
    monkeypatch.setattr(station, '_send_burst', lambda kind: True)
    monkeypatch.setattr(station, '_advance_resend',
                        lambda **kwargs: setattr(station, '_expect_level', 5))
    monkeypatch.setattr(station, '_owe_nothing', lambda: None)
    monkeypatch.setattr(station, '_read_levels', lambda: (5,))
    station._irs_nak(SimpleNamespace(level=6))
    assert io.msgs == [
        f'link-recovery requested: Recovery: {peer}: retry request transmitted; '
        'waiting for protocol progress from the peer. Receipt of the request is unknown.',
        f'rx expected DATA at level 6 failed '
        f'(expected peer {peer}; identity unverified): NAK, retry due at level 5']
    assert io.keys == 0 and not io.sent
