# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Known off-air false grants reproduced on pactor-hears-a-gateway fcd75c6aa."""
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_mfsk as MK
from hfmodem.tests.evidence import require_input
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_response_by_stream import _load_48k
from hfmodem.tests.kestrel.test_stock_caller_connect import IO


@corpora.requires_regress_fixtures
@pytest.mark.parametrize('label,ended', [
    ('oracle_pactor2_ref', 14.8017),
    ('pos_p1_twosided_14110', 7.0607),
    ('pos_pactor1_local', 8.3657),
])
def test_known_offair_false_grants_do_not_take_the_turn(label, ended):
    """Previously whole-fit 1.5–1.8 bypassed the measured weak-pair guard.
    PACTOR's persistent carriers must not grant this unrelated VARA link.
    """
    path = corpora.REGRESS_FIXTURES / (label + '.wav')
    require_input(path)
    x = _load_48k(path)
    io = IO()
    hs = VA.VaraStationHandshake(['AAAA1'], io, bw='500')
    hs.role, hs.caller, hs.called = 'initiator', 'AAAA1', 'BBBB2'
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    hs._peer_offset = 0.
    for pos in range(0, int((ended + 1) * MK.FS), 1024):
        hs.on_rx_stream(x[pos:pos + 1024])
    assert not io.keyed, io.lines
    assert not any('rx turn grant' in line for line in io.lines)
    assert hs.turn == VA._TURN_PEER
