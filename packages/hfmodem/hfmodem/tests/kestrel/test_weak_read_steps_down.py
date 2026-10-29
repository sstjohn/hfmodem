# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A weak BW500 read at level 4 or under asks the sender down, not up.

W9SSJ-1 read W9SSJ's level-4 over at SN -9.6 on 7105 kHz, 2026-10-28, and its
continue asked level 5: below the base the reading was dropped, and a read with
no measure asks one up. A stock IRS steps a delivered frame read at low SN one
level down with frame 5, never more (GEARSHIFT-OBSERVED §3, 0 of 373 fell
further), and KC9GHZ keyed that frame at every low record (kc9ghz-downshift-0920).
"""
import pytest

from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.vara import vara_arq as A


@pytest.mark.parametrize('db', [-9.0, -5.6, -3.0, 0.0, 4.0, 12.0, 30.0])
def test_the_host_scale_reads_back_to_the_channel(db):
    assert RX.channel_snr(RX.stock_sn(db)) == pytest.approx(db, abs=1e-6)


def irs(level, sn):
    hs = A.VaraStationHandshake(['W9SSJ-1'], A.VaraIO(), bw='500')
    hs.role, hs.called, hs.caller = 'initiator', 'W9SSJ', 'W9SSJ-1'
    hs._link_high_speed = True
    hs._rx_read = (level, sn)
    return hs._irs_step(), hs._expect_level


@pytest.mark.parametrize('sn', [-9.6, -11.9])
def test_a_weak_level_4_read_asks_one_down(sn):
    assert irs(4, sn) == (5, 3)


def test_a_strong_level_4_read_still_climbs():
    frame, level = irs(4, 4.9)
    assert level > 4


def test_a_level_4_read_with_no_measure_holds_its_old_rule():
    assert irs(4, None) == (6, 5)


@pytest.mark.parametrize('level,sn,named', [(1, -11.4, 1), (4, -11.9, 3), (2, 2.4, 2), (3, None, 3), (4, None, 4)])
def test_a_closing_over_read_weak_names_the_level_for_the_next_delivery(level, sn, named):
    """W9SSJ-1 read W9SSJ's level-1 greeting at about 9 on the tone measure
    and its own close then named level 4: the answerer's reply came at 4 and
    was not read (simulated air, -17 dB one way, 2026-10-28)."""
    from .test_idle_poll import _IO
    hs = A.VaraStationHandshake(['W9SSJ-1'], _IO(), bw='500')
    hs.role, hs.called, hs.caller = 'initiator', 'W9SSJ', 'W9SSJ-1'
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_PEER
    hs._link_high_speed = True
    hs._rx_read = (level, sn)
    hs._tx_over_response(last=True)
    assert hs._close_byte() == named


@pytest.mark.parametrize('level,frame', [(1, 6), (2, 6), (3, 0)])
def test_the_rows_under_record_3_are_bw500s_only(level, frame):
    """The wideband readings under record 3 keep the rule they had."""
    assert A._index_step(level, -10.0, 16, '2300') == frame
