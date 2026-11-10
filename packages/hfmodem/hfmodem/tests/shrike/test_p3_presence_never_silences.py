# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A presence verdict never silences a live PACTOR-3 link.

K4MSU 2026-11-07 went "P3 peer unsupported -> receive only; retained bytes, no
response key" twice and keyed nothing for twenty-five slots while the gateway
kept sending CRC-valid repeats. The link is answered in every slot; the silence
and CRC lifetimes, not a presence count, end a peer that has really gone.
"""
from types import SimpleNamespace

import numpy as np

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import modem, opportunity


def test_a_peer_repeating_a_crc_valid_packet_is_never_counted_dark():
    """The `;PM` run: the changeover repeated CRC-valid every window while the
    read for its structure was skipped. Each window is the peer, present."""
    a, io = modem()
    for _ in range(12):
        token = a.begin_receive_opportunity()
        a.on_rx_packet(1, b';PM', 0x00, crc_ok=True)
        assert a.settle_peer_presence()
        a.finish_receive_opportunity(token)
    assert a._dark_windows == 0
    assert a.p3_tx_allowed()


def test_a_presence_read_skipped_for_budget_is_not_a_verdict():
    """Hold 27 skipped its presence read for the cycle's budget, and calling that
    unsupported is what discarded the CRC-valid `;PM` repeat in the same window."""
    a, _ = modem()
    a.begin_receive_opportunity()
    op = a._receive_opportunity
    sessrx = SimpleNamespace(
        host=SimpleNamespace(arq=a, protocol=spec.Protocol.PACTOR3),
        _p3_presence_form=(a._link_generation, 1), _p3_row0=600000,
        _p3_cycle_n=60000, _p3_presence_gaps=(), _p3_repeat_changeover=True,
        _p3_span=38880, _p3_prekey_crc=(600000, 1), _p3_delivered_at=600000,
        P3_SL1_RESERVE_S=onair._SessionRx.P3_SL1_RESERVE_S,
        P3_WIDEBAND_RESERVE_S=onair._SessionRx.P3_WIDEBAND_RESERVE_S,
        _p3_acquisition_fits=lambda reserve: False,
        _p3_expected=lambda: (), _corrected=lambda x: x)
    # This packet's CRC, read before the key that answers it.
    raster = SimpleNamespace(boundary=lambda slot: 594000 + slot * 60000,
                             _p3_peer=(595680, 38880, 60000, 0))
    tx = SimpleNamespace(live=SimpleNamespace(lost=0), slot=1, keyings=(),
                         tx_key_up=None, tx_audio_start=None, tx_end=None)
    # The audio reaches the packet the slot answers; only the budget is short.
    onair._p3_qualify_presence(sessrx, raster, tx, np.zeros(48000, np.float32), 593000)
    assert op.presence_checked and op.qualified
    assert op.presence_source == 'current CRC'


def test_the_answered_packets_crc_counts_with_no_audio_of_it_in_hand():
    """Old audio ending before the packet the slot answers holds nothing of it,
    and the CRC-valid read of that packet still answers for the peer."""
    a, _ = modem()
    a.begin_receive_opportunity()
    op = a._receive_opportunity
    sessrx = SimpleNamespace(
        host=SimpleNamespace(arq=a, protocol=spec.Protocol.PACTOR3),
        _p3_presence_form=(a._link_generation, 1), _p3_row0=600000,
        _p3_cycle_n=60000, _p3_presence_gaps=(), _p3_repeat_changeover=True,
        _p3_span=38880, _p3_prekey_crc=(600000, 1), _p3_delivered_at=600000,
        P3_SL1_RESERVE_S=onair._SessionRx.P3_SL1_RESERVE_S,
        P3_WIDEBAND_RESERVE_S=onair._SessionRx.P3_WIDEBAND_RESERVE_S,
        _p3_acquisition_fits=lambda reserve: False,
        _p3_expected=lambda: (), _corrected=lambda x: x)
    # This packet's CRC, read before the key that answers it.
    raster = SimpleNamespace(boundary=lambda slot: 594000 + slot * 60000,
                             _p3_peer=(595680, 38880, 60000, 0))
    tx = SimpleNamespace(live=SimpleNamespace(lost=0), slot=1, keyings=(),
                         tx_key_up=None, tx_audio_start=None, tx_end=None)
    onair._p3_qualify_presence(sessrx, raster, tx, np.zeros(4800, np.float32), 500000)
    assert op.presence_checked and op.presence_source == 'current CRC'


def test_a_read_with_nothing_to_read_keeps_its_verdict():
    """Only the budget leaves a read unchecked. With no audio for the packet
    the slot answers, a CRC delivered in the opportunity is not this packet."""
    a, _ = modem()
    a.begin_receive_opportunity()
    op = a._receive_opportunity
    sessrx = SimpleNamespace(
        host=SimpleNamespace(arq=a, protocol=spec.Protocol.PACTOR3),
        _p3_presence_form=(a._link_generation, 1), _p3_row0=600000,
        _p3_cycle_n=60000, _p3_presence_gaps=(), _p3_repeat_changeover=True,
        _p3_span=38880, _p3_prekey_crc=None, _p3_delivered_at=None,
        P3_SL1_RESERVE_S=onair._SessionRx.P3_SL1_RESERVE_S,
        P3_WIDEBAND_RESERVE_S=onair._SessionRx.P3_WIDEBAND_RESERVE_S,
        _p3_acquisition_fits=lambda reserve: True,
        _p3_expected=lambda: (), _corrected=lambda x: x)
    raster = SimpleNamespace(boundary=lambda slot: 600000 + slot * 60000, _p3_peer=None)
    tx = SimpleNamespace(live=SimpleNamespace(lost=0), slot=1, keyings=(),
                         tx_key_up=None, tx_audio_start=None, tx_end=None)
    onair._p3_qualify_presence(sessrx, raster, tx, np.zeros(0, np.float32), 595000)
    assert op.presence_checked and op.presence_source == 'unsupported'


def _copy_before(delivered):
    """K4MSU 2026-11-07 15:45: the gateway's changeover `RMS` repeated every
    cycle and each copy was read a cycle late, in the next window, with the
    presence read skipped for the budget."""
    a, _ = modem()
    a.begin_receive_opportunity()
    a.on_rx_packet(1, b'RMS', 0x00, True, breakin=True)
    sessrx = SimpleNamespace(
        host=SimpleNamespace(arq=a, protocol=spec.Protocol.PACTOR3),
        _p3_presence_form=(a._link_generation, 1), _p3_row0=delivered,
        _p3_cycle_n=60000, _p3_presence_gaps=(), _p3_repeat_changeover=True,
        _p3_span=38880, _p3_prekey_crc=None, _p3_delivered_at=delivered,
        P3_SL1_RESERVE_S=onair._SessionRx.P3_SL1_RESERVE_S,
        P3_WIDEBAND_RESERVE_S=onair._SessionRx.P3_WIDEBAND_RESERVE_S,
        _p3_acquisition_fits=lambda reserve: False,
        _p3_expected=lambda: (), _corrected=lambda x: x)
    raster = SimpleNamespace(boundary=lambda slot: 654000 + slot * 60000,
                             _p3_peer=(delivered - 4320, 38880, 60000, 0))
    tx = SimpleNamespace(live=SimpleNamespace(lost=0), slot=1, keyings=(),
                         tx_key_up=None, tx_audio_start=None, tx_end=None)
    onair._p3_qualify_presence(sessrx, raster, tx, np.zeros(48000, np.float32), 653000)
    return a


def test_the_copy_before_the_answered_packet_is_the_peer_still_there():
    a = _copy_before(600000)
    assert a._receive_opportunity.presence_source == 'previous CRC'
    assert a.settle_peer_presence()
    assert a._dark_windows == 0 and a._p3_crc_wait_ticks == 0


def test_a_copy_two_cycles_old_still_says_nothing():
    a = _copy_before(540000)
    assert a._receive_opportunity.presence_source == 'unsupported'
