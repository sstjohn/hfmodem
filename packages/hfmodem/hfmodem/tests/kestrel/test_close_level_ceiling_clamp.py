# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A stock close can name a host level above what :data:`arq.phy.TX_CEILING`
lets this station key — BW500's "FS Y" over closed ``0f 82``, naming level 15
against a ceiling of 13. ``_note_close_level`` used to drop any close above the
ceiling outright and fall back to record 4, which stock NAKs; the fix clamps
the named level to the ceiling instead, matching what a stock caller in the
same spot keys and what a bench diagnostic read 556/556 B against.
"""
import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA
from .test_data_over_gate import _connected


def station(monkeypatch, bw, called, close_level, data):
    """A high speed capable caller whose peer's close named ``close_level`` —
    which may exceed this bandwidth's TX_CEILING — with ``data`` queued as its
    first over."""
    monkeypatch.setattr(VA.OF, 'data_over_tx', lambda *a, **k: np.zeros(64))
    hs, io = _connected(bw=bw)
    hs.called, hs._link_high_speed, hs.turn = called, True, VA._TURN_OURS
    body_len = phy.body_size(bw, phy.close_level(bw))
    close = phy.vara_body(b'x' * 5, hs.caller, tail=0x42, body_len=body_len,
                          level=close_level)
    hs._note_close_level([close])
    hs.send(data)
    return hs, io


def test_bw500_close_above_ceiling_clamps_to_it(monkeypatch):
    hs, io = station(monkeypatch, '500', 'WC9P-10', 15, bytes(605))
    assert phy.host_level('500', hs._tx_pending[2]) == phy.TX_CEILING['500'] == 13


def test_bw2300_close_above_ceiling_clamps_to_it(monkeypatch):
    hs, io = station(monkeypatch, '2300', 'WC9P-10', 18, bytes(605))
    assert phy.host_level('2300', hs._tx_pending[2]) == phy.TX_CEILING['2300'] == 16
