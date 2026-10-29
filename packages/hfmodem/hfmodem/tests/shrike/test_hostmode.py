# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""hostmode's own self-test, collected where the suite looks.

Found in the same sweep that caught arq's: a ``__main__``-only block carrying
six real assertions -- the manual's CRC example, both decoder round trips, the
corruption/request/repeat exchange, and byte-at-a-time framing -- that pytest
never ran.
"""
from hfmodem.core import hostmode


def test_framing_selftest():
    hostmode._selftest()


def test_tracker_and_ptc_share_transport_framing_without_modem_imports():
    from hfmodem.shrike import ptc
    from hfmodem.brubru.host import tracker

    assert hostmode.Decoder is ptc.hostmode.Decoder is tracker.hm.Decoder
    wire = hostmode.unwrap(hostmode.command(1, 'I W9SSJ'))
    [packet] = tracker.hm.Decoder(crc=False).feed(wire)
    assert packet.channel == 1 and packet.is_command
    assert packet.text == 'I W9SSJ'
