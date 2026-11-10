# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The callsign announcement goes out lowercase, as both completions on tape do.

`1dl6maa` and `1w4dna` are the two PACTOR-1 -> PACTOR-3 completions in the corpus;
every arm of ours to 2026-09-08 announced `1W9SSJ`. The grant proves the status
bits were read; it does not prove the string was.
"""
from hfmodem.shrike import pactor1
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_p3_offer import Keyed, calling_station, cs_event


def test_the_announcement_is_lowercase():
    _, keyed = calling_station()
    assert keyed.packets[0][1] == b"1w9ssj\r"


def test_the_uppercase_control():
    keyed = Keyed()
    host = PtcHost(peer=keyed, mycall="W9SSJ")
    host.announce_lower = False
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    assert keyed.packets[0][1] == b"1W9SSJ\r"
