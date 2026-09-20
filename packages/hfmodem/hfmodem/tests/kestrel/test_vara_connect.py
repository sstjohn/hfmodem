# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""VARA connect handshake: link-setup synthesis and the end-to-end sequence.

The live proof — kestrel driving a real Wine VARA to CONNECTED over BlackHole —
lives in oracle/validate_kestrel_connect.py (needs VARA). These are the
rig-free regression checks: the link-setup frame the initiator sends, and the
full CR -> connect-response -> link-setup -> connected-ack sequence between two
kestrel handshake drivers.
"""
import numpy as np

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara.vara_arq import VaraIO, VaraState, VaraStationHandshake


def test_link_setup_frame_carries_callsign_and_crc():
    fr = VF.link_setup_frame("W9SSJ")
    assert len(fr) == 92
    assert VF.crc16_genibus(fr[:90]).to_bytes(2, "big") == fr[90:92]
    assert fr[7] == 0x80 and fr[8] == 0x14 and fr[88] == 0x04


def test_link_setup_burst_round_trips_byte_exact():
    fr = VF.link_setup_frame("W9SSJ")
    got = rx.decode_burst(tx.synth_frame(fr, over=0), rx.BASE_LEVEL)
    assert got is not None and got.crc_ok
    assert bytes(got.frame_bytes) == fr


class _IO(VaraIO):
    def __init__(self, peer_inbox):
        self.out = peer_inbox
        self.conn = None
    def key(self, on): pass
    def tx(self, s): self.out.append(np.asarray(s, float))
    def pending(self): pass
    def connected(self, caller, called, bw): self.conn = (caller, called, bw)
    def log(self, m): pass


def test_two_kestrels_complete_the_connect_handshake():
    a_in, b_in = [], []
    A = VaraStationHandshake(["W9SSJ"], _IO(b_in), bw="2300")
    B = VaraStationHandshake(["W1AW"], _IO(a_in), bw="2300")
    B.listen(True)
    A.originate("W1AW")                          # keys the CR
    for _ in range(6):
        while b_in:
            B.on_rx_audio(b_in.pop(0))
        while a_in:
            A.on_rx_audio(a_in.pop(0))
    assert A.state == VaraState.CONNECTED, "initiator did not reach CONNECTED"
    assert B.state == VaraState.CONNECTED, "responder did not reach CONNECTED"
    assert A.io.conn == ("W9SSJ", "W1AW", "2300")
