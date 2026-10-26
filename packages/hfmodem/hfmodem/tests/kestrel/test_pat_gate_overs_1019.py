# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The overs of the 2026-10-19 BW500 Pat gate that stock refused while starved.

Read off the a2b cable (working/bench-1019-pat-intermittent/LEADS.txt), every
DATA over was right: its body, its record and its place in the lead stream.
"""
import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK


class IO(VA.VaraIO):
    def __init__(self):
        self.msgs = []

    def key(self, on): pass
    def tx(self, samples): pass
    def pending(self): pass
    def connected(self, *args): pass
    def data(self, payload): pass
    def log(self, message): self.msgs.append(message)
    def tx_metadata(self, **metadata): pass


def test_bw500_overs_match_the_tape(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    e1 = VF.lead_state("W9SSJ", False, "500")
    draws = {VF._lcg_advance(e1, k): k for k in range(400)}

    io = IO()
    modem = VaraHostModem(io, bandwidth="500")
    hs = modem.station
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    hs._link_high_speed, hs._rec500 = True, 7
    # The link-setup keyed at E1 itself took the stream's first 18 draws.
    hs._lead = VB6Rnd(VF._lcg_advance(e1, 18))

    keyed = []
    render = hs._render_over

    def spy(bodies, over, level, lead):
        keyed.append((over, level, draws.get(lead.state), bodies[-1]))
        return render(bodies, over, level, lead)

    hs._render_over = spy

    def write(payload):
        modem.transmit(payload)
        modem.poll()

    def granted_by_poll():
        hs._took_poll()
        hs._grant_held = .1
        hs._took_turn_grant()
        if hs._tx_pending is None:
            hs._tx_data_over()

    write(b"h" * 70)
    now[0] += .3
    hs.on_rx_stream(np.zeros(MK.FS // 100))
    write(b"p" * 29)
    hs._turn_granted(np.zeros(0), said="session-drained-responder")
    hs._took_control_burst()

    write(b"c" * 6)
    granted_by_poll()
    hs._took_control_burst()

    hs._expect_level = 10
    close = phy.vara_body(b"FS +\r", "W9SSJ", tail=VA._frame_field(0, True),
                          level=15, body_len=phy.body_size("500", 10))
    hs._note_close_level([close])

    write(b"m" * 20)
    now[0] += .3
    hs.on_rx_stream(np.zeros(MK.FS // 100))
    write(b"m" * 506)
    hs._turn_granted(np.zeros(0), said="session-drained-responder")
    hs._nak_held = .041
    hs._took_responder_nak()
    hs._nak_held = .068
    hs._took_responder_nak()
    hs._continue_frame = (hs._tx_pending, 0)
    hs._took_over_continue()
    hs._took_control_burst()

    write(b"\x04\x1c")
    granted_by_poll()

    assert [k[:3] for k in keyed] == [
        (1, 7, 19), (2, 7, 64),
        (3, 13, 109), (3, 13, 164), (3, 11, 219),
        (4, 11, 264), (5, 11, 309),
    ], io.msgs
    assert keyed[-1][3] == bytes.fromhex("041c14f6") + bytes(466) + bytes.fromhex("0a82")
