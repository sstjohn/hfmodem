# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Host multi-write NAK recovery from the 2026-10-20 failure."""
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


def _station(monkeypatch, named):
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    io = IO()
    modem = VaraHostModem(io, bandwidth="500")
    hs = modem.station
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    hs._link_high_speed, hs._rec500 = True, 7
    hs._lead = VB6Rnd(VF._lcg_advance(VF.lead_state("W9SSJ", False, "500"), 18))
    hs._expect_level = 10
    close = phy.vara_body(b"FS +\r", "W9SSJ", tail=VA._frame_field(0, True),
                          level=named, body_len=phy.body_size("500", 10))
    hs._note_close_level([close])
    keyed = []
    render = hs._render_over

    def spy(bodies, over, level, lead):
        keyed.append((over, phy.host_level("500", level), len(bodies[-1])))
        return render(bodies, over, level, lead)

    hs._render_over = spy
    return io, modem, hs, now, keyed


def _deliver(modem, hs, now, *writes):
    modem.transmit(writes[0])
    modem.poll()
    now[0] += .3
    hs.on_rx_stream(np.zeros(MK.FS // 100))
    for w in writes[1:]:
        modem.transmit(w)
        modem.poll()
    hs._turn_granted(np.zeros(0), said="session-drained-responder")


def _missing_data_naks(hs, now, n):
    for held in (.060, .065, .071)[:n]:
        now[0] += 1.2
        hs._retry_data_over()
        now[0] += .5
        hs._nak_held = held
        hs._took_responder_nak()


def test_resends_follow_the_receivers_expectation_from_seven(monkeypatch):
    io, modem, hs, now, keyed = _station(monkeypatch, 7)
    _deliver(modem, hs, now, b"h" * 70, b"p" * 29)
    _missing_data_naks(hs, now, 3)
    assert [k[:2] for k in keyed] == [(1, 7), (1, 5), (1, 3), (1, 1)], io.msgs


def test_resends_follow_the_receivers_expectation_from_fifteen(monkeypatch):
    io, modem, hs, now, keyed = _station(monkeypatch, 15)
    _deliver(modem, hs, now, b"m" * 20, b"m" * 506)
    _missing_data_naks(hs, now, 3)
    assert [k[:2] for k in keyed] == [(1, 13), (1, 13), (1, 11), (1, 9)], io.msgs
