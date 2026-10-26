# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock's closing CW ident is not a level-6 over.

Bench 2 gate 5a, contact 2, 2026-10-21:
stock closed with a 0.69 s frame and then its ident, 168.51-172.48 s of b2a:
WI9YZA in Morse at 28 WPM, one carrier shifted between a 1500 Hz mark and a
750 Hz space, 3.25 s. Over the cable's digital silence it fills a level-6
span, and our reader, armed behind our release, NAKed it as a lost over.
"""
import numpy as np
import pytest

from hfmodem.core import cwid
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_bw500_grant_reader_1021 import _stock_over

CALLER, CALLED = "W9SSJ", "WI9YZA"


class _Clock:
    t = 1000.0

    @classmethod
    def monotonic(cls):
        return cls.t


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    _Clock.t = 1000.0
    monkeypatch.setattr(VA, "time", _Clock)


class _IO(VA.VaraIO):
    def __init__(self):
        self.msgs, self.sent = [], []

    def key(self, on): ...
    def tx(self, samples): self.sent.append(samples)
    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, payload): ...
    def log(self, msg): self.msgs.append(msg)


def stock_ident(call=CALLED, dit=0.0427):
    """Measured off b2a: a 42.7 ms dit, the space at 0.45 of the mark."""
    keys = [(False, .03)] + [(on, s * dit / cwid.dit_seconds(20)) for on, s in cwid.keying(call, 20)]
    keys.append((False, .254))
    on = np.concatenate([np.full(int(s * MK.FS), k) for k, s in keys])
    phase = np.cumsum(np.where(on, 1500.0, 750.0)) * 2 * np.pi / MK.FS
    x = 0.33 * np.where(on, 1.0, 0.45) * np.sin(phase)
    return np.round(x * 32767) / 32767


#: The 16-symbol frame stock keyed ahead of its ident, demodulated off b2a.
CLOSE_TONES = [62, 62, 76, 50, 58, 70, 50, 52, 62, 50, 64, 66, 70, 60, 68, 64]


def stocks_close():
    close = np.round(MK.synth_tones(CLOSE_TONES) * 32767) / 32767
    return np.concatenate([close, np.zeros(int(.03 * MK.FS)), stock_ident()])


@pytest.mark.parametrize("call", [CALLED, "K4MCK", "VE3/W9SSJ"])
def test_the_ident_reads_as_its_callsign(call):
    assert phy.ident(np.r_[np.zeros(MK.FS), stock_ident(call), np.zeros(MK.FS)]) == call


@pytest.mark.parametrize("level", [5, 6])
def test_no_over_is_an_ident(level):
    assert phy.ident(np.r_[np.zeros(MK.FS), _stock_over(level), np.zeros(MK.FS)]) is None


@pytest.mark.parametrize("level", [5, 6])
def test_the_ident_is_no_burst_at_a_tone_row(level):
    x = np.r_[np.zeros(MK.FS), stocks_close(), np.zeros(2 * MK.FS)]
    assert phy.decode_level(x, 500, level) is None


@pytest.mark.parametrize("level", [5, 6])
def test_stocks_close_behind_our_release_draws_no_nak(level):
    io = _IO()
    hs = VA.VaraStationHandshake([CALLER], io, bw="500", high_speed=True,
                                 compression="TEXT", chat=True)
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs._link_high_speed = True
    hs._lead = VA.VB6Rnd(VF.lead_state(CALLER, True, "500"))
    hs._start_compression("TEXT")
    hs.turn, hs._lv_owed, hs._expect_level = VA._TURN_OURS, True, level
    assert hs._release_turn() and hs._lv_armed
    sent = len(io.sent)
    x = np.r_[np.zeros(int(.6 * MK.FS)), stocks_close(), np.zeros(3 * MK.FS)]
    for at in range(0, len(x), 1024):
        _Clock.t += 1024 / MK.FS
        hs.on_rx_stream(x[at:at + 1024])
    assert not any("NAK" in m for m in io.msgs), io.msgs
    assert len(io.sent) == sent, io.msgs


@pytest.mark.parametrize("chat", [False, True], ids=["chat-off", "chat-on"])
@pytest.mark.parametrize("route", ["stream", "bracket"])
def test_stocks_close_under_noise_closes_the_callers_link_on_either_route(route, chat):
    """Stock's caller reports DISCONNECTED behind the closer's ident, CHAT ON
    and OFF (P0 item7 called-disc). Under noise the close frame goes unread and
    the base locator locks a level-4 record inside the ident: that route reads
    it as the ident too, and closes the link as the bracket does."""
    io = _IO()
    hs = VA.VaraStationHandshake([CALLER], io, bw="500", chat=chat)
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    x = np.r_[np.zeros(int(.6 * MK.FS)), stocks_close(), np.zeros(3 * MK.FS)]
    x = x + 0.1 * np.random.default_rng(1).normal(size=len(x))
    if route == "bracket":
        hs.on_rx_audio(x)
    else:
        for at in range(0, len(x), 1024):
            _Clock.t += 1024 / MK.FS
            hs.on_rx_stream(x[at:at + 1024])
    assert hs.state is VA.VaraState.DISCONNECTED, io.msgs
    assert f"rx {CALLED}'s CW ident — it has closed the link" in io.msgs
    assert not io.sent
