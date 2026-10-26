# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Kestrel as BW500 caller, chat on: stock's first over behind our grant.

Bench 2, 2026-10-21.
Holding the turn empty behind a high-speed close, we answered stock's frame 0
with the caller's grant. The grant unkeyed with the turn still ours, so the
expected-level reader was never armed, and stock's level-6 over 0.1 s later
went to the base decoder: "will not decode", then NAKs, in gate 4 and in both
5b runs. The cable is noiseless: stock's audio rests on exact zeros.
"""
import warnings

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

CALLER, CALLED = "W9SSJ", "WI9YZA"
TAIL = np.zeros(int(0.13 * MK.FS))
TEXT = b"24 text from S on the slot\n"


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
        self.msgs, self.host = [], []

    def key(self, on): ...
    def tx(self, samples): ...
    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, payload): self.host.append(bytes(payload))
    def log(self, msg): self.msgs.append(msg)


def _held_after_a_high_speed_close():
    io = _IO()
    hs = VA.VaraStationHandshake([CALLER], io, bw="500", high_speed=True,
                                 compression="TEXT", chat=True)
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs._link_high_speed = True
    hs._lead = VA.VB6Rnd(VF.lead_state(CALLER, True, "500"))
    hs._start_compression("TEXT")
    hs.turn, hs._released, hs._lv_owed = VA._TURN_OURS, True, True
    hs._expect_level = 6
    hs._tx_pending = (phy.vara_body(b"x" * 20, CALLER), 9, phy.level_key("500", 10))
    return hs, io


def _stock_over(level=6):
    lead = VA.VB6Rnd(VF.lead_state(CALLER, True, "500"))
    body = phy.vara_body(TEXT, CALLER, body_len=phy.body_size("500", level))
    # stock keys 16-bit samples: its tails round to the cable's exact zeros
    return np.round(phy.render_level([body], "500", level, lead) * 32767) / 32767


@pytest.mark.parametrize("idle_first", [True, False], ids=["idle-then-grant", "grant"])
def test_stocks_first_over_behind_our_grant_is_read_at_its_level(idle_first):
    """Gate 4 answered frame 1 with the idle answer first; 5b granted at once."""
    hs, io = _held_after_a_high_speed_close()
    for frame in (1, 0) if idle_first else (0,):
        hs.on_rx_audio(np.concatenate([MK.synth_tone_pairs(VF.control_bursts(CALLER, "500", frame)[1]),
                                       TAIL]))
    # BW500 now answers a final idle with the complete release: the short
    # handback's pair phase is peer dependent. After that release, the peer's
    # request is answered as a released turn, rather than as a held grant.
    answer = "tx session-turn-release" if idle_first else "tx caller-turn-grant"
    assert any(m.startswith(answer) for m in io.msgs), io.msgs
    assert hs.turn == VA._TURN_PEER
    x = np.concatenate([np.zeros(int(0.1 * MK.FS)), _stock_over(), np.zeros(MK.FS)])
    for at in range(0, len(x), 1024):
        _Clock.t += 1024 / MK.FS
        hs.on_rx_stream(x[at:at + 1024])
    assert "rx WI9YZA's over at level 6 (asked 6)" in io.msgs, io.msgs
    assert not any("will not decode" in m for m in io.msgs), io.msgs
    assert b"".join(io.host) == TEXT


def _silent(x, lead=94 * phy._tone.H, after=MK.FS):
    return np.concatenate([np.zeros(lead), x, np.zeros(after)])


@pytest.mark.parametrize("level", [5, 6])
def test_a_tone_burst_over_digital_silence_stands_clear(level):
    """Keyed on the reader's block grid, the over leaves every block around it
    exactly zero: the floor is 0 for a third of the onsets at level 6."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert phy._tone_burst(_silent(_stock_over(level)), 500, level)


def test_digital_silence_still_refuses_itself_and_a_control_burst():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not phy._tone_burst(np.zeros(6 * MK.FS), 500, 6)
        control = MK.synth_tone_pairs(VF.control_bursts(CALLER, "500", 0)[1])
        assert not phy._tone_burst(_silent(control, after=4 * MK.FS), 500, 6)
