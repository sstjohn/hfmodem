# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Two stations over a clean audio loopback, every over rendered and read by the
real tone, OFDM and index codecs: the caller W9SSJ and the callee KB8AY connect,
the callee delivers 4 kB to the caller, the turn changes hands, and the caller
delivers 4 kB back. High speed capable, the ISS climbs by the rows the IRS's SNR
clears into the OFDM levels both ways, and an over the channel ruins draws the
caller's NAK, falls by our table and recovers; speed limited, the session stays
at level 4 as it always has.

The caller runs as it does on air: fed its receive stream in 20 ms polls on a
clock that advances with the audio. The callee's connect is its own FSM and it
keys only what its own methods render, but its data phase has no receive route
(the stream readers are the initiator's): ``_Link.callee_hears`` stands in for
one, reading what the caller keys with the same decoders and handing the
callee's own takers what they read. The callee also has no turn-request reader,
so the stand-in grants the caller's ask with the drained frame.
"""
import re
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf2300 as RX
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

FS = MK.FS
BW = "2300"
POLL = FS // 50
_FRAMES = (0, 1, 4, 5, 6, 7, 8)


class _Clock:
    t = 1000.0

    @classmethod
    def monotonic(cls) -> float:
        return cls.t


class _IO(VA.VaraIO):
    def __init__(self, out: list):
        self.out, self.msgs, self.host = out, [], []

    def key(self, on): ...

    def tx(self, samples):
        x = np.asarray(samples, float)
        self.out.append(x)
        _Clock.t += len(x) / FS

    def pending(self): ...

    def connected(self, *a): ...

    def log(self, msg): self.msgs.append(msg)

    def data(self, payload): self.host.append(bytes(payload))


def _pairs(tones):
    return [tuple(sorted(p)) for p in tones]


class _Link:
    def __init__(self, high_speed: bool, ruin: int | None = None):
        self.to_callee, self.to_caller = [], []
        self.caller = VA.VaraStationHandshake(["W9SSJ"], _IO(self.to_callee), bw=BW,
                                              high_speed=high_speed)
        self.callee = VA.VaraStationHandshake(["KB8AY"], _IO(self.to_caller), bw=BW,
                                              high_speed=high_speed)
        self.received, self.read_levels = bytearray(), []
        self.overs, self.ruin = 0, ruin

    def connect(self):
        self.callee.listen(True)
        self.caller.originate("KB8AY")
        while self.to_callee or self.to_caller:
            if self.to_callee:
                self.callee.on_rx_audio(self.to_callee.pop(0))
            else:
                self.caller_hears(self.to_caller.pop(0))

    def run(self):
        while self.to_callee or self.to_caller:
            if self.to_caller:
                self.caller_hears(self.to_caller.pop(0))
            else:
                self.callee_hears(self.to_callee.pop(0))

    def caller_hears(self, x):
        """``x`` behind 0.1 s of silence, then silence until the caller keys or
        nine seconds pass. The ``ruin``-th DATA over arrives under noise that
        leaves it locatable and unreadable."""
        if len(x) >= VA._DATA_OVER_MIN:
            self.overs += 1
            if self.overs == self.ruin:
                x = x + np.random.default_rng(7).normal(0, 3 * np.std(x), len(x))
        audio = np.concatenate([np.zeros(FS // 10), x, np.zeros(9 * FS)])
        keyed = len(self.to_callee)
        for i in range(0, len(audio), POLL):
            _Clock.t += POLL / FS
            self.caller.on_rx_stream(audio[i:i + POLL])
            if len(self.to_callee) > keyed:
                return

    def _read(self, x) -> phy.LevelRead | None:
        """An over at the level the callee asked for, or at a speed limited
        link's close, the index record below the base."""
        pad = np.zeros(FS // 10)
        x = np.concatenate([pad, x, pad])
        read = phy.decode_level(x, BW, self.callee._expect_level)
        if read is not None or self.callee._link_high_speed:
            return read
        rec = RX.index_guard(x)[0][0]
        fr = RX.decode_over(x, 0, len(x), level=rec)
        body = bytes(fr.payload)
        return phy.LevelRead(phy.host_level(BW, rec), (body,), located=1) \
            if fr.crc_ok else None

    def callee_hears(self, x):
        cs = self.callee
        if len(x) >= VA._DATA_OVER_MIN:
            read = self._read(x)
            assert read is not None and not read.lost
            body = read.frames[-1]
            self.read_levels.append(read.level)
            self.received += phy.vara_payload(body, caller=cs.caller, body_len=len(body))
            if phy.over_is_last(body, cs.caller):
                cs._tx_control_burst()
                return
            cs._rx_read = (read.level, read.quality)
            cs.io.tx(MK.synth_tone_pairs(cs._continue_tail()))
            return
        n = max(1, round((len(x) - MK.STRIDE) / MK.HOP) + 1)
        tones, pairs = MK.demod_tones(x, n), _pairs(MK.demod_tone_pairs(x, n))
        continues = VF.over_continue_frames(cs.called, BW, _FRAMES)
        step = next((f for f in _FRAMES if n == 32 and tones == VF.handshake_tones(
            cs.called, replace(VF.SESSION_OVER_RESPONSE, preadv=1 + 62 * f))), None)
        if step is None:
            step = next((f for f in _FRAMES if pairs == _pairs(continues[f][0])), None)
        if step is None and tones == VF.handshake_tones(cs.called,
                                                        VF.SESSION_OVER_RESPONSE_SHORT):
            step = 0
        if step == 1:
            cs._pending_answer_at = _Clock.t
            cs._matched_data_nak = (cs._tx_pending, cs._pending_answer_at,
                                    "session-data-pair-nak")
            cs._took_data_nak()
        elif step is not None:
            cs._continue_frame = (cs._tx_pending, step)
            cs._took_over_continue()
        elif n == 11 and pairs == _pairs(VF.control_bursts(cs.caller, BW)[0]):
            cs._took_control_burst()
        elif n == 32 and tones == VF.handshake_tones(cs.caller, VF.SESSION_TURN_REQUEST):
            cs.turn, cs._expect_level = VA._TURN_PEER, 4
            cs._send_burst(VF.SESSION_DRAINED_RESPONDER)
        else:
            cs.on_rx_audio(x)


def _keyed_levels(station) -> list[int]:
    """The host level of every DATA over ``station`` keyed, retries included."""
    out = []
    for m in station.io.msgs:
        if hit := re.match(r"tx DATA over #\d+(?: at record (\d+))?", m):
            rec = hit[1]
        elif hit := re.match(r"retry DATA over #\d+ .*?record (\d+)", m):
            rec = hit[1]
        else:
            continue
        out.append(4 if rec is None else phy.host_level(BW, int(rec)))
    return out


def _read_at(station) -> list[int]:
    """The host level of every over ``station`` read at the level it asked."""
    return [int(hit[1]) for m in station.io.msgs
            if (hit := re.match(r"rx KB8AY's over at level (\d+) \(asked", m))]


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(VA, "time", _Clock)


def _session(high_speed, size, ruin=None):
    rng = np.random.default_rng(1)
    greeting, upload = (rng.integers(0, 256, size, dtype=np.uint8).tobytes() for _ in "ab")
    link = _Link(high_speed, ruin)
    link.connect()
    assert link.caller.state == link.callee.state == VA.VaraState.CONNECTED
    link.callee.turn = VA._TURN_OURS
    link.callee.send(greeting)
    link.run()
    link.caller.send(upload)
    link.run()
    assert b"".join(link.caller.io.host) == greeting
    assert bytes(link.received) == upload
    return link


def test_high_speed_capable_stations_climb_into_ofdm_both_ways_and_recover_a_nak():
    link = _session(True, 4096, ruin=2)
    caller, callee = link.caller, link.callee
    assert "rx link-setup — caller is W9SSJ (high speed capable, ask 6)" in callee.io.msgs
    assert "the caller's confirm starts our first over at level 6" in callee.io.msgs

    # stock's 2300 ladder resends a heard-but-unread level 10 at 7
    assert "rx expected DATA at level 10 failed (expected peer KB8AY; identity unverified): " \
        "NAK, retry due at level 7" in caller.io.msgs
    nak = next(m for m in callee.io.msgs if "NAK; retaining over #2" in m)
    assert nak.endswith(f"retrying at record {phy.level_key(BW, 7)}")
    assert _keyed_levels(callee) == [6, 10, 7, 11, 14, 13]
    assert _read_at(caller) == [6, 7, 11, 14, 13]

    assert _keyed_levels(caller) == link.read_levels == [4, 6, 10, 14, 15]


def test_a_speed_limited_session_stays_at_level_four():
    link = _session(False, 1024)
    caller, callee = link.caller, link.callee
    assert "rx link-setup — caller is W9SSJ (free, ask 4)" in callee.io.msgs
    assert caller._lead is None and callee._lead is None
    assert _keyed_levels(callee) == _keyed_levels(caller) == [4] * 11 + [3]
    assert link.read_levels == [4] * 11 + [3]
    assert not _read_at(caller)
