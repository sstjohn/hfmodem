# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where the peer is transmitting, and every reader scored on it.

A VARA link is two radios that agree on a frequency to within whatever their
dials, their references and the path between them allow. Half a carrier here is
12 Hz. This file holds the receiver to reading a peer that is off our grid by
less than that, which is to say almost every peer there is, and to reading
nothing where there is nothing at any grid at all.

The failure it was written for: K0SI answered every DATA over of 2026-09-18 with
the 11-symbol control burst, delivered whole and clean 8 Hz low of us, and this
station logged silence and closed two sessions with the traffic still queued —
while reading the same gateway's single-tone frames 16 of 16 in the same
turnarounds.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel.test_turn_law import (
    _CALLED, _IO, _MYCALL, _connected, _off_frequency, _peer_control)

# A carrier low of our grid by as near half a bin as a station can be and still
# have a nearest bin: at exactly 0.5 a tone is equidistant from two of them and
# which one each symbol reads is the channel's to decide, which is a property of
# sampled spectra and not of this receiver. K0SI, the station this was written
# for, sat at 0.35.
HALF_BIN = -0.45


def _dialled(bw: str = "2300"):
    """A station part-way through calling ``_CALLED``: the stream is live and the
    connect-response has not arrived."""
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw=bw, mfsk_only=True)
    hs.originate(_CALLED, _MYCALL)
    return hs, io


def _stream(hs, x, block: int = 4800) -> None:
    for i in range(0, len(x), block):
        hs.on_rx_stream(x[i:i + block])


def _response(offset: float, bw: str = "2300") -> np.ndarray:
    kind = VF.connect_response(bw)
    x = MK.synth_burst(_CALLED, kind)
    return np.concatenate([_off_frequency(x, offset), np.zeros(MK.FS)])


# --- the measurement ---------------------------------------------------------- #

@pytest.mark.parametrize("offset", [-0.45, -0.35, -0.125, 0.0, 0.25, 0.45])
def test_the_connect_response_says_where_the_peer_is(offset):
    """The frame that opens the link is the one that measures it: fifteen tones
    keyed to the callsign we dialled, already confirmed before this is read."""
    hs, _ = _dialled()
    _stream(hs, _response(offset))
    assert hs._peer_shift == 0
    assert hs._peer_offset == pytest.approx(offset, abs=0.05)


def test_an_unheard_peer_has_no_frequency_and_is_read_on_ours():
    hs, _ = _dialled()
    _stream(hs, np.zeros(MK.FS))
    assert hs._peer_offset is None and hs._grid == 0.0


def test_the_grid_is_said_once_and_only_when_it_has_moved():
    hs, io = _dialled()
    _stream(hs, _response(HALF_BIN))
    said = [m for m in io.msgs if "off our grid" in m]
    assert len(said) == 1 and "-10." in said[0], io.msgs
    hs._peer_frame(_off_frequency(MK.synth_burst(
        _CALLED, VF.SESSION_TURN_RELEASE_RESPONDER), HALF_BIN),
        VF.SESSION_TURN_RELEASE_RESPONDER, _CALLED)
    assert [m for m in io.msgs if "off our grid" in m] == said


def test_a_later_frame_slews_the_grid_rather_than_replacing_it():
    """HF moves. One reading is not the link's frequency for the rest of a
    session, and one reading is not worth the whole of the grid either."""
    hs, _ = _dialled()
    _stream(hs, _response(0.0))
    first = hs._peer_offset
    kind = VF.SESSION_TURN_RELEASE_RESPONDER
    hs._peer_frame(_off_frequency(MK.synth_burst(_CALLED, kind), HALF_BIN),
                   kind, _CALLED)
    assert first > hs._peer_offset > HALF_BIN


def test_a_frame_that_was_not_read_leaves_the_grid_alone():
    hs, _ = _dialled()
    _stream(hs, _response(HALF_BIN))
    grid = hs._peer_offset
    kind = VF.SESSION_TURN_RELEASE_RESPONDER
    _heard, _at, matched = hs._peer_frame(
        _off_frequency(MK.synth_burst("W1AW", kind), 0.25), kind, _CALLED)
    assert not matched and hs._peer_offset == grid


# --- what the measurement is for ---------------------------------------------- #

def _off_grid_session(offset: float = HALF_BIN, bw: str = "2300"):
    """A connected station whose peer transmits ``offset`` of a carrier off our
    grid, with the link measured the way the air measures it."""
    hs, io = _dialled(bw)
    _stream(hs, _response(offset, bw))
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def test_the_control_burst_of_an_off_grid_peer_answers_our_over():
    hs, _ = _off_grid_session()
    burst = _off_frequency(_peer_control(), HALF_BIN)
    assert hs._peer_control_burst(np.pad(burst, (4800, 4800)))


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
def test_the_continue_of_an_off_grid_peer_asks_for_the_next_over(bw):
    hs, _ = _off_grid_session(bw=bw)
    burst = _off_frequency(MK.synth_tone_pairs(
        VF.over_continue(_MYCALL, bw)), HALF_BIN)
    assert hs._peer_over_continue(np.pad(burst, (4800, 4800)))


def test_an_off_grid_nak_is_still_a_nak():
    hs, _ = _off_grid_session()
    burst = _off_frequency(MK.synth_tone_pairs(VF.nak(_MYCALL, "2300")[1]),
                           HALF_BIN)
    x = np.pad(burst, (4800, 4800))
    assert hs._peer_nak(x) and not hs._peer_over_continue(x)


# --- what it must not turn into ----------------------------------------------- #

@pytest.mark.parametrize("grid", [-0.5, -0.25, 0.0, 0.25, 0.5])
@pytest.mark.parametrize("seed", range(6))
def test_band_noise_is_nothing_at_every_grid(grid, seed):
    """A moved grid must not be a second chance to accept. The whole argument for
    measuring the offset instead of searching it is that one grid is tested per
    reader either way — so noise that holds nothing at zero holds nothing here."""
    hs, _ = _connected()
    hs._peer_offset = grid
    x = np.random.default_rng(seed).normal(0, 0.05, int(1.9 * MK.FS))
    assert not hs._peer_control_burst(x)
    assert not hs._peer_over_continue(x)
    assert not hs._peer_responder_release(x)


def test_a_peer_on_our_grid_is_measured_there_and_reads_unchanged():
    """The control arm: KB8AY was 0.6 Hz off us on the same band in the same hour
    and its frames were read on our grid before any of this. They still are."""
    hs, io = _off_grid_session(offset=0.0)
    assert abs(hs._peer_offset) < 0.05
    assert not [m for m in io.msgs if "off our grid" in m]
    assert hs._peer_control_burst(np.pad(_peer_control(), (4800, 4800)))
    assert hs._peer_over_continue(np.pad(
        MK.synth_tone_pairs(VF.over_continue(_MYCALL, "2300")), (4800, 4800)))


def test_the_grid_never_reaches_a_whole_carrier():
    """Past half a bin the tone is in the next bin and ``_peer_shift`` owns it."""
    hs, _ = _off_grid_session()
    for _ in range(8):
        hs._note_peer_offset(np.full(64, -0.9), np.arange(32), hs._grid)
    assert hs._peer_offset == -0.5


# --- the two gateways this was written for ------------------------------------ #

FIXTURE = Path(__file__).parent / "fixtures" / "link-frequency-0918"


def _recorded(name: str) -> np.ndarray:
    path = FIXTURE / f"{name}.wav"
    if not path.exists():
        pytest.skip(f"on-air clip absent: {path}")
    with wave.open(str(path)) as w:
        assert w.getframerate() == MK.FS and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32768.0


def _gateway(call: str, clip: str):
    """A connected session with that gateway's own recorded connect-response put
    through the stream reader, exactly as the evening's arms put it."""
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300", mfsk_only=True)
    hs.originate(call, _MYCALL)
    _stream(hs, _recorded(clip))
    assert hs._peer_shift == 0, "the response arrived on our carriers"
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def test_k0si_is_read_off_its_own_connect_response_and_answers_our_over():
    """2026-09-18, 40 m: the gateway whose every answer this station missed.

    Both clips are its own audio, minutes apart — the frame that says where it
    transmits, and the frame that could not be read without that."""
    hs, io = _gateway("K0SI", "k0si-connect-response")
    assert hs._peer_offset == pytest.approx(-0.36, abs=0.05)
    assert any("off our grid" in m for m in io.msgs), io.msgs
    assert hs._peer_control_burst(_recorded("k0si-post-over-control"))


def test_k0si_on_our_own_grid_is_the_silence_two_sessions_read():
    """The preamble, which is what those two sessions were reading it with.

    On our own grid it holds for 0 alignments and at K0SI's own for 35, which is
    the whole of what this file is about. The frame itself is still there, and a
    reader that scores all eleven symbols against the band's own power finds it
    at either grid — 2.37 here against 2.54 measured  [see `_burst_match`]. So
    the second route takes this burst on our grid too, and the grid is still what
    the preamble test and `_cont_held` cannot do without.
    """
    hs, _ = _connected()
    assert hs._peer_offset is None
    burst = _recorded("k0si-post-over-control")
    assert VA._ack_plateau(burst, 0, None, hs._band) < VA._ACK_PLATEAU
    assert VA._burst_match(burst, VF.CONTROL_BURST_RESPONDER_2300,
                           hs._band) >= VA._BURST_MATCH


def test_kb8ay_reads_the_same_burst_the_way_it_always_did():
    """The control. It was 0.6 Hz off us, it was read on our grid before any of
    this, and a measured grid must leave it exactly where it was."""
    hs, io = _gateway("KB8AY", "kb8ay-connect-response")
    assert abs(hs._peer_offset) < 0.1
    assert not [m for m in io.msgs if "off our grid" in m]
    burst = _recorded("kb8ay-post-over-control")
    assert hs._peer_control_burst(burst)
    plain, _ = _connected()
    assert plain._peer_control_burst(burst)
