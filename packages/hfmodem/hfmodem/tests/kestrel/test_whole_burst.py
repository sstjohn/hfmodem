# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The peer's control burst read whole, and what that is allowed to cost.

The preamble test in front of this one scores three symbols of eleven. On
2026-09-17/18 eight gateways keyed twenty-seven of these bursts at this station
and it refused five of them at full strength — three of those five the answer to
a DATA over, which is a delivery abandoned with the peer already asking for the
next block. In all five the seven tail symbols were exact and the head was gone.

So the bar here is not sensitivity. It is that reading the other sixteen
carriers must not also read the band: everything below is the false-accept side
of that, and the sweep it rests on is in `_BURST_MATCH`.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel import corpora

FIXTURE = Path(__file__).parent / "fixtures" / "whole-burst-0918"
BAND = MK.band_for("2750")
_RESP = VF.CONTROL_BURST_RESPONDER_2750
_OURS = VF.CONTROL_BURST_CALLER_2750


def _recorded(name: str) -> np.ndarray:
    path = FIXTURE / f"{name}.wav"
    if not path.exists():
        pytest.skip(f"on-air clip absent: {path}")
    with wave.open(str(path)) as w:
        assert w.getframerate() == MK.FS and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32768.0


@pytest.mark.parametrize("clip", ["ns0a-post-over-control",
                                  "ve3wlr-post-over-control"])
def test_a_burst_whose_head_the_channel_took_is_still_the_burst(clip):
    """Both gateways answered a DATA over of ours and this station keyed nothing.

    The preamble is the evidence the reader in front has and it is not there;
    the tail is the evidence it does not use and it is exact.
    """
    x = _recorded(clip)
    assert VA._ack_plateau(x, 0, None, BAND) < VA._ACK_PLATEAU, (
        "the preamble test is supposed to refuse this one — if it now takes it, "
        "this clip no longer holds the failure it was cut for")
    assert VA._burst_match(x, _RESP, BAND) >= VA._BURST_MATCH


def test_a_turnaround_the_peer_keyed_nothing_into_stays_empty():
    """K9KDJ heard our link-setup and answered with a turn-request. There is no
    control burst on this clip at any alignment and the reader must say so."""
    x = _recorded("k9kdj-quiet-turnaround")
    assert VA._burst_match(x, _RESP, BAND) < VA._BURST_MATCH


def test_the_grid_does_not_move_the_floor_or_the_signal():
    """Unlike the preamble test, whose floor is a property of the grid — a
    PACTOR-1 exchange holds that preamble for 59 alignments half a carrier low
    and for none at zero — this score is taken against the band's own power and
    barely moves across the half bin either side."""
    burst, quiet = (_recorded("ns0a-post-over-control"),
                    _recorded("k9kdj-quiet-turnaround"))
    for off in (-0.5, -0.25, 0.0, 0.25, 0.5):
        assert VA._burst_match(burst, _RESP, BAND, bin_offset=off) >= VA._BURST_MATCH
        assert VA._burst_match(quiet, _RESP, BAND, bin_offset=off) < VA._BURST_MATCH


def test_our_own_burst_coming_back_through_the_mute_is_not_the_peers():
    """The two tails of one link are what role picks between, and this is the
    one confusion the preamble test cannot make at all: our own copy holds that
    preamble for 60 alignments. Keying the transmitter at it is the endless
    turn-idle `_peer_control_burst` was written to prevent."""
    ours = MK.synth_tone_pairs(_OURS)
    x = np.concatenate([np.zeros(MK.FS // 4), ours, np.zeros(MK.FS // 2)])
    assert VA._ack_plateau(x, 0, None, BAND) >= VA._ACK_PLATEAU
    assert VA._burst_match(x, _RESP, BAND) < VA._BURST_MATCH
    assert VA._burst_match(x, _OURS, BAND) >= VA._BURST_MATCH


def test_a_nak_does_not_become_an_acknowledgement():
    """A frame that says the over was NOT read must never score as one that says
    it was  [vara_frames.nak]."""
    band = MK.band_for("2300")
    for tones in VF.NAK_BY_CALLER[("W9SSJ", "2300")]:
        x = MK.synth_tone_pairs(tones)
        assert VA._burst_match(x, VF.CONTROL_BURST_RESPONDER_2300, band) < VA._BURST_MATCH


@corpora.requires_regress_fixtures
def test_nothing_without_a_vara_session_on_it_reaches_the_bar():
    """The false-accept side, on the corpus the constants above are set against.

    Every recording of the shared regression corpus that carries no VARA session,
    both tails a session can be scored on, every alignment. The three that do
    carry one are excluded by name and by the thing that makes them positives:
    `qso_vara_kc9ghz`, `qso_vara_ko2f` and `qso_vara_ns0a` hold a real gateway's
    own control bursts, and this reader finds them at 2.19 to 3.20.
    """
    held = {}
    for path in sorted(corpora.REGRESS_FIXTURES.glob("*.wav")):
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() + 1e-12)
        worst = max(VA._burst_match(x, tones, band)
                    for tones, band in ((_RESP, BAND),
                                        (VF.CONTROL_BURST_RESPONDER_2300,
                                         MK.band_for("2300"))))
        if worst >= VA._BURST_MATCH:
            held[path.name] = round(worst, 3)
    assert set(held) == {"qso_vara_kc9ghz.wav", "qso_vara_ko2f.wav",
                         "qso_vara_ns0a.wav"}, (
        f"the recordings that reach {VA._BURST_MATCH} are {held}")
