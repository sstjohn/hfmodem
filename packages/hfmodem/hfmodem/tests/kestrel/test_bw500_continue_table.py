# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The BW500 caller's continue: seven state symbols keyed to the called station.

Ninety stock-keyed continues off nine BW500 tapes say the seven symbols behind
the lead are a function of the called station, the keying station's role, and
the state of the over just answered — its speed level, plus whether its
per-frame field is ``0x81``. All four caller states of W9SSJ -> KC9GHZ are on
tape and each is a frame of the called station's answer stream
[vara_frames.over_continue_state]. The caller now keys the frame for the level
step its reads call for, taped state or not  [VaraStationHandshake._answer_step];
with no over read it uses generated short16, including recovery ladder rungs.

``fixtures/bw500-continue/caller-tails.json`` carries each copy's capture, its
SHA-256, the continue's burst start and symbol origin, and the over it answers.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

FIXTURES = Path(__file__).with_name("fixtures")
LOW = FIXTURES / "bw500-low-levels"
BAND = MK.band_for("500")
STATES = [(1, False), (2, False), (4, False), (4, True)]


@pytest.fixture(scope="module")
def tape():
    path = FIXTURES / "bw500-continue" / "caller-tails.json"
    if not path.exists():
        pytest.skip(f'stock BW500 continue metadata absent: {path}')
    data = json.loads(path.read_text())
    assert {(s['level'], s['field_is_0x81']) for s in data['states']} == set(STATES)
    return data


class _IO(VA.VaraIO):
    def __init__(self):
        self.sent: list[np.ndarray] = []
        self.msgs: list[str] = []
        self.host: list[bytes] = []

    def key(self, on): ...

    def tx(self, samples): self.sent.append(np.asarray(samples, float))

    def log(self, msg): self.msgs.append(msg)

    def data(self, payload): self.host.append(bytes(payload))


def _station(called):
    io = _IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="500")
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", called
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    return hs, io


def _low(name):
    path = LOW / f"{name}.wav"
    if not path.exists():
        pytest.skip(f'stock BW500 low-level recording absent: {path}')
    rate, x = wavfile.read(path)
    assert rate == MK.FS
    return x.astype(float) / 32768


def _tones(samples):
    return [tuple(sorted(p)) for p in MK.demod_tone_pairs(samples, 8, BAND)]


def _symbols(pairs):
    return [tuple(p) for p in pairs]


@pytest.mark.parametrize("key", STATES, ids=[f"L{level}-{field}" for level, field in STATES])
def test_every_taped_continue_regenerates_from_the_over_it_answered(key, tape):
    """One lookup per copy, keyed by nothing but the over's level and field."""
    state, = [s for s in tape['states'] if (s['level'], s['field_is_0x81']) == key]
    want = tuple([tuple(tape["lead"])] + _symbols(state["tail"]))
    assert len(state["samples"]) == state["copies"]
    for sample in state["samples"]:
        assert (sample["over_level"], sample["over_field"] == 0x81) == (
            state["level"], state["field_is_0x81"])
        got = VF.over_continue_state(tape["link"]["called"],
                                     sample["over_level"], sample["over_field"])
        assert got == want and len(got) == 8


def test_the_fields_that_share_a_tail_are_on_the_tape(tape):
    """0x81 is the only one that moves the mid-level-4 tail: the countdown
    values and the base-level close all draw the same seven symbols."""
    mid = [s for s in tape["states"] if s["level"] == 4 and not s["field_is_0x81"]]
    fields = {s["over_field"] for s in mid[0]["samples"]}
    assert fields == {0x95, 0x91, 0x8d}
    tails = {VF.over_continue_state("KC9GHZ", 4, f) for f in fields}
    assert len(tails) == 1
    assert tails.pop() != VF.over_continue_state("KC9GHZ", 4, 0x81)


def test_a_state_never_on_tape_has_no_tail():
    assert VF.over_continue_state("KC9GHZ", 3, 0x95) is None
    assert VF.over_continue_state("K5FIT", 2, 0x81) is None


def test_the_w1aw_links_taped_tails_are_its_generated_ones():
    """The W1AW link's two forms off the 2026-08-30 bench, which no table here
    ever held: mid-level 4, and the over before a delivery's last."""
    assert VF.over_continue_state("W1AW", 4, 0x95) == VF.over_continue("W1AW", "500")[0]
    assert VF.over_continue_state("W1AW", 4, 0x81) == (
        (64, 67), (62, 76), (58, 76), (50, 68), (50, 72), (60, 74), (50, 68), (52, 76))


@pytest.mark.parametrize("name,level", [
    ("bw500-level1-arm1", 1), ("bw500-level1-arm2", 1),
    ("bw500-level2-arm1", 2), ("bw500-level2-arm2", 2)])
def test_a_low_level_over_read_once_holds_its_level(name, level):
    """The stock resend our caller could not climb, answered at KC9GHZ with
    frame 0: one clean read is not grounds to ask for more. The taped tail
    here was stock's caller asking a level up on a quiet cable; ours asked
    K5FIT for two on 40 m, 2026-09-24 05:30Z, and lost the next over."""
    hs, io = _station("KC9GHZ")
    assert hs._answer_data_over(_low(name))
    assert _tones(io.sent[-1]) == list(VF.over_continue_frames("KC9GHZ", "500", (0,))[0][0])
    assert hs._next_500 == level


def _assert_short(samples, called):
    kind = VF.for_bw(VF.SESSION_OVER_RESPONSE_SHORT, "500")
    assert MK.demod_burst(samples, kind, BAND) == VF.handshake_tones(called, kind)


def test_any_link_answers_the_recorded_low_level_over_with_its_own_tail():
    """The former W1AW fallback was ignored by stock K5FIT twice per over."""
    hs, io = _station("K5FIT")
    assert hs._answer_data_over(_low("bw500-level1-arm1"))
    assert _tones(io.sent[-1]) == list(VF.over_continue_frames("K5FIT", "500", (0,))[0][0])
    assert hs._reack_frame == VA.OVER_CONTINUE_BURST


@pytest.mark.parametrize("called", ["KC9GHZ", "K5FIT"])
def test_no_over_read_uses_a_called_keyed_short_answer(called):
    hs, io = _station(called)
    hs._peer_over_state = None
    assert hs._tx_continue_answer(VA.OVER_CONTINUE_BURST) == VA.OVER_CONTINUE_SHORT
    assert len(io.sent) == 1
    _assert_short(io.sent[0], called)


@pytest.mark.parametrize("called,state", [
    ("K5FIT", (3, 0x95)), ("K5FIT", (2, 0x81)), ("KC9GHZ", (3, 0x95))])
def test_a_state_never_on_tape_keys_the_generated_continue(called, state):
    """The continue is the called station's generated frame for the level
    step, taped or not  [vara_frames.over_continue_frames]."""
    hs, io = _station(called)
    hs._peer_over_state = state
    assert hs._tx_continue_answer(VA.OVER_CONTINUE_BURST) == VA.OVER_CONTINUE_BURST
    assert _tones(io.sent[0]) == list(VF.over_continue_frames(called, "500", (0,))[0][0])


def test_no_over_read_never_gets_an_eight_symbol_tail_on_the_ladder():
    hs, io = _station("K5FIT")
    hs._peer_over_state = None
    hs._key_over_answer(last=False, owes_release=False)
    assert hs._reack_frame == VA.OVER_CONTINUE_SHORT
    assert hs._answer_owed == VA._OWED_OVER and not hs._owed_block
    for _ in range(3):
        assert hs._reack()
    assert len(io.sent) == 4 and hs._reacks == 3
    # Initial short, repeated short, generated32, then the captured rung is
    # resolved to short too. There is no unmeasured eight-symbol emission.
    for i in (0, 1, 3):
        _assert_short(io.sent[i], "K5FIT")
    kind = VF.for_bw(VF.SESSION_OVER_RESPONSE, "500")
    assert MK.demod_burst(io.sent[2], kind, BAND) == VF.handshake_tones("K5FIT", kind)
    assert not any("8-symbol continue burst" in m for m in io.msgs)


class _RefusingIO(_IO):
    def __init__(self):
        super().__init__()
        self.transmits = False

    def tx(self, samples):
        if self.transmits:
            super().tx(samples)

    def tx_went_out(self):
        return self.transmits


def test_refused_answer_keeps_positive_ack_debt_until_retry():
    hs, _ = _station("K5FIT")
    hs.io = io = _RefusingIO()
    hs._peer_over_state = None
    hs._key_over_answer(last=False, owes_release=False)
    assert io.sent == [] and hs._reacks == hs._since_progress == 0
    assert hs._answer_owed == VA._OWED_OVER and not hs._owed_block
    io.transmits = True
    assert hs._reack()
    assert len(io.sent) == 1 and hs._reacks == hs._since_progress == 1
    _assert_short(io.sent[0], "K5FIT")
    assert not hs._owed_block


def test_direct_captured_sender_refuses_with_no_over_read():
    hs, io = _station("K5FIT")
    hs._peer_over_state = None
    assert not hs._tx_over_continue()
    assert io.sent == []


def test_a_responder_keys_its_roles_continue():
    hs, io = _station("KC9GHZ")
    hs.role = "responder"
    hs._peer_over_state = (1, 0x99)
    assert hs._tx_continue_answer(VA.OVER_CONTINUE_BURST) == VA.OVER_CONTINUE_BURST
    assert _tones(io.sent[0]) == list(VF.over_continue("KC9GHZ", "500")[1])


@pytest.mark.parametrize("bw", ["2300", "2750"])
def test_other_bandwidths_keep_their_explicit_captured_answer(bw):
    hs, io = _station("K5FIT")
    hs.bw = bw
    hs._peer_over_state = None
    assert hs._tx_continue_answer(VA.OVER_CONTINUE_BURST) == VA.OVER_CONTINUE_BURST
    got = MK.demod_tone_pairs(io.sent[0], 8, MK.band_for(bw))
    assert [tuple(sorted(pair)) for pair in got] == list(VF.over_continue("K5FIT", bw)[0])
