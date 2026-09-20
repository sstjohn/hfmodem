# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The answer that is somebody else's, and the route that reports it.

Both routes that accept an answer regenerate the fifteen payload tones for the
callsign we dialled and look for them, so three different things reach the same
verdict: nobody transmitted, the gateway transmitted and we could not read it,
and somebody transmitted a well-formed connect-response addressed to a station we
cannot name.

The route here reads a connect-response by its own structure — the eight fixed
preamble tones, and a payload the generator itself accounts for — and reports it
as an answer nobody can be named for. It never brings a link up; that is what the
callsign check is for, and this exists because the callsign check said no.

It runs behind :meth:`_peer_answer`, which is what the two KB3AC-10 bursts this
file was written for turned out to belong to: they carry that gateway's own tones
at another frame of its stream, and `test_peer_answer` holds them now.

The evidence it rests on is in two halves. The preamble is eight carriers taking
seven distinct values out of the seventy the tone track can return. The payload is
the output of one generator whose only free parameter is the called callsign's own
start state, so :func:`vara_frames.payload_states` asks whether ANY station's
connect-response carries these tones rather than whether one of the few hundred
on a panel does. The second half is the rejection: the one thing in the whole population that
holds the preamble and not the generator is in this file too.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import (
    _I_CR_SENT,
    _I_LINKSETUP_SENT,
    VaraState,
    VaraStationHandshake,
)
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_response_by_stream import (
    _IO,
    _PANEL,
    _load_48k,
    band_limit,
)

FS = MK.FS
RESP = VF.CONNECT_RESPONSE
MYCALL = "W9SSJ"


def _awaiting(called: str, panel=_PANEL) -> VaraStationHandshake:
    """An initiator that has keyed a connect-request and is waiting for the answer."""
    hs = VaraStationHandshake([MYCALL], _IO(), panel=panel)
    hs.role, hs.called, hs.caller = "initiator", called, MYCALL
    hs.state, hs.step = VaraState.CONNECTING, _I_CR_SENT
    return hs


def _feed(hs: VaraStationHandshake, x: np.ndarray, block: int = 4800) -> None:
    for i in range(0, len(x), block):
        hs.on_rx_stream(x[i:i + block])


# --------------------------------------------------------------------------- #
# The generator, backwards.
@pytest.mark.parametrize("kind", [RESP, VF.CONNECT_RESPONSE_500, VF.SESSION_CONFIRM,
                                  VF.SESSION_TURN_RELEASE])
@pytest.mark.parametrize("call", ["W1AW", "KB3AC-10", "KC9GHZ", "NS0A", "VE7QRP"])
def test_the_payload_generator_runs_backwards(call, kind):
    """The state :func:`payload_bins` starts from is the state that comes back."""
    crc = VF.crc16_genibus(VF.normalize_callsign(call).encode("ascii"))
    seed = (crc + kind.seed_off) & 0x7FFF
    mult = VF._g_hash(VF.normalize_callsign(call)) + ((crc + 50) >> 15)
    want = VF._lcg_advance(VF._start(seed), mult + kind.preadv)
    assert VF.payload_states(VF.payload_bins(call, kind), kind) == (want,)


def test_a_partly_heard_payload_still_pins_one_state():
    """The tones a mute swallowed are not tones the peer got wrong, and the
    inverse has to say so: the state is pinned by what arrived."""
    whole = VF.payload_bins("KC9GHZ", RESP)
    for lost in range(0, 8):
        heard = [-1] * lost + whole[lost:]
        assert VF.payload_states(heard, RESP) == VF.payload_states(whole, RESP), (
            f"{lost} tones under the mute and the state stopped being pinned")


def test_a_tone_off_the_alphabets_own_lattice_has_no_state():
    """Parity is fixed by symbol index, so half the alphabet is unreachable at
    each position and a single misread tone is not this generator's output."""
    whole = VF.payload_bins("KC9GHZ", RESP)
    for k in range(RESP.n_payload):
        wrong = list(whole)
        wrong[k] += 1                              # one carrier off, wrong parity
        assert VF.payload_states(wrong, RESP) == ()


def test_arbitrary_tones_land_on_a_state_at_the_rate_the_arithmetic_says():
    """The bar this route stands on, measured rather than argued.

    A payload symbol draws from 35 of the alphabet's 70 carriers, so eight
    comparable tones are one of 35**8 sequences against 2**24 states and land on
    one 7.4e-6 of the time. Fifteen tones drawn from the whole alphabet — which is
    what a phantom would have to be — reach it far less often than that, and 2000
    of them reaching it never is the assertion a corpus sweep cannot make.
    """
    rng = np.random.default_rng(11)
    alphabet = sorted(VF.BW2300_TONES.carriers)
    hits = sum(bool(VF.payload_states(
        list(rng.choice(alphabet, RESP.n_payload)), RESP)) for _ in range(2000))
    assert hits == 0, f"{hits} of 2000 random tone sequences reproduce a state"


# --------------------------------------------------------------------------- #
# The two arms.
# --------------------------------------------------------------------------- #
# The floor.
def _negatives():
    """Real off-air audio that holds no connect-response at all."""
    out = [corpora.CLEAR_CHANNEL, *corpora.ONAIR_SILENT_CALLS]
    if corpora.REGRESS_FIXTURES.is_dir():
        out += sorted(corpora.REGRESS_FIXTURES.glob("*.wav"))
    return [p for p in out if p.exists()]


@corpora.requires_regress_fixtures
@corpora.requires_clear_channel
@corpora.requires_onair_silent_calls
def test_the_negative_corpus_holds_no_answer_that_names_nobody():
    """The false-positive figure, over the population the other floors in this
    package are measured on: the 31 shared regression fixtures (PACTOR-1/2/3,
    ARDOP, FT8, WSPR, band noise from four continents and four real VARA
    sessions), the verified clear channel, and the two 2026-08-06 calls nobody
    answered — 1362 s, 9,967,325 alignment-shifts.

    Zero. The preamble alone is already zero at :data:`_UNATTR_MIN_PRE`: six
    comparable symbols exact is reached four times over this population, all four
    inside recordings of real VARA sessions, and no payload behind any of the four
    is something this generator could have emitted.
    """
    named = {}
    for path in _negatives():
        hs = _awaiting("KC9GHZ")
        _feed(hs, _load_48k(path))
        if hs.unattributed:
            named[path.name] = [a.payload for a in hs.unattributed]
    assert not named, (
        f"the route names {sum(len(v) for v in named.values())} answers in audio "
        f"that holds none: {named}. Either the corpus has changed or the bar has.")


@corpora.requires_onair_unanswered_call
def test_the_preamble_alone_would_have_taken_the_one_burst_the_generator_rejects():
    """The specimen the rejection is for.

    At 44.888 s of the 2026-08-19 05:12z call — the recording kept precisely
    because nobody answered it — something holds seven comparable preamble symbols
    exact over forty consecutive alignments. Its payload reads five to ten tones
    that change from one alignment to the next and land on no generator state at
    any of them, so it is not a connect-response and this route says nothing about
    it. A preamble-only rule would have reported it, which is the phantom this
    package has already paid for once.
    """
    hs = _awaiting("KC9GHZ")
    _feed(hs, _load_48k(corpora.ONAIR_UNANSWERED_CALL))
    assert not hs.unattributed, hs.unattributed
    assert hs.step == _I_CR_SENT


# --------------------------------------------------------------------------- #
# A burst addressed to somebody the panel does hold.
def _answer_from(call: str, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(6 * FS) * 0.02
    burst = MK.synth_burst(call, RESP) * 0.08
    x[2 * FS:2 * FS + len(burst)] += burst
    return band_limit(x)


def test_an_answer_addressed_to_another_panel_station_is_named_not_taken():
    """The case requirement 3 covers where there IS a name: another gateway
    answering somebody else's call in our turnaround. It is reported with that
    gateway's callsign on it, and it still brings no link up."""
    hs = _awaiting("KC9GHZ", panel=["KB9MMT", "NS0A", "KO2F"])
    _feed(hs, _answer_from("NS0A"))
    assert len(hs.unattributed) == 1
    a = hs.unattributed[0]
    assert (a.best_call, a.best_tones) == ("NS0A", RESP.n_payload)
    assert a.states == VF.payload_states(VF.payload_bins("NS0A", RESP), RESP)
    assert hs.step == _I_CR_SENT and hs.state is VaraState.CONNECTING


def test_the_answer_we_dialled_for_never_reaches_this_route():
    """One answer is not reported twice. The callsign route takes it, and this one
    is only reached where that route declined."""
    hs = _awaiting("NS0A", panel=["NS0A"])
    _feed(hs, _answer_from("NS0A"))
    assert hs.step == _I_LINKSETUP_SENT
    assert hs.unattributed == []


def test_an_empty_panel_still_ends_the_silence():
    """A driver with no gateway list to hand reports the burst by its tones. The
    panel names the station where it can; it is not what finds the answer."""
    hs = _awaiting("KC9GHZ", panel=())
    _feed(hs, _answer_from("KB9MMT"))
    assert len(hs.unattributed) == 1
    assert hs.unattributed[0].best_call == ""
    assert hs.unattributed[0].heard == RESP.n_payload
