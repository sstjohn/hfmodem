# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The BW500 handshake bursts, against the tones a real VARA emitted.

Ground truth: five VARA HF 4.9.0 -> VARA HF 4.9.0 BW500 connects driven on the
Wine bench 2026-08-15, both cables recorded, every keying labelled by the PTT
ledger. The tones below are what ``vara_mfsk.lock_preamble`` +
``demod_tones`` read off those recordings — the modem's own symbols, not a round
trip through kestrel's encoder.

Four distinct called callsigns is the whole test. Each burst's payload pins one
24-bit generator state out of 2**24, but the seeding map is many-to-one: W1AW
alone leaves 9 candidate ``(SEED_OFF, PREADV)`` pairs for the request and 9 for
the response, and only the intersection across all four is a singleton. The two
W1AW sessions were run separately and read tone-identical, which is what a burst
keyed to the CALLED station does.
"""
from __future__ import annotations

import pytest

from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

# called callsign -> (connect-request payload, connect-response payload)
_BENCH = {
    "W1AW": (
        [60, 72, 58, 58, 52, 68, 58, 72, 66, 52, 52, 52, 68, 68, 52, 60, 64, 50,
         72, 62, 52, 64, 70, 68, 72, 54, 58, 76, 72, 58, 76],
        [54, 66, 74, 52, 68, 54, 76, 68, 62, 58, 54, 64, 58, 56, 68]),
    "N0DX": (
        [54, 62, 64, 50, 64, 58, 72, 58, 68, 68, 50, 52, 58, 54, 60, 52, 76, 54,
         66, 64, 74, 58, 68, 66, 56, 54, 74, 66, 52, 62, 68],
        [62, 70, 68, 50, 54, 74, 50, 52, 50, 52, 58, 54, 72, 52, 74]),
    "W2XY": (
        [50, 54, 68, 72, 64, 66, 64, 64, 74, 56, 58, 62, 54, 76, 72, 64, 58, 62,
         64, 76, 58, 56, 62, 50, 62, 52, 50, 56, 68, 70, 64],
        [68, 70, 52, 70, 70, 54, 54, 50, 56, 70, 60, 68, 52, 58, 74]),
    "KC2OUR": (
        [56, 58, 70, 56, 64, 70, 58, 72, 64, 54, 58, 62, 72, 68, 50, 68, 60, 70,
         76, 66, 64, 66, 62, 58, 64, 68, 52, 76, 56, 66, 62],
        [56, 60, 58, 50, 76, 74, 56, 54, 64, 66, 54, 54, 66, 72, 72]),
}
_KINDS = {"request": (VF.CR500, 0), "response": (VF.CONNECT_RESPONSE_500, 1)}


@pytest.mark.parametrize("called", sorted(_BENCH))
@pytest.mark.parametrize("which", sorted(_KINDS))
def test_the_generator_reproduces_the_modems_own_tones(called, which):
    kind, i = _KINDS[which]
    assert VF.payload_bins(called, kind) == _BENCH[called][i]


@pytest.mark.parametrize("which", sorted(_KINDS))
def test_the_bw500_alphabet_is_fourteen_even_carriers(which):
    """50..76 on a 46.9 Hz lattice, against BW2300's 70 carriers on 23.4 Hz. The
    parity term that splits the BW2300 alphabet into odd and even halves is
    absent here, so no BW500 handshake burst can emit an odd carrier."""
    kind = _KINDS[which][0]
    emitted = {t for cs in _BENCH for t in VF.payload_bins(cs, kind)}
    assert emitted <= set(range(50, 78, 2))
    assert kind.tones is VF.BW500_TONES
    assert VF.BW500_TONES.carriers == frozenset(range(50, 77, 2))


@pytest.mark.parametrize("which", sorted(_KINDS))
def test_the_preamble_is_the_bw2300_one(which):
    """Only the payload alphabet narrows; the fixed preamble is shared, which is
    why ``lock_preamble`` finds a BW500 burst with the BW2300 descriptor."""
    kind = _KINDS[which][0]
    wide = VF.CR if kind is VF.CR500 else VF.CONNECT_RESPONSE
    assert kind.preamble == wide.preamble and kind.n_payload == wide.n_payload


@pytest.mark.parametrize("called", sorted(_BENCH))
def test_a_bw500_request_round_trips_through_the_mfsk_path(called):
    tones = VF.handshake_tones(called, VF.CR500)
    assert MK.demod_burst(MK.synth_tones(tones), VF.CR500) == tones


def test_the_two_bandwidths_do_not_answer_to_each_other():
    """A BW2300 responder is measured not to answer the BW500 request and the
    reverse; the descriptors must not be able to score each other either."""
    for called in _BENCH:
        for narrow, wide in ((VF.CR500, VF.CR),
                             (VF.CONNECT_RESPONSE_500, VF.CONNECT_RESPONSE)):
            assert not VF.recognize(VF.payload_bins(called, narrow), called, wide)
            assert not VF.recognize(VF.payload_bins(called, wide), called, narrow)
