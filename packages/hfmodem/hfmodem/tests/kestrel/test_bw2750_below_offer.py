# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""K7EK-10 lower-speed offers, plus near-miss rejection at the tone gate."""
import collections

import numpy as np

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_response_by_stream import _IO, _load_48k


def _awaiting(called="K7EK-10", bw="2750"):
    io = _IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw=bw)
    hs.role, hs.called, hs.caller = "initiator", called, "W9SSJ"
    hs.state, hs.step = VA.VaraState.CONNECTING, VA._I_CR_SENT
    return hs, io


def _drive(hs, x):
    for i in range(0, len(x), 4800):
        hs.on_rx_stream(x[i:i + 4800])


def test_lattice_has_the_2750_below_offer_positions():
    assert VA._ANSWER_LATTICE_BY_BW["2750"] == (0, 1, 2)


@corpora.requires_onair_below_offer_2750
def test_k7ek_below_offer_answers_are_named_at_positions_one_and_two():
    hs, io = _awaiting()
    _drive(hs, _load_48k(corpora.ONAIR_BELOW_OFFER_2750))
    assert hs.answers, "the below-offer answers were named nothing"
    positions = collections.Counter(a.position for a in hs.answers)
    assert set(positions) == {1, 2}
    assert all(a.shift == 0 for a in hs.answers)
    assert hs.state == VA.VaraState.CONNECTING       # setup sent, but no confirmation
    assert any("BW2750 connect-response" in m and "level" in m
               for m in io.log_lines)


@corpora.requires_onair_below_offer_2750
def test_every_below_offer_answer_belongs_to_the_dialled_station():
    hs, _ = _awaiting()
    _drive(hs, _load_48k(corpora.ONAIR_BELOW_OFFER_2750))
    kind = VF.connect_response("2750")
    for a in hs.answers:
        assert VF.payload_position(a.payload, "K7EK-10", kind) == a.position


def _near_miss_offer(called="K7EK-10", bw="2750", drop=3):
    kind = VF.connect_response(bw)
    tones = list(VF.handshake_tones(called, kind))
    npre = len(kind.preamble)
    lo, hi = MK.band_for(bw)
    for j in range(npre + 2, npre + 2 + 2 * drop, 2):
        t = tones[j]
        tones[j] = t + 3 if t + 3 <= hi else t - 3
    burst = MK.synth_tones(tones)
    return np.concatenate([np.zeros(int(0.2 * MK.FS)), burst,
                           np.zeros(int(0.5 * MK.FS))])


def test_offer_one_tone_short_logs_a_near_miss_once():
    hs, io = _awaiting()
    _drive(hs, _near_miss_offer(drop=3))
    near = [m for m in io.log_lines if "short of confirming the response" in m]
    assert len(near) == 1, io.log_lines
    assert "keyed the connect-response" in near[0]
    assert hs.state == VA.VaraState.CONNECTING       # a near-miss is not a connect
    assert not hs.answers


def test_a_clean_offer_is_taken_by_the_accept_route_and_is_no_near_miss():
    """The near-miss line must not stand in front of a real offer: the accept
    route takes it, the link-setup goes out, and nothing reports a shortfall."""
    hs, io = _awaiting()
    kind = VF.connect_response("2750")
    burst = MK.synth_tones(VF.handshake_tones("K7EK-10", kind))
    seg = np.concatenate([np.zeros(int(0.2 * MK.FS)), burst,
                          np.zeros(int(0.5 * MK.FS))])
    _drive(hs, seg)
    assert not any("short of confirming the response" in m for m in io.log_lines)
    assert any("connect-response for K7EK-10 found" in m for m in io.log_lines)
    assert hs.step == VA._I_LINKSETUP_SENT      # the offer was taken
    assert any("tx link-setup" in m for m in io.log_lines)
    assert not hs.answers                       # found by the full-response search
