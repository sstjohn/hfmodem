# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The r6 default-axis self-pair followed a peer's changeover head at -12.7 Hz
and keyed every later burst there. The peer's reader at 0 Hz read our next CS3
head with 5-7 bit errors and never recovered.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, placement, rx, spec
from hfmodem.tests.shrike.test_follow_p1_cross_check import sessrx

LEAD = 2400


@pytest.fixture(params=["axis", "diagonal"])
def phase(request, monkeypatch):
    monkeypatch.setattr(placement, "CONTROL_PHASE", request.param)


def _changeover():
    audio = onair._trim_silence(
        placement.changeover_packet(b"[HF", 0, swapped=False)) * .5
    return np.concatenate([np.zeros(LEAD), audio, np.zeros(LEAD)])


def _follow(carrier=0.0):
    audio = _changeover()
    if carrier:
        audio = p3acquire.compensate(audio, -carrier)
    s = sessrx()
    got = s._p3_changeover_packet(audio)
    assert got is not None and got[0].cs == placement.BREAKIN_CS
    return s, got[0]


def test_changeover_follow_keeps_the_read_and_keys_the_carrier(phase):
    s, _ = _follow()
    assert s.p3_receive_offset_hz == pytest.approx(-12.7, abs=.3)
    assert abs(s.p3_transmit_offset_hz) <= .3 + 1e-12


def test_changeover_follow_keeps_a_real_carrier_offset(phase):
    s, _ = _follow(-6.0)
    assert s.p3_receive_offset_hz == pytest.approx(-18.7, abs=.3)
    assert s.p3_transmit_offset_hz == pytest.approx(-6.0, abs=.3)


def test_next_changeover_reads_at_the_peers_zero_hertz(phase):
    s, ev = _follow()
    tx = SimpleNamespace(p3_follow_offset="all", sessrx=s, _p3_shifted={},
                         _tx_offset_hz=None)
    keyed = onair.RadioTx._p3_offset(tx, _changeover(), control=True)
    reads = [rx.decode_control_signal(keyed, start)
             for start in range(ev.start - spec.SAMPLE_RATE // 100,
                                ev.start + spec.SAMPLE_RATE // 100 + 1, 24)]
    best = min(reads, key=lambda read: read[1])
    assert best == (placement.BREAKIN_CS, 0)


def test_bare_axis_cs1_still_keys_at_zero(monkeypatch):
    monkeypatch.setattr(placement, "CONTROL_PHASE", "axis")
    burst = placement.control_burst(arq.CS_ACK, tail=False, swapped=False)
    audio = np.concatenate([np.zeros(LEAD), burst, np.zeros(LEAD)])
    got = p3acquire.control_signal(audio)
    assert got is not None and got.event.cs == arq.CS_ACK
    s = sessrx()
    for cycle in range(2):
        s.new_cycle()
        s._p3_head(got.event, got.event.start + cycle * round(
            s.host.arq.cycle_s * spec.SAMPLE_RATE), got.offset_hz)
    assert s._p3_offset_fixed
    assert s.p3_receive_offset_hz == pytest.approx(0.0, abs=.3)
    assert s.p3_transmit_offset_hz == pytest.approx(0.0, abs=.3)


@pytest.mark.parametrize("candidate_cs,hz,cs,transmit_hz", [
    (arq.CS_ACK, 0.0, placement.BREAKIN_CS, 0.0),
    (placement.BREAKIN_CS, -12.7, arq.CS_ACK, -.2),
])
def test_corroboration_follows_the_candidates_word_convention(
        monkeypatch, candidate_cs, hz, cs, transmit_hz):
    monkeypatch.setattr(placement, "CONTROL_PHASE", "axis")
    raw = (placement.changeover_packet(b"[HF", 0, swapped=False)
           if cs == placement.BREAKIN_CS else
           placement.control_burst(cs, tail=False, swapped=False))
    head = LEAD + placement.pulse_lead(raw)
    audio = np.pad(onair._trim_silence(raw), (LEAD, LEAD))[
        :head + round(.23 * spec.SAMPLE_RATE)]
    s = sessrx()
    s._p3_head_candidate = (0, head, hz, candidate_cs)
    s._p3_acquisition_cycle = 1
    cycle_n = round(s.host.arq.cycle_s * spec.SAMPLE_RATE)
    ev = s._p3_corroborate(audio, cycle_n, head)
    assert ev is not None and ev.cs == cs
    assert s._p3_offset_fixed
    assert s.p3_receive_offset_hz == hz
    assert s.p3_transmit_offset_hz == pytest.approx(transmit_hz)
