# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Unavailable spectral SN must not become a numerical-floor signal report."""
import numpy as np
import pytest

from hfmodem.kestrel.rx.cq_stream import CQDetector, quality, beacon_quality
from hfmodem.kestrel.rx import beacon as B, varahf500 as rx
from hfmodem.kestrel.rx.cq import build_frame, parse_frame
from hfmodem.kestrel.rx.broadcast import compact, parse_compact
from hfmodem.contact.broadcast import Broadcast, decode_ui
from hfmodem.kestrel.tx.beacon import synth_beacon
from hfmodem.kestrel.tx.cq import synth_cq
from hfmodem.contact.slots import Mixer
from .test_beacon import host
from .test_cq import replay


@pytest.mark.parametrize('x', [np.zeros(2048), np.ones(512),
                              np.full(2048, np.nan), np.full(2048, np.inf)])
def test_unmeasurable_quality_is_unavailable(x):
    assert quality(x, 1500) is None


def test_valid_quality_is_gain_invariant():
    rng = np.random.default_rng(86)
    t = np.arange(12000) / 12000
    x = np.cos(2 * np.pi * 1500 * t) + rng.normal(0, .1, len(t))
    sn = quality(x, 1500)
    assert 10 < sn < 20
    assert quality(.001 * x, 1500) == pytest.approx(sn)


def test_unavailable_sn_preserves_cq_and_broadcast_payloads():
    cq = parse_frame(build_frame('K5CG', 500), sn=None)
    assert cq.call == 'K5CG' and cq.sn is None
    session, modem, commands, port = host()
    session.listen_cq = True
    session.modem_cqframe(cq.call, str(cq.bandwidth), cq.sn)
    assert commands[-2:] == ['SN --', 'CQFRAME K5CG 500']
    packet = Broadcast(1, 'K5CG', 'W9SSJ', b'hello', sn=None)
    assert parse_compact(compact(packet), sn=None) == packet
    assert decode_ui(packet.encode(), sn=None) == packet


@pytest.mark.parametrize('offset', [-140, 0, 140])
@pytest.mark.parametrize('iq', [False, True])
def test_decoded_beacon_with_flank_interference_has_measured_sn(offset, iq):
    from .test_beacon import shifted
    from scipy.signal import hilbert
    air = synth_beacon('20OEM13PBK5CG', amplitude=.1)
    x = np.r_[np.zeros(24000), shifted(air, 48000, offset), np.zeros(48000)]
    t = np.arange(len(x)) / 48000
    x += np.random.default_rng(79).normal(0, .03, len(x))
    def receive(audio):
        if iq:
            audio = hilbert(audio) * np.exp(-2j * np.pi * 1500 * t)
        return replay(CQDetector(), audio)

    baseline = receive(x)
    # Outside the beacon tones but inside the estimator's noise reference.
    x += .5 * np.cos(2 * np.pi * 2400 * t)
    got = receive(x)
    assert [r.text for r in got] == ['20OEM13PBK5CG']
    assert got[0].sn == pytest.approx(baseline[0].sn, abs=.6)
    assert got[0].sn == pytest.approx(10 * np.log10(.1**2 / .03**2), abs=1.5)
    session, modem, commands, port = host()
    session.modem_beacon(got[0])
    assert commands == [f'SN {got[0].sn:.1f}']
    assert port.delivered == ['20OEM13PBK5CG'.ljust(20)]


@pytest.mark.parametrize('sn', [-15, -5, 5, 15, 25])
@pytest.mark.parametrize('offset', [-12, 0, 12])
def test_reference_estimator_against_known_noise(sn, offset):
    # Independent known reference tones with real AWGN. At 48 kHz, noise
    # referred to 12 kHz is half the full real-sample noise power.
    r = B.RECORD
    refs, tones = rx._l3_tables(B.LEVEL)[:2]
    t = np.arange(r.dw) / 48000
    x = np.zeros((r.ncols, r.dw))
    x[refs] = np.cos(2 * np.pi * (tones[:, None] * 48000 / r.dw + offset) * t)
    estimates = []
    for seed in range(24):
        noisy = x + np.random.default_rng(seed).normal(0, 10**(-sn / 20), x.shape)
        estimates.append(beacon_quality(noisy.ravel(), 0))
    errors = np.array(estimates) - sn
    # Assess bias and single-frame variation over independent noise trials,
    # rather than treating one draw of 24 reference symbols as exact.
    assert abs(errors.mean()) < .4
    assert np.sqrt(np.mean(errors**2)) < 1.2
    assert beacon_quality(noisy.ravel() * .001, 0) == pytest.approx(estimates[-1])


def test_interference_inside_the_comb_counts_against_sn():
    air = synth_beacon('20OEM13PBK5CG', amplitude=.1)
    start = 32 * 2048 + 1536 + B.RECORD.lead * B.RECORD.dw
    x = air + np.random.default_rng(23).normal(0, .01, len(air))
    baseline = beacon_quality(x, start)
    t = np.arange(len(x)) / 48000
    interfered = beacon_quality(x + .04 * np.cos(2 * np.pi * 1406.25 * t), start)
    assert baseline > 15
    assert interfered is None or interfered < baseline - 6


def test_beacon_estimator_needs_a_complete_measurable_reference_set():
    assert beacon_quality(np.zeros(1024), 0) is None
    assert beacon_quality(np.zeros(B.RECORD.ncols * B.RECORD.dw), 0) is None
    assert beacon_quality(np.full(B.RECORD.ncols * B.RECORD.dw, np.nan), 0) is None


@pytest.mark.parametrize('offset', [-140, 0, 140])
@pytest.mark.parametrize('sigma', [.003, .03, .15])
@pytest.mark.parametrize('iq', [False, True])
def test_cq_sn_survives_the_passive_cf_filter(offset, sigma, iq):
    from scipy.signal import hilbert
    from .test_beacon import shifted
    air = synth_cq('AB4EJ-9', 500, amplitude=.1)
    x = np.r_[np.zeros(24000), shifted(air, 48000, offset), np.zeros(48000)]
    x += np.random.default_rng(79).normal(0, sigma, len(x))
    filtered = Mixer(1500)(x)
    if iq:
        rotation = np.exp(-2j * np.pi * 1500 * np.arange(len(x)) / 48000)
        x, filtered = hilbert(x) * rotation, hilbert(filtered) * rotation
    direct, passive = [replay(CQDetector(), y) for y in (x, filtered)]
    assert [r.call for r in direct] == [r.call for r in passive] == ['AB4EJ-9']
    # Body amplitude differs from the header by the measured stock factor.
    expected = 20 * np.log10(.1 * (.270 / .353) / sigma)
    assert direct[0].sn == pytest.approx(expected, abs=1.5)
    assert passive[0].sn == pytest.approx(direct[0].sn, abs=1.5)
