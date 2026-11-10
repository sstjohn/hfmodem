# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF
from hfmodem.tests.kestrel.test_bw500_peer_offset import measured, offset_audio, stream
from hfmodem.tests.kestrel.test_idle_poll import clock as clock
from hfmodem.tests.kestrel.test_responder_wide_confirm import acquired
from hfmodem.tests.kestrel.test_weak_setup_1023 import IO
from hfmodem.tests.kestrel.test_wide_link_setup_stream import CALLER, CALLED


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('hz', [-50., 50.])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_request_measures_the_caller_before_link_setup(bw, hz, route):
    io = IO()
    hs = A.VaraStationHandshake([CALLED], io, bw=bw, high_speed=True)
    hs.listen(True)
    request = offset_audio(M.synth_burst(CALLED, F.connect_request(bw)), hz)
    request += np.random.default_rng(430).normal(0, .001, len(request))
    stream(hs, request)
    assert hs._awaiting_link_setup(), io.logs
    assert hs._peer_hz == pytest.approx(hz, abs=M.carrier_to_hz(1))
    setup = offset_audio(OF.link_setup_tx(CALLER, bw=bw), hz)
    setup += np.random.default_rng(431).normal(0, .001, len(setup))
    stream(hs, setup) if route == 'stream' else hs.on_rx_audio(setup)
    assert hs.state == A.VaraState.CONNECTED and hs.caller == CALLER, io.logs


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('hz', [-50., 50.])
def test_offset_session_confirm_on_receive_stream(bw, hz):
    hs, io = acquired(bw, immediate=b'greeting')
    hs.called = CALLED if hz < 0 else {'2300': 'K2AA', '2750': 'K5AA'}[bw]
    measured(hs, hz)
    kind = F.for_bw(F.SESSION_CONFIRM, bw)
    tones = F.handshake_tones(hs.called, kind)
    lo, hi = M.band_for(bw)
    edge = min(tones[3:7]) if hz < 0 else max(tones[3:7])
    assert not lo <= edge + hs._peer_shift <= hi
    x = offset_audio(np.r_[np.zeros(4800), M.synth_burst(hs.called, kind)], hz)
    x += np.random.default_rng(432).normal(0, .001, len(x))
    stream(hs, x)
    assert any('rx session-confirm' in m and 'answering with responder control' in m
               for m in io.msgs), io.msgs
    assert not hs._awaiting_caller_confirm and hs._acked_at is None
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None
    assert len(io.sent) == 2


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('shift', [-2, 0, 2])
def test_read_band_covers_only_the_needed_extension(bw, shift):
    lo, hi = M.band_for(bw)
    peer = F.ALPHABETS[bw].carriers
    assert M.band_for(bw, shift) == (min(lo, min(peer) + shift),
                                   max(hi, max(peer) + shift))
    hs, _ = acquired(bw)
    measured(hs, M.carrier_to_hz(shift))
    assert hs._band == M.band_for(bw, shift)
    if bw == '500' or shift == 0:
        assert hs._band == (lo, hi)


@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_offset_reestimate_keeps_power_columns_until_next_scan(monkeypatch, route):
    hs, _ = acquired('2300')
    measured(hs, -50.)
    band = hs._band
    pairs = F.control_bursts(hs.caller, hs.bw)[0]
    x = offset_audio(M.synth_tone_pairs(pairs), -50.)

    def scan(samples, **kwargs):
        power = hs._cached_power(samples, hs._band, hs._grid)
        before = A._frame_fit(power, pairs, hs._band, -2)
        hs._peer_offset = None
        hs._note_peer_offset(np.zeros(32), np.arange(32), 2.)
        assert hs._peer_shift == 2 and hs._band == band
        assert A._frame_fit(power, pairs, hs._band, -2) == before

    monkeypatch.setattr(hs, '_on_rx_stream' if route == 'stream' else '_read_burst', scan)
    hs.on_rx_stream(x) if route == 'stream' else hs.on_rx_audio(x)
    assert hs._band == M.band_for('2300', 2)


def test_listening_and_a_new_contact_do_not_keep_the_old_band():
    hs, _ = acquired('2750')
    measured(hs, -50.)
    hs._narrow_to('2300')
    assert hs._band == M.band_for('2300', -2)
    hs.state = A.VaraState.DISCONNECTED
    hs.listen(True)
    assert hs.bw == '2750' and hs._band == M.band_for('2750')
    stream(hs, offset_audio(M.synth_burst(hs.called, F.connect_request('2750')), 0.))
    assert hs._awaiting_link_setup() and hs._peer_shift == 0
    assert hs._band == M.band_for('2750')
