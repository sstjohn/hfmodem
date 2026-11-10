# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.kestrel.vara import vara_arq as A, vara_ofdm as OF, vara_mfsk as M, vara_frames as F
from hfmodem.tests.kestrel.test_bw500_pairing import _build, _receiver
from hfmodem.tests.kestrel.test_weak_setup_1023 import awaiting_setup, answerer


def offset_audio(x, hz):
    x = np.pad(x, (0, 2 * rx.FS))
    return (hilbert(x) * np.exp(2j * np.pi * hz * np.arange(len(x)) / rx.FS)).real


def measured(hs, hz):
    bins = hz / M.carrier_to_hz(1)
    hs._peer_shift = round(bins)
    hs._peer_offset = bins - hs._peer_shift


def stream(hs, x):
    for at in range(0, len(x), 4800):
        hs.on_rx_stream(x[at:at + 4800])


@pytest.mark.parametrize('hz', [-47.5, 47.5])
def test_offset_data_on_receive_stream(hz):
    hs, io = _receiver('initiator', 4)
    measured(hs, hz)
    frame = _build(4)
    x = offset_audio(tx.synth_burst(frame, onset=24000), hz)
    x += np.random.default_rng(420).normal(0, .001, len(x))
    stream(hs, x)
    assert io.host == [bytes([31]) * rx._PAYLOAD[4]], io.msgs


@pytest.mark.parametrize('hz', [-47.5, 47.5])
def test_offset_link_setup_on_receive_stream(hz):
    hs, io = awaiting_setup()
    measured(hs, hz)
    x = offset_audio(OF.link_setup_tx('KD2WNY', bw='500', level=4, ask=5,
                                    high_speed=True, compression='TEXT'), hz)
    x += np.random.default_rng(421).normal(0, .001, len(x))
    stream(hs, x)
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert hs.caller == 'KD2WNY'


@pytest.mark.parametrize('hz', [-47.5, 47.5])
def test_request_measures_the_caller_before_link_setup(hz):
    hs, io = answerer()
    request = offset_audio(M.synth_burst('W9SSJ', F.for_bw(F.connect_request('500'), '500')), hz)
    stream(hs, request)
    assert hs._awaiting_link_setup(), io.logs
    assert hs._peer_hz == pytest.approx(hz, abs=.5)
    setup = offset_audio(OF.link_setup_tx('KD2WNY', bw='500'), hz)
    stream(hs, setup)
    assert hs.state == A.VaraState.CONNECTED, io.logs


@pytest.mark.parametrize('hz', [-47.5, 47.5])
def test_offset_batch_readers(hz):
    hs, _ = _receiver('initiator', 4)
    measured(hs, hz)
    frame = _build(4)
    x = offset_audio(tx.synth_burst(frame, onset=24000), hz)
    read = phy.decode_level(x, '500', 4, hz=hz, expect_pair=False)
    assert read is not None and read.frames == (frame[:-2],)
    assert hs._peer_data_over_500(x, len(x) / rx.FS) == [frame[:-2]]
    hs, _ = awaiting_setup()
    measured(hs, hz)
    setup = offset_audio(OF.link_setup_tx('KD2WNY', bw='500'), hz)
    assert hs._rx_link_setup(setup)


def test_streaming_shift_is_independent_of_callback_sizes():
    from hfmodem.kestrel.rx.bw500_shift import PeerShift

    x = np.random.default_rng(422).normal(size=40000)
    outputs = []
    for sizes in ([480], [1024], [4800], [17, 2301, 1, 4800, 937]):
        shift = PeerShift()
        chunks, at, i = [], 0, 0
        for stop, hz in ((15001, -47.5), (27003, -51.2), (len(x), 49.1)):
            shift.hz = hz
            while at < stop:
                end = min(stop, at + sizes[i % len(sizes)])
                chunks.append(shift.feed(x[at:end]))
                at, i = end, i + 1
        outputs.append(np.concatenate(chunks))
    for output in outputs[1:]:
        np.testing.assert_allclose(output, outputs[0], rtol=0, atol=1e-9)


@pytest.mark.parametrize('hz', [-10., 10.])
def test_tolerated_offsets_keep_the_original_audio(hz):
    from hfmodem.kestrel.rx import bw500_shift

    frame = _build(4)
    x = offset_audio(tx.synth_burst(frame, onset=24000), hz)
    assert bw500_shift.retune(x, hz) is x
    read = phy.decode_level(x, '500', 4, hz=hz, expect_pair=False)
    assert read is not None and read.frames == (frame[:-2],)


def test_reestimate_waits_until_the_next_receive_search():
    hs, io = _receiver('initiator', 4)
    measured(hs, 47.5)
    x = offset_audio(tx.synth_burst(_build(4), onset=24000), 47.5)
    stream(hs, x[:96000])
    front = hs._base_front
    phase = hs._base_shift.phase
    measured(hs, 51.)
    hs.on_rx_stream(x[96000:97024])
    assert hs._base_front is front and hs._base_shift.hz == pytest.approx(47.5)
    assert hs._base_shift.phase != phase
    stream(hs, x[97024:])
    assert io.host == [bytes([31]) * rx._PAYLOAD[4]], io.msgs
    hs.on_rx_stream(np.zeros(1024))
    assert hs._base_shift.hz == pytest.approx(51.)


def test_frequency_update_preserves_phase():
    from hfmodem.kestrel.rx.bw500_shift import PeerShift, _ANALYTIC
    from scipy.signal import lfilter

    x = np.random.default_rng(423).normal(size=4000)
    shift = PeerShift()
    shift.hz = -47.5
    first = shift.feed(x[:1501])
    shift.hz = 51.2
    second = shift.feed(x[1501:])
    phase = -2 * np.pi / rx.FS * np.concatenate((
        -47.5 * np.arange(1501), -47.5 * 1501 + 51.2 * np.arange(2499)))
    expected = (lfilter(_ANALYTIC, [1.], x) * np.exp(1j * phase)).real
    np.testing.assert_allclose(np.concatenate((first, second)), expected, atol=1e-12)


def noisy_offset(burst, noise=2.7):
    x = offset_audio(burst / np.max(abs(burst)), 47.5)
    return x + np.random.default_rng(424).normal(0, noise, len(x))


@pytest.mark.parametrize('expected', [4, 5])
def test_batch_quality_and_gearshift_use_unfiltered_audio(expected):
    burst = tx.synth_burst(_build(4), onset=24000)
    x = noisy_offset(burst)
    read = phy.decode_level(x, '500', expected, hz=47.5, expect_pair=False)
    assert read is not None and read.frames == (_build(4)[:-2],)
    sn = rx.channel_snr(read.quality)
    raw = rx.channel_snr(rx.quality(x[24000:len(burst)]))
    assert sn == pytest.approx(raw, abs=.1)
    assert -4.1 < sn < -3.5
    assert A._irs_frame(4, sn, phy.RX_CEILING['500'], '500') == 0


@pytest.mark.parametrize('reader', ['batch', 'stream'])
def test_data_reports_raw_snr_and_holds_level_four(reader):
    hs, io = _receiver('initiator', 4)
    hs._link_high_speed = True
    measured(hs, 47.5)
    reports = []
    io.snr = reports.append
    burst = tx.synth_burst(_build(4), onset=24000)
    x = noisy_offset(burst)
    if reader == 'batch':
        assert hs._answer_data_over(x)
    else:
        stream(hs, x)
    assert io.host == [bytes([31]) * rx._PAYLOAD[4]], io.msgs
    assert reports
    assert rx.channel_snr(reports[-1]) == pytest.approx(
        rx.channel_snr(rx.quality(x[24000:len(burst)])), abs=.2)
    assert hs._expect_level == 4


@pytest.mark.parametrize('reader', ['batch', 'stream'])
def test_setup_reports_unfiltered_snr(reader):
    hs, io = awaiting_setup()
    measured(hs, 47.5)
    reports = []
    io.snr = reports.append
    burst = OF.link_setup_tx('KD2WNY', bw='500')
    x = noisy_offset(burst, noise=.7)
    if reader == 'batch':
        assert hs._rx_link_setup(x)
    else:
        stream(hs, x)
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert reports
    assert rx.channel_snr(reports[-1]) == pytest.approx(
        rx.channel_snr(rx.quality(x[:len(burst)])), abs=.2)


def test_rearmed_responder_measures_a_new_caller_without_the_old_contact():
    hs, io = answerer()
    request = M.synth_burst('W9SSJ', F.for_bw(F.connect_request('500'), '500'))
    setup = OF.link_setup_tx('KD2WNY', bw='500')
    stream(hs, offset_audio(request, 55.))
    assert hs._awaiting_link_setup(), io.logs
    assert hs._peer_hz == pytest.approx(55., abs=2.)
    retry = M.synth_burst('W9SSJ', F.for_bw(F.connect_request_retry('500'), '500'))
    stream(hs, offset_audio(retry, 45.))
    assert hs._peer_hz == pytest.approx(50., abs=2.)
    stream(hs, offset_audio(setup, 55.))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    hs.disconnect()
    hs.listen(True)
    stream(hs, offset_audio(request, 0.))
    assert hs._awaiting_link_setup(), io.logs
    assert hs._peer_hz == pytest.approx(0., abs=.5)
    stream(hs, offset_audio(setup, 0.))
    assert hs.state == A.VaraState.CONNECTED, io.logs


@pytest.mark.parametrize('armed', [False, True])
def test_unused_shift_is_not_run(monkeypatch, armed):
    from hfmodem.kestrel.rx import bw500_shift

    hs, _ = _receiver('initiator', 4) if armed else answerer()
    measured(hs, 0. if armed else 47.5)

    def unused(*args):
        pytest.fail('the shifter output is unused in this receive search')

    monkeypatch.setattr(bw500_shift.PeerShift, 'feed', unused)
    hs.on_rx_stream(np.zeros(4800))


@pytest.mark.parametrize('setup', [False, True])
def test_stream_quality_window_removes_the_filter_delay(monkeypatch, setup):
    from hfmodem.kestrel.rx import bw500_locator, bw500_shift

    hs, _ = awaiting_setup() if setup else _receiver('initiator', 4)
    measured(hs, 47.5)
    burst = (OF.link_setup_tx('KD2WNY', bw='500') if setup
             else tx.synth_burst(_build(4), onset=24000))
    x = noisy_offset(burst, noise=.7)
    windows, overs = [], []
    quality, feed = rx.quality, bw500_locator.Receiver.feed

    def measure(samples):
        windows.append(samples.copy())
        return quality(samples)

    def locate(receiver, *args, **kwargs):
        found = feed(receiver, *args, **kwargs)
        overs.extend(found)
        return found

    monkeypatch.setattr(rx, 'quality', measure)
    monkeypatch.setattr(bw500_locator.Receiver, 'feed', locate)
    stream(hs, x)
    over = next(o for o in overs if any(f.crc_ok for f in o.frames))
    lead, tail = (0, 0) if setup else (A._OVER500_LEAD, A._OVER500_TAIL)
    start = max(0, over.candidate.start - bw500_shift.DELAY - lead)
    end = over.end - bw500_shift.DELAY + tail
    assert any(np.array_equal(window, x[start:end]) for window in windows)
