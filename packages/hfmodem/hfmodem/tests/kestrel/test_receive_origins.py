# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Transmit-referenced receive coordinates, independent of ARQ delivery."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import bw500_locator as locator, bw500_shift as shift
from hfmodem.kestrel.rx import varahf500 as rx, varaofdm as ofdm, varatone as tone
from hfmodem.kestrel.rx import varahf2300 as wide
from hfmodem.kestrel.rx.observation import CaptureContext, FrameOrigin, LocatedFrame
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.kestrel.vara import vara_arq as arq

STRIDES = {1: 253952, 2: 233472, 4: 201728}
# For _pair(4)'s measured-pulse waveform, the fitted sub-band centres are
# onset + 38 and onset + 290. Normalize the latter's half-hop stagger before
# averaging, as required by the receive coordinate convention. These fits
# differ from the pulse centres placed at onset and onset + H/2 by synth_burst.
LEVEL4_SYNTH_FITTED_ORIGIN_OFFSET = round((38 + (290 - tx.H // 2)) / 2)


def _pair(level, onset=12000):
    if level == 4:
        frames = [tx.build_frame(bytes([k + 1]) * 43, 0x99) for k in range(2)]
        columns = 9 + 2 * 394
        audio = tx.synth_burst(frames, onset=onset, ncol=columns)
        stop = onset + columns * 512
    else:
        r = rx.INDEX_RECORDS[level]
        frames = [tx.build_l3_frame(bytes([k + 1]) * r.payload, 0x99, level)
                  for k in range(2)]
        audio = tx.synth_l3_burst(frames, onset=onset, level=level)
        stop = onset + (r.lead + 2 * r.ncols) * r.dw
    return np.pad(audio, (0, 12000)), frames, stop


@pytest.fixture(scope='module', params=[1, 2, 4])
def pair(request):
    return request.param, _pair(request.param)


@pytest.mark.parametrize('crop', [0, 137, 4001])
def test_transmitted_training_origin_and_every_companion_stride(pair, crop):
    level, (audio, raw, stop) = pair
    context = CaptureContext(7, 11, 2000000 + crop, 48000)
    frames = rx.decode_burst_frames(audio[crop:], 12000 - crop, stop - crop,
                                    level, context=context)
    assert [f.frame_bytes for f in frames] == raw
    assert [f.ordinal for f in frames] == [0, 1]
    for ordinal, frame in enumerate(frames):
        origin = frame.origin
        fitted_offset = LEVEL4_SYNTH_FITTED_ORIGIN_OFFSET if level == 4 else 0
        expected = 2012000 + fitted_offset + ordinal * STRIDES[level]
        assert abs(origin.training_origin - expected) <= 16
        assert (origin.tune_epoch, origin.decoder_epoch, origin.level, origin.bandwidth_hz) == (
            7, 11, level, 500)
        support = (403 * 512 + 1 if level == 4 else
                   (rx.INDEX_RECORDS[level].lead + rx.INDEX_RECORDS[level].ncols)
                   * rx.INDEX_RECORDS[level].dw)
        assert support <= origin.end_sample - origin.training_origin <= support + 16
        assert origin.end_sample < context.sample_start + len(audio[crop:])
    assert frames[1].origin.training_origin - frames[0].origin.training_origin == STRIDES[level]


@pytest.mark.parametrize('clean_ordinal', [0, 1])
def test_only_head_or_companion_keeps_position_and_hole(pair, clean_ordinal):
    level, (audio, raw, stop) = pair
    frames = rx.decode_burst_frames(audio, 12000, stop, level,
                                    context=CaptureContext(1, 2, 5000000, 48000))
    frames = [replace(f, crc_ok=i == clean_ordinal) for i, f in enumerate(frames)]
    result = rx.DecodeResult(frames, frames)
    read = phy.LevelRead(level, observations=result.observations)
    assert read.frames == (raw[clean_ordinal][:-2],)
    assert read.located == 2 and read.lost
    assert [o.ordinal for o in read.observations] == [0, 1]
    assert read.observations[1 - clean_ordinal].body is None
    assert read.observations[1].origin.training_origin - read.observations[0].origin.training_origin == STRIDES[level]


def test_unreached_companion_has_geometry_and_no_body():
    context = CaptureContext(4, 9, 900000, 48000)
    head = rx.locate_frame(rx.FrameResult(3000, (0, 256), 9, 1., True,
                                          tx.build_frame(b'A' * 43, 0x99),
                                          frames_in_burst=2), context)
    observations = rx.DecodeResult([head], [head]).observations
    assert len(observations) == 2
    assert observations[1].ordinal == 1 and observations[1].body is None
    assert observations[1].origin.training_origin == 903000 + 201728
    assert observations[1].origin.end_sample == head.origin.end_sample + 201728


@pytest.mark.parametrize('force_full', [False, True])
def test_streaming_peer_shift_delay_and_refinement(pair, force_full, monkeypatch):
    level, (audio, raw, _) = pair
    if force_full:
        monkeypatch.setattr(locator, 'demod_local', lambda *a, **kw: ({}, (), 0.))
    peer = shift.PeerShift()
    peer.hz = 20.
    audio = (hilbert(audio) * np.exp(2j * np.pi * 20 * np.arange(len(audio)) / rx.FS)).real
    receiver = locator.Receiver(context=CaptureContext(3, 8, 7000000, 48000, shift.DELAY))
    overs = []
    delayed = np.pad(audio, (0, shift.DELAY))
    for at in range(0, len(delayed), 1003):
        overs.extend(receiver.feed(peer.feed(delayed[at:at + 1003]), may_pair=True))
    overs.extend(receiver.feed(np.empty(0), may_pair=True, eof=True))
    frames = [f for over in overs for f in over.frames]
    assert [f.frame_bytes for f in frames] == raw
    for ordinal, frame in enumerate(frames):
        assert abs(frame.origin.training_origin - (7012000 + ordinal * STRIDES[level])) <= 64
        assert frame.origin.decoder_epoch == 8
        assert frame.origin.training_origin >= receiver.context.sample_start - rx.BACKWARD_REACH


def test_segmented_retune_has_no_second_delay(pair):
    level, (audio, raw, _) = pair
    context = CaptureContext(2, 3, 4000000, 48000)
    shifted = (hilbert(audio) * np.exp(2j * np.pi * 20 * np.arange(len(audio)) / rx.FS)).real
    result = rx.decode_stream(shift.retune(shifted, 20.), expect_pair=True, context=context)
    assert [f.frame_bytes for f in result.frames] == raw
    assert abs(result.frames[0].origin.training_origin - 4012000) <= 64


@pytest.mark.parametrize('fallback', [False, True])
def test_behind_slices_advance_context_without_changing_epochs(monkeypatch, fallback):
    context = CaptureContext(12, 19, 3000000, 48000)
    audio = np.zeros(2000000)
    seen = []
    def family(samples, bw, level, *, context):
        seen.append(('family', len(samples), context))
        observations = (() if fallback else
                        (LocatedFrame(0, context.origin(400, 800, 5, 500), b'body'),))
        return tone.ToneResult(5, () if fallback else (b'body\x00\x00',),
            SimpleNamespace(snr_db=20.), lock=1.0, onset=400, end=800, observations=observations)
    monkeypatch.setattr(tone, 'decode_burst', family)
    monkeypatch.setattr(phy, 'ident', lambda x: None)
    monkeypatch.setattr(phy, '_tone_burst', lambda *a, **kwargs: True)
    def base(samples, bw, *args, context, **kwargs):
        seen.append(('base', len(samples), context))
        return phy.LevelRead(2, observations=(LocatedFrame(0, context.origin(700, 900, 2, 500), b'base'),))
    monkeypatch.setattr(phy, '_decode_base', base)
    read = phy.decode_level(audio, '500', 5, fresh=4800, under=4800, context=context)
    for _, length, capture in seen:
        assert capture == context.shifted(len(audio) - length)
    chosen = seen[-1][2]
    assert read.observations[0].origin.training_origin == chosen.sample_start + (700 if fallback else 400)
    assert read.observations[0].origin.tune_epoch == 12
    assert read.observations[0].origin.decoder_epoch == 19


def test_reset_epochs_are_distinct_and_origins_are_immutable():
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    origin = station.capture_context(1234).origin(10, 20, 4, 500)
    assert station.origin_is_current(origin)
    with pytest.raises(FrozenInstanceError):
        origin.training_origin = 0
    with pytest.raises(FrozenInstanceError):
        CaptureContext(0, 0, 0, 48000).sample_start = 1
    with pytest.raises(FrozenInstanceError):
        LocatedFrame(0, origin, None).body = b'x'
    station._key(True)
    assert station._tune_epoch == origin.tune_epoch
    assert not station.origin_is_current(origin)
    current = station.capture_context().origin(10, 20, 4, 500)
    station.rx_gap(100000)
    assert not station.origin_is_current(current)
    assert station._tune_epoch == origin.tune_epoch
    station.retuned()
    assert station._tune_epoch == origin.tune_epoch + 1


def test_locator_frontier_includes_pending_and_virtual_training():
    receiver = locator.Receiver(context=CaptureContext(1, 2, 1000000, 48000, 128))
    receiver.raw.feed(np.zeros(10000))
    pending = locator.Over(locator.Candidate(4, 2000, 1., 0, 2000))
    receiver.pending.append(pending)
    assert receiver.frontier <= 1000000 - 128 - rx.BACKWARD_REACH
    early = FrameOrigin(1, 2, 500000, 800000, 4, 500)
    pending.frames = [rx.FrameResult(2000, (0, 256), 9, 1., False, b'', origin=early)]
    assert receiver.frontier <= early.training_origin


def test_wideband_decode_over_preserves_onset_column_and_padding(monkeypatch):
    level, onset, column = 3, 37, 18
    rec = wide.RECORDS[level]
    mag = np.ones((rec.ncols + column, rec.span))
    monkeypatch.setattr(wide, '_alignments', lambda *a: iter([(onset, column, mag)]))
    monkeypatch.setattr(wide, '_guard_scores', lambda *a: (np.array([24]), None, None))
    monkeypatch.setattr(wide, 'frame_at', lambda *a: wide.Frame2300(level, True, b'A' * 92))
    context = CaptureContext(1, 2, 6000000, 48000)
    frame = wide.decode_over(np.zeros(1000), 100, 900, level=level, context=context)
    expected = 100 + onset + (column - rec.lead) * rec.dw50
    assert frame.onset == 100 + onset - rec.lead * rec.dw50
    assert frame.column == column
    assert frame.origin.training_origin == context.sample_start + expected
    assert frame.origin.end_sample == context.sample_start + expected + rec.ncols * rec.dw50
    assert frame.origin.bandwidth_hz == 2300


def test_ofdm_ordered_holes_preserve_cell_grid_origin(monkeypatch):
    row = ofdm.ROWS[500, 13]
    valid = b'A' * (row.frame_bytes - 2)
    from hfmodem.kestrel.coding.crc import crc16_genibus
    valid += crc16_genibus(valid).to_bytes(2, 'big')
    frames = iter([bytes(row.frame_bytes), valid])
    monkeypatch.setattr(ofdm, '_llr', lambda *a: (None, np.zeros(row.coded // row.bpc, int), 20.))
    monkeypatch.setattr(ofdm, 'decode_frame', lambda *a: next(frames))
    monkeypatch.setattr(ofdm, 'onair', lambda *a: np.zeros(row.coded, int))
    context = CaptureContext(3, 4, 1000000, 48000)
    result = ofdm._read([(None, 1.), (None, 1.)], row, 0.,
                        context=context, origin=-500, start=7)
    read = phy.LevelRead(13, observations=result.observations)
    assert read.observations[0].body is None
    assert read.observations[1].body == valid[:-2]
    assert read.frames == (valid[:-2],) and read.lost
    assert read.observations[0].origin.training_origin == 1000000 - 500 + 7 * ofdm.SYM
    assert read.observations[1].origin.training_origin - read.observations[0].origin.training_origin == ofdm.FRAME * ofdm.SYM


def test_real_narrow_fallback_from_high_speed_preserves_actual_geometry():
    audio, raw, _ = _pair(2)
    audio = np.pad(audio, (750000, 0))
    context = CaptureContext(4, 7, 9000000, 48000)
    read = phy.decode_level(audio, '500', 6, fresh=4800, under=4800,
                            expect_pair=True, context=context)
    assert read is not None and read.level == 2
    assert read.frames == tuple(frame[:-2] for frame in raw)
    assert len(read.observations) == 2
    assert abs(read.observations[0].origin.training_origin - 9762000) <= 16
    assert read.observations[1].origin.training_origin - read.observations[0].origin.training_origin == 233472
    assert all(o.origin.bandwidth_hz == 500 and o.origin.level == 2
               for o in read.observations)


def test_window_reads_preserves_selected_fit_and_recursive_offsets(monkeypatch):
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='2300')
    rec = wide.RECORDS[3]
    whole = rec.ncols * rec.dw50
    def alignment(x):
        return arq._IndexAlignment(3, 24, 24,
            np.ones((len(x) // rec.dw50, rec.span)), 37)
    monkeypatch.setattr(station, '_index_alignment', alignment)
    monkeypatch.setattr(wide, 'frame_at', lambda *a: wide.Frame2300(3, True, b'A' * 92))
    monkeypatch.setattr(arq, '_window_state', lambda *a: arq._QUIET)
    station._decode_context = CaptureContext(3, 8, 5000000, 48000)
    audio = np.zeros(2 * whole + 123)
    reads = station._window_reads(audio, offset=27000)
    assert len(reads) == 2
    first_end = whole + 123 - rec.dw50
    assert reads[0][3].origin.training_origin == 5027000 + 37
    assert reads[1][3].origin.training_origin == 5027000 + first_end + 37
    assert all(frame.origin.end_sample - frame.origin.training_origin == whole
               for _, _, _, frame, _, _ in reads)


def test_tone_origin_uses_selected_onset_and_pulse_support():
    from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
    row = tone.ROWS[500, 5]
    body = bytes([0x41]) * (row.frame_bytes - 2)
    audio = np.pad(phy.render_level((body,), '500', 5, VB6Rnd(4107106)), (9001, 4000))
    context = CaptureContext(2, 9, 5000000, 48000)
    result = tone.decode_burst(audio, 500, 5, context=context)
    observation, = result.observations
    assert result.ok and observation.body == body
    assert observation.origin.training_origin == context.sample_start + result.onset
    support = max(row.stagger) + (row.nsym - 1) * tone.H + len(row.taps) // 2 + 1
    assert observation.origin.end_sample == observation.origin.training_origin + support
    assert observation.origin.level == 5 and observation.origin.bandwidth_hz == 500
