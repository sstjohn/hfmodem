# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.rx import bw500_locator as locator, varahf500 as rx
from hfmodem.tests.kestrel.test_bw500_locator_ownership import _station, _frame, _WINDOWS


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('age,accepted', [(0, True), (rx.FS, True), (rx.FS + 1, False)])
def test_owned_pair_wait_ages_from_completion_decision(monkeypatch, role, age, accepted):
    station = _station(role)
    frame = _frame(_WINDOWS[0])
    end = frame.start + (rx.NSYM + 9) * rx.H
    decision = end + rx.NSYM * rx.H + rx._PAD
    candidate = locator.Candidate(4, frame.start, .7, end, frame.start)
    over = locator.Over(candidate, generation=station._front_generation,
                        frames=[frame], status='closed', end=end, decision=decision)
    station._base_ready = [over]
    station._front_heard = decision + age
    station._base_rx = SimpleNamespace(raw=SimpleNamespace(
        origin=frame.start, samples=station._front_heard,
        raw=np.zeros(station._front_heard - frame.start)))
    taken = []

    def answer(samples):
        taken.append(station._resend_decoded)
        return True

    monkeypatch.setattr(station, '_answer_data_over', answer)
    assert station._take_located_500() is accepted
    assert len(taken) == int(accepted)
    assert not station._base_ready
    assert station._resend_decoded is None
    assert not station._take_located_500()


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('tail', [2048, 12000])
def test_expected_single_does_not_invent_a_partner_at_eof(level, tail):
    from hfmodem.kestrel.tx import varahf500_tx as tx
    body = bytes([31]) * rx._PAYLOAD[level]
    frame = tx.build_frame(body, 0x95) if level == 4 else tx.build_l3_frame(body, 0x95, level)
    signal = tx.synth_burst([frame], onset=12000, level=level)
    receiver = locator.Receiver()
    found = receiver.feed(np.pad(signal, (0, tail)), eof=True)
    assert len(found) == 1
    assert found[0].result.complete
    assert [f.frame_bytes for f in found[0].frames] == [frame]
    assert found[0].result.frames_skipped == 0


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('fraction', [.002, .005, .01, .02, .1, .5, .9])
def test_eof_inside_a_second_slot_never_delivers_a_partial_pair(level, fraction):
    from hfmodem.kestrel.tx import varahf500_tx as tx
    build = tx.build_frame if level == 4 else lambda p, m: tx.build_l3_frame(p, m, level)
    frames = [build(bytes([v]) * rx._PAYLOAD[level], 0x95) for v in (31, 73)]
    signal = tx.synth_burst(frames, onset=12000, level=level)
    rec = rx.INDEX_RECORDS.get(level)
    lead = rec.lead * rec.dw if rec else 9 * rx.H
    slot = rec.ncols * rec.dw if rec else rx.NSYM * rx.H
    receiver = locator.Receiver()
    found = receiver.feed(signal[:12000 + lead + int((1 + fraction) * slot)], may_pair=True, eof=True)
    assert found
    assert not any(o.result.complete for o in found)


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_faded_partner_is_never_closed_as_a_clean_single(level):
    from hfmodem.kestrel.tx import varahf500_tx as tx
    build = tx.build_frame if level == 4 else lambda p, m: tx.build_l3_frame(p, m, level)
    frames = [build(bytes([v]) * rx._PAYLOAD[level], 0x95) for v in (31, 73)]
    signal = tx.synth_burst(frames, onset=12000, level=level)
    rec = rx.INDEX_RECORDS.get(level)
    lead = rec.lead * rec.dw if rec else 9 * rx.H
    slot = rec.ncols * rec.dw if rec else rx.NSYM * rx.H
    start = 12000 + lead + slot
    signal[start:start + 24000] *= .01
    receiver = locator.Receiver()
    found = receiver.feed(np.pad(signal, (0, 12000)), may_pair=True, eof=True)
    assert found
    assert not any(o.result.complete and len(o.frames) == 1 for o in found)


@pytest.mark.parametrize('samples', [0, 1, 1024])
def test_empty_or_short_observation_cannot_complete(samples):
    result = locator.decode(np.zeros(samples))
    assert not result.frames
    assert not result.complete


@pytest.mark.parametrize('level,kind,snr', [(lv, 'white', -6) for lv in (1, 2, 3, 4)] + [(4, 'ssb', -3)])
def test_predicted_noisy_pair_keeps_both_frames(level, kind, snr):
    from scipy.signal import firwin, lfilter
    from hfmodem.kestrel.tx import varahf500_tx as tx
    build = tx.build_frame if level == 4 else lambda p, m: tx.build_l3_frame(p, m, level)
    frames = [build(bytes([v]) * rx._PAYLOAD[level], 0x95) for v in (31, 73)]
    signal = tx.synth_burst(frames, onset=24000, level=level)
    power = np.mean(signal[24000:72000] ** 2)
    signal = np.pad(signal, (0, rx.FS * 6))
    noise = np.random.default_rng(7).standard_normal(len(signal))
    if kind == 'ssb':
        noise = lfilter(firwin(513, [300, 2700], pass_zero=False, fs=rx.FS), 1, noise)
    noise_power = np.mean(noise ** 2) * (2500 / 24000 if kind == 'white' else 1)
    x = signal + noise * np.sqrt(power / noise_power / 10 ** (snr / 10))
    receiver = locator.Receiver()
    found = []
    for j in range(0, len(x), 1024):
        found.extend(receiver.feed(x[j:j + 1024], may_pair=True))
    good = [o for o in found if any(f.crc_ok for f in o.frames)]
    assert len(good) == 1
    assert good[0].result.complete
    assert [f.frame_bytes for f in good[0].frames] == frames


@pytest.mark.parametrize('onset', [3000, 3127, 3511, 3582, 4023])
@pytest.mark.parametrize('chunk', [1024, 1003, 5003])
def test_short_eof_tail_uses_only_real_partial_block_samples(onset, chunk):
    from hfmodem.kestrel.tx import varahf500_tx as tx
    frame = tx.build_frame(bytes([31]) * rx._PAYLOAD[4], 0x95)
    signal = np.pad(tx.synth_burst(frame, onset=onset, ncol=rx.NSYM + 9), (0, 2048))
    receiver = locator.Receiver()
    found = []
    for j in range(0, len(signal), chunk):
        found.extend(receiver.feed(signal[j:j + chunk]))
    ticks = receiver.front.lattice.ticks
    found.extend(receiver.feed(np.empty(0), eof=True))
    assert receiver.front.lattice.ticks == ticks
    assert len(found) == 1
    assert found[0].result.complete
    assert [f.frame_bytes for f in found[0].frames] == [frame]
    assert receiver.front.samples + len(receiver.front.pending) == len(signal) + 8192


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('fraction', [.01, .1, .3, .45, .9])
@pytest.mark.parametrize('snr', [None, -6])
def test_batch_decode_and_phy_reject_eof_inside_second_slot(level, fraction, snr):
    from hfmodem.kestrel.arq import phy
    from hfmodem.kestrel.tx import varahf500_tx as tx
    build = tx.build_frame if level == 4 else lambda p, m: tx.build_l3_frame(p, m, level)
    frames = [build(bytes([v]) * rx._PAYLOAD[level], 0x95) for v in (31, 73)]
    signal = tx.synth_burst(frames, onset=12000, level=level)
    rec = rx.INDEX_RECORDS.get(level)
    lead = rec.lead * rec.dw if rec else 9 * rx.H
    slot = rec.ncols * rec.dw if rec else rx.NSYM * rx.H
    cut = signal[:12000 + lead + int((1 + fraction) * slot)]
    if snr is not None:
        noise = np.random.default_rng(7).standard_normal(len(cut))
        power = np.mean(signal[12000:72000] ** 2)
        cut = cut + noise * np.sqrt(power / (np.mean(noise ** 2) * 2500 / 24000)
                                   / 10 ** (snr / 10))
    result = locator.decode(cut)
    assert result.frames and result.frames[0].crc_ok
    assert not result.observation_complete
    assert not result.complete
    assert phy._decode_base(cut, '500') is None


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('tail', [2048, 12000])
def test_batch_decode_completes_whole_bursts_with_eof_tail(level, count, tail):
    from hfmodem.kestrel.tx import varahf500_tx as tx
    build = tx.build_frame if level == 4 else lambda p, m: tx.build_l3_frame(p, m, level)
    frames = [build(bytes([v]) * rx._PAYLOAD[level], 0x95) for v in (31, 73)[:count]]
    signal = tx.synth_burst(frames, onset=12000, level=level)
    result = locator.decode(np.pad(signal, (0, tail)))
    assert result.complete
    assert [f.frame_bytes for f in result.frames] == frames


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_failed_first_crc_holds_predicted_pair_until_full_span(monkeypatch, level):
    rec = rx.INDEX_RECORDS.get(level)
    slot = rec.ncols * rec.dw if rec else rx.NSYM * rx.H
    lead = rec.lead * rec.dw if rec else 9 * rx.H
    receiver = locator.Receiver()
    over = locator.Over(locator.Candidate(level, 0, 1., 0, 0))
    if rec:
        monkeypatch.setattr(rx, '_l3_align', lambda *a: (0, rec.lead, 1., None))
        monkeypatch.setattr(rx, '_l3_frames', lambda *a, **kw: [rx.FrameResult(
            0, (0,), rec.lead, 1., False, b'bad\x82\x00\x00', level=level)])
    else:
        def aligned(*args):
            yield
            return None, {0: 0, 1: 0}, 1.
        monkeypatch.setattr(rx, '_demod_aligned_steps', aligned)
        monkeypatch.setattr(rx, '_frame_gh', lambda *a: object())
        def cells(gh, index, c0):
            yield
            return 1., c0, False, b'bad\x82\x00\x00', False
        monkeypatch.setattr(rx, '_decode_cells_steps', cells)
    receiver.raw.feed(np.zeros(lead + slot + rx.FS // 2))
    for _ in range(32):
        receiver._advance(over, may_pair=True, eof=False)
        if over.pair:
            break
    assert over.pair
    assert over.status == 'open'
    assert over.frames[0].frames_in_burst == 2
    finish = over.end + slot
    assert over.next_check == finish + round(locator.ANSWER_GAP[level] * rx.FS)
    receiver.raw.feed(np.zeros(finish - 1 - receiver.raw.samples))
    receiver._advance(over, may_pair=True, eof=True)
    assert over.status == 'truncated'
    assert over.end == finish
    assert not over.result.complete
