"""Decision-point telemetry must preserve decoding and recording-clock placement."""
import numpy as np

from hfmodem.kestrel.rx import varaofdm as rx
from hfmodem.kestrel.tx import varaofdm_tx as tx
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd


def packet(key=(500, 7)):
    return tx.build_frame(bytes(rx.ROWS[key].frame_bytes - 3), 0, *key, caller='N0CALL')


def test_symbols_preserve_decode_carriers_and_sample_clock():
    key = (500, 7)
    frame = packet(key)
    wave = tx.render(frame, *key, VB6Rnd(7))
    x = np.pad(wave, (4800, 3000))
    plain = rx.decode_burst(x, key[0])
    symbols = []
    observed = rx.decode_burst(x, key[0], on_symbols=symbols.append)
    assert plain == observed and observed.frames == (frame,)
    assert len(symbols) == 1
    s = symbols[0]
    assert s.crc_ok and s.level == key[1]
    assert len(s.points) == rx.ROWS[key].coded // rx.ROWS[key].bpc
    assert set(s.carrier_hz) == {b * rx.FS / rx.TU for b in range(27, 38)}
    assert s.start_sample <= min(s.sample_indices) < max(s.sample_indices) < s.end_sample
    assert s.start_sample >= 4800 and s.end_sample <= len(x)
    shifted = []
    rx.decode_burst(np.pad(x, (317, 0)), key[0], on_symbols=shifted.append)
    assert np.max(np.abs(shifted[0].sample_indices - s.sample_indices - 317)) <= 2
    # Values are the noisy receiver estimates, not hard decisions painted back.
    rng = np.random.default_rng(99)
    noisy = []
    rx.decode_burst(x + rng.normal(0, .005, len(x)), key[0], on_symbols=noisy.append)
    assert noisy and np.mean(np.min(abs(noisy[0].points[:, None] - noisy[0].ideal)**2, axis=1)) > 0


def test_failed_crc_still_exposes_selected_hypothesis(monkeypatch):
    key = (500, 7)
    x = np.pad(tx.render(packet(key), *key, VB6Rnd(1)), (1000, 3000))
    monkeypatch.setattr(rx, 'check', lambda frame: False)
    symbols = []
    result = rx.decode_burst(x, key[0], on_symbols=symbols.append)
    assert not result.ok
    assert symbols and not symbols[0].crc_ok and len(symbols[0].points)


def test_no_acquisition_does_not_invent_a_cloud():
    symbols = []
    result = rx.decode_burst(np.zeros(220000), 500, on_symbols=symbols.append)
    assert not result.ok and symbols == []


def test_tone_symbols_preserve_results_and_band_timing():
    from hfmodem.kestrel.rx import varatone as tone
    from hfmodem.kestrel.tx import varatone_tx as tone_tx
    key = (2300, 6)
    row = tone.ROWS[key]
    frame = tone_tx.build_frame(bytes(row.frame_bytes - 3), 0, row, caller='N0CALL')
    x = np.pad(tone_tx.synth_frame(frame, *key, VB6Rnd(3)), (2400, 2400))
    plain = tone.decode_burst(x, key[0], key[1])
    frames = []
    observed = tone.decode_burst(x, key[0], key[1], on_symbols=frames.append)
    assert observed == plain and observed.frames == (frame,)
    assert len(frames) == 1 and frames[0].crc_ok
    assert set(frames[0].carrier_hz) == set(row.centres)
    assert min(frames[0].sample_indices) >= 2400
    assert max(frames[0].sample_indices) < len(x)
