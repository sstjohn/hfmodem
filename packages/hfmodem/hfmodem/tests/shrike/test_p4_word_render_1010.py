# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import resample_poly
from math import gcd

from hfmodem.shrike import onair, p4breakintx, p4chirp, p4longtx, p4oracle, p4robusttx, p4rx, p4shorttx, p4sig, p4waveforms, p4words
from hfmodem.shrike.p4receiver import Kind
from hfmodem.tests.shrike import p4_stock_peer as stock
from hfmodem.tests.shrike import test_p4_placed_carrier_0929 as placed

RATES = (8000, 12000, 24000, 28800, 44100, 48000, 96000)
FIELDS = ((b'', 0x20), (b'', 0x40), (b'data', 0x20), (b'data', 0x40), *((b'counted data', n) for n in range(4)))
FROZEN = Path(__file__).with_name('fixtures')/'p4-word-render-1010.json'


def waves(fs, mode=None):
    kw = {} if mode is None else {'word_render': mode}
    for name in p4words.STEPS:
        yield 'word:'+name, p4robusttx.robust_word(name, fs, **kw)
    for payload, status in FIELDS:
        yield f'body:{payload.hex()}:{status}', p4breakintx.changeover_packet(payload, status, fs=fs, **kw)
    for header in ('A', 'B'):
        for value in (7, 38, 60):
            yield f'short:{value}:{header}', p4robusttx.robust_short(value, header, fs)
    for n in range(3):
        yield f'clone:{n}', p4robusttx.robust_cs_clone(n, fs)
    for name, render in (('normal', p4shorttx.short_packet), ('long', p4longtx.long_packet), ('chirp', p4chirp.entry_packet)):
        yield name, render(b'packet family', 1, fs)


@pytest.mark.parametrize('fs', RATES)
def test_frozen_off_and_unaffected_families(fs):
    reference = json.loads(FROZEN.read_text())
    assert reference['schema'] == 2 and reference['grid_bits'] == 24
    frozen = reference['rates'][str(fs)]
    omitted, off, selected = (dict(waves(fs, mode)) for mode in (None, 'off', 'stock'))
    for name, wave in omitted.items():
        row = frozen[name]
        assert wave.dtype == np.dtype(row['dtype']) == np.dtype('float64')
        assert wave.shape == (row['samples'],) and np.isfinite(wave).all()
        # Platform libm/FFT implementations differ below 8e-16. A full-wave
        # fingerprint pins every sample to the fixed 2^-24 amplitude grid
        # without treating float64 implementation noise as waveform drift.
        canonical = np.rint(wave * (1 << reference['grid_bits'])).astype('<i8')
        assert sha256(canonical.tobytes()).hexdigest() == row['sha256_int64_le']
        assert np.array_equal(wave, off[name])
        if name.startswith('word:'):
            np.testing.assert_allclose(selected[name], stock.word(name[5:], fs), rtol=0, atol=1e-12)
        if not name.startswith(('word:', 'body:')) or (name.startswith('word:') and name[5:] not in p4words.CARRIER_HZ):
            assert np.array_equal(wave, selected[name])


@pytest.mark.parametrize('fs', [44100, 48000])
@pytest.mark.parametrize('name', sorted(p4words.CARRIER_HZ))
def test_cached_stock_word_despreading_and_independent_fit(fs, name):
    cache = onair._p4_link_waveforms(SimpleNamespace(values=()), word_render='stock')
    wave = cache[p4waveforms.cache_key(Kind.WORD, word=name)]
    if fs != 48000:
        wave = resample_poly(wave, fs//gcd(fs, 48000), 48000//gcd(fs, 48000))
    direct = p4robusttx.robust_word(name, fs, word_render='stock')
    np.testing.assert_allclose(direct, stock.word(name, fs), rtol=0, atol=1e-12)
    for actual in (wave, direct):
        audio = np.pad(actual, (fs//10, fs//10))
        coherence, symbols, first = p4oracle.placed_despread(audio, fs, .1, 0., count=27)
        match, score, _ = p4words.match_placed(symbols, cfo_hz=0.)
        assert coherence > p4oracle.ROBUST_FRACTION_KNEE
        assert match == name and score > p4words.MATCH_KNEE
        assert abs(first-fs//10) <= 2
        phase = np.unwrap(np.angle(symbols*np.conj(p4words.BANK[name])))
        fitted = np.polyfit(np.arange(2, 25)/112.5, phase[2:25], 1)[0]/(2*np.pi)
        assert fitted == pytest.approx(p4words.CARRIER_HZ[name], abs=.1)


@pytest.mark.parametrize('fs', [44100, 48000])
@pytest.mark.parametrize('payload,status', FIELDS)
def test_stock_body_matches_independent_peer_and_crc(fs, payload, status):
    wave = p4breakintx.changeover_packet(payload, status, fs=fs, word_render='stock')
    np.testing.assert_allclose(wave, stock.body(payload, status, fs), rtol=0, atol=1e-12)
    audio = np.pad(wave, (fs//10, fs//10))
    head = p4oracle.word_at(audio, fs, 0, .1, cfo_hz=0., expected_words={'Y27'})
    assert head is not None and head.word == 'Y27'
    frame = p4oracle.body_after(audio, fs, 0, head)
    assert frame is not None and frame.data == payload.ljust(31, b'\x1e')
    assert frame.status.byte == status


@pytest.mark.parametrize('name', sorted(p4words.STEPS))
def test_exact_chip_ramp_zero_origin_and_reset(monkeypatch, name):
    captured = []
    convolve = np.convolve
    def grab(impulses, taps):
        captured.append(impulses[::p4rx.SAMPLES_PER_CHIP].copy())
        return convolve(impulses, taps)
    monkeypatch.setattr(np, 'convolve', grab)
    p4robusttx.robust_word(name, word_render='stock')
    p4robusttx.robust_word(name, word_render='stock')
    chips = (p4words.BANK[name][:, None]*p4sig.SPREAD16).ravel()
    f = p4words.CARRIER_HZ.get(name, 0.)
    expected = chips*np.exp(2j*np.pi*f*np.arange(len(chips))/1800) if f else chips
    np.testing.assert_allclose(captured[0], expected, rtol=0, atol=1e-14)
    assert captured[0][0] == chips[0]
    assert np.array_equal(captured[0], captured[1])


@pytest.mark.parametrize('name', ['Y27', 'T27', 'Z27'])
def test_body_head_only_phase_and_gain(monkeypatch, name):
    captured = []
    convolve = np.convolve
    def grab(impulses, taps):
        captured.append(impulses[::p4rx.SAMPLES_PER_CHIP].copy())
        return convolve(impulses, taps)
    monkeypatch.setattr(np, 'convolve', grab)
    p4breakintx.changeover_packet(b'data', 1, name, word_render='stock')
    p4breakintx.changeover_packet(b'data', 1, name)
    selected, legacy = captured
    f = p4words.CARRIER_HZ.get(name, 0.)
    ramp = np.exp(2j*np.pi*f*np.arange(432)/1800)
    np.testing.assert_allclose(selected[:432], legacy[:432]*ramp, rtol=0, atol=1e-14)
    assert np.array_equal(selected[432:], legacy[432:])
    assert selected[0] == legacy[0]
    if name == 'Y27':
        assert np.angle(ramp[216], deg=True) == pytest.approx(60.48)
        assert np.angle(ramp[-1]*np.exp(2j*np.pi*f/1800), deg=True) == pytest.approx(120.96)


@pytest.mark.parametrize('cfo', [-10., 0., 10.])
@pytest.mark.parametrize('snr', [5., 10.])
def test_production_stock_partial_full_and_body_cfo(monkeypatch, cfo, snr):
    def word(name, *, cfo):
        from scipy.signal import hilbert
        cache = onair._p4_link_waveforms(SimpleNamespace(values=()), word_render='stock')
        wave = cache[p4waveforms.cache_key(Kind.WORD, word=name)]
        return (hilbert(wave)*np.exp(2j*np.pi*cfo*np.arange(len(wave))/48000)).real
    monkeypatch.setattr(stock, 'word', word)
    placed.test_stock_words_acquire_then_alternate_on_the_body_carrier(cfo, snr)


@pytest.mark.parametrize('hold', [False, True])
@pytest.mark.parametrize('cfo', [-10., 0., 10.])
@pytest.mark.parametrize('snr', [5., 10.])
def test_production_stock_held_body_crc_and_cfo(monkeypatch, hold, cfo, snr):
    def body(payload, status, *, cfo):
        wave = p4breakintx.changeover_packet(payload, status, word_render='stock')
        from scipy.signal import hilbert
        return (hilbert(wave)*np.exp(2j*np.pi*cfo*np.arange(len(wave))/48000)).real
    monkeypatch.setattr(stock, 'body', body)
    placed.test_stock_head_enriches_without_learning_its_offset_twice(hold, cfo, snr)


def test_probe_and_dynamic_payload_cache_modes():
    from hfmodem.tests.shrike.test_p4turn import machine
    for mode in ('off', 'stock'):
        probe = SimpleNamespace(values=())
        cache = onair._p4_link_waveforms(probe, word_render=mode)
        assert cache.word_render == mode
        assert onair._p4_link_waveforms(probe, word_render=mode) is cache
        with pytest.raises(ValueError, match='mode'):
            onair._p4_link_waveforms(probe, word_render='stock' if mode == 'off' else 'off')
        m, _ = machine(data='chirp')
        m.outbox.arq.on_host_data(b'dynamic payload '*4)
        before = bytes(m.outbox.arq._outbuf), m.outbox.inflight
        p4waveforms.prepare(cache, m, word_render=mode)
        assert (bytes(m.outbox.arq._outbuf), m.outbox.inflight) == before
        bodies = [k for k in cache if k[0] == Kind.BODY]
        assert any(k[4] for k in bodies)
        for k in bodies:
            assert np.array_equal(cache[k], p4breakintx.changeover_packet(k[4], k[5], word_render=mode))
        with pytest.raises(ValueError, match='mode'):
            p4waveforms.prepare(cache, m, word_render='stock' if mode == 'off' else 'off')


@pytest.mark.parametrize('mode', ['bad', '', None])
def test_invalid_render_mode(mode):
    for render in (lambda: p4robusttx.robust_word('X27', word_render=mode),
                   lambda: p4breakintx.changeover_packet(b'', 0x40, word_render=mode),
                   lambda: onair._p4_link_waveforms(SimpleNamespace(values=()), word_render=mode)):
        with pytest.raises(ValueError, match='mode'):
            render()


@pytest.mark.parametrize('refused', [False, True])
@pytest.mark.parametrize('policy', ['fixed-x27', 'counter-odd-t27', 'counter-even-t27'])
def test_b1_repeated_refused_answers_keep_selection_timing_and_source(refused, policy):
    from hfmodem.tests.shrike.test_p4_late_long_ack_1010 import accepted, held_machine
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    for status in (0x61, 0x62):
        results = []
        for mode in ('off', 'stock'):
            cache = onair._p4_link_waveforms(SimpleNamespace(values=()), word_render=mode)
            m, delivered = held_machine()
            m.long_ack = policy
            m.observe(accepted(48000, b'data', status))
            first = m.decide(48000+163200-m.lead)
            wave = cache[p4waveforms.cache_key(first.kind, first.word)]
            if refused:
                m.refused('deadline')
            else:
                m.keyed()
            m.observe(unread(Kind.LONG, 228000))
            repeated = m.decide(228000+163200-m.lead)
            assert np.array_equal(wave, cache[p4waveforms.cache_key(repeated.kind, repeated.word)])
            assert repeated.word == first.word and repeated.accepted_target == 48000
            assert not repeated.current_slot_accepted
            sent = m.keyed()
            results.append((first, repeated, sent, delivered, m._last_ack, m.acked_counter,
                            bytes(m.outbox.arq._outbuf), m.outbox.inflight))
        assert results[0] == results[1]


@pytest.mark.parametrize('mode', ['off', 'stock'])
def test_driver_mode_records_and_prepared_emissions(mode, monkeypatch, capsys):
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx, FS
    audio = np.zeros(8*FS)
    for at, wave in ((44304, stock.body(b'hello', 0)),
                     (104304, p4robusttx.robust_short(13)),
                     (192624, stock.body(b'', 0x40))):
        audio[at:at+len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_word_render = mode
    tx.p4_link_options.p4_text = 'dynamic data'
    tx.p4_link_options.p4_data = 'body'
    for module, name in ((p4robusttx, 'robust_word'), (p4breakintx, 'changeover_packet')):
        render = getattr(module, name)
        def outside_admission(*args, _render=render, **kwargs):
            assert getattr(tx, 'p4_tx_admit', None) is None
            return _render(*args, **kwargs)
        monkeypatch.setattr(module, name, outside_admission)
    actual = []
    emit = tx._tx
    def keyed(wave, what):
        actual.append(wave.copy())
        emit(wave, what)
    tx._tx = keyed
    onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    rows = tx.p4_probe.emissions[1:]
    assert rows and any(r['payload'] for r in rows)
    assert 'word_render='+mode in capsys.readouterr().out
    event, = [e for e in tx.p4_probe.link_events if e['event'] == 'WordRender']
    assert event['word_render'] == mode
    assert event['word_offsets_hz'] == (p4words.CARRIER_HZ if mode == 'stock' else {})
    keyed = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    for row, wave, event in zip(rows, actual, keyed, strict=True):
        name = row['word'] if event['kind'] == Kind.WORD else 'Y27' if event['kind'] == Kind.BODY else None
        assert event['word_render'] == row['word_render']
        assert event['word_offset_hz'] == row['word_offset_hz']
        assert row['word_render'] == mode
        assert row['word_offset_hz'] == (p4words.CARRIER_HZ.get(name, 0.) if mode == 'stock' else 0.)
        assert row['waveform_sha256'] == sha256(wave.tobytes()).hexdigest()
        assert row['audio_start'] == row['keyed_at']
        if row['payload']:
            assert np.array_equal(wave, p4breakintx.changeover_packet(bytes.fromhex(row['payload']), row['status'], word_render=mode))
