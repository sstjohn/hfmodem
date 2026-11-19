# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KO4HJO's low-contrast turn replies, through the receiver and turn machine."""
from dataclasses import replace
import hashlib
import json

import pytest

from hfmodem.shrike import p4oracle, session
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, Kind, P4Receiver, Registered, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.evidence import OPERATOR_DATA, require_input
from hfmodem.tests.shrike.test_p4_chirp_changeover_1010 import chirp
from hfmodem.tests.shrike.test_p4turn import FS, machine

CAPTURE = 'onair-20261027T051353167738Z-fac3d9b1-ad4f-4852-ae0f-7bf7c9e3dcb8'
REQUESTS = (8791862, 9151862, 9511862, 9871862)
PAYLOAD = b'KO4HJO regression\r'


@pytest.fixture(scope='module')
def recording():
    root = OPERATOR_DATA / 'captures' / CAPTURE
    for name, digest in (
        ('p4-probe.json', '4e44a6518b708f7bdbd430f14e8362838313ac92630a4c8d229d10ff34693065'),
        ('stream.flac', '0d03f3e13d607467da1d7631f2a8e4c2bb785eac0b7740eeb23e36a3e4e54986'),
    ):
        require_input(root / name)
        with (root / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
    probe = json.loads((root / 'p4-probe.json').read_text())
    rows = [r for r in probe['heard'] if r.get('word') == 'V27'
            and r.get('responds_to') in REQUESTS and r['outcome'] == 'confirmed']
    assert [r['responds_to'] for r in rows] == list(REQUESTS)
    assert [r['onset_sample'] for r in rows] == [8837388, 9197397, 9557465, 9917402]
    assert all(r['extent_s'] is None and not r['weak'] and not r['unexpected'] for r in rows)
    return session.load_wav(str(root / 'stream.flac'), FS), probe, rows


def requested(probe, row, *, state=State.IRS_BREAKIN):
    m, a = machine(role='irs', data='body', prekey_lead=.04,
                   normal_ack='parity', chirp_ack='parity')
    m.first_body = False
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=row['cfo_hz'])
    rx.enable_hold()
    last = next(r for r in reversed(probe['heard'])
                if r['kind'] == 'chirp' and r.get('status') is not None
                and r['onset_sample'] < REQUESTS[0])
    assert (last['onset_sample'], last['status']) == (8448662, 0)
    reg = rx.register(chirp(last['onset_sample'], last['status'],
                            bytes.fromhex(last['data'])).observation)
    m.observe(reg, now=last['onset_sample'] + 158400)
    ack = m.decide(last['onset_sample'] + 163200 - m.lead)
    assert (ack.word, ack.stream_sample) == ('V27', 8611862)
    m.keyed()
    a.on_host_data(PAYLOAD)
    request = row['responds_to']
    emitted = next(e for e in probe['emissions'] if e['audio_start'] == request)
    assert emitted['word'] == 'Y27' and emitted['payload'] == ''
    # Replay the actual key, independently for each recorded retry's reply.
    m._break_in(request)
    if state == State.ISS_GRANT:
        # Exercise the other gate with an empty BODY grant and the real reply.
        m._body(request, 0x40, state)
    m.keyed()
    rx.note_emission(Emission(m.last.kind, Role.ISS, request,
                             .244 if m.last.kind == Kind.WORD else .798))
    assert not m.outbox.inflight and a.tx_acked_bytes == 0
    return m, a, rx


def window(audio, row):
    at = row['onset_sample']
    origin = at - round(.15 * FS)
    return audio[origin:at + round(.90 * FS)].copy(), origin


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('state', [State.IRS_BREAKIN, State.ISS_GRANT])
@pytest.mark.parametrize('context', ['response', 'retained'])
def test_recorded_v27_without_energy_extent_grants_turn_and_data(recording, index, state, context):
    audio, probe, rows = recording
    row = rows[index]
    m, a, rx = requested(probe, row, state=state)
    samples, origin = window(audio, row)
    if context == 'retained':
        end = row['capture_end_sample']
        origin = end - 6 * FS
        samples = audio[origin:end].copy()
        for em in probe['emissions']:
            begin, stop = em['audio_start'], em['audio_end']
            if begin < end and stop > origin:
                samples[max(0, begin - origin):min(len(samples), stop - origin)] = 0
                rx.note_gap(begin, stop)
    assert p4oracle.bursts(samples, FS, gaps=rx._gaps(samples, origin)) == []
    head = rx._word(samples, origin, provisional=True)
    assert head is not None and head.word == 'V27' and head.provisional
    m.observe(rx.register(head), now=row['onset_sample'] + round(.20 * FS))
    assert not [e for e in m.link_events if e['event'] == 'turn'] and not m.outbox.inflight
    regs = rx.read_window(samples, origin, whole=True)
    full, = [r for r in regs if r.observation.word == 'V27' and not r.observation.provisional]
    assert full.outcome == Registered.CONFIRMED
    assert full.responds_to == row['responds_to']
    assert full.observation.extent_s is None and full.observation.word_span_supported
    m.observe(full, now=row['received_at_sample'])
    assert [e['event'] for e in m.link_events if e['event'] == 'turn'] == ['turn']
    assert m.peer_phase is None and m.peer_watch is None
    assert m._next.kind == Kind.BODY and m._next.payload == PAYLOAD
    assert m._next.state == State.ISS_DATA and m._next.status == 0x40
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('edge,keep_s', [('tail', .230), ('tail', .120), ('head', .230)])
def test_truncated_recorded_v27_does_not_grant_turn(recording, index, edge, keep_s):
    audio, probe, rows = recording
    row = rows[index]
    samples, origin = window(audio, row)
    at = row['onset_sample'] - origin
    begin, end = ((at + round(keep_s * FS), at + round(.27 * FS)) if edge == 'tail'
                  else (at - round(.02 * FS), at + round((.240 - keep_s) * FS)))
    # Replace the cut with the same tape's channel noise, not artificial silence.
    noise = row['onset_sample'] + FS
    samples[begin:end] = audio[noise:noise + end - begin]
    m, a, rx = requested(probe, row)
    regs = rx.read_window(samples, origin, whole=True)
    if keep_s == .230 and index == 0:
        full, = [r for r in regs if r.observation.word == 'V27']
        assert full.observation.score > .9
        assert not full.observation.word_span_supported
    for reg in regs:
        m.observe(reg, now=row['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn'] and not m.outbox.inflight and a.tx_acked_bytes == 0
    assert bytes(a._outbuf) == PAYLOAD


@pytest.mark.parametrize('path', ['placed', 'incremental_scan'])
def test_recorded_span_refinement_reaches_turn_gate_once(recording, path):
    audio, probe, rows = recording
    row = rows[0]
    m, a, rx = requested(probe, row)
    samples, origin = window(audio, row)
    obs = p4oracle.word_at(samples, FS, origin, (row['onset_sample'] - origin) / FS,
                           cfo_hz=row['cfo_hz'])
    assert obs is not None and not obs.provisional and not obs.word_span_supported
    m.observe(rx.register(obs), now=row['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn']
    if path == 'placed':
        regs = rx.read_window(samples, origin, whole=True)
    else:
        rx.scan = 'incremental'
        rx._retain_scan(audio[origin:origin + round(1.25 * FS)], origin, None)
        rx._scan_eligible = True
        regs = rx.scan_idle(now=origin + len(samples), next_event=origin + len(samples) + FS)
    refined, = regs
    assert refined.outcome == Registered.ENRICHED
    assert refined.observation.word_span_supported and refined.observation.extent_s is None
    m.observe(refined, now=row['received_at_sample'])
    assert m._next.kind == Kind.BODY and m._next.payload == PAYLOAD
    assert rx.register(refined.observation).outcome == Registered.DUPLICATE


@pytest.mark.parametrize('missing', ['head', 'tail', 'gap'])
def test_censored_recorded_word_has_no_full_span_proof(recording, missing):
    audio, probe, rows = recording
    row = rows[0]
    m, a, rx = requested(probe, row)
    samples, origin = window(audio, row)
    at = row['onset_sample']
    if missing == 'head':
        cut = at - origin + round(.01 * FS)
        samples, origin = samples[cut:], origin + cut
    elif missing == 'tail':
        samples = samples[:at - origin + round(.230 * FS)]
    else:
        rx.note_gap(at + round(.1 * FS), at + round(.11 * FS))
    regs = rx.read_window(samples, origin, whole=True)
    assert all(not r.observation.word_span_supported for r in regs)
    for reg in regs:
        m.observe(reg, now=row['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn'] and not m.outbox.inflight


@pytest.mark.parametrize('control', ['unrelated_slot', 'no_request', 'older_request', 'stale',
                                    'provisional', 'weak', 'unexpected', 'expired',
                                    'contradicted', 'duplicate', 'short_extent'])
def test_recorded_word_support_never_replaces_turn_admission(recording, control):
    audio, probe, rows = recording
    row = rows[1]
    m, a, rx = requested(probe, row)
    samples, origin = window(audio, row)
    if control == 'no_request':
        rx = P4Receiver(FS, admitted=True, cfo_hz=row['cfo_hz'])
    if control == 'unrelated_slot':
        # Preserve a supported recorded waveform while isolating association.
        shift = 9257393 - row['onset_sample']
        row = dict(row, onset_sample=9257393)
        origin += shift
        rx.clock = Clock(Expect.WORD, row['onset_sample'], .935, 3.75, row['cfo_hz'])
    regs = rx.read_window(samples, origin, whole=True)
    reg, = [r for r in regs if r.observation.word == 'V27']
    assert reg.observation.word_span_supported and reg.observation.extent_s is None
    if control in ('unrelated_slot', 'no_request'):
        assert reg.responds_to is None
    elif control == 'older_request':
        reg = replace(reg, responds_to=REQUESTS[0])
    elif control == 'stale':
        reg = replace(reg, response_generation=17)
    elif control in ('provisional', 'weak', 'unexpected'):
        reg = replace(reg, observation=replace(reg.observation, **{control: True}))
    elif control in ('expired', 'contradicted', 'duplicate'):
        reg = replace(reg, outcome=Registered(control))
    elif control == 'short_extent':
        reg = replace(reg, observation=replace(reg.observation, extent_s=.230))
    m.observe(reg, now=row['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn'] and not m.outbox.inflight and a.tx_acked_bytes == 0
    assert bytes(a._outbuf) == PAYLOAD


def recorded_word(recording, index=0):
    audio, probe, rows = recording
    row = rows[index]
    samples, origin = window(audio, row)
    _, _, rx = requested(probe, row)
    obs = rx._word(samples, origin, whole=True)
    assert obs is not None and obs.word == 'V27' and obs.word_span_supported
    return samples, origin, obs


@pytest.mark.parametrize('extents', [(.230, None, .250), (.250, .230, None)])
def test_registration_keeps_every_measured_short_extent(recording, extents):
    _, probe, rows = recording
    _, _, obs = recorded_word(recording)
    m, a, rx = requested(probe, rows[0])
    for extent in extents:
        reg = rx.register(replace(obs, extent_s=extent,
                                  word_span_supported=extent is None))
        if .230 in extents[:extents.index(extent) + 1]:
            m.observe(reg, now=rows[0]['received_at_sample'])
            assert rx._classification(obs.stream_sample).extent_s == .230
            assert reg.observation.extent_s == .230
    assert not [e for e in m.link_events if e['event'] == 'turn']
    assert not m.outbox.inflight and a.tx_acked_bytes == 0


@pytest.mark.parametrize('old_ms,new_ms', [(30, 50), (50, 30)])
def test_supported_refinement_rechecks_reply_bracket(recording, old_ms, new_ms):
    _, probe, rows = recording
    _, _, obs = recorded_word(recording)
    m, a, rx = requested(probe, rows[0])
    due = rows[0]['responds_to'] + round(.935 * FS)
    def shifted(ms, supported):
        at = due + round(ms / 1000 * FS)
        return replace(obs, stream_sample=at, first_chip=at-obs.origin,
                       word_span_supported=supported)
    first = rx.register(shifted(old_ms, False))
    assert first.responds_to == (rows[0]['responds_to'] if old_ms == 30 else None)
    reg = rx.register(shifted(new_ms, True))
    assert reg.outcome == Registered.ENRICHED
    assert reg.responds_to == (rows[0]['responds_to'] if new_ms == 30 else None)
    m.observe(reg, now=rows[0]['received_at_sample'])
    assert bool([e for e in m.link_events if e['event'] == 'turn']) == (new_ms == 30)
    assert a.tx_acked_bytes == 0


# SF16 chip timing measured on each intact word; the third probe onset is a
# multipath copy 1.4 ms later. Pulse centres lag the chip impulses by 64 samples
# at 28.8 kHz; half a chip earlier is the first symbol's waveform boundary.
CHIP_ONSETS = (8837391, 9197395, 9557397, 9917402)


def replace_symbols(recording, index, begin_symbol, end_symbol, noise_s):
    from hfmodem.shrike import p4rx, p4words
    audio, _, rows = recording
    row = rows[index]
    samples, origin, obs = recorded_word(recording, index)
    boundary = CHIP_ONSETS[index] - origin + round(56 / p4rx.SAMPLE_RATE * FS)
    begin = boundary + round(begin_symbol / p4words.SYMBOL_RATE * FS)
    end = boundary + round(end_symbol / p4words.SYMBOL_RATE * FS)
    noise = row['onset_sample'] + round(noise_s * FS)
    samples[begin:end] = audio[noise:noise + end-begin]
    return samples, origin, obs


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('edge', ['head', 'tail'])
@pytest.mark.parametrize('fraction', [.25, .50, .75])
@pytest.mark.parametrize('noise_s', [.5, 1.])
def test_partial_endpoint_symbol_has_no_span_proof(recording, index, edge, fraction, noise_s):
    begin, end = ((-2, fraction) if edge == 'head' else (26 + fraction, 30))
    samples, origin, obs = replace_symbols(recording, index, begin, end, noise_s)
    assert not p4oracle._word_span_supported(samples, FS, obs.stream_sample-origin,
                                            obs.cfo_hz, ())
    _, probe, rows = recording
    m, a, rx = requested(probe, rows[index])
    for reg in rx.read_window(samples, origin, whole=True):
        assert not reg.observation.word_span_supported
        m.observe(reg, now=rows[index]['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn']
    assert not m.outbox.inflight and a.tx_acked_bytes == 0


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('symbol', range(1, 26))
@pytest.mark.parametrize('noise_s', [.5, 1.])
def test_single_noise_filled_interior_symbol_has_no_span_proof(recording, index, symbol, noise_s):
    samples, origin, obs = replace_symbols(recording, index, symbol, symbol + 1, noise_s)
    assert not p4oracle._word_span_supported(samples, FS, obs.stream_sample-origin,
                                            obs.cfo_hz, ())


@pytest.mark.parametrize('index', range(4))
def test_word_extent_and_span_share_burst_inventory(recording, monkeypatch, index):
    samples, origin, _ = recorded_word(recording, index)
    _, probe, rows = recording
    _, _, rx = requested(probe, rows[index])
    calls = []
    bursts = p4oracle.bursts
    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return bursts(*args, **kwargs)
    monkeypatch.setattr(p4oracle, 'bursts', counted)
    obs = rx._word(samples, origin, whole=True)
    assert obs.word_span_supported and obs.extent_s is None
    assert calls == [len(samples)]


def test_acquisition_repeat_extent_and_shifted_association_sequence(recording):
    audio, probe, rows = recording
    row = rows[0]
    samples, origin, supported = recorded_word(recording)
    m, a, rx = requested(probe, row)
    def acquire():
        hit, = p4oracle.word_scan(samples, FS, row['cfo_hz'], origin=origin)
        assert hit.word == 'V27' and hit.extent_s is None
        return hit
    first = rx.register(acquire())
    assert first.outcome == Registered.NEW and first.responds_to == row['responds_to']
    repeated = rx.register(acquire())
    assert repeated.outcome == Registered.DUPLICATE
    measured = rx.register(replace(acquire(), extent_s=.230))
    assert measured.outcome == Registered.EXTENDED
    shifted = replace(supported, stream_sample=supported.stream_sample + FS // 20,
                      first_chip=supported.first_chip + FS // 20)
    refined = rx.register(shifted)
    assert refined.outcome == Registered.ENRICHED
    assert refined.observation.extent_s == .230 and refined.responds_to is None
    assert rx.register(shifted).outcome == Registered.DUPLICATE
    for reg in (first, repeated, measured, refined):
        m.observe(reg, now=row['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn']
    assert not m.outbox.inflight and a.tx_acked_bytes == 0


@pytest.mark.parametrize('control', ['noise', 'X27', 'Z27'])
def test_new_span_boundary_rejects_noise_and_other_banks(recording, monkeypatch, control):
    import numpy as np
    from hfmodem.shrike import p4robusttx
    audio, probe, rows = recording
    row = rows[0]
    samples, origin, good = recorded_word(recording)
    first = good.stream_sample - origin
    if control == 'noise':
        start = row['onset_sample'] + round(.4 * FS)
        samples = audio[start:start + round(.6 * FS)].copy()
    else:
        samples = np.zeros(round(.6 * FS))
        wave = p4robusttx.robust_word(control, FS) * .03
        samples[first:first + len(wave)] = wave
    # Force the missing-extent path: refusal must come from the word proof,
    # rather than a detectable energy boundary or the turn gate's other guards.
    monkeypatch.setattr(p4oracle, 'bursts', lambda *args, **kwargs: [])
    assert p4oracle._word_span_supported(*recorded_word(recording)[:1], FS,
                                       first, good.cfo_hz, ())
    assert not p4oracle._word_span_supported(samples, FS, first, 0., ())
    m, a, rx = requested(probe, row)
    for reg in rx.read_window(samples, origin, whole=True):
        assert not reg.observation.word_span_supported
        m.observe(reg, now=row['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn']
    assert not m.outbox.inflight and a.tx_acked_bytes == 0


@pytest.mark.parametrize('missing', ['half_tail', *range(1, 26)])
def test_coherent_partial_word_cannot_supply_a_complete_span(monkeypatch, missing):
    import numpy as np
    from hfmodem.shrike import p4robusttx, p4words
    symbols = p4words.BANK['V27'].copy()
    symbols[-1 if missing == 'half_tail' else missing] *= .5 if missing == 'half_tail' else 0
    wave = p4robusttx._render(symbols, FS)
    samples = np.zeros(round(.6 * FS))
    first = round(.15 * FS)
    samples[first:first + len(wave)] = wave
    monkeypatch.setattr(p4oracle, 'bursts', lambda *args, **kwargs: [])
    assert not p4oracle._word_span_supported(samples, FS, first, 0., ())


def test_complete_predicate_searches_bounded_offsets(monkeypatch):
    import numpy as np
    from hfmodem.shrike import p4robusttx, p4rx, p4words
    samples = np.zeros(round(.6 * FS))
    first = round(.15 * FS)
    for word, shift in [('V27', 0), ('X27', -140)]:
        wave = p4robusttx.robust_word(word, FS)
        samples[first + shift:first + shift + len(wave)] += wave
    monkeypatch.setattr(p4oracle, 'bursts', lambda *args, **kwargs: [])
    lo = first - round(.035 * FS)
    spread, power, _ = p4oracle._despread_window(samples[lo:first + round(.275 * FS)], FS, 0.)
    start = round((first-lo) * p4rx.SAMPLE_RATE / FS)
    slack = round(.003 * p4rx.SAMPLE_RATE)
    indices = start + np.arange(-slack, slack + 1)[:, None] + p4oracle._SYMBOL_N * np.arange(27)
    support = p4oracle._despread_fraction(spread[indices], power[indices], 1e-30)
    selected = spread[indices[support.mean(axis=1).argmax()]]
    reference = p4words.BANK['V27']
    match = abs(np.vdot(reference, selected)) / (np.linalg.norm(reference) * np.linalg.norm(selected))
    assert match < p4words.MATCH_KNEE
    assert p4oracle._word_span_supported(samples, FS, first, 0., ())


def test_incremental_acquisition_revisits_changed_extent(recording, monkeypatch):
    audio, probe, rows = recording
    row = rows[0]
    _, origin = window(audio, row)
    samples = audio[origin:origin + round(1.25 * FS)]
    m, a, rx = requested(probe, row)
    rx.scan = 'incremental'
    rx._retain_scan(samples, origin)
    acquired, = p4oracle.word_scan(samples, FS, row['cfo_hz'], origin=origin)
    extent = .250
    calls = []
    def spans(*args, **kwargs):
        calls.append(len(args[0]))
        return [] if extent is None else [(acquired.first_chip / FS, extent)]
    monkeypatch.setattr(p4oracle, 'bursts', spans)
    def scan():
        rx.cfo_hz = row['cfo_hz']
        rx._scan_eligible = True
        rx._scan_cursor = origin
        return rx.scan_idle(now=origin + len(samples), next_event=origin + len(samples) + FS)
    first, = scan()
    assert first.outcome == Registered.NEW and first.observation.extent_s == pytest.approx(.250)
    assert scan() == []
    extent = .231
    measured, = scan()
    assert measured.outcome == Registered.EXTENDED and measured.observation.extent_s == pytest.approx(.231)
    extent = None
    refined, = scan()
    assert refined.outcome == Registered.ENRICHED and refined.observation.word_span_supported
    assert refined.observation.extent_s == measured.observation.extent_s
    assert scan() == []
    assert calls == [len(samples)] * 5
    m.observe(measured, now=row['received_at_sample'])
    m.observe(refined, now=row['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn']
    assert not m.outbox.inflight and a.tx_acked_bytes == 0
