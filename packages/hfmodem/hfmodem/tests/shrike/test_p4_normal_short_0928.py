# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p4frame, p4short, p4shorttx, p4rx
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role, Kind, Observation, Registered
from hfmodem.shrike.p4observation import RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import machine
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx, FS

ROOT = Path(__file__).resolve().parents[5]
RASTER = 1667751
TEXT = b'119 daily minutes remaining with WS8EOC '
FX = Path(__file__).parent/'fixtures'/'p4-breakin'


def capture(name='stream.flac'):
    p = ROOT/'captures/onair-0928-0951'/name
    if not p.exists():
        pytest.skip('local WS8EOC recording absent')
    return onair.session.load_wav(str(p), FS)


def receiver(at=RASTER, cfo=-.7):
    r = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
    r.enable_hold()
    r.note_emission(Emission(Kind.WORD, Role.IRS, at-round(.330*FS), .242))
    r.peer_onset = at-60000
    r._peer_heads = [at-120000, at-60000]
    return r


def read_copy(r, audio, at):
    origin = max(0, at-round(.05*FS))
    early, end = at+round(.810*FS), at+round(.846*FS)
    records = r.read_partial(audio[origin:early], origin, deadline=early)
    records += r.read_window(audio[origin:end], origin, whole=True)
    return records


def normal(at=48000, status=1, data=TEXT):
    f = p4frame.decode_field(p4frame.encode_field(data.ljust(40, b'\x1e'),
                             p4frame.Status.from_byte(status)), data_bytes=40)
    return Observation(Kind.NORMAL, FS, 0, at, at, .805, -.7, .8,
                       lattice=.8, placed=True, frame=f)


def decide(m, obs):
    due = obs.stream_sample+round(.920*FS)
    m.observe(RegisteredObservation(obs, Registered.NEW), now=obs.stream_sample+round(.810*FS))
    return m.decide(due-m.lead)


def test_ws8eoc_0928_witness_shorts_decode_through_the_placed_read():
    audio = capture('witness.wav')
    start = RASTER+round(3.691*FS)
    r = receiver(start, -5.)
    frames = []
    for k in range(20):
        at = start+k*60000
        records = read_copy(r, audio, at)
        frames += [x.observation for x in records if x.observation.frame]
        r.new_cycle()
    assert len(frames) == 20
    assert all(o.frame.status.byte == 1 and o.frame.data.rstrip(b'\x1e') == TEXT for o in frames)
    assert all(abs(o.stream_sample-start-k*60000) <= 48 for k, o in enumerate(frames))


def test_ws8eoc_0928_stream_short_combines_by_the_third_copy():
    audio = capture()
    r = receiver()
    frames = []
    for k in range(3):
        records = read_copy(r, audio, RASTER+k*60000)
        frames += [x.observation for x in records if x.observation.frame]
        assert abs(r.clock.cfo_hz+.7) <= 3
        r.new_cycle()
    assert len(frames) == 1
    assert abs(frames[0].stream_sample-(RASTER+120000)) <= 240
    assert frames[0].frame.status.byte == 1
    assert frames[0].frame.data.rstrip(b'\x1e') == TEXT


def driver(audio):
    stream = Stream(audio)
    stream.pos = round(28*FS)
    tx = Tx(stream)
    tx.p4_probe.emissions = [dict(audio_start=1382736, audio_end=1419103, value=7, header='A')]
    tx.p4_probe.values = (7, 38)
    stream.pos = 1419103
    tx.settle = .044
    key = tx._tx
    def flushing(audio, what):
        key(audio, what)
        stream.pos = tx.tx_end
    tx._tx = flushing
    delivered = []
    tx.p4_link_host.deliver = delivered.append
    onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0, packet_n=36000,
                                                   cs_n=12000, d_max_n=4800))
    return tx, delivered


def test_ws8eoc_0928_v27_answers_the_short(monkeypatch):
    from hfmodem.shrike.p4hold import BreakinHold
    normal_decode = BreakinHold._normal_decode

    def controlled(self, deadline=None):
        allowance = None if deadline is None else deadline-self.now()
        clock = [0.]
        decode = self._decode

        def decoding(scores):
            result = decode(scores)
            clock[0] += .010
            return result

        with monkeypatch.context() as timing:
            timing.setattr(self, 'now', lambda: clock[0])
            timing.setattr(self, '_decode', decoding)
            return normal_decode(self, allowance)

    monkeypatch.setattr(BreakinHold, '_normal_decode', controlled)
    tx, delivered = driver(capture())
    requests = {e['at'] for e in tx.p4_probe.link_events
                if e['event'] == 'Keyed' and not e.get('current_slot_accepted', True)}
    emissions = [e for e in tx.p4_probe.emissions[1:] if e['audio_start'] not in requests]
    assert emissions[0]['word'] == 'X27'
    assert abs(emissions[0]['audio_start']-1651911) <= 96
    assert emissions[1]['word'] == 'V27'
    assert abs(emissions[1]['audio_start']-(RASTER+120000+44160)) <= 96
    assert all(e['word'] == 'V27' for e in emissions[1:])
    assert b''.join(delivered).count(TEXT) == 1
    assert any(e.get('event') == 'acked-repeat' for e in tx.p4_probe.link_events)
    assert any(e.get('ack_policy') == 'v27' for e in tx.p4_probe.link_events)
    normals = [row for row in tx.p4_probe.heard if row.get('kind') == Kind.NORMAL and row.get('placed')]
    assert normals
    assert all(abs((row['onset_sample']-RASTER+30000) % 60000-30000) <= 240 for row in normals)
    assert tx.p4_probe.reason != 'peer silent'


def test_ws8eoc_0928_shorts_keep_the_link_live():
    audio = capture()
    r = receiver()
    live = []
    for k in range(16):
        at = RASTER+k*60000
        regs = read_copy(r, audio, at)
        if regs:
            assert r.finish_cycle(at) == 0
        live.append(r.finish_cycle(at))
        r.new_cycle()
    assert max(live) <= 2
    quiet = np.zeros(25*FS)
    r.silent_cycles = 0
    for k in range(16):
        at = r.clock.next_due
        origin = at-round(.05*FS)
        r.read_window(quiet[:round(.9*FS)], origin, whole=True)
        r.finish_cycle(at)
        r.new_cycle()
    assert r.silent_cycles == 16


def test_n9vc_k0003_changeover_short_decodes_and_draws_the_grant():
    audio = onair.session.load_wav(str(FX/'n9vc-normal.wav'), FS)
    audio = np.r_[np.tile(audio[:2400], 20), audio]
    r = receiver(round(1.08*FS), 0.)
    regs = read_copy(r, audio, round(1.08*FS))
    frames = [x.observation for x in regs if x.observation.frame]
    assert len(frames) == 1
    obs = frames[0]
    assert obs.frame.status.byte == 0x41 and obs.frame.data == b'VC>\r'+b'\x1e'*36
    assert 1.0-1e-6 <= obs.cfo_hz <= 1.2+1e-6
    m, _ = machine(role='irs')
    p = decide(m, obs)
    assert p.kind == Kind.BODY and p.status == 0x40
    assert p.stream_sample == obs.stream_sample+44160


@pytest.mark.parametrize('status,word,kind,state', [
    (1, 'V27', Kind.WORD, State.IRS_ANSWER),
    (0x41, None, Kind.BODY, State.ISS_GRANT),
    (0x21, 'V27', Kind.WORD, State.IRS_CLOSE),
    (0x61, 'V27', Kind.WORD, State.IRS_ANSWER),
])
def test_p4turn_answers_normal_shorts_by_status(status, word, kind, state):
    m, _ = machine(role='irs')
    m.first_body = False
    obs = normal(status=status)
    p = decide(m, obs)
    assert p is not None and (p.kind, p.word, p.state) == (kind, word, state)
    assert p.stream_sample == obs.stream_sample+44160


def test_normal_enrichment_delivers_once_and_answers_once():
    r = receiver(48000)
    obs = normal()
    first = r.register(replace(obs, frame=None))
    reg = r.register(obs)
    assert first.outcome == Registered.NEW
    assert reg.outcome == Registered.ENRICHED and reg.replaced == first.observation
    assert reg.responds_to == first.responds_to
    assert r.register(obs).outcome == Registered.DUPLICATE
    m, _ = machine(role='irs')
    m.first_body = False
    m.observe(first, now=48000)
    m.observe(reg, now=48000)
    p = m.decide(48000+44160-m.lead)
    assert p.word == 'V27'
    m.keyed(p.stream_sample)
    assert m.decide(p.stream_sample) is None


def test_repeated_changeover_short_replays_the_grant():
    m, a = machine(role='irs')
    for k in range(4):
        p = decide(m, normal(48000+k*60000, 0x41))
        assert p and p.kind == Kind.BODY and p.status == 0x40 and p.payload == b''
        m.keyed(p.stream_sample)
        assert m.grants == k+1 and m.acked_counter == 1
        assert a._buffer_raw == 0
    p = decide(m, normal(48000+4*60000, 0x41))
    assert p.kind == Kind.SHORT


def test_acked_counter_updates_only_on_a_keyed_answer():
    m, _ = machine(role='irs')
    assert m.acked_counter is None
    p = decide(m, normal(status=0x41))
    m.refused('deadline')
    assert m.acked_counter is None
    p = decide(m, normal(108000, 0x41))
    m.keyed(p.stream_sample)
    assert m.acked_counter == 1


@pytest.mark.parametrize('status', [0x05, 0x81])
def test_single_copy_unsupported_source_still_ends_mail(status):
    at = 48000
    wave = p4shorttx.short_packet(TEXT, status)
    audio = np.zeros(3*FS)
    start = at+round(308*FS/28800)
    audio[start:start+len(wave)] = wave
    r = receiver(at, 0.)
    frames = [x.observation for x in read_copy(r, audio, at) if x.observation.frame]
    assert len(frames) == 1 and frames[0].frame.status.byte == status
    m, _ = machine(role='irs')
    decide(m, frames[0])
    assert m.adapter.failure.startswith(f'unsupported P4 {"extended" if status & 0x80 else "source"} status 0x{status:02x}')


def test_short_score_polarity_is_fixed_by_training():
    wave = p4shorttx.short_packet(b'polarity', 1)
    fields = []
    for sign in (1, -1):
        z = p4rx.audio_to_baseband(sign*wave, FS)
        result = p4short.short_scores(z, first_training_start=2240, frequency_offset=0.)
        assert result is not None
        fields.append(result[2])
        frame, pol = p4short.decode_scores(result[2])
        assert frame is not None and pol == 1
    assert p4short.decode_scores(sum(fields))[1] == 1
    assert p4short.decode_scores(-fields[0])[1] == -1


@pytest.mark.parametrize('delta', [-.0345, .0345, .036])
def test_acquisition_waits_for_the_whole_bracket(delta):
    audio = capture()
    at = RASTER+9*60000
    predicted = at-round(delta*FS)
    r = receiver(predicted)
    r.peer_onset = None
    r._peer_heads = []
    origin = predicted-round(.05*FS)
    early = predicted+round(.81*FS)
    assert r.read_partial(audio[origin:early], origin, deadline=early) == []
    assert r.observations == [] and r.hold.members == []
    records = r.read_window(audio[origin:predicted+round(.846*FS)], origin, whole=True)
    if delta > .035:
        assert r.peer_onset is None
        assert not any(x.observation.frame for x in records)
    else:
        assert records
        assert abs(records[-1].observation.stream_sample-at) <= 48
        assert len(r.hold.members) == 1
        assert r.read_window(audio[origin:predicted+round(.846*FS)], origin, whole=True) == []


def test_p4_status_rule_per_kind():
    from hfmodem.shrike.p4hold import BreakinHold
    h = BreakinHold(FS)
    h.acked_counter = 0
    for kind in ('body', 'normal'):
        h._select(kind)
        assert h.status_ok(normal(status=1).frame, combined=True)
        assert not h.status_ok(normal(status=3).frame, combined=True)
        assert not h.status_ok(normal(status=5).frame, combined=True)
        assert not h.status_ok(normal(status=0x81).frame, combined=True)
        assert h.status_ok(normal(status=5).frame, combined=False)
    assert h.status_ok(normal(status=0x61).frame, combined=True)
    h.acked_counter = 3
    assert [h.status_ok(normal(status=c).frame, combined=True) for c in range(4)] == [True, False, False, True]


@pytest.mark.parametrize('cost', [.054, .010])
def test_short_fallback_is_deferred_by_the_wall_clock(monkeypatch, cost):
    from hfmodem.shrike.p4hold import BreakinHold
    h = BreakinHold(FS)
    h._select('normal')
    clock = [0.]
    h.now = lambda: clock[0]
    decode = p4short.decode_scores
    calls = []
    monkeypatch.setattr(h, "quick_decode_times", [1.])
    def timed(scores, **kw):
        calls.append(clock[0])
        clock[0] += cost
        return decode(scores, **kw)
    monkeypatch.setattr(p4short, 'decode_scores', timed)
    bad = np.random.default_rng(92804).normal(0, 10, 1056)
    good = (1.-2*p4shorttx.coded_bits(p4frame.encode_field(
        normal().frame.data, normal().frame.status)).astype(float))*2
    assert h.normal(48000, 0, 0., .5, bad, None, deadline=.066) is None
    clock[0] = 1.252
    result = h.normal(108000, 0, 0., .5, good, None, deadline=1.316)
    if cost == .054:
        assert result is None and len(calls) == 2
        clock[0] = 1.316
        result = h.refine(deadline=1.566)
    assert result.data == normal().frame.data
    assert len(calls) == 3
    assert h.normal(108000, 0, 0., .5, good, None) == result
    assert len(calls) == 3
    assert h.normal(168000, 0, 0., .5, good, None) == result


@pytest.mark.parametrize('kind', ['short', 'body'])
def test_placed_normal_read_preserves_other_packets(kind):
    from hfmodem.shrike import p4robusttx, p4breakintx
    at = 48000
    wave = (p4robusttx.robust_short(13, 'A') if kind == 'short'
            else p4breakintx.changeover_packet(b'hello', 0))
    audio = np.r_[np.zeros(at), wave, np.zeros(FS)]
    r = receiver(at, 0.)
    records = read_copy(r, audio, at)
    assert records and not any(x.observation.kind == Kind.NORMAL for x in records)
    if kind == 'short':
        obs = records[-1].observation
        assert obs.value == 13 and obs.header == 'A'
    else:
        assert records[-1].observation.frame.data.startswith(b'hello')


def test_hold_reader_preserves_recorded_long_and_chirp():
    from scipy.io import wavfile
    from hfmodem.shrike.p4receiver import Clock, Expect
    from hfmodem.tests.shrike.test_p4chirp import REAL_FIELDS

    fs, pcm = wavfile.read(FX/'n9vc-long-ack.wav')
    audio = np.r_[np.tile(pcm[:600], 20), pcm].astype(float)/32768
    r = P4Receiver(fs, admitted=True)
    r.enable_hold()
    r.note_emission(Emission(Kind.WORD, Role.IRS, round(.75*fs), .244))
    early = round(1.926*fs)
    records = r.read_partial(audio[:early], 0, deadline=early)
    assert records[-1].observation.kind == Kind.LONG
    records = r.read_window(audio[:round(4.40*fs)], 0, whole=True)
    assert records[-1].observation.kind == Kind.LONG
    chirps = FX.parent/'sigidwiki/p4-chirps/audio.npz'
    if not chirps.exists():
        pytest.skip(f"{chirps} is absent")
    with np.load(chirps) as fixture:
        fs = int(fixture['fs'])
        for row, field in zip(fixture['audio'], REAL_FIELDS):
            r = P4Receiver(fs, admitted=True, chirp=True)
            r.enable_hold()
            r.expected = True
            r.clock = Clock(Expect.PACKET, fs//3, .33, 3.75, -242.4)
            records = r.read_window(np.r_[np.zeros(fs//3), row, np.zeros(fs//3)], 0, whole=True)
            assert len(records) == 1 and records[0].observation.kind == Kind.CHIRP
            frame = records[0].observation.frame
            assert p4frame.encode_field(frame.data, frame.status) == field


def test_late_normal_enrichment_is_delivered_without_a_late_key():
    m, _ = machine(role='irs')
    m.first_body = False
    obs = normal()
    m.observe(RegisteredObservation(obs, Registered.ENRICHED), now=obs.stream_sample+44160)
    assert m.adapter.last == (1, obs.frame.data)
    assert m.decide(obs.stream_sample+44160) is None
    following = replace(obs, stream_sample=obs.stream_sample+60000)
    assert decide(m, following).word == 'V27'


def test_data_response_repeat_keeps_the_same_inflight_bytes():
    m, a = machine(role='irs', data='body')
    a.on_host_data(b'FC test')
    m.state = State.IRS_BREAKIN
    obs = normal(status=0x40)
    p = decide(m, obs)
    assert p.kind == Kind.BODY and p.payload
    m.keyed(p.stream_sample)
    original = m.outbox.inflight
    count = a._buffer_raw
    repeated = decide(m, replace(obs, stream_sample=obs.stream_sample+60000))
    assert repeated.payload == original and repeated.repeat
    m.keyed(repeated.stream_sample)
    assert m.outbox.inflight == original and a._buffer_raw == count
    assert m.data_repeats == 1


def test_normal_short_false_accept_corpus():
    import json
    from hfmodem.tests.shrike.p4_normal_corpus import run
    test_ws8eoc_0928_stream_short_combines_by_the_third_copy()
    result = run()
    print('NORMAL_CORPUS_RESULT '+json.dumps(result), flush=True)


@pytest.mark.parametrize('cost, acquisition', [(.054, 0), (.010, 0), (.054, .0345), (.054, -.0345)])
def test_driver_fallback_obeys_elapsed_time_after_flushed_keys(monkeypatch, cost, acquisition):
    from hfmodem.shrike import p4breakintx
    audio = np.random.default_rng(17).normal(0, .0001, 8*FS)
    at = round(.923*FS)
    greeting = p4breakintx.changeover_packet(b'hello', 0)
    audio[at:at+len(greeting)] += greeting
    wave = p4shorttx.short_packet(TEXT, 1)
    starts = [at+60000*k for k in range(1, 4)]
    for start in starts:
        head = start+round(308*FS/28800)
        audio[head:head+len(wave)] += wave
    stream = Stream(audio)
    wall = [0]
    stream.sample_now = lambda: max(stream.pos, wall[0])
    take = stream.take_until
    def taking(end):
        part = take(end)
        wall[0] = max(wall[0], stream.pos)
        return part
    stream.take_until = taking
    stream.clamp_late = lambda at: max(0, stream.sample_now()-at)
    monkeypatch.setattr(onair.time, 'monotonic', lambda: wall[0]/FS)
    tx = Tx(stream)
    tx.settle = .044
    tx.sessrx.p4.enable_hold()
    tx.sessrx.p4.hold.now = lambda: wall[0]/FS
    if acquisition:
        rx = tx.sessrx.p4
        note = rx.note_emission
        shifted = []
        def noting(emission):
            note(emission)
            if emission.role == Role.IRS and not shifted:
                rx.peer_onset = None
                rx._peer_heads = []
                rx.clock.next_due -= round(acquisition*FS)
                shifted.append(True)
        monkeypatch.setattr(rx, 'note_emission', noting)
    key = tx._tx
    def flushing(audio, what):
        assert stream.sample_now() <= tx.raster.boundary(tx.slot)
        key(audio, what)
        stream.pos = tx.tx_end
        wall[0] = max(wall[0], stream.pos)
    tx._tx = flushing
    scores = p4short.short_scores
    scored = []
    def scoring(*a, **kw):
        result = scores(*a, **kw)
        if result is not None:
            wall[0] += round(.002*FS)
            coh, cfo, fields = result
            if not scored:
                fields = np.random.default_rng(9).normal(0, np.mean(abs(fields))*10, 1056)
            scored.append(1)
            return coh, cfo, fields
        return result
    monkeypatch.setattr(p4short, 'short_scores', scoring)
    decode = p4short.decode_scores
    calls = []
    monkeypatch.setattr(tx.sessrx.p4.hold, "quick_decode_times", [1.])
    def decoding(fields, **kw):
        calls.append(wall[0])
        wall[0] += round(cost*FS)
        return decode(fields, **kw)
    monkeypatch.setattr(p4short, 'decode_scores', decoding)
    onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0, packet_n=36000,
                                                   cs_n=12000, d_max_n=4800))
    requests = {e['at'] for e in tx.p4_probe.link_events
                if e['event'] == 'Keyed' and not e.get('current_slot_accepted', True)}
    words = [e for e in tx.p4_probe.emissions[1:] if e['audio_start'] not in requests]
    assert words[0]['word'] == 'X27'
    assert words[1]['word'] == 'V27'
    answered = starts[2] if cost == .054 else starts[1]
    assert abs(words[1]['audio_start']-(answered+44160)) <= 96
    assert len(scored) == 3


def test_settled_data_response_cannot_be_replayed_from_the_ack_cache():
    from hfmodem.tests.shrike.test_p4turn import word
    m, a = machine(role='irs', data='body')
    a.on_host_data(b'FC test')
    m.state = State.IRS_BREAKIN
    obs = normal(status=0x40)
    p = decide(m, obs)
    m.keyed(p.stream_sample)
    m.observe(word('X27', p.stream_sample+round(.92*FS), responds_to=p.stream_sample))
    assert m.outbox.inflight == b'' and a._buffer_raw == 0
    again = decide(m, replace(obs, stream_sample=obs.stream_sample+120000))
    assert again is None or again.payload == b''
    assert a._buffer_raw == 0


def test_normal_enrichment_retains_a_previously_keyed_answer():
    r = receiver(48000)
    obs = normal()
    first = r.register(replace(obs, frame=None))
    r._answered[obs.stream_sample] = 92160
    enriched = r.register(obs)
    assert enriched.outcome == Registered.ENRICHED
    assert enriched.answered == 92160 and enriched.responds_to == first.responds_to
    m, _ = machine(role='irs')
    m.observe(enriched, now=92160)
    assert m.adapter.last == (1, obs.frame.data)
    assert m.decide(92160) is None


def test_combined_body_accepts_the_next_counter():
    from hfmodem.shrike import p4breakintx
    from hfmodem.shrike.p4hold import BreakinHold, Copy
    from hfmodem.tests.shrike.test_p4turn import word
    wire = p4frame.encode_field(b'counter one'.ljust(31, b'\x1e'), p4frame.Status(1))
    fields = 2*(1.-2*p4breakintx.coded_bits(wire).astype(float))
    h = BreakinHold(FS)
    h.acked_counter = 0
    h.members = [Copy(2400, 0, 0., 2., fields.copy())]
    h.last = 2400
    path = ROOT/'captures/onair-0927-1544/stream.flac'
    if not path.exists():
        pytest.skip('local floor recording absent')
    audio = onair.session.load_wav(str(path), FS)[:round(1.1*FS)].copy()
    audio *= .001/np.sqrt(np.mean(audio**2))
    wave = p4breakintx.changeover_packet(b'counter one', 1)
    audio[2400:2400+len(wave)] += wave
    frame = h.inline(audio, 60000, word('Y27', 62400).observation)
    assert frame is not None and frame.status.byte == 1
    assert frame.data.startswith(b'counter one')
    assert len(h.members) == 2
