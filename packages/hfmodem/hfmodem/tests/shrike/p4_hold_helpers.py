# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from functools import lru_cache
from time import monotonic

from hfmodem.tests.shrike import p4_stock_peer as stock

import numpy as np

from hfmodem.shrike import p4oracle, p4turn
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role, Kind
from hfmodem.tests.shrike.test_p4turn import machine

FS = 48000


@lru_cache(maxsize=16)
def answer_audio(name, weak, *, early_full=False):
    waveform = stock.word(name)
    first = round(.05*FS)
    from hfmodem.tests.shrike.test_p4_breakin_hold_0927 import capture

    floor = capture('onair-0927-1549')
    noise = floor[-round(.38*FS):].copy()
    noise *= .15 / np.sqrt(np.mean(noise**2))
    for level in (np.linspace(.005, .18, 100) if weak else [10.]):
        audio = noise.copy()
        audio[first:first+len(waveform)] += level*waveform
        got = p4oracle.word_at(audio, FS, 0, .05, cfo_hz=0.,
                               expected_words={'V27', 'X27', 'Y27', 'Z27', 'U27'})
        if got and got.word == name and (.76 <= got.score <= .86 if weak else got.score >= .95):
            early = p4oracle.word_at(audio[:first+round(.215*FS)], FS, 0, .05,
                cfo_hz=0., provisional=True, expected_words={'V27', 'X27', 'Y27', 'Z27', 'U27'})
            if early_full:
                confirmed = p4oracle.word_at(audio[:first+round(.245*FS)], FS, 0, .05,
                    cfo_hz=0., whole=True, expected_words={'V27', 'X27', 'Y27', 'Z27', 'U27'})
                if confirmed is None or confirmed.word != name or confirmed.weak != weak:
                    continue
            if not weak or early is not None and early.word == name and early.weak:
                return audio
    raise AssertionError(f'no {name} {weak=} fixture found')


def check_settlement():
    for sequence, commits in [
        (['weak', 'full'], [False, True]),
        (['weak', 'weak'], [False, True]),
        (['weak', 'U27', 'weak', 'weak'], [False, False, False, True]),
        (['weak', None, 'weak', 'weak'], [False, False, False, True]),
        (['weak', 'U27']*4, [False]*8),
    ]:
        turn, queue = machine(data='body')
        turn.weak_pairs = True
        queue.on_host_data(b'A'*31)
        turn._data(0, p4turn.State.ISS_DATA_ACK)
        receiver = P4Receiver(FS, admitted=True, breakin=True)
        receiver.enable_hold()
        previous = None
        for kind, committed in zip(sequence, commits):
            sent = turn.keyed()
            receiver.note_emission(Emission(Kind.BODY, Role.ISS, sent.stream_sample, .80))
            receiver.expected_words = turn.expected_words()
            onset = sent.stream_sample+round(.9426*FS)
            start = onset-round(.05*FS)
            audio = np.zeros(round(.38*FS)) if kind is None else answer_audio('U27' if kind == 'U27' else 'X27', kind == 'weak')
            records = receiver.read_window(audio, start, whole=True)
            for reg in records:
                assert reg.responds_to == sent.stream_sample
                turn.observe(reg, now=onset+round(.24*FS))
                if kind == 'U27':
                    assert reg.observation.unexpected
            if turn._next is None:
                turn.silence(onset+round(.25*FS))
            assert (queue._buffer_raw == 0) == committed, (sequence, kind, records)
            assert queue._buffer_raw == len(queue._outbuf)+len(turn.outbox.inflight)
            if committed:
                assert turn.decide(onset).value == 13
            elif kind == 'weak' and turn.outbox.inflight:
                assert turn.decide(onset).stream_sample == sent.stream_sample+120000
                if previous is not None and previous == sent.stream_sample:
                    raise AssertionError('duplicate key')
            previous = sent.stream_sample
        if len(sequence) == 8:
            assert not turn.outbox.inflight and len(queue._outbuf) == 31
    turn, queue = machine(data='body')
    turn.weak_pairs = False
    queue.on_host_data(b'A'*31)
    turn._data(0, p4turn.State.ISS_DATA_ACK)
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.enable_hold()
    for _ in range(2):
        sent = turn.keyed()
        receiver.note_emission(Emission(Kind.BODY, Role.ISS, sent.stream_sample, .8))
        receiver.expected_words = turn.expected_words()
        start = sent.stream_sample+round((.9426-.05)*FS)
        for reg in receiver.read_window(answer_audio('X27', True), start, whole=True):
            turn.observe(reg)
            turn.observe(reg)
        assert queue._buffer_raw == 31


def check_corpus(root, evidence=None):
    from hfmodem.shrike.p4hold import BreakinHold, Copy
    from hfmodem.shrike import onair, p4breakintx, p4breakin
    from hfmodem.tests.shrike.test_p4turn import word
    import json
    import pytest

    rng = np.random.default_rng(92704)
    streams = []
    phases = (35.9772, 12.2445, 25.9974, 15.999)
    for arm in ('1544', '1547', '1549', '1553'):
        path = root/'captures'/f'onair-0927-{arm}'/'stream.flac'
        if not path.exists():
            pytest.skip('local false-accept corpus absent')
        streams.append(onair.session.load_wav(str(path), FS))
    counts = dict(capture_epochs=0, awgn_epochs=0, noise_copies=0, noise_accepts=0,
                  decoder_sums=0, decoder_crc_passes=0, decoder_accepts=0,
                  mixed_epochs=0, weak_epochs=0, mixed_accepts=0,
                  held_similar=0, similar_copies=0, post_key_accepts=0, safety_exits=0)
    noise_scores = []
    size, pad = round(1.1*FS), round(.05*FS)

    def epoch(copies, frames=None, cfo=0., keys=False):
        holder = BreakinHold(FS)
        holder.acked_counter = 1
        for index, audio in enumerate(copies):
            origin = index*60000
            from dataclasses import replace
            head = replace(word('Y27', origin+pad).observation, cfo_hz=cfo)
            before = len(holder.events)
            frame = holder.inline(audio, origin, head)
            frame = frame or holder.refine(deadline=monotonic()+.250)
            if frames is None:
                counts['noise_copies'] += 1
                counts['noise_accepts'] += frame is not None
                assert frame is None, (counts, holder.events)
                if holder.members:
                    noise_scores.append(holder.members[-1].llr.copy())
            elif frame is not None:
                assert (frame.status.byte, frame.data) in frames
                if len(holder.members) > 1:
                    assert frame.status.byte in (0, 0x20, 0x40)
                counts['mixed_accepts'] += 1
                if keys:
                    assert not any(e['event'] == 'held-same' for e in holder.events[before:])
                    counts['post_key_accepts'] += 1
                    holder.reset('our key')
        return holder

    for index in range(456):
        arm = index % 4
        stream = streams[arm]
        n = 10 if index < 176 else 11
        candidates = np.arange(0, len(stream)/FS-(n-1)*1.25-size/FS, .05)
        candidates = candidates[np.abs((candidates+.05-phases[arm]+.625) % 1.25-.625) >= .100]
        start = float(candidates[(index//4*37) % len(candidates)])
        copies = [stream[round((start+k*1.25)*FS):round((start+k*1.25)*FS)+size] for k in range(n)]
        epoch(copies, cfo=float(rng.uniform(-8, 8)))
        counts['capture_epochs'] += 1
        if index % 40 == 0:
            print('HOLD CAPTURE '+json.dumps(counts), flush=True)
    for index in range(250):
        epoch([rng.normal(0, .1, size) for _ in range(12)])
        counts['awgn_epochs'] += 1
        if index % 40 == 0:
            print('HOLD AWGN '+json.dumps(counts), flush=True)
    assert counts['noise_copies'] == 7840
    if evidence is not None:
        np.savez_compressed(evidence/'noise-scores.npz', scores=np.asarray(noise_scores),
                            rng_state=json.dumps(rng.bit_generator.state))
        print('HOLD EVIDENCE '+str(evidence), flush=True)
    for index in range(40000):
        selected = rng.integers(0, len(noise_scores), size=int(rng.integers(1, 13)))
        frame = p4breakin.decode_scores(sum(noise_scores[i] for i in selected))
        if frame is not None:
            counts['decoder_crc_passes'] += 1
            holder = BreakinHold(FS)
            holder.acked_counter = 1
            holder.members = [Copy(k*60000, 0, 0., float(np.mean(abs(noise_scores[i]))),
                                   noise_scores[i]) for k, i in enumerate(selected)]
            accepted = holder._accept()
            counts['decoder_accepts'] += accepted is not None
            detail = dict(index=index, selected=selected.tolist(), status=frame.status.byte,
                          data=frame.data.hex(), events=holder.events)
            print('HOLD NOISE CRC '+json.dumps(detail), flush=True)
            assert accepted is None, detail
        counts['decoder_sums'] += 1
        if index % 5000 == 0:
            print('HOLD DECODER '+str(index), flush=True)

    def render(data):
        wave = p4breakintx.changeover_packet(data, 0)
        return np.pad(wave, (pad, size-pad-len(wave)))

    def noisy(wave, snr):
        power = np.mean(wave[pad:pad+round(.8*FS)]**2)
        return wave+rng.normal(0, np.sqrt(power*10**(-snr/10)), len(wave))

    real_fields = [b'RMS Trimode 1.4.4.0 KB5LZK at t', b'RMS Trimode 1.4.4.0\r\nW9SSJ has ']
    real_waves = []
    for stream, onset in ((streams[0], 38.4772), (streams[1], 12.2445)):
        cut = stream[round((onset-.05)*FS):round((onset-.05)*FS)+size].copy()
        real_waves.append(cut/np.sqrt(np.mean(cut**2)))
    for index in range(196):
        fields = real_fields if index < 80 else [rng.bytes(31) for _ in range(3)]
        waves = real_waves if index < 80 else [render(data) for data in fields]
        choices = ([k % 2 for k in range(10)] if index % 4 == 0 else
                   [int(k >= 5) for k in range(10)] if index % 4 == 1 else
                   [int(k < 5) for k in range(10)] if index % 4 == 2 else
                   rng.integers(0, len(waves), 10))
        copies = [noisy(waves[c], float(rng.uniform(-10, -2) if index < 80 else rng.uniform(-16, -12))) for c in choices]
        frames = {(0, data) for data in fields}
        epoch(copies, frames)
        epoch(copies, frames, keys=True)
        counts['mixed_epochs'] += 1
        if index % 20 == 0:
            print('HOLD MIXED '+json.dumps(counts), flush=True)
    for _ in range(120):
        field = rng.bytes(31)
        wave = render(field)
        epoch([noisy(wave, float(rng.uniform(-19, -14))) for _ in range(12)], {(0, field)})
        counts['weak_epochs'] += 1
    counts.update(check_hold_safety(streams[0]))
    print('HOLD CORPUS '+json.dumps(counts), flush=True)
    if evidence is not None:
        (evidence/'corpus.json').write_text(json.dumps(counts, indent=2)+'\n')


def check_driver_settlement(monkeypatch, *, answer_offset=.9426, retry_at=120000):
    from hfmodem.shrike import onair
    from hfmodem.tests.shrike.test_p4_link_0927 import Tx
    from hfmodem.tests.shrike.test_ws8eoc_p4_silent_0927 import Live, grid

    for answers, pairs, should_commit in ((('weak', 'full'), True, True),
                                          (('weak', 'weak'), True, True),
                                          (('weak', 'weak'), False, False)):
        with monkeypatch.context() as patch:
            live = Live(np.zeros(20*FS))
            tx = Tx(live)
            tx.p4_probe.emissions.clear()
            tx.p4_link_options.p4_data = 'body'
            tx.p4_link_options.p4_link_timeout = 5.
            queue = tx.p4_link_host.arq
            receiver = tx.sessrx.p4
            if retry_at == 120000:
                # Stock placement can confirm this weak waveform before the
                # deadline. Charge its full decode to just after that deadline
                # so this remains the late-confirmation contract, independently
                # of carrier refinement or CPU speed. The early case is unaltered.
                late_until = 0
                now = live.sample_now
                read_window = receiver.read_window
                def sample_now():
                    return max(now(), late_until)
                def late_confirmation(*args, **kwargs):
                    nonlocal late_until
                    records = read_window(*args, **kwargs)
                    if any(r.responds_to == 0 and r.observation.word == 'X27'
                           and r.observation.weak and not r.observation.provisional
                           for r in records):
                        late_until = 60000-2131+1
                    return records
                patch.setattr(live, 'sample_now', sample_now)
                patch.setattr(receiver, 'read_window', late_confirmation)
            queue.on_host_data(b'A'*31)
            machines, emissions, settlements, confirmations = [], [], [], []
            init = p4turn.P4Turn.__init__
            settle = queue.settle_bytes
            def settled(wire):
                settlements.append(wire)
                settle(wire)
            patch.setattr(queue, 'settle_bytes', settled)
            def begin(turn, *a, **kw):
                init(turn, *a, **kw)
                machines.append(turn)
                turn.weak_pairs = pairs
                turn._data(0, p4turn.State.ISS_DATA_ACK)
                turn.keyed()
                receiver.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8))
                observe = turn.observe
                def observed(reg, *args, **kwargs):
                    if (reg.responds_to == 0 and reg.observation.word == 'X27'
                            and reg.observation.weak and not reg.observation.provisional):
                        confirmations.append(live.sample_now())
                    return observe(reg, *args, **kwargs)
                patch.setattr(turn, 'observe', observed)
            patch.setattr(p4turn.P4Turn, '__init__', begin)
            def place(at, kind):
                piece = answer_audio('X27', kind == 'weak', early_full=retry_at == 60000)
                start = at+round((answer_offset-.05)*FS)
                live.audio[start:start+len(piece)] = piece
            place(0, answers[0])
            key = tx._tx
            def transmit(wave, what):
                key(wave, what)
                proposal = machines[0]._next
                emissions.append(proposal)
                assert queue._buffer_raw == len(queue._outbuf)+len(machines[0].outbox.inflight)
                if proposal.kind == Kind.BODY and len(emissions) < len(answers):
                    place(tx.tx_audio_start, answers[len(emissions)])
                else:
                    live.audio = live.audio[:live.pos]
            patch.setattr(tx, '_tx', transmit)
            patch.setattr(onair, '_prekey_lead', lambda *a: 2131)
            onair._run_p4_link(tx, live, grid())
            assert emissions[0].kind == Kind.BODY and emissions[0].stream_sample == retry_at
            if retry_at == 60000:
                assert confirmations and confirmations[0] <= retry_at-2131
                assert not any(e['event'] == 'Refused' for e in tx.p4_probe.link_events)
            else:
                assert confirmations and confirmations[0] > 60000-2131
            assert emissions[0].payload == b'A'*31
            assert bool(settlements) == should_commit, (emissions, machines[0].events)
            assert len(settlements) <= 1
            if should_commit:
                assert emissions[1].kind == Kind.SHORT and emissions[1].value == 13
            assert queue._buffer_raw == len(queue._outbuf)


@lru_cache(maxsize=1)
def speculative_audio():
    waveform = stock.word('X27')
    first = round(.05*FS)
    base = np.pad(waveform, (first, round(.38*FS)-first-len(waveform)))
    for count in range(2, 7):
        audio = base.copy()
        audio[first+round((27-count)/112.5*FS):] *= -1
        early = p4oracle.word_at(audio[:first+round(.215*FS)], FS, 0, .05,
                                 cfo_hz=0., provisional=True, expected_words={'V27','X27','Y27'})
        full = p4oracle.word_at(audio, FS, 0, .05, cfo_hz=0., expected_words={'V27','X27','Y27'})
        if early and full and early.word == full.word == 'X27' and not early.weak and full.weak:
            return audio
    raise AssertionError('no speculative-then-weak waveform')


def check_weak_edge_cases():
    from hfmodem.shrike.p4observation import Registered

    turn, queue = machine(data='body')
    turn.weak_pairs = True
    queue.on_host_data(b'A'*31)
    turn._data(0, p4turn.State.ISS_DATA_ACK)
    turn.keyed()
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.enable_hold()
    receiver.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8))
    receiver.expected_words = turn.expected_words()
    onset = round(.9426*FS)
    start = onset-round(.05*FS)
    audio = speculative_audio()
    partial = receiver.read_partial(audio[:round(.265*FS)], start,
                                     deadline=onset+round(.215*FS))
    assert partial and partial[0].observation.word == 'X27' and not partial[0].observation.weak
    for reg in partial:
        turn.observe(reg, now=onset+round(.215*FS))
    short = turn.keyed()
    assert short.kind == Kind.SHORT and short.value == 13
    clock = receiver.clock
    receiver.note_emission(Emission(Kind.SHORT, Role.ISS, short.stream_sample, .758))
    receiver.clock = clock
    records = receiver.read_window(audio, start, whole=True)
    confirmed = next(r for r in records if r.outcome == Registered.CONFIRMED)
    assert confirmed.responds_to == 0 and confirmed.answered == short.stream_sample
    assert confirmed.observation.weak
    turn.weak_ack = 0
    turn.observe(confirmed, now=short.stream_sample+round(.8*FS))
    assert turn.weak_ack is None and queue._buffer_raw == 31
    assert turn._next.kind == Kind.BODY
    assert turn._next.stream_sample == short.stream_sample+round(1.25*FS)
    receiver.note_emission(Emission(Kind.SHORT, Role.ISS, short.stream_sample, .758))
    start = short.stream_sample+round((.9426-.05)*FS)
    for reg in receiver.read_window(answer_audio('X27', True), start, whole=True):
        assert reg.responds_to == short.stream_sample
        turn.observe(reg)
    assert queue._buffer_raw == 31 and turn.weak_ack is None
    retry = turn.keyed()
    receiver.note_emission(Emission(Kind.BODY, Role.ISS, retry.stream_sample, .8))
    start = retry.stream_sample+round((.9426-.05)*FS)
    for reg in receiver.read_window(answer_audio('X27', True), start, whole=True):
        turn.observe(reg)
        turn.observe(reg)
    assert queue._buffer_raw == 31 and turn.weak_ack == retry.stream_sample
    following = turn.keyed()
    receiver.note_emission(Emission(Kind.BODY, Role.ISS, following.stream_sample, .8))
    start = following.stream_sample+round((.9426-.05)*FS)
    for reg in receiver.read_window(answer_audio('Y27', False), start, whole=True):
        turn.observe(reg)
    assert queue._buffer_raw == 31 and turn.weak_ack is None

    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.enable_hold()
    for at in (0, 60000):
        receiver.note_emission(Emission(Kind.SHORT, Role.ISS, at, .758))
        start = at+round((.9426-.05)*FS)
        receiver.read_window(answer_audio('X27', False), start, whole=True)
    receiver.new_cycle()
    receiver.note_refusal(Kind.BODY, 120000)
    start = 120000+round((.9426-.05)*FS)
    records = receiver.read_window(answer_audio('X27', True), start, whole=True)
    assert records and all(reg.responds_to is None for reg in records)
    for reg in records:
        turn.observe(reg)
    assert queue._buffer_raw == 31 and turn.weak_ack is None

    turn, queue = machine(data='body')
    turn.weak_pairs = True
    queue.on_host_data(b'A'*31+b'B'*31)
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.enable_hold()
    for at, weak in ((0, True), (60000, False), (180000, True)):
        if at != 60000:
            turn._data(at, p4turn.State.ISS_DATA_ACK)
        sent = turn.keyed()
        receiver.note_emission(Emission(sent.kind, Role.ISS, at, .8))
        receiver.expected_words = turn.expected_words()
        start = at+round((.9426-.05)*FS)
        for reg in receiver.read_window(answer_audio('V27' if turn.counter & 1 else 'X27', weak), start, whole=True):
            turn.observe(reg)
        assert queue._buffer_raw == (62 if at == 0 else 31)
        assert queue._buffer_raw == len(queue._outbuf)+len(turn.outbox.inflight)
    assert turn.outbox.inflight == b'B'*31



def check_hold_safety(stream):
    from hfmodem.shrike.p4hold import BreakinHold
    from hfmodem.shrike import p4breakintx
    from hfmodem.tests.shrike.test_p4turn import word

    rng = np.random.default_rng(92705)
    pad, size = round(.05*FS), round(1.1*FS)
    fields = [b'RMS Trimode 1.4.4.0 KB5LZK at t', b'RMS Trimode 1.4.4.0\r\nW9SSJ has ']
    def render(data):
        wave = p4breakintx.changeover_packet(data, 0)
        return np.pad(wave, (pad, size-pad-len(wave)))
    def held(data):
        holder = BreakinHold(FS)
        holder.offset = 32
        audio = render(data)+rng.normal(0, .02, size)
        assert holder.inline(audio, 0, word('Y27', pad).observation) is not None
        return holder
    counts = dict(safety_exits=0, similar_copies=0, held_similar=0)
    for _ in range(24):
        holder = held(fields[0])
        other = render(rng.bytes(31))
        assert holder.inline(other, 60000, word('Y27', 60000+pad).observation) is None
        assert any(e.get('reason') == 'different body' for e in holder.events)
        counts['safety_exits'] += 1
    for k in range(218):
        holder = held(fields[1])
        onset = 37.2272+(k % 11)*1.25
        start = round((onset-.05)*FS)
        heard = holder.inline(stream[start:start+size], 60000,
                              word('Y27', 60000+pad).observation)
        counts['similar_copies'] += 1
        counts['held_similar'] += heard is not None and heard.data == fields[1]
    return counts
