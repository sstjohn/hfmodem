# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Trimode level steps from the September 29 recordings.

Quiet-machine variant: HFMODEM_REALTIME=1 pytest -m realtime <this module>.
"""
import os
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path

import pytest

from hfmodem.shrike import session
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Kind, Registered
from hfmodem.tests.shrike.test_p4turn import machine

ROOT = Path(__file__).resolve().parents[5]
FS = 48000
PINS = [
    ('2320', 'witness', 25.935, -1.6, 1, 0xd97b, 128),
    ('2320', 'witness', 27.185, -1.6, 1, 0xd97b, 128),
    ('2320', 'witness', 43.435, -1.6, 1, 0xd97b, 128),
    ('2320', 'stream', 43.495, 2.2, 1, 0xd97b, 128),
    ('1859', 'stream', 29.746, -4.3, 0, 0x8ce4, 84),
    ('2045', 'stream', 124.747, 4.3, 1, 0x84f0, 84),
    ('2200', 'stream', 50.997, -3.9, 0, 0x7e59, 84),
]


@lru_cache(None)
def audio(arm, source):
    path = ROOT / 'captures' / f'onair-0929-{arm}' / (source + ('.flac' if source == 'stream' else '.wav'))
    if not path.exists():
        pytest.skip(f'local recording absent: {path}')
    return session.load_wav(str(path), FS)


def receiver(at, cfo, counter):
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, cfo)
    rx.peer_onset = at - 60000
    rx._peer_heads = [at - 120000, at - 60000]
    rx.acked_counter = (counter - 1) % 4
    return rx


def replay(pin, rx=None):
    arm, source, seconds, cfo, status, crc, size = pin
    at = round(seconds * FS)
    rx = receiver(at, cfo, status) if rx is None else rx
    start = at - round(.05 * FS)
    records = rx.read_window(audio(arm, source)[start:at+round(.846*FS)], start, whole=True)
    return rx, records


@pytest.mark.parametrize('pin', PINS, ids=lambda p: f'{p[0]}-{p[1]}-{p[2]}')
def test_recorded_levelstep_registers_and_gets_normal_ack(pin):
    rx, records = replay(pin)
    normal = [r for r in records if r.observation.kind == Kind.NORMAL and r.observation.frame]
    assert len(normal) == 1
    reg = normal[0]
    frame = reg.observation.frame
    assert reg.outcome in (Registered.NEW, Registered.ENRICHED)
    assert (frame.status.byte, frame.crc, len(frame.data)) == (pin[4], pin[5], pin[6])
    turn, _ = machine(role='irs')
    received = []
    turn.adapter.host = SimpleNamespace(deliver=received.append, supervisor=lambda *a: None)
    at = reg.observation.stream_sample
    turn.observe(reg, now=at+round(.810*FS))
    proposal = turn.decide(at+round(.920*FS)-turn.lead)
    assert proposal is not None and proposal.kind == Kind.WORD
    assert proposal.word == 'V27'
    if pin[0] == '1859':
        from hfmodem.winlink import B2FSession
        mail = B2FSession('W9SSJ', target='WS8EOC', password='offline-test-password')
        for payload in received:
            mail.feed(payload)
        assert 'CMS via WS8EOC >' in mail.remote_text
    if pin[0] == '2200':
        assert frame.data == b'\x1e' * 84 and frame.status.byte == 0


def test_1859_recorded_greeting_reaches_mail_after_counter_wrap():
    from hfmodem.winlink import B2FSession
    turn, _ = machine(role='irs')
    mail = B2FSession('W9SSJ', target='WS8EOC', password='offline-test-password')
    turn.adapter.host = SimpleNamespace(deliver=mail.feed, supervisor=lambda *a: None)
    for seconds, status in ((20.996, 1), (23.496, 2), (28.496, 3), (29.746, 0)):
        _, records = replay(('1859', 'stream', seconds, -4.3, status, None, None))
        reg, = [r for r in records if r.observation.frame]
        at = reg.observation.stream_sample
        turn.observe(reg, now=at+round(.810*FS))
        assert turn.decide(at+round(.920*FS)-turn.lead).word == 'V27'
        turn.keyed()
    assert 'CMS via WS8EOC >' in mail.remote_text
    assert mail.remote_sid and mail.challenge and mail.sent_text and not mail.failure


def test_2200_peer_raster_still_read_while_request_waits_for_grant():
    from hfmodem.shrike.p4receiver import Emission, Role
    pin = PINS[-1]
    at = round(pin[2]*FS)
    rx = receiver(at, pin[3], 0)
    rx.note_emission(Emission(Kind.WORD, Role.ISS, at-round(.330*FS), .244))
    rx.peer_normal_pending = True
    grant_due = rx.clock.next_due
    _, records = replay(pin, rx)
    reg, = [r for r in records if r.observation.kind == Kind.NORMAL and r.observation.frame]
    assert reg.observation.frame.status.byte == 0
    assert rx.clock.expects == Expect.WORD and rx.clock.next_due == grant_due
    from hfmodem.shrike.p4turn import State
    turn, _ = machine(role='irs')
    turn.state = State.IRS_BREAKIN
    turn.observe(reg, now=at+round(.810*FS))
    proposal = turn.decide(reg.observation.stream_sample+round(.920*FS)-turn.lead)
    assert proposal.kind == Kind.WORD and proposal.word == 'V27'
    assert proposal.state == State.IRS_ANSWER


def test_levelstep_does_not_inherit_bpsk_payload_or_combine_a_new_body():
    from hfmodem.shrike import p4frame, p4shorttx
    pin = PINS[-1]
    at = round(pin[2]*FS)
    rx = receiver(at, pin[3], 0)
    data = b'old normal'.ljust(40, b'\x1e')
    wire = p4frame.encode_field(data, p4frame.Status.from_byte(3))
    scores = 4*(1.-2*p4shorttx.coded_bits(wire))
    rx.hold.acked_counter = 2
    rx.hold.normal(at-60000, 0, pin[3], .8, scores, None)
    assert rx.hold.frame is not None
    _, records = replay(pin, rx)
    reg, = [r for r in records if r.observation.frame]
    assert reg.observation.frame.data == b'\x1e'*84
    assert not rx.hold.members and rx.hold.frame is None
    assert rx._normal_level == 2
    # No ACK or reset between an SL7 idle and an SL8 greeting. The new CRC
    # must establish its own data, even when both have the same training.
    next_pin = PINS[0]
    next_at = at+60000
    start = next_at-round(.05*FS)
    source_at = round(next_pin[2]*FS)
    crop = audio('2320', 'witness')[source_at-round(.05*FS):source_at+round(.846*FS)]
    rx.clock = Clock(Expect.PACKET, next_at, .330, 1.25, -1.6)
    rx.acked_counter = 0
    records = rx.read_window(crop, start, whole=True)
    reg, = [r for r in records if r.observation.frame]
    assert reg.observation.frame.crc == 0xd97b and rx._normal_level == 3
    assert not rx.hold.members


def test_existing_bpsk_and_2243_robust_long_still_decode():
    _, records = replay(('2200', 'stream', 48.497, -3.9, 3, None, None))
    reg, = [r for r in records if r.observation.frame]
    assert reg.observation.frame.crc == 0x3bd2
    from hfmodem.shrike import p4oracle, p4robustrx
    for seconds, crc in ((50.997, 0x990c), (84.747, 0xeee0)):
        _, symbols, _ = p4oracle.placed_despread(audio('2243', 'stream'), FS, seconds,
                                                -10.57, count=373, slack_s=.025)
        reception = p4robustrx.decode_symbols(symbols)
        assert reception and reception.frame.crc == crc and reception.speedlevel == 3


@lru_cache(None)
def scores(pin):
    from hfmodem.shrike import p4rx, p4short
    at = round(pin[2]*FS)
    crop = audio(pin[0], pin[1])[at-round(.05*FS):at+round(.846*FS)]
    z = p4rx.audio_to_baseband(crop, FS)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM, pin[3], acquisition=True)
    equalized = []
    result = p4short.short_scores(z, first_training_start=start, frequency_offset=cfo,
                                 refine=0, on_equalized=lambda *a: equalized.extend(a))
    return *equalized, result[2]


def decoder(ranked, **kwargs):
    from hfmodem.shrike.p4shortlevels import ShortDecode
    return ShortDecode(ranked, costs={i: [] for i in (1, 2, 3)}, events=[], at=0,
                       status_ok=lambda *a, **kw: True, **kwargs)


def test_component_tables_match_archive_hashes():
    import hashlib
    import numpy as np
    from hfmodem.shrike.p4shortlevels import component_permutation
    for width, digest in ((2, '837e4997d27088d7f30052c9d9ea8fa4fbf060b81e1a4b7ed3cf2432356d8159'),
                           (3, 'c67d4f7da1ad3b758b0e84672bd5e0abf80b8a15ffe4f9fab531e83df0339c4b')):
        order = component_permutation(width)
        assert np.array_equal(np.sort(order), np.arange(352*width))
        assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == digest


@pytest.mark.parametrize('pin,width', [(PINS[0], 3), (PINS[4], 2)])
@pytest.mark.parametrize('wrong', ['level', 'bit-order', 'phase', 'null'])
def test_recorded_labels_reject_wrong_hypotheses(pin, width, wrong):
    import numpy as np
    from hfmodem.shrike.p4shortlevels import demap
    symbols, noise, _ = scores(pin)
    if wrong == 'level':
        width = 5-width
    if wrong == 'phase':
        symbols = symbols * np.exp(2j*np.pi/(1 << width))
    if wrong == 'null':
        symbols = np.zeros_like(symbols)
    fields = demap(symbols, noise, width)
    if wrong == 'bit-order':
        fields = fields.reshape(-1, width)[:, ::-1].ravel()
    assert decoder([(width, fields)]).decode() is None


def test_different_crc_valid_candidates_are_ambiguous():
    from hfmodem.shrike import p4frame
    from hfmodem.shrike.p4shortlevels import codeword
    ranked = []
    for width in (2, 3):
        size = 44*width-4
        frame = p4frame.decode_field(p4frame.encode_field(b'content'.ljust(size, b'\x1e'),
                                    p4frame.Status.from_byte(0)), data_bytes=size)
        ranked.append((width, 4*(1.-2*codeword(frame, width))))
    read = decoder(ranked)
    assert read.decode() is None
    assert len(read.crc_valid) == 2
    assert read.events[-1]['event'] == 'normal-ambiguous'


def test_deadline_pause_resumes_the_same_turbo_state(monkeypatch):
    from hfmodem.shrike import p4decode
    from hfmodem.shrike.p4shortlevels import demap
    symbols, noise, _ = scores(PINS[-1])
    wall = [0.]
    iterations = []
    original = p4decode.turbo_iterations

    def timed(*args, **kwargs):
        for i, result in enumerate(original(*args, **kwargs), 1):
            wall[0] += .006
            iterations.append(i)
            yield result

    monkeypatch.setattr(p4decode, 'turbo_iterations', timed)
    read = decoder([(2, demap(symbols, noise, 2))], now=lambda: wall[0])
    assert read.decode(.010) is None
    assert iterations == [1]
    assert read.decode(.010) is None
    assert iterations == [1]
    frame = read.decode(.100)
    assert frame and frame.crc == 0x7e59
    assert iterations == [1, 2, 3]


def costed_read_clock(monkeypatch, rx, cost_ms, placement_ms=10):
    from hfmodem.shrike import p4decode, p4short
    wall = SimpleNamespace(ticks=0, iterations=[])
    score = p4short.short_scores
    decode = p4decode.turbo_iterations

    def scored(*args, **kwargs):
        result = score(*args, **kwargs)
        wall.ticks += round(placement_ms*FS/1000)
        return result

    def decoded(*args, **kwargs):
        for iteration, result in enumerate(decode(*args, **kwargs), 1):
            wall.ticks += round(cost_ms*FS/1000)
            wall.iterations.append(iteration)
            yield result

    monkeypatch.setattr(p4short, 'short_scores', scored)
    monkeypatch.setattr(p4decode, 'turbo_iterations', decoded)
    rx.hold.now = lambda: wall.ticks/FS
    return wall


@pytest.mark.parametrize('cost_ms', [6, pytest.param(None, marks=[pytest.mark.realtime, pytest.mark.skipif(
    os.environ.get("HFMODEM_REALTIME") != "1", reason="set HFMODEM_REALTIME=1 on a quiet machine")])])
@pytest.mark.parametrize('pin', [PINS[0], PINS[3], PINS[4], PINS[5], PINS[6]])
def test_recorded_read_and_turn_ack_fit_the_receiver_budget(monkeypatch, pin, cost_ms):
    from time import monotonic
    at = round(pin[2]*FS)
    source = audio(pin[0], pin[1])
    rx = receiver(at, pin[3], pin[4])
    turn, _ = machine(role='irs')
    # Charged costs pin iteration counts and the V27 slot; the realtime case
    # checks actual read-through-decode latency. Binding admission is below.
    if cost_ms is not None:
        wall = costed_read_clock(monkeypatch, rx, cost_ms)
    now_clock = monotonic if cost_ms is None else rx.hold.now
    start = now_clock()
    rx.normal_deadline = lambda onset: start+.036+(onset-at)/FS
    records = rx.read_window(source[at-round(.05*FS):at+round(.810*FS)],
                             at-round(.05*FS), whole=True)
    now = at+round((.810+now_clock()-start)*FS)
    if cost_ms is not None:
        expected = 3 if pin == PINS[6] else 2 if pin in (PINS[3], PINS[5]) else 1
        assert wall.iterations == list(range(1, expected+1))
    reg, = [r for r in records if r.observation.frame]
    assert now <= reg.observation.stream_sample+round(.920*FS)-turn.lead
    turn.observe(reg, now=now)
    proposal = turn.decide(reg.observation.stream_sample+round(.920*FS)-turn.lead)
    assert proposal and proposal.word == 'V27'
    assert proposal.stream_sample == reg.observation.stream_sample+round(.920*FS)


@pytest.mark.parametrize('placement_ms,iterations', [(10, [1, 2]), (50, [])],
                         ids=['third-iteration-crosses-deadline', 'placement-past-deadline'])
def test_recorded_deadline_admission_pauses_and_resumes(monkeypatch, placement_ms, iterations):
    pin = PINS[-1]
    at = round(pin[2]*FS)
    source = audio(pin[0], pin[1])
    rx = receiver(at, pin[3], pin[4])
    wall = costed_read_clock(monkeypatch, rx, 10, placement_ms)
    rx.normal_deadline = lambda onset: .036+(onset-at)/FS
    origin = at-round(.05*FS)
    crop = source[origin:at+round(.810*FS)]

    # This frame needs three iterations: 10 ms placement + 3 x 10 ms would
    # cross the deadline. Admission must pause before iteration 3; placement
    # already past the deadline must admit no iteration at all.
    records = rx.read_window(crop, origin, whole=True)
    assert not any(r.observation.frame for r in records)
    assert wall.iterations == iterations
    elapsed = (placement_ms+10*len(iterations))/1000
    assert wall.ticks/FS == pytest.approx(elapsed)
    event, = [e for e in rx.hold.events if e.get('reason') == 'budget']
    assert event['iterations'] == len(iterations)+1
    assert event['allowance_ms'] < event['budget_ms']
    assert event['allowance_ms'] < 10
    if placement_ms == 50:
        assert event['allowance_ms'] < 0

    records = rx.read_window(crop, origin, whole=True)
    assert not any(r.observation.frame for r in records)
    assert wall.iterations == iterations
    assert wall.ticks/FS == pytest.approx(elapsed)

    rx.normal_deadline = lambda onset: .100+(onset-at)/FS
    records = rx.read_window(crop, origin, whole=True)
    reg, = [r for r in records if r.observation.frame]
    assert reg.observation.frame.crc == pin[5]
    assert wall.iterations == [1, 2, 3]
    assert wall.ticks/FS == pytest.approx((placement_ms+30)/1000)


def test_peer_raster_can_enrich_a_budget_deferred_normal():
    from hfmodem.shrike.p4receiver import Emission, Role
    pin = PINS[-1]
    at = round(pin[2]*FS)
    rx = receiver(at, pin[3], 0)
    rx.note_emission(Emission(Kind.WORD, Role.ISS, at-round(.330*FS), .244))
    rx.peer_normal_pending = True
    grant_due = rx.clock.next_due
    rx.normal_deadline = lambda onset: 0.
    _, records = replay(pin, rx)
    assert not any(r.observation.kind == Kind.NORMAL for r in records)
    assert rx.peer_onset == at-60000
    rx.normal_deadline = None
    _, records = replay(pin, rx)
    reg, = [r for r in records if r.observation.frame]
    assert reg.outcome == Registered.NEW and reg.observation.frame.crc == 0x7e59
    assert rx.clock.next_due == grant_due and rx.clock.expects == Expect.WORD


def test_repeated_levelstep_ack_leaves_request_state():
    from dataclasses import replace
    from hfmodem.shrike.p4turn import State
    _, records = replay(PINS[0])
    reg, = [r for r in records if r.observation.frame]
    at = reg.observation.stream_sample
    turn, _ = machine(role='irs')
    turn.observe(reg, now=at+round(.810*FS))
    assert turn.decide(at+round(.920*FS)-turn.lead).word == 'V27'
    turn.keyed()
    turn.state = State.IRS_BREAKIN
    repeated = replace(reg, observation=replace(reg.observation, stream_sample=at+60000))
    turn.observe(repeated, now=at+60000+round(.810*FS))
    ack = turn.decide(at+60000+round(.920*FS)-turn.lead)
    assert ack.word == 'V27' and ack.state == State.IRS_ANSWER

@pytest.mark.parametrize('finish', ['read', 'refine'])
def test_ranked_sl8_preserves_bpsk_memory_until_its_crc(finish):
    _, prior = replay(('2200', 'stream', 48.497, -3.9, 3, None, None))
    pin = PINS[0]
    at = round(pin[2]*FS)
    rx = receiver(at, pin[3], pin[4])
    from hfmodem.shrike import p4frame, p4shorttx
    frame = next(r.observation.frame for r in prior if r.observation.frame)
    rx.hold.acked_counter = 2
    rx.hold.normal(at-60000, 0, pin[3], .8, 4*(1.-2*p4shorttx.coded_bits(p4frame.encode_field(frame.data, frame.status))), None)
    held = rx.hold.frame
    members = list(rx.hold.members)
    epoch = rx.hold.epoch
    assert held is not None and members
    rx.normal_deadline = lambda onset: 0.
    _, records = replay(pin, rx)
    assert not any(r.observation.frame for r in records)
    assert rx.hold.frame is held and rx.hold.members == members and rx.hold.epoch == epoch
    rx.normal_deadline = None
    records = rx.refine() if finish == 'refine' else replay(pin, rx)[1]
    assert any(r.observation.frame and r.observation.frame.crc == pin[5] for r in records)
    assert not rx.hold.members and rx.hold.frame is None and rx._normal_level == 3


def test_2045_sl7_marginal_tracking_retry_is_complete_and_cached(monkeypatch):
    from hfmodem.shrike import p4short
    pin = PINS[5]
    at = round(pin[2]*FS)
    origin = at-round(.05*FS)
    rx = receiver(at, pin[3], pin[4])
    place = p4short.place_short
    calls = []

    def counted(z, nominal, cfo, **kwargs):
        if kwargs.get('acquisition'):
            assert all(p4short.complete(z, nominal+o) for o in kwargs['offsets'])
        result = place(z, nominal, cfo, **kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(p4short, 'place_short', counted)
    source = audio(pin[0], pin[1])
    records = rx.read_partial(source[origin:at+round(.810*FS)], origin,
                              deadline=at+round(.810*FS))
    assert .30 <= calls[0][1][0] < .35
    assert len(calls) == 2 and calls[1][1][0] >= .35
    assert not rx.normal_waiting
    records += rx.read_window(source[origin:at+round(.846*FS)], origin, whole=True)
    assert len(calls) == 2
    assert any(r.observation.frame and r.observation.frame.crc == pin[5] for r in records)


@pytest.mark.parametrize('breakin', ['body', 'y27'])
def test_0923_breakin_driver_reads_peer_raster_before_word_window(monkeypatch, breakin):
    from hfmodem.tests.shrike.test_p4_bare_y27_0929 import Replay
    reads = []
    partial = P4Receiver.read_partial

    def observed(rx, *args, **kwargs):
        if rx.peer_normal_pending and rx.peer_onset is not None:
            reads.append((kwargs['deadline'], rx.peer_onset+round(1.25*FS),
                          rx.clock.next_due+round((rx.clock.slack_s+.180)*FS)))
        return partial(rx, *args, **kwargs)

    monkeypatch.setattr(P4Receiver, 'read_partial', observed)
    run = Replay('0923', monkeypatch, breakin=breakin)
    assert not [(at, peer, word) for at, peer, word in reads if at < word]
    assert run.probe.reason == 'capture ended'


@pytest.mark.parametrize('pin', [PINS[0], PINS[3], PINS[4], PINS[5], PINS[6]])
@pytest.mark.parametrize('regression', ['lost-frame', 'late-ack', 'placement-overrun'])
def test_receiver_budget_gate_rejects_in_memory_regressions(monkeypatch, pin, regression):
    if regression == 'lost-frame':
        original = P4Receiver.read_window

        def unread(*args, **kwargs):
            return [r for r in original(*args, **kwargs) if not r.observation.frame]

        monkeypatch.setattr(P4Receiver, 'read_window', unread)
    elif regression == 'placement-overrun':
        from hfmodem.shrike import p4short
        original = p4short.short_scores

        def repeated(*args, **kwargs):
            # Inject 50 ms of placement work on the converter clock.
            result = original(*args, **kwargs)
            clock = rx_clock[0]
            clock.ticks += round(.050*FS)
            return result

        rx_clock = []
        clock_factory = costed_read_clock

        def clocked(*args, **kwargs):
            clock = clock_factory(*args, **kwargs)
            rx_clock.append(clock)
            return clock

        monkeypatch.setattr(p4short, 'short_scores', repeated)
        monkeypatch.setitem(globals(), 'costed_read_clock', clocked)
    else:
        from dataclasses import replace
        from hfmodem.shrike.p4turn import P4Turn
        original = P4Turn.decide

        def late(*args, **kwargs):
            proposal = original(*args, **kwargs)
            return replace(proposal, stream_sample=proposal.stream_sample+FS) if proposal else None

        monkeypatch.setattr(P4Turn, 'decide', late)
    with pytest.raises((AssertionError, ValueError)):
        test_recorded_read_and_turn_ack_fit_the_receiver_budget(monkeypatch, pin, 6)


def test_sample_admission_gate_rejects_ignored_deadline(monkeypatch):
    from hfmodem.shrike.p4shortlevels import ShortDecode
    original = ShortDecode.decode
    monkeypatch.setattr(ShortDecode, 'decode', lambda self, deadline=None: original(self, None))
    with pytest.raises(AssertionError):
        test_recorded_deadline_admission_pauses_and_resumes(monkeypatch, 10, [1, 2])
