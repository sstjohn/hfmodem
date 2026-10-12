# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hashlib import sha256
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from time import monotonic, perf_counter, sleep
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, p4frame, p4oracle, p4rx, p4sf8rx, p4turn, rxfront, session
from hfmodem.shrike.p4host import P4Host, P4LateFrame
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, Kind, P4Receiver, Registered, Role

ROOT = Path(__file__).resolve().parents[5]
if (not (ROOT/'captures/onair-1002-2154/stream.flac').is_file()
        and (ROOT.parents[2]/'captures/onair-1002-2154/stream.flac').is_file()):
    ROOT = ROOT.parents[2]
FS = 48000
DATA_SHA = 'e8acf31f32a128b6d89a3bb2780cf3fa513014358934a4ba3f8966a922704a92'


def recording(name):
    path = ROOT/f'captures/{name}/stream.flac'
    if not path.is_file():
        pytest.skip(f'{name} recording absent')
    return session.load_wav(str(path), FS)


def receiver(at, cfo, cycle=3.75):
    rx = P4Receiver(FS, admitted=True, cfo_hz=cfo, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .33, cycle, cfo)
    return rx


def test_1002_six_normal_fields_and_robust_short_controls():
    audio = recording('onair-1002-2154')
    for n in range(6):
        at = round((38.514+1.25*n)*FS)
        rx = receiver(at, .69, 1.25)
        rx.last_emission = Emission(Kind.WORD, Role.IRS, at-round(.33*FS),
                                    .24, packet_reply=True)
        obs = rx._packet(audio[at-12000:at+round(.9*FS)], at-12000)
        assert obs is not None and obs.kind == Kind.NORMAL
        assert obs.frame is not None and obs.frame.status.byte == 1
    for n in range(20):
        at = round((46.013+1.25*n)*FS)
        origin = at-10000
        field = p4oracle.robust_at(audio[origin:at+round(.83*FS)], FS,
                                   (at-origin)/FS, .69)
        if n in (10, 11):
            assert field['p4_value'] is None
        else:
            assert field['p4_value'] == (8 if n == 0 else 59)
            assert field['p4_hdr'] == ('A' if n < 2 else 'B')


def test_1002_opening_controls_and_entry_words():
    audio = recording('onair-1002-2154')
    controls = [event.text.split('/')[0] for event in rxfront.decode_events(
        audio[round(14.7*FS):round(17.55*FS)]) if event.kind == 'cs']
    assert controls == ['CS4', 'CS4', 'CS1']
    for seconds, word, cfo in ((19.764, 'Z27', 3.43),
                               (27.264, 'X27', 3.43),
                               (37.264, 'Y27', .69)):
        at = round(seconds*FS)
        origin = at-10000
        obs = p4oracle.word_at(audio[origin:at+round(.29*FS)], FS, origin,
                               (at-origin)/FS, cfo_hz=cfo,
                               expected_words={word}, whole=True)
        assert obs is not None and obs.word == word and obs.score > .98


def test_1002_live_dispatch_keeps_twelve_occupied_and_spends_four_empty(monkeypatch):
    audio = recording('onair-1002-2154')
    rx = receiver(3528586, .69)
    seen = []
    calls = []
    recover = p4rx.recover_robust

    def counted(*a, **kw):
        calls.append(kw['data_symbols'])
        return recover(*a, **kw)

    monkeypatch.setattr(p4rx, 'recover_robust', counted)
    for n in range(16):
        at = 3528586+180000*n
        rx.clock = Clock(Expect.PACKET, at, .33, 3.75, .69)
        origin = at-24000
        early = rx.read_partial(audio[origin:at+round(.845*FS)], origin,
                                deadline=at+round(.845*FS))
        if n < 12:
            assert any(r.observation.kind == Kind.LONG for r in early)
        before = len(calls)
        prepared = rx.prepare_sf8(audio[origin:at+round(3.230*FS)], origin)
        rx.prepare_sf8(audio[origin:at+round(3.230*FS)], origin)
        assert len(calls)-before == 1
        assert rx.sf8_prepared() == (prepared is not None)
        records = rx.read_window(audio[origin:at+round(3.300*FS)], origin,
                                 whole=True)
        placed = [r.observation for r in records
                  if r.outcome != Registered.EXPIRED and r.observation.kind == Kind.LONG
                  and r.observation.placed]
        if n < 12:
            assert prepared is not None and len(placed) == 1
            assert placed[0].stream_sample == at
            assert placed[0].frame is None if n == 1 else placed[0].frame is not None
            if placed[0].frame is not None:
                assert sha256(placed[0].frame.data).hexdigest() == DATA_SHA
                seen.append(placed[0].frame.data)
            assert calls[before:] == [680, 24]
            assert rx.finish_cycle(at) == 0
        else:
            assert prepared is None and not placed
            assert rx.finish_cycle(at) == n-11
    assert len(seen) == 11 and len(set(seen)) == 1


def test_late_crc_cannot_be_accepted_or_answered():
    audio = recording('onair-1002-2154')
    at = 3528586
    rx = receiver(at, .69)
    rx.sf8_deadline = lambda _: monotonic()-1
    origin = at-24000
    assert rx.prepare_sf8(audio[origin:at+round(3.230*FS)], origin) is not None
    obs = rx._packet(audio[origin:at+round(3.300*FS)], origin)
    assert obs is not None and obs.placed and obs.provisional and obs.frame is None
    sf8 = next(row for row in reversed(rx.live_events) if row['event'] == 'sf8-long')
    assert sf8['crc_valid'] and not sf8['timely']


def test_c1_is_deferred_when_the_live_decision_budget_is_short():
    audio = recording('onair-1002-2154')
    at = 3888586
    origin = at-24000
    rx = receiver(at, .69)
    rx.sf8_deadline = lambda _: monotonic()+.010
    rx.prepare_sf8(audio[origin:at+round(3.230*FS)], origin)
    obs = rx._packet(audio[origin:at+round(3.280*FS)], origin)
    assert obs is not None and obs.placed and obs.provisional and obs.frame is None
    assert not rx.live_events[-1]['crc_valid']


def test_host_entry_rechecks_after_a_stalled_delivery():
    host = PtcHost()
    adapter = P4Host(host)
    original = host.deliver

    def stalled(blob):
        sleep(.090)
        return original(blob)

    host.deliver = stalled
    deadline = monotonic()+.020
    with pytest.raises(P4LateFrame):
        adapter.accept(p4frame.Frame(b'hello', p4frame.Status(3), 0),
                       admit=lambda: monotonic() < deadline)
    assert host.rcvd_total == 0
    assert adapter.last is None


def test_header_without_complete_sf8_body_does_not_hold_liveness():
    audio = recording('onair-1002-2154')
    at = 3528586
    rx = receiver(at, .69)
    origin = at-24000
    early = rx.read_partial(audio[origin:at+round(.845*FS)], origin,
                            deadline=at+round(.845*FS))
    assert any(r.observation.kind == Kind.LONG and not r.observation.placed for r in early)
    assert rx.prepare_sf8(audio[origin:at+round(.9*FS)], origin) is None
    assert rx.finish_cycle(at) == 1


@pytest.mark.parametrize('cut_s', [.300, .900, 1.500, 3.100, 3.230,
                                    3.240, 3.250, 3.260, 3.270,
                                    3.279, 3.27925])
@pytest.mark.parametrize('prepared', [False, True])
def test_cut_body_followed_by_noise_does_not_place_or_reset_silence(cut_s, prepared):
    audio = recording('onair-1002-2154')
    at = 3528586
    origin = at-24000
    crop = audio[origin:at+round(3.300*FS)].copy()
    first = at-origin+round(cut_s*FS)
    crop[first:] = np.random.default_rng(108).normal(0, .001, len(crop)-first)
    rx = receiver(at, .69)
    rx.sf8_deadline = lambda _: monotonic()-1
    if prepared:
        rx.prepare_sf8(crop[:at-origin+round(3.230*FS)], origin)
    records = rx.read_window(crop, origin, whole=True)
    assert not [reg for reg in records if reg.observation.placed and reg.observation.kind == Kind.LONG]
    assert rx.finish_cycle(at) == 1


def test_every_five_ms_body_cut_is_absent():
    audio = recording('onair-1002-2154')
    at = 3528586
    origin = at-24000
    original = audio[origin:at+round(3.300*FS)]
    for cut_ms in range(2000, 3280, 5):
        crop = original.copy()
        first = at-origin+round(cut_ms*FS/1000)
        crop[first:] = np.random.default_rng(108).normal(0, .001, len(crop)-first)
        for prepared in (False, True):
            rx = receiver(at, .69)
            if prepared:
                rx.prepare_sf8(crop[:at-origin+round(3.230*FS)], origin)
            records = rx.read_window(crop, origin, whole=True)
            assert not any(reg.observation.sf8 and reg.observation.placed
                           for reg in records), (cut_ms, prepared)
            assert rx.finish_cycle(at) == 1, (cut_ms, prepared)


@pytest.mark.parametrize('stall_stage, cut_s, deliveries', [
    ('none', None, 1), ('prepare', None, None), ('decode', None, 0),
    ('after_read', None, 0), ('on_rx_p4', None, 0),
    ('peer_heard', None, 0), ('after_arq', None, 0),
    ('after_arq_frozen', None, 0),
    ('before_turn', None, 0),
    ('inside_accept', None, 0), ('inside_feed', None, 0),
    ('after_accept', None, 1), ('after_turn', None, 1),
    ('before_decide', None, 1), ('after_decide', None, 1),
    ('before_key', None, 1),
    ('none', 3.240, 0), ('none', 3.250, 0), ('none', 3.260, 0),
    ('none', 3.275, 0), ('none', 3.279, 0)])
def test_live_sf8_deadline_uses_recorded_driver_reserve(monkeypatch, stall_stage, cut_s, deliveries):
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx

    class Clocked(Stream):
        def __init__(self, samples):
            super().__init__(samples)
            self.wall = 0
            self.anchor = perf_counter()
            self.transmit = True
            self.key_notice = 1476
            self._blk = 128
            self.tx_latency_n = 336
            self.frozen = False

        def sample_now(self):
            if self.frozen:
                return max(self.pos, self.wall)
            return max(self.pos, self.wall+round((perf_counter()-self.anchor)*FS))

        def take_until(self, end):
            now = self.sample_now()
            part = super().take_until(end)
            self.wall = max(now, self.pos)
            self.anchor = perf_counter()
            return part

        def clamp_late(self, due):
            return max(0, self.sample_now()+self.key_notice-due)

    audio = recording('onair-1002-2154')
    at = 3528586
    samples = audio[:at+round(3.700*FS)].copy()
    if cut_s is not None:
        first = at+round(cut_s*FS)
        samples[first:] = np.random.default_rng(108).normal(0, .001, len(samples)-first)
    live = Clocked(samples)
    live.pos = at-round(.500*FS)
    live.wall = live.pos
    live.anchor = perf_counter()
    tx = Tx(live)
    tx.settle = .040
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 4.2
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_rx_timing_trace = True
    tx.p4_probe.handoff_audio = (live.pos, samples[live.pos:live.pos].copy())
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, .69)
    rx.cfo_hz = .69
    delivered = []
    tx.p4_link_host.deliver = lambda field: delivered.append((field, live.sample_now()))
    original_turn = p4turn.P4Turn
    machines = []
    stalled_at = []

    def stall_once(stage, used):
        if stall_stage == stage and not used[0]:
            used[0] = True
            sleep(.090)
            stalled_at.append(live.sample_now())

    def make_turn(*a, **kw):
        machine = original_turn(*a, **{**kw, 'role': 'irs'})
        accept = machine.adapter.accept
        feed = machine.adapter._feed
        can_key = machine.can_key
        observe = machine.observe
        decide = machine.decide
        stalled = [False]

        def delayed_accept(*args, **kwargs):
            stall_once('inside_accept', stalled)
            result = accept(*args, **kwargs)
            stall_once('after_accept', stalled)
            return result

        def delayed_feed(*args, **kwargs):
            stall_once('inside_feed', stalled)
            return feed(*args, **kwargs)

        def delayed_key(proposal):
            if proposal.ack_identity is not None:
                stall_once('before_key', stalled)
            return can_key(proposal)

        def delayed_observe(reg, *, now=None):
            if reg.observation.sf8 and reg.observation.frame is not None:
                stall_once('before_turn', stalled)
            result = observe(reg, now=now)
            if reg.observation.sf8 and reg.observation.frame is not None:
                stall_once('after_turn', stalled)
            return result

        def delayed_decide(now):
            if machine.pending:
                stall_once('before_decide', stalled)
            result = decide(now)
            if result is not None and result.ack_identity is not None:
                stall_once('after_decide', stalled)
            return result

        machine.adapter.accept = delayed_accept
        machine.adapter._feed = delayed_feed
        machine.can_key = delayed_key
        machine.observe = delayed_observe
        machine.decide = delayed_decide
        machines.append(machine)
        return machine

    monkeypatch.setattr(p4turn, 'P4Turn', make_turn)
    original_tx = tx._tx

    def emit(data, what):
        original_tx(data, what)
        live.pos = tx.tx_end
        live.wall = max(live.wall, live.pos)
        live.anchor = perf_counter()

    tx._tx = emit
    original_read = rx.read_window
    original_arq = tx.p4_link_host.arq.on_rx_p4
    original_prepare = rx.prepare_sf8
    original_decode = p4sf8rx.decode_symbols
    original_peer_heard = tx.p4_link_host.arq.note_peer_heard
    finish_ms = []
    prep_used = [False]
    decode_used = [False]
    peer_pending = [False]

    def delayed_prepare(*args, **kwargs):
        stall_once('prepare', prep_used)
        return original_prepare(*args, **kwargs)

    def delayed_decode(*args, **kwargs):
        stall_once('decode', decode_used)
        return original_decode(*args, **kwargs)

    def delayed_peer_heard():
        if peer_pending[0]:
            peer_pending[0] = False
            sleep(.090)
            stalled_at.append(live.sample_now())
        return original_peer_heard()

    tx.p4_link_host.arq.note_peer_heard = delayed_peer_heard

    def delayed_read(*a, **kw):
        started = perf_counter()
        before = len(rx.live_events)
        rows = original_read(*a, **kw)
        if any(event['event'] == 'sf8-long' for event in rx.live_events[before:]):
            finish_ms.append((perf_counter()-started)*1000)
            if stall_stage == 'after_read':
                sleep(.090)
                stalled_at.append(live.sample_now())
        return rows

    def delayed_arq(reg):
        if (stall_stage == 'on_rx_p4' and reg.observation.sf8
                and reg.observation.frame is not None and not stalled_at):
            sleep(.090)
            stalled_at.append(live.sample_now())
        if stall_stage == 'peer_heard' and reg.observation.sf8 and reg.observation.frame:
            peer_pending[0] = True
        result = original_arq(reg)
        if (stall_stage in ('after_arq', 'after_arq_frozen') and reg.observation.sf8
                and reg.observation.frame is not None and not stalled_at):
            live.frozen = stall_stage == 'after_arq_frozen'
            sleep(.090)
            stalled_at.append(live.sample_now())
        return result

    rx.read_window = delayed_read
    rx.prepare_sf8 = delayed_prepare
    tx.p4_link_host.arq.on_rx_p4 = delayed_arq
    monkeypatch.setattr(p4sf8rx, 'decode_symbols', delayed_decode)
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    lead = onair._prekey_lead(live, round(tx.settle*FS))
    assert lead == round(.049*FS)
    if stall_stage != 'none':
        if cut_s is None:
            assert len(stalled_at) == 1
    if stall_stage == 'on_rx_p4':
        assert len(stalled_at) == 1
        assert stalled_at[0] > at+round(3.400*FS)-lead
    if deliveries is not None:
        assert len(delivered) == deliveries
    assert all(sample <= at+round(3.400*FS)-lead for _, sample in delivered)
    if cut_s is None:
        assert len(finish_ms) == 1
    else:
        assert not finish_ms
    assert any(row['stage'] == 'sf8_prepare'
               for row in tx.p4_probe.timing_trace)
    assert any(row['event'] == 'Decision' and row['decision_at'] == at+round(3.400*FS)-lead
               for row in tx.p4_probe.link_events)
    acks = [row for row in tx.p4_probe.emissions if row.get('word') == 'X27']
    if stall_stage in ('after_accept', 'after_turn', 'before_decide', 'after_decide', 'before_key'):
        assert not acks
    elif deliveries is not None:
        assert len(acks) == deliveries


def test_live_twelve_sf8_copies_keep_their_own_ack_deadlines(monkeypatch):
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    from hfmodem.tests.shrike.p4_nak_qam_support import neutral_keys

    class Clocked(Stream):
        def __init__(self, samples):
            super().__init__(samples)
            self.wall = 0
            self.anchor = perf_counter()
            self.transmit = True
            self.key_notice = 1476
            self._blk = 128
            self.tx_latency_n = 336

        def sample_now(self):
            return max(self.pos, self.wall+round((perf_counter()-self.anchor)*FS))

        def take_until(self, end):
            now = self.sample_now()
            part = super().take_until(end)
            self.wall = max(now, self.pos)
            self.anchor = perf_counter()
            return part

        def clamp_late(self, due):
            return max(0, self.sample_now()+self.key_notice-due)

    audio = recording('onair-1002-2154')
    at = 3528586
    live = Clocked(audio)
    live.pos = at-round(.500*FS)
    live.wall = live.pos
    live.anchor = perf_counter()
    tx = Tx(live)
    tx.settle = .040
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 61.5
    tx.p4_link_options.p4_data = 'body'
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, .69)
    rx.cfo_hz = .69
    received = []
    tx.p4_link_host.deliver = received.append
    original_turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: original_turn(
        *a, **{**kw, 'role': 'irs'}))
    original_tx = tx._tx

    def emit(data, what):
        original_tx(data, what)
        live.pos = tx.tx_end
        live.wall = max(live.wall, live.pos)
        live.anchor = perf_counter()

    tx._tx = emit
    finish = rx.finish_cycle
    silence = []

    def completed(onset):
        count = finish(onset)
        silence.append(count)
        return count

    rx.finish_cycle = completed
    with neutral_keys() as neutral, redirect_stdout(StringIO()):
        onair._run_p4_link(tx, live, onair._MasterGrid(
            0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert onair._prekey_lead(live, round(tx.settle*FS)) == round(.049*FS)
    assert len(received) == 1
    assert len([row for row in rx.live_events if row['event'] == 'sf8-long'
                and row['crc_valid']]) == 11
    keyed=[e for e in tx.p4_probe.link_events if e.get('event')=='Keyed']
    assert all(e.get('word')=='X27' and e.get('purpose') is None for e in keyed)
    ordinary=[e for e in keyed if e['current_slot_accepted']]
    assert len(ordinary)==11 and len(keyed)==16
    # Eleven original physical answers retain their exact own LONG deadlines.
    assert [e['at'] for e in ordinary]==[at+round(3.4*FS)+n*round(3.75*FS)
                                       for n in (0,2,3,4,5,6,7,8,9,10,11)]
    valid=[e for e in rx.live_events if e.get('event')=='sf8-long' and e['crc_valid']]
    assert all(e['timely'] for e in valid)
    assert [e['at']+round(3.4*FS) for e in valid]==[e['at'] for e in ordinary]
    assert neutral==[dict(at=n,word='X27',purpose=None,kind=Kind.WORD,state='IRS_ANSWER')
                     for n in (3871786,5851786,6031786,6211786,6391786)]
    assert silence == [0]*12 + [1, 2, 3, 4]


def test_1003_ve3kpg_robust_long_replay():
    audio = recording('onair-1003-0017')
    outcomes = []
    for n in range(8):
        at = round((46.013+3.75*n)*FS)
        rx = receiver(at, 3.22)
        origin = at-24000
        rx.prepare_sf8(audio[origin:at+round(3.230*FS)], origin)
        obs = rx._packet(audio[origin:at+round(3.300*FS)], origin)
        outcomes.append(obs is not None and obs.placed and obs.frame is not None)
    assert outcomes == [True, True, True, True, True, False, True, True]


def test_recorded_repeat_delivers_once_at_the_existing_long_answer_slot():
    from hfmodem.tests.shrike.test_p4turn import machine

    audio = recording('onair-1002-2154')
    turn, _ = machine(role='irs', data='body')
    received = []
    turn.adapter.host = SimpleNamespace(deliver=received.append,
                                        supervisor=lambda *args: None)
    for n in (0, 4):
        at = 3528586+180000*n
        rx = receiver(at, .69)
        origin = at-24000
        rx.prepare_sf8(audio[origin:at+round(3.230*FS)], origin)
        records = rx.read_window(audio[origin:at+round(3.300*FS)], origin,
                                 whole=True)
        for record in records:
            turn.observe(record, now=at+round(3.310*FS))
        due = at+round(3.400*FS)
        proposal = turn.decide(due-turn.lead)
        assert proposal is not None and proposal.stream_sample == due
        turn.keyed()
    assert len(received) == 1
    assert received[0]
