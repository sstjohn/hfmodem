# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from time import monotonic

import numpy as np
import pytest

from hfmodem.tests.evidence import CAPTURES, RECORD
from hfmodem.shrike import onair, p4oracle, p4breakin, p4turn
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role, Kind
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx
from hfmodem.tests.shrike.test_ws8eoc_p4_silent_0927 import Live, grid, hold
from hfmodem.tests.shrike.test_p4turn import machine, body, word

GREETING = b'RMS Trimode 1.4.4.0 KB5LZK at t'


def capture(arm, name='stream.flac'):
    path = CAPTURES / arm / name
    if not path.exists():
        pytest.skip(f'local capture absent: {path}')
    return onair.session.load_wav(str(path), FS)


def kb_receiver(audio, *, first=35.9772, copies=12, cfo=-1.3):
    receiver = P4Receiver(FS, admitted=True, cfo_hz=cfo, breakin=True)
    receiver.enable_hold()
    receiver.note_emission(Emission(Kind.SHORT, Role.ISS, round((first-.920)*FS), .758))
    records = []
    for index in range(copies):
        onset = first + index*1.25
        start, early, end = (round(t*FS) for t in (onset-.10, onset+.215, onset+.846))
        records.extend(receiver.read_partial(audio[start:early], start, deadline=early))
        records.extend(receiver.read_window(audio[start:end], start, whole=True))
        receiver.hold.refine(deadline=monotonic()+.250)
        receiver.new_cycle()
    return receiver, records


def run_kb(monkeypatch, *, silent=False):
    audio = capture('onair-0927-1544')
    if silent:
        audio = np.concatenate((audio, np.zeros(25*FS)))
        monkeypatch.setattr(p4breakin, 'decode_scores', lambda *a: None)
    live = Live(audio)
    live.pos = round(36.1922*FS)
    tx = Tx(live)
    tx.p4_link_options.p4_link_timeout = 90.
    tx.p4_link_options.p4_data = 'body'
    tx.p4_probe.emissions = [dict(audio_start=1622708, audio_end=1659092, value=7, header='A'),
                             dict(audio_start=1682708, audio_end=1719092, value=38, header='A')]
    tx.p4_probe.received = [(0, round(34.65*FS), audio[round(34.65*FS):round(35.05*FS)]),
                           (1, 1719092, audio[1719092:live.pos])]
    tx.sessrx.p4.cfo_hz = -1.3
    receiver = tx.sessrx.p4
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    if not silent:
        key = tx._tx
        def first_key(*args):
            key(*args)
            live.audio = live.audio[:live.pos]
        monkeypatch.setattr(tx, '_tx', first_key)
    onair._run_p4_link(tx, live, grid())
    return tx, live, receiver


def test_kb5lzk_0927_greeting_combines_by_the_fourth_copy():
    receiver, records = kb_receiver(capture('onair-0927-1544'))
    bodies = [r.observation for r in records if r.observation.kind == Kind.BODY]
    assert bodies and bodies[0].frame.data == GREETING
    assert bodies[0].frame.status.byte == 0
    assert any(abs(bodies[0].stream_sample/FS-t) < .005 for t in (38.477, 39.727))
    assert all(b.stream_sample/FS >= 38.4 for b in bodies)
    assert receiver.hold.frame.data == GREETING


def test_kb5lzk_0927_x27_answers_the_combined_greeting(monkeypatch):
    tx, live, receiver = run_kb(monkeypatch)
    keys = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert keys and keys[0]['kind'] == Kind.WORD and keys[0]['word'] == 'X27'
    bodies = [o for o in receiver.observations if o.kind == Kind.BODY]
    assert abs(keys[0]['at']-bodies[0].stream_sample-round(.920*FS)) <= 96
    assert keys[0]['at']/FS >= 39.397
    assert len(tx.p4_probe.emissions) == 3


def test_kb5lzk_0927_link_holds_while_y27_continues(monkeypatch):
    tx, live, receiver = run_kb(monkeypatch, silent=True)
    assert tx.p4_probe.reason == 'peer silent'
    assert live.pos/FS > 50.6
    assert receiver.silent_cycles == 16
    assert sum(e['event'] == 'BREAKIN_UNREAD' for e in tx.p4_probe.link_events) >= 11
    assert len(tx.p4_probe.emissions) == 2


def test_kb5lzk_0927_placed_read_names_the_x27_answer():
    audio = capture('onair-0927-1544')
    start = round(34.65*FS)
    got = p4oracle.word_at(audio[start:round(35.05*FS)], FS, start, 34.724-start/FS,
                           cfo_hz=-1.3, whole=True, expected_words={'X27', 'Y27', 'Z27'})
    assert got is not None and got.word == 'X27' and got.score >= .75
    assert not got.weak and .89 < got.score < .91
    for onset in [35.9772+k*1.25+.55 for k in range(12)]+list(np.arange(.37, .37*87, .37)):
        start = max(0, round((onset-.05)*FS))
        got = p4oracle.word_at(audio[start:round((onset+.32)*FS)], FS, start, onset-start/FS,
                               cfo_hz=-1.3, expected_words={'X27', 'Y27', 'Z27'})
        assert got is None, (onset, got)


def test_kb5lzk_0927_witness_agrees_with_the_combined_frame():
    from hfmodem.shrike.p4hold import BreakinHold
    from hfmodem.shrike import p4breakintx
    witness = capture('onair-0927-1544', 'witness.wav')
    receiver, records = kb_receiver(witness, first=35.9772+3.68, copies=5, cfo=-8.)
    bodies = [r.observation for r in records if r.observation.frame is not None]
    assert bodies and bodies[0].frame.data == GREETING
    stream = capture('onair-0927-1544').copy()
    receiver, _ = kb_receiver(stream, copies=4)
    holder = receiver.hold
    assert isinstance(holder, BreakinHold) and holder.frame is not None
    onset = 35.9772+4*1.25
    at = round(onset*FS)
    wave = p4breakintx.changeover_packet(b'', 0x40)
    body = slice(round(.245*FS), round(.800*FS))
    level = np.sqrt(np.mean(stream[at:at+len(wave)][body]**2)/np.mean(wave[body]**2))
    stream[at:at+len(wave)] = level*wave
    start, end = at-round(.1*FS), at+round(.846*FS)
    head = word('Y27', at).observation
    epoch = holder.epoch
    assert holder.inline(stream[start:end], start, head) is None
    assert holder.epoch == epoch+1 and holder.frame is None
    assert any(e.get('reason') == 'different body' for e in holder.events)


def test_ve3kpg_0927_z27_draws_v7_on_header_b():
    audio = capture('onair-0927-1549')
    receiver = P4Receiver(FS, admitted=True, cfo_hz=2.3, breakin=True)
    receiver.enable_hold()
    turn, _ = machine(data='body', prekey_lead=2131/FS)
    emissions = [dict(audio_start=1202631, value=7), dict(audio_start=1262631, value=38)]
    turn.seed_opening(emissions, [])
    receiver.note_emission(Emission(Kind.SHORT, Role.ISS, 1262631, .758))
    receiver.expected_words = turn.expected_words()
    for index, onset in enumerate((27.2474, 28.4974)):
        start = round((onset-.06)*FS)
        now = round((onset+.25)*FS)
        records = receiver.read_window(audio[start:now], start, whole=True)
        assert any(r.observation.word == 'Z27' and not r.observation.provisional for r in records)
        for record in records:
            turn.observe(record, now=now)
        proposal = turn.decide(now)
        assert proposal and proposal.value == 7 and proposal.header == 'B'
        assert proposal.stream_sample == 1262631+(index+1)*60000
        assert proposal.stream_sample >= now+turn.lead
        sent = turn.keyed()
        receiver.note_emission(Emission(Kind.SHORT, Role.ISS, sent.stream_sample, .758))
    receiver.clock.next_due = round(33.4974*FS)
    start = round(33.44*FS)
    now = round(33.82*FS)
    for record in receiver.read_window(audio[start:now], start, whole=True):
        turn.observe(record, now=now)
    assert turn.index == 0


def test_ve3kpg_0927_late_z27_skips_the_missed_repeat_slot():
    audio = capture('onair-0927-1549')
    receiver = P4Receiver(FS, admitted=True, cfo_hz=2.3, breakin=True)
    receiver.enable_hold()
    turn, _ = machine(data='body', prekey_lead=2131/FS)
    turn.seed_opening([dict(audio_start=1202631, value=7),
                       dict(audio_start=1262631, value=38)], [])
    receiver.note_emission(Emission(Kind.SHORT, Role.ISS, 1262631, .758))
    receiver.expected_words = turn.expected_words()
    onset = 27.2474
    start, end = round((onset-.06)*FS), round((onset+.25)*FS)
    records = receiver.read_window(audio[start:end], start, whole=True)
    assert any(r.observation.word == 'Z27' and not r.observation.provisional for r in records)
    now = end+round(.060*FS)
    for record in records:
        turn.observe(record, now=now)
    proposal = turn.decide(now)
    assert proposal and proposal.value == 7 and proposal.header == 'B'
    assert proposal.stream_sample == 1382631
    assert proposal.stream_sample >= now+turn.lead


def test_ws8eoc_0927_greeting_is_answered_x27_not_granted(monkeypatch):
    tx, live, heard = hold(monkeypatch)
    key = tx._tx
    def first_key(*args):
        key(*args)
        live.audio = live.audio[:live.pos]
    monkeypatch.setattr(tx, '_tx', first_key)
    onair._run_p4_probe(tx, live, grid(), heard=heard)
    keyed = next(e for e in tx.p4_probe.link_events if e['event'] == 'Keyed')
    assert keyed['kind'] == Kind.WORD and keyed['word'] == 'X27'
    assert keyed['state'] == p4turn.State.IRS_ANSWER
    assert abs(keyed['at']-631897) <= 96


def test_p4turn_repeat_of_the_acked_body_requests_queued_turn():
    turn, queue = machine(role='irs')
    turn.observe(word('Y27', FS, extent=.8))
    turn.observe(body(FS))
    first = turn.decide(FS+round(.846*FS))
    assert first.word == 'X27'
    turn.keyed()
    queue.on_host_data(b'ready')
    at = FS+60000
    turn.observe(word('Y27', at, extent=.8))
    turn.observe(body(at))
    repeat = turn.decide(at+round(.846*FS))
    assert repeat.kind == Kind.BODY and repeat.status == 0x40
    assert repeat.state == p4turn.State.IRS_BREAKIN
    turn.keyed()
    receiver, records = kb_receiver(capture('onair-0927-1544'), copies=4)
    head = max(o.stream_sample for o in receiver.observations if o.word == 'Y27')
    receiver.note_emission(Emission(Kind.WORD, Role.IRS, head+round(.920*FS), .244))
    assert receiver.hold.frame is None
    stream = capture('onair-0927-1544').copy()
    source = capture('onair-0927-1547')
    decoded = []
    for index in range(3):
        peer = round((12.2445+index*1.25)*FS)
        onset = head+(index+1)*60000
        size = round(.946*FS)
        origin = onset-round(.1*FS)
        stream[origin:origin+size] = source[peer-round(.1*FS):peer-round(.1*FS)+size]
        receiver.clock.cfo_hz = 0.
        incoming = receiver.read_window(stream[origin:origin+size], origin, whole=True)
        frames = [r.observation.frame for r in incoming if r.observation.frame]
        assert all(f.data != GREETING for f in frames)
        decoded += frames
        receiver.hold.refine(deadline=monotonic()+.250)
        receiver.new_cycle()
    assert decoded and all(frame.data == b'RMS Trimode 1.4.4.0\r\nW9SSJ has ' for frame in decoded)


def test_p4_liveness_learns_the_peer_turnaround():
    audio = capture('onair-0927-1549')
    receiver = P4Receiver(FS, admitted=True, cfo_hz=2.3, breakin=True)
    receiver.enable_hold()
    receiver.note_emission(Emission(Kind.SHORT, Role.ISS, 1202631, .758))
    for k in range(36):
        onset = 25.9974 + k*1.25
        if k == 1:
            receiver.note_emission(Emission(Kind.SHORT, Role.ISS, 1262631, .758))
        start = round((onset-.05)*FS)
        records = receiver.read_window(audio[start:round((onset+.31)*FS)], start, whole=True)
        assert records, k
        assert receiver.finish_cycle(round(onset*FS)) == 0, k
        receiver.new_cycle()
    assert abs(np.median(receiver.turnarounds['short'])-.9426) < .001
    off = round((25.9974+36*1.25+.020)*FS)
    receiver.register(word('Z27', off).observation)
    assert receiver.finish_cycle(off-round(.020*FS)) == 0
    assert abs(np.median(receiver.turnarounds['short'])-.9426) < .001
    receiver.note_emission(Emission(Kind.SHORT, Role.ISS, 72*FS, .758))
    receiver.register(word('X27', round(72.955*FS)).observation)
    assert receiver.finish_cycle(round(72.955*FS)) == 0
    assert receiver.live_events[-1]['event'] == 'off-turnaround'

    audio = capture('onair-0927-1544')
    kb = P4Receiver(FS, admitted=True, cfo_hz=-1.3, breakin=True)
    kb.enable_hold()
    for index, onset in enumerate([34.7256]+[35.9772+k*1.25 for k in range(12)]):
        if index < 2:
            kb.note_emission(Emission(Kind.SHORT, Role.ISS, (1622708,1682708)[index], .758))
        start, early, end = (round(t*FS) for t in (onset-.05, onset+.215, onset+.31))
        kb.read_partial(audio[start:early], start, deadline=early)
        records = kb.read_window(audio[start:end], start)
        assert records and kb.finish_cycle(round(onset*FS)) == 0
        kb.new_cycle()
    assert len(kb.turnarounds['short']) == 1
    assert abs(kb.turnarounds['short'][0]-.9192) < .001


def test_p4_weak_data_ack_settles_only_on_qualifying_answers(monkeypatch):
    receiver = P4Receiver(FS, admitted=True, cfo_hz=2.3, breakin=True)
    receiver.enable_hold()
    audio = capture('onair-0927-1549')
    for at, onset in ((1202631, 1247878), (1262631, 1307875)):
        receiver.note_emission(Emission(Kind.SHORT, Role.ISS, at, .758))
        start = onset-round(.06*FS)
        receiver.read_partial(audio[start:onset+round(.215*FS)], start,
                              deadline=onset+round(.215*FS))
        old_clock = receiver.clock
        if at == 1262631:
            receiver.note_emission(Emission(Kind.SHORT, Role.ISS, 1322775, .758))
            receiver.clock = old_clock
        records = receiver.read_window(audio[start:onset+round(.31*FS)], start, whole=True)
        confirmed = next(r for r in records if r.outcome.value == 'confirmed')
        assert confirmed.responds_to == at
        assert confirmed.answered == (1322775 if at == 1262631 else None)
    from hfmodem.tests.shrike.p4_hold_helpers import check_settlement
    check_settlement()
    from hfmodem.tests.shrike.p4_hold_helpers import check_driver_settlement
    check_driver_settlement(monkeypatch)
    from hfmodem.tests.shrike.p4_hold_helpers import check_weak_edge_cases
    check_weak_edge_cases()


@pytest.mark.slow
def test_breakin_hold_false_accept_corpus(tmp_path):
    from hfmodem.tests.shrike.p4_hold_helpers import check_corpus
    check_corpus(RECORD, evidence=tmp_path)


@pytest.mark.parametrize('answer', [True, False])
def test_probe_reads_v7_before_admitting_v38(monkeypatch, answer):
    audio = capture('onair-0927-1744') if answer else np.zeros(17*FS)
    live = Live(audio)
    live.pos = round(12*FS)
    tx = Tx(live)
    tx.slot = 10
    tx.sessrx.host = tx.p4_link_host
    tx.p4_probe.emissions.clear()
    tx.p4_probe.values = (7, 38)
    tx.p4_probe.headers = ('A', 'A')
    tx.p4_probe.cadence = 'consecutive'
    tx.p4_probe.packet = 'robust'
    tx.p4_link_options.p4_link_timeout = 3.
    aim = tx.aim
    def aim_slot(raster, slot):
        aim(raster, slot)
        return raster.boundary(slot)
    monkeypatch.setattr(tx, 'aim', aim_slot)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    onair._run_p4_probe(tx, live, grid(), in_slot=True)
    probes = [e for e in tx.p4_probe.emissions if 'proposed_at' not in e]
    assert [e['value'] for e in probes] == ([7] if answer else [7, 38])
    assert probes[0]['audio_start'] == grid().boundary(10)
    if answer:
        y = next(o for o in tx.sessrx.p4.observations if o.word == 'Y27' and not o.provisional)
        first = next(e for e in tx.p4_probe.link_events if e['event'] == 'Keyed')
        assert first['kind'] == Kind.BODY and abs(first['at']-y.stream_sample-round(.920*FS)) <= 2
    else:
        assert probes[1]['audio_start'] == grid().boundary(11)
