# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from hfmodem.tests.shrike import p4_stock_peer as stock

import numpy as np
import pytest
from hfmodem.shrike import p4frame, p4breakintx
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Kind, Role


def scores(status=0, data=b'A'*31):
    frame = p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)), data_bytes=31)
    bits = p4breakintx.coded_bits(p4frame.encode_field(frame.data, frame.status))
    return frame, 10*(1-2*bits.astype(float))


def test_search_overrun_restores_the_inline_sum_and_members(monkeypatch):
    from hfmodem.shrike import p4hold

    frame, llr = scores()
    hold = p4hold.BreakinHold(48000)
    first = p4hold.Copy(0, 0, 0, 1, -llr)
    second = p4hold.Copy(60000, 0, 0, 1, -llr)
    hold.members = [first, second]
    hold.work = np.zeros(100), 0, 0, 60000
    hold.last = 60000
    monkeypatch.setattr(p4hold.p4breakin, 'search_scores', lambda *a, **kw: (32, 3, llr))
    monkeypatch.setattr(p4hold.p4breakin, 'decode_scores', lambda *a: frame)
    times = iter((0, 2))
    monkeypatch.setattr(p4hold, 'monotonic', lambda: next(times))
    assert hold.refine(deadline=1) is None
    assert hold.members == [first, second] and second.q == 1 and second.offset == 0
    np.testing.assert_array_equal(second.llr, -llr)
    assert hold.frame is None and hold.offset == 0
    assert hold.events == [dict(event='search overrun', at=60000)]


def test_combined_status_gate_does_not_restrict_single_copies(monkeypatch):
    from hfmodem.shrike import p4hold

    frame, llr = scores(4)
    hold = p4hold.BreakinHold(48000)
    monkeypatch.setattr(p4hold.p4breakin, 'decode_scores', lambda *a: frame)
    hold.members = [p4hold.Copy(0, 0, 0, 1, llr), p4hold.Copy(60000, 0, 0, 1, llr)]
    assert hold._accept() is None
    assert hold.events[-1] == dict(event='status-rejected', status=4)
    hold.members.pop()
    assert hold._accept() == frame


def test_only_a_key_closes_the_hold_epoch():
    from hfmodem.shrike import p4hold

    receiver = P4Receiver(48000, admitted=True, breakin=True)
    receiver.enable_hold()
    _, llr = scores()
    receiver.hold.members = [p4hold.Copy(0, 0, 0, 1, llr)]
    epoch = receiver.hold.epoch
    receiver.note_refusal(Kind.WORD, 44160)
    assert receiver.hold.epoch == epoch and receiver.hold.members
    receiver.note_emission(Emission(Kind.WORD, Role.IRS, 44160, .24))
    assert receiver.hold.epoch == epoch+1 and not receiver.hold.members


def test_late_extent_refinement_does_not_count_unread_twice():
    from hfmodem.tests.shrike.test_p4turn import machine, word
    from hfmodem.shrike.p4observation import Registered

    turn, _ = machine()
    first = word('Y27', 48000)
    turn.observe(first)
    turn.decide(48000+round(.846*48000))
    assert turn.unread == 1
    turn.observe(word('Y27', 48002, outcome=Registered.EXTENDED,
                      replaced=first.observation, extent=.8))
    turn.decide(48002+round(.846*48000))
    assert turn.unread == 1



def test_held_body_safety_exit_and_similar_body_measurement():
    from hfmodem.tests.shrike.test_p4_breakin_hold_0927 import capture
    from hfmodem.tests.shrike.p4_hold_helpers import check_hold_safety

    counts = check_hold_safety(capture('onair-0927-1544'))
    assert counts['safety_exits'] == 24
    assert counts['similar_copies'] == 218
    print(counts)


def test_radio_link_ceiling_replaces_the_entry_deadline(tmp_path, monkeypatch):
    from time import monotonic
    from hfmodem.shrike import onair, p4probe

    fs = 48000
    monkeypatch.setattr(onair, '_load', lambda _: np.zeros(150*fs, np.float32))
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.live = onair._ReplayInput('synthetic')
    tx.live.pos = 130*fs
    tx.p4_probe = p4probe.EntryProbe(timeout=120., requested=True, emitting=True,
                                    grant_wall=monotonic())
    grid = onair._MasterGrid(0, 60000, 0, packet_n=38400, cs_n=5760, d_max_n=6240)
    tx.aim(grid, 105)
    tx.p4_link_deadline = (140*fs, monotonic()+10)
    tx._tx(np.zeros(38400), 'P4 LINK body')
    assert not tx.refused and tx.slots_used == [105]
    tx.aim(grid, 106)
    tx.p4_link_deadline = (133*fs, monotonic()+10)
    tx._tx(np.zeros(38400), 'P4 LINK body')
    assert tx.refused and tx.slots_used == [105]
    assert tx.p4_probe.reason == 'session ceiling'


def test_inline_combining_has_time_after_the_body_horizon(monkeypatch):
    from hfmodem.tests.shrike.test_p4_link_0927 import test_driver_admission_accounts_for_decode_cost_and_a_stall

    test_driver_admission_accounts_for_decode_cost_and_a_stall(monkeypatch, 46, 1)


def test_unexpected_weak_winner_cannot_be_renamed_as_an_ack(monkeypatch):
    from hfmodem.shrike import p4oracle

    audio = np.concatenate((np.zeros(4800), stock.word('Z27'), np.zeros(4800)))
    def ranked(symbols, **kwargs):
        fits = [('V27', 0, 0.) for _ in symbols]
        fits[0] = ('Z27', 0, 0.)
        scores = np.full(len(symbols), .76)
        scores[0] = .84
        return fits, scores
    monkeypatch.setattr(p4oracle.p4words, 'placed_scores', ranked)
    diagnostics = []
    result = p4oracle.word_at(audio, 48000, 0, .1, cfo_hz=0., whole=True,
                             expected_words={'Y27', 'V27', 'X27'}, diagnostics=diagnostics)
    assert result is None
    assert diagnostics[0]['word'] == 'Z27'


@pytest.mark.parametrize('arm,first,copies', [('onair-0927-1549', 25.9974, 36),
                                             ('onair-0927-1553', 15.999, 35)])
def test_recorded_z27_train_never_names_a_data_ack(arm, first, copies):
    from hfmodem.tests.shrike.test_p4_breakin_hold_0927 import capture
    from hfmodem.shrike import p4oracle

    audio = capture(arm)
    names = []
    for k in range(copies):
        at = round((first + 1.25*k)*48000)
        start = at-2400
        for length, provisional in ((.215, True), (.310, False)):
            obs = p4oracle.word_at(audio[start:at+round(length*48000)], 48000, start,
                                  .05, cfo_hz=2.3, provisional=provisional, whole=not provisional,
                                  expected_words={'Y27', 'V27', 'X27'}, carrier_offsets={})
            if obs is not None:
                names.append((k, obs.word))
    assert names
    assert all(name == ('X27' if k == 6 else 'Z27') for k, name in names), names
    assert (6, 'X27') in names


@pytest.mark.parametrize('answer', [False, True])
@pytest.mark.parametrize('cost_ms', [50, pytest.param(None, marks=pytest.mark.realtime)])
def test_probe_read_cost_and_handoff_keep_the_next_short_on_time(monkeypatch, answer, cost_ms):
    from time import monotonic
    import os

    durations = []
    from hfmodem.shrike import onair, p4robusttx
    from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx
    from hfmodem.tests.shrike.test_ws8eoc_p4_silent_0927 import Live, grid

    audio = np.zeros(18*FS)
    onset = grid().boundary(10)+round(.923*FS)
    if answer:
        wave = stock.word('X27')
        audio[onset:onset+len(wave)] = wave
    live = Live(audio)
    live.pos = 12*FS
    live.wall = live.pos
    live.sample_now = lambda: max(live.pos, live.wall)
    tx = Tx(live)
    tx.slot = 10
    tx.sessrx.host = tx.p4_link_host
    tx.p4_probe.emissions.clear()
    tx.p4_probe.values, tx.p4_probe.headers = (7, 38), ('A', 'A')
    tx.p4_probe.cadence, tx.p4_probe.packet = 'consecutive', 'robust'
    tx.p4_link_options.p4_link_timeout = 2.
    aim = tx.aim
    def aim_slot(raster, slot):
        aim(raster, slot)
        return raster.boundary(slot)
    monkeypatch.setattr(tx, 'aim', aim_slot)
    key = tx._tx
    def guarded_key(wave, what):
        tx.refused = live.sample_now()+live.key_notice > tx.raster.boundary(tx.slot)
        if not tx.refused:
            key(wave, what)
    monkeypatch.setattr(tx, '_tx', guarded_key)
    for name in ('robust_short', 'robust_word'):
        render = getattr(p4robusttx, name)
        def costly_render(*args, _render=render, **kwargs):
            if tx.slots_used:
                live.wall = live.sample_now()+round(.020*FS)
            return _render(*args, **kwargs)
        monkeypatch.setattr(p4robusttx, name, costly_render)
    for name in ('read_partial', 'read_window'):
        read = getattr(tx.sessrx.p4, name)
        def costly_read(*args, _read=read, **kwargs):
            begin, now = monotonic(), live.sample_now()
            result = _read(*args, **kwargs)
            elapsed = 1000*(monotonic()-begin)
            durations.append(elapsed)
            live.wall = now+round((elapsed if cost_ms is None else cost_ms)*FS/1000)
            return result
        monkeypatch.setattr(tx.sessrx.p4, name, costly_read)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    onair._run_p4_probe(tx, live, grid(), in_slot=True)
    entries = tx.p4_probe.emissions
    assert len(entries) >= 2, tx.p4_probe.link_events
    assert entries[1]['value'] == 38 and entries[1]['header'] == 'A'
    due = grid().boundary(11)
    assert abs(entries[1]['audio_start']-due) <= 2
    print(dict(answer=answer, cost_ms=cost_ms, reads_ms=durations, load=os.getloadavg()))


def test_weak_x27_confirms_before_decision_and_retry_keys_on_time(monkeypatch):
    from hfmodem.tests.shrike.p4_hold_helpers import check_driver_settlement

    check_driver_settlement(monkeypatch, answer_offset=.920, retry_at=60000)


def test_weak_ack_pair_follows_the_retry_even_when_the_key_is_late():
    from dataclasses import replace
    from hfmodem.tests.shrike.test_p4turn import machine, word
    from hfmodem.shrike.p4turn import State

    turn, queue = machine(data='body')
    queue.on_host_data(b'A'*31)
    turn._data(0, State.ISS_DATA_ACK)
    for at in (0, 60096):
        sent = turn.keyed(at)
        reg = word('X27', sent.stream_sample+45245, responds_to=sent.stream_sample)
        reg = replace(reg, observation=replace(reg.observation, weak=True, score=.8))
        turn.observe(reg)
    assert queue._buffer_raw == 0


def test_opening_retry_uses_the_configured_probe_values():
    from hfmodem.tests.shrike.test_p4turn import machine

    turn, _ = machine()
    turn.seed_opening([dict(audio_start=0, value=5), dict(audio_start=60000, value=6)], [])
    turn.silence(110000)
    proposal = turn.decide(110000)
    assert proposal.value == 5 and proposal.header == 'B'


def test_liveness_rejects_an_unexpected_word_outside_learned_turnaround():
    from hfmodem.tests.shrike.test_p4turn import word

    receiver = P4Receiver(48000, admitted=True, breakin=True)
    receiver.enable_hold()
    receiver.expected_words = {'Y27', 'V27', 'X27'}
    receiver.turnarounds['short'] = [.923]
    receiver.note_emission(Emission(Kind.BODY, Role.ISS, 48000, .8))
    receiver.register(word('Z27', 48000+round(.955*48000)).observation)
    assert receiver.finish_cycle(48000+round(.955*48000)) == 1


def test_held_body_quality_floor_uses_inline_estimates(monkeypatch):
    from hfmodem.shrike import p4hold
    from hfmodem.tests.shrike.test_p4turn import word

    frame, llr = scores()
    hold = p4hold.BreakinHold(48000)
    member = p4hold.Copy(0, 0, 0., 100., llr)
    member.inline_q = 1.
    hold.members, hold.last = [member], 0
    hold.frame = frame
    hold.codeword = (llr < 0).astype(np.uint8)
    monkeypatch.setattr(p4hold.p4breakin, 'body_scores', lambda *a, **kw: -llr)
    assert hold.inline(np.zeros(48000), 60000, word('Y27', 60000).observation) is None
    assert hold.events[-1]['reason'] == 'different body'


def test_independent_fallback_preserves_unsupported_source_for_the_host(monkeypatch):
    from hfmodem.shrike import p4hold
    from hfmodem.tests.shrike.test_p4turn import word, FS

    _, llr = scores(4)
    hold = p4hold.BreakinHold(FS)
    hold.members = [p4hold.Copy(0, 0, 0., 10., -llr)]
    hold.last = 0
    epoch = hold.epoch
    monkeypatch.setattr(p4hold.p4breakin, 'body_scores', lambda *a, **kw: llr)
    frame = hold.inline(np.zeros(FS), 60000, word('Y27', 60000).observation)
    assert frame.status.byte == 4
    assert hold.epoch == epoch+1 and len(hold.members) == 1


def test_clock_acquisition_counts_the_observed_peer_cycle(monkeypatch):
    from hfmodem.shrike import onair
    from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx, Stream
    from hfmodem.tests.shrike.test_ws8eoc_p4_silent_0927 import grid

    audio = np.zeros(round(1.3*FS))
    at = round(.2*FS)
    wave = stock.word('Y27')
    audio[at:at+len(wave)] = wave
    live = Stream(audio)
    tx = Tx(live)
    tx.p4_probe.emissions.clear()
    receiver = tx.sessrx.p4
    counts = []
    finish = receiver.finish_cycle
    def finish_cycle(onset):
        result = finish(onset)
        counts.append((onset, result))
        return result
    monkeypatch.setattr(receiver, 'finish_cycle', finish_cycle)
    onair._run_p4_link(tx, live, grid())
    assert counts and abs(counts[0][0]-at) <= 96 and counts[0][1] == 0
