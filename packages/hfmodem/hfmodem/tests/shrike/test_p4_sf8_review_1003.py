# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Review reproductions: established presence, callback retries and final TX admission."""
from dataclasses import replace
from hashlib import sha256
from time import monotonic, sleep
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import compress, onair, ota, p4frame, p4rx, p4sf8rx, ptc, session
from hfmodem.shrike.p4credit import ReceiptContext, SourceReceipt
from hfmodem.shrike.p4host import P4Host, P4LateFrame
from hfmodem.shrike.p4receiver import Emission, Kind, Observation, Registered, Role
from hfmodem.shrike.p4observation import RegisteredObservation
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.test_p4_sf8_live_1003 import FS, receiver, recording
from hfmodem.tests.shrike.test_p4turn import machine
from hfmodem.tests.shrike.test_tx_latency_correction import _Stream


@pytest.mark.parametrize('history', ['raster', 'turnaround', 'both'])
@pytest.mark.parametrize('placed', [False, True])
def test_long_presence_requires_complete_placement_even_with_history(history, placed):
    at = 10*FS
    rx = receiver(at, .69)
    if history in ('raster', 'both'):
        rx.peer_onset = at-180000
    if history in ('turnaround', 'both'):
        em = Emission(Kind.WORD, Role.IRS, at-round(.35*FS), .24)
        rx.emissions.append(em)
        rx.turnarounds['word'] = [.35]
    obs = Observation(Kind.LONG, FS, 0, at, at, 3.28 if placed else .9,
                      .69, .8, provisional=True, placed=placed, sf8=placed)
    rx.silent_cycles = 7
    rx.register(obs)
    assert rx.finish_cycle(at) == (0 if placed else 8)
    assert bool(rx.live_onsets) == placed
    turn, arq = machine(role="irs")
    heard = []
    arq.note_peer_heard = lambda: heard.append(True)
    arq.on_rx_p4(RegisteredObservation(obs, Registered.NEW))
    assert bool(heard) == placed


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('cut_s', [.9, 3.23, 3.275, 3.27925])
def test_recorded_early_ending_body_cannot_renew_established_peer(prepared, cut_s):
    at = 3528586
    origin = at-24000
    audio = recording('onair-1002-2154')[origin:at+round(3.30*FS)].copy()
    cut = at-origin+round(cut_s*FS)
    audio[cut:] = np.random.default_rng(108).normal(0, .001, len(audio)-cut)
    rx = receiver(at, .69)
    rx.peer_onset = at-180000
    rx.emissions.append(Emission(Kind.WORD, Role.IRS, at-round(.35*FS), .24))
    rx.turnarounds['word'] = [.35]
    rx.silent_cycles = 7
    rx.read_partial(audio[:at-origin+round(.845*FS)], origin,
                    deadline=at+round(.845*FS))
    if prepared:
        rx.prepare_sf8(audio[:at-origin+round(3.23*FS)], origin)
    rx.read_window(audio, origin, whole=True)
    assert rx.finish_cycle(at) == 8
    assert not rx.live_onsets


def require(clock):
    def admit():
        if clock[0] > 1.0:
            raise onair._MissedTxSlot('offline monotonic deadline expired')
    return admit


@pytest.mark.parametrize('stage', ['none', 'drain', 'wait', 'enqueue'])
def test_radio_tx_rechecks_monotonic_deadline_with_frozen_sample_clock(tmp_path, stage):
    clock = [0.]
    stream = _Stream(tx_latency_n=336)
    rig = SimpleNamespace(ptt=lambda _: True, key_failure=lambda: None)
    radio = onair.RadioTx(rig, transmit=True, outdir=tmp_path, settle=.04)
    radio.live = stream
    radio.p4_tx_admit = require(clock)
    due = 100000
    radio.aim(onair._MasterGrid(due, 60000, 0, packet_n=36000,
                               cs_n=12000, d_max_n=4800), 0)
    radio.p4_tx_capture_end = due-onair._prekey_lead(stream, round(.04*FS))-480
    if stage == 'drain':
        original = stream.take_until
        def stalled(end):
            result = original(end)
            clock[0] = 2.
            return result
        stream.take_until = stalled
    elif stage == 'wait':
        stream.wait_until = lambda _: clock.__setitem__(0, 2.)
    elif stage == 'enqueue':
        stream._tx_done.clear = lambda: clock.__setitem__(0, 2.)
    try:
        radio._tx(np.ones(10080, np.float32), 'offline deadline test')
    except onair._MissedTxSlot:
        assert stage in ('drain', 'wait')
    if stage == 'none':
        assert radio.slots_used == [0]
        assert radio.tx_audio_start == due
    else:
        assert not radio.slots_used and not stream.slept
        assert stream._tx is None


@pytest.mark.parametrize('stage', ['warmup', 'settle', 'none'])
def test_playback_rechecks_same_deadline_at_actual_play(monkeypatch, stage):
    clock, played, keys = [0.], [], []
    monkeypatch.setattr(ota, '_warm_output',
                        lambda _: clock.__setitem__(0, 2. if stage == 'warmup' else 0.))
    monkeypatch.setattr(ota.time, 'sleep',
                        lambda _: clock.__setitem__(0, 2. if stage == 'settle' else 0.))
    monkeypatch.setattr(ota, 'play_drained', lambda *a: played.append(True))
    rig = SimpleNamespace(ptt=lambda value: keys.append(value), stop=lambda: None)
    if stage == 'none':
        ota._play(np.ones(128), None, 1., rig, settle=.04, admit=require(clock))
        assert played == [True] and keys == [True, False]
    else:
        with pytest.raises(onair._MissedTxSlot):
            ota._play(np.ones(128), None, 1., rig, settle=.04, admit=require(clock))
        assert played == []
        assert keys == ([] if stage == 'warmup' else [True, False])


class Sink:
    def __init__(self, clock, expire_after=None):
        self.clock, self.expire_after = clock, expire_after
        self.events = []
        self._p4_sf8_admit = None

    def check(self):
        if self._p4_sf8_admit is not None and not self._p4_sf8_admit():
            raise P4LateFrame

    def effect(self, event):
        self.check()
        self.events.append(event)
        if len(self.events) == self.expire_after:
            self.clock[0] = 2.

    def deliver(self, payload):
        self.effect(('data', payload))

    def supervisor(self, code, params):
        self.effect(('block', code, params))


def netframe(data, status=0x63):
    return p4frame.Frame(data, p4frame.Status.from_byte(status), 0)


@pytest.mark.parametrize('expire_after', [1, 2])
def test_partial_external_delivery_is_not_repeated_after_rollback(expire_after):
    clock = [0.]
    sink = Sink(clock, expire_after)
    host = P4Host(sink)
    frame = netframe(b'payload'+compress.supervisor_block('B first')+
                     compress.supervisor_block('B second'))
    admit = lambda: clock[0] < 1.
    with pytest.raises(P4LateFrame):
        host.accept(frame, admit=admit)
    completed = list(sink.events)
    assert len(completed) == expire_after
    assert host.last is None and host._delivery_journal is not None
    # Another source operation cannot publish or discard the pending retry.
    with pytest.raises(P4LateFrame):
        host.end_epoch(0, 'timeout', terminal=True)
    assert sink.events == completed
    clock[0] = 0.
    assert host.accept(frame, admit=admit) == b'payload'
    assert sink.events == [('data', b'payload'), ('block', 'B', b'first'),
                           ('block', 'B', b'second')]
    assert host.accept(frame, admit=admit) is None
    assert len(sink.events) == 3 and host._delivery_journal is None


def held_source(sink):
    host = P4Host(sink)
    prefix = b'held-prefix'
    receipt = SourceReceipt(1, (Kind.BODY, FS), (0, prefix))
    host.candidate(receipt, prefix, 'opening')
    host.arm(receipt)
    assert host.accept(netframe(prefix[:4], 0x61),
                       receipt_context=ReceiptContext(receipt, Kind.CHIRP)) == b''
    assert host.result.held_total == 4
    return host, receipt


@pytest.mark.parametrize('recover', [False, True])
def test_internal_release_retains_admission_after_empty_callback_rollback(recover):
    sink = ptc.PtcHost()
    delivered = []
    original_deliver = sink.deliver
    def deliver(payload):
        original_deliver(payload)
        delivered.append(payload)
    sink.deliver = deliver
    host, receipt = held_source(sink)
    prior_identity = host.last
    original_feed = host.supervisor.feed
    stalled = []
    def feed(field):
        payload = original_feed(field)
        if not stalled:
            stalled.append(True)
            sleep(.060)
        return payload
    host.supervisor.feed = feed
    turn, _ = machine(role='irs')
    turn.adapter = host
    turn._receipt = receipt
    turn._receive_epoch = 1
    turn._receive_active = True
    frame = p4frame.decode_field(p4frame.encode_field(
        b'next source'.ljust(141, b'\x1e'), p4frame.Status.from_byte(0x63)),
        data_bytes=141)
    assert frame is not None
    obs = Observation(Kind.LONG, FS, 0, 10*FS, 10*FS, 3.28, 0., .9,
                      frame=frame, placed=True, sf8=True)
    deadlines = {obs.stream_sample: monotonic()+.050}
    turn.sf8_eligible = lambda seen: monotonic() < deadlines[seen.stream_sample]
    turn.observe(RegisteredObservation(obs, Registered.NEW), now=obs.stream_sample)
    assert not turn.sf8_eligible(obs)
    assert sink.rcvd_total == 0 and not delivered
    assert host.last == prior_identity and host.result.held_total == 4
    assert host._receipts[receipt].held == b'held'
    assert host._receipts[receipt].state == 'matching' and host._active == receipt
    assert host._delivery_journal['release'] and not host._delivery_journal['done']
    with pytest.raises(P4LateFrame):
        host.accept(netframe(b'unrelated source'), admit=lambda: True)
    assert sink.rcvd_total == 0 and host._receipts[receipt].held == b'held'
    if recover:
        # A distinct received copy has its own deadline; the refused copy's
        # original deadline stays expired. Retry the exact source operation.
        later = replace(obs, stream_sample=obs.stream_sample+180000,
                        first_chip=obs.first_chip+180000)
        deadlines[later.stream_sample] = monotonic()+1.
        turn.observe(RegisteredObservation(later, Registered.NEW), now=later.stream_sample)
        assert not turn.sf8_eligible(obs) and turn.sf8_eligible(later)
        assert delivered == [b'held', b'next source'] and sink.rcvd_total == 15
        assert host.result.released_source == 4 and host.result.held_total == 0
        assert host.result.delivery == b'heldnext source'
        assert host._delivery_journal is None and host._active is None
        repeat = replace(later, stream_sample=later.stream_sample+180000,
                         first_chip=later.first_chip+180000)
        deadlines[repeat.stream_sample] = monotonic()+1.
        result = host.result
        turn.observe(RegisteredObservation(repeat, Registered.NEW), now=repeat.stream_sample)
        # The unanswered copy passed its due and is held as deferred; its repeat
        # takes that answer into its own slot without re-entering the source.
        sample, pending = turn._deferred
        assert sample == later.stream_sample and pending['current_slot'] == repeat.stream_sample
        assert pending['current_slot_accepted'] and repeat.stream_sample in turn._decided
        assert host.result is result
        assert delivered == [b'held', b'next source'] and sink.rcvd_total == 15
        deadlines[later.stream_sample] = monotonic()-1.
        assert not turn.sf8_eligible(later)
        answer = turn.decide(repeat.stream_sample+round(3.4*FS)-turn.lead)
        assert answer.ack_identity == (0x63, frame.data) and answer.current_slot_accepted
        assert turn.sf8_source(answer) is repeat and turn.can_key(answer)
        turn.link_down('session ceiling')
        assert delivered == [b'held', b'next source'] and host.failure is None
    else:
        # Production shutdown must retain the refused release and report it.
        turn.link_down('session ceiling')
        assert not turn.sf8_eligible(obs)
        assert sink.rcvd_total == 0 and not delivered
        assert host._receipts[receipt].held == b'held' and host._active == receipt
        assert host._receipts[receipt].state == 'matching'
        assert host.failure == 'P4 source delivery incomplete at link end'
        assert any(row['event'] == 'source-release-deferred' for row in turn.link_events)
        assert host._delivery_journal['release'] and not host._delivery_journal['done']


def test_late_accept_without_held_release_does_not_pin_unrelated_source():
    sink = ptc.PtcHost()
    host = P4Host(sink)
    original = host.supervisor.feed
    deadline = monotonic()+.050
    def stalled(field):
        payload = original(field)
        sleep(.060)
        return payload
    host.supervisor.feed = stalled
    with pytest.raises(P4LateFrame):
        host.accept(netframe(b'refused'), admit=lambda: monotonic() < deadline)
    assert sink.rcvd_total == 0 and host.last is None
    assert host._delivery_journal is None
    host.supervisor.feed = original
    next_deadline = monotonic()+1.
    assert host.accept(netframe(b'later source'),
                       admit=lambda: monotonic() < next_deadline) == b'later source'
    assert sink.rcvd_total == 12 and monotonic() > deadline


def test_driver_cleanup_does_not_reopen_retained_source_admission(monkeypatch):
    from hfmodem.shrike import p4turn
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx

    live = Stream(np.zeros(4*FS, np.float32))
    tx = Tx(live)
    rx = tx.sessrx.p4
    tx.p4_link_options.p4_link_timeout = 4.
    original = p4turn.P4Turn
    machines = []
    obs = Observation(Kind.LONG, FS, 0, 0, 0, 3.28, 0., .9,
                      frame=netframe(b'old source'), placed=True, sf8=True)
    admitted = []
    take_until = live.take_until
    def take(end):
        if not admitted:
            admitted.append(machines[0].sf8_eligible(obs))
        return take_until(end)
    live.take_until = take
    def make_machine(*args, **kwargs):
        turn = original(*args, **kwargs)
        machines.append(turn)
        return turn
    monkeypatch.setattr(p4turn, 'P4Turn', make_machine)
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert admitted == [True]
    assert rx.sf8_deadline is None and tx.p4_link_host.arq.p4_sf8_eligible is None
    assert live.pos >= round(3.4*FS)
    assert not machines[0].sf8_eligible(obs)
    # Reusing/resetting a receiver cannot renew a held source's old deadline.
    live.pos = 0
    assert not machines[0].sf8_eligible(obs)


@pytest.mark.parametrize('operation', ['cancel', 'epoch', 'sf8-close'])
def test_credit_release_checks_admission_before_host_callback(operation):
    clock = [0.]
    sink = Sink(clock)
    host, receipt = held_source(sink)
    original = host.supervisor.feed
    def stalled(field):
        result = original(field)
        clock[0] = 2.
        return result
    host.supervisor.feed = stalled
    admit = lambda: clock[0] < 1.
    if operation == 'sf8-close':
        turn, _ = machine(role='irs')
        turn.adapter = host
        turn._receipt = receipt
        turn._receive_epoch = 1
        turn._receive_active = True
        turn.sf8_eligible = lambda _: admit()
        obs = Observation(Kind.LONG, FS, 0, 10*FS, 10*FS, 3.28, 0., .9,
                          frame=netframe(b'', 0x23), placed=True, sf8=True)
        turn.observe(RegisteredObservation(obs, Registered.NEW), now=10*FS)
        assert turn._receipt == receipt
        assert not turn._sf8_acks
    else:
        with pytest.raises(P4LateFrame):
            if operation == 'cancel':
                host.cancel(receipt, 'peer close', admit=admit)
            else:
                host.end_epoch(1, 'peer close', terminal=True, admit=admit)
    assert not sink.events
    assert host._receipts[receipt].held == b'held'
    assert host._receipts[receipt].state == 'matching'
    assert host._active == receipt
    # Cleanup cannot turn the refused source release into an unguarded one.
    with pytest.raises(P4LateFrame):
        host.cancel(receipt, 'timeout')
    with pytest.raises(P4LateFrame):
        host.end_epoch(1, 'timeout', terminal=True)
    if operation == 'sf8-close':
        turn.link_down('timeout')
        assert host.failure == 'P4 source delivery incomplete at link end'
    assert not sink.events


@pytest.mark.parametrize('n', range(6))
def test_1123_witness_sf8_protocol3_crc_and_repeat_identity(n):
    path = require_input(CAPTURES/'onair-1003-1123/witness.wav')
    audio = session.load_wav(str(path), FS)
    at = 2268669+180000*n+round(1.103*FS)
    origin = at-round(.5*FS)
    z = p4rx.audio_to_baseband(audio[origin:at+round(3.4*FS)], FS)
    observed = p4rx.recover_robust(z, reference_start=19520, spread_factor=8,
        data_symbols=696, timing_offsets=range(-48, 49, 2), cfo_hz=(16.6,))
    result = p4sf8rx.decode_symbols(observed.symbols)
    assert result is not None
    assert (result.frame.status.byte, result.frame.status.protocol,
            result.frame.status.counter, len(result.frame.data)) == (0x63, 3, 3, 141)
    assert sha256(result.frame.data).hexdigest() == '180f29754d8232d8f1e4f7f4c401729ae4ca3bbca8140f39d8ebcd3b4040d494'
    assert result.correction == ('amplitude-c1' if n == 5 else 'systematic')
    host = P4Host()
    host.accept(result.frame)
    assert host.accept(result.frame) is None


@pytest.mark.parametrize('epoch', [False, True])
def test_partially_completed_credit_release_retries_without_duplicate(epoch):
    clock = [0.]
    sink = Sink(clock, expire_after=1)
    host = P4Host(sink)
    prefix = b'held'+compress.supervisor_block('B id')+b'end'
    receipt = SourceReceipt(1, (Kind.BODY, FS), (0, prefix))
    host.candidate(receipt, prefix, 'opening')
    host.arm(receipt)
    host.accept(netframe(prefix[:-3], 0x61),
                receipt_context=ReceiptContext(receipt, Kind.CHIRP))
    def release():
        if epoch:
            return host.end_epoch(1, 'peer close', terminal=True, admit=lambda: clock[0] < 1.)
        return host.cancel(receipt, 'peer close', admit=lambda: clock[0] < 1.)
    with pytest.raises(P4LateFrame):
        release()
    assert sink.events == [('data', b'held')]
    assert host._receipts[receipt].held == prefix[:-3]
    clock[0] = 0.
    release()
    assert sink.events == [('data', b'held'), ('block', 'B', b'id')]
    assert host._active is None and host._delivery_journal is None
