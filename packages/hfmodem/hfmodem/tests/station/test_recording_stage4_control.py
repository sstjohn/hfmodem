# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Attached-child JSON controls share IDs and independent source anchors.

core.recording exposes ChildControlSender(fd, capacity, report), send(record),
replace(fd, anchors), bind(session, anchors), close(timeout), and
ChildControlReceiver(session, fd), drain().
Records use op/contact_id/positions/anchors/provisional/keep/reason. A source
anchor maps parent sample/clock to child position/rate/clock without resampling.
"""
import json
import os
import threading
from types import SimpleNamespace as NS

import pytest

from hfmodem.tests.station.recording_support import TIMEOUT, bounded, flush, pcm, push, audio_files, readable


def module():
    from hfmodem.core import recording
    sender = getattr(recording, 'ChildControlSender', None)
    receiver = getattr(recording, 'ChildControlReceiver', None)
    assert sender is not None and receiver is not None, 'shared inherited child control transport is missing'
    return NS(Sender=sender, Receiver=receiver)


def messages(fd, count):
    data = b''
    for _ in range(count + 2):
        if data.count(b'\n') >= count:
            break
        ready = readable(fd)
        assert ready, 'child boundary delivery timed out'
        block = os.read(fd, 65536)
        if not block:
            break
        data += block
    return [json.loads(line) for line in data.splitlines()]


def anchor(sample=0, position=100, clock='kiwi-1'):
    return dict(rx=dict(clock=clock, parent_clock='card', sample=sample,
                        position=position, rate=8.25, parent_rate=8))


@pytest.mark.parametrize('provisional,keep', [(False, True), (True, True), (True, False)])
def test_child_receives_parent_id_padding_and_frozen_decision(sessions, provisional, keep):
    control = module()
    rec, root = sessions()
    lane = rec.stream('rx', rate=8, clock='kiwi-1')
    rd, wr = os.pipe()
    receiver = control.Receiver(rec, rd)
    sender = control.Sender(wr, capacity=4, report=lambda text: None)
    try:
        sender.send(dict(op='begin', contact_id='parent-c1', positions={'card': 32},
                         anchors=anchor(), provisional=provisional))
        ready = readable(rd)
        assert ready, 'begin control was not delivered'
        receiver.drain()
        flush(rec)
        assert 'parent-c1' in rec.index['contacts']
        push(rec, lane, 100, 164)
        sender.send(dict(op='end', contact_id='parent-c1', positions={'card': 64},
                         anchors=anchor(), keep=keep, reason='operator'))
        ready = readable(rd)
        assert ready, 'end control was not delivered'
        receiver.drain()
        push(rec, lane, 164, 212)
        sender.close(timeout=TIMEOUT)
        receiver.drain()
        bounded(rec.close)
        if keep:
            row = rec.index['contacts']['parent-c1']
            assert row['positions'] == {'kiwi-1': 133}
            assert row['lanes']['rx']['actual'][0] == 100
        else:
            assert audio_files(root) == []
    finally:
        os.close(rd)


@pytest.mark.parametrize('provisional,keep', [(False, True), (True, None), (True, True), (True, False)])
def test_control_eof_keeps_only_decided_contacts_as_interrupted(sessions, provisional, keep):
    control = module()
    rec, root = sessions()
    lane = rec.stream('rx', rate=8, clock='kiwi-1')
    rd, wr = os.pipe()
    receiver = control.Receiver(rec, rd)
    try:
        rows = [dict(op='begin', contact_id='c1', positions={'kiwi-1': 0},
                     provisional=provisional, anchors={})]
        if keep is not None and provisional:
            rows.append(dict(op='end', contact_id='c1', positions={'kiwi-1': 16},
                             keep=keep, reason='operator'))
        os.write(wr, b''.join((json.dumps(row)+'\n').encode() for row in rows))
        receiver.drain()
        push(rec, lane, 0, 16)
        os.close(wr)
        receiver.drain()
        bounded(rec.close)
        assert bool(audio_files(root)) == (not provisional or keep is True)
        if not provisional:
            assert rec.index['contacts']['c1']['state'] == 'partial'
    finally:
        os.close(rd)


def test_replaced_receiver_gets_active_ids_and_fresh_clock_anchors():
    control = module()
    rd, wr = os.pipe()
    rd2, wr2 = os.pipe()
    sender = control.Sender(wr, capacity=4, report=lambda text: None)
    try:
        sender.send(dict(op='begin', contact_id='c1', positions={'card': 16},
                         anchors=anchor(), provisional=True))
        first = messages(rd, 1)
        bounded(lambda: sender.replace(wr2, anchors=anchor(80, 0, 'kiwi-2')))
        replacement = messages(rd2, 1)
        assert first[0]['contact_id'] == replacement[0]['contact_id'] == 'c1'
        assert replacement[0]['anchors'] == anchor(80, 0, 'kiwi-2')
        assert replacement[0]['provisional'] is True
    finally:
        sender.close(timeout=TIMEOUT)
        os.close(rd)
        os.close(rd2)


def test_blocked_pipe_never_blocks_producer_and_reserves_end_close(monkeypatch):
    control = module()
    rd, wr = os.pipe()
    entered, release = threading.Event(), threading.Event()
    reports, threads = [], []
    write = os.write
    def stalled(fd, data):
        if fd == wr:
            threads.append(threading.get_ident())
            entered.set()
            assert release.wait(TIMEOUT), 'blocked child pipe was not released'
        return write(fd, data)
    monkeypatch.setattr(os, 'write', stalled)
    sender = control.Sender(wr, capacity=4, report=reports.append)
    producer = threading.get_ident()
    try:
        sender.send(dict(op='begin', contact_id='c1', positions={}, anchors={}, provisional=False))
        assert entered.wait(TIMEOUT)
        def flood():
            for n in range(40):
                sender.send(dict(op='anchor', positions={'card': n}, anchors={}))
            sender.send(dict(op='end', contact_id='c1', positions={'card': 40}, keep=True, reason='closed'))
        bounded(flood)
        assert all(t != producer for t in threads)
    finally:
        release.set()
        sender.close(timeout=TIMEOUT)
    rows = messages(rd, 2)
    os.close(rd)
    assert any(row['op'] == 'end' and row['contact_id'] == 'c1' for row in rows)
    assert rows[-1]['op'] == 'close'
    assert sum(row['op'] == 'anchor' for row in rows) <= 4
    assert reports, 'lost child coverage must be reported'


@pytest.mark.parametrize('policy', ('contacts', 'everything'))
def test_station_owner_shared_boundaries_fan_out_ids_and_event_clock(
        stage4, monkeypatch, policy):
    control = module()
    from hfmodem.core.audio import Block
    from hfmodem.station import capture
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setattr(capture, 'CARD_RATE_HZ', 8)
    owner = capture.StationCapture(stage4.root, 'synthetic')
    rd, wr = os.pipe()
    sender = control.Sender(wr, capacity=4, report=lambda text: None)
    try:
        sender.bind(owner.session, anchors=anchor())
        owner.push(Block(0., 0, pcm(0, 16)))
        owner.push_tx(100, pcm(0, 4))
        flush(owner.session)
        key = (owner.boundaries.run, 'vara', 'endpoint', 'K9ZZT', 'peer', 1)
        cid = owner.boundaries.established(key, 'W0QXR', owner.positions())
        owner.boundaries.established(key, 'W0QXR', owner.positions())
        owner.push(Block(2., 16, pcm(16, 24)))
        flush(owner.session)
        owner.boundaries.ended(key, owner.positions(), 'peer closed')
        owner.boundaries.ended(key, owner.positions(), 'duplicate')
        bounded(owner.close)
        sender.close(timeout=TIMEOUT)
        delivered = messages(rd, 3)
        begins = [row for row in delivered if row['op'] == 'begin']
        ends = [row for row in delivered if row['op'] == 'end']
        assert len(begins) == len(ends) == 1
        assert begins[0]['contact_id'] == ends[0]['contact_id'] == cid
        assert begins[0]['positions'] == {'card': 16, 'dac': 104}
        assert ends[0]['positions'] == {'card': 24, 'dac': 104}
        assert begins[0]['anchors'] == anchor()
    finally:
        os.close(rd)
        bounded(owner.close)


def test_inherited_owner_controls_relay_to_replaced_witness(sessions, monkeypatch):
    control = module()
    source, destination = os.pipe()
    rd, wr = os.pipe()
    rd2, wr2 = os.pipe()
    sender = control.Sender(wr, capacity=4, report=lambda text: None)
    relay = control.Receiver(sender, source)
    monkeypatch.setenv('HFMODEM_RECORDING_CONTROL_OUT_FDS', str(destination))
    rec, _ = sessions('off')
    try:
        rec.begin('shared-c1', {'card': 16}, provisional=True)
        assert readable(source)
        relay.drain()
        assert messages(rd, 1)[0]['contact_id'] == 'shared-c1'
        sender.replace(wr2, anchors=anchor(80, 0, 'kiwi-2'))
        assert messages(rd2, 1)[0]['anchors'] == anchor(80, 0, 'kiwi-2')
        rec.end('shared-c1', {'card': 88}, keep=True)
        assert readable(source)
        relay.drain()
        end = messages(rd2, 1)[0]
        assert end['op'] == 'end' and end['keep'] is True
        assert end['positions'] == {'card': 88}
        bounded(rec.close)
        relay.drain()
    finally:
        sender.close(timeout=TIMEOUT)
        for fd in (source, destination, rd, rd2):
            os.close(fd)


def test_blocked_delivery_bounds_successive_contact_lifetimes(monkeypatch):
    control = module()
    rd, wr = os.pipe()
    entered, release = threading.Event(), threading.Event()
    reports = []
    write = os.write
    def stalled(fd, data):
        if fd == wr:
            entered.set()
            assert release.wait(TIMEOUT)
        return write(fd, data)
    monkeypatch.setattr(os, 'write', stalled)
    sender = control.Sender(wr, capacity=4, report=reports.append)
    try:
        sender.send(dict(op='begin', contact_id='c0', positions={}))
        assert entered.wait(TIMEOUT)
        def flood():
            for n in range(40):
                cid = f'c{n}'
                sender.send(dict(op='begin', contact_id=cid, positions={}))
                sender.send(dict(op='end', contact_id=cid, positions={}, keep=True))
        bounded(flood)
    finally:
        release.set()
        sender.close(timeout=TIMEOUT)
    rows = messages(rd, 1)
    os.close(rd)
    begins = {row['contact_id'] for row in rows if row['op'] == 'begin'}
    ends = {row['contact_id'] for row in rows if row['op'] == 'end'}
    assert len(begins) <= 4 and ends == begins
    assert rows[-1]['op'] == 'close' and reports


def test_console_launcher_maps_delayed_source_edges_and_replacement(stage4, monkeypatch):
    import importlib
    from hfmodem.core import recording
    from hfmodem.core.audio import Block
    from hfmodem.station import capture
    station = importlib.import_module('consolelib.station')
    monkeypatch.setenv('HFMODEM_RECORDING', 'contacts')
    monkeypatch.setattr(capture, 'CARD_RATE_HZ', 8)
    monkeypatch.setattr(recording.time, 'monotonic', stage4.clock)
    plan = dict(env={'HFMODEM_RECORDING': 'contacts'})
    env, rd, sender = station.attached_control(plan)
    monkeypatch.setenv('HFMODEM_RECORDING_CONTROL_OUT_FDS', plan['env']['HFMODEM_RECORDING_CONTROL_OUT_FDS'])
    owner = capture.StationCapture(stage4.root / 'parent', 'synthetic')
    child = recording.RecordingSession('contacts', stage4.root / 'child', attached_children=False)
    lane = child.stream('witness', rate=8, clock='kiwi-1')
    lane.row.update(monotonic_origin=1000., measured_rate=8.)
    receiver = recording.ChildControlReceiver(child, rd)
    replacement_fd = None
    try:
        owner.push(Block(1000., 0, pcm(0, 8)))
        flush(owner.session)
        stage4.clock.now = 1015.
        owner.session.begin('delayed', {'card': 16})
        assert readable(rd)
        receiver.drain()
        flush(child)
        assert child.index['contacts']['delayed']['positions'] == {'kiwi-1': 16}
        env2, replacement_fd, _ = station.attached_control(plan, sender=sender)
        replacement = recording.RecordingSession('contacts', stage4.root / 'replacement', attached_children=False)
        lane2 = replacement.stream('witness', rate=8, clock='kiwi-2')
        lane2.row.update(monotonic_origin=1010., measured_rate=8.)
        receiver2 = recording.ChildControlReceiver(replacement, replacement_fd)
        assert env2['HFMODEM_RECORDING_CONTROL_FD'] == str(replacement_fd)
        assert readable(replacement_fd)
        receiver2.drain()
        flush(replacement)
        assert replacement.index['contacts']['delayed']['positions'] == {'kiwi-2': -64}
        stage4.clock.now = 1020.
        owner.session.end('delayed', {'card': 96})
        assert readable(replacement_fd)
        receiver2.drain()
        flush(replacement)
        assert replacement.index['contacts']['delayed']['end_positions'] == {'kiwi-2': 16}
    finally:
        bounded(owner.close)
        station.close_attached_controls(plan)
        sender.close(timeout=TIMEOUT)
        os.close(rd)
        if replacement_fd is not None:
            os.close(replacement_fd)


@pytest.mark.parametrize('ppm', (-200., 200.))
def test_adc_drift_preserves_contiguous_samples_and_child_bounds(sessions, monkeypatch, ppm):
    from hfmodem.core import recording
    parent, _ = sessions('everything', attached_children=False)
    child, _ = sessions('off', attached_children=False)
    rate, size = 8000, 80
    measured = rate * (1 + ppm / 1e6)
    lane = parent.stream('rx', rate=rate, clock='adc')
    witness = child.stream('tap', rate=rate, clock='tap')
    witness.row.update(monotonic_origin=1000., measured_rate=measured, clock_uncertainty_s=0.)
    rd, wr = os.pipe()
    sender = recording.ChildControlSender(wr)
    sender.bind(parent, anchors={})
    receiver = recording.ChildControlReceiver(child, rd)
    try:
        for n in range(400):
            adc = 100. + n * size / measured
            monkeypatch.setattr(recording.time, 'monotonic', lambda adc=adc: 900. + adc)
            lane.push_adc(pcm(0, size), NS(inputBufferAdcTime=adc, currentTime=adc))
            flush(parent)
        assert lane.position == 400 * size
        assert lane.gaps == []
        parent.begin('drift', {'adc': 8000})
        parent.end('drift', {'adc': 24000})
        sender.close(timeout=TIMEOUT)
        receiver.drain()
        flush(child)
        row = child.index['contacts']['drift']
        assert row['positions'] == {'tap': 8000}
        assert row['end_positions'] == {'tap': 24000}
        assert lane.row['measured_rate'] == pytest.approx(measured, rel=1e-7)
        first = lane.position
        adc = 100. + (400 * size + size) / measured
        monkeypatch.setattr(recording.time, 'monotonic', lambda: 900. + adc)
        lane.push_adc(pcm(0, size), NS(inputBufferAdcTime=adc, currentTime=adc))
        flush(parent)
        assert lane.gaps == [[first, first + size]]
        assert lane.position == first + 2 * size
    finally:
        sender.close(timeout=TIMEOUT)
        os.close(rd)


def test_control_launch_hands_one_child_only_the_ends_it_is_given(monkeypatch):
    from hfmodem.core import claim, recording
    out = 'HFMODEM_RECORDING_CONTROL_OUT_FDS'
    base = {out: '21,22', 'HFMODEM_RECORDING_CONTROL_FD': '23', 'KEEP': '1'}
    monkeypatch.setattr(recording, '_WINDOWS', False)
    env, options = recording.control_launch(base)
    assert options == {'pass_fds': (21, 22)}
    assert env == {out: '21,22', 'KEEP': '1'}
    env, options = recording.control_launch({'KEEP': '1'}, read=24)
    assert options == {'pass_fds': (24,)}
    assert env == {'KEEP': '1', 'HFMODEM_RECORDING_CONTROL_FD': '24'}
    assert recording.control_launch({out: '', 'KEEP': '1'}) == ({'KEEP': '1'}, {})

    extras = {'recording-out-0': 31}
    monkeypatch.setattr(claim, 'extra', lambda name, optional=False: extras[name])
    monkeypatch.setattr(recording, '_WINDOWS', True)
    # A launcher forwards a received extra beside its own witness writer.
    env, options = recording.control_launch({out: 'recording-out-0,22', 'KEEP': '1'}, read=24)
    assert options == {'handles': {'recording-out-0': 31, 'recording-out-1': 22,
                                   'recording-control': 24}}
    assert env == {out: 'recording-out-0,recording-out-1', 'KEEP': '1',
                   'HFMODEM_RECORDING_CONTROL_FD': 'recording-control'}
    assert recording.control_launch({'KEEP': '1'}) == ({'KEEP': '1'}, {})
    assert recording.control_fd({'HFMODEM_RECORDING_CONTROL_FD': 'recording-out-0'}) == 31


def test_an_idle_sender_writes_nothing_and_a_replacement_gets_the_live_contacts():
    """A witness that failed to place leaves no reader on the pipe: the launcher's
    controls must go nowhere, without a broken-pipe report, until a replacement."""
    from hfmodem.core import recording
    reports = []
    rd, wr = os.pipe()
    sender = recording.ChildControlSender(wr, report=reports.append)
    try:
        sender.send(dict(op='begin', contact_id='a', positions={}))
        assert messages(rd, 1)[0]['contact_id'] == 'a'
        os.close(rd)
        sender.idle()
        sender.send(dict(op='begin', contact_id='b', positions={}))
        sender.send(dict(op='end', contact_id='a', positions={}))
        rd2, wr2 = os.pipe()
        try:
            for _ in range(200):
                if not sender._retiring:
                    break
                threading.Event().wait(.01)
            else:
                pytest.fail('the idled pipe never retired')
            sender.replace(wr2, anchors={})
            assert [m['contact_id'] for m in messages(rd2, 1)] == ['b']
        finally:
            sender.close(timeout=TIMEOUT)
            os.close(rd2)
        assert reports == []
    finally:
        sender.close(timeout=TIMEOUT)


def test_idling_with_an_end_still_queued_frees_its_contacts_slot():
    from hfmodem.core import recording
    rd, wr = os.pipe()
    sender = recording.ChildControlSender(wr, capacity=1, report=lambda line: None)
    try:
        with sender._condition:
            sender.send(dict(op='begin', contact_id='a', positions={}))
            sender.send(dict(op='end', contact_id='a', positions={}))
            sender.idle()
        sender.send(dict(op='begin', contact_id='b', positions={}))
        assert set(sender._active) == {'b'}
    finally:
        sender.close(timeout=TIMEOUT)
        os.close(rd)
