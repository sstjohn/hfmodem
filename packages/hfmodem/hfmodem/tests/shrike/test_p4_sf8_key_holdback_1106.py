# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Transcript-driven timing replay; the 04:53 arm intentionally recorded no PCM.

Only the RF frontend is substituted: terminated coded symbols are reconstructed from
its printed CRC-valid frame. Decode admission, source admission, the deadline
closure, P4Turn.can_key and the driver's Refused/Keyed path are production code.
The transcript has decision positions, not individual sample_now() calls.
"""
import re
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, p4frame, p4longdecode, p4receiver, p4robustlongtx, p4turn
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, Observation
from hfmodem.tests.evidence import RECORD, require_input
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
from hfmodem.tests.shrike.p4_declared_clock_1111 import declared_time

FS = 48000
ARM = 'onair-20261106T045338027084Z-aca20de9-4170-497c-b28e-e2727baa49f1'


def recorded_frame(peer):
    directory = require_input(RECORD / 'operator-data' / 'working' / ARM)
    paths = list(directory.glob('pactor-*.log'))
    assert len(paths) == 1, f'Missing/ambiguous required transcript: {directory}'
    text = paths[0].read_text()
    assert 'Recording: off (inherited)' in text
    at = int(re.search(r"\[p4 read\] \{'at': (\d+).*?\n"
                       rf'.*?source-(?:accept|replay) at {peer}', text)[1])
    frame = re.search(rf'\[p4 frame\] long at {peer} status=0x(\w+) data=(\w+)', text)
    decision = re.search(rf'peer={peer} .*?margin=([\d.]+)ms capture_lag=([\d.]+)ms'
                         r'.*?nominal_due=([\d.]+)', text)
    decode = re.search(rf'sf8-long at {peer} .*?crc_valid=True timely=True '
                       r'placement_ms=([\d.]+) decode_ms=([\d.]+)', text)
    assert frame and decision and decode
    assert f'Refused deadline: word at {float(decision[3]):.3f}' in text
    clock = re.search(r'block (\d+), ADC->DAC (\d+) samples .*?\+ (\d+) samples '
                      r'.*?logical DAC notice (\d+) .*?holdback (\d+)', text)
    assert clock
    return SimpleNamespace(at=at, status=int(frame[1], 16), data=bytes.fromhex(frame[2]),
                           margin=float(decision[1]), lag=float(decision[2]),
                           due=round(float(decision[3])*FS),
                           placement=float(decode[1])/1000, decode=float(decode[2])/1000,
                           block=int(clock[1]), latency=int(clock[3]),
                           notice=int(clock[4]), holdback=int(clock[5]))


def recorded_symbols(row):
    field = p4frame.encode_field(row.data, p4frame.Status.from_byte(row.status))
    return p4robustlongtx.body_symbols(p4robustlongtx.coded_bits(field, 4), 4)


def replay(monkeypatch, row, *, early_ms=0., completion='recorded', key_stall=False,
           full_holdback=False):
    class Driver(Stream):
        def __init__(self):
            super().__init__(np.zeros(row.at+4*FS, np.float32))
            self.pos = row.at-FS//2
            self.wall = self.pos/FS
            self.transmit = True
            self._blk = row.block
            self.key_notice = row.notice
            self.tx_latency_n = row.latency
            self.holdback = row.holdback
            self.early = early_ms*FS/1000
            # The printed lag includes placement/decode and the other readers.
            self.delivery = round((row.lag/1000-row.decode-row.placement)*FS)
            assert 0 < self.delivery <= self.holdback
            self.decoded = []
            self.proposals = []

        def sample_now(self):
            return round(self.wall*FS)+self.early

        def take_until(self, end):
            part = super().take_until(end)
            self.wall = max(self.wall, (self.pos+self.delivery)/FS)
            self.early = 0.
            return part

        def clamp_late(self, at):
            return max(0, self.sample_now()+self.key_notice-at)

    live = Driver()
    monkeypatch.setattr(onair.time, 'monotonic', lambda: live.wall)
    monkeypatch.setattr(onair.time, 'perf_counter', lambda: live.wall)
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: live.wall)
    tx = Tx(live)
    tx.settle = .040
    tx.p4_probe.emissions.clear()
    tx.p4_probe.handoff_audio = (live.pos, np.empty(0, np.float32))
    tx.p4_link_options.p4_link_timeout = 4.
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_long_ack = 'toggle'
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, row.at, .33, 3.75, 4.06)
    machines = []
    turn_type = p4turn.P4Turn

    def make_turn(*a, **kw):
        turn = turn_type(*a, **{**kw, 'role': 'irs'})
        decide = turn.decide

        def stalled_decide(now):
            proposal = decide(now)
            if proposal is not None:
                live.proposals.append(proposal)
                if full_holdback:
                    live.delivery = live.holdback
                if key_stall:
                    live.wall = deadlines[0]['limits'].enqueue.wall_deadline+.001
            return proposal

        turn.decide = stalled_decide
        machines.append(turn)
        return turn

    monkeypatch.setattr(p4turn, 'P4Turn', make_turn)
    symbols = recorded_symbols(row)
    recovered = SimpleNamespace(symbols=symbols, symbol_coherence=np.ones(len(symbols)),
                                coherence=.68, reference_start=8000)
    tail = SimpleNamespace(symbols=symbols[-25:], symbol_coherence=np.ones(25),
                           coherence=.68, reference_start=0)
    def recover(*a, **kw):
        live.wall += row.placement
        return tail

    monkeypatch.setattr(rx, '_recover_sf8', recover)
    monkeypatch.setattr(rx, '_sf8_energy_present', lambda *a: True)
    advance = p4longdecode.LongSearch.advance
    cut = row.due-onair._prekey_lead(live, round(tx.settle*FS))
    deadlines = []

    def grant(search, **kw):
        limits = rx.long_deadline(row.at, sf8=True)
        deadlines.append(dict(limits=limits, work=kw['deadline'],
                              publication=limits.publication.wall_deadline,
                              enqueue=limits.enqueue.wall_deadline))
        result = advance(search, **kw)
        assert result.frame is not None and result.frame.status.byte == row.status
        assert result.frame.data == row.data
        completed = ((cut-round(row.margin*FS/1000))/FS if completion == 'recorded'
                     else limits.work.wall_deadline+.001 if completion == 'work-reserve'
                     else limits.publication.wall_deadline+.001)
        assert completed >= live.wall
        live.wall = completed
        return result

    monkeypatch.setattr(p4longdecode.LongSearch, 'advance', grant)

    def partial(*a, **kw):
        obs = Observation(Kind.LONG, FS, 0, row.at, row.at, .845, 4.06, .68,
                          provisional=True)
        result = []
        rx._emit(obs, result)
        return result

    def window(audio, origin, **kw):
        result = []
        if origin+len(audio) >= row.at+round(3.280*FS) and not rx._sf8_cache:
            obs = rx._sf8_long(audio, origin)
            if obs is not None:
                live.decoded.append(obs)
                rx._emit(obs, result)
        return result

    monkeypatch.setattr(rx, 'read_partial', partial)
    monkeypatch.setattr(rx, 'read_window', window)
    def prepare(*a):
        rx._sf8_prepared[row.at, 4.06] = recovered
        rx._sf8_rf_references[row.at, 4.06] = 0.
        return recovered

    monkeypatch.setattr(rx, 'prepare_sf8', prepare)
    monkeypatch.setattr(rx, 'prepare_long', lambda *a: None)
    monkeypatch.setattr(rx, 'refine', lambda **kw: [])
    emit = tx._tx

    def transmit(audio, what):
        if tx.p4_tx_admit is not None:
            tx.p4_tx_admit()
        emit(audio, what)
        live.pos = tx.tx_end
        live.wall = max(live.wall, live.pos/FS)

    tx._tx = transmit
    # Rust consumes physical monotonic timestamps; this transcript charges a
    # declared clock. The shared replay clock runs real reference LONG grants.
    with declared_time(lambda: live.wall*FS):
        onair._run_p4_link(tx, live, onair._MasterGrid(
            0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    return tx.p4_probe, rx, machines[0], live, deadlines, cut


@pytest.mark.parametrize('peer', ['37.258', '41.008', '56.008'])
def test_recorded_timely_sf8_keys_with_capture_holdback(monkeypatch, peer):
    row = recorded_frame(peer)
    probe, rx, _, _, _, _ = replay(monkeypatch, row)
    decoded, = [e for e in rx.live_events if e['event'] == 'sf8-long']
    assert decoded['crc_valid'] and decoded['timely']
    keys = [e for e in probe.link_events if e['event'] == 'Keyed']
    assert [(e['at'], e['word'], e['current_slot_accepted']) for e in keys] == [
        (row.due, 'X27', True)]
    assert not [e for e in probe.link_events if e['event'] == 'Refused']


def test_genuinely_late_sf8_decode_is_not_accepted(monkeypatch):
    row = recorded_frame('37.258')
    probe, rx, _, _, deadlines, _ = replay(monkeypatch, row, completion='late-publication')
    decoded, = [e for e in rx.live_events if e['event'] == 'sf8-long']
    assert decoded['crc_valid'] and not decoded['timely']
    assert not [e for e in probe.link_events if e['event'] == 'Keyed'
                and e['current_slot_accepted']]
    engine, = [e for e in rx.live_events if e['event'] == 'long-engine']
    assert engine['crc_valid'] and not engine['timely']
    assert not [e for e in probe.link_events if e['event'] == 'source-accept']
    assert not deadlines[0]['limits'].publishable()


def test_key_stall_still_uses_real_deadline_refusal(monkeypatch):
    row = recorded_frame('37.258')
    probe, rx, _, _, _, _ = replay(monkeypatch, row, key_stall=True)
    assert any(e['event'] == 'sf8-long' and e['timely'] for e in rx.live_events)
    refused, = [e for e in probe.link_events if e['event'] == 'Refused']
    assert (refused['at'], refused['reason']) == (row.due, 'deadline')
    assert not [e for e in probe.link_events if e['event'] == 'Keyed']


@pytest.mark.parametrize('early_ms', [-4., 4.])
def test_early_estimate_cannot_extend_decode_deadline(monkeypatch, early_ms):
    row = recorded_frame('37.258')
    probe, rx, turn, live, deadlines, cut = replay(
        monkeypatch, row, early_ms=early_ms, full_holdback=True)
    grant, = deadlines
    limits = grant['limits']
    assert limits.work.sample_deadline == cut-round(.015*FS)
    assert limits.publication.sample_deadline == row.due-row.notice-round(.005*FS)
    assert limits.enqueue.sample_deadline == row.due-row.notice
    for name in ('work', 'publication', 'enqueue'):
        part = getattr(limits, name)
        assert part.wall_deadline <= grant[name] <= part.sample_deadline/FS
    # Rereading the clocks after the completed turn cannot renew a grant.
    limits.work_limit()
    limits.publishable()
    limits.keyable()
    for name in ('work', 'publication', 'enqueue'):
        assert getattr(limits, name).wall_deadline <= grant[name]
    assert grant['work'] < grant['publication'] < grant['enqueue']
    assert any(e['event'] == 'Keyed' and e['current_slot_accepted']
               for e in probe.link_events)
    obs, = live.decoded
    live.wall = (limits.publication.sample_deadline+1)/FS
    live.pos = limits.publication.sample_deadline+1
    assert not limits.publishable()
    assert turn.sf8_eligible(obs)
    assert turn.can_key(live.proposals[0])
    live.wall = (limits.enqueue.sample_deadline+1)/FS
    live.pos = limits.enqueue.sample_deadline+1
    assert not turn.sf8_eligible(obs)
    assert not turn.can_key(live.proposals[0])


def test_sf8_grant_overrunning_work_can_publish_before_publication_limit(monkeypatch):
    row = recorded_frame('37.258')
    probe, rx, _, _, deadlines, _ = replay(monkeypatch, row, completion='work-reserve')
    engine, = [e for e in rx.live_events if e['event'] == 'long-engine']
    assert engine['crc_valid'] and engine['timely'] and engine['budget_overrun_keyable']
    grant, = deadlines
    assert grant['work']+.001 < grant['publication'] < grant['enqueue']
    keys = [e for e in probe.link_events if e['event'] == 'Keyed']
    assert [(e['at'], e['word'], e['current_slot_accepted']) for e in keys] == [
        (row.due, 'X27', True)]
    assert not [e for e in probe.link_events if e['event'] == 'Refused']
