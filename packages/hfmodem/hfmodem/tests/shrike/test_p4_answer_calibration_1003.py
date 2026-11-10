# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline chain calibration: real scheduler arithmetic, no card or RF claim."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import config
from hfmodem.core.rigs import RIGS, resolve_tx_timing
from hfmodem.shrike import onair, p4turn
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, Observation
from hfmodem.tests.shrike.test_p4_link_0927 import Tx
from hfmodem.tests.shrike.test_p4turn import ack_frame, machine, propose_ack
from hfmodem.tests.shrike.test_tx_latency_correction import _Stream, _key, SETTLE_N

FS = onair.FS
RESIDUAL = round(.0314 * FS)  # 1507 samples = 31.395833 ms; candidate only.
GRID = onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800)


def plan(tmp_path, residual=31.4, *, rig='ftdx10', evidence='1112 and 1129 witness; candidate'):
    path = tmp_path/'offline-chain.toml'
    path.write_text('schema=1\n[station]\ncontrol="local"\nregulatory="unregulated"\n'
                    'because="offline timing fixture"\n[audio.chains.ftdx10_usb]\n'
                    f'p4_answer_residual_ms={residual}\n'
                    f'p4_answer_residual_evidence="{evidence}"\n')
    return resolve_tx_timing(rig=rig, station_path=path, environ={})


def scoped(stream, timing, call):
    tx = SimpleNamespace(p4_link_options=SimpleNamespace(tx_timing_plan=timing,
        p4_answer_offset=.920), p4_probe=SimpleNamespace(link_events=[]))
    return onair._p4_answer_timing_scope(lambda *_: call())(tx, stream, GRID)


@pytest.mark.parametrize('rig', RIGS)
def test_unconfigured_chain_keeps_schedule(rig):
    timing = resolve_tx_timing(rig=rig, environ={})
    assert timing.p4_answer_residual_samples == 0
    stream = _Stream(tx_latency_n=timing.latency_samples)
    before = _key(stream, 100000)
    assert scoped(stream, timing, lambda: _key(stream, 100000)) == before


@pytest.mark.parametrize('rig', ['ft891', 'g90', 'x6100'])
def test_calibration_never_crosses_selected_chain(tmp_path, rig):
    assert plan(tmp_path, rig=rig).p4_answer_residual_samples == 0


@pytest.mark.parametrize('literal', ['nan', 'inf', '-61', '61', 'true', '"31.4"'])
def test_invalid_signed_calibration_is_refused(tmp_path, literal):
    with pytest.raises(config.ConfigError, match='p4_answer_residual_ms'):
        plan(tmp_path, literal)


def test_evidence_is_required_and_zero_restores_prior_value(tmp_path):
    with pytest.raises(config.ConfigError, match='evidence'):
        plan(tmp_path, evidence='')
    timing = plan(tmp_path, 0, evidence='')
    assert timing.p4_answer_residual_samples == 0
    assert timing.latency_samples == round(.007 * FS)


@pytest.mark.parametrize('offset', [.888, .910, .940])
def test_offset_workaround_cannot_be_combined(tmp_path, offset):
    args = SimpleNamespace(tx_timing_plan=plan(tmp_path), p4_answer_offset=offset,
                           p4_link=True)
    with pytest.raises(SystemExit, match='requires --p4-answer-offset 0.920'):
        onair._p4_entry_defaults(args)


@pytest.mark.parametrize('field', ['audio_in', 'audio_out'])
def test_endpoint_override_requires_its_own_calibration(tmp_path, field):
    args = SimpleNamespace(tx_timing_plan=plan(tmp_path), p4_answer_offset=.920,
                           **{field: 'another device'})
    with pytest.raises(SystemExit, match='endpoints'):
        onair._p4_answer_timing_defaults(args)


@pytest.mark.parametrize('residual_ms', [-31.4, 31.4])
@pytest.mark.parametrize('kind', [Kind.SHORT, Kind.BODY, Kind.NORMAL, Kind.LONG, Kind.CHIRP])
def test_all_answer_forms_share_one_scheduler_correction(tmp_path, residual_ms, kind):
    timing = plan(tmp_path, residual_ms)
    delta = timing.p4_answer_residual_samples
    stream = _Stream(tx_latency_n=timing.latency_samples)
    baseline_lead = onair._prekey_lead(stream, SETTLE_N)
    notice = stream.key_notice
    onset = 10 * FS
    due = onset + round((3.4 if kind in (Kind.LONG, Kind.CHIRP) else .920)*FS)
    original = _key(stream, due)

    def run():
        lead = onair._prekey_lead(stream, SETTLE_N)
        assert lead == baseline_lead + delta
        assert stream.key_notice == notice + delta
        assert lead-stream.key_notice == baseline_lead-notice
        m, _ = machine(role='irs', prekey_lead=lead/FS)
        frame = ack_frame(onset, b'offline', 0, kind)
        if kind == Kind.SHORT:
            frame = replace(frame, observation=replace(frame.observation, frame=None, value=13))
        proposal = propose_ack(m, frame)
        assert proposal is not None and proposal.stream_sample == due
        output, key = _key(stream, due)
        assert output == original[0] - delta
        assert key == pytest.approx(original[1] - delta/FS)
        assert stream._dac_time(output)-key == pytest.approx(SETTLE_N/FS)
        assert stream.corrected_emission_start == due
        assert m.keyed(stream.corrected_emission_start).stream_sample == due
    scoped(stream, timing, run)
    assert stream.tx_latency_n == timing.latency_samples
    assert _key(stream, due) == original


@pytest.mark.parametrize('overrun', [0, 1, RESIDUAL-1, RESIDUAL])
def test_enqueue_accepts_boundary_and_refuses_one_sample_late(tmp_path, overrun):
    timing = plan(tmp_path)
    stream = _Stream(tx_latency_n=timing.latency_samples)
    due = 100000
    baseline_notice = stream.key_notice
    stream.samples = due-baseline_notice-RESIDUAL+overrun
    assert stream.clamp_late(due) == 0  # Every case fitted the old schedule.
    def run():
        wave = np.zeros(128, np.float32)
        if overrun:
            with pytest.raises(onair._MissedTxSlot):
                stream.transmit(wave, at=due, settle=.04)
            assert stream._tx is None and not stream.slept
        else:
            assert stream.transmit(wave, at=due, settle=.04)[0] == due
    scoped(stream, timing, run)


@pytest.mark.parametrize('error', [RuntimeError('decode failure'), EOFError('capture ended')])
def test_exceptions_restore_existing_latency(tmp_path, error):
    stream = _Stream(tx_latency_n=336)
    def fail():
        assert stream.tx_latency_n == 336+RESIDUAL
        raise error
    with pytest.raises(type(error)):
        scoped(stream, plan(tmp_path), fail)
    assert stream.tx_latency_n == 336
    assert not hasattr(stream, 'p4_answer_residual_n')


# Exact published witness rows: subtract only the sample-quantized candidate.
# 1112 used .910: these are arithmetic pins, never a permitted combined trial.
@pytest.mark.parametrize('measured_ms,nominal_ms,expected_ms', [
    (941.30, 910., 909.9041666667), (941.41, 910., 910.0141666667),
    (941.44, 910., 910.0441666667), (3431.47, 3400., 3400.0741666667),
    (3431.33, 3400., 3399.9341666667), (3431.29, 3400., 3399.8941666667),
    (951.478, 920., 920.0821666667), (954.013, 920., 922.6171666667),
    (951.687, 920., 920.2911666667), (951.756, 920., 920.3601666667),
    (951.617, 920., 920.2211666667),
])
def test_1112_and_1129_witness_model_only(tmp_path, measured_ms, nominal_ms, expected_ms):
    correction = plan(tmp_path).p4_answer_residual_samples*1000/FS
    assert measured_ms-correction == pytest.approx(expected_ms)
    assert (measured_ms-nominal_ms)-correction == pytest.approx(expected_ms-nominal_ms)


class ClockedStream(_Stream):
    """Production enqueue arithmetic with synthetic capture and a finite clock."""
    def __init__(self):
        super().__init__(tx_latency_n=336)
        self.pos = self.samples = round(.9*FS)
        self.wall = self.pos
        self.outputs = []

    def sample_now(self):
        return max(self.pos, self.wall)

    def take_until(self, end):
        end = min(end, 6*FS)
        out = np.zeros(max(0, end-self.pos), np.float32)
        self.pos = self.samples = max(self.pos, end)
        return out

    def clamp_late(self, at):
        return max(0, self.sample_now()+self.key_notice-at)


def drive(monkeypatch, timing, kind, *, stall=0):
    stream = ClockedStream()
    tx = Tx(stream)
    tx.settle = .040
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.tx_timing_plan = timing
    tx.p4_link_options.p4_link_timeout = (1.3 if stall and kind not in (Kind.LONG, Kind.CHIRP) else 4.3)
    rx = tx.sessrx.p4
    long = kind in (Kind.LONG, Kind.CHIRP)
    rx.clock = Clock(Expect.PACKET, FS, .330, 3.75 if long else 1.25, 0.)
    obs = ack_frame(FS, b'offline', 0, kind).observation
    if kind == Kind.SHORT:
        obs = replace(obs, frame=None, value=13, walsh_peak=64)
    machines = []
    turn = p4turn.P4Turn
    def irs(*a, **kw):
        m = turn(*a, **{**kw, 'role': 'irs'})
        machines.append(m)
        return m
    monkeypatch.setattr(p4turn, 'P4Turn', irs)
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)
    def read(audio, origin, *, partial=False, **kw):
        if abs(rx.clock.next_due-FS) >= FS//10:
            return []
        ready = 3.28 if long and not partial else .800
        if origin+len(audio) < FS+round(ready*FS):
            return []
        records = []
        if kind == Kind.BODY:
            rx._emit(Observation(Kind.WORD, FS, 0, FS, FS, .3, 0., .9,
                                 word="Y27", score=.99), records)
        rx._emit(replace(obs, provisional=long and partial), records)
        if any(r.observation.frame and not r.observation.provisional for r in records):
            stream.wall = stream.sample_now()+round(stall*FS)
        return records
    monkeypatch.setattr(rx, 'read_partial', lambda *a, **kw: read(*a, partial=True, **kw))
    monkeypatch.setattr(rx, 'read_window', read)
    def emit(audio, what):
        due = tx.raster.boundary(tx.slot)
        tx.tx_audio_start, tx.tx_end = stream.transmit(audio, at=due, settle=tx.settle)
        stream.outputs.append((due, stream._tx_at, stream.slept[-2]))
        tx.slots_used.append(tx.slot)
        stream.pos = stream.samples = tx.tx_end
    tx._tx = emit
    onair._run_p4_link(tx, stream, GRID)
    return tx, stream, machines[0]


@pytest.mark.parametrize('kind', [Kind.SHORT, Kind.BODY, Kind.NORMAL, Kind.LONG, Kind.CHIRP])
def test_driver_nominal_and_local_clocks_and_earlier_decision(tmp_path, monkeypatch, kind):
    calibrated = plan(tmp_path)
    with monkeypatch.context() as patch:
        plain, base, _ = drive(patch, replace(calibrated, p4_answer_residual_samples=0,
                                            p4_answer_residual_ms=0), kind)
    with monkeypatch.context() as patch:
        tx, stream, _ = drive(patch, calibrated, kind)
    assert base.outputs and stream.outputs
    nominal = FS+round((3.4 if kind in (Kind.LONG, Kind.CHIRP) else .920)*FS)
    assert base.outputs[0][0] == stream.outputs[0][0] == nominal
    assert stream.outputs[0][1] == base.outputs[0][1]-RESIDUAL
    for entry in tx.p4_probe.emissions:
        assert entry['audio_start'] == entry['nominal_due']
        assert entry['local_due'] == entry['nominal_due']-RESIDUAL
    before = next(e for e in plain.p4_probe.link_events if e['event'] == 'Decision')
    after = next(e for e in tx.p4_probe.link_events if e['event'] == 'Decision')
    assert after['decision_at'] == before['decision_at']-RESIDUAL
    assert after['nominal_due'] == before['nominal_due']
    assert stream.tx_latency_n == 336


@pytest.mark.parametrize('kind', [Kind.BODY, Kind.LONG, Kind.CHIRP])
def test_driver_cost_overrun_does_not_key_late(tmp_path, monkeypatch, kind):
    tx, stream, m = drive(monkeypatch, plan(tmp_path), kind, stall=.150)
    assert not stream.outputs
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert [e for e in tx.p4_probe.link_events
            if e['event'] in ('Refused', 'answer-deferred')]
    assert m.acked_counter is None


def test_latency_override_cannot_spend_the_same_calibration_twice(tmp_path):
    timing = plan(tmp_path)
    path = tmp_path/'offline-chain.toml'
    changed = resolve_tx_timing(38.4, rig='ftdx10', station_path=path, environ={})
    # Other protocols can still select a TX latency; only P4 consumes this residual.
    with pytest.raises(SystemExit, match='station TX latency'):
        onair._p4_answer_timing_defaults(SimpleNamespace(tx_timing_plan=changed))
    same = resolve_tx_timing(7, rig='ftdx10', station_path=path, environ={})
    assert same.p4_answer_residual_samples == timing.p4_answer_residual_samples


@pytest.mark.parametrize('stage', ['before-render', 'after-drain', 'enqueue', 'capture-loss'])
def test_real_radio_tx_refuses_calibrated_deadline_or_capture_loss(tmp_path, stage):
    timing = plan(tmp_path)
    stream = _Stream(tx_latency_n=336)
    due = 100000
    radio = onair.RadioTx(SimpleNamespace(ptt=lambda _: True, key_failure=lambda: None),
                          transmit=True, outdir=tmp_path, settle=.04)
    radio.live = stream
    grid = onair._MasterGrid(due, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800)
    radio.aim(grid, 0)
    original = stream.transmit
    def inject():
        # A single sample after notice is inside the legacy 5-ms forgiveness.
        stream.samples = due-stream.key_notice+1
    if stage == 'after-drain':
        stream.wait_until = lambda _: inject()
    elif stage == 'enqueue':
        def late_enqueue(*a, **kw):
            inject()
            return original(*a, **kw)
        stream.transmit = late_enqueue
    elif stage == 'capture-loss':
        stream.lost, stream.xruns, stream.lost_at = 344, 0, []
    def attempt():
        if stage == 'before-render':
            inject()
        radio._tx(np.ones(10080, np.float32), 'offline calibrated answer')
    scoped(stream, timing, attempt)
    assert radio.refused
    assert radio.slots_used == []
    assert stream._tx is None and not stream.slept
    assert radio.slot == 0  # No silent move to the next raster.
    assert stream.tx_latency_n == 336


def test_file_replay_retains_nominal_audio_and_local_ptt(tmp_path):
    timing = plan(tmp_path)
    stream = onair._ReplayInput.__new__(onair._ReplayInput)
    stream.fs, stream.pos = FS, 0
    stream.audio = np.zeros(3*FS, np.float32)
    stream.realtime = False
    stream._bridged = np.empty(0, np.float32)
    stream.lost = stream.xruns = 0
    radio = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    radio.live = stream
    due = 2*FS
    radio.aim(onair._MasterGrid(due, 60000, 0, packet_n=36000,
                               cs_n=12000, d_max_n=4800), 0)
    scoped(stream, timing, lambda: radio._tx(np.ones(10080, np.float32), 'offline replay'))
    assert not radio.refused
    assert radio.tx_audio_start == due
    assert radio.tx_key_up == due-round(.04*FS)-RESIDUAL
    assert radio.tx_end == due+10080
    assert not hasattr(stream, 'tx_latency_n')


def test_nested_scope_cannot_apply_residual_twice(tmp_path):
    timing = plan(tmp_path)
    stream = _Stream(tx_latency_n=336)
    with pytest.raises(RuntimeError, match='already active'):
        scoped(stream, timing, lambda: scoped(stream, timing, lambda: None))
    assert stream.tx_latency_n == 336
    assert not hasattr(stream, 'p4_answer_residual_n')
