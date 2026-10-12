# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The 1103 splice rejects stale P4 entry timing without losing its query."""
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.shrike import onair, session
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_p4_link_0927 import Tx

FS = onair.FS
CASE_PATH = Path(__file__).parent / 'fixtures/p4-entry-loss-1003.json'
if not CASE_PATH.is_file():
    pytest.skip('private P4 entry recording metadata is absent from this distribution', allow_module_level=True)
CASE = json.loads(CASE_PATH.read_text())


@pytest.fixture(scope='module')
def recorded():
    paths = [evidence.require_input(evidence.RECORD / name) for name in CASE['files']]
    for path, digest in zip(paths, CASE['files'].values()):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return session.load_wav(str(paths[0]), FS)


def station(recorded, tmp_path, monkeypatch, mode):
    monkeypatch.setattr(onair, '_load', lambda _: recorded.copy())
    live = onair._ReplayInput('recorded-1103')
    live.pos = 10 * FS
    live.xruns, live.lost, live.lost_at = 1, 0, []
    seed = Tx(live)
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.live, tx.sessrx = live, seed.sessrx
    tx.p4_probe = seed.p4_probe
    probe = tx.p4_probe
    probe.emissions.clear()
    probe.grant_sample = 10 * FS
    probe.timeout = 20.
    probe.packet, probe.cadence = 'robust', 'consecutive'
    probe.values, probe.headers = (7, 38), ('A', 'A')
    tx.p4_link_options = seed.p4_link_options
    tx.sessrx.host = seed.p4_link_host
    tx.qrm_guard = False  # Recorded operator setting, never changed by recovery.
    grid = onair._MasterGrid(CASE['anchor'], 60000, 0, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    tx.aim(grid, 9)
    drains, attempts, handoffs = [], [], []
    take, key = live.take_until, tx._tx

    def loss():
        live.lost += CASE['lost_samples']
        live.lost_at.append((CASE['loss_sample'] if not live.lost_at else live.pos,
                             CASE['lost_samples'], .004))
        live.xruns += 2

    def drain(until):
        audio = take(until)
        if mode == 'after' and not probe.emitting and len(probe.emissions) == 2 and not live.lost:
            loss()
        if probe.emitting:
            drains.append((tx.slot, live.pos))
            if mode == 'repeat' or (mode in ('drain', 'deadline') and len(drains) == 1) or (
                    mode == 'second' and len(probe.emissions) == 1 and live.lost == 0):
                loss()
                if mode == 'deadline':
                    probe.timeout = (live.pos - probe.grant_sample) / FS
            elif mode == 'xrun':
                live.xruns += 1
        return audio

    def attempt(audio, what, *args, **kwargs):
        before = len(tx.slots_used)
        key(audio, what, *args, **kwargs)
        attempts.append(dict(label=what, sent=len(tx.slots_used) > before,
                             cause=getattr(tx, 'refusal_cause', None), slot=tx.slot,
                             boundary=tx.boundary, pos=live.pos))

    def handoff(tx, live, raster):
        assert probe.emissions or getattr(probe, 'peer_breakin', False)
        handoffs.append(list(probe.emissions))
        return 'keyed entry handed off'

    monkeypatch.setattr(live, 'take_until', drain)
    monkeypatch.setattr(tx, '_tx', attempt)  # Instrument real guard, never return a canned refusal.
    monkeypatch.setattr(onair, '_run_p4_link', handoff)
    if mode == 'before':
        loss()
    if mode == 'qrm':
        tx.qrm_guard = True
        grid.note_peer_codeword(grid.boundary(9)+5000, 5760, 'CS1', 'WS8EOC',
                                protocol=onair.Protocol.PACTOR1)
    return tx, live, grid, attempts, handoffs, drains


@pytest.mark.parametrize('mode', ['drain', 'before', 'xrun', 'none', 'second', 'after'])
def test_recorded_pending_query_survives_only_real_loss(recorded, tmp_path, monkeypatch, mode):
    tx, live, grid, attempts, handoffs, drains = station(recorded, tmp_path, monkeypatch, mode)
    result = onair._run_p4_probe(tx, live, grid, in_slot=True)
    assert result == 'keyed entry handed off'
    assert [e['value'] for e in tx.p4_probe.emissions] == [7, 38]
    assert len(handoffs) == 1 and len(handoffs[0]) == 2
    failed = [a for a in attempts if not a['sent']]
    assert len(failed) == (1 if mode in ('drain', 'second') else 0)
    assert all(a['cause'] == 'input-loss' for a in failed)
    assert all(e['audio_start'] == grid.boundary(e['slot']) for e in tx.p4_probe.emissions
               if mode != 'second' or e['value'] == 38)
    if mode in ('drain', 'second'):
        assert [a['label'].split(' v=')[1].split()[0] for a in attempts] == (
            ['7', '7', '38'] if mode == 'drain' else ['7', '38', '38'])
        assert attempts[1 if mode == 'drain' else 2]['slot'] > failed[0]['slot']
        assert any(e['event'] == 'entry P1 control' and e['word'] == 'CS1'
                   for e in tx.p4_probe.link_events)
    assert grid.anchor == CASE['anchor'] - (344 if mode in ('drain', 'before', 'second') else 0)
    assert live.pos <= tx.p4_probe.deadline(FS)
    if mode == 'after':
        assert live.lost == 344
        assert tx._observe_input_loss('after committed opening')
        assert len(tx.p4_probe.emissions) == 2 and len(attempts) == 2


@pytest.mark.parametrize('mode', ['repeat', 'deadline', 'qrm'])
def test_unemitted_entry_never_becomes_p4_only(recorded, tmp_path, monkeypatch, mode):
    tx, live, grid, attempts, handoffs, drains = station(recorded, tmp_path, monkeypatch, mode)
    arq = tx.sessrx.host.arq
    prior_role, prior_state = arq.role, arq.state
    result = onair._run_p4_probe(tx, live, grid, in_slot=True)
    assert (arq.role, arq.state) == (prior_role, prior_state)
    assert not tx.p4_probe.emissions and not handoffs
    assert result.startswith('P4 entry not emitted') and 'peer silent' not in result
    assert len(attempts) == (onair.REGRID_TRIES if mode == 'repeat' else 1)
    assert not any(a['sent'] for a in attempts)
    if mode != 'qrm':
        assert 'capture-loss refusal' in result and 'last peer CS1' in result
        if mode == 'repeat':
            times = [e['at']/FS for e in tx.p4_probe.link_events
                     if e['event'] == 'entry P1 control']
            assert all(any(abs(at-want) < .015 for at in times)
                       for want in (11.116, 12.363, 13.617, 14.865))
    else:
        assert 'refused' in result and tx.qrm_guard
        assert attempts[0]['cause'] is None


def test_recorded_p1_phase_jump_recovers_at_production_final_drain(
        recorded, tmp_path, monkeypatch):
    """Replay acquisition and the real shift jump, not a fabricated P4 peer."""
    import contextlib
    import io
    import shlex
    import sys

    log_name = next(n for n in CASE['files'] if n.endswith('.log'))
    log = (evidence.RECORD / log_name).read_text()
    argv = shlex.split(next(s.strip()[6:] for s in log.splitlines()
                           if s.strip().startswith('argv: ')))
    omit = {'--serial', '--ptt-port', '--audio-out', '--audio-in', '--outdir',
            '--mail-out', '--mail-send'}
    filtered, i = [], 0
    while i < len(argv):
        if argv[i] in omit:
            i += 2
            continue
        if argv[i] != '--transmit':
            filtered.append(argv[i])
        i += 1
    wave = tmp_path / 'aligned.wav'
    session.write_wav(str(wave), recorded[CASE['anchor']:])
    monkeypatch.setattr(sys, 'argv', ['onair', *filtered, '--replay', str(wave),
        '--outdir', str(tmp_path/'out'), '--mail-out', str(tmp_path/'mail')])
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **k: None)
    monkeypatch.setattr(onair, '_LiveInput', lambda *a, **k: pytest.fail('device opened'))
    key, take = onair.RadioTx._tx, onair._ReplayInput.take_until
    active, attempts, handoffs, injections = [], [], [], []

    def drain(live, until):
        audio = take(live, until)
        tx = active[-1] if active else None
        if tx is not None and tx.p4_probe.emitting and not injections:
            at = CASE['loss_sample']-CASE['anchor']
            assert live.pos >= at
            live.lost, live.xruns = 344, 3
            live.lost_at = [(at, 344, .004)]
            injections.append((at, live.pos, tx.boundary))
        return audio

    def emit(tx, audio, what, *args, **kwargs):
        active.append(tx)
        before = len(tx.slots_used)
        try:
            key(tx, audio, what, *args, **kwargs)
        finally:
            active.pop()
        if what.startswith('P4 PROBE'):
            attempts.append((what, len(tx.slots_used)>before, getattr(tx, 'refusal_cause', None), tx.slot))

    def handoff(tx, live, raster):
        assert [e['value'] for e in tx.p4_probe.emissions] == [7, 38]
        assert all(e['audio_start'] == raster.boundary(e['slot'])
                   for e in tx.p4_probe.emissions)
        handoffs.append(tx.p4_probe)
        return 'recorded entry restored'

    monkeypatch.setattr(onair._ReplayInput, 'take_until', drain)
    monkeypatch.setattr(onair.RadioTx, '_tx', emit)
    monkeypatch.setattr(onair, '_run_p4_link', handoff)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        onair.main()
    # Never put the raw arm transcript/arguments into assertion diagnostics.
    assert 'phase of the cycle before' in output.getvalue()
    assert len(injections) == 1 and injections[0][0] == 479694
    assert [sent for _, sent, _, _ in attempts] == [False, True, True]
    assert attempts[0][2] == 'input-loss'
    assert attempts[1][3] > attempts[0][3]
    assert len(handoffs) == 1
    controls = [e for e in handoffs[0].link_events if e['event'] == 'entry P1 control']
    assert controls and all(e['word'] == 'CS1' for e in controls)
