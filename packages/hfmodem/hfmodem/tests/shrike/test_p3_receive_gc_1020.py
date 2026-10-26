# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import contextlib
import gc
from types import SimpleNamespace
import weakref

import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p4probe import EntryProbe
from hfmodem.tests.shrike.test_call_cadence import session


@pytest.fixture
def gc_policy():
    enabled = gc.isenabled()
    gc.enable()
    try:
        yield
    finally:
        if not enabled:
            gc.disable()


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_receive_gc_restores_policy(monkeypatch, gc_policy, enabled, failure):
    collections = []
    monkeypatch.setattr(gc, 'collect', lambda *args: collections.append(args))
    policy = contextlib.nullcontext() if enabled else onair._receive_gc()
    with policy:
        try:
            with onair._receive_gc() as incoming:
                assert incoming is enabled
                assert not gc.isenabled()
                if failure:
                    raise ValueError('receive failed')
        except ValueError:
            assert failure
        assert gc.isenabled() is enabled
    assert gc.isenabled()
    assert not collections


def test_receive_gc_nesting(gc_policy):
    with onair._receive_gc() as outer:
        assert outer and not gc.isenabled()
        with onair._receive_gc() as inner:
            assert not inner and not gc.isenabled()
        assert not gc.isenabled()
    assert gc.isenabled()


class Cycle:
    def __init__(self):
        self.cycle = self


@pytest.fixture
def loop_trace(monkeypatch, gc_policy):
    receive_gc, collect = onair._receive_gc, gc.collect
    trace = SimpleNamespace(scopes=[], active=None, last=None, policy=receive_gc,
                            interrupt=None, pending=None, handoff=None, lives=[])

    @contextlib.contextmanager
    def receiving():
        scope = dict(steps=[], refs=[], tag=None, collected=False, normal=False)
        trace.scopes.append(scope)
        assert trace.active is None
        try:
            with receive_gc() as enabled:
                scope['enabled'] = enabled
                trace.active = scope
                try:
                    yield enabled
                    scope['normal'] = True
                finally:
                    assert not gc.isenabled()
                    assert all(ref() is not None for ref in scope['refs'])
                    trace.active = None
                    trace.last = scope
        finally:
            assert gc.isenabled() is scope['enabled']
            scope['steps'].append('exit')

    def collecting(generation):
        assert generation == 2 and gc.isenabled()
        assert trace.active is None
        scope = trace.last
        assert scope['steps'][-1] == 'exit'
        assert not scope['collected']
        result = collect(generation)
        assert all(ref() is None for ref in scope['refs'])
        scope['steps'].append('collect')
        scope['collected'] = True
        return result

    new_cycle = onair._SessionRx.new_cycle
    def cycling(rx):
        scope = trace.active
        assert scope is not None and not gc.isenabled()
        scope['tag'] = rx.tag
        scope['steps'].append('cycle')
        garbage = Cycle()
        scope['refs'].append(weakref.ref(garbage))
        if trace.pending is not None:
            trace.pending.pending = False
        return new_cycle(rx)

    listener = onair._listen_until_answer
    def listening(live, want, host, rx, *args, **kwargs):
        assert trace.active is not None and not gc.isenabled()
        trace.active['steps'].append('listen')
        trace.lives.append(live)
        if trace.interrupt is not None:
            trace.interrupt(rx)
        return listener(live, want, host, rx, *args, **kwargs)

    scan = onair._scan_frame
    def scanning(*args, **kwargs):
        if trace.active is not None:
            assert not gc.isenabled()
            trace.active['steps'].append('scan')
        else:
            assert gc.isenabled() is trace.last['enabled']
            assert not trace.last['enabled'] or trace.last['collected']
        return scan(*args, **kwargs)

    regrid = onair._regrid
    def regridding(live, raster, tx, host, rx, *args, **kwargs):
        assert trace.active is not None and not gc.isenabled()
        trace.active['steps'].append('regrid')
        result = regrid(live, raster, tx, host, rx, *args, **kwargs)
        if trace.handoff is not None:
            trace.handoff(tx, rx)
        return result

    emit = onair.RadioTx.emit_pending_cs
    def emitting(tx):
        assert trace.active is not None and not gc.isenabled()
        trace.active['steps'].append('emit')
        return emit(tx)

    monkeypatch.setattr(onair, '_receive_gc', receiving)
    monkeypatch.setattr(gc, 'collect', collecting)
    monkeypatch.setattr(onair._SessionRx, 'new_cycle', cycling)
    monkeypatch.setattr(onair, '_listen_until_answer', listening)
    monkeypatch.setattr(onair, '_scan_frame', scanning)
    monkeypatch.setattr(onair, '_regrid', regridding)
    monkeypatch.setattr(onair.RadioTx, 'emit_pending_cs', emitting)
    return trace


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('hush', [False, True])
@pytest.mark.parametrize('disabled', [False, True])
def test_run_receive_gc_cycles(tmp_path, monkeypatch, loop_trace, held, hush, disabled):
    trace = loop_trace
    policy = trace.policy() if disabled else contextlib.nullcontext()
    with policy:
        session(tmp_path, monkeypatch, '--max-cycles', '4', '--retries', '20',
                *(['--listen-every', '1'] if held and hush else
                  ['--call-hush', '1,1'] if hush else []),
                answer_from=1 if held else None)
        assert gc.isenabled() is not disabled
    scopes = [s for s in trace.scopes if (s['tag'] == 'HOLD RX') is held]
    assert len(scopes) >= 2
    assert all(s['normal'] and s['collected'] is not disabled for s in scopes)
    assert all(s['steps'].index('listen') < s['steps'].index('exit') for s in scopes)
    assert any('scan' in s['steps'] for s in scopes)
    if hush:
        assert any('regrid' not in s['steps'] for s in scopes)
    else:
        keyed = [s for s in scopes if 'emit' in s['steps']]
        assert keyed
        assert all(s['steps'].index('regrid') < s['steps'].index('emit')
                   < s['steps'].index('exit') for s in keyed)
    if not disabled:
        assert all(s['steps'][-2:] == ['exit', 'collect'] for s in scopes)


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('error', [ValueError('receive failed'), KeyboardInterrupt()])
def test_run_receive_exception_restores_gc(tmp_path, monkeypatch, loop_trace, held, error):
    trace = loop_trace
    def interrupt(rx):
        if (rx.tag == 'HOLD RX') is held:
            raise error
    trace.interrupt = interrupt
    with pytest.raises(type(error)) as caught:
        session(tmp_path, monkeypatch, '--max-cycles', '4', answer_from=1 if held else None)
    assert caught.value is error
    assert gc.isenabled() and trace.active is None
    assert trace.lives[-1].closed
    assert not trace.last['normal'] and not trace.last['collected']
    assert trace.last['steps'][-1] == 'exit'
    assert all(s['collected'] for s in trace.scopes[:-1])


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('handoff', [False, True])
def test_run_receive_early_exit(tmp_path, monkeypatch, loop_trace, held, handoff):
    trace = loop_trace
    interrupted, handed = [], []
    def early(tx, rx):
        if interrupted or (rx.tag == 'HOLD RX') is not held:
            return
        interrupted.append(trace.active)
        if handoff:
            tx.p4_link_options = SimpleNamespace()
            tx.p4_probe = EntryProbe(requested=True)
            tx.p4_probe.peer_breakin = True
            tx.p4_probe.handoff_slot = tx.slot
        else:
            trace.pending = tx.p4_hold_audio = SimpleNamespace(pending=True)
    def separate_runner(*args, **kwargs):
        assert gc.isenabled() and trace.active is None
        assert not trace.last['collected']
        handed.append(True)
        return 'handoff complete'
    trace.handoff = early
    monkeypatch.setattr(onair, '_run_p4_probe', separate_runner)
    session(tmp_path, monkeypatch, '--max-cycles', '4', answer_from=1 if held else None)
    scope, = interrupted
    assert gc.isenabled() and trace.active is None
    assert 'emit' not in scope['steps'] and not scope['collected']
    assert handed == ([True] if handoff else [])
    if not handoff:
        assert trace.scopes.index(scope) < len(trace.scopes) - 1
        assert trace.scopes[-1]['collected']


@pytest.mark.parametrize('word', [False, True])
def test_run_receive_first_ack_handoff(tmp_path, monkeypatch, loop_trace, word):
    trace = loop_trace
    interrupted, handed, words = [], [], []
    read = onair._read_held_window
    def reading(live, raster, tx, rx, *args, **kwargs):
        result = read(live, raster, tx, rx, *args, **kwargs)
        assert trace.active is not None and not gc.isenabled()
        interrupted.append(trace.active)
        rx.host.p4_after_first_ack = True
        rx.host.p1_stalled = False
        tx.p4_link_options = SimpleNamespace()
        tx.p4_probe = EntryProbe(requested=True)
        return result
    def hold_word(tx, *args):
        assert trace.active is interrupted[-1] and not gc.isenabled()
        words.append(tx.slot)
        return tx.slot if word else None
    def separate_runner(*args, **kwargs):
        assert gc.isenabled() and trace.active is None
        assert not trace.last['collected']
        assert kwargs == (dict(heard=words[-1]) if word else {})
        handed.append(True)
        return 'handoff complete'
    monkeypatch.setattr(onair, '_read_held_window', reading)
    monkeypatch.setattr(onair, '_p4_hold_word', hold_word)
    monkeypatch.setattr(onair, '_run_p4_probe', separate_runner)
    session(tmp_path, monkeypatch, '--max-cycles', '4', answer_from=1)
    scope, = interrupted
    assert gc.isenabled() and trace.active is None
    assert 'emit' not in scope['steps'] and not scope['collected']
    assert len(words) == 1 and handed == [True]
