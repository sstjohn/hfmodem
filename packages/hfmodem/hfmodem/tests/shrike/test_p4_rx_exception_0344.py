# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
import copy
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hfmodem.shrike import onair, p4shorttx, session
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, Kind, P4Receiver, Role
from hfmodem.tests.evidence import CAPTURES

FS = 48000


def receiver(next_due=1456742):
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.WORD, next_due, .935, 1.25, -4.524312774980677)
    rx.last_emission = Emission(Kind.BODY, Role.ISS, 1291467, .797625, True)
    rx.emissions = [rx.last_emission]
    rx.peer_onset = 1336742
    rx.peer_normal_pending = True
    return rx


@pytest.mark.parametrize('partial', [False, True])
def test_reconstructed_0344_partial_read_window(partial):
    path = CAPTURES / 'onair-1001-0344/stream.flac'
    if not path.exists():
        pytest.skip('local reconstructed 0344 recording absent')
    end = 1435622
    origin = end - 6 * FS
    audio = session.load_wav(str(path), FS).astype(np.float32)[origin:end]
    assert len(audio) == 288000 and np.isfinite(audio).all()
    rx = receiver()
    clock = vars(rx.clock).copy()
    assert max(0, rx.clock.next_due - round(.15 * FS) - origin) == 301920
    if partial:
        assert rx.read_partial(audio, origin, deadline=end) == []
    else:
        assert rx._reply_carrier(audio, origin) is False
    assert rx.observations == []
    assert vars(rx.clock) == clock


@pytest.mark.parametrize('length,begin', [(0, 0), (100, 101), (100, 100), (100, 99)])
def test_reply_carrier_search_boundary(length, begin):
    origin = 100000
    rx = receiver(origin + begin + round(.15 * FS))
    clock = vars(rx.clock).copy()
    assert rx._reply_carrier(np.zeros(length), origin) is False
    assert rx.observations == []
    assert vars(rx.clock) == clock


def test_reply_carrier_retains_valid_normal_detection():
    rx = receiver(12000)
    rx.clock.cfo_hz = 0
    audio = np.r_[np.zeros(12000), p4shorttx.short_packet(b'control', 0)]
    assert rx._reply_carrier(audio, 0) is True


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_reply_carrier_rejects_nonfinite_nonempty_search(value):
    rx = receiver(round(.15 * FS))
    with pytest.raises(ValueError, match='nonempty finite one-dimensional signal'):
        rx._reply_carrier(np.full(100, value), 0)


def run_reporting(read_partial, cleanup, tx, **scope):
    tree = ast.parse(inspect.getsource(onair._drive_session))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    outer = next(node for node in run.body if isinstance(node, ast.Try)
                 and any(isinstance(handler.type, ast.Name)
                         and handler.type.id == 'PttError' for handler in node.handlers))
    initial = next(node for node in run.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'ended'
                           for target in node.targets))
    summary = next(node for node in ast.walk(outer) if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Compare)
                   and ast.unparse(node.test) == 'tx.p4_probe is None'
                   and any(isinstance(child, ast.Constant)
                           and child.value == 'session ended: '
                           for child in ast.walk(node)))
    outer = copy.deepcopy(outer)
    outer.body = ast.parse('receiver.read_partial(audio, origin, deadline=end)').body
    outer.finalbody = ast.parse('cleanup()').body + copy.deepcopy(summary.orelse)
    module = ast.Module(body=[copy.deepcopy(initial), outer], type_ignores=[])
    namespace = dict(receiver=SimpleNamespace(read_partial=read_partial),
                     audio=np.zeros(1), origin=0, end=1, cleanup=cleanup,
                     tx=tx, linked=False, args=SimpleNamespace(mycall='W9SSJ', p4_link=True,
                                                              cwid_optional=False),
                     sessrx=SimpleNamespace(cs_log=[]), captures='captures',
                     code=0, PttError=onair.PttError)
    namespace.update(scope)
    from hfmodem.core import identification
    namespace.setdefault('identity', identification.resolve(
        namespace['args'].mycall, getattr(namespace['args'], 'idcall', None), environ={}))
    exec(compile(ast.fix_missing_locations(module), onair.__file__, 'exec'), namespace)
    return namespace


def test_receive_exception_verdict_propagates_and_cleans_up(capsys):
    error = ValueError('expected a nonempty finite one-dimensional signal')

    def read_partial(*args, **kwargs):
        raise error

    cleanup = Mock()
    tx = SimpleNamespace(identify=Mock())
    with pytest.raises(ValueError) as caught:
        run_reporting(read_partial, cleanup, tx)
    assert caught.value is error
    assert any(frame.name == 'read_partial' for frame in caught.traceback)
    cleanup.assert_called_once_with()
    tx.identify.assert_not_called()
    output = capsys.readouterr().out
    verdict = 'SESSION FAILED -- ValueError: expected a nonempty finite one-dimensional signal'
    assert f'session ended: {verdict};' in output
    assert f'verdict: P4 link; {verdict}' in output
    assert 'cycle budget ran out' not in output


@pytest.mark.parametrize('error,ended,code,keys', [
    (None, 'the cycle budget ran out', 0, 1),
    (KeyboardInterrupt(), 'the operator interrupted it', 0, 0),
    (onair.PttError('pin failure'), 'KEYING FAILED -- pin failure', 1, 0),
])
def test_reporting_controls(error, ended, code, keys, capsys):
    cleanup = Mock()
    tx = SimpleNamespace(identify=Mock())
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run_reporting(Mock(side_effect=error), cleanup, tx)
    else:
        namespace = run_reporting(Mock(side_effect=error), cleanup, tx)
        assert namespace['code'] == code
        assert namespace['ended'] == ended
    cleanup.assert_called_once_with()
    assert tx.identify.call_count == keys
    assert f'verdict: P4 link; {ended}' in capsys.readouterr().out
