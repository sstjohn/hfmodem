# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""An unanswered caller keys every cycle through its budget, as an SCS PTC does."""
import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1
from hfmodem.tests.shrike.test_armdefaults import _parsed
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

SLOT_N = round(1.25 * onair.FS)


class _Clock(_Bench):
    def __init__(self):
        super().__init__(seconds=60)
        self.closed = False

    def close(self):
        self.closed = True


def session(tmp_path, monkeypatch, *argv, answer_from=None):
    """Calls keyed, as (label, first sample); the peer answers call `answer_from` on."""
    clock, keys, calls = _Clock(), [], 0
    monkeypatch.setattr(onair.ota, 'Rig', lambda *a, **kw: _Rig())
    monkeypatch.setattr(onair.ota, 'ptt_device', lambda *a, **kw: None)
    monkeypatch.setattr(onair, 'find_device', lambda *a, **kw: 0)
    monkeypatch.setattr(onair, '_LiveInput', lambda *a, **kw: clock)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    emit = onair.RadioTx._tx

    def transmit(tx, audio, what, *args, **kwargs):
        nonlocal calls
        before = len(tx.keyed)
        emit(tx, audio, what, *args, **kwargs)
        if len(tx.keyed) == before:
            return
        keys.append((what, tx.tx_audio_start))
        if what.startswith('connect->'):
            calls += 1
            if answer_from is not None and calls >= answer_from:
                cs = onair._trim_silence(pactor1.control_signal(
                    pactor1.CS_SPEED, invert=calls % 2)).astype(np.float32)
                at = tx.tx_end + round(.095 * onair.FS)
                clock.audio[at:at + cs.size] = cs

    monkeypatch.setattr(onair.RadioTx, '_tx', transmit)
    onair.run(_parsed('--transmit', '--serial', '/dev/null', '--dial', '7100000',
                      '--outdir', str(tmp_path / 'capture'), '--pactor1-only', '--hold', '2',
                      *argv))
    assert clock.closed
    return keys


def _calls(keys):
    return [at for what, at in keys if what.startswith('connect->')]


def test_an_unanswered_caller_keys_every_cycle_through_its_budget(
        tmp_path, monkeypatch, capsys):
    keys = session(tmp_path, monkeypatch, '--retries', '20', '--max-cycles', '32')
    calls = _calls(keys)
    assert len(calls) == 21
    assert set(np.diff(calls)) == {SLOT_N}
    out = capsys.readouterr().out
    calling = out.split('FINAL CALL LISTEN')[0]
    assert 'HUSHED' not in calling and 'OFF THE AIR' not in calling
    assert 'FINAL CALL LISTEN' in out


def test_call_hush_reproduces_the_four_and_six_cadence(tmp_path, monkeypatch, capsys):
    keys = session(tmp_path, monkeypatch, '--retries', '20', '--max-cycles', '32',
                   '--call-hush', '4,6')
    slots = [(at - _calls(keys)[0]) // SLOT_N for at in _calls(keys)]
    assert slots == [0, 1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32]
    assert 'OFF THE AIR, 6 cycle(s)' in capsys.readouterr().out


@pytest.mark.parametrize('first', [1, 5, 12, 18])
def test_an_answer_to_any_call_is_read(tmp_path, monkeypatch, capsys, first):
    keys = session(tmp_path, monkeypatch, '--retries', '20', '--max-cycles', '32',
                   answer_from=first)
    assert any(what.startswith('P1 pkt#1') for what, _ in keys)
    assert '** CONNECTED to' in capsys.readouterr().out
    assert set(np.diff(_calls(keys))) == {SLOT_N}


@pytest.mark.parametrize('text', ['4', '0,6', '4,x'])
def test_call_hush_takes_two_positive_counts(text):
    with pytest.raises(SystemExit):
        _parsed('--call-hush', text)
