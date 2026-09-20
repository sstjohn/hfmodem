# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The last connect gets a decoded receive interval before identification."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import onair, pactor1
from hfmodem.tests.shrike.test_armdefaults import _parsed
from hfmodem.tests.shrike.test_grid import _Rig
from hfmodem.tests.shrike.test_identify import Clock

LATE_CS1 = Path(__file__).with_name('fixtures') / 'ws8eoc-0913-late-cs1.wav'


def session(tmp_path, monkeypatch, *, replies=0, retries=20, cycles=2,
            interrupt_tail=None, reply_delay=0, reply_audio=None):
    clock, rig, keys = Clock(), _Rig(), []
    monkeypatch.setattr(onair.ota, 'Rig', lambda *a, **kw: rig)
    monkeypatch.setattr(onair, 'find_device', lambda *a, **kw: 0)
    monkeypatch.setattr(onair, '_LiveInput', lambda *a, **kw: clock)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    emit = onair.RadioTx._tx
    calls = 0

    def transmit(tx, audio, what, *args, **kwargs):
        nonlocal calls
        before = len(tx.keyed)
        emit(tx, audio, what, *args, **kwargs)
        if len(tx.keyed) == before:
            return
        keys.append((what, tx.tx_audio_start, tx.tx_end))
        if what.startswith('connect->'):
            calls += 1
            # The last permitted call is followed by a peer answering on its
            # unchanged raster. Nothing answers any of the earlier calls.
            if calls == min(cycles, retries) + 1:
                for k in range(replies):
                    cs = onair._trim_silence(pactor1.control_signal(
                        pactor1.CS_SPEED, invert=k % 2)).astype(np.float32)
                    if reply_audio is not None:
                        cs = reply_audio
                    at = tx.tx_end + round((.095 + (k + reply_delay) * 1.25) * onair.FS)
                    clock.audio[at:at + len(cs)] = cs

    monkeypatch.setattr(onair.RadioTx, '_tx', transmit)
    if interrupt_tail is not None:
        listen = onair._listen_until_answer

        def interrupted(*args, **kwargs):
            if calls == min(cycles, retries) + 1:
                raise interrupt_tail
            return listen(*args, **kwargs)

        monkeypatch.setattr(onair, '_listen_until_answer', interrupted)
    args = _parsed('--transmit', '--serial', '/dev/null', '--dial', '7100000',
                   '--outdir', str(tmp_path), '--pactor1-only', '--hold', '2',
                   '--max-cycles', str(cycles), '--retries', str(retries))
    if isinstance(interrupt_tail, SystemExit):
        with pytest.raises(SystemExit, match='SIGTERM'):
            onair.run(args)
    else:
        onair.run(args)
    assert clock.closed
    return keys


@pytest.mark.parametrize('cycles,retries', [(2, 20), (20, 2)])
def test_unanswered_budget_waits_two_full_cycles_without_more_calls(
        tmp_path, monkeypatch, capsys, cycles, retries):
    keys = session(tmp_path, monkeypatch, cycles=cycles, retries=retries)
    calls = [k for k in keys if k[0].startswith('connect->')]
    ids = [k for k in keys if k[0].startswith('ID ')]
    assert len(calls) == 3 and len(ids) == 1
    gap = (ids[0][1] - calls[-1][2]) / onair.FS
    assert 2.5 <= gap < 5
    assert keys == calls + ids
    assert 'FINAL CALL LISTEN' in capsys.readouterr().out


@pytest.mark.parametrize('cycles,retries', [(2, 20), (20, 2)])
def test_late_peer_connects_and_gets_data_instead_of_cw(
        tmp_path, monkeypatch, capsys, cycles, retries):
    keys = session(tmp_path, monkeypatch, replies=5, cycles=cycles, retries=retries)
    assert len([k for k in keys if k[0].startswith('connect->')]) == 3
    first_data = next(k for k in keys if k[0].startswith('P1 pkt#1'))
    assert (first_data[1] - keys[0][1]) % round(1.25 * onair.FS) == 0
    assert not any(k[0].startswith('ID ') for k in keys)
    assert '** CONNECTED to' in capsys.readouterr().out


def test_one_unconfirmed_late_codeword_does_not_connect(tmp_path, monkeypatch, capsys):
    keys = session(tmp_path, monkeypatch, replies=1)
    assert any(k[0].startswith('ID ') for k in keys)
    assert not any(k[0].startswith('P1 pkt#') for k in keys)
    assert '** CONNECTED to' not in capsys.readouterr().out


@pytest.mark.parametrize('delay', [1, 2])
def test_candidate_near_tail_deadline_gets_time_to_confirm(
        tmp_path, monkeypatch, capsys, delay):
    keys = session(tmp_path, monkeypatch, replies=5, reply_delay=delay)
    assert any(k[0].startswith('P1 pkt#1') for k in keys)
    assert not any(k[0].startswith('ID ') for k in keys)
    trace = capsys.readouterr().out
    assert 'exact candidate extends listening' in trace
    assert '** CONNECTED to' in trace


@pytest.mark.skipif(not LATE_CS1.exists(),
                    reason=f'the recorded late CS1 {LATE_CS1.name} is not '
                           'installed')
def test_recorded_late_cs1_can_corroborate_in_the_extended_listen(
        tmp_path, monkeypatch, capsys):
    fs, pcm = wavfile.read(LATE_CS1)
    assert fs == onair.FS
    meta = json.loads(LATE_CS1.with_suffix('.json').read_text())
    assert meta['source_cycle'] == 34
    # Only the first candidate is this arm's actual evidence. Repeat its crop
    # on later slots to test recovery had the peer kept replying cleanly;
    # these continuations are explicitly constructed, not additional RF crops.
    keys = session(tmp_path, monkeypatch, replies=5, reply_delay=1,
                   reply_audio=pcm.astype(np.float32) / 32768)
    assert any(k[0].startswith('P1 pkt#1') for k in keys)
    assert not any(k[0].startswith('ID ') for k in keys)
    assert 'exact candidate extends listening' in capsys.readouterr().out


def test_candidate_extensions_have_an_absolute_ceiling():
    tail = onair._ConnectTail(1000, 60000)
    initial = tail.end
    assert tail.candidate(initial - 1)
    extended = tail.end
    assert not tail.candidate(initial - 1)  # Same observation buys no more time.
    for now in range(extended, tail.limit + 600001, 60000):
        tail.candidate(now)
        assert tail.end <= 1000 + 8 * 60000
    assert tail.end == tail.limit


@pytest.mark.parametrize('ending', [KeyboardInterrupt(), SystemExit('SIGTERM')])
def test_interrupt_during_final_listen_closes_without_cw(tmp_path, monkeypatch, ending):
    keys = session(tmp_path, monkeypatch, interrupt_tail=ending)
    assert len(keys) == 3
    assert all(k[0].startswith('connect->') for k in keys)
