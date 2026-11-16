# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC 11-14 05:18Z and 05:50Z: v6/A parity from the answered call."""
import contextlib
from functools import lru_cache
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1, p4probe, p4robusttx, session
from hfmodem.shrike.arq import PactorArq

FS, SLOT = 48000, 60000
ROOT = Path(__file__).resolve().parents[5]


@lru_cache(maxsize=2)
def _tape(flight='0518'):
    capture, anchor = {
        '0518': ('onair-20261114T051818581719Z-1ef33e2d-*', 5269),
        '0550': ('onair-20261114T055020092775Z-401a2dbf-*', 5431)}[flight]
    stream = next((ROOT / 'operator-data/captures').glob(
        f'{capture}/stream-l7278-*.flac'), None)
    if stream is None:
        pytest.skip(f'WS8EOC {flight}Z recording is not available')
    submitted = next(stream.parent.glob('submitted-tx-*.flac'), None)
    if submitted is None:
        pytest.skip(f'WS8EOC {flight}Z submitted TX recording is not available')
    audio = session.load_wav(str(stream.resolve()), FS).astype(np.float32)
    own = session.load_wav(str(submitted.resolve()), FS)
    idx = np.flatnonzero(np.abs(own) > 1e-4)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 480) + 1):
        if run.size:
            audio[max(0, run[0]+84-240):run[-1]+84+240] = 0
    return audio[anchor:]


def _replay(tmp_path, monkeypatch, parity=None, real_link=False, *, flight='0518', opening=None):
    wav = tmp_path / 'peer.wav'
    session.write_wav(str(wav), _tape(flight))
    keyed, tx = [], onair.RadioTx._tx
    accepted, awaiting_key = [], False
    answer = PactorArq.on_rx_robust_answer
    calls = tuple(pactor1.connect_signal('WS8EOC', invert=inv, variant='robust')
                  for inv in (False, True))
    read_answer = []

    def accept(self, entry):
        nonlocal awaiting_key
        accepted.append(entry)
        awaiting_key = True
        return answer(self, entry)

    def guarded(render):
        def checked(*args, **kwargs):
            assert not awaiting_key, 'render between the answer decision and key'
            return render(*args, **kwargs)
        return checked

    def key(self, audio, what, *args, **kwargs):
        nonlocal awaiting_key
        if awaiting_key:
            read_answer.append(self.p4_link_options._robust_answer.copy())
        invert = self._sent_invert
        if what.startswith('connect'):
            assert invert is not None
            np.testing.assert_array_equal(audio, calls[invert])
        before = len(self.slots_used)
        result = tx(self, audio, what, *args, **kwargs)
        if not self.refused and (len(self.slots_used) > before or self.raster is None):
            awaiting_key = False
            keyed.append((self.slot if self.raster else 0, what,
                          invert if what.startswith('connect') else self.raster.shift(self.slot)))
        return result

    monkeypatch.setattr(onair.RadioTx, '_tx', key)
    monkeypatch.setattr(PactorArq, 'on_rx_robust_answer', accept)
    monkeypatch.setattr(pactor1, 'connect_signal', guarded(pactor1.connect_signal))
    monkeypatch.setattr(p4robusttx, 'robust_short', guarded(p4robusttx.robust_short))
    if not real_link:
        monkeypatch.setattr(onair, '_run_p4_link', lambda *a, **kw: 'opening handed to link')
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    monkeypatch.setattr(p4probe, 'code_revision', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', [
        'onair', '--replay', str(wav), '--mycall', 'W9SSJ', '--dxcall', 'WS8EOC',
        '--dial', '3593500', '--hold', '0', '--connect', 'robust',
        '--after-robust', 'p4-robust', '--rig', 'ftdx10',
        '--serial', '/dev/null', '--ptt-port', '/dev/null',
        '--retries', '20', '--max-cycles', '32', '--outdir', str(tmp_path / 'out'),
        *(['--p4-opening', opening] if opening is not None else []),
        *(['--p4-robust-parity', parity] if parity is not None else [])])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    assert accepted == [0], log.getvalue()
    assert len(read_answer) == 1
    assert (read_answer[0]['slot'], read_answer[0]['sense']) == (
        (2, 0) if flight == '0518' else (3, 1)), log.getvalue()
    (tmp_path / 'replay.log').write_text(log.getvalue())
    result = json.loads((tmp_path / 'out/p4-probe.json').read_text())
    return keyed, log.getvalue(), result


@pytest.mark.parametrize('parity,calls,slots', [
    (None, [0, 1, 2], [3, 4, 5]),
    ('current', [0, 1, 2], [3, 4, 5]),
    ('stock', [0, 1, 2, 3], [4, 5, 6])])
def test_replay_opening_slots(tmp_path, monkeypatch, parity, calls, slots):
    keyed, log, result = _replay(tmp_path, monkeypatch, parity)
    assert [s for s, label, _ in keyed if label.startswith('connect')] == calls, log
    assert [(s, inv) for s, label, inv in keyed if label.startswith('connect')] == [
        (s, bool(s & 1)) for s in calls]
    opening = [(s, label, inv) for s, label, inv in keyed if label.startswith('P4 PROBE')]
    assert [s for s, _, _ in opening] == slots, log
    assert all(f'v={v} hdr=A' in label for (_, label, _), v in zip(opening, (6, 9, 8)))
    assert opening[0][2] is (parity != 'stock')
    assert [e['slot'] for e in result['emissions']] == slots


@pytest.mark.parametrize('parity,first', [(None, 4), ('current', 4), ('stock', 5)])
def test_0550_replay_places_v6_after_v9(tmp_path, monkeypatch, parity, first):
    keyed, log, result = _replay(tmp_path, monkeypatch, parity,
                               flight='0550', opening='9,6,9,8')
    calls = [(s, inv) for s, label, inv in keyed if label.startswith('connect')]
    assert calls == [(s, bool(s & 1)) for s in range(first)], log
    opening = [(s, label, inv) for s, label, inv in keyed if label.startswith('P4 PROBE')]
    assert [(s, inv) for s, _, inv in opening] == [
        (s, bool(s & 1)) for s in range(first, first+4)], log
    assert all(f'v={v} hdr={h}' in label for (_, label, _), (v, h) in zip(
        opening, ((9, 'A'), (6, 'A'), (9, 'B'), (8, 'A'))))
    assert opening[1][2] is (parity != 'stock')
    assert [e['slot'] for e in result['emissions']] == list(range(first, first+4))


@pytest.mark.parametrize('opening', ['9,6,8', '9,9,8'])
def test_stock_keeps_normal_v6_or_no_v6(tmp_path, monkeypatch, opening):
    keyed, log, _ = _replay(tmp_path, monkeypatch, 'stock', opening=opening)
    assert [s for s, label, _ in keyed if label.startswith('connect')] == [0, 1, 2], log
    opening = [(s, inv) for s, label, inv in keyed if label.startswith('P4 PROBE')]
    assert opening == [(3, True), (4, False), (5, True)], log
    assert log.count('stock: no v6/A in the opening; nothing to place') == (
        0 if any('v=6 hdr=A' in label for _, label, _ in keyed) else 1)


@pytest.mark.parametrize('values,index', [((6, 9, 8), 0), ((9, 6, 9, 8), 1),
                                        ((9, 9, 6, 8), 2), ((9, 8, 9, 6), 3),
                                        ((6, 6, 9, 8), 0), ((9, 9, 8), None)])
@pytest.mark.parametrize('sense', [0, 1])
def test_first_v6_sense_uses_answered_call(values, index, sense):
    probe = p4probe.EntryProbe(values=values)
    answer = dict(slot=3, sense=sense)
    got = onair._p4_robust_v6_sense(probe, answer, 4)
    assert got == (None if index is None else sense ^ ((1+index) & 1))
    if got == 0:
        # An already normal v6/A needs no extra call, even at a later index.
        assert onair._p4_robust_v6_sense(probe, answer, 5) == 1


def test_default_ladder_runs_its_old_length_and_names_the_repeats(tmp_path, monkeypatch):
    keyed, log, result = _replay(tmp_path, monkeypatch, None, real_link=True)
    assert sum(what.startswith('P4 LINK short') for _, what, _ in keyed) == 7, keyed
    assert 'LINK DOWN: retry budget exhausted' in log, log
    assert 'gateway still repeating robust entry 0 (052D56)' in log, log
    assert 'LINK DOWN: gateway still repeating' not in log, log


@pytest.mark.parametrize('parity,last', [('stock', 8)])
def test_real_link_names_held_answer(tmp_path, monkeypatch, parity, last):
    _, log, result = _replay(tmp_path, monkeypatch, parity, real_link=True)
    assert 'LINK DOWN: gateway still repeating robust entry 0 (052D56)' in log, log
    assert 'retry budget exhausted' not in result['reason']
    assert 'whole opening plus 2 link slots' in result['reason']
    assert 'silent 1/16' not in log and 'silent 2/16' not in log
    repeats = [e for e in result['link_events'] if e['event'] == 'robust entry repeated']
    assert [e['slot'] for e in repeats] == list(range(last-4, last+1))
    assert all(e['entry'] == 0 and e['errors'] == 0 for e in repeats)
    assert all(980 <= e['phase_ms'] <= 986 for e in repeats)


def _grid():
    return onair._MasterGrid(0, SLOT, 0, packet_n=46080, cs_n=5760,
                             d_max_n=onair._d_max_n(1.25, .04))


def _held_tx():
    args = SimpleNamespace(after_robust='p4-robust', _robust_answer=dict(
        entry=0, phase=.983, sense=0, slot=2, offset_hz=0.))
    probe = p4probe.EntryProbe(values=(6, 9, 8))
    probe.emissions = [dict(slot=s) for s in (3, 4, 5)]
    return SimpleNamespace(p4_link_options=args, p4_probe=probe)


def test_held_tape_windows_follow_phase_and_parity():
    audio, tx = _tape(), _held_tx()
    grid = _grid()
    for slot in range(3, 8):
        begin = slot*SLOT + round(.88*FS)
        window = audio[begin:(slot+1)*SLOT-round(.089*FS)]
        row = onair._p4_held_answer(tx, grid, window, begin, 'opening' if slot < 6 else 'link')
        assert row is not None, slot
        assert (row['entry'], row['word'], row['sense'], row['errors']) == (0, '052D56', slot & 1, 0)
        assert (onair._p4_held_answer_reason(tx.p4_probe) is not None) == (slot == 7)
    assert len(tx.p4_probe.link_events) == 5
    onair._p4_held_answer(tx, grid, window, begin, 'link')
    assert len(tx.p4_probe.link_events) == 5
    missing = tx.p4_probe.robust_repeats.pop(4)
    assert onair._p4_held_answer_reason(tx.p4_probe) is None
    tx.p4_probe.robust_repeats[4] = missing
    tx.p4_probe.left_call_phase = True
    assert onair._p4_held_answer_reason(tx.p4_probe) is None


@pytest.mark.parametrize('change', ['route', 'phase', 'sense', 'entry'])
def test_held_answer_rejects_other_routes_and_wrong_anchor(change):
    tx, grid = _held_tx(), _grid()
    if change == 'route':
        tx.p4_link_options.after_robust = 'p2'
    else:
        tx.p4_link_options._robust_answer[change] = dict(phase=.953, sense=1, entry=3)[change]
    assert onair._p4_held_answer(tx, grid, _tape()[3*SLOT:4*SLOT], 3*SLOT, 'link') is None
    assert not tx.p4_probe.link_events


def test_no_held_answer_leaves_other_openings_alone():
    probe = SimpleNamespace(emissions=[dict(audio_start=0, audio_end=36000)])
    assert onair._p4_held_answer_reason(probe) is None
