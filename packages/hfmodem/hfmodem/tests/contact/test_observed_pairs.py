# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Exact slot-13 recording: the level-3 over was swallowed by a guessed pair."""
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import wavfile

from hfmodem.contact import slots as S
from hfmodem.core.wav import source


def test_slot13_recording_recovers_all_three_fragments_once(monkeypatch):
    root = Path(__file__).parent / 'fixtures/varac-observed-pairs-1025'
    facts = json.loads((root / 'provenance.json').read_text())
    path = root / 'slot13.flac'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == facts['flac_sha256']
    fs, x = wavfile.read(source(path))
    assert fs == S.FS == facts['sample_rate']
    # This checks decoded recording content, independently of host CPU contention.
    # Charge each high-speed read 20 ms against the unchanged elapsed-time budget;
    # its debt and expiry behavior are checked separately with a controlled clock.
    clock = itertools.count(step=.02)
    monkeypatch.setattr(S, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    monitor = S.SlotMonitor(14_107_250, 14_105_000, passband=(1250, 1750))
    found = []
    # Enough trailing silence to settle a possible partner; no EOF-only decoder.
    x = np.concatenate([x.astype(float) / 32768, np.zeros(6 * fs)])
    for i in range(0, len(x), 4800):
        found += monitor.feed(x[i:i + 4800])
    said = [item[2] for item in found if item[0] == 'qso' and item[1] == 13]
    assert [s.text for s in said if s.kind == 'text'] == facts['expected_text']
    assert [s.text for s in said if s.kind == 'coverage'] == []
    receiver = monitor._channels[0].frames
    assert receiver.overflows == receiver.expired == 0


def test_base_candidate_loss_is_reported_and_breaks_text_carry(monkeypatch):
    monitor = S.SlotMonitor(14_107_250, 14_105_000, passband=(1250, 1750))
    channel = monitor._channels[0]
    evidence = S._Evidence(text='before missing data', last_chunk=('old', 2, 0))
    def feed(samples):
        channel.frames.overflows += 1
        return []
    monkeypatch.setattr(channel.frames, 'feed', feed)
    found = monitor._read(channel, evidence, np.zeros(S.BLOCK))
    assert any(kind == 'qso' and said.kind == 'coverage' and 'validated frames retained' in said.text
               for kind, _, said in found)
    assert evidence.text == '' and evidence.last_chunk is None


def test_retained_overs_are_not_joined_into_tags_across_a_candidate_gap(monkeypatch):
    from types import SimpleNamespace
    from hfmodem.kestrel.rx.varahf500 import FrameResult
    monitor = S.SlotMonitor(14_107_250, 14_105_000, passband=(1250, 1750))
    channel = monitor._channels[0]
    evidence = S._Evidence()
    def feed(samples):
        channel.frames.expired += 1
        return [SimpleNamespace(end=at + 1000, frames=[FrameResult(
            at, (), 4, 1., True, text + b'\x01\0\0', level=2)])
            for at, text in ((0, b'<DI'), (10000, b'SC>'))]
    monkeypatch.setattr(channel.frames, 'feed', feed)
    found = monitor._read(channel, evidence, np.zeros(S.BLOCK))
    assert [said.text for _, _, said in found if said.kind == 'text'] == ['<DI', 'SC>']
    assert not evidence.closed and 'DISC' not in evidence.tags
    assert evidence.text == ''
