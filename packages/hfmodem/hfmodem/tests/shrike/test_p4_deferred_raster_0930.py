# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""0210: a late post-key observation must not schedule the next turn in the past."""
import json
from pathlib import Path
import runpy
import sys

import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import onair
from hfmodem.shrike.p4observation import Kind
from hfmodem.tests.shrike import test_p4_link_0927 as support


@pytest.mark.parametrize('body_ms', [50, 70])
def test_0210_deferred_opening_does_not_poison_following_deadlines(monkeypatch, body_ms):
    root = Path(__file__).resolve().parents[5]
    capture = root/'captures/onair-0930-0210'
    require_input(capture/'stream.flac')
    require_input(capture/'p4-probe.json')
    require_input(root/'working/p4-plan-0930/stall-bisect/drive.py')
    probe = json.loads((capture/'p4-probe.json').read_text())
    opening = probe['emissions'][0]
    fs = support.FS

    class Capture(support.Stream):
        transmit = True
        key_notice = 1796
        tx_latency_n = 144
        _blk = 128
        wall = 0
        lost = 0

        def take_until(self, end):
            audio = super().take_until(max(self.pos, end))
            # 0210: 128-sample callbacks and about 14 ms delivery latency.
            self.wall = max(self.wall, -(-self.pos//128)*128 + round(.014*fs))
            return audio

        def sample_now(self):
            return max(self.pos, self.wall)

        def clamp_late(self, at):
            delivered = (self.sample_now()-round(.014*fs))//128*128
            return max(0, delivered + self.key_notice - at)

    monkeypatch.setattr(support, 'Stream', Capture)
    run = onair._run_p4_link
    results = []

    def drive(tx, stream, raster):
        tx.settle = .040
        read = tx.sessrx.p4.read_window
        charged_body = False

        def timed(*args, **kwargs):
            nonlocal charged_body
            records = read(*args, **kwargs)
            first = not charged_body and any(
                r.observation.kind == Kind.BODY and r.observation.frame is not None
                for r in records)
            charged_body |= first
            stream.wall = stream.sample_now() + round((body_ms if first else 15)*fs/1000)
            return records

        monkeypatch.setattr(tx.sessrx.p4, 'read_window', timed)
        transmit = tx._tx

        def emit(audio, what):
            transmit(audio, what)
            # The production transmitter blocks, then flushes through its tail.
            stream.wall = max(stream.wall, tx.tx_end)
            stream.pos = max(stream.pos, tx.tx_end)

        monkeypatch.setattr(tx, '_tx', emit)
        run(tx, stream, raster)
        results.append(tx.p4_probe)

    monkeypatch.setattr(onair, '_run_p4_link', drive)
    monkeypatch.setattr(sys, 'argv', ['drive.py', str(capture/'stream.flac'),
        str(opening['audio_start'] % 60000), str(opening['audio_start']), '1',
        str(opening['audio_end']/fs), '24'])
    runpy.run_path(str(root/'working/p4-plan-0930/stall-bisect/drive.py'), run_name='__main__')
    result, = results
    deferred = [e for e in result.link_events if e['event'] == 'answer-deferred']
    if body_ms == 50:
        # B: a 50 ms read leaves the opening keyable, so it is ACKed in its own slot.
        # Later measured reads may legitimately defer a different packet;
        # those events must not be mistaken for this opening's disposition.
        assert not [e for e in deferred if e['at'] == 527685]
        assert result.emissions[1]['audio_start'] == 527685+round(.920*fs)
    else:
        assert (deferred[0]['at'], deferred[0]['due']) == (527685, 631845)
        assert result.emissions[1]['audio_start'] == deferred[0]['due']
        # The post-key read really did leave the prior, unread copy to expire.
        assert any(h['kind'] == Kind.NORMAL and 12.24*fs < h['onset_sample'] < 12.27*fs
                   and h.get('status') is None for h in result.heard)
    assert all(e['due'] >= e['at']+round(.90*fs) for e in deferred)
    decisions = [e for e in result.link_events if e['event'] == 'Decision'
                 and e['expected'] == 'packet' and 13*fs < e['at'] < 24*fs]
    assert decisions
    assert all(e['scheduled_at'] >= e['at'] + round(.90*fs) for e in decisions)
    assert any(e.get('word') == 'V27' and 14.40*fs < e['audio_start'] < 14.43*fs
               for e in result.emissions)
