# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
import queue
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import wavfile

from hfmodem.contact import passive as P, slots as S
from hfmodem.core.wav import source
from hfmodem.tests.evidence import TREE


def test_native_missing_vmail_is_recovered_at_levels_5_6_7_without_active_protocol():
    prov = json.loads((Path(__file__).parent / 'fixtures/varac-passive-highspeed-1024/provenance.json').read_text())
    path = TREE / 'site/examples/varac-receive/audio/receiver.flac'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prov['flac_sha256']
    fs, x = wavfile.read(source(path))
    assert fs == S.FS
    monitor = S.SlotMonitor(14_105_000, 14_105_000, mycall='W9SSJ')
    found = []
    for i in range(0, len(x), 4800):
        found += monitor.feed(x[i:i + 4800] / 32768)
    said = [r[2] for r in found if r[0] == 'qso' and r[1] == 1]
    text = '\n'.join(s.text for s in said)
    assert 'Just want to say HOWDY!!' in text
    assert 'Tesla was based on the east coast' in text
    assert 'thanks for the education' in text
    assert 'by!!!' in text and 'WB5KUO VarAC' in text
    assert '<QSYR>' in text
    for level in (5, 6, 7):
        assert any(s.kind == 'level' and s.text.startswith(f'level {level}') for s in said)
    assert any('possible retransmission' in s.text for s in said)
    assert {s.kind for s in said} <= {'stage', 'text', 'level', 'coverage', 'payload'}
    assert all(s.station != 'KN6ZUA' for s in said)  # Embedded VMail author is not the RF sender.
    ch = next(c for c in monitor._channels if c.slot == 1)
    assert ch.high.attempts <= 12 and ch.high.skipped == 0


def test_families_run_once_per_emission_and_never_use_base_fallback(monkeypatch):
    calls = []
    def decode(x, bw, level):
        calls.append(level)
        return SimpleNamespace(frames=[])
    monkeypatch.setattr(P.phy, '_family', lambda bw, lv: (SimpleNamespace(decode_burst=decode), None))
    r = P.HighSpeed()
    r.raw = np.zeros(6 * P.FS)
    r.samples = len(r.raw)
    r.pending.append((P.FS, 5 * P.FS))
    assert r.read() == [] and r.read() == [] and calls == [6, 7]


def test_valid_base_emission_does_not_trigger_extra_high_speed_search(monkeypatch):
    monkeypatch.setattr(P.phy, '_family', lambda *a: (_ for _ in ()).throw(AssertionError('extra decode')))
    r = P.HighSpeed(); r.raw = np.zeros(6 * P.FS); r.samples = len(r.raw)
    r.pending.append((P.FS, 5 * P.FS)); r.note_base(P.FS + 100, 5 * P.FS)
    assert r.read() == []


def test_no_unbounded_storage_or_decode_on_continuous_busy_channel():
    r = P.HighSpeed()
    r.floor.extend([1e-8] * 3000)
    for _ in range(40):
        r.feed(np.ones(24000))
    assert len(r.raw) <= P.MAX_SPAN + 2 * P.PAD
    assert len(r.pending) <= P.MAX_PENDING
    assert r.attempts == 0 and r.skipped == 1
    assert '12 seconds' in r.drain_notices()[0]


def watch_with_full_queue():
    w = S.SlotWatch.__new__(S.SlotWatch)
    w._in, w._out = queue.Queue(maxsize=1), queue.Queue()
    w._commands, w._command_lock = deque(), threading.Lock()
    w._dropped, w._generation, w._calling = 0, 0, 14_105_000
    w._table, w._found, w._now = [], [], 0
    w._in.put(np.zeros(960))
    return w


def test_full_audio_queue_drops_without_blocking_and_precedes_next_audio_with_gap():
    w = watch_with_full_queue(); w.put(np.ones(960)); assert w._dropped == 960
    w._in.get_nowait(); w.put(np.ones(960))
    assert w._in.get_nowait() == ('gap', 960)
    assert w._dropped == 960  # Barrier used the only available slot.


def test_full_queue_cannot_make_retune_or_heard_fail_the_active_contact():
    w = watch_with_full_queue()
    w.retune(14_105_750); w.heard('KF0RKC', 1)
    assert w._generation == 1 and len(w._commands) == 2
    w.put(np.ones(960)); assert w._dropped == 960
    w._in.get_nowait(); w._controls()
    assert w._in.get_nowait() == ('tune', 14_105_750, 14_105_000, 1)
    w._controls(); assert w._in.get_nowait() == ('heard', 'KF0RKC', 1)


def test_global_elapsed_time_debt_holds_other_channels_until_credit_returns(monkeypatch):
    monitor = S.SlotMonitor(14_105_000, 14_105_000)
    now, reads = [0.], []
    class Reader:
        ready = True
        def read(self):
            reads.append(now[0]); now[0] += .6
            return []
        def drain_notices(self): return []
    monitor._channels = [SimpleNamespace(slot=i, audio_hz=750 + 750 * i,
                                        high=Reader(), mixer=lambda x: x) for i in range(2)]
    monitor._measure = lambda _: []
    monitor._read = lambda *a: []
    monitor._idle = lambda *a: []
    monitor._judge = lambda ch: SimpleNamespace(state='quiet', calls=[])
    monkeypatch.setattr(S.time, 'monotonic', lambda: now[0])
    monitor._block(np.zeros(S.BLOCK))
    assert len(reads) == 1 and monitor._high_credit < 0
    for _ in range(7): monitor._block(np.zeros(S.BLOCK))
    assert len(reads) == 1
    for _ in range(3): monitor._block(np.zeros(S.BLOCK))
    assert len(reads) == 2


def test_partial_high_speed_emission_reports_known_gap_and_retains_original_crc(monkeypatch):
    frames = [b'original-frame-and-crc']
    empty = SimpleNamespace(decode_burst=lambda *a: SimpleNamespace(frames=[]))
    ofdm = SimpleNamespace(decode_burst=lambda *a: SimpleNamespace(
        frames=frames, level=7, quality=SimpleNamespace(raw_bits=200)),
        ROWS={(500, 7): SimpleNamespace(coded=100)})
    monkeypatch.setattr(P.phy, '_family', lambda bw, lv: (empty if lv == 6 else ofdm, None))
    r = P.HighSpeed(); r.raw = np.zeros(6 * P.FS); r.samples = len(r.raw)
    r.pending.append((P.FS, 5 * P.FS))
    got = r.read()
    assert [f.frame_bytes for f in got] == frames
    assert '1 of 2 frames validated' in r.drain_notices()[0]


def test_two_separate_identical_frames_are_not_silently_deduplicated(monkeypatch):
    result = SimpleNamespace(frames=[b'wire-plus-crc'], level=6)
    receiver = SimpleNamespace(decode_burst=lambda *a: result)
    monkeypatch.setattr(P.phy, '_family', lambda *a: (receiver, None))
    r = P.HighSpeed(); r.raw = np.zeros(12 * P.FS); r.samples = len(r.raw)
    r.pending.extend([(P.FS, 5 * P.FS), (6 * P.FS, 10 * P.FS)])
    assert len(r.read()) == len(r.read()) == 1
    assert r.attempts == 2


def test_pending_band_then_slot_tune_keeps_authoritative_calling_frequency():
    w = watch_with_full_queue()
    w.retune(7_105_000, 7_105_000); w.retune(7_105_750)
    w._in.get_nowait(); w._controls()
    assert w._in.get_nowait() == ('tune', 7_105_750, 7_105_000, 2)


def test_capture_drops_instead_of_waiting_for_ui_control_lock():
    w = watch_with_full_queue()
    with w._command_lock:
        w.put(np.ones(960))
    assert w._dropped == 960


def test_retune_cannot_interleave_between_control_flush_and_audio_enqueue():
    w = watch_with_full_queue(); w._in = queue.Queue(maxsize=4)
    entering, release = threading.Event(), threading.Event()
    original = w._in.put_nowait
    def put(item):
        if isinstance(item, np.ndarray):
            entering.set()
            assert release.wait(2)
        original(item)
    w._in.put_nowait = put
    audio = threading.Thread(target=w.put, args=(np.ones(960),))
    audio.start(); assert entering.wait(2)
    tune = threading.Thread(target=w.retune, args=(14_105_750,))
    tune.start(); release.set()
    audio.join(2); tune.join(2)
    assert not audio.is_alive() and not tune.is_alive()
    assert isinstance(w._in.get_nowait(), np.ndarray)
    assert w._in.get_nowait() == ('tune', 14_105_750, 14_105_000, 1)


def test_capture_gap_resets_reader_and_text_carry_before_later_audio(monkeypatch):
    seen = []
    class Monitor:
        def __init__(self, dial, calling, **kwargs):
            self.dial, self.samples = dial, 0
            self._evidence = {1: SimpleNamespace(text='stale<', last_chunk=('old', 6, 0))}
            self._channels = [SimpleNamespace(slot=1)]
        @property
        def now(self): return self.samples / S.FS
        def retune(self, dial): seen.append('reset')
        def _said(self, ch, kind, text): return ('qso', ch.slot, S.Said(ch.slot, self.now, kind, text))
        def feed(self, x):
            assert self._evidence[1].text == '' and self._evidence[1].last_chunk is None
            self.samples += len(x); seen.append('later audio'); return []
        def table(self): return []
    monkeypatch.setattr(S, 'SlotMonitor', Monitor)
    monkeypatch.setattr(S.os, 'nice', lambda _: None)
    inbox, outbox = queue.Queue(), queue.Queue()
    for item in [('gap', 72000), np.zeros(4800), None]: inbox.put(item)
    S._watch(inbox, outbox, 14_105_000, 14_105_000, {})
    assert seen == ['reset', 'later audio']
    assert '1.50 seconds skipped' in outbox.get_nowait()[1][2].text


def test_partial_emission_never_joins_retained_fragments_across_unknown_loss(monkeypatch):
    monitor = S.SlotMonitor(14_105_000, 14_105_000)
    class Reader:
        ready = True
        def read(self): return ['first', 'third']
        def drain_notices(self): return ['partial DATA emission: 2 of 3 frames validated']
    ch = SimpleNamespace(slot=1, audio_hz=750, high=Reader(), mixer=lambda x: x)
    monitor._channels = [ch]
    monitor._measure = lambda _: []
    monitor._read = lambda *a: []
    monitor._idle = lambda *a: []
    monitor._judge = lambda _: SimpleNamespace(state='quiet', calls=[])
    carried = []
    def frame(ch, ev, data):
        carried.append(ev.text); ev.text = data; return []
    monitor._frame = frame
    monitor._block(np.zeros(S.BLOCK))
    assert carried == ['', ''] and monitor._evidence[1].text == ''
