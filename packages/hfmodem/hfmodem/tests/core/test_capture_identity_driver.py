# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Whole native driver identity/lifecycle pins using fake card, rig and codec."""
from __future__ import annotations

import json
import shutil
import signal
from pathlib import Path

import pytest

from hfmodem.core import capture_identity as ci, wav
from hfmodem.shrike import onair
from hfmodem.tests import fakeclaim
from hfmodem.tests.shrike.test_armdefaults import _parsed
from hfmodem.tests.shrike.test_grid import _Rig
from hfmodem.tests.shrike.test_identify import Clock


def _native(tmp_path, monkeypatch, *, fail_read=False, fail_close=False):
    events, cards, rigs = [], [], []
    fakeclaim.install(monkeypatch)
    monkeypatch.setattr(onair, '_WRITER', None)
    monkeypatch.setattr(onair, '_CAPTURE_ROOTS', {})
    monkeypatch.setattr(onair, '_CAPTURE_RECORDINGS', {})

    class Card(Clock):
        def __init__(self, *args, **kwargs):
            super().__init__()
            cards.append(self)
            events.append('card-open')

        def read(self, count):
            if fail_read:
                raise RuntimeError('synthetic-read-failure')
            return super().read(count)

        def close(self):
            events.append('card-close')
            super().close()
            if fail_close:
                raise LookupError('synthetic-close-failure')

    class Rig(_Rig):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            rigs.append(self)
            self.stopped = False

        def stop(self):
            events.append('rig-stop')
            self.stopped = True

    def fake_codec(*arguments):
        # Existing codec seam only: synthetic container roundtrip, not FLAC proof.
        assert cards and cards[0].closed
        assert rigs and rigs[0].stopped
        events.append('codec')
        shutil.copyfile(Path(arguments[-1]), Path(arguments[-2]))

    monkeypatch.setattr(onair.ota, 'Rig', Rig)
    monkeypatch.setattr(onair.ota, 'ptt_device', lambda *args, **kwargs: 'fake-ptt')
    monkeypatch.setattr(onair, 'find_device', lambda *args, **kwargs: 0)
    monkeypatch.setattr(onair, '_LiveInput', Card)
    monkeypatch.setattr(wav, '_flac', fake_codec)
    args = _parsed('--transmit', '--serial', 'fake', '--dial', '7091000',
                   '--outdir', str(tmp_path / 'new-native-root'),
                   '--pactor1-only', '--hold', '1', '--max-cycles', '2')
    return args, events, cards, rigs


@pytest.mark.parametrize('failure', ['none', 'read', 'close'])
def test_whole_native_driver_keeps_ownership_and_cleanup(tmp_path, monkeypatch, failure):
    args, events, cards, rigs = _native(tmp_path, monkeypatch,
                                      fail_read=failure == 'read',
                                      fail_close=failure == 'close')
    prior = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    if failure == 'none':
        assert onair.run(args) == 0
    else:
        kind = RuntimeError if failure == 'read' else LookupError
        with pytest.raises(kind, match='synthetic-'):
            onair.run(args)
    assert {s: signal.getsignal(s) for s in prior} == prior
    assert len(cards) == len(rigs) == 1
    assert cards[0].closed and rigs[0].stopped
    # Unkey before closing audio; both must finish before the codec flush.
    assert events.index('rig-stop') < events.index('card-close')
    if 'codec' in events:
        assert events.index('rig-stop') < events.index('codec')
        assert events.index('card-close') < events.index('codec')
    capture = args._capture_identity
    assert capture.ended()
    assert Path(args.outdir) == capture.root('capture')
    facts = capture.lifecycle()
    assert facts['arm_ended'] and facts['unclaimed_writers'] == []
    row = facts['writers']['modem']
    if failure == 'none':
        assert row['state'] == 'finalized'
        assert facts['capture_complete']
        assert 'codec' in events
    else:
        assert row['state'] == 'partial'
        assert not facts['capture_complete']
    for side in Path(args.outdir).glob('*.json'):
        if side.name.startswith('.'):
            continue
        data = json.loads(side.read_text())
        if 'writer_id' in data and 'artifact_id' in data:
            assert data['capture_id'] == capture.capture_id
            assert data['registry_sha256'] == capture.digest


@pytest.mark.parametrize('journal', ['end', 'status'])
def test_whole_native_primary_error_survives_journal_failure(tmp_path, monkeypatch, journal):
    args, _, cards, rigs = _native(tmp_path, monkeypatch, fail_read=True)
    def broken(*args, **kwargs):
        raise OSError('synthetic-journal-failure')
    if journal == 'end':
        monkeypatch.setattr(ci.Capture, 'end', broken)
    else:
        original_status = onair._CaptureWriter.publish_status
        def failed_shutdown(self, writer):
            if writer.capture.ended():
                return broken()
            return original_status(self, writer)
        monkeypatch.setattr(onair._CaptureWriter, 'publish_status', failed_shutdown)
    with pytest.raises(RuntimeError, match='synthetic-read-failure'):
        onair.run(args)
    assert cards[0].closed and rigs[0].stopped


@pytest.mark.parametrize('journal', ['end', 'status'])
def test_successful_native_run_reports_journal_failure(tmp_path, monkeypatch, journal):
    args, _, cards, rigs = _native(tmp_path, monkeypatch)
    def broken(*args, **kwargs):
        raise OSError('synthetic-journal-failure')
    if journal == 'end':
        monkeypatch.setattr(ci.Capture, 'end', broken)
    else:
        original_status = onair._CaptureWriter.publish_status
        def failed_shutdown(self, writer):
            if writer.capture.ended():
                return broken()
            return original_status(self, writer)
        monkeypatch.setattr(onair._CaptureWriter, 'publish_status', failed_shutdown)
    with pytest.raises(OSError, match='synthetic-journal-failure'):
        onair.run(args)
    assert cards[0].closed and rigs[0].stopped
    assert onair._CAPTURE_ROOTS[Path(args.outdir)]._status['failed']


@pytest.mark.realtime
def test_actual_native_deferred_work_finishes_only_its_original_capture(tmp_path, monkeypatch):
    """Real queued native jobs stay bound while a second fake native run ends."""
    import hashlib
    import threading
    old_args, _, old_cards, old_rigs = _native(tmp_path / 'old', monkeypatch)
    old_root = Path(old_args.outdir).absolute()
    blocked, release = threading.Event(), threading.Event()
    original_save = onair._save_capture
    original_drain = onair._drain_captures
    old_queue = None

    def held_save(path, *args, **kwargs):
        if Path(path).absolute().is_relative_to(old_root):
            blocked.set()
            if not release.wait(10):
                raise TimeoutError('synthetic-worker-release-missing')
        return original_save(path, *args, **kwargs)

    def bounded_first_drain(timeout=onair.CAPTURE_DRAIN_S, *, rig_down=False):
        queue = onair._WRITER
        if queue is not None and any(writer.root == old_root for writer, _ in queue._owned.values()):
            queue.drain(0.0)
            queue.store()
            return queue.report()
        return original_drain(timeout, rig_down=rig_down)

    def digests(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file() and not p.is_symlink()}

    monkeypatch.setattr(onair, '_save_capture', held_save)
    monkeypatch.setattr(onair, '_drain_captures', bounded_first_drain)
    try:
        assert onair.run(old_args) == 0
        old_queue = onair._WRITER
        assert blocked.wait(5)
        old_capture = old_args._capture_identity
        old_writer = next(writer for writer, _ in old_queue._owned.values() if writer.root == old_root)
        assert old_capture.ended() and old_queue.pending > 0
        assert not old_capture.lifecycle()['capture_complete']
        old_end = (old_root / '.capture-ended.json').read_bytes()
        assert old_cards[0].closed and old_rigs[0].stopped

        new_args, _, new_cards, new_rigs = _native(tmp_path / 'new', monkeypatch)
        assert onair.run(new_args) == 0
        new_capture = new_args._capture_identity
        new_root = Path(new_args.outdir)
        assert new_capture.capture_id != old_capture.capture_id
        assert new_cards[0].closed and new_rigs[0].stopped
        assert new_capture.lifecycle()['capture_complete']
        new_before = digests(new_root)
        assert not old_capture.lifecycle()['capture_complete']
        release.set()
        assert old_queue.drain(5) == 0
        old_recording = old_queue.unstored[0][1]
        assert isinstance(old_recording, ci.Recording)
        assert old_recording.audio.writer is old_writer
        old_clock_id = old_recording.audio.record['clock_id']
        old_queue.store()
        assert old_capture.lifecycle()['capture_complete']
        assert (old_root / '.capture-ended.json').read_bytes() == old_end
        assert digests(new_root) == new_before
        assert new_capture.lifecycle()['capture_complete']
        with pytest.raises(ci.IdentityError):
            ci.Recording.reserve(old_writer, 'late.wav', clock_id=old_clock_id)
    finally:
        release.set()
        if old_queue is not None:
            old_queue.drain(5)
