# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recording faults preserve subsequent PCM and bounded shutdown."""
import errno
import os
from pathlib import Path
import threading
import time
import wave

import pytest

from hfmodem.core import wav
from hfmodem.core import recording
from hfmodem.tests.station.recording_support import bounded, flush, pcm


@pytest.fixture(params=(0, .3), ids=('fast-fsync', 'slow-fsync'))
def slow_fsync(request, monkeypatch):
    """Arms an fsync delay (WSL2's virtual disk) for the cleanup close, after the timed assertions."""
    fsync = os.fsync
    def arm():
        monkeypatch.setattr(os, 'fsync', lambda fd: (time.sleep(request.param), fsync(fd))[1])
    return arm


@pytest.mark.parametrize('split', (False, True))
def test_sample_clock_resumes_after_segment_open_failure(sessions, monkeypatch, split):
    rec, root = sessions('everything')
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 48000)
    lane = rec.stream('rx', rate=8, clock='card')
    original = Path.open
    failed = []

    def opened(path, *args, **kwargs):
        if path.suffix == '.wav' and args == ('xb',) and not failed:
            failed.append(path)
            raise OSError(errno.EIO, 'segment open unavailable')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', opened)
    if split:
        lane.push(100, pcm(100, 104))
        flush(rec)
    first = 104 if split else 100
    lane.push(first, pcm(first, 116))
    flush(rec)
    lane.push(116, pcm(116, 124))
    flush(rec)
    bounded(rec.close)
    assert failed
    assert lane.gaps == [[100, 108]]
    parts = []
    for path in lane.writer.paths:
        with wave.open(str(path)) as src:
            parts.append((int(path.stem.rsplit('-s', 1)[1]), src.getnframes()))
    assert parts == [(108, 8), (116, 8)]


def test_later_close_enospc_reaches_session_failure(sessions, monkeypatch):
    reports = []
    rec, _ = sessions('everything', report=reports.append)
    lane = rec.stream('rx', rate=8, clock='card')
    lane.push(0, pcm(0, 8))
    flush(rec)
    first = OSError(errno.EIO, 'first segment close')
    second = OSError(errno.ENOSPC, 'second segment close')

    def fail_close(error):
        close = lane.writer._wav.close
        def failed():
            close()
            raise error
        monkeypatch.setattr(lane.writer._wav, 'close', failed)

    fail_close(first)
    lane.push(16, pcm(16, 24))
    flush(rec)
    fail_close(second)
    bounded(rec.close)
    assert lane.disabled, 'later ENOSPC must disable persistence'
    assert any('second segment close' in message for message in reports)
    assert lane._reported is second


def test_close_deadline_includes_stalled_origin_marker(sessions, monkeypatch, slow_fsync):
    rec, _ = sessions('everything')
    lane = rec.stream('rx', rate=8, clock='card')
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = lane._sync_origin
    def marker(path):
        assert threading.current_thread().name == 'recording-store'
        entered.set()
        assert release.wait(3)
        original(path)
    monkeypatch.setattr(lane, '_sync_origin', marker)
    close_timeout = recording._CLOSE_TIMEOUT
    monkeypatch.setattr(recording, '_CLOSE_TIMEOUT', .15, raising=False)
    producer = threading.Thread(target=lambda: lane.push(0, pcm(0, 8)), daemon=True)
    errors = []
    def close():
        try:
            rec.close()
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()
    producer.start()
    closer = threading.Thread(target=close, daemon=True)
    try:
        producer.join(.1)
        assert not producer.is_alive(), 'origin publication blocked the producer'
        assert entered.wait(2)
        started = time.monotonic()
        closer.start()
        assert finished.wait(.8), 'close exceeded its deadline on origin publication'
        assert isinstance(errors[0], TimeoutError)
        assert time.monotonic() - started < .8
    finally:
        release.set()
        monkeypatch.setattr(recording, '_CLOSE_TIMEOUT', close_timeout)
        producer.join(2)
        if closer.ident is not None:
            closer.join(2)
    slow_fsync()
    bounded(rec.close)


def test_close_closes_pcm_before_stalled_storage_wait(sessions, monkeypatch, slow_fsync):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=8, clock='card')
    lane.push(0, pcm(0, 8))
    flush(rec)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def stall():
        entered.set()
        assert release.wait(3)
    rec._enqueue(stall)
    assert entered.wait(2)
    with monkeypatch.context() as patch:
        patch.setattr(recording, '_CLOSE_TIMEOUT', .15, raising=False)
        errors = []
        def close():
            try:
                rec.close()
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()
        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        try:
            assert finished.wait(.8), 'storage wait exceeded the overall close deadline'
            assert isinstance(errors[0], TimeoutError)
            assert lane.writer._wav is None
            assert lane.writer.paths[0].is_file()
            assert rec.index['segments'][0]['state'] in ('storing', 'pending')
        finally:
            release.set()
            closer.join(2)
    slow_fsync()
    bounded(rec.close)


def test_pcm_seal_preserves_clock_for_later_queued_audio(sessions):
    rec, _ = sessions('everything')
    lane = rec.stream('rx', rate=8, clock='card')
    lane.push(100, pcm(100, 108))
    flush(rec)
    rec._pcm.put(('seal',))
    flush(rec)
    lane.push(108, pcm(108, 116))
    flush(rec)
    bounded(rec.close)
    assert lane.gaps == []
    assert [row['actual'] for row in lane.writer.coverage.values()] == [[100, 108], [108, 116]]
    assert [int(path.stem.rsplit('-s', 1)[1]) for path in lane.writer.paths] == [100, 108]
