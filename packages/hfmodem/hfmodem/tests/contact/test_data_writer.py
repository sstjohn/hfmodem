# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import socket
import threading
import time

import pytest

from hfmodem.contact.data_writer import DataWriter, WriterUnavailable


class Stream:
    def __init__(self, fail=False):
        self.entered, self.release = threading.Event(), threading.Event()
        self.writes = []
        self.fail = fail

    def sendall(self, data):
        self.entered.set()
        assert self.release.wait(2)
        if self.fail:
            raise OSError('partial write')
        self.writes.append(data)

    def shutdown(self, how):
        self.release.set()


def completions(writer, count):
    result = []
    deadline = time.monotonic() + 2
    while len(result) < count and time.monotonic() < deadline:
        result.extend(writer.poll())
        threading.Event().wait(.001)
    assert len(result) == count
    return result


def test_serial_bounded_writer_includes_inflight_bytes():
    stream = Stream()
    writer = DataWriter(stream, max_bytes=6, max_frames=2)
    try:
        first = writer.submit(b'abc', ('link', 1))
        assert stream.entered.wait(1)
        second = writer.submit(b'def', ('link', 1))
        with pytest.raises(WriterUnavailable):
            writer.submit(b'g')
        assert writer.pending
        stream.release.set()
        done = completions(writer, 2)
        assert [d.write_id for d in done] == [first, second]
        assert all(d.disposition == 'complete' for d in done)
        assert stream.writes == [b'abc', b'def']
        assert not writer.pending
    finally:
        writer.close()


def test_unknown_write_poisons_stream_and_cancels_queued_frames():
    stream = Stream(fail=True)
    writer = DataWriter(stream)
    try:
        writer.submit(b'prefix', 10)
        assert stream.entered.wait(1)
        writer.submit(b'never send', 11)
        stream.release.set()
        done = completions(writer, 2)
        assert [(d.token, d.disposition) for d in done] == [(10, 'unknown'), (11, 'cancelled')]
        assert writer.poisoned
        with pytest.raises(WriterUnavailable):
            writer.submit(b'<DISC>')
        assert stream.writes == []
    finally:
        writer.close()


def test_close_interrupts_a_real_blocked_socket():
    sending, receiving = socket.socketpair()
    sending.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    writer = DataWriter(sending, max_bytes=4 * 1024 * 1024)
    try:
        writer.submit(b'x' * (2 * 1024 * 1024), 'old generation')
        writer.close(timeout=.5)
        assert not writer._thread.is_alive()
        with pytest.raises(WriterUnavailable):
            writer.submit(b'new generation')
    finally:
        sending.close()
        receiving.close()


def test_unconsumed_completions_are_bounded():
    stream = Stream()
    stream.release.set()
    writer = DataWriter(stream, max_frames=1)
    try:
        writer.submit(b'a')
        assert stream.entered.wait(1)
        with pytest.raises(WriterUnavailable):
            writer.submit(b'b')
        completions(writer, 1)
        writer.submit(b'b')
        completions(writer, 1)
    finally:
        writer.close()
