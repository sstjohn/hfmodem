# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A shared eight-symbol shape is not positive DATA feedback."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from .test_data_over_gate import _connected
from .test_data_replies_0919 import native
from .test_data_naks_0919 import native as negative, ROWS as NAK_ROWS


def station(monkeypatch):
    """KC9GHZ's link reading W1AW's tails, so its own native reply is foreign."""
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'end')
    for name in ('over_continue', 'over_continue_down'):
        tails = getattr(VF, name)
        monkeypatch.setattr(VF, name, lambda called, bw, tails=tails: tails('W1AW', bw))
    return hs, io


def feed(hs, audio):
    for start in range(0, len(audio), 4096):
        hs.on_rx_stream(audio[start:start + 4096])


def test_unknown_reply_waits_for_query_and_teaches_nothing(monkeypatch):
    hs, io = station(monkeypatch)
    pending = hs._tx_pending
    reply = native('stock-2300-continue-1')
    feed(hs, reply)
    assert hs._tx_pending == pending and len(io.sent) == 1
    assert any('unclassified' in m for m in io.msgs), io.msgs
    assert hs._probe_intermediate_answer()
    feed(hs, native('stock-2300-query-answer'))
    assert hs._tx_pending[1] == pending[1] + 1
    sent = len(io.sent)
    feed(hs, reply)
    assert hs._tx_pending[1] == pending[1] + 1 and len(io.sent) == sent


def test_negative_feedback_after_an_unknown_reply_retries(monkeypatch):
    hs, io = station(monkeypatch)
    reply = native('stock-2300-continue-1')
    feed(hs, reply)
    assert hs._probe_intermediate_answer()
    row = next(r for r in NAK_ROWS if r['call'] == 'KC9GHZ' and r['seed'] == 288)
    feed(hs, negative(row))
    assert hs._tx_retries == 1


def test_ack_of_another_link_is_not_authority_to_retire_this_pending_frame():
    hs, io = _connected()
    hs.called = 'N0XYZ'
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'last')
    old = hs._tx_pending
    feed(hs, native('stock-2300-continue-1'))
    assert hs._tx_pending == old and len(io.sent) == 1


def unmapped_native(name):
    from pathlib import Path
    from scipy.io import wavfile
    path = (Path(__file__).with_name('fixtures') /
            'unmapped-data-replies-0919' / (name + '.wav'))
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == VA.MK.FS and x.dtype.kind == 'f'
    return x.astype(float)


def unmapped_station():
    hs, io = _connected()
    hs.called = 'N0XYZ'
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'end')
    return hs, io


def test_native_formerly_unmapped_link_reads_its_own_continue():
    hs, io = unmapped_station()
    assert hs._peer_over_continue(unmapped_native('unmapped-ack'))
    assert not [m for m in io.msgs if 'unclassified' in m], io.msgs


def test_native_long_nak_can_arrive_directly_without_a_query():
    from hfmodem.kestrel.arq import phy
    audio = unmapped_native('direct-long-nak')
    for chunk in (512, 4096, 4800):
        hs, io = unmapped_station()
        old = hs._tx_pending
        assert not hs._intermediate_query_attempted
        assert not hs._peer_data_nak(audio[:int(.65 * VA.MK.FS)])
        for at in range(0, len(audio), chunk):
            hs.on_rx_stream(audio[at:at + chunk])
        assert hs._tx_retries == 1 and hs._intermediate_query_attempts == 0
        assert hs._tx_pending[1:] == (old[1], 2)
        body = hs._tx_pending[0]
        assert (phy.vara_payload(body, caller=hs.caller, body_len=len(body))
                + b''.join(hs._txq)) == b'A' * 89 + b'B' * 89 + b'end'



def drive_to_query_deadline(monkeypatch, hs, audio, moved):
    """Feed 20 ms blocks on a clock that puts the query deadline at 1.8 s of
    audio, calling the query whenever the driver would; when did ``moved`` hold?"""
    fs = VA.MK.FS
    clock = [hs._pending_answer_at]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: clock[0])
    assert hs.intermediate_query_due_in() == pytest.approx(1.8, abs=1e-3)
    start, read = clock[0], None
    for at in range(0, len(audio), 960):
        hs.on_rx_stream(audio[at:at + 960])
        clock[0] = start + (at + 960) / fs
        due = hs.intermediate_query_due_in()
        if due is not None and due <= 0:
            hs.query_intermediate_answer()
        if read is None and moved():
            read = clock[0] - start
    return read


@pytest.mark.parametrize('tail', [1.387, 1.437, 1.487])
def test_direct_long_nak_is_read_at_its_tail_between_scans(monkeypatch, tail):
    # The NAK ends between the last turnaround scan (1.342 s) and the query
    # deadline (1.8 s); the half-second stride would next reach it at 1.84 s.
    fs = VA.MK.FS
    audio = unmapped_native('direct-long-nak')
    hs, io = unmapped_station()
    old = hs._tx_pending
    kind = VF.for_bw(VF.SESSION_DATA_NAK_QUERY, hs.bw)
    _, start = VA._payload_fit(audio, kind, np.asarray(VF.handshake_tones(hs.called, kind)),
                               band=hs._band)
    end = start + VA._span(kind)
    buf = np.pad(audio[end - int(tail * fs):], (0, 2 * fs))[:int(2.4 * fs)]
    read = drive_to_query_deadline(monkeypatch, hs, buf, lambda: hs._tx_retries)
    assert read is not None and read <= tail + .1
    assert hs._tx_retries == 1 and hs._tx_pending[1:] == (old[1], 2)
    assert hs._intermediate_query_attempts == 0 and not hs._intermediate_query_attempted
    assert len(io.sent) == 2


@pytest.mark.parametrize('tail', [1.46, 1.62, 1.78])
def test_continue_between_scans_is_read_before_the_query_keys(monkeypatch, tail):
    # Eight symbols leave no partial fit to wake on: only the deadline's own
    # scan of the buffer reads a continue that lands after the 1.342 s scan.
    fs = VA.MK.FS
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'last')
    old = hs._tx_pending
    whole_at = .58
    buf = np.pad(native('stock-2300-continue-1'),
                 (int((tail - whole_at) * fs), 2 * fs))[:int(2.4 * fs)]
    read = drive_to_query_deadline(monkeypatch, hs, buf,
                                   lambda: hs._tx_pending[1] != old[1])
    assert read is not None and round(read, 6) <= 1.8 + .02
    assert hs._intermediate_query_attempts == 0 and not hs._intermediate_query_attempted
    assert len(io.sent) == 2


@pytest.mark.parametrize('heard', [.02, 1.5])
def test_a_stopped_stream_holds_the_query_at_most_one_block_past_its_deadline(
        monkeypatch, heard):
    fs = VA.MK.FS
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'last')
    clock = [hs._unkeyed_at]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: clock[0])
    for at in range(0, int(heard * fs), 960):
        hs.on_rx_stream(np.zeros(960))
        clock[0] += 960 / fs
    deadline = hs._pending_answer_at + VA._OVER_RETRY_S
    if heard > VA._OVER_RETRY_S - VA._STREAM_STALL_S:
        clock[0] = deadline + .15
        assert hs.intermediate_query_due_in() > 0
        assert not hs.query_intermediate_answer()
    clock[0] = deadline + VA._STREAM_STALL_S + .01
    assert hs.intermediate_query_due_in() == 0
    assert hs.query_intermediate_answer()
    assert hs._intermediate_query_attempts == 1 and len(io.sent) == 2
