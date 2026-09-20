# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The live BW500 stream retains both native level-1 frames until their end.

Its old base-level carry discarded the first 1.17 seconds of a clean two-frame
level-1 burst. Batch decoding alone did not exercise that streaming boundary.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.tests.kestrel.test_bw500_low_levels import _audio, recordings
from hfmodem.tests.kestrel.test_bw500_session import _station


@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_native_two_frame_level1_survives_live_stream_carry(chunk):
    name = 'bw500-level1-two-frames'
    meta = recordings()[name]
    x = _audio(name)
    first, end = meta['burst_samples_in_slice']
    expected = bytes.fromhex(meta['expected_payload_hex'])
    hs, io = _station()
    hs.caller, hs.called = 'W9SSJ', 'KC9GHZ'
    # Lead silence forces the rolling buffer to prune before the long over ends;
    # it must discard only stale silence, retaining the first frame's head.
    lead = RX.FS * 2
    stream = np.concatenate([np.zeros(lead), x, np.zeros(RX.FS // 4)])
    captured = []
    answer = hs._answer_data_over
    def observe(window):
        captured.append(window.copy())
        return answer(window)
    hs._answer_data_over = observe
    for at in range(0, len(stream), chunk):
        part = stream[at:at + chunk]
        io.fed = at + len(part)
        hs.on_rx_stream(part)
    assert b''.join(io.host) == expected
    assert list(map(len, io.host)) == [9, 9]
    assert io.keys == 1
    assert len(captured) == 1
    # Full native span must survive; length alone cannot prove the head survived.
    # Every retained sample is from the source, so its first distinctive sample
    # gives a cheap exact-subsequence check, with no correlation or new decoder.
    native = x[first:end]
    starts = np.flatnonzero(captured[0] == native[0])
    assert any(np.array_equal(captured[0][at:at + len(native)], native) for at in starts)
    assert not any('will not decode' in message for message in io.msgs)
    assert lead + end <= io.keyed_at[0] <= lead + end + RX.FS // 4 + chunk
    # Late energy-bracket delivery of the same samples must not duplicate host
    # bytes or key a second answer after the stream already consumed the over.
    hs.on_rx_audio(x)
    assert b''.join(io.host) == expected and io.keys == 1


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_carry_covers_two_frames_at_every_supported_level(level):
    if level == RX.BASE_LEVEL:
        complete = (RX._PREAMBLE + 2 * RX.NSYM) * RX.H
    else:
        record = RX.INDEX_RECORDS[level]
        complete = (record.lead + 2 * record.ncols) * record.dw
    assert VA._OVER500_LONGEST >= complete
    assert VA._OVER500_CARRY >= complete + RX.FS


def test_silence_still_has_a_bounded_stream_buffer_and_keys_nothing():
    hs, io = _station()
    for _ in range((3 * VA._OVER500_CARRY) // 4800 + 1):
        hs._stream_over_500(np.zeros(4800))
        assert len(hs._ov_buf) < VA._OVER500_CARRY + VA._OVER500_COL
    assert not io.host and io.keys == 0
