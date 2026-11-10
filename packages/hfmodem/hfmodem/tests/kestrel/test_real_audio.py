# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Cross-payload anti-artifact acceptance gate (permanent regression test).

Decodes three REAL captured VARA HF 500 recordings with the frozen open-loop
receiver and asserts, for each: all data frames CRC-pass AND the decoded payload
equals the independently-known transmitted bytes.

  cap6  -> counter payload   00 01 .. fe ff
  capP  -> PRBS15 (256 B)     ground truth: capP/sent.bin
  capR  -> random  (256 B)    ground truth: capR/sent.bin

Ground truth for capP/capR is the exact byte stream sent to VARA-A, independently
confirmed byte-exact by VARA-B's host API during capture (see the session logs).
The decoder uses only fixed reference/permutation/PN tables (spec/tables/bw500)
+ payload-blind onset/c0 scans — no payload knowledge — so byte-exact recovery of
PRBS15 and random proves the counter success is not payload-structure fitting.
"""
import pytest

from hfmodem.kestrel.rx import varahf500 as rx
from hfmodem.tests.kestrel.corpora import (MULTIFRAME_SESSION, bw500_capture,
                                           harness, requires_bw500_captures,
                                           requires_multiframe_session, wav_mono)


@requires_bw500_captures
@pytest.mark.parametrize("name", ["cap6", "capP", "capR"])
def test_cross_payload_byte_exact(name):
    rec, sent = bw500_capture(name)
    if rec is None:
        pytest.skip(f"capture {name} not present")

    result = rx.decode_stream(rec)

    assert len(result.data_frames) == 6, (
        f"{name}: expected 6 data frames, got {len(result.data_frames)}")
    assert result.frames_skipped == 0, (
        f"{name}: single-frame bursts, nothing to skip; "
        f"columns = {[f.burst_columns for f in result.frames]}")
    assert result.complete, (
        f"{name}: CRC per frame = {[f.crc_ok for f in result.data_frames]}, "
        f"skipped {result.frames_skipped}")
    # frozen structural constant: c0 locks to 9 on every burst (incl. connect)
    assert all(f.c0 == 9 for f in result.frames), (
        f"{name}: c0 not constant 9: {[f.c0 for f in result.frames]}")

    decoded = result.payload[:len(sent)]
    assert decoded == sent, (
        f"{name}: NOT byte-exact\n decoded={decoded[:16].hex()}...\n sent   ={sent[:16].hex()}...")


def test_burst_column_classes_map_to_frame_counts():
    """The two measured burst classes and the boundary between them. 403 columns
    is preamble + one 394-column frame, 796 is preamble + two; the ~463 class in
    one session is a different waveform and must not be read as a truncated
    two-frame burst. An unmeasured burst claims one frame, never more."""
    assert [rx.frames_carried(n) for n in (0, 403, 463, 796)] == [1, 1, 1, 2]


@requires_multiframe_session
def test_a_two_frame_burst_decodes_both_frames_byte_exact():
    """A 796-column burst carries two 394-column frames; both come back. Frame 1
    is the ordinary SHORT frame; frame 2 reads the grid480 reference and the
    Stage-1 placement one step advanced (``fk=1``), and decodes to the next 43
    payload bytes byte-exact. Nothing is skipped and the decode is complete."""
    payloads = harness("payloads")
    sent = payloads.prbs(4096, 9)
    chunks = [sent[i:i + 43] for i in range(0, len(sent), 43)]

    x = wav_mono(MULTIFRAME_SESSION)
    spans = rx.burst_spans(x)
    carried = [rx.frames_carried((b - a) // rx.H) for a, b in spans]
    assert carried.count(2) == 45 and carried.count(1) == 7, (
        f"session shape changed: {sorted((b - a) // rx.H for a, b in spans)}")

    a, b = spans[carried.index(2)]
    result = rx.decode_stream(x[max(a - 20000, 0):b + 20000])

    assert len(result.data_frames) == 2
    assert all(f.crc_ok for f in result.data_frames)
    assert all(f.c0 == 9 for f in result.data_frames)
    assert result.frames_skipped == 0
    assert result.complete, "both frames demodulated and CRC-clean is a clean decode"

    # both payloads are real, consecutive 43-byte chunks of the known stream: the
    # second frame is recovered byte-exact, not merely CRC-clean.
    k0 = chunks.index(result.data_frames[0].payload)
    assert result.data_frames[1].payload == chunks[k0 + 1]


@requires_multiframe_session
def test_multiframe_session_recovers_every_byte():
    """End to end over the one session whose bursts pack two frames: every burst's
    frames come back, all CRC-clean, and the concatenated payload is the exact
    4096-byte PRBS9 stream that was sent."""
    payloads = harness("payloads")
    sent = payloads.prbs(4096, 9)

    result = rx.decode_stream(wav_mono(MULTIFRAME_SESSION))

    assert result.frames_skipped == 0
    assert result.complete
    assert len(result.data_frames) == 96          # 45*2 + 6 (one burst is connect)
    assert result.payload[:len(sent)] == sent


def test_a_burst_too_short_for_a_frame_does_not_crash():
    """detect_bursts accepts runs from 150000 samples, and the cell grid needs
    column c0+392, so a recording that ends mid-burst — or a fade that splits one —
    leaves every c0 candidate running off the end. That used to surface as a
    TypeError several frames later inside decode_stream, taking the whole decode
    down rather than skipping one unusable burst."""
    import warnings

    import numpy as np

    from hfmodem.kestrel.rx import varahf500 as rx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n, start in ((200000, 40000), (160000, 1000), (150001, 0)):
            r = rx.decode_burst(np.zeros(n), start)
            assert r.crc_ok is False, f"{n}@{start} claimed a good frame from silence"
