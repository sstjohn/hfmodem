# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Which alignment a control signal's reader reports, out of the band that decode.

Twenty bits, no CRC, six words at mutual distance twelve: on
`ws8eoc-0910/first-rms.wav` every alignment from 300 samples early to 120 late
reads CS3 at zero bit errors, fifteen of them over 8.75 ms. The hard decision
cannot choose between them, and what the reader returns is the peer's phase as
far as `p3_reply_shift`, the raster flywheel and a changeover's body offset are
concerned. So it has to be the burst's own head and not the leading edge of the
reader's capture range, which is what keeping the first strict minimum returned.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p3acquire, placement, rx, rxfront
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm

FS = rxfront.FS
SPS = rxfront.SPS
TOLERANCE = SPS // 10
"""1 ms. The raster the reading feeds is quantised far coarser than this: the
projection walks the recorded onsets by 60, 60 and 48 samples over the morning's
three real cycle spacings, so a reader inside this stays on one raster."""

METADATA = Path(__file__).with_name("fixtures") / "ws8eoc-0910" / "metadata.json"
pytestmark = pytest.mark.skipif(
    not METADATA.exists(),
    reason=f"WS8EOC morning recordings absent: {METADATA}")


def crops():
    """The morning's recorded changeovers: PCM at its own CFO, and its head."""
    for row in json.loads(METADATA.read_text())["fixtures"]:
        if row["correction_hz"] is None:
            continue
        pcm = recorded_pcm({"file": f"ws8eoc-0910/{row['file']}",
                            "sha256": row["pcm_sha256"]})
        yield (row["file"],
               p3acquire.compensate(pcm, float(row["correction_hz"])),
               row["onset_relative"])


def first_crop():
    return next(c for c in crops() if c[0] == "first-rms.wav")


def zero_error_alignments(audio: np.ndarray, at: int):
    """Every sixteenth within four symbols of `at` that reads CS3 without error."""
    pulse = rx._pulse(SPS)
    delay = (pulse.size - 1) // 2
    Z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in rxfront.HDR_TONES}
    return Z, delay, [
        st for st in range(at - 4 * SPS, at + 4 * SPS + 1, SPS // 16)
        if rx.nearest_control_signal(rx.cs_bits(Z, st, delay))
        == (placement.BREAKIN_CS, 0)]


def test_the_tracked_read_lands_on_the_head_the_recording_holds():
    """Within 1 ms of the manifest's onset, which is the acquisition's answer too."""
    _, audio, head = first_crop()
    ev = rxfront.SyncedRx().control_signal_at(audio, head)
    assert ev is not None and ev.packet is not None, "the changeover reads at all"
    assert abs(ev.start - head) <= TOLERANCE, \
        f"tracked {ev.start} against the manifest's {head}"
    assert p3acquire.changeover(audio).event.start == head, \
        "...and the acquisition, which measures the head, agrees with the manifest"


def test_the_first_alignment_that_decodes_is_six_milliseconds_early():
    """The negative control: the tie-break this replaced could not have passed.

    Nothing in the twenty bits separates the fifteen alignments -- they are one
    codeword at one distance -- so a reader that keeps the first strict minimum
    reports where its sweep began finding the word rather than where the word is.
    """
    _, audio, head = first_crop()
    _, _, zero = zero_error_alignments(audio, head)
    assert [st - head for st in zero] == list(range(-300, 121, SPS // 16)), \
        "the premise: fifteen alignments, one word, no bit errors"
    assert zero[0] - head == -300
    assert abs(zero[0] - head) > TOLERANCE, \
        "the first minimum is outside the bound the tracked read now holds"


def test_the_coherent_peak_is_what_separates_the_band():
    """And it is a peak: one alignment, strictly above every other in the band."""
    _, audio, head = first_crop()
    Z, delay, zero = zero_error_alignments(audio, head)
    q = [rxfront._cs_coherence(Z, delay, st, placement.BREAKIN_CS) for st in zero]
    assert zero[int(np.argmax(q))] == head
    assert sorted(q)[-1] > sorted(q)[-2]


def test_every_morning_changeover_is_read_on_its_own_head():
    """All five physical copies, each against the onset its manifest row records."""
    off = {}
    for name, audio, head in crops():
        ev = rxfront.SyncedRx().control_signal_at(audio, head)
        assert ev is not None and ev.packet is not None, f"{name} did not read"
        off[name] = ev.start - head
    assert len(off) == 5
    assert max(abs(d) for d in off.values()) <= TOLERANCE, off
