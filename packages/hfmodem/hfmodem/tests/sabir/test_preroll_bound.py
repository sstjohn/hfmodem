# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The window a header may be found in, bounded from both sides.

`arq.modem.PREROLL` is squeezed between two independent facts that live in
different modules and were chosen for unrelated reasons:

  * it must cover the widest pre-roll the segmenter can put in front of a burst,
    or a found burst is a burst that will not decode;
  * it must stay clear of the OFDM body's leading ZC segment, or the correlator
    finds the body's preamble instead of the header's and the header decodes as
    whatever the body happens to look like.

There are about five hundred samples between those, and nothing in either module
mentions the other. A comment saying so is worth having and is not worth relying
on: `monitor.SEG_PAD` or `SEG_FRAME` can be retuned by someone reading only the
segmenter, and the failure that follows is a decoder that quietly stops finding
headers on exactly the audio it was widened for.
"""
from __future__ import annotations

from hfmodem.sabir import monitor
from hfmodem.sabir.arq.modem import PREROLL

#: Where the body's leading ZC segment completes, past the header's end. Below
#: this the header's own preamble still wins the correlation.
BODY_PREAMBLE_SAMPLES = 5632


def test_the_window_covers_the_widest_pre_roll_the_segmenter_can_produce():
    """`SEG_PAD` quiet frames kept before the burst, plus the frame the energy
    gate opened in — the gate cannot open earlier than the frame it fired on."""
    widest = (monitor.SEG_PAD + 1) * monitor.SEG_FRAME
    assert PREROLL >= widest, (
        f"the segmenter can put {widest} samples in front of a burst and the "
        f"header window is {PREROLL}; a found burst would not decode")


def test_the_window_stops_short_of_the_body_preamble():
    assert PREROLL < BODY_PREAMBLE_SAMPLES, (
        f"a window of {PREROLL} reaches the body's ZC segment at "
        f"{BODY_PREAMBLE_SAMPLES}, which outbids the header's own preamble")


def test_the_two_bounds_have_not_met():
    """Reported rather than merely asserted: the headroom is small enough that
    knowing it has shrunk is worth more than knowing it is positive."""
    widest = (monitor.SEG_PAD + 1) * monitor.SEG_FRAME
    headroom = BODY_PREAMBLE_SAMPLES - widest
    assert headroom > 0, (
        "the segmenter's pre-roll now reaches the body preamble: the fast header "
        "needs the body excluded some other way, and widening the window cannot "
        f"do it (pre-roll {widest}, body preamble {BODY_PREAMBLE_SAMPLES})")
