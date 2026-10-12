# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded structure evidence on an already accepted P3 packet raster."""
import numpy as np
from . import p3acquire, p3frame, p3rx, placement, rxfront


def tracked_structure(audio, origin, target, levels, *, changeover=False, gaps=()):
    """Return a presence source, never a packet, byte or clock acceptance.

    Input is corrected at the receiver's existing carrier hypothesis. Only
    the current form and emitted-command candidates may be supplied as levels.
    """
    fs, sps = rxfront.FS, rxfront.SPS
    lead = p3frame.DATA_OFFSET * sps
    phase = target - lead
    if changeover:
        lo, hi = phase - round(.08 * fs), phase + round(.31 * fs)
        if origin > lo or origin + len(audio) < hi:
            return None
        if any(a < hi and b > lo for a, b in gaps):
            return None
        got = p3acquire.changeover(audio[lo-origin:hi-origin], offsets=(0.,))
        if got is not None and abs(lo + got.event.start - phase) <= sps:
            return 'tracked changeover'
        return None
    for sl in dict.fromkeys(levels):
        if sl not in placement.SPEED_PATHS:
            continue
        home = placement.SPEED_PATHS[sl]
        # Preserve the same pre-header matched-filter context as the
        # tracked packet reader; a one-symbol crop truncates its pulse.
        lo = phase - 2400
        if lo < origin:
            continue
        part = audio[lo-origin:]
        at = target-lo
        starts = range(at-sps, at+sps+1, sps//4)
        idx = np.arange(starts.start-lead, starts[-1]+1, sps//4) + rxfront.MATCHED_DELAY_N
        idx = np.unique(np.add.outer(idx, home.clock_offsets(sps)).ravel())
        if idx[-1] >= len(part):
            continue
        header_end = lo + int(idx[-1]) + 1
        if any(a < header_end and b > lo for a, b in gaps):
            continue
        z = rxfront._sampled_baseband(part[:header_end-lo], home.tones, idx)
        header = p3rx.header_of(z, starts, home)
        if header is None or header.fit < p3rx.anchor_gate(home) or sl not in header.levels:
            continue
        row = header.at + lead
        path = p3rx.path_for(sl, header)
        end = lo + row + rxfront._frame_span(path) - rxfront.UNREAD_TAIL_N + 1
        if origin + len(audio) < end or any(a < end and b > lo for a, b in gaps):
            continue
        if sl == 1:
            # This predicate needs real baseband statistics, not a sparse
            # array whose unvisited positions are padded zeros.
            pulse = p3rx.rx._pulse(sps)
            field = part[:end-lo]
            z = {cn: p3rx.rx._baseband(field, cn, fs, pulse) for cn in home.tones}
            if not p3rx.sl1_carriers_present(z, row, fs=fs):
                continue
        return f'tracked SL{sl} ' + ('long header' if header.long_cycle else 'short header')
    return None
