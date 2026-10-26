# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An IRS cycle's header and cold control reads, taken while it still listens.

0923-1321, KB5LZK-30: on the IRS holds whose tracked read missed its CRC, the
long-header read and the cold control read (anchored word, then acquisition)
stacked in front of the key and 7 of 15 slots were handed back. `_read_ahead`
takes both from the slices `_listen_until_answer` hands it; the pre-key path
then finds them done and decodes exactly what it decoded before.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session

FS = 48000
ARM = Path(__file__).parents[5] / 'captures' / 'onair-0923-1321'
recorded = pytest.mark.skipif(not ARM.is_dir(), reason='needs captures/onair-0923-1321')


class Keyed:
    """The raster as the reads see it: our last control, keyed and not sending."""

    sending = False
    keyed_slot = 1

    def __init__(self, started, emitted_end, due):
        self._p3_keyed_reply = (1, started, emitted_end, 60000)
        self.due = due

    def rx_due(self, slot):
        return self._p3_keyed_reply[1] + self.due

    def rx_due_in(self, lo, hi):
        return None


def irs(row0):
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -13.6
    s.rx._p3_row0 = row0
    s.rx.sync.packet_level = 1
    return s


def reads(monkeypatch):
    """Every call of the three readers the pre-key path used to pay for."""
    made = []
    for owner, name in ((rxfront.SyncedRx, 'control_signal_at'),
                        (rxfront.SyncedRx, 'sl1_header_at'),
                        (p3acquire, 'control_signal')):
        fn = getattr(owner, name)

        def run(*a, _fn=fn, _name=name, **k):
            made.append(_name)
            return _fn(*a, **k)
        monkeypatch.setattr(owner, name, run)
    return made


def pre_key(s, pcm, origin, g):
    header = onair._p3_long_header(s.rx, pcm, origin)
    heard = s.rx.control_signal_in(pcm, origin, g)
    return (header and (header[0], repr(header[1])), heard,
            [ev.text for ev in s.events])


@recorded
@pytest.mark.parametrize('n,row0,due,taken', [
    # The arm's own anchor, 23411 past our key (its replay): the word misses
    # and the acquisition takes the peer's repeated changeover, body and all.
    (36, 3947743, 23411, []),
    # Anchored on that head instead: the word is found on the cut and read
    # again over the whole window, which is where its body is.
    (36, 3947743, 17291, ['control_signal_at']),
    # Nothing there for either.
    (40, 4247773, 23411, []),
])
def test_the_pre_key_reads_are_taken_while_the_irs_listens(monkeypatch, n, row0,
                                                            due, taken):
    meta = json.loads((ARM / f'hold_{n:02d}.json').read_text())
    pcm = onair._load(str(ARM / f'hold_{n:02d}.flac')).astype(np.float32)
    origin = meta['end_stream_sample'] - pcm.size
    g = Keyed(origin - 11341, origin, due)

    before = pre_key(irs(row0), pcm, origin, g)

    s = irs(row0)
    lo, hi, at = s.rx._p3_reply_bracket(g, origin, origin + pcm.size)
    cut = s.rx._p3_cold_cut(at, hi, lo)
    assert cut is not None and lo + cut < origin + pcm.size - round(.25 * FS)
    step = round(.25 * FS)
    for end in range(step, pcm.size + step, step):
        onair._read_ahead(s.rx, g, pcm[:end], origin)
        if end >= cut:
            assert any(key[0] == 'word' for key in s.rx._p3_ahead)
    made = reads(monkeypatch)
    after = pre_key(s, pcm, origin, g)

    assert after == before
    assert made == taken
