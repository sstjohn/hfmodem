# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Each named receive ordering requires two independently read non-QAM bodies."""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varaofdm as ofdm
from hfmodem.kestrel.tx import varatone_tx as tone_tx
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
from .receive_window_helpers import full_body


@pytest.mark.parametrize('pin,level', [
    ('critic-A', 10), ('hint-R1', 10), ('hint-R2', 10), ('hint-R4', 10),
    ('hint-R5', 10), ('hint-R5b', 1), ('hint-R5c', 10), ('hint-R5d', 10),
    ('hint-R11', 7), ('hint-C1', 2), ('hint-C2', 3),
])
def test_named_double_reception(pin, level):
    bw, caller = '2300', 'K7NEX'
    size = phy.body_size(bw, phy.level_key(bw, level))
    payloads = (bytes(i % 251 for i in range(size - 1)),
                bytes((i + 71) % 251 for i in range(size - 1)))
    bodies = tuple(full_body(p, caller, size, 0x99) for p in payloads)
    if level <= 4:
        from hfmodem.kestrel.rx import varahf2300 as rx
        from hfmodem.kestrel.tx import varahf2300_tx as tx
        key = phy.level_key(bw, level)
        record = rx.RECORDS[key]
        first, second = [tx.synth_burst(body, key, i) for i, body in enumerate(bodies)]
        audio = np.r_[first, second[-record.ncols * record.dw50:]]
    elif level == 7:
        from hfmodem.kestrel.coding.crc import crc16_genibus
        record = tone_tx.ROWS[int(bw), level]
        frames = [b + crc16_genibus(b).to_bytes(2, 'big') for b in bodies]
        first = tone_tx.synth_frame(frames[0], int(bw), level, VB6Rnd(7))
        cells = tone_tx.cells(frames[1], record, VB6Rnd(17))
        cells[:, :tone_tx.C0] = 0
        second = tone_tx.render(cells, record)
        second *= tone_tx.RMS / np.sqrt(np.mean(second ** 2))
        shift = (record.nsym - tone_tx.C0) * tone_tx.H
        audio = np.pad(first, (0, shift))
        audio[shift:shift + len(second)] += second
    else:
        first, second = [phy.render_level((body,), bw, level, VB6Rnd(i + 10))
                         for i, body in enumerate(bodies)]
        audio = np.r_[first, second[ofdm.PSK_LEAD * ofdm.SYM:]]
    read = phy.decode_level(audio, bw, level)
    assert read is not None
    assert tuple(read.frames) == bodies, pin
    assert b''.join(phy.vara_payload(b, caller=caller, body_len=size)
                    for b in read.frames) == b''.join(payloads)
