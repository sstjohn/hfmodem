# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The first loaded PACTOR-3 field a granted link keys, against the references.

Two stations key a first loaded packet on tape and both raise STATUS BIT 5 on
it: DL6MAA at speed level 3 (`rf-corpus/PIII_Complete_1.wav`, 16.613 s, status
0x33, a full 59-byte field) and W4DNA at speed level 1
(`7101k_234600.wav`, 21.185 s, status 0x21,
`Q-6.0` filling the field exactly). Ours was 0x03: the ask was gated on the rung
having a long frame, and PACTOR-3 speed level 1 -- where every granted phase
opens -- has none, so the packet behind a grant could never raise it.

Neither reference carries its PACTOR-1 ANNOUNCEMENT into PACTOR-3 either. W4DNA
announced `1w4dna\r` in PACTOR-1 and put five fresh bytes in its first PACTOR-3
field; DL6MAA keyed six acknowledged template-filled packets after its entry
before it loaded one at all. Ours keyed `1w9ss` -- `1w9ssj\r` cut mid-callsign --
and left `j\r` to lead the login a turn later.

Run:  python -m pytest hfmodem/tests/shrike/test_first_loaded_packet.py
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import arq, placement, rx, spec
from hfmodem.tests.shrike.test_granted_entry_retry import grant
from hfmodem.tests.shrike.test_granted_progress import unread_announcement
from hfmodem.tests.shrike.test_no_p3_fallback import acknowledge, entered
from hfmodem.tests.shrike.test_p3_offer import cs_event

QUEUED = b"the application bytes the entry packet was keyed for, " * 3
"""More than one speed-level-3 field, so the train has a first loaded packet
with traffic behind it and a last one that empties the buffer."""


def loaded(tx, mark):
    """Every PACTOR-3 data packet keyed since `mark`, as (payload, status)."""
    return [(row[2], row[3]) for row in tx.emissions[mark:]
            if row[0] is spec.Protocol.PACTOR3 and row[1] == "packet"]


def test_a_level_one_field_carries_bit_five_through_the_receiver():
    """The transmitter's own round trip: what `placement.link_packet` builds at
    the floor decodes back with the bit up and the CRC valid. W4DNA's field is
    the same shape, and rebuilds here at 0 of 144 code cells."""
    status = spec.status_byte(3, long_cycle_request=True)
    assert status & spec.STATUS_LONG_CYCLE
    audio = placement.link_packet(1, b"1w9ss", status)
    pad = np.zeros(spec.SAMPLE_RATE // 2)
    fields = [f for _, f in rx.case0_accepts(np.concatenate([pad, audio, pad]))]
    assert fields, "the packet no longer decodes at all"
    assert all(f[-3] == status for f in fields), [f.hex() for f in fields]
    assert all(spec.field_payload(f[:-3]) == b"1w9ss" for f in fields)


def test_the_first_loaded_packet_behind_a_grant_raises_bit_five():
    host, tx, mark = entered(payload=QUEUED)
    entry = [row for row in tx.emissions[mark:] if row[1] == "entry"]
    assert entry and not entry[0][3] & spec.STATUS_LONG_CYCLE
    acknowledge(host)                       # the peer reads the entry
    first, status = loaded(tx, mark)[0]
    assert status & spec.STATUS_LONG_CYCLE, f"0x{status:02x}"
    assert first and QUEUED.startswith(first)


def test_an_idle_packet_does_not():
    """Bit 5 travels with a loaded field. DL6MAA's template-filled packets carry
    0x19/0x1a and our own entry 0x1a, all four with the bit clear."""
    host, tx, mark = entered(payload=QUEUED)
    acknowledge(host)
    for _ in range(20):
        if not host.arq._outbuf and host.arq._inflight is None:
            break
        acknowledge(host)
    assert not host.arq._outbuf
    idle = len(tx.emissions)
    host.tick()
    keyed = loaded(tx, idle)
    assert keyed and not any(payload for payload, _ in keyed)
    assert not any(status & spec.STATUS_LONG_CYCLE for _, status in keyed)


def test_the_bit_drops_on_the_packet_that_empties_the_buffer():
    """DL6MAA holds it through every loaded packet from 16.613 s to 44.115 s and
    clears it at 47.865 s (0x10) on the partial field that runs its buffer
    down."""
    host, tx, mark = entered(payload=QUEUED)
    acknowledge(host)
    for _ in range(20):
        if not host.arq._outbuf and host.arq._inflight is None:
            break
        acknowledge(host)
    keyed = loaded(tx, mark)
    carried = [(payload, status) for payload, status in keyed if payload]
    assert carried
    assert b"".join(payload for payload, _ in carried) == QUEUED
    assert not carried[-1][1] & spec.STATUS_LONG_CYCLE
    assert all(status & spec.STATUS_LONG_CYCLE for _, status in carried[:-1])


def test_a_repeat_requested_announcement_does_not_lead_the_loaded_field():
    """The grant is the answer to the packet it arrives behind, however many
    times the peer asked for that packet first -- nine times at KB5LZK on
    2026-09-11 -- so the application's own bytes lead the first loaded field
    from their first byte rather than following `1w9ss`."""
    host, keyed = unread_announcement()
    host.arq.on_host_data(QUEUED)
    host.on_rx_event(grant())
    host.tick()
    host.on_rx_event(cs_event(arq.CS_ACK, spec.Protocol.PACTOR3))
    assert keyed.p3[0] == b""               # the entry packet
    assert keyed.p3[1] == QUEUED[:len(keyed.p3[1])]
    assert not any(b"1w9" in payload for payload in keyed.p3)
