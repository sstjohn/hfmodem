# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The SL3 field a gateway repeated for forty seconds, read through the live path.

K0NTS, 40 m, 2026-09-15 23:57Z (`arm-post-v31-B-assessed-40-k0nts-20260915T235733Z`).
After our second CS4 the gateway went to speed level 3 with counter 1, carrying
60 bytes, and repeated them until past our teardown. The arm logged PEER'S PACKET
UNREAD on every cycle and QRT'd with `rx_seq` still 0, while the same windows
decode CRC-valid offline, byte-exact against the reference decoder, off the
arm's own capture and
off an independent monitor alike. So the demodulator and the radio input were
never in question: what failed was acquisition.

Fourteen carriers cannot reach a header gate averaged over eighteen channels, and
below that gate the only reader allowed on the block was given one CRC, on the
weak block's own carrier order, with this cycle's softs thrown away afterwards --
against a gateway that alternates the order and a field that needed the copies
summed.

The fixture is the arm's OWN hold windows, cut from its own `stream.wav` at the
boundaries the arm read them at, so what runs here is the window the receiver had
and not one chosen afterwards.

Run:  python -m pytest hfmodem/tests/shrike/test_k0nts_sl3_0915.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3frame, p3rx, placement, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session

FIXTURE = Path(__file__).with_name("fixtures") / "k0nts-sl3-0915"
PAYLOAD = b"ing ec2-34-230-165-191.compute-1.amazonaws.com\r*** W9SSJ Co"
STATUS = 0x21
"""The reference decoder's `###STATUS: SL: 3 ... LEN: 60`: sequence 1, and the long-cycle request."""


@pytest.fixture(scope="module")
def holds():
    if not (FIXTURE / "metadata.json").exists():
        pytest.skip(f"no {FIXTURE.name} cuts under {FIXTURE.parent}")
    meta = json.loads((FIXTURE / "metadata.json").read_text())
    out = []
    for row in meta["rows"]:
        path = FIXTURE / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        fs, pcm = wavfile.read(path)
        assert fs == onair.FS and len(pcm) == row["end"] - row["start"]
        out.append((row, pcm.astype(float) / 32768))
    return meta, out


def receiver(meta):
    """The arm's own state at its last CRC-valid changeover."""
    s = _Session(role=arq.IRS)
    s.rx.p3_wideband_prekey = True
    s.rx._seed_p3_changeover_clock(meta["seed_changeover"])
    s.rx._p3_clock_role = arq.IRS
    s.rx.sync.packet_level = 1
    s.rx.p3_receive_offset_hz = meta["correction_hz"]
    s.host.arq._rx_seen = True
    s.host.arq._expected_seq = 1
    s.host.arq.cfg.long_cycle = False
    return s


def replay(meta, holds, session=None):
    s = session or receiver(meta)
    delivered = []
    for row, pcm in holds:
        s.rx.new_cycle()
        before = len(s.packets)
        onair._scan_frame(s.rx, pcm, row["start"], tracked_only=True)
        delivered += [(row["hold"], ev) for ev in s.packets[before:]]
    return s, delivered


def test_the_repeated_field_reaches_the_host_and_advances_the_counter(holds):
    """Two of eight cycles deliver, which is what the copies are worth.

    The gateway sent one field over and over, so the copies sum: holds 33 and 37
    are the two the arm's own windows carry far enough for the CRC. Both return
    the same sixty bytes, and once the counter moves the station stops asking for
    the packet it has -- CS2 where it had keyed CS1 for forty seconds.
    """
    meta, rows = holds
    s, delivered = replay(meta, rows)
    assert [h for h, _ in delivered] == [33, 37]
    for _, ev in delivered:
        assert ev.packet == (3, STATUS, PAYLOAD, True)
        assert "SL3" in ev.text and "short fallback" in ev.text
    assert s.host.arq.rx_seq == 1
    assert s.host.peer.sent[-1] == ("cs", arq.CS_REQUEST)


def test_without_soft_combining_the_same_windows_deliver_nothing(holds, monkeypatch):
    """The fails-without control, and the arm's own record.

    Every copy reached the CRC and none passed it alone. Discarding them is
    exactly what the live arm did, and it is the whole of the stall.
    """
    meta, rows = holds
    monkeypatch.setattr(rxfront.SyncedRx, "_wideband_combined",
                        lambda *a, **k: None)
    s, delivered = replay(meta, rows)
    assert delivered == []
    assert s.host.arq.rx_seq == 0


def test_no_header_block_here_ever_clears_its_own_gate(holds):
    """The gate was never lowered; the declared fallback is what read these.

    Scored on speed level 3's own twelve constant-header channels -- which is
    already the better of the two readings, and what `p3rx.HEADER_COMBS` now
    gives the acquisition sweep as well -- the best block of the eight reaches
    0.785 against 0.80. Selective fading took the eight-symbol uncoded block
    while the 72-row coded field behind it still decodes.
    """
    meta, rows = holds
    gate = p3rx.anchor_gate(placement.SPEED_PATHS[3])
    sync = rxfront.SyncedRx()
    fits = []
    for row, pcm in rows:
        k = (row["end"] - meta["seed_changeover"]
             - p3frame.DATA_OFFSET * rxfront.SPS) // 60000
        row0 = (meta["seed_changeover"] + p3frame.DATA_OFFSET * rxfront.SPS
                + k * 60000)
        at = row0 - row["start"]
        header = sync._wideband_header_candidate_at(
            onair.p3acquire.compensate(pcm, meta["correction_hz"]), at)
        assert header is not None
        fits.append(header.fit)
    assert max(fits) < gate, max(fits)


def test_a_wideband_window_carries_the_matched_filters_reach(holds):
    """The read closes late enough for the last row to come off the air.

    `_sampled_baseband` reads past the end of a buffer as silence, so a window
    that stops on the packet's last sample demodulates its last rows out of the
    filter's own padding. The K0NTS holds stopped 613-901 samples inside that
    row. The term is still bounded by the admitted early tail and by the key.
    """
    meta, _ = holds
    s = receiver(meta)
    row0 = s.rx._p3_row0
    packet = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[3]))
    reach = (packet + rxfront.MATCHED_DELAY_N
             - rxfront.SyncedRx.WIDEBAND_EARLY_N + rxfront.SPS // 2)
    deadline = row0 + reach
    assert onair._p3_wideband_target(s.rx, deadline) == row0
    assert onair._p3_frame_ready(s.rx, deadline) == deadline
    # ...and never past the deadline it was handed, whatever the reach wants.
    assert onair._p3_frame_ready(s.rx, row0 + 1) <= row0 + 1
