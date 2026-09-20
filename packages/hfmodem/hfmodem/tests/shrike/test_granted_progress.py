# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A grant settles the packet it answers, and what the granted link carries after it.

KB5LZK, 2026-09-11 04:10 UTC, 3 597 000 kHz, arm R06
(`working/pactor-evening-en63bc-0910/`). The link keyed the seven-byte
announcement nine times and the gateway asked for it again every cycle; it then
granted PACTOR-3, read the entry packet and answered it -- and this station keyed
an EMPTY speed-level-3 field twenty-seven times against a held CS1, spent no
budget on any of them, and never reached the greeting the mail driver was
waiting on.

What that session was read as at the time, and what the reference tapes say:

  * the grant was fed to the FSM as a CS ACK, which settled a packet the peer had
    never taken. It IS the answer to that packet (pactor3.md 17.1) and settles it
    here -- and the announcement does not cross into PACTOR-3 either way, because
    neither reference caller's does;
  * so a granted link with nothing queued keys IDLE packets after its entry,
    which is what DL6MAA does for six acknowledged cycles before it loads a
    field, and is the state its gateway breaks in on;
  * and a repeat request spends no retry, so the loop had no end. That is what
    `arq.UNREAD_RUNG_REPEATS` bounds, and it is the whole of what ended R06.

Run:  python -m pytest hfmodem/tests/shrike/test_granted_progress.py
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, pactor1, placement
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_granted_entry_retry import grant
from hfmodem.tests.shrike.test_p3_entry_controls import recorded
from hfmodem.tests.shrike.test_p3_offer import (acknowledge, calling_station,
                                                cs_event, upgraded_station)

ANNOUNCEMENT = b"1w9ssj\r"
"""What `ptc.PtcHost._answer_link_setup` queues, and the only bytes R06 had."""

SL3_SAMPLES = 41219
"""82 symbols and the transmit filter ringing out, untrimmed. It was 43200 while
the data packets went out on the generic raised cosine, whose kernel is 3841 taps
against the protocol pulse's 1860 (`placement.PROTOCOL_RISE`)."""

LONG_TRAIN = 64
"""Repeat requests past any bound, so a test that waits for one says so by the
outcome rather than by naming the number."""


def unread_announcement():
    """The link at the instant KB5LZK granted: PACTOR-1 ISS, nine keyings of the
    announcement, every one of them answered with a request for it again."""
    host, keyed = calling_station()
    host.p1_grant_only = True
    for _ in range(8):
        host.on_rx_event(cs_event(pactor1.CS_SPEED))   # CS4 held = repeat request
        host.tick()
    assert [payload for _, payload in keyed.packets] == [ANNOUNCEMENT] * 9
    assert host.arq.tx_seq == 1 and not host.arq._outbuf
    return host, keyed


def test_the_grant_settles_the_packet_it_arrives_behind():
    """It arrives where the codeword answering that packet would, however many
    times the peer asked for it first, and nothing crosses into PACTOR-3 --
    W4DNA announced `1w4dna\r` in PACTOR-1 and keyed fresh bytes in its first
    PACTOR-3 field."""
    host, keyed = unread_announcement()
    host.on_rx_event(grant())
    assert not host.arq._outbuf
    assert host.arq._buffer_raw == 0
    assert host.protocol is Protocol.PACTOR3 and host.arq.entry_pending
    # pactor3.md, "a changeover restarts the counter": the entry packet is
    # counter 2, behind the PACTOR-1 phase's one data packet.
    assert host.arq.tx_seq == 2
    assert keyed.p3 == [b""]


QUEUED = b"pending application bytes"
"""Something behind the announcement, so "not requeued" reads as what the
granted link carries instead of as an absence."""


def test_a_grant_behind_a_read_announcement_does_not_requeue_it():
    """pactor3.md 17.1: the grant arrives where the codeword answering our
    packet would and acknowledges that packet. Requeued unconditionally, a
    first-try announcement nobody had asked again for rode the first PACTOR-3
    field into the peer's host port -- `1w9ssj\\r` at offset 0 of the mail the
    application sent, in all three audio rehearsals."""
    host, keyed = calling_station()
    host.p1_grant_only = True
    host.arq.on_host_data(QUEUED)
    assert bytes(host.arq._inflight.payload) == ANNOUNCEMENT
    assert not host.arq._inflight.retries and not host.arq._inflight.repeats
    host.on_rx_event(grant())
    assert bytes(host.arq._outbuf) == QUEUED
    assert host.arq._buffer_raw == len(QUEUED)
    assert host.protocol is Protocol.PACTOR3 and host.arq.entry_pending
    assert host.arq.tx_seq == 2 and keyed.p3 == [b""]
    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))
    assert keyed.p3 == [b"", QUEUED]


def test_the_first_pactor3_field_after_the_entry_is_idle_fill():
    """DL6MAA keyed six template-filled packets after its entry, all of them
    acknowledged, before it loaded a field at all (PIII_Complete_1, 9.113 to
    15.365 s). Ours used to key `1w9ss` here -- `1w9ssj\r` cut mid-callsign --
    and leave `j\r` to lead the login a turn later."""
    host, keyed = unread_announcement()
    host.on_rx_event(grant())
    host.tick()                             # R06's own cycle, so the evening
    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))  # runtime keys its
                                            # entry and fails on the empty field
    assert not host.arq.entry_pending and host.arq.speed_level == 3
    host.tick()
    assert keyed.p3 == [b"", b""]
    assert host.arq.tx_seq == 3 and host.arq._inflight.sl == 3
    assert not any(payload for payload in keyed.p3)


def test_a_peer_that_never_reads_the_new_waveform_is_bounded():
    host, keyed = unread_announcement()
    said = []
    host.log = said.append
    host.on_rx_event(grant())
    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))    # the entry is read
    host.tick()                             # ...and the slot behind it gets one
    for _ in range(LONG_TRAIN):
        host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))  # held CS1 = repeat
        if host.protocol is Protocol.PACTOR1:
            break
    assert host.protocol is Protocol.PACTOR1
    assert any(f"asked {arq.UNREAD_RUNG_REPEATS + 1} times for the first "
               f"PACTOR-3 data packet" in line for line in said)
    assert len(keyed.p3) == 2 + arq.UNREAD_RUNG_REPEATS
    # Nothing is owed on the retreat: the grant answered the announcement and
    # every packet behind it was idle, so no user byte went unacknowledged.
    assert not host.arq._inflight.payload and not host.arq._outbuf
    assert not any(keyed.p3)


def test_the_retreat_is_not_a_verdict_and_a_second_grant_is_taken():
    """Sixteen cycles of a fade says nothing about what a station can decode,
    so PACTOR-3 stays open and `_grant_taken` comes back off its latch."""
    host, keyed = unread_announcement()
    host.on_rx_event(grant())
    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))
    host.tick()
    for _ in range(LONG_TRAIN):
        host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))
        if host.protocol is Protocol.PACTOR1:
            break
    assert host.protocol is Protocol.PACTOR1 and not host._ruled_out
    keyed.packets.clear()
    host.on_rx_event(grant())
    assert host.protocol is Protocol.PACTOR3 and host.arq.entry_pending
    assert keyed.p3 == [b""]


def test_a_gear_command_restarts_the_rung_budget():
    """CS5 is the IRS working the ladder, which is progress: the level it
    hands us gets the whole allowance rather than what is left of it."""
    host, _ = unread_announcement()
    host.on_rx_event(grant())
    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))
    host.tick()
    for _ in range(arq.UNREAD_RUNG_REPEATS):
        host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))
    host.on_rx_event(cs_event(arq.CS_NAK, Protocol.PACTOR3))    # CS5: gear down
    assert host.arq.speed_level == 2 and host.arq._inflight.repeats == 1
    for _ in range(arq.UNREAD_RUNG_REPEATS):
        host.on_rx_event(cs_event(arq.CS_REQUEST, Protocol.PACTOR3))
        assert host.protocol is Protocol.PACTOR3


def test_an_uninvited_upgrade_is_not_bounded_by_the_rung_budget():
    """It keys no entry packet, so it has no entry answer to claim and none of
    `UNREAD_RUNG_REPEATS`'s reasoning applies; `UPGRADE_SILENCE_CYCLES` is its
    budget. The bound used to arm here and report an entry nobody keyed."""
    host, _ = upgraded_station()
    said = []
    host.log = said.append
    assert host.protocol is Protocol.PACTOR3 and not host.arq.entry_pending
    for _ in range(LONG_TRAIN):
        host.on_rx_event(cs_event(arq.CS_REQUEST, Protocol.PACTOR3))
    assert host.protocol is Protocol.PACTOR3
    assert not any("entry packet" in line for line in said)


def test_an_acknowledged_field_leaves_repeat_requests_unbudgeted():
    """The measured rule the bound must not reach (WS8EOC, 2026-08-09): a repeat
    request is the reverse channel working, not a failed receive."""
    host, _ = unread_announcement()
    host.on_rx_event(grant())
    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))      # entry read
    host.tick()                                   # the idle packet behind it
    host.on_rx_event(cs_event(arq.CS_REQUEST, Protocol.PACTOR3))  # counter 3 taken
    host.tick()                                   # ...and the slot gets another
    assert host.arq.tx_seq == 0 and not host.arq._outbuf
    assert not host.arq._entry_keyed        # the rung carried; the bound is off
    for _ in range(LONG_TRAIN):
        host.on_rx_event(cs_event(arq.CS_REQUEST, Protocol.PACTOR3))
    # The requests are still counted -- `on_rx_grant` reads that counter -- and
    # with the rung carried they buy nothing: no retry, no fallback.
    assert host.protocol is Protocol.PACTOR3 and not host.arq._inflight.retries


def test_a_grant_with_nothing_to_carry_keys_an_idle_packet_and_waits():
    """Deliberate, and the guard `upgrade` applies to an UNINVITED offer does not
    reach it: the peer spent its own answer slot asking for the waveform. What it
    gets is an idle packet, and the IRS takes the channel off that with the CS3
    the reference peer answers an entry packet with (pactor3.md)."""
    host, keyed = calling_station()
    host.p1_grant_only = True
    acknowledge(host)                       # the announcement is delivered
    assert not host.arq._outbuf and host.arq._inflight is None
    host.on_rx_event(grant())
    assert host.protocol is Protocol.PACTOR3 and keyed.p3 == [b""]
    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))
    host.tick()
    assert keyed.p3 == [b"", b""]           # an idle field, not a forged one
    host.on_rx_event(cs_event(arq.CS_BREAKIN, Protocol.PACTOR3))
    assert host.arq.role == arq.IRS and host.protocol is Protocol.PACTOR3


def test_the_recorded_gateway_answer_releases_the_entry_packet():
    """R06 again, with KB5LZK's own two answers off the air in place of the
    scripted ones -- `fixtures/p3-entry-cs/kb5lzk-arm09-hold{09,10}.wav`, the
    windows `test_p3_entry_controls` corroborates the entry from. The repeat
    train behind them is scripted: one physical window cannot answer two cycles,
    and the receiver refuses to let it (`_SessionRx`)."""
    host, keyed = unread_announcement()
    host.on_rx_event(grant())
    rx = onair._SessionRx(host, tag="R06")
    for hold in (9, 10):
        row, audio = recorded(f"kb5lzk-arm09-hold{hold:02}.wav")
        rx.new_cycle()
        rx.control_signal(audio, row["start_stream_sample"],
                          row["anchor_stream_sample"])
    assert not host.arq.entry_pending and host.arq.speed_level == 3
    host.tick()
    assert keyed.p3 == [b"", b""]
    assert host.arq.tx_seq == 3


@pytest.mark.parametrize("swapped", [False, True])
def test_a_speed_level_three_packet_renders_one_length(swapped):
    """Two calls, because the R06 log reads 0.863/0.864/0.870/0.871 for one
    packet shape and the renderer has no term that could produce it."""
    rendered = [len(placement.link_packet(3, b"", 0x1b, swapped=swapped))
                for _ in range(2)]
    assert rendered == [SL3_SAMPLES, SL3_SAMPLES]


def test_the_logged_duration_is_the_trim_and_not_the_render():
    """It is `onair._trim_silence`, which cuts a field-dependent tail off a packet
    that is always the same length -- 0.859 s of render since the data packets
    took the protocol's own pulse, 0.900 s before it. The head it cuts is stable
    to two samples, so the phase reference the peer times us against does not move
    with it -- the duration in the log is not a packet identity."""
    packets = [placement.link_packet(3, b"", status, swapped=swapped)
               for status in (0x1b, 0x5b) for swapped in (False, True)]
    assert {len(audio) for audio in packets} == {SL3_SAMPLES}
    heads = [int(np.flatnonzero(np.abs(a) > .02 * np.abs(a).max())[0])
             for a in packets]
    assert max(heads) - min(heads) <= 2
    assert len({len(onair._trim_silence(a)) for a in packets}) > 1
