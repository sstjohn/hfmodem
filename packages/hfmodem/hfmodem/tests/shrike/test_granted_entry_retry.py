# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Granted entry must not spend silence budgets on a readable grant train."""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests.shrike.recorded_pcm import recorded_pcm

from hfmodem.shrike import arq, onair, pactor1, rxfront
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p3_offer import calling_station, cs_event

GRANTS_PATH = Path(__file__).with_name("fixtures") / "ve3kpg-0909-grants.json"


def grant():
    return rxfront.Event(0.1, "unassigned", "0x59A", protocol=Protocol.PACTOR1,
                         spare=pactor1.CS_59A, sense=0)


def granted():
    host, keyed = calling_station()
    host.p1_grant_only = True
    host.arq.on_host_data(b"pending application bytes")
    host.on_rx_event(grant())
    host.tick()  # Finish the initial grant/entry cycle before its replies arrive.
    assert host.arq.entry_pending
    return host, keyed


def repeat(host):
    host.on_rx_event(grant())
    host.tick()


def test_four_grants_do_not_abandon_the_only_entry_waveform():
    host, keyed = granted()
    packet = host.arq._inflight
    queued = bytes(host.arq._outbuf)
    for _ in range(4):
        repeat(host)
    assert host.protocol is Protocol.PACTOR3
    assert host.arq.entry_pending and host.arq._inflight is packet
    assert len(keyed.p3) == 5
    assert bytes(host.arq._outbuf) == queued
    assert packet.retries == 1  # Each grant restarts the packet retry budget.


def test_grants_separate_silent_runs_and_allow_a_late_breakin():
    host, _ = granted()
    for _ in range(2):
        for _ in range(3):
            host.tick()
        repeat(host)
        assert host.protocol is Protocol.PACTOR3
        assert host.arq._unanswered_upgrade == 0
        assert host.arq._inflight.retries == 1
    host.on_rx_event(cs_event(arq.CS_BREAKIN, Protocol.PACTOR3))
    assert host.arq.role == arq.IRS
    assert not host.arq.entry_pending
    assert not host.arq.upgrade_unanswered


def test_entry_requests_remain_bounded_without_claiming_incompatibility():
    host, _ = granted()
    messages = []
    host.log = messages.append
    for _ in range(arq.ENTRY_GRANT_CYCLES - 1):
        repeat(host)
        assert host.protocol is Protocol.PACTOR3
    repeat(host)
    assert host.protocol is Protocol.PACTOR1
    assert not host.arq.entry_pending
    assert any("entry retry budget exhausted" in line for line in messages)
    assert not any("never once asked" in line or "detected one" in line
                   for line in messages)


def test_a_permanently_granting_peer_reaches_the_new_floor():
    """2026-09-11: the 40 m WS8EOC arm never saw the peer's
    PACTOR-1 answer position go quiet because the old eight-cycle limit ended
    the campaign two cycles before the point the 80 m arm saw it happen at.
    The campaign now has to survive to at least fourteen keyed entries."""
    host, _ = granted()
    for _ in range(arq.ENTRY_GRANT_CYCLES - 1):
        repeat(host)
    assert host.protocol is Protocol.PACTOR3
    assert host.arq.entry_pending
    assert host.arq._upgrade_requests == arq.ENTRY_GRANT_CYCLES - 1


def test_a_permanently_granting_peer_still_ends_the_campaign():
    """NEGATIVE CONTROL: raising the floor must not make the campaign
    open-ended -- a peer that never confirms the entry still has to be let go."""
    host, _ = granted()
    for _ in range(4 * arq.ENTRY_GRANT_CYCLES):
        repeat(host)
        if host.protocol is Protocol.PACTOR1:
            break
    else:
        raise AssertionError("a permanently granting peer never ended the campaign")
    assert not host.arq.entry_pending


def test_four_consecutive_silent_entry_cycles_still_fall_back():
    host, _ = granted()
    for _ in range(3):
        host.tick()
        assert host.protocol is Protocol.PACTOR3
    host.tick()
    assert host.protocol is Protocol.PACTOR1


def test_silence_after_grants_does_not_erase_the_recorded_answers():
    host, _ = granted()
    repeat(host)
    for _ in range(4):
        host.tick()
    assert host.protocol is Protocol.PACTOR1
    ending = next(line for line in host.log_lines if "falls back" in line)
    assert "After 1 repeated grant," in ending
    assert "consecutive unanswered" in ending
    assert "current silent run" in ending


def test_refused_transmissions_do_not_exhaust_entry_budgets():
    # Six total refusals stay below the separate eight-refusal placement ceiling.
    # This pins the two air budgets; the permanent-refusal test pins the ceiling.
    host, _ = granted()
    for _ in range(3):
        host.arq._refused_burst = True  # Previous attempt missed its key slot.
        repeat(host)
        assert host.protocol is Protocol.PACTOR3
        assert host.arq._upgrade_requests == 0
        assert not host.arq._refused_burst
    for _ in range(3):
        host.arq._refused_burst = True
        host.tick()
        assert host.protocol is Protocol.PACTOR3
        assert host.arq._unanswered_upgrade == 0


def test_transmit_seam_refusal_then_success_charges_only_keyed_entries():
    host, keyed = calling_station()
    send = keyed.send_packet
    keyed.send_packet = lambda *args, **kwargs: arq.REFUSED
    host.on_rx_event(grant())
    host.tick()
    for _ in range(6):
        repeat(host)
        assert host.protocol is Protocol.PACTOR3
        assert host.arq._upgrade_requests == 0
    keyed.send_packet = send
    repeat(host)  # First entry that the transmitter actually accepts.
    assert host.arq._upgrade_requests == 0
    repeat(host)  # Its answer can now spend one request.
    assert host.arq._upgrade_requests == 1
    assert host.arq._inflight.retries == 1


def test_optional_next_entry_still_starts_after_four_requests():
    host, _ = granted()
    host.arq.cfg.entry_ladder = ("template", "data")
    for _ in range(4):
        repeat(host)
    assert host.protocol is Protocol.PACTOR3
    assert host.arq.entry_variant == "data"
    for _ in range(arq.ENTRY_GRANT_CYCLES - 1):
        repeat(host)
        assert host.protocol is Protocol.PACTOR3
    repeat(host)
    assert host.protocol is Protocol.PACTOR1


@pytest.mark.skipif(not GRANTS_PATH.exists(),
                    reason=f"the recorded VE3KPG grants {GRANTS_PATH.name} "
                           "are not installed")
def test_recorded_ve3kpg_grants_keep_the_entry_in_flight():
    """Replay the actual initial grant and four replies that caused TX18 P1."""
    rows = json.loads(GRANTS_PATH.read_text())
    host, keyed = calling_station()
    host.p1_grant_only = True
    for row in rows:
        samples = recorded_pcm(row)
        event = onair._SessionRx._p1_cs(samples, row["anchor"])
        assert event is not None and event.spare == pactor1.CS_59A
        assert event.sense == row["sense"]
        assert onair._SessionRx._p1_cs(np.zeros_like(samples),
                                      row["anchor"]) is None
        host.on_rx_event(event)
        host.tick()
        assert host.protocol is Protocol.PACTOR3
        assert host.arq.entry_pending
    assert len(keyed.p3) == 5  # Old implementation sent only four, then P1.
    assert host.arq._inflight.retries == 1


def test_permanent_entry_refusals_end_without_spending_air_budgets():
    for answered in (False, True):
        host, keyed = granted()
        keyed.send_packet = lambda *a, **kw: arq.REFUSED
        for _ in range(200):
            if answered:
                host.on_rx_event(grant())
            host.tick()
            if host.protocol is Protocol.PACTOR1:
                break
        else:
            raise AssertionError("a permanently refused entry never ended")
        assert any("placement budget exhausted" in line for line in host.log_lines)


def test_silence_diagnostics_keep_collision_evidence_after_a_grant():
    host, _ = granted()
    repeat(host)
    for _ in range(4):
        host.arq.note_burst(-38)
        host.tick()
    assert any("WE WERE TRANSMITTING OVER THE ANSWER" in line for line in host.log_lines)


def test_a_granted_entry_replaces_a_refused_p1_send_before_the_answered_tick():
    host, keyed = calling_station()
    keyed.send_p1_packet = lambda *a, **kw: arq.REFUSED
    host.tick()
    assert host.arq._refused_burst
    # The grant starts a new packet through _on_ack, bypassing _on_nak's
    # refusal consumption. The answered tick must not retain the old P1 refusal.
    host.on_rx_event(grant())
    host.tick()
    assert host.protocol is Protocol.PACTOR3 and keyed.p3
    assert not host.arq._refused_burst
    repeat(host)
    assert host.arq._upgrade_requests == 1
