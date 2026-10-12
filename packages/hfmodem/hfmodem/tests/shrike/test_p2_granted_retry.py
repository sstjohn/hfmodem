"""WS8EOC 2026-09-20 23:25Z: repeated 0x6A9 is not silence.

The first response at 7.36 s opened P2. Subsequent slots were 0x6A9,
unread, 0x6A9, 0x6A9 (8.61/9.86/11.11/12.36 s). Previously that
sequence exhausted four 'silent' cycles and abandoned the P2 waveform.
This is a replay of decoded events, not an independent waveform decode.
"""
import pytest

from hfmodem.shrike import arq, onair, pactor1
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p2_policy import grant, linked
from hfmodem.tests.shrike.test_p3_offer import cs_event


def granted():
    host, seam = linked()
    host.on_rx_event(grant(pactor1.CS_6A9))
    host.tick()
    assert host.protocol is Protocol.PACTOR2
    return host, seam


def repeat(host):
    host.on_rx_event(grant(pactor1.CS_6A9))
    host.tick()


def test_ws8eoc_recorded_reply_sequence_keeps_the_p2_entry_alive():
    host, seam = granted()
    packet = host.arq._inflight
    for received in (True, False, True, True):
        if received:
            host.on_rx_event(grant(pactor1.CS_6A9))
        host.tick()
    assert host.protocol is Protocol.PACTOR2
    assert host.arq._inflight is packet
    assert host.arq._upgrade_requests == 3
    assert host.arq._unanswered_upgrade == 0
    assert not any('silence rule' in line for line in host.log_lines)
    assert not seam.p3


def test_p2_repeated_grants_exhaust_a_finite_entry_budget_without_acking_data():
    host, seam = granted()
    host.arq.on_host_data(b'pending mail')
    queued = bytes(host.arq._outbuf)
    packet = host.arq._inflight
    for _ in range(arq.ENTRY_GRANT_CYCLES - 1):
        repeat(host)
        assert host.protocol is Protocol.PACTOR2
        assert host.arq._inflight is packet
        assert bytes(host.arq._outbuf) == queued
    repeat(host)
    assert host.protocol is Protocol.PACTOR1
    assert not host.arq.upgrade_unanswered
    assert any('entry retry budget exhausted' in line for line in host.log_lines)
    assert not any('P3 entry confirmation' in line or 'silence rule' in line
                   for line in host.log_lines)
    assert not seam.p3


@pytest.mark.parametrize('wrong_word', [False, True])
def test_genuine_silence_and_unselected_grants_still_time_out(wrong_word):
    host, _ = granted()
    for _ in range(arq.UPGRADE_SILENCE_CYCLES):
        if wrong_word:
            host.on_rx_event(grant(pactor1.CS_59A))
        host.tick()
    assert host.protocol is Protocol.PACTOR1
    assert any('silence rule' in line for line in host.log_lines)


def test_p2_ack_confirms_the_entry_and_ends_the_upgrade_budget():
    host, _ = granted()
    repeat(host)
    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR2))
    assert host.protocol is Protocol.PACTOR2
    assert not host.arq.upgrade_unanswered
    assert not host.arq._upgrade_granted


def test_abort_and_reconnect_reset_grant_provenance():
    host, _ = granted()
    assert host.arq._upgrade_granted
    host.arq.on_host_abort()
    assert not host.arq._upgrade_granted


def test_native_p2_keyings_are_reported_even_after_fallback():
    grid = onair._MasterGrid(0, round(1.25 * onair.FS), 0,
                            packet_n=round(.960 * onair.FS),
                            cs_n=round(.120 * onair.FS),
                            d_max_n=onair._d_max_n(1.25, .04))
    assert 'NO ENTRY PACKET WAS KEYED' in grid.entry_verdict()
    for cycle in range(6, 10):
        grid.cycles = grid.keyed_slot = cycle
        grid.keying(Protocol.PACTOR2, extent_n=round(.833 * onair.FS))
    grid.keying(Protocol.PACTOR1)
    verdict = grid.entry_verdict()
    assert 'PACTOR-2 WAS KEYED: 4 burst(s), first in cycle 6' in verdict
    assert 'NO ENTRY PACKET' not in verdict
    assert 'does not confirm' in verdict
