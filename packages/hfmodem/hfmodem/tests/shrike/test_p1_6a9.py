"""Alternate P1 response and entry waveform are independent selections."""
import pytest
from hfmodem.shrike import onair, pactor1
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_p3_offer import MESSAGE, cs_event
from hfmodem.tests.shrike.test_p4chirp import _P4Keyed


def connected(experimental, ladder=('p4chirp',)):
    peer = _P4Keyed()
    host = PtcHost(peer=peer, mycall='W9SSJ')
    host.p1_6a9 = experimental
    host.arq.on_host_connect('W9SSJ', 'KB5LZK')
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    host.arq.on_host_data(MESSAGE)
    host.arq.cfg.entry_ladder = ladder
    return host, peer


def response(host, word):
    host.on_rx_event(onair.rxfront.Event(
        0.2, 'unassigned', f'word {word}', protocol='PACTOR-1', spare=word, sense=0))


@pytest.mark.parametrize('experimental,word,emits', [
    (False, pactor1.CS_6A9, False), (False, pactor1.CS_59A, True),
    (True, pactor1.CS_59A, False), (True, pactor1.CS_6A9, True),
])
def test_only_selected_response_starts_chirp(experimental, word, emits):
    host, peer = connected(experimental)
    response(host, word)
    assert bool(peer.p4_entries) == emits
    assert host._grant_taken == emits
    assert not peer.entries
    if emits:
        response(host, word)
        assert len(peer.p4_entries) == 1


def test_6a9_can_start_selected_p3_entry():
    host, peer = connected(True, ('template',))
    response(host, pactor1.CS_6A9)
    assert host._grant_taken and peer.entries and not peer.p4_entries


def test_6a9_respects_pending_termination():
    host, peer = connected(True)
    host.arq.on_host_disconnect()
    response(host, pactor1.CS_6A9)
    assert not host._grant_taken and not peer.p4_entries


def test_6a9_respects_p1_only():
    host, peer = connected(True)
    host.stay_in_pactor1 = True
    response(host, pactor1.CS_6A9)
    assert not host._grant_taken and not peer.entries and not peer.p4_entries


@pytest.mark.parametrize('long_cycles', [False, True])
def test_6a9_entry_keeps_later_long_cycle_negotiation(long_cycles):
    from hfmodem.shrike import arq, spec
    host, peer = connected(True, ('template',))
    host.arq.cfg.long_cycle = long_cycles
    host.arq.on_host_data(b'A' * 1000)
    response(host, pactor1.CS_6A9)
    assert host.arq.entry_pending and not host.arq.cycle_long
    assert not host.arq._inflight.status & spec.STATUS_LONG_CYCLE
    host.arq.on_rx_cs(arq.CS_ACK)
    host.arq.on_cycle()
    assert not host.arq.entry_pending
    assert bool(host.arq._inflight.status & spec.STATUS_LONG_CYCLE) == long_cycles
    assert not host.arq.cycle_long  # A request alone never changes the cycle.
    if long_cycles:
        host.arq.on_rx_cs(arq.CS_CYCLE_TOG)
        assert host.arq.cycle_long
