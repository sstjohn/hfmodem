# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A peer whose PACTOR-3 carrier has gone is answered in every slot until the
CRC-free lifetime ends the link on a goodbye.

WS8EOC-80 0924-0249 (`captures/onair-0923-2150`): the gateway's last packet is
in hold 56 and it hung up; holds 57-66 carry no PACTOR-3, and the silence budget
keyed nine CS2s at the empty channel before it ended the link.

KB5LZK-30 0924-1344 (`captures/onair-0924-0844`): the gateway's last packet is
in hold 40; holds 41-46 are dark and each drew a CS1, because the live driver
keys a window's answer before that window's verdict was taken.
"""
import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, onair, rxfront, spec

CAPTURES = Path(__file__).resolve().parents[5] / 'captures'
P3 = spec.Protocol.PACTOR3
WS8EOC = dict(session='onair-0923-2150', first=3, last_heard=56, last=66,
              decoded=set(range(3, 57)) - {5, 23, 28, 29, 30, 50})
KB5LZK = dict(session='onair-0924-0844', first=10, last_heard=40, last=46,
              decoded=set(range(10, 41)))


def windows(session, last):
    """Each hold's receive window, cut from the unnormalised session stream."""
    where = CAPTURES / session
    if not (where / 'stream.flac').is_file():
        pytest.skip(f'session capture not on this machine: {where}')
    stream = rxfront.load_wav(str(where / 'stream.flac'))
    for h in range(1, last + 1):
        meta = json.loads((where / f'hold_{h:02d}.json').read_text())
        end = meta['end_stream_sample']
        yield h, stream[end - meta['samples']:end]


def carrier_by_hold(case):
    """The live order: a decoded window teaches the template, an unread one asks it."""
    peer = onair._PeerSpectrum()
    lit = {}
    for h, seg in windows(case['session'], case['last']):
        peer.hear(seg)
        if h in case['decoded']:
            peer.learn(seg)
        lit[h] = h in case['decoded'] or (h >= case['first'] and peer.carries_peer(seg))
    return lit


@pytest.mark.parametrize('case', [WS8EOC, KB5LZK], ids=['ws8eoc', 'kb5lzk'])
def test_the_carrier_is_there_until_the_gateway_hangs_up_and_gone_after(case):
    lit = carrier_by_hold(case)
    assert all(lit[h] for h in range(case['first'], case['last_heard'] + 1))
    assert not any(lit[h] for h in range(case['last_heard'] + 1, case['last'] + 1))


class Io(arq.ArqIO):
    protocol = P3

    def __init__(self):
        self.keyed = []

    def send_cs(self, index):
        self.keyed.append(index)

    def send_packet(self, sl, payload, status, breakin=False):
        self.keyed.append(('packet', status))
        return arq.REFUSED


def run_link(case, *, key_inside_window):
    """Retain original holds; then model the empty receive lifetime."""
    lit = carrier_by_hold(case)
    io = Io()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    trace = []
    queued = bytes(range(8))
    for h in range(case["first"], case["last_heard"] + arq.P3_CRC_WAIT_TICKS + 2):
        before = len(io.keyed)
        if not key_inside_window:
            a.on_cycle()
        token = a.begin_receive_opportunity()
        if h in case["decoded"]:
            a.on_rx_packet(1, b"", spec.status_byte(2), True, protocol=P3)
            # The preexisting decoded-hold set supplies unit CRC evidence;
            # carrier alone does not qualify the current opportunity.
            assert a.qualify_peer(token, a._link_generation, "current CRC")
        if lit.get(h, False):
            a.note_peer_carrier()
        if key_inside_window:
            a.on_cycle()
        a.finish_receive_opportunity(token)
        if h == case["last_heard"]:
            a.on_host_data(queued)
        trace.append(dict(hold=h, keyed=len(io.keyed) - before, qrt=a._qrt_pending,
                          state=a.state.name, acked=a.tx_acked_bytes,
                          rx_seq=a._expected_seq, tx_seq=a._next_seq))
        if a.state == arq.State.DISCONNECTED:
            break
    assert a.state == arq.State.DISCONNECTED
    assert a._p3_crc_wait_ticks >= arq.P3_CRC_WAIT_TICKS
    # The goodbye was offered; this seam refuses packets, which ends the link.
    assert any(k[0] == 'packet' and k[1] & spec.STATUS_QRT for k in io.keyed
               if isinstance(k, tuple))
    assert a.transfer_unconfirmed["suffix"] == queued
    last_crc = next(row for row in trace if row["hold"] == case["last_heard"])
    assert all((row["acked"], row["rx_seq"], row["tx_seq"]) ==
               (last_crc["acked"], last_crc["rx_seq"], last_crc["tx_seq"])
               for row in trace if row["hold"] > case["last_heard"])
    a._test_lifetime_trace = trace
    return a, h


@pytest.mark.parametrize('key_inside_window', [False, True], ids=['model', 'live'])
@pytest.mark.parametrize('case', [WS8EOC, KB5LZK], ids=['ws8eoc', 'kb5lzk'])
def test_the_link_keys_every_slot_after_the_hang_up_until_its_lifetime_ends(
        case, key_inside_window):
    a, h = run_link(case, key_inside_window=key_inside_window)
    assert h <= case["last_heard"] + arq.P3_CRC_WAIT_TICKS + 1
    # From the first slot the last CRC's answer does not already cover.
    quiet = [row for row in a._test_lifetime_trace
             if row["hold"] >= case["last_heard"] + 2 and not row["qrt"]]
    assert len(quiet) >= arq.P3_CRC_WAIT_TICKS - 2
    assert all(row["keyed"] == 1 for row in quiet), quiet


@pytest.mark.parametrize("source, crc, expected", [
    ("current CRC", True, 0),
    ("current CRC", False, 2),
    ("crc", True, 2),
    ("unit-short-header", True, 2),
    (None, False, 2),
])
def test_current_crc_requires_accepted_packet_and_current_evidence(source, crc, expected):
    io = Io()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    token = a.begin_receive_opportunity()
    a.finish_receive_opportunity(token)
    assert a._p3_crc_wait_ticks == 1
    token = a.begin_receive_opportunity()
    if crc:
        a.on_rx_packet(1, b"", spec.status_byte(2), True, protocol=P3)
    if source is not None:
        assert a.qualify_peer(token, a._link_generation, source)
    else:
        a.note_peer_carrier()
        assert not a.peer_qualified
    snapshot = (a.tx_acked_bytes, a._expected_seq, a._next_seq, bytes(a._outbuf), len(io.keyed))
    a.finish_receive_opportunity(token)
    assert a._p3_crc_wait_ticks == expected
    assert (a.tx_acked_bytes, a._expected_seq, a._next_seq, bytes(a._outbuf), len(io.keyed)) == snapshot
