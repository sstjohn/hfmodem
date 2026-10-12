# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A peer whose PACTOR-3 carrier has gone is let go within `P3_DARK_WINDOWS`.

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
    """Retain original holds; then model the approved empty receive lifetime."""
    lit = carrier_by_hold(case)
    io = Io()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    after = None
    quiet = None
    trace = []
    queued = bytes(range(8))
    for h in range(case["first"], case["last_heard"] + arq.P3_CRC_WAIT_TICKS + 1):
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
            after = len(io.keyed)
            a.on_host_data(queued)
        trace.append(dict(hold=h, keys=len(io.keyed), quiet=a.p3_receive_only,
                          ticks=a._p3_crc_wait_ticks, dark=a._dark_windows,
                          state=a.state.name, acked=a.tx_acked_bytes,
                          rx_seq=a._expected_seq, tx_seq=a._next_seq))
        if a.p3_receive_only and quiet is None:
            quiet = (h, len(io.keyed))
        if a.state == arq.State.DISCONNECTED:
            break
    assert quiet is not None
    assert a.state == arq.State.DISCONNECTED
    assert h == case["last_heard"] + arq.P3_CRC_WAIT_TICKS
    assert a._p3_crc_wait_ticks == arq.P3_CRC_WAIT_TICKS
    assert all(row["keys"] == quiet[1] for row in trace if row["hold"] >= quiet[0])
    assert not a.said_goodbye and not a._qrt_pending
    assert a.transfer_unconfirmed["suffix"] == queued
    last_crc = next(row for row in trace if row["hold"] == case["last_heard"])
    assert all((row["acked"], row["rx_seq"], row["tx_seq"]) ==
               (last_crc["acked"], last_crc["rx_seq"], last_crc["tx_seq"])
               for row in trace if row["hold"] > case["last_heard"])
    a._test_lifetime_trace = trace
    # h remains the physical terminal hold; quiet cessation is separate.
    return a, h, len(io.keyed) - after


@pytest.mark.parametrize('key_inside_window', [False, True], ids=['model', 'live'])
@pytest.mark.parametrize('case', [WS8EOC, KB5LZK], ids=['ws8eoc', 'kb5lzk'])
def test_the_link_stops_keying_within_the_dark_budget_after_the_hang_up(
        case, key_inside_window):
    a, h, keyed = run_link(case, key_inside_window=key_inside_window)
    assert a.state == arq.State.DISCONNECTED
    quiet_h = next(row["hold"] for row in a._test_lifetime_trace if row["quiet"])
    assert quiet_h <= case["last_heard"] + arq.P3_DARK_WINDOWS
    assert h == case["last_heard"] + arq.P3_CRC_WAIT_TICKS
    assert keyed < arq.P3_DARK_WINDOWS


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
