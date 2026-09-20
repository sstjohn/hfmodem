# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Ignored P1 words cannot change a confirmed P3 peer's transmit geometry."""
import pytest

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_breakin_timing import transitioned, observed_reply, tx_at

FS = onair.FS


def confirmed(tmp_path):
    g = transitioned()
    observed_reply(g)
    s = _Session(role=arq.IRS)
    tx = tx_at(g, 44, tmp_path)
    tx.host = s.host
    s.host.peer = tx
    return g, s, tx


@pytest.mark.parametrize('check', ['parity', 'geometry'])
def test_ignored_p1_callback_cannot_change_fresh_p3_placement(check, tmp_path):
    g, s, tx = confirmed(tmp_path)
    # Deliberately put the foreign P1 word where borrowing its120ms extent
    # would move an otherwise accepted P3 break-in120samples early. This is
    # an adversarial placement, not a claim about the peer's actual reply.
    at = g.boundary(44) - 5760 - g.peer_read_gap - 120
    sense = int(not g.shift(g.rx_slot(at)))
    ev = rxfront.Event(at / FS, 'cs', 'P1 ignored by host',
                       protocol=spec.Protocol.PACTOR1, cs=0, sense=sense)
    s.rx.cs_log.append(ev)
    before_shift = g.shift_slot
    before_end = g.peer_packet_end(44)
    if check == 'parity':
        onair._align_shift(g, s.rx, tx, 44, at)
        assert g.shift_slot == before_shift
    else:
        onair._forecast_next_key(s.rx, tx, g, at-10)
        assert g.peer_packet_end(44) == before_end
        assert tx._breakin_key(g, 44) == g.boundary(44)
        assert g.peer_cs is None


def test_cached_lower_protocol_alias_cannot_supply_p3_packet_extent(tmp_path):
    g, _, tx = confirmed(tmp_path)
    before_end = g.peer_packet_end(44)
    at = g.boundary(44) - 5760 - g.peer_read_gap - 120
    g.note_peer_codeword(at, 5760, 'CS1/ack', 'WS8EOC', protocol=spec.Protocol.PACTOR1)
    assert g.peer_at == g._p3_peer[0]
    assert g._bare_peer_width(at) is None
    # Even an onset on the foreign word's phase cannot turn that alias into
    # the120ms duration of a P3 peer transmission.
    g.peer_onset = at
    assert g.peer_packet_end(44) == before_end
    assert tx._breakin_key(g, 44) == g.boundary(44)


def test_unconfirmed_entry_retains_p1_grant_shift_and_timing_evidence(tmp_path):
    g = transitioned()
    g._p3_peer = None  # Entry emitted; no P3 peer frame has been accepted.
    g._p3_peer_confirmed = False
    s = _Session(role=arq.ISS, entry_pending=True)
    tx = tx_at(g, 44, tmp_path)
    tx.host = s.host
    s.host.peer = tx
    at = g.rx_due(43)
    sense = int(not g.shift(g.rx_slot(at)))
    ev = rxfront.Event(at / FS, 'unassigned', 'repeated P1 grant',
                       protocol=spec.Protocol.PACTOR1,
                       spare=onair.pactor1.CS_59A, sense=sense)
    s.rx.cs_log.append(ev)
    onair._align_shift(g, s.rx, tx, 44, at)
    onair._forecast_next_key(s.rx, tx, g, at-10)
    assert g.shift(g.rx_slot(at)) == bool(sense)
    assert g.peer_cs.at == at and g.peer_cs.width == g.p1_cs_n
    assert g.peer_cs.protocol == spec.Protocol.PACTOR1
    assert g.d == 4413 and g.rx_ref_n == 17280


def test_bare_control_confirms_p3_without_reusing_cached_p1_geometry(tmp_path):
    g = transitioned()
    g._p3_peer, g._p3_peer_confirmed = None, False
    g.sending = True
    g.note_peer_codeword(2600000, 5760, 'CS1/ack', 'WS8EOC',
                         protocol=spec.Protocol.PACTOR1)
    g.peer_onset = 2600000
    s = _Session(role=arq.ISS, entry_pending=True)
    s.host.peer.raster = g
    s.host.arq.on_host_data(b'pending entry payload')
    s.host.arq._next_seq = 2
    s.host.arq._start_next_packet()
    assert s.host.arq.tx_seq == 2
    # Accepted/corroborated bare P3 CS1, through the real host callback.
    ev = rxfront.Event(2610000 / FS, 'cs', 'corroborated ACK',
                       protocol=spec.Protocol.PACTOR3, cs=arq.CS_ACK)
    s.rx._on(ev)
    assert not s.host.arq.entry_pending
    onair._grid_reversal(g, s.host)
    assert g._p3_peer_confirmed and g._p3_peer is None
    assert g.peer_at is None and g._peer_air() is None
    assert g._bare_peer_width(2600000) is None
    # A real P3 bare control has geometry, but no CRC/control timing pair
    # from which to authorize our own changeover using the inherited P1 d.
    g.note_peer_codeword(2610000, 10080, 'ACK', 'WS8EOC',
                         protocol=spec.Protocol.PACTOR3)
    tx = tx_at(g, 44, tmp_path)
    assert tx._place_breakin() == ''
    assert 'no fresh corroborated' in tx.unplaceable


def test_iss_retry_ignores_fsk_onset_while_crc_timing_is_fresh(tmp_path):
    g, s, tx = confirmed(tmp_path)
    g.reverse(to_iss=True)
    before_end = g.peer_packet_end(44)
    g.peer_onset = g._peer_raster_at = 2661704
    g.note_peer_codeword(2661704, 5760, 'CS1/ack', 'WS8EOC',
                         protocol=spec.Protocol.PACTOR1)
    assert g.peer_at == g._p3_peer[0]
    assert g.peer_packet_end(44) == before_end
    assert g._peer_air()[:2] == g._p3_peer[:2]
    assert tx._breakin_key(g, 44) == g.boundary(44)


def test_new_p1_contact_releases_confirmed_p3_consumer_filter(tmp_path):
    g, s, tx = confirmed(tmp_path)
    s.host.protocol = spec.Protocol.PACTOR1
    s.host.arq.state = arq.State.CONNECTING
    onair._grid_reversal(g, s.host)
    assert g._p3_timing is None and not g._p3_controls
    g.keying(spec.Protocol.PACTOR1)
    g.note_peer_codeword(2700000, 5760, 'CS1/ack', 'WS8EOC',
                         protocol=spec.Protocol.PACTOR1)
    g.sending = True
    assert not g._p3_peer_confirmed
    assert g._bare_peer_width(2700000) == 5760
    assert g._peer_air()[:2] == (2700000, 5760)
