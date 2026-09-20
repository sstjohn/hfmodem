# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A shared eight-symbol shape is not positive DATA feedback."""
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from .test_data_over_gate import _connected
from .test_data_replies_0919 import native
from .test_data_naks_0919 import native as negative, ROWS as NAK_ROWS


def station(monkeypatch):
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'end')
    link = (hs.caller, hs.called, hs.bw)
    monkeypatch.delitem(VF.OVER_CONTINUE_RESPONDER_BY_LINK, link)
    return hs, io, link


def feed(hs, audio):
    for start in range(0, len(audio), 4096):
        hs.on_rx_stream(audio[start:start + 4096])


def test_unknown_reply_waits_for_query_then_learns_only_its_confirmed_tail(monkeypatch):
    hs, io, link = station(monkeypatch)
    pending = hs._tx_pending
    reply = native('stock-2300-continue-1')
    feed(hs, reply)
    assert hs._tx_pending == pending and len(io.sent) == 1
    assert hs._unclassified_continue is not None
    assert not hs._confirmed_continue_pairs
    assert hs._probe_intermediate_answer()
    feed(hs, native('stock-2300-query-answer'))
    assert hs._tx_pending[1] == pending[1] + 1
    assert len(hs._confirmed_continue_pairs) == 1
    assert next(iter(hs._confirmed_continue_pairs))[:3] == link
    sent = len(io.sent)
    feed(hs, reply)
    assert hs._tx_pending[1] == pending[1] + 2
    assert len(io.sent) == sent + 1


def test_unknown_reply_is_not_learned_from_negative_feedback(monkeypatch):
    hs, io, _ = station(monkeypatch)
    reply = native('stock-2300-continue-1')
    feed(hs, reply)
    assert hs._probe_intermediate_answer()
    row = next(r for r in NAK_ROWS if r['call'] == 'KC9GHZ' and r['seed'] == 288)
    feed(hs, negative(row))
    assert hs._tx_retries == 1
    assert not hs._confirmed_continue_pairs


def test_unknown_reply_cache_is_cleared_at_connection_boundary(monkeypatch):
    hs, io, link = station(monkeypatch)
    hs._confirmed_continue_pairs.add((*link, ((1, 2),) * 8))
    hs._unclassified_continue = (hs._tx_pending, hs._pending_answer_at, ((1, 2),) * 8)
    hs._connected(confirm=False)
    assert not hs._confirmed_continue_pairs and hs._unclassified_continue is None


def test_ack_of_another_link_is_not_authority_to_retire_this_pending_frame():
    hs, io = _connected()
    hs.called = 'N0XYZ'
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'last')
    old = hs._tx_pending
    feed(hs, native('stock-2300-continue-1'))
    assert hs._tx_pending == old and len(io.sent) == 1
    assert not hs._confirmed_continue_pairs


def unmapped_native(name):
    from pathlib import Path
    from scipy.io import wavfile
    path = (Path(__file__).with_name('fixtures') /
            'unmapped-data-replies-0919' / (name + '.wav'))
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == VA.MK.FS and x.dtype.kind == 'f'
    return x.astype(float)


def unmapped_station():
    hs, io = _connected()
    hs.called = 'N0XYZ'
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'end')
    assert (hs.caller, hs.called, hs.bw) not in VF.OVER_CONTINUE_RESPONDER_BY_LINK
    return hs, io


def test_native_unmapped_link_learns_only_after_its_positive_query_reply():
    hs, io = unmapped_station()
    old = hs._tx_pending
    reply = unmapped_native('unmapped-ack')
    assert not hs._peer_over_continue(reply)
    assert hs._tx_pending == old and not hs._confirmed_continue_pairs
    assert hs._probe_intermediate_answer()
    assert hs._peer_intermediate_query_answer(unmapped_native('positive-query'))
    hs._took_intermediate_query_answer()
    assert hs._tx_pending[1] == old[1] + 1
    assert hs._peer_over_continue(reply)
    assert len(hs._confirmed_continue_pairs) == 1


def test_native_long_nak_can_arrive_directly_without_a_query():
    from hfmodem.kestrel.arq import phy
    audio = unmapped_native('direct-long-nak')
    for chunk in (512, 4096, 4800):
        hs, io = unmapped_station()
        old = hs._tx_pending
        assert not hs._intermediate_query_attempted
        assert not hs._peer_data_nak(audio[:int(.65 * VA.MK.FS)])
        for at in range(0, len(audio), chunk):
            hs.on_rx_stream(audio[at:at + chunk])
        assert hs._tx_retries == 1 and hs._intermediate_query_attempts == 0
        assert hs._tx_pending[1:] == (old[1], 2)
        body = hs._tx_pending[0]
        assert (phy.vara_payload(body, caller=hs.caller, body_len=len(body))
                + b''.join(hs._txq)) == b'A' * 89 + b'B' * 89 + b'end'
        assert not hs._confirmed_continue_pairs
