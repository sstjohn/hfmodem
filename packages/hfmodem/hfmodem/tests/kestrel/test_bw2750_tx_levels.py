# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Announced DATA descent conserves bytes and pending-frame ownership.

Stub-rendered remainder checks establish local framing, not stock acceptance
of near-capacity short trailers. Native transport qualification is separate.
"""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as va
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.tests.kestrel.test_vara_giveup import _IO

CAPACITY = {103: 89, 102: 47, 101: 22, 100: 9}


def prefix_records(level):
    return list(range(103, 99 + level, -1))


@pytest.mark.parametrize('level,cap', [(1, 9), (2, 22), (3, 47), (4, 89)])
def test_every_remainder_and_exact_multiple_preserves_bytes(level, cap, monkeypatch):
    rendered = []
    def render(body, *, over, bw, level):
        rendered.append((body, over, level))
        return np.zeros(1)
    monkeypatch.setattr(va.OF, 'data_over_tx', render)
    prefix_bytes = sum(CAPACITY[r] for r in prefix_records(level))
    for size in range(1, prefix_bytes + 3 * cap + 1):
        io = _IO()
        hs = va.VaraStationHandshake(['W9SSJ'], io, bw='2750', tx_level=level)
        hs.caller, hs.called = 'W9SSJ', 'KC9GHZ'
        payload = bytes((i % 113) + 32 for i in range(size))
        hs.send(payload)
        blocks = list(hs._txq)
        assert b''.join(blocks) == payload
        decoded = bytearray()
        rendered.clear()
        for index, block in enumerate(blocks):
            assert hs._tx_data_over()
            body, over, record = hs._tx_pending
            expected_record = max(99 + level, 103 - index)
            actual_cap = CAPACITY[record]
            assert record == expected_record and len(body) == actual_cap + 1
            assert over == index + 1
            part = phy.vara_payload(body, caller=hs.caller, body_len=len(body))
            assert part == block
            decoded.extend(part)
            assert phy.over_is_last(body, hs.caller) == (index == len(blocks) - 1)
            if index < len(blocks) - 1:
                assert len(block) == actual_cap
                if level == 4:
                    assert body[-1] == (0x89, 0x8d, 0x91)[len(blocks) - index - 2]
                else:
                    assert body[-1] == (0x81 if record > 99 + level else 0x89)
                assert hs._full_keyed == index + 1
            else:
                assert len(block) < actual_cap
                assert hs._full_keyed == 0
                # The two near-cap short forms currently overwrite part of
                # the trailer. Do not misrepresent round-trip bytes as native
                # qualification of those forms.
                if len(block) <= actual_cap - 3:
                    assert body[-2:] == b'\x04\x82'
            hs._tx_pending = None  # ACK boundary; no modem/hardware in this unit test.
        assert bytes(decoded) == payload
        assert len(rendered) == len(blocks)
        assert not hs._txq


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_refused_tx_and_retry_retain_exact_record_and_bytes(level, monkeypatch):
    io = _IO(False)
    hs = va.VaraStationHandshake(['W9SSJ'], io, bw='2750', tx_level=level)
    hs.caller, hs.called = 'W9SSJ', 'KC9GHZ'
    frames = []
    def render(body, *, over, bw, level):
        frames.append((body, over, level))
        return np.zeros(1)
    monkeypatch.setattr(va.OF, 'data_over_tx', render)
    hs.send(b'a' * 100)
    original = list(hs._txq)
    hs._tx_data_over()
    assert hs._txq == original and hs._tx_pending is None and hs._over == 0
    io.transmits = True
    hs._tx_data_over()
    pending, remaining = hs._tx_pending, list(hs._txq)
    hs.state = va.VaraState.CONNECTED
    hs._retry_data_over(peer_asked=True)
    assert hs._tx_pending == pending and hs._txq == remaining
    assert frames[0] == frames[1] == frames[2]


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_79_byte_fc_starts_and_closes_at_base(level, monkeypatch):
    monkeypatch.setattr(va.OF, 'data_over_tx', lambda *a, **kw: np.zeros(1))
    hs = va.VaraStationHandshake(['W9SSJ'], _IO(), bw='2750', tx_level=level)
    hs.caller = 'W9SSJ'
    payload = b'f' * 79
    hs.send(payload)
    assert hs._txq == [payload]
    hs._tx_data_over()
    body, over, record = hs._tx_pending
    assert (over, record, len(body)) == (1, 103, 90)
    assert phy.vara_payload(body, caller=hs.caller) == payload
    assert body[-2:] == b'\x04\x82'
    assert hs._full_keyed == 0 and not hs._txq


@pytest.mark.parametrize('level', [1, 2, 3])
def test_refused_lower_transition_does_not_advance_ladder(level, monkeypatch):
    frames = []
    def render(body, *, over, bw, level):
        frames.append((body, over, level))
        return np.zeros(1)
    monkeypatch.setattr(va.OF, 'data_over_tx', render)
    io = _IO()
    hs = va.VaraStationHandshake(['W9SSJ'], io, bw='2750', tx_level=level)
    hs.caller = 'W9SSJ'
    prefix = prefix_records(level)
    target = 99 + level
    hs.send(b'a' * (sum(CAPACITY[r] for r in prefix) + 2 * CAPACITY[target] + 3))
    for record in prefix:
        hs._tx_data_over()
        assert hs._tx_pending[2] == record
        hs._tx_pending = None
    remaining = list(hs._txq)
    io.transmits = False
    hs._tx_data_over()
    assert hs._txq == remaining and hs._tx_pending is None
    assert hs._over == hs._full_keyed == len(prefix)
    assert frames[-1][2] == target
    io.transmits = True
    hs._tx_data_over()
    pending = hs._tx_pending
    rest = list(hs._txq)
    hs.state = va.VaraState.CONNECTED
    hs._retry_data_over(peer_asked=True)  # A decoded NAK keeps the committed record.
    assert hs._tx_pending == pending and hs._txq == rest
    assert hs._over == hs._full_keyed == len(prefix) + 1
    assert frames[-3] == frames[-2] == frames[-1]


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_independent_writes_restart_ladder_without_changing_pending(level, monkeypatch):
    monkeypatch.setattr(va.OF, 'data_over_tx', lambda *a, **kw: np.zeros(1))
    hs = va.VaraStationHandshake(['W9SSJ'], _IO(), bw='2750', tx_level=level)
    hs.caller = 'W9SSJ'
    prefix = prefix_records(level)
    first = b'a' * (sum(CAPACITY[r] for r in prefix) + 2 * CAPACITY[99 + level] + 3)
    second, third = b'b' * 79, b'c' * 126
    hs.send(first)
    first_count = len(hs._txq)
    hs._tx_data_over()
    pending = hs._tx_pending
    hs.send(second)
    hs.send(third)
    assert hs._tx_pending == pending
    assert hs._over == hs._full_keyed == 1
    seen, received = [], bytearray()
    while hs._tx_pending is not None or hs._txq:
        if hs._tx_pending is None:
            hs._tx_data_over()
        body, over, record = hs._tx_pending
        seen.append(record)
        received.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
        hs._tx_pending = None
    assert bytes(received) == first + second + third
    assert seen[:first_count] == prefix + [99 + level] * 3
    assert seen[first_count:] == [103, 103, max(102, 99 + level)]
    assert hs._full_keyed == 0


@pytest.mark.parametrize('bw,cap', [('500', 43), ('2300', 89), ('2750', 89)])
def test_default_geometry_unchanged(bw, cap):
    hs = va.VaraStationHandshake(['W9SSJ'], _IO(), bw=bw)
    assert hs._tx_payload_size() == cap
    hs.send(b'x' * cap)
    assert hs._txq == [b'x' * cap, b'']


@pytest.mark.parametrize('bw,level', [('2300', 2), ('500', 4), ('2750', 0), ('2750', 5), ('2750', True)])
def test_invalid_selector_rejected(bw, level):
    with pytest.raises(ValueError, match='tx_level'):
        va.VaraStationHandshake(['W9SSJ'], _IO(), bw=bw, tx_level=level)


def test_native_mixed_low_level_training_rng():
    # Unique LCG state solved from native consecutive L2 then L3 training:
    # working/vara-bw2750-tx-0916/preamble/mixed-bpc-rng.json.
    state_after_first = 16570462
    inverse = pow(0x43FD43FD, -1, 1 << 24)
    before = ((state_after_first - 0xC39EC3) * inverse) % (1 << 24)
    rnd = tx.VB6Rnd(before)
    for record, draws in [(101, [7, 1, 7, 4]), (102, [14, 0, 12, 9, 11, 13])]:
        r = tx.RECORDS[record]
        expected = [((int(a) + r.stride * d - r.first_bin) % r.span) + r.first_bin
                    for a, d in zip(tx.preamble_alloc(record), draws)]
        assert tx.preamble_bins(rnd, record) == expected
    assert rnd.state == 13727505


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_cli_selector_reaches_station_without_devices(level, monkeypatch):
    import sys
    from hfmodem.tests.kestrel import corpora
    cli = corpora.harness('kestrel_connect')
    seen = []
    def connect_without_devices(*args, **kwargs):
        seen.append(kwargs['hs'].tx_level)
        return False
    def forbidden(*args, **kwargs):
        pytest.fail('CLI preflight opened hardware')
    monkeypatch.setattr(cli, 'connect', connect_without_devices)
    monkeypatch.setattr(cli, 'AudioVaraIO', forbidden)
    monkeypatch.setattr(cli, 'Rig', forbidden)
    monkeypatch.setattr(sys, 'argv', ['kestrel_connect', '--gateway', 'KC9GHZ',
        '--mycall', 'W9SSJ', '--bw', '2750', '--tx-level', str(level), '--dry-run'])
    assert cli.main() == 1
    assert seen == [level]


@pytest.mark.parametrize('bw,level', [('500', 2), ('2300', 3), ('2750', 5)])
def test_cli_bad_selector_fails_before_device_or_credentials(bw, level, monkeypatch):
    import sys
    from hfmodem.tests.kestrel import corpora
    cli = corpora.harness('kestrel_connect')
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid selector reached device or credentials')
    monkeypatch.setattr(cli, 'AudioVaraIO', forbidden)
    monkeypatch.setattr(cli, 'Rig', forbidden)
    monkeypatch.setattr(cli.config, 'mail_password', forbidden)
    monkeypatch.setattr(sys, 'argv', ['kestrel_connect', '--gateway', 'KC9GHZ',
        '--mycall', 'W9SSJ', '--bw', bw, '--tx-level', str(level), '--dry-run'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize('remaining', [4, 5, 29, 30, 31, 62, 1000])
def test_long_delivery_field_saturates_without_wrapping_to_final(remaining):
    assert va._frame_field(remaining, True) == 0x99
    assert va._frame_field(remaining, False) == 0x99


@pytest.mark.parametrize('level,cap', [(1, 9), (2, 22), (3, 47), (4, 89)])
def test_full_binary_suffix_is_not_a_short_trailer(level, cap):
    payload = b'x' * (cap - 2) + b'\x14\xf6'
    for field in (0x81, 0x89, 0x8d, 0x91, 0x95, 0x99):
        body = phy.vara_body(payload, 'W9SSJ', tail=field, body_len=cap + 1)
        assert phy.vara_payload(body, caller='W9SSJ', body_len=len(body)) == payload
        assert not phy.over_is_last(body, 'W9SSJ')


@pytest.mark.parametrize('level', [1, 2, 3])
def test_lower_records_query_without_inheriting_clipped_continue_heuristic(level):
    hs = va.VaraStationHandshake(['W9SSJ'], _IO(), bw='2750', tx_level=level,
                                allow_data_retries=False, probe_intermediate_query=True)
    hs.state, hs.role, hs.turn = va.VaraState.CONNECTED, 'initiator', va._TURN_OURS
    hs.caller, hs.called = 'W9SSJ', 'KC9GHZ'
    cap = CAPACITY[99 + level]
    hs._tx_pending = (phy.vara_body(b'x' * cap, hs.caller, tail=0x89,
                                  body_len=cap + 1), 1, 99 + level)
    hs._txq = [b'last']
    assert hs._intermediate_answer_candidate()
    assert not hs._peer_head_cut_continue([], ([], []))
    hs._txq = []
    hs._tx_pending = (phy.vara_body(b'last', hs.caller, body_len=cap + 1), 2, 99 + level)
    assert hs._final_ack_candidate()


@pytest.mark.parametrize('level,cap', [(1, 9), (2, 22), (3, 47), (4, 89)])
def test_long_binary_delivery_preserves_full_suffix_and_field_count(level, cap, monkeypatch):
    monkeypatch.setattr(va.OF, 'data_over_tx', lambda *a, **kw: np.zeros(1))
    hs = va.VaraStationHandshake(['W9SSJ'], _IO(), bw='2750', tx_level=level)
    hs.caller = 'W9SSJ'
    block = bytes(range(cap - 2)) + b'\x14\xf6'
    prefix = prefix_records(level)
    payload = b''.join(bytes(range(CAPACITY[r] - 2)) + b'\x14\xf6'
                       for r in prefix) + block * 64
    hs.send(payload)
    received = bytearray()
    fields = []
    while hs._txq:
        hs._tx_data_over()
        body = hs._tx_pending[0]
        part = phy.vara_payload(body, caller=hs.caller, body_len=len(body))
        received.extend(part)
        if part:
            assert not phy.over_is_last(body, hs.caller)
            fields.append(body[-1])
        else:
            assert phy.over_is_last(body, hs.caller)
        hs._tx_pending = None
    assert bytes(received) == payload
    # Lower targets use descent/stay announcements; explicitL4 preserves the
    # established countdown policy instead of changing the base-level path.
    expected = ([0x99] * 60 + [0x95, 0x91, 0x8d, 0x89] if level == 4
                else [0x81] * len(prefix) + [0x89] * 64)
    assert fields == expected
