# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace

import pytest

from hfmodem.shrike import compress, onair, traffic
from hfmodem.tests.shrike.test_tx_gear_hold import control, sender


def login():
    answer = bytes(range(48, 58))
    before = b'W9SSJ\r;PR: '
    after = b'\rK4MSU 2026-10-03 0533z\r' + bytes(range(256))
    return before + answer + after, len(before), len(answer)


def expected_mask(blob, start, size):
    return [start <= i < start + size for i in range(len(blob))]


@pytest.mark.parametrize('cut', range(23))
def test_framed_line_context_survives_every_callback_split(cut):
    blob, start, size = login()
    log = traffic.TrafficLog()
    log.withhold(blob[:cut])
    log.withhold(blob[cut:])
    mask = expected_mask(blob, start, size)
    expected = [hidden for byte, hidden in zip(blob, mask)
                for _ in compress.transparent(bytes((byte,)))]
    wire = compress.transparent(blob)
    assert log._hidden(wire) == expected
    assert not log._login_line
    for i, byte in enumerate(wire):
        shown = '\n'.join(log.packet('TX', 1, i % 4, bytes((byte,)), field=bytes((byte,))))
        assert ('<PR>' in shown) == expected[i]
        assert ('field ' in shown) != expected[i]
        log.accepted(1)
    assert blob == login()[0]


@pytest.mark.parametrize('cut', range(23))
def test_every_field_split_retries_and_recuts_preserve_positions(cut):
    blob, start, size = login()
    wire = compress.transparent(blob)
    log = traffic.TrafficLog()
    log.withhold(blob)
    for part in (wire[:cut], wire[cut:]):
        mask = log._hidden(part)
        for sl in (3, 2, 1, 2):
            assert log._hidden(part) == mask
            text = '\n'.join(log.packet('TX', sl, 1, part, field=part))
            assert ('<PR>' in text) == any(mask)
        log.accepted(len(part))
    ordinary = b'W9SSJ K4MSU 2026-10-03 0533z ' + bytes(range(48, 58))
    log.withhold(ordinary)
    assert not any(log._hidden(ordinary))
    output = '\n'.join(log.packet('TX', 3, 2, ordinary, field=ordinary))
    assert repr(ordinary) in output and ordinary.hex() in output


def test_real_arq_positions_follow_retries_and_lower_speed_recuts():
    host, seam = sender(False)
    a = host.arq
    a._outbuf.clear()
    a._inflight = None
    a._buffer_raw = 0
    a._sl = a._peer_sl = a._established_sl = 2
    blob, start, size = login()
    sent = []
    original_send = seam.send_packet
    log = traffic.for_host(host)

    def send(sl, payload, status, breakin=False):
        n = original_send(sl, payload, status, breakin)
        if n > 0:
            chunk = payload[:n]
            sent.append((a._buffer_raw, chunk, log._hidden(chunk)))
        return n

    seam.send_packet = send
    onair._mail_send_callback(host)(blob)
    a._start_next_packet()
    control(host, 5)
    while a._buffer_raw:
        control(host, 1 if a.tx_seq % 2 == 0 else 2)
    wire = compress.transparent(blob)
    masks = [hidden for byte, hidden in zip(blob, expected_mask(blob, start, size))
             for _ in compress.transparent(bytes((byte,)))]
    for remaining, chunk, hidden in sent:
        offset = len(wire) - remaining
        assert chunk == wire[offset:offset + len(chunk)]
        assert hidden == masks[offset:offset + len(chunk)]
    assert sum(hidden for _, _, mask in sent[1:] for hidden in mask) == size
    ordinary = b'9SSJ PACTOR test via K4MSU 2026-10-03 0533z\x02'
    onair._mail_send_callback(host)(ordinary)
    assert not any(log._hidden(ordinary))
    output = '\n'.join(log.packet('TX', 2, 1, ordinary, field=ordinary))
    assert repr(ordinary) in output and ordinary.hex() in output


def test_transparency_pair_has_the_same_sensitivity_and_terminator_forgets_it():
    blob = b';PR: ' + bytes((28, 29, 30)) + b'\r9 2026\x02'
    log = traffic.TrafficLog()
    log.withhold(blob)
    wire = compress.transparent(blob)
    assert log._hidden(wire) == [False] * 5 + [True] * 6 + [False] * 8
    assert not any(log._hidden(wire)[11:])


def test_binary_marker_outside_a_framed_login_line_is_visible():
    blob = b'\x02;PR: ' + bytes(range(48, 58)) + b'\r2026-10-03'
    log = traffic.TrafficLog()
    log.withhold(blob)
    assert not any(log._hidden(blob))


def test_multiple_callbacks_keep_queued_positions_after_partial_acceptance():
    blob, start, size = login()
    wire = compress.transparent(blob)
    a = SimpleNamespace(_buffer_raw=0)
    log = traffic.TrafficLog()
    log._arq = a
    log.withhold(blob)
    log.queued(len(wire))
    a._buffer_raw = len(wire) - start - 1
    assert log._hidden(wire[start + 1:start + size]) == [True] * (size - 1)
    more = b'K4MSU 2026-10-03\r'
    log.withhold(more)
    log.queued(len(more))
    a._buffer_raw += len(more)
    remaining = wire[start + 1:] + more
    assert log._hidden(remaining) == [True] * (size - 1) + [False] * (len(remaining) - size + 1)
    a._buffer_raw = len(more)
    assert not any(log._hidden(more))


@pytest.mark.parametrize('accepted', [0, 6, 12])
def test_direct_host_writes_after_callback_keep_exact_login_positions(accepted):
    host, seam = sender(False)
    a = host.arq
    a._outbuf.clear()
    a._inflight = None
    a._buffer_raw = 0
    answer = bytes(range(48, 56))
    blob = b';PR: ' + answer + b'\r'
    onair._mail_send_callback(host)(blob)
    if accepted:
        a._sl = a._peer_sl = a._established_sl = 1
        a._start_next_packet()
        for _ in range(3):
            if len(blob) - a._buffer_raw >= accepted:
                break
            control(host, 1 if a.tx_seq % 2 == 0 else 2)
        assert len(blob) - a._buffer_raw >= accepted
    remaining = a._buffer_raw
    ordinary = b'abcdefghij' + bytes((28, 29, 30)) + b'W9SSJ 2026-10-03\r'
    a.on_host_data(ordinary)
    wire = compress.transparent(blob + ordinary)
    offset = len(blob) - remaining
    pending = (a._inflight.payload if a._inflight else b'') + bytes(a._outbuf)
    assert pending == wire[offset:]
    log = traffic.for_host(host)
    assert log._hidden(pending) == expected_mask(wire, 5, len(answer))[offset:]
    more = b'K4MSU 0533z\r'
    onair._mail_send_callback(host)(more)
    assert log._hidden(pending + more) == expected_mask(wire + more, 5, len(answer))[offset:]


@pytest.mark.parametrize('transition', ['abort', 'disconnect', 'reconnect'])
@pytest.mark.parametrize('prefix', [b';P', b';PR: ', b';PR: ' + bytes(range(48, 52))])
def test_connection_reset_forgets_unterminated_login_context(transition, prefix):
    host, _ = sender(False)
    a = host.arq
    onair._mail_send_callback(host)(prefix)
    log = traffic.for_host(host)
    if transition == 'abort':
        a.on_host_abort()
    elif transition == 'disconnect':
        a._finish_disconnected()
    a._enter_connected()
    assert not log._line_prefix and not log._login_line and not log._mask
    ordinary = b'W9SSJ 2026-10-03\r'
    onair._mail_send_callback(host)(ordinary)
    assert not any(log._hidden(ordinary))
    assert bytes(a._outbuf).endswith(ordinary)
