# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import struct

import pytest

from hfmodem.station.pskreporter import Config, Encoder, Reporter, Spot


def test_official_complete_packet_fixture():
    # pskdev's public Homebrew v5.6 packet, with sequence explicitly set to one.
    encoder = Encoder('N1DQ', 'FN42hn', software='Homebrew v5.6', domain=0)
    encoder.sequence = 1
    packet = encoder.packet([
        Spot('N1DQ', 14070567, 0x47953254, 'cq'),
        Spot('KB1MBX', 14070987, 0x47953268, 'beacon'),
    ], now=0x47953272, mode='PSK')
    expected = bytes.fromhex('''
        000a00ac479532720000000100000000
        000300249992000300018002ffff0000768f8004ffff0000768f8008ffff0000768f0000
        0002002c999300058001ffff0000768f800500040000768f800affff0000768f800b00010000768f00960004
        99920020044e31445106464e3432686e0d486f6d65627265772076352e360000
        9993002c044e31445100d6b3270350534b0147953254064b42314d425800d6b4cb0350534b01479532680000
    ''')
    assert packet == expected


class UDP:
    def __init__(self):
        self.packets = []
        self.fail = False
        self.closed = False

    def settimeout(self, value):
        assert value == 2

    def sendto(self, packet, address):
        assert address == ('127.0.0.1', 4739)
        if self.fail:
            raise OSError('fake network down')
        self.packets.append(packet)
        return len(packet)

    def close(self):
        self.closed = True


def fixture(enabled=True, capacity=256):
    now = [0.0]
    udp = UDP()
    calls = []

    def resolve(host, port, **kwargs):
        calls.append((host, port))
        return [(2, 2, 17, '', ('127.0.0.1', 4739))]

    reporter = Reporter(Config('W9SSJ', 'EN61', enabled, capacity),
                        clock=lambda: now[0], wall=lambda: 1700000000 + now[0],
                        resolver=resolve, socket_factory=lambda *args: udp,
                        start_worker=False)
    return reporter, now, udp, calls


def spot(call='K1ABC'):
    return Spot(call, 14106500, 1700000000, 'beacon')


def test_disabled_never_resolves_or_creates_socket():
    r, now, udp, calls = fixture(False)
    assert not r.offer(spot())
    now[0] = 10000
    r._batch()
    r.stop()
    assert calls == [] and udp.packets == []


def test_batch_dedupe_own_call_and_stable_socket():
    r, now, udp, calls = fixture()
    assert not r.offer(spot('W9SSJ'))
    assert r.offer(spot())
    assert not r.offer(spot())
    r._batch()
    assert udp.packets == []
    for i in range(4):
        now[0] = 300 * (i + 1)
        if i:
            assert r.offer(spot())
        r._batch()
    assert len(calls) == 1 and len(udp.packets) == 4
    assert [struct.unpack('!I', p[8:12])[0] for p in udp.packets] == [0, 1, 2, 3]
    assert all(p[16:18] == b'\x00\x03' for p in udp.packets[:3])
    assert udp.packets[3][16:18] == b'\x99\x92'
    assert b'VARAC' in udp.packets[0] and b'hfmodem 0.4.0' in udp.packets[0]
    now[0] = 4500
    r.offer(spot())
    r._batch()
    assert udp.packets[-1][16:18] == b'\x00\x03'


def test_queue_cap_failure_retry_and_stop_no_flush():
    r, now, udp, calls = fixture(capacity=1)
    assert r.offer(spot())
    assert not r.offer(spot('K2ABC'))
    udp.fail = True
    now[0] = 300
    r._batch()
    assert r.status()['queued'] == 1 and r.status()['error']
    udp.fail = False
    now[0] = 600
    r._batch()
    assert r.status()['submitted'] == 1 and r.status()['error'] is None
    assert r.offer(spot('K2ABC'))
    r.stop()
    now[0] = 900
    r._batch()
    assert len(udp.packets) == 1 and udp.closed


def test_live_provenance_and_frequency_required():
    r, _, _, _ = fixture()
    record = dict(call='K1ABC', rf_source='cq', rx_frequency_hz=14106500,
                  received_unix=1700000000, dial_hz=14105750)
    assert not r.accept_heard(record)
    assert not r.accept_heard({**record, 'rf_source': 'text'}, live=True)
    assert not r.accept_heard({k: v for k, v in record.items() if k != 'rx_frequency_hz'}, live=True)
    assert r.accept_heard(record, live=True)
    assert r._queue[0].frequency_hz == 14106500


@pytest.mark.parametrize('call', ['UNKNOWN', 'K1ABC\n', '<K1ABC>', 'X-W9SSJ'])
def test_invalid_calls_rejected(call):
    with pytest.raises(ValueError):
        spot(call)


def test_mtu_bounded_and_sequence_counts_reports():
    r, now, udp, _ = fixture()
    for i in range(100):
        r.offer(spot(f'K{i}ABC'))
    now[0] = 300
    r._batch()
    submitted = r.status()['submitted']
    assert 1 < submitted < 100 and len(udp.packets[0]) <= 1200
    now[0] = 600
    r._batch()
    assert struct.unpack('!I', udp.packets[1][8:12])[0] == submitted


def test_stop_during_dns_prevents_later_send():
    import threading
    r, now, udp, _ = fixture()
    entered, release = threading.Event(), threading.Event()
    original = r._resolver
    def resolve(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    r._resolver = resolve
    r.offer(spot())
    now[0] = 300
    worker = threading.Thread(target=r._batch)
    worker.start()
    assert entered.wait(2)
    r.stop()
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert udp.packets == [] and r.status()['queued'] == 0
    assert not r.status()['stopping']
    r.stop()  # Close the fake-only socket; production worker finally owns close.


def test_stop_reports_already_admitted_send_and_discards_unsent_queue():
    import threading
    r, now, udp, _ = fixture()
    entered, release = threading.Event(), threading.Event()
    original = udp.sendto
    def send(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    udp.sendto = send
    r.offer(spot())
    now[0] = 300
    worker = threading.Thread(target=r._batch)
    worker.start()
    assert entered.wait(2)
    r.stop()
    assert r.status()['stopping'] and not r.status()['enabled']
    assert r.status()['queued'] == 0
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert not r.status()['stopping'] and r.status()['submitted'] == 1
    assert not r.offer(spot('K2ABC'))
    r.stop()


def test_failed_queued_call_is_not_duplicated_after_dedupe_window():
    r, now, udp, _ = fixture()
    r.offer(spot())
    udp.fail = True
    now[0] = 300
    r._batch()
    now[0] = 600
    assert not r.offer(spot())
    assert r.status()['queued'] == 1


def test_status_countdown_explains_wait_and_retry_without_changing_schedule():
    r, now, udp, calls = fixture()
    assert r.status()['next_upload_seconds'] is None
    r.offer(spot())
    assert r.status()['next_upload_seconds'] == 300
    now[0] = 74.2
    assert r.status()['next_upload_seconds'] == 226
    assert not calls
    now[0] = 300
    udp.fail = True
    r._batch()
    assert r.status()['next_upload_seconds'] == 300
    assert r.status()['queued'] == 1 and r.status()['error']
    now[0] = 600
    udp.fail = False
    r._batch()
    assert r.status()['next_upload_seconds'] is None
    assert r.status()['submitted'] == 1
    r.offer(spot('K2ABC'))
    assert r.status()['next_upload_seconds'] == 300
    r.stop()
    assert r.status()['next_upload_seconds'] is None
    assert r.status()['worker_state'] == 'off'


def test_status_distinguishes_active_upload_from_future_batch():
    r, now, udp, _ = fixture()
    r.offer(spot())
    now[0] = 300
    original = udp.sendto
    def send(packet, address):
        status = r.status()
        assert status['worker_state'] == 'uploading'
        assert status['next_upload_seconds'] is None
        return original(packet, address)
    udp.sendto = send
    r._batch()
    assert r.status()['submitted'] == 1


def test_normal_shutdown_flushes_all_datagrams_without_waiting_for_batch_timer():
    r, now, udp, calls = fixture()
    for i in range(100):
        r.offer(spot(f'K{i}ABC'))
    result = r.close()
    assert result['submitted'] == 100 and result['unsent_at_shutdown'] == 0
    assert result['inflight_at_shutdown'] == 0 and now[0] == 0
    assert len(udp.packets) > 1 and len(calls) == 1
    assert all(len(packet) <= 1200 for packet in udp.packets)
    sequences = [struct.unpack('!I', packet[8:12])[0] for packet in udp.packets]
    assert sequences[0] == 0 and sequences == sorted(set(sequences))
    r.close()
    assert result['submitted'] == r.status()['submitted']


def test_shutdown_failure_is_one_attempt_and_reports_unsent():
    r, _, udp, _ = fixture()
    r.offer(spot())
    udp.fail = True
    result = r.close()
    assert result['submitted'] == 0 and result['unsent_at_shutdown'] == 1
    assert result['error'] == 'fake network down'
    assert result['queued'] == 0


def test_explicit_off_never_flushes_even_when_normal_close_follows():
    r, _, udp, calls = fixture()
    r.offer(spot())
    r.stop()
    r.close()
    assert not calls and not udp.packets


def test_shutdown_dns_deadline_prevents_later_upload():
    import threading
    import time
    r, _, udp, _ = fixture()
    entered, release = threading.Event(), threading.Event()
    original = r._resolver
    def resolve(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    r._resolver = resolve
    r.offer(spot())
    start = time.monotonic()
    result = r.close(timeout=.02)
    assert entered.is_set() and time.monotonic() - start < .5
    assert result['unsent_at_shutdown'] == 1 and result['inflight_at_shutdown'] == 0
    release.set()
    r._worker.join(2)
    assert not r._worker.is_alive() and not udp.packets


def test_shutdown_during_admitted_send_counts_unknown_separately_and_never_duplicates():
    import threading
    r, now, udp, _ = fixture()
    entered, release = threading.Event(), threading.Event()
    original = udp.sendto
    def send(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    udp.sendto = send
    for i in range(100):
        r.offer(spot(f'K{i}ABC'))
    now[0] = 300
    r._worker = threading.Thread(target=r._run, daemon=True)
    r._worker.start()
    assert entered.wait(2)
    result = r.close(timeout=.01)
    active = result['inflight_at_shutdown']
    assert active > 0 and result['unsent_at_shutdown'] + active == 100
    assert not r.offer(spot('K9ZZZ'))
    release.set()
    r._worker.join(2)
    assert len(udp.packets) == 1 and r.status()['submitted'] == active


def test_completion_accounting_is_atomic_with_shutdown_snapshot():
    import threading
    r, now, udp, _ = fixture()
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original = udp.sendto
    def send(*args):
        entered.set()
        assert release.wait(2)
        count = original(*args)
        returned.set()
        return count
    udp.sendto = send
    r.offer(spot())
    now[0] = 300
    r._worker = threading.Thread(target=r._run, daemon=True)
    r._worker.start()
    assert entered.wait(2)
    with r._condition:
        release.set()
        assert returned.wait(2)
        # UDP has returned, but the worker cannot yet commit its accounting.
        result = r.close(timeout=0)
        assert result['unsent_at_shutdown'] == 0
        assert result['inflight_at_shutdown'] == 1
    r._worker.join(2)
    assert r.status()['submitted'] == 1 and len(udp.packets) == 1
