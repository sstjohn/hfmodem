# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
import pytest
from hfmodem.contact.filetransfer import RawTransferCapture, CaptureStopped, parse_offer

HEADER = '<SF:weather.txt.VarAC_gzip|1031|8B314EE2A7897B0477459E307F761967>'


def test_offer_preserves_uninterpreted_fields():
    offer = parse_offer(HEADER)
    assert offer.raw == HEADER
    assert offer.wire_name == 'weather.txt.VarAC_gzip'
    assert offer.declared_integer == 1031
    assert offer.digest_text == '8B314EE2A7897B0477459E307F761967'


@pytest.mark.parametrize('name', ['../file', '/file', 'a/b', 'a\\b', 'C:file', '.', '..',
                                 'a\nfile', 'a\x1bfile', ' trailing', 'trailing ', 'a>'])
def test_offer_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        parse_offer(HEADER.replace('weather.txt.VarAC_gzip', name))


@pytest.mark.parametrize('text', [HEADER+'tail', HEADER[:-1], HEADER.replace('1031', '-1'),
                                HEADER.replace('1031', '2147483648'),
                                HEADER.replace('1031', '9'*1000),
                                HEADER.replace('8B314', 'ZZZZZ')])
def test_invalid_offer(text):
    with pytest.raises(ValueError):
        parse_offer(text)


class Clock:
    now = 100.0
    def __call__(self):
        return self.now


def capture(tmp_path, clock=None, **kwargs):
    return RawTransferCapture(tmp_path, peer='K6HN', filename='weather.txt',
                              link_generation=3, clock=clock or Clock(), **kwargs)


def events(c):
    return [json.loads(line) for line in (c.directory/'chunks.jsonl').read_text().splitlines()]


def test_exact_bytes_order_offsets_and_partial_finish(tmp_path):
    clock = Clock()
    c = capture(tmp_path, clock)
    c.append('rx', b'\x00\xff<')
    clock.now += .5
    c.append('tx', b'6 <SFRD>')
    c.append('rx', b'SFB>\x1f\x8b')
    result = c.finish('operator stopped')
    assert c.finish() is result
    assert not result.validated_download
    assert (c.directory/'transport.bin').read_bytes() == b'\x00\xff<6 <SFRD>SFB>\x1f\x8b'
    chunks = events(c)[1:-1]
    assert [e['offset'] for e in chunks] == [0, 3, 11]
    assert [e['direction_offset'] for e in chunks] == [0, 0, 3]
    assert [e['sequence'] for e in chunks] == [0, 1, 2]
    assert chunks[1]['elapsed_s'] == .5
    with pytest.raises(CaptureStopped):
        c.append('rx', b'more')


def test_cap_preserves_prefix_and_reports_observed_length(tmp_path):
    c = capture(tmp_path, byte_limit=5)
    c.append('tx', b'ab')
    with pytest.raises(CaptureStopped) as stopped:
        c.append('rx', b'cdefgh')
    assert (c.directory/'transport.bin').read_bytes() == b'abcde'
    assert stopped.value.result.truncated
    assert events(c)[-2]['observed_length'] == 6
    assert stopped.value.result.received_bytes == 3


def test_deadline_without_rx_keeps_partial(tmp_path):
    clock = Clock()
    c = capture(tmp_path, clock, time_limit=2)
    c.append('rx', b'partial')
    clock.now += 2
    with pytest.raises(CaptureStopped) as stopped:
        c.check_deadline()
    assert stopped.value.result.reason == 'deadline'
    assert (c.directory/'transport.bin').read_bytes() == b'partial'


@pytest.mark.parametrize('kwargs', [{'byte_limit': 0}, {'byte_limit': 65537},
                                  {'byte_limit': 1.5}, {'time_limit': 121},
                                  {'time_limit': 0}])
def test_cannot_raise_capture_bounds(tmp_path, kwargs):
    with pytest.raises(ValueError):
        capture(tmp_path, **kwargs)


def test_write_failure_closes_retains_and_reports(tmp_path, monkeypatch):
    c = capture(tmp_path)
    c.append('rx', b'kept')
    monkeypatch.setattr(c, '_write_event', lambda value: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        c.append('rx', b'last')
    assert c.result.reason == 'write error'
    assert c._raw.closed and c._events.closed
    assert (c.directory/'transport.bin').read_bytes() == b'keptlast'
    assert not c.result.validated_download
