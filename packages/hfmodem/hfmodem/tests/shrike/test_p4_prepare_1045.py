# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Pin the chirp-data/BODY preparation failure at K4MSU on 1001-1045.

The authenticated, charged driver replay uses real receiver registrations.
This is a bounded conditional cache control: the earlier chirp keys differ
from the recording, so later peer audio does not predict a changed exchange.
No later state or grant is injected, and the replay stops before a retry key.
"""
import pytest

from hfmodem.shrike import p4waveforms
from hfmodem.shrike.p4observation import Kind
from hfmodem.tests.shrike.p4_chirp_deadline_support import FS, replay


GRANT_KEY = 3660622  # 76.26295833333333 s in the original stream coordinates


def assert_grant_body(result):
    refusals = [e for e in result['events'] if e['event'] == 'Refused'
                and e.get('kind') == Kind.BODY]
    bodies = [e for e in result['events'] if e['event'] == 'Keyed'
              and e.get('kind') == Kind.BODY and e.get('state') == 'ISS_DATA_ACK']
    assert [e['at'] for e in bodies] == [GRANT_KEY], (
        '1045 grant BODY did not key its admitting slot', bodies, refusals)
    source = bytes.fromhex(result['source'])
    body, = bodies
    assert body['status'] == 0 and bytes.fromhex(body['payload']) == source[:31]
    assert result['buffered'] == len(source)
    assert not refusals
    # The receiver actually decoded the recorded turn-grant BODY before keying.
    assert any(e['kind'] == Kind.BODY and e.get('status') == 0x40
               and e['onset_sample'] == 3616530 and not e['provisional']
               for e in result['heard'])
    # A 31-byte BODY must be rendered ahead of admission, even though the
    # selected data form has a 22-byte chirp field.
    renders = [m for m in result['misses'] if m['builder'] == 'changeover_packet'
               and m['payload'] == source[:31].hex()]
    assert renders and all(m['at'] < GRANT_KEY-result['lead'] for m in renders)


def test_1045_chirp_data_prepares_body_for_recorded_grant():
    assert_grant_body(replay('onair-1001-1045', stop=77*FS))


def test_1045_pin_detects_omitted_anticipatory_body_renders(monkeypatch):
    """In-memory pre-hotfix behavior: prepare only chirps until BODY is pending.

    Keep explicit pending BODY keys and all chirp variants. This removes only
    the anticipatory BODY matrix added by f1d1b57c, now in reachable_keys after
    24b32314; it does not fake a cache miss or alter the receiver/turn machine.
    """
    reachable = p4waveforms.reachable_keys

    def without_anticipatory_body(machine):
        keys = reachable(machine)
        packets = (machine._next, machine._data_packet)
        if machine.data_kind == Kind.CHIRP and not any(
                p is not None and p.kind == Kind.BODY for p in packets):
            keys = {key for key in keys if key[0] != Kind.BODY}
        return keys

    with monkeypatch.context() as mutation:
        mutation.setattr(p4waveforms, 'reachable_keys', without_anticipatory_body)
        result = replay('onair-1001-1045', stop=77*FS)
        with pytest.raises(AssertionError, match='1045 grant BODY did not key'):
            assert_grant_body(result)
        assert any(e['event'] == 'Refused' and e.get('kind') == Kind.BODY
                   and e['at'] == GRANT_KEY and e['reason'] == 'waveform not prepared'
                   for e in result['events'])
    assert p4waveforms.reachable_keys is reachable
