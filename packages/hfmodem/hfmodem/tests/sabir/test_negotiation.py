# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Fixed capability bitmap, strict envelope grammar and exact sender permissions."""
import pytest
from hfmodem.sabir.arq import wire
from hfmodem.sabir.arq.fsm import ArqConfig, ArqFsm, SessionState, DISC_LINK_FAILED
from hfmodem.sabir.arq.modem import LinkModem, HEADER_SAMPLES
from hfmodem.sabir.arq.profiles import EXTENDED
from hfmodem.sabir.frame.codec import crc16


def _fsm(name, **kw):
    class IO:
        def __init__(self): self.controls = []; self.datas = []
        def send_control(self, c): self.controls.append(c); return 8.72
        def send_data(self, *args): self.datas.append(args); return 8.0
        def connected(self, *args): pass
        def disconnected(self): pass
        def deliver(self, data): pass
        def log(self, message): pass
        def state_changed(self, state): pass
    io = IO()
    return io, ArqFsm(io, ArqConfig(callsign=name, **kw), clock=lambda: 0.)


def test_complete_capability_bitmap_roundtrip():
    word = wire.capabilities(wire.DATA_IDS, wire.FASTCTL | wire.PBACK | wire.LOADING | wire.DEFLATE)
    assert word == 0x0F3F00F8
    c = wire.Control.connect(0x81, word, 'W1AW', destination='K6XYZ')
    assert c.pack()[10:12] == bytes(2)
    assert c.pack()[12:16] == word.to_bytes(4, 'big')
    assert wire.Control.unpack(c.pack()) == c
    assert wire.supported_profiles(c.capability_word) == sorted(wire.DATA_IDS)


@pytest.mark.parametrize('bit', [0, 1, 2, 8, 15, 22, 23, 29, 30, 31])
def test_unknown_bits_grant_no_profile_or_feature_permission(bit):
    word = wire.capabilities([4], wire.FASTCTL) | (1 << bit)
    assert wire.decode_capabilities(word) == wire.decode_capabilities(wire.capabilities([4], wire.FASTCTL))
    io, fsm = _fsm('BOB')
    fsm.on_host_listen(True)
    fsm.on_control(wire.Control.connect(123, word, 'ALICE', destination='BOB'))
    assert fsm.peer_capabilities == word
    assert fsm.peer_profiles == [4]
    assert fsm.use_fastctl and not fsm.use_loading and not fsm.use_deflate and not fsm.use_pback


@pytest.mark.parametrize('offset,value', [(1, 99), (10, 1), (11, 4), (40, 1), (41, 1)])
def test_connect_rejects_invalid_discriminator_and_reserved_fields(offset, value):
    raw = bytearray(wire.Control.connect(123, wire.capabilities([4]), 'ALICE', destination='BOB').pack())
    raw[offset] = value
    raw[-2:] = crc16(raw[:-2]).to_bytes(2, 'big')
    assert wire.Control.unpack(raw) is None


def test_connect_rejects_zero_session_and_truncation():
    raw = bytearray(wire.Control.connect(123, wire.capabilities([4]), 'ALICE', destination='BOB').pack())
    assert wire.Control.unpack(raw[:-1]) is None
    raw[2:10] = bytes(8)
    raw[-2:] = crc16(raw[:-2]).to_bytes(2, 'big')
    assert wire.Control.unpack(raw) is None


@pytest.mark.parametrize('profile', [3, 21, 255, 256, 65535])
@pytest.mark.parametrize('handover', [False, True])
def test_absolute_data_profile_roundtrip(profile, handover):
    c = wire.Control(wire.DATA, 1, seq=42,
                     gear=profile | (wire.HANDOVER if handover else 0),
                     mask=wire.cw_mask([0, 63]), aux=wire.data_aux(123, 64, (0,) * 8))
    assert wire.Control.unpack(c.pack()) == c


def test_sender_never_infers_intermediate_profiles_from_highest_bit():
    _, a = _fsm('ALICE')
    a.on_host_connect('BOB')
    a.on_control(wire.Control.connect(a.session, wire.capabilities([3, 7]), 'BOB', ack=True, destination='ALICE'))
    assert a.peer_profiles == [3, 7]
    assert a.tx_profile().id == 3
    a._set_rung(4)
    assert a.tx_profile().id == 7
    a._set_rung(3)
    assert a.tx_profile().id == 3


def test_single_profile_peer_is_usable_without_lower_profiles():
    _, a = _fsm('ALICE')
    a.on_host_connect('BOB')
    a.on_control(wire.Control.connect(a.session, wire.capabilities([7]), 'BOB', ack=True, destination='ALICE'))
    assert a.tx_profile().id == 7
    a._set_rung(0)
    assert a.tx_profile().id == 7


def test_no_data_permission_fails_when_host_data_is_queued():
    io, a = _fsm('ALICE')
    a.on_host_connect('BOB')
    a.on_host_data(b'hello')
    a.on_control(wire.Control.connect(a.session, 0, 'BOB', ack=True, destination='ALICE'))
    assert a.state == SessionState.DISCONNECTED and a.disc_reason == DISC_LINK_FAILED
    assert not io.datas


def test_all_profiles_negotiate_in_exactly_two_fast_blocks():
    a = LinkModem(ArqConfig(callsign='ALICE', advertise_gears=EXTENDED), clock=lambda: 0.)
    b = LinkModem(ArqConfig(callsign='BOB', advertise_gears=EXTENDED), clock=lambda: 0.)
    b.fsm.on_host_listen(True)
    a.fsm.on_host_connect('BOB')
    assert len(a.outbox) == 1 and a.outbox[0].size == a.fast.n_samples(1)
    b.on_air(a.outbox.pop())
    assert len(b.outbox) == 1 and b.outbox[0].size == b.fast.n_samples(1)
    a.on_air(b.outbox.pop())
    assert a.fsm.state == b.fsm.state == SessionState.CONNECTED
    assert a.fsm.peer_profiles == b.fsm.peer_profiles == sorted(wire.DATA_IDS)


@pytest.mark.parametrize('bandwidth,expected', [
    (500, [18, 19, 20]),
    (1500, [3, 4, 5, 17, 18, 19, 20]),
    (2300, [3, 4, 5, 17, 18, 19, 20]),
    (2750, sorted(wire.DATA_IDS)),
])
def test_bandwidth_policy_advertises_only_eligible_profiles(bandwidth, expected):
    _, f = _fsm('ALICE', bandwidth_hz=bandwidth, advertise_gears=EXTENDED)
    assert wire.supported_profiles(f._my_capabilities()) == expected
    assert f.local_profiles() == expected


def test_beacon_uses_same_network_order_capability_bitmap():
    word = wire.capabilities([3, 17, 21], wire.BEACON | wire.FASTCTL)
    beacon = wire.Beacon.build(word, 'ALICE', profile=1)
    assert beacon.pack()[2:6] == word.to_bytes(4, 'big')
    assert wire.Control.unpack(beacon.pack()) == beacon
    assert wire.decode_capabilities(word)['profiles'] == [3, 17, 21]


def test_wrong_destination_is_ignored():
    io, b = _fsm('BOB')
    b.on_host_listen(True)
    b.on_control(wire.Control.connect(123, wire.capabilities([4]), 'ALICE', destination='SOMEONE'))
    assert b.state == SessionState.LISTENING and not io.controls


def _offer(dst, ctrl, extensions=()):
    dst.begin_burst()
    for block in extensions:
        dst.on_control(wire.Control.unpack(block.pack()))
    dst.on_control(wire.Control.unpack(ctrl.pack()))


@pytest.mark.parametrize('ack', [False, True])
@pytest.mark.parametrize('critical', [False, True])
def test_optional_extension_ignored_and_critical_extension_refused(ack, critical):
    from hfmodem.sabir.arq.fsm import DISC_REFUSED
    io, f = _fsm('BOB')
    if ack:
        f.on_host_connect('ALICE')
    else:
        f.on_host_listen(True)
    session = f.session or 123
    extension = wire.Caps.build(session, wire.pack_tlvs([
        (0x40 | (wire.CRIT if critical else 0), b'future semantics')]), 1, 0)
    offer = wire.Control.connect(session, wire.capabilities([4]), 'ALICE',
                                 ack=ack, xh=1, destination='BOB')
    _offer(f, offer, [extension])
    if critical:
        assert f.state != SessionState.CONNECTED and f.disc_reason == DISC_REFUSED
        if not ack:
            assert io.controls[-1].gear & wire.CFAIL
    else:
        assert f.state == SessionState.CONNECTED and f.peer_profiles == [4]


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'session', 'count', 'position'])
def test_extensions_must_form_complete_session_bound_offer(fault):
    io, f = _fsm('BOB')
    f.on_host_listen(True)
    extensions = [wire.Caps.build(124 if fault == 'session' else 123,
                  wire.pack_tlvs([(wire.IMPL, b'identifier')]),
                  3 if fault == 'count' else 2, 1 if fault == 'position' else 0)]
    if fault == 'duplicate':
        extensions *= 2
    elif fault not in ('missing', 'position'):
        extensions += [wire.Caps.build(123, bytes(wire.CAPS_BYTES), 2, 1)]
    offer = wire.Control.connect(123, wire.capabilities([4]), 'ALICE', xh=2, destination='BOB')
    _offer(f, offer, extensions)
    assert f.state == SessionState.LISTENING and f.peer_profiles == []
    assert not io.controls


def test_future_profile_extension_cannot_override_current_bitmap():
    assert wire.parse_tlvs(wire.pack_tlvs([(wire.GEARSET, b'\0\x04')])) is None
    io, f = _fsm('BOB')
    f.on_host_listen(True)
    extra = wire.Caps.build(123, wire.pack_tlvs([(wire.GEARSET, b'\x01\x00\xff\xff')]), 1, 0)
    _offer(f, wire.Control.connect(123, wire.capabilities([4]), 'ALICE', xh=1, destination='BOB'), [extra])
    assert f.peer_profiles == [4, 256, 65535]
    assert f.tx_profile().id == 4


@pytest.mark.parametrize('count', [1, 2, 3])
def test_full_width_extensions_roundtrip(count):
    ids = b''.join(i.to_bytes(2, 'big') for i in range(256, 270))
    block = wire.Caps.build(123, wire.pack_tlvs([(wire.GEARSET, ids)]), count, count - 1)
    assert len(block.tlvs) == 31
    assert wire.Control.unpack(block.pack()) == block


def test_extension_builder_never_truncates():
    with pytest.raises(ValueError):
        wire.Caps.build(123, bytes(32), 1, 0)


def test_optional_extension_sample_exchange():
    a = LinkModem(ArqConfig(callsign='ALICE'), clock=lambda: 0.)
    b = LinkModem(ArqConfig(callsign='BOB'), clock=lambda: 0.)
    a.fsm._my_caps = lambda: [wire.Caps.build(a.fsm.session,
        wire.pack_tlvs([(0x40, bytes(range(29)))]), 1, 0)]
    b.fsm.on_host_listen(True)
    a.fsm.on_host_connect('BOB')
    assert a.outbox[0].size == 2 * HEADER_SAMPLES
    b.on_air(a.outbox.pop())
    a.on_air(b.outbox.pop())
    assert a.fsm.state == b.fsm.state == SessionState.CONNECTED


@pytest.mark.parametrize('name', ['robust', 'workhorse', 'workhorse34', 'fast', 'max', 'doppler', 'sparse34', 'narrow', 'narrow2', 'narrow4', 'wide256'])
def test_data_preference_accepts_every_implemented_profile(name):
    _, a = _fsm('ALICE', data_profile=name)
    a.on_host_connect('BOB')
    a.on_control(wire.Control.connect(a.session, wire.capabilities(wire.DATA_IDS), 'BOB', ack=True, destination='ALICE'))
    assert a.tx_profile().name == name


def test_failed_baseline_preference_falls_back_without_sticking():
    io, a = _fsm('ALICE', data_profile='workhorse')
    a.on_host_connect('BOB')
    a.on_control(wire.Control.connect(a.session, wire.capabilities([3, 4]), 'BOB', ack=True, destination='ALICE'))
    a.on_host_data(b'prefer workhorse, rebuild robust')
    assert a._frame.profile == 'workhorse'
    a._rebuild()
    assert a._frame.profile == 'robust'


@pytest.mark.parametrize('listener_config', [
    {'bandwidth_hz': 500, 'advertise_gears': EXTENDED},
    {'fast_ctrl': False},
    {'extensions': True},
])
def test_fast_probe_falls_back_for_narrow_floor_only_or_extended_listener(listener_config):
    config = dict(listener_config)
    extensions = config.pop('extensions', False)
    a = LinkModem(ArqConfig(callsign='ALICE'), clock=lambda: 0.)
    b = LinkModem(ArqConfig(callsign='BOB', **config), clock=lambda: 0.)
    if extensions:
        b.fsm._my_caps = lambda: [wire.Caps.build(b.fsm.session, wire.pack_tlvs([(wire.IMPL, b'id')]), 1, 0)]
    b.fsm.on_host_listen(True)
    a.fsm.on_host_connect('BOB')
    assert a.fsm.bootstrap_fast
    b.on_air(a.outbox.pop())
    assert b.fsm.state == SessionState.LISTENING and not b.outbox
    a.fsm.on_timer(a.fsm.next_deadline())
    assert not a.fsm.bootstrap_fast
    b.on_air(a.outbox.pop())
    a.on_air(b.outbox.pop())
    assert a.fsm.state == b.fsm.state == SessionState.CONNECTED


def test_lost_fast_connect_ack_recovers_same_session_on_floor():
    a = LinkModem(ArqConfig(callsign='ALICE'), clock=lambda: 0.)
    b = LinkModem(ArqConfig(callsign='BOB'), clock=lambda: 0.)
    b.fsm.on_host_listen(True)
    a.fsm.on_host_connect('BOB')
    original_session = a.fsm.session
    b.on_air(a.outbox.pop())
    assert b.fsm.bootstrap_fast and b.fsm.state == SessionState.CONNECTED
    b.outbox.clear()
    a.fsm.on_timer(a.fsm.next_deadline())
    assert not a.fsm.bootstrap_fast and a.fsm.session == original_session
    b.on_air(a.outbox.pop())
    assert not b.fsm.bootstrap_fast
    a.on_air(b.outbox.pop())
    assert a.fsm.state == SessionState.CONNECTED
    assert sum('CONNECTED ALICE session=' in event for event in b.events) == 1


def test_floor_handshake_timer_covers_longest_legal_extension_reply():
    from hfmodem.sabir.phy.modem import FS
    now = [0.]
    a = LinkModem(ArqConfig(callsign='ALICE', fast_ctrl=False), clock=lambda: now[0])
    b = LinkModem(ArqConfig(callsign='BOB'), clock=lambda: now[0])
    b.fsm._my_caps = lambda: [wire.Caps.build(b.fsm.session,
        wire.pack_tlvs([(wire.IMPL, b'optional metadata')]), wire.MAX_CAPS, i)
        for i in range(wire.MAX_CAPS)]
    b.fsm.on_host_listen(True)
    a.fsm.on_host_connect('BOB')
    request = a.outbox.pop()
    now[0] = request.size / FS + a.fsm.cfg.turnaround_s
    b.on_air(request)
    reply = b.outbox.pop()
    reply_start = now[0] + b.fsm.cfg.turnaround_s
    # Fire timers after each full reply block. The old fixed 15-second wait
    # retransmitted while the extension blocks were still being received.
    for block in range(1, wire.MAX_CAPS + 2):
        now[0] = reply_start + block * HEADER_SAMPLES / FS
        a.fsm.on_timer()
        assert a.fsm.state == SessionState.CONNECTING and not a.outbox
    assert now[0] < a.fsm.next_deadline()
    a.on_air(reply)
    assert a.fsm.state == SessionState.CONNECTED


def test_floor_handshake_still_retries_when_bounded_reply_window_expires():
    now = [0.]
    a = LinkModem(ArqConfig(callsign='ALICE', fast_ctrl=False), clock=lambda: now[0])
    a.fsm.on_host_connect('BOB')
    a.outbox.clear()
    deadline = a.fsm.next_deadline()
    assert deadline == pytest.approx(45.6)
    now[0] = deadline - .001
    a.fsm.on_timer()
    assert not a.outbox
    now[0] = deadline
    a.fsm.on_timer()
    assert len(a.outbox) == 1 and a.fsm.state == SessionState.CONNECTING


def test_configured_connect_timeout_can_increase_reply_budget():
    _, a = _fsm('ALICE', fast_ctrl=False, connect_timeout_s=90.)
    a.on_host_connect('BOB')
    assert a.next_deadline() == pytest.approx(98.72)
