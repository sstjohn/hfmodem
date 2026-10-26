# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Channel changes retire samples and decoded candidates across all owners."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.station.arbiter import TxArbiter
from hfmodem.station.contact import ContactOwner


def test_scan_lease_retires_candidates_from_before_and_during_the_hold():
    audio = SimpleNamespace(samples=10)
    arb = TxArbiter(audio, None)
    before = arb.receive_boundary[0]
    token = arb.acquire_scan_hold('pactor', eligible=lambda: True)
    during = arb.receive_boundary[0]
    assert during > before and arb.receive_boundary[1] == 10
    assert arb.claim_contact('vara', receive_epoch=during) is None
    audio.samples = 100
    arb.release_scan_hold(token)
    after, floor = arb.receive_boundary
    assert after > during and floor == 100
    for epoch in (before, during):
        assert arb.claim_contact('vara', receive_epoch=epoch) is None
    assert arb.claim_contact('vara', receive_epoch=after) is not None


def test_failed_retune_still_retires_the_old_channel_without_reusing_a_scan_token():
    arb = TxArbiter(SimpleNamespace(samples=20), None)
    initial = arb.receive_boundary
    with pytest.raises(ValueError, match='tune failed'):
        with arb.try_hold() as held:
            assert held and arb.receive_boundary[0] > initial[0]
            raise ValueError('tune failed')
    final = arb.receive_boundary
    assert final[0] == initial[0]+2
    assert arb.claim_contact('rpr', receive_epoch=initial[0]) is None
    token = arb.acquire_scan_hold('pactor', eligible=lambda: True)
    boundary = arb.receive_boundary
    with pytest.raises(ValueError, match='does not belong'):
        arb.release_scan_hold(object())
    assert arb.receive_boundary == boundary
    arb.release_scan_hold(token)


def test_legacy_active_lane_and_reserved_contact_refuse_retune_without_a_new_epoch():
    arb = TxArbiter(None, None)
    initial = arb.receive_boundary
    arb.active_protocols = lambda: ('legacy',)
    with arb.try_hold() as held:
        assert not held
    arb.active_protocols = lambda: ()
    owner = ContactOwner(arb, 'ardop')
    assert owner.admit()
    with arb.try_hold() as held:
        assert not held
    assert arb.receive_boundary == initial
    owner.close()


def test_vara_retires_queued_capture_and_request_state_before_readmission():
    from hfmodem.kestrel.host.station_runtime import StationTransport
    st = SimpleNamespace(audio=SimpleNamespace(samples=0))
    st.arbiter = TxArbiter(st.audio, None)
    transport = StationTransport(SimpleNamespace(station=st))
    received, resets = [], []
    hs = SimpleNamespace(_reset_stream=lambda: resets.append(True),
                         on_rx_stream=lambda samples: received.extend(samples))
    transport.hs = hs
    transport.feed(0, np.ones(10))
    transport._bursts.append(np.ones(3))
    token = st.arbiter.acquire_scan_hold('pactor', eligible=lambda: True)
    transport.feed(10, np.ones(10)*2)
    st.audio.samples = 20
    st.arbiter.release_scan_hold(token)
    applicant = object()
    assert not transport.admit_contact(applicant)
    transport.before_step()
    assert resets == [True] and not transport._bursts
    assert transport.next_rx_burst(.01, hs) is None and not received
    transport.feed(20, np.zeros(10))
    assert transport.next_rx_burst(.01, hs) is None
    assert received == [0]*10
    assert transport.admit_contact(applicant)
    transport.retire_contact(applicant, idle=True)


def test_ardop_rejects_a_stale_channel_frame_even_with_the_current_attachment_generation():
    from hfmodem.besra.station_runtime import StationArdopModem
    from hfmodem.tests.besra.test_station_runtime import Observer, wait_for
    arb = TxArbiter(None, None)
    modem = StationArdopModem(lambda _: None, contact=ContactOwner(arb, 'ardop'))
    received = []
    modem.core.receive_frames = received.extend
    modem.start(Observer())
    try:
        old = modem.channel_epoch
        token = arb.acquire_scan_hold('pactor', eligible=lambda: True)
        arb.release_scan_hold(token)
        wait_for(lambda: modem.channel_epoch == arb.receive_boundary[0])
        modem.deliver_frames(['old channel'], channel_epoch=old)
        modem.deliver_frames(['new channel'])
        wait_for(lambda: received)
        assert received == ['new channel']
    finally:
        modem.stop()


def test_tracker_discards_queued_sabm_across_scan_release_without_key_or_credit():
    from hfmodem.brubru.host.runtime import PacketRuntime
    from hfmodem.tests.brubru.test_connected_hosts import Link
    from hfmodem.ax25.connected import SABM, State
    from hfmodem.ax25.frame import Frame, address
    arb = TxArbiter(None, None)
    link = Link()
    link.station = SimpleNamespace(arbiter=arb)
    runtime = PacketRuntime(link, mycall='W9SSJ', contact=ContactOwner(arb, 'rpr'))
    runtime.engine.listen = True
    raw = Frame((address('W9SSJ', flag=True), address('N0CALL', last=True)),
                SABM|0x10, None, b'').encode()
    before = arb.receive_boundary[0]
    runtime.send_frame(raw)
    token = arb.acquire_scan_hold('pactor', eligible=lambda: True)
    runtime.send_frame(raw)
    arb.release_scan_hold(token)
    runtime.send_frame(raw, channel_epoch=before)
    runtime._step()
    assert runtime.engine.state is State.DISCONNECTED
    assert not link.requests and not runtime.engine.controls
    assert runtime.engine.accepted == runtime.engine.delivered == 0
    runtime.send_frame(raw)
    runtime._step()
    assert runtime.engine.state is State.CONNECTED and len(link.requests) == 1
    runtime.stop()


def test_pactor_capture_and_already_decoded_call_cannot_survive_a_channel_change(monkeypatch):
    from hfmodem.tests.shrike.test_station_io import port, chunk
    from hfmodem.shrike.station_runtime import StationPtcRuntime
    from hfmodem.station.arbiter import Identity
    live, st = port()
    live.feed(chunk(0, [1, 2, 3]))
    with st.arbiter.try_hold() as held:
        assert held
        st.audio.samples = 30
    with pytest.raises(ConnectionAbortedError, match='scanner hold'):
        live._check_state()
    st.arbiter.identity = Identity('N0CALL')
    runtime = StationPtcRuntime(SimpleNamespace(station=st))
    runtime._port = live
    retired = []
    monkeypatch.setattr(runtime, '_answer_timing_ready', lambda: True)
    monkeypatch.setattr(runtime, '_retire_port', lambda: retired.append(True))
    runtime._contact(('decoded before retune', object()))
    assert retired == [True] and not st.rig.edges
    assert st.arbiter.contact_protocol is None
    live.close()


def test_rpr_discards_old_pcm_and_does_not_relabel_old_decoder_output(monkeypatch):
    from hfmodem.tests.brubru.test_station import station, feed
    st = station()
    delivered, positions = [], []
    st.link._on_frame = delivered.append
    old = st.arbiter.receive_boundary[0]
    with st.arbiter.try_hold() as held:
        assert held
        st.audio.samples = 100
    original = st.link.decimator.feed
    def decimate(at, samples):
        positions.append((at, len(samples)))
        return original(at, samples)
    monkeypatch.setattr(st.link.decimator, 'feed', decimate)
    feed(st.link, np.zeros(110), start=0)
    assert positions == [(100, 10)]
    st.link.on_frame(b'old', channel_epoch=old)
    st.link.on_frame(b'new', channel_epoch=st.arbiter.receive_boundary[0])
    assert delivered == [b'new']


def test_rpr_worker_retires_old_assessment_even_without_another_capture_poll():
    from hfmodem.tests.brubru.test_station import station
    st = station()
    st.link.access.feed(0, np.random.default_rng(731).normal(0, .001, 3*48000))
    assert st.link.access.floor is not None
    st.audio.samples = 3*48000
    token = st.arbiter.acquire_scan_hold('pactor', eligible=lambda: True)
    pending = st.link.enqueue(b'UI during scan')
    assert pending.completion == ('refused', 'station scanner lease held')
    st.audio.samples += 100
    st.arbiter.release_scan_hold(token)
    st.link._fresh()
    assert st.link.access.floor is None and not st.link._eligible()
    assert not st.rig.assertions


def test_queued_rpr_packet_is_not_reinterpreted_on_a_new_channel():
    from hfmodem.tests.brubru.test_station import station, ui_frame
    st = station()
    request = st.link.enqueue(ui_frame('N0CALL', 'CQ', b'old channel'))
    # The station's active-lane snapshot normally refuses this retune. Force
    # the race window in which the queue was published after that snapshot.
    with st.arbiter.try_hold() as held:
        assert held
    st.link._serve(request)
    assert request.completion == ('refused', 'channel changed before packet transmission')
    assert not st.rig.assertions
