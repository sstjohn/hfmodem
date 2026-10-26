# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Serialized SCS CW policy reaches the native producer without delivery credit."""
import numpy as np
import pytest

from hfhost import ptc
from hfmodem.core import cwid, levels
from hfmodem.shrike import arq, host_cwid, onair
from hfmodem.shrike.ptc import PtcHost
from .test_identify import transmitter, CALL
from .test_ptc_compatibility import Client
from .test_station_runtime import running as running, Client as SerialClient, eventually as eventually


def test_trimode_audio_cw_profile_has_effective_readback_and_bounded_speed():
    host = PtcHost(mycall=CALL)
    assert b'ERROR' not in host.feed(b'CWS 70\r')
    client = Client(host)
    assert client.cmd('#CWID 4 2').code == ptc.OK
    assert client.retry().code == ptc.OK and host.settings['CWID'] == '4 2'
    before = dict(host.settings)
    for command in ['#CWS 101', '#CWID 4 0', '#MYCALL N0CALL-1']:
        assert client.cmd(command).code == ptc.FAIL
        assert host.settings == before
    calls = dict(host._callsigns)
    assert client.cmd('I N0CALL-1').code == ptc.FAIL
    assert host._callsigns == calls
    host.arq.state = arq.State.CONNECTED
    for command in ['#MYCALL N1ABC', 'I N1ABC', '#CWID 0']:
        assert client.cmd(command).code == ptc.FAIL
        assert host.settings == before and host._callsigns == calls
    assert client.cmd('#CWID 4 2').code == ptc.OK
    host.arq.state = arq.State.LISTENING
    assert client.cmd('JHOST0').code == ptc.OK
    assert b'CWID: 4 2' in host.feed(b'CWID\r')


def test_station_refuses_requested_cw_for_a_different_configured_identity(running):
    runtime, station, server = running
    assert station.arbiter.identity.mycall == 'K1ABC'
    client = SerialClient(server.endpoint)
    try:
        assert client.command('#CWS 70').code == ptc.OK
        assert client.command('#CWID 4 2').code == ptc.FAIL
        assert runtime.host.settings['CWID'] == '0 0'
        assert not station.audio.sent and not station.rig.keyed
    finally:
        client.close()


def test_interval_uses_sample_clock_and_only_completed_output_renews_it():
    now = [100]
    schedule = host_cwid.Schedule(CALL, lambda: now[0])
    assert not schedule.due and not schedule.defer_data()
    schedule.note_transmission(100)
    assert schedule.deadline == 100 + 420*48000
    now[0] = schedule.deadline - 1
    schedule.note_transmission(now[0])
    assert not schedule.due  # Ordinary DATA does not postpone identification.
    now[0] += 1
    assert schedule.defer_data() and schedule.waiting and schedule.completed == 0
    now[0] += 4*48000
    assert schedule.defer_data() and schedule.waiting  # A request is not completion.
    schedule.identified(now[0])
    assert schedule.completed == 1 and not schedule.waiting and not schedule.due
    assert schedule.deadline == now[0] + 420*48000


def pending_interval(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    host = PtcHost(tx, mycall=CALL)
    host.feed(b'CWS 70\rFSKA 1500\rCWID 4 2\r')
    client = Client(host)
    host.arq.state, host.arq.role = arq.State.CONNECTED, arq.ISS
    host.arq.dxcall = 'N0DX'
    schedule = host_cwid.Schedule(CALL, lambda: tx.live.samples)
    host._cw_schedule = tx._host_cw_schedule = schedule
    schedule.note_transmission(tx.live.samples)
    # Inject interval expiry; the independent clock test above checks 420 s.
    schedule.deadline = tx.live.samples
    original = b'not acknowledged\x00\xff\x1c'
    assert client.exchange(original, False).code == ptc.OK
    host.arq._start_next_packet()
    assert schedule.waiting and not tx.live.sent
    return tx, host, schedule, original


def test_expired_interval_defers_data_and_renders_actual_1400_hz_fourteen_wpm_cw(tmp_path):
    tx, host, schedule, original = pending_interval(tmp_path)
    pending = host.arq._inflight
    before = (host.arq.payload_ledger.accepted, host.arq.payload_ledger.confirmed,
              host.arq.payload_ledger.pending)
    onair._service_host_cwid(tx, host, tx.live, tx.raster, tx.slot)
    assert len(tx.live.sent) == 1 and tx.rig.edges == [True, False]
    expected = levels.at_drive(cwid.audio(CALL, wpm=14, tone_hz=1400),
                               tx._fsk_drive * 10**(-3/20))
    expected = onair._trim_silence(expected)
    np.testing.assert_array_equal(tx.live.sent[-1], expected)
    assert schedule.completed == 1 and not schedule.waiting
    assert schedule.deadline == tx.tx_end + 420*48000
    assert host.arq._inflight is pending
    assert (host.arq.payload_ledger.accepted, host.arq.payload_ledger.confirmed,
            host.arq.payload_ledger.pending) == before == (len(original), 0, len(original))


def test_interval_never_interrupts_the_peer_turn(tmp_path):
    tx, host, schedule, _ = pending_interval(tmp_path)
    host.arq.role = arq.IRS
    slot = tx.slot
    assert onair._service_host_cwid(tx, host, tx.live, tx.raster, slot) == slot
    assert not tx.live.sent and schedule.waiting and schedule.completed == 0


def test_refused_cw_preserves_interval_obligation_and_pending_originals(tmp_path, capsys):
    tx, host, schedule, original = pending_interval(tmp_path)
    tx.max_key = .1
    with pytest.raises(ValueError, match='identification did not complete'):
        onair._service_host_cwid(tx, host, tx.live, tx.raster, tx.slot)
    assert 'max-key' in capsys.readouterr().out
    assert not tx.live.sent and not tx.rig.edges
    assert schedule.waiting and schedule.due and schedule.completed == 0
    assert host.arq.payload_ledger.pending == len(original)
    assert host.arq.payload_ledger.confirmed == 0


@pytest.mark.parametrize('answering', [False, True])
def test_serial_cwid_profile_drives_native_closing_audio_and_retires_its_contact(running, answering):
    from hfmodem.shrike import pactor1
    runtime, station, server = running
    station.arbiter.identity.mycall = 'N0CALL'  # Synthetic station, matching digital/CW identities.
    client = SerialClient(server.endpoint)
    try:
        assert client.command('#CWS 70').code == ptc.OK
        assert client.command('#FSKA 750').code == ptc.OK
        assert client.command('#CWID 4 2').code == ptc.OK
        if answering:
            signal = pactor1.connect_signal('N0CALL', invert=True, lead_s=0, tail_s=0)
            at = station.audio.samples + 24000
            def source(first, count):
                block = np.zeros(count, np.float32)
                lo, hi = max(first, at), min(first+count, at+len(signal))
                if hi > lo:
                    block[lo-first:hi-first] = signal[lo-at:hi-at]
                return block
            station.audio.source = source
        else:
            assert client.command('C N0DX').code == ptc.OK
        eventually(lambda: station.rig.keyed, timeout=10)
        tx = runtime.tx
        assert runtime.host._cw_schedule is tx._host_cw_schedule
        assert runtime.host._cw_schedule is not None
        assert client.command('#CWID 0').code == ptc.FAIL
        assert client.command('DD').code == ptc.OK
        eventually(lambda: len(station.audio.sent) >= 2 and not runtime.active, timeout=15)
        assert len(station.audio.sent) == 2
        expected = onair._trim_silence(levels.at_drive(
            cwid.audio('N0CALL', wpm=14, tone_hz=1400), tx._fsk_drive * 10**(-3/20)))
        np.testing.assert_allclose(station.audio.sent[-1][1], expected, rtol=1e-6, atol=2e-8)
        assert station.arbiter.last_sent['ident'] == station.audio.sent[-1][0] + len(expected)
        assert not station.arbiter.identity.due and not station.rig.keyed
        assert runtime.host._cw_schedule is tx._host_cw_schedule is None
        assert runtime.failure is None and station.audio.failure is None
    finally:
        client.close()


# The independent card presents DATA/QRT on fixed 1.25-second boundaries.
# A loaded decoder can miss the source entirely; retain its actual 25-second
# deadline in the mandatory serial nice-0 qualification slice.
@pytest.mark.wall_clock
def test_peer_qrt_after_binary_receive_emits_the_requested_closing_cw(running):
    from hfmodem.shrike import compress, pactor1
    runtime, station, server = running
    station.arbiter.identity.mycall = 'N0CALL'
    client = SerialClient(server.endpoint)
    try:
        assert client.command('#CWS 70').code == ptc.OK
        assert client.command('#CWID 4 2').code == ptc.OK
        assert client.command('#MYL 1').code == ptc.OK
        original = b'\x00\xff'
        data = b'1n0dx\r' + compress.transparent(original)
        bursts = [
            (0, pactor1.connect_signal('N0CALL', invert=True, lead_s=0, tail_s=0)),
            (60000, pactor1.packet_signal(data, 200, packet_count=1,
                                         lead_s=0, tail_s=0)),
            (120000, pactor1.packet_signal(data, 200, packet_count=1,
                                          invert=True, lead_s=0, tail_s=0)),
            (180000, pactor1.packet_signal(pactor1.qrt_field('N0CALL', 200), 200,
                packet_count=2, qrt=True, lead_s=0, tail_s=0)),
        ]
        at = station.audio.samples + 24000
        def source(first, count):
            block = np.zeros(count, np.float32)
            for offset, samples in bursts:
                start = at + offset
                lo, hi = max(first, start), min(first + count, start + len(samples))
                if hi > lo:
                    block[lo-first:hi-first] += samples[lo-start:hi-start]
            return block
        station.audio.source = source
        received, links, contact_tx = bytearray(), [], None
        def retired():
            nonlocal contact_tx
            reply = client.command('G')
            if reply.code == ptc.DATA:
                received.extend(reply.data)
            elif reply.code == ptc.LINK:
                links.append(reply.text)
                if len(links) == 1:
                    contact_tx = runtime.tx
            assert runtime.failure is None and station.audio.failure is None
            return len(links) == 2 and not runtime.active
        eventually(retired, timeout=25)
        assert received == original
        assert links == ['(31) CONNECTED to N0DX', '(31) DISCONNECTED fm N0DX']
        assert runtime.host.arq.state == arq.State.LISTENING
        assert not runtime.host.arq.transfer_unconfirmed and not station.rig.keyed
        expected = onair._trim_silence(levels.at_drive(
            cwid.audio('N0CALL', wpm=14, tone_hz=1400),
            contact_tx._fsk_drive * 10**(-3/20)))
        np.testing.assert_allclose(station.audio.sent[-1][1], expected,
                                   atol=2e-8, rtol=1e-6)
        assert runtime.host._cw_schedule is contact_tx._host_cw_schedule is None
        assert runtime.host.sent_total == 0
    finally:
        client.close()
