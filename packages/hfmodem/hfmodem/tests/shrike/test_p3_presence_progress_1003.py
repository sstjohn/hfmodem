# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Qualified absence and ACK-gated attached-mail progress; virtual elapsed time."""
import numpy as np
import pytest
from unittest.mock import Mock
from hfmodem.shrike import arq, spec, p3frame, placement, rxfront
from hfmodem.shrike.p3presence import tracked_structure
from hfmodem.shrike.p3progress import MailProgress
from hfmodem.winlink import B2FSession, MailClient, compose
from hfmodem.winlink.session import _GREETING, _ANSWER_HS, _WAIT_FS, _THEIR_TURN, _RECEIVING

FS = 48000


class IO(arq.ArqIO):
    protocol = spec.Protocol.PACTOR3
    def __init__(self):
        self.events, self.lines, self.received = [], [], []
    def send_cs(self, word):
        self.events.append(('cs', word))
    def send_packet(self, sl, payload, status, breakin=False):
        self.events.append(('packet', len(payload)))
        return len(payload)
    def log(self, text):
        self.lines.append(text)
    def deliver(self, blob):
        self.received.append(blob)


def modem():
    io = IO()
    a = arq.PactorArq(io, arq.ArqConfig(max_retries=200))
    a._enter_connected()
    a.role = arq.IRS
    return a, io


def opportunity(a, *, source=None, weak=False, generation=None, tick=True):
    token = a.begin_receive_opportunity()
    if weak:
        a.note_unreadable_answer()
        a.note_peer_carrier()
        a.note_burst(100., at_anchor=True)
        a.note_peer_raster_burst()
    a.qualify_peer(token, a._link_generation if generation is None else generation, source)
    present = a.settle_peer_presence()
    if present and tick:
        a.on_cycle()
    a.finish_receive_opportunity(token)
    return present


def mail(phase=_WAIT_FS, seconds=None, with_message=True):
    a, io = modem()
    message = compose('W9SSJ', 'WS8EOC', 'test', 'body\r\n', mid='PROGRESS120')
    session = B2FSession('W9SSJ', target='WS8EOC', outbox=[message] if with_message else [])
    clock = [0.]
    client = MailClient(session, a.on_host_data)
    config = {} if seconds is None else dict(seconds=seconds)
    progress = client.progress = MailProgress(client, a, now=lambda: clock[0], **config)
    session._state = phase
    if phase == _WAIT_FS:
        out = bytearray()
        session._our_turn(out)
        client._push(bytes(out))
    elif phase == _THEIR_TURN:
        client._push(b'FF\r')
    else:
        progress.update()
    return a, io, client, progress, clock


def acknowledge_all(a):
    a.role = arq.ISS
    while a._outbuf or a._inflight is not None and a._inflight.payload:
        if a._inflight is None:
            a._start_next_packet()
        a._on_ack()
    a.role = arq.IRS


@pytest.mark.parametrize('weak', [False, True])
def test_sixth_unsupported_window_never_keys_even_with_every_occupancy_bypass(weak):
    a, io = modem()
    a.on_host_data(b'unconfirmed')
    for _ in range(5):
        assert opportunity(a, weak=weak)
    count = len(io.events)
    assert opportunity(a, weak=weak)
    assert a.p3_receive_only and bytes(a._outbuf) == b'unconfirmed'
    for _ in range(89):
        assert opportunity(a, weak=weak)
        assert len(io.events) == count
    assert not opportunity(a, weak=weak)
    assert len(io.events) == count and a.state == arq.State.DISCONNECTED
    assert a.transfer_unconfirmed['suffix'] == b'unconfirmed'
    assert a.transfer_unconfirmed['field'] == b''


def test_current_structure_resets_absence_without_crc_or_delivery_credit():
    a, io = modem()
    for _ in range(5):
        opportunity(a)
    assert opportunity(a, source='tracked SL2 short header')
    assert a._dark_windows == 0 and a._p3_crc_wait_ticks == 6
    assert not io.received and a.acked == 0
    for _ in range(5):
        assert opportunity(a)


def test_stale_generation_and_revisited_opportunity_cannot_reset_or_double_count():
    a, _ = modem()
    token = a.begin_receive_opportunity()
    assert not a.qualify_peer(token, a._link_generation-1, 'current CRC')
    assert a.settle_peer_presence()
    assert a._dark_windows == 1
    a.settle_peer_presence()
    assert a._dark_windows == 1
    assert not a.qualify_peer(token, a._link_generation, 'late structure')
    a.finish_receive_opportunity(token)
    assert a._dark_windows == 1


@pytest.mark.parametrize('phase', [_WAIT_FS, _THEIR_TURN])
def test_62_and_110_second_supported_cms_waits_survive_default(phase):
    a, _, client, timer, now = mail(phase)
    acknowledge_all(a)
    assert not timer.poll(qualified=True)
    assert timer.deadline == 120.
    for seconds in (43.7525, 62., 110., 119.999):
        now[0] = seconds
        assert not timer.poll(qualified=True)
        assert not client.session.failure and timer.deadline == 120.
    now[0] = 120.
    assert timer.poll(qualified=True)
    assert timer.close_requested and a._qrt_pending
    assert 'unconfirmed' in client.session.failure
    if phase == _WAIT_FS:
        assert len(client.session._proposed) == 1
        assert not client.session.sent_mids


@pytest.mark.parametrize('phase', [_GREETING, _ANSWER_HS, _RECEIVING])
def test_excluded_phases_gain_no_application_timer(phase):
    _, _, client, timer, now = mail(phase)
    for seconds in (62., 120., 1000.):
        now[0] = seconds
        assert not timer.poll(qualified=True)
        assert timer.deadline is None and not client.session.failure


@pytest.mark.parametrize('phase', [_WAIT_FS, _THEIR_TURN])
def test_slow_local_transport_over_120_seconds_does_not_arm_early(phase):
    a, _, _, timer, now = mail(phase)
    now[0] = 180.
    assert not timer.poll(qualified=True) and timer.deadline is None
    acknowledge_all(a)
    assert not timer.poll(qualified=True) and timer.deadline == 300.
    # A duplicate ACK has no packet credit and cannot create another interval.
    a._on_ack()
    now[0] = 242.
    assert not timer.poll(qualified=True) and timer.deadline == 300.


def test_listing_fragment_resets_but_polls_and_empty_source_do_not():
    a, _, client, timer, now = mail()
    acknowledge_all(a)
    timer.poll(qualified=True)
    now[0] = 110.
    client.on_link_data(b'contacting ')
    assert timer.deadline == 230.
    now[0] = 120.
    client.on_link_data(b'')
    timer.poll(qualified=True)
    assert timer.deadline == 230.
    client.on_link_data(b'CMS\r')
    assert timer.deadline == 240.


def test_valid_fs_exits_wait_and_body_acks_gate_next_timer():
    a, _, client, timer, now = mail()
    acknowledge_all(a)
    timer.poll(qualified=True)
    now[0] = 110.
    client.on_link_data(b'FS ')
    assert not client.session.sent_mids
    client.on_link_data(b'Y\r')
    assert client.session.sent_mids == ['PROGRESS120']
    assert timer.phase == _THEIR_TURN and timer.deadline is None
    now[0] = 300.
    assert not timer.poll(qualified=True) and timer.deadline is None
    acknowledge_all(a)
    assert not timer.poll(qualified=True) and timer.deadline == 420.


def test_invalid_fs_does_not_authorize_body_or_extend_wait():
    a, _, client, timer, now = mail()
    acknowledge_all(a)
    timer.poll(qualified=True)
    now[0] = 110.
    client.on_link_data(b'FS invalid\r')
    assert not client.session.sent_mids and client.session.failure
    assert timer.deadline is None


def test_application_expiry_during_absence_waits_for_return_or_lifetime_no_key_abort():
    for returns in (False, True):
        a, io, client, timer, now = mail()
        acknowledge_all(a)
        timer.poll(qualified=True)
        for _ in range(5):
            opportunity(a)
        now[0] = 120.
        assert timer.poll(qualified=False)
        assert not timer.close_requested and not a._qrt_pending
        if returns:
            assert opportunity(a, source='current CRC', tick=False)
            timer.poll(qualified=True)
            assert timer.close_requested and a._qrt_pending
        else:
            count = len(io.events)
            for _ in range(90):
                assert opportunity(a)
            assert not opportunity(a)
            timer.poll(qualified=False)
            assert len(io.events) == count and not a._qrt_pending
        assert not client.session.sent_mids


def test_progress_expiry_wrapper_applies_terminal_effects_once():
    a, _, client, timer, now = mail(_THEIR_TURN, with_message=False)
    acknowledge_all(a)
    assert not timer.poll(qualified=True) and timer.deadline == 120.
    stage = client.session.stage
    for name in ('_cancel_p3_proposals', '_enter_receive_only',
                 'retain_unconfirmed', 'on_host_disconnect'):
        setattr(a, name, Mock(wraps=getattr(a, name)))
    now[0] = 120.
    assert timer.poll(qualified=False)
    assert client.session.failure == (
        f'application progress timeout while {stage}; transfer unconfirmed')
    assert a.p3_receive_only and not timer.close_requested
    assert timer.poll(qualified=False)
    assert a._cancel_p3_proposals.call_count == 2  # Wrapper and receive-only transition.
    a._enter_receive_only.assert_called_once_with()
    a.retain_unconfirmed.assert_called_once_with('application progress timeout')
    a.on_host_disconnect.assert_not_called()
    assert timer.poll(qualified=True) and timer.close_requested
    assert timer.poll(qualified=True)
    assert a._cancel_p3_proposals.call_count == 3  # One more for peer-return goodbye.
    a._enter_receive_only.assert_called_once_with()
    a.retain_unconfirmed.assert_called_once_with('application progress timeout')
    a.on_host_disconnect.assert_called_once_with(local=False)
    assert len([line for line in client.session.log_lines
                if line.startswith('session failed')]) == 1


def test_configurable_longer_interval_and_new_link_generation():
    a, _, _, timer, now = mail(seconds=180.)
    acknowledge_all(a)
    timer.poll(qualified=True)
    now[0] = 120.
    assert not timer.poll(qualified=True)
    timer.link_down()
    a._enter_connected()
    timer.update()
    assert timer.deadline is None and timer.generation == a._link_generation


@pytest.mark.parametrize('seconds', [0., -1., float('nan'), float('inf')])
def test_invalid_progress_interval_is_refused(seconds):
    with pytest.raises(ValueError):
        mail(seconds=seconds)


def rendered(sl=1, *, long=False, swapped=False):
    audio = placement.link_packet(sl, b'test', 0x21, long_cycle=long, swapped=swapped)
    pad = 10 * rxfront.SPS
    row = pad + (placement.protocol_config().pulse().size-1)//2 + p3frame.DATA_OFFSET*rxfront.SPS
    return np.pad(audio, (pad, 2*rxfront.SPS)), row


@pytest.mark.parametrize('sl,long,swapped', [(1,False,False), (1,True,True),
    (2,False,True), (2,True,False), (3,False,False), (3,True,True),
    (6,False,True), (6,True,False)])
def test_existing_tracked_forms_qualify_without_packet_or_delivery(sl, long, swapped):
    audio, row = rendered(sl, long=long, swapped=swapped)
    # Absolute amplitude cannot exclude a coherent weak peer.
    source = tracked_structure(audio * .0001, 0, row, (sl,))
    assert source == f'tracked SL{sl} ' + ('long header' if long else 'short header')


def test_energy_wrong_form_phase_and_missing_coverage_do_not_qualify():
    audio, row = rendered(1)
    assert tracked_structure(audio, 0, row, (1,))
    assert tracked_structure(audio, 0, row, (2,)) is None
    assert tracked_structure(audio, 0, row+4*rxfront.SPS, (1,)) is None
    assert tracked_structure(audio[:row+1000], 0, row, (1,)) is None
    assert tracked_structure(audio, 0, row, (1,), gaps=((row+500, row+600),)) is None
    n = np.arange(len(audio))
    for unsupported in (np.zeros_like(audio), np.random.default_rng(1053).normal(0, .1, len(audio)),
                        .1*np.cos(2*np.pi*1500*n/FS)):
        assert tracked_structure(unsupported, 0, row, (1,)) is None


def test_coherent_fixed_changeover_is_qualified_at_its_tracked_phase():
    from hfmodem.shrike import onair
    audio = onair._trim_silence(placement.historical_control_signal(placement.BREAKIN_CS))
    pad = round(.15*FS)
    audio = np.pad(audio, (pad, round(.2*FS)))
    from hfmodem.shrike import p3acquire
    got = p3acquire.changeover(audio, offsets=(0.,))
    assert got is not None
    row = got.event.start + p3frame.DATA_OFFSET*rxfront.SPS
    assert tracked_structure(audio, 0, row, (1,), changeover=True) == 'tracked changeover'
    assert tracked_structure(audio, 0, row+4*rxfront.SPS, (1,), changeover=True) is None
    assert tracked_structure(audio, 0, row, (1,), changeover=True,
                             gaps=((got.event.start+100, got.event.start+200),)) is None


def test_crc_free_supported_structure_retains_independent_96_tick_ceiling():
    a, _ = modem()
    for _ in range(arq.P3_CRC_WAIT_TICKS-1):
        assert opportunity(a, source='tracked SL1 short header')
        assert not a._qrt_pending
    opportunity(a, source='tracked SL1 short header')
    assert a._p3_crc_wait_ticks == arq.P3_CRC_WAIT_TICKS and a._qrt_pending


def test_expiry_retires_unconfirmed_wire_before_later_close_can_send():
    a, _, _, timer, now = mail()
    acknowledge_all(a)
    timer.poll(qualified=True)
    a.on_host_data(b'retained only')
    a.role = arq.ISS
    a._start_next_packet()
    field, suffix = a._inflight.payload, bytes(a._outbuf)
    now[0] = 120.
    timer.poll(qualified=False)
    assert a.transfer_unconfirmed['field'] == field
    assert a.transfer_unconfirmed['suffix'] == suffix
    assert not a._outbuf and a._inflight is None
    timer.poll(qualified=True)
    _, chunk, _, _, _ = a._next_packet(False)
    assert chunk == b'' and a._qrt_pending


def test_external_hostmode_has_presence_bound_but_no_application_timer():
    from hfmodem.tests.shrike.test_entry_answer import _Session
    session = _Session(role=arq.IRS)
    host = session.host
    assert host.app is None
    host.arq.on_host_data(b'external client owns its timer')
    for _ in range(5):
        token = host.arq.begin_receive_opportunity()
        host.arq.qualify_peer(token, host.arq._link_generation, None)
        host.tick()
        host.arq.finish_receive_opportunity(token)
    count = len(host.peer.sent)
    token = host.arq.begin_receive_opportunity()
    host.arq.qualify_peer(token, host.arq._link_generation, None)
    host.tick()
    host.arq.finish_receive_opportunity(token)
    assert len(host.peer.sent) == count
    assert host.arq.state == arq.State.CONNECTED and host.arq.p3_receive_only
    for _ in range(90):
        token = host.arq.begin_receive_opportunity()
        host.arq.qualify_peer(token, host.arq._link_generation, None)
        host.tick()
        host.arq.finish_receive_opportunity(token)
    assert len(host.peer.sent) == count and host.arq.state == arq.State.LISTENING
    assert host.arq.transfer_unconfirmed['suffix'] == b'external client owns its timer'
    assert any('transfer unconfirmed' in line for line in host.channel(host.ptchn).link)


def test_cli_rejects_invalid_progress_interval_before_opening_io():
    from types import SimpleNamespace
    from hfmodem.shrike.onair import _p3_mail_defaults
    for seconds in (0., -1., float('nan'), float('inf')):
        with pytest.raises(SystemExit):
            _p3_mail_defaults(SimpleNamespace(p3_mail=False, p3_mail_progress_seconds=seconds))


@pytest.fixture(scope='module')
def recorded_1053():
    import hashlib, json
    from pathlib import Path
    from hfmodem.tests import evidence
    from hfmodem.core.wav import read
    manifest = json.loads(Path(__file__).with_name('fixtures').joinpath(
        'p3_presence_1053_1003.json').read_text())
    path = evidence.require_input(evidence.CAPTURES / manifest['recording'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']
    return manifest, read(path)


def presence_session(row, *, cfo=0.):
    from hfmodem.tests.shrike.test_entry_answer import _Session
    session = _Session(role=arq.IRS)
    rx = session.rx
    rx._p3_row0 = row
    rx._p3_cycle_n = 60000
    rx.sync.packet_level = 1
    rx._p3_presence_form = (session.host.arq._link_generation, 1)
    rx._p3_repeat_changeover = False
    rx.p3_receive_offset_hz = cfo
    return session


def test_1053_actual_tail_production_attribution_stops_before_sixth_key(recorded_1053):
    from types import SimpleNamespace
    from hfmodem.shrike import onair
    manifest, audio = recorded_1053
    last = manifest['repeat_rows'][-1]
    assert len(manifest['tail']) == 32
    session = presence_session(last, cfo=manifest['cfo_hz'])
    host, rx = session.host, session.rx
    host.arq.on_host_data(b'retain this pending wire')
    keys = 0
    for index, window in enumerate(manifest['tail']):
        # Each original unnormalised full-stream slice is independently
        # attributed through the production helper. No occupancy truth flags.
        a = rx.host.arq
        token = a.begin_receive_opportunity()
        a._receive_opportunity.capture_lost = window['lost']
        tx = SimpleNamespace(slot=window['slot'], live=SimpleNamespace(lost=window['lost']))
        raster = SimpleNamespace(boundary=lambda slot: window['reply'])
        onair._p3_qualify_presence(rx, raster, tx, audio[window['origin']:window['end']], window['origin'])
        assert a._receive_opportunity.presence_checked and not a.peer_qualified
        before = len(host.peer.sent)
        host.tick()
        keys += len(host.peer.sent)-before
        a.finish_receive_opportunity(token)
        assert a.state == arq.State.CONNECTED
        assert a.p3_receive_only == (index >= 5)
        if index >= 5:
            assert len(host.peer.sent) == before
        assert bytes(a._outbuf) == b'retain this pending wire'
    assert keys == 5 and a._p3_crc_wait_ticks == 32
    # The tape supplies32 absent windows; the remaining64 are a labeled model.
    for index in range(64):
        token = a.begin_receive_opportunity()
        a.qualify_peer(token, a._link_generation, None)
        host.tick()
        a.finish_receive_opportunity(token)
        assert len(host.peer.sent) == keys
        assert (a.state == arq.State.CONNECTED) == (index < 63)
    assert host.arq.transfer_unconfirmed['suffix'] == b'retain this pending wire'


def test_previous_complete_packet_cannot_qualify_current_truncated_opportunity():
    from types import SimpleNamespace
    from hfmodem.shrike import onair
    audio, row = rendered()
    session = presence_session(row)
    token = session.host.arq.begin_receive_opportunity()
    tx = SimpleNamespace(slot=2)
    raster = SimpleNamespace(boundary=lambda slot: row+60000+36000)
    onair._p3_qualify_presence(session.rx, raster, tx, audio, 0)
    assert not session.host.arq.peer_qualified
    session.host.arq.finish_receive_opportunity(token)


def test_1053_repeated_crc_fragment_is_delivered_once_without_progress_reset(recorded_1053):
    from hfmodem.shrike import p3acquire
    manifest, audio = recorded_1053
    rows = manifest['repeat_rows']
    assert len(rows) == 35 and (rows[-1]-rows[0])/FS == 43.7525
    a, io, client, timer, now = mail()
    acknowledge_all(a)
    timer.poll(qualified=True)
    a._breakin_pending = False
    original = io.deliver
    def deliver(blob):
        original(blob)
        client.on_link_data(blob)
    io.deliver = deliver
    decoded = 0
    for row in rows:
        lo = row-7200
        corrected = p3acquire.compensate(audio[lo:row+38000], manifest['cfo_hz'])
        sync = rxfront.SyncedRx()
        event = None
        for delta in (0, -120, 120):
            event = sync.sl1_packet_at(corrected, row-lo+delta)
            if event is not None and event.packet is not None:
                break
        assert event is not None and event.packet is not None
        sl, status, payload, crc = event.packet
        assert crc and status == manifest['repeat_status'] and payload == b'TOR t'
        decoded += 1
        now[0] = (row-rows[0])/FS
        token = a.begin_receive_opportunity()
        a.on_rx_packet(sl, payload, status, crc)
        a.qualify_peer(token, a._link_generation, 'current CRC')
        a.finish_receive_opportunity(token)
        assert not timer.poll(qualified=True) and timer.deadline == 120.
    assert decoded == 35 and io.received == [b'TOR t']
    assert a._dark_windows == 0 and not client.session.sent_mids


@pytest.mark.parametrize('lost,gap', [(False,False), (True,False), (False,True)])
def test_current_weak_structure_requires_actual_input_coverage(lost, gap):
    from types import SimpleNamespace
    from hfmodem.shrike import onair, p3acquire
    audio, row = rendered(2)
    # The existing tracked carrier remains authoritative, including a large
    # correction like the successful 1047 SL2 control. No new CFO search.
    audio = p3acquire.compensate(audio, 101.3) * .0001
    session = presence_session(row, cfo=-101.3)
    session.rx.sync.packet_level = 2
    session.rx._p3_presence_form = (session.host.arq._link_generation, 2)
    a = session.host.arq
    token = a.begin_receive_opportunity()
    a._receive_opportunity.capture_lost = 0
    tx = SimpleNamespace(slot=209, live=SimpleNamespace(lost=int(lost)))
    if gap:
        tx.tx_audio_start, tx.tx_end = row-1000, row+1000
    raster = SimpleNamespace(boundary=lambda slot: row+37000)
    onair._p3_qualify_presence(session.rx, raster, tx, audio, 0)
    assert a.peer_qualified == (not lost and not gap)
    assert not a._receive_opportunity.crc
    a.finish_receive_opportunity(token)


def test_live_mail_tick_rechecks_presence_before_progress_or_response():
    from hfmodem.tests.shrike.test_entry_answer import _Session
    session = _Session(role=arq.IRS)
    host, a = session.host, session.host.arq
    _, _, client, _, now = mail()
    host.app = client
    client.send = a.on_host_data
    client.progress = MailProgress(client, a, now=lambda: now[0])
    client.progress.update()
    for _ in range(5):
        token = a.begin_receive_opportunity()
        a.qualify_peer(token, a._link_generation, None)
        host.tick()
        a.finish_receive_opportunity(token)
    now[0] = 120.
    before = len(host.peer.sent)
    token = a.begin_receive_opportunity()
    a.qualify_peer(token, a._link_generation, None)
    host.tick()
    a.finish_receive_opportunity(token)
    assert len(host.peer.sent) == before and not a._qrt_pending
    assert client.session.failure and 'unconfirmed' in client.session.failure


def test_accepted_form_survives_decoder_unlock_but_not_link_generation():
    from types import SimpleNamespace
    from hfmodem.shrike import onair
    audio, row = rendered(2)
    s = presence_session(row)
    a = s.host.arq
    s.rx._p3_presence_form = (a._link_generation, 2)
    s.rx.sync.packet_level = None
    tx, raster = SimpleNamespace(slot=1), SimpleNamespace(boundary=lambda _:row+37000)
    for current in (True, False):
        token = a.begin_receive_opportunity()
        if not current:
            a._link_generation += 1
        onair._p3_qualify_presence(s.rx, raster, tx, audio, 0)
        assert a.peer_qualified == current
        a.finish_receive_opportunity(token)


def test_presence_obeys_existing_admission_and_long_transform_reserve():
    from types import SimpleNamespace
    from hfmodem.shrike import onair
    for long in (False, True):
        audio, row = rendered(1, long=long)
        s = presence_session(row)
        s.rx._p3_cycle_n = 180000 if long else 60000
        observed = []
        s.rx._p3_acquisition_fits = lambda reserve: observed.append(reserve) or False
        token = s.host.arq.begin_receive_opportunity()
        tx = SimpleNamespace(slot=1)
        raster = SimpleNamespace(boundary=lambda _:row+(156000 if long else 37000))
        onair._p3_qualify_presence(s.rx,raster,tx,audio,0)
        assert not s.host.arq.peer_qualified
        assert observed == [s.rx.P3_SL1_RESERVE_S + (s.rx.P3_WIDEBAND_RESERVE_S if long else 0)]
        s.host.arq.finish_receive_opportunity(token)
