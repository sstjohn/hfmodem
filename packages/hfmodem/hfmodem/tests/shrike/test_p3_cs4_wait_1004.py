"""Approved 96-tick CS4 opportunity: synthetic models, not post-recording RF.

Reuse the existing sender's wire fixture. No new ACK eligibility rule is
modelled here: PtcHost maps parity, and ARQ requires current P3 and sent data.
"""
import pytest

from hfmodem.shrike import arq, rxfront, spec
from hfmodem.tests.shrike.test_tx_gear_hold import control, sender


def waiting(*, bound=None, refused=False):
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    if bound is not None:
        host.arq.cfg.p3_mail_cs4_wait_ticks = bound
    seam.refuse = refused
    control(host, 4)
    packet = host.arq._inflight
    assert host.arq._mail_cs4_candidate is packet
    assert host.arq._mail_cs4_wait is None
    control(host, 4)
    assert host.arq._mail_cs4_wait == 0
    return host, seam, packet


def held(host, seam):
    a = host.arq
    return (a._inflight, a._inflight.payload, bytes(a._outbuf), a.tx_seq,
            a._buffer_raw, a.tx_acked_bytes, host.sent_total, tuple(seam.frames))


def assert_held(host, seam, before):
    assert host.arq.state == arq.State.CONNECTED
    assert held(host, seam) == before
    assert not host.arq._qrt_pending


def assert_retained(host, seam, before):
    a = host.arq
    assert a.state == arq.State.DISCONNECTED
    retained = a.transfer_unconfirmed
    assert retained['field'] == before[1]
    assert retained['suffix'] == before[2]
    assert retained['counter'] == before[3]
    assert a.tx_acked_bytes == before[5] and host.sent_total == before[6]
    assert tuple(seam.frames) == before[7]
    assert a.unconfirmed_transfers[-1] is retained
    assert len(a.unconfirmed_transfers) == 1
    assert a._mail_cs4_wait is a._mail_cs4_candidate is None
    return retained


def test_default_and_explicit_old_bound():
    assert arq.ArqConfig().p3_mail_cs4_wait_ticks == 96
    assert arq.ArqConfig(p3_mail_cs4_wait_ticks=8).p3_mail_cs4_wait_ticks == 8


@pytest.mark.parametrize('value', [0, -1, True, False, 1.5, '96', None])
def test_bound_validation_unchanged(value):
    with pytest.raises(ValueError, match='p3_mail_cs4_wait_ticks'):
        arq.ArqConfig(p3_mail_cs4_wait_ticks=value)


def test_default_silent_wait_survives_old_eight_and_95_then_retains_once():
    host, seam, _ = waiting()
    before = held(host, seam)
    for elapsed, total in ((8, 8), (87, 95)):
        host.tick(elapsed_ticks=elapsed)
        assert host.arq._mail_cs4_wait == total
        assert_held(host, seam, before)
    host.tick()
    retained = assert_retained(host, seam, before)
    for cs in (1, 2, 3, 4, 5, 6):
        control(host, cs)
        host.tick()
        assert host.arq.transfer_unconfirmed is retained
        assert len(host.arq.unconfirmed_transfers) == 1
        assert tuple(seam.frames) == before[7]


def test_explicit_eight_tick_override_retains_original_boundary():
    host, seam, _ = waiting(bound=8)
    before = held(host, seam)
    host.tick(elapsed_ticks=7)
    assert_held(host, seam, before)
    host.tick()
    assert_retained(host, seam, before)


@pytest.mark.parametrize('cs', [1, 3, 4, 5, 6])
def test_nonvalidating_controls_never_restart_first_origin(cs):
    host, seam, _ = waiting()
    before = held(host, seam)
    for total in range(1, 96):
        control(host, cs)
        host.arq.note_peer_carrier()
        host.tick()
        assert host.arq._mail_cs4_wait == total
        assert_held(host, seam, before)
    control(host, cs)
    host.tick()
    assert_retained(host, seam, before)


def test_skipped_and_long_slots_charge_elapsed_once_per_opportunity():
    host, seam, _ = waiting()
    before = held(host, seam)
    for elapsed, total in ((8, 8), (3, 11), (80, 91), (4, 95)):
        token = host.arq.begin_receive_opportunity()
        host.tick(elapsed_ticks=elapsed, cycle_ticks=3)
        assert host.arq._receive_opportunity.elapsed_ticks == elapsed
        host.arq.finish_receive_opportunity(token)
        assert host.arq._mail_cs4_wait == total
        assert_held(host, seam, before)
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    assert_retained(host, seam, before)


@pytest.mark.parametrize('seq', [1, 2])
def test_late_parity_ack_uses_actual_host_mapping_and_credits_once(seq):
    host, seam, packet = waiting()
    # Exercise both parities on the same existing sent-field fixture.
    packet.status = (packet.status & ~3) | seq
    host.arq._next_seq = (seq + 1) % 4
    before = held(host, seam)
    host.tick(elapsed_ticks=9)
    control(host, 2 if seq % 2 == 0 else 1)
    assert_held(host, seam, before)
    control(host, 1 if seq % 2 == 0 else 2)
    a = host.arq
    assert a._mail_cs4_wait is a._mail_cs4_candidate is None
    assert a.tx_acked_bytes == before[5] + len(packet.payload)
    assert host.sent_total == before[6] + len(packet.payload)
    assert a._buffer_raw == before[4] - len(packet.payload)
    assert a.tx_seq == (seq + 1) % 4
    confirmed = a.tx_acked_bytes
    control(host, 1 if seq % 2 == 0 else 2)
    assert a.tx_acked_bytes == confirmed  # duplicate now has wrong parity


@pytest.mark.parametrize('protocol', [spec.Protocol.PACTOR1, spec.Protocol.PACTOR2])
def test_lower_protocol_event_does_not_release_during_added_interval(protocol):
    host, seam, _ = waiting()
    host.tick(elapsed_ticks=9)
    before = held(host, seam)
    host.on_rx_event(rxfront.Event(0, 'cs', '', protocol=protocol, cs=1))
    assert_held(host, seam, before)
    assert host.arq._mail_cs4_wait == 9


def test_crc_changeover_does_not_release_or_change_turn_during_added_interval():
    host, seam, packet = waiting()
    host.tick(elapsed_ticks=9)
    before = held(host, seam)
    host.on_rx_event(rxfront.Event(0, 'packet', '', protocol=spec.Protocol.PACTOR3,
                                 packet=(1, 0, packet.payload, True), breakin=True))
    assert host.arq.role == arq.ISS
    assert_held(host, seam, before)
    assert host.arq._mail_cs4_wait == 9


def test_unkeyed_field_is_not_validated_or_reissued_after_old_bound():
    host, seam, packet = waiting(refused=True)
    assert packet.sent_at_level == 0
    before = held(host, seam)
    host.tick(elapsed_ticks=9)
    seam.refuse = False
    control(host, 2)
    assert host.arq._mail_cs4_wait == 9
    assert_held(host, seam, before)
    host.tick(elapsed_ticks=87)
    assert_retained(host, seam, before)


def test_current_protocol_is_required_by_arq_even_for_logical_ack():
    host, seam, _ = waiting()
    before = held(host, seam)
    host.tick(elapsed_ticks=9)
    host.protocol = spec.Protocol.PACTOR2
    host.arq.on_rx_cs(arq.CS_ACK)
    assert_held(host, seam, before)


def test_reader_before_tick_ack_at_95_releases_before_expiry():
    host, seam, packet = waiting()
    host.tick(elapsed_ticks=95)
    acknowledged = host.arq.tx_acked_bytes
    control(host, 2)
    host.tick()
    assert host.arq.state == arq.State.CONNECTED
    assert host.arq._mail_cs4_wait is None
    assert host.arq.tx_acked_bytes == acknowledged + len(packet.payload)
    assert host.arq.transfer_unconfirmed is None


@pytest.mark.parametrize('method', ['on_host_disconnect', 'on_host_abort'])
def test_explicit_end_after_old_bound_retains_without_credit(method):
    host, seam, _ = waiting()
    host.tick(elapsed_ticks=9)
    before = held(host, seam)
    getattr(host.arq, method)()
    retained = assert_retained(host, seam, before)
    control(host, 2)
    assert host.arq.transfer_unconfirmed is retained
    assert tuple(seam.frames) == before[7]


def test_new_contact_clears_wait_and_does_not_replay_retained_history():
    host, seam, _ = waiting()
    host.tick(elapsed_ticks=96)
    retained = host.arq.transfer_unconfirmed
    frames = tuple(seam.frames)
    host.arq._enter_connected()
    assert host.arq._mail_cs4_wait is host.arq._mail_cs4_candidate is None
    assert host.arq.transfer_unconfirmed is None
    assert host.arq.unconfirmed_transfers[-1] is retained
    assert host.arq._inflight is None and not host.arq._outbuf
    assert tuple(seam.frames) == frames


def test_queue_growth_preserves_first_origin_and_retains_new_suffix_once():
    host, seam, packet = waiting()
    host.tick(elapsed_ticks=9)
    before = held(host, seam)
    # Reuse already queued fixture bytes; no new wire fixture is introduced.
    extra = bytes(host.arq._outbuf)
    host.arq.on_host_data(extra)
    assert host.arq._mail_cs4_wait == 9
    assert host.arq._inflight is packet
    assert bytes(host.arq._outbuf) == before[2] + extra
    assert host.arq.tx_seq == before[3]
    assert host.arq.tx_acked_bytes == before[5]
    assert tuple(seam.frames) == before[7]
    grown = held(host, seam)
    host.tick(elapsed_ticks=87)
    assert_retained(host, seam, grown)


def test_mail_watermark_is_unarmed_until_real_transport_confirmation():
    from hfmodem.shrike.p3progress import MailProgress
    from hfmodem.winlink import B2FSession, MailClient
    from hfmodem.winlink.session import _THEIR_TURN

    host, seam, packet = waiting()
    a = host.arq
    # sender seeds buffers directly. Restore the complete queued-byte ledger,
    # including the already credited setup field, before attaching the app.
    a.tx_queued_bytes = a.tx_acked_bytes + len(packet.payload) + len(a._outbuf)
    client = MailClient(B2FSession(host.mycall), a.on_host_data)
    client.session._state = _THEIR_TURN
    now = [0.0]
    timer = client.progress = MailProgress(client, a, now=lambda: now[0])
    host.app = client
    timer.update()
    watermark = timer.watermark
    before = held(host, seam)
    for elapsed, seconds in ((8, 10.0), (87, 118.75)):
        now[0] = seconds
        host.tick(elapsed_ticks=elapsed)
        assert timer.deadline is None and timer.watermark == watermark
        assert not client.session.failure
        assert_held(host, seam, before)
    control(host, 2)
    assert a._mail_cs4_wait is None
    # This ACK covers one field, not the still queued application suffix.
    assert not timer.poll(qualified=True) and timer.deadline is None
    for _ in range(16):
        if a.tx_acked_bytes >= watermark:
            break
        if a._inflight is None:
            host.tick()
        assert a._inflight is not None and a._inflight.sent_at_level > 0
        control(host, 1 if a.tx_seq % 2 == 0 else 2)
    assert a.tx_acked_bytes == watermark
    assert not timer.poll(qualified=True)
    assert timer.deadline == now[0] + 120.0
    now[0] = timer.deadline
    assert timer.poll(qualified=False)
    failure = client.session.failure
    assert failure and a._p3_app_failed and timer.expired
    frames = tuple(seam.frames)
    for cs in (1, 2, 4, 6):
        control(host, cs)
        client.on_link_data(packet.payload)
    assert client.session.failure == failure
    assert timer.expired and a._p3_app_failed
    assert tuple(seam.frames) == frames


@pytest.mark.parametrize('unbounded', [False, True], ids=['finite-hold', 'unbounded-hold'])
def test_actual_main_hold_loop_respects_outer_and_cs4_bounds(monkeypatch, tmp_path,
                                                          unbounded):
    """Synthetic dry-run main, with an admitted-event seam and no RF decoder.

    Retain the real hold budget, receive opportunities, ticks and final TX
    admission. The receive seam seeds an existing field fixture once; this is
    neither a waveform-admission test nor reconstruction of a recorded arm.
    """
    import contextlib
    import copy
    import io
    import sys
    from hfmodem.shrike import onair

    observed = {}
    original_defaults = onair._timing_trial_defaults

    def synthetic_driver_configuration(args):
        original_defaults(args)
        # Exercise the actual driver's two budget modes without claiming a
        # second CLI path for the existing unbounded experiment setting.
        args.p3_timing_unbounded = unbounded

    class Receiver(onair._SessionRx):
        def feed(self, chunk):
            if observed:
                return
            source, _ = sender(enabled=False)
            host, a = self.host, self.host.arq
            a._enter_connected()
            a.role = arq.ISS
            host.protocol = spec.Protocol.PACTOR3
            host.p3_mail_active = True
            host.p3_tx_gear_hold = False
            a.cfg.long_cycle = False
            a._inflight = copy.copy(source.arq._inflight)
            a._next_seq = source.arq._next_seq
            a._outbuf = bytearray(source.arq._outbuf)
            a._buffer_raw = source.arq._buffer_raw
            control(host, 4)
            control(host, 4)
            assert a._mail_cs4_wait == 0
            assert a._inflight.sent_at_level > 0
            observed.update(host=host, field=a._inflight.payload,
                            suffix=bytes(a._outbuf), counter=a.tx_seq,
                            credited=a.tx_acked_bytes, tx_count=host.peer.n,
                            ticks=[], opportunities=[])
            original_tick = host.tick

            def tick(**kwargs):
                observed['ticks'].append(kwargs.get('elapsed_ticks', 1))
                observed['opportunities'].append(a._receive_opportunity is not None)
                assert len(observed['ticks']) <= 100
                return original_tick(**kwargs)

            host.tick = tick

    monkeypatch.setattr(onair, '_timing_trial_defaults', synthetic_driver_configuration)
    monkeypatch.setattr(onair, '_SessionRx', Receiver)
    monkeypatch.setattr(onair.time, 'sleep', lambda _: None)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    monkeypatch.setattr(sys, 'argv', ['shrike.onair', '--hold', '12',
                                    '--max-cycles', '1', '--listen', '0',
                                    '--mycall', 'W9SSJ', '--dxcall', 'K7ABC',
                                    '--outdir', str(tmp_path / 'capture')])
    # No --transmit, --replay, hardware objects, tape or fake post-tape input.
    with contextlib.redirect_stdout(io.StringIO()):
        result = onair.main()
    host, a = observed['host'], observed['host'].arq
    assert result == 1 and a.state == arq.State.DISCONNECTED
    assert a.transfer_unconfirmed['field'] == observed['field']
    assert a.transfer_unconfirmed['suffix'] == observed['suffix']
    assert a.transfer_unconfirmed['counter'] == observed['counter']
    assert len(a.unconfirmed_transfers) == 1
    assert a.tx_acked_bytes == observed['credited']
    assert host.peer.n == observed['tx_count']
    # Setup owns its existing plain tick; all subsequent hold ticks are
    # enclosed in actual driver receive opportunities.
    assert observed['opportunities'][0] is False
    assert all(observed['opportunities'][1:])
    elapsed = sum(observed['ticks'])
    assert elapsed == 96 if unbounded else 8 < elapsed < 96


def test_recorded_stop_metadata_and_first_old_terminal_divergence():
    import json
    from pathlib import Path

    meta = json.loads(Path(__file__).with_name('fixtures').joinpath(
        'p3_cs4_wait_edf509bb_1004.json').read_text())
    assert meta['recording_log_sha256'] == (
        '60609f29967ae7edb3b0889bd7f2cc1eef2eb71ab1750e1051e4aa10d8aaf814')
    assert (meta['old_ticks'], meta['field_bytes'], meta['suffix_bytes']) == (8, 23, 79)
    assert meta['outer_hold_unbounded'] and meta['mail_deadline_unarmed']
    assert meta['scope'] == 'recorded metadata only; model stops at first divergence'
    old, old_seam, _ = waiting(bound=meta['old_ticks'])
    new, new_seam, _ = waiting()
    old_before, new_before = held(old, old_seam), held(new, new_seam)
    for _ in range(meta['old_ticks'] - 1):
        old.tick()
        new.tick()
        assert_held(old, old_seam, old_before)
        assert_held(new, new_seam, new_before)
    old.tick()
    new.tick()
    assert_retained(old, old_seam, old_before)
    assert_held(new, new_seam, new_before)
    # No comparison beyond this first old-terminal divergence and no claim
    # that the synthetic fixture's bytes or later silence were on the tape.
