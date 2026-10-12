# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Close-only P1 exception after a current owned grant, never a mail fallback."""
import pytest
from hfmodem.shrike import arq, pactor1, rxfront, spec
from hfmodem.tests.shrike.test_no_p3_fallback import entered, grant, cs, acknowledge
from hfmodem.tests.shrike.test_p3_entry_window_1003 import (
    ARMS, recording, grid, collected_end)
from hfmodem.shrike import onair


def current():
    host, tx, mark = entered(payload=b'unconfirmed application bytes')
    grant(host)
    host.tick()
    assert host.arq._entry_p1_current and not host.arq._p3_entry_confirmed
    return host, tx, mark


def retained(host, expected):
    r = host.arq.transfer_unconfirmed
    assert r is not None and r['field'] + r['suffix'] == expected
    assert len(host.arq.unconfirmed_transfers) == 1
    assert 'transfer unconfirmed' in host.channel(host.ptchn).link[-1]
    assert 'LINK FAILURE' in host.channel(host.ptchn).link[-1]
    assert 'not queued for retry' in host.channel(host.ptchn).link[-1]


def test_current_grant_closes_only_and_retains_exact_wire_bytes():
    host, tx, mark = current()
    a = host.arq
    wire = a._inflight.payload + bytes(a._outbuf)
    sent = host.sent_total
    host.arq.on_host_disconnect()
    assert host.protocol is spec.Protocol.PACTOR1 and a.terminal_pending
    assert not a.entry_pending and a._disconnect_ticks == 0
    assert a.transfer_unconfirmed['field'] + a.transfer_unconfirmed['suffix'] == wire
    assert not a._outbuf and a._inflight is None
    host.tick()
    assert a.said_goodbye and a.state is arq.State.DISCONNECTING
    assert a.tx_seq == a.transfer_unconfirmed['counter']
    assert tx.emissions[-1][0] is spec.Protocol.PACTOR1
    assert tx.emissions[-1][2] == pactor1.qrt_field('WS8EOC', host.p1_baud)
    deadline = a._goodbye_cycles
    grant(host)
    a.on_host_disconnect()
    assert a._goodbye_cycles == deadline and host.protocol is spec.Protocol.PACTOR1
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
    assert a.state is arq.State.DISCONNECTED and not a.goodbye_acked
    assert host.sent_total == sent
    retained(host, wire)


@pytest.mark.parametrize('bad', ['initial-grant-only', 'quiet', 'skipped',
                                 'truncated', 'energy', 'wrong-grant',
                                 'p2-grant', 'unowned', 'higher-bad-crc', 'new-window'])
def test_ambiguous_or_stale_evidence_keys_no_p1_goodbye(bad):
    host, tx, _ = current()
    a = host.arq
    wire = a._inflight.payload + bytes(a._outbuf)
    if bad == 'initial-grant-only':
        host, tx, _ = entered(payload=b'unconfirmed application bytes')
        a = host.arq
        wire = a._inflight.payload + bytes(a._outbuf)
    elif bad == 'quiet':host.tick()
    elif bad == 'skipped':
        grant(host)
        host.tick(elapsed_ticks=2)
    elif bad == 'truncated':a.note_entry_window_truncated()
    elif bad == 'new-window':onair._SessionRx(host).new_cycle()
    elif bad == 'energy':host.on_rx_event(rxfront.Event(.1, 'detect', 'unread carrier'))
    elif bad in ('wrong-grant', 'p2-grant'):
        host.on_rx_event(rxfront.Event(.1, 'unassigned', 'other',
            protocol=spec.Protocol.PACTOR2 if bad == 'p2-grant' else spec.Protocol.PACTOR1,
            spare=pactor1.CS_59A if bad == 'p2-grant' else pactor1.CS_6A9))
    elif bad == 'unowned':
        a._upgrade_reply_owed = False
        grant(host)
    else:
        host.on_rx_event(rxfront.Event(.1, 'packet', 'bad higher candidate',
            protocol=spec.Protocol.PACTOR3, packet=(1, 0, b'', False)))
    n = len(tx.emissions)
    a.on_host_disconnect()
    host.tick()
    assert len(tx.emissions) == n and not a.said_goodbye
    assert a.state is arq.State.DISCONNECTED
    retained(host, wire)


@pytest.mark.parametrize('higher', ['ack', 'nak', 'speed', 'packet', 'earlier-contact-phase'])
def test_any_confirmed_p3_response_forbids_p1_close_for_this_contact(higher):
    host, tx, _ = current()
    a = host.arq
    if higher in ('ack', 'earlier-contact-phase'):
        acknowledge(host)
    elif higher == 'packet':
        host.on_rx_event(rxfront.Event(.1, 'packet', 'checked P3',
            protocol=spec.Protocol.PACTOR3, packet=(1, 0, b'', True)))
    else:host.on_rx_event(cs(arq.CS_NAK if higher == 'nak' else arq.CS_SPEED_UP))
    assert a._p3_entry_confirmed
    if higher == 'earlier-contact-phase':
        # A later unconfirmed attempt cannot erase this contact's P3 history.
        host.protocol = spec.Protocol.PACTOR3
        a.entry_pending = True
        a._unanswered_upgrade = 0
        a._upgrade_reply_owed = True
    grant(host)
    assert not host.prepare_p1_entry_goodbye()
    assert host.protocol is spec.Protocol.PACTOR3


@pytest.mark.parametrize('refusal', ['seam', 'elapsed-unplaceable'])
def test_refused_p1_close_keeps_existing_elapsed_placement_ceiling(refusal):
    host, tx, _ = current()
    a = host.arq
    wire = a._inflight.payload + bytes(a._outbuf)
    tx.send_p1_packet = lambda *args, **kwargs: arq.REFUSED
    a.on_host_disconnect()
    if refusal == 'seam':
        host.tick()
        assert not a.said_goodbye and a._disconnect_ticks == 1
        a.on_host_disconnect()
        assert a._disconnect_ticks == 1
        host.tick(elapsed_ticks=arq.GOODBYE_PLACE_TICKS - 1)
    else:host.tick(elapsed_ticks=arq.GOODBYE_PLACE_TICKS)
    assert a.state is arq.State.DISCONNECTED and a.goodbye_unplaceable
    assert not a.said_goodbye
    retained(host, wire)


def test_real_p1_close_ack_does_not_credit_retained_application_bytes():
    host, tx, _ = current()
    a = host.arq
    wire = a._inflight.payload + bytes(a._outbuf)
    sent = host.sent_total
    a.on_host_disconnect()
    host.tick()
    # A modeled ACK is only a close-state pin, never recorded peer acceptance.
    seq = a.tx_seq
    host.on_rx_event(cs(pactor1.CS_ACK_A if seq & 1 else pactor1.CS_ACK_B,
                        spec.Protocol.PACTOR1))
    assert a.state is arq.State.DISCONNECTED and a.goodbye_acked
    assert host.sent_total == sent
    retained(host, wire)


def test_retained_history_survives_but_current_close_evidence_resets_next_link():
    host, tx, _ = current()
    a = host.arq
    a.on_host_disconnect()
    host.tick(elapsed_ticks=arq.GOODBYE_PLACE_TICKS)
    history = a.unconfirmed_transfers[0]
    a.on_host_connect('W9SSJ', 'KB5LZK')
    host.on_rx_event(cs(pactor1.CS_SPEED, spec.Protocol.PACTOR1))
    host.tick()
    grant(host)
    host.tick()
    assert a.transfer_unconfirmed is None and a.unconfirmed_transfers == [history]
    assert not a._entry_p1_current and not a._p3_entry_confirmed
    assert not host.prepare_p1_entry_goodbye()


@pytest.mark.parametrize('arm,positive', [('1003-0004', True), ('1003-0007', True),
                                         ('1003-0008', False), ('1003-0010', False),
                                         ('1003-0012', False)])
def test_recorded_grants_confirmed_and_ambiguous_windows(arm, positive):
    audio = recording(arm)
    host, tx, _ = entered(payload=b'unconfirmed application bytes')
    rx = onair._SessionRx(host, tag='P1-CLOSE')
    for row in ARMS[arm]['rows']:
        rx.new_cycle()
        g = grid(arm, row)
        end = collected_end(rx, g, row)
        rx.control_signal_in(audio[row['origin']:end], row['origin'], g)
        host.tick()
    assert host.prepare_p1_entry_goodbye() is positive
    if positive:
        host.arq.on_host_disconnect()
        host.tick()
        assert host.arq.said_goodbye and tx.emissions[-1][0] is spec.Protocol.PACTOR1
        assert tx.emissions[-1][2] == pactor1.qrt_field('WS8EOC', host.p1_baud)
    else:
        assert host.protocol is spec.Protocol.PACTOR3
        assert host.arq._p3_entry_confirmed is (arm != '1003-0012')


def test_host_write_during_p1_terminal_close_is_retained_not_transmitted():
    host, tx, _ = current()
    a = host.arq
    wire = a._inflight.payload + bytes(a._outbuf)
    a.on_host_disconnect()
    a.on_host_data(b'later host bytes')
    host.tick()
    assert not a._outbuf and not a._inflight.payload
    assert tx.emissions[-1][2] == pactor1.qrt_field('WS8EOC', host.p1_baud)
    host.on_rx_event(cs(a.tx_seq & 1, spec.Protocol.PACTOR3))
    assert not a.goodbye_acked and a.state is arq.State.DISCONNECTING
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
    retained(host, wire + b'later host bytes')


def test_binary_writes_and_supervisor_during_close_keep_exact_wire_escape():
    from hfmodem.shrike import compress
    host, tx, _ = current()
    a = host.arq
    wire = a._inflight.payload + bytes(a._outbuf)
    a.on_host_disconnect()
    binary = bytes(range(256))
    block = compress.supervisor_block('Bopaque')
    a.on_host_data(binary)
    a.on_host_supervisor(block)
    host.tick()
    assert not a._outbuf and not a._inflight.payload
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
    retained(host, wire + compress.transparent(binary) + block)


def test_terminal_p1_waveform_uses_real_renderer_and_radio_admission(tmp_path, monkeypatch):
    import numpy as np
    from hfmodem.shrike import p1rx
    from hfmodem.tests.shrike.test_p3_breakin_timing import duplex
    host, _, _ = current()
    a = host.arq
    counter = a.tx_seq
    g = onair._MasterGrid(5 * onair.FS, 60000, 0, packet_n=46080,
                         cs_n=5760, d_max_n=6240)
    tx = duplex(g, tmp_path, seconds=8)
    tx.attach(host)
    host.peer = tx
    tx.aim(g, 0)
    waves = []
    transmit = tx.live.transmit
    def capture(audio, **kwargs):
        waves.append(audio.copy())
        return transmit(audio, **kwargs)
    monkeypatch.setattr(tx.live, 'transmit', capture)
    monkeypatch.setattr(onair.session, 'write_wav', lambda *args: None)
    a.on_host_disconnect()
    host.tick()
    assert a.said_goodbye and a.tx_seq == counter
    assert len(waves) == 1 and tx.tx_audio_start == g.boundary(0)
    assert tx.rig.edges == [True, False]
    packets = p1rx.decode_p1_packets(np.pad(waves[0], (4800, 4800)), onair.FS)
    assert len(packets) == 1
    assert packets[0].status & spec.STATUS_QRT
    assert packets[0].status & spec.STATUS_SEQ == counter
    assert packets[0].payload == pactor1.qrt_field('WS8EOC', host.p1_baud)


@pytest.mark.parametrize('current_grant', [False, True])
def test_uninvited_attempt_needs_the_same_owned_current_p1_proof(current_grant):
    from hfmodem.tests.shrike.test_uninvited_entry import _uninvited
    link = _uninvited()
    host, a = link.host, link.host.arq
    host.no_p3_fallback = True
    assert host._uninvited_entry and not a.entry_pending
    wire = a._inflight.payload + bytes(a._outbuf)
    counter = a.tx_seq
    if current_grant:
        link.grant()
        assert a._entry_p1_current
    a.on_host_disconnect()
    host.tick()
    if current_grant:
        assert a.said_goodbye and a.tx_seq == counter
        assert host.protocol is spec.Protocol.PACTOR1
        host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
    else:
        assert not a.said_goodbye
    assert a.state is arq.State.DISCONNECTED and not a.goodbye_acked
    retained(host, wire)
