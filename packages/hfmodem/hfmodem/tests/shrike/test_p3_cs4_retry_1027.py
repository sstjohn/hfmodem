# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""K4MSU 0546: a second CS4 must not strand the first ordinary field."""
from hfmodem.shrike import arq
from hfmodem.tests.shrike.test_tx_gear_hold import sender, control


def test_recorded_k4msu_controls_reach_exact_field_retry():
    import hashlib
    import json
    from pathlib import Path
    from scipy.io import wavfile
    from hfmodem.shrike import p3acquire, rxfront

    folder = Path(__file__).with_name('fixtures')/'k4msu-1027'
    pin = json.loads((folder/'second-cs4.json').read_text())
    fs, pcm = wavfile.read(folder/'second-cs4.wav')
    assert fs == 48000 and hashlib.sha256(pcm.tobytes()).hexdigest() == pin['pcm_sha256']
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    for index in range(2):
        n = pin['window_samples']
        audio = p3acquire.compensate(pcm[index*n:(index+1)*n].astype(float)/32768,
                                     pin['offset_hz'])
        event = rxfront.SyncedRx().control_signal_tracked(audio, pin['local_phase'],
                                                        details=False)
        assert event is not None and event.cs == arq.CS_SPEED_UP
        assert '0 bit errors' in event.text
        host.on_rx_event(event)
    assert len(seam.frames) == 2 and seam.frames[0] == seam.frames[1]
    assert host.arq.tx_seq == 1 and host.arq._mail_cs4_wait == 0


def test_refused_confirmation_retry_does_not_get_replayed_or_credit_bytes():
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    control(host, 4)
    packet = host.arq._inflight
    credit = host.arq.tx_acked_bytes
    seam.refuse = True
    control(host, 4)
    seam.refuse = False
    host.tick()
    control(host, 4)
    assert len(seam.frames) == 1
    assert host.arq._inflight is packet and host.arq.tx_acked_bytes == credit


def test_second_cs4_retries_exact_field_without_credit_or_speed_change():
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    control(host, 4)
    a = host.arq
    packet = a._inflight
    field = seam.frames[-1]
    before = (a.tx_acked_bytes, host.sent_total, a._buffer_raw,
              bytes(a._outbuf), a.tx_seq)
    control(host, 4)
    assert seam.frames == [field, field]
    assert a._inflight is packet and a._mail_cs4_wait == 0
    assert (a.tx_acked_bytes, host.sent_total, a._buffer_raw,
            bytes(a._outbuf), a.tx_seq) == before
    # A peer that lost the first copy can now answer the retained counter.
    control(host, 2)
    assert a._mail_cs4_wait is None
    assert a.tx_acked_bytes == before[0]+len(packet.payload)
    credited = a.tx_acked_bytes
    control(host, 2)
    assert a.tx_acked_bytes == credited


def test_cs4_retry_is_one_bounded_attempt_and_silence_retains_all_bytes():
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    control(host, 4)
    packet = host.arq._inflight
    remaining = packet.payload+bytes(host.arq._outbuf)
    credit = host.arq.tx_acked_bytes
    control(host, 4)
    assert len(seam.frames) == 2
    for _ in range(5):
        control(host, 4)
        host.tick()
    assert len(seam.frames) == 2
    host.tick(elapsed_ticks=91)
    assert host.arq.state == arq.State.DISCONNECTED
    retained = host.arq.transfer_unconfirmed
    assert retained['field']+retained['suffix'] == remaining
    assert host.arq.tx_acked_bytes == credit
