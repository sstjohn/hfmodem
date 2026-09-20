"""A recorded ordinary role request cannot acknowledge our CS3 payload.

The request is real WS8EOC PCM; the subsequent bare CS1 is a rendered positive
control, not a claim that the failed 2325 session contained that answer.
"""
import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import onair, p3acquire, rxfront, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_entry_answer import _answer
from hfmodem.tests.shrike.test_iss_answer_placement import CS_PHASE_N
from hfmodem.tests.shrike.test_iss_turn_changeover import takes_the_channel


def test_recorded_request_preserves_changeover_until_decoded_control_ack():
    path = evidence.CAPTURES / 'onair-0919-2325' / 'hold_114.wav'
    if not path.exists():
        pytest.skip('local WS8EOC September 19 23:25 capture unavailable')
    rate, raw = wavfile.read(path)
    assert rate == onair.FS
    pcm = p3acquire.compensate(raw[:, 0].astype(float) / 32768, -8.8)
    request = rxfront.SyncedRx().sl1_packet_at(pcm, 23949)
    assert request is not None
    assert request.packet == (1, 0x40, b'', True)
    assert not request.breakin

    host, peer, login = takes_the_channel()
    pending = host.arq._inflight
    assert pending.payload == b';FW' and pending.seq == 0
    for _ in range(3):
        host.on_rx_event(request)
        assert host.arq.unconfirmed_breakin
        assert host.arq._inflight is pending
        assert host.arq._buffer_raw == len(login)
        assert len(peer.bursts) == 1

    receiver = onair._SessionRx(host, tag='TEST')
    receiver.p3_receive_offset_hz = -8.8
    lead = 9600
    ack = np.pad(_answer(0, -8.8), (lead, 4800))
    receiver.new_cycle()
    assert receiver.control_signal(ack, 0, lead + CS_PHASE_N) == 0
    assert not host.arq.unconfirmed_breakin
    assert host.arq._inflight.seq == 1
    assert host.arq._buffer_raw == len(login) - 3
    assert len(peer.bursts) == 2

    # Repeating CS1 requests retransmission of counter 1; it cannot retire it.
    ordinary = host.arq._inflight
    receiver.new_cycle()
    receiver.control_signal(ack, 60000, 60000 + lead + CS_PHASE_N)
    assert host.arq._inflight is ordinary
    assert host.arq._buffer_raw == len(login) - 3


@pytest.mark.parametrize('protocol', [spec.Protocol.PACTOR1, spec.Protocol.PACTOR2])
def test_legacy_other_protocol_request_interpretation_is_unchanged(protocol):
    host, peer, login = takes_the_channel()
    host.arq._stint_tail = False
    host.arq.on_rx_packet(1, b'', 0x40, True, protocol=protocol)
    assert not host.arq.unconfirmed_breakin
    assert host.arq._buffer_raw == len(login) - 3
