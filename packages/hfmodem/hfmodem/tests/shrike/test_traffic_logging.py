# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Traffic counters and speed reports describe observations, not ARQ intent."""
from types import SimpleNamespace

from hfmodem.shrike import onair, p3rx, placement, rxfront, traffic


def test_requested_speed_is_not_observed_speed():
    log = traffic.TrafficLog()
    log.packet("RX", 1, 3, b"\r;PQ:")
    request = "\n".join(log.control("TX", 3))
    assert "SL1 -> SL2" in request
    assert "word=0x95AC5 bits-lsb-first=10100011010110101001 payload=0B" in request
    repeat = "\n".join(log.packet("RX", 1, 3, b"\r;PQ:"))
    assert "copy=2 repeat=1" in repeat
    assert "speed-up unconfirmed" in repeat
    next_packet = "\n".join(log.packet("RX", 1, 0, b" 5152"))
    assert "copy=1 repeat=0" in next_packet
    assert "speed-up confirmed" not in next_packet
    # A changeover's synthetic level is not ordinary data speed evidence.
    log.packet("RX", 1, 0, b"RMS", kind="CHANGEOVER")
    changed = "\n".join(log.packet("RX", 2, 1, b"more"))
    assert "RX data SL1 -> SL2 observed" in changed
    assert "speed-up confirmed" in changed
    assert log.requested_rx_level is None


def test_direction_and_control_counters_do_not_conflate_packets():
    log = traffic.TrafficLog()
    log.packet("TX", 2, 0, b"data")
    log.control("RX", 1)
    assert "attempt=2 retry=1" in log.packet("TX", 1, 0, b"data")[0]
    assert log.levels == {"TX": 1}
    log.packet("RX", 4, 0, b"peer")
    assert "attempt=1 retry=0" in log.packet("TX", 1, 0, b"data")[0]
    assert log.levels == {"TX": 1, "RX": 4}


def test_cs2_acks_odd_counter_and_repeat_replies_are_counted():
    arq = SimpleNamespace(rx_seq=3, tx_seq=0, _rx_seen=True)
    log = traffic.TrafficLog()
    log.packet("RX", 1, 3, b"abc")
    assert "CS2 ACK seq=3 reply-copy=1 reply-repeat=0" in log.control("TX", 1, arq)[0]
    log.packet("RX", 1, 3, b"abc")
    assert "reply-copy=2 reply-repeat=1" in log.control("TX", 1, arq)[0]
    assert "CS2 REPEAT seq=0" in log.control("RX", 1, arq)[0]
    assert "CS1 ACK seq=0" in log.control("RX", 0, arq)[0]
    arq.tx_seq = None
    assert "no outstanding TX packet" in log.control("RX", 0, arq)[0]
    arq._rx_seen = False
    assert "REPEAT/await seq=0" in log.control("TX", 1, arq)[0]


def test_modulo_four_wrap_resets_packet_copy_count():
    log = traffic.TrafficLog()
    for seq in (0, 1, 2, 3, 0):
        assert "copy=1 repeat=0" in log.packet("RX", 1, seq, b"same")[0]
        assert "copy=2 repeat=1" in log.packet("RX", 1, seq, b"same")[0]


def test_cs4s_answering_different_tx_packets_are_not_labelled_repeats():
    log = traffic.TrafficLog()
    for sl, seq in ((1, 0), (2, 1), (3, 2)):
        log.packet("TX", sl, seq, b"data")
        assert "reply-copy=1 reply-repeat=0" in log.control("RX", 3)[0]
    log.packet("TX", 3, 2, b"data")
    assert "reply-copy=2 reply-repeat=1" in log.control("RX", 3)[0]


def test_full_decoded_information_and_binary_payload_survive(capsys):
    path = placement.SPEED_PATHS[3]
    payload = bytes(range(40)) + b"\xff\r\nEND"
    field = placement.build_field(placement.field_info(payload, path.crc_bytes-3, 2), path)
    decoded = p3rx.packet_of(field, path, 0)
    event = rxfront._packet_event(decoded)
    assert event.information == field[:-2]
    assert repr(payload) in event.text
    assert event.text.endswith("[CRC-VALID]")
    host = SimpleNamespace(arq=None)
    traffic.received(host, event)
    output = capsys.readouterr().out
    assert payload.hex() in output
    assert repr(payload) in output
    assert field[:-2].hex() in output


def test_tx_refusal_does_not_spend_retry_and_dry_run_is_labelled(tmp_path, monkeypatch, capsys):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    refuse = False

    def emit(*args, **kwargs):
        tx.refused = refuse
        if not refuse:
            tx.n += 1

    monkeypatch.setattr(tx, "_tx", emit)
    tx.send_packet(1, b"abc", 0)
    first = capsys.readouterr().out
    assert "TX(dry-run) P3 SL1 DATA" in first
    assert "attempt=1 retry=0" in first
    expected = placement.build_field(placement.field_info(b"abc", 5, 0), placement.SPEED_PATHS[1])
    assert expected.hex() in first
    refuse = True
    tx.send_packet(1, b"abc", 0)
    assert "[traffic]" not in capsys.readouterr().out
    refuse = False
    tx.send_packet(1, b"abc", 0)
    assert "attempt=2 retry=1" in capsys.readouterr().out
    tx.send_packet(1, b"TX ", 1)
    assert "bytes=b'TX '" in capsys.readouterr().out


def test_host_shares_transmit_and_receive_observations(tmp_path):
    host = SimpleNamespace(arq=None)
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.attach(host)
    event = rxfront.Event(0, "packet", "", protocol="PACTOR-3",
                          packet=(3, 0, b"abc", True))
    traffic.received(host, event)
    assert tx.traffic is traffic.for_host(host)
    assert "SL3 -> SL4" in "\n".join(tx.traffic.control("TX", 3))


def test_refused_speed_request_does_not_announce_a_request(tmp_path, monkeypatch, capsys):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.traffic.packet("RX", 1, 3, b"\r;PQ:")
    refuse = True

    def emit(*args, **kwargs):
        tx.refused = refuse
        if not refuse:
            tx.n += 1

    monkeypatch.setattr(tx, "_tx", emit)
    tx._send_p3_control(3)
    assert tx.traffic.requested_rx_level is None
    assert "[traffic]" not in capsys.readouterr().out
    refuse = False
    tx._send_p3_control(3)
    assert tx.traffic.requested_rx_level == 2
    assert "reply-copy=1 reply-repeat=0" in capsys.readouterr().out
