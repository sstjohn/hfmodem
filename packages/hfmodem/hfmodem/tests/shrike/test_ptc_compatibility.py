# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Source-backed consumer contracts, using hfhost's independent CRC codec."""
import pytest

from hfhost import ptc
from hfmodem.shrike import compress
from hfmodem.core import hostmode
from hfmodem.shrike.arq import State, IRS, _Packet
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol


class Client:
    def __init__(self, host):
        self.host, self.bit = host, 0
        self.decoder = ptc.Deframer()
        host.feed(b"PTC 31\rJHOST4\r")

    def exchange(self, data, command=True, channel=31):
        self.bit ^= 1
        self.frame = ptc.Packet(channel, command, data, counter=self.bit).encode()
        return self.decoder.feed(self.host.feed(self.frame))[0]

    def cmd(self, text, channel=31):
        return self.exchange(text.encode(), channel=channel)

    def retry(self):
        return self.decoder.feed(self.host.feed(self.frame))[0]


def connected():
    host = PtcHost()
    client = Client(host)
    host.arq.state, host.arq.role = State.CONNECTED, IRS
    host.arq.dxcall = "N0DX"
    host._link_established("N0DX")
    assert "CONNECTED" in client.cmd("G").text
    return host, client


def settle(host, size):
    wire = bytes(host.arq._outbuf[:size])
    del host.arq._outbuf[:size]
    host.arq._inflight = _Packet(0, wire, 1)
    host.arq._settle_inflight()


def test_recovery_retains_original_bytes_when_wire_stops_inside_an_escape(tmp_path):
    import hashlib
    import json
    from hfmodem.shrike.onair import _write_unconfirmed_transfer

    host, client = connected()
    original = b'x' + bytes(range(0x1c, 0x20)) + b'end'
    assert client.exchange(original, False).code == ptc.OK
    settle(host, 2)
    assert host.arq.payload_ledger.confirmed == 1
    assert client.cmd('DD').code == ptc.OK
    retained = host.arq.transfer_unconfirmed
    path = _write_unconfirmed_transfer(tmp_path, host.arq)
    assert path.read_bytes() == retained['field'] + retained['suffix']
    report = json.loads((tmp_path / 'p3-transfer-unconfirmed.json').read_text())
    originals = report['host_originals']
    assert (tmp_path / originals['file']).read_bytes() == original[1:]
    assert originals['bytes'] == len(original) - 1
    assert originals['host_accepted'] == len(original)
    assert originals['host_confirmed'] == 1
    assert originals['sha256'] == hashlib.sha256(original[1:]).hexdigest()
    assert path.read_bytes() != original[1:]
    with pytest.raises(FileExistsError):
        _write_unconfirmed_transfer(tmp_path, host.arq)


def test_trimode_initialization_and_transactional_channel_validation():
    host = PtcHost()
    for line in ("CONT 3", "ARX 0", "BR 0", "CHOB 0", "CM 0", "CWID 0 0",
                 "CWS 70", "L 0", "LF 0", "MYL 3", "MAXE 35", "PTC 31",
                 "REM 0", "MODE 0", "ST 2", "UML 0", "CSD 5", "TONES 4"):
        assert b"ERROR" not in host.feed(line.encode() + b"\r")
    assert host.ptchn == 31
    assert host._resolve("PTCC") != "PTCHN"
    assert b"ERROR" in host.feed(b"PTC 32\r")
    assert host.ptchn == 31


def test_disconnected_and_wrong_channel_data_are_rejected():
    host = PtcHost()
    client = Client(host)
    assert client.exchange(b"lost", False).code == ptc.FAIL
    assert client.exchange(b"lost", False, channel=4).code == ptc.FAIL
    assert not host.arq._outbuf


@pytest.mark.parametrize('state', [State.CONNECTING, State.CONNECTED, State.DISCONNECTING])
@pytest.mark.parametrize('route', ['terminal', 'hostmode', 'forwarded'])
def test_another_connect_cannot_replace_the_active_contact(state, route):
    host = PtcHost(mycall='N0CALL')
    calls = []
    host.connect_burst = lambda own, remote: calls.append((own, remote))
    client = Client(host) if route != 'terminal' else None
    if client:
        assert client.cmd('C N0FIRST').code == ptc.OK
    else:
        assert b'ERROR' not in host.feed(b'C N0FIRST\r')
    host.arq.state = state
    before = (host.arq.mycall, host.arq.dxcall, host.arq.last_connect,
              host.arq.connect_variant, host.arq.role, host.arq._connect_retries)
    if client:
        assert client.cmd(('#' if route == 'forwarded' else '') + 'C %N0NEXT').code == ptc.FAIL
        assert client.retry().code == ptc.FAIL
    else:
        assert b'ERROR' in host.feed(b'\x1bC %N0NEXT\r')
    assert (host.arq.mycall, host.arq.dxcall, host.arq.last_connect,
            host.arq.connect_variant, host.arq.role, host.arq._connect_retries) == before
    assert host.arq.state == state
    assert calls == [('N0CALL', 'N0FIRST')]


def test_original_byte_confirmation_echo_and_cached_retry():
    host, client = connected()
    assert client.cmd("%M1").code == ptc.OK
    original = bytes(range(256))
    assert client.exchange(original, False).code == ptc.OK
    assert client.retry().code == ptc.OK
    assert host.arq.payload_ledger.accepted == 256
    settle(host, 29)  # ends in the first half of escaped 0x1c
    assert int(client.cmd("%T").text) == 28
    echo = client.cmd("G")
    assert echo.code == ptc.SENT_ECHO and echo.data == original[:28]
    assert client.retry() == echo
    settle(host, len(host.arq._outbuf))
    assert int(client.cmd("%T").text) == 256
    assert client.cmd("G").data == original[28:]
    assert client.cmd("G").code == ptc.OK


def test_protocol_retirement_and_queue_clear_do_not_credit_delivery():
    host, client = connected()
    client.cmd("%M1")
    host.arq.queue_protocol_data(b"1N0CALL\r")
    client.exchange(b"host", False)
    assert host.arq.drop_queued_prefix(b"1N0CALL\r")
    assert client.cmd("%T").text == "0"
    assert client.cmd("#CL").code == ptc.OK
    assert not host.arq._outbuf
    assert host.arq.payload_ledger.discarded == 4
    assert client.cmd("G").code == ptc.OK
    assert client.cmd("%T").text == "0"


def test_clear_preserves_inflight_escape_tail():
    host, client = connected()
    client.exchange(b"\x1cunsent", False)
    host.arq._inflight = _Packet(0, bytes(host.arq._outbuf[:1]), 1)
    del host.arq._outbuf[:1]
    client.cmd("#CL")
    assert bytes(host.arq._outbuf) == compress.transparent(b"\x1c")[1:]
    host.arq._settle_inflight()
    assert host.sent_total == 0
    settle(host, 1)
    assert host.sent_total == 1
    assert host.arq.payload_ledger.discarded == 6


def test_clear_preserves_only_offered_plain_bytes():
    host, client = connected()
    client.cmd("%M1")
    client.exchange(b"offered and unsent", False)
    host.arq._inflight = _Packet(0, bytes(host.arq._outbuf[:3]), 1)
    del host.arq._outbuf[:3]
    assert client.cmd("#CL").code == ptc.OK
    assert not host.arq._outbuf
    host.arq._settle_inflight()
    assert client.cmd("%T").text == "3"
    assert client.cmd("G").data == b"off"
    assert host.arq.payload_ledger.discarded == len(b"ered and unsent")


def test_terminal_line_overflow_cannot_execute_a_truncated_command():
    host = PtcHost()
    host.feed(b"PTC " + b"1" * 10000)
    assert len(host._line) <= 4096
    assert b"ERROR" in host.feed(b"\r")
    assert host.ptchn == 4
    host.feed(b"PTC 31\r")
    assert host.ptchn == 31


def test_forced_close_retains_original_unconfirmed_bytes_once():
    host, client = connected()
    payload = b"host\x1c\x1d\x1e"
    client.exchange(payload, False)
    client.cmd("DD")
    record = host.arq.transfer_unconfirmed
    assert record["host_unconfirmed"] == payload
    assert record["host_confirmed"] == 0
    assert host.arq.payload_ledger.discarded == len(payload)
    assert len(host.arq.unconfirmed_transfers) == 1
    client.cmd("DD")
    assert len(host.arq.unconfirmed_transfers) == 1
    assert len(host.channel(31).link) == 1


def test_final_receive_and_echo_precede_terminal_with_real_state():
    host, client = connected()
    client.cmd("%M1")
    client.exchange(b"sent", False)
    settle(host, 4)
    host.deliver(b"last received")
    host.arq.on_host_abort()
    assert host.arq.state == State.DISCONNECTED
    assert client.cmd("L").text.split()[-1] == "0"
    assert client.cmd("G3", channel=254).data[1] == 0
    assert 32 in client.cmd("G", channel=255).data
    assert client.cmd("G1").code == ptc.OK  # selective poll cannot retire payload
    assert client.cmd("C N0NEXT").code == ptc.FAIL
    assert client.cmd("G").data == b"last received"
    assert client.cmd("G").data == b"sent"
    assert "DISCONNECTED" in client.cmd("G").text
    assert host.contact_ready()


def test_drain_timeout_requires_explicit_attachment_retirement(monkeypatch):
    host, client = connected()
    host.deliver(b"unread")
    host.arq.on_host_abort()
    deadline = host.host_output_deadline
    monkeypatch.setattr("hfmodem.shrike.ptc.time.monotonic", lambda: deadline + 1)
    host.service_host_output()
    assert host.host_output_failure == "HOST OUTPUT DRAIN TIMEOUT"
    assert client.cmd("C N0NEXT").code == ptc.FAIL
    host.feed(b"")
    host.attachment_lost()
    assert host.contact_ready()
    assert not host.channel(31).rx


def test_mylevel_intersects_operator_controls_and_forwarded_errors():
    host = PtcHost()
    client = Client(host)
    assert client.cmd("#MYL 1").code == ptc.OK
    host.offer_pactor2 = True
    assert not host._upgrade_targets()
    host._follow_peer(Protocol.PACTOR3)
    assert host.protocol == Protocol.PACTOR1
    assert client.cmd("#MYL 9").code == ptc.FAIL
    assert host.settings["MYLEVEL"] == "1"
    assert client.cmd("#MO 2").code == ptc.OK
    assert host.settings['MODE'] == '2'
    assert client.cmd('#MO 9').code == ptc.FAIL
    assert host.settings['MODE'] == '2'


def test_code8_all_serial_splits_and_crc_corruption():
    frame = hostmode.reply(31, hostmode.SENT_ECHO, b"\x00\xaa\x1c\xff")
    for split in range(len(frame) + 1):
        decoder = ptc.Deframer()
        responses = decoder.feed(frame[:split]) + decoder.feed(frame[split:])
        assert responses == [ptc.Response(31, ptc.SENT_ECHO, b"\x00\xaa\x1c\xff")]
    decoder = ptc.Deframer()
    corrupt = frame[:-1] + bytes((frame[-1] ^ 1,))
    assert not decoder.feed(corrupt)


def test_echo_overflow_does_not_reenter_an_ack_transition():
    host, client = connected()
    client.cmd("%M1")
    host.host_output_limit = 1
    client.exchange(b"confirmed", False)
    # Exercise _on_ack, which examines the packet after observer callbacks.
    wire = bytes(host.arq._outbuf)
    host.arq._outbuf.clear()
    host.arq._inflight = _Packet(0, wire, 1)
    host.arq._on_ack()
    assert host.sent_total == len(wire)
    assert len(host.channel(31).echo) <= host.host_output_limit
    assert host.host_output_failure == "HOST OUTPUT OVERFLOW"
    host.service_host_output()
    assert host.arq.state == State.DISCONNECTED


def test_later_failed_acquisition_cannot_rewrite_prior_unconfirmed_originals():
    host, client = connected()
    original = b'previous contact\x00\x1c\x1d\x1e\xaa'
    assert client.exchange(original, False).code == ptc.OK
    assert client.cmd('DD').code == ptc.OK
    previous = host.arq.transfer_unconfirmed
    snapshot = dict(previous)
    assert client.cmd('G').code == ptc.LINK
    assert host.contact_ready()
    # Invalid requests do not retire the last contact's report.
    assert client.cmd('C %!N0BAD').code == ptc.FAIL
    assert host.arq.transfer_unconfirmed is previous
    assert client.cmd('C N0NEXT').code == ptc.OK
    assert client.cmd('DD').code == ptc.OK
    assert previous == snapshot and previous['host_unconfirmed'] == original
    assert host.arq.transfer_unconfirmed is None
    assert host.arq.payload_ledger.accepted == 0
    assert list(host.arq.unconfirmed_transfers) == [previous]
    terminal = client.cmd('G')
    assert terminal.code == ptc.LINK and 'LINK FAILURE with N0NEXT' in terminal.text
    assert 'transfer unconfirmed' not in terminal.text
    assert host.sent_total == 0
