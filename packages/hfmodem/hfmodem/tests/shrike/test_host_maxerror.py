# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS MAXError controls air budgets, independently of local admission failure."""
import pytest

from hfhost import ptc
from hfmodem.shrike import arq
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_ptc_compatibility import Client


class Seam(arq.ArqIO):
    def __init__(self):
        self.calls = self.packets = 0
        self.refuse = False

    def attach(self, host):
        self.host = host

    def connect_burst(self, *args):
        self.calls += 1

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets += 1
        return arq.REFUSED if self.refuse else len(payload)

    def send_cs(self, index):
        return arq.REFUSED if self.refuse else None

    def pump(self):
        pass

    def cycle(self):
        pass


def connected(role=arq.ISS, refuse=False):
    seam = Seam()
    seam.refuse = refuse
    modem = arq.PactorArq(seam, arq.ArqConfig(host_error_limit=30, max_retries=2))
    modem.role, modem.state = role, arq.State.CONNECTED
    return modem, seam


def test_abbreviated_terminal_and_forwarded_host_commands_bind_the_same_budget():
    host = PtcHost()
    assert b'ERROR' not in host.feed(b'MAXE 35\r')
    assert host.arq.cfg.host_error_limit == 35
    assert host.arq.cfg.max_retries == 8  # local refusal/entry policy is independent
    client = Client(host)
    assert client.cmd('#MAXE 30').code == ptc.OK
    assert host.arq.cfg.host_error_limit == 30
    configuration = host.arq.cfg
    assert client.retry().code == ptc.OK
    assert host.arq.cfg is configuration
    host.arq.state = arq.State.CONNECTING
    assert client.cmd('#MAXE 31').code == ptc.FAIL
    assert host.arq.cfg.host_error_limit == 30
    assert host.settings['MAXERROR'] == '30'


@pytest.mark.parametrize('value', ['29', '256', '-1', 'bad', '35 36'])
def test_invalid_profile_never_commits_a_setting(value):
    host = PtcHost()
    original = host.arq.cfg
    assert b'ERROR' in host.feed(f'MAXE {value}\r'.encode())
    assert host.arq.cfg is original and 'MAXERROR' not in host.settings


def test_unanswered_call_stops_after_the_selected_number_of_sync_packets():
    seam = Seam()
    host = PtcHost(seam)
    host.feed(b'MAXE 30\r')
    client = Client(host)
    assert client.cmd('C N0DX').code == ptc.OK
    assert seam.calls == 1
    for _ in range(29):
        host.tick()
        assert host.arq.state == arq.State.CONNECTING
    assert seam.calls == 30
    host.tick()
    assert host.arq.state == arq.State.DISCONNECTED
    assert seam.calls == 30
    assert host.arq.payload_ledger.confirmed == 0


def test_iss_consecutive_air_errors_use_host_limit_and_keep_originals_unconfirmed():
    modem, _ = connected()
    original = b'unconfirmed original'
    modem.on_host_data(original)
    modem.on_cycle()  # first offered packet
    for _ in range(29):
        modem._on_nak(False, answered=False)
        assert modem.state == arq.State.CONNECTED and not modem._qrt_pending
    assert modem._inflight.retries == 29
    modem._on_nak(False, answered=False)
    assert modem._qrt_pending
    assert modem.payload_ledger.confirmed == 0
    assert modem.payload_ledger.unconfirmed == original


def test_valid_repeat_request_resets_errors_without_confirming_the_packet():
    modem, _ = connected()
    modem.on_host_data(b'original')
    modem.on_cycle()
    for _ in range(20):
        modem._on_nak(False, answered=False)
    modem._unrepaired, modem._silent_cycles = 20, 15
    modem.on_rx_cs(arq.CS_REQUEST)
    assert modem._inflight.retries == 0
    assert modem._unrepaired == modem._silent_cycles == 0
    assert modem.payload_ledger.confirmed == 0
    for _ in range(29):
        modem._on_nak(False, answered=False)
    assert modem.state == arq.State.CONNECTED and not modem._qrt_pending


def test_call_error_budget_resets_only_on_a_qualified_answer():
    seam = Seam()
    host = PtcHost(seam)
    host.feed(b'MAXE 30\r')
    client = Client(host)
    client.cmd('C N0DX')
    for _ in range(15):
        host.tick()
    host.arq.on_rx_cs(arq.CS_REQUEST)
    assert host.arq.state == arq.State.CONNECTING
    for _ in range(30):
        host.tick()
        assert host.arq.state == arq.State.CONNECTING
    assert seam.calls == 46
    host.tick()
    assert host.arq.state == arq.State.DISCONNECTED and seam.calls == 46


def test_unidentified_activity_cannot_extend_the_host_call_budget():
    seam = Seam()
    host = PtcHost(seam)
    host.feed(b'MAXE 30\r')
    client = Client(host)
    client.cmd('C N0DX')
    for _ in range(30):
        host.arq.note_peer_heard()
        host.tick()
    assert host.arq.state == arq.State.DISCONNECTED and seam.calls == 30


def test_irs_silence_and_bad_crc_frames_use_host_air_limit():
    modem, _ = connected(role=arq.IRS)
    for _ in range(29):
        modem._charge_irs_silence()
    assert modem.state == arq.State.CONNECTED and not modem._qrt_pending
    modem._charge_irs_silence()
    assert modem._qrt_pending

    modem, _ = connected(role=arq.IRS)
    for _ in range(29):
        modem.on_rx_packet(1, b'not delivered', 0, crc_ok=False)
        assert not modem._qrt_pending
    modem.on_rx_packet(1, b'not delivered', 0, crc_ok=False)
    assert modem._qrt_pending


def test_local_transmit_refusal_keeps_its_own_shorter_budget():
    modem, _ = connected(refuse=True)
    modem.on_host_data(b'never transmitted')
    for _ in range(4):
        modem.on_cycle()
    assert modem._qrt_pending
    assert modem.payload_ledger.confirmed == 0


def test_mixed_bad_frames_and_silence_share_one_consecutive_error_budget():
    modem, _ = connected(role=arq.IRS)
    for _ in range(20):
        modem.on_rx_packet(1, b'bad', 0, crc_ok=False)
    for _ in range(9):
        modem._charge_irs_silence()
    assert modem._host_air_errors == 29 and not modem._qrt_pending
    modem._charge_irs_silence()
    assert modem._qrt_pending


def test_valid_packet_resets_mixed_errors_but_unknown_control_does_not():
    modem, _ = connected(role=arq.IRS)
    for _ in range(20):
        modem.on_rx_packet(1, b'bad', 0, crc_ok=False)
    modem.on_rx_cs(9999)
    assert modem._host_air_errors == 20
    modem.on_rx_packet(1, b'good', 0, crc_ok=True)
    assert modem._host_air_errors == 0
    for _ in range(29):
        modem._charge_irs_silence()
    assert not modem._qrt_pending


def test_bad_crc_and_silence_settlement_charge_one_receive_opportunity():
    modem, _ = connected(role=arq.IRS)
    token = modem.begin_receive_opportunity()
    modem.on_rx_packet(1, b'bad', 0, crc_ok=False)
    modem._charge_irs_silence()
    modem.finish_receive_opportunity(token)
    assert modem._host_air_errors == 1


def test_255_disables_air_timeout_but_not_local_refusal_bounds():
    seam = Seam()
    host = PtcHost(seam)
    assert b'ERROR' not in host.feed(b'MAXE 255\r')
    client = Client(host)
    client.cmd('C N0DX')
    for _ in range(300):
        host.tick()
    assert seam.calls == 301 and host.arq.state == arq.State.CONNECTING

    modem, _ = connected()
    modem.cfg = arq.ArqConfig(host_error_limit=255, max_retries=2)
    modem.on_host_data(b'original')
    modem.on_cycle()
    for _ in range(300):
        modem._on_nak(False, answered=False)
    assert not modem._qrt_pending and modem.payload_ledger.confirmed == 0

    modem, _ = connected(refuse=True)
    modem.cfg = arq.ArqConfig(host_error_limit=255, max_retries=2)
    modem.on_host_data(b'never transmitted')
    for _ in range(4):
        modem.on_cycle()
    assert modem._qrt_pending
