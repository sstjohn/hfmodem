# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""CSDelay reaches the native answer phase, with admission before keying."""
from types import SimpleNamespace

import pytest
from hfhost import ptc

from hfmodem.shrike import onair, p1rx
from hfmodem.shrike.ptc import PtcHost, SimPeer
from .test_identify import transmitter
from .test_ptc_compatibility import Client
from .test_station_runtime import running as running, Client as SerialClient


@pytest.mark.parametrize('value', [10, 18])
def test_csdelay_places_actual_control_audio_after_the_call_frame(tmp_path, value):
    tx, _, _ = transmitter(tmp_path)
    host = PtcHost(tx)
    client = Client(host)
    assert host.settings['CSDELAY'] == '18'  # Existing native 90 ms default.
    assert client.cmd('#TXD 4').code == ptc.OK
    assert client.cmd(f'#CSD {value}').code == ptc.OK
    assert client.cmd('JHOST0').code == ptc.OK
    assert f'CSDELAY: {value}'.encode() in host.feed(b'CSD\r')
    assert tx.get_host_control('CSDELAY') == value
    start = tx.live.samples + onair.FS
    grid = onair._called_grid(start, p1rx.Connect('Normal', 'N0CALL', False),
        tx.settle, offset_n=0, response_delay_s=tx.validate_host_answer())
    tx.aim(grid, 0)
    tx.send_p1_cs(0)
    assert tx.tx_audio_start == start + round((.960 + value * .005) * onair.FS)
    assert tx.tx_audio_start - tx.tx_key_up == round(.020 * onair.FS)
    assert tx.rig.edges == [True, False]


@pytest.mark.parametrize('value', ['0', '32', '-1', 'bad'])
def test_invalid_csdelay_preserves_driver_and_readback(tmp_path, value):
    tx, _, _ = transmitter(tmp_path)
    host = PtcHost(tx)
    client = Client(host)
    before = dict(host.settings), tx.get_host_control('CSDELAY')
    assert client.cmd('#CSD '+value).code == ptc.FAIL
    assert (host.settings, tx.get_host_control('CSDELAY')) == before


def test_csdelay_policy_cannot_change_until_native_contact_retires(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    host = PtcHost(tx)
    host.enable_owner()
    client = Client(host)
    assert client.cmd('#CSD 10').code == ptc.OK
    with host._owner.contact():
        assert client.cmd('#CSD 18').code == ptc.FAIL
        assert client.cmd('#CSD 10').code == ptc.OK
        assert tx.get_host_control('CSDELAY') == 10
    assert client.cmd('#CSD 18').code == ptc.OK


def test_csdelay_requires_a_backend_effect_and_simulation_is_explicit():
    host = PtcHost(SimpleNamespace(attach=lambda host: None))
    client = Client(host)
    assert client.cmd('#CSD 10').code == ptc.FAIL
    assert 'CSDELAY' not in host.settings
    simulated = SimPeer()
    logical = PtcHost(simulated)
    assert b'CSDELAY: 5' in logical.feed(b'CSD\r')
    assert b'ERROR' not in logical.feed(b'CSD 10\r')
    assert simulated.controls['CSDELAY'] == 10


@pytest.mark.parametrize('unsupported', [5, 31])
def test_incoherent_staged_timing_refuses_contact_before_reservation_or_keying(running, unsupported):
    runtime, station, server = running
    client = SerialClient(server.endpoint)
    try:
        assert client.command(f'#CSD {unsupported}').code == ptc.OK
        assert client.command('C N0DX').code == ptc.FAIL
        assert not runtime._answer_timing_ready()
        assert station.arbiter.contact_protocol is None
        assert not station.rig.edges and not station.audio.sent
        # The actual clients finish initialization with these coherent values.
        assert client.command('#TXD 4').code == ptc.OK
        assert client.command('#CSD 10').code == ptc.OK
        assert runtime._answer_timing_ready()
        assert runtime.tx.validate_host_answer() == .050
        assert runtime.failure is None
    finally:
        client.close()
