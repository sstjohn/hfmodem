# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PTC voltage/delay commands reach rendered output and the keying clock."""
import numpy as np
import pytest
from types import SimpleNamespace

from hfhost import ptc
from hfmodem.shrike import arq, onair
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.core import cwid, levels
from hfmodem.core.occupied import occupied_hz
from hfmodem.tests.shrike.test_identify import transmitter, CALL
from hfmodem.tests.shrike.test_ptc_compatibility import Client


def test_separate_fsk_and_psk_levels_reach_every_native_producer(tmp_path, monkeypatch):
    keyed = []
    monkeypatch.setattr(onair.session, 'write_wav',
                        lambda path, audio: keyed.append(np.asarray(audio)))
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path, drive=.8, p1_drive=.1)
    host = PtcHost(tx)
    host.feed(b'FSKA 750\rPSKA 1500\r')
    assert b'FSKAMPL: 750' in host.feed(b'FSKA\r')
    client = Client(host)
    tx.connect_burst('W9SSJ', 'N0DX')
    tx.send_p1_packet(b'original', 100, 1)
    tx.send_p1_cs(0)
    tx.send_p1_breakin(b'', 100, 0)
    tx.send_entry_packet(1, b'', 0x1a)
    tx.send_cs(0)
    tx._tx_p2(np.sin(np.arange(4800) * .2).astype(np.float32), 'P2 test')
    assert len(keyed) == 7
    assert [float(np.max(np.abs(a))) for a in keyed] == pytest.approx(
        [.2, .2, .2, .2, .4, .4, .4], abs=1e-6)
    # A later PSK change cannot change FSK, nor vice versa.
    assert client.cmd('#PSKA 3000').code == ptc.OK
    tx.send_p1_cs(0)
    tx.send_cs(0)
    assert float(np.max(np.abs(keyed[-2]))) == pytest.approx(.2)
    assert float(np.max(np.abs(keyed[-1]))) == pytest.approx(.8)


def test_cw_uses_the_explicit_fsk_level_even_after_a_psk_burst(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    host = PtcHost(tx)
    assert b'ERROR' not in host.feed(b'FSKA 1500\rPSKA 3000\r')
    tx.identify(CALL)
    assert np.max(np.abs(tx.live.sent[-1])) == pytest.approx(.15 * 10 ** (-3 / 20), abs=1e-6)


def test_txdelay_reaches_actual_key_to_audio_sample_interval(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    host = PtcHost(tx)
    client = Client(host)
    assert client.cmd('#TXD 7').code == ptc.OK
    assert tx.settle == pytest.approx(.035)
    tx.identify(CALL)
    assert tx.tx_audio_start - tx.tx_key_up == round(.035 * onair.FS)


def test_trimode_cws70_renders_fourteen_wpm_at_the_actual_output(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    host = PtcHost(tx)
    assert b'CWSPEED: 100' in host.feed(b'CWS\r')
    assert b'ERROR' not in host.feed(b'CWS 70\rFSKA 1500\r')
    assert b'CWSPEED: 70' in host.feed(b'CWS\r')
    client = Client(host)
    assert client.cmd('#CWS 70').code == ptc.OK
    tx.identify(CALL)
    expected = cwid.audio(CALL, wpm=14, tone_hz=sum(occupied_hz('pactor')) / 2)
    expected = levels.at_drive(expected, .15 * 10 ** (-3 / 20))
    sounding = np.flatnonzero(np.abs(expected) > .02 * np.max(np.abs(expected)))
    expected = expected[sounding[0]:sounding[-1] + 1]
    np.testing.assert_array_equal(tx.live.sent[-1], expected)
    assert tx.tx_end - tx.tx_audio_start == len(expected)
    assert tx.keyed[-1][1] == pytest.approx(cwid.duration(CALL, 14), abs=.01)


@pytest.mark.parametrize('value', [39, 101, 160, 1200, 1201])
def test_cw_speed_refusal_preserves_effective_readback_and_rendering(tmp_path, value):
    tx, _, _ = transmitter(tmp_path)
    host = PtcHost(tx)
    client = Client(host)
    assert client.cmd('#CWS 70').code == ptc.OK
    assert client.cmd(f'#CWS {value}').code == ptc.FAIL
    assert host.settings['CWSPEED'] == '70' and tx.get_host_control('CWSPEED') == 70
    tx.identify(CALL)
    audio = cwid.audio(CALL, wpm=14, tone_hz=sum(occupied_hz('pactor')) / 2)
    sounding = np.flatnonzero(np.abs(audio) > .02 * np.max(np.abs(audio)))
    assert len(tx.live.sent[-1]) == sounding[-1] - sounding[0] + 1


def test_speed_changes_require_idle_and_logical_simulator_has_its_own_range(tmp_path):
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    host = PtcHost(tx)
    client = Client(host)
    assert client.cmd('#CWS 40').code == ptc.OK
    host.arq.state = arq.State.CONNECTING
    assert client.cmd('#CWS 70').code == ptc.FAIL
    assert client.cmd('#CWS 40').code == ptc.OK
    assert host.settings['CWSPEED'] == '40' and tx.get_host_control('CWSPEED') == 40
    peer = SimPeer()
    logical = PtcHost(peer)
    assert b'CWSPEED: 160' in logical.feed(b'CWS\r')
    assert b'ERROR' not in logical.feed(b'CWS 1200\r')
    assert peer.controls['CWSPEED'] == 1200


@pytest.mark.parametrize('command', [b'FSKA 29', b'PSKA 3001', b'TXD 0',
                                      b'TXD 32', b'FSKA bad', b'PSKA 30 60'])
def test_invalid_settings_do_not_mutate_driver_or_readback(tmp_path, command):
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    host = PtcHost(tx)
    before = dict(host.settings), tx.settle, tx._fsk_drive, tx._psk_drive
    assert b'ERROR' in host.feed(command + b'\r')
    assert (host.settings, tx.settle, tx._fsk_drive, tx._psk_drive) == before


def test_calibration_refusal_and_busy_channel_preserve_effective_settings(tmp_path):
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path, drive=.6,
                       drive_reference_mv=1000)
    host = PtcHost(tx)
    client = Client(host)
    assert client.cmd('#FSKA 500').code == ptc.OK
    assert tx._fsk_drive == pytest.approx(.3)
    assert client.cmd('#FSKA 1500').code == ptc.FAIL
    assert host.settings['FSKAMPL'] == '500' and tx._fsk_drive == pytest.approx(.3)
    host.arq.state = arq.State.CONNECTING
    assert client.cmd('#FSKA 750').code == ptc.FAIL
    assert tx._fsk_drive == pytest.approx(.3)
    assert client.cmd('#FSKA 500').code == ptc.OK


def test_missing_backend_effect_is_refused_and_simulator_is_explicit():
    host = PtcHost(SimpleNamespace(attach=lambda host: None))
    assert b'ERROR' in host.feed(b'FSKA 300\r')
    assert 'FSKAMPL' not in host.settings
    peer = SimPeer()
    host = PtcHost(peer)
    assert b'ERROR' not in host.feed(b'FSKA 300\r')
    assert peer.controls == {'FSKAMPL': 300}
