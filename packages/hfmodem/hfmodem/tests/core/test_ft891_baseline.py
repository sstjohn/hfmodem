# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Literal FT-891 consumer and sample-clock baselines; no hardware is opened."""
from dataclasses import asdict
import socket
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import audio, config, levels, rigs
from hfmodem.shrike import onair
from hfmodem.station import preflight


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ('HFMODEM_STATION', 'CAT_PORT', 'PTT_PORT'):
        monkeypatch.delenv(key, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail(f'unexpected hardware boundary: {args}')

    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setitem(__import__('sys').modules, 'sounddevice',
                        SimpleNamespace(Stream=forbidden, InputStream=forbidden,
                                        OutputStream=forbidden, query_devices=forbidden))


def test_profile_and_flight_consumers():
    assert rigs.resolve_rig() == 'ft891'
    assert rigs.RIGS['ft891'] == dict(
        model=1036, baud=38400, mode='PKTUSB', settle=.04,
        tx_latency_ms=20., tx_latency_status='INFERRED',
        ptt_type='RTS', ptt_iface=1, tr_switch_s=.055,
        tr_recovery_slack_s=.015, audio_device='USB Audio Device',
        cat_port='/dev/cu.usbserial-00A6C71B0', ptt_port='/dev/cu.usbserial-00A6C71B1',
        note='Yaesu FT-891; set CAT rate to 38400 (menu 05-06); external soundcard on DATA jack',
        tune_carrier=True,
        readback={
            'rig_levels': ('PREAMP', 'ATT', 'AF', 'RF', 'SQL', 'IF', 'NR',
                           'NOTCHF', 'AGC', 'NB', 'MONITOR_GAIN', 'RFPOWER',
                           'STRENGTH', 'RAWSTR', 'ALC', 'SWR'),
            'rig_funcs': ('NB', 'NR', 'ANF', 'APF', 'MN', 'COMP', 'MON', 'TUNER'),
            'unreported_levels': ('AGC', 'METER'),
            'governed_by': {'NR': 'NR', 'NB': 'NB', 'NOTCHF': 'MN',
                            'COMP': 'COMP', 'MONITOR_GAIN': 'MON'},
            'meters': ('STRENGTH', 'RAWSTR', 'ALC', 'SWR'),
            'preamp_meaning': {'0': 'IPO, preamp bypassed', '10': 'AMP1', '20': 'AMP2'},
            'menu_group': '08',
            'menu_items': ('0801', '0802', '0803', '0804', '0805', '0806',
                           '0807', '0808', '0809', '0810', '0811', '0812'),
            'menu_labels': {
                '0801': 'DATA MODE', '0802': 'PSK TONE', '0803': 'OTHER DISP',
                '0804': 'OTHER SHIFT', '0805': 'DATA LCUT FREQ',
                '0806': 'DATA LCUT SLOPE', '0807': 'DATA HCUT FREQ',
                '0808': 'DATA HCUT SLOPE', '0809': 'DATA IN SELECT',
                '0810': 'DATA PTT SELECT', '0811': 'DATA OUT LEVEL', '0812': 'DATA BFO'},
            'menu_values': {
                '0801': {'0': 'PSK', '1': 'OTHERS'},
                '0805': {'00': 'OFF'}, '0807': {'00': 'OFF'},
                '0809': {'0': 'MIC', '1': 'REAR'},
                '0810': {'0': 'DAKY', '1': 'RTS', '2': 'DTR'},
                '0812': {'0': 'USB', '1': 'LSB'}},
            'raw_reads': (
                ('NA0;', 'NARROW', {'0': 'off', '1': 'ON -- DATA caps at 500 Hz'}),
                ('GT0;', 'AGC', {'0': 'OFF', '1': 'FAST', '2': 'MID', '3': 'SLOW',
                                 '4': 'AUTO-FAST', '5': 'AUTO-MID', '6': 'AUTO-SLOW'}),
                ('SH0;', 'IF WIDTH', {'117': '3000 Hz, the widest DATA offers',
                                      '116': '2400 Hz', '115': '2000 Hz',
                                      '114': '1700 Hz', '113': '1400 Hz',
                                      '112': '1200 Hz', '111': '800 Hz', '110': '500 Hz'}),
                ('IS0;', 'IF SHIFT', {}),
                ('CO00;', 'CONTOUR', {'0': 'off', '1': 'ON'}),
                ('NR0;', 'DNR', {'0': 'off', '1': 'ON'}),
                ('BC0;', 'AUTO NOTCH', {'0': 'off', '1': 'ON'}),
                ('NB0;', 'NOISE BLANKER', {'0': 'off', '1': 'ON'}),
                ('RA0;', 'ATT', {'0': 'off', '1': 'ON (12 dB)'}),
                ('PA0;', 'IPO', {'0': 'IPO, preamp bypassed', '1': 'AMP'}),
                ('RG0;', 'RF GAIN', {'030': 'maximum'})),
            'mode_codes': {'1': 'LSB', '2': 'USB', '3': 'CW-U', '4': 'FM', '5': 'AM',
                           '6': 'RTTY-L', '7': 'CW-L', '8': 'DATA-L', '9': 'RTTY-U',
                           'A': 'DATA-FM', 'B': 'FM-N', 'C': 'DATA-U', 'D': 'AM-N',
                           'E': 'PSK', 'F': 'DATA-FM-N'}},
        data_widths=(50, 100, 150, 200, 250, 300, 350, 400, 450,
                     500, 800, 1200, 1400, 1700, 2000, 2400, 3000),
        rf_gain_raw_max=30, physical_data_passband_verified=False,
        cat='yaesu', id='0650', name='Yaesu FT-891', cat_rts=False, cat_dtr=False,
        min_watts=5, max_watts=100, width_prefix='1', width_modes=('8', 'C'))
    args = rigs.flight_defaults(SimpleNamespace(
        rig=None, serial=None, ptt_port=None, audio_in=None, audio_out=None))
    assert vars(args) == dict(rig='ft891', serial=None, ptt_port=None,
                             audio_in='USB Audio Device', audio_out='USB Audio Device')
    assert asdict(config.Audio()) == dict(input='', output='', rate=48000,
        blocksize=128, latency='low', input_gain=.04, tx_drive=.6,
        tx_latency_ms=None, capture_dir='')
    assert config.GAIN_RANGE == (.005, .10)
    assert config.tx_latency_ms() == 20
    assert config.tx_latency_ms(0) == 0


@pytest.mark.parametrize('path,geometry,start,notice,lead', [
    ('generic', 960, 58732, 1652, 2016),
    ('flown', 0, 57772, 2612, 2976),
    ('p2-mail', 0, 57772, 2612, 2976),
    ('p4-bench', 816, 58588, 1796, 2160),
])
@pytest.mark.parametrize('source', ['profile', 'toml-zero', 'operator-zero'])
def test_sample_placement_and_zero_precedence(monkeypatch, tmp_path, path,
                                             geometry, start, notice, lead, source):
    chain = tmp_path / 'chain.toml'
    chain.write_text('schema=1\n[station]\ncontrol="local"\nregulatory="unregulated"\n'
                     'because="offline baseline"\n[audio]\ntx_latency_ms=0\n')
    if source != 'profile':
        monkeypatch.setenv('HFMODEM_STATION', str(chain))
        geometry, start, notice, lead = 0, 58732, 1652, 2016
    plan = rigs.resolve_tx_timing(0 if source == 'operator-zero' else None,
                                 command_path=path)
    assert (plan.rig, plan.tx_latency_ms, plan.latency_samples,
            plan.geometry_samples, plan.latency_source, plan.calibration_status,
            plan.compatibility_identity) == (
        'ft891', 20. if source == 'profile' else 0.,
        960 if source == 'profile' else 0, geometry,
        {'profile': 'rig-profile', 'toml-zero': 'station-toml',
         'operator-zero': 'operator'}[source],
        {'profile': 'INFERRED', 'toml-zero': 'CHAIN OVERRIDE',
         'operator-zero': 'OPERATOR OVERRIDE'}[source], source == 'profile')
    live = onair._LiveInput.__new__(onair._LiveInput)
    live.fs, live._duplex, live._blk, live._lat = 48000, True, 128, 1268
    live.samples, live.pos, live.holdback = 12800, 12800, 660
    live.tx_latency_n, live.geometry_n = plan.latency_samples, geometry
    live._dac, live._tx_end_time = (0., 0), 0.
    live._tx_done = SimpleNamespace(clear=lambda: None, wait=lambda _: True)
    live.sample_now = lambda: 12800
    sleeps, buffers, keys = [], [], []
    live._sleep_until = lambda t: (sleeps.append(t), buffers.append(live._tx.tobytes()))
    # The watchdog is inert and the key callback is a list append.
    monkeypatch.setattr(onair.threading, 'Thread', lambda **kw: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(onair.time, 'monotonic', lambda: .25)
    samples = np.array([-.5, -.25, 0, .25, .5], dtype='>f8')
    assert live.transmit(samples, at=60000, settle=.04, key=keys.append) == (60000, 60005)
    assert live._tx_at == start
    assert live.corrected_emission_start == 60000 + geometry
    assert live.corrected_emission_end == 60005 + geometry
    assert live.key_notice == notice
    assert onair._prekey_lead(live, 1920) == lead
    assert sleeps == [pytest.approx(start / 48000 - .04), 0.]
    assert buffers == [bytes.fromhex('000000bf000080be000000000000803e0000003f')] * 2
    assert keys == [True, False]
    with pytest.raises(onair._MissedTxSlot):
        live.transmit(samples, at=12800, key=keys.append)
    assert keys == [True, False]


@pytest.mark.parametrize('gain', [None, .005, .04, .10])
def test_capture_samples_are_not_scaled_by_gain(gain):
    station = audio.StationAudio(input_device='fake-in', output_device='fake-out', gain=gain)
    blocks = []
    station.subscribe(SimpleNamespace(push=blocks.append))
    captured = np.array([[-.5], [-.25], [0], [.25], [.5]], dtype=np.float32)
    output = np.ones((5, 1), np.float32)
    station._callback(captured, output, 5, SimpleNamespace(
        inputBufferAdcTime=0., outputBufferDacTime=1268 / 48000), SimpleNamespace())
    assert station.failure is None
    assert len(blocks) == 1
    assert blocks[0].n0 == 0
    assert blocks[0].samples.tobytes() == bytes.fromhex('000000bf000080be000000000000803e0000003f')
    assert output.tobytes() == bytes(20)
    assert station.samples == 5


def test_normalized_sample_bytes_and_level_refusals():
    samples = np.array([-.5, -.25, 0, .25, .5], dtype=np.float32)
    assert levels.at_drive(samples).tobytes() == bytes.fromhex(
        '9a9919bf9a9999be000000009a99993e9a99193f')
    assert levels.at_drive(samples, 0).tobytes() == bytes.fromhex(
        '0000008000000080000000000000000000000000')
    assert levels.usable(np.zeros(100)) == (
        False, 'digital silence — the capture stalled or the device is wrong')
    assert levels.usable(np.ones(100)) == (
        False, '100.00% of samples against the rail at 0.0 dBFS — '
        'the input is hot; the measured working point is 0.04')
    assert (levels.RX_FLOOR_DBFS, levels.RX_HOT_RMS, levels.RX_SILENT_RMS,
            levels.PEAK_LIMIT_DBFS, levels.PEAK_AIM_DBFS) == (-34.9, .30, .001, -3., -6.)


def test_preflight_and_cycle_budget_inputs_and_decisions():
    assert onair.TR_SWITCH_S == preflight.TR_RECOVERY_S == .055
    assert preflight.TR_RECOVERY_SLACK_S == .015
    assert onair._budget(1.25, .21, .04) == (True,
        'cycle budget: settle 0.040 + packet 0.960 = 1.000 keyed of 1.250 s, '
        'leaving 250 ms clear; holds an answer whole for d 55-130 ms, of which '
        '55-107 can be read, against a measured peer at 87-134 (median 96); '
        'TX offset 0.210 assumes d 80 ms')
    assert onair._budget(1.25, .21, .10) == (False,
        'cycle budget: settle 0.100 + packet 0.960 = 1.060 keyed of 1.250 s, '
        'leaving 190 ms clear; holds an answer whole for d 55-70 ms, NONE of '
        'which is long enough to read, against a measured peer at 87-134 (median 96); '
        'TX offset 0.210 assumes d 80 ms -- OUTSIDE that band, so the first answer '
        'is still running when we key until the measured d replaces it -- SHORT '
        'OF THE MEDIAN ANSWER: this rig cannot read a peer at 96 ms, and a station '
        'that answers reads as silence')
