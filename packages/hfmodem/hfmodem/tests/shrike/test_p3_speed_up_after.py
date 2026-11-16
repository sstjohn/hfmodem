# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace

import pytest

from hfmodem.shrike import arq, onair, spec


@pytest.fixture
def configured_host(tmp_path, monkeypatch):
    class Configured(Exception):
        pass

    found = []

    def capture(host):
        found.append(host)
        raise Configured

    parser = onair.build_parser()
    monkeypatch.setattr(onair, 'RadioTx', lambda *a, **kw: SimpleNamespace(
        attach=lambda host: None, prepare_p1_setup=lambda: None))
    monkeypatch.setattr(onair, '_SessionRx', capture)

    def configure(*flags):
        args = parser.parse_args([
            '--dxcall', 'N0CALL', '--rig', 'ft891', '--outdir', str(tmp_path / str(len(found))), *flags])
        with pytest.raises(Configured):
            onair.run(args)
        return found[-1]

    return configure


@pytest.mark.parametrize('threshold', [1, 9])
def test_modem_speed_up_after_reaches_arq_config(configured_host, threshold):
    host = configured_host('--p3-speed-up-after', str(threshold))
    host.protocol = spec.Protocol.PACTOR3
    assert host.arq.cfg.p3_speed_up_after == threshold
    assert [host.arq._gear_cs(1) for _ in range(threshold)] == (
        [arq.CS_ACK] * (threshold - 1) + [arq.CS_SPEED_UP])


def test_modem_speed_up_after_defaults_to_three(configured_host):
    host = configured_host()
    assert host.arq.cfg.p3_speed_up_after == 3


@pytest.mark.parametrize('threshold', [1, 9])
def test_modem_speed_up_after_leaves_p2_cadence_at_three(configured_host, threshold):
    host = configured_host('--p3-speed-up-after', str(threshold))
    for protocol, cadence in [(spec.Protocol.PACTOR2, 3),
                              (spec.Protocol.PACTOR3, threshold),
                              (spec.Protocol.PACTOR2, 3)]:
        host.protocol = protocol
        host.arq._clean_run = 0
        assert [host.arq._gear_cs(1) for _ in range(cadence)] == (
            [arq.CS_ACK] * (cadence - 1) + [arq.CS_SPEED_UP])


@pytest.mark.parametrize('value', [None, 'repeat', 'ack'])
def test_modem_repeated_cs4_reaches_arq_config(configured_host, value):
    flags = [] if value is None else ['--p3-repeated-cs4', value]
    host = configured_host(*flags)
    assert host.arq.cfg.p3_repeated_cs4 == (value or 'repeat')


@pytest.mark.parametrize('value', [None, 'parity', 'repeat'])
def test_modem_unread_after_gear_reaches_arq_config(configured_host, value):
    flags = [] if value is None else ['--p3-unread-after-gear', value]
    host = configured_host(*flags)
    assert host.arq.cfg.p3_unread_after_gear == (value or 'parity')


@pytest.mark.parametrize('value', [None, 'parity', 'repeat'])
def test_console_forwards_unread_after_gear_deviation(tmp_path, value, tools):
    import console
    from consolelib import pactor

    flags = [] if value is None else ['--p3-unread-after-gear', value]
    args = console.parser_for(pactor.PROFILE).parse_args([
        'fly', 'WS8EOC', '--sense', '--no-mail', *flags])
    words, adjustments = pactor.flags({}, pactor.LEVELS['current'], args, tmp_path, None)
    expected = value or 'parity'
    assert adjustments['p3_unread_after_gear'] == expected
    assert ('--p3-unread-after-gear' in words) == (expected != 'parity')
    parsed = onair.build_parser().parse_args(['--dxcall', 'WS8EOC', *words])
    assert parsed.p3_unread_after_gear == expected
