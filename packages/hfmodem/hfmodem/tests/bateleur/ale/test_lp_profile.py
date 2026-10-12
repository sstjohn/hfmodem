# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import config
from hfmodem.station.link import AleLink
from hfmodem.tests.bateleur.ale.test_station import _cfg


def station(profile, dial, level=2):
    cfg = _cfg(channels=(dial,), lp_level=level, lp_key='0123456789abcd')
    cfg = replace(cfg, station=replace(cfg.station, regulatory=profile,
                  regulatory_settings={'because': 'bench'} if profile == 'unregulated'
                  else {'licence': 'extra'}))
    sent = []
    return SimpleNamespace(cfg=cfg, lanes={}, rig=SimpleNamespace(dial=lambda: dial, tune=lambda _: None), audio=None,
                           arbiter=SimpleNamespace(submit=lambda req: sent.append(req) or 123,
                                                   try_hold=lambda: nullcontext(True)),
                           sent=sent)


@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('dial', [1_850_000, 3_596_000, 5_357_000, 7_102_000, 10_145_500,
                                  14_250_000, 18_150_000, 21_300_000, 24_950_000, 28_500_000])
def test_unregulated_warns_at_startup_and_each_key(dial, level, capsys):
    st = station('unregulated', dial, level)
    link = AleLink(st)
    startup = capsys.readouterr().err
    assert 'warning' in startup and 'amateur' in startup
    assert 'your chosen unregulated profile applies a warning here' in startup
    for _ in range(2):
        assert link._send(np.zeros(120), link.waveform.rate) == 123
        warning = capsys.readouterr().err
        assert 'warning' in warning and 'amateur' in warning and str(dial) in warning
    assert len(st.sent) == 2


@pytest.mark.parametrize('dial,level', [(5_000_000, 2), (8_000_000, 2), (27_185_000, 2), (7_102_000, 0)])
def test_no_warning_off_band_or_with_protection_off(dial, level, capsys):
    st = station('unregulated', dial, level)
    link = AleLink(st)
    assert 'warning' not in capsys.readouterr().err
    assert link._send(np.zeros(120), link.waveform.rate) == 123
    assert 'warning' not in capsys.readouterr().err


@pytest.mark.parametrize('dial', [7_102_000, 8_000_000])
@pytest.mark.parametrize('level', [1, 2])
def test_part97_refuses_protection_in_config_and_lane(dial, level):
    st = station('part97', dial, level)
    with pytest.raises(config.ConfigError, match='chosen.*part97.*linking protection'):
        config._validate(st.cfg)
    with pytest.raises(config.ConfigError, match='chosen.*part97.*linking protection'):
        AleLink(st)
    assert not st.sent


def test_profile_is_checked_again_before_transmit(capsys):
    st = station('unregulated', 7_102_000)
    link = AleLink(st)
    capsys.readouterr()
    st.cfg = replace(st.cfg, station=replace(st.cfg.station, regulatory='part97',
                     regulatory_settings={'licence': 'extra'}))
    reason = link._send(np.zeros(120), link.waveform.rate)
    assert 'chosen' in reason and 'part97' in reason and 'linking protection' in reason
    assert not st.sent
