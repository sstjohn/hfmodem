# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, p3frame, p3rx, placement, rxfront


@pytest.mark.parametrize('hold', ['held', 'once'])
def test_maxdown_hold_accepts_supported_policies(hold):
    assert arq.ArqConfig(p3_maxdown_hold=hold).p3_maxdown_hold == hold


@pytest.mark.parametrize('hold', ['', 'hold', 'ONCE', None, True, 4])
def test_maxdown_hold_rejects_unknown_policies(hold):
    with pytest.raises(ValueError, match='p3_maxdown_hold'):
        arq.ArqConfig(p3_maxdown_hold=hold)


@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('long_cycle', [False, True])
@pytest.mark.parametrize('swapped', [False, True])
def test_narrow_header_preserves_rendered_geometry(level, long_cycle, swapped):
    audio = np.pad(placement.link_packet(
        level, b'header', 0x03, long_cycle=long_cycle, swapped=swapped), (4800, 4800))
    at = (4800 + (placement.protocol_config().pulse().size - 1) // 2
          + p3frame.DATA_OFFSET * rxfront.SPS)
    sync = rxfront.SyncedRx()
    before = dict(vars(sync))
    header = sync.narrow_header_at(audio[:at + rxfront.SPS], at, level)
    assert header is not None
    assert level in header.levels
    assert header.long_cycle == long_cycle
    assert header.swapped == swapped
    assert vars(sync) == before


@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('fit,levels', [
    (p3rx.VH_FIT - .001, (1, 2)),
    (p3rx.VH_FIT, ()),
    (1., (3,)),
])
def test_narrow_header_rejects_unreliable_or_other_level(monkeypatch, level, fit, levels):
    sync = rxfront.SyncedRx()
    monkeypatch.setattr(sync, '_narrow_header_at',
                        lambda audio, at: SimpleNamespace(fit=fit, levels=levels))
    assert sync.narrow_header_at(np.zeros(0), 0, level) is None
