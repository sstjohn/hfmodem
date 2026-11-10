# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Construct the offline interop receive owner without a probe or devices."""
from contextlib import ExitStack
import importlib.util
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from . import corpora


@pytest.fixture
def xara(monkeypatch):
    directory = corpora.ROOT / 'tools' / 'interop' / 'xara'
    if not directory.is_dir():
        pytest.skip('interop harness is not included in this distribution')
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setenv('HFMODEM_INTEROP_ROOT', str(corpora.ROOT))
    for name in ('paths', 'channel'):
        spec = importlib.util.spec_from_file_location(name, directory / f'{name}.py')
        dependency = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, dependency)
        spec.loader.exec_module(dependency)
    spec = importlib.util.spec_from_file_location('xara_span_run', directory / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('snr,replay', [('clean', False), ('10', False), ('clean', True)])
def test_xara_receive_path_and_wrapper_share_indexed_context(xara, snr, replay):
    args = SimpleNamespace(condition=xara.Condition(snr=snr), registered=True, compression='OFF')
    with ExitStack() as resources:
        kestrel = xara.Kestrel(args, '500', lambda *args: None, resources, replay=replay)
        kestrel.runtime.step(0)
        rx, wire, hs = kestrel.rx, kestrel.wire, kestrel.modem.station
        context = kestrel.runtime.receive_context
        assert rx.hs is wire.hs is hs
        assert rx.receive_context is wire.receive_context is hs.receive_context is context
        assert hs.cq_detector is context.detector
        assert hs._tune_epoch == wire.tune_epoch == rx.tune_epoch == 0
        assert hs._decoder_epoch >= rx._decoder_epoch > 0
        delivered = []
        wire.cq_sink = delivered.append
        assert rx.cq_sink is wire.cq_sink
        rx.feed(10000, np.zeros(4800))
        assert wire.input_clock == rx.input_clock == 14800
        wire.next_rx_burst(.01, hs)
        if snr != 'clean':
            assert context.input_clock == 14800 and hs._input_end == 14800
        rx.burst_start = 12345
        assert wire.burst_start == 12345
        wire.retuned()
        assert hs._tune_epoch == wire.tune_epoch == context.ledger.tune_epoch == 1
