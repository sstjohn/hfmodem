# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import sys
from importlib import import_module
from pathlib import Path

import pytest

from hfmodem.tests.kestrel import corpora


@pytest.mark.parametrize('cached', ['viewer', 'harness'])
def test_rehear_modules_keep_their_identity_with_either_cached_first(monkeypatch, cached):
    monkeypatch.syspath_prepend(str(corpora.TOOLS.parent))
    monkeypatch.syspath_prepend(str(corpora.TOOLS / 'logbook_view'))
    first = (import_module('tools.logbook_view.rehear_arms') if cached == 'viewer'
             else corpora.harness('rehear'))
    monkeypatch.setitem(sys.modules, 'rehear', first)

    harness = corpora.harness('rehear')
    viewer = import_module('tools.logbook_view.rehear_arms')

    assert corpora.harness('rehear') is harness
    assert Path(harness.__file__).resolve() == corpora.TOOLS / 'rehear' / '__init__.py'
    assert corpora.harness('rehear.sessionlog') is harness.sessionlog
    from tools.logbook_view.rehear_arms import plan
    assert plan is viewer.plan
    assert not hasattr(viewer, 'recordings')
    assert callable(harness.recordings)
