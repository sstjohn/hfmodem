# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import os
from types import SimpleNamespace as NS

import pytest

from hfmodem.core import claim, winclaim
from hfmodem.lanner import qso


def test_windows_standalone_terminal_has_no_optional_owner(monkeypatch):
    monkeypatch.setattr(qso, 'os', NS(name='nt'))
    monkeypatch.setattr(claim, 'os', NS(name='nt'))
    monkeypatch.setattr(winclaim, '_extras', {})
    monkeypatch.setattr(winclaim, 'take_handle', lambda name: None)
    assert qso.owner_fd() is None
    assert winclaim._extras == {}
    with pytest.raises(claim.ClaimError, match='missing launch extra owner'):
        claim.extra('owner')
    assert winclaim._extras == {}
    assert claim.extra('owner', optional=True) is None


def test_optional_owner_still_refuses_a_retired_or_reused_windows_handle(monkeypatch):
    original = object()
    monkeypatch.setattr(winclaim, '_extras', {'owner': (99, original)})
    monkeypatch.setattr(winclaim, 'entries', lambda: {99: original})
    assert winclaim.extra('owner', optional=True) == 99
    monkeypatch.setattr(winclaim, 'entries', lambda: {99: object()})
    with pytest.raises(claim.ClaimError, match='no longer live'):
        winclaim.extra('owner', optional=True)
    monkeypatch.setattr(winclaim, '_extras', {'owner': (None, None)})
    with pytest.raises(claim.ClaimError, match='no longer live'):
        winclaim.extra('owner', optional=True)


def test_posix_optional_extra_distinguishes_absence_from_malformed_metadata(monkeypatch):
    monkeypatch.setattr(claim, 'os', NS(name='posix', environ=os.environ))
    monkeypatch.delenv('HFMODEM_CLAIM_EXTRAS', raising=False)
    assert claim.extra('owner', optional=True) is None
    monkeypatch.setenv('HFMODEM_CLAIM_EXTRAS', json.dumps({'owner': 73}))
    assert claim.extra('owner', optional=True) == 73
    for malformed in ('[]', '{', '{"owner": "invalid"}'):
        monkeypatch.setenv('HFMODEM_CLAIM_EXTRAS', malformed)
        with pytest.raises(claim.ClaimError):
            claim.extra('owner', optional=True)
