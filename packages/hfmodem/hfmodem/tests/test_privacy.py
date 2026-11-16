# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import re
from pathlib import Path

import pytest

from hfmodem.tests import d_evidence
from hfmodem.tests.evidence import require_input


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("relative, mids", [
    (
        'packages/hfmodem/hfmodem/tests/fixtures/receive-d-0930/ardop-day-03-w6ids-third.log',
        ('WKYC52O6ZESJ',),
    ),
    (
        'packages/hfmodem/hfmodem/tests/fixtures/receive-d-0930/ardop-day-08-w6ids.log',
        ('WTDQ6MCT16VW', 'D1AFM2QY1J0V', 'JNKZAKP5FK5B'),
    ),
    (
        'tools/consolelib/vara-fetch.log',
        ('WKYC52O6ZESJ', 'QKBU1DBHK3GY', 'PL9J4MQ2L3F8', 'V9QZ9YDXL5J7'),
    ),
    (
        'tools/logbook_view/test_link.py',
        ('WKYC52O6ZESJ', 'WTDQ6MCT16VW', '31WB58MN5Q1L'),
    ),
    (
        'packages/hfmodem/hfmodem/tests/shrike/test_empty_turnaround_0924.py',
        ('1L96LAEHDZLC',),
    ),
    (
        'packages/hfmodem/hfmodem/tests/winlink/test_session.py',
        ('WKYC52O6ZESJ', 'WTDQ6MCT16VW', 'D1AFM2QY1J0V', 'JNKZAKP5FK5B'),
    ),
])
def test_transcripts_use_placeholder_identifiers(relative, mids):
    text = require_input(ROOT / relative).read_text()
    assert "op@example.net" in text
    assert not re.search(r"[\w.+-]+@gmail\.co", text)
    for mid in mids:
        assert mid in text


@pytest.mark.parametrize("name", [
    "ardop-day-03-w6ids-third.log",
    "ardop-day-08-w6ids.log",
])
def test_scrubbed_transcripts_match_the_integrity_manifest(name):
    assert d_evidence.log(name).is_file()


@pytest.mark.parametrize("relative, mid", [
    ('packages/hfmodem/hfmodem/tests/besra/test_arq_unreadable_budget.py', 'WTDQ6MCT16VW'),
    ('packages/hfmodem/hfmodem/tests/besra/test_memory_arq.py', 'WKYC52O6ZESJ'),
    ('packages/hfmodem/hfmodem/tests/besra/test_arq_robustness.py', 'WTDQ6MCT16VW'),
    ('packages/hfmodem/hfmodem/tests/gates/test_stale_negatives.py', 'WKYC52O6ZESJ'),
    ('packages/hfmodem/hfmodem/besra/arq/session.py', 'WTDQ6MCT16VW'),
    ('tools/logbook_view/test_legacy.py', '31WB58MN5Q1L'),
])
def test_regression_sources_use_placeholder_identifiers(relative, mid):
    text = require_input(ROOT / relative).read_text()
    assert mid in text
    assert not re.search(r"[\w.+-]+@gmail\.co", text)
