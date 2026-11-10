# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
from pathlib import Path

import pytest

from hfmodem.shrike import p4frame, p4oracle, p4robustrx, session
from hfmodem.shrike.p4host import P4Host
from hfmodem.shrike.p4observation import Kind, Observation
from hfmodem.winlink import B2FSession

ROOT = Path(__file__).resolve().parents[5]
GREETING = ROOT/'working/p4-normal-budget-0929/greeting-fields.json'
if not (GREETING).exists():
    pytest.skip('working/p4-normal-budget-0929/greeting-fields.json is not in this tree', allow_module_level=True)
FIELDS = json.loads(GREETING.read_text())['frames']


def test_recorded_fields_have_distinct_counters_and_exact_repeated_prefix():
    fields = [bytes.fromhex(row['field_hex']) for row in FIELDS]
    assert [row['counter'] for row in FIELDS] == [0, 1, 2, 3]
    assert [len(field) for field in fields] == [31, 40, 40, 69]
    assert fields[1].startswith(fields[0])
    assert fields[0].endswith(b'W9SSJ has ')


def test_nonoverlapping_continuation_and_exact_repeats_keep_b2f_greeting_intact():
    adapter = P4Host()
    mail = B2FSession('W9SSJ', target='VE3KPG', password='offline-test-password')
    for index, row in enumerate(FIELDS):
        field = bytes.fromhex(row['field_hex'])
        if index == 1:
            field = b'103 daily'.ljust(40, b'\x1e')
        frame = p4frame.decode_field(p4frame.encode_field(field,
            p4frame.Status.from_byte(row['status'])), data_bytes=len(field))
        payload = adapter.accept(frame)
        assert adapter.accept(frame) is None
        mail.feed(payload)
    assert [line for line in mail.log_lines if line.startswith('banner:')] == [
        'banner: RMS Trimode 1.4.4.0',
        'banner: W9SSJ has 103 daily minutes remaining with VE3KPG (FN04VE)',
    ]
    assert mail.remote_sid == '[WL2K-5.0-B2FWIHJM$]'
    assert mail.challenge == '25286779'
    assert mail.sent_text and not mail.failure


def test_local_1024_capture_contains_the_restarted_greeting_prefix():
    path = ROOT/'captures/onair-0929-1024/stream.flac'
    if not path.exists():
        pytest.skip('local September29 1024 capture absent')
    recording = session.load_wav(str(path), 48000)
    for row in FIELDS:
        origin, at = row['origin'], row['nominal_sample']
        audio = recording[origin:row['end']]
        if row['kind'] == 'body':
            head = Observation(Kind.WORD, 48000, origin, at-origin, at, .240,
                               row['cfo_hz'], .264, word='Y27', score=.981)
            frame = p4oracle.body_after(audio, 48000, origin, head)
        else:
            coherence, symbols, _ = p4oracle.placed_despread(
                audio, 48000, (at-origin)/48000, row['cfo_hz'], count=373)
            assert coherence >= .25
            decoded = p4robustrx.decode_symbols(symbols)
            assert decoded is not None
            frame = decoded.frame
        assert frame is not None
        assert frame.status.byte == row['status']
        assert frame.data.hex() == row['field_hex']
