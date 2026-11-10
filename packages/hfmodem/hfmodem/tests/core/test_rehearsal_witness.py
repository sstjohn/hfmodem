# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Synthetic observer-coverage checks; no station, device or timing run."""
import pytest

from hfmodem.tests.core.rehearsal_witness import _callback_coverage


def callbacks():
    result = []
    for number in range(350, 412):
        due = 10.0 + number * 128 / 48000
        result.append((number, due, due, due + .024, due + .0001, due + .00012, 10.0))
    return result


def covered(rows):
    return _callback_coverage(rows, 10.0, 128, 1152, 46848, 10.960, 10.965)


def test_contiguous_records_cover_keying_write_and_first_conversion():
    assert covered(callbacks())


@pytest.mark.parametrize('defect', ['key_before_window', 'missing_block', 'duplicate',
    'wrong_grid', 'wrong_epoch', 'reversed_delivery', 'missing_output',
    'late_delivery', 'conversion_after_window'])
def test_incomplete_or_inconsistent_records_cannot_be_conclusive(defect):
    rows = callbacks()
    if defect == 'key_before_window':
        rows = [row for row in rows if row[0] >= 365]
        # The original guard accepted this despite losing the keying interval.
        assert any(row[0] * 128 <= 46848 < (row[0] + 1) * 128 for row in rows)
    elif defect == 'missing_block':
        rows.pop(11)
    elif defect == 'duplicate':
        rows.insert(11, rows[10])
    elif defect == 'wrong_grid':
        row = list(rows[11]); row[2] += .001; rows[11] = tuple(row)
    elif defect == 'wrong_epoch':
        row = list(rows[11]); row[6] += .001; rows[11] = tuple(row)
    elif defect == 'reversed_delivery':
        row = list(rows[11]); row[4], row[5] = row[5], row[4]; rows[11] = tuple(row)
    elif defect == 'missing_output':
        rows = [row for row in rows if row[0] != 366]
    elif defect == 'late_delivery':
        rows = [(*row[:4], row[4] + .1, row[5] + .1, row[6]) for row in rows]
    else:
        rows = [row for row in rows if row[0] <= 367]
    assert not covered(rows)
