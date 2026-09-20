# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Final ACK tails from two independent stock-VARA BW2750 sessions.

Original recordings and recorder-clock alignments, not synthesized test input.
The two caller fixtures were followed by stock responder turn releases. The
two responder fixtures are connected ACKs; the archive audit also measured the
same responder tails at final-DATA ACKs in both sessions.
"""
import numpy as np
import pytest
import json
from functools import cache
from pathlib import Path
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_vara_giveup import _connected

FIXTURES = Path(__file__).parent / 'fixtures/bw2750-controls'
CASE_FILES = [f'20260904-{stamp}-continue-2750-{run}-{role}.wav'
              for stamp, run in [('031917', 1), ('032037', 2)]
              for role in ['A', 'B']]


@cache
def cases():
    path = FIXTURES / 'provenance.json'
    if not path.exists():
        pytest.skip(f'stock BW2750 control metadata absent: {path}')
    rows = {row['file']: row for row in json.loads(path.read_text())}
    assert set(rows) == set(CASE_FILES)
    return rows


def pairs(samples):
    # Rectangular windows resolve the responder's adjacent63/65 carriers;
    # Hann skirts overlap and can incorrectly select the empty bin64.
    rows = []
    for i in range(11):
        at = MK._WOFF + i * MK.HOP
        mag = np.abs(np.fft.rfft(samples[at:at + MK.NFFT]))
        rows.append(tuple(sorted((np.argsort(mag[22:106])[-2:] + 22).tolist())))
    return tuple(rows)


@pytest.mark.parametrize('name', CASE_FILES)
def test_measured_control_tails_match_stock_cables(name):
    case = cases()[name]
    path = FIXTURES / case['file']
    if not path.exists():
        pytest.skip(f'stock BW2750 control recording absent: {path}')
    role = 0 if case['station'] == 'A' else 1
    at = case['clip_lock_sample']
    rate, audio = wavfile.read(path, mmap=True)
    assert rate == MK.FS
    measured = pairs(np.asarray(audio[at:at + 12 * MK.HOP], dtype=float))
    ours = VF.control_bursts('W9SSJ', '2750')
    assert measured == ours[role]
    assert all(a != b for a, b in zip(measured[4:], ours[1 - role][4:]))
    assert all(a != b for a, b in zip(measured[4:],
                                     VF.control_bursts('W9SSJ', '2300')[role][4:]))


@pytest.mark.parametrize('role,index', [('initiator', 0), ('responder', 1)])
def test_transmitter_keys_the_measured_tail_for_its_role(role, index):
    hs, io = _connected()
    hs.bw, hs.role = '2750', role
    assert hs._tx_control_burst()
    assert pairs(io.sent[-1]) == VF.control_bursts('W9SSJ', '2750')[index]


def test_unmeasured_caller_does_not_claim_a_measured_tail():
    assert VF.control_bursts('N0XYZ', '2750') is None
    assert VF.control_bursts('w9ssj', '2750') == VF.control_bursts('W9SSJ', '2750')
