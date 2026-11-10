# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Contact recordings deeper than the Windows MAX_PATH limit."""
import os
import wave

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_support import RATE, bounded, module, push

MAX_PATH = 260


def _form(path, extended):
    return '\\\\?\\' + str(path) if extended else str(path)


@pytest.mark.parametrize('extended_path', [False, True])
@pytest.mark.parametrize('extended_root', [False, True])
def test_recorded_names_are_relative_posix_from_either_path_form(tmp_path, extended_path, extended_root):
    if os.name != 'nt' and (extended_path or extended_root):
        pytest.skip('extended-length paths exist only on Windows')
    relative = module()._relative
    path = _form(tmp_path / 'contact-c1' / 'lane' / 'receive.wav', extended_path)
    assert relative(path, _form(tmp_path, extended_root)) == 'contact-c1/lane/receive.wav'


def test_contact_recording_deeper_than_max_path_writes_and_reads_back(sessions, tmp_path):
    recording = module()
    deep = tmp_path.joinpath(*['station-data-' + 'd' * 48] * 5)
    rec, root = sessions(directory=deep)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 24)
    rec.begin('c1', {'card': 24})
    push(rec, lane, 24, 40)
    rec.end('c1', {'card': 40})
    push(rec, lane, 40, 40 + 6 * RATE)
    bounded(rec.close)

    index = recording.load_recording_index(root)
    ref = index['contacts']['c1']['lanes']['rx']
    names = [d['source'] if isinstance(d, dict) else d for d in ref['segments']]
    assert names and max(len(str(root / name)) for name in names) > MAX_PATH
    for name in [*names, index['lanes']['rx']['source']]:
        assert '\\' not in name and not name.startswith('/'), name
    chunks = []
    for name in names:
        with wav.opened_source(root / name) as stream, wave.open(stream) as src:
            chunks.append(np.frombuffer(src.readframes(src.getnframes()), '<i2'))
    np.testing.assert_array_equal(np.concatenate(chunks), np.arange(*ref['actual']) % 30000)
