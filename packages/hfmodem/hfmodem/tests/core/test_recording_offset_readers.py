# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Absolute-offset segment names and legacy recording families."""
import json
import wave

import numpy as np
import pytest

from hfhost import recordings
from hfmodem.core import wav
from hfmodem.station.contactlog import cut_audio


def family(tmp_path, offsets, *, legacy=False):
    paths = []
    for i, first in enumerate(offsets):
        suffix = f'20261101T00{i:02d}Z' if legacy else f'20261101T0000Z-s{first:020d}'
        path = tmp_path / f'receive-{suffix}.wav'
        with wave.open(str(path), 'wb') as out:
            out.setparams((1, 2, 8, 0, 'NONE', 'not compressed'))
            out.writeframes(np.arange(first, first + 480, dtype='<i2').tobytes())
        paths.append(path)
    base = tmp_path / 'receive.wav'
    base.with_suffix('.json').write_text(json.dumps({
        'segments': [p.name for p in paths], 'segment_samples': 480,
        'samplerate': 8, 'started_utc': '2026-11-01T00:00:00Z'}))
    return base, paths


@pytest.mark.parametrize('legacy', [False, True])
def test_read_contiguous_family_and_resolve_base(tmp_path, legacy):
    base, paths = family(tmp_path, [0, 480], legacy=legacy)
    assert wav.segments(base) == paths
    assert recordings.base(paths[1]) == base
    target = tmp_path / 'cut.wav'
    cut_audio(base, target, 478, 482)
    with wave.open(str(target)) as audio:
        assert np.frombuffer(audio.readframes(4), '<i2').tolist() == [478, 479, 480, 481]


def test_absolute_cutout_preserves_gap_and_nonzero_origin(tmp_path):
    base, paths = family(tmp_path, [1000, 1500])
    with pytest.raises(ValueError):
        wav.segments(base)
    target = tmp_path / 'cut.wav'
    with pytest.raises(ValueError):
        cut_audio(base, target, 1478, 1502)
    assert not target.exists()
    cut_audio(base, target, 1500, 1502)
    with wave.open(str(target)) as audio:
        assert np.frombuffer(audio.readframes(2), '<i2').tolist() == [1500, 1501]


def test_offset_family_refuses_reuse_without_index(tmp_path):
    base, paths = family(tmp_path, [1000])
    base.with_suffix('.json').unlink()
    with pytest.raises(FileExistsError):
        recordings.refuse_occupied(base)


@pytest.mark.parametrize('end,first', [(None, 0), (1480, 1000)])
def test_a_window_sidecar_places_the_window_on_its_stream_end(tmp_path, end, first):
    path = tmp_path / 'window.wav'
    with wave.open(str(path), 'wb') as out:
        out.setparams((1, 2, 8, 0, 'NONE', 'not compressed'))
        out.writeframes(np.zeros(480, '<i2').tobytes())
    path.with_suffix('.json').write_text(json.dumps({'samples': 480, 'end_stream_sample': end}))
    assert recordings.intervals(path) == [((first, first + 480), path)]
