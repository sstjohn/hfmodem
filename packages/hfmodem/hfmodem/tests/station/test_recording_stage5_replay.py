# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Replay transports carry absolute selections without loading session hints."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.tests.station.recording_stage5_support import (
    RATE, START, STOP, entry, reader, save_selection, wave_file,
)


@pytest.mark.parametrize('gaps', [[], [[8008, 8016]]])
def test_wav_replay_frames_preserve_source_offsets_and_holes(tmp_path, gaps):
    from hfhost.audio import WavReplaySource, FS
    selected = entry(tmp_path, gaps=gaps)
    path = save_selection(tmp_path, selected)
    with WavReplaySource(path, realtime=False, selection=selected['recording']) as source:
        frames = list(source.frames())
    ratio = FS // RATE
    assert frames[0][0] == START * ratio
    assert sum(len(pcm) // 2 for _, pcm in frames) == (STOP - START - (8 if gaps else 0)) * ratio
    for first, pcm in frames:
        assert START * ratio <= first < STOP * ratio
        assert first + len(pcm) // 2 <= STOP * ratio
        assert not gaps or first + len(pcm) // 2 <= 8008 * ratio or first >= 8016 * ratio


def test_monitor_decode_passes_selection_to_transport_before_pcm_allocation(tmp_path, monkeypatch):
    module = reader('logbook_view.rehear_arms')
    from hfhost import audio
    selected = entry(tmp_path)
    path = save_selection(tmp_path, selected)
    def source(path, **kwargs):
        assert kwargs.get('selection') == selected['recording'], 'monitor read the entire source without its contact selection'
        return SimpleNamespace(frames=lambda: iter([]))
    monkeypatch.setattr(audio, 'WavReplaySource', source)
    monkeypatch.syspath_prepend(str(tmp_path))
    args = SimpleNamespace(wav=path, selection=selected['recording'], out=tmp_path,
        base=tmp_path, deep=False, runners=[], p4=False, grace=1)
    module.decode(args)


def test_station_file_replay_is_bounded_by_contact_reference(tmp_path, monkeypatch):
    from hfmodem.core import config
    from hfmodem.station import process
    selected = entry(tmp_path)
    path = save_selection(tmp_path, selected)
    wave_file(path, count=64)
    from hfmodem.tests.station.recording_stage5_support import write_json
    write_json(path.with_suffix('.json'), {'recording': selected['recording'],
        'start_stream_sample': START - 16, 'end_stream_sample': STOP + 16,
        'samplerate': RATE, 'source_id': 'source-rx', 'index_id': 'index-rx'})
    cfg = config.StationConfig(station=config.Station(mycall='N0CALL'), rig=config.Rig(cat='none'),
        audio=config.Audio(input='replay:' + str(path)), protocols={})
    station = process.Station(cfg)
    monkeypatch.setattr(station, 'validate_options', lambda: None)
    monkeypatch.setattr(cfg.__class__, 'require_audio', lambda *a, **k: None)
    class Replay:
        def __init__(self, pcm): self.pcm = pcm
        def open(self): pass
        def close(self): pass
    monkeypatch.setattr(process, 'ReplayAudio', Replay)
    audio = station.open_audio()
    try:
        assert len(audio.pcm) == 4 * process.CARD_RATE_HZ
    finally:
        audio.close()


@pytest.mark.parametrize('waveform', ['voice', 'cw'])
def test_voice_cw_heard_replay_respects_available_selected_spans(tmp_path, waveform):
    from hfmodem.contact.runtime import Session, RATE as output_rate
    selected = entry(tmp_path, gaps=[[8008, 8016]])
    owner = object.__new__(Session)
    owner.latest = {'audio': np.arange(32, dtype=np.float32) / 32768, 'rate': RATE,
                    'recording': selected['recording']}
    owner.playing = None
    owner.draft = None
    owner.waveform = SimpleNamespace(name=waveform)
    heard = []
    owner.play = lambda **kwargs: heard.append(owner.draft.copy())
    owner.replay_heard()
    assert sum(len(draft['audio']) for draft in heard) == 3 * output_rate
    assert len(heard) == 2, 'local replay must preserve the missing interval rather than play it as retained audio'
    assert [draft['source_start_sample'] for draft in heard] == [8000, 8016]
