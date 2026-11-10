# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Station recording policy is independent of audio destinations."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import tomllib

import pytest
from hfhost.config import ConfigError

from hfmodem.core import config


def station():
    return tomllib.loads((Path(__file__).resolve().parents[5] /
                          'examples/station.toml').read_text())


def test_recording_defaults_are_frozen():
    recording = config.Recording()
    assert recording.policy == 'everything'
    with pytest.raises(FrozenInstanceError):
        recording.policy = 'off'


@pytest.mark.parametrize('table', [None, {}, {'policy': 'everything'},
                                 {'policy': 'off'}, {'policy': 'contacts'}])
def test_recording_table_parses(table, tmp_path):
    raw = station()
    if table is not None:
        raw['recording'] = table
    cfg = config.parse(raw, environ={})
    assert cfg.recording.policy == (table or {}).get('policy', 'everything')
    path = tmp_path / 'station.toml'
    text = (Path(__file__).resolve().parents[5] / 'examples/station.toml').read_text()
    if table is not None:
        text += '\n[recording]\n' + (f'policy = "{table["policy"]}"\n' if table else '')
    path.write_text(text)
    assert config.load(path, environ={}).recording == cfg.recording


@pytest.mark.parametrize('value', ['all', 'OFF', '', 'contact', 1, True, [], {}])
def test_recording_rejects_unknown_values(value):
    raw = station()
    raw['recording'] = {}
    config.parse(raw, environ={})
    raw['recording'] = {'policy': value}
    with pytest.raises(ConfigError, match=r'\[recording\].*policy'):
        config.parse(raw, environ={})


@pytest.mark.parametrize('key', ['polciy', 'duration', 'codec', 'directory'])
def test_recording_rejects_unknown_keys(key):
    raw = station()
    raw['recording'] = {}
    config.parse(raw, environ={})
    raw['recording'] = {'policy': 'contacts', key: 'unused'}
    with pytest.raises(ConfigError, match=rf'\[recording\].*{key}'):
        config.parse(raw, environ={})


@pytest.mark.parametrize('destination', ['', 'captures'])
@pytest.mark.parametrize('policy', ['off', 'contacts', 'everything'])
def test_capture_destination_does_not_select_policy(destination, policy):
    raw = station()
    raw['audio']['capture_dir'] = destination
    raw['recording'] = {'policy': policy}
    cfg = config.parse(raw, environ={})
    assert cfg.recording.policy == policy
    assert cfg.audio.capture_dir == destination
