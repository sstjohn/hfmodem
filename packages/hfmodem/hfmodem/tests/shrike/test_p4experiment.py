# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An experimental response comparison never turns a signature into an ACK."""
import numpy as np
import pytest

from hfmodem.shrike import p4experiment, pactor1
from hfmodem.tests.shrike.test_p4sig import _synth_header


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
def test_experiment_retains_sample_timing(tmp_path, host):
    import json
    from hfmodem.core.audio import timestamp_limitation
    from hfmodem.shrike.p4probe import EntryProbe
    limitation = timestamp_limitation(host)
    probe = EntryProbe()
    probe.request(1000, b'', 0x61)
    probe.timing = (dict(timestamp_limitation=limitation, driver_offset_provenance='nominal',
                         timing_estimated=True, timing_basis='estimated sample timing; nominal driver offset')
                    if limitation else {})
    probe.emissions = [dict(audio_start=2000, audio_end=3000)]
    probe.windows = [dict(kind='post-chirp', start=3000, end=3480)]
    probe.received = [(0, 3000, np.zeros(480, np.float32))]
    result = p4experiment.write_report(probe, tmp_path)
    saved = json.loads((tmp_path / 'p4-mail-experiment.json').read_text())
    assert saved == result
    assert saved['grant_sample'] == 1000 and saved['observations'][0]['start_sample'] == 3000
    for row in [saved, *saved['emissions'], *saved['receive_windows'], *saved['observations']]:
        if limitation:
            assert row['timestamp_limitation'] == limitation
            assert row['timing_estimated'] is True
            assert 'nominal' in row['timing_basis']
        else:
            assert not {'timestamp_limitation', 'timing_estimated'} & row.keys()


def test_measured_p1_response():
    x = np.zeros(48000, np.float32)
    cs = pactor1.control_signal(pactor1.CS_ACK_A)
    x[12000:12000 + len(cs)] = cs
    got = p4experiment.observe(x, 96000)
    assert any(c['index'] == pactor1.CS_ACK_A for c in got['p1_controls'])
    assert got['p4_candidate'] is None


def test_measured_p4_header_candidate():
    x = np.zeros(48000, np.float32)
    header = _synth_header(14, 3, cfo=6)
    x[12000:12000 + len(header)] = header
    got = p4experiment.observe(x, 48000)
    assert got['p4_candidate'] is not None
    assert got['p4_candidate']['root'] == 14
    assert got['p4_candidate']['shift'] == 3


@pytest.mark.parametrize('keys,expected', [
    ([1, 1, 2], 'candidate-pattern-consistent-with-counter-sensitive-replies'),
    ([1, 2, 2], 'repeatability-prediction-not-observed'),
    ([1, 1, 1], 'counter-change-prediction-not-observed'),
])
def test_preregistered_counter_predictions(keys, expected):
    observations = [dict(p1_controls=[], p4_candidate=dict(root=k, shift=0, header_score=.5))
                    for k in keys]
    assert p4experiment.verdicts(observations, True)[1] == expected
    assert p4experiment.verdicts(observations, False) == ('inconclusive', 'inconclusive')


def test_persistent_p1_and_silence_have_different_outcomes():
    rows = [dict(p1_controls=[dict(index=4)], p4_candidate=None)] * 3
    assert p4experiment.verdicts(rows, True)[0].startswith('P1-replies-persist')
    assert p4experiment.verdicts([], False) == ('inconclusive', 'inconclusive')
