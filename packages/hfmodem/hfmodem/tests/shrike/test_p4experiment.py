"""An experimental response comparison never turns a signature into an ACK."""
import numpy as np
import pytest

from hfmodem.shrike import p4experiment, pactor1
from hfmodem.tests.shrike.test_p4sig import _synth_header


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
