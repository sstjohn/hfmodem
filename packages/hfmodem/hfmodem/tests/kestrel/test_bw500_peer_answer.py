# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Recorded BW500 replies select robust setup speeds."""
import numpy as np
import pytest
import sys
from dataclasses import replace
from pathlib import Path
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F

PAYLOADS = {
    23: [56, 50, 72, 56, 72, 70, 58, 58, 64, 70, 54, 76, 50, 52, 50],
    21: [54, 76, 66, 60, 60, 66, 66, 56, 60, 54, 56, 72, 52, 76, 54],
}


class IO:
    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def key(self, on):
        pass

    def tx(self, samples):
        pass


def awaiting(call='KB0WZI', bw='500'):
    station = A.VaraStationHandshake(['W9SSJ'], IO(), bw=bw)
    station.called, station.caller, station.role = call, 'W9SSJ', 'initiator'
    station.state, station.step = A.VaraState.CONNECTING, A._I_CR_SENT
    return station


def observe(station, payload):
    count = int(A._PAY_OFF[-1]) + 1
    track = np.zeros((count, 3), dtype=np.int32)
    clear = np.full((count, 2), 20.0)
    live = np.ones(count, dtype=bool)
    track[A._PAY_OFF, 0] = payload
    return station._peer_answer(track, clear, live, 1, 0)


@pytest.mark.parametrize('position', [21, 23])
def test_measured_narrow_reply_selects_setup_speed(position):
    station = awaiting()
    assert observe(station, PAYLOADS[position])
    assert [a.position for a in station.answers] == [position]
    assert station.answers[0].tones == 15
    assert station.step == (A._I_LINKSETUP_SENT if station.answers else A._I_CR_SENT)
    assert station.state == A.VaraState.CONNECTING
    assert not station.answer_retry  # Reset stock trials do not establish retry cadence.
    assert f'level {position - 20}' in station.io.logs[0]
    assert station._setup_level == position - 20
    assert 'could not read' not in station.io.logs[0]


@pytest.mark.parametrize('position', [21, 23])
@pytest.mark.parametrize('call,bw', [('W1AW', '500'), ('KB0WZI', '2300'),
                                   ('KB0WZI', '2750')])
def test_measured_reply_does_not_name_a_different_call_or_bandwidth(position, call, bw):
    station = awaiting(call, bw)
    assert not observe(station, PAYLOADS[position])
    assert not station.answers
    assert not station.answer_retry


def test_actual_narrow_offer_belongs_to_the_offer_path():
    station = awaiting()
    offer = F.connect_response('500')
    assert (offer.preadv - 1) // F.lattice_step(offer) == 24
    assert not observe(station, F.payload_bins('KB0WZI', offer))
    assert not station.answers


@pytest.mark.parametrize('position', [14, 15, 16])
def test_wide_grade_positions_do_not_define_tactical_reply_semantics(position):
    station = awaiting(bw='2750')
    kind = F.connect_response('2750')
    tones = F.payload_bins('KB0WZI', replace(kind, preadv=1 + 30 * position))
    assert not observe(station, tones)
    assert not station.answers
    assert not station.answer_retry


@pytest.mark.parametrize('position', [21, 23])
def test_a_partial_unconfirmed_reply_does_not_become_evidence(position):
    station = awaiting()
    assert not observe(station, PAYLOADS[position][:7] + [0] * 8)
    assert not station.answers


@pytest.mark.parametrize('name,start,stop,call,positions', [
    # Leave right context for the response-sized search span and its next
    # half-second batch. Ending at22 s truncates that scan despite retaining the
    # position21 payload itself (20.844 s); it is not a different FFT grid.
    ('20260910T055105Z-W9SSJ-KB0WZI.wav', 15, 23, 'KB0WZI', [23, 21]),
    ('20260910T051041Z-W9SSJ-N5TW.wav', 31, 34, 'N5TW', [23]),
])
def test_native_recording_selects_setup_speed(name, start, stop, call, positions):
    path = Path(__file__).resolve().parents[5] / 'logs' / 'onair' / name
    if not path.exists():
        pytest.skip('Local September 10 on-air recording unavailable')
    rate, samples = wavfile.read(path, mmap=True)
    assert rate == 48000
    audio = samples[int(start * rate):int(stop * rate)].astype(float)
    if np.issubdtype(samples.dtype, np.integer):
        audio /= np.iinfo(samples.dtype).max
    for target, expected in ((call, positions), ('W1AW', [])):
        station = awaiting(target)
        for i in range(0, len(audio), 4800):
            station.on_rx_stream(audio[i:i + 4800])
        assert [a.position for a in station.answers] == expected
        assert station.step == (A._I_LINKSETUP_SENT if station.answers else A._I_CR_SENT)
        assert not station.answer_retry


@pytest.mark.parametrize('position,damage', [(23, 14), (21, 16)])
def test_independent_stock_responses_select_setup_speed(position, damage):
    # Actual stock output, after 14 and16 corrupted payload tones respectively.
    # Each trial was ABORT/rearmed; it supplies no in-session retry permission.
    path = (Path(__file__).resolve().parent / 'fixtures' / 'bw500-damaged-request'
            / f'position{position}-corrupt{damage}.wav')
    if not path.exists():
        pytest.skip('Stock damaged-request recording unavailable (fixtures/bw500-damaged-request)')
    rate, samples = wavfile.read(path, mmap=True)
    assert rate == 48000
    audio = samples.astype(float)
    if np.issubdtype(samples.dtype, np.integer):
        audio /= np.iinfo(samples.dtype).max
    for target, expected in (('W1AW', [position]), ('W9SSJ', [])):
        station = awaiting(target)
        for i in range(0, len(audio), 4800):
            station.on_rx_stream(audio[i:i + 4800])
        assert [answer.position for answer in station.answers] == expected
        assert station.step == (A._I_LINKSETUP_SENT if station.answers else A._I_CR_SENT)
        assert station.state == A.VaraState.CONNECTING
        assert not station.answer_retry


def test_attempt_summary_reports_only_measured_narrow_semantics(monkeypatch, capsys):
    from hfmodem.tests.kestrel import corpora
    driver = corpora.harness('kestrel_connect')

    def connect(*args, hs=None, **kwargs):
        hs.answers = [A.PeerAnswer(1.0, 23, 0, 15, 15, ())]
        return False

    monkeypatch.setattr(driver, 'connect', connect)
    monkeypatch.setattr(sys, 'argv', ['kestrel_connect.py', '--gateway', 'KB0WZI',
        '--mycall', 'W9SSJ', '--bw', '500', '--dry-run', '--listen-first', '0',
        '--timeout', '1', '--no-record'])
    assert driver.main() == 1
    output = capsys.readouterr().out
    assert 'BW500 connect offers (frames 23)' in output
    assert 'setup was not confirmed' in output
    assert 'none was a connect offer' not in output
    assert 'could not read' not in output
