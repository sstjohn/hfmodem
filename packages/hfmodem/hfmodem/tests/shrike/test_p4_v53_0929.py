# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import hilbert

from hfmodem.shrike import onair, p4chirprx, p4oracle, p4words
from hfmodem.shrike.p4receiver import Kind, Observation, Registered
from hfmodem.shrike.p4observation import RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.corpora import n9vc_0802
from hfmodem.tests.shrike.test_p4turn import machine
from hfmodem.tests.shrike.test_p4_peer_ladder_0928 import receiver

ROOT = Path(__file__).resolve().parents[5]
FS = 48000


def control(at=FS, **kwargs):
    return Observation(Kind.SHORT, FS, 0, at, at, .758, 4., .5,
                       header=kwargs.pop('header', 'A'), value=53,
                       walsh_peak=64, placed=True, **kwargs)


@pytest.mark.parametrize('header', ['A', 'B', '?'])
@pytest.mark.parametrize('queued', [False, True])
def test_peer_53_gets_stock_u27_at_its_own_deadline(header, queued):
    m, arq = machine()
    m.state = State.IRS_ANSWER
    m.allow_breakin = True
    if queued:
        arq.on_host_data(b'mail remains queued')
    before = arq._buffer_raw
    obs = control(header=header)
    m.observe(RegisteredObservation(obs, Registered.NEW), now=FS+round(.81*FS))
    proposal = m.decide(FS+44160-m.lead)
    assert proposal is not None and proposal.word == 'U27'
    assert proposal.stream_sample == FS+44160
    assert proposal.peer_control == 53 and proposal.responds_to == FS
    assert arq._buffer_raw == before


def test_53_no_progress_counts_physical_copies_not_our_answers():
    m, _ = machine()
    m.state = State.IRS_ANSWER
    for k in range(8):
        at = FS+k*60000
        obs = control(at)
        m.observe(RegisteredObservation(replace(obs, provisional=True), Registered.NEW))
        assert len(m._unsupported_copies) == k
        m.observe(RegisteredObservation(obs, Registered.CONFIRMED))
        m.observe(RegisteredObservation(replace(obs,stream_sample=at+2), Registered.DUPLICATE))
        assert len(m._unsupported_copies) == k+1
        proposal = m.decide(at+44160-m.lead)
        assert proposal.word == 'U27'
        m.keyed()
    assert m.state == State.IRS_ANSWER
    assert m.reason is None


def test_latest_recorded_53_is_answered_without_reader_threshold_changes():
    path = ROOT/'captures/onair-0929-0126/stream.flac'
    if not path.exists():
        pytest.skip('local KB5LZK 0126 recording absent')
    audio = onair.session.load_wav(str(path), FS)
    at = 1906829
    rx = receiver(at, cfo=4.)
    end = at+round(.810*FS)
    records = rx.read_window(audio[at-2400:end],at-2400,whole=True)
    obs = next(r.observation for r in records if r.observation.value == 53)
    assert not obs.provisional and obs.walsh_peak == 64
    m, _ = machine()
    m.state = State.IRS_ANSWER
    for reg in records:
        m.observe(reg, now=end)
    proposal = m.decide(obs.stream_sample+44160-m.lead)
    assert proposal.word == 'U27'
    assert proposal.stream_sample == obs.stream_sample+44160
    assert proposal.stream_sample >= end+m.lead


def test_stock_53_u27_then_crc_valid_first_chirp():
    fs, pcm = n9vc_0802()
    base = 4.  # Source 1721.0 s, relative to the joined excerpt's 1717.0 s.
    audio = pcm[round(base*fs):round(10.3*fs)]/32768
    audio = (hilbert(audio)*np.exp(-2j*np.pi*400*np.arange(len(audio))/fs)).real
    field = p4oracle.robust_at(audio,fs,5.410-base,0.)
    assert field['p4_value'] == 53 and field['p4_walsh_peak'] == 64
    fraction, symbols, at = p4oracle.placed_despread(audio,fs,6.330-base,-8.,count=27)
    assert .19 < fraction < .25
    name, score = p4words.match(symbols)
    assert name == 'U27' and score > .94
    unit = symbols/np.maximum(abs(symbols),1e-30)
    runners = [p4words.correlate(word, unit)[0] for name,word in p4words.BANK.items() if name != 'U27']
    assert max(runners) < .58
    assert abs((base+at/fs)-5.410-.920) < .003
    for target in (6.030,6.630):
        _, null, _ = p4oracle.placed_despread(audio,fs,target-base,-8.,count=27)
        assert p4words.match(null)[0] != 'U27'
    result = p4chirprx.decode_packet(audio,fs=fs,start_sample=round((6.660-base)*fs))
    assert result is not None and result.marker == 8
    assert result.frame.status.byte == 0x61
    assert result.frame.data == b'[BPQ-6.0.25.1-B2FWIHJM'
    assert abs(6.660-(base+at/fs)-.330) < .003


@pytest.mark.parametrize('failure', ['provisional', 'unexpected', 'late'])
def test_unconfirmed_or_late_53_does_not_negotiate_or_spend_queued_mail(failure):
    m, arq = machine()
    m.state = State.IRS_ANSWER
    m.allow_breakin = True
    arq.on_host_data(b'pending mail')
    obs = control(**({failure: True} if failure != 'late' else {}))
    now = FS+44160-m.lead+(1 if failure == 'late' else 0)
    m.observe(RegisteredObservation(obs, Registered.NEW), now=now)
    assert m.decide(now) is None
    assert arq._buffer_raw == len(b'pending mail')


def test_known_u27_fixture_agrees_with_stock_answer_bank():
    fs, pcm = wavfile.read(Path(__file__).parent/'fixtures/p4-breakin/n9vc-u27.wav')
    (obs,) = p4oracle.word_scan(pcm.astype(float)/32768, fs)
    assert obs.word == 'U27' and obs.score > .98
