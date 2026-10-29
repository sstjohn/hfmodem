# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Idle replies remain visible on CF after a local move, without driving ARQ."""
import numpy as np
import pytest
import hashlib
import json
import wave
from pathlib import Path
from hfmodem.contact import passive as P
from hfmodem.contact.passive import IdleReplies
from hfmodem.contact import slots as S
from hfmodem.kestrel.vara import vara_frames as VF, vara_mfsk as MK


@pytest.mark.parametrize('center,expected', [(2250, ['W9SSJ']), (1500, [])])
def test_recorded_k9aho_idle_is_on_old_cf_after_local_move(center, expected):
    p = Path(__file__).parent/'fixtures'
    audio = p/'k9aho-1025-cf-idle.wav'
    provenance = json.loads((p/'k9aho-1025-cf-idle.json').read_text())
    assert hashlib.sha256(audio.read_bytes()).hexdigest() == provenance['wave_sha256']
    with wave.open(str(audio), 'rb') as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1,2,S.FS)
        x = np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
    reader, mixer, found = IdleReplies(), S.Mixer(center), []
    # The cut ends 0.7 s after the reply; the next search is up to a second on.
    x = np.r_[x, np.zeros(S.FS)]
    for at in range(0,len(x),4800):
        found += reader.feed(mixer(x[at:at+4800]), ['W9SSJ'])
    assert found == expected


@pytest.mark.parametrize('center,expected', [(2250, ['W9SSJ']), (1500, [])])
def test_recorded_k5cg_two_tone_frame1_is_visible_on_old_cf(center, expected):
    p = Path(__file__).parent/'fixtures'
    f = p/'k5cg-1025-cf-frame1.wav'
    provenance = json.loads((p/'k5cg-1025-cf-frame1.json').read_text())
    assert hashlib.sha256(f.read_bytes()).hexdigest() == provenance['wave_sha256']
    with wave.open(str(f), 'rb') as w:
        x = np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
    reader, mixer, found = IdleReplies(), S.Mixer(center), []
    x = np.r_[x, np.zeros(S.FS)]
    for at in range(0, len(x), 4800):
        found += reader.feed(mixer(x[at:at+4800]), ['W9SSJ'])
    assert found == expected


def idle(call='W9SSJ'):
    return MK.synth_burst(call, VF.for_bw(VF.SESSION_TURN_IDLE_RESPONDER, '500'))


@pytest.mark.parametrize('block', [960, 4800, 24000])
def test_complete_idle_is_reported_once_and_repeats_are_not_hidden(block):
    reader = IdleReplies()
    burst = idle()
    x = np.r_[np.zeros(4800), burst, np.zeros(2*S.FS), burst, np.zeros(S.FS)]
    found = []
    for at in range(0, len(x), block):
        found += reader.feed(x[at:at+block], ['W9SSJ'])
    assert found == ['W9SSJ', 'W9SSJ']
    assert len(reader.raw) <= P.IDLE_WINDOW


@pytest.mark.parametrize('negative', ['unknown-key', 'truncated', 'noise', 'other-control'])
def test_idle_requires_the_supplied_key_and_complete_matching_payload(negative):
    burst = idle('K9AHO' if negative == 'unknown-key' else 'W9SSJ')
    if negative == 'truncated':
        burst = burst[:len(burst)//2]
    elif negative == 'noise':
        burst = np.random.default_rng(5).normal(0, .1, 2*S.FS)
    elif negative == 'other-control':
        burst = MK.synth_burst('W9SSJ', VF.for_bw(VF.SESSION_TURN_REQUEST_RESPONDER, '500'))
    x = np.r_[np.zeros(4800), burst, np.zeros(S.FS)]
    reader, found = IdleReplies(), []
    for at in range(0, len(x), 24000):
        found += reader.feed(x[at:at+24000], ['W9SSJ'])
    assert not found


def test_cf_idle_after_retune_is_a_passive_control_not_chat_or_sender_identity():
    cf = 7105000
    m = S.SlotMonitor(cf, cf, mycall='W9SSJ')
    m.retune(cf-750)
    # Moving down puts the unchanged remote CF at 2250 Hz audio.
    burst = idle()
    # Translate the synthetic 1500 Hz waveform to the old channel.
    from scipy.signal import hilbert
    high = np.real(hilbert(burst)*np.exp(2j*np.pi*750*np.arange(len(burst))/S.FS))
    x = np.r_[np.zeros(4800), high, np.zeros(S.FS)]
    found = []
    for at in range(0, len(x), 4800):
        found += m.feed(x[at:at+4800])
    controls = [r[2] for r in found if r[0] == 'qso' and r[2].kind == 'control']
    assert len(controls) == 1
    assert controls[0].slot is None and not controls[0].station
    assert controls[0].text == 'idle reply addressed to W9SSJ'
    activity = next(a for a in m.table() if a.slot is None)
    assert activity.state == 'busy' and not activity.calls


@pytest.mark.parametrize('lead', range(0, S.FS + 1, S.FS // 8))
@pytest.mark.parametrize('block', [4800, S.BLOCK])
def test_an_idle_is_caught_wherever_it_falls_between_scans(lead, block):
    noise = np.random.default_rng(lead).normal(0, .02, 3 * S.FS)
    x = np.r_[noise[:S.FS + lead], idle(), noise[S.FS + lead:]]
    reader, found = IdleReplies(), []
    for at in range(0, len(x), block):
        found += reader.feed(x[at:at + block], ['W9SSJ', 'K9AHO'])
    assert found == ['W9SSJ']


def test_the_idle_search_runs_about_once_a_second_of_audio(monkeypatch):
    from hfmodem.kestrel.vara import vara_arq as VA
    scans = []
    track = VA._top3_track
    monkeypatch.setattr(VA, '_top3_track', lambda *a, **k: scans.append(1) or track(*a, **k))
    reader, x = IdleReplies(), np.random.default_rng(3).normal(0, .02, 20 * S.FS)
    for at in range(0, len(x), 4800):
        reader.feed(x[at:at + 4800], ['W9SSJ'])
    assert len(scans) <= 20
