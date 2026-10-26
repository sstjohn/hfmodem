# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KD2WNY calls W9SSJ at BW500 on the 40 m calling frequency, 2026-10-23 11:13Z.

The station answered the second and third keyings, never read the 4.25 s
link-setup that followed its answer, and the connect died at CANCELPENDING
twice. The recording is the rig's receive audio, cut around the call
(rf-corpus varac-weak-setup-1023; the fixture's provenance.json has the offsets).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core.wav import source
from hfmodem.tests import evidence
from hfmodem.kestrel.monitor import Segmenter
from hfmodem.kestrel.rx import bw500_locator as loc
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF

HERE = Path(__file__).with_name('fixtures') / 'varac-weak-setup-1023'
CUT = evidence.CORPUS / 'varac-weak-setup-1023'
pytestmark = pytest.mark.skipif(not CUT.is_dir(),
                                reason=f'recording not present ({evidence.CORPUS})')
FS = M.FS
CHUNK = 4800                       # the host's capture frame  [host.station_runtime.CHUNK]
CALLER, CALLED = 'KD2WNY', 'W9SSJ'
REQUEST = (40 * M.HOP + M.STRIDE) / FS       # the 41-symbol request on the air
# Seconds after 11:13:00Z, from the audit and the offline decode.
FIRST_REQUEST, RETRY, SETUP_END, SECOND_ATTEMPT = 5.87, 9.56, 16.23, 49.0


class Clock:
    t = 1000.

    @classmethod
    def monotonic(cls):
        return cls.t


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    Clock.t = 1000.
    monkeypatch.setattr(A, 'time', Clock)


class IO(A.VaraIO):
    def __init__(self):
        self.sent, self.logs, self.heard = [], [], 0
        self.connected_at = None

    def tx(self, samples):
        self.sent.append((self.heard, np.asarray(samples, float)))

    def connected(self, caller, called, bw):
        self.connected_at = self.heard

    def log(self, msg):
        self.logs.append(msg)


def answerer():
    io = IO()
    hs = A.VaraStationHandshake([CALLED], io, bw='500', high_speed=True,
                                compression='TEXT', chat=True)
    hs.listen(True)
    return hs, io


def replay(hs, io, x):
    """The host's order: every capture frame to the stream, then the gate's brackets."""
    gate = Segmenter()
    brackets = 0
    for at in range(0, len(x), CHUNK):
        chunk = x[at:at + CHUNK]
        io.heard = at + len(chunk)
        Clock.t += len(chunk) / FS
        hs.on_rx_stream(chunk)
        for _, burst in gate.push(chunk):
            brackets += 1
            hs.on_rx_audio(burst)
    return brackets


@pytest.fixture(scope='module')
def cut():
    prov = json.loads((HERE / 'provenance.json').read_text())['call-1113']
    rate, x = wavfile.read(source(CUT / prov['file']))
    assert rate == FS and x.dtype == np.int16
    t0 = 12 * 60 + 54.948                        # sample 0 is 11:12:54.948Z
    return x.astype(float) / 32768, lambda s: int(round((13 * 60 + s - t0) * FS))


def chat_offer(level):
    return M.synth_burst(CALLED, F.for_bw(F.chat_setup_response('500', level), '500'))


# -- the recording ----------------------------------------------------------

def test_the_first_keying_is_answered_and_graded(cut):
    """We answered the retry at 11:13:10.95; stock answers the first readable
    keying about 0.2 s after its last symbol. The 41-symbol request read 18
    clear tones of 21 and was refused for want of 24."""
    x, at = cut
    hs, io = answerer()
    replay(hs, io, x[:at(RETRY + 2.0)])
    assert len(io.sent) == 2
    first, retry = io.sent
    assert at(FIRST_REQUEST + REQUEST) <= first[0] <= at(FIRST_REQUEST + REQUEST + 0.2)
    assert at(RETRY) < retry[0] <= at(RETRY + 32 * M.HOP / FS + 0.2)
    # Stock offers the setup level the request earned: this one reached us at
    # -8 to -10 dB on the bench's measure, where stock's responder offers 1 and 2.
    assert np.array_equal(first[1], chat_offer(1))
    assert np.array_equal(retry[1], chat_offer(2))


def test_the_link_setup_is_read_on_the_stream(cut):
    """The setup stands -2.3 dB over the band in 1150-1850 Hz. The energy gate
    never opens on it; the locator reads it and the connect completes."""
    x, at = cut
    hs, io = answerer()
    brackets = replay(hs, io, x[:at(SETUP_END + 3.0)])
    assert brackets == 0
    assert hs.state == A.VaraState.CONNECTED
    assert (hs.caller, hs.called) == (CALLER, CALLED)
    assert hs.peer_high_speed and hs.session_compression == 'TEXT'
    assert at(SETUP_END) < io.connected_at <= at(SETUP_END + 0.5)
    assert any('caller is KD2WNY (high speed capable, ask 5)' in m for m in io.logs)


def test_the_frame_decodes_offline(cut):
    x, at = cut
    frames = [f for f in loc.decode(x[at(11.0):at(SETUP_END + 2.0)]).frames if f.crc_ok]
    assert len(frames) == 1 and F.is_link_setup(bytes(frames[0].frame_bytes))
    assert F.caller_from_link_setup(frames[0].frame_bytes) == CALLER
    assert F.link_setup_marks(frames[0].frame_bytes) == (True, 5)


# -- synthetic ladders ------------------------------------------------------

def flat_noise(n, seed, lo=300, hi=2700):
    """Unit-RMS noise, flat between ``lo`` and ``hi``."""
    z = np.fft.rfft(np.random.default_rng(seed).normal(size=n))
    f = np.fft.rfftfreq(n, 1 / FS)
    z[(f < lo) | (f >= hi)] = 0
    x = np.fft.irfft(z, n)
    return x / np.sqrt(np.mean(x * x))


def band_power(x, lo=1150, hi=1850):
    f = np.fft.rfftfreq(len(x), 1 / FS)
    z = np.fft.rfft(x)
    return float(2 * np.sum(abs(z[(f >= lo) & (f < hi)]) ** 2) / len(x) ** 2)


def in_noise(burst, snr_db, seed, lead=6 * FS, tail=3 * FS):
    """``burst`` over noise at ``snr_db`` of active-burst power in 1150-1850 Hz."""
    sig = np.concatenate([np.zeros(lead), burst, np.zeros(tail)])
    nn = flat_noise(len(sig), 21000 + seed)
    a, b = lead, lead + len(burst)
    nn *= np.sqrt(band_power(sig[a:b]) / band_power(nn[a:b])) * 10 ** (-snr_db / 20)
    return sig + nn


def awaiting_setup():
    hs, io = answerer()
    hs.role, hs.called, hs._chat_profile = 'responder', CALLED, True
    hs.state, hs.step = A.VaraState.CONNECTING, A._R_RESP_SENT
    return hs, io


SETUP = OF.link_setup_tx(CALLER, bw='500', level=4, ask=5, high_speed=True, compression='TEXT')


@pytest.mark.parametrize('seed', range(3))
def test_a_setup_at_minus_three_db_completes_the_connect(seed):
    """Before the stream reader the responder needed the gate to open, which took
    +20 dB in this band; KD2WNY's setup was at -2.3."""
    hs, io = awaiting_setup()
    assert replay(hs, io, in_noise(SETUP, -3.0, seed)) == 0
    assert hs.state == A.VaraState.CONNECTED and hs.caller == CALLER


@pytest.mark.parametrize('name, burst', [
    ('noise only', None),
    ('request', M.synth_burst(CALLED, F.for_bw(F.connect_request('500'), '500'))),
    ('data over L4', OF.data_over_tx(bytes(range(44)), level=4, bw='500')),
    ('data over L2', OF.data_over_tx(bytes(range(35)), level=2, bw='500')),
])
def test_nothing_but_a_link_setup_completes_the_connect(name, burst):
    """A repeated request draws the connect-response again; nothing else draws
    a keying, and only a link-setup completes the connect."""
    for snr in (20.0, 0.0):
        hs, io = awaiting_setup()
        x = (in_noise(burst, snr, 7) if burst is not None
             else 0.05 * flat_noise(13 * FS, 31007))
        replay(hs, io, x)
        assert hs.state == A.VaraState.CONNECTING, name
        assert not io.connected_at and (name == 'request' or not io.sent), name


def request_train(call, noise_rms, seed, gap=int(1.5 * FS)):
    """A stock caller's keyings -- the 41-symbol request, then the 32-symbol retry
    twice -- at the 2026-09-09 bench's drive, in its flat noise to 2900 Hz."""
    full = M.synth_burst(call, F.for_bw(F.connect_request('500'), '500'))
    retry = M.synth_burst(call, F.for_bw(F.connect_request_retry('500'), '500'))
    lead = 3 * FS
    x = np.concatenate([np.zeros(lead), full, np.zeros(gap), retry, np.zeros(gap), retry, np.zeros(2 * FS)])
    first_end = lead + len(full)
    return np.clip(x + noise_rms * flat_noise(len(x), 1000 + seed, 0, 2900), -0.98, 0.98), first_end


@pytest.mark.parametrize('noise_rms, level', [(0.40, 4), (0.90, 2), (1.20, 1)])
def test_the_first_keying_draws_the_offer_stock_makes(noise_rms, level):
    """Stock's responder offered 4, 2 and 1 at these noise levels over our
    request (2026-09-09 bench); the first keying is answered
    inside 0.2 s of its last symbol down to -10.6 dB."""
    for seed in range(3):
        x, first_end = request_train(CALLED, noise_rms, seed)
        hs, io = answerer()
        replay(hs, io, x)
        assert io.sent and first_end <= io.sent[0][0] <= first_end + 0.2 * FS
        assert np.array_equal(io.sent[0][1], chat_offer(level)), (noise_rms, seed)


@pytest.mark.parametrize('call', ['K1ABC', 'W9SSJ-4', CALLER])
def test_a_request_to_somebody_else_is_never_answered(call):
    for noise_rms in (0.0, 0.70, 1.20):
        x, _ = request_train(call, noise_rms, 5)
        hs, io = answerer()
        replay(hs, io, x)
        assert not io.sent and hs.state == A.VaraState.LISTENING, (call, noise_rms)


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_a_wide_responder_offers_the_base_level_it_can_read(bw):
    """A wide setup is read only at its base level, so a weak wide request is not
    graded down to a setup we could not take."""
    hs = A.VaraStationHandshake([CALLED], IO(), bw=bw, high_speed=True)
    hs._request_db = 0.0
    assert hs._offer_for() == 4
