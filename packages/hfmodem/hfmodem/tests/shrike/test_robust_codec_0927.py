"""Recorded robust-entry waveform and codec checks, independent of the bridge."""
import numpy as np
from hfmodem.shrike import compress, p3acquire, pactor2, spec
from hfmodem.tests.shrike.test_robust_answer_0923 import _k4mck

FS = 48000
SLOT = 60000
FIELD = bytes.fromhex('0f0f2a3ab44512320f8f87c7c3e31958eb')


def test_pmc_recorded_callsign():
    assert compress.pmc('k4mck\r') == FIELD[:8]
    for call in ('w9ssj', 'ai4y', 've3kpg', 'kb5lzk', 'w9ssj-10'):
        coded = compress.pmc(call + '\r')
        assert compress.decompress(coded, 6, spec.Protocol.PACTOR2) == (call + '\r').encode()






def test_recorded_waveform_matches_only_swapped():
    from scipy.signal import butter, sosfiltfilt
    x = _k4mck()[2*SLOT:3*SLOT]
    sos = butter(4,85,fs=FS,output='sos')
    def lanes(x):
        t = np.arange(len(x))/FS
        return [sosfiltfilt(sos,x*np.exp(-2j*np.pi*f*t)) for f in (1400,1600)]
    recorded = lanes(x)
    for swapped in (False,True):
        raw = pactor2.data_burst(FIELD,pactor2.PATHS[1],swapped=swapped)
        ours = lanes(np.pad(raw,(1000,1000)))
        lo,hi = 2000,len(raw)-2000
        peaks = []
        for at in range(-48,121):
            pairs = [(a[1000+lo:1000+hi],b[at+lo:at+hi]) for a,b in zip(ours,recorded)]
            score = sum(abs(np.vdot(a,b)) for a,b in pairs)/np.sqrt(
                sum(np.vdot(a,a).real for a,b in pairs)*sum(np.vdot(b,b).real for a,b in pairs))
            peaks.append((score,at/FS))
        score,at = max(peaks)
        if swapped:
            assert score >= .8 and abs(at-.0008) < .001
        else:
            assert score < .3



def test_fsk_entry_three_at_190_hz():
    from hfmodem.shrike import p1rx
    from hfmodem.tests.shrike.test_robust_answer_0923 import _burst, _window
    wave = _window(_burst(spec.CONTROL_SIGNALS[3]), at=.05, size=.30, noise=0)
    wave = p3acquire.compensate(wave,-190)
    got = p1rx.acquire_robust_answer_search(wave,.02,.10)
    assert got is not None and got.entry == 3 and got.errors == 0
    assert abs(got.offset_hz-190) < 5
