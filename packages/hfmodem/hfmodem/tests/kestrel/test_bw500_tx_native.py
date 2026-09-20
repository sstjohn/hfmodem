# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Base BW500 TX must reproduce held-out stock waveforms, not only self decode.

Native fixtures are excluded from publication and skipped visibly when absent.
Only delay and constant gain/phase per carrier are fitted, never pulse samples.
"""
from functools import cache
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags, hilbert

from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.tx import varahf500_tx as TX

FIXTURES=Path(__file__).with_name('fixtures')/'bw500-tx-native'

@cache
def metadata():
    path=FIXTURES/'provenance.json'
    if not path.exists():pytest.skip(f'native BW500 TX metadata absent: {path}')
    return json.loads(path.read_text())


def audio(name):
    path=FIXTURES/(name+'.wav')
    if not path.exists():pytest.skip(f'native BW500 TX recording absent: {path}')
    fs,x=wavfile.read(path)
    assert fs==TX.FS
    return x.astype(float)


@pytest.mark.parametrize('name',['greeting-heldout','fetch-heldout'])
def test_base_tx_reproduces_independent_stock_payload(name,monkeypatch):
    x=audio(name);frame=bytes.fromhex(metadata()[name]['frame_hex'])
    native=RX.decode_stream(x)
    assert native.complete and len(native.data_frames)==1
    assert native.data_frames[0].frame_bytes==frame
    rendered=TX.synth_burst(frame,onset=3000)
    own=RX.decode_stream(rendered)
    assert own.complete and own.data_frames[0].frame_bytes==frame
    parts=[]
    for sub in (0,1):
        pulses=dict(TX._PULSE)
        pulses[1-sub]=np.zeros_like(pulses[1-sub])
        with monkeypatch.context() as patch:
            patch.setattr(TX,'_PULSE',pulses)
            parts.append(TX.synth_burst(frame,onset=3000))
    np.testing.assert_allclose(parts[0]+parts[1],rendered,rtol=1e-14,atol=1e-14)
    analytic=[hilbert(p) for p in parts]
    cc=correlate(x,analytic[0],method='fft')
    approximate=int(correlation_lags(len(x),len(rendered))[np.argmax(abs(cc))])
    first,last=TX.FS//2,len(rendered)-TX.FS//2
    basis=np.column_stack([v for z in analytic for v in (z[first:last].real,z[first:last].imag)])
    normal=basis.T@basis
    best=None
    for lag in range(approximate-16,approximate+17):
        if first+lag<0 or last+lag>len(x):continue
        target=x[first+lag:last+lag]
        gain=np.linalg.solve(normal,basis.T@target)
        prediction=basis@gain
        residual=np.linalg.norm(target-prediction)/np.linalg.norm(target)
        if best is None or residual<best[0]:best=(residual,target,prediction)
    assert best is not None
    residual,target,prediction=best
    # The old pulse and simultaneous subbands leave roughly70% residual;
    # the corrected measured pulse is below0.3% on independent stock bytes.
    assert residual<.01,residual
    def crest(y):return 20*np.log10(np.max(abs(y))/np.sqrt(np.mean(y*y)))
    assert abs(crest(target)-crest(prediction))<.1


@pytest.mark.parametrize('ncol',[TX.C0+TX.NSYM,TX.C0+TX.NSYM+8])
def test_default_length_retains_the_upper_carrier_tail(ncol,monkeypatch):
    frame=TX.build_frame(bytes(range(43)),0x99)
    symbols=TX.symbols(TX.encode_coded(frame),ncol)
    # Current reference coverage happens to make terminal filler cells zero.
    # Force a live final upper-band cell so it exercises allocation/retention,
    # rather than passing vacuously because a skipped pulse carried no energy.
    symbols[1][-1]=1.+1j
    monkeypatch.setattr(TX,'symbols',lambda *a,**kw:symbols)
    original=TX.synth_burst(frame,onset=3000,ncol=ncol)
    padded=TX.synth_burst(frame,onset=3000,ncol=ncol,length=len(original)+TX.H)
    # Extending the container must not suddenly restore a skipped final pulse.
    np.testing.assert_array_equal(original,padded[:len(original)])
    np.testing.assert_array_equal(padded[len(original):],0.)
    assert np.count_nonzero(original[-TX.H:])>0
    assert not np.any(original[:3000-TX.H//2])
