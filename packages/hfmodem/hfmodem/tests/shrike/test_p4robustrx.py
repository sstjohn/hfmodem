# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from pathlib import Path
import numpy as np
import pytest

from hfmodem.shrike import p4coding, p4decode, p4frame, p4oracle, p4robustrx, session

FX = Path(__file__).parent/'fixtures/p4-long-0929/symbols.npz'
ROOT = Path(__file__).resolve().parents[5]
SID = b'[BPQ-6.0.25.1-B2FWIHJM$]\r\rde N9VC>\r'+b'\x1e'*5
BANNER = b'RMS Trimode 1.4.4.0\r\nW9SSJ has 109 daily minutes remaining with VE3KP'


def recorded(name):
    with np.load(FX, allow_pickle=False) as fixture:
        return fixture[name]


@pytest.mark.parametrize('name,level,status,data,corrected', [
    ('n9vc_0',2,0x41,SID,0),('n9vc_1',2,0x41,SID,0),
    ('ve3kpg_0',3,0x61,BANNER,0),('ve3kpg_1',3,0x61,BANNER,7),
])
def test_recorded_long_field_has_unique_crc_and_measured_geometry(name,level,status,data,corrected):
    result = p4robustrx.decode_symbols(recorded(name))
    assert result is not None and result.speedlevel == level
    assert result.frame.status.byte == status and result.frame.data == data
    assert result.fields.shape == (696,)
    assert result.syndrome_score >= .84 and result.corrected_bits == corrected
    assert result.packet_symbol == result.field_start-20
    assert p4robustrx.syndrome(result.fields,5-level) < .80


def test_weak_copy_requires_channel_correction_not_raw_crc_acceptance():
    symbols = recorded('ve3kpg_1')
    assert p4robustrx.decode_symbols(symbols,iterations=0) is None
    assert p4robustrx.decode_symbols(symbols,iterations=1) is None
    assert p4robustrx.decode_symbols(symbols,iterations=2).frame.data == BANNER


def test_published_c1_and_c2_parities_match_strong_sl3_recording():
    result = p4robustrx.decode_symbols(recorded('ve3kpg_0'))
    bits = (result.fields < 0).astype(np.uint8)
    systematic = bits[:580]
    order = np.concatenate([np.arange(j,580,24) for j in range(24)])
    c1 = np.array(p4coding.rsc_encode(systematic,initial_state=(0,0,0)).parity)
    c2 = np.array(p4coding.rsc_encode(systematic[order],initial_state=(0,0,0)).parity)
    np.testing.assert_array_equal(c1[::10][:-2],bits[580:638][:-2])
    np.testing.assert_array_equal(c2[4::10][:-2],bits[638:][:-2])


@pytest.mark.parametrize('turn', [0,1,2,3])
@pytest.mark.parametrize('name',['ve3kpg_0','ve3kpg_1'])
def test_carrier_quarter_turn_ambiguity_still_has_one_crc(turn,name):
    symbols = recorded(name)
    symbols *= (1j**turn)**np.arange(len(symbols))
    result = p4robustrx.decode_symbols(symbols)
    assert result is not None and result.frame.data == BANNER


@pytest.mark.parametrize('kind',['truncated','silence','conjugate','wrong_walk','corrupt'])
def test_incomplete_or_wrong_field_is_not_payload(kind):
    symbols = recorded('ve3kpg_0')
    if kind == 'truncated':
        symbols = symbols[:350]
    elif kind == 'silence':
        symbols[:] = 0
    elif kind == 'conjugate':
        symbols = symbols.conjugate()
    elif kind == 'wrong_walk':
        symbols[21:369] = symbols[21:369][np.roll(np.arange(348),17)]
    else:
        symbols[80:220] *= np.exp(1j*np.random.default_rng(17).uniform(-np.pi,np.pi,140))
    assert p4robustrx.decode_symbols(symbols) is None


def test_random_coherent_quaternary_fields_do_not_pass_crc():
    rng = np.random.default_rng(929)
    for _ in range(64):
        symbols = np.exp(.5j*np.pi*np.cumsum(rng.integers(0,4,373)))
        assert p4robustrx.decode_symbols(symbols) is None


@pytest.mark.parametrize('at,offset',[(2688057,21),(2868057,22)])
def test_causal_0121_audio_reaches_decoder_before_long_deadline(at,offset):
    path = ROOT/'captures/onair-0929-0121/stream.flac'
    if not path.exists():
        pytest.skip('local VE3KPG0121 recording absent')
    audio = session.load_wav(str(path),48000)
    end = at+round(3.326*48000)
    fraction,symbols,first = p4oracle.placed_despread(audio[at-4800:end],48000,.1,1.54,count=373)
    result = p4robustrx.decode_symbols(symbols)
    assert fraction >= .25 and result.frame.data == BANNER
    assert result.field_start == offset
    placed = at-4800+first+round(result.packet_symbol*48000/112.5)
    assert abs(placed-at) < 64
    assert end < placed+round(3.4*48000)-2131


@pytest.mark.parametrize('kwargs',[dict(starts=[17]),dict(starts=[20,20]),dict(iterations=3)])
def test_invalid_decoder_dimensions(kwargs):
    with pytest.raises(ValueError):
        p4robustrx.decode_symbols(recorded('ve3kpg_0'),**kwargs)



def test_complemented_soft_candidate_preserves_erased_posterior_tie(monkeypatch):
    reference = p4robustrx.decode_symbols(recorded('ve3kpg_0'))
    wire = p4frame.encode_field(reference.frame.data, reference.frame.status)
    bits = np.r_[np.unpackbits(np.frombuffer(wire,np.uint8),bitorder='little'),[0,0,0,1]]
    zero = int(np.flatnonzero(bits[:576] == 0)[0])
    posterior = (2*bits.astype(float)-1)*5
    posterior[zero] = 0.
    calls = []
    def decode(*args, **kwargs):
        calls.append(1)
        return p4decode.TurboDecode(posterior,(posterior < 0).astype(np.uint8),())
    monkeypatch.setattr(p4decode,'turbo_decode',decode)
    symbols = recorded('ve3kpg_1')*(-1)**np.arange(len(recorded('ve3kpg_1')))
    result = p4robustrx.decode_symbols(symbols)
    assert calls == [1]
    assert result is not None and result.frame == reference.frame
