# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock-caller evidence for speed negotiation; guarded production acceptance."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from hfmodem.kestrel.vara import vara_frames as F, vara_mfsk as M, vara_ofdm as O, vara_arq as A
from hfmodem.kestrel.rx import varahf2300 as R, varahf500 as N
from .test_bw2750_peer_answer import awaiting

EVIDENCE_PATH = (Path(__file__).with_name('fixtures') /
    'kd7uhr-response-zero-0920/stock-setup-frames.json')
EVIDENCE = json.loads(EVIDENCE_PATH.read_text()) if EVIDENCE_PATH.is_file() else []

@pytest.mark.parametrize('row', EVIDENCE, ids=[r['run'] for r in EVIDENCE])
def test_setup_frame_matches_independent_stock_caller(row):
    assert row['crc_ok'] and not any(row['capture_statuses'].values())
    frame = F.link_setup_frame(row['caller'], row['bw'], row['level'])
    assert frame == bytes.fromhex(row['frame'])
    assert F.is_link_setup(frame)
    assert F.caller_from_link_setup(frame) == row['caller']

@pytest.mark.parametrize('bw,first', [('2750',0),('2300',14),('500',21)])
@pytest.mark.parametrize('level', [1,2,3,4])
def test_every_offer_selects_its_setup_level_on_tone_and_stream_paths(bw, first, level):
    k=F.connect_response(bw,level)
    assert (k.preadv-1)//F.lattice_step(k) == first+level-1
    for route in ('tones','stream','audio'):
        hs=awaiting('KD7UHR',bw)
        hs.mfsk_only=True
        if route=='tones':
            hs.on_rx_tones(F.handshake_tones('KD7UHR',k),F.connect_response(bw))
        else:
            # Short-preamble repeat: the payload fallback must work too.
            x=M.synth_burst('KD7UHR',replace(k,preamble=(62,)),amplitude=.25)
            x=np.concatenate([x,np.zeros(48000)])
            if route=='audio':hs.on_rx_audio(x)
            else:
                for at in range(0,len(x),4800):hs.on_rx_stream(x[at:at+4800])
        assert hs._setup_level == level, (bw,level,route,hs.io.logs)
        assert hs._linksetup_tx == 1
        assert hs.step == A._I_LINKSETUP_SENT
        assert hs.state == A.VaraState.CONNECTING
        assert not hs.answer_retry

@pytest.mark.parametrize('bw',['2750','2300','500'])
@pytest.mark.parametrize('level',[1,2,3,4])
def test_negotiated_waveform_carries_stock_frame(bw,level):
    x=O.link_setup_tx('W9SSJ-10',bw=bw,level=level)
    if bw=='500':
        # Narrow low levels have different alignment from the base waveform.
        spans=N.burst_spans(np.concatenate([x,np.zeros(4800)]))
        fr=N.decode_burst(x,O.ONSET_500) if level==4 else N.decode_burst(x,*spans[0])
    else:fr=R.decode_over(x,0,len(x),tries=2,level=level-1+(100 if bw=='2750' else 0))
    assert fr.crc_ok
    assert bytes(fr.frame_bytes)==F.link_setup_frame('W9SSJ-10',bw,level)


def test_repeated_offer_can_lower_speed_without_resetting_setup_budget():
    hs=awaiting('KD7UHR');hs.mfsk_only=True
    for level in (4,1,2,3):
        hs.on_rx_tones(F.handshake_tones('KD7UHR',F.connect_response('2750',level)),
                       F.connect_response('2750'))
    assert hs._linksetup_tx==hs.max_link_setups
    assert hs._setup_level==2  # fourth offer cannot spend another transmission
    assert hs.state==A.VaraState.CONNECTING


def test_refused_transmission_preserves_budget_and_selected_retry_speed():
    hs=awaiting('KD7UHR');hs.io.tx_went_out=lambda:False
    hs.on_rx_tones(F.handshake_tones('KD7UHR',F.connect_response('2750',1)),
                   F.connect_response('2750'))
    assert hs._setup_level==1 and hs._linksetup_tx==0
    assert not hs.resend_link_setup()
    assert hs._setup_level==1 and hs._linksetup_tx==0


@pytest.mark.parametrize('bw,position', [('2750',4),('2750',5),('2300',13),('2300',18),('500',20),('500',25)])
def test_adjacent_session_positions_do_not_become_connect_offers(bw, position):
    hs = awaiting('KD7UHR', bw)
    kind = F.connect_response(bw)
    kind = replace(kind, preadv=1 + 30 * position)
    audio = np.concatenate([M.synth_burst('KD7UHR', kind), np.zeros(48000)])
    for at in range(0, len(audio), 4800):
        hs.on_rx_stream(audio[at:at + 4800])
    assert hs._linksetup_tx == 0 and not hs.io.keyed
    assert hs.step == A._I_CR_SENT
