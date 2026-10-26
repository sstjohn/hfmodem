# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The native final ACK had a faded first fixed preamble pair, not a missing tail."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.tests.evidence import CORPUS, require_input
from hfmodem.contact.slots import Mixer
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_responder_control_stream import station, feed
from .test_idle_poll import clock as clock

ORIGINAL=b'15 de W9SSJ <R-08>12 <LOC:EN63bb>7 hello?\n'


def pending():
    hs,io=station('W9SSJ')
    hs.caller='KQ4QWH'
    hs.turn=A._TURN_OURS
    hs._link_high_speed=True
    hs._rec500=A._phy.level_key('500',6)
    hs._lead=A.VB6Rnd(F.lead_state(hs.caller,True,'500'))
    hs._start_compression('TEXT')
    hs.send(ORIGINAL)
    assert len(ORIGINAL)==42 and hs._tx_pending is not None
    assert hs.payload_ledger.pending==42 and hs.payload_ledger.confirmed==0
    return hs,io


def recorded():
    prov=json.loads((Path(__file__).parent/'fixtures/kq4qwh-turn-loop-1024/provenance.json').read_text())
    p=require_input(CORPUS/prov['corpus_path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==prov['sha256']
    fs,x=wavfile.read(p);assert fs==48000
    return Mixer(1500)(x.astype(float)/32768)


@pytest.mark.parametrize('block',[960,4096,24000])
def test_native_final_ack_settles_original_once_and_releases_turn(block):
    hs,io=pending();before=len(io.sent)
    x=recorded();feed(hs,x,block)
    assert hs.payload_ledger.confirmed==42, io.msgs
    assert hs.payload_ledger.pending==0 and hs._tx_pending is None
    assert hs.turn==A._TURN_PEER and hs._answer_query_at is None
    assert len(io.sent)==before+1
    assert np.array_equal(io.sent[-1],M.synth_tone_pairs(F.turn_grant(hs.called,'500')))
    feed(hs,np.r_[np.zeros(24000),x],block)
    assert hs.payload_ledger.confirmed==42


@pytest.mark.parametrize('kind',['wrong-call','wrong-role','missing-tail','missing-payload-pair'])
def test_faded_head_exception_never_credits_wrong_identity_or_incomplete_payload(kind):
    hs,io=pending();before=len(io.sent)
    pairs=F.control_bursts('N0ZZZ' if kind=='wrong-call' else hs.caller,'500')[1 if kind=='wrong-role' else 0]
    wave=M.synth_tone_pairs(pairs)
    if kind=='missing-tail':wave=wave[:-3*M.HOP]
    if kind=='missing-payload-pair':wave[7*M.HOP:8*M.HOP]=0
    feed(hs,np.r_[np.zeros(4800),wave,np.zeros(24000)])
    assert hs.payload_ledger.confirmed==0 and hs.payload_ledger.pending==42
    assert hs._tx_pending is not None and len(io.sent)==before
