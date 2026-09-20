# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""First real CS3 body split across the pre-key/recovered-slot read boundary."""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

FS = onair.FS
META_PATH = (Path(__file__).with_name("fixtures") /
             "ws8eoc-0912-first-changeover.json")
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else None
pytestmark = pytest.mark.skipif(
    META is None or not META_PATH.with_suffix(".wav").exists(),
    reason="recorded WS8EOC first-changeover fixture is not installed")


def scene(tmp_path, *, wrong_body=False):
    session = _Session(entry_pending=True)
    host, receiver = session.host, session.rx
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(host)
    host.peer = tx
    tx.defer_p3_cs = True
    live = _Bench(seconds=30)
    crop = recorded_pcm(META)
    first = META["first_sample"]
    live.audio[first:first+len(crop)] = crop
    if wrong_body:
        # Keep the real coherent head; replace only the following body with
        # deterministic noise. A head alone must not authorize a role change.
        at = round(21.25*FS)
        live.audio[at:first+len(crop)] = np.random.default_rng(31).normal(
            0,.01,first+len(crop)-at)
    end = META["original_read_end_sample"]
    live.pos = live.now = end
    grid = onair._MasterGrid(2724,60000,round(.185*FS),
                            packet_n=46080,cs_n=5760,d_max_n=6240)
    grid.protocol = spec.Protocol.PACTOR3
    grid.d_n = .0921*FS
    grid.d_ref_n = 46080
    tx.raster = grid
    tx.slots_used = [16]
    tx.aim(grid,17)
    receiver.new_cycle()
    seg = live.audio[first:end].copy()
    # Actual old P1 answer anchor from hold_02/03, not a supplied CS3 head.
    assert receiver.control_signal(seg,first,1013225) is None
    assert host.arq.role == arq.ISS
    assert receiver._p3_delivered_at is None
    return session,tx,live,grid,seg,first


@pytest.mark.parametrize("overrun", [False,True])
def test_first_recorded_body_is_delivered_before_next_entry(tmp_path,overrun):
    session,tx,live,grid,seg,first = scene(tmp_path)
    if overrun:
        # The real arm missed slot17. The other case is the important control:
        # preserving the body must not depend on an accidental decoder overrun.
        live.spend(round(.08*FS))
    slot,whole,origin = onair._regrid(live,grid,tx,session.host,session.rx,
                                     17,seg,first,1920)
    assert session.host.arq.role == arq.IRS
    assert not session.host.arq.entry_pending
    assert [e.packet[2] for e in session.packets] == [b"RMS"]
    assert not grid.sending
    assert tx.boundary == grid.boundary(slot)
    assert tx.n == 0  # Receiving a frame deferred its ACK; it emitted no entry.
    assert tx._pending_p3_cs is not None
    assert live.pos < META["slot18_boundary"]
    session.host.tick()
    assert tx.n == 0
    assert tx._pending_p3_cs == 0  # Next emission is the deferred CS1 ACK.
    # The later generic scan sees the same physical frame. Delivery and timing
    # watermark remain singular despite the two paths sharing the audio.
    delivered = session.rx._p3_delivered_at
    session.rx.new_cycle()
    onair._scan_frame(session.rx,whole,origin)
    assert [e.packet[2] for e in session.packets] == [b"RMS"]
    assert session.rx._p3_delivered_at == delivered


def test_wrong_body_expires_and_the_same_head_cannot_extend_hold(tmp_path):
    session,tx,live,grid,seg,first = scene(tmp_path,wrong_body=True)
    receiver=session.rx
    slot,whole,origin=onair._regrid(live,grid,tx,session.host,receiver,17,seg,first,1920)
    assert session.host.arq.role == arq.ISS
    assert session.host.arq.entry_pending
    assert not session.packets
    assert receiver._p3_delivered_at is None
    assert slot == 18  # One bounded receive extension; entry can resume here.
    position=live.pos
    receiver.new_cycle()
    assert receiver.control_signal(seg,first,1013225) is None
    _,_,_=onair._regrid(live,grid,tx,session.host,receiver,slot,whole,origin,1920)
    assert live.pos == position
    assert not session.packets


def test_noise_does_not_inhibit_an_entry(tmp_path):
    session,tx,live,grid,seg,first=scene(tmp_path)
    receiver=session.rx
    # Fresh receiver state, so only this negative-control window is evidence.
    receiver._p3_head_candidate=None
    receiver._p3_entry_body_candidate=None
    noise=np.random.default_rng(4).normal(0,.01,len(seg))
    assert receiver.control_signal(noise,first,1013225) is None
    position=live.pos
    onair._regrid(live,grid,tx,session.host,receiver,17,noise,first,1920)
    assert live.pos == position
    assert session.host.arq.role == arq.ISS


def test_missing_audio_is_not_joined_into_a_valid_frame(tmp_path):
    session,tx,live,grid,seg,first=scene(tmp_path)
    live.pos += 480  # Ten real milliseconds are absent from the receive window.
    live.now = live.pos
    onair._regrid(live,grid,tx,session.host,session.rx,17,seg,first,1920)
    assert session.host.arq.role == arq.ISS
    assert session.host.arq.entry_pending
    assert not session.packets
    assert session.rx._p3_delivered_at is None
    assert session.rx.p3_receive_offset_hz == 0


def test_later_distinct_head_gets_its_own_bounded_body_read(tmp_path):
    session,tx,live,grid,seg,first=scene(tmp_path,wrong_body=True)
    receiver=session.rx
    onair._regrid(live,grid,tx,session.host,receiver,17,seg,first,1920)
    checked=receiver._p3_entry_body_checked_at
    # Outside the two-cycle bare-head corroboration budget. This must acquire
    # anew, not get a role change from the earlier unvalidated head.
    shift=3*60000
    crop=recorded_pcm(META)
    origin=first+shift
    live.audio[origin:origin+len(crop)]=crop
    live.pos=live.now=META["original_read_end_sample"]+shift
    for _ in range(3):
        receiver.new_cycle()
    seg=live.audio[origin:live.pos].copy()
    assert receiver.control_signal(seg,origin,1013225+shift) is None
    assert session.host.arq.role == arq.ISS
    assert receiver._p3_entry_body_candidate[0] > checked
    onair._regrid(live,grid,tx,session.host,receiver,20,seg,origin,1920)
    assert [e.packet[2] for e in session.packets] == [b"RMS"]
    assert receiver._p3_entry_body_checked_at > checked
    assert session.host.arq.role == arq.IRS


@pytest.mark.parametrize("cs5", [False, True])
def test_first_recorded_candidate_emits_one_correctly_ordered_ack(tmp_path,monkeypatch,cs5):
    session,tx,live,grid,seg,first=scene(tmp_path)
    session.host.p3_changeover_cs5=cs5
    control=arq.CS_NAK if cs5 else arq.CS_ACK
    tx.live,tx.sessrx=live,session.rx
    tx.transmit,tx.rig=True,_Rig()  # Simulated rig + sample-clock duplex only.
    # The assertions below are the staggered SCS control's own geometry, which
    # is the experiment now that the defaults are historical/audio-start.
    tx.p3_control_waveform,tx.p3_control_placement="current","pulse-center"
    # ...and it is a CARRIER ORDER, read back against nominal templates. The
    # recorded peer sits at -15 Hz and `--p3-follow-offset` would key the reply
    # there, which is its own subject (`test_control_offset_follow`).
    tx.p3_follow_offset="none"
    waveforms=[]
    transmit=live.transmit

    def record(samples,**kwargs):
        waveforms.append(samples.copy())
        return transmit(samples,**kwargs)

    monkeypatch.setattr(live,"transmit",record)
    slot,_,_=onair._regrid(live,grid,tx,session.host,session.rx,17,seg,first,1920)
    assert not live.emissions
    assert [e.packet[2] for e in session.packets] == [b"RMS"]
    assert session.packets[0].carrier_swapped is False
    # Deliberately oppose the corrected caller reply. Its order comes from
    # the peer's CRC permutation plus connection origin, not old FSK parity.
    monkeypatch.setattr(grid,"shift",lambda slot:False)
    session.host.tick()
    tx.emit_pending_cs()
    assert len(live.emissions) == len(waveforms) == 1
    assert tx.n == 1 and not tx.refused
    assert waveforms[0].size < FS//2  # No intervening entry packet.
    phase=tx.tx_audio_start+min(tx.tx_pulse_offsets)
    # Pin the existing target phase against the recorded head, including the
    # decoder's neighboring valid alignments. The CS5 pilot changes no timing;
    # this is not evidence of the stock peer's receive-window tolerance.
    assert abs(phase-(1008780+round(.890*FS))) <= round(.005*FS)
    assert live.keyed_at > 1008780+round(.810*FS)
    # Read back the actual submitted PCM against both physical-order controls.
    # This exercises RadioTx's render, trim, admission and duplex enqueue.
    fits=[]
    for swapped in (False,True):
        expected=onair._trim_silence(placement.control_signal(control,swapped=swapped))
        n=min(len(expected),len(waveforms[0]))
        a,b=expected[:n],waveforms[0][:n]
        fits.append(abs(float(a@b))/(np.linalg.norm(a)*np.linalg.norm(b)))
    assert fits[1] > .999999
    assert fits[0] < .95
    assert tx.tx_pulse_offsets[0]-tx.tx_pulse_offsets[1] == FS//200
    assert tx.rig.edges == [True,False]
