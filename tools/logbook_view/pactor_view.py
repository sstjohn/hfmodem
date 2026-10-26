# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Single-modem PACTOR replay adapter for the shared QSO inspector."""
import dataclasses
import hashlib
import json
from pathlib import Path

from vara_packets import packet_bytes
import numpy as np
from hfmodem.shrike import rxfront, p3rx, placement, rx, spec, p2rx, pactor2, p1rx

FS=spec.SAMPLE_RATE


def write_points(out, name, points, times, carriers, ideal, stage, normalization):
    points=np.asarray(points).ravel();times=np.asarray(times).ravel();carriers=np.asarray(carriers).ravel()
    valid=np.isfinite(points)&np.isfinite(times)
    data={'i':points.real[valid].round(6).tolist(),'q':points.imag[valid].round(6).tolist(),
          't':times[valid].round(6).tolist(),'carrier':carriers[valid].tolist(),
          'ideal':[[float(z.real),float(z.imag)] for z in ideal], 'stage':stage,
          'normalization':normalization,'omitted_nonfinite_or_outside':int((~valid).sum())}
    (Path(out)/name).write_text(json.dumps(data,allow_nan=False,separators=(',',':')))
    return name


def p3_points(audio, packet):
    """Read the accepted frame's exact matched-filter clock and phase axes.

    P3's ordinary demodulator uses differential phase only. Preserve that unit
    magnitude instead of manufacturing amplitude noise around ideal points.
    """
    path=(placement.LONG_PATHS if packet.long_cycle else placement.SPEED_PATHS)[packet.sl]
    if packet.carrier_swapped:
        path=dataclasses.replace(path,tones=tuple(spec.CARRIER_SWAP[cn] for cn in path.tones))
    sps=FS//100; pulse=rx._pulse(sps);delay=(len(pulse)-1)//2
    # Filter only this frame with sufficient original lookback and tail.
    lo=max(0,packet.start-len(pulse)-2*sps)
    hi=min(len(audio),packet.start+(path.n_symbols+2)*sps+len(pulse))
    points=[];times=[];carriers=[]
    for cn,offset in zip(path.tones,path.clock_offsets(sps)):
        z=rx._baseband(audio[lo:hi],cn,FS,pulse)
        at=packet.start-lo+offset+np.arange(-1,path.n_symbols)*sps+delay
        if at[0]<0 or at[-1]>=len(z):
            continue
        y=z[at]; phase=np.angle(y[1:]*y[:-1].conj())-packet.phase_rotation
        points.extend(np.exp(1j*phase));times.extend((at[1:]-delay+lo)/FS)
        carriers.extend([spec.channel_freq_hz(cn)]*path.n_symbols)
    ideal=(np.exp(1j*np.array(list(placement.BIT_PHASE.values()))) if path.bits_per_cell==1
           else np.exp(1j*placement.DIBIT_PHASE.ravel()))
    return np.asarray(points),np.asarray(times),np.asarray(carriers),ideal,path


def replay(audio,bw,calls,out,progress=print,tx=(),decode_tx=False, transcript=""):
    events=[];duration=len(audio)/FS
    # Chunk the general front end with overlap. P3 data is decoded separately
    # below so its accepted geometry and combining provenance remain available.
    seen=[]
    for lo in range(0,len(audio),20*FS):
        hi=min(len(audio),lo+22*FS);segment=audio[lo:hi]
        for ev in rxfront.decode_events(segment,p3_packets=False,p3_envelope=False,mycall=calls[-1]):
            t=lo/FS+ev.t
            if any(kind==ev.kind and protocol==ev.protocol and abs(t-old)<.18 for old,kind,protocol in seen):
                continue
            seen.append((t,ev.kind,ev.protocol))
            protocol=ev.protocol or 'PACTOR-1'
            status='CRC pass' if ev.packet and ev.packet[3] else 'decoded control' if ev.kind in ('cs','connect','callb','unassigned') else 'signal candidate'
            # Several front-end families timestamp the END of their window.
            # Preserve that fact; do not present guessed bounds as frame edges.
            start=t;end=t
            if ev.kind=='cs' and protocol=='PACTOR-3' and ev.start is not None:
                start=lo/FS+ev.start/FS;end=start+.21
            kind=(f'{protocol} SL{ev.packet[0]} data' if ev.packet else
                  f'{protocol} CS{ev.cs}' if ev.kind=='cs' else
                  f'{protocol} '+{'connect':'connect request','callb':'robust call','unassigned':'unassigned control','detect':'carrier pair','fsk':'FSK activity','p1reply':'burst candidate'}.get(ev.kind,ev.kind))
            e={'id':f'pactor-{len(events):05d}','t0':max(0,start),'t1':min(duration,end),
               'kind':kind,'protocol':protocol,'direction':'observed','status':status,'detail':ev.text,
               'frame_type':'data' if ev.packet else 'observation','symbols':None,
               'origin':'PACTOR receive front end',
               'context_before_s':1.2,'context_after_s':.25,
               'timing_precision':'detector timestamp; preceding audio shown as context' if start==end else 'decoder control-symbol clock'}
            if ev.kind=='cs' and protocol=='PACTOR-1':
                e['t1']=min(duration,e['t0']+.12)
                e['timing_precision']='control-word detection; nominal 120 ms duration'
            if ev.packet:
                sl,st,payload,ok=ev.packet
                e['packet_sha256']=hashlib.sha256(protocol.encode()+bytes([st])+payload).hexdigest()
                if ok:e['payloads']=[packet_bytes(payload, 'PACTOR decoded payload · may contain source coding or padding', status_byte=st)]
                if protocol=='PACTOR-3' and ev.text.startswith('HEADER field='):
                    e.update(kind='PACTOR-3 SL1 header', frame_type='observation',
                             status='decoded control · CRC-verified header', timestamp_only=True)
                    e.pop('payloads', None)
            if ev.connect:
                e['handshake_key']=f'P1 {ev.connect.variant} {ev.connect.callsign}'
            events.append(e)
        progress(f'  PACTOR {lo/FS:.0f}–{hi/FS:.0f}s: {len(events)} observations',flush=True)
    if 'robust' in transcript.lower():
        from hfmodem.shrike.onair import _candidate_excess, ANSWER_CODEWORD_X
        answers=[]
        for lo in range(0,len(audio),FS):
            segment=audio[lo:min(len(audio),lo+int(1.4*FS))]
            got=p1rx.acquire_robust_answer_search(segment,0,min(1.2,len(segment)/FS-.2))
            if got is None or got.errors or got.bits!=20 or _candidate_excess(segment,got.at)<ANSWER_CODEWORD_X:
                continue
            t=lo/FS+got.at
            if any(abs(t-old)<.15 for old in answers):
                continue
            answers.append(t)
            events.append({'id':f'robust-{len(events):05d}','t0':t,'t1':min(duration,t+.2),
                'kind':f'Robust answer {got.entry}','protocol':'PACTOR-1','direction':'observed',
                'frame_type':'observation','status':'decoded control','symbols':None,
                'handshake_key':f'robust answer {got.word:06x}', 'origin':'PACTOR robust-answer audio decoder',
                'detail':str(got),'offset_hz':got.offset_hz})
    # The existing P3 scanner holds its ARQ memory across the full recording.
    scan=p3rx.decode_p3_packets(audio,envelope=False)
    for packet in scan.packets:
        points,times,carriers,ideal,path=p3_points(audio,packet)
        stage=f'P3 differential {"BPSK" if path.bits_per_cell==1 else "QPSK"}'
        number=len(events);name=f'symbols-pactor-{number:05d}.json'
        write_points(out,name,points,times,carriers,ideal,stage,'phase-only receiver decision input; unit magnitude; individual transmission')
        events.append({'id':f'p3-{number:05d}','t0':max(0,packet.start/FS-.09),
            't1':min(duration,(packet.start+path.n_symbols*(FS//100))/FS),
            'kind':f'P3 SL{packet.sl} data','protocol':'PACTOR-3','level':packet.sl,
            'direction':'observed','status':'CRC pass after combining' if packet.combined else 'CRC pass',
            'frame_type':'data','symbols':name,'stage':stage,'carrier_hz':sorted(set(carriers.tolist())),
            'origin':'PACTOR-3 packet decoder','combined':packet.combined,
            'anatomy':[{'start_s':0,'end_s':min(.09,packet.start/FS),'label':'Pre-data context','basis':'Context preceding the decoder payload clock; not a measured preamble boundary.'}, {'start_s':min(.09,packet.start/FS),'end_s':min(.09,packet.start/FS)+path.n_symbols/100,'label':'Coded payload symbols','basis':'Accepted P3 carrier/symbol clock.'}],
            'detail':f'{len(packet.payload)} payload bytes; sequence {packet.seq}; phase-only unit-magnitude display. '+('CRC recovered across retransmissions; plotted symbols are this transmission only.' if packet.combined else 'CRC verified on this transmission.'),
            'payloads':[packet_bytes(packet.payload, 'PACTOR-3 payload · may contain source coding or padding', status_byte=packet.status, sequence=packet.seq)],
            'packet_sha256':hashlib.sha256(b'PACTOR-3'+bytes([packet.status])+packet.payload).hexdigest()})
    events=[e for e in events if not(e.get('protocol')=='PACTOR-1' and e.get('frame_type')=='data')]
    # Individual P1 data and changeover frames use the existing header/eye/CRC
    # gates. Never use the CRC by itself as an acquisition test.
    for lo in range(0,len(audio),20*FS):
        segment=audio[lo:min(len(audio),lo+22*FS)]
        for breakin in (False,True):
            for packet in p1rx.decode_p1_packets(segment,breakin=breakin):
                if packet.start is None:
                    continue
                t=(lo+packet.start)/FS
                if any(e.get('origin')=='PACTOR-1 header/eye/CRC decoder' and abs(e['t0']-t)<.2 for e in events):
                    continue
                events.append({'id':f'p1-{len(events):05d}','t0':t,'t1':min(duration,t+.96),
                    'kind':f'P1 {packet.baud} Bd '+('changeover' if breakin else 'data'),
                    'protocol':'PACTOR-1','direction':'observed','frame_type':'data',
                    'status':'CRC pass','symbols':None,'origin':'PACTOR-1 header/eye/CRC decoder',
                    'detail':f'{len(packet.payload)} payload bytes; sequence {packet.packet_count}',
                    'payloads':[packet_bytes(packet.payload, 'PACTOR-1 payload · may contain source coding or padding', status_byte=packet.status, sequence=packet.packet_count)],
                    'packet_sha256':hashlib.sha256(b'PACTOR-1'+bytes([packet.status])+packet.payload).hexdigest()})
    def p2_frame(t,field,path,points,bin_pair,swapped,combined):
        number=len(events)
        carriers=[p2rx.ACQ_LO_HZ+25*(b if b<16 else b-32) for b in p2rx.lane_tones(bin_pair,swapped)]
        name=None;stage='P2 differential PSK'
        if points is not None:
            name=f'symbols-pactor-{number:05d}.json'
            times=np.array([t-lead/p2rx.ACQ_RATE+(np.arange(len(z))+1)/100 for z,lead in zip(points,(0,p2rx.LANE_LEAD_FRAMES))])
            write_points(out,name,np.concatenate(points),times.ravel(),np.repeat(carriers,path.n_symbols),
                         np.exp(-1j*pactor2.cell_steps(path.bits_per_cell)),stage,
                         'receiver differential magnitudes normalized to burst median; fitted per-carrier phase')
        events.append({'id':f'p2-{number:05d}','t0':max(0,t),'t1':min(duration,t+path.n_symbols/100),
            'kind':f'P2 SL{path.level+1} data','protocol':'PACTOR-2','level':path.level+1,'direction':'observed',
            'frame_type':'data','status':'CRC pass after combining' if combined else 'CRC pass',
            'symbols':name,'stage':stage,'carrier_hz':carriers,'combined':combined,
            'anatomy':[{'start_s':0,'end_s':path.n_symbols/100,'label':'Coded payload symbols','basis':'Accepted P2 burst clock; marker not included.'}],
            'origin':'PACTOR-2 recording decoder','detail':f'{path.crc_bytes-3} field bytes; '+('CRC recovered across retransmissions; no single-copy constellation assigned.' if combined else 'CRC verified on this transmission.'),
            'payloads':[packet_bytes(field[:path.crc_bytes-3], 'PACTOR-2 payload · may contain source coding or padding', status_byte=field[path.crc_bytes-3])],
            'packet_sha256':hashlib.sha256(b'PACTOR-2'+field[:path.crc_bytes]).hexdigest()})
    markers=p2rx.find_markers(audio,FS,.90)
    for long_frame in (False,True):
        for level in sorted({(k>>1)&3 for _t,_b,k,_s,score,_sw in markers if bool((k>>3)&1)==long_frame and score>=p2rx.MARKER_ARM}):
            p2rx.decode_bursts(audio,FS,level,long_frame,on_frame=p2_frame)
    if 'P4' in transcript or 'PACTOR-4' in transcript:
        events.extend(p4_replay(audio,out,progress,tx=tx))
    if len(events)>10000:
        raise ValueError('PACTOR replay event limit exceeded.')
    return sorted(events,key=lambda e:(e['t0'],e['id']))


def p4_replay(audio,out,progress=print,tx=()):
    """Acquire from audio; the transcript only admits the P4 decoder family.

    Control words seed bounded normal-short searches on the 1.25 s raster.
    Each candidate must independently pass the training gate; only a CRC pass
    becomes decoded data. No original-session payload is substituted for audio.
    """
    from hfmodem.shrike import p4oracle,p4short,p4rx,p4words
    events=[];duration=len(audio)/FS
    heads=p4oracle.word_scan(audio,FS,whole=True)
    for head in heads:
        t=head.stream_sample/FS
        frame=p4oracle.body_after(audio,FS,0,head) if head.word=='Y27' else None
        length=.796 if frame or (head.extent_s or 0)>.5 else head.duration_s
        fraction,points,start=p4oracle.placed_despread(audio,FS,t,head.cfo_hz,count=27)
        name=None
        if start is not None and len(points)>1:
            points=points[1:]*points[:-1].conj()
            scale=np.median(abs(points))
            if scale>0:
                points/=scale
            times=start/FS+np.arange(1,len(points)+1)/p4words.SYMBOL_RATE
            name=f'symbols-p4-word-{len(events):05d}.json'
            write_points(out,name,points,times,np.full(len(points),1500),
                         np.exp(1j*np.arange(4)*np.pi/2),'P4 spread DQPSK control',
                         'measured differential despread symbols; median magnitude; control word only')
        e={'id':f'p4-word-{len(events):05d}','t0':t,'t1':min(duration,t+length),
           'kind':f'P4 {head.word}'+(' + body' if frame else ''),'protocol':'PACTOR-4',
           'direction':'observed','status':'CRC pass' if frame else 'decoded control',
           'frame_type':'data' if frame else 'observation','symbols':name,
           'stage':'spread DQPSK control'+(' + BPSK body' if frame else ''),
           'origin':'PACTOR-4 audio word/body decoder','offset_hz':head.cfo_hz,
           'anatomy':[{'start_s':0,'end_s':min(length,head.duration_s),'label':'Spread control word','basis':'Acquired word clock and nominal symbol duration.'}]+([{'start_s':head.duration_s,'end_s':length,'label':'Body interval','basis':'Nominal remaining word/body interval; internal training boundaries not exported.'}] if frame and length>head.duration_s else []),
           'detail':'Control-word constellation; body uses a different modulation.' if frame else 'Control word decoded from audio.'}
        if frame:
            e['packet_sha256']=hashlib.sha256(b'PACTOR-4'+bytes([frame.status.byte])+frame.data).hexdigest()
            e['payloads']=[packet_bytes(frame.data, 'PACTOR-4 source-coded data · not decompressed application text', status_byte=frame.status.byte)]
            e['detail']+=f' {len(frame.data)} source-coded bytes; status 0x{frame.status.byte:02x}.'
        events.append(e)
    # Prefer low-offset hypotheses when word acquisition finds carrier aliases.
    # Multiple heads on the same raster must not produce duplicate packets.
    tried=[]
    for head in sorted((h for h in heads if h.word=='Y27'),key=lambda h:abs(h.cfo_hz)):
        for at in range(head.stream_sample+round(1.25*FS),len(audio),round(1.25*FS)):
            if any(abs(at-old)<.02*FS for old in tried):
                continue
            tried.append(at)
            lo=max(0,at-round(.05*FS));hi=min(len(audio),at+round(.95*FS))
            z=p4rx.audio_to_baseband(audio[lo:hi],FS)
            nominal=round((at-lo)*28800/FS)+p4short.GEOM
            placed=p4short.place_short(z,nominal,head.cfo_hz)
            if placed is None or placed[0]<p4short.NORMAL_COHERENCE_KNEE:
                continue
            coherence,training,cfo=placed
            captured=[]
            scored=p4short.short_scores(z,first_training_start=training,frequency_offset=cfo,refine=0,
                                       on_symbols=lambda points,times:captured.append((points,times)))
            if scored is None:
                continue
            frame,_=p4short.decode_scores(scored[2])
            t=at/FS+(training-nominal)/28800
            name=f'symbols-p4-normal-{len(events):05d}.json'
            points,times=captured[0]
            write_points(out,name,points,lo/FS+times/28800,np.full(len(points),1500),
                         np.array([-1+0j,1+0j]),'P4 normal-short BPSK',
                         'measured receiver equalizer output; training-fitted channel gain; individual transmission')
            e={'id':f'p4-normal-{len(events):05d}','t0':max(0,t),'t1':min(duration,t+.805),
               'kind':'P4 normal-short data' if frame else 'P4 normal-short training',
               'protocol':'PACTOR-4','direction':'observed','status':'CRC pass' if frame else 'training lock; CRC not recovered',
               'frame_type':'data' if frame else 'observation','symbols':name,'stage':'BPSK · 1800 Bd · normal short (SL5/6 body geometry)',
               'origin':'PACTOR-4 normal-short audio decoder','offset_hz':cfo,
               'detail':f'Training coherence {coherence:.2f}. Speed level is not resolved from this body geometry.'}
            if frame:
                e['packet_sha256']=hashlib.sha256(b'PACTOR-4'+bytes([frame.status.byte])+frame.data).hexdigest()
                e['payloads']=[packet_bytes(frame.data, 'PACTOR-4 source-coded data · not decompressed application text', status_byte=frame.status.byte)]
                e['detail']+=f' {len(frame.data)} source-coded bytes; status 0x{frame.status.byte:02x}; CRC verified on this transmission.'
            events.append(e)
    # A closed energy envelope is not required to read a robust field: AGC
    # and noise can join a whole run of overs into one apparent long burst.
    # Read bounded symbol windows on independently acquired word clocks.
    from hfmodem.shrike import p4robusttx
    candidates=[(round(e['t0']*FS),0.) for e in tx if 'P4' in e['kind']]
    for head in sorted(heads,key=lambda h:abs(h.cfo_hz)):
        candidates.extend((at,head.cfo_hz) for at in range(head.stream_sample,len(audio),round(1.25*FS)))
    checked=[]
    for at,cfo in candidates:
        if any(abs(at-old)<.02*FS for old in checked):
            continue
        checked.append(at)
        t=at/FS
        if any(abs(e['t0']-t)<.06 and (e['status']=='CRC pass' or 'word decoder' in e.get('origin','') or e['kind'].startswith(('P4 X','P4 Y','P4 V'))) for e in events):
            continue
        fraction,points,start=p4oracle.placed_despread(audio,FS,t,cfo,count=p4robusttx.SHORT_SYMBOLS)
        if start is None or fraction<p4oracle.ROBUST_FRACTION_KNEE:
            continue
        field=p4oracle.robust_symbols(points,p4robusttx.SHORT_SYMBOLS)
        header,value=field['p4_hdr'],field['p4_value']
        if header not in ('A','B') and value is None:
            continue
        t=start/FS
        if any(abs(e['t0']-t)<.06 and e.get('origin')=='PACTOR-4 robust-field audio decoder' for e in events):
            continue
        points=points[1:]*points[:-1].conj()
        rotation=np.angle(np.mean((points/np.maximum(abs(points),1e-30))**4))/4
        points*=np.exp(-1j*rotation)
        scale=np.median(abs(points))
        if scale>0:points/=scale
        name=f'symbols-p4-robust-{len(checked):05d}.json'
        write_points(out,name,points,t+np.arange(1,len(points)+1)/112.5,np.full(len(points),1500),
                     np.exp(1j*np.arange(4)*np.pi/2),'P4 robust spread differential PSK',
                     'measured despread differential symbols; common phase removed; median magnitude; header and Walsh field')
        events=[e for e in events if not(e['status'].startswith('training lock') and abs(e['t0']-t)<.06)]
        events.append({'id':f'p4-robust-{len(checked):05d}','t0':t,'t1':min(duration,t+p4robusttx.EMISSION_S),
                       'kind':f'P4 robust {header or "?"}'+(f' · value {value}' if value is not None else ''),
                       'protocol':'PACTOR-4','direction':'observed','frame_type':'observation',
                       'status':'decoded control' if header in ('A','B') and value is not None else 'partial robust field',
                       'symbols':name,'stage':'spread differential PSK · 112.5 symbols/s · robust field',
                       'origin':'PACTOR-4 robust-field audio decoder',
                       'detail':f'Spread coherence {fraction:.2f}; Walsh peak {field["p4_walsh_peak"]}/64. Header uses quarter turns; the Walsh field uses binary phase steps. This is a decoded symbol window, not a measured RF envelope or a payload CRC.',
                       'timing_precision':'85-symbol window on the acquired despreader clock'})
    progress(f'  PACTOR-4: {len(events)} observations; {sum(e["status"]=="CRC pass" for e in events)} CRC-valid bodies',flush=True)
    return events
