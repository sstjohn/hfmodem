"""ARDOP recording observations using the active modem's rolling receiver."""
import hashlib
import re
import sys
from pathlib import Path
import numpy as np
from hfmodem.besra.phy.demodulator import Demodulator, SAMPLE_RATE, frame_span, _carrier_freq, _HEADER_LEN
from hfmodem.besra.frame import frame as F
from hfmodem.besra.radio import RollingDecoder
from hfmodem.core.resample import from_card
from pactor_view import write_points
import tx_timing
from vara_packets import packet_bytes
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.rehear import sessionlog, besra


def intervals(text,recording,kind,duration,log=None):
    explicit=tx_timing.intervals(text,recording,kind,duration,'ARDOP')
    if kind!='rig rx' or log is None or Path(recording).stem not in text:
        return explicit,'No matched original-session ARDOP timing; sender attribution is incomplete.'
    from vara_view import load_audio
    audio,_=load_audio(Path(recording))
    edges,quiet,level=besra._key_edges(from_card(audio,SAMPLE_RATE))
    started=besra._started(Path(recording))
    try:
        where=sessionlog.align(log,[a for a,b in edges],besra._quieter(level,log.keyed),
                               log.predicted(started) if started else None)
    except sessionlog.Ambiguous:
        where=None
    if where is None:
        return explicit,'Logged ARDOP transmissions could not be uniquely placed on this recording.'
    names=re.findall(r'\bTX (\S+) [\d.]+s',text)
    out=[]
    for i,(a,b) in enumerate(log.keyed):
        start,end=a-where.offset,b-where.offset
        if end<=0 or start>=duration:continue
        out.append({'id':f'ardop-tx-{i}','t0':max(0,start),'t1':min(duration,end),
                    'log_t0':a,'log_offset':where.offset,'kind':names[i] if len(names)==len(log.keyed) else 'ARDOP transmission',
                    'protocol':'ARDOP','evidence_type':'ptt','direction':'tx','role':'caller','tx_confirmed':True,
                    'status':'logged PTT interval','symbols':None,'origin':'original ARDOP TX/PTT log',
                    'detail':str(where),'attribution':'local transmitter; original key/unkey log aligned to recording',
                    'timing_precision':'log edges aligned to receiver mute; software timing'})
    return tx_timing.merge(explicit,out),f'{len(out)}/{len(log.keyed)} ARDOP transmit intervals aligned from original TX/PTT records. {where}'


def replay(audio,bw,calls,out,progress=print,tx=(),decode_tx=False,log=None):
    keyed=next((e for e in tx if 'log_t0' in e),None)
    offset=keyed['log_t0']-keyed['t0'] if keyed else 0
    party=besra._Party.of(log if keyed else None,offset)
    decoder=Demodulator(expect_session=party.expect,rx_epoch=party.epoch,collect_symbols=True)
    events=[]
    def heard(pos,f):
        fd=F.FRAMES[f.type];t=pos/SAMPLE_RATE;end=min(len(audio)/48000,t+frame_span(f.type)/SAMPLE_RATE)
        if t>=end:return
        # Rolling acquisition may replace a failed body with a later successful
        # attempt at the same physical frame. Retain one observation.
        events[:]=[e for e in events if abs(e['t0']-t)>.12]
        number=f'{pos}-{f.type}';name=None
        if f.symbol_rows:
            points=[];times=[];carriers=[]
            for car,phase,mag,clock in f.symbol_rows:
                scale=max(float(mag[0]),1e-30)
                points.extend(mag[1:]/scale*np.exp(1j*phase/1000))
                times.extend(t+clock/SAMPLE_RATE);carriers.extend([_carrier_freq(car,fd.carriers)]*len(phase))
            order=4 if fd.mod is F.Mod.PSK4 else 8
            ideal=np.exp(2j*np.pi*np.arange(order)/order)
            if fd.mod is F.Mod.QAM16:ideal=np.r_[ideal,ideal*.5]
            name=f'symbols-ardop-{number}.json'
            write_points(out,name,points,times,carriers,ideal,fd.mod.value,
                         'measured differential phase and carrier magnitude relative to reference symbol; this transmission only')
        data=fd.has_crc and not f.unverified
        status=('header candidate' if f.header_only else 'truncated body' if f.truncated else
                'header-only control; no payload check' if f.unverified else
                ('CRC pass after combining' if f.combined else 'CRC pass') if f.ok and data else
                'decoded control (RS or repetition check)' if f.ok else 'body decode failed')
        e={'id':f'ardop-{number}','t0':t,'t1':end,'kind':f.name,'protocol':'ARDOP','direction':'observed',
           'status':status,'frame_type':'data' if data and not f.header_only else 'observation',
           'symbols':name,'stage':fd.mod.value,'modulation':fd.mod.value,'carrier_count':fd.carriers,
           'symbol_rate':fd.baud,'combined':f.combined,'origin':'ARDOP rolling audio receiver',
           'detail':f'Session 0x{f.session_id:02X}; '+(f'{len(f.payload)} payload bytes. ' if f.ok else '')+
                    ('CRC uses retransmission memory; plotted symbols are this copy. ' if f.combined else '')+
                    'Time begins at the decoded header; preceding leader remains in the waterfall.',
           'quality':f.quality,'offset_hz':f.cfo_hz,'carrier_hz':sorted(set(carriers)) if name else [],
           'timing_precision':'header clock and frame-type body length',
           'anatomy':[{'start_s':0,'end_s':min(end-t,_HEADER_LEN/SAMPLE_RATE),'label':'Frame-type header','basis':'Decoded header clock · 4FSK, 50 Bd; leader precedes this interval.'}] +
                     ([{'start_s':_HEADER_LEN/SAMPLE_RATE,'end_s':end-t,'label':'Coded body','basis':'Nominal span from decoded frame type; includes its reference/training symbols where present.'}] if end-t>_HEADER_LEN/SAMPLE_RATE else [])}
        if f.ok and data:
            e['packet_sha256']=hashlib.sha256(bytes([f.type&254])+f.payload).hexdigest()
            e['payloads']=[packet_bytes(f.payload, 'ARDOP frame payload', session_id=f.session_id, frame_type=f.type)]
        if f.ok and f.caller and f.target:
            e['handshake_key']=f'ARDOP {f.name} {f.caller} {f.target}'
            if f.caller==calls[0]:e.update(direction='tx',tx_confirmed=True,attribution='decoded connection request names the local caller')
        events.append(e)
    rolling=RollingDecoder(decoder,heard,resample=True)
    for lo in range(0,len(audio),4800):
        party.at=(lo+4800)/48000
        rolling.push(audio[lo:lo+4800]);rolling.pump()
        if lo%(20*48000)==0:progress(f'  ARDOP {lo/48000:.0f}s: {len(events)} observations',flush=True)
    rolling.flush()
    return sorted(events,key=lambda e:e['t0'])
