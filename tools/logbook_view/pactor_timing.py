# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PACTOR transmit intervals on the named session stream's sample clock."""
import re
import tx_timing
from pathlib import Path


def intervals(text, recording, kind, duration, fs=48000, p4_record=None):
    if kind!='rig rx':
        return [], 'PACTOR transmit grid requires the session stream recording.'
    structured=tx_timing.intervals(text,recording,kind,duration,'PACTOR')
    paths=re.findall(r'session stream: (\S+)',text)
    if len(paths)!=1 or Path(paths[0]).with_suffix('')!=Path(recording).with_suffix(''):
        return structured, 'Explicit local transmit events on the recording clock.' if structured else 'The transcript does not uniquely name this session stream; audio correspondence remains unassigned.'
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    anchors=list(re.finditer(r'master grid: anchor @ sample (\d+), cycle ([\d.]+) s',text))
    slots=list(re.finditer(r'\[grid\] (?:hold \d+ )?slot (\d+)(?: boundary @ sample (\d+))?',text))
    keys=list(re.finditer(r'^\s*TX\[(\d+)\]\s+(.+?)\s+\([\d.]+s\)\s+-- keying',text,re.M))
    used=set();parsed=[];previous_rf=0
    for i,key in enumerate(keys):
        end=keys[i+1].start() if i+1<len(keys) else len(text)
        rf=re.search(r'RF started (?:([-+][\d.]+) ms from the slot boundary|with no slot).*?PTT keyed ([\d.]+) s',text[key.end():end])
        if not rf:
            continue
        rf_end=key.end()+rf.end()
        anchor=next((a for a in reversed(anchors) if a.start()<key.start()),anchors[0] if anchors else None)
        preceding=[(j,s) for j,s in enumerate(slots) if previous_rf<s.start()<key.start() and j not in used]
        following=[(j,s) for j,s in enumerate(slots) if rf_end<s.start()<end and j not in used and 'hold ' in s[0]]
        picked=preceding[-1] if preceding else following[0] if following else None
        slot=None;boundary=None
        if picked:
            j,s=picked;used.add(j);slot=int(s[1]);boundary=int(s[2]) if s[2] else None
        elif i==0 and anchor is not None:
            slot=0
        parsed.append(dict(key=key,anchor=anchor,slot=slot,boundary=boundary,
                           offset=float(rf[1] or 0)/1000,length=float(rf[2]),inferred=False))
        previous_rf=rf_end
    # The entry transition omits its slot line. Interpolate only when the two
    # neighboring, explicitly logged slots leave exactly one slot for this key.
    for i,e in enumerate(parsed[1:-1],1):
        a,b=parsed[i-1],parsed[i+1]
        if e['slot'] is None and a['slot'] is not None and b['slot']==a['slot']+2 and e['anchor']==a['anchor']==b['anchor']:
            e['slot']=a['slot']+1;e['inferred']=True
    out=[]
    for e in parsed:
        if e['anchor'] is None or e['slot'] is None:
            continue
        anchor,cycle=int(e['anchor'][1])/fs,float(e['anchor'][2])
        start=(e['boundary']/fs if e['boundary'] is not None else anchor+e['slot']*cycle)+e['offset']
        if start>=duration or start+e['length']<=0:
            continue
        out.append({'id':f'ptt-{e["key"][1]}','t0':max(0,start),'t1':min(duration,start+e['length']),
                    'kind':e['key'][2],'evidence_type':'ptt','direction':'tx','role':'caller',
                    'status':'logged PTT interval','symbols':None,'origin':'original PACTOR session grid',
                    'detail':e['key'][2], 'attribution':'local caller; grid and RF-start timing',
                    'timing_precision':('slot interpolated between adjacent logged slots; ' if e['inferred'] else '')+'software timing, not measured RF edges'})
    # Old P4 sessions saved audio positions outside the human transcript.
    # Match the complete ordered emission list to the transcript, never zip a
    # partial list and silently assign later keys to the wrong transmission.
    p4keys=[k for k in keys if k[2].startswith('P4 ')]
    emissions=(p4_record or {}).get('emissions', [])
    rate=(p4_record or {}).get('sample_rate')
    if rate==fs and len(emissions)==len(p4keys) and emissions:
        valid=all(isinstance(e.get('audio_start'),int) and isinstance(e.get('audio_end'),int)
                  and 0<=e['audio_start']<e['audio_end'] and not e.get('refused')
                  and (not e.get('word') or f"word {e['word']}" in k[2])
                  for k,e in zip(p4keys,emissions))
        valid=valid and all(a['audio_end']<=b['audio_start'] for a,b in zip(emissions,emissions[1:]))
        if valid:
            for key,e in zip(p4keys,emissions):
                start,end=e['audio_start']/fs,e['audio_end']/fs
                if start>=duration or end<=0:
                    continue
                event={'id':f'ptt-{key[1]}','t0':start,'t1':min(duration,end),
                       'kind':key[2],'protocol':'PACTOR-4','evidence_type':'ptt',
                       'direction':'tx','role':'caller','tx_confirmed':True,
                       'status':'logged PTT interval','symbols':None,
                       'origin':'original PACTOR P4 emission record','detail':key[2],
                       'attribution':'local transmitter; saved audio sample positions',
                       'timing_precision':'audio output samples on session clock; excludes PTT lead and tail'}
                out=[old for old in out if old['id']!=event['id']]+[event]
    out=tx_timing.merge(structured,out)
    return out, f'{len(out)}/{max(len(keys),len(out))} TX intervals mapped from PACTOR session timing records. Direction is separate from PACTOR ISS/IRS roles; software timing.'
