// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
let waveforms={};
export function setWaveforms(catalog){waveforms=catalog;}
export function waveformRows(protocol,bw){return Object.values(waveforms).filter(r=>protocol==='VARA'?r.protocol==='VARA'&&r.bandwidth_hz===bw:r.protocol.startsWith(protocol)).sort((a,b)=>a.protocol.localeCompare(b.protocol)||a.level-b.level);}
// Replay evidence only: no future events, no inferred delivery or negotiated rate.
const good = e => /^(CRC pass|decoded control|logged PTT)/.test(e.status || '');
const localTx = e => e.evidence_type==='ptt' || !!e.transmit_label || e.tx_confirmed===true;
const txStart = e => e.burst_t0 ?? e.t0;
const txEnd = e => e.burst_t1 ?? e.t1;
const text = e => [e.kind,e.transmit_label].filter(Boolean).join(' ');
export function radioMode(e, protocol, bw=null) {
  if (!e) return 'Not observed';
  if(e.tx_waveform?.modulation)e={...e,...e.tx_waveform};
  const s=text(e);
  const p=e.protocol||(/^P3 /.test(s)?'PACTOR-3':/^P2 /.test(s)?'PACTOR-2':/^P4 /.test(s)?'PACTOR-4':protocol);
  const level=e.level ?? Number(s.match(/(?:SL|L)(\d+)/)?.[1]||0);
  const row=waveforms[`${p}:${p==='VARA'?(e.bandwidth_hz||bw):0}:${level}`];
  if(row&&good(e)){return `${p} · ${p==='VARA'?'L':'SL'}${level} · ${row.family} · ${row.modulation} · ${row.carrier_count} carriers · ${Number(row.symbol_rate.toFixed(2))} Bd/carrier`;}
  if(localTx(e)&&e.modulation)return `${p}${e.level!=null?` · ${p==='VARA'?'L':'SL'}${e.level}`:''} · ${e.modulation}${e.carrier_count?` · ${e.carrier_count} carriers`:''}${e.symbol_rate?` · ${e.symbol_rate} Bd/carrier`:''} · transmitter record`;
  if(p==='ARDOP')return `ARDOP${e.status==='header candidate'?' · candidate':''} · ${e.modulation||e.stage||s}${e.carrier_count?` · ${e.carrier_count} carriers`:''}${e.symbol_rate?` · ${e.symbol_rate} Bd/carrier`:''}`;
  if(p==='PACTOR-4' && e.stage)return `PACTOR-4 · ${e.stage}`;
  if (!good(e)) return localTx(e)?'Local TX · waveform not decoded':'Unconfirmed signal';
  if (/P3 SL|PACTOR-3/.test(s+' '+p) && e.level != null)
    return `PACTOR-3 · SL${e.level} · ${e.stage?.includes('BPSK')?'DBPSK':'DQPSK'}${e.carrier_hz?.length?` · ${e.carrier_hz.length} carriers`:''} · 100 Bd/carrier`;
  if (/P2 SL/.test(s) && e.level != null)
    return `PACTOR-2 · SL${e.level} · ${['DBPSK','DQPSK','8-DPSK','16-DPSK'][e.level-1] || 'DPSK'} · 2 carriers · 100 Bd/carrier`;
  if (/P1 \d+ Bd/.test(s)) return `PACTOR-1 · ${s.match(/\d+ Bd/)[0]} · FSK`;
  if (/Robust answer|robust call|PACTOR-1.*(connect|CS)|connect->/.test(s)) return 'PACTOR-1 · FSK control';
  if (protocol==='VARA') {
    if (e.stage) return `VARA${e.level!=null?` · L${e.level}`:''} · ${e.stage}`;
    if (/Base.*L\d+/.test(s)) return `VARA · ${s.match(/L\d+/)[0]} · index modulation`;
    return 'VARA · control';
  }
  return `${p} · control`;
}
function transition(e, protocol) {
  if (!good(e)) return null;
  const s=text(e);
  if(protocol==='ARDOP' && /ConReq/.test(s))return ['Calling',1];
  if(protocol==='ARDOP' && /ConAck/.test(s))return [e.direction==='rx'&&!localTx(e)?'Peer answered · negotiating':'Local connection acknowledgement',2];
  if (/disconnect|link.closed|closing link/i.test(s)) return ['Disconnect observed',0];
  if (/ENTRY/i.test(s)) return ['PACTOR-3 entry attempts',3];
  if (/Robust answer/.test(s)) return ['Peer answered · negotiating',2];
  if (/connect-response/.test(s)) return ['Peer answered · negotiating',2];
  if (/connected-ack/.test(s)) return ['Link acknowledgement observed',4];
  if (/link-setup/.test(s)) return ['Link setup',3];
  if (/connect->|connect.request|\bCR\b|robust call/.test(s)) return ['Calling',1];
  if (e.frame_type==='data' && e.status.startsWith('CRC pass')) return ['Data frames observed',4];
  return null;
}
export function linkState(events,time,protocol,selected=null,bw=null) {
  // Decoded frames are known at their end. Selection explicitly inspects a whole frame.
  const host=events.filter(e=>localTx(e)&&txStart(e)<=time).map(e=>({...e,t0:txStart(e),t1:txStart(e),kind:e.transmit_label||e.kind,status:'logged PTT interval',frame_type:'observation',direction:'tx'}));
  const available=events.filter(e=>(e.available_at_s ?? e.t1 ?? Infinity)<=time || e===selected).concat(host).sort((a,b)=>a.t1-b.t1);
  let phase='No link evidence yet',rank=0,evidence=null;
  const last={tx:null,rx:null};
  for (const e of available) {
    if (good(e) && e.direction in last) last[e.direction]=e;
    const next=transition(e,protocol);
    if (next && (next[1]>=rank || next[1]===0)) { [phase,rank]=next; evidence=e; }
  }
  const transmitting=events.find(e=>localTx(e)&&txStart(e)<=time&&time<(txEnd(e)??Infinity));
  const active=selected || transmitting || events.filter(e=>e.t0<=time && time<(e.t1 ?? Infinity)).sort((a,b)=>Number(!!b.symbols)-Number(!!a.symbols))[0];
  const direction=active&&(localTx(active)?'tx':active.direction);
  const activity=active?(direction==='tx'?'Caller transmitting':direction==='rx'?'Called station received':'Signal · direction uncertain'):'Between observed transmissions';
  return {phase,activity,active,last,evidence,mode:active?radioMode(active,protocol,bw):'No current packet',age:evidence?Math.max(0,time-evidence.t1):null};
}

export function packetAt(events,time) {
  const usable=events.filter(e=>e.views || e.symbols || e.frame_type==='data');
  const active=usable.filter(e=>e.t0<=time && time<(e.t1 ?? Infinity));
  active.sort((a,b)=>Number(!!b.symbols)-Number(!!a.symbols) || Number(b.frame_type==='data')-Number(a.frame_type==='data'));
  return active[0] || usable.filter(e=>e.t1!=null&&e.t1<=time).at(-1) || null;
}
export function packetProgress(e,time) {
  return e?.t1!=null?Math.max(0,Math.min(1,(time-e.t0)/Math.max(.001,e.t1-e.t0))):null;
}
export function symbolVisible(t,time,whole=false,window=null) {
  return whole || (t<=time && (window===null || t>=time-window));
}
export function exchangeNote(events,time) {
  const past=events.filter(e=>(e.available_at_s ?? e.t1 ?? Infinity)<=time);
  const entries=past.filter(e=>good(e)&&/ENTRY/.test(text(e)));
  if(entries.length) {
    const start=entries[0].t0;
    const replies=past.filter(e=>e.t0>=start&&e.direction==='rx'&&good(e));
    const data=replies.filter(e=>e.frame_type==='data');
    return `${entries.length} entry transmission${entries.length===1?'':'s'} observed. ${data.length?`${data.length} peer data frame${data.length===1?'':'s'} decoded.`:replies.length?`${replies.length} peer control repl${replies.length===1?'y':'ies'} decoded; no peer data decoded yet.`:'No peer reply decoded since entry began.'}`;
  }
  const last=past.filter(good).at(-1);
  if(last?.status?.startsWith('CRC pass after'))return 'This frame needed combined evidence from retransmissions.';
  if(last?.direction==='tx')return 'Local transmit audio is available. A clean local constellation does not show what reached the peer.';
  return 'Compare the same packet across recordings to separate transmitted audio from receiver conditions.';
}

// Evidence categories describe the check that ran, never application delivery.
export function evidenceClass(e) {
  const s=e.status||'';
  if(s.startsWith('CRC pass'))return 'verified';
  if(/CRC failed|body decode failed|truncated body/.test(s))return 'failed';
  if(/partial|training lock|header-only control|header candidate/.test(s))return 'partial';
  if(s.startsWith('decoded control'))return 'control';
  if(localTx(e)||s.startsWith('logged PTT'))return 'tx';
  return 'candidate';
}
export function evidenceAvailable(e,time){return (e.available_at_s ?? e.t1 ?? Infinity)<=time;}
export function directionLabel(e,manifest={}) {
  const local=manifest.local_role==='called'?manifest.called:manifest.caller;
  const peer=manifest.local_role==='called'?manifest.caller:manifest.called;
  if(localTx(e)||e.direction==='tx')return `Local TX · ${local||'caller'}`;
  if(e.direction==='rx')return `RX · ${peer||manifest.call||'other station'}${/assumed/.test(e.attribution||'')?' (assumed)':''}`;
  return 'Unattributed signal';
}
export function playbackSummary(events,time) {
  const completed=events.filter(e=>evidenceAvailable(e,time));
  const inbound=completed.filter(e=>e.direction==='rx'&&!localTx(e));
  const verified=inbound.filter(e=>evidenceClass(e)==='verified');
  const latest=inbound.reduce((a,b)=>!a||b.t1>a.t1?b:a,null);
  return {tx:events.filter(e=>(localTx(e)||e.direction==='tx')&&txStart(e)<=time).length,
    verified:verified.length,combined:verified.filter(e=>e.combined||/combining/.test(e.status)).length,
    failed:inbound.filter(e=>evidenceClass(e)==='failed').length,
    partial:inbound.filter(e=>['partial','candidate'].includes(evidenceClass(e))).length,
    controls:inbound.filter(e=>evidenceClass(e)==='control').length,
    lastRxAge:latest?Math.max(0,time-latest.t1):null};
}
export function matchesEvent(e,direction='all',result='all',query='') {
  const way=localTx(e)?'tx':e.direction;
  return (direction==='all'||way===direction)&&
    (result==='all'||evidenceClass(e)===result)&&
    (!query||[e.kind,e.status,e.transmit_label,e.detail].join(' ').toLowerCase().includes(query.toLowerCase()));
}

export function byteRecords(event, source=null) {
  if(!event)return [];
  // A chosen receiver must stand on its own evidence; never borrow another's bytes.
  const rows=source&&source!==event.source_id?(event.views?.[source]?.decoded||[]):[event];
  return rows.filter(e=>(e.status||'').startsWith('CRC pass')).flatMap(e=>(e.payloads||[]).map((p,index)=>({
    ...p,index,kind:e.kind,status:e.status,t0:e.t0??event.t0,t1:e.t1??event.t1,
    available_at_s:e.available_at_s??e.t1??event.t1,combined:!!e.combined
  })));
}
export function hexDump(hex) {
  const bytes=hex.match(/../g)||[];
  return Array.from({length:Math.ceil(bytes.length/16)},(_,row)=>{
    const at=row*16,part=bytes.slice(at,at+16);
    const text=part.map(h=>{const n=parseInt(h,16);return n>=32&&n<=126?String.fromCharCode(n):'·';}).join('');
    return `${at.toString(16).padStart(4,'0')}  ${part.join(' ').padEnd(47)}  ${text}`;
  }).join('\n');
}
export function readableBytes(hex) {
  return (hex.match(/../g)||[]).map(h=>{const n=parseInt(h,16);return n===10?'\n':n===13?'␍':n===9?'⇥':n>=32&&n<=126?String.fromCharCode(n):`\\x${h}`;}).join('');
}
export function trafficBytes(events,time) {
  return events.flatMap(e=>byteRecords(e).filter(p=>evidenceAvailable(p,time)&&p.payload_hex!==null).map(p=>({
    ...p,hex:p.payload_hex??p.hex,length:p.payload_length??p.length,layer:p.payload_layer??p.layer,event:e})))
    .sort((a,b)=>a.available_at_s-b.available_at_s);
}

export function exchangeRows(events,time,whole=false) {
  const prior=new Map();
  return [...events].sort((a,b)=>a.t0-b.t0).filter(e=>whole||e.t0<=time).map(e=>{
    const ready=evidenceAvailable(e,time),identified=whole||ready;
    const direction=localTx(e)?'tx':e.direction||'observed';
    const key=identified&&e.status?.startsWith('CRC pass')&&e.packet_sha256?`${direction}:${e.packet_sha256}`:null;
    const repeat=key?prior.get(key):null;if(key)prior.set(key,e);
    return {event:e,direction,identified,ready,repeat,
      active:e.t0<=time&&time<(e.t1??Infinity),future:e.t0>time,
      label:identified?e.kind:direction==='tx'?(e.transmit_label||'Local transmission'):'Signal · identification pending',
      result:identified?e.status:direction==='tx'?'Local TX recorded':'Decode pending'};
  });
}
export function packetAnatomy(event,time) {
  if(!event)return {parts:[],current:null,progress:null};
  const duration=event.t1==null?null:Math.max(0,event.t1-event.t0);
  const raw=event.anatomy?.length?event.anatomy:[{start_s:0,end_s:duration,label:event.evidence_type==='ptt'?'Recorded keying interval':'Packet interval',basis:'Section boundaries were not retained in this replay.'}];
  const parts=raw.map(p=>({...p,start_s:Math.max(0,p.start_s),end_s:p.end_s==null?duration:Math.min(duration??Infinity,p.end_s)})).filter(p=>p.end_s==null||p.end_s>p.start_s);
  const position=Math.round((time-event.t0)*1e9)/1e9;
  return {parts,current:parts.find(p=>p.start_s<=position&&(p.end_s==null||position<p.end_s))||null,progress:packetProgress(event,time)};
}
