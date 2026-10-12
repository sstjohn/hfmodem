import test from 'node:test';
import assert from 'node:assert/strict';
import {linkState,radioMode} from './link_state.mjs';
const call={t0:1,t1:2,kind:'connect->PEER',direction:'tx',status:'logged PTT interval'};
const answer={t0:3,t1:3.2,kind:'Robust answer 0',direction:'rx',status:'decoded control'};
const entry={t0:4,t1:4.8,kind:'P3 SL1 data',transmit_label:'SL1 ENTRY',level:1,stage:'P3 differential BPSK',carrier_hz:[1400,1600],direction:'tx',status:'CRC pass',frame_type:'data'};
const noise={t0:5,t1:5,kind:'PACTOR-2 carrier pair',direction:'rx',status:'signal candidate'};
const events=[call,answer,entry,noise];
test('seeking does not leak future phase or completed decode',()=>{
 assert.equal(linkState(events,0,'PACTOR').phase,'No link evidence yet');
 assert.equal(linkState(events,2.1,'PACTOR').phase,'Calling');
 assert.equal(linkState(events,3.3,'PACTOR').phase,'Peer answered · negotiating');
 assert.equal(linkState(events,4.2,'PACTOR').phase,'PACTOR-3 entry attempts');
 assert.equal(linkState(events,4.2,'PACTOR').activity,'Caller transmitting');
 assert.equal(linkState(events,6,'PACTOR').phase,'PACTOR-3 entry attempts');
 assert.equal(linkState(events,6,'PACTOR').last.rx,answer);
 assert.equal(linkState(events,0,'PACTOR').last.tx,null);
});
test('selection inspects whole frame, entry is never established data',()=>{
 const s=linkState(events,4,'PACTOR',entry);
 assert.equal(s.phase,'PACTOR-3 entry attempts');
 assert.match(s.mode,/SL1 · DBPSK · 2 carriers/);
 assert.equal(linkState(events,100,'PACTOR').activity,'Between observed transmissions');
 assert.equal(linkState(events,100,'PACTOR').age,95.2);
});
test('unconfirmed levels stay unconfirmed and P2 alphabets are explicit',()=>{
 assert.equal(radioMode({...entry,status:'CRC failed',transmit_label:''},'PACTOR'),'Unconfirmed signal');
 assert.match(radioMode({...entry,kind:'P2 SL3 data',transmit_label:'',protocol:'PACTOR-2',level:3},'PACTOR'),/8-DPSK/);
});
test('VARA setup is not data, acknowledgement and disconnect update phase',()=>{
 const e=[{...call,kind:'tx link-setup'},{...entry,t0:3,t1:4,kind:'connected-ack',transmit_label:'',frame_type:'observation'}, {...call,t0:5,t1:6,kind:'disconnect'}];
 assert.equal(linkState(e,2.5,'VARA').phase,'Link setup');
 assert.equal(linkState(e,4,'VARA').phase,'Link acknowledgement observed');
 assert.equal(linkState(e,6,'VARA').phase,'Disconnect observed');
});

test('recorded packet opens at start and points arrive on capture time',async()=>{
 const {packetAt,packetProgress,symbolVisible}=await import('./link_state.mjs');
 const e={...entry,views:{rx:{}}};
 assert.equal(packetAt([e],4.01),e);
 assert.equal(packetAt([e],3.99),null);
 assert.ok(Math.abs(packetProgress(e,4.4)-.5)<1e-10);
 assert.equal(symbolVisible(4.5,4.4),false);
 assert.equal(symbolVisible(4.3,4.4),true);
 assert.equal(symbolVisible(4.3,4.4,false,.05),false);
 assert.equal(symbolVisible(4.5,4.4,true),true);
});
test('open live burst has no progress denominator and delayed results stay pending',async()=>{
 const {packetAt,packetProgress}=await import('./link_state.mjs');
 const burst={t0:4,t1:null,kind:'Burst',status:'acquiring',views:{rx:{}}};
 assert.equal(packetAt([burst],10),burst);
 assert.equal(packetProgress(burst,10),null);
 const late={...entry,transmit_label:'',direction:'rx',available_at_s:7};
 assert.equal(linkState([late],6,'PACTOR').phase,'No link evidence yet');
 assert.equal(linkState([late],7,'PACTOR').phase,'Data frames observed');
});

test('host TX is known at keying, independent of decode and acquisition edges',()=>{
 const failed={...entry,t0:4.1,t1:4.8,burst_t0:4,burst_t1:4.9,status:'CRC failed',direction:'observed'};
 const s=linkState([failed],4.02,'PACTOR');
 assert.equal(s.activity,'Caller transmitting');
 assert.equal(s.phase,'PACTOR-3 entry attempts');
 assert.equal(s.last.tx.status,'logged PTT interval');
 assert.equal(s.mode,'Local TX · waveform not decoded');
 assert.equal(linkState([failed],4.88,'PACTOR').activity,'Caller transmitting');
 assert.equal(linkState([failed],3.9,'PACTOR').last.tx,null);
});

test('operator mode labels distinguish OFDM from its symbol modulation',async()=>{
 const {readFileSync}=await import('node:fs');
 const {setWaveforms,radioMode}=await import('./link_state.mjs');
 setWaveforms(JSON.parse(readFileSync(new URL('./waveforms.json',import.meta.url))));
 assert.match(radioMode({kind:'OFDM L12',level:12,status:'CRC pass'},'VARA',500),/OFDM · 16-QAM · 11 carriers/);
 assert.match(radioMode({kind:'Tone L10',level:10,status:'CRC pass'},'VARA',2750),/Multicarrier tones · DQPSK · 13 carriers/);
 assert.match(radioMode({kind:'P3 SL3 data',level:3,status:'CRC pass'},'PACTOR'),/Multicarrier · DBPSK · 14 carriers/);
 assert.equal(radioMode({kind:'OFDM L12',level:12,status:'CRC failed'},'VARA',500),'Unconfirmed signal');
 setWaveforms({});
});

test('ARDOP names its modulation while keeping header guesses provisional',()=>{
  assert.match(radioMode({protocol:'ARDOP',kind:'8PSK.500.100.E',stage:'8PSK',carrier_count:2,symbol_rate:100,status:'CRC pass'},'ARDOP'),/8PSK · 2 carriers · 100 Bd/);
  assert.match(radioMode({protocol:'ARDOP',stage:'16QAM',status:'header candidate'},'ARDOP'),/candidate/);
  assert.equal(linkState([{protocol:'ARDOP',kind:'ConAck500',t0:1,t1:2,status:'decoded control (RS or repetition check)',direction:'rx'}],3,'ARDOP').phase,'Peer answered · negotiating');
});

test('operator summaries wait for results and do not count local decodes as RX',async()=>{
 const {playbackSummary,evidenceClass,directionLabel,matchesEvent}=await import('./link_state.mjs');
 const rx={t0:1,t1:3,available_at_s:4,direction:'rx',status:'CRC pass after combining',kind:'Data',combined:true,attribution:'assumed peer'};
 assert.equal(playbackSummary([rx],3.5).verified,0);
 assert.equal(playbackSummary([rx],4).verified,1);
 assert.equal(playbackSummary([rx],4).combined,1);
 assert.equal(playbackSummary([{...rx,tx_confirmed:true}],4).verified,0);
 assert.equal(evidenceClass({status:'truncated body'}),'failed');
 assert.equal(evidenceClass({status:'partial robust field'}),'partial');
 assert.match(directionLabel(rx,{called:'PEER'}),/PEER \(assumed\)/);
 assert.match(directionLabel({...rx,tx_confirmed:true}),/Local TX/);
 assert.equal(matchesEvent(rx,'rx','verified','combining'),true);
 assert.equal(matchesEvent(rx,'tx'),false);
 assert.equal(matchesEvent({...rx,tx_confirmed:true},'tx'),true);
});


test('a local connection acknowledgement is not evidence of a peer answer',()=>{
 const ack={kind:'ConAck500',direction:'tx',transmit_label:'ConAck500',t0:1,t1:2,status:'logged PTT interval'};
 assert.equal(linkState([ack],1.1,'ARDOP').phase,'Local connection acknowledgement');
});

test('bytes preserve binary values, decoder timing, repeats and receiver boundaries',async()=>{
 const {byteRecords,trafficBytes,hexDump,readableBytes}=await import('./link_state.mjs');
 const p={hex:'00413cff0d0a',length:6,layer:'payload'};
 const e={id:'one',kind:'data',source_id:'rx',status:'CRC pass',t0:1,t1:2,available_at_s:3,payloads:[p],views:{tap:{decoded:[{status:'CRC failed',payloads:[p]}]}}};
 assert.equal(byteRecords(e,'rx')[0].hex,p.hex);
 assert.deepEqual(byteRecords(e,'tap'),[]);
 assert.deepEqual(byteRecords({...e,status:'CRC failed'}),[]);
 assert.equal(trafficBytes([e],2.9).length,0);
 assert.equal(trafficBytes([e,{...e,id:'repeat'}],3).length,2);
 assert.match(hexDump(p.hex),/00 41 3c ff 0d 0a/);
 assert.equal(readableBytes(p.hex),'\\x00A<\\xff␍\n');
});

test('VARA traffic shows DATA payload and keeps setup fields out of the stream',async()=>{
 const {byteRecords,trafficBytes}=await import('./link_state.mjs');
 const e={kind:'Tone L9',source_id:'rx',status:'CRC pass',t0:1,t1:2,
  payloads:[{hex:'46460d14f6000009821234',length:10,layer:'frame',payload_hex:'46460d',payload_length:3,payload_layer:'VARA DATA payload'}]};
 assert.equal(byteRecords(e)[0].hex,e.payloads[0].hex);
 assert.equal(trafficBytes([e],2)[0].hex,'46460d');
 assert.equal(trafficBytes([e],2)[0].length,3);
 assert.equal(trafficBytes([{...e,payloads:[{...e.payloads[0],payload_hex:null}]}],2).length,0);
});

test('exchange ladder hides future identities and qualifies repeated bytes',async()=>{
 const {exchangeRows}=await import('./link_state.mjs');
 const rx={id:'a',kind:'Data',status:'CRC pass',direction:'rx',t0:1,t1:2,available_at_s:3,packet_sha256:'same'};
 const repeat={...rx,id:'b',t0:4,t1:5,available_at_s:5};
 assert.equal(exchangeRows([rx,repeat],0).length,0);
 const active=exchangeRows([rx,repeat],1.5)[0];
 assert.equal(active.active,true);assert.equal(active.identified,false);assert.match(active.label,/pending/);
 assert.equal(exchangeRows([rx,repeat],2.5)[0].identified,false);
 assert.equal(exchangeRows([rx,repeat],5)[1].repeat,rx);
 assert.equal(exchangeRows([rx,{...repeat,direction:'tx'}],5)[1].repeat,undefined);
 assert.equal(exchangeRows([rx,repeat],0,true)[1].future,true);
 assert.equal(exchangeRows([{...rx,transmit_label:'tx connect',direction:'observed'}],1.5)[0].direction,'tx');
});

test('anatomy follows section boundaries and supports open live intervals',async()=>{
 const {packetAnatomy}=await import('./link_state.mjs');
 const e={t0:2,t1:3,anatomy:[{start_s:0,end_s:.2,label:'Header'},{start_s:.2,end_s:1,label:'Body'}]};
 assert.equal(packetAnatomy(e,2.1).current.label,'Header');
 assert.equal(packetAnatomy(e,2.2).current.label,'Body');
 assert.equal(packetAnatomy(e,3).current,null);
 assert.equal(packetAnatomy({t0:2,t1:null},10).progress,null);
 assert.equal(packetAnatomy({t0:2,t1:null},10).current.label,'Packet interval');
 assert.match(packetAnatomy({t0:0,t1:1},0).parts[0].basis,/not retained/);
});
