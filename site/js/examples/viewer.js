// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import {frequencyView,eventLabel,signalDescription,setWaveforms,waveformRows,linkState,radioMode,packetAt,packetProgress,symbolVisible,exchangeNote,evidenceClass,evidenceAvailable,directionLabel,playbackSummary,matchesEvent,byteRecords,hexDump,readableBytes,trafficBytes,exchangeRows,packetAnatomy} from './link-state.js?v=signal-1029-1';
import {installDeskEmbedding} from './desk-embed.mjs';
const $ = id => document.getElementById(id);
const state = {time:0, playing:false, selected:null, loop:null, events:[], cache:new Map(), tiles:new Map(), generation:0, eventPage:0};
let manifest, base, artifacts, lastTick=0, lastDraw=0;
let toneTiles=new WeakMap();
function audioOffset(){return manifest?.sources?.find(s=>s.id===$('listenSource').value)?.offset_s||0;}
function syncAudio(){const a=$('audio');const source=manifest?.sources?.find(s=>s.id===$('listenSource').value);const url=source?.url||manifest.recording.url;if(a.getAttribute('src')!==url)a.src=url;a.currentTime=Math.max(0,state.time+audioOffset());a.playbackRate=+$('speed').value;a.preservesPitch=false;}

function decodeRaster(value){return Uint8Array.from(atob(value),c=>c.charCodeAt(0));}
const seconds = t => `${t.toFixed(2)} s`;
const color = e => ({verified:'#59d9c0',failed:'#efb56f',partial:'#e6cc87',control:'#79a9ee',tx:'#b8a2ef',candidate:'#a8b9cd'})[evidenceClass(e)];
const senderColor = e => e.direction==='tx'?'#b8a2ef':e.direction==='rx'?'#59d9c0':'#a8b9cd';
async function json(url){const r=await fetch(url);if(!r.ok)throw Object.assign(Error(`Could not load ${url} (${r.status})`),{status:r.status});const data=await r.json();if(data.map_base64){data.map=decodeRaster(data.map_base64);delete data.map_base64;}return data;}
function memo(cache,key,loader,max){if(!cache.has(key)){const pending=loader().catch(error=>{if(cache.get(key)===pending)cache.delete(key);throw error;});cache.set(key,pending);while(cache.size>max)cache.delete(cache.keys().next().value);}return cache.get(key);}
function symbols(e){return memo(state.cache,e.symbols,()=>json(base+e.symbols),24);}
function context(id){const c=typeof id==='string'?$(id):id,r=c.getBoundingClientRect(),d=devicePixelRatio||1;const w=Math.round(r.width*d),h=Math.round(r.height*d);if(c.width!==w||c.height!==h){c.width=w;c.height=h;}const ctx=c.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,r.width,r.height);return [ctx,r.width,r.height];}
function view(){const span=+$('span').value;const start=Math.max(0,Math.min(state.time-span*.35,manifest.duration_s-span));return [start,Math.min(manifest.duration_s,start+span)];}
function setTime(t){state.time=Math.max(0,Math.min(manifest.duration_s,t));$('seek').value=state.time;$('clock').textContent=seconds(state.time);$('seek').setAttribute('aria-valuetext',`${seconds(state.time)} of ${seconds(manifest.duration_s)}`);}
function stop(){state.playToken=(state.playToken||0)+1;state.playing=false;state.preparing=false;$('play').setAttribute('aria-disabled','false');$('play').setAttribute('aria-busy','false');$('play').textContent='Play';$('audio').pause();}
function clearSelection(){state.selected=null;$('selection').textContent='Following playback';document.querySelectorAll('#events .selected').forEach(n=>n.classList.remove('selected'));}
async function play(){
 if(state.preparing)return;
 clearSelection();if(state.time>=manifest.duration_s)setTime(0);
 const requested=state.time,token=state.playToken=(state.playToken||0)+1;
 state.playing=false;state.preparing=true;$('play').setAttribute('aria-disabled','true');$('play').setAttribute('aria-busy','true');
 try {
  if($('listen').checked){
   const a=$('audio');syncAudio();
   if(a.readyState<1)await new Promise((resolve,reject)=>{
    const done=()=>{a.removeEventListener('loadedmetadata',ready);a.removeEventListener('error',failed);};
    const ready=()=>{done();resolve();},failed=()=>{done();reject(Error('Recording could not be loaded'));};
    a.addEventListener('loadedmetadata',ready);a.addEventListener('error',failed);a.load();
   });
   if(token!==state.playToken)return;
   const target=Math.max(0,requested+audioOffset());
   a.currentTime=target;
   if(a.seeking)await new Promise((resolve,reject)=>{
    const done=()=>{a.removeEventListener('seeked',ready);a.removeEventListener('error',failed);clearTimeout(timer);};
    const ready=()=>{done();resolve();},failed=()=>{done();reject(Error('Recording seek failed'));};
    const timer=setTimeout(()=>{done();reject(Error('Recording seek timed out'));},10000);
    a.addEventListener('seeked',ready);a.addEventListener('error',failed);
   });
   if(token!==state.playToken)return;
   if(Math.abs(a.currentTime-target)>.1)throw Error('The server did not seek to the requested recording position');
   await a.play();
   if(token!==state.playToken){a.pause();return;}
   if($('notice').textContent.startsWith('Audio playback failed:'))$('notice').textContent='';
  }
  state.playing=true;lastTick=0;$('play').textContent='Pause';
 } catch(e){if(token===state.playToken){stop();$('notice').textContent=`Audio playback failed: ${e.message}`;}}
 finally {if(token===state.playToken){state.preparing=false;$('play').setAttribute('aria-disabled','false');$('play').setAttribute('aria-busy','false');}}
}
function select(e){stop();state.loop=null;state.selected=e;setTime(e.t0);$('selection').textContent=`Inspecting ${e.id}`;document.querySelectorAll('#events button').forEach(n=>n.classList.toggle('selected',n.dataset.id===e.id));$('cloud').value='frame';draw();}
function fields(id,data){const key=JSON.stringify(data);if($(id).dataset.content===key)return;$(id).dataset.content=key;$(id).replaceChildren();for(const [k,v] of Object.entries(data)){if(v==null||v==='')continue;const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=String(v);$(id).append(dt,dd);}}
function details(e){
 fields('details',e?{Event:eventLabel(e),Waveform:radioMode(e,manifest.protocol,manifest.bw_hz),Result:e.status,Time:`${seconds(e.t0)} – ${e.t1==null?'Open':seconds(e.t1)}`,Direction:directionLabel(e,manifest),Details:e.detail}:{Selection:'Choose a frame, or play the recording.'});
 fields('metrics',e?{'Symbol SNR':e.snr_db!=null?`${e.snr_db.toFixed(1)} dB · decoder estimate`:'Not measured','Carrier offset':e.offset_hz!=null?`${e.offset_hz.toFixed(2)} Hz`:'Not measured','Modem quality':e.quality!=null?`${e.quality} / 100 · decoder score, not SNR`:null,Combining:e.combined||/combining|CRC pass after/.test(e.status)?'Decoded using multiple transmissions; dots show this transmission.':null}:{});
 fields('evidence',e?{Attribution:e.attribution||'Attribution basis not stored in this replay',Source:e.origin,'Timing precision':e.timing_precision||'See recording alignment below','Result available':`${seconds(e.available_at_s??e.t1??e.t0)}${e.available_at_s==null?' · frame-end proxy; decoder latency not recorded':''}`,'Original receiver log':e.original_observations?.join(' → '),'Session log':e.transmit_label,'Supporting evidence':e.observations?.map(o=>`${eventLabel(o)} · ${o.status} · ${o.origin}`).join('\n')}:{});
}
function filterEvents(reset=false){
 if(reset)state.eventPage=0;
 const buttons=[...document.querySelectorAll('#events button')];
 const visible=buttons.filter(b=>matchesEvent(state.events.find(e=>e.id===b.dataset.id),$('eventDirection').value,$('eventResult').value,$('eventSearch').value.trim()));
 const pages=Math.max(1,Math.ceil(visible.length/25));state.eventPage=Math.min(pages-1,state.eventPage);
 const shown=new Set(visible.slice(state.eventPage*25,(state.eventPage+1)*25));
 for(const b of buttons)b.hidden=!shown.has(b);
 const message=`${visible.length} of ${state.events.length} observations · page ${state.eventPage+1} of ${pages} · up to 25 per page`;
 if($('eventCount').textContent!==message)$('eventCount').textContent=message;
 $('eventsPrevious').setAttribute('aria-disabled',String(state.eventPage===0));$('eventsNext').setAttribute('aria-disabled',String(state.eventPage===pages-1));
}
function replaceReading(host,nodes){
 const focused=host.contains(document.activeElement)?document.activeElement:null;
 const old=[...host.querySelectorAll('button,a')],index=old.indexOf(focused),id=focused?.dataset.id;
 host.replaceChildren(...nodes);
 if(focused){const controls=[...host.querySelectorAll('button,a')];const same=id&&controls.find(b=>b.dataset.id===id);const next=same||controls[Math.max(0,Math.min(index,controls.length-1))]||host;next.tabIndex=next===host?-1:next.tabIndex;next.focus({preventScroll:true});}
}
function signalText(e,data,file,frame=e){
 const description=$('signal-description'),link=$('signal-data'),values=$('signal-values');
 if(!e||!data){delete values.dataset.key;description.textContent='No packet signal retained at this position; unidentified audio is unknown.';link.hidden=true;values.textContent='';return;}
 const key=base+file;if(values.dataset.key===key)return;values.dataset.key=key;
 description.textContent=signalDescription(e,frame,manifest);
 link.href=base+file;link.hidden=false;
 if(data.spectrum){values.textContent=`Stored audio RMS: ${data.rms_dbfs} dBFS; strongest bin: ${data.peak_dbfs} dBFS.\nAverage power spectrum (frequency Hz: dBFS), 0–3000 Hz:\n`+data.spectrum.map((v,i)=>`${(i*3000/(data.bins-1)).toFixed(1)}: ${v.toFixed(2)}`).join('\n');}
 else {values.textContent=`${data.stage||'Decoded symbols'} · ${data.normalization||'stored decoder coordinates'}\nTime seconds | carrier Hz | in-phase | quadrature\n`+data.i.map((v,i)=>`${data.t[i]} | ${data.carrier[i]} | ${v} | ${data.q[i]}`).join('\n');}
}
function sessionOverview(){
 $('waterfallLocal').textContent=`${manifest.caller||'Local station'} · TX`;
 $('waterfallPeer').textContent=`${manifest.called||manifest.call||'Peer'} · RX`;
 $('sessionOutcome').textContent=manifest.comparison?.original_summary||'Original session outcome not stored in this replay.';
 const session=manifest.session||{};
 $('sessionFacts').textContent=[`${manifest.caller||'Local station'} → ${manifest.called||manifest.call} · local station is the caller`,session.dial_hz!=null?`${(session.dial_hz/1e6).toFixed(4)} MHz dial`:null,session.tx_w!=null?`${session.tx_w} W configured`:null,manifest.bw_hz?`${manifest.bw_hz} Hz bandwidth`:null,`${manifest.sources?.length||1} recording sources`,`${seconds(manifest.duration_s)} primary recording`,manifest.outcome_note||'Mail delivery comes from the original session log; replay CRCs do not establish delivery.'].filter(Boolean).join(' · ');
}
function timeline(){
const [c,w,h]=context('timeline'),duration=manifest.duration_s||1,left=82,pw=w-left;
const labels=['Local / TX','Peer / RX'];
for(let j=0;j<labels.length;j++){c.fillStyle='#172335';c.fillRect(left,12+j*25,pw,18);c.fillStyle='#a8b9cd';c.font='10px system-ui';c.fillText(labels[j],2,25+j*25);}
for(const e of state.events){const lane=e.direction==='tx'?0:e.direction==='rx'?1:2;const x=left+e.t0/duration*pw,ww=Math.max(3,(e.t1-e.t0)/duration*pw);c.globalAlpha=(state.selected && state.selected !== e) ? .35 : .85;
if((e.evidence_type==='ptt'||e.kind==='PTT / transmit')){c.strokeStyle='#9a88cb';c.lineWidth=1;c.strokeRect(x,11,ww,20);}else{c.fillStyle=color(e);c.fillRect(x,12+lane*25,ww,18);}
if(state.selected===e){c.globalAlpha=1;c.strokeStyle='#ffce64';c.lineWidth=2;c.strokeRect(x-1,10+lane*25,ww+2,22);}}
c.globalAlpha=1;c.lineWidth=1;const [lo,hi]=view();c.strokeStyle='#a9bed877';c.strokeRect(left+lo/duration*pw,3,(hi-lo)/duration*pw,88);c.fillStyle='#ffffff';c.fillRect(left+state.time/duration*pw,0,1,h);
}
async function waterfall(generation){const [lo,hi]=view(),wf=manifest.waterfall,frequency=frequencyView(manifest,$('frequencyRange').value);const tiles=wf.tiles.filter(t=>(t.first+t.rows)*wf.hop_s>=lo&&t.first*wf.hop_s<=hi);const results=await Promise.all(tiles.map(t=>memo(state.tiles,t.file,async()=>{const r=await fetch(base+t.file);if(!r.ok)throw Error('Waterfall tile unavailable');const bytes=decodeRaster((await r.json()).raster_base64);if(bytes.length!==t.rows*wf.bins)throw Error('Invalid waterfall tile');const img=new ImageData(t.rows,wf.bins);for(let row=0;row<t.rows;row++)for(let bin=0;bin<wf.bins;bin++){const v=bytes[row*wf.bins+bin]/255;const i=((wf.bins-1-bin)*t.rows+row)*4;const rgb=window.HFSignalPalette?.rgb(v)||[Math.min(255,Math.max(0,(v-.45)*400)),Math.min(255,v*v*360),Math.min(255,20+v*220)];[img.data[i],img.data[i+1],img.data[i+2]]=rgb;img.data[i+3]=255;}const tile=document.createElement('canvas');tile.width=t.rows;tile.height=wf.bins;tile.getContext('2d').putImageData(img,0,0);tile.values=bytes;return tile;},16)));
if(generation!==state.generation)return;
$('waterfall').dataset.frequencyLow=frequency.low;$('waterfall').dataset.frequencyHigh=frequency.high;
$('waterfall-text').textContent=`${frequency.low}–${frequency.high} Hz audio · fixed −100 to 0 dBFS scale · ${$('frequencyRange').value==='signal'?'nominal signal band, centered at '+(manifest.audio_center_hz??1500)+' Hz':'full retained spectrum; receiver filtering is visible in the recording'} · outlines identify the sender; brightness shows recorded power.`;
if($('waterfall-values').open){
 const lines=['Time seconds | strongest bin Hz | dBFS'];
 for(let i=0;i<tiles.length;i++){const tile=tiles[i],bytes=results[i].values;
  for(let row=0;row<tile.rows;row++){const time=(tile.first+row)*wf.hop_s;if(time<lo||time>hi)continue;
   let peak=frequency.firstBin;for(let bin=frequency.firstBin+1;bin<=frequency.lastBin;bin++)if(bytes[row*wf.bins+bin]>bytes[row*wf.bins+peak])peak=bin;
   lines.push(`${time.toFixed(2)} | ${(peak*wf.hz_per_bin).toFixed(2)} | ${(wf.db_min+bytes[row*wf.bins+peak]/255*(wf.db_max-wf.db_min)).toFixed(2)}`);
  }
 }
 const text=lines.join('\n');if($('waterfall-peaks').textContent!==text)$('waterfall-peaks').textContent=text;
}
const [c,w,h]=context('waterfall');const left=48,top=12,pw=w-left-8,ph=h-38;c.save();c.beginPath();c.rect(left,top,pw,ph);c.clip();tiles.forEach((t,i)=>{c.drawImage(results[i],0,frequency.sourceTop,t.rows,frequency.sourceHeight,left+(t.first*wf.hop_s-lo)/(hi-lo)*pw,top,t.rows*wf.hop_s/(hi-lo)*pw,ph);});
for(const e of state.events){
 if(e.timestamp_only||e.t1<=e.t0)continue;
 const start=e.burst_t0??e.t0,end=e.burst_t1??e.t1;
 if(end<=lo||start>=hi)continue;
 const x=left+(start-lo)/(hi-lo)*pw,width=Math.max(2,(end-start)/(hi-lo)*pw),ink=senderColor(e);
 c.fillStyle=ink+'12';c.fillRect(x,top+3,width,ph-6);
 c.strokeStyle='#080b10';c.lineWidth=3;c.strokeRect(x,top+3,width,ph-6);
 c.strokeStyle=ink;c.lineWidth=1;c.strokeRect(x,top+3,width,ph-6);
 // Separate rail positions also distinguish senders without relying on hue.
 c.fillStyle=ink;c.fillRect(x,e.direction==='tx'?top+4:top+ph-8,width,4);
}
if(state.selected){
const x0=left+(state.selected.t0-lo)/(hi-lo)*pw,x1=Math.max(x0+3,left+(state.selected.t1-lo)/(hi-lo)*pw);
c.fillStyle='rgba(3,8,16,.55)';c.fillRect(left,top,Math.max(0,x0-left),ph);c.fillRect(x1,top,Math.max(0,left+pw-x1),ph);
c.lineWidth=6;c.strokeStyle='#080b10';c.strokeRect(x0,top+2,x1-x0,ph-4);
c.lineWidth=2;c.strokeStyle='#ffce64';c.strokeRect(x0,top+2,x1-x0,ph-4);
for(const x of [x0,x1]){c.fillStyle='#ffce64';c.beginPath();c.moveTo(x-6,top);c.lineTo(x+6,top);c.lineTo(x,top+10);c.closePath();c.fill();}
const text=eventLabel(state.selected);c.font='bold 10px system-ui';const tw=c.measureText(text).width+14,bx=Math.max(left+4,Math.min(x0+6,left+pw-tw-4));c.fillStyle='#ffce64';c.fillRect(bx,top+12,tw,20);c.fillStyle='#131924';c.fillText(text,bx+7,top+26);
}else{c.fillStyle='#fff';c.fillRect(left+(state.time-lo)/(hi-lo)*pw,top,1,ph);}c.lineWidth=1;c.restore();c.font='10px monospace';c.fillStyle='#9dacbf';for(let k=0;k<=4;k++)c.fillText(`${Math.round(frequency.high-(frequency.high-frequency.low)*k/4)}`,4,top+k*ph/4+4);for(let k=0;k<=4;k++)c.fillText(`${(lo+(hi-lo)*k/4).toFixed(1)}s`,left+k*pw/4-12,h-8);}
function sourceName(id){const s=manifest.sources?.find(s=>s.id===id);return {'rig rx':'Receiver input','rig tap':'TX monitor tap','sdr witness':'SDR witness'}[s?.kind]||s?.kind||'Recording';}
function jsonCached(file){return memo(state.cache,file,()=>json(base+file),48);}
function plotConstellation(canvas,data,whole=!!state.selected,window=null,carrier='all'){
 const [c,w,h]=context(canvas),phaseOnly=data.normalization?.includes('phase-only'),scale=Math.min(w,h)*(phaseOnly?.25:.30),cx=w/2,cy=phaseOnly?h*.38:h/2;
 c.strokeStyle='#24364c';c.lineWidth=1;
 for(let i=-1;i<=1;i++){c.beginPath();c.moveTo(cx+i*scale,14);c.lineTo(cx+i*scale,phaseOnly?h*.72:h-14);c.moveTo(14,cy+i*scale);c.lineTo(w-14,cy+i*scale);c.stroke();}
 c.font='10px monospace';c.fillStyle='#a8b9cd';c.fillText('I',w-16,cy-7);c.fillText('Q',cx+7,18);
 for(const [i,q]of data.ideal){c.strokeStyle='#d6e0ea';c.strokeRect(cx+i*scale-4,cy-q*scale-4,8,8);}
 const histogram=Array(72).fill(0);let count=0;
 for(let j=0;j<data.i.length;j++){
  if(!symbolVisible(data.t[j],state.time,whole,window)||(carrier!=='all'&&data.carrier[j]!==+carrier))continue;
  const age=state.time-data.t[j];c.fillStyle=!whole&&age>=0&&age<.07?'#dbfff5':phaseOnly?'#66ecd135':'#66ecd17a';
  c.fillRect(cx+data.i[j]*scale-1,cy-data.q[j]*scale-1,2,2);count++;
  const phase=Math.atan2(data.q[j],data.i[j]);histogram[Math.min(71,Math.floor((phase+Math.PI)/(2*Math.PI)*72))]++;
 }
 if(phaseOnly){
  const left=26,pw=w-52,baseline=h-26,height=h*.16,peak=Math.max(1,...histogram);
  c.font='10px system-ui';c.fillStyle='#a8b9cd';c.fillText('Differential phase · amplitude removed',left,baseline-height-12);
  c.fillStyle='#59d9c0';histogram.forEach((n,j)=>c.fillRect(left+j/72*pw,baseline-n/peak*height,Math.max(1,pw/72-1),n/peak*height));
  c.strokeStyle='#c6d2e3';c.setLineDash([2,3]);
  for(const [i,q] of data.ideal){const x=left+(Math.atan2(q,i)+Math.PI)/(2*Math.PI)*pw;c.beginPath();c.moveTo(x,baseline-height);c.lineTo(x,baseline);c.stroke();}
  c.setLineDash([]);c.font='9px monospace';c.fillStyle='#9dacbf';c.fillText('−180°',left,baseline+14);c.fillText('0°',left+pw/2-5,baseline+14);c.fillText('+180°',left+pw-30,baseline+14);
 }
 return count;
}
function plotSignal(canvas,data,mode,progress=null){const [c,w,h]=context(canvas),left=38,top=24,pw=w-left-12,ph=h-88;const floor=mode==='spectrum'?-100:Math.max(-100,data.peak_dbfs-55),ceil=mode==='spectrum'?0:Math.max(floor+20,data.peak_dbfs);c.font='10px monospace';c.fillStyle='#8ea9c1';
if(mode==='spectrum'){for(let j=0;j<=3;j++){const y=top+j*ph/3;c.strokeStyle='#203248';c.beginPath();c.moveTo(left,y);c.lineTo(left+pw,y);c.stroke();c.fillText(String(Math.round(ceil-j*(ceil-floor)/3)),2,y+3);}c.beginPath();data.spectrum.forEach((db,k)=>{const x=left+k/(data.bins-1)*pw,y=top+(1-Math.max(0,Math.min(1,(db-floor)/(ceil-floor))))*ph;k?c.lineTo(x,y):c.moveTo(x,y);});c.strokeStyle='#64e6ce';c.lineWidth=1.6;c.stroke();c.lineTo(left+pw,top+ph);c.lineTo(left,top+ph);const fill=c.createLinearGradient(0,top,0,top+ph);fill.addColorStop(0,'#3ab9cb66');fill.addColorStop(1,'#3ab9cb00');c.fillStyle=fill;c.fill();c.fillStyle='#8ea9c1';for(let j=0;j<=3;j++)c.fillText(`${j}k`,left+j*pw/3-8,top+ph+14);c.fillText('dBFS',left,14);
}else{let tile=toneTiles.get(data);if(!tile){const img=new ImageData(data.rows,data.bins);for(let row=0;row<data.rows;row++)for(let bin=0;bin<data.bins;bin++){const db=data.map[row*data.bins+bin]/2.55-100,v=Math.max(0,Math.min(1,(db-floor)/(ceil-floor))),i=((data.bins-bin-1)*data.rows+row)*4;const rgb=window.HFSignalPalette?.rgb(v)||[Math.max(0,(v-.5)*480),Math.min(255,v*v*390),Math.min(255,25+v*200)];[img.data[i],img.data[i+1],img.data[i+2]]=rgb;img.data[i+3]=255;}tile=document.createElement('canvas');tile.width=data.rows;tile.height=data.bins;tile.getContext('2d').putImageData(img,0,0);toneTiles.set(data,tile);}c.drawImage(tile,left,top,pw,ph);for(let j=0;j<=3;j++)c.fillText(`${3-j}k`,8,top+j*ph/3+3);c.fillText('0 s',left,top+ph+14);c.fillText(`${data.duration_s.toFixed(2)} s`,left+pw-48,top+ph+14);c.fillText(`Detail gain · ${floor.toFixed(0)} to ${ceil.toFixed(0)} dBFS`,left,14);}
const y0=h-29;c.strokeStyle='#a390f3';c.lineWidth=1.4;c.beginPath();data.envelope.forEach((db,j)=>{const x=left+j/(data.envelope.length-1)*pw,y=y0-Math.max(0,Math.min(1,(db+100)/100))*25;j?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();c.fillStyle='#8ea9c1';c.fillText('RMS envelope · fixed −100…0 dBFS',left,h-7);
if(progress!==null&&mode!=='spectrum'){const x=left+Math.max(0,Math.min(1,progress))*pw;c.fillStyle='#0a101a80';c.fillRect(x,top,left+pw-x,ph);c.strokeStyle='#ffda85';c.lineWidth=1.5;c.beginPath();c.moveTo(x,top);c.lineTo(x,y0+2);c.stroke();c.fillStyle='#ffda85';c.beginPath();c.moveTo(x-4,top);c.lineTo(x+4,top);c.lineTo(x,top+6);c.fill();}}
function toneProgress(e,data,source){if(state.selected)return null;const shift=manifest.sources?.find(s=>s.id===source)?.offset_s||0;const lo=Math.max(-shift,e.views?.[source]?.context?e.t0-(e.context_before_s||.15):e.t0);return (state.time-lo)/data.duration_s;}

async function comparisons(e,generation){
 const host=$('comparisons');if(!$('compare').checked){host.replaceChildren();state.comparisonKey=null;return;}
 const sources=manifest.sources||[];$('alignmentNote').textContent=sources.filter(s=>s.offset_s!==0).map(s=>`${sourceName(s.id)}: ${s.alignment}`).join(' ')+' Levels refer to the stored audio; receiver gain and SDR normalization can differ.';
 if(!e?.views){host.replaceChildren();state.comparisonKey=null;return;}
 const key=base+e.id+$('signalMode').value;
 if(state.comparisonKey===key){for(const r of state.comparisonCards){if(r.points)plotConstellation(r.canvas,r.points);else plotSignal(r.canvas,r.data,$('signalMode').value,toneProgress(e,r.data,r.source.id));}return;}
 const cards=await Promise.all(sources.filter(s=>e.views[s.id]).map(async source=>{const view=e.views[source.id],data=await jsonCached(view.signal),frame=view.frames.find(f=>f.status==='CRC pass')||view.frames[0];const points=frame&&$('signalMode').value==='auto'?await symbols(frame):null;return {source,view,data,frame,points};}));if(generation!==state.generation)return;
 state.comparisonKey=key;state.comparisonCards=[];host.replaceChildren(...cards.map(({source,view,data,frame,points})=>{const card=document.createElement('article');card.className='receiver';const title=document.createElement('h3');title.textContent=sourceName(source.id)+(source.id===e.preferred_source?' · preferred':'');const link=document.createElement('a');link.href=source.url;link.target='_blank';link.rel='noopener';link.textContent=title.textContent;link.title=source.alignment;title.replaceChildren(link);const note=document.createElement('p');note.className='caption';note.textContent=frame?`${radioMode(frame,manifest.protocol,manifest.bw_hz)} · rehear result: ${frame.status}${view.frames.length>1?` · ${view.frames.length} symbol frames in this interval`:''}`:view.decoded?.length?view.decoded.map(d=>`${d.kind} · ${d.status}`).join(' / '):'Measured tone / frequency energy · no decoded constellation';const stats=document.createElement('span');stats.className='stats';stats.textContent=`${data.rms_dbfs} whole-packet recording dBFS RMS · peak bin ${data.peak_dbfs} dBFS`;const canvas=document.createElement('canvas');canvas.setAttribute('role','img');canvas.setAttribute('aria-label',sourceName(source.id)+' packet signal');card.append(title,stats,canvas,note);state.comparisonCards.push({canvas,points,data,source});requestAnimationFrame(()=>{if(card.isConnected){if(points)plotConstellation(canvas,points);else plotSignal(canvas,data,$('signalMode').value,toneProgress(e,data,source.id));}});return card;}));
}
async function cloud(e,generation){
const chosen=$('signalSource').value==='auto'?e?.preferred_source:$('signalSource').value;
const view=e?.views?.[chosen];
const frames=view?.frames||[];
const frame=frames.find(f=>f.status.startsWith('CRC pass'))||frames[0]||(chosen===e?.source_id?e:null);
$('plotBasis').textContent=chosen?`${sourceName(chosen)} · ${frame?.status||view?.decoded?.map(d=>d.status).join(' / ')||'No decoded symbols; measured audio only'} · whole-frame replay evidence`:'';
const whole=!!state.selected, rolling=!whole&&$('cloud').value==='recent';
if(frame?.symbols&&$('signalMode').value==='auto'){
 const data=await symbols(frame);if(generation!==state.generation)return;
 $('plotBasis').textContent+=data.normalization?` · ${data.normalization}`:'';
 const carriers=[...new Set(data.carrier)],identity=carriers.join(',');
 $('carrier').disabled=false;if($('carrier').dataset.identity!==identity){$('carrier').dataset.identity=identity;const prior=$('carrier').value;$('carrier').replaceChildren(new Option('All','all'),...carriers.map(b=>new Option(`${b.toFixed(1)} Hz`,String(b))));if(prior==='all'||carriers.includes(+prior))$('carrier').value=prior;}
 let points=data;
 if(rolling){const others=state.events.filter(f=>f!==e&&f.t0<=state.time&&f.t1>=state.time-+$('window').value).flatMap(f=>f.views?.[chosen]?.frames||[]).filter(f=>f.level===frame.level&&f.stage===frame.stage&&f.symbols!==frame.symbols);const unique=[...new Map(others.map(f=>[f.symbols,f])).values()];const batches=await Promise.all(unique.map(symbols));if(generation!==state.generation)return;points={...data};for(const k of ['i','q','t','carrier'])points[k]=[...data[k],...batches.flatMap(b=>b[k])];}
 const count=plotConstellation($('constellation'),points,whole,rolling?+$('window').value:null,$('carrier').value);
 signalText(e,data,frame.symbols,frame);
 $('signalTitle').textContent=data.normalization?.includes('phase-only')?'Differential phase':'Constellation';$('cloudnote').textContent=`${sourceName(chosen)} · ${data.stage}${data.normalization?.includes('phase-only')?' · unit radius; phase spread only':''} · ${count} symbols${whole?' · whole frame':rolling?' · recent history':' · arriving with audio'}`;
 return;
}
$('carrier').disabled=true;
if(view){const data=await jsonCached(view.signal);if(generation!==state.generation)return;signalText(e,data,view.signal);plotSignal($('constellation'),data,$('signalMode').value,toneProgress(e,data,chosen));$('signalTitle').textContent=$('signalMode').value==='spectrum'?'Packet spectrum · whole frame':'Tone map & envelope';$('cloudnote').textContent=`${sourceName(chosen)} · ${data.rms_dbfs} dBFS whole-packet RMS (stored audio, not RF power)${view.context?' · detection context':''} · ${$('signalMode').value==='spectrum'?'Average over this packet':whole?'Whole packet':'Playhead follows audio; upcoming tones are dimmed'}`;return;}
signalText(null);context('constellation');$('signalTitle').textContent='Packet signal';$('cloudnote').textContent='No packet at this position.';
}
function packetCaption(e){
const whole=!!state.selected,active=e&&e.t0<=state.time&&state.time<e.t1;
$('packetName').textContent=e?eventLabel(e):'Waiting for the first observed packet';
$('packetMode').textContent=e?radioMode(e,manifest.protocol,manifest.bw_hz):'';
$('packetResult').textContent=e?`${whole||!active?'Rehear result':'Hindsight result'}${e.source_id?' · '+sourceName(e.source_id):''}: ${e.status}`:'';
$('packetPosition').textContent=e?whole?'Whole frame':active?`${Math.max(0,state.time-e.t0).toFixed(2)} / ${(e.t1-e.t0).toFixed(2)} s`:`Ended ${Math.max(0,state.time-e.t1).toFixed(1)} s ago`:'';
$('packetFill').style.width=`${whole?100:packetProgress(e,state.time)*100}%`;
$('replayFrame').disabled=!e;$('replayFrame').textContent=state.loop?'Stop loop':'Loop this packet';
$('exchangeNote').textContent=exchangeNote(state.events,state.time);
state.currentPacket=e;
}
function showLinkState(){
const s=linkState(state.events,state.time,manifest.protocol,null,manifest.bw_hz);
$('linkPhase').textContent=s.phase;$('linkActivity').textContent=s.active?directionLabel(s.active,manifest):s.activity;
$('linkMode').textContent=s.active&&!evidenceAvailable(s.active,state.time)?(directionLabel(s.active,manifest).startsWith('Local TX')?(s.active.tx_waveform?radioMode({...s.active,status:'acquiring'},manifest.protocol,manifest.bw_hz):'Local TX · waveform decode pending'):'Receiving · decode pending'):s.mode;
const counts=playbackSummary(state.events,state.time);$('playbackCounts').textContent=`RX observations so far: ${counts.verified} CRC verified (${counts.combined} combined), ${counts.failed} failed, ${counts.partial} provisional, ${counts.controls} checked controls. Gaps mean no identified packet, not necessarily silence.`;
for(const [direction,id] of [['tx','linkTx'],['rx','linkRx']]){const e=s.last[direction];$(id).textContent=e?`${radioMode(e,manifest.protocol,manifest.bw_hz)} · ${seconds(e.t0)}`:'Not observed';}
$('linkEvidence').textContent=s.evidence?`Playback · ${eventLabel(s.evidence)} · ${s.evidence.status} · ${seconds(s.evidence.t1)}${s.age>0?` · ${(s.age).toFixed(1)} s since evidence`:''}. Speed levels describe individual transmissions.`:'No completed link-state evidence at this point. Signal candidates do not establish a link or speed level.';
}
function showBytes(e){
 const source=$('signalSource').value==='auto'?e?.preferred_source:$('signalSource').value;
 const rows=byteRecords(e,source),whole=!!state.selected;
 const visible=rows.filter(p=>whole||evidenceAvailable(p,state.time));
 const key=[base,e?.id,source,$('byteMode').value,whole,visible.length].join('|');
 if(key!==state.byteKey){state.byteKey=key;
  $('byteNote').textContent=!manifest.byte_schema?'This older replay did not retain bytes. Rehear this QSO to capture them.':rows.length&&!visible.length?'Decoding this packet. Bytes appear when its checked result becomes available.':visible.length?`${sourceName(source)} · ${whole?'whole-frame inspection; may be ahead of playback':'available at the playhead'} · checked decoder bytes, not an acknowledged application stream.`:'No CRC-verified bytes for this packet in the chosen recording. Control fields remain in frame inspection.';
  $('packetBytes').replaceChildren(...visible.map(p=>{const box=document.createElement('article'),title=document.createElement('p'),pre=document.createElement('pre'),download=document.createElement('button');
   pre.tabIndex=0;pre.setAttribute('role','region');pre.setAttribute('aria-label','Decoded packet bytes; scroll horizontally');
   const raw=$('byteMode').value==='frame',data=raw?p.hex:(p.payload_hex??p.hex),length=data.length/2,layer=raw?p.layer:(p.payload_layer??p.layer);
   title.className='caption';title.textContent=`${length} bytes · ${layer}${p.payload_hex!=null&&!raw?' · frame fields, CRC and padding removed':''}${p.sequence!=null?' · sequence '+p.sequence:''}${p.status_byte!=null?' · status 0x'+p.status_byte.toString(16).padStart(2,'0'):''}${p.combined?' · combined receptions':''}`;
   const hex=data.slice(0,16384);pre.textContent=($('byteMode').value==='text'?readableBytes(hex):hexDump(hex))+(data.length>hex.length?'\n… display limited to 8192 bytes; download contains all bytes.':'');
   download.textContent=raw?'Download decoder bytes':'Download bytes';download.onclick=()=>{const bytes=Uint8Array.from(data.match(/../g)||[],h=>parseInt(h,16));const url=URL.createObjectURL(new Blob([bytes],{type:'application/octet-stream'}));const a=document.createElement('a');a.href=url;a.download=`${manifest.arm}-${e.id}-${source}-${p.index}${raw?'-frame':''}.bin`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};box.append(title,pre,download);return box;}));
 }
 const traffic=trafficBytes(state.events,state.time),trafficKey=base+'|'+traffic.length;
 if(trafficKey!==state.trafficKey){state.trafficKey=trafficKey;
  $('trafficNote').textContent=`${traffic.length} checked blocks · ${traffic.reduce((n,p)=>n+p.length,0)} payload bytes including repeats. VARA frame fields, CRC and padding are removed where retained. Each packet uses its recorded source; repeats are not deduplicated and compressed mail remains binary. Nonprinting bytes use escapes. Showing the latest 100 blocks.`;
  replaceReading($('byteTraffic'),traffic.slice(-100).map(p=>{const b=document.createElement('button');b.textContent=`${seconds(p.available_at_s)} · ${directionLabel(p.event,manifest)} · ${sourceName(p.event.source_id)} · ${p.length} B · ${readableBytes(p.hex.slice(0,160)).replace(/\n/g,' ↵ ')}${p.hex.length>160?'…':''}`;b.onclick=()=>select(p.event);return b;}));
  $('byteTraffic').scrollTop=$('byteTraffic').scrollHeight;
 }
}
function showExchange(){
 const whole=$('exchangeScope').value==='whole';const rows=exchangeRows(state.events,state.time,whole);
 const key=base+'|'+whole+'|'+rows.map(r=>`${r.event.id}:${r.identified}:${r.active}:${r.future}`).join(',');
 $('ladderLocal').textContent=`${manifest.caller||'Local station'} · local`;$('ladderPeer').textContent=`${manifest.called||manifest.call||'Peer'} · called`;
 $('ladderNote').textContent=(whole?'Whole recording · includes future results. ':'Evidence up to the playhead. ')+'Showing the latest 40 packets in this view; use Frames and observations for the full list. Arrows show direction, not confirmed delivery. ACKs stand alone unless a response relationship was recorded.';
 if(key!==state.ladderKey){state.ladderKey=key;
  const host=$('ladder'),oldScroll=host.scrollTop;
  replaceReading(host,rows.slice(-40).map(r=>{const e=r.event,b=document.createElement('button'),t=document.createElement('time'),flight=document.createElement('span'),note=document.createElement('small');
   b.dataset.id=e.id;b.className=[r.direction,r.active?'active':'',r.future?'future':''].join(' ');
   t.textContent=seconds(e.t0);flight.className='flight';flight.textContent=r.identified?eventLabel(e):r.label;flight.style.color=r.identified?color(e):'#cad7e5';
   note.textContent=[r.result,r.identified?radioMode(e,manifest.protocol,manifest.bw_hz):'',e.direction==='rx'&&/assumed/.test(e.attribution||'')?'Peer attribution assumed':'',r.repeat?`Same checked bytes as ${seconds(r.repeat.t0)}; retry relationship unconfirmed`:''].filter(Boolean).join(' · ');
   b.append(t,flight,note);b.onclick=()=>select(e);return b;}));
  host.scrollTop=oldScroll;
  if(state.playing){const row=host.querySelector('.active')||host.lastElementChild;if(row)host.scrollTop=Math.max(0,row.offsetTop-host.offsetTop-host.clientHeight/2);}
 }
 for(const b of $('ladder').children)b.classList.toggle('selected',b.dataset.id===state.selected?.id);
 if(state.selected&&state.ladderSelection!==state.selected.id){const host=$('ladder'),row=host.querySelector('.selected');if(row)host.scrollTop+=row.getBoundingClientRect().top-host.getBoundingClientRect().top-host.clientHeight/2;}
 state.ladderSelection=state.selected?.id;
}
function showAnatomy(e){
 const a=packetAnatomy(e,state.time),duration=e?.t1!=null?e.t1-e.t0:null;
 $('anatomyName').textContent=e?`${eventLabel(e)} · ${directionLabel(e,manifest)}`:'Choose a packet or start playback.';
 const key=base+'|'+e?.id;
 if(key!==state.anatomyKey){state.anatomyKey=key;replaceReading($('anatomyParts'),a.parts.map((p,i)=>{const b=document.createElement('button');b.textContent=p.label;b.style.flex=`${p.end_s==null?1:Math.max(.001,p.end_s-p.start_s)} 1 0`;b.setAttribute('aria-label',p.label);b.title=`${p.label} · ${p.start_s.toFixed(3)}–${p.end_s==null?'open':p.end_s.toFixed(3)} s from packet start. ${p.basis}`;b.onclick=()=>{stop();clearSelection();setTime(e.t0+p.start_s);draw();};return b;}));}
 [...$('anatomyParts').children].forEach((b,i)=>b.classList.toggle('current',a.parts[i]===a.current));
 $('anatomyCursor').hidden=a.progress===null||!e||state.time<e.t0||state.time>e.t1;
 $('anatomyCursor').style.left=`${(a.progress||0)*100}%`;
 $('anatomyPosition').textContent=e?duration==null?'Open interval · duration unknown':`${Math.max(0,Math.min(duration,state.time-e.t0)).toFixed(2)} / ${duration.toFixed(2)} s`:'';
 $('anatomyNote').textContent=e?(a.current?`${a.current.label} · ${a.current.basis}`:a.parts.map(p=>`${p.label}: ${p.basis}`).join(' '))+' Recording inspection shows known section boundaries ahead of playback. Click a section to seek.':'';
}
function draw(){if(!manifest)return;const gen=++state.generation;showLinkState();timeline();const e=state.selected||packetAt(state.events,state.time);packetCaption(e);details(state.selected||e);showBytes(e);showExchange();showAnatomy(e);Promise.all([waterfall(gen),cloud(e,gen),comparisons(e,gen)]).then(()=>{if(gen===state.generation&&state.plotLoadError){state.plotLoadError=false;$('notice').textContent=(manifest.sources||[]).filter(s=>s.offset_s===null).map(s=>`${sourceName(s.id)} is unaligned. ${s.alignment}`).join('\n');}}).catch(error=>{if(gen!==state.generation)return;state.plotLoadError=true;$('notice').textContent=`${error.message}. The affected plot retains its last loaded view. Seek again to retry.`;});}
async function load(index){stop();state.loop=null;state.generation++;const a=artifacts[index];manifest=a.manifest;base=a.base;state.cache.clear();state.tiles.clear();state.events=await json(manifest.events);clearSelection();setTime(0);$('seek').max=manifest.duration_s;$('title').textContent=`${manifest.call} · ${manifest.protocol}${manifest.protocol==='VARA'?' BW'+manifest.bw_hz:''}`;$('subtitle').textContent=`${manifest.started_utc} · ${manifest.recording.kind} · ${seconds(manifest.duration_s)}`;$('audio').src=manifest.sources.find(s=>s.url)?.url||'';$('badge').textContent=manifest.protocol;
const frequency=frequencyView(manifest);$('frequencyRange').replaceChildren(new Option(`Full recording · 0–${frequency.maximum} Hz`,'full'),...(frequency.expected?[new Option(`Expected signal · ${manifest.bw_hz} Hz`,'signal')]:[]));
const rows=waveformRows(manifest.protocol,manifest.bw_hz);$('modeGuide').closest('table').hidden=!rows.length;$('modeGuide').replaceChildren(...rows.map(r=>{const tr=document.createElement('tr');for(const value of [`${r.protocol} ${r.protocol==='VARA'?'L':'SL'}${r.level}`,`${r.family} · ${r.modulation}`,`${r.carrier_count}`]){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;}));
$('listenSource').replaceChildren(...(manifest.sources||[{id:'rx',kind:manifest.recording.kind,url:manifest.recording.url,offset_s:0}]).filter(s=>s.offset_s!==null&&s.url).map(s=>new Option(sourceName(s.id),s.id)));
$('sourcekind').textContent=({'rig rx':'Waterfall & RX: rig receiver input.','rig tap':'Source: rig monitor tap.','sdr witness':'Source: remote SDR witness.'})[manifest.recording.kind]||manifest.recording.kind;
if(manifest.sources?.some(s=>s.kind==='rig tap'&&s.offset_s!==null))$('sourcekind').textContent+=' TX inspection: aligned monitor tap.';
$('sourcefile').href=manifest.recording.url||'./manifest.json';$('sourcefile').textContent=manifest.recording.url?decodeURIComponent(manifest.recording.url.split('/').at(-1)):'Recording details';
const comparison=manifest.comparison;
$('comparison').textContent=comparison?.logged_rx_overs!=null?`Original session: ${comparison.logged_rx_overs} incoming data overs logged. This replay: ${comparison.replayed_rx_frames} CRC-valid received data frames. Replay may be incomplete; the logbook outcome comes from the original session.`:manifest.protocol==='PACTOR'?'PACTOR replay · packets, control words and signal candidates are distinguished. Combined decodes are labelled; replay does not replace the original session outcome.':'Replay findings are incomplete observations of the recording; they do not replace the original session outcome.';
$('provenance').textContent=`${manifest.clock} ${manifest.timing||''} Decoder source ${manifest.decoder.source_sha256.slice(0,12)}. Recording SHA256 ${manifest.recording.sha256}.`;$('limits').replaceChildren(...manifest.limits.map(t=>{const li=document.createElement('li');li.textContent=t;return li;}));$('events').replaceChildren(...state.events.map(e=>{const b=document.createElement('button');b.dataset.id=e.id;const t=document.createElement('time');t.textContent=seconds(e.t0);const kind=document.createElement('span');kind.textContent=eventLabel(e);kind.style.color=color(e);const small=document.createElement('small');small.textContent=`${directionLabel(e,manifest)} · ${e.status}`;b.append(t,kind,small);b.onclick=()=>select(e);return b;}));$('signalSource').replaceChildren(new Option('Best available','auto'),...(manifest.sources||[]).map(s=>{const option=new Option(sourceName(s.id)+(s.offset_s===null?' · unaligned':''),s.id);option.disabled=s.offset_s===null;option.title=s.alignment;return option;}));sessionOverview();filterEvents();$('workspace').hidden=false;$('notice').textContent=(manifest.sources||[]).filter(s=>s.offset_s===null).map(s=>`${sourceName(s.id)} is available but could not be aligned. ${s.alignment}`).join('\n');draw();}
for(const id of ['eventDirection','eventResult','eventSearch'])$(id).oninput=()=>filterEvents(true);
$('eventsPrevious').onclick=()=>{if($('eventsPrevious').getAttribute('aria-disabled')==='true')return;state.eventPage--;filterEvents();};$('eventsNext').onclick=()=>{if($('eventsNext').getAttribute('aria-disabled')==='true')return;state.eventPage++;filterEvents();};
$('replayFrame').onclick=()=>{if(state.loop){state.loop=null;stop();draw();return;}const e=state.currentPacket;if(!e)return;state.loop=e;setTime(Math.max(0,e.t0-.15));play();};
$('play').onclick=()=>state.playing?stop():play();$('follow').onclick=play;
$('seek').oninput=()=>{stop();state.loop=null;clearSelection();setTime(+$('seek').value);$('audio').currentTime=Math.max(0,state.time+audioOffset());draw();};
for(const id of ['frequencyRange','span','cloud','carrier','window','signalMode','signalSource','compare','byteMode','exchangeScope'])$(id).onchange=draw;
$('speed').onchange=()=>{$('audio').playbackRate=+$('speed').value;$('audio').preservesPitch=false;};
$('listenSource').onchange=()=>{syncAudio();if(state.playing&&$('listen').checked)play();};
$('listen').onchange=()=>{if(state.preparing)stop();else if(state.playing)play();else $('audio').pause();};
$('audio').onerror=()=>{if($('listen').checked){stop();$('notice').textContent='Recording audio is unavailable or unsupported in this browser. Visual replay still works with Listen off.';}};
for(const [id,delta] of [['previous',-1],['next',1]])$(id).onclick=()=>{const idx=state.selected?state.events.indexOf(state.selected):state.events.findIndex(e=>e.t0>=state.time);const e=state.events[Math.max(0,Math.min(state.events.length-1,idx+delta))];if(e)select(e);};
$('timeline').onclick=event=>{const r=$('timeline').getBoundingClientRect(),t=Math.max(0,(event.clientX-r.left-82)/(r.width-82))*manifest.duration_s;const lane=Math.max(0,Math.min(2,Math.floor((event.clientY-r.top-12)/25))),direction=['tx','rx','observed'][lane],candidates=state.events.filter(e=>e.direction===direction),inside=candidates.filter(e=>e.t0<=t&&e.t1>=t);inside.sort((a,b)=>Number(!!b.symbols)-Number(!!a.symbols)||Number(b.frame_type==='data')-Number(a.frame_type==='data'));const nearest=inside[0]||candidates.reduce((a,b)=>!a||Math.abs(b.t0-t)<Math.abs(a.t0-t)?b:a,null);if(nearest)select(nearest);};
$('waterfall').onclick=event=>{state.loop=null;const r=$('waterfall').getBoundingClientRect(),[lo,hi]=view();stop();clearSelection();setTime(lo+Math.max(0,Math.min(1,(event.clientX-r.left-48)/(r.width-56)))*(hi-lo));draw();};
// Moving interactive packet content stops before a keyboard/switch user reads it.
$('workspace').addEventListener('focusin',event=>{
 if((state.playing||state.preparing)&&(event.target.closest('.panel,.sourcebar')||event.target.id==='seek')){stop();$('notice').textContent='Playback paused for inspection. Use Play to resume.';}
});
$('waterfall-values').addEventListener('toggle',()=>{if($('waterfall-values').open)draw();});
$('waterfall-download').onclick=async()=>{
 stop();
 try {
  const wf=manifest.waterfall,parts=['time_s,frequency_hz,audio_dbfs\n'];
  for(const tile of wf.tiles){const data=await json(base+tile.file),bytes=decodeRaster(data.raster_base64);const rows=[];
   for(let row=0;row<tile.rows;row++)for(let bin=0;bin<wf.bins;bin++)rows.push(`${((tile.first+row)*wf.hop_s).toFixed(3)},${(bin*wf.hz_per_bin).toFixed(3)},${(wf.db_min+bytes[row*wf.bins+bin]/255*(wf.db_max-wf.db_min)).toFixed(3)}\n`);
   parts.push(rows.join(''));
  }
  const url=URL.createObjectURL(new Blob(parts,{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download=`${manifest.arm}-waterfall.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 }catch(error){$('notice').textContent=`Waterfall download failed: ${error.message}`;}
};
window.addEventListener('resize',draw);
function tick(now){if(state.playing&&manifest){const elapsed=lastTick?(now-lastTick)/1000:0;const t=$('listen').checked?$('audio').currentTime-audioOffset():state.time+elapsed*(+$('speed').value);setTime(t);if(state.loop&&state.time>state.loop.t1+.15){setTime(Math.max(0,state.loop.t0-.15));$('audio').currentTime=Math.max(0,state.time+audioOffset());}if(state.time>=manifest.duration_s||($('listen').checked&&$('audio').ended))stop();if(now-lastDraw>33){draw();lastDraw=now;}}lastTick=now;requestAnimationFrame(tick);}requestAnimationFrame(tick);
try {
 setWaveforms(await json('../waveforms.json'));
 artifacts=[{manifest:await json('./manifest.json'),base:'./data/'}];
 await load(0);
 const start=Number(new URLSearchParams(location.search).get('t'));
 if(Number.isFinite(start)&&start>0){setTime(start);draw();}
} catch(e) { $('notice').textContent=e.message; }

installDeskEmbedding(stop,()=>{state.tiles.clear();toneTiles=new WeakMap();draw()});
