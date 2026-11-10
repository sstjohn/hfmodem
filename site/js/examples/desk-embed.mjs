// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Same-origin, opt-in embedding in the static operating desktop.
export function installDeskEmbedding(stop,redraw){
 if(window.parent===window||new URLSearchParams(location.search).get('desk')!=='1')return;
 const style=document.createElement('link');style.rel='stylesheet';style.href=new URL('../../examples/desk-embed.css',import.meta.url);document.head.append(style);
 const palettes={green:[[20,38,30],[70,105,62],[168,197,125],[245,255,215]],blueyellow:[[12,22,45],[39,84,143],[78,163,192],[240,225,123],[255,254,233]],gray:[[10,10,10],[250,250,250]]};
 let palette='green';
 function readPalette(){try{palette=localStorage.getItem('hfmodem-signal-palette')||'green'}catch{}if(!palettes[palette])palette='green'}
 readPalette();
 window.HFSignalPalette={rgb(v){const stops=palettes[palette],x=Math.max(0,Math.min(1,v))*(stops.length-1),lo=Math.min(stops.length-2,Math.floor(x)),f=x-lo;return stops[lo].map((a,i)=>Math.round(a+(stops[lo+1][i]-a)*f))}};
 function theme(value){document.documentElement.dataset.theme=value==='light'?'light':'dark';redraw()}
 theme(new URLSearchParams(location.search).get('theme'));
 const back=document.createElement('button');back.textContent='Back to logbook';back.addEventListener('click',()=>{stop();parent.postMessage({kind:'back-book'},location.origin)});document.querySelector('header').prepend(back);
 window.addEventListener('message',event=>{if(event.origin!==location.origin||event.source!==parent)return;if(event.data?.kind==='desk-theme')theme(event.data.theme);if(event.data?.kind==='desk-pause-review')stop()});
 window.addEventListener('storage',event=>{if(event.key==='hfmodem-signal-palette'){readPalette();redraw()}});
 document.addEventListener('keydown',event=>{if(event.key==='Escape'){stop();parent.postMessage({kind:'stop-radio'},location.origin)}});
 window.addEventListener('pagehide',stop);
}
