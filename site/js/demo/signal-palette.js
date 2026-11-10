// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
/* Shared display preference only; it never changes signal data or decoding. */
(() => {
 const key='hfmodem-signal-palette';
 const palettes={green:{name:'Green',stops:[[20,38,30],[70,105,62],[168,197,125],[245,255,215]]},blueyellow:{name:'Blue–yellow',stops:[[12,22,45],[39,84,143],[78,163,192],[240,225,123],[255,254,233]]},gray:{name:'Grayscale',stops:[[10,10,10],[250,250,250]]}};
 let selected='green';try{selected=localStorage.getItem(key)||selected}catch{}
 if(!palettes[selected])selected='green';
 function rgb(value){const stops=palettes[selected].stops,x=Math.max(0,Math.min(1,value))*(stops.length-1),lo=Math.min(stops.length-2,Math.floor(x)),f=x-lo;return stops[lo].map((v,i)=>Math.round(v+(stops[lo+1][i]-v)*f))}
 const css=value=>'rgb('+rgb(value).join(',')+')';
 function refresh(){document.querySelectorAll('[data-signal-palette]').forEach(s=>s.value=selected);document.querySelectorAll('[data-palette-ramp]').forEach(s=>s.style.background='linear-gradient(to right,'+palettes[selected].stops.map(v=>'rgb('+v.join(',')+')').join(',')+')');window.dispatchEvent(new Event('hfmodem-palette-change'))}
 function set(id){if(!palettes[id])return;selected=id;try{localStorage.setItem(key,id)}catch{}refresh()}
 function bind(id){const select=document.getElementById(id);if(!select)return;select.replaceChildren(...Object.entries(palettes).map(([id,p])=>new Option(p.name,id)));select.value=selected;select.onchange=()=>set(select.value);document.querySelectorAll('[data-palette-ramp]').forEach(s=>s.style.background='linear-gradient(to right,'+palettes[selected].stops.map(v=>'rgb('+v.join(',')+')').join(',')+')')}
 window.HFSignalPalette={rgb,css,set,bind,get id(){return selected}};
 window.addEventListener('storage',e=>{if(e.key===key&&palettes[e.newValue]){selected=e.newValue;refresh()}});
})();
