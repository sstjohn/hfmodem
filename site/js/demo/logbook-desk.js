// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
/* Static review of the six curated public recordings. No station API. */
let logbookInspection=null,publicReviewError='',publicReviewReady=false;
const preparedReplays=[];
function reviewTheme(){document.querySelectorAll('#logbook-desk iframe').forEach(f=>f.contentWindow?.postMessage({kind:'desk-theme',theme:state.theme},location.origin))}
function pauseReview(){document.querySelectorAll('#logbook-desk iframe').forEach(f=>f.contentWindow?.postMessage({kind:'desk-pause-review'},location.origin))}
function openPublicReplay(route){
 if(!publicReviewReady||!preparedReplays.some(([,id])=>id===route))return;
 pauseReview();state.profile='logbook';logbookInspection=route;render();document.getElementById('inspect-frame').focus();
}
function returnToBook(){pauseReview();logbookInspection=null;render();document.getElementById('book-frame').focus()}
function renderLogbook(){
 const frame=document.getElementById('book-frame'),inspect=document.getElementById('inspect-frame');
 if(!frame.getAttribute('src')){frame.src='logbook.html?theme='+state.theme;frame.onload=reviewTheme;}
 frame.hidden=!!logbookInspection;inspect.hidden=!logbookInspection;
 if(logbookInspection){const url='../examples/'+logbookInspection+'/?desk=1&theme='+state.theme;if(inspect.dataset.arm!==logbookInspection){inspect.dataset.arm=logbookInspection;inspect.src=url;inspect.onload=reviewTheme;}inspect.title=preparedReplays.find(([,id])=>id===logbookInspection)?.[0]+' · recorded QSO inspector';}
 reviewTheme();
 const toolbar=document.getElementById('sessionbar');
 toolbar.hidden=!logbookInspection;
 HFAccessibility.render(toolbar,logbookInspection?'<button class="small" id="return-book">← Back to logbook</button><strong>Recorded QSO inspector</strong><span class="badge">READ-ONLY</span>':'');
 if(document.getElementById('return-book'))document.getElementById('return-book').onclick=returnToBook;
}
window.addEventListener('message',event=>{
 if(event.origin!==location.origin||![document.getElementById('book-frame')?.contentWindow,document.getElementById('inspect-frame')?.contentWindow].includes(event.source))return;
 if(event.data?.kind==='inspect-arm')openPublicReplay(event.data.arm);
 if(event.data?.kind==='back-book')returnToBook();
 if(event.data?.kind==='stop-radio')stop();
});
import('./public-qsos.mjs').then(async({loadPublicQsos})=>{
 const rows=await loadPublicQsos();preparedReplays.push(...rows.map(row=>[`${row.peer} · ${row.protocol} ${row.role}`,row.route]));publicReviewReady=true;
 document.getElementById('open-logbook-replay').disabled=false;if(state.profile==='logbook')render();
}).catch(error=>{publicReviewError=error.message;if(state.profile==='logbook')render()});
