// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Without JavaScript, both figures and their replay links remain available.
for (const widget of document.querySelectorAll('[data-evidence-widget]')) {
 const list=widget.querySelector('.evidence-tabs');
 const tabs=[...list.querySelectorAll('a')];
 const panels=tabs.map(tab=>document.getElementById(tab.hash.slice(1)));
 list.setAttribute('role','tablist');
 function select(index,focus=false){
  tabs.forEach((tab,i)=>{
   tab.setAttribute('aria-selected',String(i===index));tab.tabIndex=i===index?0:-1;
   panels[i].hidden=i!==index;
  });
  if(focus)tabs[index].focus();
 }
 tabs.forEach((tab,i)=>{
  tab.setAttribute('role','tab');tab.setAttribute('aria-controls',panels[i].id);
  panels[i].setAttribute('role','tabpanel');panels[i].tabIndex=0;
  tab.addEventListener('click',event=>{event.preventDefault();select(i);});
  tab.addEventListener('keydown',event=>{
   const next=event.key==='ArrowRight'?(i+1)%tabs.length:event.key==='ArrowLeft'?(i+tabs.length-1)%tabs.length:event.key==='Home'?0:event.key==='End'?tabs.length-1:null;
   if(next!==null){event.preventDefault();select(next,true);}
   else if(event.key===' '){event.preventDefault();select(i);}
  });
 });
 select(0);
}
