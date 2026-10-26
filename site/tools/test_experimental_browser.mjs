// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import assert from 'node:assert/strict';
export async function checkExperimentalGuides(page,baseURL,axeSource){
 const checked=[],failures=[];
 const audit=async(name)=>{
  await page.addScriptTag({content:axeSource});
  const result=await page.evaluate(()=>axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}}));
  if(result.violations.length)failures.push({name,violations:result.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});
  checked.push(name);
 };
 for(const theme of ['light','dark']){
  await page.emulateMedia({colorScheme:theme,reducedMotion:'reduce'});
  await page.goto(new URL('robust-packet/',baseURL).href);
  await page.locator('#level').focus();await page.locator('#level').selectOption('1');
  assert.equal(await page.evaluate(()=>document.activeElement.id),'level');
  assert.match(await page.locator('#rateResult').innerText(),/600 bit\/s/);
  await page.locator('#capacity').selectOption('825');
  const phase=await page.locator('#phasePlot').innerHTML();
  await page.locator('#phaseNext').press('Enter');
  assert.notEqual(await page.locator('#phasePlot').innerHTML(),phase);
  assert.equal(await page.evaluate(()=>document.activeElement.id),'phaseNext');
  for(const region of await page.locator('.figure-scroll').all())assert.equal(await region.getAttribute('tabindex'),'0');
  await audit(`${theme} robust packet level1 largest packet`);
  await page.goto(new URL('stanag/',baseURL).href);
  await page.locator('#frame-family').selectOption('1');
  assert.match(await page.locator('#frame-summary').innerText(),/1200|213\.3/);
  await page.locator('#constellation-mode').focus();await page.locator('#constellation-mode').selectOption('256-QAM');
  assert.equal(await page.locator('#constellation-values tr').count(),256);
  assert.equal(await page.evaluate(()=>document.activeElement.id),'constellation-mode');
  await audit(`${theme} STANAG4529 and256QAM`);
  await page.goto(new URL('rtty/',baseURL).href);
  await page.locator('#letter-code').selectOption('8'); // H /US#; international unassigned.
  await page.locator('#figure-variant').selectOption('itu');
  assert.match(await page.locator('#rtty-readout').innerText(),/H \/ unassigned/);
  assert.equal(await page.locator('#rtty-values tr').count(),7);
  await page.locator('#diversity-case').selectOption('conflict');
  await page.locator('#diversity .diagram-data summary').click();
  assert.match(await page.locator('#diversity-decision').innerText(),/unresolved/);
  await page.locator('#navtex-case').focus();await page.locator('#navtex-case').selectOption('partial');
  assert.match(await page.locator('#navtex-readout').innerText(),/Pending/);
  assert.equal(await page.evaluate(()=>document.activeElement.id),'navtex-case');
  await audit(`${theme} text-mode ambiguity and partial bulletin`);
  await page.locator('#navtex-case').selectOption('bad');
  assert.match(await page.locator('#navtex-readout').innerText(),/rejected/);
 }
 await page.setViewportSize({width:320,height:900});
 for(const mode of ['robust-packet','stanag','rtty']){
  await page.goto(new URL(mode+'/',baseURL).href);
  const width=await page.evaluate(()=>({page:document.documentElement.scrollWidth,viewport:innerWidth,nodes:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&e.getClientRects().length).slice(0,12).map(e=>({tag:e.tagName,id:e.id,cls:e.getAttribute('class'),width:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right}))}));
  assert.ok(width.page<=width.viewport+1,JSON.stringify({mode,...width}));
  for(const region of await page.locator('[tabindex="0"][role=region]').all()){
   if(await region.isVisible()&&await region.evaluate(e=>e.scrollWidth>e.clientWidth+1)){
    await region.focus();await region.press('ArrowRight');await page.waitForTimeout(80);
    assert.ok(await region.evaluate(e=>e.scrollLeft>0),JSON.stringify({mode,label:await region.getAttribute('aria-label'),size:await region.evaluate(e=>({client:e.clientWidth,scroll:e.scrollWidth,left:e.scrollLeft,active:document.activeElement===e}))}));
   }
  }
  await audit('320px '+mode);
 }
 await page.setViewportSize({width:1280,height:900});
 await page.emulateMedia({reducedMotion:'no-preference',colorScheme:'light'});
 assert.deepEqual(failures,[],JSON.stringify(failures));
 return checked;
}
