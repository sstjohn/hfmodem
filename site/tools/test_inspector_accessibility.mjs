// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import assert from 'node:assert/strict';
export async function checkInspectors(page, baseURL) {
 const checked=[];
 for(const route of ['vara-send','vara-receive','pactor3-send','pactor3-receive','ardop-send','ardop-receive']){
  await page.goto(new URL(`examples/${route}/`,baseURL).href);
  await page.locator('#workspace').waitFor({state:'visible'});
  await page.locator('.skip').press('Tab');
  await page.getByRole('link',{name:'Skip to content',exact:true}).press('Enter');
  assert.equal(await page.evaluate(()=>document.activeElement.id),'main',route+' skip focus');
  assert.ok(await page.locator('#events button:visible').count()<=25,route+' manageable event page');
  if(route==='vara-send'){
   await page.route('**/*.flac',async request=>{await new Promise(resolve=>setTimeout(resolve,200));await request.continue();});
   await page.locator('#listen').check();
   await page.locator('#play').click();
   assert.equal(await page.evaluate(()=>document.activeElement.id),'play','audio loading retains Play focus');
   await page.waitForFunction(()=>document.getElementById('play').textContent==='Pause');
   assert.equal(await page.evaluate(()=>document.activeElement.id),'play','audio playback retains Play focus');
  }else await page.locator('#play').click();
  assert.equal(await page.locator('#play').innerText(),'Pause');
  await page.locator('#eventSearch').click();
  assert.equal(await page.locator('#play').innerText(),'Play',route+' focus pauses');
  assert.equal(await page.evaluate(()=>document.activeElement.id),'eventSearch');
  await page.locator('#eventSearch').fill('no-matching-event-xyz');
  assert.equal(await page.locator('#events button:visible').count(),0);
  assert.match(await page.locator('#eventCount').innerText(),/^0 of /);
  await page.locator('#eventSearch').fill('');
  for(let count=0;count<100&&await page.locator('#eventsNext').getAttribute('aria-disabled')!=='true';count++)await page.locator('#eventsNext').click();
  assert.equal(await page.locator('#eventsNext').getAttribute('aria-disabled'),'true');
  if((await page.locator('#eventCount').innerText()).includes('page 1 of 1')===false)assert.equal(await page.evaluate(()=>document.activeElement.id),'eventsNext',route+' last page preserves focus');
  const packet=page.locator('#events button:visible').first();
  await packet.click();
  assert.equal(await page.evaluate(()=>document.activeElement.dataset.id),await packet.getAttribute('data-id'));
  await page.locator('#exchangeScope').selectOption('whole');
  const ladder=page.locator('#ladder button').first();
  await ladder.click();
  assert.equal(await page.evaluate(()=>document.activeElement.dataset.id),await ladder.getAttribute('data-id'),route+' ladder focus');
  assert.ok(await page.locator('#ladder button').count()<=40);
  assert.ok(await page.locator('#details').innerText(),route+' text frame details');
  await page.locator('#signal-text summary').click();
  assert.ok(await page.locator('#signal-description').innerText(),route+' text signal description');
  // Changes requested through a native select remain on that control.
  await page.locator('#signalMode').selectOption('spectrum');
  await page.locator('#signalMode').press('Tab');
  assert.notEqual(await page.evaluate(()=>document.activeElement.tagName),'BODY');
  if(route==='vara-send'){
   await page.locator('#waterfall-values summary').click();
   await page.waitForFunction(()=>document.getElementById('waterfall-peaks').textContent.startsWith('Time seconds'));
   const downloaded=page.waitForEvent('download');
   await page.locator('#waterfall-download').click();
   const download=await downloaded,stream=await download.createReadStream();let csv='';
   for await(const chunk of stream)csv+=chunk.toString();
   assert.match(csv,/^time_s,frequency_hz,audio_dbfs/);
   const geometry=await (await page.request.get(new URL(`examples/${route}/manifest.json`,baseURL).href)).json();
   assert.equal(csv.trim().split('\n').length,1+geometry.waterfall.rows*geometry.waterfall.bins,'full numeric waterfall equivalent');
  }
  checked.push(route);
 }
 await page.goto(new URL('examples/logbook/',baseURL).href);
 await page.getByText('Keyboard preferences',{exact:true}).click();
 await page.getByLabel('Allow letter-key shortcuts',{exact:true}).uncheck();
 await page.getByLabel('Shortcut sequence',{exact:true}).fill('ordinary words');
 await page.getByRole('button',{name:'Run shortcut',exact:true}).click();
 assert.equal(await page.locator('#shortcut-status').innerText(),'Shortcut not recognized.');
 await page.getByLabel('Shortcut sequence',{exact:true}).fill('uuddllrrab');
 await Promise.all([page.waitForURL('**/demo/'),page.getByRole('button',{name:'Run shortcut',exact:true}).click()]);
 assert.match(await page.locator('.demo-banner').innerText(),/Every radio action is simulated/);
 return checked;
}

export async function checkReflow(page,baseURL){
 const results=[],overflows=[];
 await page.setViewportSize({width:320,height:900});
 for(const route of ['','compared/','pactor/','vara/','ardop/','sabir/','examples/logbook/','examples/vara-send/','examples/pactor3-receive/','examples/ardop-receive/','accessibility/','demo/','demo/logbook.html','demo/console-tools.html']){
  await page.goto(new URL(route,baseURL).href);
  if(route.startsWith('examples/')&&await page.locator('#workspace').count())await page.locator('#workspace').waitFor({state:'visible'});
  const width=await page.evaluate(()=>({page:document.documentElement.scrollWidth,viewport:innerWidth}));
  if(width.page>width.viewport+1)overflows.push({route,...width,nodes:await page.evaluate(()=>Array.from(document.querySelectorAll('body *')).filter(e=>e.clientWidth>0&&e.scrollWidth>e.clientWidth+1&&getComputedStyle(e).overflowX==='visible').map(e=>({tag:e.tagName,id:e.id,cls:e.className,c:e.clientWidth,s:e.scrollWidth,txt:e.textContent.slice(0,60)})).slice(0,12))});
  results.push(route||'homepage');
 }
 await page.setViewportSize({width:1280,height:900});
 assert.deepEqual(overflows,[],JSON.stringify(overflows));
 return results;
}
