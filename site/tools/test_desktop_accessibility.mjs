// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import assert from 'node:assert/strict';
const routes=['vara-send','vara-receive','pactor3-send','pactor3-receive','ardop-send','ardop-receive'];
export async function checkDesktop(page,baseURL,axeSource){
 const checks=[],violations=[];
 const audit=async(scope,name)=>{
  await scope.addScriptTag({content:axeSource});
  const r=await scope.evaluate(async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}}));
  if(r.violations.length)violations.push({name,violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});
  checks.push(name);
 };
 await page.goto(new URL('demo/',baseURL).href);
 await page.locator('#open-logbook-replay').waitFor({state:'visible'});
 await page.waitForFunction(()=>!document.getElementById('open-logbook-replay').disabled);
 for(const theme of ['light','dark']){
  if(await page.locator('html').getAttribute('data-theme')!==theme)await page.locator('#theme').click();
  for(const profile of ['chat','contest','traffic']){
   await page.locator(`[data-profile=${profile}]`).click();
   assert.equal(await page.evaluate(()=>document.activeElement.dataset.profile),profile,'desk focus');
   await audit(page,`${theme} ${profile}`);
  }
  await page.locator('[data-traffic=net]').click();await audit(page,`${theme} net traffic`);
  await page.locator('#new-message').click();await audit(page,`${theme} compose exercise`);
  await page.locator('#compose [data-close]').press('Escape');
  await page.locator('[data-traffic=mail]').click();
  await page.locator('#new-mail').click();await audit(page,`${theme} compose mail`);
  await page.locator('#close-mail-compose').click();
  for(const dialog of ['tune','diagnostics','accessibility']){
   await page.locator(dialog==='tune'?'#dial':`#${dialog}-open`).click();
   await page.locator(`#${dialog}`).waitFor({state:'visible'});
   await audit(page,`${theme} ${dialog} dialog`);
   await page.locator(`#${dialog} [data-close]`).press('Escape');
   assert.equal(await page.locator(`#${dialog}`).isVisible(),false,'Escape closes dialog');
   assert.equal(await page.evaluate(()=>document.activeElement.id),dialog==='tune'?'dial':`${dialog}-open`,'dialog restores focus');
  }
  await page.locator('[data-profile=logbook]').click();
  await page.locator('.skip').press('Enter');
  assert.equal(await page.evaluate(()=>document.activeElement.id),'logbook-desk','logbook skip focus');
  const book=()=>page.frames().find(f=>f.url().includes('/demo/logbook.html'));
  await page.frameLocator('#book-frame').locator('#rows .row').first().waitFor({state:'visible'});
  assert.equal(await page.locator('#sessionbar').isVisible(),false,'The embedded logbook owns its header and controls');
  assert.equal(await page.locator('[data-replay]').count(),0,'No duplicate outer QSO shortcuts');
  assert.match(await page.locator('#book-frame').getAttribute('title'),/Public logbook/,'Logbook iframe keeps an accessible name');
  assert.equal(await book().locator('#rows .row').count(),6,'only six records');
  assert.equal(await book().locator('.release-notice').isVisible(),false,'parent warning replaces duplicate iframe warning');
  assert.equal(await page.locator('.release-notice').isVisible(),true,'parent warning remains visible');
  await book().locator('#protocol').selectOption('ARDOP');await book().locator('#role').selectOption('Send');
  assert.equal(await book().locator('#rows .row:visible').count(),1);
  await book().getByRole('button',{name:/WW2MI .* details/}).click();
  assert.match(await book().locator('#detail-ardop-send').innerText(),/failure|expected SOH|parser/i,'receive failure retained');
  await audit(book(),`${theme} filtered logbook details`);
  await book().locator('#summaryView').click();await audit(book(),`${theme} logbook summary`);
  await book().locator('#bookView').click();await book().locator('#reset').click();
  await book().locator('#rows [data-route=vara-send] .inspect-link').click();
  await page.frameLocator('#inspect-frame').locator('#workspace').waitFor({state:'visible'});
  assert.match(await page.locator('#inspect-frame').getAttribute('src'),/examples\/vara-send\//,'logbook Inspect opens public recording');
  await page.locator('#return-book').click();
  await page.locator('#book-frame').waitFor({state:'visible'});
  for(const route of routes){
   const inspect=book().locator(`#rows [data-route=${route}] .inspect-link`);
   await inspect.focus();await inspect.press('Enter');
   await page.frameLocator('#inspect-frame').locator('#workspace').waitFor({state:'visible'});
   const view=page.frames().find(f=>f.url().includes(`/examples/${route}/`));
   await view.waitForFunction(theme=>document.documentElement.dataset.theme===theme,theme);
   assert.equal(await view.locator('.release-notice').isVisible(),false,'embedded inspector does not repeat warning');
   await audit(view,`${theme} ${route} embedded`);
   if(route==='vara-send'){
    await view.locator('#listen').check();await view.locator('#play').click();
    await view.waitForFunction(()=>document.getElementById('play').textContent==='Pause');
    await page.locator('#stop').click();
    await view.waitForFunction(()=>document.getElementById('play').textContent==='Play'&&document.getElementById('audio').paused);
    await view.locator('#play').click();await view.waitForFunction(()=>document.getElementById('play').textContent==='Pause');
    await view.locator('#play').press('Escape');
    await view.waitForFunction(()=>document.getElementById('audio').paused&&document.getElementById('play').textContent==='Play');
   }
   await view.getByRole('button',{name:'Back to logbook',exact:true}).click();
   await page.waitForFunction(()=>document.activeElement.id==='book-frame'&&document.getElementById('inspect-frame').hidden);
   assert.equal(await page.locator('#sessionbar').isVisible(),false,'Returning to the logbook restores its single header');
  }
  await page.locator('[data-profile=chat]').click();
  assert.equal(await page.locator('#sessionbar').isVisible(),true,'Other desks keep their own session controls');
 }
 await page.setViewportSize({width:320,height:900});
 for(const profile of ['chat','contest','traffic','logbook']){
  await page.locator(`[data-profile=${profile}]`).click();
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),profile+' narrow reflow');
  await audit(page,`320px ${profile}`);
 }
 await page.setViewportSize({width:1280,height:900});
 // Source messages from the parent page are not accepted as frame requests.
 await page.evaluate(()=>window.postMessage({kind:'inspect-arm',arm:'unapproved-route'},location.origin));
 assert.equal(await page.locator('#inspect-frame').isVisible(),false);
 await page.goto(new URL('demo/console-tools.html',baseURL).href);
 await page.locator('#commands [data-command]').first().waitFor({state:'visible'});
 for(const theme of ['light','dark']){
  if(await page.locator('html').getAttribute('data-theme')!==theme)await page.locator('#tools-theme').click();
  for(const command of ['vara fly','pactor fly']){
   await page.locator(`[data-command=\"${command}\"]`).click();
   assert.match(await page.locator('#command-preview').inputValue(),/console (vara|pactor) fly/);
   await audit(page,`${theme} ${command} command preview`);
  }
 }
 await page.route('**/examples/logbook/entries.json',route=>route.fulfill({json:[]}));
 await page.goto(new URL('demo/logbook.html',baseURL).href);
 await page.locator('#retry').waitFor({state:'visible'});
 assert.equal(await page.locator('#rows .row').count(),0,'missing public roster never fabricates rows');
 assert.match(await page.locator('#loadStatus').innerText(),/could not be loaded/);
 await page.unroute('**/examples/logbook/entries.json');
 await page.locator('#retry').click();await page.waitForFunction(()=>document.querySelectorAll('#rows .row').length===6);
 checks.push('public data error and retry');
 assert.deepEqual(violations,[],JSON.stringify(violations));
 return checks;
}
