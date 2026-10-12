// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Automated WCAG 2.2 AA checks supplement manual assistive-technology review.
// Usage: NODE_PATH=/path/to/dev/node_modules node site/audit_accessibility.mjs _site http://localhost:8767
import {checkDesktop} from './test_desktop_accessibility.mjs';
import {checkExperimentalGuides} from './test_experimental_browser.mjs';
import {checkGuides} from './test_browser_accessibility.mjs';
import {checkPactor4} from './test_pactor4_browser.mjs';
import {checkInspectors,checkReflow} from './test_inspector_accessibility.mjs';
import {createRequire} from 'node:module';
import {readdir, readFile, writeFile} from 'node:fs/promises';
import {resolve, relative} from 'node:path';
const require = createRequire(process.argv[4] ? resolve(process.argv[4], 'package.json') : import.meta.url);
const {chromium} = require('playwright');
const axe = require('axe-core');
const root = resolve(process.argv[2] || '_site');
const routeFilter = (process.argv[7] || process.env.A11Y_ROUTES)?.split(',');
const origin = process.argv[3] || 'http://127.0.0.1:8767';
const executablePath = process.argv[5] || process.env.A11Y_CHROME;
const browser = await chromium.launch({headless:true, ...(executablePath ? {executablePath} : {})});
const pages = [];
async function collect(dir) { for (const item of await readdir(dir,{withFileTypes:true})) {
 const path = resolve(dir,item.name);
 if(item.isDirectory())await collect(path);
 else if(item.name.endsWith('.html'))pages.push(path);
}}
await collect(root);
if(routeFilter)pages.splice(0,pages.length,...pages.filter(p=>routeFilter.some(prefix=>relative(root,p).startsWith(prefix))));
const results = [], runtimeErrors = [];
for (const theme of ['light','dark']) {
 const page = await browser.newPage({colorScheme:theme,viewport:{width:1280,height:900}});
 // Ignore analytics; the audit inspects only the static local build.
 await page.route('**/*',route=>new URL(route.request().url()).origin===new URL(origin).origin ? route.continue() : route.abort());
 let current;
 page.on('pageerror',error=>runtimeErrors.push({route:current,theme,message:error.message}));
 for (const file of pages) {
  if(results.length && process.env.A11Y_FAIL_FAST)break;
  current = relative(root,file).replaceAll('\\','/');
  await page.goto(new URL(current,origin+'/').href);
  if(current.startsWith('examples/')&&await page.locator('#workspace').count())await page.locator('#workspace').waitFor({state:'visible'});
  await page.addScriptTag({content:axe.source});
  const result = await page.evaluate(async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}}));
  if(result.violations.length)results.push({route:current,theme,violations:result.violations.map(v=>({id:v.id,impact:v.impact,help:v.help,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});
 }
 await page.close();
}
const functional = {};
console.log(`axe checked ${pages.length} routes in light and dark themes; starting interaction checks`);
const page = await browser.newPage({viewport:{width:1280,height:900}});
page.on('pageerror',error=>runtimeErrors.push({route:'functional checks',message:error.message}));
await page.route('**/*',route=>new URL(route.request().url()).origin===new URL(origin).origin ? route.continue() : route.abort());
try {
 functional.guides = await checkGuides(page,origin+'/');
 functional.pactor4 = await checkPactor4(page,origin+'/');
 functional.experimentalGuides = await checkExperimentalGuides(page,origin+'/',axe.source);
 functional.inspectors = await checkInspectors(page,origin+'/');
 functional.desktop = await checkDesktop(page,origin+'/',axe.source);
 functional.reflow320 = await checkReflow(page,origin+'/');
} catch(error) { runtimeErrors.push({route:'functional checks',message:error.stack}); }
// Optional review image uses the same local build and curated recordings.
if(process.argv[8]) {
 await page.setViewportSize({width:1280,height:1400});
 const reviewRoute=process.argv[9] || 'demo/';
 await page.goto(new URL(reviewRoute,origin+'/').href);
 if(reviewRoute==='demo/') {
  await page.locator('[data-profile=logbook]').click();
  await page.frameLocator('#book-frame').locator('#rows .row').first().waitFor({state:'visible'});
  await page.frameLocator('#book-frame').locator('.release-notice').waitFor({state:'hidden'});
 } else await page.locator('main').waitFor({state:'visible'});
 await page.screenshot({path:process.argv[8]});
}
await page.close();
await browser.close();
const report = {pages:pages.length,themes:['light','dark'],violations:results,runtimeErrors,functional};
if(process.argv[6] || process.env.A11Y_REPORT)await writeFile(process.argv[6] || process.env.A11Y_REPORT,JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
if(results.length||runtimeErrors.length)process.exitCode=1;
