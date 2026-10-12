// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Functional checks supplement axe: input must change accessible text and retain focus.
import assert from 'node:assert/strict';

export async function checkGuides(page, baseURL) {
  await page.goto(baseURL);
  for (const width of [1280, 320]) {
    await page.setViewportSize({width, height:900});
    const layout = await page.evaluate(() => {
      const notice = document.querySelector('.release-notice p + p');
      const plain = notice.cloneNode(true);
      for (const link of plain.querySelectorAll('a')) link.replaceWith(document.createTextNode(link.textContent));
      plain.setAttribute('aria-hidden', 'true');
      Object.assign(plain.style, {position:'absolute', visibility:'hidden', width:notice.getBoundingClientRect().width+'px'});
      notice.parentElement.append(plain);
      const noticeHeight = notice.getBoundingClientRect().height;
      const plainHeight = plain.getBoundingClientRect().height;
      plain.remove();
      const paragraphs = [...document.querySelectorAll('#run .workbench > div > p')];
      return {noticeHeight, plainHeight,
        sizes:paragraphs.map(p=>getComputedStyle(p).fontSize),
        gaps:paragraphs.slice(1).map((p,i)=>p.getBoundingClientRect().top-paragraphs[i].getBoundingClientRect().bottom),
        installFirst:!!(document.querySelector('#run').compareDocumentPosition(document.querySelector('#operate')) & Node.DOCUMENT_POSITION_FOLLOWING)};
    });
    assert.ok(Math.abs(layout.noticeHeight-layout.plainHeight)<=1, 'Protocol links flow like ordinary sentence text');
    assert.equal(new Set(layout.sizes).size, 1, 'Setup explanation uses consistent paragraph typography');
    assert.ok(layout.gaps.every(gap=>gap>=20), 'Setup paragraphs have visible separation');
    assert.ok(layout.installFirst, 'Installation precedes commands that require hfmodem');
  }
  await page.setViewportSize({width:1280, height:900});
  const go = async mode => {
    await page.goto(new URL(`${mode}/`, `${baseURL.replace(/\/$/, '')}/`).href);
    await page.locator('main').waitFor();
  };
  const image = id => page.locator(`#${id}`).evaluate(canvas => canvas.toDataURL());
  for (const mode of ['pactor', 'vara']) {
    await go(mode);
    const play = page.locator('#wfPlay');
    assert.equal(await play.textContent(), 'Play diagrams', `${mode}: animation starts paused`);
    assert.equal(await play.getAttribute('aria-pressed'), 'false');
    const canvas = mode === 'pactor' ? 'toneCanvas' : 'trellisCanvas';
    const paused = await image(canvas);
    await page.waitForTimeout(120);
    assert.equal(await image(canvas), paused, `${mode}: secondary diagram remains still`);
    await play.click();
    assert.equal(await play.textContent(), 'Pause diagrams');
    assert.equal(await play.getAttribute('aria-pressed'), 'true');
    await page.waitForTimeout(180);
    await play.click();
    assert.equal(await play.getAttribute('aria-pressed'), 'false');
    const stopped = await image(canvas);
    await page.waitForTimeout(120);
    assert.equal(await image(canvas), stopped, `${mode}: page-wide pause includes secondary diagrams`);
    // The contents control must never leave focus on a link in a collapsed panel.
    await page.locator('.page-contents summary').click();
    const link = page.locator('.page-contents a').first();
    const target = (await link.getAttribute('href')).slice(1);
    await link.click();
    assert.equal(await page.locator('.page-contents').getAttribute('open'), null);
    assert.equal(await page.evaluate(() => document.activeElement.id), target);
    if (mode === 'pactor') {
      await page.locator('#slSeg button[data-sl="6"]').click();
      assert.equal(await page.locator('#slSeg button[data-sl="6"]').getAttribute('aria-pressed'), 'true');
      assert.equal(await page.locator('#toneCanvas-data tbody tr').count(), 18);
      assert.match(await page.locator('#toneCanvas-data summary').textContent(), /SL6/);
      await page.locator('#callIn').fill('W9SSJ');
      assert.match(await page.locator('#fskCanvas-data').textContent(), /W9SSJ/);
      assert.match(await page.locator('#fskReadout').textContent(), /57 39 53 53 4a/i);
    } else {
      await page.locator('#callIn').fill('W9SSJ');
      await page.locator('#burstSeg button[data-k="RESP"]').click();
      assert.equal(await page.locator('#burstSeg button[data-k="RESP"]').getAttribute('aria-pressed'), 'true');
      assert.match(await page.locator('#hsCanvas-data summary').textContent(), /W9SSJ/);
      assert.match(await page.locator('#hsReadout').textContent(), /response/i);
      const row = page.locator('#hsCanvas-data tbody tr').first();
      const bin = Number(await row.locator('td').nth(1).textContent());
      const hz = Number(await row.locator('td').nth(0).textContent());
      assert.ok(hz > bin, 'VARA: text equivalent converts bin number to audio frequency');
      await page.locator('#carrierSeg button[data-b="2300"]').click();
      assert.match(await page.locator('#carrierCanvas-data summary').textContent(), /BW2300/);
      await page.locator('#highConstSeg button[data-pos="4"]').click();
      assert.equal(await page.locator('#highConstellation-data tbody tr').count(), 32);
      assert.match(await page.locator('#highConstellation-data summary').textContent(), /32-cross/);
      if (await page.locator('#wfRight').isVisible()) {
        const angle = await page.evaluate(() => wfAngle);
        await page.locator('#wfRight').click();
        assert.ok(await page.evaluate(() => wfAngle) > angle, 'VARA: rotation has a native-button equivalent');
        await page.locator('#wfView').click();
        assert.equal(await page.locator('#wfView').textContent(), 'Show 3D view');
      }
      await page.locator('#hsBins').focus();
      assert.equal(await page.evaluate(() => document.activeElement.id), 'hsBins');
    }
  }

  await go('ardop');
  const before = await image('waterfall');
  await page.waitForTimeout(180);
  assert.equal(await image('waterfall'), before, 'ARDOP: waterfall is initially still');
  assert.equal(await page.locator('#motion').textContent(), 'Play waterfall');
  await page.locator('#mode').selectOption({label: '4FSK.500.100'});
  await page.locator('#symbol-next').click();
  assert.match(await page.locator('#symbol-note').textContent(), /Selected tone \d at [\d.]+ Hz/);
  assert.match(await page.locator('#symbol-status').textContent(), /Selected tone/);
  await page.locator('#repair').click();
  assert.match(await page.locator('#repair-note').textContent(), /Corrects up to \d+ byte errors.*within the Reed/);
  assert.equal(await page.locator('#repair').textContent(), 'Reset errors');
  await page.locator('#next').click();
  for (let i = 0; i < 20 && await page.locator('#next').getAttribute('aria-disabled') !== 'true'; i++) await page.locator('#next').click();
  assert.equal(await page.locator('#next').getAttribute('aria-disabled'), 'true');
  assert.equal(await page.evaluate(() => document.activeElement.id), 'next', 'ARDOP: final step retains focus');
  await page.locator('#exchange-play').click();
  assert.equal(await page.locator('#step-content').getAttribute('aria-live'), 'off');
  await page.locator('#previous').click();
  assert.equal(await page.locator('#step-content').getAttribute('aria-live'), 'polite');
  assert.equal(await page.locator('#exchange-play').textContent(), 'Play exchange');
  assert.equal(await page.locator('#waterfall-data tbody tr').count(), 192);

  await go('compared');
  assert.equal(await page.locator('#wf-data tbody tr').count(), 8, 'Recorded waterfall has a concise time/peak summary');
  await page.locator('#wf-data summary').click();
  assert.equal(await page.locator('#wf-data button').count(), 0, 'No numerical dump download replaces the plot explanation');
  await page.locator('#wfBtn').click();
  assert.match(await page.locator('#wfSrc').textContent(), /ConAck.*failed connection attempt/);
  await page.locator('#specSeg button').filter({hasText:'Recorded spectrum'}).click();
  assert.ok(await page.locator('#specCanvas-data tbody tr').count() <= 6);
  for(const [selection,rate,context,route] of [
    ['2','PACTOR-3 SL6',/rate probe; no mail payload/, 'pactor3-send'],
    ['3','VARA Tone L9',/acknowledged L9.*tone frame, not OFDM/, 'vara-send']
  ]){
    await page.locator('#spectrumRecordingSelect').focus();
    await page.locator('#spectrumRecordingSelect').selectOption(selection);
    assert.equal(await page.locator('#recordingSelect').inputValue(),selection,'Both recording selectors stay synchronized');
    assert.match(await page.locator('#wfBadge').textContent(),new RegExp(rate));
    assert.match(await page.locator('#wfSrc').textContent(),context);
    assert.match(await page.locator('#wfSrc').textContent(),/local W9SSJ transmitted audio, not a peer receive measurement/);
    assert.ok(await page.locator('#specCanvas-data tbody tr').count() <= 6);
    assert.match(await page.locator('#waveCanvas-data').textContent(),/48000.*original PCM samples/);
    assert.equal(await page.evaluate(()=>document.activeElement.id),'spectrumRecordingSelect','Changing recordings keeps native selector focus');
    for(const selector of ['#wfSrc','#specSrc'])for(const href of await page.locator(selector+' a').evaluateAll(nodes=>nodes.map(n=>n.href))){
      const response=await page.request.get(href);assert.equal(response.status(),200,'Public recording/inspector link resolves: '+href);
    }
    assert.match(await page.locator('#wfSrc a').first().getAttribute('href'),new RegExp(route));
  }
  await page.locator('#waveSeg button').filter({hasText:'Frame RMS envelope'}).click();
  assert.match(await page.locator('#waveCanvas-data').textContent(),/64 equal-duration RMS bins/);
  await page.locator('#specSeg button').filter({hasText:'Calculated carriers'}).click();
  await page.locator('#geometryMode').selectOption('vara');
  for(const [band,family,rows,note] of [['500','index',4,/two-subband DBPSK/],['2300','tone',5,/10 simultaneous carriers/],['2750','ofdm',7,/59 simultaneous carriers/]]){
    await page.locator('#varaBand').selectOption(band);await page.locator('#varaFamily').selectOption(family);
    assert.equal(await page.locator('#specCanvas-data tbody tr').count(),rows);
    assert.match(await page.locator('#specCanvas-data').textContent(),note);
    assert.match(await page.locator('#specSrc').textContent(),/Calculated geometry, not recorded audio/);
  }
  await page.locator('#ladderSeg button').filter({hasText: 'VARA'}).click();
  assert.match(await page.locator('#ladderCanvas-data summary').textContent(), /VARA/);
  await page.locator('#constSeg button').filter({hasText: 'measured only'}).click();
  assert.equal(await page.locator('#constSeg button').filter({hasText: 'measured only'}).getAttribute('aria-pressed'), 'true');
  assert.doesNotMatch(await page.locator('#constCanvas-data summary').textContent(), /ideal/);
  for(const [mode,points,bytes,source,context] of [
    ['ardop','396','64',/Remote SDR witness.*WW2MI/,/receive parsing failed and END was not confirmed/],
    ['pactor3','1296','0',/unit magnitude/,/rate probe; no mail payload/],
    ['vara','3430','425',/receiver decision-input gain/,/acknowledged L9/],
    ['vara-ofdm','1584','197',/BW500 L7.*Independently decoded/,/OFDM data frame carrying the Winlink FQ close request/]
  ]){
    await page.locator('#constFrame').focus();await page.locator('#constFrame').selectOption(mode);
    assert.equal(await page.evaluate(()=>document.activeElement.id),'constFrame','Selecting a constellation preserves native focus');
    assert.equal(await page.locator('#tilePoints').textContent(),points);
    assert.equal(await page.locator('#tileBytes').textContent(),bytes);
    assert.match(await page.locator('#constSrc').textContent(),source);
    assert.match(await page.locator('#constNote').textContent(),context);
    assert.equal(await page.locator('#constCanvas-data tbody tr').filter({hasText:'Ideal direction'}).count(),4,'All ideal directions have coordinate/count text equivalents');
    assert.ok(await page.locator('#constCanvas-data tbody tr').count()<=14,'Constellation text alternative is compact');
    assert.equal(await page.locator('#ardopQuality').isVisible(),mode==='ardop');
    assert.equal(await page.locator('#tileBytesLabel').textContent(),mode==='vara-ofdm'?'decoded frame bytes (including control/CRC)':'decoded payload bytes');
    if(mode==='ardop')assert.equal(await page.locator('#qualInline').textContent(),'89 of 100');
    for(const href of await page.locator('#constSrc a').evaluateAll(nodes=>nodes.map(n=>n.href))){const response=await page.request.get(href);assert.equal(response.status(),200,'Selected constellation source link resolves: '+href)}
    await page.locator('#constSeg button').filter({hasText:'with ideal grid'}).click();
    assert.match(await page.locator('#constCanvas-data summary').textContent(),/with ideal grid/);
    await page.locator('#constSeg button').filter({hasText:'measured only'}).click();
    assert.doesNotMatch(await page.locator('#constCanvas-data summary').textContent(),/ideal/);
  }
  await page.setViewportSize({width:320,height:900});
  await page.locator('#constFrame').selectOption('vara-ofdm');
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Selected OFDM constellation reflows at320px');
  await page.setViewportSize({width:1280,height:900});

  await go('sabir');
  await page.locator('#profile').selectOption('fast');
  assert.match(await page.locator('#geometry').textContent(), /56 carriers/);
  assert.match(await page.locator('#spectrum-data summary').textContent(), /fast/);
  assert.ok(await page.locator('#spectrum-data tbody tr').count() > 20);
  return ['pactor','vara','ardop','compared','sabir'];
}
