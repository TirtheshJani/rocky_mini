// Playwright end-to-end test of the Rocky Mini settings UI.
const { chromium } = require('playwright');

const TARGET_URL = process.env.ROCKY_URL || 'http://127.0.0.1:8042';
const SHOT = process.env.SHOT_DIR || '.';

function assert(cond, msg) {
  if (!cond) throw new Error('ASSERT FAILED: ' + msg);
  console.log('  ok - ' + msg);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(e.message));

  try {
    console.log('1. Load page');
    // Use domcontentloaded, not networkidle: the SSE metrics stream keeps a request
    // open forever, so networkidle never fires.
    await page.goto(TARGET_URL, { waitUntil: 'domcontentloaded' });
    assert((await page.title()) === 'Rocky Mini', 'title is Rocky Mini');
    await page.waitForSelector('#chip-words');
    assert((await page.textContent('#chip-words')).includes('knows 0 words'), 'starts at 0 words');

    console.log('2. Teach Rocky a fact via chat');
    await page.fill('#chat-input', 'a taco is food');
    await page.click('#chat-send');
    await page.waitForSelector('.msg.rocky');
    const reply = await page.textContent('.msg.rocky');
    assert(/Rocky learn/i.test(reply), 'Rocky acknowledges learning: ' + JSON.stringify(reply.slice(0, 60)));

    console.log('3. Fact appears in the table');
    await page.waitForSelector('#facts-body tr[data-fact-id]');
    const factText = await page.textContent('#facts-body tr[data-fact-id] td');
    assert(factText.trim() === 'taco is food', 'fact row shows "taco is food"');
    assert((await page.textContent('#chip-words')).includes('knows 1 words'), 'word count went to 1');

    console.log('4. Metrics panel populated');
    const ttfa = await page.textContent('#m-ttfa');
    assert(ttfa !== '—', 'time-to-first-audio metric rendered: ' + ttfa);
    const meterStatus = await page.textContent('#meter-status');
    assert(/budget/.test(meterStatus), 'latency meter shows budget status: ' + meterStatus);

    console.log('5. Naivety deflection on an Earth-knowledge probe');
    await page.fill('#chat-input', 'what is the capital of Italy?');
    await page.click('#chat-send');
    await page.waitForFunction(() => document.querySelectorAll('.msg.rocky').length >= 2);
    const replies = await page.$$eval('.msg.rocky', (els) => els.map((e) => e.textContent));
    assert(/not know/i.test(replies[replies.length - 1]), 'Rocky stays naive (deflects capital of Italy)');
    const leakMsgs = await page.$$('.msg.rocky.leak');
    assert(leakMsgs.length === 0, 'no leak flagged on the deflection');

    console.log('6. Confirm a fact');
    await page.click('#facts-body tr[data-fact-id] .act-confirm');
    await page.waitForFunction(() =>
      document.querySelector('#facts-body tr[data-fact-id] .conf')?.textContent === 'confirmed');
    assert((await page.textContent('#facts-body tr[data-fact-id] .conf')) === 'confirmed', 'fact confirmed');

    console.log('7. Fire an emote');
    await page.click('.emote-btn[data-emote="jazz_hands"]');
    assert(true, 'jazz hands emote clicked (POST /api/emote)');

    console.log('8. Toggle the model');
    await page.selectOption('#model-select', 'rocky:latest');
    await page.click('#settings-save');
    await page.waitForFunction(() => document.querySelector('#chip-model').textContent.includes('rocky:latest'));
    assert((await page.textContent('#chip-model')).includes('rocky:latest'), 'model toggled to rocky:latest');

    console.log('9. Delete the fact');
    await page.click('#facts-body tr[data-fact-id] .act-delete');
    await page.waitForSelector('#facts-body .empty-row');
    assert(await page.$('#facts-body .empty-row'), 'fact table empty after delete');

    console.log('10. Screenshots (desktop + mobile)');
    // Re-teach so the screenshot shows a populated UI.
    await page.fill('#chat-input', 'the sun is a star');
    await page.click('#chat-send');
    await page.waitForTimeout(300);
    await page.screenshot({ path: SHOT + '/rocky-ui-desktop.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(200);
    await page.screenshot({ path: SHOT + '/rocky-ui-mobile.png', fullPage: true });

    assert(errors.length === 0, 'no console/page errors (' + JSON.stringify(errors) + ')');

    console.log('\nALL UI CHECKS PASSED');
  } catch (e) {
    console.error('\nFAILED:', e.message);
    if (errors.length) console.error('console errors:', errors);
    await page.screenshot({ path: SHOT + '/rocky-ui-failure.png', fullPage: true }).catch(() => {});
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
