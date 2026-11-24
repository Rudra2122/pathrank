// Lighthouse (mobile form factor, the default) on the production build's login page.
// Needs Chrome installed. Serves dist/ itself, so run `npm run build` first.
//   npm run build && npm run lighthouse  ->  ../../results/ads/lighthouse.json
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import * as chromeLauncher from 'chrome-launcher';
import lighthouse from 'lighthouse';

const dist = resolve('dist');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.map': 'application/json' };
const server = createServer(async (req, res) => {
  const path = req.url === '/' || !extname(req.url) ? '/index.html' : req.url.split('?')[0];
  try {
    const body = await readFile(join(dist, path));
    res.writeHead(200, { 'Content-Type': types[extname(path)] || 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(5182);

const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless=new'] });
try {
  const r = await lighthouse('http://localhost:5182/', {
    port: chrome.port, output: 'json', onlyCategories: ['performance', 'accessibility', 'best-practices'],
  });
  const scores = Object.fromEntries(Object.entries(r.lhr.categories).map(([k, v]) => [k, Math.round(v.score * 100)]));
  const out = { url: '/', formFactor: r.lhr.configSettings.formFactor, scores,
    lcp_ms: Math.round(r.lhr.audits['largest-contentful-paint'].numericValue) };
  await mkdir('../../results/ads', { recursive: true });
  await writeFile('../../results/ads/lighthouse.json', JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out, null, 2));
} finally {
  await chrome.kill();
  server.close();
}
